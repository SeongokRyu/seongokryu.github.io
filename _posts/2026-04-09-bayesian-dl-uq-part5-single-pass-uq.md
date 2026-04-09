---
title: "Bayesian DL & UQ Part 5: Single-Pass UQ — Evidential Deep Learning and Distance-Aware Methods"
date: 2026-04-09 14:25:00 +0900
categories: [AI & ML, ML Theory]
tags: [bayesian-deep-learning, evidential-deep-learning, sngp, duq, single-pass-uq]
math: true
---

**Bayesian Deep Learning and Uncertainty Quantification — A Journey Toward Trustworthy AI**

*This is Part 5 of an 8-part series.*

- **Part 0:** [Beyond Predictions — Why Uncertainty Matters](/posts/bayesian-dl-uq-part0-why-uncertainty/)
- **Part 1:** [The Language of Bayesian Inference](/posts/bayesian-dl-uq-part1-bayesian-language/)
- **Part 2:** [The Art of Approximation — From Variational to Ensemble](/posts/bayesian-dl-uq-part2-art-of-approximation/)
- **Part 3:** [The Anatomy of Uncertainty — Aleatoric vs. Epistemic](/posts/bayesian-dl-uq-part3-anatomy-of-uncertainty/)
- **Part 4:** [Calibration and Conformal Prediction](/posts/bayesian-dl-uq-part4-calibration-conformal/)
- **Part 5:** Single-Pass UQ — Evidential Deep Learning and Distance-Aware Methods (this post)
- **Part 6:** [UQ in Science — Molecules, Proteins, and Materials](/posts/bayesian-dl-uq-part6-uq-in-science/)
- **Part 7:** [The Future of UQ — Uncertainty in the Age of Foundation Models](/posts/bayesian-dl-uq-part7-future-foundation-models/)

---

## Hook: You need to evaluate millions of molecules, but cannot afford 10 forward passes

Imagine a virtual screening pipeline in drug discovery. The goal is to identify candidate molecules with activity against a target protein. The search space spans **millions to billions of molecules**. The ZINC database alone contains over 750 million compounds, and Enamine REAL exceeds 40 billion.

From Part 2, we know that Deep Ensembles deliver the best calibration. But what if you need to evaluate 1 billion molecules with an ensemble of 5 models? That is **5x the inference cost**. Running MC Dropout with 20 forward passes? **20x the inference cost**. Translated into GPU hours, this is the difference between days and weeks.

Even more extreme scenarios exist. Real-time object detection in autonomous vehicles must complete inference within 16 ms per frame. Industrial anomaly detection demands millisecond-level latency. In these settings, ensembles or sampling are **physically impossible**.

The solution to this problem is **single-pass uncertainty quantification** — methods that output both predictions and uncertainty estimates **simultaneously** from a single forward pass. In this post, we examine three major lineages of this approach — Evidential Deep Learning, SNGP, and DUQ — from their mathematical foundations to their practical performance.

---

## 1. The Computational Cost Problem — Why We Need Single-Pass Methods

### 1.1 Cost structure of existing methods

Let us summarize the computational costs of the methods covered in Part 2:

```
+-------------------+---------------+----------------+-------------------+
| Method            | Training Cost | Inference Cost | Memory Cost       |
+-------------------+---------------+----------------+-------------------+
| Point Estimate    | 1x            | 1x             | 1x                |
| MC Dropout        | 1x            | T x  (T~20)    | 1x                |
| Deep Ensemble     | M x  (M~5)    | M x            | M x               |
| SWAG              | ~1.2x         | T x  (T~30)    | ~1x + low-rank    |
| Laplace (post-hoc)| ~1x           | ~1x *          | 1x + Hessian      |
| Single-Pass UQ    | 1x            | 1x             | ~1x               |
+-------------------+---------------+----------------+-------------------+
  * Last-layer Laplace is close to single-pass,
    but incurs additional cost for full posterior sampling
```

- **Deep Ensemble**: Training $M$ models independently multiplies training cost by $M$. Inference also requires a forward pass through each of the $M$ models, multiplying by $M$. With $M = 5$, the total cost is 5x.
- **MC Dropout**: Training is identical to standard dropout training, but inference requires $T$ forward passes. In practice, $T = 10 \sim 50$ is typical, resulting in **10-50x inference cost**.
- **SWAG**: Training is slightly more expensive due to covariance estimation on top of SWA, and inference requires sampling $T$ weight configurations from the SWAG posterior, each needing a forward pass — yielding $T\times$ inference cost.

### 1.2 Practical constraints at scale

Let us put concrete numbers to the problem to appreciate its magnitude.

- **Virtual screening**: Predicting binding affinity for 750 million molecules in the ZINC-15 database
  - Point estimate (1x): ~12 hours on a single A100 GPU
  - Deep Ensemble (5x): 60 hours = 2.5 days
  - MC Dropout (20x): 240 hours = 10 days
- **Real-time inference**: Autonomous driving object detection with a 16 ms budget per frame
  - 1x forward pass: 8 ms — feasible
  - 5x ensemble: 40 ms — **infeasible** (frame drops)
- **Neural Network Potentials**: Energy and force computation at every timestep of a molecular dynamics simulation
  - $10^6$ timesteps $\times$ $10^3$ atoms = $10^9$ inference calls
  - An ensemble extends the simulation wall time by 5x

### 1.3 The core idea behind single-pass UQ

The common strategy across single-pass UQ methods can be summarized as follows:

> **Expand the network's output so that a single forward pass produces both the prediction and an uncertainty estimate.**

Whereas existing methods extract uncertainty from **disagreement among multiple models** or **variance across multiple samples**, single-pass methods **train a single model to predict its own uncertainty directly**. We now examine three mathematically distinct frameworks that make this possible.

---

## 2. Evidential Deep Learning — Classification

### 2.1 Core idea: A distribution over class probabilities

A conventional classification network uses softmax to output a class probability vector $\mathbf{p} = (p_1, ..., p_K)$. This $\mathbf{p}$ represents "the probability that the data point belongs to each class," but provides **no information about how trustworthy that probability itself is**.

Sensoy, Kaplan & Kandemir (NeurIPS 2018) proposed a groundbreaking approach grounded in **Subjective Logic** and **Dempster-Shafer Evidence Theory**. The key idea:

- Instead of outputting $\mathbf{p}$ directly, the network outputs a **distribution over $\mathbf{p}$**
- This distribution is modeled as a **Dirichlet distribution**
- The Dirichlet concentration parameters reflect **evidence** gathered from the data

### 2.2 Mathematical framework

**Dirichlet Distribution**: A distribution over the categorical probability vector $\mathbf{p} = (p_1, ..., p_K)$ for $K$ classes:

$$\text{Dir}(\mathbf{p} \mid \boldsymbol{\alpha}) = \frac{\Gamma\big(\sum_{k=1}^K \alpha_k\big)}{\prod_{k=1}^K \Gamma(\alpha_k)} \prod_{k=1}^K p_k^{\alpha_k - 1}$$

where $\boldsymbol{\alpha} = (\alpha_1, ..., \alpha_K)$ are the **concentration parameters**, with all $\alpha_k > 0$. Key properties of the Dirichlet distribution:

- **Mean**: $$\mathbb{E}[p_k] = \alpha_k / S$$ where $$S = \sum_{k=1}^K \alpha_k$$ (Dirichlet strength)
- **Variance**:

$$\text{Var}[p_k] = \frac{\alpha_k(S - \alpha_k)}{S^2(S + 1)}$$
- Larger $S$ → narrower distribution → greater **confidence** in the class probabilities
- Smaller $S$ → wider distribution → greater **uncertainty** in the class probabilities

**From evidence to concentration parameters**: The network outputs non-negative **evidence** $e_k \geq 0$ for each class (enforced via ReLU or similar activations). The concentration parameters are:

$$\alpha_k = e_k + 1$$

When $e_k = 0$, we get $\alpha_k = 1$ → uniform prior (no evidence for that class).

**Belief mass and vacuity**: Within the Subjective Logic framework, the **belief mass** $b_k$ for each class and the overall **vacuity** $u$ are defined as:

$$b_k = \frac{e_k}{S} = \frac{\alpha_k - 1}{S}, \quad u = \frac{K}{S}$$

where:
- $b_k$: evidence-based belief assigned to class $k$ → $\sum_k b_k + u = 1$
- $u$: **uncertainty mass** → larger when evidence is insufficient
- $K$: number of classes
- When all $e_k = 0$, $S = K$, so $u = 1$ → total ignorance

Visualizing these relationships on the Dirichlet simplex:

```
Dirichlet Simplex Visualization (K=3 classes)

Case 1: Strong evidence for Class 1     Case 2: No evidence (vacuity = 1)
    alpha = (10, 1, 1), S=12                alpha = (1, 1, 1), S=3
    u = 3/12 = 0.25                         u = 3/3 = 1.0

         C3                                      C3
        /\                                       /\
       /  \                                     /  \
      /    \                                   /    \
     / *    \    <-- mass near C1             / *  * \   <-- spread
    /  *     \                               / * ** * \      uniformly
   /  **      \                             / * * ** * \
  /____________\                           /____________\
 C1            C2                         C1            C2

Case 3: Balanced strong evidence         Case 4: Evidence for C1 and C2
    alpha = (10, 10, 10), S=30               alpha = (8, 8, 1), S=17
    u = 3/30 = 0.1                           u = 3/17 = 0.18

         C3                                      C3
        /\                                       /\
       /  \                                     /  \
      /    \                                   /    \
     /      \                                 /      \
    /   **   \    <-- mass at center         /        \
   /   ***   \                              /   **     \  <-- mass on
  /____________\                           /____________\     C1-C2 edge
 C1            C2                         C1            C2
```

**Key insight**: Compare Case 2 and Case 3. Both have the same expected class probabilities of $(1/3, 1/3, 1/3)$. However, Case 2 means "I don't know because there is no evidence" (high vacuity), while Case 3 means "I am confident, based on ample evidence, that the three classes are roughly equal" (low vacuity). **A softmax output cannot distinguish between these two situations.** This is the fundamental advantage of EDL.

### 2.3 Loss function: Bayes risk + KL regularizer

EDL training is derived from a **Type II Maximum Likelihood** (Empirical Bayes) perspective. For a data point $(\mathbf{x}_i, y_i)$, where $y_i$ is represented as a one-hot vector $\mathbf{y}_i$:

**Bayes Risk (expected loss under the Dirichlet)**:

$$\mathcal{L}_{\text{BR}}(\boldsymbol{\alpha}_i) = \sum_{k=1}^{K} y_{i,k} \big(\psi(S_i) - \psi(\alpha_{i,k})\big)$$

where $\psi(\cdot)$ is the digamma function. In practice, it is common to replace this with the expected sum-of-squared-errors loss:

$$\mathcal{L}_{\text{MSE}}(\boldsymbol{\alpha}_i) = \sum_{k=1}^{K} \big(y_{i,k} - \hat{p}_{i,k}\big)^2 + \frac{\hat{p}_{i,k}(1 - \hat{p}_{i,k})}{S_i + 1}$$

where

$$\hat{p}_{i,k} = \alpha_{i,k}/S_i$$

- First term: **prediction error** — how close the expected class probability is to the true label
- Second term: **variance penalty** — increases as the Dirichlet strength $S_i$ decreases (i.e., as uncertainty grows)

**KL Regularizer**: Encourages the evidence for incorrect classes to shrink toward zero:

$$\mathcal{L}_{\text{KL}}(\boldsymbol{\alpha}_i) = \text{KL}\big(\text{Dir}(\mathbf{p} \mid \tilde{\boldsymbol{\alpha}}_i) \,\|\, \text{Dir}(\mathbf{p} \mid \mathbf{1})\big)$$

where

$$\tilde{\boldsymbol{\alpha}}_i = \mathbf{y}_i + (1 - \mathbf{y}_i) \odot \boldsymbol{\alpha}_i$$

is the concentration parameter with the evidence for the correct class removed. This regularizer shrinks the evidence for incorrect classes ($e_k$ for $y_k = 0$) toward zero, converting that evidence into vacuity.

**Full loss**:

$$\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N}\Big[\mathcal{L}_{\text{MSE}}(\boldsymbol{\alpha}_i) + \lambda_t \cdot \mathcal{L}_{\text{KL}}(\boldsymbol{\alpha}_i)\Big]$$

### 2.4 The importance of KL regularization annealing

The weight $\lambda_t$ of the KL regularizer starts at 0 and is gradually increased over training (typically ramped from 0 to 1 over at least 10 epochs):

$$\lambda_t = \min\big(1.0,\; t / T_{\text{anneal}}\big)$$

Why this annealing is necessary:

- **Early in training**: The network has not yet learned the correct patterns. Applying the KL regularizer strongly at this stage **drives all evidence to zero** → the network converges to a trivial solution that only outputs "I don't know"
- **Later in training**: Once sufficient evidence for the correct classes has accumulated, the KL regularizer performs its intended role of eliminating evidence for incorrect classes

**Practical experience**: The annealing schedule is highly sensitive to EDL performance. If $T_{\text{anneal}}$ is too short, underfitting results; if too long, the model becomes overconfident on OOD data. This hyperparameter sensitivity is one of EDL's primary weaknesses.

---

## 3. Evidential Deep Learning — Regression

### 3.1 Normal-Inverse-Gamma prior

Where the Dirichlet serves as the prior over categorical probabilities in classification, the **Normal-Inverse-Gamma (NIG)** distribution serves as the joint prior over the mean and variance in regression. Amini, Schwarting, Soleimany & Rus (NeurIPS 2020) introduced this framework.

**Model structure**: Assume the observation $y$ follows a Gaussian likelihood:

$$y \mid \mu, \sigma^2 \sim \mathcal{N}(\mu, \sigma^2)$$

where both the mean $\mu$ and variance $\sigma^2$ are **uncertain**. We place a NIG prior on their joint distribution:

$$(\mu, \sigma^2) \sim \text{NIG}(\gamma, \nu, \alpha, \beta)$$

That is:

$$\sigma^2 \sim \text{Inv-Gamma}(\alpha, \beta), \quad \mu \mid \sigma^2 \sim \mathcal{N}\big(\gamma, \sigma^2/\nu\big)$$

The meaning of each parameter:
- $\gamma$: expected value of the mean → **the prediction**
- $\nu > 0$: pseudo-count of mean observations → **inversely related to epistemic uncertainty**
- $\alpha > 1$: pseudo-count of variance observations
- $\beta > 0$: scale of the variance

**Network output**: A single network outputs all four parameters $(\gamma, \nu, \alpha, \beta)$ for a given input $\mathbf{x}$. To satisfy the constraints:
- $\gamma$: unconstrained (linear output)
- $\nu$: softplus activation ($\nu > 0$)
- $\alpha$: softplus + 1 ($\alpha > 1$)
- $\beta$: softplus activation ($\beta > 0$)

```
NIG Distribution Concept

Network Output:  (gamma, nu, alpha, beta)
                    |      |     |      |
                    v      v     v      v
              prediction  epistemic  aleatoric
                          strength   strength

Uncertainty Decomposition:

  Aleatoric = E[sigma^2] = beta / (alpha - 1)
    --> irreducible noise in data

  Epistemic = Var[mu]    = beta / (nu * (alpha - 1))
    --> uncertainty about the mean

                 +-------------------------------+
                 |        Prediction: gamma      |
                 +-------------------------------+
                 |                               |
          +------+------+              +---------+---------+
          | Aleatoric   |              | Epistemic         |
          | b/(a-1)     |              | b/(v*(a-1))       |
          |             |              |                   |
          | High when:  |              | High when:        |
          | - Noisy data|              | - Sparse training |
          | - Inherent  |              | - Far from train  |
          |   randomness|              |   distribution    |
          +-------------+              +-------------------+
```

### 3.2 Uncertainty decomposition

The NIG prior yields a natural decomposition of uncertainty:

**Aleatoric Uncertainty** (inherent noise in the data):

$$\mathbb{E}[\sigma^2] = \frac{\beta}{\alpha - 1}$$

This is the expected value of the observation noise, reflecting the intrinsic variability of the data.

**Epistemic Uncertainty** (uncertainty about the mean):

$$\text{Var}[\mu] = \frac{\beta}{\nu(\alpha - 1)}$$

This is simply the aleatoric uncertainty divided by $\nu$. Since $\nu$ acts as a pseudo-count for "how many times the mean has been observed," regions with abundant evidence have large $\nu$, driving epistemic uncertainty down.

**Key relationship**: Epistemic = Aleatoric / $\nu$. In **data-rich regions**, $\nu$ is large and epistemic uncertainty is much smaller than aleatoric uncertainty. In **data-sparse regions**, $\nu$ is small and epistemic uncertainty becomes comparable to aleatoric uncertainty. This aligns precisely with the aleatoric/epistemic decomposition principles discussed in Part 3.

### 3.3 Loss function

We minimize the negative log-marginal likelihood under the NIG. For observation $y_i$:

$$\mathcal{L}_{\text{NLL}}^{(i)} = \frac{1}{2}\log\frac{\pi}{\nu_i} - \alpha_i \log \Omega_i + (\alpha_i + \frac{1}{2})\log\big((y_i - \gamma_i)^2 \nu_i + \Omega_i\big) + \log\frac{\Gamma(\alpha_i)}{\Gamma(\alpha_i + \frac{1}{2})}$$

where

$$\Omega_i = 2\beta_i(1 + \nu_i)$$

An additional evidence regularizer is applied:

$$\mathcal{L}_{\text{reg}}^{(i)} = |y_i - \gamma_i| \cdot (2\nu_i + \alpha_i)$$

This regularizer **penalizes high evidence when the prediction error is large**, preventing the model from being simultaneously wrong and confident.

### 3.4 Extension to molecular property prediction — Soleimany et al. (2021)

Soleimany et al. (ACS Central Science, 2021) presented the first systematic application of Deep Evidential Regression to **molecular property prediction and drug discovery**. Key contributions of this work:

**Data efficiency on the QM9 benchmark**:
- Applied evidential regression to 12 quantum chemical properties (HOMO, LUMO, gap, dipole moment, etc.)
- Achieved accuracy comparable to full-training (100% data) performance using **only 40% of the data**
- Core mechanism: filtering out high-uncertainty predictions dramatically improves the accuracy of the remaining predictions
- This aligns with the fundamental principle of active learning — a model that "knows what it doesn't know" learns more efficiently

**Confidence filtering in antibiotic screening**:
- Applied to the antibiotic discovery dataset from Stokes et al. (Cell, 2020)
- Baseline model hit rate: ~78%
- After applying evidential uncertainty-based confidence filtering: **hit rate above 95%**
- Simply excluding the top-20% most uncertain predictions sharply boosted the hit rate
- This translates to **tangible cost savings in virtual screening** — reducing the number of molecules that need to be synthesized and experimentally tested while improving the success rate

**The practical bottom line**: These results empirically validate the central thesis from Part 0 — **"The activity of this molecule is X" is far less useful for decision-making than "The activity of this molecule is X, with uncertainty Y."**

### 3.5 Extension to Neural Network Potentials

A recent study published in Nature Communications (2025) reported notable results applying Evidential Deep Learning to **interatomic potentials** (neural network potentials, NNPs).

**The UQ challenge in NNPs**:
- Molecular dynamics (MD) simulations compute energy and forces at every timestep (~1 fs intervals)
- Running an ensemble at every step across millions of timesteps is impractical
- EDL, which provides uncertainty from a single model, is ideally suited here

**Per-atom uncertainty (spatially resolved uncertainty)**:
- The total energy in an NNP is a sum of atomic contributions: $E = \sum_i \epsilon_i$
- Applying EDL yields $$(\gamma_i, \nu_i, \alpha_i, \beta_i)$$ for each atom $i$
- This enables identification of **which atoms are uncertain** → physically interpretable
- Example: High epistemic uncertainty around a reactive center indicates that the corresponding chemical environment is underrepresented in the training data

**Practical significance**:
- When uncertainty exceeds a threshold, the simulation can halt and switch to DFT (density functional theory) calculations — an **adaptive switching** strategy
- This implements a workflow where "fast NNPs are used only in regions of guaranteed accuracy, and slow but accurate ab initio methods fill in where uncertainty is high"

---

## 4. SNGP — Distance-Aware Uncertainty

### 4.1 Feature collapse: Why standard networks are overconfident on OOD data

The most fundamental obstacle to single-pass UQ is **feature collapse**. Since this concept motivates both SNGP and DUQ, we examine it in detail first.

**The phenomenon**: A standard ReLU network can map arbitrary OOD inputs into the **same feature region as in-distribution (ID) data**. This occurs because the hidden representations are optimized exclusively for ID data.

```
Feature Collapse Problem

Input Space                 Feature Space (hidden layer)
+-----------+               +-----------+
|           |               |  * * *    |
|  ID data  |   ------>     |  * * *    |  <-- ID and OOD mapped
|  (o o o)  |   Network     |  x x     |      to the same region!
|           |               |  x       |
|  OOD data |               |           |
|  (x x x)  |               |           |
+-----------+               +-----------+

                            Softmax over collapsed features
                            --> high confidence for OOD (x)!


Distance-Aware Feature Space (with bi-Lipschitz)
+-----------+               +-----------+
|           |               |  * * *    |
|  ID data  |   ------>     |  * * *    |  <-- ID and OOD
|  (o o o)  |   SNGP        |           |      separated!
|           |               |           |
|  OOD data |               |     x x   |
|  (x x x)  |               |     x     |
+-----------+               +-----------+

                            GP over separated features
                            --> high uncertainty for OOD (x)
```

**Why this is dangerous**: When feature collapse occurs, any output head — whether softmax, sigmoid, or even Dirichlet — produces outputs for OOD inputs that are indistinguishable from those for ID data. **The root of the problem lies in the feature extractor, not the output layer.**

**Mathematical perspective**: A standard ReLU network $f: \mathbb{R}^d \to \mathbb{R}^h$ is a piecewise linear function. Within each linear region, $f(\mathbf{x}) = \mathbf{A}\mathbf{x} + \mathbf{b}$, and if an OOD input falls within the same linear region as ID data, the network treats both identically. More critically, the saturation region of ReLU ($\text{ReLU}(z) = 0$ for $z < 0$) zeroes out many features of OOD inputs, effectively "collapsing" them into a specific region of the feature space.

**An example from molecular property prediction**: Consider molecular representations learned by a GNN. When the training data is concentrated on drug-like molecules (MW < 500, LogP < 5), the feature vectors of polymers or organometallic compounds may be mapped to the same region as drug-like molecules. In such cases, the model produces high-confidence predictions for molecules from an entirely different chemical space. In my own Bayesian GCN research (Ryu et al., 2019), we observed precisely this phenomenon — it was one of the main reasons epistemic uncertainty was reported lower than expected.

### 4.2 The two modifications in SNGP

Liu, Padhy, Ren, Lin & Lakshminarayanan (JMLR 2023, "A Simple Approach to Improve Single-Model Deep Uncertainty via Distance-Awareness") showed that **just two modifications to an existing network** are sufficient to obtain distance-aware uncertainty from a single model.

#### Modification 1: Spectral Normalization → Bi-Lipschitz Constraint

**Goal**: Force the feature extractor to preserve distances in input space.

**Lipschitz continuity**: A function $f$ is Lipschitz continuous if there exists a constant $L > 0$ such that:

$$\|f(\mathbf{x}_1) - f(\mathbf{x}_2)\| \leq L \cdot \|\mathbf{x}_1 - \mathbf{x}_2\|$$

**Bi-Lipschitz** additionally requires a lower bound:

$$l \cdot \|\mathbf{x}_1 - \mathbf{x}_2\| \leq \|f(\mathbf{x}_1) - f(\mathbf{x}_2)\| \leq L \cdot \|\mathbf{x}_1 - \mathbf{x}_2\|$$

In other words, **points that are far apart in input space must also be far apart in feature space**. This is what prevents feature collapse.

**Spectral Normalization**: The spectral norm (largest singular value) $\sigma(\mathbf{W}_l)$ of each weight matrix $\mathbf{W}_l$ is constrained to be at most a constant $c$:

$$\mathbf{W}_l \leftarrow c \cdot \frac{\mathbf{W}_l}{\max(1, \sigma(\mathbf{W}_l) / c)}$$

where $c$ is the upper bound on the spectral norm (typically $c = 0.95 \sim 6.0$). This normalization bounds the Lipschitz constant of each layer, and since the overall Lipschitz constant of a deep network is the product of layer-wise constants, a global upper bound is guaranteed.

**Additional effect in ResNets**: In a residual connection

$$\mathbf{h}_{l+1} = \mathbf{h}_l + f_l(\mathbf{h}_l)$$

when the spectral norm is at most 1,

$$\|f_l(\mathbf{h}_l)\| \leq \|\mathbf{h}_l\|$$

so the skip connection dominates. This naturally guarantees a lower bound $l$ as well, meaning **ResNet + Spectral Normalization provides an approximate bi-Lipschitz property**.

#### Modification 2: GP Output Layer → Reflecting distance from training data

**Goal**: For the features at the last hidden layer, generate uncertainty that scales with distance from the training data.

The standard softmax layer is replaced by a **Random Feature Gaussian Process (RFGP)**. The key equation:

$$\text{logit}_k(\mathbf{x}) = \boldsymbol{\beta}_k^T \hat{\boldsymbol{\phi}}(\mathbf{h}(\mathbf{x}))$$

where $\hat{\boldsymbol{\phi}}$ is a random Fourier feature approximation of the RBF kernel, and $\boldsymbol{\beta}_k$ are the GP weights.

**Predictive distribution**: The GP provides a closed-form predictive mean and variance:

$$\text{logit}_k(\mathbf{x}^{\ast}) \sim \mathcal{N}\big(\hat{\mu}_k(\mathbf{x}^{\ast}),\, \hat{\sigma}_k^2(\mathbf{x}^{\ast})\big)$$

where the predictive variance $\hat{\sigma}_k^2(\mathbf{x}^{\ast})$ **directly reflects the distance from the training data in feature space**. In regions densely populated with training data, $\hat{\sigma}_k^2$ is small; in regions far from training data, $\hat{\sigma}_k^2$ is large.

**The posterior is computed efficiently via Laplace approximation**: The GP posterior precision matrix is incrementally updated during training:

$$\boldsymbol{\Sigma}^{-1} = \boldsymbol{\Sigma}_0^{-1} + \hat{\boldsymbol{\Phi}}^T \hat{\boldsymbol{\Phi}}$$

where $\hat{\boldsymbol{\Phi}}$ is the random feature matrix of the training data. Rather than computing the matrix inverse directly, the matrix inversion lemma is used for efficient computation.

### 4.3 SNGP architecture overview

```
SNGP Architecture

Standard Network:
+-------+     +--------+     +--------+     +---------+     +--------+
| Input | --> | Conv/  | --> | Conv/  | --> | Hidden  | --> | Softmax|
|       |     | FC + BN|     | FC + BN|     | Features|     | Output |
+-------+     +--------+     +--------+     +---------+     +--------+

SNGP (two modifications only):
+-------+     +--------+     +--------+     +---------+     +---------+
| Input | --> | Conv/  | --> | Conv/  | --> | Hidden  | --> | GP      |
|       |     | FC + SN|     | FC + SN|     | Features|     | Output  |
+-------+     +--------+     +--------+     +---------+     +---------+
                  ^              ^                              ^
                  |              |                              |
            Spectral Norm  Spectral Norm               Gaussian Process
            (Modification 1)                          (Modification 2)
                                                    Returns mean + variance
                  |                                        |
                  v                                        v
         Preserves distance                   Distance-dependent
         in feature space                     uncertainty estimate
```

### 4.4 Why a GP provides distance awareness — an intuitive explanation

Let us build intuition for why a GP output layer provides distance-aware uncertainty.

**Standard softmax layer**: After learning a weight matrix $\mathbf{W}$ and bias $\mathbf{b}$, it outputs $\text{softmax}(\mathbf{W}\mathbf{h} + \mathbf{b})$. This function has **the same level of confidence everywhere in feature space** — it applies the same linear boundary even to feature vectors far from any training data. The softmax **inherently lacks any notion of** "how close is this feature vector to the training data?"

**GP output layer**: The critical difference is that the GP's predictive variance takes the form:

$$\hat{\sigma}^2(\mathbf{x}^{\ast}) = k(\mathbf{x}^{\ast}, \mathbf{x}^{\ast}) - \mathbf{k}_{\ast}^T (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{k}_{\ast}$$

where $\mathbf{k}_{\ast}$ is the kernel similarity vector between the test point and the training data:

- **Far from training data**: the entries of $\mathbf{k}_{\ast}$ are near zero, the second term becomes small, and the predictive variance approaches the prior variance $k(\mathbf{x}^{\ast}, \mathbf{x}^{\ast})$ — i.e., uncertainty is high
- **Near training data**: $\mathbf{k}_{\ast}$ is large, the second term cancels most of the prior, and the variance shrinks

This is the mathematical mechanism behind "distance awareness." Through the kernel function, the GP **automatically incorporates distance from training data into its uncertainty**.

**Synergy with spectral normalization**: For this mechanism to work properly, distances in feature space must be meaningful. If feature collapse occurs, OOD features end up close to ID features, and the GP returns low variance — the exact opposite of the intended behavior. Spectral normalization prevents this.

### 4.5 Performance and practical implications

Key experimental results from Liu et al. (2023):

- **CIFAR-10/100, ImageNet**: SNGP achieves calibration (ECE) and OOD detection (AUROC) performance **on par with or superior to** Deep Ensembles (5 models), at only **1x cost**
- **Selective prediction**: Achieves AUPRC comparable to Deep Ensembles in confidence-based abstention
- **Computational overhead**: Spectral normalization adds ~3-5% training overhead; the GP layer adds ~5% → total overhead of **~10% or less**
- **Robustness to dataset shift**: On CIFAR-10-C (corrupted images), calibration degradation is as gradual as that of Deep Ensembles — markedly better than a single standard model

**Key findings from the ablation study**:
- Spectral normalization **alone** (with standard softmax, no GP): minimal improvement in OOD detection
- GP output layer **alone** (without spectral norm): some improvement, but limited
- **Only when both modifications are applied together** does performance match Deep Ensembles → the two modifications are complementary

This ablation result clearly illustrates SNGP's design philosophy. Feature collapse prevention (spectral norm) and distance-based uncertainty (GP) are not independent — **one is the prerequisite for the other**.

**Limitations of SNGP**:
- **No aleatoric/epistemic decomposition**: The GP's predictive variance reflects a mixture of both uncertainty types, with no natural way to disentangle them
- **Complexity of the GP layer implementation**: Requires additional hyperparameters including the number of random features ($D$), kernel lengthscale ($l$), and prior precision ($\lambda$), all of which significantly affect performance
- **Spectral norm upper bound $c$**: Too small and it degrades the network's expressiveness (constraining each layer's representational capacity); too large and feature collapse prevention weakens. In practice, $c = 3 \sim 6$ is typical
- **Precision matrix updates**: The precision matrix must be incrementally updated during training, adding implementation complexity compared to a standard training loop

---

## 5. DUQ — Deterministic Uncertainty Quantification

### 5.1 Dropping softmax and measuring distance instead

van Amersfoort, Smith, Teh & Gal (ICML 2020, "Uncertainty Estimation Using a Single Deep Deterministic Neural Network") start from the fundamental problem with softmax.

**The softmax problem**: For any logit vector $\mathbf{z}$, softmax always outputs a valid probability distribution — even when $\mathbf{z}$ comes from a completely meaningless OOD input. Softmax has no built-in mechanism to express "I don't know."

**DUQ's alternative**: Replace softmax with **RBF (Radial Basis Function) kernel distances to class centroids**.

For each class $k$, a **centroid** $\mathbf{c}_k$ is maintained in feature space. For the feature representation $\mathbf{h}(\mathbf{x})$ of input $\mathbf{x}$:

$$K_k(\mathbf{x}) = \exp\Big(-\frac{\|\mathbf{W}_k \mathbf{h}(\mathbf{x}) - \mathbf{e}_k\|^2}{2\sigma^2}\Big)$$

where:
- $\mathbf{W}_k$: class-specific projection matrix (maps features into the centroid space)
- $\mathbf{e}_k$: centroid for class $k$ (initialized as a unit vector, updated via exponential moving average during training)
- $\sigma$: RBF kernel lengthscale

**Key property**: $K_k(\mathbf{x}) \in [0, 1]$, and:
- If $\mathbf{x}$ is **close** to the centroid of class $k$ → $K_k \approx 1$
- If $\mathbf{x}$ is **far from all** centroids → $K_k \approx 0$ for all $k$ → **"I don't know"**

This is the decisive difference from softmax. Even when all logits are low, softmax normalizes and assigns high probability somewhere. An RBF kernel, by contrast, returns **values near zero for all classes** when the input is far from every centroid.

### 5.2 Bi-Lipschitz guarantee via gradient penalty

DUQ must also address the same problem as SNGP — feature collapse. Instead of spectral normalization, DUQ uses a **two-sided gradient penalty**.

The gradient norm of the RBF output $K_k$ with respect to the input $\mathbf{x}$ is controlled:

$$\mathcal{L}_{\text{GP}} = \lambda \sum_k \Big(\big\|\nabla_{\mathbf{x}} K_k(\mathbf{x})\big\| - 1\Big)^2$$

This penalty keeps the gradient norm close to 1, which:
- **Upper bound**: Prevents the gradient from being too large → Lipschitz upper bound
- **Lower bound**: Prevents the gradient from being too small → ensures changes in input space are reflected in feature space

### 5.3 Training and inference

**Training**: Binary cross-entropy loss + gradient penalty

$$\mathcal{L} = -\sum_{i=1}^{N}\sum_{k=1}^{K}\Big[y_{i,k}\log K_k(\mathbf{x}_i) + (1-y_{i,k})\log(1 - K_k(\mathbf{x}_i))\Big] + \mathcal{L}_{\text{GP}}$$

**Centroid update**: Updated via exponential moving average at each mini-batch:

$$\mathbf{e}_k \leftarrow (1 - \gamma)\mathbf{e}_k + \gamma \cdot \text{mean}_{\{i: y_i = k\}} \mathbf{W}_k \mathbf{h}(\mathbf{x}_i)$$

**Inference**:
- **Prediction**: $$\hat{y} = \arg\max_k K_k(\mathbf{x})$$
- **Uncertainty**: $$\text{uncertainty}(\mathbf{x}) = 1 - \max_k K_k(\mathbf{x})$$
- When all $K_k$ are near zero → high uncertainty (likely OOD)

### 5.4 Limitations of DUQ

- **Difficulty scaling to many classes**: A separate projection matrix $\mathbf{W}_k$ must be maintained for each of $K$ classes, so costs escalate for problems with many classes (e.g., ImageNet with 1,000 classes)
- **Gradient penalty cost**: Computing input gradients adds overhead during training
- **No aleatoric/epistemic decomposition**: The RBF distance is a single scalar, with no mechanism for distinguishing between the two types of uncertainty
- **Extension to regression**: The original design is specialized for classification, and extending it naturally to regression is not straightforward

---

## 6. Comparing the Methods — The Current State of Single-Pass UQ

### 6.1 Comprehensive comparison table

```
+-----------------+----------+-----------+----------+----------+----------+
| Criterion       | EDL      | EDL       | SNGP     | DUQ      | Deep     |
|                 | (Class.) | (Regr.)   |          |          | Ensemble |
+-----------------+----------+-----------+----------+----------+----------+
| Training Cost   | 1x       | 1x        | ~1.1x    | ~1.2x    | 5x       |
| Inference Cost  | 1x       | 1x        | ~1.05x   | 1x       | 5x       |
| Memory          | 1x       | 1x        | ~1.1x    | ~1.1x    | 5x       |
+-----------------+----------+-----------+----------+----------+----------+
| Aleatoric/      | Yes      | Yes       | No       | No       | Yes*     |
| Epistemic Split |          |           |          |          |          |
+-----------------+----------+-----------+----------+----------+----------+
| OOD Detection   | Moderate | Moderate  | Strong   | Strong   | Strong   |
| (AUROC)         |          |           |          |          |          |
+-----------------+----------+-----------+----------+----------+----------+
| Calibration     | Good     | Good      | Very     | Good     | Very     |
| (ECE)           |          |           | Good     |          | Good     |
+-----------------+----------+-----------+----------+----------+----------+
| Theoretical     | Evidence | NIG       | GP +     | RBF +    | Empirical|
| Foundation      | Theory/  | conjugate | Spectral | Lipschitz| (later   |
|                 | Dirichlet| prior     | Norm     |          | Bayesian |
|                 |          |           |          |          | interp.) |
+-----------------+----------+-----------+----------+----------+----------+
| Key             | KL anneal| Evidence  | GP hyper-| Multi-   | M x cost |
| Weakness        | sensitiv.| regulariz.| params   | class    |          |
|                 |          | tuning    | tuning   | scaling  |          |
+-----------------+----------+-----------+----------+----------+----------+
| Regression      | N/A      | Native    | Possible | Not      | Native   |
| Support         |          |           | w/ adapt.| natural  |          |
+-----------------+----------+-----------+----------+----------+----------+
  * Deep Ensemble: when using heteroscedastic members
```

### 6.2 A practitioner's decision guide

Which method to choose depends on **constraints and requirements**:

**Scenario 1: Large-scale virtual screening (regression)**
- Requirements: Evaluating millions to billions of molecules, aleatoric/epistemic decomposition needed
- **Recommendation: Evidential Regression (NIG)**
- Rationale: 1x inference cost, natural uncertainty decomposition, confidence filtering to boost hit rates

**Scenario 2: Real-time classification + OOD detection**
- Requirements: Strict latency constraints, robust OOD detection
- **Recommendation: SNGP**
- Rationale: Deep Ensemble-level OOD detection at ~1x inference cost, minimal modifications to an existing network

**Scenario 3: Small-scale classification + simplicity is the priority**
- Requirements: Few classes ($K < 20$), fast prototyping
- **Recommendation: DUQ**
- Rationale: Simple implementation, intuitive uncertainty metric (centroid distance), robustness via gradient penalty

**Scenario 4: Best possible performance, cost is no object**
- Requirements: Top-tier calibration and OOD detection
- **Recommendation: Deep Ensemble (see Part 2)**
- Rationale: Still the most robust method overall. When cost is not a constraint, the motivation for single-pass diminishes

**Scenario 5: Neural Network Potentials**
- Requirements: Millions of timesteps, per-atom uncertainty, adaptive accuracy
- **Recommendation: Evidential Regression (NIG)**
- Rationale: Per-atom $(\gamma, \nu, \alpha, \beta)$ output provides spatially resolved UQ, enabling adaptive DFT switching

### 6.3 An open question: The debate over EDL's theoretical justification

Despite its practical successes, EDL — and single-pass UQ more broadly — is the subject of **an active debate regarding its theoretical soundness**.

**Critique by Bengs, Hoehne, & Krueger (NeurIPS 2023, "Second opinion needed: communicating uncertainty in medical AI")**: This work points out that EDL's loss function is not theoretically **well-specified** in certain cases. Specifically:

- **In regression**: The way the NIG loss regularizer combines prediction error and evidence creates **perverse incentives** in some scenarios — the model can reduce its loss by deliberately making inaccurate predictions while reporting high uncertainty
- **In classification**: Training is unstable without KL annealing, and the annealing schedule amounts to **heuristic tuning rather than principled posterior inference**

**An alternative perspective by Meinert & Lavin (2023)**: They re-derive the evidential regression loss and propose a **proper scoring rule-based** loss for the NIG parameters. This eliminates the perverse incentives while preserving the uncertainty decomposition.

**Current consensus**:
- EDL is practically useful, but the loss function design requires careful attention
- **Evidence regularizer strength**, **KL annealing schedule**, and **choice of activation function** (ReLU vs. softplus vs. exponential) all significantly impact performance
- A single-pass UQ method with complete theoretical justification remains an open research problem

This is an interesting contrast with the story of Deep Ensembles from Part 2. Deep Ensembles were also proposed without theoretical justification, yet proved empirically outstanding — and a Bayesian interpretation followed later. Whether EDL will follow a similar trajectory, or whether fundamental limitations will be uncovered, is a question that future research will answer.

### 6.4 Practical tips: Pitfalls when implementing single-pass UQ

Key practical considerations when deploying single-pass UQ methods in real projects:

**When implementing EDL**:
- **Activation function choice**: Using **softplus** instead of ReLU for the evidence output can mitigate gradient vanishing. Some studies use $\exp(\cdot)$, but this carries a risk of numerical overflow and should be used with caution
- **KL annealing schedule**: Linear annealing is the most stable choice. Step functions or exponential annealing can cause training instability. Setting $T_{\text{anneal}}$ to 30-50% of the total number of epochs is empirically sound
- **OOD validation**: During training, use an OOD validation set (or a corrupted version of the ID validation set) to monitor whether vacuity/uncertainty is actually elevated for OOD samples

**When implementing SNGP**:
- **Number of random features $D$**: Too few and the kernel approximation is inaccurate; too many and memory usage grows. In practice, $D = 1024 \sim 2048$ is typical
- **Scope of spectral normalization**: While the principle is to apply it to all layers, when BatchNorm is present, apply it only to the convolution/linear layers **before** BatchNorm. Do not apply it to BatchNorm itself
- **Precision matrix reset**: Whether to reset the precision matrix to the prior at the start of each epoch or accumulate it depends on dataset size. For small datasets, resetting tends to work better

**Common pitfalls**:
- **Do not confuse OOD detection with calibration**: A high AUROC (OOD detection) from single-pass UQ does not guarantee good calibration. Always evaluate separately with the calibration metrics discussed in Part 4
- **Combining with post-hoc calibration**: Applying temperature scaling or conformal prediction on top of single-pass UQ output can further improve calibration — the two approaches are not mutually exclusive but **complementary**

---

## 7. Situating Single-Pass UQ in the Bigger Picture

The three single-pass UQ methods discussed in this post embody a fundamentally different **design philosophy** from the Bayesian approximation methods explored in Part 2.

```
UQ Methods Landscape

                    Accuracy of Uncertainty
                    (calibration, OOD detection)
                    ^
                    |
       HMC (gold   |   * Deep Ensemble
       standard) *  |  *
                    | *  SNGP
                    |*
         SWAG *     |   * DUQ
                    |
    MC Dropout *    |   * EDL (Regression)
                    |
         VI *       |   * EDL (Classification)
                    |
      Laplace *     |
                    +-------------------------------> Computational
                    1x     2x     5x    10x   20x     Efficiency
                                                      (inference)

        Legend:
        Left cluster: sampling/ensemble methods (multi-pass)
        Right cluster: single-pass methods (this post)
```

**Summary of the key trade-off**:

- **Multi-pass methods (Ensemble, MC Dropout, SWAG)**: Extract uncertainty from the **disagreement** among multiple models or samples. Theoretically sound (they approximate the posterior) and empirically strong, but expensive
- **Single-pass methods (EDL, SNGP, DUQ)**: Train a single model to **predict its own uncertainty directly**. Minimally expensive, but the fundamental question remains: "Can a model accurately know what it doesn't know?"

Where to position yourself along this trade-off depends on the original question from Part 0 — "How much can we trust this prediction?" — and specifically, **what degree of precision is required in the uncertainty estimate**.

- **When decision costs are low** (e.g., ranking in a web recommendation system): Approximate uncertainty suffices → single-pass
- **When decision costs are high** (e.g., selecting molecules for clinical trials): Precise uncertainty is essential → ensemble or SNGP
- **When inference cost is the binding constraint** (e.g., real-time control, large-scale screening): Single-pass is the only viable option

---

## Conclusion: A single model can say "I don't know"

Let us distill the key messages of this post.

1. **Single-pass UQ was born out of practical necessity.** For virtual screening of millions of molecules, real-time inference, and large-scale simulations, ensembles and sampling are physically impossible. Methods that quantify uncertainty in a single forward pass are **not a nice-to-have but a must-have**.

2. **The three approaches rest on distinct mathematical foundations.** EDL leverages conjugate priors for higher-order uncertainty (Dirichlet over probabilities, NIG over mean + variance). SNGP achieves distance awareness through spectral normalization + Gaussian processes. DUQ relies on RBF kernels to measure centroid distances. Each has clear strengths and weaknesses.

3. **Feature collapse is the central adversary of single-pass UQ.** When a network maps OOD inputs into the same feature region as ID data, no output head can yield meaningful uncertainty. SNGP's spectral normalization and DUQ's gradient penalty represent two different solutions to this problem.

4. **Theoretical justification is not yet complete.** The design of EDL's loss function and the role of its regularizers remain actively debated. Practical success does not guarantee theoretical completeness, and closing this gap is a vibrant research direction.

In the next installment, Part 6, we shift our focus **from methodology to applications**. We will examine how the UQ methods covered in this and preceding posts operate in **real scientific discovery** — from my experience with Bayesian GCNs, to why AlphaFold's pLDDT is not true Bayesian uncertainty, to UQ in Neural Network Potentials, and active learning in autonomous laboratories (lab-in-the-loop). Beyond theoretical rigor, we will explore the tangible value that uncertainty quantification provides in deciding **which experiment to run next at the lab bench**.

---

## References

### Evidential Deep Learning
1. Sensoy, M., Kaplan, L. & Kandemir, M. (2018). "Evidential Deep Learning to Quantify Classification Uncertainty." *NeurIPS*.
2. Amini, A., Schwarting, W., Soleimany, A. & Rus, D. (2020). "Deep Evidential Regression." *NeurIPS*.
3. Soleimany, A. P. et al. (2021). "Evidential Deep Learning for Guided Molecular Property Prediction and Discovery." *ACS Central Science*, 7(8), 1356-1367.
4. Nature Communications (2025). "Evidential Deep Learning for Uncertainty Quantification of Interatomic Potentials."

### Distance-Aware Methods
5. Liu, J. Z., Padhy, S., Ren, J., Lin, Z. & Lakshminarayanan, B. (2023). "A Simple Approach to Improve Single-Model Deep Uncertainty via Distance-Awareness." *JMLR*, 24(42), 1-63.
6. van Amersfoort, J., Smith, L., Teh, Y. W. & Gal, Y. (2020). "Uncertainty Estimation Using a Single Deep Deterministic Neural Network." *ICML*.

### Theoretical Critiques and Extensions
7. Bengs, V., Hoehne, J. & Krueger, D. (2023). "Second opinion needed: communicating uncertainty in medical AI." *NeurIPS*.
8. Meinert, N. & Lavin, A. (2023). "The Unreasonable Effectiveness of Deep Evidential Regression." *AAAI*.

### Background (from earlier parts)
9. Lakshminarayanan, B., Pritzel, A. & Blundell, C. (2017). "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles." *NeurIPS*.
10. Gal, Y. & Ghahramani, Z. (2016). "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning." *ICML*.
11. Ryu, S., Kwon, Y. & Kim, W. Y. (2019). "A Bayesian Graph Convolutional Network for Reliable Prediction of Molecular Properties with Uncertainty Quantification." *Chemical Science*, 10, 8438-8446.
12. Stokes, J. M. et al. (2020). "A Deep Learning Approach to Antibiotic Discovery." *Cell*, 180(4), 688-702.
