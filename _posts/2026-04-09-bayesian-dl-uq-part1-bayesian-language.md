---
title: "Bayesian DL & UQ Part 1: The Language of Bayesian Inference"
date: 2026-04-09 14:05:00 +0900
categories: [AI & ML, ML Theory]
tags: [bayesian-deep-learning, bayesian-inference, weight-posterior, predictive-distribution, marginal-likelihood]
math: true
---

**Bayesian Deep Learning and Uncertainty Quantification — A Journey Toward Trustworthy AI**

*This is Part 1 of an 8-part series.*

- **Part 0:** [Beyond Predictions — Why Uncertainty Matters](/posts/bayesian-dl-uq-part0-why-uncertainty/)
- **Part 1:** The Language of Bayesian Inference (this post)
- **Part 2:** [The Art of Approximation — From Variational to Ensemble](/posts/bayesian-dl-uq-part2-art-of-approximation/)
- **Part 3:** [The Anatomy of Uncertainty — Aleatoric vs. Epistemic](/posts/bayesian-dl-uq-part3-anatomy-of-uncertainty/)
- **Part 4:** [Calibration and Conformal Prediction](/posts/bayesian-dl-uq-part4-calibration-conformal/)
- **Part 5:** [Single-Pass UQ — Evidential Deep Learning and Distance-Aware Methods](/posts/bayesian-dl-uq-part5-single-pass-uq/)
- **Part 6:** [UQ in Science — Molecules, Proteins, and Materials](/posts/bayesian-dl-uq-part6-uq-in-science/)
- **Part 7:** [The Future of UQ — Uncertainty in the Age of Foundation Models](/posts/bayesian-dl-uq-part7-future-foundation-models/)

---

## Hook: Your Model Stands on a Single Point in the Loss Landscape

When you train a deep learning model, you end up with a single weight vector $\mathbf{w}^{\ast}$. This vector is **one point** — the point where SGD (or Adam) converged in a loss landscape spanning millions of dimensions. We use this one point to make predictions for every new input.

But pause for a moment — if you retrain the same model on the same data with a different random seed, the optimizer lands in a different local minimum. The training losses of the two models may be nearly identical, yet their predictions on certain inputs can differ substantially. Which one is correct?

**The Bayesian neural network gives a clear answer: consider both.** Instead of choosing a single weight, **average over all weights that are compatible with the data.** This is the essence of Bayesian inference, and it is the entirety of what this post is about.

---

## 1. The Limitations of Point Estimates

### 1.1 MLE and MAP — Picking a Single Point on the Loss Landscape

Standard deep learning training performs **Maximum Likelihood Estimation (MLE)** or **Maximum A Posteriori (MAP)** estimation.

**MLE** finds the weight that maximizes the data likelihood:

$$\mathbf{w}_{\text{MLE}} = \arg\max_{\mathbf{w}} \; p(\mathcal{D} \mid \mathbf{w}) = \arg\max_{\mathbf{w}} \; \prod_{i=1}^{N} p(y_i \mid \mathbf{x}_i, \mathbf{w})$$

- Taking the log, this is equivalent to **minimizing the negative log-likelihood (NLL)**
- Under a Gaussian likelihood assumption, regression reduces to **minimizing MSE loss**
- Under a Categorical likelihood assumption, classification reduces to **minimizing cross-entropy loss**

**MAP** adds a prior on top:

$$\mathbf{w}_{\text{MAP}} = \arg\max_{\mathbf{w}} \; p(\mathbf{w} \mid \mathcal{D}) = \arg\max_{\mathbf{w}} \; p(\mathcal{D} \mid \mathbf{w}) \, p(\mathbf{w})$$

- A Gaussian prior $p(\mathbf{w}) = \mathcal{N}(\mathbf{0}, \lambda^{-1}\mathbf{I})$ is equivalent to **L2 regularization (weight decay)**
- A Laplace prior $p(\mathbf{w}) \propto \exp(-\lambda \|\mathbf{w}\|_1)$ is equivalent to **L1 regularization**
- In other words, the weight decay we routinely use is actually **MAP estimation under a Gaussian prior**

Both approaches ultimately produce **a single point estimate** $\mathbf{w}^{\ast}$. Prediction is then performed deterministically with this fixed weight:

$$\hat{y} = f(\mathbf{x}^{\ast}; \mathbf{w}^{\ast})$$

The fundamental problem with this approach is that it completely ignores the fact that **countless other points in the loss landscape explain the data equally well**.

### 1.2 Same Data, Different Local Minima, Different Predictions

The loss landscape of modern neural networks is riddled with local minima and saddle points. Training the same architecture on the same dataset with **different random seeds** causes the optimizer to converge to different local minima.

```
Loss Landscape (simplified 2D cross-section)

Loss ^
     |
     |   .                           .
     |  / \         .               / \
     | /   \       / \     .       /   \
     |/     \     /   \   / \     /     \
     |       \   /     \_/   \   /       \
     |        \_/              \_/        \___
     |     w_A*       w_B*       w_C*
     +---------------------------------------------> w
           ^           ^           ^
           |           |           |
        Seed 1      Seed 2      Seed 3

     All three points have similar training loss,
     yet their predictions on certain test inputs can differ.
```

- **w_A***: Predicts activity = 7.2 nM for a new molecule X
- **w_B***: Predicts activity = 5.1 nM for the same molecule X
- **w_C***: Predicts activity = 9.8 nM for the same molecule X

Which one is the "correct" answer? Point estimate methods cannot address this question. The fact that an arbitrary choice — the random seed — determines the prediction is the first limitation of point estimates.

### 1.3 With Limited Data: Overfitting and Overconfidence

The problems with point estimates **grow more severe as data becomes scarcer**.

- **With sufficient data**: Multiple local minima yield similar predictions, and the loss landscape is relatively smooth. Point estimates are reasonable.
- **With limited data**: The loss landscape contains many wide, shallow valleys, and very different solutions can achieve similar loss. The model begins fitting noise (overfitting). Yet no uncertainty is reflected in the predictions whatsoever.

This is the **root cause of overconfidence**. Even in situations where the model should say "I don't know" due to insufficient data, a point estimate model expresses 100% confidence in its predictions — because it simply has no mechanism to express uncertainty.

### 1.4 Reframing the Core Question

To summarize the discussion so far:

- **What point estimates do**: Select a single point $\mathbf{w}^{\ast}$ on the loss landscape and predict solely from that point
- **What we actually want**: Consider **all** weights compatible with the data and produce an answer that includes predictive uncertainty

The mathematical realization of this shift is precisely **Bayesian inference**.

```
Point Estimate                    Bayesian Inference
                                  
   Loss                              Posterior p(w|D)
    ^                                     ^
    |       *  <-- pick one point         |    @@@@
    |      / \                            |   @@@@@@  <-- use the full distribution
    |     /   \                           |  @@@@@@@@
    |    /     \                           | @@@@@@@@@@
    |---/-------\----> w                  |@@@@@@@@@@@@
                                          +-----------> w
                                  
   Prediction: f(x*; w*)          Prediction: integral f(x*;w) p(w|D) dw
   (a single number)              (a distribution — mean + uncertainty)
```

---

## 2. Bayesian Inference 101 — Describing Learning in the Language of Probability

### 2.1 Three Building Blocks

Bayesian inference describes the learning process through the relationships among three probability distributions.

#### (1) Prior: $p(\mathbf{w})$

The **prior** encodes our **prior belief** about the weights before observing any data.

$$p(\mathbf{w}) = \mathcal{N}(\mathbf{0}, \lambda^{-1}\mathbf{I})$$

- This expression encodes the belief that "weights are likely to be near zero, and extremely large values are unlikely"
- $\lambda$ is the **precision** (inverse variance) — larger values pull the weights more strongly toward zero
- The prior is a subjective choice, which is one of the most common criticisms of the Bayesian approach — but as we will see, this subjectivity can be a **strength, not a weakness**
- Practically, the prior acts as regularization: Gaussian prior = L2 regularization, Laplace prior = L1 regularization

**Intuition**: The prior is the "starting point when nothing is known." When data is absent, the prior dominates the prediction; as data accumulates, the influence of the prior fades.

#### (2) Likelihood: $p(\mathcal{D} \mid \mathbf{w})$

The **likelihood** is the probability of the observed data given a particular weight $\mathbf{w}$, where:

$$\mathcal{D} = \lbrace(\mathbf{x}_i, y_i)\rbrace_{i=1}^{N}$$

$$p(\mathcal{D} \mid \mathbf{w}) = \prod_{i=1}^{N} p(y_i \mid \mathbf{x}_i, \mathbf{w})$$

- Under the assumption that data points are **independent and identically distributed (i.i.d.)**, the likelihood factorizes as a product
- **Regression** (Gaussian likelihood):

$$p(y_i \mid \mathbf{x}_i, \mathbf{w}) = \mathcal{N}(y_i \mid f(\mathbf{x}_i; \mathbf{w}),\; \sigma^2)$$

  where $f(\mathbf{x}_i; \mathbf{w})$ is the neural network output and $\sigma^2$ is the observation noise variance

- **Classification** (Categorical likelihood):

$$p(y_i \mid \mathbf{x}_i, \mathbf{w}) = \text{Cat}(y_i \mid \text{softmax}(f(\mathbf{x}_i; \mathbf{w})))$$

**Intuition**: The likelihood measures "how well does this weight explain the data?" Weights that assign high likelihood to the data are weights that fit the data well.

#### (3) Posterior: $p(\mathbf{w} \mid \mathcal{D})$

The **posterior** is the updated belief about the weights after observing the data. By **Bayes' theorem**:

$$\boxed{p(\mathbf{w} \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \mathbf{w}) \, p(\mathbf{w})}{p(\mathcal{D})}}$$

The role of each term is summarized below:

| Term | Name | Role |
|:---|:-----|:-----|
| $p(\mathbf{w} \mid \mathcal{D})$ | Posterior | Belief about the weights after observing the data |
| $p(\mathcal{D} \mid \mathbf{w})$ | Likelihood | How well a given weight explains the data |
| $p(\mathbf{w})$ | Prior | Belief about the weights before observing the data |
| $p(\mathcal{D})$ | Marginal Likelihood (Evidence) | Normalizing constant (average goodness-of-fit across all weights) |

**Intuition**: The posterior is a **compromise** between the prior and the likelihood.

- When data is scarce: the prior dominates, and posterior $\approx$ prior
- When data is abundant: the likelihood dominates, and the posterior concentrates near the MLE
- In between: the prior and likelihood balance each other, and this balance provides a natural form of regularization

```
The Posterior as a Compromise Between Prior and Likelihood

         p(w)
          ^
          |     Prior             Likelihood          Posterior
          |      @@                    @@              @@
          |     @@@@                  @@@@            @@@@
          |    @@@@@@                @@@@@@          @@@@@@
          |   @@@@@@@@              @@@@@@@@        @@@@@@@@
          |  @@@@@@@@@@    +      @@@@@@@@@@   =  @@@@@@@@@@
          | @@@@@@@@@@@@        @@@@@@@@@@@@     @@@@@@@@@@@@
          +---------|------- w ------|----------- w ----|------> w
                  w_prior          w_MLE            w_MAP
          
          Region favored        Region indicated    Posterior forms at the
          by the prior          by the data         compromise between the two
```

### 2.2 Marginal Likelihood: The Meaning of the Denominator

The denominator of Bayes' theorem, the **marginal likelihood** (also called **model evidence**), is defined as:

$$p(\mathcal{D}) = \int p(\mathcal{D} \mid \mathbf{w}) \, p(\mathbf{w}) \, d\mathbf{w}$$

- This integral is the **prior-weighted average of the likelihood over all possible weights**
- Its role is to normalize the posterior into a valid probability distribution (one that integrates to 1)
- When only the shape of the posterior is needed (e.g., for MCMC sampling), $p(\mathcal{D})$ does not need to be computed directly — the proportionality relation $p(\mathbf{w} \mid \mathcal{D}) \propto p(\mathcal{D} \mid \mathbf{w}) \, p(\mathbf{w})$ is sufficient

However, the marginal likelihood carries profound significance in its own right — it is the cornerstone of **model selection**. We return to this in Section 4.

### 2.3 Predictive Distribution: The Heart of Bayesian Prediction

The ultimate goal of Bayesian inference is to make **predictions for a new input $\mathbf{x}^{\ast}$**. Instead of using a single weight, we use the entire posterior:

$$\boxed{p(y^{\ast} \mid \mathbf{x}^{\ast}, \mathcal{D}) = \int p(y^{\ast} \mid \mathbf{x}^{\ast}, \mathbf{w}) \, p(\mathbf{w} \mid \mathcal{D}) \, d\mathbf{w}}$$

This is called the **posterior predictive distribution**, and it is the single most important equation in Bayesian neural networks. Let us unpack it term by term:

- $p(y^{\ast} \mid \mathbf{x}^{\ast}, \mathbf{w})$ — the output distribution for input $\mathbf{x}^{\ast}$ when the weight $\mathbf{w}$ is **fixed**. This is a standard forward pass.
- $p(\mathbf{w} \mid \mathcal{D})$ — the posterior. It encodes the "plausibility" of each weight configuration.
- $\int \cdot \, d\mathbf{w}$ — integration over all possible weights (**marginalization**). Each weight's prediction is weighted by its posterior probability.

**Intuition**: Each $\mathbf{w}$ defines a distinct "model." The posterior predictive is an **ensemble of infinitely many models**, where each model's contribution is proportional to how well it fits the data (its posterior probability).

This is fundamentally different from a point estimate:

| | Point Estimate | Bayesian Predictive |
|:---|:---|:---|
| **Weights used** | A single $\mathbf{w}^{\ast}$ | The full posterior $p(\mathbf{w} \mid \mathcal{D})$ |
| **Prediction** | $f(\mathbf{x}^{\ast}; \mathbf{w}^{\ast})$ (a single number) | $p(y^{\ast} \mid \mathbf{x}^{\ast}, \mathcal{D})$ (a distribution) |
| **Uncertainty** | None | Naturally included (variance of the distribution) |
| **Analogy** | Viewing a landscape from one mountain peak | A composite panorama seen from many peaks |

### 2.4 Mean and Variance of the Predictive Distribution

Two directly useful statistics can be extracted from the posterior predictive distribution.

**Predictive Mean**:

$$\mathbb{E}[y^{\ast} \mid \mathbf{x}^{\ast}, \mathcal{D}] = \int f(\mathbf{x}^{\ast}; \mathbf{w}) \, p(\mathbf{w} \mid \mathcal{D}) \, d\mathbf{w}$$

- This is the **posterior-weighted average of the model outputs**
- It is analogous to the point estimate prediction $f(\mathbf{x}^{\ast}; \mathbf{w}^{\ast})$, but more robust

**Predictive Variance**:

$$\text{Var}[y^{\ast} \mid \mathbf{x}^{\ast}, \mathcal{D}] = \underbrace{\mathbb{E}_{\mathbf{w}}[\text{Var}[y^{\ast} \mid \mathbf{x}^{\ast}, \mathbf{w}]]}_{\text{aleatoric uncertainty}} + \underbrace{\text{Var}_{\mathbf{w}}[\mathbb{E}[y^{\ast} \mid \mathbf{x}^{\ast}, \mathbf{w}]]}_{\text{epistemic uncertainty}}$$

This decomposition follows from the **Law of Total Variance** and naturally separates uncertainty into two types:

- **Aleatoric uncertainty** (first term): The average prediction variance across weights — arising from **inherent noise in the data** and irreducible even with more data
- **Epistemic uncertainty** (second term): The variance of the prediction mean across weights — arising from **insufficient knowledge of the model** and reducible by collecting more data

A detailed analysis of this decomposition is reserved for Part 3, but the key takeaway here is this: **a single operation — marginalization — simultaneously solves both prediction and uncertainty quantification.** No separate uncertainty estimation module is required. In the Bayesian framework, uncertainty is not a byproduct of prediction — it is an integral part of the prediction process itself.

---

## 3. Why Exact Bayesian Inference Is Intractable — A Preview

Despite the mathematical elegance of Bayesian inference, performing it exactly in neural networks is **fundamentally infeasible**. A detailed analysis of this problem is the central topic of Part 2, but here we briefly introduce the four barriers.

### 3.1 Non-Conjugacy: No Closed-Form Solution

For the posterior to be computed in closed form, the prior and likelihood must be in a **conjugate relationship** — that is, the posterior must belong to the same distributional family as the prior.

**An example where conjugacy holds (linear regression)**:

$$\text{Gaussian prior} + \text{Gaussian likelihood (linear model)} \rightarrow \text{Gaussian posterior}$$

In linear regression, the posterior over weights is exactly Gaussian, with analytically computable mean and covariance:

$$p(\mathbf{w} \mid \mathcal{D}) = \mathcal{N}(\mathbf{w} \mid \boldsymbol{\mu}_N, \boldsymbol{\Sigma}_N)$$

where:

$$\boldsymbol{\Sigma}_N = (\sigma^{-2} \mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I})^{-1}, \quad \boldsymbol{\mu}_N = \sigma^{-2} \boldsymbol{\Sigma}_N \mathbf{X}^\top \mathbf{y}$$

**Why conjugacy breaks down in neural networks**: Even a single hidden layer introduces:

$$f(\mathbf{x}; \mathbf{w}) = \mathbf{W}_2 \cdot \sigma(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2$$

The nonlinear activation $\sigma(\cdot)$ destroys the conjugate relationship between the likelihood and the prior. Even with a Gaussian prior, the posterior is no longer Gaussian — no closed-form solution exists.

### 3.2 The Curse of Dimensionality: Integration Over Millions of Dimensions

Without a closed-form solution, we must resort to numerical methods. But the weight space of neural networks spans millions to hundreds of billions of dimensions.

- ResNet-50: $d = 25.6\text{M}$ dimensions
- GPT-3: $d = 175\text{B}$ dimensions

If we attempt numerical integration with a modest grid of 10 points per dimension, ResNet-50 would require $10^{25,600,000}$ function evaluations — a physically impossible scale.

### 3.3 Multimodality: A Complex Posterior Landscape

The posterior of a neural network has numerous modes.

- **Functionally distinct modes** corresponding to different local minima
- **Equivalent modes** arising from permutation symmetry of hidden neurons ($h!$ of them)
- Transitions between these modes are extremely difficult even for MCMC methods

### 3.4 Storage Costs: The Expense of Representing the Posterior Itself

Even the simplest Gaussian approximation $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ to the posterior requires:

- Full covariance matrix: $O(d^2)$ storage — approximately **1.3 PB** for ResNet-50
- Matrix inversion / Cholesky decomposition: $O(d^3)$ computation
- Completely infeasible in practice

### 3.5 The Four Barriers Reinforce Each Other

```
Non-conjugacy ─────→ No closed-form solution
       │                        │
       ▼                        ▼
High dimensionality ──→ Numerical integration infeasible
       │                        │
       ▼                        ▼
Multimodality ────────→ Simple approximations inadequate
       │                        │
       ▼                        ▼
Storage costs ────────→ Rich representations impractical
```

These four barriers are not independent. The absence of conjugacy forces reliance on numerical methods, but high dimensionality renders numerical integration infeasible. Even if the posterior could be explored efficiently, multimodality makes simple approximations inaccurate, and richer approximations are prohibited by storage costs.

**In Part 2, we analyze each of these four barriers in detail and systematically compare the approximation methods developed over the past 30 years to circumvent them — Variational Inference, MC Dropout, Deep Ensembles, SWAG, Laplace Approximation, and HMC.**

---

## 4. Bayesian vs. Frequentist — A Pragmatic Perspective

"Granted, Bayesian inference is theoretically elegant. But if exact inference is intractable, why bother with the Bayesian framework at all?"

The answer lies in four practical advantages.

### 4.1 Bayesian Occam's Razor: Automatic Regularization

The marginal likelihood $p(\mathcal{D})$ possesses a remarkable property: **it automatically penalizes models that are overly complex.**

Here is the intuition. The marginal likelihood is the average likelihood under the prior:

$$p(\mathcal{D}) = \int p(\mathcal{D} \mid \mathbf{w}) \, p(\mathbf{w}) \, d\mathbf{w}$$

- **An overly simple model $\mathcal{M}_1$**: Lacks the expressiveness to explain the data well regardless of the weight choice. The likelihood $p(\mathcal{D} \mid \mathbf{w})$ is low everywhere, so the integral is low.

- **An overly complex model $\mathcal{M}_3$**: Has a vast weight space, so the prior probability is spread thinly. Some weight configurations explain this particular dataset very well, but the majority do not. Because the prior is spread widely, the prior probability mass assigned to the "good" region is small. The integral ends up at a middling value.

- **A model of appropriate complexity $\mathcal{M}_2$**: The prior concentrates on the weight region that explains the data well, yielding a high marginal likelihood.

```
Bayesian Occam's Razor

  p(D|M)
    ^
    |                     @@@
    |                    @@@@@
    |        @@         @@@@@@@
    |       @@@@       @@@@@@@@@         @@@@@@@@@@@@@@
    |      @@@@@@     @@@@@@@@@@@       @@@@@@@@@@@@@@@@
    |     @@@@@@@@   @@@@@@@@@@@@@     @@@@@@@@@@@@@@@@@@
    |    @@@@@@@@@@  @@@@@@@@@@@@@@   @@@@@@@@@@@@@@@@@@@@
    +----@@@@@@@@@@--@@@@@@@@@@@@@@---@@@@@@@@@@@@@@@@@@@@@@---->
         M_1 (too    M_2 (just       M_3 (too complex)
          simple)     right)
    
    M_1: Cannot explain the data (low likelihood everywhere)
    M_2: Appropriate complexity (highest marginal likelihood)
    M_3: Prior mass spread too thinly across a vast space
```

This is the **Bayesian Occam's razor**. Without a separate validation set or cross-validation, the marginal likelihood itself automatically trades off model complexity against data fit.

### 4.2 Model Selection via Marginal Likelihood

The Bayesian Occam's razor directly leads to a principle for **model selection**. Given competing models $\mathcal{M}_1, \mathcal{M}_2, \ldots$, we can select the best model by comparing their marginal likelihoods:

$$\frac{p(\mathcal{M}_i \mid \mathcal{D})}{p(\mathcal{M}_j \mid \mathcal{D})} = \frac{p(\mathcal{D} \mid \mathcal{M}_i)}{p(\mathcal{D} \mid \mathcal{M}_j)} \cdot \frac{p(\mathcal{M}_i)}{p(\mathcal{M}_j)}$$

- The left-hand side is the **posterior odds**: the relative probability of the two models after observing the data
- The first factor $p(\mathcal{D} \mid \mathcal{M}_i) / p(\mathcal{D} \mid \mathcal{M}_j)$ is the **Bayes factor**: the evidence the data provides in favor of one model over the other
- The second factor is the **prior odds**: the prior preference between the two models (typically set to be equal)

This is the Bayesian counterpart of information criteria such as AIC and BIC, and it provides a theoretically more coherent framework.

In practice, computing the marginal likelihood exactly is difficult (it is itself an integration problem), but several approximation methods exist:

- **Laplace approximation**:

$$\log p(\mathcal{D}) \approx \log p(\mathcal{D} \mid \mathbf{w}_{\text{MAP}}) + \log p(\mathbf{w}_{\text{MAP}}) - \frac{1}{2} \log \det(\mathbf{H}) + \frac{d}{2} \log(2\pi)$$
- **ELBO (Evidence Lower Bound)**: A lower bound obtained as a byproduct of Variational Inference
- MacKay (1992) used this Laplace-based marginal likelihood approximation to propose **automatic hyperparameter optimization (regularization strength)** for neural networks — one of the founding contributions of BNN research

### 4.3 Active Learning: Uncertainty Guides the Next Experiment

One of the most direct practical applications of the Bayesian framework is **active learning**.

In drug discovery, one must choose which molecules to test experimentally from among millions of candidates. Testing every molecule is prohibitive in both time and cost. Which molecules should be tested first?

- **Exploitation**: Select molecules for which the model predicts the highest activity — risks repeating what is already known
- **Exploration**: Select molecules about which the model is most uncertain — explores new chemical space
- **Bayesian approach**: Automatically balances the exploration-exploitation trade-off by **jointly considering the mean and variance of the predictive distribution**

A representative **acquisition function**:

$$\alpha(\mathbf{x}) = \mu(\mathbf{x}) + \kappa \cdot \sigma(\mathbf{x})$$

where $\mu(\mathbf{x})$ is the predictive mean, $\sigma(\mathbf{x})$ is the predictive standard deviation, and $\kappa$ is a hyperparameter controlling the exploration-exploitation balance. This is the **Upper Confidence Bound (UCB)** acquisition function, a core element of Bayesian optimization.

**Key insight**: In the Bayesian framework, uncertainty is not a byproduct of prediction — it is the **essential ingredient for decision-making**. High-uncertainty regions = regions where information can be gained = regions worth exploring.

### 4.4 Continual Learning: The Posterior Becomes the Next Prior

Another structural advantage of the Bayesian framework is its natural extension to **sequential learning**.

Suppose we train on a first dataset $\mathcal{D}_1$ and obtain the posterior $p(\mathbf{w} \mid \mathcal{D}_1)$. When new data $\mathcal{D}_2$ arrives:

$$p(\mathbf{w} \mid \mathcal{D}_1, \mathcal{D}_2) = \frac{p(\mathcal{D}_2 \mid \mathbf{w}) \, p(\mathbf{w} \mid \mathcal{D}_1)}{p(\mathcal{D}_2 \mid \mathcal{D}_1)}$$

- $p(\mathbf{w} \mid \mathcal{D}_1)$ now serves as the **new prior**
- Information learned from the previous data is naturally preserved
- This provides a principled approach to the **catastrophic forgetting** problem

```
Bayesian Update Cycle (Continual Learning)

   Prior          Data D_1         Posterior_1
  p(w)    ----->  Train    ----->  p(w|D_1)
                                      |
                                      | (use posterior as the next prior)
                                      v
                  Data D_2         Posterior_2
                  Train    ----->  p(w|D_1, D_2)
                                      |
                                      | (use posterior as the next prior)
                                      v
                  Data D_3         Posterior_3
                  Train    ----->  p(w|D_1, D_2, D_3)
                                      |
                                      v
                                    ...

   At each stage, the result of previous learning (posterior)
   becomes the starting point (prior) for the next round,
   allowing knowledge to accumulate progressively.
```

This is impossible with point estimate methods. The weight $\mathbf{w}^{\ast}$ obtained via SGD carries no information about "the degree of confidence in this weight." The posterior, by contrast, encodes the certainty in each weight direction through the shape of the distribution, enabling the model to decide what to preserve and what to update when new data arrives.

In practice, **Elastic Weight Consolidation (EWC)** (Kirkpatrick et al., PNAS 2017) implements this idea via the Laplace approximation — weights with high posterior precision (Fisher information) resist change, while weights with low precision are updated freely.

---

## 5. A BNN in Action: 1D Regression Example

To ground the theoretical discussion in something concrete, let us examine how a Bayesian neural network behaves in the simplest possible setting — **1D regression**.

### 5.1 Problem Setup

- **Data**: $N$ pairs where $y\_i = \sin(x\_i) + \epsilon\_i$ and $\epsilon\_i \sim \mathcal{N}(0, 0.1^2)$
- **Observed region**: $N = 20$ data points sampled from $x \in [-2, 2]$
- **Prediction region**: The full range $x \in [-5, 5]$ (including regions with no data)
- **Model**: A 1-hidden-layer neural network with 50 hidden neurons and ReLU activation

### 5.2 Point Estimate vs. Posterior Predictive

**Point Estimate (MAP)**:

```
Point Estimate (MAP) Prediction

  y ^
    |
  2 |                                          .....
    |                               ...........
  1 |               @@    @  ......
    |            @@   @@@@  .
  0 |----@@----@@----------.--------------------------> x
    |  .      @          ..
 -1 |  .  @@           ..
    | .               ..
 -2 |.              ...
    |
    +----|----|----|----|----|----|----|----|----|---->
     -5  -4  -3  -2  -1   0   1   2   3   4   5

    @ = data points     ... = MAP prediction (single curve)

    Problem: Even in regions [-5,-2] and [2,5] where no data exists,
    the model confidently predicts a single value.
    There is no information about uncertainty whatsoever.
```

**Bayesian Posterior Predictive**:

```
Bayesian Posterior Predictive Distribution

  y ^
    |
  3 |:                                          ::::
    |::                                       ::::::
  2 | ::                                     :::..::
    |  :::                           ::::.......::::
  1 |   :::         @@    @  ...............  ::::
    |    ::::    @@   @@@@  .@@@@@@@@@@@@.  ::::
  0 |-----::::-@@----@@---.@@@@@@@@@@@@.-:::---------> x
    |  ..   :::     @   ..@@@@@@@@@@..  :::
 -1 | ...   ::::       ..@@@@@@@@..   ::::
    |  ..     ::::   ...            ::::
 -2 |  :::     :::::..            ::::
    |   ::::     ::::           ::::
 -3 |    :::::                :::::
    |
    +----|----|----|----|----|----|----|----|----|---->
     -5  -4  -3  -2  -1   0   1   2   3   4   5

    @ = data points
    . = predictive mean (posterior-weighted average)
    : = uncertainty band (predictive std)
    @@ (dense) = narrow band (data-rich region: low uncertainty)

    Key observations:
    1. In [-2, 2] where data exists: narrow band (confident)
    2. In data-free regions: wide band (uncertain)
    3. Band widens progressively farther from the data
```

The key observations from this visualization are:

- **Data-rich region ($x \in [-2, 2]$)**: The posterior is tightly concentrated, so different weights yield similar predictions. The predictive band is narrow.
- **Data-free regions ($x < -2$ or $x > 2$)**: Weights sampled from the posterior produce wildly different predictions. The predictive band widens.
- **Transition regions**: Near the data boundary, the band broadens gradually, and uncertainty increases smoothly.

**This is the core value proposition of Bayesian neural networks**: without any separate calibration or additional module, marginalization — a single operation — automatically captures "where we know and where we don't."

### 5.3 The Influence of the Prior: Strong vs. Weak

Let us examine how the prior strength ($\lambda$, the precision) affects the behavior of a BNN.

**Strong Prior (large $\lambda$)**:

- Weights are tightly constrained near zero
- In function space: favors simple, smooth functions
- With limited data: the prior dominates predictions, potentially preventing the model from learning complex patterns (underfitting)
- Uncertainty: in data-free regions, predictions regress toward zero with a relatively narrow band

**Weak Prior (small $\lambda$)**:

- Weights have high degrees of freedom
- In function space: permits complex, oscillating functions
- With limited data: the likelihood dominates, risking fitting to noise (overfitting)
- Uncertainty: in data-free regions, the band becomes very wide

```
Posterior Predictive Comparison by Prior Strength

  Strong Prior (lambda = 10)        Weak Prior (lambda = 0.01)
  
  y ^                                y ^
    |                                  |::::
    | ::                               |:::::             ::::
    |::::  @@  @@ ..........           |:::::  @@  @@ ....:::::
    |:::@@  @@@@.@@@@@@@@@@::          |:::@@  @@@@.@@@@@@:::::
    |--@@---@@-.@@@@@@@@@--::--        |-@@--@@--.@@@@@@@@-::::-->
    |  @   ..@@@@@@@@@@  ::::          |.@  ..@@@@@@@@    :::::
    |  ....               :::          |......             ::::
    |                                  |::::
    +----|----|----|----|--->           +----|----|----|----|--->
     -5  -2   0   2   5                -5  -2   0   2   5

  Outside data: gentle convergence    Outside data: widely diverging
  toward zero, narrow band            broad band

  → Risk of underfitting              → Honest uncertainty representation
    but may underestimate uncertainty   but predictions may be unstable
```

**Practical implications**:

- The prior choice is "subjective," but this is not a weakness — it is a **channel for injecting domain knowledge into the model**
- Example: In molecular property prediction, the domain knowledge that "binding energies are not extremely large" can be encoded through the scale of the prior
- Whether a prior is appropriate can be assessed via **posterior predictive checks** (verifying that the model's predictions are consistent with the actual data distribution)
- **Automatic relevance determination (ARD)** (MacKay, 1992), which uses the marginal likelihood, is a method for automatically optimizing prior hyperparameters from data

### 5.4 Intuition Behind Posterior Sampling

Let us walk through the process that generates the visualizations above, step by step:

1. **Sample weights from the posterior**: $\mathbf{w}_1, \mathbf{w}_2, \ldots, \mathbf{w}_T \sim p(\mathbf{w} \mid \mathcal{D})$
2. **Predict with each weight**: $f_t(x) = f(x; \mathbf{w}_t)$ for $t = 1, \ldots, T$
3. **Compute summary statistics**:
   - Predictive mean: $\hat{\mu}(x) = \frac{1}{T} \sum_{t=1}^{T} f_t(x)$
   - Predictive variance: $\hat{\sigma}^2(x) = \frac{1}{T} \sum_{t=1}^{T} (f_t(x) - \hat{\mu}(x))^2 + \sigma_{\text{noise}}^2$

```
Individual Functions Sampled from the Posterior

  y ^
    |
  2 |    ......                            -----
    |  ..      ..                      ---/
  1 | .    @@   @.  ------            /
    |.  @@   @@@@ /       \          /
  0 |--@@---@@--/--------- \--------/-----------> x
    | .    @  /              \     /
 -1 |.     --                 \  /
    |------                    \/
 -2 | ....                      ......
    |
    +----|----|----|----|----|----|----|---->
     -5  -3  -2  -1   0   1   2   3   5

    Each curve (... --- ) represents the prediction function
    of a single weight sampled from the posterior.

    Data region: all functions agree → low variance
    Outside data: functions diverge widely → high variance
```

This is the core mechanism of BNNs. Each weight sampled from the posterior represents one plausible model. In regions constrained by data, the models converge; in unconstrained regions, the models diverge. The degree of this divergence is precisely the epistemic uncertainty.

---

## Mathematical Supplement: Closed-Form Solution for Bayesian Linear Regression

To deepen the understanding of BNNs, we briefly present the case of **linear models**, where the posterior can be computed exactly. This serves as the "ideal limit" of BNNs, showing precisely how Bayesian inference works without any approximation.

**Model**: $y = \mathbf{w}^\top \mathbf{x} + \epsilon$, $\epsilon \sim \mathcal{N}(0, \sigma^2)$

**Prior**: $p(\mathbf{w}) = \mathcal{N}(\mathbf{0}, \lambda^{-1} \mathbf{I})$

**Posterior** (closed-form):

$$p(\mathbf{w} \mid \mathcal{D}) = \mathcal{N}(\mathbf{w} \mid \boldsymbol{\mu}_N, \boldsymbol{\Sigma}_N)$$

where:

$$\boldsymbol{\Sigma}_N = \left(\sigma^{-2} \mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I}\right)^{-1}$$

$$\boldsymbol{\mu}_N = \sigma^{-2} \boldsymbol{\Sigma}_N \mathbf{X}^\top \mathbf{y}$$

**Predictive distribution** (closed-form):

$$p(y^{\ast} \mid \mathbf{x}^{\ast}, \mathcal{D}) = \mathcal{N}\left(y^{\ast} \mid \boldsymbol{\mu}_N^\top \mathbf{x}^{\ast}, \; \sigma^2 + {\mathbf{x}^{\ast}}^\top \boldsymbol{\Sigma}_N \mathbf{x}^{\ast}\right)$$

The two terms in the predictive variance have clear interpretations:

- $\sigma^2$ — **aleatoric uncertainty**: the irreducible uncertainty due to observation noise
- ${\mathbf{x}^{\ast}}^\top \boldsymbol{\Sigma}_N \mathbf{x}^{\ast}$ — **epistemic uncertainty**: uncertainty about the weights propagated into prediction space. As more data is collected, $\boldsymbol{\Sigma}_N$ shrinks and this term converges to zero.

This decomposition holds exactly for linear models and approximately for BNNs. Part 3 will explore this uncertainty decomposition in full depth within the BNN context.

**Marginal likelihood** (closed-form):

$$\log p(\mathcal{D}) = -\frac{N}{2}\log(2\pi\sigma^2) - \frac{1}{2}\mathbf{y}^\top(\sigma^2\mathbf{I} + \lambda^{-1}\mathbf{X}\mathbf{X}^\top)^{-1}\mathbf{y} - \frac{1}{2}\log\det(\sigma^2\mathbf{I} + \lambda^{-1}\mathbf{X}\mathbf{X}^\top)$$

The third term $-\frac{1}{2}\log\det(\cdot)$ acts as a **complexity penalty** — it grows with the effective number of model parameters, thereby reducing the marginal likelihood. This is the mathematical realization of the Bayesian Occam's razor described in Section 4.1.

---

## Summary: Key Messages of Part 1

Here is a recap of the main ideas covered in this post:

**1. The limitations of point estimates**
- MLE/MAP select a single point on the loss landscape and cannot express uncertainty
- The same data with different random seeds produces different predictions, with no way to determine which is correct
- The overconfidence problem is especially severe when data is limited

**2. The core structure of Bayesian inference**
- **Prior** $p(\mathbf{w})$: belief about the weights before learning
- **Likelihood** $p(\mathcal{D} \mid \mathbf{w})$: the data evaluates the weights
- **Posterior** $p(\mathbf{w} \mid \mathcal{D}) \propto p(\mathcal{D} \mid \mathbf{w}) \, p(\mathbf{w})$: updated belief after observing the data
- **Predictive distribution**: $p(y^{\ast} \mid \mathbf{x}^{\ast}, \mathcal{D}) = \int p(y^{\ast} \mid \mathbf{x}^{\ast}, \mathbf{w}) \, p(\mathbf{w} \mid \mathcal{D}) \, d\mathbf{w}$

**3. Marginalization = the source of uncertainty**
- Averaging over all possible weights (marginalization) naturally yields uncertainty
- No separate uncertainty module is needed — uncertainty is built into the prediction process itself

**4. The intractability of exact inference**
- Four mutually reinforcing barriers: non-conjugacy, the curse of dimensionality, multimodality, and storage costs
- This is what necessitates approximation methods, the subject of Part 2

**5. Practical advantages of the Bayesian framework**
- Bayesian Occam's razor (automatic regularization)
- Model selection via marginal likelihood
- Natural exploration-exploitation balance in active learning
- Continual learning: the posterior becomes the next prior

---

## Next Up: Part 2 — The Art of Approximation

Having laid the mathematical foundations of Bayesian inference in Part 1, Part 2 turns to the practical question:

**"If the exact posterior is intractable, how do we work around it?"**

We will systematically compare the approximation methods developed over the past three decades — Variational Inference (Bayes by Backprop), MC Dropout, Deep Ensembles, SWAG, Laplace Approximation, and Hamiltonian Monte Carlo — analyzing which of the four barriers each method trades off and how.

---

## References

1. MacKay, D. J. C. "A Practical Bayesian Framework for Backpropagation Networks." *Neural Computation*, 4(3):448-472, 1992.
2. Neal, R. M. *Bayesian Learning for Neural Networks*. Springer, Lecture Notes in Statistics, 1996.
3. Jospin, L. V., Laga, H., Boussaid, F., Buntine, W., & Bennamoun, M. "Hands-On Bayesian Neural Networks — A Tutorial for Deep Learning Users." *IEEE Computational Intelligence Magazine*, 17(2):29-48, 2022.
4. Blundell, C., Cornebise, J., Kavukcuoglu, K., & Wierstra, D. "Weight Uncertainty in Neural Networks." *Proceedings of the 32nd International Conference on Machine Learning (ICML)*, 2015.
5. Wilson, A. G. & Izmailov, P. "Bayesian Deep Learning and a Probabilistic Perspective of Generalization." *Advances in Neural Information Processing Systems (NeurIPS)*, 33, 2020.
6. Bishop, C. M. *Pattern Recognition and Machine Learning*. Springer, Chapter 3 (Linear Models for Regression), 2006.
7. Murphy, K. P. *Machine Learning: A Probabilistic Perspective*. MIT Press, Chapters 5 & 7, 2012.
8. Kirkpatrick, J. et al. "Overcoming Catastrophic Forgetting in Neural Networks." *Proceedings of the National Academy of Sciences (PNAS)*, 114(13):3521-3526, 2017.
9. Ryu, S., Kwon, Y., & Kim, W. Y. "A Bayesian Graph Convolutional Network for Reliable Prediction of Molecular Properties with Uncertainty Quantification." *Chemical Science*, 10(36):8438-8446, 2019.
