---
title: "Bayesian DL & UQ Part 7: The Future of UQ — Uncertainty in the Age of Foundation Models"
date: 2026-04-09 14:35:00 +0900
categories: [AI & ML, ML Theory]
tags: [bayesian-deep-learning, foundation-models, llm-uncertainty, bayesian-lora, autonomous-labs]
math: true
---

**Bayesian Deep Learning and Uncertainty Quantification — A Journey Toward Trustworthy AI**

*This is Part 7 of an 8-part series.*

- **Part 0:** [Beyond Predictions — Why Uncertainty Matters](/posts/bayesian-dl-uq-part0-why-uncertainty/)
- **Part 1:** [The Language of Bayesian Inference](/posts/bayesian-dl-uq-part1-bayesian-language/)
- **Part 2:** [The Art of Approximation — From Variational to Ensemble](/posts/bayesian-dl-uq-part2-art-of-approximation/)
- **Part 3:** [The Anatomy of Uncertainty — Aleatoric vs. Epistemic](/posts/bayesian-dl-uq-part3-anatomy-of-uncertainty/)
- **Part 4:** [Calibration and Conformal Prediction](/posts/bayesian-dl-uq-part4-calibration-conformal/)
- **Part 5:** [Single-Pass UQ — Evidential Deep Learning and Distance-Aware Methods](/posts/bayesian-dl-uq-part5-single-pass-uq/)
- **Part 6:** [UQ in Science — Molecules, Proteins, and Materials](/posts/bayesian-dl-uq-part6-uq-in-science/)
- **Part 7:** The Future of UQ — Uncertainty in the Age of Foundation Models (this post)

---

## Hook: Standing Before Hundreds of Billions of Parameters

In 2019, when I was writing the Bayesian GCN paper, the models I worked with had on the order of hundreds of thousands of parameters. Learning per-layer dropout rates with Concrete Dropout and estimating predictive distributions via MC sampling was enough to boost the hit rate of EGFR virtual screening by 138%. The question we wrestled with back then was: **"Does Bayesian treatment help with molecular property prediction?"**

Seven years later, in 2026, the question has changed entirely.

GPT-4 is estimated to exceed one trillion parameters, Llama 3.1 has 405 billion, and ESM-2 — a scientific foundation model for proteins — has 15 billion. As these models become core infrastructure for scientific decision-making, from drug candidate proposals to materials property prediction to protein structure generation, the new question is this:

**"How much can we trust the answers that these hundreds of billions of parameters produce?"**

Papamarkou et al. (ICML 2024) took a clear stance on this question: **"Bayesian Deep Learning is Needed in the Age of Large-Scale AI."** The larger the model and the more consequential the decisions it informs, the less optional uncertainty quantification becomes — it becomes a necessity.

But there is a problem. As we saw in Part 2, even the full covariance posterior for ResNet-50 with its 25.6 million parameters would require 1.3 petabytes of storage. What about Llama 3.1 with 405 billion parameters? The resulting number cannot fit in the margins of this post.

**A new paradigm is needed.**

---

## 1. UQ in the LLM Era — New Challenges

### 1.1 The Scale Barrier — Confronting the Numbers

Let us look concretely at the scale of the challenges that UQ faces in the Foundation Model era:

```
Model Scale vs. Bayesian Inference Feasibility
=============================================

Model            Params        Full Covariance    Diagonal VI     Status
-----------      ----------    ---------------    -----------     ------
Bayesian GCN     ~100K         ~40 GB             ~400 KB         Feasible
(Ryu et al. 2019)

ResNet-50        25.6M         1.3 PB             ~100 MB         Partial
(2015)                                                            (Last-layer)

BERT-Base        110M          48 PB              ~440 MB         Partial
(2018)                                                            (Subnetwork)

Llama 3.1        405B          ~10^23 bytes       ~1.6 TB         Impossible
(2024)                         (= 10 million      (diagonal       (full)
                                Exabytes)          alone!)

GPT-4            ~1T+          ???                ~4 TB+          Impossible
(2023)                                                            (full)

=============================================
               Full Bayesian inference becomes
               fundamentally impossible beyond
               ~10M parameters
```

- **Full covariance posterior**: $O(d^2)$ storage and $O(d^3)$ computation for $d$ parameters — physically impossible once $d$ reaches the billions
- **Diagonal VI (Bayes by Backprop)**: 2x memory for each parameter (mean + variance) — terabytes at the hundreds-of-billions scale
- **Deep ensemble**: $M$ full copies of the model — $M \times$ inference cost, $M \times$ memory
- **MC Dropout**: $T$ forward passes at inference time — a $T$-fold increase on top of already expensive LLM inference

**The bottom line**: None of the traditional BDL methods from Part 2 work directly at Foundation Model scale. This is not a problem that "slightly more efficient implementations" will solve — it is a scale barrier that demands a paradigm shift.

### 1.2 Token-level vs. Sequence-level Uncertainty

UQ for LLMs involves a structural complexity that is fundamentally different from traditional discriminative models:

- **Token-level uncertainty**: Uncertainty in predicting the next token
  - The entropy of $p(x_t \mid x_{\lt t})$ at each step of an autoregressive model
  - Relatively simple to compute — directly measurable from the entropy of the softmax output
  - However, **high token entropy does not necessarily imply high sequence uncertainty**
  - Example: "The capital of France is ___" — many tokens are grammatically possible, but semantically there is virtually no uncertainty

- **Sequence-level uncertainty**: Uncertainty over the entire generated output
  - Uncertainty over $p(\mathbf{x}_{1:T})$ — the distribution across all possible sequences
  - Raising the sampling temperature for the same prompt produces diverse answers
  - **The semantic equivalence problem**: "8.2 nM" and "approximately eight nanomolar" are different token sequences with the same meaning
  - Quantifying uncertainty at the semantic level remains an open problem

**Token uncertainty:**

$$H[p(x_t \mid x_{\lt t})] = -\sum_{v \in \mathcal{V}} p(v \mid x_{\lt t}) \log p(v \mid x_{\lt t})$$

**Sequence uncertainty:**

$$H[p(\mathbf{x}_{1:T})] \neq \sum_{t=1}^{T} H[p(x_t \mid x_{\lt t})]$$

The inequality holds because of **conditional dependencies** between tokens: sequence entropy is not a simple sum of individual token entropies. The choice of an early token can completely reshape the distribution of all subsequent tokens, and this chain of dependencies makes sequence-level uncertainty fundamentally more complex.

### 1.3 Hallucination and UQ

**Hallucination** in LLMs — the phenomenon of confidently generating content that is factually wrong — is particularly interesting from a UQ perspective:

- **An extreme failure of epistemic uncertainty**: The model fails to produce an "I don't know" signal and responds with high confidence even to questions far outside its training distribution
- **Calibration breakdown**: In regions where hallucination occurs, there is a systematic gap between the model's internal confidence and its actual accuracy
- **A structural cause**: Autoregressive generation is inherently a process of selecting "the most plausible next token," with no built-in mechanism for judging "is there factual evidence for this answer?"
- **Can UQ mitigate this?**: Some attempts flag high-token-entropy regions as hallucination risk zones, but **confidently wrong** hallucinations — the most dangerous kind — cannot be caught by token entropy alone

This is the central challenge of Foundation Model UQ: **when the model is wrong, uncertainty should be high, but the most dangerous errors (hallucinations) often exhibit low uncertainty.**

---

## 2. Bayesian Fine-tuning — Solutions in Low-Dimensional Subspaces

### 2.1 The Key Insight: "Where to Apply Bayesian Treatment"

At the close of Part 2, we confirmed an important lesson: **a precise approximation in a critical subspace outperforms a crude approximation across the entire weight space.** Last-Layer Laplace was the first piece of evidence, and the Bayesian LoRA family extends this principle into the Foundation Model era.

**LoRA (Low-Rank Adaptation)** freezes the pretrained weight matrix $\mathbf{W}_0 \in \mathbb{R}^{d \times k}$ and learns only a low-rank update $\Delta\mathbf{W} = \mathbf{B}\mathbf{A}$:

$$\mathbf{W} = \mathbf{W}_0 + \mathbf{B}\mathbf{A}, \quad \mathbf{B} \in \mathbb{R}^{d \times r}, \; \mathbf{A} \in \mathbb{R}^{r \times k}, \; r \ll \min(d, k)$$

Here the rank $r$ is typically 4–64. Fine-tuning Llama 3.1 (405B parameters) with LoRA ($r=16$) means training only a few million parameters — less than 0.001% of the total. **This low-dimensional subspace is the ideal target for Bayesian treatment.**

```
Bayesian LoRA: Uncertainty in the Low-Rank Subspace
====================================================

Frozen pretrained weights          Bayesian low-rank adapter
+---------------------------+      +---------------------------+
|                           |      |                           |
|   W_0 (fixed, d x k)     |  +   |  B * A  (Bayesian, r<<d) |
|                           |      |                           |
|   405 billion params      |      |  ~2 million params        |
|   (deterministic)         |      |  (probabilistic)          |
|                           |      |                           |
+---------------------------+      +---------------------------+
         |                                   |
         +------------ W = W_0 + BA ---------+
                           |
                    p(BA | D) = posterior
                    over adapter weights
                           |
              +------------+------------+
              |                         |
         Mean prediction        Uncertainty estimate
         E[f(x; W_0+BA)]       Var[f(x; W_0+BA)]
```

### 2.2 Bayesian LoRA (Yang et al., 2024)

Yang et al. (2024) is a pioneering study that revealed the **structural correspondence** between LoRA fine-tuning and GP (Gaussian Process) inference:

- **Key observation**: The output of a low-rank adapter is mathematically equivalent to the output of a random-feature-approximated GP
- Placing a Gaussian prior on the LoRA parameters $\mathbf{A}, \mathbf{B}$ yields a predictive distribution in output space that matches the GP predictive distribution
- **Practical implication**: The well-known closed-form GP posterior can be applied directly to LoRA
- Post-hoc approach: Bayesian treatment is added on top of an already fine-tuned LoRA, so **no additional training is required**

$$p(\mathbf{w}_{\text{LoRA}} \mid \mathcal{D}) = \mathcal{N}(\boldsymbol{\mu}_{\text{post}}, \boldsymbol{\Sigma}_{\text{post}})$$

Here $\mathbf{w}_{\text{LoRA}}$ is the vectorized LoRA parameters, and the posterior is given in closed-form Gaussian.

### 2.3 SWAG-LoRA (2024)

This method applies the core idea of **SWAG (Stochastic Weight Averaging Gaussian)** from Part 2 — approximating the posterior using the first two moments of the SGD trajectory — to LoRA parameters:

- Collects the **running mean** and **low-rank covariance** of LoRA parameters from the SGD trajectory during fine-tuning
- Near-zero additional computational cost: simply save checkpoints during fine-tuning
- Low-rank ($K$ rank) + diagonal covariance representation:

$$q(\mathbf{w}_{\text{LoRA}}) = \mathcal{N}\left(\bar{\mathbf{w}}, \frac{1}{2}\left(\boldsymbol{\Sigma}_{\text{diag}} + \frac{1}{K}\mathbf{D}\mathbf{D}^T\right)\right)$$

where $\mathbf{D}$ is the deviation matrix collected from the SGD trajectory.

### 2.4 BLoB — Bayesian LoRA by Backpropagation (Wang et al., 2024)

BLoB applies the **Bayes by Backprop** principle from Part 2 to LoRA:

- Each LoRA parameter is treated as a **Gaussian random variable**: $w \sim \mathcal{N}(\mu, \sigma^2)$
- The mean $\mu$ and variance $\sigma^2$ (more precisely, $\rho = \log(\exp(\sigma) - 1)$ ) are **jointly learned via backpropagation**
- **Reparameterization trick**: $w = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1)$
- ELBO optimization: standard fine-tuning loss + KL divergence regularizer

$$\mathcal{L}_{\text{BLoB}} = \mathbb{E}_{q(\mathbf{w})}[\text{NLL}(\mathbf{w})] + \beta \cdot \text{KL}[q(\mathbf{w}) \| p(\mathbf{w})]$$

- **Advantage**: Uncertainty is automatically calibrated during training
- **Cost**: ~2x the parameters of standard LoRA (mean + variance) — but still a vanishingly small fraction of the full model

### 2.5 C-LoRA (2025) — Context-Aware Uncertainty

C-LoRA addresses a limitation shared by the methods above — **input-agnostic uncertainty**:

- Existing Bayesian LoRA methods: sample from the posterior over LoRA parameters -> same weight uncertainty for all inputs
- **C-LoRA**: Adjusts the uncertainty of adapter weights in a **sample-dependent** manner based on the input $\mathbf{x}$
- Extracts a context vector at each layer to generate a **conditional distribution** of the LoRA parameters for the given input
- Captures "how uncertain is the model about this specific query" on a per-layer basis

$$p(\mathbf{w}_{\text{LoRA}} \mid \mathbf{x}, \mathcal{D}) \quad \text{vs.} \quad p(\mathbf{w}_{\text{LoRA}} \mid \mathcal{D})$$

The left-hand side is C-LoRA's context-aware posterior; the right-hand side is the context-agnostic posterior of existing methods.

### 2.6 Comparison and Key Takeaways

| Method | Approach | Training | Inference | Input-Aware |
|:-------|:---------|:--------:|:---------:|:-----------:|
| Bayesian LoRA | GP posterior on LoRA | Post-hoc | $O(1)$ forward + posterior | No |
| SWAG-LoRA | SGD trajectory moments | During FT (free) | $T$ samples | No |
| BLoB | Variational (BBB on LoRA) | During FT (~2x) | $T$ samples | No |
| C-LoRA | Context-conditioned adapter | During FT | $O(1)$ forward | **Yes** |

**Key takeaway**: The common principle underlying all of these methods is the same:

- **Freeze the pretrained representation** (it is already a sufficiently good feature extractor)
- **Model uncertainty only in the task-specific adaptation layer** (a low-dimensional subspace)
- **Fewer than 1,000 additional parameters** can yield effective calibration

This is a natural extension of the principle demonstrated by Last-Layer Laplace in Part 2: **"where to apply Bayesian treatment" matters as much as — or more than — "how precisely to apply it."**

---

## 3. Scientific Foundation Models and UQ

### 3.1 Chemistry and Materials — The Missing UQ Problem

As the Neural Network Potentials (NNPs) and molecular property prediction models from Part 6 grow to Foundation Model scale, the absence of UQ becomes an increasingly serious problem:

- **MACE-MP-0** (Batatia et al., 2024): A general-purpose NNP pretrained on 150,000 structures from the Materials Project
  - Impressive accuracy, but no systematic approach to predictive uncertainty
  - Users cannot know "how reliable is this energy prediction?"
  - Risk of silent failure in extrapolation regions (element combinations absent from training data)

- **GNoME** (Merchant et al., *Nature*, 2023): Predicted 2.2 million stable crystal structures
  - Generated candidate structures using graph networks + energy-based filters
  - Used energy-threshold-based filtering **rather than uncertainty-based filtering**
  - Subsequent experimental validation found a significant number of structures to be actually unstable — UQ could have filtered these out in advance

- **Nature Reviews Chemistry (2025)**: A review of foundation models for atomistic simulation that identified the absence of UQ as a key challenge
  - "Current scientific foundation models focus on the accuracy race while lacking systematic evaluation of reliability"
  - A known NNP failure mode: energies and forces diverging to unphysical values under extrapolation — detectable in advance with UQ

### 3.2 Biology — The Gap Between Confidence Scores and Bayesian UQ

In Part 6, we critically analyzed why AlphaFold's pLDDT is not a genuine Bayesian uncertainty measure. In the Foundation Model era, this issue becomes far more widespread:

- **ESM-2** (Lin et al., 2023): A 15-billion-parameter protein language model
  - Softmax outputs from masked token prediction serve as a form of per-position confidence
  - But these are **conditional probabilities, not epistemic uncertainty**
  - High softmax probabilities are possible even for protein families absent from the training data — the same structural problem as hallucination

- **AlphaFold3** (Abramson et al., *Nature*, 2024)
  - Transitioned to diffusion-based structure prediction
  - **Multiple sample generation is now possible**, allowing indirect uncertainty estimation from output diversity
  - But it is impossible to distinguish whether this diversity represents posterior samples or mere generation noise
  - **Aleatoric (structural flexibility) vs. epistemic (lack of model knowledge)** decomposition remains impossible

### 3.3 The Structural Problem With Benchmarks

```
Current Benchmark Paradigm (The Race to the Bottom)
====================================================

                   What We Measure
                   ===============
                   |  Accuracy  |  Speed  |  Memory  |
                   +------------+---------+----------+
     Model A       |   0.952    |   1.2s  |   4 GB   |
     Model B       |   0.961    |   2.1s  |   8 GB   |
     Model C       |   0.958    |   0.8s  |   3 GB   |  <-- "best" by
                   +------------+---------+----------+      accuracy/cost

                   What We DON'T Measure
                   =====================
                   |  Calibration | OOD Detection | Failure Alert |
                   +--------------+---------------+---------------+
     Model A       |     ???      |      ???      |      ???      |
     Model B       |     ???      |      ???      |      ???      |
     Model C       |     ???      |      ???      |      ???      |
                   +--------------+---------------+---------------+

     ==> Models that "know when they don't know"
         receive NO credit in current benchmarks
```

Problems with current scientific Foundation Model benchmarks:

- **Only accuracy is evaluated**: MAE, RMSE, AUC, and other point-prediction metrics are the sole determinants of ranking
- **No UQ evaluation**: Calibration error, negative log-likelihood, and prediction interval coverage are not included in the standard metrics
- **"Race to the bottom"**: Every team competes on the same accuracy metric at the third decimal place — leaderboard climbing with no meaningful improvement in practical utility
- **No evaluation under distribution shift**: Rankings are based on in-distribution performance only, so robustness in real-world settings (novel molecules, unseen proteins) remains unknown

**A proposal — UQ-Aware Benchmarks**:

- **Expand the required metrics**: Mandate reporting **ECE (Expected Calibration Error)**, **NLL**, and **AUCE (Area Under the Calibration Error curve)** alongside accuracy
- **Separate OOD performance**: Report in-distribution and out-of-distribution results separately, and evaluate **selective prediction performance** under OOD conditions (i.e., how much accuracy improves when high-uncertainty predictions are excluded)
- **Prediction interval coverage**: For regression tasks, report the actual coverage of 95% prediction intervals
- **Failure detection**: The fraction of severe errors detected by uncertainty (AUROC for error detection)

If such benchmarks become the standard, models with built-in UQ will gain a systematic advantage, and the entire field will shift from "accurate but unaware of when it fails" to "knows when it can be trusted."

---

## 4. New Theoretical Developments

### 4.1 Closed-form UQ for Deep Residual Networks

Traditionally, UQ in deep networks has always required **sampling** (MC Dropout, HMC, ensembles) or **approximations** (VI, Laplace). Recent work has shown that, for certain architectures, **analytical (closed-form) uncertainty estimation without sampling** is possible:

- In **deep residual networks**, the structure of skip connections can be exploited to decompose the output uncertainty as a sum of per-layer contributions
- Approximating each layer's contribution via local linearization allows the total uncertainty to be computed via **Jacobian propagation in a single forward pass**
- This approach is philosophically similar to SNGP (Part 5), but instead of a GP output layer, it **interprets the residual structure itself probabilistically**

$$\text{Var}[f(\mathbf{x})] \approx \sum_{l=1}^{L} \mathbf{J}_l^T \boldsymbol{\Sigma}_l \mathbf{J}_l$$

where $\mathbf{J}_l$ is the Jacobian of layer $l$ and $\boldsymbol{\Sigma}_l$ is the posterior covariance of that layer's weights. The residual connections are what make this decomposition possible.

### 4.2 Nature Communications (2026) — The Return of Metropolis-Hastings

Here we revisit in greater detail the work briefly mentioned in Part 2.

**The problem with existing SG-MCMC**: Stochastic Gradient Langevin Dynamics (SGLD) and SG-HMC "recycle" mini-batch gradient noise as the stochastic noise in Langevin/Hamiltonian dynamics to approximately sample from the posterior. However, these methods carry an **asymptotic bias** — they do not sample from the exact posterior unless the step size is annealed to zero.

**The solution**: A computationally lightweight **Metropolis-Hastings (MH) acceptance step** integrated into SG-HMC

- The MH step **corrects** the bias introduced by mini-batch gradient noise
- Guarantees exact (or near-exact) posterior sampling even at finite step sizes
- **Cost**: Each MH step requires an additional likelihood computation, but subsampling techniques keep it lightweight

The significance of this work:

- **Reopens the possibility that sampling-based Bayesian inference can be practical at deep learning scale**
- Bias-free posterior samples enable **theoretical guarantees on calibration**
- Future potential for exact posterior sampling over the last layer or adapters of Foundation Models

### 4.3 ACM Transactions on Probabilistic Machine Learning (2025)

The launch of **TOPML**, ACM's dedicated journal for probabilistic ML, in 2025 is a symbolic milestone for the maturity of the field:

- **Significance**: UQ, Bayesian ML, and probabilistic modeling have been elevated from a "supplementary concern" within ML to a **standalone research discipline**
- The existence of a dedicated venue changes career incentives for researchers: UQ work can now be published in a top venue
- The BDL community finally has a "home" — previously it depended on NeurIPS/ICML workshops or partial acceptance at general venues

### 4.4 Weight-space vs. Function-space Inference

This debate touches one of the deepest theoretical questions in BDL:

- **Weight-space inference**: Directly infer $p(\mathbf{w} \mid \mathcal{D})$ — the traditional BDL approach
  - Pros: Conceptually straightforward, can leverage existing tools
  - Cons: Weight symmetries and overparameterization make the posterior unnecessarily complex

- **Function-space inference**: Directly infer $p(f \mid \mathcal{D})$ — the philosophy of GPs
  - Pros: The permutation symmetry problem is automatically resolved (same function = same point in function space), spurious modes vanish
  - Cons: Inference in an infinite-dimensional function space is difficult to implement in practice

- **Recent advances**: Function-space variational inference, function-space particle methods
  - Connection to LoRA: The output of LoRA is effectively a variation in a low-rank function space — a natural meeting point with function-space inference

**An intriguing perspective is that the Bayesian LoRA family is effectively a compromise between weight-space and function-space inference**: weights are treated probabilistically in a low-dimensional subspace, and that subspace is aligned with directions of meaningful variation in function space.

---

## 5. Autonomous Laboratories — The Ultimate Testing Ground for UQ

### 5.1 UQ in the Autonomous Loop

Part 6 covered active learning and lab-in-the-loop workflows, but in a **fully autonomous laboratory**, the role of UQ is even more fundamental. In an environment where AI makes every decision without human intervention, UQ is not "nice to have" — it is a **prerequisite for the system to function at all**.

```
Autonomous Laboratory Loop
===========================

    +------------------+
    |  Foundation Model |
    |  (Prediction +   |
    |   Uncertainty)   |
    +--------+---------+
             |
             | predictions + UQ
             v
    +------------------+
    | Acquisition       |
    | Function          |<---- UQ drives decision:
    | (UCB / EI /       |      explore (high uncertainty)
    |  Thompson)        |      vs. exploit (high value)
    +--------+---------+
             |
             | selected candidates
             v
    +------------------+
    |  Robotic          |
    |  Synthesis        |<---- automated wet lab
    |  Platform         |      (liquid handling,
    +--------+---------+       reactors, etc.)
             |
             | synthesized compounds
             v
    +------------------+
    |  Automated        |
    |  Characterization |<---- spectroscopy, assay,
    |  & Measurement    |      physical property
    +--------+---------+       measurement
             |
             | experimental results
             v
    +------------------+
    |  Model Update     |
    |  (Online / Batch  |<---- posterior update with
    |   Learning)       |      new data, recalibration
    +--------+---------+
             |
             +----------> back to Foundation Model
                          (next cycle)

    Each cycle: hours to days
    Typical campaign: 10-50 cycles
    Total candidates explored: 100-1,000
    out of combinatorial space: 10^6 - 10^12
```

### 5.2 Real-World Examples

**Polymer Design**:
- The design space for functional polymers: monomer types x ratios x polymerization conditions -> combinatorial explosion
- UQ-driven active learning efficiently searches for compositions meeting target properties (Tg, modulus, conductivity) among $10^8$+ candidates
- In Bayesian optimization, **prioritizing regions of high epistemic uncertainty** rapidly expands the model's predictive coverage

**Catalyst Screening**:
- Autonomous electrochemistry labs (e.g., A-Lab): robots automate electrode fabrication, electrochemical measurement, and data analysis
- UQ -> propose next catalyst composition -> synthesize -> measure activity -> update model
- **Multi-fidelity BO**: Combining DFT calculations (low cost, moderate accuracy) with experiments (high cost, high accuracy)
  - **UQ decides** when DFT is sufficient and when an experiment is necessary

**Nanomedicine Optimization**:
- Design variables for drug delivery vehicles (e.g., lipid nanoparticles): lipid type, ratio, PEG density, particle size, drug loading, etc.
- **Number of possible nano-formulations: $\sim 1.7 \times 10^{10}$** (17 billion)
- Exhaustive search is impossible — active learning is the only practical approach
- Cases where Bayesian optimization reached the optimal formulation with only a few hundred experiments:
  - Out of 17 billion candidates, only ~500 experiments achieved organ-specific delivery
  - Without UQ, random screening would have required tens of thousands of experiments

### 5.3 UQ Requirements in Autonomous Laboratories

The autonomous laboratory setting imposes special requirements on UQ:

- **Real-time calibration**: The model's calibration must be maintained after each experimental round
  - The "catastrophic confidence" problem from Part 6: overconfidence after fine-tuning leads to misguided exploration
  - **Online recalibration** is essential

- **Robustness to distribution shift**: Each exploration round shifts the region of chemical space being sampled
  - There is no guarantee that the model from round $n$ is well-calibrated for the candidates in round $n+1$
  - **Conformal prediction + continual calibration** is currently the best available strategy

- **Failure detection**: Real-time detection of cases where the model's predictions are completely wrong
  - Cost implications: An erroneous synthesis instruction wastes robotic time and reagents
  - Safety implications: Failure to predict hazardous reaction conditions risks equipment damage
  - UQ-based **safety thresholds**: If uncertainty exceeds a specified level, **halt synthesis and request human review**

---

## 6. Open Questions and Outlook

### 6.1 The Cold Posterior Effect — Resolved or Unresolved?

Let us revisit the cold posterior effect introduced in Part 2:

- **The phenomenon**: With the tempered posterior $p(\mathbf{w} \mid \mathcal{D})^{1/T}$, $T < 1$ (cold temperature) yields better predictive performance than $T = 1$ (the Bayesian optimum)
- **Izmailov et al. (2021)'s explanation**: When data augmentation is used, the augmented data provides more "redundant information" than actual data -> the likelihood is overestimated -> correction with $T < 1$ is needed
- **However**: Cases where a cold posterior is advantageous even without data augmentation have been reported
- **The fundamental question**: Is the cold posterior effect **an artifact of the specific practice of data augmentation**, or does it reflect **a deeper discrepancy between BDL theory and practice**?
- **The prior misspecification hypothesis**: Isotropic Gaussian priors fail to reflect the true prior distribution of weights -> when the prior is wrong, $T = 1$ is not optimal
- **Still open**: As of 2026, there is no full consensus, and the resolution of this issue remains a pivotal question for the theoretical completeness of BDL

### 6.2 Prior Specification — The Limits of the Isotropic Gaussian

Throughout this series, we have implicitly used an **isotropic Gaussian** $\mathcal{N}(\mathbf{0}, \lambda^{-1}\mathbf{I})$ as the prior $p(\mathbf{w})$. This is, in fact, a very strong assumption:

- **"All weights are independent with equal variance"**: This prior completely ignores layer depth, neuron role, and network architecture
- **The actual distribution of weights**: In a well-trained network, weights exhibit highly structured patterns — different scales across layers, sparse structure, and correlations
- **Impact**: A wrong prior leads to a posterior that fails to reflect the true data structure, resulting in degraded calibration
- **Alternatives**:
  - **Empirical Bayes**: Estimate the prior's hyperparameters from data (MacKay 1992's evidence framework)
  - **Hierarchical priors**: Learn different prior variances for each layer
  - **Structured priors**: Incorporate the low-rank structure, sparsity, and other properties of the weight matrix
  - **Function-space priors**: Directly specify a prior distribution over functions rather than weights (e.g., GP priors)

**Implications for the Foundation Model era**: Pretrained weights $\mathbf{w}_0$ effectively serve as an **informative prior**. Setting $p(\mathbf{w}) = \mathcal{N}(\mathbf{w}_0, \sigma^2 \mathbf{I})$ during fine-tuning encodes pretrained knowledge into the prior. The Bayesian LoRA family leverages precisely this principle.

### 6.3 The Rise of Conformal Prediction — Coexistence or Replacement of Bayesian Methods?

Conformal prediction, covered in Part 4, has been gaining rapid attention:

- **Advantages of conformal methods**: **Coverage guarantees** without distributional assumptions, model-agnostic
- **Advantages of Bayesian methods**: Mechanistic understanding (why is it uncertain?), **aleatoric/epistemic decomposition**, natural use in active learning
- **Coexistence, not replacement**:
  - Bayesian UQ for aleatoric/epistemic decomposition -> used for scientific understanding and exploration strategy
  - Conformal prediction for coverage guarantees on prediction intervals -> a safety net for decision-making
  - **Conformal + Bayesian**: Calibrate the Bayesian model's predictive distribution via a conformal procedure -> "best of both worlds"

- **Caveat**: Conformal prediction's **exchangeability assumption** is violated in active learning, time series, and iterative design settings
  - "Conformal prediction under feedback covariate shift" (PNAS, 2022), discussed in Part 6, partially addresses this
  - A complete solution remains an open problem

### 6.4 Does UQ Get Easier or Harder as Models Grow Larger?

This question sounds simple, but the answer is surprisingly ambiguous:

**The case that it gets easier**:
- Larger models learn **richer representations** -> distance-based UQ in feature space becomes more meaningful
- **Ensemble diversity**: Ensemble members of large models learn more diverse functions -> epistemic uncertainty estimates become more accurate
- **Effectiveness of last-layer/adapter methods**: The better the representation, the more effective Bayesian treatment of just the final layer becomes

**The case that it gets harder**:
- **Exploding computational cost**: The cost of traditional methods (ensembles, MC sampling) scales with model size
- **The overparameterization paradox**: With very many parameters, different weight combinations represent the same function -> the posterior becomes structurally more complex
- **Hallucination**: Larger models tend to be more "confidently wrong" -> the need for UQ increases, but so does the difficulty

**The current answer**: Probably **both** are right. As models grow, traditional full-Bayesian UQ becomes harder, but UQ in low-dimensional subspaces (Bayesian LoRA, etc.) and post-hoc calibration (conformal methods, etc.) actually become more effective. **"UQ methodology must evolve to match the scale of the models"** — that is the key lesson.

---

## 7. Closing the Series — Toward Trustworthy AI

### 7.1 The Journey from 2019 to 2026

As I bring this series to a close, let me trace the field's evolution through the lens of my own journey.

**2019 — The Starting Point with Bayesian GCN**:

The question we had then was modest: *"When a GNN predicts molecular properties, can we know how much to trust those predictions?"* The key contribution was applying Concrete Dropout to a GCN, estimating predictive distributions via MC sampling, and separating aleatoric from epistemic uncertainty. The most unexpected finding of that paper was not that uncertainty acted as a filter to improve prediction accuracy — boosting the top-100 actives in EGFR virtual screening from 29 to 69, a 138% improvement — but rather that **tracing regions of anomalously high aleatoric uncertainty in the Harvard CEP dataset uncovered a data quality issue where PCE = 0**. The insight that UQ is not only a tool for "knowing when the model is wrong" but also a tool for "finding where the data is wrong" foreshadowed the direction the field would take.

**2020–2022 — The Gap Between Benchmarks and Reality**:

After the Bayesian GCN, the systematic benchmarks of Hirschfeld et al. (2020) and Scalia et al. (2020) forced us to confront an uncomfortable truth: **there is no single best method.** In some datasets deep ensembles won, in others MC Dropout prevailed, and in still others simple bootstrapping was the champion. The most important lesson the field learned during this period was that **"evaluating UQ methods is itself a hard problem."**

**2023–2024 — Foundation Models and the Paradigm Shift**:

The arrival of scientific foundation models — ESM-2, MACE-MP-0, AlphaFold3, among others — changed the rules of the game. When general-purpose models with billions of parameters, rather than task-specific models with millions, became the basis for scientific decision-making, both the importance and the difficulty of UQ surged simultaneously. It is no coincidence that Bayesian LoRA, BLoB, and C-LoRA — **Bayesian fine-tuning in low-dimensional subspaces** — emerged during this period. The scale barrier forced a new paradigm into existence.

**2025–2026 — Convergence and Expansion**:

The launch of ACM TOPML, the ICML 2025 Workshop on Reliable and Responsible Foundation Models, the Nature Communications paper on MH-corrected SG-HMC — all of these signal that UQ is moving from the **periphery to the center** of ML research. At the same time, UQ-driven discovery in autonomous laboratories and active learning for nanomedicine optimization demonstrate that UQ is transitioning from **a concept in papers to a practical tool on the laboratory bench**.

### 7.2 Core Lessons of the Series — Principles Spanning All Seven Parts

Here are the core principles that run through the entire series:

- **Calibration is independent of accuracy** (Parts 0, 4)
  - An accurate model is not necessarily calibrated, and calibration is what gives a model practical value
  - The fact that **a single parameter — temperature scaling** — can dramatically improve calibration shows that this problem is not unsolvable

- **"Where to apply Bayesian treatment" matters as much as "how to apply it"** (Parts 2, 7)
  - The trajectory from Last-Layer Laplace to Bayesian LoRA: concentrating on the critical subspace outperforms a crude application to the whole model
  - **Fewer than 1,000 additional parameters** can calibrate models with hundreds of billions of parameters

- **Separating aleatoric and epistemic uncertainty is a guide to action** (Parts 3, 6)
  - High aleatoric uncertainty -> check data quality, or obtain more precise measurements
  - High epistemic uncertainty -> collect more data (or data from a different region)
  - Without this distinction, knowing that uncertainty is high provides no guidance on **what to do next**

- **The ultimate purpose of UQ is to determine the next action** (Parts 6, 7)
  - Active learning: collect data where uncertainty is high
  - Autonomous labs: uncertainty determines the next synthesis
  - Virtual screening: uncertainty-based filtering improves hit rates
  - **Uncertainty itself is not the goal — value is created when uncertainty is linked to action**

- **No perfect method exists; context determines the choice** (Parts 2, 5)
  - Already-trained model -> Laplace, MC Dropout
  - Training from scratch with computational budget -> Deep Ensemble
  - Foundation Model fine-tuning -> Bayesian LoRA family
  - Coverage guarantees needed -> Conformal Prediction
  - Real-time, large-scale screening -> Single-Pass methods (EDL, SNGP)

### 7.3 Key Challenges for the Next Five Years

From 2026 to 2031, three key challenges that the UQ field must address:

**Scalable UQ**:
- As foundation models continue to grow, UQ methods must scale alongside them
- The Bayesian LoRA family represents the current state of the art, but **validation on trillion-parameter models** remains incomplete
- Hardware-software co-design: UQ-optimized accelerators may eventually be necessary

**Calibrated UQ**:
- Scalable but miscalibrated UQ is useless — or worse, dangerous
- **Maintaining calibration under distribution shift** is the hardest challenge
- Conformal + Bayesian combinations are promising, but the exchangeability violation problem persists
- **Continual calibration**: A framework for re-evaluating calibration each time the model is updated with new data

**Interpretable UQ**:
- Going beyond "this prediction has high uncertainty" to explaining **"why it is uncertain"**
- Aleatoric/epistemic decomposition is a first step, but the goal is to identify **which features contribute to uncertainty**
- Example: "The solubility prediction for this molecule is uncertain because the training data contains only 3 analogues of this scaffold" — this level of explanation is the ultimate target
- **Explainable UQ**: Combining UQ with attention mechanisms, feature attribution, and related techniques

### 7.4 Final Reflections

Seven years ago, when I was writing the Bayesian GCN paper, my greatest concern was convincing reviewers that "Bayesian treatment is necessary." At the time, Bayesian deep learning was often regarded in the ML community as "interesting but impractical." The common perception was that the computational cost was too high, implementation too complex, and accuracy gains over point estimates too marginal.

In 2026, that perception has been thoroughly overturned. Papamarkou et al. declaring at ICML 2024 that "Bayesian Deep Learning is **Needed**" symbolizes the shift from "nice-to-have" to "must-have." In an era where foundation models propose drug candidates and autonomous laboratories carry out syntheses based on those proposals, making decisions without knowing "how confident the model is" is no longer acceptable.

But in all honesty, we are still in the middle of the journey. Full Bayesian inference for models with hundreds of billions of parameters remains impossible, and we lack a complete understanding of whether compromise methods like Bayesian LoRA are "good enough" in theory. The debate over the origins of the cold posterior effect continues, and the fundamental problem of prior specification is not in a vastly different state from MacKay's era three decades ago.

And yet, the direction is clear.

**Trustworthy AI** is no longer an academic slogan. It is a core requirement that directly determines the productivity of scientific research, the success rate of drug development, and the efficiency of materials design. And the foundation of trust lies not in knowing "is this answer correct?" but in knowing **"how much can we trust this answer?"**

I hope this series has helped organize the current state of answers to that question, and I look forward to the day when readers bring UQ into their own research and practice not as an "optional extra" but as a "default feature."

Quantifying uncertainty — that is the language of science.

---

## References

### Foundation Model Era BDL
1. Papamarkou, T. et al. (2024). "Position: Bayesian Deep Learning is Needed in the Age of Large-Scale AI." *ICML 2024*.
2. Yang, A. et al. (2024). "Bayesian Low-Rank Adaptation for Large Language Models." *arXiv:2308.13111*.
3. Sharma, A. et al. (2024). "SWAG-LoRA: Low-Rank Adaptation via Stochastic Weight Averaging." *arXiv:2405.03425*.
4. Wang, Z. et al. (2024). "BLoB: Bayesian Low-Rank Adaptation by Backpropagation for Large Language Models." *arXiv:2406.11675*.
5. C-LoRA (2025). "Context-Aware Low-Rank Adaptation for Calibrated Uncertainty in Foundation Models."

### Scientific Foundation Models
6. Batatia, I. et al. (2024). "A Foundation Model for Atomistic Simulation: MACE-MP-0." *arXiv:2401.00096*.
7. Merchant, A. et al. (2023). "Scaling Deep Learning for Materials Discovery." *Nature*, 624, 80-85.
8. Lin, Z. et al. (2023). "Evolutionary-Scale Prediction of Atomic-Level Protein Structure with a Language Model." *Science*, 379, 1123-1130.
9. Abramson, J. et al. (2024). "Accurate Structure Prediction of Biomolecular Interactions with AlphaFold 3." *Nature*, 630, 493-500.
10. Nature Reviews Chemistry (2025). "Foundation Models for Atomistic Simulation."

### Theoretical Developments
11. Nature Communications (2026). "Reliable Uncertainty via Metropolis-Hastings for Deep Networks."
12. Welling, M. & Teh, Y. W. (2011). "Bayesian Learning via Stochastic Gradient Langevin Dynamics." *ICML*.
13. ACM Transactions on Probabilistic Machine Learning (2025). *Inaugural Issue*.
14. ICML 2025 Workshop on Reliable and Responsible Foundation Models.

### Autonomous Labs and Active Learning
15. Angelopoulos, A. N. et al. (2022). "Conformal Prediction under Feedback Covariate Shift for Biomolecular Design." *PNAS*.
16. Angelopoulos, A. N. & Bates, S. (2023). "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification." *Foundations and Trends in ML*.

### Series Origin
17. Ryu, S., Kwon, Y. & Kim, W. Y. (2019). "A Bayesian Graph Convolutional Network for Reliable Prediction of Molecular Properties with Uncertainty Quantification." *Chemical Science*, 10, 8438-8446.
18. Hirschfeld, L. et al. (2020). "Uncertainty Quantification Using Neural Networks for Molecular Property Prediction." *JCIM*.
19. Scalia, G. et al. (2020). "Evaluating Scalable Uncertainty Estimation Methods for Deep Learning-Based Molecular Property Prediction." *JCIM*.

### Earlier Parts of This Series
20. Part 0: Beyond Prediction — Why Uncertainty Matters
21. Part 1: The Language of Bayesian Inference
22. Part 2: The Art of Approximation — From Variational Methods to Ensembles
23. Part 3: The Anatomy of Uncertainty
24. Part 4: Calibration and Conformal Prediction
25. Part 5: Single-Pass UQ — Evidential Deep Learning and Distance-Aware Methods
26. Part 6: UQ in Science — Molecules, Proteins, and Materials
