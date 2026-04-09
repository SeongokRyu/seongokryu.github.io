---
title: "Bayesian DL & UQ Part 2: The Art of Approximation — From Variational to Ensemble"
date: 2026-04-09 14:10:00 +0900
categories: [AI & ML, ML Theory]
tags: [bayesian-deep-learning, variational-inference, mc-dropout, deep-ensemble, swag, laplace-approximation, hmc]
math: true
---

**Bayesian Deep Learning and Uncertainty Quantification — A Journey Toward Trustworthy AI**

*This is Part 2 of an 8-part series.*

- **Part 0:** [Beyond Predictions — Why Uncertainty Matters](/posts/bayesian-dl-uq-part0-why-uncertainty/)
- **Part 1:** [The Language of Bayesian Inference](/posts/bayesian-dl-uq-part1-bayesian-language/)
- **Part 2:** The Art of Approximation — From Variational to Ensemble (this post)
- **Part 3:** [The Anatomy of Uncertainty — Aleatoric vs. Epistemic](/posts/bayesian-dl-uq-part3-anatomy-of-uncertainty/)
- **Part 4:** [Calibration and Conformal Prediction](/posts/bayesian-dl-uq-part4-calibration-conformal/)
- **Part 5:** [Single-Pass UQ — Evidential Deep Learning and Distance-Aware Methods](/posts/bayesian-dl-uq-part5-single-pass-uq/)
- **Part 6:** [UQ in Science — Molecules, Proteins, and Materials](/posts/bayesian-dl-uq-part6-uq-in-science/)
- **Part 7:** [The Future of UQ — Uncertainty in the Age of Foundation Models](/posts/bayesian-dl-uq-part7-future-foundation-models/)

---

## Hook: The Perfect Answer Is Uncomputable

In Part 1, we arrived at the core formula of Bayesian neural networks:

$$p(\mathbf{w} \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \mathbf{w})\, p(\mathbf{w})}{p(\mathcal{D})}$$

If we knew this posterior $p(\mathbf{w} \mid \mathcal{D})$ exactly, everything would follow. The prediction for a new input $\mathbf{x}^{\ast}$ is given by the expectation over the posterior:

$$p(y^{\ast} \mid \mathbf{x}^{\ast}, \mathcal{D}) = \int p(y^{\ast} \mid \mathbf{x}^{\ast}, \mathbf{w})\, p(\mathbf{w} \mid \mathcal{D})\, d\mathbf{w}$$

The variance of this predictive distribution naturally quantifies uncertainty. In data-rich regions, the posterior concentrates tightly and uncertainty is low; in data-scarce regions, it spreads out and uncertainty is high.

There is one problem: **this integral cannot be computed.**

This is not merely a matter of "the computation takes too long." In modern neural networks, exact posterior inference is fundamentally intractable — the kind of problem that cannot be brute-forced even given the age of the universe. In this post, we analyze **why** from four distinct perspectives, and then survey the approximation methods developed over the past thirty years to circumvent this impossibility.

---

## 1. Four Reasons Why the Exact Posterior Is Intractable

### 1.1 The Normalization Problem — The Intractable Marginal Likelihood

The denominator $p(\mathcal{D})$ of Bayes' theorem is called the **marginal likelihood** (or **model evidence**):

$$p(\mathcal{D}) = \int p(\mathcal{D} \mid \mathbf{w})\, p(\mathbf{w})\, d\mathbf{w}$$

This integral marginalizes the product of likelihood and prior over all possible weight configurations $\mathbf{w}$. It is essential for normalizing the posterior into a proper probability distribution, yet closed-form evaluation is only possible in very restricted cases.

**When a closed-form solution exists**: When the prior and likelihood are **conjugate** — meaning the prior and posterior belong to the same distributional family — the marginal likelihood can be computed analytically. Classic examples:

- Gaussian prior + Gaussian likelihood → Gaussian posterior (linear regression)
- Beta prior + Bernoulli likelihood → Beta posterior (coin flip)
- Dirichlet prior + Multinomial likelihood → Dirichlet posterior

**Why conjugacy breaks down for neural networks**: The likelihood $p(\mathcal{D} \mid \mathbf{w})$ of a neural network is **highly nonlinear** in the weights. Even a single-hidden-layer network:

$$f(\mathbf{x}; \mathbf{w}) = \mathbf{W}_2 \cdot \sigma(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2$$

where $\sigma$ is a nonlinear activation function, yields a likelihood that is not conjugate to any known distributional family. Even if we assign a Gaussian prior $p(\mathbf{w}) = \mathcal{N}(\mathbf{0}, \lambda^{-1}\mathbf{I})$, the posterior $p(\mathbf{w} \mid \mathcal{D})$ is not Gaussian — the nonlinear transformation completely distorts the shape of the distribution.

As a result, the marginal likelihood integral **has no analytical solution**, and computing it numerically requires an exhaustive search over the entire weight space.

### 1.2 The Curse of Dimensionality — Integration in Millions of Dimensions

The dimensionality $d$ of the weight space equals the number of parameters in the network. To appreciate the scale:

| Model | Parameters ($d$) | Era |
|:------|:----------------:|:---:|
| LeNet-5 (1998) | 60K | Classical |
| ResNet-50 (2015) | 25.6M | Modern CV |
| BERT-Base (2018) | 110M | NLP |
| GPT-3 (2020) | 175B | Foundation |
| Llama 3.1 (2024) | 405B | Frontier |

The simplest approach to numerically computing the marginal likelihood is **grid integration**: divide each dimension into $K$ grid points, evaluate the integrand at every combination, and sum. The cost scales as $O(K^d)$.

Even with a modest $K = 10$ (ten grid points per dimension):

- LeNet-5: $10^{60,000}$ function evaluations — exceeding the number of atoms in the observable universe ($\sim 10^{80}$) by a factor of $10^{59,920}$
- ResNet-50: $10^{25,600,000}$ evaluations — a number beyond any meaningful analogy

This is the essence of the **curse of dimensionality**. Grid-based integration in high-dimensional spaces suffers **exponential** growth in computational cost with respect to dimensionality. Switching to Monte Carlo integration does not fundamentally resolve this — in naive Monte Carlo sampling, halving the variance requires quadrupling the number of samples, and achieving reasonable accuracy in a complex posterior over millions of dimensions demands an astronomical number of samples.

**Key insight**: The crux of the problem is not "the integral is hard to compute" but rather "we cannot efficiently locate the region where probability mass concentrates (the typical set) in high-dimensional space." The vast majority of weight configurations carry negligibly low posterior probability, while the high-probability region occupies a vanishingly small fraction of the total space. Without finding this region, any estimate of the integral is meaningless.

### 1.3 Multimodality and Complex Geometry — A Rugged Posterior Landscape

Even if the curse of dimensionality could be overcome, the neural network posterior exhibits **complex geometric structure** that makes approximation all the more challenging.

**Multimodality from nonlinear transformations**: The loss landscape of a neural network typically harbors numerous local minima. This means multiple distinct weight configurations can explain the data comparably well, and the posterior $p(\mathbf{w} \mid \mathcal{D})$ becomes a **multimodal distribution** with probability mass concentrated around each such mode.

Approximating this multimodal posterior with a single Gaussian (e.g., Laplace approximation) captures only one mode and ignores the rest. If different modes yield different predictions — for instance, mode A predicts "this molecule is active" while mode B predicts "inactive" — a single-mode approximation systematically underestimates the true uncertainty.

**Equivalent modes from permutation symmetry**: Neural networks possess a structural property that further amplifies posterior multimodality. Arbitrarily permuting the $h$ neurons in a hidden layer — rearranging both their incoming and outgoing weights together — leaves the network's input-output mapping completely unchanged. Since the number of permutations of $h$ neurons is $h!$, every single functional solution corresponds to $h!$ equivalent modes in weight space.

Concretely, for a single-hidden-layer network with 512 hidden neurons:

$$512! \approx 3.5 \times 10^{1166}$$

equivalent modes exist. They all represent the same function, yet occupy different locations in weight space. This geometric symmetry renders the posterior structure extraordinarily complex and underscores the limitations of simple unimodal approximations.

**Asymmetric connectivity between modes**: Recent work (Izmailov et al., 2021) reported intriguing findings from large-scale BNN experiments using HMC. Within a single mode, the posterior is relatively well-connected, meaning a sufficiently long MCMC chain can reasonably explore the intra-mode posterior. However, transitions between distinct (functionally non-equivalent) modes occur very rarely. This explains why multi-chain or multi-model approaches (e.g., deep ensembles) represent the posterior more faithfully than a single chain.

### 1.4 Storage and Computation — The Cost of Merely Representing the Posterior

Even if the exact posterior were miraculously computed, **storing and using it** would itself be a formidable challenge.

The most natural posterior approximation is a **Gaussian distribution** $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$. In this case:

- **Mean vector** $\boldsymbol{\mu}$: $d$ parameters → $O(d)$ storage
- **Covariance matrix** $\boldsymbol{\Sigma}$: $d \times d$ symmetric matrix → $O(d^2/2)$ storage

To store the full covariance for ResNet-50's 25.6M parameters:

$$\frac{(25.6 \times 10^6)^2}{2} \times 4 \text{ bytes} \approx 1.3 \times 10^{15} \text{ bytes} = 1.3 \text{ PB}$$

1.3 petabytes — tens of thousands of times beyond the memory of a single GPU. Computing the matrix inverse or Cholesky decomposition costs $O(d^3)$, which is equally impractical.

Because of these storage costs, most practical approximation methods simplify the posterior representation itself:

| Approximation | Storage Cost | What Is Sacrificed |
|:---|:---:|:---|
| Full Gaussian | $O(d^2)$ | — (impractical in practice) |
| KFAC (Kronecker-Factored) | $O(d)$ | Cross-layer correlations |
| Diagonal Gaussian | $O(d)$ | All inter-weight correlations |
| Low-rank + Diagonal | $O(dk)$ | Correlation structure beyond rank $k$ |
| Point estimate (MAP) | $O(d)$ | The distribution itself |

This table illustrates the core trade-off in approximation design: **richer posterior representations demand more memory and computation, and within any practical budget, something must be sacrificed.**

### Summary: How the Four Barriers Reinforce Each Other

These four barriers are not independent but mutually reinforcing:

```
Non-conjugacy ────→ No closed-form solution
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

Without conjugacy, we must resort to numerical methods — but high dimensionality renders numerical integration infeasible. Even if we could efficiently explore the posterior, multimodality makes simple approximations inaccurate, and richer approximations incur prohibitive storage costs.

Given this situation, the strategic question for researchers is clear: **which of the four barriers to compromise on, and how.** The approximation methods developed over the past three decades each choose a different point in this space of trade-offs.

---

## 2. Historical Development of Approximation Methods — A Thirty-Year Journey

### 2.1 Timeline: The Evolution of Practical Bayesian Inference

The timeline below marks the major milestones in approximate Bayesian inference for neural networks:

```
1992  MacKay         Laplace Approximation for NNs
 │
1995  Hinton & Neal  HMC for BNNs (Neal's thesis)
 │
1996  Barber & Bishop Ensemble Learning (variational approach)
 │
2011  Graves         Practical VI with stochastic gradients
 │
2015  Blundell et al. Bayes by Backprop (reparameterization trick)
 │
2016  Gal & Ghahramani MC Dropout as approximate inference
 │
2016  Liu & Wang      Stein Variational Gradient Descent
 │
2017  Lakshminarayanan Deep Ensembles
 │
2018  Izmailov et al.  SWA (weight averaging → better point estimate)
 │
2019  Maddox et al.   SWAG (SWA + covariance → posterior sampling)
 │
2020  Wilson & Izmailov Ensembles as Bayesian marginalization
 │
2021  Daxberger et al. Laplace Redux (modular post-hoc framework)
 │
2021  Izmailov et al.  Large-scale HMC study of BNN posteriors
 │
2024  Bayesian LoRA, BLoB, C-LoRA  ── BDL in the Foundation Model era
 │
2026  Nature Comm.    MH-corrected SG-HMC for deep networks
```

A clear arc emerges from this timeline. The early era prioritized **accuracy** (HMC, full Laplace). As deep learning models grew in scale, the field shifted toward **scalability** (MC Dropout, SWAG, Last-Layer Laplace). Most recently, the focus has moved to **lightweight Bayesian methods that operate at foundation model scale** (Bayesian LoRA and its variants).

Let us now examine how each method addresses the four barriers from Section 1.

---

### 2.2 Laplace Approximation — Surveying the Surroundings from the Peak

**Core idea**: Approximate the posterior with a second-order Taylor expansion around the MAP estimate. Find the peak (MAP) of the posterior, measure the local curvature via the Hessian, and fit a Gaussian.

$$q(\mathbf{w}) = \mathcal{N}\big(\mathbf{w}_{\text{MAP}},\, \mathbf{H}^{-1}\big)$$

where $\mathbf{H}$ is the Hessian of the negative log-posterior density:

$$\mathbf{H} = -\nabla^2 \log p(\mathbf{w} \mid \mathcal{D}) \big|_{\mathbf{w}_{\text{MAP}}}$$

**Historical context**: David MacKay (1992) was the first to systematically apply this method to neural networks. His contribution was pioneering in two respects:

1. **Posterior approximation**: Quantifying weight uncertainty via the Hessian
2. **Model comparison via evidence**: Estimating the marginal likelihood $p(\mathcal{D})$ through the Laplace approximation to compare network architectures (e.g., number of hidden units) — a principled approach to model selection without a validation set

MacKay's work planted the initial seed of "Bayesian methods can be useful" in the neural network community, though practical applications at the time were limited to small networks.

**Trade-offs against the four barriers**:

- *Non-conjugacy*: Bypassed by forcing a Gaussian approximation → captures only a single mode
- *Dimensionality*: Full Hessian requires $O(d^2)$ storage and $O(d^3)$ computation → infeasible for large networks
- *Multimodality*: Captures only the mode around the MAP → completely ignores uncertainty from other modes
- *Storage*: Requires storing the full covariance, limiting scalability

**A modern revival — Laplace Redux (Daxberger et al., NeurIPS 2021)**: Thirty years later, the Laplace approximation experienced a remarkable resurgence. Daxberger et al. proposed a modular framework that strategically restricts the scope of the Hessian computation:

| Variant | Scope | Cost | Quality |
|:--------|:------|:----:|:-------:|
| Full Laplace | All weights | $O(d^2)$ | Theoretically best (impractical) |
| KFAC Laplace | All weights, Kronecker factored | $O(d)$ | Good |
| Diagonal Laplace | All weights, diagonal only | $O(d)$ | Moderate |
| Last-Layer Laplace | Final layer only | $O(d_L^2)$ | **Surprisingly good** |
| Subnetwork Laplace | Selected subnetwork | $O(d_{\text{sub}}^2)$ | Variable |

**Last-Layer Laplace** deserves special attention. It treats most of the network as a fixed feature extractor and applies Bayesian treatment only to the final linear layer. Since the final layer's parameter count is a tiny fraction of the total, full covariance storage becomes realistic. The key observation: **Bayesian inference on top of a well-learned representation can outperform crude inference over the entire weight space.** This insight provides the theoretical underpinning for subsequent Bayesian fine-tuning methods (SWAG-LoRA, BLoB, etc.).

---

### 2.3 Hamiltonian Monte Carlo — The Light and Shadow of the Gold Standard

**Core idea**: Draw samples directly from the posterior, using Hamiltonian dynamics from physics to navigate high-dimensional space efficiently.

Radford Neal's 1995 doctoral thesis was a landmark that systematically established HMC for BNNs. The core mechanism:

1. Treat the weight vector $\mathbf{w}$ as "position" and an auxiliary variable $\mathbf{p}$ as "momentum"
2. Define the Hamiltonian $H(\mathbf{w}, \mathbf{p}) = -\log p(\mathbf{w} \mid \mathcal{D}) + \frac{1}{2}\mathbf{p}^T\mathbf{p}$
3. Simulate Hamiltonian dynamics with a leapfrog integrator to propose a new $(\mathbf{w}', \mathbf{p}')$
4. Apply a Metropolis-Hastings acceptance step to correct for energy conservation errors

**Why HMC outperforms random walk**: Random walk Metropolis moves in small steps and thus requires $O(d^2)$ steps to explore a $d$-dimensional space. HMC follows gradient-informed Hamiltonian trajectories, "leaping" to distant regions while maintaining a high acceptance rate. Theoretically, HMC's mixing efficiency scales as $O(d^{5/4})$, markedly better than the $O(d^2)$ of random walk.

**HMC today — lessons from large-scale experiments**: Izmailov et al. (ICML 2021) conducted the first systematic study of HMC on modern architectures (ResNet, CNN) at scale. Some chains ran for over 60 million epochs on CIFAR-10. Key findings:

- **Single long chain**: Within a single mode, the posterior can be explored reasonably well. A single long chain outperforms several short chains in posterior representation
- **The truth about the cold posterior effect**: It had been reported that a "tempered posterior" $p(\mathbf{w} \mid \mathcal{D})^{1/T}$ with $T < 1$ outperforms the standard $T = 1$ posterior. Izmailov et al. discovered this to be **an artifact of data augmentation** — without data augmentation, the standard $T = 1$ posterior performs well
- **Deep ensemble ≈ multi-modal HMC**: Deep ensembles are closer to the HMC posterior than VI is — because each ensemble member occupies a different posterior mode
- **Limitations**: BNNs also fail to generalize under distribution shift — Bayesian is not a silver bullet

**Trade-offs against the four barriers**:

- *Non-conjugacy*: No workaround needed. Samples directly without assumptions on the posterior shape
- *Dimensionality*: Gradient-based exploration is efficient, but convergence guarantees remain elusive
- *Multimodality*: Mode transitions are theoretically possible but practically very slow
- *Storage*: Stores samples $\lbrace\mathbf{w}_1, ..., \mathbf{w}_T\rbrace$ → cost of $T$ models

**Practical standing**: HMC primarily serves as a "ground truth" reference for evaluating the quality of other approximation methods. Its computational cost is too high for production deployment.

---

### 2.4 Variational Inference — Turning Inference into Optimization

**Core idea**: Instead of computing the exact posterior $p(\mathbf{w} \mid \mathcal{D})$ directly, find the closest distribution within a tractable family $q_{\boldsymbol{\theta}}(\mathbf{w})$. "Closeness" is measured by the KL divergence.

$$\boldsymbol{\theta}^{\ast} = \arg\min_{\boldsymbol{\theta}}\, \text{KL}\big(q_{\boldsymbol{\theta}}(\mathbf{w}) \,\|\, p(\mathbf{w} \mid \mathcal{D})\big)$$

This minimization cannot be performed directly (since the posterior is unknown), but it is equivalent to maximizing the **ELBO (Evidence Lower Bound)**:

$$\text{ELBO}(\boldsymbol{\theta}) = \mathbb{E}_{q_{\boldsymbol{\theta}}(\mathbf{w})}\big[\log p(\mathcal{D} \mid \mathbf{w})\big] - \text{KL}\big(q_{\boldsymbol{\theta}}(\mathbf{w}) \,\|\, p(\mathbf{w})\big)$$

The first term is the **data fit** (the expected log-likelihood of the data under the variational posterior), and the second is the **complexity penalty** (how much the variational posterior deviates from the prior). Since the ELBO is a lower bound on $\log p(\mathcal{D})$, maximizing it minimizes the KL divergence.

**Bayes by Backprop (Blundell et al., ICML 2015)**: A practical algorithm that scaled VI to neural network size. Key technical contributions:

1. **Mean-field Gaussian variational family**: Each weight $w\_i$ is modeled as an independent Gaussian:

$$q_{\boldsymbol{\theta}}(\mathbf{w}) = \prod_i \mathcal{N}(w_i \mid \mu_i, \sigma_i^2)$$

The variational parameters to learn are the $(\mu\_i, \sigma\_i)$ pairs, totaling $2d$

2. **Reparameterization trick**: Rewrite $w\_i = \mu\_i + \sigma\_i \cdot \epsilon\_i$ with $\epsilon\_i \sim \mathcal{N}(0, 1)$, so that gradients flow through $\mu\_i$ and $\sigma\_i$ → trainable via standard backpropagation
3. **Mini-batch training**: Approximate the ELBO expectation with Monte Carlo estimates over mini-batches → same workflow as SGD

**The fundamental limitation of VI — directionality of KL divergence**: The ELBO minimizes $\text{KL}(q \| p)$, not $\text{KL}(p \| q)$. This direction of KL divergence is **mode-seeking**: $q$ tends to concentrate on high-density regions of $p$ rather than covering all of $p$. For a multimodal posterior, this results in capturing only one mode while ignoring the rest, systematically underestimating uncertainty.

**The cost of the mean-field assumption**: Assuming all weights are independent discards all inter-weight correlations — for example, the constraint that "if this weight is large, that one must be small." When the true posterior contains strong correlations, the uncertainty estimates from mean-field approximations are inaccurate.

**Trade-offs against the four barriers**:

- *Non-conjugacy*: Fixes the shape of the approximate distribution $q$ in advance (usually Gaussian)
- *Dimensionality*: Mean-field requires only $O(2d)$ parameters → scalable
- *Multimodality*: Mode-seeking $\text{KL}(q\|p)$ captures only a single mode
- *Storage*: $O(2d)$ — twice the original model

---

### 2.5 MC Dropout — Turning a Pre-trained Model into a BNN

**Core idea**: Keep dropout enabled at test time and perform $T$ forward passes. This is mathematically equivalent to approximate variational inference.

The insight of Gal and Ghahramani (ICML 2016) was remarkably simple yet profoundly influential:

**Mathematical argument**: In each forward pass with dropout, every row of the weight matrix $\mathbf{W}_l$ is zeroed out with probability $p$. Viewed as a variational distribution:

$$q(\mathbf{W}_l) = \mathbf{M}_l \cdot \text{diag}(\mathbf{z}_l), \quad z_{l,i} \sim \text{Bernoulli}(1-p)$$

where $\mathbf{M}_l$ is the learned weight matrix and $\mathbf{z}_l$ is the dropout mask. Gal and Ghahramani proved that training a network with dropout is equivalent to maximizing an approximation to the ELBO with $q$ as the variational posterior.

**Predictive uncertainty estimation**: At test time, perform $T$ forward passes:

$$\hat{\mu}(\mathbf{x}^{\ast}) = \frac{1}{T}\sum_{t=1}^{T} f(\mathbf{x}^{\ast}; \hat{\mathbf{w}}_t), \quad \hat{\sigma}^2(\mathbf{x}^{\ast}) = \frac{1}{T}\sum_{t=1}^{T}\big(f(\mathbf{x}^{\ast}; \hat{\mathbf{w}}_t) - \hat{\mu}(\mathbf{x}^{\ast})\big)^2$$

where $\hat{\mathbf{w}}_t$ is the weight vector with the $t$-th dropout mask applied. $\hat{\sigma}^2$ estimates the epistemic uncertainty.

**Concrete Dropout (Gal et al., 2017)**: A major weakness of MC Dropout is that the dropout rate $p$ — a variational hyperparameter — is set manually. Concrete Dropout makes $p$ learnable through gradient descent via a differentiable relaxation (the Concrete distribution). This is also the approach adopted in my Bayesian GCN paper (Ryu et al., 2019) — the optimal dropout rate is learned automatically per layer, yielding low dropout rates (high confidence) for data-rich tasks and high dropout rates (high uncertainty) for data-scarce tasks.

**Practical limitations of MC Dropout**:

1. **Restricted variational family**: The distributional family defined by Bernoulli dropout masks is extremely limited. Each weight is either present or absent — a binary choice — making it difficult to capture the continuous uncertainty structure of the posterior.
2. **Underestimation for OOD inputs**: When Concrete Dropout converges to near-zero dropout rates, the model reports almost no epistemic uncertainty even for inputs far from the training distribution.
3. **Theoretical debate**: The rigor of the dropout-as-VI interpretation has been challenged. Osband (2016) and others argued that MC Dropout systematically misses important aspects of the posterior.

Despite these limitations, MC Dropout remains one of the most widely used BDL methods due to its **simplicity of implementation**. The ability to obtain uncertainty estimates from a model already trained with dropout simply by running inference in `model.train()` mode is a tremendously practical advantage.

**Trade-offs against the four barriers**:

- *Non-conjugacy*: Forces a Bernoulli variational family
- *Dimensionality*: No additional parameters ($O(d)$ maintained)
- *Multimodality*: Stochastic combinations of dropout masks capture some diversity, but coverage is limited
- *Storage*: $O(d)$ — identical to the original model

---

### 2.6 Deep Ensembles — The Simplest and Most Powerful Method

**Core idea**: Train $M$ independently initialized networks on the same data and average their outputs at prediction time. The disagreement among predictions is the uncertainty.

Deep ensembles, proposed by Lakshminarayanan et al. (NeurIPS 2017), are technically the simplest method:

$$\hat{\mu}(\mathbf{x}^{\ast}) = \frac{1}{M}\sum_{m=1}^{M} \mu_m(\mathbf{x}^{\ast}), \quad \hat{\sigma}^2_{\text{epi}}(\mathbf{x}^{\ast}) = \frac{1}{M}\sum_{m=1}^{M}\big(\mu_m(\mathbf{x}^{\ast}) - \hat{\mu}(\mathbf{x}^{\ast})\big)^2$$

If each member is a heteroscedastic model (outputting both $\mu_m$ and $\sigma^2_m$), aleatoric and epistemic uncertainty can be separated:

$$\hat{\sigma}^2_{\text{ale}}(\mathbf{x}^{\ast}) = \frac{1}{M}\sum_{m=1}^{M}\sigma^2_m(\mathbf{x}^{\ast}), \quad \hat{\sigma}^2_{\text{total}} = \hat{\sigma}^2_{\text{ale}} + \hat{\sigma}^2_{\text{epi}}$$

**Is it "Bayesian"?** — One of the most interesting debates in the field.

Deep ensembles were originally proposed as a non-Bayesian method. There was no Bayesian interpretation, and the sole justification was that "it works well." But a series of subsequent studies shifted this perspective:

**Wilson & Izmailov (NeurIPS 2020)**: In "Bayesian Deep Learning and a Probabilistic Perspective of Generalization," they present a key argument:

> All BDL methods are approximations anyway. What matters is not "was this method derived from Bayesian axioms?" but rather "how well does it approximate the true Bayesian predictive distribution?"

Each ensemble member converges to a different mode of the loss landscape (since the random initializations differ). Therefore, the ensemble's predictive distribution **captures multiple modes of the multi-modal posterior**. By contrast, mean-field VI is by definition a unimodal Gaussian and thus explores only within a single mode. The HMC experiments of Izmailov et al. (2021) provided empirical evidence that deep ensembles are closer to the HMC posterior than VI is.

**Pearce et al. (2023)**: Established a **rigorous mathematical connection** between deep ensembles and variational Bayesian methods. They showed that ensemble training can, under certain conditions, be interpreted as optimization of a variational objective.

**The practical reality**: Across nearly all benchmarks for calibration, OOD detection, and predictive performance, deep ensembles are the strongest or among the strongest methods. There is an irony here — the method with the weakest theoretical justification performs the best.

**The cost issue and mitigation**: Training $M$ models independently requires $M\times$ the cost. In practice, $M = 5$ is standard. Several cost-reducing variants have been developed:

- **BatchEnsemble**: Decomposes weights into shared + rank-1 perturbation to save memory
- **Hyperensemble**: A single hypernetwork generates the weights for all ensemble members
- **Snapshot Ensembles**: Harvests multiple models from a single training run via cyclical learning rates

**Trade-offs against the four barriers**:

- *Non-conjugacy*: Sidestepped entirely. No distributional assumptions made
- *Dimensionality*: Each member trains via standard optimization
- *Multimodality*: **Core strength** — random initialization naturally captures different modes
- *Storage*: $O(Md)$ — $M$-fold cost

---

### 2.7 SWA and SWAG — Two Kinds of Information Hidden in the SGD Trajectory

When SGD "wanders" near convergence across the loss landscape, the trajectory contains two useful pieces of information: **where it wanders around** (the average position) and **how widely it wanders** (the variance structure). SWA uses only the former; SWAG uses both. The two methods share a core idea but **diverge fundamentally at the final estimation step**. Let us examine each.

#### 2.7.1 SWA — Stochastic Weight Averaging: A Better Point Estimate

**Core idea**: Simply average multiple checkpoints along the SGD trajectory to obtain **a single weight vector**. This has the effect of finding a solution located in a flatter region near the mode of the posterior.

SWA was proposed by Izmailov et al. (UAI 2018, "Averaging Weights Leads to Wider Optima and Better Generalization"):

**Step 1 — Standard training**: Train the network normally (or use a pre-trained model).

**Step 2 — Cyclical/High LR SGD**: Continue running SGD with a sufficiently high (or cyclical) learning rate, periodically collecting weight snapshots $\mathbf{w}_1, \mathbf{w}_2, ..., \mathbf{w}_K$.

**Step 3 — Weight Averaging**: Compute the simple average of the collected snapshots:

$$\mathbf{w}_{\text{SWA}} = \frac{1}{K}\sum_{k=1}^{K}\mathbf{w}_k$$

**Step 4 — Recompute Batch Normalization**: Since activation statistics change for the averaged weights, BN statistics are recomputed once over the training data.

**Prediction**: SWA produces a **deterministic prediction** using this single weight vector:

$$\hat{y}(\mathbf{x}^{\ast}) = f(\mathbf{x}^{\ast};\, \mathbf{w}_{\text{SWA}})$$

**Why does simple averaging yield a good solution?** The explanation is geometric. With a cyclical learning rate, SGD visits multiple "peripheral" solutions within a mode of the loss landscape. Their average tends to lie at the **center** of the mode — a broader, flatter region. It is empirically known that wide, flat minima generalize better (Keskar et al., 2017), and SWA achieves this effect indirectly.

```
Loss landscape (1D cross-section):

Loss │     SGD checkpoints (periphery)
     │     ·  ·        ·  ·
     │    ╱    ╲      ╱    ╲
     │   ╱      ╲    ╱      ╲
     │  ╱   SWA  ╲  ╱        ╲
     │ ╱   (center)╲╱          ╲
     └──────────────────────────── w
              ▲
         Flat minimum
         (better generalization)
```

**Limitation of SWA**: SWA is **not** a Bayesian method. It yields a single point estimate and provides no information about uncertainty. There is no way to gauge the confidence of the prediction. SWA is a better **regularization** technique, not an **uncertainty quantification** method.

#### 2.7.2 SWAG — Stochastic Weight Averaging Gaussian: From Point Estimate to Posterior

**Core idea**: Where SWA uses only the **first-order statistics (the mean)** of the SGD trajectory, SWAG additionally uses the **second-order statistics (the covariance)**. It models the variance structure as a low-rank Gaussian to approximate the posterior, then **samples** weights from it to construct a predictive distribution.

SWAG was proposed by Maddox et al. (NeurIPS 2019, "A Simple Baseline for Bayesian Uncertainty in Deep Learning"):

**Steps 1-2**: Identical to SWA. After standard training, collect weight snapshots $\mathbf{w}_1, ..., \mathbf{w}_K$ via cyclical/high LR SGD.

**Step 3 — Construct the Gaussian Posterior** (the point of divergence from SWA):

SWA takes only the average and stops, but SWAG estimates the **covariance structure** as well:

- **Mean** (same as SWA):

$$\bar{\mathbf{w}} = \frac{1}{K}\sum_{k=1}^{K}\mathbf{w}_k$$

- **Diagonal variance**:

$$\text{diag}(\boldsymbol{\Sigma}_{\text{diag}}) = \frac{1}{K}\sum_{k=1}^{K}(\mathbf{w}_k - \bar{\mathbf{w}})^2$$

- **Low-rank component**: Deviation vectors form a rank-$K$ matrix:

$$\mathbf{d}_k = \mathbf{w}_k - \bar{\mathbf{w}}$$

These are combined into a Gaussian posterior:

$$q_{\text{SWAG}}(\mathbf{w}) = \mathcal{N}\Big(\bar{\mathbf{w}},\, \frac{1}{2}\boldsymbol{\Sigma}_{\text{diag}} + \frac{1}{2}\mathbf{D}\mathbf{D}^T\Big)$$

where:

$$\mathbf{D} = [\mathbf{d}_1, ..., \mathbf{d}_K] / \sqrt{K-1}$$

**Step 4 — Weight Sampling and Predictive Distribution** (the decisive difference from SWA):

SWAG **samples** $T$ weights from this Gaussian:

$$\tilde{\mathbf{w}}_t \sim q_{\text{SWAG}}(\mathbf{w}), \quad t = 1, ..., T$$

Each sampled weight produces an independent forward pass, and the results are aggregated into a **predictive distribution**:

$$\hat{\mu}(\mathbf{x}^{\ast}) = \frac{1}{T}\sum_{t=1}^{T} f(\mathbf{x}^{\ast};\, \tilde{\mathbf{w}}_t), \quad \hat{\sigma}^2(\mathbf{x}^{\ast}) = \frac{1}{T}\sum_{t=1}^{T}\big(f(\mathbf{x}^{\ast};\, \tilde{\mathbf{w}}_t) - \hat{\mu}(\mathbf{x}^{\ast})\big)^2$$

This is where SWAG fundamentally differs from SWA: **SWA uses one weight to make one prediction, whereas SWAG draws multiple weights from the posterior and obtains a distribution over predictions, whose variance quantifies uncertainty.**

#### 2.7.3 SWA vs. SWAG — A Side-by-Side Comparison

The differences between the two methods, stated precisely:

| | SWA | SWAG |
|:---|:---|:---|
| **Statistics used** | 1st-order (mean only) | 1st + 2nd-order (mean + covariance) |
| **Output** | Single weight $\mathbf{w}_{\text{SWA}}$ | Gaussian posterior $q(\mathbf{w})$ |
| **Prediction** | Deterministic: $f(\mathbf{x}^{\ast};\, \mathbf{w}_{\text{SWA}})$ | Stochastic: sample $T$ times, then average |
| **Uncertainty** | Not provided | $\hat{\sigma}^2$ provided (epistemic) |
| **Nature** | Better regularization | Approximate Bayesian inference |
| **Inference cost** | 1× | $T$× (typically $T = 30$) |
| **Theoretical standing** | Improved point estimate | Posterior approximation |

```
SGD trajectory snapshots: w₁, w₂, w₃, ..., wₖ
            │
            ├──→ SWA:  average → w̄  → single prediction
            │                         (no uncertainty)
            │
            └──→ SWAG: average → w̄  ─┐
                 + covariance → Σ  ─┤→ Gaussian q(w)
                                    │   → sample w̃₁, w̃₂, ..., w̃ₜ
                                    │   → T predictions
                                    │   → mean + variance
                                    └── (uncertainty quantified)
```

#### 2.7.4 Theoretical Justification — Why Does the SGD Trajectory Encode Posterior Information?

The common premise underlying both SWA and SWAG is that **the SGD trajectory carries information about the posterior**. The theoretical basis for this intuition lies in the connection to Stochastic Gradient Langevin Dynamics (SGLD; Welling & Teh, 2011).

In SGLD, the weight update is:

$$\mathbf{w}_{t+1} = \mathbf{w}_t + \frac{\epsilon_t}{2}\nabla_{\mathbf{w}}\log p(\mathbf{w}_t \mid \mathcal{D}) + \boldsymbol{\eta}_t, \quad \boldsymbol{\eta}_t \sim \mathcal{N}(\mathbf{0}, \epsilon_t \mathbf{I})$$

Under appropriate conditions (learning rate $\epsilon_t \to 0$), the stationary distribution of SGLD converges to the posterior $p(\mathbf{w} \mid \mathcal{D})$. Standard SGD with mini-batching is essentially SGLD without explicit noise injection, but the stochasticity from mini-batching serves a similar role. Hence, the phenomenon of SGD "wandering" around a particular region of the loss landscape even after convergence suggests that this region corresponds to a high-posterior-probability area (the typical set).

SWA estimates the center of this typical set, while SWAG summarizes its **shape** as a Gaussian.

#### 2.7.5 Strengths, Limitations, and Extensions

**Strengths**:

- Applicable with minimal modification to standard SGD pipelines (SWA: average checkpoints; SWAG: additionally compute variance)
- SWAG's low-rank approximation captures some inter-weight correlations (richer posterior representation than diagonal-only VI)
- The transition from SWA to SWAG is incremental: obtain a good point estimate with SWA, then add covariance to unlock uncertainty

**Limitations**:

- Approximates the posterior within a single mode only — unable to capture the multi-modal structure of the loss landscape
- Gaussian assumption: Approximation quality degrades when the posterior is non-Gaussian (asymmetric, heavy-tailed, etc.)
- Sensitive to learning rate schedule: Requires tuning hyperparameters such as cycle length and min/max learning rate

**MultiSWAG**: To overcome SWAG's single-mode limitation, run SWAG independently from different random initializations and combine the results. This is a hybrid of deep ensembles and SWAG, where each ensemble member captures the **local Gaussian posterior of a different mode**. In effect, the multi-modal posterior is approximated as a Gaussian mixture.

**Trade-offs against the four barriers** (for SWAG):

- *Non-conjugacy*: Approximated with a Gaussian
- *Dimensionality*: Low-rank decomposition with $O(dK)$ storage ($K \ll d$)
- *Multimodality*: Captures a single mode only (partially mitigated by MultiSWAG)
- *Storage*: $O(dK)$ — modest additional cost

---

### 2.8 Stein Variational Gradient Descent — A Concerto of Particles

**Core idea**: Place $K$ "particles" $\lbrace\mathbf{w}_1, ..., \mathbf{w}_K\rbrace$ in weight space and **deterministically** move them to match the posterior. Each particle is drawn toward high-probability regions via the gradient, while a kernel function pushes particles apart to maintain diversity.

SVGD, proposed by Liu and Wang (NeurIPS 2016), uses the update rule:

$$\mathbf{w}_i \leftarrow \mathbf{w}_i + \epsilon \cdot \hat{\phi}^{\ast}(\mathbf{w}_i)$$

$$\hat{\phi}^{\ast}(\mathbf{w}) = \frac{1}{K}\sum_{j=1}^{K}\Big[\underbrace{k(\mathbf{w}_j, \mathbf{w})\nabla_{\mathbf{w}_j}\log p(\mathbf{w}_j \mid \mathcal{D})}_{\text{attractive: gradient term}} + \underbrace{\nabla_{\mathbf{w}_j} k(\mathbf{w}_j, \mathbf{w})}_{\text{repulsive: kernel term}}\Big]$$

- **Attractive term**: Kernel-weighted posterior gradient. Pulls each particle toward high-posterior-probability regions
- **Repulsive term**: Gradient of the kernel. Pushes particles apart when they cluster too closely, preventing mode collapse

When $K=1$, SVGD degenerates to MAP estimation; as $K \to \infty$, it converges to the true posterior. For finite $K$, SVGD can be seen as a "principled generalization" of deep ensembles — whereas ensembles train members independently without interaction, SVGD explicitly models repulsive interactions between particles.

**Practical standing**: Theoretically more elegant than ensembles, but requires kernel selection and bandwidth tuning. In high dimensions, the kernel's repulsive force weakens (variance collapse), so SVGD does not clearly surpass deep ensembles.

---

## 3. Comparative Analysis — What, How, and When

### 3.1 Summary: How Each Method Handles the Four Barriers

| Method | Non-conjugacy | Dimensionality | Multimodality | Storage | Provides UQ |
|:-------|:---:|:---:|:---:|:---:|:---:|
| **Laplace** | Gaussian forced | KFAC/Diagonal/Last-Layer | Single mode | $O(d)$ ~ $O(d^2)$ | Yes |
| **HMC** | No assumption | Gradient-based | Theoretically possible | $O(Td)$ | Yes |
| **VI (BbB)** | Gaussian forced | Mean-field $O(2d)$ | Single mode | $O(2d)$ | Yes |
| **MC Dropout** | Bernoulli forced | $O(d)$ maintained | Limited | $O(d)$ | Yes |
| **Deep Ensemble** | No assumption | $O(Md)$ | **Multiple modes** | $O(Md)$ | Yes |
| **SWA** | — (point est.) | $O(d)$ | Single mode | $O(d)$ | **No** |
| **SWAG** | Gaussian forced | Low-rank $O(dK)$ | Single mode | $O(dK)$ | Yes |
| **SVGD** | No assumption | Particle $O(Kd)$ | Theoretically possible | $O(Kd)$ | Yes |

*Note: SWA is not a Bayesian method; it is a regularization technique for obtaining a better point estimate. It serves as the precursor (weight averaging step) to SWAG but does not by itself quantify uncertainty.*

### 3.2 Empirical Performance Comparison

Synthesizing consistent patterns reported across various benchmarks:

**Calibration (ECE / NLL)**:
```
Deep Ensemble ≥ SWAG > Laplace (Last-Layer) > MC Dropout > VI (Mean-field)
```

**OOD Detection (AUROC)**:
```
Deep Ensemble ≥ SVGD > SWAG > MC Dropout > VI
```

**Computational Efficiency**:
```
MC Dropout > Laplace (post-hoc) > SWAG > VI > Deep Ensemble > HMC
```

**Ease of Implementation**:
```
MC Dropout > Deep Ensemble > SWAG > Laplace > VI > SVGD > HMC
```

### 3.3 When to Use What — A Decision Tree

```
Do you already have a trained model?
├── Yes ─→ Does the model use dropout?
│          ├── Yes ─→ MC Dropout (easiest)
│          └── No ──→ Last-Layer Laplace (post-hoc)
│
└── No ──→ Is the computational budget sufficient?
           ├── Yes (M×) ─→ Deep Ensemble (most robust)
           ├── Moderate (1.2-2×) ─→ SWAG or VI
           └── No (1×) ──→ Single-pass methods covered in Part 5
                            (EDL, SNGP, DUQ)
```

---

## 4. Open Questions and Ongoing Research

### 4.1 Prior Specification — An Unresolved Fundamental Problem

Every Bayesian method requires a prior $p(\mathbf{w})$. Currently, most BDL work uses an isotropic Gaussian prior $p(\mathbf{w}) = \mathcal{N}(\mathbf{0}, \lambda^{-1}\mathbf{I})$, which amounts to the very weak assumption that "all weights are independently near zero."

The inadequacy of this prior is clear: what we actually know about the prior distribution of weights in real neural networks far exceeds what an isotropic Gaussian captures — for instance, weight magnitudes systematically vary with layer depth, and in certain architectures the weight matrices should exhibit approximate orthogonality. Yet methods for encoding such structural knowledge into priors have not been standardized.

### 4.2 Weight Space vs. Function Space — Where Should Inference Be Performed?

All methods discussed so far compute the posterior in **weight space**. But what we truly care about is not the weights but the **function** $f(\cdot; \mathbf{w})$. The complex symmetry structure of weight space (the permutation symmetry of Section 1.3) vanishes in function space.

This observation has sparked interest in **function-space inference**. Gaussian Processes (GPs) natively perform Bayesian inference in function space, and Neural Tangent Kernel (NTK) theory shows that infinitely wide neural networks converge to GPs. Linearized Laplace practically leverages this connection by treating a trained network as a generalized linear model in feature space and performing GP inference.

### 4.3 Bayesian Inference in the Foundation Model Era

Computing the posterior over the full weight space of models with billions to hundreds of billions of parameters will remain infeasible for the foreseeable future. Recent practical alternatives:

- **Bayesian LoRA (2024)**: Shows that GP inference is structurally equivalent to LoRA's low-rank factorization, and performs Bayesian treatment only on the LoRA parameters
- **SWAG-LoRA (2024)**: Applies SWAG to the SGD trajectory of LoRA parameters
- **BLoB — Bayesian LoRA by Backpropagation (2024)**: Jointly learns the mean and covariance of LoRA parameters via backpropagation
- **C-LoRA (2025)**: Provides sample-dependent, context-aware uncertainty at each layer

Their shared insight: **Bayesian inference in a low-dimensional subspace can be more effective than crude approximation over the entire weight space.** Effective calibration is achievable with fewer than 1,000 additional parameters.

### 4.4 The Return of Metropolis-Hastings (Nature Communications, 2026)

A recent study published in Nature Communications (2026) proposed integrating a computationally lightweight Metropolis-Hastings acceptance step into Stochastic Gradient HMC, eliminating the asymptotic bias inherent in conventional SG-MCMC while maintaining practical cost. This reopens the possibility that sampling-based methods can be viable at the scale of deep learning.

---

## Closing Thoughts: What the Art of Approximation Teaches Us

The seven methods covered in this post are all different compromises for the same problem — an **intractable posterior**. Summarizing the current landscape after thirty years of research:

1. **Theoretical elegance and practical performance are separate matters.** The most theoretically justified methods (HMC, VI) do not necessarily yield the best calibration. Deep ensembles received their theoretical justification retrospectively, yet they have been empirically the strongest throughout.

2. **"Where to approximate" matters as much as "how to approximate."** The success of Last-Layer Laplace and Bayesian LoRA shows that precise inference in a carefully chosen subspace can outperform crude inference over the entire weight space.

3. **The ability to handle multimodality determines calibration quality.** The primary reason deep ensembles and SVGD outperform VI and single-mode Laplace is that they capture multiple modes of the posterior. This was confirmed by the HMC experiments of Izmailov et al. (2021).

4. **No method is perfect; context determines the choice.** If you need to add uncertainty to a pre-trained model, use Laplace or MC Dropout. If you can train from scratch with sufficient compute, use deep ensembles. If you are fine-tuning a foundation model, consider the Bayesian LoRA family.

In the next installment, Part 3, we dissect the resulting uncertainty into **what kind of uncertainty it represents** — the mathematical foundations of the aleatoric vs. epistemic distinction, and how this distinction provides practical value for real-world decision-making (e.g., "should we collect more data, or improve the model?").

---

## References

### Foundational
1. MacKay, D. J. C. (1992). "A Practical Bayesian Framework for Backpropagation Networks." *Neural Computation*, 4(3), 448-472.
2. Neal, R. M. (1996). *Bayesian Learning for Neural Networks.* Springer.
3. Hinton, G. E. & Van Camp, D. (1993). "Keeping Neural Networks Simple by Minimizing the Description Length of the Weights." *COLT*.

### Variational Methods
4. Graves, A. (2011). "Practical Variational Inference for Neural Networks." *NeurIPS*.
5. Blundell, C., Cornebise, J., Kavukcuoglu, K. & Wierstra, D. (2015). "Weight Uncertainty in Neural Networks." *ICML*.
6. Gal, Y. & Ghahramani, Z. (2016). "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning." *ICML*.
7. Gal, Y., Hron, J. & Kendall, A. (2017). "Concrete Dropout." *NeurIPS*.
8. Liu, Q. & Wang, D. (2016). "Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm." *NeurIPS*.

### Ensembles and Scalable Methods
9. Lakshminarayanan, B., Pritzel, A. & Blundell, C. (2017). "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles." *NeurIPS*.
10. Izmailov, P., Podoprikhin, D., Garipov, T., Vetrov, D. & Wilson, A. G. (2018). "Averaging Weights Leads to Wider Optima and Better Generalization." *UAI*. (SWA)
11. Maddox, W. J., Garipov, T., Izmailov, P., Vetrov, D. & Wilson, A. G. (2019). "A Simple Baseline for Bayesian Uncertainty in Deep Learning." *NeurIPS*. (SWAG)
12. Wilson, A. G. & Izmailov, P. (2020). "Bayesian Deep Learning and a Probabilistic Perspective of Generalization." *NeurIPS*.
13. Pearce, T. et al. (2023). "A Rigorous Link between Deep Ensembles and (Variational) Bayesian Methods." *arXiv:2305.15027*.

### Modern Developments
14. Daxberger, E. et al. (2021). "Laplace Redux — Effortless Bayesian Deep Learning." *NeurIPS*.
15. Izmailov, P., Vikram, S., Hoffman, M. D. & Wilson, A. G. (2021). "What Are Bayesian Neural Network Posteriors Really Like?" *ICML*.
16. Wenzel, F. et al. (2020). "How Good is the Bayes Posterior in Deep Neural Networks Really?" *ICML*.
17. Welling, M. & Teh, Y. W. (2011). "Bayesian Learning via Stochastic Gradient Langevin Dynamics." *ICML*. (SGLD — theoretical basis for SWA/SWAG)

### Scientific Applications
18. Ryu, S., Kwon, Y. & Kim, W. Y. (2019). "A Bayesian Graph Convolutional Network for Reliable Prediction of Molecular Properties with Uncertainty Quantification." *Chemical Science*, 10, 8438-8446.

### Foundation Model Era
19. Bayesian LoRA (2024). *arXiv:2601.21003*.
20. SWAG-LoRA (2024). *arXiv:2405.03425*.
21. BLoB — Bayesian LoRA by Backpropagation (2024). *arXiv:2406.11675*.
22. Nature Communications (2026). "Reliable Uncertainty via Metropolis-Hastings for Deep Networks."
23. Papamarkou, T. et al. (2024). "Position: Bayesian Deep Learning is Needed in the Age of Large-Scale AI." *ICML*.
24. Keskar, N. S. et al. (2017). "On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima." *ICLR*.
