---
title: "Causal Inference Part 5: Beyond the Average — Heterogeneous Effects and Causal Discovery"
date: 2026-04-06 14:00:00 +0900
categories: [AI & ML, ML Theory]
tags: [causal-inference, heterogeneous-effects, cate, causal-forests, causal-discovery]
math: true
---

**Causal Inference — From First Principles to Automated Reasoning**

*This is Part 5 of an 8-part series.*

- **Part 0:** [Beyond Correlation — Why Causal Inference?](/posts/causal-inference-part0-beyond-correlation/)
- **Part 1:** [Potential Outcomes and the Fundamental Problem](/posts/causal-inference-part1-potential-outcomes/)
- **Part 2:** [Structural Causal Models and DAGs](/posts/causal-inference-part2-structural-causal-models/)
- **Part 3:** [Identification — From Design to Estimand](/posts/causal-inference-part3-identification/)
- **Part 4:** [Estimation — From Estimand to Estimate](/posts/causal-inference-part4-estimation/)
- **Part 5:** Beyond the Average — Heterogeneous Effects and Causal Discovery (this post)
- **Part 6:** [End-to-End Practice — The Causal Pipeline in Code](/posts/causal-inference-part6-causal-pipeline/)
- **Part 7:** [The Causal Agent — LLM-Driven Causal Reasoning](/posts/causal-inference-part7-causal-agent/)

---

## Hook: Your Patient Is Not Average

The average treatment effect is a useful fiction. It tells you that a drug works *on average* — but your patient is not average. A job training program raises earnings by 2,000 USD overall, but perhaps 8,000 USD for workers without a college degree and nothing for those with one. A new catalyst increases yield by 5% across substrates, but 20% for electron-poor aromatics and $-3$% for sterically hindered ones.

**An ATE of zero can hide large positive and negative effects that cancel in aggregation.** Worse, a positive ATE can mask subgroups that are actively harmed. If we stop at the average, we leave value on the table and risk on the patient.

This post develops two extensions. First, *heterogeneous treatment effects* — the machinery for estimating $\tau(\mathbf{x})$, the causal effect as a function of covariates. This is the applied frontier: personalized medicine, policy targeting, and optimal treatment assignment all require it. Second, *causal discovery* — algorithms that learn the causal graph itself from data, inverting the problem we have solved so far. Both capabilities feed directly into the automated pipeline of Parts 6 and 7.

---

## 1. Why Heterogeneity Matters

### 1.1 The Hidden Structure Behind the Average

Parts 3-4 estimated a single number: the ATE $\tau = E[Y_i(1) - Y_i(0)]$ or the ATT $\tau_{\text{ATT}}$. **This summary is complete only when the treatment effect is constant across units — a condition that almost never holds in practice.**

Consider a clinical trial for a blood pressure medication. The ATE is $-5$ mmHg. But suppose:

- Patients over 65: $\tau = -12$ mmHg (large benefit)
- Patients under 40 with normal baseline: $\tau = +2$ mmHg (slight harm)
- Patients with kidney disease: $\tau = -3$ mmHg (modest benefit, high side-effect risk)

The aggregate number masks clinically actionable variation. A physician who prescribes uniformly based on the ATE overtreats some patients and undertreats others.

### 1.2 A Numerical Example: ATE of Zero, Maximal Heterogeneity

Consider an even starker scenario. A policy intervention has zero ATE — it appears useless.

| Subgroup | $N$ | $E[Y(1)]$ | $E[Y(0)]$ | $\tau(\mathbf{x})$ | Interpretation |
|----------|:---:|:---------:|:---------:|:-------------------:|----------------|
| Young, low-income | 500 | 45,000 | 35,000 | $+10{,}000$ | Large benefit |
| Young, high-income | 300 | 62,000 | 65,000 | $-3{,}000$ | Slight harm |
| Older, low-income | 400 | 38,000 | 32,000 | $+6{,}000$ | Moderate benefit |
| Older, high-income | 800 | 70,000 | 78,000 | $-8{,}000$ | Substantial harm |
| **Population** | **2,000** | — | — | **$\approx 0$** | **Appears useless** |

The weighted average across subgroups is approximately zero. **A policy-maker who sees only the ATE would cancel a program that delivers \$10,000 in earnings gains to young, low-income workers.** The CATE function $\tau(\mathbf{x})$ reveals the structure hidden by the average: the program works powerfully for some subgroups and harms others. Optimal policy targets the former and excludes the latter.

### 1.3 The Conditional Average Treatment Effect

> **Definition 5.1 (CATE)**: The Conditional Average Treatment Effect is the expected treatment effect conditional on covariates:
>
> $$\tau(\mathbf{x}) = E[Y_i(1) - Y_i(0) \mid \mathbf{X}_i = \mathbf{x}]$$
>
> The CATE is a *function* $\tau : \mathbb{R}^p \to \mathbb{R}$, not a scalar. It maps each covariate profile to the expected effect for units with that profile.

The CATE inherits the same identification challenges as the ATE. Under unconfoundedness (Assumption 1.2 from Part 1) and overlap:

$$\tau(\mathbf{x}) = E[Y_i \mid W_i = 1, \mathbf{X}_i = \mathbf{x}] - E[Y_i \mid W_i = 0, \mathbf{X}_i = \mathbf{x}] = \mu_1(\mathbf{x}) - \mu_0(\mathbf{x})$$

This looks simple — just estimate two conditional expectations and subtract. The difficulty is statistical: we need to estimate a *function* rather than a *scalar*, with data that may be sparse in many regions of $\mathbf{X}$-space.

### 1.4 Why CATE Estimation Is Harder Than ATE Estimation

| Property | ATE | CATE |
|----------|-----|------|
| **Target** | Scalar $\tau \in \mathbb{R}$ | Function $\tau(\mathbf{x}) : \mathbb{R}^p \to \mathbb{R}$ |
| **Convergence rate** | $O(N^{-1/2})$ (parametric) | $O(N^{-2/(2+p)})$ (nonparametric, curse of dimensionality) |
| **Evaluation** | Compare to RCT benchmark | No ground truth per subgroup |
| **Bias-variance** | Averaging reduces variance | Local estimation amplifies variance |
| **Overlap requirement** | $0 < e(\mathbf{x}) < 1$ globally | $e(\mathbf{x})$ bounded away from 0, 1 *locally* at each $\mathbf{x}$ |

**The curse of dimensionality hits CATE estimation directly: estimating a treatment effect at a specific covariate value requires sufficient treated and control units *near* that value.** This is why structured estimators — meta-learners and causal forests — are essential.

### 1.5 Applications

- **Personalized medicine**: assign treatments based on predicted individual benefit rather than population averages
- **Policy targeting**: allocate scarce program slots to individuals with the highest expected gain, maximizing social welfare per dollar spent
- **Optimal treatment rules**: learn a decision function $\pi^*(\mathbf{x}) = \mathbf{1}[\hat{\tau}(\mathbf{x}) > 0]$ that assigns treatment only to units with positive predicted effect. The policy value is $V(\pi) = E[Y_i(\pi(\mathbf{X}_i))]$.
- **Scientific discovery**: identify which features moderate a causal effect (effect modification), generating testable hypotheses about mechanisms
- **Computational science**: in molecular design, identify which scaffold features predict large activity differences between paired compounds; in materials science, determine which process parameters yield heterogeneous responses to a fabrication change

---

## 2. Meta-Learners — A Modular Framework

### 2.1 The Core Idea

A meta-learner is not a specific algorithm but a *strategy for decomposing the CATE estimation problem into standard supervised learning subproblems*. The prefix "meta" means that any base learner (random forest, gradient boosting, neural network) can be plugged into the framework.

**The key insight: a meta-learner defines *how* to decompose the CATE problem, not *which* base ML model to use.**

```
META-LEARNER TAXONOMY

                        CATE Estimation
                              │
              ┌───────────────┼───────────────┐
              │               │               │
         Direct Modeling   Cross-Group    Residualized
              │             Imputation     Approaches
         ┌────┴────┐          │           ┌────┴────┐
      S-learner  T-learner  X-learner  R-learner  DR-learner
      (single    (two       (Kunzel    (Robinson  (doubly
       model)    models)    et al.)    decomp.)   robust)
```

We define each learner formally, then compare.

### 2.2 S-Learner (Single Model)

**Strategy**: Fit a single outcome model that includes treatment as a feature. Estimate the CATE by varying the treatment input.

$$\hat{\mu}(\mathbf{x}, w) \approx E[Y_i \mid \mathbf{X}_i = \mathbf{x}, W_i = w]$$

$$\hat{\tau}^S(\mathbf{x}) = \hat{\mu}(\mathbf{x}, 1) - \hat{\mu}(\mathbf{x}, 0)$$

The model sees all data at once. Treatment is simply another covariate.

- **Strength**: Maximum sample size for fitting; simple to implement
- **Weakness**: If the base learner regularizes $W$ toward zero (as tree-based methods do when $W$ contributes little to prediction), the estimated CATE shrinks toward zero. **The S-learner is biased toward homogeneity.**
- **Best for**: Settings where the treatment effect is small relative to the outcome variation

### 2.3 T-Learner (Two Models)

**Strategy**: Fit separate outcome models for treated and control groups. Estimate the CATE by subtraction.

$$\hat{\mu}_1(\mathbf{x}) \approx E[Y_i \mid W_i = 1, \mathbf{X}_i = \mathbf{x}], \quad \hat{\mu}_0(\mathbf{x}) \approx E[Y_i \mid W_i = 0, \mathbf{X}_i = \mathbf{x}]$$

$$\hat{\tau}^T(\mathbf{x}) = \hat{\mu}_1(\mathbf{x}) - \hat{\mu}_0(\mathbf{x})$$

Each model is trained on its respective subgroup.

- **Strength**: No assumption that the treatment and control outcome surfaces share structure; can capture arbitrary heterogeneity
- **Weakness**: Each model uses only a fraction of the data. When one group is small (e.g., $n_1 \ll n_0$), the treated-group model is unstable. **The T-learner is biased toward spurious heterogeneity when groups are unbalanced.**
- **Best for**: Balanced designs with large sample sizes in both groups

### 2.4 X-Learner (Cross-Group Imputation)

Kunzel et al. (2019) introduced the X-learner to handle imbalanced treatment groups — a common situation in observational studies.

**Strategy** (three stages):

**Stage 1**: Fit outcome models as in the T-learner: $\hat{\mu}_0(\mathbf{x})$ and $\hat{\mu}_1(\mathbf{x})$.

**Stage 2**: Impute individual treatment effects using the *other* group's model:

$$\hat{\tau}_i^1 = Y_i - \hat{\mu}_0(\mathbf{X}_i) \quad \text{for treated units } (W_i = 1)$$

$$\hat{\tau}_i^0 = \hat{\mu}_1(\mathbf{X}_i) - Y_i \quad \text{for control units } (W_i = 0)$$

For treated units, we observe $Y_i(1)$ and impute $Y_i(0)$. For control units, the reverse. Each imputed effect is a noisy estimate of $\tau(\mathbf{X}_i)$.

**Stage 3**: Fit two CATE models on the imputed effects, then combine using the propensity score as a weight:

$$\hat{\tau}^X(\mathbf{x}) = e(\mathbf{x}) \cdot \hat{\tau}^{1,\text{model}}(\mathbf{x}) + (1 - e(\mathbf{x})) \cdot \hat{\tau}^{0,\text{model}}(\mathbf{x})$$

The propensity-weighted combination assigns more weight to the model trained on the larger group, where estimation is more reliable.

- **Strength**: Leverages the larger group's model to stabilize estimation in the smaller group; excels when treatment prevalence is low
- **Weakness**: Three-stage procedure introduces complexity; performance depends on quality of Stage 1 models
- **Best for**: Observational studies with unbalanced treatment groups (e.g., rare treatment)

### 2.5 R-Learner (Robinson Decomposition)

The R-learner (Nie and Wager, 2021) reformulates CATE estimation as a single loss minimization, building on Robinson's (1988) partially linear model decomposition.

**Strategy**: Write the outcome model as:

$$Y_i = \mu(\mathbf{X}_i) + \tau(\mathbf{X}_i)(W_i - e(\mathbf{X}_i)) + \epsilon_i$$

where $\mu(\mathbf{x}) = E[Y_i \mid \mathbf{X}_i = \mathbf{x}]$ is the marginal outcome and $e(\mathbf{x})$ is the propensity score. Define residualized quantities:

$$\tilde{Y}_i = Y_i - \hat{\mu}(\mathbf{X}_i), \quad \tilde{W}_i = W_i - \hat{e}(\mathbf{X}_i)$$

Then estimate $\tau(\cdot)$ by minimizing the R-loss:

$$\hat{\tau}^R = \arg\min_{\tau} \sum_{i=1}^{N} \left( \tilde{Y}_i - \tau(\mathbf{X}_i) \tilde{W}_i \right)^2 + \Lambda(\tau)$$

where $\Lambda(\tau)$ is a regularization penalty.

- **Strength**: Orthogonalizes the CATE estimation from nuisance components ($\mu$, $e$), reducing sensitivity to misspecification of either. Inherits the double-robustness flavor of DML (Part 4).
- **Weakness**: Requires good estimates of both $\mu(\mathbf{x})$ and $e(\mathbf{x})$; the R-loss is non-convex in general when $\tau$ is nonparametric.
- **Best for**: High-dimensional settings where DML-style orthogonalization matters

### 2.6 DR-Learner (Doubly Robust)

Kennedy (2023) showed that the AIPW pseudo-outcome — the same doubly robust score from Part 4 — can serve directly as a target for CATE regression.

**Strategy**: Compute the doubly robust pseudo-outcome for each unit:

$$\hat{\Gamma}_i = \hat{\mu}_1(\mathbf{X}_i) - \hat{\mu}_0(\mathbf{X}_i) + \frac{W_i(Y_i - \hat{\mu}_1(\mathbf{X}_i))}{\hat{e}(\mathbf{X}_i)} - \frac{(1 - W_i)(Y_i - \hat{\mu}_0(\mathbf{X}_i))}{1 - \hat{e}(\mathbf{X}_i)}$$

This is the AIPW influence function evaluated at unit $i$. Its conditional expectation equals the CATE:

$$E[\hat{\Gamma}_i \mid \mathbf{X}_i = \mathbf{x}] = \tau(\mathbf{x})$$

Then regress $\hat{\Gamma}_i$ on $\mathbf{X}_i$ using any nonparametric method:

$$\hat{\tau}^{DR}(\mathbf{x}) = \hat{E}[\hat{\Gamma}_i \mid \mathbf{X}_i = \mathbf{x}]$$

- **Strength**: Inherits double robustness — $\hat{\tau}^{DR}$ is consistent if either $\hat{\mu}_w$ or $\hat{e}$ is consistent. The pseudo-outcome is an unbiased signal for the CATE, enabling standard nonparametric regression tools. Achieves minimax-optimal rates under regularity conditions.
- **Weakness**: Pseudo-outcomes can have high variance when $\hat{e}(\mathbf{x})$ is near 0 or 1 (overlap violations); requires trimming or clipping.
- **Best for**: General-purpose CATE estimation; the default recommendation when both outcome and propensity models are available.

### 2.7 Comparison Table

| Learner | # Models | Nuisance Inputs | Double Robust? | Handles Imbalance? | Complexity |
|---------|:--------:|:---------------:|:--------------:|:------------------:|:----------:|
| **S-learner** | 1 | None | No | Neutral | Low |
| **T-learner** | 2 | None | No | Poor | Low |
| **X-learner** | 3+2 | $\hat{e}(\mathbf{x})$ | No | Excellent | Medium |
| **R-learner** | 1+2 | $\hat{\mu}(\mathbf{x})$, $\hat{e}(\mathbf{x})$ | Partial | Good | Medium |
| **DR-learner** | 1+2 | $\hat{\mu}_w(\mathbf{x})$, $\hat{e}(\mathbf{x})$ | Yes | Good | Medium |

> **Pitfall**: A common mistake is to evaluate meta-learners by their predictive performance on $Y$. The target is $\tau(\mathbf{x})$, which is never observed for any individual unit. Evaluation requires either (a) a randomized experiment with known subgroup effects, (b) semi-synthetic benchmarks where $\tau(\mathbf{x})$ is known by construction, or (c) policy value metrics (e.g., mean outcome under the learned treatment rule).

### 2.8 Practical Recommendations

**Default choice**: Start with the DR-learner. It is doubly robust, handles moderate imbalance, and its pseudo-outcome construction naturally separates nuisance estimation from effect estimation.

**When treatment is rare** ($e(\mathbf{x}) < 0.1$ for many units): Use the X-learner. Its cross-imputation strategy borrows strength from the large control group.

**When interpretability matters**: Use the T-learner with interpretable base models (e.g., gradient-boosted trees with SHAP). The separate models allow direct inspection of what drives outcomes in each treatment arm.

**When double robustness is critical but resources are limited**: The R-learner provides partial orthogonalization with a single CATE model fit, whereas the DR-learner requires computing pseudo-outcomes as an intermediate step.

In all cases, cross-validate the nuisance models ($\hat{\mu}_w$, $\hat{e}$) using out-of-fold predictions to avoid overfitting — the same cross-fitting strategy from DML in Part 4 applies here.

---

## 3. Causal Forests

### 3.1 From Random Forests to Causal Forests

Meta-learners decompose the CATE problem into supervised learning subproblems, then rely on off-the-shelf ML. Causal forests, introduced by Wager and Athey (2018), take a different approach: **modify the splitting criterion of random forests to directly target treatment effect heterogeneity.**

A standard random forest splits to maximize predictive accuracy — reducing variance of $Y$ within child nodes. A causal forest splits to maximize heterogeneity in $\hat{\tau}$ across child nodes.

### 3.2 The Splitting Criterion

At each candidate split of a parent node $P$ into children $C_L$ and $C_R$, the causal forest evaluates:

$$\Delta(\text{split}) = \frac{N_{C_L} \cdot N_{C_R}}{N_P^2} \left( \hat{\tau}_{C_L} - \hat{\tau}_{C_R} \right)^2$$

where $\hat{\tau}\_{C} = \bar{Y}\_{C}^{\text{treated}} - \bar{Y}\_{C}^{\text{control}}$ is the within-leaf treatment effect estimate and $N\_{C}$ is the number of units in child $C$. The split that maximizes $\Delta$ creates the greatest separation in treatment effects.

**The forest splits where treatment effects differ, not where outcomes differ.** A region where $Y$ varies dramatically but $\tau$ is constant will not be split.

### 3.3 Honesty — The Key to Valid Inference

Standard random forests use the same data to determine the tree structure and to compute predictions within leaves. This double use invalidates standard confidence intervals.

> **Definition 5.2 (Honest Estimation)**: A tree is *honest* if it uses one subsample ($\mathcal{I}^{\text{split}}$) to determine the tree structure (which splits, where) and a disjoint subsample ($\mathcal{I}^{\text{est}}$) to estimate the treatment effect within each leaf.

**Honesty is the price of inference. By splitting the sample, each leaf's $\hat{\tau}$ is computed from data that played no role in selecting the leaf — eliminating the adaptive bias that makes standard forest predictions overconfident.**

The procedure:

1. Draw a subsample of size $s$ from the training data
2. Split into $\mathcal{I}^{\text{split}}$ (for tree building) and $\mathcal{I}^{\text{est}}$ (for leaf estimation)
3. Grow the tree on $\mathcal{I}^{\text{split}}$ using the causal splitting criterion
4. Drop $\mathcal{I}^{\text{est}}$ into the fitted tree; compute $\hat{\tau}_{\text{leaf}}$ from the estimation sample only
5. Repeat across $B$ trees; average predictions

Wager and Athey (2018) proved that honest causal forests are asymptotically normal:

$$\frac{\hat{\tau}(\mathbf{x}) - \tau(\mathbf{x})}{\hat{\sigma}(\mathbf{x})} \xrightarrow{d} \mathcal{N}(0, 1)$$

This result enables pointwise confidence intervals for the CATE — a property that meta-learners generally lack without additional assumptions.

### 3.4 Conceptual Diagram

```
CAUSAL FOREST: HONEST SPLITTING

Training Data (N units)
    │
    ├─── Subsample 1 ──────────────────── Subsample 2 ── ... ── Subsample B
    │        │                                 │                      │
    │   ┌────┴────┐                       ┌────┴────┐           ┌────┴────┐
    │   │ I_split │  │ I_est │            │ I_split │ │ I_est │ │  ...    │
    │   └────┬────┘  └───┬───┘            └────┬────┘ └───┬───┘ └────────┘
    │        │            │                    │           │
    │   Build tree   Estimate τ           Build tree  Estimate τ
    │   (causal      in leaves            (causal     in leaves
    │    splits)     (honest)              splits)    (honest)
    │        │            │                    │           │
    │        └─────┬──────┘                    └─────┬─────┘
    │           Tree 1                            Tree 2          ...
    │              │                                │
    └──────────────┴────────────────────────────────┴──────────────┘
                                    │
                           Average across B trees
                                    │
                            τ̂(x) ± confidence interval
```

### 3.5 Generalized Random Forests (GRF)

Athey, Tibshirani, and Wager (2019) generalized the causal forest idea into the GRF framework, which solves *local moment equations* at each covariate value $\mathbf{x}$:

$$E[\psi_\theta(Y_i, W_i, \mathbf{X}_i) \mid \mathbf{X}_i = \mathbf{x}] = 0$$

The forest defines an adaptive kernel — a set of weights $\alpha_i(\mathbf{x})$ based on how often unit $i$ falls in the same leaf as $\mathbf{x}$ — and solves the weighted moment condition:

$$\sum_{i=1}^{N} \alpha_i(\mathbf{x}) \psi_{\hat{\theta}(\mathbf{x})}(Y_i, W_i, \mathbf{X}_i) = 0$$

For CATE estimation, $\psi$ is the score of a partially linear model. But GRF extends naturally to quantile treatment effects, local average treatment effects (IV settings), and heterogeneous policy effects.

### 3.6 Causal Forests vs. DML

| Property | Double/Debiased ML (Part 4) | Causal Forest |
|----------|----------------------------|---------------|
| **CATE form** | Parametric: $\tau(\mathbf{x}) = \mathbf{x}^\top \beta$ | Fully nonparametric |
| **Base learner** | Any ML (for nuisance) | Modified random forest |
| **Inference** | Standard (asymptotic normality of $\hat{\beta}$) | Pointwise CIs from honesty |
| **Interpretability** | Linear coefficients on $\mathbf{x}$ | Variable importance, partial dependence |
| **Best for** | Low-dimensional effect modification | High-dimensional, nonlinear heterogeneity |

**DML assumes a parametric form for the CATE and uses ML only for nuisance estimation. Causal forests estimate the CATE nonparametrically, using ML as the primary estimator itself.** The choice depends on whether the researcher has a structural prior on the form of heterogeneity.

---

## 4. Causal Discovery — Learning the Graph

### 4.1 The Inverted Problem

Everything in Parts 1-4 assumed a known causal graph (DAG). The graph was drawn by the researcher based on domain knowledge, and all identification and estimation followed from that graph. **Causal discovery inverts this: given data, what can we learn about the causal graph?**

The motivation is practical. In many domains — genomics, climate science, neuroscience — the causal structure is unknown, the number of variables is large, and human expertise cannot specify every edge. Discovery algorithms provide a principled starting point.

But the inversion comes with a fundamental limitation, stated upfront:

> **Theorem 5.1 (Markov Equivalence)**: From observational data alone, under faithfulness and causal sufficiency, the best we can identify is the *Markov equivalence class* — the set of all DAGs that encode the same conditional independence relations. This equivalence class is represented by a Completed Partially Directed Acyclic Graph (CPDAG), which may leave some edge directions undetermined.

Two DAGs are Markov equivalent if and only if they have the same skeleton (undirected edges) and the same *v-structures* (collider patterns $X \to Z \gets Y$ where $X$ and $Y$ are non-adjacent).

### 4.2 Constraint-Based Algorithms: PC and FCI

#### The PC Algorithm

The PC algorithm (Spirtes, Glymour, and Scheines, 2000) is the foundational constraint-based method. It exploits a direct consequence of the causal Markov condition: in a faithful DAG, $X \perp\!\!\!\perp Y \mid \mathbf{Z}$ if and only if $\mathbf{Z}$ blocks every path between $X$ and $Y$.

**The PC algorithm works backward: start with a fully connected undirected graph and remove edges wherever a conditional independence is found.**

```
PC ALGORITHM — STEP BY STEP

Step 0: Start with complete undirected graph on p variables
        (every pair connected)

        A ─── B ─── C ─── D
        │ ╲       ╱ │
        │   ╲   ╱   │
        E ─── F ─── G

Step 1: Test marginal independences (conditioning set size = 0)
        Remove edge (X,Y) if X ⊥⊥ Y
        → Sparse graph

Step 2: Test conditional independences (size = 1, 2, ...)
        For each remaining edge (X,Y):
          For each subset Z of adjacents of X (or Y):
            If X ⊥⊥ Y | Z, remove edge, record Z as "sepset"
        → Skeleton

Step 3: Orient v-structures (colliders)
        For each triple X — Z — Y where X and Y are
        non-adjacent:
          If Z ∉ sepset(X,Y), orient as X → Z ← Y
        → Partially oriented graph

Step 4: Apply orientation rules (Meek rules)
        Propagate orientations to avoid new v-structures
        and directed cycles
        → CPDAG (output)
```

The PC algorithm assumes:

- **Causal sufficiency**: no latent common causes (all confounders measured)
- **Faithfulness**: all observed independences reflect structural separations (no exact cancellations)
- **Correct conditional independence tests**: the tests have appropriate size and power

**The output is a CPDAG, not a single DAG.** Undirected edges in the CPDAG represent directions that cannot be determined from the data alone — they belong to edges where both orientations yield Markov-equivalent DAGs.

The computational cost depends on the maximum size of the conditioning sets tested. Let $q$ denote the maximum neighborhood size in the true graph. The PC algorithm runs at most $O(p^{q+2})$ conditional independence tests, where $p$ is the number of variables. For sparse graphs ($q \ll p$), this is tractable even for hundreds of variables. For dense graphs, the cost is prohibitive.

**Practical consideration**: The choice of conditional independence test matters. For continuous data, partial correlation tests (Fisher's $z$) are standard under Gaussianity. For mixed or non-Gaussian data, kernel-based tests (HSIC) or mutual-information-based tests provide more flexibility at higher computational cost. The significance level $\alpha$ for CI tests acts as a tuning parameter: lower $\alpha$ removes fewer edges, yielding denser graphs with fewer false negatives but more false positives.

#### FCI: Allowing Latent Confounders

The Fast Causal Inference (FCI) algorithm (Spirtes et al., 2000) relaxes causal sufficiency — the assumption that all common causes are measured. **When latent common causes exist, the PC algorithm can produce incorrect edge orientations because unmeasured confounders create spurious dependencies that mimic direct effects.**

FCI addresses this by introducing additional conditioning and orientation rules that account for the possibility of hidden variables. The algorithm begins similarly to PC (skeleton discovery via CI tests) but adds a second pass to detect potential latent confounders through "discriminating paths."

FCI outputs a *Partial Ancestral Graph* (PAG) with four edge types:

| Edge Mark | Meaning |
|-----------|---------|
| $X \to Y$ | $X$ is an ancestor of $Y$ (causal direction identified) |
| $X \leftrightarrow Y$ | Latent common cause of $X$ and $Y$ (bidirected edge) |
| $X \circ\!\!\to Y$ | $X$ is either a cause of $Y$ or shares a latent common cause with $Y$ |
| $X \circ\!\!\!-\!\!\circ Y$ | Both direction and latent-cause status undetermined |

The circle mark ($\circ$) represents genuine ambiguity: the data cannot distinguish between the possible interpretations, and the PAG explicitly represents this uncertainty.

- **Strength**: Handles latent confounding — essential for observational studies where causal sufficiency is implausible. Sound and complete for the class of ancestral graphs.
- **Weakness**: More conservative; many edges left undetermined, reducing the actionable information. Higher computational cost from additional conditioning rounds. Interpretation of PAG edge marks requires care.
- **When to prefer FCI over PC**: Whenever you suspect unmeasured common causes — which is the default in observational science.

### 4.3 Score-Based Algorithms: GES and NOTEARS

#### Greedy Equivalence Search (GES)

Score-based methods treat discovery as a model selection problem. Define a score function $S(G; \mathbf{D})$ over DAGs $G$ given data $\mathbf{D}$, and search the DAG space for the highest-scoring structure.

GES (Chickering, 2002) uses the BIC score and searches in two phases:

1. **Forward phase**: Start from the empty graph. Greedily add the single edge that most improves BIC. Repeat until no addition improves the score.
2. **Backward phase**: Greedily remove edges that improve BIC. Repeat until no removal improves the score.

Under the assumptions of causal sufficiency, faithfulness, and correct model specification, GES recovers the true CPDAG in the large-sample limit.

- **Strength**: Does not require choosing conditional independence tests or significance levels; naturally handles multiple edges simultaneously
- **Weakness**: Greedy search is exponential in the worst case; can get trapped in local optima for finite samples

#### NOTEARS: Acyclicity as a Continuous Constraint

Zheng et al. (2018) reformulated causal discovery as a continuous optimization problem by expressing the acyclicity constraint algebraically.

For a $p \times p$ weighted adjacency matrix $\mathbf{A}$, the graph is a DAG if and only if:

$$h(\mathbf{A}) = \text{tr}(e^{\mathbf{A} \circ \mathbf{A}}) - p = 0$$

where $\circ$ denotes element-wise product and $e^{(\cdot)}$ is the matrix exponential. This smooth equality constraint enables gradient-based optimization:

$$\min_{\mathbf{A}} \| \mathbf{Y} - \mathbf{Y}\mathbf{A} \|_F^2 + \lambda \|\mathbf{A}\|_1 \quad \text{subject to} \quad h(\mathbf{A}) = 0$$

The $L_1$ penalty encourages sparsity. The continuous acyclicity constraint $h(\mathbf{A}) = 0$ replaces the combinatorial search over DAG structures.

- **Strength**: Scales to hundreds of variables; leverages modern optimization infrastructure (autodiff, GPU)
- **Weakness**: Assumes linear relationships with Gaussian noise in the basic formulation (extensions exist for nonlinear and non-Gaussian cases); the acyclicity constraint surface is non-convex, so convergence to the global optimum is not guaranteed

### 4.4 Functional Approach: LiNGAM

The Linear Non-Gaussian Acyclic Model (Shimizu et al., 2006) exploits a remarkable identifiability result:

> **Theorem 5.2 (LiNGAM Identifiability)**: If the true data-generating process is a linear DAG with non-Gaussian, independent exogenous noise terms, then the DAG is *uniquely* identifiable from the data distribution — not merely up to Markov equivalence.

The structural model is:

$$\mathbf{X} = \mathbf{B}\mathbf{X} + \mathbf{E}, \quad E_j \perp\!\!\!\perp E_k \text{ for } j \neq k, \quad E_j \text{ non-Gaussian}$$

where $\mathbf{B}$ is a strictly lower-triangular matrix (after appropriate permutation of variables) encoding the DAG structure. The key is that Independent Component Analysis (ICA) can recover $\mathbf{B}$ because non-Gaussian independent sources are uniquely separable.

- **Strength**: Recovers the *full DAG* (not just equivalence class) — a strictly stronger result than PC or GES
- **Weakness**: Requires linearity and non-Gaussianity; violation of either assumption invalidates the identification
- **Best for**: Domains where non-Gaussianity is plausible (e.g., financial data, neural recordings with heavy-tailed distributions)

### 4.5 Discovery Algorithm Comparison

| Algorithm | Type | Key Assumption | Output | Scalability | Latent Confounders? |
|-----------|------|----------------|--------|:-----------:|:-------------------:|
| **PC** | Constraint | Sufficiency, faithfulness | CPDAG | $O(p^q)$, $q$ = max neighborhood | No |
| **FCI** | Constraint | Faithfulness (no sufficiency) | PAG | Slower than PC | Yes |
| **GES** | Score | Sufficiency, faithfulness, BIC | CPDAG | Moderate (greedy) | No |
| **NOTEARS** | Continuous opt. | Linearity (basic), acyclicity | DAG | Good ($O(p^3)$ per step) | No |
| **LiNGAM** | Functional | Linearity, non-Gaussianity | Full DAG | Good (ICA-based) | No |

> **Pitfall**: Discovery algorithms are sensitive to sample size, distributional assumptions, and the faithfulness condition. A graph learned from 500 samples should be treated as a hypothesis, not as ground truth. Edges that appear in only 60% of bootstrap resamples deserve skepticism, not confidence intervals.

### 4.6 The Fundamental Limitation

**Observational data, even infinite, cannot distinguish between Markov-equivalent DAGs without additional assumptions (functional form, non-Gaussianity, or interventional data).** This is not a limitation of any particular algorithm — it is a mathematical impossibility.

Consider three variables $X$, $Y$, $Z$ with the chain structure $X \to Y \to Z$. The DAGs $X \to Y \to Z$, $X \gets Y \gets Z$, and $X \gets Y \to Z$ all encode the same conditional independence: $X \perp\!\!\!\perp Z \mid Y$. From observational data alone, these three DAGs are indistinguishable (they form an equivalence class).

Only a v-structure ($X \to Y \gets Z$) is identifiable, because it encodes a *different* independence pattern: $X \perp\!\!\!\perp Z$ but $X \not\!\perp\!\!\!\perp Z \mid Y$.

```
MARKOV EQUIVALENCE — SAME DATA, DIFFERENT DAGS

Equivalence class 1:              Equivalence class 2:
All encode X ⊥⊥ Z | Y            Encodes X ⊥⊥ Z but X ⊬⊥ Z | Y

  X → Y → Z                        X → Y ← Z
  X ← Y ← Z                       (unique — v-structure)
  X ← Y → Z

  CPDAG: X — Y — Z                 CPDAG: X → Y ← Z
  (directions unknown)              (directions known)
```

### 4.7 Practical Decision Guide

Given the variety of discovery algorithms, the practitioner needs a decision heuristic. **The right algorithm depends on three questions: do you assume causal sufficiency, is the data plausibly Gaussian, and how many variables are there?**

```
CHOOSING A DISCOVERY ALGORITHM

                     Latent confounders possible?
                        /              \
                      Yes               No
                       │                 │
                      FCI          Data Gaussian?
                                   /           \
                                 Yes            No
                                  │              │
                            p < 100?        LiNGAM
                            /      \        (full DAG)
                          Yes       No
                           │         │
                       PC or GES   NOTEARS
                       (CPDAG)     (scalable)
```

When in doubt, run multiple algorithms and look for consensus edges. An edge that appears in PC, GES, and LiNGAM outputs simultaneously is far more credible than one appearing in only a single method's output.

---

## 5. From Discovery to Inference — The Full Workflow

### 5.1 The End-to-End Pipeline

Parts 0-5 of this series now assemble into a complete causal analysis pipeline with two possible entry points:

```
FULL CAUSAL WORKFLOW

Entry A: Expert Knowledge          Entry B: Data-Driven Discovery
         │                                   │
    Draw DAG from                     Run PC / GES / LiNGAM
    domain expertise                  on observational data
         │                                   │
         └──────────┬────────────────────────┘
                    │
                    ▼
              DAG (possibly CPDAG)
                    │
              ┌─────┴─────┐
              │  IDENTIFY  │  Find valid adjustment set
              │            │  (backdoor, frontdoor, IV)
              └─────┬──────┘
                    │
              ┌─────┴─────┐
              │  ESTIMATE  │  IPW, AIPW, DML, meta-learner,
              │            │  causal forest
              └─────┬──────┘
                    │
              ┌─────┴─────┐
              │   REFUTE   │  Sensitivity analysis, placebo,
              │            │  bootstrap
              └─────┬──────┘
                    │
                    ▼
         Causal estimate + uncertainty
         τ̂ (ATE) or τ̂(x) (CATE) + CI
```

### 5.2 When to Use Discovery vs. Expert DAG

The choice between Entry A and Entry B is itself a causal modeling decision.

| Criterion | Expert DAG (Entry A) | Discovery (Entry B) |
|-----------|---------------------|---------------------|
| **Domain knowledge** | Strong — can specify edges confidently | Weak — many plausible structures |
| **Number of variables** | Small ($p < 20$) | Large ($p > 50$) |
| **Risk tolerance** | Low — wrong graph leads to wrong policy | Higher — discovery is hypothesis generation |
| **Sample size** | Any | Large ($N \gg p^2$ for reliable tests) |
| **Latent confounders** | Can mark explicitly in DAG | PC cannot handle; FCI can but at cost |
| **Recommended** | Clinical trials, policy evaluation | Genomics, neuroscience, exploratory analysis |

### 5.3 Discovery as Hypothesis Generation

**Causal discovery is most valuable not as a replacement for domain knowledge but as a structured hypothesis generator.** The recommended workflow:

1. **Run discovery** on the data to produce a candidate CPDAG or PAG
2. **Consult domain experts** to resolve undirected edges, reject implausible edges, and add known edges the algorithm missed
3. **Refine the DAG** by combining algorithmic output with expert input
4. **Identify and estimate** using the refined DAG, following the standard pipeline
5. **Sensitivity analysis**: test how conclusions change if specific discovered edges are reversed or removed

This hybrid approach — algorithm proposes, expert disposes — combines the scalability of discovery with the correctness guarantees of expert knowledge.

### 5.4 Stability and Robustness Checks for Discovered Graphs

A single run of PC or GES on a dataset produces one graph. **A responsible practitioner never reports this graph without assessing its stability.** Standard robustness checks include:

- **Bootstrap stability**: Run the discovery algorithm on $B$ bootstrap resamples. Report each edge with its frequency of appearance. Edges appearing in fewer than 50% of resamples are unreliable.
- **Sensitivity to $\alpha$**: For constraint-based methods, vary the CI test significance level. Edges that appear only at $\alpha = 0.1$ but vanish at $\alpha = 0.01$ are weakly supported.
- **Cross-algorithm agreement**: Run PC, GES, and (if applicable) LiNGAM on the same data. Edges that appear across all three methods deserve higher confidence than those appearing in only one.
- **Interventional validation**: If any interventional data exist (e.g., from gene knockouts, A/B tests, or natural experiments), check whether the discovered graph is consistent with the known interventional effects.

### 5.5 Connection to the Agent Pipeline

Part 7 explores how an LLM agent can partially automate the hybrid workflow of Section 5.3. The agent:

1. Runs causal discovery algorithms as tools
2. Uses the LLM's compressed domain knowledge to adjudicate undirected edges — functioning as a scalable (if imperfect) substitute for human expert consultation
3. Proposes identification strategies based on the refined DAG
4. Executes estimation and refutation steps

The CATE estimation methods of Section 2 and the causal forest of Section 3 become callable tools in this agent architecture. The discovery algorithms of Section 4 provide the graph-construction step. **This post provides the statistical machinery; Part 7 provides the orchestration.**

---

## Summary

This post developed two extensions that move causal inference beyond a single average number.

1. **The CATE $\tau(\mathbf{x})$ captures treatment effect heterogeneity — the variation in causal effects across subpopulations.** Meta-learners (S, T, X, R, DR) decompose CATE estimation into standard supervised learning subproblems, differing in how they handle nuisance estimation, group imbalance, and double robustness. The DR-learner, which regresses AIPW pseudo-outcomes on covariates, is the recommended default.

2. **Causal forests modify random forests to split on treatment effect heterogeneity rather than outcome prediction, with honest estimation enabling valid confidence intervals.** The GRF framework unifies causal forests with other local moment estimation problems.

3. **Causal discovery algorithms — constraint-based (PC, FCI), score-based (GES, NOTEARS), and functional (LiNGAM) — learn the causal graph from data, but face a fundamental limitation: observational data identify only the Markov equivalence class, not a unique DAG.** Discovery is best used as hypothesis generation, not as ground truth.

4. **The full pipeline runs: Discover (or Draw) $\to$ Identify $\to$ Estimate $\to$ Refute.** The choice between expert DAG and algorithmic discovery depends on domain knowledge, variable count, and risk tolerance.

Part 6 translates this entire pipeline — from meta-learners to causal forests to discovery — into working Python code using DoWhy, EconML, and causal-learn.

---

## References

- Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized random forests. *Annals of Statistics*, 47(2), 1148-1178.
- Chickering, D. M. (2002). Optimal structure identification with greedy search. *Journal of Machine Learning Research*, 3, 507-554.
- Kennedy, E. H. (2023). Towards optimal doubly robust estimation of heterogeneous causal effects. *Electronic Journal of Statistics*, 17(2), 3008-3049.
- Kunzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019). Metalearners for estimating heterogeneous treatment effects using machine learning. *Proceedings of the National Academy of Sciences*, 116(10), 4156-4165.
- Nie, X., & Wager, S. (2021). Quasi-oracle estimation of heterogeneous treatment effects. *Biometrika*, 108(2), 299-319.
- Robinson, P. M. (1988). Root-N-consistent semiparametric regression. *Econometrica*, 56(4), 931-954.
- Shimizu, S., Hoyer, P. O., Hyvarinen, A., & Kerminen, A. (2006). A linear non-Gaussian acyclic model for causal discovery. *Journal of Machine Learning Research*, 7, 2003-2030.
- Spirtes, P., Glymour, C. N., & Scheines, R. (2000). *Causation, Prediction, and Search* (2nd ed.). MIT Press.
- Wager, S., & Athey, S. (2018). Estimation and inference of heterogeneous treatment effects using random forests. *Journal of the American Statistical Association*, 113(523), 1228-1242.
- Zheng, X., Aragam, B., Ravikumar, P., & Xing, E. P. (2018). DAGs with NO TEARS: Continuous optimization for structure learning. *Advances in Neural Information Processing Systems*, 31, 9472-9483.

---
