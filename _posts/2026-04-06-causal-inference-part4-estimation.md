---
title: "Causal Inference Part 4: Estimation — From Estimand to Estimate"
date: 2026-04-06 13:59:00 +0900
categories: [AI & ML, ML Theory]
tags: [causal-inference, estimation, matching, ipw, doubly-robust, dml]
math: true
---

**Causal Inference — From First Principles to Automated Reasoning**

*This is Part 4 of an 8-part series.*

- **Part 0:** [Beyond Correlation — Why Causal Inference?](/posts/causal-inference-part0-beyond-correlation/)
- **Part 1:** [Potential Outcomes and the Fundamental Problem](/posts/causal-inference-part1-potential-outcomes/)
- **Part 2:** [Structural Causal Models and DAGs](/posts/causal-inference-part2-structural-causal-models/)
- **Part 3:** [Identification — From Design to Estimand](/posts/causal-inference-part3-identification/)
- **Part 4:** Estimation — From Estimand to Estimate (this post)
- **Part 5:** [Beyond the Average — Heterogeneous Effects and Causal Discovery](/posts/causal-inference-part5-heterogeneous-effects/)
- **Part 6:** [End-to-End Practice — The Causal Pipeline in Code](/posts/causal-inference-part6-causal-pipeline/)
- **Part 7:** [The Causal Agent — LLM-Driven Causal Reasoning](/posts/causal-inference-part7-causal-agent/)

---

## Hook: The Gap Between Logic and Statistics

Identification tells you *what* to estimate. Estimation tells you *how*. The first is a logic problem; the second is a statistical problem. Conflating them is the most common mistake in applied causal inference.

Part 3 ended with a clean estimand — a mathematical expression that, under stated assumptions, equals the causal effect. But an estimand is not a number. It is a population-level quantity written in terms of distributions we do not observe. To convert it into a point estimate with a confidence interval, we need an *estimator*: a function that maps a finite dataset to a number.

This conversion is not trivial. The choice of estimator determines whether your estimate is consistent, efficient, and robust to model misspecification. A correct estimand paired with a poor estimator yields misleading conclusions — wide confidence intervals, regularization bias, or sensitivity to functional form.

**This post covers six estimation strategies — from classical matching to machine-learning-powered debiased estimation — with particular emphasis on the two estimators that underpin the AlphaFold impact study: Poisson Pseudo-Maximum Likelihood (PPML) for count outcomes and OLS with fixed effects for binary outcomes.**

---

## Notation Recap

This post inherits the notation convention from the series. Key symbols used throughout:

| Symbol | Meaning |
|--------|---------|
| $i = 1, \dots, N$ | Unit index; $N$ = sample size |
| $t = 1, \dots, T$ | Time index |
| $W_i \in \{0, 1\}$ | Treatment indicator |
| $Y_i$ | Observed outcome; $Y_i(w)$ = potential outcome under treatment $w$ |
| $\mathbf{X}_i \in \mathbb{R}^p$ | Pre-treatment covariate vector |
| $\tau = E[Y_i(1) - Y_i(0)]$ | Average Treatment Effect (ATE) |
| $\tau_{\text{ATT}}$ | Average Treatment Effect on the Treated |
| $e(\mathbf{x}) = P(W_i=1 \mid \mathbf{X}_i=\mathbf{x})$ | Propensity score |
| $\mu_w(\mathbf{x}) = E[Y_i \mid W_i=w, \mathbf{X}_i=\mathbf{x}]$ | Outcome regression function |
| $\alpha_i$, $\gamma_t$ | Unit and time fixed effects |
| $\delta$ | DiD coefficient (treatment effect in panel models) |
| $\hat{\tau}$ | Estimated treatment effect |
| $K$ | Number of folds in cross-fitting |

---

## 1. Matching and Propensity Score Methods

### 1.1 The Core Idea

**Matching makes the treatment group and the control group look alike on observed covariates, so that the remaining difference in outcomes can be attributed to treatment.**

Suppose we have identified $\tau_{\text{ATT}}$ via the backdoor criterion, so the estimand is:

$$\tau_{\text{ATT}} = E\left[Y_i(1) - Y_i(0) \mid W_i = 1\right]$$

Under unconfoundedness (Assumption 1.2 from Part 1), for each treated unit $i$ with covariates $\mathbf{X}\_{i}$, we seek a control unit $j$ with $\mathbf{X}\_{j} \approx \mathbf{X}\_{i}$. The matched difference $Y\_{i} - Y\_{j}$ is an unbiased estimate of the individual treatment effect, and the average across matched pairs estimates $\tau\_{\text{ATT}}$.

The problem is dimensionality. When $\mathbf{X}_i \in \mathbb{R}^p$ with $p > 5$, exact matching becomes infeasible — the number of possible covariate cells grows exponentially. This is the *curse of dimensionality* in matching.

### 1.2 The Propensity Score

Rosenbaum and Rubin (1983) proved a dimension-reduction result that transformed applied causal inference.

> **Theorem 1.1 (Propensity Score Theorem)**: If unconfoundedness holds given $\mathbf{X}_i$, then it also holds given the scalar propensity score $e(\mathbf{X}_i)$ alone:
>
> $$\{Y_i(0), Y_i(1)\} \perp\!\!\!\perp W_i \mid \mathbf{X}_i \implies \{Y_i(0), Y_i(1)\} \perp\!\!\!\perp W_i \mid e(\mathbf{X}_i)$$
>
> where $e(\mathbf{x}) = P(W_i = 1 \mid \mathbf{X}_i = \mathbf{x})$.

**This reduces a $p$-dimensional matching problem to a one-dimensional matching problem.** Instead of matching on all of $\mathbf{X}$, we match on the scalar $e(\mathbf{X})$.

In practice, the propensity score is estimated — typically by logistic regression, but any classifier that produces calibrated probabilities suffices (gradient-boosted trees, random forests with calibration, etc.). The estimated $\hat{e}(\mathbf{X}_i)$ replaces the unknown $e(\mathbf{X}_i)$.

### 1.3 Propensity Score Matching (PSM)

**PSM pairs each treated unit with its nearest control in propensity score space.**

The algorithm:

1. Estimate $\hat{e}(\mathbf{X}_i)$ for all units.
2. For each treated unit $i$, find the control unit $j$ that minimizes $|\hat{e}(\mathbf{X}_i) - \hat{e}(\mathbf{X}_j)|$.
3. Estimate $\hat{\tau}\_{\text{ATT}} = \frac{1}{N\_{1}} \sum\_{i: W\_i=1} (Y\_{i} - Y\_{j(i)})$.

Variants include caliper matching (discard treated units without a close match), $k$:1 matching (use $k$ controls per treated), and matching with replacement (a control can serve multiple treated units).

**After matching, verify covariate balance.** The standardized mean difference (SMD) for each covariate should be below 0.1. If balance is poor, the propensity score model is misspecified.

The SMD for covariate $X^{(j)}$ is:

$$\text{SMD}^{(j)} = \frac{\bar{X}^{(j)}_{\text{treated}} - \bar{X}^{(j)}_{\text{matched control}}}{\sqrt{(s^2_{\text{treated}} + s^2_{\text{control}})/2}}$$

A *love plot* displays SMD values before and after matching, providing a visual diagnostic of whether matching achieved balance.

```
LOVE PLOT (Balance Diagnostic)

                Before matching          After matching
Covariate       |                        |
                |                        |
  X₁    ───────|────●                   |──●
  X₂    ───────|───────────●            |─●
  X₃    ───────|──●                     |●
  X₄    ───────|─────────────────●      |───●
  X₅    ───────|────────●               |●
                |                        |
          ─────|────────────────────    ─|────────────
              0.0  0.1  0.2  0.3       0.0  0.1
                  │                      │
                  SMD                    SMD
                  (poor balance)         (good: all < 0.1)
```

### 1.4 Beyond PSM

| Method | Mechanism | Advantage | Limitation |
|--------|-----------|-----------|------------|
| **Subclassification** | Stratify on $\hat{e}(\mathbf{x})$ quintiles; estimate within-stratum effects | Uses all data; simple | Residual bias within strata |
| **Kernel matching** | Weight all controls by kernel distance to treated unit's $\hat{e}$ | Smooth; uses all controls | Bandwidth selection |
| **CEM** | Coarsen covariates; exact-match on coarsened cells | Transparent; no model | Requires few, discrete covariates |

> **Pitfall**: PSM does not solve unobserved confounding. It implements unconfoundedness (Assumption 1.2 from Part 1) — it does not relax it. If an unmeasured variable $U$ confounds the treatment-outcome relationship, balancing on $e(\mathbf{X})$ leaves the bias from $U$ intact. PSM is an estimation strategy, not an identification strategy.

---

## 2. Inverse Probability Weighting (IPW)

### 2.1 The Horvitz-Thompson Idea

**IPW reweights the observed data to create a pseudo-population in which treatment is independent of covariates.**

Instead of finding a matched partner for each treated unit, IPW upweights control units that look like treated units and vice versa. The estimator for the ATE is:

$$\hat{\tau}_{\text{IPW}} = \frac{1}{N} \sum_{i=1}^N \left[ \frac{W_i Y_i}{\hat{e}(\mathbf{X}_i)} - \frac{(1 - W_i) Y_i}{1 - \hat{e}(\mathbf{X}_i)} \right]$$

**Verbal interpretation**: each treated unit $i$ receives weight $1/\hat{e}(\mathbf{X}_i)$. A treated unit with a low propensity score — one that "should not" have been treated based on covariates — gets a large weight, because it is informative about what would happen if we imposed treatment on a unit that would normally be untreated. Symmetrically, untreated units with high propensity scores get large weights.

The logic traces back to Horvitz and Thompson (1952), who developed inverse-probability-weighted estimators for survey sampling. The connection to causal inference was formalized by Robins, Rotnitzky, and Zhao (1994).

**Why IPW works**: under unconfoundedness, weighting by the inverse propensity score creates a pseudo-population where treatment assignment is independent of $\mathbf{X}$. Formally:

$$E\left[\frac{W_i Y_i}{e(\mathbf{X}_i)}\right] = E\left[\frac{W_i Y_i(1)}{e(\mathbf{X}_i)}\right] = E\left[E\left[\frac{W_i}{e(\mathbf{X}_i)} \mid \mathbf{X}_i\right] Y_i(1)\right] = E[Y_i(1)]$$

The first equality uses the consistency assumption ($Y_i = Y_i(1)$ when $W_i = 1$). The second uses iterated expectations. The third uses $E[W_i \mid \mathbf{X}_i] = e(\mathbf{X}_i)$, so $E[W_i / e(\mathbf{X}_i) \mid \mathbf{X}_i] = 1$. The same argument applied to the control arm gives $E[(1-W_i)Y_i / (1-e(\mathbf{X}_i))] = E[Y_i(0)]$, and the difference is the ATE.

### 2.2 The Reweighting Visual

```
OBSERVED POPULATION                   PSEUDO-POPULATION (after IPW)
                                      
Treated  (W=1):                       Treated  (W=1):
  ● high e(x) → weight 1/0.9 = 1.1     ● weight ≈ 1
  ● high e(x) → weight 1/0.8 = 1.25    ● weight ≈ 1
  ● low  e(x) → weight 1/0.2 = 5.0     ●●●●● amplified
  ● low  e(x) → weight 1/0.3 = 3.3     ●●● amplified
                                      
Control  (W=0):                       Control  (W=0):
  ○ low  e(x) → weight 1/0.8 = 1.25    ○ weight ≈ 1
  ○ low  e(x) → weight 1/0.9 = 1.1     ○ weight ≈ 1
  ○ high e(x) → weight 1/0.2 = 5.0     ○○○○○ amplified
  ○ high e(x) → weight 1/0.3 = 3.3     ○○○ amplified
                                      
Result: treatment correlated           Result: treatment independent
        with covariates                         of covariates
```

**IPW creates balance by inflating the representation of underrepresented covariate regions in each treatment arm.** The pseudo-population mimics what we would observe under randomization.

### 2.3 The Extreme Weight Problem

**When $\hat{e}(\mathbf{X}_i)$ is close to 0 or 1, the weights $1/\hat{e}$ or $1/(1-\hat{e})$ explode.** A single unit with $\hat{e} = 0.01$ receives weight 100, dominating the entire estimate. This inflates variance and makes the estimator fragile.

Three standard remedies:

1. **Trimming**: drop units with $\hat{e}(\mathbf{X}_i) < \epsilon$ or $\hat{e}(\mathbf{X}_i) > 1 - \epsilon$ (typically $\epsilon = 0.01$ or $0.05$). This changes the estimand from ATE to the ATE among the overlap population.

2. **Stabilized weights**: replace $1/\hat{e}(\mathbf{X}_i)$ with $P(W=1)/\hat{e}(\mathbf{X}_i)$ — multiply by the marginal treatment probability. This does not change the point estimate but reduces variance.

3. **Overlap weighting** (Li, Morgan, and Zaslavsky, 2018): use weights $\hat{e}(\mathbf{X}_i)(1 - \hat{e}(\mathbf{X}_i))$ for both groups. This targets the ATE in the overlap population and produces the most stable estimates.

> **Assumption (Overlap / Positivity)**: For IPW to be valid, the propensity score must satisfy $0 < e(\mathbf{x}) < 1$ for all $\mathbf{x}$ in the covariate support. Violations indicate structural non-overlap — there exist covariate regions where treatment (or control) is deterministic, and the causal effect is not identified for those units.

---

## 3. Doubly Robust / AIPW

### 3.1 Two Models, One Insurance Policy

**The doubly robust (DR) estimator combines a propensity score model $\hat{e}(\mathbf{X}_i)$ and an outcome model $\hat{\mu}_w(\mathbf{X}_i)$, and is consistent if *either one* is correctly specified.**

The augmented inverse probability weighted (AIPW) estimator is:

$$\hat{\tau}_{\text{DR}} = \frac{1}{N} \sum_{i=1}^N \left[ \hat{\mu}_1(\mathbf{X}_i) - \hat{\mu}_0(\mathbf{X}_i) + \frac{W_i (Y_i - \hat{\mu}_1(\mathbf{X}_i))}{\hat{e}(\mathbf{X}_i)} - \frac{(1-W_i)(Y_i - \hat{\mu}_0(\mathbf{X}_i))}{1 - \hat{e}(\mathbf{X}_i)} \right]$$

**Verbal interpretation**: start with the outcome regression estimate $\hat{\mu}_1(\mathbf{X}_i) - \hat{\mu}_0(\mathbf{X}_i)$, then add an IPW-weighted correction for the residual error $(Y_i - \hat{\mu}_w(\mathbf{X}_i))$. If the outcome model is perfect, the residuals are zero and the correction vanishes. If the outcome model is wrong but the propensity model is correct, the correction removes the bias. Either way, the estimator is consistent.

### 3.2 Why "Doubly Robust"?

Let $\hat{e}$ and $\hat{\mu}_w$ denote estimated nuisance functions. The DR estimator satisfies:

$$\hat{\tau}_{\text{DR}} \xrightarrow{p} \tau \quad \text{if } \hat{e} \xrightarrow{p} e \text{ or } \hat{\mu}_w \xrightarrow{p} \mu_w$$

**You get two chances to get the model right.** In practice, both models will be somewhat wrong, but the bias of the DR estimator is *multiplicative* in the errors of the two models. If both models are slightly wrong, the bias is second-order small — much smaller than the bias of either IPW or outcome regression alone.

### 3.3 Semiparametric Efficiency

The DR estimator is not just robust — it is *efficient*. Robins, Rotnitzky, and Zhao (1994) showed that AIPW achieves the semiparametric efficiency bound when both nuisance models are consistently estimated. This means:

**No regular estimator can have lower asymptotic variance than AIPW when both models converge to the truth.**

The influence function of the AIPW estimator is:

$$\psi_i = \hat{\mu}_1(\mathbf{X}_i) - \hat{\mu}_0(\mathbf{X}_i) - \tau + \frac{W_i (Y_i - \hat{\mu}_1(\mathbf{X}_i))}{\hat{e}(\mathbf{X}_i)} - \frac{(1-W_i)(Y_i - \hat{\mu}_0(\mathbf{X}_i))}{1 - \hat{e}(\mathbf{X}_i)}$$

This influence function is *Neyman orthogonal* — its derivative with respect to the nuisance parameters is zero at the truth. This property is the foundation of Double/Debiased Machine Learning, covered in Section 6.

### 3.4 The Bridge to Modern Methods

The progression from matching to IPW to AIPW follows a clear logic:

```
ESTIMATION METHOD EVOLUTION

Matching (Sec 1)                 IPW (Sec 2)
  Uses: outcome model only         Uses: propensity model only
  Risk: model misspec.             Risk: extreme weights
        │                                │
        └──────────┐    ┌────────────────┘
                   ▼    ▼
               AIPW / DR (Sec 3)
                 Uses: both models
                 Risk: both wrong
                       │
                       ▼
                 DML (Sec 6)
                   Uses: both models + ML
                   Risk: both wrong + not cross-fit
```

**Each step adds robustness. DML (Section 6) takes the final step: allowing the nuisance models to be estimated by arbitrary machine learning methods while preserving valid inference on $\hat{\tau}$.**

---

## 4. PPML for Count Outcomes

### 4.1 Why Not Log-OLS?

**This section directly supports Strategy I of the AlphaFold impact study, where the outcome is a count variable (e.g., number of PDB deposits, citation counts).**

A natural first instinct for count data is to log-transform the outcome and run OLS:

$$\log(Y_i) = \mathbf{X}_i'\boldsymbol{\beta} + \epsilon_i$$

This approach has three problems, any one of which can invalidate inference.

**Problem 1: Jensen's inequality.** The conditional expectation of the log is not the log of the conditional expectation:

$$E[\log Y_i \mid \mathbf{X}_i] \neq \log E[Y_i \mid \mathbf{X}_i]$$

When we exponentiate the OLS prediction $\exp(\mathbf{X}_i'\hat{\boldsymbol{\beta}})$ to recover the level of $Y$, we get a biased estimate of $E[Y_i \mid \mathbf{X}_i]$ unless the error term is homoskedastic and normal — conditions that count data almost never satisfy.

**Problem 2: Zeros.** $\log(0)$ is undefined. Researchers often add a small constant $\log(Y + 1)$, but this is ad hoc and the results are sensitive to the constant chosen. In the AlphaFold context, many protein families may have zero deposits in a given year — dropping these observations or adding constants distorts the estimand.

**Problem 3: Heteroskedasticity-driven inconsistency.** Santos Silva and Tenreyro (2006) proved that log-OLS is *inconsistent* under heteroskedasticity in the level equation. The bias is not a small-sample problem — it persists as $N \to \infty$.

### 4.2 The PPML Estimator

**Poisson Pseudo-Maximum Likelihood (PPML) maximizes the Poisson log-likelihood, but requires only correct specification of the conditional mean — not that the data actually follow a Poisson distribution.**

The model specifies:

$$E[Y_i \mid \mathbf{X}_i] = \exp(\mathbf{X}_i'\boldsymbol{\beta})$$

The Poisson pseudo-log-likelihood is:

$$\ell(\boldsymbol{\beta}) = \sum_{i=1}^N \left[ Y_i \cdot \mathbf{X}_i'\boldsymbol{\beta} - \exp(\mathbf{X}_i'\boldsymbol{\beta}) \right]$$

The first-order condition is:

$$\sum_{i=1}^N \mathbf{X}_i \left[ Y_i - \exp(\mathbf{X}_i'\hat{\boldsymbol{\beta}}) \right] = \mathbf{0}$$

This condition depends only on the conditional mean $E[Y_i \mid \mathbf{X}_i] = \exp(\mathbf{X}_i'\boldsymbol{\beta})$. It does not require equidispersion ($\text{Var}(Y) = E[Y]$), and it does not require $Y$ to be an integer. **PPML is consistent whenever the conditional mean is correctly specified as exponential, regardless of the true distribution of $Y$.**

### 4.3 PPML in the DiD Context

For the AlphaFold study, the PPML DiD specification is:

$$Y_{it} = \exp(\alpha_i + \gamma_t + \delta \cdot W_i \times \text{Post}_t + \mathbf{X}_{it}'\boldsymbol{\beta}) \cdot \eta_{it}$$

where:
- $\alpha_i$: unit (protein family) fixed effect
- $\gamma_t$: time (year) fixed effect
- $W_i$: treatment indicator (AlphaFold-affected family)
- $\text{Post}_t$: post-treatment period indicator
- $\delta$: the DiD coefficient of interest
- $\eta\_{it}$: multiplicative error with $E[\eta\_{it} \vert \alpha\_{i}, \gamma\_{t}, W\_{i}, \text{Post}\_{t}, \mathbf{X}\_{it}] = 1$

**Interpretation of $\hat{\delta}$**: the treatment effect on the treated in proportional terms. Specifically:

$$\exp(\hat{\delta}) - 1 \approx \text{proportional change in } E[Y_{it}] \text{ due to treatment}$$

If $\hat{\delta} = 0.30$, then $\exp(0.30) - 1 = 0.35$, meaning a 35% increase in the expected count outcome due to treatment.

### 4.4 PPML vs. Log-OLS: Head-to-Head

| Axis | Log-OLS | PPML |
|------|---------|------|
| **Handles zeros** | No — $\log(0)$ undefined; requires $\log(Y + c)$ hack | Yes — $\exp(\cdot) \geq 0$ naturally accommodates zeros |
| **Jensen's bias** | Present — $E[\log Y] \neq \log E[Y]$; retransformation needed | Absent — directly models $E[Y]$ |
| **Heteroskedasticity** | Inconsistent if $\text{Var}(\epsilon) = f(\mathbf{X})$ (Santos Silva & Tenreyro, 2006) | Consistent under arbitrary heteroskedasticity in $Y$ |
| **Efficiency** | More efficient *only if* errors are homoskedastic normal | Less efficient under homoskedasticity; more robust otherwise |
| **Interpretation** | $\hat{\beta}$ ≈ semi-elasticity only under restrictive assumptions | $\exp(\hat{\beta}) - 1$ = proportional effect on $E[Y]$; clean |

**The verdict for causal inference with count outcomes: PPML dominates log-OLS on every axis except one (efficiency under a knife-edge assumption that rarely holds).** This is why the AlphaFold proposal adopts PPML as the primary specification for Strategy I.

### 4.5 Multiplicative vs. Additive Parallel Trends

The choice between PPML and OLS for DiD estimation reflects a deeper assumption about how the parallel trends condition is specified.

**Additive parallel trends** (OLS DiD):

$$E[Y_{it}(0) \mid W_i = 1, t = \text{Post}] - E[Y_{it}(0) \mid W_i = 1, t = \text{Pre}] = E[Y_{it}(0) \mid W_i = 0, t = \text{Post}] - E[Y_{it}(0) \mid W_i = 0, t = \text{Pre}]$$

The *level change* in the untreated potential outcome is the same for treated and control groups.

**Multiplicative parallel trends** (PPML DiD):

$$\frac{E[Y_{it}(0) \mid W_i = 1, t = \text{Post}]}{E[Y_{it}(0) \mid W_i = 1, t = \text{Pre}]} = \frac{E[Y_{it}(0) \mid W_i = 0, t = \text{Post}]}{E[Y_{it}(0) \mid W_i = 0, t = \text{Pre}]}$$

The *proportional change* (ratio) in the untreated potential outcome is the same for treated and control groups.

**Which is more plausible depends on the data.** For count outcomes with large variation in baseline levels (some protein families have 1 deposit per year, others have 100), multiplicative trends are typically more credible. A family with 100 deposits per year is likely to show larger absolute changes over time than a family with 1 deposit — but the percentage changes may be comparable. The pre-trend test remains the primary diagnostic: plot $\hat{\delta}_k$ for $k < 0$ and test whether they are jointly zero.

> **Assumption (Correct conditional mean)**: PPML requires $E[Y_i \mid \mathbf{X}_i] = \exp(\mathbf{X}_i'\boldsymbol{\beta})$. If the true conditional mean is not exponential (e.g., if the effect is additive rather than multiplicative), PPML is misspecified. In the DiD context, this corresponds to the multiplicative parallel trends assumption: absent treatment, the *ratio* of outcomes for treated and control would remain constant over time.

---

## 5. OLS with Fixed Effects / Linear Probability Model

### 5.1 The LPM for Binary Outcomes

**This section directly supports Strategy II of the AlphaFold impact study, where the outcome is binary (e.g., whether a protein family has any experimental structure deposited in the PDB).**

The Linear Probability Model (LPM) specifies:

$$Y_{it} = \alpha_i + \gamma_t + \delta \cdot (W_i \times \text{Post}_t) + \mathbf{X}_{it}'\boldsymbol{\beta} + \epsilon_{it}$$

where $Y_{it} \in \{0, 1\}$ and $\hat{\delta}$ estimates the change in probability (in percentage points) of $Y_{it} = 1$ due to treatment.

**Verbal interpretation**: if $\hat{\delta} = 0.05$ with $Y_{it}$ = "has PDB deposit," then AlphaFold-affected families experience a 5-percentage-point increase in the probability of having an experimental structure deposited.

### 5.2 Why Not Logit with Fixed Effects?

The natural alternative for a binary outcome is logistic regression with fixed effects:

$$\log \frac{P(Y_{it}=1)}{1 - P(Y_{it}=1)} = \alpha_i + \gamma_t + \delta \cdot (W_i \times \text{Post}_t) + \mathbf{X}_{it}'\boldsymbol{\beta}$$

This raises the *incidental parameters problem* (Neyman and Scott, 1948). With $N$ unit fixed effects, the maximum likelihood estimator of $\alpha_i$ is inconsistent when the number of time periods $T$ is small — and its inconsistency contaminates the estimate of $\delta$.

**The LPM avoids the incidental parameters problem entirely.** OLS with fixed effects is consistent for $\delta$ even when $T$ is small, because the within transformation eliminates $\alpha_i$ without requiring its estimation.

| Feature | LPM (OLS + FE) | Logit + FE |
|---------|:-:|:-:|
| **Incidental parameters** | Not a problem | Biased when $T$ small |
| **Interpretation** | $\hat{\delta}$ = p.p. change in probability | Requires marginal effects computation |
| **Interaction terms** | Straightforward | Nonlinear — Ai and Norton (2003) correction needed |
| **Predictions in $[0,1]$** | Not guaranteed | Guaranteed |
| **Heteroskedasticity** | Inherent: $\text{Var}(Y) = p(1{-}p)$ | Handled by model |

### 5.3 Dealing with LPM's Limitations

**The LPM has two well-known problems, both manageable in practice.**

**Problem 1: Predictions outside $[0, 1]$.** The linear functional form can produce $\hat{Y}\_{it} < 0$ or $\hat{Y}\_{it} > 1$ for extreme covariate values. In a causal inference context, this is rarely a concern — we care about $\hat{\delta}$ (the treatment effect), not about fitted values. The OLS estimate of $\hat{\delta}$ is consistent regardless of whether some predictions fall outside the unit interval.

**Problem 2: Heteroskedasticity.** Since $\text{Var}(Y\_{it} \vert \mathbf{X}\_{it}) = P(Y\_{it}{=}1)(1 - P(Y\_{it}{=}1))$ depends on covariates, the errors are necessarily heteroskedastic. The remedy is standard: use heteroskedasticity-robust standard errors (HC1 or HC3), or — in panel data — cluster standard errors at the unit level.

$$\hat{V}^{\text{cluster}} = \left(\mathbf{X}'\mathbf{X}\right)^{-1} \left(\sum_{i=1}^{N} \mathbf{X}_i' \hat{\boldsymbol{\epsilon}}_i \hat{\boldsymbol{\epsilon}}_i' \mathbf{X}_i \right) \left(\mathbf{X}'\mathbf{X}\right)^{-1}$$

**Cluster at the level at which treatment varies.** In the AlphaFold study, treatment is assigned at the protein-family level, so standard errors should be clustered at the family level.

### 5.4 LPM in Practice: Specification and Reporting

A complete LPM specification for the AlphaFold study takes the form:

$$\text{HasDeposit}_{it} = \alpha_i + \gamma_t + \delta \cdot (\text{Treated}_i \times \text{Post}_t) + \mathbf{X}_{it}'\boldsymbol{\beta} + \epsilon_{it}$$

The coefficient $\hat{\delta}$ is reported as:

- **Point estimate**: e.g., $\hat{\delta} = 0.032$
- **Interpretation**: AlphaFold-affected protein families experience a 3.2-percentage-point increase in the probability of having at least one experimental structure deposited in the PDB in a given year.
- **Standard error**: clustered at the protein-family level.
- **Baseline rate**: report the pre-treatment mean of $Y_{it}$ among the treated group. If the baseline probability is 0.15, then $\hat{\delta} = 0.032$ represents a $0.032 / 0.15 = 21.3\%$ relative increase.

> **Pitfall**: A common error is to interpret LPM coefficients as percentages. They are *percentage points*. If $\hat{\delta} = 0.05$, the effect is 5 percentage points, not 5 percent. The distinction matters enormously when the baseline probability is small.

---

## 6. Double/Debiased Machine Learning (DML)

### 6.1 The Problem: Regularization Bias

**Plugging machine learning predictions directly into causal formulas introduces regularization bias that invalidates inference.**

Consider the partially linear model:

$$Y_i = \tau W_i + g(\mathbf{X}_i) + U_i$$

where $g(\cdot)$ is an unknown nuisance function and $\tau$ is the causal parameter of interest. A natural approach: estimate $\hat{g}$ with a machine learning model (random forest, LASSO, neural net), compute residuals $\tilde{Y}_i = Y_i - \hat{g}(\mathbf{X}_i)$, and regress $\tilde{Y}$ on $W$ to get $\hat{\tau}$.

This fails. ML estimators use regularization (penalty terms, early stopping, pruning) that introduces systematic bias in $\hat{g}$. When we plug $\hat{g}$ into the estimation formula, this bias does not average out — it propagates into $\hat{\tau}$. The resulting $\hat{\tau}$ converges at a rate slower than $\sqrt{N}$, standard errors are wrong, and confidence intervals have incorrect coverage.

**The bias is not a finite-sample problem. It is an asymptotic failure: the regularization bias dominates the sampling variability.**

### 6.2 The DML Solution

Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey, and Robins (2018) proposed Double/Debiased Machine Learning, which solves the problem with two innovations.

**(a) Neyman orthogonality.** Construct a score function $\psi$ for $\tau$ whose derivative with respect to the nuisance parameters (propensity score, outcome regression) is zero at the true values. This means small errors in nuisance estimation produce only second-order bias in $\hat{\tau}$.

**(b) Cross-fitting.** Split the sample into $K$ folds. For each fold $k$, train nuisance models on the remaining $K-1$ folds and predict on fold $k$. This avoids overfitting bias — the predictions used to construct $\hat{\tau}$ are out-of-sample.

### 6.3 The Partially Linear Model: Step by Step

The DML procedure for the partially linear model:

**Data-generating process:**

$$Y_i = \tau W_i + g(\mathbf{X}_i) + U_i, \quad E[U_i \mid \mathbf{X}_i, W_i] = 0$$

$$W_i = m(\mathbf{X}_i) + V_i, \quad E[V_i \mid \mathbf{X}_i] = 0$$

where $g(\mathbf{X}_i) = E[Y_i \mid \mathbf{X}_i, W_i = 0]$ is the outcome nuisance and $m(\mathbf{X}_i) = E[W_i \mid \mathbf{X}_i]$ is the propensity nuisance.

**Step 1: Split data into $K$ folds** ($K = 5$ is standard).

**Step 2: For each fold $k = 1, \dots, K$:**

- Train $\hat{g}^{(-k)}$ on all data *except* fold $k$.
- Train $\hat{m}^{(-k)}$ on all data *except* fold $k$.
- Compute residuals for units in fold $k$:
  - $\tilde{Y}_i = Y_i - \hat{g}^{(-k)}(\mathbf{X}_i)$
  - $\tilde{W}_i = W_i - \hat{m}^{(-k)}(\mathbf{X}_i)$

**Step 3: Pool all residuals and estimate:**

$$\hat{\tau}_{\text{DML}} = \left(\sum_{i=1}^N \tilde{W}_i^2\right)^{-1} \sum_{i=1}^N \tilde{W}_i \tilde{Y}_i$$

**Verbal interpretation**: first, partial out the effect of confounders $\mathbf{X}$ from both the outcome $Y$ and the treatment $W$ using flexible ML models. Then regress the residualized outcome on the residualized treatment. The Robinson (1988) decomposition guarantees that this yields a consistent estimate of $\tau$ — and cross-fitting ensures the ML estimation errors do not contaminate inference.

### 6.4 The Cross-Fitting Diagram

```
DML CROSS-FITTING PROCEDURE (K=5)

DATA: [  Fold 1  |  Fold 2  |  Fold 3  |  Fold 4  |  Fold 5  ]

Round 1:
  Train on: [  Fold 2  |  Fold 3  |  Fold 4  |  Fold 5  ]  → ĝ⁻¹, m̂⁻¹
  Predict:  [  Fold 1  ]  → Ỹ₁, W̃₁

Round 2:
  Train on: [  Fold 1  |  Fold 3  |  Fold 4  |  Fold 5  ]  → ĝ⁻², m̂⁻²
  Predict:  [  Fold 2  ]  → Ỹ₂, W̃₂

Round 3:
  Train on: [  Fold 1  |  Fold 2  |  Fold 4  |  Fold 5  ]  → ĝ⁻³, m̂⁻³
  Predict:  [  Fold 3  ]  → Ỹ₃, W̃₃

Round 4:
  Train on: [  Fold 1  |  Fold 2  |  Fold 3  |  Fold 5  ]  → ĝ⁻⁴, m̂⁻⁴
  Predict:  [  Fold 4  ]  → Ỹ₄, W̃₄

Round 5:
  Train on: [  Fold 1  |  Fold 2  |  Fold 3  |  Fold 4  ]  → ĝ⁻⁵, m̂⁻⁵
  Predict:  [  Fold 5  ]  → Ỹ₅, W̃₅

Pool: τ̂_DML = (Σ W̃ᵢ²)⁻¹ Σ W̃ᵢỸᵢ    over ALL folds
```

**Cross-fitting is not the same as cross-validation.** In cross-validation, we average *predictions* across folds to select a model. In cross-fitting, we use out-of-sample *predictions* to construct the final estimate — every unit's residual is computed from a model that never saw that unit during training.

### 6.5 The Key Theorem

> **Theorem 6.1 (DML Convergence, Chernozhukov et al. 2018)**: Under regularity conditions, if the nuisance estimators $\hat{g}$ and $\hat{m}$ converge at rates such that $\|\hat{g} - g\|_2 \cdot \|\hat{m} - m\|_2 = o(N^{-1/2})$, then:
>
> $$\sqrt{N}(\hat{\tau}_{\text{DML}} - \tau) \xrightarrow{d} \mathcal{N}(0, \sigma^2)$$
>
> where $\sigma^2$ is the semiparametric efficiency bound.

**Verbal interpretation**: each nuisance model needs to converge faster than $N^{-1/4}$ (in $L_2$ norm). This is a mild requirement that most modern ML methods satisfy — random forests, boosted trees, LASSO, and neural networks all achieve $N^{-1/4}$ rates under standard conditions. The causal parameter $\hat{\tau}$ then converges at the parametric $\sqrt{N}$ rate with valid Gaussian inference, regardless of the complexity of $g$ and $m$.

**This is the central appeal of DML: use any ML method for nuisance estimation while maintaining classical statistical inference for the causal parameter.**

### 6.6 Choosing the ML Method

The DML framework is agnostic about which ML algorithm estimates the nuisance functions. In practice, the choice matters for finite-sample performance, though not for asymptotic validity.

| ML Method | Strengths for Nuisance Estimation | Caveats |
|-----------|----------------------------------|---------|
| **LASSO** | Fast; interpretable; $N^{-1/4}$ rate under sparsity | Assumes approximate sparsity in $g$ and $m$ |
| **Random Forest** | Handles interactions; nonparametric | May undersmooth; tuning required |
| **Gradient Boosting** | High predictive accuracy; flexible | Overfitting risk without early stopping |
| **Neural Network** | Universal approximation; handles complex $g$ | Requires large $N$; tuning overhead |
| **Ensemble (stacking)** | Combines strengths; robust to single-method failure | Computational cost |

**A practical default**: gradient-boosted trees (XGBoost or LightGBM) with moderate depth and early stopping. These achieve strong predictive performance across a wide range of data-generating processes and satisfy the $N^{-1/4}$ rate requirement under standard regularity conditions.

**Sensitivity check**: run DML with two or three different ML methods for the nuisance functions. If $\hat{\tau}_{\text{DML}}$ is stable across choices, the result is robust. If it is sensitive, investigate whether one method is underfitting or overfitting the nuisance functions.

### 6.7 Connection to AIPW

DML with the AIPW score function recovers the doubly robust estimator from Section 3, but with cross-fitted nuisance models. The combination of Neyman orthogonality (from the AIPW influence function) and cross-fitting (from DML) yields an estimator that is:

1. Doubly robust — consistent if either nuisance model is correct.
2. Semiparametrically efficient — achieves the variance bound.
3. ML-compatible — $\hat{e}$ and $\hat{\mu}_w$ can be estimated with flexible nonparametric methods.
4. Inferentially valid — standard errors, $t$-tests, and confidence intervals are asymptotically correct.

---

## 7. Choosing an Estimator: Decision Framework

### 7.1 The Estimation Decision Tree

**Matching the estimator to the problem requires answering three questions: what is the outcome type, what is the identification strategy, and how complex are the confounders?**

```
ESTIMATION DECISION TREE

Start: "I have an identified estimand. How do I estimate it?"
  │
  ├─ Q1: What is the outcome type?
  │   │
  │   ├─ Continuous → OLS / AIPW / DML
  │   │
  │   ├─ Binary → LPM with FE (Sec 5) or AIPW
  │   │
  │   └─ Count (with zeros) → PPML (Sec 4)
  │
  ├─ Q2: What is the identification strategy?
  │   │
  │   ├─ Selection on observables (backdoor)
  │   │   └─ Matching / IPW / AIPW / DML
  │   │
  │   ├─ DiD (panel data)
  │   │   ├─ Count outcome → PPML + unit & time FE
  │   │   ├─ Binary outcome → LPM + unit & time FE
  │   │   └─ Continuous → OLS + unit & time FE
  │   │
  │   ├─ IV → 2SLS / LIML
  │   │
  │   └─ RDD → Local polynomial regression
  │
  └─ Q3: Are confounders high-dimensional / nonlinear?
      │
      ├─ No (p < 20, linear) → Parametric methods suffice
      │
      └─ Yes (p >> 20, interactions) → DML with flexible ML
```

### 7.2 Six Estimators Compared

| Estimator | Nuisance Input | Key Assumption | Output | Best For |
|-----------|---------------|----------------|--------|----------|
| **PSM** | $\hat{e}(\mathbf{x})$ | Unconfoundedness; correct PS model | $\hat{\tau}_{\text{ATT}}$ | Low-dimensional $\mathbf{X}$; interpretable pairs |
| **IPW** | $\hat{e}(\mathbf{x})$ | Unconfoundedness; overlap; correct PS model | $\hat{\tau}$ (ATE) | Full-sample estimation; survey-like designs |
| **AIPW / DR** | $\hat{e}(\mathbf{x})$, $\hat{\mu}_w(\mathbf{x})$ | Unconfoundedness; one model correct | $\hat{\tau}$ (ATE) | Robustness to single model failure |
| **PPML** | Unit FE, time FE | Multiplicative parallel trends; correct cond. mean | $\hat{\delta}$ (proportional) | Count outcomes with zeros; DiD |
| **LPM + FE** | Unit FE, time FE | Additive parallel trends | $\hat{\delta}$ (p.p.) | Binary outcomes; small $T$ panels |
| **DML** | $\hat{g}(\mathbf{x})$, $\hat{m}(\mathbf{x})$ (any ML) | Unconfoundedness; $N^{-1/4}$ nuisance rates | $\hat{\tau}$ (ATE) | High-dim. confounders; flexible modeling |

### 7.3 Mapping to the AlphaFold Impact Study

The AlphaFold proposal uses two identification strategies, each requiring a specific estimator:

| Strategy | Outcome Variable | Outcome Type | Estimator | Parallel Trends |
|----------|-----------------|:------------:|-----------|:---------------:|
| **I** | PDB deposits per family per year | Count (many zeros) | **PPML + FE** | Multiplicative |
| **II** | Any deposit (0/1) per family per year | Binary | **LPM + FE** | Additive |
| **Robustness** | Either | Either | **DML** | Selection on observables |

**Strategy I (PPML)** asks: "By what proportion did AlphaFold increase the rate of experimental structure deposition?" The answer is $\exp(\hat{\delta}) - 1$.

**Strategy II (LPM)** asks: "By how many percentage points did AlphaFold increase the probability that a family has at least one experimental deposit?" The answer is $\hat{\delta}$ directly.

**Robustness (DML)** provides a cross-check: under a selection-on-observables assumption (rather than DiD), does the estimated effect remain in the same range? DML allows flexible ML-based confounding adjustment without functional form commitments.

---

## 8. Practical Guidance

### 8.1 What to Report

**Every applied causal analysis should report a minimum of five quantities.**

1. **Point estimate** $\hat{\tau}$ (or $\hat{\delta}$) with interpretation in natural units.
2. **Standard error** — robust or clustered, with clustering level stated.
3. **95% confidence interval** — $\hat{\tau} \pm 1.96 \times \text{SE}(\hat{\tau})$.
4. **Sample size** $N$ and effective sample size (after trimming, matching, or weighting).
5. **Balance diagnostics** — covariate balance table or love plot (for matching/IPW).

### 8.2 Common Errors

| Error | Why It Fails | Fix |
|-------|-------------|-----|
| Using PSM and claiming causal identification | PSM implements unconfoundedness; it does not establish it | State the identification assumption explicitly |
| Running log-OLS on count data with zeros | $\log(0)$ undefined; Jensen's bias; heteroskedasticity inconsistency | Use PPML |
| Plugging ML into IPW without cross-fitting | Regularization bias → invalid inference | Use DML |
| Logit + FE with short panels | Incidental parameters bias | Use LPM + robust/clustered SE |
| Ignoring extreme weights in IPW | Single unit dominates estimate | Trim, stabilize, or use overlap weighting |

### 8.3 The Estimation-Identification Interface

A recurring theme deserves final emphasis. **Estimation and identification are separate steps, and each can fail independently.**

| Scenario | Identification | Estimation | Result |
|----------|:-:|:-:|--------|
| RCT + OLS | Correct (randomization) | Correct (consistent) | Valid |
| RCT + ML without cross-fitting | Correct | Wrong (regularization bias) | Invalid inference |
| Bad DAG + perfect AIPW | Wrong (unidentified) | Correct machinery | Consistently wrong number |
| Good DAG + wrong outcome model + right PS | Correct | Partially correct (DR rescues) | Valid (double robustness) |

**No estimator can rescue a failed identification strategy.** The most sophisticated DML implementation yields a precise estimate of the wrong quantity if the estimand does not equal the causal effect. Conversely, a correct estimand paired with a biased estimator yields a number that may not converge to the truth. Both steps must succeed.

### 8.4 The Big Picture

The estimation methods in this post transform identified estimands into numbers with valid uncertainty quantification. The progression is:

1. **Matching** — intuitive but limited to low-dimensional settings.
2. **IPW** — uses all data but is fragile with extreme propensity scores.
3. **AIPW** — doubly robust and semiparametrically efficient.
4. **PPML** — the right tool for multiplicative models with count outcomes.
5. **LPM + FE** — the pragmatic choice for binary outcomes in panel data.
6. **DML** — the frontier: ML-powered nuisance estimation with parametric inference on the causal parameter.

**The methods are complementary, not competing.** A rigorous applied study runs multiple estimators and checks whether conclusions are robust across them. If PPML and LPM tell different stories, the difference is informative — it reveals whether the effect operates on the intensive margin (how many structures) or the extensive margin (whether any structure), and whether the parallel trends assumption is better specified in multiplicative or additive form.

The full estimation pipeline, from estimand to reported result, is:

```
ESTIMATION PIPELINE

  Identified Estimand (from Part 3)
        │
        ▼
  Choose Estimator
    ├── Outcome type (count / binary / continuous)
    ├── Identification strategy (DiD / IV / RDD / selection on observables)
    └── Confounder complexity (low-dim parametric / high-dim ML)
        │
        ▼
  Estimate Nuisance Parameters
    ├── Propensity score ê(x)
    ├── Outcome regression μ̂_w(x)
    └── Cross-fit if using ML
        │
        ▼
  Compute Point Estimate τ̂
        │
        ▼
  Standard Errors
    ├── Robust (heteroskedasticity-consistent)
    ├── Clustered (at treatment-assignment level)
    └── Bootstrap (if closed-form SE unavailable)
        │
        ▼
  Report: τ̂ ± 1.96 × SE, with balance diagnostics
```

---

## 9. Looking Ahead

This post covered the machinery for converting an identified estimand into an estimate with valid inference. But we estimated only a single number — the *average* treatment effect (or its DiD analog). What if the effect varies across units? What if AlphaFold helps some protein families enormously and others not at all?

**Part 5 (Heterogeneous Effects and Causal Discovery)** addresses this question. We move from "what is the average effect?" to "who benefits most?" — using conditional average treatment effects (CATE), causal forests, and meta-learners. We also reverse the problem entirely: instead of assuming a causal graph and estimating effects, can we *learn* the graph from data?

---

## References

- Ai, C., & Norton, E. C. (2003). Interaction terms in logit and probit models. *Economics Letters*, 80(1), 123-129.
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1-C68.
- Horvitz, D. G., & Thompson, D. J. (1952). A generalization of sampling without replacement from a finite universe. *Journal of the American Statistical Association*, 47(260), 663-685.
- Li, F., Morgan, K. L., & Zaslavsky, A. M. (2018). Balancing covariates via propensity score weighting. *Journal of the American Statistical Association*, 113(521), 390-400.
- Neyman, J., & Scott, E. L. (1948). Consistent estimates based on partially consistent observations. *Econometrica*, 16(1), 1-32.
- Robins, J. M., Rotnitzky, A., & Zhao, L. P. (1994). Estimation of regression coefficients when some regressors are not always observed. *Journal of the American Statistical Association*, 89(427), 846-866.
- Robinson, P. M. (1988). Root-N-consistent semiparametric regression. *Econometrica*, 56(4), 931-954.
- Rosenbaum, P. R., & Rubin, D. B. (1983). The central role of the propensity score in observational studies for causal effects. *Biometrika*, 70(1), 41-55.
- Santos Silva, J. M. C., & Tenreyro, S. (2006). The log of gravity. *The Review of Economics and Statistics*, 88(4), 641-658.

---
