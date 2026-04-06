---
title: "Causal Inference Part 1: The Language of Causation — Potential Outcomes"
date: 2026-04-06 13:56:00 +0900
categories: [AI & ML, ML Theory]
tags: [causal-inference, potential-outcomes, rubin-causal-model, ate, selection-bias]
math: true
---

**Causal Inference — From First Principles to Automated Reasoning**

*This is Part 1 of an 8-part series.*

- **Part 0:** [Beyond Correlation — Why Causal Inference?](/posts/causal-inference-part0-beyond-correlation/)
- **Part 1:** The Language of Causation — Potential Outcomes (this post)
- **Part 2:** [Structural Causal Models](/posts/causal-inference-part2-structural-causal-models/)
- **Part 3:** [Identification](/posts/causal-inference-part3-identification/)
- **Part 4:** [Estimation](/posts/causal-inference-part4-estimation/)
- **Part 5:** [Heterogeneous Effects & Discovery](/posts/causal-inference-part5-heterogeneous-effects/)
- **Part 6:** [The Causal Pipeline](/posts/causal-inference-part6-causal-pipeline/)
- **Part 7:** [The Causal Agent](/posts/causal-inference-part7-causal-agent/)

---

## Hook

> *"Every causal claim is a claim about a world we did not observe."*

You train a model that predicts patient survival with 0.95 AUC. A clinician asks: *"Should I give this drug to the next patient?"* Your model cannot answer. It learned $P(Y \mid X, W)$ — the association between treatment $W$ and outcome $Y$ given covariates $X$ — but association is not causation. The drug might correlate with survival simply because healthier patients are more likely to receive it.

**To answer the clinician's question, you need a formal language for counterfactual reasoning.** That language is the *Rubin Causal Model* (RCM), also called the *potential outcomes framework*. It reduces every causal question to a missing data problem: for each patient, we observe the outcome under the treatment they received, but the outcome under the treatment they *did not* receive is forever missing.

This post builds that language from scratch. By the end you will be able to:

1. Write down the four core causal estimands (ATE, ATT, CATE, LATE).
2. State the *fundamental problem of causal inference*.
3. Decompose a naive comparison into causal effect + selection bias.
4. List the three identifying assumptions and explain when each fails.
5. Explain exactly *why* randomization works — not as magic, but as mechanism.

---

## 1. Potential Outcomes — The Missing Data Framework

### 1.1 Two Worlds per Unit

Consider $N$ units indexed by $i = 1, \dots, N$. Each unit receives a binary treatment $W_i \in \{0, 1\}$ (1 = treated, 0 = control). **For every unit, two potential outcomes exist — one for each treatment state — regardless of which treatment is actually assigned.**

$$
Y_i(1) \quad \text{(outcome if unit } i \text{ is treated)}
$$

$$
Y_i(0) \quad \text{(outcome if unit } i \text{ is in control)}
$$

These are *fixed attributes* of unit $i$, not random variables conditioned on treatment. The treatment does not create the potential outcomes; it reveals one of them.

Think of it this way: before any treatment decision is made, each unit already "carries" two numbers — the outcome it would show if treated and the outcome it would show if left alone. The treatment assignment simply selects which number nature reveals.

### 1.2 The Switching Equation

The observed outcome $Y_i$ follows the *switching equation*:

$$
Y_i = W_i \cdot Y_i(1) + (1 - W_i) \cdot Y_i(0)
$$

This is a deterministic link between the potential outcomes and the observation. If $W_i = 1$, we observe $Y_i(1)$; if $W_i = 0$, we observe $Y_i(0)$. The other potential outcome is *counterfactual* — it describes what would have happened under a different treatment.

> **Pitfall**: The switching equation is not a model. It is a *definition*. It holds by construction, not by assumption.

### 1.3 The Fundamental Problem of Causal Inference

The individual causal effect for unit $i$ is:

$$
\tau_i = Y_i(1) - Y_i(0)
$$

**The individual causal effect $\tau_i$ is fundamentally unobservable because we can never observe both $Y_i(1)$ and $Y_i(0)$ for the same unit at the same time** (Holland, 1986). This is not a limitation of our instruments or our data; it is a logical impossibility. A patient either takes the drug or does not; we cannot simultaneously observe both realities.

This is the *fundamental problem of causal inference*.

One might object: "Can we observe the same patient twice — once with and once without treatment?" This is the *crossover design*, and it works only if (a) the treatment effect is reversible, (b) there are no carryover effects, and (c) the unit does not change between periods. These conditions are rarely met in practice, so the fundamental problem remains.

### 1.4 The Science Table

The science table makes the missing data structure explicit. Question marks denote counterfactual (unobserved) values:

```
┌──────┬─────┬──────┬──────┬────────┬──────────────────┐
│ Unit │  W  │ Y(1) │ Y(0) │ Y(obs) │ τ_i = Y(1)-Y(0) │
├──────┼─────┼──────┼──────┼────────┼──────────────────┤
│  1   │  1  │   7  │  ?   │    7   │        ?         │
│  2   │  0  │  ?   │  3   │    3   │        ?         │
│  3   │  1  │   5  │  ?   │    5   │        ?         │
│  4   │  0  │  ?   │  6   │    6   │        ?         │
│  5   │  1  │   8  │  ?   │    8   │        ?         │
│  6   │  0  │  ?   │  4   │    4   │        ?         │
└──────┴─────┴──────┴──────┴────────┴──────────────────┘
```

**Every row has exactly one missing potential outcome — this is the defining structure of the causal inference problem.** All methods in this series are strategies for filling in, averaging over, or bounding those question marks.

Suppose (hypothetically) we could peek behind the curtain and see the full table:

```
┌──────┬─────┬──────┬──────┬────────┬──────────────────┐
│ Unit │  W  │ Y(1) │ Y(0) │ Y(obs) │ τ_i = Y(1)-Y(0) │
├──────┼─────┼──────┼──────┼────────┼──────────────────┤
│  1   │  1  │   7  │  4   │    7   │       +3         │
│  2   │  0  │   5  │  3   │    3   │       +2         │
│  3   │  1  │   5  │  5   │    5   │        0         │
│  4   │  0  │   8  │  6   │    6   │       +2         │
│  5   │  1  │   8  │  6   │    8   │       +2         │
│  6   │  0  │   3  │  4   │    4   │       -1         │
└──────┴─────┴──────┴──────┴────────┴──────────────────┘
                                      ATE = (3+2+0+2+2-1)/6
                                          = 8/6 ≈ 1.33
```

The individual effects range from $-1$ to $+3$, and no single row's effect equals the average. The ATE is a *population summary*, not a unit-level truth. In reality, we never have this full table — the question marks are permanent.

### 1.5 Contrast with Prediction

| Aspect | Prediction | Causal Inference |
|--------|-----------|------------------|
| **Goal** | Estimate $E[Y \mid X, W]$ | Estimate $E[Y(1) - Y(0)]$ |
| **Missing data** | None (outcome is observed) | Half the potential outcomes |
| **Key threat** | Overfitting, distribution shift | Confounding, selection bias |
| **Validation** | Hold-out test set | No direct validation possible without design |
| **Core formalism** | Statistical learning theory | Potential outcomes / SCMs |

The prediction task asks: *"What outcome do we expect to see?"* The causal task asks: *"What outcome would change if we intervened?"* These are fundamentally different questions, and conflating them is the root of most causal errors in applied ML.

---

## 2. From Individual to Average — Causal Estimands

Since $\tau_i$ is unobservable, we target *population-level* summaries. **Four estimands dominate applied causal inference, each averaging the individual effect over a different population.**

### 2.1 Average Treatment Effect (ATE)

$$
\tau = E[Y_i(1) - Y_i(0)]
$$

The ATE is the expected causal effect across the *entire population*. It answers: "If we randomly assigned treatment to a unit drawn from the population, what effect would we expect?" This is the policy-relevant quantity when a government considers rolling out a universal program.

By linearity of expectation:

$$
\tau = E[Y_i(1)] - E[Y_i(0)]
$$

### 2.2 Average Treatment Effect on the Treated (ATT)

$$
\tau_{\text{ATT}} = E[Y_i(1) - Y_i(0) \mid W_i = 1]
$$

The ATT is the expected causal effect among *units that actually received treatment*. It answers: "Did the treatment work for the people who got it?" This is the natural estimand for program evaluation: we want to know whether the program helped its participants, not a hypothetical random person.

### 2.3 Conditional Average Treatment Effect (CATE)

$$
\tau(\mathbf{x}) = E[Y_i(1) - Y_i(0) \mid \mathbf{X}_i = \mathbf{x}]
$$

The CATE is the expected causal effect for the subpopulation with covariates $\mathbf{x} \in \mathbb{R}^p$. It answers: "How does the treatment effect vary across individuals?" The CATE is the foundation of *personalized* treatment assignment — Part 5 of this series covers heterogeneous treatment effects in depth.

Note the relationship: the ATE is the CATE averaged over the covariate distribution:

$$
\tau = E_{\mathbf{X}}[\tau(\mathbf{X}_i)]
$$

### 2.4 Local Average Treatment Effect (LATE)

$$
\tau_{\text{LATE}} = E[Y_i(1) - Y_i(0) \mid \text{unit } i \text{ is a complier}]
$$

The LATE is the expected causal effect among *compliers* — units whose treatment status is determined by the instrument. It answers: "What is the effect for units who would change their behavior in response to the encouragement?" This estimand arises naturally in instrumental variable designs (Part 3).

To define compliers precisely, consider a binary instrument $Z_i \in \{0,1\}$ (e.g., randomized encouragement to take the drug). Units partition into four latent types:

| Type | Behavior if $Z=0$ | Behavior if $Z=1$ | Responds to instrument? |
|------|:-----------------:|:-----------------:|:----------------------:|
| **Complier** | $W=0$ | $W=1$ | Yes |
| **Always-taker** | $W=1$ | $W=1$ | No |
| **Never-taker** | $W=0$ | $W=0$ | No |
| **Defier** | $W=1$ | $W=0$ | Perversely |

Under a monotonicity assumption (no defiers), the Wald estimator $\tau_{\text{LATE}} = \frac{E[Y \mid Z=1] - E[Y \mid Z=0]}{E[W \mid Z=1] - E[W \mid Z=0]}$ identifies the effect for compliers (Angrist, Imbens, and Rubin, 1996). We return to this in Part 3.

### 2.5 When Do Estimands Differ?

**The four estimands coincide only when the treatment effect is constant ($\tau_i = \tau$ for all $i$); otherwise, selection into treatment drives them apart.**

Consider a drug that benefits sick patients (high $\tau_i$) but not healthy ones (low $\tau_i$). If sicker patients are more likely to take the drug:
- $\tau_{\text{ATT}} > \tau$ because the treated group is enriched for high-effect individuals.
- $\tau(\mathbf{x})$ varies across health status.
- $\tau_{\text{LATE}}$ depends on which subpopulation complies with the instrument.

### 2.6 Estimand Comparison Table

| | **ATE** $\tau$ | **ATT** $\tau\_{\text{ATT}}$ | **CATE** $\tau(\mathbf{x})$ | **LATE** $\tau\_{\text{LATE}}$ |
|---|---|---|---|---|
| **Definition** | $E[Y(1) - Y(0)]$ | $E[Y(1) - Y(0) \vert W=1]$ | $E[Y(1) - Y(0) \vert \mathbf{X}=\mathbf{x}]$ | $E[Y(1) - Y(0) \vert \text{complier}]$ |
| **Target pop.** | Entire population | Treated units only | Subgroup with $\mathbf{X}=\mathbf{x}$ | Complier subpopulation |
| **Use case** | Universal policy | Program evaluation | Personalized treatment | Imperfect compliance |
| **Identified by** | RCT, matching, IPW | DiD, matching | Causal forests, meta-learners | IV / Wald estimator |
| **Requires** | Unconfoundedness | Parallel trends / unconf. | Unconf. + flexible model | Instrument validity |

---

## 3. The Selection Bias Decomposition

### 3.1 The Naive Comparison

The most natural comparison — the difference in mean outcomes between treated and control groups — is:

$$
E[Y_i \mid W_i = 1] - E[Y_i \mid W_i = 0]
$$

**This quantity is observable, but it does not, in general, equal any causal estimand.** The gap between the naive comparison and the causal effect is *selection bias*.

### 3.2 The Decomposition

We derive the decomposition step by step. Start with the treated group's mean outcome and apply the switching equation:

$$
E[Y_i \mid W_i = 1] = E[Y_i(1) \mid W_i = 1]
$$

For the control group:

$$
E[Y_i \mid W_i = 0] = E[Y_i(0) \mid W_i = 0]
$$

Now add and subtract $E[Y_i(0) \mid W_i = 1]$ from the difference:

$$
\underbrace{E[Y_i \mid W_i = 1] - E[Y_i \mid W_i = 0]}_{\text{naive comparison}} = \underbrace{E[Y_i(1) - Y_i(0) \mid W_i = 1]}_{\tau_{\text{ATT}}} + \underbrace{E[Y_i(0) \mid W_i = 1] - E[Y_i(0) \mid W_i = 0]}_{\text{selection bias}}
$$

The key step is the add-and-subtract trick: inserting $E[Y_i(0) \mid W_i = 1]$ creates two differences — one that compares potential outcomes *within* the treated group (the causal effect), and one that compares baseline potential outcomes *across* groups (the bias).

The verbal interpretation: the naive comparison equals the causal effect on the treated *plus* a bias term that captures systematic differences in baseline outcomes between the treatment and control groups.

**The naive comparison equals the causal effect only when the selection bias term is exactly zero** — that is, when treated and control units would have had the same average outcome *in the absence of treatment*.

### 3.3 Anatomy of the Decomposition (Diagram)

```
                    Naive Comparison
          E[Y | W=1] - E[Y | W=0]
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
    Causal Effect           Selection Bias
    (ATT = τ_ATT)      E[Y(0)|W=1] - E[Y(0)|W=0]
          │                       │
          │                       │
  "Treatment moved            "Groups differed
   outcomes for the            even before
   treated group"              treatment"
```

### 3.4 Concrete Example: Drug Efficacy

Suppose we study a blood-pressure drug. Sicker patients (higher baseline BP) are more likely to receive the drug. The data show:

- Treated group mean outcome: $E[Y \mid W=1] = 140$ mmHg
- Control group mean outcome: $E[Y \mid W=0] = 125$ mmHg
- Naive comparison: $140 - 125 = +15$ mmHg

This suggests the drug *raises* blood pressure. But decompose:

- True ATT: $\tau_{\text{ATT}} = -10$ mmHg (the drug lowers BP by 10)
- Selection bias: $E[Y(0) \mid W=1] - E[Y(0) \mid W=0] = 150 - 125 = +25$ mmHg

The treated group would have had 150 mmHg *without* the drug (they were sicker). The drug brought them down to 140. The naive comparison of +15 is the sum of the true effect ($-10$) and the selection bias ($+25$).

| Component | Value | Interpretation |
|-----------|------:|----------------|
| Naive comparison | $+15$ | Treated have higher BP than control |
| $\tau_{\text{ATT}}$ | $-10$ | Drug lowers BP by 10 for treated |
| Selection bias | $+25$ | Treated were 25 mmHg sicker at baseline |
| Check: $-10 + 25$ | $= +15$ | Decomposition is exact |

> **Pitfall**: Observational studies that report unadjusted mean differences are reporting the naive comparison, not the causal effect. The sign of the selection bias can flip the apparent direction of the effect, as this example illustrates.

---

## 4. Assumptions That Enable Identification

The fundamental problem tells us individual effects are unobservable. Selection bias tells us naive comparisons are biased. **Identification requires assumptions — there is no assumption-free causal inference.** The three core assumptions of the potential outcomes framework transform the causal question into a statistical one.

### 4.1 SUTVA

> **Assumption 1.1 (Stable Unit Treatment Value Assumption — SUTVA)**:
>
> (a) **No interference**: The potential outcomes of unit $i$ depend only on $i$'s own treatment assignment, not on the treatments assigned to other units. Formally, $Y_i(w_1, \dots, w_N) = Y_i(w_i)$ for all assignment vectors.
>
> (b) **No hidden versions of treatment**: There is only one version of each treatment level. If $W_i = w$, then $Y_i = Y_i(w)$ (consistency).

**Plain English**: My outcome depends only on *my* treatment, and "treatment" means the same thing for everyone.

**When it fails**: Vaccination. If your neighbor is vaccinated, your probability of infection drops even if *you* are not vaccinated (interference). SUTVA also fails when the treatment is ill-defined — "exercise more" admits many versions (running vs. swimming vs. cycling), each potentially producing different outcomes.

**Consequence of failure**: The potential outcomes $Y_i(0)$ and $Y_i(1)$ are not well-defined. The entire framework collapses — we cannot even *write down* the estimand, let alone estimate it.

```
SUTVA Satisfied:                    SUTVA Violated (Interference):

Unit 1 ←── W₁ ──→ Y₁(w₁)          Unit 1 ←── W₁ ──→ Y₁(w₁, w₂)
                                                          ↑
Unit 2 ←── W₂ ──→ Y₂(w₂)          Unit 2 ←── W₂ ───────┘
                                         └──→ Y₂(w₁, w₂)
(Independent)                       (Outcome depends on
                                     other units' treatment)
```

### 4.2 Ignorability (Unconfoundedness)

> **Assumption 1.2 (Ignorability / Unconfoundedness)**:
>
> $$(Y_i(1), Y_i(0)) \perp\!\!\!\perp W_i \mid \mathbf{X}_i$$
>
> Conditional on observed pre-treatment covariates $\mathbf{X}_i$, treatment assignment $W_i$ is independent of the potential outcomes.

**Plain English**: After controlling for $\mathbf{X}_i$, the treatment and control groups are comparable — any remaining differences in treatment assignment are "as good as random."

This is also called *conditional exchangeability* or *selection on observables*. The word "ignorability" comes from Rubin: the assignment mechanism is ignorable for inference because, once we condition on $\mathbf{X}_i$, the reason a unit was assigned to treatment carries no additional information about its potential outcomes.

A weaker variant, *mean ignorability*, requires only $E[Y_i(w) \mid W_i, \mathbf{X}_i] = E[Y_i(w) \mid \mathbf{X}_i]$ for $w \in \{0,1\}$. This suffices for identifying mean effects but does not support distributional or quantile causal statements.

**When it fails**: An unobserved variable $U$ affects both treatment and outcome. Example: patients with a genetic predisposition (unobserved) are both more likely to develop the disease and more likely to seek treatment. No amount of adjusting for observed covariates removes this confounding.

**Consequence of failure**: Selection bias persists even after conditioning on $\mathbf{X}_i$. All estimators that rely on unconfoundedness (matching, IPW, regression adjustment, DML) are biased. Sensitivity analysis (Rosenbaum bounds, E-values) can quantify how strong the unmeasured confounding would need to be to overturn the result, but cannot eliminate the bias.

> **Pitfall**: Ignorability is *untestable* from observed data alone. You cannot verify from your dataset that all confounders are measured. Domain knowledge, not statistical tests, justifies this assumption.

### 4.3 Positivity (Overlap)

> **Assumption 1.3 (Positivity / Overlap)**:
>
> $$0 < P(W_i = 1 \mid \mathbf{X}_i = \mathbf{x}) < 1 \quad \text{for all } \mathbf{x} \text{ with } P(\mathbf{X}_i = \mathbf{x}) > 0$$
>
> Equivalently: $0 < e(\mathbf{x}) < 1$ where $e(\mathbf{x}) = P(W_i = 1 \mid \mathbf{X}_i = \mathbf{x})$ is the propensity score.

**Plain English**: For every subgroup defined by covariates, there must be a nonzero chance of receiving either treatment or control. No subgroup is *deterministically* assigned to one arm.

**When it fails**: In a study of surgery vs. medication, very elderly patients might *never* receive surgery (clinical guidelines prohibit it). For those patients, $e(\mathbf{x}) = 0$, and the causal effect is not identified because we have no surgical outcomes to learn from. Near-violations ($e(\mathbf{x}) \approx 0$ or $\approx 1$) produce extreme IPW weights and high-variance estimates even when the strict condition holds.

**Consequence of failure**: The conditional expectation $E[Y(w) \mid \mathbf{X} = \mathbf{x}]$ is not identified for the affected subgroup. Extrapolation, not interpolation, would be required. Estimators based on inverse propensity weighting become unstable or undefined (division by near-zero).

### 4.4 The Identification Chain

**Under SUTVA + Ignorability + Positivity, the ATE is identified from observational data.**

Here is the derivation, step by step:

$$
\tau = E[Y_i(1)] - E[Y_i(0)]
$$

Apply the law of iterated expectations over $\mathbf{X}_i$:

$$
= E_{\mathbf{X}}\left[E[Y_i(1) \mid \mathbf{X}_i]\right] - E_{\mathbf{X}}\left[E[Y_i(0) \mid \mathbf{X}_i]\right]
$$

Apply ignorability: $(Y_i(1), Y_i(0)) \perp\!\!\!\perp W_i \mid \mathbf{X}_i$ implies $E[Y_i(w) \mid \mathbf{X}_i] = E[Y_i(w) \mid W_i = w, \mathbf{X}_i]$:

$$
= E_{\mathbf{X}}\left[E[Y_i(1) \mid W_i = 1, \mathbf{X}_i]\right] - E_{\mathbf{X}}\left[E[Y_i(0) \mid W_i = 0, \mathbf{X}_i]\right]
$$

Apply consistency (from SUTVA): $Y_i = Y_i(w)$ when $W_i = w$:

$$
= E_{\mathbf{X}}\left[E[Y_i \mid W_i = 1, \mathbf{X}_i] - E[Y_i \mid W_i = 0, \mathbf{X}_i]\right]
$$

Positivity ensures that both conditional expectations are well-defined (the conditioning events have positive probability).

The left-hand side involves *potential outcomes* (unobservable). The right-hand side involves *observed data* only (conditional means of observed $Y$ given observed $W$ and $\mathbf{X}$). The three assumptions bridge the gap.

```
 SUTVA                  Ignorability              Positivity
   │                        │                        │
   ▼                        ▼                        ▼
 Y_i(w) is              Treatment is             Both treatment
 well-defined            "as good as              levels exist in
 for each unit           random" given X          every X-stratum
   │                        │                        │
   └──────────┬─────────────┘                        │
              │                                      │
              ▼                                      │
   E[Y(w)] = E_X[E[Y|W=w, X]]  ◄───────────────────┘
              │
              ▼
   τ = E_X[E[Y|W=1, X] - E[Y|W=0, X]]
              │
              ▼
      IDENTIFIED FROM DATA
```

### 4.5 Assumption Summary Table

| Assumption | Formal Statement | Plain English | Failure Example | Consequence |
|------------|-----------------|---------------|-----------------|-------------|
| **SUTVA** | $Y_i(\mathbf{w}) = Y_i(w_i)$; consistency | My outcome depends only on my treatment | Vaccination (herd immunity), network effects | Estimand is ill-defined |
| **Ignorability** | $(Y(1), Y(0)) \perp\!\!\!\perp W \vert \mathbf{X}$ | No unmeasured confounders | Unobserved genetic risk, self-selection | Persistent bias in all estimators |
| **Positivity** | $0 < e(\mathbf{x}) < 1$ for all relevant $\mathbf{x}$ | Both arms possible in every subgroup | Elderly patients never receive surgery | Extrapolation; unstable weights |

---

## 5. Why Randomization Works

### 5.1 Randomization as Assumption Satisfaction

In a randomized controlled trial (RCT), treatment is assigned by a known, researcher-controlled mechanism — typically a coin flip independent of everything else:

$$
W_i \perp\!\!\!\perp (Y_i(1), Y_i(0), \mathbf{X}_i)
$$

This is *unconditional* independence — stronger than Assumption 1.2, which requires independence only *conditional on* $\mathbf{X}_i$.

**Randomization is not magic; it mechanically satisfies the identifying assumptions.** Let us verify each one:

| Assumption | Status under RCT | Why |
|------------|:-----------------:|-----|
| **SUTVA** | Must still be assumed | Randomization does not prevent interference or multiple treatment versions |
| **Ignorability** | Satisfied by design | $W_i$ is independent of $(Y(1), Y(0))$ — no confounders, observed or unobserved |
| **Positivity** | Satisfied by design | Each unit has a known, bounded probability of assignment (e.g., 0.5) |

### 5.2 Selection Bias Vanishes

Recall the decomposition:

$$
E[Y \mid W=1] - E[Y \mid W=0] = \tau_{\text{ATT}} + \underbrace{E[Y(0) \mid W=1] - E[Y(0) \mid W=0]}_{\text{selection bias}}
$$

Under randomization, $W_i \perp\!\!\!\perp Y_i(0)$, so:

$$
E[Y_i(0) \mid W_i = 1] = E[Y_i(0) \mid W_i = 0] = E[Y_i(0)]
$$

The selection bias term is exactly zero. By an identical argument, $E[Y_i(1) \mid W_i = 1] = E[Y_i(1) \mid W_i = 0] = E[Y_i(1)]$. It follows that $\tau_{\text{ATT}} = \tau$ because treatment assignment carries no information about which units have larger or smaller effects. The naive comparison *is* the ATE:

$$
E[Y \mid W=1] - E[Y \mid W=0] = \tau
$$

**This is why the simple difference-in-means is the gold-standard estimator in an RCT.** No regression adjustment, no propensity scores, no matching — just means.

Note that regression adjustment *can* still be used in an RCT — not for bias reduction (there is no bias to reduce) but for *variance reduction*. Conditioning on prognostic covariates $\mathbf{X}_i$ shrinks the standard error of $\hat{\tau}$, increasing statistical power. This is the logic behind ANCOVA and the Lin (2013) estimator.

### 5.3 The Observational Challenge

Every observational method in Parts 3-5 of this series is an attempt to *approximate* what randomization *guarantees*:

```
                          ┌──────────────────────┐
                          │    RANDOMIZATION      │
                          │  W ⊥ (Y(1), Y(0))    │
                          │  Selection bias = 0   │
                          │  Simple diff-in-means │
                          └──────────┬───────────┘
                                     │
                    "What if we can't randomize?"
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
              ▼                      ▼                      ▼
     Matching / IPW          DiD / Synth Control      IV / RDD
     (Part 4)                (Part 3)                 (Part 3)
              │                      │                      │
              ▼                      ▼                      ▼
     Assume:                 Assume:                 Assume:
     No unmeasured           Parallel trends         Valid instrument
     confounders                                     (exclusion, etc.)
```

Each design replaces the unconditional independence that randomization provides with a weaker but (hopefully) defensible assumption about the data-generating process.

**The strength of a causal analysis is determined by the plausibility of its assumptions — not by the sophistication of its estimator.** A simple difference-in-means in a well-designed RCT is more credible than the most advanced doubly robust estimator applied to a confounded observational dataset.

> **Pitfall**: "Controlling for covariates" does not make an observational study into an RCT. Regression adjustment removes *measured* confounding; it cannot address *unmeasured* confounding. The gap between a well-designed RCT and a well-analyzed observational study is the gap between $W \perp\!\!\!\perp (Y(1), Y(0))$ (unconditional) and $(Y(1), Y(0)) \perp\!\!\!\perp W \mid \mathbf{X}$ (conditional, and unverifiable).

---

## Summary

This post established the potential outcomes framework — the formal language underlying all of modern causal inference. The key ideas:

1. **Two potential outcomes per unit**: $Y_i(1)$ and $Y_i(0)$ exist for every unit; only one is observed (the fundamental problem).

2. **Four causal estimands**: ATE, ATT, CATE, and LATE target the same quantity — $Y(1) - Y(0)$ — but average it over different subpopulations. They coincide only under constant effects.

3. **Selection bias decomposition**: The naive comparison = causal effect + selection bias. The bias reflects baseline differences between treatment groups, and it can reverse the apparent sign of the effect.

4. **Three identifying assumptions**: SUTVA (well-defined potential outcomes), Ignorability (no unmeasured confounders), and Positivity (overlap). Together, they bridge the gap between potential outcomes and observed data.

5. **Randomization as mechanism**: An RCT mechanically satisfies ignorability and positivity, zeroing out selection bias. Every observational method substitutes a weaker assumption.

---

## What Comes Next

Part 2 introduces an alternative — and complementary — language for causation: *structural causal models* (SCMs) and *directed acyclic graphs* (DAGs). Where potential outcomes define *what* a causal effect is, SCMs encode *why* variables are related and provide graphical tools ($d$-separation, the do-calculus) for determining *when* a causal effect is identifiable.

---

## Key Notation Reference

For convenience, the core notation introduced in this post:

| Symbol | Meaning |
|--------|---------|
| $i = 1, \dots, N$ | Unit index |
| $W_i \in \{0, 1\}$ | Binary treatment (1 = treated, 0 = control) |
| $Y_i$ | Observed outcome |
| $Y_i(w)$ | Potential outcome under treatment $w$ |
| $\mathbf{X}_i \in \mathbb{R}^p$ | Pre-treatment covariates |
| $\tau_i = Y_i(1) - Y_i(0)$ | Individual causal effect (unobservable) |
| $\tau = E[Y_i(1) - Y_i(0)]$ | Average Treatment Effect (ATE) |
| $\tau_{\text{ATT}}$ | Average Treatment Effect on the Treated |
| $\tau(\mathbf{x})$ | Conditional Average Treatment Effect (CATE) |
| $\tau_{\text{LATE}}$ | Local Average Treatment Effect (compliers) |
| $e(\mathbf{x}) = P(W_i = 1 \vert \mathbf{X}_i = \mathbf{x})$ | Propensity score |

---

## References

- Holland, P. W. (1986). Statistics and causal inference. *Journal of the American Statistical Association*, 81(396), 945-960.
- Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics, Social, and Biomedical Sciences: An Introduction*. Cambridge University Press.
- Rubin, D. B. (1974). Estimating causal effects of treatments in randomized and nonrandomized studies. *Journal of Educational Psychology*, 66(5), 688-701.
- Angrist, J. D., Imbens, G. W., & Rubin, D. B. (1996). Identification of causal effects using instrumental variables. *Journal of the American Statistical Association*, 91(434), 444-455.
- Hernán, M. A., & Robins, J. M. (2020). *Causal Inference: What If*. Chapman & Hall/CRC.
- Lin, W. (2013). Agnostic notes on regression adjustments to experimental data: Reexamining Freedman's critique. *The Annals of Applied Statistics*, 7(1), 295-318.
