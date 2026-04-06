---
title: "Causal Inference Part 3: Identification — From Design to Estimand"
date: 2026-04-06 13:58:00 +0900
categories: [AI & ML, ML Theory]
tags: [causal-inference, identification, did, instrumental-variables, rdd, synthetic-control]
math: true
---

**Causal Inference — From First Principles to Automated Reasoning**

*This is Part 3 of an 8-part series.*

- **Part 0:** [Beyond Correlation — Why Causal Inference?](/posts/causal-inference-part0-beyond-correlation/)
- **Part 1:** [The Language of Causation: Potential Outcomes](/posts/causal-inference-part1-potential-outcomes/)
- **Part 2:** [Graphs and Interventions — Structural Causal Models](/posts/causal-inference-part2-structural-causal-models/)
- **Part 3:** Identification — From Design to Estimand (this post)
- **Part 4:** [Estimation — From Estimand to Estimate](/posts/causal-inference-part4-estimation/)
- **Part 5:** [Heterogeneous Effects — Who Benefits?](/posts/causal-inference-part5-heterogeneous-effects/)
- **Part 6:** [The Causal Pipeline — From Data to Decision](/posts/causal-inference-part6-causal-pipeline/)
- **Part 7:** [The Causal Agent — Automated Causal Reasoning](/posts/causal-inference-part7-causal-agent/)

---

> **TL;DR**: Identification is the step where you declare what you believe about the world — and those beliefs determine whether data can speak about causes at all. This post surveys five canonical identification strategies (DiD, IV, RDD, Synthetic Control, and RCT), states each strategy's identifying assumption, and provides a decision tree for choosing the right design. If you read only one post in this series, read this one.

---

## Hook: The Hardest Part Is Not the Code

Identification is the hardest part of causal inference. Not estimation, not coding — identification.

It is the step where you declare *what you believe about the world* that allows data to speak about causes. Get identification right and a simple regression recovers the truth. Get it wrong and the most sophisticated machine learning pipeline produces a confident, precise, wrong number.

Every empirical debate in economics, epidemiology, and policy evaluation ultimately reduces to a disagreement about identification. "Are the parallel trends plausible?" "Is the instrument truly excludable?" "Is there manipulation at the cutoff?" These are not statistical questions. They are questions about the structure of the world — and no amount of data can settle them.

**This post is the most practically important in the series.** It equips you to choose a research design, state its identifying assumption, and defend that assumption to a skeptical reviewer. Parts 4-6 handle the estimation machinery, but the machinery is worthless without a valid design.

---

## 1. What Is Identification?

**Identification and estimation are fundamentally different activities.** Conflating them is the single most common mistake in applied causal inference.

- **Identification** asks: *Can* the causal effect be expressed as a function of observable quantities?
- **Estimation** asks: *How* do we compute that function from a finite sample?

Identification is a property of the model and assumptions. Estimation is a property of the data and algorithm. A causal effect can be identified but poorly estimated (small sample). A causal effect can be unidentified — and then no estimator, no matter how sophisticated, can recover it.

> **Core principle**: If you cannot identify the effect, no estimator can save you.

Recall the four-step pipeline from [Part 0](/posts/causal-inference-part0-beyond-correlation/):

```
┌─────────┐     ┌────────────┐     ┌────────────┐     ┌──────────┐
│  MODEL  │────→│  IDENTIFY  │────→│  ESTIMATE  │────→│  REFUTE  │
│         │     │            │     │            │     │          │
│ State   │     │ Map causal │     │ Compute    │     │ Stress-  │
│ causal  │     │ estimand → │     │ from data  │     │ test     │
│ assump- │     │ statistical│     │            │     │ results  │
│ tions   │     │ estimand   │     │            │     │          │
└─────────┘     └────────────┘     └────────────┘     └──────────┘
```

**This post lives entirely in the second box.** We take the causal estimands defined in Parts 1-2 ($\tau$, $\tau_{\text{ATT}}$, $\tau_{\text{LATE}}$) and ask under what assumptions each can be written as a function of observable joint distributions.

Five identification strategies dominate applied work. Each makes a different structural assumption; each identifies a different estimand; each requires different data. The rest of this post surveys them in order of practical relevance.

| # | Strategy | Core Assumption | Estimand | Data Requirement |
|:-:|----------|----------------|----------|-----------------|
| 1 | **DiD** | Parallel trends | $\tau_{\text{ATT}}$ | Panel or repeated cross-section |
| 2 | **IV** | Exclusion restriction | $\tau_{\text{LATE}}$ | Instrument + cross-section |
| 3 | **RDD** | Continuity at cutoff | Local $\tau$ at $c$ | Running variable + outcome |
| 4 | **Synthetic Control** | Weighted donor match | $\tau$ for treated unit | Long pre-treatment panel |
| 5 | **RCT** | Randomization | $\tau$ (ATE) | Experimental data |

---

## 2. Difference-in-Differences (DiD)

DiD is the workhorse of policy evaluation, applied microeconomics, and — increasingly — computational science impact studies. **It compares the change in outcomes over time between a treated group and a control group, attributing the differential change to the treatment.**

### 2.1 The 2x2 Setup

The simplest DiD design involves two groups and two time periods:

- **Groups**: treated ($W_i = 1$) and control ($W_i = 0$)
- **Periods**: pre-treatment ($t = 0$) and post-treatment ($t = 1$)
- **Treatment timing**: the treated group receives treatment between periods 0 and 1; the control group never receives treatment

The two-way fixed effects (TWFE) regression takes the form:

$$Y_{it} = \alpha_i + \gamma_t + \delta \cdot (W_i \times \text{Post}_t) + \epsilon_{it}$$

where $\alpha_i$ absorbs permanent unit-level differences, $\gamma_t$ absorbs common time shocks, and $\delta$ is the DiD estimand — the causal effect of treatment under the identifying assumption.

**The coefficient $\delta$ captures the treatment effect because it differences out both time-invariant confounders (via $\alpha_i$) and common trends (via $\gamma_t$).** What remains is the differential change attributable to treatment.

The DiD estimand can be written without regression notation as:

$$\hat{\delta}_{\text{DiD}} = \underbrace{(\bar{Y}_{1,\text{post}} - \bar{Y}_{1,\text{pre}})}_{\text{change in treated}} - \underbrace{(\bar{Y}_{0,\text{post}} - \bar{Y}_{0,\text{pre}})}_{\text{change in control}}$$

"Subtract the control group's change from the treated group's change."

### 2.2 The Identifying Assumption

> **Assumption 3.1 (Parallel Trends)**: In the absence of treatment, the average outcome for the treated group would have changed by the same amount as the average outcome for the control group:
>
> $$E[Y_{it}(0) \mid W_i = 1] - E[Y_{i,t-1}(0) \mid W_i = 1] = E[Y_{it}(0) \mid W_i = 0] - E[Y_{i,t-1}(0) \mid W_i = 0]$$
>
> Equivalently: the counterfactual trend for the treated group equals the observed trend for the control group.

**Parallel trends requires parallel *trends*, not parallel *levels*.** The treated and control groups can have permanently different outcome levels — a hospital that publishes 50 papers per year versus one that publishes 10. The fixed effect $\alpha_i$ absorbs these level differences. What the assumption requires is that *absent treatment*, both groups would have experienced the same *change* over time.

This distinction is critical. Researchers sometimes reject a DiD design because the two groups "look different." But DiD does not require the groups to look similar in levels — only in trends.

### 2.3 Numeric Example

Consider two research labs. Lab A (treated) adopts AlphaFold in mid-2021. Lab B (control) does not.

|                    | Lab A (treated) | Lab B (control) | Difference |
|:------------------:|:---------------:|:---------------:|:----------:|
| **Pre (2020)**     | 12 papers       | 8 papers        | 4          |
| **Post (2022)**    | 20 papers       | 11 papers       | 9          |
| **Change**         | +8              | +3              | —          |

$$\hat{\delta}_{\text{DiD}} = (20 - 12) - (11 - 8) = 8 - 3 = 5 \text{ papers}$$

Under parallel trends, Lab A would have increased by 3 papers (like Lab B) without AlphaFold. The additional 5 papers are attributed to the treatment.

### 2.4 DiD Diagram: Parallel Trends

```
Papers
  │
20│                              ● Lab A (observed, treated)
  │                           ╱
  │                        ╱
  │                     ╱
15│                  ╱
  │               ╱                    ← treatment effect = 5
  │            ╱
  │         ............● Lab A (counterfactual, no treatment)
12│        ●╱...........
  │      ╱ .        ╱
  │    ╱  .      ╱  ← parallel trend
11│   ╱  .    ● Lab B (observed, control)
  │  ╱  .  ╱
  │ ╱  .╱
 8│●..╱
  │╱
  └──────────────────────────────────── Time
       2020          2021          2022
              (Pre)   ▲   (Post)
                  Treatment
```

The dashed line shows the counterfactual path for Lab A under parallel trends. The vertical gap between the observed and counterfactual outcomes for Lab A in 2022 equals $\delta = 5$.

### 2.5 Event Study Extension

When multiple pre- and post-treatment periods are available, the event study specification replaces the single $\delta$ with period-specific coefficients:

$$Y_{it} = \alpha_i + \gamma_t + \sum_{k \neq -1} \delta_k \cdot (W_i \times \mathbb{1}[t = k]) + \epsilon_{it}$$

Here $k$ indexes event time relative to treatment adoption. The period $k = -1$ (one period before treatment) is the reference category, so all $\delta_k$ are measured relative to it.

**The event study serves two purposes:**

1. **Pre-trends test**: the coefficients $\delta_k$ for $k < -1$ should be approximately zero and statistically insignificant. If they are not, the parallel trends assumption is suspect.
2. **Dynamic effects**: the coefficients $\delta_k$ for $k > 0$ trace how the treatment effect evolves over time — immediate, delayed, or fading.

A typical event study plot looks like:

- Pre-treatment coefficients hovering near zero (supporting parallel trends)
- A jump at $k = 0$ (treatment onset)
- Possibly growing or shrinking effects for $k > 0$

### 2.6 The TWFE Critique: When the Standard Estimator Fails

**The canonical TWFE regression can produce severely biased estimates when treatment timing varies across units.** This insight, developed independently by three research groups, has reshaped applied practice since 2020.

The problem arises with staggered adoption — different units adopt treatment at different times. In this setting, TWFE implicitly uses *already-treated* units as controls for *newly-treated* units. When treatment effects are heterogeneous across groups or over time, this comparison can assign negative weights to some group-time ATTs, producing an estimate that does not correspond to any meaningful causal parameter.

| Paper | Key Finding |
|-------|------------|
| Goodman-Bacon (2021) | Decomposed TWFE into a weighted average of all possible 2x2 DiDs; showed some weights can be negative |
| Callaway & Sant'Anna (2021) | Proposed group-time ATT $\text{ATT}(g, t)$ — estimate effects separately for each adoption cohort, then aggregate |
| de Chaisemartin & D'Haultfoeuille (2020) | Showed TWFE weights can be negative under heterogeneous effects; proposed robust estimator |

**Practical recommendation**: for staggered adoption designs, use the Callaway-Sant'Anna estimator (or equivalent) rather than TWFE. Report the group-time ATTs and their aggregation. TWFE remains valid when treatment timing is uniform (the simple 2x2 case) or when effects are homogeneous.

> **Pitfall**: Running `Y ~ unit_FE + time_FE + treated*post` in a staggered-adoption setting and interpreting the coefficient as ATT is incorrect when treatment effects vary. This was standard practice until 2020. Check your treatment timing structure before choosing an estimator.

---

## 3. Instrumental Variables (IV)

**IV identifies a causal effect by exploiting an external source of variation that shifts treatment but does not directly affect the outcome.** It is the strategy of choice when treatment assignment is endogenous and no natural experiment (DiD, RDD) is available.

### 3.1 Setup and Assumptions

The goal is to estimate the causal effect of $W_i$ on $Y_i$, but an unobserved confounder $U$ jointly affects both. Direct regression of $Y$ on $W$ is biased. IV introduces a variable $Z_i$ — the instrument — that affects $Y_i$ only through its effect on $W_i$.

> **Assumption 3.2 (IV Conditions)**:
>
> (a) **Relevance**: $\text{Cov}(Z_i, W_i) \neq 0$ — the instrument affects treatment.
>
> (b) **Exclusion**: $Z_i$ affects $Y_i$ only through $W_i$ — no direct path from $Z$ to $Y$.
>
> (c) **Independence**: $Z_i \perp\!\!\!\perp (Y_i(w), W_i(z))$ — the instrument is as good as randomly assigned (unconditionally or conditional on covariates).

### 3.2 The IV DAG

```
         Z ──────→ W ──────→ Y
                   ▲          ▲
                   │          │
                   └─── U ────┘
                   (unobserved)

Z affects W (relevance)             ✓ required
Z does not directly affect Y (exclusion)   ✓ required
Z is independent of U (independence)       ✓ required
```

**The instrument creates exogenous variation in treatment.** By isolating this exogenous variation, IV recovers the causal effect despite the confounding by $U$.

### 3.3 Two-Stage Least Squares (2SLS)

The standard estimation procedure is 2SLS:

**Stage 1** — regress treatment on the instrument:

$$W_i = \pi_0 + \pi_1 Z_i + \nu_i$$

Obtain the fitted values $\hat{W}_i = \hat{\pi}_0 + \hat{\pi}_1 Z_i$.

**Stage 2** — regress the outcome on the fitted treatment:

$$Y_i = \beta_0 + \beta_1 \hat{W}_i + \eta_i$$

The coefficient $\hat{\beta}_1$ is the IV estimate. It uses only the variation in $W$ that is driven by $Z$ — the exogenous part.

For a single instrument and single treatment, the IV estimator reduces to the Wald estimator:

$$\hat{\tau}_{\text{IV}} = \frac{\text{Cov}(Z_i, Y_i)}{\text{Cov}(Z_i, W_i)} = \frac{\text{reduced form}}{\text{first stage}}$$

"The causal effect equals the total effect of the instrument on the outcome, scaled by the instrument's effect on treatment."

### 3.4 LATE: What IV Actually Estimates

**IV does not estimate the ATE.** Under the assumptions above plus monotonicity (no defiers), IV estimates the Local Average Treatment Effect (LATE):

$$\tau_{\text{LATE}} = E[Y_i(1) - Y_i(0) \mid \text{complier}]$$

where compliers are units whose treatment status is changed by the instrument: $W_i(1) > W_i(0)$.

This is the landmark result of Imbens & Angrist (1994). It means the IV estimate applies only to a subpopulation — the compliers — not to the full population. For policy, the LATE may be exactly what you want (the effect on people who would respond to the policy lever). For science, it requires careful interpretation.

| Subpopulation | Definition | Example (draft lottery → military → earnings) |
|:-------------|:-----------|:----------------------------------------------|
| **Complier** | $W(1) = 1, W(0) = 0$ | Serves if drafted, does not serve otherwise |
| **Always-taker** | $W(1) = 1, W(0) = 1$ | Volunteers regardless of lottery |
| **Never-taker** | $W(1) = 0, W(0) = 0$ | Avoids service regardless |
| **Defier** | $W(1) = 0, W(0) = 1$ | Assumed away by monotonicity |

### 3.5 Weak Instruments

**When the instrument is weakly correlated with treatment, IV estimates are biased toward the OLS estimate and have inflated standard errors.** The classic diagnostic is the first-stage F-statistic:

- $F > 10$: historically considered adequate (Staiger & Stock, 1997)
- $F > 104$: more conservative threshold for 5% worst-case bias (Lee et al., 2022)
- Weak instrument robust inference: Anderson-Rubin test, conditional likelihood ratio test

> **Pitfall**: A first-stage F-statistic of 8 does not mean "almost valid." Weak instruments produce estimates that are biased, have non-normal distributions, and whose confidence intervals have incorrect coverage. Do not use 2SLS with weak instruments — use weak-instrument-robust methods instead.

---

## 4. Regression Discontinuity Design (RDD)

**RDD exploits a sharp rule that assigns treatment based on whether a running variable crosses a known cutoff.** Near the cutoff, units are "as-if randomly" assigned, creating a local experiment embedded in observational data.

### 4.1 Setup

- **Running variable** $R_i$: a continuous score that determines treatment assignment (test score, income, age, vote share)
- **Cutoff** $c$: the threshold value
- **Treatment rule**: $W_i = \mathbb{1}[R_i \geq c]$ (sharp RDD)

Units just above and just below the cutoff are nearly identical in all observed and unobserved characteristics — they differ only in whether they received treatment. This makes the local comparison at the cutoff highly credible.

### 4.2 The Identifying Assumption

> **Assumption 3.3 (Continuity)**: The conditional expectation functions $E[Y_i(0) \mid R_i = r]$ and $E[Y_i(1) \mid R_i = r]$ are continuous in $r$ at the cutoff $c$.
>
> "Potential outcomes do not jump at the cutoff. Any jump in the observed outcome must be due to treatment."

This assumption fails if units can precisely manipulate the running variable to sort above or below the cutoff (e.g., precisely choosing a test score). The McCrary (2008) density test checks for bunching at the cutoff — a sign of manipulation.

### 4.3 Sharp RDD Estimand

The sharp RDD estimand is:

$$\tau_{\text{RDD}} = \lim_{r \downarrow c} E[Y_i \mid R_i = r] \;-\; \lim_{r \uparrow c} E[Y_i \mid R_i = r]$$

"The causal effect at the cutoff equals the discontinuity in the conditional expectation of the outcome."

This is a limit of two one-sided regressions evaluated at the cutoff. In practice, researchers fit local polynomial regressions within a bandwidth $h$ of the cutoff:

- Fit $E[Y \mid R]$ separately on each side of $c$
- Extrapolate each fit to $c$
- The difference is the RDD estimate

### 4.4 Fuzzy RDD

In many settings, crossing the cutoff does not deterministically assign treatment — it merely changes the *probability* of treatment. This is the fuzzy RDD:

$$\tau_{\text{fuzzy}} = \frac{\lim_{r \downarrow c} E[Y_i \mid R_i = r] - \lim_{r \uparrow c} E[Y_i \mid R_i = r]}{\lim_{r \downarrow c} E[W_i \mid R_i = r] - \lim_{r \uparrow c} E[W_i \mid R_i = r]}$$

**Fuzzy RDD is IV at the cutoff.** The instrument is $Z_i = \mathbb{1}[R_i \geq c]$, and the estimand is a LATE for compliers at the cutoff.

### 4.5 RDD: Conceptual Diagram

```
Outcome
(Y)
  │
  │           ○
  │        ○    ○
  │     ○  ────────── fitted line (above cutoff)
  │   ○  ○  ○
  │  ○ ○  ○     ○
  │ ○     ○
  │────────────────── } τ_RDD (discontinuity)
  │        ○ ○
  │   ○  ○
  │  ○ ○────────── fitted line (below cutoff)
  │ ○   ○  ○
  │○  ○    ○
  │ ○   ○
  │○  ○
  └──────────┬───────────── Running variable (R)
             c
          (cutoff)
```

The vertical gap between the two fitted lines at the cutoff equals $\tau_{\text{RDD}}$. Points are individual observations scattered around the conditional expectation functions.

### 4.6 Strengths and Limitations

**RDD is widely regarded as the most internally valid quasi-experimental design.** Units near the cutoff are comparable, making the identification assumption highly credible. Lee & Lemieux (2010) call it "close to a randomized experiment."

But this credibility comes at a cost:

- **Locality**: the effect is identified only at the cutoff. Extrapolation to units far from the cutoff requires additional assumptions.
- **Statistical power**: only observations near the cutoff contribute meaningfully, reducing effective sample size.
- **Bandwidth choice**: results can be sensitive to the bandwidth $h$. Imbens & Kalyanaraman (2012) and Calonico, Cattaneo & Titiunik (2014) provide optimal bandwidth selectors.

> **Pitfall**: Fitting a single polynomial across the entire range of $R$ and including a treatment dummy is not a valid RDD estimator. High-order polynomials can produce arbitrary effects. Use local linear regression within a narrow bandwidth around the cutoff.

---

## 5. Synthetic Control

**Synthetic control constructs a data-driven counterfactual for a single treated unit by optimally weighting untreated donor units.** It is designed for settings with one (or a few) treated units and a long pre-treatment panel — the regime where DiD is weakest.

### 5.1 Setup

- **One treated unit** (unit 1): e.g., a country that adopts a policy, a company that releases a tool
- **$J$ donor units** (units $2, \ldots, J+1$): untreated throughout the study period
- **Panel data**: outcomes observed for all units over $T$ periods, with treatment occurring at $T_0$

The synthetic control is a weighted average of donor outcomes:

$$\hat{Y}^{SC}_{1t} = \sum_{j=2}^{J+1} w_j \, Y_{jt}$$

where the weights $w_j \geq 0$ and $\sum_{j=2}^{J+1} w_j = 1$.

**The weights are chosen to minimize the pre-treatment discrepancy between the treated unit and the synthetic control:**

$$\min_{\mathbf{w}} \sum_{t=1}^{T_0} \left( Y_{1t} - \sum_{j=2}^{J+1} w_j \, Y_{jt} \right)^2 \quad \text{s.t.} \quad w_j \geq 0, \; \sum w_j = 1$$

If the synthetic control closely tracks the treated unit before treatment, it provides a credible counterfactual after treatment.

### 5.2 Estimand

The treatment effect for unit 1 at time $t > T_0$ is:

$$\hat{\tau}_{1t} = Y_{1t} - \hat{Y}^{SC}_{1t}$$

"The effect is the gap between the treated unit's observed outcome and the synthetic control's predicted outcome in the post-treatment period."

### 5.3 Inference: Placebo Tests

Standard inference (t-tests, confidence intervals) is not applicable because there is only one treated unit. Instead, synthetic control uses permutation-based inference:

1. Apply the synthetic control method to each *donor* unit in turn, pretending it was treated at $T_0$.
2. Compute the placebo effect for each donor.
3. Compare the treated unit's effect to the distribution of placebo effects.
4. If the treated unit's effect is extreme relative to the placebos, the effect is statistically significant.

**This is a Fisher-style exact test.** The p-value is the fraction of placebo effects at least as large as the treated unit's effect.

### 5.4 Key References and Extensions

| Contribution | Reference | Innovation |
|:------------|:----------|:-----------|
| Original method | Abadie & Gardeazabal (2003) | Basque Country terrorism study |
| Formalization | Abadie, Diamond & Hainmueller (2010) | California tobacco control |
| Synthetic DiD | Arkhangelsky et al. (2021) | Combines DiD and synthetic control; allows many treated units |
| Augmented SCM | Ben-Michael, Feller & Rothstein (2021) | Bias correction for imperfect pre-treatment fit |

> **Pitfall**: If the synthetic control cannot closely match the treated unit in the pre-treatment period, the post-treatment gap is uninterpretable. Always inspect the pre-treatment fit before reporting effects.

---

## 6. Choosing a Design — Decision Tree

**The choice of identification strategy is determined by the structure of the research setting, not by the researcher's statistical preferences.** The following decision tree encodes the key questions.

```
Is treatment randomly assigned?
├── Yes ──────────────────────────→ RCT
│                                   Estimand: τ (ATE)
│                                   Gold standard; no further ID needed
│
└── No → Is there a sharp threshold on a running variable?
    │
    ├── Yes ──────────────────────→ RDD (Section 4)
    │                               Estimand: local τ at cutoff
    │                               Check: no manipulation at cutoff
    │
    └── No → Is there a valid instrument?
        │
        ├── Yes ──────────────────→ IV (Section 3)
        │                           Estimand: τ_LATE (compliers)
        │                           Check: F > 10, exclusion credible
        │
        └── No → Is there a clear pre/post event?
            │
            ├── Yes → Comparable control group available?
            │   │
            │   ├── Yes ──────────→ DiD (Section 2)
            │   │                   Estimand: τ_ATT
            │   │                   Check: parallel pre-trends
            │   │
            │   └── No, but multiple
            │       donor units ──→ Synthetic Control (Section 5)
            │                       Estimand: τ for treated unit
            │                       Check: pre-treatment fit
            │
            └── No → Selection on observables plausible?
                │
                ├── Yes ──────────→ Matching / IPW / DML (Part 4)
                │                   Estimand: τ or τ_ATT
                │                   Check: unconfoundedness
                │
                └── No ───────────→ Bounds / partial identification
                                    Only ranges, not point estimates
```

**Walk the tree from top to bottom.** At each node, the question is about the research setting — what data and institutional features are available — not about statistical convenience. Always prefer the highest branch you can credibly claim.

---

## 7. Five Designs at a Glance

### Table 1: Comparison of Identification Strategies

| Design | Identifying Assumption | Estimand | Data Requirement | Internal Validity | Key Limitation |
|:-------|:----------------------|:---------|:----------------|:-----------------|:---------------|
| **RCT** | Randomization | $\tau$ (ATE) | Experimental data | Highest | Often infeasible; external validity concerns |
| **RDD** | Continuity at cutoff | Local $\tau$ at $c$ | Running variable + outcome | Very high | Local to cutoff; limited extrapolation |
| **IV** | Exclusion + relevance + independence | $\tau_{\text{LATE}}$ | Instrument + outcome | High (if valid) | LATE ≠ ATE; weak instruments dangerous |
| **DiD** | Parallel trends | $\tau_{\text{ATT}}$ | Panel or repeated cross-section | Moderate-high | Trends assumption untestable; staggered adoption issues |
| **Synthetic Control** | Weighted donor match | $\tau$ for treated unit | Long panel, few treated units | Moderate-high | Requires good pre-treatment fit; few treated units |

**Reading this table**: internal validity is ranked from top (RCT) to bottom. But internal validity is only one criterion. The right design is the one whose assumptions are most defensible in your specific setting. A credible DiD beats a questionable IV every time.

### Table 2: Mapping to the AlphaFold Impact Study

This series supports a concrete research proposal: measuring the causal impact of AlphaFold on structural biology research. The following table maps each dimension of the proposal to the appropriate identification strategy.

| Research Dimension | Design | Rationale | Treated | Control | Key Assumption |
|:-------------------|:-------|:----------|:--------|:--------|:---------------|
| Publication volume (structural biology vs. other fields) | DiD | Clear pre/post (Jul 2021), natural control group | Structural biology labs | Non-structural biology labs | Parallel publication trends pre-2021 |
| Methodological composition (experimental vs. computational) | DiD | Same temporal shock, field-level variation | Experimentally-focused labs | Computationally-focused labs | Parallel composition trends |
| Citation impact (AlphaFold-using vs. non-using papers) | IV | AlphaFold availability as instrument for structure coverage | Papers using AF structures | Papers using experimental structures | Exclusion: AF availability affects citations only through structure access |
| Research entry (new researchers entering structural biology) | RDD | Funding threshold + AF availability interaction | Labs above funding cutoff | Labs below funding cutoff | No manipulation of funding score; continuity |
| Country-level diffusion | Synthetic Control | Few treated "countries" (early-adopter nations) | Early-adopter nations | Donor pool of late/non-adopters | Pre-treatment trend match |

**No single design answers the full question.** The proposal requires a multi-design approach, with each design targeting a different dimension of impact. This is typical of large-scale impact evaluation.

---

## 8. Common Pitfalls and Cross-Cutting Themes

### 8.1 The Identification-Estimation Separation

**Never let the availability of an estimator drive your choice of identification strategy.** Researchers sometimes choose DiD because they have panel data and know how to run a fixed-effects regression. But having panel data does not make parallel trends hold. The data structure is necessary but not sufficient.

### 8.2 No Assumption Is Free

Every identification strategy requires an untestable assumption:

- DiD: parallel trends is about *counterfactual* trends — what would have happened without treatment. This is fundamentally unobservable.
- IV: the exclusion restriction cannot be tested from data. It is a claim about the absence of a direct effect.
- RDD: continuity is testable in principle (the McCrary test) but only partially — manipulation can be subtle.
- Synthetic Control: the quality of the pre-treatment fit is observable, but it does not guarantee the post-treatment counterfactual is correct.

**The role of the researcher is to argue — with institutional knowledge, domain expertise, and supporting evidence — that the assumption is plausible.** Formal tests (pre-trends, first-stage F, density tests) provide supporting evidence but can never prove an assumption true.

### 8.3 Estimand Clarity

Different designs identify different estimands. This matters:

| Design | Estimand | Population |
|:-------|:---------|:-----------|
| RCT | ATE: $E[Y(1) - Y(0)]$ | Full sample |
| DiD | ATT: $E[Y(1) - Y(0) \mid W = 1]$ | Treated units |
| IV | LATE: $E[Y(1) - Y(0) \mid \text{complier}]$ | Compliers only |
| RDD | Local ATE: $\lim_{r \to c} E[Y(1) - Y(0) \mid R = r]$ | Units at cutoff |
| Synthetic Control | Unit-specific: $Y_1(1) - Y_1(0)$ | Single treated unit |

**Comparing estimates across designs without accounting for estimand differences is a category error.** An IV estimate of 0.5 and a DiD estimate of 0.3 for the "same" treatment are not contradictory — they answer different questions about different subpopulations.

---

## 9. Summary and Bridge to Part 4

This post surveyed five identification strategies. The key takeaways:

1. **Identification comes before estimation.** No algorithm can fix a design that does not identify the effect.
2. **Every design requires a structural assumption.** State it explicitly. Defend it with institutional knowledge.
3. **Different designs identify different estimands.** Know what your design estimates — ATE, ATT, LATE, or local ATE — and interpret results accordingly.
4. **The TWFE critique is real.** For staggered DiD, use modern robust estimators (Callaway-Sant'Anna, de Chaisemartin-D'Haultfoeuille).
5. **Walk the decision tree.** Let the research setting — not statistical convenience — choose the design.

**[Part 4](/posts/causal-inference-part4-estimation/) takes identification as given and asks: how do we compute the estimate?** We cover matching, inverse probability weighting, doubly robust estimation, PPML for count outcomes, and double/debiased machine learning. The shift is from "can we identify the effect?" to "how do we estimate it efficiently and with valid inference?"

---

## References

- Abadie, A., & Gardeazabal, J. (2003). The economic costs of conflict: A case study of the Basque Country. *American Economic Review*, 93(1), 113-132.
- Abadie, A., Diamond, A., & Hainmueller, J. (2010). Synthetic control methods for comparative case studies. *Journal of the American Statistical Association*, 105(490), 493-505.
- Angrist, J. D., & Imbens, G. W. (1994). Identification and estimation of local average treatment effects. *Econometrica*, 62(2), 467-475.
- Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & Wager, S. (2021). Synthetic difference-in-differences. *American Economic Review*, 111(12), 4088-4118.
- Ben-Michael, E., Feller, A., & Rothstein, J. (2021). The augmented synthetic control method. *Journal of the American Statistical Association*, 116(536), 1789-1803.
- Callaway, B., & Sant'Anna, P. H. C. (2021). Difference-in-differences with multiple time periods. *Journal of Econometrics*, 225(2), 200-230.
- Calonico, S., Cattaneo, M. D., & Titiunik, R. (2014). Robust nonparametric confidence intervals for regression-discontinuity designs. *Econometrica*, 82(6), 2295-2326.
- Card, D., & Krueger, A. B. (1994). Minimum wages and employment: A case study of the fast-food industry in New Jersey and Pennsylvania. *American Economic Review*, 84(4), 772-793.
- de Chaisemartin, C., & D'Haultfoeuille, X. (2020). Two-way fixed effects estimators with heterogeneous treatment effects. *American Economic Review*, 110(9), 2964-2996.
- Goodman-Bacon, A. (2021). Difference-in-differences with variation in treatment timing. *Journal of Econometrics*, 225(2), 254-277.
- Holland, P. W. (1986). Statistics and causal inference. *Journal of the American Statistical Association*, 81(396), 945-960.
- Imbens, G. W., & Kalyanaraman, K. (2012). Optimal bandwidth choice for the regression discontinuity estimator. *Review of Economic Studies*, 79(3), 933-959.
- Imbens, G. W., & Angrist, J. D. (1994). Identification and estimation of local average treatment effects. *Econometrica*, 62(2), 467-475.
- Lee, D. S., & Lemieux, T. (2010). Regression discontinuity designs in economics. *Journal of Economic Literature*, 48(2), 281-355.
- Lee, D. S., McCrary, J., Moreira, M. J., & Porter, J. (2022). Valid t-ratio inference for IV. *American Economic Review*, 112(10), 3260-3290.
- McCrary, J. (2008). Manipulation of the running variable in the regression discontinuity design: A density test. *Journal of Econometrics*, 142(2), 698-714.
- Staiger, D., & Stock, J. H. (1997). Instrumental variables regression with weak instruments. *Econometrica*, 65(3), 557-586.
