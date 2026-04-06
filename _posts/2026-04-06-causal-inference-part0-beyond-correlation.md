---
title: "Causal Inference Part 0: Beyond Correlation — Why Causal Inference?"
date: 2026-04-06 13:55:00 +0900
categories: [AI & ML, ML Theory]
tags: [causal-inference, pearl, ladder-of-causation, simpson-paradox, prediction-vs-causation]
math: true
---

**Causal Inference — From First Principles to Automated Reasoning**

*This is Part 0 of an 8-part series.*

- **Part 0:** Beyond Correlation — Why Causal Inference? (this post)
- **Part 1:** [The Language of Causation — Potential Outcomes](/posts/causal-inference-part1-potential-outcomes/)
- **Part 2:** [Structural Causal Models](/posts/causal-inference-part2-structural-causal-models/)
- **Part 3:** [Identification](/posts/causal-inference-part3-identification/)
- **Part 4:** [Estimation](/posts/causal-inference-part4-estimation/)
- **Part 5:** [Heterogeneous Effects & Discovery](/posts/causal-inference-part5-heterogeneous-effects/)
- **Part 6:** [The Causal Pipeline](/posts/causal-inference-part6-causal-pipeline/)
- **Part 7:** [The Causal Agent](/posts/causal-inference-part7-causal-agent/)

---

## Hook: When Prediction Kills

A hospital builds a machine learning model to predict 30-day mortality among pneumonia patients. The model is excellent — AUC above 0.95 on held-out data. Among its strongest predictors: a history of asthma is associated with *lower* mortality risk.

The reason is simple and deadly. Asthmatic pneumonia patients had historically been triaged directly to the ICU, where aggressive treatment drove their mortality down. The model learned the *observational association* — asthma predicts survival — rather than the *causal mechanism* — ICU care causes survival. If a hospital used this model to *decide* who receives ICU care, asthmatic patients would be routed away from intensive treatment. The prediction is correct. The decision is fatal.

The causal structure behind this failure looks like:

```
  Asthma ──→ ICU triage ──→ Survival
    │                          ▲
    └──────────────────────────┘
         (direct risk factor)
```

The model captures the *net* association between asthma and survival (positive, because the ICU pathway dominates), but the *direct* causal effect of asthma on survival is negative — asthma makes pneumonia worse. The model conflates the two because it has no representation of the causal graph.

This is not a contrived thought experiment. Caruana et al. (2015) documented exactly this failure in a real clinical risk model deployed at a major medical center.

**Prediction answers "what will happen?" Causation answers "what should I do?" These are different questions with different math.**

The distinction matters whenever a model moves from *describing* the world to *changing* it:

- A drug trial analyst who conditions on post-treatment variables introduces bias that an observational correlation model would never flag.
- A policy researcher who estimates a program's effect by comparing participants to non-participants confuses selection with impact.
- A computational biologist who asks "did AlphaFold accelerate structural biology?" cannot answer by plotting citations over time, because the hardest proteins — the ones that needed AlphaFold most — attracted the most attention regardless.

Every one of these failures has the same root cause: **the question is causal, but the method is associational.** This post establishes why that gap exists, why it cannot be closed by collecting more data or fitting a better model, and what it takes to close it properly.

---

## 1. Three Questions You Cannot Answer with Regression

### 1.1 The Drug Question

A pharmaceutical company runs an observational study on a new drug. Among patients who received the drug, the survival rate is 60%. Among those who did not, it is 70%. Should the drug be abandoned?

Not necessarily. Suppose sicker patients are more likely to receive the drug (physicians prescribe it when standard therapy fails). Then the treated group is systematically sicker than the control group, and comparing raw survival rates conflates the drug's effect with the severity of illness.

**The variable we want — the causal effect of treatment — is entangled with the variable that determines who gets treated.**

Formally, let $W_i \in \{0, 1\}$ denote treatment and $Y_i$ the observed outcome for unit $i = 1, \dots, N$. A naive regression estimates:

$$E[Y_i \mid W_i = 1] - E[Y_i \mid W_i = 0]$$

This is the *associational difference in means*. It equals the causal effect only when treatment assignment is independent of potential outcomes — a condition that observational data almost never satisfy.

To see why, decompose the associational difference:

$$E[Y_i \mid W_i = 1] - E[Y_i \mid W_i = 0] = \underbrace{E[Y_i(1) - Y_i(0) \mid W_i = 1]}_{\text{ATT}} + \underbrace{E[Y_i(0) \mid W_i = 1] - E[Y_i(0) \mid W_i = 0]}_{\text{Selection bias}}$$

The first term is the average treatment effect on the treated (ATT) — the quantity we want. The second term is *selection bias*: the difference in baseline outcomes between those who received treatment and those who did not, in the absence of treatment. When sicker patients are more likely to receive the drug, $E[Y_i(0) \mid W_i = 1] < E[Y_i(0) \mid W_i = 0]$, and the selection bias is negative. This negative bias can make an effective drug ($\text{ATT} > 0$) look harmful in the raw comparison.

> **Assumption**: The decomposition above requires only the *consistency assumption* — that $Y_i = Y_i(W_i)$ — and is otherwise identity-level algebra. No causal assumptions are needed to write it. The problem is that neither term on the right-hand side is separately identifiable without further assumptions, because $Y_i(0)$ is unobserved for treated units. Part 1 develops this point formally as the *Fundamental Problem of Causal Inference*.

### 1.2 The Policy Question

A government introduces a job training program. Participants earn \$3,000 more per year than non-participants. Does the program work?

Again, not necessarily. People who voluntarily enroll in job training may be more motivated, better networked, or already on an upward trajectory. **Selection into treatment is itself caused by variables that also affect the outcome.**

This is the same confounding structure as the drug question, but in a social science context. Ordinary least squares (OLS) regression on $W_i$ absorbs the selection effect into the treatment coefficient unless every confounder is measured and correctly specified.

The classic study that exposed this problem is LaLonde (1986), who compared experimental estimates of a job training program (from a randomized trial) to non-experimental estimates (from regression on observational data). The regression estimates ranged from &minus;15,000 USD to +1,600 USD depending on the comparison group chosen — while the experimental benchmark was +886 USD. **The observational estimates were not just imprecise; they had the wrong sign.**

### 1.3 The Scientific Tool Question

Did AlphaFold accelerate protein structure determination? A natural approach: compare the rate of new structures deposited in the PDB before and after AlphaFold's release. But the proteins studied after AlphaFold may differ systematically from those studied before — in difficulty, in scientific interest, in the availability of complementary data.

**The difficulty of the target is a confounder: hard proteins attract both more attention and more tool adoption.**

The causal graph looks like:

```
Protein difficulty ──→ AlphaFold adoption
        │                     │
        │                     ▼
        └──────────────→ Scientific output
                         (structures solved,
                          publications, etc.)
```

A naive before-after comparison captures the combined effect of AlphaFold adoption *and* the changing composition of studied proteins. Disentangling the two requires a causal design — difference-in-differences with appropriate controls, for instance — which is exactly the kind of method Parts 3-4 will develop.

| Question | Treatment $W$ | Outcome $Y$ | Confounder $\mathbf{X}$ | Why regression fails |
|----------|:-------------:|:-----------:|:------------------------:|---------------------|
| Drug effect | Drug administered | Survival | Disease severity | Sicker patients get the drug |
| Policy effect | Program enrollment | Earnings | Motivation, prior skills | Self-selection into program |
| Tool effect | AlphaFold usage | Scientific output | Protein difficulty | Hard proteins attract both tool use and attention |

The common thread across all three: **each question asks what happens if we intervene, not what we observe.** The difference between $E[Y \mid W = 1]$ (what we see among the treated) and $E[Y(\text{do}(W = 1))]$ (what would happen if we forced treatment) is exactly the gap that causal inference is designed to close.

> **Pitfall**: Adding more control variables to a regression does not guarantee that confounding is removed. If a variable is a *collider* — a common effect of treatment and outcome — conditioning on it introduces new bias rather than removing old bias. The correct adjustment set depends on the causal structure, not on statistical significance or $R^2$ improvement. (Part 2 makes this precise with $d$-separation.)

---

## 2. Pearl's Ladder of Causation

Judea Pearl organized all statistical and causal questions into a three-level hierarchy. **Each level is strictly more informative than the one below, and no amount of data at a lower level can answer a question at a higher level without additional assumptions.**

This is the single most important conceptual framework in the series. Every method we develop in Parts 1-6 is a tool for climbing this ladder.

### 2.1 Level 1 — Association

$$P(Y \mid X = x)$$

This is the domain of standard machine learning: observe features, predict outcomes. "Patients who take the drug have higher survival." "Stocks that rose yesterday tend to rise today." "Proteins with high pLDDT scores have accurate folds."

Association answers the question: **"Given that I observe $X = x$, what do I expect for $Y$?"**

All of supervised ML — linear regression, random forests, deep neural networks — operates at this level. The data-generating process is treated as a black box; we care only about the joint distribution $P(X, Y)$ or the conditional $P(Y \mid X)$.

| Property | Level 1 (Association) |
|----------|----------------------|
| Typical question | "What is $Y$ given $X = x$?" |
| Mathematical object | $P(Y \mid X)$ |
| Data requirement | Observational data |
| Answers intervention? | No |
| Answers counterfactual? | No |

### 2.2 Level 2 — Intervention

$$P(Y \mid \mathrm{do}(X = x))$$

This is the domain of causal inference proper. The $\mathrm{do}(\cdot)$ operator, introduced by Pearl (1995), denotes an *intervention* — physically setting $X$ to the value $x$, regardless of the natural process that would otherwise determine $X$.

**The critical distinction: $P(Y \mid X = x)$ conditions on *observing* $X = x$; $P(Y \mid \mathrm{do}(X = x))$ conditions on *forcing* $X = x$.**

Why do these differ? Because in the observational case, $X = x$ carries information about the confounders that caused $X$ to take that value. When we intervene, we sever the causal arrows *into* $X$ and eliminate that confounding information.

> "Conditioning on an observation is not the same as intervening to set a value. The difference is not philosophical but mathematical: the two operations correspond to different probability distributions."
> — Pearl, *Causality* (2009), Ch. 3

Consider a concrete example. Among patients who *choose* to take a statin, cholesterol is lower. This is $P(\text{cholesterol} \mid \text{statin} = 1)$. But these patients may also exercise more, eat better, and see their doctors more frequently. The observed association between statin use and low cholesterol is a mixture of the drug's pharmacological effect and the healthy behaviors that correlate with choosing to take a drug.

Now consider an RCT where patients are *randomly assigned* to statin or placebo. The randomization severs the link between statin assignment and lifestyle. The resulting estimate is $P(\text{cholesterol} \mid \mathrm{do}(\text{statin} = 1))$ — the interventional distribution, free of confounding.

**A randomized controlled trial (RCT) is the gold standard precisely because randomization implements $\mathrm{do}(X = x)$ physically** — the coin flip ensures that $X$ is set independently of all confounders.

The methods in Parts 3-4 of this series — difference-in-differences, instrumental variables, regression discontinuity — are all strategies for recovering $P(Y \mid \mathrm{do}(X = x))$ from observational data under stated assumptions. Each method imposes a different set of structural or design-based assumptions to compensate for the absence of randomization.

### 2.3 Level 3 — Counterfactual

$$P(Y_x \mid X = x', Y = y')$$

This is the domain of individual-level "what if" reasoning. "This patient died without the drug. Would they have survived *had we given it*?" "This protein was solved by crystallography. Would AlphaFold have predicted its structure accurately?"

**Counterfactual queries are about a specific unit that was observed in one state, asking what would have happened in an alternative state.**

Note the conditioning on the right-hand side: $X = x', Y = y'$. The query takes as given that we *observed* the unit in state $(X = x', Y = y')$ and asks what $Y$ *would have been* under a different value of $X$. This is fundamentally harder than the interventional query because it requires reasoning about a specific unit's unobserved noise terms, not just population-level distributions.

Answering Level 3 questions requires a *structural causal model* (SCM) — a set of equations that specify how each variable is generated from its parents and exogenous noise. The three-step procedure is:

1. **Abduction**: Use the observed evidence $(X = x', Y = y')$ to infer the exogenous noise $U$ for this unit.
2. **Action**: Modify the structural equation to set $X = x$ (the intervention).
3. **Prediction**: Compute $Y$ under the modified model with the inferred $U$.

Parts 1 and 2 build this machinery in detail. The key takeaway for now: counterfactual reasoning is not speculation — it is a well-defined mathematical operation, but one that requires stronger assumptions than intervention.

| Level | Question type | Formal object | What it requires |
|:-----:|--------------|:-------------:|-----------------|
| 1 | Association | $P(Y \mid X)$ | Data only |
| 2 | Intervention | $P(Y \mid \mathrm{do}(X = x))$ | Causal graph or experiment |
| 3 | Counterfactual | $P(Y_x \mid X = x', Y = y')$ | Full structural model |

### 2.4 The Ladder as a Diagram

```
Level 3: Counterfactual     "Would Y have been y' had X been x?"
  P(Y_x | X=x', Y=y')       Requires: Structural model
         ▲
         │ strictly stronger
         │
Level 2: Intervention       "What happens to Y if I set X = x?"
  P(Y | do(X=x))            Requires: Causal graph or experiment
         ▲
         │ strictly stronger
         │
Level 1: Association        "What is Y given that I observe X = x?"
  P(Y | X=x)                Requires: Data only
```

The arrows are one-directional. A structural model can answer any interventional or associational query. A causal graph can answer interventional and associational queries but not all counterfactuals. Observational data alone can answer only associational queries.

The "strictly stronger" claim is not informal. Bareinboim and Pearl (2016) proved that there exist interventional quantities that are *provably* not computable from any observational distribution — regardless of sample size, model complexity, or computational resources — without additional causal assumptions. Similarly, there exist counterfactual quantities that are not computable from interventional data alone.

> "The do-operator is not a notational trick. It represents a mathematical operation — graph surgery — that has no counterpart in classical probability theory."
> — Pearl, *The Book of Why* (2018), Ch. 1

### 2.5 The No Free Lunch Theorem of Causation

The hierarchy is not merely a pedagogical convenience. It reflects a mathematical impossibility result.

> **Assumption 0.1 (No Free Lunch in Causation)**: Observational data alone — however large — cannot identify causal effects without structural or design-based assumptions that go beyond the data.

This is worth pausing on. In supervised learning, more data generally improves prediction. In causal inference, more data improves *precision* (narrower confidence intervals) but cannot address *identification* (whether the estimand is correct). **A biased estimator does not become unbiased as $N \to \infty$; it converges to the wrong number with ever-greater confidence.**

An analogy clarifies the distinction. Consider estimating the height of a building from its shadow. With more measurements of the shadow (more data), you get a more precise estimate of the shadow's length. But to convert shadow length to building height, you need the angle of the sun — a piece of structural knowledge that no amount of shadow data can provide. In causal inference, the "angle of the sun" is the causal graph or design-based assumption. Without it, the most precise shadow measurement in the world gives you the wrong building height.

The practical consequence: every causal analysis must state its identifying assumptions explicitly. These assumptions are untestable from the data alone — they must be defended on substantive, domain-specific grounds. The series will return to this point repeatedly.

| Prediction (Level 1) | Causal Inference (Levels 2-3) |
|----------------------|-------------------------------|
| More data $\Rightarrow$ better answers | More data $\Rightarrow$ more precise *same* answer (biased or not) |
| Model-agnostic metrics (AUC, RMSE) | Sensitivity to untestable assumptions |
| Cross-validation for evaluation | Placebo tests, refutation checks |
| Features selected by predictive power | Variables selected by causal structure |
| Black-box models welcome | Structural transparency required |

---

## 3. Simpson's Paradox — The Canonical Warning

Simpson's Paradox is the clearest demonstration that causal questions cannot be answered by data alone. **The paradox is not statistical — it is causal.** The same dataset supports two opposite conclusions, and the correct one depends on the causal graph, not on the numbers.

### 3.1 The UC Berkeley Admissions Example

In 1973, UC Berkeley's graduate admissions appeared to discriminate against women. The overall admission rate was:

| | Applied | Admitted | Rate |
|----------|--------:|--------:|-----:|
| **Men** | 8,442 | 3,738 | 44.3% |
| **Women** | 4,321 | 1,494 | 34.6% |

A 10-percentage-point gap, prima facie evidence of gender bias.

But when Bickel, Hammel, and O'Connell (1975) disaggregated by department, the pattern reversed. In four of the six largest departments, women were admitted at *higher* rates than men. The explanation: women disproportionately applied to departments with low overall admission rates (humanities, social sciences), while men disproportionately applied to departments with high overall admission rates (engineering, physical sciences).

Here is a simplified numerical example that reproduces the paradox with two departments:

| Department | Gender | Applied | Admitted | Rate |
|:----------:|--------|--------:|--------:|-----:|
| A (easy) | Men | 800 | 480 | **60%** |
| A (easy) | Women | 200 | 120 | **60%** |
| B (hard) | Men | 200 | 40 | **20%** |
| B (hard) | Women | 800 | 180 | **22.5%** |
| **Overall** | **Men** | **1,000** | **520** | **52%** |
| **Overall** | **Women** | **1,000** | **300** | **30%** |

In both departments, women are admitted at equal or higher rates than men. Yet the overall rate for women (30%) is far below that for men (52%). **The aggregated data reverses the department-level conclusion because the confounding variable — department choice — is correlated with both gender and admission difficulty.**

Verify the arithmetic to build intuition:

- **Men**: $(800 \times 0.60) + (200 \times 0.20) = 480 + 40 = 520$ admitted out of 1,000 $\Rightarrow$ 52%
- **Women**: $(200 \times 0.60) + (800 \times 0.225) = 120 + 180 = 300$ admitted out of 1,000 $\Rightarrow$ 30%

The reversal happens because 80% of women apply to the hard department (20-22.5% admission) while 80% of men apply to the easy department (60% admission). The marginal totals are dominated by the department with the most applicants from each gender — and those departments have very different base rates.

### 3.2 Two DAGs, Two Conclusions

The correct analysis depends on the causal graph. Consider two possible structures:

**DAG 1: Department is a confounder**

```
    Gender
    /    \
   ▼      ▼
Department → Admission
```

Here, gender influences department choice, and department influences admission. Department is a common cause (confounder) of the treatment-outcome pair if we think of "gender's effect on admission" as the causal question. To identify the direct effect of gender on admission, we must *condition on* department.

Under DAG 1, the correct analysis is the department-stratified one. The overall rate is misleading because it fails to block the confounding path through department.

**DAG 2: Department is a mediator**

```
Gender → Department → Admission
```

Here, gender affects admission *only through* department choice. If the causal question is "what is the total effect of gender on admission (including the pathway through department choice)?", then conditioning on department blocks the very pathway we want to measure.

Under DAG 2, the correct analysis is the *aggregated* one.

**DAG 3: Department is a collider**

```
Gender ──→ Admission
              ▲
Department ───┘
```

Here, gender and department independently affect admission, and department is a *collider* on the path Gender $\to$ Admission $\leftarrow$ Department. Conditioning on department opens a spurious association between gender and the unobserved factors that determine department choice. In this (admittedly less plausible) structure, conditioning on department would *introduce* bias rather than remove it.

The formal criterion for deciding which variables to condition on is called $d$-separation, and it is the centerpiece of Part 2. For now, the point is simpler: **three different causal graphs, applied to the same data, yield three different correct analyses.**

> **Pitfall**: Simpson's Paradox is often presented as a statistical curiosity with a clear "correct" answer (disaggregate). This is misleading. Whether to condition on a variable depends on its causal role — confounder, mediator, or collider — not on whether conditioning changes the estimate. **The data cannot tell you which DAG is correct. The causal graph is an assumption, not a discovery.**

### 3.3 The Lesson

Simpson's Paradox teaches three principles that recur throughout this series:

1. **Aggregation and disaggregation are causal operations.** Choosing which variables to condition on is a causal decision, not a statistical convenience.

2. **The same data can support opposite causal conclusions.** Without a causal graph, the data are ambiguous. No sample size resolves the ambiguity.

3. **Causal assumptions must come from outside the data.** Domain knowledge, experimental design, or structural theory — something must constrain the set of admissible DAGs before estimation begins.

| Scenario | Correct analysis | Why |
|----------|-----------------|-----|
| Department is a confounder (DAG 1) | Stratify by department | Blocks backdoor path |
| Department is a mediator (DAG 2) | Do not stratify | Preserves total effect pathway |
| Department is a collider | Do not condition | Conditioning opens spurious path |

---

## 4. Prediction vs. Causation — A Systematic Comparison

Before mapping the series roadmap, it is useful to crystallize the distinction between predictive and causal inference along multiple axes. **These are not different philosophies — they are different mathematical problems with different data requirements, different evaluation criteria, and different failure modes.**

| Axis | Prediction (ML) | Causal Inference |
|------|-----------------|-----------------|
| **Question** | "What will $Y$ be given $\mathbf{X}$?" | "What will $Y$ be if I set $W = 1$?" |
| **Formal target** | $E[Y \mid \mathbf{X}]$ | $E[Y(1) - Y(0)]$ or $P(Y \mid \mathrm{do}(W = 1))$ |
| **Key threat** | Overfitting (variance) | Confounding (bias) |
| **Evaluation** | Out-of-sample metrics (AUC, MSE) | Sensitivity analysis, placebo tests, refutation |
| **Role of DAG** | Optional (feature selection heuristic) | Essential (defines adjustment set) |
| **More data helps with** | Variance reduction, generalization | Precision only — bias persists |
| **When to use** | Forecasting, ranking, scoring | Treatment decisions, policy design, mechanism discovery |

> **Pitfall**: A common mistake is to train a predictive model, interpret its feature importance as causal effects, and recommend interventions on the most "important" features. This is invalid. A feature can be highly predictive because it is a *consequence* of the outcome (reverse causation), a *proxy* for a confounder, or a *collider* that absorbs variation from multiple sources. Predictive importance and causal importance are different quantities with different estimation strategies.

### When Do You Need Causal Inference?

A useful heuristic: **if your research question contains (or implies) any of the following verbs, you need causal methods, not predictive ones.**

| Verb | Example question | Why it is causal |
|------|-----------------|-----------------|
| *Cause* | "Does smoking cause cancer?" | Directly asks for a causal mechanism |
| *Affect* | "Does class size affect test scores?" | Asks about an interventional effect |
| *Prevent* | "Does the vaccine prevent infection?" | Asks about a counterfactual outcome |
| *Improve* | "Does the new catalyst improve yield?" | Implies comparison to a no-intervention baseline |
| *Would* | "Would this patient have survived with the drug?" | Explicitly counterfactual |
| *Should* | "Should we fund this program?" | Requires knowing the program's causal effect |

If the question uses verbs like *predict*, *forecast*, *classify*, or *rank*, standard ML is appropriate. The confusion arises when a prediction verb is used but a causal answer is desired — "predict which treatment works best" is a causal question disguised in predictive language.

### A Concrete Example: Same Data, Different Questions

Consider a dataset of 10,000 patients with two features — age and blood pressure — and an outcome — stroke within 5 years. A predictive modeler and a causal analyst receive the same data but ask different questions:

| | Predictive modeler | Causal analyst |
|---|---|---|
| **Question** | Which patients are at highest risk of stroke? | Does lowering blood pressure reduce stroke risk? |
| **Target** | $P(\text{stroke} \mid \text{age}, \text{BP})$ | $E[\text{stroke}(\mathrm{do}(\text{BP} = b))]$ |
| **Approach** | Train a classifier; maximize AUC | Specify a DAG; identify confounders; estimate causal effect |
| **Uses age as** | A predictive feature | A confounder that must be adjusted for |
| **Evaluation** | Held-out AUC = 0.82 | Sensitivity analysis to unmeasured confounding |
| **Actionable?** | Ranks patients; does not guide treatment | Guides treatment decision if assumptions hold |

The predictive model can be excellent (high AUC) yet completely silent on whether treating blood pressure reduces stroke. The causal analysis can answer the treatment question but requires assumptions that the predictive model does not need. **Neither is better — they answer different questions.**

---

## 5. What This Series Will Build

### 5.1 The Roadmap

The remaining seven posts build on this motivation in a specific order:

```
SERIES ARCHITECTURE

Part 0  Beyond Correlation (this post)
  │     WHY causal inference?
  │
  ├──→ Part 1  Potential Outcomes (Rubin Causal Model)
  │     │      WHAT is a causal effect? Y(1) - Y(0)
  │     │
  │     └──→ Part 2  Structural Causal Models (Pearl)
  │           │      HOW do graphs encode assumptions?
  │           │
  │           └──→ Part 3  Identification
  │                 │      WHEN can we estimate from observational data?
  │                 │      DiD, IV, RDD, Synthetic Control
  │                 │
  │                 └──→ Part 4  Estimation
  │                       │      HOW do we compute the effect?
  │                       │      Matching, IPW, DML
  │                       │
  │                       └──→ Part 5  Heterogeneous Effects & Discovery
  │                             │      WHO benefits most? CATE, Causal Forests
  │                             │
  │                             └──→ Part 6  Code Pipeline
  │                                   │      DoWhy + EconML end-to-end
  │                                   │
  │                                   └──→ Part 7  The Causal Inference Agent
  │                                          CAN LLMs automate causal reasoning?
  │
  └──────────────────────────────────────────────────────────────────────┘
                        (Part 0 motivation referenced throughout)
```

Here is what each part delivers:

| Part | Title | You will learn to... |
|:----:|-------|---------------------|
| 0 | Beyond Correlation (this post) | Distinguish causal from predictive questions |
| 1 | Potential Outcomes | Define $\tau = E[Y(1) - Y(0)]$, state SUTVA, formalize the fundamental problem |
| 2 | Structural Causal Models | Draw DAGs, apply $d$-separation, use $\mathrm{do}$-calculus |
| 3 | Identification | Apply DiD, IV, RDD, synthetic control to derive estimands |
| 4 | Estimation | Implement matching, IPW, doubly robust, and DML estimators |
| 5 | Heterogeneous Effects | Estimate CATE with causal forests and meta-learners |
| 6 | Code Pipeline | Run a full DoWhy + EconML analysis end-to-end in Python |
| 7 | The Causal Agent | Design an LLM-driven agent for automated causal reasoning |

**Reading paths** (choose based on your goal):

- **Full sequence** (0 $\to$ 7): For researchers with no causal background who want complete understanding.
- **Fast track** (0 $\to$ 1 $\to$ 3 $\to$ 4 $\to$ 6): For researchers who need to run a difference-in-differences study now and will backfill theory later.
- **Agent-focused** (0 $\to$ 6 $\to$ 7): For engineers designing causal inference tooling or LLM-powered analysis agents.

### 5.2 The Four-Step Pipeline

Every causal analysis in this series follows a common pipeline, formalized by Microsoft's DoWhy library and adopted as our organizing structure from Part 3 onward:

```
┌─────────┐     ┌────────────┐     ┌────────────┐     ┌──────────┐
│  MODEL   │────→│  IDENTIFY   │────→│  ESTIMATE   │────→│  REFUTE  │
│          │     │             │     │             │     │          │
│ Draw the │     │ Find a valid│     │ Compute the │     │ Stress-  │
│ causal   │     │ estimand    │     │ numerical   │     │ test the │
│ graph    │     │ from the    │     │ estimate    │     │ result   │
│ (DAG)    │     │ graph       │     │ from data   │     │          │
└─────────┘     └────────────┘     └────────────┘     └──────────┘
   Part 2          Part 3             Part 4          Parts 4-6
```

| Step | Core question | Where in series | Key output |
|------|--------------|:---------------:|------------|
| **Model** | What are the causal assumptions? | Part 2 | DAG with nodes and edges |
| **Identify** | Under these assumptions, what formula recovers the causal effect? | Part 3 | Estimand (e.g., backdoor, IV) |
| **Estimate** | Given the estimand and data, what is the numerical effect? | Part 4 | Point estimate + confidence interval |
| **Refute** | How sensitive is the result to violations of assumptions? | Parts 4-6 | Sensitivity analysis, placebo tests |

**The pipeline enforces a discipline: no estimation without identification, no identification without a model, and no conclusion without refutation.** This is the fundamental difference from predictive ML, where evaluation is automated by cross-validation. In causal inference, evaluation requires adversarial reasoning about unobserved confounders.

### 5.3 Notation Convention

All posts in this series use a unified notation. The essentials are:

| Symbol | Meaning |
|--------|---------|
| $i = 1, \dots, N$ | Unit index |
| $W_i \in \{0, 1\}$ | Binary treatment (1 = treated, 0 = control) |
| $Y_i$ | Observed outcome |
| $Y_i(w)$ | Potential outcome under treatment $w$ |
| $\mathbf{X}_i \in \mathbb{R}^p$ | Pre-treatment covariates |
| $\tau = E[Y_i(1) - Y_i(0)]$ | Average Treatment Effect (ATE) |
| $e(\mathbf{x}) = P(W_i = 1 \mid \mathbf{X}_i = \mathbf{x})$ | Propensity score |
| $\mathrm{do}(X = x)$ | Pearl's intervention operator |
| $X \rightarrow Y$ | Direct cause in a DAG |
| $X \perp\!\!\!\perp Y \mid Z$ | Conditional independence |

When notation departs from this convention (e.g., time indices in Part 3's DiD section, or instrument $Z_i$ in Part 3's IV section), the departure is flagged inline.

---

## Summary

This post established three claims:

1. **Causal questions are mathematically distinct from predictive questions.** The target estimand $E[Y(1) - Y(0)]$ differs from the predictive conditional $E[Y \mid X]$ in both definition and identification requirements.

2. **Pearl's Ladder of Causation organizes all statistical queries into a strict hierarchy — association, intervention, counterfactual — where each level requires strictly stronger assumptions than the one below.** No amount of Level 1 data can answer a Level 2 question without assumptions that go beyond the data.

3. **Simpson's Paradox demonstrates that the correct analysis of the same data depends on the causal graph, not the data alone.** Whether to aggregate or disaggregate is a causal decision, and the data cannot make it for you.

The rest of this series provides the mathematical tools and computational pipeline to move from "I have a causal question" to "I have a defensible answer."

### What Comes Next

Part 1 begins with the *Rubin Causal Model* and the *Fundamental Problem of Causal Inference* — the formal statement of why causal effects are hard to estimate: for any unit $i$, we observe either $Y_i(1)$ or $Y_i(0)$, but never both. This missing data problem is the engine that drives the entire field.

With the Fundamental Problem in hand, we will develop two complementary frameworks for overcoming it:

- **The Potential Outcomes framework** (Part 1): defines causal effects as contrasts between potential outcomes and derives identification conditions from assignment mechanisms.
- **The Structural Causal Model framework** (Part 2): defines causal effects through graphical models and derives identification conditions from $d$-separation and the $\mathrm{do}$-calculus.

These two frameworks are not competitors — they are two languages for the same mathematics, and a working causal analyst uses both. The rest of the series (Parts 3-7) builds on both frameworks simultaneously.

---

## References

- Bareinboim, E., & Pearl, J. (2016). Causal inference and the data-fusion problem. *Proceedings of the National Academy of Sciences*, 113(27), 7345-7352.
- Bickel, P. J., Hammel, E. A., & O'Connell, J. W. (1975). Sex bias in graduate admissions: Data from Berkeley. *Science*, 187(4175), 398-404.
- Caruana, R., Lou, Y., Gehrke, J., Koch, P., Sturm, M., & Elhadad, N. (2015). Intelligible models for healthcare: Predicting pneumonia risk and hospital 30-day readmission. *Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 1721-1730.
- Hernán, M. A., & Robins, J. M. (2020). *Causal Inference: What If*. Chapman & Hall/CRC.
- Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics, Social, and Biomedical Sciences: An Introduction*. Cambridge University Press.
- LaLonde, R. J. (1986). Evaluating the econometric evaluations of training programs with experimental data. *The American Economic Review*, 76(4), 604-620.
- Pearl, J. (1995). Causal diagrams for empirical research. *Biometrika*, 82(4), 669-688.
- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.
- Pearl, J., & Mackenzie, D. (2018). *The Book of Why: The New Science of Cause and Effect*. Basic Books.

---

*Next: [Part 1 — The Language of Causation: Potential Outcomes and the Rubin Causal Model](/posts/causal-inference-part1-potential-outcomes/)*
