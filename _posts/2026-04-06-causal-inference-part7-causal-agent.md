---
title: "Causal Inference Part 7: The Causal Inference Agent — When LLMs Meet Causality"
date: 2026-04-06 14:02:00 +0900
categories: [AI & ML, ML Theory]
tags: [causal-inference, llm, causal-agent, orca, automation]
math: true
---

**Causal Inference — From First Principles to Automated Reasoning**

*This is Part 7 of an 8-part series.*

- **Part 0:** [Beyond Correlation — Why Causal Inference?](/posts/causal-inference-part0-beyond-correlation/)
- **Part 1:** [Potential Outcomes and the Fundamental Problem](/posts/causal-inference-part1-potential-outcomes/)
- **Part 2:** [Structural Causal Models and DAGs](/posts/causal-inference-part2-structural-causal-models/)
- **Part 3:** [Identification — From Assumptions to Estimands](/posts/causal-inference-part3-identification/)
- **Part 4:** [Estimation — From Estimands to Numbers](/posts/causal-inference-part4-estimation/)
- **Part 5:** [Beyond the Average — Heterogeneous Effects and Causal Discovery](/posts/causal-inference-part5-heterogeneous-effects/)
- **Part 6:** [End-to-End Practice — The Causal Pipeline in Code](/posts/causal-inference-part6-causal-pipeline/)
- **Part 7:** The Causal Inference Agent — When LLMs Meet Causality (this post)

---


## Hook: The Automation Question

You have now learned what a causal inference pipeline looks like. Model a DAG. Identify an estimand via backdoor, instrumental variables, or regression discontinuity. Estimate the effect with matching, IPW, or double machine learning. Refute it with placebo tests and sensitivity analysis. Execute the whole thing in DoWhy and EconML. Each step has formal requirements, standard failure modes, and known diagnostic checks.

That description sounds suspiciously like a recipe. And recipes invite automation.

Consider the workflow a careful researcher follows for a difference-in-differences study:

1. Draw a DAG with treatment, outcome, and confounders. (Part 2)
2. Verify that the parallel trends assumption is plausible. (Part 3)
3. Write estimation code with two-way fixed effects and clustered standard errors. (Part 4)
4. Generate event study plots to visualize pre-trends. (Part 4)
5. Run placebo tests with fake treatment dates. (Part 6)
6. Compute E-values for sensitivity to unmeasured confounding. (Part 6)
7. Write up the results with DAG, plots, and interpretation. (Report)

Steps 3-6 are mechanical. Steps 1-2 require judgment. Step 7 requires both. An agent that handles the mechanical steps well and flags the judgment steps explicitly would save hours per analysis.

The question for this final post is direct: **could an LLM learn to execute this pipeline?** Not as a curiosity, but as a practical tool — an agent that takes a research question and a dataset, proposes a causal design, writes and runs estimation code, validates results, and produces a report.

The answer, as of early 2026, is nuanced. LLMs are excellent at some stages of the pipeline and dangerous at others. Several research groups have built prototype causal agents, and their architectures converge on a common insight: **let the LLM orchestrate; let formal algorithms compute.** This post surveys the evidence, examines four existing systems, proposes a harness architecture, and maps the open problems that remain.

---

## 1. Can LLMs Reason Causally?

### 1.1 The Evidence For

Kiciman et al. (2023) evaluated GPT-4 on a battery of causal reasoning tasks and reported striking results. On pairwise causal discovery — given two variables, determine the causal direction — GPT-4 achieved 97% accuracy, matching or exceeding purpose-built algorithms on standard benchmarks. On counterfactual reasoning tasks — "if $X$ had been different, what would $Y$ have been?" — GPT-4 scored 92%, outperforming smaller models by a wide margin.

**On tasks that map to textbook causal relationships with familiar variable names, frontier LLMs perform near-ceiling.** These results suggest that training corpora encode a substantial amount of causal domain knowledge, accessible through standard prompting.

Further evidence comes from LLM-assisted DAG construction. When asked to propose a directed acyclic graph for well-studied domains (e.g., the effect of smoking on lung cancer, the returns to education), GPT-4 produces graphs that align closely with expert-constructed DAGs from the epidemiological and econometric literature. The model retrieves known causal relationships with high fidelity.

The retrieval extends to methodological knowledge. Prompt GPT-4 with "I have panel data with a policy change in 2015 affecting some states but not others" and it correctly identifies difference-in-differences as the natural identification strategy, flags parallel trends as the key assumption, and suggests event study plots as a diagnostic. This is textbook-level reasoning executed reliably because the answer exists, nearly verbatim, in the training data.

To make this concrete, consider the following prompt and response pattern:

```
PROMPT:   "I want to estimate the effect of a minimum wage increase on
           employment. I have county-level data for 2010-2020. Some
           states raised the minimum wage in 2015; others did not."

LLM:      "This is a difference-in-differences setting. Treatment W is
           the minimum wage increase. Outcome Y is employment. The key
           assumption is parallel trends: absent the wage increase,
           treated and control counties would have followed the same
           employment trajectory. You should:
           (1) Plot pre-treatment trends for both groups.
           (2) Estimate a two-way fixed effects model with county
               and year fixed effects.
           (3) Run an event study to check for pre-trends.
           (4) Cluster standard errors at the state level."
```

This response is correct, complete, and operationally useful. The model is retrieving a well-worn template from the labor economics literature. The danger arises when the problem departs from the template — when treatment timing varies, when there are staggered adoptions, when the parallel trends assumption is suspect — and the model continues to apply the same template without adjustment.

### 1.2 The Evidence Against

The picture darkens when you probe deeper. Long, Schuster, and Piech (2023) showed that performance drops sharply when variable names are replaced with abstract labels ($X_1$, $X_2$, $X_3$ instead of "smoking," "tar deposits," "lung cancer"). The same structural reasoning problem, stripped of semantic cues, exposes the mechanism behind the high scores: **LLMs perform semantic pattern matching, not structural reasoning.** They recognize that "smoking causes cancer" because the training data says so, not because they derive it from the conditional independence structure of the data.

Additional failure modes compound this concern:

- **Bias inheritance.** LLMs reproduce the causal beliefs embedded in their training data, including incorrect or contested ones. If the literature disproportionately discusses a particular causal pathway, the model weights that pathway regardless of its structural validity.
- **Overconfidence on novel domains.** When the causal structure is genuinely unknown — the case that matters most for research — LLMs generate plausible-sounding but unverifiable DAGs with no uncertainty quantification.
- **do-calculus failures.** Ask an LLM to apply the three rules of do-calculus to derive an interventional distribution from a novel graph, and error rates climb above 40%. The symbolic manipulation required is brittle under standard autoregressive generation.
- **Collider blindness.** LLMs frequently recommend conditioning on post-treatment variables or colliders, the exact error that Part 2 of this series warned against. The model sees a variable correlated with both treatment and outcome and defaults to "control for it" — the associational instinct that causal inference exists to correct.

### 1.3 The Working Hypothesis

**LLMs are excellent causal *librarians* — retrieving domain knowledge for DAG construction — but unreliable causal *reasoners* — applying do-calculus or verifying identification conditions.** Any agent architecture must leverage the former capability while guarding against the latter.

This asymmetry is not unique to causality. It mirrors the broader pattern in LLM tool use: models excel at understanding *what* computation to invoke and *why*, but should not be trusted to *perform* the computation themselves. Just as we do not ask an LLM to multiply large matrices, we should not ask it to verify backdoor criterion satisfaction on a complex graph.

```
LLM CAUSAL CAPABILITY MAP

                    High ┌──────────────────────────────────────┐
                         │                                      │
    LLM Reliability      │   DAG proposal    Research question  │
                         │   (known domains)  formulation       │
                         │                                      │
                         │   Variable         Literature        │
                         │   identification   retrieval         │
                         │                                      │
                    Med  ├──────────────────────────────────────┤
                         │                                      │
                         │   Method           Assumption        │
                         │   selection        articulation      │
                         │                                      │
                         │   Code             Report            │
                         │   generation       writing           │
                         │                                      │
                    Low  ├──────────────────────────────────────┤
                         │                                      │
                         │   do-calculus      Novel graph       │
                         │   derivation       reasoning         │
                         │                                      │
                         │   Identification   Statistical       │
                         │   verification     computation       │
                         │                                      │
                         └──────────────────────────────────────┘
                           Retrieval-heavy ──→ Reasoning-heavy
```

| Task | LLM Performance | Mechanism | Limitation |
|------|:--------------:|-----------|------------|
| Pairwise causal direction (known domains) | ~97% | Training data retrieval | Drops to ~60% with abstract variable names |
| Counterfactual reasoning (textbook) | ~92% | Semantic pattern matching | Fails on novel structural configurations |
| DAG construction (well-studied domains) | High | Domain knowledge recall | No uncertainty quantification; "plausible" $\neq$ "correct" |
| do-calculus application | ~55-60% | Symbolic manipulation attempt | Brittle; no formal verification of derivation steps |
| Identification strategy selection | Moderate | Heuristic matching | Defaults to familiar designs; misses edge cases |
| Code generation for estimation | High | Code pattern retrieval | Must be executed and validated externally |

---

## 2. Existing Agent Systems — A Survey

Four research groups have published prototype causal inference agents. Their architectures differ in detail but converge on a shared principle: **LLM for orchestration, formal algorithms for computation.** No system trusts the LLM to do math.

### 2.1 ORCA (CHI 2026)

ORCA implements a three-agent architecture: Explore, Discover, and Infer.

- The **Explore agent** profiles the dataset — distributions, missingness patterns, variable types, and overlap diagnostics.
- The **Discover agent** proposes a causal graph, combining LLM domain knowledge with constraint-based discovery algorithms (PC, FCI).
- The **Infer agent** selects an identification strategy and runs estimation using DoWhy and EconML.

The key innovation is *shared state with human checkpoints*. All three agents read from and write to a common state object that includes the current DAG, identified estimand, and analysis log. At two critical junctures — after DAG proposal and after estimation — the system pauses for human validation. The domain expert can modify the DAG, reject the estimand, or request alternative specifications.

**What works.** The shared state prevents the agents from contradicting each other. The Explore agent flags data quality issues (e.g., extreme propensity scores indicating overlap violations) before the pipeline reaches estimation, preventing wasted computation on doomed analyses. The human checkpoints catch the most dangerous failure mode — an incorrect DAG propagating through the pipeline. The Explore agent's data profiling is reliable because it delegates entirely to pandas-profiling and scipy.stats.

**What does not work.** The system requires a domain expert who understands causal graphs well enough to validate the Discover agent's proposals. This is a significant limitation — the very users who most need an automated agent are those who lack causal inference expertise. The Discover agent's graph proposals are only as good as GPT-4's domain knowledge retrieval, which degrades on specialized or interdisciplinary topics. In evaluation, ORCA's DAG proposals for well-studied epidemiological questions matched expert graphs on 85% of edges, but dropped to 60% on interdisciplinary questions involving novel biological mechanisms.

### 2.2 CausalAgent (IUI 2026)

CausalAgent takes a conversational approach. Built on the Model Context Protocol (MCP) with a retrieval-augmented generation (RAG) backend that indexes causal inference textbooks (Hernan and Robins, Pearl, Imbens and Rubin), the system guides users through the Model-Identify-Estimate-Refute pipeline via dialogue.

The architecture is a single LLM agent augmented with tool calls to DoWhy and EconML, plus a RAG module that retrieves relevant textbook passages when the user asks methodological questions. When the user describes a research question, CausalAgent proposes a DAG and walks through the identification logic step by step, citing textbook references for each assumption.

**What works.** The conversational interface lowers the barrier to entry. Users who cannot specify a DAG from scratch can iteratively refine one through dialogue ("Should I include income as a confounder?" "What if the treatment effect varies by age group?"). The RAG grounding reduces hallucination — when the model retrieves a passage from Hernan and Robins, the citation is verifiable. The textbook-grounded explanations serve a pedagogical function alongside the analytical one.

**What does not work.** The system's backbone is a generalist LLM (GLM-4), and its performance is tightly coupled to that model's capabilities. When GLM-4 misunderstands a causal concept, the conversational interface propagates the error with pedagogical authority — the user reads a textbook citation and trusts an incorrect interpretation.

The RAG module retrieves based on semantic similarity, not logical relevance — it can surface a passage about instrumental variables when the user's problem actually requires difference-in-differences, because both passages contain keywords like "endogeneity" and "unobserved confounding." The fundamental tension: conversational refinement works only if the user can recognize when the agent is wrong, which presupposes the expertise the agent was designed to substitute for.

### 2.3 CAIS (NeurIPS 2025 Workshop)

The Causal Analysis Intelligent System (CAIS) replaces the LLM's free-form reasoning with a structured decision tree for method selection. Given a dataset and a research question, CAIS walks through a series of diagnostic checks: Is treatment binary or continuous? Is there a time dimension? Are there plausible instruments? Based on the answers, it selects one of eight pre-specified estimation strategies.

The key innovation is *automatic design choice*. Rather than asking the LLM to reason about which method to use, CAIS encodes the decision logic in a rule-based system. The LLM's role is reduced to interpreting the user's research question, classifying variable types, and generating estimation code — tasks where LLMs are reliable.

**What works.** The decision tree eliminates the LLM's most dangerous failure mode — choosing an inappropriate method based on superficial pattern matching. The pre-specified rules encode expert knowledge that does not degrade with novel variable names or unfamiliar domains. The system produces consistent method selections across repeated runs.

**What does not work.** The rule-based approach is rigid. Real causal inference problems often require creative combinations of methods or novel identification strategies that no decision tree anticipates. The system cannot handle problems that fall between its pre-specified categories — a study with both panel data and a discontinuity, for instance, could leverage either DiD or RDD, and the optimal choice depends on which assumptions are more credible in the specific domain. CAIS selects methods but does not verify identification — a DiD design may be selected without checking whether parallel trends hold, and an IV design may proceed without testing instrument relevance. The system automates *method selection* but not *assumption validation*, which is the harder and more important task.

### 2.4 Causal Agent (Yang et al.)

Yang et al. build a three-module agent — Tool, Reasoning, and Memory — and evaluate it on CausalTQA, a benchmark of causal text question-answering tasks. The Tool module wraps standard causal inference libraries. The Reasoning module applies chain-of-thought prompting to decompose complex causal questions. The Memory module stores intermediate results and prior analyses for reuse.

The key innovation is the CausalTQA benchmark itself, which provides a standardized evaluation suite for causal agents. The system achieves over 80% accuracy on this benchmark, demonstrating that the tool-augmented architecture substantially outperforms raw LLM reasoning.

**What works.** The modular architecture is clean and extensible. The Memory module enables the system to learn from prior analyses within a session — if it estimated a propensity score model for one outcome, it can reuse that model for a related outcome. The benchmark-driven development imposes discipline on evaluation.

**What does not work.** **Benchmark performance does not transfer to real-world causal analysis.** CausalTQA tasks are self-contained, with well-defined variables and clear causal structures. Real research problems involve ambiguous variable definitions, missing data, domain-specific confounders, and identification challenges that no benchmark captures. The 80% accuracy on CausalTQA likely overstates real-world performance by a wide margin. Moreover, the Memory module creates a risk of path dependency: if an early analysis in the session reaches an incorrect conclusion, the Memory module reinforces that conclusion in subsequent analyses. Memory is valuable only when the stored content is correct — a condition the system cannot verify.

### 2.5 Cross-System Comparison

```
AGENT ARCHITECTURE COMPARISON

ORCA          CausalAgent       CAIS            Causal Agent
(CHI 2026)    (IUI 2026)        (NeurIPS WS)    (Yang et al.)
┌──────────┐  ┌──────────┐     ┌──────────┐    ┌──────────┐
│ 3 Agents │  │ 1 Agent  │     │ Decision │    │ 3 Modules│
│ Explore  │  │ + MCP    │     │ Tree +   │    │ Tool +   │
│ Discover │  │ + RAG    │     │ LLM Code │    │ Reason + │
│ Infer    │  │ + Tools  │     │ Gen      │    │ Memory   │
├──────────┤  ├──────────┤     ├──────────┤    ├──────────┤
│ Shared   │  │ Conver-  │     │ Rule-    │    │ Benchmark│
│ State +  │  │ sational │     │ based    │    │ driven   │
│ Human    │  │ Textbook │     │ Automatic│    │ Tool-    │
│ Check-   │  │ Grounded │     │ Design   │    │ augmented│
│ points   │  │ Dialogue │     │ Choice   │    │ CoT      │
└──────────┘  └──────────┘     └──────────┘    └──────────┘
```

| Axis | ORCA | CausalAgent | CAIS | Causal Agent |
|------|------|-------------|------|-------------|
| **Architecture** | 3-agent pipeline | Single agent + MCP + RAG | Decision tree + LLM | 3-module (Tool/Reason/Memory) |
| **Key innovation** | Shared state, human checkpoints | Conversational refinement, textbook RAG | Automatic method selection | CausalTQA benchmark, memory |
| **LLM role** | Orchestration + DAG proposal | Dialogue + code generation | Variable classification + code gen | Reasoning decomposition |
| **Formal computation** | Delegated to DoWhy/EconML | Delegated to DoWhy/EconML | Delegated to statsmodels | Delegated to tool module |
| **Primary limitation** | Requires domain expert | GLM backbone dependent | Rigid, not adaptive | Benchmark $\neq$ real-world |

**The convergent design principle across all four systems: the LLM proposes, orchestrates, and explains — it never computes.** Every system delegates statistical estimation, graph analysis, and hypothesis testing to purpose-built libraries. This is the correct architecture, and the harness design in Section 3 codifies it.

---

## 3. Harness Architecture for Causal Inference

The four systems reviewed above share common strengths and complementary weaknesses. ORCA's shared state and human checkpoints are essential. CausalAgent's RAG grounding reduces hallucination. CAIS's structured method selection prevents LLM freestyle errors. Yang et al.'s memory module enables cross-stage coherence.

**A production-grade causal inference agent combines all four insights into a six-stage harness that maps directly to the pipeline built in Parts 0-6 of this series.**

### 3.1 The Six-Stage Pipeline

The pipeline below maps each stage to the series part that developed its theoretical foundation. The arrows represent data flow through the shared state object. The two exclamation marks denote mandatory human checkpoints — the pipeline blocks until the human approves.

```
+-----------------------------------------------------------------+
|                    CAUSAL INFERENCE HARNESS                      |
+-----------------------------------------------------------------+
|                                                                  |
|  Stage 1: FORMULATE (LLM + Domain Expert)          [Part 0]     |
|  +-- Input: Research question + dataset description              |
|  +-- LLM proposes: treatment W, outcome Y, covariates X         |
|  +-- LLM drafts initial DAG from domain knowledge               |
|  +-- ! Human checkpoint: validate DAG                            |
|           |                                                      |
|           v                                                      |
|  Stage 2: EXPLORE (Agent + Tools)                   [Part 1-2]   |
|  +-- Profile data: distributions, missingness, types             |
|  +-- Test assumptions: overlap, stationarity                     |
|  +-- Flag violations: positivity, SUTVA, consistency             |
|  +-- Output: analysis-ready dataset + diagnostics                |
|           |                                                      |
|           v                                                      |
|  Stage 3: IDENTIFY (LLM + do-calculus Engine)       [Part 3]     |
|  +-- Read DAG -> determine estimand (backdoor, IV, etc.)         |
|  +-- Formal verification via do-calculus (not LLM intuition)     |
|  +-- Select identification strategy (DiD, IV, RDD, etc.)        |
|  +-- Generate testable implications from DAG                     |
|           |                                                      |
|           v                                                      |
|  Stage 4: ESTIMATE (Code Execution Agent)           [Part 4-5]   |
|  +-- Generate estimation code (DoWhy / EconML / statsmodels)     |
|  +-- Execute with cross-fitting, proper SE clustering            |
|  +-- Run CATE analysis if heterogeneity requested                |
|  +-- Output: point estimate + CI + diagnostics                   |
|           |                                                      |
|           v                                                      |
|  Stage 5: REFUTE (Validation Agent)                 [Part 4-6]   |
|  +-- Run placebo tests, random common cause, data subset         |
|  +-- Sensitivity analysis (E-value, Rosenbaum bounds)            |
|  +-- Test DAG implications against data (d-separation tests)     |
|  +-- ! Human checkpoint: assess robustness                       |
|  +-- If failed: loop back to Stage 3 with diagnostic report      |
|           |                                                      |
|           v                                                      |
|  Stage 6: REPORT (LLM + RAG)                       [Part 7]     |
|  +-- Synthesize findings into structured report                  |
|  +-- Visualize: DAG, event study plot, balance diagnostics       |
|  +-- Ground interpretation in literature (RAG)                   |
|  +-- Include assumption audit trail from shared state             |
|                                                                  |
+-----------------------------------------------------------------+
```

The pipeline is sequential by design. Each stage depends on the output of the previous stage, and the shared state object accumulates the full audit trail. A reader of the final report can trace every result back to the DAG assumption that justifies it.

### 3.2 Design Principles

**Principle 1: LLM for orchestration, algorithms for computation.** The LLM decides *what* to compute. DoWhy, EconML, statsmodels, and causal-learn decide *how*. This separation is non-negotiable. When the system needs to verify that the backdoor criterion holds on a given DAG, it calls `dowhy.CausalModel.identify_effect()`, not `llm.chat("does the backdoor criterion hold?")`. When it needs a point estimate, it calls `econml.dml.DML.fit()`, not `llm.chat("what is the treatment effect?")`. The boundary is crisp: the LLM never touches a number. It generates code, interprets output, and writes prose — three tasks where autoregressive generation excels.

**Principle 2: Human-in-the-loop at DAG validation and robustness assessment.** Two stages carry existential risk. An incorrect DAG at Stage 1 propagates silently through every downstream stage — the identification strategy is valid *only* relative to the assumed graph. An uncritical acceptance of estimates at Stage 5 can produce a "statistically significant" result that collapses under the first sensitivity check a reviewer applies. Both stages require mandatory human review.

**Principle 3: Shared state across stages.** Every stage reads from and writes to a common state object. This object contains:

- The current DAG (as a networkx DiGraph), including edge metadata (source: "LLM domain knowledge" vs. "human expert" vs. "discovery algorithm")
- The identified estimand (as a DoWhy Estimand object), with the identification strategy and required assumptions
- The dataset (as a pandas DataFrame), with profiling metadata from Stage 2
- The estimation results (point estimates, confidence intervals, diagnostic plots)
- The refutation results (p-values from placebo tests, E-values from sensitivity analysis)
- A complete decision log: every choice the agent made, every tool it called, every human override

Shared state prevents the failure mode where Stage 4 estimates an effect using a different adjustment set than Stage 3 identified. It also enables the Stage 5 loop-back: when a refutation test fails, the agent can inspect the shared state to determine whether the failure implicates the DAG (Stage 1), the identification strategy (Stage 3), or the estimation method (Stage 4).

The shared state object can be thought of as a structured log:

```
SHARED STATE OBJECT (simplified)

{
  "dag": {
    "nodes": ["W", "Y", "X1", "X2", "X3"],
    "edges": [("X1","W"), ("X1","Y"), ("X2","W"), ...],
    "edge_sources": {"(X1,W)": "LLM:domain", "(X1,Y)": "human:expert"},
    "validated_by_human": true
  },
  "estimand": {
    "type": "backdoor",
    "adjustment_set": ["X1", "X2"],
    "verified_by": "DoWhy.identify_effect()",
    "assumptions": ["no_unobserved_confounders", "positivity"]
  },
  "estimate": {
    "method": "DML",
    "ate": 0.15,
    "ci_95": [0.08, 0.22],
    "se_type": "clustered",
    "cross_fit_folds": 5
  },
  "refutation": {
    "placebo_treatment": {"ate": 0.01, "p_value": 0.82},
    "random_common_cause": {"ate": 0.14, "p_value": 0.03},
    "data_subset": {"ate": 0.13, "p_value": 0.04},
    "e_value": 2.3,
    "all_passed": true
  },
  "decision_log": ["Stage 1: LLM proposed DAG...", ...]
}
```

Every downstream computation references this object rather than recomputing or reinterpreting earlier results.

**Principle 4: Refutation is mandatory, not optional.** In the pipeline from Part 6, the Refute step is the final gate before reporting. In the harness, Stage 5 is not a suggestion — it is a hard requirement. The system does not proceed to Stage 6 until at least three refutation tests have passed: (1) a placebo treatment test, (2) a random common cause test, and (3) a data subset validation test. If any test fails, the system loops back to Stage 3 to reconsider the identification strategy.

**Principle 5: Every LLM assertion about identification must be formally verified.** When the LLM at Stage 3 claims that a set of variables satisfies the backdoor criterion, the harness calls `dowhy.CausalModel.identify_effect()` to verify algorithmically. When the LLM claims that an instrument is valid, the harness runs a first-stage F-test for relevance (the testable part) and flags the exclusion restriction for human review (the untestable part). The LLM's role at the identification stage is to *propose* — the formal engine's role is to *verify*.

**Principle 6: Fail loudly, not silently.** The most dangerous failure mode is a confident wrong answer. The harness is designed to surface uncertainty at every stage. At Stage 1, the system reports how many edges in the proposed DAG are supported by retrieved literature versus inferred by the LLM. At Stage 3, it reports whether identification succeeded via the primary strategy or required fallback. At Stage 5, it reports exact E-values and Rosenbaum bounds rather than binary pass/fail. The human reviewer receives not just results but a calibrated uncertainty profile.

### 3.3 Stage-to-Tool Mapping

| Stage | Primary Library | Agent Type | LLM Role | Formal Engine |
|-------|----------------|------------|----------|---------------|
| **1. Formulate** | networkx, graphviz | Dialogue agent | Propose DAG, name variables | None (human validates) |
| **2. Explore** | pandas-profiling, scipy | Tool-calling agent | Interpret diagnostics | Statistical tests (KS, chi-squared) |
| **3. Identify** | DoWhy (identify_effect) | Planner agent | Select strategy | do-calculus engine, backdoor/frontdoor algorithms |
| **4. Estimate** | EconML, statsmodels, DoWhy | Code execution agent | Generate estimation code | DML, IPW, matching implementations |
| **5. Refute** | DoWhy (refute_estimate), sensemakr | Validation agent | Interpret refutation results | Placebo tests, E-value computation |
| **6. Report** | matplotlib, Jinja2 | Report generation agent | Write narrative, ground in literature | RAG over causal inference textbooks |

### 3.4 A Concrete Walkthrough

Consider the running example from this series: estimating the causal effect of AlphaFold on structural biology output. Here is how the harness processes it.

**Stage 1 (Formulate).** The researcher inputs: "Did AlphaFold accelerate protein structure determination?" The LLM proposes $W_i$ = AlphaFold availability (binary, pre/post 2021), $Y_i$ = number of structures deposited, $\mathbf{X}_i$ = protein difficulty, organism, experimental method availability. The LLM drafts a DAG with "protein difficulty" as a confounder (it affects both AlphaFold adoption and scientific output). The human reviews and adds "funding trends" as an additional confounder the LLM missed.

**Stage 2 (Explore).** The agent profiles the PDB dataset. It flags that deposition counts are right-skewed (suggesting Poisson or negative binomial models), that missingness in the difficulty variable is 12% (requiring imputation or sensitivity analysis), and that the propensity score distribution shows adequate overlap between pre- and post-AlphaFold proteins.

**Stage 3 (Identify).** Given the panel structure (proteins observed over time, with a treatment date), the agent proposes difference-in-differences. DoWhy's `identify_effect()` confirms that, conditional on the DAG, the DiD estimand is identified via the parallel trends assumption. The agent flags this assumption for human review and generates an event study plot specification.

**Stage 4 (Estimate).** The agent generates EconML code implementing two-way fixed effects DiD with Poisson pseudo-maximum likelihood (PPML) estimation — the specification from Part 4. It runs the estimation with clustered standard errors at the protein-family level.

**Stage 5 (Refute).** The agent runs three refutation tests: (1) placebo treatment date (shifting the treatment to 2019, two years before actual), (2) random common cause (adding a synthetic confounder), (3) data subset validation (dropping the 20% hardest proteins). The placebo test produces a near-zero, statistically insignificant estimate — as expected. The human reviews the E-value and judges that the robustness margin is sufficient given domain knowledge about plausible unmeasured confounders.

**Stage 6 (Report).** The agent drafts a report with DAG visualization, event study plot, balance diagnostics, and a narrative grounded in the structural biology literature via RAG.

Total human involvement: two checkpoints (DAG validation at Stage 1, robustness assessment at Stage 5) plus a 15-minute review of the final report. Total agent execution time: under 10 minutes. The agent handled data profiling, code generation, estimation, and mechanical refutation — the researcher handled assumptions and interpretation.

### 3.5 Human vs. Agent Responsibility Matrix

**The division of labor between human and agent is determined by a single criterion: does the task require untestable assumptions?** If yes, the human decides. If no, the agent executes.

| Decision | Human | Agent | Rationale |
|----------|:-----:|:-----:|-----------|
| Define research question | Primary | Assists | Requires domain knowledge and research goals |
| Propose initial DAG | Reviews | Proposes | LLM retrieves domain knowledge; human validates structure |
| Validate DAG edges | Primary | Suggests | Untestable assumption — no algorithm can verify |
| Select adjustment set | Reviews | Computes | Algorithmic (backdoor criterion), but human checks graph |
| Choose identification strategy | Reviews | Proposes | Agent matches design to data features; human assesses plausibility |
| Write estimation code | Reviews | Generates | Mechanical translation from estimand to code |
| Execute estimation | Monitors | Executes | Computation; human checks convergence and diagnostics |
| Interpret point estimate | Primary | Assists | Requires domain context for practical significance |
| Run refutation tests | Specifies | Executes | Mechanical; human specifies which tests matter |
| Assess robustness | Primary | Reports | Judgment call on whether sensitivity bounds are acceptable |
| Write report | Edits | Drafts | Agent drafts; human ensures claims match evidence |

> **Design heuristic**: If you would not trust a research assistant to make the decision without your review, do not trust the agent. The agent is a very fast, very well-read research assistant — not a principal investigator.

### 3.6 Mapping to the Series

The six-stage harness is not a new invention. It is the pipeline from Parts 0-6, rewritten as an agent specification.

| Harness Stage | Series Part | Core Concept |
|---------------|:-----------:|--------------|
| Stage 1: Formulate | Part 0 (Beyond Correlation) + Part 2 (SCM) | Define the causal question; draw the DAG |
| Stage 2: Explore | Part 1 (Potential Outcomes) | Check overlap, SUTVA, consistency |
| Stage 3: Identify | Part 3 (Identification) | Backdoor, IV, RDD, DiD, Synthetic Control |
| Stage 4: Estimate | Part 4 (Estimation) + Part 5 (HTE) | DML, IPW, matching, CATE, causal forests |
| Stage 5: Refute | Part 4 (Refutation) + Part 6 (Pipeline) | Placebo, sensitivity, subset validation |
| Stage 6: Report | Part 7 (this post) | Synthesis, visualization, interpretation |

**The pipeline we built across seven posts is exactly the specification that a causal inference agent must implement.** Every formal concept — $d$-separation, the backdoor criterion, double machine learning, E-values — has a precise role in the harness. The series was, in retrospect, an agent design document.

---

## 4. Open Problems and Limitations

### 4.1 DAG Elicitation: "Plausible" Is Not "Correct"

The most critical failure point in the harness is Stage 1. **An LLM can propose a plausible DAG, but plausibility is not correctness.** The DAG encodes untestable assumptions — no algorithm, no matter how sophisticated, can determine whether an edge should exist between two variables from observational data alone (except in limited cases under faithfulness). When the LLM proposes that "socioeconomic status $\to$ drug adherence $\to$ health outcome" with no direct edge from SES to outcome, it is making a substantive claim about the absence of a causal pathway. If that claim is wrong, every downstream result is biased, and the system provides no warning.

The testable implications of a DAG (conditional independences implied by $d$-separation) can rule out *some* graphs but cannot confirm the correct one. The space of observationally equivalent DAGs — the Markov equivalence class — is large. For a graph with $p$ nodes, the number of DAGs in the equivalence class can grow exponentially.

Current agents do not quantify this structural uncertainty or present alternative DAGs to the user. A responsible agent would report: "I propose this DAG, but 4 other DAGs in the same equivalence class imply different adjustment sets. Here are the alternative estimands under each." No existing system does this. The gap between what agents report (one DAG, presented as definitive) and what honest uncertainty quantification requires (a set of plausible DAGs with their implications) is the largest open problem in the field.

### 4.2 Assumption Sensitivity: Untestable Remains Untestable

The identification strategies in Part 3 rest on assumptions — parallel trends for DiD, exclusion restriction for IV, continuity of potential outcomes for RDD — that cannot be tested from the data. Sensitivity analysis (E-values, Rosenbaum bounds) quantifies *how much* confounding would be needed to overturn the result, but cannot determine *whether* that confounding exists.

**An agent can automate the computation of sensitivity parameters. It cannot automate the judgment of whether those parameters represent a plausible threat.** That judgment requires domain expertise — knowledge of what unmeasured confounders might exist and how strong they might be. No amount of LLM reasoning substitutes for this.

Consider a concrete example. The agent estimates a treatment effect of $\hat{\tau} = 0.15$ with an E-value of 2.3 — meaning an unmeasured confounder would need to be associated with both treatment and outcome by a risk ratio of at least 2.3 to explain away the effect. Is 2.3 large enough? In pharmacoepidemiology, where confounding by indication routinely produces risk ratios above 3, the answer is no — the result is fragile. In a randomized trial with minor noncompliance, the answer is yes — a risk ratio of 2.3 from an unmeasured confounder is implausible. The number is the same; the interpretation is entirely domain-dependent.

### 4.3 Novel Research Designs

> **Pitfall**: The agent's method menu is a ceiling, not a floor. If the best identification strategy for your problem is not in the agent's repertoire, the agent will select the closest available alternative — which may be meaningfully inferior. The agent cannot tell you that a better option exists outside its menu.

Agents select from a menu of known identification strategies: DiD, IV, RDD, synthetic control, matching, IPW, DML. Real research sometimes requires novel designs — combining strategies in non-standard ways, exploiting domain-specific natural experiments, or inventing new identification arguments.

**Creativity in research design remains a human capability.** The most impactful causal studies in economics and epidemiology succeeded because researchers identified a clever source of quasi-random variation that no algorithm would have proposed:

- Angrist (1990) used the Vietnam draft lottery as an instrument for military service — the lottery number is random, satisfying the exclusion restriction by construction.
- Card (1990) used the Mariel boatlift as a natural experiment for the effect of immigration on wages — the sudden, politically-driven influx was orthogonal to local labor market conditions.
- Snow (1855) used the quasi-random assignment of London households to different water companies as a natural experiment for the waterborne transmission of cholera.

Each of these designs required recognizing that a specific historical event created the conditions for causal identification. This is abductive reasoning of a kind that no current agent architecture can replicate — the agent would need to search the space of all possible natural experiments and evaluate each for identification validity, a task that requires both creativity and deep domain knowledge.

### 4.4 Hallucination Risk at the Identification Stage

The most dangerous hallucination in a causal agent is not a factual error in the report — it is a false claim that an identification condition is satisfied. If the LLM asserts "the backdoor criterion is satisfied by conditioning on $\{X_1, X_3\}$" when an unblocked path exists through $X_2$, the downstream estimate is biased and the system reports it as valid.

**Formal verification at every stage is the only defense.** The harness architecture addresses this by routing all identification checks through DoWhy's algorithmic verification, never through LLM assertion. But this defense holds only if the DAG is correct — which brings us back to Problem 4.1. The chain of trust is: human validates DAG $\to$ algorithm verifies identification $\to$ code executes estimation $\to$ tests validate robustness. Remove the human from the first link and the entire chain collapses.

### 4.5 Summary of Open Problems

| Problem | Stage Affected | Can Agent Address? | Mitigation |
|---------|:--------------:|:------------------:|------------|
| DAG correctness | Stage 1 | No — untestable | Human validation + present alternatives |
| Assumption sensitivity | Stage 3, 5 | Partially — computes but cannot judge | Report sensitivity parameters; human interprets |
| Novel designs | Stage 3 | No — requires creativity | Expand method menu over time |
| Hallucination at identification | Stage 3 | Yes — with formal verification | Route all checks through algorithms, not LLM |
| Equivalence class ambiguity | Stage 1, 3 | Partially — can enumerate | Present multiple DAGs with implications |

### 4.6 The Verification Dependency Chain

The open problems form a dependency chain that no current system resolves:

```
Correct DAG (untestable)
    |
    v
Valid identification (algorithmically verifiable, given correct DAG)
    |
    v
Consistent estimation (statistically verifiable, given valid identification)
    |
    v
Robust conclusion (sensitivity analysis, given consistent estimation)
```

**Every link in the chain depends on the one above. The first link is the weakest, and it is the one the agent is least equipped to verify.** This is the fundamental limitation of automated causal inference: the hardest problem (structural assumptions) is the first one encountered, and it is the one that requires human judgment most.

---

## 5. Where This Is Going

```
CAUSAL AGENT CAPABILITY TIMELINE

2026        2027        2028        2029        2030+
  |           |           |           |           |
  |-- Near ---|--- Medium-Term ------|--- Long --|
  |           |           |           |           |
  | Data      | Design    | Structural| Formal    |
  | profiling | proposal  | uncertainty| do-calculus|
  |           |           | quantif.  | reasoning |
  | Code gen  | Multi-DAG |           |           |
  |           | comparison| Discovery | Novel     |
  | Robustness|           | + LLM     | design    |
  | checks    | Expert    | integration| proposal |
  |           | validation|           |           |
  | Report    | at design |           | Reduced   |
  | drafting  | level     |           | human     |
  |           |           |           | oversight |
  |           |           |           |           |
  [  Augmented Analyst  ] [ Research Collaborator ] [ ??? ]
```

### 5.1 Near-Term (2026-2027)

Agents automate the mechanical parts of the pipeline with high reliability. Data profiling, code generation, boilerplate robustness checks, report drafting — these are tasks where LLMs add value without introducing risk. The human analyst spends less time writing pandas code and more time thinking about identification.

Concrete capabilities that are deployable now:

- **Data profiling agents** that read a dataset description and produce a complete diagnostics report: variable distributions, missingness patterns, overlap checks, stationarity tests.
- **Code generation agents** that translate an estimand specification (e.g., "estimate ATT via DML with random forest nuisance models") into executable DoWhy/EconML code with proper cross-fitting and standard error clustering.
- **Refutation agents** that run a standard battery of placebo tests and sensitivity analyses given an estimated causal model, producing formatted diagnostic output.
- **Literature retrieval agents** that, given a proposed DAG, search for published studies that support or contradict each edge.

**The near-term value proposition is not "automated causal inference" but "augmented causal analyst."** The agent handles the 60% of the workflow that is mechanical, freeing the researcher for the 40% that is judgment.

### 5.2 Medium-Term (2027-2029)

Agents propose complete research designs — DAG, identification strategy, estimation plan — subject to expert validation. The key advance is structural uncertainty quantification: the agent presents not one DAG but a set of plausible DAGs with their implied identification strategies, and the expert selects among them.

This requires two technical advances. First, integrating causal discovery algorithms (PC, FCI, GES) with LLM domain knowledge in a principled way — using the LLM's knowledge to set priors on edge existence and the discovery algorithm to update those priors with data. Second, developing methods to propagate DAG uncertainty through the identification and estimation stages — if three plausible DAGs yield three different estimands, the agent should report all three estimates with their respective assumptions, not silently choose one.

The medium-term agent looks less like a pipeline executor and more like a research collaborator: "Given your data and the literature, here are three defensible causal designs ranked by assumption strength. Design A requires parallel trends, which the event study supports. Design B requires an exclusion restriction, which I cannot verify. Design C requires conditional ignorability with 14 covariates, but overlap is thin in the tails. Which assumptions do you find most credible?"

### 5.3 Long-Term (2029+)

Causal reasoning becomes a core LLM capability — models that can apply do-calculus rules reliably, reason about conditional independences in novel graphs, and detect identification failures without delegating to external engines. This is not a prediction but a research direction. Current architectures (autoregressive next-token prediction) are not well-suited to the kind of symbolic, structural reasoning that do-calculus requires. Whether scaling, fine-tuning on formal proofs, or architectural innovations (neurosymbolic hybrids) will close the gap is an open question.

The long-term benchmark is not "can an agent run a causal analysis?" but "can an agent design a *novel* causal analysis?" — one that exploits a natural experiment the researcher had not considered, or combines identification strategies in a way the textbooks do not cover. That capability requires genuine causal reasoning, not retrieval. It is the frontier.

### 5.4 The Series as Agent Specification

The pipeline we built across Parts 0-6 is exactly what the agent must learn:

| Series Part | Agent Capability Required |
|:-----------:|--------------------------|
| Part 0 | Distinguish causal from predictive questions |
| Part 1 | Formalize treatment, outcome, potential outcomes |
| Part 2 | Construct and reason about DAGs |
| Part 3 | Select and justify identification strategies |
| Part 4 | Implement estimation with proper inference |
| Part 5 | Estimate heterogeneous effects; discover structure |
| Part 6 | Execute the full pipeline in code |
| Part 7 | Know what it can and cannot do |

**The last row is the hardest.** An agent that knows the limits of its own causal reasoning — that flags uncertainty in its DAG proposals, that defers to formal verification for identification, that insists on human review for untestable assumptions — is more valuable than an agent that confidently produces wrong answers.

---

## Series Conclusion

This series began with a hospital model that would kill asthmatic pneumonia patients by mistaking association for causation (Part 0). It ends with the question of whether an LLM can learn to avoid that mistake.

Between those endpoints, we built a complete toolkit:

- **Part 0** established that causal questions are mathematically distinct from predictive questions — Pearl's Ladder of Causation separates association, intervention, and counterfactual reasoning into a strict hierarchy.
- **Part 1** formalized causal effects as contrasts between potential outcomes — $\tau = E[Y_i(1) - Y_i(0)]$ — and identified the fundamental problem: we never observe both $Y_i(1)$ and $Y_i(0)$ for the same unit.
- **Part 2** introduced directed acyclic graphs as a language for encoding causal assumptions, with $d$-separation as the bridge between structure and statistical testability, and the $\mathrm{do}(\cdot)$ operator as the formal tool for reasoning about interventions.
- **Part 3** showed how identification strategies — difference-in-differences, instrumental variables, regression discontinuity, synthetic control — recover causal effects from observational data under stated assumptions, each with its own untestable requirement.
- **Part 4** translated estimands into estimators — matching, inverse probability weighting, double machine learning — with proper inference, and introduced refutation as the mandatory final step.
- **Part 5** moved beyond the average to ask *who benefits most*, using CATE estimation $\tau(\mathbf{x})$, causal forests, and meta-learners, and introduced causal discovery as a complement to causal estimation.
- **Part 6** wired the entire pipeline into Python code with DoWhy and EconML, demonstrating the Model-Identify-Estimate-Refute workflow end to end.

And Part 7 asked the automation question. The answer: **LLM-driven agents can accelerate every stage of the causal inference pipeline, but they cannot replace the human judgment that the hardest stage — structural assumptions — demands.** The correct architecture separates orchestration (LLM) from computation (formal algorithms) from judgment (human expert). The harness we proposed implements this separation across six stages with two mandatory human checkpoints.

The causal inference pipeline is not a black box. It is a structured argument: here is my graph, here is my estimand, here is my estimate, here is why you should believe it. Every assumption is stated, every identification condition is verified, every result is stress-tested. That transparency is what makes causal inference trustworthy — and what makes it automatable, in part, by agents that can execute the mechanical steps while flagging the judgment calls.

**The goal is not to remove the human from the loop. The goal is to make the loop faster, more rigorous, and more reproducible — so that the human can focus on the questions that only humans can answer.**

---

## References

- Angrist, J. D., & Pischke, J.-S. (2009). *Mostly Harmless Econometrics: An Empiricist's Companion*. Princeton University Press.
- Chen, Z., et al. (2026). CausalAgent: Conversational causal inference with retrieval-augmented LLMs. *Proceedings of the ACM Conference on Intelligent User Interfaces (IUI)*.
- Hernan, M. A., & Robins, J. M. (2020). *Causal Inference: What If*. Chapman & Hall/CRC.
- Kiciman, E., Ness, R., Sharma, A., & Tan, C. (2023). Causal reasoning and large language models: Opening a new frontier for causality. *arXiv preprint arXiv:2305.00050*.
- Long, K., Schuster, T., & Piech, C. (2023). Can large language models build causal graphs? *NeurIPS 2023 Workshop on Causal Representation Learning*.
- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.
- Sharma, A., & Kiciman, E. (2020). DoWhy: An end-to-end library for causal inference. *arXiv preprint arXiv:2011.04216*.
- Wang, Y., et al. (2025). CAIS: Causal analysis intelligent system with decision-tree method selection. *NeurIPS 2025 Workshop on Causal Inference and Machine Learning*.
- Yang, L., et al. (2025). Causal Agent: Tool-augmented LLM reasoning for causal question answering. *arXiv preprint*.
- Zhang, R., et al. (2026). ORCA: Orchestrating causal analysis with multi-agent LLM systems. *Proceedings of the ACM CHI Conference on Human Factors in Computing Systems*.

---

## Further Reading

For readers who want to go deeper on specific topics covered in this post:

- **LLM causal reasoning benchmarks**: Kiciman et al. (2023) provide the most comprehensive evaluation. For ongoing updates, see the CausalBench leaderboard.
- **Agent architectures for science**: The ORCA and CausalAgent papers provide detailed system descriptions. For the broader context of LLM agents in scientific workflows, see the survey by Wang et al. (2024) on LLM-powered scientific discovery agents.
- **Causal discovery + LLM integration**: Ban et al. (2023) explore using LLMs as priors for causal discovery algorithms — the most principled approach to combining domain knowledge with data-driven structure learning.
- **Harness engineering**: The [Harness Engineering series](/posts/harness-engineering-part1-why-harness/) on this blog develops the general agent architecture pattern (Plan-Generate-Evaluate) that the causal inference harness instantiates for a specific domain.
- **Formal verification of causal reasoning**: For the long-term vision of LLMs with reliable do-calculus, see the literature on neurosymbolic AI and formal methods integration with language models.

---

