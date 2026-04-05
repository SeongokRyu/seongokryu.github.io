---
title: "Harness Engineering Part 2: Self-Improving Harnesses — Lessons from Meta-Harness Research"
date: 2026-04-06 00:40:00 +0900
categories: [AI & ML, AI Agent]
tags: [harness-engineering, meta-harness, claude-code, self-improvement, ai-agent, research]
math: true
---

**Harness Engineering: From Concept to Practice**

*This is Part 2 of a 3-part series on Harness Engineering.*

- **Part 1:** [Why Harness Engineering Matters — The Shift Beyond Prompts and Context](/posts/harness-engineering-part1-why-harness/)
- **Part 2** (this post): Self-Improving Harnesses — Lessons from Meta-Harness Research
- **Part 3:** [Harness Engineering in Practice — Tools, Patterns, and Starting Points](/posts/harness-engineering-part3-practical-toolkit/)

---

## What If AI Designed Its Own Harness?

In [Part 1](/posts/harness-engineering-part1-why-harness/), we established that the bottleneck for long-running agents isn't the model — it's the scaffolding around the model. We name this scaffolding **the harness**, and designing it is a discipline.

But here's a problem: **harness engineering is mostly trial and error**. You try a planner/generator split, measure how often your agent gets stuck, tweak the evaluator prompt, add a retry loop, add a budget guard, re-measure, rewrite. Weeks of hand-tuning.

What if you automated that loop?

That's the bet of a recent paper — **Meta-Harness** (arxiv 2603.28052) — which uses **Claude Code itself as the harness designer**. The system gives the model **full filesystem access to every previous candidate**: its code, its execution traces, its failure logs, its score. Then it asks: *propose a better harness*.

The result is both **a research contribution** and a **practical playbook** for anyone hand-tuning harnesses today. This post walks through what Meta-Harness found, why it works, and the five architectural lessons you can apply starting tomorrow.

---

## Prior Text Optimization and Its Compression Problem

Before Meta-Harness, the state of the art in "LLM-as-optimizer" research looked like this:

| Method | Feedback to optimizer | Scale |
|---|---|---|
| **OPRO** (Google DeepMind) | Scalar scores | 0.002M tokens / iteration |
| **TextGrad** (Stanford) | Short text summaries | 0.015M tokens / iteration |
| **ACE** (various) | Structured manual pipelines | Hand-designed |
| **Meta-Harness** | Full execution traces + code + scores | **10.0M tokens / iteration** |

**The central observation**: every prior method **compresses feedback** before passing it to the optimizer. OPRO hands the LLM a number. TextGrad hands it a sentence. ACE bakes the feedback shape into hand-designed scaffolding.

Compression throws away information. **Meta-Harness refuses to compress.**

Instead, the proposer LLM gets the entire history — code, traces, errors — as files it can selectively read. Access is **adaptive**: the proposer chooses which files to open.

```
Prior text optimizers (OPRO, TextGrad, ACE):

   [Candidate] → [Evaluation]
                      │
                      ▼
              [Compress to scalar or summary]
                      │
                      ▼
              [Propose next candidate]


Meta-Harness:

   [Candidate] → [Evaluation] → [Filesystem: code + trace + score]
                                          │
                                          ▼
                          [Proposer selectively inspects files]
                                          │
                                          ▼
                                [Propose next candidate]
```

The difference sounds small. The results suggest it isn't.

---

## The Meta-Harness Breakthrough: Filesystem as Feedback

The architectural insight of Meta-Harness:

> **Don't summarize the history. Store it as files. Let the proposer decide what to read.**

This is **non-Markovian feedback** — the optimizer isn't restricted to a single recent state. It can reach back to a failing candidate from 10 iterations ago, open its trace, and say "ah, it failed because of X".

### Why This Matters Architecturally

The key properties Meta-Harness gets from filesystem-based feedback:

- **Adaptive inspection**: Median 82 files read per iteration across ~20 prior candidates. The proposer chooses scope.
- **Full execution traces**: Up to 10M tokens of diagnostic data per evaluation (5000x more than OPRO).
- **Persistent state**: History doesn't decay with window size.
- **Human-readable artifacts**: You can inspect what the proposer inspected.

### The Search Loop

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  1. Initialize baseline harness population                  │
│                      │                                      │
│                      ▼                                      │
│  2. Proposer (Claude Code) inspects filesystem              │
│       • Reads selected prior candidates                     │
│       • Reads scores, traces, failure info                  │
│                      │                                      │
│                      ▼                                      │
│  3. Proposer generates new harness candidate                │
│                      │                                      │
│                      ▼                                      │
│  4. Evaluate candidate on search-set tasks                  │
│                      │                                      │
│                      ▼                                      │
│  5. Store results (code, scores, traces) in filesystem      │
│                      │                                      │
│                      └────── loop ──────────┐               │
│                                             ▼               │
│                                     Back to step 2          │
└─────────────────────────────────────────────────────────────┘
```

The proposer gets **minimal domain-specific scaffolding** — mostly, it decides what to do.

---

## Experiment 1: Online Text Classification

### Setup

- **Datasets**: LawBench, Symptom2Disease, USPTO-50k
- **Base model**: GPT-OSS-120B
- **Proposer**: Claude Code Opus 4.6

### Results

| System | Accuracy | Context tokens used |
|---|---|---|
| **ACE** (hand-designed SOTA baseline) | 40.9 | 50.8K |
| **Meta-Harness (discovered)** | **48.6** | **11.4K** |

Meta-Harness beats ACE by **+7.7 points** while using **4× fewer context tokens**.

It also matches competing text optimizers (OpenEvolve, TTT-Discover) with **10× fewer evaluations**.

### Generalization Test

The discovered harness was evaluated on 9 out-of-distribution datasets it had never seen during search:

| System | OOD average |
|---|---|
| ACE | 70.2 |
| **Meta-Harness** | **73.1** (+2.9) |

**The harness generalized.** This wasn't overfitting to the search set.

---

## Experiment 2: Math Reasoning Retrieval

### Setup

- **Search set**: 250 Olympiad-difficulty problems
- **Test set**: 200 IMO-level problems, evaluated on **5 held-out models**

### Discovered Architecture

The proposer discovered a **four-route lexical router** with BM25-based retrieval:

```
[Problem] → [Router: classify domain]
                │
                ├── combinatorics → [Fetch 20 candidates → dedupe to 8 → rerank → keep top 3]
                ├── geometry      → [1 fixed reference + 2 BM25 neighbors]
                ├── number theory → [12 candidates + technique-aware reranking]
                └── default       → [10 candidates + adaptive selection]
```

Each route has **different retrieval hyperparameters** — different fetch counts, different rerankers, different strategies. The proposer chose all of this through search.

### Results

Across 5 held-out models, the discovered harness delivered a **+4.7 point improvement** over no-retrieval baselines. The strategy transferred across models it wasn't designed for.

---

## Experiment 3: TerminalBench-2 (Agentic Coding)

The most practical benchmark: **89 autonomous long-horizon coding tasks** requiring the agent to explore, modify, and verify code across a real codebase.

### Results

| Model | Pass rate | Leaderboard position |
|---|---|---|
| **Opus 4.6** + discovered harness | **76.4%** | **#2** (above Terminus-KIRA at 74.7%) |
| **Haiku 4.5** + discovered harness | **37.6%** | **#1 among all Haiku agents** |

### The Discovered Modification: Environment Bootstrapping

The winning harness modification was small and elegant. A **pre-execution shell command** that runs *before the agent loop starts*:

```bash
# Injected at the very beginning
uname -a
which python node go rustc
ls -la /app
cat /app/README* /app/*.md 2>/dev/null | head -50
# etc.
```

This single addition **eliminated 3–5 wasted exploratory steps** on dependency-heavy tasks. The agent didn't need to "look around" anymore — it knew the environment upfront.

> **Engineering lesson**: Harness improvements aren't always clever. Sometimes they're "gather all the obvious information at the start."

---

## The Ablation That Should Rewrite How You Build Harnesses

Buried in the paper is the single most important finding. The authors asked: **how much does the richness of feedback actually matter?**

They ran the same search loop with different levels of feedback richness:

| Feedback interface | Median accuracy |
|---|---|
| Scores only (scalar) | 34.6 |
| Scores + summaries | 34.9 |
| **Full interface (execution traces)** | **50.0** |

```
Scores only        ████████████████████████████████████░░░  34.6
Scores + summaries █████████████████████████████████████░░  34.9
Full traces        ████████████████████████████████████████████████████  50.0
```

Compressed feedback gets you nowhere. Raw access almost doubles your score.

**This is the most important architectural insight in the paper**:

> **Richer feedback beats more structure.** Pouring effort into cleverer proposer prompts, cleverer search strategies, or cleverer reward shaping buys less than simply **giving the proposer more information to read**.

> **Apply this to your own harness**: If you're designing a feedback loop today, log *everything*. Keep full traces. Resist the urge to pre-summarize.

---

## Proposer as Scientist: Causal Reasoning in the Wild

The TerminalBench-2 search log contains a remarkable qualitative moment.

After **two consecutive failures** from candidate modifications, the proposer didn't blindly try another variant. Instead, it:

```
Step 1: Noticed regression.
Step 2: Hypothesized: "Maybe the prompt template changes,
        not the structural bugfix, caused the regression."
Step 3: Designed a controlled test to isolate the variable.
Step 4: Confirmed the prompt changes were the issue.
Step 5: Observed: "Control-flow modifications remain fragile."
Step 6: Pivoted strategy: "Try purely additive modifications."
Step 7: Additive modification became the winning candidate.
```

This is **variable isolation, hypothesis testing, and strategic pivot** — the behavior of a scientist, not a dumb search process.

The paper's authors didn't program this in. It emerged because the proposer had the information and the freedom to reason.

---

## Implications for Your Harness — 5 Concrete Recommendations

Even if you never build a Meta-Harness-style self-improvement loop, the paper's findings change how you should hand-design harnesses today.

### 1. Log Everything. Summarize Nothing (Until You Have To).

**Instead of**: compressing traces into "summary: agent failed at step 4".

**Do**: store the full trace on disk. If something later needs a summary, summarize then — on demand.

### 2. Make State Live on Disk, Not in Context

**Instead of**: passing 50 turns of conversation history into each agent call.

**Do**: checkpoint state to files. Agents read only what they need.

### 3. Design for Adaptive Inspection

**Instead of**: deciding upfront what each agent sees.

**Do**: let agents choose what to read. Give them a file tree and tools to navigate it.

### 4. Prefer Additive Modifications

**Instead of**: rewriting control flow to fix a bug.

**Do**: add a new pre/post-processing step. The Meta-Harness proposer independently learned that additive changes are safer.

### 5. Bootstrap Your Environment

**Instead of**: letting the agent discover its environment turn-by-turn.

**Do**: run a "dump everything" command at the start. OS, versions, file structure, recent logs. Inject it all upfront.

---

## Comparison: Meta-Harness vs. Competing Optimizers

| Method | Feedback bandwidth | Search space | Scalability | Interpretability |
|---|---|---|---|---|
| **OPRO** | Scalar score | Prompt text | Cheap | Low (black-box score chase) |
| **TextGrad** | NL summary | Prompt text | Moderate | Medium |
| **ACE** | Manual shaping | Pipeline templates | High effort upfront | High but rigid |
| **Meta-Harness** | Full traces | Arbitrary code | Expensive per iter, few iters needed | **High** (readable code) |

The key trade-off: Meta-Harness iterations are **expensive** (10M tokens each), but it needs **far fewer iterations** to reach or beat competing methods. And the output is **inspectable human-readable code** — not weight-space deltas, not prompt templates, but actual harness source that you can read, edit, and reuse.

---

## The Optimization Objective

For the research-minded, here's the core formulation:

$$H^* = \arg\max_H \mathbb{E}[r(\tau, x)] \quad \text{where } \tau \sim p_M(H, x)$$

- $H$: the harness (code surrounding the model)
- $M$: the base model
- $x$: task input
- $\tau$: trajectory (the agent's full execution)
- $r$: reward function

Translation: **find the harness code that maximizes expected task reward, given the underlying model and task distribution**.

In practice, this is an **agentic code search problem**, solved by Claude Code reading and rewriting harness source.

---

## Limitations & What's Next

The Meta-Harness paper is candid about limitations:

- **Single proposer**: all experiments use Claude Code Opus 4.6. Generalization to other proposer models is unverified.
- **Search-eval leakage**: TerminalBench-2 results use the same benchmark for both search and evaluation (mitigated by held-out model generalization tests).
- **Cost**: 10M tokens per iteration is expensive. Hobby projects won't run this.

The authors point to **harness + model weight co-evolution** as the next frontier. If harnesses discovered today can be folded back into fine-tuning objectives tomorrow, the distinction between "what the model knows" and "what the harness adds" starts to blur.

---

## Key Takeaways

- **Meta-Harness automates harness design** by using Claude Code as an agentic proposer with full filesystem access to prior attempts.
- **Filesystem-as-feedback** is the core architectural innovation — 10M tokens of raw information per iteration, adaptively inspected.
- **Beats hand-designed SOTA** by +7.7 points on text classification, ranks #1 on TerminalBench-2 for Haiku, matches best text optimizers in 10× fewer iterations.
- **The ablation is the headline**: full execution traces (50.0) nearly double the accuracy of scalar scores (34.6). **Richer feedback beats cleverer structure.**
- **Discovered harnesses generalize** across OOD datasets and unseen models.
- **The proposer exhibits scientific reasoning** — variable isolation, hypothesis testing, strategy pivots.
- **Five practical lessons** for hand-designed harnesses: log everything, state on disk, adaptive inspection, additive modifications, environment bootstrapping.

---

## What's Next

You now understand *why* harnesses matter (Part 1) and *where the research frontier is* (Part 2).

[**Part 3**](/posts/harness-engineering-part3-practical-toolkit/) is the practical toolkit: the OSS landscape, framework selection matrix, deep dives into LangGraph and `revfactory/harness`, five real-world scenarios from small to large, evaluation protocols, and a 5-week roadmap to go from reading this series to shipping a harness.

If you've ever opened your editor and thought "okay, but what do I actually *do*?" — Part 3 answers that.
