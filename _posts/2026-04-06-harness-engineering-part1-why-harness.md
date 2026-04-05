---
title: "Harness Engineering Part 1: Why It Matters — The Shift Beyond Prompts and Context"
date: 2026-04-06 00:30:00 +0900
categories: [AI & ML, AI Agent]
tags: [harness-engineering, ai-agent, claude-code, workflow, multi-agent, anthropic]
---

**Harness Engineering: From Concept to Practice**

*This is Part 1 of a 3-part series on Harness Engineering.*

- **Part 1** (this post): Why Harness Engineering Matters — The Shift Beyond Prompts and Context
- **Part 2:** [Self-Improving Harnesses — Lessons from Meta-Harness Research](/posts/harness-engineering-part2-meta-harness-research/)
- **Part 3:** [Harness Engineering in Practice — Tools, Patterns, and Starting Points](/posts/harness-engineering-part3-practical-toolkit/)

---

## The Moment You Need a Harness

Your agent worked great for the first 30 minutes.

Then it shipped broken code, **confidently praised its own work**, and quietly stopped halfway through a task you thought was 80% done. You scroll through the logs and find the agent rewriting the same file three times, each pass slightly worse than the last.

If you've built anything with Claude Code, Cursor, aider, Cline, Devin, or your own agent loop, you've seen this movie. **The problem isn't the model. The model is fine.** The problem is the scaffolding around the model — the loop, the state, the tools, the feedback.

That scaffolding has a name now: **the harness**. And designing it is a distinct engineering discipline.

By the end of this post, you'll know what a harness is, why it's a separate discipline from prompt or context engineering, and when you actually need one (and when you don't).

---

## Three Failure Modes of Long-Running Agents

Before defining a harness, let's name the diseases it treats. Every long-running agent eventually hits at least one of these three failure modes.

### 1. Context Anxiety

**Models rush to finish as they approach their token limit.**

As the conversation window fills up, the model starts exhibiting what Anthropic calls "context anxiety" — it wraps things up prematurely, skips verification steps, and declares the task done before it is. You can literally see quality degrade as tokens pile up.

```
Token usage    │ Quality
─────────────  │ ───────
  20% used     │ ★★★★★
  40% used     │ ★★★★☆
  60% used     │ ★★★☆☆
  80% used     │ ★★☆☆☆
  95% used     │ "Task complete!" ✗
```

### 2. Self-Evaluation Bias

**Agents grade their own homework — and always give themselves an A.**

Ask a model "did you complete the task?" and 9 times out of 10 it says yes, even when the output is obviously broken. Anthropic observed that agents "confidently praise the work — even when, to a human observer, the quality is obviously mediocre."

This isn't stupidity. It's a structural bias: **the same context that produced the answer is the context evaluating the answer**. You need external judgment.

### 3. Coherence Drift

**Agents go off the rails over time.**

Extended projects cause agents to slowly lose track of the original goal. They over-correct on small issues, get stuck in refactoring loops, or pursue tangents that felt relevant 10 turns ago but aren't now.

> "Breaking work into tractable chunks prevents degradation." — Anthropic Engineering

These three failure modes are not model bugs. They are **structural properties of single-context, single-agent loops**. The only known fix is structural: change the architecture around the model.

That structural fix is the harness.

---

## Harness, Defined

> "Harness design refers to the structured scaffolding that enables AI agents to effectively complete complex, extended tasks. Rather than relying on a single model pass, harnesses decompose work into specialized agent roles with feedback loops." — Anthropic Engineering

A harness is **the code and structure around your model** — the loop, the sub-agents, the file system, the tools, the evaluators. It's everything that isn't the model call itself.

### The Anatomy of a Modern Harness (Planner–Generator–Evaluator)

Anthropic's recommended pattern is inspired by Generative Adversarial Networks. Three specialized roles:

```
    ┌──────────┐
    │ Planner  │ ← expands user request into detailed spec
    └────┬─────┘
         │ contract
         ▼
    ┌──────────┐
    │Generator │ ← implements features incrementally
    └────┬─────┘
         │ artifact
         ▼
    ┌──────────┐
    │Evaluator │ ← tests via actual interaction, returns feedback
    └────┬─────┘
         │
         └──── (feedback loop back to Generator)
```

Each role has a **separate context**, **separate role**, and **separate failure mode**. Critically, the Evaluator is never the same agent as the Generator. This fixes self-evaluation bias at the architectural level.

### Why "Harness"?

The word comes from two places:

1. **Automotive/aerospace**: a harness is the wiring that connects many components into a functioning system.
2. **Testing**: a "test harness" is the scaffolding that exercises code under realistic conditions.

Both metaphors fit. Your LLM is a component. The harness is what makes the component into a system.

---

## The Evolution: Prompt → Context → Harness

Harness Engineering is the third wave of LLM engineering disciplines.

```
  2022             2023             2024+
  ────             ────             ─────
 Prompt   →      Context   →      Harness
Engineering   Engineering    Engineering
  ────             ────             ─────
"What do I ask?" "What do I show?" "How do I loop?"
```

### Era 1: Prompt Engineering (2022–2023)

**Unit of work**: A single message.

Techniques: few-shot examples, chain-of-thought, role prompting, delimiters, output formatting.

**Why it stopped being enough**: Single calls can't handle tasks with unpredictable steps. You can't prompt your way out of a 100-step task.

### Era 2: Context Engineering (2023–2024)

**Unit of work**: The input window.

Techniques: RAG, retrieval, chunking, compression, re-ranking, dynamic context assembly.

**Why it stopped being enough**: A well-constructed context doesn't prevent the agent from drifting over 50 turns. Context engineering assumes one shot. Long-running tasks are many shots.

### Era 3: Harness Engineering (2024+)

**Unit of work**: The entire system around the model.

Techniques: multi-agent roles, feedback loops, file-system state, checkpoints, evaluators, budgets, orchestration.

**Why it's emerging now**: Models are finally good enough that the bottleneck has moved. When the model can reliably do a 5-step task, the question becomes "can I chain 50 of them together without the whole thing collapsing?"

### Summary Table

| Dimension | Prompt Engineering | Context Engineering | Harness Engineering |
|---|---|---|---|
| **Focus** | Single message | Input window | Entire system |
| **Time scale** | One turn | Multi-turn chat | Hours to days |
| **Key techniques** | Few-shot, CoT, role | RAG, chunking | Multi-agent, loops, state |
| **Failure mode** | Model misunderstands | Context overflow, irrelevance | Drift, self-praise, collapse |
| **Core question** | What do I write? | What do I include? | How do I structure? |

One-line summary:

- **Prompt Engineering**: Ask better questions.
- **Context Engineering**: Include better information.
- **Harness Engineering**: Build a better environment.

---

## Workflow vs Agent — A Critical Distinction

Anthropic draws a sharp line that every harness designer needs to internalize:

> **Workflows**: "LLMs and tools operate through predefined code paths" — orchestration is determined upfront.
>
> **Agents**: "LLMs dynamically direct their own processes and tool usage, maintaining control over how they accomplish tasks."

**This distinction matters because workflows are cheaper, more debuggable, and more predictable than agents.**

| | Workflow | Agent |
|---|---|---|
| **Control flow** | Predefined | LLM-decided |
| **Debugging** | Easy | Hard |
| **Cost predictability** | High | Low |
| **Adapts to novel situations** | No | Yes |
| **Best for** | Predictable tasks | Open-ended tasks |
| **Examples** | Prompt chains, routers | Claude Code, Devin |

### The Decision Tree

```
Do you know the steps upfront?
│
├── Yes ──→ Can a single LLM call solve it?
│          │
│          ├── Yes ──→ Use a single call. Stop.
│          │
│          └── No ───→ Use a WORKFLOW
│                      (prompt chain, router, parallel fan-out)
│
└── No ───→ Do the steps depend on runtime feedback?
           │
           ├── Yes ──→ Use an AGENT
           │          (with guardrails, budgets, and evaluators)
           │
           └── No ───→ Reconsider. You probably know more than you think.
```

> **Tip**: Most "agent" projects should be workflows in disguise. Start with a workflow, escalate only when you hit a real dead-end.

---

## The 5 Building Blocks (Workflow Patterns)

Anthropic's engineering team distilled agent patterns into **5 reusable building blocks**. Understand these and you can assemble most harnesses.

### 1. Prompt Chaining

Decompose a task into sequential steps; each step's output feeds the next.

```
[Input] → [Step 1: Extract] → [Step 2: Classify] → [Step 3: Format] → [Output]
```

**Use when**: The task has clear sub-steps that each benefit from focused prompting.

### 2. Routing

Classify the input, then dispatch to a specialized handler.

```
                 ┌──→ [Handler A: refund queries]
[Input] → [Router] ──→ [Handler B: technical issues]
                 └──→ [Handler C: general questions]
```

**Use when**: Different categories of input need very different prompts or tools.

### 3. Parallelization

Run independent subtasks simultaneously (sectioning), or run the same task multiple times for diverse outputs (voting).

```
                ┌──→ [Subtask 1]
[Input] ────────┼──→ [Subtask 2] ──→ [Aggregate]
                └──→ [Subtask 3]
```

**Use when**: Subtasks are independent, or you want consensus from multiple attempts.

### 4. Orchestrator–Workers

One LLM breaks down the task dynamically and delegates to worker LLMs.

```
[Input] → [Orchestrator]
              │
              ├── dispatches → [Worker A]
              ├── dispatches → [Worker B]
              └── dispatches → [Worker C]
                      │
                      └── [Orchestrator synthesizes]
```

**Use when**: You can't predict the subtasks in advance, but the overall structure is stable.

### 5. Evaluator–Optimizer

One LLM generates; another LLM critiques. Iterate until the critic is satisfied.

```
[Input] → [Generator] → [Draft]
              ▲            │
              │            ▼
              │        [Evaluator]
              │            │
              └── feedback ┘
```

**Use when**: The output is refinable (translations, writing, code), and you have a concrete quality signal.

### How These Compose

Real harnesses combine these. Claude Code, for instance, uses **orchestrator–workers** at the top level (main agent → subagents), **evaluator–optimizer** inside code generation (write → test → fix), and **routing** when deciding which tool to use.

---

## When You Don't Need a Harness

Before you build anything multi-agent, read this carefully:

> **Common Pitfall: Over-engineering.**

Anthropic's advice is blunt: *"Start by using LLM APIs directly: many patterns can be implemented in a few lines of code."*

You do **not** need a harness when:

- A **single LLM call** solves your task.
- **Retrieval augmentation** (RAG) is enough.
- **In-context examples** get you where you need to go.
- Your task has **clear, short steps** and no unpredictability.
- You're still in **exploration mode** and don't know what you're measuring.

You **do** need a harness when:

- Tasks have **unpredictable step counts**.
- You need **autonomous operation across many turns**.
- The agent must **respond to environmental feedback** to make decisions.
- You need **meaningful human oversight** at specific points.
- You're willing to invest in **sandboxed testing** and **iteration**.

### Cognition's Counter-Point

Not everyone agrees multi-agent is the answer. Cognition Labs (makers of Devin) published a widely-cited post titled **"Don't Build Multi-Agents"**, arguing that:

- Multi-agent systems have **coordination overhead** that often exceeds their benefits.
- A **single powerful agent with good tools** outperforms naive multi-agent designs.
- **Context sharing** between agents is the hidden cost that kills throughput.

The takeaway isn't "multi-agent bad" — it's that **harness complexity must be earned through measured improvement**, not assumed.

> **Try this**: Before adding an agent, write down the specific failure you're solving. If you can't, you don't need the agent yet.

---

## The Core Discipline: Keep It Simple, Iterate Ruthlessly

The single most important principle of Harness Engineering:

> **"The best harness is the simplest one that still works."** — Anthropic

As models improve, assumptions baked into your harness become obsolete. The patterns you invented to compensate for GPT-4's weaknesses aren't needed for Claude Opus 4.6. Your harness will keep collecting cruft unless you actively remove it.

The practice:

1. **Start with the minimal harness** that could possibly work.
2. **Measure** where it fails.
3. **Add structure** only to address measured failures.
4. **Test removal** of components periodically — they may no longer be load-bearing.

---

## Key Takeaways

- **The bottleneck has moved.** Model quality is good enough; now the scaffolding around the model is what limits you.
- **Three failure modes** plague long-running agents: context anxiety, self-evaluation bias, and coherence drift. All three are structural, not model-based.
- **Harness Engineering is the third wave** after prompt and context engineering, operating at the system level rather than the message or window level.
- **Planner–Generator–Evaluator** is the canonical harness anatomy. The separation of evaluation from generation is the most important architectural decision.
- **Workflows beat agents** when the steps are predictable. Escalate only when necessary.
- **Five building blocks** (chaining, routing, parallelization, orchestrator–workers, evaluator–optimizer) compose into most real harnesses.
- **Simple first, complex later.** Every harness component must earn its place through measured improvement.

---

## What's Next

In [**Part 2**](/posts/harness-engineering-part2-meta-harness-research/), we'll look at what happens when you let the AI design its own harness.

A recent paper — *Meta-Harness* — uses Claude Code as an autonomous proposer with full filesystem access to previous attempts. It beats hand-designed state-of-the-art baselines by **7.7 points** on text classification, ranks **#1 on TerminalBench-2** for Haiku, and reveals a counter-intuitive truth about harness design: **rich feedback beats clever structure**.

If you're building harnesses by hand today, Part 2 will change how you think about the work.
