---
title: "oh-my-* — When One AI Agent Isn't Enough, Build a Team"
date: 2026-04-06 09:44:00 +0900
categories: [AI & ML, AI Agent]
tags: [ai-agent, claude-code, multi-agent, oh-my-claudecode, codex, gemini, orchestration]
---

*A review of the multi-agent orchestration wrappers for Claude Code, Codex, and Gemini CLI*

---

> **TL;DR**: oh-my-claudecode (24.5k stars), oh-my-codex (16.6k stars), and oh-my-gemini are tmux-based orchestration layers that turn single-agent AI coding CLIs into multi-agent teams. The architecture is sound — pipeline stages, parallel workers, skill routing — but the value depends entirely on task scale. For single-file fixes, it's overkill. For feature-level development, it's where things get interesting.

---

## The Problem Everyone Knows But Nobody Solved

You ask Claude Code to implement a login feature. Halfway through, it hits a rate limit. You wait. It resumes. The result has a bug in the auth middleware. You ask it to fix. It fixes the middleware but breaks the test. You sigh.

**The problem isn't the model. It's that one agent is doing everything — planning, coding, testing, reviewing — in a single thread, with a single context window.**

The oh-my-* series asks: **what if we applied software engineering to AI agents themselves?** Divide the work. Run agents in parallel. Verify with a separate reviewer. Auto-recover from failures.

This review examines whether the added complexity pays off — and under what conditions.

---

## What oh-my-* Actually Is

Three projects. One author ([Yeachan Heo](https://github.com/Yeachan-Heo)). One design pattern applied to three different AI coding CLIs.

|                      | oh-my-claudecode       | oh-my-codex              | oh-my-gemini             |
|----------------------|------------------------|--------------------------|--------------------------|
| **Wraps**            | Claude Code CLI        | OpenAI Codex CLI         | Google Gemini CLI        |
| **GitHub**           | 24.5k stars            | 16.6k stars              | Fork-based               |
| **npm**              | oh-my-claude-sisyphus  | —                        | oh-my-gemini-sisyphus    |
| **Core skills**      | 19 specialized agents  | 4 canonical skills       | Skill/role system        |
| **Execution modes**  | Team, Autopilot, Ralph, Ultrawork | $team, $ralph, $ralplan | Team orchestration |

**All three share the same backbone:**

- **tmux** as the parallel execution runtime
- **Structured pipelines** (plan → execute → verify → fix)
- **Skill/role-based** context injection per worker
- **Persistent state** across sessions (`.omc/`, `.omx/`)

Here is the common architecture:

```
┌──────────────────────────────────────────────────────┐
│                    oh-my-* Layer                     │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │ Pipeline │  │   Skill  │  │  State   │           │
│  │  Engine  │  │  Router  │  │ Manager  │           │
│  └─────┬────┘  └─────┬────┘  └─────┬────┘           │
│        └──────────────┼─────────────┘                │
│                       ▼                              │
│              tmux Session Manager                    │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐       │
│  │Worker 1│ │Worker 2│ │Worker 3│ │Worker N│       │
│  └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘       │
└──────┼──────────┼──────────┼──────────┼─────────────┘
       ▼          ▼          ▼          ▼
┌──────────┐┌──────────┐┌──────────┐┌──────────┐
│Claude Code││Claude Code││ Codex CLI││Gemini CLI│
└──────────┘└──────────┘└──────────┘└──────────┘
```

**The wrapper doesn't modify the CLIs.** It launches them in isolated tmux panes, feeds them scoped prompts, and collects the results. The CLIs have no idea they're part of a team.

---

## How the Pipeline Works

**The core value isn't "run more agents." It's "enforce structure before execution."**

The team mode pipeline — the flagship feature across all three projects — runs in five stages:

```
┌──────────┐   ┌──────────┐   ┌──────────────────────┐   ┌──────────┐   ┌──────────┐
│   PLAN   │──→│   PRD    │──→│       EXECUTE        │──→│  VERIFY  │──→│   FIX    │
│          │   │          │   │                      │   │          │   │          │
│ 1 agent  │   │ 1 agent  │   │  N agents parallel   │   │ 1 agent  │   │ targeted │
│ decompose│   │ spec each│   │  ┌────┐┌────┐┌────┐ │   │ review   │   │ repairs  │
│ tasks    │   │ task     │   │  │ W1 ││ W2 ││ W3 │ │   │ all work │   │ only     │
└──────────┘   └──────────┘   │  └────┘└────┘└────┘ │   └──────────┘   └──────────┘
                              └──────────────────────┘
                                                               │
                                                        fail? ─┘──→ back to FIX
```

- **PLAN** — A single agent decomposes the task into subtasks and builds a dependency graph. Decides what can run in parallel.
- **PRD** — Each subtask gets a detailed specification. This becomes the scoped context each worker receives.
- **EXECUTE** — N workers launch in separate tmux panes and code **simultaneously**. Each worker modifies only its assigned scope. Atomic task claims prevent collisions.
- **VERIFY** — A separate agent reviews the combined output. Checks builds, tests, and spec compliance.
- **FIX** — If verification fails, only the broken parts get targeted repairs. No full re-execution.

**Without this pipeline, here's what typically happens with a base CLI:** the agent starts coding immediately, discovers mid-way that the approach won't work, backtracks, loses context, and delivers a half-finished result.

**The pipeline forces the expensive thinking (planning, specification) to happen before the expensive execution (parallel coding).** This is not a new idea — it's how human engineering teams have operated for decades. The novelty is applying it to AI agents.

---

## Three Design Decisions Worth Examining

### Decision 1: tmux as the Runtime

**The wrapper strategy: don't touch the CLI internals — control them from outside.**

Why tmux specifically:

- Each worker gets an **isolated pane** — process separation for free
- Uses the CLI's **stdin/stdout as-is** — no internal API knowledge required
- Developers can `tmux attach` to **watch agents work in real time** — instant debuggability
- Already installed on virtually every development machine

**tmux is the poor man's container orchestration.** What Kubernetes does for microservices, tmux does for AI agents. It's crude, it's simple, and it works.

### Decision 2: Smart Model Routing

oh-my-claudecode routes tasks to different models based on complexity:

```
Task arrives
    │
    ▼
┌───────────┐
│ Complexity│
│ Classifier│
└─────┬─────┘
      │
 ┌────┴────┐
 ▼         ▼
Simple   Complex
 │         │
 ▼         ▼
Haiku    Opus
(fast,   (slow,
cheap)   powerful)

Claim: 30-50% token savings
```

**Not every task needs the strongest model.** Renaming a file with Opus is waste. A variable rename goes to Haiku. Architecture decisions go to Opus. The claimed 30-50% token savings is plausible — most real-world coding tasks are a mix of trivial and complex subtasks.

### Decision 3: Skill Learning System

The most forward-looking feature. How it works:

- **Auto-extracts** recurring problem-solving patterns from sessions
- **Stores** them in `.omc/skills/` (project scope) or `~/.omc/skills/` (user scope)
- **Auto-injects** matching skills when similar problems appear in future sessions
- **Shareable via Git** — the entire team benefits from accumulated patterns

**This is the same compound loop that Karpathy's knowledge base uses — the system gets smarter through use.** The difference: Karpathy's system compounds *knowledge*. oh-my-* compounds *coding patterns*.

---

## When Does the Wrapper Pay Off?

**oh-my-* is not universally better. The value scales with task complexity.**

|                    | Base CLI                    | oh-my-*                          |
|--------------------|-----------------------------|----------------------------------|
| **Execution**      | Single agent, sequential    | N agents, parallel               |
| **Workflow**       | Freeform                    | Structured pipeline              |
| **Context**        | One window, one session     | Per-worker scoped context        |
| **Verification**   | Self-review (same agent)    | Separate verifier agent          |
| **Rate limits**    | Manual wait                 | Auto-resume daemon               |
| **State**          | Session-scoped (mostly)     | Persistent (.omc/, .omx/)        |
| **Learning**       | None (or manual memory)     | Auto-extracted skills            |
| **Overhead**       | Zero                        | tmux + config + N x token cost   |

Where each approach fits on the task spectrum:

```
Task complexity / scope
─────────────────────────────────────────────────────▶

│◄── Base CLI wins ──►│◄── Gray zone ──►│◄── oh-my-* wins ──►│
│                     │                 │                     │
│  - Bug fix          │  - New endpoint │  - Feature module   │
│  - 1-2 file edit    │  - Small feature│  - Multi-component  │
│  - Quick refactor   │  - 3-5 files    │  - 10+ files        │
│  - Explanation      │                 │  - Cross-cutting    │
│                     │                 │    refactor         │
│                     │                 │                     │
│  Overhead > Value   │  Depends on     │  Parallel + Pipeline│
│                     │  your patience  │  > Sequential       │
```

**The simplest heuristic: if you would assign the task to one developer, use the base CLI. If you would form a small team, consider oh-my-*.**

---

## Trade-offs and Honest Concerns

A fair review requires acknowledging the costs:

- **Token cost multiplier** — N agents times orchestration overhead equals **3-10x token consumption** versus a single CLI session. Smart routing claims 30-50% savings, but the absolute cost is still significantly higher.
- **Complexity tax** — tmux configuration, skill management, debugging multi-agent coordination. There's a learning curve to learn the tool that helps you use the tool.
- **CLI dependency risk** — When the base CLI updates, the wrapper can break. Every upstream breaking change is an emergency for the wrapper maintainer.
- **Coordination overhead** — Workers can step on each other's changes. Atomic task claims mitigate this but don't eliminate it entirely.
- **Diminishing returns** — Doubling workers doesn't halve execution time. The parallelizable portion of a codebase has limits. Amdahl's Law applies to AI agents too.

**The existential risk for oh-my-\* is that the base CLIs absorb its features.** Claude Code already has sub-agents and parallel tool calls. Codex and Gemini are adding similar capabilities. If the CLIs natively support team-style orchestration, the wrapper layer becomes redundant.

The counter-argument: oh-my-* is CLI-agnostic and can orchestrate *across* providers — a Claude agent and a Codex agent working on the same codebase. Native features are vendor-locked. Whether that cross-provider value justifies the wrapper remains to be seen.

---

## Verdict

| Aspect                            | Rating      | Note                                    |
|-----------------------------------|-------------|-----------------------------------------|
| **Problem identification**        | Excellent   | Single-agent limits are real            |
| **Architecture**                  | Strong      | Pipeline + parallel + verify is sound   |
| **Practical value (small tasks)** | Low         | Overhead exceeds benefit                |
| **Practical value (large tasks)** | High        | Parallelism + structure pays off        |
| **Long-term viability**           | Uncertain   | Depends on base CLI evolution           |
| **Community traction**            | Strong      | 24.5k + 16.6k stars, active development|

**oh-my-\* solves a real problem with a sound architecture.** The insight — that AI agents need the same engineering principles humans do (division of labor, pipelines, verification, recovery) — is correct and likely to remain relevant even as base CLIs evolve.

The execution is impressive: 24.5k stars on oh-my-claudecode alone, active development across three CLI ecosystems, and a skill learning system that points toward compounding value over time.

But the value proposition has a clear condition. For a bug fix or a two-file refactor, the pipeline overhead costs more than it saves. For building a feature module that touches ten files across three layers — that's where parallel workers, structured verification, and scoped context start earning their keep.

**Use it when your task is too big for one agent. Skip it when it isn't.** That's the entire decision framework.
