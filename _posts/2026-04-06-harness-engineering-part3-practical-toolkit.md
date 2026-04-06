---
title: "Harness Engineering Part 3: In Practice — Tools, Patterns, and Starting Points"
date: 2026-04-06 00:50:00 +0900
categories: [AI & ML, AI Agent]
tags: [harness-engineering, langgraph, crewai, autogen, claude-code, revfactory, gstack, openharness, ai-agent]
---

**Harness Engineering: From Concept to Practice**

*This is Part 3 of a 3-part series on Harness Engineering.*

- **Part 1:** [Why Harness Engineering Matters — The Shift Beyond Prompts and Context](/posts/harness-engineering-part1-why-harness/)
- **Part 2:** [Self-Improving Harnesses — Lessons from Meta-Harness Research](/posts/harness-engineering-part2-meta-harness-research/)
- **Part 3** (this post): In Practice — Tools, Patterns, and Starting Points

---

## Tomorrow Morning, You Open Your Editor

You've read [Part 1](/posts/harness-engineering-part1-why-harness/) on why harness engineering matters.

You've read [Part 2](/posts/harness-engineering-part2-meta-harness-research/) on how AI can design its own harness.

Now it's Monday morning. You have an agent to ship. Your Slack is pinging. You open your editor.

**What do you actually type?**

This post is the practical toolkit. We'll walk through the OSS landscape, how to pick a framework, how to use `revfactory/harness` specifically, five real-world scenarios you're probably facing, how to evaluate what you build, and a 5-week roadmap to become competent at this.

No more theory. Let's build.

---

## Core Principles Cheat Sheet

Pin this to your wall. These principles come from Anthropic's engineering team plus the Meta-Harness research we covered in Part 2.

### The Big Three

1. **Simplicity first.** Start with LLM API calls. Add structure only when measured failure demands it.
2. **Transparency.** Every decision, every tool call, every reasoning step — log it. Invisible agents are undebuggable agents.
3. **ACI (Agent-Computer Interface).** Tool docs deserve as much care as UX docs. A tool with a bad description is a broken tool.

### Do / Don't

**Do**

- Separate your **planner, generator, and evaluator** into different agents
- Use **files as source of truth**, not context window
- Make subjective goals **measurable** before building
- **Log everything** (you learned this in Part 2)
- Set **budgets** — max iterations, max tokens, max time
- Include **human checkpoints** for high-stakes decisions
- **Test removal** of harness components periodically

**Don't**

- Let agents **self-evaluate** critical work
- Build **agents when workflows suffice**
- Let **context grow unbounded**
- Trust **frameworks blindly** without understanding the primitives
- Add agents **prematurely**
- Skip **tool documentation** — agents read it

---

## The Workflow vs Agent Decision

Before you pick a framework, decide what kind of system you're building.

```
                  ┌──────────────────────────────┐
                  │ Do you know the steps in     │
                  │ advance?                     │
                  └──────┬──────────────┬────────┘
                         │ Yes          │ No
                         ▼              ▼
              ┌──────────────────┐  ┌──────────────────┐
              │ Can one LLM call │  │ Does branching   │
              │ handle it?       │  │ depend on runtime│
              └──┬────────────┬──┘  │ feedback?        │
                 │ Yes        │ No  └──┬────────────┬──┘
                 ▼            ▼        │ Yes        │ No
              ┌──────┐  ┌──────────┐   ▼            ▼
              │Single│  │Workflow  │ ┌──────┐  ┌──────────┐
              │ call │  │(5 ptns)  │ │Agent │  │Workflow  │
              └──────┘  └──────────┘ │(full)│  │(router)  │
                                     └──────┘  └──────────┘
```

**Rule of thumb**: If you can draw a flowchart of the steps before you run it, you want a **workflow**. If you can't, you want an **agent**.

> **Tip**: Most "I need an agent" projects are actually "I need a workflow with a router and an evaluator."

---

## The 3-Layer OSS Landscape

The harness tooling ecosystem is starting to stratify into clear layers.

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 3: Meta / Auto-Harness                               │
│  ─────────────────────────────────                          │
│  • gstack — opinionated sprint workflow (23 skills)         │
│  • revfactory/harness (Claude Code plugin)                  │
│  • harness-100 (template library)                           │
│  • Meta-Harness (research, self-improvement loops)          │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  LAYER 2: High-Level Agent Frameworks                       │
│  ─────────────────────────────────                          │
│  • CrewAI — role-playing agents + flows                     │
│  • AutoGen — conversational multi-agent                     │
│  • OpenAI Swarm — lightweight handoff                       │
│  • LangChain Agents — high-level abstractions               │
│  • Claude Code — agentic CLI                                │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  LAYER 1: Low-Level Orchestration                           │
│  ─────────────────────────────────                          │
│  • LangGraph — StateGraph + checkpointing + HITL            │
│  • OpenHarness — lightweight open-source agent harness      │
│  • Temporal — durable workflow orchestration                │
│  • Prefect — dataflow orchestration                         │
│  • Raw SDK (Anthropic, OpenAI) — direct primitives          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Direction of dependency**: Layer 2 sits on top of Layer 1 conceptually; Layer 3 sits on top of Layer 2.

**Rule of thumb for starting out**:

- **Experimenting?** Start at Layer 1 (raw SDK or LangGraph).
- **Building production?** Layer 1 (LangGraph) or Layer 2 (CrewAI/AutoGen).
- **Shipping fast in Claude Code?** Layer 3 (`revfactory/harness`).

> **Common Pitfall**: Picking Layer 2 without understanding Layer 1. When CrewAI does something weird, you need to understand what's happening underneath.

---

## Framework Selection Matrix

| Framework | Layer | Philosophy | Best for | Avoid if |
|---|---|---|---|---|
| **Raw SDK** | L1 | No framework | Learning, prototyping | You need state, retries |
| **LangGraph** | L1 | Low-level orchestration | Production stateful agents | You want fast DX |
| **Temporal** | L1 | Code-as-workflow | Long-running, durability critical | Task is short/stateless |
| **AutoGen** | L2 | Conversational multi-agent | Agents that "talk to each other" | You want tight control |
| **CrewAI** | L2 | Role metaphor | Non-developer-friendly team design | You need low-level control |
| **OpenAI Swarm** | L2 | Minimal handoff | Learning, simple multi-agent | Production systems |
| **Claude Code** | L2/3 | Agentic CLI | Coding tasks, file ops | Non-coding workflows |
| **OpenHarness** | L1 | Lightweight open-source harness | Multi-provider agents, research | Need battle-tested production infra |
| **revfactory/harness** | L3 | Meta-skill for Claude Code | Quick team-of-agents prototyping | Non-Claude-Code users |
| **gstack** | L3 | Opinionated sprint workflow | Shipping production code in Claude Code | Non-standard dev workflows |

### Quick Decision Tree

```
What environment are you in?
│
├── Already using Claude Code?
│     │
│     ├── Shipping production code (plan → build → QA → deploy)?
│     │     └─→ gstack (opinionated sprint workflow)
│     │
│     ├── Need domain-specific agent teams?
│     │     └─→ revfactory/harness (auto-generated teams)
│     │
│     └── Unsure?
│           └─→ Start with gstack for shipping, harness for exploration
│
├── Building production customer-facing?
│     │
│     ├── Needs HITL + checkpoints?
│     │     └─→ LangGraph
│     │
│     └── Multi-agent conversational?
│           └─→ AutoGen or CrewAI
│
├── Want multi-provider flexibility (open source)?
│     │
│     └─→ OpenHarness (Claude + OpenAI + GitHub Copilot)
│
├── Research / experimentation?
│     │
│     └─→ Raw SDK, then LangGraph or OpenHarness
│
└── Non-developer stakeholders in the loop?
      │
      └─→ CrewAI (roles are readable)
```

---

## Deep Dive 1: LangGraph

LangGraph is the **low-level orchestration framework** of choice for production stateful agents. Companies like Klarna, Uber, and J.P. Morgan use it in production.

### Core Concepts

- **StateGraph**: your agent is a graph where nodes are functions and edges are control flow
- **Nodes**: processing functions that read/write shared state
- **Edges**: static or conditional, dictate flow between nodes
- **Checkpointing**: every state change saved, agent can resume from failure
- **Human-in-the-loop (HITL)**: pause at any node, wait for human input, resume

### Minimal Example

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class State(TypedDict):
    messages: list
    plan: str
    output: str

def planner(state: State) -> State:
    # LLM call that produces a plan
    state["plan"] = generate_plan(state["messages"])
    return state

def executor(state: State) -> State:
    # LLM call that executes the plan
    state["output"] = execute(state["plan"])
    return state

def should_continue(state: State) -> str:
    if needs_more_work(state["output"]):
        return "executor"
    return END

graph = StateGraph(State)
graph.add_node("planner", planner)
graph.add_node("executor", executor)
graph.add_edge(START, "planner")
graph.add_edge("planner", "executor")
graph.add_conditional_edges("executor", should_continue)

app = graph.compile(checkpointer=MemorySaver())
```

### Why Teams Pick LangGraph

- **Durable execution**: failures don't lose state; resume from checkpoint
- **HITL built in**: pause and wait for human approval at any node
- **Memory**: short-term (thread) + long-term (cross-session)
- **Production deployment**: mature infra, observability integrations

### When LangGraph Is Overkill

If your whole app is a single LLM call and a retry loop, you don't need a graph framework. Use the raw SDK.

> **Try this**: Re-implement Anthropic's 5 workflow patterns from Part 1 as LangGraph StateGraphs. It's the fastest way to learn.

---

## Deep Dive 2: revfactory/harness + harness-100

If you use Claude Code, `revfactory/harness` is worth 30 minutes of your time. It's a **meta-skill plugin** that automatically designs domain-specific agent teams.

### Installation

**Via Claude Code marketplace**:

```bash
/plugin marketplace add revfactory/harness
/plugin install harness@harness
```

**Direct install**:

```bash
cp -r skills/harness ~/.claude/skills/harness
```

**Required environment variable**:

```bash
export CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1
```

### How You Use It

Just ask in natural language:

```
"Build a harness for this project"
"Design an agent team for API documentation"
"Set up a harness for code review with parallel specialists"
```

The plugin will analyze your domain, choose an orchestration pattern, and generate agents + skills.

### The 6 Orchestration Patterns

The plugin supports six patterns for team composition:

| Pattern | Flow | Best for |
|---|---|---|
| **Pipeline** | `A → B → C` | Sequential dependent tasks |
| **Fan-out / Fan-in** | `A → {B1, B2, B3} → C` | Parallel independent work |
| **Expert Pool** | Dispatcher → chosen expert | Context-dependent specialization |
| **Producer–Reviewer** | Producer → Reviewer (loop) | Quality gates |
| **Supervisor** | Central agent distributes | Dynamic task allocation |
| **Hierarchical Delegation** | Top-down recursive | Deep task decomposition |

### What It Generates

After invocation, the plugin creates this structure:

```
.claude/
├── agents/
│   ├── analyst.md       ← domain analysis agent
│   ├── builder.md       ← implementation agent
│   └── qa.md            ← quality verification agent
└── skills/
    ├── analyze/
    │   └── SKILL.md     ← scoped skill for analyst
    └── build/
        └── SKILL.md     ← scoped skill for builder
```

Each agent is a Markdown file with role, responsibilities, and tools. Each skill is a focused capability the agent can invoke.

### harness-100: Template Library

`revfactory/harness-100` is a companion repo with **100 production-ready harness templates** across 10 domains:

- Content creation (webtoon, YouTube, blog series)
- Software development (code review, refactoring, debugging)
- AI/data (pipeline design, evaluation, dataset curation)
- Healthcare
- Marketing
- Research synthesis
- API documentation
- And more

**Use it as reference**, not blind copy-paste. Each template shows how a real harness was composed — which pattern, which agents, which skills.

### Reported Performance

A/B testing by the author showed:

- **+60% quality improvement** (79.3 vs 49.5 baseline)
- **100% win rate** across 15 software engineering tasks
- Effectiveness scales with task complexity

> **Caveat**: token consumption is significant. Users with smaller Claude Max plans reported hitting limits quickly. Budget accordingly.

---

## Deep Dive 3: gstack — The Sprint Workflow Harness

If `revfactory/harness` is "generate an agent team for any domain," **gstack is "here's the exact team you need to ship production code."** Built by YC CEO Garry Tan, who used it to ship **600,000+ lines of code in 60 days** while running YC full-time. 64.8k GitHub stars.

### Philosophy

gstack treats the software development cycle as a structured sprint:

```
Think → Plan → Build → Review → Test → Ship → Reflect
```

Each stage has **dedicated skills** that embody specific professional roles — not a generalist assistant, but a CEO reviewer, an engineering lead, a QA tester, a security officer. The model isn't doing everything; each skill constrains the agent to one perspective.

### 23 Specialized Skills

| Stage | Skills | What they do |
|---|---|---|
| **Plan** | `/office-hours`, `/plan-ceo-review`, `/plan-eng-review`, `/autoplan` | Product discovery, architecture validation, sprint planning |
| **Design** | `/design-consultation`, `/design-shotgun`, `/design-html` | Design systems, variant generation, production HTML |
| **Build** | `/browse`, `/codex`, `/investigate` | Persistent browser automation, multi-AI code review, debugging |
| **QA** | `/qa` | Real browser testing, atomic commits per bug, health scoring |
| **Review** | `/review` | Multi-stage PR analysis, parallel specialist review, adversarial review (Claude + Codex) |
| **Security** | `/cso` | 15-phase vulnerability scan (OWASP + STRIDE) |
| **Ship** | `/ship`, `/land-and-deploy`, `/canary` | Pre-merge automation, merge-to-prod, staged rollout |
| **Reflect** | `/retro` | Weekly metrics across projects |

### The Killer Feature: Persistent Browser QA

gstack runs a **long-lived headless Chromium daemon** — not a fresh browser per command. This means:

- **100-200ms command execution** (vs. 2-3 seconds for fresh launches)
- **Persistent login state** — the QA agent stays authenticated across tests
- **Real UI interactions** — clicks, forms, navigation, not mocked endpoints
- **Before/after diffs** — visual regression detection

```
┌─────────────────────────────────────────────────────┐
│                gstack /qa                            │
│                                                     │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐   │
│  │ Browse   │────→│ Find bug │────→│ Fix +    │   │
│  │ real app │     │ (UI/UX)  │     │ atomic   │   │
│  │ (100ms)  │     │          │     │ commit   │   │
│  └──────────┘     └──────────┘     └────┬─────┘   │
│                                         │          │
│                                         ▼          │
│                                    [Next bug]      │
│                                    loop until       │
│                                    health score OK  │
└─────────────────────────────────────────────────────┘
```

### Adversarial Code Review

The `/review` skill doesn't just run one AI reviewer — it dispatches **independent Claude + Codex reviewers**, then deduplicates findings. Different models catch different blind spots. This is defense-in-depth for code quality.

### When gstack Fits

- **Individual developers or small teams shipping production code** — the sprint workflow matches real eng practice
- **You want strong opinions** — gstack tells you *how* to work, not just *what tools* to use
- **You need real QA** — persistent browser automation is unique in this space

### When gstack Doesn't Fit

- **Non-standard development workflows** — the 23 skills are opinionated and may clash
- **You're not on Claude Code** — gstack is tightly coupled to the Claude Code runtime
- **You need domain-specific agents** — gstack is for software development, not arbitrary domains (use `revfactory/harness` for that)

### gstack vs revfactory/harness

| | gstack | revfactory/harness |
|---|---|---|
| **Approach** | Prescriptive: "follow this sprint" | Generative: "describe your domain, I'll design a team" |
| **Skills** | 23 pre-built, software-dev focused | Auto-generated per domain |
| **Customization** | Templates (.tmpl) | Natural language prompts |
| **Strength** | Shipping code fast with quality gates | Flexible agent composition for any domain |
| **Stars** | 64.8k | 2.5k |

**They're complementary, not competing.** Use gstack for your daily coding workflow. Use `revfactory/harness` when you need a custom agent team for a non-coding domain.

---

## Deep Dive 4: OpenHarness — The Lightweight Open-Source Reference

[OpenHarness](https://github.com/HKUDS/OpenHarness) (HKUDS, Hong Kong University) is the **first open-source implementation of the formal "Agent Harness" pattern** defined in recent research (Meta-Harness, Natural-Language Agent Harnesses). It delivers Claude-Code-like functionality in **~11,700 lines of Python** — roughly 3% of Claude Code's codebase.

### Why It Matters

OpenHarness is not trying to replace Claude Code or LangGraph. It's a **reference implementation** that answers: "What does a minimal, functional agent harness actually look like in code?"

If LangGraph is the production-grade framework and Claude Code is the polished product, OpenHarness is the **textbook implementation** — readable, hackable, and provider-agnostic.

### Architecture (10 Subsystems)

```
┌──────────────────────────────────────────────┐
│              OpenHarness Core                 │
│                                              │
│  ┌────────┐  ┌────────┐  ┌────────┐         │
│  │ Engine │  │Toolkit │  │ Skills │         │
│  │ (loop) │  │(43+tools)│ │(40+ md)│        │
│  └───┬────┘  └───┬────┘  └───┬────┘         │
│      └───────────┼───────────┘               │
│                  ▼                            │
│  ┌────────┐  ┌────────┐  ┌────────┐         │
│  │Plugins │  │Permis- │  │Memory  │         │
│  │        │  │sions   │  │& State │         │
│  └────────┘  └────────┘  └────────┘         │
│                  ▼                            │
│  ┌────────┐  ┌────────┐  ┌────────┐         │
│  │ Swarm  │  │ Hooks  │  │ Config │         │
│  │(multi) │  │        │  │ & UI   │         │
│  └────────┘  └────────┘  └────────┘         │
└──────────────────────────────────────────────┘
         │              │              │
         ▼              ▼              ▼
    Anthropic       OpenAI-       GitHub
    (Claude)      compatible      Copilot
                  (DeepSeek,
                   Ollama, ...)
```

### Multi-Provider: The Key Differentiator

Unlike Claude Code (Anthropic-only) or gstack (Claude Code-dependent), OpenHarness supports:

- **Anthropic format**: Claude, Moonshot/Kimi, Bedrock, Vertex
- **OpenAI format**: OpenAI, DeepSeek, Groq, Ollama, local instances
- **GitHub Copilot**: OAuth device flow, no API keys required

**This makes it the only harness in this list that lets you switch providers without changing your harness code.** For cost-conscious teams or researchers benchmarking across models, this is significant.

### What's Good

- **Code density**: 11,700 lines for near-parity with Claude Code is remarkable engineering
- **Anthropic ecosystem compatibility**: CLAUDE.md files, skills, and plugins work unchanged
- **Production signals**: 114+ tests, exponential backoff, token/cost tracking, permission model
- **Hackability**: Small codebase means you can read and modify the entire harness

### What's Early

- **v0.1.0** (released April 1, 2026) — 5,450 stars but only days old
- **No published benchmarks** against Claude Code or LangGraph
- **Multi-agent patterns exist** (Swarm subsystem) but lack real-world examples
- **Documentation** is solid at README level but thin on architecture deep-dives

### When to Use OpenHarness

- **You want to understand harness internals** — read the source as a learning exercise
- **You need multi-provider flexibility** — swap between Claude, GPT, DeepSeek, Ollama
- **You're extending the harness for research** — small codebase means fast iteration
- **You want Claude-Code-like features without vendor lock-in**

### When Not to Use OpenHarness

- **You need battle-tested production infrastructure** — use LangGraph
- **You want a polished UX** — use Claude Code directly
- **You need a rich ecosystem of integrations** — LangChain/LangGraph has far more

> **Watch this space**: If OpenHarness delivers on its benchmarks and the community grows, it could become the "SQLite of agent harnesses" — the small, embeddable, reliable option that everyone reaches for when they don't need a full framework.

---

## 5 Real-World Scenarios (And What to Use)

Here are the scenarios you're probably facing, matched to tools and patterns.

### Scenario A: Small Task, 1-2 Agents

**Example**: "Summarize this document and classify it into one of 5 categories."

**Recommendation**: Raw SDK + prompt chaining.

```
[Input] → [Summarize] → [Classify] → [Output]
```

- **Tools**: Anthropic/OpenAI SDK directly
- **Pattern**: Prompt chaining
- **Don't**: Reach for a framework. You don't need it yet.

### Scenario B: Medium Team, 3-5 Agents

**Example**: "Write a technical blog post that's researched, drafted, fact-checked, and edited."

**Recommendation**: CrewAI or AutoGen with Producer-Reviewer.

```
[Researcher] → [Writer] → [Fact-checker] → [Editor]
                              │                │
                              └── feedback ────┘
```

- **Tools**: CrewAI (readable roles) or AutoGen (conversational)
- **Pattern**: Orchestrator-workers + evaluator-optimizer
- **Don't**: Let Writer be the Fact-checker. Separate them.

### Scenario C: Long-Running Stateful

**Example**: "Customer support agent that handles multi-day escalations with human approval gates."

**Recommendation**: LangGraph with checkpointing and HITL.

```
[Triage] → [Route] → [Handler] → [Human approval] → [Resolve]
                         ▲                             │
                         └── retry if rejected ────────┘
```

- **Tools**: LangGraph (mandatory for durability), LangSmith for tracing
- **Pattern**: Router + HITL
- **Don't**: Try to keep state in prompts. Use checkpointer.

### Scenario D: Claude Code Native

**Example**: "I'm already in Claude Code, I need to ship a feature with plan, build, QA, and review."

**Recommendation**: gstack for the sprint workflow. Use `/autoplan` → build → `/qa` → `/review` → `/ship`.

```
[/autoplan] → [Build] → [/qa (browser)] → [/review (adversarial)] → [/ship]
                             │                      │
                             └── fix + atomic ───────┘
                                 commit per bug
```

- **Tools**: gstack (23 skills covering the full sprint cycle)
- **Pattern**: Pipeline with evaluator-optimizer loops at QA and review stages
- **Don't**: Skip `/qa`. The persistent browser automation catches real UI bugs that unit tests miss.

**Alternative**: If your task is a custom domain (not standard software dev), use `revfactory/harness` to generate a domain-specific agent team instead.

```
[Auditor]
    ├── dispatch → [Security specialist]
    ├── dispatch → [Performance specialist]
    ├── dispatch → [Architecture specialist]
    └── merge ────── [Report aggregator]
```

- **Tools**: `revfactory/harness` + harness-100 templates
- **Pattern**: Fan-out / Fan-in

### Scenario E: Research / Experimental

**Example**: "I want to find the best harness for my task. Let the AI figure it out."

**Recommendation**: Meta-Harness-style loop (covered in Part 2).

```
[Baseline] → [Evaluate] → [Log to filesystem] → [Proposer reads history]
                                                         │
                                                         ▼
                                                  [New candidate]
                                                         │
                                                         └── loop
```

- **Tools**: Claude Code as proposer, simple eval harness, filesystem logging
- **Pattern**: Self-improvement loop
- **Don't**: Underestimate token cost. This is expensive per iteration.

---

## Evaluation and A/B Testing Your Harness

A harness you can't evaluate is a harness you can't improve. Here's the minimum evaluation setup.

### Quantitative Metrics

- **Task completion rate** — did the harness finish the task correctly?
- **Average iterations to completion** — did it converge efficiently?
- **Token cost per successful task** — what does each success actually cost?
- **Failure mode distribution** — where does it break?
- **Wall-clock time** — especially for long-running scenarios

### Qualitative Metrics

- **Human preference** in A vs B comparisons
- **Readability** of produced artifacts
- **Recoverability** from errors

### A/B Test Protocol

```
1. Define baseline harness H0
2. State hypothesis: "Splitting planner from generator → +X% success"
3. Build variant H1 (minimal change from H0)
4. Run H0 and H1 each N times on the same task set
5. Paired comparison on identical tasks
6. Compute 95% confidence interval on the delta
7. If significant, promote H1. If not, discard.
```

> **Try this**: Before your next harness change, write down what you expect to see. If you're wrong more than half the time, your mental model of the harness is miscalibrated — that's worth knowing.

### Observability Tools

| Tool | Strength |
|---|---|
| **LangSmith** (LangChain) | Best for LangGraph, rich trace UI |
| **Arize Phoenix** | Open-source, framework-agnostic |
| **Braintrust** | Eval-focused, strong comparisons |
| **Helicone** | Routing + observability combined |

All four will give you tracing. Pick based on your stack.

---

## Meta-Harness-Style Self-Improvement: Minimal Sketch

You don't need a research lab to run Meta-Harness-style optimization. Here's a minimal sketch:

```python
import json
from pathlib import Path

HISTORY_DIR = Path("./harness_history")
HISTORY_DIR.mkdir(exist_ok=True)

def log_candidate(iteration, code, score, traces):
    """Save everything to disk, not just the score."""
    candidate_dir = HISTORY_DIR / f"iter_{iteration:03d}"
    candidate_dir.mkdir(exist_ok=True)
    (candidate_dir / "harness.py").write_text(code)
    (candidate_dir / "score.json").write_text(json.dumps({"score": score}))
    (candidate_dir / "traces.jsonl").write_text("\n".join(traces))

def propose_next(history_dir):
    """Claude Code (as proposer) reads history, proposes new candidate."""
    # Claude Code gets filesystem access to history_dir
    # Reads whatever files it chooses
    # Writes a new harness.py
    pass

def evaluate(harness_code, task_set):
    """Run harness on tasks, return (score, traces)."""
    pass

# Main loop
for i in range(N_ITERATIONS):
    code = propose_next(HISTORY_DIR) if i > 0 else baseline_code()
    score, traces = evaluate(code, task_set)
    log_candidate(i, code, score, traces)
```

**Key architectural choices**:

- **History on disk**, not in context window
- **Full traces stored** (Meta-Harness lesson)
- **Proposer chooses what to read** (adaptive inspection)

Start simple: 10 iterations, 5 tasks. See what happens.

---

## 5-Week Learning Roadmap

Want to go from reading this series to shipping a harness? Here's a sequenced path:

### Week 1: Foundations

- [ ] Read Anthropic's "Building effective agents" essay
- [ ] Implement each of the 5 workflow patterns (Part 1) by hand with raw SDK
- [ ] Don't use a framework

### Week 2: Low-level Orchestration

- [ ] Work through LangGraph quickstart
- [ ] Re-implement one workflow pattern as a StateGraph
- [ ] Add a checkpointer, simulate a failure, resume

### Week 3: Claude Code Harness Tools

- [ ] Install gstack and run `/qa` + `/review` on a real project
- [ ] Install `revfactory/harness` and generate a custom agent team
- [ ] Compare: gstack's opinionated sprint vs harness's flexible composition
- [ ] Read 3 templates from harness-100 and analyze their structure

### Week 4: Ship Something

- [ ] Pick a real task you have
- [ ] Design the harness (Planner + Generator + Evaluator)
- [ ] Build it. Measure it. Iterate.
- [ ] Keep notes on what worked, what didn't

### Week 5: Evaluation + Meta-Loops

- [ ] Add LangSmith/Phoenix/Braintrust tracing
- [ ] Run an A/B test on one harness change
- [ ] Sketch a minimal Meta-Harness loop for your task
- [ ] (Optional) Let Claude Code iterate for 5-10 rounds

After week 5, you'll know what you're missing — and you'll know how to find it.

---

## 7 Takeaways (Print This)

1. **Simple first, complex later.** Single call → workflow → agent. Escalate only when necessary.
2. **Separate evaluation from generation.** Never let an agent grade its own work.
3. **Files > context window.** State lives on disk. Context is scratch space.
4. **Rich feedback beats clever structure.** (Meta-Harness lesson: log everything.)
5. **Frameworks hide; direct code reveals.** Know the primitives before you adopt abstractions.
6. **Harness is iterative.** It's a system you evolve, not a template you apply.
7. **Start with templates.** `revfactory/harness-100` or equivalent. Don't start from zero.

---

## Series Wrap-Up

Three posts in, here's the arc:

**Part 1** established that **the scaffolding around your model is now the bottleneck** — and that "harness" is the name of that scaffolding.

**Part 2** showed that **AI can design its own harness** when given the right information diet, and that **richer feedback consistently beats cleverer structure**.

**Part 3** gave you **the tools, patterns, and scenarios** to start building harnesses that work in production today.

If you take one thing from the series: **harness engineering is a measurement-driven discipline**. You don't design a harness once — you evolve it, with your finger on the numbers.

Model capability is catching up. Harness quality is still a moat.

---

## References and Further Reading

**Engineering Blogs**
- Anthropic: "Harness design for long-running LLM applications"
- Anthropic: "Building effective agents"
- Cognition Labs: "Don't Build Multi-Agents"

**Research**
- Meta-Harness (arxiv 2603.28052)

**Research**
- Natural-Language Agent Harnesses (arxiv 2603.25723)

**OSS Projects**
- [gstack](https://github.com/garrytan/gstack) — Sprint workflow harness for Claude Code (64.8k stars)
- [OpenHarness](https://github.com/HKUDS/OpenHarness) — Lightweight open-source agent harness (5.4k stars)
- [revfactory/harness](https://github.com/revfactory/harness) — Meta-skill for Claude Code agent teams
- [revfactory/harness-100](https://github.com/revfactory/harness-100) — 100 harness templates
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [AutoGen](https://github.com/microsoft/autogen)
- [CrewAI](https://github.com/crewAIInc/crewAI)
- [OpenAI Swarm](https://github.com/openai/swarm)
