---
title: "Agent Tool Interfaces Part 2: Orchestrating Tool Interfaces — From Harness Design to GraphRAG"
date: 2026-04-06 16:00:00 +0900
categories: [AI & ML, AI Agent]
tags: [harness-engineering, graphrag, tool-orchestration, neo4j, ai-agent, mcp, knowledge-graph, agentic-rag, agent-architecture]
---

*You know the interfaces. Now learn how to compose them — and how GraphRAG makes your agent smarter about which tools to use.*

**Agent Tool Interfaces: From Landscape to Orchestration**

*This is Part 2 of a 2-part series on Agent Tool Interfaces.*

- **Part 1:** [How Agents Connect to Tools — The Complete Interface Landscape](/posts/agent-tool-interface-landscape/)
- **Part 2** (this post): Orchestrating Tool Interfaces — From Harness Design to GraphRAG

---

> **TL;DR**: Having 43 tools is not a feature — it's a **liability** if your agent can't pick the right one. This post covers three things: (1) how to **orchestrate** CLI, MCP, and Code Execution in a single harness using the Three-Loop Architecture, (2) how to **route, manage state, and recover from failures** across interfaces, and (3) how **GraphRAG** enables intelligent tool discovery that collapses the schema bloat problem from O(n) to O(1). The punchline: a Tool Knowledge Graph with graph-guided selection achieves **43% accuracy vs 14% baseline** while cutting token cost by 95%.

---

## The 50-Tool Problem

Your agent has access to 43 GitHub tools, 8 Slack tools, 5 database tools, and a handful of file operations. That's 60+ tools loaded into context.

**What happens?**

- **44,000+ tokens** consumed by schemas before any work begins
- The LLM picks the wrong tool **28% of the time** at this scale
- **Perplexity removed MCP support entirely** citing these costs

Vercel discovered the fix: they **cut their tool set by 80%** and performance *improved*. The agent got better when it had *fewer* choices.

> "The best harness is the simplest one that still works." — Anthropic Engineering

But you still need those 60+ tools — just not all at once. The question is: **how do you give the agent the right tools at the right time?**

That's orchestration. And in 2026, the answer involves three interconnected problems: **routing**, **state management**, and **discovery**.

---

## The Three-Loop Architecture

In [Part 1](/posts/agent-tool-interface-landscape/), we mapped six tool interfaces. In production, the top three — CLI, Code Execution, and MCP — layer into a three-loop architecture:

```
┌──────────────────────────────────────────────────────────────┐
│                    Agent System                              │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │              Inner Loop (CLI)                          │  │
│  │                                                        │  │
│  │  git ─── pytest ─── eslint ─── cargo ─── docker        │  │
│  │                                                        │  │
│  │  ● Stateless subprocess   ● Zero auth overhead         │  │
│  │  ● Millisecond latency    ● Billions of training data  │  │
│  │  ● 100% reliability       ● ~1,365 tokens/op           │  │
│  └────────────────────────────────────────────────────────┘  │
│                          │                                    │
│  ┌────────────────────────────────────────────────────────┐  │
│  │             Middle Loop (Code Execution)               │  │
│  │                                                        │  │
│  │  Multi-tool chains ─── Data transforms ─── API combos  │  │
│  │                                                        │  │
│  │  ● Sandboxed (E2B/V8)    ● N steps → 1 round-trip     │  │
│  │  ● Code is inspectable   ● 99.9% cheaper than MCP     │  │
│  │  ● ~600 tokens/op        ● State within sandbox       │  │
│  └────────────────────────────────────────────────────────┘  │
│                          │                                    │
│  ┌────────────────────────────────────────────────────────┐  │
│  │              Outer Loop (MCP + Browser)                │  │
│  │                                                        │  │
│  │  Slack ─── Notion ─── Jira ─── Salesforce ─── Legacy   │  │
│  │                                                        │  │
│  │  ● OAuth 2.1 auth        ● Audit trail                │  │
│  │  ● Structured I/O        ● Self-describing schemas     │  │
│  │  ● Cross-network         ● Browser for no-API targets  │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │        State Layer (Filesystem + Checkpoints)          │  │
│  │  Files as source of truth │ Survives truncation/crash  │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

**Why three loops, not two?**

The original Two-Loop model (CLI inner, MCP outer) missed the middle ground: **multi-step workflows that are too complex for a single CLI command but don't need OAuth or external authentication**. Code Execution fills this gap — it's the place where you chain `fetch → filter → aggregate → format` in a single sandboxed pass instead of 4 sequential MCP calls.

### Claude Code: The Canonical Implementation

Claude Code is the clearest production example of this pattern [[Penligent](https://www.penligent.ai/hackinglabs/inside-claude-code-the-architecture-behind-tools-memory-hooks-and-mcp/)]:

- **Inner Loop**: 8 built-in CLI tools (Bash, Read, Edit, Write, Grep, Glob, Task, TodoWrite)
- **Middle Loop**: Subagents via Task tool — nested conversations with isolated context windows
- **Outer Loop**: MCP servers for external services via `claude mcp add`
- **Unified permissions**: `Bash(git diff *)` and `mcp__server__tool` use identical syntax
- **Subagent scoping**: Security-review agents get Read/Grep/Glob but not Edit/Bash

The LLM routes **naturally** between tools based on task descriptions — no explicit routing logic required. When you ask "check the latest PR and post a summary to Slack," it uses `gh` (CLI) for the PR and the Slack MCP server for posting.

### BKit: PDCA as a Harness Plugin

While Claude Code implements the Three-Loop Architecture **internally**, **BKit** implements it **externally** — as a plugin that wraps Claude Code, Gemini CLI, and Codex CLI with a structured PDCA (Plan-Do-Check-Act) workflow [[GitHub](https://github.com/popup-studio-ai/bkit-claude-code)].

```
┌──────────────────────────────────────────────────────────┐
│                     BKit Layer                           │
│                                                          │
│  Plan ──▶ Design ──▶ Do ──▶ Check ──▶ Act ──▶ Report    │
│                              │                           │
│                     gap-detector agent                   │
│                     (design vs implementation            │
│                      match rate ≥ 90%)                   │
│                              │                           │
│              ┌───────────────┼───────────────┐           │
│              ▼               ▼               ▼           │
│         Opus agents    Sonnet agents    Haiku agents     │
│         (11: complex   (19: implement) (2: light        │
│          reasoning)                     tasks)          │
└──────────────────────────────────────────────────────────┘
                         │
                         ▼
              Claude Code / Gemini CLI / Codex CLI
              (Three-Loop execution underneath)
```

Key design choices that map to harness engineering principles:

- **37 skills** (18 workflow / 18 capability / 1 hybrid) loaded **per phase** — not all at once. This avoids the 50-Tool Firehose anti-pattern.
- **Separate evaluator**: A `gap-detector` agent compares implementation against design docs. Match rate < 70% triggers automatic iteration (max 5 cycles). This is the Planner–Generator–Evaluator pattern enforced at the system level.
- **6-layer hook system**: 18 event types (`SessionStart`, `PreToolUse`, `PostToolUse`, `PreCompact`, `Stop`, etc.) inject context at distinct lifecycle points.
- **Cross-platform**: ~95% code reuse across Claude Code, Gemini CLI, and Codex CLI. Only manifest files and hook event names differ.
- **Context preservation**: PDCA state survives compaction, improving context retention from 30-40% to 75-85%.

BKit demonstrates that **the Three-Loop Architecture isn't just for framework builders** — it can be imposed on existing agents from the outside via hooks, skills, and MCP servers (`bkit-pdca`, `bkit-analysis`).

**References:**
- BKit for Claude Code — [github.com/popup-studio-ai/bkit-claude-code](https://github.com/popup-studio-ai/bkit-claude-code)
- BKit for Gemini — [github.com/popup-studio-ai/bkit-gemini](https://github.com/popup-studio-ai/bkit-gemini)
- BKit for Codex — [github.com/popup-studio-ai/bkit-codex](https://github.com/popup-studio-ai/bkit-codex)

---

## Tool Routing: How to Pick the Right Interface

### Three Routing Strategies

```
┌──────────────────────────────────────────────────────────────┐
│                   Routing Strategy Spectrum                   │
│                                                              │
│  Fast ◄────────────────────────────────────────────► Smart   │
│                                                              │
│  Rule-Based          Hybrid               LLM-Based          │
│  ──────────          ──────               ─────────          │
│  ● 0ms overhead      ● Fast rules first   ● 50-100ms        │
│  ● Deterministic     ● Semantic middle     ● Flexible        │
│  ● Rigid             ● LLM fallback       ● Handles novel    │
│  ● Misclassifies     ● Best of both         combinations     │
│    edge cases        ● 39% cost savings   ● Explainable      │
│                                                              │
│  "if file_op: CLI"   Rules → Router → LLM  "Model picks"    │
└──────────────────────────────────────────────────────────────┘
```

**Rule-Based**: Predefined conditions per task type. If the task matches a known pattern (e.g., "run tests" → CLI), route immediately. Zero latency, but can't handle ambiguity.

**LLM-Based**: The model itself decides which tool to call. This is Claude Code's native approach — tool schemas are in context, and the model generates the appropriate tool call. Maximum flexibility, but 50-100ms additional inference latency per routing decision.

**Hybrid** (consensus best practice): Layer all three for optimal cost/quality [[Patronus AI](https://www.patronus.ai/ai-agent-development/ai-agent-routing), [Requesty](https://www.requesty.ai/blog/intelligent-llm-routing-in-enterprise-ai-uptime-cost-efficiency-and-model)]:

1. **Fast rules** filter obvious cases (file ops → CLI, Slack → MCP)
2. **Semantic router** handles the middle tier (embedding similarity to known patterns)
3. **LLM** tackles edge cases and novel combinations

Measured results: **37-46% reduction in LLM usage**, **32-38% latency improvement**, **39% cost reduction**.

### Cost-Aware Routing

oh-my-claudecode implements **model-level routing**: simple tasks (variable rename) go to Haiku (fast, cheap), complex tasks (architecture decisions) go to Opus. Claimed **30-50% token savings** [[oh-my-claudecode GitHub](https://github.com/anthropics/oh-my-claudecode)].

BKit takes a different approach: **role-based routing with fixed model assignments**. 32 agents are pre-assigned — 11 to Opus (complex reasoning), 19 to Sonnet (implementation), 2 to Haiku (lightweight). The routing is deterministic by agent role, not dynamic by task complexity. This trades flexibility for consistency — every architecture decision always goes through Opus, every code generation always goes through Sonnet.

**The cost hierarchy to internalize:**

| Interface | Cost/op | When to route here |
|-----------|---------|-------------------|
| **CLI** | ~1,365 tokens | CLI tool exists, simple auth |
| **Code Exec** | ~600 tokens | Multi-step, data transforms |
| **MCP** | ~44,026 tokens | OAuth required, no CLI exists |
| **Browser** | ~100K+ tokens | No API at all (vision + actions) |

**Always prefer the cheaper interface** unless a specific requirement (auth, audit, structured output) forces you upward.

---

## State Management: The Hardest Unsolved Problem

Each interface has a **different state model**. This is where most multi-interface harnesses break.

```
┌──────────────────────────────────────────────────────────────┐
│              State Model per Interface                        │
│                                                              │
│  CLI (Bash)         MCP (stdio)        Code Execution        │
│  ──────────         ──────────         ──────────────         │
│  Stateless          Session-based      Sandbox-scoped         │
│  subprocess.        Process persists   State persists         │
│  Env resets         for client         within sandbox         │
│  between calls.     lifetime.          lifetime.              │
│                                                              │
│  MCP (HTTP)         Subagents                                │
│  ──────────         ─────────                                │
│  Stateless.         Separate context                         │
│  Each request       window. Returns                          │
│  is independent.    summary only.                            │
└──────────────────────────────────────────────────────────────┘
```

### Solution 1: Filesystem-as-State (Anthropic's Recommendation)

The core insight from Meta-Harness research [[arxiv:2603.28052](https://arxiv.org/abs/2603.28052)]: **externalize state to the filesystem**.

Three enforced properties:

- **Externalized**: State is written to artifacts, not held in transient context
- **Path-addressable**: Later stages reopen the exact object by file path
- **Compaction-stable**: State survives context truncation, restart, and delegation

This is how Claude Code works — CLAUDE.md, project files, and TODO lists are the state store. The conversation window is disposable; the filesystem is truth.

> "Use files as source of truth, not context window." — Anthropic Engineering [[Best Practices](https://www.anthropic.com/engineering/claude-code-best-practices)]

### Solution 2: LangGraph Checkpointing

Every state change is saved as a checkpoint. The agent can resume from any prior point after failure [[LangGraph Docs](https://www.langchain.com/langgraph)]:

- **Short-term**: Thread-scoped memory (within a session)
- **Long-term**: Cross-session memory (across conversations)
- **Recovery**: If the agent fails at node N, resume from checkpoint N-1
- **Production users**: Klarna, Uber, J.P. Morgan

### Solution 3: Deep Agents CompositeBackend

LangChain's Deep Agents SDK mixes state backends [[Blog](https://www.langchain.com/deep-agents)]:

- **In-memory** for speed
- **Filesystem** for persistence
- **LangGraph Store** for cross-thread memory
- The agent offloads large context to a virtual filesystem to prevent overflow

### Practical Recommendation

| Tool count | State strategy |
|-----------|----------------|
| **< 10 tools** | Filesystem-as-State is sufficient |
| **10-30 tools** | Add LangGraph checkpointing for crash recovery |
| **30+ tools** | CompositeBackend with selective offloading |

---

## Error Handling and Fallback Chains

### The Reliability Math

If each step succeeds 95% of the time, chaining 20 steps gives:

```
0.95^20 = 36% end-to-end success rate
```

This is why production harnesses invest heavily in fallback chains.

### Fallback Patterns

```
┌──────────────────────────────────────────────────────────────┐
│                    Fallback Chain                             │
│                                                              │
│  ┌───────────┐  timeout  ┌───────────┐  fail  ┌──────────┐  │
│  │ MCP call  │──────────▶│ CLI       │───────▶│ Human-in-│  │
│  │ (primary) │           │ fallback  │        │ the-loop │  │
│  └───────────┘           └───────────┘        └──────────┘  │
│                                                              │
│  ┌───────────┐  sandbox  ┌───────────┐  fail  ┌──────────┐  │
│  │ Code Exec │  error    │ Sequential│───────▶│ Log +    │  │
│  │ (batch)   │──────────▶│ tool calls│        │ retry    │  │
│  └───────────┘           └───────────┘        └──────────┘  │
└──────────────────────────────────────────────────────────────┘
```

**MCP Gateway** (the most impactful single optimization):

A middleware layer that solves MCP's reliability problem [[StackOne](https://www.stackone.com/blog/mcp-token-optimization/)]:

- **Connection pooling**: TCP timeout failures drop from **28% → ~1%**
- **Schema filtering**: Token cost drops **90%** (3 relevant tools instead of 43)
- **Auth centralization**: Single OAuth flow for all downstream servers
- **Implementations**: Lasso MCP Gateway, WunderGraph (GraphQL-based), Portkey

**LangGraph checkpoint recovery**: If an agent fails at node N, resume from checkpoint N-1 with preserved state. No restart from scratch.

**oh-my-\*'s targeted repair**: The VERIFY stage checks combined output. If verification fails, only the broken parts get targeted repairs in the FIX stage — no full re-execution. Auto-resume daemon handles rate limit interruptions [[GitHub](https://github.com/anthropics/oh-my-claudecode)].

---

## GraphRAG: How Agents Find the Right Tool

This is where the orchestration story gets interesting. Everything above — routing, state, fallbacks — assumes the agent **already knows which tools exist**. But with 50+ tools across multiple MCP servers, **tool discovery itself becomes the bottleneck**.

### The Schema Bloat Problem, Quantified

```
MySQL MCP Server:      106 tools = 207KB = ~54,600 tokens
GitHub MCP Server:      43 tools = ~44,000 tokens
7+ MCP servers:        67,000+ tokens consumed before any work

All loaded on every request, even if you need 2 tools.
```

This is the "hidden token tax" [[Layered.dev](https://layered.dev/mcp-tool-schema-bloat-the-hidden-token-tax-and-how-to-fix-it/)]. Current solutions (lazy loading, dynamic toolsets, gateways) help, but they're **syntactic** optimizations — they reduce what's loaded, not how intelligently tools are selected.

The **semantic** solution is retrieval: don't load all tools — **search for the right ones**.

### Vector RAG vs GraphRAG for Tool Selection

**Vector RAG** embeds tool descriptions and retrieves the top-k most similar. Simple, fast, but fundamentally limited:

| Metric | Vector RAG | GraphRAG |
|--------|-----------|----------|
| **Overall accuracy** | 56.2% | ~90% (FalkorDB) |
| **Multi-entity queries (5+ entities)** | Degrades to **0%** | Stable |
| **Multi-hop reasoning** | Poor | Strong |
| **Hallucination rate** | Baseline | Up to **90% reduction** |

**Source**: Diffbot KG-LM Benchmark [[FalkorDB](https://www.falkordb.com/blog/graphrag-accuracy-diffbot-falkordb/)]

**Why the gap?** Vector search treats tools as independent documents. But tools have **relationships** — dependencies, composition patterns, input/output type compatibility — that vectors can't capture.

```
Vector RAG sees:                    GraphRAG sees:
┌──────────┐                        ┌──────────┐
│ Tool A   │   (independent)        │ Tool A   │──output──▶(:DataType)
└──────────┘                        └──────────┘              │
┌──────────┐                        ┌──────────┐              │
│ Tool B   │   (independent)        │ Tool B   │◀──input───────┘
└──────────┘                        └──────────┘
┌──────────┐                        ┌──────────┐
│ Tool C   │   (independent)        │ Tool C   │──alt_to──▶(Tool B)
└──────────┘                        └──────────┘

Vector finds Tool A.                Graph finds the chain: A → B
                                    and the alternative: A → C
```

### Building a Tool Knowledge Graph

The schema design for tool orchestration:

```
┌──────────────────────────────────────────────────────────────┐
│              Tool Knowledge Graph Schema                     │
│                                                              │
│  (:Tool {name, description, version, cost, latency})         │
│     │                                                        │
│     ├──[:REQUIRES_INPUT]──▶ (:DataType {schema, format})     │
│     ├──[:PRODUCES_OUTPUT]──▶ (:DataType)                     │
│     ├──[:DEPENDS_ON]──▶ (:Tool)                              │
│     ├──[:COMPOSES_WITH]──▶ (:Tool {pattern})                 │
│     ├──[:ALTERNATIVE_TO]──▶ (:Tool {tradeoff})               │
│     ├──[:BELONGS_TO]──▶ (:ToolCategory)                      │
│     ├──[:HAS_PRECONDITION]──▶ (:Condition {expression})      │
│     └──[:HAS_EFFECT]──▶ (:Effect {description})              │
│                                                              │
│  What graph structure captures that flat schemas miss:        │
│  ● Tool A's output type matches Tool B's input (composable)  │
│  ● Tool C requires Tool D to run first (dependency)          │
│  ● Tools E, F, G solve the same problem differently (alts)   │
│  ● Tool H has a precondition only Tool I can satisfy         │
└──────────────────────────────────────────────────────────────┘
```

**Research validation:**

- **SciToolAgent** (Nature Computational Science, 2025): Knowledge graph of scientific tools enables informed selection and combination across biology, chemistry, and materials science [[Nature](https://www.nature.com/articles/s43588-025-00849-y)]
- **Agent-as-a-Graph** (arxiv:2511.18194): Represents tools AND agents as graph nodes. **+14.9% Recall@5** and **+14.6% nDCG@5** over prior retrievers on LiveMCPBenchmark [[arxiv](https://arxiv.org/abs/2511.18194)]
- **GAP: Graph-Based Agent Planning** (NeurIPS 2025 Workshop): Models inter-task dependencies as a DAG. Enables parallel execution of independent tools while respecting sequential dependencies [[arxiv](https://arxiv.org/abs/2510.25320)]

### Graph-Guided Tool Selection in Action

```
User query: "Find the customer's order history and send them a summary email"

Step 1: Query the Tool Knowledge Graph
        ─────────────────────────────
        MATCH (t:Tool)-[:PRODUCES_OUTPUT]->(d:DataType)
              <-[:REQUIRES_INPUT]-(t2:Tool)
        WHERE t.description CONTAINS 'customer'
        RETURN path

Step 2: Graph returns the chain
        ─────────────────────────
        search_customers ──output: customer_id──▶ get_order_history
                                                   ──output: order_list──▶ send_email

Step 3: Present only 3 tools (not 106)
        ──────────────────────────────
        Token cost: ~3,000 (vs ~54,600 for full schema load)
        Accuracy: 43.13% vs 13.62% baseline
```

**Source**: RAG-MCP [[arxiv:2505.03275](https://arxiv.org/abs/2505.03275)] — formally addresses MCP schema bloat via semantic retrieval. **>50% token reduction**, **3x accuracy improvement**.

### The Hybrid Retrieval Architecture

Production systems in 2026 use **three-pillar retrieval** — vectors for breadth, graphs for depth, keywords for precision:

```
┌──────────────────────────────────────────────────────────────┐
│                Hybrid Tool Retrieval                         │
│                                                              │
│  User Query: "Deploy the staging build and notify the team"  │
│       │                                                      │
│       ├──▶ Vector Index ──▶ "deploy", "staging" → kubectl,   │
│       │    (semantic)       docker, terraform (by similarity) │
│       │                                                      │
│       ├──▶ Graph Index ──▶ kubectl ──[:DEPENDS_ON]──▶        │
│       │    (relational)    docker_build                      │
│       │                   kubectl ──[:COMPOSES_WITH]──▶       │
│       │                    slack_notify                       │
│       │                                                      │
│       └──▶ BM25 Index ──▶ exact match: "staging"            │
│            (keyword)       → staging_deploy_script           │
│                                                              │
│  Reciprocal Rank Fusion ──▶ Final ranked tool list           │
│                                                              │
│  Result: [docker_build, kubectl_deploy, slack_notify]        │
│  Token cost: ~3,000 (vs loading all schemas)                 │
└──────────────────────────────────────────────────────────────┘
```

**Performance**: Hybrid approaches show **10-30% improvement** in retrieval accuracy over single-strategy systems [[Calmops](https://calmops.com/ai/hybrid-search-rag-complete-guide-2026/)].

---

## Practical Graph DB Implementations

### Neo4j: The Default Knowledge Layer

Neo4j positions itself as the knowledge layer for agentic systems, with the most mature integration ecosystem:

**MCP Servers** (6 official/Labs servers) [[Neo4j MCP](https://neo4j.com/developer/genai-ecosystem/model-context-protocol-mcp/)]:

| Server | Purpose |
|--------|---------|
| **Official Neo4j MCP** | Schema retrieval, Cypher execution (read/write), GDS algorithms |
| **MCP-Neo4j-Memory** | Knowledge graph persistence — entity/relationship management |
| **MCP-Neo4j-Data-Modeling** | Schema design, constraint generation, 7 templates |
| **MCP-Neo4j-GDS** | Graph algorithms — PageRank, Louvain, Leiden, Dijkstra |

**Framework integration**: LangChain (`langchain-neo4j`), LlamaIndex (PropertyGraphIndex), CrewAI, Semantic Kernel, Google ADK [[Neo4j Blog](https://neo4j.com/blog/developer/neo4j-graphrag-workflow-langchain-langgraph/)].

**Text2Cypher**: LLM generates Cypher from natural language. **92% accuracy** on Spider-Graph benchmarks, 150ms end-to-end latency. Fine-tuned models available on HuggingFace [[Neo4j Medium](https://medium.com/neo4j/text2cypher-guide-cc161518a509)].

**GraphRAG Retrievers as MCP Server**: Neo4j published a pattern to expose vector, text2cypher, and hybrid retrievers directly as MCP tools [[Neo4j Blog](https://neo4j.com/blog/developer/neo4j-graphrag-retrievers-as-mcp-server/)].

### FalkorDB: Performance-First

Purpose-built for GraphRAG speed [[FalkorDB](https://www.falkordb.com/)]:

- **496x faster** than Neo4j for point lookups (Redis hash tables, O(1) in-memory)
- **6x better** memory efficiency
- **2.9x faster** 2-hop traversal
- Batch ingestion at 5,000: **22,784/s**
- Own MCP server for graph-based AI integration [[FalkorDB MCP](https://www.falkordb.com/blog/mcp-integration-falkordb-graphrag/)]
- GraphRAG SDK: **90%+ accuracy** for schema-heavy enterprise queries

### Microsoft GraphRAG

The open-source reference implementation [[GitHub](https://github.com/microsoft/graphrag)]:

- **6-phase indexing**: Chunk → Extract entities → Leiden community detection → Summarize
- **Three search modes**: Local (entity neighborhood), Global (community summaries), DRIFT (hybrid)
- **LazyGraphRAG** (June 2025): **700x lower query cost** than full GraphRAG with comparable quality. Indexing cost identical to vector RAG (0.1% of full GraphRAG) [[Microsoft Research](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/)]

### Zep / Graphiti: Temporal Knowledge Graphs

For agents that need to **remember** across sessions [[GitHub](https://github.com/getzep/graphiti)]:

- Three-tier subgraph: Episode (raw events) → Semantic Entity → Community
- **Bitemporal** modeling: `event_time` + `ingestion_time` on every node/edge
- **94.8% accuracy** on Deep Memory Retrieval (vs MemGPT's 93.4%) [[arxiv:2501.13956](https://arxiv.org/abs/2501.13956)]
- Non-lossy updates: full history of fact validity periods

### Amazon Bedrock + Neptune

Fully managed GraphRAG (GA March 2025) [[AWS Blog](https://aws.amazon.com/blogs/machine-learning/announcing-general-availability-of-amazon-bedrock-knowledge-bases-graphrag-with-amazon-neptune-analytics/)]:

- Automatic entity extraction during document ingestion
- Two-step query: vector search → graph traversal for multi-hop reasoning
- No graph DB management required

---

## The Production Architecture

Putting it all together — the full stack for a production agent system with intelligent tool discovery:

```
┌────────────────────────────────────────────────────────────────┐
│                 Production Agent System                        │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                   Agent (LLM)                            │  │
│  │  Planner → Generator → Evaluator (separate contexts)     │  │
│  └──────────────────────┬───────────────────────────────────┘  │
│                         │                                      │
│  ┌──────────────────────▼───────────────────────────────────┐  │
│  │              Tool Knowledge Graph                        │  │
│  │         (Neo4j / FalkorDB / Neptune)                     │  │
│  │                                                          │  │
│  │  (:Tool)──[:REQUIRES_INPUT]──▶(:DataType)                │  │
│  │  (:Tool)──[:DEPENDS_ON]──▶(:Tool)                        │  │
│  │  (:Tool)──[:COMPOSES_WITH]──▶(:Tool)                     │  │
│  │  (:Tool)──[:ALTERNATIVE_TO]──▶(:Tool)                    │  │
│  │                                                          │  │
│  │  Hybrid Retrieval: Vector + Graph + BM25                 │  │
│  └──────────────────────┬───────────────────────────────────┘  │
│                         │ Graph-Guided Discovery               │
│  ┌──────────────────────▼───────────────────────────────────┐  │
│  │                 Hybrid Router                            │  │
│  │  Rule-based fast path → Semantic → LLM fallback          │  │
│  └───┬──────────────────┬──────────────────┬────────────────┘  │
│      │                  │                  │                    │
│      ▼                  ▼                  ▼                    │
│  ┌────────┐      ┌───────────┐      ┌───────────┐             │
│  │  CLI   │      │   Code    │      │    MCP    │             │
│  │ Inner  │      │  Middle   │      │   Outer   │             │
│  │  Loop  │      │   Loop    │      │    Loop   │             │
│  └────────┘      └───────────┘      └───────────┘             │
│      │                  │                  │                    │
│      └──────────────────┼──────────────────┘                   │
│                         ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │          State Layer                                     │  │
│  │  Filesystem-as-State + LangGraph Checkpoints             │  │
│  │  (survives truncation, crash, delegation)                │  │
│  └──────────────────────────────────────────────────────────┘  │
│                         │                                      │
│  ┌──────────────────────▼───────────────────────────────────┐  │
│  │          MCP Gateway                                     │  │
│  │  Schema filtering (90%) │ Connection pooling (28%→1%)    │  │
│  │  Auth centralization    │ PII detection                  │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

### How It Flows

1. **Query arrives**: "Find the customer's order history and send them a summary email"
2. **Graph-guided discovery**: Tool Knowledge Graph finds the chain: `search_customers → get_order_history → send_email` (3 tools, not 106)
3. **Hybrid router**: `search_customers` and `get_order_history` are database tools → MCP outer loop. `send_email` is Slack → MCP outer loop. All three need OAuth → no CLI fallback.
4. **But wait — it's a 3-step chain**: Router recognizes multi-step pattern → Middle Loop (Code Execution). Agent writes a Python script that calls all three via the MCP SDK in one sandbox pass.
5. **State persisted**: Results written to filesystem. If the sandbox crashes at step 2, LangGraph checkpoint enables resume from step 1's result.
6. **MCP Gateway**: All three MCP calls routed through gateway. Connection pooling prevents TCP timeouts. Schema filtering means only 3 tool schemas loaded, not 106.

### Where RTK and BKit Fit

```
┌────────────────────────────────────────────────────────────────┐
│  Methodology Layer ──── BKit (PDCA workflow, quality gates)    │
│       │                 37 skills, 32 agents, gap-detector     │
│       ▼                                                        │
│  Harness Layer ──────── Three-Loop + GraphRAG + Routing        │
│       │                                                        │
│       ▼                                                        │
│  Tool Interface Layer ─ CLI ─── MCP ─── Code Exec ─── Browser │
│       │                  │                                     │
│       ▼                  ▼                                     │
│  Infrastructure ──────── RTK (CLI output compression, 60-90%) │
│                         ICM (persistent memory, KG)            │
│                         Grit (parallel agent git locking)      │
└────────────────────────────────────────────────────────────────┘

BKit pushes structure DOWN: methodology → harness → tool selection
RTK pushes efficiency UP: infrastructure → tool interface → harness
```

BKit and RTK represent two complementary directions of harness optimization:

- **BKit** operates **top-down** — it imposes methodology (PDCA) onto the agent, controlling *which* tools are available at each phase and *how* their outputs are evaluated. The gap-detector ensures quality; phase-aware skill loading ensures focus.
- **RTK** operates **bottom-up** — it optimizes the *infrastructure* that tools run on, making every CLI call cheaper without changing what the agent does. A 70% token reduction across a session means more room for reasoning.

Both validate the core thesis of this series: **tool interface orchestration is a multi-layer engineering discipline**, not a single protocol choice. The best systems optimize at every layer simultaneously.

---

## Eight Anti-Patterns to Avoid

Lessons extracted from production harness failures:

### 1. The 50-Tool Firehose

**Problem**: Giving the model 50+ tool schemas and hoping it picks correctly.

**Fix**: Per-phase tool scoping. Planning step gets read-only tools. Execution gets write tools. QA gets test tools. Vercel cut tools by 80% and performance improved.

### 2. Self-Evaluation

**Problem**: Letting the same agent evaluate its own output → 90% false-positive "task complete" signals.

> "Agents confidently praise their own work — even when quality is obviously mediocre." — Anthropic

**Fix**: Architecturally separate evaluator with its own context. The Planner–Generator–Evaluator pattern from [Harness Engineering Part 1](/posts/harness-engineering-part1-why-harness/).

### 3. Context Window as State Store

**Problem**: At 80%+ context usage, models rush to finish (context anxiety), skip verification, declare premature completion.

**Fix**: Externalize state to the filesystem. The context window is disposable; files are truth.

### 4. MCP Schema Bloat

**Problem**: Loading all schemas on connect. GitHub MCP = 44,000 tokens before any work.

**Fix**: Lazy loading (95%), Code Mode (99.9%), MCP Gateway (90%), or GraphRAG-based discovery (95%+).

### 5. Over-Summarizing Feedback

**Problem**: Compressing execution traces to scalar scores before passing to the evaluator.

**Data**: Full traces achieve **50.0 accuracy** vs **34.6 for scalar scores** (Meta-Harness ablation [[arxiv:2603.28052](https://arxiv.org/abs/2603.28052)]).

**Fix**: Log everything. Summarize on demand, not by default.

### 6. Premature Multi-Agent

**Problem**: Adding agents before measuring the specific failure you're solving.

> "Multi-agent systems have coordination overhead that often exceeds their benefits." — Cognition Labs

**Data**: Forrester 2025: 75% of firms building custom agentic architectures will fail due to infrastructure complexity.

**Fix**: Start single-agent. Measure. Escalate only when measured.

### 7. Ignoring Amdahl's Law

**Problem**: Assuming 2x workers = 2x speed. The parallelizable portion of a codebase has limits.

**Fix**: oh-my-\* acknowledges this — for a bug fix or two-file refactor, pipeline overhead exceeds savings. Multi-agent pays off only at **feature-module scale** (10+ files, cross-cutting changes).

### 8. Data Overload (The Firehose Effect)

**Problem**: Tools returning database dumps, binary data, massive JSON payloads.

**Fix**: Tools should return filtered, relevant data. If a tool returns more than the model can process, the tool's interface is broken — fix the tool, not the model. For CLI specifically, **RTK** intercepts command output and applies 12 filtering strategies (stats extraction, error-only, pattern grouping, deduplication) to compress 60-90% of noise before it reaches the agent — turning `cargo test` from 4,823 tokens to 11 [[github.com/rtk-ai/rtk](https://github.com/rtk-ai/rtk)].

---

## Scenario-Based Recommendations

| Scenario | Tool count | Recommended stack | Tool discovery |
|----------|-----------|-------------------|----------------|
| **Single-file bug fix** | 3-5 | CLI only, static routing | None needed |
| **Feature development (10+ files)** | 10-15 | CLI + Code Exec, LLM routing | Lazy loading |
| **External service integration** | 15-25 | Three-Loop, Hybrid routing | MCP Gateway |
| **Enterprise platform (50+ tools)** | 50+ | Three-Loop + GraphRAG | Tool Knowledge Graph |
| **Multi-agent team** | Variable | Per-agent tool scoping + Gateway | Graph per agent role |

---

## The Minimal Viable Harness Roadmap

**Add complexity only when measured failure demands it.**

```
Step 1: Single LLM + CLI
        ─────────────────
        Measure: What fails? What's the completion rate?
        If ≥ 90% success → Stop here.

Step 2: + MCP (for external services)
        ─────────────────────────────
        Measure: Token cost? How much is schema overhead?
        If schema overhead < 10% of total → Stop here.

Step 3: + Code Execution (for multi-step workflows)
        ─────────────────────────────────────────────
        Measure: How many round-trips? What's the latency?
        If round-trips < 3 on average → Stop here.

Step 4: + MCP Gateway (for reliability and token cost)
        ─────────────────────────────────────────────
        Measure: MCP failure rate? Schema bloat impact?
        If failure rate < 5% → Stop here.

Step 5: + GraphRAG tool discovery (for 50+ tools)
        ─────────────────────────────────────────
        Measure: Tool selection accuracy? Discovery latency?
        Start with LazyGraphRAG (cheapest), graduate to full.

        "측정 없이 추가하지 마라."
```

### Starting Stack Recommendation

For most teams starting today:

- **LazyGraphRAG** for knowledge retrieval — indexing cost identical to vector RAG, 700x cheaper queries [[Microsoft Research](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/)]
- **Neo4j MCP Server** for tool discovery — query the graph via the same MCP interface [[Neo4j](https://github.com/neo4j/mcp)]
- **Embedding-based tool selection** for MCP schema bloat — RAG-MCP pattern [[arxiv:2505.03275](https://arxiv.org/abs/2505.03275)]
- Graduate to **full Tool Knowledge Graph** with dependency-aware execution (GAP pattern [[arxiv:2510.25320](https://arxiv.org/abs/2510.25320)]) when you exceed 50 tools.

---

## Connections to the Harness Engineering Series

This post fills gaps in the [Harness Engineering series](/posts/harness-engineering-part1-why-harness/):

| Harness Engineering | This Post |
|---|---|
| **Part 1**: Three failure modes (context anxiety, self-eval bias, coherence drift) | Adds: the **50-tool firehose** as a fourth failure mode |
| **Part 2**: "Richer feedback beats cleverer structure" (Meta-Harness) | Extends: **richer tool discovery** (GraphRAG) beats loading more schemas |
| **Part 3**: Framework landscape (LangGraph, CrewAI, gstack) | Extends: **tool interface layer** underneath the framework layer |
| **Part 3**: "ACI — tool docs deserve as much care as UX docs" | Extends: **Tool Knowledge Graph** as the structured version of ACI |

The Planner–Generator–Evaluator architecture from Part 1 benefits directly from GraphRAG:

- **Planner** uses graph-guided discovery to find the minimal tool set
- **Generator** receives only relevant tool schemas (95%+ token savings)
- **Evaluator** can query the graph for expected tool behaviors and verify results

---

## References

### Tool Orchestration and Harness Design

1. **Inside Claude Code Architecture** — [penligent.ai](https://www.penligent.ai/hackinglabs/inside-claude-code-the-architecture-behind-tools-memory-hooks-and-mcp/)
2. **Claude Code Agent Harness Architecture** — [wavespeed.ai](https://wavespeed.ai/blog/posts/claude-code-agent-harness-architecture/)
3. **Claude Code Best Practices** — [anthropic.com/engineering](https://www.anthropic.com/engineering/claude-code-best-practices)
4. **Claude Code Sandboxing** — [anthropic.com/engineering](https://www.anthropic.com/engineering/claude-code-sandboxing)
5. **Anatomy of an Agent Harness** — [blog.langchain.com](https://blog.langchain.com/the-anatomy-of-an-agent-harness/)
6. **LangChain Deep Agents SDK** — [langchain.com/deep-agents](https://www.langchain.com/deep-agents)
7. **Microsoft Agent Framework v1.0** — [devblogs.microsoft.com](https://devblogs.microsoft.com/agent-framework/microsoft-agent-framework-version-1-0/)
8. **Natural-Language Agent Harnesses** — [arxiv:2603.25723](https://arxiv.org/html/2603.25723v1)
9. **Meta-Harness Research** — [arxiv:2603.28052](https://arxiv.org/abs/2603.28052)

### Tool Routing and Optimization

10. **AI Agent Routing Tutorial** — [patronus.ai](https://www.patronus.ai/ai-agent-development/ai-agent-routing)
11. **Intelligent LLM Routing in Enterprise AI** — [requesty.ai](https://www.requesty.ai/blog/intelligent-llm-routing-in-enterprise-ai-uptime-cost-efficiency-and-model)
12. **Toward Super Agent System with Hybrid AI Routers** — [arxiv:2504.10519](https://arxiv.org/html/2504.10519v1)
13. **MCP Token Optimization: 4 Approaches** — [stackone.com](https://www.stackone.com/blog/mcp-token-optimization/)
14. **MCP Tool Schema Bloat: The Hidden Token Tax** — [layered.dev](https://layered.dev/mcp-tool-schema-bloat-the-hidden-token-tax-and-how-to-fix-it/)
15. **MCP vs CLI Benchmark** — [scalekit.com](https://www.scalekit.com/blog/mcp-vs-cli-use)

### GraphRAG and Knowledge Graphs

16. **Microsoft GraphRAG** — [github.com/microsoft/graphrag](https://github.com/microsoft/graphrag)
17. **LazyGraphRAG: New Standard for Quality and Cost** — [microsoft.com/research](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/)
18. **GraphRAG: Dynamic Community Selection** — [microsoft.com/research](https://www.microsoft.com/en-us/research/blog/graphrag-improving-global-search-via-dynamic-community-selection/)
19. **RAG-MCP: Mitigating Prompt Bloat** — [arxiv:2505.03275](https://arxiv.org/abs/2505.03275)
20. **Agent-as-a-Graph** — [arxiv:2511.18194](https://arxiv.org/abs/2511.18194)
21. **SciToolAgent** (Nature Computational Science) — [nature.com](https://www.nature.com/articles/s43588-025-00849-y)
22. **GAP: Graph-Based Agent Planning** (NeurIPS 2025) — [arxiv:2510.25320](https://arxiv.org/abs/2510.25320)
23. **Agentic RAG with Knowledge Graphs** — [arxiv:2507.16507](https://arxiv.org/abs/2507.16507)

### Graph Database Implementations

24. **Neo4j MCP Servers** — [neo4j.com](https://neo4j.com/developer/genai-ecosystem/model-context-protocol-mcp/) | [github.com/neo4j/mcp](https://github.com/neo4j/mcp) | [github.com/neo4j-contrib/mcp-neo4j](https://github.com/neo4j-contrib/mcp-neo4j)
25. **Neo4j GraphRAG Workflow with LangGraph** — [neo4j.com/blog](https://neo4j.com/blog/developer/neo4j-graphrag-workflow-langchain-langgraph/)
26. **Neo4j GraphRAG Retrievers as MCP Server** — [neo4j.com/blog](https://neo4j.com/blog/developer/neo4j-graphrag-retrievers-as-mcp-server/)
27. **Text2Cypher Production Guide** — [medium.com/neo4j](https://medium.com/neo4j/text2cypher-guide-cc161518a509)
28. **FalkorDB GraphRAG Accuracy Benchmark** — [falkordb.com](https://www.falkordb.com/blog/graphrag-accuracy-diffbot-falkordb/)
29. **FalkorDB MCP Integration** — [falkordb.com](https://www.falkordb.com/blog/mcp-integration-falkordb-graphrag/)
30. **Amazon Bedrock GraphRAG with Neptune** — [aws.amazon.com](https://aws.amazon.com/blogs/machine-learning/announcing-general-availability-of-amazon-bedrock-knowledge-bases-graphrag-with-amazon-neptune-analytics/)
31. **Zep / Graphiti: Temporal Knowledge Graphs** — [github.com/getzep/graphiti](https://github.com/getzep/graphiti) | [arxiv:2501.13956](https://arxiv.org/abs/2501.13956)
32. **GraphRAG MCP Server (community)** — [github.com/rileylemm/graphrag_mcp](https://github.com/rileylemm/graphrag_mcp)

### Agent Frameworks and Patterns

33. **LangGraph Documentation** — [langchain.com/langgraph](https://www.langchain.com/langgraph)
34. **LangGraph MCP Integration** — [latenode.com](https://latenode.com/blog/ai-frameworks-technical-infrastructure/langgraph-multi-agent-orchestration/langgraph-mcp-integration-complete-model-context-protocol-setup-guide-working-examples-2025)
35. **CrewAI Role-Based Orchestration** — [digitalocean.com](https://www.digitalocean.com/community/tutorials/crewai-crash-course-role-based-agent-orchestration)
36. **AIO Sandbox** — [github.com/agent-infra/sandbox](https://github.com/agent-infra/sandbox)
37. **Tool RAG: Next Breakthrough in Scalable AI Agents** — [Red Hat](https://next.redhat.com/2025/11/26/tool-rag-the-next-breakthrough-in-scalable-ai-agents/)
38. **Cognee: Knowledge Engine for AI Agents** — [github.com/topoteretes/cognee](https://github.com/topoteretes/cognee)
39. **LlamaIndex GraphRAG v2** — [developers.llamaindex.ai](https://developers.llamaindex.ai/python/examples/cookbooks/graphrag_v2/)
40. **Graphs Meet AI Agents: Taxonomy and Opportunities** — [arxiv:2506.18019](https://arxiv.org/html/2506.18019v1)

### Production Harness Implementations

41. **BKit for Claude Code** — [github.com/popup-studio-ai/bkit-claude-code](https://github.com/popup-studio-ai/bkit-claude-code)
42. **BKit for Gemini CLI** — [github.com/popup-studio-ai/bkit-gemini](https://github.com/popup-studio-ai/bkit-gemini)
43. **BKit for Codex CLI** — [github.com/popup-studio-ai/bkit-codex](https://github.com/popup-studio-ai/bkit-codex)
44. **RTK: CLI Output Compression** (18.6k stars) — [github.com/rtk-ai/rtk](https://github.com/rtk-ai/rtk)
45. **ICM: Persistent Memory for Agents** — [github.com/rtk-ai/icm](https://github.com/rtk-ai/icm)
46. **Grit: Git for Parallel Agents** — [github.com/rtk-ai/grit](https://github.com/rtk-ai/grit)
