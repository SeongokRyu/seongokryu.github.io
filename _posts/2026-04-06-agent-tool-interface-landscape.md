---
title: "Agent Tool Interfaces Part 1: How Agents Connect to Tools — The Complete Interface Landscape"
date: 2026-04-06 15:30:00 +0900
categories: [AI & ML, AI Agent]
tags: [mcp, cli, code-execution, model-context-protocol, ai-agent, tool-use, computer-use, a2a, wasm, agent-architecture]
---

*Every architectural decision in an agent system starts with one question: how does it touch the outside world?*

**Agent Tool Interfaces: From Landscape to Orchestration**

*This is Part 1 of a 2-part series on Agent Tool Interfaces.*

- **Part 1** (this post): How Agents Connect to Tools — The Complete Interface Landscape
- **Part 2:** [Orchestrating Tool Interfaces — From Harness Design to GraphRAG](/posts/orchestrating-tool-interfaces-graphrag/)

---

> **TL;DR**: MCP and CLI get all the attention, but agents actually connect to the world through **at least six distinct interfaces** — each with different cost, reliability, and security profiles. The biggest shift in 2026 isn't MCP vs CLI. It's the rise of **Code-as-Tool-Use**, which bypasses MCP's schema bloat entirely by letting agents write and execute code against typed SDKs. This post maps the complete landscape, benchmarks the Big Three, and provides a decision framework for choosing the right interface for each system your agent touches.

---

## The Question Nobody Asked Until It Broke

Your agent needs to list open pull requests. Two paths:

**Path A — CLI:**

```bash
gh pr list --repo owner/repo --state open --json title,number
```

**Path B — MCP:**

```
Connect to GitHub MCP server → OAuth handshake → Load 43 tool schemas
→ Call list_pull_requests({owner: "owner", repo: "repo", state: "open"})
→ Parse structured JSON response
```

Both return the same data. But **Path A costs 1,365 tokens**. **Path B costs 44,026 tokens** — 32x more — before a single result comes back. Path A succeeds 100% of the time. Path B fails 28% of the time due to TCP timeouts.

So why does MCP exist? And why is it winning adoption anyway?

**Because tool connection is an architectural decision, not a feature toggle.** And in 2026, the choice is no longer binary. There are at least six distinct approaches competing for your agent's attention — and the most disruptive one isn't MCP or CLI at all.

---

## The Tool Interface Landscape (2026)

Before diving into any single interface, let's see the full map.

```
┌──────────────────────────────────────────────────────────────────┐
│               Tool Interface Landscape (2026)                    │
│                                                                  │
│  Structured ◄──────────────────────────────────► Unstructured    │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────────┐ │
│  │  Native     │  │    MCP      │  │         CLI              │ │
│  │  Function   │  │  Protocol   │  │    Shell / Subprocess    │ │
│  │  Calling    │  │  (JSON-RPC) │  │    (stdin/stdout)        │ │
│  └──────┬──────┘  └──────┬──────┘  └────────────┬─────────────┘ │
│         │                │                       │               │
│  ┌──────┴──────┐  ┌──────┴──────┐  ┌────────────┴─────────────┐ │
│  │   Code      │  │  Browser /  │  │     API-Direct           │ │
│  │  Execution  │  │  Computer   │  │   (HTTP generation)      │ │
│  │  (sandbox)  │  │    Use      │  │                          │ │
│  └─────────────┘  └─────────────┘  └──────────────────────────┘ │
│                                                                  │
│  ──── Supporting Infrastructure ────                             │
│  Message Queues │ Framework Abstractions │ WebAssembly Sandbox   │
└──────────────────────────────────────────────────────────────────┘
```

### Maturity Matrix

| Approach | Maturity | 2026 Trend | Best For |
|----------|----------|------------|----------|
| **CLI** | Mature | CLI renaissance (SKILL.md) | Local dev tools, git, testing |
| **MCP** | Maturing | 97M+ downloads/month | OAuth services, audit, discovery |
| **Code Execution** | Mature + accelerating | **Most important shift** | Multi-step workflows, data transforms |
| **Native Function Calling** | Mature | Converging across providers | Tight app control, single-turn |
| **Browser/Computer Use** | Early-mid | Web approaching production | No-API legacy systems |
| **API-Direct** | Mature (mediated) | Consolidating via platforms | OpenAPI-spec'd services |
| **Message Queue** | Infra mature | Agent standardization needed | Async enterprise pipelines |
| **WebAssembly** | Emerging | MS/NVIDIA momentum | Maximum sandbox security |

The rest of this post focuses on **the Big Three** — the interfaces that handle 90%+ of production tool calls — then surveys the remaining five.

---

## The Big Three

### 1. CLI: The Unix Philosophy, Reborn

The CLI tool interface is dead simple. The agent spawns a subprocess, passes arguments, and reads the output.

```
Agent                    Shell                    System
  │                        │                        │
  │  "git diff HEAD~1"     │                        │
  │───────────────────────▶│                        │
  │                        │  fork + exec           │
  │                        │───────────────────────▶│
  │                        │                        │
  │                        │◀───────────────────────│
  │   stdout (diff text)   │   exit code 0          │
  │◀───────────────────────│                        │
  │                        │                        │
  │  (LLM interprets diff) │                        │
```

**No schema. No handshake. No OAuth.** Just a command, a result, and a model that knows what to do with both.

#### Why LLMs Are Unreasonably Good at CLI

LLMs didn't learn tool use from protocol specifications. They learned it from **billions of real terminal sessions** — Stack Overflow answers, GitHub Actions logs, tutorial transcripts, man pages.

> When an LLM generates `git log --oneline -10 | grep "fix"`, it's not following a schema. **It's pattern-matching against millions of similar commands it saw during training.**

- **Self-correction**: Agent runs `--help` when uncertain, reads error messages, retries with different flags
- **Pipe composition**: Models naturally chain `curl | jq | grep` without being taught
- **Novel combinations**: LLMs improvise pipe chains they've never seen, because they understand the *grammar* of shell composition

MCP has **zero training data**. Every MCP tool call is a cold-start inference from schema descriptions alone.

#### Strengths and Limitations

| Strength | Detail |
|----------|--------|
| **Token efficiency** | ~1,365 tokens for a GitHub PR list vs ~44,026 for MCP |
| **100% reliability** | 25/25 in benchmarks. No TCP timeouts. |
| **Unix composability** | Pipes, redirects, subshells — small tools into complex workflows |
| **Self-documenting** | `--help`, `man`, error messages are natural language LLMs already understand |
| **Ubiquitous** | git, docker, kubectl, curl, jq, python — every tool has a CLI |

| Limitation | Detail |
|------------|--------|
| **Unstructured output** | Plain text requires LLM interpretation; no guaranteed schema |
| **Security surface** | Broad shell permissions; prompt injection → malicious commands |
| **No discovery** | No standard way to enumerate tools or parameters at runtime |
| **Stateless** | Environment resets between subprocess calls |

#### Who Uses CLI as Primary Interface

- **Claude Code** — 8 built-in tools; Bash is the universal adapter; ~135K GitHub commits/day (Feb 2026) [[Anthropic Engineering](https://www.anthropic.com/engineering/claude-code-best-practices)]
- **Devin** — Cloud sandbox with Bash + VS Code + Chrome [[Devin Agents 101](https://devin.ai/agents101)]
- **Open Interpreter** — Natural language → Python/JS/shell via `exec()` [[GitHub](https://github.com/openinterpreter/open-interpreter)]
- **Codex CLI** — Adopted SKILL.md from Claude Code [[OpenAI](https://openai.com/index/introducing-codex/)]
- **Aider** — Git-aware CLI coding assistant [[GitHub](https://github.com/Aider-AI/aider)]

#### The CLI Output Problem — and RTK's Fix

CLI's biggest weakness — unstructured output flooding the context window — has spawned its own optimization layer. **RTK** (Rust Token Killer) is a CLI proxy written in Rust that intercepts command output and compresses it before it reaches the agent [[GitHub, 18.6k stars](https://github.com/rtk-ai/rtk)].

| Command | Raw tokens | After RTK | Reduction |
|---------|-----------|-----------|-----------|
| `cargo test` | ~4,823 | ~11 | 99% |
| `git diff HEAD~1` | ~21,500 | ~1,259 | 94% |
| `pytest -v` | ~756 | ~24 | 96% |

A typical 30-minute Claude Code session drops from **~150,000 tokens to ~45,000** (70% savings). RTK processes commands through a six-phase pipeline (parse → route → execute → filter → print → track) with **12 filtering strategies** — stats extraction, error-only filtering, pattern grouping, deduplication, and more.

The integration is invisible: a `PreToolUse` hook rewrites shell commands to `rtk` equivalents. The agent never knows the compression happened. This is **harness-level optimization** — no model changes, no prompt changes, just better infrastructure between the agent and the shell.

RTK is part of a broader agent infrastructure ecosystem:

- **ICM** — Persistent memory for agents via MCP-native knowledge graphs with typed relationships (`depends_on`, `contradicts`, `refines`) [[GitHub](https://github.com/rtk-ai/icm)]
- **Grit** — Git for parallel agents with AST-level locking to prevent merge conflicts across 50+ concurrent agents [[GitHub](https://github.com/rtk-ai/grit)]

---

### 2. MCP: The USB-C of AI

MCP (Model Context Protocol) was created by Anthropic in **November 2024**, inspired by the **Language Server Protocol (LSP)**. It uses **JSON-RPC 2.0** as the wire format.

```
┌─────────────────────────────────────────────────────────────┐
│                        Host                                 │
│                  (Claude Desktop, Cursor, VS Code)          │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                 │
│  │ MCP      │  │ MCP      │  │ MCP      │                 │
│  │ Client 1 │  │ Client 2 │  │ Client 3 │  ← 1:1 mapping  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                 │
└───────┼──────────────┼──────────────┼───────────────────────┘
        │              │              │
        ▼              ▼              ▼
   ┌─────────┐   ┌─────────┐   ┌─────────┐
   │ GitHub  │   │  Slack  │   │Database │   ← MCP Servers
   │ Server  │   │ Server  │   │ Server  │
   │(43 tools)│  │(8 tools)│  │(5 tools)│
   └─────────┘   └─────────┘   └─────────┘
```

**Three primitives** per server:

- **Tools**: Actions the agent can invoke (e.g., `create_issue`, `send_message`)
- **Resources**: Data the agent can read (e.g., file contents, database rows)
- **Prompts**: Reusable prompt templates for common operations

**Two transports**: stdio (local, low latency) and Streamable HTTP (remote, SSE)

#### The 2025-11-25 Spec (1-Year Anniversary)

Major upgrades in the anniversary release [[MCP Blog](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/)]:

- **Tasks Primitive**: Long-running operation tracking (`working` → `completed` → `failed`)
- **Simplified Auth**: OAuth 2.1 with URL-based client registration
- **Extensions Framework**: Composable additions outside the core spec
- **Sampling with Tools**: Servers execute agentic loops using client-provided LLM tokens

#### Strengths and Limitations

| Strength | Detail |
|----------|--------|
| **Standardization** | One protocol for all integrations — no custom adapters |
| **Self-description** | Servers declare capabilities; agents discover tools at runtime |
| **Enterprise security** | OAuth 2.1, per-server scoping, audit logging |
| **Bidirectional** | Server-initiated notifications, progress updates |
| **Ecosystem** | 97M+ monthly SDK downloads, 10K+ public servers |

| Limitation | Detail |
|------------|--------|
| **Schema bloat** | GitHub MCP = 43 tools = ~44,000 tokens before any work |
| **Reliability gap** | 72% success vs CLI's 100%. TCP timeouts. |
| **Security immaturity** | 30+ CVEs early 2026; 82% vulnerable to path traversal |
| **Zero training data** | LLMs never saw MCP patterns during pre-training |

#### Ecosystem Scale

- **97M+** monthly SDK downloads
- **10,000+** public MCP servers; **~2,000** registry entries (407% growth since Sep 2025)
- **Governance**: Donated to **AAIF** (Linux Foundation) Dec 2025 — co-founded by Anthropic, OpenAI, Google, Microsoft, AWS, Block [[Anthropic](https://www.anthropic.com/news/donating-the-model-context-protocol-and-establishing-of-the-agentic-ai-foundation)]
- **Spec**: [modelcontextprotocol.io/specification/2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25)

---

### 3. Code Execution: The Third Way

This is the part most people miss. **The biggest shift in 2026 isn't MCP vs CLI. It's Code-as-Tool-Use.**

Instead of calling tools one at a time (sequential tool calling), the agent **writes code that orchestrates multiple tools in a single execution**. The code runs in a sandboxed environment and returns only the final result.

```
┌────────────────────────────────────────────────────────────┐
│           Sequential Tool Calling vs Code Execution        │
│                                                            │
│   Traditional (N round-trips)      Code-as-Tool-Use        │
│   ──────────────────────────       ──────────────────       │
│   call tool_1(args)                sandbox.run("""          │
│   ← result_1                         r1 = tool_1(args)     │
│   call tool_2(result_1)              r2 = tool_2(r1)       │
│   ← result_2                         r3 = tool_3(r2)       │
│   call tool_3(result_2)              return r3              │
│   ← result_3                       """)                    │
│                                    ← result_3              │
│                                                            │
│   Round-trips: 3                   Round-trips: 1          │
│   Schema overhead: 3x             Schema overhead: 1x      │
│   Token cost: HIGH                Token cost: LOW           │
└────────────────────────────────────────────────────────────┘
```

#### Key Implementations

**Cloudflare Code Mode** [[Blog](https://blog.cloudflare.com/code-mode/)]

The most radical implementation. Instead of 2,500 individual tool schemas, expose just **2 meta-tools**: `search()` and `execute()`. The agent writes TypeScript against a typed SDK, executed in a V8 isolate (Dynamic Worker).

```
Before: 2,500 API endpoints as individual tools
  Total schema cost: ~1,170,000 tokens

After: 2 tools (search + execute)
  Total schema cost: ~1,000 tokens

  Token reduction: 99.9%
```

Dynamic Workers boot in milliseconds, use megabytes of memory — claimed **100x faster and 100x more memory-efficient** than containers [[Blog](https://blog.cloudflare.com/dynamic-workers/)].

**Anthropic Programmatic Tool Calling (PTC)** [[Docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling)]

Claude writes a Python script that orchestrates multiple tool calls in a single sandboxed execution. The script processes intermediate results internally — Claude only sees the final output.

- **37% reduction** in round-trips
- **Up to 37% token savings** on multi-step workflows
- Combined with Anthropic's code execution tool [[Engineering](https://www.anthropic.com/engineering/advanced-tool-use)]

**E2B Sandboxes** [[e2b.dev](https://e2b.dev/)]

Open-source, Firecracker microVM-based sandboxes purpose-built for AI agents:

- Boot in **<200ms**, 3-5MB memory overhead per instance
- **~15 million sessions/month** (March 2025)
- **~50% of Fortune 500** using it
- SDKs for Python and JavaScript

**OpenAI Code Interpreter** [[Docs](https://developers.openai.com/api/docs/guides/tools-code-interpreter)]

Runs Python in sandboxed containers via the Responses API. Handles data analysis, file transforms, chart generation, iterative debugging. Internet access disabled during execution.

#### Why Code Execution Changes Everything

The traditional tool-calling model — LLM decides action, system executes, LLM reads result, repeat — has a fundamental **O(n) round-trip problem**. Each tool call requires a full LLM inference pass to decide the next action.

Code execution collapses this to **O(1)**. The LLM plans the entire workflow upfront, writes it as code, and executes it in a single pass. The sandbox handles the sequential logic that would otherwise burn N inference calls.

This matters most when:

- **Multi-step data transformations** (filter → aggregate → format → send)
- **Large API surfaces** (Cloudflare's 2,500 endpoints, GitHub's 43 tools)
- **Cost-sensitive operations** (code is 99.9% cheaper than N sequential MCP calls)

---

## The Benchmark: Three-Way Comparison

Benchmarks from Scalekit [[Blog](https://www.scalekit.com/blog/mcp-vs-cli-use)], Cloudflare [[Blog](https://blog.cloudflare.com/code-mode/)], and Anthropic [[Docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling)]:

### Token Cost

```
Task: List pull requests for a repository

CLI (gh pr list --json)         ██  1,365 tokens
Code Mode (search + execute)    █  ~600 tokens
MCP (GitHub MCP Server)         ████████████████████████████████████████████  44,026 tokens

                                CLI is 32x cheaper than MCP
                                Code Mode is 73x cheaper than MCP
```

### Monthly Cost at Scale (10,000 operations)

| Interface | Tokens/op | Monthly cost | Reliability |
|-----------|-----------|-------------|-------------|
| **Code Mode** | ~600 | **~$1.50** | ~98% |
| **CLI** | 1,365 | **$3.20** | 100% |
| **MCP** | 44,026 | **$55.20** | 72% |

### Why These Numbers Are Partially Misleading

The benchmarks compare **CLI-accessible operations**. MCP's value isn't replacing CLIs — it's connecting to services that **have no CLI**: Figma, Notion, Salesforce, internal APIs.

And Code Mode's value isn't replacing simple CLI calls — it's replacing **multi-step MCP workflows** where N sequential tool calls become 1 code execution.

**The right question: what's the right interface for each system your agent touches?**

> **Perplexity publicly removed MCP support**, citing token cost and reliability. [[jannikreinhard.com](https://jannikreinhard.com/2026/02/22/why-cli-tools-are-beating-mcp-for-ai-agents/)]

---

## The Remaining Five

These interfaces handle specific niches where the Big Three don't reach.

### 4. Native Function Calling — The Engine Under MCP

This is the **model API's own structured tool-use mechanism** — the layer MCP sits on top of.

- **OpenAI**: Function calling via Responses API; `strict: true` for schema-guaranteed arguments [[Docs](https://platform.openai.com/docs/guides/function-calling)]
- **Anthropic**: `tool_use` blocks in Messages API; server-side tools (`web_search`, `code_execution`) [[Docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview)]
- **Google**: Gemini function calling; multi-tool combination in Gemini 3 [[Docs](https://ai.google.dev/gemini-api/docs/function-calling)]

**Relationship to MCP**: Function calling is **vendor-specific tight control**. MCP is **standardized portability**. OpenAI treats them as complementary [[OpenAI Agents SDK](https://openai.github.io/openai-agents-python/mcp/)].

**Use when**: You need maximum control over a small, fixed set of tools within a single provider's API.

### 5. Browser/Computer Use — The Last Resort

When there's no API, no CLI, no MCP server — the agent uses its **eyes and hands**.

```
Agent                    Screen                   Application
  │                        │                        │
  │  take_screenshot()     │                        │
  │───────────────────────▶│                        │
  │   ◀── image bytes ─────│                        │
  │                        │                        │
  │  (LLM: "I see a       │                        │
  │   login form...")      │                        │
  │                        │                        │
  │  click(x=340, y=220)  │                        │
  │───────────────────────▶│───────────────────────▶│
  │                        │                        │
```

- **Anthropic Computer Use**: Screenshots → mouse/keyboard actions. Beta since Oct 2024. [[Docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool)]
- **OpenAI CUA/Operator**: GPT-4o vision + RL-trained GUI interaction. 87% on WebVoyager, 38% on OSWorld. [[Blog](https://openai.com/index/computer-using-agent/)]
- **Stagehand v3** (Browserbase): `act()`, `extract()`, `observe()`. Talks directly to Chrome DevTools Protocol, 44% faster. [[stagehand.dev](https://www.stagehand.dev/)]
- **Browserbase**: Cloud browser infra. $40M Series B, 50M sessions in 2025. [[browserbase.com](https://www.browserbase.com/)]

**Use when**: The target has a visual interface but no API. Web automation is approaching production; desktop remains experimental.

### 6. API-Direct — LLM-Generated HTTP

The agent generates raw HTTP requests from OpenAPI specs.

- **Composio**: 1,000+ pre-built connectors, managed OAuth, type-safe interfaces [[composio.dev](https://composio.dev/)]
- **GPT Actions**: OpenAPI-spec-defined API calls in Custom GPTs — being replaced by MCP Apps
- **Cloudflare Code Mode**: The typed SDK approach is essentially API-Direct via generated code

**Use when**: An OpenAPI spec exists and you want maximum flexibility without building a tool wrapper. Typically mediated through a framework rather than raw generation.

### 7. Message Queue / Event-Driven

Agents communicate with tools via **Kafka, Redis Streams, or Pulsar**.

- Tool invocations published as events; consumers execute and publish results
- **Inherently async**: perfect for long-running agent tasks
- **Durable**: messages survive crashes; exactly-once semantics
- **Scalable**: millions of events/second
- Patterns documented by Red Hat [[Blog](https://developers.redhat.com/articles/2025/06/16/how-kafka-improves-agentic-ai)]

**Use when**: Enterprise multi-agent systems where durability and async processing matter more than latency. Overkill for single-agent synchronous workflows.

### 8. WebAssembly Sandboxing

AI-generated code compiled to **Wasm modules** and executed in maximally isolated runtimes.

- **Microsoft Wassette**: WebAssembly Components via MCP; modules fetched from OCI registries [[Microsoft](https://opensource.microsoft.com/blog/2025/08/06/introducing-wassette-webassembly-based-tools-for-ai-agents/)]
- **NVIDIA**: Wasm for sandboxing agentic AI workflows [[NVIDIA](https://developer.nvidia.com/blog/sandboxing-agentic-ai-workflows-with-webassembly/)]
- Wasm modules are **inert by default** — zero host access unless explicitly granted
- WebAssembly 3.0: 64-bit memory, garbage collection, exception handling

**Use when**: Maximum sandbox security is non-negotiable. The safest execution environment available, but still early-stage for agent use cases.

---

## The Protocol Ecosystem Map (2026)

These interfaces don't exist in isolation. A layered protocol stack is emerging:

```
┌───────────────────────────────────────────────────────────────┐
│                   Protocol Stack (2026)                       │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Layer 4: Agent Commerce                                │  │
│  │  ACP / UCP — Payment, procurement, transactions         │  │
│  │  Status: Early stage                                    │  │
│  └─────────────────────────────────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Layer 3: Agent-to-Web                                  │  │
│  │  WebMCP — Standardized web interaction (replaces scrape)│  │
│  │  Status: Emerging                                       │  │
│  └─────────────────────────────────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Layer 2: Agent-to-Agent                                │  │
│  │  A2A (Google) — Discovery, delegation, coordination     │  │
│  │  Status: v1.0 (gRPC + JSON-RPC, signed Agent Cards)     │  │
│  └─────────────────────────────────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Layer 1: Agent-to-Tool                                 │  │
│  │  MCP (Anthropic → AAIF) — Tool/data/API access          │  │
│  │  Status: Dominant (97M+ downloads/month)                │  │
│  └─────────────────────────────────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Layer 0: Local Execution                               │  │
│  │  CLI + Code Execution — Direct OS/sandbox access        │  │
│  │  Status: Universal                                      │  │
│  └─────────────────────────────────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Substrate: Execution Environments                      │  │
│  │  E2B (microVM) │ Dynamic Workers (V8) │ Wasm (isolate)  │  │
│  │  Status: Mature (E2B) / Emerging (Wasm)                 │  │
│  └─────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
```

**Key relationships:**

- **MCP + A2A**: MCP connects agents to tools; A2A connects agents to each other. IBM's ACP merged into A2A Aug 2025 [[Google](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/)]
- **AAIF governance**: All protocols under the Linux Foundation [[AAIF](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents)]
- **Function Calling + MCP**: OpenAI treats them as complementary — function calling for tight control, MCP for portability

---

## Security: Three Models Compared

Each interface has a fundamentally different security profile:

```
┌────────────────────────────────────────────────────────────────┐
│                  Security Model Comparison                     │
│                                                                │
│  CLI                  MCP                   Code Execution     │
│  ────                 ────                  ──────────────      │
│  Threat:              Threat:               Threat:             │
│  Prompt injection     Tool poisoning,       Sandbox escape,     │
│  → shell command      path traversal,       arbitrary code      │
│                       token theft                               │
│                                                                │
│  Mitigation:          Mitigation:           Mitigation:         │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐   │
│  │ Seatbelt/     │    │ OAuth 2.1     │    │ Firecracker   │   │
│  │ bubblewrap    │    │ per-server    │    │ microVM       │   │
│  │ (OS sandbox)  │    │ scoping       │    │ (hardware     │   │
│  │               │    │               │    │  isolation)   │   │
│  │ Command       │    │ Audit logging │    │               │   │
│  │ allowlist     │    │               │    │ No network    │   │
│  │               │    │ AAIF working  │    │ by default    │   │
│  │ Human-in-     │    │ groups        │    │               │   │
│  │ the-loop      │    │               │    │ V8 isolate    │   │
│  └───────────────┘    └───────────────┘    └───────────────┘   │
│                                                                │
│  Maturity:            Maturity:             Maturity:           │
│  Decades of Unix      30+ CVEs in 2026     E2B: production     │
│  security practice    OWASP MCP Top 10     Wasm: emerging      │
│                       82% path traversal                       │
└────────────────────────────────────────────────────────────────┘
```

> **CLI is a known risk with known mitigations.** MCP's attack surface is novel and poorly understood. Code execution sandboxes (E2B, Wasm) offer the strongest isolation but are newest.

**References:**
- OWASP MCP Top 10 [[owasp.org](https://owasp.org/www-project-mcp-top-10/)]
- MCP Security: 30 CVEs in 60 Days [[heyuan110.com](https://www.heyuan110.com/posts/ai/2026-03-10-mcp-security-2026/)]
- Simon Willison on MCP Prompt Injection [[simonwillison.net](https://simonwillison.net/2025/Apr/9/mcp-prompt-injection/)]
- Claude Code Sandboxing [[Anthropic Engineering](https://www.anthropic.com/engineering/claude-code-sandboxing)]

---

## The Decision Framework

Use this flowchart when choosing a tool interface:

```
                    ┌─────────────────────────────┐
                    │  Does a CLI tool exist for   │
                    │  this operation?              │
                    └──────┬──────────────┬────────┘
                           │ Yes          │ No
                           ▼              ▼
                    ┌──────────┐   ┌─────────────────────────┐
                    │ Is it a  │   │ Is it a multi-step       │
                    │ single   │   │ workflow against an API?  │
                    │ command? │   └──────┬──────────┬────────┘
                    └──┬───┬──┘          │ Yes      │ No
                       │   │             ▼          ▼
                       │   │      ┌──────────┐  ┌────────────────┐
                       │Yes│No    │Use Code  │  │ Does it need   │
                       ▼   ▼      │Execution │  │ OAuth / audit? │
                 ┌──────┐ ┌────┐  │(sandbox) │  └──┬──────────┬──┘
                 │ Use  │ │Use │  └──────────┘     │ Yes      │ No
                 │ CLI  │ │Code│                    ▼          ▼
                 │      │ │Exec│             ┌──────────┐ ┌──────────┐
                 └──────┘ │    │             │ Use MCP  │ │ Visual-  │
                          └────┘             │          │ │ only UI? │
                                             └──────────┘ └──┬────┬──┘
                                                             │Yes │ No
                                                             ▼    ▼
                                                       ┌───────┐┌──────┐
                                                       │Browser││ API- │
                                                       │  Use  ││Direct│
                                                       └───────┘└──────┘
```

### Rules of Thumb

1. **Start with CLI.** Cheapest, most reliable, LLM already knows it.
2. **Use Code Execution for multi-step workflows.** Collapses N round-trips to 1.
3. **Add MCP when you need auth, audit, or API-only services.** Not for things with good CLIs.
4. **Browser/Computer Use is the last resort.** Only when no API exists at all.
5. **Lazy-load everything.** Don't pay 44,000 tokens for 43 schemas when you'll use 2.
6. **Monitor token costs.** MCP's 32x overhead is invisible until you check your bill.
7. **Compress CLI output.** Tools like RTK cut CLI token cost by 60-90% — the cheapest interface gets even cheaper.

---

## Token Optimization: The War on Schema Bloat

MCP's schema bloat problem has spawned an entire subfield:

| Strategy | Token Reduction | Complexity | Reference |
|----------|---------------|------------|-----------|
| **Lazy Loading** (Claude Code v2.1.7) | 108K → 5K (95%) | Low | [[Claude Code Docs](https://code.claude.com/docs/en/overview)] |
| **Code Mode** (Cloudflare) | 1.17M → 1K (99.9%) | Medium | [[Cloudflare Blog](https://blog.cloudflare.com/code-mode/)] |
| **Dynamic Toolsets** | 44K → 3K (93%) | Medium | [[Speakeasy](https://www.speakeasy.com/blog/how-we-reduced-token-usage-by-100x-dynamic-toolsets-v2)] |
| **MCP Gateway** | 90% (schema filtering) | Medium | [[StackOne](https://www.stackone.com/blog/mcp-token-optimization/)] |
| **SKILL.md Pattern** | 33% fewer tool calls | Low | [[Claude Code → Codex CLI → Gemini CLI](https://www.anthropic.com/engineering/claude-code-best-practices)] |
| **RTK** (CLI output compression) | 60-90% CLI tokens | Low | [[GitHub](https://github.com/rtk-ai/rtk)] |
| **RAG-MCP** | >50% + 3x accuracy | High | [[arxiv:2505.03275](https://arxiv.org/abs/2505.03275)] |

The last entry — **RAG-MCP** — is a bridge to [Part 2](/posts/orchestrating-tool-interfaces-graphrag/), where we explore how **GraphRAG enables intelligent tool discovery** that makes schema bloat a non-issue.

---

## What's Next: From Landscape to Orchestration

Knowing the interfaces is step one. The harder question is **how to combine them** into a well-designed harness:

- How does an agent **decide** which interface to use for each step?
- How is **state maintained** when switching between CLI (stateless), MCP (session), and code execution (sandbox)?
- When one interface **fails** (MCP timeout), how do you **fall back** to another (CLI)?
- With 50+ tools available, how does the agent **find the right one** without loading all schemas?

These questions are the subject of **[Part 2: Orchestrating Tool Interfaces — From Harness Design to GraphRAG](/posts/orchestrating-tool-interfaces-graphrag/)**.

---

## References

### Specifications and Official Docs

1. **MCP Specification (2025-11-25)** — [modelcontextprotocol.io/specification/2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25)
2. **One Year of MCP: Anniversary Spec Release** — [blog.modelcontextprotocol.io](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/)
3. **Anthropic: Donating MCP to AAIF** — [anthropic.com](https://www.anthropic.com/news/donating-the-model-context-protocol-and-establishing-of-the-agentic-ai-foundation)
4. **Google A2A Protocol** — [developers.googleblog.com](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/)
5. **OpenAI Function Calling** — [platform.openai.com](https://platform.openai.com/docs/guides/function-calling)
6. **Claude Tool Use** — [platform.claude.com](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview)
7. **Gemini Function Calling** — [ai.google.dev](https://ai.google.dev/gemini-api/docs/function-calling)

### Benchmarks and Analysis

8. **MCP vs CLI: Benchmarking Cost & Reliability** — [scalekit.com](https://www.scalekit.com/blog/mcp-vs-cli-use)
9. **Why CLI Tools Are Beating MCP** — [jannikreinhard.com](https://jannikreinhard.com/2026/02/22/why-cli-tools-are-beating-mcp-for-ai-agents/)
10. **MCP vs CLI for AI-Native Development** — [circleci.com](https://circleci.com/blog/mcp-vs-cli/)
11. **CLI vs MCP, or CLI + MCP** — [shubhdeepchhabra.in](https://www.shubhdeepchhabra.in/blog/cli-mcp-ai)
12. **AI Agent Protocol Ecosystem Map 2026** — [digitalapplied.com](https://www.digitalapplied.com/blog/ai-agent-protocol-ecosystem-map-2026-mcp-a2a-acp-ucp)

### Code Execution and Sandboxing

13. **Cloudflare Code Mode** — [blog.cloudflare.com](https://blog.cloudflare.com/code-mode/)
14. **Cloudflare Dynamic Workers** — [blog.cloudflare.com](https://blog.cloudflare.com/dynamic-workers/)
15. **Anthropic Programmatic Tool Calling** — [platform.claude.com](https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling)
16. **Anthropic Advanced Tool Use** — [anthropic.com/engineering](https://www.anthropic.com/engineering/advanced-tool-use)
17. **E2B Sandboxes** — [e2b.dev](https://e2b.dev/)
18. **OpenAI Code Interpreter** — [developers.openai.com](https://developers.openai.com/api/docs/guides/tools-code-interpreter)
19. **Microsoft Wassette (WebAssembly)** — [opensource.microsoft.com](https://opensource.microsoft.com/blog/2025/08/06/introducing-wassette-webassembly-based-tools-for-ai-agents/)
20. **NVIDIA Wasm for Agentic AI** — [developer.nvidia.com](https://developer.nvidia.com/blog/sandboxing-agentic-ai-workflows-with-webassembly/)

### Browser and Computer Use

21. **Anthropic Computer Use** — [platform.claude.com](https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool)
22. **OpenAI CUA / Operator** — [openai.com](https://openai.com/index/computer-using-agent/)
23. **Stagehand v3** — [stagehand.dev](https://www.stagehand.dev/)

### Security

24. **OWASP MCP Top 10** — [owasp.org](https://owasp.org/www-project-mcp-top-10/)
25. **MCP Security: 30 CVEs in 60 Days** — [heyuan110.com](https://www.heyuan110.com/posts/ai/2026-03-10-mcp-security-2026/)
26. **Claude Code Sandboxing** — [anthropic.com/engineering](https://www.anthropic.com/engineering/claude-code-sandboxing)
27. **Simon Willison: MCP Prompt Injection** — [simonwillison.net](https://simonwillison.net/2025/Apr/9/mcp-prompt-injection/)

### Architecture and Patterns

28. **Inside Claude Code Architecture** — [penligent.ai](https://www.penligent.ai/hackinglabs/inside-claude-code-the-architecture-behind-tools-memory-hooks-and-mcp/)
29. **AI Agent CLI + MCP Hybrid Architecture** — [stackone.com](https://www.stackone.com/blog/ai-agent-cli-mcp-hybrid-architecture/)
30. **MCP Token Optimization: 4 Approaches** — [stackone.com](https://www.stackone.com/blog/mcp-token-optimization/)
31. **Reducing MCP Token Usage by 100x** — [speakeasy.com](https://www.speakeasy.com/blog/how-we-reduced-token-usage-by-100x-dynamic-toolsets-v2)
32. **RAG-MCP: Mitigating Prompt Bloat** — [arxiv:2505.03275](https://arxiv.org/abs/2505.03275)
33. **The Protocol Wars** — [theregister.com](https://www.theregister.com/2026/01/30/agnetic_ai_protocols_mcp_utcp_a2a_etc/)

### CLI Optimization

34. **RTK: Rust Token Killer** (18.6k stars) — [github.com/rtk-ai/rtk](https://github.com/rtk-ai/rtk)
35. **RTK Architecture** — [github.com/rtk-ai/rtk/ARCHITECTURE.md](https://github.com/rtk-ai/rtk/blob/master/ARCHITECTURE.md)
36. **ICM: Persistent Memory for Agents** — [github.com/rtk-ai/icm](https://github.com/rtk-ai/icm)
37. **Grit: Git for Parallel Agents** — [github.com/rtk-ai/grit](https://github.com/rtk-ai/grit)
