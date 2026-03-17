---
title: "From OpenClaw to NemoClaw: The Technical Landscape of the Claw AI Agent Ecosystem"
date: 2026-03-17 11:00:00 +0900
categories: [Engineering]
tags: [ai-agent, openclaw, security, systems-engineering, rust, zig, container-isolation]
---

In November 2025, Austrian developer Peter Steinberger published a project called Clawdbot — a TypeScript wrapper that connected a chat application to Claude Code, enabling an LLM to execute system-level tasks through messaging platforms. Four months later, the project (now renamed OpenClaw) had accumulated over 240,000 GitHub stars, been acquired by OpenAI, and generated a family of derivative frameworks spanning from 678-kilobyte Zig binaries to NVIDIA's enterprise agent platform.

This post traces the technical evolution of the Claw ecosystem: OpenClaw's architecture and its security limitations, the specific engineering decisions each variant made to address those limitations, and the applications each approach enables.

---

## OpenClaw: Architecture and Adoption

### How It Works

OpenClaw is a pnpm workspace monorepo written in TypeScript/Node.js, organized into several packages:

| Package | Role |
|---------|------|
| `packages/core` | Framework core, provider interfaces, base classes |
| `packages/gateway` | WebSocket-based control plane, session management, channel routing |
| `packages/agent` | Agent runtime with RPC mode, tool streaming |
| `packages/cli` | CLI for onboarding and gateway control |
| `packages/sdk` | SDK for custom tools and plugins |
| `packages/ui` | WebChat interface and control dashboard |

The **skill system** is central to OpenClaw's extensibility. Each skill is a directory containing a `SKILL.md` file with YAML frontmatter and natural-language instructions. When a session starts, eligible skills are injected as a compact XML list into the system prompt (195 bytes base overhead + 97 bytes per skill). Skills are discovered through a precedence chain: workspace-level → user-level → bundled defaults, with additional directories configurable via `skills.load.extraDirs`.

**ClawHub**, the public skill registry, uses a React + TanStack Start frontend with a Convex backend. Skill discovery relies on embedding-based vector search rather than keyword matching — a design choice that would later become a vector for supply chain attacks.

### Timeline

- **November 2025**: Steinberger publishes Clawdbot on GitHub. Built on top of Clawd (later renamed Molty), an AI virtual assistant he had developed.
- **Late January 2026**: Adoption accelerates rapidly. The project gains tens of thousands of stars within days, reaching 100,000. Developers adopt it for chaining multi-step workflows — email management, calendar operations, bash command execution — across WhatsApp, Discord, and Slack.
- **January 27, 2026**: Anthropic files a trademark complaint over the "Clawd" name. Project renamed to Moltbot.
- **January 28, 2026**: Moltbook launches — an AI-agent-only forum (Reddit-style, restricted to bot participation). It claims 1.6 million registered agents by February, though a misconfigured Supabase database later reveals these belong to only 17,000 human owners.
- **January 30, 2026**: Renamed again to OpenClaw. Three name changes in one week.
- **February 14, 2026**: Steinberger announces he is joining OpenAI. The project moves to an independent open-source foundation with OpenAI's financial sponsorship. By this point, 247,000 GitHub stars.
- **March 2026**: Chinese authorities restrict state agencies from running OpenClaw on office computers, citing security risks.

### The Robotics Integration

The `unitree-robot` skill demonstrated OpenClaw's scope beyond text-based automation. It provides text-to-motion control for the Unitree G1 humanoid robot through messaging platforms — users issue commands like `"forward 1m"` or `"turn left 45 degrees"` without needing the vendor SDK.

More technically interesting is the spatial agent memory layer. The skill integrates RGB-D frames from the robot's stereo camera, LiDAR data, and odometry readings to construct a voxelized world model (**SpatialRAG**). This gives the agent a persistent spatial representation it can query — e.g., "where did I see the red box?" — rather than operating frame-by-frame.

Other integrations include a 3D-printed 16-joint Aero Hand Open (controlled via USB camera feedback) and a NERO 7-axis robotic arm via the pyAgxArm SDK.

---

## The Security Problem

### CVE-2026-25253 (CVSS 8.8)

The most critical vulnerability exposed a fundamental architectural weakness. Before version 2026.1.29, OpenClaw accepted a `gatewayUrl` parameter from a query string and established a WebSocket connection to that URL without user confirmation, transmitting the authentication token.

The attack chain:

1. Victim clicks a malicious link while OpenClaw is running locally.
2. Cross-site WebSocket hijacking succeeds because OpenClaw's server does not validate the WebSocket `Origin` header — it accepts connections from any website, bypassing localhost network restrictions.
3. Attacker receives the authentication token.
4. Attacker sends `exec.approvals.set` to disable user confirmation prompts.
5. Attacker sends `config.patch` to force commands to run on the host machine (not in a Docker container).
6. Attacker sends `node.invoke` to execute arbitrary shell commands.

As of February 2026, Censys found over 135,000 OpenClaw instances exposed to the public internet, with 15,000+ specifically vulnerable to this RCE.

### ClawHavoc: The Supply Chain Attack

The ClawHavoc campaign targeted ClawHub's skill registry directly. Researchers traced 335 core malicious skills to a single threat actor, with the total eventually reaching 1,184+ confirmed malicious packages across 10,700+ total registry entries — roughly 20% of the registry.

The attack mechanism was multi-staged:

1. **Skill injection**: Malicious instructions hidden in `SKILL.md` files. Skills appeared legitimate (e.g., `solana-wallet-tracker`, `youtube-summarize-pro`) with professional documentation.
2. **Prerequisite injection**: A "Prerequisites" section instructed users to install an `openclaw-agent` utility — on macOS, by copy-pasting an installation script from `glot.io` into Terminal.
3. **Social engineering**: A deceptive human-in-the-loop dialog tricked users into entering their password.
4. **Payload**: A variant of Atomic macOS Stealer (AMOS), a commodity malware ($500-1000/month), delivered as a universal Mach-O binary. It harvested browser credentials, keychain passwords, cryptocurrency wallets, SSH keys, Apple Notes, and KeePass databases.

The fundamental issue was architectural: OpenClaw ran with the user's full privileges on the host OS. Application-level permission checks could not substitute for real process isolation. This gap motivated every Claw variant that followed.

---

## The Claw Variants: Engineering Decisions and Trade-offs

Each variant addressed OpenClaw's limitations through different engineering choices. What follows is a technical examination of how each one works, what trade-offs it makes, and what applications it enables.

### NanoClaw — Container Isolation with a Minimal Core

**Core architecture.** NanoClaw is a single Node.js process with approximately 15 source files and ~500 lines of core logic (roughly 4,000 lines including channel adapters). It runs directly on Anthropic's Claude Agent SDK with no abstraction layer — no multi-provider support, no custom LLM wrappers. Message ingestion for WhatsApp uses the Baileys library; other channels (Telegram, Slack, Discord, Signal, Gmail) use their respective adapters. Persistence is handled by SQLite for memory, message queues, and per-group concurrency control.

**Execution model.** The core differentiator is the container-per-invocation lifecycle:

1. A message arrives on a channel (e.g., WhatsApp group).
2. The orchestrator checks if the message matches the trigger pattern (default: `@Andy`).
3. A fresh, ephemeral Linux container is spawned with only the relevant group's directory mounted.
4. Inside the container, the Claude Agent SDK handles reasoning and tool use.
5. The container is destroyed after the invocation completes.

The agent runs as an unprivileged user. A mount allowlist at `~/.config/nanoclaw/mount-allowlist.json` blocks sensitive paths by default (`.ssh`, `.gnupg`, `.aws`, `.env`, `credentials`). The allowlist file lives outside the project directory, so a compromised agent cannot modify its own permissions. On macOS, NanoClaw uses Apple Containers (native sandboxing); on Linux, Docker containers.

**Docker partnership (March 2026).** The collaboration with Docker added a second isolation layer: MicroVM-based sandboxing. The architecture is two-tiered:

```
Host (macOS / Windows WSL)
  └── Docker Sandbox (MicroVM with isolated kernel)
        └── NanoClaw process + channel adapters
              └── Nested Docker containers (one per agent)
```

The MicroVM runs its own kernel and its own Docker daemon. Even a container escape stays within the MicroVM boundary. Credentials are managed through a MITM proxy at `host.docker.internal:3128` that injects the Anthropic API key — credentials never need to exist inside the sandbox.

**Applications.** The NanoClaw creators run a production instance ("Andy") for sales pipeline management: daily briefings at 9:00 AM with lead statuses and task assignments, parsing WhatsApp notes and email threads, updating databases, and setting follow-up reminders. The multi-agent collaboration feature allows teams of specialized agents to work within the same chat, each sub-agent isolated in its own container with its own memory context.

**Trade-offs.** NanoClaw is Claude-only (no multi-provider support). The container overhead adds latency per invocation. And the ~400MB RAM footprint, while much smaller than OpenClaw's 1.5GB, is still substantial compared to the systems-language variants.

### ZeroClaw — Trait-Driven Rust Agent Runtime

**Core architecture.** ZeroClaw's design is organized around Rust traits that abstract every major subsystem:

- **`Provider` trait**: Abstracts AI backends (OpenAI, Anthropic, Ollama, OpenRouter, any OpenAI-compatible endpoint). A `create_provider` factory enables runtime provider selection via config, with normalized token-cost tracking across providers.
- **`Channel` trait**: Abstracts messaging platforms (11 implementations: Telegram, Discord, Slack, Matrix, etc.). All channels share a single bounded `tokio::sync::mpsc` message bus for unified message routing.
- **`Tool` trait**: Agent capabilities (35+ built-in).
- **`Memory` trait**: Persistence backends.
- **`Peripheral` trait**: Hardware interfaces.

**Concurrency model.** ZeroClaw uses the Tokio async runtime. In daemon mode, three concurrent subsystems run as supervised tasks with automatic restart on failure using exponential backoff. If an LLM call exceeds 90 seconds, the agent sends a timeout error and continues processing the next message — one slow request never blocks the system. Thread safety is enforced through Rust's `Send`/`Sync` traits with minimal `Arc` usage.

**Memory and persistence.** ZeroClaw builds a hybrid search engine on top of SQLite with no external dependencies:

- **Vector search**: Embeddings stored as SQLite BLOBs, cosine similarity computed in-process.
- **Keyword search**: SQLite FTS5 extension with BM25 scoring.
- **Weighted merge**: Default 70% vector (cosine similarity) + 30% keyword (BM25).

Everything persists in a single `memory.db` file. On a Raspberry Pi Zero 2 W, total retrieval takes under 3ms (0.3ms FTS5, 2ms vector, 0.1ms merge).

**Performance.** 3.4MB binary, <5MB RAM, cold start under 10ms. Built-in cron (`zeroclaw cron add "0 9 * * *" --tz "Asia/Shanghai" "task"`) plus a `HEARTBEAT.md` mechanism for periodic agent wake-ups.

**Applications.** ZeroClaw targets self-hosted deployments on constrained hardware: Raspberry Pis, old laptops, $5 VPS instances. Documented use cases include edge/IoT assistants that switch between local and cloud models while keeping data on-premise, modular moderation/automation platforms routing messages across Telegram/Discord/Slack, and enterprise document processing (contract compliance, insurance claims, medical case summarization) with workspace isolation.

**Trade-offs.** No built-in container isolation — security relies on the smaller attack surface of ~3.4MB compiled Rust rather than process-level sandboxing. Users who need strong isolation must compose ZeroClaw with external containerization.

### NullClaw — Zig for Minimum Footprint

**Core architecture.** NullClaw is written in Zig with manual memory management — no allocator overhead, no garbage collector, no runtime. The binary compiles to 678KB and requires only libc. LLM API calls use Zig's `std.http.Client.fetch()` with `std.Io.Writer.Allocating` for response body capture — the standard library HTTP client with no external dependencies.

Every subsystem is implemented as a **vtable interface** (Zig's equivalent of traits/interfaces), enabling zero-cost polymorphism. The binary supports 50+ LLM providers and 19 communication channels, all through this abstraction.

**Memory system.** Like ZeroClaw, NullClaw uses a SQLite-based hybrid search engine: FTS5 virtual tables with BM25 scoring for keyword search, embeddings stored as compressed BLOBs for vector search, and a weighted merge of both result sets. This avoids the common failure mode where pure vector search misses specific identifiers, and pure keyword search misses semantic intent. The entire system runs in-process on SQLite — no external vector database, no network overhead.

**Sandboxing.** NullClaw implements sandboxing as a vtable interface with multiple backends: Landlock (Linux Security Module for filesystem restriction at the kernel level), Firejail (namespace-based sandboxing with seccomp filters), Bubblewrap (lightweight unprivileged container sandboxing), and Docker. The runtime auto-detects host environment capabilities and selects the strongest available sandbox. These are not layered simultaneously — the system picks the best option based on kernel features and installed tools. API keys are encrypted with ChaCha20-Poly1305 using a local key file, and every action is recorded in a cryptographically signed event trail.

**Performance.** 678KB binary, ~1MB RAM, <2ms boot on Apple Silicon, <8ms on 0.8 GHz edge cores. The codebase comprises ~45,000 lines of Zig with 2,738 tests.

**Applications.** NullClaw's per-instance footprint (~1MB) enables deployment patterns that would be impractical with heavier runtimes. Researchers have deployed hundreds of NullClaw agent instances on a single commercial server for multi-agent simulations. On IoT hardware, it provides native interfaces for Serial, Arduino, Raspberry Pi GPIO, and STM32, enabling use cases like sensor monitoring and device control with zero cloud dependency (when paired with Ollama for local inference). It runs on the same class of low-power hardware typically used for DNS servers or ad blockers.

**Trade-offs.** Manual memory management in Zig means memory safety is the developer's responsibility — no borrow checker. The auto-select sandbox model means security guarantees vary depending on the host environment.

### PicoClaw — Embedded Agent Orchestration in Go

**Core architecture.** PicoClaw compiles to a single Go binary with zero external dependencies. It is not an inference engine — it is an **agentic orchestration shell** that handles tool routing, chat platform gateways, scheduling, and provider configuration. LLM calls go to remote API providers (OpenRouter, Anthropic, OpenAI, DeepSeek, Zhipu AI, Groq) via a model-centric config format (`vendor/model`, e.g. `zhipu/glm-4.7`).

Created by Sipeed, a hardware company known for RISC-V development boards, PicoClaw was launched on **February 9, 2026** and built in a single day. The primary target hardware is Sipeed's own **$9.90 LicheeRV-Nano** board.

**Local inference.** For on-device inference without cloud APIs, a companion project **PicoLM** (`github.com/RightNow-AI/picolm`) serves as the "local brain." It uses memory-mapped model files (mmap on Linux/macOS), 4-bit quantized weights (Q4_K_M format), and keeps only ~30MB of model in physical memory at any time. This allows running a 1B-parameter LLM on a board with 256MB RAM. Ollama is also supported but has known timeout issues (120s fixed limit).

**The "95% agent-generated" claim.** PicoClaw originated as a refactor of NanoBot (a TypeScript/Zig project). The AI agent drove the entire Go migration — architecture decisions, code generation, and optimization. A human developer provided high-level direction and refinement. The result reflects the aggressiveness of AI-driven optimization: <10MB RAM footprint, sub-second boot on 0.6GHz hardware.

**Performance.** <10MB RAM, sub-1-second boot on 0.6GHz single core, single binary across RISC-V, ARM, MIPS, and x86. 17,000+ GitHub stars within 11 days of launch.

**Applications.** Documented deployments include always-on home assistants on the $9.90 LicheeRV-Nano (reminders, monitoring, automation via Ethernet or WiFi6), automated server maintenance bots on NanoKVM ($30-50), and general-purpose assistants on Raspberry Pi hardware. The gateway system (`picoclaw gateway`) supports Telegram, Discord, QQ, and DingTalk for messaging integration. Built-in cron handles one-time reminders, recurring tasks, and standard cron expressions.

**Trade-offs.** PicoClaw is in early development with acknowledged unresolved network security issues. No sandboxing mechanism. The project explicitly warns against production use before v1.0.

### IronClaw — TEE-Based Security for Always-On Agents

**Core architecture.** IronClaw is built in Rust by NEAR AI and launched at NEARCON on **February 23, 2026**. The project was championed by **Illia Polosukhin** — co-author of the 2017 "Attention Is All You Need" (Transformer) paper and co-founder of NEAR Protocol.

IronClaw uses a **four-layer defense-in-depth** architecture:

1. **TEE Enclave**: The agent instance boots inside a hardware Trusted Execution Environment on NEAR AI Cloud. The computation is encrypted in memory from boot to shutdown — not just data at rest, but active computation. This provides hardware-enforced confidentiality: even the infrastructure operator cannot inspect the agent's state.
2. **AES-256-GCM Encrypted Vault**: API keys and passwords are stored with AES-256-GCM encryption. Each credential is bound to a policy rule restricting its use to a single specified domain.
3. **WebAssembly Sandbox**: Every third-party tool runs in its own Wasmtime WASM container with capability-based permissions, allowlisted endpoints, and strict resource limits.
4. **Leak Detection**: Outbound requests and responses are scanned for secret exfiltration attempts.

**Credential injection — the boundary model.** This is IronClaw's most technically distinctive feature. WASM tools specify URLs and headers using placeholder syntax (e.g., `{{SECRET:my_api_key}}`). The WASM module never receives the decrypted secret value — it only sees the placeholder. At the network boundary (when the HTTP request is actually sent), the host runtime replaces placeholders with decrypted values. Credentials are only decrypted for approved endpoints matching policy rules. The LLM never sees credentials at any point in the pipeline.

**Multi-agent orchestration.** Every agent action is gated through policy checks covering approved tools, data scopes, spending limits, model permissions, and access boundaries. The core principle: LLM output is treated as untrusted intent. All proposed actions pass a non-bypassable gating step before execution. A real-time dashboard enables monitoring agents, reviewing actions, and approving escalations.

**Applications.** IronClaw targets use cases where agents handle sensitive assets. In crypto/DeFi, agents perform deal flow analysis, portfolio monitoring, and automated trading — with the critical property that private keys can be used for signing transactions without the LLM ever seeing the key. The free Starter tier (one agent instance) and paid scaling tiers make it the most commercially oriented Claw variant. NEAR positions it as part of a broader "Confidential Cross-Chain Infrastructure for the Agentic Economy."

**Trade-offs.** Requires NEAR AI Cloud for TEE — no self-hosted option for the TEE layer. The WASM sandbox adds overhead to tool execution. The credential boundary injection model, while secure, adds complexity to tool development.

---

## NVIDIA NemoClaw: Enterprise Policy Enforcement

On **March 16, 2026**, NVIDIA CEO Jensen Huang announced NemoClaw at GTC 2026. Rather than building a new agent runtime, NVIDIA built a policy enforcement layer around OpenClaw, integrating it with NVIDIA's NeMo Agent Toolkit.

### Architecture

NemoClaw is structured as an **OpenClaw plugin for OpenShell** with three layers:

1. **Plugin layer**: A thin TypeScript plugin registering commands under the `openclaw nemoclaw` namespace.
2. **Blueprint system**: A versioned Python artifact containing sandbox creation logic, policy application, and inference configuration. The plugin resolves, verifies, and executes the blueprint as a subprocess.
3. **Sandbox environment**: A pre-configured OpenShell sandbox with strict policies applied from first boot.

### OpenShell: Kernel-Level Policy Enforcement

OpenShell uses kernel-level, out-of-process policy enforcement — not behavioral prompts. Even a compromised agent cannot override policies. Enforcement operates across four domains:

- **Filesystem**: Prevents reads/writes outside `/sandbox` and `/tmp`. Policies are locked at sandbox creation.
- **Network**: Every outbound connection is intercepted. Only explicitly allowlisted endpoints (declared in a YAML policy file) are permitted. Unlisted hosts are blocked and surfaced in a TUI for operator approval. Network policies are hot-reloadable at runtime via `openshell policy set`.
- **Process**: Blocks privilege escalation and dangerous syscalls. Locked at sandbox creation.
- **Inference**: Reroutes model API calls to controlled backends. Hot-reloadable.

Policies are declarative YAML files that can be version-controlled alongside the project.

### Privacy Router

The inference routing system intercepts every model API call from the agent inside the sandbox. Agent traffic never leaves the sandbox directly. OpenShell routes requests to one of three configured providers depending on the selected profile:

- Cloud-hosted Nemotron 3 Super 120B
- A local NIM (NVIDIA Inference Microservice) instance
- vLLM

This enables enterprises to route sensitive data to on-premise models while using cloud models for less sensitive tasks.

### Integration and Deployment

NemoClaw integrates NeMo (model training and agent reasoning), the Nemotron model family (released December 2025), and NIM inference microservices. A single command through the NVIDIA Agent Toolkit sets up Nemotron + OpenShell together. Supported hardware includes RTX PCs, DGX Station, and DGX Spark. The platform is hardware-agnostic at the agent level — it does not require NVIDIA GPUs — though GPU acceleration provides performance advantages when available.

NVIDIA has been working with Salesforce, Cisco, Google Cloud, Adobe, and CrowdStrike ahead of the launch. Cisco published a blog post on "Securing Enterprise Agents with NVIDIA OpenShell and Cisco AI Defense." Target use cases include customer service with compliance boundaries, SOC automation for security operations, autonomous software engineering, and multi-department agent deployment in supply chain and finance.

NemoClaw is currently in early-stage alpha. NVIDIA's own documentation states: "Expect rough edges."

---

## Comparison

### Technical Specifications

| Variant | Language | Binary | RAM | Boot | Providers | Sandbox Model |
|---------|----------|--------|-----|------|-----------|---------------|
| **OpenClaw** | TypeScript | ~28MB | ~1.5GB | — | Multi (50+) | None (host process) |
| **NanoClaw** | TypeScript | ~15MB | ~400MB | — | Claude only | Container + MicroVM |
| **ZeroClaw** | Rust | 3.4MB | <5MB | <10ms | 22+ | None (minimal surface) |
| **NullClaw** | Zig | 678KB | ~1MB | <2ms | 50+ | Auto-select (Landlock/Firejail/Bubblewrap/Docker) |
| **PicoClaw** | Go | ~5MB | <10MB | ~1s | Multi (API) | None |
| **IronClaw** | Rust | ~8MB | ~50MB | — | Multi | TEE + WASM |
| **NemoClaw** | TS + Python | Enterprise | Enterprise | — | Nemotron/NIM/vLLM | Kernel-level policy (OpenShell) |

### Security Approaches

The variants represent five distinct strategies for the same problem — how to give an agent system access while limiting blast radius:

- **NanoClaw**: OS-level container isolation. The agent literally cannot see files or processes outside its container. The MicroVM adds a second boundary so that even a container escape is contained.
- **ZeroClaw / NullClaw**: Minimal attack surface. A 678KB-3.4MB compiled binary has fewer exploitable code paths than a 400K-line TypeScript monorepo. Security through reduction rather than containment.
- **PicoClaw**: Physical isolation by deployment target. An agent running on a $10 dedicated board is inherently separated from your primary workstation. (Though PicoClaw itself has no sandboxing, which limits this argument.)
- **IronClaw**: Hardware-enforced encryption via TEE. The computation itself is encrypted in memory. The WASM sandbox prevents tools from seeing credentials. The boundary injection model ensures secrets never exist in the LLM's context.
- **NemoClaw**: Declarative policy enforcement at the kernel level. YAML-defined rules control filesystem, network, process, and inference routing. Policies are locked at sandbox creation and (for some domains) immutable at runtime.

None of these is strictly superior. Container isolation (NanoClaw) provides the strongest general-purpose containment but adds per-invocation overhead. TEE (IronClaw) provides the strongest credential protection but requires specific cloud infrastructure. Kernel-level policies (NemoClaw) provide the most granular control but depend on correct policy authoring. Minimal footprint (ZeroClaw/NullClaw) reduces attack surface but doesn't prevent a compromised binary from acting within its permissions.

### Memory Architecture Convergence

An interesting technical convergence: ZeroClaw and NullClaw independently arrived at the same hybrid memory design — SQLite FTS5 for keyword search (BM25 scoring) combined with in-process vector similarity (embeddings stored as SQLite BLOBs, cosine similarity computed without external dependencies). Both use weighted merging of results (ZeroClaw defaults to 70/30 vector/keyword). This pattern — avoiding external vector databases by embedding search directly into SQLite — appears to be a pragmatic solution for self-hosted agents where minimizing dependencies matters more than scaling to millions of documents.

### Language Choice as Design Statement

Each language choice reflects the variant's priorities:

- **TypeScript** (OpenClaw, NanoClaw): Maximizes developer ecosystem access and iteration speed. Pays the cost in binary size (~28MB) and RAM (~1.5GB).
- **Rust** (ZeroClaw, IronClaw): Compile-time memory safety without garbage collection. The borrow checker eliminates data races; no GC pauses in a real-time agent loop.
- **Go** (PicoClaw): Straightforward cross-compilation to RISC-V/ARM/MIPS/x86 from a single codebase. Good concurrency primitives for gateway multiplexing.
- **Zig** (NullClaw): Manual memory management yields the smallest binary (678KB) and lowest RAM (~1MB). The trade-off is that memory safety is the developer's responsibility.
- **Python** (NemoClaw blueprint layer): Integrates with NVIDIA's existing ML/AI stack (NeMo, NIM). Not the agent runtime itself — the TypeScript plugin calls Python subprocess.

---

## How Applications Evolved Across the Claw Ecosystem

The Claw variants did not just differ in implementation language or binary size — they progressively expanded the scope of what AI agent applications could do and where they could run. This evolution unfolded in five distinct phases, each defined by a shift in the security model that unlocked new categories of use cases.

### Phase 1: Personal Automation on the Host OS — OpenClaw (Nov 2025 – Jan 2026)

OpenClaw's initial applications centered on personal productivity automation. Users chained multi-step workflows — email management, calendar operations, bash command execution — through messaging platforms. The skill ecosystem (5,000+ packages on ClawHub) extended this to web browsing, file management, and API integrations.

The `unitree-robot` skill pushed applications into physical robotics: text-to-motion control for the Unitree G1 humanoid, spatial memory via SpatialRAG (voxelized world model from RGB-D + LiDAR + odometry), and integrations with robotic arms and hands.

The constraint: all of this ran with the user's full OS privileges. As application scope grew, so did the attack surface — ClawHavoc (20% of the skill registry compromised) and CVE-2026-25253 (1-click RCE on 15,000+ exposed instances) demonstrated that the security model could not scale with the application ambitions.

### Phase 2: Isolated Workflow Automation — NanoClaw (Feb 2026~)

NanoClaw did not introduce fundamentally new application types. Instead, it made existing applications safe to run in production. The representative deployment is "Andy" — a sales pipeline management bot that delivers daily lead briefings at 9:00 AM, parses WhatsApp and email threads, updates databases, and sets follow-up reminders. OpenClaw could do the same work, but NanoClaw's container isolation ensures the agent cannot access `.ssh` keys or AWS credentials while generating a briefing.

The more significant application development was multi-agent collaboration (Agent Swarms). Multiple specialized agents operate within the same chat, each running in its own container with its own memory context. A sales agent cannot see the marketing agent's credentials — isolation is structural, not policy-based. The Docker MicroVM partnership (March 2026) added a second boundary: even a container escape stays within the MicroVM.

### Phase 3: Edge and IoT Deployment — ZeroClaw, NullClaw, PicoClaw (Feb 2026~)

These three variants expanded where agents could run, opening application categories that were physically impossible with OpenClaw's 1.5GB RAM footprint.

**ZeroClaw** (3.4MB, <5MB RAM) enabled agents as system daemons on constrained hardware. Documented applications include edge AI assistants on Raspberry Pi that switch between local and cloud models while keeping data on-premise, modular community automation platforms routing messages across Telegram/Discord/Slack, and enterprise document processing — contract compliance review, insurance claims, medical case summarization — with workspace isolation to prevent cross-document data leakage. The SQLite FTS5 + vector hybrid memory achieves sub-3ms retrieval on a Raspberry Pi Zero 2 W, enabling RAG-based document search without external infrastructure.

**NullClaw** (~1MB per instance) enabled applications at scales impractical with heavier runtimes. Researchers deploy hundreds to thousands of agent instances on a single server for multi-agent simulations — a deployment pattern that would require 1.5TB of RAM with OpenClaw. On IoT hardware, NullClaw provides native interfaces for Serial, Arduino, Raspberry Pi GPIO, and STM32, enabling sensor monitoring and device control on the same class of low-power hardware used for DNS servers or ad blockers. Paired with Ollama for local inference, these agents operate with zero cloud dependency.

**PicoClaw** (<10MB RAM) targeted the most extreme edge: always-on agents on $9.90 LicheeRV-Nano boards scattered around homes and offices for reminders, monitoring, and automation. The companion PicoLM project enables running a 1B-parameter LLM locally on 256MB RAM boards via mmap and 4-bit quantization (Q4_K_M), though most deployments use remote API providers. Automated server maintenance bots on NanoKVM ($30-50) hardware represent another documented use case.

### Phase 4: Sensitive Asset Management — IronClaw (Feb 23, 2026~)

IronClaw shifted the application question from "what can the agent do?" to "what can the agent not see while still acting?" The TEE + WASM + boundary credential injection architecture enabled applications involving real money and sensitive assets.

The primary application is crypto/DeFi: agents perform deal flow analysis, portfolio monitoring, and automated trading strategies where the critical property is that private keys can sign transactions without the LLM ever seeing the key (`{{SECRET:private_key}}` placeholders are decrypted only at the network boundary). Every action passes through policy gating — approved tools, data scopes, spending limits, model permissions — producing investor-grade audit trails.

Fintech and healthtech companies are experimenting with agentic workflows on regulated data where compliance constraints require that credentials and sensitive records never become LLM-visible text. IronClaw's commercial model (free Starter tier, paid scaling) reflects the willingness of these sectors to pay for verifiable security guarantees.

### Phase 5: Enterprise Fleet Management — NemoClaw (Mar 16, 2026~)

NemoClaw moved the application scope from individual agents to organizational infrastructure — deploying and governing hundreds of agents across departments under unified policy control.

Target applications include customer service agents operating against proprietary data within compliance boundaries, SOC automation for alert-heavy security operations where network access is controlled via policy YAML, autonomous software engineering (code writing, testing, debugging) with filesystem access restricted to `/sandbox` and `/tmp`, and multi-department agent deployment in supply chain and finance. The Privacy Router enables a key enterprise pattern: routing sensitive data to on-premise Nemotron/NIM instances while using cloud models for less sensitive tasks, all governed by declarative privacy profiles.

Partnerships with Cisco (AI Defense integration), Salesforce, Adobe, Google Cloud, and CrowdStrike indicate that NemoClaw's applications target IT infrastructure teams managing agent fleets, not individual users running personal assistants.

### Application Evolution Summary

| Phase | Representative | Application Scope | Key Transition |
|-------|---------------|-------------------|----------------|
| 1 | OpenClaw | Personal productivity, robotics | LLM directly operates the OS |
| 2 | NanoClaw | Workflow automation, multi-agent | Same tasks, but in isolated containers |
| 3 | ZeroClaw / NullClaw / PicoClaw | Edge AI, IoT, multi-agent simulation | Deployment targets expand to $5 boards |
| 4 | IronClaw | Finance, crypto, sensitive assets | Defining what agents *cannot see* |
| 5 | NemoClaw | Organization-wide agent infrastructure | Personal tool → enterprise fleet management |

Across these five phases, the pattern is consistent: **the security model determined the application boundary**. OpenClaw's lack of isolation limited it to low-stakes personal automation. Container isolation (NanoClaw) made production business workflows viable. Minimal footprint (ZeroClaw/NullClaw/PicoClaw) brought agents to hardware that previously couldn't run them. TEE-based credential isolation (IronClaw) unlocked financial applications. Kernel-level policy enforcement (NemoClaw) enabled organization-wide deployment. Each security architecture didn't just protect existing applications — it made new categories of applications possible.

---

## Observations

**The Claw layer is an orchestration problem, not a model problem.** All variants use external LLMs (via API or local inference servers) — none of them train or fine-tune models. What they build is the infrastructure between the model and the real world: tool invocation, context management, scheduling, persistence, multi-channel communication, and security mediation. This layer did not have a standard framework before OpenClaw.

**Security approaches are diverging, not converging.** Four months in, the ecosystem has produced at least five distinct security architectures with no signs of standardization. This suggests the industry has not yet found a broadly acceptable trust model for autonomous agents. Different deployment contexts (personal device vs. enterprise server vs. IoT board vs. crypto wallet) may require fundamentally different security approaches.

**The footprint spectrum is wider than expected.** From 1.5GB (OpenClaw) to 1MB (NullClaw) — a 1,500x range in RAM usage for functionally similar agent capabilities. This spread reflects genuinely different deployment targets (cloud server vs. $5 ARM board), but it also suggests that much of the weight in larger frameworks is ecosystem overhead rather than core agent functionality.

**AI-generated agent infrastructure is already here.** PicoClaw's 95% agent-generated codebase is a concrete example of AI bootstrapping its own infrastructure. The fact that the resulting Go binary is competitive on performance metrics with hand-written Rust (ZeroClaw) and Zig (NullClaw) variants raises questions about how much of systems programming will be human-authored in the near future.
