---
title: "Building an AI-Native Organization: From Copilots to Company Intelligence"
date: 2026-04-03 09:00:00 +0900
categories: [AI & ML, AI in Practice]
tags: [ai-native, organization, knowledge-management, ai-agent, compound-intelligence]
---

*What Dorsey and Karpathy's experiments reveal about the real architecture of AI-first teams*

---

> "Treat the company as a mini AGI... it really is. If you look at a company, it is an intelligence."
> — Jack Dorsey, Sequoia Capital podcast (Feb 2026)

In February 2026, two very different experiments surfaced the same idea.

**Jack Dorsey** went top-down. He cut 40% of Block's workforce, flattened five management layers to two, and declared the company "intelligence-native." Every employee now has access to a queryable intelligence layer built from the company's Slack messages, emails, code, and documents. Meetings no longer feature slide decks — people bring working prototypes instead.

**Andrej Karpathy** went bottom-up. Working alone, he built a personal knowledge system where raw sources — papers, articles, datasets — are dumped into a folder, and an LLM automatically compiles them into an indexed, interlinked Markdown wiki. He asks questions; the LLM writes everything. Every answer enriches the wiki for the next query.

The scale could not be more different. One reshaped a 10,000-person fintech company. The other built a solo research tool. But the underlying insight is the same:

**AI-native is not about using more AI tools. It is about restructuring how an organization creates, stores, and acts on knowledge — so that AI participates as a first-class member.**

This post extracts the common architecture from both experiments and examines what it actually takes to make this work.

---

## AI-Enhanced vs AI-Native: The Distinction That Matters

**Most companies are using AI as a faster typewriter.** They bolt ChatGPT or Copilot onto existing workflows — the same five-step approval chain, the same weekly status meetings, the same report-driven decisions. The process stays identical; it just runs a little faster.

This is AI-Enhanced. And it has a subtle trap: **it preserves existing inefficiencies while accelerating them.**

AI-Native is a fundamentally different operating mode. The distinction comes down to who leads the workflow:

|                | AI-Enhanced              | AI-Native                    |
|----------------|--------------------------|------------------------------|
| **Who leads**  | Human works, AI assists  | AI executes, human steers    |
| **Process**    | Existing + Copilot bolted on | Redesigned around intelligence |
| **Knowledge**  | Stays in people's heads  | Compiled, indexed, queryable |
| **Meetings**   | Review presentations     | Review prototypes            |
| **Improvement**| Linear (more tools)      | Compound (every answer accumulates) |

The core shift: *"Human writes, AI helps"* becomes *"AI writes, human steers."*

> "You never write the wiki. The LLM writes everything. You just steer."
> — Andrej Karpathy

This is not a philosophical difference. It changes how teams operate day to day:

- **Meetings**: reviewing a slide deck → reviewing a working prototype that was built in hours
- **Decisions**: waiting for a report → querying the organization's knowledge layer in real time
- **Onboarding**: reading documentation → asking the intelligence layer questions
- **Quality control**: humans inspect everything → AI inspects and flags, humans handle exceptions

The workflow inversion looks like this:

```
AI-Enhanced:
  Human ── work ──→ AI assists ──→ Human reviews ──→ Output

AI-Native:
  Human ── direction ──→ AI executes ──→ Human steers ──→ Knowledge
                                                            │
                                             ←── compounds ─┘
```

Notice the feedback arrow at the bottom. In AI-Enhanced mode, the output is consumed and discarded. In AI-Native mode, **the output feeds back into the knowledge base**, making the next cycle smarter. This feedback loop is what separates the two models — and it is the subject of Section 3.

---

## The 4-Layer Architecture: What to Build

**If you extract the common structure from Karpathy's personal system and Dorsey's organizational redesign, four layers emerge.** They are not a theoretical framework — they are a description of what both experiments actually built, whether they named the layers or not.

```
┌─────────────────────────────────────────────────┐
│  Layer 4: Human-in-the-Loop                     │
│  Steer direction · Handle exceptions · Verify   │
├─────────────────────────────────────────────────┤
│  Layer 3: Agent Layer                           │
│  Domain-specific autonomous execution           │
├─────────────────────────────────────────────────┤
│  Layer 2: Intelligence Layer          ← core    │
│  Compile, index, and query org knowledge        │
├─────────────────────────────────────────────────┤
│  Layer 1: Data Foundation                       │
│  Collect, normalize, and preserve all sources   │
└─────────────────────────────────────────────────┘
        ▲ dependencies flow upward — each layer
          requires the one below it to function
```

Each layer answers one question. If you cannot answer "yes" at a given layer, the layers above it will not work properly.

| Layer | The question to ask | Karpathy's answer | Dorsey's answer |
|---|---|---|---|
| **1. Data** | Can AI read all our organizational knowledge? | `raw/` folder — originals preserved as-is | All Slack, email, code, docs unified |
| **2. Intelligence** | Can AI understand our full context and answer queries? | `wiki/` + `_index.md` + Lint & Heal | "Queryable company intelligence layer" |
| **3. Agents** | Does AI autonomously execute repeatable work? | Q&A agent, slide generator, plot generator | ICs amplified 10x via AI agents |
| **4. Human** | Do people steer rather than execute? | Ask questions; LLM writes the wiki | IC / DRI / Player-Coach roles |

### Why the Order Matters

**The dependency direction is what makes or breaks the transition.**

- Layer 1 missing → Layer 2 hallucinates (AI answers without organizational context)
- Layer 2 missing → Layer 3 automates blindly (fast execution, but in the wrong direction)
- Most failures follow the same pattern: **jumping straight to Layer 3**

This is the most common mistake. A team connects an LLM to their Slack API, builds an agent, and launches it — without first ensuring the organization's knowledge is collected (Layer 1) or structured into something the agent can query (Layer 2). The agent has no context. It hallucinates. The team concludes "AI isn't ready yet."

**The agent was fine. The foundation was missing.**

Karpathy's system works not because his Q&A agent is exceptional, but because `wiki/` and `_index.md` are meticulously organized. The agent is simple — it reads the index, identifies relevant files, reads them, and synthesizes an answer. The intelligence comes from the knowledge layer, not the agent.

### Starting Points Differ by Scale

The right entry point depends on organizational size:

|                      | Startups (< 50)         | Enterprises (500+)          |
|----------------------|-------------------------|-----------------------------|
| **Layer 1 difficulty** | Low — few tools, small data | High — silos, legacy, permissions |
| **Layer 2 approach**   | Non-RAG (hundreds of docs suffice) | RAG + knowledge graph likely needed |
| **Biggest obstacle**   | "We don't have enough data yet" | Process inertia + organizational politics |
| **Where to start**     | Layer 1–2 simultaneously | Layer 1 pilot in one team → expand |

For startups, the barrier is often a misconception. *"We're only 20 people — what data do we even have?"* The answer: Slack history, Git repos, meeting notes, customer conversations, design docs. It is already there. It is just not in a form AI can consume.

For enterprises, the barrier is structural. Data lives in siloed systems with different permissions, formats, and owners. Starting with one team — one Layer 1 pilot — and expanding from demonstrated results is more realistic than a company-wide transformation.

---

## The Compound Loop: The Real Moat

**The real value of an AI-native system is not cost reduction. It is that the system gets smarter with every use.** This compound effect is what creates a durable advantage — and it is the least understood part of the AI-native model.

> "Every answer makes the wiki smarter."
> — Karpathy's core design principle

Here is how the loop works:

```
┌─────────┐    ┌─────────────┐    ┌──────────┐
│  Query  │──→─│ Intelligence │──→─│  Answer  │
└─────────┘    │   Layer     │    └────┬─────┘
               └──────┬──────┘         │
                      │           ┌────▼─────┐
                      │           │ Worth     │
                      │           │ keeping?  │
                      │           └────┬──────┘
                      │      yes ──────▼
                 ┌────┴─────┐    ┌──────────┐
                 │ Richer   │◀───│ Re-absorb │
                 │ knowledge│    │ into base │
                 └──────────┘    └──────────┘
```

In Karpathy's system, when a Q&A answer synthesizes insights across multiple documents, the system evaluates whether the answer is worth keeping. If it connects ideas in a novel way, it gets re-absorbed into the wiki — becoming source material for future queries. The wiki does not just store information. **It grows through use.**

This is the fundamental difference between AI-native and conventional tools:

**Cost-cutting is a one-time event.** You reduce 10 people to 5, and you are done. The saving is fixed.

**Compound knowledge is an ongoing process.** The system at month 6 is meaningfully smarter than the system at month 1. The gap between an organization running this loop and one that is not widens over time.

Dorsey's claim that "100 people + AI = 1,000 people" only holds if this compound loop is actually running. Without it, you just have fewer people doing the same work with slightly better tools. The multiplier comes from accumulated organizational intelligence, not from headcount arithmetic.

**Signs the compound loop is *not* working:**

- Six months after adopting AI tools, output quality is the same as month one
- Team members keep asking AI the same questions — answers are not accumulating
- Generated artifacts are consumed once and never referenced again
- New employees get no benefit from questions that previous employees already asked

If these symptoms sound familiar, the issue is probably not the AI model. It is the absence of a re-absorption mechanism — Layer 2 is not capturing the value that Layer 3 creates.

---

## The Direction: Bottom-Up, Not Top-Down

**Most AI-native attempts fail because they start in the wrong place.** The instinct is to start with the most visible layer — deploy an agent, automate a workflow, demo something impressive. But without Layers 1 and 2 underneath, the agent has nothing meaningful to work with.

```
Common failure pattern:
  Layer 3 (Agents)        ← start here    ❌ no context → hallucination
  Layer 2 (Intelligence)                   ⬜ skipped
  Layer 1 (Data)                           ⬜ skipped

What actually works:
  Layer 1 (Data)          ← start here     ✓ collect and normalize
    ↓
  Layer 2 (Intelligence)  ← then here      ✓ compile and index
    ↓
  Layer 3 (Agents)        ← then here      ✓ automate with context
    ↓
  Layer 4 (Human roles)   ← finally        ✓ redefine how people work
```

**One principle above all: knowledge before agents.**

Karpathy's Q&A agent works because `wiki/` and `_index.md` are well-organized. Dorsey's "queryable company intelligence layer" says the same thing in corporate language. The precondition for useful agents is queryable knowledge. Without it, agents are just faster ways to produce wrong answers.

This does not mean the transition needs to be slow. It means the *sequence* matters. Layer 1 can be as simple as consolidating Slack exports, Git repos, and meeting transcripts into a single searchable location. Layer 2 can start with an LLM that reads an index file and retrieves relevant documents — no vector database required at small scale.

*How long each step takes depends entirely on your organization's size, data maturity, and domain complexity. What matters is not speed — it is sequence.*

---

## Closing: Intelligence Is a Structure, Not a Tool

**Dorsey's** most valuable insight: treat the company as a single intelligence, not a hierarchy of people.

**Karpathy's** most valuable insight: let AI compile the knowledge, and let every answer compound.

The intersection of these two ideas points to a clear direction:

**AI-native is not about which AI product you adopt. It is about building a structure where organizational knowledge is compiled by AI, queried by AI, and accumulated through use.**

Get this structure right, and it works regardless of which model you use or which agent framework you deploy. Models will keep changing — the state of the art six months from now will be different from today. But a well-organized knowledge layer endures. It is the asset that compounds.

The question is not whether AI will reshape how organizations operate. It is whether your organization's knowledge is in a form where AI can participate — or whether it is still locked in slide decks, email threads, and people's heads.
