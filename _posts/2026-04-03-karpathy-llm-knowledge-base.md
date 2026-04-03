---
title: "How Karpathy Turned an LLM into a Self-Improving Research Wiki — And Why RAG Wasn't Needed"
date: 2026-04-03 10:00:00 +0900
categories: [AI & ML, AI in Practice]
tags: [llm, knowledge-base, rag, karpathy, compile, obsidian, wiki]
---

*Dissecting the architecture of a knowledge system where the LLM reads, writes, indexes, lints, and heals*

---

> "Something I'm finding very useful recently: using LLMs to build personal knowledge bases for various topics of research interest. A large fraction of my recent token throughput is going less into manipulating code, and more into manipulating knowledge."
> — Andrej Karpathy

Karpathy recently shared the architecture of his personal knowledge system. The setup: drop raw sources — papers, articles, repos, datasets — into a folder, and an LLM automatically compiles them into an indexed Markdown wiki. From there, the LLM answers complex questions, generates slide decks and plots, and periodically runs health checks to find gaps and fix inconsistencies.

**What makes this system worth studying is not a novel technology stack. It is the deliberate simplicity.** No vector database. No RAG pipeline. No embedding model. Just plain Markdown files, a maintained index, and an LLM that does not merely *read* — it *writes, indexes, lints, and heals* the entire knowledge base.

This post dissects the architecture, extracts the design principles behind each decision, and compares the Compile approach with traditional RAG to clarify when each is the right choice.

---

## The Architecture: Five Steps, No Magic

Karpathy's system breaks into a five-step main pipeline and three supporting tools. Each component is simple on its own. The power comes from how they compose.

```
MAIN PIPELINE

┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Step 1  │    │  Step 2  │    │  Step 3  │    │  Step 4  │    │  Step 5  │
│ SOURCES  │──→─│  raw/    │──→─│  WIKI    │──→─│ Q&A Agent│──→─│  OUTPUT  │
│          │    │          │    │          │    │          │    │          │
│ articles │    │ as-is    │    │ compiled │    │ complex  │    │ .md files│
│ papers   │    │ .md +    │    │ summaries│    │ questions│    │ Marp     │
│ repos    │    │ images   │    │ backlinks│    │ against  │    │ slides   │
│ datasets │    │ local    │    │ concepts │    │ full wiki│    │ matplotlib│
│ images   │    │ storage  │    │_index.md │    │ no RAG   │    │ plots    │
└──────────┘    └──────────┘    └──────────┘    └────┬─────┘    └──────────┘
                                     ▲                │
                                     └── re-absorb ───┘
                                     (valuable answers filed back)

SUPPORT LAYER

┌────────────┐  ┌────────────┐  ┌────────────┐
│  Obsidian  │  │ Lint+Heal  │  │ CLI Tools  │
│ IDE for    │  │ find gaps  │  │ search     │
│ viewing    │  │ fix issues │  │ web UI     │
│ raw + wiki │  │ web search │  │ CLI for LLM│
└────────────┘  └────────────┘  └────────────┘
```

Here is what happens at each step:

- **Sources → raw/**: Papers, articles, and repos are converted to Markdown via Obsidian's Web Clipper and stored in `raw/` as-is. **Originals are never modified** — this preserves the option to recompile everything when a better model arrives.
- **raw/ → Wiki**: The LLM reads `raw/` and **compiles** `wiki/` — writing summaries, creating backlinks, categorizing concepts, and auto-maintaining an `_index.md` file. **Humans never edit the wiki directly.**
- **Wiki → Q&A**: The LLM reads `_index.md` to identify relevant files, opens them, and synthesizes an answer. At ~100 documents and ~400K words, this works **without a vector database**.
- **Q&A → Output**: Answers are rendered not as terminal text but as **Marp slide decks, matplotlib charts, or Markdown reports** — all viewable in Obsidian.
- **Re-absorption loop**: When a Q&A answer synthesizes ideas across multiple documents in a novel way, it gets filed back into the wiki. The knowledge base grows through use.

**In this entire pipeline, the human does exactly two things:**

1. Drop sources into `raw/`
2. Ask questions

Everything else — compilation, indexing, linking, linting, output generation — is the LLM's job.

---

## Three Design Principles That Make It Work

The architectural choices above are not accidental. Three principles run through the entire system.

### Principle 1: Non-RAG by Design

**Karpathy deliberately chose not to use RAG.** Instead, the LLM auto-maintains `_index.md` — a table of contents with one-line summaries of every document — and uses it as a navigation map to find and open relevant files directly.

> "Rather than using fancy RAG, the LLM auto-maintains index files and brief summaries of documents and reads important related data fairly easily at this small scale."

Why this works at his scale:

- **`_index.md` provides a map of the entire wiki.** The LLM reads it once and knows which files to open — no embedding similarity search needed.
- **~100 documents fit comfortably.** The index plus a few selected files fit within modern context windows.
- **Transparency.** You can open `_index.md` and see exactly what the LLM is referencing. RAG's embedding similarity scores offer no equivalent visibility.

**Non-RAG is not "RAG not yet built." It is a better choice at this scale.** The detailed comparison with RAG — and when RAG becomes necessary — follows in Section 4.

### Principle 2: LLM as Sole Author

**The LLM is the only writer of the wiki.** The human curates what goes into `raw/` and asks questions. The LLM handles everything in between: summarizing, linking, categorizing, formatting.

This solves three chronic problems of human-maintained knowledge bases:

- **Knowledge rot**: Human-written wikis decay because nobody maintains them. An LLM-maintained wiki gets refreshed on every compilation cycle.
- **Inconsistency**: Multiple human authors produce inconsistent structure and style. The LLM follows prompt instructions consistently.
- **Missed connections**: Humans only link what they already know is related. The LLM reads the entire wiki and discovers connections humans would miss.

**The human is the content curator (what goes in). The LLM is the content compiler (how it is structured).**

### Principle 3: Compound Through Re-absorption

**Q&A results that synthesize knowledge across documents are filed back into the wiki.** The system gets richer with every use.

```
Query ──→ Wiki search ──→ Answer
                             │
                        Worth keeping?
                          │      │
                         yes     no → reply only
                          │
                   Re-absorb into wiki
                          │
                   Next query draws on
                   richer knowledge
```

This is fundamentally different from conventional tools. You can pile documents into Google Docs — search does not improve. You can add pages to Notion — existing pages do not get updated. **In this system, usage itself improves the system.**

---

## What Lint & Heal Reveals About the Long Game

**Most knowledge systems peak the day they are created and degrade from there.** Karpathy's system does the opposite — it maintains and improves itself over time.

> "I run LLM 'health checks' over the wiki to find inconsistent data, impute missing data with web searches, and find interesting connections for new article candidates."

The maintenance loop has two phases:

- **Lint (detect)**: Find inconsistent data across articles, identify broken links, flag orphaned notes that nothing references.
- **Heal (repair)**: Fill information gaps using **web search**, suggest new connections between articles, propose new article candidates for emerging topics.

Compare this to how traditional knowledge tools age:

|                    | Traditional Wiki (Notion, Confluence) | Karpathy's LLM Wiki     |
|--------------------|---------------------------------------|--------------------------|
| **Who writes**     | Humans                                | LLM                      |
| **Who maintains**  | Humans (in theory)                    | LLM (Lint & Heal)        |
| **Quality over time** | Degrades (knowledge rot)           | Improves (compound loop) |
| **Gap detection**  | Manual review or none                 | Automated health checks  |
| **Gap filling**    | Requires human effort                 | Web search + LLM imputation |
| **New connections**| Only what humans notice               | LLM discovers across all docs |

**Lint & Heal is what transforms this from a "tool" into a self-maintaining knowledge organism.** Even when the human is not actively using it, the system resists decay and continues to grow.

---

## RAG vs Compile: Two Paradigms of Knowledge Retrieval

**Karpathy's Non-RAG approach and traditional RAG are not competing solutions to the same problem. They are different architectures that excel under different conditions.** The question is not which is "better" — it is when each one is the right choice.

### The Structural Difference

```
RAG pipeline:
┌────────┐  ┌─────────┐  ┌─────────┐  ┌────────┐  ┌────────┐
│  Docs  │─→│ Chunk + │─→│Embedding│─→│Vector  │  │  LLM   │
│        │  │  split  │  │ model   │  │  DB    │  │        │
└────────┘  └─────────┘  └─────────┘  └───┬────┘  └───┬────┘
                                          │            │
                           Query ─ embed ─┤  top-k ──→─┤─→ Answer
                                          └────────────┘

Compile pipeline (Karpathy):
┌────────┐  ┌─────────┐  ┌─────────┐  ┌────────┐
│  raw/  │─→│  LLM    │─→│  wiki/  │  │  LLM   │
│ sources│  │ compile │  │_index.md│  │        │
└────────┘  └─────────┘  └───┬─────┘  └───┬────┘
                              │            │
               Query ────────→┤  read ───→─┤─→ Answer
                              │ index+files│      │
                              └────────────┘      │
                                   ▲              │
                                   └─ re-absorb ──┘
```

|                      | RAG                              | Compile (Karpathy)               |
|----------------------|----------------------------------|----------------------------------|
| **Core operation**   | Retrieve chunks by similarity    | Compile structured knowledge     |
| **Knowledge format** | Raw chunks + embeddings          | LLM-written summaries + index    |
| **Retrieval method** | Vector similarity search         | LLM reads index, picks files     |
| **Infrastructure**   | Embedding model + Vector DB      | Plain Markdown files             |
| **Pre-processing**   | Chunk + embed (one-time)         | LLM compile (incremental)        |
| **Transparency**     | Similarity scores (opaque)       | `_index.md` (human-readable)     |
| **Knowledge quality**| Same as source (no synthesis)    | LLM-synthesized summaries + links|
| **Maintenance**      | Re-embed on update, drift mgmt   | LLM Lint & Heal                  |
| **Feedback loop**    | None (static index)              | Re-absorption (answers enrich wiki) |

**The fundamental difference is not the retrieval method — it is what happens to knowledge *before* the query arrives.**

RAG stores raw chunks and retrieves them at query time. The LLM sees fragments. Compile processes raw sources into structured knowledge *before* any query is asked — the LLM has already read, summarized, linked, and indexed everything. The query hits pre-digested knowledge.

### When Compile Wins

**The following conditions favor the Compile approach over RAG:**

- **Scale under ~500 documents**: `_index.md` plus selected files fit in a context window. Vector DB adds complexity with no benefit.
- **Synthesis matters more than lookup**: RAG retrieves relevant chunks but does not connect them. Compile delivers pre-connected, cross-referenced knowledge.
- **Transparency is required**: Open `_index.md` and you see exactly what the LLM references. Embedding similarity scores are opaque.
- **Knowledge accumulates over time**: The re-absorption loop makes the system smarter with use. RAG indexes are static.
- **Infrastructure simplicity**: No database, no embedding pipeline. Markdown files and an LLM.

### When RAG Becomes Necessary

**When any of the following conditions apply, RAG is the right tool — or a necessary complement:**

| Scenario                          | Why Compile alone falls short           | What RAG provides                    |
|-----------------------------------|-----------------------------------------|--------------------------------------|
| **1,000+ documents**              | `_index.md` exceeds context window      | Scalable vector search, no context limit |
| **Real-time data streams**        | Compile is batch, not streaming         | Incremental embedding on ingest      |
| **Exact chunk attribution**       | Wiki summaries abstract away source     | Direct chunk-to-source traceability  |
| **Multi-team access control**     | Flat wiki, no permission model          | Per-document ACL in retrieval layer  |
| **Regulatory / compliance**       | Need to cite exact source passages      | Chunk-level provenance tracking      |
| **Heterogeneous media at scale**  | Compiling 10K PDFs is token-expensive   | Embed once, retrieve many times      |
| **Low-latency, high-throughput**  | LLM reads files per query (token-heavy) | Pre-computed embeddings, fast lookup |

**The clearest trigger is scale.** When `_index.md` — the table of contents with one-line summaries — no longer fits in a single context window, the Compile-only approach breaks down. At ~100 documents this is comfortable. At ~500 it gets tight. At 1,000+ you need a retrieval layer that operates outside the context window — and that is exactly what RAG was designed for.

### The Hybrid: Compile + RAG

**This is not an either/or choice. The two approaches can layer.**

```
┌─────────────────────────────────────────────────┐
│            Compile Layer (Karpathy)              │
│                                                 │
│  raw/ ──→ LLM compile ──→ wiki/ + _index.md     │
│  (core knowledge, ~100s docs, actively curated) │
│                                                 │
│       Q&A: LLM reads index + files              │
│       Re-absorption: answers → wiki             │
├─────────────────────────────────────────────────┤
│            RAG Layer (fallback)                  │
│                                                 │
│  archive/ ──→ chunk + embed ──→ Vector DB       │
│  (long-tail, 1000s+ docs, infrequently accessed)│
│                                                 │
│       Q&A: similarity search when               │
│       wiki doesn't have the answer              │
└─────────────────────────────────────────────────┘
```

Three patterns where hybrid makes sense:

- **Hot/Cold separation**: Actively referenced core knowledge lives in the Compile layer (wiki). Infrequently accessed archives live in RAG.
- **Wiki-first, RAG-fallback**: Query the wiki first. If the answer is insufficient, fall back to RAG. When RAG surfaces something valuable, re-absorb it into the wiki.
- **Compile for synthesis, RAG for lookup**: Cross-document analysis goes through the wiki. Pinpointing a specific passage or fact goes through RAG.

**Karpathy did not skip RAG because RAG is bad. He skipped it because his scale did not require it.** As scale grows, RAG becomes not a replacement but a complement — and the Compile layer remains the core where knowledge is synthesized, connected, and accumulated.

---

## Limitations and Open Questions

**Every architecture has a scope where it thrives and boundaries where it struggles.** The Compile pattern is no exception.

- **LLM dependency**: Compilation quality varies across models. When you switch models, the wiki may compile differently. The `raw/` preservation policy mitigates this — you can always recompile — but the cost and quality variance of full recompilation remain open.
- **Multi-user scaling**: As a solo research wiki, this system is excellent. Scaling to a team introduces edit conflicts, divergent perspectives, and access control — none of which the current architecture addresses.
- **Factual accuracy**: LLM-compiled summaries may subtly diverge from the source. Can Lint catch this systematically? At what point does human verification become necessary?
- **Looking ahead**: Karpathy himself hinted at the next frontier — fine-tuning an LLM on the wiki data itself. "Knowledge in weights, not just context." If realized, this would transcend the context window limitation entirely.

**These open questions do not invalidate the system.** They define where it currently works well — solo or small-team, research-oriented, hundreds of documents — and where expansion requires additional design decisions.

---

## Closing: The Shift from Retrieve to Compile

**RAG asks**: "Given a query, which chunks are most similar?"

**Compile asks**: "Given all sources, what is the structured knowledge?"

These are different questions, and they produce different kinds of systems. RAG finds fragments. Compile builds structure.

**The deeper lesson is not about Obsidian or Markdown or any specific tool. It is about the role of the LLM shifting from reader to writer — from answering questions about documents to compiling documents from raw knowledge.**

At small scale, Compile alone is sufficient — and simpler, more transparent, and self-improving. At larger scale, RAG becomes a necessary complement. But even then, the Compile layer remains the core: the place where knowledge is synthesized, connected, and accumulated.

When the LLM writes, indexes, lints, and heals the knowledge base, the human is freed to do what humans do best: decide what questions are worth asking.
