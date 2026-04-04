---
title: "Synthesis AI Part 1: The Synthesis Bottleneck — Why \"Make\" Lags Behind"
date: 2026-04-04 09:00:00 +0900
categories: [Drug Discovery, Small Molecule Design]
tags: [synthesis, drug-discovery, dmta, ai-chemistry, bottleneck]
math: true
---

**AI-Driven Synthesis in Drug Discovery**

*This is Part 1 of a 5-part series on AI-driven synthesis in drug discovery.*

- **Part 1** (this post): The Synthesis Bottleneck — Why "Make" Lags Behind
- **Part 2:** [Reaction Prediction — Can AI Predict What Chemistry Will Do?](/posts/synthesis-ai-part2-reaction-prediction/)
- **Part 3:** [Retrosynthesis — Can AI Plan How to Make a Molecule?](/posts/synthesis-ai-part3-retrosynthesis/)
- **Part 4:** [Synthesis-Aware Design — Making AI-Generated Molecules Makeable](/posts/synthesis-ai-part4-synthesis-aware-design/)
- **Part 5:** [From Algorithm to Lab — CRO Integration and the Remaining Gap](/posts/synthesis-ai-part5-algorithm-to-lab/)


---

## 1. Introduction: The Amdahl's Law of Drug Discovery

Drug discovery runs on cycles. We design a molecule, make it, test it, learn from the results, and design the next one. This **Design-Make-Test-Analysis (DMTA)** loop is the heartbeat of every drug program — a typical small-molecule lead optimization campaign runs 10-15 full cycles per year. AI has compressed the Design step by orders of magnitude. But a strange asymmetry has emerged.

**The overall cycle time is governed by its slowest stage, not its fastest — and Make is overwhelmingly the slowest.**

Consider the arithmetic, which we explored in detail in the companion essay "Toward Accelerating the Entire Design-Make-Test Cycle with AI":

```
  Typical DMTA cycle (small molecule lead optimization):

    Design:  2 days      (med chem ideation + computational scoring)
    Make:    21 days      (synthesis, purification, QC)
    Test:    10 days      (biochemical + cellular + Tier 1 ADME panel)
    Learn:   2 days       (data analysis, team review)
    ─────────────────
    Total:   35 days      (~ 5 weeks)

  After 100x Design acceleration (AI generative models):
    Design:  0.02 days    (minutes, not days)
    Make:    21 days       (unchanged)
    Test:    10 days       (unchanged)
    Learn:   2 days        (unchanged)
    ─────────────────
    Total:   33.02 days   → 5.7% improvement

  After 2x Make + Test acceleration (automation + prediction):
    Design:  2 days        (unchanged)
    Make:    10.5 days     (route pre-validation, robotic synthesis)
    Test:    5 days        (active learning, focused assay panel)
    Learn:   2 days        (unchanged)
    ─────────────────
    Total:   19.5 days    → 44% improvement
```

AstraZeneca — the pharma company that has published most extensively on DMTA optimization — reports that a full cycle takes **4-6 weeks**, with synthesis alone consuming **3-6 weeks** per round (Thakkar et al., 2021, Chem. Sci.). Design, by contrast, can now be completed in hours with modern generative models.

This is Amdahl's law applied to chemistry: **optimizing the fast component of a sequential process has negligible impact when the slow component dominates.**

This series asks what it would take to break that bottleneck. Across five parts, we will trace the landscape of AI-driven synthesis — from predicting what a reaction produces, to planning multi-step routes backward from a target molecule, to generating only molecules that can actually be made, and finally to executing AI-proposed routes in physical laboratories.

- **Part 1** (this post): Why Make resists AI — the structural barriers and data landscape
- **Part 2**: Reaction prediction — "What forms when these reactants meet?"
- **Part 3**: Retrosynthesis — "How do we build this target molecule?"
- **Part 4**: Synthesis-aware design — "Can we generate only makeable molecules?"
- **Part 5**: Lab execution — "Does the AI-proposed route actually work?"

Here is the DMTA cycle and where each part of this series intervenes:

```
       ┌──────────┐      ┌──────────┐      ┌──────────┐      ┌──────────┐
       │  DESIGN  │─────▶│   MAKE   │─────▶│   TEST   │─────▶│  LEARN   │
       │          │      │          │      │          │      │          │
       │ propose  │      │ synth /  │      │ assay /  │      │ analyze  │
       │ molecule │      │ purify   │      │ screen   │      │ update   │
       └──────────┘      └──────────┘      └──────────┘      └────┬─────┘
            ▲                  │                                   │
            │            ┌─────┴──────┐                           │
            │            │ THIS SERIES│                           │
            │            │ Parts 2-5  │                           │
            │            └────────────┘                           │
            └─────────────────────────────────────────────────────┘
```

---

## 2. Why Chemistry Resists AI: Three Structural Barriers

AI has transformed molecular design. Structure prediction runs in seconds. Generative models propose candidates by the thousands. Docking evaluates binding in milliseconds. Why, then, does the Make step remain so stubbornly resistant?

**The answer lies in three structural barriers that make chemical synthesis fundamentally harder for AI than molecular design.**

These are not temporary gaps waiting for more data or bigger models. They reflect deep properties of chemistry itself.

---

### Barrier 1: Combinatorial Explosion of Conditions

A chemical reaction is not determined by its reactants alone. Consider a Suzuki coupling — the reactants (aryl halide + boronic acid) are only the beginning. The outcome depends on:

- **Catalyst**: Pd(PPh3)4 vs. Pd(dppf)Cl2 vs. Pd(OAc)2/XPhos — each favors different substrates
- **Base**: K2CO3 vs. Cs2CO3 vs. K3PO4 — affects transmetalation rate
- **Solvent**: DMF vs. THF vs. dioxane/water — governs solubility and reaction pathway
- **Temperature**: 60 C vs. 100 C vs. microwave 150 C
- **Concentration**: 0.1 M vs. 0.5 M — dilution effects on selectivity
- **Atmosphere**: air vs. N2 vs. Ar — oxygen sensitivity
- **Time**: 2 hours vs. 24 hours vs. 3 days

**The same pair of reactants can give completely different products — or no product at all — under different conditions.** A textbook example: the reaction of an enolizable ketone with an aldehyde can give an aldol product (kinetic control, low temperature, LDA) or an elimination product (thermodynamic control, high temperature, NaOH). Same reactants, different conditions, different molecules.

The combinatorial space is enormous. Even discretizing each variable coarsely — say, 10 common solvents, 5 catalysts, 5 bases, 5 temperature ranges, 3 concentration regimes, 2 atmospheres — gives:

> 10 x 5 x 5 x 5 x 3 x 2 = **7,500 condition combinations per reaction**

For a structure prediction model, the input is a sequence and the output is a structure — a well-defined mapping. For a reaction prediction model, the input must somehow encode not just the molecular graphs of reactants but the entire physicochemical context. This is a fundamentally higher-dimensional prediction problem.

---

### Barrier 2: Exception-Rich Rules

Organic chemistry is taught through rules: Markovnikov's rule, Baldwin's rules for ring closure, Bredt's rule, Woodward-Hoffmann rules. These rules are genuinely useful — they capture real thermodynamic and kinetic tendencies.

**But every rule in organic chemistry has exceptions that are almost as well-known as the rule itself.**

Consider a few:

| Rule | What It Says | Famous Exceptions |
|------|-------------|-------------------|
| Markovnikov's rule | HX adds to the more substituted carbon of an alkene | Anti-Markovnikov addition with peroxides (Kharasch, 1933) |
| Baldwin's rules | Certain ring closure modes are kinetically disfavored | 4-exo-dig and 5-endo-dig cyclizations that violate the rules but proceed readily |
| Bredt's rule | No double bond at bridgehead of small bicyclic systems | Stable bridgehead olefins in large rings (Wiseman, 1967) |
| Woodward-Hoffmann | Thermal [2+2] cycloadditions are forbidden | Ketene [2+2] cycloadditions proceed thermally |

This creates a dilemma for AI:

- **Physics-based simulation (QM/DFT)** can in principle handle all of these correctly by computing transition state energies. But a single transition state calculation takes minutes to hours at DFT level, and days at coupled-cluster level. Running QM on every proposed reaction step in a multi-step synthesis is computationally prohibitive.
- **Data-driven approaches** (neural networks trained on reaction databases) are fast — milliseconds per prediction — but they learn statistical patterns, not physical laws. They perform well on reactions similar to their training data and degrade unpredictably on novel substrates or unusual reaction types.

**The result is a fundamental tension: physics is accurate but slow; data is fast but brittle.** In protein structure prediction, end-to-end differentiable models (AlphaFold2, Boltz-1) have unified physics and data into a single framework. No comparable unification exists yet for reaction prediction.

---

### Barrier 3: Negative Data Scarcity

Perhaps the most insidious barrier is what we do not see in the data. When a chemist tries a reaction and it fails — no product, wrong product, decomposition — that experiment almost never enters any database.

**Failed reactions are the dark matter of chemistry: they shape the landscape but are nearly invisible to data-driven models.**

The reasons are structural:

- **Publication bias**: Journals publish successful reactions. A paper titled "A Highly Efficient Palladium-Catalyzed Cross-Coupling" will be accepted; "Fourteen Conditions Under Which This Coupling Failed" will not.
- **Electronic lab notebooks (ELNs)** capture failures within organizations, but this data is proprietary, unstructured, and rarely shared.
- **Patent data** (the basis of USPTO) reports optimized procedures, not the dozens of failed attempts that preceded them.

A model trained exclusively on successful reactions learns "what works" but has no signal for "what doesn't work." This is like training a medical diagnosis model only on patients who survived.

Here is the positive/negative ratio in commonly used datasets:

| Dataset | Approximate Size | Positive Reactions | Negative/Failed Reactions | Pos:Neg Ratio |
|---------|-----------------|-------------------|--------------------------|---------------|
| USPTO | ~3.5M reactions | ~3.5M | ~0 | ~inf:1 |
| Reaxys | ~130M reactions | ~130M | minimal (some low-yield) | >>100:1 |
| Pistachio | ~16M reactions | ~16M | ~0 | ~inf:1 |
| ORD | ~1M reactions | ~900K | ~100K (includes null results) | ~9:1 |
| Typical pharma ELN | varies | ~40-60% | ~40-60% | ~1:1 |

The contrast between public datasets (nearly 100% positive) and actual lab experience (roughly 50/50 success/failure) is striking. The Open Reaction Database (ORD) is the first major effort to systematically include negative results, but it remains small compared to the positive-only giants.

**Without negative data, models cannot learn decision boundaries — they can only interpolate within the space of known successes.**

---

## 3. The Data Landscape: What AI Models Learn From

Every AI model for chemical synthesis is shaped by the data it trains on. The reaction databases available today vary enormously in size, quality, coverage, and accessibility. Understanding their strengths and limitations is essential for interpreting what any synthesis AI model can — and cannot — do.

---

### 3.1 The Major Databases

**USPTO (United States Patent and Trademark Office)**

The workhorse of academic reaction prediction research. Daniel Lowe's 2012 extraction of ~3.5 million reactions from US patents (Lowe, 2012, PhD Thesis, Cambridge) created the most widely used public benchmark.

- **Strengths**: Large, public, free, well-established benchmarks (USPTO-50K, USPTO-MIT, USPTO-STEREO)
- **Weaknesses**: Noisy automated extraction from patent text. Condition data sparse. Biased toward industrial reaction types (cross-couplings, amide formations, reductions).

**Reaxys (Elsevier)**

The gold standard for reaction data in the pharmaceutical industry. Approximately 130 million reactions manually curated from the scientific literature since the 1800s.

- **Strengths**: Massive coverage, expert curation, structured condition data for many entries, spans two centuries
- **Weaknesses**: Commercial — expensive institutional license. Not available for ML training at scale.

**Pistachio (NextMove Software)**

A structured extraction of ~16 million reactions from patent literature, classified using the NextMove NameRxn system.

- **Strengths**: Cleaner than raw USPTO, reactions classified by type (~1,000 named reaction categories)
- **Weaknesses**: Commercial. Shares USPTO's patent-derived biases. Condition data partial.

**ORD (Open Reaction Database)**

A community effort launched in 2021 (Kearnes et al., 2021, JACS) to create a standardized, open repository including conditions, yields, analytical data, and crucially, negative results.

- **Strengths**: Standardized schema (Protocol Buffers), systematic conditions and outcomes, ML-ready, open-source
- **Weaknesses**: Still small (~1M reactions as of early 2026). Coverage patchy — depends on voluntary contributions.

**CAS SciFinder (American Chemical Society)**

The largest curated reaction database, maintained by the Chemical Abstracts Service since 1907.

- **Strengths**: Most comprehensive coverage, high curation quality, integrated substance and reference databases
- **Weaknesses**: Highly restricted access — no bulk download for ML training. Essentially unavailable for model development.

---

### 3.2 Database Comparison

| Feature | USPTO | Reaxys | Pistachio | ORD | CAS SciFinder |
|---------|-------|--------|-----------|-----|---------------|
| **Size** | ~3.5M | ~130M | ~16M | ~1M | ~200M+ |
| **Source** | US patents | Literature | Patents | Contributed | Literature + patents |
| **Condition data** | Sparse, noisy | Structured, partial | Partial | Standardized, systematic | Structured |
| **Yield data** | Rare | Partial (~30%) | Partial | Systematic | Partial |
| **Negative results** | No | Rare | No | Yes (by design) | No |
| **Reaction classification** | No | Yes | Yes (NameRxn) | Partial | Yes |
| **Access** | Public, free | Commercial | Commercial | Public, free | Restricted |
| **Noise level** | High | Low-moderate | Moderate | Low | Low |
| **ML-ready format** | SMILES/RXN | Requires extraction | SMILES/RXN | Protocol Buffers | Requires extraction |

**The critical observation: most datasets lack systematic reaction condition and yield data.** A model trained on USPTO can learn to predict "does A + B give product C?" with reasonable accuracy. But it cannot reliably answer "under what conditions?" or "in what yield?" — because that information was never consistently recorded in the training data.

---

### 3.3 The Conditions-and-Yield Gap

Consider what a medicinal chemist actually needs versus what current models provide:

| What the chemist asks | What the model can answer | What is missing |
|-----------------------|--------------------------|-----------------|
| "What product do I get from A + B?" | Major product prediction (good) | Minor products, side reactions |
| "What conditions should I use?" | Generic defaults from literature | Substrate-specific optimization |
| "What yield can I expect?" | Rough binary (works/doesn't) | Quantitative yield prediction |
| "Will this work on my substrate?" | Analogy to training data | OOD generalization confidence |
| "What could go wrong?" | Almost nothing | Failure mode prediction |

**The gap between "this reaction is known" and "this reaction will work in your lab, under these conditions, at this yield" is where most of the practical value lies — and where data is scarcest.**

The ORD is the most promising effort to close this gap — its schema captures inputs with quantities, conditions, outcomes, and crucially **null outcomes**. If the field can sustain and scale this effort, it will fundamentally change what synthesis AI models can learn. But for now, the landscape remains dominated by large, positive-only, condition-sparse databases.

---

## 4. The Landscape of AI Synthesis: A Roadmap

The barriers described above — combinatorial conditions, exception-rich rules, missing negative data — are not problems that a single model can solve. They require a layered approach, where different AI capabilities build on one another.

**This series maps four layers of AI-driven synthesis, each addressing a distinct question and depending on the layers below it.**

```
  ┌─────────────────────────────────────────────────────────────┐
  │              AI-Driven Synthesis Landscape                  │
  │                                                             │
  │   Part 2                Part 3                Part 4        │
  │  ┌─────────────┐     ┌──────────────┐     ┌─────────────┐  │
  │  │  Reaction   │     │   Retro-     │     │ Synthesis-  │  │
  │  │ Prediction  │────▶│  synthesis   │────▶│   Aware     │  │
  │  │             │     │  Planning    │     │  Design     │  │
  │  │ "What       │     │ "How to     │     │ "Generate   │  │
  │  │  forms?"    │     │  build it?" │     │  makeable   │  │
  │  │             │     │             │     │  molecules" │  │
  │  └─────────────┘     └──────────────┘     └──────┬──────┘  │
  │        │                    │                     │         │
  │        │                    │                     ▼         │
  │        │                    │              ┌─────────────┐  │
  │        └────────────────────┴─────────────▶│    Lab      │  │
  │                                            │ Execution   │  │
  │                                            │ (Part 5)    │  │
  │                                            │ "Does it    │  │
  │                                            │  work?"     │  │
  │                                            └─────────────┘  │
  └─────────────────────────────────────────────────────────────┘
```

Here is what each layer addresses and why the dependencies matter:

**Part 2 — Reaction Prediction: "What forms?"**

Given reactants and (optionally) conditions, predict the product. This is the forward model — the atomic operation of computational chemistry. Every retrosynthesis engine needs one. Three paradigms compete: template-based, sequence-to-sequence, and graph-based.

- Core challenge: Generalizing beyond training data, especially to novel reaction types
- Key models: Molecular Transformer (Schwaller et al., 2019, ACS Cent. Sci.), Electron Flow Matching (Joung et al., 2025, Nature)

**Part 3 — Retrosynthesis Planning: "How to build it?"**

Given a target molecule, propose a synthetic route — a tree of reactions leading back to purchasable starting materials. Requires single-step retrosynthetic models and multi-step search algorithms (MCTS, A*, beam search).

- Core challenge: Evaluating route quality beyond step count — considering yield, cost, practicality
- Key tools: ASKCOS (Coley et al., MIT), AiZynthFinder (Thakkar et al., AstraZeneca), Synthia (Grzybowski, UNIST/Allchemy)
- **Depends on Part 2**: Each proposed retrosynthetic step must be validated by a forward model

**Part 4 — Synthesis-Aware Design: "Generate makeable molecules"**

Embed synthesis constraints directly into the generative model, rather than generating first and filtering after. This is where molecular design meets synthesis planning — and where connections to co-folding models and bespoke libraries become concrete.

- Core challenge: Balancing drug-likeness, target affinity, and synthetic accessibility simultaneously
- Key approaches: SynFlowNet (Cretu et al., 2024, Mila), SynFormer (Gao et al., 2025, PNAS), REINVENT + synthesis filters (AstraZeneca)
- **Depends on Part 3**: The generative model needs a retrosynthesis oracle to evaluate proposed molecules

**Part 5 — Lab Execution: "Does it work?"**

The ultimate test: can an AI-proposed route be executed in a physical lab? Covers automated synthesis platforms, CRO handoff, and the gap between computational proposals and bench reality.

- Core challenge: Translating algorithmic instructions into physical procedures that work on real equipment
- Key efforts: Chemify/XDL (Cronin, Glasgow), PostEra's AI-to-CRO pipeline, ORD data standardization
- **Depends on Parts 2-4**: Lab execution validates the entire upstream stack

---

## 5. Who Is Working on This: The Research Landscape

The field is structured around four research axes. Understanding who is working on what — and where axes intersect — provides the map we will navigate throughout this series.

---

### Axis 1: Reaction Understanding and Prediction

> Core question: "Can we accurately predict what a chemical reaction produces?"

| Research Group | Affiliation | Key Contributions | Direction |
|---------------|-------------|-------------------|-----------|
| **Philippe Schwaller** | EPFL | Molecular Transformer (2019), RXNMapper (2021), DRFP (2022), yield prediction | Building a "language model for reactions" — treating SMILES as natural language. Unified approach from atom mapping to fingerprints to yield |
| **Connor Coley** | MIT | Electron Flow Matching (Nature, 2025), reaction condition prediction | Mechanism-level prediction — modeling electron flow rather than just product SMILES. Bridging data-driven speed with QM-level understanding |
| **Frank Glorius** | U. Munster | Generalization benchmarks, out-of-distribution evaluation | Systematically testing whether models generalize to new reaction types. Designing evaluation protocols that expose overfitting to USPTO |

**Key tension in this axis**: Schwaller's approach treats reactions as a language problem (sequence-to-sequence translation), while Coley's Electron Flow Matching treats them as a physical process (electron redistribution). Both achieve strong benchmark results. The question is which approach degrades more gracefully on truly novel chemistry.

---

### Axis 2: Retrosynthesis Planning

> Core question: "Can we automatically design synthetic routes to target molecules?"

| Research Group | Affiliation | Key Contributions | Direction |
|---------------|-------------|-------------------|-----------|
| **Connor Coley** | MIT | ASKCOS platform, higher-level retro strategies (2026) | Moving beyond simple disconnections to strategy-level planning — protecting groups, functional group interconversions, convergent routes. "Think like a chemist" |
| **AstraZeneca** (Thakkar, Genheden et al.) | AstraZeneca | AiZynthFinder, RAscore, 3-year industrial deployment report | Industrial deployment at scale. Configurable expansion policies, stock management, integration with med chem workflows |
| **Bartosz Grzybowski** | UNIST / Allchemy | Synthia (Chematica), 50K+ expert-encoded rules | Expert knowledge + ML hybrid. Hand-coded rules for accuracy, ML for scalability. Has demonstrated total synthesis-level route planning |
| **Marwin Segler** | Microsoft Research | Early neural retrosynthesis (2017), Syntheseus framework (2024) | Unified benchmarking framework. Systematic comparison of retro models + search algorithms in all combinations |

**Key tension in this axis**: Rules vs. learning. Grzybowski's Synthia encodes 50,000+ expert rules and achieves remarkable accuracy on complex targets, but scaling requires ongoing expert curation. Coley's and AstraZeneca's ML-based approaches learn from data and scale easily, but struggle with rare or novel disconnections. Segler's Syntheseus framework enables fair comparison across these approaches.

---

### Axis 3: Synthesis-Aware Molecular Generation

> Core question: "Can generative models produce only molecules that are actually synthesizable?"

| Research Group | Affiliation | Key Contributions | Direction |
|---------------|-------------|-------------------|-----------|
| **Yoshua Bengio group** | Mila | GFlowNet framework, SynFlowNet (2024), RGFN (2024) | GFlowNet-based generation where the action space is defined by reactions and building blocks. Diversity + synthesizability by construction |
| **Connor Coley / Wenhao Gao** | MIT | SynFormer (PNAS, 2025), SynNet (2022), SCScore | Transformer-based synthetic pathway generation. SynFormer uses Transformer + diffusion for building block selection across local and global chemical space |
| **AstraZeneca** | AstraZeneca | REINVENT, Lib-INVENT (2022) | RL-based molecular generation with synthesis filters. Library-focused design for combinatorial chemistry. Industrial-scale deployment |
| **CGFlow group** | Various | CGFlow (2025) | Structure-based SynFlowNet with co-folding confidence as reward. First integration of synthesis-aware generation with protein-ligand co-folding |

**Key tension in this axis**: Two competing architectures are emerging:

- **GFlowNet-based** (Bengio/Mila): Actions = reactions + building blocks. Synthesizability is guaranteed by construction. Strong on diversity. Scales with available reaction templates.
- **Transformer-based** (Coley/MIT): SynFormer generates full synthetic pathways directly. Greater flexibility and scalability. Synthesizability is a learned property rather than a hard constraint.

Both approaches converge on the same principle: **synthesis constraints should be embedded at generation time, not applied as a post-hoc filter.**

---

### Axis 4: Lab Execution and Automation

> Core question: "Can AI-proposed synthesis be automatically executed in a physical lab?"

| Research Group / Company | Affiliation | Key Contributions | Direction |
|--------------------------|-------------|-------------------|-----------|
| **Lee Cronin** | U. Glasgow | Chemify platform, XDL (Chemical Description Language) | Programming language for synthesis. XDL makes procedures machine-readable and robot-executable. Demonstrated automated synthesis of complex molecules |
| **PostEra** | Startup (US) | COVID Moonshot, AI-to-CRO pipeline | Largest open-science AI synthesis experiment. Demonstrated end-to-end AI retrosynthesis to CRO execution at scale |
| **Insilico Medicine** | Startup (HK) | Chemistry42, Rentosertib (Phase IIa) | End-to-end validation from AI target identification through molecule generation to synthesis and clinical trials |
| **ORD consortium** | Multi-institution | Open Reaction Database, standardized reaction schema | Data infrastructure. Standardizing how reaction conditions, outcomes, and failures are recorded — enabling the next generation of condition-aware models |

**Key observation**: The bottleneck in this axis is not algorithms but standards and integration. Cronin's XDL, the ORD's data schema, and PostEra's CRO handoff experience address different pieces of the same infrastructure gap: how to translate computational proposals into physical experiments reproducibly.

---

### Cross-Axis Map

The four axes are not independent. Research groups and their outputs flow across boundaries:

```
               Axis 1: Reaction Prediction
              ┌─────────────────────────────┐
              │  Schwaller (EPFL)            │
              │  Coley (MIT) ──────────────────────┐
              │  Glorius (Munster)           │     │
              └──────────────┬──────────────┘     │
                             │ forward models      │
                             │ feed into            │
                             ▼                      │
               Axis 2: Retrosynthesis              │ Coley spans
              ┌─────────────────────────────┐     │ Axes 1-3
              │  Coley (MIT) ──────────────────────┤
              │  AstraZeneca                │     │
              │  Grzybowski (UNIST)         │     │
              │  Segler (Microsoft)         │     │
              └──────────────┬──────────────┘     │
                             │ route evaluation     │
                             │ enables               │
                             ▼                      │
               Axis 3: Synthesis-Aware Design      │
              ┌─────────────────────────────┐     │
              │  Bengio group (Mila)        │     │
              │  Coley / Gao (MIT) ────────────────┘
              │  AstraZeneca                │
              │  CGFlow group               │
              └──────────────┬──────────────┘
                             │ generates routes
                             │ that must be
                             ▼
               Axis 4: Lab Execution
              ┌─────────────────────────────┐
              │  Cronin (Glasgow)           │
              │  PostEra                    │
              │  Insilico Medicine          │
              │  ORD consortium             │
              └─────────────────────────────┘
```

Three observations stand out:

- **Connor Coley (MIT) spans Axes 1-3** — from Electron Flow Matching through ASKCOS to SynFormer. This breadth positions his group uniquely for vertically integrated synthesis AI.
- **AstraZeneca leads in industrial deployment** across Axes 2-3. Their 3-year AiZynthFinder retrospective (Genheden et al., 2024, J. Cheminform.) is the most detailed account of retrosynthesis AI in production.
- **The Mila/Bengio group introduced GFlowNet** — a framework that natively encodes synthesis constraints. SynFlowNet and RGFN shift the paradigm from "generate then filter" to "generate within constraints." CGFlow extends this to structure-based objectives, connecting to co-folding models traced in our Protein AI series.

---

## Looking Ahead

This post has established three claims:

1. **The synthesis bottleneck is the rate-limiting step of AI-driven drug discovery.** Accelerating Design further yields diminishing returns; the leverage is in Make.
2. **Chemistry resists AI for structural reasons** — combinatorial conditions, exception-rich rules, and the near-total absence of negative data — not merely for lack of effort.
3. **The data landscape is large but lopsided.** Millions of reactions are recorded, but systematic condition and yield data remain scarce, and failed reactions are nearly invisible.

These barriers are real, but they are not insurmountable. In the next four parts, we will examine the tools researchers are building to overcome them — layer by layer, from predicting individual reactions to executing full routes in automated laboratories.

In Part 2, we examine the first building block: **reaction prediction** — can AI reliably predict what happens when molecules meet reagents under specified conditions?

---

*Next: Part 2 — Reaction Prediction: Can AI Predict What Chemistry Will Do?*
