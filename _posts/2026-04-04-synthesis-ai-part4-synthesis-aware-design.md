---
title: "Synthesis AI Part 4: Synthesis-Aware Design — Making AI-Generated Molecules Makeable"
date: 2026-04-04 09:30:00 +0900
categories: [Drug Discovery, Small Molecule Design]
tags: [synthesis-aware-design, generative-model, synthesizability, co-folding, bespoke-library]
math: true
---

**AI-Driven Synthesis in Drug Discovery**

*This is Part 4 of a 5-part series on AI-driven synthesis in drug discovery.*

- **Part 1:** [The Synthesis Bottleneck — Why "Make" Lags Behind](/posts/synthesis-ai-part1-synthesis-bottleneck/)
- **Part 2:** [Reaction Prediction — Can AI Predict What Chemistry Will Do?](/posts/synthesis-ai-part2-reaction-prediction/)
- **Part 3:** [Retrosynthesis — Can AI Plan How to Make a Molecule?](/posts/synthesis-ai-part3-retrosynthesis/)
- **Part 4** (this post): Synthesis-Aware Design — Making AI-Generated Molecules Makeable
- **Part 5:** [From Algorithm to Lab — CRO Integration and the Remaining Gap](/posts/synthesis-ai-part5-algorithm-to-lab/)


---

## The Core Question

**Can we build generative models that only propose molecules we can actually synthesize — and tell us how?**

Parts 2 and 3 addressed two foundational capabilities: predicting what a reaction will produce (forward prediction) and figuring out how to build a target molecule (retrosynthesis). But in a real drug discovery campaign, we rarely start with a fixed target molecule. We start with a *target protein* and ask a generative model to propose molecules that bind it. The problem is that many of those proposals are synthetic dead ends.

This Part traces the evolution from crude synthesizability scores to models that generate molecules and their synthesis routes simultaneously — and examines how these approaches integrate with co-folding models and bespoke library strategies.

---

## 1. The Synthesizability Problem

Generative models for drug design have made extraordinary progress. REINVENT (Blaschke et al., J Chem Inf Model, 2020), diffusion-based models, and fragment-growing approaches can now propose molecules optimized for predicted binding affinity, selectivity, and drug-likeness. There is one persistent failure mode, however: **a significant fraction of AI-generated molecules cannot be synthesized in practice**.

The numbers are sobering:

- Typical SMILES-based generators: 30-60% of outputs flagged as "hard to synthesize" by experienced chemists
- Structure-based diffusion models: often propose strained ring systems, exotic heterocycles, or stereocenters that require prohibitively complex routes
- Even when a molecule looks reasonable on paper, the *route* to make it may require 15+ steps or unavailable reagents

Two strategies have emerged to address this:

- **Post-hoc filtering**: Generate freely, then score or filter for synthesizability
- **Generation-time integration**: Embed synthesizability constraints directly into the generative process

**The field has been shifting decisively from the first approach to the second.** This Part documents that transition.

---

## 2. Scoring Approaches: Post-Hoc Synthesizability Filters

The earliest and simplest approach is to let the generator propose whatever it wants, then apply a synthesizability score to filter or rank the outputs. Three scores dominate the literature.

### 2.1 SAScore: The Fragment Frequency Heuristic

**SAScore** (Ertl & Schuffenhauer, J Cheminform, 2009) computes synthetic accessibility from two components:

- **Fragment contribution**: How frequently the molecule's substructures appear in PubChem — common fragments are assumed easier to make
- **Complexity penalty**: Ring complexity, stereocenters, macrocycles, unusual atom types

The score ranges from 1 (easy) to 10 (hard). **SAScore is fast and widely used as a baseline, but it conflates novelty with difficulty** — a genuinely novel but synthetically straightforward molecule may score poorly simply because its fragments are rare in PubChem.

### 2.2 SCScore: Learning Complexity from Reactions

**SCScore** (Coley et al., J Chem Inf Model, 2018) takes a data-driven approach. It trains a neural network on reaction data with one constraint: the product of a reaction should always have a higher complexity score than any of its reactants.

- Trained on ~1M reactions from Reaxys
- Produces a continuous score from 1 (simple, purchasable) to 5 (complex, multi-step)
- Better correlation with chemist intuition than SAScore for complex molecules

The key insight is that **synthetic complexity is relative** — it flows from simple starting materials toward complex products through reaction sequences, and this directionality can be learned.

### 2.3 RAscore: Retrosynthetic Accessibility

**RAscore** (Thakkar et al., Chem Sci, 2021) connects synthesizability scoring directly to retrosynthetic planning. It works by:

1. Running AiZynthFinder on a large set of molecules
2. Labeling each molecule as "route found" (1) or "no route found" (0)
3. Training a classifier (random forest or neural network) to predict this label from molecular features

**RAscore is the first score that explicitly reflects whether a retrosynthesis tool can find a route** — making it more actionable than heuristic scores. It also inherits AiZynthFinder's biases and limitations.

### 2.4 Comparison Table

| Score | Method | Speed | Correlation with Chemist Judgment | Key Limitation |
|---|---|---|---|---|
| **SAScore** | Fragment frequency + complexity penalty | ~10 us/mol | Moderate — penalizes novelty | Confuses rare with hard |
| **SCScore** | Neural network trained on reaction directionality | ~100 us/mol | Good for relative ranking | No absolute threshold; no route |
| **RAscore** | Classifier trained on AiZynthFinder outcomes | ~50 us/mol | Best among scores | Inherits retro-tool biases |

### 2.5 The Common Limitation

All three scores answer a binary or continuous version of the same question: *"Is this molecule synthesizable?"* None of them answer the more important question: *"How do we synthesize it?"*

**A molecule with a high RAscore is likely synthesizable, but the score provides no route — the chemist still needs to run a retrosynthesis tool separately.** This disconnect between scoring and route generation is the fundamental limitation of post-hoc approaches.

---

## 3. Generation-Time Integration: Building Synthesizability into the Generator

The alternative to post-hoc scoring is to make the generative model itself synthesizability-aware. Three families of approaches have emerged, each with distinct philosophies:

```
Generation-Time Integration — Three Families

┌────────────────────┐   ┌────────────────────┐   ┌────────────────────┐
│   RL-based         │   │   GFlowNet-based   │   │  Transformer-based │
│                    │   │                    │   │                    │
│ REINVENT           │   │ SynFlowNet         │   │ SynNet             │
│ + synth reward     │   │ CGFlow             │   │ SynFormer          │
│ Lib-INVENT         │   │ RGFN               │   │ SynthFormer        │
│                    │   │                    │   │                    │
│ Synthesizability   │   │ Synthesizability   │   │ Synthesizability   │
│ as REWARD          │   │ by CONSTRUCTION    │   │ by CONSTRUCTION    │
│ (no route)         │   │ (route included)   │   │ (route included)   │
└────────────────────┘   └────────────────────┘   └────────────────────┘
```

### 3.1 RL-Based: REINVENT + Synthesis Filter

The most straightforward integration is to include a synthesizability score as one component of the reinforcement learning reward during generation.

**REINVENT** (Blaschke et al., J Chem Inf Model, 2020; AstraZeneca) is the most widely used RL-based molecular generator. In its standard workflow:

- A prior model (RNN or Transformer) generates SMILES strings
- A scoring function evaluates each molecule on multiple objectives (docking score, QED, similarity, etc.)
- Policy gradient RL steers the generator toward high-scoring molecules

Adding synthesizability is simple: include SAScore or RAscore as a reward component with a user-defined weight. **Lib-INVENT** (Fialkova et al., J Chem Inf Model, 2022; AstraZeneca) extends this for library design, enumerating R-groups on a validated scaffold to ensure compatibility with parallel synthesis.

**Limitations of the RL approach**:

- The synthesizability signal is still a *score*, not a *route* — the generator doesn't learn chemistry, it learns to avoid low-scoring regions
- Score gaming: the model can exploit scoring function artifacts rather than genuinely improving synthesizability
- No guarantee of synthesizability — just a statistical tendency

### 3.2 GFlowNet Family: SynFlowNet, CGFlow, RGFN

The GFlowNet family represents a fundamentally different philosophy: **don't score molecules for synthesizability — generate them through synthetic actions so they are synthesizable by construction**.

#### SynFlowNet: The Paradigm Shift

**SynFlowNet** (Cretu et al., ICLR 2025, Mila / Yoshua Bengio group) is, in our view, the most conceptually significant advance in synthesis-aware design. It reframes molecular generation as a *synthetic planning* process:

```
SynFlowNet Action Space

State: current intermediate molecule (or empty)

Step 1: Select building block B_i from library
        ┌──────────────────────────────────┐
        │  Building Block Library          │
        │  (~100-300 validated fragments)  │
        │  Enamine, eMolecules, etc.       │
        └──────────┬───────────────────────┘
                   │
                   ▼
Step 2: Select reaction template R_j
        ┌──────────────────────────────────┐
        │  Reaction Template Set           │
        │  (~70-90 validated reactions)    │
        │  Hartenfeller et al. templates   │
        └──────────┬───────────────────────┘
                   │
                   ▼
Step 3: Apply reaction: intermediate + B_i → new intermediate
        (or B_i + B_j → initial intermediate)

        ┌─────┐   ┌─────┐         ┌──────────┐
        │ B_1 │ + │ B_2 │ ──R_1──→│ Intermed │
        └─────┘   └─────┘         └────┬─────┘
                                       │
                                  ┌────▼─────┐
                                  │ Intermed │ + B_3 ──R_2──→ Product
                                  └──────────┘

Step 4: Repeat or terminate
        → Final molecule + complete synthesis DAG
```

The key properties:

- **Synthesizability by construction**: Every molecule is assembled from purchasable building blocks via validated reaction templates. There is no post-hoc check needed — if the model generated it, a synthesis route exists.
- **GFlowNet sampling**: Unlike RL, which mode-collapses toward a single high-reward region, GFlowNets sample proportionally to the reward. This produces *diverse* sets of synthesizable molecules.
- **Synthesis DAG as output**: The generation trajectory *is* the synthesis route — molecule and pathway emerge together.

**SynFlowNet demonstrated that we do not need to separate molecular generation from synthesis planning — they can be the same process.** This is the paradigm shift.

Technically, SynFlowNet trains a policy over this action space using the GFlowNet objective: the probability of generating a molecule is proportional to its reward. This is fundamentally different from RL, which maximizes expected reward and tends to collapse onto a single high-reward mode. GFlowNet's proportional sampling means the model naturally discovers *diverse* high-reward molecules across different synthesis routes — a property that is extremely valuable in drug discovery, where we want multiple backup candidates, not a single optimized structure.

#### CGFlow: Adding Structure-Based Objectives

**CGFlow** (Shen et al., ICML 2025) extends SynFlowNet by incorporating structure-based binding objectives directly into the GFlowNet reward:

- Uses docking score (e.g., Vina) or co-folding model confidence as the reward signal
- Generates molecules that are simultaneously synthesizable (by construction) and predicted to bind a target protein
- First model to jointly optimize synthesizability and structure-based binding in a single generative process

The reward formulation in CGFlow typically takes the form:

- $R(\text{molecule}) = f(\text{docking score}) \times g(\text{drug-likeness}) \times h(\text{additional constraints})$
- Because the generation process itself enforces synthesizability, there is no need for a synthesizability term in the reward — it is guaranteed by construction

This is a significant integration point. Prior approaches optimized binding and synthesizability sequentially (generate for binding, then filter for synthesis). **CGFlow optimizes both simultaneously, avoiding the information loss that comes from sequential optimization.**

#### RGFN: Reaction-Level GFlowNet

**RGFN** (Koziarski et al., 2024, Mila) operates at the reaction level rather than the building-block level:

- Actions correspond to selecting specific reactions from a curated reaction space
- Explores synthesis routes as first-class objects
- Complementary to SynFlowNet — focuses more on the reaction sequence than on building block selection

### 3.3 Transformer Family: SynNet, SynFormer, SynthFormer

The Transformer family takes a different route to the same destination: generate synthesis pathways directly using the scalability of Transformer architectures.

#### SynNet: The Early Attempt

**SynNet** (Gao et al., 2022, MIT) was among the first to frame molecular generation as a sequence of synthetic actions:

- Three action types: *add* a building block, *react* two intermediates, *modify* a functional group
- Actions are predicted sequentially by a set of neural networks
- Molecules are synthesizable by design — each action is a valid synthetic operation

SynNet proved the concept but was limited by its action vocabulary and scalability. It handled only a small building block set (~1,000 building blocks) and short synthesis sequences (typically 2-3 steps). The architecture also used separate networks for each action type, making it hard to scale the approach or train end-to-end.

#### SynFormer: The Scalable Solution

**SynFormer** (Gao, Luo & Coley, PNAS, 2025, MIT) is the most capable Transformer-based synthesis-aware generator to date. It addresses SynNet's limitations through a carefully designed architecture:

```
SynFormer Architecture

Input: target molecule (for local exploration)
       or property objective (for global exploration)
       ┌──────────────────────────────────────┐
       │                                      │
       ▼                                      │
┌─────────────────┐                           │
│ Transformer     │                           │
│ Encoder         │ Encodes molecular or      │
│                 │ property specification     │
└────────┬────────┘                           │
         │                                    │
         ▼                                    │
┌─────────────────┐                           │
│ Autoregressive  │ Predicts sequence of      │
│ Pathway Decoder │ (reaction, building block)│
│ (Transformer)   │ pairs step by step        │
└────────┬────────┘                           │
         │                                    │
         │  At each step:                     │
         │  1. Select reaction template       │
         │  2. Select building block(s)       │
         ▼                                    │
┌─────────────────┐                           │
│ Diffusion-Based │ Selects building blocks   │
│ Building Block  │ from large catalogs       │
│ Selector        │ (~130K Enamine BBs)       │
│ (operates in    │                           │
│  fingerprint    │                           │
│  space)         │                           │
└────────┬────────┘                           │
         │                                    │
         ▼                                    │
   Synthesis pathway                          │
   (molecule + route)                         │
         │                                    │
         └────────────────────────────────────┘
              (iterative refinement via
               autoregressive decoding)
```

The critical innovation is the **diffusion-based building block selector**. Instead of classifying over a fixed, enumerated set of building blocks (which doesn't scale), SynFormer:

1. Represents each building block as a molecular fingerprint
2. Uses a diffusion model to *generate* the fingerprint of the ideal building block at each step
3. Retrieves the nearest real building block from the catalog

This means SynFormer can work with building block catalogs of **130,000+ compounds** — far beyond what GFlowNet-based approaches currently handle.

**SynFormer operates in two modes**:

- **Local exploration**: Given a query molecule, generate synthesizable analogs. Useful for lead optimization where the chemist wants molecules similar to a hit but with a clear synthesis route.
- **Global exploration**: Given a property oracle (e.g., docking score), generate synthesizable molecules that optimize it. Useful for de novo design campaigns.

On standard benchmarks, SynFormer achieves:

- Top-3 AUC on PMO benchmark tasks among synthesis-constrained methods
- Generates synthesis routes of 2-6 steps using commercially available building blocks
- Handles building block sets 10-100x larger than GFlowNet methods

#### SynthFormer: 3D-Aware Synthesis Planning

**SynthFormer** (2024) adds 3D structural awareness to synthesis-aware generation:

- Uses a 3D equivariant GNN to encode pharmacophore-level information
- Transformer decoder generates synthesis actions conditioned on 3D target features
- Bridges the gap between structure-based design and synthesis-aware design

### 3.4 Comparison Table: Generation-Time Approaches

| Model | Family | Key Innovation | Synth. Guarantee | Building Block Scale | Structure-Aware? |
|---|---|---|---|---|---|
| **REINVENT + filter** | RL | Synth score as RL reward | No (statistical) | N/A (SMILES-based) | Via reward only |
| **Lib-INVENT** | RL | Library-focused R-group design | Partial (scaffold) | ~1K R-groups | No |
| **SynFlowNet** | GFlowNet | Reaction template + BB as action space | Yes (by construction) | ~300 BBs | No |
| **CGFlow** | GFlowNet | + Structure-based binding reward | Yes (by construction) | ~300 BBs | Yes (docking/co-folding) |
| **RGFN** | GFlowNet | Reaction-level action space | Yes (by construction) | ~300 BBs | No |
| **SynNet** | Transformer | Sequential synthetic actions | Yes (by construction) | ~1K BBs | No |
| **SynFormer** | Transformer | Diffusion-based BB selection | Yes (by construction) | ~130K BBs | Via oracle |
| **SynthFormer** | Transformer | 3D equivariant GNN encoder | Yes (by construction) | ~10K BBs | Yes (pharmacophore) |

### 3.5 GFlowNet vs Transformer: The Two Leading Paradigms

The two most promising families — GFlowNet (SynFlowNet/CGFlow) and Transformer (SynFormer) — have complementary strengths:

**GFlowNet family (SynFlowNet, CGFlow)**:

- Excels at **diversity** — GFlowNets sample proportionally to reward, naturally avoiding mode collapse
- Principled exploration of the chemical space without getting stuck in local optima
- Native multi-objective handling via reward decomposition
- Current limitation: restricted to relatively small building block libraries (~100-300)

**Transformer family (SynFormer)**:

- Excels at **scalability** — handles 130K+ building blocks via diffusion-based selection
- Faster inference for individual molecules
- Flexible input specification (molecule for analogs, property oracle for de novo)
- Current limitation: less principled diversity mechanism — relies on sampling temperature rather than GFlowNet's proportional sampling

**The ideal system would combine GFlowNet's diversity-aware sampling with SynFormer's scalable building block handling.** We expect this convergence to happen within the next 1-2 years.

---

## 4. Integration with Co-Folding and Generative Design

The models described above are powerful on their own. But the real impact comes when we integrate them with the rest of the drug discovery pipeline — specifically, with co-folding models for binding prediction and bespoke library strategies for chemical space navigation.

### 4.1 The Fragmented Pipeline Problem

Today's typical AI-driven drug discovery workflow looks like this:

```
Current Fragmented Pipeline
═══════════════════════════

Step 1:  Generative model (REINVENT, diffusion, etc.)
              │
              ▼
         Candidate molecules (optimized for predicted binding)
              │
              ▼
Step 2:  SAScore / RAscore filter
              │
              ├── FAIL → discard (30-60% of candidates)
              │
              ▼
Step 3:  Docking or co-folding (AF3, Boltz-2, Chai-1)
              │
              ▼
Step 4:  Retrosynthesis planning (ASKCOS, AiZynthFinder)
              │
              ├── FAIL → discard (another 20-40%)
              │
              ▼
Step 5:  Medicinal chemist review → CRO synthesis

         Total attrition: 50-80% of generated molecules discarded
         Information loss: each step is independent
```

The problem is not just the high attrition rate — it is the **information loss at each handoff**. The generative model doesn't know about synthesis constraints. The docking model doesn't know about synthesis feasibility. The retrosynthesis tool doesn't know about the binding context. Each component operates in isolation, and the sequential filtering discards potentially valuable candidates.

The integrated vision looks fundamentally different:

```
Integrated Pipeline (Emerging)
══════════════════════════════

Step 1:  Synthesis-aware generator
         (SynFlowNet / CGFlow / SynFormer)
              │
              │  Generates: molecule + synthesis route
              │  Constraint: uses validated reactions + purchasable BBs
              │
              ▼
Step 2:  Co-folding validation (AF3 / Boltz-2 / Chai-1)
              │
              │  Predicts: binding pose + confidence score
              │  OR: used as reward signal during generation (CGFlow)
              │
              ▼
Step 3:  Multi-objective ranking
         (binding + synthesizability + ADMET)
              │
              ▼
Step 4:  Route validation + CRO execution (Part 5)

         Attrition: minimal — molecules are synthesizable by design
         Information flow: binding and synthesis considered jointly
```

**The shift from sequential filtering to joint optimization is the architectural insight that connects synthesis-aware design to the broader drug discovery pipeline.**

### 4.2 Co-Folding Model Connection

Co-folding models — AlphaFold3 (Abramson et al., Nature, 2024), Boltz-2 (Wohlwend et al., 2025), Chai-1 (Chai Discovery, 2024) — predict the 3D structure of a protein-ligand complex given the protein sequence and ligand structure. Their relevance to synthesis-aware design comes through two mechanisms:

**Mechanism 1: Co-folding confidence as a filter**

After a synthesis-aware model proposes candidates, co-folding can evaluate their binding potential:

- Run AF3/Boltz-2/Chai-1 on (target protein, candidate molecule) pairs
- Use confidence metrics (ipTM, pLDDT at the interface) as binding quality estimates
- Rank candidates by confidence and prioritize for synthesis

This is still a sequential pipeline, but with a better binding evaluator than traditional docking.

**Mechanism 2: Co-folding confidence as a generative reward**

CGFlow (Shen et al., ICML 2025) demonstrates the more powerful integration:

- The GFlowNet's reward function directly incorporates docking or co-folding confidence
- During generation, each candidate is evaluated for binding as it is assembled
- The generator learns to build molecules that are *simultaneously* synthesizable and predicted to bind

**This closes the loop between structure-based drug design and synthesis-aware generation for the first time.**

There is an important caveat. Co-folding confidence scores are not the same as binding affinity:

| Metric | What It Measures | Correlation with Affinity | Use Case |
|---|---|---|---|
| ipTM | Interface predicted TM-score | Moderate (r ~ 0.3-0.5) | Complex formation quality |
| pLDDT (interface) | Per-residue confidence at binding site | Weak-moderate | Local structure reliability |
| Boltz-2 affinity head | Direct affinity prediction | Better (r ~ 0.6) | Relative ranking |
| Docking score (Vina) | Physics-based binding estimate | Moderate (r ~ 0.4-0.5) | Fast screening |

The gap between co-folding confidence and actual binding affinity remains a significant limitation. CGFlow optimizes for the *predicted* reward, which may not correlate perfectly with experimental binding. This is an active area of research — Boltz-2's dedicated affinity prediction head (Pearson r ~ 0.62) is a promising step, but we are not yet at the point where co-folding confidence reliably predicts experimental potency.

### 4.3 Bespoke Library Docking Connection

There is a deep conceptual connection between synthesis-aware generative models and the bespoke library paradigm that we should make explicit.

**What is a bespoke library?** A bespoke (or make-on-demand) virtual library is a combinatorial enumeration of molecules that can be synthesized from validated reactions and available building blocks. Examples include:

- **Enamine REAL**: ~40B molecules enumerable from ~200 validated reactions + ~230K building blocks
- **WuXi GalaXi**: ~30B molecules, similar principle
- **V-SYNTHES / V-SYNTHES2** (Lyu et al., Shoichet Lab): Iterative docking-guided enumeration of bespoke libraries. V-SYNTHES fragments each reaction into building block positions, docks the fragments, scores them, and iteratively grows the most promising fragments into full molecules — achieving ultra-large virtual screening without exhaustive enumeration of the entire combinatorial space

**SynFlowNet and SynFormer are essentially AI-guided explorations of bespoke chemical space.** Both operate within the same constraints as bespoke libraries — validated reactions, purchasable building blocks — but navigate the space using learned reward signals rather than exhaustive enumeration. The connection becomes clear when we compare the approaches:

| Aspect | Exhaustive Enumeration (Enamine REAL) | Docking-Guided (V-SYNTHES) | AI-Guided (SynFlowNet/SynFormer) |
|---|---|---|---|
| **Chemical space** | Fixed combinatorial library | Fixed library, iterative docking | Dynamic, reward-driven |
| **Synthesis guarantee** | Yes (validated reactions) | Yes (validated reactions) | Yes (validated reactions + BBs) |
| **Exploration strategy** | Enumerate all, dock all | Iterative: dock fragment, grow, dock | Generative: sample proportional to reward |
| **Scale** | ~10^10 molecules | ~10^10, but docks ~10^6 | ~10^6-10^8 reachable, samples ~10^4 |
| **Multi-objective** | Docking score only | Docking score only | Any differentiable reward |
| **Novelty** | Limited to pre-defined reactions | Limited to pre-defined reactions | Constrained by template set |

**The key difference is in how chemical space is navigated.** Exhaustive enumeration is complete but computationally prohibitive for the full space. V-SYNTHES uses a clever iterative docking strategy to explore without full enumeration. AI-guided approaches use learned reward signals to navigate the space intelligently, potentially finding high-reward regions that exhaustive methods would take much longer to reach.

These approaches are complementary, not competitive:

- Bespoke library docking excels when the chemical space is well-defined and the objective is purely binding — it provides systematic coverage with no risk of missing high-affinity molecules within the enumerated space
- AI-guided generation excels when multi-objective optimization is needed or when the chemical space should be explored more creatively — it can balance binding, ADMET, and synthesis cost simultaneously
- A hybrid strategy — using AI-guided methods to identify promising chemotype regions, then exhaustively enumerating and docking within those regions — may be the most powerful approach

The convergence of these strategies is already visible. Several groups are working on using synthesis-aware generators to propose novel reaction templates or building block combinations, which then get added to bespoke library enumeration platforms for exhaustive evaluation. This "AI-guided library design" paradigm may become the dominant approach for large-scale hit finding.

---

## 5. The End-to-End Vision and Remaining Gaps

### 5.1 The Ideal Pipeline

Combining the advances from Parts 2-4, we can now sketch the end-to-end pipeline that the field is converging toward:

```
The End-to-End Vision
═════════════════════

   Target protein structure (experimental or predicted)
        │
        ▼
   ┌──────────────────────────────────────────────────┐
   │  Synthesis-Aware Generative Model                │
   │  (SynFormer / CGFlow / next-generation hybrid)   │
   │                                                  │
   │  Reward signal:                                  │
   │    - Co-folding confidence (AF3/Boltz-2)         │
   │    - ADMET predictions                           │
   │    - Selectivity constraints                     │
   │    - Synthesizability (inherent, by construction) │
   └──────────────────┬───────────────────────────────┘
                      │
                      ▼
   Candidate molecule + synthesis route + predicted binding pose
                      │
                      ▼
   ┌──────────────────────────────────────────────────┐
   │  Route Validation                                │
   │                                                  │
   │  - Reaction condition prediction (Part 2)        │
   │  - Reagent availability check                    │
   │  - Yield estimation per step                     │
   │  - Cost estimation                               │
   └──────────────────┬───────────────────────────────┘
                      │
                      ▼
   ┌──────────────────────────────────────────────────┐
   │  CRO Execution (Part 5)                          │
   │                                                  │
   │  - Automated route translation (XDL, Chemify)    │
   │  - Robotic synthesis                             │
   │  - Analytical verification                       │
   └──────────────────────────────────────────────────┘
                      │
                      ▼
              Synthesized compound
              ready for biological assay
```

**This pipeline does not yet exist as a fully integrated system, but every component is now available in some form.** The challenge is integration.

### 5.2 Remaining Gaps

Several significant gaps remain between the current state and the end-to-end vision:

**Gap 1: Multi-objective balancing**

Real drug design requires simultaneous optimization of 5-10 objectives:

- Binding affinity to the target
- Selectivity against off-targets
- Synthesizability and route quality
- ADMET properties (absorption, distribution, metabolism, excretion, toxicity)
- Physicochemical properties (solubility, permeability, stability)

Current synthesis-aware models handle 2-3 objectives at most. Scaling to the full multi-objective landscape — with complex trade-offs and Pareto frontiers — remains unsolved.

**Gap 2: Conformational diversity**

Co-folding models typically predict a single binding pose. But proteins are dynamic — the binding site may adopt multiple conformations, and the ligand's bioactive conformation may differ from its lowest-energy conformation. Synthesis-aware models don't yet account for this conformational diversity. Ensemble methods — generating multiple co-folding predictions and aggregating confidence — are a partial solution, but they multiply computational cost and introduce new aggregation challenges.

**Gap 3: ADMET integration**

Most synthesis-aware generators optimize for binding and synthesizability but ignore ADMET properties. Integrating ADMET prediction models (e.g., ADMET-AI, pkCSM) as additional reward components is straightforward in principle but adds computational cost and complicates the multi-objective landscape.

**Gap 4: Route condition optimization**

Synthesis-aware models generate *what* reactions to run and *which* building blocks to use. They do not specify reaction *conditions* — temperature, solvent, catalyst, stoichiometry. This is the domain of reaction condition prediction models (Part 2), but integrating condition optimization into the generative loop remains an open problem.

**Gap 5: Experimental validation feedback**

The ultimate test of any synthesis-aware pipeline is whether the proposed routes actually work in the lab. Closing the loop — feeding experimental success/failure back into the generative model — requires:

- Standardized data formats (ORD) for recording outcomes
- Automated synthesis platforms that can report results programmatically
- Learning algorithms that can update on sparse, delayed, and noisy feedback
- A culture shift toward recording and sharing negative results (failed reactions)

These gaps are not merely technical inconveniences — they represent the difference between a research prototype and a production system that medicinal chemists will trust and use daily. Addressing them will require not just better algorithms, but better data infrastructure, better benchmarks, and closer collaboration between computational and experimental scientists.

### 5.3 The Bridge to Part 5

At this point in our series, we have traced the full computational pipeline:

- **Part 2**: AI can predict what reactions produce (forward prediction)
- **Part 3**: AI can plan multi-step synthesis routes (retrosynthesis)
- **Part 4**: AI can generate molecules that are synthesizable by construction, with routes included

The AI has proposed a molecule, a synthesis route, and a predicted binding pose. The final question remains: **can this route actually be executed in a lab?**

That question involves challenges far removed from algorithms — reagent procurement, equipment availability, CRO communication protocols, analytical verification, and the gap between computational prediction and experimental reality. That is the subject of Part 5.

---

## Key Takeaways

- **Post-hoc scores** (SAScore, SCScore, RAscore) answer "is this synthesizable?" but not "how?" — they are useful as filters but insufficient for route generation.
- **SynFlowNet** (GFlowNet-based) introduced the paradigm of synthesizability by construction — molecules are assembled from building blocks via reaction templates, making the generation trajectory itself the synthesis route.
- **SynFormer** (Transformer-based) achieved the same guarantee with superior scalability — handling 130K+ building blocks via a diffusion-based selection mechanism.
- **CGFlow** closed the loop between synthesis-aware generation and structure-based drug design by using docking/co-folding confidence as a GFlowNet reward.
- The **integrated pipeline** — synthesis-aware generation + co-folding validation + bespoke library strategies — is conceptually clear but not yet fully realized as an end-to-end system.
- **Remaining gaps** include multi-objective optimization, ADMET integration, conformational diversity, route condition prediction, and experimental feedback loops.

---

## References

- Blaschke, T. et al. "REINVENT 2.0: An AI Tool for De Novo Drug Design." *J. Chem. Inf. Model.* (2020).
- Coley, C. W. et al. "SCScore: Synthetic Complexity Learned from a Reaction Corpus." *J. Chem. Inf. Model.* (2018).
- Cretu, A. et al. "SynFlowNet: Towards Molecule Design with Guaranteed Synthesis Pathways." *ICLR* (2025).
- Ertl, P. & Schuffenhauer, A. "Estimation of Synthetic Accessibility Score of Drug-like Molecules based on Molecular Complexity and Fragment Contributions." *J. Cheminform.* (2009).
- Fialkova, V. et al. "LibINVENT: Reaction-based Generative Scaffold Decoration for in Silico Library Design." *J. Chem. Inf. Model.* (2022).
- Gao, W. et al. "Amortized Tree Generation for Bottom-up Synthesis Planning and Synthesizable Molecular Design." *ICLR* (2022).
- Gao, W., Luo, S. & Coley, C. W. "SynFormer: Synthesizable Molecular Generation with Transformer-based Building Block Selection." *PNAS* (2025).
- Koziarski, M. et al. "RGFN: Synthesizable Molecular Generation Using GFlowNets." *NeurIPS Workshop* (2024).
- Lyu, J. et al. "Ultra-Large Library Docking for Discovering New Chemotypes." *Nature* (2019).
- Shen, S. et al. "CGFlow: Synthesizable Molecular Generation with Co-folding Guidance." *ICML* (2025).
- Thakkar, A. et al. "Retrosynthetic Accessibility Score (RAscore) — Rapid Machine Learned Synthesizability Classification from AI Retrosynthetic Planning." *Chem. Sci.* (2021).
- Abramson, J. et al. "Accurate Structure Prediction of Biomolecular Interactions with AlphaFold 3." *Nature* (2024).
- Wohlwend, J. et al. "Boltz-2: Exploring the Frontiers of Biomolecular Prediction." (2025).

---

*Next in the series — Part 5: "From Algorithm to Lab — Can AI Synthesis Plans Actually Be Executed?"*
