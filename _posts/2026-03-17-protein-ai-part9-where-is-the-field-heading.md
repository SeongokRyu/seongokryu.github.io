---
title: "Protein AI Series Part 9: Where Is the Field Heading?"
date: 2026-03-17 10:30:00 +0900
categories: [Drug Discovery]
tags: [protein-ai, future-directions, drug-discovery, open-source, benchmarks]
math: true
---

**The Technical Evolution of Protein AI — A Record of Key Design Decisions**

*This is Part 9 of a 10-part series tracing the architectural choices behind modern protein structure prediction and design models. This is the final installment.*

---

## The Core Question

**What technical choices have converged across the field, what remains divergent, and what fundamental challenges await the next generation of protein AI models?**

Over the previous eight Parts, we traced a remarkable arc of innovation: from AlphaFold2's Evoformer revolution (2021) through the co-folding era (AF3, Boltz, Chai), the emergence of design models (RFdiffusion, BoltzGen, La-Proteina), the conformational diversity challenge (Frozen Trunk, BioEmu, Vilya-1), the engineering that makes it all trainable at scale, and the data landscape — from PDB experimental structures to AFDB's proteome-scale quaternary expansion — that increasingly determines competitive advantage. This final Part steps back to identify the patterns, gaps, and open frontiers.

---

## 1. Where the Field Has Converged — Eight Technical Axes

Each Part of this series examined one technical axis. Across all eight, clear convergence directions have emerged — alongside questions that remain stubbornly open.

| Technical Axis (Part) | Convergence Direction | Open Question |
|---|---|---|
| **Input** (Part 1) | Hybrid: MSA + PLM | Can PLMs fully replace MSA? |
| **Trunk** (Part 2) | Pairformer-based; Triangle Multiplication survives all variants | Is attention in the Trunk necessary at all? |
| **Generation** (Part 3) | Flow Matching for new models | Does ensemble generation still need SDEs? |
| **Scope** (Part 4) | All-atom co-folding; open-source ecosystem maturing | Can open-source match or exceed proprietary models? |
| **Design** (Part 5) | "Keep the Trunk, Replace the Head" pattern | What is the optimal sequence representation for generation? |
| **Diversity** (Part 6) | Frozen Trunk identified as the bottleneck | No universal solution yet |
| **Training** (Part 7) | FlashAttention + BF16 + FSDP + activation checkpointing | FP8 adoption? Scaling beyond 3B parameters? |
| **Data** (Part 8) | Distillation essential; PDB + synthetic data mixing | Optimal experimental-to-synthetic ratio? Data moat vs. open data? |

### 1.1 The Strongest Convergences

**Pairformer as de facto standard**: Every major co-folding model released since 2024 — AF3, Boltz-1/2, Chai-1, Protenix, SeedFold, OpenFold3 — uses a Pairformer-variant Trunk. The variants differ in efficiency (SeedFold: linear attention; Pairmixer: attention-free), but the core data structure — pair representation $z_{ij}$ updated through triangular operations — is universal.

**Flow Matching for generation**: Among models released in 2025–2026, Flow Matching has become the default generative framework: NP3, Proteina, La-Proteina, Complexa, FrameFlow, AlphaFlow. The shift from diffusion SDEs to flow matching ODEs reduces sampling steps (40 vs. 200) and simplifies training (no noise schedule tuning). EDM-style diffusion persists in AF3-family models (Boltz, Chai) that predated this convergence.

**All-atom co-folding**: The era of protein-only models is over. Every new model handles proteins, nucleic acids, small molecules, ions, and covalent modifications in a unified framework. The tokenization strategy — residue-level for polymers, atom-level for non-standard molecules — introduced by AF3 has been adopted almost universally.

**Engineering stack standardization**: As detailed in Part 7, the combination of FlashAttention + BF16 mixed precision + FSDP + activation checkpointing is now the default for GPU-based training. Models deviate only for architecture-specific reasons (Pairmixer has no attention to flash-optimize; AF2/AF3 run on TPUs).

### 1.2 The Deepest Disagreements

**Sequence representation for design**: Three competing approaches — geometric encoding (BoltzGen), latent variables (La-Proteina), and discrete tokens (ESM3) — address the same fundamental problem: how to jointly generate discrete amino acid sequences and continuous 3D coordinates. None has emerged as clearly superior, and the evaluation criteria themselves are disputed (in silico metrics vs. experimental validation).

**Conformational diversity**: The Frozen Trunk problem (Part 6) is well-diagnosed but unsolved. Vilya-1's unified diffusion transformer is architecturally elegant but limited to macrocycles. BioEmu produces thermodynamically correct ensembles but only for monomers. NP3's polymer prior improves ConfBench scores but remains at 52%. No approach simultaneously achieves broad molecular scope, thermodynamic correctness, and computational tractability.

---

## 2. Patterns Across the Series — Meta-Observations

### 2.1 "Keep the Trunk, Replace the Head"

This pattern appeared in nearly every Part:

```
Part 2:  Evoformer (Trunk) stays while structure module changes
         AF2 → AF3: same Trunk concept, new diffusion head

Part 3:  IPA → Diffusion → Flow Matching
         Generation head evolves; Trunk representations persist

Part 5:  RF1 Trunk → RFdiffusion denoiser (SE(3) diffusion head)
         Boltz-2 Trunk → BoltzGen design head (geometric encoding)
         Trunk = reusable asset; Head = task-specific adapter

Part 6:  Frozen Trunk problem
         The very success of "Keep the Trunk" creates the
         conformational diversity bottleneck — z_ij is too static
```

The recurrence of this pattern reflects a deeper truth: **learning rich representations of protein physics is the hardest part**. Once a model has internalized the relationship between sequence, evolutionary signal, and 3D geometry in its Trunk, adapting that knowledge to new tasks (generation, design, affinity prediction) is comparatively straightforward. The Trunk is the most valuable asset — and also the hardest to change.

### 2.2 The Persistence of the Triangle

Across all Trunk variants — Evoformer, Pairformer, SeedFold, Pairmixer — one operation survives every simplification attempt: **Triangle Multiplication**.

```
Evoformer (2021):    Tri Mult ✓   Tri Att ✓   MSA Att ✓   Seq Att (in MSA)
Pairformer (2024):   Tri Mult ✓   Tri Att ✓   ────────    Seq Att ✓
SeedFold (2025):     Tri Mult ✓   Linear TA   ────────    Seq Att ✓
Pairmixer (2026):    Tri Mult ✓   ────────    ────────    ────────

                     ▲ SURVIVES    removed     removed     removed
                       ALL          or          early       by
                       variants     linearized               Pairmixer
```

Triangle Attention, MSA Attention, and Sequence Attention have all been removed or replaced in at least one competitive variant. Triangle Multiplication has not. The operation:

$$z_{ij}^{\text{out}} = \sum_k a_{ik} \cdot b_{jk}$$

implements third-party reasoning — updating the relationship between residues $i$ and $j$ through a mediating residue $k$ — and maps naturally to GPU matrix multiplication. It is both the core inductive bias and the most hardware-efficient operation in the Trunk. Pairmixer's central finding — "Triangle Multiplication Is All You Need" — is the strongest evidence that this operation, not attention, is the irreducible core of protein representation learning.

### 2.3 The Return of Physics-Based Priors

The field has traced an arc from physics through pure data back toward physics:

```
2021  AF2: End-to-end learning, minimal physics priors
      → "Let the model learn physics from data"

2024  AF3/Boltz: Gaussian noise prior for diffusion
      → Generic prior, no structural knowledge

2025  NP3: Polymer Prior (Langevin dynamics with bonding potentials)
      → Physics-informed starting point for flow matching
      → 40 steps instead of 200

2025  BioEmu: Boltzmann distribution p(x|seq) ∝ exp(-E(x)/kT)
      → Explicit thermodynamic objective
      → PPFT: free energy measurements as training signal

2025  Vilya-1: Pure chemical features (element, bond, charge, chirality)
      → No molecule-type labels → physics-first representation
```

The progression suggests that **pure data-driven learning hits diminishing returns** — physics-based priors provide the right inductive biases for problems where data is scarce (conformational ensembles, rare folds, out-of-distribution molecules). This parallels similar trends in other scientific ML domains.

### 2.4 Data as the New Competitive Axis

A pattern that emerged most clearly in Part 8: as architectures converge, **data becomes the primary differentiator**.

```
2021  AF2: Architecture is the breakthrough
      → Evoformer + IPA = unprecedented accuracy
      → Data (PDB) is necessary but not the innovation

2024  AF3/Boltz/Chai: Architecture converges (Pairformer + Diffusion)
      → Open-source matches AF3 on architecture alone
      → Differentiation begins to shift toward data strategy

2025  SeedFold: 26.5M distillation data is essential, not optional
      → Removing distillation mid-training causes immediate degradation
      → Architecture needs data volume to compensate for lost inductive bias

2026  IsoDDE: Proprietary experimental data creates the performance gap
      → Same architecture family as open-source models
      → Decisive advantage: internal binding data at scale

2026  AFDB Quaternary: Open data ecosystem expands to complexes
      → 1.8M high-confidence dimeric structures now publicly available
      → Open-source community gains interface data that was previously scarce
```

The implication: the "data pyramid" (Part 8) — sequences (billions) > synthetic structures (millions) > experimental structures (hundreds of thousands) > functional data (tens of thousands) — defines the landscape of competitive advantage. Each layer is harder to acquire than the one below it, and the scarcest layer (proprietary functional data) is increasingly where competitive moats form.

### 2.5 "What We Thought Was Essential, Wasn't"

A recurring pattern of elimination:

| Year | Discovery | Impact |
|---|---|---|
| 2022 | ESMFold: MSA not required for structure prediction | PLM as MSA replacement |
| 2023 | RF2: Neither IPA nor Triangle Attention needed for AF2-level accuracy | Opened path to Pairmixer |
| 2024 | AF3: Recycling through the Trunk can be removed | Simplified training |
| 2025 | SeedFold: Triangle Attention can be linearized without quality loss | Sub-cubic Trunk |
| 2026 | Pairmixer: All attention in the Trunk can be removed | Attention-free protein AI |

Each "removal" discovery required first understanding why the component was included (usually for good theoretical reasons), then empirically demonstrating that the model could achieve equivalent performance without it. The accumulation of these discoveries suggests the field is converging toward a minimal sufficient architecture — one where every remaining component (Triangle Multiplication, pair representation, flow matching) has survived multiple elimination attempts.

---

## 3. The Wet-Lab Validation Gap

### 3.1 Computational Success $\neq$ Experimental Success

The most consequential gap in protein AI is between **in silico metrics and experimental outcomes**. A designed protein that scores well on designability (self-consistency via AF2 refolding), novelty, and diversity may still fail to:

- Express in a cell (solubility, folding kinetics)
- Fold into the predicted structure (kinetic traps, aggregation)
- Bind the intended target (affinity, specificity)
- Function as intended (catalysis, signaling)

### 3.2 Experimental Validation Status by Model

| Model | Design Target | Experimental Scale | Key Result | Publication |
|---|---|---|---|---|
| **RFdiffusion** | De novo binders, scaffolds | Hundreds of designs tested | ~15–30% success rates for binders; multiple designs validated in vivo | *Nature* 2023 |
| **RFdiffusion2** | Enzyme active sites (all-atom) | Dozens of designs | Functional enzymes with designed active sites | 2025 |
| **RFAntibody** | Antibody CDR design | Multiple targets | Competitive with experimental methods | 2024–25 |
| **Chai-2** | De novo antibody binders | 52 targets, ~20 designs each | **~16% hit rate** (~100 $\times$ over previous methods); 50% of targets hit with $\leq 20$ designs | 2025 |
| **BoltzGen** | Universal binder design | Limited experimental data | 6 design protocols validated computationally | 2025 |
| **La-Proteina** | Backbone + sequence co-design | Computational benchmarks | 75%+ co-designability (in silico) | 2025 |
| **Complexa** | Binder design | Computational benchmarks | SOTA on binder design benchmarks (in silico) | 2025–26 |

The maturity gradient is striking: **RFdiffusion** (2023) has three years of experimental validation and hundreds of confirmed designs. **BoltzGen** and **La-Proteina/Complexa** (2025) show strong computational results but lack large-scale experimental confirmation. This lag is inherent — wet-lab validation takes 3–12 months per design cycle.

### 3.3 The In Silico–Experimental Correlation Problem

The standard computational validation pipeline:

```
Generated structure → AF2/Boltz refolding → RMSD to design → "designability"
```

This self-consistency check asks: "Does a prediction model agree that this design folds as intended?" But the prediction model shares many of the same biases as the generation model — especially when both are Pairformer-based. High self-consistency may reflect **shared model biases** rather than genuine foldability.

Metrics that correlate better with experimental success:
- **Rosetta energy**: Physics-based energy function, orthogonal to learned models
- **Molecular dynamics stability**: Does the design maintain its structure in simulation?
- **Expression prediction**: Sequence-level features predicting soluble expression
- **Experimental hit rate**: The only ground truth — but expensive and slow

The field needs a "PoseBusters for protein design" — a filter that catches physically implausible designs before they reach the wet lab.

### 3.4 The Tightening Design Loop

The practical impact of AI protein design depends on the speed of the **design–make–test–learn** cycle:

```
2023:  RFdiffusion (backbone)
         → ProteinMPNN (sequence)
           → Expression + binding assay (months)
             → Manual analysis → next round

2025:  BoltzGen/Chai-2 (end-to-end: structure + sequence)
         → Automated expression (Recursion, Emerald Cloud Lab)
           → High-throughput binding (weeks)
             → Active learning → next round

Future: Unified AI model (design + predict + score)
          → Robotic synthesis + characterization (days)
            → Automatic data feedback → model update
              → Continuous improvement cycle
```

Each acceleration of this loop — from months to weeks to days — multiplicatively increases the value of AI design models. The models that integrate most tightly with automated experimental platforms will have the fastest learning cycles and the best real-world performance.

---

## 4. The Limits of Current Benchmarks

### 4.1 From CASP to Continuous Evaluation

**CASP** (Critical Assessment of protein Structure Prediction) has been the gold standard since 1994, running biennial blind prediction competitions. But its limitations have become apparent:

- **Biennial cadence**: Two-year gaps cannot track a field where major models appear quarterly
- **Static targets**: Test proteins are selected once; models can overfit to the distribution of CASP targets
- **Structure-only focus**: No evaluation of dynamics, binding, or design

**CAMEO** (Continuous Automated Model EvaluatiOn) provides weekly blind evaluation against newly deposited PDB structures, offering real-time tracking of model performance. **PLINDER** specifically benchmarks protein-ligand interaction prediction, filling a gap that CASP never addressed.

### 4.2 PoseBusters: Physical Validity

**PoseBusters** (2024) introduced a critical filter: checking whether predicted structures satisfy basic **physical constraints** — bond lengths within tolerance, no steric clashes, correct chirality, plausible torsion angles. Surprisingly, many AI-generated structures that score well on RMSD and lDDT fail PoseBusters checks:

```
Scenario:
  AI predicts protein-ligand complex
  lDDT: 0.85 (good)
  Ligand RMSD: 1.2 Å (good)

  BUT:
  → C-C bond stretched to 2.1 Å (should be ~1.5 Å)
  → Steric clash between ligand and protein
  → Inverted chirality at stereocenter

  PoseBusters verdict: FAIL
```

This reveals a gap between **geometric accuracy** (learned from data) and **physical validity** (constrained by chemistry). Models trained purely on structural data can produce geometrically close but physically impossible structures — a problem that matters critically for drug design, where incorrect bond geometry invalidates binding affinity predictions.

### 4.3 Data Leakage and Temporal Splits

A pervasive concern: **training data overlapping with test data**. The PDB grows continuously; structures deposited after a model's training cutoff may share high sequence identity with training examples. Proper evaluation requires:

- **Temporal splits**: Only evaluate on structures deposited after the training data cutoff
- **Sequence identity filtering**: Remove test targets with $\gt 30$% sequence identity to any training example
- **Structural novelty**: Test on genuinely novel folds not present in the training distribution

ConfBench (Part 6) exemplifies good benchmark design: it tests a capability (conformational discrimination) that models were not explicitly trained for, revealing genuine weaknesses rather than memorization.

### 4.4 Benchmark Overfitting

When every model optimizes for the same benchmarks, the field risks **Goodhart's Law**: the metric ceases to be a good measure once it becomes the target. Signs of this in protein AI:

- Models achieving near-identical lDDT scores on standard test sets while differing substantially on out-of-distribution targets
- Hyperparameter tuning against benchmark leaderboards rather than downstream applications
- New benchmarks (ConfBench, PoseBusters, PLINDER) consistently exposing weaknesses that existing benchmarks missed

The healthiest dynamic is a **co-evolution of models and benchmarks**: new models expose benchmark limitations, new benchmarks expose model limitations, and both improve.

---

## 5. Industrial Applications — Where Do These Models Fit in Drug Discovery?

### 5.1 Structure-Based Virtual Screening

The most immediate industrial impact of protein structure prediction: **enabling virtual screening for targets without experimental structures**.

```
Before AI structure prediction:

  Target protein → X-ray crystallography (months-years, may fail)
                   → IF structure obtained: docking-based virtual screening
                   → IF not: phenotypic screening (blind)

After AI structure prediction:

  Target protein → AF3/Boltz-2 prediction (minutes)
                   → Docking against AI structure
                   → Prioritized compound library for experimental testing
```

This has already changed practice at major pharmaceutical companies. AI-predicted structures are routinely used for virtual screening when experimental structures are unavailable — particularly for targets in disease-relevant conformations that resist crystallization (membrane proteins, IDPs, transient complexes).

### 5.2 Binding Affinity: The Remaining Gap

**Binding affinity prediction** is the holy grail of computational drug design. Two approaches compete:

**Physics-based** (FEP+, TI): Free Energy Perturbation calculates relative binding free energies between congeneric ligands using molecular dynamics. Accuracy: ~1 kcal/mol for well-parameterized systems. Cost: GPU-hours per ligand pair.

**AI-based** (Boltz-2, IsoDDE): End-to-end prediction of $K_d$ or $\Delta G$ from structure. Boltz-2's affinity head (Part 7, Stage 3) predicts binding affinity as an additional output. IsoDDE reportedly integrates affinity prediction into a multi-task framework.

Current status: **AI affinity prediction has not yet matched FEP+ accuracy** for lead optimization (where 0.5 kcal/mol differences matter). However, AI models are orders of magnitude faster and can handle targets where FEP+ requires expensive setup (novel scaffolds, protein-protein interactions, non-congeneric series). The likely convergence: **AI for initial ranking, FEP+ for final optimization**.

### 5.3 Biologics vs. Small Molecules

AI protein design's impact differs dramatically between molecule types:

**Biologics** (antibodies, nanobodies, peptides): AI design is genuinely transformative.
- Traditional approach: immunization campaigns or phage display → months of screening
- AI approach: Chai-2/RFAntibody generates candidates in silico → immediate experimental testing
- Chai-2's ~16% hit rate across diverse targets represents a step change in efficiency
- BoltzGen's six design protocols (protein, peptide, antibody, nanobody, small molecule, redesign) provide a unified design platform

**Small molecules**: AI's primary contribution is structure prediction for virtual screening, not molecule design.
- Small molecule generative models (diffusion-based, autoregressive) exist but operate in a different design space
- The binding pocket prediction from protein AI feeds into traditional SBDD (Structure-Based Drug Design) pipelines
- Protein-ligand co-folding (AF3, Boltz-2, NP3) improves pose prediction, which directly benefits small molecule optimization

### 5.4 The Automation Frontier

The companies integrating AI design with automated experimental platforms — Recursion (robotic biology), Isomorphic Labs (DeepMind spin-off), EvolutionaryScale (ESM3), and Chai Discovery — are positioned to close the design–test loop fastest. The competitive advantage is not the model alone but the **speed of iteration**: a slightly worse model with 10 $\times$ faster experimental feedback can outperform a better model with slow feedback.

---

## 6. Looking Ahead: 2026–2027

### 6.1 Will Unified Architectures Prevail?

Vilya-1's unified diffusion transformer (Part 6) merges Trunk and Diffusion Module, making $z_{ij}$ dynamic at every denoising step. This solves the Frozen Trunk problem architecturally but at cost: $O(T \times L^2)$ instead of $O(L^2)$ for the pair representation.

**Prediction**: Unified architectures will expand beyond macrocycles to protein-ligand complexes within 2026–2027, but with approximations — updating $z_{ij}$ every $k$ steps rather than every step, or using lightweight update mechanisms. The full $O(T \times L^2)$ cost is unlikely to be acceptable for large complexes, but the principle of dynamic pair representations will become standard.

### 6.2 Will PLMs Replace MSA?

The trend from Part 1 is clear: PLM contribution is growing while MSA dependency is shrinking. ESMFold demonstrated MSA-free prediction (with accuracy loss); Chai-1 showed that PLMs can compensate when MSA quality is poor.

**Prediction**: By 2027, the best single-sequence (MSA-free) model will match the best MSA-dependent model on standard benchmarks. MSA will remain useful for edge cases (orphan proteins, de novo designs with no evolutionary history) but cease to be the default requirement.

### 6.3 Active Learning: Closing the Loop

The most impactful near-term development is not a new architecture but a new **workflow**: automated design–test–learn cycles where AI models propose designs, robotic platforms test them, and results feed back into model training.

```
Current (open-loop):

  AI model (fixed) → designs → experiments → publications
                                              (months later)

Future (closed-loop):

  AI model v1 → designs → robotic testing (days)
       ↑                         │
       └── retrain on results ←──┘

  AI model v2 → better designs → ...
```

This requires:
- **Standardized experimental protocols** that produce machine-readable results
- **Uncertainty quantification** in AI models (knowing which designs to test first)
- **Efficient fine-tuning** on small batches of new experimental data

### 6.4 Multi-Task Foundation Models

IsoDDE (Isomorphic Labs, 2026) signals the direction: a single model predicting structure, binding affinity, ADMET properties, and selectivity. The "multi-task drug design engine" treats each downstream application as a head on a shared Trunk — the "Keep the Trunk, Replace the Head" pattern scaled to its logical conclusion.

**Prediction**: By 2027, the leading protein AI models will be multi-task: structure prediction, confidence, affinity, design, and property prediction sharing a common Trunk. The competitive differentiation will shift from architecture (converged) to **data** (proprietary experimental datasets) and **integration** (speed of the design–test loop).

### 6.5 The Data Ecosystem: Expansion and Risk

The data landscape (Part 8) is evolving along three dimensions that will shape 2026–2027:

**AFDB Quaternary Expansion**: The 2026 expansion of the AlphaFold Database to proteome-scale quaternary structures (Han et al.) added 31M dimeric complex predictions, of which 1.8M meet high-confidence thresholds. Unlike monomeric AFDB, these contain **interface information** — directly usable for training binder design and PPI prediction models. This public resource partially addresses the data scarcity that previously limited complex-aware model training to PDB's ~tens of thousands of experimental multimers.

**Continuous distillation as requirement**: SeedFold's ablation (Part 8) established that distillation data is not merely useful for pre-training — it must be maintained throughout training. Removing it at step 47,612 caused immediate accuracy degradation. This finding has architectural implications: post-AF2 models that replaced IPA's geometric inductive bias with general-purpose diffusion transformers fundamentally need more data to generalize. The 180K PDB structures are insufficient; distillation at scale (SeedFold: 26.5M, OpenFold3: 13M) is now a requirement, not an optimization.

**The data flywheel and model collapse risk**: As each generation of models generates synthetic training data for the next (AF2 → AFDB → Boltz-2/NP3 → next generation), a recursive loop forms. This flywheel drives improvement but carries the risk of **model collapse** — systematic errors reinforced rather than corrected across generations. The role of experimental PDB data as a "ground truth anchor" becomes increasingly critical. Whether the open-source community can sustain the flywheel without access to proprietary experimental data (IsoDDE's advantage) remains the central strategic question.

### 6.6 Democratization of Access

The barrier to using protein AI has dropped dramatically:

| Year | Access Model | User Requirement |
|---|---|---|
| 2021 | AF2: Download code, install dependencies, run MSA search | ML engineering + biology |
| 2023 | AlphaFold Server: Web interface for prediction | Biology only |
| 2025 | Boltz API, Chai API: Cloud endpoints for prediction + design | API call |
| 2026+ | Integrated platforms: Design → predict → score in one interface | Domain question only |

This democratization means that the **bottleneck shifts from computational access to biological insight**: knowing which protein to target, which conformer matters, and how to validate results experimentally.

---

## 7. The Open-Source Ecosystem's Long-Term Impact

### 7.1 Where Open Source Won

The open-source reproduction race (Part 4) has produced models that **match or exceed AF3** on most benchmarks:

```
AF3 (2024, academic license)
  ├── Boltz-2 (MIT) ─── matches AF3 + affinity head
  ├── OpenFold3 (Apache 2.0) ─── matches AF3, best RNA
  ├── Chai-1 (Apache 2.0) ─── matches AF3 + ESM-2 integration
  └── Protenix (open) ─── matches AF3 with minor improvements

Verdict: Open-source collectively matches AF3 across all benchmarks
         AND adds capabilities (affinity, PLM integration) AF3 lacks
```

### 7.2 Where Proprietary Models Lead

**IsoDDE** represents capabilities that open-source has not yet matched:
- Multi-task: structure + affinity + ADMET in one model
- Trained on proprietary experimental data (Isomorphic Labs' internal datasets)
- 1000-seed sampling with learned confidence ranking
- Integrated into a drug discovery pipeline (not just a prediction tool)

The pattern: **open-source matches on architecture; proprietary leads on data and integration**.

### 7.3 The Second-Order Innovation Effect

The most important impact of open-source is not the models themselves but the **innovation they enable**:

```
OpenFold (2022, Apache 2.0)
  → AlphaFlow (2024): AF2 fine-tuned for conformational ensembles
  → Training recipe shared → informed Boltz-1 development

Boltz-1/2 (MIT, training code open)
  → BoltzGen (2025): design capability added
  → BoltzDesign1: further design extensions
  → Community can experiment with training modifications

Proteina (open, 2025)
  → La-Proteina: latent flow matching extension
  → Complexa: binder design extension
  → Scaling law findings inform entire field
```

Each open-source release creates a platform for derivative innovation that the original authors could not have anticipated. The cumulative effect: the rate of innovation in protein AI is faster than any single organization could achieve alone.

### 7.4 License Matters

The choice between restrictive (AF3: academic-only) and permissive (Boltz-2: MIT, OpenFold3: Apache 2.0) licenses has had measurable consequences:

- **Academic-only**: AF3's code is available but cannot be used commercially → pharmaceutical companies must rely on open alternatives or proprietary reproductions
- **Permissive**: Boltz-2 and OpenFold3 are directly usable in commercial drug discovery pipelines → faster industrial adoption
- **Result**: The open-source models with permissive licenses have seen wider adoption than AF3 despite AF3's first-mover advantage

---

## 8. The Full Technical Timeline

```
[Timeline]

2018  AF1: Distance distribution prediction + gradient descent refinement

2021  AF2: Evoformer + IPA (end-to-end revolution)
      RF1: 3-track architecture (1D + 2D + 3D simultaneous)

2022  ESMFold: PLM-only prediction (no MSA required)
      ProteinMPNN: Inverse folding becomes standard
      OpenFold: AF2 open-source reproduction (Apache 2.0)
      UniFold: AF2 reproduction by DP Technology

2023  RFdiffusion: Design revolution (SE(3) diffusion)
      RF2: Triangle Attention shown unnecessary
      FrameDiff: SE(3) diffusion theoretical foundation

2024  AF3: Pairformer + EDM Diffusion (co-folding era begins)
      RFAA: All-atom co-folding (Baker Lab)
      Boltz-1, Chai-1: Open-source AF3 reproductions
      BioEmu: Boltzmann distribution learning (conformational ensembles)
      ESM3: Discrete multimodal generation (structure + sequence + function)

2025  Boltz-2: Affinity head added (structure + confidence + affinity)
      SeedFold: Linear Triangle Attention (sub-cubic Trunk)
      NP3: Flow Matching + Polymer Prior + ConfBench
      BoltzGen: Universal binder design (geometric encoding)
      Proteina → La-Proteina → Complexa: NVIDIA flow matching lineage
      OpenFold3: Complete open-source AF3 (Apache 2.0)
      Chai-2: Antibody design (~16% hit rate)
      Vilya-1: Unified Diffusion Transformer (dynamic z_ij)
      RFAntibody, RFdiffusion2: Baker Lab design extensions

2026  IsoDDE: Multi-task drug design engine (Isomorphic Labs)
      Pairmixer: Attention-free Trunk ("Triangle Multiplication Is All You Need")
      AFDB Quaternary: Proteome-scale complex predictions (31M dimers, 1.8M HC)


[Technical Axis Transitions]

Input:       MSA → PLM → Hybrid (MSA + PLM)
Trunk:       Evoformer → Pairformer → Pairmixer/SeedFold
Generation:  IPA → Diffusion (EDM) → Flow Matching
Scope:       Monomer → Complex → All-atom co-folding
Purpose:     Prediction → Design → Ensemble
Training:    DDP → FSDP, FP32 → BF16, single-stage → multi-stage
Data:        PDB only → PDB + distillation → PDB + AFDB + synthetic complexes


[Lineage Map]

[Structure Prediction]                   [Design / Generation]
━━━━━━━━━━━━━━━━━━━━                   ━━━━━━━━━━━━━━━━━━━

AF1 (2018)
 └→ AF2 (2021) ─── OpenFold             FrameDiff (2023) → FrameFlow
     │              UniFold              ProteinMPNN (2022)
     ├→ AF2-Multimer
     └→ AF3 (2024) ─── OpenFold3
         │               Protenix
         │               SeedFold
         ├→ Boltz-1 → Boltz-2 ────────→ BoltzGen, BoltzDesign1
         ├→ Chai-1 ───────────────────→ Chai-2
         └→ IsoDDE (2026)

RF1 (2021)
 ├→ RF2 (2023)
 └→ RFAA (2024) ──────────────────────→ RFdiffusion → RFdiff-AA
                                          → RFAntibody → RFdiff2

NP1 (2022) → NP2 → NP3 (2025)

ESM-1b → ESM-2 → ESMFold ────────────→ ESM3

BioEmu (2024) ────────────────────────→ PPFT

                                        Proteina (backbone FM)
                                         → La-Proteina (latent FM)
                                           → Complexa (binder design)

Vilya-1 (2025) ─── Unified Diffusion Transformer
PLACER (2025) ─── Conformational ensemble
AlphaFlow ──── Flow Matching + AF2 fine-tuning
Pairmixer (2026) ─── Attention-free Trunk
```

---

## 9. Closing Thoughts

### What this series traced

Over nine installments, we followed one core question: **why did each model make the technical choices it did?** The answers revealed a field that is simultaneously converging (Pairformer, Flow Matching, all-atom scope, standard training stack) and diversifying (design representations, conformational diversity approaches, commercial vs. open-source strategies).

### The four most important lessons

**1. Representations matter more than generation methods.** The Trunk — how a model represents residue-pair relationships — has been the most stable and valuable component across all model generations. Generation methods (IPA, diffusion, flow matching) have changed rapidly, but the pair representation $z_{ij}$ and the triangular operations that update it have persisted from Evoformer through Pairmixer. Investing in better representations pays off across all downstream tasks.

**2. Elimination is as important as innovation.** Some of the field's most impactful contributions were removals: RF2 removing IPA, Pairmixer removing attention, ESMFold removing MSA dependence. Each removal simplified the architecture, reduced computational cost, and clarified what is truly essential. The minimal architecture — Triangle Multiplication + FFN for the Trunk, flow matching for generation — may be close to the irreducible core.

**3. Engineering and data are first-class design decisions.** Part 7 showed that the same architecture trained with different engineering produces different results. Part 8 showed that the same architecture trained with different data produces different results. The gap between "published architecture" and "working system" is filled by crop strategies, precision management, distillation pipelines, and data mixing ratios. As architectures converge, the competitive frontier shifts to data — who has the best experimental measurements (IsoDDE), who generates the best synthetic structures (AFDB, Teddymer), and who mixes them most effectively (Boltz-2's 60/40, SeedFold's 50/50).

**4. Open data accelerates the entire field.** The AFDB — from 360K monomers (2021) to 214M monomers (2024) to 31M quaternary complexes (2026) — has been the single most impactful public resource in protein AI. Models trained on AFDB-derived data (NP3, Boltz-2, SeedFold) collectively outperform models without it. The expansion to quaternary structures provides interface information that was previously available only through the PDB's limited set of experimental complexes. Open data creates a rising tide that lifts all models — while proprietary data creates moats that only individual organizations can cross.

### What comes next

The next generation of protein AI will likely be defined not by a new architecture but by **closing the loop between computation and experiment**. The models are good enough to generate plausible designs; the remaining bottleneck is knowing which designs to test and learning from the results. The organizations that solve this — integrating AI design, automated experimentation, and continuous model improvement — will define the field's next chapter.

---

*Part of the series: The Technical Evolution of Protein AI — A Record of Key Design Decisions*
