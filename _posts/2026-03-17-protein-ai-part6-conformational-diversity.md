---
title: "Protein AI Series Part 6: The Conformational Diversity Problem"
date: 2026-03-17 10:00:00 +0900
categories: [Drug Discovery]
tags: [protein-ai, conformational-diversity, bioemu, vilya, frozen-trunk]
math: true
---

**The Technical Evolution of Protein AI — A Record of Key Design Decisions**

*This is Part 6 of a 10-part series tracing the architectural choices behind modern protein structure prediction and design models.*

---

## The Core Question

**The same protein can adopt multiple structures — can models capture this?**

Parts 1–5 implicitly assumed that each protein has a single "correct" structure. But real proteins are dynamic: a kinase switches between DFG-in (active) and DFG-out (inactive) conformations; an enzyme opens and closes upon substrate binding; a receptor rearranges its loops when an antibody docks. The difference between these conformers can be the difference between a drug that works and one that doesn't.

This Part examines why current co-folding models struggle with conformational diversity, and traces five distinct approaches to solving the problem — from simple seed variation to fundamental architectural redesign.

---

## 1. Why Conformational Diversity Matters

### 1.1 Apo vs. Holo: The Drug Design Challenge

A protein's structure depends on its binding partners. The **apo** (unbound) state often differs significantly from the **holo** (ligand-bound) state:

```
Apo (no ligand):              Holo (ligand bound):

   ╭──╮                           ╭──╮
   │  │    binding site            │  ╰─╮  binding site
   │  │ ←── open loop              │    │ ←── closed loop
   │  │                            │ ●  │     (ligand ●
   ╰──╯                           ╰──╯       enclosed)
```

**Kinase DFG motif** — one of the most therapeutically important conformational switches:
- **DFG-in** (active): Asp faces into the ATP-binding site → catalysis proceeds
- **DFG-out** (inactive): Asp rotates out → kinase inactive, Type II inhibitor binding site exposed

A structure prediction model that always returns the DFG-in conformation will miss the binding pocket that Type II inhibitors (imatinib, sorafenib) exploit. For drug design, predicting **which** conformer forms under specific conditions is as important as predicting any single structure accurately.

### 1.2 ConfBench: Quantifying the Problem

**ConfBench** (introduced with NP3, 2025) provides the first systematic benchmark for ligand-induced conformational change prediction, built from apo-holo pairs in the PLINDER dataset.

**Scoring function**:

$$\text{score} = \frac{\text{RMSD}(q, \text{alt}) - \text{RMSD}(q, \text{ref})}{\sqrt{(\text{RMSD}^2(q, \text{alt}) + \text{RMSD}^2(q, \text{ref}) + \text{RMSD}^2(\text{alt}, \text{ref})) / 2}}$$

where $q$ is the predicted structure, ref is the target state (apo or holo), and alt is the opposite state. A score of $+1$ means perfect match to the target state; $0$ means indistinguishable; $-1$ means the prediction matches the wrong state.

**Results**:

| Model | Overall Accuracy (score $\gt$ 0) | Apo Accuracy | Holo Accuracy |
|---|---|---|---|
| AF2-Multimer 2.3 | **29.6%** | — | — |
| NP3 | **51.9%** | 67.4% | 69.9% |

AF2-Multimer performs **worse than random** (50%) on conformational discrimination — it is systematically biased toward one state. NP3 improves substantially but still correctly classifies only half of cases. This benchmark reveals that **conformational diversity remains a largely unsolved problem**.

---

## 2. Root Cause: The Frozen Trunk

### 2.1 The Architectural Bottleneck

The AF3/Boltz architecture (Parts 2–4) has a fundamental structural limitation for conformational diversity. The information flow is:

```
Sequence + MSA ──→ Trunk (Pairformer, 1 pass) ──→ z_ij (FIXED)
                                                       │
                                                       ▼ (read-only)
                                              Diffusion Module
                                                       │
                                                       ▼
                                              3D coordinates
```

The pair representation $z\_{ij}$ is computed **once** by the Trunk and then passed to the Diffusion Module as a fixed conditioning signal. The Diffusion Module reads $z\_{ij}$ but **cannot modify it**. This means:

1. **$z\_{ij}$ is ligand-unaware**: The Trunk processes sequence and MSA — both ligand-independent. Whether a small molecule is present or absent, $z\_{ij}$ is essentially identical: $z\_{ij}^{\text{apo}} \approx z\_{ij}^{\text{holo}}$.

2. **Structure is determined at Trunk time**: Since $z_{ij}$ encodes residue-pair relationships and the Diffusion Module only performs coordinate-level refinement conditioned on this fixed blueprint, the protein's global conformation is effectively decided before the diffusion process begins.

3. **MSA averages over conformers**: The MSA encodes evolutionary covariance patterns averaged over millions of years and multiple functional states. For a kinase, both DFG-in and DFG-out states contribute to the MSA signal — the result is a single averaged pattern that cannot distinguish between states.

### 2.2 Formal Statement

In the standard AF3/Boltz pipeline:

$$z_{ij} = \text{Trunk}(\mathbf{s}_{\text{seq}}, \mathbf{m}_{\text{MSA}})$$

$$\mathbf{x}^{(t-1)} = \text{Denoise}(\mathbf{x}^{(t)}, z_{ij}, t) \quad \text{for } t = T, T-1, \ldots, 1$$

$z\_{ij}$ is constant across all denoising steps $t$. Even when the ligand's coordinates $\mathbf{x}\_{\text{ligand}}^{(t)}$ approach the binding site during denoising, the "who should be close to whom" information ($z\_{ij}$) cannot respond. Only fine-grained coordinate adjustments are possible — large-scale structural rearrangements (loop reordering, domain motions) are beyond the Diffusion Module's capacity.

---

## 3. Approach 1: Seed Variation (AF3, Boltz, Chai)

### 3.1 The Simplest Strategy

The most straightforward approach exploits the stochastic nature of diffusion: **run multiple predictions with different random seeds** and hope that different noise realizations lead to different conformers.

```
Same sequence + MSA + ligand
        │
        ├── Seed 1 → Trunk → z_ij → Diffusion → Structure A
        ├── Seed 2 → Trunk → z_ij → Diffusion → Structure B
        ├── Seed 3 → Trunk → z_ij → Diffusion → Structure C
        ...
        └── Seed N → Trunk → z_ij → Diffusion → Structure N
```

Structures are ranked by confidence (pLDDT, pTM), and the best is selected. IsoDDE (Part 4) runs **1000 seeds** per prediction, demonstrating that brute-force sampling can improve results.

### 3.2 Limitations

Since $z_{ij}$ is **identical** across all seeds (same sequence, same MSA → same Trunk output), the diversity is limited to what the Diffusion Module can express with a fixed structural blueprint. In practice:

- **Local variations**: Side-chain rotamers, small loop movements — well captured
- **Global rearrangements**: Domain motions, large loop reconfigurations, apo↔holo transitions — poorly captured

The ConfBench results (29.6% for AF2-Multimer) confirm that seed variation alone is insufficient for meaningful conformational diversity.

---

## 4. Approach 2: Boltzmann Distribution Learning (BioEmu)

### 4.1 A Fundamentally Different Goal

**BioEmu** (Microsoft Research, 2024) redefines the objective: instead of predicting the single most likely structure, **sample from the equilibrium distribution** over all structures:

$$p(\mathbf{x} \mid \text{seq}) \propto \exp\left(-\frac{E(\mathbf{x})}{k_B T}\right)$$

where $E(\mathbf{x})$ is the free energy of conformation $\mathbf{x}$ and $k_B T$ is the thermal energy. This Boltzmann distribution assigns probability to every accessible conformation proportional to its thermodynamic stability — naturally producing both apo and holo states, folded and unfolded ensembles, and rare cryptic pocket openings.

```
AF3/Boltz output:                   BioEmu output:
  1 structure + confidence            10,000 independent structures
                                      → population-weighted ensemble
                                      → free energy landscape
```

### 4.2 Three-Stage Training

**Stage 1 — AFDB Pre-training**: Score matching on ~200M AlphaFold Database structures (50K sequence clusters). The model learns the general landscape of protein structure space.

**Stage 2 — MD Fine-tuning**: Fine-tuning on molecular dynamics (MD) trajectories totaling $\gt$ 200 ms of cumulative simulation time. Critically, **Markov State Model (MSM) reweighting** ensures thermodynamic correctness — raw MD frames are reweighted so the learned distribution matches the true Boltzmann distribution rather than the kinetic sampling distribution of the MD simulation.

**Stage 3 — PPFT (Property-Prediction Fine-Tuning)**: The most innovative stage. Rather than learning from structures, PPFT uses **thermodynamic measurements** (folding free energies $\Delta G$) as supervision:

$$p_{\text{folded}}^{\text{model}} = \frac{N_{\text{folded}}}{N_{\text{total}}}, \quad p_{\text{folded}}^{\text{exp}} = \frac{1}{1 + \exp(\Delta G / k_B T)}$$

$$\mathcal{L}_{\text{PPFT}} = \left| p_{\text{folded}}^{\text{model}} - p_{\text{folded}}^{\text{exp}} \right|$$

The model generates a fast ensemble (8-step sampling), classifies each structure as folded or unfolded, computes the folded fraction, and compares against the experimental value. **No structural data is needed** — only $\Delta G$ measurements from high-throughput stability assays (e.g., MEGAscale: 502,442 sequences with $\Delta G_{\text{folding}}$).

### 4.3 Results

| Capability | BioEmu Performance | Comparison |
|---|---|---|
| Free energy accuracy | MAE ~0.74 kcal/mol (fast-folding) | Equivalent to force-field differences |
| Domain motion coverage | 83% (within 3 Å RMSD of experiment) | — |
| Local unfolding | 70% folded + 81% unfolded recreation | — |
| Cryptic pocket prediction | 86% holo state prediction | vs. 56% apo |
| Computational cost | ~1 GPU-hour for 10,000 structures | 4–5 orders of magnitude cheaper than MD |

### 4.4 Limitations

BioEmu demonstrates that thermodynamically correct ensemble generation is achievable, but with significant scope restrictions:

- **Monomer only**: Single protein chains — no complexes, no ligands, no nucleic acids
- **Backbone only**: Side-chain atoms absent → ligand binding interactions unavailable
- **Ligand-blind**: Cannot model ligand-induced conformational changes (the very problem drug design needs to solve)
- **Fixed conditions**: Temperature 300K, standard pH — no condition-dependent ensemble generation

The gap between BioEmu's elegant physics and practical drug design needs (protein-ligand complexes with explicit atomic detail) remains wide.

---

## 5. Approach 3: Unified Diffusion Transformer (Vilya-1)

### 5.1 The Architectural Solution

**Vilya-1** (Baker Lab/Vilya, 2025) addresses the Frozen Trunk problem at its root: **merge the Trunk and Diffusion Module into a single unified transformer** where $z_{ij}$ is updated at every denoising step.

```
AF3/Boltz (separated):                Vilya-1 (unified):

  Trunk (1 pass)                       Unified Transformer
  ─────────────                        ───────────────────
  seq, MSA → z_ij (fixed)             For each denoising step t:
       │                                 z_ij^(t), s_i^(t) = Block(
       ▼ (read-only)                        z_ij^(t-1), s_i^(t-1),
  Diffusion (T steps)                       x_protein^(t), x_ligand^(t),
  ───────────────────                       t
  z_ij + noise → coords                 )
                                         x^(t-1) = Predict(z_ij^(t), s_i^(t))
```

### 5.2 How Dynamic $z_{ij}$ Enables Conformational Diversity

In Vilya-1, the pair representation responds to the evolving coordinates at each denoising step:

$$z_{ij}^{(t)} = \text{PairUpdate}\left(z_{ij}^{(t-1)},\; \mathbf{x}_{\text{protein}}^{(t)},\; \mathbf{x}_{\text{ligand}}^{(t)},\; t\right)$$

This creates a **feedback loop** between structural evolution and the pair representation:

```
Denoising step t:
  Ligand coordinates x_ligand^(t) approach binding site
       │
       ▼
  z_ij^(t) updates: binding site residue pairs strengthen
       │
       ▼
  Triangle propagation: distal z_ij values update
       │
       ▼
  Coordinate update: loop/domain motions respond
       │
       ▼
  New binding environment → z_ij^(t+1) reflects changes
       │
       ▼
  ... continues each step ...
```

This feedback captures **induced fit** — the mechanism by which a ligand reshapes the protein's binding site — at the representation level. In the separated AF3 architecture, this feedback is architecturally impossible.

### 5.3 Key Design Decisions

Vilya-1 makes three additional choices that maximize generality:

| Design Decision | AF3/Boltz | Vilya-1 | Rationale |
|---|---|---|---|
| **Tokenization** | Residue-level | **Atom-level** | Handles non-canonical amino acids, macrocycles |
| **Input features** | Molecule type, atom name, MSA | **Pure chemical features** (element, bond, charge, chirality) | Prevents memorization, maximizes generalization |
| **Architecture** | Separated Trunk + Diffusion | **Unified transformer** | Dynamic $z_{ij}$ for conformational diversity |

### 5.4 Performance

Vilya-1 currently specializes in **macrocyclic molecules** — cyclic peptides and non-canonical structures:

| Benchmark | Vilya-1 | Best Alternative |
|---|---|---|
| X-ray cyclic peptides (RMSD $\lt$ 1 Å) | **89.2%** | Schrödinger: 37.6% |
| Receptor-bound macrocycles | **93%** | — |
| NMR non-canonical amino acids | High accuracy | Boltz-2: crashes |
| CSD small molecules (7,192) | High | — |

### 5.5 Current Limitations

Vilya-1's macrocycle specialization means it has not yet been validated on the full scope of protein co-folding benchmarks. Extending the unified architecture to large protein complexes — where the computational cost of updating $z_{ij}$ at every diffusion step scales as $O(T \times L^2)$ instead of $O(L^2)$ — is an open engineering challenge.

---

## 6. Approach 4: Flow Matching + Polymer Prior (NP3)

### 6.1 Encoder-Decoder Separation

**NP3** (Part 3, Part 4) takes a different architectural path: instead of merging Trunk and Diffusion, it cleanly separates them into an **encoder** and **decoder** with no recycling:

```
NP3 Architecture:

  Encoder (~350M params)           Decoder (~350M params)
  ────────────────────            ────────────────────
  Seq + MSA + PLM                  Polymer prior
       │                                │
       ▼                                ▼
  PairFormer                       DiT (Flow Matching)
       │                           conditioned on z_ij
       ▼                                │
  z_ij, s_i ─────────────────────→      │
                                        ▼
                                   3D coordinates
```

While $z_{ij}$ is still fixed after the encoder (similar to AF3), NP3's design compensates through two innovations:

### 6.2 Polymer Prior

Instead of starting from Gaussian noise (AF3), NP3 initializes the flow matching process with a **physics-informed polymer prior** generated by Langevin dynamics:

$$\mathbf{x}_0 \leftarrow \mathbf{x}_0 + \delta t \cdot \mathbf{f}_{\text{drift}}(\mathbf{x}_0) + \sqrt{2 \delta t} \cdot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, I)$$

where the drift term encodes physical constraints:

$$\mathbf{f}_{\text{drift}} = 2 \cdot \mathbf{d}_{\text{bond}} + \frac{\mathbf{d}_{\text{entity}}}{r_{\text{ent}}^2} + \frac{\mathbf{d}_{\text{res}}}{r_{\text{res}}^2} - \frac{\mathbf{x}_0}{r_{\text{sphere}}^2}$$

- $\mathbf{d}\_{\text{bond}}$: harmonic bonding potential (maintains covalent connectivity)
- $\mathbf{d}\_{\text{entity}}$: entity-level clustering (keeps chains together)
- $\mathbf{d}\_{\text{res}}$: residue-level clustering (keeps atoms within residues close)
- $-\mathbf{x}\_0 / r\_{\text{sphere}}^2$: global spherical confinement

After 64 Langevin steps, the prior produces a **topologically plausible** starting point — a polymer-like coil rather than random Gaussian noise. The flow matching process then needs only to refine this into the correct fold, requiring far fewer steps (40 vs. AF3's 200).

```
Gaussian Prior (AF3):     Polymer Prior (NP3):

  Random cloud              Connected coil
  (hundreds of Å away)      (tens of Å away)
        │                         │
        │  200 steps              │  40 steps
        │  (long journey)         │  (short journey)
        ▼                         ▼
  Folded protein             Folded protein
```

### 6.3 Impact on Conformational Diversity

The polymer prior helps conformational diversity in two ways:

1. **Shorter denoising trajectory**: Less "distance" to travel means the model can explore local conformational alternatives more easily — the generative process doesn't need to reconstruct the entire fold from scratch each time.

2. **Ligand-conditioned starting point**: When the ligand is included in the input, the encoder produces $z_{ij}$ that reflects the ligand's presence. Combined with the polymer prior's physically reasonable starting configuration, the decoder can access different conformational states depending on whether the ligand is present.

NP3's ConfBench results (51.9% vs. AF2-Multimer's 29.6%) demonstrate measurable improvement, with particularly strong kinase discrimination: 77.8% for apo and 72.5% for holo predictions.

---

## 7. Approach 5: Flow Matching Fine-Tuning (AlphaFlow, PLACER)

### 7.1 AlphaFlow: Repurposing Prediction for Diversity

**AlphaFlow** (MIT, 2024) takes a pragmatic approach: **fine-tune AlphaFold2 in a flow matching framework** to generate diverse conformational ensembles:

```
AlphaFold2 (single structure)     AlphaFlow (ensemble)
────────────────────────         ──────────────────────

  Seq + MSA → AF2 → 1 structure   Seq + MSA → AF2 weights
                                        │
                                   Flow matching
                                   fine-tuning
                                        │
                                        ▼
                                   ODE integration
                                   (multiple samples)
                                        │
                                        ▼
                                   Diverse ensemble
```

By reusing AF2's learned representations (the "Keep the Trunk" pattern from Part 5) and adding a flow matching head, AlphaFlow produces conformational ensembles that correlate with MD simulation trajectories. The deterministic ODE formulation (vs. BioEmu's stochastic SDE) offers more controllable generation but potentially less diversity.

### 7.2 PLACER: Protein-Ligand Conformational Ensembles

**PLACER** (Baker Lab, 2025, *PNAS*) specifically targets the protein-ligand conformational ensemble problem:

- Generates both apo and holo ensembles for protein-small molecule systems
- Quantifies ligand-induced conformational changes
- Bridges the gap between monomer ensemble methods (BioEmu, AlphaFlow) and the practical need for complex-aware conformational sampling

PLACER demonstrates that conformational ensemble generation can be extended beyond monomers to protein-ligand systems — a critical step toward drug design applications.

---

## 8. Comparison: Five Approaches

| | Seed Variation | BioEmu | Vilya-1 | NP3 | AlphaFlow / PLACER |
|---|---|---|---|---|---|
| **Representative** | AF3, Boltz, Chai | MS Research | Baker Lab/Vilya | Iambic | MIT / Baker Lab |
| **$z_{ij}$ behavior** | Fixed | N/A (no pair repr.) | **Dynamic** (per step) | Fixed (encoder) | Fixed (AF2) |
| **Diversity source** | Random seed | Boltzmann sampling | Dynamic pair update | Polymer prior + FM | FM fine-tuning |
| **Target molecules** | All biomolecules | Monomer only | Macrocycles | All biomolecules | Monomer / Prot-Lig |
| **Ensemble quality** | Limited | **Thermodynamic** | High (for macrocycles) | Improved | MD-correlated |
| **Conformer types** | Local variations | Full ensemble | Induced fit | Apo/holo | Ensemble |
| **ConfBench** | ~30% | — | — | **52%** | — |
| **Computational cost** | $N \times$ single prediction | ~1 GPU-hr / 10K structures | Standard diffusion | 40 steps (fast) | AF2 + FM overhead |
| **Maturity** | Production | Research | Early (macrocycle) | Production | Research |

---

## 9. Convergence and Open Questions

### Where the field agrees

- **The Frozen Trunk is a real bottleneck**: ConfBench quantitatively confirms that AF3-family models struggle with conformational diversity. The architectural cause — fixed $z_{ij}$ — is well understood.
- **Thermodynamic correctness matters**: BioEmu's PPFT demonstrates that physics-based training objectives (free energy matching) produce more meaningful ensembles than geometric diversity alone.
- **Prior design impacts diversity**: NP3's polymer prior and BioEmu's MD-informed initialization both show that the starting point of the generative process significantly affects the achievable conformational range.

### What remains unresolved

**Universal conformational co-folding**: Each approach solves part of the problem but none solves all of it. BioEmu captures thermodynamic ensembles but only for monomers. Vilya-1 achieves dynamic $z_{ij}$ but only for macrocycles. NP3 handles all biomolecules but its ConfBench accuracy is only 52%. The holy grail — thermodynamically correct conformational ensembles for arbitrary protein-ligand complexes — remains unachieved.

**Scaling the unified architecture**: Vilya-1's unified diffusion transformer updates $z\_{ij}$ at every denoising step, multiplying the cost of the pair representation by the number of diffusion steps $T$. For a 1000-residue protein with $T = 200$ steps, this means $200 \times$ the pair computation of a standard separated architecture. Whether efficient approximations (e.g., updating $z\_{ij}$ only every $k$ steps, or using lightweight update mechanisms) can maintain diversity while controlling cost is an open engineering question.

**Connecting ensemble generation to drug design**: Even if we could perfectly predict conformational ensembles, the practical question remains: how do we use ensemble information to design better drugs? Scoring a candidate molecule against an ensemble of conformers, rather than a single structure, requires new computational workflows and scoring functions that the field has yet to standardize.

**Apo-holo training data scarcity**: ConfBench reveals the problem, but solving it requires training data with matched apo-holo pairs — which are rare in the PDB. Most structures are crystallized under specific conditions that favor one state. Generating synthetic apo-holo pairs through MD simulation (as in BioEmu's training) or learning the apo↔holo transformation as a conditional generation task are promising but unvalidated directions.

---

**Next: Part 7 — How Are These Models Actually Trained? Engineering, Scaling, and Training Techniques**

Papers report "trained on N GPUs for M days" — but behind that single sentence lie dozens of engineering decisions. Part 7 examines FlashAttention, distributed training strategies, mixed precision, scaling laws, and multi-stage training pipelines that make large-scale protein AI possible.

---

*Part of the series: The Technical Evolution of Protein AI — A Record of Key Design Decisions*
