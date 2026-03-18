---
title: "Protein AI Series Part 0: What Is Protein Structure Prediction?"
date: 2026-03-17 09:00:00 +0900
categories: [Drug Discovery, Foundation Model]
tags: [protein-ai, structure-prediction, alphafold, deep-learning]
math: true
---

**The Technical Evolution of Protein AI — A Record of Key Design Decisions**

*This is Part 0 of a 10-part series tracing the architectural choices behind modern protein structure prediction and design models.*

---

## Introduction

Every protein in biology begins as a linear chain of amino acids. Yet this one-dimensional sequence encodes a three-dimensional structure that determines the protein's function — catalyzing reactions, transducing signals, recognizing pathogens. The central question of structural biology has long been: **given only the amino acid sequence, can we predict the 3D structure?**

This series traces how AI models have answered that question — and how their answers have evolved from AlphaFold2 (2021) to the latest generation of co-folding and design models in 2025–2026. Each installment examines a specific architectural decision point and compares how different model families responded. Before diving into those comparisons, this Part 0 establishes the shared vocabulary and conceptual framework that every subsequent post builds on.

---

## 1. Protein Structure Fundamentals

### 1.1 Amino Acids and the Peptide Bond

Proteins are linear polymers of amino acids. Nature uses 20 standard amino acid types, each sharing an identical backbone but carrying a distinct side-chain (R group):

```
        H    O
        |    ‖
   H₂N─Cα──C─OH
        |
        R  (side-chain, unique per amino acid type)
```

Adjacent amino acids are linked by a **peptide bond** between the carbonyl carbon of residue $i$ and the amide nitrogen of residue $i+1$:

```
   ...─[NH─Cα─C(═O)]─[NH─Cα─C(═O)]─[NH─Cα]─...
         residue i       residue i+1     residue i+2
```

The repeating N–Cα–C pattern forms the **backbone**. The variable R groups attached at each Cα are the **side-chains**. Every residue contributes four backbone heavy atoms: N, Cα, C, and O.

### 1.2 Torsion Angles: The Internal Coordinates of Structure

Rather than specifying Cartesian coordinates for every atom, protein structure can be compactly described by **torsion (dihedral) angles** — the rotation around each covalent bond.

**Backbone torsion angles** (three per residue):

$$\phi_i: \quad \text{C}_{i-1} - \text{N}_i - \text{C}_{\alpha,i} - \text{C}_i \qquad \in [-\pi, +\pi]$$

$$\psi_i: \quad \text{N}_i - \text{C}_{\alpha,i} - \text{C}_i - \text{N}_{i+1} \qquad \in [-\pi, +\pi]$$

$$\omega_i: \quad \text{C}_{\alpha,i} - \text{C}_i - \text{N}_{i+1} - \text{C}_{\alpha,i+1} \qquad (\approx \pi, \text{trans})$$

Since bond lengths and bond angles are nearly fixed across all proteins, specifying $(\phi, \psi)$ for each residue (with $\omega \approx \pi$) is sufficient to reconstruct the full backbone trace.

**Side-chain torsion angles** $\chi_1, \chi_2, \chi_3, \chi_4$ (0–4 per residue depending on amino acid type) describe the rotational state of the side-chain. Discretized combinations of $\chi$ angles are called **rotamers**.

The **Ramachandran plot** — a 2D scatter of $(\phi, \psi)$ values — reveals that only certain regions of torsion-angle space are sterically permitted:

```
  ψ
  ↑
  |   ■■■■              ← β-sheet region (φ ≈ -120°, ψ ≈ +130°)
  |   ■■■■
  |
  |         ■■■■        ← α-helix region (φ ≈ -60°, ψ ≈ -45°)
  |         ■■■■
  +──────────────→ φ
```

Structure prediction models must produce $(\phi, \psi)$ pairs that fall within these physically allowed regions.

### 1.3 Secondary Structure

Local, repeating backbone conformations give rise to **secondary structure** elements:

| Element | Hydrogen Bond Pattern | Geometry | Residues per Turn |
|---|---|---|---|
| **α-helix** | C=O(i) ↔ N-H(i+4) | Right-handed helix, rise 5.4 Å | 3.6 |
| **β-sheet** | Inter-strand C=O ↔ N-H | Extended strands, parallel or antiparallel | — |
| **Loop / Coil** | Irregular | Flexible, often solvent-exposed | — |

Loops are the most structurally diverse and hardest to predict. Antibody CDRs (Complementarity Determining Regions), the antigen-binding loops, are canonical examples of structurally variable loops.

### 1.4 Tertiary, Quaternary Structure, and Rigid Body Frames

**Tertiary structure** is the full 3D arrangement of a single polypeptide chain — secondary structure elements packed together through hydrophobic interactions, disulfide bonds, hydrogen bonds, and ionic interactions.

**Quaternary structure** arises when multiple chains assemble into a complex (e.g., an antibody: 2 heavy + 2 light chains).

A key representational concept introduced by AlphaFold2 is the **rigid body frame**. Each residue is treated as a rigid body defined by:

$$T_i = (R_i, \vec{t}_i), \qquad R_i \in \mathrm{SO}(3),\ \vec{t}_i \in \mathbb{R}^3$$

where $R_i$ is a $3 \times 3$ rotation matrix encoding the residue's orientation, and $\vec{t}_i$ is a translation vector encoding its position. The frame is constructed from the three backbone atoms N, $\text{C}_\alpha$, C using a Gram-Schmidt-like procedure. Side-chain atoms are then placed relative to this frame via the predicted $\chi$ angles.

---

## 2. The Sequence-to-Structure Problem

### 2.1 Levinthal's Paradox

A protein of $L$ residues has roughly $2L$ backbone torsion degrees of freedom ($\phi, \psi$ per residue). Even discretizing each angle into just 3 states yields $3^{2L}$ possible conformations. For a modest 100-residue protein:

$$3^{200} \approx 10^{95}$$

Exhaustive enumeration is computationally intractable — yet real proteins fold on the millisecond-to-second timescale. This is **Levinthal's paradox**: the conformational search space is astronomically large, but nature navigates it efficiently through the energy landscape encoded by the sequence.

### 2.2 Experimental Structure Determination

Before AI, 3D structures were determined experimentally:

| Method | Resolution | Throughput | Limitations |
|---|---|---|---|
| **X-ray crystallography** | ~1–2 Å | Months–years | Requires crystallization; bias toward rigid, ordered structures |
| **Cryo-EM** | ~2–4 Å | Weeks–months | Large complexes preferred; resolution varies across map |
| **NMR spectroscopy** | ~2–5 Å | Months | Limited to small proteins (~30 kDa) |

The Protein Data Bank (PDB) contains ~220K experimentally determined structures — a tiny fraction of the ~250M+ known protein sequences. This gap between sequence space and structure space is the fundamental motivation for computational structure prediction.

### 2.3 Why AI Succeeded Where Physics Struggled

Classical molecular dynamics (MD) simulates atomic forces at femtosecond ($10^{-15}$ s) timesteps. Folding a 100-residue protein (folding time ~ms) requires $\sim 10^{12}$ steps — feasible on specialized hardware (e.g., Anton) but impractical at scale. Moreover, force field inaccuracies accumulate over long trajectories.

AI approaches sidestep direct physics simulation by learning the **sequence → structure mapping** from experimental data. The key insight: evolution has already explored sequence-structure relationships across billions of years. By analyzing patterns in known sequences and structures, neural networks can infer the mapping without simulating the physical folding process.

---

## 3. The Common Architecture: Trunk + Structure Generation

Nearly every modern protein structure prediction model shares a two-stage architecture. Understanding this shared blueprint is essential for following the rest of this series.

```
┌───────────────────────────────────────────────────────────────┐
│  INPUTS                                                       │
│    Sequence → one-hot / PLM embedding                         │
│    MSA → coevolutionary statistics                            │
│    Templates → known similar structures (optional)            │
└──────────────────────────┬────────────────────────────────────┘
                           ↓
┌───────────────────────────────────────────────────────────────┐
│  INPUT EMBEDDING                                              │
│    Sequence → s_i ∈ R^{c_s}  (single representation)         │
│    MSA → Outer Product Mean → z_ij ∈ R^{c_z}  (pair repr.)  │
│    Templates → pair features added to z_ij                    │
└──────────────────────────┬────────────────────────────────────┘
                           ↓
┌───────────────────────────────────────────────────────────────┐
│  TRUNK  (the "representation engine")                         │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Evoformer (AF2) / Pairformer (AF3, Boltz) / ...       │  │
│  │  × N blocks (48–64)                                     │  │
│  │                                                         │  │
│  │  Iteratively refines:                                   │  │
│  │    z_ij : pair representation  (L × L × c_z)           │  │
│  │    s_i  : single representation (L × c_s)              │  │
│  │                                                         │  │
│  │  × R recycling iterations (typically R = 3)             │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                               │
│  Output: z_trunk, s_trunk                                     │
└──────────────────────────┬────────────────────────────────────┘
                           ↓
┌───────────────────────────────────────────────────────────────┐
│  STRUCTURE GENERATION MODULE                                  │
│                                                               │
│    AF2:        IPA Structure Module  (deterministic, 8 steps) │
│    AF3/Boltz:  Diffusion Module      (stochastic, 200 steps)  │
│    NP3:        Flow Matching Decoder (stochastic, 40 steps)   │
│    BioEmu:     Langevin Dynamics     (stochastic, ~500 steps) │
│                                                               │
│  Output: 3D atomic coordinates                                │
└──────────────────────────┬────────────────────────────────────┘
                           ↓
┌───────────────────────────────────────────────────────────────┐
│  OUTPUT HEADS                                                 │
│    Confidence:  pLDDT, pTM, pAE, pDE                         │
│    (Optional)   Affinity: K_d, ΔΔG                           │
│    (Optional)   Distogram: pairwise distance distributions    │
└───────────────────────────────────────────────────────────────┘
```

### 3.1 Pair Representation $z_{ij}$

The pair representation is an $L \times L \times c_z$ tensor (typically $c_z = 128$) where each entry $z_{ij} \in \mathbb{R}^{c_z}$ encodes the relationship between residues $i$ and $j$ — their spatial proximity, relative orientation, and interaction type.

**Initialization** combines multiple signals:
- **Relative positional encoding**: function of $\|i - j\|$
- **Outer product mean** from MSA: column-pair statistics capturing coevolutionary signal
- **Template pair features**: Cα–Cα distances from known similar structures

After the Trunk refines $z_{ij}$ through dozens of layers, it serves as a **structural blueprint**: "$z_{ij}$ encodes that residues $i$ and $j$ should be within 5 Å, oriented in a particular way." The Structure Generation Module reads this blueprint to produce 3D coordinates.

**Memory cost** scales quadratically:

$$\text{Memory} = L^2 \times c_z \times \text{sizeof(dtype)}$$

For $L = 2000$, $c_z = 128$, BF16: $2000^2 \times 128 \times 2 \approx 976\;\text{MB}$ — just for the pair representation alone. This $O(L^2)$ scaling is the primary bottleneck limiting the size of structures that can be processed.

### 3.2 Single Representation $s_i$

The single representation $s \in \mathbb{R}^{L \times c_s}$ (typically $c_s = 384$) captures per-residue properties: amino acid identity, local structural environment, and aggregated information from MSA columns or PLM embeddings.

In the Pairformer architecture (AF3, Boltz-2), $s_i$ is updated via **Attention with Pair Bias**:

$$\text{Attn}(Q, K, V, z) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}} + W_z \cdot z_{ij}\right) V$$

where $Q = W_q s$, $K = W_k s$, $V = W_v s$. The pair representation $z_{ij}$ acts as a learned bias on the attention weights, injecting structural context directly into the self-attention over residues.

### 3.3 The Trunk: Where Representations Are Built

The Trunk is the computational core of every model — the module that transforms raw input features into rich structural representations $(s, z)$. Its design has been the primary axis of architectural innovation:

| Generation | Trunk Architecture | Key Feature |
|---|---|---|
| AF2 (2021) | Evoformer | MSA + pair jointly processed |
| AF3 (2024) | Pairformer | MSA separated; cleaner pair-only trunk |
| Pairmixer (2026) | Attention-free | Triangle multiplication only; 4× faster |

**Part 2** of this series examines this evolution in detail.

### 3.4 Structure Generation: From Deterministic to Generative

The Structure Generation Module converts the Trunk's output $(s, z)$ into 3D atomic coordinates. This is where the most dramatic paradigm shifts have occurred:

| Paradigm | Representative | Mechanism | Nature |
|---|---|---|---|
| IPA | AF2 | SE(3)-equivariant point attention over rigid frames | Deterministic |
| EDM Diffusion | AF3, Boltz-1/2, Chai-1 | Iterative denoising: $D_\theta(x_t, \sigma; z) \approx \mathbb{E}[x_0 \mid x_t]$ | Stochastic |
| SE(3) Diffusion | RFdiffusion, FrameDiff | Noise on frames $(R, t)$ in SE(3) | Stochastic |
| Flow Matching | NP3, Proteina, FrameFlow | ODE: $\frac{dx}{dt} = v_\theta(x, t)$, linear interpolation path | Stochastic/ODE |
| Langevin | BioEmu | Score-based sampling from $p(x \mid \text{seq}) \propto e^{-E(x)/kT}$ | Stochastic |

**Part 3** dives deep into these five paradigms.

---

## 4. Evaluation Metrics

To compare models, the field relies on a standard set of metrics. Understanding these is essential for interpreting benchmarks throughout the series.

### 4.1 Structural Accuracy Metrics

**RMSD (Root Mean Square Deviation)** measures the average atomic displacement between predicted and true structures after optimal superposition:

$$\text{RMSD} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \| \vec{x}_i^{\,\text{pred}} - \vec{x}_i^{\,\text{true}} \|^2}$$

Typically computed over Cα atoms. Interpretation: $< 1$ Å excellent, $< 2$ Å good, $< 5$ Å approximate fold correct.

**lDDT (Local Distance Difference Test)** evaluates local structural accuracy without requiring global superposition. For all pairs of atoms within 15 Å in the true structure, it checks whether their predicted distances are preserved within tolerance thresholds:

$$\text{lDDT} = \frac{1}{|\mathcal{P}|} \sum_{(i,j) \in \mathcal{P}} \frac{1}{4} \sum_{\tau \in \{0.5, 1, 2, 4\}} \mathbb{1}\!\left[ |d_{ij}^{\text{pred}} - d_{ij}^{\text{true}}| < \tau \right]$$

where $\mathcal{P}$ is the set of atom pairs within 15 Å in the reference. Range: $[0, 1]$, higher is better. Unlike RMSD, lDDT is robust for proteins with flexible domains.

**TM-score (Template Modeling score)** provides a length-normalized measure of global structural similarity:

$$\text{TM-score} = \frac{1}{L} \sum_{i=1}^{L_{\text{aligned}}} \frac{1}{1 + (d_i / d_0)^2}$$

where $d_0 = 1.24 \sqrt[3]{L - 15} - 1.8$ is a length-dependent normalization. Range: $(0, 1]$; $> 0.5$ indicates the same fold, $> 0.7$ high similarity.

### 4.2 Complex / Docking Metrics

**DockQ** is a composite score for protein–protein interface quality, combining:
- $F_{\text{nat}}$: fraction of native contacts preserved
- $L_{\text{rms}}$: ligand RMSD (after interface alignment)
- $i_{\text{rms}}$: interface RMSD

Range: $[0, 1]$; $> 0.23$ acceptable, $> 0.49$ medium, $> 0.80$ high quality.

### 4.3 Model Confidence Scores

Modern models output self-assessed confidence alongside their predictions:

| Score | What It Estimates | Range | Interpretation |
|---|---|---|---|
| **pLDDT** | Per-residue lDDT | [0, 100] | > 90 very high; < 50 likely disordered |
| **pTM** | Global TM-score | [0, 1] | Overall fold confidence |
| **pAE** | Pairwise aligned error | Å | Inter-chain relative positioning |
| **pDE** | Pairwise distance error | Å | Atom-pair distance accuracy |

Confidence scores are critical for **ranking**: models like AF3 and Boltz-2 generate multiple structures with different random seeds, then rank them by confidence to select the best prediction.

### 4.4 Conformational Diversity Metrics

**ConfBench** (introduced by NP3) evaluates a model's ability to distinguish apo (ligand-free) and holo (ligand-bound) conformations:

$$\text{score} = \frac{\text{RMSD}_{\text{alt}} - \text{RMSD}_{\text{ref}}}{\sqrt{(\text{RMSD}_{\text{alt}}^2 + \text{RMSD}_{\text{ref}}^2 + \text{RMSD}_{\text{ref-alt}}^2) / 2}}$$

where $+1$ means the correct state is predicted, $0$ means indistinguishable, and $-1$ means the wrong state. Current results: AF2-Multimer ~30% (near random), NP3 ~52% — conformational diversity remains an open challenge (**Part 6**).

---

## 5. Series Roadmap

This post established the foundation. The remaining parts each pose a core architectural question and trace how models have diverged in answering it:

```
Parts 1–3: How has the prediction pipeline evolved?

  Part 1 ─ How to read evolutionary information?
            MSA vs PLM vs Hybrid

  Part 2 ─ How to reason about residue-pair relationships?
            Evoformer → Pairformer → Pairmixer

  Part 3 ─ How to output 3D structure?
            IPA → Diffusion → Flow Matching


Parts 4–5: From prediction to design

  Part 4 ─ Proteins only, or all biomolecules?
            The rise of co-folding + open-source competition

  Part 5 ─ How to turn a prediction model into a design model?
            Four strategies compared


Parts 6–7: Open problems and the future

  Part 6 ─ One structure or many?
            The conformational diversity problem

  Part 7 ─ Where is the field heading?
            Points of convergence and unsolved problems
```

Each post follows a consistent structure: a core question → why it matters → how different models answered differently → a comparison table → current consensus and remaining open questions.

---

**Next: Part 1 — How to Read Evolutionary Information? MSA vs PLM vs Hybrid**

We examine the first major design decision: should a model rely on multiple sequence alignments, protein language models, or both?

---

*Part of the series: The Technical Evolution of Protein AI — A Record of Key Design Decisions*
