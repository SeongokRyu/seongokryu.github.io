---
title: "Protein AI Series Part 1: MSA vs PLM vs Hybrid"
date: 2026-03-17 09:10:00 +0900
categories: [Drug Discovery]
tags: [protein-ai, msa, protein-language-model, alphafold, esmfold]
math: true
---

**The Technical Evolution of Protein AI — A Record of Key Design Decisions**

*This is Part 1 of a 10-part series tracing the architectural choices behind modern protein structure prediction and design models.*

---

## The Core Question

**Can a model predict protein structure from a single sequence alone, or does it need thousands of evolutionarily related sequences?**

This is the first design decision every protein structure prediction model must make: how to extract the information needed to build the pair representation $z_{ij}$ that serves as the structural blueprint (introduced in Part 0). The answer has shifted dramatically — from "MSA is essential" (AF2, 2021) to "a single sequence might suffice" (ESMFold, 2022) to "both together is best" (Chai-1, 2024).

---

## 1. MSA: What Evolution Tells Us About Structure

### 1.1 The Principle of Coevolution

A Multiple Sequence Alignment (MSA) is a matrix of evolutionarily related sequences aligned position-by-position:

```
  Position:    1  2  3  4  5  6  7  8  9  10 ...
  Query:       M  K  F  L  I  L  L  F  N  I  ...
  Homolog 1:   M  K  F  L  V  L  L  F  N  I  ...
  Homolog 2:   M  R  F  L  I  L  L  Y  N  I  ...
  Homolog 3:   M  K  Y  L  I  L  L  F  N  V  ...
  Homolog 4:   M  R  Y  L  V  L  L  Y  N  V  ...
               ...
               ↑              ↑
          Position 2      Position 8
          (K↔R vary       (F↔Y vary
           together)       together)
```

The key insight: **residues that covary across evolution tend to be spatially close in the 3D structure**. If position 2 mutates from K→R, position 8 often compensates by mutating F→Y. This **coevolution** signal arises because contacting residue pairs must maintain complementary physicochemical properties to preserve the fold.

Formally, given an MSA matrix $M \in \{1, \ldots, 20\}^{N_{\text{seq}} \times L}$, the coevolutionary signal can be quantified through the empirical covariance of amino acid frequencies. Early methods (DCA, GREMLIN, plmDCA) used this to predict contact maps — binary matrices indicating which residue pairs are within ~8 Å. AlphaFold2 dramatically advanced this by learning to extract richer pairwise features directly from the MSA through its Evoformer.

### 1.2 From MSA to Pair Representation: Outer Product Mean

In AlphaFold2, the MSA's coevolutionary information is injected into the pair representation through the **Outer Product Mean (OPM)** operation. Given the MSA representation $m \in \mathbb{R}^{N_{\text{seq}} \times L \times c_m}$:

$$\text{OPM}(m)_{ij} = \frac{1}{N_{\text{seq}}} \sum_{s=1}^{N_{\text{seq}}} \left( W_a \, m_{si} \right) \otimes \left( W_b \, m_{sj} \right)$$

where $W\_a, W\_b$ are learned linear projections and $\otimes$ denotes the outer product. The result is a $(L \times L \times c\_a \cdot c\_b)$ tensor that captures column-pair statistics from the MSA — effectively a learned, differentiable version of covariance analysis.

This OPM output is added to the pair representation $z_{ij}$ at every Evoformer block, continuously feeding evolutionary information into the structural blueprint.

### 1.3 The MSA Generation Pipeline

Constructing a high-quality MSA requires searching large sequence databases:

```
Query sequence
     │
     ├──→ JackHMMER (iterative HMM search)
     │      └──→ UniRef90 (~150M seqs), Mgnify (~600M seqs)
     │
     ├──→ HHblits (HMM-HMM search)
     │      └──→ Uniclust30 / BFD (~2.5B seqs)
     │
     └──→ MMseqs2 (fast k-mer + alignment)
            └──→ ColabFold DB
     │
     ▼
MSA matrix: (N_seq × L)
     N_seq: hundreds to tens of thousands
     L: aligned sequence length
```

The search is computationally expensive — **minutes to hours** per query depending on database size and sequence length. AlphaFold2's genetic search pipeline alone accounts for a large fraction of its total wall-clock time.

### 1.4 Limitations of MSA

MSA is powerful but not universal. Three failure modes are particularly important:

**Orphan proteins**: Proteins with few or no detectable homologs yield shallow MSAs. The coevolutionary signal degrades rapidly as $N_{\text{seq}}$ drops below ~100. De novo designed proteins, by definition, have no evolutionary history and thus no meaningful MSA.

**Antibody CDRs**: The Complementarity Determining Regions (CDRs) — especially CDR-H3 — are hypervariable by design. Somatic hypermutation generates enormous sequence diversity precisely in the regions most critical for antigen recognition. In an MSA, these positions show near-random variation, producing no useful coevolutionary signal. Yet CDR structure is what determines binding specificity.

**Computational cost**: Searching terabyte-scale databases (UniRef: ~70 GB, BFD: ~1.8 TB) requires substantial I/O and compute. For high-throughput applications — screening millions of sequences or rapid design iteration — this cost becomes prohibitive.

---

## 2. PLM: Grammar Learned from Context

### 2.1 Protein Language Models

Protein Language Models (PLMs) take a fundamentally different approach: rather than aligning a query against a database of homologs, they learn the "grammar" of protein sequences from massive unsupervised pretraining.

**ESM-2** (Meta, 2022) is the most widely adopted PLM in structure prediction:

- **Training data**: ~65M protein sequences from UniRef50
- **Training objective**: Masked Language Modeling (MLM) — mask 15% of residues, predict them from context
- **Architecture**: Transformer encoder, up to 36 layers, 3B parameters (ESM-2 3B)
- **Output**: Per-residue embedding $h_i \in \mathbb{R}^{d}$ ($d = 2560$ for ESM-2 3B)

The MLM objective forces the model to learn which amino acids are plausible at each position given the surrounding context. This implicitly captures:

- **Local structural preferences**: secondary structure propensities, turn motifs
- **Long-range contacts**: the model learns that certain positions covary, even without explicit alignment
- **Functional constraints**: active site residues, binding interfaces

Remarkably, attention maps from the middle layers of ESM-2 correlate with contact maps — the model discovers spatial proximity from sequence context alone, mirroring the coevolutionary signal that MSA provides explicitly.

### 2.2 ESMFold: Structure Prediction Without MSA

**ESMFold** (Lin et al., 2022) demonstrated that a PLM alone can drive structure prediction. It replaces the entire MSA pipeline with ESM-2:

```
                    AlphaFold2                        ESMFold
                    ──────────                        ────────
  Input:     Sequence + MSA + Templates        Sequence only
                   │                                │
  Embedding: MSA features → OPM → z_ij        ESM-2 → per-residue h_i
                   │                                │
  Trunk:     Evoformer (48 blocks × 3 recycle) Folding Trunk (48 blocks)
                   │                                │
  Output:    IPA Structure Module              IPA Structure Module
```

**Performance** (CAMEO test set, 2022):

| Model | Median lDDT | MSA Required | Inference Time |
|---|---|---|---|
| AlphaFold2 | ~0.88 | Yes (minutes–hours) | Minutes |
| ESMFold | ~0.84 | No | Seconds |

ESMFold's accuracy was slightly lower than AF2, but the key insight was clear: **a single forward pass through a pretrained PLM captures enough structural information for competitive folding**, eliminating the MSA bottleneck entirely.

The gap was most pronounced for proteins with abundant homologs (where MSA excels) and smallest for orphan proteins (where MSA provides little signal anyway).

### 2.3 Why PLMs Work: Implicit Coevolution

The connection between PLMs and coevolution can be understood through the lens of the model's attention mechanism. For a masked position $i$, the PLM must predict the amino acid using context from all other positions. This forces the model to learn:

$$p(x_i \mid x_{-i}) \propto \exp\left(\sum_j w_{ij}(x) \cdot \phi(x_j)\right)$$

where $x\_{-i}$ denotes all positions except $i$, and $w\_{ij}$ captures the pairwise coupling between positions. This is functionally analogous to the **Potts model** used in Direct Coupling Analysis (DCA), where the Boltzmann distribution over sequences is:

$$p(\mathbf{x}) = \frac{1}{Z} \exp\left(\sum_i h_i(x_i) + \sum_{i \lt j} J_{ij}(x_i, x_j)\right)$$

The coupling parameters $J_{ij}$ capture coevolution. PLMs learn an analogous (but far richer, nonlinear) coupling through their attention layers. The critical difference: **PLMs learn from the entire protein sequence universe at once**, while MSA-based coevolution is computed per-query from that query's homologs.

---

## 3. Hybrid: The Best of Both Worlds

### 3.1 Chai-1: Dual-Track MSA + PLM Integration

**Chai-1** (Chai Discovery, 2024) introduced the most successful hybrid architecture, integrating ESM-2 3B as a dedicated input track alongside MSA:

```
  ┌─────────────────────────────────┐
  │  Track 1: MSA features          │──┐
  ├─────────────────────────────────┤  │
  │  Track 2: Template features     │──┤
  ├─────────────────────────────────┤  ├──→ Trunk → Diffusion → 3D
  │  Track 3: ESM-2 3B embeddings   │──┤
  ├─────────────────────────────────┤  │
  │  Track 4: Experimental restraints│──┘
  └─────────────────────────────────┘
```

**ESM-2 integration details**:
- Uses `esm2_t36_3B_UR50D` (36 layers, 3B parameters)
- Per-residue embeddings extracted via a single forward pass for each protein chain
- Non-protein entities (DNA, RNA, ligands) receive zero embeddings
- MSA and ESM-2 tracks can be used simultaneously or independently

This design creates a natural **complementarity**:
- When MSA is abundant → MSA track dominates, ESM-2 provides redundant confirmation
- When MSA is sparse (orphan proteins) → ESM-2 compensates for the missing coevolutionary signal
- For antibody CDRs → MSA is uninformative, but ESM-2 captures sequence-context patterns

**Performance** (selected benchmarks):

| Benchmark | Chai-1 (Hybrid) | AF2.3 (MSA only) | Note |
|---|---|---|---|
| CASP15 Monomer lDDT | 0.849 | — | Top tier |
| PoseBusters Ligand | 77% | — | AF3: 76% |
| Multimer DockQ > 0.23 | **75.1%** | 67.7% | +7.4% |
| Antibody-Antigen (no MSA) | AF2.3+MSA level | Baseline | ESM-2 compensates |

The antibody result is particularly striking: **Chai-1 without MSA matched AF2.3 with MSA** on antibody-antigen complexes, directly demonstrating the PLM's value where MSA fails.

### 3.2 NeuralPLexer 3: Multi-Modal PLM Integration

**NP3** (Iambic, 2025) extends the hybrid concept to multiple molecular modalities:

```
  Protein sequences ──→ ESM-2 embeddings  ──┐
  RNA sequences ─────→ RiNALMo embeddings ──┼──→ Encoder (PairFormer)
  MSA features ──────→ U-shaped MSA Module──┘        │
                                                       ▼
                                              Decoder (Flow Matching)
                                                       │
                                                       ▼
                                                  3D coordinates
```

**Key architectural choices**:
- **ESM-2** for protein chains (same as Chai-1)
- **RiNALMo** (RNA Language Model) for RNA chains — a PLM trained on RNA sequences
- MSA processed through a **U-shaped MSA Module** (6 blocks) before entering the PairFormer encoder
- Encoder produces single ($s \in \mathbb{R}^{L \times 384}$) and pair ($z \in \mathbb{R}^{L \times L \times 128}$) representations

**Critical finding**: NP3 achieves AF3-level RNA structure prediction **using only RiNALMo, without RNA MSA**. This demonstrates that for RNA — where MSA construction is even more challenging than for proteins — a domain-specific PLM alone is sufficient.

| Modality | NP3 | AF2-M 2.3 | Note |
|---|---|---|---|
| Protein monomer (lDDT) | 87.1% | 86.2% | +0.9% |
| Protein-Protein (DockQ > 0.23) | 52.7% | 52.3% | Comparable |
| Protein-Peptide (DockQ > 0.23) | **85.1%** | 76.6% | **+8.5%** |
| RNA (CASP15, lDDT) | 46.5% | — | RiNALMo only |
| Protein-DNA (DockQ > 0.23) | 56.2% | — | — |

---

## 4. The Current Landscape: Where Each Approach Stands

### 4.1 Comparison Table

| | MSA-Dependent | PLM-Only | Hybrid |
|---|---|---|---|
| **Representative models** | AF2, AF3, Boltz-1/2 | ESMFold | Chai-1, NP3 |
| **Input** | Thousands of aligned sequences | Single sequence | Both |
| **Speed** | Minutes–hours (DB search) | Seconds (forward pass) | Seconds–minutes |
| **Homologs required** | Essential | Not needed | Beneficial but not essential |
| **Antibody CDRs** | Uninformative | Informative | PLM compensates |
| **Orphan proteins** | Degrades | Robust | Robust |
| **Information type** | Coevolution → contacts | Context → structure | Both integrated |
| **Accuracy (general)** | Highest | Slightly lower | Highest |
| **RNA support** | Requires RNA MSA | RiNALMo sufficient (NP3) | PLM sufficient |

### 4.2 The Strategic Question for Model Developers

A key open question remains: **would AF3 and Boltz-2 benefit from adding PLM embeddings?**

Neither AF3 nor Boltz-2 currently integrates a PLM. Given Chai-1's demonstrated gains — particularly on antibodies and low-homology targets — this is a clear avenue for improvement. The fact that Chai-1 is the only AF3-family model with ESM-2 integration, and that it achieves the best multimer DockQ scores, suggests this is not coincidental.

The counterargument: PLMs add significant memory and compute overhead (ESM-2 3B requires ~12 GB just for model weights). For models already operating near GPU memory limits with large pair representations, integrating a 3B-parameter PLM is non-trivial.

---

## 5. Convergence and Open Questions

### Where the field agrees

- **MSA remains the most reliable source of structural information** for well-studied protein families with abundant homologs
- **PLMs are essential** for antibodies, designed proteins, and other sequences where MSA is uninformative
- **Hybrid approaches represent the Pareto frontier** — no pure MSA or pure PLM model matches the best hybrid results

### What remains unresolved

**Can PLMs fully replace MSA?** ESMFold showed the gap is small but persistent. As PLMs scale (ESM-2 went from 650M to 3B parameters with clear improvements), this gap may close. But for now, MSA still provides information that even large PLMs miss — particularly the query-specific coevolutionary signal from the protein's own evolutionary family.

**What is the optimal PLM for structure prediction?** ESM-2 was trained with a general MLM objective. Would a PLM specifically trained for structure prediction (e.g., with structure-aware objectives) outperform it? ESM-3's multimodal training (sequence + structure + function tokens) hints at this direction, though its structure prediction performance has not surpassed specialized models.

**How should PLM embeddings be integrated architecturally?** Chai-1 uses a separate track. NP3 feeds PLM embeddings into its encoder. Are there better fusion strategies — e.g., using PLM attention maps directly as pair bias, or distilling PLM knowledge into a smaller, specialized module?

---

**Next: Part 2 — How to Reason About Residue-Pair Relationships? Evoformer → Pairformer → Pairmixer**

We examine the Trunk — the computational core where pair and single representations are refined — and trace how its architecture evolved from AF2's Evoformer through AF3's Pairformer to the attention-free Pairmixer.

---

*Part of the series: The Technical Evolution of Protein AI — A Record of Key Design Decisions*
