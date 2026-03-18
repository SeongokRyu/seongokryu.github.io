---
title: "Protein AI Series Part 2: Evoformer → Pairformer → Pairmixer"
date: 2026-03-17 09:20:00 +0900
categories: [Drug Discovery, Foundation Model]
tags: [protein-ai, evoformer, pairformer, pairmixer, triangle-attention]
math: true
---

**The Technical Evolution of Protein AI — A Record of Key Design Decisions**

*This is Part 2 of a 10-part series tracing the architectural choices behind modern protein structure prediction and design models. This is the most technically dense installment — it covers the computational core of every modern protein AI model.*

---

## The Core Question

**What is the optimal way to update the pair representation $z_{ij}$?**

In Part 0, we introduced the pair representation — an $L \times L \times c\_z$ tensor encoding the relationship between every residue pair. In Part 1, we discussed how MSA and PLM provide the initial signal. This Part examines the **Trunk**: the module that iteratively refines $z\_{ij}$ (and the single representation $s\_i$) through dozens of layers until it contains enough structural information to guide 3D coordinate generation.

The Trunk's architecture has been the primary axis of innovation: from AlphaFold2's Evoformer (2021) through AlphaFold3's Pairformer (2024) to the attention-free Pairmixer (ICLR 2026). Understanding this evolution reveals what actually matters for protein structure prediction — and what turns out to be surprisingly unnecessary.

---

## 1. Why Triangles? The Geometric Intuition

Before examining specific architectures, we need to understand the geometric principle that underlies all of them.

In Euclidean space, any three points $i$, $j$, $k$ satisfy the **triangle inequality**:

$$d(i,j) \leq d(i,k) + d(k,j)$$

For protein structure prediction, this means: **the relationship between residues $i$ and $j$ is constrained by their respective relationships with every other residue $k$**. If we know that residue $i$ is close to residue $k$, and residue $k$ is close to residue $j$, we can infer that $i$ and $j$ are likely within a bounded distance.

This principle motivates the core operation shared across all Trunk architectures: **updating $z_{ij}$ by aggregating information through intermediary residues $k$**:

```
           k
          / \
    z_ik /   \ z_jk
        /     \
       i ───── j
         z_ij

"Update z_ij using information from z_ik and z_jk for all k"
```

The different Trunk architectures implement this principle through two distinct operations — triangle multiplication and triangle attention — and, as we will see, one of them turns out to be dispensable.

---

## 2. Triangle Multiplication and Triangle Attention

### 2.1 Triangle Multiplication

Triangle multiplication updates $z_{ij}$ by aggregating element-wise products of pair features through all intermediary residues $k$:

**Outgoing edges** (aggregate over shared starting node $i$ and $j$'s neighbors):

$$z_{ij} \leftarrow \sum_k a_{ik} \odot b_{jk}$$

**Incoming edges** (aggregate over shared ending node):

$$z_{ij} \leftarrow \sum_k a_{ki} \odot b_{kj}$$

where $a = W\_a \cdot z$, $b = W\_b \cdot z$ are learned linear projections with gating, and $\odot$ denotes element-wise multiplication.

In practice, this is implemented as a single **einsum** operation:

```python
# Outgoing triangle multiplication
a = linear_a(z)          # (L, L, D)
b = linear_b(z)          # (L, L, D)
out = einsum("bikd, bjkd -> bijd", a, b)   # sum over k
z = z + linear_out(out)
```

**Complexity**: $O(L^3 \cdot C\_z)$ time, $O(L^2 \cdot C\_z)$ memory.

The einsum maps directly to batched matrix multiplication (GEMM) — one of the most optimized operations on modern GPUs. This hardware alignment is a key advantage of triangle multiplication.

### 2.2 Triangle Attention

Triangle attention applies standard self-attention along the rows (starting node) or columns (ending node) of the pair representation, using $z$ itself as an attention bias:

**Starting node** (attention along rows — for each fixed $i$, attend across all $j$):

$$z_{ij} \leftarrow \text{Attention}(Q = z_{ij},\; K = z_{ik},\; V = z_{ik},\; \text{bias} = z_{jk}) \quad \forall k$$

**Ending node** (attention along columns — for each fixed $j$, attend across all $i$):

$$z_{ij} \leftarrow \text{Attention}(Q = z_{ij},\; K = z_{kj},\; V = z_{kj},\; \text{bias} = z_{ki}) \quad \forall k$$

This performs **$L$ independent attention operations** — one per row (or column) of the pair matrix, each over sequences of length $L$.

**Complexity**: $O(L^3 \cdot H \cdot D)$ time, $O(L^3 \cdot H)$ memory (for attention maps).

### 2.3 The Critical Difference: Hardware Efficiency

Both operations are $O(L^3)$ in time, but they differ dramatically in memory and GPU utilization:

| Metric | Triangle Multiplication | Triangle Attention |
|---|---|---|
| Time complexity | $O(L^3 \cdot C_z)$ | $O(L^3 \cdot H \cdot D)$ |
| Memory complexity | $O(L^2 \cdot C_z)$ | $O(L^3 \cdot H)$ |
| Implementation | Single einsum (batched GEMM) | $L$ independent attention ops |
| GPU utilization | High (maps to matmul) | Low (kernel launch overhead) |

The memory difference is stark. For $L = 1024$, $H = 4$, FP16:
- Triangle attention maps alone: $L^2 \times L \times H \times 2 = 8$ GB
- With 48 layers and gradients: hundreds of GB required

This memory footprint **directly determines the maximum processable sequence length** and is the primary reason triangle attention is the bottleneck.

---

## 3. Evoformer (AlphaFold2, 2021)

AlphaFold2's Evoformer processes three representations simultaneously: the MSA representation $m$, the pair representation $z$, and the single representation $s$ (extracted from the first row of $m$).

### 3.1 Block Structure

One Evoformer block consists of:

```
┌─────────────────────────────────────────────────────┐
│  Evoformer Block                                    │
│                                                     │
│  MSA track:                                         │
│    m → Row-wise Attention (with pair bias z) → m    │
│    m → Column-wise Attention → m                    │
│    m → Transition (FFN) → m                         │
│                                                     │
│  MSA → Pair:                                        │
│    m → Outer Product Mean → Δz                      │
│    z = z + Δz                                       │
│                                                     │
│  Pair track:                                        │
│    z → Triangle Multiplication (outgoing) → z       │
│    z → Triangle Multiplication (incoming) → z       │
│    z → Triangle Attention (starting node) → z       │
│    z → Triangle Attention (ending node) → z         │
│    z → Pair Transition (FFN) → z                    │
│                                                     │
└─────────────────────────────────────────────────────┘

× 48 blocks × 3 recycling iterations = 144 total passes
```

The defining feature of Evoformer is the **tight coupling between MSA and pair processing**. At every block, information flows from MSA to pair representation through the Outer Product Mean (see Part 1), continuously feeding coevolutionary signal into the structural blueprint. The MSA itself is refined through row-wise and column-wise attention, allowing the model to learn which sequences and which positions are most informative.

### 3.2 Strengths and Limitations

**Strengths**:
- Joint MSA-pair processing extracts maximum information from evolutionary data
- Recycling enables long-range information propagation across the pair matrix

**Limitations**:
- MSA processing inside the Trunk makes it **tightly coupled to protein-specific inputs** — extending to ligands, nucleic acids, or other molecular types requires rethinking the entire MSA track
- The MSA track is computationally expensive: row-wise attention is $O(N\_{\text{seq}} \cdot L \cdot d)$ per head, column-wise is $O(N\_{\text{seq}} \cdot L)$
- Total parameter count and FLOP budget are dominated by MSA processing

### 3.3 OpenFold: The Open-Source Foundation

**OpenFold** (Columbia University et al., 2022, Apache 2.0) provided a complete open-source reimplementation of AlphaFold2:

- Full training code and weights publicly available
- **First application of FlashAttention** to protein structure prediction — enabling significant memory savings
- Compatible with AF2 weights — can reproduce identical predictions
- Became the foundation for subsequent research: AlphaFlow, various fine-tuning studies, and architectural explorations

OpenFold's significance extends beyond AF2 reproduction. By making the full training pipeline accessible, it enabled the community to experiment with AF2's architecture — experiments that ultimately led to insights like those in RF2 and Pairmixer.

### 3.4 UniFold: Another AF2 Reproduction

**UniFold** (DP Technology, 2022, Apache 2.0) provided an independent open-source reimplementation of AF2's Evoformer architecture:

- Full training code and pretrained weights publicly available
- Demonstrated that AF2's Evoformer could be faithfully reproduced outside DeepMind
- Served as the foundation for subsequent models in the DP Technology ecosystem (e.g., Uni-Fold Symmetry for symmetric complexes)

Together with OpenFold, UniFold established that the Evoformer architecture was fully reproducible — an important validation that the published AF2 methodology was complete and that the community could build on it independently.

---

## 4. Pairformer (AlphaFold3 / Boltz, 2024)

### 4.1 The Key Change: Separating MSA from the Trunk

AlphaFold3's Pairformer makes one architectural decision that changes everything: **move MSA processing outside the Trunk**.

```
AlphaFold2:                          AlphaFold3:

  MSA + Pair ──→ Evoformer ──→ z,s     MSA ──→ MSA Module ──→ initial z
  (jointly processed)                         (separate)      │
                                        Seq ──→ Input Embed ──→ initial s
                                                               │
                                                    z,s ──→ Pairformer ──→ z,s
                                                    (pair + single only)
```

The MSA is processed once in a dedicated MSA Module that produces an initial pair representation. The Pairformer Trunk then refines only the pair and single representations — it never sees the raw MSA.

### 4.2 Block Structure

One Pairformer block:

```
┌─────────────────────────────────────────────────────┐
│  Pairformer Block                                   │
│                                                     │
│  Pair track:                                        │
│    z → Triangle Multiplication (outgoing) → z       │
│    z → Triangle Multiplication (incoming) → z       │
│    z → Triangle Attention (starting node) → z       │
│    z → Triangle Attention (ending node) → z         │
│    z → Pair Transition (FFN) → z                    │
│                                                     │
│  Single track:                                      │
│    s → Attention with Pair Bias (s, z) → s          │
│    s → Single Transition (FFN) → s                  │
│                                                     │
└─────────────────────────────────────────────────────┘

× 48 blocks (AF3) or 64 blocks (Boltz-2) × 3 recycling
```

### 4.3 Why This Change Matters

**Modularity**: By decoupling MSA processing from the Trunk, the Pairformer can accept pair features from any source — MSA, PLM embeddings, template features, or even a combination. This is precisely what enabled Chai-1's multi-track input architecture (Part 1).

**Non-protein molecules**: Ligands, nucleic acids, and ions have no MSA. In Evoformer, they would need special handling within the MSA track. In Pairformer, they simply enter through the pair and single representations — the Trunk is agnostic to the input source.

**Simplicity**: Removing the MSA track eliminates roughly half the operations per block. The Pairformer block is conceptually cleaner: update pair features via triangle operations, then update single features informed by the pair.

### 4.4 Adoption

Pairformer became the **de facto standard** after AF3. Every major co-folding model adopted it:
- **Boltz-1/2**: Pairformer with 64 blocks
- **Chai-1**: Evoformer-like variant but with MSA separated
- **NP3**: PairFormer in the encoder
- **Protenix**: Pairformer (AF3 reproduction)
- **SeedFold**: Pairformer with linear triangle attention and wider representations

---

## 5. RoseTTAFold's Independent Path

While the AF2 → AF3 lineage refined the Evoformer into the Pairformer, the Baker Lab pursued a fundamentally different architectural philosophy.

### 5.1 RF1: Three-Track Architecture

RoseTTAFold (RF1, 2021) introduced a **three-track architecture** that processes 1D (sequence), 2D (pairwise), and 3D (coordinate) information simultaneously:

```
┌────────────────────────────────────────────────────┐
│  1D Track (Sequence)                               │
│    Amino acid patterns, MSA column features        │
│         ↕                                          │
│  2D Track (Pairwise)          ← Biaxial Attention  │
│    Residue-pair interactions, distance maps         │
│         ↕                                          │
│  3D Track (Coordinates)       ← SE(3) Transformer  │
│    Backbone coordinates, spatial geometry           │
└────────────────────────────────────────────────────┘
    All three tracks exchange information bidirectionally
```

Crucially, RF1 used **biaxial attention** on the 2D track instead of triangle attention/multiplication. Biaxial attention applies standard attention along rows and columns alternately — a simpler mechanism that nonetheless captured pairwise relationships effectively.

The 3D track maintained actual coordinates throughout the network, updated via SE(3)-equivariant transformers. This stands in contrast to AF2, where 3D coordinates only appear in the final Structure Module.

### 5.2 RF2: The Precursor Observation

RoseTTAFold2 (RF2, 2023) extended the three-track design with FAPE loss, recycling, and distillation from AF2 predictions. Its most important contribution was an empirical observation:

> *"Excellent performance can be achieved without hallmark features of AlphaFold2 — invariant point attention and triangle attention — indicating that these are not essential for high accuracy prediction."*

RF2 replaced triangle attention with **structure-biased attention** — a simpler variant that uses the 3D track's coordinates to bias attention weights. This achieved AF2-level accuracy on monomers and AF2-Multimer-level on complexes.

This finding was a **precursor** to Pairmixer's more rigorous demonstration: if triangle attention is not essential, what is the minimal set of operations needed?

---

## 6. SeedFold (ByteDance, 2025): Efficient Triangle Attention via Linearization

While Pairmixer would later propose removing triangle attention entirely, **SeedFold** (ByteDance, 2025) took a different approach: **make triangle attention efficient rather than eliminate it**.

### 6.1 Linear Triangle Attention

SeedFold replaces the standard $O(L^3)$ triangle attention with a **linear attention** variant that reduces the complexity to sub-cubic scaling. Standard triangle attention computes full $L \times L$ attention maps for each of $L$ rows (or columns):

$$\text{TriAtt}(z)_{ij} = \sum_k \text{softmax}_k(Q_{ij} K_{ik}^T / \sqrt{d}) \cdot V_{ik}$$

SeedFold approximates this via kernel-based linear attention, avoiding the explicit materialization of the $L \times L$ attention matrix:

$$\text{LinearTriAtt}(z)_{ij} \approx \frac{\sum_k \phi(Q_{ij}) \phi(K_{ik})^T V_{ik}}{\sum_k \phi(Q_{ij}) \phi(K_{ik})^T}$$

where $\phi(\cdot)$ is a feature map (e.g., ELU + 1) that enables the computation to be rewritten as matrix multiplications with $O(L^2 \cdot d^2)$ complexity instead of $O(L^3 \cdot d)$.

### 6.2 Wider Pairformer Representations

Beyond efficiency, SeedFold explores **widening the pair representation** — increasing the channel dimension $C_z$ beyond AF3's default of 128. This gives the model more capacity per layer to encode complex pairwise relationships, trading compute per layer for improved representation quality.

### 6.3 Results

SeedFold reports performance **exceeding AF3 on most protein-related tasks**, demonstrating that the Pairformer architecture with efficient triangle attention and wider representations can surpass the original AF3.

### 6.4 SeedFold vs. Pairmixer: Two Philosophies

SeedFold and Pairmixer represent two contrasting responses to the same bottleneck:

| | SeedFold | Pairmixer |
|---|---|---|
| **Strategy** | Make triangle attention efficient | Remove triangle attention entirely |
| **Triangle Attention** | Linearized (sub-cubic) | Removed |
| **Triangle Multiplication** | Retained | Retained |
| **Pair dim** | Wider ($C_z \gt 128$) | Standard |
| **Philosophy** | "Attention is useful if made efficient" | "Attention is unnecessary" |

Both approaches confirm that standard $O(L^3)$ triangle attention is the bottleneck, but disagree on whether the solution is optimization or elimination.

---

## 7. Pairmixer (ICLR 2026): Triangle Multiplication Is All You Need

### 6.1 The Hypothesis

Pairmixer asks a simple question: **which components of the Pairformer block are actually necessary?**

Starting from the Pairformer block, the authors systematically ablated components:

| Configuration | Removed | lDDT | GPU-days |
|---|---|---|---|
| Full Pairformer (baseline) | — | 0.74 | 82 |
| − Sequence Attention (with Pair Bias) | Single update | 0.73 | 80 |
| − Triangle Attention | Row/col attention on pair | 0.70 | 66 |
| − Triangle Multiplication | Outgoing/incoming mult | 0.70 | 71 |
| **Pairmixer (− TriAtt − SeqAtt)** | Both attention types | **0.78** | **269** (large) |

*Note: The ablation was performed on small 12-layer models. The final Pairmixer result (lDDT 0.78) is from the large model trained with more compute, matching the full Pairformer.*

### 6.2 What Pairmixer Removes

```
Pairformer Block:                    Pairmixer Block:

  z → TriMult (out)    → z           z → TriMult (out)    → z
  z → TriMult (in)     → z           z → TriMult (in)     → z
  z → TriAtt (start)   → z           ╳ REMOVED
  z → TriAtt (end)     → z           ╳ REMOVED
  z → Pair FFN         → z           z → Pair FFN         → z
  s → Att + PairBias   → s           ╳ REMOVED (s = s_init)
  s → Single FFN       → s           ╳ REMOVED
```

Pairmixer retains **only triangle multiplication and feed-forward networks**. The single representation $s$ is not updated at all within the Trunk — it remains fixed at its initial embedding value $s_{\text{init}}$.

### 6.3 Results

On the full-scale model:

| Metric | Pairformer | Pairmixer | Difference |
|---|---|---|---|
| Mean lDDT | 0.78 | 0.78 | **Identical** |
| DockQ > 0.23 | 0.64 | 0.63 | -0.01 |
| Training cost | 421 GPU-days | 269 GPU-days | **-34%** |
| Inference (2048 tokens) | ~1000 s | ~250 s | **4× faster** |

The speedup scales with sequence length due to the cubic term reduction:

| Token count | Speedup |
|---|---|
| 128 | 1.25× |
| 512 | 1.6× |
| 1024 | 2× |
| 2048 | **4×** |

### 6.4 FLOP Analysis: Why the Speedup Grows

The cubic-term FLOPs for a single block:

| Operation | Cubic Term | Quadratic Term |
|---|---|---|
| Triangle Attention | $8L^3 C\_z$ | $20L^2 C\_z^2$ |
| Triangle Multiplication | $4L^3 C\_z$ | $24L^2 C\_z^2$ |

Removing triangle attention eliminates $8L^3 C_z$ FLOPs — **two-thirds of the cubic compute**. At small $L$, the quadratic terms dominate and the savings are modest. At large $L$, the cubic terms dominate and the speedup approaches 3–4×:

$$\text{Pairmixer FLOPs} \approx \frac{4L^3 C_z + 24L^2 C_z^2}{12L^3 C_z + 44L^2 C_z^2} \times \text{Pairformer FLOPs}$$

For large $L$: $\frac{4}{12} = 33\%$ — explaining the ~4× speedup at 2048 tokens.

### 6.5 Why Triangle Attention Was Dispensable

**Redundant information propagation**: Triangle multiplication already aggregates information through intermediary residues $k$. Triangle attention does the same — it attends along rows/columns of $z$ with $z$ as bias. The information theoretic content is largely overlapping.

**Hardware mismatch**: Triangle attention requires $L$ independent attention operations (one per row or column), each with $O(L^2)$ attention maps. This translates to many small CUDA kernel launches with poor GPU occupancy. Triangle multiplication, by contrast, maps to a single batched matrix multiply — one of the most optimized operations on modern hardware.

**Learned sparsity in triangle multiplication**: Analysis shows that triangle multiplication learns sparse interaction patterns through magnitude modulation — most elements of $a_{ik} \odot b_{jk}$ have near-zero norm, with only high-norm elements contributing meaningfully. When 75% of low-norm entries are dropped, performance is maintained. This suggests triangle multiplication efficiently learns to select the relevant intermediary residues without explicit attention.

### 6.6 Memory Implications

By eliminating triangle attention's $O(L^3 \cdot H)$ memory footprint, Pairmixer achieves approximately **30% memory expansion**: structures up to ~650 residues can be processed where Pairformer would OOM at ~500 residues on the same hardware. For binder design applications targeting large protein complexes, this is practically significant.

---

## 8. Comparison Table

| | Evoformer (AF2) | Pairformer (AF3/Boltz) | SeedFold (ByteDance) | Pairmixer (ICLR 2026) |
|---|---|---|---|---|
| **MSA processing** | Inside Trunk | Outside Trunk | Outside Trunk | Outside Trunk |
| **Triangle Multiplication** | Yes | Yes | Yes | Yes |
| **Triangle Attention** | Yes (standard) | Yes (standard) | **Linearized** (sub-cubic) | **Removed** |
| **Sequence Attention** | Inside MSA track | Attention + Pair Bias | Yes | **Removed** |
| **Uses attention** | Extensively | Yes | Yes (efficient) | **Attention-free** |
| **Pair dim ($C_z$)** | 128 | 128 | Wider ($\gt$ 128) | 128 |
| **Representative models** | AF2, OpenFold, UniFold | AF3, Boltz-2, Chai-1 | SeedFold | Pairmixer |
| **vs AF3 accuracy** | — | Baseline | Exceeds AF3 | Matches |
| **Key advantage** | MSA-pair coupling | Modular, extensible | Efficient + wider | Fastest, attention-free |

---

## 9. Convergence and Open Questions

### Where the field agrees

- **Triangle multiplication is the essential operation** for pair representation updates. Every architecture that achieves SOTA includes it. RF2 and Pairmixer independently confirmed that it alone is sufficient.
- **Standard triangle attention is the bottleneck** — both SeedFold (linearization) and Pairmixer (removal) address this, confirming the consensus that vanilla $O(L^3)$ triangle attention is too expensive.
- **Separating MSA from the Trunk** (Pairformer design) is superior to joint processing (Evoformer) for extensibility to non-protein molecules.
- **Recycling** (re-running the Trunk with its own outputs as input) is valuable but not the only path — NP3 eliminates recycling entirely through its encoder-decoder design.

### What remains unresolved

**$O(L^2)$ memory scaling**: All current architectures store an explicit $L \times L$ pair representation. For a 3000-residue complex, this alone requires multi-gigabyte memory. Linear-scaling alternatives (analogous to linear attention for sequences) remain an open research direction.

**Will Pairmixer's findings transfer?** Pairmixer was demonstrated on a Pairformer-like model trained from scratch. Whether Boltz-2, Chai-1, or other production models can drop triangle attention without retraining — or whether they can benefit from a mixed approach (e.g., triangle attention in early layers only) — is untested.

**Linear attention vs. no attention?** SeedFold and Pairmixer offer competing visions. SeedFold's linear triangle attention preserves the attention mechanism's expressiveness at reduced cost, while Pairmixer shows attention is unnecessary altogether. Whether linearized attention provides benefits over pure triangle multiplication at larger scales — or whether the added complexity is unjustified — remains to be determined through direct comparison.

**Optimal depth vs width**: Pairmixer uses the same block count as Pairformer. Given the reduced per-block cost, would a deeper Pairmixer (more blocks) outperform a wider one? SeedFold's wider pair representations suggest that dimension may matter more than previously assumed. The scaling dynamics of attention-free vs. efficient-attention architectures may differ fundamentally.

---

**Next: Part 3 — How to Output 3D Structure? IPA → Diffusion → Flow Matching**

We move from the Trunk to the Structure Generation Module — examining how the field transitioned from deterministic coordinate prediction (AF2's IPA) to stochastic generation via diffusion (AF3) and flow matching (NP3, Proteina).

---

*Part of the series: The Technical Evolution of Protein AI — A Record of Key Design Decisions*
