---
title: "Protein AI Series Part 7: Training Engineering and Scaling"
date: 2026-03-17 10:10:00 +0900
categories: [Drug Discovery]
tags: [protein-ai, training-engineering, flashattention, distributed-training, scaling-laws]
math: true
---

**The Technical Evolution of Protein AI — A Record of Key Design Decisions**

*This is Part 7 of a 10-part series tracing the architectural choices behind modern protein structure prediction and design models.*

---

## The Core Question

**What engineering is required to train a protein AI model with hundreds of millions of parameters?**

Parts 1–6 traced the architectural evolution of protein AI: how models read evolutionary information, reason about residue pairs, generate 3D structures, handle diverse biomolecules, design novel proteins, and capture conformational diversity. But every architecture choice must survive contact with **hardware reality**. A Pairformer with 48 blocks processing an $L = 2000$ residue complex generates pair representations consuming tens of gigabytes of memory. A diffusion module running 200 denoising steps multiplies the computational cost of structure generation by two orders of magnitude. Papers report "trained on $N$ GPUs for $M$ days" — but behind that single sentence lie dozens of engineering decisions that determine whether training succeeds or fails, whether convergence takes weeks or months, and whether the final model matches or exceeds the reported benchmarks.

This Part examines the engineering techniques that make large-scale protein AI training possible — from GPU kernel optimizations to distributed training strategies to scaling laws — and explains why reproducing an architecture without reproducing the engineering often fails to reproduce the results.

---

## 1. Why Engineering Matters: The Unique Challenges of Protein AI

### 1.1 The $O(L^2)$ Memory Wall

The pair representation $z\_{ij} \in \mathbb{R}^{L \times L \times d\_z}$ is the central data structure of modern protein AI (Part 2). For a complex with $L$ tokens and pair dimension $d\_z = 128$:

$$\text{Memory}(z) = L^2 \times d_z \times \text{bytes per element}$$

| $L$ (tokens) | BF16 Memory for $z$ | FP32 Memory for $z$ |
|---|---|---|
| 384 (AF3 crop) | 36 MB | 72 MB |
| 768 | 144 MB | 288 MB |
| 1024 | 256 MB | 512 MB |
| 2048 | 1.0 GB | 2.0 GB |

This is the memory for a **single** pair tensor. A Pairformer block stores intermediate activations for Triangle Multiplication, Triangle Attention, projections, and gating — easily 5–10 $\times$ the raw $z$ size. With 48 blocks and no memory optimization, the activation memory for a 2048-token complex can exceed **100 GB** — well beyond any single GPU.

### 1.2 The Triple Cost Multiplier

Protein AI models face three compounding cost factors:

```
1. O(L²) pair representation    → quadratic memory and compute
2. N_blocks deep Trunk           → 48 blocks × per-block activations
3. T diffusion/FM steps          → 200 (AF3) or 40 (NP3) forward passes through structure module

Total: O(L² × N_blocks) for Trunk  +  O(L × T) for diffusion
```

For comparison, a standard language model with sequence length $L$ requires $O(L^2)$ memory only for the attention matrix within each layer, and processes each token once. Protein models require $O(L^2)$ for the pair representation that persists across all layers, plus $O(L^3)$ operations for Triangle Attention/Multiplication in every block.

### 1.3 What the Papers Report

| Model | Hardware | Training Budget | Parameters |
|---|---|---|---|
| AF2 | 128 TPUv3 | ~11 days initial + fine-tuning | ~93M |
| AF3 | TPU pod (details undisclosed) | ~1400 GPU-days equiv. | ~90M |
| Boltz-2 | 64 A100 80GB | ~450 GPU-days | ~90M |
| OpenFold3 | Multi-node GPU | Not disclosed | ~90M |
| Proteina | Multi-node GPU | Variable (scaling experiments) | 50M–3B |
| Pairmixer | A100 | 269 GPU-days | ~70M |

Behind each row lies a specific combination of the techniques described below.

---

## 2. Attention Optimization: FlashAttention and Its Impact

### 2.1 The Memory Bottleneck of Standard Attention

Standard scaled dot-product attention computes:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

The naive implementation **materializes the full $L \times L$ attention matrix** in GPU High Bandwidth Memory (HBM), requiring $O(L^2)$ memory. For protein models, this matrix appears in:

- **AttentionPairBias** (Pairformer): $L \times L$ attention over single representation, biased by pair representation
- **Triangle Attention** (Pairformer): $L$ independent $L \times L$ attention operations over pair representation → $O(L^3)$ total

The data movement pattern is the critical bottleneck: GPU SRAM (on-chip, fast, ~20 MB) is orders of magnitude faster than HBM (off-chip, slow, ~80 GB), but standard attention writes the full $L \times L$ matrix to HBM and reads it back for the softmax and value multiplication — an **IO-bound** operation.

### 2.2 FlashAttention: Tiling for IO Efficiency

**FlashAttention** (Dao et al., 2022) eliminates the materialization of the attention matrix through **tiling**: instead of computing the entire $L \times L$ matrix at once, it processes $Q$, $K$, $V$ in blocks that fit in SRAM.

```
Standard Attention:                  FlashAttention:

  Q (L×d)  ×  K^T (d×L)              Q block (B×d) × K block (d×B)
       │                                    │
       ▼                                    ▼
  S = Q·K^T  (L×L in HBM)            S_block (B×B in SRAM)
       │                                    │
       ▼                                    ▼
  P = softmax(S)  (L×L in HBM)       Online softmax accumulation
       │                                    │    (running max + sum)
       ▼                                    ▼
  O = P·V  (L×d)                      O_block accumulated in SRAM
                                            │
  Memory: O(L²)                             ▼
  IO: O(L²·d)                         Write final O to HBM
                                       Memory: O(L)
                                       IO: O(L²·d² / SRAM_size)
```

The key algorithmic insight is **online softmax**: by maintaining running statistics (max value and sum of exponentials), each block's contribution to the softmax can be correctly accumulated without ever seeing the full row. This reduces memory from $O(L^2)$ to $O(L)$ and achieves 2–4 $\times$ wall-clock speedup by minimizing HBM access.

### 2.3 OpenFold's Contribution: FlashAttention Meets Protein AI

**OpenFold** (2022), the first complete open-source reproduction of AlphaFold2, made a critical engineering contribution: **applying FlashAttention to protein structure prediction for the first time**. This provided immediate benefits:

- MSA row-wise and column-wise attention: direct FlashAttention application → significant memory reduction for deep MSAs
- Pair-biased attention: FlashAttention supports additive attention bias, enabling direct application to AttentionPairBias

This engineering choice became the foundation for all subsequent GPU-based protein models. Boltz-1/2, Chai-1, Protenix, and OpenFold3 all build on FlashAttention as a baseline assumption.

### 2.4 Triangle Attention: FlashAttention's Limits

Triangle Attention presents a particular challenge. For "starting node" Triangle Attention, the operation is:

$$z_{ij}^{\text{out}} = \sum_k \text{softmax}_k\left(\frac{q_i^\top k_k}{\sqrt{d}}\right) v_{kj}$$

This is $L$ independent attention operations (one per row $j$), each of size $L \times L$. FlashAttention accelerates each individual attention, but the **total complexity remains $O(L^3)$** — $L$ calls to an $O(L^2)$ kernel.

```
Triangle Attention with FlashAttention:

  For j = 1 to L:
    FlashAttn(Q[:, j, :], K[:, j, :], V[:, j, :])   ← O(L²) each
                                                       ← L calls
  Total: O(L³) — FlashAttention helps constant, not complexity class
```

This fundamental limitation motivated two responses (Part 2):
- **SeedFold**: Replace Triangle Attention with **Linear Triangle Attention** → sub-cubic complexity
- **Pairmixer**: **Remove Triangle Attention entirely** → only Triangle Multiplication remains

### 2.5 FlashAttention-2 and FlashAttention-3

| Version | Key Improvements | Impact on Protein AI |
|---|---|---|
| FlashAttention-2 (2023) | Better work partitioning, non-square tiling, ~2 $\times$ over FA-1 | Default for all new protein models |
| FlashAttention-3 (2024) | Asynchronous WGMMA, FP8 support, pipelining | H100/B200 optimization; FP8 exploration for protein models |

---

## 3. Memory Management: Activation Checkpointing and Crop Strategies

### 3.1 Activation Checkpointing

During backpropagation, gradient computation requires the intermediate activations from the forward pass. For a 48-block Pairformer, naively storing all activations requires memory proportional to:

$$\text{Activation Memory} \propto N_{\text{blocks}} \times L^2 \times d_z$$

**Activation checkpointing** (also called **gradient checkpointing**) trades compute for memory: instead of storing intermediate activations, it discards them during the forward pass and **recomputes** them during the backward pass.

```
Without checkpointing:              With checkpointing:

Forward: store all 48 blocks        Forward: store only block boundaries
  Block 1 → save activations          Block 1 → discard activations
  Block 2 → save activations          Block 2 → discard activations
  ...                                  ...
  Block 48 → save activations         Block 48 → discard activations

Backward: use stored activations     Backward: recompute then use
  ∂L/∂Block 48 ← stored acts          ∂L/∂Block 48 ← recompute Block 48
  ∂L/∂Block 47 ← stored acts          ∂L/∂Block 47 ← recompute Block 47
  ...                                  ...

Memory: O(48 × L² × d_z)            Memory: O(L² × d_z) + checkpoints
Compute: 1× forward + 1× backward   Compute: 1× forward + ~1.3× backward
```

The trade-off: **~60% memory reduction** at the cost of **~30% additional computation**. For protein models where memory is the binding constraint, this is almost universally adopted. Every model in the table above uses activation checkpointing.

**Selective checkpointing** goes further: instead of checkpointing every block uniformly, checkpoint only the most memory-intensive operations (Triangle Multiplication, Triangle Attention) while keeping cheaper operations (transitions, projections) stored. This reduces the recomputation overhead while maintaining most of the memory savings.

### 3.2 Crop Strategies: Training on Fragments

Real protein complexes range from ~50 residues (small peptides) to 10,000+ tokens (large multi-chain complexes with ligands). Training on full structures is infeasible for large complexes due to $O(L^2)$ memory scaling. The solution: **crop** the complex to a fixed number of tokens during training.

**Spatial cropping** (AF3, Protenix):
1. Randomly select a seed atom
2. Select the $N_{\text{crop}}$ tokens closest to the seed in 3D space
3. Result: a spatially contiguous fragment that preserves local interactions

**Contiguous cropping** (Boltz-2, initial stage):
1. Select a contiguous segment along the chain
2. Simpler but misses inter-chain interactions

```
Full complex (2000 tokens):            Spatial crop (384 tokens):

  Chain A: ●●●●●●●●●●●●●●●●●           Seed atom ★
  Chain B: ○○○○○○○○○○○○○○               ┌────────────────┐
  Ligand:  △                             │ ●●●●●●         │
                                          │ ○○○○○  ★       │
  O(2000²) = 4M token pairs              │ △              │
  → infeasible for training               └────────────────┘
                                          O(384²) ≈ 147K token pairs
                                          → fits in GPU memory
```

### 3.3 Boltz-2's Multi-Stage Crop Strategy

Boltz-2 demonstrates a sophisticated curriculum over crop sizes:

| Stage | Crop Size | Purpose |
|---|---|---|
| Stage 1 | 256 | Fast initial convergence — small crops enable large batch sizes |
| Stage 2 | 384 | Refinement — matches AF3's crop size for fair comparison |
| Stage 3 | Mixed (256–512) | Generalization — variable crops prevent overfitting to fixed size |

The insight: **start small, grow large**. Small crops allow rapid iteration during early training when the model is learning basic structural motifs. Larger crops become important later when the model needs to capture long-range interactions and inter-chain contacts.

### 3.4 The Crop-Inference Gap

A persistent challenge: models trained on crops of 384 tokens must generalize to full complexes of 2000+ tokens at inference. This **distribution shift** can cause:

- Edge effects at crop boundaries (trained to expect "missing" neighbors)
- Degraded long-range contacts (never seen pairs separated by > 384 tokens)
- Inconsistent confidence scores (pLDDT calibrated on cropped structures)

Some models address this with inference-time strategies: running overlapping crops and stitching results, or using lower precision to fit larger structures. But the fundamental tension between training efficiency (small crops) and inference quality (full structures) remains.

---

## 4. Distributed Training: Data Parallelism and Model Parallelism

### 4.1 Why Distribution Is Necessary

A single A100 80GB GPU cannot hold the full training state of a 90M-parameter protein model processing 384-token crops:

```
Model parameters (BF16):          ~180 MB
Optimizer state (AdamW, FP32):    ~1.1 GB  (params + momentum + variance, FP32)
Gradients (BF16):                 ~180 MB
Activations (48 blocks, BF16):    ~10-30 GB (with checkpointing)
Pair representation workspace:     ~2-8 GB (depending on crop size)
──────────────────────────────────
Total:                             ~15-40 GB for crop=384
```

This fits on a single GPU for small crops, but scaling to crop=768 or larger models (Proteina 3B) requires distributing across multiple GPUs.

### 4.2 DDP: The Baseline

**Distributed Data Parallelism (DDP)** is the simplest strategy: replicate the entire model on each GPU, split the data, and synchronize gradients after each step.

$$g_{\text{global}} = \frac{1}{N_{\text{GPU}}} \sum_{i=1}^{N_{\text{GPU}}} g_i \quad \text{(AllReduce)}$$

```
GPU 0: Full model copy + Batch 0 → g₀ ─┐
GPU 1: Full model copy + Batch 1 → g₁ ─┤── AllReduce → g_avg → update all copies
GPU 2: Full model copy + Batch 2 → g₂ ─┤
GPU 3: Full model copy + Batch 3 → g₃ ─┘
```

**Advantage**: Minimal communication overhead, linear scaling of throughput.
**Limitation**: Each GPU must hold the full model + optimizer + activations → only works when the model fits in a single GPU's memory.

Early protein models (AF2 on TPUs, smaller OpenFold runs, Pairmixer) use DDP because the ~90M parameter models are small enough.

### 4.3 FSDP: Sharding Everything

**Fully Sharded Data Parallelism (FSDP)** extends DDP by sharding not just data but also **parameters, gradients, and optimizer states** across GPUs. This is based on the ZeRO (Zero Redundancy Optimizer) concept:

```
DDP (each GPU stores everything):     FSDP (each GPU stores 1/N):

  GPU 0: [params₀₋₃] [opt₀₋₃]          GPU 0: [params₀] [opt₀]
  GPU 1: [params₀₋₃] [opt₀₋₃]          GPU 1: [params₁] [opt₁]
  GPU 2: [params₀₋₃] [opt₀₋₃]          GPU 2: [params₂] [opt₂]
  GPU 3: [params₀₋₃] [opt₀₋₃]          GPU 3: [params₃] [opt₃]

  Memory per GPU: Full model             Memory per GPU: 1/4 model
```

When a layer needs its full parameters for forward/backward computation, FSDP performs an **AllGather** to temporarily reconstruct the full parameters, computes, and then discards the non-local shards. The communication overhead is offset by the dramatic memory savings.

**Boltz-2** and **OpenFold3** use FSDP as their primary distribution strategy. This enables training on 64 GPUs with effective batch sizes large enough for stable convergence.

### 4.4 Tensor Parallelism: Splitting Layers

**Tensor Parallelism (TP)** splits individual layers across GPUs. For a linear layer $Y = XW$ where $W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$:

$$W = [W_1 \;|\; W_2], \quad Y = X \cdot [W_1 \;|\; W_2] = [XW_1 \;|\; XW_2]$$

Each GPU computes $XW_i$ on its local shard, then results are combined via AllReduce or AllGather. For attention layers, the heads are naturally distributed: GPU $i$ computes heads $i \cdot H/N$ through $(i+1) \cdot H/N - 1$.

```
Tensor Parallelism on Attention (4 GPUs, 16 heads):

  GPU 0: heads 0-3  ─┐
  GPU 1: heads 4-7  ─┤── AllReduce → combined output
  GPU 2: heads 8-11 ─┤
  GPU 3: heads 12-15─┘
```

TP requires **high-bandwidth interconnect** (NVLink within a node: ~900 GB/s) because communication happens within every layer's forward/backward pass. It is typically used **intra-node** (within a single 8-GPU machine).

### 4.5 Pipeline Parallelism: Splitting Stages

**Pipeline Parallelism (PP)** assigns different layers to different GPUs and passes activations between them sequentially. A 48-block Pairformer on 4 GPUs:

```
GPU 0: Blocks 1-12 → activations → GPU 1: Blocks 13-24 → ...
                                     → GPU 2: Blocks 25-36 → ...
                                       → GPU 3: Blocks 37-48
```

The **bubble problem**: GPUs are idle while waiting for upstream activations. Micro-batching (splitting a batch into smaller chunks) and interleaved scheduling (1F1B — one forward, one backward) reduce but cannot eliminate this idle time.

PP is less common in protein AI than in LLM training because protein models are relatively small (~90M vs. LLMs' 70B+) and the pair representation's $O(L^2)$ memory dominates over parameter storage.

### 4.6 Practical Combinations

Real training setups combine strategies:

```
Typical large-scale protein AI setup:

  Intra-node (8 GPUs, NVLink):
    FSDP (parameter sharding)
    Optional: TP for very large models (Proteina 3B)

  Inter-node (multiple nodes, InfiniBand):
    DDP (data parallelism across nodes)

  Example: 8 nodes × 8 GPUs = 64 GPUs
    FSDP within each node, DDP across nodes
    Effective batch size: 64 × micro_batch_size
```

| Strategy | What It Shards | Communication | Best For |
|---|---|---|---|
| DDP | Data only | AllReduce (gradients) | Small models, simple setup |
| FSDP | Params + grads + optimizer | AllGather + ReduceScatter | Medium-large models, memory-constrained |
| TP | Layer internals | AllReduce per layer | Very large models, intra-node |
| PP | Model stages | Point-to-point activations | Extremely deep models |

---

## 5. Numerical Stability: Mixed Precision Training

### 5.1 Why Precision Matters

Training in FP32 (32-bit floating point) uses 4 bytes per parameter; **BF16** (Brain Float16) uses 2 bytes — halving memory and doubling throughput on NVIDIA Tensor Cores. Virtually all modern protein AI models train in BF16.

```
FP32:  1 bit sign | 8 bits exponent | 23 bits mantissa
BF16:  1 bit sign | 8 bits exponent |  7 bits mantissa
FP16:  1 bit sign | 5 bits exponent | 10 bits mantissa
```

**BF16 vs. FP16**: BF16 has the same exponent range as FP32 (can represent values from $\sim 10^{-38}$ to $\sim 10^{38}$) but lower precision (7 vs. 23 mantissa bits). FP16 has higher precision (10 bits) but a narrower range ($\sim 10^{-5}$ to $6.5 \times 10^4$) — requiring **loss scaling** to avoid underflow in gradients. Protein AI models strongly prefer BF16 for its numerical stability.

### 5.2 Protein-Specific Numerical Challenges

Protein models present unique numerical difficulties that go beyond standard BF16 training:

**Triangle Multiplication overflow**: The outgoing Triangle Multiplication computes:

$$z_{ij}^{\text{out}} = \sum_k a_{ik} \cdot b_{jk}$$

where $a, b \in \mathbb{R}^{L \times L \times d}$. This einsum accumulates $L$ terms — for $L = 2048$ with BF16 values, the sum can overflow BF16's limited mantissa precision, producing NaN or degraded gradients.

**Attention logit overflow**: For large pair representations, the attention logits $QK^\top / \sqrt{d_k} + z_{\text{bias}}$ can produce values outside BF16's representable range, causing softmax to return 0 or 1 (saturated gradients).

**Diffusion noise schedule**: EDM-style diffusion (Part 3) uses noise levels $\sigma$ spanning from 0.01 to 160 — a dynamic range of $10^4$. The preconditioning functions $c\_{\text{skip}}(\sigma)$, $c\_{\text{out}}(\sigma)$, $c\_{\text{in}}(\sigma)$ must handle this range without precision loss.

### 5.3 Chunk-and-Accumulate: Boltz's Solution

Boltz addresses the Triangle Multiplication overflow problem with **chunk-and-accumulate**: instead of computing the full einsum in BF16, split the summation into chunks and accumulate in FP32:

```
Standard (BF16 throughout):

  z_ij = einsum("bikd,bjkd->bijd", a, b)     ← BF16, sum over L terms
                                                 overflow risk for large L

Chunk-and-Accumulate:

  z_ij = 0  (FP32 accumulator)
  for chunk in split(k_dim, chunk_size=128):
    z_ij += einsum("biCd,bjCd->bijd", a_chunk, b_chunk).float()
                                  ↑ BF16 einsum    ↑ accumulate in FP32

  z_ij = z_ij.bfloat16()         ← cast back to BF16 for next operation
```

This preserves BF16's speed for the per-chunk einsum (Tensor Core utilization) while using FP32 only for the accumulation — a small memory and compute overhead that prevents numerical instability.

### 5.4 FP8: The Next Frontier

NVIDIA H100 and B200 GPUs support **FP8** (8-bit floating point) on Tensor Cores, offering 2 $\times$ throughput over BF16. FlashAttention-3 includes FP8 support. However, FP8 adoption in protein AI is still exploratory:

- **E4M3** format (4-bit exponent, 3-bit mantissa): useful for forward pass, max value ~448
- **E5M2** format (5-bit exponent, 2-bit mantissa): useful for backward pass, wider range

The challenge: protein models' numerical sensitivity (Triangle Multiplication accumulation, diffusion preconditioning) makes FP8 training nontrivial. Selective FP8 — using FP8 for attention kernels while keeping critical accumulations in higher precision — is the likely path forward.

---

## 6. Scaling Laws: Do Bigger Models Work Better?

### 6.1 Proteina's Discovery

**Proteina** (NVIDIA, ICLR 2025 Oral) provided the first systematic evidence that **scaling laws apply to protein structure generation** — not just language models.

The experiment: train flow-matching backbone generation models at four scales and measure structural quality:

| Model Size | Parameters | Key Observation |
|---|---|---|
| Proteina-S | 50M | Baseline |
| Proteina-M | 200M | Consistent improvement over S |
| Proteina-L | 600M | Further improvement |
| Proteina-XL | 3B | Best on all metrics |

All structural quality metrics (designability, diversity, novelty) showed **log-linear improvement** with model size — the same pattern observed in language model scaling laws (Kaplan et al., 2020). This was not obvious *a priori*: protein structure is a physical system with hard geometric constraints, and it was plausible that small models could capture these constraints as well as large ones.

### 6.2 The Two Axes of Scaling

Scaling operates along two dimensions:

**Parameter scaling**: Increasing the number of Trunk blocks, hidden dimensions, and attention heads. The standard Pairformer uses $d\_z = 128$ and 48 blocks; Proteina-XL scales to much larger dimensions across its flow-matching architecture. SeedFold explores a related direction: wider pair representations (increased $d\_z$) while keeping the block count constant.

**Data scaling**: The Protein Data Bank (PDB) contains ~220K experimental structures — a tiny dataset by deep learning standards. Scaling data requires **synthetic structures**:

```
Data scaling strategies:

  PDB (~220K experimental structures)
    │
    ├── AlphaFold Database (~200M predicted structures)
    │     → Proteina: pre-train on ESMFold-generated structures
    │     → OpenFold3: 13M AF3-distilled structures
    │
    ├── Self-distillation
    │     → Boltz-2: use own predictions as training data
    │     → Quality filtering critical (keep only high-confidence)
    │
    └── Synthetic complexes
          → Complexa/Teddymer: synthetic binder-target pairs
          → Expand complex diversity beyond PDB coverage
```

### 6.3 Scaling Limits

Several factors constrain how far protein AI can scale:

**Memory wall**: $O(L^2)$ pair representations limit the maximum crop size (and thus context window) regardless of model size. Increasing parameters adds proportionally to model memory but the pair representation dominates for large $L$.

**Data quality vs. quantity**: Synthetic structures inherit the errors of the generating model. OpenFold3's 13M distilled structures are AF3 predictions — any systematic biases in AF3 propagate to OpenFold3. Quality filtering (discarding low-confidence predictions) is essential but reduces effective data size.

**Diminishing returns for small problems**: Scaling laws show improvements on *average*, but many single-domain proteins are already predicted near-perfectly by 90M-parameter models. The gains from scaling are concentrated on hard cases — large complexes, disordered regions, novel folds — that are precisely where training data is scarcest.

**Scale comparison**: Proteina-XL at 3B parameters is the largest protein AI model to date. For context, modern language models reach 70B–400B+ parameters. The gap reflects both the smaller available dataset and the $O(L^2)$ memory constraint that limits effective batch sizes.

---

## 7. Multi-Stage Training Pipelines

### 7.1 Why Not Train Everything at Once?

Protein AI models learn multiple tasks — structure prediction, confidence estimation, and increasingly affinity prediction and design — that have different loss landscapes, data requirements, and convergence dynamics. Training all objectives simultaneously from the start leads to:

- **Conflicting gradients**: Structure prediction loss and confidence loss can push representations in opposite directions during early training
- **Data imbalance**: Affinity data (experimental binding measurements) is ~100 $\times$ scarcer than structural data
- **Wasted compute**: Confidence and affinity heads cannot learn meaningfully until the model produces reasonable structures

Multi-stage training addresses this with curriculum: teach structure first, then build higher-level capabilities on top.

### 7.2 AF2/AF3: Two-Stage Training

AlphaFold models use a relatively simple two-stage pipeline:

```
AF2/AF3 Training Pipeline:

  Stage 1: Initial Training
  ─────────────────────────
    Crop size: 256-384
    Recycling: 3 cycles
    Objective: Structure prediction (FAPE loss + auxiliary losses)
    Duration: ~90% of total compute

  Stage 2: Fine-tuning
  ────────────────────
    Crop size: 384-512 (larger)
    Recycling: fewer cycles
    Objective: Same + refined loss weighting
    Duration: ~10% of total compute
```

### 7.3 Boltz-2: Three-Stage Pipeline

Boltz-2 introduces a more sophisticated pipeline that progressively builds capabilities:

```
Boltz-2 Training Pipeline:

  Stage 1: Structure Prediction
  ─────────────────────────────
    Data:     PDB experimental structures (+ optional distillation data)
    Loss:     Diffusion MSE + smooth_LDDT + distogram + bond geometry
    Crop:     256 → 384 (curriculum)
    Output:   Model that predicts 3D structures from sequence/MSA

         │ weights inherited
         ▼

  Stage 2: Confidence Head Training
  ──────────────────────────────────
    Data:     Same as Stage 1
    Loss:     pTM + pLDDT + PAE prediction (predicting own error)
    Freeze:   Structure prediction weights partially frozen
    Output:   Model that predicts structures AND estimates confidence

         │ weights inherited
         ▼

  Stage 3: Affinity Head Training
  ────────────────────────────────
    Data:     PDBbind + BindingDB (experimental K_d, K_i, IC50)
    Loss:     Affinity prediction MSE
    Freeze:   Structure + confidence weights frozen
    Output:   Full model: structure + confidence + affinity
```

Each stage inherits the previous stage's weights, ensuring that structural understanding is established before confidence estimation, and confidence estimation before affinity prediction. This curriculum is critical: a model that cannot fold a protein correctly cannot meaningfully predict its binding affinity.

### 7.4 Proteina's Curriculum: Small to Large

Proteina takes a different approach to curriculum learning, organized around **model size and crop size** rather than task objectives:

1. **Hyperparameter search** on small models (50M) → identify optimal learning rate, noise schedule, data mixing
2. **Transfer to large models** (3B) → same hyperparameters, scaled compute
3. **Crop curriculum**: train with small crops initially (faster iteration), increase crop size as training progresses

This strategy amortizes the cost of hyperparameter search: experimenting on a 50M model is ~60 $\times$ cheaper than on a 3B model.

### 7.5 Distillation

**Knowledge distillation** uses a trained teacher model to generate synthetic training data for a student model:

```
Teacher (AF3)                  Student (OpenFold3)
──────────                     ───────────────────

  PDB + UniRef sequences         PDB experimental structures
       │                              +
       ▼                         13M teacher-generated structures
  AF3 predictions ────────────→        │
  (13M structures)                     ▼
                                Train from scratch on combined data
```

OpenFold3's use of 13M AF3-distilled structures demonstrates both the power and risk of this approach:
- **Power**: Dramatically expands effective training set size, enabling the model to match AF3's performance on most benchmarks
- **Risk**: The student inherits the teacher's systematic biases. Where AF3 consistently fails (certain RNA structures, rare folds), the distilled data reinforces rather than corrects these failures

Boltz-2 provides explicit ablation studies comparing training with and without distillation data, showing that distillation particularly helps on targets where PDB training data is sparse (novel folds, unusual ligand types).

---

## 8. Comparison: Training Engineering Across Models

| | AF2/AF3 | Boltz-2 | Proteina | OpenFold3 | Pairmixer |
|---|---|---|---|---|---|
| **Hardware** | 128 TPUv3 | 64 A100 80GB | Multi-GPU | Multi-GPU | A100 |
| **Distribution** | DP (TPU) | FSDP | FSDP + TP | FSDP | DDP |
| **Precision** | BF16 | BF16 | BF16 | BF16 | BF16 |
| **FlashAttention** | N/A (TPU) | Yes | Yes | Yes | N/A (no attention) |
| **Act. Checkpoint** | Yes | Yes | Yes | Yes | Yes |
| **Crop Strategy** | Spatial, 384 | Multi-stage (256→384→mixed) | Curriculum (small→large) | Spatial | Spatial |
| **Training Stages** | 2-stage | 3-stage (struct→conf→affinity) | Scaling curriculum | 2-stage + distillation | 1-stage |
| **Synthetic Data** | — | Optional distillation | ESMFold structures | 13M AF3 distillation | — |
| **Total GPU-days** | ~1,400 | ~450 | Variable | — | 269 |
| **Parameters** | ~90M | ~90M | 50M–3B | ~90M | ~70M |

### Key Observations

**Convergence to a standard stack**: FlashAttention + BF16 + FSDP + activation checkpointing is now the de facto standard for GPU-based protein AI training. Models that deviate (AF2/AF3 on TPUs, Pairmixer without attention) do so for architecture-specific reasons.

**Crop strategy as a design choice**: There is no consensus on optimal cropping. AF3's fixed spatial crop, Boltz-2's multi-stage curriculum, and Proteina's progressive growth all achieve competitive results — suggesting that the choice depends on the specific model architecture and training budget.

**Engineering determines reproducibility**: Protenix (ByteDance) uses nearly the same architecture as AF3 but with different training engineering (modified MSA signal flow, different crop strategies). The result: measurably different performance despite architectural near-identity. This demonstrates that **reproducing an architecture without reproducing the training engineering is insufficient**.

---

## 9. Convergence and Outlook

### What has become standard

- **FlashAttention**: Universal on GPU platforms. Future kernel fusions (Triton, CUTLASS) will push further.
- **BF16 mixed precision**: No protein AI model trains in FP32 anymore.
- **FSDP**: Standard for multi-GPU training; DDP only for single-node experiments.
- **Activation checkpointing**: Universally adopted; the memory savings are too large to forgo.
- **Multi-stage training**: Single-stage training is increasingly rare as models add confidence and affinity heads.

### What remains unsolved

**Optimal crop strategy**: The best crop size, spatial vs. contiguous selection, and curriculum schedule remain model-dependent. No theoretical framework predicts the optimal strategy for a given architecture.

**FP8 for protein AI**: H100/B200 hardware supports FP8 with 2 $\times$ throughput over BF16, but protein models' numerical sensitivity (Triangle Multiplication accumulation, diffusion preconditioning) makes adoption nontrivial. Selective FP8 — using FP8 for attention while keeping accumulations in higher precision — is the likely path.

**Scaling beyond 3B**: Proteina-XL at 3B is the largest protein AI model. Scaling further requires either overcoming the $O(L^2)$ memory wall (Pairmixer and SeedFold's efficiency improvements help) or developing protein-specific parallelism strategies that exploit the structure of pair representations.

**Dedicated protein AI kernels**: Current models rely on general-purpose deep learning primitives (FlashAttention, cuBLAS einsum). Custom CUDA/Triton kernels for Triangle Multiplication, pair-biased attention, and diffusion preconditioning could yield significant speedups but require substantial engineering investment.

### The Core Lesson

The same architecture trained with different engineering produces different results. AF3's closed training pipeline, Boltz-2's open multi-stage approach, and Protenix's architectural near-clone with different engineering all achieve measurably different performance. In protein AI, **engineering is not implementation detail — it is a design decision as consequential as the choice of attention mechanism or generation framework**.

---

**Next: Part 8 — What Data Do These Models Train On? The Experimental, Synthetic, and Sequence Data Landscape**

Before concluding the series, we examine the often-overlooked foundation of all protein AI models: training data. From the ~220K experimental structures in the PDB to the 214M+ synthetic structures in AFDB (now expanded with proteome-scale quaternary complexes), from knowledge distillation pipelines to synthetic binder-target datasets — we map the full data landscape and show why data, not architecture, may be the decisive competitive advantage.

---

*Part of the series: The Technical Evolution of Protein AI — A Record of Key Design Decisions*
