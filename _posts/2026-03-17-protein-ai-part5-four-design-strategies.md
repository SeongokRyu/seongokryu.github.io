---
title: "Protein AI Series Part 5: Four Design Strategies Compared"
date: 2026-03-17 09:50:00 +0900
categories: [Drug Discovery]
tags: [protein-ai, protein-design, rfdiffusion, boltzgen, esm3, flow-matching]
math: true
---

**The Technical Evolution of Protein AI — A Record of Key Design Decisions**

*This is Part 5 of a 10-part series tracing the architectural choices behind modern protein structure prediction and design models.*

---

## The Core Question

**Can a model that "predicts" protein structure also "design" novel proteins?**

Parts 1–4 traced the evolution of structure prediction: how to read evolutionary information, reason about residue pairs, generate 3D coordinates, and handle heterogeneous molecular complexes. But prediction answers the question "given this sequence, what is the structure?" — design asks the inverse: "given a desired function (e.g., binding a target), what sequence and structure would achieve it?"

This Part examines how the field has repurposed prediction models for design, identifying four distinct strategies — each with different trade-offs in expressiveness, maturity, and scalability.

---

## 0. The Common Pattern: Keep the Trunk, Replace the Head

Before examining individual strategies, it is worth noting a recurring architectural pattern: **successful design models reuse the Trunk (representation learning) from prediction models, replacing or extending only the output head**.

```
  Structure Prediction Model              Design Model
  ─────────────────────────              ────────────

  Input features                         Input features + design spec
       │                                      │
       ▼                                      ▼
  ┌──────────┐                          ┌──────────┐
  │  Trunk   │  ← shared weights →      │  Trunk   │  (reused / fine-tuned)
  └────┬─────┘                          └────┬─────┘
       │                                      │
       ▼                                      ▼
  ┌──────────┐                          ┌──────────────┐
  │ Predict  │                          │  Generate    │
  │  Head    │                          │  Head        │
  └──────────┘                          └──────────────┘
       │                                      │
       ▼                                      ▼
  Structure                             Novel structure + sequence
```

The rationale is intuitive: the Trunk has learned rich representations of protein physics and geometry from training on experimental structures. This knowledge is directly useful for generation — understanding what makes a protein stable is prerequisite to designing one. The output head, however, must change: prediction heads produce a single deterministic output, while design heads must sample from a distribution over possible solutions.

This pattern appears across all four strategies:

| Strategy | Prediction Model | Reused Component | New Component |
|---|---|---|---|
| SE(3) Diffusion | RoseTTAFold | RF Trunk → denoiser | SE(3) diffusion on frames |
| Conditional Generation | Boltz-2 | Pairformer + Diffusion Module | Masking + geometric encoding |
| Latent Flow Matching | — | — | Flow matching + VAE |
| Discrete Multimodal | ESM-2 | Transformer backbone | Multi-track tokenization |

---

## Part A: The Four Design Strategies

## 1. Strategy 1: SE(3) Diffusion on Backbone Frames

### 1.1 RFdiffusion: Denoising as Design

**RFdiffusion** (Watson et al., 2023, *Nature*) converts the RoseTTAFold structure prediction model into a generative model through a conceptually simple but powerful insight: **structure prediction is denoising, and denoising is generative modeling**.

The key idea: instead of predicting structure from sequence features (as in prediction), corrupt a known structure with noise and train the model to recover it. At inference time, start from pure noise and iteratively denoise — the result is a novel protein structure.

**Representation**: Each residue $i$ is represented as a rigid frame $T\_i = (R\_i, \vec{t}\_i)$ where $R\_i \in \mathrm{SO}(3)$ is a rotation matrix encoding the backbone orientation (N-Cα-C plane) and $\vec{t}\_i \in \mathbb{R}^3$ is the Cα position.

**Forward process**: The forward noising process operates on both components:

$$\vec{t}_i^{(t)} = \vec{t}_i^{(0)} + \sigma_t \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I_3)$$

$$R_i^{(t)} = R_i^{(0)} \cdot \exp(\sigma_t^{\mathrm{rot}} \cdot \hat{\omega}), \quad \hat{\omega} \sim \mathcal{N}(0, I_3)$$

where translations are corrupted with isotropic Gaussian noise and rotations with Brownian motion on SO(3) via the exponential map.

**Reverse process**: The RoseTTAFold trunk, originally trained for structure prediction, is fine-tuned to predict the clean structure $T\_i^{(0)}$ from the noisy input $T\_i^{(t)}$:

$$\hat{T}_i^{(0)} = f_\theta(T_1^{(t)}, \ldots, T_L^{(t)}, t, c)$$

where $c$ encodes conditioning information (e.g., target protein, hotspot residues, symmetry constraints). The loss is:

$$\mathcal{L} = \sum_{i} \left[ \| \hat{\vec{t}}_i^{(0)} - \vec{t}_i^{(0)} \|^2 + \lambda_{\mathrm{rot}} \cdot d_{\mathrm{SO}(3)}(\hat{R}_i^{(0)}, R_i^{(0)})^2 \right]$$

where $d_{\mathrm{SO}(3)}$ is the geodesic distance on SO(3).

**Critical limitation**: RFdiffusion generates **backbone frames only** — it produces N-Cα-C coordinates and orientations but **no amino acid identities**. The generated backbone is a geometric scaffold without sequence information.

### 1.2 ProteinMPNN: Inverse Folding to Complete the Pipeline

To obtain a sequence for the generated backbone, RFdiffusion relies on **ProteinMPNN** (Dauparas et al., 2022), a graph neural network for inverse folding (structure → sequence):

```
  RFdiffusion                ProteinMPNN              AlphaFold2
  ──────────                ───────────              ──────────
  Pure noise                Backbone                 Designed
  → denoise                 → inverse fold           → refold
  → backbone                → sequence               → validate

  ┌────────────┐           ┌────────────┐           ┌────────────┐
  │ SE(3) Diff │  Cα+ori   │  GNN       │  sequence │ Structure  │
  │ on RF      │ ────────→ │  Encoder + │ ────────→ │ Prediction │
  │ Trunk      │           │  AR Decoder│           │            │
  └────────────┘           └────────────┘           └────────────┘
```

**ProteinMPNN architecture**:
- **Input**: Backbone coordinates (N, Cα, C, O, virtual Cβ) → K-nearest neighbor graph ($k = 30$)
- **Encoder**: 3-layer message passing GNN with edge features from inter-residue distances and orientations
- **Decoder**: Autoregressive sequence generation — residues predicted one at a time, conditioned on previously generated residues
- **Performance**: 52.4% sequence recovery on native backbones (vs. 32.9% for Rosetta)

The three-stage pipeline — RFdiffusion → ProteinMPNN → AF2 validation — became the most experimentally validated computational design workflow, with hundreds of designs confirmed in wet-lab experiments.

### 1.3 The RFdiffusion Family

RFdiffusion spawned several specialized variants:

```
RoseTTAFold (prediction)
     │
     ▼
RFdiffusion (2023)────── Backbone-only design
     │
     ├── RFdiffusion-AA ── All-atom design (RFAA-based, small molecules)
     │
     ├── RFAntibody ─────── CDR loop design (template track)
     │
     └── RFdiffusion2 ──── Enzyme active site design (atomic resolution)
```

| Variant | Base Model | Scope | Key Innovation |
|---|---|---|---|
| RFdiffusion | RF1 | Backbone-only | SE(3) denoising fine-tune |
| RFdiffusion-AA | RFAA | All-atom + ligands | Atom-level diffusion with molecular graphs |
| RFAntibody | RF + templates | CDR loop design | Template conditioning for framework regions |
| RFdiffusion2 | RF | Enzyme active sites | Atomic-resolution catalytic geometry design |

### 1.4 FrameDiff and FrameFlow: Academic Alternatives

**FrameDiff** (Yim et al., 2023) and **FrameFlow** (Yim et al., 2024) provided academic alternatives operating on the same SE(3) frame representation:

- **FrameDiff**: Score-based diffusion on SE(3) frames, trained from scratch (no prediction model reuse)
- **FrameFlow**: Riemannian flow matching on SE(3), replacing stochastic diffusion with deterministic ODE flow — achieving comparable quality with fewer sampling steps

These models demonstrated that the SE(3) frame representation is effective for protein backbone generation regardless of whether the underlying generative framework is diffusion or flow matching.

---

## 2. Strategy 2: Conditional Generation on Co-Folding Models

### 2.1 The Key Insight: Prediction Models Already Know How to Generate

Strategy 1 requires a separate inverse folding model because backbone diffusion cannot produce sequences. Strategy 2 takes a fundamentally different approach: **modify a co-folding model (which already generates all-atom structures) to also generate sequences**, producing structure and sequence simultaneously in a single pass.

### 2.2 BoltzGen: Geometric Encoding of Residue Type

**BoltzGen** (2025) extends Boltz-2 (Part 4) from structure prediction to protein design. Its core innovation is a **geometric encoding** that represents amino acid identity as atomic coordinates — converting the discrete sequence design problem into a continuous one that the existing diffusion module can handle directly.

**The discrete-continuous challenge**: Protein design requires generating both:
- **Structure** (continuous): 3D coordinates of all atoms
- **Sequence** (discrete): amino acid identity at each position (20 categories)

Diffusion models operate in continuous space — they cannot natively generate discrete tokens. BoltzGen's solution: encode amino acid type as geometry.

**14-atom fixed representation**: Every residue, regardless of its amino acid type, is represented by exactly 14 atoms:

```
Atoms 1-4:     Backbone atoms (N, Cα, C, O)
               → Identical positions for all 20 amino acid types

Atoms 5-14:    Virtual atoms
               → Positions encode amino acid identity
               → Placed at specific locations relative to
                 backbone atoms (above N, Cα, or O)
```

The encoding scheme places virtual atoms at defined positions relative to backbone reference points. The number and placement of virtual atoms above each backbone atom uniquely identifies the amino acid type:

```
Example encodings:
  Glycine (G):      All virtual atoms collapsed to Cα
                    (smallest amino acid, no side chain)

  Alanine (A):      1 virtual atom above Cα
                    (single methyl side chain)

  Threonine (T):    3 atoms above N, 4 atoms above O
                    (hydroxyl + methyl branches)

  Tryptophan (W):   Complex distribution across all positions
                    (largest side chain, indole ring)
```

**Decoding**: At inference time, the amino acid type is recovered by counting virtual atoms within 0.5 Å of each backbone reference atom — a simple geometric classification.

**Why this works**: By encoding sequence as geometry, the entire design problem — structure AND sequence — becomes a single continuous coordinate prediction task. The existing EDM diffusion module (Part 3) generates all 14 × $L$ atom coordinates simultaneously, and the amino acid identities emerge from the geometric pattern of the generated virtual atoms.

### 2.3 BoltzMasker: From Prediction to Conditional Generation

BoltzGen converts Boltz-2 from a prediction model to a conditional generative model through **masking at design positions**:

```
Input to Boltz-2 (prediction):
  Target protein + Binder sequence + Binder structure features
  → Predicts: binder structure (given known sequence)

Input to BoltzGen (design):
  Target protein + [MASKED binder] + Design specification
  → Generates: binder structure AND sequence (both unknown)
```

At design positions, the masker replaces:
- **Token level**: residue type → UNK, MSA profile → uniform, templates → zero
- **Atom level**: element types → mask token, charges → 0
- **Optional**: backbone coordinates can also be masked for de novo scaffold design

The model learns the conditional distribution:

$$p(\mathbf{X}_{\mathrm{design}}, \mathbf{s}_{\mathrm{design}} \mid \mathbf{X}_{\mathrm{target}}, \mathbf{s}_{\mathrm{target}})$$

where $\mathbf{X}$ denotes coordinates (in the 14-atom representation) and $\mathbf{s}$ denotes sequence — but since sequence is encoded geometrically, both are realized as coordinate generation.

### 2.4 Dilated Noise Schedule

BoltzGen introduces a **dilated noise schedule** — a modification to the standard EDM schedule that allocates more denoising steps to the critical noise level where amino acid identity is determined.

**Observation**: Analysis of the denoising trajectory reveals that residue types are resolved in a narrow noise interval $\tau \in [0.6, 0.8]$ of the normalized schedule. Below this range (low noise), backbone geometry is refined. Above it (high noise), only gross topology is established.

**Standard EDM schedule** (Part 3):

$$t_i = \sigma_{\mathrm{data}} \cdot \left( s_{\max}^{1/\rho} + \frac{i}{N-1} \left( s_{\min}^{1/\rho} - s_{\max}^{1/\rho} \right) \right)^\rho$$

**Dilated schedule**: The normalized time variable $\tau \in [0, 1]$ is remapped through a piecewise linear function $\varphi(\tau)$ that expands the interval $[\tau_s, \tau_e] = [0.6, 0.8]$ by a factor $\lambda \approx 8/3$:

$$\varphi(\tau) = \begin{cases} \tau / r & \text{if } \tau \lt l \\ (\tau - l) / \lambda + l & \text{if } l \leq \tau \leq u \\ (\tau - u) / r + u & \text{if } \tau \gt u \end{cases}$$

where $r = \frac{1 - \lambda(\tau\_e - \tau\_s)}{1 - (\tau\_e - \tau\_s)}$, $l = r \cdot \tau\_s$, and $u = l + \lambda(\tau\_e - \tau\_s)$.

This allocates approximately $2.7 \times$ more function evaluations to the amino acid determination phase, improving sequence recovery without increasing total step count significantly (200 → 300 steps in batch mode).

### 2.5 Multi-Task Training

BoltzGen maintains Boltz-2's prediction capability while adding design through a multi-task training distribution:

| Task | Description | Sampling Probability |
|---|---|---|
| Folding | Structure prediction (preserves Boltz-2 accuracy) | 5–10% |
| Binder design (full chain) | Design entire protein against target | ~40% |
| Binder design (interface) | Design interface residues on fixed scaffold | ~10% |
| Motif scaffolding | Design scaffold around functional motif | 10–20% |
| Unconditional generation | Generate without target conditioning | 10–40% |

**Training data**: 60% PDB experimental structures, 30% AlphaFold DB distillation, 10% Boltz-1 distillation (ligand, RNA, DNA complexes).

### 2.6 BoltzGen Inverse Folding Refinement

Although the geometric encoding already determines residue types, BoltzGen includes an optional **inverse folding refinement** step using a lightweight GNN:

- **Encoder**: 3-layer MLP-Attention GNN on Cα K-nearest neighbor graph ($k = 30$)
- **Decoder**: Autoregressive sequence prediction, design positions in random order
- **Temperature**: $\tau = 0.1$ (near-greedy sampling)
- **Purpose**: Correct residue type errors from imperfect virtual atom placement

### 2.7 Quality-Diversity Selection

For practical design campaigns, BoltzGen generates large candidate pools (e.g., 60,000 designs per target) and selects a diverse, high-quality subset:

$$x^* = \arg\max_{x \in S \setminus A} \left[ \alpha \cdot \mathrm{Diversity}(x, A) + (1 - \alpha) \cdot \mathrm{Quality}(x) \right]$$

where $\mathrm{Diversity}(x, A) = 1 - \left( w_{\mathrm{struct}} \cdot \max_{a \in A} \mathrm{TM}(x, a) + w_{\mathrm{seq}} \cdot \max_{a \in A} \mathrm{SeqSim}(x, a) \right)$ and $A$ is the already-selected set. Greedy selection adds designs iteratively with $\alpha = 0.001$ for protein binders (quality-focused) and $\alpha = 0.01$ for peptides (diversity-balanced).

### 2.8 Experimental Validation

BoltzGen was validated on **9 novel targets** with less than 30% sequence identity to any known binder:

| Design Type | Candidates | Tested | Hits (nM affinity) | Success Rate |
|---|---|---|---|---|
| Nanobody binders | 60,000/target | 15/target | 6/9 targets | 66% |
| Protein binders | 60,000/target | 15/target | 6/9 targets | 66% |

Notable results: PHYH nanobody at 7.8 nM, PMVK nanobody at 6.1 nM. The system also demonstrated success across diverse modalities: linear peptides, cyclic peptides, disulfide-bonded peptides, and antimicrobial peptides (19.5% growth inhibition rate from 1,808 designs against *E. coli*).

### 2.9 Chai-2: Antibody Design from Improved Prediction

**Chai-2** (2025) extends Chai-1 (Part 1, Part 4) to de novo antibody design with a key observation: **design performance scales directly with prediction accuracy**.

Chai-2's improvements:
- **Folding accuracy**: 2× improvement in antibody-antigen DockQ > 0.8 (17% → 34%)
- **Design capability**: multimodal generation conditioned on target structure + ESM-2 embeddings

**Experimental results** (52 therapeutically relevant targets):

| Format | Hit Rate | Comparison |
|---|---|---|
| Miniproteins | 68% (picomolar) | 3× RFdiffusion/AlphaProteo |
| VHH (nanobody) | 20% | vs. $\lt$ 0.1% prior computational |
| scFv | 14% | vs. $\lt$ 0.1% prior computational |
| Full-length IgG | 86% met developability | 88 designs tested |

Landmark achievements include: first computational de novo binder for TNFα (where all previous methods failed), computational antibody agonists for GPCRs (CCR8, CXCR4), and pMHC-selective binders (KRAS G12V at 1.5 nM with single-residue selectivity).

---

## 3. Strategy 3: Flow Matching in Latent Space

### 3.1 Proteina: Scaling Laws for Protein Design

**Proteina** (NVIDIA, ICLR 2025 Oral) established that protein backbone generation follows **scaling laws** — larger models trained on more data produce consistently better designs, with no saturation observed:

- **Framework**: Flow matching on Cα coordinates (deterministic ODE, not stochastic diffusion)
- **Architecture**: Transformer-based denoiser with pair conditioning
- **Key finding**: Log-linear improvement in designability with model size and training data
- **Large-scale synthetic data**: Millions of AF2-predicted structures used for training

Like RFdiffusion, Proteina generates **backbone only** — requiring a separate inverse folding step for sequence design.

### 3.2 La-Proteina: Partially Latent Flow Matching

**La-Proteina** (2025) addresses the backbone-only limitation through **Partially Latent Flow Matching** — an elegant solution to the discrete-continuous challenge that differs fundamentally from BoltzGen's geometric encoding.

**Core idea**: Decompose each residue into two components with different representations:

```
Each residue i:
  ┌──────────────────────────────────────────────┐
  │  Cα coordinate  x_i ∈ R³                     │  ← Explicit (flow matching)
  │                                               │
  │  Sequence + side-chain  →  z_i ∈ R^d          │  ← Latent (VAE-encoded)
  │    amino acid type (discrete, 20 classes)      │
  │    side-chain torsion angles (continuous)      │
  │    all non-Cα atom positions (continuous)      │
  └──────────────────────────────────────────────┘
```

**The VAE component**:

A Variational Autoencoder maps each residue's sequence identity and side-chain geometry into a continuous latent vector $z_i \in \mathbb{R}^d$:

- **Encoder**: $q\_\phi(z\_i \mid s\_i, \chi\_i, x\_i^{\mathrm{bb}})$ — encodes amino acid type $s\_i$, side-chain torsion angles $\chi\_i$, and local backbone context into latent $z\_i$
- **Decoder**: $p_\psi(s_i, \chi_i \mid z_i, x_i^{\mathrm{Cα}})$ — reconstructs sequence and side-chain from latent + Cα position
- **Training**: Standard VAE objective with KL regularization

**The flow matching component**:

Flow matching operates on the **joint** space of Cα coordinates and latent variables:

$$\frac{d}{dt} \begin{pmatrix} x^{\mathrm{Cα}}_t \\ z_t \end{pmatrix} = v_\theta\left( \begin{pmatrix} x^{\mathrm{Cα}}_t \\ z_t \end{pmatrix}, t, c \right)$$

where $v_\theta$ is a learned velocity field, $t \in [0, 1]$ is the flow time, and $c$ encodes conditioning (e.g., target structure). The flow transforms a simple prior (Gaussian) at $t = 0$ to the data distribution at $t = 1$.

**The generation pipeline**:

```
  t=0: Gaussian noise                    t=1: Protein
  ┌─────────────┐                       ┌─────────────┐
  │ x_Cα ~ N(0,I)│   Flow matching      │ x_Cα (coords)│
  │ z ~ N(0,I)   │ ─────────────────→   │ z (latent)   │
  └─────────────┘    ODE integration    └──────┬──────┘
                                                │
                                         VAE Decoder
                                                │
                                                ▼
                                       ┌──────────────┐
                                       │ Sequence s_i  │
                                       │ Side-chain χ_i│
                                       │ All-atom X_i  │
                                       └──────────────┘
```

**Why "partially latent"?** The Cα coordinates remain in explicit 3D space (interpretable, physically meaningful), while only the sequence and side-chain information is encoded into latent space. This hybrid is more interpretable than a fully latent model and avoids the need for a separate inverse folding step.

**Performance**: Co-designability > 75% (2× previous methods), generation up to 800 residues.

### 3.3 Complexa: Extending to Binder-Target Complexes

**Complexa** (Proteina-Complexa, 2025) extends La-Proteina from unconditional protein generation to **binder design against specific targets**:

```
La-Proteina (unconditional)         Complexa (conditional)
─────────────────────────          ──────────────────────

  Generate protein in vacuum        Generate binder given target
  No target conditioning             Target structure as condition
  Monomer only                       Binder-target complex
```

**Key innovations**:

- **Teddymer dataset**: Large-scale synthetic binder-target pairs generated for training
- **Test-time optimization**: Gradient-based refinement of generated designs against target-specific objectives
- **Conditional flow matching**: Target structure encoded as fixed conditioning, binder generated via flow

Complexa achieves state-of-the-art on binder design benchmarks, demonstrating that the La-Proteina framework scales from unconditional generation to conditional complex design.

### 3.4 BoltzGen vs. La-Proteina: Two Solutions to the Same Problem

Both BoltzGen and La-Proteina solve the same fundamental challenge — **simultaneously generating discrete sequences and continuous structures** — but through diametrically opposite approaches:

```
The Challenge: Generate both sequence (discrete) and structure (continuous)

BoltzGen's Answer:                    La-Proteina's Answer:
"Make sequence continuous"            "Make sequence latent"

  Amino acid type                      Amino acid type
       │                                    │
       ▼                                    ▼
  14-atom geometric                    VAE encoder
  encoding                                  │
       │                                    ▼
       ▼                              Latent vector z_i
  Atom coordinates                          │
  (all continuous)                          ▼
       │                              Flow matching on
       ▼                              (Cα, z) jointly
  EDM Diffusion                             │
  generates all coords                      ▼
       │                              VAE Decoder
       ▼                              recovers sequence
  Count virtual atoms                 + side-chain
  → amino acid type
```

| Aspect | BoltzGen (Geometric) | La-Proteina (Latent) |
|---|---|---|
| **Discrete → continuous mapping** | Explicit: virtual atom coordinates | Implicit: VAE latent space |
| **Generative framework** | EDM diffusion (200–300 steps) | Flow matching (ODE, fewer steps) |
| **Backbone source** | Reuses Boltz-2 diffusion module | Dedicated flow matching |
| **Sequence recovery** | From geometry + inverse fold refinement | From VAE decoder |
| **Advantages** | No additional encoder/decoder training; reuses existing pipeline | Natural continuous representation; scaling-friendly |
| **Limitations** | 14-atom representation is hand-designed | VAE training adds complexity; latent quality limits output |
| **Complex design** | Native (Boltz-2 handles all molecular types) | Via Complexa extension |
| **Experimental validation** | 66% hit rate on novel targets (9 targets) | Co-designability > 75% |

The comparison reveals a broader tension in the field: **explicit engineering** (BoltzGen's geometric encoding) versus **learned abstraction** (La-Proteina's VAE latent). The explicit approach is simpler to implement and debug but may not generalize to new molecular types. The learned approach is more flexible but introduces an additional training stage and potential information bottleneck.

---

## 4. Strategy 4: Discrete Multimodal Generation

### 4.1 ESM3: A Fundamentally Different Paradigm

**ESM3** (EvolutionaryScale, 2024) departs entirely from the diffusion/flow matching paradigm. Instead of generating continuous coordinates, ESM3 treats **all modalities — sequence, structure, and function — as discrete tokens** and generates them via masked prediction, identical to how large language models generate text.

**Three-track tokenization**:

```
  Track 1 — Sequence:    Standard amino acid tokens (20 + special)
  Track 2 — Structure:   VQ-VAE encoded structure tokens (4,096 codes)
  Track 3 — Function:    InterPro annotations, keywords (tokenized)
```

The structure track is the most innovative: a **Vector-Quantized VAE (VQ-VAE)** is trained to compress local 3D structure around each residue into a single discrete token from a codebook of 4,096 entries. This lossy compression enables the model to reason about structure using the same discrete attention mechanisms it uses for sequence and function.

**Generation via iterative unmasking**:

$$p(\mathbf{x}_{\mathrm{masked}} \mid \mathbf{x}_{\mathrm{observed}}) = \prod_{t=1}^{T} p(x_{m_t} \mid \mathbf{x}_{\mathrm{observed}}, x_{m_1}, \ldots, x_{m_{t-1}})$$

where $m_1, \ldots, m_T$ is an ordering of masked positions (typically highest-confidence-first). Starting from a fully masked input, the model iteratively predicts the most confident position, unmasks it, and conditions subsequent predictions on the growing set of revealed tokens.

```
Step 0: [MASK] [MASK] [MASK] [MASK] [MASK]   (all masked)
Step 1: [MASK]   M   [MASK] [MASK] [MASK]    (highest confidence)
Step 2: [MASK]   M   [MASK]   L   [MASK]
Step 3:   K      M   [MASK]   L   [MASK]
Step 4:   K      M     F      L   [MASK]
Step 5:   K      M     F      L     I        (complete)
```

This process runs simultaneously across all three tracks — sequence, structure, and function tokens are unmasked in an interleaved fashion, allowing cross-modal conditioning at every step.

### 4.2 Scale and Capability

| Model | Parameters | Training Compute | Training Data |
|---|---|---|---|
| ESM3-open | 1.4B | — | — |
| ESM3 Large | 98B | $\sim 10^{24}$ FLOPs | 771B tokens |

**Landmark result — esmGFP**: ESM3 generated a novel fluorescent protein (esmGFP) with only ~58% sequence similarity to the nearest natural GFP — corresponding to approximately 500 million years of natural evolutionary distance. The protein was experimentally confirmed to fluoresce, demonstrating that discrete multimodal generation can produce functional proteins.

### 4.3 Diffusion vs. Discrete Generation

| Dimension | Diffusion / Flow Matching | Discrete (ESM3) |
|---|---|---|
| **Structure representation** | Continuous 3D coordinates | Discrete VQ-VAE tokens |
| **Generation process** | Noise → denoise (continuous) | Mask → unmask (discrete) |
| **Sequence generation** | Geometric encoding or VAE latent | Native (same token space) |
| **Function integration** | Difficult (separate head) | Seamless (third track) |
| **Atomic precision** | Sub-angstrom | Limited by VQ-VAE codebook |
| **Diversity control** | Sampling noise level | Masking rate + temperature |
| **Scaling properties** | Proven for backbone | LLM-style scaling (favorable) |

**The precision trade-off**: ESM3's VQ-VAE compression inevitably loses fine structural detail. While sufficient for global fold prediction, it may limit accuracy for applications requiring sub-angstrom precision — such as enzyme active site design or small molecule binding pose prediction. Diffusion and flow matching models, operating in continuous coordinate space, do not face this limitation.

**The scaling advantage**: Discrete token models benefit from the enormous infrastructure (hardware, software, training recipes) developed for LLMs. ESM3's 98B parameter model demonstrates that protein generation can scale to sizes comparable to frontier language models — a regime where diffusion models have not yet been explored.

---

## Part B: Synthesis and Outlook

## 5. Four-Strategy Comparison

| | Strategy 1 | Strategy 2 | Strategy 3 | Strategy 4 |
|---|---|---|---|---|
| **Approach** | SE(3) Diffusion | Conditional Generation | Latent Flow Matching | Discrete Multimodal |
| **Representative** | RFdiffusion | BoltzGen, Chai-2 | La-Proteina, Complexa | ESM3 |
| **Generates** | Backbone only | Structure + Sequence | Structure + Sequence | Struct + Seq + Function |
| **Sequence method** | Separate (ProteinMPNN) | Geometric encoding | VAE latent | Native tokens |
| **Framework** | SE(3) diffusion | EDM diffusion | Flow matching | Masked LM |
| **Base model** | RoseTTAFold | Boltz-2 / Chai-1 | Trained from scratch | ESM-2 |
| **Complex design** | Limited | Full (any molecular type) | Via Complexa | No |
| **Experimental hits** | Hundreds validated | 66% on novel targets | Early stage | esmGFP confirmed |
| **Maturity** | Most mature | Rapidly advancing | Early but promising | Early but scalable |
| **Scaling** | Moderate | Moderate | Favorable | Most favorable |

---

## 6. Convergence and Open Questions

### Where the field agrees

- **Trunk reuse is effective**: Fine-tuning or extending prediction model representations for generation consistently outperforms training design models from scratch. The Trunk captures protein physics that transfers directly to generation.
- **Joint structure-sequence generation is superior**: Strategies 2–4, which co-generate structure and sequence, avoid the information loss inherent in the two-stage pipeline (Strategy 1), where the inverse folding model has no access to the design objective.
- **Experimental validation is the bottleneck**: Computational metrics (designability, co-designability, TM-score) are necessary but insufficient. The gap between computational success rates and experimental hit rates remains significant.

### What remains unresolved

**Which discrete-continuous bridge will win?** BoltzGen's geometric encoding, La-Proteina's VAE latent, and ESM3's VQ-VAE tokens represent three very different solutions to the same fundamental problem. Each has trade-offs in precision, flexibility, and scalability. The optimal approach likely depends on the application: geometric encoding for atomic-precision tasks, latent space for scalable generation, discrete tokens for multi-property optimization.

**Can design models achieve reliable function?** Current models optimize for structural metrics (fold stability, binding geometry) as proxies for function. But designing a protein that actually catalyzes a reaction, signals through a pathway, or achieves therapeutic efficacy requires functional understanding that goes beyond structure prediction. The gap between designed-to-fold and designed-to-function remains the field's central challenge.

**Will scaling continue to help?** Proteina demonstrated scaling laws for backbone generation, and ESM3 showed that 98B parameters produce better designs than 1.4B. But it is unclear whether scaling alone can overcome the fundamental challenges of protein design — or whether architectural innovations (better discrete-continuous bridges, physics-informed losses, active learning from experimental data) are equally necessary.

---

**Next: Part 6 — How to Predict Protein Dynamics and Ensembles? Beyond Static Structures**

We move from single-structure prediction and design to the frontier of conformational ensemble modeling — examining how models like AlphaFlow, BioEmu, and Boltzmann generators capture the dynamic nature of proteins.

---

*Part of the series: The Technical Evolution of Protein AI — A Record of Key Design Decisions*
