---
title: "Protein AI Series Part 4: Co-Folding and the Open-Source Race"
date: 2026-03-17 09:40:00 +0900
categories: [Drug Discovery, Foundation Model]
tags: [protein-ai, co-folding, alphafold3, boltz, open-source]
math: true
---

**The Technical Evolution of Protein AI — A Record of Key Design Decisions**

*This is Part 4 of a 10-part series tracing the architectural choices behind modern protein structure prediction and design models.*

---

## The Core Question

**How do we handle proteins, nucleic acids, ligands, ions, and modified residues within a single model?**

Parts 1–3 focused on the core prediction pipeline: input representation → Trunk → Structure Generation. But real biology involves **heterogeneous molecular complexes** — a kinase bound to a small-molecule inhibitor, an antibody recognizing a glycosylated antigen, a ribosome with mRNA and tRNA. This Part traces how models evolved from protein-only prediction to universal co-folding, and examines the open-source ecosystem that emerged in parallel.

---

## Part A: The Technical Evolution of Co-Folding

## 1. The Representation Problem

Different molecular types have fundamentally different structures:

| Molecular Type | Structure | Natural Unit | Examples |
|---|---|---|---|
| Proteins | Linear polymer | Residue (backbone + side-chain) | Enzymes, antibodies |
| Nucleic acids | Linear polymer | Nucleotide (sugar + base + phosphate) | DNA, RNA |
| Small molecules | Arbitrary graph | Atom | Drug-like ligands, cofactors |
| Ions / water | Single atom | Atom | Zn²⁺, Mg²⁺, Ca²⁺ |
| Modified residues | Residue + modification | Hybrid | Phosphoserine, glycosylation |

The core challenge: **polymers are naturally described at the residue level (one token per residue), but ligands and ions require atom-level description**. How do you run a unified Trunk (Part 2) and Structure Generation (Part 3) over tokens that represent fundamentally different things?

---

## 2. AF2 → AF2-Multimer → AF3: The DeepMind Progression

### 2.1 AF2: One Token = One Residue, Proteins Only

AlphaFold2 tokenizes exclusively at the residue level. Each token represents one amino acid — its backbone atoms (N, Cα, C, O) and side-chain atoms are grouped under a single token. The pair representation $z_{ij}$ encodes residue-residue relationships.

This design is elegant for proteins but **structurally incapable of representing non-polymeric molecules**. A drug molecule with 30 atoms doesn't have residues, backbone, or side-chains.

### 2.2 AF2-Multimer: Multi-Chain Extension

AF2-Multimer (2021) extended AF2 to protein complexes with multiple chains:

- **Paired MSA**: Cross-chain evolutionary coupling via paired sequence search
- **Chain permutation**: Handle symmetric complexes where chain labels are interchangeable
- **Relative chain encoding**: Add inter-chain positional features to the pair representation

But the fundamental tokenization remained protein-residue-only — no ligands, no nucleic acids.

### 2.3 AF3: Mixed Tokenization + AtomAttention

AlphaFold3 (2024) introduced **mixed tokenization** — the key architectural innovation enabling universal co-folding:

```
Standard residues (protein, DNA, RNA):
  1 token = 1 residue (as in AF2)
  → reference atoms define the token's position

Non-standard entities (ligands, ions, modified residues):
  1 token = 1 atom
  → each atom is its own token
```

This creates a heterogeneous token sequence where some tokens represent entire residues (~14 atoms) and others represent single atoms. To bridge these scales, AF3 introduced the **AtomAttention Encoder/Decoder**:

```
                All atoms (raw coordinates)
                         │
                         ▼
              ┌─────────────────────┐
              │  Atom Encoder       │
              │  (3 layers)         │
              │  Atom → Token       │
              │  Window attention   │
              │  (Q=32, K=128)      │
              └──────────┬──────────┘
                         │
              Token-level representations
                         │
                         ▼
              ┌─────────────────────┐
              │  Pairformer Trunk   │
              │  (48 blocks)        │
              │  + Token Transformer│
              │  in diffusion       │
              └──────────┬──────────┘
                         │
              Token-level representations
                         │
                         ▼
              ┌─────────────────────┐
              │  Atom Decoder       │
              │  (3 layers)         │
              │  Token → Atom       │
              │  Predict Δ coords   │
              └─────────────────────┘
                         │
                         ▼
                All-atom coordinates
```

The Atom Encoder uses **windowed attention** (query window Q=32, key window K=128) to aggregate atom-level features into token-level representations. Within each window, atoms belonging to the same token (e.g., all atoms of a residue) attend to each other and to nearby atoms from other tokens. The Atom Decoder reverses this mapping, producing per-atom coordinate updates from token-level predictions.

This design means the Trunk (Pairformer) and the Token Transformer in the diffusion module operate at the **token level** — the same resolution regardless of molecular type. The atom-to-token and token-to-atom conversions happen in thin wrapper layers (3 layers each), keeping the core architecture unchanged.

---

## 3. RFAA: The Hybrid Alternative

**RoseTTAFold All-Atom** (RFAA, Baker Lab, 2024) took a different approach to multi-molecular modeling:

- **Proteins and nucleic acids**: processed through the standard RF2 three-track architecture (1D/2D/3D) at residue resolution
- **Small molecules**: represented as **atom graphs** with learned embeddings, processed through a separate graph neural network
- **Cross-modal interaction**: protein-residue and ligand-atom features interact through cross-attention layers

This hybrid design contrasts with AF3's unified tokenization. Instead of forcing everything into the same token space, RFAA maintains type-specific representations and lets them interact through explicit cross-modal layers.

**Trade-offs**:
- RFAA's hybrid approach requires designing cross-modal interaction mechanisms for each pair of molecular types
- AF3's unified tokenization handles any combination automatically through the same attention mechanism
- RFAA's design may be more expressive for individual molecular types but less scalable to new entity types

RFAA's significance extends beyond structure prediction — it became the backbone for **RFdiffusion-AA**, which extended backbone design (RFdiffusion) to include small molecule interactions (e.g., designing proteins that bind specific drug molecules).

---

## 4. NP1 → NP2 → NP3: Incremental Expansion

The NeuralPLexer lineage shows a gradual expansion of scope:

| Version | Year | Scope | Architecture | Generation |
|---|---|---|---|---|
| NP1 | 2022 | Protein-ligand only | SE(3)-equivariant GNN | SE(3) diffusion |
| NP2 | 2024 | All biomolecules | Extended GNN | SE(3) diffusion |
| NP3 | 2025 | All biomolecules | Encoder-Decoder (PairFormer + DiT) | Flow matching |

NP3's transition to the PairFormer-based encoder (Part 2) and flow matching decoder (Part 3) represents a convergence with the AF3 architectural paradigm — while adding its own innovations (polymer prior, encoder-decoder separation, Flash-TriangularAttention).

NP3 introduces a two-level atom handling strategy:
- **Anchor atoms** (one per residue/nucleotide, or each atom for ligands): processed with dense attention in the encoder
- **All heavy atoms**: processed with sliding window attention in the decoder, achieving $O(N)$ scaling

This design allows NP3 to handle large complexes efficiently while maintaining atomic resolution where it matters most (ligand binding sites, protein-nucleic acid interfaces).

---

## Part B: The Open-Source Reproduction Race

## 5. The Landscape After AF3

When AlphaFold3 was published (May 2024), its code was not initially released. The paper's restricted academic license (released November 2024) further motivated the community to build open alternatives. The result was an unprecedented race to reproduce — and extend — AF3's capabilities.

```
AF2 (2021)
  ├── OpenFold (2022) ─── Apache 2.0, AF2 reproduction
  └── UniFold (2022) ──── Apache 2.0, AF2 reproduction (DP Technology)

AF3 paper (2024.05)
  │
  ├── Boltz-1 (2024.09) ─── MIT license, full training code
  │     └── Boltz-2 (2025) ─── + Affinity head
  │           └── BoltzGen (2025) ─── + Design capability
  │
  ├── Chai-1 (2024.09) ─── Apache 2.0, inference only
  │     └── Chai-2 (2025) ─── Antibody design (closed)
  │
  ├── Protenix (2024.10) ─── ByteDance, MSA module changes
  │
  ├── SeedFold (2025) ──── ByteDance, linear TriAtt + wider Pairformer
  │
  ├── AF3 code release (2024.11) ─── Academic license
  │
  └── OpenFold3 (2025.10) ─── Apache 2.0, full reproduction
```

### 5.1 AF2 Reproductions: OpenFold and UniFold

**OpenFold** (Columbia University et al., 2022, Apache 2.0) established the template: a complete, permissively licensed reimplementation of AF2 with full training code and weights. Its contributions went beyond reproduction:

- First application of **FlashAttention** to protein structure prediction
- Compatible with AF2 weights (identical predictions)
- Became the foundation for AlphaFlow and numerous fine-tuning studies

**UniFold** (DP Technology, 2022, Apache 2.0) independently reproduced AF2's Evoformer architecture, providing another fully open training pipeline. UniFold served as the foundation for DP Technology's subsequent models, including Uni-Fold Symmetry for symmetric complexes. Together with OpenFold, it validated that AF2's methodology was complete and reproducible — establishing the expectation that major models should be openly reimplementable.

**OpenFold3** (October 2025, Apache 2.0) applied the same philosophy to AF3:

| Aspect | Detail |
|---|---|
| License | Apache 2.0 (commercial use permitted) |
| Training data | 300K experimental structures + 13M synthetic structures |
| Consortium | Columbia University, LLNL, Seoul National University, SandboxAQ |
| Key strength | **RNA structure prediction on par with AF3** — the only open-source model to achieve this |
| Training code | Fully open |

OpenFold3's RNA performance is notable because RNA structure prediction is widely considered harder than protein structure prediction (fewer training examples, more conformational flexibility), and most AF3 reproductions underperform significantly on RNA benchmarks.

### 5.2 Boltz-1/2: The Community Workhorse

Boltz-1 (MIT license, September 2024) was the first fully open AF3 reproduction with complete training code. Boltz-2 (2025) added the most significant architectural extension:

**Affinity Dual Head**: Boltz-2 predicts binding affinity alongside structure — a binary binder/non-binder classification and a continuous affinity value — using a dedicated head that reads the same Trunk representations.

- Won the **CASP16 affinity prediction challenge** (outperforming all other methods)
- Achieves FEP+-comparable accuracy at **~1000× lower computational cost** (Pearson $r = 0.62$ vs FEP+ benchmarks)
- Complete three-stage training pipeline publicly available: structure → confidence → affinity

Boltz-2's MIT license and complete training pipeline make it the most adopted starting point for new research — from fine-tuning on specialized protein families to developing new output heads.

### 5.3 Protenix and SeedFold: ByteDance's Two Approaches

**Protenix** (ByteDance, 2024) reproduced AF3 with modifications to the MSA module's signal flow, achieving modest improvements over the AF3 baseline. Its training code is public, though the license terms are less permissive than Boltz or OpenFold3.

**SeedFold** (ByteDance, 2025) takes a more ambitious approach — rather than faithfully reproducing AF3, it modifies the Pairformer architecture itself:

- **Linear Triangle Attention**: Replaces the standard $O(L^3)$ triangle attention with a linearized variant, reducing complexity to sub-cubic scaling (see Part 2 for details)
- **Wider Pairformer representations**: Expands the pair representation dimension beyond AF3's default of 128, increasing per-layer capacity
- **Result**: Outperforms AF3 on most protein-related benchmarks

ByteDance's dual strategy — faithful reproduction (Protenix) alongside architectural innovation (SeedFold) — illustrates that reproduction is a stepping stone, not the end goal.

### 5.4 Chai-1: Inference-Only but Architecturally Distinct

**Chai-1** (Apache 2.0, September 2024) is architecturally the most distinct AF3-family model:

- **ESM-2 3B integration** as a separate input track (Part 1) — unique among AF3-family models
- **Heun sampler** (second-order ODE solver) instead of Euler — 2× model evaluations per step but better trajectory accuracy
- Multi-track input: MSA + templates + ESM-2 + experimental restraints

However, Chai-1's **training code is not publicly available** — only inference code and weights are released. This limits its utility as a research platform despite its permissive license.

### 5.5 Comparison Table: Open-Source Ecosystem

**AF2 Reproductions**:

| | OpenFold | UniFold |
|---|---|---|
| **Developer** | Columbia University et al. | DP Technology |
| **Year** | 2022 | 2022 |
| **License** | Apache 2.0 | Apache 2.0 |
| **Key contribution** | FlashAttention, AF2 weight compatibility | Independent reproduction, Uni-Fold Symmetry |

**AF3-family Models**:

| | OpenFold3 | Boltz-2 | Protenix | SeedFold | Chai-1 | AF3 (official) |
|---|---|---|---|---|---|---|
| **Developer** | Columbia et al. | MIT/Recursion | ByteDance | ByteDance | Chai Discovery | DeepMind |
| **License** | Apache 2.0 | MIT | Mixed | — | Apache 2.0 | Academic only |
| **Training code** | Full | Full | Full | — | Inference only | Limited |
| **Commercial use** | Yes | Yes | Varies | — | Yes | No |
| **Unique feature** | RNA strength | Affinity head | MSA variant | Linear TriAtt, wider pair | ESM-2 3B | Reference model |
| **vs AF3 accuracy** | Matches | Matches | Slightly better | **Exceeds** | Matches | Baseline |
| **Community adoption** | Growing | Highest | Moderate | — | Moderate | Reference |

---

## 6. IsoDDE: Why Reproduction ≠ Surpassing

**IsoDDE** (Isomorphic Labs, 2026) — the "Drug Design Engine" — demonstrates the gap between open-source reproductions and the frontier:

| Benchmark | IsoDDE | AF3 | Best Open-Source |
|---|---|---|---|
| Antibody-Antigen (DockQ > 0.8) | **39%** | ~17% | Similar to AF3 |
| CDR-H3 loop (RMSD ≤ 2 Å) | **70%** | 58% | ~60% |
| Pocket prediction (AUPRC) | **1.5× P2Rank** | — | — |

IsoDDE's advantages come not from architectural novelty but from:

1. **Multi-task learning**: Structure prediction, binding affinity, and pocket prediction are trained jointly on a shared representation — each task improves the others
2. **Scale of training data**: Access to proprietary experimental data (Isomorphic Labs' internal assay data) far exceeding public databases
3. **Compute budget**: Training at a scale that academic groups cannot match
4. **1000-seed multi-state inference**: Running 1000 independent diffusion trajectories per prediction, then selecting the best by confidence

This gap highlights a sobering reality: **architecture changes alone (the focus of open-source efforts) account for only part of model performance**. Training data diversity, compute scale, and multi-task synergies are equally — perhaps more — important.

---

## 7. Co-Folding Technical Comparison

| | AF3 | RFAA | NP3 | Boltz-2 |
|---|---|---|---|---|
| **Molecular representation** | Unified tokens | Hybrid (residue + graph) | Encoder-Decoder | Unified tokens |
| **Tokenization** | Residue + atom | Sequence + atom graph | Anchor + atom levels | Residue + atom |
| **Trunk** | Pairformer (48) | 3-track RF2 | PairFormer encoder | Pairformer (64) |
| **Structure generation** | EDM diffusion (200 steps) | SE(3) IPA | Flow matching (40 steps) | EDM diffusion (200 steps) |
| **Affinity prediction** | No | No | No | **Yes** (dual head) |
| **PLM integration** | No | No | ESM-2 + RiNALMo | No |
| **License** | Academic | BSD | Proprietary | MIT |
| **Training code** | Limited | Yes | No | Yes |

---

## 8. Convergence and Open Questions

### Where the field agrees

- **Unified tokenization** (AF3-style mixed residue + atom tokens) has become the dominant approach, adopted by Boltz, Chai, and OpenFold3. The alternative hybrid approach (RFAA) has not been widely adopted by other groups.
- **The Pairformer Trunk is universal** — every co-folding model (AF3, Boltz, Chai, NP3, OpenFold3, SeedFold) uses a Pairformer variant as its core representation engine, though SeedFold and Pairmixer (Part 2) differ on whether triangle attention should be linearized or removed.
- **Open-source models match or exceed AF3 on standard benchmarks** for protein structure prediction and protein-ligand docking. SeedFold exceeds AF3 on most tasks; the gap has largely closed for well-characterized targets.

### What remains unresolved

**Affinity prediction accuracy**: Boltz-2 and IsoDDE have added affinity prediction, but achieving physics-based accuracy (FEP+ level) for arbitrary protein-ligand pairs remains elusive. Current models excel at relative ranking but struggle with absolute $\Delta G$ prediction.

**The data moat**: IsoDDE's performance gap suggests that proprietary training data (especially experimental binding data) provides advantages that architectural innovation alone cannot overcome. Whether the open-source community can close this gap through synthetic data generation, data augmentation, or novel training strategies is an open question.

**RNA and non-canonical molecules**: While OpenFold3 matches AF3 on RNA, most other open-source models significantly underperform. Nucleic acid structure prediction remains less mature than protein prediction, with fewer training examples and greater conformational complexity.

---

**Next: Part 5 — How to Turn a Prediction Model into a Design Model? Four Strategies Compared**

We shift from prediction to design — examining how co-folding models like Boltz-2 can be repurposed for generating novel proteins, and comparing four distinct strategies: SE(3) diffusion, conditional generation, latent flow matching, and discrete multimodal generation.

---

*Part of the series: The Technical Evolution of Protein AI — A Record of Key Design Decisions*
