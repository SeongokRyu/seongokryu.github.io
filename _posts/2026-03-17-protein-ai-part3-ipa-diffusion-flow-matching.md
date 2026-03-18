---
title: "Protein AI Series Part 3: IPA → Diffusion → Flow Matching"
date: 2026-03-17 09:30:00 +0900
categories: [Drug Discovery, Foundation Model]
tags: [protein-ai, diffusion-model, flow-matching, structure-generation]
math: true
---

**The Technical Evolution of Protein AI — A Record of Key Design Decisions**

*This is Part 3 of a 10-part series tracing the architectural choices behind modern protein structure prediction and design models.*

---

## The Core Question

**How do we convert the Trunk's learned representations $(z, s)$ into 3D atomic coordinates?**

Parts 1–2 traced how the pair representation $z\_{ij}$ is built and refined. But $z\_{ij}$ is an abstract tensor — it encodes spatial relationships but is not itself a 3D structure. The **Structure Generation Module** bridges this gap, and its design has undergone the most dramatic paradigm shifts in the field:

- **IPA** (AF2, 2021): deterministic, 8 refinement steps
- **EDM Diffusion** (AF3/Boltz/Chai, 2024): stochastic, 200 denoising steps
- **SE(3) Diffusion** (RFdiffusion/FrameDiff, 2023): diffusion on rigid body frames
- **Flow Matching** (NP3/Proteina/FrameFlow, 2024–25): ODE-based, 40 steps
- **Langevin Dynamics** (BioEmu, 2024): thermodynamic ensemble sampling

Each paradigm reflects a different answer to the fundamental tension between **accuracy** (getting the single best structure right) and **diversity** (capturing the multiple conformations a protein can adopt).

---

## 1. IPA Structure Module (AlphaFold2) — Deterministic Prediction

### 1.1 The Frame-Based Approach

AlphaFold2's Structure Module operates on **rigid body frames** — one per residue:

$$T_i = (R_i, \vec{t}_i), \qquad R_i \in \mathrm{SO}(3),\; \vec{t}_i \in \mathbb{R}^3$$

Starting from identity frames ($R\_i = I$, $\vec{t}\_i = \vec{0}$), the module iteratively updates each residue's position and orientation through 8 layers of **Invariant Point Attention (IPA)**.

### 1.2 Invariant Point Attention

IPA combines three attention signals:

$$\alpha_{ij} = \text{softmax}\!\left(\underbrace{\frac{q_i^T k_j}{\sqrt{d}}}_{\text{sequence}} + \underbrace{b_{ij}(z)}_{\text{pair bias}} - \underbrace{\frac{w}{2}\|T_i \circ q_{\text{pt}} - T_j \circ k_{\text{pt}}\|^2}_{\text{point distance}}\right)$$

where:
- **Sequence term**: standard dot-product attention on single representation $s$
- **Pair bias**: learned projection of $z_{ij}$ (Part 2's structural blueprint)
- **Point distance**: learnable query/key points are placed in each residue's local frame, then compared in global coordinates

The point attention term is the key innovation: it injects **3D geometric awareness** directly into the attention weights. Points that are close in 3D space (after frame transformation) receive higher attention — a built-in inductive bias for spatial reasoning.

**SE(3)-invariance guarantee**: Because the point distance $\|T_i \circ q - T_j \circ k\|^2$ depends only on relative positions, the entire attention computation is invariant to global rotations and translations.

### 1.3 Pipeline

```
s_trunk, z_trunk ──→ IPA Layer 1 ──→ ... ──→ IPA Layer 8
                     (update frames)          (update frames)
                                                   │
                                                   ▼
                                         Backbone: N, Cα, C coordinates
                                                   │
                                                   ▼
                                         Side-chain: predict χ angles
                                                   │
                                                   ▼
                                         All-atom coordinates
```

### 1.4 Strengths and Limitations

**Strengths**: Fast (only 8 steps), accurate for single-structure prediction, physically grounded through SE(3)-equivariance.

**Limitations**: **Deterministic** — one input always produces one output. No mechanism for sampling alternative conformations. Additionally, the frame-based representation is inherently protein-centric: ligands, ions, and non-standard residues don't naturally have backbone frames, making all-atom extension awkward.

---

## 2. EDM Diffusion (AF3, Boltz-1/2, Chai-1) — The Generative Pivot

### 2.1 Why Diffusion?

AlphaFold3's switch from IPA to diffusion was motivated by two needs:

1. **All-atom generation**: Ligands, ions, and modified residues lack backbone frames. Diffusion operates on raw Cartesian coordinates — any atom can be denoised regardless of molecular type.
2. **Stochastic sampling**: By varying the random seed, diffusion generates different plausible structures from the same input, enabling confidence-based ranking and limited conformational exploration.

### 2.2 EDM Formulation

All three major co-folding models (AF3, Boltz-1/2, Chai-1) adopt the **EDM** (Elucidating the Design Space of Diffusion Models; Karras et al., 2022) formulation with identical hyperparameters.

**Forward process** (training — add noise to ground truth):

$$x_t = x_0 + \sigma_t \cdot \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I)$$

**Denoiser** (the neural network's task — predict the clean structure):

$$D_\theta(x_t, \sigma_t;\; z) \approx \mathbb{E}[x_0 \mid x_t]$$

**EDM preconditioning** (rescale network inputs/outputs for training stability):

$$D_\theta(x, \sigma) = c_{\text{skip}}(\sigma) \cdot x + c_{\text{out}}(\sigma) \cdot F_\theta\!\left(c_{\text{in}}(\sigma) \cdot x,\; c_{\text{noise}}(\sigma);\; z\right)$$

where:

$$c_{\text{skip}}(\sigma) = \frac{\sigma_{\text{data}}^2}{\sigma^2 + \sigma_{\text{data}}^2}, \qquad c_{\text{out}}(\sigma) = \frac{\sigma \cdot \sigma_{\text{data}}}{\sqrt{\sigma^2 + \sigma_{\text{data}}^2}}$$

$$c_{\text{in}}(\sigma) = \frac{1}{\sqrt{\sigma^2 + \sigma_{\text{data}}^2}}, \qquad c_{\text{noise}}(\sigma) = \frac{1}{4}\ln\!\frac{\sigma}{\sigma_{\text{data}}}$$

These ensure that $F\_\theta$'s inputs and outputs have approximately unit variance at every noise level — critical for stable gradient flow across the wide range $\sigma \in [\sigma_{\min}, \sigma_{\max}]$.

**Training loss**:

$$\mathcal{L}(\theta) = \mathbb{E}_{\sigma, x_0, \varepsilon}\left[w(\sigma) \cdot \|D_\theta(x_0 + \sigma\varepsilon,\; \sigma;\; z) - x_0\|^2\right]$$

with noise-level weighting $w(\sigma) = (\sigma^2 + \sigma_{\text{data}}^2) / (\sigma \cdot \sigma_{\text{data}})^2$ that emphasizes intermediate noise levels where the learning signal is strongest.

### 2.3 Shared Hyperparameters

Remarkably, AF3, Boltz, and Chai use **identical noise schedule parameters**:

| Parameter | Value | Meaning |
|---|---|---|
| $\sigma_{\min}$ | $4 \times 10^{-4}$ | Minimum noise (nearly clean) |
| $\sigma_{\max}$ | 160 | Maximum noise (pure noise) |
| $\sigma_{\text{data}}$ | 16 | Data standard deviation |
| $P_{\text{mean}}$ | -1.2 | Log-normal noise sampling center |
| $P_{\text{std}}$ | 1.5 | Log-normal noise sampling spread |
| $\rho$ | 7 | Inference schedule exponent |

Training samples noise as $\sigma \sim \sigma_{\text{data}} \cdot \exp(P_{\text{mean}} + P_{\text{std}} \cdot \mathcal{N}(0,1))$, yielding a median $\sigma \approx 4.8$.

### 2.4 Denoising Network Architecture

The denoising network $F_\theta$ follows a common three-stage design:

```
Noisy atom coords (x_t)
    │
    ▼
┌───────────────────────────────────┐
│  Atom Encoder (3 layers)          │
│  Window attention (Q=32, K=128)   │
│  Atom-level → Token-level         │
└────────────────┬──────────────────┘
                 │
                 ▼
┌───────────────────────────────────┐
│  Token Transformer (24 layers)    │
│  16 heads, 768 dim                │
│  Conditioned on: z_trunk, s_trunk,│
│    σ (noise level via Fourier)    │
└────────────────┬──────────────────┘
                 │
                 ▼
┌───────────────────────────────────┐
│  Atom Decoder (3 layers)          │
│  Token-level → Atom-level         │
│  Predicts Δx (coordinate update)  │
└───────────────────────────────────┘
```

The Atom Encoder compresses raw atomic coordinates into per-token (per-residue or per-ligand-atom) representations. The Token Transformer — the core of the denoising network — refines these using the Trunk's outputs as conditioning. The Atom Decoder maps back to atomic coordinates.

### 2.5 Inference

**ODE sampler** (first-order Euler, 200 steps):

$$x_{i-1} = x_i + (\sigma_{i-1} - \sigma_i) \cdot \frac{x_i - D_\theta(x_i, \sigma_i;\; z)}{\sigma_i}$$

starting from $x_T \sim \mathcal{N}(0, \sigma_{\max}^2 I)$.

**SDE sampler** (optional — adds stochastic noise at each step for greater diversity):

$$x_{i-1} = x_i + (\sigma_{i-1} - \sigma_i) \cdot \frac{x_i - D_\theta(x_i, \sigma_i;\; z)}{\sigma_i} + \sqrt{\sigma_{i-1}^2 - \sigma_i^2} \cdot \gamma \cdot \varepsilon_i$$

Chai-1 notably uses a **second-order Heun sampler** (2 model evaluations per step), which improves trajectory accuracy at the cost of doubled compute.

### 2.6 Model-Specific Variations

| Aspect | AF3 | Boltz-1/2 | Chai-1 |
|---|---|---|---|
| Default steps | 200 (48 practical) | 200 | 200 |
| Sampler | 1st-order Euler | 1st-order Euler | **2nd-order Heun** |
| Stochasticity | $\gamma$ (variable) | $\gamma\_0 = 0.8$ | $S\_{\text{churn}} = 80$ |
| Model evals/step | 1 | 1 | **2** (Heun) |
| Atom weighting | Protein: 1, DNA/RNA: 5, Ligand: 10 | Same | Similar |

---

## 3. SE(3) Diffusion (RFdiffusion, FrameDiff) — Design-Oriented

### 3.1 Diffusion on Frames, Not Coordinates

While EDM diffuses in Cartesian coordinate space $\mathbb{R}^{3N}$, SE(3) diffusion operates on the **rigid body manifold** — each residue's frame $(R_i, \vec{t}_i)$:

**Forward process**:
- **Translations** $\vec{t}_i$: standard Gaussian noise in $\mathbb{R}^3$
- **Rotations** $R_i$: Brownian motion on SO(3) — the Lie group of 3D rotations

The SO(3) component requires special treatment because rotations don't live in Euclidean space. The noise distribution on SO(3) is the **isotropic Gaussian on SO(3)** (IGSO(3)), parameterized by a concentration parameter that decreases during the forward process.

### 3.2 FrameDiff: Theoretical Foundation

**FrameDiff** (Yim et al., ICML 2023) established the rigorous mathematical framework for SE(3) diffusion:

- Defined the proper forward process on $\text{SE}(3)^N$ (product manifold of $N$ frames)
- Derived the score function $\nabla \log p_t$ on this manifold
- Showed that the reverse-time SDE on SE(3) is well-defined and tractable
- **FrameFlow**: the flow matching counterpart, replacing the SDE with an ODE on SE(3)

### 3.3 RFdiffusion: Practical Design Engine

**RFdiffusion** (Watson et al., Nature 2023) applied SE(3) diffusion to protein backbone design:

```
Noise (random frames) → 200 denoising steps → Backbone Cα frames
                                                     │
                                          ProteinMPNN (inverse folding)
                                                     │
                                              Designed sequence
```

RFdiffusion **generates backbone structure only** (Cα frames) — no sequence, no side chains, no ligands. The sequence is designed separately by ProteinMPNN, which predicts amino acid identities that are compatible with the generated backbone geometry.

**Experimental validation**: Hundreds of RFdiffusion-designed proteins have been experimentally validated, making it the most battle-tested generative model for protein design. Applications include de novo binder design, symmetric oligomers, and enzyme active site scaffolding.

**Limitation**: The two-step pipeline (backbone generation → inverse folding) means that structure and sequence are not jointly optimized — a gap that later models (BoltzGen, La-Proteina) address directly.

---

## 4. Flow Matching (NP3, Proteina, AlphaFlow, FrameFlow) — The Straight Path

### 4.1 From SDE to ODE

The fundamental difference between diffusion and flow matching lies in the transport path:

```
Diffusion (SDE):                    Flow Matching (ODE):

  x₁ (noise)                         x₁ (noise/prior)
   \                                    \
    \ curved, stochastic                 \ straight, deterministic
     \ path (Brownian)                    \ path (linear interpolation)
      \                                    \
       x₀ (structure)                      x₀ (structure)

  dX = f(X,t)dt + g(t)dW             dX/dt = v_θ(X, t)
  Score: ∇ log p_t                    Velocity: v_t = x₀ - x₁
  Steps: ~200                         Steps: ~40
```

**Flow matching** defines a probability path $p_t$ that linearly interpolates between a prior distribution ($t = 0$) and the data distribution ($t = 1$):

$$x_t = (1 - t) \cdot x_1 + t \cdot x_0$$

The **velocity field** is simply:

$$v_t = \frac{dx_t}{dt} = x_0 - x_1$$

The neural network learns to predict this velocity:

$$\mathcal{L}(\theta) = \mathbb{E}_{t \sim U[0,1]}\, \mathbb{E}_{x \sim p_t} \|v_\theta(x_t, t) - v_t\|^2$$

**Inference** solves the ODE from the prior to the data distribution:

$$x_0 = x_1 + \int_0^1 v_\theta(x_t, t)\, dt \qquad (\text{numerically, ~40 steps})$$

### 4.2 Why Fewer Steps Suffice

The key advantage emerges from **path geometry**. Diffusion's SDE generates curved, stochastic trajectories — the particle wanders before reaching its destination. Flow matching's ODE generates nearly straight paths — the trajectory curvature is approximately 1.02 (versus ~3.45 for diffusion).

Straighter paths require fewer discretization steps to integrate accurately. NP3 achieves AF3-level accuracy with **40 ODE steps versus 200 diffusion steps** — a 5× reduction in model evaluations.

Additionally, flow matching training is **simulation-free**: the loss at each timestep $t$ can be computed independently, without rolling out the entire trajectory. This simplifies training and avoids the numerical instabilities that can arise from long SDE rollouts.

### 4.3 NP3: Polymer Prior + Encoder-Decoder

NP3 (Iambic, 2025) makes two additional innovations beyond adopting flow matching:

**Polymer Prior**: Instead of sampling initial coordinates from an isotropic Gaussian, NP3 generates a physics-informed prior through 64 Langevin dynamics steps with four force terms:

1. **Bond distance**: Maintains covalent bond lengths near physical values (~1.47 Å)
2. **Entity clustering**: Groups atoms belonging to the same molecular entity
3. **Residue clustering**: Keeps amino acid atoms compact
4. **Sphere confinement**: Prevents divergence to infinity

```
for step in 1..64:
    drift = 2·F_bond + F_entity/r_ent² + F_residue/r_res² - x/r_sphere²
    x ← x + dt·drift + √(2dt)·ε     (Langevin update)
```

The result: a prior that already encodes chemical connectivity and molecular topology. The ODE only needs to "refine" from this structured starting point, not build structure from scratch — explaining why 40 steps suffice.

**Encoder-Decoder separation**: Unlike AF3/Boltz (which run Trunk once, then Diffusion $N$ times referencing frozen $z$), NP3 uses a dedicated Encoder (~350M params) that produces representations consumed by a separate Decoder (~350M params) running flow matching. This eliminates the need for recycling — NP3 runs the encoder once and the decoder with 40 ODE steps, without iterating the encoder.

### 4.4 Proteina: Scaling Laws for Flow Matching

**Proteina** (NVIDIA, ICLR 2025 Oral) demonstrated that flow matching obeys **scaling laws** for protein backbone generation: model performance improves predictably with model size and training data volume.

This is significant because it suggests that flow matching models can benefit from the same "scale up model + data" recipe that has driven progress in language models — unlike SE(3) diffusion models (RFdiffusion), where scaling behavior was less characterized.

Proteina's scaling results directly led to its successors: **La-Proteina** (all-atom latent flow matching) and **Complexa** (binder design), which we discuss in Part 5.

---

## 5. Langevin Dynamics (BioEmu) — Thermodynamic Ensembles

### 5.1 A Different Goal

All models discussed so far aim to predict **the** structure (or a few structures) for a given sequence. BioEmu (Microsoft Research, 2024) pursues a fundamentally different objective: sampling from the **Boltzmann distribution**:

$$p(x \mid \text{seq}) = \frac{1}{Z} \exp\!\left(-\frac{E(x)}{k_B T}\right)$$

This means generating a thermodynamic ensemble — the full distribution of conformations a protein explores at equilibrium, weighted by their free energies.

### 5.2 Score Matching + Langevin MCMC

**Training**: BioEmu learns the score function $\nabla_x \log p_t(x)$ from molecular dynamics (MD) simulation trajectories using denoising score matching.

**Sampling** via Langevin dynamics:

$$x_{k+1} = x_k + \eta \cdot \nabla_x \log p(x_k) + \sqrt{2\eta} \cdot \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I)$$

This Markov chain provably converges to the Boltzmann distribution. Each sample is an independent conformation drawn from thermodynamic equilibrium — providing **true conformational diversity**, not just random seed variation.

### 5.3 Architecture and Limitations

BioEmu uses an SE(3)-equivariant denoiser based on IPA (inherited from AF2), operating on atomic coordinates. **PPFT** (Property Prediction Fine-Tuning) extends BioEmu by fine-tuning the model on ensemble observables ($\Delta G$, NMR chemical shifts) — learning thermodynamic properties without explicit structural supervision.

**Current limitation**: BioEmu supports only protein monomers. Extending Langevin-based ensemble sampling to complexes with ligands, nucleic acids, and cofactors remains open.

---

## 6. The Current Trend: Convergence Toward Flow Matching

### 6.1 Timeline

```
2021  AF2: IPA (deterministic, 8 steps)
      RF1: SE(3)-equivariant updates (integrated in 3-track)

2023  RFdiffusion: SE(3) diffusion (200 steps, design)
      FrameDiff: SE(3) diffusion theory (ICML)

2024  AF3/Boltz-1/Chai-1: EDM diffusion (200 steps, prediction)
      BioEmu: Langevin dynamics (ensemble)
      AlphaFlow: flow matching fine-tuning of AF2
      FrameFlow: SE(3) flow matching

2025  NP3: flow matching + polymer prior (40 steps)
      Proteina: flow matching + scaling laws
      La-Proteina: latent flow matching (all-atom)
      Complexa: flow matching (binder design)
      Boltz-2: EDM diffusion (retains 200 steps)
      Vilya-1: unified diffusion transformer
```

The trend is clear: **new models overwhelmingly adopt flow matching**. The practical advantages — 5× fewer steps, simulation-free training, nearly linear trajectories — make it the preferred framework for new development.

However, EDM diffusion is not dead. Boltz-2 (2025) retains it, and the established AF3 ecosystem ensures its continued use. BioEmu's Langevin approach occupies a distinct niche for ensemble sampling that neither diffusion nor flow matching addresses.

### 6.2 Comparison Table

| | IPA | EDM Diffusion | SE(3) Diffusion | Flow Matching | Langevin |
|---|---|---|---|---|---|
| **Representative** | AF2 | AF3, Boltz, Chai | RFdiffusion, FrameDiff | NP3, Proteina, FrameFlow | BioEmu |
| **Nature** | Deterministic | Stochastic (SDE) | Stochastic (SDE) | ODE-based | Stochastic (MCMC) |
| **Steps** | 8 | 200 | 200 | 40 | ~500 |
| **Operates on** | Residue frames | Atom coordinates | Residue frames | Atom coordinates | Atom coordinates |
| **Primary use** | Prediction | Prediction + generation | Backbone design | Prediction + generation | Ensemble sampling |
| **Conformers** | Single | Seed variation | Seed variation | ODE (+ noise injection) | True ensemble |
| **Speed** | Very fast | Moderate | Moderate | Fast | Slow |
| **All-atom capable** | Limited | Yes | Backbone only | Yes | Yes (monomer) |

---

## 7. Convergence and Open Questions

### Where the field agrees

- **Generative approaches are superior to deterministic prediction** for any task requiring structural diversity, all-atom modeling, or design applications
- **Flow matching is the emerging standard** for new model development, with clear advantages in inference speed and training simplicity
- **The Trunk → Structure Generation two-stage pattern persists** across all paradigms, with the Trunk producing conditioning signals that the generative module consumes

### What remains unresolved

**Unified generation for all paradigms**: Vilya-1 (Part 6) demonstrated that merging the Trunk and Structure Generation into a single unified transformer — where $z_{ij}$ and coordinates update jointly at every step — can improve conformational diversity. But this approach has only been demonstrated for macrocycles, not general protein complexes.

**Optimal prior design**: NP3's polymer prior improved efficiency dramatically. Could task-specific priors (e.g., antibody-specific, enzyme-specific) further reduce the required steps? How far can the prior-to-target gap be compressed?

**Ensemble sampling at scale**: BioEmu provides true Boltzmann ensembles but only for monomers. Extending thermodynamic ensemble sampling to large complexes with ligands — where conformational diversity is arguably most important for drug design — remains a fundamental challenge.

---

**Next: Part 4 — Proteins Only, or All Biomolecules? The Rise of Co-Folding and Open-Source Competition**

We examine how the field expanded from protein-only models to universal co-folding systems that handle proteins, nucleic acids, ligands, and ions in a single framework — and the open-source race to reproduce AlphaFold3.

---

*Part of the series: The Technical Evolution of Protein AI — A Record of Key Design Decisions*
