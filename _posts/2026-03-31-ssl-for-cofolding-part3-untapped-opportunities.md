---
title: "SSL for Co-Folding Part 3: Untapped Opportunities and the Confidence Calibration Trap"
date: 2026-03-31 09:20:00 +0900
categories: [Drug Discovery, Foundation Model]
tags: [semi-supervised-learning, co-folding, confidence-calibration, adaptive-thresholding, mean-teacher]
math: true
---

**What Protein AI Can Learn from Semi-Supervised Learning**

*This is Part 3 of a 4-part series examining how semi-supervised learning techniques from computer vision can improve protein structure prediction models.*

- **Part 1:** [The Semi-Supervised Revolution in Computer Vision](/posts/ssl-for-cofolding-part1-ssl-revolution-in-cv/)
- **Part 2:** [How Co-Folding Models Use Synthetic Data — An SSL Perspective](/posts/ssl-for-cofolding-part2-synthetic-data-in-cofolding/)
- **Part 3** (this post): Untapped Opportunities and the Confidence Calibration Trap
- **Part 4:** [The Road Ahead — Data Flywheels, Foundation Models, and Open Questions](/posts/ssl-for-cofolding-part4-road-ahead/)


---

## The Core Question

**Which proven SSL techniques from computer vision remain unadopted in protein AI, and what would their application look like?**

Part 2 mapped the distillation strategies of every major co-folding model onto the SSL framework we established in Part 1. The conclusion was striking: every model uses pseudo-labeling with fixed confidence thresholds — techniques that correspond to Era 1-2 of SSL development (2013-2017). Meanwhile, the most impactful advances from Era 3-4 (2019-2025) — adaptive thresholding, soft weighting, weak-to-strong augmentation asymmetry — remain entirely unadopted.

This Part is the key contribution of the series. We propose four concrete techniques that could be transferred from CV to protein AI, assess their implementation difficulty and expected impact, and then turn to what we believe is the most important and under-discussed problem in the field: the confidence calibration trap. AF3's decision to train confidence heads on PDB data only was not an arbitrary design choice — it was a deliberate defense against a subtle failure mode that most other models leave unaddressed.

Why does this gap exist? Part of the answer is sociological: the protein AI community and the SSL community operate in largely separate circles, publishing in different venues and citing different literatures. A researcher building a co-folding model is immersed in Evoformer attention patterns and diffusion schedules, not in the CIFAR-10 semi-supervised benchmarks where FlexMatch and SoftMatch were developed. Another part is structural: protein structure prediction involves continuous 3D coordinates rather than discrete class labels, and it is not always obvious how classification-oriented SSL techniques should be adapted. But as we will show, the adaptations are often straightforward — sometimes requiring as little as a single line of code.

---

## 1. The Gap at a Glance

Before we dive into individual proposals, let us summarize where the field stands relative to the SSL toolkit available in CV:

| SSL Technique | CV Era | Adopted in Protein AI? | Current State |
|---|---|---|---|
| Pseudo-labeling | Era 1 (2013) | Yes | All co-folding models |
| Fixed confidence threshold | Era 3 (2020) | Yes | pLDDT/ipTM hard cutoffs |
| Teacher-student distillation | Era 2 (2017) | Yes | AF2 to AF3, Boltz-1 to Boltz-2 |
| Multi-source data mixing | Era 3 (2019) | Yes | AF3 multi-teacher, Boltz-2 |
| **Weak-to-strong augmentation** | **Era 3 (2020)** | **No** | Teacher/student use same augmentation |
| **Adaptive thresholding** | **Era 4 (2021)** | **No** | All models use fixed thresholds |
| **Soft weighting** | **Era 4 (2023)** | **No** | Binary include/exclude only |
| **Online Mean Teacher** | **Era 2 (2017)** | **No** | Offline distillation only |
| **Confidence-weighted loss** | **Era 4 (2023)** | **No** | Equal loss for all included samples |

Everything below the bold line represents opportunities. We address them one by one.

---

## 2. Opportunity 1: Weak-to-Strong Augmentation

### The CV Principle

FixMatch (Sohn et al., 2020, NeurIPS) introduced a deceptively simple idea: generate pseudo-labels from *weakly augmented* (clean) inputs, but train the student on *strongly augmented* (noisy) inputs. The intuition is that the teacher's prediction is most reliable when the input is minimally perturbed, while the student gains robustness by learning to match that clean prediction under challenging conditions.

This single principle allowed FixMatch to outperform the far more complex MixMatch, which combined pseudo-labeling, consistency regularization, and entropy minimization into one framework. Sometimes less is more — provided the "less" targets the right asymmetry.

### Current State in Protein AI

In every co-folding model we surveyed, the teacher and student operate under the same augmentation regime. AF2's teacher predicts from full MSAs; the student trains on the same full MSAs (with identical subsampling). There is no deliberate separation between the conditions under which pseudo-labels are generated and the conditions under which the student learns.

### Protein AI Augmentation Axes

Protein structure prediction has a rich set of input perturbation axes that map naturally onto the weak/strong dichotomy:

| Augmentation Axis | Weak (for pseudo-label) | Strong (for training) |
|---|---|---|
| MSA depth | Full MSA (all sequences) | Subsampled (50% or fewer) |
| Sequence coverage | Full-length sequence | Random crop (256-384 residues) |
| Template | Template provided | Template dropout |
| Coordinate noise | No noise injection | Gaussian noise ($\sigma = 0.5$-$1.0$ A) |
| Recycling iterations | Full recycling (3 rounds) | Reduced recycling (1 round) |

Each axis controls a different aspect of input quality. MSA depth determines the richness of evolutionary signal. Sequence cropping removes global context. Template dropout eliminates explicit structural priors. Coordinate noise perturbs the diffusion denoiser's starting point. Recycling iterations control iterative refinement depth.

### Concrete Proposal

The weak-to-strong protocol for co-folding would look like this:

```
Pseudo-label generation (weak path):
  Input:  Full MSA, full-length sequence, templates included, no noise
  Model:  Teacher with full recycling (3 rounds)
  Output: High-quality structure prediction → pseudo-label x̂₀

Student training (strong path):
  Input:  MSA subsampled to 50%, random crop (384 tokens), template dropout
  Model:  Student with reduced recycling (1 round)
  Target: Match the teacher's clean prediction x̂₀
```

The consistency loss under this weak-to-strong regime takes the form:

$$\mathcal{L}_{\text{ws}} = \text{FAPE}\left(D_\theta(\mathbf{x}_t, t;\; z_{\text{strong}}),\; \hat{\mathbf{x}}_0^{\text{weak}}\right)$$

where $\hat{\mathbf{x}}_0^{\text{weak}}$ is the teacher's denoised prediction under weak augmentation, $z_{\text{strong}}$ is the pair representation derived from the strongly augmented input, $D_\theta$ is the student's denoiser, and $\mathbf{x}_t$ denotes the noised coordinates at diffusion timestep $t$.

### Expected Impact

The student would learn to produce accurate structures even when MSA signal is degraded, templates are unavailable, and context is limited — precisely the conditions that arise for orphan proteins, recently diverged sequences, and novel targets. This is implicit regularization: by training against a harder task, the model develops more robust internal representations.

Why should we expect this to work? Because it exploits a natural information hierarchy in protein inputs. The teacher's prediction from full MSA and full sequence is strictly more informed than the student's input under strong augmentation. This information gap creates a non-trivial learning signal: the student must learn to extract from a sparse MSA what the teacher could read directly from a rich one. In CV, the analogous gap between a flip+crop (weak) and RandAugment+Cutout (strong) was enough to drive state-of-the-art results. The augmentation gap in protein AI — full MSA vs. 50% subsampled MSA — is arguably even more meaningful, because MSA depth is directly related to the quality of coevolutionary signal available for contact prediction.

| Aspect | CV (FixMatch) | Protein AI (Proposed) |
|---|---|---|
| Weak augmentation | Flip + crop | Full MSA, full sequence, templates |
| Strong augmentation | RandAugment + Cutout | MSA subsampling + crop + template dropout |
| Information gap | Moderate (spatial) | Large (evolutionary signal) |
| Pseudo-label quality | Depends on model | High for well-studied proteins |
| Training signal | Classification consistency | Structural consistency (FAPE) |

### Implementation Difficulty: LOW

MSA subsampling and cropping are already part of standard training pipelines. The only change is *when* they are applied — during student training but not during teacher inference. No architectural modifications are required.

---

## 3. Opportunity 2: Adaptive and Soft Thresholding

### The Current Problem

Every co-folding model uses a fixed confidence threshold to filter pseudo-labeled data. SeedFold requires pLDDT $\geq$ 0.8. Boltz-2 uses lDDT $\geq$ 0.5 for monomers and ipTM $\geq$ 0.85 for complexes. These thresholds are set once and never change during training.

The consequence is severe data waste. When AF3 removed the pLDDT $>$ 0.8 filter that AF2 applied, they implicitly acknowledged the problem: strict thresholds discard too much useful data. For quaternary structure predictions in the AlphaFold Database, approximately 93% of entries would fail a pLDDT $\geq$ 0.8 cutoff. This means the vast majority of complex-structure pseudo-labels — the ones we need most — are thrown away.

Fixed thresholds also create a bias toward easy structure types. Alpha-helical domains tend to have high pLDDT; disordered regions and novel folds have low pLDDT. A single threshold enriches the training set with structures the model already handles well, while excluding precisely the difficult cases it needs to learn from.

### FlexMatch: Structure-Type-Adaptive Thresholds

FlexMatch (Zhang et al., 2021, NeurIPS) addressed an analogous problem in CV classification: fixed thresholds cause easy classes to dominate pseudo-label selection while hard classes are systematically excluded. The solution was to maintain a per-class threshold that adapts based on learning progress.

Translated to protein structure prediction, we define structure type $c$ (e.g., $\alpha$-helix domain, $\beta$-sheet domain, loop region, protein-protein interface, protein-ligand interface) and track the model's learning progress $\sigma_c(t)$ for each type. The adaptive threshold becomes:

$$\tau_c(t) = \frac{\sigma_c(t)}{\max_{c'} \sigma_{c'}(t)} \cdot \tau$$

where $\tau$ is a global base threshold and $\sigma_c(t)$ measures how well the model currently handles type $c$ — for instance, the fraction of type-$c$ predictions that exceed a validation accuracy criterion. When the model struggles with protein-protein interfaces ($\sigma_{\text{interface}}$ is low), the threshold for including interface pseudo-labels drops, allowing more of them into training.

### SoftMatch: Continuous Weighting

SoftMatch (Chen et al., 2023, ICLR) went further: instead of any threshold at all, it assigns a continuous weight to each pseudo-label based on a truncated Gaussian centered on the model's current confidence distribution.

Applied to protein AI, this means replacing the binary include/exclude decision with a smooth weight:

$$w_i = \exp\left(-\frac{(\text{pLDDT}_i - \mu_t)^2}{2\sigma_t^2}\right) \cdot \mathbb{1}[\text{pLDDT}_i \geq \tau_{\min}]$$

where $\mu_t$ and $\sigma_t$ are exponential moving averages of the model's confidence distribution during training, and $\tau_{\min}$ is a minimal quality floor (e.g., 0.4) to exclude clearly unreliable predictions. The key insight: a structure with pLDDT 0.77 — currently discarded under a 0.8 threshold — would receive a partial weight (say, $w \approx 0.5$) rather than being thrown away entirely.

### Comparison of Threshold Strategies

| Strategy | Method | Filtering Rate | Data Utilization | Quality Control | Implementation |
|---|---|---|---|---|---|
| Fixed hard | FixMatch-style | ~93% discarded (complexes) | Low | Binary (in or out) | Trivial |
| Adaptive per-type | FlexMatch-style | Structure-type dependent | Medium | Per-type threshold | Moderate |
| Soft continuous | SoftMatch-style | ~0% fully discarded | High | Continuous weight | Low-Moderate |

The progression from fixed to adaptive to soft thresholding represents a clear improvement trajectory. Each step increases data utilization while maintaining (or improving) quality control through more nuanced mechanisms.

```
Threshold Strategy Evolution:

  Era 3 (2020)              Era 4 (2021)              Era 4 (2023)
  ┌───────────────┐         ┌───────────────┐         ┌───────────────┐
  │  FixMatch      │         │  FlexMatch     │         │  SoftMatch     │
  │  Fixed τ=0.95  │────────→│  τ_c adaptive  │────────→│  Continuous w  │
  │  Binary: 0/1   │         │  per class     │         │  Gaussian wt   │
  │                │         │  Binary: 0/1   │         │  Soft: [0,1]   │
  └───────────────┘         └───────────────┘         └───────────────┘
        │                         │                         │
        ▼                         ▼                         ▼
   Co-folding:               Co-folding:               Co-folding:
   pLDDT ≥ 0.8              τ per structure type       w = f(pLDDT)
   (current state)           (proposed)                 (proposed)
```

### A Practical Hybrid

In practice, the most pragmatic approach may be a hybrid: soft weighting within each structure type, with type-specific parameters. The combined loss for pseudo-labeled data becomes:

$$\mathcal{L}_{\text{soft}} = \sum_{i \in \hat{\mathcal{D}}_U} w_c(\text{pLDDT}_i) \cdot \mathcal{L}_i$$

where $w_c$ is the SoftMatch weight function with parameters $(\mu_{c,t}, \sigma_{c,t})$ specific to structure type $c$. This captures both the FlexMatch insight (different types need different treatment) and the SoftMatch insight (continuous weights beat binary decisions).

### Implementation Difficulty: LOW to MODERATE

SoftMatch-style weighting requires only multiplying the per-sample loss by a scalar — a one-line change in the training loop. FlexMatch-style adaptation additionally requires tracking per-type learning progress, which needs a structure-type classifier and running statistics. Neither requires architectural changes.

---

## 4. Opportunity 3: Online Mean Teacher

### The Current Problem

Every co-folding model uses offline distillation. The teacher model is frozen, generates pseudo-labels once, and these static labels are used throughout student training. This means pseudo-labels cannot improve even as the student surpasses the teacher on some structure types. It also means that any systematic errors in the teacher's predictions are permanently baked into the training data.

### The Mean Teacher Principle

Mean Teacher (Tarvainen and Valpola, 2017, NIPS) proposed maintaining an exponential moving average (EMA) of the student's own weights as the teacher. As the student improves, the teacher automatically improves too — creating a positive feedback loop where better predictions generate better pseudo-labels, which drive further improvement.

For co-folding, the online EMA teacher update rule is:

$$\theta'_t = \alpha \theta'_{t-1} + (1-\alpha)\theta_t, \quad \alpha \in [0.999, 0.9999]$$

The consistency loss between the student and its EMA teacher becomes:

$$\mathcal{L}_{\text{MT}} = \text{FAPE}\left(D_{\theta}(\mathbf{x}_t, t;\; z_{\text{strong}}),\; D_{\theta'}(\mathbf{x}_t, t;\; z_{\text{weak}})\right)$$

where $D_\theta$ is the student denoiser operating on strongly augmented input and $D_{\theta'}$ is the EMA teacher denoiser operating on weakly augmented input. Note that this naturally combines with the weak-to-strong augmentation from Opportunity 1.

### Offline Distillation vs Online Mean Teacher

```
Offline Distillation (current):

  ┌──────────┐                    ┌──────────┐
  │  Teacher  │──── predict ────→ │  Fixed    │──── train ────→ ┌──────────┐
  │  (frozen) │                   │  Pseudo-  │                 │  Student  │
  │           │                   │  Labels   │                 │           │
  └──────────┘                    └──────────┘                  └──────────┘
       ✗ no update                     ✗ static


Online Mean Teacher (proposed):

  ┌──────────┐                                       ┌──────────┐
  │  Teacher  │◄──── EMA update ──────────────────────│  Student  │
  │  (EMA)    │                                       │          │
  │  θ'       │──── predict (weak aug) ──→ target     │  θ       │
  └──────────┘                              │         └──────────┘
                                            ▼              ▲
                                   FAPE(target, pred) ─────┘
                                                     train on strong aug
       ✓ improves with student         ✓ dynamic labels
```

### Expected Impact

In CV, Mean Teacher solved the scalability limitation of Temporal Ensembling (Laine and Aila, 2017, ICLR), which required storing predictions for every unlabeled sample. More importantly, the EMA teacher provides a smoother, more stable learning signal than the student's own rapidly-changing predictions. For co-folding, the benefit would be reduced error propagation: as the student learns to correctly fold a difficult protein family, the EMA teacher's pseudo-labels for that family automatically improve, creating a virtuous cycle.

The Mean Teacher also enables an important capability that offline distillation cannot provide: the teacher can generate pseudo-labels for *new* sequences during training, not just those pre-computed before training began. This opens the door to curriculum-based sampling strategies where the training set evolves over time.

### Practical Considerations

| Aspect | Offline Distillation | Online Mean Teacher |
|---|---|---|
| Memory | 1x model | 2x model (student + EMA copy) |
| Compute per step | Forward pass (student only) | 2x forward pass (student + teacher) |
| Pseudo-label quality | Fixed at teacher's level | Improves during training |
| Storage | Pre-computed labels on disk | Generated on-the-fly |
| Error propagation | Static errors persist | Errors self-correct over time |
| Implementation | Simple (pre-process once) | Complex (online inference pipeline) |

### Implementation Difficulty: HIGH

Maintaining an EMA copy of a co-folding model requires approximately 2x GPU memory — a significant cost when models already consume 40-80GB per GPU. Online pseudo-label generation adds a forward pass per training step (though the teacher forward pass can be done without gradients, saving memory on activations). The infrastructure for online inference during training is substantially more complex than pre-computing labels offline.

That said, the EMA mechanism itself is trivial to implement — it is a single line of code executed after each optimizer step. The difficulty is purely one of engineering and compute budget, not algorithmic complexity.

---

## 5. Opportunity 4: Confidence-Weighted Loss

### The Current State

Among the models that do not separate confidence loss by data source (i.e., everyone except AF3 and OpenFold3), all included pseudo-labeled samples receive equal weight in the loss function. A structure with pLDDT 0.95 contributes the same gradient magnitude as one with pLDDT 0.81 (assuming the threshold is 0.80). This ignores a clear signal: the teacher's own confidence in its prediction.

### The Proposal

Use pLDDT directly as a continuous loss weight for pseudo-labeled samples:

$$\mathcal{L}_{\text{cw}} = \sum_{i \in \mathcal{D}_L} \mathcal{L}_i + \lambda \sum_{i \in \hat{\mathcal{D}}_U} w(\text{pLDDT}_i) \cdot \mathcal{L}_i$$

where $\mathcal{D}_L$ is the labeled PDB data (full weight), $\hat{\mathcal{D}}_U$ is the pseudo-labeled synthetic data, and $w(\text{pLDDT})$ is a monotonically increasing weight function. The labeled data always receives full weight; the synthetic data is downweighted according to the teacher's confidence.

The weight function $w$ could take several forms:

| pLDDT | Linear | Sigmoid | Gaussian ($\mu=0.9, \sigma=0.15$) |
|---|---|---|---|
| 0.95 | 0.95 | ~1.0 | ~0.95 |
| 0.85 | 0.85 | ~0.9 | ~0.95 |
| 0.75 | 0.75 | ~0.5 | ~0.60 |
| 0.65 | 0.65 | ~0.2 | ~0.15 |
| 0.50 | 0.50 | ~0.02 | ~0.00 |

The choice matters less than the principle: high-confidence pseudo-labels should dominate the gradient, while low-confidence ones contribute proportionally less. This is the most minimal possible intervention — multiply the loss by a scalar per sample.

### Implementation Difficulty: LOW

This requires a single multiplication in the loss computation. The pLDDT values are already available as metadata for every pseudo-labeled structure. No additional forward passes, no architectural changes, no new hyperparameters beyond the choice of weight function.

---

## 6. The Confidence Calibration Trap

This section addresses what we believe is the most important and under-discussed design decision in co-folding model training. It concerns not the structure prediction itself, but the *confidence estimate* — the model's self-assessment of how accurate its own prediction is.

### AF3's Design: Loss Separation by Data Source

AlphaFold3's total loss decomposes as:

$$\mathcal{L} = \underbrace{\alpha_{\text{diff}} \mathcal{L}_{\text{diffusion}} + \alpha_{\text{dist}} \mathcal{L}_{\text{distogram}}}_{\text{all data (PDB + synthetic)}} + \underbrace{\alpha_{\text{conf}} \left(\mathcal{L}_{\text{pLDDT}} + \mathcal{L}_{\text{PDE}} + \alpha_{\text{PAE}} \mathcal{L}_{\text{PAE}} + \mathcal{L}_{\text{resolved}}\right)}_{\text{PDB experimental data only}}$$

The structure losses (diffusion, distogram) are trained on all data — both experimental PDB structures and synthetic pseudo-labeled structures. But the confidence losses (pLDDT, PDE, PAE, experimentally resolved) are trained exclusively on PDB data, where the ground truth structure is known from experiment.

This is not a minor implementation detail. It is a deliberate architectural decision that protects the model against a cascade of failure modes. Most other models — Boltz-2, SeedFold, and to a lesser extent AF2 and Boltz-1 — do not make this separation. Let us examine why they should.

### Problem 1: Systematic Overconfidence

The most immediate danger. Teacher model errors are not random — they are *systematic*. AF2 makes characteristic mistakes on specific structure types:

- **Disordered regions** are predicted as compact structures, with moderate pLDDT (70-80)
- **Novel folds** with sparse MSAs are predicted as the nearest known fold, with moderate-to-high pLDDT (75-85)
- **Multimer interfaces** predicted from monomer models receive artificially high per-residue pLDDT

When a student model learns confidence from these pseudo-labels, it inherits the teacher's mis-assessment. The student sees a compact prediction for a disordered region with a teacher-assigned pLDDT of 78, and learns: "for this type of input, a pLDDT of ~78 is appropriate." But the actual accuracy of this prediction is far lower — the region is disordered and should not be modeled as a single conformation at all.

With PDB experimental data, the model can learn the corrective signal: "I predicted a compact structure, but the actual experimental structure shows this region is unresolved — my confidence should be very low." Synthetic data provides no such correction, because the pseudo-label and the confidence estimate come from the same teacher model.

### Problem 2: Calibration Distortion

A confidence score is only useful if it is *calibrated* — meaning that a pLDDT of 90 should correspond to approximately 90% of predictions at that confidence level being accurate. Formally, calibration requires:

$$P\left(\text{lDDT}_i \geq s \;\middle|\; \text{pLDDT}_i = s\right) \approx s, \quad \forall s \in [0, 1]$$

Well-calibrated confidence enables trustworthy decision-making downstream. A drug discovery team that sees pLDDT = 0.90 can allocate wet-lab resources accordingly; a protein engineer who sees pLDDT = 0.55 knows to seek alternative validation. This decision-making value collapses if the calibration curve is distorted.

The teacher model's own calibration is imperfect in specific, predictable ways:

- **Small proteins** ($<$ 100 residues): pLDDT tends to be overestimated because these are generally well-folded and highly represented in training
- **Metal binding sites**: pLDDT is high for the overall residue backbone but coordination geometry (bond angles, distances to metal) is often wrong
- **RNA-proximal protein regions**: AF2 cannot model RNA, so protein regions near RNA binding sites have distorted confidence
- **Protein-protein interfaces**: Per-residue pLDDT can be high even when the relative chain orientation (captured by ipTM/PAE) is incorrect

When the student trains confidence on millions of these systematically mis-calibrated pseudo-labels, the calibration anchor shifts. The relationship between predicted confidence and actual accuracy becomes unreliable. Even if the structure prediction itself is reasonable, the confidence estimate may be dangerously misleading.

### Problem 3: Confirmation Bias Amplification

Self-training has a well-known failure mode: confirmation bias. The teacher makes an error, the student learns it, and the error persists. For structure prediction, this bias is partially mitigated by mixing in PDB experimental data — the ground truth structures provide a corrective signal.

But confidence errors are more insidious:

```
Confidence error propagation cycle:

  Teacher makes wrong prediction ─────────────────────────────────┐
       + assigns moderate-high confidence                         │
                    │                                             │
                    ▼                                             │
  Student learns: "this pattern → confidence ~0.78"               │
                    │                                             │
                    ▼                                             │
  Student reproduces similar error with similar confidence        │
                    │                                             │
                    ▼                                             │
  If student becomes next teacher (data flywheel):                │
  Error + false confidence reinforced across generations ─────────┘
```

Structure errors can be detected — the model's prediction can be compared against new experimental structures, and the discrepancy is measurable. But confidence errors are invisible unless you explicitly evaluate calibration. A model can achieve excellent structure prediction metrics (low RMSD, high GDT) while having badly miscalibrated confidence — and no one notices until the confidence score leads to a costly wrong decision in drug discovery or protein design.

### Problem 4: Noise Tolerance Asymmetry

This is the key insight that unifies the previous three problems. Structure loss and confidence loss have fundamentally different tolerances for noise in pseudo-labels:

```
Structure loss (diffusion loss):
  ┌────────────────────────────────────────────────────────────┐
  │  Target: 3D coordinates                                    │
  │  Teacher accuracy: Most AF2 predictions are fairly         │
  │    accurate (median GDT > 70 on globular domains)          │
  │  Noise averaging: Millions of diverse examples mean        │
  │    random errors average out across training                │
  │  Impact of errors: Slightly wrong coordinates still        │
  │    teach correct fold topology                              │
  │                                                            │
  │  Conclusion: Synthetic data is USEFUL                      │
  │    Slight noise is tolerable; coverage gains dominate      │
  └────────────────────────────────────────────────────────────┘

Confidence loss:
  ┌────────────────────────────────────────────────────────────┐
  │  Target: "How accurate is this prediction?"                │
  │    — a meta-judgment about prediction quality               │
  │  Teacher accuracy: Systematic errors in confidence         │
  │    are NOT random and do NOT average out                    │
  │  Noise averaging: A few consistently mis-calibrated        │
  │    regions undermine overall trustworthiness                │
  │  Impact of errors: Wrong confidence directly causes        │
  │    wrong downstream decisions                               │
  │                                                            │
  │  Conclusion: Synthetic data can be HARMFUL                 │
  │    Systematic mis-calibration is not diluted by scale      │
  └────────────────────────────────────────────────────────────┘
```

We can state this asymmetry concisely: **structure predictions tolerate approximate correctness; confidence estimates demand exact correctness.** A 3D coordinate that is 2 angstroms off still teaches the model something useful about protein geometry. But a confidence score that is 20 percentage points off actively misleads the model about when to trust itself.

This noise tolerance asymmetry is the fundamental reason why AF3's loss separation is sound engineering. Structure loss benefits from the massive scale of synthetic data (41M+ pseudo-labels). Confidence loss would be corrupted by the systematic biases embedded in that same data. Separating the two respects the different noise characteristics of each learning signal.

### Model-by-Model Confidence Estimator Assessment

Given this analysis, we can assess the reliability of each model's confidence estimates:

| Model | Confidence Loss Data | Protection Mechanism | Risk Level |
|---|---|---|---|
| **AF3** | PDB only | Loss separation by design | Low |
| **OpenFold3** | PDB only (AF3 protocol) | Same as AF3 | Low |
| **AF2** | Distillation + PDB | Low-confidence residues masked ($c_i < 0.5$) | Medium |
| **Boltz-1** | Distillation + PDB | Last 15K steps PDB only (partial recalibration) | Medium |
| **Boltz-2** | All data (lDDT $\geq$ 0.5) | None — sampling ratio only | High |
| **SeedFold** | All data (pLDDT $\geq$ 0.8) | None — threshold only | High |

AF3 and OpenFold3 are protected by design. AF2 has partial protection through residue-level masking — residues with KL-divergence $c_i < 0.5$ are excluded from loss, which removes the most obviously uncertain regions. However, residues with $c_i > 0.5$ that are still systematically wrong remain included. Boltz-1's switch to PDB-only data in the final 15K training steps may partially recalibrate confidence, though it is unclear whether this was an intentional design for calibration or simply a training schedule choice.

Boltz-2 and SeedFold are the most vulnerable. Boltz-2 trains confidence on all data with only a very permissive lDDT $\geq$ 0.5 threshold, meaning substantially inaccurate pseudo-labels contribute to confidence learning. SeedFold trains on 26.5M synthetic structures — 147 times the size of PDB — with no confidence loss separation, meaning the vast majority of the confidence learning signal comes from pseudo-labels rather than ground truth.

### Practical Recommendation

When using confidence scores from co-folding models in downstream applications — virtual screening, binder design, lead optimization — **trust relative rankings rather than absolute values** for models other than AF3 and OpenFold3. A Boltz-2 prediction with pLDDT 0.85 may not mean the same thing as an AF3 prediction with pLDDT 0.85. The ranking ("structure A is predicted more reliably than structure B") is more robust to calibration distortion than the absolute score ("this prediction is 85% likely to be accurate").

For any team building a new co-folding model, the recommendation is unambiguous: **separate confidence loss from structure loss by data source.** Train confidence heads on PDB experimental data only. The implementation cost is minimal — it is a conditional statement in the loss function — and the calibration benefit is substantial.

### Why This Matters for Downstream Applications

The confidence calibration trap is not an abstract concern. It has direct consequences for every application that uses co-folding model confidence scores as input to decision-making:

| Application | How Confidence Is Used | Cost of Mis-calibration |
|---|---|---|
| Virtual screening | Filter/rank docking poses by pLDDT | Wrong compounds advance to synthesis ($10K-100K each) |
| Binder design | Assess designed interface quality | Failed binders waste expression + assay cycles |
| Variant effect prediction | Confidence as proxy for structural impact | Misclassified pathogenic variants |
| Cryo-EM model building | pLDDT guides map-to-model fitting | Incorrect backbone tracing in ambiguous density |
| Target assessment | Confidence indicates modelability | Resources allocated to un-modelable targets |

In each case, the distinction between "pLDDT 0.85 means 85% reliable" and "pLDDT 0.85 means somewhere between 60-90% reliable" is the difference between a calibrated tool and an unreliable one. The cost of overconfidence is wasted experimental resources; the cost of underconfidence is missed opportunities. Both are expensive, and both are avoidable if confidence is trained on the right data.

---

## 7. Closing: Prioritized Recommendations

We have proposed four concrete SSL techniques for co-folding and analyzed the confidence calibration problem in depth. How should a practitioner prioritize these opportunities? We rank them by the ratio of expected impact to implementation difficulty:

| Rank | Technique | Difficulty | Expected Impact | Rationale |
|---|---|---|---|---|
| 1 | Soft weighting (SoftMatch) | Low | High | Multiply loss by pLDDT weight. Directly addresses 93% filtering waste |
| 2 | Weak-to-strong augmentation | Low | High | MSA depth separation. FixMatch's core insight, near-zero implementation cost |
| 3 | Confidence loss separation | Low | High | AF3's proven approach — restrict confidence loss to PDB. One conditional |
| 4 | Adaptive thresholding | Moderate | High | Structure-type thresholds address easy-example bias |
| 5 | Confidence-weighted loss | Low | Medium | Per-sample loss weighting by pLDDT. Minimal code change |
| 6 | Curriculum learning | Moderate | Medium | Train on easy structures first, hard structures later |
| 7 | Online Mean Teacher | High | High | 2x memory, online inference. Highest potential but highest cost |

The first three items share a common trait: they are low-cost interventions with clear theoretical motivation and practical precedent. Soft weighting and weak-to-strong augmentation are proven in CV with minimal adaptation needed. Confidence loss separation is already validated by AF3's design — it simply needs to be adopted by other models. Any team could implement all three in a single training run.

Items 4-6 require more infrastructure but remain within reach of well-resourced groups. The online Mean Teacher (item 7) is the most ambitious proposal — its memory and compute requirements are significant for models that already push hardware limits — but its potential for breaking the static pseudo-label ceiling makes it a compelling long-term investment.

Notice that the top three recommendations are all *low difficulty*. This is not a coincidence — it reflects the fact that co-folding models have not yet adopted even the simplest advances from modern SSL. The low-hanging fruit has not been picked. Before investing in complex online learning infrastructure, the field should first exhaust the easy wins that CV demonstrated years ago.

To put concrete numbers on the potential: if soft weighting recovers even 10% of the 93% of complex-structure pseudo-labels currently discarded, that represents hundreds of thousands of additional training examples for the structure types where data is scarcest. If weak-to-strong augmentation provides even half the relative improvement that FixMatch showed over naive pseudo-labeling in CV benchmarks, the gains would be measurable on CASP-style evaluations. And confidence loss separation is not speculative at all — it is proven engineering, already validated by AF3's results.

### What We Did Not Cover

Two important directions remain outside the scope of this Part. **Iterative self-training** — running multiple rounds of distillation where the student becomes the next teacher — is the natural extension of Noisy Student (Xie et al., 2020, CVPR) to protein AI. Currently, every co-folding model performs exactly one round of distillation. **Learned quality predictors** — training an independent model to assess pseudo-label quality, in the spirit of SemiReward (Wang et al., 2024, ICLR) — could replace pLDDT-based self-assessment with a more objective quality signal. Both of these are longer-term research directions that we address in Part 4.

---

**Next: Part 4 — The Road Ahead: Data Flywheels, Foundation Models, and Open Questions**

In the final Part, we look beyond individual techniques to the systemic questions: Can the data flywheel (better teacher $\to$ better data $\to$ better student $\to$ repeat) sustain indefinite improvement, or does model collapse set a ceiling? How do protein language models like ESM-2 change the role of SSL? And what fundamental challenges in confidence calibration and data quality remain unsolved?

---

*Part of the series: What Protein AI Can Learn from Semi-Supervised Learning*

---

## Notation Reference

| Symbol | Meaning |
|---|---|
| $\theta$ | Student model parameters |
| $\theta'$ | Teacher (EMA) model parameters |
| $\mathcal{D}_L$ | Labeled dataset (PDB experimental structures) |
| $\hat{\mathcal{D}}_U$ | Pseudo-labeled dataset (synthetic/distilled structures) |
| $\alpha(x)$ | Weak augmentation of input $x$ |
| $\mathcal{A}(x)$ | Strong augmentation of input $x$ |
| $\tau$ | Confidence threshold |
| $w(\cdot)$ | Sample weight function |
| pLDDT | Predicted Local Distance Difference Test (confidence metric) |
| FAPE | Frame Aligned Point Error (structure loss) |
| $\mathbf{x}_0$ | Clean 3D coordinates |
| $\mathbf{x}_t$ | Noised coordinates at diffusion timestep $t$ |
| $z$ | Pair representation |
| $D_\theta$ | Denoiser network parameterized by $\theta$ |

## References

- Sohn, K., Berthelot, D., et al. "FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence." NeurIPS 2020.
- Zhang, B., Wang, Y., et al. "FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling." NeurIPS 2021.
- Chen, H., Tao, R., et al. "SoftMatch: Addressing the Quantity-Quality Trade-off in Semi-supervised Learning." ICLR 2023.
- Wang, Y., Chen, H., et al. "FreeMatch: Self-adaptive Thresholding for Semi-supervised Learning." ICLR 2023.
- Tarvainen, A. and Valpola, H. "Mean Teachers are Better Role Models." NIPS 2017.
- Laine, S. and Aila, T. "Temporal Ensembling for Semi-Supervised Learning." ICLR 2017.
- Berthelot, D., Carlini, N., et al. "MixMatch: A Holistic Approach to Semi-Supervised Learning." NeurIPS 2019.
- Xie, Q., Luong, M.-T., et al. "Self-training with Noisy Student improves ImageNet classification." CVPR 2020.
- Wang, Y., et al. "SemiReward: A General Reward Model for Semi-supervised Learning." ICLR 2024.
- Jumper, J., et al. "Highly accurate protein structure prediction with AlphaFold." Nature 2021.
- Abramson, J., et al. "Accurate structure prediction of biomolecular interactions with AlphaFold 3." Nature 2024.
- Wohlwend, J., et al. "Boltz-1: Democratizing Biomolecular Interaction Modeling." bioRxiv 2024.
- Passaro, S., Wohlwend, J., et al. "Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction." bioRxiv 2025.
- SeedFold team. "SeedFold: Scaling Biomolecular Structure Prediction." arXiv 2025.
- OpenFold Consortium. "OpenFold3." Technical Report 2025.
