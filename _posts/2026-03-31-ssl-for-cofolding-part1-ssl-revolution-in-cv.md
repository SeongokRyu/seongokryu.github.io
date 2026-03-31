---
title: "SSL for Co-Folding Part 1: The Semi-Supervised Revolution in Computer Vision"
date: 2026-03-31 09:00:00 +0900
categories: [Drug Discovery, Foundation Model]
tags: [semi-supervised-learning, co-folding, pseudo-labeling, consistency-regularization, computer-vision]
math: true
---

**What Protein AI Can Learn from Semi-Supervised Learning**

*This is Part 1 of a 4-part series examining how semi-supervised learning techniques from computer vision can improve protein structure prediction models.*

- **Part 1** (this post): The Semi-Supervised Revolution in Computer Vision
- **Part 2:** [How Co-Folding Models Use Synthetic Data — An SSL Perspective](/posts/ssl-for-cofolding-part2-synthetic-data-in-cofolding/)
- **Part 3:** [Untapped Opportunities and the Confidence Calibration Trap](/posts/ssl-for-cofolding-part3-untapped-opportunities/)
- **Part 4:** [The Road Ahead — Data Flywheels, Foundation Models, and Open Questions](/posts/ssl-for-cofolding-part4-road-ahead/)


---

## The Core Question

**What are the key ideas behind semi-supervised learning, and how did they evolve in computer vision?**

Before we can argue that protein AI should adopt techniques from semi-supervised learning (SSL), we need to understand what those techniques actually are, where they came from, and why they work. This Part equips you with the conceptual vocabulary for the rest of the series. Even if you have never worked on image classification, the patterns here will look strikingly familiar once we map them to protein structure prediction in Part 2.

---

## 1. Introduction: The Labeled Data Bottleneck

Deep learning is hungry for labeled data. In computer vision, the landmark success of AlexNet (Krizhevsky et al., 2012, NeurIPS) was built on ImageNet's 1.28 million hand-labeled images — a dataset that took years and millions of dollars to construct. Yet even ImageNet covers only 1,000 categories. Scaling to 10,000 classes, or to fine-grained distinctions like bird species or skin lesion subtypes, requires annotation budgets that most research groups simply do not have.

The internet, meanwhile, overflows with unlabeled images. Flickr alone hosts billions of photos. The question that launched semi-supervised learning in the deep learning era was deceptively simple: **can we extract useful training signal from data that has no labels?**

The scale of the problem is worth quantifying. Even in the "data-rich" regime of ImageNet, supervised learning requires about 1,000 labeled examples per class. For medical imaging, getting 1,000 expert-annotated radiology scans per pathology type can cost over \$100,000. For satellite imagery, labeling requires domain expertise that simply does not exist at scale. The common thread: **labels are expensive, domain-specific, and bottleneck the entire pipeline.**

This is not merely a computer vision problem. In protein AI, the asymmetry is arguably more extreme:

| Data Type | Approximate Count | Annotation Cost |
|---|---|---|
| Protein sequences (UniProt) | 250,000,000+ | Essentially free (sequencing) |
| Experimental structures (PDB) | ~220,000 | Years of crystallography / cryo-EM per structure |
| High-quality complex structures | ~50,000 | Even more expensive |

The ratio of unlabeled to labeled data in protein AI exceeds **1,000:1** — far worse than the typical CV benchmark where we might have 250 labels out of 50,000 images (a 200:1 ratio on CIFAR-10). If SSL revolutionized computer vision with a 200:1 ratio, what could it do for protein AI at 1,000:1?

This series argues that every modern co-folding model — AlphaFold2, AlphaFold3, Boltz, SeedFold — is already performing semi-supervised learning. They just do not call it that, and they are stuck using techniques from 2013 while a decade of advances sits untapped.

SSL's premise is that we can bridge this gap: use the abundant unlabeled data (sequences) to improve models trained on scarce labeled data (experimental structures). The question is not whether to do this — every major co-folding model already does — but whether we are doing it well.

**Series roadmap:**
- **Part 1** (this post): SSL concepts and their CV evolution
- **Part 2**: How co-folding models use synthetic data — an SSL perspective
- **Part 3**: Untapped opportunities and the confidence calibration trap
- **Part 4**: Data flywheels, foundation models, and open questions

---

## 2. Three Pillars of Semi-Supervised Learning

SSL methods in deep learning rest on three core principles. Every major method from 2013 to 2025 can be understood as a combination of these three ideas.

### 2.1 Pseudo-Labeling

The simplest idea: **use the model's own predictions as labels for unlabeled data**. Given an unlabeled input $x$, the model produces a prediction, and we convert that prediction into a "pseudo-label" that we treat as ground truth for training.

Formally, the pseudo-label is the most confident class:

$$\hat{y} = \arg\max_c \; p_\theta(y = c \mid x)$$

where $p_\theta(y = c \mid x)$ is the model's predicted probability of class $c$ given input $x$, and $\theta$ denotes the model parameters. The training loss on unlabeled data then becomes:

$$\mathcal{L}_{\text{pl}} = \mathbb{1}[\max_c \; p_\theta(c \mid x) \geq \tau] \cdot H(\hat{y}, \; p_\theta(y \mid \alpha(x)))$$

Here $\tau$ is a confidence threshold (e.g., 0.95), $\alpha(x)$ denotes a (potentially augmented) version of $x$, $H(\cdot, \cdot)$ is the cross-entropy loss, and $\mathbb{1}[\cdot]$ is the indicator function. The threshold ensures we only train on pseudo-labels the model is confident about.

**Intuition**: If the model is already 95% sure an image is a cat, training on that pseudo-label reinforces a likely-correct decision and pushes the decision boundary into low-density regions. The risk, of course, is confirmation bias — wrong but confident predictions get amplified.

### 2.2 Consistency Regularization

A different angle: **the model's prediction should not change when we perturb the input in ways that do not change the true label**. Flipping an image horizontally, adding noise, or applying color jitter should not turn a cat into a dog.

$$\mathcal{L}_{\text{cons}} = \| p_\theta(y \mid \alpha_1(x)) - p_\theta(y \mid \alpha_2(x)) \|^2$$

where $\alpha_1(x)$ and $\alpha_2(x)$ are two different augmentations of the same input. No labels are needed — we are only asking the model to be self-consistent.

**Intuition**: This acts as a smoothness regularizer. The model learns that the function $p_\theta(y \mid x)$ should be locally flat — small perturbations in input space should not cross decision boundaries. This is closely related to the cluster assumption in SSL: data points in the same cluster should share labels, and clusters should be separated by low-density regions.

### 2.3 Entropy Minimization

The third principle: **encourage the model to make confident (low-entropy) predictions on unlabeled data**, regardless of which class it is confident about.

$$\mathcal{L}_{\text{ent}} = -\sum_c p_\theta(c \mid x) \log p_\theta(c \mid x)$$

This is simply the entropy of the predicted distribution. Minimizing it pushes predictions toward one-hot vectors — the model must "commit" to a class rather than hedging.

**Intuition**: Entropy minimization implements the low-density separation assumption. By forcing confident predictions everywhere, we push decision boundaries away from data-dense regions and into the gaps between clusters. It is implicit in pseudo-labeling (hard pseudo-labels are zero-entropy) but can also be applied as a soft, continuous objective.

### 2.4 How the Three Pillars Relate

These three principles are not independent — they overlap and reinforce each other:

```
    ┌──────────────────────────────────────────────────┐
    │              Semi-Supervised Learning             │
    │                                                   │
    │   ┌─────────────┐       ┌─────────────────────┐  │
    │   │  PSEUDO-     │       │  CONSISTENCY        │  │
    │   │  LABELING    │       │  REGULARIZATION     │  │
    │   │              │       │                     │  │
    │   │ "Trust the   ├───┐   │ "Perturbed inputs   │  │
    │   │  model's own │   │   │  should give same   │  │
    │   │  predictions"│   │   │  predictions"       │  │
    │   │              │   │   │                     │  │
    │   └──────┬───────┘   │   └──────────┬──────────┘  │
    │          │      ┌────┴─────┐        │             │
    │          │      │ COMBINED │        │             │
    │          └──────┤ METHODS  ├────────┘             │
    │                 │(FixMatch)│                      │
    │                 └────┬─────┘                      │
    │                      │                            │
    │             ┌────────┴────────┐                   │
    │             │    ENTROPY      │                   │
    │             │  MINIMIZATION   │                   │
    │             │                 │                   │
    │             │ "Be confident   │                   │
    │             │  about unlabeled│                   │
    │             │  data"          │                   │
    │             └─────────────────┘                   │
    └──────────────────────────────────────────────────┘
```

Pseudo-labeling implicitly performs entropy minimization (hard labels have zero entropy). Consistency regularization combined with pseudo-labeling gives you FixMatch — arguably the most important SSL method of the past decade. And entropy minimization underpins both: it is the shared assumption that decision boundaries belong in low-density regions.

| Principle | Core Equation | Intuition | Representative Method |
|---|---|---|---|
| **Pseudo-labeling** | $\hat{y} = \arg\max_c \; p_\theta(c \mid x)$ | Trust confident model predictions as ground truth | Pseudo-Label (Lee, 2013) |
| **Consistency regularization** | $\lVert p_\theta(\alpha_1(x)) - p_\theta(\alpha_2(x)) \rVert^2$ | Same input, different view, same prediction | Mean Teacher (Tarvainen & Valpola, 2017) |
| **Entropy minimization** | $-\sum_c p_\theta(c \mid x) \log p_\theta(c \mid x)$ | Force the model to commit to one class | Minimum Entropy (Grandvalet & Bengio, 2005) |

---

## 3. The Evolution: Four Eras

The history of SSL in deep learning is not a smooth gradient — it moved through distinct phases, each marked by a conceptual shift. We organize this as four eras, focusing on the key turning points rather than an exhaustive chronology. For readers primarily interested in the protein AI applications (Parts 2-4), this section provides the historical context needed to understand why certain SSL advances matter and which ones remain unexplored in our field.

### 3.1 Era 1 (2013--2016): Foundations

The deep learning revolution of 2012 created both the need and the opportunity for SSL. Neural networks were finally powerful enough to generate useful pseudo-labels, and deep representations made consistency regularization meaningful. But no one had a unified framework — the field explored multiple directions simultaneously.

**Pseudo-Label (Lee, 2013, ICML Workshop).** The foundational paper for deep SSL. The idea is almost trivially simple: train a neural network on labeled data, use it to predict labels for unlabeled data, and then train on both. Lee showed this works as a form of entropy regularization — the hard pseudo-labels push the model toward confident predictions, implicitly minimizing entropy. Despite its simplicity, this paper defined a paradigm that every subsequent method builds upon. The central risk — confirmation bias, where wrong predictions reinforce themselves — would take years to adequately address.

**Semi-Supervised VAE (Kingma et al., 2014, NIPS).** A generative approach: model the joint distribution $p(x, y)$ using a variational autoencoder, treating the label $y$ as a latent variable when it is missing. Elegant in theory, but the generative modeling overhead proved difficult to scale and was eventually overtaken by discriminative approaches.

**Ladder Networks (Rasmus et al., 2015, NIPS).** Layer-wise denoising applied to each layer of a deep network. The model learns to reconstruct clean activations from noisy ones at every level, combining supervised classification with unsupervised denoising. The headline result was remarkable: **1.06% error on MNIST with only 100 labels** — approaching fully supervised performance. This demonstrated the raw power of SSL but the architecture was complex and did not generalize easily beyond the specific experimental setup.

**Semi-Supervised GAN (Salimans et al., 2016, NIPS).** Instead of using a binary real/fake discriminator, extend it to $K+1$ classes: the $K$ real classes plus "fake." This forces the discriminator to learn class-discriminative features as a byproduct of distinguishing real from generated. Clever, but unstable training and mode collapse limited practical adoption.

**Summary of Era 1:** Four different angles on the same problem — self-training, generative modeling, denoising, adversarial training. Each demonstrated that SSL works, but there was no unified framework and no clear winner for practical use. The field had proven the concept; what it lacked was a coherent recipe.

| Method | Approach | Best Result | Key Limitation |
|---|---|---|---|
| **Pseudo-Label** (2013) | Self-training with hard labels | Improved over supervised baseline | No quality control on pseudo-labels |
| **Semi-Sup. VAE** (2014) | Generative latent variable model | Principled probabilistic framework | Scaling difficulty, reconstruction overhead |
| **Ladder Networks** (2015) | Layer-wise denoising | 1.06% error, MNIST, 100 labels | Complex architecture, hard to generalize |
| **Semi-Sup. GAN** (2016) | K+1 class discriminator | Good feature learning | Mode collapse, training instability |

### 3.2 Era 2 (2017--2019): Consistency Regularization Takes the Lead

The second era established consistency regularization as the dominant paradigm and shifted the field's center of gravity from generative to discriminative approaches. The key insight: instead of trying to generate data or reconstruct inputs, simply require that predictions be stable under perturbation. This proved far more scalable and general.

**Pi-Model / Temporal Ensembling (Laine & Aila, 2017, ICLR).** Two related ideas that established the paradigm. The Pi-Model applies different random augmentations to the same input twice in the same forward pass and penalizes disagreement. Temporal Ensembling maintains an exponential moving average (EMA) of each sample's prediction across training epochs and uses the averaged prediction as the target. Both implement the same principle — consistency — but Temporal Ensembling is more stable because the target accumulates information over many epochs rather than relying on a single stochastic forward pass.

**Mean Teacher (Tarvainen & Valpola, 2017, NIPS).** The breakthrough refinement. Instead of averaging predictions (Temporal Ensembling), average the model weights themselves:

$$\theta'_t = \alpha \theta'_{t-1} + (1 - \alpha) \theta_t$$

where $\theta_t$ is the student's weights at step $t$, $\theta'_t$ is the teacher's weights (the EMA), and $\alpha$ is a momentum coefficient (typically 0.999). The student is trained on both labeled data and a consistency loss against the teacher's predictions. The teacher never receives gradient updates — it evolves only through the EMA.

Why does this work better than Temporal Ensembling? The teacher provides a consistency target that updates every training step (not every epoch), scales to large datasets without per-sample memory, and produces smoother, more reliable targets because weight averaging is more stable than prediction averaging.

**Virtual Adversarial Training / VAT (Miyato et al., 2018, TPAMI).** Rather than using random perturbations, find the adversarial perturbation that maximally changes the prediction and penalize that change:

$$r_{\text{adv}} = \arg\max_{\|r\| \leq \epsilon} \; D_{\text{KL}}\left(p_\theta(y \mid x) \;\|\; p_\theta(y \mid x + r)\right)$$

This is the worst-case version of consistency regularization — instead of hoping that random augmentation is "hard enough," explicitly find the most challenging perturbation. Elegant and effective, but computationally expensive (requires multiple forward/backward passes per sample).

**UDA / Unsupervised Data Augmentation (Xie et al., 2020, NeurIPS; arXiv 2019).** The paper that proved augmentation quality is the bottleneck. UDA replaced simple random augmentations with strong, learned augmentations — RandAugment for images, back-translation for text — and showed dramatic improvements. The core message: **better augmentation = better SSL**. This insight directly led to Era 3.

A useful way to compare Era 2 methods is by how they construct the consistency target:

| Method | Target Source | Perturbation | Update Frequency | Memory Overhead |
|---|---|---|---|---|
| **Pi-Model** | Same model, different dropout | Random noise/dropout | Every step | None |
| **Temporal Ens.** | EMA of predictions per sample | Random noise/dropout | Every epoch | O(N) per-sample |
| **Mean Teacher** | EMA of model weights | Random noise/dropout | Every step | 1x model copy |
| **VAT** | Same model, adversarial input | Learned adversarial | Every step | None (but costly) |
| **UDA** | Same model, weak aug | Task-specific strong aug | Every step | None |

The evolution of perturbation strategies across Era 2 shows a clear trend:

```
   Perturbation Strategy Evolution (Era 2)
   ═══════════════════════════════════════

   Pi-Model          Mean Teacher         VAT               UDA
   (2017)            (2017)               (2018)            (2019)
      │                  │                  │                 │
      ▼                  ▼                  ▼                 ▼
   ┌────────┐      ┌──────────┐      ┌───────────┐    ┌──────────┐
   │Random  │      │Weight    │      │Adversarial│    │Learned / │
   │noise / │ ───▶ │EMA for   │ ───▶ │worst-case │ ──▶│strong    │
   │dropout │      │stability │      │direction  │    │augment.  │
   └────────┘      └──────────┘      └───────────┘    └──────────┘
       │                │                  │                 │
       │                │                  │                 │
       ▼                ▼                  ▼                 ▼
   "Any noise      "Smooth the       "Find the         "Quality of
    is better       teacher, not      hardest            augmentation
    than none"      the noise"        perturbation"      matters most"
```

### 3.3 Era 3 (2019--2021): Unification and Simplification

Era 3 is where the field crystallized. Researchers realized that the three pillars — pseudo-labeling, consistency regularization, and entropy minimization — could be combined into unified frameworks rather than treated as competing alternatives. The trajectory moved from complex combinations to radical simplification, culminating in FixMatch — a method so clean that it became the default backbone for all subsequent SSL research.

**MixMatch (Berthelot et al., 2019, NeurIPS).** The first unification. MixMatch combines all three pillars in a single algorithm: (1) generate pseudo-labels by averaging predictions across $K$ augmentations (consistency + pseudo-labeling), (2) sharpen the averaged prediction by raising it to a power $1/T$ where $T$ is a temperature parameter (entropy minimization), and (3) apply MixUp interpolation to both labeled and pseudo-labeled data. MixMatch achieved state-of-the-art results across benchmarks and proved that combining the three pillars is better than any one alone. However, it involved many moving parts — temperature sharpening, multiple augmentations, MixUp, and careful hyperparameter tuning — making it difficult to analyze which component contributed most.

**ReMixMatch (Berthelot et al., 2020, ICLR).** Two key additions to MixMatch. First, **distribution alignment**: adjust pseudo-label distributions to match the marginal class distribution of labeled data, preventing the model from ignoring rare classes. Second, **weak-to-strong augmentation anchoring**: generate pseudo-labels from weakly augmented inputs but train on strongly augmented ones. This asymmetry — clean target, noisy input — proved crucial and became the defining feature of the next method.

**FixMatch (Sohn et al., 2020, NeurIPS).** The radical simplification that redefined the field. FixMatch strips away MixMatch's complexity and retains only the essential elements: weak-to-strong augmentation plus a confidence threshold.

The total loss has two terms:

$$\mathcal{L} = \underbrace{\frac{1}{B} \sum_{b=1}^{B} H(y_b, \; p_\theta(y \mid x_b))}_{\text{supervised}} \;+\; \lambda \underbrace{\frac{1}{\mu B} \sum_{b=1}^{\mu B} \mathbb{1}[\max(q_b) \geq \tau] \cdot H(\hat{q}_b, \; p_\theta(y \mid \mathcal{A}(u_b)))}_{\text{unsupervised}}$$

where:
- $B$ is the labeled batch size, $\mu B$ is the unlabeled batch size (with $\mu$ typically 7)
- $q_b = p_\theta(y \mid \alpha(u_b))$ is the model's prediction on the weakly augmented unlabeled input
- $\hat{q}_b = \arg\max(q_b)$ is the pseudo-label (one-hot)
- $\mathcal{A}(u_b)$ is a strong augmentation (e.g., RandAugment + CTAugment)
- $\alpha(u_b)$ is a weak augmentation (e.g., random horizontal flip + crop)
- $\tau = 0.95$ is the confidence threshold
- $\lambda$ is the unsupervised loss weight

The algorithm is beautifully simple:
1. For each unlabeled image, apply weak augmentation, get model prediction
2. If the model is confident enough ($\max \geq 0.95$), convert prediction to a hard pseudo-label
3. Apply strong augmentation to the same image
4. Train the model to match the pseudo-label on the strongly augmented image

FixMatch is important enough to warrant understanding why it works so well. The weak-to-strong asymmetry is key: the pseudo-label comes from an "easy" version of the input (minimal augmentation), so the model is more likely to be correct. But the model must learn to be correct even on a "hard" version (heavy augmentation). This forces the model to learn robust, augmentation-invariant features — which, by the cluster assumption, correspond to semantically meaningful features.

**Noisy Student (Xie et al., 2020, CVPR).** Self-training at ImageNet scale. A teacher model trained on ImageNet labels 300 million unlabeled images from JFT. A student model — equal or larger than the teacher — is trained on both the labeled and pseudo-labeled data, crucially with noise (dropout, stochastic depth, RandAugment) injected into the student but not the teacher. The noise-injected student eventually surpasses the teacher: **88.4% top-1 accuracy on ImageNet**, a new state of the art at the time. The key insight: the student should be noisier than the teacher, not cleaner. Noise during training acts as a regularizer that forces the student to be robust.

The simplification trajectory across Era 3 is striking:

```
   MixMatch → ReMixMatch → FixMatch: The Simplification Arc
   ═══════════════════════════════════════════════════════════

   MixMatch (2019)              ReMixMatch (2020)           FixMatch (2020)
   ┌─────────────────┐          ┌─────────────────┐         ┌─────────────────┐
   │ K augmentations  │         │ K augmentations  │         │ 1 weak aug      │
   │ Average + sharpen│         │ Dist. alignment  │         │ 1 strong aug    │
   │ MixUp on both    │   ───▶  │ Weak-to-strong   │   ───▶  │ Threshold only  │
   │ MSE loss         │         │ Rotation self-sup│         │ CE loss         │
   │ T, K, α params   │         │ T, K, α, τ_a    │         │ τ = 0.95 only   │
   └─────────────────┘          └─────────────────┘         └─────────────────┘
        6+ hyperparams               8+ hyperparams             1 hyperparameter

   Error on CIFAR-10 (40 labels):
        11.08%                        5.44%                     4.26%
```

The pattern is unmistakable: **simpler is better**. FixMatch has essentially one hyperparameter ($\tau$), yet it outperforms the far more complex MixMatch. This is a recurring theme in machine learning — the right inductive bias, cleanly implemented, beats elaborate machinery.

| Method | Year | Pillars Used | Key Innovation | CIFAR-10 (40 labels) | Hyperparams |
|---|---|---|---|---|---|
| **MixMatch** | 2019 | All three + MixUp | First unification | 11.08% | 6+ |
| **ReMixMatch** | 2020 | All three + dist. align | Weak-to-strong anchoring | 5.44% | 8+ |
| **FixMatch** | 2020 | Pseudo-label + consistency | Radical simplification | 4.26% | 1 |
| **Noisy Student** | 2020 | Pseudo-label + noise | ImageNet-scale self-training | — (ImageNet 88.4%) | 3 |

### 3.4 Era 4 (2021--2025): Adaptive Thresholding and Beyond

FixMatch set the template that persists today. Era 4 asks: **can we do better than a fixed confidence threshold?** The answer is yes — through adaptive, soft, and learned thresholding. This is the frontier of SSL research, and it is the era whose innovations have not yet reached protein AI.

The core problem with fixed thresholds becomes clear in practice. Consider CIFAR-100 with only 4 labels per class (400 total). Some classes — say, "bicycle" — are visually distinctive and the model quickly reaches 95% confidence. Other classes — say, "maple tree" vs. "oak tree" — are inherently harder, and the model may never reach 95% confidence even after extensive training. A fixed threshold systematically overrepresents easy classes in the pseudo-labeled pool and starves hard classes of training signal.

**FlexMatch (Zhang et al., 2021, NeurIPS).** The problem with FixMatch's fixed $\tau = 0.95$ becomes acute in imbalanced or fine-grained settings: easy classes quickly generate many pseudo-labels, while hard classes (with consistently lower confidence) generate few or none. On CIFAR-100 with 400 labels, some classes may produce zero pseudo-labels for thousands of training steps while others are saturated. The model over-learns easy classes and under-learns hard ones. FlexMatch introduces **class-adaptive curriculum thresholds**:

$$\tau_c(t) = \frac{\sigma_c(t)}{\max_{c'} \sigma_{c'}(t)} \cdot \tau$$

where $\sigma_c(t)$ is a "learning status" measure — the fraction of unlabeled samples for class $c$ that exceed the base threshold $\tau$ at time $t$. Classes the model has already learned well get a higher threshold (harder to generate pseudo-labels), while struggling classes get a lower threshold (more pseudo-labels to help learning catch up). This is curriculum learning applied to thresholds.

**FreeMatch (Wang et al., 2023, ICLR).** Instead of per-class heuristics, maintain a single global threshold that adapts via exponential moving average of model confidence:

$$\tau_t = \beta \cdot \tau_{t-1} + (1 - \beta) \cdot \frac{1}{|\mathcal{U}_B|} \sum_{x \in \mathcal{U}_B} \max_c \; p_\theta(c \mid x)$$

The threshold tracks the model's evolving competence — low early in training when the model is uncertain, high later when the model is confident. This avoids both the rigidity of FixMatch (threshold too high initially) and the need for per-class tracking of FlexMatch.

**SoftMatch (Chen et al., 2023, ICLR).** The most conceptually elegant approach: **replace the hard threshold entirely with a soft, continuous weighting function**. Instead of including or excluding pseudo-labels based on a binary $\mathbb{1}[\max \geq \tau]$, weight each sample by how close its confidence is to the current training-progress mean:

$$w(x) = \exp\left(-\frac{(\max p_\theta(x) - \mu_t)^2}{2\sigma_t^2}\right) \cdot \mathbb{1}[\max p_\theta(x) \geq \tau_{\min}]$$

Here $\mu_t$ and $\sigma_t$ are the EMA mean and standard deviation of model confidence at step $t$, and $\tau_{\min}$ is a minimal quality floor. The Gaussian weighting means: samples near the current learning frontier (confidence close to $\mu_t$) get the highest weight, while very easy samples (already learned) and very hard ones (probably wrong) are downweighted. This is a truncated Gaussian sample reweighting scheme that naturally implements curriculum learning.

**SemiReward (Wang et al., 2024, ICLR).** A fundamentally different approach: instead of using the model's own confidence to judge pseudo-label quality, **train a separate reward model** to predict whether a pseudo-label is correct. The reward model is trained on labeled data to predict whether a pseudo-label matches the true label, then applied to unlabeled data. This decouples quality estimation from the prediction model itself — addressing the circular dependency of "the model judging its own work." The idea has a natural analog in protein AI: instead of relying on pLDDT (the model's self-assessed confidence), one could train an independent quality predictor on PDB data to evaluate predicted structures. We will explore this in detail in Part 3.

| Method | Threshold Type | Mechanism | Strength | Weakness |
|---|---|---|---|---|
| **FixMatch** | Fixed global | $\tau = 0.95$ for all classes, all time | Simple, robust | Ignores class difficulty, wastes easy/hard data |
| **FlexMatch** | Class-adaptive | $\tau_c \propto$ learning progress per class | Balances class difficulty | Per-class heuristic, sensitive to $\sigma_c$ estimate |
| **FreeMatch** | Self-adaptive global | EMA of model confidence | Adapts to training stage | Single threshold, no class distinction |
| **SoftMatch** | Soft continuous | Truncated Gaussian weighting | No hard boundary, curriculum | Two EMA statistics to track |
| **SemiReward** | Learned | Separate reward model | Decouples quality from prediction | Requires reward model training |

To ground these comparisons in numbers, here is how the FixMatch family performs across standard benchmarks. All methods share the same FixMatch backbone — the only difference is the thresholding strategy:

| Method | CIFAR-10 (40) | CIFAR-10 (250) | CIFAR-100 (400) | CIFAR-100 (2500) | STL-10 (40) |
|---|---|---|---|---|---|
| **FixMatch** | 7.47 | 4.86 | 46.42 | 28.03 | 14.28 |
| **FlexMatch** | 4.97 | 4.51 | 39.94 | 26.49 | 6.59 |
| **FreeMatch** | 4.90 | 4.43 | 38.41 | 25.28 | 5.63 |
| **SoftMatch** | 4.84 | 4.37 | 37.96 | 24.95 | 5.47 |

*Error rates (%, lower is better). Numbers in parentheses indicate number of labeled samples.*

The trend is clear: adaptive and soft thresholding consistently improve upon fixed thresholds, with the largest gains in low-label and many-class settings (CIFAR-100 with 400 labels, STL-10 with 40 labels). These are precisely the regimes most relevant to protein AI, where labeled data is scarce and the "class" space (structural diversity) is enormous.

### 3.5 Master Timeline

The full trajectory from Pseudo-Label to SemiReward spans a decade of rapid innovation:

| Year | Method | Venue | Core Contribution | Key Limitation |
|---|---|---|---|---|
| 2013 | Pseudo-Label | ICML-W | Self-training for deep nets | Confirmation bias, no quality control |
| 2014 | Semi-Sup. VAE | NIPS | Generative SSL | Scaling difficulty, mode coverage |
| 2015 | Ladder Networks | NIPS | Layer-wise denoising | Complex architecture, limited generality |
| 2016 | Semi-Sup. GAN | NIPS | Adversarial regularization | Training instability |
| 2017 | Pi-Model / Temp. Ens. | ICLR | Consistency paradigm established | Stochastic targets, per-sample memory |
| 2017 | Mean Teacher | NIPS | Weight EMA for stable targets | Still uses random perturbations |
| 2018 | VAT | TPAMI | Adversarial consistency | Computational cost (inner loop) |
| 2019 | MixMatch | NeurIPS | Unifies three pillars | Many hyperparameters |
| 2019 | UDA | NeurIPS | Strong augmentation matters | Augmentation design is task-specific |
| 2020 | ReMixMatch | ICLR | Distribution alignment, weak-to-strong | Still complex |
| 2020 | FixMatch | NeurIPS | Radical simplification | Fixed threshold ignores class balance |
| 2020 | Noisy Student | CVPR | ImageNet-scale self-training | Offline, single distillation round |
| 2021 | FlexMatch | NeurIPS | Class-adaptive thresholds | Per-class heuristics |
| 2023 | FreeMatch | ICLR | Self-adaptive EMA thresholds | No class granularity |
| 2023 | SoftMatch | ICLR | Continuous sample weighting | Additional EMA statistics |
| 2024 | SemiReward | ICLR | Learned pseudo-label quality | Reward model training overhead |

---

## 4. Two Key Evolutionary Threads

Looking across all four eras, two threads run through the entire history of SSL. These threads are not just historical curiosities — they represent the two fundamental design axes of any SSL system. Understanding them is essential for Part 3, where we will map them onto protein AI and identify exactly where the field has stopped innovating.

### Thread 1: Perturbation Strategy

How do we create "different views" of the same input for consistency regularization? The trajectory shows a clear progression toward more sophisticated and task-aware perturbations:

- **Random noise / dropout** (Pi-Model, 2017): Any stochastic variation in the forward pass. Cheap and universal, but the perturbations may be too weak to force meaningful invariance.
- **Temporal smoothing** (Temporal Ensembling, 2017): Average predictions over time for stability. Stabilizes the target but does not change the perturbation itself.
- **Weight EMA** (Mean Teacher, 2017): Average the model itself for smoother targets. The teacher becomes a more reliable oracle, making consistency targets more trustworthy.
- **Adversarial** (VAT, 2018): Find the maximally disruptive perturbation. Guarantees the model is tested against worst-case inputs, but expensive to compute.
- **Weak-to-strong augmentation** (FixMatch, 2020): Different augmentation strengths for target vs. training. The key insight: the target should come from an easy view, the training signal from a hard one.

The overall direction is from "any noise" to "carefully designed asymmetric perturbation."

This matters for protein AI because the concept of augmentation maps directly to choices like MSA subsampling depth, template availability, sequence cropping strategy, and coordinate noise injection. All of these are already used in co-folding training, but not in a systematic weak-to-strong framework where the teacher sees the "easy" view and the student must learn from the "hard" one.

### Thread 2: Pseudo-Label Quality Control

How do we prevent the model from training on its own mistakes? This thread tracks the evolution from no filtering to learned quality estimation:

- **None** (Pseudo-Label, 2013): Trust all model predictions. Simple but vulnerable to confirmation bias.
- **Fixed threshold** (FixMatch, 2020): $\tau = 0.95$ for all samples. Effective but ignores that some classes are inherently harder than others.
- **Class-adaptive** (FlexMatch, 2021): Per-class threshold $\tau_c$ based on learning progress. Addresses class imbalance but introduces per-class heuristics.
- **Self-adaptive** (FreeMatch, 2023): Global $\tau_t$ tracks model confidence via EMA. Adapts to training phase without per-class tracking.
- **Soft weighting** (SoftMatch, 2023): Continuous Gaussian weight $w(x)$ replaces binary decision. Eliminates the hard boundary entirely, treating quality as a spectrum.
- **Learned** (SemiReward, 2024): Separate model predicts pseudo-label correctness. Breaks the circular dependency of self-assessed confidence.

The direction here is from "all or nothing" to "nuanced, continuous, and externally validated."

In protein AI, the analogous progression would be from "discard all structures with pLDDT < X" (the current standard) to "weight each predicted structure by a learned quality estimate that accounts for structure type, sequence complexity, and prediction uncertainty." The potential payoff is substantial: current fixed thresholds discard up to 93% of predicted quaternary structures in AFDB, leaving enormous amounts of potentially useful training signal on the table.

The two threads can be visualized together on a timeline:

```
   Two Evolutionary Threads of SSL
   ═══════════════════════════════════════════════════════════════════

   2013    2015    2017      2018    2019    2020      2021   2023  2024
    │       │       │         │       │       │         │      │      │
    ▼       ▼       ▼         ▼       ▼       ▼         ▼      ▼      ▼

   PERTURBATION STRATEGY (Thread 1)
   ─────────────────────────────────────────────────────────────────────
    ·               Random    Adver-          Weak-to-
                    noise/    sarial          strong
                    Weight    (VAT)           (FixMatch)
                    EMA                       ───────────────────────▶
                    (Mean T.)                 [standard since 2020]

   QUALITY CONTROL (Thread 2)
   ─────────────────────────────────────────────────────────────────────
    None                                      Fixed     Class-  Soft    Learned
    (Pseudo-                                  τ=0.95    adapt.  weight  reward
    Label)                                    (Fix-     (Flex-  (Soft-  (Semi-
                                              Match)    Match)  Match)  Reward)
    ·───────────────────────────────────────▶ ·────────▶ ·─────▶ ·─────▶ ·
```

| Era | Perturbation (Thread 1) | Quality Control (Thread 2) | Status in Protein AI |
|---|---|---|---|
| **Era 1** (2013--2016) | Random noise, dropout | None (trust all) | Partially adopted |
| **Era 2** (2017--2019) | Weight EMA, adversarial | None to implicit | Partially adopted |
| **Era 3** (2019--2021) | Weak-to-strong | Fixed threshold $\tau$ | Currently used |
| **Era 4** (2021--2025) | Same as Era 3 | Adaptive, soft, learned | **Not yet adopted** |

The right column — "Status in Protein AI" — is the punchline of this table and the central thesis of this series. Protein AI has adopted ideas from Eras 1-3 (pseudo-labeling, self-distillation, fixed confidence thresholds) but has not yet incorporated the adaptive thresholding and soft weighting advances of Era 4. We will make this case precisely in Parts 2 and 3.

---

## 5. Closing: Why This Matters for Protein AI

If you work on protein structure prediction and have read this far, you may have noticed something: **every co-folding model is already doing semi-supervised learning — they just do not call it that.**

Consider the parallels:
- **AlphaFold2's self-distillation** uses a trained model to predict structures for unlabeled sequences, then retrains on those predictions. This is pseudo-labeling — the foundational SSL technique from 2013.
- **pLDDT-based filtering** discards predicted structures below a confidence threshold before using them for training. This is FixMatch's confidence thresholding — the 2020 technique.
- **Teacher-student training** in Boltz-2 and SeedFold, where one model generates pseudo-structures and another learns from them, is self-training — the overarching SSL paradigm.

The mapping is not superficial — it is structural:

| Protein AI Concept | SSL Equivalent | SSL Era |
|---|---|---|
| Self-distillation (AF2) | Pseudo-labeling | Era 1 (2013) |
| pLDDT filtering (fixed cutoff) | Fixed confidence threshold | Era 3 (2020) |
| Teacher-student distillation | Self-training | Era 1 (2013) |
| EMA model averaging | Mean Teacher | Era 2 (2017) |
| Confidence loss on PDB only (AF3) | Loss-type-aware filtering | No direct CV analog |
| Adaptive thresholds per structure type | FlexMatch class-adaptive $\tau_c$ | **Era 4 (2021) -- not yet adopted** |
| Soft confidence weighting | SoftMatch Gaussian weighting | **Era 4 (2023) -- not yet adopted** |
| Learned quality predictor for structures | SemiReward | **Era 4 (2024) -- not yet adopted** |

But here is the critical observation: the protein AI field stopped at Era 1-2 techniques. AlphaFold's self-distillation is Pseudo-Label (2013). pLDDT filtering with a fixed threshold is FixMatch (2020) at best. No co-folding model uses class-adaptive thresholds, soft weighting, online teacher updates, or learned quality estimation — all techniques proven effective in CV between 2021 and 2024.

The gap is not accidental. Protein AI evolved its training recipes independently, under different terminology (distillation, synthetic data, confidence filtering), without explicitly connecting to the SSL literature. This series aims to bridge that gap — not to argue that CV techniques should be copied blindly, but to show that the SSL framework provides a systematic lens for identifying which improvements are likely to transfer and which require adaptation.

The stakes are real. Every co-folding model trains on millions of synthetic structures. The quality of those structures, and how they are weighted and filtered during training, directly determines model accuracy. A 2-3% improvement in synthetic data utilization — the kind of gain adaptive thresholding delivers in CV — could translate to meaningful improvements in structure prediction for the most challenging targets: multi-domain complexes, disordered regions, and protein-ligand interfaces.

**Next: Part 2 — How Co-Folding Models Use Synthetic Data.** We dissect the distillation strategies of AlphaFold2, AlphaFold3, Boltz-1/2, SeedFold, and OpenFold3 through an SSL lens, mapping each model's choices to specific SSL concepts from this Part.

---

**Next: Part 2 — How Co-Folding Models Use Synthetic Data**

---

*Part of the series: What Protein AI Can Learn from Semi-Supervised Learning*
