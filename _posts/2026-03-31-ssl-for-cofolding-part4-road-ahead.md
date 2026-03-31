---
title: "SSL for Co-Folding Part 4: The Road Ahead — Data Flywheels, Foundation Models, and Open Questions"
date: 2026-03-31 09:30:00 +0900
categories: [Drug Discovery, Foundation Model]
tags: [semi-supervised-learning, co-folding, data-flywheel, foundation-model, open-questions]
math: true
---

**What Protein AI Can Learn from Semi-Supervised Learning**

*This is Part 4 of a 4-part series examining how semi-supervised learning techniques from computer vision can improve protein structure prediction models.*

- **Part 1:** [The Semi-Supervised Revolution in Computer Vision](/posts/ssl-for-cofolding-part1-ssl-revolution-in-cv/)
- **Part 2:** [How Co-Folding Models Use Synthetic Data — An SSL Perspective](/posts/ssl-for-cofolding-part2-synthetic-data-in-cofolding/)
- **Part 3:** [Untapped Opportunities and the Confidence Calibration Trap](/posts/ssl-for-cofolding-part3-untapped-opportunities/)
- **Part 4** (this post): The Road Ahead — Data Flywheels, Foundation Models, and Open Questions


---

## The Core Question

**Where is the intersection of SSL and protein AI heading, and what fundamental challenges remain?**

Let us briefly recap where we have been. In Part 1, we established the SSL framework from computer vision — three pillars (pseudo-labeling, consistency regularization, entropy minimization) and four eras of methodological evolution, from the foundational Pseudo-Label (Lee, 2013) through the radical simplification of FixMatch (Sohn et al., 2020, NeurIPS) to the adaptive and soft thresholding of SoftMatch (Chen et al., 2023, ICLR) and SemiReward (Wang et al., 2024, ICLR). Two evolutionary threads — perturbation strategy and pseudo-label quality control — provided the organizing lens for the entire series.

In Part 2, we dissected every major co-folding model through that SSL lens. AlphaFold2's self-distillation is pseudo-labeling. AlphaFold3's multi-teacher strategy with confidence-loss separation is a sophisticated but still Era-2-level design. Boltz-2's multi-source pipeline, SeedFold's 26.5-million-structure distillation, and OpenFold3's 97%-synthetic training mix all map cleanly to SSL concepts — yet none venture beyond fixed thresholds and offline teachers.

In Part 3, we identified four untapped opportunities — soft weighting, adaptive thresholds, online Mean Teacher, and confidence-weighted loss — and exposed the confidence calibration trap: training confidence heads on synthetic data teaches models to be confidently wrong. AF3's decision to restrict confidence loss to PDB data was highlighted as the single most important design insight in current practice.

Now we look forward. What fundamental challenges remain? Where do the trajectories of SSL and protein AI converge — or diverge? And what should the field prioritize in the next two to five years?

This final Part is deliberately more speculative than the preceding three, but every claim is grounded in either established SSL theory, published protein AI results, or clearly identified analogies. Where we speculate, we say so explicitly.

The structure of this Part follows a natural progression. We start with the data flywheel — the feedback loop that is already running in every co-folding group — and analyze its risks. We then ask what happens when you iterate the flywheel (multi-round distillation) and what happens when foundation models change the underlying dynamics. We propose a concrete alternative to pLDDT-based quality control, examine the evolving strategic value of experimental data, and close with open questions and actionable recommendations.

---

## 1. The Data Flywheel: Promise and Peril

The central dynamic of modern protein AI is a feedback loop: prediction models generate synthetic data, which trains better models, which generate better synthetic data. This is the data flywheel, and every co-folding model from AlphaFold2 onward depends on it.

```
┌─────────────────────────────────────────────────────────────┐
│                      DATA FLYWHEEL                          │
│                                                             │
│    Better teacher ──→ Higher-quality synthetic structures    │
│         ↑                           │                       │
│         │                           ▼                       │
│    Student becomes           Better student model           │
│    next teacher                     │                       │
│         ↑                           ▼                       │
│         └────── Train on PDB + synthetic data ──────┘       │
│                                                             │
│    ⚠ Risk: Model collapse if PDB anchor is too weak         │
└─────────────────────────────────────────────────────────────┘
```

The flywheel is powerful. SeedFold's 26.5 million synthetic structures enabled a model that approaches AlphaFold3's accuracy at a fraction of the compute. Boltz-2 combined AF2 and Boltz-1 predictions to reach competitive multi-target performance. The premise is sound: unlabeled sequences are abundant, prediction models are good enough to generate useful pseudo-labels, and students trained on both real and synthetic data outperform those trained on PDB alone.

But the flywheel has a dark side.

### Model Collapse

When the same lineage of models generates all synthetic data, systematic errors are not just inherited — they are reinforced. Each generation of teacher produces structures with characteristic failure modes, and each generation of student learns those failures as ground truth. This is model collapse: the recursive consumption of model-generated data leading to progressive degradation.

In language modeling, model collapse is now well documented. Shumailov et al. (2024, Nature) showed that training language models on text generated by previous-generation language models causes irreversible degradation — the tails of the distribution collapse first, then the distribution narrows until the model produces only generic, mode-collapsed output. The mechanism is straightforward: low-probability but valid outputs are underrepresented in synthetic data, so each generation has less diversity than the last.

The protein structure analog is concerning. If AlphaFold2 consistently mispredicts a certain fold family — say, transmembrane beta-barrels with unusual loop conformations — then every downstream model trained on AF2 distillation inherits that blind spot. Worse, because the student never sees the correct structure for those sequences, it has no signal to correct the error. The flywheel does not self-correct; it self-reinforces.

We can formalize this risk. Let $p^*$ denote the true distribution of protein structures, and let $p^{(k)}$ denote the distribution of structures generated by the round-$k$ model. Model collapse occurs when:

$$D_{\text{KL}}\left(p^* \;\|\; p^{(k)}\right) > D_{\text{KL}}\left(p^* \;\|\; p^{(k-1)}\right)$$

That is, each generation moves further from the true distribution rather than closer. The PDB acts as a regularizer that pulls $p^{(k)}$ back toward $p^*$ at each round. The critical question is whether the PDB ratio is sufficient to counteract the drift introduced by synthetic data.

### PDB as Ground Truth Anchor

This reframes the Protein Data Bank's role in a critical way. The PDB's strategic value is no longer primarily about data volume — 220,000 structures is a rounding error compared to 200 million synthetic ones. Its value lies in **drift prevention**. The PDB is the only source of structures that are independent of the prediction model lineage. Without sufficient PDB anchoring, the flywheel drifts: predictions converge to a self-consistent but potentially incorrect reality.

Consider the contrast in current practice. Boltz-2 uses a 60:40 PDB-to-synthetic ratio — a strong anchor. OpenFold3 uses roughly 3:97 — the PDB is a thin thread preventing the flywheel from spinning into the void. Which strategy is more sustainable as synthetic data volumes continue to grow? We do not yet have a definitive answer, but the model collapse literature strongly suggests that maintaining a robust experimental anchor is not optional.

Multi-teacher diversity offers a complementary mitigation. If synthetic data comes from structurally independent teachers — say, an Evoformer-based model, a diffusion model, and a physics-based method — the systematic biases of each teacher are less likely to overlap. AF3 already uses multiple teachers (AF2 for monomers, AFM for disorder, AF3 for RNA), but this diversity was driven by modality coverage rather than explicit collapse prevention. A principled multi-teacher strategy, where teacher diversity is a design objective, remains unexplored.

One concrete approach would be to measure teacher agreement as a proxy for pseudo-label reliability. When multiple independent teachers agree on a structure, we have higher confidence that the prediction is correct (or at least that the shared prediction reflects a genuine structural mode rather than a shared artifact). When teachers disagree, the structure should receive lower weight or be excluded. This is analogous to co-training in SSL (Blum & Mitchell, 1998, COLT), where two classifiers trained on different feature subsets provide independent votes — agreement indicates reliability.

The current co-folding landscape actually provides a natural starting point for multi-teacher diversity. AlphaFold3, Boltz-2, Chai-1, and SeedFold use sufficiently different architectures that their prediction errors are not perfectly correlated. A study measuring the overlap in their failure modes — which proteins each model gets wrong, and whether those sets are the same — would directly inform the value of multi-teacher distillation.

| Risk | Mechanism | Mitigation | Current Adoption |
|---|---|---|---|
| **Model collapse** | Recursive synthetic training erodes diversity | Maintain high PDB ratio per round | Partial (varies by model) |
| **Systematic bias amplification** | Teacher errors inherited by student | Multi-teacher diversity | AF3 only |
| **Confidence drift** | Calibration degrades each generation | Confidence loss on PDB only | AF3, OpenFold3 |
| **Convergence to local optima** | All models from same lineage | Independent teacher architectures | Not systematically pursued |
| **Tail distribution loss** | Rare folds underrepresented in synthetic data | Stratified sampling by fold family | Not adopted |

---

## 2. Iterative Self-Training: Multiple Rounds

Noisy Student's key insight was not just that a student can learn from a teacher's pseudo-labels — it was that the student becomes the next teacher, and you iterate. Xie et al. (2020, CVPR) ran three rounds of self-training on ImageNet, with each round producing a student that surpassed the previous teacher. The gains were not monotonic — each round contributed less — but three rounds meaningfully outperformed one.

Every co-folding model in current practice performs exactly one round of distillation. A teacher generates pseudo-labels once, a student trains on them once, and the pipeline stops. Nobody iterates.

### The Iterative Formulation

At round $k$, the student is trained on the union of labeled data and pseudo-labels generated by the round-$k$ model:

$$\theta^{(k+1)} = \arg\min_\theta \mathcal{L}\left(\theta;\; \mathcal{D}_L \cup \hat{\mathcal{D}}_U^{(k)}\right)$$

where the pseudo-labeled set at round $k$ is:

$$\hat{\mathcal{D}}_U^{(k)} = \left\{(x_i, f_{\theta^{(k)}}(x_i)) \mid x_i \in \mathcal{U}\right\}$$

Here $\theta^{(k)}$ denotes the model parameters at iteration $k$, $\mathcal{D}_L$ is the PDB labeled set, $\mathcal{U}$ is the pool of unlabeled sequences, and $f_{\theta^{(k)}}(x_i)$ is the predicted structure for sequence $x_i$ at round $k$.

The appeal is clear: the round-1 student is better than the original teacher, so its pseudo-labels for round 2 should be higher quality. The convergence condition for this process to be beneficial is that pseudo-label quality improves monotonically:

$$\mathbb{E}_{x \sim \mathcal{U}}\left[\text{lDDT}(f_{\theta^{(k+1)}}(x), s^*(x))\right] > \mathbb{E}_{x \sim \mathcal{U}}\left[\text{lDDT}(f_{\theta^{(k)}}(x), s^*(x))\right]$$

That is, the average quality of predicted structures on unlabeled data improves from one round to the next. Whether this condition holds depends on the interplay between improved pseudo-label quality and error accumulation.

### Open Questions for Multi-Round Training

**How many rounds are optimal?** Noisy Student found diminishing returns after three rounds on ImageNet. In that setting, each round improved top-1 accuracy by a decreasing margin: round 1 gave +2.0%, round 2 gave +0.8%, and round 3 gave +0.3%. Protein structure prediction is a harder problem with a more complex output space — the optimal number of rounds is unknown and may depend on the diversity of the unlabeled pool, the quality of the initial teacher, and the heterogeneity of the target distribution (some fold families may benefit from more rounds than others).

**Does each round require full re-prediction?** Re-running inference on millions of sequences is expensive. A single AF2 prediction takes seconds to minutes per sequence; re-predicting 26 million sequences (SeedFold's scale) is a substantial computational investment per round. An incremental strategy — re-predicting only the lowest-confidence structures from the previous round — could reduce cost substantially, but its effectiveness relative to full re-prediction is unknown.

**How should the PDB-to-synthetic ratio change across rounds?** In early rounds, when pseudo-labels are lower quality, a higher PDB ratio may be needed to anchor learning. As pseudo-label quality improves in later rounds, the ratio could potentially shift toward more synthetic data. No principled framework exists for this schedule, but one could imagine an adaptive ratio:

$$r^{(k)} = r_0 \cdot \left(1 - \gamma \cdot \Delta q^{(k)}\right)$$

where $r^{(k)}$ is the PDB ratio at round $k$, $r_0$ is the initial ratio, $\gamma$ is a scaling factor, and $\Delta q^{(k)}$ is the measured quality improvement from round $k-1$ to $k$. When quality improves, the PDB ratio can decrease; when quality stagnates or degrades, it increases.

| Aspect | Single Round (current) | Multi-Round (proposed) |
|---|---|---|
| **Pseudo-label quality** | Fixed at teacher quality | Improves per round |
| **Computational cost** | 1x teacher inference | $k$x teacher inference |
| **Error propagation** | Teacher errors frozen in pseudo-labels | Can self-correct OR amplify |
| **Implementation complexity** | Simple pipeline | Requires iterative infrastructure |
| **Model collapse risk** | Low (one generation) | Higher (accumulates across rounds) |
| **Empirical evidence (CV)** | Baseline | 3 rounds optimal on ImageNet |

The interaction between iterative self-training and model collapse is the key tension. Each round risks amplifying systematic errors, but each round also gives the student an opportunity to improve on the teacher. Whether the improvement or the amplification dominates likely depends on the strength of the PDB anchor and the diversity of the teacher ensemble. This is, in our view, one of the most important empirical questions the field should address.

---

## 3. Foundation Models Change the Game

In computer vision, the rise of foundation models has fundamentally altered SSL's role. A striking result from Zhang et al. (2025, NeurIPS) demonstrated that with powerful vision foundation models like CLIP (Radford et al., 2021, ICML) and DINOv2 (Oquab et al., 2024, TMLR), parameter-efficient fine-tuning (PEFT) on labeled data alone often matches or exceeds classical SSL performance. When your backbone already encodes rich, general-purpose representations, the marginal value of pseudo-labels shrinks.

This represents a paradigm shift in SSL's purpose:

```
┌───────────────────────────────────────────────────────────┐
│  Pre-foundation model era:                                │
│    SSL's role = "Learn good representations               │
│                  from unlabeled data"                      │
│                                                           │
│  Foundation model era:                                    │
│    SSL's role = "Refine already-strong representations    │
│                  with task-specific pseudo-labels"         │
│                                                           │
│  Protein AI question:                                     │
│    Which era are we in? And does SSL's role change        │
│    when the output is 3D structure rather than a class?   │
└───────────────────────────────────────────────────────────┘
```

### Protein Language Models as Foundation Models

Does protein AI have its own foundation models? Arguably, yes. ESM-2 (Lin et al., 2023, Science) is a 15-billion-parameter protein language model trained on hundreds of millions of sequences. It learns evolutionary patterns, secondary structure propensities, and contact maps directly from sequence data — no experimental structures required. Smaller but powerful models like ESM-2 3B serve as embedding layers in several co-folding models: Chai-1 uses ESM-2 3B embeddings, and Nucleic-acid Prediction 3 (NP3) combines ESM-2 with RiNALMo for RNA.

But there is a crucial difference between protein language models and vision foundation models. Vision foundation models like DINOv2 learn representations that directly encode spatial and semantic structure — the kind of information classification needs. Protein language models encode evolutionary information and sequence-level patterns, but they do not directly capture 3D structural geometry. The gap between a PLM embedding and a predicted 3D structure is wider than the gap between a DINOv2 feature and a class label.

This gap is not merely quantitative. Consider what each foundation model provides versus what the downstream task requires:

- **DINOv2 for image classification**: The foundation model outputs a feature vector that linearly separates most classes. A linear probe on DINOv2 features achieves 80%+ accuracy on ImageNet. The remaining SSL/fine-tuning work is incremental — pushing from 80% to 86%.
- **ESM-2 for structure prediction**: The language model outputs per-residue embeddings that encode evolutionary conservation, secondary structure propensity, and some contact information. But converting these embeddings to full 3D atomic coordinates requires substantial additional computation — the Evoformer/Pairformer trunk, the structure module/diffusion head, and iterative refinement. The PLM provides the starting point, not the near-final answer.

This means the CV conclusion — "foundation models make SSL unnecessary" — does not straightforwardly transfer to protein AI. The structural prediction task still requires substantial computation beyond what the PLM provides, and pseudo-labels from teacher models remain valuable for bridging the gap between evolutionary embeddings and 3D coordinates.

| Era | Computer Vision | Protein AI | SSL's Function |
|---|---|---|---|
| **Pre-FM** | No pretrained features | No PLM embeddings | Learn representations from scratch |
| **Early FM** | ImageNet pretrained CNN | ESM-2 embeddings as input | Augment pretrained features |
| **Current FM** | CLIP / DINOv2 | PLM + co-folding model | Refine already-strong representations |
| **Future** | FM + PEFT sufficient? | PLM + minimal fine-tuning? | Pseudo-labels for distribution expansion? |

### The Convergence Question

The question the field will need to answer over the next few years: **with a strong PLM backbone, does classical distillation still help?** If the PLM already captures the evolutionary information that MSA provides, and the co-folding architecture is powerful enough to convert that to 3D structure, then the value of pseudo-labeled structures may diminish.

Alternatively, pseudo-labels may remain essential precisely because they provide 3D supervision that no sequence-only model can supply. A PLM trained on sequences sees evolutionary covariance — which residues co-evolve — but never sees an actual 3D coordinate. Synthetic structures from a teacher model provide explicit geometric supervision that complements the PLM's evolutionary signal. In this view, PLMs and SSL are not substitutes but complements, each contributing a different type of information.

Our expectation is the latter — PLMs and distillation will remain complementary for the foreseeable future. But this is an empirical question, and the answer may depend on model scale and the richness of the PLM's internal representations.

There is a subtler point worth making. Even if PLMs eventually encode enough information for accurate structure prediction without distillation, the SSL framework remains relevant for a different reason: **distribution expansion**. PLMs are trained on naturally occurring sequences, and co-folding models are trained on PDB structures — both are limited to the distribution of sequences and structures that evolution has explored. For applications like de novo protein design, antibody engineering, or binding to non-natural targets, the relevant structures lie outside the training distribution. Pseudo-labeling strategies that systematically generate and validate structures for designed or non-natural sequences could extend the reach of co-folding models into these therapeutically critical regions, regardless of how powerful the PLM backbone becomes.

---

## 4. Learned Quality Estimation: Beyond pLDDT

In Part 3, we analyzed the confidence calibration trap: training confidence heads on synthetic data teaches the model to be confidently wrong. AF3's solution — restricting confidence loss to PDB data — is effective but does not address a deeper issue. Even with PDB-only confidence training, the quality of pseudo-labels is still assessed by the teacher's own confidence metric. pLDDT is the teacher evaluating its own homework.

The SemiReward framework (Wang et al., 2024, ICLR) offers an alternative: train a separate "rewarder" network — independent of both teacher and student — to predict pseudo-label quality. Applied to protein AI, this translates to an independent quality predictor for synthetic structures.

### The Concrete Proposal

We have a dataset of PDB proteins for which we can compute both the predicted structure (from any teacher model) and the actual experimental structure. We can compute the true lDDT between prediction and experiment. This gives us training data for a quality predictor: input a (sequence, MSA features, predicted structure) tuple, and predict what the lDDT would be if we had the experimental structure.

The quality predictor objective is:

$$\mathcal{L}_{\text{QP}} = \sum_{i \in \mathcal{D}_L} \left(Q_\phi(x_i, \hat{s}_i) - \text{lDDT}(\hat{s}_i, s_i^*)\right)^2$$

where $Q_\phi$ is the quality predictor with parameters $\phi$, $\hat{s}_i$ is the teacher's predicted structure for sequence $x_i$, $s_i^*$ is the experimental structure, and $\text{lDDT}(\hat{s}_i, s_i^*)$ is the actual local distance difference test score.

At deployment, the quality predictor evaluates synthetic structures generated for sequences that lack experimental structures. Its estimate $Q_\phi(x_i, \hat{s}_i)$ replaces pLDDT as the filtering or weighting criterion.

```
pLDDT (self-assessed):
  ┌──────────┐
  │  Teacher  │──── predict structure ──→ structure
  │  (same)   │──── predict pLDDT ─────→ "I'm 85% confident"
  └──────────┘     ⚠ self-evaluation bias

Learned Quality Predictor (third-party):
  ┌──────────┐
  │  Teacher  │──── predict structure ──→ structure ─┐
  └──────────┘                                       │
                                                     ▼
  ┌──────────────┐                              ┌─────────┐
  │   Quality    │◄── (sequence, MSA, struct) ──│ combine  │
  │   Predictor  │──── "Actually 72% accurate"  └─────────┘
  │   (separate) │     ✓ independent evaluation
  └──────────────┘
```

### Why This Could Work — and the Challenges

Why would a learned quality predictor outperform pLDDT? Three reasons. First, it can identify patterns of teacher failure that the teacher itself cannot detect — for instance, structures that look internally self-consistent (high pLDDT) but deviate from experimental ground truth in characteristic ways. Second, it is trained on actual quality labels (true lDDT), not on the teacher's self-assessed confidence. Third, it can incorporate features the teacher does not explicitly model, such as MSA depth, sequence conservation patterns, or structural motifs associated with prediction difficulty.

There is precedent for this kind of external model quality assessment (MQA) in protein structure prediction. Tools like QMEANDisCo (Studer et al., 2020, Bioinformatics), VoroMQA (Olechnovič & Venclovas, 2017, Proteins), and the more recent GNN-based quality predictors used in CASP evaluations all estimate structure quality from sequence and structural features. The SemiReward-inspired proposal extends this idea by training the quality predictor specifically to assess pseudo-label quality for the purposes of SSL — optimizing for the downstream use case of sample weighting or filtering rather than for general quality assessment.

One could integrate the quality predictor directly into the soft weighting scheme from Part 3. Instead of using pLDDT to compute sample weights, use $Q_\phi$:

$$w_i = \exp\left(-\frac{(Q_\phi(x_i, \hat{s}_i) - \mu_t)^2}{2\sigma_t^2}\right) \cdot \mathbb{1}[Q_\phi(x_i, \hat{s}_i) \geq \tau_{\min}]$$

This combines two ideas — soft weighting from SoftMatch and learned quality estimation from SemiReward — into a single, principled framework for pseudo-label utilization.

The practical challenge is generalization. The quality predictor is trained on PDB proteins, but must evaluate predictions for sequences that are, by definition, not in the PDB. If the distribution of prediction errors differs substantially between PDB-like and non-PDB sequences, the quality predictor may not transfer well. This is a form of distribution shift that requires careful validation. However, it is worth noting that this distribution shift is no worse than the shift already implicit in using pLDDT-filtered synthetic data — pLDDT is also calibrated on PDB-like proteins and may be less reliable on distant sequences.

---

## 5. The Strategic Value of Experimental Data

Here is a counter-intuitive observation: as synthetic data becomes abundant, experimental data becomes more valuable, not less.

This seems paradoxical. If we can generate 200 million synthetic structures, why does the PDB's 220,000 matter? The answer lies in three irreplaceable roles that experimental data plays.

**First, calibration anchor.** As we discussed in Section 1, the PDB is the only dataset independent of the prediction model lineage. Without it, the data flywheel has no fixed point — predictions converge to whatever self-consistent attractor the model finds, whether or not it matches physical reality. Every model in the flywheel can agree on a wrong answer; only experimental data can break that consensus.

**Second, systematic error correction.** Some prediction errors are consistent across all models because all models share similar inductive biases. Metal coordination geometries with unusual ligand arrangements, non-canonical RNA base pairing, and certain membrane protein topologies are examples where every model in the current lineage struggles. Only new experimental structures for these cases can provide the corrective signal. Synthetic data generated by models that share these blind spots will never fix them — they will only reinforce the shared misconception.

**Third, genuinely novel structural information.** The cryo-EM revolution is producing structures of large complexes, membrane proteins, and dynamic assemblies that have never been seen before. These are not incremental additions to existing fold families — they represent genuinely new structural knowledge that cannot be interpolated from existing data. Each new cryo-EM structure of a novel complex expands the boundary of what prediction models can learn. Recent structures of the nuclear pore complex, large ribosomal assemblies, and multi-subunit membrane channels fall into this category.

Beyond structure, functional measurements represent an even scarcer and more strategically valuable data type. Binding affinities ($K_d$, $K_i$), inhibition constants (IC$_{50}$), and other activity data require wet-lab experiments that no prediction model can synthesize. These measurements are the ultimate bottleneck for drug discovery applications, and they define the next frontier for the data flywheel — one where the gap between labeled and unlabeled data is even wider than for structures.

| Data Type | Scale | Can Synthetic Replace? | Strategic Value | Trend |
|---|---|---|---|---|
| **Protein sequences** | ~2.5B | N/A (input, not label) | Foundation | Saturating |
| **Synthetic structures** | ~200M | Self-referential | Coverage expansion | Growing rapidly |
| **PDB experimental** | ~220K | No — irreplaceable anchor | Ground truth + calibration | Growing (cryo-EM) |
| **Functional data** ($K_d$, IC$_{50}$) | ~10--50K | No — requires wet lab | Highest per-datum value | Scarce bottleneck |

The implication for resource allocation is clear. Investing in experimental structure determination — particularly for protein families where predictions are systematically poor — yields outsized returns in the SSL framework. Each new experimental structure does not just add one training example; it anchors the flywheel for an entire class of related sequences. Similarly, curating and expanding functional measurement databases may be the highest-leverage investment for downstream applications.

This argument has a direct analog in the active learning literature. When we have a fixed budget for labeling, we should label the examples that are most informative — not a random sample, and not the easy ones. For protein AI, this translates to: the next cryo-EM structure to solve should be the one that maximally reduces uncertainty across the largest family of related sequences. Developing principled selection criteria for experimental targets, informed by the SSL framework, is an actionable research direction.

The interplay between data types also matters for how SSL techniques should be applied. Structures provide geometric supervision — the loss is FAPE, distogram, or diffusion denoising. Functional data provides scalar supervision — the loss is regression on $K_d$ or classification on activity. The pseudo-labeling strategies that work for geometric outputs may not transfer directly to scalar functional outputs, and vice versa. This suggests that a multi-task SSL framework — one that handles structural and functional pseudo-labels with different quality control mechanisms — may be necessary for next-generation models that predict both structure and function.

---

## 6. Open Questions

We close the analytical portion of this series with a set of open questions. These are not rhetorical — they represent genuine research gaps where we believe focused effort would yield meaningful advances.

- **Optimal experimental-to-synthetic ratio.** Does a universal optimum exist, or is it architecture-dependent? Boltz-2's 60:40 PDB-to-synthetic split and OpenFold3's 3:97 split represent radically different philosophies. Systematic ablation studies varying this ratio would be informative, but the computational cost of training full co-folding models makes controlled experiments difficult. A formal framework connecting the ratio to generalization bounds — analogous to the labeled-to-unlabeled ratio theory in SSL (Wei et al., 2021, NeurIPS) — would be valuable.

- **Multi-teacher diversity versus quality.** Is it better to have multiple mediocre teachers or one excellent teacher? AF3's multi-teacher strategy (AF2 for monomers, AFM for disorder, AF3 itself for RNA) suggests diversity matters. But this diversity was driven by practical necessity (different modalities), not by a principled analysis of how teacher diversity affects student learning. The SSL literature on multi-teacher knowledge distillation could provide guidance here.

- **Adaptive thresholding in continuous structure space.** All adaptive thresholding methods from CV (FlexMatch, FreeMatch, SoftMatch) assume discrete class labels. Protein structures live in a continuous space — there is no clean analog of "class." Does adaptive thresholding still help when the output is a 3D coordinate set rather than a class probability? The answer likely depends on how we define the "classes" — by fold type, by structural difficulty, by sequence identity to PDB — and the right granularity is an open question.

- **Online Mean Teacher computational overhead.** The online Mean Teacher requires maintaining two copies of the model in memory and running online teacher inference during training. For co-folding models that already push the limits of GPU memory (AF3's diffusion module, Boltz-2's confidence module), this may be impractical without architectural compromises. Is the improvement justified by the 2x memory cost? Could approximations — updating the teacher every $N$ steps rather than every step — preserve most of the benefit at reduced cost?

- **Foundation models and SSL convergence.** Will protein language models eventually make classical distillation unnecessary for structure prediction? If a sufficiently powerful PLM encodes enough structural information that a lightweight decoder can recover 3D coordinates, then the role of synthetic structures as training data diminishes. We believe this convergence is far off — the gap between evolutionary embeddings and 3D geometry is substantial — but monitoring this trajectory is important.

- **Confidence calibration standards.** The field lacks a standardized benchmark for evaluating confidence calibration across models. CASP and CAMEO evaluate prediction accuracy, but do not systematically assess whether a model's confidence estimates are reliable. Should future CASP rounds include calibration metrics — expected calibration error, reliability diagrams, calibration-conditioned accuracy — as standard evaluation criteria? We argue yes. Miscalibrated confidence directly impacts downstream applications: a drug discovery campaign that trusts a model's 90% confidence prediction may invest months of wet-lab effort on a structure that is actually 60% accurate.

- **Cross-modal transfer.** Can insights from protein structure SSL transfer to RNA structure prediction, molecular dynamics simulation, or drug-binding affinity prediction? RNA structure prediction (RoseTTAFold2NA, NP3) already uses synthetic data, but the SSL analysis has not been applied. Molecular dynamics involves continuous trajectories rather than static structures, raising new questions about what constitutes a "pseudo-label" in a dynamical context. Binding affinity prediction faces an even more extreme labeled-data bottleneck than structure prediction, making it a natural candidate for SSL techniques.

These questions span different time horizons. Some — like optimal data ratios and confidence calibration standards — could be addressed with focused empirical studies in the near term. Others — like the foundation model convergence question — will play out over years as both PLMs and co-folding architectures evolve. The following table provides a rough categorization.

| Time Horizon | Open Question | Required Effort |
|---|---|---|
| **Near-term** (6--12 months) | Optimal PDB-to-synthetic ratio ablation | Single model, systematic sweep |
| **Near-term** | Confidence calibration benchmark | Community agreement on metrics |
| **Medium-term** (1--2 years) | Multi-round distillation empirics | Iterative pipeline + compute |
| **Medium-term** | Learned quality predictor validation | Train + evaluate quality model |
| **Medium-term** | Adaptive thresholding for continuous outputs | Algorithmic adaptation + experiments |
| **Long-term** (2--5 years) | Foundation model vs SSL convergence | Depends on PLM scaling trajectory |
| **Long-term** | Cross-modal SSL transfer | Requires multi-domain expertise |

---

## 7. Series Conclusion

Across four Parts, we have made a single core argument: **the co-folding field is operating at SSL Era 1--2 maturity, and systematically adopting computer vision's decade of SSL advances can yield meaningful improvements in both accuracy and training efficiency.**

### The Series at a Glance

```
┌──────────────────────────────────────────────────────────────────┐
│        SERIES OVERVIEW: SSL × Protein AI                         │
│                                                                  │
│  Part 1: SSL Framework from CV                                   │
│  ├── Three pillars: pseudo-labeling, consistency, entropy min.   │
│  ├── Four eras: Foundations → Consistency → Unification → Adapt. │
│  └── Two threads: perturbation strategy + quality control        │
│           │                                                      │
│           ▼                                                      │
│  Part 2: Co-Folding Models as SSL Systems                        │
│  ├── AF2 self-distillation = pseudo-labeling (2013)              │
│  ├── pLDDT filtering = FixMatch thresholding (2020)              │
│  ├── AF3 multi-teacher = cross-distillation                      │
│  └── All models: Era 1-2 techniques only                         │
│           │                                                      │
│           ▼                                                      │
│  Part 3: Untapped Opportunities + Confidence Trap                │
│  ├── Soft weighting (SoftMatch, 2023) — not adopted              │
│  ├── Adaptive thresholds (FlexMatch, 2021) — not adopted         │
│  ├── Online Mean Teacher (2017) — not adopted                    │
│  └── Confidence calibration trap — AF3 fixed, others not         │
│           │                                                      │
│           ▼                                                      │
│  Part 4: The Road Ahead                                          │
│  ├── Data flywheel risks (model collapse, drift)                 │
│  ├── Iterative self-training (multi-round distillation)          │
│  ├── Foundation models (PLM × SSL interaction)                   │
│  └── Learned quality estimation (beyond pLDDT)                   │
└──────────────────────────────────────────────────────────────────┘
```

### Three Immediately Actionable Recommendations

Three recommendations deserve re-emphasis because they are immediately actionable with minimal engineering effort:

**1. Soft weighting.** Replace hard pLDDT thresholds with continuous weights. The SoftMatch-style Gaussian weighting function:

$$w_i = \exp\left(-\frac{(\text{pLDDT}_i - \mu_t)^2}{2\sigma_t^2}\right) \cdot \mathbb{1}[\text{pLDDT}_i \geq \tau_{\min}]$$

requires minimal code changes and immediately makes millions of currently discarded synthetic structures usable at reduced weight. The cost is one line of code. The potential impact is substantial — structures with pLDDT 60-70 that are currently thrown away could contribute meaningful training signal at low weight, particularly for difficult targets like disordered regions and loop conformations.

**2. Confidence loss separation.** Follow AF3's lead — train confidence heads (pLDDT, PAE, PDE) on PDB data only. Training confidence on synthetic data teaches the model to replicate the teacher's confidence errors, which degrades calibration and undermines downstream decision-making. The cost is a minor training loop modification. The impact is trustworthy confidence estimates that correctly reflect prediction uncertainty.

**3. Weak-to-strong augmentation asymmetry.** Generate pseudo-labels with full MSA depth and no noise (weak augmentation of the teacher's input), but train the student with subsampled MSAs, random cropping, and coordinate noise (strong augmentation). This is FixMatch's core principle applied to protein structure prediction. The cost is a data pipeline adjustment. The impact is improved robustness and better generalization to sequences with shallow MSAs — a critical capability for predicting structures of orphan proteins and metagenomic sequences.

| Recommendation | SSL Analog | Implementation Cost | Expected Impact | Evidence Base |
|---|---|---|---|---|
| **Soft weighting** | SoftMatch (2023) | One line of code | Millions of structures recovered | Strong (CV ablations) |
| **Confidence loss separation** | Domain-specific insight | Training loop modification | Calibrated confidence | AF3 ablation |
| **Weak-to-strong augmentation** | FixMatch (2020) | Data pipeline adjustment | Robustness to MSA depth | Strong (CV theory + practice) |

### Closing Thought

We began this series by observing that the ratio of unlabeled to labeled data in protein AI exceeds 1,000:1 — far more extreme than any SSL benchmark in computer vision. We showed that every co-folding model already performs SSL under different nomenclature, but with techniques from 2013-2020 while a decade of advances sits on the shelf. We identified specific, actionable techniques and analyzed the risks of the data flywheel that the entire field depends on.

The most exciting aspect of this intersection is not any single technique, but the realization that protein AI has a parallel field — one with over a decade of rigorous theory and extensive empirical results — whose insights are directly transferable. Semi-supervised learning in computer vision went through years of trial and error to learn that soft weighting beats hard thresholds, that teacher confidence should not train student confidence, that weak-to-strong augmentation asymmetry is essential, and that iterative self-training with diverse teachers outperforms single-round distillation. Protein AI does not need to re-discover these lessons from scratch.

The data flywheel will continue to spin. Foundation models will grow more powerful. Experimental data will remain irreplaceable. And the techniques we have outlined — from simple soft weighting to learned quality estimation — offer a clear path from the current Era 1-2 practice to the frontier of SSL methodology. The gaps we have identified are not speculative possibilities but demonstrated techniques, proven in a closely analogous domain, waiting to be adapted.

The question is no longer whether these techniques could help, but how quickly the field will adopt them.

We look forward to seeing these ideas tested, challenged, and refined by the community. The bridge between SSL and protein AI is built; it is time to cross it.

---

## Notation Reference

| Symbol | Meaning |
|---|---|
| $\theta^{(k)}$ | Model parameters at iteration $k$ |
| $\mathcal{D}_L$ | Labeled dataset (PDB experimental structures) |
| $\hat{\mathcal{D}}_U^{(k)}$ | Pseudo-labeled dataset at iteration $k$ |
| $\mathcal{U}$ | Unlabeled sequence pool |
| $f_{\theta^{(k)}}(x)$ | Model prediction (structure) at iteration $k$ |
| $Q_\phi$ | Quality predictor with parameters $\phi$ |
| $s^*$ | Experimental (ground truth) structure |
| $\hat{s}$ | Predicted structure |
| lDDT | Local Distance Difference Test |
| pLDDT | Predicted lDDT (model confidence metric) |
| $w_i$ | Sample weight for pseudo-labeled example $i$ |
| $\mu_t, \sigma_t$ | EMA mean and standard deviation of model confidence at step $t$ |
| $p^{(k)}$ | Distribution of structures generated by round-$k$ model |

---

**End of Series: What Protein AI Can Learn from Semi-Supervised Learning**

---

*Part of the series: What Protein AI Can Learn from Semi-Supervised Learning*
