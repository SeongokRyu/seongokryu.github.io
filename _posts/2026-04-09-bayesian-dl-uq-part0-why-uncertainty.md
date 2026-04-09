---
title: "Bayesian DL & UQ Part 0: Beyond Predictions — Why Uncertainty Matters"
date: 2026-04-09 14:00:00 +0900
categories: [AI & ML, ML Theory]
tags: [bayesian-deep-learning, uncertainty-quantification, calibration, overconfidence, reliability-diagram]
math: true
---

**Bayesian Deep Learning and Uncertainty Quantification — A Journey Toward Trustworthy AI**

*This is Part 0 of an 8-part series.*

- **Part 0:** Beyond Predictions — Why Uncertainty Matters (this post)
- **Part 1:** [The Language of Bayesian Inference](/posts/bayesian-dl-uq-part1-bayesian-language/)
- **Part 2:** [The Art of Approximation — From Variational to Ensemble](/posts/bayesian-dl-uq-part2-art-of-approximation/)
- **Part 3:** [The Anatomy of Uncertainty — Aleatoric vs. Epistemic](/posts/bayesian-dl-uq-part3-anatomy-of-uncertainty/)
- **Part 4:** [Calibration and Conformal Prediction](/posts/bayesian-dl-uq-part4-calibration-conformal/)
- **Part 5:** [Single-Pass UQ — Evidential Deep Learning and Distance-Aware Methods](/posts/bayesian-dl-uq-part5-single-pass-uq/)
- **Part 6:** [UQ in Science — Molecules, Proteins, and Materials](/posts/bayesian-dl-uq-part6-uq-in-science/)
- **Part 7:** [The Future of UQ — Uncertainty in the Age of Foundation Models](/posts/bayesian-dl-uq-part7-future-foundation-models/)

---

## Hook: 99% Confident, 100% Wrong

In 2024, an AI team at a pharmaceutical company synthesized ten candidate drug molecules. Their deep learning model predicted the binding affinity of these molecules to a target protein with high confidence, and the team prioritized synthesis in order of predicted potency. The result? Eight out of ten showed no meaningful activity whatsoever. The model's prediction said "this molecule is promising," but it said **nothing about how much that prediction should be trusted**.

This is not a hypothetical story. Modern deep learning models are **astonishingly self-assured**, and that confidence is often unfounded. This series is a journey to diagnose — and resolve — this problem.

---

## 1. The Overconfidence Problem — Models Never Say "I Don't Know"

### 1.1 Medical Imaging: When Lives Hinge on Confidence

Consider a deep learning model that detects lung nodules in chest X-rays. When the model outputs **"96% probability of malignancy,"** how should a clinician interpret that number?

- If the model is well **calibrated** — meaning roughly 96% of cases where it says 96% are indeed malignant — this number serves as a **sound basis for clinical decisions**
- If the model is **miscalibrated** — meaning only about 60% of its 96%-confidence predictions are actually malignant — this number is a **dangerous illusion**

The trouble is that most modern deep learning models fall into the latter category. Guo et al. (2017) demonstrated that contemporary architectures (ResNet, DenseNet, etc.) trained on CIFAR-100 and ImageNet are **systematically overconfident**. When a model claims "90% confidence," its actual accuracy is often only 70–80%.

### 1.2 Autonomous Driving: False Certainty About the Unfamiliar

An object recognition model in a self-driving vehicle classifies objects into categories like "car," "pedestrian," and "bicycle." What happens when the model encounters **an object entirely absent from its training data** — say, a refrigerator lying on the road, or a person in a Halloween costume?

- An ideal model: **"I don't know what this is"** (high uncertainty)
- A real-world model: **"It's a pedestrian, 99% confident"** (low uncertainty, high error)

The structural limitation of the softmax function lies at the heart of this problem. Softmax **always** outputs a probability distribution — no matter how unfamiliar the input, it must assign high probability to some class. There is simply **no mechanism** for the model to express "this input is unlike anything I was trained on."

### 1.3 Drug Discovery: The $100K Lesson in Synthesis Costs

This very problem motivated my 2019 work on Bayesian GCN (Ryu, Kwon & Kim, *Chemical Science*).

- A molecular property prediction model says **"this molecule has high solubility"**
- But there is no information about **how far this molecule lies from the chemical space** covered by the training data
- Only after weeks of synthesis and experimentation does the team discover the prediction was wrong

This experience exposes a fundamental limitation of point predictions. **"This molecule has high solubility"** is far less useful to a researcher than **"This molecule is predicted to have high solubility, but the uncertainty is large — the training data contains very few molecules with a similar scaffold."** Had the latter been available, the team would have **prioritized synthesizing molecules whose predictions were trustworthy** instead.

### 1.4 The Guo et al. Finding — Most Modern Networks Are Miscalibrated

The landmark paper that systematically analyzed this issue is Guo et al. (2017), **"On Calibration of Modern Neural Networks."** Its key findings can be summarized as follows:

- **Older models (e.g., LeNet) were relatively well calibrated**
- **Modern models (ResNet, DenseNet, etc.) are systematically overconfident** — accuracy improved, but calibration actually deteriorated
- **Miscalibration worsens as model capacity grows**
- **Modern training practices such as batch normalization and reduced weight decay** contribute to calibration degradation

Intuitively, this is **a distinct phenomenon from overfitting**. Even when test accuracy is high, confidence scores can be systematically inflated. The models have become "more accurate and simultaneously more overconfident."

---

## 2. Visualizing Calibration with Reliability Diagrams

### 2.1 What Perfect Calibration Means

**Calibration** refers to the degree to which a model's predicted probabilities match the actual frequency of correct outcomes. Formally:

$$P(\hat{Y} = Y \mid \hat{P} = p) = p, \quad \forall\, p \in [0, 1]$$

In other words, among all samples for which the model predicts confidence $p$, exactly a fraction $p$ should actually be correct.

- The model says **"80% confident"** on 100 predictions, and 80 turn out correct → **perfect calibration**
- The model says **"80% confident"** on 100 predictions, and only 60 turn out correct → **overconfident**
- The model says **"80% confident"** on 100 predictions, and 95 turn out correct → **underconfident**

The most intuitive tool for visualizing this is the **reliability diagram**.

### 2.2 Anatomy of a Reliability Diagram

A reliability diagram plots predicted confidence (horizontal axis) against actual accuracy (vertical axis):

```
  Accuracy (fraction correct)
  1.0 |                                          *
      |                                     *  /
      |                                   /  /
  0.8 |                              *  /  /
      |                            /  *  /
      |                      *  /     /
  0.6 |                     /  /    /
      |                *  /  /    /
      |              /  *  /    /       * = perfect calibration
  0.4 |         *  /     /    /             (diagonal)
      |        /  /    /    /
      |   *  /  /    /    /          o = actual modern network
  0.2 |  /  /  /    /    /               (below diagonal)
      | / o/ o/  o/  o/
      |/ /  /  /  /
  0.0 +--+--+--+--+--+--+--+--+--+--+--> Confidence
      0    0.2  0.4  0.6  0.8  1.0
```

```
  Ideal calibration              Reality of modern networks
  (Perfect)                      (Overconfident)

  Acc                             Acc
  1.0 |        /                  1.0 |        /
      |       /                       |       / .
      |      /                        |      / .
      |     /                         |     /.
      |    /                          |   ./
      |   /                           |  ./ <- Gap =
      |  /                            | ./    miscalibration
      | /                             |./
      |/                              |/
      +-----------> Conf              +-----------> Conf
      0    0.5   1.0                  0    0.5   1.0
```

**Key observation**: For most modern deep learning models, the reliability diagram lies **below the diagonal**. This means the confidence reported by the model is **systematically higher** than its actual accuracy.

### 2.3 Expected Calibration Error (ECE)

The standard metric for summarizing calibration as a single number is the **ECE (Expected Calibration Error)**:

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

where:
- $M$: number of confidence bins
- $B_m$: set of samples falling in the $m$-th bin
- $|B_m|$: number of samples in that bin
- $\text{acc}(B_m)$: actual accuracy within the bin
- $\text{conf}(B_m)$: average confidence within the bin

**ECE = 0 means perfect calibration**; larger values indicate worse miscalibration.

Results from Guo et al. (2017):

| Model | Test Accuracy | ECE (%) |
|:------|:---:|:---:|
| LeNet-5 (1998) | 75.3% | 3.05 |
| ResNet-110 (2016) | 93.0% | 4.41 |
| DenseNet-BC (2017) | 93.5% | 5.35 |

Accuracy improved substantially, yet **ECE actually got worse**. The models became smarter but more overconfident at the same time.

### 2.4 Temperature Scaling — A Surprisingly Simple One-Parameter Fix

Guo et al. also proposed a remarkably simple remedy: **Temperature Scaling**. The idea is to divide the logits $\mathbf{z}$ by a single scalar parameter $T > 0$ before applying softmax:

$$\hat{q}_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

- $T = 1$: original softmax output
- $T > 1$: the output distribution becomes **softer** (softening) → confidence decreases
- $T < 1$: the output distribution becomes **sharper** (sharpening) → confidence increases

For overconfident modern networks, $T > 1$ is typically appropriate. Finding just **one parameter** $T$ that minimizes NLL (Negative Log-Likelihood) on a validation set can dramatically reduce ECE.

However, temperature scaling has fundamental limitations:

- **It does not change the model's representations**: the learned features and decision boundaries remain unchanged
- **It is a global, input-agnostic scaling**: the same $T$ is applied to every sample, so it cannot capture per-sample differences in uncertainty
- **It does not help with OOD (Out-of-Distribution) detection**: there is no mechanism to raise uncertainty for inputs outside the training distribution

This is precisely why **a more principled approach — Bayesian Deep Learning — is needed**.

---

## 3. Two Faces of Uncertainty — A Preview

Not all uncertainty is of the same kind. Uncertainty arises from two fundamentally different sources.

### 3.1 Aleatoric Uncertainty — Noise Inherent in the Data

**Aleatoric uncertainty** (data uncertainty, irreducible uncertainty) is the inherent stochastic variability in the data-generating process itself.

- **Coin flips**: Even after flipping a fair coin a million times, the outcome of the next flip remains uncertain. No amount of additional data can reduce this uncertainty.
- **Experimental measurement error**: Measuring the solubility of the same molecule ten times yields slightly different values each time, due to limitations of the measurement apparatus and subtle variations in experimental conditions.
- **Intrinsic ambiguity**: When a lesion boundary is unclear in a blurry medical image, even expert clinicians disagree on the interpretation.

Key property: **It cannot be reduced by collecting more data.**

### 3.2 Epistemic Uncertainty — A Lack of Knowledge

**Epistemic uncertainty** (model uncertainty, reducible uncertainty) arises from the model's insufficient learning from the available data.

- **Fairness of a die**: If you roll a die only three times and get [6, 6, 5], it is hard to tell whether the die is fair. But after 10,000 rolls, you can estimate the frequency of each face with precision. More data **reduces** this uncertainty.
- **Scaffolds absent from training data**: If a model was trained exclusively on benzene-ring-based molecules, its predictions for azaindole-based molecules should carry high epistemic uncertainty.
- **Choice of model architecture**: If a 3-layer network and a 10-layer network yield different predictions for the same data, this disagreement is itself a form of epistemic uncertainty.

Key property: **In principle, it can be reduced by collecting more data.**

### 3.3 Comparing the Two Types of Uncertainty

```
  ┌─────────────────────────────────────────────────────────────────┐
  │           Two Faces of Uncertainty                              │
  ├────────────────────────────┬────────────────────────────────────┤
  │    Aleatoric Uncertainty   │     Epistemic Uncertainty          │
  │    (Data uncertainty)      │     (Model uncertainty)            │
  ├────────────────────────────┼────────────────────────────────────┤
  │  Source: Noise in the data │  Source: Lack of knowledge/data    │
  │  itself                    │                                    │
  │                            │                                    │
  │  Outcome of a coin flip    │  Whether the coin is fair          │
  │  Variability in experiment │  Chemical space unseen in training │
  │  measurements              │                                    │
  │                            │                                    │
  │  More data → does not      │  More data → decreases             │
  │  decrease (irreducible)    │  (reducible)                       │
  │                            │                                    │
  │  Response: Provide         │  Response: Collect more data /     │
  │  prediction intervals      │  active learning                   │
  └────────────────────────────┴────────────────────────────────────┘
```

**Why does the distinction matter?** This is not merely an academic taxonomy — it has **direct practical implications for decision-making**:

- **When aleatoric uncertainty is high**: Collecting more data will not help. Instead, one must acknowledge the inherent uncertainty and incorporate it into decision-making (e.g., widening prediction intervals).
- **When epistemic uncertainty is high**: This represents an **opportunity**. Gathering additional data in that region can reduce the uncertainty. This is the core idea behind **active learning** — preferentially collecting data where the model is most uncertain, thereby learning more efficiently.

In the context of molecular property prediction:

- For properties with large experimental measurement error (e.g., certain ADMET properties), **aleatoric uncertainty** should be explicitly modeled to provide prediction intervals
- For chemical space not covered by training data (e.g., novel scaffolds), **epistemic uncertainty** should guide the prioritization of which molecules to synthesize next

**Disentangling and quantifying these two types of uncertainty** is a central capability of Bayesian Deep Learning, and Part 3 covers this in detail.

---

## 4. Why UQ Is Especially Urgent Now

The need for Uncertainty Quantification (UQ) has been recognized for some time. Yet three developments in the 2024–2026 AI landscape have made this problem **particularly pressing**.

### 4.1 The Foundation Model Era: The Black-Box Problem at Scale

Large language models (LLMs) such as GPT-4, Claude, and Gemini contain hundreds of billions of parameters, making their internal workings exceedingly difficult to understand.

- **The larger the model, the harder it is to identify where it fails**: failure modes of small models can be analyzed relatively easily, but the failures of LLMs are unpredictable and nearly impossible to debug
- **The hallucination problem**: LLMs confidently generating factually incorrect content is, at its core, a calibration failure — because the model has no way to express "I don't know"
- **Growing use of FMs in science**: Protein structure prediction (AlphaFold), molecular generation (diffusion models), reaction prediction — the adoption of large-scale models in the sciences is accelerating rapidly, and in this context prediction reliability is **directly tied to experimental planning and resource allocation**

Papamarkou et al. (2024) make this case explicitly in their ICML 2024 position paper: **"Bayesian Deep Learning is Needed in the Age of Large-Scale AI."** As models grow larger, uncertainty estimation for predictions ceases to be optional — it becomes essential.

### 4.2 The Rise of Autonomous Laboratories: UQ Decides the Next Experiment

**Autonomous laboratories (self-driving labs)** — systems where robots conduct experiments and AI designs the next ones — are rapidly becoming reality.

- **Bayesian optimization**: The most widely used experimental design framework, whose core component is precisely **the model's predictive uncertainty**
- **Acquisition functions**: When deciding "where to search next," the balance between predicted value (exploitation) and uncertainty (exploration) is critical
- **Requirements of closed-loop systems**: In autonomous experimental loops where humans do not intervene, the model's uncertainty serves as both a **safety mechanism** and an **engine for efficiency**

```
  ┌───────────┐      ┌───────────┐      ┌──────────────────┐
  │  AI Model  │──────│  Predict  │──────│  UQ: "How certain │
  │  (trained) │      │  + UQ     │      │  is this?"        │
  └───────────┘      └───────────┘      └──────┬───────────┘
                                                │
                        ┌───────────────────────┤
                        │                       │
                        ▼                       ▼
               ┌──────────────┐       ┌──────────────────┐
               │ Confident     │       │ Uncertain         │
               │ prediction:   │       │ prediction:       │
               │ use now       │       │ next experiment   │
               │ (exploit)     │       │ candidate         │
               └──────────────┘       │ (explore)         │
                                      └────────┬─────────┘
                                                │
                                                ▼
                                      ┌──────────────────┐
                                      │  Autonomous lab:  │
                                      │  run experiment   │
                                      └────────┬─────────┘
                                                │
                                                ▼
                                      ┌──────────────────┐
                                      │  Result → update  │
                                      │  model            │
                                      │  (closed loop)    │
                                      └──────────────────┘
```

Without UQ in this scenario, the model either repeatedly explores regions it already knows well (biased toward exploitation) or wastes resources on random exploration (losing directionality). **Good UQ is a prerequisite for efficient exploration.**

### 4.3 The Regulatory Perspective: Demands for Transparency and Reliability

As AI is deployed in high-stakes domains, regulatory requirements are becoming increasingly concrete.

- **FDA (United States)**: The 2021 AI/ML-based Software as a Medical Device (SaMD) Action Plan recommends not only reporting model performance but also **explicitly disclosing uncertainty and limitations**
- **EU AI Act (2024)**: Requires transparency and explainability for high-risk AI systems. Providing information about "where the model can be trusted and where it cannot" is a core component of compliance
- **EMA (European Medicines Agency)**: Increasingly expects **confidence intervals** or **prediction intervals** for model predictions in AI-driven drug development

Simply stating "this model has 95% average accuracy" is no longer sufficient. We are now in an era where **"how reliable is this particular prediction for this specific input"** must be provided at the level of individual predictions.

---

## 5. What This Series Covers — A Roadmap

Over eight installments, this series systematically covers the core theory, methods, and scientific applications of Bayesian Deep Learning and Uncertainty Quantification.

### Series Structure

```
  ┌──────────────────────────────────────────────────────────────────┐
  │                                                                  │
  │   Part 0: Beyond Predictions — Why Uncertainty Matters           │
  │   (Motivation & Problem Statement)  ◄── You are here            │
  │                                                                  │
  │         │                                                        │
  │         ▼                                                        │
  │   Part 1: The Language of Bayesian Inference                     │
  │   (Weight posterior, predictive distribution)                     │
  │                                                                  │
  │         │                                                        │
  │         ├──────────────────────┐                                 │
  │         ▼                      ▼                                 │
  │   Part 2: The Art of        Part 3: The Anatomy of               │
  │   Approximation             Uncertainty                          │
  │   (VI, Dropout, Ensemble,   (Aleatoric vs. Epistemic,           │
  │    SWAG, Laplace, HMC)      variance decomposition)             │
  │         │                      │                                 │
  │         ├──────────┬───────────┤                                 │
  │         ▼          ▼           ▼                                 │
  │   Part 4:      Part 5:     Part 6:                               │
  │   Calibration  Single-Pass UQ in Science                         │
  │   & Conformal  UQ (EDL,    (Bayesian GCN,                       │
  │   Prediction   SNGP, DUQ)  AlphaFold, NNP)                      │
  │         │          │           │                                 │
  │         └──────────┴───────────┘                                 │
  │                    │                                             │
  │                    ▼                                             │
  │   Part 7: The Future of UQ — Uncertainty in the                  │
  │   Foundation Model Era                                           │
  │   (LLM UQ, Bayesian LoRA, Autonomous Labs)                      │
  │                                                                  │
  └──────────────────────────────────────────────────────────────────┘
```

### Core Question of Each Installment

| Part | Title | Core Question |
|:----:|-------|---------------|
| **0** | Beyond Predictions — Why Uncertainty Matters | How much can we trust a deep learning model's predictions? |
| **1** | The Language of Bayesian Inference | What happens when we put probabilities on neural networks? |
| **2** | The Art of Approximation — From Variational to Ensemble | How do we handle an intractable posterior? |
| **3** | The Anatomy of Uncertainty | Aleatoric vs. Epistemic — why does the distinction matter? |
| **4** | Calibration and Conformal Prediction | Does "90% confidence" really mean 90% accuracy? |
| **5** | Single-Pass UQ | Can we quantify uncertainty without ensembles? |
| **6** | UQ in Science | How does UQ accelerate real scientific discovery? |
| **7** | The Future of UQ | How does uncertainty change as models grow larger? |

### Suggested Reading Order

- **Sequential reading is recommended**: Part 0 → 1 → 2/3 (can be read in parallel) → 4/5/6 (can be read in parallel) → 7
- **Parts 2 and 3 are independent**: approximation methods (Part 2) and uncertainty decomposition (Part 3) address different axes
- **Parts 4, 5, and 6 build on Parts 2 and 3**, but the interdependencies among the three are weak
- **Part 7 is a synthesis of the entire series** — it assumes familiarity with the preceding installments

---

## Closing: Toward Trustworthy AI

This series begins from a simple observation: **modern deep learning models make good predictions, but they do not know their own limitations.**

This is not merely the trite claim that "models are not perfect." The key insight is that these models are **designed in a way that prevents them from recognizing their own imperfections**. Softmax outputs are invariably self-assured, and point predictions conceal the very existence of uncertainty.

Bayesian Deep Learning offers a principled solution to this problem:

- **It provides not only predictions but also a measure of confidence** in those predictions
- **It disentangles aleatoric and epistemic uncertainty**, transforming them into actionable information for decision-making
- **It automatically expresses high uncertainty** in data-sparse regions

In the next installment (Part 1), we formalize these ideas mathematically — in the language of weight posteriors, predictive distributions, and marginal likelihoods.

---

## References

- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. "On Calibration of Modern Neural Networks." *ICML*, 2017.
- Ovadia, Y., Fertig, E., Ren, J., Nado, Z., Sculley, D., Nowozin, S., Dillon, J. V., Lakshminarayanan, B., & Snoek, J. "Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift." *NeurIPS*, 2019.
- Papamarkou, T., Skoularidou, M., Palla, K., Aitchison, L., Arbel, J., Dunson, D., Filippone, M., Fortuin, V., Hennig, P., Hernandez-Lobato, J. M., et al. "Position: Bayesian Deep Learning is Needed in the Age of Large-Scale AI." *ICML*, 2024.
- Ryu, S., Kwon, Y., & Kim, W. Y. "A Bayesian Graph Convolutional Network for Reliable Prediction of Molecular Properties with Uncertainty Quantification." *Chemical Science*, 10, 8438-8446, 2019.

---

*Next: [Part 1 — The Language of Bayesian Inference](./01_bayesian_inference_language.md)*
