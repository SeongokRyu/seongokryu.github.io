---
title: "Bayesian DL & UQ Part 4: Calibration and Conformal Prediction"
date: 2026-04-09 14:20:00 +0900
categories: [AI & ML, ML Theory]
tags: [bayesian-deep-learning, calibration, conformal-prediction, ece, temperature-scaling]
math: true
---

**Bayesian Deep Learning and Uncertainty Quantification — A Journey Toward Trustworthy AI**

*This is Part 4 of an 8-part series.*

- **Part 0:** [Beyond Predictions — Why Uncertainty Matters](/posts/bayesian-dl-uq-part0-why-uncertainty/)
- **Part 1:** [The Language of Bayesian Inference](/posts/bayesian-dl-uq-part1-bayesian-language/)
- **Part 2:** [The Art of Approximation — From Variational to Ensemble](/posts/bayesian-dl-uq-part2-art-of-approximation/)
- **Part 3:** [The Anatomy of Uncertainty — Aleatoric vs. Epistemic](/posts/bayesian-dl-uq-part3-anatomy-of-uncertainty/)
- **Part 4:** Calibration and Conformal Prediction (this post)
- **Part 5:** [Single-Pass UQ — Evidential Deep Learning and Distance-Aware Methods](/posts/bayesian-dl-uq-part5-single-pass-uq/)
- **Part 6:** [UQ in Science — Molecules, Proteins, and Materials](/posts/bayesian-dl-uq-part6-uq-in-science/)
- **Part 7:** [The Future of UQ — Uncertainty in the Age of Foundation Models](/posts/bayesian-dl-uq-part7-future-foundation-models/)

---

## Hook: What Does "90% Confident" Actually Mean?

Your model claims to be "90% confident" about a particular molecule's activity. How should you interpret that number?

Intuitively, it should mean that **when the model makes 100 predictions at 90% confidence, roughly 90 of them should be correct**. This is the essence of **calibration**. Yet real-world deep learning models systematically violate this expectation.

- **ResNet-110** (CIFAR-100): Among predictions where the model shows confidence above 90%, the **actual accuracy is only about 72%** (Guo et al., 2017)
- In drug discovery, compounds recommended with "high confidence" sometimes have a real hit rate **no better than random selection**
- In medical imaging, the consequences of a misdiagnosis made with 99% confidence

In Parts 2 and 3, we learned how to approximate the posterior and decompose uncertainty. But now it is time to confront the most uncomfortable truth of this series: **having an uncertainty estimate is an entirely different matter from having a trustworthy one.** Poorly calibrated uncertainty can be more dangerous than having no uncertainty at all — because it creates a false sense of security.

This post addresses two core topics:

- **Calibration**: How to measure and correct whether a model's confidence aligns with its actual accuracy
- **Conformal Prediction**: A groundbreaking framework that constructs statistically valid prediction sets **without any assumptions** about the model or data distribution

---

## 1. What Is Calibration?

### 1.1 Definition: Agreement Between Confidence and Accuracy

The mathematical definition of **calibration** is remarkably simple:

$$P(\hat{Y} = Y \mid \hat{P} = p) = p, \quad \forall\, p \in [0, 1]$$

- $\hat{Y}$: the model's predicted class
- $Y$: the true class
- $\hat{P}$: the confidence assigned by the model (maximum softmax output)

In plain terms:

- Collect all predictions where the model is **70% confident** — among those, **70% should be correct**
- Collect all predictions where the model is **95% confident** — among those, **95% should be correct**
- When this holds at every confidence level, the model is **perfectly calibrated**

This property is defined identically for **regression**. If the model assigns 95% probability to the prediction interval $[\hat{y} - 2\sigma, \hat{y} + 2\sigma]$, then the true value should fall within that interval approximately 95% of the time.

### 1.2 How to Read a Reliability Diagram

The most widely used visual diagnostic for calibration is the **reliability diagram**:

```
  Accuracy
  1.0 |                                          o (perfect)
      |                                       o
      |                                    o
  0.8 |                              . o
      |                           o
      |                     .  o
  0.6 |                  o
      |               o
      |         .  o
  0.4 |      o
      |   o
      |o
  0.2 |
      |
      |
  0.0 +----+----+----+----+----+----+----+----+----+----+
      0.0  0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1.0
                              Confidence

      Legend:  o = perfect calibration (diagonal)
               . = typical modern deep network (below diagonal = overconfident)
```

**How to interpret it**:

- **On the diagonal (o)**: perfectly calibrated model. Confidence = Accuracy
- **Below the diagonal (.)**: **overconfident** model. Actual accuracy is lower than the model's confidence
- **Above the diagonal**: **underconfident** model. The model is more accurate than it claims to be

**Modern deep learning models almost universally fall below the diagonal** — that is, they are systematically overconfident.

### 1.3 Why Modern Deep Learning Is Overconfident

The key finding of Guo et al. (2017) is this: **Model accuracy has improved dramatically over recent years, but calibration has actually gotten worse.** The causes of this paradox include:

- **Model depth**: As networks grow deeper, logit magnitudes tend to increase. Since softmax responds exponentially to differences in logits, even a modest increase in logit values can push confidence to extremes
- **Batch Normalization**: Stabilizes training but systematically alters logit scales during internal covariate shift correction, with unpredictable effects on calibration
- **Weight decay (L2 regularization)**: Prevents overfitting and improves accuracy, but does not necessarily improve calibration in terms of NLL. The way weight decay reshapes the loss landscape has counter-intuitive effects on calibration
- **Side effects of NLL training**: Training with cross-entropy loss minimizes NLL, but this can induce **overfitted confidence**. Training NLL keeps decreasing while validation calibration deteriorates

**Key insight**: Accuracy and calibration are **independent properties**. High accuracy does not guarantee calibration, and techniques that boost accuracy can actually degrade calibration. This is the fundamental reason why calibration must be treated as a separate problem.

---

## 2. Calibration Metrics

### 2.1 Expected Calibration Error (ECE)

The most widely used calibration metric. It partitions all predictions into $M$ bins by confidence, then computes the weighted average of the gap between accuracy and confidence in each bin:

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

- $B_m$: the set of predictions falling in the $m$-th bin
- $|B_m|$: the number of samples in that bin
- $N$: total number of samples
- $\text{acc}(B_m)$: accuracy within the bin
- $\text{conf}(B_m)$: average confidence within the bin

**Strengths**:

- Intuitive interpretation: "On average, how much do confidence and accuracy disagree?"
- Simple to compute and directly corresponds to the reliability diagram
- Summarizes calibration quality in a single number

**Pitfalls**:

- **Sensitive to bin count**: ECE values can differ substantially between $M=10$ and $M=100$. The choice between equal-width bins and equal-mass bins also affects the result
- **Cancellation effect**: If the model is overconfident in some bins and underconfident in others, these errors can cancel out, yielding a low ECE despite poor calibration overall
- **Maximum Calibration Error (MCE)**: An alternative that focuses on the worst-case bin. In safety-critical domains, MCE may be more appropriate than ECE

### 2.2 Negative Log-Likelihood (NLL)

$$\text{NLL} = -\frac{1}{N} \sum_{i=1}^{N} \log \hat{p}(y_i \mid \mathbf{x}_i)$$

- **Proper scoring rule**: Minimizing NLL theoretically induces calibration automatically
- If the model learns the true conditional distribution exactly, NLL is minimized and calibration is achieved

**However, practical limitations arise**:

- **Extreme sensitivity to outliers**: If the model assigns $\hat{p} = 0.001$ to a sample that turns out to be correct, $-\log(0.001) \approx 6.9$ has a disproportionate impact on NLL. A single outlier can dominate the entire metric
- **Does not measure calibration alone**: NLL captures calibration + sharpness simultaneously. A sharp and calibrated model achieves the lowest NLL, but disentangling the two properties is difficult

### 2.3 Brier Score

$$\text{BS} = \frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} (\hat{p}_{i,k} - y_{i,k})^2$$

where $y_{i,k}$ is the one-hot encoded true label.

**The Brier Score's key strength — decomposability** (Murphy, 1973):

$$\text{BS} = \underbrace{\text{Reliability}}_{\text{calibration error}} - \underbrace{\text{Resolution}}_{\text{discriminative ability}} + \underbrace{\text{Uncertainty}}_{\text{inherent data uncertainty}}$$

- **Reliability**: The discrepancy between confidence and accuracy at each confidence level — lower is better (= well calibrated)
- **Resolution**: The degree to which the model assigns different confidence levels to different samples — higher is better (= useful discrimination)
- **Uncertainty**: The intrinsic uncertainty of the data — beyond the model's control

This decomposition is powerful because it allows us to **diagnose calibration and usefulness separately**.

### 2.4 An Important Caveat: ECE = 0 Does Not Mean a Good Model

Consider the following counterexample:

- **Dataset**: 90% class A, 10% class B (extreme class imbalance)
- **Model strategy**: Always output "class A, confidence 90%" for every input
- **Result**: Every prediction has 90% confidence, and the actual accuracy is also 90% → **ECE = 0**

This model is **perfectly calibrated** yet **completely useless**. It cannot distinguish class B at all. Through the Brier Score decomposition, Reliability = 0 (calibrated), but Resolution is also 0 (no discriminative ability whatsoever).

**Lesson**: Calibration is a **necessary condition, not a sufficient one**. Evaluating a model by calibration metrics alone is inadequate — accuracy, resolution, and calibration must be assessed together. The Brier Score decomposition makes this possible.

---

## 3. Post-hoc Calibration Methods

These methods **retroactively correct the outputs of an already-trained model** without modifying the training process itself. They are surprisingly simple yet effective.

### 3.1 Temperature Scaling — The Power of a Single Parameter

**Core idea**: Divide the logits by a temperature $T$ before feeding them into the softmax:

$$\hat{q}_i = \text{softmax}(z_i / T)$$

- $T > 1$: **softens** the softmax output (lowers confidence) → corrects overconfident models
- $T < 1$: **sharpens** the softmax output (raises confidence) → corrects underconfident models
- $T = 1$: equivalent to the original model

**Training procedure**: Find the $T$ that minimizes NLL on the validation set. Since this is a one-dimensional optimization problem, a single call to scipy's minimize suffices.

**Why does this work so well** (key finding of Guo et al., 2017):

- A **single parameter** $T$ is enough to effectively calibrate most modern networks
- More complex methods (bin-wise scaling, matrix scaling) show little additional benefit
- The reason: miscalibration in modern networks primarily stems from the **overall scale of the logits**. The relative ordering among logits is generally correct, but their absolute magnitudes are too large

**An important property**: Temperature scaling **does not change the predicted class**. Since $\arg\max$ softmax($z/T$) = $\arg\max$ softmax($z$), accuracy is preserved while only calibration improves.

### 3.2 Platt Scaling

Originally proposed for producing probabilistic outputs from SVMs (Platt, 1999), adapted here for neural networks:

$$\hat{q} = \sigma(az + b)$$

- $a$, $b$: two parameters learned on the validation set
- Primarily used for binary classification
- Multi-class extension: class-specific $a_k, b_k$ → $2K$ parameters

Temperature scaling can be viewed as a special case of Platt scaling ($a = 1/T$, $b = 0$, with the same $a$ for all classes). Guo et al. (2017) showed that the additional degrees of freedom tend to cause overfitting in most cases, actually degrading performance.

### 3.3 Isotonic Regression — A Nonparametric Approach

- Instead of assuming a parametric form, learn a **monotonically non-decreasing function** $f$ to recalibrate confidence:

$$\hat{q} = f(\hat{p}), \quad f \text{ is isotonic (non-decreasing)}$$

- Efficiently solved by the **Pool Adjacent Violators (PAV) algorithm**
- Strength: makes no functional form assumption
- Weakness: prone to overfitting when the validation set is small. Not consistently better than temperature scaling

### 3.4 Limitations of Post-hoc Calibration

```
  Before Temperature Scaling        After Temperature Scaling

  Accuracy                           Accuracy
  1.0 |                    .         1.0 |                    o
      |                 .                |                 o
      |              .                   |              o
  0.8 |           .                  0.8 |           o
      |        .                         |        o
      |     .                            |     o
  0.6 |  .                           0.6 |  o
      |.                                 |o
      |                                  |
  0.4 |                              0.4 |
      +----+----+----+----+----+         +----+----+----+----+----+
      0.0  0.2  0.4  0.6  0.8  1.0      0.0  0.2  0.4  0.6  0.8  1.0
             Confidence                          Confidence

  (overconfident: curve below         (calibrated: curve matches
   the diagonal)                       the diagonal)
```

The diagram above illustrates the before-and-after of temperature scaling. A single parameter can produce this dramatic improvement, but fundamental limitations remain:

- **Dependence on validation set representativeness**: If the calibration set's distribution differs from the test set, the learned $T$ becomes meaningless. This is fatal under **distribution shift**
- **Global rather than instance-level correction**: Temperature scaling applies the same $T$ to every input. It cannot handle cases where high-confidence and low-confidence predictions require different corrections
- **Non-trivial extension to regression**: Softmax calibration for classification does not directly translate to prediction interval calibration for regression
- **Does not address the model's own limitations**: If the model has learned the wrong features, post-hoc calibration merely adjusts surface-level numbers without solving the underlying problem

These limitations naturally lead to the question: **Is there a method that provides statistical guarantees without depending on the model's form or the data distribution?**

---

## 4. Conformal Prediction — Distribution-Free Guarantees

### 4.1 A Revolutionary Idea

Conformal prediction is a framework systematized by **Vovk, Gammerman & Shafer (2005)** that has seen a surge of interest in recent years. Its core idea is as follows:

- Instead of a single point prediction, construct a **prediction set** $C(\mathbf{x})$
- Provide a **finite-sample guarantee** on the probability that this prediction set contains the true label
- This guarantee holds **for any model, under any data distribution**

$$P(Y_{\text{test}} \in C(X_{\text{test}})) \geq 1 - \alpha$$

- This is not an asymptotic guarantee but a **finite-sample** one
- It does not matter whether the model is a random forest or a deep network
- It does not matter whether the data is Gaussian or follows any other distribution

The only assumption is **exchangeability**: the calibration data and the test data must be exchangeable. This is automatically satisfied under i.i.d. sampling.

### 4.2 Split Conformal Algorithm

The most practical and widely used form is **split conformal prediction**:

```
  Split Conformal Prediction Pipeline
  ====================================

  [Training Data] ──────> [Train Model f]
                                │
                                ▼
  [Calibration Data]      [Compute nonconformity scores]
  (X_cal, Y_cal)          s_i = score(X_i, Y_i, f)
       │                        │
       │                        ▼
       │                  [Sort scores: s_(1) <= ... <= s_(n)]
       │                        │
       │                        ▼
       │                  [Find quantile q_hat]
       │                  q_hat = s_( ceil((n+1)(1-alpha)) )
       │                        │
       └───────────────────────>│
                                ▼
  [New test input X_new] ──> [Prediction set C(X_new)]
                              C(X_new) = { y : score(X_new, y, f) <= q_hat }
```

**Step-by-step explanation**:

1. **Data splitting**: Separate the data into a training set and a calibration set. The model is trained only on the training set
2. **Nonconformity score computation**: For each $(X_i, Y_i)$ in the calibration set, compute a score $s_i$ that measures "how unusual this (input, output) pair is"
3. **Quantile computation**: Sort all scores and find the $\lceil (n+1)(1-\alpha) \rceil / n$ quantile ($\hat{q}$)
4. **Prediction set construction**: For a new input $X_{\text{new}}$, include in the prediction set every $y$ whose score is at most $\hat{q}$

**Intuition behind the coverage guarantee**:

- Under exchangeability, the score of a new test point is **uniformly distributed** among all possible ranks relative to the calibration scores (every rank is equally likely)
- Therefore, the probability that the test score falls at or below the $(1-\alpha)$ quantile is exactly $\geq 1-\alpha$
- This argument depends neither on the form of the model $f$ nor on the data distribution $P_{X,Y}$

### 4.3 Regression: Conformalized Quantile Regression (CQR)

For regression, the most natural nonconformity score is the **absolute residual**:

$$s_i = |Y_i - \hat{Y}_i|$$

In this case, the prediction interval is:

$$C(X_{\text{new}}) = [\hat{Y}_{\text{new}} - \hat{q},\; \hat{Y}_{\text{new}} + \hat{q}]$$

The limitation of this approach is that it produces **intervals of uniform width for all inputs**. Some inputs are inherently easy to predict (a narrow interval would suffice) while others are difficult (a wide interval is needed), but this method cannot distinguish between them.

**Conformalized Quantile Regression (CQR)** (Romano, Patterson & Candes, 2019) solves this problem elegantly:

- **Step 1**: Train a quantile regression model as the base model. Estimate the conditional quantile functions:

$$\hat{q}_{\alpha/2}(x) \quad \text{and} \quad \hat{q}_{1-\alpha/2}(x)$$
- **Step 2**: Define the nonconformity score as:

$$s_i = \max\left(\hat{q}_{\alpha/2}(X_i) - Y_i,\; Y_i - \hat{q}_{1-\alpha/2}(X_i)\right)$$

- **Step 3**: Compute $\hat{q}$ (the score quantile) on the calibration set and construct the prediction interval:

$$C(X_{\text{new}}) = [\hat{q}_{\alpha/2}(X_{\text{new}}) - \hat{q},\; \hat{q}_{1-\alpha/2}(X_{\text{new}}) + \hat{q}]$$

**Strengths of CQR**:

- When the base model's quantile estimates are good, intervals have **adaptive widths that vary with the input**
- Even when the base model's quantile estimates are wrong, the conformal correction ($\hat{q}$) still guarantees coverage
- Best case: narrow and accurate intervals. Worst case: wide but still valid intervals

### 4.4 Classification: Adaptive Prediction Sets (APS)

In classification, conformal prediction outputs a **prediction set** — a set of possible classes.

**Naive approach**: Include every class whose softmax output exceeds a threshold

$$C_{\text{naive}}(X) = \{k : \hat{p}_k(X) \geq \tau\}$$

However, this requires the model's softmax to be well calibrated, and the threshold choice lacks solid theoretical grounding.

**Adaptive Prediction Sets (APS)** (Romano, Sesia & Candes, 2020):

- Sort the softmax probabilities in descending order
- Use as the score the cumulative probability up to the point where the true class is first included:

$$s(X, Y) = \sum_{j=1}^{k} \hat{p}_{\pi(j)}(X), \quad \text{where } \pi(k) = Y$$

- For confident (easy) predictions, the prediction set is small (often a single class)
- For uncertain (hard) predictions, the prediction set grows larger

**Intuition behind APS**: Saying "this image could be a cat or a tiger" honestly is often more useful for decision-making than saying "this is a cat (confidence 51%)."

---

## 5. Limitations and Extensions of Conformal Prediction

### 5.1 Marginal Coverage != Conditional Coverage

The coverage guarantee of conformal prediction is **marginal**:

$$P(Y \in C(X)) \geq 1 - \alpha$$

This means that **on average**, a fraction $1-\alpha$ of test points will be included in the prediction set. However, coverage is not guaranteed for specific subgroups or specific regions of the input space.

**A concrete problem scenario**:

- In molecular property prediction, even if overall 90% coverage is achieved:
  - **Common scaffolds** (abundant in training data): 99% coverage (overly conservative → wide intervals)
  - **Rare scaffolds** (scarce in training data): 60% coverage (**insufficient coverage** → decision errors)
- These two effects cancel out so that marginal coverage reaches 90%, but at the places that actually matter (rare scaffolds) the guarantee breaks down

What we ideally want is **conditional coverage**:

$$P(Y \in C(X) \mid X = x) \geq 1 - \alpha, \quad \forall x$$

However, distribution-free conditional coverage is theoretically impossible to achieve (Vovk, 2012; Lei & Wasserman, 2014). This is a fundamental limitation of conformal prediction.

### 5.2 Violation of the Exchangeability Assumption

Practical scenarios where conformal prediction's sole assumption of **exchangeability** is violated:

- **Time-series data**: Stock prices, weather, reaction kinetics — past and future are not exchangeable
- **Active learning**: Selecting the next data point based on the model's predictions introduces selection bias that violates exchangeability
- **Distribution shift**: Data distributions that change over time (concept drift)
- **Drug discovery pipelines**: During hit-to-lead optimization, the chemical space progressively narrows → early calibration molecules and later test molecules are not exchangeable

Such violations invalidate the coverage guarantee. A promised 90% coverage may in practice deliver only 70%.

### 5.3 Extensions: Pushing Beyond the Limits

**Group-conditional conformal prediction** (Barber et al., 2023):

- Partition the data into meaningful subgroups and perform conformal prediction separately within each group
- If exchangeability holds within each group, group-level coverage is guaranteed
- Limitation: if groups are too fine-grained, each group's calibration set becomes small, leading to wide intervals

**Weighted conformal prediction** (Tibshirani et al., 2019):

- Addresses covariate shift by weighting calibration points with the likelihood ratio:

$$w(x) = p_{\text{test}}(x)/p_{\text{cal}}(x)$$
- Constructs prediction sets using weighted quantiles
- Limitation: estimating the likelihood ratio itself is challenging

**Adaptive Conformal Inference (ACI)** (Gibbs & Candes, 2021):

- For settings where exchangeability is violated, such as time series, dynamically adjusts $\alpha$
- Feeds back past coverage errors to update $\alpha$ — an online learning approach
- Converges to the desired coverage rate in the long run

### 5.4 Conformal Prediction in Cheminformatics

Conformal prediction finds a particularly natural application in cheminformatics:

- **Prediction intervals for QSAR models** (Svensson et al., ACS Omega, 2024): Applying conformal prediction to drug activity prediction provides, for each molecule, "the 90% prediction interval for this prediction is [a, b]." This enables **decision support in drug development** that was impossible with point predictions alone
- **Defining the applicability domain**: Chemists have traditionally defined "the chemical space where this model is reliable" in qualitative terms. Conformal prediction **quantifies this through prediction set size**. Molecules with excessively wide prediction intervals are deemed outside the model's applicability domain
- **Controlling false positives in virtual screening** (ScienceDirect, 2025): Filtering out molecules whose conformal prediction sets include both "active" and "inactive" labels reduces wasted resources on synthesis experiments

**Author's perspective**: In our 2019 Bayesian GCN paper, we took the approach of filtering unreliable predictions using epistemic uncertainty (Ryu, Kwon & Kim, 2019). Conformal prediction is complementary to this — while Bayesian methods excel at **decomposing the sources** of uncertainty, conformal prediction excels at providing **statistical guarantees** without distributional assumptions. Combining the two yields a more powerful framework.

---

## 6. Practical Guide: When to Use What

### 6.1 Decision Tree

```
  What is your situation?
  =======================

  Q1: Is it sufficient to just recalibrate
      the confidence of an already-trained model?
         |
    YES -+--- NO
    |         |
    v         v
  [Temperature    Q2: Do you need distribution-free
   Scaling]        coverage guarantees?
  (simple,              |
   effective,      YES -+--- NO
   just 1 param)   |         |
                   v         v
         Q3: Do you need   [Use Bayesian methods
          adaptive intervals for uncertainty
          per input?         estimation +
              |              reliability diagrams
         YES -+--- NO       for diagnostics]
         |         |
         v         v
  [Conformalized   [Split Conformal
   Quantile         Prediction]
   Regression]    (simplest conformal
  (adaptive width,  method)
   coverage guarantee)
```

### 6.2 Situation-Specific Recommendations

**Situation 1: You have a well-trained classification model and only need to recalibrate its confidence**

- **Recommendation**: Temperature Scaling
- **Rationale**: Extremely simple to implement (a single line of scipy.optimize), does not alter accuracy, and is sufficient in most cases
- **Caveat**: The validation set must be similar in distribution to the test set

**Situation 2: You need to present a "90% confidence interval for this prediction" to regulators or decision-makers**

- **Recommendation**: Conformal Prediction (CQR for regression, APS for classification)
- **Rationale**: The distribution-free coverage guarantee minimizes assumptions, making it easy to explain to external stakeholders
- **Caveat**: Coverage is marginal, so guarantees for specific subgroups require separate verification

**Situation 3: You need uncertainty-based filtering in large-scale virtual screening**

- **Recommendation**: Combine conformal prediction with a well-calibrated base model
- **Specific strategy**:
  - Estimate uncertainty with Deep Ensembles or MC Dropout as the base model
  - Calibrate the base model with temperature scaling
  - Apply conformal prediction to construct a prediction set for each molecule
  - Prioritize molecules whose prediction set contains a single class (high confidence)
  - Route molecules with multi-class prediction sets to further experimental testing

**Situation 4: Time series, active learning, or other settings where exchangeability is violated**

- **Recommendation**: Adaptive Conformal Inference (ACI)
- **Rationale**: Online coverage control that adapts to distribution shift
- **Caveat**: The guarantee weakens to a long-run average. Coverage at any individual time step is not guaranteed

### 6.3 Principles of Combination

The most powerful approach is to **combine calibration and conformal prediction**. These two methods are not competitors but complements:

- **Calibration (Temperature Scaling, etc.)**: Corrects the base model's softmax outputs, making the nonconformity scores used in conformal prediction more meaningful
- **Conformal Prediction**: Acts as a safety net that guarantees coverage even when calibration is imperfect

Applying conformal prediction to a well-calibrated model yields:

- **Smaller prediction sets** (at the same coverage level): Well-calibrated scores are more informative
- **Closer approximation to conditional coverage**: Perfect conditional coverage is impossible, but the better the base model, the smaller the gap between marginal and conditional coverage

---

## Closing Remarks: Toward Honest Uncertainty

Synthesizing everything covered in this post, a complete pipeline for uncertainty quantification comes into view:

1. **Uncertainty estimation** (Part 2): Obtain predictive uncertainty through posterior approximation
2. **Uncertainty decomposition** (Part 3): Separate into aleatoric and epistemic components to diagnose the source
3. **Calibration verification** (this post): Check whether the uncertainty matches reality using reliability diagrams, ECE, and the Brier Score
4. **Post-hoc correction**: If discrepancies exist, apply temperature scaling or similar methods
5. **Statistical guarantees**: Use conformal prediction to provide distribution-free coverage guarantees

Within this pipeline, the role of this post is **verification and assurance**. If Parts 2 and 3 were about "producing" uncertainty, Part 4 is about confirming that the uncertainty is "honest" and "attaching guarantees" to it.

Let me close by emphasizing three key takeaways:

- **Calibration is a mandatory checkpoint, but it is not sufficient on its own.** Remember that a perfectly useless model with ECE = 0 can exist. A perspective that examines calibration, resolution, and uncertainty together — as in the Brier Score's three-component decomposition — is essential.

- **Recognize both the power and the limits of temperature scaling.** A single parameter can achieve remarkable calibration improvements, but this depends on the representativeness of the validation set and is vulnerable to distribution shift. When exploring novel chemical spaces in scientific research, there is no guarantee that a $T$ learned from existing data will remain valid.

- **Conformal prediction is an indispensable member of the modern UQ toolkit.** This framework, which guarantees coverage without distributional assumptions, does not compete with Bayesian methods — it **complements** them. If Bayesian methods are the scalpel that dissects the sources of uncertainty, conformal prediction is the seal that stamps a statistical certificate onto that uncertainty.

In the next installment, Part 5, we address another practical question: **Can uncertainty be quantified in a single forward pass, without Deep Ensembles or MC Dropout?** We will examine the theory and practice of single-pass UQ methods such as Evidential Deep Learning, SNGP, and DUQ.

---

## References

### Calibration
1. Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. Q. (2017). "On Calibration of Modern Neural Networks." *ICML*.
2. Platt, J. (1999). "Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods." *Advances in Large Margin Classifiers*.
3. Naeini, M. P., Cooper, G. & Hauskrecht, M. (2015). "Obtaining Well Calibrated Probabilities Using Bayesian Binning into Quantiles." *AAAI*.
4. Nixon, J. et al. (2019). "Measuring Calibration in Deep Learning." *CVPR Workshops*.
5. Murphy, A. H. (1973). "A New Vector Partition of the Probability Score." *Journal of Applied Meteorology*, 12(4), 595-600.

### Conformal Prediction — Foundations
6. Vovk, V., Gammerman, A. & Shafer, G. (2005). *Algorithmic Learning in a Random World.* Springer.
7. Angelopoulos, A. N. & Bates, S. (2023). "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification." *Foundations and Trends in Machine Learning*.
8. Romano, Y., Patterson, E. & Candes, E. (2019). "Conformalized Quantile Regression." *NeurIPS*.
9. Romano, Y., Sesia, M. & Candes, E. (2020). "Classification with Valid and Adaptive Coverage." *NeurIPS*.

### Conformal Prediction — Extensions
10. Barber, R. F., Candes, E. J., Ramdas, A. & Tibshirani, R. J. (2023). "Conformal Prediction Beyond Exchangeability." *Annals of Statistics*.
11. Tibshirani, R. J., Barber, R. F., Candes, E. J. & Ramdas, A. (2019). "Conformal Prediction Under Covariate Shift." *NeurIPS*.
12. Gibbs, I. & Candes, E. (2021). "Adaptive Conformal Inference Under Distribution Shift." *NeurIPS*.

### Conformal Prediction in Chemistry
13. Svensson, F. et al. (2024). "Conformal Prediction for QSAR." *ACS Omega*.
14. Conformal prediction for virtual screening and molecular property prediction (2025). *ScienceDirect*.

### Scientific Applications
15. Ryu, S., Kwon, Y. & Kim, W. Y. (2019). "A Bayesian Graph Convolutional Network for Reliable Prediction of Molecular Properties with Uncertainty Quantification." *Chemical Science*, 10, 8438-8446.

### Background
16. Lei, J. & Wasserman, L. (2014). "Distribution-Free Prediction Bands for Non-Parametric Regression." *JRSSB*.
17. Vovk, V. (2012). "Conditional Validity of Inductive Conformal Predictors." *AISTATS*.
