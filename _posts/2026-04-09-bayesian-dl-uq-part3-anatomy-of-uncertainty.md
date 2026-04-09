---
title: "Bayesian DL & UQ Part 3: The Anatomy of Uncertainty — Aleatoric vs. Epistemic"
date: 2026-04-09 14:15:00 +0900
categories: [AI & ML, ML Theory]
tags: [bayesian-deep-learning, aleatoric-uncertainty, epistemic-uncertainty, variance-decomposition, ood-detection]
math: true
---

**Bayesian Deep Learning and Uncertainty Quantification — A Journey Toward Trustworthy AI**

*This is Part 3 of an 8-part series.*

- **Part 0:** [Beyond Predictions — Why Uncertainty Matters](/posts/bayesian-dl-uq-part0-why-uncertainty/)
- **Part 1:** [The Language of Bayesian Inference](/posts/bayesian-dl-uq-part1-bayesian-language/)
- **Part 2:** [The Art of Approximation — From Variational to Ensemble](/posts/bayesian-dl-uq-part2-art-of-approximation/)
- **Part 3:** The Anatomy of Uncertainty — Aleatoric vs. Epistemic (this post)
- **Part 4:** [Calibration and Conformal Prediction](/posts/bayesian-dl-uq-part4-calibration-conformal/)
- **Part 5:** [Single-Pass UQ — Evidential Deep Learning and Distance-Aware Methods](/posts/bayesian-dl-uq-part5-single-pass-uq/)
- **Part 6:** [UQ in Science — Molecules, Proteins, and Materials](/posts/bayesian-dl-uq-part6-uq-in-science/)
- **Part 7:** [The Future of UQ — Uncertainty in the Age of Foundation Models](/posts/bayesian-dl-uq-part7-future-foundation-models/)

---

## Hook: Same Uncertainty, Entirely Different Prescriptions

Imagine an AI team at a pharmaceutical company running a virtual screen for EGFR inhibitor candidates. The model reports the same predictive uncertainty, $\sigma = 1.5$ pIC50, for molecules A and B. On the surface the two look equally unreliable.

Decompose A's uncertainty, however, and it turns out to originate from **noise intrinsic to the data** — poor reproducibility of the assay. B's uncertainty, by contrast, arises because **the model has never seen this chemical structure before**.

The prescriptions are completely different:
- **Molecule A**: Repeating the experiment will not reduce the uncertainty — it is a fundamental limitation of the assay itself
- **Molecule B**: Measuring just a few more molecules from this scaffold family can reduce the uncertainty dramatically

This distinction is precisely that between **aleatoric uncertainty** (inherent in the data, irreducible) and **epistemic uncertainty** (stemming from lack of knowledge, reducible with more data). In this post we develop the mathematical foundation of this decomposition, show how to implement it, and explore the practical value it delivers.

---

## 1. The Law of Total Variance

### 1.1 Mathematical Derivation

The starting point for decomposing the variance of a Bayesian predictive distribution is the **law of total variance**. Given model parameters $\theta$ and an input $\mathbf{x}$, the total variance of the prediction $y$ decomposes as follows:

$$\text{Var}[y \mid \mathbf{x}] = \underbrace{\mathbb{E}_{\theta}[\text{Var}[y \mid \mathbf{x}, \theta]]}_{\text{Aleatoric Uncertainty}} + \underbrace{\text{Var}_{\theta}[\mathbb{E}[y \mid \mathbf{x}, \theta]]}_{\text{Epistemic Uncertainty}}$$

Let us dissect each term precisely.

**First term: $\mathbb{E}_{\theta}[\text{Var}[y \mid \mathbf{x}, \theta]]$ — Aleatoric Uncertainty**
- Compute the predictive variance $\text{Var}[y \mid \mathbf{x}, \theta]$ under a fixed set of parameters $\theta$, then take the **expectation** over all plausible $\theta$
- Interpretation: the noise that **remains even if we knew the model parameters exactly** — the fundamental randomness inherent in the data-generating process
- Cannot be reduced no matter how much data we collect (expected value of the irreducible noise)

**Second term: $\text{Var}_{\theta}[\mathbb{E}[y \mid \mathbf{x}, \theta]]$ — Epistemic Uncertainty**
- Measures how much the predictive **means** $\mathbb{E}[y \mid \mathbf{x}, \theta]$ **vary** across different $\theta$
- Interpretation: the degree to which different models (different weight configurations) make **different predictions** for the same input
- As data accumulates the posterior narrows, and this term **converges to 0**

### 1.2 Proof Sketch

The proof is concise. Expanding $\text{Var}[y \mid \mathbf{x}]$:

$$\text{Var}[y \mid \mathbf{x}] = \mathbb{E}[y^2 \mid \mathbf{x}] - (\mathbb{E}[y \mid \mathbf{x}])^2$$

Apply the tower property (iterated expectation):

$$\mathbb{E}[y^2 \mid \mathbf{x}] = \mathbb{E}_{\theta}[\mathbb{E}[y^2 \mid \mathbf{x}, \theta]]$$

Meanwhile, for any random variable $Z$ we have $\mathbb{E}[Z^2] = \text{Var}[Z] + (\mathbb{E}[Z])^2$, so:

$$\mathbb{E}_{\theta}[\mathbb{E}[y^2 \mid \mathbf{x}, \theta]] = \mathbb{E}_{\theta}[\text{Var}[y \mid \mathbf{x}, \theta] + (\mathbb{E}[y \mid \mathbf{x}, \theta])^2]$$

$$= \mathbb{E}_{\theta}[\text{Var}[y \mid \mathbf{x}, \theta]] + \mathbb{E}_{\theta}[(\mathbb{E}[y \mid \mathbf{x}, \theta])^2]$$

Similarly, $(\mathbb{E}[y \mid \mathbf{x}])^2 = (\mathbb{E}_{\theta}[\mathbb{E}[y \mid \mathbf{x}, \theta]])^2$. Subtracting:

$$\text{Var}[y \mid \mathbf{x}] = \mathbb{E}_{\theta}[\text{Var}[y \mid \mathbf{x}, \theta]] + \left(\mathbb{E}_{\theta}[(\mathbb{E}[y \mid \mathbf{x}, \theta])^2] - (\mathbb{E}_{\theta}[\mathbb{E}[y \mid \mathbf{x}, \theta]])^2\right)$$

The expression inside the parentheses is exactly $\text{Var}_{\theta}[\mathbb{E}[y \mid \mathbf{x}, \theta]]$. QED.

### 1.3 Visual Intuition

The following ASCII diagram illustrates the decomposition at a glance.

```
    Total Predictive Variance: Var[y|x]
    =============================================

    +-------------------------------------------+
    |                                           |
    |   Model 1 (theta_1):  mu_1 +/- sigma_1   |
    |   Model 2 (theta_2):  mu_2 +/- sigma_2   |
    |   Model 3 (theta_3):  mu_3 +/- sigma_3   |
    |       ...                                 |
    |   Model T (theta_T):  mu_T +/- sigma_T   |
    |                                           |
    +-------------------------------------------+
                    |
         +----------+----------+
         |                     |
         v                     v
    +-----------+        +-----------+
    | Aleatoric |        | Epistemic |
    |           |        |           |
    | Average   |        | Spread of |
    | of the    |        | the means |
    | variances |        |           |
    |           |        | Var over  |
    | E[sigma^2]|        | {mu_1,    |
    |           |        |  mu_2,    |
    |           |        |  ...,     |
    |           |        |  mu_T}    |
    +-----------+        +-----------+
    "Even if we         "Models
     knew theta          disagree
     perfectly,          about the
     noise remains"      prediction"
```

### 1.4 Monte Carlo Estimation in Practice

When we run $T$ stochastic forward passes with a deep ensemble or MC Dropout, each pass $t$ yields a predicted mean $\hat{\mu}_t(\mathbf{x})$ and a predicted variance $\hat{\sigma}_t^2(\mathbf{x})$. The decomposition is then estimated as:

$$\text{Aleatoric} \approx \frac{1}{T} \sum_{t=1}^{T} \hat{\sigma}_t^2(\mathbf{x})$$

$$\text{Epistemic} \approx \frac{1}{T} \sum_{t=1}^{T} (\hat{\mu}_t(\mathbf{x}) - \bar{\mu}(\mathbf{x}))^2, \quad \bar{\mu}(\mathbf{x}) = \frac{1}{T}\sum_{t=1}^{T}\hat{\mu}_t(\mathbf{x})$$

$$\text{Total} = \text{Aleatoric} + \text{Epistemic}$$

**Key point**: For this decomposition to work, the network must **output its own estimate of aleatoric uncertainty** — that is, it must predict not only the mean $\mu(\mathbf{x})$ but also the variance $\sigma^2(\mathbf{x})$, making it a **heteroscedastic model**. If the network only produces point estimates, the variance captured by MC Dropout or an ensemble reflects epistemic uncertainty alone, and separation from the aleatoric component is impossible.

---

## 2. Aleatoric Uncertainty — The Irreducible Kind

### 2.1 Fundamental Nature

**Aleatoric uncertainty** (from Latin *alea* = dice) is the randomness inherent in the data-generating process itself. Following the classical definition of Der Kiureghian & Ditlevsen (2009), it arises from unobservable variables or fundamental physical stochasticity.

Concrete examples:
- **Drug activity prediction**: Repeated measurements of the same molecule in the same assay yield different values — cell state variation, temperature fluctuations, operator differences
- **Autonomous driving**: LiDAR readings at the same location differ with weather, airborne particles, and surface reflections
- **Molecular simulation**: Thermodynamic fluctuations at finite temperature

**Key property: no matter how much data we collect, this uncertainty does not shrink.** The next face of a fair die cannot be predicted regardless of how many rolls we have observed.

### 2.2 Homoscedastic vs. Heteroscedastic Noise

There are two ways to model aleatoric uncertainty.

**Homoscedastic (constant variance)**:
- The noise variance is the same for every input $\mathbf{x}$: $\sigma^2 = \text{const}$
- A single learnable parameter ($\log \sigma^2$) is optimized
- Simple but often a poor reflection of reality

**Heteroscedastic (input-dependent variance)**:
- The noise variance depends on the input: $\sigma^2(\mathbf{x})$
- The network predicts two outputs simultaneously: $[\mu(\mathbf{x}),\, \log \sigma^2(\mathbf{x})]$
- Far more useful in practice — for instance, the experimental noise for certain molecular scaffolds can be substantially larger than for others

```
    Homoscedastic vs. Heteroscedastic

    Homoscedastic:                 Heteroscedastic:
    y                              y
    |    .  .                      |    .  .
    |  .. ... .                    |  .. ... .
    | ........... .                | ............
    |.................             |.......+++++++++++++
    |  .............. .            |  .....++++++++++++++++++
    |    ...........               |    .......++++++++++++++++
    |      .. ..                   |      .. ...++++++++++++++
    |        .                     |        .    ++++++++++
    +-------------------> x        +-------------------> x
    (noise band constant)          (noise band widens with x)
                                    . = data,  + = wider noise
```

### 2.3 Heteroscedastic Loss Function

The cornerstone of a heteroscedastic model is the **design of the loss function**. Assume the observations $y$ are generated under Gaussian noise:

$$y = \mu(\mathbf{x}) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2(\mathbf{x}))$$

Writing out the negative log-likelihood gives:

$$\mathcal{L} = \frac{1}{2\sigma^2(\mathbf{x})}(y - \mu(\mathbf{x}))^2 + \frac{1}{2}\log \sigma^2(\mathbf{x})$$

The two terms form a **natural balancing mechanism**:

- **First term** $\frac{1}{2\sigma^2(\mathbf{x})}(y - \mu(\mathbf{x}))^2$: **precision-weighted squared error**
  - If the model predicts a large $\sigma^2(\mathbf{x})$ (declares high uncertainty), the contribution of this term decreases
  - In other words, **"confessing ignorance" about a data point reduces the penalty for prediction error**
- **Second term** $\frac{1}{2}\log \sigma^2(\mathbf{x})$: **log-variance regularizer**
  - Prevents $\sigma^2(\mathbf{x})$ from being sent to infinity
  - The larger the declared uncertainty, the more this term **grows**, imposing a cost

The competition between these two terms is the key. In regions where the model can fit the data well, it keeps $\sigma^2(\mathbf{x})$ small to receive a precise learning signal from the first term. In regions where the data is inherently noisy, it increases $\sigma^2(\mathbf{x})$ to reduce the penalty from large residuals. The net effect is that the model **learns the intrinsic noise structure of the data**.

**Implementation notes**:
- The network should output $s(\mathbf{x}) = \log \sigma^2(\mathbf{x})$ rather than $\sigma^2(\mathbf{x})$ directly — this ensures numerical stability (positivity is guaranteed, and gradients are better behaved)
- In the loss, recover $\sigma^2(\mathbf{x}) = \exp(s(\mathbf{x}))$
- $s(\mathbf{x})$ can be unstable early in training, so appropriate initialization (e.g., $s = 0$, i.e., $\sigma^2 = 1$) is recommended

### 2.4 Case Study: Data Quality Diagnosis in the Harvard Clean Energy Project

An intriguing finding emerged from our Bayesian GCN research (Ryu, Kwon & Kim, *Chemical Science*, 2019). When predicting the power conversion efficiency (PCE) of organic solar-cell candidate molecules from the **Harvard Clean Energy Project (HCEP)** dataset, we observed **anomalously large aleatoric uncertainty for molecules labeled with PCE = 0**.

Interpretation of this phenomenon:
- A label of PCE = 0 may not actually mean "the efficiency of this molecule is exactly zero"
- Cases where DFT (density functional theory) calculations failed to converge, or where computational limitations led to **"unmeasurable" being recorded as 0**, are likely included
- In short, the labels themselves harbor **systematic error**

**Aleatoric uncertainty acted as a data-quality diagnostic tool.** The model learning that "the labels of these data points are inherently noisy" can be viewed as an automated mechanism for questioning the reliability of those labels.

Implications:
- **An aleatoric uncertainty map can serve as a quality map of the dataset**
- Flagging data points with high aleatoric uncertainty for re-verification can greatly improve the efficiency of dataset curation
- This approach is especially valuable for large-scale automated experimental or simulation datasets

---

## 3. Epistemic Uncertainty — The Knowledge Gap

### 3.1 Three Sources

**Epistemic uncertainty** (from Greek *episteme* = knowledge) arises from the model's incomplete knowledge. In principle it can be eliminated given infinite data. There are three major sources:

**1) Parameter uncertainty**
- Finite training data makes it impossible to **pin down the optimal weights exactly**
- From a Bayesian perspective: the posterior $p(\theta \mid \mathcal{D})$ is a distribution, not a point mass
- As data grows the posterior tightens, converging theoretically to zero as $N \to \infty$

**2) Model / architecture uncertainty**
- The possibility that **the wrong model structure was chosen**
- Example: using a linear model when the true relationship is nonlinear, or using too few message-passing steps in a GNN to capture long-range interactions
- This form of uncertainty is not captured by Bayesian inference within a single model class — it requires model selection or model averaging
- Leads to the problem of **model misspecification**, which we examine critically in Section 6

**3) Distributional shift / Out-of-distribution (OOD)**
- Situations where the model must make predictions for inputs outside the training distribution $p_{\text{train}}(\mathbf{x})$
- The most practically important source of epistemic uncertainty
- Example: in drug-candidate screening, molecules with chemical scaffolds absent from the training data

```
    Epistemic Uncertainty: High in Data-Sparse Regions

    Prediction
    |
    |         Data-rich region       Data-sparse region
    |         (low epistemic)        (HIGH epistemic)
    |
    |     ....****....               .............
    |   ..**        **..           ..               ..
    |  .*              *.        ..                   ..
    | .*    True f(x)   *.      .  ?? True f(x) ??     .
    | *       ----       *     .                        .
    | *      /    \      *    .    Models disagree!      .
    |  *    /      \    *    .   /   |    \    |    \     .
    |   *  / models \  *    .  /    |     \   |     \     .
    |    */  agree   \*    . /     |      \  |      \     .
    |     *          *    ./      |       \ |       \     .
    |      **      **    .       |        \|        \     .
    |        ******     .       |         X         \     .
    +----+--------+--------+--------+--------+--------+---> x
         |  training data  |        |  NO training data  |
         +--------+--------+        +--------+-----------+
    
    Narrow confidence band           Wide confidence band
    (models converge)                (models diverge)
```

### 3.2 Case Study: Epistemic Uncertainty in EGFR Virtual Screening

The most striking practical result from Ryu, Kwon & Kim (2019) was the use of epistemic uncertainty in **EGFR (Epidermal Growth Factor Receptor) virtual screening**.

Experimental setup:
- A Bayesian GCN predicts the activity of molecules against EGFR
- Roughly 20,000 molecules are ranked by predicted activity
- Two strategies are compared:
  - **Baseline**: select the top-100 molecules by predicted activity
  - **UQ-guided**: select molecules with high predicted activity **and epistemic uncertainty below a threshold**

Results:
- **Baseline top-100**: **29** true actives (hit rate 29%)
- **UQ-guided top-100**: **69** true actives (hit rate 69%)
- **138% improvement** — filtering by epistemic uncertainty alone more than doubled the hit rate

What this means:
- High predicted activity + high epistemic uncertainty = **the model is guessing optimistically about something it does not understand**
- Such molecules are likely false positives
- Epistemic uncertainty filtering implements the strategy **"trust only predictions where the model is confident"**, which has direct value for allocating expensive experimental resources efficiently

```
    EGFR Virtual Screening: UQ-Guided Selection

    Without UQ filter:          With UQ filter:
    
    Top-100 molecules           Top-100 molecules
    (by predicted activity)     (high activity + low epistemic UQ)
    
    +---+---+---+---+---+       +---+---+---+---+---+
    | A | . | . | A | . |       | A | A | . | A | A |
    | . | A | . | . | . |       | A | A | A | . | A |
    | . | . | A | . | A |       | A | . | A | A | . |
    | A | . | . | . | . |  -->  | . | A | A | A | A |
    | . | . | A | . | . |       | A | A | . | A | A |
    | A | . | . | A | . |       | A | . | A | A | A |
    | . | . | . | . | A |       | A | A | A | A | . |
    | . | A | . | . | . |       | A | A | A | . | A |
    +---+---+---+---+---+       +---+---+---+---+---+
    
    A = Active (true hit)       A = Active (true hit)
    . = Inactive (false pos)    . = Inactive (false pos)
    
    29/100 = 29% hit rate       69/100 = 69% hit rate
                                (+138% improvement)
```

### 3.3 Epistemic Uncertainty and Data Efficiency

The most important property of epistemic uncertainty is that **it can be reduced with data**. This property is the driving force behind **active learning**.

The basic active-learning loop:
1. Train the model on the current dataset
2. From the unlabeled pool, select the sample with the **highest epistemic uncertainty**
3. Acquire the label for that sample (run the experiment)
4. Add it to the dataset and return to step 1

Why this strategy is effective: regions of high epistemic uncertainty are where the model is **most ignorant**, so adding data there maximizes information gain. Compared with random sampling, the amount of data needed to reach the same accuracy can be reduced substantially. Concrete methodology (e.g., BALD) is discussed in Section 4.

---

## 4. Uncertainty Decomposition in Classification

### 4.1 From Regression to Classification

The variance decomposition in regression is intuitive, but classification requires a decomposition based on **entropy** rather than variance. In classification the prediction is a probability distribution $p(y \mid \mathbf{x})$, and entropy is the natural way to measure how "spread out" that distribution is.

### 4.2 Entropy-Based Decomposition

In a $K$-class classification problem the Bayesian predictive distribution is:

$$p(y = k \mid \mathbf{x}, \mathcal{D}) = \int p(y = k \mid \mathbf{x}, \theta)\, p(\theta \mid \mathcal{D})\, d\theta \approx \frac{1}{T}\sum_{t=1}^{T} p(y = k \mid \mathbf{x}, \theta_t)$$

where $\theta_t$ are $T$ samples from the posterior. We decompose the uncertainty of this predictive distribution as follows.

**Total Uncertainty: Predictive Entropy**

$$\mathbb{H}[y \mid \mathbf{x}, \mathcal{D}] = -\sum_{k=1}^{K} p(y=k \mid \mathbf{x}, \mathcal{D}) \log p(y=k \mid \mathbf{x}, \mathcal{D})$$

**Aleatoric Uncertainty: Expected Entropy**

$$\mathbb{E}_{p(\theta|\mathcal{D})}[\mathbb{H}[y \mid \mathbf{x}, \theta]] = -\frac{1}{T}\sum_{t=1}^{T}\sum_{k=1}^{K} p(y=k \mid \mathbf{x}, \theta_t) \log p(y=k \mid \mathbf{x}, \theta_t)$$

- The average of each individual model $\theta_t$'s predictive entropy
- The degree to which **all models are equally confused**

**Epistemic Uncertainty: Mutual Information (BALD)**

$$\mathbb{I}[y; \theta \mid \mathbf{x}, \mathcal{D}] = \mathbb{H}[y \mid \mathbf{x}, \mathcal{D}] - \mathbb{E}_{p(\theta|\mathcal{D})}[\mathbb{H}[y \mid \mathbf{x}, \theta]]$$

$$= \underbrace{\text{Total Uncertainty}}_{\text{Predictive Entropy}} - \underbrace{\text{Aleatoric Uncertainty}}_{\text{Expected Entropy}}$$

This mutual information is known as the **BALD (Bayesian Active Learning by Disagreement)** score (Houlsby et al., 2011). The core idea: a high **mutual information** between $y$ and $\theta$ means that learning $\theta$ would greatly reduce uncertainty about $y$ — in other words, **the current uncertainty stems from the model's ignorance**.

### 4.3 Understanding the Decomposition Through Three Scenarios

```
    Scenario 1: BOTH low  (Confident & Correct)
    ─────────────────────────────────────
    Model 1:  Cat 0.95  |  Dog 0.05
    Model 2:  Cat 0.93  |  Dog 0.07
    Model 3:  Cat 0.96  |  Dog 0.04

    Predictive Entropy:  LOW   (all models say "Cat")
    Expected Entropy:    LOW   (each model is confident)
    Mutual Information:  LOW   (= Low - Low)
    => Confident prediction, low noise, low ignorance

    Scenario 2: HIGH aleatoric, LOW epistemic  (Inherently Ambiguous)
    ─────────────────────────────────────
    Model 1:  Cat 0.52  |  Dog 0.48
    Model 2:  Cat 0.50  |  Dog 0.50
    Model 3:  Cat 0.49  |  Dog 0.51

    Predictive Entropy:  HIGH  (prediction is ~uniform)
    Expected Entropy:    HIGH  (each model is uncertain)
    Mutual Information:  LOW   (= High - High)
    => All models agree: "this is genuinely ambiguous"
    => More data won't help — the input is inherently noisy

    Scenario 3: LOW aleatoric, HIGH epistemic  (Ignorant)
    ─────────────────────────────────────
    Model 1:  Cat 0.95  |  Dog 0.05
    Model 2:  Cat 0.10  |  Dog 0.90
    Model 3:  Cat 0.80  |  Dog 0.20

    Predictive Entropy:  HIGH  (averaged prediction is uncertain)
    Expected Entropy:    LOW   (each model is confident!)
    Mutual Information:  HIGH  (= High - Low)
    => Models confidently DISAGREE — this is epistemic!
    => More data in this region will resolve the disagreement
```

### 4.4 Active Learning with BALD

The BALD score is used directly as an **acquisition function** in active learning. The key procedure:

1. Compute the BALD score for each point in the unlabeled pool

$$\mathcal{U} = \lbrace\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_N\rbrace$$

2. Select $\mathbf{x}^{\ast} = \arg\max_{\mathbf{x} \in \mathcal{U}} \mathbb{I}[y; \theta \mid \mathbf{x}, \mathcal{D}]$
3. Acquire the label for $\mathbf{x}^{\ast}$ and update the model

Why BALD outperforms raw predictive entropy: **it avoids the trap of selecting inherently noisy data.** Predictive entropy alone gives high scores to Scenario 2 (inherently ambiguous) inputs as well, but adding such data to the training set does not improve the model. BALD selects only Scenario 3 (models disagree), **focusing experimental resources on data that will actually yield information gain**.

```
    BALD Acquisition vs. Predictive Entropy

    Unlabeled Pool:
    +------+-------+---------+----------+---------+
    | x_i  | Total | Aleat.  | Epist.   | BALD    |
    |      | Ent.  | (E[H])  | (MI)     | Select? |
    +------+-------+---------+----------+---------+
    | x_1  | 0.12  | 0.10    | 0.02     |   No    |
    | x_2  | 0.95  | 0.90    | 0.05     |   No    | <-- High total
    | x_3  | 0.88  | 0.15    | 0.73     |  YES    | <-- but actually
    | x_4  | 0.30  | 0.25    | 0.05     |   No    |     noisy (x_2)
    | x_5  | 0.91  | 0.20    | 0.71     |  YES    |
    +------+-------+---------+----------+---------+

    Predictive Entropy would pick: x_2, x_5, x_3
    BALD would pick:               x_3, x_5         (skips noisy x_2!)
```

---

## 5. OOD Detection and Distributional Uncertainty

### 5.1 The Softmax Trap

Detecting out-of-distribution (OOD) inputs is a core requirement for safe AI systems. Intuitively, it is the ability to recognize "this input is unlike anything I was trained on."

Using softmax probabilities as an uncertainty indicator is **dangerous**. The structural problem with the softmax function:

$$p(y=k \mid \mathbf{x}) = \frac{\exp(z_k)}{\sum_{j} \exp(z_j)}$$

where $z_k$ denotes the logits. The issues are:

- **Softmax always outputs a valid probability distribution** — no matter how bizarre the input, $\sum_k p(y=k) = 1$
- **Saturation in high-dimensional logit space**: logits of ReLU networks can grow proportionally to the input norm, causing the softmax to assign extremely high probability to a single class even for OOD inputs
- **Hein et al. (2019)**: proved theoretically that ReLU networks produce **high-confidence predictions no matter how far the input is moved from the training distribution**

```
    The Softmax Trap

    Input space:
    +----------------------------------+
    |           training               |
    |           distribution           |
    |        +----------+              |
    |        |  ID data |              |
    |        |  (seen)  |              |
    |        +----------+              |
    |                                  |
    |  x_ood (far from training)       |
    |    *                             |
    +----------------------------------+

    Standard NN:
    x_ood --> [Neural Net] --> logits: [12.3, 0.1, 0.5]
                           --> softmax: [0.9999, 0.0001, 0.0004]
                           --> "99.99% confident it's class 1!"
                           --> WRONG! Should say "I don't know"

    Bayesian NN:
    x_ood --> [Model 1] --> "class 1" (confident)
          --> [Model 2] --> "class 3" (confident)
          --> [Model 3] --> "class 2" (confident)
          --> Models DISAGREE --> High epistemic uncertainty
          --> "I don't know" (correct!)
```

### 5.2 Why Epistemic Uncertainty Is Well-Suited to OOD Detection

The effectiveness of epistemic uncertainty for OOD detection has a clear theoretical basis:

- OOD regions contain **no training data**, so the posterior $p(\theta \mid \mathcal{D})$ provides **no constraint** on predictions in those regions
- As a result, different posterior samples (or ensemble members) produce **different predictions** for OOD inputs
- This "inter-model disagreement" is captured as high epistemic uncertainty

Aleatoric uncertainty, by contrast, is ill-suited for OOD detection:
- Aleatoric uncertainty reflects the **noise level of the data**, a pattern learned from the training distribution
- In OOD regions the aleatoric estimate may be high or low — the model has never learned the noise characteristics of that region, so the estimate is meaningless

### 5.3 The Feature Collapse Problem

There is, however, a **fundamental limitation** to epistemic-uncertainty-based OOD detection: the problem of **feature collapse** (or feature collapse to in-distribution).

Core mechanism:
- The feature representation extracted at the last hidden layer of a neural network **may not preserve distances in input space**
- An OOD input may be **mapped to the same region** in feature space as in-distribution (ID) data
- When this happens, every uncertainty estimation method operating in feature space **fails to detect the OOD input**

```
    Feature Collapse Problem

    Input Space:                  Feature Space:
    +------------------+          +------------------+
    |    ID cluster    |          |    ID cluster    |
    |     ++++++       |   f(x)  |     ++++++       |
    |    ++++++++      |  ---->  |    ++++++++      |
    |     ++++++       |          |     +OO+++      | <-- OOD mapped
    |                  |          |      ++++        |     INTO ID region!
    |           OOO    |          |                  |
    |          OOOOO   |          |                  |
    |           OOO    |          |                  |
    +------------------+          +------------------+

    + = In-Distribution data
    O = Out-of-Distribution data

    In input space: clearly separated
    In feature space: COLLAPSED together
    => Uncertainty methods in feature space FAIL
```

Solutions to this problem include:
- **SNGP (Spectral-Normalized Neural Gaussian Process)**: applies spectral normalization to the feature extractor, enforcing that input-space distances are preserved in feature space (a bi-Lipschitz constraint)
- **DUQ (Deterministic Uncertainty Quantification)**: uses an RBF kernel to directly leverage distances in feature space
- These distance-aware methods will be covered in detail in Part 5

---

## 6. A Critical Perspective — Limitations of the Traditional Decomposition

### 6.1 The Challenge Raised by Wimmer et al. (2024)

The aleatoric-epistemic decomposition presented so far is the standard framework widely adopted in the AI/ML community. Recently, however, Wimmer et al. (2024, "Quantifying Aleatoric and Epistemic Uncertainty in Machine Learning: Are Conditional Entropy and Mutual Information Appropriate Measures?") raised **fundamental objections** to this traditional decomposition.

**Core critique: Aleatoric $\neq$ Irreducible under Model Misspecification**

In the traditional decomposition, aleatoric uncertainty is used synonymously with **"irreducible uncertainty."** But this equation holds only when the **model is well-specified** — that is, the hypothesis space $\mathcal{H}$ contains the true data-generating process.

In practice this condition is almost never satisfied. Specifically:

- If the model class

$$\mathcal{F} = \lbrace f_\theta : \theta \in \Theta \rbrace$$

**does not contain the true function** $f^{\ast}$, then even with infinite data the posterior will not converge to $f^{\ast}$
- In this case there is a **residual error** that persists even after arbitrarily large data collection. Part of this residual arises because the model is misspecified (reducible by choosing a better model), and only part is genuine irreducible noise
- Yet the traditional decomposition classifies **all** of this residual as aleatoric

### 6.2 Concrete Example

Suppose $y = f^{\ast}(\mathbf{x}) + \epsilon$ (true model) with $\epsilon \sim \mathcal{N}(0, \sigma_{\text{true}}^2)$. If our model class $\mathcal{F}$ does not contain $f^{\ast}$, then even the best approximation $f_{\theta^{\ast}} \in \mathcal{F}$ cannot exactly reproduce $f^{\ast}$:

$$y - f_{\theta^{\ast}}(\mathbf{x}) = \underbrace{[f^{\ast}(\mathbf{x}) - f_{\theta^{\ast}}(\mathbf{x})]}_{\text{approximation error}} + \underbrace{\epsilon}_{\text{true noise}}$$

In the traditional decomposition, $\text{Var}[y \mid \mathbf{x}, \theta^{\ast}]$ treats **both terms combined** as aleatoric. Because collecting more data only drives $\theta \to \theta^{\ast}$ and the approximation error does not vanish, it is deemed "irreducible."

However, the approximation error **can be reduced by choosing a better model** — so it is semantically closer to **epistemic**. The traditional framework misses this distinction.

### 6.3 An Alternative Framework: Reducible vs. Irreducible

Wimmer et al. propose the following alternative decomposition:

| Traditional Decomposition | Alternative Decomposition | Relationship |
|:---:|:---:|:---|
| Aleatoric | Irreducible | Coincide only when the model is well-specified |
| Epistemic | Reducible | Traditional epistemic is a subset of reducible |

Alternative definitions:
- **Irreducible uncertainty**: uncertainty that cannot be removed regardless of the model class or the amount of data — genuine noise
- **Reducible uncertainty**: uncertainty that can be diminished through a better model or more data — includes residuals from model misspecification

```
    Traditional vs. Alternative Decomposition

    Traditional:
    +---------------------------+------------------+
    |   Aleatoric               |   Epistemic      |
    | (data noise +             | (parameter       |
    |  approx. error when       |  uncertainty     |
    |  model misspecified)      |  from finite     |
    |                           |  data)           |
    +---------------------------+------------------+

    Alternative (Wimmer et al., 2024):
    +-----------------+---------+------------------+
    |  Irreducible    | Approx. |   Reducible      |
    |  (true noise    | Error   | (parameter       |
    |   only)         | (model  |  uncertainty     |
    |                 | mispec.)|  + approx error) |
    +-----------------+---------+------------------+
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                      = Things we CAN fix
```

### 6.4 Practical Implications

The practical impact of this critique is as follows:

**1) Aleatoric uncertainty should not be dismissed as "uncertainty we can ignore"**
- One must determine whether high aleatoric uncertainty reflects genuine noise or model misspecification
- If the latter, it can be reduced through model improvement, so ignoring it is inappropriate

**2) Ablation tests are needed**
- Gradually increase model capacity and observe how aleatoric uncertainty changes
- If aleatoric uncertainty decreases as the model grows, **it was not truly aleatoric**
- Only the portion that converges stably represents genuine irreducible noise

**3) Ensemble diversity analysis**
- In a deep ensemble, monitor inter-member variance of the predicted means (epistemic) alongside each member's predicted variance (aleatoric)
- A large drop in aleatoric uncertainty after an architecture improvement is evidence that model misspecification was present

**A balanced view**: The critique by Wimmer et al. is theoretically important, but it does **not** render the traditional decomposition useless. In most practical scenarios:
- The traditional decomposition remains useful **as a heuristic**
- OOD detection, active learning, and selective prediction based on epistemic uncertainty work well regardless of whether model misspecification is present
- What requires caution is interpreting aleatoric uncertainty as an **"absolute irreducible lower bound"**

---

## 7. Synthesis — The Practical Value of Uncertainty Decomposition

### 7.1 "What Should We Do Next?"

The central message of this post converges on a single point: **knowing the type of uncertainty determines the next action.**

```
    Decision Framework Based on Uncertainty Type

    High Uncertainty Detected
            |
            v
    +------------------+
    | Which type is    |
    | dominant?        |
    +--------+---------+
             |
      +------+------+
      |             |
      v             v
    Aleatoric     Epistemic
      |             |
      v             v
    +----------+  +----------+
    | Actions: |  | Actions: |
    | - Better |  | - Collect|
    |   sensor |  |   more   |
    | - Reduce |  |   data   |
    |   noise  |  | - Improve|
    | - Accept |  |   model  |
    |   as     |  | - Active |
    |   limit  |  |   learn  |
    | - Widen  |  | - Flag   |
    |   conf.  |  |   as OOD |
    |   band   |  | - Defer  |
    +----------+  |   to     |
                  |   human  |
                  +----------+
```

Scenario-by-scenario summary:

| Scenario | Dominant Uncertainty | Prescription |
|:---|:---:|:---|
| Low hit rate in drug screening | Epistemic | Expand the chemical space of the training data |
| Poor reproducibility of a particular assay | Aleatoric | Improve the experimental protocol or ensemble multiple assay results |
| Prediction failure on OOD scaffolds | Epistemic | Add data from that scaffold family (active learning) |
| Intrinsically noisy experimental values (e.g., solubility) | Aleatoric | Accept the limitation and widen the prediction interval |
| Systematic model error in a specific region | Model mispec. | Improve the architecture (caution: traditionally may be misclassified as aleatoric) |

### 7.2 Summary Equations

For regression:

$$\boxed{\text{Var}[y \mid \mathbf{x}] = \underbrace{\mathbb{E}_{\theta}[\sigma^2(\mathbf{x}, \theta)]}_{\text{Aleatoric: limitation of the data}} + \underbrace{\text{Var}_{\theta}[\mu(\mathbf{x}, \theta)]}_{\text{Epistemic: limitation of the model}}}$$

For classification:

$$\boxed{\mathbb{H}[y \mid \mathbf{x}] = \underbrace{\mathbb{E}_{\theta}[\mathbb{H}[y \mid \mathbf{x}, \theta]]}_{\text{Aleatoric: inherent ambiguity}} + \underbrace{\mathbb{I}[y; \theta \mid \mathbf{x}]}_{\text{Epistemic: inter-model disagreement (BALD)}}}$$

---

## Closing: The Challenges Beyond Anatomy

Decomposing uncertainty is only the beginning. Key questions remain unanswered:

1. **When a model says "90% confident," is it truly right 90% of the time?** — The **calibration** problem: verifying that the *magnitude* of uncertainty is correct. No matter how sophisticated the aleatoric-epistemic decomposition, if the absolute scale is wrong, it will lead to flawed decisions.

2. **Can we guarantee uncertainty intervals in a distribution-free manner?** — The Bayesian framework depends on model assumptions, but **conformal prediction** provides valid coverage guarantees for any model from finite samples.

The next installment, Part 4 **"Calibration and Conformal Prediction,"** addresses these questions — moving beyond the anatomical distinction of uncertainty to methods for verifying and guaranteeing the **reliability of its absolute magnitude**.

---

## References

### Core — Uncertainty Decomposition
1. Kendall, A. & Gal, Y. (2017). "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?" *NeurIPS*.
2. Der Kiureghian, A. & Ditlevsen, O. (2009). "Aleatory or Epistemic? Does It Matter?" *Structural Safety*, 31(2), 105-112.
3. Houlsby, N., Huszar, F., Ghahramani, Z. & Lengyel, M. (2011). "Bayesian Active Learning for Classification and Preference Learning." *arXiv:1112.5745*.

### Critical Perspectives
4. Wimmer, L., Sale, Y., Hofman, P. & Huellermeier, E. (2024). "Quantifying Aleatoric and Epistemic Uncertainty in Machine Learning: Are Conditional Entropy and Mutual Information Appropriate Measures?" *arXiv:2209.03302*.

### OOD Detection and Feature Collapse
5. Hein, M., Andriushchenko, M. & Bitterwolf, J. (2019). "Why ReLU Networks Yield High-Confidence Predictions Far Away From the Training Data and How to Mitigate the Problem." *CVPR*.
6. Liu, J. Z. et al. (2020). "Simple and Principled Uncertainty Estimation with Deterministic Deep Learning via Distance Awareness." *NeurIPS*. (SNGP)
7. Van Amersfoort, J., Smith, L., Teh, Y. W. & Gal, Y. (2020). "Uncertainty Estimation Using a Single Deep Deterministic Neural Network." *ICML*. (DUQ)

### Scientific Applications
8. Ryu, S., Kwon, Y. & Kim, W. Y. (2019). "A Bayesian Graph Convolutional Network for Reliable Prediction of Molecular Properties with Uncertainty Quantification." *Chemical Science*, 10, 8438-8446.

### Heteroscedastic Modeling
9. Nix, D. A. & Weigend, A. S. (1994). "Estimating the Mean and Variance of the Target Probability Distribution." *IEEE International Conference on Neural Networks*.
10. Le, Q. V., Smola, A. J. & Canu, S. (2005). "Heteroscedastic Gaussian Process Regression." *ICML*.
