---
title: "Bayesian DL & UQ Part 6: UQ in Science — Molecules, Proteins, and Materials"
date: 2026-04-09 14:30:00 +0900
categories: [AI & ML, ML Theory]
tags: [bayesian-deep-learning, drug-discovery, bayesian-gcn, alphafold, neural-network-potentials, active-learning]
math: true
---

**Bayesian Deep Learning and Uncertainty Quantification — A Journey Toward Trustworthy AI**

*This is Part 6 of an 8-part series.*

- **Part 0:** [Beyond Predictions — Why Uncertainty Matters](/posts/bayesian-dl-uq-part0-why-uncertainty/)
- **Part 1:** [The Language of Bayesian Inference](/posts/bayesian-dl-uq-part1-bayesian-language/)
- **Part 2:** [The Art of Approximation — From Variational to Ensemble](/posts/bayesian-dl-uq-part2-art-of-approximation/)
- **Part 3:** [The Anatomy of Uncertainty — Aleatoric vs. Epistemic](/posts/bayesian-dl-uq-part3-anatomy-of-uncertainty/)
- **Part 4:** [Calibration and Conformal Prediction](/posts/bayesian-dl-uq-part4-calibration-conformal/)
- **Part 5:** [Single-Pass UQ — Evidential Deep Learning and Distance-Aware Methods](/posts/bayesian-dl-uq-part5-single-pass-uq/)
- **Part 6:** UQ in Science — Molecules, Proteins, and Materials (this post)
- **Part 7:** [The Future of UQ — Uncertainty in the Age of Foundation Models](/posts/bayesian-dl-uq-part7-future-foundation-models/)

---

## Hook: Out of 200 Million Molecules, Which Ones Do You Synthesize?

Suppose you are running a virtual screening campaign in a drug discovery project. A deep learning model has predicted binding affinities for 200 million candidate molecules, and you select the top 100 for synthesis. After testing, 29 out of 100 show actual activity — a **hit rate of 29%**. Not bad, but given the cost of synthesis, disappointing.

Now suppose that the same model also provides **"how uncertain this prediction is"** alongside each prediction. You prioritize molecules that have both high predicted activity and low uncertainty — molecules for which the model is saying "this molecule is active, and I am confident in this prediction." When you synthesize the same number — 100 molecules — this time **69 show activity**, a hit rate of 69%.

These are real numbers. In 2019, my colleagues and I reported this result in our Bayesian GCN paper, observed during virtual screening for EGFR (epidermal growth factor receptor) inhibitors. A single piece of uncertainty information more than **doubled** the practical value of the same model.

This experience gave me a conviction: **UQ is not an academic curiosity — it is an engineering tool that determines the speed of scientific discovery.** From hit rates in virtual screening to the next experiment chosen by an autonomous lab, uncertainty quantification has a direct impact on the laboratory bench, far beyond theory.

In this post, I develop the evidence for this conviction across seven sections.

---

## 1. Molecular Property Prediction and UQ — My Experience

### 1.1 The Design Philosophy of Bayesian GCN

In 2017-2018, applying graph neural networks (GNNs) to molecular property prediction was an active area of research. However, most studies focused exclusively on **prediction accuracy**, treating prediction **reliability** as an afterthought. My motivation was straightforward:

- **In drug discovery, a single wrong prediction can waste six months and tens of thousands of dollars in synthesis costs**
- If a model can say "this prediction should not be trusted," a researcher can skip that molecule and allocate resources to more certain candidates
- If the **source** of uncertainty can be identified (data noise vs. insufficient training data), even more targeted decisions become possible

Bayesian GCN was designed from this philosophy (Ryu, Kwon & Kim, *Chemical Science*, 2019).

**Architecture**:

- **Molecular representation**: augmented graph convolutional network (GCN) — **3 GCN layers** for learning atom-level features
- **Attention readout**: **4 self-attention heads** for generating graph-level representations — instead of simple sum/mean pooling, the contribution of each atom is determined through learning
- **Feature dimension**: **256-dimensional** hidden features
- **Prediction head**: **2 fully-connected (FC) layers** producing the final prediction
- **Bayesian treatment**: **Concrete Dropout** applied to all layers

```
Input Molecule (SMILES -> Graph)
       |
       v
+---------------------+
|  Augmented GCN x 3  |  <- atom-level feature learning
|  (256-dim features)  |     + Concrete Dropout per layer
+---------+-----------+
          |
          v
+---------------------+
|  Self-Attention x 4  |  <- learned atom importance weighting
|  (multi-head)        |     + Concrete Dropout
+---------+-----------+
          |
          v
+---------------------+
|  FC Layer x 2        |  <- property prediction
|  + Concrete Dropout  |     outputs: mean + variance
+---------+-----------+
          |
          v
   Prediction: y_hat +/- sigma
   (aleatoric + epistemic)
```

### 1.2 Concrete Dropout — Why Per-Layer Dropout Rate Learning Matters

As discussed in Part 2, MC Dropout (Gal & Ghahramani, 2016) showed that dropout is a form of variational inference. However, standard MC Dropout has practical limitations:

- **Fixed dropout rate**: the same $p$ is applied to all layers — an unrealistic assumption that the posterior has the same shape across all layers
- **Hyperparameter search cost**: determining the dropout rate through grid search often leads to suboptimal values
- **Ignoring per-layer role differences**: early layers (general chemical patterns) and later layers (task-specific patterns) require different levels of regularization

**Concrete Dropout** (Gal, Hron & Kendall, *NeurIPS*, 2017) solves this problem:

- The dropout rate $p_l$ for each layer $l$ is made a **learnable parameter**
- A **Binary Concrete distribution** (continuous Bernoulli relaxation) enables gradient-based optimization
- An **entropy regularizer** in the objective prevents the dropout rate from collapsing to 0 or 1

$$\mathcal{L}_{\text{concrete}} = \frac{1}{N} \sum_{i=1}^{N} \ell(y_i, \hat{y}_i) + \sum_{l=1}^{L} \frac{\lambda_l}{N} \|\mathbf{W}_l\|^2 \cdot (1 - p_l) - \sum_{l=1}^{L} H(p_l)$$

where $H(p_l)$ is the entropy of the dropout rate:

$$H(p_l) = -p_l \log p_l - (1-p_l) \log(1-p_l)$$

Observations in Bayesian GCN:

- **Early GCN layers**: learned dropout rates were relatively **low** (0.05-0.15) — high certainty for basic atomic feature extraction
- **Later FC layers**: dropout rates were relatively **high** (0.2-0.4) — more uncertainty in task-specific prediction
- This aligns with intuition: recognizing a benzene ring is far less uncertain than predicting binding affinity to a specific target

### 1.3 EGFR Virtual Screening — What the Numbers Tell Us

The practical value of Bayesian GCN was clearly demonstrated in EGFR inhibitor virtual screening. After training on EGFR assay data from the ChEMBL database, virtual screening was performed on a separate test set.

**Key Result Table — Active molecule count (hit count) in top-N molecules**:

| Top-N | Standard ML | MAP (point estimate) | Bayesian GCN |
|:-----:|:-----------:|:-------------------:|:------------:|
| 100   | 29          | 57                  | **69**       |
| 200   | 67          | 130                 | **140**      |
| 500   | 277         | 346                 | **368**      |

Key takeaways from this table:

- **Standard ML to MAP**: even with just point estimates, the architectural improvements (GCN + attention) are substantial (29 to 57)
- **MAP to Bayesian GCN**: adding **uncertainty-based filtering** alone to the same architecture yields further improvement (57 to 69)
- **Hit rate in the top 100**: Standard ML 29% vs. Bayesian GCN 69% — a **2.4x improvement**
- **The gap narrows as top-N increases**: at top 500, it is 277 vs. 368 (1.3x) — UQ's value is maximized under **resource-constrained settings**

**Uncertainty-based selection strategy**:

- Naive criterion: rank by predicted value (standard approach)
- Bayesian criterion: rank by **predicted value - k x epistemic uncertainty** (inverse UCB)
  - Prioritize molecules that have both high predicted activity and low epistemic uncertainty
  - This strategy focuses resources on molecules for which the model has "sufficient evidence to predict activity"

### 1.4 Harvard Clean Energy Project — UQ as a Data Quality Diagnostic Tool

Another application of Bayesian GCN was **data quality diagnosis**. The Harvard Clean Energy Project (CEP) is a large-scale dataset of power conversion efficiency (PCE) predictions for organic solar cell candidate molecules computed via DFT.

**A noteworthy observation**:

- The dataset contains a substantial number of molecules with **PCE = 0**
- When predicted with Bayesian GCN, these PCE = 0 molecules exhibited **abnormally large aleatoric uncertainty**
- This directly reflects the meaning of aleatoric uncertainty discussed in Part 3: noise and inconsistency in the data itself

**Interpretation**: the label PCE = 0 encompasses two possibilities:

- The molecule genuinely has PCE = 0: zero solar cell efficiency
- The DFT calculation failed to converge or produced a non-physical result: effectively a **missing value**

Since the model cannot distinguish between these two cases, it expresses the ambiguity as high aleatoric uncertainty. This is an empirical demonstration that **UQ can serve as a tool for diagnosing data quality issues**, not just prediction accuracy.

### 1.5 Subsequent Developments — Lessons from Benchmarks

After Bayesian GCN, UQ methodology in molecular ML began to be systematically organized through benchmark studies.

**Hirschfeld et al. (*JCIM*, 2020)** — "Uncertainty Quantification Using Neural Networks for Molecular Property Prediction":

- Compared UQ methods across diverse molecular property datasets including QM9, Lipophilicity, and FreeSolv
- Methods: MC Dropout, deep ensemble, mean-variance estimation, bootstrapping
- **Key finding**: deep ensemble generally performs best, but **is not always the winner across all datasets**
- In particular, **bootstrapping is competitive when data size is small**

**Scalia et al. (*JCIM*, 2020)** — "Evaluating Scalable Uncertainty Estimation Methods for DNN-Based Molecular Property Prediction":

- Included larger datasets and industrially relevant endpoints
- Evaluated UQ performance under scaffold split (simulating distribution shift)
- **Key finding**: under scaffold split, UQ quality of all methods **degrades significantly**
- This reveals the gap between benchmarks and real-world scenarios

**Common lessons from both benchmarks**:

- **There is no single best UQ method** — the optimal approach varies depending on the dataset, split strategy, and evaluation metric
- Deep ensemble is generally a safe choice, but **not always justified relative to its computational cost**
- **Distribution shift is the real test for UQ** — good UQ performance under random split does not reflect reality

### 1.6 UQ Under Distribution Shift — The Rise of Error Models

The most recent development is UQ evaluation under **temporal distribution shift** (temporal split).

**JCIM (2026) Benchmark**:

- **Data scale**: 293,000 compounds
- **Split strategy**: temporal split — train on data before a specific year, evaluate on data after
  - This accurately reflects the real drug discovery scenario: building a model on historical data and applying it to future molecules
- **Endpoints**: 7 ADME (absorption, distribution, metabolism, excretion) endpoints
- **Compared methods**: MC Dropout, deep ensemble, conformal prediction, error model (meta-model), EDL

**Key result — Robustness of the Error Model**:

- Error model (or meta-model): a separate Random Forest model that takes the base model's predictions and molecular descriptors as input and **predicts the magnitude of the base model's error**
- Under temporal split, the error model showed the **most robust UQ performance**
- Deep ensemble also performed well, but exhibited **increasing performance degradation under larger temporal shifts**
- Conformal prediction: the coverage guarantee is maintained, but prediction intervals become unnecessarily wide

**Why is the error model effective?**

- It learns **patterns** in the chemical space where the model struggles — specific functional groups, scaffolds, and physicochemical property ranges where errors tend to be large
- Relatively robust to distribution shift because patterns in chemical descriptor space are partly preserved under temporal shift
- **Conceptually simple**: can be implemented by simply adding a single RF on top of an existing model

**Virtual Screening Pipeline with UQ — Complete Workflow**:

```
Chemical Library (10^6 - 10^9 molecules)
              |
              v
+--------------------------+
|  Molecular Representation |  <- SMILES -> graph / fingerprint
+------------+-------------+
             |
             v
+--------------------------+
|  Property Prediction      |  <- GNN, Transformer, etc.
|  + UQ Method              |     MC Dropout / Ensemble / EDL
|                           |     -> y_hat, sigma_epistemic,
|                           |       sigma_aleatoric
+------------+-------------+
             |
             v
+--------------------------+
|  Uncertainty-Aware        |  <- Filter: high prediction +
|  Ranking & Filtering      |     low epistemic uncertainty
|                           |  <- Or: UCB, EI, Thompson
+------------+-------------+
             |
             v
+--------------------------+
|  Synthesis & Testing      |  <- Top-N candidates
|  (Resource-Constrained)   |     N = 50 ~ 500
+------------+-------------+
             |
             v
      Hit Rate: 2-3x improvement
      with UQ vs. without UQ
```

---

## 2. AlphaFold's Confidence Is Not Bayesian Uncertainty

### 2.1 pLDDT — What It Is and How It Is Trained

AlphaFold2 (Jumper et al., *Nature*, 2021) revolutionized protein structure prediction. Its confidence score, **pLDDT** (predicted Local Distance Difference Test), provides a per-residue confidence estimate on a 0-100 scale.

**How pLDDT is trained**:

- An **auxiliary prediction head** attached to AlphaFold's structure module
- Training target: the LDDT score between the predicted and experimental structures (computed empirically)
- Essentially a **self-supervised target** — the model learns to predict its own error
- Implemented as a classification head: outputs probabilities over 4 LDDT bins for each residue, then converts to an expected value

**The key point**: pLDDT is **not an uncertainty derived from a probability distribution, but a learned self-assessment**. This is fundamentally different from the Bayesian uncertainty discussed in Parts 1-5.

### 2.2 What pLDDT Does Well

pLDDT is useful in several respects:

- **Distinguishing globular domains from disordered regions**: high pLDDT for well-defined 3D structures, low pLDDT for intrinsically disordered regions (IDRs)
- **Confidence of secondary structure elements**: high pLDDT for regular structures such as alpha-helices and beta-sheets
- **Filtering large-scale structure databases**: effective for selecting high-confidence predicted structures from the AlphaFold Protein Structure Database
- **Overall trend**: a meaningful correlation between pLDDT and actual structural accuracy

### 2.3 What pLDDT Cannot Do — Systematic Failures

However, pLDDT has systematic limitations, and accepting it uncritically can lead to incorrect biological conclusions.

**Problem 1 — High-confidence wrong predictions (~10% false high-confidence)**

According to Terwilliger et al.'s AlphaFold validation study:

- Among residues with pLDDT > 70, **approximately 10%** differ from the experimental structure by **more than 2 Angstroms**
- This is especially pronounced at loop regions and domain boundaries
- When users rely on pLDDT > 70 as a threshold for "reliable," a **10% false positive rate** can cause serious errors in downstream research (drug design, mutagenesis, etc.)

**Problem 2 — Failure to capture binding-partner-induced structural changes**

- Many proteins change structure upon binding to ligands, other proteins, or nucleic acids (**induced fit**, **conformational selection**)
- AlphaFold tends to predict the **apo (unbound) state** by default
- pLDDT does not reflect such **binding-dependent flexibility** — the information "this residue's structure may vary depending on the binding partner" is absent

**Problem 3 — Limitations in antibody-antigen complexes**

- When predicting antibody-antigen complexes with AlphaFold-Multimer, the success rate is around **51-63%**
- CDR (Complementarity-Determining Region) loops are inherently flexible and adopt diverse conformations
- Even when high pLDDT is assigned, the actual binding pose may differ substantially

### 2.4 The Core Critique — Inability to Distinguish Aleatoric from Epistemic

As emphasized in Part 3, knowing the **source** of uncertainty is as important as knowing its **magnitude**. The fundamental limitation of pLDDT is precisely that it cannot provide this distinction.

- **Aleatoric uncertainty** (structural flexibility): this residue inherently fluctuates between multiple conformations and cannot be pinned down to a single structure
- **Epistemic uncertainty** (insufficient training data): the PDB lacks experimental structures of this fold type, so the model has not learned this region well

When pLDDT is low, we cannot answer the following questions:

- "Would this prediction improve if more experimental structures were added to the PDB?" (epistemic — yes)
- "Or is this residue inherently flexible, so no model could predict a single structure?" (aleatoric — no)

This distinction **determines the direction of follow-up research**:

- If epistemic uncertainty is high — determine the relevant experimental structure, or attempt ensemble prediction
- If aleatoric uncertainty is high — run molecular dynamics simulations to generate a conformational ensemble

### 2.5 Alternatives — Ensembles and Conformal Prediction

Several approaches have been proposed to complement the limitations of pLDDT:

**Ensemble AlphaFold2**:

- Run AlphaFold2 **multiple times** with different random seeds and MSA subsampling
- Use **structural variance** among predicted structures as a proxy for epistemic uncertainty
- Advantage: straightforward to implement, and captures uncertainty that pLDDT alone cannot
- Disadvantage: 5-10x computational cost — prohibitive at proteome scale

**Conformal prediction wrapping**:

- Apply conformal prediction (discussed in Part 4) to AlphaFold
- Compute the nonconformity score distribution on a calibration set (proteins with known experimental structures)
- Provide **prediction intervals with coverage guarantees** for new predictions
- Advantage: distribution-free, with finite-sample coverage guarantee
- Limitation: only marginal coverage is guaranteed — conditional coverage for a specific fold family is not

**Summary — The right attitude toward AlphaFold confidence**:

- pLDDT is a **useful filter**, but it is **not Bayesian uncertainty**
- "pLDDT > 90, so this structure is safe for drug design" is **dangerous reasoning**
- For important decisions, always combine **additional UQ** (ensemble, conformal, MD simulation)

---

## 3. Neural Network Potentials and UQ

### 3.1 Molecular Simulation in the Foundation Model Era

Neural Network Potentials (NNPs) learn quantum mechanical energies to deliver DFT-level accuracy at MD-level speed. In recent years, the NNP field has entered a **foundation model era**:

- **MACE-MP-0** (Batatia et al., 2023): an equivariant message-passing model pretrained on the vast DFT data of the Materials Project
- **ANI** (Smith et al.): an NNP specialized for organic molecules, trained on energies across diverse conformations
- **CHGNet** (Deng et al., 2023): a graph neural network potential that incorporates charge information
- **GNoME** (Merchant et al., *Nature*, 2023): Google DeepMind's materials discovery effort — using NNPs to predict 2.2M stable crystal structures

These models predict energies, forces, and stresses for systems containing **millions of atoms**. A single wrong prediction can destabilize an MD simulation, producing non-physical trajectories, or lead to incorrect candidates in stable structure searches. UQ is a critical safety mechanism for detecting such failures before they propagate.

### 3.2 A Hierarchy of UQ Methods

UQ methods for NNPs form a hierarchy along the **accuracy-cost trade-off**:

| Method | Additional Cost | Calibration Quality | Key Characteristic |
|:-------|:---------------:|:-------------------:|:-------------------|
| **EDL** (Evidential) | 1x | Good | Single forward pass, aleatoric/epistemic separation |
| **Readout ensemble** | 3-5x | Good | Shared backbone, multiple readout heads |
| **Deep ensemble** | 5-10x | Best | Independently trained models |

**Details of each method**:

**EDL (1x cost)** — applied to NNPs in Nature Communications (2025):

- Evidential regression with a NIG (Normal-Inverse-Gamma) prior
- Outputs 4 parameters $(\gamma, \nu, \alpha, \beta)$ for atomic energies
- Enables **per-atom uncertainty decomposition**: aleatoric vs. epistemic uncertainty for each atom
- Near-zero additional computational cost (only the output head is modified)

**Readout ensemble (3-5x cost)**:

- A single shared message-passing backbone with **multiple readout heads** trained independently
- Feature extraction runs only once, making this more efficient than a full ensemble
- If the backbone is well-trained, readout-level diversity alone provides substantial UQ quality

**Deep ensemble (5-10x cost)**:

- M independently trained NNPs from different initializations
- Epistemic uncertainty is estimated from the variance of predictions
- **Highest calibration quality**, but training cost scales by a factor of M

### 3.3 Spatially Resolved Uncertainty — The Value of Per-Atom Uncertainty

A distinctive value of UQ in NNPs is **spatially resolved uncertainty**.

```
Total system energy uncertainty:
  E_total +/- sigma_total

         | decomposition

Per-atom uncertainty:
  atom 1: E_1 +/- sigma_1  (low  -- bulk region)
  atom 2: E_2 +/- sigma_2  (low  -- well-represented)
  atom 3: E_3 +/- sigma_3  (HIGH -- surface defect)
  atom 4: E_4 +/- sigma_4  (HIGH -- unusual coordination)
  ...

-> If sigma_3, sigma_4 are large:
  "This atomic environment is underrepresented in training data"
  -> Prioritize additional DFT calculations accordingly
```

This spatial resolution enables the following applications:

- **Active learning**: prioritize DFT calculations for atomic environments with high uncertainty, efficiently expanding the training set
- **Failure diagnosis**: when an MD simulation becomes unstable, immediately identify which atom is the culprit
- **Domain of applicability**: when applying an NNP to a new material system, determine which atomic environments lie outside the training distribution

### 3.4 UQ in MD Simulation — On-the-Fly Uncertainty Monitoring

A particularly critical application of NNP UQ is **molecular dynamics (MD) simulation**. In MD, the NNP computes forces at every time step, so a single erroneous prediction can drive the entire trajectory into a non-physical regime.

**On-the-fly UQ monitoring workflow**:

```
MD Step t
    |
    +-- Compute forces with NNP
    +-- Compute per-atom uncertainty
    |
    +-- IF max(sigma_atom) > threshold_warn:
    |       -> Log warning, flag this frame
    |
    +-- IF max(sigma_atom) > threshold_halt:
    |       -> Halt simulation
    |       -> Trigger DFT calculation for this config
    |       -> Retrain NNP with new data point
    |       -> Resume simulation
    |
    +-- ELSE: continue normally
```

Practical value of this approach:

- **Prevention of non-physical trajectories**: when high uncertainty is detected, halt the simulation and verify with DFT — preemptively block the MD from exploring a "wrong energy landscape"
- **Automatic training data expansion**: automatically add high-uncertainty configurations to the DFT training set — on-the-fly active learning
- **Computational efficiency**: use only the NNP (~1/1000th the cost of DFT) for most time steps, and invoke DFT only at uncertain moments

### 3.5 The Misspecification Problem — When UQ Also Fails

An important caveat: **when the model class itself is wrong, UQ fails with it.**

For example:

- If an NNP **only accounts for pairwise interactions** but the real system is dominated by **many-body effects**
- Because the model's assumptions are wrong, uncertainty estimates can be systematically underestimated
- If all members of a deep ensemble are wrong in the same direction, inter-model variance shrinks, delivering a prediction that is "confidently wrong"

This is a concrete instance of the **model misspecification** problem discussed in Part 3.

**Specific failure modes**:

| Situation | Cause | UQ Behavior | Risk Level |
|:----------|:------|:------------|:----------:|
| Element combinations outside training range | Epistemic | High uncertainty (normal) | Low |
| Pressure/temperature outside training range | Epistemic | High uncertainty (normal) | Low |
| Wrong physical assumptions (e.g., 2-body only) | Misspecification | **Low uncertainty with wrong prediction** | **High** |
| Inaccurate DFT level in training data | Data quality | High aleatoric (partially captured) | Medium |

The third row is the most dangerous: a situation where the model does not even know what it does not know (**unknown unknowns**).

**Mitigation strategies**:

- **Ensemble of diverse architectures**: instead of repeating the same architecture, ensemble MACE + ANI + CHGNet — models with different inductive biases are less likely to fail in the same way
- **Physics-informed constraints**: limit the range of misspecification with conservation of energy, symmetry, virial theorem, and other physical constraints
- **Out-of-distribution detection**: measure distance from the training distribution in atomic environment space (applying the SNGP/DUQ ideas from Part 5 to NNPs)
- **Cross-validation with different levels of theory**: periodically compare NNP predictions against a small number of high-accuracy calculations (CCSD(T)) to detect systematic bias

---

## 4. Active Learning — UQ Designs the Experiment

### 4.1 The Active Learning Loop — Principles of Data-Efficient Discovery

Active learning is the most direct application of UQ: using model uncertainty to **decide what data to collect next**.

```
+----------------------------------------------------------+
|                  ACTIVE LEARNING LOOP                     |
|                                                          |
|   +---------+     +------------+     +--------------+    |
|   |  Train   |---->|  Predict   |---->|  Acquisition  |   |
|   |  Model   |     |  + UQ      |     |  Function     |   |
|   +----^-----+     +------------+     +------+-------+   |
|        |                                      |           |
|        |           +------------+             |           |
|        +-----------| Update     |<------------+           |
|                    | Dataset    |                         |
|                    +-----^------+                         |
|                          |                                |
|                    +-----+------+                         |
|                    | Experiment |  <- wet-lab or DFT      |
|                    | / Oracle   |                         |
|                    +------------+                         |
|                                                          |
|   Repeat until budget exhausted or goal achieved         |
+----------------------------------------------------------+
```

Core components:

- **Surrogate model**: a low-cost predictive model (GNN, GP, etc.)
- **UQ method**: uncertainty quantification on predictions (ensemble, dropout, EDL, etc.)
- **Acquisition function**: selects the next experimental target based on uncertainty information
- **Oracle**: the expensive ground truth — a wet-lab assay, DFT calculation, etc.

### 4.2 Acquisition Functions — Balancing Exploration and Exploitation

| Acquisition Function | Strategy | Formula | Characteristic |
|:----|:-----|:-----|:-----|
| **Uncertainty Sampling** | Pure exploration | $a(\mathbf{x}) = \sigma(\mathbf{x})$ | Where uncertainty is greatest |
| **Expected Improvement (EI)** | Balanced | $a(\mathbf{x}) = \mathbb{E}[\max(f(\mathbf{x}) - f^+, 0)]$ | Expected gain over current best |
| **Upper Confidence Bound (UCB)** | Tunable | $a(\mathbf{x}) = \mu(\mathbf{x}) + \kappa \sigma(\mathbf{x})$ | $\kappa$ controls exploration-exploitation ratio |
| **Thompson Sampling** | Stochastic | $a(\mathbf{x}) = f_{\text{sample}}(\mathbf{x})$ | Sample from posterior, then select optimum |

Practical characteristics of each:

- **Uncertainty Sampling**: simplest to implement, but completely ignores the predicted value — can waste resources on inactive but uncertain molecules
- **EI**: the most widely used default in BO — weights improvement potential relative to the current best, concentrating search near already promising regions
- **UCB**: explicitly controls the exploration-exploitation ratio via $\kappa$ — a strategy of gradually reducing $\kappa$ as the project progresses can be effective
- **Thompson Sampling**: stochastic sampling from the posterior provides natural exploration — produces good diversity in batch selection

### 4.3 Case Studies — Real-World Results of UQ-Based Active Learning

**Case 1: HDAC Inhibitor Discovery (ACS Central Science, 2025)**

- **Method**: Multi-fidelity Bayesian Optimization
  - Low-fidelity: rapid docking score (low-cost, lower accuracy)
  - High-fidelity: FEP (Free Energy Perturbation) calculation or experimental assay
- **UQ's role**: acquisition function based on uncertainty from the multi-fidelity surrogate model to select targets for high-fidelity experiments
- **Result**: **significant hit rate improvement** over random screening, yielding potent HDAC inhibitors within a limited synthesis/testing budget
- **Lesson**: in a multi-fidelity setting, UQ is the key tool for deciding "when to run the expensive experiment"

**Case 2: CDK2/KRAS Inhibitors (Communications Chemistry, 2025)**

- **Challenge**: CDK2 and KRAS were considered "undruggable" — high-difficulty targets
- **Approach**: UQ-guided active learning to efficiently explore chemical space
- **Result**: out of 9 synthesized molecules, **8 were active** (hit rate 89%)
  - Among them, **1 achieved nM-level potency** — drug lead quality
- **Significance**: with synthesis and evaluation of just 9 molecules, a drug lead was identified against a challenging target — without UQ, random screening would have required hundreds to thousands of syntheses

**Case 3: BATCHIE — Drug Combination Screening (Nature Communications, 2024)**

- **Problem**: the search space for drug combinations is quadratic in the number of single agents — 1,000 drugs yield ~500,000 pairwise combinations
- **BATCHIE approach**: batch active learning to select the next drug combinations to test
  - UQ prioritizes combinations with the highest information gain
  - A repulsive term maintains batch diversity
- **Result**: **5-10x hit rate improvement** over random screening
- **Key insight**: the larger the search space, the greater the value of UQ-guided selection

### 4.4 Mathematical Foundations of Bayesian Optimization (Brief)

A concise overview of the key structure of Bayesian Optimization (BO), the formal framework underlying active learning:

**1. Surrogate Model**: a probabilistic model that approximates the expensive function $f(\mathbf{x})$

$$f(\mathbf{x}) \sim p(f \mid \mathcal{D})$$

- Classical choice: Gaussian Process (GP) — more recently: GNN with UQ, deep kernel learning
- **Key requirement**: the surrogate must provide both predictions and uncertainty

**2. Acquisition Function**: uses the surrogate's posterior to select the next experimental target

$$\mathbf{x}_{\text{next}} = \arg\max_{\mathbf{x}} \alpha(\mathbf{x} \mid \mathcal{D})$$

**3. Batch Strategy**: in practice, multiple molecules are synthesized and tested simultaneously

- Greedy batch selection: sequentially add phantom observations one at a time
- Diverse batch selection: ensure intra-batch diversity via DPP (Determinantal Point Process) or similar
- Thompson sampling: drawing multiple independent samples naturally yields a diverse batch

---

## 5. Lab-in-the-Loop — UQ Closes the Design-Make-Test Cycle

Section 4's active learning mostly addresses **single-round** settings: one round of acquisition selects the next experimental batch, and the results update the model. In reality, drug discovery is a **multi-round iterative design** workflow. This section goes beyond single-round active learning to examine how UQ serves as a core decision engine in iterative design cycles.

### 5.1 UQ in the DMTA Cycle — Four Decision Points

The basic unit of drug discovery is the **DMTA cycle** (Design-Make-Test-Analyze):

```
Design --> Make --> Test --> Analyze
  |         |        |         |
  |  2-4 wk |  3-6 wk|  1-2 wk|  1-2 wk
  |         |        |         |
  +---------+--------+---------+
         Total: 8-14 weeks per cycle
```

Traditionally, a single DMTA cycle takes 8-14 weeks. As reported by Hillisch et al. (*Drug Discovery Today*, 2024) through AstraZeneca's experience, predictive AI accelerates this cycle primarily through **information-driven decision-making at each stage**. UQ is what determines the quality of those decisions.

**UQ intervenes at four decision points**:

**1. Design — Which chemical space to explore**

- **Epistemic uncertainty-based space selection**: promising molecules may lie in regions where the model is uncertain
- Filter molecules proposed by generative models (VAE, diffusion, etc.) using UQ
- High predicted value + high epistemic uncertainty = **candidates with high exploration value**

**2. Make prioritization — When synthesis resources are limited**

- A synthetic chemist's time and reagents are finite resources
- Without UQ: synthesize in order of predicted activity
- With UQ: synthesize in order of **information gain** (UCB or Thompson sampling)
- High predicted value + low uncertainty = confirmation synthesis (exploitation)
- High uncertainty + reasonable predicted value = exploratory synthesis (exploration)

**3. Test sequencing — Maximizing information gain across multiple assays**

- Multiple assays can be run for a single molecule: binding, selectivity, ADME, toxicity
- Running all assays is prohibitively expensive and time-consuming
- UQ-based sequential decision: measure the most uncertain property first, then decide whether to proceed with or halt subsequent assays based on the result

**4. Learn — Model update and next-cycle strategy**

- After updating the model with experimental results, analyze how the **uncertainty map** has changed
- Use the pattern of epistemic uncertainty reduction to **determine the next cycle's strategy**:
  - If epistemic uncertainty has decreased substantially in a specific chemical region, that region is sufficiently learned — move on to other regions
  - If epistemic uncertainty remains high overall — more diverse molecular exploration is needed

**Key insight**: Section 4's active learning addresses only **point 1 (Design)** among these four. A real project requires UQ at all four.

### 5.2 Prescient Design's Lab-in-the-Loop — The Most Complete Example

Frey et al. (*bioRxiv*, 2025) reported Genentech/Roche's Prescient Design team's **lab-in-the-loop therapeutic antibody design** framework. This is one of the most fully realized real-world examples of multi-round UQ-driven iterative design.

**Architecture — Semi-autonomous design loop**:

```
+--------------------------------------------------------------+
|                LAB-IN-THE-LOOP WORKFLOW                        |
|                                                               |
|  Round 1 (No project data)                                    |
|  +-----------+   +----------+   +------------+               |
|  | Pre-trained|-->| Generate |-->| Predict +  |               |
|  | Foundation |   | Variants |   | UQ (high   |               |
|  | Model      |   | (6 mut.) |   | epistemic) |               |
|  +-----------+   +----------+   +-----+------+               |
|                                        |                      |
|                                        v                      |
|                              +------------------+             |
|                              | Select: maximize  |             |
|                              | EXPLORATION       |             |
|                              | (uncertainty      |             |
|                              |  sampling)        |             |
|                              +--------+---------+             |
|                                       |                       |
|                                       v                       |
|                              +------------------+             |
|                              | Synthesize + Test |             |
|                              | (~100 variants)   |             |
|                              +--------+---------+             |
|                                       |                       |
|                                       v                       |
|  Round 2-3 (Accumulating data)                                |
|  +-----------+   +----------+   +------------+               |
|  | Fine-tuned|-->| Generate |-->| Predict +  |               |
|  | Model     |   | Variants |   | UQ (lower  |               |
|  | (+ data)  |   | (8 mut.) |   | epistemic) |               |
|  +-----------+   +----------+   +-----+------+               |
|                                        |                      |
|                                        v                      |
|                              +------------------+             |
|                              | Select: mixed     |             |
|                              | UCB strategy      |             |
|                              +--------+---------+             |
|                                       |                       |
|                                       v                       |
|  Round 4 (Rich project data)                                  |
|  +-----------+   +----------+   +------------+               |
|  | Well-tuned|-->| Generate |-->| Predict +  |               |
|  | Model     |   | Variants |   | UQ (low    |               |
|  |           |   | (12 mut.)|   | epistemic) |               |
|  +-----------+   +----------+   +-----+------+               |
|                                        |                      |
|                                        v                      |
|                              +------------------+             |
|                              | Select: maximize  |             |
|                              | EXPLOITATION (EI) |             |
|                              +------------------+             |
+--------------------------------------------------------------+
```

**How the strategy evolves across rounds**:

- **Round 1**: no project-specific data available, only the pre-trained foundation model is used, epistemic uncertainty is high, **exploration-focused** (uncertainty sampling)
  - **Mutation budget: 6** — conservative exploration due to low model confidence
- **Round 2-3**: experimental data from prior rounds accumulates, the model is fine-tuned, epistemic uncertainty begins to decrease, **mixed strategy** (UCB)
  - **Mutation budget: 8** — exploration scope modestly expanded as model confidence grows
- **Round 4**: hundreds of project-specific data points, well-calibrated model, **exploitation-focused** (EI)
  - **Mutation budget: 12** — the model can provide reliable predictions across a wider sequence space

**The significance of progressively expanding the mutation budget**:

- More mutations enable exploration of a wider sequence space, but also increase the risk of extrapolation
- As UQ calibration improves, confidence in extrapolated predictions grows, justifying bolder exploration
- This is an elegant strategy in which **UQ confidence dynamically determines the scope of exploration**

### 5.3 Multi-Round UQ Evolution — The Value of Accumulating Project Data

**Epistemic Uncertainty Decay Curve**:

In theory, as rounds progress and project data accumulates, epistemic uncertainty should decrease:

```
Epistemic Uncertainty
    |
  H | xxxxxx
  i |       xxxx
  g |           xxx
  h |              xxx
    |                 xx        <- Aleatoric floor
  L | - - - - - - - - - -xxx- - - - - -
  o |                      xxxxxxxx
  w |
    +------------------------------------>
      R1     R2     R3     R4     R5
                  Round
```

- **Theoretical expectation**: epistemic uncertainty decreases monotonically after each round
- **What is actually observed**: generally decreasing, but **non-monotonic fluctuations** can occur
  - Temporary increase in epistemic uncertainty when entering a new chemical region
  - Catastrophic forgetting during fine-tuning
- **Aleatoric floor**: the irreducible noise that no amount of data can eliminate — experimental measurement error, biological variability

**Post-Fine-Tuning Overconfidence (Catastrophic Confidence)**:

A subtle but serious problem that arises when fine-tuning a foundation model on project data:

- **Phenomenon**: prediction performance improves after fine-tuning, but **UQ calibration degrades**
- **Cause**: overfitting to a small fine-tuning dataset narrows the posterior unrealistically, making the model "confident about everything"
- This is the reverse of the cold posterior effect discussed in Part 2: a **warm posterior problem when data is scarce**

**Solution — Continual Calibration**:

- Perform recalibration after each round using **held-out experimental data** (a portion of previous rounds' results)
- Reapply temperature scaling (Part 4) at every round
- Alternatively, reconstruct the conformal prediction wrapper at each round
- **Key point**: UQ calibration is not a one-time setup — it is a **dynamic property that must be maintained every round**

### 5.4 Methodological Challenges in Iterative Design

Multi-round lab-in-the-loop entails unique methodological challenges absent from single-round active learning.

**Challenge 1 — Cumulative Distribution Shift**

- At each round, the acquisition function selects a particular chemical space, shifting the training data distribution for the next round
- Cumulative effect: the chemical space in Round 4 may differ significantly from Round 1's training distribution
- Assumptions underlying the UQ model from earlier rounds become **systematically violated**

**Challenge 2 — Exchangeability Violation in Conformal Prediction**

- As discussed in Part 4, the core assumption of conformal prediction is that data points are **exchangeable**
- In lab-in-the-loop, compounds in each round are selected **conditionally** on previous rounds' results, violating exchangeability
- As a result, conformal prediction's coverage guarantee may no longer hold

**Alternative**: Angelopoulos et al. (*PNAS*, 2022) proposed **conformal prediction under feedback covariate shift**:

- Uses biomolecular design (protein engineering) as a concrete case study
- Applies importance weighting at each round to correct for covariate shift
- The corrected conformal prediction provides approximate coverage guarantees

**Challenge 3 — Determining the Exploration-to-Exploitation Transition Point**

- When should one switch from exploration (searching uncertain regions) to exploitation (optimizing the best candidates)?
- **Practical criterion**: when the median epistemic uncertainty across the entire candidate pool drops to the level of aleatoric uncertainty, it is a signal to shift toward exploitation
  - "There is little information left to gain from exploration" = "We already know enough of what can be known"

$$\text{Transition condition}: \text{median}[\sigma_{\text{epistemic}}(\mathbf{x})] \lesssim \mathbb{E}[\sigma_{\text{aleatoric}}(\mathbf{x})]$$

**Challenge 4 — Risk of Model Collapse**

- Over-exploitation leads to repeatedly learning similar compounds, causing the model to be calibrated only within a narrow chemical space, with UQ failing in new regions — **lost discovery opportunities**
- This is analogous to "mode collapse" in generative models

**Mitigation strategies**:

- **Periodic diversity injection**: even in exploitation-heavy rounds, include a fixed proportion (e.g., 20%) of diverse/random molecules
- **Thompson sampling**: stochastic sampling from the posterior maintains natural exploration, ensuring a baseline level of diversity at all times
- **Chemical diversity constraint**: set an upper bound on Tanimoto similarity within each batch

### 5.5 Practical Guide — Recommended Strategy by Round

Practical guidelines for lab-in-the-loop deployment:

| Round | Data State | Recommended Acquisition | UQ Role | Mutation Budget |
|:-----:|:----------:|:-----------------------:|:-------:|:---------------:|
| **1-2** | No/little project data | **Uncertainty Sampling** | Maximize exploration | Conservative (small) |
| **3-4** | 100-300 compounds | **UCB** (gradually decrease $\kappa$) | Balance exploration-exploitation | Medium |
| **5+** | 500+ compounds | **EI** + 20% diversity | Exploitation-focused | Expandable |

**Per-round checklist**:

- **At the start of each round**: update the model with previous round's experimental results + recalibrate UQ
- **At the end of each round**: assess the degree of epistemic uncertainty reduction and determine the next round's strategy
- **Warning sign**: epistemic uncertainty is not decreasing or is increasing — revisit model architecture or explore a new chemical region

**Practical limitations in small-scale projects**:

- In projects with fewer than 100 compounds, the statistical power of UQ-based iterative design is limited
- Minimum requirements: 20-30+ experimental data points per round, at least 2-3 rounds
- Alternative: leverage UQ from pre-trained foundation models as much as possible while fine-tuning conservatively

---

## 6. Bayesian Optimization in Materials Science

### 6.1 Target-Oriented BO — Searching for a Specific Value, Not a Maximum

Bayesian Optimization in materials science has unique characteristics that distinguish it from drug discovery. The most significant difference lies in the nature of the objective:

- **Drug discovery**: "**Maximize** binding affinity" — standard optimization
- **Materials science**: "Find a material with a bandgap of **exactly 1.8 eV**" — target-oriented search

In target-oriented BO, the acquisition function is modified as follows:

$$a_{\text{target}}(\mathbf{x}) = p(|f(\mathbf{x}) - y_{\text{target}}| < \epsilon)$$

This can be computed directly from the surrogate model's predictive distribution:

$$a_{\text{target}}(\mathbf{x}) = \Phi\left(\frac{y_{\text{target}} + \epsilon - \mu(\mathbf{x})}{\sigma(\mathbf{x})}\right) - \Phi\left(\frac{y_{\text{target}} - \epsilon - \mu(\mathbf{x})}{\sigma(\mathbf{x})}\right)$$

where $\Phi$ is the CDF of the standard normal distribution, and $\mu(\mathbf{x})$ and $\sigma(\mathbf{x})$ are the surrogate model's predicted mean and standard deviation.

**UQ's key role**: regions with high uncertainty also have a higher probability of containing the target value, so exploration and target matching are naturally coupled.

### 6.2 Multi-Fidelity BO — Combining Information Sources of Different Costs

In materials discovery, property calculations span a range of accuracy-cost levels:

```
Accuracy   Cost       Method
--------------------------
  ^      High     CCSD(T) / Experiment
  |               |
  |      Medium   DFT (hybrid functional)
  |               |
  |      Low      DFT (GGA functional)
  |               |
  v      Minimal  Semi-empirical / ML
--------------------------
```

**Multi-fidelity BO** exploits this hierarchy:

- Collect **low-cost/low-accuracy data** in large quantities to understand the overall landscape
- Collect **high-cost/high-accuracy data** only at a small number of key points guided by UQ
- The surrogate model learns the **correlation** between fidelities, effectively transferring low-fidelity information to improve high-fidelity predictions

**Mathematical structure**:

$$f_{\text{high}}(\mathbf{x}) = \rho \cdot f_{\text{low}}(\mathbf{x}) + \delta(\mathbf{x})$$

where $\rho$ is the inter-fidelity correlation coefficient and $\delta(\mathbf{x})$ is a correction function. In a GP-based multi-fidelity model, posteriors are maintained over both $\rho$ and $\delta$, so the acquisition function also determines **at which fidelity to perform the next calculation**.

### 6.3 Autonomous Laboratories — A-Lab and Robotic Synthesis

Autonomous laboratories represent the ultimate implementation of UQ-driven BO: a fully automated loop in which robots synthesize and AI designs the next experiment, with no human intervention.

**A-Lab** (Szymanski et al., *Nature*, 2023):

- **Setup**: powder-handling robots + diverse synthesis equipment (ball milling, furnace, etc.) + XRD analysis + AI controller
- **BO's role**: surrogate model over synthesis conditions (temperature, time, composition ratio) + acquisition function to determine the next experimental conditions
- **Result**: 41 novel inorganic materials synthesized in 17 days — automating months of work by a human researcher
- **UQ's importance**: failure cost is high in robotic experiments (time, reagents) — UQ selects the conditions that are "most likely to succeed" or "most informative," maximizing experimental efficiency

**Special UQ requirements in autonomous labs**:

- **Real-time decision-making**: the next experiment must be decided as soon as results arrive — fast UQ inference is essential (EDL or readout ensemble)
- **Safety constraints**: UQ-based safety filters for dangerous reaction conditions (high temperature, toxic reagents) — automatically block high-uncertainty hazardous conditions
- **Equipment constraints**: limited number of simultaneously available instruments — requires batch acquisition
- **Non-stationary noise**: as equipment conditions change over time, aleatoric uncertainty may vary — not i.i.d.
- **Feedback latency**: some experiments (e.g., annealing, sintering) take hours to produce results — asynchronous batch BO is needed

**Autonomous lab vs. human researcher comparison**:

```
                    Human Researcher       Autonomous Lab
Experimental throughput   ~5-10 / day          ~50-200 / day
UQ utilization           Intuition + experience Mathematical acquisition function
Failure cost             Reagents + time        Reagents + time + equipment damage risk
Search strategy          Experience-based heuristic  BO + calibrated UQ
Adapting to new domains  Literature + expertise  Epistemic signal from UQ
```

In autonomous labs, UQ's value goes beyond "which experiment to run next" to encompass **"what happens if this experiment fails."** High uncertainty + high potential value = an informative experiment. High uncertainty + high failure cost = an experiment to avoid. UQ makes both of these judgments quantitatively possible.

---

## 7. The Calibration Crisis — A Hidden Problem in Molecular ML

### 7.1 Discovery of Systematic Miscalibration

While the preceding sections championed the value of UQ, we must address one uncomfortable truth: **the majority of current molecular ML models are systematically miscalibrated.**

**Key finding from a PMC (2025) study**:

- When hyperparameter tuning is performed using **accuracy or AUC** (the current standard practice) for molecular property prediction, models become systematically **overconfident** or **underconfident**
- This means that the same miscalibration problem discussed in Part 0 for general deep learning also exists in molecular ML
- **An especially dangerous scenario**: a model exhibits high AUC while being severely miscalibrated — the user judges it a "good model," but its probabilistic predictions are unreliable

### 7.2 Why Accuracy-Based Tuning Hurts Calibration

**Root cause**:

- Accuracy/AUC optimization only optimizes the **position of the decision boundary**
- Calibration requires that the **predicted probability values** are meaningful
- These two objectives are independent: even with a perfect decision boundary (AUC = 1.0), the probability estimates can be completely wrong

**Intuitive example**:

```
Ground truth:    P(active) = 0.7

Model A (good AUC, bad calibration):
  Predicts: P(active) = 0.99  -> Correct decision, wrong probability
  
Model B (good AUC, good calibration):
  Predicts: P(active) = 0.72  -> Correct decision, right probability

In virtual screening:
  - Model A ranking -> order is correct, but
    estimates 99 hits in the top 100 (actual: 70)
  - Model B ranking -> order is correct, and
    estimates 72 hits in the top 100 (actual: 70)
```

Using Model A causes decision-makers to allocate resources based on **unrealistic expectations**.

### 7.3 Solutions — Calibration-Aware Training

**Approach 1: Tune with BCE Loss**

- Binary Cross-Entropy (BCE) is a **proper scoring rule** — optimization naturally improves calibration
- Replace the standard accuracy/AUC-based early stopping with **validation BCE for early stopping**

$$\text{BCE} = -\frac{1}{N}\sum_{i=1}^{N} [y_i \log \hat{p}_i + (1-y_i) \log(1-\hat{p}_i)]$$

**Approach 2: Tune with Adaptive Calibration Error (ACE)**

- An improved version of ECE (Expected Calibration Error) from Part 4
- Adaptive binning evaluates calibration even in low-density regions
- Use ACE directly as the optimization target during hyperparameter selection

**Approach 3: HMC Bayesian Last Layer — An Efficient Alternative**

- No need to make the entire model Bayesian — apply HMC Bayesian treatment to the **last layer only**
- An extension of Laplace Redux (Daxberger et al., 2021) discussed in Part 2
- Can be applied to an already-trained model with minimal additional cost
- Empirically yields significant calibration improvements

### 7.4 Calibration Diagnostics — Reading Reliability Diagrams in Molecular ML

Here we concretize the reliability diagrams from Part 4 in the molecular ML context.

**Ideal calibration for a molecular ML model**:

- If the model predicts "this molecule has a 90% probability of being active," then 90% of such molecules should indeed be active
- For regression: a predicted 95% interval $\hat{y} \pm 2\sigma$ should contain 95% of observed values

**Common miscalibration patterns in molecular ML**:

```
Reliability Diagram (Classification)

Observed Fraction
    |
1.0 |                           /
    |                         /
0.8 |                       /
    |                    . / .   <- Typical molecular ML model
0.6 |                 .  /       (underconfident in middle,
    |              .   /          overconfident at extremes)
0.4 |           .    /
    |        .     /
0.2 |     .      /
    |   .      / <- Perfect calibration
0.0 |.       /
    +--------------------------->
    0.0  0.2  0.4  0.6  0.8  1.0
         Mean Predicted Probability
```

- **Low probability region (0.0-0.3)**: generally well calibrated — the model assigns low probabilities to inactive molecules
- **High probability region (0.7-1.0)**: systematic overconfidence — even when the model says "95% confident," the actual hit rate is 70-80%
- **Middle region (0.3-0.7)**: slightly underconfident — the model expresses uncertainty, but reality is more decisive

The cause of this pattern: cross-entropy loss optimization focuses on making probabilities accurate **near the decision boundary** and tends to neglect calibration in the extreme regions.

### 7.5 Summary of Practical Recommendations

1. **Include calibration metrics in hyperparameter selection**: choose from the Pareto front of AUC + ECE — a model with slightly lower AUC but markedly better ECE may be more useful in practice
2. **Apply post-hoc calibration as a default**: at minimum, always perform temperature scaling — significant calibration improvement from a single parameter
3. **Monitor calibration periodically**: when data distribution changes (temporal shift, new assay), recalibration is needed — calibration is a dynamic property, not a static one
4. **Do not trust probability outputs from models without UQ**: softmax probability is not equal to calibrated probability — make this distinction explicit within your team
5. **Check calibration in regression too**: periodically compute the empirical coverage of prediction intervals — verify that a 95% interval actually contains 95% of observations
6. **Include calibration results in reports**: build a culture of always reporting ECE/coverage alongside AUC/RMSE when presenting model performance

---

## Closing: UQ Determines the Speed of Scientific Discovery

Here we distill the key insights from the seven sections covered in this post.

**1. UQ's value is maximized under resource-constrained settings**

- When the number of molecules that can be synthesized in virtual screening is limited
- When experimental time and reagents in an autonomous lab are finite
- When each round's results in a multi-round design determine the next round's strategy
- UQ provides the answer to "what to do first" in all of these situations

**2. Bayesian uncertainty and learned confidence are fundamentally different**

- Bayesian UQ (MC Dropout, ensemble, EDL): uncertainty derived from the posterior, with aleatoric/epistemic separation
- Learned confidence (AlphaFold pLDDT): a self-assessment learned in a self-supervised manner, unable to distinguish the source of uncertainty
- Both are useful, but they differ in the depth of information they provide for decision-making

**3. Distribution shift is the real test for UQ**

- Good UQ performance under random split does not reflect reality
- UQ performance under temporal split and scaffold split determines real-world value
- Surprisingly simple approaches like error models (meta-models) can be remarkably robust

**4. Single-round active learning is only the beginning**

- Real scientific discovery is a multi-round iterative process
- Lab-in-the-loop requires UQ to operate at every stage of the DMTA cycle
- Unique challenges exist: cumulative distribution shift, exchangeability violation, risk of model collapse
- However, methodologies for addressing these challenges are advancing rapidly

**5. UQ without calibration is dangerous**

- Even if uncertainty estimates exist, miscalibrated estimates lead to wrong decisions
- Calibration is not a one-time setup but a property that must be continuously maintained
- In multi-round design, recalibration at every round is essential

When I published Bayesian GCN in 2019, UQ was a "nice-to-have" add-on in the molecular ML community. As of 2026, UQ has become a **must-have** core component. Just as AlphaFold's pLDDT became the standard for structure prediction, we are rapidly approaching an era in which every molecular and materials prediction model is expected to provide calibrated uncertainty by default.

In the next Part 7, we turn to the final destination of this journey — **UQ in the Foundation Model Era**. We will examine what new challenges arise in quantifying uncertainty for billion-parameter LLMs, protein language models, and molecular foundation models, and survey emerging solutions from Bayesian LoRA to conformal prediction for foundation models.

---

## References

### Author's Work
1. Ryu, S., Kwon, Y. & Kim, W. Y. (2019). "A Bayesian Graph Convolutional Network for Reliable Prediction of Molecular Properties with Uncertainty Quantification." *Chemical Science*, 10, 8438-8446.

### Molecular Property Prediction UQ
2. Hirschfeld, L., Swanson, K., Yang, K., Barzilay, R. & Coley, C. W. (2020). "Uncertainty Quantification Using Neural Networks for Molecular Property Prediction." *Journal of Chemical Information and Modeling*, 60(8), 3770-3780.
3. Scalia, G., Grambow, C. A., Perniciani, B., Li, Y. P. & Green, W. H. (2020). "Evaluating Scalable Uncertainty Estimation Methods for Deep Neural Network-Based Molecular Property Prediction." *Journal of Chemical Information and Modeling*, 60(6), 2697-2717.
4. JCIM (2026). Benchmark study on UQ under temporal distribution shift for ADME endpoint prediction. *Journal of Chemical Information and Modeling*.

### AlphaFold and Protein Structure
5. Jumper, J. et al. (2021). "Highly Accurate Protein Structure Prediction with AlphaFold." *Nature*, 596, 583-589.
6. Terwilliger, T. C. et al. "AlphaFold predictions are useful for validating and improving experimental structures." *PNAS*.

### Neural Network Potentials
7. Batatia, I. et al. (2023). "A foundation model for atomistic simulations." *arXiv:2401.00096*.
8. Merchant, A. et al. (2023). "Scaling deep learning for materials discovery." *Nature*, 624, 80-85.
9. Nature Communications (2025). "Evidential Deep Learning for Uncertainty-Aware Interatomic Potentials."

### Active Learning and Bayesian Optimization
10. Soleimany, A. P. et al. (2021). "Evidential Deep Learning for Guided Molecular Property Prediction and Discovery." *ACS Central Science*, 7(8), 1356-1367.
11. ACS Central Science (2025). Multi-fidelity Bayesian optimization for HDAC inhibitor discovery.
12. Communications Chemistry (2025). UQ-guided active learning for CDK2/KRAS inhibitor discovery.
13. Nature Communications (2024). "BATCHIE: Batch active learning for drug combination screening."

### Lab-in-the-Loop and Iterative Design
14. Frey, N. et al. (2025). "Lab-in-the-loop therapeutic antibody design with deep learning." *bioRxiv*.
15. Angelopoulos, A. N. et al. (2022). "Conformal prediction under feedback covariate shift for biomolecular design." *Proceedings of the National Academy of Sciences*.
16. Hillisch, A. et al. (2024). "Augmenting the design-make-test-analyse cycle using predictive AI modelling at AstraZeneca." *Drug Discovery Today*, 29(2), 103830.

### Calibration
17. Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. Q. (2017). "On Calibration of Modern Neural Networks." *ICML*.
18. PMC (2025). Calibration study in molecular machine learning — systematic miscalibration from accuracy/AUC-based hyperparameter tuning.
19. Daxberger, E. et al. (2021). "Laplace Redux — Effortless Bayesian Deep Learning." *NeurIPS*.

### Bayesian Deep Learning Methods (from earlier parts)
20. Gal, Y. & Ghahramani, Z. (2016). "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning." *ICML*.
21. Gal, Y., Hron, J. & Kendall, A. (2017). "Concrete Dropout." *NeurIPS*.
22. Lakshminarayanan, B., Pritzel, A. & Blundell, C. (2017). "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles." *NeurIPS*.

### Autonomous Labs
23. Szymanski, N. J. et al. (2023). "An autonomous laboratory for the accelerated synthesis of novel materials." *Nature*, 624, 86-91.
