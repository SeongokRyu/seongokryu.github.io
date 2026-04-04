---
title: "Synthesis AI Part 2: Reaction Prediction — Can AI Predict What Chemistry Will Do?"
date: 2026-04-04 09:10:00 +0900
categories: [Drug Discovery, Small Molecule Design]
tags: [reaction-prediction, template-based, template-free, graph-neural-network, synthesis]
math: true
---

**AI-Driven Synthesis in Drug Discovery**

*This is Part 2 of a 5-part series on AI-driven synthesis in drug discovery.*

- **Part 1:** [The Synthesis Bottleneck — Why "Make" Lags Behind](/posts/synthesis-ai-part1-synthesis-bottleneck/)
- **Part 2** (this post): Reaction Prediction — Can AI Predict What Chemistry Will Do?
- **Part 3:** [Retrosynthesis — Can AI Plan How to Make a Molecule?](/posts/synthesis-ai-part3-retrosynthesis/)
- **Part 4:** [Synthesis-Aware Design — Making AI-Generated Molecules Makeable](/posts/synthesis-ai-part4-synthesis-aware-design/)
- **Part 5:** [From Algorithm to Lab — CRO Integration and the Remaining Gap](/posts/synthesis-ai-part5-algorithm-to-lab/)


---

## 1. Introduction: Forward Prediction as the Foundation

In Part 1, we examined the synthesis bottleneck in drug discovery and the data landscape that fuels AI approaches. Now we turn to the most fundamental computational question in chemistry: **given reactants and reagents, what product will form?**

This is forward reaction prediction, and it is the atomic building block on which everything else in this series rests:

- **Retrosynthesis** (Part 3) runs forward prediction in reverse
- **Synthesis planning** chains multiple forward steps into routes
- **Synthesis-aware generation** (Part 4) uses forward models to score feasibility
- **Lab execution** (Part 5) depends on accurate predictions to avoid wasted experiments

Think of it this way: if we cannot reliably predict the outcome of a single reaction step, there is no hope of stringing together a multi-step synthesis route or filtering generated molecules for synthesizability. Forward prediction is the foundation.

But "reaction prediction" is not a single question. It is really three distinct questions bundled together:

| Question | Input | Output | Difficulty |
|---|---|---|---|
| **What product forms?** | Reactants + reagents | Major product structure | Moderate (current focus) |
| **Under what conditions?** | Desired transformation | Solvent, catalyst, temp, etc. | Hard (under-explored) |
| **How well does it work?** | Reaction + conditions | Yield (%) | Very hard (data-sparse) |

Most research has focused almost exclusively on the first question. As we will see, this creates a critical gap between what models predict and what chemists actually need to plan a synthesis campaign.

The data landscape from Part 1 sets the stage here: USPTO provides the bulk of training data (~1.8M reactions), but it records mainly reactants and products — conditions are sparse, yields are often missing, and the reaction distribution skews toward patent-worthy novelty rather than everyday medicinal chemistry.

Let us walk through the three dominant paradigms, where they excel, and where they fall short.

---

## 2. Three Paradigms of Reaction Prediction

The field has converged on three fundamentally different ways to approach forward reaction prediction. Each encodes chemical knowledge differently, and each makes different trade-offs.

```
┌──────────────────────────────────────────────────────────────────────┐
│              Three Paradigms of Reaction Prediction                  │
├──────────────────────┬──────────────────────┬───────────────────────┤
│   TEMPLATE-BASED     │  TEMPLATE-FREE       │   GRAPH-BASED         │
│                      │  (Seq2seq)           │                       │
│  Reactants           │  Reactants           │  Reactants            │
│     │                │     │                │     │                 │
│     ▼                │     ▼                │     ▼                 │
│  ┌────────┐          │  ┌────────┐          │  ┌────────┐           │
│  │ Match  │          │  │Tokenize│          │  │ Build  │           │
│  │template│          │  │ SMILES │          │  │ graph  │           │
│  └───┬────┘          │  └───┬────┘          │  └───┬────┘           │
│      ▼               │      ▼               │      ▼                │
│  ┌────────┐          │  ┌────────┐          │  ┌────────┐           │
│  │ Apply  │          │  │Encoder │          │  │  GNN   │           │
│  │template│          │  │Decoder │          │  │layers  │           │
│  └───┬────┘          │  └───┬────┘          │  └───┬────┘           │
│      ▼               │      ▼               │      ▼                │
│  ┌────────┐          │  ┌────────┐          │  ┌────────┐           │
│  │Validate│          │  │Decode  │          │  │Predict │           │
│  │ & rank │          │  │SMILES  │          │  │bond Δ  │           │
│  └───┬────┘          │  └───┬────┘          │  └───┬────┘           │
│      ▼               │      ▼               │      ▼                │
│   Product            │   Product            │   Product             │
│  (guaranteed         │  (may be             │  (structurally        │
│   valid)             │   invalid)           │   grounded)           │
└──────────────────────┴──────────────────────┴───────────────────────┘
```

### 2.1 Template-Based: Matching Known Reaction Patterns

The oldest and most chemically intuitive approach encodes reactions as **transformation rules** — essentially SMARTS patterns that describe which bonds break and form.

The workflow is straightforward:

- Extract reaction templates from a database of known reactions
- For a new set of reactants, find all templates whose reactant pattern matches
- Apply each matching template to generate candidate products
- Rank candidates using a learned scoring function

**Template-based methods guarantee chemically valid outputs because every transformation is derived from an observed reaction.** This is their defining strength and the reason many chemists still trust them.

Key tools and approaches include:

- **RDChiral** (Coley et al., JCIM, 2019): handles stereochemistry-aware template extraction and application using RDKit, producing chirally correct products
- **Reaction fingerprint methods**: encode reactions as fixed-length vectors for similarity search and classification
- **Rule-based expert systems**: hand-curated rules from decades of organic chemistry knowledge (the Synthia/Chematica lineage)

To make this concrete, consider how a template works:

```
Example: Amide Bond Formation Template (SMARTS)

  Pattern:   [C:1](=O)[OH] . [N:2]([H])  >>  [C:1](=O)[N:2]
  English:   "A carboxylic acid reacting with an amine forms an amide bond"

  Step 1:  Match the pattern against input reactants
  Step 2:  If matched → apply the transformation (remove OH, remove NH, form C-N)
  Step 3:  Output the product with correct atom mapping
```

The limitations, however, are fundamental:

- **Coverage ceiling**: if a reaction type is not in the template library, the model cannot predict it — zero generalization to novel chemistry
- **Template explosion**: medicinal chemistry alone involves thousands of reaction types, each with substrate-specific variants; comprehensive libraries can contain 100K+ templates
- **Maintenance burden**: templates need curation, validation, and updating as new chemistry is published
- **Selectivity blindness**: when multiple sites on a molecule match the same template, the method cannot easily predict which site reacts preferentially

### 2.2 Template-Free (Seq2seq): Chemistry as Translation

The template-free paradigm reframes reaction prediction as a **sequence-to-sequence translation problem**. Reactant SMILES go in, product SMILES come out — just like translating English to French.

```
Input:   CC(=O)Cl . c1ccc(N)cc1  >>  ?

                    ┌─────────────┐
  Reactant SMILES ──┤  Transformer ├──► Product SMILES
                    │  Encoder-   │
                    │  Decoder    │    CC(=O)Nc1ccccc1
                    └─────────────┘
```

The breakthrough came with the **Molecular Transformer** (Schwaller et al., ACS Cent Sci, 2019), which applied the Transformer architecture — the same architecture behind GPT and BERT — to chemical reactions. The key insight was that SMILES strings, despite being designed for human readability, contain enough structural information for a Transformer to learn chemical transformation patterns.

**The Molecular Transformer demonstrated that treating SMILES as a "chemical language" could achieve >90% top-1 accuracy on standard benchmarks, rivaling or exceeding template-based methods.** This was a watershed moment for the field.

The approach has several compelling advantages:

- **No template extraction needed** — learns directly from reaction SMILES pairs
- **Can generalize** to reaction types not explicitly seen during training (in principle)
- **Scalable** — adding more training data improves performance without manual curation

But it also introduced new failure modes:

- **Invalid SMILES generation**: the decoder can produce strings that do not parse into valid molecules (~3-5% of predictions)
- **Black-box reasoning**: no explicit representation of which bonds break or form, making error diagnosis difficult
- **Hallucination risk**: can generate plausible-looking but chemically nonsensical products

Building on this foundation, **Chemformer** (Irwin et al., Mach Intell, 2022) from AstraZeneca introduced pre-training on large unlabeled chemical corpora before fine-tuning on reaction data. This follows the same pre-train-then-fine-tune recipe that made BERT dominant in NLP.

Chemformer's contributions:

- Pre-trained on ZINC and PubChem (~100M molecules) using masked SMILES reconstruction
- Fine-tuned on USPTO reaction data
- Achieved competitive accuracy with fewer labeled reaction examples
- Released as open-source, accelerating adoption in pharma

IBM's **RXN for Chemistry** (2020) took the Molecular Transformer to production, offering a cloud API that any chemist can use. This is notable because it demonstrated that seq2seq reaction prediction could move from a research prototype to an industrial tool — even if adoption remains limited by the accuracy gap we discuss in Section 4.

### 2.3 Graph-Based: Predicting Bond Changes Directly

Graph-based methods take a middle path: instead of matching templates or translating strings, they operate directly on the **molecular graph** and predict which bonds change.

The core idea is chemically elegant:

- Represent reactants as a molecular graph (atoms = nodes, bonds = edges)
- Use a graph neural network (GNN) to learn atom and bond representations
- Predict the **reaction center** — which atoms and bonds are involved in the transformation
- Apply the predicted bond changes to the reactant graph to produce the product graph

**LocalRetro** (Chen and Jung, Chem Sci, 2021) exemplifies this approach. It identifies reaction centers by analyzing **local atom environments** — the immediate neighborhood of each atom — and predicts transformations based on these local contexts.

Key strengths of graph-based methods:

- **Structurally grounded**: predictions are expressed as bond changes on real molecular structures
- **Atom mapping internalized**: the model implicitly learns which atoms in reactants correspond to which atoms in products
- **Interpretable reaction centers**: we can visualize exactly where the model thinks the reaction happens

Limitations remain:

- **Stereochemistry handling**: encoding and predicting 3D chirality in 2D graphs is non-trivial; many graph-based models struggle with stereocenters
- **Multi-component reactions**: reactions involving three or more components are harder to represent as graph edits
- **Scalability to complex transformations**: rearrangements that change the molecular skeleton are difficult to express as local bond edits

A useful way to think about graph-based methods: they sit conceptually between template-based and template-free approaches. Like templates, they reason about specific bond changes. Like seq2seq models, they learn these changes from data rather than hand-coded rules. This hybrid character makes them attractive for interpretability-sensitive applications, but the stereochemistry limitation is a real barrier for drug discovery where chiral centers are ubiquitous.

### 2.4 Head-to-Head Comparison

| Criterion | Template-Based | Template-Free (Seq2seq) | Graph-Based |
|---|---|---|---|
| **Top-1 Accuracy (USPTO)** | 85-90% | 90-93% | 88-92% |
| **Generalization to novel rxns** | None (limited to library) | Moderate (learned patterns) | Moderate (local edits) |
| **Interpretability** | High (template is readable) | Low (black-box decoder) | Medium (reaction center visible) |
| **Chemical validity** | 100% (by construction) | 95-97% (SMILES may be invalid) | ~99% (graph edits preserve valence) |
| **Stereochemistry** | Good (RDChiral) | Variable (often ignored) | Weak (2D graph limitation) |
| **Scalability** | Limited by library size | Scales with data | Scales with data |
| **Key limitation** | Cannot go beyond known templates | Can hallucinate invalid products | Struggles with complex rearrangements |
| **Representative work** | RDChiral (Coley, 2019) | Molecular Transformer (Schwaller, 2019) | LocalRetro (Chen & Jung, 2021) |

The trend line is clear: **template-free methods are winning on benchmark accuracy**, and the field is moving in that direction. But as we will see in Section 4, benchmark accuracy is not the whole story.

---

## 3. The Condition Prediction Gap

### 3.1 Why Product Prediction Alone Is Insufficient

Suppose a model correctly predicts that reactant A and reactant B will form product C. A medicinal chemist's immediate follow-up question is: **"Great, but under what conditions?"**

The same reaction can give wildly different outcomes depending on the conditions:

```
Same Reaction, Different Conditions:
───────────────────────────────────────────────────────────────────
  Suzuki coupling: ArBr + ArB(OH)2  →  Ar-Ar

  Condition Set A                    Condition Set B
  ─────────────                      ─────────────
  Catalyst: Pd(PPh3)4               Catalyst: Pd(OAc)2 / XPhos
  Solvent:  DMF / H2O               Solvent:  THF / H2O
  Base:     K2CO3                    Base:     Cs2CO3
  Temp:     80°C                     Temp:     rt (25°C)
  Time:     16 h                     Time:     2 h
  ─────────────                      ─────────────
  Yield:    35%                      Yield:    92%
  ─────────────                      ─────────────
```

**The difference between a 35% yield and a 92% yield is not a minor optimization detail — it is the difference between a failed campaign and a viable drug candidate.** In early-stage drug discovery, where tens of milligrams of compound are needed for biological assays, a low-yielding reaction can block an entire program.

### 3.2 Conditions That Matter

The space of reaction conditions is vast and highly combinatorial:

- **Solvent**: from dozens of common options (DMF, DMSO, THF, DCM, MeOH, toluene, water, etc.)
- **Temperature**: continuous, typically 0-200 degC
- **Catalyst/ligand**: hundreds of metal catalysts and ligand combinations
- **Base/acid**: inorganic vs organic, strength, stoichiometry
- **Atmosphere**: air, N2, Ar (matters for sensitive chemistry)
- **Concentration**: dilute vs concentrated can change selectivity
- **Time**: minutes to days
- **Additives**: phase-transfer catalysts, molecular sieves, etc.

This is a mixed discrete-continuous optimization problem with strong interactions between variables (e.g., catalyst choice constrains viable solvents).

### 3.3 Current Approaches

The field has begun to address this gap, but progress is early-stage:

- **Reaction condition recommendation** (Maser et al., ACS Cent Sci, 2021): given a reaction type, recommend a ranked list of condition sets using a neural network trained on historical reaction data
- **Yield prediction** (Schwaller et al., Mach Intell, 2021): given a reaction and its conditions, predict the yield as a continuous value; useful for ranking condition sets but requires conditions as input
- **DRFP** (Probst et al., Digital Discovery, 2022): differential reaction fingerprints that encode both transformation and conditions, enabling condition-aware similarity search
- **Open Reaction Database (ORD)** (Kearnes et al., JACS, 2021): a standardized schema for recording reactions with full conditions and outcomes — the data infrastructure that condition prediction models desperately need

### 3.4 The Core Gap

We can visualize the disconnect between model capability and chemist need:

```
What Models Currently Predict        What Chemists Need to Know
─────────────────────────────        ──────────────────────────

  Reactants ──► Product              Reactants ──► Product
      (structure only)                    │
                                          ├──► Conditions (which?)
                                          ├──► Yield (how much?)
                                          ├──► Selectivity (which isomer?)
                                          ├──► Scalability (mg → g → kg?)
                                          └──► Purification (how to isolate?)

          ◄── Current AI ──►                    ◄── Full Picture ──►
```

**The transition from "this reaction works" (binary classification) to "under these conditions, at this yield" (continuous, multi-dimensional prediction) is the most critical unsolved problem in computational reaction prediction.** Until we close this gap, forward prediction models remain useful for suggesting what is possible but insufficient for telling chemists what to actually do.

The ORD initiative is particularly important here. Before we can train condition-prediction models, we need data in a standardized format that captures the full experimental context — not just the reactants and products that patent filings emphasize.

Consider the data hierarchy:

```
Data Richness for Reaction Prediction:

  Level 1:  Reactants → Product                 (USPTO, most models)
  Level 2:  Reactants + Conditions → Product     (sparse, some ORD)
  Level 3:  Reactants + Conditions → Product +   (very sparse)
            Yield + Selectivity
  Level 4:  Full experimental protocol with       (almost nonexistent
            scale, purification, analytical data    in ML-ready format)
```

Most models operate at Level 1. Chemists need Level 3 or 4. The gap is not just a modeling challenge — it is fundamentally a data challenge. We will revisit this in Part 5 when we discuss the hand-off from algorithm to laboratory.

---

## 4. Benchmark Performance vs Real-World Utility

### 4.1 The 90% Illusion

On the standard USPTO benchmark, top-performing models achieve >90% top-1 accuracy. This is impressive. It is also misleading.

**When medicinal chemists evaluate these same models on their own reactions, perceived accuracy drops to roughly 70-80% — and that number is generous.** The gap is not a single issue but a convergence of multiple systematic biases.

### 4.2 Root Causes of the Gap

**Distribution shift.** USPTO reactions come from patents. Patent chemistry has a very different distribution from real medicinal chemistry:

- Patents over-represent novel reaction types (that is why they are patented)
- Patents under-represent routine transformations (amide couplings, Boc deprotections)
- Patent substrates tend to be simpler than real drug intermediates
- The 50-reaction toolkit a typical med chem team uses daily is poorly represented

**Atom mapping noise.** Most training data relies on automated atom mapping tools (e.g., RXNMapper by Schwaller et al., Sci Adv, 2021). These tools are good but not perfect:

- ~5-10% of mappings in USPTO contain errors
- Models trained on noisy mappings learn noisy patterns
- Error compounds through multi-step predictions

**Multi-product simplification.** Real reactions often produce multiple products — regioisomers, stereoisomers, side products. Benchmarks evaluate only the major product:

- A model that predicts the correct major product but misses a significant side product gets full credit
- Chemists care about selectivity, not just the top product

**Stereochemistry blindness.** Many models either ignore stereochemistry entirely or handle it poorly:

- Dropping chirality from SMILES during preprocessing is common
- Models that do handle stereochemistry often confuse R/S or E/Z assignments
- For drug molecules, where a single stereocenter can determine efficacy vs toxicity, this is not acceptable

### 4.3 Benchmark vs Reality

| Metric | Benchmark Score | Real-World Assessment | Gap Explanation |
|---|---|---|---|
| **Top-1 accuracy** | 90-93% | 70-80% (estimated) | Distribution shift from patents to med chem |
| **Chemical validity** | 95-100% | 90-95% | Edge cases in complex substrates |
| **Stereochemistry** | Often not evaluated | Poor to moderate | Most models drop or mishandle chirality |
| **Reaction conditions** | Not evaluated | Not predicted | Models only predict product structure |
| **Multi-product** | Major product only | All products matter | Selectivity not captured in benchmarks |
| **Yield prediction** | MAE ~10-15% | Highly variable | Sparse condition data, high noise |
| **Out-of-distribution** | Rarely tested | Significant failures | Models memorize USPTO distribution |

### 4.4 Where Are We Practically?

The honest assessment: **current forward prediction models function at the level of an experienced chemist's sanity check, not as decision-replacement tools.**

They are useful for:

- Quickly screening whether a proposed reaction is plausible
- Suggesting possible products for brainstorming
- Filtering obviously infeasible transformations in automated planning

They are not yet reliable for:

- Replacing a chemist's judgment on reaction feasibility
- Predicting exact outcomes for complex, multi-functional substrates
- Guiding condition optimization without additional experimental input

This is not a failure of AI — it reflects the genuine difficulty of chemistry. But it means we should calibrate expectations accordingly.

---

## 5. The LLM Frontier

### 5.1 Chemistry Meets Large Language Models

The explosive growth of large language models has inevitably reached chemistry. The question is whether LLMs bring something genuinely new to reaction prediction or merely repackage existing capabilities.

**The distinction between "understanding SMILES" and "understanding chemistry" is critical here.** A model that can manipulate SMILES strings fluently is not necessarily reasoning about electron density, orbital symmetry, or steric effects. But it might not need to — if pattern matching over SMILES is sufficient for practical accuracy.

### 5.2 Current Approaches

Several directions are emerging, each with a different philosophy:

**LLM as orchestrator.** ChemCrow (Bran et al., Nat Mach Intell, 2024) uses an LLM agent to orchestrate chemistry-specific tools — reaction predictors, property calculators, literature search — through natural language reasoning. The LLM does not predict reactions itself; it decides which tool to call and how to interpret the result. This is the "tool-use" paradigm: the LLM provides reasoning and planning, while specialized models handle the chemistry.

**Generative mechanism prediction.** Electron Flow Matching (Joung, Fong, and Coley, Nature, 2025) takes a fundamentally different approach. Instead of predicting just the product, it generates full **reaction mechanisms** — the step-by-step electron flow from reactants to products. This is arguably the most scientifically ambitious direction: it asks models to learn the *why* of chemistry, not just the *what*.

```
Traditional Model:          Mechanism-Aware Model:

  A + B → C                  A + B → [TS1] → Int → [TS2] → C
  (black box)                (electron flow at each step)
```

**Chemistry-specialized LLMs.** Various groups (2024-2026) have fine-tuned foundation models on chemical literature, reaction databases, and textbook knowledge. Early results show improved reaction-related question answering but mixed results on quantitative prediction tasks. The jury is still out on whether these models truly learn chemistry or merely memorize correlations in chemical text.

### 5.3 What LLMs Can and Cannot Do

| Capability | LLM Status | Assessment |
|---|---|---|
| Reaction product prediction | Moderate | Comparable to smaller specialized models, not yet superior |
| Condition recommendation | Promising | Can leverage literature knowledge, but lacks quantitative precision |
| Mechanistic reasoning | Early stage | Electron Flow Matching is a breakthrough, but not yet LLM-native |
| Literature synthesis | Strong | Summarizing reaction precedent from papers is a genuine strength |
| Multi-step planning | Promising | Natural language reasoning about strategy, but search is still needed |
| Yield prediction | Weak | Numerical regression is not an LLM strength |

### 5.4 Looking Ahead: Multi-Modal Reaction Models

The most exciting frontier is **multi-modal models** that combine:

- **Molecular graphs** for structural reasoning
- **Reaction conditions** as structured input
- **Natural language** for context and constraints
- **3D conformations** for steric and electronic effects

**The ultimate goal is a model that takes a chemist's natural language description — "I need to couple these two fragments under mild conditions compatible with this acid-sensitive protecting group" — and returns a ranked list of fully specified reaction protocols.** We are not there yet, but the pieces are coming together.

---

## 6. Convergence and Open Questions

### 6.1 Where the Field Stands

Three trends are clear:

- **Template-free methods have won the accuracy race.** Transformer-based seq2seq models consistently outperform template-based and graph-based approaches on benchmarks. The field is converging on this paradigm for product prediction.

- **Interpretability remains unsolved.** Chemists want to understand *why* a model predicts a given product — which bonds break, which form, and what mechanistic pathway is implied. Black-box seq2seq models cannot provide this. Graph-based methods and mechanistic approaches like Electron Flow Matching offer a path forward.

- **Condition prediction is the critical bottleneck.** Product prediction alone is necessary but not sufficient. The field needs to move from "what forms?" to "how do I make it work?" — and this requires both better models and better data (ORD).

### 6.2 Open Questions

Key unresolved challenges for the next 2-3 years:

- **Out-of-distribution robustness**: can models predict outcomes for reaction types not in USPTO? Glorius and colleagues have led systematic evaluation here, and the results are sobering — performance drops sharply for novel reaction classes.
- **Condition-yield co-prediction**: can we jointly predict optimal conditions and expected yield? This requires richer training data (ORD) and models that handle mixed discrete-continuous outputs.
- **Stereochemical accuracy**: can we reach chemist-level performance on chirality prediction? Drug molecules are overwhelmingly chiral; ignoring stereochemistry is not an option for real applications.
- **Uncertainty quantification**: can models tell us when they do not know? A model that confidently predicts the wrong product is more dangerous than one that says "I am not sure."
- **Mechanism-grounded prediction**: will models that learn electron flow outperform pattern-matching approaches? Electron Flow Matching (Joung, Fong, and Coley, Nature, 2025) suggests yes, but the approach is still young.
- **Data quality and standardization**: can the community move beyond USPTO to curated, condition-rich datasets? The ORD is a start, but coverage remains limited.

### 6.3 Bridge to Part 3

**Reaction prediction gives us the forward direction — reactants to products. In Part 3, we reverse it: given a target molecule, can AI figure out how to make it?**

This is retrosynthesis, and it turns out to be a fundamentally harder problem. Forward prediction asks "what will happen?" — a single-answer question. Retrosynthesis asks "what *could* have led here?" — a combinatorial explosion of possible pathways, each composed of multiple forward steps.

The connection is direct: every retrosynthesis engine needs a forward model (or its inverse) as a core component. The accuracy, speed, and coverage of the forward model constrain what the retrosynthesis planner can achieve. We will see how the models from this Part become building blocks for the search algorithms of Part 3.

---

**Next**: Part 3 — *Retrosynthesis: Can AI Design the Route Backward?*

---

### References

- Schwaller, P. et al. "Molecular Transformer: A Model for Uncertainty-Calibrated Chemical Reaction Prediction." *ACS Central Science* 5, 1572-1583 (2019).
- Coley, C. W. et al. "RDChiral: An RDKit Wrapper for Handling Stereochemistry in Retrosynthetic Template Extraction and Application." *J. Chem. Inf. Model.* 59, 2529-2537 (2019).
- Chen, S. and Jung, Y. "Deep Retrosynthetic Reaction Prediction using Local Reactivity and Global Attention." *Chemical Science* 12, 10416-10427 (2021).
- Irwin, R. et al. "Chemformer: a pre-trained transformer for computational chemistry." *Machine Intelligence* 4, 1256-1264 (2022).
- Schwaller, P. et al. "Prediction of chemical reaction yields using deep learning." *Machine Intelligence* 3, 144-152 (2021).
- Schwaller, P. et al. "Extraction of organic chemistry grammar from unsupervised learning of chemical reactions." *Science Advances* 7, eabe4166 (2021).
- Maser, M. R. et al. "Multilabel Classification Models for the Prediction of Cross-Coupling Reaction Conditions." *ACS Central Science* 7, 258-268 (2021).
- Probst, D. et al. "Mapping the space of chemical reactions using attention-based neural networks." *Digital Discovery* 1, 258-265 (2022).
- Kearnes, S. M. et al. "The Open Reaction Database." *J. Am. Chem. Soc.* 143, 18820-18826 (2021).
- Bran, A. M. et al. "Augmenting large language models with chemistry tools." *Nature Machine Intelligence* 6, 525-535 (2024).
- Joung, J. F., Fong, M., and Coley, C. W. "Electron Flow Matching for Reaction Prediction." *Nature* 639, 378-385 (2025).
