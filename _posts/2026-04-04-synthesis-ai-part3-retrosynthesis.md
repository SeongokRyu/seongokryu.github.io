---
title: "Synthesis AI Part 3: Retrosynthesis — Can AI Plan How to Make a Molecule?"
date: 2026-04-04 09:20:00 +0900
categories: [Drug Discovery, Small Molecule Design]
tags: [retrosynthesis, askcos, aizynthfinder, synthesis-planning, single-step-model]
math: true
---

**AI-Driven Synthesis in Drug Discovery**

*This is Part 3 of a 5-part series on AI-driven synthesis in drug discovery.*

- **Part 1:** [The Synthesis Bottleneck — Why "Make" Lags Behind](/posts/synthesis-ai-part1-synthesis-bottleneck/)
- **Part 2:** [Reaction Prediction — Can AI Predict What Chemistry Will Do?](/posts/synthesis-ai-part2-reaction-prediction/)
- **Part 3** (this post): Retrosynthesis — Can AI Plan How to Make a Molecule?
- **Part 4:** [Synthesis-Aware Design — Making AI-Generated Molecules Makeable](/posts/synthesis-ai-part4-synthesis-aware-design/)
- **Part 5:** [From Algorithm to Lab — CRO Integration and the Remaining Gap](/posts/synthesis-ai-part5-algorithm-to-lab/)


---

## The Core Question

**Given a target molecule, can AI design a complete synthesis route back to commercially available starting materials?**

In Part 2, we asked whether AI can predict the products of a chemical reaction — the forward direction. Now we reverse the arrow entirely. Retrosynthesis starts with a desired molecule and works backward, asking: what precursors and reactions could produce this target? And then, recursively: how do we make those precursors?

This is the question that E.J. Corey formalized in the 1960s, earning him the 1990 Nobel Prize. He introduced the concept of "retrosynthetic analysis" — a systematic way of reasoning backward from a target molecule through strategic bond disconnections. Decades later, AI is attempting to automate what Corey's logic-driven approach began.

---

## 1. The Two Halves of Retrosynthesis

Retrosynthetic planning has two distinct sub-problems, and conflating them is a common source of confusion.

**Single-step retrosynthesis** asks: given a target molecule, what is one plausible set of precursors that could produce it in a single reaction? This is the reverse of Part 2's forward prediction — instead of "reactants to product," we predict "product to reactants."

**Multi-step retrosynthesis** asks the harder question: starting from the target, can we find a complete route — possibly 3, 5, or 10 steps long — all the way back to building blocks we can actually purchase?

The relationship between them:

```
Single-Step Retro (one disconnection)
──────────────────────────────────────

  Target ──→ [Retro Model] ──→ Precursor A + Precursor B
                                  (one step back)


Multi-Step Retro (full route search)
──────────────────────────────────────

  Target
    │
    ├── Precursor A ──── commercially available? YES ✓
    │
    └── Precursor B ──── commercially available? NO ✗
            │
            ├── Precursor C ── available? YES ✓
            │
            └── Precursor D ── available? YES ✓

  Route: C + D → B ; A + B → Target  (2 steps)
```

**The single-step model is the "expansion function" that the multi-step planner calls at each node of its search tree.** Without an accurate single-step model, multi-step planning cannot work. Without multi-step planning, a single-step model only gives you one disconnection — useless if the resulting precursors are themselves complex molecules that nobody sells.

Both halves must work together. We address them in turn.

---

## 2. Single-Step Retrosynthetic Models

The single-step retro problem can be stated simply: given a product SMILES (or graph), predict a set of reactant SMILES (or graphs) that could produce it in one reaction step. Three families of approaches have emerged, mirroring the forward-prediction taxonomy from Part 2 but with retro-specific challenges.

The retro direction is inherently harder than the forward direction. A forward reaction typically has one major product. A retrosynthetic disconnection has many plausible answers — there are often dozens of ways to assemble the same target molecule.

### 2.1 Template-Based Retrosynthesis

Template-based methods maintain a library of reaction templates — SMARTS-encoded transformation rules extracted from known reactions. At inference time, the model:

- Identifies which templates are applicable to the target molecule
- Ranks them (often by a neural network classifier)
- Applies the top-ranked template(s) to generate precursors

The pioneering work by Coley et al. (2017, ACS Central Science) at MIT used a neural network to prioritize templates from a library of ~300K rules extracted from USPTO reactions. Given a target molecule, the model produces a molecular fingerprint, passes it through the network, and outputs a probability distribution over the template library. This approach became the backbone of the ASKCOS platform.

**Template-based methods guarantee chemical validity — every suggested disconnection corresponds to a known reaction type.** This is their greatest strength and their greatest limitation.

Key characteristics:

- **Validity**: High. Templates enforce chemical rules by construction
- **Interpretability**: High. Each prediction maps to a named reaction type
- **Coverage**: Limited by template library size. Novel reaction types are invisible
- **Scalability**: Template libraries can grow to 100K+ rules, but long-tail reactions remain uncovered

The coverage problem is fundamental. If a reaction type does not appear in the training data and therefore has no extracted template, the model cannot propose it — period. This is especially problematic for novel chemistry emerging from methodology research.

### 2.2 Template-Free Retrosynthesis

Template-free methods treat retrosynthesis as a sequence-to-sequence translation problem: product SMILES in, reactant SMILES out. No template library needed.

The most direct approach reverses the Molecular Transformer architecture (Schwaller et al., 2019, ACS Central Science). Instead of training on "reactants -> product" pairs, we train on "product -> reactants" pairs. The model learns retrosynthetic disconnections implicitly from data, treating the task as a machine translation problem where the source language is "products" and the target language is "reactants."

Other notable template-free approaches:

- **MEGAN** (Sacha et al., 2021, JCIM): Predicts retrosynthesis as a sequence of graph edits guided by electron flow. Rather than generating SMILES strings, MEGAN operates on the molecular graph and predicts bond changes corresponding to reaction mechanisms. This grounds the model in electron-level chemistry
- **Graph2Retro** (various groups, 2022-2023): Graph-edit-based retrosynthesis where the model learns to remove bonds and add leaving groups directly on the molecular graph representation
- **Retroformer** (Wan et al., 2022, ICML): Augments the Transformer with local attention on the molecular graph to improve reaction center awareness, bridging the sequence and graph paradigms

**Template-free models excel at generalization — they can propose disconnections for reaction types never seen as explicit templates.** However, they sometimes generate chemically invalid SMILES or propose disconnections that no known reaction mechanism supports. The SMILES string is not a natural representation of chemistry, and errors in one character can render the entire output meaningless.

### 2.3 Semi-Template Retrosynthesis

Semi-template methods split the problem into two stages:

1. **Identify the reaction center** (which bonds to break in the target)
2. **Complete the precursors** (add leaving groups, adjust atoms around the break)

This two-stage design combines the chemical grounding of template methods with the flexibility of template-free generation. The reaction center identification constrains the problem to a manageable subspace, while the completion step can handle diverse chemistry.

Key models:

- **LocalRetro** (Chen & Jung, 2021, JCIM): Uses local atom environments to predict reaction centers, then applies local templates only at the identified sites. The "local" in the name refers to the fact that templates are applied only to atoms near the reaction center, rather than to the entire molecule. Achieved state-of-the-art top-1 accuracy on USPTO-50K (~53.4%) at the time of publication
- **RetroXpert** (Yan et al., 2020, NeurIPS): Identifies reaction centers via an edge-labeling GNN that classifies each bond as "broken or not," then completes the reactants with a Transformer decoder that generates the appropriate leaving groups and functional group adjustments

Semi-template methods often achieve the best balance of accuracy and chemical validity, because the reaction center identification step constrains the search space while the completion step retains flexibility for novel chemistry.

### 2.4 Comparison Table: Single-Step Retro Approaches

| Aspect | Template-Based | Template-Free | Semi-Template |
|---|---|---|---|
| **Coverage** | Limited by library (10K-300K templates) | Theoretically unlimited | Broad (center ID is flexible) |
| **Top-1 Accuracy** (USPTO-50K) | ~45-50% | ~48-53% | ~53-55% |
| **Top-10 Accuracy** (USPTO-50K) | ~75-85% | ~80-90% | ~85-92% |
| **Diversity of Suggestions** | Multiple templates can match | Beam search yields diverse SMILES | Moderate diversity |
| **Chemical Validity** | ~100% (by construction) | ~85-95% (invalid SMILES possible) | ~95-99% |
| **Interpretability** | High (named reaction type) | Low (black-box translation) | Medium (center is explicit) |
| **Handling Novel Reactions** | Cannot | Can attempt | Partially |
| **Speed** | Fast (template lookup + ranking) | Moderate (autoregressive decoding) | Moderate (two-stage) |

A few notes on reading this table:

- The accuracy numbers are approximate ranges from published benchmarks on the standard USPTO-50K split. Real-world performance is generally lower due to distribution shift, as we discussed in Part 2
- Top-10 accuracy matters more than top-1 for multi-step planning. The search algorithm needs multiple candidate disconnections per target to explore alternative routes. A model with 85% top-10 accuracy gives the planner enough options to work with, even if the top-1 choice is wrong
- Chemical validity of template-free models has improved with constrained decoding and post-processing, but the fundamental risk of invalid outputs remains

---

## 3. Multi-Step Planning: Tree Search Over Chemistry Space

A single-step retro model can suggest how to disconnect a molecule once. But drug-like molecules typically require 3-8 synthetic steps from available starting materials. **Multi-step retrosynthesis is fundamentally a tree search problem — and the tree can be enormous.**

### 3.1 The AND/OR Tree Formulation

The canonical formulation of multi-step retrosynthesis is an AND/OR tree:

- **OR nodes** represent molecules. A molecule is "solved" if ANY of its child reactions can produce it (OR logic: we only need one working route)
- **AND nodes** represent reactions. A reaction is "solved" only if ALL of its required precursors are available or themselves solved (AND logic: we need every reactant)

Here is a concrete example:

```
                        Target (OR)
                       /     |      \
                     /       |        \
              Rxn A (AND)  Rxn B (AND)  Rxn C (AND)
              /    \        |    \         |     \
            /       \       |     \        |      \
          P1(OR)  P2(OR)  P3(OR) P4(OR)  P5(OR) P6(OR)
          [avail]   |      [avail][avail]   |    [avail]
                    |                       |
               Rxn D (AND)            Rxn E (AND)
               /        \             /        \
             /           \          /            \
          P7(OR)      P8(OR)     P9(OR)       P10(OR)
          [avail]     [avail]    [avail]      [avail]

  Legend:
    (OR)  = molecule node  — solved if ANY child reaction works
    (AND) = reaction node  — solved if ALL precursors are available/solved
    [avail] = commercially available building block
```

In this example, three possible first-step disconnections exist (Rxn A, B, C). Rxn B leads directly to two available precursors — the shortest route (1 step). Rxn A requires one additional step (Rxn D) because P2 is not purchasable, giving a 2-step route. Rxn C also requires an extra step (Rxn E). The search algorithm must explore this tree efficiently, balancing breadth (trying many disconnections) against depth (following promising routes to completion).

The combinatorics are staggering. If each molecule has ~50 possible disconnections (a reasonable number for a drug-like molecule with a large template library), and the route is 5 steps long, the naive tree has up to $50^5 \approx 3 \times 10^8$ leaf nodes. Efficient search is essential.

### 3.2 Search Algorithms

**Monte Carlo Tree Search (MCTS)** has become the dominant algorithm for retrosynthetic planning, borrowed from the game-playing AI community (most famously, AlphaGo). MCTS balances exploration and exploitation through four repeated phases:

```
MCTS Iteration Cycle
────────────────────

  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
  │  SELECTION   │────→│  EXPANSION   │────→│   ROLLOUT    │────→│ BACKPROP    │
  │             │     │             │     │             │     │             │
  │ Walk down   │     │ Call retro  │     │ Estimate    │     │ Update      │
  │ tree using  │     │ model at    │     │ probability │     │ values of   │
  │ UCB score   │     │ leaf node   │     │ of reaching │     │ all         │
  │ to pick     │     │ to generate │     │ buyable     │     │ ancestor    │
  │ promising   │     │ candidate   │     │ building    │     │ nodes       │
  │ path        │     │ precursors  │     │ blocks      │     │             │
  └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                                                      │
        ┌─────────────────────────────────────────────────────────────┘
        │
        ▼
  Repeat for N iterations, then return best route found
```

Each MCTS iteration:

1. **Selection**: Navigate down the tree, choosing child nodes that balance high estimated value with low visit count using the UCB (Upper Confidence Bound) formula
2. **Expansion**: At a leaf node, call the single-step retro model to generate candidate disconnections. Each candidate becomes a new child AND-node
3. **Rollout / Evaluation**: Estimate how likely this partial route is to reach available building blocks. This can be done via random simulation (classic rollout), a learned value function, or a simple heuristic like molecular complexity reduction
4. **Backpropagation**: Update value estimates for all ancestor nodes along the path

Other search algorithms have also been applied:

- **Retro\*** (Chen et al., 2020, ICML): An A\*-like best-first search with a learned value function that estimates the "cost-to-go" from any molecule to available building blocks. Provably optimal under certain assumptions about the value function
- **Breadth-first search with pruning**: Simpler but can work well with strong single-step models. Explores all disconnections at each depth before going deeper
- **Proof-number search**: Particularly efficient for AND/OR trees. Used in some Syntheseus configurations with competitive results

### 3.3 Major Platforms and Tools

Let us survey the major tools that combine single-step models with multi-step search into usable retrosynthesis platforms.

**ASKCOS (MIT, Coley group)**

ASKCOS (Automated System for Knowledge-based Continuous Organic Synthesis) was the first practical, web-accessible retrosynthesis platform powered by AI (Coley et al., 2019, JCIM). It marked a turning point: retrosynthesis moved from academic papers to a tool bench chemists could actually use.

Key features:

- Template-based expansion using ~160K reaction templates from USPTO
- MCTS for multi-step planning with configurable search depth and time limits
- Integrated modules beyond retrosynthesis: forward prediction, condition recommendation, impurity prediction, and atom mapping
- Web interface designed for medicinal chemists, not just ML researchers
- Continuously updated; recent versions incorporate neural scoring and improved template prioritization

**ASKCOS demonstrated that AI retrosynthesis could be packaged as a usable tool, not just an academic benchmark exercise.** It remains the most widely cited academic retrosynthesis platform and has been adopted by several pharmaceutical companies for internal pilot programs.

**AiZynthFinder (AstraZeneca)**

AiZynthFinder (Genheden et al., 2020, JCIM) emerged from AstraZeneca's internal efforts to deploy AI retrosynthesis at industrial scale. Its distinguishing design philosophy is modularity:

- **Expansion policy** (single-step model) is swappable — users can plug in template-based, template-free, or custom models without changing the search infrastructure
- **Stock databases** are configurable — internal corporate compound inventories, Enamine catalog, eMolecules, or custom building block lists
- **Search algorithm** defaults to MCTS but supports alternatives
- **Route scoring** incorporates practical metrics: step count, estimated feasibility, building block availability

The platform has evolved significantly since its initial release. AiZynthFinder 4.0 (Genheden et al., 2024, JCHEMINF) incorporates three years of industrial deployment learnings:

- Improved route scoring that better correlates with expert chemist preferences
- Better handling of stereochemistry in retrosynthetic disconnections
- Integration with AstraZeneca's internal compound management systems
- Enhanced stock management for real-world building block availability

**AiZynthFinder's strength is its industrial pragmatism — it was designed by medicinal chemists and computational chemists working together, and it shows in every design decision.**

**Synthia (formerly Chematica)**

Synthia stands apart from the ML-first approaches. Developed over two decades by Bartosz Grzybowski's group (originally at Northwestern, then KAIST/Allchemy), Synthia encodes expert chemical knowledge directly:

- **50,000+ hand-coded reaction rules** curated by expert chemists, covering named reactions, strategic transformations, and protecting group chemistry
- Network-based search (not MCTS) over a massive graph of chemical transformations
- Rules are encoded with substrate scope, functional group tolerance, and strategic priority — going well beyond simple SMARTS patterns
- Successfully planned total syntheses of complex natural products, validated experimentally in the lab (Klucznik et al., 2018, Chem)
- Acquired by Merck/MilliporeSigma and commercialized as a subscription service

**Synthia represents the "expert knowledge" pole of retrosynthesis — fewer rules than ML models extract from data, but each rule is carefully validated and annotated with practical constraints.** The trade-off is scalability: curating 50K rules required years of expert effort, and extending to new reaction domains is slow and labor-intensive.

The 2018 validation study was particularly impressive: Synthia planned synthesis routes for eight medicinally relevant targets, and all eight were successfully executed in the lab, some with routes that expert chemists described as creative and non-obvious.

**SynPlanner (Gao et al., 2024)**

SynPlanner is a recent open-source retrosynthesis platform that combines modern single-step models with MCTS search (Gao et al., 2024, JCIM). It emphasizes:

- Reproducibility and ease of benchmarking
- Support for multiple expansion policies and search configurations out of the box
- Clean Python API for integration into larger workflows
- Modern model architectures as default expansion policies

SynPlanner fills an important gap: a modern, well-maintained open-source baseline that researchers can use for fair comparison of new methods.

**Syntheseus (Microsoft Research)**

Syntheseus, led by Marwin Segler's group at Microsoft Research (Segler et al., 2024), takes a meta-level approach. Rather than proposing yet another retro model or search algorithm, Syntheseus provides a unified framework for combining any single-step model with any search algorithm:

- Plug in different retro models (template-based, template-free, semi-template)
- Plug in different search algorithms (MCTS, Retro\*, breadth-first, proof-number search)
- Run systematic benchmarks across all model-search combinations
- Compare results on a level playing field with standardized evaluation metrics

A key insight from the Syntheseus benchmarking: **the choice of search algorithm matters as much as the choice of single-step model.** A mediocre retro model with an excellent search algorithm can outperform a state-of-the-art retro model with naive search. This finding is often overlooked when papers report only single-step accuracy numbers.

**Syntheseus demonstrated that retrosynthesis performance depends on the interaction between the expansion model and the search algorithm, not on either component alone.**

### 3.4 Comparison Table: Multi-Step Planning Tools

| Tool | Approach | Single-Step Model | Search | Open-Source? | Industrial Adoption | Key Strength |
|---|---|---|---|---|---|---|
| **ASKCOS** | Template + MCTS | Template-based (~160K) | MCTS | Yes (MIT license) | Academic, pharma pilots | First practical platform; web UI |
| **AiZynthFinder** | Modular + MCTS | Swappable (default: template) | MCTS | Yes (MIT license) | AstraZeneca; external pharma | Industrial pragmatism; 3+ yrs deployment |
| **Synthia** | Expert rules + network | 50K+ hand-coded rules | Network search | No (commercial) | Merck/MilliporeSigma | Expert-validated; total synthesis |
| **SynPlanner** | Modular + MCTS | Multiple supported | MCTS | Yes | Early academic | Modern, reproducible baseline |
| **Syntheseus** | Meta-framework | Any (plug-in) | Any (plug-in) | Yes (MIT license) | Research benchmarking | Systematic model-search comparison |
| **Retro\*** | Learned value + A\* | Various | A\*-like | Yes | Academic | Provably optimal search |

A clear trend emerges from this table: **the field is converging on modular, open-source frameworks** where the single-step model and the search algorithm are interchangeable components. This is a healthy development — it cleanly separates the two research problems and enables systematic comparison of contributions.

The practical consequence for a drug discovery team evaluating these tools: start with AiZynthFinder or ASKCOS (mature, well-documented, community-supported), and use Syntheseus when you need to benchmark alternative models or search strategies.

---

## 4. "Route Suggested ≠ Route Works"

Here we arrive at the most important — and most uncomfortable — section of this post. Everything above describes impressive technical progress. Single-step models achieve over 50% top-1 accuracy. Multi-step planners find routes in seconds that would take a chemist hours to draft. The tools are deployed at major pharmaceutical companies.

**And yet, the gap between a suggested route and a route that actually works in the lab remains enormous.**

### 4.1 Failure Modes

When an AI-proposed synthesis route fails in the laboratory, the failure rarely comes from gross chemical nonsense. The models are good enough to avoid suggesting that carbon form five bonds. The failures are subtler, more practical, and harder to fix algorithmically:

- **Reagent availability mismatch**: The route calls for a building block listed in the Enamine catalog, but it has a 6-week lead time, is available only in 5 mg quantity, or has been discontinued since the database snapshot. Building block availability is dynamic; the planner's stock database is static

- **Reaction condition optimization is absent**: The retro model says "use a Suzuki coupling here" but does not specify the palladium catalyst, ligand, base, solvent, temperature, or reaction time. As we discussed in Part 2, condition prediction remains an unsolved sub-problem. The chemist must optimize each step, which can take days to weeks per reaction

- **Protecting group strategies are not reflected**: Complex molecules with multiple functional groups require strategic protection and deprotection sequences. Most AI retro models treat each disconnection independently, without considering the global protecting group strategy that an expert chemist would plan from the very beginning

- **Scale-up chemistry diverges from milligram-scale**: A reaction that works at 10 mg scale in a screening vial may fail at 1 g scale due to heat transfer differences, mixing limitations, or solubility changes. AI models are trained on literature reactions, most of which are reported at small scale

- **Order-dependent side reactions**: Regioselectivity and chemoselectivity issues that the model fails to anticipate. A nucleophilic addition might attack the wrong electrophilic site. A cross-coupling might give homo-coupling as the major product under certain conditions

- **Functional group incompatibility**: The route proposes a step using a strong base in the presence of a base-sensitive ester, or a reduction step that would also reduce an amide the route intends to keep. Models trained on individual reactions may not capture these cross-step incompatibilities

The following diagram illustrates how these failures compound across a multi-step route:

```
AI-Proposed Route (5 steps)
───────────────────────────

Step 1: Suzuki coupling          → conditions unknown, needs optimization
Step 2: Boc deprotection         → straightforward, likely works
Step 3: Amide coupling           → HATU coupling, but free amine from
                                    Step 2 may cause side reactions
Step 4: Reduction (LiAlH4)       → too aggressive; will reduce the
                                    amide from Step 3 as well
Step 5: Final cyclization        → never tested at this scale

Result: 1 clean step, 1 needing optimization, 3 with potential issues
        Route is NOT executable as written
```

### 4.2 Quantitative Evidence

How often do AI-proposed routes actually work? The honest answer is: we have limited systematic data, and what exists is sobering.

**Route-finding rate vs. route-working rate:**

Genheden et al. (2020, JCIM), in their AiZynthFinder paper, tested AI-proposed routes on a set of pharmaceutically relevant targets. The route-finding success rate — simply finding any route to purchasable building blocks — was around 85-95% depending on the target set. But "finding a route" and "the route works in the lab" are very different things.

**Retrospective analysis at AstraZeneca:**

Thakkar et al. (2021, Chemical Science) from the AstraZeneca group conducted one of the most informative studies. They analyzed AI-proposed routes against routes that had actually been executed in AstraZeneca's internal pipeline:

- AI tools could find routes for most targets, but the proposed routes often differed significantly from what expert chemists would choose
- Routes that aligned with expert-chosen paths had substantially higher experimental success rates
- The biggest discrepancies were in protecting group strategy and reagent choices — exactly the areas where current models are weakest
- Expert chemists frequently modified AI-suggested routes before execution, indicating that the routes served as starting points rather than final plans

**Real-world synthesis at scale (COVID Moonshot):**

PostEra's COVID Moonshot project (2020-2021) provided perhaps the largest-scale real-world test of AI-assisted synthesis planning. Hundreds of AI-designed molecules were synthesized via CRO networks. The project demonstrated that AI-to-synthesis pipelines can function at scale, but it also revealed that significant human expert intervention was needed to convert AI route suggestions into executable synthesis plans.

**Summary of validation evidence:**

| Metric | Approximate Range | Source |
|---|---|---|
| Route-finding rate (any route found) | 85-95% | AiZynthFinder, ASKCOS benchmarks |
| Route agreement with expert choice | 30-50% | AstraZeneca retrospective |
| Routes executable without modification | 30-50% (estimated) | Cross-study synthesis |
| Routes working on first attempt | Lower still | Limited data available |

**A reasonable estimate, synthesizing the available evidence: perhaps 30-50% of AI-proposed routes for drug-like molecules can be executed without major modifications, and even fewer work on the first attempt without condition optimization.**

### 4.3 The Nature of the Gap

It is tempting to frame this as "the models need to be more accurate." But that misses the deeper issue.

**The gap between route suggestion and route execution is not primarily a model accuracy problem — it is a condition prediction and practical constraint problem.**

Consider what happens when an expert chemist plans a synthesis:

```
Expert Chemist's Mental Model
─────────────────────────────

  Step 1: "I'll use a Buchwald-Hartwig amination"
          → I know Pd2(dba)3 / XPhos works for this substrate class
          → I need to check if the aryl chloride is electron-poor enough
          → Solvent: toluene, 100°C, overnight
          → I have this reagent in my freezer

  Step 2: "Then a Boc deprotection"
          → TFA in DCM, room temperature, 1 hour
          → Standard, should be fine

  Step 3: "Amide coupling"
          → HATU / DIPEA in DMF
          → But wait — will the free amine from Step 2
             interfere? Maybe I should change the order...


AI Retrosynthesis Model's Output
─────────────────────────────────

  Step 1: "Buchwald-Hartwig amination"  (disconnection only)
  Step 2: "Boc deprotection"            (disconnection only)
  Step 3: "Amide coupling"              (disconnection only)

  No conditions. No strategic reasoning. No cross-step analysis.
```

The chemist integrates disconnection logic, condition knowledge, reagent availability, and strategic thinking into a single unified plan. Current AI retrosynthesis delivers only the first part — disconnection logic — and leaves everything else to the human. The AI is doing perhaps 20% of the total intellectual work of synthesis planning.

### 4.4 Current Practical Level

Where does this leave us? The most accurate description of current AI retrosynthesis tools is: **an expert chemist's brainstorming partner.**

This is not a dismissal. Brainstorming partners are valuable. Specifically, AI retro tools are useful for:

- **Rapid exploration**: Generating 10-50 candidate routes in minutes, giving the chemist a broader menu of options than they might conceive alone
- **Identifying non-obvious disconnections**: Sometimes the model suggests a route the chemist would not have considered, leading to a shorter or more elegant synthesis
- **Building block sourcing**: Connecting target structures to specific purchasable starting materials across catalogs of millions of compounds
- **Cross-checking**: Validating that a chemist's planned route is not missing an obviously better alternative
- **Onboarding and education**: Helping less experienced chemists quickly generate reasonable starting points for synthesis planning

They are not yet reliable for:

- **Autonomous synthesis planning** without human review and modification
- **Condition-complete route specification** (catalyst, solvent, temperature, time for each step)
- **Cost or time optimization** across the full route
- **Handling of stereochemistry-critical** targets with high fidelity
- **Strategic decisions** about convergent vs. linear routes, protection strategies, or step ordering

---

## 5. Convergence and Open Questions

Let us step back and assess where the field stands as of early 2026.

### 5.1 The State of Single-Step Retrosynthesis

**Template-free and semi-template methods lead in accuracy, while template-based methods lead in interpretability and chemical validity.** The accuracy gap is narrowing as template-free models incorporate more chemical inductive biases (graph attention, electron flow). Semi-template approaches like LocalRetro currently offer the best trade-off for practical deployment.

An interesting convergence is emerging: the best models from each category are borrowing ideas from the others. Template-free models now use graph-based attention to localize predictions. Semi-template models use learned completion rather than fixed templates. The boundaries between categories are blurring.

### 5.2 The Error Accumulation Problem

**On multi-step planning, the primary bottleneck is error accumulation.** If a single-step model has 50% top-1 accuracy, and a 5-step route requires all 5 disconnections to be correct, the naive probability of a fully correct route is:

$$
P(\text{correct route}) = (0.5)^5 \approx 3\%
$$

In practice, beam search and diverse suggestions mitigate this somewhat — the planner does not rely solely on the top-1 prediction at each step. But the exponential penalty of sequential errors remains the central mathematical challenge of multi-step planning.

This is why top-10 accuracy matters so much. If the correct disconnection appears somewhere in the top-10 suggestions with 90% probability, the math improves dramatically:

$$
P(\text{correct route, top-10}) = (0.9)^5 \approx 59\%
$$

But the search algorithm must then navigate a much larger tree to find the right combination, which brings its own computational challenges.

### 5.3 Open Questions

The most pressing research questions in retrosynthesis, as we see them:

- **Condition-aware retrosynthesis**: Can we build retro models that predict not just "what disconnection" but "what disconnection under what conditions, with what expected yield"? This would close the gap between route suggestion and route execution. Early work by the Coley group on reaction condition prediction points the way, but integration into multi-step planning is nascent

- **Automatic route scoring and ranking**: When a planner finds 20 routes, which one should the chemist try first? Scoring must integrate step count, estimated yield per step, cost of building blocks, availability, lead time, and strategic complexity. Current route scoring is rudimentary — typically just step count or a simple heuristic

- **Building block set optimization**: The definition of "commercially available" is itself a design choice. Optimizing which building blocks to stock in-house, or which virtual building blocks to enumerate for bespoke libraries, is an underexplored combinatorial problem with large practical impact

- **Strategy-level planning**: Current models operate at the level of individual disconnections. Expert chemists think at a higher strategic level — "install the stereocenter early," "use a convergent strategy," "protect the amine before the coupling sequence." Encoding these meta-level heuristics into AI planners is an emerging research direction, with recent work from the Coley group (2025-2026) beginning to address this gap

- **Confidence calibration**: When an AI tool proposes a route, how confident should we be? Current models provide scores, but these scores are poorly calibrated — a route scored 0.9 is not necessarily more likely to work than one scored 0.7. Trustworthy uncertainty estimates would help chemists prioritize which routes to attempt

### 5.4 Looking Ahead

Retrosynthesis tells us how to make a specific molecule. It takes a fixed target and works backward to find a route. But what if we could turn this relationship around?

What if, instead of designing a molecule first and then asking whether it can be made, we could generate molecules that are inherently synthesizable — with a synthesis route included at the moment of generation? Instead of the sequential pipeline of "design, then check synthesizability, then plan route," what if all three happened simultaneously?

That is the promise of synthesis-aware molecular design, and it is the subject of Part 4.

---

*Next: Part 4 — Synthesis-Aware Design: Can We Generate Only Molecules We Can Actually Make?*

---

## References

- Chen, B., & Jung, J. (2021). "LocalRetro: Improving Retrosynthetic Analysis with Local Atom Environments." *JCIM*.
- Chen, B., Li, C., Dai, H., & Song, L. (2020). "Retro*: Learning Retrosynthetic Planning with Neural Guided A* Search." *ICML*.
- Coley, C. W., Rogers, L., Green, W. H., & Jensen, K. F. (2017). "Computer-Assisted Retrosynthesis Based on Molecular Similarity." *ACS Central Science*.
- Coley, C. W., Thomas, D. A., Lummiss, J. A. M., et al. (2019). "A Robotic Platform for Flow Synthesis of Organic Compounds Informed by AI Planning." *Science*.
- Gao, W., et al. (2024). "SynPlanner: An Open-Source Tool for Retrosynthetic Planning." *JCIM*.
- Genheden, S., Thakkar, A., Chadimova, V., et al. (2020). "AiZynthFinder: A Fast, Robust and Flexible Open-Source Software for Retrosynthetic Planning." *JCIM*.
- Genheden, S., et al. (2024). "AiZynthFinder 4.0: Developments Based on Learnings from 3 Years of Industrial Application." *JCHEMINF*.
- Klucznik, T., Miber, B., Szymkuc, S., et al. (2018). "Efficient Syntheses of Diverse, Medicinally Relevant Targets Planned by Computer." *Chem*.
- Sacha, M., Blaz, M., Byrski, P., et al. (2021). "Molecule Edit Graph Attention Network: Modeling Chemical Reactions as Sequences of Graph Edits." *JCIM*.
- Schwaller, P., Laino, T., Gaudin, T., Bolgar, P., et al. (2019). "Molecular Transformer: A Model for Uncertainty-Calibrated Chemical Reaction Prediction." *ACS Central Science*.
- Segler, M. H. S., et al. (2024). "Syntheseus: A Benchmark Framework for Retrosynthesis." *Microsoft Research*.
- Thakkar, A., Chadimova, V., Bjerrum, E. J., et al. (2021). "Retrosynthetic Accessibility Score (RAscore)." *Chemical Science*.
- Wan, Y., Hassen, B., Hsieh, C.-Y., & Shang, J. (2022). "Retroformer: Pushing the Limits of Interpretable End-to-End Retrosynthesis Transformer." *ICML*.
- Yan, C., Ding, Q., Zhao, P., et al. (2020). "RetroXpert: Decompose Retrosynthesis Prediction Like a Chemist." *NeurIPS*.
