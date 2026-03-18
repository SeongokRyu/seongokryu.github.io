---
title: "Toward Accelerating the Entire Design–Make–Test Cycle with AI"
date: 2026-03-18 09:00:00 +0900
categories: [Drug Discovery, Agentic Discovery]
tags: [drug-discovery, dmt-cycle, ai-agent, lab-automation, closed-loop]
math: true
---

The pharmaceutical industry measures progress in cycles: design a molecule, make it, test it, learn from the results, and design the next one. This **Design–Make–Test–Learn (DMTL)** loop — often abbreviated as DMT — is the heartbeat of drug discovery. Every marketed drug passed through hundreds, sometimes thousands, of these iterations before reaching patients. A typical small-molecule program runs 10–15 full DMT cycles per year; a biologics program, fewer still.

AI has compressed one part of this loop dramatically. We can now generate novel protein binders, small-molecule candidates, and even macrocyclic peptides in minutes. But a curious imbalance has emerged: the Design step has been accelerated by orders of magnitude, while Make and Test remain stubbornly slow. The community celebrates each new generative model, each percentage-point improvement in predicted binding affinity — but the calendar time from idea to experimental validation has barely changed.

This essay examines that imbalance, asks what direct AI acceleration of Make and Test would look like, and considers the orchestration architecture needed to close the entire loop. The argument is simple: **the marginal return on further Design acceleration is diminishing; the highest-leverage investments are now in Make, Test, and the infrastructure that connects all stages.**

Two recent papers frame the discussion. Zhavoronkov, Gennert, and Shi's "From Prompt to Drug: Toward Pharmaceutical Superintelligence" [1] paints a sweeping vision of autonomous drug discovery pipelines driven by agentic AI. The paper is deliberately provocative — the title invokes "superintelligence" — but beneath the rhetoric lies a substantive argument about pipeline-level orchestration. Orlov et al.'s "ChemSpace Copilot" [2] offers a concrete implementation: an agentic system built on Generative Topographic Mapping (GTM) that automates chemical space exploration with an Observe–Plan–Act–Reflect loop. Together, they bracket the spectrum from vision to execution — and both expose the same blind spot.

---

## 1. The DMT Cycle as the Unit of Drug Discovery Progress

### 1.1 Anatomy of a Cycle

A single DMT cycle consists of four coupled stages:

```
  ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
  │ DESIGN  │────▶│  MAKE   │────▶│  TEST   │────▶│  LEARN  │
  │         │     │         │     │         │     │         │
  │ propose │     │ synth / │     │ assay / │     │ analyze │
  │ molecule│     │ express │     │ screen  │     │ update  │
  └─────────┘     └─────────┘     └─────────┘     └────┬────┘
       ▲                                               │
       └───────────────────────────────────────────────┘
```

**Design** generates candidate molecules — small molecules, peptides, proteins, or antibodies — guided by hypotheses about the target. **Make** synthesizes or expresses those candidates in physical form. **Test** measures their properties: binding affinity, selectivity, ADMET, cellular activity. **Learn** interprets results and updates the mental (or computational) model that guides the next round of Design.

The distinction between DMT and DMTL (with an explicit Learn step) matters. Many organizations treat Learn as implicit — the chemist "just knows" what to do next. Making it explicit exposes a critical gap: the Learn step is where AI could provide the most leverage, yet it receives the least attention.

### 1.2 The Time Asymmetry

Industry data quantifies the time asymmetry across stages. AstraZeneca — which has published more extensively on DMTA cycle optimization than any other pharma company [18] — reports that a traditional DMTA cycle takes **4–6 weeks**, with synthesis alone consuming **3–6 weeks** per round. Tier 1 in vitro ADME assays require **5 working days** turnaround as an industry benchmark; a full assay cascade from compound reception to data upload takes approximately **10 calendar days**. Design, by contrast, can now be completed in hours or less with AI tools.

| Stage  | Small Molecules | Biologics | Source |
|--------|----------------|-----------|--------|
| Design | Hours (AI) to 1–3 days (manual) | Hours (AI) to days | General; Schrödinger FEP+ benchmarks |
| Make   | 3–6 weeks (traditional); 1–2 weeks (optimized) | 6–12 weeks (antibody); 2–3 weeks (nanobody, phage) | AstraZeneca; ProteoGenix |
| Test   | 5–10 working days (in vitro panel) | 1–4 weeks (binding + biophysics) | Industry standard |
| Learn  | 1–3 days | 1–3 days | General |
| **Full cycle** | **4–6 weeks** (typical); <5 days (AZ iLab target) | **8–16 weeks** | AstraZeneca; industry surveys |

Note that the asymmetry is even more extreme for biologics. Antibody discovery via immunization alone takes 1–3 months before any DMTA cycling can begin. Even with synthetic phage display libraries (bypassing immunization), nanobody discovery and screening requires 2–3 weeks for a single round. Designing the protein sequence took 10 minutes.

This asymmetry has a critical consequence. Borrowing from parallel computing: **the overall cycle time is dominated by its slowest stage**, much as Amdahl's law dictates that the speedup from parallelizing one component is limited by the sequential fraction.

Let's make this concrete using industry-representative timelines:

```
  Typical DMTA cycle (small molecule lead optimization):
  Based on AstraZeneca and industry-reported timelines

    Design:  2 days     (med chem ideation + computational scoring)
    Make:    21 days     (synthesis, purification, QC — 3-week median)
    Test:    10 days     (biochemical + cellular + Tier 1 ADME panel)
    Learn:   2 days      (data analysis, team review, next-round planning)
    ─────────────────
    Total:   35 days     (≈ 5 weeks — consistent with 4-6 week industry norm)

  After 100x Design acceleration (AI generative models):
    Design:  0.02 days   (← minutes, not days)
    Make:    21 days      (unchanged)
    Test:    10 days      (unchanged)
    Learn:   2 days       (unchanged)
    ─────────────────
    Total:   33.02 days   (← 5.7% improvement)

  After 2x Make + Test acceleration (automation + prediction):
    Design:  2 days       (unchanged)
    Make:    10.5 days    (← route pre-validation, robotic synthesis)
    Test:    5 days       (← active learning, focused assay panel)
    Learn:   2 days       (unchanged)
    ─────────────────
    Total:   19.5 days    (← 44% improvement)
```

The numbers speak for themselves. A 100x improvement in Design yields ~6% cycle time reduction. A 2x improvement in Make and Test yields 44%. This is Amdahl's law in action: optimizing the fast component of a sequential process has negligible impact when the slow components dominate.

The contrast is even starker for biologics, where Make alone can consume 6–12 weeks. AstraZeneca's stated ambition of reducing full DMTA cycles to **<5 days** via their iLab initiative implicitly acknowledges that this requires compressing Make and Test, not just Design. Exscientia has demonstrated that AI-guided DMTA cycling can deliver development candidates in **12–15 months** (vs. industry average 4.5 years) by synthesizing only **150–250 compounds** instead of the typical 500–1,200 — but this efficiency gain comes primarily from better Design (fewer wasted cycles) rather than faster Make or Test.

This is not a hypothetical concern. It is the central bottleneck of modern AI-driven drug discovery.

### 1.3 Two Papers, One Diagnosis

Zhavoronkov et al. [1] articulate the vision most clearly. Their "pharmaceutical superintelligence" concept envisions an AI system that not only designs molecules but orchestrates their synthesis, testing, and iterative optimization — coordinating robotic labs, managing experimental queues, and learning from results in real time. The paper identifies the right problem: that isolated AI models, no matter how powerful, cannot accelerate drug discovery alone because they operate on only one stage of a multi-stage process.

The paper's most interesting contribution is its framing of **prompt-to-drug** as an end-to-end pipeline, analogous to prompt-to-image or prompt-to-code in the generative AI world. A researcher describes a therapeutic need in natural language; the system autonomously generates candidates, plans synthesis, executes experiments, and iterates. This framing highlights just how far the field is from that vision — not because Design is inadequate, but because the downstream stages lack the APIs, data, and automation to participate in such a pipeline.

ChemSpace Copilot [2] provides a more modest but instructive proof of concept. By wrapping GTM-based chemical space visualization with an agentic Observe–Plan–Act–Reflect loop, Orlov et al. demonstrate that even within the Design stage, autonomous exploration outperforms static tool usage. The system can navigate chemical space, identify promising regions, and generate candidates with minimal human guidance. Their integration of SynPlanner for retrosynthetic analysis hints at cross-stage connections — but the system stops at "suggesting a synthetic route." Whether that route will work in practice, and what to do when it fails, remains outside the agent's scope.

Both papers, in different ways, confirm the same observation: **AI in drug discovery is heavily concentrated on Design, with Make and Test receiving at best indirect assistance.** Zhavoronkov acknowledges this implicitly by describing the full pipeline as aspirational. Orlov demonstrates it concretely by building a sophisticated agentic system that operates entirely within the computational domain.

---

## 2. Where AI Has Made the Most Progress — and Where It Hasn't

### 2.1 Design: The Mature Frontier

The Design stage has been transformed by AI, but it is worth noting that Design itself has two layers. The first — **structural target characterization** — involves understanding the target's 3D structure, identifying druggable binding sites or epitopes, and formulating a design strategy (orthosteric vs. allosteric, competitive vs. covalent, small molecule vs. biologic). This upstream layer has been dramatically accelerated by structure prediction (AlphaFold2 [15], Boltz-1, Chai-1), pocket detection (P2Rank, FPocket, SiteMap), and epitope prediction tools. A target that once required months of crystallography and mutagenesis to characterize structurally can now be computationally mapped in hours. This layer is a prerequisite for everything that follows — the quality of Design depends on the quality of target understanding.

The second layer — **molecular generation and optimization** — is where the most visible progress has occurred:

**Protein design** has progressed from physics-based energy minimization (Rosetta) through diffusion-based backbone generation (RFdiffusion) to joint structure-sequence generation (BoltzGen, Chai-2). The trajectory, traced in detail across the Protein AI series [3], shows rapid convergence on Pairformer-based architectures and flow matching as the generative paradigm. By early 2026, protein design models can generate plausible binders for arbitrary targets in minutes. The experimental success rate for computationally designed binders has climbed from <1% (pre-AF2) to 10–30% for well-defined targets, with some groups reporting >50% for nanobody and miniprotein scaffolds.

**Small-molecule design** has followed a parallel arc: variational autoencoders, reinforcement learning on molecular graphs, and diffusion models in 3D coordinate space. Tools like REINVENT, MolMIM, and 3D-conditional generative models can propose thousands of candidates per hour, scored against predicted ADMET profiles and binding affinity. The integration of structure-based design with generative chemistry — conditioning generation on protein pocket geometry — has become standard practice.

**Scoring and filtering** has matured alongside generation. Structure-based scoring with ML potentials, PoseBusters-style physical validity checks [13], and multi-objective optimization frameworks allow rapid prioritization of generated candidates. The entire Design pipeline — from target structure to ranked candidate list — can execute in an afternoon.

**Antibody and nanobody design** represents a particularly mature subfield. CDR loop grafting, humanization, and de novo CDR design have all been addressed with varying degrees of success. The combination of language models (trained on OAS and other antibody sequence databases) with structure prediction (IgFold, ABlooper, ABodyBuilder2) enables rapid in-silico antibody engineering.

Why has Design advanced so rapidly? Three structural advantages:

1. **Benchmarks exist.** CASP, CAMEO, PoseBusters, and numerous molecular generation benchmarks (MOSES, GuacaMol, PHARM) allow quantitative comparison and publication-driven competition. The competitive dynamics of benchmark-driven research generate rapid progress.
2. **Public data is abundant.** The PDB (220k+ structures), UniProt (250M+ sequences), ZINC (2B+ molecules), ChEMBL (2.4M compounds with bioactivity data), and synthetic structure databases (AFDB with 200M+ predicted structures, ESMAtlas) provide massive training sets. The data pyramid is steep but sufficient for training foundation models.
3. **Pure computation suffices.** Design is fundamentally an in-silico task. No physical experiments are needed to train or evaluate design models — only to validate their outputs. This means iteration cycles for model development are minutes, not months.

These advantages do not extend to Make and Test.

### 2.2 Make: Where AI Stops at the Lab Door

AI's contribution to the Make stage remains largely advisory — offering suggestions that a human expert must evaluate, adapt, and execute.

**Retrosynthetic analysis** is the most developed capability. Tools like SynPlanner (integrated into ChemSpace Copilot [2]), ASKCOS (MIT), AiZynthFinder [14] (AstraZeneca), and commercial platforms from PostEra and Synthia (Merck/Sigma-Aldrich) can propose multi-step synthetic routes for target molecules. These represent genuine progress — a medicinal chemist can get route suggestions in seconds rather than spending hours in literature search. Modern retrosynthetic tools use template-based approaches, template-free neural models, or hybrids, and can handle complex natural product-like scaffolds.

But route suggestion is not synthesis. The gap between "here is a plausible route" and "this route will produce 50 mg of pure product in your lab next week" is enormous. Current retrosynthetic tools:

- **Do not predict failure modes.** A proposed route may involve a step with 20% yield that makes the overall synthesis impractical, or a protecting group strategy that fails for the specific substrate. A chemist looks at a retrosynthetic proposal and immediately spots potential issues — steric clashes, competing side reactions, difficult purifications — that the model has no mechanism to flag.
- **Do not optimize reaction conditions.** Temperature, solvent, catalyst loading, concentration, and reaction time are typically suggested as literature defaults, not optimized for the specific context. The difference between "this reaction works at 80°C in DMF" and "this specific substrate requires 120°C in NMP with 5 mol% Pd(OAc)₂ and added CuI" can be the difference between success and failure.
- **Do not account for laboratory constraints.** Reagent availability, equipment limitations (no glovebox, no high-pressure reactor), safety considerations (pyrophoric reagents, exothermic reactions), and practical scale requirements are outside the model's scope.

**Synthetic accessibility scores** (SA scores, SCScore, RAscore) provide a rough filter — flagging molecules that are likely difficult to synthesize — but they are coarse signals. A molecule with SA score = 4.5 might be trivially synthesized by a lab with the right starting materials, or impossible for a lab without access to specific chiral building blocks. The score conveys difficulty without actionable guidance.

For **biologics**, AI contributes to codon optimization (tools like IDT's Codon Optimization Tool, GenSmart) and, increasingly, to predicting expression levels from sequence features. But the vast space of construct design decisions — expression system, tag selection, purification strategy, refolding protocols, culture conditions, scale-up parameters — remains largely manual, guided by institutional knowledge and trial-and-error that varies enormously between laboratories.

**What is missing in Make:**

| Capability | Current State | Needed |
|-----------|--------------|--------|
| Route feasibility | Binary (possible/not) | Probability with failure diagnosis |
| Reaction conditions | Literature defaults | Context-specific optimization |
| Failure prediction | None | Pre-synthesis risk assessment |
| Lab constraints | Ignored | Constraint-aware planning |
| Biologics construct | Manual design | Automated with yield prediction |
| Equipment interface | No standard API | Standardized submission/retrieval |

### 2.3 Test: Where AI Prioritizes but Doesn't Execute

AI's role in the Test stage is similarly indirect — helping decide what to test, but not how to test it or how to interpret the results holistically.

**Activity prediction** models — ranging from simple QSAR to graph neural networks trained on binding data — help prioritize which candidates to test first. This is valuable: testing 20 high-confidence candidates instead of 100 random ones saves time and resources. But the experimental work itself remains unchanged. The assays take the same time whether the compounds were selected by AI or by a chemist.

**Molecular dynamics and free energy perturbation (FEP)** calculations provide computational surrogates for certain binding measurements, reducing the number of physical experiments needed. FEP+ (Schrödinger) and similar tools can predict relative binding free energies with useful accuracy (RMSE ~1 kcal/mol for congeneric series), effectively replacing some experimental iterations with computational ones. This is genuine acceleration of the Test stage — but it is limited to binding affinity for well-characterized targets and does not extend to cellular assays, ADMET properties, or in vivo endpoints.

**Virtual screening** with docking or ML-based scoring functions also serves as a Test surrogate, filtering millions of candidates down to thousands before any physical testing. This reduces the volume of experimental work but does not change the per-experiment timeline.

What AI does not yet do for the Test stage:

- **Assay design automation.** Choosing the right assay format, designing controls, setting concentration ranges, and estimating required sample sizes are currently expert tasks. The knowledge that "for a kinase target, start with a TR-FRET biochemical assay before moving to cellular NanoBRET" is encoded in organizational experience, not in any AI system. An AI system that could recommend assay designs — or flag problematic designs before execution — would directly accelerate testing.
- **Active learning for experimental selection.** Instead of testing all candidates or selecting them by predicted score alone, an active learning framework could select the subset of experiments that maximizes information gain. This is well-studied in machine learning but poorly integrated into drug discovery workflows, partly because the timescales don't match typical active learning assumptions.
- **Multi-readout integration.** A real drug discovery campaign generates data from dozens of assay types — binding, cellular potency, selectivity, metabolic stability, permeability, hERG, solubility, in vivo PK. Integrating these into a coherent signal for the next design cycle is a cognitive task that current AI systems barely attempt. Each endpoint is modeled independently; the cross-endpoint reasoning is left to human experts.
- **Experimental quality assessment.** Detecting when an assay has produced unreliable results — due to compound aggregation, fluorescence interference, edge effects in plate assays, or other artifacts — requires domain expertise that is not yet encoded in automated systems.

### 2.4 The Core Diagnosis

The pattern is clear:

```
                    AI Impact on DMT Cycle (2026)

  Design ████████████████████████████████████████ High
         (generative models, scoring, filtering)

  Make   ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ Low
         (route suggestion only)

  Test   ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ Low
         (prioritization, FEP surrogates)

  Learn  ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░ Moderate
         (SAR analysis, some automated interpretation)
```

Accelerating Design by another 10x — from minutes to seconds — will not meaningfully reduce overall cycle time when Make takes weeks and Test takes months. The marginal return on Design acceleration is diminishing rapidly. **The next high-impact frontier is direct acceleration of Make and Test.**

This is not merely a technology problem. It is simultaneously:

- **A data problem.** Make and Test generate proprietary data that rarely enters public training sets. Failed syntheses are not published. Assay conditions are buried in methods sections or not reported at all.
- **A benchmark problem.** There is no CASP equivalent for synthesis success prediction, no PoseBusters for experimental design quality. Without benchmarks, there is no competitive pressure to improve and no way to measure progress.
- **An interface problem.** Connecting AI systems to physical laboratory operations requires standardized APIs, machine-readable protocols, and real-time feedback mechanisms that do not yet exist at scale.
- **An incentive problem.** Academic researchers are rewarded for publishing novel models (Design), not for building infrastructure (Make/Test integration). Industry has the data but not the incentive to share it.

---

## 3. What Would Direct Acceleration of Make Look Like?

The Make step differs fundamentally between small molecules and protein therapeutics. For small molecules, Make means chemical synthesis — a sequence of bond-forming and bond-breaking reactions. For proteins (antibodies, nanobodies, enzymes), Make means biological production — convincing a living cell to express, fold, and secrete a functional protein. These two processes have different bottlenecks, different data landscapes, and require different AI strategies. We treat them separately.

### 3A. Small Molecule Make

#### 3A.1 The Make Problem Is Being Absorbed into Design

A remarkable shift is underway in small-molecule drug discovery: **the Make problem is increasingly being solved at the Design stage** by constraining molecular generation to synthesizable chemical space. Two complementary paradigms drive this trend.

**Paradigm A: On-demand libraries and ultra-large virtual screening.** Companies like Enamine (REAL Space: 78 billion make-on-demand molecules) and WuXi (GalaXi: 26 billion) have built massive virtual libraries where every molecule has a validated synthetic route from in-stock building blocks. Tools like V-SYNTHES [8] use synthon-based hierarchical search to navigate these spaces efficiently, docking only a tiny fraction (<0.1%) of the full library. The result: when a hit emerges from screening, its synthesis route is already known and validated, with typical success rates >80%. Make is no longer a separate problem — it is a precondition of the search space.

**Paradigm B: Synthesis-aware generative models.** Rather than generating molecular structures and retrofitting synthesis, newer models generate synthetic pathways directly. SynFormer [4] (Gao, Luo & Coley, MIT) uses a transformer architecture to produce sequences of reactions from commercially available building blocks, ensuring that every generated molecule is synthesizable by construction. The Shoichet lab (UCSF) has pioneered **bespoke library design** — constructing custom virtual libraries of millions of molecules around underexplored but synthetically accessible scaffolds, then screening them with structure-based docking. Their work on tetrahydropyridines [6] and isoquinuclidines [7] demonstrated that bespoke libraries can achieve hit rates of 30–50%, far exceeding standard library screening, while accessing chemical space that standard on-demand libraries miss due to their bias toward common reaction types (71% of Enamine REAL uses amide couplings).

These paradigms share a key insight: **"design, then figure out synthesis" is being replaced by "design within synthesizable space."** The most sophisticated version of this idea is SPARROW [5] (Fromer & Coley), which jointly optimizes which molecules to synthesize and test by considering both expected information gain and synthetic cost — including the exploitation of shared intermediates across a batch of molecules. SPARROW directly bridges Design, Make, and Test in a single optimization framework.

```
  The Shift in Small Molecule Make:

  Traditional:  Design ──▶ "Can we make this?" ──▶ Retrosynthesis ──▶ Synthesis
                                  ↑ often fails

  Current:      Design within synthesizable space ──▶ Synthesis (route pre-validated)
                (on-demand libraries, SynFormer, bespoke libraries)
```

#### 3A.2 The Remaining Gap: When Validated Routes Still Fail

The absorption of Make into Design is impressive but incomplete. On-demand libraries report >80% synthesis success rates — which means ~20% still fail. Bespoke libraries use modular reactions with broad substrate scope, but specific building block combinations can introduce steric clashes, electronic mismatches, or competing side reactions that the enumeration logic does not capture.

The remaining gaps fall into two categories:

**Synthesis failure prediction.** Current retrosynthetic tools (SynPlanner, ASKCOS, AiZynthFinder) generate routes that are chemically plausible — each step has literature precedent. But precedent-based plausibility is a low bar. A Suzuki coupling that works for simple substrates may fail when the substrate contains a free amine that poisons the palladium catalyst. What is needed is a **synthesis outcome predictor** that estimates success probability, expected yield, and likely failure points for a specific route applied to a specific substrate.

The data requirements are substantial. This model would need training on Electronic Lab Notebook (ELN) records that include both successful and failed syntheses. The publication bias in chemistry is extreme: journals publish successful syntheses but reject failed ones. Pharmaceutical companies have accumulated large ELN datasets internally (GSK: >5 million reactions; AstraZeneca, Roche, and Pfizer have similar repositories), but these remain proprietary. ChemSpace Copilot's [2] integration of SynPlanner represents a starting point; the next step is adding a confidence layer that predicts which proposed routes are likely to succeed in practice.

**Reaction condition optimization.** For syntheses that proceed but with suboptimal yield, optimizing conditions (temperature, solvent, catalyst, concentration) is time-consuming. Bayesian optimization can identify optimal conditions in 5–15 experiments where traditional screening requires 50–100 [12]. An AI agent that combines automated literature extraction, BO-driven experimental design, and robotic execution could compress condition optimization from weeks to days. Several groups — Aspuru-Guzik's self-driving labs (Toronto), the Acceleration Consortium, Emerald Cloud Lab — are building toward this, but standardized APIs for robotic synthesis platforms remain absent.

#### 3A.3 Data Landscape

Small molecule synthesis benefits from the richest data ecosystem in drug discovery:

| Resource | Scale | Content |
|----------|-------|---------|
| Reaxys / SciFinder | >100M reactions | Published reactions with conditions |
| Open Reaction Database (ORD) | ~1M reactions | Structured, machine-readable |
| Enamine REAL Space | 78B virtual molecules | Validated routes from building blocks |
| Pharma ELN (proprietary) | 1–10M per company | Success/failure with conditions |
| USPTO patent reactions | ~3M | Extracted from patents |

The public data is sufficient for retrosynthetic planning. The critical gap is **failure data** — the ELN records of syntheses that didn't work, which would enable failure prediction models but remain locked in proprietary systems.

### 3B. Protein Make

#### 3B.1 The Sequence Is the Manufacturing Instruction

For protein therapeutics, the relationship between Design and Make is fundamentally different from small molecules — and fundamentally tighter.

When a chemist designs a small molecule, the target structure and its synthesis are separable problems. The same molecule can be made by multiple routes; if one fails, alternatives exist. The molecular structure determines the drug's properties, while the synthesis route is a means to an end.

For a protein, **the amino acid sequence simultaneously determines everything**: binding function, folding stability, expression level in the host cell, aggregation propensity, viscosity at high concentration, immunogenicity, and serum half-life. There is no "alternative synthesis route" for a protein that doesn't express — the molecule itself must be redesigned. This makes Design and Make inseparable in a way that has no analog in small molecule chemistry.

```
  Small Molecule:
    Structure ──▶ Properties (binding, ADMET)
    Structure ──▶ Synthesis route 1, route 2, route 3... (separable)

  Protein:
    Sequence ──▶ Properties (binding, stability, function)
    Sequence ──▶ Expression, folding, aggregation, immunogenicity (inseparable)
    Sequence IS the manufacturing instruction
```

This tight coupling means that accelerating protein Make is not primarily about optimizing the production process (though that matters) — it is about **predicting manufacturability from sequence and incorporating those predictions into Design.**

#### 3B.2 Developability: The State of Prediction

The biologics field uses the term **developability** to encompass the full set of properties that determine whether a protein candidate can be manufactured, formulated, and administered as a drug:

| Property | Why It Matters | Prediction Maturity |
|----------|---------------|-------------------|
| Expression yield | Must produce enough protein cost-effectively | Low |
| Thermal stability (Tm) | Shelf life, cold chain requirements | Moderate |
| Aggregation propensity | Safety (immunogenicity), formulation | Moderate |
| Viscosity | Subcutaneous injection requires <20 cP at 100+ mg/mL | Low |
| Chemical stability | Oxidation, deamidation degrade product | Low–Moderate |
| Immunogenicity | Anti-drug antibodies reduce efficacy | Low |
| Polyreactivity | Off-target binding indicates poor specificity | Moderate |

Computational tools exist for assessing these properties. The Therapeutic Antibody Profiler (TAP, Oxford OPIG) [10] flags antibodies with atypical CDR properties relative to clinical-stage molecules. CamSol predicts solubility. FoldX estimates stability changes from mutations. Aggrescan3D identifies aggregation-prone surface patches.

However, **ML-based developability prediction remains unreliable.** The FLAb2 benchmark [9] — the largest public antibody fitness benchmark with data from >4 million antibodies across 32 studies — found that protein AI models produce statistically non-significant correlations for **80% of developability datasets.** The 2025 Ginkgo Datapoints AbDev Competition, a blinded benchmark across five properties using clinical antibodies, reported best-case Spearman correlations of 0.71 for hydrophobicity but only 0.31 for expression titer and 0.34 for self-association. Cross-validation scores consistently exceeded test performance, indicating overfitting and poor out-of-distribution generalization.

The root cause is data scarcity. Unlike binding affinity (where ChEMBL contains millions of data points) or structure prediction (where the PDB provides hundreds of thousands of structures), standardized developability data is sparse, heterogeneous, and mostly proprietary. Each company measures different properties, under different conditions, with different assay formats — making aggregation across organizations nearly impossible.

#### 3B.3 Integrating Developability into Design

Given that sequence determines manufacturability, the natural response is to incorporate developability predictions into the Design stage. Two paradigms are emerging:

**Post-hoc filtering (traditional).** Generate candidates optimized for binding affinity, then filter using TAP, CamSol, or other tools to remove candidates with poor predicted developability. This is simple to implement but wasteful — many candidates are discarded, and the design space explored is not guided by developability.

**Co-optimization during generation (emerging).** Integrate developability as an objective or constraint directly within the generative model. Recent approaches include guided diffusion with Soft Value-based Decoding (SVDD), which biases antibody generation toward favorable developability without retraining the base model, and constrained preference optimization frameworks (AbNovo) that fine-tune generators with binding affinity as reward while enforcing biophysical constraints. The Sormanni group [11] demonstrated automated simultaneous optimization of stability and solubility, validating on approved therapeutics — with the critical finding that mutations improving one property often harm another, making co-optimization essential.

**Construct design automation** addresses a different aspect of protein Make: the decisions that surround the sequence itself. Expression system, affinity tag, linker design, signal peptide, codon optimization, culture conditions, and purification strategy are currently chosen by institutional knowledge and trial-and-error. An AI recommender trained on even a few thousand expression experiments (with metadata on conditions and outcomes) could propose construct designs with predicted expression level, solubility, and purification difficulty.

#### 3B.4 The Remaining Gaps

Even with developability-aware design, several gaps persist:

**Expression failure is multi-causal.** A protein may fail to express for reasons spanning codon usage bias, mRNA secondary structure near the ribosome binding site, co-translational folding kinetics, chaperone overload, or host-cell proteolysis. Current models treat these as independent features; the interactions between them are poorly understood and hard to model with limited data.

**Scale-up is nonlinear.** A protein that expresses well in shake flasks (100 mL) may behave differently in a bioreactor (10 L or 1000 L). Oxygen transfer, mixing dynamics, nutrient gradients, and cell viability all change with scale. This process-dependent variability has no direct analog in small molecule synthesis, where scale-up follows more predictable engineering principles.

**Data scarcity is the fundamental bottleneck.** Unlike small molecules (with Reaxys, ORD, and patent databases providing hundreds of millions of reaction records), protein expression data is scarce and fragmented:

| Resource | Scale | Content |
|----------|-------|---------|
| SAbDab (antibody structures) | ~8,000 | Structures only, no developability |
| TDCommons developability | ~2,400 | Binary classification, limited scope |
| FLAb2 (benchmark) | 4M+ antibodies, 32 studies | Heterogeneous, multi-property |
| Ginkgo AbDev | 246 clinical antibodies | 5 properties, small scale |
| Pharma internal | Thousands per company | Not shared |

Building a community resource for protein developability data — analogous to what ChEMBL did for bioactivity or ORD for reactions — is arguably the single highest-leverage investment for accelerating protein Make.

### 3C. Protocol Design and the Physical Interface

Regardless of modality, connecting computational agents to physical laboratory operations remains a shared challenge. This challenge has two layers: generating the experimental protocol itself, and executing it on physical equipment.

#### Protocol Design Agents

Before a synthesis or assay can be run, someone must design the protocol: what reagents, in what order, at what concentrations, with what controls, for how long. This is traditionally a manual task requiring domain expertise — and it is a surprisingly large fraction of the Make/Test timeline.

**Biomni** [17] (Huang, Leskovec et al., Stanford) demonstrates that general-purpose biomedical AI agents can generate wet-lab protocols at expert level. Built on an LLM reasoning core (Claude 3.7) with access to 150 specialized biomedical tools, 105 software packages, and 59 databases, Biomni autonomously designed a complete molecular cloning protocol — oligo design, Golden Gate assembly, heat-shock transformation, and sequencing validation — that was executed in the lab and produced correct results confirmed by Sanger sequencing. In a blinded benchmark of 10 cloning tasks, Biomni matched the accuracy of an experienced PostDoc while significantly outperforming a trainee-level scientist.

The relevance to DMT acceleration is direct. In each cycle, protocol design consumes time and expertise:

- **Make protocols**: What expression system, what culture conditions, what purification strategy? (biologics) What solvent, temperature, catalyst loading, workup procedure? (small molecules)
- **Test protocols**: What assay format, what concentration range, what controls, what readout? (Section 4.1)
- **Workflow scheduling**: In what order should experiments be run? Which can be parallelized? What are the dependencies?

An AI agent that generates publication-quality protocols — with appropriate controls, reagent specifications, and step-by-step instructions — removes a bottleneck that is easy to overlook because it is diffuse: spread across many small decisions rather than concentrated in a single slow step. Biomni's architecture (retrieval-augmented planning + code-based execution) provides a template for how such agents might integrate into DMT pipelines.

#### The Physical Interface

The deeper challenge is connecting protocol design to physical execution. Zhavoronkov et al. [1] envision "humanoid-in-the-loop" — robotic systems executing AI-designed protocols with human oversight. This highlights the current reality: most laboratories operate with equipment designed for human operators, not computational clients.

The interface problem has several layers:

1. **Equipment control.** Robotic platforms need standardized APIs for experiment submission and result retrieval. Currently, most lab equipment communicates through vendor-specific software with no programmatic interface.
2. **Protocol translation.** Converting a computational "recipe" into executable instructions for specific equipment — pump settings, valve positions, timing sequences that vary by platform. This is the gap between Biomni's protocol (human-readable) and a robotic system's instructions (machine-readable).
3. **Error handling.** Detecting when a physical experiment has gone wrong (precipitation, equipment malfunction, contamination) and deciding whether to retry, modify, or escalate to a human.
4. **Sample tracking.** Maintaining chain of custody from virtual design to physical sample to test result, with full provenance for regulatory and IP purposes.

These are engineering challenges, not research problems — but they are prerequisites for any AI system that aims to directly accelerate Make. The gap between "AI designs a protocol" and "AI executes a protocol" is fundamentally an infrastructure gap — and Biomni's demonstration that the first half is already achievable makes the second half more urgent.

---

## 4. What Would Direct Acceleration of Test Look Like?

### 4.1 Assay Design Agents

The Test stage begins with assay design: choosing what to measure, how to measure it, and what controls to include. This is currently an expert task, drawing on deep knowledge of the target biology, available assay technologies, and practical constraints (budget, timeline, sample quantity).

An **assay design agent** would operate analogously to Google's DORA or Co-Scientist, but specialized for experimental design rather than hypothesis generation:

- **Input:** Target profile, candidate molecules (with predicted properties), available equipment, budget constraints, project stage
- **Output:** Recommended assay panel with protocol details, control design, sample requirements, estimated cost and timeline, expected information gain

The agent would need to reason about:

- Which assay format (biochemical, cell-based, biophysical) is most informative at this stage of the project
- What concentration range to test, given predicted potency and solubility (avoiding the common mistake of testing above the solubility limit)
- Which controls are necessary to distinguish genuine activity from artifacts (aggregation, fluorescence interference, redox cycling)
- How to design the experiment for maximum statistical power given sample constraints
- Which assays to run in parallel vs. sequentially based on decision dependencies

Consider a concrete example. A team has 15 candidate molecules from an AI design round targeting a novel kinase. The assay design agent might recommend:

```
  Recommended Test Panel (Stage: Hit Validation)
  ─────────────────────────────────────────────────

  Tier 1 (all 15 compounds, 1 week):
    - TR-FRET kinase activity assay (10-point dose-response)
    - Kinetic solubility (nephelometry)
    - Estimated cost: $2,400 | Turnaround: 5 days

  Tier 2 (compounds passing Tier 1, ~5-8 expected):
    - NanoBRET cellular target engagement
    - Selectivity panel (5 closest kinases)
    - Microsomal stability (human, mouse)
    - Estimated cost: $8,000 | Turnaround: 10 days

  Tier 3 (top 2-3 compounds):
    - SPR binding kinetics (full kinetic characterization)
    - Cell viability (counter-screen)
    - Permeability (PAMPA or Caco-2)
    - Estimated cost: $5,000 | Turnaround: 14 days

  Controls: staurosporine (positive), DMSO vehicle,
  known inhibitor at 3 concentrations (reference standard)
```

This kind of structured experimental planning currently requires a senior scientist with years of kinase program experience. Encoding this expertise into a recommender system would democratize access to high-quality experimental design and reduce wasted experiments.

### 4.2 Intelligent Experiment Selection

A common proposal for accelerating the Test stage is active learning — selecting the subset of candidates that maximizes information gain rather than testing all of them or just the top-ranked ones. This is a valid idea, but it is important to recognize what it does and does not do. Active learning **does not make any individual experiment faster.** An SPR assay still takes the same time whether the compound was selected by active learning or by a chemist's intuition. What active learning does is **reduce the number of DMTA cycles** needed to reach a given optimization target — it is fundamentally a Learn-stage intervention that improves inter-cycle efficiency, not a Test-stage acceleration.

We discuss active learning in its proper context — as part of the "lab-in-the-loop" paradigm — in Section 5.

### 4.3 Multi-Readout Integration

A real drug discovery campaign does not produce a single number per molecule. It produces a profile across dozens of measurements:

```
  Molecule X-42 — Full Profile:
  ───────────────────────────────────────────────────────────────
  Target engagement:
    Binding affinity (SPR):      Ki = 12 nM          ✓ (target: <50 nM)
    Cell potency (reporter):     IC50 = 340 nM       ✗ (target: <100 nM)
    Residence time (SPR):        τ = 45 min           ✓ (target: >30 min)

  Selectivity:
    Kinase panel (50 kinases):   S(10) = 0.08        ✓ (target: <0.15)
    hERG IC50:                   >30 μM              ✓ (target: >10 μM)

  ADMET:
    Metabolic stability (HLM):   t½ = 45 min         ~ (target: >60 min)
    Metabolic stability (MLM):   t½ = 22 min         ✗ (target: >30 min)
    Permeability (PAMPA):        Papp = 8 × 10⁻⁶    ✓ (target: >5 × 10⁻⁶)
    Solubility (kinetic):        23 μg/mL            ✗ (target: >50 μg/mL)
    Plasma protein binding:      fu = 0.03           ~ (target: >0.05)

  Safety:
    CYP inhibition (3A4):       IC50 = 8 μM          ~ (target: >10 μM)
    Ames test:                   Negative             ✓
  ───────────────────────────────────────────────────────────────
  Overall assessment: Promising potency and selectivity, but cell
  potency gap suggests permeability or efflux issue. Solubility
  limits formulation options. Priority: improve cell penetration
  while maintaining binding affinity.
```

Interpreting this profile — deciding what to optimize next, which properties to prioritize, where to accept trade-offs — is the core skill of medicinal chemistry. It requires integrating quantitative data with qualitative understanding of structure–property relationships, historical precedent, and project-specific constraints. A single property deficiency might be tolerable if other properties compensate; two moderate deficiencies might be worse than one severe one if they affect the same downstream endpoint.

Current AI systems typically treat each endpoint independently: one model for binding, another for ADMET, another for selectivity. **Multi-readout integration** — taking the full experimental profile and recommending the next optimization direction — is a much harder problem that current systems barely attempt. The challenge is not just technical but epistemological: optimizing a molecule requires understanding causal relationships between structural features and properties, not just correlations.

An agent that could ingest a full assay profile and produce structured recommendations ("the 28x potency drop from biochemical to cellular suggests either poor permeability or active efflux; the PAMPA value is adequate, so test for P-gp efflux. If confirmed, reduce the number of hydrogen bond donors while maintaining the aminopyrimidine pharmacophore") would directly accelerate the Learn→Design transition, even if the experimental execution itself remains unchanged.

This is perhaps where large language models could contribute most directly. The reasoning required — integrating diverse data types, drawing on broad medicinal chemistry knowledge, and generating actionable hypotheses — maps naturally onto LLM capabilities. The barrier is not the reasoning ability but the structured integration of quantitative assay data with the LLM's chemical knowledge.

### 4.4 "PoseBusters for Experimental Design"

PoseBusters [13] validated docking poses by checking physical plausibility — flagging poses with steric clashes, impossible bond geometries, or violated interaction constraints. The analogy extends naturally to experimental design: a quality filter that catches errors before they consume physical resources.

Examples of detectable errors:

- **Incompatible assay conditions.** Testing a compound above its kinetic solubility limit yields unreliable dose-response curves. If predicted solubility is 10 μM and the top assay concentration is 100 μM, the dose-response will be artifactually flat.
- **Missing controls.** An assay design without a positive control, without appropriate vehicle controls for compound solubility, or without a counter-screen for assay-specific artifacts (e.g., fluorescence quenching in a fluorescence-based assay).
- **Statistical underpowering.** Too few replicates to detect the expected effect size at the desired significance level. With n=2 replicates and expected variability of 20%, a 2-fold difference may not reach statistical significance.
- **Redundant experiments.** Proposing to test two compounds that differ only in a position far from the binding site, when existing SAR data already shows that position is tolerant. The information gain is near zero.
- **Temporal conflicts.** Specifying an assay readout time that is incompatible with the expected kinetics. A 30-minute kinase assay for a slow-binding inhibitor with t½ > 2 hours will underestimate potency.

These checks are not AI-hard — many are rule-based or statistical. But implementing them as an automated filter in the DMT pipeline would prevent wasted experiments, just as PoseBusters prevents wasted follow-up on physically impossible binding poses. The analogy is precise: both serve as reality checks between a computational stage and a physical one.

---

## 5. Closing the Loop — The Learn Step and Lab-in-the-Loop

### 5.1 From Manual Interpretation to Automated Model Update

The Learn step connects Test back to Design, completing the cycle. Today, this connection is largely manual:

```
  Current State:
  Test results → Medicinal chemist reviews → Mental model update →
  → Intuition-guided design → Next cycle

  Desired State:
  Test results → Automated ingestion → Model fine-tuning →
  → Updated predictions → AI-guided design → Next cycle
  (with human oversight at decision checkpoints)
```

The manual path is not just slow — it is lossy. Human experts develop intuitions from experience, but those intuitions are difficult to transfer, may contain biases, and are limited by working memory. A chemist running three concurrent projects cannot hold the full SAR of all three in mind simultaneously. An automated Learn step would preserve the full information content of experimental results, update quantitative models in a principled way, and make the updated predictions immediately available to the Design agent.

The current state of Learn is also inconsistent. Different chemists interpret the same data differently. Organizational knowledge is lost when team members change. The rationale behind design decisions is often undocumented. An automated Learn step creates an auditable record of how each experimental result influenced subsequent design choices.

### 5.2 Lab-in-the-Loop: The Learn Step Made Concrete

The most complete realization of automated Learn to date is **lab-in-the-loop** [16], a framework developed by Prescient Design at Genentech. Lab-in-the-loop orchestrates generative models, multi-task property predictors, active learning selection, and in vitro experimentation in a semi-autonomous iterative loop. It is the clearest demonstration of what the Learn step looks like when done well — and it shows why Learn, not Test, is where active learning belongs.

```
  Lab-in-the-Loop Architecture (Prescient Design):

  ┌───────────────────────────────────────────────────────────┐
  │                    DESIGN (minutes)                        │
  │  Generative models (dWJS, SeqVDM, LaMBO-2, PropEn, DyAb) │
  │  → ~30,000 candidate variants per lead                    │
  └──────────────────────┬────────────────────────────────────┘
                         ▼
  ┌───────────────────────────────────────────────────────────┐
  │                    LEARN/SELECT                            │
  │  Multi-task property predictor (Cortex/LBSTER)            │
  │  → Predict affinity, expression, non-specificity          │
  │  Active learning selection (NEHVI acquisition function)   │
  │  → Rank and select batch for experimental testing         │
  │  OOD detection + chemical liability filtering             │
  └──────────────────────┬────────────────────────────────────┘
                         ▼
  ┌───────────────────────────────────────────────────────────┐
  │                    MAKE (days)                             │
  │  Linear DNA expression workflow (HEK293, 1 mL scale)     │
  │  Purification, OD quantification                          │
  └──────────────────────┬────────────────────────────────────┘
                         ▼
  ┌───────────────────────────────────────────────────────────┐
  │                    TEST (days)                             │
  │  SPR binding kinetics (ka, kd, KD)                        │
  │  Expression yield, non-specificity (BV ELISA)             │
  └──────────────────────┬────────────────────────────────────┘
                         ▼
  ┌───────────────────────────────────────────────────────────┐
  │                    LEARN/UPDATE                            │
  │  Ingest experimental data → Retrain property predictor    │
  │  Update generative model priors                           │
  │  → Next round with improved models                        │
  └──────────────────────┬────────────────────────────────────┘
                         │
                         └──▶ Next cycle (4-6 weeks per round)
```

Applied to four clinically relevant antibody targets (EGFR, IL-6, HER2, OSM), the system designed and tested over **1,800 unique variants** across **four rounds**. The results demonstrate the power of iterative learning:

| Round | Mutation budget | % designs with ≥3× better binding |
|-------|----------------|-----------------------------------|
| 1 | ≤6 edits from lead | Baseline (no prior project data) |
| 2 | ≤8 edits | Improving |
| 3 | ≤12 edits | Improving |
| 4 | ≤12 edits | >26% |

Best binders reached the **therapeutically relevant ~100 pM range**, representing 3–100× improvements over starting leads. Critically, the system maintained expression yield and developability — the multi-task predictor simultaneously optimized for affinity, expression, and non-specificity, avoiding the trap of improving binding at the expense of manufacturability.

**Why this is a Learn-stage story, not a Test-stage story.** The SPR assays in each round took the same amount of time regardless of whether the variants were selected by active learning or by random sampling. What improved across rounds was not experimental speed but **model quality**: the property predictor became more accurate with each round of experimental data, the generative models explored more productive regions of sequence space, and the selection strategy (NEHVI acquisition function) became better at identifying the Pareto frontier of affinity vs. expression. The Make and Test steps were unchanged — what changed was the intelligence of the Design→Learn feedback loop.

This clarifies a broader point. **Active learning does not accelerate the DMT cycle by making Make or Test faster. It accelerates the overall campaign by reducing the total number of cycles needed.** Exscientia's achievement of reaching development candidates with 150–250 compounds instead of 500–1,200 is the same principle: each cycle takes the same time, but fewer cycles are needed because each one is more informative.

### 5.3 Project-Specific Fine-Tuning

Foundation models for drug design are trained on public data that may be only loosely related to the specific target and chemical series of a given project. ChEMBL contains broad bioactivity data, but a project targeting a novel kinase allosteric site will find little relevant data. As a project generates its own experimental data, that data becomes the most valuable signal for the next design cycle.

Lab-in-the-loop demonstrates this concretely: Round 1 begins with **no labeled neighborhood data** — the property predictor relies entirely on pre-training. By Round 4, the predictor has been refined on hundreds of project-specific data points and produces substantially better-calibrated predictions. This is project-specific fine-tuning in action.

**The spectrum of adaptation strategies:**

| Approach | Data Required | Latency | Risk | Best When |
|----------|--------------|---------|------|-----------|
| Full fine-tuning | 1,000+ data points | Hours | Overfitting | Late-stage optimization |
| LoRA / adapter tuning | 100–500 data points | Minutes | Moderate | Mid-campaign |
| In-context learning | 10–50 data points | Seconds | Limited capacity | Early hits |
| Retrieval-augmented | Any | Seconds | Retrieval quality | Cross-project transfer |
| Meta-learning | Pre-trained on many projects | Minutes | Requires diverse data | New target class |

The right approach depends on the project stage. Early in a campaign, when only a handful of compounds have been tested, in-context learning or retrieval augmentation may be sufficient. As data accumulates, adapter-based fine-tuning becomes viable. Full fine-tuning is rarely appropriate for project-level data due to overfitting risk — 200 data points cannot meaningfully update millions of model parameters.

The key insight is that **the Learn step should be continuous, not episodic.** Every new data point should propagate into updated predictions, changing the priority queue for the next round of Design. This requires infrastructure for streaming experimental data into model update pipelines — infrastructure that most drug discovery organizations do not yet have.

### 5.4 The Model Collapse Risk

An automated DMT loop creates a feedback circuit: the Design model generates candidates, which are tested, and the results are used to update the Design model. This is precisely the setup for **model collapse** — the well-documented failure mode where a model trained on its own outputs progressively loses diversity and accuracy.

Lab-in-the-loop provides a concrete example of both the risk and its mitigation. The Prescient Design team observed that some leads with starting affinities ≥8.3 pKD showed minimal improvement across rounds — suggesting that the optimization had converged to local optima in sequence space. Their mitigation strategy was to ensemble multiple generative models (dWJS, SeqVDM, LaMBO-2, PropEn, DyAb), each exploring orthogonal regions of sequence space, and to use OOD (out-of-distribution) detection to flag predictions in unexplored regions.

In the drug discovery context more broadly, model collapse manifests as:

- **Narrowing chemical diversity.** The model converges on a small region of chemical/sequence space, ignoring potentially superior solutions elsewhere.
- **Amplified biases.** Systematic errors in the training data are reinforced rather than corrected by project data that was itself generated from biased models.
- **False confidence.** The model becomes highly confident about molecules similar to those it has already seen, while remaining unreliable in unexplored regions.

The antidote is experimental data serving as a **ground truth anchor** — each cycle of physical testing provides measurements independent of the model's predictions. Additional safeguards from the lab-in-the-loop experience include:

1. **Ensemble generation.** Multiple generative models explore different regions of sequence/chemical space, reducing dependence on any single model's biases.
2. **Expanding mutation budgets.** Progressively allowing more edits per round (6 → 8 → 12 in lab-in-the-loop) forces the system to explore further from the starting point.
3. **Out-of-distribution detection.** Flagging when the model makes predictions far from its training data, with appropriate uncertainty quantification.
4. **Multi-objective selection.** Using acquisition functions like NEHVI that optimize the Pareto frontier across multiple properties, rather than greedy optimization on a single objective.
5. **Human-in-the-loop review.** Regular expert review of the model's design tendencies to catch narrowing that automated metrics might miss.

This creates a productive tension: the DMT loop should be as fast and automated as possible, but it must include periodic injections of experimental reality and deliberate exploration.

---

## 6. The Orchestration Architecture

### 6.1 Lessons from ChemSpace Copilot

ChemSpace Copilot's [2] architecture provides a useful template for DMT orchestration, despite operating only within the Design stage. Its key elements:

- **Master Agent** that coordinates workflow and maintains conversation state
- **Specialist tools** (GTM visualization, SynPlanner, descriptor calculation) accessed through a unified interface
- **Observe–Plan–Act–Reflect loop** that structures each interaction into distinct phases
- **Persistent state** that carries context across multiple actions

The Observe–Plan–Act–Reflect pattern is particularly instructive. In the Design stage, each phase takes seconds. But the same pattern can structure longer timescale operations:

```
  DMT-Level Observe–Plan–Act–Reflect:

  OBSERVE: Review results from last cycle, current pipeline status,
           model predictions, available resources
           (automated data ingestion + dashboard)

  PLAN:    Determine next actions across all active cycles
           - Which new designs to generate
           - Which syntheses to prioritize
           - Which assays to run on newly available compounds
           (agent reasoning + human approval)

  ACT:     Execute planned actions
           - Submit design jobs (synchronous, minutes)
           - Queue syntheses (asynchronous, days-weeks)
           - Initiate assays (asynchronous, days-weeks)
           (mixed sync/async execution)

  REFLECT: Evaluate outcomes against predictions
           - Did synthesis succeed as predicted?
           - Do assay results match model expectations?
           - What did we learn? Update models accordingly
           (automated analysis + human interpretation)
```

Extending this pattern to the full DMT cycle raises specific challenges that the Design-only case does not encounter.

### 6.2 The Time Asymmetry Problem

The most fundamental architectural challenge is managing operations that span vastly different timescales.

```
  Timeline of a Single DMT Cycle:

  Day 0        Day 1        Day 7        Day 14       Day 28       Day 42
  ─────────────────────────────────────────────────────────────────────────
  │ Design    │             │            │             │             │
  │ (minutes) │             │            │             │             │
  │           │◀── Make ──▶│            │             │             │
  │           │ (1 week)    │            │             │             │
  │           │             │◀────── Test ──────────▶│             │
  │           │             │ (3 weeks)  │             │             │
  │           │             │            │             │◀── Learn ─▶│
  │           │             │            │             │ (2 weeks)   │
```

A Design-stage agent operates synchronously: submit a prompt, get results in seconds or minutes. A Make-stage agent operates asynchronously over days or weeks. A Test-stage agent may need to wait months for results.

This means the orchestration system must:

1. **Manage concurrent cycles.** While one cycle waits in the Make stage, the system should be running Design for the next cycle, analyzing Test results from a previous cycle, and updating models from completed Learn stages. A typical active project might have 5–10 cycles in flight simultaneously at different stages.

2. **Maintain long-lived state.** Unlike a chatbot conversation that lasts minutes, a DMT orchestrator must maintain project state over months — tracking which molecules are in synthesis, which assays are pending, which results have been incorporated into models, and which design decisions are still valid given new data.

3. **Handle interrupts and replanning.** Experimental results arrive asynchronously and may invalidate ongoing work. A breakthrough result from Cycle N may make the molecules designed in Cycle N+2 obsolete. The orchestrator must detect these events and trigger replanning — potentially canceling syntheses that are no longer worth completing.

4. **Prioritize across cycles.** With limited laboratory resources, the orchestrator must decide which cycle's synthesis or testing should take priority. A compound from Cycle 3 that addresses a critical SAR question might be more important than a compound from Cycle 5 that is merely incrementally better.

```
  Concurrent Cycle Management (Realistic View):

  Cycle 1:  D ──── M ──────── T ─────────────── L ─┐
  Cycle 2:       D ──── M ──────── T ──────────── L ├──▶ feeds Cycle 6
  Cycle 3:            D ──── M ─ [FAIL] ──▶ redesign│
  Cycle 4:                 D ──── M ──────── T ─────┘
  Cycle 5:                      D ──── M ──────── ...

  ──────────────────────────────────────────────────────▶ Time

  Note: Cycle 3 synthesis failed; agent detects failure,
  redesigns molecule, and re-enters Make. Cycle 6 is
  informed by results from Cycles 1, 2, and 4.
```

### 6.3 API-Based Integration

Zhavoronkov et al. [1] envision a future where all components of the drug discovery pipeline — legacy databases, laboratory equipment, AI models, and human experts — are connected through APIs. This mirrors a pattern from the Protein AI series: "Keep the Trunk, Replace the Head" — using a stable core representation while swapping out modular components.

Applied at the pipeline level:

| Component | API Inputs | API Outputs |
|-----------|-----------|-------------|
| Design Agent | Target structure, constraints, project history | Ranked candidate list with predicted properties |
| Retrosynthesis | Candidate molecule, lab constraints | Proposed routes with feasibility scores |
| Synthesis Robot | Route specification, reagents, scale | Status updates, yield, purity, analytical data |
| Assay Platform | Compounds, assay specification, controls | Raw data, processed results, QC flags |
| Analysis Agent | All results for cycle N, model state | Updated model, recommendations for N+1 |
| Human Expert | Recommendations, data summary | Approval, modifications, strategic direction |

The critical insight is that **the API contracts — the input/output schemas — matter more than the implementations behind them.** A retrosynthesis module powered by SynPlanner can be swapped for one powered by ASKCOS without changing the orchestrator, as long as both conform to the same schema. This modularity enables gradual adoption: organizations can automate one stage at a time while maintaining the same orchestration framework.

This is the "Keep the Trunk, Replace the Head" principle operating at the pipeline level rather than the model level. The orchestration layer is the trunk; the individual tools and models are swappable heads. Just as Pairformer emerged as the universal trunk for structure prediction, a well-designed orchestration protocol could become the universal trunk for drug discovery pipelines.

### 6.4 Infrastructure Requirements

Making this architecture real requires several infrastructure components that are currently underdeveloped:

**Standardized data formats.** The field lacks agreed-upon schemas for representing experimental results in machine-readable form. A binding assay result might be stored as a Ki value, an IC50, a percent inhibition at a single concentration, or a full dose-response curve — each in a different format depending on the laboratory and assay platform. Proposals like the Pistoia Alliance's IDMP data standards and the Allotrope Data Format represent steps in the right direction but lack universal adoption.

**Experiment–computation provenance tracking.** In an automated loop, it becomes critical to know: which model version generated which candidates, which experimental data was used to train which model, and which decisions were made by AI vs. human experts. This provenance chain is essential for debugging (why did the model suggest this molecule?), regulatory compliance (IND filings require clear documentation of the design rationale), and intellectual property tracking (who/what invented this compound?).

**Graceful degradation.** Not every stage will be automated simultaneously. The orchestration system must handle hybrid workflows where some stages are AI-driven and others are manual, with human checkpoints at configurable positions in the loop. An organization might start by automating Design and Learn while keeping Make and Test manual, then gradually automate each stage as infrastructure matures.

**Security and access control.** Drug discovery data is commercially sensitive. An orchestration system must enforce access controls, audit logging, and data isolation between projects — requirements that are well-understood in enterprise software but often missing from research prototypes. The competitive nature of drug discovery means that a single data breach could expose a pipeline worth billions.

**Fault tolerance and recovery.** Physical experiments fail. Equipment breaks. Reagents expire. The orchestration system must handle these events gracefully — detecting failures, diagnosing causes, and either retrying, rerouting, or escalating to human intervention. This is fundamentally different from software orchestration where retries are cheap; a failed synthesis may mean days of lost work and wasted reagents.

---

## 7. Research Directions and Recommendations

The analysis above points to several concrete research directions that could meaningfully accelerate the DMT cycle beyond the Design stage.

### 7.1 Build Benchmarks for Make and Test

The Design stage benefits from CASP, PoseBusters, MOSES, and dozens of other benchmarks. Make and Test have nothing comparable. Without benchmarks, progress is unmeasurable and academic incentives do not align with the most impactful work.

Two specific benchmark proposals:

1. **Synthesis Outcome Prediction Benchmark (SOPBench).** Curate a dataset of proposed synthetic routes paired with outcomes (success/failure, yield, purity, failure mode). Pharmaceutical companies collectively have millions of such records in their ELNs. A consortium-based approach — where companies contribute anonymized data to a shared benchmark without revealing proprietary molecules — could unlock this resource. The anonymization could involve replacing specific structures with fingerprints or learned embeddings that preserve reaction-relevant features while obscuring molecular identity.

2. **Lab-in-the-Loop Benchmark (LitLBench).** Using retrospective data from completed drug discovery campaigns, evaluate iterative optimization strategies by simulating sequential rounds: given the results from rounds 1–N, which candidates would each strategy select for round N+1, and how does the resulting campaign compare in total cycles to reach a given optimization target? Prescient Design's lab-in-the-loop work demonstrates the value of this approach but uses proprietary data. A public benchmark with temporal ordering would enable academic groups to develop and compare Learn-stage strategies.

### 7.2 Release Anonymized ELN Data

The Make stage suffers from a data scarcity that is artificial: the data exists but is locked in proprietary systems. Initiatives like the Open Reaction Database (ORD) have made progress for published reactions, but the most valuable data — failed syntheses, actual yields, optimized conditions — remains proprietary.

Pharmaceutical companies should consider releasing anonymized ELN datasets, possibly with structure obfuscation to protect IP while preserving reaction-level patterns. The precedent exists: ChEMBL aggregates bioactivity data from patents and publications; a similar effort for synthesis outcomes would be transformative. The Open Reaction Database provides a starting framework, but needs scale — currently it contains ~1M reactions, mostly from publications. A contribution of even 1% of a major pharma company's ELN would multiply this several-fold.

### 7.3 Open-Source DMT Orchestration Frameworks

No open-source framework currently exists for orchestrating the full DMT cycle. ChemSpace Copilot's agentic architecture is a starting point, but it operates only within Design. A full DMT orchestrator would need to handle:

- Asynchronous multi-timescale operations (seconds to months)
- Concurrent cycle management with inter-cycle dependencies
- Model versioning and continuous update pipelines
- Human-in-the-loop checkpoints with configurable autonomy levels
- Equipment integration adapters (abstraction layer over heterogeneous lab hardware)
- Provenance tracking and audit logging
- Fault tolerance and graceful degradation

Building this as an open-source framework — analogous to what LangChain/LangGraph did for LLM applications, or what MLflow did for ML experiment tracking — would lower the barrier to entry and accelerate adoption across the industry. The framework need not automate Make and Test themselves; it needs only to provide the scaffolding for connecting automated Design and Learn with manual (or semi-automated) Make and Test.

### 7.4 Academia–Industry Collaboration Models

The current division of labor — academia develops Design models, industry generates Make and Test data — creates a structural bottleneck. Neither side alone can accelerate the full DMT cycle.

A more productive collaboration model would pair:

| Academic Contribution | Industry Contribution | Joint Output |
|----------------------|----------------------|-------------|
| Active learning algorithms | Retrospective project data | Validated AL benchmarks |
| Synthesis prediction models | ELN data (anonymized) | SOPBench benchmark |
| Orchestration frameworks | Workflow requirements | Open-source DMT orchestrator |
| Multi-readout integration methods | Multi-assay datasets | Profile interpretation tools |

The key enabler is data. Academic groups have the modeling expertise but lack the experimental data that would make their models relevant to real drug discovery. Industry has the data but often lacks the bandwidth or incentive to develop novel ML methods. Structured collaboration — with clear data sharing agreements and co-publication rights — could unlock both resources.

### 7.5 Three Actionable Research Proposals

1. **Synthesis Success Predictor.** Train a model on ELN data (even if initially proprietary and internal) that predicts the probability of synthetic route success, conditioned on the specific substrate and available equipment. Validate against held-out synthesis attempts. The key metric: if the model can reliably identify routes with <10% success probability, it saves weeks of wasted effort per project. Even a binary classifier (feasible / likely to fail) with >80% accuracy would be transformative. Start with one reaction type (e.g., Suzuki coupling) where sufficient data exists, then generalize.

2. **Assay Design Recommender.** Build a system that, given a target profile and project stage, recommends an assay panel with protocol parameters. Train on retrospective project records where experienced teams made these decisions. Evaluate by comparing AI recommendations to expert decisions on held-out projects. The comparison metric: do AI-recommended panels generate equivalent or better decision quality (measured by next-cycle outcomes) compared to expert-designed panels?

3. **Cross-Stage Lab-in-the-Loop Optimization.** Extend the lab-in-the-loop paradigm beyond the Learn→Design interface to encompass Make and Test decisions. Current implementations (Prescient Design) optimize which candidates to design and test, but not which to synthesize (considering synthetic difficulty and time) or which assays to run (considering information value and cost). A full DMTL optimizer would jointly select molecules, synthesis routes, and assay panels using a unified acquisition function — as SPARROW [5] does for the Design→Make interface, but extended across all stages.

---

## Closing

The protein AI revolution has largely been a Design revolution. Models that generate novel proteins, predict structures, and score candidates have improved by orders of magnitude over five years. The convergence on Pairformer architectures, flow matching, and joint structure-sequence generation has created a powerful and increasingly commoditized toolkit for molecular design.

But drug discovery is not a Design problem — it is a cycling problem. The value of a perfect design model is bounded by the speed at which its outputs can be made and tested. A team that generates 1,000 perfect candidates per hour but can only synthesize and test 10 per month has a Design surplus and a Make/Test bottleneck. Further Design improvements yield diminishing returns.

Zhavoronkov et al. [1] see this clearly: their vision of pharmaceutical superintelligence is fundamentally about orchestration, not about any single model. The prompt-to-drug pipeline requires every stage to be computationally accessible — not just Design, but Make, Test, and Learn. ChemSpace Copilot [2] demonstrates that agentic architectures can automate complex multi-tool workflows — but its scope stops at the lab door, precisely where the hard problems begin.

The next frontier is not a better generative model or a more accurate structure predictor. It is the infrastructure, benchmarks, data, and orchestration systems that connect Design to physical reality. Specifically:

- **Benchmarks** for synthesis success prediction and lab-in-the-loop optimization
- **Data sharing** frameworks that unlock proprietary ELN and assay data for model training
- **Orchestration architectures** that manage the time asymmetry of concurrent DMT cycles
- **Interface standards** that connect computational agents to physical laboratory operations
- **Continuous learning** pipelines that update models as experimental data arrives

The research community that builds these bridges — between computation and experiment, between Design and Make and Test — will define the next era of AI-driven drug discovery.

The tools to design molecules have outpaced the tools to make and test them. Closing that gap is the highest-leverage problem in the field today.

---

**References**

1. Zhavoronkov, A., Gennert, D. & Shi, J. ["From Prompt to Drug: Toward Pharmaceutical Superintelligence."](https://pubs.acs.org/doi/10.1021/acscentsci.5c00867) *ACS Central Science* (2026).
2. Orlov, A. A. et al. ["ChemSpace Copilot: Agentic AI for Interactive Visualization and Exploration of Chemical Space."](https://chemrxiv.org/engage/chemrxiv/article-details/67ee3e4b81d2151a022157c5) *ChemRxiv* (2026).
3. Ryu, S. ["Protein AI Series, Part 0–9."](https://github.com/SeongokRyu/research_brainstorming/tree/main/blog_draft/protein_ai_series) *Research Brainstorming Blog* (2025–2026).
4. Gao, W., Luo, S. & Coley, C. W. ["Generative Artificial Intelligence for Navigating Synthesizable Chemical Space."](https://www.pnas.org/doi/10.1073/pnas.2415665122) *PNAS* (2025).
5. Fromer, J. C. & Coley, C. W. ["Computer-aided multi-objective optimization of synthetic pathways and conditions."](https://www.nature.com/articles/s43588-024-00639-y) *Nature Computational Science* (2024).
6. Kaplan, A. L. et al. ["Bespoke library docking for 5-HT2A receptor agonists with antidepressant activity."](https://www.nature.com/articles/s41586-022-05258-z) *Nature* 610, 582–591 (2022).
7. Vigneron, S. F. et al. ["Docking 14 Million Virtual Isoquinuclidines against the μ and κ Opioid Receptors."](https://pubs.acs.org/doi/10.1021/acscentsci.5c00052) *ACS Central Science* (2025).
8. Sadybekov, A. A. et al. ["Synthon-based ligand discovery in virtual libraries of over 11 billion compounds."](https://www.nature.com/articles/s41586-021-04220-9) *Nature* (2022).
9. Chungyoun, M. & Gray, J. J. ["FLAb2: Benchmarking Reveals That Protein AI Models Cannot Yet Consistently Predict Developability."](https://www.biorxiv.org/content/10.1101/2025.12.27.696706v1) *bioRxiv* (2025).
10. Rabia, L. A. et al. ["Blueprint for antibody biologics developability."](https://pmc.ncbi.nlm.nih.gov/articles/PMC10012935/) *mAbs* (2023).
11. Sormanni, P. et al. ["Automated optimisation of solubility and conformational stability of antibodies."](https://www.nature.com/articles/s41467-023-37668-6) *Nature Communications* (2023).
12. Shields, B. J. et al. ["Bayesian reaction optimization as a tool for chemical synthesis."](https://www.nature.com/articles/s41586-021-03213-y) *Nature* (2021).
13. Gao, W. et al. ["PoseBusters: AI-based docking methods fail to generate physically valid ligand poses or generalise to novel sequences."](https://pubs.rsc.org/en/content/articlelanding/2024/sc/d3sc04185a) *Chemical Science* (2024).
14. Genheden, S. et al. ["AiZynthFinder: a fast, robust and flexible open-source software for retrosynthetic planning."](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-020-00472-1) *Journal of Cheminformatics* (2020).
15. Jumper, J. et al. ["Highly accurate protein structure prediction with AlphaFold."](https://www.nature.com/articles/s41586-021-03819-2) *Nature* (2021).
16. Frey, N. C., Hötzel, I., Stanton, S. D. et al. ["Lab-in-the-loop therapeutic antibody design with deep learning."](https://www.biorxiv.org/content/10.1101/2025.02.19.639050v1) *bioRxiv* (2025).
17. Huang, K., Leskovec, J. et al. ["Biomni: A General-Purpose Biomedical AI Agent."](https://www.biorxiv.org/content/10.1101/2025.05.30.656746v1) *bioRxiv* (2025).
18. Hillisch, A. et al. ["Augmenting DMTA using predictive AI modelling at AstraZeneca."](https://www.sciencedirect.com/science/article/pii/S1359644624000709) *Drug Discovery Today* (2024).
19. Blakemore, D. C. et al. ["Closing the Loop: Developing an Integrated Design, Make, and Test Platform for Discovery."](https://pmc.ncbi.nlm.nih.gov/articles/PMC6580368/) *ACS Medicinal Chemistry Letters* (2019).
20. Morgan, P. et al. ["Can the flow of medicines be improved?"](https://pubmed.ncbi.nlm.nih.gov/22227532/) *Drug Discovery Today* (2012).
