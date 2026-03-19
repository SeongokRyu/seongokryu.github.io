---
title: "From Millions to Billions — and Back: Ultra-Large Virtual Screening and the Case for Bespoke Libraries"
date: 2026-03-18 10:00:00 +0900
categories: [Drug Discovery, Small Molecule Design]
tags: [virtual-screening, docking, chemical-library, drug-discovery, small-molecule]
math: true
---

In 2019, docking 170 million molecules against a protein target was a landmark achievement [1]. By 2025, the make-on-demand chemical space available for virtual screening has grown to **78 billion** molecules — a 450-fold expansion in six years. Tools like V-SYNTHES [9] navigate 36 billion compounds by docking less than 0.1% of them, and ML-accelerated methods [16] reduce computational cost by another 1,000-fold.

This explosive scaling rests on a compelling principle: **the larger the library, the better the hits**. More compounds searched means higher hit rates, stronger affinities, and more novel chemotypes. The evidence for this is now substantial, spanning dozens of targets and multiple research groups.

But scaling has also exposed inconvenient truths. Larger libraries accumulate more artifacts — molecules that score well for the wrong reasons [2]. The dominant on-demand libraries are biased toward a narrow set of reaction types, systematically missing entire classes of bioactive scaffolds. And every screening campaign, no matter how vast, operates under the same fundamental constraint: the docking scoring function is an imperfect proxy for actual binding.

This essay traces the evolution of ultra-large virtual screening (ULVS) from its origins to the present, examines the complementary strategy of bespoke library design, and confronts the fundamental limitations that define the field's next frontier.

---

## 1. The Scaling Principle — Why Bigger Libraries Find Better Hits

### 1.1 Structure-Based Virtual Screening in Brief

Structure-based virtual screening (SBVS) uses the three-dimensional structure of a protein target to computationally evaluate how well candidate molecules fit into a binding site. The process involves:

- **Docking**: placing each molecule into the binding site in many orientations and conformations
- **Scoring**: estimating the binding strength using a physics-based or empirical scoring function
- **Ranking**: ordering all molecules by score and selecting top candidates for experimental testing

The key variable is the **size of the chemical library**. A larger library means more chemical diversity explored, increasing the probability of finding molecules that complement the target's shape and electrostatics.

### 1.2 The Landmark Study: Lyu et al., Nature 2019

The study that established ULVS as a paradigm came from the Shoichet lab at UCSF [1]. Key facts:

- **Library**: 170 million make-on-demand compounds from Enamine, constructed from 130 well-characterized reactions and 70,000 building blocks
- **Targets**: D4 dopamine receptor and AmpC β-lactamase
- **Scale**: 10.7 million unique scaffolds — the vast majority unavailable in any physical collection

Results:

| Target | Compounds Tested | Hit Rate | Best Affinity | Notable |
|--------|-----------------|----------|---------------|---------|
| D4 dopamine receptor | 549 | Monotonic with score | **180 pM** | Subtype-selective agonist |
| AmpC β-lactamase | 44 | High | 77 nM | Novel phenolate chemotype, among most potent non-covalent inhibitors known |

The most important finding was not any single hit, but the **systematic relationship between library size and hit quality**:

- Hit rates fell monotonically with docking score rank
- The score-versus-rank curve predicted ~453,000 ligands for D4 in the full library
- 81 new chemotypes were discovered, 30 with submicromolar activity
- Larger libraries consistently produced better-fitting, more potent molecules

This established the scaling principle: **more compounds → better hits**.

### 1.3 The Scaling Law: Lyu, Irwin & Shoichet, Nat Chem Biol 2023

Four years later, the same group provided a theoretical analysis of what happens as libraries grow from millions to billions [2]. This paper is essential for understanding both the promise and the limits of ULVS.

Key findings:

- **Docking scores improve log-linearly** with library size — each 10-fold expansion yields a roughly constant improvement in the best score
- **Artifacts grow with library size** — rare molecules that exploit scoring function weaknesses become more frequent as the library expands
- **Bio-likeness decreases dramatically** — the proportion of "bio-like" molecules in tangible make-on-demand libraries is **19,000-fold lower** than in curated "in-stock" collections
- Better-fitting molecules are found, but so are more false positives

The implication: **bigger is better, but with diminishing returns and growing noise**. This insight frames everything that follows — it is simultaneously the justification for ULVS and the theoretical basis for the bespoke library alternative (Section 5) and the scoring function critique (Section 6).

### 1.4 The Rise of Make-on-Demand Chemical Space

What made ULVS possible was not just faster computers, but a revolution in chemical space itself. Make-on-demand libraries — vast collections of molecules that do not physically exist but can be reliably synthesized from validated reactions and in-stock building blocks — changed the game:

- **Enamine REAL Space**: the largest, built from validated reactions with >80% synthesis success rates. Every molecule comes with a pre-defined synthesis route.
- **ZINC-22** [21]: the Shoichet/Irwin lab's public database — 5.9 billion compounds with precomputed 3D structures, 54.9 billion in 2D representation
- **WuXi GalaXi**: 26 billion tangible molecules via BioSolveIT partnership

The key innovation: instead of screening molecules that already exist in physical collections, screen molecules that **could exist** — enormously expanding the accessible chemical space while guaranteeing synthesizability.

---

## 2. Scaling Up — Platforms, Libraries, and Applications

### 2.1 The ULVS Timeline

The growth trajectory from 2019 to 2025 tells a story of relentless expansion:

| Year | Study | Library Size | Target | Best Affinity | Key Innovation | Ref |
|------|-------|-------------|--------|---------------|----------------|-----|
| 2019 | Lyu et al. | 170M | D4, AmpC | 180 pM | First ULVS at scale | [1] |
| 2020 | Stein et al. | 150M | Melatonin MT1 | 470 pM | Circadian clock modulation | [5] |
| 2020 | VirtualFlow | 1.4B | KEAP1 | 114 nM | Open-source platform | [3] |
| 2022 | V-SYNTHES | 11B (synthon) | Cannabinoid | 33% hit rate | Synthon-based navigation | [9] |
| 2022 | Chem space dock | ~1B | ROCK1 | 39% hit rate | Fragment linker vectors | [11] |
| 2023 | VirtualFlow 2.0 | 69B ready-to-dock | — | — | GPU, RNA/DNA targets | [4] |
| 2025 | CB1R screen | 74M | Cannabinoid CB1R | 0.95 nM | Reduced side effects | [6] |
| 2025 | GPR139 screen | 235M | Orphan GPCR | 160 nM | Cryo-EM validation | [7] |
| 2025 | V-SYNTHES2 | 36B | Multiple | — | Full automation, $300 | [10] |

### 2.2 VirtualFlow: Democratizing ULVS

**VirtualFlow** (Gorgulla et al., Nature 2020) [3] was the first open-source platform purpose-built for ULVS:

- **Scale**: screened 1.4 billion compounds against KEAP1 (NRF2 pathway regulator)
- **Architecture**: two modules — VFLP (ligand preparation) + VFVS (virtual screening)
- **Performance**: perfect scaling to 160,000+ CPUs
- **Result**: 114 nM binder (iKeap1), disrupting KEAP1-NRF2 interaction
- **Key validation**: confirmed that hit quality improves with library size, now at 10x the scale of Lyu 2019

**VirtualFlow 2.0** (bioRxiv 2023) [4] extended this further:

- 69 billion ready-to-dock molecules (full Enamine REAL Space)
- 1,500+ docking methods — enabling RNA, DNA, and non-protein targets
- GPU acceleration and ARM CPU support
- Adaptive Target-Guided Virtual Screens (ATG-VS): focused sampling of ultra-large libraries for a fraction of the cost of exhaustive screening

### 2.3 The Library Explosion

The growth of make-on-demand chemical space has been exponential:

| Library | 2019 | 2022 | 2025 |
|---------|------|------|------|
| Enamine REAL | ~170M | ~11B | **78B** |
| WuXi GalaXi | — | — | **26B** |
| ZINC (2D) | ~750M | ~37B | **54.9B** |
| ZINC (3D, ready-to-dock) | — | — | **5.9B** |

This polynomial growth is driven by two factors:

- **More validated reactions**: Enamine expanded from 130 to 185+ reaction types
- **More building blocks**: from ~70,000 to 115,000+ unique reactants
- Each new reaction type, combined with existing building blocks, produces a combinatorial explosion of new products

### 2.4 Recent Applications from the Shoichet Lab

By 2025, ULVS has moved from landmark demonstrations to a routine tool for GPCR drug discovery:

**Cannabinoid CB1R agonists** [6]:
- 74 million compounds docked
- 46 synthesized and tested → 9 active (**20% hit rate**)
- Structure-based optimization → compound '1350: **0.95 nM**, full CB1R agonist
- Strong analgesic effects in mice with **2–20x therapeutic window** over hypolocomotion, sedation, and catalepsy — a key limitation of existing cannabinoid drugs

**Orphan receptor GPR139** [7]:
- 235 million compounds docked
- 68 evaluated → 5 full agonists (**7% hit rate**)
- Potencies from 160 nM to 3.6 μM
- Structure-guided optimization identified one of the most potent GPR139 agonists known
- Cryo-EM structure of receptor-ligand complex confirmed predicted binding mode

**MRGPRD** [8]:
- High-affinity agonists discovered, revealing recognition motifs for this pain-related GPCR
- Published in Cell Reports 2024

These examples show that ULVS is no longer experimental — it is a production-grade tool for GPCR-focused drug discovery. But they also underscore a limitation: all targets above are GPCRs with well-defined, deep binding pockets. Targets with shallow, solvent-exposed, or allosteric sites remain more challenging.

### 2.5 The Brute-Force Wall

Despite platform advances, screening 78 billion compounds by docking each one individually remains computationally prohibitive for most groups. At ~1 second per compound per CPU core:

- 78B compounds × 1 sec = ~2,500 CPU-years
- At cloud pricing (~$0.03/CPU-hour): **~$660,000** per screen

This is within reach of large pharma but not of academic labs. Even with GPU acceleration, brute-force at this scale is unsustainable as libraries continue to grow toward trillions. New strategies are needed — and they have arrived.

---

## 3. Navigating Vast Space Without Enumerating It

The central insight of post-2022 ULVS is that you do not need to dock every molecule to find the best ones. Instead, you can exploit the **combinatorial structure** of make-on-demand libraries:

```
  Traditional (brute-force):
    Enumerate all N molecules → Dock each one → Rank
    Cost: O(N)  — infeasible beyond ~1-2B

  New paradigm (fragment/synthon-based):
    Dock building blocks / fragments / synthons: O(√N)
    Combine best fragments → Dock selected combinations
    Cost scales with #synthons, not #products
```

This paradigm shift has produced multiple complementary approaches.

### 3.1 Synthon-Based Approaches

**V-SYNTHES (Sadybekov et al., Nature 2022)** [9]

The first synthon-based method for navigating ultra-large make-on-demand libraries:

- **Library**: 11 billion compounds in Enamine REAL Space
- **Method**:
  1. Decompose library into scaffolds + synthons (building blocks)
  2. Dock synthons into the binding site → identify best scaffold-synthon "seed" combinations
  3. Iteratively elaborate seeds by adding building blocks, docking at each stage
  4. Only the final top-scoring complete molecules are fully evaluated
- **Key result**: <0.1% of the library was actually docked
- **Targets**: cannabinoid receptors — **33% experimental hit rate**, including 14 sub-micromolar ligands

```
  V-SYNTHES Workflow:

  Step 1: Dock synthons (thousands)
           ┌─────┐  ┌─────┐  ┌─────┐
           │ S1  │  │ S2  │  │ S3  │  ...
           └──┬──┘  └──┬──┘  └──┬──┘
              ▼        ▼        ▼
  Step 2: Score scaffold + synthon combinations
           Select best "seeds"

  Step 3: Elaborate seeds with additional building blocks
           Dock elaborated molecules

  Step 4: Full docking of top candidates only
           → Experimental testing
```

The computational advantage is dramatic: cost scales with the number of synthons (~100K), not the number of products (billions).

**V-SYNTHES2** [10]

The next generation, published in 2025:

- **Scale**: 36 billion compounds (expanded REAL Space)
- **Automation**: CapSelect — geometry-based, fully automated fragment selection
- **Validation**: GPCRs, RNA-binding sites, shallow pockets, phospholipid-binding enzymes
- **Cost**: 40,000 CPU hours / **~$300** — making billion-scale screening accessible to academic labs
- Available via GitHub (katritchlab/V-SYNTHES)

**Chemical Space Docking (Lyu et al., Nat Commun 2022)** [11]

A complementary approach from the Shoichet lab:

- Uses **fragment linkers as reaction vectors** for product library enumeration
- Each fragment represents a library of ~10K molecules
- Applied to ~1 billion compounds against ROCK1 kinase
- **39% hit rate** (27/69 purchased compounds had Ki < 10 μM)
- Two X-ray structures confirmed docked poses
- Cost scales with number of reagents, orders of magnitude faster than traditional docking

### 3.2 Virtual Fragment Screening → Elaboration

While synthon-based methods start from the library's combinatorial structure, an alternative starts from fragments and grows into chemical space.

**Luttens et al., Nat Commun 2025** [12]

- **Target**: OGG1 (8-oxoguanine DNA glycosylase) — a "difficult" target involved in DNA repair, cancer, and inflammation
- **Approach**: three-phase strategy
  1. **Fragment screening**: dock 14 million fragments to OGG1 → test 29 top-ranked → 4 confirmed binders (X-ray crystallography validates binding modes)
  2. **Fragment elaboration**: pattern matching based on validated hit topology → coupling reactions → 6,720 virtual products not available in make-on-demand catalogs
  3. **Docking-guided selection**: dock elaborated products → synthesize selected compounds → **165-fold potency improvement** over initial fragments
- **Result**: 600 nM IC50 inhibitors with anti-inflammatory and anti-cancer activity in cells
- **Connection to V-SYNTHES**: same underlying principle ("dock small, then grow") but starting from validated fragment hits rather than the library's synthon decomposition

### 3.3 ML-Accelerated Approaches

A different strategy: use machine learning to predict which molecules would score well, avoiding the need to dock most of them.

**Deep Docking (Gentile et al., ACS Cent Sci 2020)** [13]

- Train QSAR model on docking scores from a random subset → predict scores for remaining compounds → iteratively refine
- **50–100x acceleration** without significant loss of top hits
- COVID-19 application: 40 billion compounds screened in 19 days (previously estimated at 10 years)
- Open-source (D5 platform)

**HASTEN (Kalliokoski, 2021→2023)** [14]

- Chemprop-based ML model + iterative docking
- **1.56 billion** Enamine REAL compounds: docking only 1% achieved **90% recall** of the true top-1,000 hits
- **99% reduction** in required docking experiments
- Open-source (GitHub)

**RosettaVS (Mulligan et al., Nat Commun 2024)** [15]

- AI-accelerated virtual screening with a unique feature: **receptor flexibility modeling**
- Applied to multi-billion compound libraries
- Results:
  - KLHDC2: 7 hits out of 50 tested (**14% hit rate**), all single-digit μM
  - NaV1.7: 4 hits out of 9 tested (**44% hit rate**)
- X-ray crystallography validated KLHDC2 docking pose
- Completed in less than 7 days
- Open-source

**ML-Guided Docking (Luttens et al., Nat Comp Sci 2025)** [16]

- CatBoost classifier trained on docking scores of 1 million compounds
- **Conformal prediction framework** for calibrated uncertainty — selects which compounds to dock from the full library
- Applied to **3.5 billion** compounds: **1,000-fold computational cost reduction**
- Discovered a dual-target ligand modulating A2A adenosine and D2 dopamine receptors
- Notable: from the same Carlsson lab that published the fragment elaboration work [12] — demonstrating that fragment-based and ML-acceleration strategies are **complementary, not competing**

**ML Acceleration Comparison:**

| Method | Mechanism | Speedup | Recall of Top Hits | Library Tested | Open-Source |
|--------|-----------|---------|-------------------|----------------|-------------|
| Deep Docking [13] | QSAR iterative filter | 50–100x | High | 40B (COVID) | Yes |
| HASTEN [14] | Chemprop iterative | 99% reduction | 90% (top-1K) | 1.56B | Yes |
| RosettaVS [15] | AI + receptor flexibility | — | — | Multi-billion | Yes |
| ML-guided [16] | CatBoost + conformal pred. | 1,000x | — | 3.5B | — |

### 3.4 Ligand-Based Chemical Space Navigation

Not all navigation requires docking. Ligand-based methods use known active molecules as queries to search combinatorial chemical spaces.

**SpaceGrow (BioSolveIT, JCAMD 2024)** [17]:

- Shape-based 3D virtual screening in combinatorial fragment spaces
- **Billions of compounds on a single CPU** within hours
- Comparable pose reproduction to conventional superposition tools, superior ranking
- Exploits combinatorial structure: cost scales with synthon count, not product count

**SpaceLight** (BioSolveIT):
- Topological fingerprint similarity (ECFP4 + Connected Subgraph Fingerprints)
- Scaffold hopping capability

**infiniSee** (BioSolveIT):
- Fragment-space decomposition platform
- Navigates up to **10^15** compounds
- Supports multiple chemical spaces (Enamine, WuXi, Chemspace)

### 3.5 Summary: How the Approaches Compare

| Approach | Principle | Scale | Speed | Structure-Based? | Key Advantage | Key Limitation |
|----------|-----------|-------|-------|-----------------|---------------|----------------|
| Brute-force | Enumerate all, dock all | ~1–2B max | Slow | Yes | Complete coverage | Cost-prohibitive at scale |
| Synthon-based | Dock synthons, grow | 36B+ | Fast | Yes | Scales with √N | Limited to defined reaction space |
| Fragment→elaborate | Dock fragments, expand | 14M→billions | Medium | Yes | Validated starting points | Requires initial fragment hits |
| Chemical space dock | Fragment linker vectors | ~1B | Fast | Yes | Combines SBVS + combinatorics | Specific library architecture needed |
| ML-accelerated | Train on subset, predict | Billions | Very fast | Hybrid | Dramatic cost reduction | Inherits scoring function bias |
| Shape/fingerprint | Ligand similarity | Billions+ | Very fast | No | No target structure needed | Requires known active query |

The evolution is clear: from exhaustive enumeration (2019–2020) to intelligent navigation (2022–2025). But all approaches share a common dependency — they rely on the same underlying scoring functions. We return to this in Section 6.

---

## 4. The Bias Problem — Why Bigger Isn't Always Better

### 4.1 The Theory

The scaling law analysis by Lyu et al. [2] revealed a crucial nuance: as libraries grow, **quality and noise both increase**:

- Docking scores improve log-linearly — but the **marginal gain per additional 10-fold expansion decreases**
- Artifacts — molecules that score well due to scoring function weaknesses rather than genuine complementarity — accumulate
- The proportion of drug-like, bio-like molecules decreases dramatically as the space is filled with synthetically accessible but biologically implausible compounds

This means there is a **practical ceiling** to the value of library expansion, even before computational cost becomes limiting.

### 4.2 The Reaction-Type Bias

The structural composition of make-on-demand libraries is determined by the reactions used to construct them. This creates systematic blind spots.

**Enamine REAL Space composition**:
- **71.4% of molecules** are based on amide coupling reactions
- This produces predominantly **rod-like, linear molecules** with limited 3D complexity
- Scaffolds with high sp3 content, bridged ring systems, macrocycles, and natural-product-like topologies are systematically underrepresented

```
  Chemical Space Coverage:

  ┌─────────────────────────────────────────────────┐
  │           Theoretical Drug-Like Space            │
  │                                                  │
  │  ┌──────────────────────────┐                    │
  │  │   Enamine REAL (78B)     │                    │
  │  │   Broad but biased:      │   ○ ○             │
  │  │   - Amide-heavy          │    ○  ○ Bioactive │
  │  │   - Rod-like             │   ○ ○   scaffolds │
  │  │   - Low sp3              │    ○    not in    │
  │  │                          │   ○ ○   REAL      │
  │  └──────────────────────────┘    ○              │
  │                                                  │
  └─────────────────────────────────────────────────┘
```

### 4.3 Practical Consequences

This bias has real consequences for drug discovery:

- Many **GPCRs, ion channels, and PPI targets** prefer 3D-complex scaffolds that occupy distinct regions of chemical space
- **Natural product-like structures** — which have historically been a rich source of drugs — are poorly represented in make-on-demand libraries
- A library of 78 billion molecules can still miss the optimal chemotype for a given target if that chemotype requires reactions not in the library's repertoire

The uncomfortable conclusion: **78 billion compounds is not enough if the right chemical space is not included.** This is the theoretical justification for bespoke libraries.

---

## 5. Bespoke Libraries — Designing Chemical Space for a Target

### 5.1 The Concept

Bespoke libraries are custom-designed virtual compound collections built around specific chemical scaffolds that address the blind spots of standard make-on-demand libraries. The Shoichet lab at UCSF pioneered this approach, defining three criteria for a good bespoke library:

1. **Undersampled**: the scaffold is poorly represented in standard on-demand libraries (Enamine REAL, WuXi GalaXi)
2. **Readily synthesizable**: molecules can be constructed from a modular reaction scheme with available building blocks
3. **Bioactive chemotype**: the scaffold class has established biological relevance (natural products, known drug scaffolds, privileged structures)

The logic is simple: if the standard library is biased, build a better library for your specific target.

### 5.2 Case Study: Tetrahydropyridine Library for 5-HT2A

**Kaplan et al., Nature 2022** [18]

- **Scaffold**: tetrahydropyridines — well-suited to aminergic GPCRs but poorly sampled in billion-molecule general libraries
- **Library**: 75 million virtual molecules, constructed from modular reactions
- **Target**: serotonin 5-HT2A receptor (5-HT2AR)

Results:
- 17 initial molecules synthesized and tested → 4 with low-μM activity against 5-HT2A or 5-HT2B
- Structure-based optimization → agonists (R)-69 and (R)-70:
  - **41 nM and 110 nM EC50** at 5-HT2AR
  - Unusual signaling kinetics, different from classic psychedelic agonists
- **No psychedelic activity** — in contrast to all known 5-HT2AR agonists
- **Potent antidepressant activity** in mouse models: same efficacy as fluoxetine at **1/40th the dose**

Why this matters: standard ULVS against 5-HT2A with a general library would likely have missed these compounds — the tetrahydropyridine scaffold is simply not well-represented in Enamine REAL. The bespoke approach accessed a region of chemical space that billion-scale general screening could not reach.

### 5.3 Case Study: Isoquinuclidine Library for Opioid Receptors

**Vigneron et al., ACS Cent Sci 2025** [19]

- **Scaffold**: isoquinuclidines — [2.2.2] bicyclic amines with disk/sphere-like 3D geometry, high sp3 content
- **Library**: 14.6 million molecules from a modular four-component reaction (primary amines × enones × alkynes × activated alkenes)
- **Targets**: μ-opioid receptor (MOR) and κ-opioid receptor (KOR)

Why this scaffold:
- Nearly absent from existing libraries: ~95,000 out of 5.4 billion Enamine REAL compounds (0.002%)
- **98% of final compounds had unique anonymous graphs** in the Smallworld database — unprecedented chemical novelty
- Topologically complex, caged core — unlike the rod-like molecules dominating on-demand space

Results:
- 18 prioritized compounds synthesized → **9 active (50% hit rate)** — the highest reported for opioid receptor docking
- Initial hits: MOR Ki = 0.98 μM, KOR Ki = 1.2 μM
- Structure-based optimization (31 additional compounds, guided by cryo-EM):
  - **1,000-fold improvement** → sub-nM dual MOR antagonist / KOR inverse agonist
  - Best compound: 0.91 nM

In vivo results:
- Reversed morphine-induced analgesia (equivalent to naloxone)
- **Significantly less severe withdrawal symptoms** than naloxone
- **No conditioned-place aversion** (no dysphoria) — consistent with KOR inverse agonism
- Clinical potential for opioid overdose reversal with fewer side effects

Acknowledged limitation: "we did not have a method that would ensure that the isoquinuclidine scaffold we enumerated was appropriate for the opioid receptors, instead relying on gross apparent compatibility." Scaffold selection remains intuition-driven.

### 5.4 On-Demand ULVS vs Bespoke: A Comparison

| Dimension | On-Demand ULVS (Enamine REAL) | Bespoke Library (Shoichet lab) |
|-----------|-------------------------------|-------------------------------|
| **Size** | 78 billion | 14.6M–75M |
| **Diversity** | Broad but reaction-biased | Narrow but target-tailored |
| **Typical hit rate** | 0.01–1% (general); 7–44% (focused) | 30–50% |
| **Chemical novelty** | Moderate (common reactions) | High (98% unique graphs) |
| **Synthesis** | Pre-validated (>80% success) | Modular but per-case validation |
| **Scaffold selection** | Not needed (enumerate all) | Critical (requires expertise) |
| **3D complexity** | Low (amide-coupling dominated) | High (sp3-rich, bridged) |
| **Best for** | Initial broad exploration | Target-specific depth |

The key insight: **these approaches are not competing — they are complementary.** On-demand ULVS provides broad coverage; bespoke libraries provide targeted depth in regions that general libraries miss.

---

## 6. The Glass Ceiling — Fundamental Limitations of Docking

Every approach described above — brute-force, synthon-based, ML-accelerated, bespoke — shares a common dependency: they all rely on **docking scoring functions** to evaluate molecular fitness. This section examines why that dependency creates a fundamental ceiling on what ULVS can achieve.

### 6.1 Scoring Functions Are Not Binding Affinity Predictors

Docking scoring functions (Glide, AutoDock Vina, DOCK, Smina) were designed primarily for **pose prediction** — determining how a molecule sits in a binding pocket — not for **affinity ranking** — predicting which molecule binds most tightly.

The distinction matters:

- **Enrichment** (separating binders from non-binders): scoring functions perform reasonably well. This is why ULVS works — it identifies molecules that are more likely to bind than random compounds.
- **Ranking** (ordering binders by affinity): scoring functions perform poorly. The correlation between docking score and experimental binding affinity is typically weak (R² = 0.1–0.3 for congeneric series).

Practical implications:
- Hit rates of 20–50% in the best ULVS campaigns mean **50–80% of top-scored compounds are false positives**
- The scaling law analysis [2] shows that larger libraries find more artifacts — molecules that exploit scoring function weaknesses rather than genuinely complementing the target
- Scoring functions cannot reliably distinguish a 1 nM binder from a 1 μM binder — they can only distinguish binders from non-binders with moderate reliability

### 6.2 Rigid Receptors and Missing Physics

Most ULVS campaigns use a **single, rigid receptor conformation** — a necessary simplification for screening billions of compounds, but one that ignores fundamental biology.

**Induced fit and conformational selection**:
- Proteins are flexible. Ligand binding often triggers conformational changes that cannot be captured by a static structure.
- The isoquinuclidine study [19] provides a striking example: rigid docking produced a **180° pose error** for the initial hit. Only cryo-EM structures of the actual complex revealed the correct binding mode — including Asp3.32 displacement and membrane-proximal pocket opening that rigid docking could not predict.
- RosettaVS [15] introduced receptor flexibility modeling, improving results but at significantly higher computational cost.

**Solvation and water networks**:
- Structural water molecules in binding sites often mediate protein-ligand interactions and contribute substantially to binding affinity
- Water displacement entropy is a major thermodynamic component of binding
- Most scoring functions either ignore water entirely or treat it with crude implicit models

**Entropy contributions**:
- Ligand conformational entropy loss upon binding
- Protein side-chain entropy changes
- These contributions can be 1–5 kcal/mol — comparable to the total binding energy for many drug-like molecules — yet are poorly captured by scoring functions

**Non-classical interactions**:
- Halogen bonds, cation-π interactions, CH-π interactions, and sulfur-mediated contacts are increasingly recognized as important but are often absent or inaccurate in scoring functions

### 6.3 The Multi-Stage Scoring Funnel

In practice, successful ULVS campaigns do not rely on a single docking score. They use a multi-stage funnel with increasing accuracy and decreasing throughput:

```
  The Practical ULVS Pipeline:

  Stage 1: Fast docking (Vina, Glide HTVS)       Billions  → Millions
           Speed: ~1 sec/compound
           Purpose: Coarse filter

  Stage 2: Precise docking (Glide SP/XP)          Millions  → Thousands
           Speed: ~1 min/compound
           Purpose: Pose refinement, better scoring

  Stage 3: FEP / MM-GBSA rescoring                Thousands → Hundreds
           Speed: ~1 hour/compound
           Purpose: Physics-based affinity estimation

  Stage 4: Expert visual inspection                Hundreds  → Tens
           Speed: ~10 min/compound (human)
           Purpose: Chemical intuition, artifact removal

  Stage 5: Synthesis and experimental testing      Tens      → Hits
```

Most published ULVS studies report results primarily from Stages 1–2. Stage 3 (FEP/MM-GBSA) is computationally expensive but can significantly improve affinity predictions. Stage 4 — human expert review — remains essential and is rarely discussed in methods sections.

The gap between Stages 2 and 5 is where most value is lost. Automating and improving Stages 3–4 is arguably as important as scaling Stage 1.

### 6.4 ML Surrogates Inherit the Ceiling

ML-accelerated methods (Deep Docking [13], HASTEN [14], CatBoost [16]) achieve dramatic speedups by learning to predict docking scores without actually docking. But this creates a critical dependency:

**The ML model learns the scoring function, not reality.**

- If the scoring function overestimates binding for certain molecular features, the ML surrogate will too — but 1,000x faster
- Artifacts that exploit scoring function weaknesses will be efficiently identified and prioritized
- The ML model's accuracy is bounded above by the scoring function's accuracy

This is not an argument against ML acceleration — the speedups are genuine and valuable. But it is an argument for honesty about what is being optimized: **computational docking score, not experimental binding affinity.**

Additional ML-specific challenges:
- **Generalization**: models are target-specific; a surrogate trained for D4 dopamine receptor does not transfer to ROCK1 kinase
- **Uncertainty quantification**: the conformal prediction framework used by Luttens et al. [16] is a meaningful advance, providing calibrated confidence intervals. Most other ML surrogates lack such guarantees.
- **Domain shift**: as the ML model guides selection toward high-scoring regions, the distribution of compounds shifts away from the training set, potentially degrading predictions

### 6.5 Beyond Classical Docking — Emerging Alternatives

Several approaches aim to bypass or complement traditional scoring functions:

**End-to-end learned docking** (DiffDock, Uni-Mol, NeuralPLexer):
- Learn pose prediction and scoring directly from structural data, without physics-based scoring
- Potential to capture patterns that classical scoring functions miss
- Current limitations:
  - Not yet validated at ULVS scale
  - PoseBusters benchmarks reveal recurring physical validity issues (steric clashes, impossible geometries)
  - Training data bias toward crystallographic poses

**Co-folding / structure prediction** (AlphaFold3, Chai-1, Boltz-1):
- Predict protein-ligand complex structures directly from sequence and ligand information
- Could fundamentally bypass docking scoring functions
- Current limitations:
  - **Throughput**: minutes to hours per molecule vs. seconds for docking — incompatible with screening billions
  - Ranking capability is unclear — predicting a plausible pose is different from predicting relative binding affinity
  - Best suited for post-docking validation of top candidates rather than primary screening

**Physics-informed ML scoring**:
- Train scoring functions directly on experimental binding affinity data rather than on physics-based approximations
- The most direct path to raising the scoring ceiling
- Current bottleneck: insufficient high-quality training data linking 3D poses to experimental affinities at scale

**Comparison of scoring paradigms:**

| Approach | Scoring Basis | Throughput | Physical Validity | Maturity for ULVS |
|----------|-------------|-----------|------------------|-------------------|
| Classical docking | Empirical/physics | High (sec/mol) | Moderate | Mature |
| ML surrogate of docking | Learned from docking scores | Very high | Inherits docking | Mature |
| End-to-end learned | Learned from structural data | Medium | Variable | Early |
| Co-folding (AF3 etc.) | Structure prediction | Low (min/mol) | High | Not ready for ULVS |
| Physics-informed ML | Learned from experimental ΔG | Medium | High potential | Emerging |

### 6.6 The Core Message

> ULVS has revolutionized chemical space exploration, but it operates under a **glass ceiling defined by scoring function accuracy**. Screening 78 billion compounds is meaningless if the scoring function cannot distinguish genuine binders from artifacts. ML acceleration reduces cost by 1,000x but inherits the same ceiling. The next breakthrough will not come from screening more compounds — it will come from **scoring them more accurately**.

---

## 7. The Convergent Future

### 7.1 Fragment/Synthon Thinking as the Unifying Principle

At first glance, the approaches in Sections 3–5 appear diverse: synthon-based docking, fragment elaboration, ML acceleration, bespoke libraries. But they share a common intellectual structure:

**All decompose chemical space into building blocks and reassemble them intelligently.**

- V-SYNTHES: decomposes on-demand libraries into synthons, docks synthons, grows
- Chemical space docking: uses fragment linkers as reaction vectors
- Fragment→elaborate: docks fragments, validates experimentally, then grows
- Bespoke libraries: constructs a focused space from modular reactions and building blocks
- ML acceleration: learns which building block combinations are worth evaluating

This convergence suggests that **fragment/synthon-based thinking is the natural language for navigating combinatorial chemical space**, regardless of whether the goal is broad ULVS or focused bespoke screening.

### 7.2 Integrating On-Demand and Bespoke

The future likely combines both approaches in a two-stage strategy:

```
  Stage 1: Broad exploration in on-demand space
           - Screen billions via synthon-based / ML-accelerated methods
           - Identify active scaffolds and SAR trends
           - Assess which chemotypes are accessible vs. missing

  Stage 2: Focused exploration in bespoke space
           - For undersampled but promising chemotypes:
             build bespoke libraries around target-tailored scaffolds
           - Apply the same synthon-based navigation to bespoke space
           - Achieve 30-50% hit rates in focused chemical territory
```

This mirrors a well-established principle in optimization: **explore broadly, then exploit locally**.

### 7.3 Multi-Objective ULVS

Current ULVS optimizes for a single objective: docking score as a proxy for binding affinity. Real drug discovery requires simultaneous optimization of multiple properties:

- Binding affinity and selectivity
- ADMET properties (metabolism, permeability, solubility)
- Synthetic accessibility and cost
- Developability (for biologics)

SPARROW (Fromer & Coley, 2024) demonstrated joint optimization of molecular design and synthetic cost, including exploitation of shared intermediates. Extending this to the full ULVS pipeline — incorporating predicted ADMET and selectivity into the screening funnel — is a natural next step.

### 7.4 Adaptive Screening and Experimental Feedback

ULVS is currently a **one-shot** process: screen the library, pick top candidates, synthesize, test. The results inform the next project but not the current screen.

An adaptive approach would integrate experimental feedback:

1. **Round 1**: ULVS → synthesize and test top candidates
2. **Learn**: update scoring model with experimental binding data
3. **Round 2**: re-screen with improved model → focus on regions where model was wrong
4. Repeat

This connects directly to the lab-in-the-loop paradigm — where iterative model updates progressively improve hit quality across rounds. The combination of ULVS (broad chemical space coverage) with lab-in-the-loop (iterative refinement) is a powerful but unexplored intersection.

### 7.5 Automating the Full Pipeline

The practical ULVS pipeline (Section 6.3) has automated first stages but manual later stages. Full automation would require:

- **Automated FEP rescoring** of top docking hits (commercially available but expensive)
- **AI-based visual inspection** replacing or augmenting human expert review
- **Automated synthesis prioritization** considering route feasibility and cost
- **Integration with robotic synthesis and testing** for closed-loop discovery

### 7.6 The Role of AI in Bespoke Library Design

The critical bottleneck in bespoke library design is **scaffold selection** — choosing which underexplored scaffold to build a library around for a given target. This currently relies on human intuition, as the Shoichet lab has acknowledged [19].

AI could contribute through:
- **Generative models** (e.g., SynFormer) proposing novel synthesizable scaffolds that complement a target's binding site
- **Co-folding** (AlphaFold3, Chai-1) to pre-evaluate whether a scaffold can adopt a productive binding mode — catching induced fit issues before library construction
- **Systematic chemical space analysis**: identifying which regions of bioactive chemical space are undersampled by existing on-demand libraries

---

## 8. Open Questions

**Scaffold selection remains intuition-driven.** The bespoke library approach has produced remarkable results, but the initial choice of scaffold — "which underexplored chemotype should we build a library around?" — has no systematic method. The isoquinuclidine team acknowledged relying on "gross apparent compatibility." Automating this decision is arguably the highest-impact open problem in the field.

**Prospective validation is inconsistent.** Hit rates across ULVS studies range from 7% to 50%, but direct comparison is misleading — targets differ in difficulty, assay conditions vary, and "hit" definitions are not standardized. The field lacks a shared benchmark akin to CASP for structure prediction.

**Accessibility remains unequal.** VirtualFlow, HASTEN, and Deep Docking are open-source. V-SYNTHES is available via GitHub but Enamine REAL Space access requires commercial agreements. Schrödinger's Glide is proprietary. Academic groups can navigate billions of compounds for hundreds of dollars with V-SYNTHES2, but the full ecosystem of tools remains fragmented.

**ULVS optimizes for binding only.** Selectivity, ADMET properties, and developability are not part of the docking screen. A molecule that scores perfectly in docking may be metabolically unstable, impermeable, or toxic. Integrating multi-property prediction into the screening funnel — rather than applying it as a post-hoc filter — is an important but unsolved challenge.

**Standardization is lacking.** Different groups use different scoring functions, different library preparations, different hit rate definitions, and different experimental validation protocols. This makes systematic comparison of methods nearly impossible and slows the field's ability to identify which approaches genuinely outperform others.

**The environmental cost is underexplored.** Screening billions of compounds requires substantial computational resources. As libraries grow toward trillions, the energy consumption and carbon footprint of ULVS deserve consideration — both as a practical cost and as a factor in method design.

---

## Closing

Ultra-large virtual screening has transformed structure-based drug discovery. In six years, the accessible chemical space has grown from 170 million to 78 billion make-on-demand compounds. Synthon-based methods navigate this space by docking less than 0.1% of it. ML acceleration reduces computational cost by another 1,000-fold. The scaling principle — bigger libraries yield better hits — is well-established and continues to deliver potent, novel molecules against therapeutically important targets.

But two fundamental questions remain:

1. **What** chemical space to explore — the reaction-type bias of on-demand libraries means that larger is not always more diverse, and bespoke libraries have shown that small, carefully designed collections can outperform billion-scale screens for specific targets

2. **How** to evaluate what we find — the docking scoring function is the glass ceiling of the entire enterprise, and every acceleration method, from V-SYNTHES to ML surrogates, inherits its limitations

The field is evolving from a **competition of scale** to a **competition of intelligence**: not "how many compounds can we screen?" but "which compounds should we screen, and how accurately can we evaluate them?"

On-demand ULVS provides breadth. Bespoke libraries provide depth. Fragment/synthon thinking unifies both. And the next frontier — better scoring, multi-objective optimization, and experimental feedback loops — will determine whether virtual screening fulfills its promise of systematically discovering the best possible molecule for any target.

---

**References**

1. Lyu, J. et al. ["Ultra-large library docking for discovering new chemotypes."](https://www.nature.com/articles/s41586-019-0917-9) *Nature* 566, 224–229 (2019).
2. Lyu, J., Irwin, J. J. & Shoichet, B. K. ["Modeling the expansion of virtual screening libraries."](https://www.nature.com/articles/s41589-022-01234-w) *Nature Chemical Biology* 19, 712–718 (2023).
3. Gorgulla, C. et al. ["An open-source drug discovery platform enables ultra-large virtual screens."](https://www.nature.com/articles/s41586-020-2117-z) *Nature* 580, 663–668 (2020).
4. Gorgulla, C. et al. ["VirtualFlow 2.0 — The Next Generation Drug Discovery Platform Enabling Adaptive Screens of 69 Billion Molecules."](https://www.biorxiv.org/content/10.1101/2023.04.25.537981v1) *bioRxiv* (2023).
5. Stein, R. M. et al. ["Virtual discovery of melatonin receptor ligands to modulate circadian rhythms."](https://www.nature.com/articles/s41586-020-2027-0) *Nature* 579, 609–614 (2020).
6. Kaplan, A. L. et al. ["Virtual library docking for cannabinoid-1 receptor agonists with reduced side effects."](https://www.nature.com/articles/s41467-025-57136-7) *Nature Communications* (2025).
7. Fink, E. A. et al. ["Ultra-large virtual screening unveils potent agonists of the neuromodulatory orphan receptor GPR139."](https://www.nature.com/articles/s41467-025-66845-y) *Nature Communications* (2025).
8. Wang, S. et al. ["High-affinity agonists reveal recognition motifs for the MRGPRD GPCR."](https://www.cell.com/cell-reports/fulltext/S2211-1247(24)01293-2) *Cell Reports* (2024).
9. Sadybekov, A. A. et al. ["Synthon-based ligand discovery in virtual libraries of over 11 billion compounds."](https://www.nature.com/articles/s41586-021-04220-9) *Nature* 601, 452–459 (2022).
10. Sadybekov, A. A. et al. ["V-SYNTHES2 — the Next Generation Tool for Structure-based Virtual Screening of Giga-scale Chemical Spaces."](https://pubmed.ncbi.nlm.nih.gov/41282073/) (2025).
11. Lyu, J. et al. ["Chemical space docking enables large-scale structure-based virtual screening to discover ROCK1 kinase inhibitors."](https://www.nature.com/articles/s41467-022-33981-8) *Nature Communications* 13, 6447 (2022).
12. Luttens, A. et al. ["Virtual fragment screening for DNA repair inhibitors in vast chemical space."](https://www.nature.com/articles/s41467-025-56893-9) *Nature Communications* 16, 1741 (2025).
13. Gentile, F. et al. ["Deep Docking: A Deep Learning Platform for Augmentation of Structure Based Drug Discovery."](https://pubs.acs.org/doi/10.1021/acscentsci.0c00229) *ACS Central Science* 6, 939–949 (2020).
14. Kalliokoski, T. et al. ["Machine Learning-Boosted Docking Enables the Efficient Structure-Based Virtual Screening of Giga-Scale Enumerated Chemical Libraries."](https://pubs.acs.org/doi/10.1021/acs.jcim.3c01239) *Journal of Chemical Information and Modeling* (2023).
15. Mulligan, M. J. et al. ["An artificial intelligence accelerated virtual screening platform for drug discovery."](https://www.nature.com/articles/s41467-024-52061-7) *Nature Communications* (2024).
16. Luttens, A. et al. ["Rapid traversal of vast chemical space using machine learning-guided docking screens."](https://www.nature.com/articles/s43588-025-00777-x) *Nature Computational Science* 5, 301–312 (2025).
17. Lessel, U. et al. ["SpaceGrow: efficient shape-based virtual screening of billion-sized combinatorial fragment spaces."](https://link.springer.com/article/10.1007/s10822-024-00551-7) *Journal of Computer-Aided Molecular Design* (2024).
18. Kaplan, A. L. et al. ["Bespoke library docking for 5-HT2A receptor agonists with antidepressant activity."](https://www.nature.com/articles/s41586-022-05258-z) *Nature* 610, 582–591 (2022).
19. Vigneron, S. F. et al. ["Docking 14 Million Virtual Isoquinuclidines against the μ and κ Opioid Receptors."](https://pubs.acs.org/doi/10.1021/acscentsci.5c00052) *ACS Central Science* (2025).
20. Corrêa Veríssimo, G. et al. ["Ultra-Large Virtual Screening: Definition, Recent Advances, and Challenges in Drug Design."](https://onlinelibrary.wiley.com/doi/10.1002/minf.202400305) *Molecular Informatics* (2025).
21. Irwin, J. J. et al. ["ZINC-22 — A Free Multi-Billion-Scale Database of Tangible Compounds for Ligand Discovery."](https://pubs.acs.org/doi/10.1021/acs.jcim.2c01253) *Journal of Chemical Information and Modeling* (2023).
