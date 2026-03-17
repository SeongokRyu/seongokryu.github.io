---
title: "Protein AI Series Part 8: The Data Landscape"
date: 2026-03-17 10:20:00 +0900
categories: [Drug Discovery]
tags: [protein-ai, training-data, pdb, alphafold-database, knowledge-distillation]
math: true
---

**The Technical Evolution of Protein AI — A Record of Key Design Decisions**

*This is Part 8 of a 10-part series tracing the architectural choices behind modern protein structure prediction and design models.*

---

## The Core Question

**What training data do protein AI models use, where does it come from, and why do different models make radically different data choices?**

Part 7 examined the engineering required to train protein AI models — FlashAttention, FSDP, mixed precision, multi-stage pipelines. But engineering answers the question of *how* to train. This Part addresses the equally consequential question of *what* to train on.

The same Pairformer + Diffusion architecture, trained with different data, produces measurably different models. IsoDDE outperforms all open-source alternatives not through architectural innovation but through proprietary experimental data. SeedFold's ablation study shows that removing synthetic distillation data mid-training causes immediate accuracy degradation — the 180K experimental structures in the PDB are simply not enough. In protein AI, **data is architecture's equal as a determinant of model quality**, and increasingly, the primary axis of competitive differentiation.

---

## 1. The Data Pyramid: Four Orders of Magnitude

The data available for protein AI training spans four orders of magnitude in scale, with an inverse relationship between quantity and information density:

```
                    ┌─────────┐
                    │Functional│  ~10K–50K
                    │  Data    │  (K_d, K_i, IC50, ADMET)
                    ├─────────┤
                    │   PDB    │  ~220K
                    │Structures│  (experimental 3D coordinates)
                ┌───┴─────────┴───┐
                │   Synthetic      │  ~10M–200M
                │   Structures     │  (AF2/AF3/ESMFold predictions)
            ┌───┴──────────────────┴───┐
            │        Sequences          │  ~250M–2.5B
            │   (UniProt, BFD, MGnify)  │  (amino acid strings)
            └──────────────────────────┘
```

Each layer serves a distinct purpose in the training pipeline:

- **Sequences** (~250M–2.5B): Raw material for MSA construction and protein language model (PLM) pre-training. No structural information, but enormous evolutionary signal through covariation.
- **Synthetic structures** (~10M–200M): Predicted by teacher models (AF2, AF3, ESMFold). Dramatically expand the effective training set but inherit the teacher's systematic biases.
- **Experimental structures** (~220K): Ground truth for supervised learning. Small by deep learning standards but irreplaceable — no synthetic data can correct errors that are consistent across all prediction models.
- **Functional data** (~10K–50K): Binding affinities, kinetics, thermodynamic measurements. The scarcest and most valuable layer — enables affinity prediction heads (Boltz-2, IsoDDE) but is too sparse for primary training.

This asymmetry — 1,000 $\times$ more sequences than structures, 10 $\times$ more structures than functional measurements — drives every data strategy decision in the field.

---

## 2. Experimental Data: The Protein Data Bank

### 2.1 What the PDB Contains

The Protein Data Bank (PDB) is the sole repository of experimentally determined 3D biomolecular structures. As of 2026, it contains approximately **220,000 structures**, determined by:

| Method | Share | Typical Resolution | Strength |
|---|---|---|---|
| X-ray crystallography | ~85% | 1.5–3.0 Å | High resolution, well-established |
| Cryo-EM | ~12% | 2.0–4.0 Å | Large complexes, membrane proteins |
| NMR | ~3% | N/A (ensemble) | Solution-state dynamics |

By deep learning standards, 220K structures is a remarkably small dataset. ImageNet contains 14M images; the Common Crawl used for LLM training contains trillions of tokens. Protein AI operates in a regime where **every training example matters**.

### 2.2 PDB Biases

The PDB is not a representative sample of protein space. Experimental structure determination is biased toward:

**Crystallization-amenable proteins**: Proteins that readily crystallize (stable, compact, soluble globular domains) are heavily overrepresented. Membrane proteins (~30% of all proteins) and intrinsically disordered proteins (~40% of eukaryotic proteins contain disordered regions) are underrepresented.

**Organisms**: *Homo sapiens*, *Escherichia coli*, and *Saccharomyces cerevisiae* dominate. Structures from underrepresented lineages (e.g., tropical parasites, archaea) are rare — a gap that the AFDB quaternary expansion specifically addresses.

**Size**: Most structures fall in the 100–500 residue range. Very large complexes (>2000 residues) are increasingly accessible through cryo-EM but remain a minority.

**Drug targets**: Proteins of pharmaceutical interest (kinases, GPCRs, proteases) are overrepresented relative to their genomic prevalence.

These biases directly affect trained models: accuracy is highest for domains well-represented in PDB and degrades for underrepresented categories.

### 2.3 Using PDB for Training

All protein AI models use PDB as their primary supervised training data, with several standard preprocessing steps:

**Temporal split**: Structures deposited after a cutoff date (typically September 30, 2021 for AF2/AF3-era models) are reserved for evaluation. This prevents data leakage — ensuring that benchmarks like CASP15/16 and CAMEO assess genuine generalization.

**Sequence clustering**: Sequences are clustered at 40% identity to prevent training on near-duplicates. Sampling is typically performed at the cluster level, ensuring fold-level diversity rather than overrepresenting well-studied protein families.

**Quality filtering**: Structures below a resolution threshold (typically 9 Å) are excluded. Multi-chain structures require additional filtering for stoichiometric consistency and chain completeness.

---

## 3. Sequence Databases: The 1,000 $\times$ Advantage

While the PDB provides structural ground truth, sequence databases provide the raw material for evolutionary signal extraction through MSA construction (Part 1) and PLM pre-training.

### 3.1 The Database Hierarchy

```
Sequence databases used in protein AI:

  UniProt (~250M sequences)
    ├── Swiss-Prot (~570K): manually curated, high quality
    └── TrEMBL (~249M): automatically annotated, large scale
         │
         └── UniRef clustered versions:
              ├── UniRef100: identical sequences
              ├── UniRef90: 90% identity clusters
              ├── UniRef50: 50% identity clusters (ESM-2 training set)
              └── UniRef30: 30% identity clusters (ColabFold MSA search)

  BFD (Big Fantastic Database, ~2.5B sequences)
    └── Includes metagenomic sequences from soil, ocean, gut microbiomes

  MGnify (~600M+ sequences)
    └── Metagenomic protein sequences from environmental samples
```

### 3.2 How Models Use Sequence Databases

**For MSA construction**: AF2/AF3, Boltz, Chai, and SeedFold search sequence databases using JackHMMER, HHblits, or MMseqs2 to build multiple sequence alignments. The choice of search database and parameters directly affects prediction quality — MSA depth and diversity correlate strongly with structure prediction accuracy (Part 1).

**For PLM pre-training**: ESM-2 was trained on UniRef50 (~65M cluster representatives) using masked language modeling. The resulting embeddings capture evolutionary patterns without requiring MSA search at inference time. Chai-1 and NP3 integrate ESM-2 embeddings as an alternative to MSA (Part 1).

**As a distillation source**: SeedFold's largest synthetic data source is MGnify — 23M sequences whose structures were predicted by OpenFold/AF2. These metagenomic sequences provide structural diversity beyond what the PDB covers, particularly for domains from uncultured organisms.

### 3.3 The Sequence-Structure Asymmetry

The fundamental tension in protein AI data: sequences grow exponentially (metagenomic sequencing adds billions per year), but structures for those sequences remain unknown. This creates a natural role for synthetic structure data — using prediction models to fill the structural gap.

---

## 4. Synthetic Structure Data

### 4.1 AlphaFold Database: Democratizing Protein Structure

The AlphaFold Protein Structure Database (AFDB) is the largest collection of predicted protein structures, created by running AlphaFold2 on sequences from UniProt.

**AFDB Monomer (2021–2024)**:

| Version | Year | Scale | Coverage |
|---|---|---|---|
| AFDB v1 | 2021 | ~360K structures | 21 model organisms |
| AFDB v2 | 2022 | ~200M structures | Nearly all of UniProt |
| AFDB v4 | 2024 | 214M sequences | Updated interface, refined coverage |

Quality is assessed using pLDDT (predicted Local Distance Difference Test):
- Very high confidence: pLDDT > 90
- Confident: 70 < pLDDT < 90
- Low confidence: 50 < pLDDT < 70
- Very low confidence: pLDDT < 50

As training data for downstream models, AFDB provides an enormous expansion beyond PDB — but with a crucial caveat: **all structures inherit AlphaFold2's systematic errors**. Regions where AF2 consistently fails (disordered regions, certain membrane proteins, novel folds without MSA support) will be consistently wrong across all 214M predictions.

**AFDB Quaternary Expansion (2026, Han et al.)**:

In 2026, the AFDB underwent its most significant qualitative expansion: from monomeric structures to **proteome-scale protein complex predictions**. This represents a paradigm shift from single-chain coverage to interactome-scale structural modeling.

The expansion involved predicting 3D structures for approximately **31 million dimeric complexes** across 4,777 proteomes:

```
AFDB Quaternary Data Generation Pipeline:

  UniProt 2025_04 sequences (15–1,500 aa)
    │
    ├── Homodimers: 23.4M
    │     → Every monomer sequence duplicated as a complex
    │
    └── Heterodimers: 7.6M
          → STRING physical interaction database
          → 16 model organisms + 30 WHO global health proteomes
    │
    ▼
  MSA generation: MMseqs2-GPU (ColabFold)
    → UniRef30 search, best hit per taxon (orthology filter)
    │
    ▼
  Structure prediction: AlphaFold-Multimer
    → ColabFold or accelerated OpenFold (cuEquivariance + TensorRT)
    → DGX H100 superpod, Slurm array parallelization
    │
    ▼
  Quality filtering:
    → ipSAE_min ≥ 0.6 (interface quality)
    → pLDDT_avg ≥ 70 (overall confidence)
    → backbone clashes ≤ 10 (physical plausibility)
    │
    ▼
  Result: 1.8M high-confidence homodimers (~7% retention)
          57K tentatively high-confidence heterodimers
```

The central quality metric is **ipSAE** (interaction prediction Score from Aligned Errors) — a metric derived from PAE (Predicted Aligned Error) matrices that quantifies interface quality between chains. AlphaFold-Multimer produces PAE matrices predicting the positional error of each residue when the structure is aligned on a different residue. ipSAE converts these inter-chain PAE values into a scalar interface confidence score. Crucially, ipSAE is **directional**: the score for chain A aligned relative to chain B differs from B relative to A. The study defines $\text{ipSAE}\_{min} = \min(\text{ipSAE}\_{A \to B},\; \text{ipSAE}\_{B \to A})$ as a conservative, single-value estimate of interface quality. Among the four candidate metrics evaluated (ipTM, ipSAE, LIS, pDockQ2), $\text{ipSAE}\_{min}$ showed the clearest distributional separation between true homodimers and monomers, with F1 = 0.744 (precision 0.859, recall 0.655) at the $\geq 0.6$ cutoff.

The quality filtering reveals a key asymmetry: only ~7% of homodimer predictions meet the high-confidence threshold, and heterodimer predictions have even lower retention rates. This reflects the fundamental difficulty of interface prediction — modeling how two protein surfaces interact requires capturing subtle coevolutionary signals that are diluted or absent in MSAs.

The high-confidence complexes are classified into three tiers:

| Confidence Tier | $\text{ipSAE}\_{min}$ | Count (homodimers) |
|---|---|---|
| Very high-confidence | $\geq 0.8$ | 972,625 |
| Confident | $0.7$ to $< 0.8$ | 438,879 |
| Low-confidence | $0.6$ to $< 0.7$ | 342,738 |

Several key findings emerged from the analysis:

**Emergent structures**: Some protein folds appear only in the oligomeric context. Proteins with low monomeric confidence ($\text{pLDDT}\_{avg}$ = 50–60) can achieve high confidence as homodimers ($\text{pLDDT}\_{avg}$ = 80–85) through domain swapping — where each chain contributes structural elements that complete the fold of its partner. These emergent structures are invisible to monomeric prediction.

**Compressibility of complex space**: Clustering the 1.8M high-confidence structures yields 224,862 clusters — an 8-fold compression. The top 1% of cluster representatives account for ~25% of all complexes, indicating that predicted complex space is concentrated around a relatively small number of recurrent structural solutions.

**Evolutionary conservation**: Approximately 9% of non-singleton clusters contain members from at least two superkingdoms, suggesting that these oligomeric architectures originated in a common ancestor and have been maintained as universal building blocks of cellular life.

**Taxonomic variation**: Prokaryotic proteomes show 3 $\times$+ higher homodimer prediction success rates compared to eukaryotes. This likely reflects the shorter, more compact architecture of prokaryotic proteins and the higher prevalence of homo-oligomeric assemblies in prokaryotic biology.

**For ML training**: Unlike monomeric AFDB, quaternary predictions contain **interface information** — residue-residue contacts between chains, binding site geometry, and oligomeric assembly patterns. This data is directly relevant for training binder design models (such as Complexa/Teddymer) and PPI prediction models, addressing a key gap where PDB complex data (~tens of thousands) is orders of magnitude smaller than what large-scale training requires.

### 4.2 Knowledge Distillation: Teacher to Student

Knowledge distillation uses a trained teacher model to generate synthetic structures that augment the student model's training data. This strategy has become essential in protein AI, where the PDB alone is too small for modern transformer architectures.

Each major model has adopted a distinct distillation strategy:

**AF2 Self-distillation**: AlphaFold2 predictions on UniRef sequences were filtered by confidence and added to training data. The AFDB itself is the product of this approach — and when subsequent models use AFDB structures for training, they are performing indirect AF2 distillation.

**OpenFold3 — AF3 as Teacher (13M structures)**: OpenFold3 used AlphaFold3 to predict structures for UniProt sequences, yielding 13M distilled structures. After high-confidence filtering, these were combined with 300K PDB experimental structures for training. The distilled data constitutes ~97% of the total training set by count — making OpenFold3 overwhelmingly trained on synthetic data. The risk: AF3's systematic biases, particularly around small molecules and RNA, propagate directly to the student.

**Boltz-2 — Multi-Source Distillation**: Boltz-2 employs a carefully balanced data mixture:

| Source | Ratio | Content |
|---|---|---|
| PDB experimental | 60% | Ground truth structures |
| AFDB distillation (AF2) | 30% | Monomer structures, ~5M proteins |
| Boltz-1 distillation | 10% | Protein-ligand, RNA, DNA complexes |

This is the most explicit multi-source distillation strategy in the field. Boltz-2's ablation studies demonstrate that distillation particularly improves performance on targets where PDB data is sparse — novel folds, unusual ligand types — confirming that synthetic data's primary value is in **filling coverage gaps**, not merely adding volume.

**Proteina — ESMFold Structures for Scaling**: NVIDIA's Proteina used ESMFold-generated structures as synthetic training data. ESMFold is faster but less accurate than AF2/AF3, representing a quality-speed tradeoff. The key contribution: Proteina's scaling experiments demonstrated that increasing synthetic data volume produces **log-linear improvement** in structural quality metrics — the first evidence that data scaling laws apply to protein structure generation.

**SeedFold — The Largest Distillation at 26.5M Structures**: SeedFold (ByteDance, 2025) employed the largest distillation dataset in the field: 26.5M structures, a 147 $\times$ expansion over PDB alone. The teacher model was OpenFold running AlphaFold2 weights — notably using AF2 rather than AF3 as the teacher.

| Data Source | Samples | Sampling Weight | Filtering |
|---|---|---|---|
| PDB experimental | 180K | 0.50 | Cutoff 2021-09-30 |
| AFDB (short monomers) | 3.3M | 0.08 | <200 residues, pLDDT $\geq$ 0.8, 50% seq. identity clustering |
| MGnify (longer sequences) | 23M | 0.42 | Median 435 residues, 30% seq. identity clustering |

SeedFold's most important contribution to the data discussion is an ablation that answers a question other models left open: **is distillation data needed throughout training, or only for pre-training?**

The answer is unambiguous: distillation must be maintained continuously. When SeedFold removed distillation data from training at step 47,612, intra-protein structure prediction accuracy **degraded immediately**. The authors provide a compelling theoretical explanation: the architectural transition from AF2 to AF3 replaced the Invariant Point Attention (IPA) structure module — which encodes strong geometric inductive biases — with a general-purpose diffusion module. This transformer-based denoiser lacks geometric priors and therefore requires substantially more data to learn the same structural regularities. The 180K PDB structures are insufficient; the 26.5M distillation set compensates for the lost inductive bias with data volume.

### 4.3 Synthetic Complex Data

Beyond distilling single-chain structures, several models generate synthetic **complex** data to address the scarcity of protein-protein and protein-ligand complex structures in the PDB.

**Teddymer** (Complexa/NVIDIA): A large-scale dataset of synthetic binder-target pairs, purpose-built for training binder design models. Teddymer addresses the fundamental data bottleneck for protein design: the PDB contains only ~tens of thousands of well-characterized protein-protein complexes, far too few for training generative models that must explore the vast space of possible binding interfaces.

**Complexa training data**: Combines PDB experimental structures, Teddymer synthetic binder-target pairs, and AFDB structures — mixing sources to achieve both coverage and quality.

### 4.4 The Quality-Quantity Tradeoff

All distillation strategies face a fundamental tension:

```
More synthetic data → better coverage of protein space
                   → but: teacher errors propagated to student
                   → quality filtering reduces effective data size

Concrete example:
  OpenFold3: 13M AF3-distilled structures
    → AF3's RNA structure errors → propagated to OpenFold3
    → Quality filtering (high-confidence only) → effective size << 13M

  AFDB quaternary: 31M dimer predictions
    → Only 7% meet high-confidence threshold → 1.8M usable
    → Heterodimer calibration biased toward homodimer-like complexes
```

The field has converged on a pragmatic approach: **use distillation broadly but maintain a substantial proportion of experimental data in the training mix**. Boltz-2's 60/40 experimental-to-synthetic ratio and SeedFold's 50/50 sampling weight reflect this balance.

---

## 5. Comparison: Training Data Strategies Across Models

| | AF2/AF3 | Boltz-2 | SeedFold | Proteina | OpenFold3 | NP3 |
|---|---|---|---|---|---|---|
| **Primary data** | PDB | PDB (60%) | PDB (50%) | PDB | PDB | PDB |
| **Teacher model** | AF2 (self) | AF2 + Boltz-1 | OpenFold/AF2 | ESMFold | AF3 | AF2 |
| **Distillation source** | UniRef | AFDB + Boltz-1 | AFDB + MGnify | UniRef (ESMFold) | UniProt (AF3) | AFDB |
| **Distillation scale** | AFDB (200M+) | ~millions | 26.5M | Millions | 13M | ~200M |
| **Distill usage** | Indirect | Mixed training | Continuous (50:50) | Scaling experiments | Primary training | Pre-training |
| **Synthetic complexes** | — | Optional | — | Teddymer | — | — |
| **Sequence DB for MSA** | UniRef, BFD | UniRef | UniRef30, colabfold_envdb | — | UniRef | UniRef |
| **MSA tool** | JackHMMER | MMseqs2 | ColabFold (MMseqs2-GPU) | — | MMseqs2 | MMseqs2 |
| **PLM integration** | — | — | — | — | — | ESM-2 |
| **Functional data** | — | PDBbind/BindingDB | — | — | — | — |
| **Key data insight** | Created AFDB ecosystem | Multi-source mixing ratios | Continuous distillation required | Scaling law on data | Teacher bias propagation | AFDB pre-training |

### Key Observations

**No consensus on teacher model**: The field has not converged on which model should generate distillation data. AF2 (SeedFold, Boltz-2), AF3 (OpenFold3), ESMFold (Proteina), and Boltz-1 (Boltz-2 for complexes) are all used as teachers. The choice reflects availability, speed, and the type of structures needed (monomers vs. complexes).

**Experimental data proportion matters**: Despite the availability of tens of millions of synthetic structures, no successful model trains exclusively on synthetic data. The experimental-to-synthetic ratio ranges from 3% (OpenFold3) to 60% (Boltz-2), but PDB data is always present and typically upsampled relative to its raw count.

**Distillation is not optional**: SeedFold's ablation provides the strongest evidence to date: removing distillation data mid-training causes immediate degradation. This suggests that the transformer-based architectures used by post-AF2 models fundamentally require more data than the PDB provides.

**The metagenomic frontier**: SeedFold's use of MGnify sequences (23M, 87% of distillation data) is unique. Metagenomic sequences provide structural diversity from uncultured organisms — domains, folds, and functional motifs absent from the well-studied proteomes that dominate the PDB.

---

## 6. AFDB Quaternary Expansion: Implications for Model Training

The 2026 AFDB quaternary expansion (Han et al.) represents a qualitative shift in the available synthetic data — from monomeric structures to protein complexes. This has direct implications for downstream model training.

### 6.1 New Training Data for Complex Prediction

Prior to this expansion, models that needed complex structures for training relied primarily on:
- PDB experimental multimers (~tens of thousands)
- Self-generated distillation (Boltz-2's Boltz-1 complex distillation)
- Purpose-built synthetic datasets (Teddymer)

The AFDB now provides **1.8M high-confidence homodimeric structures** with standardized quality metrics. Unlike self-generated distillation, this is a publicly available, quality-calibrated resource that any model can incorporate.

### 6.2 Interface Information

The critical difference from monomeric AFDB: quaternary predictions contain **interface geometry** — the specific residue-residue contacts, buried surface areas, and binding modes that govern protein-protein interactions. This information is directly relevant for:
- Binder design models (Complexa, BoltzGen, RFdiffusion)
- PPI prediction and scoring
- Protein engineering targeting oligomeric assemblies

### 6.3 Limitations and Caveats

**Homodimer bias**: The high-confidence set is overwhelmingly homodimeric. The heterodimer confidence calibration is still under development, and current filtering criteria favor homodimer-like complexes — those with high sequence identity and small chain-length differences between partners.

**Low retention rate**: Only ~7% of homodimer predictions meet the high-confidence threshold. This means 93% of predictions are too unreliable for direct use. Using lower-confidence predictions as training data requires careful weighting or confidence-aware training strategies.

**Model limitations**: All quaternary structures were predicted using AlphaFold-Multimer, which has known limitations for heteromeric complexes and cannot model higher-order assemblies (trimers, tetramers). Future re-predictions using AF3 or Boltz-2 could improve both quality and scope.

---

## 7. The Data Ecosystem: Three Emerging Trends

### 7.1 The Synthetic Data Quality Competition

A **data flywheel** is emerging in protein AI:

```
Better teacher model
       │
       ▼
Higher-quality synthetic structures
       │
       ▼
Better student model
       │
       ▼
Student becomes the next teacher
       │
       ▼
  (repeat)
```

AF2 generated AFDB → models trained on AFDB (NP3, Boltz-2) → these models generate better predictions → next generation trains on those predictions. This flywheel drives continuous improvement, but carries a risk: **model collapse**. If successive generations of synthetic data are all derived from the same lineage of models, systematic errors may be reinforced rather than corrected. The role of experimental PDB data becomes increasingly important as the "ground truth anchor" that prevents the flywheel from drifting.

### 7.2 Experimental Data as Strategic Asset

As synthetic data becomes abundant, the strategic value of **experimental data** increases rather than decreases.

**Cryo-EM revolution**: The rapid growth of cryo-EM is producing structures of large complexes, membrane proteins, and dynamic assemblies that were previously inaccessible. These structures fill gaps in the PDB that synthetic data cannot address.

**Functional measurements**: Binding affinity ($K\_d$, $K\_i$, $\text{IC}\_{50}$), thermodynamic parameters, and kinetic rates represent the scarcest data tier. Boltz-2's affinity head was trained on PDBbind and BindingDB — databases of only ~20K–50K experimental measurements. IsoDDE's competitive advantage likely stems from access to Isomorphic Labs' proprietary binding data at scale. Functional data is the next bottleneck: it cannot be synthesized by existing prediction models.

**Irreplaceable corrections**: Experimental structures provide the only ground truth for correcting systematic prediction errors. A region where all prediction models consistently fail (e.g., certain metal coordination geometries, non-canonical base pairing in RNA) can only be corrected by experimental structures in the training set.

### 7.3 Data Access and the Competitive Landscape

The availability and licensing of training data is becoming a key competitive dimension:

| Data Source | License | Who Benefits |
|---|---|---|
| PDB | CC0 (public domain) | Everyone equally |
| AFDB (monomer + quaternary) | CC-BY-4.0 | Everyone equally |
| UniProt / UniRef | CC-BY-4.0 | Everyone equally |
| Boltz-2 distillation | MIT (model); data pipeline reproducible | Open-source community |
| OpenFold3 distillation | Apache 2.0 | Open-source community |
| Teddymer | Public (with model release) | Open-source community |
| IsoDDE training data | Proprietary | Isomorphic Labs only |
| Pharmaceutical binding data | Proprietary | Individual companies |

The **data moat** hypothesis: IsoDDE's performance gap over open-source models may be primarily attributable to proprietary experimental data — particularly large-scale binding affinity measurements from internal drug discovery programs. If true, the open-source community cannot close this gap through architecture improvements alone. The response strategies include:
- Generating synthetic binding data at scale (computationally expensive, accuracy uncertain)
- Community efforts to aggregate and share experimental binding data
- Expanding public databases like BindingDB and ChEMBL

---

## 8. Convergence and Outlook

### What has become standard

- **PDB as ground truth**: Universal and irreplaceable, despite its biases and small size
- **Distillation as essential supplement**: Every competitive model uses synthetic structures; PDB alone is insufficient for modern transformer architectures
- **Quality filtering**: All distillation pipelines apply confidence thresholds (pLDDT, ipSAE) to filter synthetic data
- **Multi-source data mixing**: Combining experimental and synthetic data with explicit sampling ratios, rather than naive concatenation

### What remains open

**Optimal experimental-to-synthetic ratio**: Boltz-2 uses 60:40, SeedFold uses 50:50, OpenFold3 uses ~3:97. No systematic study has compared these ratios across architectures to determine whether there is an optimal balance.

**Teacher model selection**: AF2, AF3, ESMFold, and Boltz-1 are all used as distillation teachers. Whether teacher quality matters more than teacher diversity — and whether using multiple teachers produces better students — remains unexplored.

**Functional data scaling**: As models add affinity prediction capabilities, the bottleneck shifts from structural to functional data. How to scale functional training data (synthetic binding data? transfer from related tasks? active learning?) is an emerging challenge.

**AFDB quaternary data utilization**: The 1.8M high-confidence dimeric complexes from the 2026 expansion are available but have not yet been incorporated into any published model's training pipeline. Their impact on complex prediction and binder design accuracy remains to be measured.

**The model collapse question**: As synthetic data dominates training sets (97% for OpenFold3, 50% for SeedFold), how many generations of synthetic-data-trained models can improve before systematic errors accumulate? The field lacks both theoretical analysis and empirical measurement of this risk.

### The Core Lesson

In the first era of protein AI (AF1–AF2), the breakthrough was architectural — attention mechanisms, invariant point attention, end-to-end differentiable structure prediction. In the current era, architectures have largely converged (Part 2, Part 7). The competitive frontier has shifted to data: who has the best experimental data (IsoDDE), who generates the best synthetic data (AFDB, Teddymer), and who mixes them most effectively (Boltz-2, SeedFold). The data landscape is not a static resource to be consumed but a dynamic ecosystem where each model's predictions become the next model's training data — a recursive loop whose limits we are only beginning to understand.

---

**Next: Part 9 — Where Is the Field Heading? Convergence, Gaps, and the Road Ahead**

We conclude the series with a panoramic view: what technical choices have converged across all eight prior Parts, what meta-patterns recur throughout the series, where the wet-lab validation gap stands, how benchmarks are evolving, where these models fit in drug discovery, and what 2026–2027 may bring.
