---
title: "Impact of AI on Scientific Knowledge Production — Part 3: Empirical Evidence from AlphaFold and Open Questions"
date: 2026-03-22 11:00:00 +0900
categories: [AI & ML, Social Impact of AI]
tags: [alphafold, ai-in-science, knowledge-production, empirical-evidence, augmentation]
math: true
---

*This is Part 3 of a three-part series examining how AI affects scientific knowledge production.*

- **Part 1:** [Three Theoretical Frameworks](/posts/impact-of-ai-on-science-part1-three-theoretical-frameworks/)
- **Part 2:** [Beyond the Productivity Debate — How AI Distorts Research Direction](/posts/impact-of-ai-on-science-part2-beyond-productivity-debate/)
- **Part 3** (this post): Empirical Evidence from AlphaFold and Open Questions

---

## Introduction

Parts 1 and 2 of this series reviewed theoretical frameworks for understanding AI's impact on scientific knowledge production. Agrawal et al. model AI as a bottleneck-constrained augmentation tool; Acemoglu et al. warn of knowledge collapse through substitution of human learning effort; Hong et al. show that augmentation and convergence can coexist at different levels of analysis; and Gans identifies the mechanism by which AI can distort research direction — the "Work to the AI" phenomenon.

These frameworks make specific, testable predictions. In this final post, I examine the most thoroughly studied empirical case of AI in science — AlphaFold 2 — through the lens of each framework. The goal is to assess which theoretical predictions hold, which do not, and what questions remain open.

---

## Empirical Analysis: AlphaFold 2's Scientific Impact (IGL, 2025)

**Source:** David Ampudia Vicente and George Richardson, "AI in Science: Emerging Evidence from AlphaFold 2," Innovation Growth Lab (IGL) Full Literature Review, November 2025.

### Methodology

The IGL study is a large-scale bibliometric analysis of AlphaFold 2's impact on scientific research, covering 2018–2025. The methodological design is rigorous:

- **Data sources:** OpenAlex (publications), PDB (protein structures), The Lens (patents), PubMed iCite (clinical citations), Semantic Scholar (citation intent)
- **Analysis units:** Individual papers, established researchers, and research labs (PI-level)
- **Statistical approach:** Coarsened Exact Matching (CEM) + Difference-in-Differences + Poisson regression
- **Control groups:** Three carefully constructed comparators — AI frontier papers, non-AI protein prediction methods, and other structural biology frontier methods (selected based on high-citation papers)

The use of three distinct control groups is a particular strength, allowing the study to distinguish AlphaFold 2's effects from general trends in AI adoption, protein research, and structural biology.

### Finding 1: Scientific Reach

AlphaFold 2's diffusion through the scientific literature has been extraordinary:

- Approximately **681,000 papers** cite AlphaFold 2 directly or indirectly
- Approximately **269,900 papers** are methodologically strongly connected
- Approximately **1,957,000 unique scientists** are associated with these papers (778,000 for methodological users)
- Direct citations concentrate in biochemistry; indirect citations spread into medicine and applied domains

### Finding 2: Experimental Structural Biology

This is the most consequential finding for the theoretical debate:

- AlphaFold 2 users submitted **45–49% more structures** to the Protein Data Bank (PDB) than matched controls
- Users explored structurally **more novel proteins** (lower TM Score similarity to existing structures)
- Mapping of **uncharacterized proteins** increased
- Research on **rare organisms** increased

The critical interpretation: **AI did not replace experimental work — it complemented it.** Researchers used AlphaFold 2's computational predictions to guide and prioritize their experimental efforts, resulting in *more* experiments on *more diverse* targets, not fewer experiments on familiar ones.

### Finding 3: Academic Productivity

Productivity effects were positive and statistically significant across units of analysis:

| Unit | Publication Increase | Citation Increase |
|------|---------------------|-------------------|
| Individual researcher | +2.5% | +8.1% |
| Research lab | +5.1% | +10.4% |
| Lab (methodological use) | +11.5% | — |

Notably, experienced researchers and larger labs benefited more — suggesting that absorptive capacity matters. AI augmentation is not uniformly distributed; it favors those with the domain expertise to leverage AI outputs effectively.

### Finding 4: Applied Research and Innovation

- **Disease research:** Individual paper-level effects were not significant, but at the researcher level, the probability of engaging in disease-related research increased by **+9.3%**.
- **Clinical citations:** Paper-level probability of clinical citation **doubled** — twice the rate of other AI or non-AI methods.
- **Patent citations:** +36.8% at the paper level, +34.2% at the lab level, +22.6% at the researcher level. Patent quality (measured by subsequent patent citations) also showed positive association.

These results suggest that AlphaFold 2 is not only accelerating basic research but also facilitating translational connections — though the clinical translation pipeline remains in its early stages.

---

## Cross-Examination: Interpreting AlphaFold Through Each Framework

### Through the Agrawal Model

Agrawal et al.'s bottleneck framework predicts that AI will dramatically improve specific stages of the scientific process while leaving others unchanged — and that the overall impact will be constrained by the weakest link.

AlphaFold 2 fits this prediction precisely:

- **$\gamma$ (Design Generation) was dramatically enhanced.** AlphaFold 2 provides structure predictions for over 200 million proteins, fundamentally changing how researchers design experiments. The +45–49% increase in experimental structure submissions is a direct consequence.
- **$\delta$ (Testing) remains the bottleneck.** The IGL data shows that while paper-level clinical citations doubled, the researcher- and lab-level translational effects are not yet statistically significant. The pipeline from structural prediction to clinical application — through target validation, lead optimization, preclinical testing, and clinical trials — remains largely unaccelerated.

The multiplicative structure $\omega = \alpha \cdot \beta \cdot \gamma \cdot \delta$ predicts exactly this pattern: even a large increase in $\gamma$ produces a modest increase in $\omega$ when $\delta$ remains small. The empirical evidence confirms the bottleneck hypothesis.

### Through the Acemoglu Model

Acemoglu et al.'s knowledge collapse model predicts that AI will substitute for human effort, reducing learning incentives and eroding collective general knowledge. AlphaFold 2 appears to be a **counterexample** — but understanding why requires careful analysis.

**Why AlphaFold 2 resists knowledge collapse:**

1. **Complementarity, not substitution.** AlphaFold 2 users increased their experimental activity by 45–49%, rather than reducing it. The computational predictions served as guides for experiments, not replacements for them.
2. **Public knowledge expansion.** The AlphaFold Database provides over 200 million predicted structures as an open-access public good. In Acemoglu's terms, this increases the **knowledge aggregation capacity** ($I$) — the parameter that unconditionally improves welfare and increases resilience against knowledge collapse.
3. **Exploration diversification.** Users explored more novel, uncharacterized proteins — the opposite of the knowledge convergence that Acemoglu et al. predict.

**Why AlphaFold 2 is not a complete counterexample:**

The complementarity arises in part from **domain-specific structural features** that may not generalize:

- Structure prediction is an *upstream* step that precedes experimental validation. The prediction does not eliminate the need for the experiment — it redirects and prioritizes it. In domains where AI output is the *final product* rather than an input to further work, substitution effects may dominate.
- AlphaFold 2 provides a **calibrated uncertainty signal** (pLDDT scores) that enables researchers to assess prediction reliability. AI tools without such calibration — particularly LLMs, which can "hallucinate confidently" — may be more prone to triggering substitution effects.
- Evidence from other domains is less encouraging: Stack Overflow activity declined after ChatGPT, Wikipedia contribution decreased in ChatGPT-substitutable areas, and studies report cognitive and creative decrements from AI writing assistance.

### Through the Hong Framework

Hong et al.'s micro-macro separation framework asks: does the individual-level augmentation observed in AlphaFold translate to collective-level diversification, or does it mask collective-level convergence?

The IGL data provides partial answers:

- **Individual-level augmentation is confirmed:** Each researcher explores more, publishes more, and targets more novel proteins. This is unambiguous within-individual diversification.
- **Collective-level diversification is suggested but not proven.** The TM Score analysis shows that AlphaFold users target structurally more novel proteins — but we need to ask whether different labs are targeting *different* novel proteins (between-variance increase) or all converging on the *same* novel proteins that AlphaFold predicts well (between-variance decrease masked by within-variance increase).

The distinction between AlphaFold (a database/infrastructure) and LLM-based tools (recommendation/dialogue systems) is important here. AlphaFold presents users with a vast structural landscape and lets them navigate it independently. LLM-based tools provide specific recommendations that may channel all users toward similar "optimal" directions. The structural difference in how these tools mediate between user and knowledge space likely produces different collective-level dynamics.

### Through the Gans Model

Gans's three-regime model makes the most specific testable prediction: is AlphaFold operating in the **Truncate** regime (streetlight effect — research converges toward AI's capability range) or the **Enlarge** regime (true exploration liberation)?

The IGL evidence leans toward **Enlarge**: researchers explored more novel proteins, including uncharacterized ones and those from rare organisms. This is consistent with AlphaFold's capability range ($R_D$) being sufficiently broad that the Enlarge condition ($R_D \geq d^*_E$) is met.

However, a critical question remains unanswered: **is the observed novelty within or beyond AlphaFold's confident prediction range?**

- If the "novel" proteins that researchers are now studying are all in the **high-pLDDT** (high-confidence prediction) range, then what appears to be Enlarge may actually be Truncate at a larger scale — a broader streetlight, but still a streetlight.
- If researchers are also pursuing proteins in the **low-pLDDT** range (intrinsically disordered proteins, membrane proteins, multi-state conformational ensembles) — where AlphaFold's predictions are less reliable — then the Enlarge interpretation is supported.

Distinguishing these scenarios requires a **pLDDT-stratified analysis** of the IGL results: decomposing the observed diversification by AlphaFold's own confidence scores. This analysis has not yet been performed and represents an important open research question.

### AlphaFold vs. LLM: Structural Differences in AI Type

A recurring theme across all four framework interpretations is that **AlphaFold's characteristics are structurally different from those of LLM-based AI tools**, and these differences matter for the substitution-complementarity dynamics:

| Dimension | AlphaFold | LLM-Based Tools |
|-----------|-----------|-----------------|
| **Nature** | Open database/infrastructure | Recommendation/dialogue agent |
| **Uncertainty quantification** | pLDDT provides calibrated confidence | Hallucination delivered with confidence; no calibrated uncertainty |
| **Independent verification** | Experimental structure determination (X-ray, cryo-EM) | Verification depends on human expertise, which AI may simultaneously erode |
| **User-knowledge mediation** | User navigates a landscape independently | Tool provides specific recommendations that channel behavior |
| **Domain knowledge structure** | Physical ground truth exists (actual protein structures) | Many application domains lack clear ground truth |

This comparison suggests that the positive findings from AlphaFold **cannot be straightforwardly generalized to LLM-based AI tools**. The conditions that made AlphaFold a complement rather than a substitute — calibrated uncertainty, independent verification pathways, and structured domain knowledge — are precisely the conditions that many LLM-based tools lack.

---

## Limitations and Open Questions

### Limitations of the IGL Study

- **Short observation window:** AlphaFold 2 was released in 2021, giving approximately 3.5 years of observation. The medium-term convergence and long-term collapse dynamics predicted by Acemoglu et al. and Hong et al. may not yet be visible.
- **Selection bias:** Despite CEM matching, researchers who adopted AlphaFold 2 may be inherently more innovative, productive, or resource-rich. The observed effects may partially reflect pre-existing differences rather than causal impact.
- **Citation intent noise:** Classification of methodological vs. incidental citations involves measurement error.
- **Correlation, not causation:** The DID design strengthens causal inference but does not eliminate confounders.

### Open Questions

1. **pLDDT-stratified analysis:** Does AlphaFold's positive impact concentrate in high-confidence prediction regions (Truncate/streetlight) or extend across the full confidence spectrum including low-confidence regions (Enlarge/liberation)?

2. **Medium-term convergence:** Will the diversification observed in the first 3.5 years persist, or will research gradually converge toward AlphaFold's capability frontier as newer versions (AlphaFold 3) expand coverage?

3. **Generalizability across AI types:** Can the conditions that made AlphaFold a complement be deliberately engineered into other AI tools? Or are they domain-specific features that cannot be transplanted?

4. **Between-variance analysis:** Is the observed novelty increase a between-lab diversification (different labs exploring different novel directions) or a within-lab diversification that masks between-lab convergence?

---

## Conclusion: Key Design Challenges for AI for Science

Across this three-part series, I have reviewed theoretical frameworks and empirical evidence on how AI affects scientific knowledge production. The picture that emerges is more nuanced than either "AI accelerates science" or "AI threatens knowledge" — both are true, at different levels of analysis and over different time horizons.

From this review, three design challenges for AI for Science emerge:

**1. Calibrated honesty in uncertainty reporting.** AI tools that report their prediction confidence honestly — as AlphaFold does with pLDDT scores — enable researchers to selectively adopt AI outputs and maintain independent judgment. Tools that deliver outputs with false confidence (as LLMs do when hallucinating) undermine the human capacity for selective adoption and accelerate the substitution dynamics that lead to knowledge collapse.

**2. Preservation of AI-independent verification pathways.** AlphaFold's complementarity with experimental structural biology is sustained because an independent verification pathway (X-ray crystallography, cryo-EM) exists and remains actively used. When AI tools operate in domains where independent verification is costly, slow, or dependent on the same human expertise that AI is eroding, the substitution-collapse dynamic is more likely to dominate.

**3. Promotion of exploration beyond AI's capability frontier.** The "Work to the AI" phenomenon and its consequence — dynamic stagnation at the capability frontier — represent a subtle but systemic risk. AI tools should be designed not merely to assist research within their current capability range, but to actively encourage exploration beyond it. Incentive structures, funding mechanisms, and tool design should all be evaluated against this criterion.

These are not abstract principles — they are conditions that can be assessed empirically by examining how different AI tools, deployed in different scientific domains, affect the direction and diversity of research. Whether AlphaFold meets all three conditions, and whether these conditions can be deliberately engineered into future AI for Science systems, are questions I intend to explore in subsequent posts.

---

## References

- Ampudia Vicente, D. & Richardson, G. (2025). [AI in Science: Emerging Evidence from AlphaFold 2.](https://www.innovationgrowthlab.org/resources/ai-in-science-alphafold-2) *Innovation Growth Lab (IGL) Full Literature Review*.
- Agrawal, A. K., McHale, J., & Oettl, A. (2026). [AI in Science.](https://www.nber.org/papers/w34953) *NBER Working Paper 34953*.
- Acemoglu, D., Kong, D., & Ozdaglar, A. (2026). [AI, Human Cognition and Knowledge Collapse.](https://economics.mit.edu/sites/default/files/2026-02/AI,%20Human%20Cognition%20and%20Knowledge%20Collapse%2002-20-26.pdf) *NBER Working Paper 34910*.
- Gans, J. S. (2025). [A Quest for AI Knowledge.](https://www.nber.org/papers/w33566) *NBER Working Paper 33566*.
- Hong, J., Yoon, S., Park, S., & Han, S. P. [How User Adoption of ChatGPT Influences Commercial Search Patterns.](https://sites.google.com/view/juwonhong/research) Under revision at *Management Science*.
- Hao, Y., et al. (2025). [Artificial Intelligence Tools Expand Scientists' Impact but Contract Science's Focus.](https://www.nature.com/articles/s41586-025-09922-y) *Nature*.
- Jumper, J., et al. (2021). [Highly accurate protein structure prediction with AlphaFold.](https://www.nature.com/articles/s41586-021-03819-2) *Nature*, 596, 583–589.
- del Rio-Chanona, R. M., et al. (2024). [Large Language Models Reduce Public Knowledge Sharing on Online Q&A Platforms.](https://arxiv.org/abs/2307.07367) *PNAS Nexus*, 3, pgae400.
- Brynjolfsson, E., Li, D., & Raymond, L. R. (2025). [Generative AI at Work.](https://www.nber.org/papers/w31161) *The Quarterly Journal of Economics*, 140(2), 889–938.
