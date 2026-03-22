---
title: "Impact of AI on Scientific Knowledge Production — Part 1: Three Theoretical Frameworks"
date: 2026-03-22 09:00:00 +0900
categories: [AI & ML, Social Impact of AI]
tags: [knowledge-collapse, ai-in-science, knowledge-production, augmentation, variance-convergence]
math: true
---

*This is Part 1 of a three-part series examining how AI affects scientific knowledge production. The series reviews recent theoretical and empirical work to identify the opportunities, risks, and open questions that should guide the design of AI for Science.*

- **Part 1** (this post): Three theoretical frameworks — augmentation, knowledge collapse, and the micro-macro separation
- **Part 2:** [Beyond the Productivity Debate — How AI Distorts Research Direction](/posts/impact-of-ai-on-science-part2-beyond-productivity-debate/)
- **Part 3:** [Empirical Evidence from AlphaFold and Open Questions](/posts/impact-of-ai-on-science-part3-empirical-evidence-alphafold/)

---

## Introduction

AI is increasingly embedded in the practice of science. AlphaFold predicts protein structures, GNoME discovers new materials, and LLM-based agents assist with literature review, experimental design, and code generation. The question is no longer whether AI will transform scientific research, but *how* — and whether the transformation will be uniformly beneficial.

The answer depends on which lens you use. Some frameworks emphasize AI's potential to augment scientific productivity; others warn that AI dependence could erode the collective knowledge base over time. These perspectives are not merely different opinions — they formalize different mechanisms and make different predictions.

To chart a direction for AI for Science, we need to understand these mechanisms clearly. This post reviews three theoretical frameworks that, taken together, provide a more complete picture than any single one:

1. **Agrawal, McHale, and Oettl (2026)** model AI as a prediction machine that augments the scientific production function — but is subject to bottleneck effects.
2. **Acemoglu, Kong, and Ozdaglar (2026)** model the dynamic risk of *knowledge collapse* — where AI dependency erodes human learning incentives and destroys collective knowledge over time.
3. **Hong et al.** provide empirical evidence that both augmentation and convergence can occur simultaneously — the difference lies in the level of analysis (individual vs. collective).

---

## Framework 1: AI as a Prediction Machine (Agrawal, McHale, and Oettl, 2026)

**Source:** NBER Working Paper 34953, "AI in Science" (March 2026)

### The Four-Stage Model of Science

Agrawal et al. define AI as a **prediction machine** that operates over combinatorial search spaces. They decompose the scientific process into four stages, each with its own productivity parameter:

| Stage | Parameter | Description |
|-------|-----------|-------------|
| Question Generation | $\alpha$ | Identifying productive research questions |
| Idea Generation | $\beta$ | Generating hypotheses and candidate solutions |
| Design Generation | $\gamma$ | Designing experiments, models, or computational pipelines |
| Testing | $\delta$ | Validating hypotheses through experiments or trials |

The knowledge production function takes the form:

$$\dot{A} = \omega \cdot A \cdot S$$

where $A$ is the existing knowledge stock, $S$ is the number of scientists, and $\omega$ is the total factor productivity of science. The key insight is that $\omega$ is the **product** of the four stage-specific productivities:

$$\omega = \alpha \cdot \beta \cdot \gamma \cdot \delta$$

### The Bottleneck Effect

Because $\omega$ is a product rather than a sum, the overall productivity of science is constrained by its weakest stage. If any one parameter is low, the entire product collapses — regardless of how high the others are.

This has a direct practical implication. AlphaFold, for example, dramatically increased $\gamma$ (design generation) for structural biology. But if $\delta$ (experimental testing, clinical trials) remains slow, the overall acceleration of knowledge production is limited. The IGL study of AlphaFold's impact, which we examine in Part 3, confirms exactly this pattern: strong effects on research productivity, but translational impact still in its early stages.

### Augmentation, Not Automation

Agrawal et al. argue that AI's primary role in science is **augmentation** of human scientists, not replacement. Human judgment remains essential for:

- **Abductive inference** — forming causal hypotheses from incomplete data
- **Contextual nuance** — interpreting results within domain-specific knowledge
- **Ethical judgment** — navigating the normative dimensions of research
- **Novel question generation** — identifying genuinely new directions

AI is most effective in data-rich, interpolative tasks — exactly the tasks where large training datasets exist and the goal is pattern recognition within known distributions. For genuinely creative discovery, which often requires extrapolation beyond known distributions, human judgment remains critical.

### The Jagged Frontier

Not all scientific domains benefit equally from AI. Agrawal et al. adopt Ethan Mollick's concept of the **jagged frontier** to describe this unevenness:

| Domain | AI Impact | Characteristics |
|--------|-----------|-----------------|
| Biology | Very high | Data-rich, AlphaFold and drug discovery already transformative |
| Materials Science | High | GNoME accelerating new material discovery |
| Physics | Limited | Anomalies are rare; AI's interpolative strength is less useful |
| Economics | Early stage | LLM-based simulation and data pattern exploration emerging |

This unevenness matters for policy and investment: the domains where AI can contribute most are those with abundant data, well-defined search spaces, and tasks that are primarily interpolative. Domains requiring fundamental conceptual breakthroughs may benefit less.

### AI as a General Purpose Meta-Technology

Finally, Agrawal et al. characterize AI as a **General Purpose Meta-Technology (GPMT)** — a technology for inventing new technologies. This framing implies that AI's impact will compound over time, but realizing this potential requires complementary investments in both **upstream** capabilities (AI models themselves) and **downstream** capabilities (scientists' ability to use AI effectively, workflow redesign, institutional adaptation).

---

## Framework 2: Knowledge Collapse (Acemoglu, Kong, and Ozdaglar, 2026)

**Source:** NBER Working Paper 34910, "AI, Human Cognition and Knowledge Collapse" (February 2026)

While Agrawal et al. focus on AI's augmentation potential, Acemoglu et al. examine a fundamentally different mechanism: the possibility that AI dependency could **destroy** the collective knowledge base over time.

### Two Types of Knowledge

The model distinguishes two types of knowledge that are **complements** in good decision-making:

- **General knowledge** ($\theta_t$): Community-level shared knowledge that evolves over time. Examples include understanding of financial instruments, knowledge of disease mechanisms, or established scientific principles.
- **Context-specific knowledge** ($\theta_{i,t}$): Knowledge unique to an individual's situation — a patient's specific symptoms, a researcher's particular experimental conditions, an investor's risk tolerance. This is drawn independently each period.

### Economies of Scope in Learning

A crucial assumption drives the model's results: human learning effort exhibits **economies of scope**. When an individual invests effort to acquire context-specific knowledge (private signals), they simultaneously produce **public signals** that contribute to the community's general knowledge stock. This side effect is an **externality** — the individual does not internalize the social benefit of their contribution to general knowledge.

### The Substitution Mechanism

Agentic AI enters the model as a provider of context-specific recommendations. When AI provides accurate context-specific guidance, it **substitutes** for human effort — reducing the individual's incentive to learn. This is the source of the dynamic tension:

- **Static effect (short-term):** AI improves individual decision quality by providing better context-specific recommendations.
- **Dynamic effect (long-term):** Reduced human effort means fewer public signals are generated, slowing the accumulation of general knowledge. Over time, the community's general knowledge base erodes.

### Knowledge Collapse

The model identifies conditions under which the economy converges to a **knowledge-collapse steady state** — a situation where general knowledge has completely vanished:

- Multiple steady states can coexist: a high-knowledge state and a knowledge-collapse trap.
- As AI accuracy increases, the **basin of attraction** of the collapse trap expands.
- Beyond a critical accuracy threshold ($\tau_A^c$), the high-knowledge state disappears entirely, and complete collapse becomes the only outcome.

### Non-Monotonicity of Welfare

Perhaps the most striking result is that welfare is **non-monotonic** in AI accuracy. There exists an interior optimal level of AI precision — meaning that **AI that is too accurate can reduce welfare**. This provides a theoretical motivation for **information design regulation**: deliberately limiting the effective precision of agentic AI recommendations to preserve human learning incentives.

### Policy Implications

The model suggests two policy directions:

1. **Garbling policy:** Intentionally limiting the precision of AI recommendations to maintain human learning incentives. This is counterintuitive — we typically want AI to be as accurate as possible — but the model shows that accuracy beyond a threshold is socially harmful.
2. **Strengthening knowledge aggregation capacity ($I$):** Improving the ability of communities to pool and share general knowledge unconditionally improves welfare and increases resilience against knowledge collapse.

### Empirical Indicators

Acemoglu et al. cite several empirical patterns consistent with substitution effects:

- **Stack Overflow:** Activity, human participation, and new knowledge generation declined after the introduction of ChatGPT (del Rio-Chanona et al., 2024).
- **Wikipedia:** Article reading and creation decreased in domains where ChatGPT is an effective substitute (Lyu et al., 2025).
- **Cognitive abilities:** Users of ChatGPT writing assistance showed reduced memory and argumentation capabilities, with measurable changes in neural connectivity (Kosmyna et al., 2025).
- **Creativity:** LLMs reduced users' creativity, particularly among younger users (Gerlich, 2025).

At the same time, they note positive examples: AlphaFold is cited as a case where AI augmented rather than replaced human scientific effort, and Brynjolfsson et al. (2025) showed that AI providing relevant background information to customer service agents improved decision-making quality.

---

## Framework 3: The Micro-Macro Separation (Hong et al.)

The first two frameworks present apparently contradictory conclusions: Agrawal et al. argue that AI augments scientific productivity, while Acemoglu et al. warn that AI erodes collective knowledge. Hong et al.'s research portfolio provides a framework for understanding why **both can be simultaneously correct** — the key is the level of analysis.

### Study 1: Search Augmentation at the Individual Level

**Paper:** "How User Adoption of ChatGPT Influences Commercial Search Patterns in Traditional Search Engines" (Hong, Yoon, Park, and Han; under major revision at *Management Science*)

Using propensity-score-matched difference-in-differences with Nielsen Korea panel data (November 2022 – April 2023), the study finds that ChatGPT adopters experienced:

- **+31.7%** increase in commercial search volume
- **+29.9%** increase in query composition diversity

The study conceptualizes ChatGPT as a **cognitive amplifier**: it translates vague exploratory needs into more precise, articulated queries. At the individual level, AI expands the scope and diversity of information-seeking behavior.

This finding aligns directly with Agrawal et al.'s augmentation thesis. Just as AlphaFold augments structural biologists' experimental capabilities, ChatGPT augments users' search and exploration capabilities. At the micro level, the evidence for augmentation is robust.

### Study 2: The Variance-Convergence Paradox

Hong's second research stream examines what happens when individual-level augmentation is aggregated to the population level. The central thesis is:

> Even if AI increases exploratory diversity for each individual, when **all users diversify through the same AI tool**, the collective outcome can be **convergence** rather than divergence.

This connects directly to Acemoglu et al.'s warning and to the empirical finding of Hao et al. (2025) that AI use in science is associated with a ~5% reduction in topic diversity and a 22% reduction in cross-scientist interaction — even as individual productivity increases.

### Within-Variance vs. Between-Variance

The mechanism behind the paradox becomes clear when we decompose diversity into two components:

- **Within-individual variance:** How diverse is a single person's search/research behavior?
- **Between-individual variance:** How different are individuals from each other?

If AI suggests similar "optimal" directions to all users, it is entirely possible for within-individual variance to increase (each person explores more) while between-individual variance decreases (everyone explores in similar directions). The net effect on collective knowledge diversity depends on which component dominates.

This distinction is analytically important. When IGL reports that AlphaFold users explored more structurally novel proteins (Part 3), we need to ask: is this within-variance (each lab exploring more) or between-variance (different labs exploring different things)? The answer has very different implications for the health of the knowledge ecosystem.

### A Unified Temporal Framework

Combining all three frameworks, a temporal dynamic emerges:

| Time Horizon | Dominant Effect | Supporting Evidence |
|--------------|----------------|---------------------|
| **Short-term** | Individual augmentation + exploration diversity | Hong (search diversity +30%), IGL (PDB submissions +45%) |
| **Medium-term** | Collective convergence begins | Variance-convergence paradox, Hao et al. (topic diversity -5%) |
| **Long-term** | Knowledge collapse risk | Acemoglu (learning incentive erosion → general knowledge depletion) |

This temporal structure is important because it explains why the empirical evidence is currently mixed. Most studies observe the short-term augmentation effects because they measure outcomes within the first few years of AI tool adoption. The medium-term convergence and long-term collapse effects may take longer to manifest — and by the time they do, they may be difficult to reverse.

### Two Key Implications

The micro-macro separation framework yields two implications that should inform the design of AI for Science:

1. **The type of AI tool matters.** AlphaFold (an open database/infrastructure) and ChatGPT (a recommendation/dialogue agent) have structurally different substitution-complementarity dynamics. Treating all AI tools as equivalent is an analytical error that leads to misleading policy conclusions.

2. **The level of analysis must be explicit.** When discussing AI's impact on science, failing to distinguish individual-level gains from collective-level losses allows contradictory conclusions to coexist as if both are unqualifiedly true. Productive debate requires specifying which level of analysis a given claim applies to.

---

## Synthesis: Why All Three Frameworks Are Needed

Each framework captures something that the others miss:

| Framework | What It Models | What It Misses |
|-----------|---------------|----------------|
| Agrawal et al. | AI's augmentation potential across scientific stages; bottleneck effects | Dynamic risks from reduced human learning effort |
| Acemoglu et al. | Long-term knowledge erosion from AI substitution | Conditions under which AI acts as complement rather than substitute |
| Hong et al. | Why augmentation and convergence coexist; the role of analysis level | Domain-specific mechanisms (addressed through within-tool variation) |

Taken together, they suggest that the impact of AI on scientific knowledge production is:

- **Stage-dependent:** AI accelerates some stages of science (design, hypothesis generation) more than others (testing, validation), and the overall effect is limited by the weakest link (Agrawal et al.).
- **Temporally non-trivial:** Short-term augmentation can coexist with medium-term convergence and long-term knowledge collapse (Acemoglu et al., Hong et al.).
- **Level-dependent:** Individual-level gains do not automatically translate to collective-level benefits — and can even coexist with collective-level losses (Hong et al.).

For those of us working on AI for Science, the takeaway is that optimizing for individual-level productivity gains is insufficient. The design of AI tools must also account for collective knowledge dynamics, the preservation of human learning incentives, and the avoidance of research direction distortion. How AI distorts research direction — and what the empirical evidence says about it — is the subject of Parts 2 and 3.

---

## References

- Agrawal, A. K., McHale, J., & Oettl, A. (2026). AI in Science. *NBER Working Paper 34953*.
- Acemoglu, D., Kong, D., & Ozdaglar, A. (2026). AI, Human Cognition and Knowledge Collapse. *NBER Working Paper 34910*.
- Hong, J., Yoon, S., Park, S., & Han, S. P. How User Adoption of ChatGPT Influences Commercial Search Patterns in Traditional Search Engines. Under revision at *Management Science*.
- Hao, Y., et al. (2025). AI and Scientific Research: Individual Productivity and Collective Scope.
- Brynjolfsson, E., et al. (2025). Generative AI at Work.
- del Rio-Chanona, R. M., et al. (2024). The Effect of ChatGPT on Stack Overflow.
- Lyu, H., et al. (2025). ChatGPT as a Substitute for Wikipedia.
- Kosmyna, N., et al. (2025). AI Writing Assistance and Cognitive Change.
- Gerlich, M. (2025). LLMs and Creativity.
- Mollick, E. (2024). The Jagged Frontier.
