---
title: "Impact of AI on Scientific Knowledge Production — Part 2: Beyond the Productivity Debate — How AI Distorts Research Direction"
date: 2026-03-22 10:00:00 +0900
categories: [AI & ML, Social Impact of AI]
tags: [knowledge-collapse, ai-economics, work-to-the-ai, research-direction, gans-model]
math: true
---

*This is Part 2 of a three-part series examining how AI affects scientific knowledge production.*

- **Part 1:** [Three Theoretical Frameworks](/posts/impact-of-ai-on-science-part1-three-theoretical-frameworks/)
- **Part 2** (this post): Beyond the Productivity Debate — How AI Distorts Research Direction
- **Part 3:** [Empirical Evidence from AlphaFold and Open Questions](/posts/impact-of-ai-on-science-part3-empirical-evidence-alphafold/)

---

## Introduction

In Part 1, I reviewed three frameworks for understanding AI's impact on scientific knowledge production. Agrawal et al. model AI as a prediction machine that augments the scientific production function; Acemoglu et al. warn that AI dependency can erode collective knowledge over time; and Hong et al. show that augmentation and convergence can coexist at different levels of analysis.

This post examines two questions. First, how robust is Acemoglu's pessimistic forecast? A substantial body of criticism exists, and understanding the debate landscape is essential for calibrating how seriously to take the knowledge collapse risk. Second — and more fundamentally — is the productivity debate even asking the right question? Joshua Gans's model suggests that a more important concern is not *how much* AI changes scientific output, but *in what direction* it steers research.

---

## The Debate Landscape: Critiques of Acemoglu's Pessimism

Acemoglu has maintained a consistently cautious position on AI's economic and social impact across multiple publications. His "Simple Macroeconomics of AI" (NBER Working Paper 32487, 2024) estimates that AI will increase TFP by approximately +0.66% and GDP by +0.90–1.16% over the next decade — roughly +0.07 percentage points per year. He explicitly excludes the possibility of AI accelerating scientific discovery, stating that it is "unlikely within the ten-year time frame."

This position has attracted significant pushback from multiple directions.

### Counterargument 1: Same Formula, Different Conclusions

Philippe Aghion — the 2025 Nobel laureate in Economics — and Bunel (2024) used **Acemoglu's own task-based formula** but substituted different empirical parameter estimates. The result: annual productivity growth of **+0.68–1.3 percentage points** — approximately **10 times** Acemoglu's estimate.

The key disagreements are in the estimates of (a) what fraction of economic tasks are automatable, and (b) how large the cost savings per automated task are. Using Acemoglu's own parameters as a lower bound and their own as an upper bound, Aghion et al. obtain a range of 0.07–1.24%p, with a median of 0.68%p. This demonstrates that the same analytical framework, with different but defensible empirical inputs, produces dramatically different conclusions.

The Anthropic Economic Index (Levine, 2025) provides another data point. By mapping 4 million Claude conversations to O*NET occupational classifications, it found that **23.7%** of wage-weighted labor tasks are already being automated or augmented — compared to Acemoglu's estimate of 4.6%. The implied TFP impact is +3.46%, approximately 5 times Acemoglu's figure.

### Counterargument 2: Missing Channels

Maxwell Tabarrok's "Contra Acemoglu on AI" offers the most detailed methodological critique. He argues that Acemoglu's paper claims to address three channels of AI's economic impact but effectively ignores two of them:

**Deepening automation** — the application of AI to tasks previously automated by earlier technologies — is dismissed in a single sentence. Tabarrok points out that transformer-based models are already deployed in robotics, autonomous driving, and fraud detection, and that deepening automation was the engine of "the fastest economic growth in American history" during the Second Industrial Revolution.

**New task creation** is included only in its negative form (disinformation generation). Acemoglu's own prior research (Acemoglu & Restrepo, 2019) emphasized the importance of new task creation, yet this paper includes only destructive new tasks. Goldman Sachs data shows that 60% of today's workers hold jobs that did not exist in 1940, and that 85% of employment growth over the past 80 years came from technology-created new occupations.

**Research productivity** is excluded with the assertion that AI-driven scientific breakthroughs are "unlikely within the ten-year time frame." This ignores compounding effects: AlphaFold (released 2021) is already transforming structural biology, and AI coding agents are rapidly increasing software development productivity.

### Counterargument 3: Macroeconomists' Direct Critiques

**Lawrence Summers** (Harvard) stated that while he respects Acemoglu, "the analysis is not persuasive." His core criticism is that the framework completely excludes the possibilities of accelerated scientific progress, improved decision-making, and social-scientific advances. He compared it to IBM's analysis concluding that "the global computer market is five mainframes" — a framework that structurally cannot capture transformative second- and third-order effects.

**Chad Jones and Tonetti** (Stanford, 2026) analyzed automation's impact through the lens of "weak links" — tasks that are difficult to automate and therefore constrain output. They find that output by 2040 would be 4% higher than a no-acceleration baseline, and 19% higher by 2060. More optimistic than Acemoglu, but far from explosive growth.

**Korinek and Trammell** (NBER, 2024) make a structural point: Acemoglu's conclusion that AI is not transformative depends on the assumption that labor remains a binding bottleneck. If AI can automate the labor bottleneck itself — through self-improving R&D — the premise collapses.

### The Prediction Spectrum

The range of estimates across researchers and institutions is striking:

| Position | Representative | Annual Productivity Growth Added |
|----------|---------------|----------------------------------|
| Very pessimistic | Acemoglu (2024) | +0.07%p |
| Pessimistic-conservative | Robert Gordon, IMF | +0.1–0.8%p |
| Moderate-conservative | Tyler Cowen, OECD | +0.25–0.7%p |
| Moderate | Aghion & Bunel, Goldman Sachs | +0.68–1.5%p |
| Optimistic | Brynjolfsson, Anthropic/Clark | +2.8–5%p |
| Very optimistic | Amodei (Anthropic) | +5–15%p |
| Extremely optimistic | Korinek & Suh (AGI scenario) | +18%p |

As Tom Cunningham has observed, most economist forecasts treat AI as a **one-time productivity shock** and assume no further AI improvement. Given AI's ongoing development trajectory, this is a remarkable assumption.

### Critiques of the Knowledge Collapse Model Itself

Beyond the productivity debate, the knowledge collapse model (NBER 34910) has drawn specific methodological criticisms:

| Assumption | Critique |
|------------|----------|
| **Complete substitution** | The model assumes AI fully substitutes human learning effort, but complementarity is more common empirically. AlphaFold stimulated rather than replaced experimental work. |
| **Homogeneous community** | The "island" model ignores institutional knowledge preservation mechanisms — universities, research institutes, open-source communities, and experienced practitioners who serve as repositories of knowledge. |
| **Fixed knowledge taxonomy** | The model underestimates the possibility that AI interaction creates **new cognitive capabilities** rather than simply substituting existing ones (Cordasco, 2026). |
| **Single cohort** | No overlapping generations or preference heterogeneity; each period features an identical cohort making a single equilibrium effort choice. |

**Thiemo Fetzer** provides a structural reinterpretation: knowledge collapse is not a technological inevitability but a consequence of **institutional design failure**. The current architecture of capitalism rewards privatizable prediction while underinvesting in verification, reproduction, public reasoning, and the slow accumulation of shared knowledge. With the right institutional incentives, knowledge collapse can be prevented.

This reframing matters: it shifts the conversation from "should we limit AI accuracy?" to "how should we design the institutions around AI?"

---

## The Core Problem: How AI Distorts Research Direction (Gans, 2025)

**Source:** NBER Working Paper 33566, "A Quest for AI Knowledge" (March 2025, revised December 2025)

The productivity debate — how much does AI increase scientific output? — is important but incomplete. Joshua Gans's model identifies a more fundamental question: **does AI change where scientists choose to look?**

### Model Setup: Knowledge as a Spatial Structure

Gans extends Carnehl and Schneider's (2025, *Econometrica*) framework to model knowledge as a continuous space on the real line, governed by Brownian motion. Each point $x$ has a true answer $y(x)$, and the uncertainty about unknown questions increases with distance from the nearest known point.

A scientist chooses where to place the next knowledge point: either **deepening** (filling gaps between existing knowledge points) or **expanding** (pushing beyond the current frontier).

### Two Types of AI

The model distinguishes two types of AI tools based on their economic function:

| AI Type | Role | Effect |
|---------|------|--------|
| **S-AI (Scientist AI)** | Reduces research cost (supply side) | Discounts the cost of filling existing gaps by factor $\alpha_S$. No help at the frontier. |
| **DM-AI (Decision-Maker AI)** | Improves decision precision (demand side) | Reduces posterior variance within its range $R_D$ by factor $\alpha_D$. Expands the actionable scope of existing knowledge. |

Both AI types are **interpolative** — they operate between existing knowledge points but cannot extrapolate beyond the frontier. This assumption is critical and empirically well-grounded: current AI systems are much better at interpolation than extrapolation.

### The Three Regimes of DM-AI

The central result concerns how DM-AI affects the scientist's choice between deepening and expansion. The impact is **non-monotonic** in AI capability ($R_D$), falling into three regimes:

| Regime | Condition | Scientist Behavior | Novelty |
|--------|-----------|-------------------|---------|
| **Ignore** | $R_D < \tilde{R}_D$ | AI too weak to matter; scientist chooses original optimum $d_0$ | No change |
| **Truncate** | $\tilde{R}_D \leq R_D < d^*_E$ | Scientist sets $d = R_D$ — "works to the AI" | **Decreases** |
| **Enlarge** | $R_D \geq d^*_E$ | Scientist pursues true frontier expansion $d^*_E > d_0$ | **Increases** |

### "Work to the AI": The Streetlight Effect Formalized

The **Truncate regime** is the most concerning. Here, the scientist voluntarily pays a **novelty tax** — choosing a less novel research location than their original optimum — in exchange for an **AI subsidy**: the DM-AI can leverage the discovery for better decision-making, increasing its practical value.

This is the economic formalization of the **streetlight effect** (searching where the light is, not where the answer might be). The scientist optimizes for the intersection of scientific value and AI usability, which systematically pulls research toward areas where AI is already capable — and away from genuinely novel frontiers.

The analogy is precise: just as someone searches for dropped keys under the streetlight because visibility is better there, a scientist researches problems within AI's capability range because the downstream value is higher — even though the scientifically optimal location lies beyond the AI's reach.

### Extensions: Deeper Consequences

Gans develops several extensions that deepen the analysis:

#### Dynamic Stagnation (Proposition 5)

In the Truncate regime, scientists set $d_t = R_{D,t}$, which means AI receives **no out-of-distribution training data**. The AI's capability range evolves as:

$$R_{D,t+1} = (1-\delta)R_{D,t} + \gamma \max\{0, d_t - R_{D,t}\}$$

When $d_t = R_{D,t}$, the second term vanishes. The AI's range stagnates permanently. This creates a **self-reinforcing trap**: scientists work within AI's range → AI gets no frontier data → AI cannot expand its range → scientists continue working within the stagnant range.

This is arguably the most dangerous scenario for AI in Science: not a dramatic failure, but a quiet settling into a fixed capability frontier that becomes self-perpetuating.

#### Scientific Kill Zone

In domains where AI achieves very high interpolation accuracy, the marginal value of human scientific verification drops to near zero. Scientists have an incentive to research only areas where they can **beat the AI** — leading to a **bifurcation**: either abandon research within the AI's range entirely, or leap to radically novel territories far beyond it.

#### Competitive Polarization (Proposition 6)

When two scientists compete in the Truncate regime, a **polarized equilibrium** can emerge:

- One scientist plays it safe: $d = R_D$ (AI-compatible research)
- The other goes fully independent: $d = d_0 > R_D$ (AI-independent research)
- No one researches the gap between $R_D$ and $d_0$ — the **middle ground is hollowed out**

This means AI could make incremental innovation economically non-viable, leaving only the extremes of safe AI-compatible work and risky AI-independent exploration.

#### Flight to Rigour

When AI hallucination is significant ($\alpha_{perc} < \alpha_{real}$), the relative value of verifying and consolidating existing knowledge increases. Science shifts from "speculation" (frontier expansion) toward "verification" (deepening). While this might seem positive, it comes at the cost of reduced exploration at the frontier.

### Connecting to Acemoglu

Gans's and Acemoglu's models are complementary:

- **Acemoglu** models the reduction in *how much* humans learn — the quantity of learning effort.
- **Gans** models the distortion of *where* humans learn — the direction of research effort.
- Gans's Truncate regime combined with dynamic stagnation (Proposition 5) can be understood as a **spatial version** of Acemoglu's knowledge collapse: not a total loss of knowledge, but a permanent confinement of knowledge to the regions where AI is already capable.

---

## Synthesis: The Real Question for AI for Science

The productivity debate — whether AI will add 0.07 or 18 percentage points to annual growth — is ultimately an empirical question that will be resolved by time. But the more actionable question is structural: **does AI distort the direction of scientific inquiry, and if so, under what conditions?**

From the frameworks reviewed in Parts 1 and 2, we can identify three key concerns:

1. **Bottleneck persistence** (Agrawal): AI accelerates some stages of science but not others. The weakest link constrains overall progress. Celebrating AI-driven acceleration of the Design stage while ignoring persistent Testing bottlenecks overstates AI's net impact.

2. **Knowledge erosion** (Acemoglu): AI that substitutes for human learning effort can destroy collective knowledge over time. The risk is highest for agentic AI that provides context-specific recommendations without requiring human engagement.

3. **Direction distortion** (Gans): Even without knowledge collapse, AI can systematically steer research toward areas where AI is already capable — the "Work to the AI" phenomenon — creating self-reinforcing stagnation at the capability frontier.

These are not merely theoretical concerns. In Part 3, we examine what the empirical evidence from AlphaFold 2 — the most thoroughly studied case of AI in science — tells us about which of these dynamics is actually operating.

---

## References

- Acemoglu, D. (2024). [The Simple Macroeconomics of AI.](https://www.nber.org/papers/w32487) *NBER Working Paper 32487*.
- Acemoglu, D. & Johnson, S. (2023). [*Power and Progress: Our Thousand-Year Struggle Over Technology and Prosperity*.](https://shapingwork.mit.edu/power-and-progress/) PublicAffairs.
- Aghion, P. & Bunel, S. (2024). [AI and Growth: Where Do We Stand?](https://www.frbsf.org/wp-content/uploads/AI-and-Growth-Aghion-Bunel.pdf)
- Gans, J. S. (2025). [A Quest for AI Knowledge.](https://www.nber.org/papers/w33566) *NBER Working Paper 33566*.
- Gans, J. S. (2026). [Growth in AI Knowledge.](https://www.nber.org/papers/w33907) *NBER Working Paper 33907*.
- Jones, C. & Tonetti, C. (2026). Past Automation and Future AI: How Weak Links Tame the Growth Explosion.
- Korinek, A. & Trammell, P. (2024). [Economic Growth under Transformative AI.](https://www.nber.org/papers/w31815) *NBER Working Paper 31815*.
- Levine, D. (2025). [Anthropic Economic Index.](https://www.anthropic.com/news/the-anthropic-economic-index) Anthropic.
- Tabarrok, M. (2024). [Contra Acemoglu on AI.](https://www.maximum-progress.com/p/contra-acemoglu-on-ai) *Maximum Progress*.
- Carnehl, C. & Schneider, J. (2025). [A Quest for Knowledge.](https://onlinelibrary.wiley.com/doi/full/10.3982/ECTA22144) *Econometrica*, 93(2), 623–659.
