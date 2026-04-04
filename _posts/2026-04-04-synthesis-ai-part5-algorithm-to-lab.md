---
title: "Synthesis AI Part 5: From Algorithm to Lab — CRO Integration and the Remaining Gap"
date: 2026-04-04 09:40:00 +0900
categories: [Drug Discovery, Small Molecule Design]
tags: [cro-integration, lab-automation, self-driving-lab, last-mile, synthesis]
math: true
---

**AI-Driven Synthesis in Drug Discovery**

*This is Part 5 of a 5-part series on AI-driven synthesis in drug discovery.*

- **Part 1:** [The Synthesis Bottleneck — Why "Make" Lags Behind](/posts/synthesis-ai-part1-synthesis-bottleneck/)
- **Part 2:** [Reaction Prediction — Can AI Predict What Chemistry Will Do?](/posts/synthesis-ai-part2-reaction-prediction/)
- **Part 3:** [Retrosynthesis — Can AI Plan How to Make a Molecule?](/posts/synthesis-ai-part3-retrosynthesis/)
- **Part 4:** [Synthesis-Aware Design — Making AI-Generated Molecules Makeable](/posts/synthesis-ai-part4-synthesis-aware-design/)
- **Part 5** (this post): From Algorithm to Lab — CRO Integration and the Remaining Gap


---

## The Last Mile Problem

An AI retrosynthesis tool produces a route. It looks clean: three steps, commercially available starting materials, known reaction types, predicted yields above 60%. The output is a SMILES string for each intermediate, a reaction class label for each step, and maybe a confidence score. **But what does the CRO that will actually synthesize this molecule receive?**

Not this. Not directly. The AI speaks in SMILES, reaction templates, and probability distributions. The CRO chemist speaks in standard operating procedures, catalog numbers, and safety data sheets. This gap — the "last mile" between computational proposal and physical execution — is the subject of this final Part, and the most practically important gap in the AI synthesis pipeline.

Let us briefly recall where we are:

- **Part 2** established that AI can predict reaction products with ~90% top-1 accuracy on benchmarks, but condition prediction remains underdeveloped
- **Part 3** showed that retrosynthesis tools can propose plausible routes, though route quality is hard to evaluate without experimental validation
- **Part 4** demonstrated that synthesis-aware generators (SynFlowNet, SynFormer) can produce molecules with routes guaranteed by construction

All of this computational machinery produces *proposals*. This Part asks the blunt question: **do any of these proposals actually work when someone tries to execute them?**

---

## 1. The CRO Handoff: What Actually Happens Today

### 1.1 The Current Workflow

Here is what the AI-to-synthesis pipeline looks like in practice at most pharmaceutical companies and biotech startups today:

```
The Real AI-to-Lab Pipeline (2024-2026)
════════════════════════════════════════

  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
  │  AI Route   │     │  Med Chem   │     │    SOP      │     │  CRO        │     │    CRO      │
  │  Proposal   │────▶│  Review &   │────▶│  Document   │────▶│  Receives   │────▶│  Executes   │
  │             │     │ Modification│     │  Creation   │     │  Package    │     │  (or not)   │
  │ SMILES +    │     │             │     │             │     │             │     │             │
  │ route +     │     │ "This step  │     │ Natural     │     │ Catalog #s, │     │ Actual      │
  │ confidence  │     │  won't work,│     │ language +  │     │ masses,     │     │ chemistry   │
  │             │     │  use X      │     │ specific    │     │ volumes,    │     │ on a bench  │
  │             │     │  instead"   │     │ conditions  │     │ equipment   │     │             │
  └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
        AI                 Human               Human               Human               Human
```

**The critical observation: CROs do not receive AI routes directly.** A human translation layer always intervenes — the medicinal chemist reviews the proposal, modifies steps, specifies conditions, and packages it with catalog numbers and safety notes. This is not a failure of adoption; it is a rational response to the current state of AI synthesis planning.

### 1.2 The Translation Problems

Why can't the AI output go straight to a CRO? The reasons are specific and instructive:

- **Route format non-standardization**: AI tools output routes in SMILES, RInChI, XDL, reaction SMARTS, or natural language — all different formats, none universally accepted by CROs. There is no "PDF" equivalent for synthesis routes.

- **Reagent availability mismatch**: The AI says "use 4-bromopyridine." But which supplier? Which catalog number? Is it in stock, or back-ordered for 12 weeks? A catalog ID in Enamine does not equal actual availability at the CRO's location.

- **Equipment constraints**: AI models assume generic equipment — "reflux," "column chromatography." The CRO has specific reactors and purification systems. A route requiring a photochemical flow reactor is useless if the CRO has none.

- **Scale mismatch**: AI retrosynthesis thinks in milligrams (the scale of reaction databases). CROs work in grams. Scale-up chemistry is different chemistry — mixing, heat transfer, and mass transfer all change behavior.

- **Safety and handling gaps**: AI routes may propose organolithium reagents at -78 C or high-pressure hydrogenation without safety flagging. A human must assess hazard, waste disposal, and regulatory compliance.

- **Condition underspecification**: As we discussed in Part 2, most AI retrosynthesis tools specify the reaction type but not the exact conditions — solvent, catalyst loading, temperature profile, workup procedure. The CRO chemist fills in these blanks from experience.

### 1.3 PostEra and the COVID Moonshot: A Case Study

**The COVID Moonshot project (Chodera et al., Science, 2023) represents the largest open-science experiment in AI-to-CRO synthesis execution to date.** PostEra served as the computational hub, using AI retrosynthesis (Manifold platform) to plan routes for community-designed molecules targeting the SARS-CoV-2 main protease. Enamine and other CROs executed the synthesis. All data was made publicly available.

Key lessons about the AI-to-lab gap:

- **Route modification rate was high**: The majority of AI-proposed routes were modified by CRO chemists before execution. Routes served as starting points, not as final instructions.
- **Availability was a constant bottleneck**: Building blocks that appeared available in catalogs were frequently out of stock, back-ordered, or available only in quantities too small for the required scale.
- **Simple chemistry dominated**: The most successful routes used well-established reaction types — amide couplings, Suzuki couplings, reductive aminations. Novel or unusual transforms proposed by AI tools had lower success rates.
- **Turnaround was still weeks**: Even with AI-accelerated route planning, the make cycle remained 2-4 weeks per compound, dominated by procurement and CRO queue times rather than by the chemistry itself.

**AI routes are useful as conversation starters with CRO chemists — they narrow the design space and provide a starting framework — but they are not yet executable specifications.**

---

## 2. Automated Synthesis Platforms

If human translation is the bottleneck, the obvious question is: can we remove the human from the loop entirely? Several platforms are attempting exactly this.

### 2.1 Chemify and XDL (Cronin Group, Glasgow)

**Chemify is the most ambitious attempt to date at making synthesis routes directly executable by machines** (Mehr et al., Science, 2020; Rohrbach et al., Science, 2022; Chemify startup, 2023).

The core idea is XDL — the Chemical Description Language:

```
XDL: A Programming Language for Chemistry
══════════════════════════════════════════

  Traditional synthesis procedure (natural language):
  ──────────────────────────────────────────────────
  "Add 4-bromobenzaldehyde (1.0 g, 5.4 mmol) to a
   round-bottom flask. Dissolve in THF (20 mL).
   Cool to 0 C. Add vinylmagnesium bromide (1.0 M
   in THF, 6.5 mL) dropwise over 15 min..."

              │
              ▼  XDL translation

  XDL procedure (machine-readable):
  ──────────────────────────────────
  <Synthesis>
    <Hardware>
      <Component id="reactor" type="flask" />
      <Component id="addition_funnel" type="funnel"/>
    </Hardware>
    <Procedure>
      <Add vessel="reactor"
           reagent="4-bromobenzaldehyde"
           mass="1.0 g" />
      <Add vessel="reactor"
           reagent="THF"
           volume="20 mL" />
      <SetTemperature vessel="reactor"
                      temp="0 C" />
      <Add vessel="reactor"
           reagent="vinylmagnesium bromide 1M THF"
           volume="6.5 mL"
           rate="0.43 mL/min" />
    </Procedure>
  </Synthesis>
```

The XDL procedure is parsed by a Chemputer (Cronin's robotic synthesis platform) and executed step by step. **The key insight is not that the current Chemputer can run every reaction — it cannot — but that the idea of a standard, machine-readable synthesis language is the correct abstraction.**

Current limitations:

- **Reaction scope is restricted**: Handles additions, heating, filtering, and extractions well, but cannot yet perform many common med chem transformations (e.g., Pd-catalyzed cross-couplings under inert atmosphere with complex workups)
- **Hardware coupling**: XDL procedures are tied to Chemputer hardware. A universal XDL that maps to any robotic platform does not yet exist.
- **Scale**: Demonstrations at mmol to gram scale only. Multi-gram synthesis is out of scope.

Despite these limitations, **the Cronin group has demonstrated synthesis of over 700 compounds from XDL-encoded procedures, including complex natural products, with minimal human intervention** (Rohrbach et al., Science, 2022). This proves the concept.

### 2.2 Cloud Laboratories

Cloud labs offer a different model: send your experiment request to a remote facility via an API.

**Emerald Cloud Lab (ECL)**:

- Cloud-based remote laboratory — users submit experiments through a web interface or programmatic API
- Full analytical suite: HPLC, MS, NMR, all operated remotely
- Chemistry support expanding but still limited — primarily routine transformations
- The API-driven model is conceptually aligned with AI integration: an AI system could submit synthesis requests programmatically

**Strateos (formerly Transcriptic)**:

- Originally biology-focused — cell culture, assay automation, compound management
- Expanding into chemistry workflows: compound dissolution, plate preparation, simple derivatizations
- High throughput for standardized operations; less suited for bespoke med chem synthesis

**The cloud lab model solves one part of the problem — removing geographic constraints and enabling programmatic access — but it does not solve the chemistry problem.** The range of executable reactions remains narrow compared to what a skilled CRO chemist can handle.

### 2.3 Self-Driving Laboratories

Self-driving labs represent the most complete vision: a closed loop where AI proposes an experiment, a robot executes it, instruments measure the outcome, and the AI updates its model and proposes the next experiment.

```
Self-Driving Lab: The Closed Loop
═════════════════════════════════

  ┌───────────────┐
  │   AI Model    │
  │  (Propose     │
  │  experiment)  │
  └──────┬────────┘
         │
         ▼
  ┌───────────────┐       ┌───────────────┐
  │   Robotic     │──────▶│  Analytical   │
  │   Synthesis   │       │  Instruments  │
  │   Platform    │       │  (HPLC, MS,   │
  │               │       │   NMR, etc.)  │
  └───────────────┘       └──────┬────────┘
                                 │
                                 ▼
                          ┌───────────────┐
                          │   Data        │
                          │   Processing  │
                          │   & Learning  │
                          └──────┬────────┘
                                 │
                                 │  Update model
                                 │
                          ┌──────▼────────┐
                          │   AI Model    │
                          │  (Updated     │
                          │  proposals)   │──── next cycle
                          └───────────────┘
```

Key players:

- **Kebotix**: Materials-focused self-driving lab; reaction optimization and materials discovery
- **Atinary**: Bayesian optimization platform for experimental design; chemistry and materials
- **IBM RXN / RoboRXN**: Integration of IBM's Molecular Transformer with robotic execution — one of the earliest attempts to connect AI prediction directly to robotic synthesis (Schwaller et al., 2020)
- **Acceleration Consortium** (University of Toronto): Large-scale initiative for self-driving labs across chemistry and materials science

**The most mature application is reaction condition optimization** — using Bayesian optimization to explore temperature, solvent, catalyst, and stoichiometry space for a known reaction type (Shields et al., Nature, 2021). The scope is narrow enough for robotic execution, and the feedback loop is fast enough for iterative learning.

Full multi-step synthesis in a closed loop — where the AI plans an entire route and the robot executes all steps, handling purification, characterization, and troubleshooting — remains out of reach. **The gap between optimizing conditions for one step and executing an entire multi-step route is enormous.**

### 2.4 Platform Comparison

| Platform | Type | Chemistry Coverage | Approach | Maturity |
|---|---|---|---|---|
| **Chemify / Chemputer** | Local robotic synthesis | ~100 reaction types demonstrated | XDL → robot execution | Research prototype; startup phase |
| **Emerald Cloud Lab** | Remote cloud lab | Moderate — expanding | API-driven experiment submission | Commercial; chemistry still limited |
| **Strateos** | Remote cloud lab | Narrow (bio-focused) | Robotic cloud operations | Commercial; chemistry is secondary |
| **Kebotix** | Self-driving lab | Materials + reaction opt. | Closed-loop Bayesian optimization | Commercial; narrow chemistry scope |
| **Atinary** | Optimization platform | Reaction conditions | Bayesian optimization + DoE | Commercial; planning layer only |
| **IBM RoboRXN** | AI + robotic synthesis | Narrow demonstration | Molecular Transformer → robot | Research demonstration |
| **Acceleration Consortium** | Multi-lab initiative | Broad ambition | Federated self-driving labs | Early-stage; infrastructure phase |

**The pattern across all platforms is consistent: each solves a piece of the problem, but none yet provides a general-purpose solution for executing arbitrary AI-proposed synthesis routes.**

---

## 3. Standardization: The Missing Infrastructure

### 3.1 The Fundamental Barrier

We have arrived at what is arguably the most important insight of this series: **the absence of a reaction description standard is the fundamental barrier to AI-to-lab integration.**

Consider the analogy from software engineering. Web applications communicate with servers through REST APIs — standardized interfaces with defined request and response formats. Code compiles to machine instructions through well-defined intermediate representations. The entire software stack is connected by standards. Chemistry has nothing equivalent.

```
Software vs Chemistry: The Standards Gap
═════════════════════════════════════════

  Software Stack (standardized):        Chemistry Stack (fragmented):
  ─────────────────────────────         ──────────────────────────────

  High-level code                       AI route proposal
       │                                     │
       ▼                                     ▼
  Compiler / Interpreter                ??? (human translation)
       │                                     │
       ▼                                     ▼
  Intermediate Representation           ??? (no standard IR)
       │                                     │
       ▼                                     ▼
  Machine code                          ??? (vendor-specific robot code)
       │                                     │
       ▼                                     ▼
  Hardware execution                    Physical synthesis
       │                                     │
       ▼                                     ▼
  Output / logs                         Product + analytical data
       │                                     │
       ▼                                     ▼
  Standard format (JSON, XML)           ??? (PDF, email, ELN entry)
```

Every "???" in the chemistry stack is a point where human intervention is currently required. **This is an infrastructure problem, not an algorithm problem.** No amount of improvement in retrosynthesis accuracy or generative model quality will close this gap without standardized interfaces.

### 3.2 Current Standardization Efforts

Several efforts are attempting to build this missing infrastructure:

**ORD (Open Reaction Database)**

The Open Reaction Database (Kearnes et al., J Am Chem Soc, 2021) is the most significant data standardization effort in reaction chemistry:

- Protocol buffer schema for recording reactions — reactants, products, conditions, yields, analytical data, all structured and machine-readable
- Designed to capture *complete* reaction records, including failed attempts and negative results
- Open-source, community-governed

Current limitations:

- Adoption growing but limited — primarily academic contributors
- Industry participation sparse due to IP concerns
- As of 2026, ~2M reactions — growing, but orders of magnitude smaller than Reaxys

**XDL / Chemical JSON**

XDL (discussed in Section 2.1) and Chemical JSON aim to make synthesis *procedures* machine-readable — encoding what to do (add reagent X, heat to Y degrees, stir for Z minutes) in structured format.

**The gap**: ORD standardizes reaction *data* (what happened). XDL standardizes synthesis *procedures* (what to do). Neither alone connects AI route proposals to physical execution.

**RInChI (Reaction InChI)** extends InChI to reactions — providing unique, canonical identifiers for reaction lookup and deduplication. It does not encode conditions or procedures; it is an identifier, not a description.

### 3.3 The "Synthesis API" Concept

What the field needs, but does not yet have, is something we might call a **Synthesis API** — a standardized interface between computational proposals and physical execution:

```
The Synthesis API Vision
════════════════════════

  AI System (any)                    Physical Lab (any)
  ───────────────                    ──────────────────
  SynFormer output                   Chemputer
  AiZynthFinder route                Emerald Cloud Lab
  ASKCOS proposal                    CRO bench chemist
  CGFlow generation                  Custom robotic platform
       │                                  ▲
       │                                  │
       ▼                                  │
  ┌────────────────────────────────────────────┐
  │            Synthesis API                    │
  │                                            │
  │  Input:   Route (standardized format)      │
  │           + building blocks (with IDs)     │
  │           + conditions (from prediction)   │
  │           + scale requirement              │
  │           + safety constraints             │
  │                                            │
  │  Output:  Execution plan (XDL or equiv.)   │
  │           + procurement list               │
  │           + equipment requirements         │
  │           + estimated timeline             │
  │           + hazard assessment              │
  │                                            │
  │  Feedback: Outcome data (ORD format)       │
  │            → feeds back to AI models       │
  └────────────────────────────────────────────┘
```

This interface does not exist today. Building it requires convergence of ORD (outcome data), XDL (procedure encoding), condition prediction models (Part 2), reagent availability APIs from suppliers (Enamine, Sigma-Aldrich), and equipment ontologies describing what a given lab or robot can do.

**The groups that will have the greatest practical impact are not necessarily those building better algorithms — they are those building better interfaces.**

---

## 4. Where We Are and What Remains

### 4.1 Series Summary: The Maturity Landscape

Across this five-part series, we have traced the AI synthesis pipeline from individual reaction prediction to lab execution. Here is where each component stands:

```
AI Synthesis Pipeline — Maturity Assessment
════════════════════════════════════════════

┌────────────────┐   ┌────────────────┐   ┌────────────────┐   ┌────────────────┐
│    Reaction    │   │    Retro-      │   │  Synthesis-    │   │      Lab       │
│   Prediction   │   │   synthesis    │   │    Aware       │   │   Execution    │
│    (Part 2)    │──▶│    (Part 3)    │──▶│    Design      │──▶│    (Part 5)    │
│                │   │                │   │    (Part 4)    │   │                │
│  Benchmark:    │   │  Route         │   │  Generate      │   │  CRO handoff   │
│  ~90% top-1    │   │  suggestion    │   │  makeable      │   │  is manual     │
│  Practice:     │   │  =/= route     │   │  molecules     │   │  automation    │
│  useful but    │   │  success       │   │  + routes      │   │  is early      │
│  incomplete    │   │                │   │                │   │                │
└────────────────┘   └────────────────┘   └────────────────┘   └────────────────┘

    Maturity:            Maturity:            Maturity:            Maturity:
    ** * * oo             ** oo                * oo                 ooo
   (maturing)         (functional           (emerging)           (nascent)
                       but fragile)

Legend:  * = solved   * = partially solved   o = open problem
```

What each maturity level means in practice:

| Component | What Works | What Doesn't Yet |
|---|---|---|
| **Reaction Prediction** | Product prediction ~90% top-1; atom mapping (RXNMapper) | Condition prediction (active research); yield prediction (r ~ 0.5-0.7, insufficient) |
| **Retrosynthesis** | Single-step retro; multi-step for common scaffolds | Novel scaffolds fragile; route quality assessment unsolved; experimental success rate poorly characterized (~50-70% per step) |
| **Synthesis-Aware Design** | Scoring (SAScore, RAscore); generation with guarantee (SynFlowNet, SynFormer) in research | Production deployment; binding integration (CGFlow) at proof-of-concept stage |
| **Lab Execution** | Single-step condition optimization (Bayesian opt.) | CRO handoff universally manual; multi-step closed-loop not at scale |

### 4.2 Highest-Leverage Investment Areas

Given this maturity landscape, where should resources — research effort, funding, engineering talent — be concentrated? We see four high-leverage areas:

**1. Condition prediction (connecting Part 2 to Part 5)**

This is, in our view, **the single most impactful unsolved problem in AI-driven synthesis**. Today's retrosynthesis tools tell you *what* reactions to run but not *how* to run them. The connection is direct: better condition prediction leads to more complete route specifications, which means less human translation, which means faster CRO execution and eventually direct robotic execution.

Promising directions include condition prediction models from Gao et al. (NeurIPS, 2023) at MIT, Schwaller's work on condition recommendation at EPFL, and the integration of condition prediction directly into retrosynthesis planning.

**2. Route validation (connecting Part 3 to Part 5)**

We need systematic studies of AI route success rates. The field currently operates on anecdote — a CRO chemist tries an AI route, it works or it does not, and the outcome is rarely recorded in structured format. What is needed:

- Large-scale prospective studies: 500+ AI routes executed at CROs, every outcome recorded in ORD format
- Stratified analysis by reaction type, target complexity, and AI tool
- Public release of data — including failures

AstraZeneca's work with AiZynthFinder (Genheden et al., J Cheminform, 2024) is the closest thing to this, but data remains largely internal.

**3. Standardization infrastructure (XDL/ORD adoption)**

Increasing ORD adoption for outcome recording and XDL for procedure encoding would unlock the pipeline. If CROs recorded outcomes in ORD format, we would rapidly build a dataset of AI route success/failure. If AI tools output in a standard format mapping to XDL, the human translation layer could be progressively automated.

**4. Closed-loop integration for specific reaction families**

The near-term opportunity is closed-loop systems for specific, well-characterized reaction families:

| Reaction Family | Suitability for Automation | Reason |
|---|---|---|
| Amide coupling | High | Well-understood, robust, few side reactions |
| Suzuki coupling | High | Standard conditions, good functional group tolerance |
| Reductive amination | High | Reliable with standard reductants |
| Buchwald-Hartwig amination | Moderate | Ligand/catalyst selection matters; optimization needed |
| C-H activation | Low | Highly condition-dependent, poor generalization |
| Organolithium chemistry | Low | Cryogenic, moisture-sensitive, safety concerns |

**Building these narrow-scope closed loops and expanding them incrementally is a more realistic path than attempting general-purpose automated synthesis.**

### 4.3 Timeline and Outlook

**Within 2-3 years (by ~2028)**:

- Condition-aware retrosynthesis tools that output complete reaction specifications (reactants + conditions + predicted yield) for common reaction types
- Standardized route interchange formats adopted by at least 2-3 major AI synthesis tools
- Closed-loop optimization systems for amide coupling, Suzuki coupling, and reductive amination — operating at scale in at least a few pharma companies
- ORD reaching 10M+ reactions with meaningful industry contribution

**Within 5 years (by ~2031)**:

- Condition-aware retrosynthesis + automated execution working in closed loop for 20-30 well-characterized reaction families
- AI route proposals that CRO chemists accept without modification >50% of the time (currently estimated at <20%)
- Cloud lab platforms capable of executing 3-4 step routes for standard medicinal chemistry transformations
- Integration of synthesis-aware generators (SynFormer-class) into production drug discovery pipelines at multiple pharma companies

**10+ year horizon**:

- General-purpose automated synthesis from arbitrary AI routes — handling novel reaction types, complex workups, and multi-step sequences without human intervention
- Self-driving synthesis labs that execute full DMTA cycles: design molecule, synthesize it, test it, learn, repeat
- Chemistry's equivalent of a "compiler" — a system that takes a target molecule and produces an executable synthesis program, end to end

**The bottleneck is shifting.** In 2020, the limiting factor was algorithm quality. In 2026, the algorithms have matured significantly. The limiting factors are now infrastructure (standards, data formats, APIs), data (conditions, negatives, outcomes), and integration (connecting computation to the physical world).

---

## 5. Closing: The Convergence Ahead

The synthesis bottleneck in drug discovery is real. "Make" consumes 3-6 weeks per DMTA cycle while "Design" has been compressed to hours. AI alone will not solve this. **What is needed is a convergence of better reaction data, better condition prediction, synthesis-aware generation, and standardized interfaces between computational proposals and physical labs.**

The pieces are being built:

- **Better data**: ORD establishing the standard for reaction outcome recording. Critical mass adoption would finally provide the condition and yield data that Part 2 identified as missing.
- **Better prediction**: Coley (MIT) pushing condition prediction and mechanism-level understanding. Schwaller (EPFL) advancing Transformer-based reaction models. Both moving toward condition-complete route specifications.
- **Better generation**: GFlowNet community (Mila — SynFlowNet, CGFlow, RGFN) and Transformer community (MIT — SynFormer) have demonstrated synthesis-guaranteed generation. Next step is production deployment.
- **Better automation**: Cronin's Chemify proving robotic execution from encoded procedures. AstraZeneca leading industrial integration. PostEra's Moonshot providing public data on the AI-to-CRO gap.
- **Better interfaces**: The Synthesis API concept does not yet exist, but components (ORD, XDL, supplier APIs) are emerging.

Convergence is beginning:

```
Convergence Map
═══════════════

  Reaction Data (ORD) ─────────────┐
                                   │
  Condition Prediction ────────────┤
  (Coley, MIT / Schwaller, EPFL)   │
                                   ├───▶  Integrated AI-to-Lab
  Synthesis-Aware Generation ──────┤      Synthesis Pipeline
  (Mila GFlowNet / MIT SynFormer) │
                                   │
  Automation + Standardization ────┤
  (Cronin XDL / Cloud Labs)        │
                                   │
  Industrial Validation ───────────┘
  (AstraZeneca / PostEra / 
   Insilico Medicine)
```

The challenge now is integration. Each piece works in isolation. Connecting them into a pipeline where an AI proposes a molecule, plans its synthesis with full conditions, encodes the procedure in a standard format, submits it to a robotic platform or CRO, receives structured outcome data, and updates its models — that is the engineering challenge of the next decade.

We opened this series with Amdahl's law applied to drug discovery: optimizing the fast component of a sequential process has negligible impact when the slow component dominates. Five parts later, we can be more specific. **The slow component — Make — is not a single bottleneck but a chain of bottlenecks: data scarcity (Part 1), condition prediction (Part 2), route quality (Part 3), synthesizability constraints (Part 4), and lab execution (Part 5).** Breaking the chain at any single point helps. Breaking it at all five points transforms the field.

The groups working on this — Coley at MIT, Schwaller at EPFL, AstraZeneca's chemistry AI team, Cronin's automation lab at Glasgow, the GFlowNet and SynFormer communities, the ORD consortium — are building the pieces. The challenge now is integration.

---

## Key Takeaways

- **The CRO handoff is universally manual today**: AI route proposals are reviewed, modified, and translated by human chemists before CRO execution. This translation layer exists because AI outputs lack condition details, reagent specifics, and safety assessments.

- **Automated synthesis platforms are promising but narrow**: Chemify (XDL-based robotic synthesis), cloud labs (Emerald Cloud Lab, Strateos), and self-driving labs (Kebotix, Atinary) each solve part of the problem. None yet provides general-purpose execution of AI-proposed routes.

- **The core barrier is infrastructure, not algorithms**: Chemistry lacks the standardized interfaces (APIs, data formats, procedure encodings) that software engineering takes for granted. ORD, XDL, and RInChI are early steps, but adoption is limited.

- **Condition prediction is the highest-leverage unsolved problem**: Bridging the gap between "what reaction to run" (retrosynthesis output) and "how to run it" (conditions, workup, scale) would transform AI proposals from suggestions into specifications.

- **The realistic near-term path is narrow-scope closed loops**: Closed-loop AI + robotic systems for specific reaction families (amide coupling, Suzuki, reductive amination) are achievable now. General-purpose automated synthesis is a 10+ year horizon.

- **The bottleneck is shifting from algorithms to integration**: The computational pieces — prediction, planning, generation — have matured significantly. The challenge now is connecting them to the physical world through standards, data, and engineering.

---

## References

- Chodera, J. et al. "Open Science Discovery of Potent Noncovalent SARS-CoV-2 Main Protease Inhibitors." *Science* (2023).
- Genheden, S. et al. "AiZynthFinder 4.0: Developments Based on Learnings from 3 Years of Industrial Application." *J. Cheminform.* (2024).
- Gao, H. et al. "Using Machine Learning To Predict Suitable Conditions for Organic Reactions." *ACS Cent. Sci.* (2018).
- Kearnes, S. et al. "The Open Reaction Database." *J. Am. Chem. Soc.* (2021).
- Mehr, S. H. M. et al. "A Universal System for Digitization and Automatic Execution of the Chemical Synthesis Literature." *Science* (2020).
- Rohrbach, S. et al. "Digitization and Validation of a Chemical Synthesis Literature Database in the ChemPU." *Science* (2022).
- Schwaller, P. et al. "Predicting Retrosynthetic Pathways Using Transformer-Based Models and a Hyper-Graph Exploration Strategy." *Chem. Sci.* (2020).
- Shields, B. J. et al. "Bayesian Reaction Optimization as a Tool for Chemical Synthesis." *Nature* (2021).
- Thakkar, A. et al. "Artificial Intelligence and Automation in Computer Aided Synthesis Planning." *React. Chem. Eng.* (2021).
- Gao, W., Luo, S. & Coley, C. W. "SynFormer: Synthesizable Molecular Generation with Transformer-based Building Block Selection." *PNAS* (2025).
- Cretu, A. et al. "SynFlowNet: Towards Molecule Design with Guaranteed Synthesis Pathways." *ICLR* (2025).
- Shen, S. et al. "CGFlow: Synthesizable Molecular Generation with Co-folding Guidance." *ICML* (2025).

---

*This concludes the 5-part series "AI-Driven Synthesis in Drug Discovery." Parts 1-4 are available in this repository.*
