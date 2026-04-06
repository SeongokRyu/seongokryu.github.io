---
title: "Causal Inference Part 2: Graphs and Interventions — Structural Causal Models"
date: 2026-04-06 13:57:00 +0900
categories: [AI & ML, ML Theory]
tags: [causal-inference, scm, dag, d-separation, do-calculus, backdoor-criterion]
math: true
---

**Causal Inference — From First Principles to Automated Reasoning**

*This is Part 2 of an 8-part series.*

- **Part 0:** [Beyond Correlation — Why Causal Inference?](/posts/causal-inference-part0-beyond-correlation/)
- **Part 1:** [The Language of Causation: Potential Outcomes](/posts/causal-inference-part1-potential-outcomes/)
- **Part 2:** Graphs and Interventions — Structural Causal Models (this post)
- **Part 3:** [Identification — From Design to Estimand](/posts/causal-inference-part3-identification/)
- **Part 4:** [Estimation — From Estimand to Estimate](/posts/causal-inference-part4-estimation/)
- **Part 5:** [Heterogeneous Effects — Who Benefits?](/posts/causal-inference-part5-heterogeneous-effects/)
- **Part 6:** [The Causal Pipeline — From Data to Decision](/posts/causal-inference-part6-causal-pipeline/)
- **Part 7:** [The Causal Agent — Automated Causal Reasoning](/posts/causal-inference-part7-causal-agent/)

---

> "A DAG is not a picture. It is a set of testable claims about the world."

In Part 1 we built the potential outcomes framework: define $Y_i(1)$ and $Y_i(0)$, acknowledge the fundamental problem of causal inference, and state assumptions under which $\tau = E[Y_i(1) - Y_i(0)]$ is identified from data. The framework is powerful for *defining* what we want. It is silent on *how to decide what to adjust for*.

Suppose you face a new observational study. You have twenty covariates. Which should you condition on? Which should you leave alone? Condition on the wrong set and you introduce bias. Condition on the right set and you identify the causal effect. **The potential outcomes framework alone cannot answer this question. You need a graph.**

This post introduces Judea Pearl's Structural Causal Model (SCM) framework. By the end, you will be able to:

1. Write down an SCM and draw its DAG.
2. Classify every path as a fork, chain, or collider.
3. Apply $d$-separation to read conditional independencies from a graph.
4. Use the $\mathrm{do}$-operator to distinguish intervention from observation.
5. State the backdoor and frontdoor criteria.
6. Understand how PO and SCM complement each other.

---

## 1. Structural Causal Models — Equations + Graph

### The Four Components

**A structural causal model is a mathematical object with four components that jointly specify how the world generates data.** Formally, an SCM is a tuple:

$$\mathcal{M} = (\mathbf{U}, \mathbf{V}, \mathcal{F}, P(\mathbf{U}))$$

| Component | Meaning | Role |
|-----------|---------|------|
| $\mathbf{U}$ | Exogenous variables | Unobserved background causes; determined outside the model |
| $\mathbf{V}$ | Endogenous variables | Observed variables; determined inside the model |
| $\mathcal{F}$ | Structural equations | One equation per endogenous variable: $V_j := f_j(Pa_j, U_j)$ |
| $P(\mathbf{U})$ | Noise distribution | Joint distribution over exogenous variables |

Each structural equation is an *assignment*, not an algebraic equality. The symbol $:=$ means "is determined by." The equation $Y := f(W, U_Y)$ asserts that $Y$ is generated as a function of $W$ and noise $U_Y$. It does *not* assert that $W$ is a function of $Y$.

### From Equations to Graph

The directed acyclic graph (DAG) of an SCM is built by a single rule:

- For each structural equation $V_j := f_j(Pa_j, U_j)$, draw an arrow from every variable in $Pa_j$ to $V_j$.

**The DAG is the graph of the structural equations — nothing more, nothing less.** Every arrow represents a direct causal claim. Every missing arrow represents an independence claim. The graph is a *commitment* about the data-generating process.

Why *acyclic*? The graph must contain no directed cycles because each variable is determined by its parents, which are determined by their parents, and so on. A cycle ($A \rightarrow B \rightarrow A$) would mean $A$ causes $B$ and $B$ causes $A$ simultaneously — this is not a well-defined assignment. Feedback loops that operate over time are handled by unrolling the time dimension (separate nodes for $A_t$ and $A_{t+1}$), not by adding cycles.

### A Concrete Example

Consider a study of whether a drug ($W$) reduces blood pressure ($Y$), where patient age ($A$) influences both drug assignment and blood pressure.

**Structural equations:**

$$A := U_A$$

$$W := f_W(A, U_W)$$

$$Y := f_Y(W, A, U_Y)$$

Here $U_A, U_W, U_Y$ are mutually independent noise terms.

**DAG:**

```
    A
   / \
  v   v
  W ──> Y
```

Three arrows encode three causal claims: age causes drug assignment, age causes blood pressure, and the drug causes blood pressure. The *absence* of an arrow from $Y$ to $W$ encodes the claim that blood pressure does not cause drug assignment.

### The Markov Factorization

Every SCM induces a factorization of the joint distribution over $\mathbf{V}$. If $\mathcal{G}$ is the DAG of $\mathcal{M}$, then:

$$P(V_1, V_2, \dots, V_n) = \prod_{j=1}^{n} P(V_j \mid Pa_j)$$

This is the *causal Markov condition*: each variable is independent of its non-descendants given its parents. The factorization is not merely a mathematical convenience — it reflects the modular structure of the data-generating process. Intervening on $V_j$ replaces $P(V_j \mid Pa_j)$ with a point mass, leaving all other factors unchanged.

### SCM vs. Regression

> **Pitfall**: An SCM is not a regression model. The equation $Y := f(W, U_Y)$ means $W$ *causes* $Y$, not that $W$ *predicts* $Y$. A regression equation $\hat{Y} = \hat{\beta}_0 + \hat{\beta}_1 W$ is a statistical summary. A structural equation is a causal claim. Reversing a regression equation gives an equivalent prediction; reversing a structural equation gives a different causal model.

| Property | Regression model | Structural equation |
|----------|-----------------|-------------------|
| Symbol | $=$ (equality) | $:=$ (assignment) |
| Direction | Symmetric | Asymmetric |
| Interpretation | Statistical association | Causal mechanism |
| Reversible? | Yes (predict $W$ from $Y$) | No: $Y := f(W, U\_{Y}) \nLeftrightarrow W := g(Y, U\_{W})$ |

---

## 2. Three Structures, Three Rules

Every path in a DAG is a sequence of arrows. The three elementary building blocks determine whether a path transmits statistical association.

### The Three Elementary Structures

```
Confounder (Fork):         Mediator (Chain):        Collider:

      Z                         W                     W     Z
     / \                        |                      \   /
    v   v                       v                       v v
    W   Y                       M                        C
                                |
                                v
                                Y
```

**Every DAG, no matter how complex, is built from forks, chains, and colliders — and the rules for how information flows through each structure are the entire foundation of causal graphical reasoning.**

### Fork ($W \leftarrow Z \rightarrow Y$)

$Z$ is a *common cause* of $W$ and $Y$. Information flows from $W$ to $Y$ through $Z$, creating a spurious association.

- **Default**: Path is *open*. $W$ and $Y$ are statistically associated.
- **Condition on $Z$**: Path is *blocked*. $W \perp\!\!\!\perp Y \mid Z$.
- **Intuition**: Once you know the common cause, the two effects carry no information about each other.

### Chain ($W \rightarrow M \rightarrow Y$)

$M$ *mediates* the effect of $W$ on $Y$. Information flows sequentially.

- **Default**: Path is *open*. $W$ and $Y$ are associated through $M$.
- **Condition on $M$**: Path is *blocked*. $W \perp\!\!\!\perp Y \mid M$.
- **Intuition**: If you hold the mediator fixed, the upstream cause cannot influence the downstream outcome through this path.

> **Pitfall**: Conditioning on a mediator blocks the *causal* path. If you want the total effect of $W$ on $Y$, do not condition on $M$.

### Collider ($W \rightarrow C \leftarrow Z$)

$C$ is a *common effect* of $W$ and $Z$. This structure behaves opposite to the other two.

- **Default**: Path is *blocked*. $W \perp\!\!\!\perp Z$ (the causes are independent).
- **Condition on $C$**: Path is *opened*. $W \not\perp\!\!\!\perp Z \mid C$.
- **Intuition**: Knowing the common effect creates an artificial link between its causes. If the floor is wet ($C$), learning that it did not rain ($W=0$) makes a sprinkler ($Z=1$) more likely.

> **Pitfall**: Conditioning on a collider (or any of its descendants) creates bias. This is the most counterintuitive rule in causal inference and the most common mistake in applied work.

### Summary Table

| Structure | Path type | Default status | Effect of conditioning on middle node |
|-----------|-----------|---------------|--------------------------------------|
| Fork ($W \leftarrow Z \rightarrow Y$) | Non-causal (confounding) | Open | **Blocked** (removes spurious association) |
| Chain ($W \rightarrow M \rightarrow Y$) | Causal (mediation) | Open | **Blocked** (removes causal flow) |
| Collider ($W \rightarrow C \leftarrow Z$) | Neither | Blocked | **Opened** (creates spurious association) |

### Worked Example: Good vs. Bad Conditioning

Consider this five-node DAG:

```
    A ──────> Y
    |         ^
    v         |
    W ──> M ──┘
          |
          v
          C <── Z
```

Structural equations:

- $A := U_A$
- $W := f_W(A, U_W)$
- $M := f_M(W, U_M)$
- $Y := f_Y(A, M, U_Y)$
- $C := f_C(M, Z, U_C)$

Paths from $W$ to $Y$:

1. $W \rightarrow M \rightarrow Y$ — **causal chain**. Open. Do not block.
2. $W \leftarrow A \rightarrow Y$ — **confounding fork**. Open. Must block.

**Good conditioning**: Condition on $\{A\}$. Blocks the confounding fork (path 2). Leaves the causal chain (path 1) open. The effect of $W$ on $Y$ is identified.

**Bad conditioning**: Condition on $\{A, M\}$. Blocks both paths. You now estimate neither the total effect nor the confounding — you have removed the causal signal.

**Worse conditioning**: Condition on $\{A, C\}$. Blocks the fork via $A$ but opens $W \rightarrow M \rightarrow C \leftarrow Z$ by conditioning on the collider $C$. If $Z$ and $Y$ share any path (even through unobserved variables), this creates bias.

### Practitioner's Decision Rule

When deciding whether to condition on a variable $V$:

1. **Is $V$ a confounder (common cause)?** Condition on it. Blocking spurious association is almost always correct.
2. **Is $V$ a mediator (on the causal path)?** Do not condition on it if you want the total effect. Condition on it only for direct/indirect effect decomposition.
3. **Is $V$ a collider (common effect)?** Do not condition on it. Do not condition on its descendants either.
4. **Unsure?** Draw the DAG. Apply $d$-separation. The graph gives a definitive answer.

---

## 3. $d$-Separation — Reading Independence from the Graph

The three building-block rules combine into a single algorithm for reading conditional independencies from any DAG.

### Formal Definition

> **Definition 2.1 ($d$-separation).** A path $p$ between nodes $X$ and $Y$ in a DAG is *blocked* by a set $\mathbf{Z}$ if and only if:
>
> (a) $p$ contains a fork $\cdots \leftarrow V \rightarrow \cdots$ or a chain $\cdots \rightarrow V \rightarrow \cdots$ such that $V \in \mathbf{Z}$, **or**
>
> (b) $p$ contains a collider $\cdots \rightarrow V \leftarrow \cdots$ such that $V \notin \mathbf{Z}$ and no descendant of $V$ is in $\mathbf{Z}$.
>
> $X$ and $Y$ are $d$-separated by $\mathbf{Z}$ (written $X \perp_d Y \mid \mathbf{Z}$) if *every* path between $X$ and $Y$ is blocked by $\mathbf{Z}$.

In words: a path is blocked if you condition on a non-collider along it, or if you fail to condition on a collider (or its descendants) along it. $d$-separation holds when *all* paths are blocked.

### From Graph to Distribution

**$d$-separation in the graph implies conditional independence in any distribution generated by the SCM.** Formally, if $\mathcal{M}$ is an SCM with DAG $\mathcal{G}$, then:

$$X \perp_d Y \mid \mathbf{Z} \text{ in } \mathcal{G} \quad \implies \quad X \perp\!\!\!\perp Y \mid \mathbf{Z} \text{ in } P_{\mathcal{M}}$$

The converse — conditional independence implies $d$-separation — holds under the *faithfulness* assumption: the distribution has no independencies beyond those implied by the graph.

> **Assumption 2.1 (Faithfulness).** The distribution $P$ generated by SCM $\mathcal{M}$ has no conditional independencies beyond those entailed by $d$-separation in $\mathcal{G}$.

Faithfulness can fail in knife-edge cases — for example, when two causal paths produce effects that exactly cancel. In practice, such exact cancellations are measure-zero events in continuous parameter spaces, so faithfulness is a mild assumption for most applied work. Causal discovery algorithms (Part 5) rely heavily on faithfulness to infer graph structure from data.

### Step-by-Step Algorithm

Given a DAG $\mathcal{G}$, nodes $X$ and $Y$, and a conditioning set $\mathbf{Z}$:

1. **Enumerate** all paths between $X$ and $Y$ (ignoring arrow directions). A path is any sequence of adjacent nodes that does not revisit any node.
2. **For each path**, walk along the nodes and classify each interior node as part of a fork, chain, or collider by examining the two arrows adjacent to it.
3. **Check blocking**: a path is blocked if *any* interior node along it satisfies condition (a) or (b) of Definition 2.1.
4. **Conclude**: if every path is blocked, $X \perp_d Y \mid \mathbf{Z}$. If any path remains unblocked, $X$ and $Y$ are $d$-connected given $\mathbf{Z}$.

A common mistake is to forget step 1: you must consider *all* paths, not just directed paths. Two variables can be $d$-connected through paths that reverse direction multiple times.

| Step | What to do | Common mistake |
|------|-----------|----------------|
| Enumerate paths | Include paths with arrows in both directions | Only considering directed paths |
| Classify nodes | Check arrow directions at each interior node | Confusing chain with collider |
| Check colliders | A collider is blocked unless conditioned on (or its descendant is) | Forgetting that descendants of colliders also open paths |
| Conclude | All paths must be blocked for $d$-separation | Stopping after finding one blocked path |

### Worked Example

Consider this five-node DAG:

```
  X ──> M ──> Y
  |           ^
  v           |
  A ──> B ────┘
```

Question: Is $X \perp_d Y \mid \{A\}$?

**Path 1**: $X \rightarrow M \rightarrow Y$
- $M$ is a chain node. $M \notin \{A\}$. Not blocked here. Path is **open**.

Since we found an open path, $X$ and $Y$ are $d$-connected given $\{A\}$. We do not need to check further paths.

Question: Is $X \perp_d Y \mid \{M, B\}$?

**Path 1**: $X \rightarrow M \rightarrow Y$
- $M$ is a chain node. $M \in \{M, B\}$. **Blocked**.

**Path 2**: $X \rightarrow A \rightarrow B \rightarrow Y$
- $A$ is a chain node. $A \notin \{M, B\}$. Check next.
- $B$ is a chain node. $B \in \{M, B\}$. **Blocked**.

**Path 3**: $X \rightarrow M \rightarrow Y \leftarrow B \leftarrow A \leftarrow X$ — this revisits $X$, so it is not a valid path (paths do not revisit nodes).

All valid paths are blocked. Conclusion: $X \perp_d Y \mid \{M, B\}$.

We now have a mechanical procedure for reading conditional independencies from any DAG. The next section shows why this matters: it lets us distinguish what we can *observe* from what we can *intervene on*.

---

## 4. The $\mathrm{do}$-Operator and Graph Surgery

### Observation vs. Intervention

The central insight of Pearl's framework is the distinction between *seeing* and *doing*:

$$P(Y \mid \mathrm{do}(W = w)) \neq P(Y \mid W = w) \quad \text{in general}$$

- $P(Y \mid W = w)$ is the **observational conditional**: the distribution of $Y$ among units where $W$ *happened* to equal $w$.
- $P(Y \mid \mathrm{do}(W = w))$ is the **interventional distribution**: the distribution of $Y$ if we *set* $W$ to $w$ by external action, overriding its natural causes.

**The entire causal inference enterprise reduces to computing $P(Y \mid \mathrm{do}(W = w))$ from observational data $P(\mathbf{V})$ and the causal graph.**

### Graph Surgery

The $\mathrm{do}$-operator has a precise graphical definition:

1. Start with the original DAG $\mathcal{G}$.
2. **Delete all arrows into $W$** — sever $W$ from its parents.
3. **Set $W = w$** — $W$ is now a constant, not a random variable.
4. The result is the *mutilated graph* $\mathcal{G}_{\overline{W}}$.

This models a *perfect intervention*: an external force (a randomized experiment, a policy mandate) sets $W$ to $w$ regardless of the factors that normally determine $W$.

### Original vs. Mutilated DAG

Returning to our drug example:

```
Original DAG:                      Mutilated DAG (do(W = w)):

      A                                  A
     / \                                  \
    v   v                                  v
    W ──> Y                          W=w ──> Y
```

In the original graph, $A$ confounds $W$ and $Y$ — both $A \rightarrow W$ and $A \rightarrow Y$ are present, creating a backdoor path. In the mutilated graph, the arrow $A \rightarrow W$ is deleted. $W$ is no longer influenced by $A$. The only remaining path from $W$ to $Y$ is the direct causal arrow $W \rightarrow Y$.

### Numeric Example

Suppose all variables are binary. The SCM is:

$$A := U_A, \quad U_A \sim \text{Bernoulli}(0.5)$$

$$W := A, \quad \text{(deterministic: older patients always get the drug)}$$

$$Y := (1 - W) \cdot A + W \cdot U_Y, \quad U_Y \sim \text{Bernoulli}(0.3)$$

Let us compute $P(Y = 1 \mid W = 1)$ and $P(Y = 1 \mid \mathrm{do}(W = 1))$.

**Observational**: $W = 1$ implies $A = 1$ (deterministic). So:

$$P(Y = 1 \mid W = 1) = P(U_Y = 1) = 0.3$$

**Interventional**: Under $\mathrm{do}(W = 1)$, we break the link $A \rightarrow W$. Now $A$ still follows $\text{Bernoulli}(0.5)$, but $W = 1$ regardless. So:

$$P(Y = 1 \mid \mathrm{do}(W = 1)) = P(W \cdot U_Y = 1) = P(U_Y = 1) = 0.3$$

Now compare with the untreated case:

$$P(Y = 1 \mid W = 0) = P((1 - 0) \cdot A = 1 \mid W = 0) = P(A = 1 \mid A = 0) = 0$$

$$P(Y = 1 \mid \mathrm{do}(W = 0)) = P((1 - 0) \cdot A = 1) = P(A = 1) = 0.5$$

| Quantity | $W = 1$ | $W = 0$ | Difference |
|----------|---------|---------|------------|
| $P(Y=1 \mid W=w)$ (observational) | 0.3 | 0.0 | 0.3 |
| $P(Y=1 \mid \mathrm{do}(W=w))$ (interventional) | 0.3 | 0.5 | $-0.2$ |

**The observational comparison suggests the drug helps (difference = 0.3). The interventional comparison reveals the drug hurts (difference = $-0.2$).** The discrepancy arises because $A$ (age) confounds the observational comparison: older patients both receive the drug and have different baseline outcomes. Graph surgery eliminates this confounding.

---

## 5. Backdoor and Frontdoor Criteria

The $\mathrm{do}$-operator defines what we want. The identification criteria tell us when we can compute it from data.

### Backdoor Criterion

> **Theorem 2.1 (Backdoor Criterion).** A set of variables $\mathbf{Z}$ satisfies the backdoor criterion relative to an ordered pair $(W, Y)$ in a DAG $\mathcal{G}$ if:
>
> (a) No node in $\mathbf{Z}$ is a descendant of $W$.
>
> (b) $\mathbf{Z}$ blocks every path between $W$ and $Y$ that contains an arrow *into* $W$ (i.e., every "backdoor path").
>
> **Then the causal effect is identified by the backdoor adjustment formula:**
>
> $$P(Y \mid \mathrm{do}(W = w)) = \sum_{\mathbf{z}} P(Y \mid W = w, \mathbf{Z} = \mathbf{z}) \cdot P(\mathbf{Z} = \mathbf{z})$$

In words: adjust for $\mathbf{Z}$, and the conditional association equals the causal effect. The right-hand side uses only observational quantities.

**Backdoor adjustment is the graphical justification for "control for confounders."** When a researcher says "I controlled for age, sex, and income," they are implicitly claiming that $\lbrace \text{age, sex, income} \rbrace$ satisfies the backdoor criterion in the true (undrawn) causal graph.

### Connection to Potential Outcomes

The backdoor criterion provides the *graphical* conditions under which the *statistical* assumption of ignorability (Assumption 1.2 from [Part 1](/posts/causal-inference-part1-potential-outcomes/)) holds:

$$\text{Backdoor criterion for } \mathbf{Z} \quad \implies \quad Y(0), Y(1) \perp\!\!\!\perp W \mid \mathbf{Z} \quad \text{(ignorability)}$$

This is the bridge between the two frameworks. The DAG tells you *which* variables to condition on. Ignorability tells you *what follows* statistically. The two are complementary, not competing.

### Backdoor Example

```
    A ──> W ──> Y
    |           ^
    v           |
    B ──────────┘
```

Backdoor paths from $W$ to $Y$: $W \leftarrow A \rightarrow B \rightarrow Y$.

- Does $\{A\}$ satisfy backdoor? Check: (a) $A$ is not a descendant of $W$. (b) $A$ blocks $W \leftarrow A \rightarrow B \rightarrow Y$ (fork at $A$). **Yes.**
- Does $\{B\}$ satisfy backdoor? Check: (a) $B$ is not a descendant of $W$. (b) $B$ blocks $W \leftarrow A \rightarrow B \rightarrow Y$ (chain at $B$). **Yes.**
- Does $\{A, B\}$ satisfy backdoor? Check: (a) Neither is a descendant of $W$. (b) Both conditions hold. **Yes** (over-adjustment is valid but may reduce efficiency).
- Does $\{\}$ (empty set) satisfy backdoor? Check: (b) The path $W \leftarrow A \rightarrow B \rightarrow Y$ is unblocked. **No.**

### Frontdoor Criterion

What if the confounder is unobserved? The backdoor criterion fails because we cannot condition on a variable we do not measure. The frontdoor criterion provides an alternative identification strategy when a mediator is available.

Consider this DAG:

```
    U (unobserved)
   / \
  v   v
  W ──> M ──> Y
```

$U$ confounds $W$ and $Y$, so the backdoor criterion cannot be satisfied — we cannot condition on $U$. But $M$ mediates the entire effect of $W$ on $Y$.

> **Theorem 2.2 (Frontdoor Criterion).** A set of variables $\mathbf{M}$ satisfies the frontdoor criterion relative to $(W, Y)$ if:
>
> (a) $W$ blocks all backdoor paths from $W$ to $\mathbf{M}$ (i.e., $\mathbf{M}$ has no unblocked non-causal path from $W$).
>
> (b) $\mathbf{M}$ intercepts all directed paths from $W$ to $Y$.
>
> (c) All backdoor paths from $\mathbf{M}$ to $Y$ are blocked by $W$.
>
> **Then the causal effect is identified by the frontdoor formula:**
>
> $$P(Y \mid \mathrm{do}(W = w)) = \sum_m P(M = m \mid W = w) \sum_{w'} P(Y \mid W = w', M = m) \, P(W = w')$$

The formula works in two stages:

1. **$W \rightarrow M$**: Because $U$ does not directly affect $M$, the effect of $W$ on $M$ is unconfounded. We estimate $P(M \mid W)$ directly.
2. **$M \rightarrow Y$**: The path from $M$ to $Y$ is confounded by $U$ (through $W$), but conditioning on $W$ blocks this backdoor path. We estimate $P(Y \mid M, W)$ and marginalize over $W$.

### When Is Frontdoor Useful?

The frontdoor criterion applies in a specific but important scenario:

| Condition | Backdoor | Frontdoor |
|-----------|----------|-----------|
| Confounder observed? | Yes | No |
| Mediator available? | Not required | Yes, and it must capture *all* directed paths |
| Key requirement | $\mathbf{Z}$ blocks backdoor paths, not a descendant of $W$ | $W$ blocks backdoor to $M$; $W$ blocks backdoor from $M$ to $Y$ |
| Practical frequency | Common | Rare (strong assumptions on mediator) |

**The frontdoor criterion is rarely satisfied in practice, but it demonstrates a powerful principle: even with unobserved confounders, causal effects can sometimes be identified by exploiting the graph structure.**

A classic example: estimating the effect of smoking ($W$) on lung cancer ($Y$). The tobacco industry argued that an unobserved genetic factor ($U$) might cause both the desire to smoke and cancer susceptibility. The frontdoor criterion offers a path forward: if tar deposits in the lungs ($M$) mediate the entire effect and satisfy the three conditions, the causal effect is identified without measuring the genetic factor.

### A Larger Identification Example

```
        U (unobserved)
       / \
      v   v
  X ──> W ──> M ──> Y
              ^
              |
              Z
```

Can we identify the effect of $W$ on $Y$?

- **Backdoor from $W$ to $Y$**: $W \leftarrow U \rightarrow Y$? No direct arrow $U \rightarrow Y$ — but $U \rightarrow W$ and $U$ is unobserved. Check the paths: $W \leftarrow U \rightarrow Y$ is not present (no $U \rightarrow Y$ edge). The only confounding path would run through $U$, but $U$ connects to $W$ and back to $Y$ only through $M$. Actually, re-read the DAG: $U$ has arrows to both $W$ and... let us be precise. $U$ points to $X$ and $W$? No — $U$ points to $W$ and $Y$.

Let me redraw for clarity:

```
        U (unobserved)
       / \
      v   v
      W   Y
      |   ^
      v   |
      M ──┘
      ^
      |
      Z
```

Now: $U$ confounds $W$ and $Y$. The causal path is $W \rightarrow M \rightarrow Y$. $Z$ is a parent of $M$.

- **Frontdoor via $M$?** Check condition (a): backdoor paths from $W$ to $M$ — is there a path $W \leftarrow U \rightarrow Y \leftarrow M$? That requires going through the collider $Y$, which is blocked by default. So condition (a) holds. Check condition (b): all directed paths from $W$ to $Y$ go through $M$. Yes. Check condition (c): backdoor from $M$ to $Y$ — path $M \leftarrow Z$... no path from $Z$ to $Y$ except through $M$. Path $M \leftarrow W \leftarrow U \rightarrow Y$: conditioning on $W$ blocks this. **Frontdoor is satisfied.**

---

## 6. Connecting the Two Frameworks

### PO and SCM Are Complementary

**The potential outcomes framework and the SCM framework are not rival theories — they are complementary tools that answer different parts of the causal inference problem.**

- **Potential outcomes** define *what* we want: $\tau = E[Y_i(1) - Y_i(0)]$, $\tau(\mathbf{x})$, $\tau_{\text{ATT}}$.
- **SCMs** determine *how* to get it: which variables to adjust for, whether identification is possible, what assumptions are needed.

In practice, the two frameworks merge into a single workflow:

1. **Draw the DAG** (SCM). Encode domain knowledge as arrows and missing arrows.
2. **Read the adjustment set** ($d$-separation, backdoor/frontdoor). The graph tells you which covariates to condition on.
3. **Define the estimand** (PO). Write the causal effect in potential outcomes notation: $\tau = E[Y(1) - Y(0)]$.
4. **Estimate** (statistical methods). Use matching, IPW, doubly robust estimation, or machine learning.

### Comparison Table

| Axis | Potential Outcomes (Rubin) | Structural Causal Models (Pearl) |
|------|---------------------------|----------------------------------|
| **Primitive object** | Potential outcome $Y_i(w)$ | Structural equation $V := f(Pa_V, U_V)$ |
| **Causal effect** | $\tau = E[Y(1) - Y(0)]$ | $P(Y \mid \mathrm{do}(W=w))$ |
| **Identification tool** | Ignorability assumptions | $d$-separation, do-calculus |
| **Main strength** | Precise estimand definitions; close to experimental design; transparent assumptions | Encodes domain knowledge graphically; algorithmic identification; handles complex structures |
| **Main limitation** | Does not tell you *which* variables to condition on; graph-free | Requires specifying a complete DAG; strong qualitative commitments |

### Where Each Shines

**Use PO when** you have a clear treatment-control comparison and need to define estimands precisely — randomized trials, natural experiments, policy evaluations.

**Use SCM when** you have a complex system with many variables and need to determine which adjustment strategy is valid — observational studies with multiple potential confounders, mediation analysis, transportability of results across populations.

**Use both when** you face a real research problem. Draw the graph to identify the adjustment set. Write down the potential outcomes estimand. Estimate with your preferred method. Report assumptions from both perspectives.

### A Concrete Workflow

Suppose you want to estimate the effect of a new training protocol ($W$) on model accuracy ($Y$) across ML experiments, where team size ($A$) and compute budget ($B$) vary.

**Step 1 — Draw the DAG (SCM):**

```
  A ──> W ──> Y
  |           ^
  v           |
  B ──────────┘
```

Domain knowledge: larger teams adopt new protocols earlier ($A \rightarrow W$) and have bigger compute budgets ($A \rightarrow B$). Compute budgets directly affect accuracy ($B \rightarrow Y$). The protocol itself affects accuracy ($W \rightarrow Y$).

**Step 2 — Read the adjustment set:** The backdoor path is $W \leftarrow A \rightarrow B \rightarrow Y$. Either $\{A\}$ or $\{B\}$ or $\{A, B\}$ satisfies the backdoor criterion. Choose $\{A, B\}$ for robustness.

**Step 3 — Define the estimand (PO):** $\tau = E[Y_i(1) - Y_i(0)]$, identified by:

$$\tau = \sum_{a, b} \big[E[Y \mid W=1, A=a, B=b] - E[Y \mid W=0, A=a, B=b]\big] \cdot P(A=a, B=b)$$

**Step 4 — Estimate:** Apply doubly robust estimation, IPW, or regression adjustment conditioning on $\{A, B\}$.

The graph guided the choice of adjustment set. The potential outcomes framework gave the estimand. The estimation method computes it. Each framework contributed where it is strongest.

### Embedding PO in SCM

Pearl showed that potential outcomes can be formally derived from an SCM. Given an SCM $\mathcal{M}$ with structural equation $Y := f_Y(W, \mathbf{X}, U_Y)$, the potential outcome under treatment $w$ is:

$$Y_i(w) = f_Y(w, \mathbf{X}_i, U_{Y_i})$$

This is the value $Y$ would take if we set $W = w$ by intervention, holding the structural equations and exogenous noise fixed. Every potential outcomes quantity — ATE, ATT, CATE — can be expressed as a functional of the SCM.

The converse is not true: an SCM contains more information than a set of potential outcomes. The graph encodes independence structure, mediating pathways, and instrumental variable relationships that potential outcomes notation leaves implicit.

---

## Key Takeaways

1. **An SCM = equations + graph + noise.** The structural equations $V := f(Pa_V, U_V)$ define the data-generating process. The DAG is the graph of these equations.

2. **Three structures determine information flow.** Forks and chains transmit association by default (block by conditioning). Colliders block association by default (open by conditioning).

3. **$d$-separation reads independence from the graph.** If every path between $X$ and $Y$ is blocked by $\mathbf{Z}$, then $X \perp\!\!\!\perp Y \mid \mathbf{Z}$ in any distribution faithful to the graph.

4. **The $\mathrm{do}$-operator models intervention by graph surgery.** Delete arrows into $W$, set $W = w$, and compute in the mutilated graph.

5. **Backdoor and frontdoor criteria provide sufficient conditions for identification.** Backdoor requires observed confounders. Frontdoor works with unobserved confounders if a clean mediator exists.

6. **PO and SCM are complementary.** The graph tells you *what to adjust for*. Potential outcomes tell you *what to estimate*. Use both.

---

## Looking Ahead

We now have two complementary languages for causal inference: potential outcomes ([Part 1](/posts/causal-inference-part1-potential-outcomes/)) and structural causal models (Part 2). Both define causal effects. Both specify assumptions. But neither tells us how to go from "identification is possible in principle" to "here is a concrete research design that works with my data."

**[Part 3 — Identification: From Design to Estimand](/posts/causal-inference-part3-identification/)** bridges this gap. We will survey the five canonical identification strategies: difference-in-differences, instrumental variables, regression discontinuity, synthetic control, and selection on observables. Each strategy makes a different assumption, applies in a different setting, and identifies a different estimand. After Part 3, you will be able to choose the right design for your research question and state its identifying assumption.

---

## References

- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.
- Pearl, J., Glymour, M., & Jewell, N. P. (2016). *Causal Inference in Statistics: A Primer*. Wiley.
- Pearl, J. (2018). *The Book of Why: The New Science of Cause and Effect*. Basic Books.
- Imbens, G. W. (2020). "Potential Outcome and Directed Acyclic Graph Approaches to Causality." *Journal of Machine Learning Research*, 21(210), 1-45.
- Richardson, T. S., & Robins, J. M. (2013). "Single World Intervention Graphs (SWIGs)." Technical Report, University of Washington.
