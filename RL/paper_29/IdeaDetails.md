## 1. Research Background and Existing Pain Points

Scaling the reasoning capabilities of Large Language Models (LLMs) has traditionally required massive, human-curated datasets for supervised fine-tuning (SFT). This reliance on costly data collection has motivated a shift toward test-time adaptation, where models adapt "on the fly" using only unlabeled problems and available compute resources. Within this paradigm, test-time reinforcement learning (TTRL) has emerged as a particularly promising approach, enabling models to improve their reasoning by leveraging self-generated feedback signals without the need for external supervision. 

Current TTRL approaches typically involve generating multiple solution candidates and selecting a "pseudo-label" for the model to learn from, often through self-consistency or majority voting. However, this mechanism suffers from a critical vulnerability: if the model exhibits a systematic reasoning flaw, it may produce a specific incorrect answer more frequently than the correct one. In such cases, majority voting not only fails to correct the error but can actively amplify it by selecting the flawed solution as a training target. Specifically, when $p(\text{Correct} | x) < p(\text{Wrong} | x)$, as we draw more samples, the chance that majority voting recovers the correct answer actually converges to zero—meaning that majority voting amplifies the wrong solution instead of correcting it. This "majority-vote trap" creates an "echo chamber" of errors. Furthermore, a straightforward extension to dual-view majority voting (selecting the answer with the highest combined probability sum across views) remains vulnerable; it fails when different wrong answers dominate each view, as their summed probabilities can easily overwhelm that of the correct answer.

## 2. Core Research Motivation and Scientific Questions

The core research motivation stems from a common human robustness heuristic: when confronted with uncertainty, people often check solutions across multiple perspectives or reformulate the problem to verify consistency. We hypothesize that the correct answer, even if not the most popular, should appear robustly across different, faithfully paraphrased versions of the question. Inspired by this, we argue that pseudo-labels should not be judged solely by popularity within a single question description, but rather by invariance across multiple views. The scientific question addressed is: How to generate robust reward signals and reliably escape the model's own "echo chamber" of errors in a fully self-contained, label-free test-time setting? To overcome the limitations of majority voting, there is a need for a TTRL method that can generate robust reward signals based on information-theoretic principles that filter out spurious solutions driven by biased reasoning.

## 3. Overall Core Idea and Design Philosophy

The overall core idea is a novel TTRL framework called Self-Harmony that adapts self-play to generate reliable pseudo-labels in a fully self-contained manner. The core intuition is that the correct answer should appear robustly across two questions that are semantically equivalent but stylistically distinct, as fragile or spurious reasoning paths are often disrupted by changes in phrasing. Self-Harmony operationalizes this by having a single model dynamically assume two cooperative roles: a Solver, which generates answers to the original problem, and a Reframer, which rephrases the problem to provide a diverse perspective. The design philosophy is grounded in the Infomax principle, encouraging representations that remain invariant across different "views" of the data. By employing the harmonic mean of answer frequencies across both views, the mechanism inherently rewards answers that appear consistently in both distributions while penalizing those that arise only in one, thereby filtering out solutions driven by biased reasoning.

## 4. Core Innovation Points

1.  **Novel TTRL Framework**: A novel test-time reinforcement learning framework, Self-Harmony, that adapts cooperative self-play to improve reasoning without labels or external models.
2.  **Robust Pseudo-label Selection Mechanism**: A robust pseudo-label generation strategy based on the harmonic mean of answer frequencies, which mitigates common failure modes of majority voting by favoring view-invariant solutions.
3.  **Theoretical Justification**: A theoretical proof showing that the Harmonic Mean Selector emerges as a principled second-order approximation of the View-Invariant Infomax objective, enforcing that selected answers are robust to surface-level transformations.
4.  **Practical Fused Action Mechanism**: A practical implementation that fuses the reframing and the second solving steps into a single, structured generative action, combined with a multiplicative reward structure that ensures only correct answers receive reward while shaping the signal to favor well-formed and meaningfully different reformulations.

## 5. Overview of the Overall Technical Solution

The overall technical solution follows a "solve → reframe → solve" conceptual sequence. Given an unlabeled test dataset and a pre-trained LLM, the model generates answers for the original query $x$. Then, the Reframer generates a semantically equivalent paraphrase $x'$. Finally, the Solver is invoked again to generate answers for $x'$. To create a practical and scalable algorithm, this flow is optimized by fusing the reframing and the second solving steps into a single, structured generative action using a system prompt. This reduces the process to two efficient model calls: one for the initial solution and one for the joint "reframe-and-solve" trajectory. A rule-based verifier assigns rewards to each response based on its correctness against a target pseudo-label $y^*$ obtained by the harmonic mean instead of majority voting. These rewards are aggregated into objective functions, which guide the optimization of the pre-trained LLM.

## 6. Detailed Module Design

### 6.1 Single Model, Dual Roles for View Generation
Self-Harmony uses a single model ($M_\theta$) that switches between two roles via prompting. 
*   **The Solver role $\pi_\theta$**: Generates answers to a query $x$.
*   **The Reframer role $\rho_\theta$**: Creates a semantically equivalent paraphrase, $x'$. The Reframer is tasked with creatively rephrasing the original problem before solving it, promoting diversity and robustness. The prompt explicitly instructs the model to first transform the problem using one of several strategies (Concretize, Generalize, Domain Shift, Add Noise, Reverse, Incremental Complexity, Focus on Constraints, POV Shift), then solve the transformed version. The transformation must preserve logical and numerical consistency with the original problem, while discouraging trivial paraphrasing. The policy $\pi_\theta$ and reframer $\rho_\theta$ share parameters.

### 6.2 Pseudo-Label Selection via Harmonic Mean Score (HMS)
Following the theoretical analysis, the pseudo-label $y^*$ is selected by identifying the candidate answer $a$ that maximizes the Harmonic Mean Score of its empirical support from the original $\hat{p}_0(a)$ and reframed $\hat{p}_1(a)$ views. This method inherently rewards answers that appear consistently in both distributions while aggressively penalizing those that are strong in one view but weak in another.

### 6.3 Policy Optimization with Fused Actions
The fused reframing-and-solving action requires a reward that jointly evaluates paraphrase quality and final-answer correctness. A standard additive reward is suboptimal, as it grants partial credit for a well-formed paraphrase even when the final answer is wrong. A multiplicative reward design is used where answer correctness acts as a success gate. A base reward is given only when the answer is correct and is modulated by two penalty terms:
*   **Format Penalty ($R^{penalty}_{format}$)**: For structural violations.
*   **Diversity Penalty ($R^{penalty}_{div}$)**: The Jensen-Shannon divergence between the original and reframed queries’ answer distributions. This incentivizes the model to generate creative and semantically rich reformulations that provide a more challenging and effective consistency check. Without a strong diversity signal, the model may produce trivial modifications that fail to test robustness.

## 7. All Mathematical Formulas and Symbol Definitions

### 7.1 Problem Formulation and Majority Voting
*   $x, x'$: Original query and reformulated query.
*   $\pi_\theta$: Pre-trained LLM policy.
*   $p_0(a)$: Probability that the policy $\pi_\theta$ generates answer $a$ for the original query $x$.
*   $p_1(a)$: Probability that the policy $\pi_\theta$ generates answer $a$ for a reformulated query $x'$.
*   $y^*$: Pseudo-label inferred from rollouts.
*   **Dual-view majority voting**: $y^* = \text{argmax}_a (p_0(a) + p_1(a))$

### 7.2 View-Invariant Infomax Objective
*   $C$: Correct label.
*   **Assumption 3.1 (View-Invariance Assumption)**: For any two semantically equivalent queries $x$ and $x'$, the correct label $C$ has an approximately constant probability mass: $p(A = C | x) = p(A = C | x')$ where $A$ is the random variable of answer $a$. In contrast, the probabilities of incorrect labels $A \neq C$ vary across different views.
*   $A$: Random variable for the model rollout candidate answer $a$.
*   $X \in \{x, x'\}$: The view variable.
*   $Z_a = \mathbb{I}\{A = a\}$: Indicator for whether the model’s answer is $a$ or not.
*   **View-Invariant Infomax Objective**:
    $$J_\lambda(a) = I(Z_a; A) - \lambda I(Z_a; X)$$
    where the hyperparameter $\lambda$ balances the trade-off between accuracy and view-invariance.

### 7.3 Theorem 3.2 and Harmonic Mean Selector
**Theorem 3.2 (Harmonic Mean Selector from Invariant Infomax)**. Assume the view-invariance condition (Assumption 3.1) holds. Suppose moreover that the following conditions are satisfied:
*   **A1. Non-degeneracy**. For every label $a$, $p_0(a) + p_1(a) < 1$.
*   **A2. Balanced–Confidence**. There exists a constant $\kappa \in (0, 1)$ such that for every maximiser $a^*$ of $J_\lambda(\cdot)$:
    $$|p_0(a^*) - p_1(a^*)| \leq \kappa [p_0(a^*) + p_1(a^*)]$$
*   **A3. Uniform View Prior**. The view variable $X \in \{x, x'\}$ is assumed to be sampled uniformly, i.e., $p(X = x) = p(X = x') = 1/2$.

Then, for the penalty weight $\lambda = 2$, the pseudo label that maximises the second-order approximation of the view-invariant Infomax objective $J_2(a) = I(Z_a; A) - 2I(Z_a; X)$ is obtained by the harmonic mean of the two view-probabilities:
$$y^* = \text{argmax}_a \frac{2 p_0(a) p_1(a)}{p_0(a) + p_1(a)} \in \text{argmax}_a J_2(a)$$

### 7.4 Proof of Theorem 3.2 (Derivation Steps)
Define: $\bar{p}(a) = \frac{1}{2}(p_0(a) + p_1(a))$, $\delta(a) = \frac{1}{2}(p_0(a) - p_1(a))$.
$I(Z_a; A) = H(Z_a) - H(Z_a | A) = H(Z_a) = h(\bar{p}(a))$, where $h(x) = -x \ln x - (1-x) \ln(1-x)$.
$I(Z_a; X) = \frac{1}{2} D_{KL}(\text{Bern}(p_0(a)) \| \text{Bern}(\bar{p}(a))) + \frac{1}{2} D_{KL}(\text{Bern}(p_1(a)) \| \text{Bern}(\bar{p}(a)))$.
Using Taylor expansion of $g(p) = D_{KL}(\text{Bern}(p) \| \text{Bern}(q)) \approx \frac{(p-q)^2}{2q(1-q)}$:
$I(Z_a; X) \approx \frac{\delta(a)^2}{2\bar{p}(a)(1-\bar{p}(a))}$.
$J_2(a) \approx h(\bar{p}(a)) - \frac{\delta(a)^2}{\bar{p}(a)(1-\bar{p}(a))}$.
Using Assumption 2 and expanding denominator $1/(1-\bar{p}(a))$, discarding third-order terms:
$J_2(a) \approx h(\bar{p}(a)) - \frac{\delta(a)^2}{\bar{p}(a)}$.
Since $h$ is strictly increasing on $(0, 1/2)$ (by A1), maximizing $J_2(a)$ is equivalent to maximizing $\bar{p}(a) - \frac{\delta(a)^2}{\bar{p}(a)}$.
Using identity: $\bar{p}(a) - \frac{\delta(a)^2}{\bar{p}(a)} = \frac{2 p_0(a) p_1(a)}{p_0(a) + p_1(a)}$, the theorem follows.

### 7.5 Generalization of Theorem: Tunable $\lambda$ and Generalized Means
$J_\lambda(a) \approx \bar{p}(a) - \lambda \frac{\delta(a)^2}{2\bar{p}(a)}$.
Generalized Mean with exponent $k$: $M_k(p_0, p_1) = (\frac{p_0^k + p_1^k}{2})^{1/k} \approx \bar{p}(a) + (k-1)\frac{\delta(a)^2}{2\bar{p}(a)}$.
Mapping: $-\lambda = k - 1 \implies k = 1 - \lambda$.
*   $\lambda = -1 \implies k = 2$ (Quadratic Mean / RMS): Risk-seeking.
*   $\lambda = 0 \implies k = 1$ (Arithmetic Mean): Standard Voting.
*   $\lambda = 1 \implies k = 0$ (Geometric Mean): Penalizes variance.
*   $\lambda = 2 \implies k = -1$ (Harmonic Mean): Doubles penalty compared to Geometric Mean.

### 7.6 Multiple Views Case and Generalized Harmonic Mean
**Theorem G.2 (Generalized Harmonic Mean Selector)**. Let there be $K$ views defined by a random variable $X$ uniformly distributed over $\{x_1, \dots, x_K\}$. Let $p_k(a) = p(A = a | X = x_k)$. Maximizing the second-order approximation of $J_2(a) = I(Z_a; A) - 2I(Z_a; X)$ recovers the Generalized Harmonic Mean of the $K$ probabilities:
$$y^* = \text{argmax}_a \left( \frac{1}{K} \sum_{k=1}^K \frac{1}{p_k(a)} \right)^{-1}$$

### 7.7 Non-Uniform Prior and Weighted Invariant Score
Let $p(X = x) = \pi$, $p(X = x') = 1 - \pi$.
$\bar{p}_\pi(a) = \pi p_0(a) + (1-\pi) p_1(a)$.
$J_2(a) \approx \bar{p}_\pi(a) - \frac{\pi(1-\pi)(p_0(a) - p_1(a))^2}{\bar{p}_\pi(a)}$.

### 7.8 Reward Functions
*   **Reward for the Initial Solving Action**: $R_{solve}(y) = \mathbb{I}[y = y^*]$
*   **Reward for the Fused Reframing-and-Solving Action**: 
    $$R_{fused}(y') = (1 - w_f R^{penalty}_{format}(y'))(1 - w_d R^{penalty}_{div}(y', y))\mathbb{I}[y' = y^*]$$
    where $w_f$ is format coefficient, $w_d$ is diversity coefficient, and $\mathbb{I}[y' = y^*]$ is 1 if the predicted answer is correct and 0 otherwise.

## 8. Algorithm Pseudocode

**Algorithm 1 Self-Harmony (Conceptual Flow)**
1: Input: Language Model $M_\theta$, Test Dataset $D$, Number of rollouts $N$
2: Define: Training Step $T$; Problem Solver $\pi_\theta$, Reframer $\rho_\theta$. They shared the parameters
3: Output: Adapted model parameters $\theta_T$
4: for $t = 1$ to $T$ do
5:  Sample minibatch $B \subset D$
6:  for query $x \in B$ do
7:      Generate $N$ extracted rollout answer $\{y\}$ via $y \sim \pi_\theta(\cdot | x)$ // Solver on $x$
8:      Generate $x' \sim \rho_\theta(\cdot | x)$ // Reframe question from x to x’
9:      Generate $\{y'\}$ via $y' \sim \pi_\theta(\cdot | x')$ // Solver on $x'$
10:     Calculate a (each answer candidate)’s frequency in $\{y_i\}$: $\hat{p}_0(a)$ and in $\{y'_i\}$: $\hat{p}_1(a)$
11:     Compute the pseudo-label $y^*$ via $HMS(\hat{p}_0(a), \hat{p}_1(a))$.
12:     Compute rewards: $R_{solve}, R'_{solve}$, and diversity reward $R_{reframe}$
13:     Compute $\nabla J(\theta) + \nabla J'(\theta)$ via rewards
14:     Update $\theta$ by ascending $\nabla J(\theta) + \nabla J'(\theta)$
15: end for
16: end for