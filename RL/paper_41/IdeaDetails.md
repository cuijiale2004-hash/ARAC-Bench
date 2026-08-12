1. Research Background and Existing Pain Points
Reinforcement Learning with Verifiable Rewards (RLVR) has become an essential component for developing advanced reasoning skills in language models. However, contemporary studies have documented training plateaus after thousands of optimization steps, showing notable decreases in performance gains despite increased computational investment. This limitation stems from the sparse exploration patterns inherent in current RLVR practices, where models rely on limited rollouts that often miss critical reasoning paths and fail to provide systematic coverage of the solution space. Current RLVR approaches remain constrained by sparse exploration patterns during training, while models are expected to demonstrate sophisticated search behaviors only at inference time. Even recent advances in prolonged RL training have shown that performance plateaus after thousands of steps, with diminishing returns to allocating more compute to deeper training. This suggests that simply scaling the number of training steps, the primary axis explored in prior work, may not be sufficient to fully realize RLVR’s potential. Furthermore, existing methods typically treat structured search as an inference-only mechanism, leaving untapped potential to integrate systematic exploration into the training process itself. Direct rollouts from the policy model behave like blind sampling: they quickly collapse into high-probability but low-diversity regions, rarely reaching deeper reasoning paths, and fail to uncover the informative trajectories required to correct systematic reasoning errors.

2. Core Research Motivation and Scientific Questions
The core insight driving this research is to focus on training-time exploration as the driver of improved reasoning. While traditional RLVR relies on limited rollouts that may miss critical reasoning paths, structured search systematically expands the reasoning frontier during training through principled tree search. The fundamental motivation is to address the bottleneck of insufficient exploration, which leads to diminishing performance gains over prolonged training. The scientific question is how to effectively integrate Monte Carlo Tree Search (MCTS) directly into RLVR training to enable systematic exploration and fine-grained credit assignment across reasoning steps, thereby overcoming the performance plateau inherent in depth-first scaling of training steps.

3. Overall Core Idea and Design Philosophy
The overall core idea is to introduce DeepSearch, a framework that embeds Monte Carlo Tree Search (MCTS) directly into RLVR training, representing a fundamental shift from scaling training depth to scaling training breadth. By coupling structured search with verifiable rewards during training, DeepSearch enables models to learn not only from correct solutions but also from the systematic exploration process itself, providing richer supervision than outcome-based or direct rollout methods. The design philosophy advances three key objectives: (i) expanding reasoning coverage beyond what direct policy rollouts can achieve, (ii) providing fine-grained credit assignment to intermediate reasoning steps through tree-structured backpropagation, and (iii) maintaining computational efficiency through intelligent node selection and solution caching strategies. It emphasizes systematic exploration over prolonged training and algorithmic innovation over resource-driven scaling.

4. Core Innovation Points
1. Integration of MCTS into RLVR Training: Embedding structured search into the training loop to enable systematic exploration and fine-grained credit assignment across reasoning steps, bridging the gap between inference-time search and training-time learning.
2. Global Frontier Selection Strategy: Prioritizing the most promising nodes across the entire search tree globally, moving beyond traditional root-to-leaf UCT traversals that can be computationally wasteful and myopic.
3. Selection with Entropy-based Guidance: Systematically identifying confident incorrect reasoning paths (with lowest average entropy) for supervision, targeting areas where the model is most confident in its decisions and would benefit from additional training.
4. Adaptive Training Strategy with Replay Buffer: Progressively filtering challenging problems and caching verified solutions (solution caching) for efficiency, preventing catastrophic forgetting and avoiding redundant computation across training iterations.
5. Hybrid Selection Strategy: Combining traditional UCT-based local selection for sibling comparison with novel global frontier selection for next expansion, leveraging complementary strengths for computational efficiency and enhanced exploration coverage.
6. Tree-GRPO Training Objective: Combining q-value regularization with policy optimization to learn effectively from tree-structured reasoning traces, utilizing soft clipping to address q-value explosion and mean-only advantages normalization to mitigate miscalibration and uncontrolled length growth.

5. Overview of the Overall Technical Solution
Given a problem $x$ and a policy model $\pi_\theta$, the framework adopts a modified MCTS framework to build a search tree for incremental step-by-step solution exploration. It replaces traditional root-to-leaf selection with global frontier-based node selection. The root node represents the question $x$, and child nodes correspond to intermediate steps $s$ generated by $\pi_\theta$. A root-to-leaf path ending at a terminal node $s_{end}$ forms a trajectory $t = x \oplus s_1 \oplus s_2 \oplus \dots \oplus s_{end}$. Solution trajectories $T = \{t_1, t_2, \dots, t_n\}$ are extracted from the search tree $\mathcal{T}$. The MCTS iterations are conducted through three subsequent components: Expansion with Entropy-based Guidance, Heuristic Score Backup, and Hybrid Selection Strategy. Following the search, an adaptive training strategy with a replay buffer is employed to construct the training dataset iteratively, filtering hard problems and utilizing cached solutions. Finally, the policy is updated using the Tree-GRPO training objective, which applies q-value soft clipping and sequence-level advantage normalization.

6. Detailed Module Design
Module 1: Expansion with Entropy-based Guidance
In step $i$, the latest reasoning trajectory $o_i = x \oplus s_1 \oplus s_2 \oplus \dots \oplus s_{i-1}$ is collected as the current state (observation). Based on this state, the policy model $\pi_\theta(s_i|o_i)$ generates $n$ candidates for the next-step reasoning trail $\{s_{i,j}\}_{j=1}^n$. This expansion repeats until terminal nodes $s_{end} \in S_{end}$ are reached (arriving at final answers or hitting maximum depth $d_T$). During each expansion, newly generated terminal nodes $S^{(k)}_{end}$ at iteration $k$ are evaluated for correctness using a verification function $V$. Terminal nodes are partitioned into correct and incorrect/incomplete subsets. If no correct solutions are found ($S^{(k)}_{correct} = \emptyset$), an entropy-based selection identifies the most confident wrong rollout by selecting the terminal node with the lowest average entropy along its root-to-leaf trajectory. This strategy prioritizes incorrect reasoning sequences with low decision uncertainty, targeting areas where the model is most confident in its mistakes.

Module 2: Heuristic Score Backup
The selected trajectory for backpropagation $t^*$ is either a correct solution trajectory or the most confident negative trajectory. The iterative q-value update rule for nodes along the selected trajectory uses a depth decay function that assigns higher weights to nodes closer to the terminal node. The constrained update rule preserves the invariant that $q^{(m)}(s_i) \ge 0$ for all intermediate nodes leading to correct solutions, while allowing negative values only for nodes not observed on any correct trajectory. The rationale behind this asymmetric rule includes:
- Same-sign reinforcement: Nodes that frequently appear on correct trajectories accumulate stronger positive evidence, while nodes consistently involved in incorrect trajectories accumulate stronger negative evidence.
- Negative-to-positive transitions: If a node previously held a negative q-value but now lies on a correct trajectory, the accumulated negative value is discarded and overwritten with the decayed positive reward, reclassifying it as potentially valuable.
- Positive-to-negative suppression: If a node already has a positive q-value, new negative signals are not propagated to it, ensuring a node proven capable of contributing to a correct solution is not downgraded due to occasional failed attempts.
Guaranteed invariants: Any node that has ever appeared on a correct trajectory retains a non-negative q-value; only nodes that have never been observed on a correct trajectory are allowed to accumulate stable negative values.

Module 3: Hybrid Selection Strategy
Local Selection for Sibling Comparison: During expansion of a selected node, multiple candidate children are generated. The UCT algorithm is employed for local sibling comparison to ensure optimal decisions among sibling nodes sharing the same parent and context.
Global Frontier Selection for Next Expansion: After score backup, the most promising node across the entire search tree is identified for the next expansion round. A global view of all leaf nodes (frontier nodes) is maintained. The frontier set $F$ consists of nodes that have no children, are not terminal, and have depth less than $d_T$. For each frontier node, a frontier priority score is computed combining Quality Potential (parent's value mapped via tanh), Uncertainty Bonus (policy's entropy), and Depth Bonus (encouraging deeper exploration using square root of depth ratio). The node with the highest frontier score is selected for the next expansion. This hybrid design eliminates redundant root-to-leaf traversals, prevents the algorithm from getting trapped in locally promising but globally suboptimal subtrees, and leverages the policy's entropy to target regions expected to benefit from additional training supervision.

Module 4: Adaptive Training Strategy with Replay Buffer
Iterative Training with Progressive Filtering: The training process follows an iterative approach refining the training subset based on model performance. The initial hard subset is constructed using the base policy evaluated via direct rollouts with a filtering threshold $\delta$. After each training phase, the updated policy is re-evaluated on the current hard subset, applying threshold-based filtering to create the next iteration's training set. This concentrates computational resources on increasingly challenging problems.
Replay Buffer with Cached Solutions: To prevent catastrophic forgetting and efficiently leverage previously discovered solutions, a replay buffer $R$ stores correct reasoning trajectories. Candidates for the buffer are problems that obtained correct solutions through MCTS but still fail to meet the filtering threshold. A deterministic strategy utilizes cached solutions when available, eliminating redundant MCTS computation. A hybrid rollout strategy is applied: for problems with cached solutions, the stored correct trajectory is incorporated and supplemented with direct rollouts from the current policy; for problems without cached solutions, the complete MCTS search process is applied. Justification for the fixed threshold $\delta=25\%$ is that it is a simple yet empirically effective heuristic that filters out mastered problems while preserving challenging cases, prioritizing methodological simplicity.
Justification for Maximum Depth and Expansion Length: The maximum search depth is set by a rollout budget of 16,384 tokens and an expansion length of 256 tokens per node, yielding an upper bound of 64 expansion steps. This reflects a principled trade-off between modeling requirements and training efficiency. The 256-token expansion length provides the best balance between search-space exploration (longer expansions reduce tree breadth) and computational cost (shorter expansions increase prefix re-encoding costs).

Module 5: Tree-GRPO Training Objective
Q-Value Soft Clipping: To address the q-value explosion problem for intermediate nodes while preserving meaningful gradients, soft clipping is applied using the hyperbolic tangent function. This bounds q-values without hard discontinuities, preserves gradients everywhere, and maintains the relative ordering of q-values while compressing extreme outliers.
Training Objective: With regularized q-values, the Tree-GRPO objective is maximized. It uses the importance ratio and follows the Clip-Higher strategy of DAPO, removing the KL regularization term. An overlong buffer penalty is applied to responses exceeding 4096 tokens. The advantage function for a node is computed using sequence-level normalization (mean-only), where the advantage is the node's q-value minus the average reward of the terminal nodes throughout the tree. This normalization mitigates uncontrolled growth in response length and addresses miscalibration issues.

7. All Mathematical Formulas and Symbol Definitions
Symbol Definitions:
$x$: Problem.
$\pi_\theta$: Policy model.
$s_i$: Intermediate step generated by $\pi_\theta$.
$s_{end}$: Terminal node.
$t$: Trajectory, $t = x \oplus s_1 \oplus s_2 \oplus \dots \oplus s_{end}$.
$T$: Set of solution trajectories $\{t_1, t_2, \dots, t_n\}$.
$\mathcal{T}$: Search tree.
$d(s)$: Depth of node $s \in \mathbb{Z}^+$.
$N(s)$: Number of visits to $s$.
$\xi(s)$: Number of children nodes of $s$.
$o_i$: Current state/observation at step $i$, $o_i = x \oplus s_1 \oplus s_2 \oplus \dots \oplus s_{i-1}$.
$S^{(k)}_{end}$: Set of newly generated terminal nodes at iteration $k$.
$V$: Verification function, $V: S_{end} \to \{0, 1\}$.
$S^{(k)}_{correct}$: Correct terminal nodes at iteration $k$.
$S^{(k)}_{incorrect}$: Incorrect/incomplete terminal nodes at iteration $k$.
$s^*_{neg}$: Most confident wrong rollout (terminal node with lowest average entropy).
$H(\pi_\theta(s_i | o_i))$: Monte Carlo estimation of Shannon entropy of token distribution at step $i$.
$\bar{H}(t(s))$: Average trajectory entropy.
$t^*$: Selected trajectory for backpropagation.
$q^{(m)}(s_i)$: Q-value for node $s_i$ after the $m$-th rollout backpropagation.
$\gamma(i, l)$: Depth decay function, $\gamma: \mathbb{Z}^+ \times \mathbb{Z}^+ \to [0, 1]$.
$\gamma_{min}$: Minimum decay threshold (0.1).
$UCT(s)$: Upper Confidence Bounds for Trees score for node $s$.
$Q(s)$: Average reward per visit for node $s$, $Q(s) = q(s)/N(s)$.
$\lambda$: Balance parameter for exploitation and exploration in UCT.
$F$: Frontier set.
$\mathcal{F}(s)$: Frontier priority score.
$\lambda_1, \lambda_2, \lambda_3$: Coefficients for quality potential, uncertainty bonus, and depth bonus.
$D(d(s))$: Depth bonus function.
$D^{(i)}_{hard}$: Hard subset at iteration $i$.
$\delta^{(i)}$: Filtering threshold at iteration $i$.
$Pass1@K(x, \pi)$: Success rate when sampling $K$ solutions.
$R$: Replay buffer.
$R^{(i)}_{candidates}$: Candidate trajectories for replay buffer at iteration $i$.
$t_{cached}$: Cached correct trajectory.
$\beta$: Fraction of additional solution attempts from current policy.
$B$: Standard sampling budget.
$\mathcal{T}^{(i)}_{train}$: Training dataset for iteration $i$.
$q_{max}$: Maximum allowable q-value magnitude.
$\epsilon_q$: Temperature parameter for soft clipping.
$k_{max}$: Maximum rollout iterations.
$\rho_{j,k}(\theta)$: Importance ratio.
$\epsilon_{high}, \epsilon_{low}$: Clipping thresholds.
$A_{j,k}$: Advantage function for node $s_j$ in trajectory $t_i$.
$\mu_t$: Average reward of terminal nodes $S_{end}$ throughout tree $\mathcal{T}$.
$\sigma_t$: Standard deviation of rewards of terminal nodes.

Formulas:
(1) $S^{(k)}_{correct} = \{s \in S^{(k)}_{end} | V(s) = 1\}, S^{(k)}_{incorrect} = \{s \in S^{(k)}_{end} | V(s) = 0\}$
(2) $s^*_{neg} = \arg\min_{s \in S^{(k)}_{incorrect}} \bar{H}(t(s))$
where $\bar{H}(t(s)) = \frac{1}{|t(s)|} \sum_{i=1}^{|t(s)|} H(\pi_\theta(s_i | o_i))$, and $H(\pi_\theta(s_i | o_i)) = -\sum_{a_{i,k}} \pi_\theta(a_{i,k} | o_i, a_{i,<k}) \log \pi_\theta(a_{i,k} | o_i, a_{i,<k})$
(3) $q^{(m)}(s_i) = q^{(m-1)}(s_i) + \gamma(i, l) \cdot q^{(m)}(s_{end})$
where $\gamma(i, l) = \max(\frac{i}{l}, \gamma_{min})$
(4) $q(s_{end}) = \begin{cases} +1 & \text{if } V(s_{end}) = 1 \text{ (correct)} \\ -1 & \text{if } V(s_{end}) = 0 \text{ (incorrect)} \lor d(s_{end}) < d_T \text{ (incomplete)} \end{cases}$
(5) $q^{(m)}(s_i) = \begin{cases} q^{(m-1)}(s_i) + \gamma(i, l) \cdot q^{(m)}(s_{end}) & \text{if } q^{(m-1)}(s_i) \cdot q^{(m)}(s_{end}) \ge 0 \\ \gamma(i, l) \cdot q^{(m)}(s_{end}) & \text{elif } q^{(m)}(s_{end}) > 0 \\ q^{(m-1)}(s_i) & \text{elif } q^{(m-1)}(s_i) > 0 \end{cases}$
(6) $F = \{s \in \mathcal{T} | \xi(s) = 0, s \notin S_{end}, d(s) < d_T\}$
(7) $\mathcal{F}(s) = \underbrace{\lambda_1 \times \tanh(Q_{parent}(s))}_{\text{Quality Potential}} + \underbrace{\lambda_2 \times H(\pi_\theta(s | o_s))}_{\text{Uncertainty Bonus}} + \underbrace{\lambda_3 \times D(d(s))}_{\text{Depth Bonus}}$
where $D(d(s)) = \sqrt{d(s)/d_T}$
(8) $D^{(0)}_{hard} = \{x \in D_{train} | Pass1@K(x, \pi_{\theta^{(0)}}) < \delta^{(0)}\}$
(9) $D^{(i+1)}_{hard} = \{x \in D^{(i)}_{hard} | Pass1@K(x, \pi_{\theta^{(i)}}) < \delta^{(i)}\}$
(10) $R^{(i)}_{candidates} = \{(x, t_{correct}) | x \in D^{(i)}_{hard}, \exists t_{correct} \in T(x), Pass1@K(x, \pi_{\theta^{(i)}}) < \delta^{(i)}\}$
(11) $Rollout(x) = \begin{cases} t_{cached} \cup DirectRollouts(x, \beta) & \text{if } (x, t_{cached}) \in R^{(i)} \\ MCTS_{full}(x) & \text{otherwise} \end{cases}$
(12) $\mathcal{T}^{(i)}_{train} = \underbrace{\bigcup_{x:(x, t_{cached}) \in R^{(i)}} \{t_{cached} \cup DirectRollouts(x, \beta)\}}_{\text{Cached problems}} \cup \underbrace{\bigcup_{x:(x, t_{cached}) \notin R^{(i)}} MCTS_{full}(x)}_{\text{Unsolved problems}}$
(13) $q(s_j) = \tanh(q^{(k_{max})}(s_j)/\epsilon_q) \cdot q_{max}$ for all $s_j \in \mathcal{T} \setminus S_{end}$
(14) $J(\theta) = \mathbb{E}_{T \sim \mathcal{T}, t_i \sim T, (s_j, o_j) \sim t_i} \left[ \frac{1}{|s_j|} \sum_{k=1}^{|s_j|} \min(\rho_{j,k}(\theta)\hat{A}_{j,k}, \text{clip}(\rho_{j,k}(\theta), 1-\epsilon_{low}, 1+\epsilon_{high})\hat{A}_{j,k}) \right]$
where $\rho_{j,k}(\theta) = \frac{\pi_\theta(a_{j,k}|o_j, a_{j,<k})}{\pi_{\theta_{old}}(a_{j,k}|o_j, a_{j,<k})}$
(15) $\hat{A}_{j,k} = q(s_j) - \mu_t$

8. Algorithm Pseudocode
Algorithm 1 DeepSearch with Global Frontier Selection and Iterative Filtering
Require: Initial policy $\pi_{\theta^{(0)}}$, training set $D_{train}$, verifier $V$, filtering threshold $\delta$
1: Initialize $D^{(0)}_{hard} \leftarrow \{x \in D_{train} | Pass1@K(x, \pi_{\theta^{(0)}}) < \delta^{(0)}\}, R^{(0)} = \emptyset$
2: for training iteration $i = 0, 1, 2, \dots$ do
3: Initialize training trajectories $\mathcal{T}^{(i)}_{train} \leftarrow \emptyset$
4: for each batch $B^{(i)} \in D^{(i)}_{hard}$ do
5: for each problem $x \in B^{(i)}$ do
6: if $(x, t_{cached}) \in R^{(i)}$ then $\triangleright$ Use cached solution
7: $T_x \leftarrow \{t_{cached}\} \cup DirectRollouts(x, \beta)$
8: $\mathcal{T}^{(i)}_{train} \leftarrow \mathcal{T}^{(i)}_{train} \cup T_x$
9: else $\triangleright$ Apply full MCTS search
10: MCTS Search:
11: Initialize search tree $\mathcal{T}$ with root node $x$
12: for rollout iteration $k = 1, 2, \dots$ do
13: if $k = 1$ then $\triangleright$ Initial expansion from root
14: Select root node $s^* = x$ for expansion
15: else
16: Global Frontier Selection:
17: Compute frontier set $F = \{s \in \mathcal{T} | \xi(s) = 0, s \notin S_{end}, d(s) < d_T\}$
18: Compute frontier priority scores (Eq. 7)
19: Select node $s^* = \arg\max_{s \in F} \mathcal{F}(s)$ for expansion
20: end if
21: Local Expansion with UCT Selection:
22: Generate $n$ candidates $\{s_j\}_{j=1}^n \sim \pi_\theta(\cdot | o_{s^*})$ from $s^*$
23: Continue expansion until terminal nodes $S^{(k)}_{end}$ are reached
24: Evaluation with Entropy-based Guidance
25: Partition: $S^{(k)}_{correct} = \{s \in S^{(k)}_{end} | V(s) = 1\}, S^{(k)}_{incorrect} = \{s \in S^{(k)}_{end} | V(s) = 0\}$
26: if $|S^{(k)}_{correct}| \ge 1$ then
27: Extract trajectories $T(x)$ from search tree $\mathcal{T}$
28: $\mathcal{T}^{(i)}_{train} \leftarrow \mathcal{T}^{(i)}_{train} \cup T(x)$
29: else
30: Select most confident negative: $s^*_{neg} = \arg\min_{s \in S^{(k)}_{incorrect}} \bar{H}(t(s))$
31: end if
32: Heuristic Score Backup:
33: Select trajectory $t^*$ (correct solution or $t(s^*_{neg})$)
34: Assign terminal rewards (Eq. 4)
35: for each node $s_j$ in $t^*$ do
36: Update Q-values using constrained backup rule (Eq. 5)
37: end for
38: end for
39: end if
40: Replay Buffer Update:
41: if MCTS found correct solutions but $Pass1@K(x, \pi_{\theta^{(i)}}) < \delta^{(i)}$ then
42: Add $(x, t_{correct})$ to $R^{(i+1)}$ for any correct $t_{correct} \in T(x)$
43: end if
44: end for
45: Policy Update:
46: Update policy $\pi_{\theta^{(i+1)}}$ using Tree-GRPO objective on $\mathcal{T}^{(i)}_{train}$ (Eq. 13 and Eq. 14)
47: end for
48: Re-evaluate and filter: $D^{(i+1)}_{hard} = \{x \in D^{(i)}_{hard} | Pass1@K(x, \pi_{\theta^{(i+1)}}) < \delta^{(i+1)}\}$
49: end for