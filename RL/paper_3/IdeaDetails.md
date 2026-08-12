**1. Research Background and Existing Pain Points**

**Research Background:**
Advancing the complex reasoning capabilities of Large Language Models (LLMs) remains a significant challenge, particularly in domains like mathematical problem-solving. Reinforcement Learning with Verifiable Reward (RLVR), especially through policy optimization methods like GRPO, has emerged as a powerful paradigm. In this framework, the space of potential solution paths for a query can be modeled as a specific ‘Reasoning Tree’, where each node represents an intermediate reasoning step and each path represents a potential solution trajectory. From this perspective, RLVR operates as a dynamic ‘node-editing’ process of the reasoning tree: by rewarding correct paths and penalizing incorrect ones, the model iteratively refines its decision policy at each tree node, gradually pruning branches that lead to low-quality or incorrect solutions. Furthermore, data scheduling—originating from curriculum learning—plays a critical role in model performance, positing that models learn more effectively when training examples are organized in a meaningful sequence.

**Existing Pain Points:**
Existing data scheduling strategies for RLVR typically pre-define a ‘difficulty’ metric for queries and schedule them from easy to hard. However, from a reasoning tree perspective, current difficulty measure strategies exhibit a critical limitation: current methods estimate difficulty primarily via final solution accuracy (path-based metrics), overlooking richer query-level characteristics such as the structural complexity of the reasoning tree. Accuracy alone is insufficient—low accuracy does not necessarily indicate that a query is inherently hard, and high accuracy does not guarantee ease of optimization. For instance, a query with low initial accuracy but a simple tree structure may require modifying only a few key decision nodes to yield substantial accuracy gains (high learning efficiency), whereas a query with higher initial accuracy but a fragmented tree structure where correct paths are scattered across disparate subtrees requires more extensive edits across numerous tree nodes, resulting in higher training difficulty and lower learning efficiency. Path-based metrics will misinterpret the first query’s low accuracy as high difficulty, assigning it a lower training weight, while incorrectly prioritizing the more difficult second query. This inconsistency undermines the efficacy of accuracy-based scheduling approaches.

**2. Core Research Motivation and Scientific Questions**

**Core Research Motivation:**
The central motivation is to address the discrepancy between path-based metrics and true learning difficulty. As demonstrated by the discrepancy between "Potential Samples" (low accuracy but high learning efficiency) and "Stagnant Samples" (high accuracy but low learning efficiency), path-based metrics like accuracy are biased measurements for learning difficulty. This motivates the need to move beyond path-based metrics to directly quantify a query’s true learning potential and difficulty from the topological structure of its reasoning tree.

**Scientific Questions:**
How can we move beyond path-based metrics to directly quantify a query’s true learning difficulty from its reasoning-tree structure?

**3. Overall Core Idea and Design Philosophy**

**Overall Core Idea:**
The core idea is to formalize the reinforcement learning training process as an optimization problem under a finite ‘node editing budget’. Instead of measuring difficulty by current accuracy, we quantify a query’s learning potential based on its reasoning tree structure. A novel metric, the Reasoning Score (r-score), is introduced to measure the maximum potential accuracy gain achievable within a limited editing budget. A high r-score signifies a more tractable reasoning structure and greater learning efficiency, indicating that substantial accuracy improvements can be made by correcting just a few critical reasoning steps.

**Design Philosophy:**
Based on the r-score, the design philosophy proposes a curriculum learning schedule (Re-Schedule) that progresses from structurally simple (high r-score) to complex (low r-score) queries. The method consists of three main stages: First, an offline approximation of each query’s reasoning tree is constructed by sampling multiple solution trajectories. Second, the approximated reasoning tree is used to calculate each query’s reasoning score by simulating the node editing process under a budget. Finally, the r-score is integrated as a dynamic weight into the RLVR loss function to form a schedule. This schedule prioritizes high-scoring (simple) queries in the initial training phases to accelerate convergence, and gradually shifts the weighting to lower-scoring (difficult) queries to enhance generalization.

**4. Core Innovation Points**

1. **Introduction of the Reasoning Score (r-score):** A novel tree-based metric that measures a query’s learning efficiency rather than its path-based solution accuracy. It quantifies the maximum potential accuracy gain achievable under a limited node editing budget.
2. **Proposal of Reasoning Tree Schedule (Re-Schedule):** A data scheduling algorithm that uses the r-score to create an effective, easy-to-hard curriculum for RLVR, dynamically adjusting data prioritization during training.
3. **Structural Viewpoint of Learning Difficulty:** Conceptualizing the RLVR training process as a dynamic ‘node-editing’ process of the reasoning tree, and establishing that a structural understanding of the reasoning tree provides a more powerful and principled foundation for quantifying learnability than outcome-based proxies.
4. **Finite Budget Optimization Formulation:** Formulating the difficulty measurement as an optimization problem under a finite ‘node editing budget’, which directly evaluates structural learnability by finding the maximum achievable accuracy gain over the reasoning tree via combinations of node modifications.
5. **Dynamic Weighting Strategy:** A dynamic weighting framework that balances data diversity and data scheduling by adaptively assigning weights determined by both the training step and the query's r-score, mitigating catastrophic forgetting of underrepresented data distributions.

**5. Overview of the Overall Technical Solution**

The overall technical solution, Reasoning Tree Schedule (Re-Schedule), enhances reinforcement learning performance by creating a curriculum based on the r-score. It operates through three sequential steps:
1. **Tree Construction:** For each query, construct a manageable, fixed-structure k-ary approximation of the reasoning tree by sampling multiple solution paths from a base model using a periodic branching strategy.
2. **R-Score Calculation:** Analyze the tree’s structure to compute the r-score. This involves calculating the accuracy function for nodes and determining the maximum sum of r-scores from a set of non-conflicting nodes under a finite modification budget $M$.
3. **Dynamic Weighting:** Integrate the r-score as a dynamic weight into the RLVR loss function. Compute a monotonically varying function $\alpha(R(q), t)$ that down-weights high-scoring queries over time while up-weighting lower-scoring ones, and map this to a training weight $\omega$ to form the curriculum schedule.

**6. Detailed Module Design**

**Module 1: Tree Construction**
Since the entire reasoning tree is computationally intractable, a fixed-structure k-ary approximation is constructed for each query $q$. The structure of the tree $T$ is defined by a branching factor $k$, a maximum depth $d$, and a token interval $l$. The construction process begins at the root node (the query $q$) and proceeds via a periodic branching strategy during response generation. Specifically, a branch is triggered immediately at the beginning of the response and subsequently at every $l$-token interval. At each trigger, the current path splits into $k$ independent sub-paths that continue to generate in parallel. This recursive branching continues until the maximum depth $d$ is reached. To minimize computational overhead, the Key-Value (KV) Cache is used, as all sibling branches share the same prefix. The quality of any intermediate node $n_i$ is defined as the average accuracy of its leaf descendants.

**Module 2: R-Score Calculation**
The r-score quantifies the learning potential of a node or query by measuring the maximum achievable accuracy gain under a limited policy refining cost (a limited node editing budget). For any non-leaf node $n_i$, its r-score $R(n_i)$ is defined as the maximal accuracy gain achievable by selecting its single best child branch and pruning all others. The overall r-score for a query, $R(q)$, estimates the total accuracy gain achievable under a budget that limits modifications to a maximum of $M$ nodes. It is the maximum sum of r-scores from any set of $M$ non-conflicting nodes. Two nodes are considered conflicting if one is located in a subtree that is implicitly pruned by the optimal branch selection of the other. A higher $R(q)$ indicates a structurally simple and efficient-to-learn query.

**Module 3: Dynamic Weighting**
To strike a balance between data diversity and data scheduling, a weighted scheduling framework dynamically adjusts data prioritization. Queries are assigned adaptive weights determined by both training step $t$ and r-score $R$. In the early training stage, higher weights are assigned to samples with higher r-scores (lower learning difficulty) to stabilize RL. In the later training phase, weights are redistributed gradually towards lower-r-score samples (higher learning difficulty) to enhance generalization. The training weight $\omega$ is formulated through an intermediate variable $\alpha(R(q), t)$ and mapped using the reverse order of $\alpha$ in the dataset. The $\alpha(R(q), t)$ is a monotonically varying function that down-weights high-scoring (simple) queries over time while up-weighting lower-scoring (difficult) ones.

**7. All Mathematical Formulas and Symbol Definitions**

**GRPO Objective Function:**
$J (\theta) = \mathbb{E}_{q\sim D,\{o_i\}_{i=1}^G \sim \pi_{old}(\cdot|q)} \left[ \frac{1}{\sum_{i=1}^G |o_i|} \sum_{i=1}^G \sum_{t=1}^{|o_i|} \min (r_{i,t}A_{i,t}, \text{clip}(r_{i,t}, 1- \varepsilon, 1 + \varepsilon)A_{i,t}) \right]$
where $r_{i,t} = \frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{old}(o_{i,t}|q, o_{i,<t})}$ is the probability ratio.
$A_{i,t} = \frac{R_i - \text{mean}(\{R_k\}_{k=1}^G)}{\text{std}(\{R_k\}_{k=1}^G) + \delta}$
where $\delta$ is a small constant for numerical stability.

**Scheduled Objective Function:**
$J_{schedule}(\theta) = \mathbb{E}_{q\sim D,...} [\omega(q, t) \cdot (\text{original objective term for } q)]$

**Accuracy-based Weighting (Baseline):**
$\alpha(ACC(q), t) = (1- \gamma(t))ACC(q) + \gamma(t)(1- ACC(q))$
$\omega = \text{rank}(\alpha)\% \cdot \omega_{max} + (1 - \text{rank}(\alpha)\%) \cdot \omega_{min}$
Here, $\omega_{max}$ and $\omega_{min}$ are hyperparameters (e.g., $\omega_{max} = 0.8, \omega_{min} = 0.2$). $\text{rank}(\alpha)$ calculates the reverse order of $\alpha$ in the entire dataset. $\gamma(t)$ is a scheduling function: linear $\gamma(t) = t/T$ or sigmoid $\gamma(t) = \sigma\left(\left(\frac{t}{T} - 0.5\right)\right)$, $\sigma(x) = (1 + e^{-x})^{-1}$.

**Reasoning Tree Notation:**
$T = (N, E, R)$, where $N$ is the set of nodes, $E$ is the set of edges, and $R$ defines the parent-child relationships.
$N_{leaf} \subset N$: set of leaf nodes.
$C(n_i)$: set of immediate children of node $n_i$.
$L(n_i)$: set of leaf descendants of node $n_i$. If $n_i$ is a leaf, $L(n_i) = \{n_i\}$.

**Accuracy Function:**
$ACC(S) = \frac{\sum_{n_j \in S} \mathbb{I}(n_j \text{ is correct})}{|S|}$
where $S$ is a set of leaf nodes and $\mathbb{I}(\cdot)$ is the indicator function.

**Node R-Score Calculation:**
$R(n_i) = \max_{n_{child} \in C(n_i)} ACC\left[ N_{leaf} \setminus L(n_i) \cup L(n_{child}) \right] - ACC\left[ N_{leaf} \right]$

**Query R-Score Calculation:**
$R(q) = \max_{N^* \subseteq N, |N^*|=M} \sum_{n_i \in N^*} R(n_i)$
where $M$ is the maximum number of node modifications allowed by the budget. Two nodes are conflicting if one is located in a subtree that is implicitly pruned by the optimal branch selection of the other.

**Dynamic Weighting Function:**
$\alpha(R(q), t) = (1- \gamma(t))R(q) + \gamma(t)(1-R(q))$
$\omega = \text{rank}(\alpha)\% \cdot \omega_{max} + (1 - \text{rank}(\alpha)\%) \cdot \omega_{min}$

**8. Algorithm Pseudocode**

The paper does not provide explicit algorithm pseudocode blocks, but defines the logic flow of the Re-Schedule algorithm in Section 5, which is summarized as follows based on the textual description:

**Reasoning Tree Schedule (Re-Schedule) Algorithm Logic:**

1. **Input:** Query dataset $D$, base model $\pi_{old}$, branching factor $k$, maximum depth $d$, token interval $l$, budget $M$, hyperparameters $\omega_{min}, \omega_{max}, T$.
2. **Offline Tree Construction Phase:**
   FOR each query $q \in D$ DO:
     a. Initialize the root node with $q$.
     b. Generate responses using $\pi_{old}$ with a periodic branching strategy:
        At every interval $l$ tokens, split the current path into $k$ independent sub-paths.
        Continue until maximum depth $d$ is reached.
     c. Construct the approximated reasoning tree $T = (N, E, R)$ for query $q$.
     d. Determine the correctness of all leaf nodes $N_{leaf}$.
3. **R-Score Calculation Phase:**
   FOR each query $q \in D$ DO:
     a. Calculate the accuracy function $ACC(N_{leaf})$ for the tree.
     b. FOR each non-leaf node $n_i \in N \setminus N_{leaf}$ DO:
          Calculate $R(n_i) = \max_{n_{child} \in C(n_i)} ACC\left[ N_{leaf} \setminus L(n_i) \cup L(n_{child}) \right] - ACC\left[ N_{leaf} \right]$
     c. Calculate the query r-score:
          $R(q) = \max_{N^* \subseteq N, |N^*|=M} \sum_{n_i \in N^*} R(n_i)$ 
          (subject to no conflicts among nodes in $N^*$)
4. **RLVR Training Phase with Dynamic Weighting:**
   FOR epoch $t = 1$ to $T$ DO:
     a. FOR each query $q \in D$ DO:
          Compute scheduling parameter: $\alpha(R(q), t) = (1- \gamma(t))R(q) + \gamma(t)(1-R(q))$
        END FOR
     b. Compute $\text{rank}(\alpha)\%$ for all queries in $D$ (reverse order of $\alpha$).
     c. FOR each query $q \in D$ DO:
          Compute training weight: $\omega = \text{rank}(\alpha)\% \cdot \omega_{max} + (1 - \text{rank}(\alpha)\%) \cdot \omega_{min}$
        END FOR
     d. Optimize the scheduled objective:
          $J_{schedule}(\theta) = \mathbb{E}_{q\sim D,...} [\omega(q, t) \cdot (\text{original objective term for } q)]$
   END FOR