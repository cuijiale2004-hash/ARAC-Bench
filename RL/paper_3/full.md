## ABSTRACT

Using Reinforcement Learning with Verifiable Rewards (RLVR) to optimize Large Language Models (LLMs) can be conceptualized as progressively editing a query’s ‘Reasoning Tree’. This process involves exploring nodes (tokens) and dynamically modifying the model’s policy at each node. When combined with data scheduling, this process yields further gains in data efficiency and accuracy. However, existing RLVR data scheduling methods typically rely on path-based metrics to rank queries, overlooking the reasoning tree structures of these queries. In this paper, we introduce a novel metric, namely Reasoning Score (r-score), which measures the query’s learning difficulty based on the structure of its reasoning tree. Based on the r-score, we propose the Reasoning Tree Schedule (Re-Schedule), a scheduling algorithm that constructs a curriculum progressing from structurally simple (high r-score) to complex (low r-score) queries. Experiments on six math-reasoning benchmarks show that Re-Schedule significantly improves average accuracy, achieving gains of up to 3.2%. These strong results validate our approach and demonstrate that a structural understanding of the reasoning tree provides a more powerful and principled foundation for RLVR data scheduling<sup>1</sup>.

## 1 INTRODUCTION

Advancing the complex reasoning capabilities of Large Language Models (LLMs) remains a significant challenge, particularly in domains like mathematical problem-solving. Reinforcement Learning with Verifiable Reward (RLVR) (Gao et al., 2024; DeepSeek-AI et al., 2025), especially through policy optimization methods like GRPO (Shao et al., 2024), has emerged as a powerful paradigm to address this challenge. As shown in Figure 1 (a), in this framework, the space of potential solution paths for a query can be modeled as a specific ‘Reasoning Tree’ (Wang et al., 2025e; Yang et al., 2025b), where each node represents an intermediate reasoning step and each path represents a potential solution trajectory. From this perspective, RLVR operates as a dynamic ‘node-editing’ process of the reasoning tree: by rewarding correct paths and penalizing incorrect ones, the model iteratively refines its decision policy at each tree node. This optimization process gradually prunes branches that lead to low-quality or incorrect solutions, thereby improving overall reasoning accuracy.

In this paradigm, data scheduling plays a critical role in model performance (Hu et al., 2025; Li et al., 2025; Yang et al., 2026; Ren et al.). The concept of data scheduling originates from curriculum learning (Bengio et al., 2009), which posits that models learn more effectively when training examples (queries) are organized in a meaningful sequence. Existing data scheduling strategies typically pre-define a‘difficulty’ metric for queries, and and schedule them from easy to hard to improve data efficiency and final performance (Xi et al., 2024; Chen et al., 2025b;a; Dai et al., 2025) However, from a reasoning tree perspective, current difficulty measure strategies exhibits a critical limitation: current methods estimate difficulty primarily via final solution accuracy, overlooking richer query-level characteristics such as the structural complexity of the reasoning tree. Accuracy alone is insufficient — low accuracy does not necessarily indicate that a query is inherently hard, and high accuracy does not guarantee ease of optimization. This inconsistency can undermine the efficacy of accuracy-based scheduling approaches. We illustrate this issue with the following examples.

![](images/3acb6223318620c8ed82a68419bab56a0751badb71c081e4b1aeeac679eda571.jpg)  
Figure 1: (a) A simple reasoning tree (q1) requires less node editing for performance improvement than a complex one (q2). (b) Consequently, q1 shows high training efficiency (steep learning curve) despite low initial accuracy, while q2’s complex structure leads to low efficiency. (c) Our method leverages this structural insight to significantly outperform baselines on various datasets.

To illustrate, consider two representative queries, q1 and q2, whose reasoning trees are shown in Figure 1(a). As depicted in Figure 1(b), LLMs may exhibit low initial accuracy on q1, due to the presence of many incorrect solution trajectories (reasoning paths). However, its simple tree structure means that modifying a few key decision nodes can yield substantial accuracy gains, indicating high learning efficiency despite the poor initial performance. In contrast, q2 achieves higher initial accuracy, with roughly half of its trajectories being correct, yet these correct paths are scattered across disparate subtrees. This fragmented structure requires more extensive edits across numerous tree nodes, typically resulting in higher training difficulty and lower learning efficiency. Critically, existing path-based metrics will misinterpret q1’s low accuracy as high difficulty, thus assigning it a lower training weight, while incorrectly prioritizing the more difficult q2. Such path-based metrics may lead to a less efficient training process. This motivates our central research question: How can we move beyond path-based metrics to directly quantify a query’s true learning difficulty from its reasoning-tree structure?

To address this question, we introduce the Reasoning Score (r-score), a novel metric that quantifies a query’s learning potential based on its reasoning tree structure. We formalize this by framing the reinforcement learning training process as an optimization problem under a finite ’node editing budget’, which we define as a fixed number of node editing operations. Consequently, a query’s r-score is its maximum potential accuracy gain achievable within this limited editing budget. This metric clearly explains the discrepancy in our example: q1, with its ‘concentrated’ error structure, yields a high r-score because a small budget (e.g., two edits) produces a massive accuracy gain (+75%). Conversely, q2’s ‘diffuse’ structure results in a low r-score, as the same budget only yields marginal improvement (+25%). Therefore, a higher r-score signifies a more tractable reasoning structure and greater learning efficiency, offering a more comprehensive assessment of difficulty than path-based metrics.

Building on the Reasoning Score, we propose the Reasoning Tree Schedule (Re-Schedule), a novel data scheduling algorithm designed to guide RLVR training more efficiently. Our method consists of three main stages. First, an offline approximation of each query’s reasoning tree is constructed by sampling multiple solution trajectories from a base model. Second, this approximated reasoning tree is used to calculate each query’s reasoning score by simulating the editing process. Finally, we integrate the r-score as a dynamic weight into the RLVR loss function to form a schedule. This schedule prioritizes high-scoring (simple) queries in the initial training phases to accelerate convergence on simple queries. As training progresses, the weighting gradually shifts to lowerscoring (difficult) queries, enabling the model to master more challenging problems.

In summary, the main contributions of this paper are:

• We introduce the Reasoning Score (r-score), a new tree-based metric that measures a query’s learning efficiency rather than its path-based solution accuracy.

• We propose Re-Schedule, a data scheduling algorithm that uses the r-score to create an effective, easy-to-hard curriculum for RLVR.

• As shown in Figure 1(c), we empirically demonstrate that our approach significantly improves average accuracy, achieving gains of up to 3.2%, on complex reasoning tasks.

## 2 RELATED WORK

## 2.1 REINFORCEMENT LEARNING WITH VERIFIABLE REWARDS IN LLMS

Reinforcement learning with verifiable reward (RLVR), where the reward is computed by a rulebased verification function, has been shown to be effective in improving the reasoning capabilities of LLMs (Gao et al., 2024; DeepSeek-AI et al., 2025; Kimi et al., 2025; Zeng et al., 2025; Wen et al., 2025; Song et al., 2025; Wei et al., 2025b;a;e;d; Ma et al.). Typically, RLVR frameworks assign a binary reward by comparing the model’s generated output against a ground-truth solution, indicating whether it is correct or incorrect. This reward design obviates the need for complex outcome-based or process-based reward models, offering a straightforward yet potent approach. Recent advancements in policy optimization algorithms, such as PPO and GRPO, have further refined this paradigm (Schulman et al., 2017; Kazemnejad et al., 2024; Yuan et al., 2025; Yue et al., 2025; Shao et al., 2024; Yu et al., 2025; Liu et al., 2025; Zhang et al., 2025a; Hu, 2025; Hao et al., 2025b; Wei et al., 2025c; Liu et al., 2026; Yang et al., 2025a). In contrast to these studies, which focus on algorithmic improvements, our work builds upon the standard GRPO framework with a primary focus on designing a more effective training data schedule.

## 2.2 DATA SCHEDULING ALGORITHM IN LLM REINFORCEMENT LEARNING

Various data scheduling strategies have been proposed to enhance the reasoning capabilities in LLM Reinforcement Learning. These can be broadly categorized into static selection and dynamic adjustment methods. Representative of static selection is LIMR (Li et al., 2025), which selected 1.4k examples from an 8.5k set for RLVR to match the performance of using the full set. In contrast, dynamic strategies make real-time adjustments during training. For instance, $R ^ { 3 }$ employs reverse curriculum reinforcement learning to simplify the model’s exploration space (Xi et al., 2024). LPPO (Chen et al., 2025b) utilize the gradient of accuracy to prioritize data, effectively treating learning difficulty as a derivative of performance. Similarly, Seed-GRPO (Chen et al., 2025a) employs semantic diversity (uncertainty) as a proxy for difficulty. Furthermore, DELT leverages training gradients to measure the quality and learnability of data (Dai et al., 2025), subsequently adjusting sample weights. However, these methods rely on outcome-based proxies (e.g., accuracy), effectively treating reasoning as a flat sequence (Zhao et al., 2025; Zhang et al., 2025b). They overlook the inherent tree-structured solution space of reasoning tasks. In contrast, our approach explicitly leverages this topological structure. By analyzing the Reasoning Tree, we directly quantify a query’s ‘structural learnability’, providing a more precise and principled measure of difficulty than performance statistics alone.

## 3 PRELIMINARIES

## 3.1 GROUP RELATIVE POLICY OPTIMIZATION

The objective of the GRPO algorithm is to optimize a policy $\pi _ { \theta }$ based on a group of generated responses (Shao et al., 2024; Yu et al., 2025). For a query $q$ from a dataset $\mathcal { D } _ { : }$ , the policy generates G responses $\{ o _ { i } \} _ { i = 1 } ^ { G }$ . The token-level objective function is formulated as:

$$
\mathcal {J} (\theta) = \mathbb {E} _ {q \sim \mathcal {D}, \{o _ {i} \} _ {i = 1} ^ {G} \sim \pi_ {\text {old}} (\cdot | q)} \left[ \frac {1}{\sum_ {i = 1} ^ {G} | o _ {i} |} \sum_ {i = 1} ^ {G} \sum_ {t = 1} ^ {| o _ {i} |} \min (r _ {i, t} A _ {i, t}, \operatorname{clip} (r _ {i, t}, 1 - \varepsilon , 1 + \varepsilon) A _ {i, t}) \right],\tag{1}
$$

where $\begin{array} { r } { r _ { i , t } = \frac { \pi _ { \theta } \left( o _ { i , t } | { q } , o _ { i , < t } \right) } { \pi _ { \mathrm { o l d } } \left( o _ { i , t } | { q } , o _ { i , < t } \right) } } \end{array}$ is the probability ratio of the token $o _ { i , t }$ between the current and old policies. The advantage term $A _ { i , t }$ is constant for all tokens within a single response and is calculated by normalizing the response’s reward $R _ { i }$ relative to the other responses in the group:

$$
A _ {i, t} = \frac {R _ {i} - \mathrm{mean} (\{R _ {k} \} _ {k = 1} ^ {G})}{\mathrm{std} (\{R _ {k} \} _ {k = 1} ^ {G}) + \delta}, \quad \forall t,\tag{2}
$$

where $\delta$ is a small constant for numerical stability.

Data scheduling algorithms can be formulated by introducing a weighting function $\omega ( \boldsymbol { q } , t )$ that modulates the contribution of each query $q \in \mathcal { D }$ and current epoch t to the overall objective. Specifically, the objective in Equation 1 is modified as follows:

$$
\mathcal {J} _ {\text { schedule }} (\theta) = \mathbb {E} _ {q \sim \mathcal {D}, \dots} [ \omega (q, t) \cdot (\text { original   objective   term   for } q) ].\tag{3}
$$

Note: In the equations above, we have abbreviated the full objective for clarity. For example, in an accuracy-based curriculum learning, the training weight ω is formulated as a function of the query’s accuracy $\operatorname { A C C } ( q )$ and current epoch t:

$$
\alpha (\mathbf {A C C} (q), t) = (1 - \gamma (t)) \mathbf {A C C} (q) + \gamma (t) (1 - \mathbf {A C C} (q)),\tag{4}
$$

$$
\omega = \operatorname{rank} (\alpha) \% \cdot \omega_ {\max} + (1 - \operatorname{rank} (\alpha) \%) \cdot \omega_ {\min}.\tag{5}
$$

Here, $\omega _ { \mathrm { m a x } }$ and $\omega _ { \mathrm { m i n } }$ are hyperparameters that define the maximum and minimum training weights $( \mathrm { e . g . , } \omega _ { \mathrm { m a x } } = 0 . 8 , \omega _ { \mathrm { m i n } } = 0 . 2 )$ ; And rank $\scriptstyle ( \alpha )$ means calculating the reverse order of α in the entire dataset. The term $\gamma ( t )$ is a scheduling function that progresses over time. Common choices for $\gamma ( t )$ include a linear mapping, $\gamma ( t ) = t / T$ , or a sigmoid function, $\begin{array} { r } { \gamma ( t ) = \sigma \big ( \big ( \frac { t } { T } - 0 . 5 \big ) \big ) , \sigma ( x ) = } \end{array}$ $( 1 + e ^ { - x } ) ^ { - 1 }$ , where $T$ is the total number of epochs.

## 3.2 REASONING TREE

For complex reasoning tasks, the process of generating a solution can be conceptualized as traversing a ‘Reasoning Tree’. In this context, the root of the tree is the initial prompt, and each node represents a partial solution or an intermediate reasoning step. The branches extending from a node correspond to the possible next tokens or thought segments that the LLM can generate.

Due to the combinatorial explosion of possible solution paths, the complete reasoning tree is typically computationally intractable. Therefore, analysis often relies on a structured approximation (e.g., a fixed-structure k-ary reasoning tree). Formally, an approximated reasoning tree is defined as a triplet $\boldsymbol { T } = ( \mathcal { N } , \mathcal { E } , \mathcal { R } )$ , where $\bar { \mathcal { N } }$ is the set of nodes, E is the set of edges, and R defines the parent-child relationships.

The components of the tree are described using the following notation: $\mathcal { N } _ { \mathrm { l e a f } } \subset \mathcal { N }$ is the set of leaf nodes; For a given node $n _ { i } \in \mathcal { N } , C ( n _ { i } )$ denotes the set of its immediate children and $\mathcal { L } ( n _ { i } )$ denotes the set of its leaf descendants. If $n _ { i }$ is a leaf node, then ${ \mathcal { L } } ( n _ { i } ) = \{ n _ { i } \}$ . Within this framework, each non-leaf node $n _ { i } \in \mathcal { N } \backslash \mathcal { N } _ { \mathrm { l e a f } }$ represents a partial reasoning path, while a complete path to a leaf node $n _ { j } \in \mathcal { N } _ { \mathrm { l e a f } }$ corresponds to a full solution trajectory.

From this perspective, the RLVR optimization process is a dynamic ‘node editing’ of this reasoning tree. By rewarding correct paths and penalizing incorrect ones, the policy optimization algorithm adjusts the token probabilities at each node, effectively strengthening the branches that lead to correct answers and weakening those that lead to errors. The structure of this tree—the distribution of correct and incorrect paths—is intrinsic to each problem sample and, as we will argue, is a key clue to its learning dynamics.

![](images/4bcf32a4ea939891efb5b5ca767e43727ebc2957f84e5a6d554effe35c367803.jpg)  
Figure 2: Accuracy Progression During Training. The solid line represents the average accuracy, and the shaded area indicates the range.

![](images/b3eca72d0ed9ec87e891608546bda90e54b70e38c17134a03be32676084329ee.jpg)  
Figure 3: Overview of the Reasoning Tree Schedule (Re-Schedule) Algorithm.(a) Tree Construction: For each query, an approximate reasoning tree is constructed by sampling multiple solution paths from a base model (Note: This figureis for illustrative purposes only; our experiments use a tree with a depth of 4 and a width of $4 , \mathrm { i . e . , } k = 4 , d = 4 . )$ . (b) R-Score Calculation: The tree’s structure is analyzed to compute the r-score, a metric quantifying the query’s learning potential. (c) Dynamic Weighting: The r-scores are used to dynamically weight each query during training, forming a curriculum that progresses from structurally simple (easy) to complex (hard) examples.

## 4 MOTIVATION

The premise of this work is that path-based metrics such as accuracy are poor indicators of a query’s true learning difficulty. To illustrate our point, we supplement the example from the introduction with an experiment. As shown in Figure 2, we selected two distinct sets of 100 queries each from the DAPA-Math-17K dataset, using the Qwen2.5-Math-7B model. The blue line represents ‘Stagnant Samples’—queries with high initial accuracy but complex reasoning structures (low r-score). Their flat learning curve indicates that despite high initial performance, they are difficult to improve further. In contrast, the red line represents ‘Potential Samples’—queries with low initial accuracy but simple tree structures (high r-score). Their steep learning curve demonstrates high learnability, where a small amount of training yields significant gains. This discrepancy highlights that pathbased metrics, like accuracy, are biased measurements for learning difficulty. This finding motivates us to design a new metric based on the structure of the reasoning tree.

## 5 METHOD

As illustrated in Figure 3, the Reasoning Tree Schedule (Re-Schedule) enhances reinforcement learning performance by creating a curriculum based on our novel metric, the Reasoning Score (r-score). The r-score quantifies a query’s learning difficulty a priori based on the structure of its reasoning tree. Next, we will introduce the specific implementation details.

## 5.1 TREE CONSTRUCTION

As the entire reasoning tree is computationally intractable, we construct a manageable, fixedstructure k-ary approximation for each query q. The structure of this tree, T, is defined by a branching factor k, a maximum depth d, and a token interval $l \left( \mathrm { e } . \mathrm { g } . , k = 4 , d = 4 , l = 2 0 0 \right)$ .

The construction process begins at the root node (the query q) and proceeds via a periodic branching strategy during response generation. Specifically, a branch is triggered immediately at the beginning of the response and subsequently at every l-token interval. As shown in Figure 3 (a), at each trigger, the current path splits into k independent sub-paths that continue to generate in parallel. This recursive branching process continues until a predefined maximum depth d is reached. To minimize computational overhead from this multi-path sampling, we use the Key-Value (KV) Cache, as all sibling branches share the same prefix.

In RLVR tasks, a solution’s quality is determined by the correctness of its final answer, which corresponds to a leaf node in our framework. Therefore, we define the quality of any intermediate node $n _ { i }$ as the average accuracy of its leaf descendants, $\mathcal { L } ( n _ { i } )$ . This is quantified using an accuracy function:

$$
\operatorname{ACC} (S) = \frac {\sum_ {n _ {j} \in S} \mathbb {I} (n _ {j} \text {   is   correct })}{| S |},\tag{6}
$$

where $S$ is a set of leaf nodes and $\mathbb { I } ( \cdot )$ is the indicator function. This allows us to assess quality at different levels: the quality of a reasoning segment via $\mathsf { A C C } ( \mathcal { L } ( n _ { i } ) )$ and the model’s aggregate performance on the query via $\mathbf { A C C } ( \mathcal { N } _ { \mathrm { l e a f } } )$ .

## 5.2 R-SCORE CALCULATION

The r-score quantifies the learning potential of a node or query by measuring the maximum achievable accuracy gain under a limited policy refining cost, like a limited node editing budget. Given this idea, for any non-leaf node $n _ { i }$ , we define its r-score, $R ( n _ { i } )$ , as the maximal accuracy gain achievable by selecting its single best child branch and pruning all others. This is formulated as:

$$
R (n _ {i}) = \max _ {n _ {\text { child }} \in \mathcal {C} (n _ {i})} \operatorname{ACC} \left[ \mathcal {N} _ {\text { leaf }} \setminus \mathcal {L} (n _ {i}) \cup \mathcal {L} (n _ {\text { child }}) \right] - \operatorname{ACC} \left[ \mathcal {N} _ {\text { leaf }} \right].\tag{7}
$$

The overall r-score for a query, $R ( q )$ , estimates the total accuracy gain achievable under a budget that limits modifications to a maximum of M nodes. It is the maximum sum of r-scores from any set of M non-conflicting nodes (e.g., for a budget of $M = 4 )$

$$
R (q) = \max _ {\mathcal {N} ^ {*} \subseteq \mathcal {N}, | \mathcal {N} ^ {*} | = M} \sum_ {n _ {i} \in \mathcal {N} ^ {*}} R (n _ {i}).\tag{8}
$$

Two nodes are considered conflicting if one is located in a subtree that is implicitly pruned by the optimal branch selection of the other.

Intuitively, solving Equation (7) represents the evaluation process of the sub-tree’s structure, while a simpler structure of reasoning tree starting from $n _ { i }$ yields a higher $R ( n _ { i } )$ . Combining the evaluation $R ( n _ { i } )$ of each node $n _ { i }$ under a limited budget $M ,$ , solving Equation (8) is to find the maximum achievable accuracy gain over the reasoning tree, like exploring possible combinations of M nodes and picking the best combination. Thus, a higher $R ( q )$ indicates that substantial accuracy improvements can be made by correcting just a few critical reasoning steps, signifying a structurally simple and efficient-to-learn query.

## 5.3 DYNAMIC WEIGHTING

To strike a balance between data diversity and data scheduling, we propose a weighted scheduling framework that dynamically adjusts data prioritization. Specifically, queries are assigned adaptive weights determined by both training step t and r-score R. Specifically, when it is an early training stage, higher weights are assigned to samples with higher r-scores (indicating lower learning difficulty), stabilizing the reinforcement learning. When RL training meets the later training phase, queries’ weights will be redistributed gradually towards lower-r-score samples (higher learning difficulty) to enhance model generalization.

Motivated by this, the training weight ω of each query is formulated as

$$
\alpha (R (q), t) = (1 - \gamma (t)) R (q) + \gamma (t) (1 - R (q)),\tag{9}
$$

$$
\omega = \operatorname{rank} (\alpha) \% \cdot \omega_ {\max} + (1 - \operatorname{rank} (\alpha) \%) \cdot \omega_ {\min},\tag{10}
$$

where t is the current epoch; $\omega _ { \mathrm { m a x } }$ and $\omega _ { \mathrm { m i n } }$ are hyperparameters that define the maximum and minimum training weights; And rank(α) means calculating the reverse order of α in the entire dataset;

![](images/3bf37408a8ec90f9a3c2d3cb49be70bf3d573175878750cb9f38bfc7471f1fc6.jpg)  
(a)

![](images/22528d4379459d1a24c8773d5f1cb17966875afb5a6249eec1b9dccb4845ced3.jpg)  
(b)

![](images/76c83b5f903975af439ebeffb455045fc213744e7cd34de983b15e7837fd56a8.jpg)  
(c)  
Figure 4: (a) The average MCN decreases over time, indicating successful tree optimization. (b) & (c) To compare metrics, we train models on the top 1/3 of data selected by each. The plots show the resulting (b) training accuracy and (c) test accuracy. The model used is Qwen2.5-Math-7B.

$\gamma ( t )$ can be either linear mapping $\begin{array} { r } { \gamma ( t ) = \frac { t } { T } } \end{array}$ or sigmoid $\begin{array} { r } { \gamma ( t ) = \sigma \left( \left( \frac { t } { T } - 0 . 5 \right) \right) } \end{array}$ . The $\alpha ( R ( q ) , t )$ is a monotonically varying function that down-weights high-scoring (simple) queries over time while up-weighting lower-scoring (difficult) ones. This scheduling approach balances exploitation of easily learnable patterns and exploration of challenging instances, mitigating catastrophic forgetting of underrepresented data distributions.

## 6 ANALYSIS

## 6.1 TRAINING AS REASONING TREE OPTIMIZATION

To empirically validate that the training process is optimizing reasoning trees, we conducted an experiment centered on a new metric: the Minimum Corrective Nodes (MCN). This metric is defined as the minimum number of node modifications required for the reasoning tree to achieve a specified target accuracy. A single node modification is counted as one token change; thus, a lower MCN signifies a well-structured reasoning tree. In our experiment, we tracked the MCN on the DAPA-Math-17K training set during the training of Qwen2.5-Math-7B, excluding queries where the base model’s accuracy was below 10%.

As shown in Figure 4(a), the average MCN across the training set exhibits a consistent downward trend as training progresses, regardless of the target accuracy. This result demonstrates that the reinforcement learning process effectively refines the model’s policy at critical decision nodes, thereby validating our central assumption that training is a process of reasoning tree optimization.

## 6.2 THE RELATIONSHIP BETWEEN R-SCORE AND LEARNING DIFFICULTY

In this experiment, we want to see which metric best identifies valuable queries for early-stage training. The process is as follows: First, we use each metric to select the top one-third of the data, creating several distinct subsets. Second, we train a separate model on each of these subsets for a single epoch. Finally, we evaluate the resulting models on both the training and test sets.

As shown in Figure 4(b), the subset selected by the ACC-based method initially shows higher average accuracy on the training set, as expected from its selection criteria. However, as training progresses, the model trained on the r-score-selected subset quickly surpasses it. This indicates that the r-score is more effective at identifying queries with low learning difficulty, rather than just initial accuracy.

The advantage of r-score is even more evident on the test set, as shown in Figure 4(c). Here, the model trained on the r-score-selected queries consistently outperforms both the ACC-based selection and a baseline with random query selection (GRPO). This confirms that the queries identified by the r-score provide the most effective learning signal, leading to better performance improvement and validating its capability in identifying the real difficulty of queries.

## 7 EXPERIMENT

## 7.1 RL TRAINING SETUPS

Training setting We conduct experiments on two different models, including Qwen2.5-Math-7B and Qwen2.5-7B. We adapt our training codebase from verl (Sheng et al., 2025) and follow the training recipe of standard GRPO. Our training data is DAPO-Math-17k (Yu et al., 2025), containing only math problems with integer ground-truth answers. Both the KL-divergence and entropy loss terms are removed in our experiments (Hao et al., 2025b). Generation batch size is set to 512. Training is performed with top-p value of 1.0 and temperature = 1.0.

Evulation We evaluated our models and baselines on six widely used mathematical reasoning benchmarks: AIME24, AIME25, AMC23 (Li et al., 2024), MATH-500 (Hendrycks et al., 2021), Minerva Math (Lewkowycz et al., 2022), and OlympiadBench (He et al., 2024). Validation is performed with a top-p value of 0.7 and temperature = 1.0 across all models and test sets. We use Math-Verify for training, validation, and final evaluation. We report avg@32 for all datasets. All results are presented as percentages.

Baselines For the throughout comparison, we compare our method against 7 baselines, including standard GRPO (Shao et al., 2024), SimpleRL-Zoo (Zeng et al., 2025), Eurus-PRIME(Cui et al., 2025), OPO (Hao et al., 2025a), ACC (curriculum learning based on accuracy, using sigmoid weighting), LPPO (Chen et al., 2025b), and Seed-GRPO (Chen et al., 2025a).

Our Methods Re-Schedule is implemented with two weighting schemes: ‘linear’ and ‘sigmoid’. Unless otherwise specified, the reasoning trees in our experiments are constructed with a branching factor of $k = 4 ,$ a maximum depth of $d = 4$ , and a token interval of $l = 2 0 0$ . The weighting schemes are defined as follows: 1. The ‘linear’ scheme uses $\gamma ( t ) = t / T ; 2 .$ The ‘sigmoid’ scheme uses $\begin{array} { r } { \gamma ( t ) = \sigma \left( \left( \frac { t } { T } - 0 . 5 \right) \right) } \end{array}$ . For both, we set the total number of epochs $T = 1 0$ . Details of the training setup can be found in the Appendix C.

## 7.2 MAIN EXPERIMENT

Table 1: Main benchmark results on Qwen2.5-Math-7B. All values are accuracies multiplied by 100. Best results are in bold.

<table><tr><td>Model</td><td>AIME24</td><td>AIME25</td><td>AMC23</td><td>MATH500</td><td>Minerva</td><td>Olympiad</td><td>Avg.</td></tr><tr><td>Qwen2.5-Math-7B</td><td>13.8</td><td>5.3</td><td>44.6</td><td>39.6</td><td>9.9</td><td>13.8</td><td>21.2</td></tr><tr><td colspan="8">Classical RLVR Methods</td></tr><tr><td>GRPO</td><td>28.0</td><td>14.3</td><td>66.2</td><td>78.6</td><td>37.5</td><td>40.9</td><td>44.3</td></tr><tr><td>SimpleRL-Zoo</td><td>30.8</td><td>14.2</td><td>65.4</td><td>79.2</td><td>37.1</td><td>40.8</td><td>44.6</td></tr><tr><td>Eurus-PRIME</td><td>20.9</td><td>13.0</td><td>65.2</td><td>79.8</td><td>37.5</td><td>40.6</td><td>42.8</td></tr><tr><td>OPO</td><td>32.2</td><td>13.4</td><td>71.5</td><td>82.2</td><td>38.6</td><td>41.0</td><td>46.5</td></tr><tr><td colspan="8">Scheduling Methods</td></tr><tr><td> $ACC_{sigmoid}$ </td><td>31.5</td><td>15.6</td><td>70.9</td><td>80.8</td><td>38.6</td><td>42.2</td><td>46.6</td></tr><tr><td>LPPO</td><td>32.8</td><td>14.9</td><td>63.3</td><td>79.2</td><td>39.0</td><td>40.6</td><td>45.0</td></tr><tr><td>Seed-GRPO</td><td>30.7</td><td>14.0</td><td>71.0</td><td>80.0</td><td>38.2</td><td>38.5</td><td>45.4</td></tr><tr><td colspan="8">Our Methods</td></tr><tr><td>Re-Schedule $linear$ </td><td>34.2</td><td>15.6</td><td>72.4</td><td>81.2</td><td>36.4</td><td>42.5</td><td>47.1</td></tr><tr><td>Re-Schedulesigmoid</td><td>35.2</td><td>16.0</td><td>72.3</td><td>82.2</td><td>42.3</td><td>44.4</td><td>48.5</td></tr></table>

As shown in Tables 1 and 2, our Re-Schedule method consistently sets a new state-of-the-art, achieving average scores of 48.5 on $\mathbf { Q } \mathbf { w e n } 2 . 5 \mathbf { - } \mathbf { M } \mathbf { a t h - } 7 \mathbf { B }$ and 44.5 on Qwen2.5-7B. It significantly outperforms both scheduling baselines like $\mathbf { A C C } _ { s i g m o i d }$ (by up to 3.2%) and classical RLVR methods like OPO/GRPO (by up to 3.8%). These results validate our central claim: that the reasoning tree’s structure, captured by our r-score, is a more effective way to measure the real learning difficulty of a query than path-based metrics like accuracy.

Table 2: Main benchmark results on Qwen2.5-7B. All values are accuracies multiplied by 100. Best results are in bold.

<table><tr><td>Model</td><td>AIME24</td><td>AIME25</td><td>AMC23</td><td>MATH500</td><td>Minerva</td><td>Olympiad</td><td>Avg.</td></tr><tr><td>Qwen2.5-7B</td><td>5.1</td><td>2.5</td><td>27.8</td><td>34.4</td><td>5.9</td><td>13.5</td><td>14.9</td></tr><tr><td colspan="8">Classical RLVR Methods</td></tr><tr><td>GRPO</td><td>15.6</td><td>8.8</td><td>62.5</td><td>78.2</td><td>38.6</td><td>40.4</td><td>40.7</td></tr><tr><td>SimpleRL-Zoo</td><td>17.0</td><td>9.6</td><td>64.7</td><td>76.6</td><td>31.6</td><td>40.3</td><td>40.0</td></tr><tr><td>OPO</td><td>16.6</td><td>8.4</td><td>64.6</td><td>74.6</td><td>31.6</td><td>40.3</td><td>39.4</td></tr><tr><td colspan="8">Scheduling Methods</td></tr><tr><td> $ACC_{sigmoid}$ </td><td>16.7</td><td>9.8</td><td>68.6</td><td>79.0</td><td>34.2</td><td>39.4</td><td>41.3</td></tr><tr><td>LPPO</td><td>15.8</td><td>9.4</td><td>64.0</td><td>76.8</td><td>35.3</td><td>36.7</td><td>39.7</td></tr><tr><td>Seed-GRPO</td><td>13.3</td><td>6.0</td><td>63.3</td><td>76.6</td><td>32.4</td><td>36.3</td><td>38.0</td></tr><tr><td colspan="8">Our Methods</td></tr><tr><td>Re-Schedulelinear</td><td>18.4</td><td>12.2</td><td>68.6</td><td>80.4</td><td>41.2</td><td>42.1</td><td>43.8</td></tr><tr><td>Re-Schedulesigmoid</td><td>18.2</td><td>14.0</td><td>69.2</td><td>81.0</td><td>41.5</td><td>43.3</td><td>44.5</td></tr></table>

## 7.3 ABLATION EXPERIMENT

Table 3: Ablation study on tree construction parameters. The default configuration (branching factor $k = 4 ,$ depth d = 4) achieves the best performance.

<table><tr><td>Branch k</td><td>Depth d</td><td>AIME24</td><td>AIME25</td><td>AMC23</td><td>MATH500</td><td>Minerva</td><td>Olympiad</td><td>Avg.</td></tr><tr><td>4</td><td>4</td><td>34.2</td><td>16.0</td><td>71.1</td><td>81.8</td><td>42.3</td><td>44.4</td><td>48.3</td></tr><tr><td>3</td><td>5</td><td>33.8</td><td>14.8</td><td>68.4</td><td>79.6</td><td>42.3</td><td>42.8</td><td>46.9</td></tr><tr><td>5</td><td>3</td><td>31.7</td><td>14.2</td><td>70.4</td><td>81.0</td><td>41.9</td><td>43.0</td><td>47.0</td></tr></table>

We investigate the impact of the reasoning tree’s structure by varying the branching factor k and maximum depth $d .$ The choice of these parameters determines the fidelity of the approximated reasoning tree. While larger values for k and d theoretically provide a more accurate approximation and thus a more effective r-score, they also introduce a significant computational overhead. As shown in Table 3, our default configuration of k = 4 and $d = 4$ yields the best average performance (48.3%). For more detailed analysis, please see Appendices D.3 and D.4.

Table 4: Ablation study on the weight function hyperparameters, $\omega _ { \mathrm { m i n } }$ and $\omega _ { \mathrm { m a x } }$ . The default setting (0.5, 2.0) performs best.

<table><tr><td> $\omega_{\min}$ </td><td> $\omega_{\max}$ </td><td>AIME24</td><td>AIME25</td><td>AMC23</td><td>MATH500</td><td>Minerva</td><td>Olympiad</td><td>Avg.</td></tr><tr><td>0.5</td><td>2.0</td><td>35.2</td><td>16.0</td><td>72.3</td><td>82.2</td><td>42.3</td><td>44.4</td><td>48.5</td></tr><tr><td>0.5</td><td>1.5</td><td>31.4</td><td>15.4</td><td>72.3</td><td>81.8</td><td>38.1</td><td>42.5</td><td>46.9</td></tr><tr><td>0.5</td><td>3.0</td><td>33.5</td><td>15.0</td><td>69.1</td><td>81.8</td><td>37.5</td><td>41.0</td><td>46.3</td></tr><tr><td>0.8</td><td>2.0</td><td>36.6</td><td>13.6</td><td>71.1</td><td>81.6</td><td>37.1</td><td>43.8</td><td>47.3</td></tr><tr><td>0.2</td><td>2.0</td><td>33.5</td><td>13.9</td><td>71.0</td><td>80.0</td><td>38.2</td><td>41.6</td><td>46.4</td></tr></table>

We analyze the sensitivity of our method to the minimum $\omega _ { \mathrm { m i n } }$ and maximum $\omega _ { \mathrm { m a x } }$ weight hyperparameters, which control the dynamic range of the curriculum. Results in Table 4 show that our default setting of $\omega _ { \mathrm { m i n } } = 0 . 5$ and $\omega _ { \mathrm { m a x } } = 2 .$ 0 achieves the highest average score (48.5). Decreasing the dynamic range by either reducing $\omega _ { \mathrm { m a x } }$ (to 1.5) or increasing $\omega _ { \mathrm { m i n } }$ (to 0.8) leads to performance degradation. This indicates that a sufficiently large weighting range is crucial for the curriculum to effectively differentiate between easy and hard samples. Conversely, an overly extreme range (e.g., $\omega _ { \mathrm { m i n } } = 0 . 2 )$ also degrades performance, possibly because the curriculum excessively under-weights difficult queries. By assigning them a minimal weight for a prolonged period, the model is prevented from learning difficult queries. Furthermore, for additional experiments on the design choices for the r-score calculation, please see Appendix D.1.

Table 5: Computational cost vs. Performance gain. “Additional $\mathrm { C o s t } ^ { \prime \mathrm { { \rangle } } }$ is relative to the total training time.

<table><tr><td>Tree Size ( $k^{d}$ )</td><td> $3^{3}$ </td><td> $4^{3}$ </td><td> $3^{4}$ </td><td> $4^{4}$  (Default)</td></tr><tr><td>Time Cost (hours)</td><td>3.48</td><td>6.21</td><td>6.70</td><td>22.67</td></tr><tr><td>Additional Cost</td><td>+7.45%</td><td>+13.30%</td><td>+14.35%</td><td>+48.54%</td></tr><tr><td>Avg Performance Gain</td><td>+3.2</td><td>+3.0</td><td>+3.2</td><td>+4.0</td></tr></table>

## 7.4 ANALYSIS EXPERIMENTS

## 7.4.1 COMPUTATIONAL COST ANALYSIS

As shown in Figure 5, we analyzed the trade-off between the offline tree construction cost and the resulting performance gain.

Table 5 presents the time cost measured on 8 × H20 GPUs. While larger trees (4<sup>4</sup>) incur higher preprocessing costs compared to smaller trees $( 3 ^ { 3 } )$ , the cost remains manageable relative to the total training time (approx. 46 hours for 5 epochs), and the performance gains are substantial.

![](images/238842ae3548a8802c012994503e0ad0cbc0c00acec17ae73c4065414bdff6d6.jpg)  
Figure 5: Average performance gain versus reasoning tree size and computational cost.

## 7.4.2 IMPACT OF ORDERING

Table 6: Comparison between Re-Schedule (Easy-to-Hard) and Reverse Schedule (Hard-to-Easy).

<table><tr><td>Schedule</td><td>AIME24</td><td>AIME25</td><td>AMC23</td><td>MATH500</td><td>Minerva</td><td>Olympiad</td><td>Avg</td></tr><tr><td colspan="8">Linear Mapping</td></tr><tr><td>Re-Schedule (Ours)</td><td>34.2</td><td>15.6</td><td>72.4</td><td>81.2</td><td>36.4</td><td>42.5</td><td>47.1</td></tr><tr><td>Reverse Schedule</td><td>31.9</td><td>14.0</td><td>67.6</td><td>81.0</td><td>37.8</td><td>41.8</td><td>45.7</td></tr><tr><td colspan="8">Sigmoid Mapping</td></tr><tr><td>Re-Schedule (Ours)</td><td>34.2</td><td>16.0</td><td>71.1</td><td>81.8</td><td>42.3</td><td>44.4</td><td>48.3</td></tr><tr><td>Reverse Schedule</td><td>30.2</td><td>15.4</td><td>67.1</td><td>80.6</td><td>34.9</td><td>40.2</td><td>44.7</td></tr></table>

To validate the “easy-to-hard” curriculum design, we compared our method against a “Reverse Schedule” where lower r-score (harder) samples are prioritized first. As shown in Table 6, the Reverse Schedule leads to a significant drop in performance, confirming that starting with structurally simple samples is crucial for effective learning.

## 8 CONCLUSIONS

In this work, we challenged the reliance on path-based metrics for RLVR data scheduling. We introduced the r-score, a novel metric that quantifies learnability based on the structure of a query’s reasoning tree, and proposed Re-Schedule, a curriculum learning algorithm built upon it. Extensive experiments demonstrated that Re-Schedule consistently outperforms classical RLVR and existing scheduling methods, validating that r-score is a more effective proxy for learnability than path-based accuracy. Our findings establish that a structural understanding of the reasoning process provides a more powerful and principled foundation for creating efficient training curricula in RLVR.