## 1. Research Background and Existing Pain Points

**Research Background:**
Large Language Models (LLMs) have demonstrated remarkable versatility across diverse domains, ranging from mathematical problem-solving, search-result integration, and tool-use to long-horizon agentic interaction. A pivotal recent advancement involves the use of reinforcement learning (RL) to bolster complex reasoning capabilities. By optimizing models through reward signals tied to verifiable outcomes—such as final answer accuracy—RL provides a scalable framework for eliciting sophisticated problem-solving behaviors, including self-reflection.

**Existing Pain Points:**
1. **Failure of Reinforcement Learning with Verifiable Rewards (RLVR) on Hard Problems:** The effectiveness of outcome-based RL methods fundamentally depends on the policy model’s ability to discover correct solutions within a limited rollout budget. This learning paradigm struggles on challenging problems from the training data, where the model’s success rate is effectively zero (i.e., the pass@k rate remains zero even after sampling k rollouts). For these problems, defined as $D_{hard}$, an incorrect intermediate step can derail the entire reasoning chain for a 7B-scale LLM, resulting in negative learning signals regardless of any partially correct solutions. Naively penalizing all incorrect final outputs can introduce training instability and hinder progress, making these difficult reasoning tasks largely intractable for standard outcome-based RL methods.
2. **Overfitting and Shallow Reasoning in Supervised Fine-Tuning (SFT):** An alternative approach is imitation learning via SFT on expert demonstrations. While SFT can instill valuable reasoning behaviors, its next-token prediction objective enforces rigid, token-level imitation, limiting the model’s ability to generalize beyond the training data. This problem becomes particularly pronounced when training data are modest in scale and the model itself is relatively less capable. Under such conditions, long and complex demonstrations often lead to overfitting and shallow reasoning behaviors, resulting in performance degradation rather than improvement.
3. **Vanishing Advantage in RL Optimization:** A key challenge for RL algorithms emerges when input queries are either too easy or too hard, resulting in uniform correctness within policy rollouts. In such cases, the advantage estimate vanishes, yielding an uninformative policy gradient and preventing model updates. While dynamic sampling is used to mitigate this, standard RLVR still fails to provide dense enough signals for problems where zero correct rollouts are found.

## 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:**
Both SFT and outcome-based RL struggle on challenging reasoning tasks, leaving a critical gap for training small open-source models to effectively learn difficult problems. There is a need for a framework that can provide rich learning signals even when all rollouts are incorrect, while encouraging flexible reasoning guided by expert demonstrations. The motivation is to bridge the gap between rigid imitation learning (SFT) and sparse outcome-based reinforcement learning (RLVR) by reformulating problem solving as a sequential decision-making process where intermediate "actions" can be supervised effectively.

**Scientific Questions:**
How can we design a training framework that decomposes complex problem-solving into a sequence of logical actions, trains the model to generate these actions via an RL-style objective while maintaining flexibility in its internal reasoning, and provides dense, efficiently computable supervision signals that scale to large datasets even for problems where the model initially fails to find correct final answers?

## 3. Overall Core Idea and Design Philosophy

**Overall Core Idea:**
The core idea is Supervised Reinforcement Learning (SRL), a framework that reformulates problem-solving as a sequential decision-making process. Rather than optimizing for a final answer or imitating an entire expert trajectory, SRL trains the model to reproduce the sequence of key actions underlying expert reasoning, following an RL-style objective. Expert demonstrations are decomposed into a series of intermediate actions. During training, the model first generates an internal monologue to articulate its reasoning and then commits to an "action". At every step, SRL provides a reward based on the similarity between the model’s predicted action and the corresponding expert action, thereby providing fine-grained, efficiently computable supervision.

**Design Philosophy:**
1. **Action-based Formulation:** Decompose solutions into logical actions (e.g., algebraic manipulation, shell command) rather than treating the response as a monolithic token sequence.
2. **Separation of Thought and Action:** Train the model to generate an internal reasoning monologue before committing to each action. The reward is computed only on the logical action, not the internal monologue. This grants the model flexibility to develop its own internal reasoning style while ensuring its external actions align with the expert’s strategy.
3. **Dense Similarity Rewards:** Provide smoother rewards based on the similarity between the model’s actions and expert actions in a step-wise manner, overcoming the sparsity of final-outcome rewards in RLVR.
4. **Curriculum Learning:** Initialize training with SRL before refining with RLVR to establish a strong foundation for reasoning before optimizing for final correctness.

## 4. Core Innovation Points

1. **Novel SRL Framework for Hard Reasoning Tasks:** Proposing Supervised Reinforcement Learning (SRL), a framework designed to enable effective learning on difficult reasoning tasks where SFT and RLVR struggle, by providing dense and smooth rewards based on similarity with expert actions.
2. **Action-based Problem Formulation and Step-wise Data Construction:** Reformulating the problem as generating a sequence of logical actions and constructing training data where the input is a partial context $x_{step_k} = [x, y_{step_1}, \dots, y_{step_{k-1}}]$ and the target is the subsequent step action $y_{step_k}$. This transforms one expert solution into a rich set of training instances.
3. **Internal Reasoning Monologue with Step-wise Similarity Reward:** Training the model to generate an internal monologue encapsulated by "think" tags before the action, and providing a dense reward based on the sequence similarity ratio $R(y'_{step_k}, y_{step_k}) = \frac{2M}{T}$ between the model's action and the expert's action, rather than sparse final correctness.
4. **Dynamic Sampling Generalization for SRL:** Generalizing the dynamic sampling strategy designed for outcome accuracy to SRL by filtering samples with near-zero variance in sequence similarity rewards, ensuring meaningful policy updates.
5. **SRL -> RLVR Pipeline for Optimal Performance:** Establishing that initializing training with SRL before refining with RLVR yields the strongest overall performance, acting as a robust curriculum learning strategy.

## 5. Overview of the Overall Technical Solution

The overall technical solution involves leveraging a powerful teacher model to generate solution trajectories, which are then decomposed into sequences of step-wise actions. For each step in the trajectory, a partial context is created consisting of the problem and the preceding steps. The policy model is prompted to generate a subsequent action step along with its own inner monologue (formatted with specific tags). A dense reward is computed based on the similarity between the generated action and the expert action using a sequence matching algorithm (specifically Python’s `difflib.SequenceMatcher`). The policy is then optimized using the Group Relative Policy Optimization (GRPO) objective with this sequence similarity reward. Additionally, a dynamic sampling strategy filters out samples where the standard deviation of rewards across rollouts is below a threshold, preventing weak learning signals. The process can be followed by a standard RLVR phase for final refinement.

## 6. Detailed Module Design

**Module 1: Action-based Problem Formulation**
Given an expert solution trajectory $y$ that leads to a correct final answer, the trajectory is decomposed into a sequence of tuples: $y = \{y_{step_n}\}_{n=1}^N$. Each step represents a logical action: the concrete action to be operated. This formulation is domain-agnostic; an action in mathematical reasoning could be an algebraic manipulation, while for a software agent, it could be a command executed in a code repository.

**Module 2: Step-wise Training Data Construction**
To create training data for SRL, a powerful teacher model $\theta_{expert}$ is used to generate solution trajectories. From a single complete solution with $N$ steps, $N-1$ partial trajectories are constructed. For each step $k \in \{1, \dots, N-1\}$, a new input prompt is created: $x_{step_k} = [x, y_{step_1}, \dots, y_{step_{k-1}}]$, where the model’s task is to predict the subsequent step $y_{step_k}$. Empirically, providing the subsequent step title (e.g., "2. **Coprime Pairs**") as additional context for the learner to predict the rest of the step content can further boost performance.

**Module 3: Learning with Sequence Similarity Reward and Own Inner Monologue**
Given a partial context $x_{step_k}$, the policy model $p_\theta$ is prompted to generate the subsequent action step along with its own inner monologue $y'_{think}$, which is encapsulated by "think" tags. The prediction is formally specified as: $y' \sim p_\theta(\cdot|x_{step_k}) = [y'_{think}, y'_{step_k}]$.

**Module 4: Reward Function Calculation**
The reward function measures the similarity between the generated action and the expert action using the ratio $R(y'_{step_k}, y_{step_k}) = \frac{2M}{T}$, where:
- $T$ (Total elements): The total number of elements in both sequences combined, calculated as $T = |S_1| + |S_2|$.
- $M$ (Matched elements): The total count of elements found in all non-overlapping matching blocks between the two sequences. The algorithm finds the longest contiguous matching subsequence and recursively searches for more matches. $M = \sum_{(i,j,n) \in \text{MatchingBlocks}} n$.
The similarity ratio is $R = \frac{2 \sum_{(i,j,n) \in \text{MatchingBlocks}} n}{|S_1| + |S_2|}$. If the generated output $y'$ fails to adhere to the required format, a negative reward is assigned.

**Module 5: Dynamic Sampling for SRL**
To ensure meaningful policy updates, samples with near-zero variance in rewards across rollouts are filtered out. A sample is retained if the standard deviation of the reward scores of its rollouts exceeds a threshold $\epsilon > 0$:
$\sqrt{\frac{\sum_{i=1}^G (r(o_i, y) - \bar{r})^2}{G}} > \epsilon$
where $G$ is the number of generated rollouts, $r(o_i, y)$ is the sequence similarity reward for the $i$-th rollout $o_i$, and $\bar{r}$ is the mean reward.

**Module 6: Extension to Software Engineering Agentic Reasoning**
For software engineering tasks, trajectories are composed of multiple steps defined by the agent’s interactions. A single step contains natural language reasoning followed by an executable action. The "action" is defined as the environment-consumable command (e.g., the bash call inside function tags). Following this decomposition, full trajectories are processed to create step-wise training instances where the model predicts the next executable action based on past observations and actions.

## 7. All Mathematical Formulas and Symbol Definitions

**LLM Probability Distribution:**
$p_\theta(y|x) = \prod_{j=1}^m p_\theta(y_j|x, y_1, \dots, y_{j-1})$
Where:
- $\theta$: Model weights
- $x = [x_1, \dots, x_n]$: Input prompt token sequence
- $y = [y_1, \dots, y_m]$: Response sequence
- $p_\theta(y_j|x, y_1, \dots, y_{j-1})$: Probability of token $y_j$ given prompt and preceding tokens

**SFT Loss Function:**
$L_{SFT}(\theta) = -\sum_{i=1}^N \log p_\theta(y^{(i)}|x^{(i)})$
Where:
- $D = \{(x^{(i)}, y^{(i)})\}_{i=1}^N$: SFT dataset

**GRPO Objective Function:**
$E\left[\frac{1}{G}\sum_{i=1}^G \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \min\left(\frac{p_\theta(o_{i,t}|x, o_{i,<t})}{p_{\theta_{old}}(o_{i,t}|x, o_{i,<t})}\hat{A}_{i,t}, \text{clip}\left(\frac{p_\theta(o_{i,t}|x, o_{i,<t})}{p_{\theta_{old}}(o_{i,t}|x, o_{i,<t})}, 1-\epsilon, 1+\epsilon\right)\hat{A}_{i,t}\right) - \beta D_{KL}[p_\theta \| p_{ref}]\right]$
Where:
- $G$: Number of sampled trajectories
- $\{o_i\}_{i=1}^G$: Response trajectories
- $\theta_{old}$: Policy from previous iteration
- $\hat{A}_{i,t} = (\tilde{r}_i - \text{mean}(\tilde{r}))/\text{std}(\tilde{r})$: Group-level normalized advantage
- $\epsilon$: Clipping range
- $\beta$: KL-divergence penalty coefficient

**Hard Problem Set Definition:**
$D_{hard} = \{x^{(i)}, a^{(i)}\}_{i=1}^N$
Where the condition is: $\frac{1}{k}\sum_{j=1}^k \mathbb{I}(\text{ExtractAnswer}(y^{(j)}) == a) \le \epsilon$
- $y^{(j)} \sim p_\theta(\cdot|x)$: Solution attempt
- $\epsilon > 0$: Small constant

**Sequence Similarity Reward:**
$R(y'_{step_k}, y_{step_k}) = \frac{2M}{T}$
Where:
- $T = |S_1| + |S_2|$: Total elements in both sequences
- $M = \sum_{(i,j,n) \in \text{MatchingBlocks}} n$: Total count of elements in non-overlapping matching blocks

**Similarity Ratio:**
$R = \frac{2 \sum_{(i,j,n) \in \text{MatchingBlocks}} n}{|S_1| + |S_2|}$

**Final SRL Reward:**
$r(y'_{step_k}, y_{step_k}) = \begin{cases} R(y'_{step_k}, y_{step_k}) & \text{if } y' \text{ follows format} \\ -1 & \text{otherwise} \end{cases}$

**Dynamic Sampling Condition:**
$\sqrt{\frac{\sum_{i=1}^G (r(o_i, y) - \bar{r})^2}{G}} > \epsilon$
Where:
- $G$: Number of generated rollouts
- $r(o_i, y)$: Sequence similarity reward for $i$-th rollout
- $\bar{r}$: Mean reward for the sample
- $\epsilon$: Threshold

## 8. Algorithm Pseudocode

The paper does not provide a formal pseudocode block, but the logical flow of the algorithm is described as follows based on the methodology:

**Algorithm: Supervised Reinforcement Learning (SRL)**

1. **Input:** Expert dataset $D_{expert}$, Policy model $p_\theta$, Number of rollouts $G$, Similarity threshold $\epsilon_{sample}$
2. **Data Preparation:**
   - For each expert trajectory $y \in D_{expert}$:
     - Decompose $y$ into sequence of steps $\{y_{step_n}\}_{n=1}^N$.
     - For $k = 1$ to $N-1$:
       - Construct partial context $x_{step_k} = [x, y_{step_1}, \dots, y_{step_{k-1}}]$.
       - Store pair $(x_{step_k}, y_{step_k})$ in training set $D_{train}$.
3. **Training Loop:**
   - While not converged:
     - Sample a batch of partial contexts $B$ from $D_{train}$.
     - For each context $x_{step_k} \in B$:
       - Generate $G$ rollouts: $o_i \sim p_\theta(\cdot|x_{step_k})$ for $i=1 \dots G$. Each rollout format: $[y'_{think}, y'_{step_k}]$.
       - Compute reward for each rollout: $r(o_i, y_{step_k}) = R(y'_{step_k}, y_{step_k})$ if format valid, else $-1$.
     - **Dynamic Sampling:** Filter out contexts from the batch where $\sqrt{\frac{\sum_{i=1}^G (r(o_i, y) - \bar{r})^2}{G}} \le \epsilon_{sample}$.
     - Re-sample to fill batch size if needed.
     - Compute advantage $\hat{A}_{i,t}$ using group-level normalized rewards.
     - Update policy $p_\theta$ using GRPO objective function.
4. **Output:** Trained policy model $p_\theta$