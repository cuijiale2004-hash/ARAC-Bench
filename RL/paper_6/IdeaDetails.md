1. **Research Background and Existing Pain Points**
Recent breakthroughs in complex reasoning across code generation, mathematical problem-solving, and logical deduction have been driven by large language models. A key factor behind these advances is the combination of reinforcement learning (RL) with chain-of-thought (CoT), which enables models to iteratively refine intermediate reasoning steps. Building on this foundation, research has increasingly focused on scaling reasoning abilities in code generation, developing automated large-scale code data generation pipelines and applying RL using reward models and test case pass rewards. However, standard reinforcement learning with verifiable reward (RLVR) primarily focuses on generating responses and lacks mechanisms to explicitly foster critique or reflection. This paradigm can hardly elicit the internal critique or reflection behavior on existing solutions. While several recent studies, like Critique-Fine-Tuning (CFT) and Critique-Guided-Distillation (CGD), have shown the benefits of explicitly teaching LLMs how to critique, standard RL training solely incentivizes the model to predict a correct solution to a given query without exposing the model to reflective or critique-based reasoning patterns. Training exclusively on critique-oriented data also biases the model toward evaluative behaviors, encouraging it to focus on judging or analyzing candidate solutions rather than directly producing task-oriented outputs, which degrades its ability to generate complete answers.

2. **Core Research Motivation and Scientific Questions**
The core research motivation is to incorporate critique mechanisms into the reinforcement learning paradigm to explicitly reward models for accurate reflection, addressing the lack of critique and reflection incentives typically found in standard RL frameworks. The scientific questions explored are: How can we design a novel reinforcement learning framework that integrates critique learning and explicitly incentivizes the model's critique abilities via rewards for correctly judging a response's correctness? Can a model trained on a hybrid of standard RL and Critique Reinforcement Learning (CRL) data leverage the complementary strengths of both paradigms—where CRL helps develop critical thinking and reasoning abilities while RLVR focuses on enhancing problem-solving performance? What is the optimal ratio of CRL to RL data to prevent the model from being biased toward judgment-oriented behaviors that introduce a mismatch with inference-time requirements for long-horizon solution generation? Furthermore, can the critique and reasoning abilities learned through CRL on coding datasets transfer effectively to broader general reasoning domains?

3. **Overall Core Idea and Design Philosophy**
The overall core idea is to propose Critique Reinforcement Learning (CRL), a novel reinforcement learning training framework that incorporates critique learning within the RL paradigm by incorporating feedback on the correctness of critiques predicted by the model. Building on this foundation, the design philosophy introduces CRITIQUE-CODER, a model combining CRL and RL to leverage the strengths of both. CRL fosters critical thinking and reasoning, while RL focuses on optimizing problem-solving. Instead of substituting RL, CRL serves as a powerful complement to it. The model is tasked with generating a critique for a given (question, solution) pair, where the reward is determined solely by whether the final judgment label of the generated critique aligns with the ground-truth judgment. This is combined with standard RL data in a unified optimization process using Group Relative Policy Optimization (GRPO), allowing the policy to align with both execution correctness and reflective judgment. A hybrid dataset is constructed that randomly assigns 20% of the data to be CRL data, with the remaining 80% consisting of standard RL data, mitigating the risk of format shift and improving the robustness of the learning process.

4. **Core Innovation Points**
1. **Critique Reinforcement Learning (CRL) Paradigm**: Proposes CRL, where the model is prompted with a question–solution pair to predict a binarized judgment label. The reward is derived from comparing this prediction with the annotated label, explicitly incentivizing the model's critique abilities based on its self-generated judgment instead of using teacher-provided critique traces.
2. **Hybrid RL and CRL Training Framework**: Introduces CRITIQUE-CODER, trained on a hybrid of RL and CRL data by substituting 20% of the standard RL data with CRL data. This unified framework enables the model to jointly learn evaluative judgment and direct solution synthesis, ensuring CRL acts as a great complement rather than a replacement for standard RL.
3. **Unified Optimization via Advantage Aggregation**: Employs GRPO where the advantage estimation is jointly shaped by task outcomes (solution-level rewards from RL) and critique guidance (critique-level rewards from CRL). Two reward signals are combined together to update policy parameters, making the GRPO update fundamentally different from standard RL.
4. **Transferable General Reasoning Ability**: Demonstrates that the application of CRL on coding datasets enhances general reasoning and critique abilities, which are transferable across a broad range of tasks, as evidenced by substantial improvements on logical reasoning benchmarks (BIG-Bench Extra Hard) beyond the coding domain.
5. **Iterative Context Lengthening Strategy**: Adopts an iterative context lengthening approach where the training context length is extended from 16k to 32k tokens once rewards stabilize, allowing the model to first develop reasoning skills on shorter contexts which can then be applied to longer reasoning chains.

5. **Overview of the Overall Technical Solution**
The overall technical solution begins with dataset construction. The CRL dataset is built from the human-seeded RL dataset of rStar-Coder. To improve efficiency, test cases longer than 200 tokens are discarded and 30 cases are randomly sampled per problem. For critique data generation, QWEN3-CODER-30B-A3B-INSTRUCT is prompted to generate candidate solutions, which are executed on the filtered test cases. A pass rate threshold of 80% is adopted to determine the judgment label (True if pass rate > 80%, False otherwise). The hybrid dataset combines 20% CRL data and 80% standard RL data. The training procedure initializes the policy model from a pretrained checkpoint and trains on this hybrid dataset using GRPO. For CRL data, the model produces multiple critique predictions, parses the judgment from a specific field, and computes binary rewards by comparing with ground-truth labels. For RL data, the model generates multiple solution candidates, extracts code blocks, and evaluates them against test cases to obtain execution-based pass rate rewards. These rewards are transformed into advantage estimates and jointly used to update the policy. Training follows a two-phase strategy with maximum response lengths of 16k and 32k, using rule-based rewards tailored to data type, with CRL rewards scaled by 0.8 in the 16k phase to reduce dominance.

6. **Detailed Module Design**
**Dataset Construction Module**:
- **RL Dataset Selection**: Filters the rStar-Coder seed dataset by discarding test cases longer than 200 tokens and randomly sampling 30 cases for each problem. This reduces the average test case input characters from 96,208 to 40 and average cases from 87 to 24.
- **Critique Data Generation Module**: Prompts QWEN3-CODER-30B-A3B-INSTRUCT to generate outputs, extracting code blocks as candidate solutions. Empty code blocks are discarded. Each candidate solution is executed on filtered test cases, and its pass rate is computed. To handle timeouts, a pass rate threshold of 80% is adopted: candidate solutions are labeled as True if their test pass rate exceeds this percentage, and False otherwise.
- **Hybrid Data Integration Module**: Constructs a hybrid dataset combining both CRL and standard RL data. 20% of the data is randomly assigned as CRL data with judgment labels, and the remaining 80% consists of standard RL data with associated test cases, mitigating the risk of format shift caused by overexposure to critique-style supervision.

**Reward Function Module**:
- For CRL samples, the model is prompted to store the final judgment in a specific format, from which the predicted label c is extracted. A reward of 1 is assigned if c matches the ground truth c*; otherwise, including missing predictions, the reward is 0.
- For RL samples, the reward is defined as the pass rate across test cases (K/N, where K is successfully solved cases out of N total cases), ranging from 0.0 to 1.0.
- During the 16k phase, the CRL reward is scaled by a factor of 0.8 to reduce its dominance relative to RL signals. At the batch level, each instance receives its reward according to its data type.

**Optimization Module (GRPO)**:
- Utilizes Group Relative Policy Optimization (GRPO) which enhances performance by leveraging relative performance-based updates, yielding more stable and efficient policy refinement compared to PPO.
- The policy input can be either a single question q from RL or a question–solution pair from CRL. The advantage estimation is jointly shaped by the two distinct reward signals (solution-level and critique-level), allowing the policy to align with both execution correctness and reflective judgment.
- Hyperparameters include a batch size of 128, learning rate of 1e-6, 8 sampled outputs per prompt, and an asymmetric clipping ratio (upper bound 0.3, lower bound 0.2).

7. **All Mathematical Formulas and Symbol Definitions**
**Problem Definition Symbols**:
- $q$: Input question.
- $s, s_i$: Solution, or the i-th candidate solution.
- $\pi_\theta$: Policy model.
- $T$: Annotated test cases.
- $R_{rl,i}$: Solution-level reward signal for the i-th candidate from standard RL.
- $D = \{([q^k; s^k], c^{*k})\}_{k=1}^N$: Annotated dataset for CRL, where each pair consists of a question and a solution with an associated binary judgment label.
- $c^* \in \{0, 1\}$: Ground-truth binary judgment label.
- $c, c_i$: Predicted binary judgment label, or the i-th prediction.
- $R_{crl,i}$: Critique-level reward signal for the i-th prediction from CRL.

**GRPO Objective Function**:
$$J(\theta) = E \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min \left( \rho_{i,t}\hat{A}_{i,t}, \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon)\hat{A}_{i,t} \right) - \beta D_{KL}(\pi_\theta \| \pi_{ref}) \right]$$
Where:
$$\rho_{i,t} = \frac{\pi_\theta(o_{i,t} | x, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t} | x, o_{i,<t})}$$
- $\rho_{i,t}$: Probability ratio of generating output $o_{i,t}$ under the new policy $\pi_\theta$ and old policy $\pi_{\theta_{old}}$.
- $\hat{A}_{i,t}$: Calculated advantage within each output group.
- $D_{KL}$: KL divergence measuring the divergence between $\pi_\theta$ and the reference policy $\pi_{ref}$, preventing the policy from drifting too far.
- $x$: Policy input, either a single question $q$ from RL or a question–solution pair $([q; s])$ from CRL.

**Reward Function Formulas**:
$$R_{crl}(c, c^*) = \begin{cases} 1, & \text{if } c = c^* \\ 0, & \text{otherwise} \end{cases}$$
$$R_{rl}(s, T) = \frac{K}{N}$$
Where $T$ is the set of $N$ test cases and $K$ denotes the number of cases successfully solved by the model's output $s$. $R_{rl} \in [0, 1]$.

**Batch Level Reward**:
$$R_i = \begin{cases} R_{crl}(c_i, c_i^*), & \text{if } i \in B_{CRL} \\ R_{rl}(s_i, T_i), & \text{if } i \in B_{RL} \end{cases}$$
Where $B_{CRL}$ and $B_{RL}$ denote the CRL and RL subsets within the batch, respectively.

8. **Algorithm Pseudocode**
**Algorithm 1 Training procedure of CRITIQUE-CODER**
Input dataset $D = \{([q_i; s_i], c_i^*)\}_{i=1}^{N_1} \cup \{(q_j, t_j)\}_{j=1}^{N_2}$, policy $\pi_\theta$
1: Initialize policy model $\pi_\theta \leftarrow \pi_\theta^{init}$
2: for each step do
3:   Sample a batch $B \subset D$
4:   for each data instance $d \in B$ do
5:     if $d = ([q; s], c^*)$ ▷ CRL data with judgment then
6:       Sample G outputs $\{o_i\}_{i=1}^G \sim \pi_\theta([q; s])$
7:       Parse each $o_i$ to extract judgment $c_i$ inside \conclusion{}
8:       Compute reward $R_{crl,i}(c_i, c^*)$ for each $c_i$
9:     else if $d = (q, t)$ ▷ RL data with test cases then
10:      Sample G outputs $\{o_i\}_{i=1}^G \sim \pi_\theta(q)$
11:      Parse each $o_i$ to extract solution $s_i$ enclosed by ''' [code] '''
12:      Evaluate $s_i$ on test cases $t$, obtain reward $R_{rl,i}(s_i, t)$
13:    end if
14:  end for
15:  Compute $\hat{A}_{i,t}$ from reward $R_i$, where $R_i = R_{crl,i}(c_i, c^*)$ for CRL or $R_{rl,i}(s_i, t)$ for RL.
16:  Update the policy model $\pi_\theta$ with GRPO ( Equation 1)
17: end for