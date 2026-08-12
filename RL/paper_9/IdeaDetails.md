## 1. Research Background and Existing Pain Points

**Research Background:**
While reinforcement learning (RL)-based fine-tuning has significantly improved Large Language Models (LLMs) reasoning and planning, models still struggle with seemingly simple tasks and incur high token costs during inference-time search. Textual reasoning excels at semantics and commonsense, but falls short in precise computation, symbolic manipulation, optimization, and algorithmic processing. In contrast, symbolic code generation handles these rigorously and benefits from external tools (e.g., equation solvers). Prompting LLMs to generate and execute code often outperforms pure textual reasoning. OpenAI’s GPT models address this by incorporating a Code Interpreter, allowing iterative code generation and reasoning over outputs. However, current public research still lacks a comprehensive understanding of how to fine-tune LLMs to integrate with Code Interpreter for robust, generalizable performance across hundreds of tasks.

**Existing Pain Points:**
1. **Steering Difficulty:** Guiding LLMs to decide when to rely on textual reasoning versus programmatic solutions is a key challenge, given that most input questions lack explicit cues about which approach is best and the possible text/code solution space is large.
2. **Underutilization of Symbolic Capabilities:** Current Code Interpreter implementations struggle to effectively steer between text and code. LLM-generated code frequently degenerates into hard-coded text-like scripts, limiting its symbolic utility.
3. **Limited Scope of Prior Work:** Recent work such as ToRL and ReTool investigates training reasoning models to integrate with Code Interpreters, but their training and evaluation are limited to math problems, leaving a significant gap from real-world applications that demand effectiveness across broader benchmarks. ToolRL focuses on teaching models to select among multiple tools, where the Code Interpreter is used only for generating relatively simple code.
4. **Task Heterogeneity and Sparse Samples:** Applying traditional DeepSeek-style RL training to a general Code Interpreter across 144 challenging tasks yields only marginal gains (+3.4%). This difficulty arises from task heterogeneity and the scarcity of effective samples.
5. **Vanishing Policy Gradient Signal:** In batches dominated by too-easy or too-hard items, the Bernoulli variance of the reward vanishes (as $p(1-p) \to 0$). The update is then governed by the KL term, contracting $\pi_\theta$ toward $\pi_{ref}$, preventing optimization headway.
6. **Training Inefficiency:** Time-consuming code execution significantly reduces GPU utilization and accounts for a large portion of RL training time in the Code Interpreter, limiting the maximum batch size and hindering parallel efficiency.

## 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:**
To tackle the challenges of training a general-purpose Code Interpreter across diverse reasoning and planning tasks. There is a critical need to address task heterogeneity and the scarcity of effective samples that hinder RL training. Furthermore, the vanishing policy gradient signal in mixed-difficulty batches must be resolved to allow effective optimization. Additionally, the computational inefficiency caused by the coupling of code execution and gradient computation needs to be mitigated to make training feasible.

**Scientific Questions:**
1. How can we effectively fine-tune open-source LLMs to integrate with a Code Interpreter for robust, generalizable performance across 100+ diverse reasoning and planning tasks?
2. How can we overcome the vanishing policy gradient signal in mixed-difficulty batches during GRPO training, where trivially easy or excessively difficult samples dilute the reward signal?
3. How can we decouple time-consuming code execution from GPU-based gradient computation to improve training efficiency and GPU utilization?

## 3. Overall Core Idea and Design Philosophy

**Overall Core Idea:**
The core idea is to present R1-Code-Interpreter, an extension of a text-only LLM trained via multi-turn supervised fine-tuning (SFT) and reinforcement learning (RL) to autonomously generate multiple code queries during step-by-step reasoning. The model iteratively reasons, optionally generates code for execution, and refines its reasoning based on the results, continuing this process until the final answer is produced.

**Design Philosophy:**
Unlike conventional curriculum learning, which progresses from easy to difficult samples, the proposed approach orders samples by their "improvement potential," i.e., their expected benefit to model optimization. Samples that are either trivially easy (almost always solved) or excessively difficult (almost never solved) provide limited training signal. In contrast, samples that the model solves correctly about half the time offer the strongest optimization opportunities. The RL training prioritizes samples with higher potential and gradually shifts to lower-potential ones. Furthermore, to address training inefficiency, the design philosophy incorporates a specialized Code Execution Sandbox on multiple CPU nodes, decoupling execution from GPU-based gradient computation.

## 4. Core Innovation Points

1. **First General Code Interpreter Training Across Multiple Tasks/Domains:** To the best of the authors' knowledge, this is the first published work to train general Code Interpreter across multiple tasks and domains. Unlike prior work focused on single tasks or simple reasoning, 144 challenging reasoning and planning tasks are curated, each with over 200 samples of varying difficulty, standardized into a unified format.
2. **Multi-Stage Curriculum Learning Framework Guided by Improvement Potential:** Analysis of RL limitations led to the proposal of an effective multi-stage curriculum learning framework guided by improvement potential. By tuning the number of training tasks, it was found that RL for general-purpose Code Interpreter is substantially more challenging due to task heterogeneity and scarcity of effective samples. The proposed multi-stage curriculum learning with measured improvement potential effectively mitigates this bottleneck, raising RL performance gains from +3.4% to +9.3%.
3. **Cost-Efficient Training via Decoupled Code Execution Sandbox:** Time-consuming code execution significantly reduces GPU utilization. To address this, a specialized Code Execution Sandbox on multiple CPU nodes is designed, decoupling execution from GPU-based gradient computation. This approach reduces overall training time by about 39%.
4. **Comprehensive Comparison of Training Strategies:** Comparison of training strategies shows that: 1) The multi-turn Code Interpreter framework proves more generalizable and effective than single-turn text or code generation frameworks. 2) Warm-starting with SFT significantly improves the training of Code Interpreter. 3) Qwen-2.5 models as base are better than DeepSeek R1-distilled reasoning models.

## 5. Overview of the Overall Technical Solution

The overall technical solution begins by curating 144 reasoning and planning tasks from Sym-Bench, Big-Bench-Hard, and Reasoning-Gym. 6.5k multi-turn text/code trajectories are synthesized for SFT, followed by Group Relative Policy Optimization (GRPO). The model iteratively reasons, optionally generates code for execution, and refines reasoning based on the results. The Code Interpreter is invoked only when deemed beneficial; otherwise, the model relies on pure textual reasoning. The system instruction directs the model to enclose code between the natural tokens ‘``` python’ and ‘```’. Upon detecting a code block, the system extracts and executes it via the Code Interpreter, then appends the result (prefixed with ‘Code Execution Results:’) to the ongoing generation. This loop continues until either (1) the maximum of 8 code calls is reached, or (2) the model emits a final answer enclosed between ‘<<<’ and ‘>>>’.

To solve the RL training bottleneck, a multi-stage curriculum learning method is proposed. After the SFT stage, the model is prompted with diverse single- and multi-turn agent strategies to answer the same question. The accuracy discrepancies among these answers are used to estimate improvement potential. RL training then proceeds in four stages, beginning with high-potential samples and moving to lower-potential ones. Each stage runs for 150 steps, gradually expanding the training distribution until the full dataset is included by stage 4. To save training time, a specialized Code Execution Sandbox on five 64-core CPU nodes decouples gradient computation from code execution.

## 6. Detailed Module Design

**Response Format Module:**
The head prompt guides the initial LLM to follow a predefined structure. The prompt organizes the output into three iterative parts: reasoning, optional Code Interpreter invocation, and the final answer. No content-specific constraints (e.g., enforcing reflective reasoning or code calls) are imposed to preserve the model’s natural learning dynamics during RL. Unlike prior work that enforces section tags like ‘THOOK’, ‘<answer>’, or ‘<search>’, this design relies solely on the final answer marker ‘<<<answer content>>>’ for answer extraction. For code, it leverages the LLM’s pretrained behavior of naturally starting code blocks with ‘``` python’, which serves as implicit tagging. A code query must involve only a single script that uses ‘print’ function for the output.

**GRPO with Code Interpreter Module:**
The RL objective integrates external code execution. For each input $x$, GRPO samples a group of responses $\{y_1, y_2, \ldots, y_G\}$ from the reference policy $\pi_{ref}$ and optimizes the policy. Code execution tokens are masked, and the policy gradient is computed only over LLM-generated tokens. The total rule-based reward $R$ is defined as a weighted combination of correctness, format compliance, and efficiency. The agent receives +1.0 if the final outcome is correct, +0.1 if all intermediate responses satisfy the format requirements (and −0.1 otherwise), and −0.1 if the number of generation turns exceeds six.

**Code Execution Sandbox Module:**
Code execution allows up to 8 code calls and a 60-second timeout per script. Generated codes are executed directly in a specialized sandbox on five 64-core CPU nodes in parallel during batch inference, reducing overall RL training time by about 39% and avoiding memory overflow on GPUs.

**Improvement Potential Measurement Module:**
To capture variability in problem-solving strategies, four agent variants are implemented: All Text, All Code, Code Agent, and CodeSteer. Starting from the SFT-initialized base model, each agent generates 5 samples at different temperatures, yielding $N = 20$ answers per question. Let $a_{i,j}$ denote the $j$-th answer for sample $x_i$, and $y_{i,j} \in \{0, 1\}$ its correctness label. The empirical correctness rate and the improvement potential score are calculated to determine the training order.

**Multi-Stage Curriculum Module:**
All training samples are sorted by $\Pi_i$ and partitioned into four equally sized groups (four group IP ranges are: [0.0, 0.32], [0.32, 0.48], [0.48, 0.64], and [0.64, 1.00]), from highest to lowest potential. The partition is sample-wise rather than task-wise. GRPO training begins with the highest-potential group, then progressively incorporates lower-potential groups in subsequent stages. Each stage runs for 150 steps.

## 7. All Mathematical Formulas and Symbol Definitions

**GRPO Objective with Code Interpreter:**
$$ \max_{\pi_\theta} \mathbb{E}_{x \sim D, y \sim \pi_\theta(\cdot|x;C)} [r_\phi(x, y)] - \beta D_{KL}[\pi_\theta(y | x; C) \| \pi_{ref}(y | x; C)] $$
Where $\pi_\theta$ is the policy LLM, $\pi_{ref}$ is the reference model, $r_\phi$ is the reward, and $D_{KL}$ is the KL divergence.

**GRPO Optimization Objective:**
$$ J_{GRPO}(\theta) = \mathbb{E}_{x \sim D, y_{1:G} \sim \pi_{old}(\cdot|x;C)} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \min \left( \frac{\pi_\theta(y_{i,t} | x, y_{i,<t}; C)}{\pi_{ref}(y_{i,t} | x, y_{i,<t}; C)} \hat{A}_{i,t}, \text{clip} \left( \frac{\pi_\theta(y_{i,t} | x, y_{i,<t}; C)}{\pi_{ref}(y_{i,t} | x, y_{i,<t}; C)}, 1-\epsilon, 1+\epsilon \right) \hat{A}_{i,t} \right) \right] - \beta D_{KL}[\pi_\theta \| \pi_{ref}] $$
Where $\epsilon$ and $\beta$ are hyperparameters, and $\hat{A}_{i,t}$ is the advantage.

**GRPO Gradient Decomposition:**
Let $r_i \in \{0, 1\}$ be the final reward for rollout $y_i$ in a group of size $G$, with group mean $\bar{r} = \frac{1}{G} \sum_{j=1}^G r_j$. Broadcasting the sequence-level advantage to tokens (and omitting clipping for clarity), the GRPO gradient splits into a policy term and a KL regularizer:
$$ \nabla_\theta J_{GRPO}(\theta) = \frac{1}{G} \sum_{i=1}^G (r_i - \bar{r}) v_i - \beta \nabla_\theta D_{KL}[\pi_\theta \| \pi_{ref}], \quad v_i := \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \nabla_\theta \log \pi_\theta(y_{i,t} | x, y_{i,<t}; C). $$

**Variance Identity for Group-Relative Bernoulli Rewards:**
If $r_1, \ldots, r_G \stackrel{i.i.d.}{\sim} \text{Bernoulli}(p)$ and $\bar{r} = \frac{1}{G} \sum_{j=1}^G r_j$, then:
$$ \mathbb{E}[(r_i - \bar{r})^2] = p(1-p) \left(1 - \frac{1}{G}\right) $$

**Bound on the Policy-Gradient Magnitude:**
Let $g_{pol} := \frac{1}{G} \sum_{i=1}^G (r_i - \bar{r}) v_i$ be the policy term. Then:
$$ \mathbb{E}[\|g_{pol}\|^2] \leq p(1-p) \left(1 - \frac{1}{G}\right) \mathbb{E}\left[ \frac{1}{G} \sum_{i=1}^G \|v_i\|^2 \right] $$
The upper bound vanishes as $p \to 0$ or $p \to 1$, and is maximized at $p = 1/2$.

**Improvement Potential Score:**
Let $p_i = \frac{1}{N} \sum_{j=1}^N y_{i,j}$ be the empirical correctness rate. The improvement potential score is defined as:
$$ \Pi_i = \frac{p_i(1-p_i)}{1/4} = 4 p_i(1-p_i) $$
By construction, $\Pi_i \in [0, 1]$, maximized when $p_i = 0.5$ (mixed outcomes) and minimized when $p_i \in \{0, 1\}$ (uniformly correct or incorrect).

**Descent Lemma View: Expected Improvement Per Step:**
Assume $J_{GRPO}$ is $L$-smooth and consider a stochastic gradient step $\theta^+ = \theta + \eta \hat{g}$, where $\hat{g}$ is an unbiased estimator of $\nabla_\theta J_{GRPO}$. The descent lemma yields:
$$ \mathbb{E}[J_{GRPO}(\theta^+) - J_{GRPO}(\theta)] \geq \eta \|\mathbb{E}[\hat{g}]\|^2 - \frac{L\eta^2}{2} \mathbb{E}[\|\hat{g}\|^2] $$

**Concentration of the Potential Estimator:**
For any $\epsilon > 0$:
$$ \Pr(|\hat{p}_i - p_i| \geq \epsilon) \leq 2 \exp(-2N\epsilon^2) $$
Hence $\Pi_i$ concentrates around $4p_i(1-p_i)$ as $N$ grows.

## 8. Algorithm Pseudocode

The paper does not provide explicit algorithm pseudocode blocks, but the operational logic of the Multi-Stage Curriculum Learning with Potential Measurement is described in detail as follows:

**Algorithm: Multi-Stage GRPO with Improvement Potential**
1. **Input:** Training dataset $D$ with questions $x_i$, SFT-initialized LLM $\pi_\theta$, Code Interpreter $C$.
2. **Estimate Improvement Potential:**
   - For each question $x_i$:
     - Generate $N=20$ answers using 4 agent variants (All Text, All Code, Code Agent, CodeSteer) at different temperatures.
     - Compute correctness labels $y_{i,j} \in \{0, 1\}$.
     - Calculate empirical correctness rate $p_i = \frac{1}{N} \sum_{j=1}^N y_{i,j}$.
     - Calculate improvement potential $\Pi_i = 4 p_i(1-p_i)$.
3. **Partition Dataset:**
   - Sort all training samples by $\Pi_i$ in descending order.
   - Partition into 4 equally sized groups:
     - Stage 1: $\Pi_i \in [0.64, 1.00]$ (High potential)
     - Stage 2: $\Pi_i \in [0.48, 0.64]$ (Moderate potential)
     - Stage 3: $\Pi_i \in [0.32, 0.48]$ (Emerging potential)
     - Stage 4: $\Pi_i \in [0.0, 0.32]$ (Low potential)
4. **Multi-Stage Training:**
   - For stage $s = 1$ to $4$:
     - Merge data from groups $1$ to $s$ to form the training set for the current stage.
     - For step $= 1$ to $150$:
       - Sample a batch of prompts $x$ from the current training set.
       - For each prompt, sample a group of responses $\{y_1, \dots, y_G\}$ using $\pi_{old}(\cdot|x;C)$ with Code Interpreter $C$.
       - Execute code blocks in the Code Execution Sandbox on CPU nodes.
       - Compute rule-based rewards $r_i \in \{0, 1\}$ based on correctness, format, and efficiency.
       - Compute advantages $\hat{A}_{i,t}$ based on group relative rewards.
       - Update $\pi_\theta$ via GRPO objective $J_{GRPO}(\theta)$.
5. **Output:** Final trained model R1-Code-Interpreter.