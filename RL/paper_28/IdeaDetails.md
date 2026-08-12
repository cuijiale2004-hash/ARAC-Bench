## 1. Research Background and Existing Pain Points

**Research Background:**
Large Language Model (LLM)-based judges are emerging as a critical component in the LLM ecosystem, typically used with scoring and ranking model outputs. This evaluation capability is essential at multiple stages of LLM development: during post-training, judges provide preference signals for alignment; at inference time, judges verify and select responses through best-of-N decoding; and during evaluation, judges deliver reliable assessments without manual human assessment. Thus, training accurate LLM-based judges is of great importance for building powerful language models.

**Existing Pain Points:**
1. Classical evaluation with reward models often outputs scores directly, which cannot fully harvest the inherent reasoning capability of LLMs. 
2. Recent progress in generative reward modeling and reinforcement learning equips judges with thinking before producing final predictions via long chains of textual reasoning traces. However, they remain inherently limited in scenarios that require precise computation or symbolic reasoning – capabilities that are much more challenging for text-only models.
3. Early attempts exploring the equipping of LLM judges with tool-use abilities reveal two major limitations: (i) Inference-time restriction: most methods integrate tool-use only at the inference stage, preventing deeper integration between reasoning processes and tool execution. (ii) Narrow task coverage: many are tailored to specific domains or specialized task types, which limits their applicability in general-purpose judging scenarios.

## 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:**
Recent advances in LLM tool-use provide a promising avenue to overcome the limitations of text-only judges. By granting access to executable interfaces for enumeration, verification, and computation, tools enable exact validation of reasoning steps rather than relying on potentially error-prone text-based inference. For example, code execution can automatically verify outputs on certain instructions or check intermediate calculations in math reasoning. There is a critical need for robust judges that tightly couple reasoning with tool execution and are optimized end-to-end.

**Scientific Questions:**
How to develop an LLM judge that can reliably integrate reasoning with code interpreter execution, and how to fully unleash the potential of reinforcement learning (RL) to optimize this tool-integrated reasoning process end-to-end from the training time, enabling self-improvement even without distillation?

## 3. Overall Core Idea and Design Philosophy

**Overall Core Idea:**
Incorporating tool-integrated reasoning (TIR), we propose TIR-Judge, a framework that leverages reinforcement learning (RL) to teach models to generate code, execute it with interpreters, and iteratively refine their reasoning based on the resulting outputs. By reinforcing this cycle of reasoning and tool-use, TIR-Judge equips LLM judges with the ability at the training time to deliver more accurate and verifiable evaluations across diverse tasks.

**Design Philosophy:**
To fully unleash the potential of RL for TIR-Judge, the framework is built on three key design principles:
1. Task diversity: To balance between different tasks, construct training prompts spanning both verifiable domains and non-verifiable domains, allowing the model to learn when tool invocation is beneficial and when pure reasoning suffices.
2. Judgment flexibility: To accommodate different input/output formats, diversify the evaluation tasks to cover pointwise, pairwise, and listwise ranking, ensuring broad applicability across practical use cases.
3. Data efficiency: Demonstrate that TIR-Judge can bootstrap from the initial checkpoint without distillation. TIR-Judge-Zero trains purely with iterative reinforcement learning for achieving self-improvement.

## 4. Core Innovation Points

1. **First Tool-Integrated RL Judge Framework:** Introduction of TIR-Judge, the first tool-integrated framework for training LLM-based judges with end-to-end multi-turn reinforcement learning. To the best of our knowledge, this is the first approach that jointly optimizes reasoning and tool-use for training LLM-based judges via RL.
2. **Key Strategies to Exploit RL Power:** Design of several key strategies to fully exploit the power of reinforcement learning, including task diversification across verifiable and non-verifiable domains, flexible judgment formats (pointwise, pairwise, listwise), as well as an iterative RL scheme that enables self-improvement in tool use even without distillation.
3. **Strong Parameter Efficiency and Performance:** TIR-Judge shows strong parameter efficiency: With only 8B parameters, it surpasses the 32B reasoning reward models on the PPE dataset, and reaches 96% of the performance of Claude-Opus-4 in the listwise setting in RewardBench 2. It consistently outperforms strong reasoning-based judges, achieving gains of up to 6.4% (pointwise) and 7.7% (pairwise).
4. **Self-Improvement via Pure RL (TIR-Judge-Zero):** TIR-Judge-Zero, the judge trained without any distillation, achieves a 1.2% gain over its distilled counterpart at 4B scale, highlighting the power of RL to bootstrap reasoning and tool-use capabilities entirely from a base model and offering a scalable path toward self-improving judges.

## 5. Overview of the Overall Technical Solution

The overall technical solution of TIR-Judge consists of four main components: (1) data collection and filtering for RL, (2) the RL framework for training judges with integrated code execution tools, (3) reward design for RL, and (4) cold-start and iterative training strategies in RL.

The problem setup considers the task of LLM-based judgment across three evaluation settings: Pointwise evaluation (assigning a scalar score), Pairwise evaluation (selecting the preferred response), and Listwise evaluation (returning the index of the best response). The judge is extended with the ability to call an external Python execution environment. The judgment trajectory interleaves reasoning, code execution, and tool feedback, enabling the judge to ground its decision in verifiable evidence. 

The RL training adopts DAPO, an improved variant of GRPO, utilizing a clipped policy gradient objective with KL divergence penalty. The training utilizes a diverse dataset comprising real preference pairs and synthetic preference pairs, totaling approximately 26k preference pairs. The reward design is structured covering three aspects: Correctness Reward, Format Reward, and Tool-Specific Reward. Two training strategies are introduced: TIR-Judge-Distill (leverages a stronger teacher for SFT initialization) and TIR-Judge-Zero (alternates between RL, rejection sampling, and SFT for self-bootstrapping).

## 6. Detailed Module Design

### 6.1 Data Collection and Filtering Module
High-quality training data is curated as a collection of (prompt, responses) tuples spanning multiple tasks.
*   **Real Preference Pairs:** Sampled from human-labeled preference pairs across domains: general helpfulness (HelpSteer 3), reasoning (UltraInteract, S1), coding (CodeRM), instruction following (preference pairs from Tulu 3), and safety (Safe-RLHF). Each prompt is paired with one preferred (chosen) response and one or more rejected responses.
*   **Synthetic Preference Pairs:** Augmented from verifiable prompts. For each prompt, responses are sampled from multiple open-source models (Qwen3-8B/14B, Gemma-2-9B, Gemma-3-12B). The responses are automatically evaluated against verifiable functions (for IF tasks using Tulu-3) or ground-truth solutions (for reasoning tasks using MATH, DAPO-Math, WebInstruct, and Loong).
*   **Filtering:** Strict 8-gram decontamination is applied to eliminate any overlap between training prompts and evaluation benchmarks. For listwise data, 3–5 negatives per instance are sampled and enforced to yield different final answers from the positive. For HelpSteer3, only examples where one response is explicitly annotated as better or significantly better are retained.

### 6.2 Tool-Integrated RL Framework Module
*   **RL Algorithm:** DAPO (variant of GRPO). Given a prompt-answer pair (q, a), a group of G rollouts is sampled from the current policy. Each rollout is assigned a scalar reward with access to the oracle answer. The policy is updated with a clipped policy gradient objective.
*   **Error Message Processing:** Outputs from Interpreter I are truncated to only the final error line to avoid excessive context length while preserving useful feedback in the trajectory.
*   **Sandbox Output Masking:** Since execution results may cause the model to overfit by memorizing outputs, execution results are masked during loss computation. This prevents reliance on exact strings and improves training stability.

### 6.3 Reward Design Module
A structured reward covering three aspects:
*   **Correctness Reward $R_c$:** Measures whether the judge’s prediction aligns with the reference preference label. $R_c = 1$ if the judge’s decision matches the ground-truth preference, and $R_c = 0$ otherwise (incorrect predictions or parsing errors).
*   **Format Reward $R_f$:** Ensures the judge strictly follows a predefined structured output format. Prediction scores must be in `<score></score>` tags, preference labels in `<preference></preference>` tags, and code in ` ```python ``` `. For safety and general helpfulness prompts, a positive format reward is granted only if the model produces a valid output without invoking tools. $R_f = 1$ if all formatting constraints are satisfied, else $0$.
*   **Tool-Specific Reward $R_t$:** Encourages accurate and efficient tool use by penalizing errors or excessive executions. Max number of tool calls per trajectory is 3. $R_t = 1$ only when code blocks are error-free and within the call budget; otherwise $0$.
*   **Combined Reward $R$:** Assigns full credit only when correctness, format, and tool-use are all satisfied.

### 6.4 Training Strategies Module
*   **TIR-Judge-Distill:** Leverages a stronger teacher (Gemini-2.5-Flash with code execution) to generate high-quality trajectories via rejection sampling. Only trajectories producing correct answers are retained for SFT dataset. Interpreter feedback tokens are masked to prevent learning on execution results. In total, about 10k tool-integrated trajectories are collected for SFT, serving as initialization before RL.
*   **TIR-Judge-Zero:** Investigates self-bootstrapping without distillation. The process alternates between RL, rejection sampling, and supervised fine-tuning. Starting from the initial model, obtain checkpoint via direct RL. Then, sample multiple trajectories (G=4) from the current policy. Retain only valid trajectories (correct answer, valid format, no interpreter errors). For efficiency, keep only one trajectory per prompt, preferring the shortest response or the one with the fewest tool calls. The dataset is then used for SFT, and the fine-tuned model initializes the next RL round. After each cycle, select the best checkpoint based on held-out validation accuracy and repeat the loop.

### 6.5 Implementation Details
*   **Backbones:** Qwen3-8B and Qwen3-4B-Instruct-2507 (without enabling thinking mode). Training implemented with Verl-Tool.
*   **SFT Settings:** Batch size 64, learning rate 2e-6, context length 8192, for 1 epoch.
*   **RL Settings:** Micro batchsize per gpu 4, mini batchsize 128, number of rollout 8. $\epsilon_{low} = 0.2, \epsilon_{high} = 0.3, \beta = 0.01$, max response length 8192, learning rate 1e-6, trained for 2 epochs.
*   **Hardware:** 8 NVIDIA H100 80G GPUs.
*   **Data Collection Settings:** Generate 2 rollouts for each model with $t = 0.9, p = 0.95$. No external feedback used.
*   **Inference Settings:** $t = 0$ for generating responses.

## 7. All Mathematical Formulas and Symbol Definitions

**Problem Setup Definitions:**
*   $x \in \mathcal{X}$: User prompt.
*   $Y = \{y_1, y_2, \ldots, y_n\}$: Model-generated responses.
*   $J_\theta$: Judge model parameterized by $\theta$.
*   Pointwise evaluation: $J_\theta(x, y) = s_\theta(x, y) \in \mathbb{R}$
*   Pairwise evaluation: $J_\theta(x, y_a, y_b) = \text{argmax}_{i \in \{a,b\}} s_\theta(x, y_i)$, where $s_\theta$ denotes a learned scoring function.
*   Listwise evaluation: $J_\theta(x, Y) = \text{argmax}_{i \in \{1,...,n\}} s_\theta(x, y_i)$.

**Tool-Augmented Judge Trajectory:**
*   $\mathcal{I}$: External Python execution environment.
*   $s_k$: Judgment trajectory at step $k$, represented as $s_k = \{r_1, c_1, o_1, \ldots, r_k, c_k, o_k\}$, where $r_i$ is a natural language reasoning step, $c_i$ is generated code, and $o_i = \mathcal{I}(c_i)$ is the execution result.
*   Iterative process definition:
    $$(r_k, c_k) \sim J(x \oplus s_{k-1}), \quad o_k = \mathcal{I}(c_k), \quad s_k = s_{k-1} \oplus r_k \oplus c_k \oplus o_k$$
*   Final prediction: $a_i \sim J(x \oplus s_T)$ with $T$ being the final step.

**Tool-Integrated RL Objective (DAPO):**
*   $\pi_\theta$: Policy parameterized by $\theta$.
*   $G$: Number of rollouts sampled from $\pi_{\theta_{old}}$.
*   $R_i = R(s_i, a)$: Scalar reward for rollout $s_i$ with oracle answer $a$.
*   $r_{i,t}(\theta) = \frac{\pi_\theta(s_{i,t}|q, s_{i,<t})}{\pi_{\theta_{old}}(s_{i,t}|q, s_{i,<t})}$: Token-level importance weight.
*   $\hat{A}_{i,t} = \frac{R_i - \text{mean}(\{R_i\}_{i=1}^G)}{\text{std}(\{R_i\}_{i=1}^G)}$: Advantage at the token level.
*   $\epsilon_{low}, \epsilon_{high}$: Clipping range hyperparameters.
*   $\beta$: KL divergence penalty coefficient.
*   Objective function:
    $$J(\theta) = E_{(q,a) \sim D, \{s_i\}_{i=1}^G \sim \pi_{\theta_{old}}(\cdot|q)} \left[ \frac{1}{\sum_{i=1}^G |s_i|} \sum_{i=1}^G \sum_{t=1}^{|s_i|} \left( \min(r_{i,t}(\theta)\hat{A}_{i,t}, \text{clip}(r_{i,t}(\theta), 1-\epsilon_{low}, 1+\epsilon_{high})\hat{A}_{i,t}) - \beta D_{KL}(\pi_\theta \| \pi_{ref}) \right) \right]$$
    subject to: $0 < |\{s_i : \text{is\_equivalent}(a, s_i)\}| < G$

**Reward Design Formulas:**
*   **Correctness Reward $R_c$:**
    $$R_c = \begin{cases} \mathbb{I}(s_\theta(x, y_{pos}) > s_\theta(x, y_{neg})) , & \text{for pointwise evaluation}, \\ \mathbb{I}(J_\theta(x, Y) = l) , & \text{for pairwise or listwise evaluation}, \\ 0, & \text{otherwise}, \end{cases}$$
    where $\mathbb{I}(\cdot)$ is the indicator function, $l$ is the ground-truth preferred response.
*   **Format Reward $R_f$:** $R_f = 1$ if output satisfies all formatting constraints (and no-tool heuristic when applicable), $0$ otherwise.
*   **Tool-Specific Reward $R_t$:** $R_t = 1$ if code blocks $c_i$ are error-free and within call budget, $0$ otherwise.
*   **Final Reward $R$:**
    $$R = \begin{cases} 1, & \text{if } R_t = 1 \land R_f = 1 \land R_c = 1, \\ 0.1, & \text{if } R_c = 1 \text{ but } (R_t = 0 \lor R_f = 0), \\ 0, & \text{if } R_c = 0. \end{cases}$$

**SFT Loss Function (for TIR-Judge-Distill):**
*   $T_{SFT} = \{(x, Y, s, a) \mid R(s, a, l) = 1\}$: Dataset of correct trajectories from teacher.
*   $\tau = (s, a)$: Target trajectory with reasoning and code steps.
*   Objective:
    $$\mathcal{L}_{SFT} = -E_{(x,\tau) \sim T_{SFT}} \left[ \sum_{i=1}^{|y|} \log f_\theta(\tau_i | \tau_{<i}, x) \right]$$

**Iterative Training (TIR-Judge-Zero) Update Rules:**
*   $\pi_{\theta_1} \leftarrow RL(\pi_{\theta_0})$
*   $T_t = \{(x, s, a) \mid R(s, a, l) = 1\}$
*   $T_{t+1} \leftarrow RS(\pi_{\theta_t})$
*   $\pi_{\theta_{t+1}} \leftarrow SFT(\pi_{\theta_0}, T_{t+1})$
*   $\pi_{\theta_{t+1}} \leftarrow RL(\pi_{\theta_{t+1}})$

## 8. Algorithm Pseudocode

**Iterative Training without Distillation (TIR-Judge-Zero)**

1: Initialize base model $\pi_{\theta_0}$
2: **Stage 1: Initial RL**
3: $\pi_{\theta_1} \leftarrow RL(\pi_{\theta_0})$ using training data (Sec. 4.2)
4: Set current policy $\pi_{\theta_t} \leftarrow \pi_{\theta_1}$
5: **Stage 2: Iterative RS $\to$ SFT $\to$ RL Loop**
6: **while** not converged **do**
7: $\quad$ **Rejection Sampling (RS):**
8: $\quad$ For each prompt $x$, sample multiple trajectories $\{s_i\}_{i=1}^G \sim \pi_{\theta_t}(\cdot | x)$ where $G=4$
9: $\quad$ $\quad$ Each trajectory $s_i = \{r_1, c_1, o_1, \ldots, r_k, c_k, o_k\}$
10: $\quad$ Retain only valid trajectories that:
11: $\quad$ $\quad$ (i) produce the correct answer $l$
12: $\quad$ $\quad$ (ii) satisfy the output format
13: $\quad$ $\quad$ (iii) execute without interpreter errors
14: $\quad$ $\quad$ $\to$ Form dataset $T_t = \{(x, s, a) \mid R(s, a, l) = 1\}$
15: $\quad$ For each prompt $x$, keep only one trajectory (prefer shortest response or fewest tool calls)
16: $\quad$ **Supervised Fine-Tuning (SFT):**
17: $\quad$ $\pi_{interim} \leftarrow SFT(\pi_{\theta_0}, T_{t+1})$
18: $\quad$ **Reinforcement Learning (RL):**
19: $\quad$ $\pi_{\theta_{t+1}} \leftarrow RL(\pi_{interim})$
20: $\quad$ Select best checkpoint $\pi_{\theta_{t+1}}$ based on held-out validation accuracy
21: $\quad$ Update current policy $\pi_{\theta_t} \leftarrow \pi_{\theta_{t+1}}$
22: **end while**
23: Return final model $\pi_{\theta_t}$