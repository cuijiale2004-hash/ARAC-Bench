1. Research Background and Existing Pain Points
Parallel thinking has emerged as a novel approach for enhancing the reasoning capabilities of large language models (LLMs) by exploring multiple reasoning paths concurrently. In cognitive science, humans depend on such parallel exploration, considering multiple possibilities before converging on coherent conclusions, which fosters divergent thought and reduces the risk of premature closure to suboptimal solutions. Moreover, parallel thinking leverages the concurrent execution capability of modern GPUs. However, activating such capabilities through training remains challenging due to the following pain points:
1) The sequential nature of current LLMs inherently constrains this ability. While test-time strategies can elicit such behavior, they only work during inference without internalizing the capability.
2) Existing training-based approaches based on supervised fine-tuning (SFT) depend on costly pipelines to curate high-quality trajectories, rarely yield performance gains beyond faster inference, and achieve parallelism via superficial teacher-forcing rather than free exploration, severely limiting the generalization of underlying parallel thinking strategies.
3) Although reinforcement learning (RL) offers a more scalable approach, current autoregressive LLMs are not exposed to parallel thinking behaviors during pre-training or SFT, leading to a severe cold-start problem where models are unable to produce such trajectories natively.
4) The optimal reward function for reinforcing parallel thinking with RL remains an open question. Relying solely on outcome-based rewards drives models toward shortcuts bypassing genuine parallel thinking, whereas enforcing structure-based rewards may lead to unnecessary parallelism.
5) The strategic role and underlying mechanisms of parallel thinking remain insufficiently understood, limiting the ability to fully harness their potential.

2. Core Research Motivation and Scientific Questions
The core research motivation is to permanently instill parallel thinking capabilities into LLMs through training, moving beyond inference-time tricks and superficial teacher-forcing. The work aims to investigate how to effectively incorporate parallel thinking into LLMs via reinforcement learning to solve complex real-world reasoning tasks.
The scientific questions addressed are:
1) How to address the cold-start problem in training parallel thinking with RL, given the lack of native parallel thinking data for complex tasks?
2) What is the optimal reward design to balance task performance and the activation of parallel thinking behaviors?
3) How do reasoning behaviors evolve and benefit training, and what is the underlying mechanism of parallel thinking during RL training?
4) Can parallel thinking serve as a strategic exploration scaffold during mid-training to unlock higher performance ceilings?

3. Overall Core Idea and Design Philosophy
The overall core idea is to bootstrap parallel thinking with simple tasks to generate cold-start data and then use RL to transfer and generalize this capability to harder problems, thus avoiding complex data generation for challenging cases. The design philosophy separates the learning of format, behavior, and core reasoning into distinct stages through a progressive training curriculum. It posits that RL-induced parallel thinking is more generic and promising than SFT as it uncovers novel, highly adaptive reasoning behaviors. Furthermore, the framework investigates two complementary settings: one without architectural modifications (autoregressive) and another with explicit architectural changes (Multiverse) to enforce path isolation. A key design philosophy is that parallel thinking can act as a mid-training exploration scaffold, where a temporary forced-exploration phase helps the model reach a stronger final policy.

4. Core Innovation Points
1) Proposing Parallel-R1, the first reinforcement learning (RL) framework that instills parallel thinking for complex real-world reasoning tasks (e.g., mathematical reasoning) beyond synthetic tasks.
2) Designing a progressive training curriculum enabled by a lightweight data pipeline that successfully bootstraps this complex skill by separating the learning of format (SFT on easy math), behavior (RL on easy math), and core reasoning (RL on general math) into distinct stages, addressing the cold-start problem.
3) Proposing and investigating multiple reward schemes, finding an effective alternating reward strategy for structured models that switches between an outcome-based reward and a reward encouraging parallel thinking behaviors within fixed windows, achieving a superior balance.
4) Delving deep into the learning dynamics to reveal that the target of parallel thinking evolves from exploration in early stages to verification in later stages, identifying a strategic shift towards risk-averse behavior.
5) Identifying the concept of parallel thinking as a mid-training exploration scaffold, which empirically boosts performance by unlocking a higher performance ceiling after RL.

5. Overview of the Overall Technical Solution
The overall technical solution, Parallel-R1, is a framework to learn parallel thinking via reinforcement learning. It formulates parallel thinking as a two-stage process (Exploration and Summary) using specific control tags. It utilizes a progressive curriculum:
- Stage 1 (Cold-Start): SFT on the curated Parallel-GSM8K dataset (generated by prompting LLMs on easy math) to instill basic format knowledge.
- Stage 2 (RL on Easy Math): Optional small-scale RL on GSM8K to stabilize format and behavior.
- Stage 3 (RL on General Math): Large-scale RL on the DAPO dataset to generalize the capability to complex problems.
The framework explores two variants:
- Parallel-Seen: Autoregressive models without architectural modifications.
- Parallel-Unseen: Multiverse models with explicit inductive biases (customized attention masks and position ids) to enforce path isolation, requiring a different training recipe and alternating reward schedule.

6. Detailed Module Design
6.1 Formulation of Parallel Thinking Behaviors
Parallel thinking is formalized as a two-stage process:
1) Exploration: When the model identifies a critical step, it temporarily suspends the main path and launches a multi-branch search, generating N independent reasoning branches simultaneously.
2) Summary: After exploration, the model aggregates the outcomes from all branches, extracts their unique key insights, and resolves conflicts to arrive at the most promising final conclusion. Subsequently, it automatically resumes the main reasoning path and inputs this summarized conclusion.
To realize this, three control tags are introduced: <Parallel>...</Parallel>, <Path>...</Path>, and <Summary>...</Summary>.
Workflow at Inference Phase: The model begins with autoregressive generation. When a <Parallel> token is predicted, it pauses the main reasoning thread and concurrently generates multiple reasoning threads within separate <Path>...</Path> blocks. Once completed, outputs are consolidated into a <Summary>...</Summary> block. The summarized context resumes the main reasoning thread.

6.2 The Lightweight Data Pipeline for Parallel Thinking Cold-Start
Since collecting high-quality parallel thinking data is challenging, the pipeline leverages the observation that lightweight prompting is highly effective for simpler tasks (GSM8K) but not complex ones (DAPO).
- Generator: DeepSeek-R1-0528-Qwen-3-8B.
- Seed Data: 7,473 samples of the GSM8K training set.
- Process: Prompting LLMs with detailed instructions to generate high-quality parallel thinking trajectories. Extract the non-thinking parts to serve as gold annotations, creating the Parallel-GSM8K dataset.
- Filtering: A Parallel Thinking Format Check (Algorithm 1) is performed to ensure strict adherence to the format for the structured model variant.

6.3 Learning Parallel Thinking via RL in Autoregressive Models (Parallel-Seen)
This module explores strategies without modifying the model architecture, using Group Relative Policy Optimization (GRPO).
Curriculum Training and Reward Modeling:
- Cold-Start Stage: Apply SFT on Parallel-GSM8K to initialize the RL actor with basic format capability.
- RL on Easy Math: Perform small-scale RL on GSM8K using GRPO to enhance format learning. Final reward format: Rfinal = R⟨Parallel⟩ & Racc. Accuracy Reward (Racc) evaluates correctness; Parallel Reward (R⟨Parallel⟩) incentivizes parallel thinking. Binary strict reward: +1 if output contains at least one parallel thinking unit AND final answer is correct; -1 otherwise.
- RL on General Math: Apply RL to general math datasets (DAPO) to generalize ability. Sole reward is accuracy reward (Racc).

6.4 Eliciting Parallel Thinking via RL in Multiverse Models (Parallel-Unseen)
This structured variant incorporates explicit inductive biases into the attention mechanism to enforce path isolation.
6.4.1 Multiverse Attention Mechanism
- Attention masks: Restricts tokens within a <Path> block to attend only to their own path and the shared context, preventing information leakage across sibling paths.
- Shared position encodings: Allocates each path a disjoint set of position indices, ensuring parallel paths can begin decoding from identical positions without interference.
6.4.2 The Training Recipe and Reward Modeling
Directly applying the progressive curriculum proves ineffective due to poor generalization of attention masks from easy to hard problems. Stage 1 RL is removed, and the reward schedule is redesigned:
- (S1) Accuracy-only: Optimizes solely for correctness without encouraging parallel thinking.
- (S2) Alternating accuracy and parallel: Alternates between two different reward functions within fixed windows of W=10 steps. For 80% of steps, uses standard accuracy-only reward (Racc). For the remaining 20%, uses a tiered reward system: (i) +1.2: If generated output contains at least one parallel thinking unit AND final answer is correct; (ii) +1.0: If generated output does not contain parallel thinking unit AND final answer is correct; (iii) -1.0: For all other cases with incorrect answers.

6.5 Parallel Thinking as a Mid-Training Exploration Strategy
A two-stage training curriculum designed to use parallel thinking as an exploration scaffold:
- Stage-1 (Exploration Phase, before 200 steps): Maximize exploration using the training approach of Parallel-R1-Unseen (S2) to ensure a high parallel ratio and encourage exploration of a wide breadth of reasoning paths.
- Stage-2 (Exploitation Phase, after 200 steps): Switch focus to exploitation. Training objective switched to optimize for accuracy alone, allowing the model to exploit effective strategies discovered during exploration.

7. All Mathematical Formulas and Symbol Definitions
GRPO Algorithm Formulation:
Let q be a question, and let {oi}_{i=1}^G be G candidate responses sampled from the old policy πθold(· | q). We denote ri as the reward for oi. We define:
ρi = \frac{πθ(oi | q)}{πθold(oi | q)}, \bar{r} = \frac{1}{G}\sum_{j=1}^G rj, Ai = \frac{ri - \bar{r}}{\sqrt{\frac{1}{G}\sum_{j=1}^G (rj - \bar{r})^2 + εstab}},
where εstab is a constant for numerical stability and Ai is the advantage. The GRPO loss is then:
LGRPO(θ) = E q∼D {oi}∼πθold [\frac{1}{G}\sum_{i=1}^G min(ρiAi, clip(ρi, 1− α, 1 + α)Ai) − β DKL(πθ ∥πref)].

Reward Definitions:
Autoregressive RL on Easy Math:
Rfinal = R⟨Parallel⟩ & Racc
(+1 if output contains at least one parallel thinking unit AND final answer is correct; -1 otherwise).

Multiverse Alternating Reward (S2):
Window W=10 steps.
80% of steps: Racc
20% of steps: Tiered Reward
(i) +1.2: If the generated output contains at least one parallel thinking unit AND the final answer is correct;
(ii) +1.0: If the generated output does not contain a parallel thinking unit AND the final answer is correct;
(iii) -1.0: For all other cases with incorrect answers.

8. Algorithm Pseudocode
Algorithm 1 Parallel Thinking Format Check
Input: tokens – list of tokens from the parallel-thinking trace;
tag pairs – set of valid (opening, closing) tag pairs, e.g. {(<Path>...</Path>), . . .}
Output: format valid – boolean indicating whether the trace is well-formed
1: S ← ∅
2: format valid ← true
3: for all t in tokens do
4:   if t is an opening tag then
5:     push t onto S
6:   else if t is a closing tag then
7:     if S is empty then
8:       format valid ← false
9:       break
10:    end if
11:    top tag ← Top(S)
12:    if (top tag , t) ∈ tag pairs then
13:      pop S
14:    else
15:      format valid ← false
16:      break
17:    end if
18:  end if
19: end for
20: if format valid and S ̸= ∅ then
21:   format valid ← false
22: end if
23: return format valid