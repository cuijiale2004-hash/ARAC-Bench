1. Research Background and Existing Pain Points
Post-training Multimodal Large Language Models (MLLMs) to build interactive agents holds promise across domains such as computer-use, web navigation, and robotics. However, a key challenge in scaling such post-training is the lack of high-quality downstream agentic task datasets with tasks that are diverse, feasible, and verifiable. Relevant interaction data resides on closed systems such as personal devices and commercial hardware, making it not readily available at web-scale. Existing approaches for task generation rely heavily on human annotation, which is prohibitively expensive and scales poorly. Alternatively, methods that prompt MLLMs with limited downstream environment information yield tasks with limited coverage and offer no guarantees of feasibility or verifiability, as current MLLMs lack explicit grounding in environment dynamics and current state information. Methods that iteratively propose and execute short-horizon subtasks are constrained by previous subtask states, progressively limiting the space of exploratory trajectories and resulting in less diverse and easier tasks. Methods without exploration suffer from hallucination since they only rely on limited functionality descriptions rather than the actual states of the environment.

2. Core Research Motivation and Scientific Questions
The core research motivation is that the fundamental bottleneck to enable scalable post-training of agentic MLLMs is the lack of large high-quality task definition datasets with tasks that are diverse, feasible to execute, verifiable, and aligned with real-world use cases. With access to such task datasets, MLLMs could be post-trained in a scalable manner with synthetically generated demonstrations using supervised finetuning (SFT) or via Reinforcement Learning (RL) without any human annotation. The central scientific question is how to build an automatic and reliable pipeline that synthesizes tasks at scale with broad coverage, feasibility, and verifiability. This requires explicit knowledge of current state information and what interactions are feasible in an environment—knowledge that can only be obtained through direct interaction with the environment. The paper seeks to answer how to actively explore the agent's environment exhaustively to collect relevant information to ground MLLMs for scalable task generation without human supervision.

3. Overall Core Idea and Design Philosophy
The overall core idea is to introduce AUTOPLAY, a scalable pipeline for task generation that explicitly explores interactive environments to discover possible interactions and current state information to synthesize environment-grounded tasks. The design philosophy is that actively exploring the agent's environment in an exhaustive manner is necessary to collect relevant information that can be used to ground MLLMs in the environment for scalable task generation. AUTOPLAY incorporates two phases: (i) an exploration phase, where an MLLM explorer agent equipped with memory systematically uncovers novel environment states and functionalities, and (ii) a task generation phase, where a task generator MLLM leverages exploration trajectories and a set of task guideline prompts as context to synthesize diverse, executable, and verifiable tasks. The tasks generated must be verifiable to filter high-quality trajectories for SFT or to build a reward model for RL.

4. Core Innovation Points
1. Explicit Environment Exploration with Memory: Unlike prior works relying on limited domain knowledge or static environment context, AUTOPLAY actively explores the environment using a goal-agnostic explorer agent equipped with explicit memory of past interactions to maximize coverage over novel states and functionalities of the environment.
2. Exploration-Conditioned Task Generation with Guidelines: AUTOPLAY uses exploration trajectories as environment context rather than successful task executions, uncovering underlying information of the transition function and state space. It introduces domain-specific task guideline prompts (e.g., Feature-Use, Information Retrieval, Feature Composition, Subtask Repetition) to steer the task generator towards diverse categories and good domain coverage.
3. Scalable Demonstration Synthesis and Verification without Human Annotation: AUTOPLAY enables large-scale task demonstration synthesis by employing an MLLM task executor and verifier. The verifier evaluates trajectories using a Chain-of-Thought reasoning process to issue a final judgment of success or failure, enabling the creation of high-quality SFT datasets without relying on privileged environment information or human annotation.
4. Enabling Reinforcement Learning with Verifier-based Rewards: AUTOPLAY generated tasks combined with MLLM verifier-based rewards enable scaling reinforcement learning training of UI agents. By scoring trajectories with the task verifier as a binary reward (1 or 0), AUTOPLAY unlocks RL training with verifier feedback without requiring any human annotation.
5. Modular Task Executor Agent Architecture: The task executor is instantiated as a modular agent decomposing execution into high-level task planning using an MLLM planner, a reflection tool for transition summaries, and a low-level grounding model for pixel-level coordinate translation, improving data collection success rates.

5. Overview of the Overall Technical Solution
AUTOPLAY is defined in the context of multi-step decision making domains expressed as Partially Observable Markov Decision Processes (POMDP). The overall technical solution operates in two main stages: Environment Exploration and Task Generation. In Stage 1, a goal-agnostic explorer agent interacts with the environment for $K$ steps to produce an exploration trajectory. This process is repeated $M$ times, where an MLLM summarizer summarizes prior exploration trajectories into concise text representations to serve as episodic memory, encouraging the explorer to cover novel states. The full set of exploration trajectories forms the environment context $E$. In Stage 2, each environment context is combined with a task guideline prompt from a set $P$. A task generator MLLM uses this context and the guidelines to produce a sequence of goals, which are paired with the initial state of the trajectory to form the task dataset $D$. Subsequently, an MLLM task executor agent attempts to execute these tasks, and an MLLM task verifier evaluates the trajectories to filter successful executions for SFT. Finally, the generated tasks and verifier are used to provide rewards for RL training.

6. Detailed Module Design
1. Explorer Agent: Implemented using an MLLM agent equipped with an explicit memory of past interactions. At each timestep, the explorer agent is provided with the current environment observation, past interaction memory, and prompted to select diverse actions. It interacts for $K$ steps to produce an exploration trajectory. The prompt is a generic exploration goal: "Explore the APP_NAME app exhaustively to access all features, functionalities and data stored on the app."
2. Summarizer MLLM: Used to convert a high-dimensional exploration trajectory into a text representation for episodic memory. It takes the sequence of observations from the exploration trajectory and outputs a structured response describing the functionalities of the environment observed and the data/configs stored that would be relevant for task curation. Formally, it generates $m = \text{MLLM}(\text{summary\_prompt}, \tau)$. The memory $M$ is updated as $M = M \cup \{m\}$.
3. Task Generator MLLM: Uses the exploration context $\tau \in E$ and a task guideline prompt $p \in P$ to produce a sequence of goals. The environment context $\tau$ serves as context of what interactions are feasible and the current state of the environment, uncovering underlying information of the POMDP transition function $T$ and state space $S$. The task generator outputs a sequence of goals: $(g_1, \cdots, g_N) \sim \text{MLLM}(\text{task\_generator\_prompt}, p, \tau)$. Each goal paired with the initial state of $\tau$ results in the task dataset $D = D \cup \{(g_1, s_1), \ldots, (g_K, s_1)\}$.
4. Task Guideline Prompts: Domain-specific prompts tailored to the target domain. For the UI domain, these include: (1) Feature-Use: tasks using basic features and providing broad coverage over all features shown; (2) Feature-Composition: tasks composing multiple feature-use tasks to create complex tasks with multiple subtasks; (3) Information Retrieval: tasks requiring searching for specific information or answering queries about the state of the environment; (4) Feature-Repetition: tasks requiring executing feature-use tasks over multiple entities stored in an application.
5. Task Executor Agent: A modular agent consisting of an MLLM planner, a reflection tool, and a grounding model. Given the task instruction, current screenshot, previously executed actions, and a reflection trace of the last action, the MLLM planner generates a high-level action in natural language. If the action is coordinate-based, the grounding model translates it into a pixel-level coordinate. The reflection tool takes the previous action, prior observation, and current observation to generate a reflection trace describing the effect of the action and whether it was executed successfully.
6. Task Trajectory Verifier: An MLLM that takes the task instruction and executed trajectory (interleaved images and actions). It operates in three stages: (i) summarizing what the agent is doing in the trajectory (screen_details); (ii) producing a Chain-of-Thought reasoning for whether the task has been completed; (iii) issuing a final judgment of "success" or "failure" (result). It labels a trajectory as successful only if the screenshots show the intended task being achieved without unintended changes.

7. All Mathematical Formulas and Symbol Definitions
POMDP Definition: A POMDP is defined as a tuple $(S, O, A, P, R, \rho_0)$.
$S$: Underlying state space.
$O$: Observation space.
$A$: Action space.
$P: S \times A \rightarrow \Delta(S)$: Transition function.
$R$: Reward function.
$\rho_0$: Initial state distribution.
$G$: Goal distribution.
$R(s, g)$: Reward formulated for $s \in S$ and $g \in G$.
$\pi(a_t|o_t, g)$: Goal-conditioned policy mapping from observation $o_t$ at timestep $t$ and goal $g$ to an action $a_t$ to achieve the goal state $s_g$.
$D$: Dataset of tasks consisting of tuples $(g, s_0)$ where $g$ is a goal-specification in natural language, and $s_0$ is the initial state of the environment.
$\tau = (o_0, a_0, \ldots, o_T)$: Trajectory defined as a sequence where the outcome of achieving goal $g$ is verified by a reward model.
$R(\tau, g)$: Reward model relying only on the trajectory without privileged access to the environment state.
$E = \{\tau_1, \cdots, \tau_M\}$: Full environment context obtained at the end of Stage 1.
$m = \text{MLLM}(\text{summary\_prompt}, \tau)$: Concise and comprehensive text representation describing exploration trajectory $\tau$ generated using a summarizer MLLM.
$P = \{p_1, \ldots, p_N\}$: Set of task guidelines tailored to the target domain.
$(g_1, \cdots, g_N) \sim \text{MLLM}(\text{task\_generator\_prompt}, p, \tau)$: Sequence of goals sampled using the task generator MLLM conditioned on the task generator prompt, task guideline prompt $p$, and trajectory $\tau$.
$D = D \cup \{(g_1, s_1), \ldots, (g_K, s_1)\}$: Update rule for the task dataset where $s_1$ is the first state in $\tau$.

8. Algorithm Pseudocode
Algorithm 1 AUTOPLAY
Stage 1: Environment Exploration
Parameters:
N # of apps
M : # of exploration turns
E = ∅
for j = 1 . . . N do
Sample s0 ∼ S, Initialize context M = ∅
for k = 1 . . .M do
Sample trajectory τ using MLLM(M, explorer_prompt)
Summarize as m = MLLM(τ, summary_prompt)
E = E ∪ {τ}, M = M ∪ {m}
end for
end for

Stage 2: Task Generation
Parameters:
P : task guidelines
K # of tasks per guideline and context
D = ∅
for s = 1 . . . S do
Sample p ∼ P, τ ∼ E
Sample (g1, . . . , gK) using
MLLM(task_generator_prompt, p, τ, )
D = D ∪ {(g1, s1), . . . , (gK , s1)}
where s1 is the first state in τ
end for
return D