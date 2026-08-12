## 1. Research Background and Existing Pain Points

**Research Background:** 
The development of autonomous agents for complex, long-horizon tasks is a central goal in AI, significantly propelled by the rise of Large Language Models (LLMs). Current dominant training paradigms for these agents primarily fall into two categories: Supervised Fine-Tuning (SFT) on expert trajectories and Reinforcement Learning (RL) from environmental feedback.

**Existing Pain Points:**
1. **SFT creates efficient but brittle policies that fail to generalize:** Supervised Fine-Tuning on expert trajectories can teach agents efficient behaviors on tasks seen during training, but these policies are brittle and fail to generalize to novel situations. When faced with novel situations (e.g., unseen task categories in L2 splits), the agent falls into non-productive loops, demonstrating that SFT teaches mimicry without instilling a generalizable reasoning process.
2. **Outcome-only RL fosters inefficient, flawed reasoning:** Reinforcement learning methods that optimize solely for final task success (sparse outcome rewards) often reinforce flawed or inefficient reasoning paths, a problem termed "inefficient exploration." This leads to agents that are brittle and fail to generalize, as they learn to find solutions without learning how to reason coherently. Standard RL inadvertently reinforces any successful trajectory, failing to distinguish between robust and flawed reasoning processes. Even when an agent successfully completes a task, its path to a solution is often littered with redundant or illogical steps (high invalid and repetitive action rates).
3. **Scaling model size does not fix the underlying reasoning deficiencies:** While scaling from a smaller to a larger model improves overall success rates, it does not resolve the inefficient exploration issue. A larger model’s enhanced capacity can be misdirected to more effectively exploit flawed strategies rather than to reason more coherently. The limitation is rooted in the training objective itself, not merely model capacity.
4. **Fundamental trade-off in current paradigms:** Current paradigms force a trade-off between brittle efficiency (SFT) and inefficient generalization (Outcome-only RL). Neither paradigm effectively teaches the agent how to reason well.

## 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:**
To build truly robust and generalizable agents, we must move beyond rewarding only the final outcome and begin to supervise the reasoning process itself. Inspired by metacognitive theory, which posits that effective problem-solving depends on “thinking about thinking”, the motivation is to directly reward beneficial cognitive behaviors. High-level skills like planning, monitoring progress, exploring alternatives, and reflecting on errors can be operationalized as distinct, verifiable steps within an agent’s reasoning process to mitigate inefficient exploration.

**Scientific Questions:**
Are agents learning to reason coherently, or are they just finding brittle shortcuts to success? How can we provide direct, process-level supervision to guide agents to not only find solutions but to do so robustly and intelligently, thereby resolving the pervasive "inefficient exploration" issue where agents are rewarded for successful outcomes even when their path is built on flawed or redundant reasoning?

## 3. Overall Core Idea and Design Philosophy

**Overall Core Idea:**
The paper introduces Reinforcement Learning with Verifiable Meta-Reasoning Rewards (RLVMR), a novel framework that integrates dense, process-level supervision into end-to-end RL. RLVMR contrasts with standard RL by rewarding not only the final outcome but also the intermediate reasoning steps. The framework defines a set of core meta-reasoning behaviors — planning, exploration, and reflection/monitoring — and enables the agent to articulate its cognitive state through special tags. During online interaction, lightweight, programmatic rules are used to grant verifiable rewards for these behaviors. These process-centric rewards are combined with the global outcome reward and optimized using a policy gradient method. 

**Design Philosophy:**
1. **Operationalizing Meta-Reasoning:** Extend the ReAct framework by introducing a structured set of meta-reasoning tags to explicitly represent distinct cognitive functions, decoupling reasoning from actions and enabling fine-grained analysis and supervision.
2. **Verifiable Reward Shaping:** Provide dense, programmatic, rule-based rewards for actions that contribute to effective problem-solving (e.g., exploration rewarded for discovering a new state; reflection rewarded for correcting a prior mistake).
3. **Critic-free Policy Optimization:** Combine the process-centric rewards with the final outcome signal and optimize using a critic-free policy gradient method (GRPO-MR) that computes step-level advantages by combining global trajectory performance with local, context-aware reasoning quality.

## 4. Core Innovation Points

1. **Identification and Analysis of the Inefficient Exploration Problem:** The paper identifies and analyzes a critical inefficient exploration issue in outcome-only end-to-end RL for long-horizon LLM agents. It empirically demonstrates that spurious state–action correlations override genuine reasoning, leading to redundant reasoning steps and illogical action loops, and proves that this issue persists even with model scaling.
2. **Introduction of the RLVMR Framework with Verifiable Meta-Reasoning Rewards:** The paper introduces a novel framework, RLVMR, that provides dense, verifiable rewards for meta-reasoning behaviors like planning, exploration, and reflection, enabling more robust and efficient problem-solving. This moves beyond sparse, outcome-only signals to provide direct, process-level supervision.
3. **Meta-Reasoning-Aware Reward Shaping Mechanism:** The paper operationalizes metacognitive theory into a reward shaping mechanism. It defines specific meta-reasoning tags (`<planning>`, `<explore>`, `<reflection>`, `<monitor>`) and designs corresponding programmatic, rule-based reward functions (e.g., exploration reward for new states, reflection reward for corrective actions after failure) that are verifiable at each step.
4. **Group Relative Policy Optimization with Meta-Reasoning (GRPO-MR):** The paper designs a novel policy optimization algorithm, GRPO-MR, which computes a step-level advantage by combining global trajectory performance with local, context-aware reasoning quality. It groups steps by meta-reasoning tag type to normalize rewards, balancing the outcome advantage and meta-reasoning advantage to optimize the policy.

## 5. Overview of the Overall Technical Solution

The RLVMR framework trains LLM agents in two sequential phases:
1. **Cold Start Phase (Initial Meta-Reasoning Acquisition via SFT):** 
   - Collect a dataset of successful task trajectories containing only observation-action pairs.
   - Employ a more powerful teacher model (e.g., GPT-4) to annotate these trajectories with meta-reasoning tags, inferring the most likely cognitive step preceding each action. This creates synthetic, reasoning-rich expert demonstrations.
   - Fine-tune the target LLM on these annotated trajectories (using only 200 trajectories for 5 epochs) so it learns to imitate the expert’s meta-reasoning and action generation patterns and acquire the foundational ability to generate structured meta-reasoning syntax.
2. **Reinforcement Learning Phase (GRPO-MR):**
   - The agent interacts with the environment online, generating trajectories with meta-reasoning tags and actions.
   - At the end of an episode, an Outcome Reward $R(\tau)$ is given.
   - At each step, a Meta-Reasoning Reward $r_{MR}^t$ and Format Reward $r_{format}^t$ are computed programmatically based on rules tied to the tag and action validity.
   - The GRPO-MR algorithm computes the Trajectory-level Relative Advantage and the Meta-reasoning Level Relative Advantage.
   - The final step-level advantage is a weighted combination of these two signals, used to update the policy via a clipped surrogate objective with KL divergence regularization.

## 6. Detailed Module Design

**1. Task Formulation as a Markov Decision Process (MDP)**
The interaction between an agent and its environment is formalized as an MDP defined by a tuple $(S, A, O, F, R)$:
- $S$: set of environment states
- $A$: action space
- $O$: observation space
- $F: S \times A \rightarrow S$: state transition function
- $R: S \times A \rightarrow \mathbb{R}$: reward function
State, action, and observation spaces are represented as natural language sequences. At timestep $t$, the policy $\pi_\theta$ generates a thought process $th_t$ and an action $a_t$ based on current state $s_t$: $(th_t, a_t) \sim \pi_\theta(\cdot | s_t)$. The trajectory is $\tau = \{(o_1, th_1, a_1), (o_2, th_2, a_2), \ldots, (o_n, th_n, a_n)\}$.

**2. Meta-Reasoning Framework Module**
This module extends the ReAct framework by introducing a structured set of meta-reasoning tags (enclosed in XML-style tags) to explicitly represent distinct cognitive functions, while all actions are contained within the `<action>` tag:
- **Planning (`<planning>`):** Decomposes the task into high-level steps to formulate an overall strategy. Used at the start of a task or when replanning is needed.
- **Exploration (`<explore>`):** Generates hypotheses or options to navigate uncertainty or bottlenecks, encouraging creative problem-solving.
- **Reflection (`<reflection>`):** Reviews history to analyze errors and formulate corrective actions. Typically triggered after unsuccessful attempts.
- **Monitoring (`<monitor>`):** Tracks task progress against the overall plan, ensuring actions remain aligned with subgoals. Applied during routine execution.

**3. Meta-Reasoning-Aware Reward Shaping Module**
The composite reward signal combines task completion with the quality of the reasoning process:
- **Outcome Reward ($R(\tau)$):** A binary signal awarded at the end of a trajectory: $R(\tau) = r_s$ for task success and $0$ otherwise, where $r_s$ is a positive constant.
- **Meta-Reasoning Reward ($r_{MR}^t$):** A dense reward assigned at each step $t$ to incentivize locally beneficial behaviors:
  - **Planning Reward ($r_{planning}$):** Awarded for a `<planning>` step if the trajectory succeeds.
  - **Exploration Reward ($r_{explore}$):** Awarded if the current action targets a new object or location, discouraging redundancy.
  - **Reflection Reward ($r_{reflection}$):** Awarded if a `<reflection>` step is followed by a corrective action after a sequence of failures.
- **Format Reward ($r_{format}^t$):** A penalty, $-\lambda_{format}$, is applied if the model’s output at step $t$ does not conform to the expected `<tag>...</tag><action>...</action>` structure.
The total step-level reward is the sum: $r_t = r_{MR}^t + r_{format}^t$.

**4. Group Relative Policy Optimization with Meta-Reasoning (GRPO-MR) Module**
This module computes a step-level advantage by combining global trajectory performance with local reasoning quality:
- **Trajectory-level Relative Advantage:** For a batch of $K$ trajectories, normalized trajectory-level advantage is:
  $A^{traj}_k = \frac{R(\tau_k) - \mu_R}{\sigma_R}$
  where $\mu_R$ and $\sigma_R$ are the mean and standard deviation of outcome rewards across the batch.
- **Meta-reasoning Level Relative Advantage:** Steps within a batch sharing the same meta-reasoning tag are grouped, and their rewards are normalized within that group:
  $A^{MR}_{t,tag} = \frac{r^{MR}_{t,tag} - \mu_{tag}}{\sigma_{tag}}$
  where $\mu_{tag}$ and $\sigma_{tag}$ are the mean and standard deviation of meta-reasoning rewards for all steps with that specific tag.
- **Combined Advantage:** The final step-level advantage $A_t$ is a weighted combination:
  $A_t = \alpha \cdot A^{traj}_k + (1 - \alpha) \cdot A^{MR}_{t,tag}$
  where $\alpha \in [0, 1]$ is a hyperparameter.
- **Policy Optimization:** The policy $\pi_\theta$ is optimized using a clipped surrogate objective with KL divergence regularization:
  $L_{final} = \mathbb{E}_t [\min (r_t(\theta)A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t)] - \lambda_{KL} D_{KL}(\pi_\theta \| \pi_{ref})$
  where $r_t(\theta)$ is the importance sampling ratio, $\epsilon$ is the clipping hyperparameter, and $\lambda_{KL}$ controls the KL penalty against a reference policy $\pi_{ref}$.

## 7. All Mathematical Formulas and Symbol Definitions

**Formulas:**
1. Agent's objective to learn an optimal policy:
   $\max_\theta \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)]$
2. Trajectory-level Relative Advantage:
   $A^{traj}_k = \frac{R(\tau_k) - \mu_R}{\sigma_R}$
3. Meta-reasoning Level Relative Advantage:
   $A^{MR}_{t,tag} = \frac{r^{MR}_{t,tag} - \mu_{tag}}{\sigma_{tag}}$
4. Final step-level Combined Advantage:
   $A_t = \alpha \cdot A^{traj}_k + (1 - \alpha) \cdot A^{MR}_{t,tag}$
5. Clipped surrogate objective with KL divergence regularization (Final Loss):
   $L_{final} = \mathbb{E}_t [\min (r_t(\theta)A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t)] - \lambda_{KL} D_{KL}(\pi_\theta \| \pi_{ref})$
6. Combined Advantage in Algorithm 2:
   $A^{(i)} = A^{(i)}_{outcome} + \lambda_{meta} \cdot A^{(i)}_{meta}$
7. Standard Discounted Return:
   $G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$

**Symbol Definitions:**
- $S$: Set of environment states
- $A$: Action space
- $O$: Observation space
- $F: S \times A \rightarrow S$: State transition function
- $R: S \times A \rightarrow \mathbb{R}$: Reward function
- $\pi_\theta$: Agent's policy parameterized by $\theta$
- $th_t$: Thought process at timestep $t$
- $a_t$: Action at timestep $t$
- $s_t$: Current state at timestep $t$
- $\tau$: Trajectory of interaction $\{(o_1, th_1, a_1), \dots, (o_n, th_n, a_n)\}$
- $R(\tau)$: Outcome Reward (binary: $r_s$ for success, $0$ otherwise)
- $r_s$: Positive constant for task success
- $r_{MR}^t$: Meta-Reasoning Reward at step $t$
- $r_{planning}$: Planning Reward
- $r_{explore}$: Exploration Reward
- $r_{reflection}$: Reflection Reward
- $r_{format}^t$: Format Reward (penalty $-\lambda_{format}$)
- $r_t$: Total step-level reward ($r_{MR}^t + r_{format}^t$)
- $K$: Number of trajectories in a batch
- $\mu_R, \sigma_R$: Mean and standard deviation of outcome rewards across the batch
- $A^{traj}_k$: Trajectory-level relative advantage for trajectory $k$
- $\mu_{tag}, \sigma_{tag}$: Mean and standard deviation of meta-reasoning rewards for all steps with a specific tag
- $A^{MR}_{t,tag}$: Meta-reasoning level relative advantage for step $t$ with a specific tag
- $\alpha \in [0, 1]$: Hyperparameter balancing the influence of the global outcome and local reasoning quality
- $r_t(\theta)$: Importance sampling ratio
- $\epsilon$: Clipping hyperparameter
- $\lambda_{KL}$: KL penalty coefficient against a reference policy
- $\pi_{ref}$: Reference policy
- $\lambda_{meta}$: Weight for meta-reasoning advantage in Algorithm 2
- $\gamma \in (0, 1]$: Discount factor for temporal discounting

## 8. Algorithm Pseudocode

**Algorithm 1: RLVMR: Reinforcement Learning with Verifiable Meta-Reasoning Rewards**
Require: Policy $\pi_\theta$, Environment $\mathcal{E}$, Reward function $R$, Hyperparameter $\lambda_{meta}$
Ensure: Optimized policy parameters $\theta$
1: for iteration $t = 1, 2, \ldots, T$ do
2: Initialize trajectory set $D = \emptyset$
3: for episode $i = 1, 2, \ldots, N$ do
4: Get initial state from environment $s^{(i)}_0 \sim \mathcal{E}$
5: Initialize trajectory $\tau^{(i)} = \{\}$
6: $t \leftarrow 0$
7: while episode not terminated do
8: Sample action $a^{(i)}_t \sim \pi_\theta(\cdot|s^{(i)}_t)$
9: Execute action and observe $s^{(i)}_{t+1}, r^{(i)}_t = \mathcal{E}(s^{(i)}_t, a^{(i)}_t)$
10: // Extract reasoning type tag
11: $tag^{(i)}_t \leftarrow \text{ExtractReasoningTag}(a^{(i)}_t)$
12: // $\{\langle\text{planning}\rangle, \langle\text{explore}\rangle, \langle\text{reflection}\rangle, \langle\text{monitor}\rangle\}$
13: $\tau^{(i)} \leftarrow \tau^{(i)} \cup \{(s^{(i)}_t, a^{(i)}_t, r^{(i)}_t, tag^{(i)}_t)\}$
14: $t \leftarrow t + 1$
15: end while
16: // Compute Outcome Reward
17: $R^{(i)}_{outcome} \leftarrow R(\tau^{(i)})$
18: end for
19: // Compute Meta Reasoning Rewards for all trajectories
20: for episode $i = 1, 2, \ldots, N$ do
21: for each step $t$ in $\tau^{(i)}$ do
22: $r^{(i)}_{meta,t} \leftarrow \text{ComputeMetaReward}(a^{(i)}_t, tag^{(i)}_t, \tau^{(i)}, R^{(i)}_{outcome})$
23: $\tau^{(i)}[t] \leftarrow \tau^{(i)}[t] \cup \{r^{(i)}_{meta,t}\}$ ▷ Attach meta-reasoning reward to step $t$
24: end for
25: $D \leftarrow D \cup \{(\tau^{(i)}, R^{(i)}_{outcome})\}$
26: end for
27: // Compute group relative advantage
28: $\{A^{(i)}\}^N_{i=1} \leftarrow \text{ComputeGroupRelativeAdvantage}(D)$
29: // Group Relative Policy Optimization
30: for update step $k = 1, 2, \ldots, K$ do
31: Compute policy gradient: $\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim D}[\nabla_\theta \log \pi_\theta(a|s) \cdot A]$
32: Update policy parameters: $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$
33: end for
34: end for

**Algorithm 2: Step-Level Group Relative Advantage Computation**
Require: Trajectory data $D = \{(\tau^{(i)}, R^{(i)}_{outcome})\}^N_{i=1}$, Weight $\lambda_{meta}$
Ensure: Advantage estimates $\{A^{(i)}\}^N_{i=1}$
1: // ===== Outcome Advantage Computation =====
2: Group by enviroment index: $G_{outcome} = \{g_j\}$ where $g_j = \{i : \text{env\_idx}(i) = j\}$
3: for each group $g_j \in G_{outcome}$ do
4: Compute group mean: $\mu_j = \frac{1}{|g_j|} \sum_{i \in g_j} R^{(i)}_{outcome}$
5: Compute group std: $\sigma_j = \sqrt{\frac{1}{|g_j|} \sum_{i \in g_j} (R^{(i)}_{outcome} - \mu_j)^2}$
6: for $i \in g_j$ do
7: $A^{(i)}_{outcome} = \frac{R^{(i)}_{outcome} - \mu_j}{\sigma_j + \epsilon}$
8: end for
9: end for
10: // ===== Meta Reasoning Advantage Computation =====
11: Group by (enviroment index, reasoning tag): $G_{meta} = \{g_{j,k}\}$
12: where $g_{j,k} = \{i : \text{env\_idx}(i) = j \land \text{tag}(i) = k\}$
13: for each group $g_{j,k} \in G_{meta}$ do
14: Compute group mean: $\mu_{j,k} = \frac{1}{|g_{j,k}|} \sum_{i \in g_{j,k}} r^{(i)}_{meta}$
15: Compute group std: $\sigma_{j,k} = \sqrt{\frac{1}{|g_{j,k}|} \sum_{i \in g_{j,k}} (r^{(i)}_{meta} - \mu_{j,k})^2}$
16: for $i \in g_{j,k}$ do
17: $A^{(i)}_{meta} = \frac{r^{(i)}_{meta} - \mu_{j,k}}{\sigma_{j,k} + \epsilon}$
18: end for
19: end for
20: // ===== Final Advantage Combination =====
21: for $i = 1, 2, \ldots, N$ do
22: $A^{(i)} = A^{(i)}_{outcome} + \lambda_{meta} \cdot A^{(i)}_{meta}$
23: end for
return $\{A^{(i)}\}^N_{i=1}$

**Algorithm 3: Meta-Reasoning Reward Computation**
Require: Action $a_t$, State $s_t$, Reasoning tag $tag_t$, Trajectory $\tau$, Outcome reward $R(\tau)$
Require: Reward hyperparameters $\{r_{plan}, r_{explore}, r_{reflect}\}$, discount factor $\gamma$
Ensure: Meta-reasoning reward $r_{meta,t}$
1: $r_{meta,t} \leftarrow 0$
2: $valid_t \leftarrow \text{IsActionValid}(a_t)$
3: if $valid_t = \text{False}$ then
4: return 0 ▷ Invalid actions receive no reward
5: end if
6: if $tag_t = \langle\text{planning}\rangle$ then
7: if $R(\tau) > 0$ then ▷ Planning rewarded only on successful trajectories
8: $k \leftarrow \text{NumPlanningAfter}(t, \tau)$
9: $r_{meta,t} \leftarrow r_{plan} \cdot \gamma^k$
10: else
11: $r_{meta,t} \leftarrow 0$
12: end if
13: end if
14: if $tag_t = \langle\text{explore}\rangle$ then
15: $isRepeated \leftarrow \text{False}$
16: for $t' = 0$ to $t-1$ do
17: Extract transition $(s_{t'}, a_{t'}, s_{t'+1})$
18: if $(s_t, a_t, s_{t+1}) = (s_{t'}, a_{t'}, s_{t'+1})$ then
19: $isRepeated \leftarrow \text{True}$
20: break
21: end if
22: end for
23: if $isRepeated = \text{False}$ then
24: $r_{meta,t} \leftarrow r_{explore}$ ▷ Novel transition
25: else
26: $r_{meta,t} \leftarrow 0$
27: end if
28: end if
29: if $tag_t = \langle\text{reflection}\rangle$ then
30: if $t > 0$ then
31: Extract previous transition $(s_{t-1}, a_{t-1})$
32: $valid_{t-1} \leftarrow \text{IsActionValid}(a_{t-1})$
33: if $valid_{t-1} = \text{False}$ and $(s_t, a_t) \neq (s_{t-1}, a_{t-1})$ then
34: $r_{meta,t} \leftarrow r_{reflect}$ ▷ Effective reflection
35: else
36: $r_{meta,t} \leftarrow 0$
37: end if
38: else
39: $r_{meta,t} = 0$
40: end if
41: end if
return $r_{meta,t}$