### 1. Research Background and Existing Pain Points

**Research Background:**
Recent advances in language models have unlocked impressive reasoning capabilities in domains such as mathematics and programming. However, many emerging applications unfold in complex environments that require multi-step reasoning, such as web navigation, deep research, and computer/phone-use tasks. Success in these domains depends not only on the ability to decompose goals and reflect on past progress but also on AI agents’ ability to construct accurate world models that capture the structure and dynamics of increasingly complex environments. Insights from human cognition indicate that the ability to model and simulate complex environments is critical. Neuroscience research highlights "vicarious trial and error": mentally simulating possible futures, evaluating their consequences, and selecting advantageous actions without directly experiencing each option. This ability greatly enhanced adaptability and decision-making, which is equally essential for reasoning in long-horizon AI agent tasks.

**Existing Pain Points:**
1.  **Lack of Simulation Capacity in AI Agents:** Current AI agents need the capacity for "vicarious trial and error" to enhance their understanding and performance in complex interactive environments. Their expert-level abilities in math and coding contrast sharply with their performance in long-horizon, interactive tasks.
2.  **Limitations of Existing Methods:** Early reactive agents directly prompt a (V)LM to make decisions on immediate observations without simulation or planning, hindering performance on complex tasks. Search-based methods and hierarchical methods introduce substantial overheads during inference (e.g., additional interactions or complex heuristics).
3.  **Limitations of Synthetic Data:** Initial attempts to address this limitation, such as Dyna-Think, integrate simulation into reasoning through distilling simplified traces and adding auxiliary next-state prediction tasks. However, these methods rely on the strong capability of reasoning models (e.g., DeepSeek-R1) to directly generate synthetic simulation data, which can embed errors and biases. Empirical evidence shows that while DeepSeek-R1 can simulate structured environments, its performance drops sharply in more complex domains both in simulation accuracy and overall task success.

### 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:**
The core motivation stems from the finding that an agent’s ability to model and simulate the environment strongly correlates with its ability to correctly reason and complete long-horizon, planning-intensive tasks. To overcome the limitations of relying on strong reasoning models for synthetic data generation (which embeds errors), there is a need to explicitly teach AI agents to simulate the environment by directly learning from real experiences.

**Scientific Questions:**
1.  How to synergize world simulations with reasoning?
2.  How to improve the simulation ability to help improve the policy?

### 3. Overall Core Idea and Design Philosophy

**Overall Core Idea:**
The core idea is Dyna-Mind, a two-stage training framework that explicitly teaches (V)LM agents to integrate simulation into their reasoning. Unlike prior work that leverages superior LLMs to generate synthetic simulation data, Dyna-Mind constructs simulation-guided reasoning traces using search trees built from real environment interactions. It then utilizes online reinforcement learning with intermediate state feedback to further refine the agent's simulation and decision-making ability.

**Design Philosophy:**
The design philosophy posits that strong agents implicitly store and use a (compressed) representation of the world to enhance their decision-making. The framework grounds the agent’s reasoning in faithful world dynamics and equips it with the ability to anticipate future states in its reasoning, explicitly integrating "vicarious trial and error" into the agent's internal reasoning process.

### 4. Core Innovation Points

1.  **Introduction of "Vicarious Trial and Error" for AI Agents:** The paper explicitly identifies and implements the capacity to mentally simulate alternative futures before acting, drawn from human cognition, as a central mechanism for enhancing understanding and performance in complex interactive environments.
2.  **Reasoning with Simulations (RESIM) from Real Experience:** Unlike prior work that relies on distilling from strong reasoning models (which can hallucinate), RESIM algorithmically constructs reasoning traces using expanded search trees obtained from real environment interactions. This grounds the agent’s reasoning in faithful world dynamics.
3.  **Simulation-Guided Distillation:** The RESIM data collection process aggregates entire search trees into a single reasoning response. This data is then used for SFT to train the model to directly generate simulation-guided reasoning traces without any algorithm support during inference.
4.  **SimRollout Mechanism:** A novel rollout process for reinforcement learning where, at each state, the agent's proposed plan is executed in the environment to obtain ground truth next-states, which are then fed back to the agent to refine its response. This bridges the gap between simulation and reality.
5.  **Dyna-GRPO Algorithm:** A modification of GRPO that iterates between simulation improvement (learning from refined policies that use future states information from SimRollout) and direct policy improvement (trained on standard rollouts without future-state access). It utilizes a modified advantage function ($A_{\text{refine}}$) to reward refinements that correctly solve the task and improve upon standard policy rollouts.

### 5. Overview of the Overall Technical Solution

The overall technical solution follows a two-stage pipeline:

**Stage 1: RESIM (Reasoning with Simulations) Training:**
1.  **Data Collection:** Use a rollout model to generate $b$ rollouts from a state $s$ up to depth $d$. Use a value function to estimate the quality of partial rollouts. Use a generic (V)LM to aggregate the search tree into a single reasoning response $a^{\text{RESIM}}$ which encapsulates the simulation.
2.  **Distillation:** Perform SFT on the policy model using the trajectories $\tau = \{s_0, a^{\text{RESIM}}_0, s_1, a^{\text{RESIM}}_1, \cdots, s_T, a^{\text{RESIM}}_T\}$ to teach the model to generate simulation-guided reasoning without search algorithms.

**Stage 2: Dyna-GRPO Training:**
1.  **Simulation Improvement:**
    *   Perform SimRollout with group size $G/2$: For each state, sample a response, extract the plan, execute it to get real next-states, and prompt the model to refine the response. Collect refined trajectories $\tau'$ and $\tau'_{\text{refine}}$.
    *   Perform standard rollouts with group size $G/2$.
    *   Combine these into a single group of size $G$ and perform GRPO.
    *   Utilize $\tau'_{\text{refine}}$ to improve the model’s simulation refinement ability using the modified advantage $A_{\text{refine}}$.
2.  **Policy Improvement:** Perform standard policy rollouts without future state information, optimized by GRPO using episode-level advantage.

### 6. Detailed Module Design

**6.1 RESIM Data Collection Module**
Given a state $s$:
1.  **Rollout Generation:** Use a rollout model $\pi_\theta$ to generate $b$ rollouts from $s$ up to depth $d$.
2.  **Value Estimation:** Use a value function $V_\nu$ to provide an estimate of the quality of each of the partial rollouts.
3.  **Aggregation:** Use a generic (V)LM to aggregate all these rollouts and their values into a single response $a^{\text{RESIM}}$ by prompting the (V)LM to:
    *   First independently summarize each partial rollout (which contains ground-truth future states information from the environment).
    *   Then aggregate all these summaries into a coherent response conditioned on the current state $s$ and previous $h$ actions and states, and choose the best plan and the next immediate action for execution.
4.  **Execution:** Execute the final chosen action from $a^{\text{RESIM}}$ in the environment.

**6.2 RESIM Distillation Module**
Given a collection of trajectories $\tau = \{s_0, a^{\text{RESIM}}_0, s_1, a^{\text{RESIM}}_1, \cdots, s_T, a^{\text{RESIM}}_T\}$:
*   Use Supervised Fine-Tuning (SFT) to train the model to directly generate each $a^{\text{RESIM}}_t$ given the current state $s_t$ as well as a maximum history of $h$ previous actions and states.

**6.3 SimRollout Module**
At each state $s_t$ during RL rollouts:
1.  Sample a response $a \sim \pi_\theta(\cdot|s_t)$.
2.  Extract the final chosen plan $\{\hat{a}_1, \hat{a}_2, \cdots, \hat{a}_d\}$ up to depth $d$ from $a$.
3.  Execute the plan in the environment to obtain ground truth next-states $\{s'_{t+1}, s'_{t+2}, \cdots, s'_{t+d}\}$.
4.  Prompt $\pi_\theta$ again to refine its response $a$ given these real future states: $a_{\text{refine}} \sim \pi_\theta(\cdot|s^{\text{refine}}_t)$, where $s^{\text{refine}}_t \equiv \{s_t \oplus a \oplus s'_{t+1} \oplus \hat{a}_2 \oplus \cdots \oplus s'_{t+d}\}$.

**6.4 Dyna-GRPO Training Module**
Iterates between:
1.  **Simulation Improvement:** For each task:
    *   Perform SimRollout with group size $G/2$, collecting $\tau' = \{s_0, a^{\text{refine}}_0, s_1, a^{\text{refine}}_1, \cdots\}$ and $\tau'_{\text{refine}} = \{s^{\text{refine}}_0, a^{\text{refine}}_0, s^{\text{refine}}_1, a^{\text{refine}}_1, \cdots\}$.
    *   Perform standard rollouts with group size $G/2$.
    *   Combine these standard rollouts $\tau$ with refined trajectories $\tau'$ into a single group of size $G$ and perform GRPO on this combined group.
    *   Utilize $\tau'_{\text{refine}}$ to improve the model’s simulation refinement ability using the modified advantage $A_{\text{refine}}$.
2.  **Policy Improvement:** Perform standard policy rollouts without future state information, optimized by GRPO using episode-level advantage.

### 7. All Mathematical Formulas and Symbol Definitions

**Markov Decision Process Formulation:**
*   $S, A, T, R$: State space, action space, transition function, reward function.
*   $\pi_\theta$: Agent policy.
*   $s_t \sim S$: State at time step $t$.
*   $a_t \sim \pi_\theta(\cdot|s_t)$: Action at time step $t$.
*   $s_{t+1} \sim T(s_t, a_t)$: Next state.
*   $r_T \sim R(s_T, a_T)$: Terminal reward.

**GRPO Objective:**
$J_{\text{GRPO}} = \mathbb{E}_{\tau \sim \pi_{\theta_{\text{old}}}} \left[ \frac{1}{G T} \sum_{i=1}^G \sum_{t=1}^T \min \left( \rho_\theta(a_t^{(i)}) A(a_t^{(i)}), \text{clip}(\rho_\theta(a_t^{(i)}), 1\pm \epsilon) A(a_t^{(i)}) \right) - \beta D_{KL}(\pi_\theta || \pi_{\theta_{\text{ref}}}) \right]$

**Importance Sampling Ratio:**
$\rho_\theta(a) = \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{ref}}}(a|s)}$

**Advantage Function (Episode-level):**
$A(a_t^{(i)}) = A_{\text{GRPO}}(\tau^{(i)}) = \frac{R(\tau^{(i)}) - \text{mean}(\{R(\tau^{(j)})\}_{j=1}^G)}{\text{std}(\{R(\tau^{(j)})\}_{j=1}^G)}$

**Reward Definition:**
$R(\tau^{(i)}) = \sum_{t=1}^T R(s_t, a_t)$
Where:
*   $R(s_t, a_t) = -0.1$ for non-terminal steps.
*   $R(s_T, a_T) = 10.0$ for terminal steps when task succeeded.
*   $R(s_T, a_T) = 0.0$ for terminal steps when task failed.

**SimRollout Refinement Input:**
$s^{\text{refine}}_t \equiv \{s_t \oplus a \oplus s'_{t+1} \oplus \hat{a}_2 \oplus \cdots \oplus s'_{t+d}\}$

**Dyna-GRPO Simulation Improvement Advantage:**
$A_{\text{refine}}(\tau^{(i)}_{\text{refine}}) = \begin{cases} 1.0, & \text{if } \tau^{(i)}_{\text{refine}} \text{ is correct and } R(\tau^{(i)}_{\text{refine}}) > \max(\bar{R}, \bar{R}_{\text{refine}}) \\ 0.0, & \text{otherwise} \end{cases}$
Where:
*   $\bar{R} = \frac{1}{G/2} \sum_{i=1}^{G/2} R(\tau^{(i)})$ is the mean reward of the standard policy rollouts.
*   $\bar{R}_{\text{refine}} = \frac{1}{G/2} \sum_{i=1}^{G/2} R(\tau^{(i)}_{\text{refine}})$ is mean reward from SimRollout.

**Value Function for RESIM (Appendix D.3):**
$V(s_t) = \gamma^{t_{\max}-t} \frac{1}{|\mathcal{T}|} \sum_{\tau \in \mathcal{T}} \mathbb{1}[\tau \text{ is successful}]$
Where:
*   $\mathcal{T} \equiv \{\tau_1, \tau_2, \cdots | s_t \in \tau_i\}$
*   $\gamma$: Discount factor.
*   $t_{\max}$: Maximum number of steps in a trajectory.

### 8. Algorithm Pseudocode

**Algorithm 1: DYNA-GRPO**
Require: policy $\pi_\theta$, environment $\mathcal{T}$, group size $G$
Require: hyperparameters $G, N, n_T, n_\pi$
1: for $N$ training iterations do
2:  // simulation improvement
3:  for $n_T$ steps do
4:   // see Algorithm 2
5:   $\{\tau'\}, \{\tau'_{\text{refine}}\} \leftarrow \text{SimRollout}(\pi_\theta, \mathcal{T}, G/2)$
6:   $\{\tau\} \leftarrow \text{Rollout}(\pi_\theta, \mathcal{T}, G/2)$
7:   Update $\pi_\theta$ with $\text{GRPO}(\{\tau\} \cup \{\tau'\})$
8:   Update $\pi_\theta$ with $\text{GRPO}(\{\tau'_{\text{refine}}\})$ using $A_{\text{refine}}$
9:  end for
10: // policy improvement
11: for $n_\pi$ steps do
12:  $\{\tau\} \leftarrow \text{Rollout}(\pi_\theta, \mathcal{T}, G)$
13:  Update $\pi_\theta$ with $\text{GRPO}(\{\tau\})$
14: end for
15: end for
16: return $\pi_\theta$

**Algorithm 2: Simulation Refinement Rollout (SIMROLLOUT)**
Require: policy $\pi_\theta$, environment $\mathcal{T}$, group size $G$
1: repeat the following $G$ times:
2: $\tau' \leftarrow \emptyset, \tau'_{\text{refine}} \leftarrow \emptyset, t = 0, s_0 \leftarrow \mathcal{T}$
3: while not done and $t < t_{\max}$ do
4:  $a \leftarrow \pi_\theta(s_t)$
5:  $\{\hat{a}_1, \cdots, \hat{a}_n\} \leftarrow \text{extract\_plan}(a)$
6:  // improve action $a$ using next-state information
7:  $\{s_{t+1}, \cdots, s_{t+n}\} \leftarrow \{\mathcal{T}(s_t, \hat{a}_1), \cdots, \mathcal{T}(s_{t+n-1}, \hat{a}_n)\}$
8:  $s^{\text{refine}}_t \leftarrow \text{refinement prompt}(a|s_t, a, \{s_{t+1}, \hat{a}_1, \cdots, s_{t+n}\})$ // see Table A3
9:  $a_{\text{refine}} \leftarrow \pi_\theta(s^{\text{refine}}_t)$
10: // update episode buffer
11: $\tau' \leftarrow \tau' \cup \{s_t, a_{\text{refine}}\}$ // learn improved policy
12: $\tau'_{\text{refine}} \leftarrow \tau'_{\text{refine}} \cup \{s^{\text{refine}}_t, a_{\text{refine}}\}$ // learn to refine simulations
13: $s_{t+1} \leftarrow \mathcal{T}(s_t, a_{\text{refine}})$
14: $t \leftarrow t + 1$
15: end while
16: return $\tau', \tau'_{\text{refine}}$

**Algorithm 3: RESIM**
Require: policy $\pi_\theta$, value function $V_\nu$, environment $\mathcal{T}$, (V)LM $\mathcal{M}$
Require: hyperparameters $b, d, t_{\max}, b_{\text{train}}$
1: $\tau \leftarrow \emptyset, t = 0, s_0 \leftarrow \mathcal{T}$
2: while not done and $t < t_{\max}$ do
3:  $\{\tau_i\}_{i=1}^b \leftarrow$ sample $b$ rollouts using $\pi_\theta$ starting from $s_t$ for max $d$ steps
4:  $\{\tau_i\}_{i=1}^{b'} \leftarrow$ deduplicate $\{\tau_i\}_{i=1}^b$
5:  $\{v_i\}_{i=1}^{b'} \leftarrow$ estimate value $\{V_\nu(s^{i}_{t+d})\}_{i=1}^b$
6:  // subsample rollouts
7:  $\tau_* \leftarrow \tau_{\arg\max_i v_i}$
8:  $\{\tau_i\}_{i=1}^{b_{\text{train}}} \leftarrow \{\tau_*\} \cup$ subsample $b_{\text{train}} - 1$ rollouts from the rest of $\{\tau_i\}_{i=1}^{b'}$
9:  // aggregate rollouts into a single reasoning response
10: $\{plan_i\}_{i=1}^{b_{\text{train}}} \leftarrow$ summarize $\{\mathcal{M}(\tau_i, v_i)\}_{i=1}^{b_{\text{train}}}$
11: $a^{\text{RESIM}} \leftarrow$ aggregate $\mathcal{M}(s_t, \{plan_i\}_{i=1}^{b_{\text{train}}})$
12: // next step
13: $s_{t+1} \leftarrow \mathcal{T}(s_t, a^{\text{RESIM}})$
14: $\tau \leftarrow \tau \cup \{s_t, a^{\text{RESIM}}\}$
15: $t \leftarrow t + 1$
16: end while
17: return $\tau$