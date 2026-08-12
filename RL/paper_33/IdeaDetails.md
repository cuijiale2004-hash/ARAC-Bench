**1. Research Background and Existing Pain Points**

**Research Background:**
Recent advances in Large Language Models (LLMs) have shifted the focus from building conversational systems to developing decision-making agents capable of reasoning about and interacting with their environments. To accomplish goals, language agents operate in multi-turn, textual observation-action loops and must adapt quickly using memory across turns. Central to such adaptation is exploration, which allows agents to test uncertain actions, acquire new knowledge, and avoid premature convergence on suboptimal strategies. Recent works have begun addressing the exploration limitation by guiding LLMs toward exploratory behaviors at test time. For example, some methods train models offline to distill exploration strategies from trajectories across diverse environments, while others induce such strategies from offline search traces or train models to learn to explore in-context as a better way of spending test-time compute.

**Existing Pain Points:**
1.  **Inability to Actively Explore:** Unlike humans who can explore systematically and make fast adaptations in new environments, LLM agents do not robustly engage in exploration without substantial interventions. RL-trained agents often struggle in tasks that require active exploration.
2.  **Failure to Adapt from Trial-and-Error:** Current RL-trained agents often learn a fixed policy during training and fail to efficiently adapt from trial-and-error experiences at test time.
3.  **Limitations of Existing Exploration Methods:** Existing works addressing exploration either focus on single-turn non-agentic reasoning problems or rely on offline data, which limits them to imitation rather than active exploration.
4.  **Sparse Success Signals:** Multi-turn tasks often have a sparse success signal only after an episode concludes. Standard single-episode Reinforcement Learning optimizes for immediate payoff and fails to incentivize exploration in early stages, leading to suboptimal convergence.

**2. Core Research Motivation and Scientific Questions**

**Core Research Motivation:**
The core motivation is to take a step toward agents that can actively explore their environment, gather feedback, and leverage this experience for more effective exploitation. Since multi-turn tasks often have a sparse success signal after an episode, the work considers a multi-episode regime where an episode is the unit of exploration and exploitation. Balancing exploration and exploitation can be naturally formulated as a cross-episode reinforcement learning framework. Training across many similar but different environments under this framework leads to Meta-Reinforcement Learning (Meta-RL), where the agent is forced to discover general strategies that work in unseen and potentially harder environments.

**Scientific Questions:**
1.  How to balance exploration and exploitation over multiple attempts at a task in LLM agents?
2.  How to efficiently adapt the policy during training and evaluation without relying on expensive gradient-based updates, naturally leveraging the in-context learning abilities of LLMs?

**3. Overall Core Idea and Design Philosophy**

**Overall Core Idea:**
The paper proposes LAMER (LLM Agent with Meta-RL), a general Meta-RL framework for LLM agent training. LAMER is designed around a multi-episode structure to train the agent to solve the problem through trial and error. In early episodes, the agent is encouraged to gather diverse experiences and informative feedback from the environment, which are then used to adapt its policy in later episodes. By maximizing long-term rewards across episodes, the agent internalizes a learning algorithm that explicitly incentivizes exploration for improved downstream exploitation.

**Design Philosophy:**
The design philosophy is rooted in the Meta-RL principle where the policy learned at the meta-level encodes how to adaptively switch between information-gathering (exploration) and reward-maximizing behaviors (exploitation) depending on the stage of interaction with a new task. At both training and test time, the agent effectively leverages the feedback and reflections from previous episodes to determine its strategy for the next episode. This essentially implements an RL algorithm in context, making the approach naturally suited for LLM agents without parameter updates at test time. Meta-RL produces more diverse samples while simultaneously achieving higher performance, reaching a better balance between exploration and exploitation than standard RL.

**4. Core Innovation Points**

1.  **First Meta-RL Framework for LLM Agent Training:** To the best of the authors' knowledge, this is the first time a meta-RL framework is used for LLM agent training, formulating the balance of exploration and exploitation as a cross-episode reinforcement learning problem.
2.  **Cross-Episode Training Framework:** LAMER introduces a cross-episode training scheme that treats each trial as a sequence of episodes. Unlike standard single-episode RL, LAMER extends the credit assignment across multiple episodes using a trajectory discount factor ($\gamma_{traj}$) to incentivize exploration in the early stages and maximize long-term rewards.
3.  **In-Context Policy Adaptation via Self-Reflection:** Instead of relying on gradient-based updates, LAMER uses self-reflection as an in-context adaptation mechanism. The agent summarizes past experiences and adjusts its strategy accordingly by modifying the context window, leveraging the in-context capability of LLMs.
4.  **Explicit Training of Self-Reflection:** The self-reflection step is explicitly trained in LAMER using the reward obtained in the subsequent episode, ensuring the reflections are grounded in environmental feedback rather than just hallucinations.
5.  **Superior Exploration-Exploitation Trade-off:** LAMER retains higher sample diversity (trajectory entropy) from the base model while achieving better success rates compared to standard RL, demonstrating a better trade-off between exploration and exploitation.

**5. Overview of the Overall Technical Solution**

The overall technical solution is the LAMER framework, which optimizes a Meta-RL objective for LLM agents. The process considers an LLM agent interacting with an environment formulated as a Markov Decision Process (MDP). Instead of independent rollouts as in standard RL, LAMER utilizes a "Trial" structure consisting of $N$ sequential episodes. If an episode fails, the agent generates a textual reflection on the past attempt and updates its inter-episode memory. The next episode starts from the same initial state but with the updated context (policy adapted in-context). The objective function maximizes the expected discounted return across the trial, combining within-episode discounting ($\gamma_{step}$) and cross-episode discounting ($\gamma_{traj}$). The gradient is estimated using standard policy gradient methods based on the cross-episode advantage estimation.

**6. Detailed Module Design**

**Module 1: Cross-Episode Training Framework**
*   **Mechanism:** In the training of LAMER, each trial consists of $N$ episodes sequentially generated by the agent. The policy at episode $n$ is updated from the accumulated history of previous episodes through an adaptation strategy. The rollout process terminates at episode $n$ if the trajectory is successful (indicated by environment feedback). Otherwise, the agent starts a new episode from the same initial state, repeating until the maximum episode budget is reached.
*   **Credit Assignment:** To enhance exploration and maximize the long-term reward, the framework defines a discounted return across the episodes of the trial. The trajectory discount factor $\gamma_{traj}$ controls how rewards are propagated within a trial. A small $\gamma_{traj}$ biases the objective towards early episodes, leading to rapid exploitation. A larger $\gamma_{traj}$ emphasizes long-horizon return, encouraging more exploration at the early stage.

**Module 2: In-Context Policy Adaptation with Self-Reflection**
*   **Mechanism:** After each episode finishes, the agent is prompted to generate a textual reflection on the previous attempt. This reflection provides specific feedback and a plan to guide the next episode.
*   **Policy Update:** The policy is updated through modifying the context, defined as $\pi_{\theta}^{(n)}(\cdot) = \pi_{\theta}(\cdot|H^{(n)})$, where $H^{(n)}$ denotes the inter-episode memory containing both the history trajectories and reflections.
*   **Memory Configuration:** The content $H^{(n)}$ can be adjusted. The default configuration retains both history and reflection. Alternative configurations include "Trajectory-only" or "Reflection-only" memory to reduce context length. The "Reflection-only" configuration often outperforms the default by presenting more concise and focused guidance.
*   **Training of Reflection:** The self-reflection step is explicitly trained using the reward obtained in the next episode.
*   **Prompting Structure:**
    *   **Standard Prompt:** Used for agent interaction. Includes environment description, rules, current observation, `{past experience reflection}` variable, `{history actions}` variable, and instructions to reason step-by-step and output actions within `<action> </action>` tags.
    *   **Reflection Prompt:** Used for reflection generation. Includes the history of a past experience (initial state, history actions, final state). Instructions require the agent to reflect on the past sequence, identify mistakes/inefficiencies, and devise a concise, improved plan. The output is placed within `<remark> </remark>` tags.

**Module 3: Optimization Strategy**
*   **Mechanism:** The proposed Meta-RL objective is optimized with standard policy gradient methods. Given the per-action cross-episode return, the gradient is estimated using the advantage estimation derived from the cross-episode return.
*   **Algorithm:** The framework is compatible with widely used optimizers such as PPO, critic-free approaches like GRPO, and GiGPO (the default used in experiments).
*   **Training Budget:** To ensure fair comparison with standard RL, the group size for standard RL is set to be three times larger than that of Meta-RL, guaranteeing the same total number of trajectories used for each gradient update step.

**7. All Mathematical Formulas and Symbol Definitions**

*   **MDP Formulation:** $M = (S, A, P, R, \gamma_{step})$
    *   $S$: State space
    *   $A$: Action space
    *   $R$: Reward function
    *   $P(\cdot | s_t, a_t)$: Transition function
    *   $\gamma_{step} \in [0, 1]$: Within-the-episode discount factor

*   **Trajectory Definition:** $\tau = (s_0, a_0, r_0, ..., s_{T-1}, a_{T-1}, r_{T-1})$

*   **Standard RL Objective:**
    $$E_{\tau \sim \pi_{\theta}} \left[ \sum_{t=0}^{T-1} \gamma_{step}^{t} r_t \right]$$

*   **Trial Definition (Cross-Episode):**
    $$\mathcal{T} = (\tau^{(0)}, \tau^{(1)}, \ldots, \tau^{(N-1)}), \text{ where } \tau^{(n)} \sim \pi_{\theta}^{(n)}(\cdot), n \in [0, N-1]$$
    *   $\pi_{\theta}^{(n)}(\cdot)$: Policy at episode $n$ updated from accumulated history.

*   **Within-the-Episode Discounted Return:**
    $$g_t^{(n)} = \sum_{l=t}^{T-1} \gamma_{step}^{l-t} r_l^{(n)}$$

*   **Cross-Episode Discounted Return:**
    $$G_t^{(n)} = \underbrace{g_t^{(n)}}_{\text{within-the-episode}} + \underbrace{\sum_{m=n+1}^{N-1} \gamma_{traj}^{m-n} g_0^{(m)}}_{\text{cross-episode}}$$
    *   $\gamma_{traj} \in [0, 1]$: Cross-episode discount factor.

*   **Meta-RL Objective:**
    $$J(\theta) = E_{\mathcal{T} \sim \pi_{\theta}} \left[ \sum_{n=0}^{N-1} \gamma_{traj}^{n} \sum_{t=0}^{T-1} \gamma_{step}^{t} r_t^{(n)} \right] = E_{\mathcal{T} \sim \pi_{\theta}} \left[ G_0^{(0)} \right]$$

*   **Gradient Estimation:**
    $$\nabla_{\theta} J(\theta) = E_{\mathcal{T} \sim \pi_{\theta}} \left[ \sum_{n=0}^{N-1} \sum_{t=0}^{T-1} \nabla_{\theta} \log \pi_{\theta}(a_t^{(n)} | s_t^{(n)}, H^{(n)}) A_t^{(n)} \right]$$
    *   $A_t^{(n)}$: Advantage estimation derived from $G_t^{(n)}$.
    *   $H^{(n)}$: Inter-episode memory containing history trajectories and reflections.

**8. Algorithm Pseudocode**

The paper does not provide a single consolidated pseudocode block but describes the iterative process and optimization steps clearly. Based on the text in Section 4 and Section 5, the logic is as follows:

**Training Loop for LAMER (Meta-RL):**
1.  **Input:** LLM agent with parameters $\theta$, environment distribution, number of episodes per trial $N=3$, group size $K=8$, discount factors $\gamma_{step}$, $\gamma_{traj}$.
2.  **For each training iteration:**
3.  $\quad$ **For each task in the batch:**
4.  $\quad\quad$ Initialize trial $\mathcal{T} = \emptyset$.
5.  $\quad\quad$ Initialize inter-episode memory $H^{(0)} = \emptyset$.
6.  $\quad\quad$ **For episode** $n = 0$ to $N-1$:
7.  $\quad\quad\quad$ Generate trajectory $\tau^{(n)} \sim \pi_{\theta}(\cdot | H^{(n)})$.
8.  $\quad\quad\quad$ Add $\tau^{(n)}$ to trial $\mathcal{T}$.
9.  $\quad\quad\quad$ If $\tau^{(n)}$ is successful: Break loop (proceed to next task).
10. $\quad\quad\quad$ Else:
11. $\quad\quad\quad\quad$ Generate reflection using Reflection Prompt based on $\tau^{(n)}$.
12. $\quad\quad\quad\quad$ Update inter-episode memory $H^{(n+1)}$ with history $\tau^{(n)}$ and reflection.
13. $\quad$ **Compute Returns:** For all generated trajectories in the batch, compute within-the-episode returns $g_t^{(n)}$ and cross-episode returns $G_t^{(n)}$ using $\gamma_{step}$ and $\gamma_{traj}$.
14. $\quad$ **Compute Advantage:** Estimate advantage $A_t^{(n)}$ derived from $G_t^{(n)}$.
15. $\quad$ **Update Parameters:** Optimize $\theta$ using policy gradient (e.g., GiGPO) based on the gradient estimation formula:
    $$\nabla_{\theta} J(\theta) = E_{\mathcal{T} \sim \pi_{\theta}} \left[ \sum_{n=0}^{N-1} \sum_{t=0}^{T-1} \nabla_{\theta} \log \pi_{\theta}(a_t^{(n)} | s_t^{(n)}, H^{(n)}) A_t^{(n)} \right]$$