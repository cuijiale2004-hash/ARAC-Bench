**1. Research Background and Existing Pain Points**

**Research Background:**
Large Language Model (LLM) agents are task-specific workflows that utilize LLMs as key components for decision making, action taking, and tool use. To improve the performance of LLM agents, two complementary approaches are widely adopted: Multi-Agent Systems (MAS) and Reinforcement Learning (RL). MAS strengthens task-specialized performance via role-based orchestration, where multiple agents collaborate. RL leverages environment rewards to train stronger policies, such as Group Relative Policy Optimization (GRPO)-style optimization, which iteratively updates model weights to strengthen decision-making. While MAS typically employs prompt-only augmentation on a shared LLM policy for role-based coordination, recent studies highlight the potential of role-specialized MAS, which adopts distinct models for different roles. However, the effectiveness of applying on-policy RL training to MAS is underexplored.

**Existing Pain Points:**
1.  **Algorithmic Challenge - Grouping Assumptions:** Standard GRPO grouping assumptions fail in MAS. GRPO's advantage calculation hinges on a fair comparison among all candidates within a group, requiring identical prompts. In MAS, a "prompt" embeds role-specific context and cross-agent interaction history, meaning prompts differ across turns and roles. If common parallel sampling is used (sampling K full trajectories from the initial state), each group size becomes 1 when $t > 1$ because no other sample shares the identical prompt. This eliminates GRPO's variance-reduction effect and yields unstable updates.
2.  **System Challenge - Training Infrastructure:** Existing on-policy RL frameworks for LLM agents (e.g., TRL, VERL, AReaL, OpenRLHF) primarily support single-agent RL training. They typically involve a single agent-environment interaction pattern, a single policy operating, and a single LLM resource pool. This makes it difficult to (i) train multiple models in on-policy RL, (ii) maintain clean on-policy training data, and (iii) support diverse MAS workflows.
3.  **Optimization Instability:** Directly applying GRPO to MAS (MAS-GRPO) violates the identical-state assumption. As multi-turn interaction histories diverge, the group-averaged baseline incorrectly aggregates heterogeneous states. This structural misalignment biases advantage estimation and destabilizes optimization, often leading to performance degradation.

**2. Core Research Motivation and Scientific Questions**

**Core Research Motivation:**
The natural next step to improve LLM agents is to combine MAS and RL: using RL to train MAS, such that we gain both stronger learned policies and role-specialized collaboration. However, bringing RL into MAS raises coupled algorithmic and system challenges that need to be addressed to unlock this potential synergy.

**Scientific Questions:**
1.  How to adapt group-relative optimization (like GRPO) for MAS where prompts vary by role and turn, ensuring fair credit assignment and valid advantage comparisons?
2.  How to design a training system that supports concurrently launching multiple models, orchestrating inter-agent environment interactions, and performing independent on-policy parameter updates?

**3. Overall Core Idea and Design Philosophy**

**Overall Core Idea:**
Introduce AT-GRPO (Agent- and Turn-wise GRPO), consisting of (i) an Agent- and Turn-wise grouped RL algorithm tailored for MAS and (ii) a system to support both single-policy and multi-policy training.

**Design Philosophy:**
The core design philosophy is to ensure valid and fair GRPO advantage comparisons in a multi-agent, multi-turn environment. Since fairness is enforced by the reward mechanism and advantage is determined solely by the response quality, a valid comparison is only possible when all responses in a group originate from an identical prompt. Because MAS prompts differ by role and interaction history, grouping must be Agent- and Turn-wise. To ensure valid group sizes (K) for turns beyond the first, sampling must be Tree-structured (branching at each turn) rather than parallel sampling of full trajectories. Furthermore, credit assignment must balance global team objectives with role-specific incentives, and the system must isolate resources per policy to maintain on-policy regimes.

**4. Core Innovation Points**

1.  **AT-GRPO Algorithm:** Introduction of an Agent- and Turn-wise grouped RL algorithm, AT-GRPO, specifically tailored for MAS, adapting group-relative optimization for multi-agent settings.
2.  **Tree-structured Sampling:** Development of a tree-structured sampling scheme where K candidate actions are sampled at each turn for each agent. This ensures valid comparison groups of size K for every turn, overcoming the group size=1 limitation of parallel sampling.
3.  **Agent-wise Credit Assignment:** Formulation of a credit assignment mechanism using a mixture of global team reward and agent-specific local reward, balancing shared team objectives with role-specific incentives.
4.  **MAS Training System:** Design of a novel training system consisting of independent LLM Resource Pools (GPU) per policy (with RolloutWorkers and UpdateWorkers) and a CPU Environment Resource Pool (with EnvWorkers), supporting on-policy RL updates for multiple policies and diverse workflows.
5.  **Role-sharing vs. Role-specialized Policy Optimization:** Definition and implementation of two optimization regimes (role-sharing and role-specialized) that differ in how training data is batched and routed, enabling flexible training configurations.

**5. Overview of the Overall Technical Solution**

The technical solution operates in two main phases per iteration: On-Policy Rollout & Data Collection, and Per-Model Policy Update.
In the **Rollout Phase**, $E$ environments are resampled. For each environment instance, agents interact sequentially or in parallel according to the MAS workflow. At each turn $t$, for each agent $i$, $K$ candidate actions are sampled from the current state. The advantages for these $K$ candidates are calculated within this group (Agent- and Turn-wise grouping). The data tuple (group key, observation, K actions, K advantages) is added to the dataset specific to the agent's policy. To proceed with the environment rollout, the candidate with the highest reward is greedily selected (Tree-structured sampling).
In the **Update Phase**, trajectories are routed to the corresponding UpdateWorker based on the policy assignment $\sigma(i)$. Per-model batches are constructed, and the policy is updated using a PPO-style clipped surrogate objective based on the advantages computed during rollouts.

**6. Detailed Module Design**

**6.1 Algorithm Design: AT-GRPO**

*   **Tree-structured Sampling:** At each turn $t$, for each agent $i$, sample $K$ candidate actions and rewards from the current state. The advantages for these $K$ candidates are calculated within this group. The full data tuple is added to dataset $D_i$. To proceed with the environment rollout, greedily select the candidate with the highest reward to be the executed action. This greedy selection concentrates exploration on coordination-critical decisions and maintains a balanced mix of positive and negative samples.
*   **Agent– and Turn-wise Grouping:** Group experiences based on the acting agent and the turn number within each parallel environment instance. A unique group key $g$ is defined for each agent $i$ at each turn $t$ in each environment $e$ using a lightweight hash function. All data generated from the K-branch sampling at that step is stored together under this group key.
*   **Agent-wise Credit Assignment:** Assign credit using a mixture of global and local rewards. At each turn $t$, the environment provides a global team reward $r^{\text{team}}_t$ and an agent-specific local reward $r^{\text{loc}}_{t,i}$.

**6.2 MAS Training System**

*   **LLM Resource Pools (GPU):** Each policy is managed within an independent resource pool. Each pool comprises two workers: a RolloutWorker for inference and an UpdateWorker for optimization. During rollouts, all policies interact collectively. Trajectories are routed to the corresponding UpdateWorker to maintain an on-policy learning regime.
*   **Environment Execution (CPU) and Data Flow:** Environment steps run in a fleet of CPU EnvWorkers, each managing a single sandboxed instance. EnvWorkers stream observations, tool logs, and rule-based rewards to a Router. The Router dispatches collected experience based on policy assignment: experiences generated by agent $i$ are sent to the UpdateWorker of its designated policy $\sigma(i)$.

**6.3 Reward Design Module**

*   **Generic Local Reward Formula:**
    $r^{\text{loc}}_{t,i} = m_{t,i} \sum_{\ell \in \{\text{fmt}, \text{tool}, \text{step}\}} c^i_\ell s^i_{\ell,t}, \quad \sum_\ell c^i_\ell = 1$
    where $m_{t,i} \in \{0, 1\}$ is a verifiability mask.
*   **Math Reward:**
    *   **Team:** $r^{\text{team}}_t = \mathbb{1}\{\text{CHECK}_{\text{FINAL}}\text{MATHVERIFIER+NUMEQ}(h)=\text{pass}\} \in \{0, 1\}, \forall t.$
    *   **Reasoner Local:** $c^{\text{Reasoner}}_{\text{fmt}} = 0.20, c^{\text{Reasoner}}_{\text{tool}} = 0.00, c^{\text{Reasoner}}_{\text{step}} = 0.80.$
        $s^{\text{Reasoner}}_{\text{fmt},t} = \mathbb{1}\{\text{required output schema matched at turn } t\}$
        $s^{\text{Reasoner}}_{\text{step},t} = \begin{cases} \text{NUMEQ}_\delta(\hat{y}, y^\star) & \text{if MATHVERIFIER extracts a numeric } \hat{y} \\ 0 & \text{otherwise} \end{cases}$
*   **Code Reward:**
    *   **Team:** $r^{\text{team}}_t = p = \frac{1}{|S|} \sum_{t \in S} \mathbb{1}\{\text{RUN}(t, \text{code}) = \text{pass}\}, \forall t.$
    *   **Coder Local:** $c^{\text{Coder}}_{\text{build}} = 0.10, c^{\text{Coder}}_{\text{run}} = 0.10, c^{\text{Coder}}_{\text{nr}} = 0.80.$
        $s^{\text{Coder}}_{\text{nr},t} = \frac{1}{|T^{\text{gold}}|} \sum_{u \in T^{\text{gold}}} \mathbb{1}\{\text{RUN}(u, \text{code}) = \text{pass}\}$
    *   **Tester Local:** $c^{\text{Tester}}_{\text{valid}} = 0.20, c^{\text{Tester}}_{\text{nr}} = 0.80.$
        $s^{\text{Tester}}_{\text{nr},t} = \frac{1}{|U|} \sum_{u \in U} \mathbb{1}\{\text{RUN}(u, \text{code}^\star) = \text{pass} > \tau_{\text{mut}}\}$
*   **Sudoku Reward:**
    *   **Team:** $r^{\text{team}}_t = \mathbb{1}\{\text{SOLVED}(h)=\text{true}\}, \forall t.$
    *   **Reasoner Local:** $c^{\text{Reasoner}}_{\text{fmt}} = 0.1, c^{\text{Reasoner}}_{\text{legal}} = 0.1, c^{\text{Reasoner}}_{\text{prog}} = 0.80.$
        $s^{\text{Reasoner}}_{\text{prog},t} = \frac{1}{N^2} \sum_{r,c} \mathbb{1}\{G_{t-1}[r, c]=0, G_t[r, c]\neq0\}$
    *   **Tool Local:** $c^{\text{Tool}}_{\text{fmt}} = 0.10, c^{\text{Tool}}_{\text{exec}} = 0.10, c^{\text{Tool}}_{\text{san}} = 0.80.$
*   **Plan-Path Reward:**
    *   **Team:** $r^{\text{team}}_t = \begin{cases} 1 & \text{if at goal at } t \\ \max(0, (d_{t-1} - d_t)/d_0) & \text{otherwise} \end{cases}$
    *   **Planner Local:** $c^{\text{Planner}}_{\text{fmt}} = 0.10, c^{\text{Planner}}_{\text{leg}} = 0.10, c^{\text{Planner}}_{\text{sp}} = 0.80.$
    *   **Tool Local:** $c^{\text{Tool}}_{\text{fmt}} = 0.10, c^{\text{Tool}}_{\text{exec}} = 0.10, c^{\text{Tool}}_{\text{shape}} = 0.80.$
*   **Sokoban Reward:**
    *   **Team:** $r^{\text{team}}_t = \begin{cases} 1 & \text{if all boxes on goals at } t \\ b_t/B & \text{otherwise} \end{cases}$
    *   **Planner Local:** $c^{\text{Planner}}_{\text{fmt}} = 0.10, c^{\text{Planner}}_{\text{leg}} = 0.10, c^{\text{Planner}}_{\text{dlk}} = 0.80.$
    *   **Tool Local:** $c^{\text{Tool}}_{\text{fmt}} = 0.10, c^{\text{Tool}}_{\text{exec}} = 0.10, c^{\text{Tool}}_{\text{pot}} = 0.80.$

**7. All Mathematical Formulas and Symbol Definitions**

**MAS Setting Notation:**
*   $N$-agent LLM system modeled as Markov game $\mathcal{M} = \langle S, \{A_i\}_{i=1}^N, \mathcal{T}, \{r_i\}_{i=1}^N, T, H \rangle$.
*   $S$: state space. $A_i$: action space of agent $i$.
*   $\mathcal{T}$: transition function inducing intra-turn micro-transitions $s_{t,0} = s_t$ and $s_{t,i} = \mathcal{T}(s_{t,i-1}, a_{t,i}, i)$, culminating in $s_{t+1} = s_{t,N}$.
*   $r_i: A_i \to [0, 1]$: reward for agent $i$. $T$: turn horizon. $H$: optimization step horizon.
*   $o_{t,i} = o_i(s_t, h_t)$: observation of agent $i$.
*   $P_i(\cdot)$: role-specific prompt template.
*   $\Theta = \{\theta^{(m)}\}_{m=1}^M$: set of LLM parameter vectors ($1 \le M \le N$).
*   $\sigma: \{1, \dots, N\} \to \{1, \dots, M\}$: mapping assigning each agent to an LLM.
*   $a_{t,i}$: one LLM rollout (macro-action).
*   $g$: group key $g = \text{hash}(e, i, t)$.

**Group-based RL Advantage (Eq. 1):**
$\text{Ag}(a_t^{(c)}) = \frac{R(a_t^{(c)}) - \text{mean}(\{R(a_t^{(c)})\}_{c=1}^K)}{\text{Fnorm}(\{R(a_t^{(c)})\}_{c=1}^K)}$
Where $G = \{(a_t^{(1)}, R(a_t^{(1)})), \dots, (a_t^{(K)}, R(a_t^{(K)}))\}$ is the comparison group.

**Policy Optimization Loss (Eq. 2):**
$L(\theta^{(m)}) = -\mathbb{E}_{g \in \mathcal{B}_m} \left[ \frac{1}{K} \sum_{c=1}^K \min(r_g^{(c,m)}(\theta^{(m)}) A_g^{(c)}, \text{clip}(r_g^{(c,m)}(\theta^{(m)}), 1-\epsilon, 1+\epsilon) A_g^{(c)}) \right]$
Where $r(\theta) = \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)}$.
*   **Role-sharing policy ($M=1$):** $\mathcal{B}_1 = \bigcup_{i=1}^N D_i$. Single joint update.
*   **Role-specialized policies ($M=N$):** $\mathcal{B}_i = D_i$. Independent update.

**Agent-wise Credit Assignment (Eq. 3):**
$r_{t,i} = \alpha r_{t}^{\text{team}} + r_{t,i}^{\text{loc}}$

**Theoretical Justification:**
*   **Assumption 1 (Monotonicity of Verification Feedback):** $r_{\text{ver}}(s, a_1) > r_{\text{ver}}(s, a_2) \implies Q^*(s, a_1) \ge Q^*(s, a_2)$.
*   **Lemma 1 (Equivalence of Maximizers):** $\text{argmax}_a r_{\text{ver}}(s, a) \subseteq \text{argmax}_a Q^*(s, a)$.
*   **Proposition 1 (Optimality of Verifier-Greedy Policy):** Let $\pi_{\text{ver}}$ be a deterministic policy such that $\pi_{\text{ver}}(s) \in \text{argmax}_a r_{\text{ver}}(s, a)$ for all states $s$. Under Assumption 1, $\pi_{\text{ver}}$ is an optimal policy.

**8. Algorithm Pseudocode**

**Algorithm 1 AT-GRPO: Agent- and Turn-wise MAS RL Training**
**Require:** Markov game $\mathcal{M}$, policies $\Theta = \{\theta^{(m)}\}_{m=1}^M$, role mapping $\sigma$, sampling temperature $T_{\text{samp}}$, branches $K$, total steps $S$, batch size $E$, turn horizon $T$, termination condition $I_{\text{term}}$.
1:  for training step $s = 1, \dots, S$ do
2:      Initialize per-agent datasets $\{D_i\}_{i=1}^N \leftarrow \emptyset$. Resample $E$ environments.
3:      for each environment instance $e \in \{1, \dots, E\}$ in parallel do
4:          for $t = 0$ to $T - 1$ do
5:              $s_{t,0,e} \leftarrow s_{t,e}$ $\triangleright$ Initialize micro-step state
6:              for each agent $i \in \{1, \dots, N\}$ do
7:                  $\forall c \in \{1, \dots, K\}, a_{t,i,e}^{(c)} \sim \pi_{\theta(\sigma(i))}(\cdot | o_{t,i,e}; T_{\text{samp}})$; compute $r_{t,i,e}^{(c)}$ (Eq. 3)
8:                  Define group key $g \leftarrow \text{hash}(e, i, t)$ and compute advantages $\{A_g^{(c)}\}_{c=1}^K$ (Eq. 1).
9:                  Append $(g, o_{t,i,e}, \{a_{t,i,e}^{(c)}\}_{c=1}^K, \{A_g^{(c)}\}_{c=1}^K)$ to $D_i$.
10:                 $c^\star \leftarrow \text{argmax}_c r_{t,i,e}^{(c)}$; $a_{t,i,e} \leftarrow a_{t,i,e}^{(c^\star)}$. (Tree-structured sampling.)
11:                 $s_{t,i,e} \leftarrow \mathcal{T}(s_{t,i-1,e}, a_{t,i,e}, i)$ $\triangleright$ Agent-wise micro-transition
12:             end for
13:             $s_{t+1,e} \leftarrow s_{t,N,e}$ $\triangleright$ End-of-turn state
14:             if $I_{\text{term}}(s_{t+1,e})$ then break
15:             end if
16:         end for
17:     end for
18:     for each model $m \in \{1, \dots, M\}$ in parallel do
19:         Construct per-model batch $\mathcal{B}_m$, loss $L(\theta^{(m)})$ on $\mathcal{B}_m$ using Eq. 2 and update policy $m$.
20:     end for
21: end for