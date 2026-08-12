# Research Idea and Implementation Plan Extraction

## 1. Research Background and Existing Pain Points
**Research Background:** 
Cooperative multi-agent reinforcement learning (c-MARL) has achieved remarkable performance in areas such as autonomous driving, 5G networks, robotics, and smart grids, as it allows agents to learn distributed policies for complex sequential tasks. Nonetheless, the failure or the compromise of even a single agent, either through direct manipulation of its actions or by corrupting its observations, can degrade the overall team performance, calling for policies that are robust against faults and adversarial attacks. Existing approaches for obtaining robust policies rely on dataset augmentation or on adversarial training. Dataset augmentation involves introducing one or more adversarial perturbations during training, allowing agents to learn under adversarial and nominal conditions simultaneously. The alternative approach is based on jointly training the benign and the adversarial agents, typically formulated as a zero-sum Stackelberg game, and a saddle-point equilibrium in policies is sought after.

**Existing Pain Points:**
Robust learning based on saddle-point equilibria against a worst case adversary found using gradient descent has three fundamental limitations:
1. **Assumption of worst case adversary:** It relies on the assumption of a worst case adversary, which fails to capture adversaries with an objective other than minimizing the team reward as well as non-cooperative behavior due to failure. These may deviate substantially from worst-case attacks, and thus the defender’s max–min policy may be far from optimal considering the actual adversarial strategy, resulting in poor team performance.
2. **Local stationary points:** The optimization problem solved is inherently non-convex, and learning algorithms are prone to converge to local stationary points, which may not be globally optimal. As a result, the saddle-point policies are local Stackelberg equilibria, potentially far from the equilibrium sought after.
3. **Overfitting to single adversarial policy:** Exposure to perturbed versions of a single adversarial policy during training can cause the agents’ representation of adversarial dynamics to overfit. Consequently, when faced with a different type of adversarial behavior at deployment, the agents may fail to adapt their policies to the previously learned max–min strategy. In such cases, they may not even achieve the minimum performance guarantee that the max–min strategy is theoretically expected to provide.

## 2. Core Research Motivation and Scientific Questions
**Core Research Motivation:** 
To address the limitations of learning a single max-min policy, the research aims to train robust MARL policies that can adapt to a diverse set of adversarial behaviors. Instead of learning a single max–min policy, the approach partitions the set of adversarial policies into disjoint subsets, defined by the range of team reward they would impose, and then computes a policy that is robust to a representative adversarial policy from each subset. Although this cannot completely eliminate the problem of local stationary points, it mitigates the problem by restricting the search to smaller, isolated feasible sets. Moreover, the subsets are constructed so that adversarial policies in different subsets exhibit distinct behaviors. The defender’s MARL policy is then trained to adapt based on its belief of adversarial behavior.

**Scientific Questions:**
- How to model the uncertainty about the adversarial objective in c-MARL at deployment time?
- How to discretize the continuous adversarial type space to ensure exposure to a diverse set of adversarial policies during training?
- How to compute the adversarial policies for each type when the objective and constraints correspond to different MDPs (externally constrained RL)?
- How to compute a perfect Bayesian equilibrium (PBE) of the resulting finite-type Bayesian game to obtain robust Bayesian c-MARL policies?

## 3. Overall Core Idea and Design Philosophy
The overall core idea is to introduce a Bayesian Dec-POMDP game model with a continuum of adversarial types, corresponding to distinct attack objectives, to address the uncertainty about the adversarial objective. To compute a perfect Bayesian equilibrium (PBE) of the game, a novel partitioning scheme of adversarial policies based on their performance against a reference c-MARL policy is introduced. This allows casting the problem as finding a PBE in a finite-type Bayesian game. To compute the adversarial policies, the concept of an externally constrained reinforcement learning problem is introduced, where the objective and constraints correspond to different MDPs, and a provably convergent algorithm along with a practically efficient variant is presented. Building on this, a simultaneous gradient update scheme is used to obtain robust Bayesian c-MARL policies.

## 4. Core Innovation Points
1. **Bayesian Dec-POMDP Game Model and Type Discretization:** Introduction of a Bayesian Dec-POMDP game model with a continuum of adversarial types and a novel criterion for discretizing the type space based on reference value to ensure exposure to a diverse set of adversarial policies during training. Based on the perfect Bayesian equilibrium of the game, the Bayesian regret is formulated as the objective to characterize the robustness of a policy.
2. **Externally-Constrained RL Concept:** Introduction of the concept of an externally-constrained RL to find the adversarial policies of different types. This is fundamentally different from standard constrained RL as the objective and constraints correspond to different MDPs and consequently different trajectories.
3. **Convergent Algorithm for Externally-Constrained RL:** Proposal of a provably convergent algorithm and a practically efficient variant (EC-PPO) to solve the externally-constrained RL problem, incorporating PPO loss to mitigate noise sensitivity and eliminate adaptive step sizes.
4. **BATPAL Framework:** Design of an end-to-end adversarial learning framework, termed BATPAL (Bayesian Type-Partitioned Adversarial Learning), which uses simultaneous gradient updates (two-timescale stochastic simultaneous gradient descent–ascent) to derive Bayesian robust c-MARL policies.

## 5. Overview of the Overall Technical Solution
The overall technical solution consists of the following steps:
1. **Pre-training Phase:** Train a cooperative reference policy $\pi_0$ in a non-adversarial environment to obtain the maximum expected initial state value $V_{max}$ and the minimum expected initial state value $V_v^{min}$ under reference policy for each victim.
2. **Reference-Value Based Partitioning:** Define the severity of an adversarial policy based on the expected initial state value it imposes under the reference policy. Partition the continuous adversarial type space into $K$ discrete subsets based on severity ranges.
3. **Adversarial Policy Computation via Externally Constrained RL:** For each discrete adversarial type, formulate the problem of finding the worst-case policy within that severity partition as an externally constrained RL problem. Use the log barrier method and EC-PPO to solve for the representative adversarial policies.
4. **Bayesian Adversarial MARL Training:** Formulate the problem as a partially observable stochastic game with $N+1$ players. Train a belief network for non-victim agents to estimate the adversarial type. Use simultaneous gradient updates where the adversarial policy is updated faster than the c-MARL policy to approximate the min-oracle, ultimately finding the Perfect Bayesian Equilibrium policies.

## 6. Detailed Module Design

### 6.1 Bayesian Dec-POMDP Game Model
The model is defined as $M_B = (\mathcal{N}, S, \{\Theta_i\}_{i\in\mathcal{N}}, \{A_i\}_{i\in\mathcal{N}}, R, P, \{\Omega_i\}_{i\in\mathcal{N}}, O, \mu, \gamma)$, where $\Theta_i$ is the type space of agent $i$, extending the Dec-POMDP formulation. The type $\theta_i \in \Theta_i$ captures the uncertainty about the reward function. $\Theta_i = [0, 1]$. Type $\theta_i = 0$ corresponds to maximizing team reward; $\theta_i > 0$ corresponds to maximizing some other reward function; $\theta_i = 1$ is the worst-case adversary minimizing team reward. The policy $\pi_i(a_i^t|\tau_i^t, \theta_i)$ depends on its type and beliefs $b_i(\tau_i^t)$ about the types of other agents.

### 6.2 Reference-Value Based Partitioning Module
The partitioning maps adversarial types $\Theta_v$ to discrete types $\hat{\Theta}_i = \{0, 1, \ldots, K\}$ based on severity against a reference policy $\pi_0 \in \arg\max_\pi V^\pi$.
- **Severity Definition:** $\eta_{\rho_v} = \frac{V_{max} - V^{\pi_0, \rho_v}}{V_{max} - V_v^{min}}$, where $V_{max} = V^{\pi_0}$ and $V_v^{min} = \min_{\rho_v} V^{\pi_0, \rho_v}$.
- **Partitioning Logic:** A policy $\rho_v$ belongs to adversarial type $z = (v, k)$ if $\eta_{\rho_v} \in (\frac{k-1}{K}, \frac{k}{K}]$. The set of all such policies is $\Pi_z$. The discrete type space support is $Z = \{(v, k) : v \in \mathcal{N}, k \in \{1, 2, \ldots, K\}\} \cup \{0\}$.
- **PBE Formulation:** The PBE in $\hat{M}_B$ corresponds to a policy $\pi^* = (\pi_1^*, \ldots, \pi_N^*)$ such that for all $i \in \mathcal{N}$ and $\tau^i$:
  $\pi_i^*(.| \tau^i, \theta_i = 0) \in \arg\max_{\pi_i} \mathbb{E}_{b_i(z|\tau^i)} [ \min_{\rho_v \in \Pi_z} V^{\pi^*, \rho_v} ]$.
- **Diversity Guarantee:** Proposition 3.2 establishes a bound on the KL divergence between two arbitrary policies: $\mathbb{E}_{s\sim d^{\pi_0, \rho_{v,\theta_1}}_\mu}[D_{KL}(\rho_{v,\theta_1}(s)||\rho_{v,\theta_2}(s))] \ge \frac{(1-\gamma)^2}{2} |V^{\pi_0, \rho_{v,\theta_1}} - V^{\pi_0, \rho_{v,\theta_2}}|^2$. Proposition 3.3 provides a lower bound on the diversity of generated adversarial policies: $Div(\{\rho_{v,k}^*\}_{k=1}^K) \ge \frac{(1-\gamma)^2(V_{max} - V_v^{min})^2(K-2)}{12K}$.
- **Regret Bound:** Proposition 3.4 bounds the regret for an arbitrary adversarial policy $\hat{\rho}_v \in \Pi_z$: $R_{\hat{\rho}_v}(\pi_z^*) \le \frac{k(V_{max} - V_v^{min})}{K}$.

### 6.3 Externally Constrained RL Module
To solve the inner minimization for a given $z=(v,k)$ and non-victim policy $\pi$, the problem is formulated as:
$\min_\rho \mathbb{E}_{s\sim\mu}[V^{\rho(1)}(s)] \quad \text{s.t.} \quad l \le \mathbb{E}_{s\sim\mu}[V^{\rho(0)}(s)] \le h$
where $V^{\rho(1)}$ and $V^{\rho(0)}$ denote the value functions under POMDP1 (induced by fixing $\pi$) and POMDP0 (induced by fixing $\pi_0$), and $l$ and $h$ are boundaries of severity type $k$.

- **Log Barrier Approximation:** The constrained problem is approximated via an unconstrained problem:
  $\min_\rho V^{\rho(1)} - \lambda \log(V^{\rho(0)} - l) - \lambda \log(h - V^{\rho(0)})$
- **EC-PPO Gradient Calculation:** To address noise sensitivity near boundaries and avoid adaptive step sizes, PPO clipping is incorporated. The gradient calculation is:
  $\hat{g}^{EC-PPO}_{\psi_n} = \begin{cases} \nabla^{PPO,\psi_n}_{(1)} - \lambda(\frac{1}{\hat{V}^{\psi_n}_{(0)} - l} - \frac{1}{h - \hat{V}^{\psi_n}_{(0)}})\hat{\nabla}^{\psi_n}_{(0)}, & \text{if } l+\zeta \le \hat{V}^{\psi_n}_{(0)} \le h-\zeta \\ \text{sign}(\hat{V}^{\psi_n}_{(0)} - \frac{1}{2}(l+h))\hat{\nabla}^{\psi_n}_{(0)}, & \text{otherwise} \end{cases}$
  where $\nabla^{PPO,\psi_n}_{(1)}$ is the gradient of the PPO objective function and $\zeta > 0$ is a small value to prevent gradient explosion.

### 6.4 Bayesian Adversarial MARL Training Module
The game $\hat{M}_B$ is equivalent to a partially observable stochastic game $G$ with $N+1$ players. The objective is to find $\arg\max_\omega \min_\psi \bar{V}^{\omega, \psi}$, where $\pi_{\omega_i}$ is the non-adversarial policy and $\rho_\psi$ is the adversarial policy.
- **Belief Network:** A parametrized function approximator $b_{\chi_i}(\theta^{-i}|\tau^i)$, implemented using an RNN, takes $\tau^i$ as input and is trained against the true type using cross-entropy loss.
- **Simultaneous Gradient Updates (Two-timescale):**
  Adversary update: $\psi_{n+1} = \psi_n - \alpha_n \hat{g}^{EC-PPO}_\psi(\omega_n, \psi_n)$
  Defender update: $\omega_{n+1} = \omega_n + \beta_n \hat{g}_\omega(\omega_n, \psi_n)$
  By selecting $\alpha_n \ge \beta_n$, the adversary's policy serves as an approximate min-oracle. $\hat{g}_\omega(\omega_n, \psi_n)$ is computed by feeding $(b_i, \tau^i)$ into the policy network and using the value estimate from critic $V_{\phi(1)}(\bar{s})$ in a standard actor-critic algorithm.

## 7. All Mathematical Formulas and Symbol Definitions
- $\mathcal{N} = \{1, 2, ..., N\}$: Set of agents.
- $S, A_i, \Omega_i$: Set of states, actions of agent $i$, observations of agent $i$.
- $R(s_t, a_t), P(s_{t+1}|s_t, a_t), O(o_t|s_t)$: Reward function, state transition probability, conditional observation probabilities.
- $\mu, \gamma$: Initial state distribution, discount factor.
- $V^\pi(s) = \mathbb{E}[\sum_{t=0}^\infty \gamma^t R(s_t, a_t) | s_0 = s]$: Value function.
- $V^\pi = \mathbb{E}_{s_0\sim\mu}[V^\pi(s_0)]$: Expected initial state value.
- $x_{-i}$: Collection of $x_j$ for all agents $j \neq i$.
- $\Theta_i$: Type space of agent $i$.
- $\rho_{v,\theta_v} = \pi_v(\cdot|\tau_v, \theta_v)$: Adversarial policy.
- $R(\pi) = \mathbb{E}_{(v,\theta_v)\sim b_0}[R_{\rho_{v,\theta_v}}(\pi)] = \mathbb{E}_{(v,\theta_v)\sim b_0}[\max_{\pi'}(V^{\pi',\rho_{v,\theta_v}}) - V^{\pi,\rho_{v,\theta_v}}]$: Bayesian regret.
- $\hat{\Theta}_i = \{0, 1, \ldots, K\}$: Discrete type space for agent $i$.
- $Z = \{(v, k) : v \in \mathcal{N}, k \in \{1, 2, ..., K\}\} \cup \{0\}$: Set representing support of discrete types.
- $V_{max} = V^{\pi_0}$: Maximum expected initial state value under reference policy.
- $V_v^{min} = \min_{\rho_v} V^{\pi_0, \rho_v}$: Minimum expected initial state value under reference policy for victim $v$.
- $\eta_{\rho_v} = \frac{V_{max} - V^{\pi_0, \rho_v}}{V_{max} - V_v^{min}}$: Severity of adversarial policy $\rho_v$.
- $\Pi_z$: Set of policies belonging to adversarial type $z = (v, k)$.
- $\pi_i^*(.| \tau^i, \theta_i = 0) \in \arg\max_{\pi_i} \mathbb{E}_{b_i(z|\tau^i)} [ \min_{\rho_v \in \Pi_z} V^{\pi^*, \rho_v} ]$: PBE formulation.
- $\min_\rho \mathbb{E}_{s\sim\mu}[V^{\rho(1)}(s)] \quad \text{s.t.} \quad l \le \mathbb{E}_{s\sim\mu}[V^{\rho(0)}(s)] \le h$: Externally constrained RL problem.
- $\min_\rho V^{\rho(1)} - \lambda \log(V^{\rho(0)} - l) - \lambda \log(h - V^{\rho(0)})$: Log barrier approximation.
- $g_\psi = \frac{1}{1-\gamma} \mathbb{E}_{s\sim d^{(1)}, a\sim \rho_\psi(.|s)}[\nabla_\psi \log \rho_\psi(a|s) A^{\rho_\psi}_{(1)}(s, a)] - \lambda \frac{1}{1-\gamma} (\frac{1}{\mathbb{E}_{s\sim\mu}[V^{\rho_\psi}_{(0)}(s)] - l} - \frac{1}{h - \mathbb{E}_{s\sim\mu}[V^{\rho_\psi}_{(0)}(s)]}) \mathbb{E}_{s\sim d^{(0)}, a\sim \rho_\psi(.|s)}[\nabla_\psi \log \rho_\psi(a|s) A^{\rho_\psi}_{(0)}(s, a)]$: Policy gradient of the log-barrier objective.
- $\psi_{n+1} = \psi_n - \alpha_n \hat{g}_{\psi_n}$: Stochastic update rule for policy parameters.
- $\hat{g}_{\psi_n} = \frac{1}{1-\gamma} [\hat{\nabla}^{\psi_n}_{(1)} - \lambda(\frac{1}{\hat{V}^{\psi_n}_{(0)} - l} - \frac{1}{h - \hat{V}^{\psi_n}_{(0)}}) \hat{\nabla}^{\psi_n}_{(0)}]$: Gradient estimate.
- $\hat{V}^{\psi_n}_{(0)} = \frac{1}{M} \sum_{m=1}^M [(\sum_{t=0}^{T_m-1} \gamma^t r_{t,m,(0)}) + V_{\phi(0)}(s_{T_m, m,(0)})]$: Value estimate.
- $\hat{g}^{EC-PPO}_{\psi_n} = \begin{cases} \nabla^{PPO,\psi_n}_{(1)} - \lambda(\frac{1}{\hat{V}^{\psi_n}_{(0)} - l} - \frac{1}{h - \hat{V}^{\psi_n}_{(0)}})\hat{\nabla}^{\psi_n}_{(0)}, & \text{if } l+\zeta \le \hat{V}^{\psi_n}_{(0)} \le h-\zeta \\ \text{sign}(\hat{V}^{\psi_n}_{(0)} - \frac{1}{2}(l+h))\hat{\nabla}^{\psi_n}_{(0)}, & \text{otherwise} \end{cases}$: EC-PPO gradient.
- $\psi_{n+1} = \psi_n - \alpha_n \hat{g}^{EC-PPO}_\psi(\omega_n, \psi_n)$: Adversary policy update in BATPAL.
- $\omega_{n+1} = \omega_n + \beta_n \hat{g}_\omega(\omega_n, \psi_n)$: Defender policy update in BATPAL.

## 8. Algorithm Pseudocode
**Algorithm 1 Adversarial Learning in BATPAL**
**Input Networks:** The reference policy networks $\omega_0^i$, the policy networks $\omega_i$, the critic $\phi(1)$, the reference critic $\phi(0)$, the belief networks $\chi_i$, and the adversarial policies $\psi_z$
1: Pretraining:
2: Train the c-MARL team in a non-adversarial environment and obtain $\omega_0$, $V_{max}$, and $V_v^{min}$
3: Adversarial Training:
4: for each iteration do
5:  Sample $\hat{\theta} \sim p$, and set $z = (k, v)$ or $z = 0$.
6:  for $m = 1, 2, \ldots, M$ do ▷ Storing experiences corresponding to POMDP1
7:   Sample the initial state and observations $s_{0,(1)}, o_{0,(1)}$
8:   for $t = 0, 1, 2, \ldots, T_m - 1$ do
9:    Sample non-victim actions $a_{i,(1)}^t \sim \pi_{\omega_i}(.|\tau_{i,(1)}^t, b_i^t)$, where $b_i^t = b_{\chi_i}(\tau_{i,(1)}^t)$
10:   Sample adversary action if $z \neq 0$, $a_v^t \sim \rho_{\psi_z}$
11:   Set the joint action profile $a^t = (a_v^t, a_{-v}^t)$ if $z \neq 0$
12:   Environment transitions and $s_{t+1,(1)}, r_{t,(1)}$ are obtained
13:   Store the transition history $H_{t,(1)}$
14:  end for
15: end for
16: if $z \neq 0$ then ▷ Storing experiences corresponding to POMDP0
17:  for $m = 1, 2, \ldots, M$ do
18:   Sample the initial state and observations $s_{0,(0)}, o_{0,(0)}$
19:   for $t = 0, 1, 2, \ldots, T_m - 1$ do
20:    Sample non-victim actions $a_{i,(0)}^t \sim \pi_{\omega_0^i}(.|\tau_{i,(0)}^t)$
21:    Sample adversary action $a_v^t \sim \rho_{\psi_z}$
22:    Set the joint action profile $a^t = (a_v^t, a_{-v}^t)$
23:    Environment transitions and $s_{t+1,(0)}, r_{t,(0)}$ are obtained
24:    Store the transition history $H_{t,(0)}$
25:   end for
26:  end for
27: end if
28: Update the critics $\phi(0), \phi(1)$ using the advantages obtained by $H_{t,(0)}$ and $H_{t,(1)}$
29: Update $\chi_i$ using true types in $H_{t,(1)}$
30: Update $\omega$ using $\omega_{n+1} = \omega_n + \beta_n \hat{g}_\omega(\omega_n, \psi_n)$
31: Update $\psi_z$ using $\psi_{n+1} = \psi_n - \alpha_n \hat{g}^{EC-PPO}_\psi(\omega_n, \psi_n)$
32: end for