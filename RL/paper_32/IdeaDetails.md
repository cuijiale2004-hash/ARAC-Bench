### 1. Research Background and Existing Pain Points

**Research Background:**
Reinforcement Learning (RL) has demonstrated exceptional performance and achieved major breakthroughs across a diverse spectrum of decision-making challenges. The foundation of this success lies primarily in Deep RL, initiated by the introduction of the Deep Q-Network (DQN), which marked the first successful application of deep neural networks in RL. To make that happen, DQN introduces various techniques to mitigate mainly the deadly triad issue due to the usage of function approximators, off-policy data, and target bootstrapping. Particularly, DQN introduces the concept of target networks to mitigate the negative impact of the deadly triad issue, where the regression target is computed using a lagged copy of the online network to promote stability during training. This prevents the online network from directly chasing its own rapidly changing estimates, thereby mitigating the problem of moving targets.

**Existing Pain Points:**
While the target network has been highly successful in improving stability and convergence, it inherently slows down learning since updates are based on delayed targets. This naturally raises an important question: how can we accelerate learning by leveraging the most recent online estimates while still preserving the stability of the training process? Conversely, using the online network as a bootstrapped target is intuitively appealing, albeit well-known to lead to unstable learning. Recent studies have suggested that relying solely on the online network for target bootstrapping can improve performance in certain methods. Surprisingly, later findings indicate that reinstating a target network in these same approaches can further enhance results. In deep RL, the problem of moving targets is especially evident due to the use of neural networks and the resulting uncontrolled fluctuations in the values of unseen states. This issue becomes even more critical in value-based methods, where overestimation bias drives the online estimates to steadily increase over time. Building the regression target from the online network accelerates the overestimation bias, which degrades performance.

### 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:**
The aim is to identify a suitable criterion for incorporating recent online estimates without sacrificing the stability traditionally provided by the target network. This naturally suggests that the online and target networks should work side by side to achieve both fast and stable learning. The motivation is to leverage online estimates only when they are unlikely to introduce harmful overestimation or rapid fluctuations in the target. In cases where the online estimates are higher than the target one, the target network should be relied upon to ensure stability. The search is for an appropriate selection criterion that can mitigate the impact of overestimation bias when employing the online network for target computation, hence faster, yet stable learning.

**Scientific Questions:**
(Q1) Is the minimum operator an appropriate criterion for combining online and target estimates?
(Q2) Can the empirical evidence substantiate the rationale for adopting the minimum operator?

### 3. Overall Core Idea and Design Philosophy

The core idea is to obtain the best out of both worlds (fast learning and stability) by introducing a novel update rule that computes the target using the MINimum estimate between the Target and Online network, giving rise to the method, MINTO. The design philosophy is that by relying on the target network when the online estimate is relatively higher, MINTO reduces overestimation bias, alleviates the moving-target problem, and ensures stable learning. At the same time, by incorporating the online network when its estimate is lower, MINTO leverages fresher information, enabling faster learning. Through this simple, yet effective modification, MINTO enables faster and stable value function learning, by mitigating the potential overestimation bias of using the online network for bootstrapping. Notably, MINTO can be seamlessly integrated into a wide range of value-based and actor-critic algorithms with a negligible cost.

### 4. Core Innovation Points

1. Advocating for a principled combination of online and target networks when computing bootstrapped targets, enabling faster learning while preserving stability.
2. Introducing MINTO, a technique that is simple in design yet effective in practice, that computes regression targets as the minimum between online and target estimates, thereby mitigating the moving-target problem and reducing overestimation bias.
3. Demonstrating MINTO’s broad applicability and effectiveness by integrating it into both value-based and actor–critic methods, across online and offline RL settings.
4. Providing theoretical convergence guarantees for MINTO by casting it as a special case of the Generalized Q-learning framework and proving its operator satisfies non-expansion properties.

### 5. Overview of the Overall Technical Solution

The overall technical solution involves modifying the target computation in temporal-difference learning. The problem is defined as a Markov Decision Process (MDP) $\langle S, A, P, r, \mu, \gamma \rangle$, where $S$ is the state space, $A$ is the action space, $P : S \times A \rightarrow \Delta(S)$ is the transition distribution, $r : S \times A \rightarrow \Delta(\mathbb{R})$ is the reward distribution, $\mu$ is the initial state distribution, and $\gamma \in [0, 1)$ is a discount factor. A policy $\pi : S \rightarrow \Delta(A)$ induces an action-value function $Q^\pi(s, a) = \mathbb{E}_\pi[\sum_{t=0}^\infty \gamma^t r(s_t, a_t)|s_0 = s, a_0 = a]$. Q-Learning is an off-policy algorithm that aims to learn the state-action value function $Q$, of the optimal policy $\pi^*$, $Q^{\pi^*} = Q^*$, by utilizing the Bellman optimality equation. In the target value of standard DQN, the maximum expected value of the next state is approximated by applying the maximum operator on a single estimate (the target network). This introduces a maximization bias. MINTO modifies this bootstrapped target by applying the MINimum operator to the estimated values of the Target and Online networks. The method is lightweight, requiring only a single additional feedforward pass of the online network on the next state.

### 6. Detailed Module Design

**MINTO for Value-Based Methods (DQN):**
The bootstrapped target is computed using both the online parameters $\theta$ and target parameters $\bar{\theta}$. The minimum operator is applied element-wise over the action space before the maximum operator selects the greedy action. The regression loss is computed using the stop-gradient operator on the target to prevent backpropagation to the online network presented in the regression target.

**MINTO for Distributional RL (IQN):**
MINTO is integrated into Implicit Quantile Networks (IQN). The target computation is modified: the optimal action $a^*$ is selected by maximizing the minimum of the expected quantiles from the target and online networks. Then, the target quantiles are computed by taking the minimum between the target and online quantile values for the selected action. The loss is the standard quantile regression loss.

**MINTO for Offline RL (CQL):**
In Conservative Q-Learning (CQL), the target Q-values for next states are computed using the MINTO operator (minimum of target and online estimates), replacing the standard target network estimate. The total loss function combines the standard TD loss computed with the MINTO target and the conservative regularizer that penalizes out-of-distribution actions.

**MINTO for Actor-Critic Methods (SAC):**
For Soft Actor-Critic (SAC), the target computation for the critic network uses the MINTO operator. Given a sampled next action $a'_b \sim \pi_\phi(\cdot|s'_b)$, the target value is computed by taking the minimum between the target and online Q-values for this action, and then subtracting the entropy term (log probability of the policy). MINTO can be applied with a single Q-function critic or adapted to the Clipped Double Q-Learning (CDQ) trick. In the CDQ variant with MINTO, the online estimates from both critics can be aggregated to compute a single shared target value, or each critic is updated using only its corresponding online estimate when forming the target.

### 7. All Mathematical Formulas and Symbol Definitions

**Symbol Definitions:**
- $S, A, P, r, \mu, \gamma$: State space, action space, transition distribution, reward distribution, initial state distribution, discount factor.
- $\pi$: Policy mapping each state to a distribution over the action space.
- $Q^\pi(s, a)$: Action-value function, expected discounted cumulative return.
- $\pi^*$: Optimal policy.
- $Q^*$: Optimal action-value function.
- $y$: Target value for Q-learning update.
- $\alpha(s, a)$: Step-size for Q-learning update.
- $Q_\theta(s, a)$: Online neural network parameterized by $\theta$.
- $Q_{\bar{\theta}}(s, a)$: Target neural network parameterized by $\bar{\theta}$.
- $\lceil \cdot \rceil$: Stop gradient operator.
- $\mathcal{G}_{\text{MINTO}}$: MINTO target operator.
- $T$: Set of historical time indices.
- $N$: Ensemble size of action-value functions.
- $K$: Number of historical time action-values.
- $Z_\theta$: Quantile function parameterized by $\theta$.
- $\tau$: Quantile value sampled from $U[0, 1]$.
- $\rho_{\tau_i}^\kappa$: Quantile regression loss function.
- $\mathcal{L}_{\text{TD}}$: Standard TD loss.
- $\mathcal{L}_{\text{CQL}}$: Conservative regularizer loss.
- $\alpha$: Tradeoff factor for CQL or temperature parameter for SAC.
- $H_{\text{target}}$: Target entropy for SAC.

**Mathematical Formulas:**

Bellman optimality equation:
$$Q^*(s, a) = \mathbb{E}[r(s, a) + \gamma \max_{a' \in A} Q^*(s', a')]$$

Q-learning update rule:
$$Q(s, a) \leftarrow Q(s, a) + \alpha(s, a)[y - Q(s, a)]$$

Standard target value:
$$y = r + \gamma \max_{a' \in A} Q(s', a')$$

MINTO bootstrapped target:
$$y = r + \gamma \max_{a' \in A} \min(Q_{\bar{\theta}}(s', a'), Q_{\theta}(s', a'))$$

MINTO regression loss:
$$\mathcal{L}(\theta) = \frac{1}{2} (\lceil y \rceil - Q_{\theta}(s, a))^2$$

MINTO operator definition for convergence analysis:
$$\mathcal{G}_{\text{MINTO}}(Q_s) = \max_{a \in A} \min_{j \in T} Q_{sa}(j)$$

Condition A1.1 Proof:
Assume $Q_{sa}(j)$ identical for all $j \in T$ and all $a \in A$.
$$\mathcal{G}_{\text{MINTO}}(Q_s) = \max_{a \in A} (\min_{j \in T} Q_{sa}(j)) = \max_{a \in A} Q_{sa}(j)$$

Condition A1.2 Proof:
Let $Q_s$ and $Q'_s$ be two distinct sets of historical Q-values for a given state $s$, assuming $N = 1$.
$$|\mathcal{G}_{\text{MINTO}}(Q_s) - \mathcal{G}_{\text{MINTO}}(Q'_s)| = |\max_{a \in A} (\min_{j \in T} Q_{sa}(j)) - \max_{a \in A} (\min_{j \in T} Q'^{sa}_j)|$$
$$\le \max_{a \in A} |\min_{j \in T} Q_{sa}(j) - \min_{j \in T} Q'^{sa}_j|$$
$$\le \max_{a \in A} (\max_{j \in T} |Q_{sa}(j) - Q'^{sa}_j|)$$
$$= \max_{a \in A, j \in T} |Q_{sa}(j) - Q'^{sa}_j|$$
$$\le \max_{a, j} |Q_{sa}(j) - Q'^{sa}_j|$$

IQN+MINTO target quantiles computation:
$$a^* = \text{argmax}_{a'} \min(\frac{1}{N} \sum_{i=1}^N Z_{\bar{\theta}}(s'_b, a', \tau_i), \frac{1}{N} \sum_{i=1}^N Z_{\theta}(s'_b, a', \tau_i))$$
$$y_j = r_b + \gamma \min(Z_{\bar{\theta}}(s'_b, a^*, \tau'_j), Z_{\theta}(s'_b, a^*, \tau'_j))$$

IQN+MINTO quantile regression loss:
$$\mathcal{L}(\theta) = \frac{1}{N N' B} \sum_{b=1}^B \sum_{i=1}^N \sum_{j=1}^{N'} \rho_{\tau_i}^\kappa (\lceil y_j \rceil - Z_\theta(s_b, a_b, \tau_i))$$

CQL+MINTO standard TD loss:
$$\mathcal{L}_{\text{TD}}(\theta) = \frac{1}{2B} \sum_{b=1}^B (\lceil y_b \rceil - Q_\theta(s_b, a_b))^2$$

CQL conservative regularizer:
$$\mathcal{L}_{\text{CQL}}(\theta) = \alpha \cdot \frac{1}{B} \sum_{b=1}^B [\log \sum_a \exp(Q_\theta(s_b, a)) - Q_\theta(s_b, a_b)]$$

CQL+MINTO total loss:
$$\mathcal{L}(\theta) = \mathcal{L}_{\text{TD}}(\theta) + \mathcal{L}_{\text{CQL}}(\theta)$$

SAC+MINTO target:
$$y_b = r_b + \gamma [\min (Q_{\bar{\theta}}(s'_b, a'_b), Q_{\theta}(s'_b, a'_b)) - \alpha \log \pi_\phi(a'_b|s'_b)]$$

SAC+MINTO critic loss:
$$\mathcal{L}(\theta) = \frac{1}{2B} \sum_{b=1}^B (\lceil y_b \rceil - Q_\theta(s_b, a_b))^2$$

SAC policy gradient:
$$J_\pi(\phi) = \frac{1}{B} \sum_{b=1}^B (\alpha \log \pi_\phi(a_b|s_b) - Q_\theta(s_b, a_b))$$

SAC temperature gradient:
$$J(\alpha) = \frac{1}{B} \sum_{b=1}^B -\alpha (\log \pi_\phi(a_b|s_b) + H_{\text{target}})$$

### 8. Algorithm Pseudocode

**Algorithm 1 MINTO**
1: Initialize online and target paramters $\bar{\theta}, \theta$, an empty replay buffer $\mathcal{B}$, and $t_{\text{total}} = 0$.
2: repeat
3:   Sample an initial state $s_0$ from $\mu$
4:   for $t = 0$ to $n_{\text{horizon}}$ do
5:     Sample an action $a_t \sim \epsilon$-greedy($Q_\theta(s_t, \cdot)$)
6:     Execute action $a_t$ in environment, observe reward $r_t$ and next state $s_{t+1}$
7:     Store transition $(s_t, a_t, r_t, s_{t+1})$ in $\mathcal{B}$
8:     Sample a batch of $B$ transitions $(s_b, a_b, r_b, s'_b)_{b=1}^B$ from $\mathcal{B}$
9:     Compute the TD-target $y_b = r_b + \gamma \max_{a'} \min(Q_{\bar{\theta}}(s'_b, a'), Q_{\theta}(s'_b, a'))$
10:    if $s'_b$ is terminal then $y_b \leftarrow r_b$
11:    Compute the loss $\mathcal{L}(\theta) = \frac{1}{2B} \sum_{b=1}^B (\lceil y_b \rceil - Q_\theta(s_b, a_b))^2$
12:    Obtain the gradient $\nabla_\theta \mathcal{L}$ and perform an update step
13:    Every $T$ steps update the target network $\bar{\theta} \leftarrow \theta$
14:    if $s_{t+1}$ is terminal break
15:  end for
16:  $t_{\text{total}} \leftarrow t_{\text{total}} + t$
17: until $t_{\text{total}} \ge n_{\text{total}}$

**Algorithm 2 IQN+MINTO**
1: Initialize online and target network parameters $\theta, \bar{\theta}$, and replay buffer $\mathcal{B}$, and $t_{\text{total}} = 0$.
2: repeat
3:   Sample initial state $s_0 \sim \mu$
4:   for $t = 0$ to $n_{\text{horizon}}$ do
5:     Select action $a_t \sim \epsilon$-greedy$( \frac{1}{N} \sum_{i=1}^N Z_\theta(s_t, a, \tau_i) )$ where $\tau_i \sim U [0, 1]$
6:     Execute $a_t$, observe reward $r_t$ and next state $s_{t+1}$
7:     Store transition $(s_t, a_t, r_t, s_{t+1})$ in $\mathcal{B}$
8:     Sample minibatch $(s_b, a_b, r_b, s'_b)_{b=1}^B$ from $\mathcal{B}$
9:     Sample $\{\tau_i\}_{i=1}^N, \{\tau'_j\}_{j=1}^{N'} \sim U [0, 1]$
10:    Compute target quantiles:
        $a^* = \text{argmax}_{a'} \min( \frac{1}{N} \sum_{i=1}^N Z_{\bar{\theta}}(s'_b, a', \tau_i), \frac{1}{N} \sum_{i=1}^N Z_{\theta}(s'_b, a', \tau_i) )$
        $y_j = r_b + \gamma \min(Z_{\bar{\theta}}(s'_b, a^*, \tau'_j), Z_{\theta}(s'_b, a^*, \tau'_j))$
11:    if $s'_b$ terminal then $y_j \leftarrow r_b$
12:    Compute quantile regression loss:
        $\mathcal{L}(\theta) = \frac{1}{N N' B} \sum_{b=1}^B \sum_{i=1}^N \sum_{j=1}^{N'} \rho_{\tau_i}^\kappa (\lceil y_j \rceil - Z_\theta(s_b, a_b, \tau_i))$
13:    Perform gradient step on $\nabla_\theta \mathcal{L}$
14:    Every $T$ steps update target network $\bar{\theta} \leftarrow \theta$
15:    if $s_{t+1}$ is terminal break
16:  end for
17:  $t_{\text{total}} \leftarrow t_{\text{total}} + t$
18: until $t_{\text{total}} \ge n_{\text{total}}$

**Algorithm 3 CQL+MINTO**
1: Initialize online and target critic parameters $\theta, \bar{\theta}$, an empty replay buffer $\mathcal{B}$, and $t_{\text{total}} = 0$.
2: Load offline dataset $\mathcal{D}$ into $\mathcal{B}$
3: repeat
4:   Sample a batch of $B$ transitions $(s_b, a_b, r_b, s'_b)_{b=1}^B$ from $\mathcal{B}$
5:   Compute target Q-values for next states:
      $y_b = r_b + \gamma \max_{a'} \min(Q_{\bar{\theta}}(s'_b, a'), Q_{\theta}(s'_b, a'))$
6:   if $s'_b$ is terminal then $y_b \leftarrow r_b$
7:   Compute standard TD loss:
      $\mathcal{L}_{\text{TD}}(\theta) = \frac{1}{2B} \sum_{b=1}^B (\lceil y_b \rceil - Q_\theta(s_b, a_b))^2$
8:   Compute conservative regularizer:
      $\mathcal{L}_{\text{CQL}}(\theta) = \alpha \cdot \frac{1}{B} \sum_{b=1}^B [ \log \sum_a \exp(Q_\theta(s_b, a)) - Q_\theta(s_b, a_b) ]$
9:   Total loss:
      $\mathcal{L}(\theta) = \mathcal{L}_{\text{TD}}(\theta) + \mathcal{L}_{\text{CQL}}(\theta)$
10:  Update critic parameters: $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}(\theta)$
11:  Every $T$ steps update target network: $\bar{\theta} \leftarrow \theta$
12:  $t_{\text{total}} \leftarrow t_{\text{total}} + 1$
13: until $t_{\text{total}} \ge n_{\text{steps}}$

**Algorithm 4 SAC+MINTO.**
1: Initialize policy parameters $\phi$, critic’s online and target parameters $\theta, \bar{\theta}$, an empty replay buffer $\mathcal{B}$, and $t_{\text{total}} = 0$.
2: repeat
3:   Sample an initial state $s_0$ from $\mu$
4:   for $t = 0$ to $n_{\text{horizon}}$ do
5:     Sample action $a_t \sim \pi_\phi(\cdot|s_t)$
6:     Execute $a_t$ in environment, observe reward $r_t$ and next state $s_{t+1}$
7:     Store transition $(s_t, a_t, r_t, s_{t+1})$ in $\mathcal{B}$
8:     Sample a batch of $B$ transitions $(s_b, a_b, r_b, s'_b)_{b=1}^B$ from $\mathcal{B}$
9:     Sample $a'_b \sim \pi_\phi(\cdot|s'_b)$ and compute $\log \pi_\phi(a'_b|s'_b)$
10:    Compute target:
        $y_b = r_b + \gamma [ \min (Q_{\bar{\theta}}(s'_b, a'_b), Q_{\theta}(s'_b, a'_b)) - \alpha \log \pi_\phi(a'_b|s'_b) ]$
11:    if $s'_b$ is terminal then $y_b \leftarrow r_b$
12:    Update critic by minimizing
        $\mathcal{L}(\theta) = \frac{1}{2B} \sum_{b=1}^B (\lceil y_b \rceil - Q_\theta(s_b, a_b))^2$
13:    Update policy $\phi$ using gradient of
        $J_\pi(\phi) = \frac{1}{B} \sum_{b=1}^B ( \alpha \log \pi_\phi(a_b|s_b) - Q_\theta(s_b, a_b) )$
14:    Optionally update temperature $\alpha$ by minimizing
        $J(\alpha) = \frac{1}{B} \sum_{b=1}^B -\alpha (\log \pi_\phi(a_b|s_b) + H_{\text{target}})$
15:    Every $T$ steps update target critic: $\bar{\theta} \leftarrow \tau\theta + (1 - \tau)\bar{\theta}$
16:    if $s_{t+1}$ is terminal break
17:  end for
18:  $t_{\text{total}} \leftarrow t_{\text{total}} + t$
19: until $t_{\text{total}} \ge n_{\text{total}}$