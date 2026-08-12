
## 1. Research Background and Existing Pain Points

**Research Background:**
Learning how to reach goals in an environment is a longstanding challenge in AI. It is natural for humans to use inherent ideas of distances to represent task progress (e.g., GPS distance to destination, recipe duration). The AI problem of reaching goals presents a rich structure (formally, an optimal substructure property) that can be exploited to decompose hard problems into easy problems, and reinforcement learning (RL) has been used to address such problems. In Goal-Conditioned Reinforcement Learning (GCRL), the key question is how to estimate the temporal distance between pairs of observations. Current offline approaches tend to separate Temporal Difference (TD) learning and Monte Carlo (MC) learning.

**Existing Pain Points:**
1.  **Separation of TD and MC methods:** TD learning leverages local updates to provide optimality guarantees (theoretically recovering the optimal Q function $Q^*$), but it often performs worse than Monte Carlo methods that perform global updates. Conversely, MC methods (e.g., with multi-step returns) perform well in practice but lack optimality guarantees and can only recover the behavior Q function $Q^\beta$.
2.  **Degradation with Horizon Length:** The effectiveness of TD methods degrades with increasing horizon length due to compounding TD errors. This makes offline RL easy to scale on short-horizon tasks but difficult for long-horizon tasks requiring complex reasoning.
3.  **Difficulties in Finding Optimal Temporal Distance:** MC methods face difficulties in finding the optimal temporal distance.
4.  **Stitching and Horizon Generalization Deficits:** Many past attempts at leveraging the optimal substructure property in modern high-dimensional, stochastic settings still need much work. Existing methods struggle with horizon generalization (executing longer tasks when only similar shorter tasks were seen in training) and stitching (taking the shortest route possible by combining segments of trajectories).

## 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:**
The effectiveness degradation of both TD and MC methods from a combination of increasing horizon length (for TD) or difficulties in finding optimal temporal distance (for MC) presents an intriguing opportunity to find methods that can leverage the advantages of both approaches, where one can perform both local and global value propagation at once. The motivation is to integrate these approaches into a practical offline GCRL method.

**Scientific Questions:**
How can we integrate the local update consistency of TD methods with the global value propagation efficiency of MC methods? How can we incorporate multistep returns with a quasimetric distance parameterization to enable long-horizon reasoning, stitching, and horizon generalization without needing explicit hierarchy? How can we stabilize such training for real-world robotic learning problems from noisy, unlabeled offline datasets?

## 3. Overall Core Idea and Design Philosophy

**Overall Core Idea:**
The core idea is Multistep Quasimetric Estimation (MQE), an offline GCRL method that incorporates both multistep value learning and quasimetric architectures without needing explicit hierarchy. MQE fits a quasimetric distance using a multistep Monte-Carlo return.

**Design Philosophy:**
The design philosophy is based on combining the unique benefits of TD and MC methods. Instead of applying onestep invariance only to the immediate future state, the principle is extended to any state (a "waypoint") between the current state and the goal state. This transforms a onestep optimization into an on-policy multistep optimization procedure. The quasimetric architecture enforces structural constraints (like the triangle inequality), which aids in value propagation and stitching. Furthermore, action invariance is explicitly enforced to stabilize value learning and link the Q function and value function, enabling the extraction of stable goal-reaching policies from noisy data.

## 4. Core Innovation Points

1.  **First Integration of Multistep Returns and Quasimetric Architectures:** MQE is the first method to effectively combine multistep TD returns with global value propagation through quasimetric architectures, achieving superior results to previous methods that use TD returns, contrastive learning, or hierarchical methods.
2.  **Multistep Backup via Sampled Waypoints:** The design extends the fitted onestep Q iteration to a multistep optimization by sampling waypoints using a combination of Bernoulli and geometric distributions. This allows the value to propagate globally across any waypoint while maintaining local consistency.
3.  **Action Invariance for Value Learning:** The method imposes action invariance ($V^*_g(s) = \max_a Q^*_g(s, a)$) as a form of value learning using a smooth loss formulation ($L_I$). This prevents trivial solutions and stabilizes training dynamics, removing the need for hyperparameter tuning for gradient magnitude regulation.
4.  **Theoretical Policy Improvement Guarantee:** The paper proves that in a tabular setting under standard assumptions (full support over state-action pairs), the algorithm is able to perform policy improvement on top of the behavior policy $\pi_\beta$.
5.  **Robust Horizon Generalization and Compositional Generalization:** The combination of multistep backup and quasimetric constraints allows the learned policy to display a much stronger level of horizon generalization and "stitching" behavior than prior methods, enabling compositionality in real-world robotic learning problems.

## 5. Overview of the Overall Technical Solution

The overall technical solution defines a distance (quasi)metric over states and state-action pairs proportional to the goal-conditioned Q and V functions. The solution consists of three main parts:
1.  **Multistep Backup under a Quasimetric Architecture:** Parameterizing the Q and V functions using a Metric Residual Network (MRN) such that $Q_g(s, a) = V_g(g)e^{-d((s,a),g)}$ and $V_g(s) = V_g(g)e^{-d(s,g)}$. Instead of a onestep TD target, the method optimizes a multistep backup objective that minimizes the Bregman divergence (LINEX loss) between the current distance and the distance from a sampled waypoint to the goal, discounted by the number of steps to the waypoint.
2.  **Imposing Action Invariance:** To satisfy the property $V^*_g(s) = \max_a Q^*_g(s, a)$, the method enforces $d(\psi(s), \phi(s, a)) = 0$ for each action. This is optimized using a specific loss function $L_I$ that scales with the magnitude of deviation, allowing relaxed enforcement when the violation is low.
3.  **Policy Extraction:** The goal-conditioned policy is extracted using a behavior-regularized deep-deterministic policy gradient (DDPG + BC) that minimizes the quasimetric distance produced by the actor actions while maximizing the log-probability of the behavior actions.

## 6. Detailed Module Design

### 6.1 Quasimetric Distance Parameterization Module
This module parameterizes the goal-conditioned Q and V functions as distances. It uses learned state representations $\psi(s) \in \mathbb{R}^{NM}$ and state-action representations $\phi(s, a) \in \mathbb{R}^{NM}$. The distance function is defined using the Metric Residual Network (MRN) architecture. MRN splits the representation into $N$ equally sized components. In each part, it takes the sum of an asymmetric component (maximum of ReLU) and a symmetric component (l2 norm) of the difference between the two embeddings. The Q and V functions are defined via the exponential of the negative distance.

### 6.2 Multistep Backup Module
This module transforms onestep Q iteration into a multistep procedure. Given a waypoint $s_{w_t}$ sampled from the future states in the trajectory (between the current state and the goal), the objective regresses the current distance to the discounted distance from the waypoint to the goal. The waypoint is sampled based on a specific distribution: with probability $p$, the waypoint is the next state ($k'=1$), and with probability $1-p$, it is sampled from a geometric distribution capped at the index of the future state. The loss function uses a form of Bregman divergence with LINEX losses, which does not incur vanishing gradients when distances are close in value.

### 6.3 Action Invariance Module
This module enforces the consistency between the value function and the Q function. Since the construction of Q and V does not inherently observe the property $V^*_g(s) = \max_a Q^*_g(s, a)$, the distance between the state representation and the state-action representation is forced to zero ($d(\psi(s), \phi(s, a)) \leftarrow 0$). A smoothed squared loss formulation is used to avoid trivial solutions where representations collapse to zero.

### 6.4 Policy Extraction Module
The policy is extracted by minimizing the distance produced by the quasimetric network for the selected action, thereby maximizing the Q value. It also incorporates a behavior cloning (BC) regularization term to stay close to the offline data distribution, tuned by a coefficient $\alpha$.

## 7. All Mathematical Formulas and Symbol Definitions

**Symbol Definitions:**
*   $\mathcal{M}$: Controlled Markov process with state space $\mathcal{S}$, action space $\mathcal{A}$, transition dynamics $P(s'|s, a)$, discount factor $\gamma$.
*   $\pi(a|s, g)$: Goal-reaching policy.
*   $\pi_\beta$: Behavior policy.
*   $Q^\pi_g(s, a)$: Goal-conditioned Q function (maximum discounted likelihood).
*   $V^\pi_g(s)$: Goal-conditioned Value function.
*   $Q^*_g(s, a)$: Optimal Q function.
*   $s_{+t}$: Future state.
*   $s_{w_t}$: Waypoint between current state and goal state.
*   $\mathcal{D}$: Offline dataset.
*   $\Phi$: Class of representations.
*   $\mathcal{Q}_X$: Space of quasimetric distances over set $X$.
*   $\mathcal{D}_X$: General class of distances satisfying non-negativity and identity.
*   $\mathcal{P}$: Path relaxation operator.

**Mathematical Formulas:**

1.  Goal-conditioned Q function:
    $$Q^\pi_g(s, a) \triangleq E_{\{s_t, a_t\} \sim \pi} \left[ \sum_{t=0}^\infty \gamma^t P(s_t = g | s_0 = s, a_0 = a) \right]$$

2.  Goal-conditioned Value function:
    $$V^\pi_g(s) \triangleq E_{\{s_t\} \sim \pi} \left[ \sum_{t=0}^\infty \gamma^t P(s_t = g | s_0 = s) \right]$$

3.  Future state sampling (Geometric distribution):
    $$s_{+t} \triangleq s_{t+K}, K \sim \text{Geom}(1 - \gamma)$$

4.  Metric Residual Network (MRN) architecture:
    $$d_{MRN}(x, y) \triangleq \frac{1}{N} \sum_{k=1}^N \max_{m=1...M} \max(0, x_{kM+m} - y_{kM+m}) + \|x_{kM+m} - y_{kM+m}\|_2$$

5.  Q and V parameterized as distance:
    $$Q_g(s, a) = V_g(g)e^{-d((s,a),g)}, \quad V_g(s) = V_g(g)e^{-d(s,g)}$$

6.  Distance definitions using representations $\phi, \psi$:
    $$d((s, a), g) \triangleq d_{MRN}(\phi(s, a), \psi(g)), \quad d(s, g) \triangleq d_{MRN}(\psi(s), \psi(g))$$

7.  Fitted onestep Q iteration (Objective $\mathcal{T}$):
    $$e^{-d((s,a),g)} \leftarrow E_{\{s' \sim P(s'|s,a)\} \sim \mathcal{D}}[\gamma \cdot e^{-d(s',g)}]$$

8.  Waypoint sampling distribution:
    $$s_{w_t} \leftarrow s_{t+k'} \text{ for } \begin{cases} k' \sim \min(\text{Geom}(1-\lambda), K) & \text{with probability } 1-p, \\ k' = 1 & \text{with probability } p. \end{cases}$$

9.  Multistep backup objective (Objective $\mathcal{T}_\beta$):
    $$e^{-d((s,a),g)} \leftarrow - E_{\{(s_t, a_t), s_{w_t}\} \sim \mathcal{D}}[\gamma^{k'} \cdot e^{-d(s_{w_t},g)}]$$
    (Note: The arrow contains a minus sign in the paper text likely denoting assignment/regression direction, copied exactly: $\leftarrow-$)

10. Bregman divergence with LINEX losses:
    $$D_T(d, d') \triangleq \exp(d - d') - d'$$

11. Loss function for multistep backup $L_{\mathcal{T}_\beta}$:
    $$L_{\mathcal{T}_\beta}(\phi, \psi; \{s_i, a_i, s_{w_i}, g_i, k'_i\}_{i=1}^N) = \sum_{i=1}^N \sum_{j=1}^N D_T(d((s_i, a_i), g_j), d(s_{w_i}, g_j)) - k'_i \log \gamma$$

12. Action invariance objective:
    $$d(\psi(s), \phi(s, a)) \leftarrow 0$$

13. Loss function for action invariance $L_I$:
    $$L_I(\phi, \psi; \{s_i, a_i\}_{i=1}^N) = \sum_{i=1, j=1}^{N, N} \left( e^{-d(\psi(s_i), \phi(s_i, a_j))} - 1 \right)^2$$

14. Policy extraction loss (DDPG + BC):
    $$L_\mu(\pi; \{s_i, a_i, g_i\}_{i=1}^N) = E \left[ \sum_{i=1}^N \sum_{j=1}^N d((s_i, \pi(s_i, g_j)), g_j) - \alpha \log \pi(a_i | s_i, g_i) \right]$$

15. Tabular learning objective (Step 1 of analysis):
    $$\min_{\hat{d} \in \mathcal{D}_{\mathcal{S} \times \mathcal{A} \cup \mathcal{S}}} E_{(s,a) \sim p^{\pi_\beta}, g \sim p^{\pi_\beta}} \left[ D_T\left(d((s, a), g), E_{p^{\pi_\beta}(s_w|s,a)}[\gamma^{k'}e^{-d(s_w,g)}]\right) + (e^{-d(s,(s,a))} - 1)^2 \right]$$

16. Path relaxation operator $\mathcal{P}$:
    $$\mathcal{P}(d)(x, z) \triangleq \min_{y \in X} [d(x, y) + d(y, z)]$$

17. Tabular policy extraction objective (Step 3 of analysis):
    $$\min_\pi E_{(s,a) \sim p^{\pi_\beta}, g \sim p^{\pi_\beta}} [\tilde{d}((s, \pi(s, g)), g)]$$

## 8. Algorithm Pseudocode

**Algorithm 1: Multistep Quasimetric Estimation**
Require: Dataset $\mathcal{D}$, Batch size $B$, training iteration $T$, Probability $p$
1: Initialize quasimetric network $Q$ with parameters $(\phi, \psi)$, goal-reaching policy $\pi_\mu$
2: for $t = 1...T$ do
3:   Sample $\{s_i, a_i, s'_i, s_{w_i}, g_i\}_{i=1}^B \sim \mathcal{D}$ (Eq. 8)
4:   Update $Q$ with multistep backup by minimizing $L_{\mathcal{T}_\beta}(\phi, \psi; \{s_i, a_i, s_{w_i}, k'_i\}_{i=1}^B)$ (Eq. 9)
5:   Update $Q$ with action invariance constraints by minimizing $L_I(\phi, \psi; \{s_i, a_i\}_{i=1}^B)$ (Eq. 14)
6:   Update policy $\pi_\mu$ with DDPG+BC by minimizing $L_\mu(\pi_\mu; \{s_i, a_i, g_i\}_{i=1}^B)$ (Eq. 15)
7: return $\pi_\mu$