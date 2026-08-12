**1. Research Background and Existing Pain Points**

**Research Background:**
Reinforcement learning (RL) has proven effective across a range of sequential decision-making tasks, from mastering games like Go to controlling complex systems such as robots and plasma reactors. However, in many real-world applications, designing an appropriate reward function is a daunting challenge, as these tasks often involve objectives that are difficult to formalize with numerical rewards. Preference-based RL (PbRL) has emerged as a promising paradigm, leveraging human feedback in the form of pairwise preferences, which are inherently more interpretable yet still information-rich. This paradigm allows agents to learn from relative judgments rather than numerical reward signals, significantly reducing the complexity of reward design. Recent advancements in PbRL have illustrated its efficacy in enabling agents to learn novel behaviors and in achieving better alignment with human preferences, which are often difficult to encapsulate in a reward function.

**Existing Pain Points:**
Despite the advantages of PbRL, obtaining human feedback for preferences can be expensive and time-consuming, which forms a strong barrier for PbRL and limits its scalability in real-world applications. Specifically, the problem of low query efficiency in offline PbRL is caused by two primary reasons:
1. **Inefficient exploration:** Existing methods for generating queries, such as disagreement-based approaches and information-gain-based approaches, may lead the algorithm to focus on refining reward estimates in regions of the state space that are irrelevant to the optimal policy. This results in queries that do not maximally contribute to learning the optimal policy.
2. **Overoptimization of learned reward functions:** A learned reward function is prone to overoptimization, particularly in regions with high uncertainty. This leads to overestimation of the value function and, subsequently, a suboptimal policy. Learning from preference feedback is more vulnerable to this issue, as the feedback is binary and sparse.

**2. Core Research Motivation and Scientific Questions**

**Core Research Motivation:**
To address the strong barrier of expensive and time-consuming human feedback in PbRL, this work aims to systematically enhance the query efficiency of offline PbRL. The motivation is to design an algorithm that generates informative queries ensuring each query maximally contributes to learning the optimal policy, while simultaneously mitigating the overoptimization of the learned reward functions in high-uncertainty regions.

**Scientific Questions:**
1. How to select queries that maximize the information gain about the optimal policy rather than just the reward function, ensuring that the algorithm avoids unnecessary explorations in regions irrelevant to the optimal policy?
2. How to prevent overoptimization of the learned reward function, particularly in regions with high uncertainty, without compromising the learning of the optimal policy?

**3. Overall Core Idea and Design Philosophy**

**Overall Core Idea:**
The paper proposes Offline PbRL via In-Dataset Exploration (OPRIDE), a novel algorithm designed to enhance the query efficiency of offline PbRL. The key idea is to conduct optimistic exploration with in-dataset queries and then utilize the learned reward function pessimistically with discount factor scheduling. OPRIDE consists of two key features: a principled exploration strategy that maximizes the informativeness of the queries and a discount scheduling mechanism aimed at mitigating overoptimization of the learned reward functions.

**Design Philosophy:**
The design philosophy follows a two-phase structure:
1. **Optimistic Exploration for Query Selection:** In the first offline phase, queries are selected based on an exploration mechanism that maximizes the difference of value differences. This strategy aims to minimize the diameter of the uncertainty set for the value function, which is proportional to the information gain from a query, ensuring proper exploitation by avoiding unnecessary explorations.
2. **Pessimistic Policy Extraction:** In the second stage, after learning a reward function based on the preference dataset and annotating the reward-free dataset, a discount factor scheduling mechanism is applied. Based on the pessimistic property of a smaller discount factor, the algorithm dynamically adjusts the discount based on the variance in reward estimation to address the overestimation issue of the value function and achieve better policy performance and higher query efficiency.

**4. Core Innovation Points**

1. **Introduction of the OPRIDE algorithm:** A novel offline PbRL algorithm that achieves superior query efficiency through in-dataset exploration.
2. **Principled exploration strategy for query selection:** A query selection method that identifies the most informative queries by analyzing value differences between trajectories, ensuring maximal contribution to learning the optimal policy. It maximizes the information gain about the optimal policy rather than the reward function.
3. **Variance-based discount scheduling mechanism:** A dynamic adjustment mechanism for the discount factor based on the variance in the reward estimation to mitigate overoptimization of the learned reward functions.
4. **Theoretical guarantees of efficiency:** Theoretical analyses establishing the provable efficiency of the algorithm involving a principled exploration strategy under mild assumptions, decomposing the suboptimality into offline error and preference error terms.

**5. Overview of the Overall Technical Solution**

The overall technical solution is a two-phase procedure as depicted in Figure 1:
**Phase 1: Query Selection via In-Dataset Exploration**
- Start with an unlabeled offline dataset of trajectories and an empty preference dataset.
- Train $M$ ensembles of reward networks and corresponding value functions using bootstrapping and offline algorithms (like IQL).
- Select two trajectories that maximize the exploration objective (difference of value differences) according to Equation 7.
- Receive the preference label between the two selected trajectories and add it to the preference dataset.
- Iteratively update the reward functions and value/Q functions after each preference query.

**Phase 2: Policy Extraction**
- Train the reward function using the cross-entropy loss in Equation 6 and annotate the reward-free dataset using the averaged ensemble reward.
- Adjust the discount factor $\gamma$ to $\hat{\gamma}$ based on the variance-based discount scheduling mechanism in Equation 8 to reduce the impact of noise and overestimation in reward learning.
- Extract the policy via standard offline reinforcement learning algorithms (like IQL) from the labeled datasets.

**6. Detailed Module Design**

**Module 1: Offline Query Selection with In-Dataset Exploration**
- **Mechanism:** To maximize the information gain about the optimal policy, the algorithm uses the difference of value differences as the exploration criteria.
- **Steps:**
  1. Train a set of reward functions $\{r_{\theta_i}\}_{i=1}^M$ using bootstrapping.
  2. Train a set of value functions $\{V_{\psi_i}\}_{i=1}^M$ using offline algorithms like IQL with the reward functions.
  3. Select two trajectories $\tau_1$ and $\tau_2$ that maximize the difference of value differences between the two trajectories (Equation 7).
- **Intuition:** Equation 7 aims to select queries that most effectively minimize the diameter of the uncertainty set for the value function. The diameter represents the maximum possible disagreement between any two candidate value functions on any two policies. Reducing this diameter is proportional to the information gain from a query. Maximizing the information gain $I(R;D)$ directly corresponds to reducing the diameter. Mathematically, the sample complexity of reward function estimation is proportional to the Eluder dimension of the reward function class, $d_{Elu}(R)$, while Equation 7’s complexity relates to the Eluder dimension of the optimal value function class, where it is often the case that $d_{Elu}(V^*) \ll d_{Elu}(R)$.

**Module 2: Policy Extraction with Variance-Based Discount Scheduling**
- **Mechanism:** After obtaining the preference feedback, the reward function is trained, and the reward-free dataset is annotated. To solve the overoptimization issue, the discount factor is adjusted based on the variance of the value function estimates.
- **Steps:**
  1. Train the return model $\hat{R}$ minimizing cross-entropy loss (Equation 6).
  2. Annotate the dataset using $\hat{r} = \frac{1}{M}\sum_{i=1}^M r_{\theta_i}$.
  3. Adjust the discount factor $\gamma$ to $\hat{\gamma}$ using Equation 8. If the variance of the value estimation for a data point is greater than the top $m\%$ in the batch, use a smaller discount factor $\gamma_{small}$; otherwise, use $\gamma$.
  4. Learn a corresponding Q-value function and extract the policy from the labeled datasets by adopting standard offline RL algorithms, like IQL.
- **Intuition:** Using a smaller discount factor is known to provide pessimistic and robust guarantees and performs well in various settings like imitation learning. Reducing the discount factor where there is a higher variance in value estimation alleviates the impact of reward function overestimation.

**Module 3: Theoretical Strategy**
- **Construct Confidence Set:** Use cross-entropy loss to get the maximum likelihood estimator for the return function $\hat{R}_k$ (Equation 9) and construct the confidence set $C_k(R)$ (Equation 10).
- **Construct Candidate Policy Set:** Construct the candidate policy set $\Pi_k$ (Equation 11) using pessimistic value estimation as the criteria, consisting of policies that are possibly optimal within the current level of uncertainty over reward and dynamics.
- **Select Exploratory Policies:** Select explorative policies via Equation 12, choosing two policies such that there is a $R_1 \in C_k(R)$ that strongly prefers $\pi_1$ over $\pi_2$, and a $R_2 \in C_k(R)$ that strongly prefers $\pi_2$ over $\pi_1$.

**7. All Mathematical Formulas and Symbol Definitions**

**MDP and Value Functions:**
- $V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{t=1}^\infty r(s_t, a_t) \bigg| s_0 = s \right]$
- $Q^\pi(s, a) = \mathbb{E}_\pi \left[ \sum_{t=1}^\infty r(s_t, a_t) \bigg| s_0 = s, a_0 = a \right]$
- Bellman evaluation operator: $(\mathcal{T}^\pi f)(s, a) = \mathbb{E}_{s'\sim P(\cdot|s,a), a'\sim \pi(\cdot|s')} \left[ r(s, a) + \gamma f(s', a') \right]$
- Bellman optimality equation: $V^*(s) = \max_{a \in \mathcal{A}} Q^*(s, a)$, $Q^*(s, a) = \mathbb{E}_{s'\sim P(\cdot|s,a)} \left[ r(s, a) + \gamma V^*(s') \right]$
- Sub-optimality metric: $\text{SubOpt}(\pi) = V^{\pi^*}(s_0) - V^\pi(s_0)$

**Bellman Consistent Pessimism:**
- Definition 1 (Bellman shift coefficient): $C(\nu; \mu, Q, \pi) := \max_{q \in \mathcal{Q}} \frac{\|q - \mathcal{T}^\pi q\|_{2, \nu}^2}{\|q - \mathcal{T}^\pi q\|_{2, \mu}^2}$

**Preference-based RL:**
- Bradley-Terry pairwise preference model: $P\left(\tau_i \succ \tau_j \bigg| R\right) = \frac{1}{\exp (R(\tau_j) - R(\tau_i)) + 1}$
- Cross-entropy loss: $L_{CE}(R) = - \mathbb{E}_{(\tau_1, \tau_2, o) \sim \mathcal{D}_{pref}} \left[ o \log P\left(\tau_1 \succ \tau_2 \bigg| R\right) + (1 - o) \log(1 - P\left(\tau_1 \succ \tau_2 \bigg| R\right)) \right]$

**Eluder Dimension and Generalized Linear Preference Model:**
- Definition 2 (Eluder Dimension): Suppose $\mathcal{F}$ is a function class defined in $\mathcal{X}$, the $\alpha$-Eluder dimension $d_{Elu}(\mathcal{F}, \alpha)$ is the longest sequence $\{x_1, x_2, \cdots, x_n\} \in \mathcal{X}$ such that there exists $\alpha' \ge \alpha$ where $x_i$ is $\alpha'$-independent of $\{x_1, \cdots, x_{i-1}\}$ for all $i \in [n]$.
- Definition 3 (Generalized Linear Preference Model): In d-dimensional generalized linear models, the preference function can be represented as $P(\tau_1 \succ \tau_2|\theta) = \sigma(\langle \phi(\tau_1, \tau_2), R \rangle)$ where $\sigma$ is an increasing Lipschitz continuous function, $\phi: \text{Traj} \times \text{Traj} \to \mathbb{R}^d$ is a known feature map satisfying $\|\phi(\tau_1, \tau_2)\|_2 \le H$ and $\theta \in \mathbb{R}^d$ is the unknown parameter.

**Query Selection via In-Dataset Exploration:**
- Exploration objective: $\arg\max_{(\tau_1, \tau_2) \in D} \arg\max_{i, j \in [M]} |(V_{\psi_i}(\tau_1) - V_{\psi_j}(\tau_1)) - (V_{\psi_i}(\tau_2) - V_{\psi_j}(\tau_2))|$

**Variance-Based Discount Scheduling:**
- Discount factor adjustment: $\hat{\gamma}(s, a) = \begin{cases} \gamma_{small}, & \text{if } \text{Var}\{Q_{\phi_i}(s, a)\}_{i=1}^M > \text{Top } m\%(\{\text{Var}_j\{Q_{\phi_i}(s_j, a_j)\}_{i=1}^M\}_{j=1}^{|\text{Batch}|}) \\ \gamma, & \text{else} \end{cases}$

**Theoretical Formulations:**
- MLE for return function: $\hat{R}_k = \arg\min_{R \in \mathcal{R}} L_k(R)$
- Confidence set: $C_k(R) = \left\{ R \in \mathcal{R} \bigg| \sum_{i=1}^k \left( (R(\tau_1^i) - R(\tau_2^i)) - (\hat{R}_k(\tau_1^i) - \hat{R}_k(\tau_2^i)) \right)^2 \le \beta_k \right\}$
- Candidate policy set: $\Pi_k = \left\{ \hat{\pi} \bigg| \exists R \in C_k(R), \hat{\pi} = \arg\max_{\pi \in \Pi} \hat{q}_R(s_1, \pi). \right\}$
- Explorative policy selection: $\pi_k^1, \pi_k^2 = \arg\max_{\pi_1, \pi_2 \in \Pi_k} \max_{R_1, R_2 \in C_k(R)} \left( (\hat{v}^{\pi_1}_{R_1} - \hat{v}^{\pi_1}_{R_2}) - (\hat{v}^{\pi_2}_{R_1} - \hat{v}^{\pi_2}_{R_2}) \right)$
- Theorem 4 Suboptimality Bound: $\text{SubOpt}(\bar{\pi}) \le O \left( \underbrace{\sqrt{\frac{C^\dagger \log(N|\mathcal{Q}||\Pi|)}{N(1-\gamma)^2}}}_{\text{Offline Error}} + \underbrace{\sqrt{\frac{\kappa d_{Elu}(\Delta R, 1/K) \log (K|\Delta R|)}{K(1-\gamma)}}}_{\text{Preference Error}} \right)$

**BCP Loss and Confidence Set:**
- Loss function: $L(q, q', \pi; D) = \sum_{k=1}^K \sum_{t=0}^T \left( q_t(s_t^k, a_t^k) - (r(s_t^k, a_t^k) + \gamma q'_{t+1}(s_{t+1}^k, \pi_{t+1})) \right)^2$
- Confidence set: $\mathcal{V}(\pi, \epsilon) = \left\{ q \in \mathcal{V} : L(q, q, \pi; D) - \min_{q' \in \mathcal{V}} L(q', q, \pi; D) \le \epsilon \right\}$
- Pessimistic policy: $\hat{\pi} = \arg\max_{\pi \in \Pi} \min_{q \in \mathcal{V}(\pi, \epsilon)} q_1(s_1, \pi)$
- Pessimistic value function: $\hat{q} = \arg\min_{q \in \mathcal{V}(\hat{\pi}, \epsilon)} q_1(s_1, \hat{\pi})$

**8. Algorithm Pseudocode**

**Algorithm 1: Offline Preference-Based Reinforcement Learning with In-Dataset Exploration**
1: Input: Unlabeled offline dataset $\mathcal{D} = \{\tau_n = \{(s_t^n, a_t^n)\}_{t=0}^T\}_{n=1}^N$, query budget $K$, ensemble number $M$
2: Initialized the preference dataset $\mathcal{D}_{pref} \leftarrow \emptyset$.
3: for episode $k = 1, \cdots, K$ do
4: Train $M$ ensembles of reward network $r_{\theta_i}$ with $\mathcal{D}_{pref}$ using $L_{CE}$ in Equation 6.
5: Train $M$ corresponding value functions $V_{\psi_i}, Q_{\phi_i}$ with each reward function $r_{\theta_i}$.
6: Select trajectories $\tau_{k,1}, \tau_{k,2}$ that maximize the exploration objective according to Equation 7.
7: Receive the preference $o_k$ between $\tau_{k,1}$ and $\tau_{k,2}$ and add it to the preference dataset, i.e., $\mathcal{D}_{pref} \leftarrow \mathcal{D}_{pref} \cup \{(\tau_{k,1}, \tau_{k,2}, o_k)\}$.
8: end for
9: Annotate the unlabeled offline dataset $\mathcal{D}$ with the reward function $\hat{\theta}$ and obtain $\hat{\mathcal{D}}$.
10: Adjust the discount facto $\gamma$ to $\hat{\gamma}$ based on Equation 8.
11: Extract policy $\pi_\xi$ via offline RL from $\hat{\mathcal{D}}$.
12: Output: The learned policy $\pi_\xi$

**Algorithm 2: Theoretical Analysis Version of OPRIDE**
1: Input: Unlabeled offline dataset $D$, query budget $K$
2: Initialized preference dataset $\mathcal{D}_{pref} \leftarrow \emptyset$.
3: for $k = 1, \cdots, K$ do
4: Calculate confidence set $C_k(R)$ for reward function based on $\mathcal{D}_{pref}$ with Equation 10.
5: Calculate pessimistic value function $\hat{q}(\cdot)$ using Algorithm 3 for each reward function in $C_k(R)$.
6: Construct the near-optimal policy set $\Pi_k$ using Equation 11.
7: Select explorative policies $\pi_k^1, \pi_k^2$ within $\Pi_k$ based on Equation 12.
8: Sample trajectories $\tau_k^1, \tau_k^2$ with selected policy $\pi_k^1, \pi_k^2$.
9: Receive the preference $o_k$ between $\tau_k^1$ and $\tau_k^2$ and add it to the preference dataset $\mathcal{D}_{pref} \leftarrow \mathcal{D}_{pref} \cup \{(\tau_k^1, \tau_k^2, o_k)\}$.
10: end for
11: Output: Average policy $\bar{\pi} = \frac{1}{2K} \cdot \sum_{k=1}^K (\pi_k^1 + \pi_k^2)$.

**Algorithm 3: Bellman-consistent Pessimism (BCP)**
1: Input: Offline Dataset $D_{off} = \{\tau_k = \{(s_t^k, a_t^k)\}_{t=0}^T\}_{k=1}^K$, reward function $r$.
2: Set the loss function as $L(q, q', \pi; D) = \sum_{k=1}^K \sum_{t=0}^T (q_t(s_t^k, a_t^k) - (r(s_t^k, a_t^k) + \gamma q'_{t+1}(s_{t+1}^k, \pi_{t+1})))^2$.
3: Set the confidence set of value functions as $\mathcal{V}(\pi, \epsilon) = \{q \in \mathcal{V} : L(q, q, \pi; D) - \min_{q' \in \mathcal{V}} L(q', q, \pi; D) \le \epsilon\}$.
4: Compute pessimistic policy and value function as $\hat{\pi} = \arg\max_{\pi \in \Pi} \min_{q \in \mathcal{V}(\pi, \epsilon)} q_1(s_1, \pi)$. and $\hat{q} = \arg\min_{q \in \mathcal{V}(\hat{\pi}, \epsilon)} q_1(s_1, \hat{\pi})$.
5: Output: $\hat{\pi}$ and $\hat{q}$.

**Algorithm 4: Bellman-consistent Pessimism Evaluation**
1: Input: Offline Dataset $D_{off} = \{\tau_k = \{(s_t^k, a_t^k)\}_{t=0}^T\}_{k=1}^K$, reward function $r$, policy $\pi$
2: Set the loss function as $L(q, q', \pi; D) = \sum_{k=1}^K \sum_{t=0}^T (q_t(s_t^k, a_t^k) - (r(s_t^k, a_t^k) + \gamma q'(s_{t+1}^k, \pi_{t+1})))^2$.
3: Set the confidence set of value functions as $\mathcal{V}(\pi, \epsilon) = \{q \in \mathcal{V} : L(q, q, \pi; D) - \min_{q' \in \mathcal{V}} L(q', q, \pi; D) \le \epsilon\}$.
4: Compute pessimistic value function as $\hat{q} = \arg\min_{q \in \mathcal{V}(\pi, \epsilon)} q(s_0, \pi)$.
5: Output: $\hat{v}$ and $\hat{q}$.