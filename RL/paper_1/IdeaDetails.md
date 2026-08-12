**1. Research Background and Existing Pain Points**
Reinforcement learning from verifiable rewards (RLVR) is an emerging paradigm for improving the reasoning ability of large language models. However, standard on-policy training discards rollout experiences after a single update, leading to computational inefficiency and instability. While prior work on RL has highlighted the benefits of reusing past experience, the role of experience characteristics in shaping learning dynamics of large reasoning models remains underexplored. Existing experience replay methods for LRMs largely overlook the impact of data quality within the replay buffer. A critical pain point is the risk of a "snowball effect": replaying high-entropy trajectories—which often contain flawed reasoning chains that accidentally yield correct final answers (lucky hits)—can systematically degrade the learned policy, leading to entrenched reasoning errors (e.g., models generating invalid code blocks for math problems). Furthermore, current methods lack a principled way to identify what makes a reasoning experience valuable and how to manage the replay process based on customized learning schedules.

**2. Core Research Motivation and Scientific Questions**
Core Research Motivation: AI is at the cusp of a new period in which experience will become the dominant medium of improvement (experience-based RL). Efficiently leveraging previously collected interactions helps models stabilize training and accelerate convergence. However, a non-trivial challenge lies in how to exploit past experiences based on their differing "values" and manage the replay process. 
Scientific Questions: How can the reasoning model effectively exploit its own stream of experience to maximize learning toward scaling RL compute for LRMs? Specifically, what makes a reasoning experience valuable? Are all questions equally useful for training, or do certain difficulty levels provide stronger learning signals? Does lower entropy imply better reasoning and serve as an effective proxy for trajectory selection?

**3. Overall Core Idea and Design Philosophy**
The overall core idea is to investigate what constitutes a valuable reasoning experience and leverage these insights to design a principled experience management and replay framework. The design philosophy is based on two key findings: (1) Medium-difficulty questions (intermediate rollout correctness) provide the most valuable optimization signals, and (2) Entropy minimization is an effective heuristic for trajectory selection (low-entropy trajectories correlate with correct reasoning chains, mitigating the risk of learning from flawed logic). Based on this, ExGRPO strategically identifies, manages, and replays valuable experiences. It maintains a replay buffer organized by correctness buckets, prioritizes medium-difficulty questions and low-entropy trajectories, and employs a mixed-policy objective to balance fresh exploration with experience exploitation, using policy shaping to preserve entropy and stabilize training.

**4. Core Innovation Points**
1. **Identification of Valuable Experience Indicators:** The paper is the first to systematically investigate what makes a reasoning experience valuable in RLVR, identifying rollout correctness (for questions) and trajectory entropy (for trajectories) as effective online proxy metrics. It establishes that tasks of intermediate difficulty and their associated low-entropy trajectories are most beneficial for RLVR optimization.
2. **Principled Experience Management Pipeline:** Proposing a three-stage management pipeline: (1) Collection of successful trajectories and their correctness rates; (2) Partition of the buffer into buckets by correctness rate, including a "Retired Set" for fully solved questions to prevent overfitting; (3) Selection using a Gaussian sampling strategy prioritizing medium-difficulty questions and selecting the lowest-entropy trajectory under the current policy.
3. **Experiential Mixed-Policy Optimization:** Introducing a unified objective that mixes on-policy exploration with off-policy experience replay. It forms mixed advantage groups containing both replayed and fresh rollouts, uses importance sampling to correct distribution shift for replayed trajectories, and employs a mixed-policy objective parameterized by an experience ratio.
4. **Entropy Preservation and Stability Mechanisms:** Introducing policy shaping $f(w) = w/(w+\beta)$ to replace hard clipping for replayed trajectories, which amplifies low-probability signals and dampens high-probability ones to encourage learning from novel aspects and prevent entropy collapse. Additionally, a delayed start mechanism ensures experiential samples are of higher quality.

**5. Overview of the Overall Technical Solution**
ExGRPO operates iteratively with two main phases: Experience Management and Experiential Policy Optimization.
In the Experience Management phase, the rollout policy generates trajectories for sampled questions. Successful rollouts are stored in a replay buffer mapped by question. The buffer is partitioned into buckets based on the question's latest rollout correctness rate, and questions with 100% correctness are moved to a Retired Set. For optimization, a mini-batch is constructed by sampling questions using a Gaussian distribution centered at medium difficulty and selecting the trajectory with the lowest entropy under the current policy.
In the Experiential Policy Optimization phase, the mini-batch consists of a proportion $\rho$ of experiential samples and $(1-\rho)$ of fresh on-policy samples. For on-policy data, the standard GRPO objective is used. For experiential data, a mixed advantage group is formed containing the replayed trajectory and fresh rollouts. The replayed trajectory is reweighted by importance sampling. To balance exploration and exploitation, policy shaping is applied to the importance weight. The unified objective optimizes the policy model using this mixed batch.

**6. Detailed Module Design**
*   **Experience Collection Module:** Given a question $q$, the rollout policy $\pi_{\theta_{old}}$ generates $K$ trajectories $G^*_q = \{o^*_i\}_{i=1}^K$. The correctness rate is computed as $Acc(q^*) = k/K$ where $k$ is the number of successful trajectories verified by the reward model. All successful rollouts are stored in the replay buffer $\mathcal{E}$, recorded as a mapping $q^* \mapsto \{o^*\}$, along with the correctness rate.
*   **Experience Partition Module:** The replay buffer $\mathcal{E}$ is partitioned into buckets by each question's latest correctness rate $Acc(q^*)$, with $q^*$ as the indexing key. To prevent overfitting on trivial cases, a Retired Set is introduced: questions solved in all rollouts are removed from $\mathcal{E}$, ensuring optimization focuses on partially solved questions.
*   **Experience Selection Module:** Retrieves a mini-batch of experiential samples in two steps:
    1.  **Question Sampling:** To prioritize experiences, each bucket is sampled with probability $p \propto \mathcal{N}(Acc(q^*); \mu, \sigma = 1)$, where $\mu$ is set to 0.5 in practice. This ensures medium-difficulty questions are sampled more often.
    2.  **Trajectory Selection:** For each sampled question, the trajectory with the lowest entropy under the current policy is selected without relying on an external judge: $o^* \leftarrow \arg\min_{o_i \in \{o^*\}} H(o_i; \pi_\theta)$, prioritizing trajectories which indicate better CoT quality.
*   **Experiential Policy Optimization Module:**
    *   **Mini-batch Construction:** A hyperparameter $\rho \in [0, 1)$ determines the experiential proportion, with $|B_{exp}| = \min(\lfloor\rho B\rfloor, |\mathcal{E}|)$ and $|B_{on}| = B - |B_{exp}|$.
    *   **On-policy Objective:** For $q \sim B_{on}$, $K$ trajectories $\{o_i\}_{i=1}^K$ are sampled from $\pi_{\theta_{old}}$ to form $G_q$, with objective $J_{GRPO}(q, G_q; \theta)$.
    *   **Experiential Off-Policy Objective:** For $q^* \sim B_{exp}$, a mixed advantage estimation group $G_{q^*} = \{o^*\} \cup \{o_i\}_{i=1}^{K-1}$ is formed, where $o^*$ is from $\pi_{\theta_{past}}$ and $\{o_i\}$ are from $\pi_{\theta_{old}}$. The advantage is estimated via $\hat{A}$. The replayed trajectory $o^*$ is reweighted by $w^*_t(\theta) = \pi_\theta(o^*_t|q^*, o^*_{<t}) / \pi_{\theta_{past}}(o^*_t|q^*, o^*_{<t})$.
    *   **Policy Shaping:** The "CLIP" term for $o^*$ is replaced with $f(w^*(\theta)) \cdot \hat{A}(o^*, G_{q^*})$, where $f(x) = x/(x+\beta)$ with $\beta = 0.1$.
    *   **Delayed Start:** ExGRPO is activated only after the Pass@1 metric for a training batch surpasses a predefined threshold.
*   **Extension to Continuous Rewards:** For RLHF with continuous rewards, replace rollout success rates with within-query reward variance $\sigma_q$. Normalize via $\tilde{\sigma}_q = (\sigma_q - \sigma_{min}) / \max(\sigma_{max} - \sigma_{min}, \epsilon)$ and discretize into buckets.

**7. All Mathematical Formulas and Symbol Definitions**
*   **Verifiable Reward Function:** 
    $r(q, o) = \begin{cases} 1 & \text{if } o \text{ contains the correct final answer to } q \\ 0 & \text{otherwise} \end{cases}$
*   **GRPO Advantage Estimation:** 
    $\hat{A}_i = \frac{r(q, o_i) - \mu_{G_q}}{\sigma_{G_q}}$, where $\mu_{G_q} = \frac{1}{K}\sum_{j=1}^K r(q, o_j)$ and $\sigma_{G_q}$ are the mean and standard deviation of rewards in group $G_q$.
*   **GRPO Objective:** 
    $J_{GRPO}(\theta) = \mathbb{E}_{q \sim \mathcal{D}, \{o_i\} \sim \pi_{\theta_{old}}(\cdot|q)} \left[ \frac{1}{K} \sum_{i=1}^K \text{CLIP}(w_i(\theta), \hat{A}_i) \right]$
*   **CLIP Definition:** 
    $\text{CLIP}(w, A) = \frac{1}{|o|}\sum_{t=1}^{|o|} \min(w_t A, \text{clip}(w_t, 1-\epsilon, 1+\epsilon)A)$
*   **Importance Weight:** 
    $w_{i,t}(\theta) = \pi_\theta(o_{i,t}|q, o_{i,<t}) / \pi_{\theta_{old}}(o_{i,t}|q, o_{i,<t})$
*   **Entropy:** 
    $H(o) = - \frac{1}{|o|}\sum_t \pi(o_t | q, o_{<t}) \log \pi(o_t | q, o_{<t})$
*   **Correctness Rate:** 
    $Acc(q^*) = k/K$
*   **Question Sampling Probability:** 
    $p \propto \mathcal{N}(Acc(q^*); \mu, \sigma = 1)$
*   **Trajectory Selection:** 
    $o^* \leftarrow \arg\min_{o_i \in \{o^*\}} H(o_i; \pi_\theta)$
*   **Experiential Importance Weight:** 
    $w^*_t(\theta) = \pi_\theta(o^*_t|q^*, o^*_{<t}) / \pi_{\theta_{past}}(o^*_t|q^*, o^*_{<t})$
*   **Policy Shaping Function:** 
    $f(x) = x/(x+\beta)$ with $\beta = 0.1$
*   **ExGRPO Objective:** 
    $J_{ExGRPO}(\theta) = (1- \rho) \cdot \mathbb{E}_{q \sim B_{on}} \left[ \frac{1}{K} \sum_{i=1}^K \text{CLIP}(w_i(\theta), \hat{A}(o_i, G_q)) \right] + \rho \cdot \mathbb{E}_{q^* \sim B_{exp}} \left[ \frac{1}{K} \left( \text{CLIP}(w^*(\theta), \hat{A}(o^*, G_{q^*})) + \sum_{i=1}^{K-1} \text{CLIP}(w_i(\theta), \hat{A}(o_i, G_{q^*})) \right) \right]$
*   **Theoretical Formulas:**
    *   Assumption A2: $\mathbb{E}_{q \sim \mathcal{D}, o \sim \pi_\theta(\cdot|q)} [X^2] < \infty$
    *   Assumption A3: $\sup_{o,t} \frac{\pi_\theta(o_t|\cdot)}{\pi_{\theta_{past}}(o_t|\cdot)} \le m$, $\sup_o W(o; \theta, \theta_{past}) \le M := m^{|o|}$
    *   Importance Ratios: $w_t(o_i; \theta, \theta') := \frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\pi_{\theta'}(o_{i,t} | q, o_{i,<t})}$, $W(o_i; \theta, \theta') := \prod_{t=1}^{|o|} w_t(o_i; \theta, \theta')$
    *   Experiential Variance Bound (General): $\text{Var}(G_{exp}) \le \frac{2}{K^2}(E[W^2U^2] + (K-1)^2 E[U^2])$
    *   Experiential Variance Bound (Tighter): $\text{Var}(G_{exp}) \le \frac{2}{K^2}(E[W^2U^2] + (K-1)E[U^2])$
    *   Continuous Reward Normalization: $\tilde{\sigma}_q = \frac{\sigma_q - \sigma_{min}}{\max(\sigma_{max} - \sigma_{min}, \epsilon)}$

**8. Algorithm Pseudocode**
**Algorithm 1 ExGRPO: Experiential Group Relative Policy Optimization**
Require: Dataset $\mathcal{D}$, batch size $B$, experience ratio $\rho \in [0, 1]$, rollout trails $K$, reward model $\mathcal{R}(\cdot)$.
1: Initialize: Policy model $\pi_\theta$, experience replay buffer $\mathcal{E} \leftarrow \emptyset$, retired set $\mathcal{S} \leftarrow \emptyset$.
2: for each training step do
3:  ▷ Phase 1: Experience Management.
4:  if $|\mathcal{E}| > 0$ then
5:   Partition $\mathcal{E}$ to $k$ buckets $U_k$ by last rollout correctness $Acc = \frac{1}{K}\sum_{k=1}^K \mathbb{1}[r_k = 1]$.
6:   Obtain sampling probability: $p_k = \mathcal{N} (k/K; 0.5, \sigma = 1)$ for nonempty $U_k$
7:   Calculate experience sampling amount: $n \leftarrow \min(\lfloor\rho B\rfloor, |\mathcal{E}|)$
8:   Sample $n$ experience: $\{e_1, \ldots, e_n\} \sim \text{BukcetSample}(\mathcal{U}, \{p_k\})$ ▷ cf. Algorithm 2.
9:   for each experience $e \in \{e_1, \ldots, e_n\}$ do ▷ cf. $\mathcal{E}: q^* \mapsto \{o^*\}$
10:    for each experience trajectory $o_i \in e$ do
11:     Compute trajectory entropy under $\pi_\theta$: $H(o_i; \pi_\theta) = - \frac{1}{|o_i|}\sum_t \log \pi_\theta(o^t_i|q^*, o^{<t}_i)$
12:    end for
13:    Select the trajectory with the lowest entropy: $o^* \leftarrow \arg\min_{e \in \mathcal{E}} H(o_i; \pi_\theta)$
14:   end for
15:   Experiential policy optimization data $B_{exp} \leftarrow \{(q^*, o^*)\}^n$
16:   On-policy optimization data $B_{on} \leftarrow \{q\}^{B-n} \sim \mathcal{D}$
17:  else
18:   $B_{exp} \leftarrow \emptyset$, $B_{on} \leftarrow \{q\}^B \sim \mathcal{D}$
19:  end if
20:  ▷ Phase 2: Experiential Policy Optimization.
21:  Construct batch data: $B \leftarrow B_{exp} \cup B_{on}$
22:  for each sample $q \in B$ do
23:   Generate rollouts: $\{o_1, \ldots, o_K\} \sim \pi_\theta(\cdot|q)$, $\{o_1, \ldots, o_{K-1}\} \sim \pi_\theta(\cdot|q^*)$
24:   Compute rewards and success: $\{r_j\}_{j=1}^K = \{\mathcal{R}(q, o_j)\}_{j=1}^K$; $s = \sum_{j=1}^K \mathbb{I}[r_j = 1]$
25:   if $s = K$ then ▷ All successful
26:    $\mathcal{S} \leftarrow \mathcal{S} \cup \{q\}$ ▷ Add to retired set
27:   else if $s < K$ then ▷ Partial success
28:    Record experience: $\mathcal{E} [q] \cup \{o_j \mid r(o_j) = 1\}$
29:   end if
30:  end for
31:  Compute advantage and update policy model with Eq. 4.
32:  Remove well-learned samples: $\mathcal{E} \leftarrow \mathcal{E} \setminus \{q : q \in \mathcal{S}\}$
33: end for
34: return Optimized policy model $\hat{\pi}_\theta$

**Algorithm 2 BucketSample(). Bucketed Multinomial & Within-Bucket Uniform Sampling**
Require: Valid buckets $\mathcal{U} = \{(Acc, \{e\})\}_{k=1}^K$; probabilities $p \in [0, 1]^K$; sample size $n \in \mathbb{N}$;
Ensure: probabilities $p \in [0, 1]^K$ with $\sum_{k=1}^K p_k = 1$
1: function MULTINOMIAL(n,p)
2:  # Note: sample $X \in \mathbb{N}^d$ such that $\sum_{i=1}^d X_i = n$
3:  Initialize $X \leftarrow (0, \ldots, 0) \in \mathbb{N}^d$, $m \leftarrow n$
4:  for $i = 1$ to $d-1$ do
5:   Draw $X_i \sim \text{Binomial}(m, \frac{p_i}{1-\sum_{j<i} p_j})$
6:   Update $m \leftarrow m - X_i$
7:  end for
8:  return collection of X
9: end function
10: # Note: sampling within a bucket is uniform without replacement; requires $c_k \le |U_{Acc}|$ for all $k$.
11: $c \sim \text{Multinomial}(n,p)$ ▷ Draw all bucket counts in one shot
12: sampled items $\leftarrow \{\}$
13: for $k \leftarrow 1$ to $K$ do
14:  $S \leftarrow \text{UniformSample}(U_k, c_k)$ ▷ Without replacement.
15:  sampled items$[U_k] \leftarrow S$
16: end for
17: return (sampled items, [])