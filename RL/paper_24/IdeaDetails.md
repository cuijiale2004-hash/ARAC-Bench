**1. Research Background and Existing Pain Points**

Robotics has seen significant progress on challenging real-world tasks by training expressive policies on large datasets via imitation learning. Despite promising results, imitation learning methods often struggle to achieve the high reliability and performance needed for real-world use-cases, even when scaled to large datasets. Fine-tuning these policies with reinforcement learning (RL) can in principle address this problem by enabling high performance through online self-improvement. Yet, existing online reinforcement learning methods are typically designed for simple Gaussian policies and do not effectively leverage expressive pre-trained policies, such as diffusion or flow-matching policies typically used in imitation learning. 

Fine-tuning expressive policies with online RL comes with a unique challenge not present in fine-tuning simpler Gaussian policies. Expressive policies like diffusion or flow-matching policies are parameterized by a long chain of denoising steps, which hinders stable gradient propagation from the action output to the policy parameters whenever we want to optimize their actions against some value functions. The backpropagation can get prohibitively expensive and unstable as the number of denoising steps grows large. In the adjacent purely offline or purely online settings, many approaches have sought to avoid the gradient propagation instability by incorporating losses at intermediate denoising steps to guide the denoising process towards high-value actions, but it is still not obvious how to best perform stable value maximization for efficient online fine-tuning.

**2. Core Research Motivation and Scientific Questions**

The core research motivation is to design an efficient and effective RL fine-tuning method for expressive policy classes. The primary scientific question is: Can we design a sample-efficient online RL algorithm that addresses the unique challenge of stable value maximization for expressive policies parameterized by long denoising chains? Specifically, how can we avoid the unstable explicit optimization over the value function with the expressive policy itself, while still effectively maximizing the Q-value and leveraging the expressive capacity of pre-trained models? Furthermore, how can we construct a policy that allows for stable gradient propagation, enables effective online exploration beyond the behavior distribution, and ensures that changes in the Q-function are immediately reflected in the agent’s behavior and TD targets?

**3. Overall Core Idea and Design Philosophy**

The overall core idea is that value maximization of expressive policy classes can be made much more effective and stable by avoiding direct optimization over value of the expressive policy itself. Instead, the base expressive policy is trained using a stable supervised learning objective, and an on-the-fly policy is constructed to maximize value through two steps: (1) a light-weight, one-step edit policy that refines the action samples from the base expressive policy, and (2) a non-parametric post-processing step that takes multiple action candidates from the base and edit policy and selects the highest-value action among base and edited actions. 

The design philosophy enforces an edit distance constraint on the edit policy such that the edited actions remain close to the original actions from the base policy. This restricts the edit policy to solve a simpler, local optimization problem, allowing it to be much smaller than the base expressive policy and enabling efficient and stable optimization. The local edits refine actions within modes of the base policy’s action distribution, which is complemented by the second on-the-fly, non-parametric post-processing step that considers multiple pairs of base and edited actions potentially from different modes and selects the best actions.

**4. Core Innovation Points**

1. **Avoiding Direct Optimization over Value with Expressive Policies**: Unlike prior methods that attempt to backpropagate value gradients through the long denoising chain, EXPO avoids direct optimization over value with the expressive policy. The base expressive policy is exclusively trained with a stable imitation learning objective, fundamentally circumventing the instability of gradient propagation through long denoising chains.

2. **Light-weight Gaussian Edit Policy for Action Refinement**: A small Gaussian edit policy is introduced to locally optimize the Q-function and maximize action entropy. By enforcing an edit distance constraint, the edit policy transforms actions from the base policy toward a higher Q-value Gaussian action distribution without deviating too far from reasonable behavior. This policy also allows for convenient state-dependent action noises for online exploration.

3. **On-the-Fly Parameterization of the RL Policy**: An on-the-fly (OTF) policy is constructed to implicitly perform value maximization by generating action samples using the base and edit policies and selecting the highest Q-value action. This OTF policy is used for both online sampling and the temporal-difference (TD) backup. This ensures that any changes in the Q-function are more immediately reflected in both the agent’s behavior and the TD Q-value target, unlike standard policy extraction methods that require slow parameter updates.

4. **Entropy Backup Framework for Data-Limited Regimes**: For scenarios where the offline dataset is not sufficiently large or broad, a framework is proposed to incorporate an entropy bonus into the training. By constructing a soft sampling distribution that first samples actions from the base policy and edits them, then chooses actions following a softmax probability distribution based on Q-values, a closed-form equation for the sampling distribution's entropy is obtained to perform the backup, addressing the lack of closed-form entropy in expressive policies like diffusion models.

**5. Overview of the Overall Technical Solution**

EXPO is a sample-efficient online RL algorithm that utilizes an on-the-fly policy to maximize value with two parameterized policies: a larger expressive base policy trained with a stable imitation learning objective and a light-weight Gaussian edit policy that edits the actions sampled from the base policy toward a higher value distribution. 

The base policy generates action candidates. The edit policy takes a state and a base action as input and outputs an action edit to refine the base action. The on-the-fly policy optimizes the actions from the base policy with the learned edit policy and chooses the value-maximizing action from the base and edited actions for both sampling and temporal-difference (TD) backup. The Q-function is trained using standard TD learning where the target is computed using the action selected by the on-the-fly policy. The base policy is updated purely via imitation learning on offline and online data, while the edit policy is updated using an entropy-regularized policy loss to maximize Q-value. For data-limited regimes, an entropy-regularized target is utilized by defining a soft sampling distribution.

**6. Detailed Module Design**

**Module 1: Base Expressive Policy**
The base expressive policy (e.g., a diffusion policy) is responsible for capturing complex behavior distributions. It is initialized from offline pre-training and then online fine-tuned with an imitation learning objective. It is never trained to explicitly maximize value. By keeping the base policy optimization stable through supervised imitation learning, it avoids the unstable gradient propagation issue inherent in expressive policy classes.

**Module 2: Gaussian Edit Policy**
The edit policy refines actions generated by the base expressive policy. Given a state $s$ and a base action $a \sim \pi_{base}(\cdot|s)$, the edit policy $\pi_{edit}(\hat{a}|s, a)$ outputs an edit $\hat{a}$. The edited action is computed as $\tilde{a} \leftarrow a + \hat{a}$. The edit policy is trained with a standard entropy-regularized policy loss to locally optimize the Q-function and maximize action entropy. To prevent the edited actions from shifting too far from the behavior distribution, an edit distance constraint is enforced by scaling $\hat{a}$ to be between $[-\beta, \beta]$, where $\beta$ is a hyperparameter. This allows the policy to continuously improve upon the actions generated by the base policy while not deviating too far from reasonable behavior.

**Module 3: On-the-Fly (OTF) Policy Extraction**
Given the base and edit policies, the on-the-fly policy $\pi_{OTF}$ effectively extracts value-maximizing actions. It performs implicit value-maximization in two steps: (1) generating action samples using the base and the edit policy, and (2) selecting the highest Q-value action. The OTF policy $\pi_{OTF}(a|s, \pi_{base}, \pi_{edit}, \phi)$ is defined as $\text{argmax}_{a=\bigcup_{i=1}^N \{a_i,\tilde{a}_i\}} Q_\phi(s, a)$, where $a_i$ is an action sampled from $\pi_{base}$ and $\tilde{a}_i = a_i + \hat{a}_i$ is the action after edit for each of $N$ action samples. This action selection is used for both environment interaction sampling and computing the target in the TD backup.

**Module 4: Entropy Backup with Soft Sampling Distribution**
In scenarios where the offline dataset is not sufficiently large, the agent must explore more broadly. Because expressive policies such as diffusion do not have a closed form expression for entropy, a soft sampling distribution $\pi_{sampling}$ is constructed. It first samples $N$ actions $a_i$ from $\pi_{base}$ and edits them into $\tilde{a}_{i+N} = a_i + \hat{a}_i$ for each action, and then chooses actions following the probability distribution defined by a softmax over Q-values. This provides a closed-form equation for this sampling distribution, and its entropy is used to perform the backup.

**7. All Mathematical Formulas and Symbol Definitions**

**Markov Decision Process (MDP) Definition:**
MDP is defined as $\{S, A, r, \gamma, T, \rho\}$ where:
- $S$: state space
- $A$: action space
- $r : S \times A \rightarrow \mathbb{R}$: reward function
- $T(s'|a, s)$: transition dynamics
- $\gamma \in [0, 1]$: discount factor
- $\rho(s)$: initial state distribution
Goal of RL: $\mathbb{E}_\pi[\sum_{t=0}^T \gamma^t r(s_t, a_t)]$

**Action Edit Formulation:**
$a \sim \pi_{base}(\cdot|s)$
$\tilde{a} \leftarrow a + \hat{a}$
where $\hat{a}$ is the edit sampled from $\pi_{edit}(\hat{a}|s, a)$, constrained such that $\hat{a} \in [-\beta, \beta]$.

**Edit Policy Loss Function:**
$L(\pi_{edit}) = -\mathbb{E}_{(s,a)\sim D, \hat{a}\sim \pi_{edit}(\cdot|s,a)}[Q_\phi(s, a+ \hat{a})- \alpha \log \pi_{edit}(\hat{a}|s, a)]$
where $Q_\phi(s, a)$ is the critic value to maximize, and $\alpha$ is the temperature parameter for entropy regularization.

**On-the-Fly Policy Parameterization:**
$\pi_{OTF}(a|s, \pi_{base}, \pi_{edit}, \phi)$ is defined as $\text{argmax}_{a=\bigcup_{i=1}^N \{a_i,\tilde{a}_i\}} Q_\phi(s, a)$
where $a_i \sim \pi_{base}$ and $\tilde{a}_i = a_i + \hat{a}_i$.

**Q-Function Objective (Critic Loss):**
$\min_\phi \mathbb{E}_{(s_t,a_t,s_{t+1})\sim D}[(r_t + \gamma Q_{\phi'}(s_{t+1}, \tilde{a}^*_{t+1})-Q_\phi(s_t, a_t))^2]$, where $\tilde{a}^*_{t+1} \sim \pi_{OTF}(\cdot|s_{t+1})$

**Entropy Backup Target for Data-Limited Regimes:**
$y = r_t + \gamma[Q_{\phi'}(s_{t+1}, \tilde{a}^*_{t+1})- \alpha \log \pi_{OTF}(\tilde{a}^*_{t+1}|s_{t+1})]$
$L(\phi) = \mathbb{E}_{(s_t,a_t,s_{t+1})\sim D}[(y -Q_\phi(s_t, a_t))^2], \tilde{a}^*_{t+1} \sim \pi_{OTF}(\cdot|s_{t+1})$
$L(\pi_{OTF}) = -\mathbb{E}_{(s,a)\sim D,\hat{a}\sim \pi_{OTF}(\cdot|s,a)}[Q_\phi(s, a+ \hat{a})- \alpha \log \pi_{OTF}(\hat{a}|s, a)]$

**Soft Sampling Distribution:**
$\pi_{sampling}(a_i|s) = \frac{\exp \beta Q(s,a_i)}{\sum_k \exp \beta Q(s,a_k)}$

**Base Policy (Diffusion Policy) Imitation Learning Objective:**
$\min_\psi \mathbb{E}_{t\sim U(\{1,\cdots ,T\}),\epsilon\sim \mathcal{N}(0,I),(s,a)\sim D}[\|\epsilon- \epsilon_\psi(\sqrt{\bar{\alpha}_t}a+\sqrt{1-\bar{\alpha}_t}\epsilon, s, t)\|]$
where $\psi$ represents the parameters of the noise prediction network $\epsilon_\psi$, $t$ is the diffusion timestep uniformly sampled from $1$ to $T$, $\epsilon$ is the noise sample, and $\bar{\alpha}_t$ is the variance schedule parameter.

**8. Algorithm Pseudocode**

**Algorithm 1 Expressive Policy Optimization (EXPO)**
Require: Prior dataset $D_{data} = \{(s_i, a_i)\}$; optionally, expressive policy initialization $\pi_{base}$.
Randomly initialize action edit policy $\pi_{edit}$, critic $Q_\phi$, target critic $Q_{\phi'}$, UTD ratio $G$.
while training do
    for each environment step $t$ do
        Collect rollouts:
        Sample $\tilde{a}^*_t$ from $\pi_{OTF}(\cdot|s, \pi_{base}, \pi_{edit}, \phi')$
        Take action $\tilde{a}^*_t$ and observe $r_t$ and $s_{t+1}$ from the environment
        Store $(s_t, a_t, r_t, s_{t+1})$ in RL replay buffer
        Update policy and critic:
        for $g = 1, \ldots, G$ do
            Sample mini-batch $(s, a, r, s')$ from the replay buffer
            Sample $\tilde{a}^*_{\prime}$ from $\pi_{OTF}(\cdot|s', \pi_{base}, \pi_{edit}, \phi')$
            Compute target as $y = r + \gamma Q_{\phi'}(s', \tilde{a}^*_{\prime})$
            Update $\phi$ minimizing loss: $L = (y -Q_\phi(s, a))^2$
            Update target networks: $\theta' \leftarrow \rho\theta' + (1-\rho)\theta$
            Update $\pi_{base}$ using the last mini-batch with supervised learning objective $L_{IL}(\pi_{base})$
            Update $\pi_{edit}$ using the last mini-batch maximizing objective $Q_\phi(s, a+ \hat{a})-\alpha \log \pi_{edit}(\hat{a}|s), \hat{a} \sim \pi_{edit}(\cdot|s)$