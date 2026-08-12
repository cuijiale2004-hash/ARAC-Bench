1. **Research Background and Existing Pain Points**
Reinforcement learning (RL) plays a crucial role in the post-training of large language models (LLMs). A series of powerful reasoning models have employed large-scale reinforcement learning to achieve strong performance on highly challenging benchmarks. The evolution of RL algorithms for LLM reasoning has progressed through several key stages: REINFORCE provides a solid baseline that is easy to implement and efficient in simple settings; PPO improves upon REINFORCE with better stability and efficiency in complex settings; GRPO simplifies PPO training by eliminating the learning of a separate value function and relying on group comparisons. However, all these methods share a fundamental limitation in their reward-maximizing objective. Reward-maximizing RL methods tend to overfit to the dominant mode of the reward distribution. As illustrated in the paper, representative RL methods such as GRPO neglect other meaningful modes, which often results in limited diversity among generated reasoning paths and reduces generalization to less frequent yet valid logical outcomes. These drawbacks become especially pronounced in complex long-chain-of-thought (CoT) reasoning, where capturing a diverse distribution of plausible solutions is essential for effective generalization. Recent approaches adjust the clip ratio, apply entropy-based advantage shaping, or selectively promote high-entropy tokens, thereby dynamically adapting the data distribution and implicitly increasing diversity. This raises a fundamental question: How can we promote diverse exploration to prevent convergence to dominant solution patterns in RL training?

2. **Core Research Motivation and Scientific Questions**
The core research motivation is to promote diverse exploration to prevent convergence to dominant solution patterns in RL training, thereby addressing the inherent mode-collapse limitations of previous RL approaches. FlowRL achieves more efficient exploration by fundamentally shifting from reward maximization to reward distribution matching. The primary scientific question posed is: How can we promote diverse exploration to prevent convergence to dominant solution patterns in RL training? Furthermore, how can we effectively integrate distribution matching (specifically GFlowNets trajectory balance) with LLM reinforcement learning for long CoT reasoning, addressing practical challenges such as gradient explosion and sampling mismatch?

3. **Overall Core Idea and Design Philosophy**
The overall core idea of FlowRL is to align the policy model with the full reward distribution, encouraging mode coverage. FlowRL introduces a learnable partition function that normalizes scalar rewards into a target distribution, and minimizes the reverse KL divergence between the policy and this reward-induced distribution. The design philosophy is rooted in transforming scalar rewards into a normalized target distribution using a learnable partition function and minimizing the reverse KL divergence between the policy and target distribution. This approach is developed based on the trajectory balance formulation from GFlowNets, providing a gradient equivalence proof that bridges generative modeling and policy optimization. To address the challenges of long CoT training, the design incorporates two key technical solutions: length normalization to tackle gradient explosion issues that occur with variable-length CoT reasoning, and importance sampling to correct for the distribution mismatch between generated rollouts and the current policy.

4. **Core Innovation Points**
1. Propose FlowRL, a policy optimization algorithm that shifts from reward maximization to reward distribution matching via flow balancing, encouraging diverse reasoning path exploration while addressing the inherent mode-collapse limitations of existing RL methods.
2. Introduce a learnable partition function $Z_\phi(x)$ to normalize scalar rewards into a valid target distribution, and minimize the reverse KL divergence between the policy and the target distribution. A gradient equivalence proof is provided establishing that minimizing this KL objective is equivalent to minimizing the trajectory balance loss used in GFlowNet.
3. Introduce length normalization to tackle gradient explosion issues that occur with variable-length CoT reasoning. The log-probability term is rescaled as $\frac{1}{|y|} \log \pi_\theta(y | x)$, balancing the contributions of long and short sequences and stabilizing the learning signal.
4. Introduce importance sampling to correct for the distribution mismatch between generated rollouts and the current policy, employing PPO-style clipping and gradient detachment to stabilize policy updates with off-policy data.
5. Redefine the reward function by incorporating a reference model as a prior constraint on the reward distribution, modifying the original $\exp(\beta r(x,y))$ to include the reference model: $\exp (\beta r(x,y)) \cdot \pi_{ref}(y | x)$.

5. **Overview of the Overall Technical Solution**
The overall technical solution formulates distribution matching in reinforcement learning through reverse KL divergence and establishes its connection to trajectory balance from GFlowNets. To address the challenges of gradient explosion and sampling mismatch encountered during long CoT training, it incorporates length normalization and importance sampling. First, scalar rewards are normalized into a target distribution using a learnable partition function $Z_\phi(x)$, and the reverse KL divergence between the policy and this target distribution is minimized. This objective is proven equivalent to the trajectory balance loss. Second, the reward function is redefined to incorporate the reference model. Third, length normalization is applied to the log-probability term to prevent gradient explosion in long trajectories. Fourth, importance sampling with clipping and gradient detachment is applied to mitigate sampling mismatch, allowing data-efficient training with stale trajectories. Finally, these components are integrated into the FlowRL objective, which is used to update the policy parameters $\theta$ during training.

6. **Detailed Module Design**
**Learnable Partition Function Module ($Z_\phi$)**:
The learnable partition function $Z_\phi(x)$ is used to normalize scalar rewards into a valid target distribution. Following Lee et al. (2024), a randomly initialized 3-layer MLP with hidden dimensions matching those of the base model is used to parameterize $Z_\phi$. The input to $Z_\phi$ is the mean of the language model’s hidden states after encoding the input $x$, and the output is a scalar value. From the flow perspective, $Z_\phi$ measures the probability flow from the initial state $S_0$. It estimates the denominator—the sum of rewards across all possible paths—enabling conversion to a probability distribution via $\frac{r(x,y)}{Z_\phi(x)}$.

**Reference Model Constraint Module**:
To incorporate a reference model as a prior constraint on the reward distribution, the original reward formulation $\exp(\beta r(x,y))$ is modified to include the reference model:
$\exp (\beta r(x,y)) \cdot \pi_{ref}(y | x)$
where $r(x,y)$ denotes the outcome reward commonly used in reinforcement learning and $\pi_{ref}$ is the initial pre-trained model. Group normalization is applied to $r(x,y)$ as $\hat{r}_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$, where $r = \{r_1, r_2, \ldots, r_G\}$ denotes the set of rewards within a sampled group.

**Length Normalization Module**:
Trajectory balance treats both the initial flow and the outcome reward as sequence-level quantities. For trajectories of varying lengths (e.g., CoT responses), the log-probability term $\log \pi_\theta(y | x) = \sum_{t=1}^{|y|} \log \pi_\theta(y_t | y_{<t},x)$ can scale with sequence length, potentially causing exploding gradients. To address this, a form of reward shaping is applied by normalizing log-probabilities with respect to sequence length. Specifically, the term is rescaled as $\frac{1}{|y|} \log \pi_\theta(y | x)$, balancing the contributions of long and short sequences and stabilizing the learning signal.

**Importance Sampling Module**:
Mainstream RL algorithms commonly perform micro-batch updates and reuse trajectories collected from an old policy $\pi_{\theta_{\text{old}}}$, enabling data-efficient training. To mitigate sampling mismatch, importance sampling inspired by PPO is employed to stabilize policy updates with off-policy data. Stale trajectories are re-weighted using the importance ratio $w = \pi_\theta(y | x) / \pi_{\text{old}}(y | x)$, which serves as a coefficient in the surrogate loss. Since the objective focuses on optimizing trajectory balance rather than expected return, the gradient from the current policy is detached to prevent excessive policy drift: $w = \text{detach}[\pi_\theta(y | x)] / \pi_{\text{old}}(y | x)$. For additional stability, PPO-style clipping is incorporated to bound the importance weights: $w = \text{clip}\left(\frac{\pi_\theta(y|x)}{\pi_{\text{old}}(y|x)}, 1-\epsilon, 1+\epsilon\right)_{\text{detach}}$.

7. **All Mathematical Formulas and Symbol Definitions**

**GRPO Objective**:
$J_{\text{GRPO}}(\theta) = \mathbb{E}_{[x \sim P(X), \{y_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(Y|x)]} \frac{1}{G} \sum_{i=1}^G \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \left\{ \min \left[ \frac{\pi_\theta(y_{i,t}|x,y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t}|x,y_{i,<t})} \hat{A}_{i,t}, \text{clip} \left( \frac{\pi_\theta(y_{i,t}|x,y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t}|x,y_{i,<t})}, 1-\epsilon, 1+\epsilon \right) \hat{A}_{i,t} \right] - \lambda D_{\text{KL}} [\pi_\theta || \pi_{\text{ref}} ] \right\}$

$D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) = \frac{\pi_{\text{ref}}(y_i|x)}{\pi_\theta(y_i|x)} - \log \frac{\pi_{\text{ref}}(y_i|x)}{\pi_\theta(y_i|x)} - 1$

**Distribution Matching Objective**:
$\min_\theta D_{\text{KL}} \left( \pi_\theta(y | x) \left\| \frac{\exp(\beta r(x,y))}{Z_\phi(x)} \right. \right) \Rightarrow \pi_\theta(y | x) \propto \exp(\beta r(x,y))$

**Trajectory Balance Equivalence (Proposition 1)**:
$\min_\theta D_{\text{KL}} \left( \pi_\theta(y | x) \left\| \frac{\exp(\beta r(x,y))}{Z_\phi(x)} \right. \right) \iff \min_\theta \left( \log Z_\phi(x) + \log \pi_\theta(y | x) - \beta r(x,y) \right)^2$

**Redefined Reward with Reference Model**:
$\exp (\beta r(x,y)) \cdot \pi_{\text{ref}}(y | x)$

**FlowRL Objective (without IS/length norm)**:
$\min_\theta \left( \log Z_\phi(x) + \log \pi_\theta(y | x) - \beta \hat{r}_i(x,y) - \log \pi_{\text{ref}}(y | x) \right)^2$

**Final FlowRL Objective**:
$\mathcal{L}_{\text{FlowRL}} = w \cdot \left( \log Z_\phi(x) + \frac{1}{|y|} \log \pi_\theta(y | x) - \beta \hat{r}(x,y) - \frac{1}{|y|} \log \pi_{\text{ref}}(y | x) \right)^2$

**Importance Weight and Normalized Reward**:
$w = \text{clip}\left(\frac{\pi_\theta(y | x)}{\pi_{\text{old}}(y | x)}, 1-\epsilon, 1+\epsilon\right)_{\text{detach}}, \quad \hat{r}_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$

**Proof of Proposition 1**:
$\nabla_\theta D_{\text{KL}} \left( \pi_\theta(y | x) \left\| \frac{\exp(\beta r(x,y))}{Z_\phi(x)} \right. \right) = \mathbb{E}_{y \sim \pi_\theta(\cdot|x)} \left[ \log \left( \frac{Z_\phi(x)\pi_\theta(y | x)}{\exp(\beta r(x,y))} \right) \cdot \nabla_\theta \log \pi_\theta(y | x) \right]$

$L(y,x; \theta) = \left( \log \frac{Z_\phi(x)\pi_\theta(y | x)}{\exp(\beta r(x,y))} \right)^2$

$\nabla_\theta L(\theta) = 2 \cdot \mathbb{E}_{y \sim \pi_\theta(\cdot|x)} \left[ \left( \log \frac{Z_\phi(x) \cdot \pi_\theta(y | x)}{\exp(\beta r(x,y))} \right) \cdot \nabla_\theta \log \pi_\theta(y | x) \right]$

**Theoretical Analysis (Proposition 5)**:
$\max_\theta \mathbb{E}_{y \sim \pi_\theta} \left[ \underbrace{\beta r(x,y)}_{\text{reward}} - \log Z_\phi(x) + \log \pi_{\text{ref}}(y|x) \right] + \underbrace{H(\pi_\theta)}_{\text{entropy}}$

$D_{\text{KL}} \left( \pi_\theta(y | x) \left\| \frac{\exp (\beta r(x,y)) \cdot \pi_{\text{ref}}(y | x)}{Z_\phi(x)} \right. \right) = \int \pi_\theta(y | x) \log \left[ \frac{Z_\phi(x)\pi_\theta(y | x)}{\exp (\beta r(x,y)) \cdot \pi_{\text{ref}}(y | x)} \right] dy$

$\text{argmin}_\theta D_{\text{KL}} \left( \pi_\theta(y | x) \left\| \frac{\exp (\beta r(x,y)) \cdot \pi_{\text{ref}}(y | x)}{Z_\phi(x)} \right. \right) = \text{argmax}_\theta \left\{ \mathbb{E}_{y \sim \pi_\theta(\cdot|x)} \log \left[ \frac{\exp (\beta r(x,y)) \cdot \pi_{\text{ref}}(y | x)}{Z_\phi(x)} \right] - \int \pi_\theta(y | x) \log \pi_\theta(y | x) dy \right\} = \text{argmax}_\theta \left\{ \mathbb{E}_{y \sim \pi_\theta(\cdot|x)} \log \left[ \frac{\exp (\beta r(x,y)) \cdot \pi_{\text{ref}}(y | x)}{Z_\phi(x)} \right] + H(\pi_\theta) \right\}$

$\max_\theta \mathbb{E}_{y \sim \pi_\theta(\cdot|x)} \left[ \underbrace{\beta r(x,y)}_{\text{reward}} - \underbrace{\log Z_\phi(x)}_{\text{normalization}} + \underbrace{\log \pi_{\text{ref}}(y|x)}_{\text{reference model constraint}} \right] + \underbrace{H(\pi_\theta)}_{\text{entropy}}$

**GFlowNets Detailed Balance**:
$\forall (s \to s') \in \mathcal{A}, F_\theta(s)P_F(s' | s; \theta) = F_\theta(s')P_B(s | s'; \theta)$

**GFlowNets Trajectory Balance**:
$Z_\theta \prod_{t=1}^n P_F(s_t | s_{t-1}; \theta) = R(x) \prod_{t=1}^n P_B(s_{t-1} | s_t; \theta)$

8. **Algorithm Pseudocode**
No explicit algorithm pseudocode block is provided in the paper. The iterative process is defined by the FlowRL objective in Equation 6. The training steps consist of:
1. For each question $x$, sample a group of answers $\{y_1, y_2, \ldots, y_G\}$ from old policy $\pi_{\text{old}}$.
2. Compute the outcome rewards $r = \{r_1, r_2, \ldots, r_G\}$ and normalize them as $\hat{r}_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$.
3. Compute the clipped importance weight $w = \text{clip}\left(\frac{\pi_\theta(y | x)}{\pi_{\text{old}}(y | x)}, 1-\epsilon, 1+\epsilon\right)_{\text{detach}}$.
4. Compute the scalar partition function value $Z_\phi(x)$ using the 3-layer MLP taking the mean of the language model's hidden states of $x$ as input.
5. Compute the FlowRL loss $\mathcal{L}_{\text{FlowRL}} = w \cdot \left( \log Z_\phi(x) + \frac{1}{|y|} \log \pi_\theta(y | x) - \beta \hat{r}(x,y) - \frac{1}{|y|} \log \pi_{\text{ref}}(y | x) \right)^2$.
6. Update the policy parameters $\theta$ and partition function parameters $\phi$ using this loss.