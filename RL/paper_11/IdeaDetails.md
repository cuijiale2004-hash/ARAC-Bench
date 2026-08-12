1. **Research Background and Existing Pain Points**
As single-center computing approaches power constraints, decentralized training becomes essential. However, traditional Reinforcement Learning (RL) methods, crucial for enhancing large model post-training, cannot adapt to decentralized distributed training due to the tight coupling between parameter learning and rollout sampling. This architectural incompatibility manifests in two critical bottlenecks. First, synchronous frameworks force computational resources (e.g., GPUs) to idle while waiting for the slowest processes—such as generating long reasoning chains—severely constraining efficiency. Second, and more fundamentally, the inevitable network latency inherent in decentralized, Internet-connected environments creates a temporal gap (policy staleness) between the sampler (generating data) and the learner (updating parameters). Most existing RL algorithms, designed for homogeneous, low-latency clusters, are ill-equipped to handle this staleness. As the analysis reveals, high latency significantly inflates KL divergence, causing the variance of importance weights to explode—ultimately leading to training instability or reward collapse. This renders conventional RL methods impractical for real-world, geographically distributed training scenarios.

2. **Core Research Motivation and Scientific Questions**
The core research motivation is to tackle the systemic bottlenecks in decentralized RL, enabling efficient and stable training of LLMs even under high network latency and computational heterogeneity. The central scientific questions are: How to design a heterogeneous RL architecture that decouples rollout sampling and parameter learning? How to stabilize asynchronous RL under high latency? How to address the explosion of variance of importance weights caused by stale policies and high KL divergence?

3. **Overall Core Idea and Design Philosophy**
The overall core idea is to propose HeteroRL, a heterogeneous RL architecture that decouples rollout sampling and parameter learning, enabling stable training across geographically distributed nodes connected via the Internet. The core design philosophy is to address the instability arising from KL divergence under high-latency conditions by introducing Group Expectation Policy Optimization (GEPO), a novel policy gradient algorithm that stabilizes asynchronous RL under high latency by replacing fragile token/sample-level importance weights with robust group-level importance weights. This shift fundamentally improves the quality of gradient estimation—transforming a high-variance, unstable estimator into a low-variance, robust one, especially under large policy divergence. The progression—from token-level (GRPO) to sequence-level (GSPO) to group-level (GEPO) coefficients—demonstrates that coarser importance-weight granularity significantly reduces gradient variance.

4. **Core Innovation Points**
*   **Framework:** We propose HeteroRL, an asynchronous reinforcement learning framework designed for heterogeneous compute networks, enabling decentralized training of large language models (LLMs) on mathematical reasoning tasks.
*   **Insight:** We identify a strong correlation between latency and the KL divergence between the rollout sampler and the learner. High latency induces high KL divergence, leading to training instability and reward collapse.
*   **Algorithm:** We introduce Group Expectation Policy Optimization (GEPO), which improves upon the importance sampling mechanism in GRPO. We theoretically show that GEPO exponentially reduces the variance of importance weights, and empirically demonstrate its superior stability and efficiency—not only under high-latency conditions, but also in the ideal zero-latency setting.
*   **Group Expectation Importance Weight (GEIW):** Replaces the individual proposal probability in the standard importance weight with its group-wise expected value under the current prompt, which is numerically stable, gradient-effective, biased yet low-variance.
*   **Engineering Optimization:** Localized Reward Computation to reduce communication overhead, removing the global gather operation and ensuring group statistics are computed locally.

5. **Overview of the Overall Technical Solution**
HeteroRL decouples the two computationally intensive phases of the RL pipeline—rollout sampling and parameter learning—and deploys them on physically or logically independent nodes with potentially heterogeneous hardware (e.g., mixing NVIDIA and Ascend chips). The sampler nodes continuously generate reasoning trajectories without interruption, while the learner node asynchronously consumes this data to update model parameters. Critically, neither component waits for the other: communication occurs infrequently and tolerates high latency, with model checkpoints and rollout batches exchanged over the Internet. The core component is Group Expectation Policy Optimization (GEPO), an asynchronous RL algorithm robust to latency caused by network delays or heterogeneity in computational resources. GEPO mitigates the issue of variance explosion by using group expectation weighting to exponentially reduce the variance of importance weights, with theoretical guarantees.

6. **Detailed Module Design**
*   **HeteroRL Architecture:** The architecture forms a star-shaped network topology centered at the learner. It consists of one parameter update node (learner) and multiple data generation nodes (sampler). During training, the sampler nodes generate rollout data, which is transmitted over the network to the learner node in a streaming fashion. The learner updates the model parameters and periodically broadcasts the updated weights back to the sampler nodes. The learner processes incoming rollouts in the order they arrive, operating within a fixed time window for data eligibility. Since data is transmitted in batch units—each containing text, generation probabilities, and rewards—a maximum delay of 1800 seconds is sufficient for typical network conditions. Network delays between the sampler and learner nodes are explicitly modeled and can be simulated using stochastic distributions such as the log-normal or Weibull distribution.
*   **Group Expectation Importance Weighting Module:** To enhance the stability of importance weights, the GEIW replaces the individual proposal probability $q(y|x)$ in the standard importance weight $\frac{p(y|x)}{q(y|x)}$ with its group-wise expected value under the current prompt $x$, denoted as $\hat{E}_q[q(y|x)]$. Inspired by GRPO, for each input $x$, it generates a group of $G$ responses $\{y_1, \ldots, y_G\} \sim q(\cdot|x)$ to form a sampling group. Since $G$ is typically much smaller than the full policy space and top-P/top-K sampling leads to $\sum_{i=1}^G q(y_i|x) \gg 1$, the vector $(q(y_1|x), \ldots, q(y_G|x))$ does not constitute a valid probability distribution. Simply using the arithmetic mean $\frac{1}{G}\sum_{i=1}^G q(y_i|x)$ would introduce bias due to ignoring the relative sampling probabilities. To obtain a more accurate estimate, a weighted expectation is employed. The key advantages of this mechanism are: Numerically stable and gradient-effective (the denominator is decoupled from any single $q(y|x)$, avoiding extreme weight values when individual proposal probabilities approach zero); Biased yet low-variance (by leveraging within-group statistical information, GEIW provides a more reliable scale estimate, effectively preventing gradient explosion).
*   **Localized Reward Computation Module:** Instead of gathering rewards from all processes, the system ensures that each group of generations (e.g., $G$ responses per prompt) is entirely generated and scored within the same process or node. This design guarantees that all samples belonging to a single group reside locally, enabling the calculation of group statistics (mean, std) without any cross-process communication. The core change is implemented by removing the global gather operation in the reward calculation function.

7. **All Mathematical Formulas and Symbol Definitions**
**Notation:**
*   $\pi_\theta$: Language model policy (Actor) parameterized by $\theta$.
*   $x$: Input prompt of a Dataset $D$.
*   $y$: Output sequence generated by the model.
*   $\pi_{\theta_k}$ (short for $q$): The policy used by the sampler at time step $k$ to generate rollout trajectories.
*   $\pi_{\theta_{k+\tau}}$ (short for $p$): The latest policy at the learner at time step $k+\tau$, used for gradient updates.
*   $\tau (\ge 0)$: Policy staleness, representing the discrepancy in policy versions between the sampler and the learner.
*   $y_i^t$: The $t$-th token of the $i$-th response in a group.
*   $r(x,y)$: The reward for response $y$ given input $x$.
*   $A(x,y)$: The advantage for response $y$ to input $x$, defined as $A(x,y) = r(x,y) - b(x)$, where $b(x)$ is a baseline reward computed for input $x$. The within-group average reward is used as the baseline $b(x) = \frac{1}{G}\sum_{i=1}^G r(x, y_i)$ where $G$ is the group size.

**Formulas:**
Objective:
$$L(\theta) = \mathbb{E}_{x \sim D, y \sim \pi_{\theta_k}(\cdot|x)} \left[ \frac{\pi_{\theta_{k+\tau}}(y|x)}{\pi_{\theta_k}(y|x)} \cdot A(x, y) \right]$$

Group Expectation Estimate:
$$\hat{E}_q[q(y|x)] \approx \sum_{i=1}^G \div q(y_i|x) \cdot q(y_i|x) = \frac{\sum_{i=1}^G q(y_i|x)^2}{\sum_{i=1}^G q(y_i|x)}$$
where $\div q(y_i|x) = \frac{q(y_i|x)}{\sum_{i=1}^G q(y_i|x)}$

GEIW importance weight:
$$w_{GEIW}(y|x) = \frac{p(y|x)}{\hat{E}_q[q(y|x)]}$$

Gradient Comparison Across Tokens:
$$\frac{\partial L(\theta)}{\partial \theta} = A \odot \left[ \begin{matrix} \frac{p'_{1,1}(\theta)}{q_{1,1}} & \cdots & \frac{p'_{1,T}(\theta)}{q_{1,T}} \\ \vdots & \ddots & \vdots \\ \frac{p'_{G,1}(\theta)}{q_{G,1}} & \cdots & \frac{p'_{G,T}(\theta)}{q_{G,T}} \end{matrix} \right]_{\text{GRPO}} \text{ or } \left[ \begin{matrix} \frac{p'_{1,1}(\theta)}{q_1} & \cdots & \frac{p'_{1,T}(\theta)}{q_1} \\ \vdots & \ddots & \vdots \\ \frac{p'_{G,1}(\theta)}{q_G} & \cdots & \frac{p'_{G,T}(\theta)}{q_G} \end{matrix} \right]_{\text{GSPO}} \text{ or } \left[ \begin{matrix} \frac{p'_{1,1}(\theta)}{Eqq} & \cdots & \frac{p'_{1,T}(\theta)}{Eqq} \\ \vdots & \ddots & \vdots \\ \frac{p'_{G,1}(\theta)}{Eqq} & \cdots & \frac{p'_{G,T}(\theta)}{Eqq} \end{matrix} \right]_{\text{GEPO (ours)}}$$

Variance Comparison (Theorem 1):
$$\text{Var}\left[\frac{p(y|x)}{q(y|x)}\right] - \text{Var}\left[\frac{p(y|x)}{\hat{E}_q[q(y|x)]}\right] \ge \exp (D_{KL}(p\|q)) - C$$

Variance Expressions (Appendix A.1):
$$\text{Var}_{std} = \int \frac{p(y|x)^2}{q(y|x)} dy - 1$$
$$\text{Var}_{new} = \frac{1}{(\int q(y|x)^2 dy)^2} \left[ \int p(y|x)^2 q(y|x) dy - \left(\int p(y|x)q(y|x) dy\right)^2 \right]$$

Discrete Space Variance Difference $\Delta$:
$$A = \left(\sum_{i=1}^n q_i^2\right)^2, \quad B = \left(\sum_{i=1}^n p_i q_i\right)^2, \quad I_1 = \sum_{i=1}^n \frac{p_i^2}{q_i}, \quad I_2 = \sum_{i=1}^n p_i^2 q_i$$
$$\Delta = I_1 + \frac{B - A - I_2}{A}$$
$$\Delta \ge \exp(D_{KL}(p\|q)) - (n^2 + 1)$$

Bias of GEPO (Theorem 2):
$$\text{Bias}(\text{GEPO}) = \left| \mathbb{E}_p[A(x,y)] - \mathbb{E}_q\left[\frac{p(y|x)}{E_q[q]} A(x,y)\right] \right| < \frac{\|p\|_2}{\|q\|_2}$$

Variance of GEPO (Theorem 3):
$$\text{Var}_q\left[A(x,y) \cdot \frac{p(y|x)}{q(y|x)}\right] - \text{Var}_q\left[A(x,y) \cdot \frac{p(y|x)}{E_q[q(y|x)]}\right] \ge A_{min}^2 \cdot \exp(D_{KL}(p\|q)) - C_{adv}$$

8. **Algorithm Pseudocode**
```python
1 if self.loss_type in ["grpo","dr_grpo","bnpo"]: # Token level
2     coef_1 = learner_token_p / sampler_token_p
3 elif self.loss_type == "gspo": # Sequence level
4     coef_1 = learner_seq_p / sampler_seq_p
5 elif self.loss_type == "gepo": # Group level
6     hat_q = sampler_seq_p.detach () / (sampler_seq_p.sum().detach ())
7     coef_1 = learner_seq_p / (hat_q * sampler_seq_p).sum()
```