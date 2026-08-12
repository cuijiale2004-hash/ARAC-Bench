# Research Idea and Implementation Plan Extraction

## 1. Research Background and Existing Pain Points
Reinforcement Learning (RL), particularly through policy gradient methods, has played a central role in enabling reasoning capabilities of Large Language Models (LLMs). Despite its success in LLM fine-tuning and decision-making tasks, RL still faces fundamental challenges that limit its broader practicality and scalability. In particular, policy gradients suffer from optimization instabilities driven by the non-stationary nature of the RL objective and the high variance of estimates. These problems are further compounded by the known pathologies of training deep networks. These factors lead to several undesired consequences, such as catastrophic updates and policy collapse, plasticity loss, sample inefficiency, and hyperparameter sensitivity. These challenges persist and may be even more pronounced in the LLM setting, since training involves billion-parameter models with very deep architectures and sampling horizons that can extend arbitrarily. In practice, current implementations of RL for LLMs typically rely on conservative hyperparameters to ensure stability, such as low learning rates (e.g., $3 \times 10^{-6}$ or less) and large batch sizes (e.g., thousands of generations per policy update). These choices substantially increase the number of LLM generations required for learning, raising computational costs. Therefore, stabilizing these algorithms in sample-efficient regimes is crucial to further scale RL for LLM reasoning.

## 2. Core Research Motivation and Scientific Questions
The optimization dynamics of RL in the context of LLMs remains underexplored. Existing implementations often resort to conservative hyper-parameter choices to ensure stability, which requires more training samples and increases computational costs. Hence, developing models for reliably tracking the underlying optimization dynamics and leveraging them into training enables more sample-efficient regimes and further unleashes scalable post-training. The core scientific question is: How can we explicitly model second-order geometry in the optimization landscape of policy gradients for LLMs in a computationally tractable manner, and leverage this information to anticipate and intervene in potentially unstable updates to achieve sample-efficient reinforcement learning?

## 3. Overall Core Idea and Design Philosophy
The overall core idea is to formalize the stochastic optimization problem of policy gradients with explicit consideration of second-order geometry. The design philosophy involves proposing a tractable computational framework that tracks and leverages curvature information during policy updates. This approach can be viewed as a form of model-based RL, but instead of modeling components of the MDP, the optimization process itself is modeled. This allows "planning" gradient estimates by anticipating policy updates that potentially induce sudden shifts in the objective or policy distribution. The framework further employs this model to design interventions in the optimization process through data selection. The resultant algorithm, Curvature-Aware Policy Optimization (CAPO), identifies samples that contribute to unstable updates and masks them out, enforcing a trust region without explicitly materializing intractable curvature matrices.

## 4. Core Innovation Points
1.  **Formalization of Second-Order Geometry**: Formalizing the RL optimization problem accounting for curvature terms, namely the Hessian of the objective and the Fisher Information Matrix of the policy distribution, to track sudden shifts in the objective or policy.
2.  **Tractable Computational Model**: Introducing a computationally and numerically tractable model of optimization dynamics that approximates curvature information via a last-layer model, scales to billion-parameter models, and provides analytical expressions for directional curvatures without materializing the full Hessian or Fisher matrices.
3.  **Data Selection Intervention Mechanism**: Proposing a simple data selection mechanism as intervention that identifies particular samples (tokens) that heavily contribute to abrupt shifts and masks them out of the policy gradient estimation, resulting in Curvature-Aware Policy Optimization (CAPO).
4.  **Theoretical Guarantees**: Establishing monotonic policy improvement guarantees under CAPO with practical assumptions, proving that the data selection mechanism ensures stable updates.

## 5. Overview of the Overall Technical Solution
The overall technical solution begins by modeling the optimization landscape. The objective function and policy distribution shifts after an update step $\Delta\theta$ are formulated using second-order Taylor expansions involving the Hessian and Fisher Information Matrix. Since computing these matrices is intractable for LLMs, a last-layer model is adopted where the policy is parameterized by a pre-softmax layer. Under this model, the gradient, Hessian, and Fisher matrices are expressed using Kronecker products of policy error vectors and feature vectors. To further reduce complexity, directional curvatures ($\Delta\theta^\top C \Delta\theta$) are computed without materializing the full matrices by exploiting Kronecker-vector identities. Gradient sparsity from selective sampling (top-k) is exploited to reduce memory and computation. Finally, the CAPO algorithm evaluates proposed update steps on disjoint subsets of a batch. If the estimated objective shift $m_H$ and policy shift $m_F$ violate trust-region constraints ($\delta_H \le m_H(\Delta\psi_i) \le \delta_H^{high}$, $m_F(\Delta\psi_i) \le \delta_F$), the subset is rejected; otherwise, it is accepted and used for the actual LLM policy gradient update.

## 6. Detailed Module Design

### Module 1: Second-Order Geometry Modeling
This module tracks the optimization landscape using higher-order objective expansion and policy divergence approximation. For an update $\Delta\theta$, the new objective $J(\theta + \Delta\theta)$ is approximated by a Taylor expansion capturing the Hessian contribution. Simultaneously, the policy shift is measured by the average KL divergence, approximated by the directional curvature under the Fisher geometry. This provides the mathematical foundation for detecting unstable updates.

### Module 2: Last-Layer Computational Model
To avoid intractable full-parameter curvature matrices, the model restricts attention to the last layer. The LLM is treated as a softmax policy where the pre-softmax layer is represented as $f_\theta(s_t) = W h_{\bar{\theta}}(s_t)$. The parameter vector is defined as $\psi = \text{vec}(W)$. Under this parameterization, the gradient, Hessian, and Fisher Information Matrix are derived analytically. They take the form of Kronecker products involving the policy error vector $(e_a - \pi_\theta(s_t))$ and the outer product of feature vectors $h_{\bar{\theta}}(s_t)h_{\bar{\theta}}(s_t)^\top$. 

### Module 3: Directional Curvature Computation
Since even the last-layer curvature matrices are too large to materialize ($K d_i \times K d_i$), this module computes only the directional curvatures required for the trust region. Using the Kronecker-vector identity $\text{vec}(X)^\top (A \otimes B) \text{vec}(X) = \text{Tr}(A X B X^\top)$, the quadratic forms $\Delta\psi^\top \tilde{F}(\psi)\Delta\psi$ and $\Delta\psi^\top \tilde{H}(\psi)\Delta\psi$ are reduced to simple vector dot products and traces. Specifically, defining $U = \text{unvec}(\Delta\psi)$, $u_t = e_{a_t} - \pi_\theta(s_t)$, and $v_t = U h_t$, the directional curvatures simplify to expectations of scalar quantities $(u_t^\top v_t)^2$ and $(u_t^\top v_t)^2 - v_t^\top F(s_t) v_t$.

### Module 4: Data Selection Intervention (CAPO)
Given a batch of trajectories, it is partitioned into disjoint subsets. For each subset, a proposed step $\Delta\psi_i$ is computed (modeled as SGD or Adam steps). The shifts $m_H$ and $m_F$ are estimated. A local trust-region acceptance test is applied: a subset is accepted only if $\delta_H \le m_H(\Delta\psi_i) \le \delta_H^{high}$ and $m_F(\Delta\psi_i) \le \delta_F$. Accepted subsets are aggregated to compute the actual policy gradient update for the LLM, effectively masking out tokens that cause unstable curvature.

## 7. All Mathematical Formulas and Symbol Definitions

**MDP and Policy Gradient Definitions:**
*   $M = (S, A, P, R, \rho_0, \gamma, T)$: MDP tuple.
*   $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \gamma^t R(s_t, a_t) \right]$: Expected cumulative reward.
*   $\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \gamma^t \nabla_\theta \log \pi_\theta(a_t | s_t) R(s_t, a_t) \right]$: Policy Gradient.

**GRPO Objective and Advantage:**
*   $J_{\text{GRPO}}(\theta) = \mathbb{E}_{\tau \sim \pi_\beta} \left[ \frac{1}{|\tau_i|} \sum_{t=0}^{|\tau_i|} \min \left( r_\theta(s_t, a_t), \text{clip}(r_\theta(s_t, a_t), 1-\epsilon, 1+\epsilon) \right) \hat{A}_{\text{GRPO}}(s_t, a_t) - \beta D_{KL}(\pi_\theta(\cdot | s_t) \| \pi_{\text{base}}(\cdot | s_t)) \right]$
*   $r_\theta(s_t, a_t) = \frac{\pi_\theta(a_t|s_t)}{\pi_\beta(a_t|s_t)}$
*   $\hat{A}_{\text{GRPO}}(s_t, a_t) = \frac{\hat{R}(\tau) - \bar{R}}{\hat{\sigma}_R + \varepsilon}$, where $\bar{R} = \frac{1}{G}\sum_{i=1}^G \hat{R}(\tau_i)$, $\hat{\sigma}_R = \sqrt{\frac{1}{G}\sum_{i=1}^G (\hat{R}(\tau_i) - \bar{R})^2}$.

**Second-Order Geometry:**
*   $J(\theta + \Delta\theta) = J(\theta) + \nabla_\theta J(\theta)^\top \Delta\theta + \underbrace{\frac{1}{2} \Delta\theta^\top H(\theta) \Delta\theta}_{m_H(\Delta\theta)} + O(\|\Delta\theta\|^3)$
*   $\bar{D}_{KL}(\pi_\theta \| \pi_{\theta+\Delta\theta}) = \underbrace{\frac{1}{2}\Delta\theta^\top F(\theta) \Delta\theta}_{m_F(\Delta\theta)} + O(\|\Delta\theta\|^3)$
*   $F(\theta) := \mathbb{E}_{s \sim d^\pi, a \sim \pi_\theta(\cdot|s)} \left[ \nabla_\theta \log \pi_\theta(a | s) \nabla_\theta \log \pi_\theta(a | s)^\top \right]$

**Last-Layer Model:**
*   $\pi_\theta(a | s) = \frac{\exp(f_\theta(s,a))}{\sum_{a'} \exp(f_\theta(s,a'))}$, $f_\theta(s_t) = W h_{\bar{\theta}}(s_t)$, $\psi = \text{vec}(W) \in \mathbb{R}^{K \cdot d_i}$.
*   Model Gradient: $\tilde{g}(\psi) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \gamma^t A(s_t, a_t) (e_a - \pi_\theta(s_t)) \otimes h_{\bar{\theta}}(s_t) \right]$
*   Model Hessian: $\tilde{H}(\psi) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \gamma^t A(s, a) \left( (e_a - \pi_\theta(s_t))(e_a - \pi_\theta(s_t))^\top - F(s_t) \right) \otimes h_{\bar{\theta}}(s_t)h_{\bar{\theta}}(s_t)^\top \right]$
*   Model Fisher: $\tilde{F}(\psi) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \left( (e_{a_t} - \pi_\theta(s_t))(e_{a_t} - \pi_\theta(s_t))^\top \right) \otimes h_{\bar{\theta}}(s_t)h_{\bar{\theta}}(s_t)^\top \right]$

**Directional Curvatures:**
Let $U = \text{unvec}(\Delta\psi)$, $u_t = e_{a_t} - \pi_\theta(s_t)$, $v_t = U h_t$.
*   Fisher Curvature: $\Delta\psi^\top \tilde{F}(\psi)\Delta\psi = \mathbb{E}_{\tau \sim \pi_\theta} \left[ (u_t^\top v_t)^2 \right]$
*   Hessian Curvature: $\Delta\psi^\top \tilde{H}(\psi)\Delta\psi = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \gamma^t A(s_t, a_t) \left( (u_t^\top v_t)^2 - v_t^\top F(s_t) v_t \right) \right]$
*   Sample Estimators:
    $\widehat{\Delta\psi^\top \tilde{F}\Delta\psi} = \frac{1}{N}\sum_{i=1}^N (u_i^\top \hat{v}_i)^2$
    $\widehat{\Delta\psi^\top \tilde{H}\Delta\psi} = \frac{1}{N}\sum_{i=1}^N \gamma^{t_i} A(s_i, a_i) \left( (u_i^\top \hat{v}_i)^2 - \hat{v}_i^\top F(s_i) \hat{v}_i \right)$

**CAPO Shifts and Constraints:**
*   $m_H(\psi) = \tilde{g}(\psi)^\top \Delta\psi + \frac{1}{2}\Delta\psi^\top \tilde{H}(\psi)\Delta\psi$
*   $m_F(\psi) = \frac{1}{2}\Delta\psi^\top \tilde{F}(\psi)\Delta\psi$
*   Constraints: $\delta_H \le m_H(\Delta\psi_i) \le \delta_H^{high}$, $m_F(\Delta\psi_i) \le \delta_F$

**Theoretical Guarantee (Theorem 5.1):**
*   Assumption E.1: $\|H(\theta)\|_{\text{op}} \le M$ and $\|\Delta\theta\| \le r$.
*   If $m_H(\Delta\theta_i) \ge \delta_H = \omega + \frac{1}{2}Mr^2$ and $m_F(\Delta\theta_i) \le \delta_F$, then for aggregated update $\Delta\theta = \frac{1}{B}\sum_{i \in B_{\text{acc}}} \Delta\theta_i$:
    $J(\pi_{\theta+\Delta\theta}) - J(\pi_\theta) \ge \omega - C\sqrt{\delta_F}$, where $C = \frac{2\gamma}{(1-\gamma)^2}\epsilon\sqrt{2}$.
    Choosing $\omega \ge C\sqrt{\delta_F}$ guarantees monotonic improvement.

## 8. Algorithm Pseudocode

**Algorithm 1: Curvature-Aware Policy Optimization (CAPO)**
Input : Policy $\pi_\theta$; batch $B$ of sampled trajectories; thresholds $(\delta_F, \delta_H, \delta_H^{high})$; optimizer for the last-layer model (e.g., SGD or Adam).
Output: Updated policy parameters $\theta$

while not done do
    // Collect data with the current policy
    Sample a batch $B = \{\tau\}_i^N$ of trajectories, $\tau \sim \pi_\theta$.
    Partition $B$ into disjoint subsets $\{b_i\}_{i=1}^N$.
    for $i = 1, \ldots, N$ in parallel do
        // Build last-layer meta-model stats on subset $b_i$
        Estimate model-based gradient $\tilde{g}(\psi)$ using Equation 7;
        Propose $\Delta\psi_i$ with the optimizer model (e.g., $\Delta\psi_i = \alpha \tilde{g}(\psi)$ for SGD, or Adam’s rule)
        Compute directional curvatures $\frac{1}{2}\Delta\psi^\top \tilde{H}(\psi)\Delta\psi$, $\Delta\psi^\top \tilde{F}(\psi)\Delta\psi$ as in Appendix D;
        Compute objective and policy shifts under the last-layer model:
        $m_H(\Delta\psi) \leftarrow \tilde{g}(\psi)^\top \Delta\psi + \frac{1}{2}\Delta\psi^\top \tilde{H}(\psi)\Delta\psi$, $m_F(\Delta\psi) \leftarrow \frac{1}{2}\Delta\psi^\top \tilde{F}(\psi)\Delta\psi$.
        
        // Local trust-region acceptance test
        if $\delta_H \le m_H(\Delta\psi_i) \le \delta_H^{high}$ and $m_F(\Delta\psi_i) \le \delta_F$ then
            Mark subset $b_i$ as ACCEPT; add to $B_{\text{acc}}$.
        else
            REJECT $b_i$.
            
    // Compute the actual policy update on accepted data
    if $B_{\text{acc}} \neq \emptyset$ then
        Estimate the objective on accepted samples (e.g., GRPO/PPO surrogate):
        $\hat{J}(\theta) = \text{pg-objective}(\pi_\theta; \bigcup_{b_i \in B_{\text{acc}}} b_i)$.
        // Policy Gradient and parameter update
        $\theta \leftarrow \theta + \alpha \hat{\nabla}_\theta J$
return $\theta$