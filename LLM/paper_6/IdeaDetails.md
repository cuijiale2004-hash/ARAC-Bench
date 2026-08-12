# Research Idea and Implementation Plan Extraction

## 1. Research Background and Existing Pain Points

**Research Background:**
Reinforcement learning (RL) has become central to enhancing reasoning in large language models (LLMs). On-policy algorithms such as Group Relative Policy Optimization (GRPO) have been widely applied in large-scale LLM training pipelines, demonstrating substantial improvements on mathematical reasoning and showing that RL signals alone can induce emergent reasoning behaviors across various domains.

**Existing Pain Points:**
Despite these successes, GRPO inherits structural inefficiencies that make training fragile in LLM reasoning:
1. **Noisy Gradients and Unstable Updates:** During the early stages of training, when rollouts are particularly weak or uninformative, stochastic rewards induce high-variance gradients that destabilize updates. Group normalization partially dampens this effect but remains sensitive to fluctuations within each batch.
2. **Inefficient Data Use:** Common practice in GRPO implementations is to use a single optimization step per rollout batch. Such a one-shot update tends to produce a noisier, less reliable step update direction, especially at the early training stages, thereby underutilizing each batch and discarding information that could be further exploited.
3. **Degradation from Naive Off-Policy Updates:** Although reusing rollout data (off-policy updates) has been explored, simply applying this off-policy update to the policy model can degrade performance as training proceeds due to distribution mismatch. How to use rollout data effectively to build strong reasoning abilities in LLMs with satisfying sample efficiency remains an open question.

## 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:**
To address the instability and inefficiency of one-shot policy updates in GRPO. The motivation is to design a simple yet effective enhancement to on-policy policy-gradient methods that stabilizes noisy updates and improves sample efficiency while leaving the base loss, rollout collection, and KL/clip regularization unchanged.

**Scientific Questions:**
How to reuse rollout batches effectively to stabilize the search direction and control off-policy drift? How to transform noisy one-shot updates into structured trajectories that yield more stable optimization, higher sample efficiency, and faster convergence without additional rollouts or changes to the underlying objective?

## 3. Overall Core Idea and Design Philosophy

**Overall Core Idea:**
Introduce Slow–Fast Policy Optimization (SFPO), a simple yet effective enhancement to on-policy policy-gradient methods that stabilizes noisy updates and improves sample efficiency. Each training step is restructured into three coordinated stages: (i) fast—multiple inner updates on the same batch to stabilize the search direction; (ii) reposition—interpolation back toward the starting point to control off-policy drift; and (iii) slow—an extra gradient correction to align with local curvature. This reposition-before-update design preserves the objective and rollout process unchanged.

**Design Philosophy:**
The design philosophy is plug-compatibility with existing policy-gradient pipelines. SFPO introduces no changes to the underlying objective, rollout generation, or regularization, enabling drop-in integration into existing LLM reasoning pipelines. It transforms noisy one-shot updates into well-structured update trajectories by decomposing the update into a fast–reposition–slow sequence, yielding more stable optimization and higher sample efficiency.

## 4. Core Innovation Points

1. **Fast–Reposition–Slow Decomposition:** Proposes SFPO, a plug-compatible update rule that reuses rollout batches via a fast–reposition–slow decomposition, directly addressing the instability and inefficiency of one-shot policy updates.
2. **Theoretical Insights for Each Stage:** Provides compact mathematical intuition showing that the fast trajectory acts as a curvature-aware low-pass filter, the reposition step is equivalent to solving a linearized proximal subproblem acting as an implicit trust-region, and the slow correction provides a predictor-corrector structure.
3. **Preservation of Objective and Rollout Process:** SFPO introduces no changes to the underlying objective, rollout generation, or regularization, making it orthogonal to existing GRPO variants and enabling drop-in integration.
4. **Adaptive Entropy-Triggered Scheduling of $\alpha$:** Introduces an adaptive rule for the interpolation factor $\alpha$ based on policy entropy fluctuations (z-score). Sharp entropy fluctuations signal proximity to a local optimum where noise dominates, so the schedule exploits the fast trajectory in the high-signal early phase and reverts to pure on-policy updates near convergence.
5. **Empirical Efficacy and Efficiency:** Extensive experiments demonstrate consistent improvements in stability, sample efficiency, and convergence speed, outperforming GRPO by up to 2.80 points in average accuracy on math reasoning benchmarks, using up to 4.93× fewer rollouts and an up to 4.19× reduction in wall-clock time.

## 5. Overview of the Overall Technical Solution

The overall technical solution restructures each training step into three stages: 
- **Stage I (Fast Trajectory):** Starting from the current policy $\pi_{\theta_{s,0}}$, generate rollouts for training. Apply K successive gradient updates on the same batch to obtain $\theta_{s,K}$. This stage stabilizes the search direction by capturing the cumulative effect of K sequential corrections.
- **Stage II (Reposition):** Interpolate between $\theta_{s,K}$ and the starting point $\theta_{s,0}$ to form $\tilde{\theta}_{s,K}$, controlling off-policy drift. This stage regulates how much of the stabilized fast trajectory is exploited versus staying close to the on-policy point.
- **Stage III (Slow Correction):** Perform one additional update on $\tilde{\theta}_{s,K}$, yielding $\pi_{\theta_{s+1,0}}$ for the next iteration. This stage applies a curvature-aware adjustment that realigns updates with the correct optimization trajectory.
The solution also integrates an adaptive scheduling mechanism for $\alpha$ based on the entropy of the policy, ensuring stability near convergence by disabling interpolation when entropy fluctuations indicate noise dominance.

## 6. Detailed Module Design

### 6.1 Stage I: Fast Trajectory
Starting from parameters $\theta_{s,0}$ at the beginning of iteration $s$, SFPO executes a short fast trajectory of $K$ inner updates on the same batch of rollouts. Unlike one-shot updates that rely on a single noisy gradient, the displacement captures the cumulative effect of $K$ sequential corrections. Even if some individual steps are perturbed by noise, their composition tends to damp idiosyncratic fluctuations and align with the underlying gradient direction. Geometrically, this can be viewed as integrating the local gradient field along a short trajectory in parameter space rather than trusting a single noisy vector at $\theta_{s,0}$. 

**Compact Mathematical Intuition:** Let $\theta_k := \theta_{s,k}$ and consider the $K$ inner steps starting at $\theta_0 = \theta_{s,0}$. In a small neighborhood, linearizing the gradient field, $\nabla L(\theta) \approx g_0 + H_0(\theta - \theta_0)$ with $g_0 = \nabla L(\theta_0)$ and $H_0 = \nabla^2 L(\theta_0)$, yields the closed-form displacement. Along an eigen-direction with curvature $\lambda \geq 0$, the scalar gain behaves like $K\eta$ for small $\lambda$ (steadily accumulating progress in gentle directions), while for larger $\lambda$ it saturates below 1 (damping stiff-direction oscillations). Thus the fast trajectory acts as a curvature-aware low-pass filter that stabilizes the endpoint direction relative to a one-shot step.

### 6.2 Stage II: Reposition
While the fast trajectory improves stability, it changes the update from on-policy to off-policy. Since all inner steps reuse the same rollouts generated at $\theta_{s,0}$, the endpoint $\theta_{s,K}$ no longer corresponds to the distribution that produced those samples. SFPO introduces a reposition step that interpolates the fast trajectory back toward its starting point. Here $\alpha$ regulates the degree of off-policy drift: smaller values keep the update close to the original on-policy iterate, while larger values rely more on the fast trajectory at the risk of greater mismatch. 

**Intuition:** The interpolation is equivalent to solving a linearized proximal subproblem around $\theta_{s,0}$. Thus, $\alpha$ acts as an implicit trust-region radius: smaller $\alpha$ implies a larger proximal weight $\lambda$, enforcing stronger contraction toward the on-policy point.

### 6.3 Stage III: Slow Correction
After repositioning, SFPO applies one more (slow) correction step at the interpolated point. This yields a predictor—corrector structure: Stage I produces a stabilized fast trajectory, Stage II tempers off-policy drift via reposition, and Stage III applies a slow correction aligned with the local curvature at the update point. 

**Theoretical intuition:** Under L-smoothness and sufficiently small $\eta$, the descent lemma implies a bound where $F(K,\alpha)$ represents the combined effect of Stage I (fast trajectory length $K$) and Stage II (reposition factor $\alpha$). Intuitively, $F(K,\alpha)$ reflects a balance between exploiting more gradient information and controlling distributional drift: increasing $K$ leverages the same rollout data across multiple steps but also amplifies off-policy mismatch, while larger $\alpha$ interpolates more aggressively toward the fast trajectory at the risk of greater instability.

### 6.4 Scheduling $\alpha$
A nonzero $\alpha$ is essential to exploit the stabilized fast trajectory when $K > 0$. However, the same aggressiveness that helps early can be counter-productive near a minimizer. When $\|\nabla L\|$ is large, a nonzero $\alpha$ accelerates progress; but as we approach a minimum, the signal weakens while curvature and stochastic noise dominate, so a large $\alpha$ amplifies drift and instability. This motivates an adaptive schedule for $\alpha$. $\alpha$ is scheduled online at each iteration. After generating rollouts with the current policy, we compute the policy entropy $H_s$ and maintain a rolling buffer of the past $\omega$ entropy values. Let $\mu_s$ and $\sigma_s$ denote the mean and standard deviation within this buffer. We define the one-sided z-score. If $|Z_s| \geq \tau$ for a threshold $\tau$, we mark the current step $s^\star$ and set $\alpha = 0$ for all $s \geq s^\star$. Otherwise we keep $\alpha = \alpha_0$. Intuitively, sharp entropy fluctuations signal that the policy is close to a local optimum where noise dominates, so interpolation should be disabled.

### 6.5 Informal Definition and Analysis of $F(K,\alpha)$
We adopt a local quadratic-with-noise model around $\theta_{s,0}$: $L(\theta) \approx \frac{1}{2} (\theta - \theta^\star)^\top H (\theta - \theta^\star)$ with $H \succeq 0$, $g(\theta) = \nabla L(\theta) = H(\theta - \theta^\star)$, and $e_0 = \theta_{s,0} - \theta^\star$. Stage I uses stochastic gradients $g_k = g(\theta_{s,k}) + \xi_k$ where $\mathbb{E}[\xi_k] = 0$, $\mathbb{E}\|\xi_k\|^2 \leq \sigma_f^2$, and $\text{Cov}(\xi_k, \xi_\ell)$ has correlation coefficients $\rho^{|k-\ell|} \in [0, 1]$. The Stage III query at $\tilde{\theta}_{s,K}$ has noise $\tilde{\xi}$ with $\mathbb{E}[\tilde{\xi}] = 0$, $\mathbb{E}\|\tilde{\xi}\|^2 \leq \sigma_s^2$.

**Deterministic fast trajectory:** In the quadratic model, Stage I obeys $e_{k+1} = (I - \eta H)e_k$, hence $e_K = (I - \eta H)^K e_0$ and the drift magnitude $\text{Drift}(K,\alpha) \propto \alpha^2 \|\Delta_K\|_H^2 = \alpha^2 \Delta_K^\top H \Delta_K = \alpha^2 g_0^\top H^\dagger [I - (I - \eta H)^K]^2 g_0$. Spectrally, for an eigenpair $(u_\lambda, \lambda > 0)$, $\text{Drift}_\lambda(K,\alpha) = \alpha^2 \frac{(1-(1-\eta\lambda)^K)^2}{\lambda} \langle g_0, u_\lambda \rangle^2$, which increases monotonically with $K$ and is quadratic in $\alpha$, and saturates as $K \to \infty$.

**Stochastic components:** The fast trajectory accumulates noise $\alpha \sum_{k=0}^{K-1} \xi_k$ with $\mathbb{E} \|\alpha \sum_{k=0}^{K-1} \xi_k\|^2 \leq \alpha^2 \sigma_f^2 S(K, \rho)$, $S(K, \rho) = K + 2 \sum_{\Delta=1}^{K-1} (K - \Delta) \rho^\Delta \in [K, K^2]$, and the slow query contributes $\mathbb{E}\|\tilde{\xi}\|^2 \leq \sigma_s^2$. We informally define $F(K,\alpha)$ as the sum of off-policy drift (bias), fast noise, and slow noise. For fixed $\alpha$, both drift and $S(K, \rho)$ are non-decreasing in $K$, so $F(K,\alpha)$ increases with $K$. For fixed $K$, $F(K,\alpha)$ grows quadratically in $\alpha$.

## 7. All Mathematical Formulas and Symbol Definitions

**GRPO Normalized Advantage:**
$$ \hat{A}_i = \frac{r_i - \text{mean}(\{r_i\}_{i=1}^G)}{\text{std}(\{r_i\}_{i=1}^G)} $$

**GRPO Training Objective:**
$$ J_{GRPO}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(O|q)} \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min(r_{i,t}(\theta)\hat{A}_{i,t}, \text{clip}(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_{i,t}) - \beta D_{KL}[\pi_\theta \| \pi_{ref}] $$

**Token-level Importance Ratio:**
$$ r_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|q, o_{i,<t})} $$

**Standard On-Policy Update:**
$$ \theta_{s+1} = \theta_s - \eta \nabla_\theta L(\theta_s) $$

**Stage I: Fast Trajectory Update:**
$$ \theta_{s,k+1} = \theta_{s,k} - \eta \nabla_\theta L(\theta_{s,k}), \quad k = 0, \ldots, K-1 $$

**Displacement of Fast Trajectory:**
$$ \theta_{s,K} - \theta_{s,0} = -\eta \sum_{k=0}^{K-1} \nabla_\theta L(\theta_{s,k}) $$

**Closed-form Displacement (Linearized Gradient Field):**
$$ \theta_K - \theta_0 \approx -[I - (I - \eta H_0)^K] H_0^\dagger g_0 $$

**Stage II: Reposition Interpolation:**
$$ \tilde{\theta}_{s,K} = \theta_{s,0} + \alpha(\theta_{s,K} - \theta_{s,0}), \quad \alpha \in [0, 1] $$

**Linearized Proximal Subproblem:**
$$ \min_\theta \langle \sum_{k=0}^{K-1} \nabla_\theta L(\theta_{s,k}), \theta - \theta_{s,0} \rangle + \frac{\lambda}{2} \|\theta - \theta_{s,0}\|^2 $$
where $\lambda = \frac{1}{\alpha \eta}$.

**Stage III: Slow Correction Update:**
$$ \theta_{s+1} = \tilde{\theta}_{s,K} - \eta \nabla_\theta L(\tilde{\theta}_{s,K}) $$

**Descent Lemma Intuition:**
$$ \mathbb{E}[L(\theta_{s+1})] \leq L(\theta_{s,0}) - c \eta \|\nabla L(\theta_{s,0})\|^2 + O(\eta^2 L \cdot F(K,\alpha)) $$

**Entropy Z-score:**
$$ Z_s = \frac{H_s - \mu_s}{\sigma_s} $$

**Unified SFPO Update:**
$$ \theta_{s+1} = \theta_{s,0} - \eta \left[ \alpha \sum_{k=0}^{K-1} \nabla_\theta L(\theta_{s,k}) + \nabla_\theta L(\tilde{\theta}_{s,K}) \right] $$

**Deterministic Fast Trajectory Displacement (Quadratic Model):**
$$ \Delta_K \stackrel{\text{def}}{=} \theta_{s,K} - \theta_{s,0} = e_K - e_0 = -[I - (I - \eta H)^K] H^\dagger g_0 $$

**Drift Magnitude:**
$$ \text{Drift}(K,\alpha) \propto \alpha^2 \|\Delta_K\|_H^2 = \alpha^2 \Delta_K^\top H \Delta_K = \alpha^2 g_0^\top H^\dagger [I - (I - \eta H)^K]^2 g_0 $$

**Spectral Drift:**
$$ \text{Drift}_\lambda(K,\alpha) = \alpha^2 \frac{(1-(1-\eta\lambda)^K)^2}{\lambda} \langle g_0, u_\lambda \rangle^2 $$

**Stochastic Noise Accumulation:**
$$ \mathbb{E} \left\| \alpha \sum_{k=0}^{K-1} \xi_k \right\|^2 \leq \alpha^2 \sigma_f^2 S(K, \rho), \quad S(K, \rho) = K + 2 \sum_{\Delta=1}^{K-1} (K - \Delta) \rho^\Delta \in [K, K^2] $$

**Informal Definition of $F(K,\alpha)$:**
$$ F(K,\alpha) \approx \underbrace{\alpha^2 g_0^\top H^\dagger [I - (I - \eta H)^K]^2 g_0}_{\text{off-policy drift (bias)}} + \underbrace{\alpha^2 \sigma_f^2 S(K, \rho)}_{\text{fast noise}} + \underbrace{\sigma_s^2}_{\text{slow noise}} $$

## 8. Algorithm Pseudocode

**Algorithm 1 SFPO: unified fast–reposition–slow update.**
Require: Initial policy $\pi_{\theta_{0,0}}$, dataset $\mathcal{D}$, hyperparameters $S, K, \alpha_0, \eta, \omega, \tau$, loss $L(\theta)$
Initialize rolling buffer $\mathcal{H} \leftarrow \emptyset$, stats $(\mu, \sigma) \leftarrow (0, 1)$, trigger index $s^\star \leftarrow +\infty$
for $s = 0, 1, \ldots, S - 1$ do
&emsp;Generate rollouts with the current policy $\pi_{\theta_{s,0}}$ on prompts from $\mathcal{D}$.
&emsp;$\alpha \leftarrow \alpha_0 \cdot \mathbb{1}[ s < s^\star ]$ // Set $\alpha$ for this iteration from past trigger (non-anticipatory)
&emsp;if $\alpha = 0$ then
&emsp;&emsp;$\tilde{\theta}_{s,K} \leftarrow \theta_{s,0}$ // Skip fast trajectory & reposition
&emsp;else
&emsp;&emsp;for $k = 0, 1, \ldots, K - 1$ do
&emsp;&emsp;&emsp;$\theta_{s,k+1} \leftarrow \theta_{s,k} - \eta \nabla_\theta L(\theta_{s,k})$ // Stage I: Fast Trajectory
&emsp;&emsp;end for
&emsp;&emsp;$\tilde{\theta}_{s,K} \leftarrow \theta_{s,0} + \alpha (\theta_{s,K} - \theta_{s,0})$ // Stage II: Reposition
&emsp;end if
&emsp;$\theta_{s+1,0} \leftarrow \tilde{\theta}_{s,K} - \eta \nabla_\theta L(\tilde{\theta}_{s,K})$ // Stage III: Slow Correction
&emsp;Compute entropy $H_s$; update rolling buffer $\mathcal{H}$ (keep last $\omega$ ones) and $(\mu_s, \sigma_s)$.
&emsp;$Z_s \leftarrow \frac{H_s - \mu_s}{\sigma_s + \varepsilon}$ ($\varepsilon$ for numerical stability)
&emsp;if $s^\star = +\infty$ and $|Z_s| \geq \tau$ then
&emsp;&emsp;$s^\star \leftarrow s + 1$ // will set $\alpha = 0$ for all future $s' \geq s^\star$
&emsp;end if
end for
return final policy $\pi_{\theta_{S,0}}$