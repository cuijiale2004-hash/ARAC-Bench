### 1. Research Background and Existing Pain Points

In continuous-action reinforcement learning (RL), particularly within offline and offline-to-online settings, there exists a long-standing tension between policy expressivity and optimization tractability with respect to a parameterized Q-function (critic). 

Simple policies, such as single-step Gaussian policies, offer high optimization tractability because they can directly exploit the critic’s action gradient (i.e., $\nabla_a Q(s, a)$) via the reparameterization trick. However, this tractability comes at the cost of expressivity, limiting the ability to represent complex, multi-modal action distributions. Conversely, highly expressive policy classes like flow or diffusion policies generate actions through a multi-step denoising process, allowing them to represent complex behavior distributions. Exploiting the critic's action gradient for these policies requires backpropagation through the entire denoising process, which is numerically unstable. 

Existing methods work around this issue in three suboptimal ways:
1.  **Discarding Action Gradients**: Methods like FAWAC only use the critic's value and discard the gradient information, sacrificing learning efficiency.
2.  **Policy Distillation**: Methods like FQL distill expressive, multi-step flow policies into one-step noise-conditioned approximations to stabilize optimization, thereby compromising policy expressivity.
3.  **Intermediate Fine-tuning Approximations**: Methods like classifier guidance apply the critic’s action gradient directly to intermediate noisy actions. This relies on the assumption that the critic's gradient at a noisy action is a good proxy for its gradient at the denoised action. This assumption often breaks down in offline RL where limited action coverage means the critic is well-trained only on a narrow distribution of noiseless actions, rendering gradients for noisy actions unreliable.

### 2. Core Research Motivation and Scientific Questions

**Core Motivation**: The central motivation is to resolve the expressivity-tractability trade-off. Specifically, the paper asks: *Can we somehow keep the full expressivity of flow policies while incorporating the critic’s action gradient directly into the denoising process without backpropagation instability?*

**Scientific Questions**:
- How can we optimize a flow-matching policy against a critic such that the loss landscape avoids the instability inherent in direct backpropagation through time (BPTT)?
- How can we transform the critic's gradient evaluated at noiseless actions into unbiased gradient estimates for the intermediate denoising steps of a flow policy?
- Can we construct an optimization objective that guarantees the policy converges to the optimal behavior-constrained distribution ($\pi \propto \pi_\beta e^{\tau Q}$) at its optimum, without requiring approximations that bias the learned policy?

### 3. Overall Core Idea and Design Philosophy

The core idea is **Q-learning with Adjoint Matching (QAM)**, which leverages "adjoint matching," a technique recently developed in generative modeling, to perform policy extraction. 

The design philosophy centers on reformulating the policy extraction problem as a Stochastic Optimal Control (SOC) objective. The target optimal policy is defined as a tilt distribution of the behavior prior: $\pi_\theta(a|s) \propto \pi_\beta(a|s)e^{\tau Q(s,a)}$. 

Instead of solving the SOC objective using the continuous adjoint method (which has the same loss landscape as unstable BPTT), QAM leverages a modified objective admitting the same optimal solution but circumventing the instability challenge. The key insight is to use the **"lean" adjoint state**. In standard BPTT or basic adjoint matching, the adjoint state depends on the policy model $f_\theta$ being optimized, allowing ill-conditioned gradients in $f_\theta$ to compound and destabilize training. In QAM, the lean adjoint state is computed strictly using the base behavior velocity field $f_\beta$, independent of the ill-conditioned $f_\theta$. The critic’s gradient at noiseless actions is transformed via a reverse ODE constructed from $f_\beta$ to form a step-wise squared loss objective for $f_\theta$. This ensures stable optimization and preserves the full expressivity of multi-step flow models while guaranteeing convergence to the target distribution.

### 4. Core Innovation Points

1.  **Novel Application of Adjoint Matching to RL**: The paper introduces adjoint matching into the RL domain as a policy extraction mechanism, allowing the direct alignment of a flow policy with the optimal prior-regularized policy without suffering from BPTT instability.
2.  **Lean Adjoint State for Stability**: By deriving and utilizing the "lean" adjoint state—which removes terms zero at the optimum from the original adjoint state—QAM computes adjoint states using the base flow model $f_\beta$ rather than the target policy $f_\theta$. This prevents the compounding of ill-conditioned gradients during backpropagation, solving the core instability issue.
3.  **Unbiased and Expressive Optimization**: Unlike classifier guidance which relies on noisy-action gradients (unreliable out-of-distribution), QAM transforms the critic's gradient at the final noiseless action into a step-wise objective. This provides an unbiased, step-wise supervision signal for the intermediate denoising steps while retaining the full expressivity of multi-step flow models.
4.  **Theoretical Guarantee**: The paper extends theoretical results from adjoint matching to prove that as long as the proposed objective is optimized to convergence, the learned policy coincides exactly with the optimal behavior-constrained policy ($\pi^\star(a|s) \propto \pi_\beta(a|s)\exp(\tau Q(s,a))$).
5.  **Constraint Relaxation Variants**: To handle support mismatch between optimal and behavior actions, QAM introduces two variants—QAM-FQL and QAM-Edit—that expand the constraint set beyond KL divergence to include Wasserstein distance constraints, allowing actions close to the behavior distribution but not strictly within its support.

### 5. Overview of the Overall Technical Solution

The QAM algorithm consists of three main learning components operating concurrently:
1.  **Behavior Policy Learning**: A flow-matching behavior policy $f_\beta$ is trained via the standard flow-matching objective to model the offline dataset distribution.
2.  **Critic Learning**: A critic $Q_\phi$ is trained using temporal-difference (TD) backup with an ensemble of networks and pessimistic target value backup to mitigate overestimation.
3.  **Policy Extraction via Adjoint Matching**: The fine-tuned policy $f_\theta$ is optimized using the adjoint matching objective. This involves sampling a trajectory from a "memory-less" SDE, computing the lean adjoint states via a reverse ODE using $f_\beta$ and the critic's gradient at $t=1$, and minimizing the squared difference between $f_\theta$ and the target velocity defined by $f_\beta$ plus the lean adjoint term.

For handling out-of-distribution actions, the algorithm offers two extensions:
-   **QAM-FQL**: Learns a 1-step noise-conditioned policy under a $W_2$ Wasserstein constraint relative to the QAM-fine-tuned flow policy.
-   **QAM-Edit**: Learns an edit policy under a $W_1$ Wasserstein constraint ($L_\infty$ norm) to modify actions sampled from the QAM flow policy.

### 6. Detailed Module Design

**Module 1: Flow-Matching Generative Model**
The flow model approximates intermediate states $X_t$ via an ODE starting from noise $X_0 \sim \mathcal{N}$. It uses a time-variant velocity field $v: \mathbb{R}^d \times [0,1] \to \mathbb{R}^d$. It defines a family of marginal-preserving SDEs admitting the same marginals as the ODE, utilizing a "memory-less" noise schedule where $X_0$ and $X_1$ are independent, specifically $\sigma_t = \sqrt{2(1-t)/t}$.

**Module 2: Adjoint Matching Objective Derivation**
The goal is to modify $f_\beta$ to generate the tilt distribution $p_\theta(X_1) \propto p_\beta(X_1)e^{Q(X_1)}$. The standard SOC objective involves backpropagation through time. To circumvent this, the "basic" adjoint matching objective is derived using the adjoint state $g$. The paper further refines this into the "lean" adjoint state $\tilde{g}$, which satisfies an ODE dependent only on $f_\beta$. The lean adjoint matching objective matches $f_\theta$ to $f_\beta$ plus a correction term involving $\tilde{g}$.

**Module 3: QAM Policy Optimization**
The QAM algorithm applies the lean adjoint matching to the RL setting.
-   **Forward SDE Sampling**: A trajectory $\{a_t\}$ is sampled using the memory-less SDE discretized with step size $h = 1/T$.
-   **Lean Adjoint Calculation**: The boundary condition is set as the critic's action gradient $\tilde{g}_1 = -\tau \nabla_{a_1} Q_\phi(s, a_1)$. The adjoint states are propagated backward using the behavior model $f_\beta$ via Vector-Jacobian Products (VJPs).
-   **Loss Computation**: The objective minimizes the squared difference between the fine-tuned velocity field and the target velocity field defined by the behavior policy and the adjoint state.

**Module 4: Constraint Relaxation (QAM-Edit & QAM-FQL)**
-   **QAM-FQL**: Learns a 1-step policy $\mu_\omega(s, z)$ to maximize $Q$ while staying close to the QAM flow policy under $W_2$ distance, implemented as a BC-regularized loss.
-   **QAM-Edit**: Learns a Gaussian edit policy $\pi_\omega(\Delta a | s, a)$ constrained by $W_1$ distance ($L_\infty$ norm $\|\Delta a\|_\infty \le \sigma_a$) using entropy regularization.

**Module 5: Network Architecture and Training Details**
-   **Critic Network**: An ensemble of $K=10$ critic networks. Uses pessimistic target value backup ($Q_{mean} - \rho Q_{std}$).
-   **Flow Policy**: Network width 512, depth 4 hidden layers. Gradient clipping of magnitude 1 is applied element-wise.
-   **Optimization**: Adam optimizer with learning rate $3 \times 10^{-4}$. Target network update rate $\lambda = 0.005$.

### 7. All Mathematical Formulas and Symbol Definitions

**MDP Formulation**:
$M = (\mathcal{S}, \mathcal{A}, P, \gamma, R, \mu)$
Discounted return: $U^\pi = \mathbb{E}_{s_0 \sim \mu, s_{k+1} \sim P(\cdot|s_k, a_k), a_k \sim \pi(\cdot|s_k)} [\sum_{k=0}^\infty \gamma^k R(s_k, a_k)]$

**Flow Matching**:
Intermediate state: $X_t = (1-t)X_0 + tX_1$
ODE: $d\hat{X}_t = f(\hat{X}_t, t)dt$
Flow matching loss: $L_{FM}(\theta) = \mathbb{E}_{t \sim \mathcal{U}[0,1], x_0 \sim \mathcal{N}, x_1 \sim \mathcal{D}} \left[ \| f_\theta((1-t)x_0 + tx_1, t) - x_1 + x_0 \|_2^2 \right]$

**Memory-less SDE**:
$dX_t = (2f(X_t, t) - X_t/t) dt + \sqrt{2(1-t)/t} dB_t$

**Adjoint State Definitions**:
Boundary condition: $g(X, t=1) = -\nabla_{X_1} Q(X_1)$
Lean adjoint ODE: $d\tilde{g}(X, t) = -\nabla_{X_t} [2f_\beta(X_t, t) - X_t/t] \tilde{g}(X, t) dt$

**Adjoint Matching Objective (Generative Modeling)**:
$L_{AM}(\theta) = \mathbb{E}_X \left[ \int_0^1 \| 2(f_\theta(X_t, t) - f_\beta(X_t, t))/\sigma_t + \sigma_t \tilde{g}(X, t) \|_2^2 dt \right]$

**QAM Policy Optimization**:
Optimal policy: $\pi^\star(a | s) \propto \pi_\beta(a | s) e^{\tau(s) Q_\phi(s, a)}$
Behavior loss: $L_{FM}(\beta) = \mathbb{E}_{(s,a)\sim D, t\sim[0,1], z\sim\mathcal{N}} \left[ \| f_\beta(s, (1-t)z + ta, t) - a + z \|_2^2 \right]$
QAM Loss: $L_{AM}(\theta) = \mathbb{E}_{s \sim D, \{a_t\}_t} \left[ \int_0^1 \| 2(f_\theta(s, a_t, t) - f_\beta(s, a_t, t))/\sigma_t + \sigma_t \tilde{g}_t \|_2^2 dt \right]$
Lean adjoint ODE (QAM): $d\tilde{g}_t = -\nabla_{a_t} [2f_\beta(s, a_t, t) - a_t/t] \tilde{g}_t dt$

**Discrete Approximations**:
Forward SDE: $a_{t+h} \leftarrow a_t + h \cdot (2f_\theta(s, a_t, t) - a_t/t) + \sqrt{2h(1-t)/t} z_t$
Backward Adjoint: $\tilde{g}_{t-h} \leftarrow \tilde{g}_t + h \cdot \text{VJP}(\nabla_{a_t}(2f_\beta(s, a_t, t) - a_t/t), \tilde{g}_t)$

**Critic Learning**:
$L(\phi_j) = \left( Q_{\phi_j}(s, a) - r - \gamma [ \bar{Q}_{mean}(s', a') - \rho \bar{Q}_{std}(s', a') ] \right)^2$
where $\bar{Q}_{mean}(s', a') := \frac{1}{K}\sum_k Q_{\bar{\phi}_k}(s', a')$ and $\bar{Q}_{std}(s', a') = \frac{1}{K}\sqrt{\sum_k (Q_{\bar{\phi}_k}(s', a') - \bar{Q}_{mean}(s', a'))^2}$

**QAM-FQL Loss**:
$L_{QAM-FQL}(\omega) = \mathbb{E}_{z \sim \mathcal{N}} \left[ -Q_\phi(s, \mu_\omega(s, z)) + \alpha \| \mu_\omega(s, z) - ODE(f_\theta(s, \cdot, \cdot), z) \|_2^2 \right]$

**QAM-Edit Loss**:
$L_{QAM-EDIT}(\omega) = \mathbb{E}_{\Delta a \sim \pi_\omega(\cdot|s,a)} \left[ -Q_\phi(s, \Delta a + a) + \eta \log \pi_\omega(\Delta a | s, a) \right]$
$L(\eta) = \eta \left[ \mathbb{E}_{\Delta a \sim \pi_\omega(\cdot|s,a)} [\log \pi_\omega(\Delta a | s, a)] - H_{target} \right]$

### 8. Algorithm Pseudocode

**Algorithm 1 Learning procedure in QAM.**
**Input:** $(s, a, s', r)$: off-policy transition tuple, $f_\beta$: behavior velocity field, $f_\theta$: fine-tuned velocity field, $Q_\phi$: critic function.

1: Optimize $\phi$ w.r.t $L(\phi) = \left[ Q_\phi(s, a) - r - \gamma Q_{\bar{\phi}}(s', a' \sim \pi_\theta(\cdot | s')) \right]^2$ $\triangleright$ TD backup
2: $\mathbf{a} = \{a_0, a_h, \cdots, a_1\} \leftarrow \text{SDE}(f_\theta(s, \cdot, \cdot))$ $\triangleright$ Memory-less SDE; Equation (24)
3: $\tilde{g}_1 \leftarrow -\tau \nabla_{a_1} Q_\phi(s, a_1)$ $\triangleright$ Compute the critic’s action gradient
4: $\tilde{g}_0, \tilde{g}_h, \cdots, \tilde{g}_{1-h} \leftarrow \text{LeanAdj}(f_\beta(s, \cdot, \cdot), \tilde{g}_1, \mathbf{a})$ $\triangleright$ Lean adjoint states; Equation (25)
5: Optimize $\theta$ w.r.t $L(\theta) = \sum_t \left\| \frac{2(f_\theta(s, a_t, t) - f_\beta(s, a_t, t))}{\sigma_t} + \sigma_t \tilde{g}_t \right\|_2^2$ $\triangleright$ Adjoint matching; Equation (21)

**Output:** $f_\theta, Q_\phi$