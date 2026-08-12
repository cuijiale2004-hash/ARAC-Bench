1. **Research Background and Existing Pain Points**
Diffusion models have demonstrated remarkable empirical success in recent years and are considered one of the state-of-the-art generative models in modern AI. These models consist of a forward process, which gradually diffuses the data distribution to a noise distribution spanning the whole space, and a backward process, which inverts this transformation to recover the data distribution from noise. Most of the existing literature assumes that the underlying space is Euclidean. However, in many practical applications, the data are constrained to lie on a submanifold of Euclidean space. Many scientific domains are intrinsically non-Euclidean; examples include orientations on SO(3), directions on spheres, toroidal angles, articulated poses, and symmetric positive definite (SPD) matrices, which are naturally modeled on Riemannian manifolds. 

While De Bortoli et al. (2022) introduced Riemannian Score-Based Generative Models (RSGMs) with convergence guarantees in the Wasserstein distance, their convergence bound suffers from critical pain points: 1) It requires an exponentially small stepsize, leading to a possibly exponential iteration complexity in some of the manifold parameters; 2) It requires $L_\infty$-accurate score estimates, which are impractical in deep learning; and 3) The data distribution is required to be smooth and strictly positive on compact manifolds.

2. **Core Research Motivation and Scientific Questions**
The core research motivation is to strengthen the theory of Riemannian diffusion models, making them theoretically efficient and practically applicable by removing strong assumptions on score estimates and data distributions. The central scientific question raised is: "Can we achieve polynomial iteration complexity for manifold diffusion models using $L_2$-accurate score estimates under mild data assumptions?"

3. **Overall Core Idea and Design Philosophy**
The overall core idea is to provide a discrete-time analysis of the RSGM sampler assuming $L_2$-accurate score estimates, establishing that polynomial stepsizes suffice for accurate sampling on manifolds in the total variation (TV) distance. The design philosophy involves decomposing the total variation error between the true data distribution and the algorithm output into four distinct error components: initialization error, score error, drift discretization error, and Brownian motion (BM) simulation error. To handle the drift discretization error without positivity/smoothness assumptions, the design employs Li-Yau gradient bounds with early stopping. To address the BM simulation error—a unique challenge in manifold settings where the transition kernel cannot be simulated exactly by discrete-time processes—the design utilizes Minakshisundaram-Pleijel parametrix expansion and a localization scheme to separate the effects of discretizing scores and Brownian motion.

4. **Core Innovation Points**
1. **Polynomial Convergence Complexity**: The paper establishes that a polynomial stepsize is sufficient for accurate sampling, achieving a polynomial iteration complexity $N \asymp \frac{\text{poly}(d, K, \delta^{-1})}{(\lambda_1 \varepsilon)^2}$, which avoids the exponential blow-up in dimension or curvature present in prior manifold work.
2. **Relaxed Score Estimation Assumptions**: The convergence guarantee relies solely on $L_2$-accurate score estimates rather than the impractical $L_\infty$-accurate score estimates.
3. **Relaxed Data Distribution Assumptions**: The theory does not require the data distribution to be smooth or strictly positive, utilizing early stopping time $\delta > 0$ to approximate $p_0$.
4. **Novel Analytical Tools for Manifolds**: 
   - High-probability Li-Yau gradient bounds for the manifold heat kernel together with early stopping to control $\|\nabla \log p_t\|$.
   - A localization scheme that "freezes" drifts across nearby tangent spaces but preserves continuous Brownian motion, separating the effects of discretizing scores and BM.
   - Quantitative estimates for the Minakshisundaram–Pleijel parametrix that control one-step deviations between the manifold heat flow and its discretized proxy.

5. **Overview of the Overall Technical Solution**
The technical solution is based on a driftless forward process defined by geometric Brownian motion on a compact Riemannian manifold $M$, followed by a reverse-time SDE driven by the score function. The algorithm discretizes the reverse SDE using a geometric Euler–Maruyama scheme with an estimated score, projecting updates onto the manifold via the exponential map and employing rejection sampling to remain within the injectivity radius. The theoretical analysis bounds the TV error by decomposing it into:
- **Initialization error**: Bounded using the mixing rate of the heat flow and the spectral gap of the Laplace-Beltrami operator.
- **Score error**: Controlled using Girsanov's theorem and $L_2$ score estimation assumptions.
- **Drift discretization error**: Managed by introducing auxiliary kernels with localized frozen drifts, applying Itô/Stratonovich calculus to cancel third-order derivatives, and bounding remaining terms via Li-Yau estimates.
- **BM simulation error**: Controlled by comparing the manifold heat kernel to the Euclidean density using Minakshisundaram-Pleijel parametrix expansion and Volterra series representations, bounding the KL divergence, and handling exceptional exit events via stopping time arguments.

6. **Detailed Module Design**
**6.1 Forward and Reverse Process Module**
The forward process is a geometric Brownian motion solving $dX_t = U_{X_t} \circ dW_t$ with $X_0 \sim p_0$, where $U_x: \mathbb{R}^d \to T_x M$ is an orthonormal frame and $\circ$ denotes the Stratonovich integral. The marginal density $p_t$ satisfies the heat equation $\partial_t p_t(\cdot, y) = \frac{1}{2}\Delta_M p_t(\cdot, y)$. The reverse-time SDE is given by:
$dY_t = \nabla \log p_\tau(Y_\tau) dt + U_{Y_\tau} \circ dW_t, \quad \tau = T - t, \quad Y_T \sim p_T$.

**6.2 Discretization and Rejection Sampling Module**
The discrete-time sampler uses an estimated score $\hat{s}_t$. In each reverse step, it selects an orthonormal frame $U_k$, samples Gaussian noise $\xi_k \sim \mathcal{N}(0, I_d)$, and lifts it to the tangent space $G_k = U_k \xi_k$. A tangent update is proposed: $\Delta_k = h \hat{s}_{t_k}(Y_k) + \sqrt{h} G_k$. To ensure the update stays within the injectivity radius, a rejection sampling step is applied: if $\|\Delta_k\| \leq h^{1/4}$, the state is updated via the exponential map $Y_{k-1} = \exp_{Y_k}(\Delta_k)$; otherwise, $Y_{k-1}$ is drawn from the uniform distribution $\mu$.

**6.3 Initialization Error Module**
This module bounds the error arising from initializing with the uniform distribution $\mu$ instead of the true marginal $p_N$. Using the spectral gap $\lambda_1 > 0$ of $-\Delta_M$ and Li-Yau upper bounds, the mixing rate of the heat flow in total variation is bounded:
$\text{TV}(p_N, \mu) \leq \sqrt{A} e^{-\frac{1}{2}\lambda_1 (T - 1/2)}$, where $A = \sup_{t \leq 1} \sup_{x \in M} t^{d/2} H(t, x, x)$, leading to $\text{TV}(p_N, \mu) \leq e^{C(K + d \log(Kd))} e^{-\frac{1}{2}\lambda_1 (T - 1/2)}$.

**6.4 Drift Discretization Error Module**
To isolate this error, an auxiliary kernel $K_k^{aux}$ is constructed using a localized vector field:
$S_{t,x}(y) = (d \exp_x)_{\log_x y} (\eta_\omega(\rho(x, y)) \cdot \hat{s}_t(x)) \in T_y M$,
where $\eta_\omega$ is a smooth cutoff function. The auxiliary SDE is $dY_\tau = S_{t_k, Y_{t_k}}(Y_\tau) dt + U_{Y_\tau} \circ dW_t$. The difference between the true score and frozen score involves third-order derivatives. Using Itô's formula:
$d\nabla \log p_t(Y_t) = \frac{1}{2} (\Delta_M \nabla - \nabla \Delta_M) \log p_t d\tau + \nabla^2 \log p_t \cdot U_{Y_t} \circ dW_t$.
By Bochner's identity, $(\Delta_M \nabla - \nabla \Delta_M)f = \text{Ric}^\#(\nabla f, \cdot)$, which cancels the third-order derivatives. The remaining terms are bounded using Li-Yau estimates and stopping time arguments for exit probabilities, yielding a bound on the drift discretization error:
$\sum_{k=1}^N \int_{t_{k-1}}^{t_k} \mathbb{E} \|\nabla \log p_t(Y_t) - S^\star_{t_k, Y_{t_k}}(Y_t)\|^2 dt \leq \frac{C d^6 K^8}{\delta^3} h^2 N$.

**6.5 BM Simulation Error Module**
This module compares the manifold heat kernel $F_H$ (for operator $H = \frac{1}{2}\Delta_M + \langle S_{t_k, Y_{t_k}}, \nabla \rangle$) with the Euclidean density proxy $\Phi$. Using the Minakshisundaram-Pleijel parametrix expansion:
$F_H(t, x, y) = \Psi(t, x, y) \left(1 + \sum_{i=1}^\infty t^i u_i(x, y)\right)$,
where $\Psi(t, x, y) = G(t, x, y) \exp(\psi(x, y)) \sqrt{\Delta(x, y)}$ and $G(t, x, y) = (2\pi t)^{-d/2} \exp(-\frac{d^2(x, y)}{2t})$. The error is bounded using the Volterra series representation $F_H = \Psi_1 + \sum_{k=1}^\infty (-1)^k \Psi_1 * r_1^{*k}$, separating integrals into local and outlier parts, and utilizing a "freezing" trick to handle singular factors. Exceptional events (exiting the polynomially small neighborhood) are handled via rejection probability bounds $P(\text{rejection}) \leq \exp(-\frac{1}{16 h^{1/2}})$. The final BM simulation error is bounded as:
$\text{TV}(p_0^{aux}, q_0^\star) \leq \sqrt{h T \text{poly}(d, K, \delta^{-1})}$.

7. **All Mathematical Formulas and Symbol Definitions**
- $dX_t = U_{X_t} \circ dW_t, X_0 \sim p_0$ (Forward SDE)
- $\partial_t p_t(\cdot, y) = \frac{1}{2}\Delta_M p_t(\cdot, y)$ (Heat equation)
- $dY_t = \nabla \log p_\tau(Y_\tau) dt + U_{Y_\tau} \circ dW_t, \tau = T - t, Y_T \sim p_T$ (Reverse SDE)
- $s_t(x) := \nabla \log p_t(x)$ (Score function)
- $y_{k-1} = y_k + h\hat{s}_{t_k}(y_k) + \sqrt{h}g_k, g_k \sim \mathcal{N}(0, I_d)$ (Euclidean Discretization)
- $\text{TV}(p, q) = \int_M |\frac{dp}{d\mu} - \frac{dq}{d\mu}|$
- $\text{KL}(p \| q) = \int_M (\log \frac{dp}{dq}) dp$
- $\Delta_M f := \nabla_\alpha \nabla^\alpha f$ (Laplace-Beltrami operator)
- $\text{Diam}(M) := \sup_{x, y \in M} \rho(x, y)$
- **Assumption 1 (Regularity)**: Let $(M, g)$ be a connected, compact d-dimensional Riemannian manifold.
  - (A1) Positive injectivity radius: $\text{inj}(M) \geq 1/K$.
  - (A2) Uniform curvature bounds: $\max\{\text{Diam}(M), \|\text{Rm}\|_{L^\infty}, \|\nabla \text{Rm}\|_{L^\infty}, \|\nabla^2 \text{Rm}\|_{L^\infty}\} \leq K$.
  - (A3) Regularity of score estimates: $\|\hat{s}_{t_k}(x)\| \leq \text{poly}(d, K) (\|\nabla \log p_{t_k}(x)\| + t_k^{-1})$.
- **Assumption 2 (Score estimation error)**: $\sum_{k=1}^N (t_k - t_{k-1})\mathbb{E}\|\hat{s}_{t_k}(Y_{t_k}) - \nabla \log p_{t_k}(Y_{t_k})\|^2 \leq \varepsilon_{\text{score}}^2$.
- **Theorem 1**: Assume Assumptions 1 and 2 hold. $\exists C, C' > 0$ such that if $T \geq \frac{C}{\lambda_1}(d \log(Kd) + K + \log(N\varepsilon))$, then:
  $\text{TV}(p_\delta, \text{Law}(Y_0)) \leq \varepsilon + C'\varepsilon_{\text{score}} + \sqrt{h T \text{poly}(d, K, \delta^{-1})}$.
- $\omega := \frac{c_\omega}{K d^4}, \eta_\omega(r) = \eta(\frac{4r^2}{\omega^2}), r \geq 0$ (Localization cutoff)
- $S_{t,x}(y) = (d \exp_x)_{\log_x y} (\eta_\omega(\rho(x, y)) \cdot \hat{s}_t(x)) \in T_y M$ (Frozen drift vector field)
- $S^\star_{t,x}(y) = (d \exp_x)_{\log_x y} (\eta_\omega(\rho(x, y)) \cdot \nabla \log p_t(x)) \in T_y M$ (True score frozen field)
- $dY_\tau = S_{t_k, Y_{t_k}}(Y_\tau)dt + U_{Y_\tau} \circ dW_t, \tau = T - t$ (Auxiliary SDE)
- $\text{KL}(\text{Law}(Y) \| \text{Law}(Y^{aux})) \leq \sum_{k=1}^N \int_{t_{k-1}}^{t_k} \mathbb{E}\|\nabla \log p_t(Y_t) - S_{t_k, Y_{t_k}}(Y_t)\|^2 dt$
- $\partial_\tau \nabla \log p_t = -\frac{1}{2}\nabla (\frac{\Delta_M p_t}{p_t}) = -\frac{1}{2}\nabla \Delta_M \log p_t - \nabla^2 \log p_t \cdot \nabla \log p_t$
- $d\nabla \log p_t(Y_t) = \frac{1}{2}\text{Ric}^\#(\nabla \log p_t, \cdot)d\tau + \nabla^2 \log p_t \cdot U_{Y_t} \circ dW_t$ (Bochner's Identity application)
- $J(x, u) := |\det d \exp_x(u)| = \sqrt{\det g_{ij}(\exp_x u)}$ (Jacobian of exponential map)
- $\Delta(x, y) := \frac{|\det d \log_x y|^2}{\det g(y)}$ (Van Vleck-Morette determinant)
- $\Psi(t, x, y) = G(t, x, y) \exp(\psi(x, y)) \sqrt{\Delta(x, y)}$ (Parametrix approximation)
- $G(t, x, y) = (2\pi t)^{-d/2} \exp(-\frac{d^2(x, y)}{2t})$
- $\psi(x, y) := \int_0^1 \langle S_{t_k, Y_{t_k}}(\gamma(s)), \frac{d}{ds}\gamma(s) \rangle ds$
- $F_H = \Psi_1 + \sum_{k=1}^\infty (-1)^k \Psi_1 * r_1^{*k}$ (Volterra series)
- $\mathbb{P}(\text{rejection at step } k) \leq \exp(-\frac{1}{16 h^{1/2}})$

8. **Algorithm Pseudocode**
**Algorithm 1 Riemannian Score-Based Generative Models (RSGM)**
1: Manifold $(M, g)$; score $\hat{s}_t(x)$; early stopping time $\delta > 0$; reverse time grid $\delta = t_0 < t_1 < \cdots < t_N = T$; step size $h = t_k - t_{k-1}$; initial $x_N \sim \mu$ (uniform distribution);
2: for $k \in \{N, \ldots, 1, 0\}$ do
3:     Choose an orthonormal frame $U_k$ at $Y_k$, which is a linear map from $\mathbb{R}^d$ to $T_{Y_k}M$.
4:     $\xi_k \sim \mathcal{N}(0, I_d)$ in $\mathbb{R}^d$; $G_k \leftarrow U_k \xi_k \in T_{Y_k}M$.
5:     $b_k \leftarrow \hat{s}_{t_k}(Y_k) \in T_{Y_k}M$
6:     $\Delta_k \leftarrow h b_k + \sqrt{h} G_k \in T_{Y_k}M$
7:     if $\|\Delta_k\| \leq h^{1/4}$ then
8:         $Y_{k-1} \leftarrow \exp_{Y_k}(\Delta_k)$
9:     else
10:        $Y_{k-1} \sim \mu$
11: return $Y_0$