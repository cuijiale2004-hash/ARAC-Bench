### 1. Research Background and Existing Pain Points

**Research Background:**
Inverse problems (IPs) have wide applications in many domains, including computer vision, protein science, medical imaging, and scientific computing. The goal is to reconstruct an unknown $x \in \mathbb{R}^n$ from noisy measurements $y \in \mathbb{R}^m$:
$$y = A(x) + \eta$$
where $A$ is a known forward operator, and $\eta \in \mathbb{R}^m$ is additive noise. Diffusion models (DMs) have recently shown powerful capabilities in modeling complex data distributions, which can provide a powerful class of priors for high-dimensional data $x$ in solving IPs. Existing diffusion-based methods for IPs can be broadly classified into MAP inference (which aims to find the single most probable $x$) and the Bayesian framework (which samples from the posterior distribution $p(x|y)$). Most approaches are based on the Denoising Diffusion Probabilistic Models (DDPM) framework, which consists of forward and reverse diffusion processes. To accelerate denoising, Denoising Diffusion Implicit Models (DDIM) define a non-Markovian and fully deterministic forward/reverse process, mapping initial noise $x_T$ to a clean sample $x_0$ via a deterministic trajectory.

**Existing Pain Points:**
Current diffusion-based methods suffer from three complementary limitations and issues:
1.  **Iterative Guidance Methods (e.g., DPS, DDRM, DDNM, $\Pi$GDM, TMPD):** These methods use the likelihood term to shift intermediate images directly, systematically pushing intermediate states off the learned data manifold and violating the training-time noise-conditioning of the denoiser, resulting in accumulated artifacts.
2.  **Stochastic MAP Methods (e.g., ReSample, DiffPIR, DAPS, SITCOM, DIIP):** These methods optimize in the image space and can match $y$ well with very sharp details, but they require carefully tuned hyperparameters to not overfit to noise. This limits their effectiveness in high or unknown noise settings. For example, DAPS uses a heuristic $\hat{\sigma}_y$ that is much smaller than its true value, making it effectively MAP-like and inheriting sensitivity to noise.
3.  **Deterministic MAP Methods (e.g., DMPlug):** These methods optimize in the DM noise space, removing randomness, but often get stuck in a single mode, especially in severely ill-posed problems like phase retrieval, due to a lack of posterior exploration. Enforcing data consistency mid-diffusion can break prior adherence, while optimizing only for fidelity leads to overfitting or mode collapse.

### 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:**
To address the trilemma of manifold infeasibility, noise overfitting, and local-mode collapse in existing methods. Enforcing data consistency mid-diffusion breaks prior adherence, while optimizing only for fidelity leads to overfitting or mode collapse. There is a need for a method that performs comprehensive exploration of the solution space, avoiding local optima, while keeping proposals strictly on the learned data manifold. Furthermore, practical challenges require methods that handle IPs with unknown noise types and levels without task-specific hyperparameter tuning.

**Scientific Questions:**
1. How to perform posterior sampling for inverse problems that strictly maintains manifold feasibility (avoiding intermediate approximations)?
2. How to avoid overfitting to measurement noise in high-noise regimes using a principled regularization approach?
3. How to effectively handle unknown noise types and levels without relying on heuristic hyperparameters or early stopping criteria?
4. How to prevent the sampler from being trapped in early local modes in severely ill-posed problems?

### 3. Overall Core Idea and Design Philosophy

The core idea is to propose Noise-space Hamiltonian Monte Carlo (N-HMC), a posterior sampling method that treats reverse diffusion as a deterministic mapping from initial noise to clean images. By moving inference entirely into the initial-noise space $x_T$, N-HMC keeps proposals on the learned data manifold and enables comprehensive exploration of the solution space. The deterministic DDIM process is treated as a parameterized generator function $x_0 = D(x_T)$, transforming the problem into evaluating the posterior distribution of noise $p(x_T|y)$. Sampling from the noise space offers two advantages: (i) the prior $p(x_T)$ is a simple Gaussian distribution, and (ii) the likelihood $p(y|x_T) = p(y|D(x_T))$ is directly accessible without intermediate approximations. HMC is used for efficient posterior sampling in the noise space.

To handle unknown noise types and levels, the method is extended to a noise-adaptive variant (NA-NHMC). Instead of requiring a fixed noise level, a principled Bayesian approach is taken by placing a non-informative Jeffreys prior on the noise variance and marginalizing it out. This yields a parameter-free likelihood term that automatically adapts to the true underlying noise in the measurements. Additionally, an annealing schedule for noise standard deviation $\sigma_y$ is employed to promote early exploration and prevent local-mode collapse.

### 4. Core Innovation Points

1. **N-HMC Framework for Posterior Sampling:** Proposes N-HMC, a posterior sampling method that operates in the noise space using reverse diffusion as a deterministic mapping, addressing manifold infeasibility, noise overfitting, and local-mode collapse simultaneously.
2. **Noise-space Formulation with Manifold Feasibility:** Formally defines manifold feasibility (Definition 3.1) and proposes sampling from $p(x_T|y)$ instead of intermediate $p(x_t|y)$. This formulation treats the initial noise $x_T$ as the sole latent variable, allowing exact likelihood computation $p(y|x_T) = p(y|D(x_T))$ without intermediate approximations that push states off-manifold.
3. **Theoretical Robustness to Measurement Noise:** Provides a comprehensive theoretical analysis proving that the Gaussian prior in the noise space acts as a regularization, keeping the noise vector close to the hypersphere of radius $\sqrt{n}$. Proposition 1 and Corollary 1.1 formally establish that the residual magnitude aligns with the true measurement noise, indicating robustness without overfitting.
4. **Noise-Adaptive Extension (NA-NHMC):** Introduces a principled noise-adaptive variant by marginalizing the noise variance $\sigma_y^2$ with a Jeffreys prior. This results in an exact marginalized likelihood term $\propto (\frac{1}{2}\|y - A(D(x_T))\|^2)^{-m/2}$ that eliminates the need for task-specific hyperparameter tuning for unknown noise types and levels (Proposition 2).
5. **Annealing Schedule for Exploration:** Implements an annealing schedule for $\sigma_y$ during the initial HMC iterations to allow $x_T$ to explore the noise space freely with a larger step size, effectively preventing local-mode collapse in severely ill-posed tasks like phase retrieval.

### 5. Overview of the Overall Technical Solution

The overall technical solution formulates the inverse problem as a latent variable model where the initial noise $x_T$ is the sole latent variable. The unconditional DDIM process with $N$ steps is treated as a deterministic parameterized generator function $D : \mathbb{R}^n \to \mathbb{R}^n$, mapping $x_T$ to a clean image $\hat{x}_0 = D(x_T)$. The measurement generation process is defined by $x_T \xrightarrow{D} \hat{x}_0 \xrightarrow{A, \eta} y$. 

The posterior sampling is performed in the noise space using HMC. The conditional score is computed using Bayes' rule:
$$\nabla_{x_T} \log p(x_T|y) = \nabla_{x_T} \log p(x_T) + \nabla_{x_T} \log p(y|x_T)$$
Since $x_T \sim \mathcal{N}(0, I)$, the prior score is $-x_T$. The likelihood score is computed via automatic differentiation through the deterministic denoising trajectory. For known Gaussian noise, the likelihood is $\mathcal{N}(y; A(D(x_T)), \sigma_y^2 I)$. For unknown noise, the likelihood is marginalized using the Jeffreys prior. The HMC sampler uses leapfrog integration with Metropolis-Hastings correction. An annealing schedule for $\sigma_y$ is applied in the startup stage to encourage exploration and avoid tiny step sizes.

### 6. Detailed Module Design

**Module 1: Deterministic Decoder Module ($D(x_T)$)**
This module utilizes unconditional DDIM to map the initial noise $x_T \sim \mathcal{N}(0, I)$ to a clean image $\hat{x}_0$. The entire denoising trajectory is treated as a deterministic mapping without stochasticity. For computational efficiency, as few as two denoising steps can be used.

**Module 2: N-HMC Sampler Module**
This module performs posterior sampling in the noise space.
- **Prior Term:** $p(x_T) = \mathcal{N}(0, I)$. The gradient is $\nabla_{x_T} \log p(x_T) = -x_T$.
- **Likelihood Term:** $p(y|x_T) = \mathcal{N}(y; A(D(x_T)), \sigma_y^2 I)$. The gradient is $-\frac{1}{2\sigma_y^2} \nabla_{x_T} \|y - A(D(x_T))\|^2$.
- **Hamiltonian Dynamics:** Defined as $H = U + V$, where $U = -\log p(x_T) - \log p(y|x_T)$ and $V = \frac{1}{2}v^\top M^{-1}v$. 
- **Leapfrog Integrator:** Discretizes the trajectory with step size $\delta$. Updates momentum $p$ and position $x_T$ iteratively.
- **Metropolis-Hastings Correction:** Applied at the end of each trajectory with acceptance probability $\alpha = \min(1, \exp(-H_1 + H_0))$ to correct discretization errors.
- **Annealing Schedule:** Uses a schedule for $\sigma_y$ during initial iterations to allow larger step sizes and better exploration, gradually declining to the target level.

**Module 3: NA-NHMC Sampler Module**
This module extends N-HMC for unknown noise types and levels.
- **Noise Prior:** Uses the Jeffreys prior $p(\sigma_y^2) \sim \frac{1}{\sigma_y^2}$.
- **Marginalized Likelihood:** Integrates out $\sigma_y^2$ to yield $p(y|x_T) \propto (\frac{1}{2}\|y - A(D(x_T))\|^2)^{-m/2}$.
- **Likelihood Gradient:** $\nabla_{x_T} \log p(y|x_T) = (\frac{-m}{2\|y - A(D(x_T))\|^2}) \nabla_{x_T} \|y - A(D(x_T))\|^2$. This is mathematically equivalent to the N-HMC gradient with an adaptively estimated $\sigma_y^2$.

### 7. All Mathematical Formulas and Symbol Definitions

**Inverse Problem Formulation:**
$$y = A(x) + \eta$$
$y \in \mathbb{R}^m$: noisy measurements; $x \in \mathbb{R}^n$: unknown signal; $A$: known forward operator; $\eta \in \mathbb{R}^m$: additive noise.

**Forward Diffusion SDE:**
$$dx = -\frac{\beta_t}{2}x dt + \sqrt{\beta_t} dw$$
$w$: standard Wiener process; $\beta_t$: variance schedule.

**Forward Markov Chain Discretization:**
$$q(x_{1:T}|x_0) := \prod_{t=1}^T q(x_t|x_{t-1}), \quad q(x_t|x_{t-1}) := \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$$

**Reverse Diffusion SDE:**
$$dx = -\frac{\beta_t}{2}x dt - \beta_t \nabla_x \log p_t(x) dt + \sqrt{\beta_t} dw$$
$w$: time-reversed standard Wiener process; $p_t(x)$: marginal probability; $\nabla_x \log p_t(x)$: score function.

**Reverse Process (DDPM):**
$$p_\theta(x_{0:T}) := p(x_T) \prod_{t=1}^T p_\theta(x_{t-1}|x_t), \quad p_\theta(x_{t-1}|x_t) := \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

**Definition 3.1 (Manifold Feasibility):**
For a pretrained diffusion model, let $p_t(x_t)$ denote the marginal distribution at noise level $t$, and let $\mathcal{M}_t$ be its high-probability generative manifold. An inverse-problem solver maintains manifold feasibility if the intermediate states $\{x_t\}$ fed into the denoiser remain close to $\mathcal{M}_t$ for all $t$, ensuring the final reconstruction $x_0$ lies on the learned data manifold.

**N-HMC Conditional Score:**
$$\nabla_{x_T} \log p(x_T|y) = \nabla_{x_T} \log p(x_T) + \nabla_{x_T} \log p(y|x_T)$$
$$\nabla_{x_T} \log p(x_T) = -\nabla_{x_T} \|x_T\|_2^2 = -x_T$$
$$\nabla_{x_T} \log p(y|x_T) = \nabla_{x_T} \log p(y|D(x_T)) = -\frac{\nabla_{x_T} \|y - A(D(x_T))\|_2^2}{2\sigma_y^2}$$

**Leapfrog Integrator (single step with step size $\delta$):**
$$v(t + \delta/2) = v(t) - \frac{\delta}{2} \frac{\partial U}{\partial x} \Big|_{x(t)}$$
$$x(t + \delta) = x(t) + \delta M^{-1} v(t + \delta/2)$$
$$v(t + \delta) = v(t + \delta/2) - \frac{\delta}{2} \frac{\partial U}{\partial x} \Big|_{x(t+\delta)}$$
$v$: momentum variable ($v(0) \sim \mathcal{N}(v; 0, M)$); $U$: potential energy; $M$: mass matrix.

**Metropolis-Hastings Acceptance Probability:**
$$\alpha = \min(1, \exp(-H_1 + H_0))$$
$H_0, H_1$: initial and proposed Hamiltonian.

**Proposition 1 (Robustness to Measurement Noise):**
Assume $p_\theta(\hat{x}_0) \approx \mathcal{N}(\hat{x}_0; x_0^*, \sigma_0^2 I_n)$. The residual satisfies:
$$\mathbb{E}_{(\hat{x}_0, y) \sim p_\theta(\hat{x}_0, y|x_0^*)} \|y - A\hat{x}_0\|^2 = \sigma_y^2 \text{tr}(BB^\top) + \text{tr}(A \Sigma_{post} A^\top)$$
where:
$$\Sigma_{post} = \left( \frac{A^\top A}{\sigma_y^2} + \frac{I_n}{\sigma_0^2} \right)^{-1}, \quad B = \left( I_m - \frac{A \Sigma_{post} A^\top}{\sigma_y^2} \right)$$
$I_m$: $m \times m$ identity matrix; $m$: dimension of $y$.

**Corollary 1.1:**
Under assumptions of Proposition 1, if $\sigma_0 / \sigma_y \ll 1$:
$$\mathbb{E}_{(\hat{x}_0, y) \sim p_\theta(\hat{x}_0, y|x_0^*)} \|y - A(x_0)\|^2 \to m\sigma_y^2$$

**Noise-Adaptive Prior:**
$$p(\sigma_y^2) \sim \frac{1}{\sigma_y^2}$$

**Marginalization of $\sigma_y^2$:**
$$p(y|x_T) = \int_0^\infty p(y|x_T, \sigma_y^2) p(\sigma_y^2) d\sigma_y^2$$
$$\propto \int_0^\infty \frac{1}{(2\pi \sigma_y^2)^{m/2}} \exp\left[ -\frac{\|y - A(D(x_T))\|^2}{2\sigma_y^2} \right] \frac{1}{\sigma_y^2} d\sigma_y^2$$
$$\propto \int_0^\infty (\sigma_y^2)^{-\frac{m}{2} - 1} \exp\left[ -\frac{(1/2)\|y - A(D(x_T))\|^2}{\sigma_y^2} \right] d\sigma_y^2$$
$$\propto \left( \frac{1}{2} \|y - A(D(x_T))\|^2 \right)^{-m/2}$$

**Marginalized Log-Likelihood:**
$$\log p(y|x_T) = \left( -\frac{m}{2} \right) \log \left( \frac{1}{2} \|y - A(D(x_T))\|^2 \right)$$

**Proposition 2 (NA-NHMC Update Rule):**
Under assumptions of Proposition 1 and $\sigma_0 / \sigma_y \ll 1$:
$$\nabla_{x_T} \log p(y|x_T)_{\text{NA-NHMC}} = -\frac{1}{2\sigma_y^2} \nabla_{x_T} \|y - A(D(x_T))\|^2$$
Proof derivation:
$$\nabla_{x_T} \log p(y|x_T) = \left( \frac{-m}{2 \|y - A(D(x_T))\|^2} \right) \nabla_{x_T} \|y - A(D(x_T))\|^2 = \left( -\frac{1}{2\sigma_y^2} \right) \nabla_{x_T} \|y - A(D(x_T))\|^2$$

**Lemma 1:** Product of two Gaussian PDFs $q_1(x) = \mathcal{N}(x; \mu_1, \Sigma_1)$ and $q_2(x) = \mathcal{N}(x; \mu_2, \Sigma_2)$ is proportional to $\mathcal{N}(x, \mu, \Sigma)$, where $\mu = \Sigma(\Sigma_1^{-1} \mu_1 + \Sigma_2^{-1} \mu_2)$, $\Sigma = (\Sigma_1^{-1} + \Sigma_2^{-1})^{-1}$.

**Lemma 2:** Bias-variance decomposition for random variable $x$ with mean $\mu$ and covariance $\Sigma$: $\mathbb{E}[\|x - a\|^2] = \|\mu - a\|^2 + \text{tr}(\Sigma)$.

### 8. Algorithm Pseudocode

**Algorithm 1: N-HMC**
Require: # HMC iterations $K$, # leapfrog steps $L$, initial integration step size $\delta$, measurement noise schedule $\{\sigma_{y,k}\}$, $x_T$, $y$, $A$, $\gamma$
1: for $k = 0$ to $K - 1$ do
2:   repeat
3:     $p \sim \mathcal{N}(0, I)$ // Initial momentum
4:     $\hat{x}_0 = \text{DDIM}(x_T)$
5:     $H_0 = \frac{1}{2}\|x_T\|^2 + \frac{1}{2\sigma_{y,k}^2}\|y - A(\hat{x}_0)\|^2 + \frac{1}{2}p^\top p$ // Current Hamiltonian
6:     $x_T^* \leftarrow x_T$ // Initialize proposal $x_T$
7:     for $l = 0$ to $L-1$ do
8:       $p \leftarrow p - \frac{\delta}{2} \left( x_T^* + \frac{1}{2\sigma_{y,k}^2} \nabla_{x_T^*} \|y - A(\hat{x}_0^*)\|^2 \right)$ // Update momentum
9:       $x_T^* \leftarrow x_T^* + \delta p$ // Update $x_T^*$
10:      $\hat{x}_0^* = \text{DDIM}(x_T^*)$
11:      $p \leftarrow p - \frac{\delta}{2} \left( x_T^* + \frac{1}{2\sigma_{y,k}^2} \nabla_{x_T^*} \|y - A(\hat{x}_0^*)\|^2 \right)$ // Update momentum
12:    end for
13:    $H_1 = \frac{1}{2}\|x_T^*\|^2 + \frac{1}{2\sigma_{y,k}^2}\|y - A(\hat{x}_0^*)\|^2 + \frac{1}{2}p^\top p$ // Proposal Hamiltonian
14:    $u \sim \text{Unif}(0, 1)$
15:    if $u < \exp(H_0 - H_1)$ then
16:      Accept proposal
17:    else
18:      $\delta \leftarrow \gamma \delta$ // Anneal step size $\delta$
19:    end if
20:  until Proposal accepted
21:  $x_T \leftarrow x_T^*$ // Accept the proposal
22: end for
23: return $x_T$

**Algorithm 2: DDIM**
Require: # diffusion steps $T$, diffusion model $s_\theta$, initial seed $x_T$
1: for $t = T - 1$ to $0$ do
2:   $\hat{\epsilon}_{t+1} = s_\theta(x_{t+1}, t+1)$ // Compute the score
3:   $\hat{x}_0(x_{t+1}) = \frac{1}{\sqrt{\alpha_{t+1}}} \left( x_{t+1} - \sqrt{1 - \alpha_{t+1}} \hat{\epsilon}_{t+1} \right)$ // Predict $\hat{x}_0$ with Tweedie's formula
4:   $x_t = \sqrt{\alpha_t} \hat{x}_0(x_{t+1}) + \sqrt{1 - \alpha_t} \hat{\epsilon}_{t+1}$ // Unconditional DDIM step
5: end for
6: return $x$

**Algorithm 3: NA-NHMC**
Require: # HMC iterations $K$, # leapfrog steps $L$, initial integration step size $\delta$, $x_T$, $y$, $A$, $\gamma$
1: for $k = 0$ to $K - 1$ do
2:   repeat
3:     $p \sim \mathcal{N}(0, I)$ // Initial momentum
4:     $\hat{x}_0 = \text{DDIM}(x_T)$
5:     $H_0 = \frac{1}{2}\|x_T\|^2 + \frac{m}{2} \log \left( \|y - A(\hat{x}_0)\|^2 \right) + \frac{1}{2}p^\top p$ // Current Hamiltonian
6:     $x_T^* \leftarrow x_T$ // Initialize proposal $x_T$
7:     for $l = 0$ to $L-1$ do
8:       $p \leftarrow p - \frac{\delta}{2} \left( x_T^* + \frac{m}{2\|y - A(\hat{x}_0^*)\|^2} \nabla_{x_T^*} \|y - A(\hat{x}_0^*)\|^2 \right)$ // Update momentum
9:       $x_T^* \leftarrow x_T^* + \delta p$ // Update $x_T^*$
10:      $\hat{x}_0^* = \text{DDIM}(x_T^*)$
11:      $p \leftarrow p - \frac{\delta}{2} \left( x_T^* + \frac{m}{2\|y - A(\hat{x}_0^*)\|^2} \nabla_{x_T^*} \|y - A(\hat{x}_0^*)\|^2 \right)$ // Update momentum
12:    end for
13:    $H_1 = \frac{1}{2}\|x_T^*\|^2 + \frac{m}{2} \log \left( \|y - A(\hat{x}_0^*)\|^2 \right) + \frac{1}{2}p^\top p$ // Proposal Hamiltonian
14:    $u \sim \text{Unif}(0, 1)$
15:    if $u < \exp(H_0 - H_1)$ then
16:      Accept proposal
17:    else
18:      $\delta \leftarrow \gamma \delta$ // Anneal step size $\delta$
19:    end if
20:  until Proposal accepted
21:  $x_T \leftarrow x_T^*$ // Accept the proposal
22: end for
23: return $x_T$