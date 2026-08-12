**1. Research Background and Existing Pain Points**

**Research Background:** 
The nature of the relationship between Bayesian sampling and stochastic gradient descent (SGD) in neural networks has been a long-standing open question in the theory of deep learning. One of the core problems in developing a scientific theory of deep learning models is giving a descriptive theory of how the internal model structure evolves during training as the model gains "knowledge" about its training distribution and how this evolution relates to the generalization ability of deep learning models. Classical methods for understanding model generalization such as the Bayesian Information Criterion (BIC) fail to give accurate descriptions of the generalization behavior of deep learning, due to its "singular" nature. This has led recent research to utilize Watanabe’s singular learning theory (SLT) as the basis for studying deep learning models. The key result of singular learning theory is the widely applicable Bayesian information criterion (WBIC) which says that the generalization error of a model with parameter $w$ is controlled by the learning coefficient $\lambda(w)$, which corresponds to the "complexity" of some local area around the parameter. 

**Existing Pain Points:**
1. **Failure of Classical BIC:** Classical BIC assumes that models are "regular statistical models," requiring identifiability (unique parameters) and a positive definite Fisher Information matrix near true parameters (non-degenerate Hessian of the loss). Neural networks are highly singular and generally admit many equivalent parametrizations, causing the determinant of the Hessian to be 0, rendering BIC undefined.
2. **Disconnection Between SGD Dynamics and Bayesian Description:** Despite the use of SLT, it is not clear how the dynamical picture of SGD interacts with the purely Bayesian description of SLT. 
3. **Inadequacy of Standard Langevin Models:** Theoretical work often models SGD as an Ornstein-Uhlenbeck process or uses the Stochastic Gradient Noise (SGN) model based on Langevin stochastic differential equations. This framework requires that the minima of the loss be quadratic (non-degenerate). This assumption is known to be false in general for neural networks due to the degeneracy of local minima.
4. **Failure to Capture Anomalous Diffusion:** Standard Langevin equations imply Brownian motion with displacement $R(t) \propto t^{1/2}$. However, experiments show that neural networks trained under SGD can behave super-diffusively early in training, becoming sub-diffusive as training continues (displacement $R(t) \propto t^{1/\nu}$ for $\nu \ge 2$ or logarithmically $R(t) \propto \ln t$). The traditional Langevin equation cannot capture this anomalous diffusion.

**2. Core Research Motivation and Scientific Questions**

**Core Research Motivation:** 
To shed light on the long-standing open question regarding the relationship between Bayesian sampling and SGD by modeling the long runtime behavior of SGD as diffusion on porous media. The motivation is to extend the connection between SGD and Bayesian inference to the general singular case (which characterizes deep learning) by describing the late stage training dynamics of SGD using a fractional Fokker-Planck equation. This allows for the derivation of steady-state solutions that account for the degeneracies of the loss surface which impact the dynamics.

**Scientific Questions:** 
How do the late-stage dynamics of SGD behave on singular (degenerate) loss surfaces? How does the degeneracy of the loss surface (quantified by singular learning theory) impact the diffusive behavior of SGD? And fundamentally, what is the mathematical relationship between the steady-state distribution of SGD and the Bayesian posterior over the weights in the context of singular models?

**3. Overall Core Idea and Design Philosophy**

**Overall Core Idea:** 
The long runtime dynamics of SGD are captured by taking the corresponding Fokker-Planck equation to describe diffusion on a porous geometry. This porous geometry is described by the local learning coefficient (LLC), drawing a direct relationship between the dynamics of SGD and Bayesian statistics via singular learning theory. By introducing a fractional derivative to the Fokker-Planck equation to account for sub-diffusion, and approximating the diffusion coefficient based on fractal dimensions (LLC and spectral dimension), the steady-state solution of the SGD system can be solved. This solution reveals that the SGD steady-state distribution is effectively a tempered version of the Bayesian posterior, which accounts for local accessibility constraints.

**Design Philosophy:** 
The design philosophy bridges the gap between the stochastic differential equation (SDE) picture and the probabilistic distribution picture. Instead of tracking a single trajectory, the Fokker-Planck equation describes the deterministic evolution of the probability distribution. Because the loss surface is singular (degenerate), the medium is treated as "porous," where the volume of accessible states (pores) scales with a fractal dimension (the LLC). The dynamics of diffusion on this porous medium are sub-diffusive, necessitating fractional calculus. Finally, by approximating the complex diffusion tensor as a scalar function dependent on the geometry, the steady-state equation becomes solvable, linking the optimization dynamics back to the Bayesian posterior via tempering.

**4. Core Innovation Points (List all innovations in the original text completely)**

1. **Modeling SGD Dynamics as Fractal Diffusion on Porous Media:** The paper proposes modeling the long runtime behavior of SGD as diffusion on porous media, utilizing a fractional Fokker-Planck equation (FFPE) with a Caputo fractional derivative to capture the sub-diffusive dynamics that standard Langevin equations miss.
2. **Linking Singular Learning Theory to Fractal Dimensions:** The paper establishes that the local learning coefficient (LLC) from SLT behaves as a localized mass (Minkowski-Bouligand) fractal dimension determining the geometry of degenerate near-critical points.
3. **Derivation of the Alexander-Orbach Relation for SGD:** The paper introduces the spectral dimension $d_s$ and walk dimension $d_{walk}$ to capture SGD dynamics, and establishes the relationship $d_{walk}(t) = \frac{2\lambda(w_t)}{d_s}$ near critical points, linking the geometry (LLC) to the dynamics.
4. **Scalar Approximation of the Diffusion Tensor:** The paper proves that for reasonable choices of learning rate in the large batch size regime, the diffusion tensor can be approximated by a scalar function $D_\xi(w) = \xi^{2-\frac{2\lambda(w_t)}{d_s}}$ for long runtimes, derived from the characteristic length scale $\xi$ and fractal dimensions.
5. **Derivation of the Tempered Bayesian Posterior:** The paper solves the FFPE for steady states and proves that the local steady-state distribution of SGD is a tempered version of the Bayesian posterior, specifically $p(w|X_m) = \frac{\rho(w)p_s(w)}{mD_\xi Z^{mD_\xi}}$, where the tempering $mD_\xi$ accounts for local accessibility constraints determined by the learning coefficient.
6. **Bounding Dynamics by Geometry:** The paper establishes the inequality $d_s \le \lambda(w(t))$ in the small learning rate regime, proving that large local volumes (small LLC implies greater local volume) trap the spread of SGD, slowing it down.

**5. Overview of the Overall Technical Solution**

1. **Formalize SGD Weight Updates:** Define the SGD weight update as an overdamped Langevin SDE separating the population loss gradient and stochastic noise.
2. **Construct the Fractional Fokker-Planck Equation:** Move from the SDE picture to the probability density picture using the Fokker-Planck equation. Replace the standard time derivative with the Caputo fractional derivative to formulate the Time Fractional Fokker-Planck Equation (FFPE) capturing sub-diffusion.
3. **Characterize Singular Geometry:** Define the local learning coefficient (LLC) as the fractal dimension (mass dimension) describing the volume of "pores" (low-loss parameters) in the loss landscape.
4. **Characterize Diffusive Dynamics:** Define the spectral dimension $d_s$ (volume of states SGD can reach over time) and the walk dimension $d_{walk}$ (scaling of displacement). Link them to the LLC via the Alexander-Orbach relation.
5. **Approximate Diffusion Coefficient:** Prove the diffusion tensor reduces to a scalar effective diffusion coefficient $D_\xi$ dependent on the length scale $\xi$ and the walk dimension $d_{walk}$.
6. **Solve for Steady States:** Solve the FFPE with the scalar diffusion coefficient to find the local steady-state probability distribution.
7. **Establish Bayesian Correspondence:** Relate the steady-state distribution to the Bayesian posterior, demonstrating that SGD samples from a tempered version of the posterior scaled by accessibility constraints.

**6. Detailed Module Design (Complete mechanisms of each layer/sub-module)**

**Module 1: Gradient Noise and SDE Formulation**
This module models the discrete SGD updates as a continuous stochastic differential equation. The weight updates are decomposed into a deterministic gradient descent term and a random gradient noise vector (assumed anisotropic Gaussian).
- Mechanism: The system is described by the Langevin SDE (Eq 1). The displacement of this system scales as $R(t) \propto t^{1/2}$, characteristic of Brownian motion.

**Module 2: Fractional Fokker-Planck Equation (FFPE)**
This module shifts from the single-trajectory SDE view to the deterministic probability distribution view.
- Mechanism: The standard Fokker-Planck equation (Eq 2) describes the evolution of the probability density $p(w,t)$. To account for the observed anomalous sub-diffusion ($R(t) \propto t^{1/\nu}$ for $\nu \ge 2$) in the late stage of training, the standard time derivative is replaced with the Caputo fractional derivative operator (Eq 3). This forms the FFPE (Eq 4), which models diffusion on porous media where the steady state does not depend on the early super-diffusive stage.

**Module 3: Singular Geometry via Local Learning Coefficient (LLC)**
This module defines the geometry of the loss surface using SLT.
- Mechanism: The "pores" of the media are the set of parameters with loss $K_m[w] < \epsilon$ within a ball of radius $r$. The singular integral $V(\epsilon)$ (Eq 5) measures the volume of these accessible states. The local learning coefficient $\lambda(w^*)$ (Eq 6) is defined as the scaling exponent such that $V(\epsilon) \propto \epsilon^{\lambda(w^*)}$ (Eq 7). The LLC acts as a localized mass fractal dimension determining the geometry of degenerate near-critical points.

**Module 4: Fractal Diffusion Dimensions**
This module defines the dynamics of the diffusion process on the singular geometry.
- Mechanism: The spectral dimension $d_s$ is defined by the volume of occupied states $V_s(t) \sim t^{\frac{d_s}{2}}$ (Eq 8, 9, 10). It determines how fast diffusion fills out the medium. The walk dimension $d_{walk}$ defines the displacement scaling $R(t) \sim t^{\frac{1}{d_{walk}}}$ (Eq 11). For sub-diffusion, $d_{walk} > 2$. The geometry and dynamics are linked by the Alexander-Orbach relation (Theorem 3.1), stating $d_{walk}(t) = \frac{2\lambda(w_t)}{d_s}$ near critical points.

**Module 5: Effective Diffusion Coefficient**
This module simplifies the complex diffusion tensor into a usable scalar form.
- Mechanism: Theorem 3.2 establishes that small-scale dynamics and first passage times are LLC dependent. Lemma 3.1 proves that the diffusion tensor can be approximated by a scalar function for long runtimes and large batch sizes because the effective diffusion tensor is low rank (most eigenvalues of the Hessian are approx 0). Lemma 3.2 defines the effective diffusivity $D_\xi = \frac{\xi^2}{\xi^{d_{walk}}} = \xi^{2-d_{walk}}$. Combining this with the Alexander-Orbach relation yields the Fractal Effective Diffusion Coefficient (Corollary 3.1, Eq 13).

**Module 6: Steady States and Bayesian Correspondence**
This module solves the FFPE and relates the solution to Bayesian inference.
- Mechanism: Solving the FFPE with the constant scalar diffusion coefficient $D_\xi$ yields the local steady-state solution $p_s(w) \propto e^{-\frac{\gamma L_m[w]}{D_\xi}}$ (Lemma 3.3). By setting $\gamma = 1$ and linking $e^{-m L_m[w]}$ to the likelihood $p(X_m|w)$, the relationship between the SGD steady state and the Bayesian posterior is derived (Corollary 3.2). The Bayesian posterior $p(w|X_m)$ is proportional to the tempered SGD steady state, scaled by $mD_\xi$. Furthermore, the dynamics are bounded by the geometry such that $d_s \le \lambda(w(t))$ (Lemma 3.4).

**7. All Mathematical Formulas and Symbol Definitions**

- **Eq 1 (SGD Langevin SDE):**
  $\frac{dw}{dt} = -\gamma \nabla L(w_{t-1}) + \Sigma_{w_{t-1}}$
  where $L$ is the population loss, $t$ is the timestep, $\gamma$ is the learning rate, and $\Sigma_{w_{t-1}}$ is a random vector.

- **Eq 2 (Fokker-Planck Equation):**
  $\frac{\partial p(w,t)}{\partial t} = \nabla \cdot (D(w,t)\nabla p(w,t) - \gamma p(w,t)\nabla L(w))$
  where $p$ is a probability density function, $D$ is the diffusion coefficient, $\gamma$ is a scalar (friction), and $L$ is a loss function.

- **Eq 3 (Caputo Fractional Derivative Operator):**
  $D_t^\alpha f(t) = \frac{1}{\Gamma(1-\alpha)} \int_0^t \frac{f'(t)}{(t-\tau)^\alpha} d\tau$
  where $0 < \alpha < 1$ is a real number.

- **Eq 4 (Time Fractional Fokker-Planck Equation):**
  $D_t^\alpha p(w,t) = \nabla \cdot (D(w,t)\nabla p(w,t) - \gamma p(w,t)\nabla L_m[w])$

- **Eq 5 (Singular Integral):**
  $V(\epsilon) = \int_{B_r(w^*,\epsilon)} \rho(w)dw$
  where $\rho(w)$ is some arbitrary choice of prior distribution.

- **Eq 6 (Local Learning Coefficient):**
  $\lambda(w^*) = \lim_{\epsilon \to 0} \frac{\log \frac{V(a\epsilon)}{V(\epsilon)}}{\log(a)}$
  where $0 < a < 1$ is an arbitrary constant.

- **Eq 7 (LLC Asymptotic Scaling):**
  $V(\epsilon) \propto \epsilon^{\lambda(w^*)}$

- **Eq 8 (Spectral Dimension):**
  $V_s(t) \sim t^{\frac{d_s}{2}}$

- **Eq 9 (Multi-timescale Spectral Dimension):**
  $V_s(t) = t^{\frac{d_s(t_i)}{2}}$ if $t$ belongs to timescale $t_i$.

- **Eq 10 (Asymptotic Spectral Dimension):**
  $V_s(t) \sim t^{\frac{d_s^\infty}{2}}$ as $t \to \infty$

- **Eq 11 (Walk Dimension):**
  $R(t) \sim t^{\frac{1}{d_{walk}}}$
  where $d_{walk} > 2$ for sub-diffusion.

- **Eq 12 (Alexander-Orbach Relation):**
  $d_{walk}(t) = \frac{2\lambda(w_t)}{d_s}$

- **Eq 13 (Fractal Effective Diffusion Coefficient):**
  $D_\xi(w) = \xi^{2-\frac{2\lambda(w_t)}{d_s}}$

- **Eq 14 (Steady State & Likelihood Relation):**
  $\frac{p_s(w)}{mD_\xi} \propto p(X_m|w)$

- **Eq 15 (Tempered Bayesian Posterior):**
  $p(w|X_m) = \frac{\rho(w)p_s(w)}{mD_\xi Z^{mD_\xi}}$
  where $Z^{mD_\xi}$ is the partition function and $\rho$ is the prior.

- **Eq 16 (Average Learning Coefficient):**
  $\bar{\lambda}(w(t)) = \lim_{\tau \to \infty} \frac{1}{\tau} \int_0^\tau \lambda(w(t))dt$

- **Eq 17 (Linear Regression for Spectral Dimension):**
  $\log(R(t)) = \frac{d_s}{2\lambda(w)} \log(t) + c$

- **Eq 25 (Local Learning Coefficient Estimator):**
  $\hat{\lambda}(w^*) = \frac{n}{\log n} [E_{w|B_r(w^*)}(L_n(w)) - L_n(w^*)]$

**Symbol Definitions:**
$W$: Parameter space; $X$: Set of tuples $(x_i, f(x_i))$; $L[X, w]$: Population loss; $L_m[X_m, w]$: Empirical loss; $\gamma$: Learning rate or friction coefficient; $\Sigma$: Stochastic noise vector; $p(w,t)$: Probability density function; $D(w,t)$: Diffusion coefficient/tensor; $D_t^\alpha$: Caputo fractional derivative; $\rho(w)$: Prior distribution; $B_r(w^*)$: Ball of radius $r$ about parameter $w^*$; $\lambda(w^*)$: Local learning coefficient; $V_s(t)$: Total volume of occupied states; $d_s$: Spectral dimension; $R(t)$: Displacement of a particle at time $t$; $d_{walk}$: Walk dimension; $\xi$: Characteristic length scale; $D_\xi$: Effective local diffusion coefficient; $p_s(w)$: Steady-state distribution; $Z^{mD_\xi}$: Partition function.

**8. Algorithm Pseudocode**

The paper does not provide explicit algorithm pseudocode blocks in the standard format. The implementation logic relies on the iterative SDE formulation and the regression formula for estimation. The procedural steps derived from the text are:

**Logic 1: Iterative SDE Update**
Given $w_{t-1}$, update weights via:
$w_t = w_{t-1} - \gamma \nabla L(w_{t-1}) + \Sigma_{w_{t-1}}$

**Logic 2: Spectral Dimension Estimation**
1. Compute $\log(R(t))$ where $R(t)$ is the total weight displacement at time $t$.
2. Solve the linear regression problem defined in Eq 17:
   $\log(R(t)) = \frac{d_s}{2\lambda(w)} \log(t) + c$
   to find $d_s$.

**Logic 3: Local Learning Coefficient Estimation**
Compute the estimator defined in Eq 25:
$\hat{\lambda}(w^*) = \frac{n}{\log n} [E_{w|B_r(w^*)}(L_n(w)) - L_n(w^*)]$