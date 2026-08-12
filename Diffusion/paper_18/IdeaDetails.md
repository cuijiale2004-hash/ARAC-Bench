## 1. Research Background and Existing Pain Points

**Research Background:**
Diffusion models produce high-quality synthetic data by gradually adding Gaussian noise through a diffusion process and approximating the time-reversal of this noising process using a learned denoiser network. For conditional sampling, Classifier-Free Guidance (CFG) has become a prominent technique. CFG modifies the generating process by incorporating a guidance term calculated as the difference between conditional and unconditional denoiser networks, scaled by a guidance weight $\omega$. Empirically, CFG drastically improves the performance of diffusion models compared to purely conditional generation.

**Existing Pain Points:**
1. **Distributional Mismatch:** Standard CFG uses a static, large guidance weight $\omega$ to improve visual results. However, this comes at the cost of poorer distributional alignment. Standard CFG was initially justified as a method for sampling from a modified distribution $p^\omega(x_0|c) \propto p(x_0)^{-\omega}p(x_0|c)^{1+\omega}$, but subsequent work has established that the standard CFG sampling procedure does not produce samples from this target distribution. Instead, it shifts the generated samples towards the modes of the original conditional distribution.
2. **Impractical Correction Methods:** Correcting the discrepancy introduced by CFG requires computationally intensive methods like Sequential Monte Carlo (SMC) or Markov Chain Monte Carlo (MCMC), which are often impractical for large-scale applications.
3. **Suboptimal Static Weights:** Since the learned denoiser $\hat{x}_\theta(x_t, c)$ is only an approximation of the true denoiser $\mathbb{E}[x_0|x_t, c]$, CFG empirically acts as a correction to this approximation. A constant guidance weight $\omega$ is arbitrary and suboptimal for modeling the target conditional distribution $p(x_0|c)$ across different timesteps and conditions.

## 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:**
Given that CFG boosts quality by acting as a correction to the approximation errors of the denoiser, learning the guidance weights should allow for a better approximation of the target conditional distribution $p(x_0|c)$ than using a static weight. By adjusting the guidance weight to be a continuous function of the conditioning and the time steps, one can minimize the distributional mismatch between the noised samples from the true conditional distribution and the samples from the guided diffusion process.

**Scientific Questions:**
1. How can we parameterize and learn a guidance weight $\omega_{c,(s,t)}$ that depends on the conditioning $c$, the proposal timestep $t$, and the target timestep $s$?
2. How can we define a theoretically sound yet practical objective function with low variance to train these guidance weights by enforcing consistency conditions between the true and guided diffusion processes?
3. How can this framework be extended to reward-guided sampling to bias samples towards high-reward regions without suffering from reward hacking?

## 3. Overall Core Idea and Design Philosophy

**Overall Core Idea:**
Generalize the CFG method by allowing the guidance weight $\omega_{c,(s,t)}$ to be a continuous function of the conditioning $c$ and time steps $s, t$. Learn this weight by enforcing a "Self-Consistency" condition: starting from clean data $x_0$, noising it to time $t$, and then denoising with guidance back to time $s$ should result in a distribution that matches the true noising process from $x_0$ directly to time $s$.

**Design Philosophy:**
The design is based on enforcing consistency conditions that any valid diffusion process must satisfy. A theoretically sound "Marginal Consistency" condition is first derived but found to be impractical due to high variance in gradients. To overcome this, a stronger but more practical condition called "Self-Consistency" is introduced. By conditioning on specific data points $(x_0, c)$, the variance is significantly reduced, resulting in a simple and effective objective. The guidance weights are optimized using Maximum Mean Discrepancy (MMD) to match the distributions. Furthermore, the framework incorporates a reward function by regularizing the reward maximization loss with the self-consistency loss to prevent reward hacking.

## 4. Core Innovation Points

1. **Learned Guidance Weights:** Generalizing CFG to allow conditioning-dependent and time-dependent guidance weights $\omega_{c,(s,t)}$, breaking away from the suboptimal static weight paradigm.
2. **Self-Consistency Objective:** Deriving a low-variance training objective based on matching the distribution of guided denoised samples to the true noised distribution, conditioned on specific data points $(x_0, c)$, overcoming the high-variance issue of marginal consistency.
3. **Simplified $\ell_2$ Objective:** Introducing a computationally cheaper variant of the self-consistency loss that minimizes the $\ell_2$ distance between the guided and true noised samples, avoiding the quadratic complexity of MMD interaction terms.
4. **Reward-Guided Sampling Integration:** Extending the framework to optimize for a reward function $R(x_0, c)$ by adding a reward maximization term regularized by the self-consistency loss, thereby preventing reward hacking.
5. **Distribution $p(s,t)$ Design:** Identifying that training with larger time gaps $\delta \approx 0.1$ between $s$ and $t$ provides a more stable and informative gradient signal than the small gaps used during inference, while generalizing effectively to small-step intervals.

## 5. Overview of the Overall Technical Solution

The overall technical solution involves freezing a pre-trained conditional and unconditional denoiser and training a lightweight guidance network $\omega_\phi_{c,(s,t)}$.
1. **Process Definition:** For a given data point $(x_0, c)$, the true process generates noised samples $x_s \sim p_{s|0}(\cdot|x_0)$. The guided process noises $x_0$ to $x_t$, applies the guided denoiser $\hat{x}_\theta(x_t, c; \omega)$, and uses DDIM to step back to time $s$, yielding $\tilde{x}_s(\omega)$.
2. **Optimization:** The guidance network parameters $\phi$ are optimized to minimize the distributional mismatch between $\tilde{x}_s(\omega)$ and $x_s$ using the Self-Consistency loss (MMD) or the Simplified $\ell_2$ loss.
3. **Reward Extension:** For reward-guided sampling, a reward loss term is added to the self-consistency loss, weighted by $\gamma_R$, to tilt the distribution towards high-reward regions while maintaining distributional consistency.
4. **Inference:** At inference time, the standard DDIM sampling loop is modified such that at each step from $t$ to $s$, the guidance weight is dynamically computed by the trained guidance network $\omega_\phi_{c,(s,t)}$.

## 6. Detailed Module Design

**1. Guided Denoiser Module:**
Instead of using the standard conditional denoiser $\hat{x}_\theta(x_t, c)$, the guided denoiser linearly combines the conditional and unconditional denoisers:
$\hat{x}_\theta(x_t, c; \omega) = \hat{x}_\theta(x_t, c) + \omega \Delta_\theta(x_t, c)$
where $\Delta_\theta(x_t, c) = \hat{x}_\theta(x_t, c) - \hat{x}_\theta(x_t, \emptyset)$ and $\omega = \omega_{c,(s,t)}$ is the dynamic guidance weight.

**2. True Noising Process Module:**
For a target time $s$, the true noised sample is drawn directly from the noising process:
$x_s \sim p_{s|0}(\cdot|x_0) = \mathcal{N}(x_s; \alpha_s x_0, \sigma_s^2 I_d)$

**3. Guided Backward Process Module:**
To generate the guided sample at time $s$, the process first noises $x_0$ to time $t$:
$\tilde{x}_t \sim p_{t|0}(\cdot|x_0) = \mathcal{N}(\tilde{x}_t; \alpha_t x_0, \sigma_t^2 I_d)$
Then, it computes the guided clean prediction $\hat{x}_0 = \hat{x}_\theta(\tilde{x}_t, c; \omega_{c,(s,t)})$ and samples $\tilde{x}_s(\omega)$ using the DDIM transition kernel:
$\tilde{x}_s(\omega) \sim p_{s|t,0}(\cdot|\tilde{x}_t, \hat{x}_\theta(\tilde{x}_t, c; \omega_{c,(s,t)}))$

**4. Guidance Network Module:**
The guidance weights are parameterized by a neural network:
$\omega_\phi_{c,(s,t)} = \omega(s, t, c; \phi)$
Architecture: A 6-layer MLP with hidden size 64/512. The time steps are converted to log SNR and processed by a 2-layer MLP. A ReLU activation is applied at the end to ensure $\omega \ge 0$.

**5. Distribution $p(s,t)$ Sampling Module:**
To sample the time steps for training, the target time $s$ is sampled uniformly: $s \sim U[S_{min}, 1-\zeta-\delta]$. The time increment is sampled $\Delta t \sim U[\delta, 1-\zeta-s]$, and the proposal time is set as $t = s + \Delta t$.

## 7. All Mathematical Formulas and Symbol Definitions

**Symbol Definitions:**
- $p_t$: probability distribution $p(x_t)$
- $p_{s|t}$: conditional distribution $p(x_s|x_t)$
- $p_{0|c}$: conditional distribution $p(x_0|c)$
- $p_{s|t,0}$: conditional distribution $p(x_s|x_t, x_0)$
- $p_{s|t,c}$: conditional distribution $p(x_s|x_t, c)$
- $p_{0,c}$: joint distribution $p(x_0, c)$
- $\mathcal{N}(x; \mu, \Sigma)$: Gaussian density of argument $x$, mean $\mu$ and covariance $\Sigma$
- $\alpha_t, \sigma_t$: noise schedule parameters where $\alpha_0=1, \sigma_0=0$ and $\alpha_1=0, \sigma_1=1$
- $\varepsilon$: churn parameter $\in [0,1]$ controlling stochasticity in DDIM

**Mathematical Formulas:**

Diffusion process:
$p(x_{t_1:t_N}|x_{t_0}) = p_{t_N|t_0}(x_{t_N}|x_{t_0}) \prod_{k=1}^{N-1} p_{t_k|t_{k+1},0}(x_{t_k}|x_{t_{k+1}}, x_{t_0})$ (1)

DDIM transition:
$p_{s|t,0}(x_s|x_t, x_0) = \mathcal{N}(x_s; \mu_{s,t}(x_0, x_t), \Sigma_{s,t})$ (2)

Noising process:
$p_{t|0}(x_t|x_0) = \mathcal{N}(x_t; \alpha_t x_0, \sigma_t^2 I_d)$ (3)

Conditional sampling:
$p_{s|t,c}(x_s|x_t, c) = \int p_{s|t,0}(x_s|x_t, x_0) p_{0|t,c}(x_0|x_t, c) dx_0$ (4)

Denoiser training loss:
$L(\theta) = \int_0^1 \lambda(t) \mathbb{E}_{(x_0,c)\sim p_{0,c}, x_t\sim p_{t|0}} [\|x_0 - \hat{x}_\theta(x_t, c)\|^2] dt$ (5)

DDIM sampling step:
$x_{t_k} \sim \mathcal{N}(\mu_{t_k, t_{k+1}}(\hat{x}_\theta(x_{t_{k+1}}, c), x_{t_{k+1}}), \Sigma_{t_k, t_{k+1}})$ (6)

CFG denoiser:
$\hat{x}_\theta(x_t, c; \omega) = \hat{x}_\theta(x_t, c) + \omega \Delta_\theta(x_t, c)$ (7)

CFG sampling step:
$x_{t_k} \sim \mathcal{N}(\mu_{t_k, t_{k+1}}(\hat{x}_\theta(x_{t_{k+1}}, c; \omega), x_{t_{k+1}}), \Sigma_{t_k, t_{k+1}})$ (8)

Maximum Mean Discrepancy (MMD):
$\text{MMD}(\beta, \lambda)[p_\omega, p] = \mathbb{E}_{p_\omega \otimes p}[\|x-y\|_2^\beta] - \frac{\lambda}{2} (\mathbb{E}_{p_\omega \otimes p_\omega}[\|x-x'\|_2^\beta] + \mathbb{E}_{p \otimes p}[\|y-y'\|_2^\beta])$ (9)

True marginal:
$p_s(x_s) = \int p_{s|0}(x_s|x_0) p_{0,c}(x_0, c) dx_0 dc$ (10)

Marginal via denoising:
$p_s(x_s) = \int \int \int p_{s|t,c}(x_s|x_t, c) p_{t|0}(x_t|x_0) p_{0,c}(x_0, c) dx_t dx_0 dc$ (11)

Guided marginal:
$p_s^{t,(\theta,\omega)}(x_s) = \int \int \int p_{s|t,c}^{(\theta,\omega)}(x_s|x_t, c) p_{t|0}(x_t|x_0) p_{0,c}(x_0, c) dx_t dx_0 dc$ (12)

Guided transition approximation:
$p_{s|t,c}^{(\theta,\omega)}(x_s|x_t, c) = p_{s|t,0}(x_s|x_t, \hat{x}_\theta(x_t, c; \omega))$ (13)

Marginal consistency loss:
$L_m(\omega) = \mathbb{E}_{(s,t)\sim p(s,t)}[\text{MMD}(\beta, \lambda)[p_s^{t,(\theta,\omega)}(\cdot), p_s(\cdot)]]$ (15)

Self-consistency distribution:
$p_{s|0,c}^{t,(\theta,\omega)}(x_s|x_0, c) = \int p_{s|t,c}^{(\theta,\omega)}(x_s|x_t, c) p_{t|0}(x_t|x_0) dx_t$ (16)

Self-consistency condition:
$p_{s|0,c}^{t,(\theta,\omega)}(x_s|x_0, c) \approx p_{s|0,c}(x_s|x_0, c) = p_{s|0}(x_s|x_0)$ (17)

Self-consistency loss:
$L_{\beta,\lambda}(\omega) = \mathbb{E}_{(x_0,c)\sim p_{0,c}, s,t\sim p(s,t)}[\text{MMD}(\beta, \lambda)[p_{s|0,c}^{t,(\theta,\omega)}(\cdot|x_0, c), p_{s|0,c}(\cdot|x_0)]]$ (18)

Expanded self-consistency loss:
$L_{\beta,\lambda}(\omega) = \mathbb{E}_{(x_0,c)\sim p_{0,c}, s,t\sim p(s,t)}[\mathbb{E}[\|\tilde{x}_s(\omega) - x_s\|_2^\beta] - \frac{\lambda}{2}\mathbb{E}[\|\tilde{x}_s(\omega) - \tilde{x}'_s(\omega)\|_2^\beta]]$ (20)

Simplified $\ell_2$ objective:
$L_{\ell_2}(\omega) = L_{\beta,0}(\omega) = \mathbb{E}_{(x_0,c)\sim p_{0,c}, s,t\sim p(s,t)}[\mathbb{E}[\|\tilde{x}_s(\omega) - x_s\|_2^2]]$ (21)

Empirical objective:
$\hat{L}_{\beta,\lambda}(\phi) = \frac{1}{n} \sum_{i=1}^n [\frac{1}{m^2} \sum_{j,k=1}^m \|\tilde{x}_{s_i}^j(\omega_\phi) - x_{s_i}^k\|_2^\beta - \frac{\lambda}{2} \frac{1}{m(m-1)} \sum_{j \neq k} \|\tilde{x}_{s_i}^j(\omega_\phi) - \tilde{x}_{s_i}^k(\omega_\phi)\|_2^\beta]$ (22)

Reward loss:
$L_R(\phi) = -\mathbb{E}_{(s,t)\sim p(s,t), x_0,c\sim p(x_0,c), x_t\sim p_{t|0}(\cdot|x_0)}[R(\hat{x}_\theta(x_t, c; \omega_\phi_{c,(s,t)}), c)]$

Total loss with reward:
$L_{tot}(\phi) = \hat{L}_{\beta,\lambda}(\phi) + \gamma_R L_R(\phi)$

DDIM Mean and Covariance:
$\mu_{s,t}(x_0, x_t) = (\varepsilon^2 r_{1,2}(s, t) + (1-\varepsilon^2)r_{0,1})x_t + \alpha_s(1-\varepsilon^2 r_{2,2}(s, t) - (1-\varepsilon^2)r_{1,1}(s, t))x_0$
$\Sigma_{s,t} = \sigma_s^2(1 - (\varepsilon^2 r_{1,1}(s, t) + (1-\varepsilon^2))^2)I_d$ (23)
where $r_{i,j}(s, t) = (\alpha_t/\alpha_s)^i (\sigma_s^2/\sigma_t^2)^j$

## 8. Algorithm Pseudocode

**Algorithm 1 Learning to Guide**
1: Input: Init. guidance parameters $\phi$; (frozen) denoiser $\hat{x}_\theta$; data distribution $p_0$; learning rate $\eta$; $\zeta > 0, S_{min} > 0, \delta > 0$, b.s. $n$, n. of particles $m$, $\lambda \in [0, 1]$, $\beta \in [0, 2]$, DDIM churn $\varepsilon \in [0, 1]$.
2: repeat
3: Sample batch of clean data and their conditionings $\{x_0^i, c^i\}_{i=1}^n$ i.i.d. $\sim p_{0,c}$.
4: Sample $\{s_i\}_{i=1}^n$ i.i.d. $\sim U[S_{min}, 1-\zeta-\delta]$, $\{\Delta t_i\}_{i=1}^n$ i.i.d. $\sim U[\delta, 1-\zeta-s_i]$, let $t_i = s_i + \Delta t_i$
5: (True process) Sample $m$ particles $\{x_{s_i}^j\}_{j=1}^m$ i.i.d. $\sim p_{s_i|0}(\cdot|x_0^i)$ from noising process (3)
6: (Guided process) Sample $m$ particles $\{\tilde{x}_{t_i}^j\}_{j=1}^m \sim p_{t_i|0}(\cdot|x_0^i)$ from noising process (3)
7: Compute guidance weights $\omega_i = \omega_\phi_{c^i,(s_i,t_i)}$ and $\hat{x}_\theta(\tilde{x}_{t_i}^j, c^i; \omega_i)$ using (7)
8: Sample $\tilde{x}_{s_i}^j(\omega_\phi) \sim p_{s_i|t_i,0}(\cdot|\tilde{x}_{t_i}^j, \hat{x}_\theta(\tilde{x}_{t_i}^j, c^i; \omega_i))$ from DDIM (2) with churn parameter $\varepsilon$
9: (Loss) Compute loss
10: $\hat{L}_{\beta,\lambda}(\phi) = \frac{1}{n} \sum_{i=1}^n [\frac{1}{m^2} \sum_{j,k=1}^m \|\tilde{x}_{s_i}^j(\omega_\phi) - x_{s_i}^k\|_2^\beta - \frac{\lambda}{2} \frac{1}{m(m-1)} \sum_{j \neq k} \|\tilde{x}_{s_i}^j(\omega_\phi) - \tilde{x}_{s_i}^k(\omega_\phi)\|_2^\beta]$
11: Update $\phi \leftarrow \phi - \eta \nabla_\phi \hat{L}_{\beta,\lambda}(\phi)$
12: until convergence
13: Output: Optimized guidance network parameters $\phi$

**Algorithm 4 Learning to Guide with reward**
1: Input: Init. guidance parameters $\phi$; (frozen) denoiser $\hat{x}_\theta$; data distribution $p_0$; learning rate $\eta$; $\zeta > 0, S_{min} > 0, \delta > 0$, b.s. $n$, n. of particles $m$, $\lambda \in [0, 1]$, $\beta \in [0, 2]$, DDIM churn $\varepsilon \in [0, 1]$. Reward function $R(x_0, c)$, reward loss weight $\gamma_R \ge 0$.
2: repeat
3: Sample batch of clean data and their conditionings $\{x_0^i, c^i\}_{i=1}^n$ i.i.d. $\sim p_{0,c}$.
4: Sample $\{s_i\}_{i=1}^n$ i.i.d. $\sim U[S_{min}, 1-\zeta-\delta]$, $\{\Delta t_i\}_{i=1}^n$ i.i.d. $\sim U[\delta, 1-\zeta-s_i]$, let $t_i = s_i + \Delta t_i$
5: (True process) Sample $m$ particles $\{x_{s_i}^j\}_{j=1}^m$ i.i.d. $\sim p_{s_i|0}(\cdot|x_0^i)$ from noising process (3)
6: (Guided process) Sample $m$ particles $\{\tilde{x}_{t_i}^j\}_{j=1}^m \sim p_{t_i|0}(\cdot|x_0^i)$ from noising process (3)
7: Compute guidance weights $\omega_i = \omega_\phi_{c^i,(s_i,t_i)}$ and $\hat{x}_\theta(\tilde{x}_{t_i}^j, c^i; \omega_i)$ using (7)
8: Sample $\tilde{x}_{s_i}^j(\omega_\phi) \sim p_{s_i|t_i,0}(\cdot|\tilde{x}_{t_i}^j, \hat{x}_\theta(\tilde{x}_{t_i}^j, c^i; \omega_i))$ from DDIM (2) with churn parameter $\varepsilon$
9: (Loss) Compute loss
10: $\hat{L}_{\beta,\lambda}(\phi) = \frac{1}{n} \sum_{i=1}^n [\frac{1}{m^2} \sum_{j,k=1}^m \|\tilde{x}_{s_i}^j(\omega_\phi) - x_{s_i}^k\|_2^\beta - \frac{\lambda}{2} \frac{1}{m(m-1)} \sum_{j \neq k} \|\tilde{x}_{s_i}^j(\omega_\phi) - \tilde{x}_{s_i}^k(\omega_\phi)\|_2^\beta]$
11: (Reward loss) Compute reward loss
12: $\hat{L}_R(\phi) = \frac{1}{nm} \sum_{i=1}^n \sum_{j=1}^m R(\hat{x}_\theta(\tilde{x}_{t_i}^j, c^i; \omega_i), c^i)$
13: (Total loss) Compute total loss
14: $\tilde{L}_{tot}(\phi) = \hat{L}_{\beta,\lambda}(\phi) + \gamma_R \hat{L}_R(\phi)$
15: Update $\phi \leftarrow \phi - \eta \nabla_\phi \tilde{L}_{tot}(\phi)$
16: until convergence
17: Output: Optimized guidance network parameters $\phi$