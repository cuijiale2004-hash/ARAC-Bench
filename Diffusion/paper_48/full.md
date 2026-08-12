## ABSTRACT

Diffusion models (DMs) have recently shown remarkable performance on inverse problems (IPs). Optimization-based methods can fast solve IPs using DMs as powerful regularizers, but they are susceptible to local minima and noise overfitting. Although DMs can provide strong priors for Bayesian approaches, enforcing measurement consistency during the denoising process leads to manifold infeasibility issues. We propose Noise-space Hamiltonian Monte Carlo (N-HMC), a posterior sampling method that treats reverse diffusion as a deterministic mapping from initial noise to clean images. N-HMC enables comprehensive exploration of the solution space, avoiding local optima. By moving inference entirely into the initial-noise space, N-HMC keeps proposals on the learned data manifold. We provide a comprehensive theoretical analysis of our approach and extend the framework to a noise-adaptive variant (NA-NHMC) that effectively handles IPs with unknown noise type and level. Extensive experiments across four linear and three nonlinear inverse problems demonstrate that NA-NHMC achieves superior reconstruction quality with robust performance across different hyperparameters and initializations, significantly outperforming recent state-of-the-art methods. The code is available at https://github.com/NA-HMC/NA-HMC.

## 1 INTRODUCTION

Inverse problems (IPs) have wide applications in many domains, including computer vision (Janai et al., 2021; Quan et al., 2024), protein science (Yi et al., 2023; Ouyang-Zhang et al., 2023; Yang et al., 2019), medical imaging (Song et al., 2022b; Chu et al., 2025; Dao et al., 2024), scientific computing (Zheng et al., 2025; Xia & Zabaras, 2022; Xu et al., 2024). The goal is to reconstruct an unknown x $\in \mathbb { R } ^ { n }$ from noisy measurements $\pmb { y } \in \mathbb { R } ^ { m }$

$$
\boldsymbol {y} = \mathcal {A} (\boldsymbol {x}) + \eta ,\tag{1}
$$

where A is a known forward operator, and $\eta \in \mathbb { R } ^ { m }$ is additive noise. Diffusion models (DMs) have recently shown powerful capabilities in modeling complex data distributions, which can provide a powerful class of priors for high-dimensional data x in solving IPs.

Existing diffusion-based methods have demonstrated remarkable success across diverse inverse problems (Chung et al., 2023; Daras et al., 2024; Zheng et al., 2025; Song et al., 2022b). Although remarkable progress has been made, as illustrated in Figure 1, current diffusion-based methods suffer from three complementary limitations and issues: (1) Iterative guidance methods such as DPS (Chung et al., 2023), DDRM (Kawar et al., 2022), DDNM(Wang et al., 2023), ΠGDM (Song et al., 2023), and TMPD (Boys et al., 2024) use the likelihood term to shift intermediate images directly, systematically pushing intermediate states off the learned data manifold and violating the trainingtime noise-conditioning of the denoiser, resulting in various failure reconstructions like accumulated artifacts as shown in Figure 1 (a). (2) Stochastic MAP methods that optimize in image space, including ReSample (Song et al., 2024), DiffPIR (Zhu et al., 2023), DAPS (Zhang et al., 2024), SITCOM (Alkhouri et al., 2025a), and DIIP (Chihaoui & Favaro, 2025a) can match y well with very sharp details but require carefully tuned hyperparameters to not overfit to noise. This limits their effectiveness in high or unknown noise settings. (3) Deterministic MAP methods that optimize in the DM noise space (DMPlug, (Wang et al., 2024)) remove randomness but often get stuck in a single mode, especially in severely ill-posed problems like phase retrieval, due to a lack of posterior exploration. In short, enforcing data consistency mid-diffusion can break prior adherence, while optimizing only for fidelity leads to overfitting or mode collapse.

![](images/4caa9b91a606294e5ce6a8ddbae6ae8b394470f73b645c758c7f369a7ce47aea.jpg)  
Figure 1: Comparison of existing methods and their limitations with the N-HMC method. (a) Iterative Guidance Methods (DPS) lead to manifold infeasibility. (b) Stochastic MAP methods (Re-Sample) (Song et al., 2024) are susceptible to overfitting to noise. (c) Deterministic MAP methods $( D M P l u g )$ (Wang et al., 2024) become trapped in a local mode. (d) Our method performs sampling in the noise space $\mathbf { \nabla } _ { \mathbf { x } _ { T } }$ and maps samples to images via a deterministic mapping ${ \bf x } _ { 0 } = \mathcal { D } ( { \bf x } _ { T } )$ .

Sampling from the full posterior ensures that the learned prior acts automatically as a regularizer, while an annealing schedule for noise standard deviation $\sigma _ { y }$ promotes efficient exploration and prevents the sampler from being trapped in early local modes. Importantly, the method relies only on a fixed set of hyperparameters that remain constant across tasks, datasets, levels of measurement noise, avoiding the repeated tuning required by many existing approaches.

To address the practical challenges that the measurement noise level is often unknown, we further introduce a Noise-Adaptive N-HMC (NA-NHMC). Instead of requiring a fixed noise level, we take a principled Bayesian approach, placing a non-informative prior on the noise variance and marginalizing it out. This yields a parameter-free likelihood term that automatically adapts to the true underlying noise in the measurements. As shown in experiments, this allows NA-NHMC to achieve robust, high-quality reconstructions across varying and even unknown noise types and levels without any task-specific hyperparameter tuning. In contrast, the performance of other methods depends on the hyperparameters listed in Section A.5, which were specifically tuned for Gaussian noise. Our key contributions include: (1) In Section 3.1, we propose N-HMC, a posterior sampling method that addresses the three key limitations of existing state-of-the-art (SOTA) approaches. We further analyze its sampling behavior and provide a theoretical guarantee of its robustness to measurement noise in Section 3.2 and Appendix A.2. (2) In Section 3.3, we extend our method to settings with unknown noise types and levels. We show that it outperforms SOTA methods on most metrics (Section 4.3), especially for non-linear and high noise problems. (3) In extensive experiments, NA-NHMC method solves diverse inverse problem tasks under unknown noise types and levels without any task- or noise-specific hyperparameter tuning, in contrast to many existing methods. (4) We demonstrate in Section 4.1 that the annealing schedule for $\sigma _ { y }$ helps promote early exploration and prevent local-mode collapse, especially in severely ill-posed tasks like phase retrieval.

## 2 PRELIMINARIES

## 2.1 DIFFUSION MODELS FOR INVERSE PROBLEMS

Daras et al. (2024) broadly classifies methods for solving inverse problems (IPs) into two categories. The first is maximum a posteriori (MAP) inference, which aims to find the single most probable x. An alternative is the Bayesian framework, where the goal becomes generating plausible reconstructions by sampling from the posterior distribution $p ( { \pmb x } | { \pmb y } )$ , where $p ( { \pmb x } | { \pmb y } )$ can be decomposed into the prior $p ( { \pmb x } )$ and the likelihood $p ( \pmb { y } | \pmb { x } )$ . MAP delivers fast optimization, but struggles with high noise and multimodal posteriors, easily converging to local minima. In contrast, the Bayesian approach samples from $p ( { \pmb x } | { \pmb y } )$ to generate plausible reconstructions, quantify uncertainty, and handle multimodality. Both approaches critically depend on powerful prior models like DMs that encode the complex statistical structure of complex data and prior knowledge.

Most diffusion-based approaches to IPs are based on the denoising diffusion probabilistic models (DDPM) framework (Ho et al., 2020; Song & Ermon, 2020). The framework consists of forward and reverse diffusion processes. The forward process gradually corrupts the clean images $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ towards standard Gaussian noise $\mathbf { \nabla } _ { \mathbf { x } _ { T } }$ . This process can be described by a stochastic differential equation (SDE), dx $\begin{array} { r } { \mathbf { \sigma } = - \frac { \beta _ { t } } { 2 } \pmb { x } d t + \sqrt { \beta _ { t } } d \pmb { w } } \end{array}$ , where w is the standard Wiener process. In practice, the process is discretized via a variance schedule $\{ \beta _ { t } \} _ { t = 1 } ^ { T }$ , forming a Markov chain:

$$
q (\boldsymbol {x} _ {1: T} | \boldsymbol {x} _ {0}) := \prod_ {t = 1} ^ {T} q (\boldsymbol {x} _ {t} | \boldsymbol {x} _ {t - 1}), \quad q (\boldsymbol {x} _ {t} | \boldsymbol {x} _ {t - 1}) := \mathcal {N} (\boldsymbol {x} _ {t}; \sqrt {1 - \beta_ {t}} \boldsymbol {x} _ {t - 1}, \beta_ {t} \boldsymbol {I})\tag{2}
$$

In order to generate clean images, the reverse process begins with a noisy sample ${ \pmb x } _ { T } \sim \mathcal { N } ( { \pmb x } _ { T } ; { \mathbf 0 } , I )$ and recursively refines it according to the reverse SDE, $\begin{array} { r } { d \pmb { x } = - \frac { \beta _ { t } } { 2 } \pmb { x } d t - \beta _ { t } \nabla _ { \pmb { x } } \log p _ { t } ( \pmb { x } ) d t + \sqrt { \beta _ { t } } d \pmb { \overline { { w } } } , } \end{array}$ where w is the time-reversed standard Wiener process, and $p _ { t } ( \pmb { x } )$ is the marginal probability of the noisy manifold at time t. $\nabla _ { \pmb { x } } \log p _ { t } ( \pmb { x } )$ is called the score function and is usually approximated by a neural network θ trained through score-matching methods.

Using the same discretization, clean images can be generated from the prior using an iterative denoising process.

$$
p _ {\theta} (\boldsymbol {x} _ {0: T}) := p (\boldsymbol {x} _ {T}) \prod_ {t = 1} ^ {T} p _ {\theta} (\boldsymbol {x} _ {t - 1} | \boldsymbol {x} _ {t}), \quad p _ {\theta} (\boldsymbol {x} _ {t - 1} | \boldsymbol {x} _ {t}) := \mathcal {N} (\boldsymbol {x} _ {t - 1}; \boldsymbol {\mu} _ {\theta} (\boldsymbol {x} _ {t}, t), \boldsymbol {\Sigma} _ {\theta} (\boldsymbol {x} _ {t}, t))\tag{3}
$$

Building on the DDPM framework, to accelerate the denoising process, Song et al. (2022a) proposes Denoising Diffusion Implicit Models (DDIM), which define a non-Markovian and fully deterministic forward/reverse process $( \beta _ { t } = 0 )$ . Unlike DDPM, which injects stochasticity at each step to improve robustness, DDIM iteratively maps the initial noise $\mathbf { \nabla } _ { \mathbf { \mathcal { X } } \mathcal { T } }$ to a clean sample $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ via a deterministic trajectory. For our method, this property is particularly beneficial, as it allows us to consider the entire reverse process as a deterministic mapping from $\mathbf { \nabla } _ { \mathbf { \mathcal { X } } \mathcal { T } }$ to $\scriptstyle { \mathbf { { \vec { x } } } } _ { 0 }$

Among successful DM-based methods for inverse problems, DPS and its variants (Chung et al., 2023; Kawar et al., 2022; Wang et al., 2023; Song et al., 2023; Chung et al., 2022) are best-known reconstruction algorithms. But they suffer from approximation errors from Tweedie’s formula corrections. To mitigate noise sensitivity, TMPD incorporates second-order information to correct the guidance trajectory; however, like other iterative methods, it relies on modifying intermediate states, which risks drifting off the learned manifold. SITCOM (Alkhouri et al., 2025b) operates on the noisy image at each diffusion step and enforces a triple-consistency constraint: data fidelity, backward consistency with the diffusion posterior mean, and forward consistency along the diffusion trajectory. DIIP (Chihaoui & Favaro, 2025b) updates the initial noise $\mathbf { \nabla } _ { \mathbf { x } _ { T } }$ with data fidelity gradients after the standard diffusion sampling process. DMPlug (Wang et al., 2024) proposes a noise-space formulation but treats inverse problems as optimization tasks, making it sensitive to noise. While early stopping can mitigate this, its criterion is task- and noise-dependent. At the high noise levels considered here, the optimizer often becomes trapped in a local mode, rendering early stopping ineffective. Similar behavior is observed in other Maximum a Posteriori (MAP) methods such as ReSample (Song et al., 2024), which optimizes directly in clean-image space (leading to noisy or blurry images under early stopping). DAPS (Zhang et al., 2024), despite being formulated as a posterior sampling method, uses a heuristic $\hat { \sigma } _ { y }$ that is much smaller than its true value to strengthen the consistency of the measurement. This deviation from true posterior sampling makes DAPS effectively MAP-like, inheriting the same sensitivity to noise.

## 2.2 HAMILTONIAN MONTE CARLO (HMC)

Hamiltonian Monte Carlo (HMC) (Duane et al., 1987) is an MCMC (Metropolis et al., 1953) sampling method that utilizes a fictitious momentum variable and simulates Hamiltonian dynamics to efficiently explore distant regions. Due to its superior scaling properties in high dimensions compared to other simpler Metropolis methods Brooks et al. (2011), HMC is particularly well suited for sampling in high-dimensional space, such as the $3 \times 2 5 6 \times 2 5 6$ pixel space of images.

The Hamiltonian is defined as $H = U + V$ , where $U = - \log p ( { \pmb x } )$ and $V = \textstyle { \frac { 1 } { 7 } } v ^ { \top } M ^ { - 1 } v$ . Then, we discretize the trajectory using the leapfrog integrator. For a single leapfrog step with step size δ, we have

$$
\boldsymbol {v} (t + \delta / 2) = \boldsymbol {v} (t) - \frac {\delta}{2} \frac {\partial U}{\partial \boldsymbol {x}} \Big | _ {\boldsymbol {x} (t)},\tag{4}
$$

$$
\pmb {x} (t + \delta) = \pmb {x} (t) + \delta \pmb {M} ^ {- 1} \pmb {v} (t + \delta / 2),\tag{5}
$$

$$
\boldsymbol {v} (t + \delta) = \boldsymbol {v} (t + \delta / 2) - \frac {\delta}{2} \frac {\partial U}{\partial \boldsymbol {x}} \Big | _ {\boldsymbol {x} (t + \delta)},\tag{6}
$$

where $\pmb { v } ( 0 ) \sim \mathcal { N } ( \pmb { v } ; \mathbf { 0 } , M )$ . This process is repeated L times to form a full trajectory. Due to a discretization error, the Hamiltonian is no longer preserved, which introduces bias and violates the detailed balance. To correct for this, a Metropolis-Hastings (MH) correction step is applied at the end of each trajectory with acceptance probability of $\alpha = \mathbf { \bar { m i n } } ( 1 , \exp ( - H _ { 1 } + H _ { 0 } ^ { \cdot } ) ;$ ), where $H _ { 0 } , H _ { 1 }$ denotes the initial and proposed Hamiltonian, respectively.

## 3 METHODOLOGY

In this section, we propose a posterior sampling method, Noise-space Hamiltonian Monte Carlo (N-HMC), to solve IPs with pretrained DMs. We show its derivation in Section 3.1 and discuss its robustness to measurement noise in Section 3.2. In Section 3.3, our method is modified to allow for unknown types and levels of measurement noise.

## 3.1 NOISE-SPACE HAMILTONIAN MONTE CARLO (N-HMC)

The goal in solving inverse problems is to sample from the posterior distribution $p ( \pmb { x } _ { 0 } | \pmb { y } )$ ∝ $p ( \pmb { x } _ { 0 } ) p ( \pmb { y } | \pmb { x } _ { 0 } )$ . Since direct sampling from $p ( \pmb { x } _ { 0 } )$ is intractable, pretrained diffusion models are employed to provide a powerful prior. Standard diffusion-based approaches draw ${ \pmb x } _ { T } \sim p ( { \pmb x } _ { T } )$ from a Gaussian noise prior and iteratively denoise through intermediate timesteps, aiming to sample from $p ( \pmb { x } _ { T } \mid \pmb { y } ) , p ( \pmb { x } _ { T - 1 } \mid \pmb { y } ) , \dots , p ( \pmb { x } _ { 0 } \mid \pmb { y } )$ in sequence. The key challenge is evaluating the intractable likelihood $p ( \pmb { y } | \pmb { x } _ { t } )$ at each intermediate timestep t. To address this, iterative guidance methods (Kawar et al., 2022; Wang et al., 2023; Chung et al., 2023; Song et al., 2023; Rozet et al., 2024; Song et al., 2024; Zhang et al., 2024) introduce approximations and apply likelihood corrections of $\mathbf { \boldsymbol { x } } _ { t } \gets \mathbf { \boldsymbol { x } } _ { t } + \eta \nabla _ { \mathbf { \boldsymbol { x } } _ { t } } \log p ( \mathbf { \boldsymbol { y } } | \mathbf { \boldsymbol { x } } _ { t } )$ . However, these gradient-based corrections systematically push intermediate states $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ away from the distribution on which the denoiser is trained, leading to what we refer to as the manifoldfeasibility problem. Following SITCOM (Alkhouri et al., 2025b), we formalize this issue as:

Definition 3.1 (Manifold Feasibility). For a pretrained diffusion model, let $p _ { t } ( \pmb { x } _ { t } )$ denote the marginal distribution at noise level t, and let $\mathcal { M } _ { t }$ be its high-probability generative manifold. An inverse-problem solver maintains manifold feasibility if the intermediate states $\{ \pmb { x } _ { t } \}$ fed into the denoiser remain close to $\mathcal { M } _ { t }$ for all t, ensuring the final reconstruction $\scriptstyle { \pmb x } _ { 0 }$ lies on the learned data manifold.

Geometrically, standard guidance methods update $\mathbf { x } _ { t }$ using the likelihood gradient $\nabla _ { \mathbf { x } _ { t } } \log p ( \mathbf { y } | \mathbf { x } _ { t } )$ In high-dimensional spaces, this gradient vector often contains components orthogonal to the local tangent space of the data manifold $\mathcal { M } _ { t }$ . Consequently, adding this gradient systematically pushes the state $\mathbf { x } _ { t }$ into low-probability regions (off-manifold), feeding out-of-distribution inputs to the denoiser and causing accumulated artifacts as shown in Figure 1 (a). To avoid such approximations, we propose posterior sampling by drawing from the initial noise space. The sampled noise is then unconditionally denoised to a clean image. We adopt unconditional DDIM for the denoising process, which treats the entire denoising trajectory as a deterministic mapping $\hat { \mathbf { x } } _ { 0 } = \mathcal { D } ( \mathbf { x } _ { T } )$ , so the problem becomes evaluating the posterior distribution of noise (Xia et al., 2023), i.e., $p ( { \pmb x } _ { T } | { \pmb y } )$ We refer to our approach as noise-space sampling because HMC updates are performed exclusively on the initial noise $x _ { T } \sim \mathcal { N } ( 0 , \bar { I } )$ This differs from image-space and iterative guidance methods that directly modify intermediate states $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ using measurement-consistency gradients. Sampling from the noise space offers two advantages: (i) the prior $p ( { \pmb x } _ { T } )$ is a simple Gaussian distribution, and (ii) the likelihood $p ( { \pmb y } | { \pmb x } _ { T } ) = p ( { \pmb y } | \mathcal { D } ( \bar { \pmb x } _ { T } ) )$ is directly accessible without intermediate approximations.

We use HMC for efficient posterior sampling in the noise space. To strictly justify our sampling objective, we formulate the inference process as a latent variable model where the initial noise $x _ { T }$ is the sole latent variable. We treat the unconditional DDIM process with $N$ steps as a deterministic parameterized generator function, denoted as $\mathcal { D } : \mathbb { R } ^ { n } \ \overset { \cdot } {  } \ \mathbb { R } ^ { n }$ , which maps $\mathbf { \nabla } _ { \mathbf { \mathcal { X } } \mathcal { T } }$ to a clean image $\hat { \pmb x } _ { 0 } = \mathcal { D } ( { \pmb x } _ { T } )$ . Under this formulation, the measurement generation process is defined by $\textbf { \textit { x } } _ { T } \overset { \mathcal { D } } { \longrightarrow } \hat { \textbf { \textit { x } } } _ { 0 } \overset { \mathcal { A } , \eta } { \longrightarrow } \textbf { \textit { y } }$ Consequently, the conditional distribution of y given $x _ { T }$ depends entirely on the generated image $\scriptstyle { \hat { \mathbf { x } } } _ { 0 }$ . The likelihood term is thus mathematically exact: $\dot { p } ( { \pmb y } | { \pmb x } _ { T } ) = \dot { p } ( { \pmb y } | \hat { \pmb x } _ { 0 } = \mathbf { \bar { \mathcal { D } } } ( { \pmb x } _ { T } ) ) = \ddot { N } ( { \pmb y } ; { \pmb A } ( { \pmb { \mathcal { D } } } ( { \pmb x } _ { T } ) ) , \sigma _ { u } ^ { 2 } I )$ . This allows us to perform posterior sampling directly in the noise space using the exact gradient of the likelihood with respect to $x _ { T }$ Then we can compute the conditional score using $\mathrm { B a y e s } ^ { \prime }$ rule:

$$
\nabla_ {\boldsymbol {x} _ {T}} \log p (\boldsymbol {x} _ {T} | \boldsymbol {y}) = \nabla_ {\boldsymbol {x} _ {T}} \log p (\boldsymbol {x} _ {T}) + \nabla_ {\boldsymbol {x} _ {T}} \log p (\boldsymbol {y} | \boldsymbol {x} _ {T}).\tag{7}
$$

Since $\mathbfit { \mathbf { x } } _ { T }$ is Gaussian noise in the DDIM framework, the first term is simply

$$
\nabla_ {\pmb {x} _ {T}} \log p (\pmb {x} _ {T}) = - \nabla_ {\pmb {x} _ {T}} \frac {\| \pmb {x} _ {T} \| ^ {2}}{2} = - \pmb {x} _ {T}.\tag{8}
$$

For the case of Gaussian measurement noise, if the noise level $\sigma _ { y } ^ { 2 }$ is known, the likelihood term becomes

$$
\nabla_ {\pmb {x} _ {T}} \log p (\pmb {y} | \pmb {x} _ {T}) = \nabla_ {\pmb {x} _ {T}} \log p (\pmb {y} | \mathcal {D} (\pmb {x} _ {T})) = - \nabla_ {\pmb {x} _ {T}} \frac {\| \pmb {y} - \mathcal {A} (\mathcal {D} (\pmb {x} _ {T})) \| ^ {2}}{2 \sigma_ {y} ^ {2}}.\tag{9}
$$

We define $p ( { \pmb y } | { \pmb x } _ { T } ) = p ( { \pmb y } | \mathcal { D } ( { \pmb x } _ { T } ) )$ by viewing the denoising trajectory as a deterministic mapping $\hat { \mathbf { x } } _ { 0 } = \mathcal { D } ( \mathbf { x } _ { T } )$ . This term can be computed directly using automatic differentiation. Because $\mathcal { D } ( \pmb { x } _ { T } )$ results from a multi-step denoising process, backpropagating through multiple score networks can be computationally expensive. Following Wang et al. (2024), we illustrate in Appendix A.9 that accurate samples can still be obtained with as few as two denoising steps.

Once the conditional score $\nabla _ { { \pmb x } _ { T } } \log p ( { \pmb x } _ { T } | { \pmb y } )$ is computed, our method proceeds with standard Hamiltonian Monte Carlo (HMC) sampling. We use the identity matrix as the mass matrix for momentum sampling. During implementation, we observed that the initial noise may lie in regions of very low posterior probability, which forces HMC to adopt a tiny step size in order to maintain a proper acceptance rate. To address this issue, we use an annealing schedule for $\sigma _ { y } .$ , allowing $\mathbf { \nabla } _ { \mathbf { \mathcal { X } } \mathcal { T } }$ to explore the noise space more freely with a larger step size in the start-up stage. Once $\sigma _ { y }$ gradually declines to the target level, posterior samples are collected. The complete procedure is summarized in Algorithm 1, along with the unconditional DDIM denoising process in Algorithm 2.

## 3.2 ROBUSTNESS TO MEASUREMENT NOISE

An additional benefit of N-HMC over MAP methods is that the Gaussian prior acts as a regularization term in the noise space, keeping the noise vector $\mathbf { \nabla } _ { \mathbf { x } _ { T } }$ close to the hypersphere of radius $\sqrt { n }$ Therefore, N-HMC produces samples that are robust to measurement noise, as justified by Proposition 1. For simplicity, we assume Gaussian measurement noise and that the forward operator A is approximately linear along the clean image manifold.

Proposition 1. Assume that the distribution of the decoded sample $\scriptstyle { \pmb x } _ { 0 }$ around the ground truth $\pmb { x } _ { 0 } ^ { * }$ is well-approximated by a Gaussian distribution $p _ { \theta } ( \hat { \pmb x } _ { 0 } ) \approx \mathcal { N } ( \hat { \pmb x } _ { 0 } ; \pmb x _ { 0 } ^ { * } , \sigma _ { 0 } ^ { 2 } \pmb I _ { n } )$ . Then, the residual $\pmb { y } - \pmb { A } \hat { \pmb { x } } _ { 0 }$ satisfies

$$
\mathbb {E} _ {(\hat {\boldsymbol {x}} _ {0}, \boldsymbol {y}) \sim p _ {\theta} (\hat {\boldsymbol {x}} _ {0}, \boldsymbol {y} | \boldsymbol {x} _ {0} ^ {*})} \| \boldsymbol {y} - \boldsymbol {A} \hat {\boldsymbol {x}} _ {0} \| ^ {2} = \sigma_ {y} ^ {2} \mathrm{tr} (\boldsymbol {B} \boldsymbol {B} ^ {\top}) + \mathrm{tr} (\boldsymbol {A} \boldsymbol {\Sigma} _ {\mathrm{post}} \boldsymbol {A} ^ {\top}),
$$

where

$$
\boldsymbol {\Sigma} _ {\text {post}} = \left(\frac {\boldsymbol {A} ^ {\top} \boldsymbol {A}}{\sigma_ {y} ^ {2}} + \frac {\boldsymbol {I} _ {n}}{\sigma_ {0} ^ {2}}\right) ^ {- 1}, \quad \boldsymbol {B} = \left(\boldsymbol {I} _ {m} - \frac {\boldsymbol {A} \boldsymbol {\Sigma} _ {\text {post}} \boldsymbol {A} ^ {\top}}{\sigma_ {y} ^ {2}}\right),
$$

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1: N-HMC
Require: # HMC iterations K, # leapfrog steps L, initial integration step size δ, measurement noise schedule {σy,k}, xT, y, A, γ
1: for k = 0 to K - 1 do
2: repeat
3: p ~ N(0, I) // Initial momentum
4: $\hat{x}_0$ = DDIM(xT)
5: $H_0$ = $\frac{1}{2}\|\boldsymbol{x}_T\|^2$ + $\frac{1}{2\sigma_{y,k}^2}\|\boldsymbol{y} - \mathcal{A}(\hat{x}_0)\|^2$ + $\frac{1}{2}\boldsymbol{p}^\top \boldsymbol{p}$ // Current Hamiltonian
6: $x_T^* \leftarrow x_T$ // Initialize proposal xT
7: for l = 0 to L - 1 do
8: p ← p - $\frac{\delta}{2}\left(x_T^* + \frac{1}{2\sigma_{y,k}^2} \nabla_{x_T^*}\|\boldsymbol{y} - \mathcal{A}(\hat{x}_0^*)\|^2\right)$ // Update momentum
9: $x_T^* \leftarrow x_T^* + \delta p$ // Update xT*
10: $\hat{x}_0^* = \text{DDIM}(x_T^*)$
11: p ← p - $\frac{\delta}{2}\left(x_T^* + \frac{1}{2\sigma_{y,k}^2} \nabla_{x_T^*}\|\boldsymbol{y} - \mathcal{A}(\hat{x}_0^*)\|^2\right)$ // Update momentum
12: end for
13: $H_1$ = $\frac{1}{2}\|\boldsymbol{x}_T^*\|^2$ + $\frac{1}{2\sigma_{y,k}^2}\|\boldsymbol{y} - \mathcal{A}(\hat{x}_0^*)\|^2$ // Proposal Hamiltonian
14: u ~ Unif(0, 1)
15: if u &lt; exp(H0 - H1) then
16: Accept proposal
17: else
18: δ ← γδ // Anneal step size δ
19: end if
20: until Proposal accepted
21: $x_T$ ← $x_T^*$ // Accept the proposal
22: end for
23: return xT
</div>

and ${ \cal I } _ { m }$ is the $m \times m$ identity matrix. m denotes the dimension of $\mathbf { \pmb { y } } .$

The expected residual decomposes into two contributions: a noise-dependent term that appears only when measurement noise is present, and a second term that persists in all settings due to intrinsic uncertainty of prior diffusion models. In Corollary 1.1 below, we show that both terms behave in a way that yields a residual whose magnitude matches the true measurement noise.

Corollary 1.1 Under the assumptions of Proposition 1, if $\sigma _ { 0 } / \sigma _ { y } \ll 1$ , the residual $\pmb { y } - \pmb { \mathcal { A } } ( \hat { \pmb { x } } _ { 0 } )$ satisfies

$$
\mathbb {E} _ {(\hat {\boldsymbol {x}} _ {0}, \boldsymbol {y}) \sim p _ {\theta} (\hat {\boldsymbol {x}} _ {0}, \boldsymbol {y} | \boldsymbol {x} _ {0} ^ {*})} \| \boldsymbol {y} - \mathcal {A} (\boldsymbol {x} _ {0}) \| ^ {2} \to m \sigma_ {y} ^ {2}.
$$

In other words, the magnitude of residual aligns with the true known level of measurement noise, indicating that N-HMC remains robust and does not overfit to noise.

## 3.3 NOISE-ADAPTIVE NHMC

In practice, the type and level of measurement noise are often unknown, making the likelihood term $p ( \pmb { y } \mid \pmb { x } _ { T } )$ intractable. To address this, other methods usually have tunable hyperparameters that control the strength of the likelihood term or use task-specific early stopping criteria. Instead of this heuristic approach, we introduce a noise-adaptive sampling method, NA-NHMC, which extends N-HMC to the unknown noise setting without any additional hyperparameter tuning.

We treat the noise variance as a latent variable and adopt the Jeffreys prior, a principled noninformative choice due to its parameterization invariance. It is scale-invariant and represents maximal uncertainty about the noise level, making it appropriate when no prior information about $\sigma _ { y }$ is available. It can also be viewed as the limiting case of an Inverse-Gamma prior $\sigma _ { y } ^ { 2 } \sim \mathrm { I n v - } \tilde { \Gamma } ( \alpha , \beta )$ as $\alpha , \beta \to 0$ The Inverse-Gamma distribution is the conjugate prior for the variance of a Gaussian likelihood. Proposition 2 characterizes the resulting behavior under additional assumptions.

$$
p (\sigma_ {y} ^ {2}) \sim \frac {1}{\sigma_ {y} ^ {2}}.
$$

Proposition 2 Under the assumptions of Proposition 1 and that the pretrained diffusion model unconditionally generates images that lie on the high-quality manifold $( \sigma _ { 0 } / \sigma _ { y } \ll 1 )$ , then the update rule of NA-NHMC follows:

$$
\nabla_ {\boldsymbol {x} _ {T}} \log p (\boldsymbol {y} | \boldsymbol {x} _ {T}) _ {\mathrm{NA-NHMC}} = - \frac {1}{2 \sigma_ {y} ^ {2}} \nabla_ {\boldsymbol {x} _ {T}} \| \boldsymbol {y} - \mathcal {A} (\mathcal {D} (\boldsymbol {x} _ {T})) \| ^ {2}.
$$

By marginalizing $\sigma _ { y } ^ { 2 } .$ , the likelihood term becomes

$$
\begin{array}{l} p (\boldsymbol {y} \mid \boldsymbol {x} _ {T}) = \int_ {0} ^ {\infty} p (\boldsymbol {y} \mid \boldsymbol {x} _ {T}, \sigma_ {y} ^ {2})   p (\sigma_ {y} ^ {2})   d \sigma_ {y} ^ {2} \\ \qquad \propto \left(\frac {1}{2} \big \| \boldsymbol {y} - \mathcal {A} (\mathcal {D} (\boldsymbol {x} _ {T})) \big \| ^ {2}\right) ^ {- m / 2}. \end{array}\tag{10}
$$

(11)

where m denotes the dimensionality of the measurement space. The derivation of this expression is provided in Appendix A.1. Substituting this marginalized likelihood into the N-HMC framework yields our proposed noiseadaptive Algorithm 3. Proposition 2 below shows that, with an appropriate measurement noise prior, the likelihood term $\nabla _ { \pmb { x } _ { T } } \log p ( \pmb { y } | \pmb { x } _ { T } )$ of NA-NHMC is identical to that of N-HMC (with known noise level).

Figure 2 demonstrates Proposition 2 in practice. All experiments use an identical setup across different noise levels, highlighting that our method does not require any hyperparameter tuning for a specific noise level. Despite the absence of such tunable parameters, the estimated standard deviation of the measurement noise, computed as $\lVert \pmb { y } - \pmb { \mathcal { A } } ( \hat { \pmb { x } } _ { 0 } ) \rVert / \sqrt { m }$ , closely matches the true, unknown noise level $\sigma _ { y } .$ This confirms that NA-NHMC effectively adapts to varying noise without specific tuning. The complete NA-NHMC is summarized in Algorithm 3.

![](images/52f63b3826118d1d5f3cd26b1e3cfb6eae57916f7becda754f67d10ec3e2183b.jpg)  
Figure 2: Gaussian deblur task on FFHQ $( 2 5 6 \times 2 5 6 )$ with varying measurement noise levels $\sigma _ { y } .$ The estimated standard deviation of measurement noise $\| \pmb { y } - \pmb { \mathcal { A } } ( \hat { \pmb { x } } _ { 0 } ) \| / \sqrt { m }$ demonstrates that our noise-adaptive method accurately recovers the true $\sigma _ { y }$ (indicated by dashed line) without overfitting across different noise levels.

The flexibility of NA-NHMC goes beyond noise-level robustness. Although NA-NHMC is formulated assuming Gaussian measurement noise, experiments with alternative noise types (Section 4.3) show that the method remains effective across other common noise distributions, demonstrating its broader robustness.

## 4 EXPERIMENTS

Following previous approaches (Wang et al., 2024) (Zhang et al., 2024), we evaluate our method on two datasets: FFHQ 256 × 256 (Karras et al., 2019) and ImageNet 256 × 256 (Deng et al., 2009), using 100 images from the validation set of each dataset. We utilize the same pretrained DM trained by Chung et al. (2023) for FFHQ and by Dhariwal & Nichol (2021) for ImageNet except for ReSample. For ReSample, we use a pretrained LDM by Rombach et al. (2022). All measurements are corrupted by additive Gaussian noise with standard deviation $\sigma _ { y }$

We compare our method against several representative baselines, including DiffPIR (Zhu et al., 2023), RED-diff (Mardani et al., 2023), DPS (Chung et al., 2023), DAPS (Zhang et al., 2024), ReSample (Song et al., 2024), SITCOM (Alkhouri et al., 2025b), and DMPlug (Wang et al., 2024). The implementation details for all the baseline methods are provided in Appendix A.5. We evaluate reconstruction quality using three standard metrics: peak signal-to-noise ratio (PSNR), structural similarity index measure (SSIM) (Wang et al., 2004), and Learned Perceptual Image Patch Similarity (LPIPS) (Zhang et al., 2018).

NA-NHMC (Ours)

## 4.1 EXPERIMENT RESULTS

Linear IPs. We evaluate our approach on four linear inverse problems. For super-resolution tasks, we consider both 4× and 16× downsampling using 4 × 4 and 16 × 16 average pooling operations, respectively. We also examine random inpainting with 92% of pixels randomly masked, and anisotropic Gaussian deblurring using blur kernels with standard deviations of 20 and 1 in orthogonal directions. The results for linear IPs are presented in Tables 10-13.

Nonlinear IPs. We further assess performance on three challenging nonlinear inverse problems. The first is nonlinear deblurring using encoded blur kernels from Tran et al. (2021). The second is phase retrieval, where only the Fourier magnitude is observed as measurements. Finally, we consider HDR reconstruction, which aims to recover images with a higher dynamic range by a factor of 2 from tone-mapped observations. The results for nonlinear IPs are presented in Tables 1, 2, 14, 15.

Main Results. Our method achieves comparable or superior performance across most tasks, as measured by PSNR and SSIM on both the FFHQ and ImageNet datasets. Notably, the improvement over SOTA methods is more pronounced for nonlinear tasks, which are substantially more challenging than linear IPs. Many existing SOTA approaches are MAP-based by design (e.g., ReSample, SITCOM, and DMPlug) or become MAP-like heuristically (e.g., DAPS). While these methods perform well in low-noise regimes, they often overfit when the noise level is higher. Since the noise levels used in our experiments $( \sigma _ { y } = 0 . 0 5 , 0 . 2 0 )$ ) exceed those commonly reported in prior work, our results further demonstrate that the proposed noise-adaptive method is more robust and consistently outperforms alternatives across most tasks and metrics without any hyperparameter tuning. Figure 4 contains visual examples for the nonlinear deblurring problem. See Appendix A.12 for more examples. A fundamental distinction lies in the generalization capability across diverse degradation conditions. Standard guidance-based methods (e.g., DPS) inherently rely on manual hyperparameter calibration to balance measurement fidelity against the diffusion prior. As evidenced in Table 4, the optimal step size is highly task-dependent (ranging from $\zeta = 0 . 4 \ \mathrm { t o } \ \zeta = 1 0 . 0 )$ , meaning a static configuration fails to generalize. In contrast, NA-NHMC derives its dynamics from the marginalized posterior, which effectively acts as an automatic gradient normalization mechanism. This structural advantage allows a single configuration to robustly generalize across varying tasks and noise levels without task-specific recalibration.

DPS  
DAPS  
![](images/cabf1ab1b4769dadedc433413c33994291844e816dcfc48adb763e804161af4a.jpg)  
Figure 3: Comparative results are averaged over 100 independent runs. (Top) Mean absolute error (MAE) heapmaps. (Bottom) Standard deviation heatmaps across runs. Our method achieves the lowest standard deviation compared to DPS and DAPS, indicating reduced sensitivity to initialization.

![](images/bc1e20a5dc334b6f7e5144e2ead974242f57d3c11db156a4d9f82518010ad599.jpg)  
Figure 4: Nonlinear deblurring results on FFHQ (256 × 256) dataset with $\sigma _ { y } = 0 . 2 .$ . Visual comparison across state-of-the-art methods shows our approach produces high-quality reconstructions with sharp details and minimal artifacts.

## 4.2 HIGHLY ILL-POSED IPS: PHASE RETRIEVAL

Another challenge commonly faced by both MAP and sampling-based methods is becoming trapped in a local mode, particularly in highly multimodal IPs such as phase retrieval. Figure 5 illustrates this issue. While DPS and DMPlug occasionally recover the correct solution, most initializations converge to spurious local modes and are thus counted as failures. In contrast, our method incorporates early exploration through $\sigma _ { y }$ scheduling, making it more robust to initialization and substantially more likely to recover the global solution.

![](images/44065b4b1b8af086fd2d66a16625d54ffe5c3cb213dcd38c3025c23123ba28ae.jpg)

![](images/23d03ea4333de42d09612c81077097cf6e0913c2eef8ab2cff4e839ab273863a.jpg)  
Progress (%)

![](images/fe7ec4a9fb89c88cecd265c2c1fe3c4d0485388f8a63696450792965d2e81c77.jpg)  
Figure 5: Phase retrieval task on FFHQ $( 2 5 6 \times 2 5 6 )$ with $\sigma _ { y } = 0 . 0 1$ . Each curve shows the median performance, with shaded areas denoting the 5th–95th percentile interval. Our method successfully solves the IP at a much higher rate than DPS and DMPlug. This is due to the annealing schedule of $\sigma _ { y }$ that allows for initial exploration of the noise space, resulting in a lower probability of being stuck on a local mode.

We quantify robustness to initialization using the standard deviation map in Figure 3. Our method achieves a mean absolute error (MAE) comparable to that of DPS, but with substantially lower pixel-wise standard deviation. While DPS can, on average, produce accurate reconstructions, its performance is sensitive to initialization and may introduce artifacts in both the face and background (red circle). In contrast, such artifacts never appear in any of the 100 runs with our method. Notably, despite exhibiting lower overall uncertainty, our method still assigns uncertainty in complex regions, which aligns with areas of high MAE (orange).

Table 1: Non-linear IPs Results on FFHQ (256 × 256) with Gaussian Noise $\sigma _ { y } = 0 . 0 5$ . (Bold: best, underline: second best)

<table><tr><td rowspan="2"></td><td colspan="3">Nonlinear Deblurring</td><td colspan="3">Phase Retrieval</td><td colspan="3">HDR Reconstruction</td></tr><tr><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td></tr><tr><td>DiffPIR</td><td>26.12</td><td>0.743</td><td>0.289</td><td>16.77</td><td>0.482</td><td>0.543</td><td>25.20</td><td>0.814</td><td>0.223</td></tr><tr><td>RED-diff</td><td>18.12</td><td>0.217</td><td>0.680</td><td>11.83</td><td>0.213</td><td>0.769</td><td>21.44</td><td>0.525</td><td>0.458</td></tr><tr><td>DPS</td><td>23.26</td><td>0.672</td><td>0.300</td><td>10.87</td><td>0.296</td><td>0.714</td><td>27.46</td><td>0.849</td><td>0.168</td></tr><tr><td>DAPS</td><td>27.00</td><td>0.736</td><td>0.283</td><td>18.52</td><td>0.414</td><td>0.528</td><td>26.03</td><td>0.758</td><td>0.259</td></tr><tr><td>ReSample</td><td>24.57</td><td>0.637</td><td>0.432</td><td>13.95</td><td>0.377</td><td>0.677</td><td>23.65</td><td>0.722</td><td>0.386</td></tr><tr><td>SITCOM</td><td>24.97</td><td>0.569</td><td>0.328</td><td>11.89</td><td>0.216</td><td>0.723</td><td>26.97</td><td>0.753</td><td>0.256</td></tr><tr><td>DMPlug</td><td>27.15</td><td>0.784</td><td>0.266</td><td>-</td><td>-</td><td>-</td><td>25.17</td><td>0.783</td><td>0.260</td></tr><tr><td>NA-NHMC (ours)</td><td>27.66</td><td>0.792</td><td>0.249</td><td>19.30</td><td>0.554</td><td>0.482</td><td>28.45</td><td>0.849</td><td>0.217</td></tr></table>

Table 2: Non-linear IPs on ImageNet $( 2 5 6 \times 2 5 6 )$ with Gaussian Noise $\sigma _ { y } = 0 . 0 5$ . (Bold: best, underline: second best)

<table><tr><td rowspan="2"></td><td colspan="3">Nonlinear Deblurring</td><td colspan="3">HDR Reconstruction</td></tr><tr><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td></tr><tr><td>DiffPIR</td><td>24.24</td><td>0.638</td><td>0.381</td><td>23.29</td><td>0.730</td><td>0.273</td></tr><tr><td>RED-diff</td><td>17.94</td><td>0.244</td><td>0.623</td><td>20.98</td><td>0.524</td><td>0.415</td></tr><tr><td>DPS</td><td>17.60</td><td>0.427</td><td>0.482</td><td>25.31</td><td>0.763</td><td>0.248</td></tr><tr><td>DAPS</td><td>24.28</td><td>0.632</td><td>0.404</td><td>23.57</td><td>0.709</td><td>0.283</td></tr><tr><td>SITCOM</td><td>24.00</td><td>0.556</td><td>0.355</td><td>24.76</td><td>0.708</td><td>0.276</td></tr><tr><td>DMPlug</td><td>22.30</td><td>0.576</td><td>0.421</td><td>20.61</td><td>0.562</td><td>0.431</td></tr><tr><td>NA-NHMC (ours)</td><td>24.98</td><td>0.694</td><td>0.308</td><td>25.86</td><td>0.779</td><td>0.253</td></tr></table>

## 4.3 ROBUSTNESS TO UNKNOWN MEASUREMENT NOISE

In practice, the measurement noise may be unknown and its type may not be Gaussian. This can pose a problem as many methods require multiple hyperparameters that are tuned for a specific level and type of measurement noise. In this section, we evaluate our method’s robustness to unknown measurement noise on two tasks and two noise types: impulse and speckle. For impulse noise, each pixel in each channel is randomly replaced by 0 or 1 with a probability $p / 2$ each, where p ∼ Unif(0, 0.2). For speckle noise, the noise takes the form $y ( 1 + \epsilon )$ , where y is the measurement tensor and ϵ ∼ Unif(0, 0.4). Table 3 shows that our method achieves superior performance on most metrics while using the exact same hyperparameters as the Gaussian noise experiment. As illustrated in Figure 6, methods such as DiffPIR, which do not suffer from noise overfitting in the Gaussian setting, now struggle with impulse noise. In contrast, even under high noise levels, NA-NHMC remains robust, producing high-quality reconstructions.

Ground Truth  
Measurement  
DiffPIR  
![](images/31c61b529a32e61f7fd04dfa0197d8790e38d8350a929a2e7fbc0ed96814b1ee.jpg)  
RED-dif  
DPS  
DAPS  
ReSample  
DMPlug  
Ours  
Figure 6: Nonlinear deblurring results on FFHQ (256×256) dataset under different noise conditions. (Top) Impulse noise. (Bottom) Speckle noise.

Table 3: The Performance Comparison with Different Types and Levels of Measurement Noise on FFHQ (256 × 256). (Bold: best, underline: second best)

<table><tr><td rowspan="3"></td><td colspan="6">Super Resolution (×4)</td><td colspan="6">Nonlinear Deblurring</td></tr><tr><td colspan="3">Impulse</td><td colspan="3">Speckle</td><td colspan="3">Impulse</td><td colspan="3">Speckle</td></tr><tr><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td></tr><tr><td>DiffPIR</td><td>19.54</td><td>0.492</td><td>0.549</td><td>25.91</td><td>0.733</td><td>0.324</td><td>21.00</td><td>0.402</td><td>0.526</td><td>25.96</td><td>0.731</td><td>0.299</td></tr><tr><td>RED-diff</td><td>15.15</td><td>0.341</td><td>0.692</td><td>21.81</td><td>0.481</td><td>0.516</td><td>13.82</td><td>0.109</td><td>0.781</td><td>18.59</td><td>0.269</td><td>0.650</td></tr><tr><td>DPS</td><td>21.99</td><td>0.581</td><td>0.395</td><td>27.00</td><td>0.761</td><td>0.246</td><td>21.64</td><td>0.595</td><td>0.322</td><td>23.42</td><td>0.678</td><td>0.302</td></tr><tr><td>DAPS</td><td>15.00</td><td>0.361</td><td>0.702</td><td>24.48</td><td>0.597</td><td>0.442</td><td>17.94</td><td>0.259</td><td>0.657</td><td>26.46</td><td>0.698</td><td>0.308</td></tr><tr><td>ReSample</td><td>22.98</td><td>0.639</td><td>0.483</td><td>26.17</td><td>0.733</td><td>0.387</td><td>22.74</td><td>0.616</td><td>0.471</td><td>24.64</td><td>0.692</td><td>0.409</td></tr><tr><td>SITCOM</td><td>16.56</td><td>0.392</td><td>0.628</td><td>23.16</td><td>0.600</td><td>0.425</td><td>17.92</td><td>0.259</td><td>0.612</td><td>26.49</td><td>0.667</td><td>0.295</td></tr><tr><td>DMPlug</td><td>19.52</td><td>0.358</td><td>0.562</td><td>26.79</td><td>0.689</td><td>0.336</td><td>23.79</td><td>0.662</td><td>0.335</td><td>25.82</td><td>0.740</td><td>0.308</td></tr><tr><td>NA-NHMC (ours)</td><td>23.42</td><td>0.631</td><td>0.382</td><td>27.36</td><td>0.768</td><td>0.290</td><td>24.16</td><td>0.677</td><td>0.319</td><td>27.97</td><td>0.796</td><td>0.253</td></tr></table>

## 5 CONCLUSION

In this work, we introduce N-HMC, a posterior sampler that operates in the noise space using reverse diffusion as a deterministic mapping from initial noise to a clean image, enabling posterior exploration while keeping proposals on the learned data manifold. The developed noise-adaptive variant, NA-NHMC, eliminates task-specific hyperparameter tuning by automatically adapting to unknown noise types and levels, which is a significant practical advantage over existing approaches. Theory establishes the correctness and efficiency of noise-space sampling, and experiments across diverse linear and nonlinear inverse problems on FFHQ and ImageNet show state-of-the-art reconstructions, robustness to initialization and noise, competitive runtimes with a few denoising steps, and uncertainty-aware estimates. The provided analysis and experiments also show that our method can mitigate measurement-consistency drift, noise overfitting, and local-mode collapse without relying on any task-specific hyperparameter tuning.

While NA-NHMC shows promising results, it incurs higher computational cost compared to other methods such as DPS due to HMC sampling. Additionally, its reliance on a small number of diffusion steps may limit its immediate applicability to more complex applications. Moreover, the high dimensionality of the posterior leads to long warmup phases before reaching stationarity. Future work could address these challenges by developing more efficient gradient estimation techniques and incorporating faster warmup strategies that relax the requirement of exact detailed balance.