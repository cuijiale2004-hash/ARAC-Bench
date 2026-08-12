**1. Research Background and Existing Pain Points**

Modern Large Language Model (LLM) pre-training consumes vast amounts of compute and training data, making the scaling behavior (scaling laws) of different models a key distinguishing factor. Discrete diffusion language models (DLMs) have emerged as an alternative to autoregressive language models (ALMs) to address fundamental limitations such as the inability to generate multiple tokens in parallel and the inability to revise previously generated tokens. However, the scaling behavior of DLMs has not yet been fully explored. Existing pain points include:
*   **Lack of Scaling Exploration:** Prior work suggests DLMs require more data and compute to match ALM performance, but the scaling behavior of uniform and hybrid-noise DLMs remains an open question, limited to small-scale ablations.
*   **Limitations of Masked Diffusion Models (MDMs):** While MDMs are the predominant DLM archetype, they suffer from the inability to revise previously generated tokens because every token experiences exactly one state transition (masked to unmasked), prohibiting transitions between two unmasked states.
*   **Likelihood Gap:** Uniform diffusion imposes less structure on the generative process than masking, making it a strictly more difficult task (requiring the model to predict which tokens are noisy in addition to imputing them). This results in a likelihood gap at small scales compared to MDMs.
*   **Flawed Scaling Law Methodology:** Prior work on scaling MDMs made potentially undesirable design choices, such as assuming the training loss can approach zero given infinite compute (no irreducible loss) and fixing the learning rate and batch size to constant values, which may not be optimal for different model sizes and token budgets.

**2. Core Research Motivation and Scientific Questions**

The core research motivation is to determine if discrete diffusion language models, particularly uniform diffusion, can overcome their small-scale limitations and become competitive with or superior to autoregressive models at scale. The paper investigates whether the likelihood gap between masked and uniform diffusion shrinks with scale and seeks to establish rigorous scaling laws for DLMs. The scientific questions are:
*   How does the scaling behavior of DLMs depend on the noise type (masked vs. uniform vs. hybrid)?
*   Can a unified framework based on signal-to-noise ratio (SNR) simplify the theory and implementation of discrete diffusion models?
*   How do crucial hyperparameters like batch size and learning rate scale with training tokens and model size for DLMs?
*   Is uniform diffusion a viable candidate for data-bound settings compared to masked diffusion and ALMs?

**3. Overall Core Idea and Design Philosophy**

The overall core idea is to propose a new family of hybrid diffusion that smoothly interpolates between masked and uniform diffusion based on the signal-to-noise ratio (SNR) and to systematically analyze the scaling laws of these models with careful attention to hyperparameter tuning. The design philosophy argues that defining the diffusion process through SNR is more natural and principled than defining it through time, aligning with continuous-state diffusion theory. By reframing Generalized Interpolating Discrete Diffusion (GIDD) in terms of SNR, the authors simplify the theory and prove invariance to the noise schedule. The methodology abandons fixed hyperparameter assumptions, instead using CompleteP for stable learning rate transfer and tuning batch size and learning rate separately for each scale. The authors hypothesize that because uniform diffusion imposes less inductive bias, it requires more capacity but scales more favorably with compute, particularly in token-constrained (data-bound) regimes.

**4. Core Innovation Points**

*   **Diffusion Process based on SNR Interpolation:** A new family of hybrid diffusion is proposed that smoothly interpolates between masked and uniform diffusion by defining a transition point depending on the SNR, rather than time. This allows control over the proportion of masking and random perturbation at any point in the noising process.
*   **Reframing GIDD in terms of SNR:** The GIDD ELBO is reparameterized in terms of log-SNR. This simplifies both theory and implementation, closes the gap to continuous-state diffusion theory, and proves that interpolating discrete diffusion is invariant to the noise schedule.
*   **Refined Scaling Law Methodology:** The strategy refines prior work by utilizing CompleteP for learning rate transfer across width and depth and by treating batch size as a crucial hyperparameter that scales with training tokens, rather than fixing it. Scaling laws are estimated without learning rate annealing, justified by the trend of treating annealing as a distinct training stage and empirical evidence that annealing yields a constant improvement without affecting optimal hyperparameters.
*   **Discovery of Parameter-Heavy Scaling for Uniform Diffusion:** The discovered scaling laws reveal that uniform diffusion incentivizes more parameter-heavy scaling (more parameters, less data) compared to masked diffusion and ALMs. This makes uniform diffusion more token-efficient at compute-optimality and a promising candidate in data-bound settings.

**5. Overview of the Overall Technical Solution**

The technical solution involves defining the GIDD framework and reparameterizing it using log-SNR to derive a simplified ELBO. A universal hybrid mixing distribution is defined using a sigmoid function of SNR to interpolate between masking and uniform noise. For training and scaling law estimation, models ranging from 25M to 570M non-embedding parameters are trained across five noise types. Hyperparameters (learning rate, batch size) are swept systematically, using CompleteP parameterization and a constant learning rate schedule without annealing. Iso-FLOP profiles are used to fit scaling laws describing compute-optimal model size, dataset size, and training loss. Finally, the predicted scaling laws are validated by training 3B and 10B parameter models, confirming that uniform diffusion closes the likelihood gap with masked diffusion at scale and matches the scaling trend of autoregressive models like DeepSeek.

**6. Detailed Module Design**

*   **Generalized Interpolating Discrete Diffusion (GIDD) Module:** Defines the noising process as an interpolated categorical distribution between data $x$ and a mixing distribution $\pi_t$. The Markov chain transitions and marginals are defined to smoothly transition from data to noise.
*   **SNR Reparameterization Module:** Replaces the time variable $t$ with log-SNR $\lambda = \log \frac{\alpha}{1-\alpha}$. The signal strength $\alpha$ is defined via the sigmoid relation $\alpha = \sigma(\lambda)$. This module transforms the forward process into a function of $\lambda$ and derives the corresponding ELBO for importance sampling over log-SNRs.
*   **Universal Hybrid Mixing Distribution Module:** Defines the mixing distribution $\pi_\lambda$ as a convex combination of uniform noise $u$ and masking noise $m$. The interpolation is controlled by a sigmoid function of the log-SNR $\lambda$ with hyperparameters $a$ and $b$ controlling the transition point and speed.
*   **Architecture & Optimization Module:** Uses a Transformer architecture with CompleteP parameterization (where bulk parameters require larger initialization variance and learning rate than auxiliary parameters), Squared ReLU activations, RMSNorm, QK-norm, and attention logit soft-capping. Uses LaProp optimizer instead of Adam. Employs diffusion forcing (independent per-token noise levels) for 50% of samples and context augmentation with random empty tokens.
*   **Confidence-based Sampling Module (Inference):** Extends confidence-based sampling to uniform diffusion. In each step, it fully denoises the top $k=1$ position that maximizes a confidence heuristic based on the prior probability and the improvement potential of the model's prediction.

**7. All Mathematical Formulas and Symbol Definitions**

*   **GIDD Transitions and Marginals:**
    $q_t(x) = \alpha_t x + \beta_t \pi_t$
    $q_{t|s}(z_s) = \alpha_{t|s} z_s + \beta_{t|s} \pi_{t|s}$
    with $\beta_t = 1 - \alpha_t$, $\alpha_{t|s} = \alpha_t / \alpha_s$, $\beta_{t|s} \pi_{t|s} = \beta_t \pi_t - \alpha_{t|s} \beta_s \pi_s$.
*   **GIDD Negative ELBO (NELBO):**
    $-\log p_\theta(x) \le \mathbb{E}_{t \sim \mathcal{U}(0,1), z \sim q_t(x)} \left[ w_t(x)_z \left\{ D_{KL}(q_t(x) \| q_t(x_\theta)) + D_{IS}(q_t(x)_z \| q_t(x_\theta)_z) \right\} \right] + C$
*   **Itakura-Saito Divergence:**
    $D_{IS}(p \| q) = p/q - \log p/q - 1$
*   **Weighting Term (Time):**
    $w_t(x) = \frac{1}{q_t(x)} \left( \beta_t \pi'_t - \frac{\alpha'_t}{\alpha_t} \pi_t \right)$
*   **Log-SNR Definition:**
    $\lambda = \log \frac{\alpha}{1-\alpha}$, $\alpha = \sigma(\lambda) = \frac{1}{1+e^{-\lambda}}$
*   **Forward Process in terms of $\lambda$:**
    $q_\lambda(x) = \sigma(\lambda)x + \sigma(-\lambda)\pi_\lambda$
*   **Proposition 1 (ELBO in terms of SNR):**
    $-\log p(x) \le \mathbb{E}_{\lambda, z} \left[ \frac{w_\lambda(x)_z}{p(\lambda)} \left\{ D_{KL}(q_\lambda(x) \| q_\lambda(x_\theta)) + D_{IS}(q_\lambda(x)_z \| q_\lambda(x_\theta)_z) \right\} \right] + C$
    with $\lambda \sim p(\lambda)$, $z \sim q_\lambda(x)$.
*   **Weighting Term (SNR):**
    $w_\lambda(x) = \frac{\sigma(-\lambda)(\pi_\lambda - \pi'_\lambda)}{q_\lambda(x)}$
*   **Universal Hybrid Mixing Distribution:**
    $\pi_\lambda = \sigma(a\lambda + b)u + (1 - \sigma(a\lambda + b))m$
    where $u = \frac{1}{N-1}(1 - e_m)$ and $m = e_m$.
    Derivative: $\pi'_\lambda = a\sigma'(a\lambda + b)(u - m)$
*   **Batch Size and Step Count Hyperbolic Relation:**
    $\left( \left[ \frac{S}{S_{min}} \right]^\alpha - 1 \right) \left( \left[ \frac{B}{B_{min}} \right]^\alpha - 1 \right) = 1$
    Token-optimal pair: $B^* = 2^{1/\alpha} B_{min}$, $S^* = 2^{1/\alpha} S_{min}$, $D^* = 4^{1/\alpha} B_{min} S_{min}$
*   **Optimal Batch Size Scaling Law:**
    $B^* = 10^{2.66} D^{0.8368}$ (Fitted relation from Appendix)
*   **Optimal Learning Rate Scaling Law:**
    $\eta^* = 10^{2.06} B^{0.3412}$ (assuming B is optimal)
*   **NLL to bpb Conversion:**
    $bpb = 0.34124 \cdot NLL$
*   **CompleteP Parameters:**
    Initialization variances: $\sigma_{base} = 0.4, \sigma_{aux} = 0.02$
    Learning rates: $\eta_{base} = 0.3, \eta_{aux} = 0.02 \cdot \eta_{base}$
*   **FLOPs per token Approximation:**
    $M = 6P + 12LDN$ (Method 1, DeepSeek)
*   **Confidence Sampling Heuristic:**
    $conf(z_t) = p_{prior}(z_t) \cdot (\max_{z'} p_\theta(x = z' | z_t) - p_\theta(x = z_t | z_t))$

**8. Algorithm Pseudocode**

The paper does not provide a single consolidated pseudocode block for the training algorithm. However, the inference mechanism for confidence-based sampling is defined algorithmically as follows:

**Algorithm: Confidence-based Sampling for Uniform Diffusion**
Input: Model $p_\theta$, Prior distribution $p_{prior}$, Number of steps $T$, Number of tokens $N$
Output: Generated sequence $x_0$

1.  Initialize $z_T \sim p_{prior}(z_T)$ (sample pure noise sequence)
2.  For $t = T$ down to 1:
3.      For each position $i \in [1, N]$:
4.          Calculate $conf(z_t)_i = p_{prior}(z_{t,i}) \cdot (\max_{z'} p_\theta(x = z' | z_t)_i - p_\theta(x = z_{t,i} | z_t)_i)$
5.      Select top $k=1$ position $j$ such that $conf(z_t)_j$ is maximized
6.      For position $j$:
7.          $z_{t-1, j} \sim p_\theta(x | z_t)$ (fully denoise top position)
8.      For positions $i \neq j$:
9.          $z_{t-1, i} = z_{t, i}$ (keep current token/noise)
10. Return $x_0 = z_0$