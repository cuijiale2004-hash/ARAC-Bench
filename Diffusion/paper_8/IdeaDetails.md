# Research Idea and Full Implementation Plan Extraction

## 1. Research Background and Existing Pain Points
Unsupervised representation learning, particularly sequential disentanglement, aims to separate static and dynamic factors of variation in data without relying on labels. This remains a challenging problem due to several existing pain points:
1. **Complex Optimization of Existing Methods**: Existing approaches based on variational autoencoders (VAEs) and generative adversarial networks (GANs) often rely on multiple loss terms, complicating the optimization process. For instance, C-DSVAE requires five hyperparameters solely to balance its various loss terms, requiring extensive hyperparameter tuning.
2. **Failure on Real-World Data**: Sequential disentanglement methods face challenges when applied to real-world data. Most models are evaluated on toy datasets and struggle to produce high-quality samples in real-world scenarios.
3. **Lack of Evaluation Protocol**: There is currently no established evaluation protocol for assessing sequential disentanglement performance in real-world settings. Traditional quantitative benchmarks depend on labeled data and pre-trained judges, which are sensitive to expressivity and generalizability.
4. **Absence of Diffusion Formalization**: Recently, diffusion models have emerged as state-of-the-art generative models, but no theoretical formalization exists for their application to sequential disentanglement. Existing diffusion architectures do not produce disentangled representations.
5. **Modal-Dependence**: Methods tailored specifically for video (animation-based approaches relying on temporal and spatial consistency) or audio (depending on spectral or temporal cues) cannot adapt to diverse modalities like time series, video, and audio simultaneously without extensive modifications.

## 2. Core Research Motivation and Scientific Questions
**Core Research Motivation**: A diffusion-based framework can reduce hyperparameter tuning and improve sample quality, paving the way for more robust and scalable approaches to unsupervised sequential disentanglement. Existing frameworks for sequential disentanglement lack a probabilistic modeling foundation for diffusion-based modeling.

**Scientific Questions**: How to establish a novel probabilistic modeling formalization for diffusion-based sequential disentanglement that accommodates dependent static and dynamic factors? How to design a modal-agnostic architecture that effectively disentangles high-dimensional, real-world data across video, audio, and time series using a single unified loss term, while enabling zero-shot generalization and multi-factor disentanglement?

## 3. Overall Core Idea and Design Philosophy
The core idea is to introduce the Diffusion Sequential Disentanglement Autoencoder (DiffSDA), a novel, modal-agnostic probabilistic framework for sequential disentanglement grounded in diffusion processes. Unlike prior tools that model static and dynamic factors as independent, DiffSDA models them as interdependent, enhancing the expressivity of their marginal distributions. The design philosophy relies on a single standard diffusion loss term to avoid complex optimization, leveraging the inherent architecture where the static factor is shared across the sequence and the dynamic factors are low-dimensional to naturally promote disentanglement. The framework incorporates a sequential semantic encoder, a stochastic encoder/decoder based on the EDM framework, and Latent Diffusion Models (LDM) using a pre-trained VQ-VAE to handle high-dimensional real-world sequences efficiently.

## 4. Core Innovation Points
1. **Novel Probabilistic Framework with Dependent Factors**: A novel modal-agnostic probabilistic framework for sequential disentanglement grounded in diffusion processes. Unlike most existing approaches, the formulation accommodates dependent static and dynamic factors of variation, improving expressivity, efficiency (non-autoregressive sampler), and causality.
2. **Single Unified Score Estimation Loss**: The model is optimized using a single, unified score estimation loss, eschewing the complex multiple regularizers, mutual information losses, and intricate prior modeling required by VAE/GAN methods.
3. **Real-World and Zero-Shot Disentanglement**: The design enables the effective disentanglement of high-dimensional, real-world data (video, audio, time series) and supports zero-shot disentanglement tasks across unseen datasets.
4. **Multifactor Latent Factorization via PCA**: Demonstrated capability to disentangle static and dynamic information into multiple interpretable factors using a post-training unsupervised linear fashion (Principal Component Analysis) on the learned latent spaces.
5. **Novel Evaluation Protocol**: Introduction of a novel evaluation protocol specifically designed for video-based disentanglement, incorporating three high-resolution video datasets and new unsupervised swapping metrics (Average Euclidean Distance and Average Keypoint Distance) to quantitatively measure object and motion preservation.

## 5. Overview of the Overall Technical Solution
The overall technical solution consists of three main components:
1. **Sequential Semantic Encoder**: Factorizes input data into separate static ($s_0$) and dynamic ($d^{1:V}_0$) components. It utilizes a U-Net for video data and an MLP for other modalities, coupled with linear layers and LSTM modules to summarize the sequence.
2. **Stochastic Encoder**: Follows the EDM framework, adding noise $\epsilon \sim \mathcal{N}(0, \sigma^2_t I)$ to each sequence element $x^{\tau}_0$, yielding noisy latents $x^{\tau}_t$.
3. **Stochastic Decoder**: Denoises the noisy latent representation produced by the stochastic encoder, conditioned on the disentangled factors $z^{\tau}_0 = (s_0, d^{\tau}_0)$. It follows the EDM decoding formulation featuring 63 neural function evaluations (NFEs) during inference.
To support high-resolution sequences, a Latent Diffusion Module (LDM) is incorporated, utilizing a pre-trained VQ-VAE autoencoder to map high-dimensional inputs to a lower-dimensional latent space before factorization and diffusion.

## 6. Detailed Module Design
**Sequential Semantic Encoder**:
Inspired by prior work, the encoder extracts $s_0$ and $d^{1:V}_0$. It consists of a U-Net (for video) or an MLP (for other modalities), coupled with linear layers that independently process each sequence element. An LSTM module summarizes the sequence into a latent representation $h^{1:V}$. The last hidden state, $h^V$, is passed to a linear layer to produce the static factor $s_0$. The sequence of hidden states $h^{1:V}$ are processed with another LSTM and a linear layer to produce the dynamic factors $d^{1:V}_0$.

**Stochastic Encoder**:
Follows the EDM framework. Given a sequence element $x^{\tau}_0$, it adds noise $\epsilon \sim \mathcal{N}(0, \sigma^2_t I)$ to yield $x^{\tau}_t = x^{\tau}_0 + \epsilon$.

**Stochastic Decoder**:
The decoder $D_{\theta}$ takes the noisy input $x^{\tau}_t$ and disentangled factors $z^{\tau}_0 := (s_0, d^{\tau}_0)$, returning a denoised version $\tilde{x}^{\tau}_0$. The denoiser is conditioned on $z^{\tau}_0$ through Adaptive Group Normalization (AdaGN), defined as:
AdaGN$(h, t, z_{sem}) = z_s (t_s \text{GroupNorm}(h) + t_b)$
where $z_s$ is the output of a linear layer applied to $z_{sem}$, and $t_s, t_b$ are outputs of an MLP applied to time $t$.

**Latent Diffusion Module (LDM)**:
To manage high-resolution data, a perceptual image compression model (VQ-VAE) is used. The pre-trained encoder $E$ and decoder $D$ map inputs $x^{1:V}_0$ to latent features $x^{1:V}_0 = E(x^{1:V}_0)$ and back $x^{1:V}_0 = D(x^{1:V}_0)$. The VQ-VAE utilizes hyperparameters $f = 8, Z = 256, d = 4$, encoding a frame of size $3 \times 256 \times 256$ into a latent representation of size $4 \times 32 \times 32$. The EDM formulation normalizes the training data globally, keeping $\sigma_{data}$ at its default value of 0.5, ensuring latents have a zero mean.

**Prior Modeling**:
The prior static and dynamic distribution $p_{T0}(s_0, d^{1:V}_0 | s_T, d^{1:V}_T)$ is modeled by training a separate latent DDIM model based on 10 MLP layers. This allows sampling of static and dynamic factors by reversing the trained model.

**Architecture Hyperparameters**:
Video Datasets (e.g., VoxCeleb, CelebV-HQ): U-Net backbone, base channels 192, channel multipliers [1, 2, 2, 2], attention placement [2], static dim 512-1024, dynamic dim 12-16, sequence length 10.
Audio/Time Series (e.g., TIMIT, PhysioNet): MLP backbone, base channels 128-256, static dim 16-32, dynamic dim 2-4, sequence length 68-672.

## 7. All Mathematical Formulas and Symbol Definitions

**Symbol Definitions**:
- $t \in [0, T]$: Time in the diffusion process.
- $\tau \in \{1, \ldots, V\}$: Time in the sequence.
- $T$: Maximum diffusion time.
- $V$: Maximum sequence time (sequence length).
- $x^{\tau}_t$: Sequence state of the diffusion process at diffusion step $t$ and sequence index $\tau$.
- $s_0$: Static (time-invariant) latent factor.
- $d^{1:V}_0$: Dynamic (time-variant) latent factors across the sequence.
- $z^{\tau}_0 := (s_0, d^{\tau}_0)$: Disentangled factors for sequence element $\tau$.
- $D_{\theta}$: Denoising parametric map (Decoder).
- $\lambda_t \in \mathbb{R}^+$: Positive weight for the loss at time $t$.
- $c^{skip}_t, c^{in}_t, c^{out}_t, c^{noise}_t$: EDM scaling and noise mapping parameters.
- $F_{\theta}$: Neural network inside the decoder.
- $\epsilon \sim \mathcal{N}(0, \sigma^2_t I)$: Noise added in the stochastic encoder.

**Probabilistic Modeling and Distributions**:
The joint distribution is given by:
$p(x^{1:V}_0, x^{1:V}_T, s_0, s_T, d^{1:V}_0, d^{1:V}_T) = p_{T0}(s_0, d^{1:V}_0 | s_T, d^{1:V}_T) \prod_{\tau=1}^V p_{T0}(x^{\tau}_0 | x^{\tau}_T, s_0, d^{\tau}_0)$
where $p_{T0}(s_0, d^{1:V}_0 | s_T, d^{1:V}_T)$ is a standard diffusion process and $p_{T0}(x^{\tau}_0 | x^{\tau}_T, s_0, d^{\tau}_0)$ is conditioned on the latent $x^{\tau}_T$ and the factors $s_0$ and $d^{\tau}_0$.

The posterior distribution reads:
$p(x^{1:V}_t, s_0, d^{1:V}_0 | x^{1:V}_0) = p_{0t}(x^{1:V}_t | x^{1:V}_0) p(s_0 | x^{1:V}_0) \prod_{\tau=1}^V p(d^{\tau}_0 | d^{<\tau}_0, x^{\le \tau}_0)$

**Optimization Objective**:
Score matching is used to optimize the denoising parametric map $D_{\theta}$:
$\theta^* = \arg\min_{\theta} E_t \left\{ \lambda_t E \left[ \| D_{\theta} - \nabla_x \log p_{0t} \|_2^2 \right] \right\}$

**Decoder Parameterization**:
$\tilde{x}^{\tau}_0 := D_{\theta}(x^{\tau}_t, t, z^{\tau}_0) = c^{skip}_t x^{\tau}_t + c^{out}_t F_{\theta}(c^{in}_t x^{\tau}_t, z^{\tau}_0, c^{noise}_t)$

**Loss Function**:
The single unified loss term based on the optimization objective is:
$E_{t, x^{\tau}_t, z^{\tau}_0, x^{\tau}_0} \left[ \lambda_t (c^{out}_t)^2 \| F_{\theta} - \frac{1}{c^{out}_t} (x^{\tau}_0 - c^{skip}_t \cdot x^{\tau}_t) \|_2^2 \right]$

**Diffusion SDEs (Background)**:
Forward process:
$dx_t = f(x_t, t)dt + g(t)dw$
Conditioned reverse process:
$dx_t = [f(x_t, t) - g(t)^2 \nabla_x \log p_t(x_t | u)]d\bar{t} + g(t)d\bar{w}$

**Prior Modeling Loss (Latent DDIM)**:
$\mathcal{L}_{latent} = \sum_{t=1}^T E_{z^{1:V}, \epsilon_t} \left[ \| \epsilon_{\phi}(z^{1:V}_t, t) - \epsilon_t \| \right]$
where $\epsilon_t \in \mathbb{R}^{dV+s} \sim \mathcal{N}(0, I)$.

**Multifactor PCA Exploration**:
$\bar{s} = \left( \frac{s - \mu_{\hat{s}}}{\sigma_{\hat{s}}} + \alpha v_i \cdot \sqrt{h} \right) \cdot \sigma_{\hat{s}} + \mu_{\hat{s}}$
where $\mu_{\hat{s}}$ and $\sigma^2_{\hat{s}}$ are the mean and variance of the sampled static features $\{\hat{s}_j\}_{j=1}^b$, $\{v_i\}_{i=1}^h$ are principal components, and $\alpha \in [-\kappa, \kappa]$.

## 8. Algorithm Pseudocode

**Algorithm 1: Conditioned Stochastic Sampler with $\sigma(t) = t$ and $s(t) = 1$.**
1: procedure ConditionedStochasticSampler($D_{\theta}, t_{i \in \{0,\dots,N\}}, \gamma_{i \in \{0,\dots,N-1\}}, z^{1:V}_0, x^{1:V}_0, S^2_{noise}$)
2: if $x^{1:V}_0 \neq \text{None}$ then
3: $x^{1:V}_N \leftarrow$ Algorithm 2 output
4: else
5: sample $x^{1:V}_N \sim \mathcal{N}(0, t^2_N I)$
6: for $i \in \{N, \ldots, 1\}$ do ▷ $\gamma_i = \begin{cases} \min(\frac{S_{churn}}{N}, \sqrt{2}-1) & \text{if } t_i \in [S_{tmin}, S_{tmax}] \\ 0 & \text{otherwise} \end{cases}$
7: sample $\epsilon_i \sim \mathcal{N}(0, S^2_{noise} I)$
8: $\hat{t}_i \leftarrow t_i + \gamma_i t_i$ ▷ Select temporarily increased noise level $\hat{t}_i$
9: $\hat{x}^{\tau}_i \leftarrow x^{\tau}_i + \sqrt{\hat{t}^2_i - t^2_i} \epsilon_i$ ▷ Add new noise to move from $t_i$ to $\hat{t}_i$
10: $d_i \leftarrow (x^{\tau}_i - D_{\theta}(x^{\tau}_i, z^{\tau}_0; \hat{t}_i)) / \hat{t}_i$ ▷ Evaluate $dx/dt$ at $t_i$
11: $x^{\tau}_{i-1} \leftarrow x^{\tau}_i + (t_{i-1} - \hat{t}_i) d_i$ ▷ Take Euler step from $t_i$ to $t_{i-1}$
12: if $t_{i-1} \neq 0$ then
13: $d'_i \leftarrow (x^{\tau}_{i-1} - D_{\theta}(x^{\tau}_{i-1}, z^{\tau}_0; t_{i-1})) / t_{i-1}$ ▷ Apply 2nd order correction
14: $x^{\tau}_{i-1} \leftarrow \hat{x}^{\tau}_i + (t_{i-1} - \hat{t}_i) (\frac{1}{2} d_i + \frac{1}{2} d'_i)$
15: return $x_0$

**Algorithm 2: Stochastic Encoding with $\sigma(t) = t$ and $s(t) = 1$.**
1: procedure StochasticEncoder($D_{\theta}, t_{i \in \{0,\dots,N\}}, \gamma_{i \in \{0,\dots,N-1\}}, x^{1:V}_0, z^{1:V}_0$)
2: sample $x_0 \sim \mathcal{N}(0, t^2_0 I)$
3: for $i \in \{0, \ldots, N - 1\}$ do ▷ $\gamma_i = \begin{cases} \min(\frac{S_{churn}}{N}, \sqrt{2}-1) & \text{if } t_i \in [S_{tmin}, S_{tmax}] \\ 0 & \text{otherwise} \end{cases}$
4: sample $\epsilon_i \sim \mathcal{N}(0, S^2_{noise} I)$
5: $\hat{t}_i \leftarrow t_i + \gamma_i t_i$ ▷ Select temporarily increased noise level $\hat{t}_i$
6: $\hat{x}_i \leftarrow x_i + \sqrt{\hat{t}^2_i - t^2_i} \epsilon_i$ ▷ Add new noise to move from $t_i$ to $\hat{t}_i$
7: $d_i \leftarrow (x^{\tau}_i - D_{\theta}(x^{\tau}_i, z^{\tau}_0; t_i)) / t_i$ ▷ Evaluate $dx^{\tau}/dt$ at $t_i$
8: $x^{\tau}_{i+1} \leftarrow x^{\tau}_i + (t_{i+1} - t_i) d_i$ ▷ Take Euler step from $t_i$ to $t_{i+1}$
9: if $t_{i+1} \neq \sigma_{max}$ then
10: $d'_i \leftarrow (x^{\tau}_{i+1} - D_{\theta}(x^{\tau}_{i+1}, z^{\tau}_0; t_{i+1})) / t_{i+1}$ ▷ Apply 2nd order correction
11: $x^{\tau}_{i+1} \leftarrow x^{\tau}_i + (t_{i+1} - t_i) (\frac{1}{2} d_i + \frac{1}{2} d'_i)$
12: return $x^{1:V}_N$