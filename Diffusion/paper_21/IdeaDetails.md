**1. Research Background and Existing Pain Points**

Out-of-distribution (OOD) detection is critical for the safe deployment of machine learning systems in safety-sensitive domains such as medical imaging and autonomous driving. Most machine learning systems assume that test data matches the training distribution, but distribution shift or OOD data can severely degrade performance. OOD inputs may stem from sensor noise, semantic differences, or acquisition changes, leading to unreliable predictions. 

Existing OOD detection methods can be broadly categorized into four families: (i) uncertainty-based methods, which rely on signals such as softmax confidence, ensemble variance, or Bayesian inference; (ii) distance-based methods, which compare test embeddings to in-distribution features; (iii) density-based methods, including flow- and energy-based models, which attempt to estimate likelihoods but have been shown to assign spuriously high likelihoods to OOD data; and (iv) representation-learning methods.

Diffusion models have recently emerged as powerful generative models, capable of capturing complex data distributions through iterative denoising. Early work exploited their properties through likelihood- or reconstruction-based scores. However, existing diffusion-based OOD metrics face significant pain points:
1. **Likelihood-based metrics are unreliable**: Diffusion models often emphasize low-level statistics while ignoring higher-level semantics, making negative log-likelihood (NLL) unreliable. NLL can even assign higher likelihoods to OOD samples than to InD ones.
2. **Score dynamics are insufficient**: Metrics using the score function norm and its temporal derivative provide some empirical separation but remain unstable. In near-OOD settings (e.g., CIFAR-10 vs CIFAR-100), the distributions overlap substantially. In other settings, the ordering can invert, with OOD samples receiving lower scores than InD.
3. **Reconstruction-based methods fail in near-OOD settings**: Methods like DDPM-OOD implicitly assume that distribution shift manifests primarily through degraded reconstruction quality. This strategy fails when OOD inputs admit visually plausible reconstructions, which is common in near-OOD settings.
4. **Scalarized uncertainty collapses informative structure**: Scalar metrics such as MSE or score norms collapse uncertainty into a single number, discarding how variance is distributed across directions. At high noise levels, this collapse is particularly problematic as many small eigenvalues are dominated by isotropic noise, obscuring class-specific structure.

**2. Core Research Motivation and Scientific Questions**

The core research motivation is to find a consistent, interpretable, and theoretically grounded signal of distribution shift in diffusion models that overcomes the limitations of heuristic scoring rules, likelihood paradoxes, and scalarized metrics. 

The scientific questions addressed are: How can we quantify the predictive uncertainty of the denoising process in a way that reliably separates InD from OOD samples? Does the spectral structure of the posterior covariance provide a consistent marker of distribution shift that remains robust even in challenging near-OOD scenarios where likelihoods and score dynamics fail? How can we efficiently estimate this spectral structure without expensive Jacobian computations?

**3. Overall Core Idea and Design Philosophy**

The overall core idea is that posterior covariance provides a consistent signal of distribution shift, leading to larger trace and leading eigenvalues on OOD inputs, yielding a clear spectral signature. EigenScore leverages the covariance structure of the denoising diffusion process to capture uncertainty signals. Unlike reconstruction-based methods that measure input-output similarity or trajectory-based methods that analyze diffusion-path geometry, EigenScore probes the posterior uncertainty of the denoiser by analyzing the eigenvalue spectrum of the posterior covariance.

The design philosophy is based on explicitly linking posterior covariance, estimated from the denoiser’s Jacobian, to distribution mismatch. When a diffusion model trained on InD data is applied to OOD inputs, the variance of its denoising predictions inflates, leaving a characteristic signature in the eigenvalue spectrum of the score Jacobian. By retaining only the top-K eigenvalues, EigenScore preserves the most informative uncertainty directions while discarding noise-dominated components, ensuring stable and discriminative OOD detection even when reconstruction error alone becomes unreliable.

**4. Core Innovation Points**

1. **Introduction of EigenScore**: A novel unsupervised, feature-based framework for OOD detection in diffusion models that leverages the posterior covariance of the denoising process to characterize distribution shift.
2. **Theoretical Link between Uncertainty and Distribution Mismatch**: Providing supporting analysis establishing a direct connection between denoising uncertainty from posterior covariance and distribution mismatch, thereby explaining why EigenScore reliably separates InD from OOD samples.
3. **Spectral Inflation as an Indicator**: Demonstrating that applying an in-distribution diffusion model to OOD samples leads to systematic inflation of posterior covariance, and that the leading eigenvalues provide a discriminative signal that overcomes the limitations of likelihood and scalarized MSE.
4. **Efficient Estimation via Jacobian-free Subspace Iteration**: Adopting a Jacobian-free subspace iteration method to estimate the leading eigenvalues using only forward evaluations of the denoiser, ensuring tractability at scale.

**5. Overview of the Overall Technical Solution**

The overall technical solution operates as follows: Given a trained InD diffusion model and a test sample, the method simulates the forward diffusion process to generate noisy states. For each noisy state, it estimates the top-K eigenvalues of the posterior covariance matrix, which is equivalent to the scaled Jacobian of the MMSE denoiser. Instead of computing the Jacobian explicitly, it uses a Jacobian-free subspace iteration method that approximates Jacobian-vector products with finite differences of the denoiser output. The sum of these top-K eigenvalues is computed across multiple noise realizations and aggregated across selected timesteps to form a feature vector. This feature vector is then normalized using Z-scores computed on the training set, and the final OOD score is the sum of these normalized coordinates.

**6. Detailed Module Design**

**Module 1: Linking Distribution Shift to Denoising Error**
This module establishes that distribution shift manifests as excess denoising error. For MMSE denoisers $D_p(x_t) = E_p[x|x_t]$ and $D_q(x_t) = E_q[x|x_t]$, the KL divergence between InD $p$ and OOD $q$ is restated in terms of denoising error. This ensures that the denoising error of a single InD denoiser $D_p$ is larger in expectation on OOD inputs, yielding a practical detection signal without access to $q$.

**Module 2: Posterior Covariance and Eigen-spectrum Analysis**
This module connects denoising error to the posterior covariance. The mean-squared error admits a law-of-total-variance decomposition, showing that denoising error equals the total posterior variance—the trace of the conditional covariance—averaged over noisy observations. Using the Miyasawa identity, the posterior covariance is expressed through the Jacobian of the MMSE denoiser. The MSE corresponds to the sum of eigenvalues—the total uncertainty across all principal directions at noise level $t$. For InD samples, the spectrum is compact with smaller leading eigenvalues; for OOD samples, uncertainty spreads across multiple eigen-directions, inflating both the spectrum and the trace.

**Module 3: EigenScore Metric Formulation**
This module defines the EigenScore feature. For each input $x$ and diffusion step $t$, it computes the sum of the top-K eigenvalues across $I$ noise realizations. These are aggregated into a feature vector $M(x)$. Each coordinate is normalized using Z-scores with statistics $(\mu_t, \sigma_t)$ computed on the training set. The final OOD score is the sum of normalized coordinates.

**Module 4: Spectral Truncation vs. Scalar Collapse**
This module handles the issue of isotropic noise at high noise levels. Lemma 1 shows that the spectrum flattens under heavy Gaussian smoothing, causing MSE and score norms to aggregate mostly isotropic noise. By retaining only the top-K eigenvalues, EigenScore preserves the most informative uncertainty directions. Ky Fan's theorem guarantees that the top-K eigenvalues capture the maximal variance among all K-dimensional projections.

**7. All Mathematical Formulas and Symbol Definitions**

**Symbol Definitions:**
- $p(x)$: InD distribution
- $q(x)$: OOD distribution
- $x_t$: Noisy state at timestep $t$
- $z \sim \mathcal{N}(0, \sigma_t^2 I)$: Gaussian noise
- $D_\theta(x_t, t)$: Denoising network
- $D_p(x_t) = E_p[x|x_t]$: MMSE estimator trained on distribution $p$
- $\sigma_t$: Noise level standard deviation
- $p(x_t)$: Marginal distribution of noisy image
- $G_{\sigma_t}$: Gaussian density function with standard deviation $\sigma_t$
- $\text{Cov}_p[x|x_t]$: Posterior covariance
- $\Sigma_t(x_t)$: Covariance matrix equivalent to $\text{Cov}_p[x|x_t]$
- $U_t$: Matrix of eigenvectors
- $\lambda_k^t$: Eigenvalues of the covariance matrix
- $K$: Number of top eigenvalues retained
- $I$: Number of noise realizations
- $T$: Number of timesteps
- $c$: Linear approximation constant for finite differences

**Formulas:**

Forward Markov chain:
$$p(x_t|x) = \mathcal{N}(x, \sigma_t^2 I)$$

Direct sampling:
$$x_t = x + z, \quad z \sim \mathcal{N}(0, \sigma_t^2 I)$$

Training objective (MSE):
$$L_{\text{MSE}}(D_\theta) = \mathbb{E}_{x, x_t, t} \left[ \| x - D_\theta(x_t, t) \|_2^2 \right]$$

Tweedie's formula:
$$D_p(x_t) = \mathbb{E}_p[x|x_t] = x_t + \sigma_t^2 \nabla \log p(x_t)$$

Marginal distribution of noisy image:
$$p(x_t) = \int p(x_t|x) p(x) dx = \int G_{\sigma_t}(x_t - x) p(x) dx$$

KL divergence score-based representation:
$$D_{\text{KL}}(p \| q) = \int_0^T \mathbb{E}_{x, x_t} \left[ \| \nabla \log p(x_t) - \nabla \log q(x_t) \|_2^2 \right] \sigma_t dt$$

Proposition 1 (KL divergence in terms of denoising error):
$$D_{\text{KL}}(p \| q) = \int_0^T \left[ \text{MSE}(D_q, t) - \text{MSE}(D_p, t) \right] \sigma_t^{-3} dt$$
where $\text{MSE}(D_p, t) = \mathbb{E} \left[ \| x - D_p(x_t) \|_2^2 \right]$ at noise level $t$.

MSE as total posterior variance:
$$\text{MSE}(D_p, t) = \mathbb{E} \left[ \| x - D_p(x_t) \|_2^2 \right] = \mathbb{E}_{x_t} \left[ \text{tr} \left( \text{Cov}_p[x | x_t] \right) \right]$$

Miyasawa identity for posterior covariance:
$$\text{Cov}_p[x|x_t] = \sigma_t^2 \left( I + \sigma_t^2 \nabla^2 \log p(x_t) \right) = \sigma_t^2 \nabla D_p(x_t)$$

MSE as trace of Jacobian:
$$\text{MSE}(D_p, t) = \mathbb{E}_{x_t} \left[ \text{tr} \left( \text{Cov}_p[x|x_t] \right) \right] = \sigma_t^2 \mathbb{E}_{x_t} \left[ \text{tr} \left( \nabla D_p(x_t) \right) \right]$$

Eigen-decomposition of covariance:
$$\Sigma_t(x_t) := \text{Cov}_p[x|x_t] = \sigma_t^2 \nabla D_p(x_t) = U_t \text{diag}(\lambda_1^t, \cdots, \lambda_n^t) U_t^T$$

MSE as sum of eigenvalues:
$$\text{MSE}(D_p, t) = \mathbb{E}_{x_t} \left[ \text{tr} \left( \text{Cov}_p[x|x_t] \right) \right] = \mathbb{E}_{x_t} \left[ \text{tr} \left( \Sigma_t(x_t) \right) \right] = \mathbb{E}_{x_t} \left[ \sum_{k=1}^n \lambda_k^t(x_t) \right]$$

EigenScore feature calculation for input $x$ and step $t$:
$$m_t(x) = \sum_{k=1}^K \lambda_k^t(x_t), \quad x_t = x + \sigma_t z, \quad z \sim \mathcal{N}(0, I)$$

EigenScore feature vector:
$$M(x) = \left[ m_1(x), \ldots, m_T(x) \right]^T$$

Z-score normalization:
$$z_t(x) = \frac{m_t(x) - \mu_t}{\sigma_t}$$

Final OOD score:
$$S_\theta(x) = \sum_{t=1}^{T^*} z_t(x)$$

Jacobian-free eigenvalue approximation:
$$\lambda_k^t(x_t) \approx \frac{\sigma_t^2}{2c} \left\| D(x_t + c v_k) - D(x_t - c v_k) \right\|$$

Lemma 1 (Spectrum flattening):
Let $p(x_t) = p * \mathcal{N}(0, \sigma_t^2 I)$ denotes the noisy marginal and let $\Sigma_t(x_t) = \sigma_t^2 \left( I + \sigma_t^2 \nabla^2 \log p(x_t) \right)$. As $\sigma_t \to \infty$, $\| \nabla^2 \log p(x_t) \| \to 0$ uniformly on compact sets. Hence
$$\Sigma_t(x_t) = \sigma_t^2 I + o(\sigma_t^2)$$
so all eigenvalues satisfy $\lambda_k^t(x_t) = \sigma_t^2 + o(\sigma_t^2)$.

Proposition 2 (Ky Fan’s theorem):
Let $\Sigma_t(x_t) \succeq 0$ have eigenvalues $\lambda_1^t \ge \cdots \ge \lambda_n^t$ with eigenvectors forming $U_t = [u_1^t, \ldots, u_n^t]$. For any $K \in \{1, \ldots, n\}$,
$$\max_{V \in \mathbb{R}^{n \times K}: V^T V = I_K} \text{tr} \left( V^T \Sigma_t(x_t) V \right) = \sum_{k=1}^K \lambda_k$$
and a maximizer is $V^\star = [u_1^t, \ldots, u_K^t]$.

**8. Algorithm Pseudocode**

**Algorithm 1: OOD Detection with EigenScore — Train/Validation (left) and Test (right)**

Train/Validation: Select time-steps, aggregation, and compute Z-score stats
Require: Trained DM $D_p$, train set $X_{\text{train}}$, $K$ number of eigenvalues, $I$ number of repetition, $L_{\text{train}} = [ ]$
1: for $x \in X_{\text{train}}$ do
2: Compute $M(x)$ via Eq. (10)
3: Append $M(x)$ to $L_{\text{train}}$
4: end for
5: Compute $\mu_t, \sigma_t$ across $L_{\text{train}}$
6: Use validation set to tune $T$ and aggregation method (mean/median/none)
7: return $(T^\star, \text{agg}^\star, \{\mu_t, \sigma_t\}_{t=1}^{T^\star})$

Test: Compute EigenScore
Require: Trained DM $D_p$, test set $X_{\text{test}}$, number of eigenvalues $K$, number of repetitions $I$, $(T^\star, \text{agg}^\star, \{\mu_t, \sigma_t\}_{t=1}^{T^\star})$, $L_{\text{test}} = [ ]$
1: for $x \in X_{\text{test}}$ do
2: Compute $M(x)$ using $T^\star$ and $\text{agg}^\star$
3: $z_t(x) = \frac{m_t(x) - \mu_t}{\sigma_t}$ for $t = 1, \ldots, T^\star$
4: $S_\theta(x) = \sum_{t=1}^{T^\star} z_t(x)$
5: Append $S_\theta(x)$ to $L_{\text{test}}$
6: end for
7: return $L_{\text{test}}$ ▷ OOD scores for all test samples

**Algorithm 2: Efficient computation of posterior principal components**

Require: $N$ (Number of PCs), $K$ (number of iterations), $D(\cdot)$ (MSE-optimal denoiser), $y$ (noisy input), $\sigma_t^2$ (noise variance), $c \ll 1$ (linear approx. constant)
1: Initialize $\{v_0^{(i)}\}_{i=1}^N \sim \mathcal{N}(0, \sigma_t^2 I)$
2: for $k \leftarrow 1$ to $K$ do
3: for $i \leftarrow 1$ to $N$ do
4: $v_k^{(i)} \leftarrow \frac{1}{2c} \left( D(y + c v_{k-1}^{(i)}) - D(y - c v_{k-1}^{(i)}) \right)$
5: end for
6: $Q, R \leftarrow \text{QR DECOMPOSITION} \left( [v_k^{(1)} \cdots v_k^{(N)}] \right)$
7: $[v_k^{(1)} \cdots v_k^{(N)}] \leftarrow Q$
8: end for
9: $v^{(i)} \leftarrow v_K^{(i)}$
10: $\lambda^{(i)} \leftarrow \frac{\sigma_t^2}{2c} \left\| D(y + c v_{K-1}^{(i)}) - D(y - c v_{K-1}^{(i)}) \right\|$