### 1. Research Background and Existing Pain Points

Recent diffusion models demonstrate remarkable sample efficiency and fast optimization, contradicting standard estimation bounds that suffer from the curse of dimensionality $n^{-1/D}$ with the data dimension $D$. Since images are usually a union of low-dimensional manifolds, current works model the data as a union of linear subspaces with Gaussian latent and achieve a $1/\sqrt{n}$ bound. Though this modeling reflects the multi-manifold property, the Gaussian latent can not capture the multi-modal property of the latent manifold. As shown in Brown et al. (2023) and Kamkari et al. (2024), though the image dataset admits low dimension, it is a union of manifolds instead of one manifold. Wang et al. (2024) models the image data as a union of linear subspaces, assume each subspace admits a low-dimensional Gaussian (mixture of low-rank Gaussians (MoLRG)), and achieve a $1/\sqrt{n}$ estimation error. Though the union of the linear subspace is closer to the real-world image dataset, the latent Gaussian assumption is far away from the low-dimensional multi-modal manifold. The existing pain point is that the previous multi-subspace modeling (MoLRG) assumes a Gaussian latent with 0 mean, which can not capture the multi-modal property of real-world data.

### 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:** To bridge the gap between the multi-manifold property and the multi-modal property of real-world data, providing a modeling that reflects the multi-manifold multi-modal property and theoretically explaining why diffusion models only require a small training sample and enjoy a fast optimization process.

**Scientific Questions:**
1. Can we propose a modeling that reflects the multi-manifold multi-modal property of real-world data?
2. Can we escape the curse of dimensionality and enjoy a fast convergence rate based on this modeling?

### 3. Overall Core Idea and Design Philosophy

The overall core idea is to propose the mixture subspace of low-rank mixture of Gaussian (MoLR-MoG) modeling, which models the target data as a union of K linear subspaces, and each subspace admits a mixture of Gaussian latent ($n_k$ modals with dimension $d_k$). With this modeling, the corresponding score function naturally has a mixture of expert (MoE) structure, captures the multi-modal information, and contains nonlinear property. The design philosophy is to leverage the union of low-dimensional linear subspace and the latent MoG property to escape the curse of dimensionality and prove that the landscape around the ground truth parameter is strongly convex to guarantee fast convergence.

### 4. Core Innovation Points

1. **MoLR-MoG Modeling and MoE Structure Nonlinear Score:** Proposing the MoLR-MoG modeling for the target data, which captures the multi low-dimensional manifold and multi-modal property of real-world data and naturally introduces the MoE-latent MoG score.
2. **Take Advantage of MoLR-MoG to Escape the Curse of Dimensionality:** Showing that by taking advantage of the union of a low-dimensional linear subspace and the latent MoG property, diffusion models escape the curse of dimensionality, achieving the estimation error bound of $R^4\sqrt{\sum_{k=1}^K n_k}\sqrt{\sum_{k=1}^K n_k d_k}/\sqrt{n}$.
3. **Strongly Convex Property and Convergence Guarantee:** Studying the optimization of the highly non-convex score-matching objective function. Using the gradient descent (GD) algorithm and showing that the landscape around the ground truth parameter is strongly convex, proving the convergence guarantee when considering MoLR-MoG.
4. **Extension to MoG Latent Without Separation Assumption:** Extending the analysis to latent MoG with overlap by defining the pairwise overlap factor and maximum expected overlap, treating off-diagonal interference as a perturbation, and proving that the global matrix remains positive definite provided the perturbation is smaller than the effective diagonal curvature.
5. **Empirical Validation of MoLR-MoG Suitability:** Demonstrating that the MoE-latent MoG network significantly outperforms MoLRG Gaussian baselines and matches MoE-latent U-Net performance with $10\times$ fewer parameters, validating its practical suitability and the necessity of expert-specific VAEs.

### 5. Overview of the Overall Technical Solution

The technical solution consists of two main parts: Estimation Error Analysis and Optimization Analysis.
1.  **Estimation Error Analysis:** First, the MoLR-MoG distribution is defined, and the corresponding MoE-nonlinear MoG score function is derived. Under the bounded-support assumption, the Lipschitz continuity of the score function and loss function is established. By controlling the Rademacher complexity of the loss class and using a Bernstein concentration argument, the estimation error bound is obtained, successfully escaping the dimensionality curse.
2.  **Optimization Analysis:** The score-matching objective function is analyzed. For 2-modal MoG latent, under sufficient cluster separation, the Jacobian simplification and Hessian block structure are derived, yielding local strong convexity and linear convergence. This is extended to general MoG latent under a highly separated Gaussian assumption and equivalent Gaussian approximation. Finally, without the high-separation assumption, the overlap is quantified, and perturbation analysis is used to prove the Hessian remains positive definite for robust convergence.

### 6. Detailed Module Design

**Module 1: MoLR-MoG Modeling Module**
The data distribution lives near a union of K linear subspaces. For the k-th subspace (represented by a matrix $A^*_k \in \mathbb{R}^{D \times d_k}$ with orthonormal columns), a $n_k$-modal MoG is placed within that subspace:
$w_k(x) = \sum_{l=1}^{n_k} \pi_{k,l} \mathcal{N}(x; A^*_k \mu^*_{k,l}, A^*_k \Sigma^*_{k,l} A^{*\top}_k)$
where $\Sigma^*_{k,l} = U^*_{k,l} U^{*\top}_{k,l}$, $U^*_{k,l} \in \mathbb{R}^{d_k \times d_{k,l}}$ and $\mu^*_{k,l}$ is the mean.

**Module 2: MoE-Latent Nonlinear Score Module**
The linear encoder $A_k$ first encodes images to the k-th manifold, and diffusion models run the denoising process. After that, the linear decoder $A^\top_k$ decodes the denoised latent to the full-dimensional images. For the k-th low-dimensional manifold, the score function is:
$\nabla \log p_{t,k}(x_{LD}) = - \frac{1}{\gamma_t^2} \frac{\sum_{l=1}^{n_k} \pi_{k,l} \mathcal{N}(x_{LD}; s_t \mu^*_{k,l}, \Sigma^*_{k,l,t}) \delta_{k,l,t}(x_{LD})}{\sum_{l=1}^{n_k} \pi_{k,l} \mathcal{N}(x; s_t \mu^*_{k,l}, \Sigma^*_{k,l,t})}$
The parameters to be learned are $\theta^* = \{\mu^*_{k,l}, U^*_{k,l}\}_{k=1,...,K}$.

**Module 3: Estimation Error Module**
Based on Assumption 5.1 ($\|x\|_2 \le R$), the per-sample squared error is $\ell(\theta; x, t) = \|s_\theta(x, t) - s^*(x, t)\|_2^2$. The empirical loss is $\hat{\mathcal{L}}_n(\theta) = \frac{1}{n} \sum_{i=1}^n \ell(\theta; x_i, t)$. By establishing the Lipschitz constant and Rademacher complexity, the bound between population loss and empirical loss is derived.

**Module 4: 2-Modal Latent MoG Optimization Module**
For a latent 2-modal MoG with the same covariance matrix $\Sigma^*_k$ and $\mu^*_{k,1} = \mu^*_k$, $\mu^*_{k,2} = -\mu^*_k$, under Assumption 6.1 (Separation within a cluster: $\|s_t \mu^*_k - (-s_t \mu^*_k)\| \ge \Delta_{intra} \gg \gamma_t$), the responsibility $r^-_k(x) = O(e^{-\Delta_{intra}^2/(2\gamma_t^2)}) \ll 1$. The Jacobian simplifies to "self-cluster" terms. The Hessian matrix near $\theta^*$ simplifies to a block-diagonal form, yielding local strong convexity with parameter $\alpha$.

**Module 5: General MoG Latent Optimization Module**
For asymmetric Gaussian mixture, under Assumption 6.4 (Highly Separated Gaussian), the individual Gaussian distributions are highly separated. The equivalent Gaussian approximation (Corollary 6.5) is used. The Hessian at the k-th subspace is convex on a neighborhood of $\theta^*$. The local strong convexity parameter $\alpha'$ is the minimum eigenvalue of the Hessian.

**Module 6: Overlap Robustness Module**
Without the high-separation assumption, the pairwise overlap factor is defined as $\xi_{i,j}(x) \triangleq r_{k,i}(x) r_{k,j}(x)$ and maximum expected overlap as $\epsilon_{overlap} = \max_i \sum_{j \neq i} \mathbb{E}_{x \sim p_t}[\xi_{i,j}(x)]$. The full Hessian is written as $H = H_{diag} + \Delta H_{overlap}$. Using Weyl's Inequality, the minimum eigenvalue of the full Hessian is bounded, guaranteeing positive definiteness if $\epsilon_{overlap} < \frac{\lambda_{base}}{\tilde{C}}$.

### 7. All Mathematical Formulas and Symbol Definitions

**Forward Process:**
$dx_t = f(t)x_t dt + g(t) dB_t$
$p_t(x_t|x_0) = \mathcal{N}(x_t; s_t x_0, s_t^2 \sigma_t^2 I_D)$
$s_t = \exp(\int_0^t f(\xi)d\xi)$, $\sigma_t = \sqrt{\int_0^t g^2(\xi)/s^2(\xi)d\xi}$

**Reverse Process:**
$dy_t = [f(t)y_t - g(t)^2 \nabla \log p_t(y_t)] dt + g(t) d\bar{B}_t$

**Score Matching Objective:**
$\min_{s_\theta \in \mathcal{N}\mathcal{N}} \mathcal{L}_{SM} = \int_\delta^T \mathbb{E}_{x_t \sim q_t} \|\nabla \log p_t(x_t) - s_\theta(x_t, t)\|_2^2 dt$

**Denoised Score Matching Objective:**
$\min_{s_\theta \in \mathcal{N}\mathcal{N}} \mathcal{L}_{DSM} = \int_\delta^T \mathbb{E}_{x_0 \sim q_0} \mathbb{E}_{x_t|x_0} \|\nabla \log p_t(x_t|x_0) - s_\theta(x_t, t)\|_2^2 dt$

**MoLR-MoG Distribution:**
$p_0 = \sum_{k=1}^K \frac{1}{K} \sum_{l=1}^{n_k} \pi_{k,l} \mathcal{N}(x; A^*_k \mu^*_{k,l}, A^*_k \Sigma^*_{k,l} A^{*\top}_k)$

**Full Score Function:**
$\nabla \log p_t(x) = - \frac{1}{\gamma_t^2} \frac{\sum_{k=1}^K \frac{1}{K} \sum_{l=1}^{n_k} \pi_{k,l} \mathcal{N}(x; s_t \mu^*_{k,l} A^*_k, A^*_k \Sigma^*_{k,l,t,A} A^{*\top}_k) \delta_{k,l,t,A}(x)}{\sum_{k=1}^K \frac{1}{K} \sum_{l=1}^{n_k} \pi_{k,l} \mathcal{N}(x; s_t \mu^*_{k,l} A^*_k, A^*_k \Sigma_{k,l,t,A} A^{*\top}_k)}$
Where $\gamma_t = s_t \sigma_t$, $\Sigma_{k,l,t,A} = s_t^2 A^*_k U^*_{k,l} U^{*\top}_{k,l} A^{*\top}_k + \gamma_t^2 I$, $\delta_{k,l,t,A}(x) = x - s_t \mu^*_{k,l} - \frac{s_t^2}{s_t^2+\gamma_t^2} A^*_k U^*_{k,l} U^{*\top}_{k,l} A^{*\top}_k (x - s_t \mu^*_{k,l} A^*_k)$

**Latent Score Function:**
$\nabla \log p_{t,k}(x_{LD}) = - \frac{1}{\gamma_t^2} \frac{\sum_{l=1}^{n_k} \pi_{k,l} \mathcal{N}(x_{LD}; s_t \mu^*_{k,l}, \Sigma^*_{k,l,t}) \delta_{k,l,t}(x_{LD})}{\sum_{l=1}^{n_k} \pi_{k,l} \mathcal{N}(x; s_t \mu^*_{k,l}, \Sigma^*_{k,l,t})}$
Where $\Sigma_{k,l,t} = s_t^2 U^*_{k,l} U^{*\top}_{k,l} + \gamma_t^2 I$, $\delta_{k,l,t}(x_{LD}) = x_{LD} - s_t \mu^*_{k,l} - \frac{s_t^2}{s_t^2+\gamma_t^2} U^*_{k,l} U^{*\top}_{k,l} (x_{LD} - s_t \mu^*_{k,l})$

**Lipschitz Constant:**
$L \le \sqrt{\sum_{i=1}^K n_k (L_{\mu_l}^2 + L_{U_k}^2)} = O((\sum_{k=1}^K n_k)^{1/2} C_w)$
$C_w = \frac{(R + s_t B_\mu)^3 s_t^2}{\gamma_t^4}$, $B_\mu = \max_{k,l} \|\mu_{k,l}\|_2$

**Estimation Error Bound:**
$|\mathcal{L}(\theta) - \hat{\mathcal{L}}_n(\theta)| \le O(\frac{C_1 (R + s_t B_\mu)^{4s_t^2} \sqrt{\sum_{k=1}^K n_k}}{\gamma_t^6} \frac{\sqrt{\sum_{k=1}^K n_k d_k}}{\sqrt{n}} + C_2 \sqrt{\frac{\log(1/\delta)}{n}})$

**2-Modal Latent Score:**
$\nabla \log p_{t,k}(x) = - \frac{1}{\gamma_t^2} \frac{\frac{1}{2} \mathcal{N}(x; s_t \mu^*_k, \Sigma^*_k) \delta'_k(x) + \frac{1}{2} \mathcal{N}(x; -s_t \mu^*_k, \Sigma^*_k) \epsilon_k(x)}{\frac{1}{2} \mathcal{N}(x; s_t \mu^*_k, \Sigma^*_k) + \frac{1}{2} (x; -s_t \mu^*_k, \Sigma^*_k)}$
$\epsilon_k(x) = x - s_t \mu^*_k - \frac{s_t^2}{s_t^2+\gamma_t^2} U^*_k U^{*\top}_k (x - s_t \mu^*_k)$
$\delta'_k(x) = x + s_t \mu^*_k - \frac{s_t^2}{s_t^2+\gamma_t^2} U^*_k U^{*\top}_k (x + s_t \mu^*_k)$

**Jacobian Simplification:**
$J^\mu_k(x) \approx s_t(I - \alpha P_k)/\gamma_t^2$
$J^U_k(x) \approx \frac{2 s_t^2}{\gamma_t^2 (s_t^2 + \gamma_t^2)} (r^-_k(x)(U^\top_k (x+s_t \mu_k)I + (x+s_t \mu_k)U^\top_k) + r^+_k(x)(U^\top_k (x-s_t \mu_k)I + U_k(x-s_t \mu_k)^\top))$

**Hessian Eigenvalues (2-Modal):**
$\lambda_{min}(H_{\mu_k \mu_k}) = \frac{s_t^2}{(s_t^2 + \gamma_t^2)^2}$
$\lambda_{min}(H_{U_k U_k}) = \frac{4(U^\top_k \mu_k))^2 + \|U_k\|_2^2 \|\mu_k\|_2^2 - \|U_k\|_2 \|\mu_k\|_2 \sqrt{8(U^\top_k \mu_k))^2 + \|U_k\|_2^2 \|\mu_k\|_2^2}}{2}$

**Local Strong Convexity Parameter (2-Modal):**
$\alpha = \min\{\frac{s_t^2}{(s_t^2 + \gamma_t^2)^2}, \frac{4(U^\top_k \mu_k))^2 + \|U_k\|_2^2 \|\mu_k\|_2^2 - \|U_k\|_2 \|\mu_k\|_2 \sqrt{8(U^\top_k \mu_k))^2 + \|U_k\|_2^2 \|\mu_k\|_2^2}}{2}\}$

**Overlap Factor:**
$\xi_{i,j}(x) \triangleq r_{k,i}(x) r_{k,j}(x)$
$\epsilon_{overlap} = \max_i \sum_{j \neq i} \mathbb{E}_{x \sim p_t}[\xi_{i,j}(x)]$

**Minimum Curvature for 2-Mode Mixture:**
$\alpha_{2-mode} \triangleq (1 - 4 \epsilon_{overlap}) \min(\lambda_{min}(H_{\mu_k \mu_k}), \lambda_{min}(H_{U_k U_k}))$
$\lambda_{min}(H) \ge \alpha_{2-mode} - C' \epsilon_{overlap} > 0$

**Minimum Curvature for Multi-Modal:**
$\alpha_{Multi-Modal} \triangleq \min_{l \in \{1,...,n_k\}} [(\pi_{k,l} - \epsilon^{total}_{k,l}) \min(\lambda_{min}(H_{\mu_{k,l} \mu_{k,l}}), \lambda_{min}(H_{U_{k,l} U_{k,l}}))]$
$\lambda_{min}(H) \ge \alpha_{Multi-Modal} - \tilde{C} \cdot \epsilon_{overlap}$

### 8. Algorithm Pseudocode

The paper does not provide explicit algorithm pseudocode blocks but defines the iterative update rule for the optimization process as follows:

**Iterative Update Rule:**
Using Gradient Descent (GD) algorithm to optimize the objective function.
Step size: $\eta_m = \eta = 2/(\eta + L')$
Condition number: $\kappa = L'/\alpha$
Convergence Rate:
$\|\theta^{(m)} - \theta^\star\|_2 \le (\frac{\kappa - 1}{\kappa + 1})^m \|\theta^{(0)} - \theta^\star\|_2$

**Extension without Highly Separated Condition:**
Effective condition number: $\kappa_{eff} = \frac{L}{\alpha - C' \epsilon_{overlap}}$
Linear Convergence under Bounded Overlap:
$\|\theta_t - \theta^\star\|_2 \le (\frac{\kappa_{eff} - 1}{\kappa_{eff} + 1})^t \|\theta^{(0)} - \theta^\star\|_2$