**1. Research Background and Existing Pain Points**

Diffusion models have rapidly emerged as the dominant class of generative models, achieving unprecedented scalability, controllability, and fidelity by iteratively denoising random noise. However, their empirical success raises a fundamental question: in principle, the standard training objective (e.g., denoising score matching) admits a closed-form solution that merely memorizes training examples; in practice, however, real-world models consistently produce novel and diverse outputs. This distinct mismatch between theoretical expectation and observed behavior poses a critical gap in our understanding of diffusion model generalization, with direct implications for privacy, interpretability, and trustworthy deployment.

Addressing this question has drawn increasing attention, yet existing explanations remain far from satisfactory. Early works based on random feature models provide useful insights but necessarily oversimplify model architectures. Analyses of linear models on Gaussian mixtures shed light on generalization but cannot capture memorization. Another line of research explores inductive biases by constructing handcrafted closed-form solutions from empirical data to approximate U-Net performance, attributing success to principles such as locality and equivariance. While these advances are valuable, the findings remain fragmented and phenomenological, and a more unified account of how diffusion models both memorize and generalize is still lacking.

**2. Core Research Motivation and Scientific Questions**

The core research motivation is to develop a unified mathematical framework based on a theoretical analysis of a nonlinear two-layer ReLU denoising autoencoder (DAE) to characterize both memorization and generalization. This framework aims to unify the characterization of memorization and generalization while bridging distribution learning with representation learning, offering profound practical implications.

The primary scientific questions addressed are:
1. When does a parameterized network overfit (learns the empirical denoiser) versus generalize (learns the population denoiser)?
2. What is the explicit connection between the internal representation structures of diffusion models and their generalization or memorization behaviors?
3. How can theoretical insights into representation spaces be leveraged for practical applications such as memorization detection and controllable generation?

**3. Overall Core Idea and Design Philosophy**

The overall core idea is to establish a representation-centric perspective on generalization, highlighting the pivotal role of bottleneck activations in DAE networks. The design philosophy posits that the representation space is not a byproduct but a crucial and controllable factor for generation. Specifically:
- **Memorization** corresponds to the model storing raw training datasets in the learned weights for encoding and decoding, yielding localized, spiky representations where memorized samples are encoded as spiky activations concentrated on a few neurons.
- **Generalization** arises when the model captures local data statistics, producing balanced representations that reflect the underlying distribution and spread energy across multiple coordinates.

**4. Core Innovation Points**

1. **Unified framework in a nonlinear ReLU setting**: Analyzes the optimal solutions of a two-layer nonlinear ReLU DAE under different empirical data sizes, providing a unified characterization of memorization and generalization that goes beyond prior random-feature or linear model analyses.
2. **A representation-centric understanding of generalization**: Establishes a rigorous connection between representation structures and generalization, identifying distinct patterns (spiky vs. balanced) that separate memorization from generalization and validating these insights across diverse model settings.
3. **Theory-inspired tools for memorization detection**: Proposes a simple yet effective representation-based method for detecting memorization using the spikiness of representations as a signature, achieving highly accurate and efficient performance in a prompt-free manner.
4. **Theory-inspired tools for model steering**: Proposes a training-free editing technique (representation steering) that allows precise control via additions in the representation space, revealing that generalized samples are highly steerable whereas memorized samples exhibit minimal editing effects.

**5. Overview of the Overall Technical Solution**

The technical solution begins by parameterizing the DAE as a two-layer ReLU network trained with an $\ell_2$-regularized objective on data assumed to follow a Mixture of Gaussians (MoG). By characterizing the local minimizers of this training loss under a defined $(\alpha, \beta)$-separability condition, the solution derives a block-wise structure for the optimal weights. Depending on the parameterization regime (overparameterized vs. underparameterized) and data abundance, the solution specializes into three cases: Memorization Regime (overparameterized, locally sparse data storing individual samples), Generalization Regime (underparameterized, locally abundant data capturing data statistics), and Hybrid Regime (imbalanced data mixing both). Based on the representation signatures (spiky for memorized, balanced for generalized), two practical applications are constructed: a representation standard deviation-based memorization detector, and a representation-space steering method for image editing.

**6. Detailed Module Design**

**6.1 Denoising Autoencoder (DAE) Network Module**
The DAE is parameterized by a two-layer ReLU network:
$f_{W_2,W_1}(x) = W_2h(x) = W_2 [W_1^\top x]_+$
where $W_1, W_2 \in \mathbb{R}^{d \times p}$, $[\cdot]_+$ denotes ReLU, and $h(\cdot)$ stands for the representation.

**6.2 Training Objective Module**
The $\ell_2$-regularized training objective is:
$\min_{W_2,W_1} \mathcal{L}_X(W_2,W_1) = \frac{1}{n} \sum_{i=1}^n \mathbb{E}_{\epsilon \sim \mathcal{N}(0,I)} [\|f_{W_2,W_1}(x_i + \sigma \epsilon) - x_i\|_2^2] + \frac{\lambda}{2} \sum_{l=1}^2 \|W_l\|_F^2$

**6.3 Memorization Regime Module**
In the overparameterized setting ($p \ge n$), each training sample is treated as an individual cluster. The local minimizer exhibits a memorizing block-wise structure where the weights store individual training samples. The representation for a training sample $x_i$ exhibits a distinctive sparse form:
$h_{mem}(x_i + \sigma \epsilon) = [W_{mem}^\top(x_i + \sigma \epsilon)]_+ \approx (0, \ldots, 0, r_i x_i^\top(x_i + \sigma \epsilon), 0, \ldots, 0)$.

**6.4 Generalization Regime Module**
In the underparameterized setting ($p = \sum_{k=1}^K p_k \ll n$), the weights effectively capture local data statistics. The representation spreads the energy of $x_i + \sigma \epsilon$ across the $p_k$ coordinates of the active block:
$h_{gen}(x_i + \sigma \epsilon) = [W_{gen}^\top(x_i + \sigma \epsilon)]_+ \approx (0, \ldots, 0, W_{X_k,1}^\top(x_i + \sigma \epsilon), \ldots, W_{X_k,p_k}^\top(x_i + \sigma \epsilon), 0, \ldots, 0)$.

**6.5 Memorization Detection Module**
The detection module leverages the spikiness of data representations. The standard deviation of intermediate features serves as a proxy for spikiness. High variance indicates memorization; low variance corresponds to generalization. 

**6.6 Representation-Space Steering Module**
The steering function is defined as:
$f_\theta^{steered}(x_t, t, c) = g_\theta(h_\theta(x_t, t, c) + a v)$
where $v = \frac{1}{|S|} \sum_{\tilde{x} \in S} h_\theta(\tilde{x}_{\tilde{t}}, \tilde{t}, \bar{c})$.
Here, $S$ denotes samples from the target concept/style, $h_\theta$ and $g_\theta$ represent the encoder and decoder components, $a \in \mathbb{R}$ controls the steering strength, $c$ is the text prompt, and $\bar{c}$ denotes the desired concept/style prompt.

**7. All Mathematical Formulas and Symbol Definitions**

**Forward and Reverse Processes:**
$x_t = x_0 + \sigma_t \epsilon$ with $\epsilon \sim \mathcal{N}(0, I)$
$x_{t-1} = x_t - (\sigma_t - \sigma_{t-1})\sigma_t \nabla \log p_t(x_t)$

**Empirical Denoiser:**
$f_{emp}(y) = \mathbb{E}[x \mid x + \sigma_t \epsilon = y; x \sim p_{emp}] = \frac{\sum_{i=1}^n \mathcal{N}(y; x_i, \sigma_t^2 I) x_i}{\sum_{i=1}^n \mathcal{N}(y; x_i, \sigma_t^2 I)}$

**Data Distribution:**
$x \sim p_{gt} := \sum_{k=1}^K \rho_k \mathcal{N}(\mu_k, \Sigma_k), \quad \sum_{k=1}^K \rho_k = 1$

**Definition 3.1 (($\alpha, \beta$)-Separability of Training Data):**
Suppose the training dataset $X$ can be partitioned into $M$ clusters $X = [X_1, \ldots, X_M]$, where $X_k = [x_{k,1}, \ldots, x_{k,n_k}] \subseteq \mathbb{R}^d$ has mean $\bar{x}_k := \frac{1}{n_k} \sum_{j=1}^{n_k} x_{k,j}$. We say the dataset is $(\alpha, \beta)$-separable if
for all $k, j$ : $\frac{\|x_{k,j} - \bar{x}_k\|_2}{\|\bar{x}_k\|_2} \le \alpha$, and for all $k \neq \ell$ : $\frac{\langle \bar{x}_k, \bar{x}_\ell \rangle}{\|\bar{x}_k\|_2 \|\bar{x}_\ell\|_2} \le \beta$.

**Theorem 3.1 (Block-wise Structure of Local Minimizers in the DAE Loss):**
There exists a local minimizer with a block-wise structure, where the weights satisfy $W_2^\star = W_1^\star$ where
$W_1^\star = (W_{X_1} \, W_{X_2} \, \cdots \, W_{X_M}) + R(\sigma, \gamma)$.
$\|R(\sigma, \gamma)\|_F^2 \le C(e^{-c \gamma^2 / \sigma^2})$
$W_{X_k} = U_k^{(p_k)} (I + n_k \sigma^2 (\Lambda_k^{(p_k)})^{-1})^{-1/2} (I - n\lambda (\Lambda_k^{(p_k)})^{-1})^{1/2} O_k^\top$

**Corollary 3.2 (Memorization in Overparameterized DAEs):**
$W_2^\star = W_1^\star = (r_1 x_1 \, \cdots \, r_n x_n \, 0 \, \cdots \, 0) =: W_{mem}$
$r_i = \sqrt{\frac{\|x_i\|_2^2 - n\lambda}{\|x_i\|_4^2 + \sigma^2 \|x_i\|_2^2}}$
When $\lambda \to 0$, empirical loss: $\mathcal{L}_X(W_2^\star, W_1^\star) = \frac{1}{n} \sum_{i=1}^n \frac{\sigma^2 \|x_i\|_2^2}{\sigma^2 + \|x_i\|_2^2} < \sigma^2$

**Corollary 3.3 (Generalization in Underparameterized DAEs):**
Concentration:
$\bar{x}_k = \frac{1}{n_k} \sum_{i=1}^{n_k} x_{k,i} \approx \mu_k$
$\frac{1}{n_k} X_k X_k^\top \approx S_k := \mu_k \mu_k^\top + \Sigma_k$
$W_{X_k} W_{X_k}^\top \to [(S_k - \frac{\lambda}{\rho_k} I)(S_k + \sigma^2 I)^{-1}]_{\text{rank-}p_k \text{ approx}}$
Expected test loss when $\lambda \to 0$:
$\mathbb{E}_{X \sim p_{gt}} [\mathcal{L}_X(W_2^\star, W_1^\star)] \lesssim \sum_{k=1}^K \rho_k \left[ \sum_{j \le p_k} \text{eig}_j(S_k) \cdot \frac{\sigma^4}{(\text{eig}_j(S_k) + \sigma^2)^2} + \sum_{j > p_k} \text{eig}_j(S_k) + \frac{C_k p_k}{\sigma^2 n_k} \right]$

**Corollary 3.4 (DAE memorizes duplicates and generalizes on well-sampled modes):**
$W_2^\star = W_1^\star = (r_1 x_1 \, \cdots \, r_m x_m \, W_{X_{m+1}} \, \cdots \, W_{X_K})$

**Lemma D.1 (Global minimizers of Regularized LAE):**
$\hat{\mathcal{L}}_X(W_2, W_1) := \|W_2 W_1^\top X - X\|_F^2 + n\sigma^2 \|W_2 W_1^\top\|_F^2 + \lambda'(\|W_1\|_F^2 + \|W_2\|_F^2)$
$W_2^\star = W_1^\star = U^{(p)} (I + n\sigma^2 \Lambda^{-1}_{(p)})^{-1/2} (I - \lambda' \Lambda^{-1}_{(p)})^{1/2} O^\top := W_X$

**Theorem D.2:**
Decomposition: $\mathcal{L}_X(W_2, W_1) = \frac{1}{n} \sum_{k=1}^M \hat{\mathcal{L}}_{X_k}(W_{2,(k)}, W_{1,(k)}) + \varepsilon(\delta, \sigma, \gamma)$
$\varepsilon(\delta, \sigma, \gamma) \le C \left( \frac{\delta}{\gamma} + e^{-c \gamma^2 / \sigma^2} \right)$
Margin $\gamma := \min_k \|\bar{x}_k\|_2 \cdot \min \left\{ \frac{s_{min} c_{proj}(\alpha)}{\sqrt{p_k}} - s_{max} \alpha, \frac{s_{max} |\beta|}{2} \right\} > 0$
Distance bound for any local minimizer:
$\|(\hat{W}_2, \hat{W}_1) - (W_2^\star, W_1^\star)\|_F \le \sqrt{\frac{2 \varepsilon(\delta, \sigma, \gamma)}{m_0}} \le \sqrt{\frac{2C}{c_0}} (\sigma^2 + n\lambda)^{-1/2} \left( \frac{\delta}{\gamma} + e^{-c \gamma^2 / \sigma^2} \right)^{1/2}$

**Corollary D.4 (Sparse solution smoothness):**
Hessian w.r.t $W_1$: $\nabla^2_{W_1} \mathcal{L}_X(W_2, W_1) = \text{blkdiag}(H(x_1) + \lambda I, \ldots, H(x_n) + \lambda I, \lambda I, \ldots, \lambda I)$
$H(x_i) = a_i x_i x_i^\top, a_i > 0$
$\ell_\infty$ Lipschitz constant: $L_\infty = \|\nabla^2_{W_1} \mathcal{L}_X(W_{mem}, W_{mem})\|_{\infty \to \infty} = \max \{ \max_{1 \le i \le n} \|H(x_i) + \lambda I\|_{\infty \to \infty}, \lambda \}$

**Corollary D.5 (Generalization error explicit form):**
Deviation: $\|\hat{D}_k - f_{p_k}(S_k)\|_F \lesssim \|S_k\|_{op} \left( \frac{1}{\sigma^2} + \frac{1}{\delta_{p_k}} \right) \frac{\sqrt{p_k} (r_{eff,k} + t)}{n_k}$
Generalization loss on cluster $k$:
$\mathbb{E}[\|\hat{D}_k(x' + \sigma \epsilon) - x'\|_2^2] \le \mathcal{L}_k^{pop}(p_k) + C \|S_k\|_{op}^3 \left( \frac{1}{\sigma^2} + \frac{1}{\delta_{p_k}} \right)^2 \frac{p_k (r_{eff,k} + t)}{n_k}$

**8. Algorithm Pseudocode**

**Algorithm 1: Detection via representation standard deviation (STD)**
Input: generated image $x_0$, timestep $t$, threshold THRES
Output: intermediate representation $h$, detection flag $I_{mem}$
$ x_t \leftarrow \text{ADDFORWARDNOISE}(x_0, t); $
$ h \leftarrow h_\theta(x_t, t, \text{condition} = \emptyset); $
where $f_\theta(x_t, t) = g_\theta[h_\theta(x_t, t, \emptyset)]$ with $g$ and $h$ the decoder/encoder components;
$ I_{mem} \leftarrow (\text{STD}(h) > \text{THRES}); $
return $h, I_{mem}$;