1.  **Research Background and Existing Pain Points**
    Diffusion models have achieved remarkable success across diverse domains such as image synthesis, molecular design, and time-series modeling, establishing themselves as a leading paradigm in modern generative modeling. However, they remain vulnerable to memorization—reproducing training data rather than generating novel outputs. This phenomenon undermines the creative potential of generative modeling, threatens the promise of generalization, and raises serious concerns about data privacy and intellectual property.
    Existing empirical studies have explored mitigation strategies such as data de-duplication and modified training objectives, providing valuable heuristics yet leaving principles underneath their success underexplored. In parallel, theoretical investigations have analyzed memorization from a statistical perspective, often using asymptotic analyses where both sample size and data dimension grow proportionally. However, these analyses do not fully explain memorization in practical, finite-sample regimes, leaving a fundamental gap: Can we disentangle memorization from generalization in practical regimes and mitigate it?

2.  **Core Research Motivation and Scientific Questions**
    The core motivation is to address the lack of theoretical understanding regarding memorization in diffusion models within practical, finite-sample regimes. The paper aims to develop a non-asymptotic analysis that theoretically explains the emergence of memorization through the dual lenses of statistical estimation and neural function approximation.
    The scientific questions are:
    *   Why does a sufficiently expressive neural network trained with a strong optimizer tend to learn the empirical score function (leading to memorization) instead of the ground-truth score function (leading to generalization)?
    *   Is there an inherent statistical gap that drives this behavior?
    *   What are the representation requirements (network size) for the ground-truth versus the empirical score function?
    *   How can these theoretical insights guide empirical mitigation strategies?

3.  **Overall Core Idea and Design Philosophy**
    The overall core idea is to establish a dual-separation result between memorization and generalization.
    *   **Statistical Separation:** The ground-truth score function does not minimize the empirical denoising score matching loss, creating an inherent gap (Loss-Gap) that drives memorization. This gap is equivalent to the Fisher divergence and does not vanish with polynomially many training samples in small-$t$ regimes.
    *   **Architectural Separation:** Implementing the empirical score function requires network size to scale with the sample size, whereas the ground-truth score function admits a compact neural representation.
    The design philosophy is that memorization is fundamentally tied to the statistical properties of the training objective and the approximation capacity of score neural networks. Strong optimizers drive sufficiently expressive networks toward the empirical score due to the statistical gap. Therefore, mitigation strategies should control network capacity or smoothness to prevent the network from fitting the empirical score.

4.  **Core Innovation Points**
    *   **Statistical Separation Theory:** Demonstrated that the denoising score matching loss admits an inherent gap between the ground-truth and empirical score functions. Quantified this discrepancy for generic sub-Gaussian mixture models, providing a formal characterization of how memorization arises statistically (Proposition 4.1, Theorem 4.3).
    *   **Neural Architectural Separation Theory:** Established bounds on neural networks approximating both ground-truth and empirical score functions. Proved that the ground-truth score function admits a compact neural representation, whereas approximating the empirical score function requires the network size to grow with the sample size $n$ (Theorem 5.1).
    *   **Lipschitz Continuity Separation:** Identified a separation in Lipschitz continuity; the empirical score function becomes highly irregular when $t$ approaches 0 (Lipschitz coefficient depends on sample separation), while the ground-truth score possesses better regularity. This supports the use of weight decay to mitigate memorization (Lemma C.1).
    *   **Theory-Driven Mitigation Strategy:** Guided by the architectural separation insight that smaller networks favor generalization, proposed a one-shot pruning method for Diffusion Transformers that identifies and prunes attention heads contributing least in the small-$t$ regime, reducing memorization while maintaining generation quality.

5.  **Overview of the Overall Technical Solution**
    The technical solution begins with the continuous-time formulation of diffusion models, defining the forward and backward processes and the denoising score matching loss. It introduces sub-Gaussian Hölder density assumptions for the data distribution.
    First, it defines the "Loss-Gap" between the empirical and ground-truth scores and proves it equals the Fisher divergence.
    Second, it quantifies this gap for mixture models with well-separated components, proving a lower bound that holds for polynomially many samples.
    Third, it analyzes the representation power of ReLU networks, establishing that approximating the empirical score requires parameters scaling with $n$, while the true score requires parameters scaling only with the desired approximation error.
    Finally, it applies these insights to develop a pruning algorithm for Diffusion Transformers (DiTs), calculating importance scores for attention heads specifically in the small-$t$ regime and pruning the least important heads followed by fine-tuning.

6.  **Detailed Module Design**
    *   **Score-based Diffusion Formulation:** The forward process corrupts data $X_0 \sim P_{\text{data}}$ via SDE. The backward process reverses this using the score function $\nabla \log p_t$. The empirical score $\nabla \log \hat{p}_t$ is defined using the empirical data distribution $\hat{P}_{\text{data}}$.
    *   **Statistical Separation Module:** Defines Loss-Gap$_t$ and proves it equals $\text{Fisher}(\hat{P}_t, P_t)$. For mixture models $P_{\text{data}} = \frac{1}{K}\sum P^{(k)}$ satisfying Assumption 4.2, it decomposes the expected denoising means $\mu_{0|t}$ and $\hat{\mu}_{0|t}$ into weighted sums. It bounds dominant weights using sample separation and noise concentration bounds to derive a lower bound on the loss gap.
    *   **Architectural Separation Module:** Defines a ReLU network class $\mathcal{F}(W, L, N)$. To approximate the score function $\nabla \log \hat{p}_t(x) = \nabla \hat{p}_t(x)/\hat{p}_t(x)$, it approximates the numerator and denominator separately. It handles truncation for large $x$ and small density values. It proves that empirical score approximation requires width and parameters scaling with $n$.
    *   **Lipschitz Continuity Module:** Computes the Hessian of log density to analyze Lipschitz continuity. The empirical Hessian depends on $\text{Cov}[X_i|X_t]$, which scales with sample separation and variance, leading to large Lipschitz constants at small $t$. Weight decay is identified as a mechanism to control this.
    *   **Mitigation (Pruning) Module:** For trained DiTs, it computes importance scores for attention heads using masking variables $\xi_h$. It defines the importance score $I(h)$ as the expected gradient magnitude of the loss with respect to $\xi_h$, normalized layerwise. It samples time steps $t$ from a Beta distribution $\text{Beta}(0.8, 2)$ to focus on the small-$t$ regime. It prunes the set of heads with lowest scores and fine-tunes the remaining model.

7.  **All Mathematical Formulas and Symbol Definitions**
    *   **Forward Process:** $dX_t = -\frac{1}{2}X_t dt + dB_t$ for $X_0 \sim P_{\text{data}}$.
    *   **Backward Process:** $d\tilde{X}_t = [\frac{1}{2}\tilde{X}_t + \nabla \log p_{T-t}(\tilde{X}_t)] dt + d\tilde{B}_t$ for $\tilde{X}_0 \sim P_T$.
    *   **Denoising Score Matching Loss:**
        $\hat{L}(s) = \int_{t_0}^T \frac{1}{n}\sum_{i=1}^n \ell_t(x_i, s) dt$
        $\ell_t(x_i, s) = \mathbb{E}_{X_t|X_0=x_i}[\|-\frac{X_t-\alpha_t x_i}{\sigma_t^2} - s(X_t, t)\|_2^2]$
    *   **Time-dependent coefficients:** $\alpha_t = e^{-t/2}$ and $\sigma_t^2 = 1-e^{-t}$.
    *   **Empirical Data Distribution:** $\hat{P}_{\text{data}} = \frac{1}{n}\sum_{i=1}^n \mathbf{1}_{x_i}$.
    *   **Empirical Score Function:**
        $\nabla \log \hat{p}_t(x_t) = -\frac{1}{\sigma_t^2}\sum_{i=1}^n w_i(x_t)(x_t - \alpha_t x_i)$
    *   **Hölder Norm:** For $\beta = s + \gamma > 0$, $s = \lfloor \beta \rfloor$, $\gamma \in [0, 1)$:
        $\|f\|_{\mathcal{H}^\beta(\mathbb{R}^d)} = \max_{s:\|s\|_1 < s} \sup_x |\partial^s f(x)| + \max_{s:\|s\|_1 = s} \sup_{x \neq y} \frac{|\partial^s f(x) - \partial^s f(y)|}{\|x-y\|_2^\gamma}$
    *   **Sub-Gaussian Hölder Density:** $p(x) = \exp(-C\|x\|_2^2/2) \cdot f(x)$ where $f \in \mathcal{H}^\beta(\mathbb{R}^d, B)$ and $\inf_x f(x) \geq c_f$.
    *   **Loss-Gap Definition:** $\text{Loss-Gap}_t = \frac{1}{n}\sum_{i=1}^n (\ell_t(x_i, \nabla \log p_t) - \ell_t(x_i, \nabla \log \hat{p}_t))$
    *   **Proposition 4.1:** $\text{Loss-Gap}_t = \text{Fisher}(\hat{P}_t, P_t) = \mathbb{E}_{X \sim \hat{P}_t}[\|\nabla \log \hat{p}_t(X) - \nabla \log p_t(X)\|_2^2]$.
    *   **Theorem 4.3:** Under Assumption 4.2, separation distance $\Delta_{\min} = \Theta(\sqrt{d})$, $t_0, t_1$ verifying $\log(\sigma_{t_0}) = \Omega(-d)$ and $\log(\sigma_{t_1}) = O(-\log d)$, and $\log n = O(d)$:
        $\mathbb{E}_D[\text{Loss-Gap}_t] = \Omega(d\sigma_t^{-2} + \text{tr}(\Sigma))$ for all $t \in [t_0, t_1]$.
    *   **ReLU Network Class:** $\mathcal{F}(W, L, N) = \{f: f(x) = A_L \cdot \text{ReLU}(\dots A_1 x + b_1 \dots) + b_L \mid A_l \in \mathbb{R}^{d_{l-1} \times d_l}, d_l \leq W, \sum \|A_l\|_0 + \|b_l\|_0 \leq N\}$.
    *   **Theorem 5.1:** For empirical score approximator $s_1 \in \mathcal{F}_1(W_1, L_1, N_1)$:
        $\mathbb{E}_D[\mathbb{E}_{X_t \sim \hat{P}_t}[\|s_1(X_t, t) - \nabla \log \hat{p}_t(X_t)\|_2^2]] \leq \frac{\epsilon}{\sigma_t^4}$
        Configurations: $W_1 = \tilde{O}(n \log^3 \epsilon^{-1}), L_1 = \tilde{O}(\log^2 \epsilon^{-1}), N_1 = \tilde{O}(n \log^4 \epsilon^{-1})$.
        For ground-truth score approximator $s_2 \in \mathcal{F}_2(W_2, L_2, N_2)$:
        $\mathbb{E}_D[\mathbb{E}_{X_t \sim \hat{P}_t}[\|s_2(X_t, t) - \nabla \log p_t(X_t)\|_2^2]] \leq \frac{\epsilon}{\sigma_t^2}$
        Configurations: $W_2 = \tilde{O}(\epsilon^{-\frac{d}{2\beta}} \log^7 \epsilon^{-1}), L_2 = \tilde{O}(\log^4 \epsilon^{-1}), N_2 = \tilde{O}(\epsilon^{-\frac{d}{2\beta}} \log^9 \epsilon^{-1})$.
    *   **Lemma C.1 (Hessian):** $\nabla^2 \log p_t(x_t) = -\frac{1}{\sigma_t^2}I + \frac{\alpha_t^2}{\sigma_t^4}\text{Cov}[X_0|X_t = x_t]$.
    *   **Lipschitz Constant Bounds:** $-\frac{1}{\sigma_t^2} + \frac{\alpha_t^2}{16\sigma_t^4}\min_{i \neq j}\|x_i - x_j\|_2^2 \leq C_t \leq \frac{1}{\sigma_t^2} + \frac{\alpha_t^2}{4\sigma_t^4}\max_{i \neq j}\|x_i - x_j\|_2^2$.
    *   **Importance Score:** $I(h) = \frac{\mathbb{E}_{x \sim D, t \sim T}[|\frac{\partial L(x,t;M)}{\partial \xi_h}|]}{\sqrt{\sum_{h' \in \text{layer}(h)}(\mathbb{E}_{x \sim D, t \sim T}[|\frac{\partial L(x,t;M)}{\partial \xi_h}|])^2}}$.

8.  **Algorithm Pseudocode**

    **Algorithm 1 One-Shot Pruning for Diffusion Transformers**
    1: Input:
    2: Dataset $D$, trained DiT model $M$ with heads $H = \{h_1, \dots, h_H\}$.
    3: Time sampling distribution $T$, which shall put more density on small $t$.
    4: Pruning percentage $\eta \in [0, 1]$, fine-tuning steps $M$.
    5: Compute importance scores $\{I(h)\}_{h \in H} \leftarrow \text{IMPORTANCESCORE}(M, D, T)$.
    6: Identify the set $H_{\text{prune}}$ of $\lfloor \eta \cdot H \rfloor$ heads with the lowest importance scores.
    7: Prune all heads $h \in H_{\text{prune}}$ from the model $M$.
    8: for $m = 1, \dots, M$ do
    9:     Fine-tune the pruned model $M$ on a batch from $D$.
    10: Output: The pruned model $M$.

    **Algorithm 2 IMPORTANCESCORE($M, D, T$)**
    1: Input:
    2: Model $M$ with mask variables $\{\xi_h\}$ for all heads $h \in H$.
    3: Dataset $D$, Time Sampling Distribution $T$.
    4: Initialize: Accumulated scores $S(h) \leftarrow 0$ for all $h \in H$.
    5: for each batch of data $x \sim D$ do
    6:     Sample timestep $t \sim T$.
    7:     Compute loss $L(x, t; M)$.
    8:     Backpropagate to obtain all gradients $\{\frac{\partial L}{\partial \xi_h}\}_{h \in H}$.
    9:     Accumulate scores: $S(h) \leftarrow S(h) + |\frac{\partial L}{\partial \xi_h}|$ for all $h \in H$.
    10: for each layer $l$ in the model do
    11:     Compute layer-wise norm: $N_l \leftarrow \sqrt{\sum_{h' \in l}(S(h'))^2}$.
    12:     for each head $h$ in layer $l$ do
    13:         Normalize score: $I(h) \leftarrow S(h)/N_l$.
    14: Output: Importance scores $\{I(h)\}_{h \in H}$.