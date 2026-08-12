### 1. Research Background and Existing Pain Points

**Research Background:** Text-guided image editing with diffusion models has seen remarkable success, largely powered by Classifier-Free Guidance (CFG). However, current editing methods suffer from a persistent weakness: background regions are notoriously difficult to modify, while object-centric edits succeed more reliably. Prior approaches have mainly attacked this problem through heuristic spatial controls, asking "WHERE" an edit should occur. These techniques often involve manipulating cross-attention maps or using the guidance difference vector to generate a spatial mask that separates the image into edit and preserve zones. For instance, DiffEdit pioneered using the guidance difference vector, $\Delta\epsilon$, to automatically generate a spatial mask, but it still interprets the signal spatially, partitioning the image into a binary "edit" versus "preserve" zone. 

**Existing Pain Points:** The primary pain point is the failure of background editing. When attempting to modify backgrounds (e.g., moving an owl from "the wild" to "a school"), current methods often fail to convincingly alter the scene or inadvertently degrade the subject. Methods like DiffEdit construct a binary spatial mask from the guidance magnitude and overwrite the latent representation inside the masked region. This functions as a hard spatial filter: regions with weak guidance are entirely removed from the editing process. Consequently, DiffEdit fails to edit background regions where the guidance signal is naturally weak. This limitation stems from a theoretical oversight: DiffEdit assumes low-magnitude regions are irrelevant, filtering them out and inadvertently discarding the valid semantic signals required for background editing. This creates a persistent trade-off where global text-alignment metrics (like CLIP) may improve through global image modifications that destroy identity, while identity-preserving methods fail to achieve faithful background alterations.

### 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:** The true bottleneck in diffusion model editing lies not in "WHERE" to apply edits, but in "HOW" the guidance signal itself is structured. The guidance difference vector $\Delta\epsilon$, central to CFG, is not random noise but the gradient of a log-likelihood ratio, whose expected magnitude is governed by local Fisher information density. This framing reveals a fundamental statistical law: objects, being information-dense, naturally yield strong guidance, whereas backgrounds, being information-sparse, yield weak guidance. The motivation is to reframe background editing failure not as an incidental flaw of prior methods, but as an information-theoretic inevitability (termed "Information Imbalance") stemming from the intrinsic properties of the diffusion guidance field.

**Scientific Questions:** How is the guidance difference vector $\Delta\epsilon$ structurally organized in terms of semantic hierarchy? Why do diffusion models systematically fail to edit background regions effectively compared to object regions? Can the weak, low-information guidance signals corresponding to backgrounds be selectively amplified to achieve robust and disentangled editing without destabilizing high-information object structures?

### 3. Overall Core Idea and Design Philosophy

**Overall Core Idea:** The core idea is the **Semantic Scale Hypothesis**, which posits that the magnitude of the guidance difference vector ($\|\Delta\epsilon\|$) directly encodes the semantic scale of edits. Large-magnitude guidance corresponds to structural edits (objects), while small-magnitude guidance corresponds to stylistic/background adjustments. This phenomenon is theoretically grounded in the connection between score prediction and the variance of the underlying data distribution via Tweedie’s formula. Low-variance regions (objects) yield large-magnitude differences, whereas high-variance regions (backgrounds) yield small-magnitude differences. Therefore, $\|\Delta\epsilon\|^2$ is proportional to the local Fisher information density, making the object/background gap a statistical necessity.

**Design Philosophy:** The design philosophy shifts the editing paradigm from a spatial masking problem ("WHERE") to a principled signal processing challenge ("HOW"). Instead of creating a binary spatial mask that filters out weak signals, the method decomposes the guidance signal into distinct semantic layers based on their information-theoretic signature. It selectively amplifies the weak, low-information components corresponding to backgrounds to re-balance the Fisher information disparity inherent in diffusion guidance, enabling disentangled control where object edits and background edits can be executed independently and robustly.

### 4. Core Innovation Points

1.  **Semantic Scale Hypothesis:** The paper formalizes a new principle connecting guidance magnitude to Fisher information density, providing the first theoretical explanation for the persistent difficulty of background editing. It mathematically proves that $\Delta\epsilon$ acts as a gradient field of a log-likelihood ratio, where $\|\Delta\epsilon\|^2 \propto$ local Fisher information density, explaining background editing failure as an information-theoretic inevitability.
2.  **Prism-Edit Framework:** A simple, training-free, model-agnostic method that operationalizes the Semantic Scale Hypothesis by decomposing the guidance field into semantic layers and selectively amplifying low-information signals.
3.  **Principled Extraction of Semantic Map with Z-score Normalization:** To compensate for the Fisher information imbalance where absolute magnitudes vary significantly, the method introduces a Z-score normalized semantic map ($M_{sem}$) extracted from a specific high-noise timestep window. This transforms raw gradients into a scale-invariant semantic signal, restoring weak background signals to a comparable scale.
4.  **Disentangled Application Modalities:** The introduction of two complementary, training-free editing modalities: Dynamic Guidance Modulation (default) that dynamically modulates guidance element-wise at each step to re-balance Fisher information, and Static Mask Blending (optional) that uses morphological refinement for maximum fidelity in corner cases requiring strict identity preservation.

### 5. Overview of the Overall Technical Solution

The overall technical solution, Prism-Edit, is a two-stage process: (1) principled extraction of a multi-layered Semantic Map, and (2) disentangled application of edits via one of two complementary modalities.

**Stage 1: Semantic Map Extraction.** To compensate for the Fisher information imbalance, a $\sigma$-normalized thresholding scheme is adopted. A narrow, high-noise window (e.g., $t \in [900, 800]$ for a 1000-step schedule) is probed to maximize semantic coverage while retaining structural plasticity. An averaged guidance difference is computed, and Z-score normalization is employed to transform the raw gradients into a scale-invariant semantic signal $M_{sem}$. Based on the tails of this semantic map, two primary semantic layers are defined: a background/style layer ($M_{sem} < 0.6$) and an object-core layer ($M_{sem} \ge 3.0$).

**Stage 2: Disentangled Application Modalities.** The extracted semantic map enables two distinct editing modalities:
*   **Dynamic Guidance Modulation (Default):** Offers greater flexibility by dynamically modulating guidance at each step. The semantic map is binarized based on the Z-score of the instantaneous $\|\Delta\epsilon_t\|$ to ensure stability. The guidance is then modulated element-wise, enabling region-specific guidance scaling where background edits can be amplified with large $\gamma$ without destabilizing object regions.
*   **Static Mask Blending (Optional):** Acts as a loose, permissive spatial constraint for maximum fidelity. It defines an active editing region using a coarse threshold on $M_{sem}$, refines it with morphological closing, and blends the edited latent with the source latent at each step, guaranteeing unmasked regions remain unchanged.

### 6. Detailed Module Design

**Semantic Map Extraction Module:**
This module derives control signals directly from the model’s internal generation dynamics. It probes a specific high-noise interval (e.g., $t \in [900, 800]$) to compute an averaged guidance difference. Because raw magnitude $\|\Delta\epsilon\|$ varies significantly across editing tasks and architectures due to posterior variance shifts (Information Imbalance), absolute thresholding is infeasible. Instead, it employs Z-score normalization to yield $M_{sem}$. Empirical analysis shows that the extreme tails of this map correspond to the cleanest semantic signals (object-core and background/style), while intermediate values represent mixtures. Thus, fixed relative thresholds ($\sigma$-levels) are used to generalize across prompts and seeds. The two primary semantic layers are defined as: background/style layer ($M_{sem} < 0.6$) and object-core layer ($M_{sem} \ge 3.0$).

**Dynamic Guidance Modulation Module:**
This is the default modality that operationalizes the information-field perspective. Although theoretically defined as a continuous map, in practice, $W_{sem,t}$ is binarized based on the Z-score of the instantaneous $\|\Delta\epsilon_t\|$ (using $< 0.6\sigma$ for background edits and $\ge 3.0\sigma$ for object edits) to ensure stability and prevent boundary artifacts. The guidance is then modulated element-wise according to Eq. 9. This enables region-specific guidance scaling: background edits (low-information, high-variance) can be amplified with large $\gamma$ (e.g., 20–40) without destabilizing object regions. By locally scaling weak, high-uncertainty regions while leaving strong, low-uncertainty regions untouched, it effectively re-balances the Fisher information disparity inherent in diffusion guidance.

**Static Mask Blending Module:**
This optional module acts as a deterministic safety net for corner cases requiring strict preservation. It defines the active editing region using a coarse threshold: targeting high-magnitude areas ($M_{sem} \ge 0.6$) for object edits, and low-magnitude areas ($M_{sem} < 0.6$) for background edits. The mask is designed to be broad, preventing edits from drifting into completely irrelevant regions while leaving semantic boundaries flexible. For tasks requiring strict identity preservation, a tighter constraint is imposed by explicitly excluding high-magnitude object cores ($M_{sem} \ge 3.0$). The base mask undergoes morphological closing (dilation followed by erosion) via Algorithm 2 to ensure spatial contiguity. At each step $t$, the edited latent $x^{pred}_{t-1}$ is blended with the corresponding source latent $x^{src}_{t-1}$ via Eq. 8.

### 7. All Mathematical Formulas and Symbol Definitions

**Guidance Difference as a Log-Likelihood Ratio Gradient:**
$$\Delta\epsilon(x_t; c_1, c_2) \propto \nabla_{x_t} \log p(x_t|c_2) - \nabla_{x_t} \log p(x_t|c_1) \quad (1)$$
$$\Delta\epsilon(x_t; c_1, c_2) \propto \nabla_{x_t} \log \frac{p(x_t|c_2)}{p(x_t|c_1)} \quad (2)$$

**Semantic Scale as a Consequence of Information Density:**
$$\|\Delta\epsilon(x_t; c_1, c_2)\| \propto \frac{\|\Delta\mu_t\|}{\sigma_t}, \quad \Delta\mu_t := E[x_0 | x_t, c_2] - E[x_0 | x_t, c_1] \quad (3)$$

**Theorem 1 (KL-based bound for guidance magnitude):** Let $d$ be the dimensionality and define the Gaussian KL divergence
$$D_{KL}(N(\mu_{c_1}, \Sigma_{c_1}) \| N(\mu_{c_2}, \Sigma_{c_2})) = \frac{1}{2} \left[ \text{tr} \left( \Sigma^{-1}_{c_2} \Sigma_{c_1} \right) + (\Delta\mu_t)^\top \Sigma^{-1}_{c_2} \Delta\mu_t - d + \log \frac{\det \Sigma_{c_2}}{\det \Sigma_{c_1}} \right].$$
Let $\lambda_{max}(\Sigma)$ and $\lambda_{min}(\Sigma)$ denote the largest and smallest eigenvalues of a symmetric positive definite matrix $\Sigma$, respectively. Then, for any $t$,
$$\|\Delta\epsilon\|^2 \le \frac{\lambda_{max}(\Sigma_{c_2})}{\sigma_t^2} \left\{ 2D_{KL}(N(\mu_{c_1}, \Sigma_{c_1}) \| N(\mu_{c_2}, \Sigma_{c_2})) - \underbrace{\left[ \text{tr}(\Sigma^{-1}_{c_2}\Sigma_{c_1}) - d - \log \det(\Sigma^{-1}_{c_2}\Sigma_{c_1}) \right]}_{:= \Psi(\Sigma_{c_1}, \Sigma_{c_2})} \right\} \quad (4)$$
and symmetrically with $(c_1, c_2)$ swapped:
$$\|\Delta\epsilon\|^2 \le \frac{\lambda_{max}(\Sigma_{c_1})}{\sigma_t^2} \left\{ 2D_{KL}(N(\mu_{c_2}, \Sigma_{c_2}) \| N(\mu_{c_1}, \Sigma_{c_1})) - \Psi(\Sigma_{c_2}, \Sigma_{c_1}) \right\} \quad (5)$$
Moreover, the following lower bounds hold:
$$\|\Delta\epsilon\|^2 \ge \frac{\lambda_{min}(\Sigma_{c_2})}{\sigma_t^2} \left\{ 2D_{KL}(N(\mu_{c_1}, \Sigma_{c_1}) \| N(\mu_{c_2}, \Sigma_{c_2})) - \Psi(\Sigma_{c_1}, \Sigma_{c_2}) \right\} \quad (6)$$
and analogously with $(c_1, c_2)$ swapped.
*Interpretation:* $\Psi(\Sigma_{c_1}, \Sigma_{c_2}) := \text{tr}(\Sigma^{-1}_{c_2}\Sigma_{c_1}) - d - \log \det(\Sigma^{-1}_{c_2}\Sigma_{c_1}) \ge 0$ quantifies the covariance mismatch (vanishes iff $\Sigma_{c_1} = \Sigma_{c_2}$).

**Corollary 1 (Equal-covariance simplification):** If $\Sigma_{c_1} = \Sigma_{c_2} = \Sigma$, then $\Psi = 0$ and
$$\frac{2\lambda_{min}(\Sigma)}{\sigma_t^2} D_{KL}(N(\mu_{c_1}, \Sigma) \| N(\mu_{c_2}, \Sigma)) \le \|\Delta\epsilon\|^2 \le \frac{2\lambda_{max}(\Sigma)}{\sigma_t^2} D_{KL}(N(\mu_{c_1}, \Sigma) \| N(\mu_{c_2}, \Sigma))$$

**Connection to Fisher divergence:** From Eq. A.2, taking an expectation w.r.t. any reference density $q(x_t)$ gives
$$\mathbb{E}_q \left[ \|\Delta\epsilon\|^2 \right] = \sigma_t^2 \mathbb{E}_q \left[ \|\nabla_{x_t} \log p(x_t|c_2) - \nabla_{x_t} \log p(x_t|c_1)\|^2 \right] = \sigma_t^2 F_q(p(\cdot|c_2), p(\cdot|c_1)),$$
the (generalized) Fisher divergence between the two conditionals under $q$.

**Semantic Map Extraction:**
$$\Delta\epsilon = \frac{1}{N_{probe}} \sum_{i=1}^{N_{probe}} \Delta\epsilon_{t_i}, \quad M_{sem} = \frac{|\Delta\epsilon| - \mu_{|\Delta\epsilon|}}{\sigma_{|\Delta\epsilon|}} \quad (7)$$

**Static Mask Blending:**
$$x_{t-1} \leftarrow x^{pred}_{t-1} \odot M_{final} + x^{src}_{t-1} \odot (1 - M_{final}) \quad (8)$$

**Dynamic Guidance Modulation:**
$$\tilde{\epsilon}_\theta(x_t, c) = \epsilon_\theta(x_t, c_{src}) + \gamma \cdot (\Delta\epsilon_t \odot W_{sem,t}) \quad (9)$$

**Appendix Derivations:**
**Tweedie's formula:**
$$E[x_0 | x_t, c] = x_t + \sigma_t^2 \nabla_{x_t} \log p(x_t | c) \quad (A.1)$$
**Guidance Difference as a Difference of Scores:**
$$\Delta\epsilon(x_t; c_1, c_2) \propto \sigma_t (\nabla_{x_t} \log p(x_t | c_1) - \nabla_{x_t} \log p(x_t | c_2)) \quad (A.2)$$
**Relating Guidance Magnitude to Posterior Mean Shift:**
$$\|\Delta\epsilon(x_t; c_1, c_2)\| \propto \frac{\|\Delta\mu_t\|}{\sigma_t}, \text{ where } \Delta\mu_t := E[x_0 | x_t, c_2] - E[x_0 | x_t, c_1] \quad (A.3)$$

### 8. Algorithm Pseudocode

**Algorithm 1 Prism-Edit (Full Version with Optional Static Mask Refinement)**
Require: Source prompt $c_{src}$, target prompt $c_{tgt}$, probe interval $\{t_{900}, \ldots, t_{800}\}$
1: Initialize latent $x_T \sim \mathcal{N}(0, I)$
2: // Stage 1: Semantic Map Extraction
3: for $t_i$ in probe interval do
4: $\quad \Delta\epsilon_{t_i} \leftarrow \epsilon_\theta(x_{t_i}, c_{tgt}) - \epsilon_\theta(x_{t_i}, c_{src})$
5: end for
6: $M_{sem} \leftarrow \text{z-score}\left(\frac{1}{N}\sum_i \|\Delta\epsilon_{t_i}\|\right)$
7: // Optional: Static Mask Generation
8: if target is object then
9: $\quad M_{base} \leftarrow (M_{sem} \ge 0.6)$
10: else ▷ target is background
11: $\quad M_{base} \leftarrow (M_{sem} < 0.6)$
12: end if
13: $M_{filled} \leftarrow \text{Mask-Refinement}(M_{base})$ ▷ See Algorithm 2
14: if identity preservation mode then
15: $\quad M_{exclude} \leftarrow (M_{sem} \ge 3.0)$
16: $\quad M_{final} \leftarrow \text{clamp}(M_{filled} - M_{exclude}, 0, 1)$
17: else
18: $\quad M_{final} \leftarrow M_{filled}$
19: end if
20: for $t = T, \ldots, 1$ do ▷ Stage 2: Disentangled Application
21: // Dynamic guidance modulation (always on)
22: $\Delta\epsilon_t \leftarrow \epsilon_\theta(x_t, c_{tgt}) - \epsilon_\theta(x_t, c_{src})$
23: // Binarize based on editing intent (e.g., $\ge 3.0\sigma$ for object, $< 0.6\sigma$ for bg)
24: $W_{sem,t} \leftarrow \text{Binarize}(\text{z-score}(\|\Delta\epsilon_t\|) \text{ meets } \tau)$
25: $\tilde{\epsilon}_\theta \leftarrow \epsilon_\theta(x_t, c_{src}) + \gamma \cdot (\Delta\epsilon_t \odot W_{sem,t})$
26: $x^{pred}_{t-1} \leftarrow S(x_t, \tilde{\epsilon}_\theta, t)$
27: // Static blending (optional)
28: if static mask mode then
29: $\quad x_{t-1} \leftarrow x^{pred}_{t-1} \odot M_{final} + x^{src}_{t-1} \odot (1 - M_{final})$
30: else
31: $\quad x_{t-1} \leftarrow x^{pred}_{t-1}$
32: end if
33: end for
34: return Edited image $\hat{x}_0$

**Algorithm 2 Mask Refinement (Morphological Closing)**
Require: Base mask $M_{base} \in \{0, 1\}^{H \times W}$, number of iterations $K$
We apply morphological closing (dilation followed by erosion) to ensure the semantic mask is contiguous and free of small holes.
1: $M \leftarrow M_{base}$
2: for $k = 1$ to $K$ do
3: $\quad M \leftarrow \text{Dilate}(M)$
4: $\quad M \leftarrow \text{Erode}(M)$
5: end for
6: return $M_{filled} \leftarrow M$