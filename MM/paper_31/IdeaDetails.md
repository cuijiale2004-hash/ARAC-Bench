# GuardAlign: Test-Time Safety Alignment in Large Vision-Language Models
## Comprehensive Research Idea and Implementation Plan Extraction

### 1. Research Background and Existing Pain Points

**Research Background:**
Large vision–language models (LVLMs) have recently achieved remarkable progress on multimodal tasks such as visual question answering and image captioning, by integrating vision encoders with large language models (LLMs) to align visual features with textual embeddings for unified multimodal understanding and generation. Despite their rapid adoption and strong performance, ensuring the safety of these models remains a critical challenge. In particular, when input images carry malicious semantics, they are prone to producing harmful responses, which undermines their reliability in real-world applications.

**Existing Pain Points:**
1.  **Inaccurate Detection in Complex Scenes:** Conventional semantic alignment methods (e.g., CLIP) may fail to detect unsafe inputs in real-world applications where images contain many elements. Filtering by CLIP similarity scores produces inevitable overlaps between safe and unsafe samples, allowing unsafe content to pass through because global image embeddings often include irrelevant semantics (e.g., background), reducing alignment accuracy.
2.  **Unstable Safety Signals During Decoding:** During inference, the attention assigned to safety prefixes becomes progressively diluted, weakening the defense they activate. As layer depth increases, the attention weight to prefix tokens consistently decreases, revealing a gradual decay of the safety signal. This decline can lead to outcomes where the model initially refuses but later, after transitional words like "however," produces unsafe responses (refusal-override pattern).
3.  **Vulnerability of Continuous Visual Embeddings:** Prior work attributes the vulnerability of LVLMs to the continuous nature of visual token embeddings, which deviate from the discrete textual embeddings and thereby bypass safety mechanisms originally designed for language backbones. Visual and textual representations remain not well aligned, indicating the risk that harmful semantics in images may be overlooked.
4.  **Computational Overhead of Existing Defenses:** Tuning-based methods and multi-step inference methods (e.g., contrastive decoding) introduce additional computational and time overhead, limiting their applicability.

### 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:**
To address the issues of inaccurate detection and unstable decoding signals in existing input-side defenses, this work aims to design a training-free defense framework that integrates semantic detection and model decoding. The motivation is to explicitly model the correlation between visual inputs and unsafe semantics using distribution distances and to ensure that the model’s intrinsic safety mechanism remains consistently activated regardless of generation length.

**Scientific Questions:**
1.  How can we accurately identify unsafe elements within complex images without additional computational cost, overcoming the overlap of similarity scores in conventional methods?
2.  How can we ensure that safety guidance persists throughout generation and prevents unsafe outputs triggered by transitional phrases, avoiding excessive refusals?
3.  Does an Optimal Transport (OT) based distance metric theoretically and empirically yield lower classification error for unsafe patches compared to cosine similarity?

### 3. Overall Core Idea and Design Philosophy

**Overall Core Idea:**
The paper proposes GuardAlign, a training-free defense framework that integrates two strategies: OT-enhanced safety detection and cross-modal attentive calibration.
1.  **OT-enhanced safety detection** leverages optimal transport to measure distribution distances between image patches and unsafe semantics, enabling accurate identification of malicious regions without additional computational cost.
2.  **Cross-modal attentive calibration** strengthens the influence of safety prefixes by adaptively reallocating attention across layers, ensuring that safety signals remain consistently activated throughout generation.

**Design Philosophy:**
*   **Fine-grained Modeling:** Instead of using global image embeddings, the method models images and prompts as discrete distributions over patches and textual variants to capture distributional discrepancies.
*   **Distributional Distance Measurement:** Using Optimal Transport (OT) distance instead of cosine similarity to prioritize discriminative features that enhance separation between safe and unsafe classes.
*   **Decoding-time Intervention:** Strengthening attention to prefix tokens during modality fusion in middle layers to prevent the dilution of safety signals.

### 4. Core Innovation Points

1.  **OT-Enhanced Safety Detection Strategy:** Proposes a method that leverages optimal transport with entropy-based weights to measure the distance between image patch distributions and unsafe prompt distributions, substantially improving the detection accuracy of unsafe content over cosine similarity.
2.  **Cross-Modal Attentive Calibration Mechanism:** Designs a mechanism that adaptively reallocates safety-related attention across layers by amplifying attention toward prefix tokens during modality fusion, preventing the decay of safety signals and the "refusal-override" pattern.
3.  **Theoretical Guarantee:** Provides a theoretical analysis proving that the OT-based method achieves a lower or equal classification error ($P_{error}^{OT} \le P_{error}^{cos}$) compared to the cosine similarity baseline, with equality when OT uses uniform weights.
4.  **Unified Training-Free Framework:** Establishes GuardAlign as an efficient and effective defense framework that operates entirely at inference time, reducing unsafe response rates while preserving or even improving general utility without requiring retraining or additional data.

### 5. Overview of the Overall Technical Solution

The GuardAlign framework operates in two main stages:
1.  **OT-Enhanced Safety Detection:** Image patches and predefined unsafe prompt categories are jointly encoded. Optimal transport is used to identify patches that align with harmful semantics by measuring the OT distance between the image patch distribution and the unsafe prompt distribution. The most suspicious patches are masked to produce a sanitized image.
2.  **Cross-Modal Attention Calibration:** A lightweight safety prefix is added to the query, and the multimodal model attends over the sanitized visual tokens. In the middle layers where visual and textual modalities are most strongly integrated, the attention scores are calibrated by amplifying the attention from instruction tokens to prefix tokens. This design guides the model toward safe evidence and prevents unsafe generations.

### 6. Detailed Module Design

**Module 1: OT-Enhanced Safety Detection**
This module explicitly models the correlation between visual inputs and unsafe semantics.
*   **Input Encoding:** An input image $x$ and a textual prompt $z$ are encoded using the CLIP model into feature representations: $\mathbf{x} = \Phi_v(x)$, $\mathbf{z} = \Phi_t(z)$.
*   **Distribution Modeling:** Each image is divided into $M$ patches $\{\mathbf{x}_m\}_{m=1}^M$ with features $\{\mathbf{x}_m\}_{m=1}^M$. Each prompt $z_i$ is expanded into $N$ textual variants $\{\mathbf{z}_n^i\}_{n=1}^N$. The image and prompts are modeled as discrete distributions:
    $$P(\mathbf{x}) = \sum_{m=1}^M a_m \delta(\mathbf{x}_m - \mathbf{x}), \quad Q_i(\mathbf{z}) = \sum_{n=1}^N b_n^i \delta(\mathbf{z}_n^i - \mathbf{z})$$
*   **Entropy-based Weighting:** Importance weights $a_m$ for image patches are assigned according to entropy with respect to the average prompt embedding $\bar{\mathbf{z}}_i = \frac{1}{N}\sum_{n=1}^N \mathbf{z}_n^i$:
    $$a_m = \frac{\exp(h(\mathbf{x}_m))}{\sum_{m'} \exp(h(\mathbf{x}_{m'}))}, \quad h(\mathbf{x}_m) = -\sum_{i=1}^C p(\bar{\mathbf{z}}_i | \mathbf{x}_m) \log p(\bar{\mathbf{z}}_i | \mathbf{x}_m)$$
    Low-entropy patches (more confident predictions) are assigned higher weights. Textual weights $b_n^i$ are computed analogously.
*   **OT Distance Calculation:** The alignment is measured via OT distance:
    $$d_{OT}(P, Q_i; C_i) = \min_{T_i \ge 0} \langle T_i, C_i \rangle, \quad \text{s.t.} \quad T_i \mathbf{1}_M = a, T_i^\top \mathbf{1}_N = b_i$$
    where $a = [a_1, \ldots, a_M]^\top$, $b_i = [b_1^i, \ldots, b_N^i]^\top$, and $C_i(m,n) = 1 - \cos(\mathbf{x}_m, \mathbf{z}_n^i)$. The transport plan $T_i$ is obtained via the Sinkhorn-Knopp algorithm.
*   **Aggregation and Thresholding:** An aggregated OT score for patch $m$ is calculated:
    $$d_{OT}(m) = \sum_{i=1}^C \sum_{n=1}^N T_i(m,n) C_i(m,n)$$
    Unsafe regions are identified by thresholding: $S_{unsafe} = \{m | d_{OT}(m) \le \tau\}$.
*   **Masking:** A sanitized input is constructed by masking the patches in $S_{unsafe}$:
    $$\hat{\mathbf{x}}_m = \begin{cases} \mathbf{0}, & m \in S_{unsafe}, \\ \mathbf{x}_m, & \text{otherwise}, \end{cases} \quad \hat{x} = \{\hat{\mathbf{x}}_m\}_{m=1}^M$$

**Module 2: Cross-Modal Attentive Calibration**
This module ensures safety guidance persists throughout generation.
*   **Attention Score Adjustment:** Let $A_{l,h}$ denote the attention matrix of the $h$-th head in the $l$-th layer, and $Z_{l,h}$ the corresponding pre-softmax scores:
    $$A_{l,h} = \text{Softmax}(Z_{l,h}), \quad Z_{l,h} = \frac{Q_{l,h} K_{l,h}^\top}{\sqrt{d_k}}$$
*   **Calibration:** The scores in middle layers are adjusted by amplifying attention toward prefix tokens:
    $$\hat{Z}_{l,h} = Z_{l,h} + \gamma M^{pref}_{l,h} \circ Z_{l,h}$$
    where $\gamma > 0$ controls the amplification strength and $\circ$ denotes the Hadamard product.
*   **Mask Definition:** The mask $M^{pref}_{l,h}$ selects query–key pairs from instruction tokens $T$ to prefix tokens $R$:
    $$M^{pref}_{l,h}(i, j) := \mathbb{I} \circ S_{l,h}(i, j), \quad S_{l,h}(i, j) = \max(0, \langle Q_{l,h}(i, :), K_{l,h}(j, :) \rangle)$$
    for $i \in T$, $j \in R$. Here, $T$ denotes the set of instruction tokens, while $R$ refers to prefix tokens introduced for safety control.

### 7. All Mathematical Formulas and Symbol Definitions

**Optimal Transport Preliminaries:**
*   $P = \sum_{i=1}^{|V|} a_i \delta(v_i - v)$: Discrete distribution in feature space.
*   $Q = \sum_{j=1}^{|U|} b_j \delta(u_j - u)$: Discrete distribution in feature space.
*   $a, b$: Probability vectors that sum to 1.
*   $C \in \mathbb{R}^{|V| \times |U|}$: Cost matrix where $C(i,j)$ measures transport cost.
*   $T \in \mathbb{R}^{|V| \times |U|}$: Transport plan.
*   $d_{OT}(P,Q;C) = \min_{T} \langle T,C \rangle, \quad \text{s.t.} \quad T\mathbf{1}_{|U|} = a, T^\top\mathbf{1}_{|V|} = b$
*   Entropic regularization: $d_{OT}(P,Q;C) = \min_{T} \langle T,C \rangle - \epsilon h(T)$

**LVLM Preliminaries:**
*   $\Phi$: Vision encoder.
*   $G$: Connector module.
*   $\mathbf{h}_v = \Phi(x)$: Visual features.
*   $\mathbf{z}_v = G(\mathbf{h}_v)$: Mapped visual tokens.
*   $Z = [\mathbf{z}_v, z_1, \ldots, z_L]$: Multimodal input sequence.
*   $y_t \sim \pi_\theta(\cdot | Z, y_{<t}), t = 1, \ldots, T$: Autoregressive prediction.

**OT-Enhanced Safety Detection Formulas:**
*   $\mathbf{x} = \Phi_v(x), \mathbf{z} = \Phi_t(z)$
*   $P(\mathbf{x}) = \sum_{m=1}^M a_m \delta(\mathbf{x}_m - \mathbf{x}), Q_i(\mathbf{z}) = \sum_{n=1}^N b_n^i \delta(\mathbf{z}_n^i - \mathbf{z})$
*   $a_m = \frac{\exp(h(\mathbf{x}_m))}{\sum_{m'} \exp(h(\mathbf{x}_{m'}))}, h(\mathbf{x}_m) = -\sum_{i=1}^C p(\bar{\mathbf{z}}_i | \mathbf{x}_m) \log p(\bar{\mathbf{z}}_i | \mathbf{x}_m)$
*   $d_{OT}(P,Q_i;C_i) = \min_{T_i \ge 0} \langle T_i,C_i \rangle, \text{s.t.} \quad T_i\mathbf{1}_M = a, T_i^\top\mathbf{1}_N = b_i$
*   $d_{OT}(m) = \sum_{i=1}^C \sum_{n=1}^N T_i(m,n)C_i(m,n)$
*   $S_{unsafe} = \{m | d_{OT}(m) \le \tau \}$
*   $\hat{\mathbf{x}}_m = \begin{cases} \mathbf{0}, & m \in S_{unsafe}, \\ \mathbf{x}_m, & \text{otherwise}, \end{cases} \quad \hat{x} = \{\hat{\mathbf{x}}_m\}_{m=1}^M$

**Cross-Modal Attentive Calibration Formulas:**
*   $A_{l,h} = \text{Softmax}(Z_{l,h}), Z_{l,h} = \frac{Q_{l,h}K_{l,h}^\top}{\sqrt{d_k}}$
*   $\hat{Z}_{l,h} = Z_{l,h} + \gamma M^{pref}_{l,h} \circ Z_{l,h}$
*   $M^{pref}_{l,h}(i, j) := \mathbb{I} \circ S_{l,h}(i, j), S_{l,h}(i, j) = \max(0, \langle Q_{l,h}(i, :), K_{l,h}(j, :) \rangle)$

**Theoretical Analysis Formulas (Proof of Lower Classification Error):**
*   $d_{OT}(m) = \sum_{n=1}^N b_n (1 - \cos(\mathbf{x}_m, \mathbf{z}_n)) = 1 - \mu, \mu = \sum_{n=1}^N b_n \cos(\mathbf{x}_m, \mathbf{z}_n)$
*   $d_{cos}(m) = \sum_{n=1}^N \cos(\mathbf{x}_m, \mathbf{z}_n)$
*   $P_{error}^{OT} = \pi_0 P(d_{OT} \le \tau | y = 0) + \pi_1 P(d_{OT} > \tau | y = 1)$
*   $P_{error}^{cos} = \pi_0 P(d_{cos} \ge \tau_{cos} | y = 0) + \pi_1 P(d_{cos} < \tau_{cos} | y = 1)$
*   $d'_{OT} = \frac{\sum_n b_n \delta_n}{\sigma \sqrt{\sum_n (b_n)^2}}$
*   $d'_{cos} = \frac{\sum_n \delta_n}{\sigma \sqrt{N}}$
*   Inequality to prove: $\frac{\sum_n b_n \delta_n}{\sqrt{\sum_n (b_n)^2}} \ge \frac{\sum_n \delta_n}{\sqrt{N}}$
*   Squaring: $\frac{(\sum_n b_n \delta_n)^2}{\sum_n (b_n)^2} \ge \frac{(\sum_n \delta_n)^2}{N}$
*   Chebyshev bound: $P(s \le \tau | y = 0) \le \frac{4\sigma_0^2}{\Delta^2}$, $P(s > \tau | y = 1) \le \frac{4\sigma_1^2}{\Delta^2}$
*   Conclusion: $P_{error}^{OT} \le \frac{4}{(d'_{OT})^2} \le \frac{4}{(d'_{cos})^2} \le P_{error}^{cos}$

### 8. Algorithm Pseudocode

The paper does not provide a formal pseudocode block, but the algorithmic flow is strictly defined in the Method section and Figure 2. The logical flow is extracted as follows:

**Input:** Input Image $x$, User Prompt, Pre-defined Unsafe Prompts $Z$, Safety Prefix, Threshold $\tau$, Amplification strength $\gamma$.
**Output:** Safe Response $y$.

**Phase 1: OT-Enhanced Safety Detection**
1.  Divide image $x$ into $M$ patches $\{\mathbf{x}_m\}_{m=1}^M$.
2.  Encode patches and unsafe prompts $Z=\{z_i\}_{i=1}^C$ using CLIP: $\mathbf{x}_m = \Phi_v(\mathbf{x}_m)$, $\mathbf{z}_n^i = \Phi_t(\mathbf{z}_n^i)$.
3.  Compute entropy-based weights $a_m$ for patches using $\bar{\mathbf{z}}_i$.
4.  Compute textual weights $b_n^i$ for prompt variants.
5.  Construct distributions $P(\mathbf{x})$ and $Q_i(\mathbf{z})$.
6.  For each category $i$:
    a. Compute cost matrix $C_i(m,n) = 1 - \cos(\mathbf{x}_m, \mathbf{z}_n^i)$.
    b. Solve for transport plan $T_i$ using Sinkhorn-Knopp subject to $T_i\mathbf{1}_M = a, T_i^\top\mathbf{1}_N = b_i$.
7.  For each patch $m$:
    a. Calculate aggregated OT score $d_{OT}(m) = \sum_{i=1}^C \sum_{n=1}^N T_i(m,n)C_i(m,n)$.
8.  Identify unsafe patches: $S_{unsafe} = \{m | d_{OT}(m) \le \tau\}$.
9.  Construct sanitized image $\hat{x}$ by masking patches in $S_{unsafe}$ (setting features to 0).

**Phase 2: Cross-Modal Attentive Calibration & Generation**
1.  Prepare input sequence: Concatenate visual tokens from $\hat{x}$, Safety Prefix, and User Prompt.
2.  For each layer $l$ in the LVLM (focusing on middle layers):
    a. Compute standard attention scores $Z_{l,h} = \frac{Q_{l,h}K_{l,h}^\top}{\sqrt{d_k}}$.
    b. Compute calibration mask $M^{pref}_{l,h}(i, j) = \mathbb{I} \circ \max(0, \langle Q_{l,h}(i, :), K_{l,h}(j, :) \rangle)$ for $i \in T, j \in R$.
    c. Adjust scores: $\hat{Z}_{l,h} = Z_{l,h} + \gamma M^{pref}_{l,h} \circ Z_{l,h}$.
    d. Compute attention: $A_{l,h} = \text{Softmax}(\hat{Z}_{l,h})$.
3.  Autoregressively generate response $y$.