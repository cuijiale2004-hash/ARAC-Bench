### 1. Research Background and Existing Pain Points

**Research Background:**
Multimodal Large Language Models (MLLMs) have demonstrated superior performance on various vision-language tasks, marking a significant step forward in multimodal learning. However, their advancement has largely been driven by scaling laws, resulting in ever-larger and more computationally demanding architectures. The substantial computational and memory costs associated with such massive models pose considerable challenges for practical deployment. This has intensified interest in developing efficient MLLMs. To this end, knowledge distillation (KD) has emerged as a promising approach, transferring rich knowledge from a larger teacher model to a smaller student model.

**Existing Pain Points:**
Existing KD methods struggle to effectively distill the teacher MLLM’s rich visual perception abilities to the student, a challenge that has been largely overlooked in previous studies. Specifically, while these KD-based methods have demonstrated effectiveness in visual question answering (VQA) tasks—which require visual recognition ability (identifying objects based on features)—they fail to deliver comparable improvements on compositional reasoning (CR) tasks. CR tasks are designed to assess visual perception ability, which involves more complex capabilities such as understanding relationships among objects and accurately capturing their attributes. Through systematic analysis, it is observed that existing KD methods show significant performance improvements on VQA, but their performance on CR is on par with the standard supervised fine-tuning (SFT) model. This indicates that existing KD methods struggle to effectively distill visual perception ability from the teacher model.

### 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:**
The core motivation is to enhance the visual perception abilities of Knowledge Distillation (KD)-based Multimodal Large Language Models (MLLMs), addressing the gap where existing methods improve visual recognition but fail to transfer the ability to understand complex object relationships and attributes. By identifying the specific bottlenecks (visual attention misalignment) that hinder the effectiveness of knowledge distillation in MLLMs, the goal is to design a framework that transfers both visual recognition and perception abilities efficiently, making MLLMs more practical for real-world applications requiring fine-grained visual reasoning.

**Scientific Questions:**
1.  Why do existing KD methods fail to deliver improvements on compositional reasoning (visual perception) tasks, unlike their success on VQA (visual recognition) tasks?
2.  Is visual attention misalignment the root cause of the failure to distill visual perception?
3.  Does increasing the teacher-student visual attention similarity over visual tokens enhance performance on CR tasks?
4.  How can the student’s visual attention mechanism be explicitly aligned with the teacher’s, considering the differences in model depth and feature space incompatibility?

### 3. Overall Core Idea and Design Philosophy

**Overall Core Idea:**
The core idea is to address **visual attention misalignment** between student and teacher models as the key barrier to effectively distilling visual perception. The proposed framework, CompoDistill, explicitly aligns the student’s visual attention with that of the teacher to enhance visual perception abilities. This involves two main technical strategies:
1.  **Visual ATtention alignment (VAT):** Aligning the attention distributions over visual tokens in the "visual understanding layers" (intermediate layers) using a group matching strategy.
2.  **Teacher Adapter Fetch (TAF):** Bridging the feature space gap between the teacher and student to make the attention transfer effective and compatible, ensuring the imposed attention mechanism does not conflict with the student's inherent visual feature space.

**Design Philosophy:**
The design philosophy rests on the empirical finding that the teacher-student visual attention similarity in the visual understanding layers is the key factor for distilling visual abilities. Since the teacher’s visual attention is optimized for its own vision-language space, a simple transfer is ineffective. Thus, the philosophy is to first ensure the student processes visual input through the same "lens" as the teacher (using TAF) and then align the attention mechanism (using VAT). Furthermore, a meticulously designed three-stage training strategy (DPT, DFT, SFT) is employed to progressively stabilize the feature space, distill the attention mechanism, and consolidate the knowledge.

### 4. Core Innovation Points

1.  **Identification of Visual Attention Misalignment as a Bottleneck:** For the first time, the study identifies that existing KD methods in MLLMs fail to distill visual perception ability due to visual attention misalignment. It establishes that moderate teacher-student attention similarity in visual understanding layers (30-70% of total layers) is the cause of failure in CR tasks, while high similarity correlates with success in VQA.
2.  **Visual ATtention alignment (VAT) Module with Group Layer Matching:** A novel module that explicitly aligns the student’s visual attention with the teacher’s. It introduces a simple yet effective "Group Layer Matching" strategy, where each student layer is matched with a group of teacher layers in a one-to-many manner (sliding window approach) to capture broader teacher knowledge and handle differences in layer depth, outperforming simple or adaptive one-to-one matching.
3.  **Teacher Adapter Fetch (TAF) Module:** A module designed to bridge the feature space gap between the teacher and student. By reusing the teacher’s frozen, pretrained adapter and adding a lightweight trainable MLP for dimensional alignment, it ensures the student processes the visual input in a space compatible with the teacher’s attention mechanism, creating synergy with VAT.
4.  **Three-Stage Distillation Framework:** A comprehensive training strategy consisting of Distilled Pre-Training (DPT) to align feature spaces, Distilled Fine-Tuning (DFT) to transfer attention mechanisms, and Supervised Fine-Tuning (SFT) to consolidate knowledge and strengthen instruction-following capabilities.

### 5. Overview of the Overall Technical Solution

The overall technical solution involves a specific architecture and training pipeline for MLLMs (based on LLaVA design).
*   **Inputs:** An image $I$ and a text query $Q$.
*   **Processing:**
    *   The vision encoder extracts patch-level features $z_p$.
    *   The **TAF Module** processes these features. It uses the teacher's frozen adapter $P^t_{\psi_t}$ followed by the student's trainable adapter $P^s_{\psi_s}$ to generate student visual tokens $x_v$ that are compatible with the teacher's space.
    *   The LLM processes the concatenated sequence of visual and text tokens.
    *   During training, the **VAT Module** operates on the intermediate (visual understanding) layers of the LLM. It extracts the attention sub-matrices over visual tokens $\tilde{A}$ for both student and teacher.
    *   **Group Layer Matching** is applied to align the student layers with groups of teacher layers. The average attention matrix of the teacher group is computed.
    *   The **VAT Loss** minimizes the cosine distance between the student's attention and the averaged teacher group's attention.
*   **Training Stages:**
    1.  **DPT (Stage 1):** Align visual feature space with language space. Train only student adapter with $L_{LM} + L_{KL}$.
    2.  **DFT (Stage 2):** Enhance visual perception. Fine-tune LLM and student adapter with $L_{LM} + L_{KL} + L_{ADL}$.
    3.  **SFT (Stage 3):** Consolidate knowledge. Fine-tune LLM and student adapter with $L_{LM}$ only.

### 6. Detailed Module Design

**6.1 Visual ATtention alignment (VAT) Module**

*   **Target Layers:** The module targets the "visual understanding layers," defined as the intermediate layers (30–70% range of the total LLM layers). These layers are crucial for visual-semantic integration.
*   **Attention Matrix Extraction:** For input tokens (visual $x_v$ and text $x_t$), the attention matrix is $A = \text{softmax}(QK^\top/\sqrt{d}) \in \mathbb{R}^{(N_v+N_t)\times(N_v+N_t)}$. The module extracts a sub-matrix $\tilde{A} \in \mathbb{R}^{(N_v+N_t)\times N_v}$ that focuses on visual attention by taking the columns corresponding to visual tokens: $\tilde{A}^l = A^l[:, : N_v]$.
*   **Group Layer Matching Strategy:**
    *   Let the sequence of student’s visual understanding layer indices be $L_S = \{l^1_s, \dots, l^j_s, \dots, l^k_s\}$ and teacher be $L_T = \{l^1_t, \dots, l^j_t, \dots, l^m_t\}$, where $k < m$.
    *   For each student layer $l^j_s$, define a group of teacher layers $G_j$ consisting of $n$ consecutive teacher layers, formed using a sliding window approach.
    *   The size $n$ is defined in closed form as $m - k + 1$ to ensure the use of all teacher layers.
    *   For example, $G_1$ for $l^1_s$ is $\{l^1_t, l^2_t, \dots, l^n_t\}$, and $G_2$ for $l^2_s$ is $\{l^2_t, l^3_t, \dots, l^{n+1}_t\}$.
    *   The average attention matrix for the teacher group is computed as $\bar{A}^j_t = \frac{1}{n}\sum_{l \in G_j} \tilde{A}^l_t$.
*   **Attention Distillation Loss ($L_{ADL}$):** The objective is to minimize the cosine distance between the student layer’s attention matrix and the averaged attention matrices of its corresponding teacher group.
    $$L_{ADL} = 1 - \frac{1}{k}\sum_{j=1}^k \text{sim}(\bar{A}^j_t, \tilde{A}^{l^j_s}_s)$$
    where $\text{sim}(\cdot, \cdot)$ denotes cosine similarity. Cosine similarity is chosen over MSE or KL Divergence because it is more crucial for the student to learn the relative importance among visual patches rather than forcing an exact match of absolute values.

**6.2 Teacher Adapter Fetch (TAF) Module**

*   **Mechanism:** The adapter projects the visual space into the language space of the LLM. Since the teacher’s attention mechanism is tightly coupled with its own adapter, imposing it on a student with an incompatible space creates a conflict. The TAF module bridges this gap by directly leveraging the teacher’s frozen, pretrained adapter ($P^t_{\psi_t}$) and adding a lightweight trainable MLP ($P^s_{\psi_s}$) only for dimensional alignment.
*   **Student Visual Token Generation:** The student’s visual token is expressed as:
    $$x_v = P^s_{\psi_s}(P^t_{\psi_t}(z_p)) \in \mathbb{R}^{N_v \times d_s}$$
    where $z_p$ is the output of the vision encoder, $P^t_{\psi_t}$ is the frozen teacher adapter, and $P^s_{\psi_s}$ is the trainable student adapter. This ensures the student processes the visual input through the same lens as the teacher.

**6.3 Three-Stage Distillation Framework**

*   **Stage 1: Distilled Pre-Training (DPT):**
    *   **Goal:** Align the visual feature space with the language space.
    *   **Optimization:** Vision encoder and LLM are frozen. The student adapter $P^s_{\psi_s}$ is optimized using the language modeling loss ($L_{LM}$) and the KL-divergence loss ($L_{KL}$).
    *   **Objective:** $L_{LM} + L_{KL}$.

*   **Stage 2: Distilled Fine-Tuning (DFT):**
    *   **Goal:** Enhance the student’s visual perception by aligning its visual attention mechanism with the teacher’s via VAT.
    *   **Optimization:** Vision encoder is frozen. Both the student LLM and the student adapter $P^s_{\psi_s}$ are fine-tuned.
    *   **Objective:** $L_{LM} + L_{KL} + L_{ADL}$.

*   **Stage 3: Supervised Fine-Tuning (SFT):**
    *   **Goal:** Consolidate the rich knowledge transferred from the teacher into the student’s own parameters and strengthen instruction-following capability.
    *   **Optimization:** Vision encoder is frozen. The student LLM and the student adapter $P^s_{\psi_s}$ are trained.
    *   **Objective:** $L_{LM}$ only.

### 7. All Mathematical Formulas and Symbol Definitions

**Equation 1: Language Modeling Probability**
$$p(y_{1:K}) = \prod_{i=1}^K p(y_i | x_v, x_t, y_{<i})$$
Where:
*   $y_{<i}$: Sequence of answer tokens up to the $i$-th token.
*   $K$: Answer length.
*   $x_v$: Visual tokens.
*   $x_t$: Text tokens.

**Equation 2: Attention Distillation Loss ($L_{ADL}$)**
$$L_{ADL} = 1 - \frac{1}{k}\sum_{j=1}^k \text{sim}(\bar{A}^j_t, \tilde{A}^{l^j_s}_s), \quad \bar{A}^j_t = \frac{1}{n}\sum_{l \in G_j} \tilde{A}^l_t$$
Where:
*   $k$: Total number of visual understanding layers for the student.
*   $j$: Index of the visual understanding layer.
*   $\text{sim}(\cdot, \cdot)$: Cosine similarity.
*   $\bar{A}^j_t$: Averaged attention matrix of the teacher group $G_j$.
*   $\tilde{A}^{l^j_s}_s$: Attention matrix of the student layer $l^j_s$.
*   $n$: Size of the teacher layer group ($m - k + 1$).
*   $G_j$: Group of teacher layers matched to student layer $l^j_s$.
*   $\tilde{A}^l_t$: Attention sub-matrix of teacher layer $l$.

**Equation 3: Student Visual Token Generation (TAF)**
$$x_v = P^s_{\psi_s}(P^t_{\psi_t}(z_p)) \in \mathbb{R}^{N_v \times d_s}$$
Where:
*   $P^t_{\psi_t}$: Frozen pretrained teacher adapter.
*   $P^s_{\psi_s}$: Trainable lightweight student adapter (MLP).
*   $z_p$: Patch-level features from vision encoder ($V_\phi(I) \in \mathbb{R}^{N_v \times d_p}$).
*   $d_s$: Hidden dimension of the student LLM.

**Equation 4: KL Divergence Loss ($L_{KL}$)**
$$L_{KL} = - \frac{1}{K}\sum_{k=1}^K \sum_{n=1}^N p^t(y_n|x_v, x_t, y_{<k}) \log \frac{p^t(y_n|x_v, x_t, y_{<k})}{p^s(y_n|x_v, x_t, y_{<k})}$$
Where:
*   $p^t$: Teacher’s predictive distribution.
*   $p^s$: Student’s predictive distribution.
*   $y_n$: $n$-th vocabulary token.
*   $N$: Total vocabulary size.
*   $K$: Answer length.

**Other Symbol Definitions:**
*   $I$: Input image ($\mathbb{R}^{H \times W \times 3}$).
*   $Q$: Text query.
*   $V_\phi(\cdot)$: Vision encoder.
*   $P_\psi(\cdot)$: Adapter.
*   $LM_\theta(\cdot)$: Pre-trained LLM.
*   $z_p$: Visual patch features ($\mathbb{R}^{N_v \times d_p}$).
*   $x_v$: Visual tokens ($\mathbb{R}^{N_v \times d}$).
*   $x_t$: Text tokens ($\mathbb{R}^{N_t \times d}$).
*   $A$: Attention matrix ($\mathbb{R}^{(N_v+N_t)\times(N_v+N_t)}$).
*   $\tilde{A}$: Visual attention sub-matrix ($\mathbb{R}^{(N_v+N_t)\times N_v}$).
*   $L_S$: Sequence of student’s visual understanding layer indices.
*   $L_T$: Sequence of teacher’s visual understanding layer indices.

### 8. Algorithm Pseudocode

The paper does not provide a formal algorithm pseudocode block, but the logic is fully described in the methodology sections. The algorithmic flow for the proposed three-stage framework is as follows:

**Algorithm: CompoDistill Training Logic**

1.  **Input:** Teacher MLLM (Vision Encoder $V^t_\phi$, Adapter $P^t_{\psi_t}$, LLM $LM^t_\theta$), Student LLM $LM^s_\theta$, Vision Encoder $V^s_\phi$, Training Data $D$.
2.  **Initialize:** Student Adapter $P^s_{\psi_s}$ (random), Student LLM $LM^s_\theta$ (pre-trained).
3.  **Construct TAF:** Define Student Visual Tokens as $x_v = P^s_{\psi_s}(P^t_{\psi_t}(z_p))$.
4.  **Stage 1: Distilled Pre-Training (DPT)**
    *   Freeze $LM^s_\theta$ and $V^s_\phi$.
    *   **For** each batch in $D$ **do**:
        *   Compute Loss $L_{DPT} = L_{LM} + L_{KL}$.
        *   Update $P^s_{\psi_s}$ via gradient descent.
5.  **Stage 2: Distilled Fine-Tuning (DFT)**
    *   Freeze $V^s_\phi$. Unfreeze $LM^s_\theta$.
    *   **For** each batch in $D$ **do**:
        *   Identify visual understanding layers in $LM^s_\theta$ and $LM^t_\theta$ (30-70% range).
        *   Extract visual attention sub-matrices $\tilde{A}^l_s$ for student layers and $\tilde{A}^l_t$ for teacher layers.
        *   Perform Group Layer Matching to define groups $G_j$ and compute $\bar{A}^j_t$.
        *   Compute Attention Distillation Loss $L_{ADL}$ using Eq 2.
        *   Compute Total Loss $L_{DFT} = L_{LM} + L_{KL} + L_{ADL}$.
        *   Update $LM^s_\theta$ and $P^s_{\psi_s}$ via gradient descent.
6.  **Stage 3: Supervised Fine-Tuning (SFT)**
    *   Freeze $V^s_\phi$. Keep $LM^s_\theta$ and $P^s_{\psi_s}$ trainable.
    *   **For** each batch in $D$ **do**:
        *   Compute Loss $L_{SFT} = L_{LM}$.
        *   Update $LM^s_\theta$ and $P^s_{\psi_s}$ via gradient descent.
7.  **Output:** Trained Student MLLM.