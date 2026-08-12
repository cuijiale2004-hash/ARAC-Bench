## 1. Research Background and Existing Pain Points

**Research Background:**
Multimodal Large Language Models (MLLMs) have rapidly advanced, demonstrating strong performance on challenging video QA benchmarks. These advances reveal their ability to process temporal visual cues and perform complex reasoning over natural language queries. Such capabilities imply that MLLMs may also possess inherent localization ability in videos, enabling training-free video reasoning segmentation, a task that localizes objects corresponding to text-based queries requiring complex reasoning. Recent studies have attempted to adapt MLLMs and segmentation foundation models (e.g., SAM, SAM2) for video reasoning segmentation by joint training through efficient fine-tuning methods such as LoRA. However, these methods require model-specific training and joint optimization of two foundation models, resulting in significant computation cost and limited generalization capability.

**Existing Pain Points:**
1.  **High Computation Cost and Limited Generalization in Training-based Methods:** Existing methods that jointly train MLLMs with SAM require model-specific training and joint optimization, which is computationally expensive and limits generalization.
2.  **Noisy and Poorly Aligned Raw Attention Maps:** To directly adapt MLLMs for localization in a training-free manner, one can extract attention maps via the rollout mechanism. However, raw attention maps are noisy and poorly aligned with object regions. Irrelevant regions and visual attention sinks often dominate, reducing the relative strength of object cues.
3.  **Generalization Issues in Existing Training-free Methods (Loc-Head):** The training-free approach Loc-Head explores localization ability by selecting attention heads responsible for grounding. However, it assumes the presence of a single referring object and selects heads based on spatial entropy, making extension to multi-object and temporal video data difficult. Moreover, it relies on heuristics to mitigate the visual attention sink phenomenon (e.g., excluding heads strongly attending to the bottom row), which does not generalize across MLLMs. For example, on Qwen2VL, applying the original heuristic resulted in a score of 0.0 because attention concentrated in the right-most column.
4.  **Instability in Token-based Methods (TAM):** TAM exhibits strong sensitivity to the predicted word tokens. When object words are split into multiple tokens (e.g., "b" and "icycle" for "bicycle"), the attention map for individual tokens (like "b") can be severely misaligned with the target object, making localization highly unstable and dependent on tokenization.

## 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:**
MLLMs demonstrate strong video understanding by attending to visual tokens relevant to textual queries. The motivation is to directly adapt this inherent localization capability for video reasoning segmentation in a training-free manner. By casting video reasoning segmentation as a video QA task, one can extract attention maps via the rollout mechanism. However, since the rollout integrates signals from all heads, irrelevant regions and visual attention sinks often dominate. The core motivation is to design a mechanism that suppresses noise and enhances object-focused signals without relying on model-specific modification or training.

**Scientific Questions:**
1.  How can we extract object localization signals from MLLMs without relying on model- or task-specific design or training?
2.  How can we effectively suppress irrelevant activations and visual attention sinks in raw attention maps to highlight the target object signal?
3.  How can we leverage the distinct strengths of temporal context (video attention) and object-centric spatial details (frame attention) to achieve robust localization?
4.  How can we utilize coarse, low-resolution attention maps to generate fine-grained dense segmentation masks without training, while filtering out spurious false positives?

## 3. Overall Core Idea and Design Philosophy

**Overall Core Idea:**
The paper proposes Decomposed Attention Fusion (DecAF), which refines noisy attention maps through two mechanisms: (1) contrastive object-background fusion and (2) complementary video-frame fusion. This method suppresses irrelevant activations and enhances object-focused cues, enabling direct conversion of attention maps into coarse segmentation masks. Furthermore, it introduces attention-guided SAM2 prompting for obtaining fine-grained masks, operating entirely without retraining.

**Design Philosophy:**
1.  **Training-Free Adaptation:** Leverage the inherent reasoning and attention mechanisms of MLLMs without any fine-tuning or joint training with segmentation models.
2.  **Decomposition and Fusion:** Decompose the attention extraction process to isolate and then fuse complementary signals. Contrastive fusion isolates the object signal from the background sink, while complementary fusion combines the temporal context of video attention with the spatial precision of frame attention.
3.  **Consistency-based Filtering:** Use attention maps not only as prompts for SAM2 but also as a validation signal. An attention consistency score is proposed to measure whether the predicted masks consistently overlap with high-attention regions, effectively filtering out spurious masks from imprecise point queries.

## 4. Core Innovation Points

1.  **Vision-Aware Normalization for Attention Rollout (V-Max):** A novel normalization technique for attention rollout that computes head-wise weights based on the maximum vision attention value, reducing the effect of noisy heads and better capturing language-conditioned grounding.
2.  **Contrastive Object-Background Attention Fusion:** A mechanism that subtracts the background attention map (derived from a prompt excluding the target object) from the object attention map to effectively suppress visual attention sinks and irrelevant background activations.
3.  **Complementary Video-Frame Attention Fusion:** A multi-scale fusion strategy that combines video-level attention (capturing temporal context but sparse) with frame-level attention (capturing object-centric, fine-grained cues but lacking temporal coherence) via averaging, after applying distinct normalization strategies.
4.  **Attention-Guided SAM2 Prompting with Attention Consistency Score:** A training-free pipeline that generates point queries from thresholded attention maps for SAM2, and filters the resulting mask tracklets using an attention consistency score that penalizes masks overlapping with low-attention regions.

## 5. Overview of the Overall Technical Solution

The framework produces segmentation masks of target objects given a video and a text instruction. The pipeline consists of two stages:
**Stage 1: Coarse Segmentation via DecAF:**
1.  **Attention Rollout with V-Max:** Trace the influence of visual tokens on the model’s output by propagating attention scores through transformer layers, using vision-aware head-wise weighted aggregation.
2.  **Contrastive Fusion:** Prompt the MLLM with an object-focused query and a background-focused query (excluding the identified object category). Subtract the background attention map from the object attention map and apply Gaussian smoothing and min-max normalization.
3.  **Complementary Fusion:** Apply the rollout and contrastive fusion pipeline separately to video inputs and individual frame inputs. Upsample the video-level maps to match the frame-level resolution. Fuse the two modalities by averaging.
4.  **Coarse Mask Generation:** Convert the final fused attention maps into coarse segmentation masks via thresholding (e.g., Otsu's method).

**Stage 2: Fine-Grained Dense Segmentation via SAM2:**
1.  **Point Query Generation:** Select visual tokens with attention scores above a threshold $\tau_{pq}$ and use their center coordinates as point queries.
2.  **Frame-wise Prompting and Propagation:** Feed point queries sequentially to SAM2 to produce mask tracklets and confidence scores. Apply Non-Maximum Suppression (NMS) based on an object score to remove redundant masks.
3.  **Mask Tracklet Scoring:** Evaluate each tracklet using an attention consistency score ($s_{ac}$) that measures alignment between the mask and the attention map. Retain tracklets with a combined score $\geq \tau_{trk}$.

## 6. Detailed Module Design

### 6.1 Attention Rollout with Vision-Aware Normalization
This module traces the influence of visual tokens on the model's output by propagating attention scores. It modifies standard attention rollout with a vision-aware normalization scheme.
1.  **Standard Rollout:** Given the attention tensor $A^{(l)} \in \mathbb{R}^{h \times N \times N}$ from the $l$-th layer, compute the head-wise averaged attention matrix (Eq. 1). Incorporate the residual connection by adding the identity matrix (Eq. 2). Recursively accumulate across layers starting from $R^{(1)} = \hat{A}^{(1)}$ (Eq. 3).
2.  **Head-wise Weighted Aggregation (V-Max):** To reduce the effect of noisy heads, assign a weight to each head based on the strength of its vision attention. Extract the vision block $A_v^{(l)} \in \mathbb{R}^{h \times N \times N_v}$ from the original attention tensor $A^{(l)} \in \mathbb{R}^{h \times N \times (N_v+N_t)}$. Compute the maximum value over the visual token dimension to get $m^{(l)}$ (Eq. 4). Average $m^{(l)}$ over the token dimension to produce the head-wise weight vector $w^{(l)} \in \mathbb{R}^h$. Normalize weights so that $\max_h(w_h^{(l)}) = 1$. Use these weights to aggregate the heads into final attention weights $\hat{A}^{(l)}$.

### 6.2 Contrastive Object-Background Attention Fusion
This module addresses visual attention sinks by contrasting attention maps from object-focused and background-focused prompts.
1.  **Object Attention Map:** Prompt the model with: "{Expression}\nWhat is the main object (or objects) referred to in the given expression or question?\nFocus on the **primary subject or agent** involved in the described action or behavior.\nRespond with a single word (e.g., 'cat', 'person', 'dog') that best describes the target object(s)." Extract rollout weights from this response as the positive map.
2.  **Background Attention Map:** Prompt the model with: "Describe the background scene of the video, excluding any {Object category}.\nAnswer the question using a single word or phrase." Extract rollout weights as the negative map. The object category is explicitly excluded to prevent the target object from being attended in the background map.
3.  **Fusion:** Reshape both maps into $(T, H_p, W_p)$. Apply Gaussian smoothing to both to mitigate sparsity. Compute the contrastive map $V_{ctr}$ by subtracting the background map from the object map, clamping to remove negative values. Apply min–max normalization to scale values into [0, 1].

### 6.3 Complementary Video-Frame Attention Fusion
This module exploits the complementary properties of video (temporal context) and frame (spatial precision) attention.
1.  **Dual-Modality Processing:** Apply the identical attention rollout and contrastive fusion pipeline individually to the video input and frame input (each frame processed along the batch axis).
2.  **Object Category Selection:** Since background prompting requires an object category, select a single prediction by aggregating outputs from both video- and frame-level inputs using a Category Choice Prompt.
3.  **Normalization:** Normalize frame-level maps independently per frame, while normalizing video-level maps globally across all frames.
4.  **Multi-scale Fusion:** Utilize higher-resolution inputs for frame attention (e.g., tiling for InternVL/LLaVA-NeXT, or doubled width/height for QwenVL). Upsample low-res video attention maps to match the frame-level resolution. Fuse the two sets of maps by simple averaging.

### 6.4 SAM2 Prompting with Attention Maps
This module converts low-resolution attention maps into fine-grained masks using SAM2.
1.  **Point Query Generation:** Select visual tokens with attention scores $V_{t,y,x} \geq \tau_{pq}$. Define the set of point queries $P$ using token center coordinates (Eq. 5).
2.  **Propagation and NMS:** Feed points sequentially to SAM2 to generate mask tracklets $M_i$ and scores $s_i^{SAM}$. Calculate object score $s_i^{obj} = V_{p_i} + s_i^{SAM}$. Apply NMS: two masks overlap if IoU > 0.7; retain the one with the higher score. If a propagated mask highly overlaps with a new mask, retain the higher-scoring one.
3.  **Mask Tracklet Scoring:** Evaluate each tracklet using attention consistency score ($s_{ac}$) to filter false positives where SAM produces high-confidence masks from spurious background points. Compute a binary mask $M^{\text{Attn}}$ using the mean attention score $\mu_t$ as a threshold (Eq. 6). Assign a penalty $\delta_t = -\max(V_{t,:,:})$ to regions below $\mu_t$ to create $\hat{V}$ (Eq. 7). Downsample the mask tracklet $\tilde{M}_i$ and compute $s_{ac}$ as a ratio of inner products (Eq. 8). Calculate the combined tracklet score $s_{trk}^i = \text{Avg}(V_{p_i}, s_i^{SAM}, s_{ac}^i)$. Retain tracklets with $s_{trk}^i \geq \tau_{trk}$ and propagate across all video frames.

## 7. All Mathematical Formulas and Symbol Definitions

**Symbol Definitions:**
*   $A^{(l)}$: Attention tensor from the $l$-th transformer layer.
*   $h$: Number of attention heads.
*   $N$: Total number of tokens ($N_v + N_t$).
*   $N_v, N_t$: Number of visual and textual tokens, respectively.
*   $A_v^{(l)}$: Vision block of the attention tensor, $\in \mathbb{R}^{h \times N \times N_v}$.
*   $I$: Identity matrix.
*   $\bar{A}^{(l)}$: Head-wise averaged attention matrix.
*   $\hat{A}^{(l)}$: Attention matrix with residual connection / Normalized attention weights after head aggregation.
*   $R^{(l)}$: Rollout matrix at layer $l$.
*   $m^{(l)}$: Maximum value over the visual token dimension for head weighting, $\in \mathbb{R}^{h \times N}$.
*   $w^{(l)}$: Head-wise weight vector, $\in \mathbb{R}^{h}$.
*   $V$: Attention map / Visual token scores, $\in \mathbb{R}^{T_s \times H_p \times W_p}$.
*   $T_s$: Number of sampled frames.
*   $H_p, W_p$: Spatial resolution of visual token grid.
*   $o_x, o_y$: Half the token width and height, respectively.
*   $P$: Set of point queries.
*   $\tau_{pq}$: Threshold for point query generation.
*   $M_i$: Video mask tracklet for point query $p_i$, $\in \mathbb{R}^{T_s \times H \times W}$.
*   $s_i^{SAM}$: Confidence score predicted by SAM2 for tracklet $i$.
*   $V_{p_i}$: Attention score of the point query $p_i$.
*   $s_i^{obj}$: Object score for tracklet $i$.
*   $s_{ac}^i$: Attention consistency score for tracklet $i$.
*   $s_{trk}^i$: Combined tracklet score.
*   $\tau_{trk}$: Threshold for retaining tracklets.
*   $\mu_t$: Mean attention score per frame $t$.
*   $M^{\text{Attn}}$: Binary mask derived from thresholding attention map.
*   $\delta_t$: Negative maximum attention score per frame ($-\max(V_{t,:,:})$).
*   $\hat{V}$: Attention map with penalty applied to low-attention regions.
*   $\tilde{M}_i$: Downsampled mask tracklet, $\in \mathbb{R}^{T_s \times H_p \times W_p}$.
*   $\langle \cdot, \cdot \rangle$: Tensor inner product.

**Mathematical Formulas:**

$$ \bar{A}^{(l)} = \frac{1}{h} \sum_{i=1}^h A_i^{(l)} \quad \text{(1)} $$

$$ \hat{A}^{(l)} = (\bar{A}^{(l)} + I)/2 \quad \text{(2)} $$

$$ R^{(l)} = \hat{A}^{(l)} R^{(l-1)} \quad \text{(3)} $$

$$ m^{(l)} = \max_{j=1}^{N_v} A_v^{(l)}[:, :, j], \quad m^{(l)} \in \mathbb{R}^{h \times N} \quad \text{(4)} $$

$$ P = \{p = (t, y + o_y, x + o_x) \mid V_{t,y,x} \geq \tau_{pq}\} \quad \text{(5)} $$

$$ M^{\text{Attn}}_{t,y,x} = \begin{cases} 1, & V_{t,y,x} \geq \mu_t \\ 0, & \text{otherwise} \end{cases} \quad \text{(6)} $$

$$ \hat{V}_{t,y,x} = \begin{cases} V_{t,y,x}, & V_{t,y,x} \geq \mu_t \\ \delta_t, & \text{otherwise} \end{cases} \quad \text{(7)} $$

$$ s_{ac}^i = \frac{\langle \tilde{M}_i, \hat{V} \rangle}{\langle M^{\text{Attn}}, \hat{V} \rangle} \quad \text{(8)} $$

## 8. Algorithm Pseudocode

The paper does not provide explicit algorithm pseudocode blocks. The algorithmic flow is described as follows based on the methodology section:

**Algorithm: DecAF Pipeline**
1.  **Input:** Video $V$, Text Expression $E$.
2.  **Output:** Dense Segmentation Masks $\hat{M}$.
3.  **Process:**
    4.  // **Stage 1: Attention Extraction & Fusion**
    5.  Identify object category $C$ using $E$ and MLLM with Category Choice Prompt.
    6.  // **Contrastive Fusion (Video)**
    7.  Generate Video Object Attention Map $V_{vid}^{obj}$ using MLLM with Object-focused Prompt on $V$.
    8.  Generate Video Background Attention Map $V_{vid}^{bg}$ using MLLM with Background-focused Prompt (excluding $C$) on $V$.
    9.  Compute Video Contrastive Map: $V_{vid}^{ctr} = \text{MinMax}(\text{Smooth}(V_{vid}^{obj}) - \text{Smooth}(V_{vid}^{bg}))$.
    10. // **Contrastive Fusion (Frame)**
    11. For each frame $f$ in $V$:
    12. $\quad$ Generate Frame Object Attention Map $V_{frm}^{obj}$ using MLLM with Object-focused Prompt on $f$.
    13. $\quad$ Generate Frame Background Attention Map $V_{frm}^{bg}$ using MLLM with Background-focused Prompt (excluding $C$) on $f$.
    14. $\quad$ Compute Frame Contrastive Map: $V_{frm}^{ctr} = \text{MinMax}_{per-frame}(\text{Smooth}(V_{frm}^{obj}) - \text{Smooth}(V_{frm}^{bg}))$.
    15. // **Complementary Fusion**
    16. Upsample $V_{vid}^{ctr}$ to match resolution of $V_{frm}^{ctr}$.
    17. Compute Fused Attention Map: $V = \text{Average}(V_{vid}^{ctr}, V_{frm}^{ctr})$.
    18. // **Stage 2: SAM2 Prompting**
    19. Generate Point Queries $P = \{p = (t, y + o_y, x + o_x) \mid V_{t,y,x} \geq \tau_{pq}\}$.
    20. Initialize Tracklet Set $\mathcal{T} = \emptyset$.
    21. For each point $p_i \in P$:
    22. $\quad$ Propagate $p_i$ through SAM2 to get Mask Tracklet $M_i$ and score $s_i^{SAM}$.
    23. $\quad$ Compute Object Score $s_i^{obj} = V_{p_i} + s_i^{SAM}$.
    24. $\quad$ Add $(M_i, s_i^{obj})$ to $\mathcal{T}$.
    25. Apply NMS on $\mathcal{T}$ based on IoU and $s_i^{obj}$ (remove lower score if IoU > 0.7).
    26. For each remaining tracklet $i$ in $\mathcal{T}$:
    27. $\quad$ Compute Attention Consistency Score $s_{ac}^i$ using Eq. 6, 7, 8.
    28. $\quad$ Compute Combined Score $s_{trk}^i = \text{Avg}(V_{p_i}, s_i^{SAM}, s_{ac}^i)$.
    29. $\quad$ If $s_{trk}^i < \tau_{trk}$, discard tracklet $i$.
    30. Propagate retained tracklets across full video frames via SAM2 to generate final $\hat{M}$.
    31. Return $\hat{M}$.