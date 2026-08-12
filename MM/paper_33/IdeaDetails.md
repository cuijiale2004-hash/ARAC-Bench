**1. Research Background and Existing Pain Points**

**Research Background:**
Fine-grained image perception and understanding, which aim to associate specific image regions with contextual information (such as semantic or instance-level information), is a fundamental task in computer vision. Classical vision models remain state-of-the-art for pure detection and segmentation tasks but lack flexible language interaction and understanding, prohibiting open-vocabulary oriented visual reasoning tasks. Inspired by CLIP, many vision-language detectors incorporate language information to detect arbitrary classes. However, these methods remain vision-centric backbones augmented with language, struggling to handle complex textual descriptions and limited to structured output. Recent advances in Multimodal Large Language Models (MLLMs) couple vision encoders with Large Language Models (LLMs). Pretrained on massive multimodal datasets, these models encode rich prior knowledge and provide a strong foundation for visual perception and understanding.

**Existing Pain Points:**
To conform with the textual output space of LLMs, most existing MLLMs serialize detected regions into bounding box coordinates expressed in textual form (e.g., [x1, y1, x2, y2]). While straightforward, this strategy introduces several critical challenges:
1.  **Inconsistent Output Formats:** Output formats are often inconsistent across samples even under the same prompt (e.g., absolute vs. normalized coordinates, JSON-style vs. free-form). This increases the difficulty of parsing and structured output.
2.  **Semantic Misalignment:** Numerical coordinate representations provide precise spatial descriptions but lack semantic alignment between textual and visual modalities. This inherent misalignment can lead to repetition or hallucination between coordinate and actual visual targets.
3.  **Discontinuous Coordinate Tokens:** Since numerical coordinate representations are mapped into discrete textual tokens, a single coordinate value may be split into several unrelated tokens (e.g., “489” split into “4, 8, 9”). These discontinuous coordinate tokens can hinder prediction accuracy, leading to fragmented numbers.
4.  **Limitations of Prior Codebook Methods:** A prior art (ClawMachine) attempted to empower LLMs to output image patch tokens discretized by a global codebook. However, this remains limited in flexibility and generality due to: (i) the risk of predicting visual tokens that do not appear in the query image; (ii) the decoded visual token not having unique correspondence in the query image, risking misalignment between predicted visual tokens and query image tokens, such as confusion between similar objects.

**2. Core Research Motivation and Scientific Questions**

**Core Research Motivation:**
To overcome the challenges of indirect representations and the limitations of fixed codebooks, the research is motivated to introduce a unified paradigm that enables MLLMs to directly generate both textual and diverse visual outputs. The motivation is to ensure that visual tokens are interpretable by LLMs, being both embeddable in the input space and decodable in the output space, while preserving unique positional information for each image region to ensure coherent predictions and improve localization and differentiation among similar objects.

**Scientific Questions:**
1. How to design a tokenization scheme that embeds visual patches directly as decodable tokens within the autoregressive generation process, avoiding the limitations of a fixed dataset-level codebook?
2. How to ensure that visual reference tokens have unique correspondence in the query image to avoid ambiguity and confusion between similar objects?
3. How to convert variable predicted visual reference tokens into diverse visual representations (e.g., bounding boxes and masks) through a lightweight mechanism?
4. How to design a training strategy and loss function that stabilize training, mitigate overfitting, and encourage the model to explore diverse valid visual references?

**3. Overall Core Idea and Design Philosophy**

**Overall Core Idea:**
The paper introduces a unified paradigm, Patch-as-Decodable Token (PaDT), which enables MLLMs to directly generate both textual and diverse visual targets in a unified yet flexible way. Central to PaDT are Visual Reference Tokens (VRTs), derived from visual patch embeddings of query images and interleaved seamlessly with LLM’s output textual tokens.

**Design Philosophy:**
1.  **Dynamic Codebook Expansion:** Rather than introducing a standalone fixed codebook, the design reuses the extracted visual tokens from the input image. Since each visual token explicitly corresponds to an image patch, at each forward pass only the tokens from the current query image are dynamically expanded into the original textual codebook. This ignores patch-specific cues such as spatial location and avoids ambiguity.
2.  **Unified Input and Output Format:** With the Multi-Modal Codebook, both textual and visual information can be input and output in a unified way. VRTs are adapted from the original image tokens, sharing a feature space similar to the LLMs representation space, simplifying training.
3.  **Decoupled Visual Decoding:** The LLM is responsible for predicting a subset of VRTs. A lightweight decoder is then used to convert these predicted VRTs into task-specific structured visual outputs (detection, segmentation, grounding).
4.  **Robust Training Strategy:** To prevent the model from overfitting to a fixed set of tokens, a strategy of randomly sampling foreground visual tokens for supervision is adopted, combined with a masked supervision mechanism to improve robustness and generalization.

**4. Core Innovation Points**

1.  **Unified Paradigm (PaDT):** Introduction of a unified paradigm, Patch-as-Decodable Token (PaDT), which enables MLLMs to directly generate both textual and diverse visual targets in a unified yet flexible way. With the proposed Visual Reference Token (VRT), the method achieves superior performance across diverse fine-grained image perception and understanding.
2.  **Dynamic Multi-Modal Codebook Expansion:** Proposal of a Dynamic Embedding Module that dynamically augments the textual vocabulary codebook with visual patches (VRTs) at each forward pass. Unlike prior methods using a global codebook, this reuses extracted visual tokens from the input image, preserving rich semantic information and ensuring unique positional information for each image region.
3.  **Light-Weight PaDT Decoder:** Proposal of a lightweight yet robust VRT-based decoder, termed the PaDT Decoder. Given the generated VRTs, it can uniformly decode diverse fine-grained structured visual outputs, such as segmentation masks and bounding boxes, through a stack of two-way attention blocks and learnable task tokens.
4.  **Robust Training Strategy and Loss Function:** Proposal of an effective fine-tuning strategy together with a robust per-token cross-entropy loss. By randomly selecting VRTs for supervised fine-tuning and introducing a mechanism to mask the logits of unselected tokens, the objective improves robustness and encourages the model to explore diverse valid visual references during training.

**5. Overview of the Overall Technical Solution**

The overall technical solution extends conventional MLLMs with three key components:
1.  **Vision Encoding & Projecting:** Given an input image, a Vision Encoder (ViT) partitions the image into non-overlapping patches and encodes them into embeddings. A projector aligns dimensions and downsamples to yield patch features.
2.  **Dynamic Embedding Module:** The patch features are projected by a lightweight module (LayerNorm and low-rank linear projection) into visual reference prototypes. These prototypes are concatenated with text embeddings to form a dynamic embedding table (Multi-Modal Codebook).
3.  **LLM Backbone & PaDT Head:** The multimodal representation (fused image and text embeddings) is fed into the LLM. At the output side, the PaDT Head augments the textual classifier with the visual reference prototypes, allowing the LLM to predict both textual tokens and VRTs.
4.  **PaDT Decoder:** The hidden features of the predicted VRTs from the final LLM layer are grouped into object queries. Learnable task tokens (bounding box, mask, score) are injected into each group. After passing through three two-way attention blocks, each task token is projected into its respective output space.
5.  **Training Strategy:** The model is trained using a robust per-token cross-entropy loss where foreground VRTs are randomly sampled, and the logits of unselected tokens are masked to $-\infty$. Task-specific losses (bbox, mask, score) are applied on the decoder outputs.

**6. Detailed Module Design**

**6.1. Revisiting MLLMs**
A Multimodal Large Language Model augments a Large Language Model with a visual encoder. Given an image $I \in R^{H \times W \times 3}$ and a text sequence $T = (t_1, . . . , t_m)$, the MLLM autoregressively generates an output sequence $Y = (y_1, . . . , y_t)$. An image encoder $f_v$, typically a Vision Transformer (ViT), partitions $I$ into $N$ non-overlapping patches which are subsequently encoded into embeddings. A projector $f_p$ then aligns dimensions and downsamples, yielding $F_{patch} = f_p(F_v) \in R^{N' \times d}$. The image embeddings are fused with the text embeddings to form a hybrid textual-visual representation $Z$. At timestep $t$, the hidden state $h_t$ produces the next-token distribution.

**6.2. Dynamic Embedding Module**
To avoid the limitations of a fixed dataset-level codebook, the proposed Dynamic Embedding Module reuses the extracted visual tokens from the input image. Original patch features $F_{patch} \in R^{N' \times d}$ are projected by a lightweight module $f_{vp}$ into visual reference prototypes $P_{ref}$. The module $f_{vp}$ consists of a LayerNorm and a low-rank linear projection. These prototypes are then concatenated with text embeddings to form a dynamic embedding table (Multi-Modal Codebook).

**6.3. Unified Input and Output Format**
With the Multi-Modal Codebook, both textual and visual information can be input and output in a unified way.
*   **Input Side:** Query image tokens are indexed in the Multi-Modal Codebook and converted into the corresponding VRTs, which are then embedded into the textual input to the LLM.
*   **Output Side:** To enable the original textual classifier to output expanded indices, the PaDT Head is proposed to augment the classifier with $P_{ref}$.

**6.4. Light-Weight PaDT Decoder**
Considering that only several VRTs on a detected object are predicted, a visual decoder converts these predicted VRTs into task-specific outputs.
*   **Structure:** Implemented as a stack of three two-way attention blocks.
*   **Input:** The hidden features of predicted VRTs from the final LLM layer. These features are grouped into object queries, where each group corresponds to a sequence of VRTs separated by intervening text tokens.
*   **Task Tokens:** To enable task-specific decoding, three learnable tokens (bounding box, mask, and score tokens) are injected into each group of object queries.
*   **Output:** After passing through the three attention blocks, each task token is projected into its respective output space, producing bounding boxes, segmentation masks, and confidence scores.

**6.5. Training Strategy**
*   **VRTs Selection:** Unlike prior work that uses all foreground visual tokens as supervision, the strategy randomly samples $N_{vrt}$ foreground tokens for each forward pass. A foreground mask $M \in \{0, 1\}^{T \times N'}$ is introduced, where $M_{t,n} = 1$ indicates that token $n$ at step $t$ was not selected.
*   **Logit Masking:** For unselected tokens, their contribution to the loss is suppressed by masking their logits. Specifically, for such tokens, the logit is set to $-\infty$. This effectively removes the masked tokens from the softmax normalization, ensuring they are neither rewarded nor penalized.
*   **Task-specific Losses:** For structured outputs from the vision task decoder, task-specific objectives ($L_{bbox}, L_{mask}, L_{score}$) are adopted. The final training objective combines the robust cross-entropy loss and the task-specific losses.

**7. All Mathematical Formulas and Symbol Definitions**

**Symbol Definitions:**
*   $I \in R^{H \times W \times 3}$: Input image.
*   $T = (t_1, . . . , t_m)$: Input text sequence.
*   $Y = (y_1, . . . , y_t)$: Output sequence.
*   $f_v$: Vision Encoder (ViT).
*   $F_v = f_v(I) \in R^{N \times d_v}$: Visual embeddings from encoder.
*   $f_p$: Projector aligning dimensions and downsampling.
*   $F_{patch} = f_p(F_v) \in R^{N' \times d}$: Patch features after projection.
*   $E_{text}(T) \in R^{m \times d}$: Text embeddings.
*   $E_{text} \in R^{V_{text} \times d}$: Text embedding table.
*   $Z = [F_{patch}; E_{text}(T)]$: Hybrid textual-visual representation.
*   $h_t$: Hidden state at timestep $t$.
*   $W_{text} \in R^{V_{text} \times d}$: Classifier weights for text tokens.
*   $f_{vp}$: Lightweight projection module (LayerNorm + low-rank linear projection).
*   $P_{ref} = f_{vp}(F_{patch}) \in R^{N' \times d}$: Visual reference prototypes.
*   $E_{dyn} = [E_{text}; P_{ref}]$: Dynamic embedding table (Multi-Modal Codebook).
*   $W_{tv} = [W_{text}; P_{ref}] \in R^{(V_{text}+N') \times d}$: Augmented classifier weights (PaDT Head).
*   $M \in \{0, 1\}^{T \times N'}$: Foreground mask, where $M_{t,n} = 1$ indicates that token $n$ at step $t$ was not selected.
*   $l'_t$: Modified logits at timestep $t$.
*   $\hat{y}_t$: Ground-truth token at step $t$.

**Mathematical Formulas:**

(1) Next-token distribution in conventional MLLM:
$$p(y_t|I,T, y_{<t}) = softmax(W_{text} \cdot h_t)$$

(2) Dynamic Embedding Table construction:
$$E_{dyn} = [E_{text};P_{ref}] , P_{ref} = f_{vp}(F_{patch}) \in R^{N' \times d}$$

(3) PaDT Head classifier weights:
$$W_{tv} = [W_{text};P_{ref}] \in R^{(V_{text}+N') \times d}$$

(4) Standard Per-token Cross-Entropy Loss:
$$L_{CE} = \frac{1}{T} \sum_t -\log p(\hat{y}_t | I,T, y_{<t}) = -\log softmax_{GT} (W_{tv} \cdot h_t)$$

(5) Logit Masking for unselected tokens:
$$l'_t = W_{tv} \cdot h_t, l'_{t,n+V_{text}} = -\infty \text{ if } M_{t,n} = 1$$

(6) Robust Per-token Cross-Entropy Loss:
$$L^{robust}_{CE} = -\log softmax_{GT} (l'_t)$$

(7) Final Training Objective:
$$L = L^{robust}_{CE} + L_{bbox} + L_{mask} + L_{score}$$

(8) Bounding Box Loss:
$$L_{bbox} = \frac{1}{L} \sum_l L_{iou}(B^{pred}_l, B^{gt}_l) + ||B^{pred}_l - B^{gt}_l||_1$$

(9) Mask Loss:
$$L_{mask} = \frac{1}{L} \sum_l L_{dice}(M^{pred}_l, M^{gt}_l) + \sum_l L_{focal}(M^{pred}_l, M^{gt}_l)$$

(10) Score Loss:
$$L_{score} = \frac{1}{L} \sum_l ||S^{pred}_l - S^{gt}_l||^2_2$$

**8. Algorithm Pseudocode**

The paper does not provide explicit pseudocode. Based on the detailed description of the methodology and training strategy, the algorithmic flow of the PaDT framework is extracted as follows:

**Training Phase:**
1.  **Input:** Image $I$, Text Sequence $T$, Ground Truth Bounding Boxes/Masks $B^{gt}, M^{gt}$.
2.  **Vision Encoding:** Compute visual patch features $F_{patch} = f_p(f_v(I))$.
3.  **Dynamic Codebook Construction:**
    *   Compute visual reference prototypes: $P_{ref} = f_{vp}(F_{patch})$.
    *   Construct dynamic embedding table: $E_{dyn} = [E_{text}; P_{ref}]$.
    *   Construct augmented classifier: $W_{tv} = [W_{text}; P_{ref}]$.
4.  **Ground Truth Sequence Construction:**
    *   For each target object, randomly sample $N_{vrt}$ foreground patches.
    *   Assign VRT indices to the sampled patches and construct the target sequence $Y$ interleaving text tokens and VRTs.
    *   Generate foreground mask $M$ indicating unselected tokens.
5.  **LLM Forward Pass:**
    *   Fuse features: $Z = [F_{patch}; E_{text}(T)]$.
    *   Obtain hidden states $h_t$ for the sequence.
6.  **Logit Masking:**
    *   Compute initial logits: $l_t = W_{tv} \cdot h_t$.
    *   Apply mask: Set $l_{t,n+V_{text}} = -\infty$ if $M_{t,n} = 1$ to obtain $l'_t$.
7.  **Loss Calculation:**
    *   Compute $L^{robust}_{CE}$ using masked logits $l'_t$.
    *   Feed predicted VRT hidden features into the PaDT Decoder to obtain $B^{pred}, M^{pred}, S^{pred}$.
    *   Compute task-specific losses $L_{bbox}, L_{mask}, L_{score}$.
    *   Compute total loss: $L = L^{robust}_{CE} + L_{bbox} + L_{mask} + L_{score}$.
8.  **Update:** Backpropagate and update model parameters.

**Inference Phase:**
1.  **Input:** Image $I$, Text Prompt $T$.
2.  **Vision Encoding:** Compute $F_{patch}$.
3.  **Dynamic Codebook Construction:** Construct $E_{dyn}$ and $W_{tv}$ using $P_{ref} = f_{vp}(F_{patch})$.
4.  **Autoregressive Generation:**
    *   LLM generates tokens sequentially.
    *   If a text token is generated, map using $E_{text}$.
    *   If a VRT index is generated, map using $P_{ref}$.
5.  **Visual Decoding:**
    *   Collect the hidden features of all generated VRTs from the final LLM layer.
    *   Group VRTs into object queries based on intervening text tokens.
    *   Inject task tokens (Bbox, Mask, Score) into each group.
    *   Pass through the 3 two-way attention blocks of the PaDT Decoder.
    *   Project task tokens to output bounding boxes, segmentation masks, and scores.