**1. Research Background and Existing Pain Points**

**Research Background:**
Multimodal Large Language Models (MLLMs) have stimulated substantially growing research interests and efforts in recent years. Existing architectures for MLLMs usually consist of a pretrained vision encoder that extracts visual information and a projection module that maps visual information to visual tokens for image understanding and reasoning with Large Language Models (LLMs). Projection modules such as multilayer perceptrons (MLPs) or Q-Formers bridge visual features to the semantic space of LLMs, whereas vision encoders are primarily for the ability of capturing visual information for MLLMs. Vision Transformers (ViTs) and their variants have been widely adopted as vision encoders in MLLMs due to their exceptional capabilities in visual feature extraction and scalability. Existing ViTs are usually trained to align visual representations with textual semantics. Contrastive Language–Image Pre-training (CLIP) is one prevailing paradigm that projects images and texts into a learned shared embedding space to aggregate matched image-text pairs, and non-matching pairs are discriminated to preserve the semantic relationship. Another popular alternative is autoregressive modeling that directly maps visual features into the textual space by connecting a cascaded vision encoder and textual decoder.

**Existing Pain Points:**
1. **Over-emphasis on Global Features:** Existing approaches (contrastive learning and autoregressive modeling) over-emphasize image-level global feature extraction but neglect essential fine-grained details required for multimodal understanding. They focus on global image representations but overlook fine-grained regional analysis.
2. **Data Scarcity:** There is a scarcity of high-quality datasets with fine-grained annotated data, which limits the fine-grained perception capability of vision encoders.
3. **Lack of Fine-grained Pre-training Paradigm:** There is a lack of a dedicated framework to train fine-grained vision encoders that effectively align with LLMs. Previous vision encoders struggle with fine-grained feature extraction for local regions, while also lacking alignment between visual features and the textual feature of the LLM.

**2. Core Research Motivation and Scientific Questions**

**Core Research Motivation:**
To address the limitation of existing vision encoders overlooking fine-grained regional analysis, this paper investigates integrating fine-grained localization capabilities into the vision encoder within an LLM cascade architecture. The motivation is to build a vision encoder that not only extracts global features but also possesses strong fine-grained perception and localization capabilities, aligning these fine-grained visual features with the semantic space of LLMs via region-level autoregressive training.

**Scientific Questions:**
1. How to address the challenge of data scarcity, specifically the scarcity of high-quality datasets with fine-grained annotations for local regions?
2. How to design a dedicated fine-grained pre-training framework that effectively trains vision encoders to extract fine-grained features and successfully aligns these features with LLMs for localized region recognition and grounding?

**3. Overall Core Idea and Design Philosophy**

**Overall Core Idea:**
The paper proposes GranViT, a novel Vision Transformer that integrates fine-grained feature extraction with semantic alignment to Large Language Models (LLMs) via region-level autoregressive training. 

**Design Philosophy:**
1. **Data Construction for Fine-grained Supervision:** Construct a large-scale, high-quality annotated dataset (Gran-29M) containing both global and region-level annotations (bounding boxes and local captions) for natural and OCR images to enable large-scale fine-grained pretraining.
2. **Pretraining-Adaptation Framework:** Design a two-stage framework to decouple the optimization of the vision encoder and the LLM. In the pretraining stage, optimize the vision encoder with bounding-box-to-caption (Bbox2Caption) regression for fine-grained feature extraction. In the adaptation stage, tune the LLM with caption-to-bounding-box (Caption2Bbox) regression to improve vision feature utilization and localization for the LLM.
3. **Explicit Localization Constraints:** Incorporate a self-distillation mechanism that imposes explicit localization constraints on the vision encoder to strengthen its regional reasoning capability, addressing the issue where vision encoders are only implicitly supervised through caption loss.
4. **Transferability:** Ensure the vision encoder provides fine-grained features while the LLM is capable of utilizing these features, making the vision encoder transferable to varying LLMs of comparable or larger sizes.

**4. Core Innovation Points**

1. **Gran-29M Dataset:** Establishing Gran-29M, a large-scale pretraining dataset comprising 29 million natural and OCR images paired with over 180 million high-quality region-level annotations (183.55 million localized region annotations). This dataset addresses the data scarcity issue by providing comprehensive global annotations and fine-grained captions with rigorous filtering.
2. **Pretraining-Adaptation Framework:** Proposing a pretraining-adaptation framework that simultaneously enhances the fine-grained feature extraction ability of GranViT with Bbox2Caption regression using explicit local region supervision and adapts to varying LLMs with stronger capacity for local region localization with Caption2Bbox regression.
3. **Localized Self-Distillation Mechanism:** Developing a localized self-distillation mechanism to optimize the vision encoder by imposing explicit localization constraints, augmenting its ability to extract fine-grained features through a frozen teacher vision encoder supervising the student vision encoder's local region features.
4. **Strong Transferability and Generalization:** Demonstrating robustness and generalization ability compared with existing vision encoders. GranViT achieves state-of-the-art performance on visual grounding and OCR comprehension, and attains strong transferability to varying LLMs (e.g., Qwen2.5-3B, Qwen2.5-7B, LLaMA3-8B) through the separated pretraining and adaptation design.

**5. Overview of the Overall Technical Solution**

The overall technical solution consists of dataset construction, data reformatting, and a two-stage pretraining-adaptation framework followed by supervised fine-tuning (SFT):
1. **Gran-29M Construction:** Collect natural images from UMG-41M, LAION, and FLICKR30k, and OCR images from public sources. Use ViTDet and Qwen2.5-VL-7B for bounding box and caption generation for natural images, and PaddleOCR for OCR images. Apply rigorous filtering based on image resolution, bbox aspect ratio, area, and quantity. Reformat global and local captions into standard QA pairs (Global Caption, Bbox2Caption, Caption2Bbox) and convert absolute bbox coordinates to relative coordinates.
2. **Stage 1: Pretraining:** Tune the vision encoder and projector with the LLM frozen. Employ the Bbox2Caption task (generating localized captions from bbox coordinates) to enhance fine-grained feature extraction. Apply localized self-distillation where a teacher vision encoder supervises the student vision encoder's local region features.
3. **Stage 2: Adaptation and Transfer:** Freeze the vision encoder and tune the projector and LLM. Employ the Caption2Bbox task (predicting bbox coordinates from local captions) to further enhance the localization capability of the LLM based on fine-grained visual features and ensure transferability to other LLMs.
4. **Supervised Fine-Tuning (SFT):** Adapt the pretrained and adapted model to downstream tasks using datasets like Open-LLaVA-NeXT 1M.

**6. Detailed Module Design**

**6.1 Gran-29M Dataset Construction Module**
*   **Data Source:** 
    *   Natural Images: UMG-41M (including CC3M, IN21k, SBU, CC12M, YFCC15M, VisualGenome), LAION, FLICKR30k.
    *   OCR Images: Plain text images, chart and table images, receipt images, and rich text images from 30 distinct public datasets (e.g., Arxiv, InfoVQA, DocVQA, ChartQA, TextVQA, etc.).
*   **Data Annotations Workflow:**
    *   Natural Images: Utilize ViTDet to detect bboxes and Qwen2.5-VL-7B to generate global and local captions. For UMG-41M, directly utilize provided bbox coordinates and regenerate captions.
    *   OCR Images: Utilize PaddleOCR for localized text detection and bounding box prediction to provide accurate bboxes and textual contents simultaneously.
*   **Filtering Criteria:** For local region annotations, the shorter side of each image must be > 448 pixels; the aspect ratio of the entire image and each bbox must be between 1/3 and 3; the area of each bbox must be > 1002 square pixels; the number of bboxes per image must be at least one.
*   **Data Reformatting:** Reorganize existing global and local region captions into a standard question-answer pair structure. Convert absolute bbox coordinates into relative coordinates based on image resolutions.
    *   **Global Caption:**
        Q: Describe in detail what is shown in the image in one paragraph
        A: [global captions]
    *   **Bbox2Caption:**
        Q: Describe the content contained within the normalized bounding box coordinates [bbox coordinates] in no more than 10 words.
        A: [local captions]
    *   **Caption2Bbox:**
        Q: Please provide the bounding box coordinate of the region this sentence describes: [local captions]
        A: [bbox coordinations]

**6.2 Stage 1: Fine-Grained Pretraining Module**
*   **Objective:** Enhance the fine-grained feature extraction capability of the vision encoder.
*   **Components:** Vision Encoder (Student), Projector, LLM (Frozen).
*   **Mechanism:** The framework cascades the vision encoder with the LLM via the projector. The LLM is viewed as a decoder that converts visual features into texts. The input prompt to the LLM incorporates the bbox coordinates. Supervision is directly propagated back to the extracted local features with bboxes via the Bbox2Caption task. A lightweight LLM (e.g., Qwen2.5-VL-1.5B) is employed to compel the vision encoder to extract generic fine-grained features, rather than relying on the powerful reasoning capabilities of large LLMs.
*   **Localized Self-Distillation:** An additional frozen teacher vision encoder is introduced to constrain the feature generated by the student vision encoder. Images $x$ are fed to the student vision encoder to extract fine-grained features $x'$. Image regions $x_{crop}$ are cropped from $x$ according to the bbox coordinates and sent to the teacher vision encoder for localized feature extraction $x'_{crop}$. The student vision encoder's weights are updated by optimizing the distillation loss between $x'_{crop}$ and the ROIAligned student features. The teacher vision encoder weights are initialized from the student and updated by EMA.

**6.3 Stage 2: Adaptation and Transfer Module**
*   **Objective:** Enhance the localization capability of the LLM based on fine-grained visual features and ensure the transferability of the vision encoder to other LLMs.
*   **Components:** Vision Encoder (Frozen), Projector, LLM (Tunable).
*   **Mechanism:** Employ the Caption2Bbox task, requiring the MLLM to recognize objects present in the image according to prompts and output their bbox coordinates. The vision encoder is kept frozen because it has already been sufficiently pretrained in Stage 1; further tuning it would significantly increase adaptation and transfer costs but yield marginal performance gains. The LLM is tuned to better use the fine-grained vision feature for the Caption2Bbox task. In Stage 1, the frozen LLM restricts learning for the language-reliant Caption2Bbox task (low ACC@IOU0.5), whereas in Stage 2, training the LLM leads to a notable increase in Caption2Bbox accuracy.

**7. All Mathematical Formulas and Symbol Definitions**

**Autoregressive Caption Loss:**
$L_{caption} = CrossEntropy(O_{LLM}, T)$
Where:
*   $T$: Ground truth text.
*   $O_{LLM}$: Output text of the LLM.

**Self-Distillation Loss:**
$L_{distill} = MSE(x'_{crop}, ROIAlign(x'))$
Where:
*   $x$: Input images.
*   $x'$: Fine-grained features extracted by the student vision encoder from $x$.
*   $x_{crop}$: Image regions cropped from $x$ according to the bbox coordinates.
*   $x'_{crop}$: Extracted features of $x_{crop}$ by the teacher vision encoder.
*   $MSE$: Mean square error loss.
*   $ROIAlign$: Region of Interest Alignment operation applied to $x'$.

**Overall Loss in Stage 1:**
$L = L_{caption} + \lambda L_{distill}$
Where:
*   $\lambda$: Weighting coefficient (default is 1).

**Teacher Vision Encoder Update Rule (Exponential Moving Average):**
$\theta_{tea} = \alpha\theta_{tea} + (1-\alpha)\theta_{stu}$
Where:
*   $\theta_{tea}$: Weights of the teacher vision encoder.
*   $\theta_{stu}$: Weights of the student vision encoder.
*   $\alpha$: EMA coefficient (default is 0.9).

**8. Algorithm Pseudocode**

The paper does not provide explicit algorithm pseudocode in a standard block format, but the logical algorithmic flow and implementation plan are explicitly detailed in Sections 3, 4.1, and 4.3. The algorithmic flow is extracted and structured as follows:

**Algorithm: GranViT Pretraining and Adaptation**
---
**Input:** 
Image dataset $D$ with global captions, local captions, and bounding box coordinates.
Vision Encoder (Student) $E_{stu}$, Teacher Vision Encoder $E_{tea}$, Projector $P$, LLM $L$.
Hyperparameters: learning rate $\eta$, weighting coefficient $\lambda$, EMA coefficient $\alpha$.

**Output:** 
Trained Vision Encoder $E_{stu}$, adapted Projector $P$ and LLM $L$.

**Procedure:**
// Stage 1: Pretraining
Initialize $E_{tea}$ weights with $E_{stu}$ weights.
Freeze LLM $L$.
**for** each batch of samples $(x, T_{global}, T_{local}, bbox)$ **do**
    Extract features: $x' \leftarrow E_{stu}(x)$
    Crop local regions: $x_{crop} \leftarrow Crop(x, bbox)$
    Extract teacher features: $x'_{crop} \leftarrow E_{tea}(x_{crop})$
    
    // Forward pass for Caption Loss
    Generate output text via LLM: $O_{LLM} \leftarrow L(P(x'), prompts)$
    Calculate Caption Loss: $L_{caption} \leftarrow CrossEntropy(O_{LLM}, T)$ where $T$ corresponds to global or Bbox2Caption targets.
    
    // Calculate Self-Distillation Loss
    Calculate Distillation Loss: $L_{distill} \leftarrow MSE(x'_{crop}, ROIAlign(x'))$
    
    // Optimization Step
    Calculate Total Loss: $L_{total} \leftarrow L_{caption} + \lambda L_{distill}$
    Update $E_{stu}$ and $P$ via optimizer using $L_{total}$.
    
    // EMA Update for Teacher
    $\theta_{tea} \leftarrow \alpha\theta_{tea} + (1-\alpha)\theta_{stu}$
**end for**

// Stage 2: Adaptation and Transfer
Freeze Vision Encoder $E_{stu}$.
**for** each batch of samples $(x, T_{global}, T_{local}, bbox)$ **do**
    Extract features: $x' \leftarrow E_{stu}(x)$
    
    // Forward pass for Caption Loss (including Caption2Bbox targets)
    Generate output text via LLM: $O_{LLM} \leftarrow L(P(x'), prompts)$
    Calculate Caption Loss: $L_{caption} \leftarrow CrossEntropy(O_{LLM}, T)$ where $T$ corresponds to global or Caption2Bbox targets (relative bbox coordinates).
    
    // Optimization Step
    Update $P$ and $L$ via optimizer using $L_{caption}$.
**end for**

**Return** $E_{stu}, P, L$