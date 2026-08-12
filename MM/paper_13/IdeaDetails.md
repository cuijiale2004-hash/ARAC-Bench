# 1. Research Background and Existing Pain Points

**Research Background:**
Unified multimodal Large Language Models (LLMs) that integrate both image understanding and image generation capabilities have become increasingly prominent within the research community. The appeal of this paradigm stems from the discovery that integrating these domains unlocks emergent capabilities in generation, such as complex world reasoning, multimodal instruction following, and iterative visual editing. Models like GPT-4o demonstrate that embedding image generation capabilities directly into an autoregressive LLM can lead to stronger instruction following, improved text rendering, multi-turn visual editing, and sophisticated world knowledge reasoning.

**Existing Pain Points:**
1.  **Performance Trade-off:** In practice, adding generation often degrades understanding. Existing unified models consistently lag far behind their understanding-only counterparts, especially on text-rich benchmarks (such as DocVQA, ChartQA, and InfoVQA) that demand precise perceptual capabilities.
2.  **Conflicting Nature of Visual Tokenization:** Auto-regressive generation usually prefers discrete image tokens, while understanding typically benefits from continuous embeddings.
3.  **Dual-Tokenizer Task Conflict:** Many models adopt a dual-tokenizer strategy, using a semantic encoder for continuous features while a separate quantized tokenizer (like VQ-VAE) handles generation. This forces the language model to process two different image token types—one from a high-level semantic space versus one from a low-level spatial space—creating a significant task conflict within the LLM.
4.  **Parameter Inefficiency of Mitigation Strategies:** Solutions like Mixture-of-Transformers (MoT) can mitigate task conflict by dedicating separate pathways for each task, but they are parameter-inefficient and are often incompatible with modern Mixture-of-Experts (MoE) architectures.
5.  **Decoupling Limitations:** Bypassing the conflict by freezing a pre-trained multimodal LLM and connecting it to a diffusion decoder preserves understanding capability, but it decouples generation. This loses potential mutual benefits and limits potential gains for generation from scaling the multimodal LLM.

# 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:**
The core motivation is to overcome the performance trade-off between understanding and generation in unified multimodal models. There is a need for a simple, scalable framework that harmonizes the representations for understanding and generation, enabling scalable joint learning of both capabilities without degrading performance on text-rich understanding tasks.

**Scientific Questions:**
1.  How to design a visual tokenizer that satisfies the distinct requirements of auto-regressive generation (discrete tokens) and visual understanding (continuous embeddings) without introducing the task conflict caused by heterogeneous feature spaces?
2.  How to unify the training objective and architecture to allow a single LLM to process both text and image modalities effectively, enabling mutual benefits between understanding and generation?
3.  How to decouple high-level semantic prediction from pixel-level synthesis to facilitate independent scaling of the language model and the image decoder?

# 3. Overall Core Idea and Design Philosophy

**Overall Core Idea:**
Manzano is a simple and scalable unified framework that substantially reduces the tension between understanding and generation by coupling a hybrid image tokenizer with a well-curated training recipe. A single shared vision encoder feeds two lightweight adapters that produce continuous embeddings for image-to-text understanding and discrete tokens for text-to-image generation within a common semantic space. A unified autoregressive LLM predicts high-level semantics in the form of text and image tokens, with an auxiliary diffusion decoder subsequently translating the image tokens into pixels.

**Design Philosophy:**
1.  **Unified Hybrid Representation:** Both branches (continuous and discrete) originate from the same encoder backbone; thus, continuous and discrete tokens inhabit a common semantic space, which reduces potential task conflict.
2.  **Simplicity and Scalability:** The design keeps training losses standard and components cleanly decoupled. The clear split between semantic prediction (LLM decoder) and detail generation (image decoder) supports independent scaling of the base LLM and the image decoder. The unified LLM decoder uses a single AR objective for text-only, I2T, and T2I tasks without additional auxiliary losses or per-task heads.
3.  **Focus on Semantics over Pixels:** The LLM decoder focuses on regressing high-level semantics (text and image tokens), while the diffusion decoder is responsible for rendering high-fidelity details in pixel space. Delegating pixel-level synthesis to the diffusion decoder allows the LLM to focus more on modeling semantics.

# 4. Core Innovation Points

1.  **Hybrid Image Tokenizer with Homogeneous Source:** A single shared visual encoder feeds two lightweight, specialized adapters—a continuous adapter for understanding and a discrete adapter for generation. Because two adaptors originate from the same encoder, it yields hybrid representations from a homogeneous source, significantly mitigating task conflict in the LLM compared to dual-encoder strategies.
2.  **Unified Hybrid Representation Space:** Both continuous (for I2T) and discrete (for T2I) tokens inhabit a common semantic space, as they originate from the same encoder backbone. This design reduces the conflict between heterogeneous visual tokens that typically reside within the LLM in dual-tokenizer setups.
3.  **Decoupled Architecture for Semantic Prediction and Pixel Synthesis:** The architecture cleanly decouples semantic prediction (handled by the unified LLM decoder) and detail generation (handled by the image decoder). The LLM predicts high-level semantics via a unified AR objective, while the diffusion image decoder renders pixels conditioned on these predicted discrete image tokens. This supports independent scaling of the LLM and image decoder.
4.  **Tokenizer Pre-alignment Strategy:** The hybrid tokenizer is first pre-trained with a small LLM decoder (300M) to pre-align the image features with the LLM feature space, ensuring that the tokenizer produces features compatible with the autoregressive objective before training the main unified multimodal LLM.
5.  **Unified Autoregressive Objective:** The unified LLM decoder uses a single AR objective for text-only, I2T, and T2I tasks without additional auxiliary losses or per-task heads, simplifying unification and scaling.
6.  **Three-Stage Unified Training Recipe:** A specific training recipe consisting of a pre-training stage (web-scale data for broad cross-modal alignment), a continued pre-training stage (higher-quality data for domain-specific capabilities), and a supervised fine-tuning (SFT) stage (curated instruction data) to enhance instruction following for both understanding and generation.
7.  **DiT-Air Image Decoder Architecture:** The image decoder adopts the DiT-Air architecture, which employs a layer-wise parameter-sharing strategy that reduces the size of the standard MMDiT model by approximately 66% while maintaining comparable performance, supporting scalable high-fidelity generation.

# 5. Overview of the Overall Technical Solution

The Manzano architecture comprises three core components:
1.  **Hybrid Vision Tokenizer:** Produces both continuous and discrete visual representations. It encodes images into continuous tokens for understanding (I2T) and discrete tokens for generation (T2I), sharing the same visual encoder backbone.
2.  **Unified LLM Decoder:** Accepts text tokens and/or continuous image embeddings and auto-regressively predicts the next discrete image or text tokens from a joint vocabulary. It handles pure text, image-to-text (understanding), and text-to-image (generation) data.
3.  **Image Decoder:** Renders image pixels from the predicted discrete image tokens. It operates as an auxiliary diffusion decoder that takes the generated image tokens as conditioning to translate them into high-fidelity pixel images.

**Workflow:**
-   **Training:** The hybrid tokenizer is first pre-trained with a small LLM to align image features with the LLM feature space. Then, the unified LLM is jointly trained on a mixture of pure text, image understanding, and image generation data using a unified AR objective. The image decoder is trained to reconstruct images in pixel space from discrete image tokens using a flow-matching pipeline.
-   **Inference:** For understanding tasks, the hybrid image tokenizer extracts continuous features, which are fed with text features into the unified LLM decoder to predict the text answer. For generation tasks, the unified LLM takes text input and predicts a sequence of discrete image tokens; the image decoder then renders these tokens into image pixels.

# 6. Detailed Module Design

**6.1 Hybrid Image Tokenizer**
The tokenizer comprises three components:
1.  **Vision Backbone:** A standard vision transformer (ViT).
2.  **Continuous Adapter:** First applies a 3→3 Spatial-to-Channel (STC) layer to reduce the number of spatial tokens by a factor of 9 (e.g., from 42→42→1024 to 14→14→9216) and then uses an MLP to project each feature into the LLM feature dimension (e.g., 2048). This adapter outputs continuous embeddings used for image understanding (I2T).
3.  **Discrete Adapter:** Starts with the same STC compression step as the continuous adapter but further quantizes the features using finite scalar quantization (FSQ)—chosen for its simplicity and scalability to large codebooks (64K in experiments)—before applying an MLP projection into the LLM feature dimension. This adapter outputs discrete code indices used for image generation (T2I).

**6.2 Unified LLM Decoder**
-   Connects the hybrid image tokenizer to a standard text LLM decoder.
-   Trained on a mixture of datasets containing text, understanding, and generation data using internal pre-trained LLMs as the language backbone.
-   Uses a unified autoregressive (AR) objective for all tasks (text-only, I2T, T2I) without additional auxiliary losses.

**6.3 Image Decoder**
-   Trained on top of the pre-trained hybrid image tokenizer to reconstruct images in pixel space from discrete image tokens.
-   **Conditioning:** Given an input image, the hybrid tokenizer first encodes it into a latent representation, which serves as the conditioning input.
-   **Pipeline:** Utilizes a flow-matching pipeline that transports Gaussian noise into realistic images.
-   **Backbone:** Adopts the DiT-Air architecture, employing a layer-wise parameter-sharing strategy. Three decoder configurations are provided with parameter sizes of 0.9B, 1.75B, and 3.52B, supporting output canvas resolutions from 256 to 2048 pixels.

**6.4 Training Recipe Stages**
1.  **Pre-training Stage:** Trained on a large-scale corpus of text-only, interleaved image-text, image-to-text (IT), and text-to-image (TI) data. Builds broad cross-modal alignment.
2.  **Continued Pre-training Stage:** Trained on higher-quality IT and TI data. Strengthens domain-specific capabilities such as document and chart understanding.
3.  **Supervised Fine-tuning (SFT) Stage:** Trained on curated text, IT, and TI instruction data. Teaches the model to follow diverse user instructions and improves both understanding and generation tasks.

**6.5 Image Editing Extension**
-   For image editing, the reference image is provided simultaneously to both the LLM and the diffusion decoder.
-   The LLM is responsible for diverse instruction following and maintaining semantic coherence.
-   The diffusion decoder enforces precise pixel-level control.
-   Jointly conditioning on the reference image enables accurate semantic instruction following while preserving fine-grained visual consistency for tasks like instruction-guided editing, style transfer, inpainting, outpainting, and depth estimation.

**6.6 Evaluation Setup**
-   **Understanding:** General VQA (SeedBench, RealWorldQA, MMBench), Knowledge & Reasoning (AI2D, ScienceQA, MMMU, MathVista), Text-rich Document & Chart (ChartQA, TextVQA, DocVQA, InfoVQA, OCRBench).
-   **Generation:** Automated (GenEval, DPGBench, WISE), Human Evaluation (800 challenging prompts, assessed on structural integrity, instruction following, and aesthetic quality; grades: major issues, minor issues, or no issues; averaged across raters).

# 7. All Mathematical Formulas and Symbol Definitions

**Hybrid Image Tokenizer Dimensions and Operations:**
-   **STC Layer:** 3→3 Spatial-to-Channel (STC) layer.
-   **Spatial Token Reduction Factor:** 9.
-   **Dimension Transformation Example:** 42→42→1024 to 14→14→9216.
-   **LLM Feature Dimension:** 2048 (projected via MLP).
-   **FSQ Codebook Size:** 64K.

**Image Decoder Architecture:**
-   **Size Reduction:** Layer-wise parameter-sharing strategy reduces the size of the standard MMDiT model by approximately 66%.
-   **Flow Matching Objective:** Transports Gaussian noise into realistic images using a conditional flow matching objective.

**Unified LLM Objective:**
-   **Unified AR Objective:** Predicts the next discrete image or text tokens from a joint vocabulary using a single AR objective for text-only, I2T, and T2I tasks.

**Tokenizer Pre-alignment:**
-   **Small LLM Decoder Size:** 300M.

# 8. Algorithm Pseudocode

No algorithm pseudocode is provided in the original text.