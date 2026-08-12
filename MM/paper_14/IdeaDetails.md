### 1. Research Background and Existing Pain Points

**Research Background:**
The proliferation of Multimodal Large Language Models (MLLMs), such as GPT-4o, Gemini, and LLaVA, has revolutionized human-machine interaction and content understanding. These models possess an insatiable appetite for high-quality visual data. As MLLM applications become ubiquitous—ranging from real-time visual question answering on mobile devices to complex scene analysis in cloud-based services—the demand for transmitting and storing image and video data is growing at an explosive rate. The prevailing cloud-edge deployment of MLLMs creates a critical bottleneck: the conflict between the need for high-fidelity visual input and the constraints of limited communication bandwidth and storage resources. Consequently, developing highly efficient compression techniques tailored for this new era is imperative.

**Existing Pain Points:**
Existing compression techniques are ill-suited for the versatile, open-world nature of MLLMs.
1.  **Human Visual System (HVS) Optimization:** Conventional image codecs (e.g., JPEG, VVC, ELIC) are engineered for the HVS and optimized for fidelity. They discard information that is imperceptible to humans but potentially vital for machine analysis, leading to inconsistent performance across the diverse capabilities of MLLMs.
2.  **Narrow Task-Specificity of ICM:** Image Coding for Machine (ICM) methods target specific, narrow computer vision tasks (e.g., object detection, segmentation). This task-specificity is fundamentally at odds with the general-purpose nature of MLLMs.
3.  **Erratic Performance:** Both human-centric and machine-centric codecs exhibit erratic performance, excelling in some MLLM tasks while failing in others. They do not address how MLLMs holistically perceive and are affected by compression artifacts.
4.  **Loss of Cross-Level Features:** Existing ICM approaches only try to preserve high-level information but ignore low-level information. This leads to a significant loss of cross-level features, which are crucial for MLLMs to synthesize local details into coherent high-level semantics.
5.  **Finetuning Inadequacy:** Direct finetuning strategies (finetuning the codec or the MLLM) fail. Finetuning the codec with the MLLM frozen leads to training collapse, while finetuning the MLLM yields only marginal gains.

### 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:**
To design a codec tailored to MLLMs, it is necessary to systematically understand what visual information MLLMs require and how this information acquisition process is affected by compression artifacts. The fundamental motivation is that compression distortion unevenly impacts different-level image features, leading to varying effects on MLLMs’ downstream tasks. Existing methods ignore the fact that an effective codec must simultaneously preserve both low-level fidelity and high-level semantic information to support the synthesis of cross-level features.

**Scientific Questions:**
1.  How does visual information flow in MLLMs? Specifically, how does the vision encoder transform raw pixels into a feature representation that balances both low-level details and high-level semantics?
2.  How does compression distortion affect this visual information flow and MLLM performance? Why are certain tasks (like counting) highly susceptible to compression while others (like coarse semantic identification) are robust?

### 3. Overall Core Idea and Design Philosophy

**Overall Core Idea:**
The core idea is that an effective codec tailored to MLLMs must preserve multi-level visual information. The design must address the finding that compression-induced information loss compromises cross-level features, which rely on integrating low-level information with high-level context, whereas low-level and coarse high-level features are relatively robust.

**Design Philosophy:**
Based on the three-stage information flow analysis of the vision encoder, the design philosophy operates on a dual-strategy approach:
1.  **Encoder Side:** Since the initial layers of a vision encoder perform preliminary information filtering, the encoder should use shallow CLIP attention to guide bitrate allocation, prioritizing important regions for MLLMs.
2.  **Decoder Side:** Since standard compression is effective at preserving robust low-level structures, the decoder should use the decompressed image as a reconstruction prior to retain low-level details and avoid domain shift. It should then inject semantic enhancements via an adapter and optimize with a multi-level loss to ensure fidelity across the feature hierarchy.

### 4. Core Innovation Points

1.  **Systematic Analysis of Compression Impact on MLLMs:** The paper provides the first systematic analysis revealing that compression distortion unevenly impacts different-level features. It identifies a three-stage pattern in the vision encoder (Preliminary Screening, Local Information Extraction, Global Semantic Integration) and discovers that cross-level features (early Stage 3) are highly vulnerable to subtle low-level distortions, while low-level and coarse high-level features are robust.
2.  **Shallow CLIP-Guided Encoder:** A novel bitrate allocation mechanism that leverages the shallow layers ([CLS] attention scores of the first 3 layers) of a frozen CLIP model to generate a discrete, three-level importance mask. This allocates more bits to semantically critical regions with negligible overhead.
3.  **Multi-Level Fidelity Decoder:** A decoder design that uses the decoded image as a reconstruction prior to preserve low-level fidelity and mitigate domain shift. It integrates a lightweight Latent Feature Adapter (single transformer block) operating on the decoded latent code to inject high-level semantic enhancement without disrupting low-level structures.
4.  **Multi-Level Fidelity Loss Function:** A novel loss function $L_{total} = \lambda_{low} L_{low} + \lambda_{high} L_{high}$ that supervises fidelity at both ends of the feature spectrum. $L_{low}$ constrains shallow layer patch embeddings (low-level details), while $L_{high}$ constrains final-layer token representations (high-level semantics).
5.  **Hierarchical Guidance for High-Resolution and Video:** An extension mechanism that fuses global (downsampled) and local (patch-wise) attention maps to resolve the conflict between coarse global guidance and limited local context for high-resolution images, making the codec compatible with high-resolution and video-based MLLMs.

### 5. Overview of the Overall Technical Solution

The proposed solution is **CoTAM (Codec TAilored to MLLMs)**.
*   **Base Codec:** A variable-bitrate compression backbone using multiple sets of learned quantization vectors controlled by gain vectors.
*   **Encoder:** Processes the input image while being guided by a Shallow CLIP attention mask. The mask is derived by averaging [CLS] attention from the first 3 CLIP layers, quantized into 3 levels ($\mu \pm k\sigma$), and used to modulate quantization parameters.
*   **Decoder:** Receives the bitstream, decodes it to a latent representation, and reconstructs the image.
*   **Adapter & Prior:** The decoder uses the reconstructed image as a "reconstruction prior" (providing patch embeddings). Concurrently, a lightweight Adapter processes the decoded latent code to generate a semantic enhancement feature, which is fused with the patch embeddings via element-wise addition.
*   **Training:** The framework is trained end-to-end using the Multi-level fidelity loss ($L_{low}$ on shallow features, $L_{high}$ on deep features). The first epoch uses only $L_{low}$ for stabilization.
*   **High-Res/Video Extension:** For high-resolution inputs, Hierarchical Guidance fuses global and local attention maps. Video is processed frame-by-frame.

### 6. Detailed Module Design

**6.1. Base Codec Module**
The base codec enables variable bitrates by adapting the multi-quantizer methodology. It equips internal layers with multiple sets of learned quantization vectors for each bitrate to adaptively allocate bits for each spatial location. The semantic importance map selects a specific vector for each region.
*   **Variable Bitrate Mechanism:** For a discrete set of $N$ target bitrates $R = \{r_1, r_2, \dots, r_N\}$, there are $N$ corresponding sets of learnable vectors. For a given target rate $r \in R$, a specific vector $g_{l,r}$ is applied to the feature map $f_l$ at each modulated layer $l$. The operation is defined as:
    $$f'_l = f_l \odot g_{l,r}$$
    where $\odot$ represents element-wise multiplication, and $f'_l$ is the scaled feature map serving as input to the subsequent layer $l+1$.
*   **Training Objective:** Optimized end-to-end using the rate-distortion loss:
    $$L = D + \lambda_i R$$
    where $D$ is the distortion loss, $R$ is the estimated bit rate, and $\lambda_i$ is the trade-off parameter corresponding to the randomly sampled quality index $i$ in each iteration.

**6.2. Shallow CLIP-Guided Encoder Module**
This module creates a discrete, three-level mask to guide rate allocation based on semantic richness.
*   **Attention Aggregation:** Average the [CLS] attention scores from the first three layers of a frozen CLIP model (chosen for high attention distance) to create a small downsampled spatial map (e.g., 8x8). This quantifies the semantic richness of each region.
*   **Quantization:** Convert the continuous map into a discrete, three-level mask via a statistics-based quantization method $\mu \pm k\sigma$. The three integer levels correspond to instructions: decrease bitrate, maintain base bitrate, or increase bitrate.
*   **Modulation:** The final mask directly modulates the quantization parameters of the learned compression backbone on a patch-wise basis. The bitrate overhead is negligible (e.g., 128 bits for 336x336 input).

**6.3. Multi-Level Fidelity Decoder Module**
This module preserves fidelity across the feature hierarchy using a reconstruction prior and a latent adapter.
*   **Reconstruction Prior:** The decoded image is used as a prior. This preserves robust low-level structures and mitigates domain shift, as MLLM vision encoders are pre-trained on natural RGB images.
*   **Latent Feature Adapter:** A lightweight adapter composed of a single transformer block. It operates directly on the decoded latent code from the bitstream. It generates a semantic enhancement feature.
*   **Feature Fusion:** The semantic enhancement feature is fused with the patch embeddings extracted from the decoded image via element-wise addition. This injects high-level guidance directly into the feature domain without disrupting crucial low-level information.

**6.4. Hierarchical Guidance Module (for High-Resolution/Video)**
*   **Problem:** Guidance from a single fixed-size downsampled image is too coarse (lacks local detail), while independent patch-based local guidance lacks global semantic perception.
*   **Solution:** Fuse (via addition) both global and local maps to create a comprehensive guidance signal that is locally precise and globally aware.
*   **Feature Resizing:** Resize the decoded high-resolution features to get a global feature before processing by the adapter, to match the expected input of the high-resolution MLLM (composed of multiple high-res patches and a downsampled global patch).
*   **Video:** Apply the semantic guidance mechanism on a frame-by-frame basis.

### 7. All Mathematical Formulas and Symbol Definitions

**Information Flow Metrics:**
For a self-attention map $A \in \mathbb{R}^{N \times N}$ in a given layer, where $A_{ji}$ denotes the attention from source token $i$ to target token $j$:
*   **Inflow(k):** $\text{Inflow}(k) = \arg\max_j A_{kj}$
*   **Outflow(k):** $\text{Outflow}(k) = \arg\max_i A_{ik}$

**Attention Distance Metrics:**
Measured on 1,000 images from the CC3M dataset:
*   **Average Attention Distance:** $D_{avg} = \frac{1}{N}\sum_{i=1}^N \sum_{j=1}^N A_{ij} \cdot d(p_i, p_j)$
*   **Average Max Attention Distance:** $D_{top1} = \frac{1}{N}\sum_{i=1}^N d(p_i, p_{\arg\max_j A_{ij}})$
Where $A$ is the $N \times N$ attention map, $A_{ij}$ is the attention weight from token $j$ to token $i$. $p_i$ denotes the 2D spatial position of the $i$-th token. $d(p_i, p_j)$ represents the Euclidean distance between positions of tokens $i$ and $j$.

**Multi-Level Fidelity Loss:**
$$L_{total} = \lambda_{low} L_{low} + \lambda_{high} L_{high}$$
*   $L_{low}$: Low-level fidelity loss. Minimizes the Mean Squared Error (MSE) between the patch embedding features of the original and decoded images (constrains shallow layers).
*   $L_{high}$: High-level perceptual loss. Minimizes the MSE between the final-layer token representations of the original and the processed output (constrains deep layers).
*   Parameters: $\lambda_{low} = 0.1$, $\lambda_{high} = 1$.

**Base Codec Scaling Formula:**
$$f'_l = f_l \odot g_{l,r}$$
*   $f_l$: Feature map at the output of the $l$-th encoder layer.
*   $g_{l,r}$: Specific gain vector for layer $l$ and target rate $r$.
*   $\odot$: Element-wise multiplication.
*   $f'_l$: Scaled feature map.

**Rate-Distortion Loss (Base Codec):**
$$L = D + \lambda_i R$$
*   $D$: Distortion loss.
*   $R$: Estimated bit rate.
*   $\lambda_i$: Trade-off parameter for quality level $i$. Empirical set: $\lambda_i \in \{0.00002, 0.00005, 0.0001, 0.0002, 0.0004, 0.0008, 0.0016, 0.0032, 0.0064, 0.0128\}$ for $N=10$ levels.

**Quantization Parameter:**
*   $k = 0.75$ for the statistics-based quantization method $\mu \pm k\sigma$.

### 8. Algorithm Pseudocode

The paper does not provide explicit algorithm pseudocode blocks. The logical process derived from the text is as follows:

**Training Stage:**
1.  Initialize Base Codec with variable bitrate vectors.
2.  For Epoch 1: Train using only $L_{low}$ (MSE on patch embeddings) to stabilize optimization.
3.  For Epochs 2-5:
    a. Input Image $X$.
    b. **Shallow CLIP Guidance:** Extract [CLS] attention from first 3 layers of frozen CLIP. Average and quantize to 3-level mask $M$.
    c. **Encoding:** Encode $X$ using Base Encoder. Modulate quantization parameters using mask $M$.
    d. **Decoding:** Decode bitstream to latent code $Z$. Decode $Z$ to image $\hat{X}$.
    e. **Adapter & Prior:** Extract patch embeddings from $\hat{X}$. Pass $Z$ through Latent Feature Adapter to get semantic feature $F_{sem}$. Fuse via addition.
    f. **Loss:** Calculate $L_{total} = \lambda_{low} L_{low} + \lambda_{high} L_{high}$.
    g. **Update:** Backpropagate and update Encoder, Decoder, and Adapter weights (CLIP is frozen).

**Inference Stage:**
1.  Input Image $X$.
2.  Generate Shallow CLIP Guidance Mask $M$.
3.  Encode $X$ with mask $M$ guidance.
4.  Transmit Bitstream.
5.  Decode Bitstream to $Z$ and $\hat{X}$.
6.  Apply Latent Feature Adapter to $Z$ and fuse with features from $\hat{X}$.
7.  Output Enhanced Features to MLLM.

**High-Resolution Extension:**
1.  Input High-Res Image $X_{HR}$.
2.  **Global Map:** Downsample $X_{HR}$, compute CLIP attention, upsample map.
3.  **Local Maps:** Divide $X_{HR}$ into patches, compute CLIP attention per patch.
4.  **Fusion:** Add Global and Local maps -> Hierarchical Guidance Mask $M_{HR}$.
5.  Proceed with Encoding/Decoding using $M_{HR}$. Resize decoded features for adapter input.