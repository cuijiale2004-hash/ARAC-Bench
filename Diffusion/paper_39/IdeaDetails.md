## 1. Research Background and Existing Pain Points

**Research Background:**
Diffusion models have recently become the dominant paradigm for image generation due to their stable training dynamics and strong generative performance. Building on these advances, large-scale pretrained variants and their control frameworks have recently pushed the frontier of personalized and controllable image synthesis. 

**Existing Pain Points:**
1. **Inability to Interpret and Follow Numeric Instructions:** Current systems remain limited in their ability to understand and follow numeric instructions for adjusting semantic attributes, especially in scenarios that demand precise control over aesthetic attributes. Current models fail to interpret comparative or gradable instructions (e.g., "make it exactly 20% dimmer" or "slightly sharper textures"), leading to outputs that are misaligned with user intent.
2. **Mismatch Between Discrete Encoders and Continuous Values:** Current text encoders are designed for discrete tokens rather than continuous values, which makes it inherently difficult to capture and control aesthetic intent.
3. **Global Preference Alignment Overlooks Multifaceted Aesthetics:** Efforts on aesthetic alignment leveraging reinforcement learning, direct preference optimization, or architectural modifications primarily align models with a global notion of human preference. These approaches implicitly assume a single optimal target, overlooking the multifaceted and context-dependent nature of aesthetic judgment, and lack mechanisms for disentangling and precisely controlling individual attributes.
4. **Artifacts in Latent-Space Interpolation:** Alternative strategies, such as latent-space interpolation, blend features between two discrete endpoints, but lack explicit guidance on the attribute’s semantic manifold, producing artifacts or collapsing structures.

## 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:**
In real-world creative scenarios, especially when precise control over aesthetic attributes is required, current methods fail to provide such controllability. Professional photographers, children's book illustrators, and content creators often need subtle refinements and precise numeric control over attributes like dimness, cartoon-like abstraction, texture sharpness, or realism. Instead of relying on undifferentiated preference signals, there is a critical need to explicitly decompose aesthetic attributes, quantify them along continuous dimensions, and enable users to modulate their intensity through numeric instructions.

**Scientific Questions:**
How can generative models disentangle aesthetic attributes, understand them as continuous values, and smoothly navigate their intensity in a user-controllable manner?

## 3. Overall Core Idea and Design Philosophy

The core idea is to introduce AttriCtrl, a lightweight framework for continuous aesthetic intensity control in diffusion models that explicitly decomposes, quantifies, and independently controls aesthetic attributes. The design philosophy centers on:
1. **Hybrid Quantification Strategy:** Defining relevant aesthetic attributes and quantifying them through a hybrid strategy that maps both concrete and abstract dimensions onto a unified [0, 1] scale.
2. **Plug-and-Play Value Encoding:** Using a lightweight, plug-and-play value encoder to transform user-specified continuous scalar values into model-interpretable embeddings for controllable generation.
3. **Lightweight Adaptation with Frozen Base Model:** Implementing the framework as a lightweight adapter while keeping the diffusion model frozen, ensuring seamless integration with existing frameworks (such as ControlNet) at negligible computational cost.
4. **Modular Compositional Control:** Learning a continuous and navigable trajectory for each attribute within the model’s conditioning space to achieve disentangled, attribute-specific control vectors, allowing for independent training and plug-and-play composition at inference.

## 4. Core Innovation Points

1. **Hybrid Aesthetic Attribute Quantification Strategy:** A novel strategy that quantifies both concrete attributes (brightness, detail) using direct metric-based estimation and abstract attributes (realism, safety) using vision-language semantic similarity, mapping them onto a unified [0, 1] continuous scale.
2. **Lightweight Plug-and-Play Value Encoder:** A tailored value encoder that maps normalized scalar intensity values into multi-scale learnable token sequences via sinusoidal embedding, MLP transformation, sequence expansion, and positional embedding, allowing self-attention layers to process scalar intensity in a distributed and relational manner.
3. **Modular Multi-Attribute Composition Mechanism:** A modular strategy where each aesthetic attribute is encoded independently using its corresponding value encoder trained on single-attribute data, and the resulting embeddings are concatenated in sequence and appended to the text embedding at inference time, ensuring flexible, plug-and-play integration while minimizing mutual interference and data imbalance.
4. **Disentangled Continuous Aesthetic Control with Frozen Base Model:** Training only the value encoder on curated subsets while keeping the base diffusion model completely frozen. This ensures minimal computational overhead, preserves the base model's generative capabilities, and achieves smooth transitions and precise modulation of specific attribute intensities independently of other factors.

## 5. Overview of the Overall Technical Solution

The overall technical solution decomposes the problem of precise aesthetic attribute control into two main components:
First, the aesthetic attribute quantification component defines four semantic attributes closely related to human perceptual preferences: brightness, detail, realism, and safety. For concrete attributes (brightness, detail), direct metric-based estimation is applied. For abstract attributes (realism, safety), pretrained vision-language models are leveraged to compute cross-modal similarity between images and descriptive text prompts. Raw values are then balanced across 10 equal-width bins via oversampling and downsampling, and normalized onto a shared [0, 1] scale via rank-based normalization.

Second, the tailored aesthetic control component introduces a lightweight value encoder that transforms the normalized intensity value into a learnable token sequence. This involves sinusoidal embedding, a two-layer MLP with SiLU activations, duplication into a fixed-length sequence, and the addition of a learnable positional embedding. The final embedding is concatenated along the sequence dimension of the text embedding, forming a joint representation that enters the backbone of the diffusion model. The value encoder is trained using a standard denoising objective while the base model remains frozen. For multi-attribute control, independently trained value encoders are concatenated at inference.

## 6. Detailed Module Design

### 6.1 Aesthetic Attribute Quantification Module
This module defines and quantifies four semantic attributes:

**6.1.1 Direct Estimation Sub-module**
*   **Brightness:** Estimated in the HSV color space. The Value channel is extracted, and the mean pixel intensity is computed and normalized by 255.
*   **Detail:** Adopted Shannon entropy as the quantification metric. The image is converted to grayscale. A histogram over 256 grayscale levels is constructed and normalized into a probability distribution. The raw intensity value of detail is defined as the entropy of this distribution.

**6.1.2 Similarity-Based Estimation Sub-module**
*   **Realism:** Leverages CLIP to compute cosine similarity between an image embedding and a set of positive and negative prompts. The positive prompt is "a real photograph, realistic details and natural lighting" ($c_{pos}$) and the negative prompt is "a cartoon image, a human-created artistic representation, such as an illustration or painting" ($c_{neg}$). 
*   **Safety:** Extracts the textual embedding $e_s$ for unsafe concepts from the internal safety checker of Stable Diffusion and computes the cosine similarity with the image embedding. A threshold $t$ (set to 0.19) defines the maximum allowable similarity. The value is inverted by taking its negative so that scores greater than zero correspond to safe images.

**6.1.3 Value Mapping Sub-module**
To ensure attribute values are uniformly distributed for training and comparable across different attributes:
*   **Balanced Sampling:** The empirical value range is divided into 10 equal-width bins based on dataset statistics. Underrepresented bins are oversampled with replacement, while overrepresented bins are randomly downsampled.
*   **Rank-based Normalization:** After balancing, the raw attribute values $x_i$ are normalized onto a shared [0, 1] scale. Given a raw value $x_i$ from a collection of $n$ training samples $\{x_1, \ldots, x_n\}$, its normalized counterpart is computed.

### 6.2 Tailored Aesthetic Control Module
This module integrates the quantified aesthetic attributes into the diffusion process.

**6.2.1 Value Encoder Sub-module**
An independent value encoder is designed to transform the normalized intensity value $x^{\text{norm}}_i$ into a learnable token sequence $v$ for each aesthetic attribute. The layer-by-layer working mechanism is:
1.  **Sinusoidal Embedding Mechanism:** Converts the continuous aesthetic value into structured high-dimensional vectors, leveraging its smooth interpolation properties.
2.  **Two-layer Multilayer Perceptron (MLP):** The resulting embedding is passed through a two-layer MLP with SiLU activations to become a hidden representation.
3.  **Sequence Expansion:** The hidden representation is duplicated and expanded into a fixed-length sequence of 32 tokens. This allows the model’s self-attention layers to process the scalar intensity value in a distributed and relational manner.
4.  **Positional Embedding:** A learnable positional embedding is added to the repeated representations to obtain the final embedding $v$. This enables the model to assign distinct functional roles to each token in the sequence, creating a more expressive representation.
5.  **Integration:** The final embedding $v$ is concatenated along the sequence dimension of the text embedding $c$, forming a joint representation that enters the backbone of the diffusion model.

**6.2.2 Multi-Attribute Composition Sub-module**
To address data imbalance across attributes in joint training:
1.  **Independent Training:** Each aesthetic attribute is first encoded independently using its corresponding value encoder trained on single-attribute data.
2.  **Inference Concatenation:** At inference time, the resulting embeddings from multiple value encoders are concatenated in sequence and appended to the text embedding. This composite embedding enables joint conditioning on multiple aesthetic dimensions within a unified framework.

## 7. All Mathematical Formulas and Symbol Definitions

**Brightness Intensity Value:**
$$x^{\text{Brightness}}_I = \frac{1}{H \cdot W} \sum_{i=1}^{h} \sum_{j=1}^{w} \frac{v_{i,j}}{255}$$
*   $I$: Input image.
*   $v_{i,j}$: Value channel of the pixel $(i, j)$ in the HSV representation.
*   $H, W$: Height and width of the image, respectively.

**Detail Intensity Value:**
$$x^{\text{Detail}}_I = \text{Entropy}(\text{Hist}(I)) = -\sum_{k=1}^{256} p_k \log(p_k)$$
*   $p_k$: Probability of the $k$-th grayscale intensity level in image $I$.

**Cosine Similarity:**
$$\text{sim}(e_I, e_T) = \frac{e_I \cdot e_T}{\|e_I\| \cdot \|e_T\|}$$
*   $e_I$: Image embedding.
*   $e_T$: Text embedding.

**Realism Intensity Value:**
$$x^{\text{Realism}}_I = \text{sim}(e_I, e_{\text{pos}}) - \text{sim}(e_I, e_{\text{neg}})$$
*   $e_{\text{pos}}$: Text embedding of the positive prompt ("a real photograph, realistic details and natural lighting").
*   $e_{\text{neg}}$: Text embedding of the negative prompt ("a cartoon image, a human-created artistic representation, such as an illustration or painting").

**Safety Intensity Value:**
$$x^{\text{Safety}}_I = -(\text{sim}(e_I, e_s) - t)$$
*   $e_s$: Textual embedding for unsafe concepts extracted from Stable Diffusion's safety checker.
*   $t$: Threshold defining the maximum allowable similarity (set to 0.19).

**Rank-based Normalization:**
$$x^{\text{norm}}_i = \frac{\text{rank}(x_i) - 0.5}{n} \in [0, 1]$$
*   $x_i$: Raw value from a collection of $n$ training samples $\{x_1, \ldots, x_n\}$.
*   $\text{rank}(x_i)$: Average rank of $x_i$ among the $n$ samples.

**Training Objective:**
$$L(\theta) = \mathbb{E}_{z_t, \varepsilon, c, t} \left[ \|\varepsilon - \hat{\varepsilon}_\theta(z_t, c, v, t)\|_2^2 \right]$$
*   $\theta$: Parameters of the value encoder and denoising network (base model frozen, so effectively only value encoder is updated).
*   $z_t$: Noisy latent at timestep $t$.
*   $\varepsilon$: Added Gaussian noise.
*   $c$: Contextual embedding from text encoder.
*   $v$: Learnable token sequence from the value encoder.
*   $t$: Timestep.
*   $\hat{\varepsilon}_\theta$: The denoising network predicting the added noise.

**Control Accuracy Metric (AvgDiff):**
$$\text{AvgDiff} = \frac{1}{N} \sum_{i=1}^{N} |v^{(i)}_{\text{target}} - v^{(i)}_{\text{result}}|$$
*   $N$: Total number of generated images.
*   $v^{(i)}_{\text{target}}$: Target attribute intensity value for the $i$-th image.
*   $v^{(i)}_{\text{result}}$: Normalized attribute intensity value extracted from the $i$-th generated image.

**Removal Rate Metric (RR):**
$$\text{RR} = \frac{n_o - n_s}{n_o}$$
*   $n_o$: Number of unsafe images generated by the base model.
*   $n_s$: Number of unsafe outputs after applying safety control.

## 8. Algorithm Pseudocode

No algorithm pseudocode is explicitly provided in the original paper.