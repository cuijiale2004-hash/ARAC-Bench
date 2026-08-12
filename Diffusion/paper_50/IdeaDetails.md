### 1. Research Background and Existing Pain Points

Real-world video super-resolution (Real-VSR) targets at recovering high-resolution (HR) videos $x_{HR}$ from their low-resolution (LR) counterparts $x_{LR}$ degraded by unknown factors in real-world cases. Traditional non-generative and generative adversarial network (GAN)-based approaches often struggle under complex, mixed degradations, producing over-smoothed results or artifacts. While many diffusion models have achieved impressive results in Real-VSR by generating rich and realistic details, their reliance on multi-step sampling leads to slow inference. One-step networks like SeedVR2, DOVE, and DLoRAL alleviate this through condensing generation into one single step, yet they remain heavy, with billions of parameters and multi-second latency.

Recent adversarial diffusion compression (ADC) offers a promising path via pruning and distilling these models into a compact AdcSR network, but directly applying it to Real-VSR fails to balance spatial details and temporal consistency. This failure stems from two main pain points:
1.  **Lack of Temporal Awareness:** AdcSR is designed for Real-ISR and does not account for temporal modeling. When applied frame by frame to videos, it inevitably introduces flickers.
2.  **Limitations of Standard Adversarial Learning:** Existing distillation methods couple the objectives of details and consistency into a single adversarial signal. In practice, the discriminator often tends to prioritize one aspect (typically details) at the expense of the other (typically consistency), leading to detail-rich but flickering results or over-smoothed but consistent results. This reveals a fundamental conflict: detail enrichment requires significant pixel-level variations, while temporal consistency demands constraining these variations across frames.

### 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:**
Although 3D spatio-temporal DiTs achieve impressive results, their attention mechanisms aim to capture long-range space-time dependencies, which are important for text-to-video generation where global information must be inferred from scratch. In Real-VSR, however, the LR video already provides much of this information (e.g., structural layout and temporal continuity). Its main objectives are: (1) synthesizing details and (2) ensuring that they are temporally consistent to prevent flickering. In this setting, heavy 3D attentions might introduce redundancy. Motivated by this, the paper seeks to compress large Real-VSR diffusion networks while balancing quality and efficiency, and to resolve the conflict between optimizing detail richness and temporal consistency that plagues existing compression and distillation methods.

**Scientific Questions:**
1.  Is a 3D spatio-temporal attention mechanism strictly necessary for Real-VSR, or can a more efficient structure suffice given that the LR input provides global structure and temporal continuity?
2.  How can adversarial distillation be structured to disentangle and jointly optimize two conflicting objectives—spatial detail richness and temporal consistency—preventing the generator from collapsing toward one at the expense of the other?

### 3. Overall Core Idea and Design Philosophy

The overall core idea is to propose an improved Adversarial Diffusion Compression (ADC) method for Real-VSR that combines an effective network design with an adversarial distillation scheme.

**Design Philosophy:**
1.  **Architecture ("2D + 1D"):** The paper hypothesizes that a 2D diffusion backbone is capable of synthesizing details, and consistency can be maintained with several 1D temporal convolutions. The rationale is that maintaining consistency is inherently less challenging than synthesizing details; the objective is to constrain pixel-level variations across consecutive frames, rather than generate new fine structures. This motivates a principled "2D + 1D" network design that is expressive enough to learn the powerful Real-VSR mappings of large 3D DiTs while reducing redundant overhead.
2.  **Distillation (Dual-Head, Dual-Domain):** To address the conflict between details and consistency, the paper introduces a dual-head and dual-discriminator scheme that disentangles the assessments of details and consistency. This design ensures that neither aspect can be disregarded or down-weighted, preventing the AdcVSR generator from collapsing toward one objective.

### 4. Core Innovation Points

1.  **Improved ADC Approach:** A novel improved ADC approach is proposed that combines an effective network design with adversarial distillation to compress a heavy Real-VSR model into an efficient diffusion-GAN hybrid.
2.  **"2D + 1D" Architecture:** It is demonstrated that a 2D image diffusion backbone augmented with lightweight 1D temporal convolutions can effectively learn Real-VSR mapping from a 3D diffusion Transformer (DiT) teacher. This design achieves significantly higher efficiency than heavy 3D attention mechanisms.
3.  **Dual-Head, Dual-Discriminator Adversarial Distillation:** A new adversarial distillation scheme is introduced that decouples the discriminations of details and consistency into two heads sharing a common network backbone, applied in both VAE decoder’s feature space and the pixel space. This design enables balanced optimization, avoiding collapse into either over-smoothed outputs (loss of spatial details) or flickering (loss of temporal consistency).
4.  **Curated Data Strategy for Discriminators:** The training of the dual-head discriminators utilizes five carefully designed data types (student outputs, real videos, temporally shuffled videos, static pseudo-videos from images, and randomly sampled image sequences) with head-specific labels that vary detail and consistency independently, ensuring stable training and balanced optimization.

### 5. Overview of the Overall Technical Solution

The technical solution involves compressing the large pretrained 3D DiT model DOVE (teacher) into the "2D + 1D" AdcVSR network (student) using a two-stage training process.

**Architecture:**
The student network AdcVSR adopts a pruned SD2.1 UNet and VAE decoder as the 2D backbone (similar to AdcSR). To add temporal awareness, residual blocks consisting of 1D temporal convolutions are inserted after each UNet block.

**Distillation Targets:**
Distillation is conducted in two domains: the pixel domain and the feature domain of the AdcSR VAE decoder’s middle block. In the feature domain, DOVE’s output pixels are re-encoded by SD2.1 VAE encoder and fed into the middle block to obtain aligned features for supervision.

**Adversarial Learning:**
Two discriminators are employed: one in the pixel domain (using a frozen ConvNeXt backbone) and one in the feature domain (using a frozen augmented SD UNet). Each discriminator has two linear projection heads ("detail" and "consistency") at the tail to separately evaluate detail richness and temporal consistency.

**Training Process:**
1.  **Stage 1:** Error-minimizing distillation from DOVE without adversarial learning.
2.  **Stage 2:** Fine-tuning with the dual-head, dual-discriminator adversarial distillation scheme, utilizing curated video and image data.

### 6. Detailed Module Design

**1. AdcVSR Network (Student):**
*   **Base:** Channel-pruned SD2.1 UNet and VAE decoder (from AdcSR).
*   **Temporal Augmentation:** Residual blocks are inserted after each UNet block (both spatial residual blocks and transformer blocks).
*   **1D Residual Block Structure:** Each block consists of a 1D temporal convolution, a ReLU activation, and a second convolution with a skip connection.
*   **Parameters:** Each 1D convolution has a kernel size of 3 and the same channel number as its preceding UNet block. The inserted temporal blocks are zero-initialized.

**2. Dual-Head Discriminators:**
*   **Pixel-Domain Discriminator ($D_{pixel}$):**
    *   **Backbone:** Frozen pretrained ConvNeXt from the OpenCLIP library.
    *   **Additional Layers:** Three additional alternating 2D and 1D convolutional layers to jointly capture spatial and temporal features.
    *   **Heads:** Two linear projection heads ("detail" head: 192 output channels; "consistency" head: 64 output channels). Implemented by $1 \times 1$ convolutions.
*   **Feature-Domain Discriminator ($D_{feature}$):**
    *   **Backbone:** Frozen pretrained augmented SD UNet (same architecture as AdcVSR generator).
    *   **Additional Layers:** Three additional alternating 2D and 1D convolutional layers.
    *   **Heads:** Same structure as the pixel-domain discriminator.
*   **Channel Configuration:** The channel numbers of first convolutions are adjusted to match the dimensions of input images and features, while both channel numbers of last-layer features are set to 256.

**3. Curated Data Strategy for Discriminator Training:**
Five curated types of video and image data are used with specific labels $(y_d, y_c)$ where $-1$ is "Fake", $0$ is "Unlabeled", and $1$ is "Real":
1.  **Student Outputs** ($x_{student}$, $f_{student}$): Labeled as "Fake" for both heads $(-1, -1)$.
2.  **Real Videos** ($x_{video}$, $f_{video}$): Labeled as "Unlabeled" for details and "Real" for consistency $(0, 1)$.
3.  **Temporally Shuffled Videos** ($x_{video}^*$, $f_{video}^*$): Randomly permuting frame order. Labeled as "Unlabeled" for details and "Fake" for consistency $(0, -1)$.
4.  **Static Pseudo-Videos** ($x_{image}$, $f_{image}$): Repeating detail-rich real images. Labeled as "Real" for both heads $(1, 1)$.
5.  **Random Image Sequences** ($x_{image}^*$, $f_{image}^*$): Randomly sampled/cropped images without temporal correspondences. Labeled as "Real" for details but "Fake" for consistency $(1, -1)$.

### 7. All Mathematical Formulas and Symbol Definitions

**Symbol Definitions:**
*   $x_{HR}$: High-resolution video.
*   $x_{LR}$: Low-resolution video.
*   $x_{SR}$: Super-resolution output.
*   $x_{teacher}$: Teacher (DOVE) output pixels.
*   $x_{student}$: Student (AdcVSR) output pixels.
*   $f_{teacher}$: Aligned features from the VAE decoder's middle block corresponding to the teacher output.
*   $f_{student}$: Features from the student's VAE decoder's middle block.
*   $D_{pixel}$: Pixel-domain discriminator.
*   $D_{feature}$: Feature-domain discriminator.
*   $[D(s)]_d$: Output of the "detail" head of discriminator $D$ for input $s$.
*   $[D(s)]_c$: Output of the "consistency" head of discriminator $D$ for input $s$.
*   $y_d, y_c \in \{-1, 0, 1\}$: Labels encoding "Fake", "Unlabeled", and "Real" for detail and consistency heads, respectively.
*   $\lambda_{pixel}, \lambda_{feature}, \lambda_{adv}$: Loss weights.

**Adversarial Distillation Loss for Student Generator:**
$$ L = \lambda_{pixel}L_{pixel} + \lambda_{feature}L_{feature} $$
$$ L_{pixel} = \|x_{student} - x_{teacher}\|_1 + DISTS(x_{student}, x_{teacher}) + \lambda_{adv}Softplus(-D_{pixel}(x_{student})) $$
$$ L_{feature} = \|f_{student} - f_{teacher}\|_1 + \lambda_{adv}Softplus(-D_{feature}(f_{student})) $$

**Losses for Dual-Head Discriminators:**
$$ L_{disc} = \sum_{(s,y_d,y_c) \in S} [Softplus(-y_d[D(s)]_d) + Softplus(-y_c[D(s)]_c)] $$
$$ S = \{ (x_{student},-1,-1), (f_{student},-1,-1), (x_{video}, 0, 1), (f_{video}, 0, 1), (x_{video}^*, 0,-1), (f_{video}^*, 0,-1), (x_{image}, 1, 1), (f_{image}, 1, 1), (x_{image}^*, 1,-1), (f_{image}^*, 1,-1) \} $$

### 8. Algorithm Pseudocode

**Algorithm 1: Training of Our Proposed AdcVSR**

**Input:** Pretrained 3D DiT DOVE, AdcSR, and SD2.1 models; Weights $\lambda_{pixel}$, $\lambda_{feature}$, and $\lambda_{adv}$.

**Stage 1: Knowledge Distillation Without Adversarial Learning**
Add 1D temporal convolutions to AdcSR as described in Sec. 3.2 to obtain AdcVSR network;
**for** number of training iterations **do**
    Sample a batch of LR-HR video pairs $(x_{LR}, x_{HR})$;
    Compute the outputs $x_{teacher}$ and $f_{teacher}$ from DOVE (teacher);
    Compute the outputs $x_{student}$ and $f_{student}$ from AdcVSR (student);
    Compute the distillation loss:
    $L_{distil} = \lambda_{pixel} [\|x_{student} - x_{teacher}\|_1 + DISTS(x_{student}, x_{teacher})] + \lambda_{feature}\|f_{student} - f_{teacher}\|_1$;
    Update the model weights of AdcVSR (student) using the Adam optimizer;

**Stage 2: Dual-Head, Dual-Discriminator Adversarial Distillation**
Initialize AdcVSR (student, generator) from the pretrained model in Stage 1;
Initialize pixel- and feature-domain discriminators $D_{pixel}$ and $D_{feature}$ as described in Sec. 4.1;
**for** number of training iterations **do**
    Sample a batch of LR-HR video pairs $(x_{LR}, x_{HR})$;
    Compute the outputs $x_{teacher}$ and $f_{teacher}$ from DOVE (teacher);
    Compute the outputs $x_{student}$ and $f_{student}$ from AdcVSR (student);
    Compute the distillation loss as in Eq. (1);
    Update the model weights of AdcVSR (student) using the Adam optimizer;
    Sample three batches of video and image data: $x_{video}$, $x_{image}$, and $x_{image}^*$;
    Construct a batch of pseudo-videos $x_{video}^*$ by randomly permuting the frames within each individual video of $x_{video}$, ensuring that each real video is shuffled independently;
    Compute features $f_{video}$, $f_{video}^*$, $f_{image}$, and $f_{image}^*$ using $x_{video}$, $x_{video}^*$, $x_{image}$, and $x_{image}^*$;
    Compute losses for discriminators $D_{pixel}$ and $D_{feature}$ as in Eq. (4);
    Update the model weights of discriminators using the Adam optimizer;