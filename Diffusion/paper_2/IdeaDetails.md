# 1. Research Background and Existing Pain Points

**Research Background:**
With the rapid advancement of 4K/8K display and imaging technologies, the demand for Ultra-High-Definition (UHD) images is increasing dramatically. However, in real-world capture, UHD images inevitably suffer from degradations such as low light, haze, blur, and noise, which are often caused by insufficient illumination, adverse weather conditions, or equipment limitations. As a result, UHD image restoration (UHD-IR) has become a crucial yet highly challenging research field in computer vision, characterized by its massive resolution scale and requirements for preserving fine-grained details. Recently, Latent Diffusion Models (LDMs) have shown remarkable potential in low-level vision tasks owing to their powerful generative priors.

**Existing Pain Points:**
1.  **Global Inconsistencies and Detail Loss in LDMs:** Directly applying pre-trained LDMs to UHD-IR often results in severe global inconsistencies and loss of fine-grained details.
2.  **Patch-based Inference Artifacts:** Due to the high computational cost and memory demand of self-attention mechanisms, models cannot process an entire UHD image in a single pass, making patch-based inference unavoidable. This strategy often introduces stripe-like artifacts and color inconsistencies, while naive upsampling schemes typically cause blurring or structural distortions.
3.  **Stochasticity in Textureless Regions:** The absence of global context in patch-based inference amplifies the stochasticity of diffusion models, leading to inconsistent high-frequency details in textureless regions.
4.  **VAE Information Bottleneck:** The Variational Autoencoder (VAE), as a core component of LDMs, suffers from lossy compression that discards high-frequency information and thereby limits restoration fidelity.
5.  **Limitations of Existing Restoration Methods:** Existing UHD-IR studies primarily enhance performance by designing innovative network architectures and developing advanced training paradigms. However, they unavoidably encounter bottlenecks, as merely modifying network structures is insufficient to overcome the inherently ill-posed nature of image restoration.
6.  **Limitations of Generation-Oriented Adaptation:** Existing high-resolution adaptation of diffusion models (e.g., MultiDiffusion, DemoFusion) are designed for generation where outputs only need to be perceptually plausible. In contrast, UHD restoration requires strict structural fidelity to the degraded input, and any hallucinated or altered content violates the restoration objective. These generation-oriented strategies often struggle to maintain input–output consistency in UHD-IR.

# 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:**
How to leverage powerful diffusion priors to overcome the bottlenecks in UHD-IR remains insufficiently explored. The objective is to resolve the prevalent issues of artifacts, global inconsistencies, and detail loss that arise when directly applying pre-trained LDMs to UHD-IR, without modifying or fine-tuning the denoising U-Net. Meeting restoration requirements relies on two principles: selective injection of reliable global information and preservation of structural alignment throughout denoising.

**Scientific Questions:**
1.  How to enforce global structural consistency across patches during patch-based denoising to eliminate stripe-like artifacts and color inconsistencies?
2.  How to constrain high-frequency detail generation to suppress high-frequency hallucinations and unrealistic textures in smooth regions while preserving local detail fidelity?
3.  How to mitigate the VAE's inherent information bottleneck that discards high-frequency information, thereby improving the reconstruction fidelity of fine-grained textures?

# 3. Overall Core Idea and Design Philosophy

**Overall Core Idea:**
Introduce FreeAdapt, a unified framework that combines a plug-and-play, training-free guidance mechanism with an optional VAE fine-tuning (VAE-FT) module. FreeAdapt provides a cost-efficient way to unleash the potential of diffusion priors and adapt pre-trained LDMs and their extensions (e.g., ControlNet) to UHD-IR. The core is the Frequency Feature Synergistic Guidance (FFSG) mechanism, which enforces both global consistency and local detail fidelity at each step of patch-based denoising during inference time.

**Design Philosophy:**
1.  **Selective Injection and Alignment:** Meeting UHD restoration requirements relies on selective injection of reliable global information and preservation of structural alignment throughout denoising.
2.  **Frequency-Domain Constraints for Structure:** Use frequency-domain constraints to stabilize low-frequency structures and textures. Frequency Guidance (FreqG) selectively fuses phase information from a reference image in the frequency domain to ensure global structural consistency across patches without interfering with texture generation.
3.  **Feature-Level Attention for Artifacts:** Use a feature-level attention module to suppress high-frequency artifacts and promote cross-patch consistency. Feature Guidance (FeatG) incorporates global context into the U-Net self-attention layers to constrain local detail generation and suppress high-frequency hallucinations.
4.  **Task-Level Prior for VAE:** The VAE-FT module learns a task-level prior for detail reconstruction without learning the restoration task itself. Because this prior is task specific rather than model specific, the fine-tuned decoder can be applied across different diffusion backbones without further training.

# 4. Core Innovation Points

1.  To the best of our knowledge, FreeAdapt is the first plug-and-play diffusion prior framework for the UHD-IR task, providing an effective and generalizable solution for pre-trained LDMs and their extensions.
2.  We propose a training-free, plug-and-play synergistic guidance mechanism that, through innovative frequency and feature guidance modules, effectively resolves the artifact and detail hallucination issues in UHD-IR, significantly improving both global consistency and local fidelity.
3.  By introducing skip connection and fine-tuning the VAE decoder, we successfully mitigate the VAE’s information bottleneck and improve the fidelity of reconstructed details.
4.  Through extensive experiments across three representative LDM-based backbones (LDM, StableSR, and DiffBIR), we show that FreeAdapt consistently delivers strong performance improvements, achieving PSNR gains typically above 2 dB over patch-based inference. These gains stem from its ability to correct cross-patch inconsistencies and more effectively exploit pre-trained diffusion priors.

# 5. Overview of the Overall Technical Solution

The core of FreeAdapt is a training-free guidance mechanism that operates during the iterative, patch-based denoising process. This mechanism integrates FreqG and FeatG modules to enforce global consistency and preserve local detail fidelity. Additionally, to overcome the inherent high-frequency information loss of the VAE, we introduce an optional VAE-FT module that fine-tunes the VAE decoder with lightweight modifications to further boost the fidelity of the restored images.

The overall mechanism consists of two main stages: Reference Image Generation and Guided High-Resolution Iterative Denoising.

1.  **Reference Image Generation:** The degraded UHD input is downsampled to the native training resolution (e.g., 512×512), encoded by the VAE to obtain the conditioning latent, and then used to guide a single standard denoising process of the pre-trained LDM, producing a clean latent representation with coherent content and structure. This latent is decoded into the pixel domain, upsampled back to the UHD resolution to form the reference image, and re-encoded by the VAE encoder to obtain the reference latent for global guidance.
2.  **Guided High-Resolution Iterative Denoising:** To address GPU memory limitations in UHD-IR, a patch-based denoising strategy is adopted. At each denoising step, multiple small patches are cropped from the current high-resolution latent representation and denoised individually. Overlapping regions are blended by smooth averaging to maintain consistency across patch boundaries. Within each denoising step, FreqG and FeatG are integrated into the denoising process, jointly improving both global structural coherence and local detail fidelity.

# 6. Detailed Module Design

**Frequency Guidance (FreqG) Module:**
To overcome the structural inconsistencies introduced by patch-based denoising, FreqG is incorporated into the iterative denoising process. At each step, both the current latent representation and the noised reference latent are transformed into the frequency domain using Fast Fourier Transformation (FFT), yielding their respective amplitude and phase spectrum. To ensure that the global structure of the reference is effectively preserved without interfering with texture generation, only the phase components are fused. Specifically, a dynamically low-pass filter is applied to weight the two phase spectrums. The filter gradually decreases as the denoising step progresses, adaptively balancing global structural constraints with flexibility for detail generation. The corrected latent is then reconstructed by combining the original amplitude spectrum with the fused phase spectrum through inverse FFT.

**Feature Guidance (FeatG) Module:**
While FreqG enforces global low-frequency consistency, it cannot constrain high-frequency details generated independently within each patch. In textureless regions this randomness often introduces spurious details, leading to visual noise or artifacts. To address this issue, FeatG module injects global contextual information into the self-attention layers of the U-Net. This allows each patch to reference the global semantics provided by the guidance image, thereby promoting inter-patch coherence and suppressing unrealistic artifacts. Specifically, we first compute the Query, Key, and Value of the current high-resolution patch to obtain local attention. In parallel, we extract the patch-aligned query from the reference, together with global keys and values, and compute the global attention. The final output is obtained by linearly blending the two attentions. Importantly, this operation is applied to the 3rd–8th decoder layers of the U-Net.

**VAE Fine-Tuning (VAE-FT) Module:**
In LDMs, VAE is responsible for perceptual compression, but the lossy compression characteristic causes the loss of high-frequency details such as fine textures and text at high resolutions. To address this issue, we introduce an optional VAE-FT module that strengthens the decoder’s ability to recover fine details while keeping both the encoder and the U-Net frozen, ensuring that the diffusion process remains fully training-free. During fine-tuning, both a low-quality image and its high-quality counterpart are passed through the shared VAE encoder, producing a high-quality latent representation along with residual features extracted from the degraded input. The decoder receives the high-quality latent together with these residual features through skip connections, which provide structural cues that help restore information lost during encoding. Through this training strategy, VAE-FT learns a task-level prior for detail reconstruction without learning the restoration task itself. We enhance the VAE decoder by introducing skip connection combined with parameter-efficient LoRA. Encoder features from the degraded input are first refined through adaptive instance normalization (AdaIN), which suppresses degradation while retaining structural details, and are then injected into the corresponding upsampling layers of the decoder via Zero-Convolution modules.

# 7. All Mathematical Formulas and Symbol Definitions

**Formula 1:** Training objective of the LDM
$$L_{ldm} = E_{z,c,t,\epsilon}[||\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}z + \sqrt{1-\bar{\alpha}_t}\epsilon, c, t)||_2^2]$$
*Symbol Definitions:*
*   $\epsilon \sim \mathcal{N}(0, I)$: ground-truth noise at timestep $t$
*   $c$: conditional information
*   $\bar{\alpha}_t$: diffusion coefficient defined in DDPM

**Formula 2:** FFT of current latent
$$FFT(z_t) = A_t e^{i\phi_t}$$
*Symbol Definitions:*
*   $z_t$: current latent representation at step $t$
*   $A_t$: amplitude spectrum
*   $\phi_t$: phase spectrum

**Formula 3:** FFT of noised reference latent
$$FFT(z_{ref_t}) = A_{ref_t} e^{i\phi_{ref_t}}$$
*Symbol Definitions:*
*   $z_{ref_t}$: noised reference latent at step $t$
*   $A_{ref_t}$: amplitude spectrum of reference
*   $\phi_{ref_t}$: phase spectrum of reference

**Formula 4:** Fused phase spectrum
$$\phi_t = \arctan((1-K(t))e^{i\phi_t} + K(t)e^{i\phi_{ref_t}})$$
*Symbol Definitions:*
*   $K(t)$: dynamically low-pass filter

**Formula 5:** Dynamic low-pass filter definition
$$K(t) = \begin{cases} t/T, & \text{if } |x - w/2| < w \cdot c \cdot t/T \text{ and } |y - h/2| < h \cdot c \cdot t/T \\ 0, & \text{otherwise} \end{cases}$$
*Symbol Definitions:*
*   $c$: hyperparameter (default 0.15)
*   $w, h$: width and height
*   $T$: total denoising steps

**Formula 6:** Corrected latent reconstruction
$$z'_t = iFFT(A_t e^{i\phi_t})$$
*Symbol Definitions:*
*   $z'_t$: corrected latent
*   $A_t$: original amplitude spectrum
*   $\phi_t$: fused phase spectrum

**Formula 7:** Local attention
$$Attn_{local} = \text{softmax}\left(\frac{Q_{tile} \cdot K_{tile}^T}{\sqrt{d}}\right) V_{tile}$$
*Symbol Definitions:*
*   $Q_{tile}, K_{tile}, V_{tile}$: Query, Key, Value of the current high-resolution patch
*   $d$: feature dimension

**Formula 8:** Global attention
$$Attn_{global} = \text{softmax}\left(\frac{U(Q_{ref_{tile}}) \cdot K_{ref_{global}}^T}{\sqrt{d}}\right) V_{ref_{global}}$$
*Symbol Definitions:*
*   $Q_{ref_{tile}}$: patch-aligned query from the reference
*   $K_{ref_{global}}, V_{ref_{global}}$: global keys and values from the reference
*   $U$: upsampling operation

**Formula 9:** Final blended attention
$$Attn_{final} = (1 - \alpha) \cdot Attn_{local} + \alpha \cdot Attn_{global}$$
*Symbol Definitions:*
*   $\alpha$: blending weight (set to 0.2 by default)

**Formula 10:** VAE-FT composite loss
$$L = L_{dwt} + L_{lpips} + L_{ssim} + L_{gan}$$
*Symbol Definitions:*
*   $L_{dwt}$: L2 loss in the Discrete Wavelet Transform domain for reconstructing high-frequency details
*   $L_{lpips}$: ensures perceptual similarity
*   $L_{ssim}$: maintains structural consistency
*   $L_{gan}$: introduces adversarial feedback to improve realism and sharpness

# 8. Algorithm Pseudocode

**Algorithm 1: FreeAdapt Inference Pipeline**

Input: Degraded UHD image $I_{lq}$, pre-trained LDM ($\epsilon_\theta$, VAE), total denoising steps $T$
Output: Restored UHD image $I_{rec}$

// Stage 1: Reference Generation
$I_{lr} \leftarrow \text{Downsample}(I_{lq})$ ; // Downsample to native resolution
$c_{lr} \leftarrow \text{VAEencode}(I_{lr})$ ; // Encode the low-res image as condition
$z^T_{lr} \sim \mathcal{N}(0, I)$ ; // Initialize Gaussian noise in latent space
for $t = T$ to $1$ do
    $z^{t-1}_{lr} \leftarrow \epsilon_\theta(z^t_{lr}, t, c_{lr})$ ; // Iterative denoising with LDM condition
$I_{ref} \leftarrow \text{Upsample}(\text{VAEdecode}(z^0_{lr}))$ ; // Decode and upsample in pixel domain
$z^0_{ref} \leftarrow \text{VAEencode}(I_{ref})$ ; // Final clean reference latent

// Stage 2: Guided High-Resolution Denoising
$c_{lq} \leftarrow \text{VAEencode}(I_{lq})$ ; // Encode UHD image as the main condition
$z_T \sim \mathcal{N}(0, I)$ ; // Initialize high-resolution Gaussian noise
for $t = T$ to $1$ do
    $z_{ref_t} \leftarrow \text{add noise}(z^0_{ref}, z_T, t)$ ; // Add corresponding noise to reference
    $\mathcal{P}_t \leftarrow \text{CropPatches}(z_t)$ ; // Crop current latent into patches
    $\mathcal{C}_{lq} \leftarrow \text{CropPatches}(c_{lq})$ ; // Crop condition into corresponding patches
    $\mathcal{P}_{t-1} \leftarrow \emptyset$ ; // Initialize set for denoised patches
    foreach patch $(p_t, c_p) \in (\mathcal{P}_t, \mathcal{C}_{lq})$ do
        $p^{ref}_t \leftarrow \text{GetCorrespondingPatch}(z_{ref_t})$ ; // Get reference patch
        $p'_t \leftarrow G_{freq}(p_t, p^{ref}_t)$ ; // Apply Frequency Guidance
        $p_{t-1} \leftarrow \epsilon_\theta(p'_t, t, c_p)$ ; // Denoise patch with condition and $G_{feat}$ inside U-Net
        $\mathcal{P}_{t-1} \leftarrow \mathcal{P}_{t-1} \cup \{p_{t-1}\}$
    $z_{t-1} \leftarrow \text{StitchPatches}(\mathcal{P}_{t-1})$ ; // Stitch patches back with blending

// Stage 3: Reconstruction
$I_{rec} \leftarrow \text{VAEdecode/VAE-FT}(z_0)$ ; // Decode with standard or fine-tuned VAE
return $I_{rec}$