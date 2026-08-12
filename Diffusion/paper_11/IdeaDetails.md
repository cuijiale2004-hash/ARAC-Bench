1. Research Background and Existing Pain Points
Research Background:
Diffusion models have emerged as the touchstone of recent advancements in image generation, enabling tasks such as text-to-image generation, image-to-image translation, super-resolution, and inpainting with ease and flexibility. Various optimizations and the proliferation of accessible interfaces have made this technology accessible to users without technical know-how and high-end hardware. Generative AI now creates high-quality, diverse, and photorealistic images that are perceptually indistinguishable from real images. Regulating entities have identified the risks posed by such technology, notably the essential demand regarding the identification and traceability of AI-generated content. Among existing solutions (such as metadata and forensics), digital watermarking stands out as a key technique. Watermarking embeds imperceptible identifiers into images, making them detectable by private decoders. This mature technology has many applications, including copy protection, audience measurement, content identification and monetizing, and broadcast monitoring. It has recently been adapted to the identification of generated content to warn users that images are not real or to filter out AI-generated images from the training sets of future generative AIs to avoid model collapse. The requirement of utmost importance is a provably low false alarm rate, i.e., the probability of flagging a real image as AI-generated.

Existing Pain Points:
Post-hoc Watermarking: Traditional post-hoc watermarking embeds a watermark signal into an original image after generation. The main weakness is that it is not specific to generative AI. Although these methods demonstrate progress in robustness, they operate as external add-ons rather than integral components of the generative process. Furthermore, it is difficult in practice to be sure that the host image is not interfering with the watermark, especially in zero-bit watermarking.
VAE-based In-generation Watermarking (Stable Signature): Stable Signature requires fine-tuning the VAE, which acts like an advanced upscaling that adds high-frequency details. Therefore, this in-generation watermarking technique focuses the watermark power on the high-frequency details. This explains the relatively low robustness of Stable Signature against low-pass filtering processes like JPEG compression.
Seed-based In-generation Watermarking (Tree-Rings, Gaussian Shading, RoBIN): Seed-based methods claim there is no original image in GenAI watermarking, controlling distortion is a meaningless constraint, and PSNR is not appropriate. However, they suffer from several issues: Tree-Rings alters the semantic content of the generated image with the strength of the watermark, is not robust to crops, and its false alarm rate is not under control (the computed p-values are incorrect and not uniformly distributed). Gaussian Shading is a multi-bit watermarking with excellent robustness against valuemetric attacks but is absolutely not robust to geometric attacks (like shift, crop, rotation). RoBIN postpones the watermark embedding to maintain semantics but similarly suffers from a high false positive rate without theoretical guarantee.

2. Core Research Motivation and Scientific Questions
Core Research Motivation:
The goal is to sample images deemed as watermarked by a pre-trained detector. This conditioning of the sampling is made without any reference to an original image and as early as possible to plant the watermark in the semantic of the generated image. The method aims to strike a balance between complete modification of the semantic content (seed-based schemes) and the addition of an invisible signal (VAE-based and post-hoc schemes). It also aims to spread the energy of the watermark all over the spectrum, unlike VAE-based methods that focus on high frequencies.

Scientific Questions:
How to convert any post-hoc watermarking scheme into an in-generation embedding along the diffusion process? How to increase robustness to attacks against which the decoder was not originally robust, without retraining or fine-tuning the diffusion model or the detector? How to preserve the diversity and quality of generation while ensuring a provably low false alarm rate?

3. Overall Core Idea and Design Philosophy
The core idea is to guide the diffusion process using the gradient computed from any off-the-shelf watermark decoder. The gradient computation encompasses different image augmentations, increasing robustness to attacks against which the decoder was not originally robust, without retraining or fine-tuning. This method effectively converts any post-hoc watermarking scheme into an in-generation embedding along the diffusion process. The design philosophy holds that watermarking should not be seen as the addition of a low-amplitude signal over an original image, and semantics should not fluctuate due to the watermark. The method utilizes conditional sampling via gradient-based guidance to incorporate external signals through backpropagated gradients during the denoising process.

4. Core Innovation Points (List all innovations in the original text completely)
1. The method is the first to embed the watermark during the diffusion process itself with the use of guidance.
2. It does not necessitate any retraining of the diffusion model.
3. It inherits from the robustness of the watermark detector, but can also improve it against new targeted attacks without retraining the detector.
4. It strikes a balance between complete modification of the semantic content (seed-based schemes) and the addition of an invisible signal (VAE-based and post-hoc schemes).
5. The method patches a weakness of the decoder by encompassing an unknown attack in the gradient computation, improving robustness against unknown augmentations (e.g., 90-degree rotation, median filtering, VAE purification) without retraining the decoder.

5. Overview of the Overall Technical Solution
The technical solution is based on guiding the diffusion process of a Latent Diffusion Model (LDM) towards generating images that are intrinsically deemed watermarked by an arbitrary watermark detector. It uses conditional sampling where a differentiable detector $\phi$ along with a differentiable loss function $L$ guides the diffusion process. At each iteration, the estimated noise is modified by incorporating information from the gradient of the loss. To ensure robustness against image transformations, the loss is computed for a chosen set of augmentations, and their gradients are aggregated using the PCGrad algorithm. To reduce computational cost, the guidance is turned on only at a step $T_w$, the gradient propagation along the backward diffusion is simplified by an identity transform ($\nabla_{z_0}$ replaces $\nabla_{z_t}$), and the gradient norm is clipped to control the amount of watermark signal added.

6. Detailed Module Design (If any, complete mechanisms of each layer/sub-module)
Latent Diffusion Model (LDM):
The image generator is an LDM defined by a latent space $\mathcal{Z}$, a number of diffusion steps $T$ with an associated scheduling $(\alpha_t)_{t \in [\![1, T]\!]}$, a noise estimate model $\epsilon_\theta : \mathcal{Z} \times [\![1, T]\!] \to \mathcal{Z}$, and the function $\text{VAE} : \mathcal{Z} \to \mathcal{X}$ converting latent vectors to images. The diffusion generates an image $x_0$ from a seed $z_T$ through the abstract update process: $\forall t \in [\![1, T]\!], z_{t-1} = \text{Diffusion} (z_t, \epsilon_\theta, t) , x_0 = \text{VAE}(z_0)$.

Pre-trained Watermark Decoder/Detector:
Uses the extraction function $\phi : \mathcal{X} \to \mathbb{R}^M$ to compute $M$ raw logits. For decoding, the decoded bits are the sign of the logits: $\hat{m} = \text{sign}(\phi(x))$. From a binary message $m \in \{0, 1\}^M$, the antipodal modulation outputs the vector $u_m = -(-1)^m$ component-wise. For detection, the image $x$ is deemed watermarked if the cosine similarity $\cos(\phi(x), u_m)$ is above a threshold. A crucial assumption is that $\phi(X)$ provides a random feature with an isotropic distribution in $\mathbb{R}^M$ when applied on a random non-watermarked image $X$.

Guided-Diffusion for Watermarking:
Resorts to conditional sampling. The differentiable detector $\phi$ and loss function $L$ guide the diffusion process. At each iteration, the estimated noise is modified by incorporating information from the gradient of the loss.

Robustness against Image Transformations:
To ensure a robust watermark, at each diffusion step, the loss is computed for an individual transformation $\mathcal{T}$, defined as $L(z_t, u_m; \mathcal{T}) := 1 - \cos(\phi(\mathcal{T}(x_0(z_t))), u_m)$. The gradients are computed for each new loss and aggregated using the PCGrad algorithm, which resolves conflicting gradient directions for different transformations.

Fast and Controlled Guidance:
Two simplifications are introduced: 1) Turn on the watermarking guidance at a step $T_w$, where $0 < T_w < T$. 2) Simplify the gradient propagation along the backward diffusion by an identity transform, where $\nabla_{z_0}$ replaces $\nabla_{z_t}$. The gradient norm is clipped to control the amount of watermark signal added at each diffusion step.

7. All Mathematical Formulas and Symbol Definitions (If any, replicate exactly as in the original text)
Forward process:
$q(z_T | z_0) = \prod_{t=1}^T q(z_t | z_{t-1}), \text{ with } q(z_t | z_{t-1}) = \mathcal{N}(z_t; \sqrt{1 - \beta_t}z_{t-1}, \beta_t\mathbf{I})$
where $\beta_t$ controls the noise schedule.

Reverse process:
$p_\theta(z_{t-1} | z_t) = \mathcal{N}(z_{t-1}; \mu_\theta(z_t, t), \Sigma_\theta(z_t, t))$

DDIM ODE-like process:
$z_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \left( \frac{z_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta(z_t, t)}{\sqrt{\bar{\alpha}_t}} \right) + \sqrt{1 - \bar{\alpha}_{t-1}} \epsilon_\theta(z_t, t)$
where $\bar{\alpha}_t = \prod_{s=1}^t (1 - \beta_s)$.

Abstract update process:
$\forall t \in [\![1, T]\!], z_{t-1} = \text{Diffusion} (z_t, \epsilon_\theta, t) , x_0 = \text{VAE}(z_0)$

Antipodal modulation:
$u_m = -(-1)^m$ component-wise, in $\mathbb{R}^M$.

Guided noise estimation:
$\hat{\epsilon}(z_t, t) := \epsilon_\theta(z_t, t) - \omega \sqrt{1 - \bar{\alpha}_t} \nabla_{z_t} \log L(z_t)$
where $\omega$ is a scalar denoting the strength of the watermark guidance.

Loss function:
$L(z_t, u_m) := 1 - \frac{u_m^\top \phi(x_0(z_t))}{\sqrt{M}\|\phi(x_0(z_t))\|_2} = 1 - \cos(\phi(x_0(z_t)), u_m)$

P-value for zero-bit watermarking:
$p = \frac{1}{2} \left( 1 \pm I_{\cos^2 \theta} \left( \frac{1}{2}, \frac{M - 1}{2} \right) \right)$
with $\cos(\theta) := 1 - L(z_t, u_m)$ and $I_x(a, b)$ is the regularized incomplete beta function.

Loss for individual transformation:
$L(z_t, u_m; \mathcal{T}) := 1 - \cos(\phi(\mathcal{T}(x_0(z_t))), u_m)$

Aggregated robust gradient:
$\hat{\epsilon}_\mathbb{T}(z_t) := \epsilon_\theta(z_t) - \sqrt{1 - \bar{\alpha}_t} \text{Agg} \left( \{ \nabla_{z_t} \log L(z_t, u_m; \mathcal{T}) \mid \mathcal{T} \in \mathbb{T} \} \right)$

Clipped gradient for fast and controlled guidance:
$\hat{\epsilon}(z_t, t) := \epsilon_\theta(z_t, t) - \omega \sqrt{1 - \bar{\alpha}_t} \frac{g}{\max(\eta, \|g\|)} , \text{ with } g = \text{clip}_\tau (\nabla_{z_0} \log L(z_t))$
where $\eta$ and $\tau$ are user-chosen parameters.

Spectrum computation:
$S(\ell, c) = \log \left( \frac{1}{3N} \sum_{i=1}^N \sum_{j=1}^3 |X^{(i)}(\ell, c, j)| \right) , \forall(\ell, c) \in [\![1, L]\!]^2$
where $X^{(i)}(:, :, j) = \text{FFT}(x^{(i)}(:, :, j))$.

Whitened detector:
$\phi_w(x) = L_\phi^{-1} (\phi(x^{(i)}) - b_\phi)$
where $b_\phi = \frac{1}{n} \sum_{i=1}^n \phi(x^{(i)})$ and $\Sigma_\phi = \frac{1}{n - 1} \sum_{i=1}^n (\phi(x^{(i)}) - b_\phi)(\phi(x^{(i)}) - b_\phi)^\top$, with Cholesky decomposition $\Sigma_\phi = L_\phi L_\phi^\top$.

P-value for Gaussian-Shading:
$p_X = I_{\frac{1}{2}} \left( s(\hat{U}, u), M - s(\hat{U}, u) + 1 \right)$
where $s(\hat{U}, u) = \sum_{i=1}^M [\hat{U}_i = u_i]$.

Tree-Ring detection score:
$s(\hat{z}, U) = \sum_{j=1}^J \sum_{i=1}^{D_j} |\hat{z}_{i,j} - U_j|^2 = \sum_{j=1}^J D_j |U_j - \lambda_j|^2 - c$
with $\lambda_j = \frac{\sum_{i=1}^{D_j} \hat{z}_{i,j}}{D_j} \in \mathbb{C}$, and $c = \sum_{j=1}^J \left( \frac{|\lambda_j|^2}{D_j} - \sum_{i=1}^{D_j} \hat{z}_{i,j}^2 \right)$.

8. Algorithm Pseudocode (If any, paste the pseudocode from the paper exactly as it is)
The paper does not provide a formal algorithm pseudocode block. The algorithmic implementation is defined through the mathematical iterative update rules. Based strictly on the original text and equations, the iterative process is executed as follows:

1. Input: Seed $z_T$, prompt, watermark message $m$, secret vector $u_m$, guidance strength $\omega$, transform set $\mathbb{T}$, start step $T_w$, clip threshold $\tau$, max norm $\eta$.
2. For $t = T$ down to 1:
   a. Compute estimated noise: $\epsilon_\theta(z_t, t)$
   b. If $t \le T_w$:
      i. Complete diffusion from $t$ to 0 and use VAE to get $x_0(z_t)$.
      ii. For each transformation $\mathcal{T} \in \mathbb{T}$:
          - Compute loss $L(z_t, u_m; \mathcal{T}) := 1 - \cos(\phi(\mathcal{T}(x_0(z_t))), u_m)$
          - Compute gradient $\nabla_{z_0} \log L(z_t, u_m; \mathcal{T})$
      iii. Aggregate gradients using PCGrad: $\text{Agg}(\{\nabla_{z_0} \log L(z_t, u_m; \mathcal{T}) | \mathcal{T} \in \mathbb{T}\})$
      iv. Clip gradient: $g = \text{clip}_\tau (\text{Agg})$
      v. Modify estimated noise: $\hat{\epsilon}(z_t, t) := \epsilon_\theta(z_t, t) - \omega \sqrt{1 - \bar{\alpha}_t} \frac{g}{\max(\eta, \|g\|)}$
   c. Else:
      - Set $\hat{\epsilon}(z_t, t) = \epsilon_\theta(z_t, t)$
   d. Update latent: $z_{t-1} = \text{Diffusion}(z_t, \hat{\epsilon}, t)$
3. Output final image: $x_0 = \text{VAE}(z_0)$