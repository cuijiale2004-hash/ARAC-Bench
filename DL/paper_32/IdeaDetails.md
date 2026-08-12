
### 1. Research Background and Existing Pain Points

**Research Background:**
Deep learning-based vocoders have significantly advanced speech synthesis, producing more natural and expressive synthetic speech. Recent developments include models based on generative adversarial networks (GANs), normalizing flow-based models, and diffusion-based models. An alternative to these approaches is to synthesize speech in the spectral domain using the inverse short-time Fourier transform (iSTFT). Operating directly on complex spectrograms avoids the need for sample-by-sample generation and learned upsampling, which are common sources of increased model complexity and inference latency. iSTFT-based vocoders generate STFT-domain coefficients directly for frame-level synthesis, enabling efficient inference without waveform upsampling.

**Existing Pain Points:**
Current iSTFT-based vocoders rely on real-valued neural networks (RVNNs) that process the real and imaginary parts as separate channels. This separation limits their ability to model the coupling between these components and capture the inherent structure of complex spectrograms. Although some recent vocoders produce complex spectrograms, they still use real-valued networks that handle each spectrogram channel independently, missing cross-component interactions. Furthermore, phase reconstruction remains challenging; early methods like the Griffin-Lim algorithm often yielded suboptimal coherence between magnitude and phase. Finally, standard implementations of complex-valued operations in many autodifferentiation systems explicitly track real and imaginary components as separate real-valued tensors, leading to redundant operations, inefficient memory access, and high computational overhead during both forward and backward passes.

### 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:**
Complex-valued neural networks (CVNNs) extend standard neural networks to the complex domain by allowing both inputs and parameters to be complex-valued. Operating entirely in the complex domain enables these models to capture the intrinsic dependencies between the real and imaginary components. Motivated by the fact that data in the spectral domain naturally form complex-valued data carrying both magnitude and phase information, there is a strong rationale for adopting CVNNs to better capture structure in the complex domain. Preliminary analysis comparing RVNN and CVNN on a synthetic complex distribution reveals that both models recover the broad structure of the target distribution, but the CVNN yields samples that adhere more closely to the underlying trajectory and exhibit lower Jensen–Shannon divergence (JSD) in both magnitude and phase (CVNN JSD (mag.): 0.006548 ± 0.003, JSD (phase): 0.003911 ± 0.002 vs. RVNN JSD (mag.): 0.018350 ± 0.014, JSD (phase): 0.021110 ± 0.036). This demonstrates that modeling directly in the complex domain offers representational advantages when the data possess inherent real–imaginary dependencies. 

**Scientific Questions:**
How can complex-valued neural networks be effectively integrated into an iSTFT-based vocoder architecture such that both the generator and discriminator operate natively in the complex domain to preserve real-imaginary interactions end-to-end? How can phase transformations be guided in a structured manner to mitigate phase drift and stabilize training within a complex-valued framework? How can the computational efficiency of complex-valued operations be improved to reduce redundant operations and accelerate training without sacrificing numerical fidelity?

### 3. Overall Core Idea and Design Philosophy

**Overall Core Idea:**
The core idea is to present ComVo, a Complex-valued neural Vocoder that performs iSTFT-based waveform generation entirely in the complex domain with a GAN-based architecture. The generator uses CVNN layers to jointly model the real and imaginary components of spectrograms, thereby better capturing their algebraic structure. A complex multi-resolution discriminator (cMRD) is designed to operate directly on complex spectrograms, establishing a complex-domain adversarial training framework where feedback respects the structure of the complex domain. 

**Design Philosophy:**
The design philosophy rests on three pillars: 1) **Unified Complex-Domain Modeling**: Treating each spectrogram coefficient as a unified complex entity rather than separating it into real and imaginary channels, ensuring cross-component interactions are captured. 2) **Structured Nonlinear Transformation**: Introducing phase quantization as an inductive bias to discretize phase angles, limiting unwarranted phase variability and guiding the network toward learning more coherent and structured phase patterns. 3) **Optimized Complex Computation**: Reformulating complex operations as real-valued block-matrix multiplications to eliminate redundant computation, reduce the computational graph complexity, and enhance parallelism on GPU architectures.

### 4. Core Innovation Points

1. **CVNN-based Architecture with Complex Adversarial Training**: ComVo is the first iSTFT-based vocoder to employ complex-valued neural networks in both its generator and discriminator. Discriminator losses are designed in the complex domain, establishing an adversarial framework that operates on complex-valued representations and provides structured spectral feedback.
2. **Structured Nonlinear Transformation (Phase Quantization)**: A tailored nonlinear operation that discretizes phase angles into a fixed set of levels. It serves as an inductive bias that preserves relative phase relationships, mitigates phase drift during training, and acts as a regularizer to enhance listening quality.
3. **Block-Matrix Computation Scheme**: An efficient implementation that fuses the four real-valued multiplications required for each complex operation into a single block-matrix multiplication. This reduces the number of separate operations, eliminates redundant computation, lowers backward graph nodes significantly, and reduces training time by 25%.
4. **Superior Synthesis Performance over Parameter Scaling**: ComVo demonstrates that modeling real–imaginary correlations with CVNNs provides larger quality gains than simply scaling real-valued models with comparable memory footprints, consistently outperforming real-valued baselines in objective and subjective metrics.

### 5. Overview of the Overall Technical Solution

ComVo is an iSTFT-based GAN vocoder whose generator and discriminator operate entirely in the complex domain, preserving real-imaginary interactions end to end. The model uses an iSTFT synthesis pipeline with adversarial training objectives. The generator adapts the Vocos architecture by replacing all convolutions and normalizations with complex-domain equivalents and utilizing a split GELU activation to maintain the ConvNeXt-style block layout. A phase quantization layer is inserted after the initial complex convolution to discretize phase values and stabilize training. The discriminator comprises a proposed complex multi-resolution discriminator (cMRD) that operates directly on complex spectrograms using complex-valued layers, alongside a standard real-valued multi-period discriminator (MPD) operating on waveforms. The overall training objective combines adversarial losses, feature matching losses, and reconstruction losses, with cMRD losses applied separately to the real and imaginary parts. Finally, all parameterized CVNN layers are implemented via a block-matrix formulation to improve training efficiency by reducing redundant operations.

### 6. Detailed Module Design

**Generator:**
The generator is adapted from the Vocos architecture, synthesizing via frame-level iSTFT without requiring learned upsampling. All convolutions and normalizations are implemented in the complex domain. It uses a split GELU activation to maintain the ConvNeXt-style block layout in the complex setting. After the initial complex convolution, a phase quantization layer discretizes phase values to stabilize training. The core building block is the complex ConvNeXt block, which comprises a complex depthwise convolution, a complex layer normalization, a complex GELU activation, and a complex pointwise convolution. The generator outputs a complex-valued spectrogram, which is then transformed into a waveform via iSTFT.

**Discriminator:**
The discriminator consists of two components: the complex multi-resolution discriminator (cMRD) and the multi-period discriminator (MPD).
*   **cMRD**: Uses complex-valued layers and operates directly on complex spectrogram inputs. It comprises multiple sub-discriminators, each operating at a different STFT resolution. Unlike prior spectrogram-based discriminators that use only magnitude or concatenate real and imaginary channels as independent inputs to a real-valued network, cMRD processes complex spectrograms natively. During training, component-wise hinge losses are applied to the real and imaginary outputs separately to retain compatibility with standard real-valued GAN objectives while operating in the complex domain.
*   **MPD**: Consists of multiple sub-discriminators operating over different periods and processing reshaped waveform segments. Because the MPD operates at the waveform level, it remains a real-valued network.

**Phase Quantization Layer:**
Complex-valued networks require nonlinear transformations that jointly handle the real and imaginary components. The input Mel-spectrogram is represented as a complex value by initializing the imaginary part to zero. The phase quantization layer discretizes phase angles into a fixed set of levels. For a complex feature $z = re^{i\theta}$, the quantized phase is defined by mapping continuous angles to a fixed set of levels. To preserve end-to-end differentiability, the straight-through estimator (STE) is adopted, where the quantization operation is applied in the forward pass, while its gradient is approximated by an identity function during backpropagation. This limits unwarranted phase variability and acts as a regularizer.

**Block-Matrix Computation Scheme:**
To improve efficiency in both forward and backward passes, CVNN operations are reformulated as real-valued block-matrix multiplications. Instead of explicitly tracking real and imaginary components as separate real-valued tensors, complex values are represented as structured pairs of real values and processed jointly through unified matrix operations. The forward complex operation and the backward gradient computation both use the same block matrix structure. This unified formulation is implemented for all parameterized CVNN layers via custom autograd functions. It replaces four independent real-valued multiplies with a single block-matrix multiply, eliminating redundant computation and allowing more efficient gradient evaluation.

**Complex-Valued Neural Network Building Blocks (Appendix A):**
*   **Complex Convolutions**: Perform convolutions directly in the complex domain, jointly processing the real and imaginary parts. For input $z = x + iy$ and filter $h = a + ib$, the output $z'$ is: $z' = (x * a - y * b) + i(x * b + y * a)$.
*   **Activation Functions**: A simple split activation applies $f_{Re}$ and $f_{Im}$ separately to the real and imaginary components: $f(z) = f_{Re}(x) + if_{Im}(y)$. A phase-aware alternative applies $f_{Mag}$ to the magnitude and reattaches the original phase: $f(z) = f_{Mag}(|z|)e^{i\theta}$.
*   **Normalization**: Accounts for the joint distribution using the covariance matrix $\Sigma = \left[\begin{array}{cc} \sigma_{xx} & \sigma_{xy} \\ \sigma_{yx} & \sigma_{yy} \end{array}\right]$. The input is normalized by centering and decorrelating $z_{norm} = \Sigma^{-1/2}(z - \mu)$, followed by an affine transformation $z' = \gamma z_{norm} + \beta$ with learnable complex-valued parameters $\gamma$ and $\beta$.
*   **Gradient Optimization**: Employs Wirtinger calculus. For real-valued objectives, only the conjugate gradient $\frac{\partial L}{\partial \bar{z}}$ is used for parameter updates.

### 7. All Mathematical Formulas and Symbol Definitions

**Phase Quantization Formulas:**
*   $\theta_q = \frac{2\pi}{N_q} \cdot \text{round}\left(\frac{N_q}{2\pi}\theta\right)$
    *   $z = re^{i\theta}$: complex feature, where $r \ge 0$ denotes magnitude and $\theta \in (-\pi, \pi]$ denotes principal phase.
    *   $N_q$: number of quantization levels.
    *   $\theta_q$: quantized phase.
*   $z_q = re^{i\theta_q}$
    *   $z_q$: quantized complex value.

**Block-Matrix Computation Scheme Formulas:**
*   Forward operation:
    $\left[\begin{array}{c} \text{Re}(z') \\ \text{Im}(z') \end{array}\right] = \left[\begin{array}{cc} W_r & -W_i \\ W_i & W_r \end{array}\right] \left[\begin{array}{c} x \\ y \end{array}\right]$
    *   $z = x + iy$: complex input ($x, y$ are real and imaginary input vectors).
    *   $W = W_r + iW_i$: complex weight matrix ($W_r, W_i$ are real and imaginary parts).
    *   $z'$: resulting complex output.
*   Backward gradient computation:
    $\left[\begin{array}{c} \frac{\partial L}{\partial x} \\ \frac{\partial L}{\partial y} \end{array}\right] = \left[\begin{array}{cc} W_r & -W_i \\ W_i & W_r \end{array}\right]^\top \left[\begin{array}{c} g_r \\ g_i \end{array}\right]$
    *   $g_r, g_i$: real and imaginary components of the gradient from the next layer.
    *   $L$: scalar loss.

**CVNN Building Block Formulas (Appendix A):**
*   Complex Convolution: $z' = (x * a - y * b) + i(x * b + y * a)$
*   Split Activation: $f(z) = f_{Re}(x) + if_{Im}(y)$
*   Magnitude-preserving Activation: $f(z) = f_{Mag}(|z|)e^{i\theta}$
*   Covariance Matrix: $\Sigma = \left[\begin{array}{cc} \sigma_{xx} & \sigma_{xy} \\ \sigma_{yx} & \sigma_{yy} \end{array}\right]$
*   Normalization: $z_{norm} = \Sigma^{-1/2}(z - \mu)$
*   Affine Transformation: $z' = \gamma z_{norm} + \beta$
*   Wirtinger Gradients:
    $\frac{\partial L}{\partial z} = \frac{1}{2}\left(\frac{\partial L}{\partial x} - i\frac{\partial L}{\partial y}\right)$
    $\frac{\partial L}{\partial \bar{z}} = \frac{1}{2}\left(\frac{\partial L}{\partial x} + i\frac{\partial L}{\partial y}\right)$
*   Parameter Update: $z^{(t+1)} = z^{(t)} - \eta \frac{\partial L}{\partial \bar{z}}$

**Loss Function Formulas:**
*   MPD Loss:
    $L_D^{MPD} = \sum_{k=1}^K \left[ \mathbb{E}_y\left(\max(0, 1-D_k^{MPD}(y))\right) + \mathbb{E}_{\hat{y}}\left(\max(0, 1+D_k^{MPD}(\hat{y}))\right) \right]$
    *   $D_k^{MPD}$: $k$-th sub-discriminator of MPD.
    *   $y, \hat{y}$: ground-truth and generated waveform segments.
*   cMRD Loss:
    $L_D^{cMRD} = \sum_{k=1}^K \left[ \frac{1}{2}\mathbb{E}_z\left(\max(0, 1-[D_k^{cMRD}(z)]_R) + \max(0, 1-[D_k^{cMRD}(z)]_I)\right) + \frac{1}{2}\mathbb{E}_{\hat{z}}\left(\max(0, 1+[D_k^{cMRD}(\hat{z})]_R) + \max(0, 1+[D_k^{cMRD}(\hat{z})]_I)\right) \right]$
    *   $D_k^{cMRD}$: $k$-th sub-discriminator of cMRD.
    *   $z, \hat{z}$: ground-truth and generated complex spectrograms.
    *   $[\cdot]_R, [\cdot]_I$: real and imaginary parts of a complex output.
*   Mel-spectrogram Loss:
    $L_{Mel} = \mathbb{E} \|M(y) - M(\hat{y})\|_1$
    *   $M(\cdot)$: log-Mel transform.
*   Adversarial Generator Loss (MPD):
    $L_G^{MPD} = \sum_{k=1}^K \mathbb{E}_{\hat{y}}\left(\max(0, 1-D_k^{MPD}(\hat{y}))\right)$
*   Adversarial Generator Loss (cMRD):
    $L_G^{cMRD} = \sum_{k=1}^K \frac{1}{2}\mathbb{E}_{\hat{z}}\left(\max(0, 1-[D_k^{cMRD}(\hat{z})]_R) + \max(0, 1-[D_k^{cMRD}(\hat{z})]_I)\right)$
*   Feature Matching Loss (MPD):
    $L_{FM}^{MPD} = \sum_{k=1}^K \sum_{l=1}^{L_k} \mathbb{E} \|D_{k,l}^{MPD}(y) - D_{k,l}^{MPD}(\hat{y})\|_1$
    *   $D_{k,l}^{MPD}$: $l$-th layer feature of the $k$-th MPD sub-discriminator.
*   Feature Matching Loss (cMRD):
    $L_{FM}^{cMRD} = \sum_{k=1}^K \sum_{l=1}^{L_k} \frac{1}{2}\mathbb{E}\left(\|[D_{k,l}^{cMRD}(z)]_R - [D_{k,l}^{cMRD}(\hat{z})]_R\|_1 + \|[D_{k,l}^{cMRD}(z)]_I - [D_{k,l}^{cMRD}(\hat{z})]_I\|_1\right)$
*   Total Generator Loss:
    $L_{gen} = \lambda_{Mel} L_{Mel} + \lambda_{MPD}\left(L_G^{MPD} + L_{FM}^{MPD}\right) + \lambda_{cMRD}\left(L_G^{cMRD} + L_{FM}^{cMRD}\right)$
    *   $\lambda_{Mel}, \lambda_{MPD}, \lambda_{cMRD}$: weighting coefficients for Mel, MPD, and cMRD terms.

### 8. Algorithm Pseudocode

**Algorithm 1: Phase Quantization Forward and Backward Pass**
Input: Complex feature $z = re^{i\theta}$, Number of quantization levels $N_q$
Output: Quantized complex value $z_q$, Gradient approximation $\frac{\partial L}{\partial z}$

1. // Forward Pass
2. Compute principal phase $\theta \in (-\pi, \pi]$
3. Calculate quantized phase: $\theta_q = \frac{2\pi}{N_q} \cdot \text{round}\left(\frac{N_q}{2\pi}\theta\right)$
4. Reconstruct quantized complex value: $z_q = re^{i\theta_q}$
5. // Backward Pass (Straight-Through Estimator)
6. Approximate gradient of quantization operation as identity function:
7. $\frac{\partial L}{\partial z} \approx \frac{\partial L}{\partial z_q}$
8. Return $z_q$, $\frac{\partial L}{\partial z}$

**Algorithm 2: Block-Matrix Computation Scheme Forward and Backward Pass**
Input: Complex input vector $z = x + iy$, Complex weight matrix $W = W_r + iW_i$, Upstream gradient $[g_r; g_i]$
Output: Complex output vector $z'$, Input gradient $[\frac{\partial L}{\partial x}; \frac{\partial L}{\partial y}]$

1. // Forward Computation
2. Construct block matrix $A = \left[\begin{array}{cc} W_r & -W_i \\ W_i & W_r \end{array}\right]$
3. Stack real input vectors $v = \left[\begin{array}{c} x \\ y \end{array}\right]$
4. Compute forward pass as single block-matrix multiplication:
5. $\left[\begin{array}{c} \text{Re}(z') \\ \text{Im}(z') \end{array}\right] = A \cdot v$
6. // Backward Computation
7. Compute input gradient using the transpose of the forward block matrix:
8. $\left[\begin{array}{c} \frac{\partial L}{\partial x} \\ \frac{\partial L}{\partial y} \end{array}\right] = A^\top \left[\begin{array}{c} g_r \\ g_i \end{array}\right]$
9. Return $z'$, $[\frac{\partial L}{\partial x}; \frac{\partial L}{\partial y}]$