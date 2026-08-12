1. Research Background and Existing Pain Points
Research Background: Audio generation, also known as neural vocoding, aims to reconstruct high-resolution waveforms from compressed representations such as Mel-spectrograms and discrete audio tokens. It serves as a crucial component in various applications, including text-to-speech (TTS) and music synthesis, directly impacting the perceptual experience of end users. Deep generative models have demonstrated remarkable progress in this task, primarily dominated by Generative Adversarial Networks (GANs) and diffusion-based methods like Flow Matching.

Existing Pain Points:
1. Slow convergence and mode collapse in GANs: GANs leverage sophisticated discriminators to capture multi-grained audio details and enable high-fidelity results with efficient one-step inference. However, adversarial optimization in GANs often leads to slow convergence and the risk of mode collapse.
2. High computational overhead in Diffusion/Flow Matching: Diffusion-based methods offer stable training and strong generative performance, but they require multi-step sampling, resulting in high computational demands and considerable computational overhead during inference.
3. Velocity estimation difficulty in Flow Matching for audio: Audio signals often contain silent segments or frequency bands with zero energy. In these empty regions, the Flow Matching model must accurately estimate the velocity to precisely cancel out the noise and recover the empty target. This increases the learning difficulty as the model is required not only to capture the differences in the active regions of the audio signal, but also to approximate the negative noise in these empty regions.
4. Loss-perception mismatch: The standard Flow Matching loss uses mean squared error (MSE) criterion, treating the prediction errors uniformly across all signal regions. This does not align well with auditory perception principles, where errors in quieter spectral regions tend to be more perceptually noticeable than equivalent errors in louder regions.

2. Core Research Motivation and Scientific Questions
Core Research Motivation: To combine the strengths of both GAN and diffusion paradigms—leveraging Flow Matching for stable training and robust generative capability learning, while utilizing GAN for efficient few-step inference and fine-grained detail enhancement. Furthermore, there is a strong motivation to tailor the Flow Matching objective specifically to the unique characteristics of audio data (empty regions and perceptual sensitivity) to provide superior pretrained weights for subsequent GAN fine-tuning, and to design a network architecture that better captures audio complexity than single-resolution designs.

Scientific Questions:
1. How can the Flow Matching objective be reformulated to circumvent the difficulties of velocity estimation in silent or empty audio regions?
2. How can the Flow Matching loss be modified to align with human auditory perception by emphasizing errors in perceptually salient quieter regions?
3. How can a pre-trained Flow Matching model be effectively transformed into a few-step generator and further refined using GAN fine-tuning to achieve high-fidelity audio generation with minimal computational overhead?
4. How can a network architecture be designed to process Fourier coefficients at different time-frequency resolutions to improve modeling capabilities compared to prior single-resolution designs?

3. Overall Core Idea and Design Philosophy
Overall Core Idea: Flow2GAN follows a two-stage training strategy. In the first stage, the model is trained with an improved Flow Matching objective tailored for audio modeling to learn robust generative capabilities with stable training. In the second stage, few-step generators are constructed from the trained model and guided by GAN fine-tuning toward fine-grained generation and efficient few-step inference. Additionally, a multi-branch ConvNeXt-based network structure is designed to operate on multi-resolution Fourier coefficients.

Design Philosophy: The design philosophy is based on addressing audio-specific challenges directly within the Flow Matching formulation before applying GAN refinement. By reformulating the objective to endpoint estimation, the model avoids the difficulty of predicting velocity in empty regions. By incorporating spectral energy-based loss weighting, the model emphasizes perceptually important low-energy regions. Building upon this adapted Flow Matching, the trained model is able to generate clearly improved audio with just 2 sampling steps, providing a strong foundation for the subsequent GAN fine-tuning stage. The GAN fine-tuning then efficiently guides the initialized few-step models toward further refined audio quality by leveraging carefully designed discriminators that capture diverse audio patterns. This combines the strengths of both paradigms: improved Flow Matching for robust generative capability learning, and GAN for detail enhancement.

4. Core Innovation Points
1. Reformulating Flow Matching objective as endpoint estimation: To circumvent the difficulty of predicting the velocity in empty regions (silent segments), the Flow Matching problem is reformulated so that the network predicts the endpoint directly instead of the velocity.
2. Spectral energy-adaptive loss scaling: To address the loss-perception mismatch, the Flow Matching loss is scaled inversely proportional to the energy of the reference spectrogram, thereby encouraging the model to emphasize the perceptually salient quieter regions across both time and frequency dimensions.
3. Two-stage training strategy (Flow Matching + GAN fine-tuning): A lightweight GAN fine-tuning stage is employed to construct few-step generators from the improved Flow Matching pre-trained model. This enables few-step (1/2/4 steps) inference that produces high-quality audio, combining the stable training of Flow Matching with the efficient detail enhancement of GANs.
4. Multi-branch network architecture processing multi-resolution Fourier coefficients: A multi-branch ConvNeXt-based network structure is developed that processes Fourier coefficients at different time-frequency resolutions, serving as a powerful backbone with enhanced modeling capabilities compared to the previous single-resolution designs.

5. Overview of the Overall Technical Solution
The Flow2GAN framework operates in two stages: Stage-1 (Flow Matching training) and Stage-2 (GAN fine-tuning). 
In Stage-1, given a noise point sampled from a Gaussian distribution and a data point sampled from the data distribution, a straight flow path is defined by linear interpolation. The model, parameterized as an endpoint estimator conditioned on compressed acoustic representations (Mel-spectrograms or discrete audio tokens), is trained to predict the clean audio data point from its noisy version at varying noise levels. The training objective utilizes spectral energy-adaptive loss scaling, which converts the prediction error into the frequency domain using a power STFT followed by a linear filterbank transformation, and scales it element-wise inversely by the square root of the reference spectrogram's energy.
In Stage-2, few-step generators are constructed by forwarding the trained Flow Matching model through N steps (1, 2, or 4 steps) using a modified sampling process derived from the endpoint prediction. These few-step generators are then fine-tuned using a GAN objective. The fine-tuning utilizes a combination of HingeGAN adversarial loss, L1 feature matching loss, and multi-scale L1 Mel-spectrogram reconstruction loss, guided by multi-period discriminator (MPD) and multi-resolution discriminator (MRD).
The overall backbone utilizes a multi-branch network architecture where each branch processes Fourier coefficients from STFT at different time-frequency resolutions using ConvNeXt models. A ConvNeXt-based condition encoder extracts deeper features from the input conditions, which serve as shared generation conditions across all branches. The final output is obtained by summing the Inverse STFT outputs of all branches.

6. Detailed Module Design
1. Flow Matching Endpoint Estimator Module:
The network parameterizes the velocity field as an endpoint estimator. Instead of estimating the marginal velocity field, the network is trained to reconstruct the clean audio from its noisy version at varying noise levels. This allows the network to focus on capturing audio-relevant patterns regardless of whether the region is active or empty. During inference, the sampling process is modified to calculate the next step based on the predicted endpoint and the current noisy state.

2. Spectral Energy-Adaptive Loss Scaling Module:
To emphasize perceptually salient quieter regions, the prediction error is converted into the frequency domain. The module computes the power STFT followed by a linear filterbank transformation for energy smoothing, denoted as S(x). The obtained result is then scaled element-wise inversely by the square root of the reference spectrogram's energy. The loss scale is clamped between 0.01 and 100 for training stability. Unlike per-frame energy-based scaling, this strategy accounts for differences along both the time and frequency dimensions.

3. Few-Step Generator Module:
Few-step generators are constructed by forwarding the trained Flow Matching model through N steps following the modified sampling equation. Generators with different steps are trained and deployed as separate models rather than a single unified model. For generators with N > 1, gradients are backpropagated through both forward passes, enabling end-to-end optimization of the intermediate output from earlier steps. The model focuses on one-, two-, and four-step variants for computational efficiency.

4. GAN Fine-Tuning Module:
The few-step generators are fine-tuned using adversarial learning. The module employs a multi-period discriminator (MPD) that learns complementary structures by analyzing different periodic patterns, and a multi-resolution discriminator (MRD) that operates in the time-frequency domain across multiple spectral resolutions. The generators are fine-tuned using a combination of HingeGAN adversarial loss, L1 feature matching loss, and multi-scale L1 Mel-spectrogram reconstruction loss with window lengths of {32, 64, 128, 256, 512, 1024, 2048}, hop lengths set to window length / 4, and Mel bins of {5, 10, 20, 40, 80, 160, 320}. Small Gaussian noise can be added to the conditioning log-Mel during fine-tuning to improve robustness.

5. Multi-Resolution Network Structure Module:
The model comprises three branches, each processing Fourier coefficients at different resolutions. In each branch, the input signal is first transformed via STFT to obtain complex Fourier coefficients, whose real and imaginary components are concatenated along the feature dimension and fed into a ConvNeXt model to produce output complex coefficients. These are then converted back to waveform domain via Inverse STFT (ISTFT). The final output is obtained by summing all branch outputs. The model uses larger embedding dimensions for the low-frame-rate branch (768), and smaller embedding dimensions for the two high-frame-rate branches (512, 384). A ConvNeXt-based condition encoder extracts deeper features from input conditions (Mel-spectrograms or codec token embeddings), serving as shared generation conditions across all branches. For Flow Matching inference, this condition encoder requires only one forward pass, with output features reused across multiple sampling steps.

7. All Mathematical Formulas and Symbol Definitions
1. Flow Path Interpolation:
$x_t = (1 - t)x_0 + tx_1$
Where $x_0 \sim p_{noise}$, $x_1 \sim p_{data}$, and $t \sim \mathcal{U}[0, 1]$.

2. Velocity Definition:
$v_t = x_1 - x_0$

3. Marginal Velocity Field:
$v(x_t, t) = \mathbb{E}_{x_0, x_1}[v_t|x_t]$

4. Standard Flow Matching Loss:
$L_{FM} = \mathbb{E}_{t, x_0, x_1}[\|f_\theta(x_t, t) - v_t\|^2]$

5. Standard Sampling Process (Euler method):
$x_{t_{i+1}} = x_{t_i} + (t_{i+1} - t_i)f_\theta(x_{t_i}, t_i)$

6. Reformulated Flow Matching Loss (Endpoint Prediction):
$L_{FM} = \mathbb{E}_{t, x_0, x_1}[\|\frac{g_\theta(x_t, t|c) - x_t}{1 - t} - \frac{x_1 - x_t}{1 - t}\|^2] = \mathbb{E}_{t, x_0, x_1}[\frac{\|g_\theta(x_t, t|c) - x_1\|^2}{(1 - t)^2}]$
Where $c$ denotes the compressed acoustic representations, $g_\theta(x_t, t|c)$ is the endpoint estimator.

7. Simplified Flow Matching Loss (without weighting factor):
$L'_{FM} = \mathbb{E}_{t, x_0, x_1}[\|g_\theta(x_t, t|c) - x_1\|^2]$

8. Modified Sampling Process (Endpoint Prediction):
$x_{t_{i+1}} = x_{t_i} + (t_{i+1} - t_i)\frac{g_\theta(x_{t_i}, t_i|c) - x_{t_i}}{1 - t_i}$

9. Spectral Smoothing Function:
$S(x) = \text{LinFB}(|\text{STFT}(x)|^2)$
Denotes the power STFT followed by a linear filterbank transformation for energy smoothing.

10. Spectral Energy-Adaptive Loss:
$L''_{FM} = \mathbb{E}_{t, x_0, x_1}\left[\sum_{i,j}\left(\frac{S(g_\theta(x_t, t|c) - x_1)}{\sqrt{S(x_1) + \epsilon}}\right)_{i,j}\right]$
Where $\epsilon = 1e - 7$. The loss scale $\frac{1}{\sqrt{S(x_1) + \epsilon}}$ is clamped between 0.01 and 100.

8. Algorithm Pseudocode
The paper does not provide explicit algorithm pseudocode for the Flow2GAN method. It references external sources (Frans et al., 2024) for Shortcut model baselines. The algorithmic steps are defined by the mathematical formulations and training strategy described in the Method section.