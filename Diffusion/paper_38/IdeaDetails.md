# 1. Research Background and Existing Pain Points

**Research Background:**
Recent generative models have shown the remarkable ability to produce temporally consistent videos. The objects within them persist across frames, through occlusion, and despite variations in camera pose and lighting. These capabilities are closely related to the visual tracking problem. While generation deals with producing videos that contain temporally persistent objects, tracking deals with analyzing such videos to estimate motion. A variety of methods have exploited the connections between these two problems, such as by using trackers to supervise or control video generators and to evaluate the temporal consistency of generated videos by measuring how “trackable” they are. In the domain of self-supervised motion estimation, deep learning has significantly advanced the field, but early dense optical flow methods often struggle with long-range tracking and occlusions. Recent methods instead track individual points over time, but these models often rely on synthetic data, limiting their real-world generalization. To bridge this gap, self-supervised optical flow methods and long-range tracking methods (such as training a model to propagate color in grayscale videos or using cycle consistency) have been proposed. Models trained for semantic understanding have also been adapted for semantic and temporal correspondence. Large pretrained models have become foundational in computer vision, and diffusion models for image generation introduced generative features that capture semantic correspondences, but lack temporal reasoning needed for motion-centric tasks. Video diffusion models address temporal consistency, though many still prioritize appearance over motion.

**Existing Pain Points:**
1. **Inability to Induce Tracking via Text Prompting:** Unlike high-level understanding tasks that are naturally described by captions (like object recognition), tracking cannot easily be induced by text prompting. 
2. **Limitations of Existing Zero-Shot Methods:** Existing zero-shot emergent correspondence methods treat the pretrained models as representation learners: they extract their internal features and use them to match pairs of images. These methods lack the temporal reasoning and object permanence needed for tracking through occlusion, as they work by matching individual pairs of video frames. A concurrent work extracts features from a pretrained video model for tracking but involves a complex, architecture-dependent analysis to identify which layers provide the best features and does not handle occlusion.
3. **Strong Priors of Generative Models Conflicting with Counterfactual Modeling:** In counterfactual modeling, one carefully perturbs the input variables, then analyzes how the generation changes in response. Yet large generative models have strong priors that sometimes conflict with this goal. For example, a distinctively colored marker placed on an object may be unnatural in some environments, and so samples from a generative model may ignore it, causing the marker to disappear from the generated video within a few frames.
4. **Artifacts in Direct Image Subtraction for Counterfactuals:** Previous work on counterfactual world models generates two possible future images (one with the marker and one without) using a masked autoencoder and enhances the signal by directly subtracting the two generated images. In video diffusion models, objects often subtly change position in different samples, leading to these differences between generations containing significant artifacts, making it challenging to use this approach.
5. **Requirement for Special-Purpose Training:** Prior counterfactual world modeling methods require training a special-purpose model (based on masked autoencoders) designed specifically with the downstream use case in mind, and requires training auxiliary models to obtain high performance.

# 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:**
Trackers and video generators solve closely related problems: the former analyze motion, while the latter synthesize it. This connection suggests that the object permanence capabilities of generative models can be exploited for motion estimation. If video diffusion models can generate temporally consistent videos where objects persist through occlusions, they inherently possess an understanding of motion and object permanence. There is a need to find a way to extract these emergent tracking capabilities from off-the-shelf video diffusion models without requiring additional training or complex architectural analysis. 

**Scientific Questions:**
Do tracking capabilities emerge automatically in video diffusion models, as a consequence of the close connection between generation and tracking? Can we elicit these capabilities from a video generator using a novel approach to counterfactual modeling that allows us to directly obtain high-quality point tracks “zero shot” from pretrained image-conditioned video diffusion models, without the need for additional training or specialized models?

# 3. Overall Core Idea and Design Philosophy

**Overall Core Idea:**
The core idea is to repurpose a pretrained generative video model to track points in a video by prompting it to visually mark point trajectories. We place a distinctively colored marker (a small circular red dot) at the query point in the initial video frame, then propagate it to future video frames by regenerating the video using SDEdit (adding an intermediate level of noise and running the reverse diffusion process), which propagates the marking to subsequent frames. To ensure that the marker remains visible in this counterfactual generation, despite such markers being unlikely in natural videos, we use the unedited initial frame as a negative prompt for the diffusion model, thereby guiding the model toward samples that contain the marker. After generation, the query point’s position can be estimated in each frame by basic image processing. 

**Design Philosophy:**
The design philosophy is rooted in counterfactual modeling and visual prompting. Instead of training a special-purpose model or extracting internal features, we prompt a video diffusion model to draw a distinctive marker in each frame at the point’s position. We address the challenge of strong generative priors ignoring the manipulation by using a simple but effective negative prompting strategy that biases the score direction away from samples from the unedited conditioning. We further refine the tracking through a coarse-to-fine process using inpainting to correct spatial shifts, and color rebalancing to reduce false positives. Finally, we demonstrate that the temporal reasoning capabilities of large video diffusion models can be transferred into a lightweight tracking network through distillation.

# 4. Core Innovation Points

1. **Zero-Shot Point Tracking via Visual Prompting:** Pretrained video diffusion models can be directly used as visual trackers without additional training by simply prompting them with a distinctively colored marker on the query point and regenerating the video. This is the first use of image prompting for point tracking in video diffusion models.
2. **Novel Diffusion Prompting Strategy for Counterfactual Signal Enhancement:** A negative prompting strategy that ensures the generated video contains the marker by computing the difference between two noise estimates conditioned on the edited image (with marker) and unedited image (without marker). This biases the score direction away from samples that resemble the original video, maintaining point consistency across frames.
3. **Coarse-to-Fine Refinement via Inpainting:** Tracking performance is improved through iterative refinement using inpainting. After obtaining initial coarse estimates, a spatiotemporal binary mask is constructed around the tracked location, and the video generation is re-run restricting modifications to masked regions, correcting temporal artifacts and spatial shifts during denoising.
4. **Color Rebalancing for False Positive Suppression:** A color rebalancing technique that reduces the saturation of the marker's color (e.g., red hues) in the background of the input video. This ensures the marker remains a unique tracking cue, reducing false detections during occlusion when the marker is not present.
5. **Distillation into a Fast Tracker:** Trajectories produced by the slow, pretrained generators can be distilled into a fast, efficient tracker (like CoTracker) with comparable performance. The generative model produces pseudo-labels on unlabeled videos, which serve as effective supervision to train a feed-forward tracking network from scratch.

# 5. Overview of the Overall Technical Solution

The overall technical solution operates by taking an input video and the pixel location of a query point. First, color rebalancing is applied to the input video to suppress the marker's color (red) in the background. A distinctively colored marker (a red dot) is inserted at the query point's position in the first frame of the rebalanced video. The video diffusion model is then applied to propagate this marker across frames. To do this, an intermediate level of noise is added to the video (SDEdit), and the reverse diffusion process is run. During denoising, the model is conditioned on the edited first frame (with the marker) and uses the unedited first frame (without the marker) as a negative prompt to enhance the counterfactual signal. This generates a video where the marker persists on the moving object. A simple color-based tracker extracts the coarse trajectory by locating the red marker in each frame based on HSV color space within a local search window. To refine this trajectory, a spatiotemporal binary mask is constructed around the coarse tracked locations. The video generation is run again using an inpainting mechanism, where only the masked regions are allowed to change, preserving the rest of the video content and correcting spatial shifts. The refined marker locations are extracted to form the final track. Additionally, for efficiency, the generated trajectories from the video diffusion model can be used as pseudo-labels to distill the knowledge into a lightweight tracking architecture (CoTracker).

# 6. Detailed Module Design

**6.1 Latent Video Diffusion Models (Preliminaries)**
These models generate a sequence of $F$ RGB frames, $V \in \mathbb{R}^{F \times H \times W \times 3}$. They operate on a compact latent representation $x \in \mathbb{R}^{F' \times H' \times W' \times C}$, where $C$ is the channel dimension, which can be converted into a video via a decoder.
*   **Forward (Noising) Process:** Given a clean video latent $x_0$, the noising process is defined using a variance schedule $\beta_t$ over timesteps $t \in \{1, \ldots, T\}$. The corrupted latent is constructed by adding Gaussian noise.
*   **Reverse (Denoising) Process:** At each timestep $t$, the video diffusion model, $\epsilon_\theta(x_t, t, c)$, predicts the noise component. These models may be conditioned on additional data $c$, such as a text prompt or the desired first frame of the video. The denoising step computes the previous latent.
*   **Video Manipulation (SDEdit):** Rather than generating a latent vector from scratch, one can regenerate an existing, clean video with modifications using SDEdit by adding an intermediate level of noise, $1 < t < T$, and then running the reverse diffusion process to denoise it. This results in a video that resembles the coarse structure of the original, but with different fine-grained details.
*   **Inpainting:** Given a binary spatiotemporal mask $m \in \mathbb{B}^{F \times H \times W}$ indicating which patches of the input video can and cannot be changed, the reverse diffusion process is run and updates are constrained to the masked region. At each denoising step, the update is computed for the masked region, while the original noised latent is kept for the unmasked region, and they are combined using the mask.

**6.2 Point Prompting for Counterfactual Tracking**
*   **Marking a Point’s Trajectory:** Given an input video and the pixel location of a query point, the goal is to predict the positions of the point in the subsequent frames. A distinctive marking (a pure red circular dot) is inserted on the query point’s position in the initial frame. SDEdit is then applied using an intermediate timestep $1 < t < T$ to manipulate the video, while conditioning on the edited initial frame. This propagates the marker to the subsequent frames of the video.
*   **Enhancing the Counterfactual Signal:** To prevent the strong priors of the generative model from ignoring the unnatural marker, a simple negative prompt is used that reduces the probability of drawing samples that resemble the original video. The difference between two noise estimates is computed: one conditioned on the original image (without the marker) and another conditioned on the edited image (with the marker). The modified noise estimate is used in the denoising step.
*   **Tracking the Marker:** To extract a track from the generated videos, a simple tracker locates the marker in each frame based on color. Given the marker’s initial location $(u_0, v_0)$ in the first frame, it tracks its motion frame by frame. For each subsequent frame $k$, the tracker searches for red pixels (in HSV colorspace) within a local window of radius $r$ centered at the previous location $(u_{k-1}, v_{k-1})$, selecting the pixel closest to the previous position. Since the marker appears as a small blob, the estimate is refined by averaging the positions of nearby red pixels to obtain a more stable center, which serves as the predicted track point. If no red pixels are found within the search region, the marker is treated as occluded and the last known position is propagated forward. The search radius $r$ is expanded at each step until the marker reappears, after which $r$ is reset to its original value.

**6.3 Extensions**
*   **Coarse-to-Fine Refinement:** To correct subtle misalignments between the generated video and the original video after regeneration, inpainting is used to refine the initial estimates. After obtaining the initial estimate of marker positions, a spatiotemporal binary mask $m \in \mathbb{B}^{F \times H \times W}$ is constructed, where each frame’s mask is set to 1 within a small radius $r$ centered on the tracked location, i.e., $m[u, v]$ is set to 1 if $(u, v) \in B_r(u_k, v_k)$. The video generation is re-run, while allowing only the image regions indicated by $m$ to change, using the inpainting equation.
*   **Color Rebalancing:** Since the tracker relies on detecting a particular color, the video’s colors are rebalanced such that the marker’s color does not appear within it. This is done by reducing the saturation of the marker’s color. For example, when tracking a red marker, the saturation of red regions is reduced, effectively suppressing natural red hues while preserving overall image quality. This reduces mistakes during occlusion, since the marker is not present and thus false detections are more common.

**6.4 Distilling the Generator into a Tracker**
To obtain a fast tracker, the generation-based method is distilled into an efficient tracking architecture. Pseudo-label trajectories are collected by running the marker propagation method on 1,000 unlabeled videos from the TAP-Vid Kinetics dataset. These extracted trajectories serve as supervision to train CoTracker from scratch.

# 7. All Mathematical Formulas and Symbol Definitions

**Symbol Definitions:**
*   $F$: Number of RGB frames in the video.
*   $H, W$: Height and width of the video frames.
*   $V \in \mathbb{R}^{F \times H \times W \times 3}$: The sequence of RGB frames.
*   $x \in \mathbb{R}^{F' \times H' \times W' \times C}$: The compact latent representation of the video.
*   $C$: The channel dimension of the latent representation.
*   $x_0$: The clean video latent.
*   $\beta_t$: The variance schedule over timesteps $t \in \{1, \ldots, T\}$.
*   $\alpha_t = 1 - \beta_t$.
*   $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$.
*   $\epsilon \sim \mathcal{N}(0, I)$: Standard Gaussian noise.
*   $\epsilon_\theta(x_t, t, c)$: The video diffusion model predicting the noise component, conditioned on data $c$.
*   $\sigma_t^2$: The variance at timestep $t$.
*   $z \sim \mathcal{N}(0, I)$: Standard Gaussian noise.
*   $m \in \mathbb{B}^{F \times H \times W}$: Binary spatiotemporal mask indicating which patches can (1) and cannot (0) be changed.
*   $c_I$: The initial-frame conditioning.
*   $\phi(c_I)$: The initial frame after applying the counterfactual manipulation (i.e., adding the marker).
*   $\lambda > 0$: A weight for the negative prompt enhancement.
*   $\tilde{\epsilon}_\theta$: The noise estimate after enhancement.
*   $p(x_t)$: The probability under the model for the noisy input at time $t$.
*   $(u_0, v_0)$: The marker’s initial location in the first frame.
*   $r$: The radius of the local search window.
*   $B_r(u_k, v_k)$: The set of pixels within a small radius $r$ centered on the tracked location $(u_k, v_k)$.

**Mathematical Formulas:**

Forward (Noising) Process:
$$x_t = \sqrt{\alpha_t}x_{t-1} + \sqrt{1-\alpha_t}\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

Reverse (Denoising) Process:
$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, t, c)\right) + \sigma_t z$$

SDEdit Noising:
$$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$$

Inpainting Denoising Step:
$$\tilde{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta\right) + \sigma_t z, \quad z \sim \mathcal{N}(0, I)$$
$$x_{t-1}^{original} = \sqrt{\bar{\alpha}_{t-1}}x_0 + \sqrt{1-\bar{\alpha}_{t-1}}\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$
$$x_{t-1} = m \odot \tilde{x}_{t-1} + (1-m) \odot x_{t-1}^{original}$$

Enhancing the Counterfactual Signal (Negative Prompting):
$$\tilde{\epsilon}_\theta (x_t, c_I) = (\lambda+1) \cdot \epsilon_\theta(x_t, \phi(c_I)) - \lambda \cdot \epsilon_\theta (x_t, c_I)$$

Score Function Interpretation of Negative Prompting:
$$\nabla_{x_t} \log(p_\lambda(x_t)) = \nabla_{x_t} \log \left( p(x_t | \phi(c_I)) \left[ \frac{p(\phi(c_I) | x_t)}{p(c_I | x_t)} \right]^\lambda \right)$$

# 8. Algorithm Pseudocode

The paper does not provide explicit algorithm pseudocode blocks. The logical process extracted from the method description is as follows:

1. **Input:** Input video, query point location $(u_0, v_0)$ in the first frame.
2. **Color Rebalancing:** Rebalance the colors of the input video by reducing the saturation of the marker's color (red) to suppress natural hues.
3. **Marking:** Place a distinctively colored marker (pure red circular dot) at the query point location $(u_0, v_0)$ in the first frame of the rebalanced video to create the edited first frame $\phi(c_I)$. Let the unedited first frame be $c_I$.
4. **Point Propagation (SDEdit + Negative Prompting):** Add intermediate noise to the video to get $x_t$. For each denoising step from $t$ down to $0$:
   a. Predict noise conditioned on edited frame: $\epsilon_\theta(x_t, \phi(c_I))$.
   b. Predict noise conditioned on unedited frame: $\epsilon_\theta(x_t, c_I)$.
   c. Compute enhanced noise estimate: $\tilde{\epsilon}_\theta (x_t, c_I) = (\lambda+1) \cdot \epsilon_\theta(x_t, \phi(c_I)) - \lambda \cdot \epsilon_\theta (x_t, c_I)$.
   d. Update latent: $x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\tilde{\epsilon}_\theta\right) + \sigma_t z$.
5. **Coarse Tracking:** For each frame $k$ in the generated video:
   a. Search for red pixels in HSV colorspace within a local window of radius $r$ centered at $(u_{k-1}, v_{k-1})$.
   b. If red pixels found: Average their positions to get coarse track point $(u_k, v_k)$. Reset $r$.
   c. If no red pixels found: Propagate last known position $(u_k, v_k) = (u_{k-1}, v_{k-1})$. Expand radius $r$.
6. **Coarse-to-Fine Refinement (Inpainting):** 
   a. Construct a spatiotemporal binary mask $m$ where $m[u, v] = 1$ if $(u, v) \in B_r(u_k, v_k)$ for each frame $k$, and $0$ otherwise.
   b. Re-run video generation (SDEdit + Negative Prompting as in step 4), but apply the inpainting constraint at each step: $x_{t-1} = m \odot \tilde{x}_{t-1} + (1-m) \odot x_{t-1}^{original}$.
7. **Final Tracking:** Apply the Coarse Tracking step (Step 5) on the refined generated video to extract the final trajectories.