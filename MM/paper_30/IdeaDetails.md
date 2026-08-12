**1. Research Background and Existing Pain Points**

**Research Background:**
Reinforcement learning (RL) based post-training has recently emerged as a powerful paradigm for enhancing the alignment and reasoning capabilities of multimodal large language models (MLLMs). Following the success of Reinforcement Learning from Verifiable Reward (RLVR) in the large language models domain, which has unlocked substantial breakthroughs in complex reasoning abilities, the research community has largely shifted its focus toward replicating this success in the multimodal domain. This has led to a predominant focus on advancing text-based Chain-of-Thought (CoT) multimodal reasoning to enhance multimodal mathematical and scientific reasoning. Within this paradigm, dense visual information often serves merely as contextual evidence, from which the model extracts sparse information to support text-based reasoning.

**Existing Pain Points:**
1.  **Predominantly Text-Centric Paradigm:** Current post-training paradigms are predominantly text-centric, where dense visual inputs are only leveraged to extract sparse cues for text-based reasoning. Consequently, a deep, fine-grained understanding of the visual signal itself has been considerably undervalued.
2.  **Dependency on Additional Generative Components:** Some recent studies have shown that explicitly incorporating visual reconstruction objectives during the training of MLLMs can improve visual understanding. However, such approaches necessitate the integration of additional visual generation components and learning objectives onto the existing understanding-based MLLM architectures.
3.  **Sub-optimal Pixel-Level Reconstruction:** It remains an open question whether forcing models to achieve pixel-level reconstruction is the optimal strategy for enhancing MLLMs’ visual understanding.
4.  **Limited Applicability of Existing Jigsaw Methods:** While jigsaw-style tasks have emerged as lightweight yet effective paradigms in self-supervised learning (e.g., reordering shuffled image patches, recovering video frame order), they have generally shown weaker performance compared to more dominant approaches (like reconstruction-based or discriminative methods) and thus have not become mainstream in vision representation learning.
5.  **Inadequacy of Simple Relative Positioning:** The most closely related work, Jigsaw-R1, struggles even on simple 2×2 image jigsaws, thus focusing mainly on predicting the relative position of a pair of patches, which limits the complexity and systematic enhancement of MLLM perception.

**2. Core Research Motivation and Scientific Questions**

**Core Research Motivation:**
The core motivation is to strengthen the intrinsic visual perception and understanding capabilities of MLLMs in a lightweight, self-supervised manner during the post-training phase, without altering the model's architecture or output format. There is a need to shift the focus from text-centric reasoning back to vision-centric understanding, leveraging the structural ordering signals inherent in visual data. Jigsaw-style tasks, viewed as simpler versions of reconstruction/generation tasks, can offer effective self-supervised signals without requiring pixel-level fidelity, making them suitable for understanding-based MLLMs optimized for textual outputs.

**Scientific Questions:**
1.  Can we enhance an MLLM’s visual understanding without altering its architecture or output format (i.e., without introducing additional visual generative designs)?
2.  Can self-supervised vision-centric tasks, specifically structural ordering jigsaw tasks, be effectively formulated to fit the RLVR framework with deterministic ground-truths?
3.  Does post-training with RL on visual jigsaw tasks enable the model to better transfer the vision-centric skills acquired from the jigsaw task to downstream applications compared to Supervised Fine-Tuning (SFT)?

**3. Overall Core Idea and Design Philosophy**

**Overall Core Idea:**
The paper introduces "Visual Jigsaw," a generic self-supervised post-training framework designed to strengthen visual understanding in MLLMs. Visual Jigsaw is formulated as a general ordering task: visual inputs are partitioned, shuffled, and the model must reconstruct the visual information by producing the correct permutation in natural language. This task naturally aligns with RLVR, requires no additional visual generative components, and derives its supervisory signal automatically without any annotations. The framework is instantiated across three visual modalities: images, videos, and 3D data.

**Design Philosophy:**
1.  **Lightweight and Verifiable:** The task is formulated as a lightweight ordering problem that requires no pixel-level reconstruction. The ground-truth is a list of indices that is directly verifiable, fitting seamlessly into the RLVR framework.
2.  **Self-Supervised Post-Training:** The task is positioned in the post-training phase because solving it requires the model to already possess a foundational level of visual understanding. Post-training with RL offers stronger generalization than SFT.
3.  **Natural Language Output:** The model outputs the permutation order using natural language, integrating seamlessly with existing text-only MLLMs.
4.  **Graded Reward Design:** A partial accuracy reward is designed with a discount factor to provide learning signals for incomplete solutions and prevent reward hacking, rather than using sparse binary feedback.

**4. Core Innovation Points (List all innovations in the original text completely)**

1.  **Introduction of Visual Jigsaw as a Lightweight Post-Training Task:** Visual Jigsaw is introduced as a lightweight and verifiable self-supervised post-training task that enhances vision-centric perception and understanding capabilities in MLLMs. It requires no additional generative modules and integrates seamlessly with existing text-only models.
2.  **Instantiation across Three Visual Modalities:** The Visual Jigsaw framework is instantiated across three visual modalities—images, videos, and 3D data—demonstrating consistent improvements in fine-grained perception, temporal understanding, and 3D spatial reasoning, thereby establishing its generality and effectiveness.
3.  **Graded Reward Function for RLVR:** A specific reward design is introduced that includes a partial accuracy reward scaled by a discount factor $\gamma$ for valid but partially correct permutations, alongside a format reward, which is crucial for bootstrapping learning in complex jigsaw configurations where sparse binary feedback is insufficient.
4.  **Practical 3D Jigsaw Formulation:** A practical variant of the 3D jigsaw based on RGB-D images is designed, where points with distinct depth values are sampled and the model recovers their depth order from nearest to farthest, augmenting 3D perceptual capabilities without requiring native 3D data structures.
5.  **Highlighting Vision-Centric Self-Supervised Potential:** The work highlights the potential of self-supervised tasks focused explicitly on the visual signal as a promising, complementary direction for enhancing the vision-centric abilities of MLLMs.

**5. Overview of the Overall Technical Solution**

The overall technical solution involves a post-training phase using Group Relative Policy Optimization (GRPO) on visual jigsaw tasks. The process is as follows:
1.  **Data Preparation:** For a given visual modality (image, video, or 3D), derive a set of $K$ jigsaw elements by applying a modality-specific partitioning rule (splitting an image into patches, segmenting a video into clips, or sampling points in a 3D scene).
2.  **Shuffling:** Apply a random permutation $\pi$ to shuffle these elements.
3.  **Model Prediction:** Present the shuffled elements to the MLLM, which must predict the original structural arrangement by generating a permutation of size $K$ as a list of indices.
4.  **Reward Calculation:** Compare the predicted permutation against the ground-truth permutation using a graded reward function. Assign a format reward if the output follows the specific thinking and answer tag structure.
5.  **Optimization:** Optimize the model using the GRPO algorithm, removing both the KL regularization and the entropy loss.

**Implementation Details:**
*   **Base Model:** Qwen2.5-VL-7B-Instruct.
*   **Algorithm:** GRPO (KL regularization and entropy loss removed).
*   **Discount Factor ($\gamma$):** 0.2 for partially correct predictions.
*   **Batch Size:** 256 for image jigsaw, 128 for video & 3D jigsaw.
*   **Learning Rate:** $1 \times 10^{-6}$.
*   **Sampling:** 16 responses per prompt with a decoding temperature of 1.0.
*   **Training Steps:** 1000 steps for image and video jigsaw, 800 steps for 3D jigsaw.

**6. Detailed Module Design (If any, complete mechanisms of each layer/sub-module)**

**6.1. Reward Design Module**
The model is required to put its thinking process within `<think ></think >` and the final answer within `<answer></answer>`. A format reward of 0.2 is assigned to outputs following the correct format, while outputs with an incorrect format receive 0 values for both format and accuracy rewards. The accuracy reward function is graded to handle partial correctness and avoid reward hacking (e.g., predicting the same index for all positions).

**6.2. Image Jigsaw Module**
Given an input image $I \in \mathbb{R}^{H \times W \times 3}$, the module partitions it into a grid of $m \times n$ non-overlapping patches. Each patch is of size $\frac{H}{m} \times \frac{W}{n}$. This produces $K = m \times n$ patches arranged in raster order (row-major, top-left to bottom-right). The patches are then shuffled using a random permutation $\pi$. The model's objective is to recover the original arrangement by predicting the correct permutation of the input patch indices.
*   **Training Data:** 118K images from the COCO dataset.
*   **Grid Size:** $m = n = 3$ (yielding 9 patches per image).
*   **Filtering:** Images with side lengths smaller than 84 pixels are filtered out.

**6.3. Video Jigsaw Module**
Given a video $V \in \mathbb{R}^{T \times H \times W \times 3}$ with $T$ frames, the module segments it uniformly along the temporal axis into $K$ non-overlapping clips. Each clip contains $\frac{T}{K}$ consecutive frames. The clips are shuffled using a random permutation $\pi$. The model’s objective is to restore the original chronological order by predicting the correct permutation.
*   **Training Data:** 100K videos from the LLaVA-Video dataset.
*   **Clip Count:** $K = 6$.
*   **Trimming:** To prevent exploiting simple frame-matching cues at clip boundaries, 5% of frames are trimmed from both the beginning and end of each clip.
*   **Constraints:** Maximum 12 frames per clip, maximum resolution $128 \times 28 \times 28$ pixels per frame. Videos shorter than 24 seconds are removed.

**6.4. 3D Jigsaw Module**
Given an RGB-D image, the module randomly selects $K$ points with distinct depth values. These points form a sequence sorted by depth from nearest to farthest. A random permutation $\pi$ is applied to shuffle the sequence. Each point is annotated with its index in the shuffled sequence on the RGB image. The model is tasked with recovering the correct depth order.
*   **Training Data:** RGB-D data from ScanNet, generating 300K training samples.
*   **Point Count:** $K = 6$.
*   **Constraints:** Points lie within 0.1 m to 10 m. Any two points must be separated by at least 40 pixels in the image and differ in depth by more than 0.2 m.

**6.5. Alternative 3D Jigsaw Variants (Explored in Appendix)**
1.  **View–Motion Matching:** Given a scene from multiple camera poses, select one anchor view and sample candidate views. Construct a natural language description of the ego-motion from the anchor to each candidate. The model matches candidate views to ego-motion descriptions.
2.  **BEV–Pose Matching:** Render a bird’s-eye-view (BEV) image and select candidate views with different camera poses annotated on the BEV image. The model matches camera poses to candidate views.
*(Note: These variants did not yield significant improvements compared to the Depth Ordering formulation).*

**7. All Mathematical Formulas and Symbol Definitions (If any, replicate exactly as in the original text)**

**7.1. Image Jigsaw Formulation**
Given input image $I \in \mathbb{R}^{H \times W \times 3}$, partition into grid of $m \times n$ non-overlapping patches of size $\frac{H}{m} \times \frac{W}{n}$:
$$P = [p_1, p_2, \ldots, p_K], \quad p_i \in \mathbb{R}^{\frac{H}{m} \times \frac{W}{n} \times 3}$$
Apply random permutation $\pi : \{1, 2, \ldots, K\} \rightarrow \{1, 2, \ldots, K\}$ mapping original position index $i$ to shuffled position $\pi(i)$. Shuffled sequence:
$$P_\pi = [p_{\pi^{-1}(1)}, p_{\pi^{-1}(2)}, \ldots, p_{\pi^{-1}(K)}]$$
Objective: Predict correct permutation of input patch indices $[\pi(1), \pi(2), \ldots, \pi(K)]$.

**7.2. Video Jigsaw Formulation**
Given video $V \in \mathbb{R}^{T \times H \times W \times 3}$, segment into $K$ non-overlapping clips of $\frac{T}{K}$ consecutive frames:
$$V = [v_1, v_2, \ldots, v_K], \quad v_i \in \mathbb{R}^{\frac{T}{K} \times H \times W \times 3}$$
Apply random permutation $\pi$. Shuffled sequence:
$$V_\pi = [v_{\pi^{-1}(1)}, v_{\pi^{-1}(2)}, \ldots, v_{\pi^{-1}(K)}]$$
Objective: Predict correct chronological order permutation $[\pi(1), \pi(2), \ldots, \pi(K)]$.

**7.3. 3D Jigsaw Formulation**
Select $K$ points with distinct depth values, sorted by depth from nearest to farthest:
$$P = [p_1, p_2, \ldots, p_K], \quad d_{p_1} < d_{p_2} < \cdots < d_{p_K}$$
where $d_{p_i}$ is the depth of point $p_i$.
Apply random permutation $\pi$. Shuffled sequence:
$$P_\pi = [p_{\pi^{-1}(1)}, p_{\pi^{-1}(2)}, \ldots, p_{\pi^{-1}(K)}]$$
Objective: Recover correct depth order by predicting permutation $[\pi(1), \pi(2), \ldots, \pi(K)]$.

**7.4. Reward Function**
$$Reward(o, g) = \begin{cases} 1, & \text{if } o = g \\ \gamma \cdot \frac{1}{K}\sum_{i=1}^K 1[o_i = g_i], & \text{if ValidPermutation}(o) \land o \neq g \\ 0, & \text{otherwise} \end{cases}$$
Where:
*   $o$: Model's predicted permutation
*   $g$: Ground-truth permutation
*   $K$: Number of jigsaw pieces
*   $\gamma$: Discount factor for partial correctness
*   $\text{ValidPermutation}(o)$: Indicator of whether $o$ is a valid permutation of size $K$

**8. Algorithm Pseudocode (If any, paste the pseudocode from the paper exactly as it is)**

The paper does not provide specific algorithm pseudocode blocks. The algorithmic flow is described logically in Section 3.1 as follows:

1.  **Input:** Visual data $X$ (Image, Video, or RGB-D image).
2.  **Partitioning:** Derive a set of $K$ jigsaw elements $P = [p_1, \dots, p_K]$ by applying a modality-specific partitioning rule (spatial patches, temporal clips, or depth-sorted points).
3.  **Shuffling:** Apply a random permutation $\pi$ to obtain the shuffled sequence $P_\pi = [p_{\pi^{-1}(1)}, \dots, p_{\pi^{-1}(K)}]$.
4.  **Prompting:** Present $P_\pi$ to the MLLM with instructions to reconstruct the original structural arrangement.
5.  **Output:** Model generates output $o$ containing a thinking process in `<think >` tags and a final answer (predicted permutation) in `<answer>` tags.
6.  **Reward Calculation:**
    *   If format is incorrect: Reward = 0.
    *   If format is correct: Format Reward = 0.2.
    *   Calculate Accuracy Reward using $Reward(o, g)$ formula:
        *   If $o = g$, Reward = 1.
        *   If $o \neq g$ and $o$ is a valid permutation, Reward = $\gamma \cdot \frac{1}{K}\sum_{i=1}^K 1[o_i = g_i]$.
        *   If $o$ is not a valid permutation, Reward = 0.
    *   Total Reward = Format Reward + Accuracy Reward.
7.  **Optimization:** Update policy model using GRPO based on the calculated rewards.