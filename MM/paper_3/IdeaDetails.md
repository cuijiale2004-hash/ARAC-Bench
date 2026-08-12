**1. Research Background and Existing Pain Points**

**Research Background:** Object counting is a fundamental task in computer vision with broad applicability in many real-world scenarios. It refers to the task of estimating the number of target instances within a given image. Conventional methods mainly estimate a density map of an image where the integral over the density map gives the total number of objects. The density map is obtained by convolving Gaussian kernels with point annotations per object in the image. Recently, class-agnostic counting methods have been proposed to count objects of arbitrary categories within images. A common practice is to provide visual exemplars to indicate the target category. Since manually selecting exemplars requires extensive labeling cost, an emerging way is to replace them with text prompts that use class names to specify target object categories. Benefiting from large-scale pretraining on vision-language pairs, recent advances in Multimodal Large Language Models (MLLMs) present a new opportunity for class-agnostic object counting in a text-promptable way, where text prompts specify target categories of interest and are then converted to proper instructions. Combined with corresponding images, these multimodal instructions are fed into the MLLM to auto-regressively predict object counts.

**Existing Pain Points:** 
1. Fully-supervised counting methods require costly point-level annotations per object. Acquiring point-level annotations is labor-intensive and time-consuming, especially in dense scenes where hundreds or thousands of objects co-occur and even occlude each other.
2. Existing weakly-supervised counting methods leverage only image-level object counts as supervision, but they are still in early stage and often limited to a single object category, i.e. person. They are inherently restricted to certain object category, limiting their applications.
3. The basic object counting pipeline based on the MLLM without any fine-tuning (i.e. MLLM-Zero) produces reasonable estimates in sparse scenarios yet its performance degrades significantly in dense scenes. This is because MLLMs are mostly seen with sparse rather than dense object distributions in their pre-training corpora.
4. The naive baseline (WS-COC-Base) that directly fine-tunes the MLLM to predict the object count from the given image leads to clear underestimation. Learning a direct mapping is difficult due to the absence of object distribution supervision. Furthermore, directly optimizing the predicted count against the ground truth can still be challenging due to the modality gap between vision and text, making it difficult for the model to establish a robust mapping from high-dimensional visual features to a discrete scalar count expressed as a text token.

**2. Core Research Motivation and Scientific Questions**

**Core Research Motivation:** Given the underlying counting ability of MLLMs in sparse scenarios, this study aims to explore how to extend such potentials to accurate class-agnostic object counting with minimal fine-tuning cost. The goal is to alleviate the high annotation cost of point-level supervision by leveraging only image-level object counts, while overcoming the limitations of existing weakly-supervised methods that are restricted to class-specific counting. By bootstrapping MLLMs with simple yet effective strategies, the research seeks to bridge the modality gap and mitigate the counting bias in dense scenes, ultimately achieving performance comparable to state-of-the-art fully-supervised methods.

**Scientific Questions:** How to effectively bootstrap MLLMs for weakly-supervised class-agnostic object counting using only image-level count supervision? How to address the modality gap between high-dimensional visual features and discrete text tokens that makes direct count regression difficult? How to mitigate the significant counting bias and underestimation in dense scenes where MLLMs inherently struggle due to their pre-training data distribution?

**3. Overall Core Idea and Design Philosophy**

**Overall Core Idea:** The paper introduces WS-COC, the first MLLM-driven weakly-supervised framework for class-agnostic object counting, which bootstraps MLLMs for object counting using only image-level count supervision. Instead of directly fine-tuning MLLMs to predict object counts, the framework incorporates three simple yet effective strategies to bootstrap the counting paradigm in both training and testing.

**Design Philosophy:** The design philosophy is rooted in reformulating the challenging direct count regression task into more accessible sub-problems for MLLMs. Instead of regressing the exact count in one shot, the task is reformulated as a series of range judgment tasks to enable curriculum learning from easy to hard. Instead of directly mapping visual features to absolute numbers across a modality gap, the model is guided to judge relative count differences between images, which is more visually probed. Finally, to counter the inherent bias of MLLMs in dense scenes, the inference process is enhanced by combining global context with aggregated local details, leveraging the complementary nature of global underestimation and local overestimation.

**4. Core Innovation Points**

1. **First MLLM-driven weakly-supervised framework for class-agnostic object counting:** The paper proposes WS-COC, which novelly leverages MLLM to achieve class-agnostic object counting using only image-level object counts as supervision, significantly reducing annotation costs compared to fully-supervised methods and expanding beyond the single-category limitation of prior weakly-supervised approaches.

2. **Divide-and-discern dialogue tuning (D3T) strategy:** A strategy that reformulates count prediction as a series of range judgment tasks. It guides the MLLM to iteratively determine whether the count falls within specific ranges through multi-round dialogue, progressively breaking down the range from coarse to fine. This multi-step reasoning helps the MLLM learn to count from easy to hard in a curriculum manner.

3. **Compare-and-rank count optimization (CRCO) strategy:** A strategy that addresses the modality gap by training the MLLM to optimize the relative ranking of multiple images according to their object counts. Guiding the MLLM to judge the relative count difference between images is intuitively more accessible than directly regressing absolute counts, thereby establishing a more robust mapping.

4. **Global-and-local counting enhancement (GLCE) strategy:** An inference strategy that aggregates and fuses local and global count predictions to improve counting performance in dense scenes. It mitigates the counting bias in dense crowds by partitioning the input image into smaller sub-images to obtain local counts, which are then averaged with the global count to leverage their complementary error characteristics.

**5. Overview of the Overall Technical Solution**

The overall technical solution of WS-COC is built upon a baseline model that fine-tunes an MLLM (specifically LLaVA-OneVision-7B) using a LoRA-based finetuning paradigm. The baseline constructs text instructions and ground truth responses using fixed templates to predict the global object count. Building upon this baseline, WS-COC introduces three strategies:
During training, the **Divide-and-discern dialogue tuning (D3T)** strategy is applied to guide the model through multi-round dialogue to progressively narrow down the count range, terminating when the range is small enough to predict the actual count. Concurrently during training, the **Compare-and-rank count optimization (CRCO)** strategy clusters images by count intervals, samples multiple images to form an inherently ranked set, shuffles them, and trains the MLLM to predict their relative ranking using a language modeling loss. 
During inference, the **Global-and-local counting enhancement (GLCE)** strategy is activated. It first predicts a global count for the image. If the count exceeds a threshold indicating a dense scene, the image is partitioned into sub-images for independent local counting. The local counts are summed and averaged with the global count to produce the final enhanced prediction.

**6. Detailed Module Design**

**Baseline Model (WS-COC-Base):** 
The baseline model fine-tunes an MLLM for weakly-supervised object counting. For each image $I$ and its global object count $c$, the text instruction $T_{inst}$ and the ground truth response $T_{gt}$ are constructed using fixed templates. $T_{inst}$ is formulated as “How many [obj] are there in the image?”, while $T_{gt}$ follows “a photo of [num] [obj]”, where the [num] token is replaced with $c$ and the [obj] token denotes a specific object category. The MLLM is optimized using the LoRA-based finetuning paradigm with a language modeling loss that minimizes the cross-entropy between the predicted response and the ground truth response.

**Divide-and-discern dialogue tuning (D3T):**
This module reformulates the count prediction task as a series of range judgment tasks through multi-round dialogue. 
Given an image $I$ and its global object count $c$, the initial range $[L_1, U_1]$ is set for the first round dialogue (e.g., minimal and maximum object counts $[1, 2000]$ in the FSC-147 dataset). The midpoint of this range is calculated as $\tau_1 = \lfloor \frac{L_1+U_1}{2} \rfloor$.
The text instruction $Q_1$ is constructed using the template “Are there more than $\tau_1$ [obj] in the image?”, where the [obj] token is replaced by the target object category. The ground truth response $R^g_1$ is set to “yes” if $c > \tau_1$ and “no” otherwise. The MLLM is queried using $I$ and $Q_1$ to predict the response $R_1$, which is optimized against $R^g_1$ using the language modeling loss.
The range is progressively halved and the process is repeated for multiple rounds. For the $t$-th round, the range $[L_t, U_t]$ is updated according to the ground truth response $R^g_{t-1}$ of the $(t-1)$-th round:
$[L_t, U_t] = \begin{cases} [\tau_{t-1} + 1, U_{t-1}], & \text{if } R^g_{t-1} = Yes, \\ [L_{t-1}, \tau_{t-1}], & \text{if } R^g_{t-1} = No. \end{cases}$
The dialogue terminates when $U_t - L_t$ is lower than $\delta = 0.2 \times c$. Then, the MLLM is asked to predict the actual object count.

**Compare-and-rank count optimization (CRCO):**
This module trains the model to compare multiple images and optimize their relative ranking according to object counts. 
First, to ensure both sparse and dense scenes are covered despite the long-tailed distribution, a sampling scheme clusters images according to their counts. The minimum and maximum counts (i.e., $[\underline{c}, \bar{c}]$) for each object category in the training set are obtained, and $[\underline{c}, \bar{c}]$ is partitioned into $K$ (4 by default) equal-length intervals. The images of each category are divided into four groups based on intervals in which their object counts fall. During each training iteration, one image is randomly sampled from each group to construct an image set $\mathbf{I} = \{I_1, I_2, I_3, I_4\}$, where the corresponding counts inherently satisfy $c_1 < c_2 < c_3 < c_4$. The order of images in $\mathbf{I}$ is randomly shuffled to $\tilde{\mathbf{I}}$ and treated as the visual input to the MLLM.
Next, the associated text instruction $T$ is constructed using the template “Given four images, rank them in ascending order based on their counts of [obj]”, and the ground truth response $T_g$ specifies the correct ranking in the form “Image i < · · · < Image j”, where “Image i” and “Image j” denote the $i$-th and $j$-th image in $\tilde{\mathbf{I}}$, respectively. The MLLM is queried with $T$ and $\tilde{\mathbf{I}}$ to obtain the response $T_r$, which is optimized with respect to $T_g$ using the language modeling loss.

**Global-and-local counting enhancement (GLCE):**
This module operates during the inference stage to mitigate the counting bias in dense crowds by aggregating global and local predictions. 
Specifically, the MLLM is first queried to predict the global count $c_g$ for the given image $I$. If $c_g$ is lower than the threshold $c_h$ (100 by default), $c_g$ is directly treated as the final prediction. Otherwise, the image $I$ is considered a dense scene. 
In this case, $I$ is evenly partitioned into $L \times L$ (2 × 2 by default) non-overlapping sub-images, and local predictions $\{c_k\}_{k=1}^{L^2}$ are obtained from the MLLM by respectively querying each sub-image. The local predictions are summed to obtain $c_l$. Due to the edge effect when aggregating sub-images, $c_l$ often overestimates the total count compared to $c_g$. Empirically, the final prediction is obtained by simply averaging $c_l$ and $c_g$, allowing them to complement each other.

**7. All Mathematical Formulas and Symbol Definitions**

**Symbol Definitions:**
$I$: Input image.
$c$: Global object count for image $I$.
$T_{inst}$: Text instruction for the baseline model.
$T_{gt}$: Ground truth response for the baseline model.
[num]: Token replaced with the object count $c$.
[obj]: Token denoting a specific object category.
$[L_t, U_t]$: The range for the count judgment in the $t$-th round of dialogue.
$\tau_t$: The midpoint of the range in the $t$-th round of dialogue.
$Q_t$: Text instruction for the $t$-th round of dialogue.
$R^g_t$: Ground truth response for the $t$-th round of dialogue.
$R_t$: Predicted response for the $t$-th round of dialogue.
$\delta$: Stopping threshold for the dialogue, defined as $0.2 \times c$.
$\mathbf{I}$: Image set constructed for count comparison, $\mathbf{I} = \{I_1, I_2, I_3, I_4\}$.
$c_k$: Object count for the $k$-th image in the set $\mathbf{I}$.
$\tilde{\mathbf{I}}$: Shuffled order of images in $\mathbf{I}$.
$T$: Text instruction for compare-and-rank optimization.
$T_g$: Ground truth response for compare-and-rank optimization.
$T_r$: Predicted response for compare-and-rank optimization.
$[\underline{c}, \bar{c}]$: Minimum and maximum counts for each object category in the training set.
$K$: Number of equal-length intervals for partitioning the count range (default 4).
$c_g$: Global count prediction for the input image during inference.
$c_h$: Threshold to determine if the scene is dense (default 100).
$L$: Grid size for partitioning the image into sub-images (default 2).
$c_k$: Local count prediction for the $k$-th sub-image.
$c_l$: Aggregated local count prediction, calculated as the sum of local predictions.

**Mathematical Formulas:**
1. Baseline instruction template: 
   $T_{inst} =$ “How many [obj] are there in the image?”
   $T_{gt} =$ “a photo of [num] [obj]”

2. Midpoint calculation in D3T:
   $\tau_1 = \lfloor \frac{L_1+U_1}{2} \rfloor$

3. D3T range update rule:
   $[L_t, U_t] = \begin{cases} [\tau_{t-1} + 1, U_{t-1}], & \text{if } R^g_{t-1} = Yes, \\ [L_{t-1}, \tau_{t-1}], & \text{if } R^g_{t-1} = No. \end{cases}$

4. D3T termination condition:
   $U_t - L_t < \delta$, where $\delta = 0.2 \times c$

5. D3T dialogue instruction template:
   $Q_1 =$ “Are there more than $\tau_1$ [obj] in the image?”
   $R^g_1 =$ “yes” if $c > \tau_1$ and “no” otherwise.

6. CRCO sampling constraint:
   $\mathbf{I} = \{I_1, I_2, I_3, I_4\}$, where $c_1 < c_2 < c_3 < c_4$

7. CRCO instruction template:
   $T =$ “Given four images, rank them in ascending order based on their counts of [obj]”
   $T_g =$ “Image i < · · · < Image j”

8. GLCE aggregation logic:
   $c_l = \sum_{k=1}^{L^2} c_k$
   Final prediction = average($c_l$, $c_g$)

**8. Algorithm Pseudocode**

The paper does not provide explicit algorithm pseudocode blocks. The algorithmic logic is fully described in the detailed module design section above, strictly adhering to the iterative processes, constraint conditions, and logical sequences presented in the original text.