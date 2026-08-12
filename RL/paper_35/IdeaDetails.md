# 1. Research Background and Existing Pain Points

**Research Background:**
The image captioning task, which generates a natural language description for a given image, bridges the gap between the visual and linguistic worlds. The captioning capability is fundamental to various applications, including vision-language models like CLIP, which learn a shared embedding space for images and text. Furthermore, captions are often a core component in the pre-training stage of Large Vision-Language Models (LVLMs), where the model learns to align visual information with linguistic descriptions on a massive scale before being fine-tuned for other downstream tasks. Given the importance of image captioning, there is a strong need for captioning models that can provide dense and accurate descriptions.

**Existing Pain Points:**
1.  **Limitations of Supervised Fine-Tuning (SFT):** Most modern captioning models are trained based on LVLMs using Supervised Fine-Tuning (SFT). While effective, SFT requires large datasets annotated by humans or proprietary models, which are expensive and not scalable. Furthermore, image captioning is an inherently open-ended problem, where a single image can be accurately described by a wide variety of captions. Since SFT models are trained to match a single ground-truth description for each image, they tend to memorize specific answers rather than learning the underlying concepts. As a result, the SFT models become less general and struggle to generate the diverse range of valid captions possible for a single image.
2.  **Challenges in Applying RLVR to Open-Ended Tasks:** The limitations of SFT have led to a recent paradigm shift toward Reinforcement Learning with Verifiable Rewards (RLVR). RLVR is the paradigm that trains models by providing clear and objective reward from the verifier, such as a binary signal of correctness for mathematical reasoning. Unlike SFT, which teaches a model to mimic a single ground-truth response, RLVR encourages the model to generate more diverse and robust outputs that meet the verifiable criteria. However, applying RLVR to open-ended tasks like image captioning is challenging, primarily due to the difficulty of designing an objective reward function. A good caption can be subjective, with multiple valid descriptions possible for the same image.
3.  **Vulnerability of Subjective Rewards:** Using reward models or LLM-as-a-judge to provide feedback is vulnerable to reward hacking. The captioning model learns to exploit weaknesses in the reward models (e.g., verbosity or brevity outputs) rather than producing a high-quality response. It is difficult to create effective rubrics or evaluation prompts for LVLM-as-a-judge methods because captions are free-form and encode substantial information. Using reference answers as rewards like ROUGE and BLEU is constrained when evaluating complex and long-form captions. As shown in the paper, existing reward models suffer from limitations like rewarding verbosity or brevity, leading to low-quality captions and reward hacking (e.g., training collapse or outputting generic terms like "description").

# 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:**
To overcome the limitation of SFT, the research aims to apply the Reinforcement Learning with Verifiable Rewards (RLVR) paradigm to the open-ended task of image captioning. The objective is to design a powerful and scalable RLVR training paradigm for the image captioning task to generate more creative and more general variety of accurate descriptions, moving beyond the limitations of traditional SFT-based image captioning models which memorize specific ground-truth answers.

**Scientific Questions:**
1.  How to design an objective reward function for the inherently subjective nature of what constitutes a "good" caption in an open-ended task?
2.  How to avoid reward hacking (such as models exploiting verbosity or brevity biases) that plagues existing subjective reward models or LLM-as-a-judge approaches in the context of image captioning?
3.  How can a caption's quality be redefined through its utility in a way that provides a clear, verifiable, and objective reward signal for reinforcement learning training?

# 3. Overall Core Idea and Design Philosophy

**Overall Core Idea:**
The paper introduces a novel perspective where a caption’s quality is proportional to its utility: a high-quality caption should enable a non-visual language model to accurately answer questions about the corresponding image. When the image caption is detailed and accurate, a text-based LLM that can’t directly “see” the image can still answer Visual Question Answering (VQA) questions about the image.

**Design Philosophy:**
1.  **Decoupled Perception-Reasoning:** Redefine caption quality through a decoupled two-stage pipeline. The first stage uses LVLMs to generate rich and accurate captions (Perception). The second stage evaluates caption quality by using a vision-free LLM to perform the QA task (Reasoning).
2.  **Verifiable Rewards:** Replace subjective scores with objective accuracy. The reward is derived from the accuracy of a separate, vision-free LLM answering Multiple-Choice Questions (MCQs) based solely on that caption. Since LLMs exhibit high stability in answering multiple-choice questions, and the evaluation of their responses only requires exact matching, the accuracy of the LLM’s responses can therefore serve as a reliable indicator of caption quality.
3.  **Sparse Supervision:** The design philosophy acknowledges that sparse QA supervision is sufficient. Even with only a single QA pair per image, the framework achieves significant improvement, highlighting efficiency.

# 4. Core Innovation Points

1.  **First Application of RLVR to Image Captioning:** The paper contributes the first study of applying Reinforcement Learning with Verifiable Rewards for the open-ended and subjective image captioning task. Unlike traditional Supervised Fine-Tuning, which can lead to models memorizing a limited set of annotated captions, this method allows the model to explore and generate a broader range of creative and general descriptions.
2.  **Novel Decoupled Two-Stage Pipeline (CapRL):** The paper presents CapRL, a new training paradigm featuring a decoupled two-stage pipeline. The initial stage uses LVLMs to generate rich and accurate captions. Subsequently, the second stage evaluates caption quality by using a vision-free LLM to perform the QA task. This redefines caption quality based on its utility for enabling a non-visual LLM to answer questions.
3.  **Specific QA Curation Pipeline with Leakage Prevention:** A specific QA curation pipeline is developed to ensure the quality of the questions and answers used for the second stage. This includes a stringent QA filtering stage to prevent information leakage, verifying that all questions are strictly visually-grounded and answerable exclusively through analysis of the image content rather than external knowledge or cues within the question itself.
4.  **Construction of CapRL-5M Dataset:** The paper constructs the CapRL-5M dataset annotated by the powerful CapRL-3B model. This dataset demonstrates strong scaling properties for multimodal pretraining, enabling the construction of high-quality, scalable datasets at very low annotation cost.
5.  **Objective Reward Design against Hacking:** The reward design eliminates potential bias in the LLM’s preference for specific answer choices by randomly shuffling the options and averaging the reward over sampled questions, which successfully mitigates reward hacking observed in subjective reward models.

# 5. Overview of the Overall Technical Solution

The overall technical solution is the Captioning Reinforcement Learning (CapRL) framework, which employs a decoupled two-stage process for RLVR training. 

**Pipeline Overview:**
1.  **Input:** An image and an instruction.
2.  **Stage 1 (Caption Generation):** The policy model (an LVLM, specifically Qwen2.5-VL-3B) generates a set of candidate captions for the input image.
3.  **Stage 2 (VQA Verification):** Each caption is paired with curated Multiple-Choice Questions (MCQs) related to the image and passed to a Large Language Model (LLM, specifically Qwen2.5-3B-Instruct) for answering. Since the LLM does not have access to the image directly, its ability to answer correctly depends entirely on how comprehensive and accurate the caption is.
4.  **Reward Computation:** The reward for a caption is determined by the accuracy of the LLM’s answers. The accuracy is computed using a simple exact-match criterion. Options are shuffled and questions are sampled N times to ensure stability and remove bias.
5.  **Optimization:** The policy model is updated via policy gradient optimization (GRPO), incorporating a KL-divergence penalty for training stability.

**Data Preparation:**
To train CapRL effectively, a high-quality VQA dataset composed exclusively of multiple-choice questions is constructed using a structured three-stage curation pipeline: (1) Image Collection (diverse sources), (2) QA Generation (using Qwen2.5-VL-72B), and (3) QA Filtering (using Qwen2.5-VL-3B to prevent leakage).

**Dataset Generation:**
Using the CapRL training scheme, CapRL-3B is obtained. This model is then used to annotate 5M images (sourced from ShareGPT4V-1M, DenseFusion-1M, and filtered web images), forming the CapRL-5M dataset for LVLM pretraining.

# 6. Detailed Module Design

**Module 1: Policy Model ($M_V$)**
*   **Function:** Generates candidate captions based on the input image and instruction.
*   **Mechanism:** During the GRPO training process, an image and an instruction are first provided as input to the policy model to sample a set of candidate captions $\{c_1, c_2, \ldots, c_G\}$.
*   **Architecture:** Initialized with Qwen2.5-VL-3B.

**Module 2: LLM Answering Module ($M_L$)**
*   **Function:** Evaluates the quality of the generated caption by answering questions based solely on the text of the caption.
*   **Mechanism:** Receives a caption $c_i$ and a question $q_m$ related to the image. Since it does not have access to the image, its ability to answer correctly depends entirely on the comprehensiveness and accuracy of the caption. It outputs an answer $a_m$.
*   **Architecture:** Qwen2.5-3B-Instruct is used as $M_L$ by default to ensure high training efficiency.

**Module 3: Reward Computation**
*   **Function:** Computes the objective verifiable reward for the policy model based on the LLM's accuracy.
*   **Mechanism:** 
    1.  **Single Question Reward:** Computed using a simple exact-match criterion. If the LLM's answer matches the ground truth, the reward is 1; otherwise, 0.
    2.  **Bias Elimination:** To eliminate potential bias in the LLM’s preference for specific answer choices, the options are randomly shuffled each time a question is presented.
    3.  **Robustness Assurance:** To ensure the stability of caption scoring, $N$ questions are sampled from all questions related to the image and the LLM answers them independently. The final reward is the average accuracy over these $N$ sampled questions.
*   **Integration:** The reward is used to calculate the mean and variance of rewards across the group to derive the advantage for each caption.

**Module 4: QA Curation Pipeline**
*   **Function:** Constructs the high-quality VQA dataset required to provide reliable reward signals.
*   **Mechanism:**
    1.  **Image Collection:** Sourcing diverse images from the web and existing open-source datasets, including natural scenes, charts, and documents.
    2.  **QA Generation:** For each image, using Qwen2.5-VL-72B to automatically generate five question-answer pairs. The questions should be challenging and focus on the image content.
    3.  **QA Filtering:** A stringent QA filtering process to prevent information leakage. The filter ensures that each selected QA pair requires the image context to be answered correctly. This is implemented by checking if an LVLM can answer the question *with* the image but *cannot* answer it *without* the image. Qwen2.5-VL-3B is used for this filtering step.
    4.  **Result:** Approximately 75k images along with their corresponding QA pairs are retained.

**Module 5: CapRL-5M Dataset Construction**
*   **Function:** Annotate a large-scale dataset for LVLM pretraining.
*   **Mechanism:**
    1.  **Image Collection:** Incorporate all images from ShareGPT4V-1M and DenseFusion-1M. Gather additional images from the web (natural photographs, documents, charts, user interfaces).
    2.  **Image Processing:** Apply rigorous filtering (inspired by SemDeDup) to remove images with redundant semantics, discard low-resolution/overly simple images, filter out violence/pornography, avoid benchmark leakage by clustering and eliminating similar images, and conduct human safety inspection. Retain 3M high-quality web images, totaling 5M images with the open-source datasets.
    3.  **Caption Annotation:** Use CapRL-3B (the policy model trained with CapRL) as the captioner to annotate the 5M images.

**Module 6: GRPO Optimization Module**
*   **Function:** Update the policy model using the computed rewards.
*   **Mechanism:** Calculate the mean and variance of rewards across the group to derive the advantage for each caption. Incorporate a KL-divergence penalty against a reference model to ensure training stability. Update the policy model via policy gradient optimization.

# 7. All Mathematical Formulas and Symbol Definitions

**Formula 1: LLM Answering Process**
$$ a_m = M_L(c_i, q_m) $$
Where:
*   $q_m$: the $m$-th question associated with current image $I$.
*   $c_i$: the $i$-th caption generated by the policy model.
*   $M_L$: the Large Language Model used for answering.
*   $a_m$: the LLM’s answer to that question.

**Formula 2: Single Question Reward**
$$ r(a_m) = \begin{cases} 1, & \text{if } a_m = GT_m \\ 0, & \text{otherwise} \end{cases} $$
Where:
*   $r(a_m)$: the reward for a single question.
*   $GT_m$: the ground-truth answer to question $q_m$.

**Formula 3: Final Caption Reward**
$$ R_{c_i} = \frac{1}{N} \sum_{k=1}^N r\left(M_L(c_i, \text{Shuffle}(q_{m_k}))\right), \quad m_k \sim \{1, \ldots, M\} $$
Where:
*   $R_{c_i}$: the final reward for caption $c_i$.
*   $N$: the number of sampling times from all the questions related to the image.
*   $M$: the number of questions associated with the current image $I$.
*   $\text{Shuffle}(q_{m_k})$: the operation of randomly shuffling the options of the question $q_{m_k}$ to eliminate potential bias.

**Formula 4: QA Filtering Set Definition**
$$ Q = \{(q, a) \in D \mid M_V^f(q, I) = a \land M_V^f(q) \neq a\} $$
Where:
*   $Q$: the filtered set of QA pairs.
*   $D$: the initial generated dataset of QA pairs.
*   $(q, a)$: a question-answer pair.
*   $I$: the corresponding input image.
*   $M_V^f$: the LVLM used in QA Filtering (Qwen2.5-VL-3B).
*   $M_V^f(q, I)$: the answer generated when conditioned on both the question $q$ and the image $I$.
*   $M_V^f(q)$: the answer generated when the image is omitted.

# 8. Algorithm Pseudocode

Based on the methodology described in the paper, the algorithm process is as follows:

**Algorithm 1: CapRL Training Process**

**Input:** Policy Model $M_V$, LLM Verifier $M_L$, Image Dataset with QA pairs $\{(I, \{(q_m, GT_m)\}_{m=1}^M)\}$, Number of samples $N$, KL penalty coefficient $\beta$.

1:  **Initialize** Policy Model parameters $\theta$ (from pretrained LVLM), Reference Model parameters $\theta_{ref}$.
2:  **for** each training iteration **do**
3:  $\quad$ Sample a batch of images $I$.
4:  $\quad$ **for** each image $I$ **do**
5:  $\quad\quad$ Generate a group of captions $\{c_1, c_2, \ldots, c_G\}$ using $M_V(I)$.
6:  $\quad\quad$ **for** each caption $c_i$ in the group **do**
7:  $\quad\quad\quad$ Initialize total reward $R_{c_i} = 0$.
8:  $\quad\quad\quad$ **for** $k = 1$ to $N$ **do**
9:  $\quad\quad\quad\quad$ Sample a question index $m_k \sim \{1, \ldots, M\}$.
10: $\quad\quad\quad\quad$ Shuffle the options of question $q_{m_k}$ to get $q_{m_k}^{shuffled}$.
11: $\quad\quad\quad\quad$ Generate answer $a_{m_k} = M_L(c_i, q_{m_k}^{shuffled})$.
12: $\quad\quad\quad\quad$ Compute reward $r(a_{m_k})$ using Exact Match:
13: $\quad\quad\quad\quad$ **if** $a_{m_k} == GT_{m_k}$ **then** $r(a_{m_k}) = 1$ **else** $r(a_{m_k}) = 0$.
14: $\quad\quad\quad\quad$ $R_{c_i} \leftarrow R_{c_i} + r(a_{m_k})$
15: $\quad\quad\quad$ **end for**
16: $\quad\quad\quad$ $R_{c_i} \leftarrow R_{c_i} / N$  // Average reward over N samples
17: $\quad\quad$ **end for**
18: $\quad\quad$ Compute advantage $A_i$ for each caption $c_i$ based on group rewards $\{R_{c_1}, \ldots, R_{c_G}\}$ (mean and variance normalization).
19: $\quad$ **end for**
20: $\quad$ Compute Policy Gradient Loss $\mathcal{L}_{PG}$ based on advantages $A_i$ and log-probabilities.
21: $\quad$ Compute KL Divergence Penalty $\mathcal{L}_{KL}$ between $M_V$ and Reference Model.
22: $\quad$ Update Policy Model parameters $\theta$ using $\nabla(\mathcal{L}_{PG} + \beta \mathcal{L}_{KL})$.
23: **end for**
24: **Return** Trained Policy Model $M_V$ (CapRL-3B).

**Algorithm 2: QA Curation Pipeline**

**Input:** Image Collection, LVLM Generator $M_{Gen}$ (Qwen2.5-VL-72B), LVLM Filter $M_V^f$ (Qwen2.5-VL-3B).

1:  **Initialize** Filtered QA Dataset $Q = \emptyset$.
2:  **for** each Image $I$ in Image Collection **do**
3:  $\quad$ Generate a set of QA pairs $D_I = \{(q_1, a_1), \ldots, (q_K, a_K)\}$ using $M_{Gen}(I)$.
4:  $\quad$ **for** each QA pair $(q, a)$ in $D_I$ **do**
5:  $\quad\quad$ Compute answer with image: $a_{with} = M_V^f(q, I)$.
6:  $\quad\quad$ Compute answer without image: $a_{without} = M_V^f(q)$.
7:  $\quad\quad$ **if** $a_{with} == a$ AND $a_{without} \neq a$ **then**
8:  $\quad\quad\quad$ Add $(q, a)$ to $Q$.
9:  $\quad\quad$ **end if**
10: $\quad$ **end for**
11: **end for**
12: **Return** Filtered QA Dataset $Q$.