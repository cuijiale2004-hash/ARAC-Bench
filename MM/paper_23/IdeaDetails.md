**1. Research Background and Existing Pain Points**

**Research Background:**
Human knowledge is acquired through multiple sensory experiences, with vision playing a dominant role in understanding the environment and accumulating knowledge. Inspired by this principle, recent advances in Large Language Models (LLMs) naturally extend toward Multimodal LLMs (MLLMs), specifically large vision language models, to foster visual intelligence. The standard paradigm for MLLMs involves combining independently pretrained LLMs and vision models, which enabled MLLMs to reach strong initial capabilities. The training strategy for these models has traditionally been Supervised Finetuning (SFT). However, a recent shift has occurred from SFT to Reinforcement Learning (RL) in MLLM training paradigms, mirroring the shift that RL brought to LLMs (e.g., RLHF). Recent studies demonstrate that incorporating human preference data via RL enhances MLLM performance and mitigates hallucination.

**Existing Pain Points:**
A dominant assumption in MLLM research is that its performance is largely inherited from the LLM backbone, given its immense parameter scale and remarkable capabilities. This has created a significant void in the understanding of the vision encoder, which determines "how MLLMs perceive images." The shift from SFT to RL magnifies this oversight—namely, the significant lack of analysis on how such training reshapes the vision encoder as well as the MLLM. Specifically, the field lacks a systematic comparison within MLLMs between SFT for instruction-following and RL for preference alignment, including an analysis of model scaling in common benchmarks. The lack of understanding is especially notable for the vision encoder. Research has progressed little beyond the preliminary finding that fine-tuning the vision encoder yields better outcomes than keeping it frozen. Such oversight can be attributed to an implicit, LLM-centric assumption about the source of MLLM capabilities, leaving a significant void in our understanding of how SFT and RL differ in reshaping visual representations.

**2. Core Research Motivation and Scientific Questions**

**Core Research Motivation:**
Despite the efficacy of RL in the MLLM, a comprehensive understanding of its effects compared to SFT—and critically, its influence on the vision encoder—remains largely absent from the literature. There is a need to investigate the differential impacts of SFT and RL on both MLLMs and their vision encoders. If RL improves downstream tasks, it is critical to understand if this is merely a language model phenomenon or if it fundamentally alters how the model perceives visual input.

**Scientific Questions:**
1.  How do SFT and RL (e.g., DPO) affect MLLMs on diverse vision-language (VQA) tasks?
2.  Is RL (DPO) actually superior to SFT?
3.  Does the trend of RL's superiority hold with model scaling (scaling vision encoders and language models)?
4.  How does MLLM training affect the visual representations within the vision encoder?
5.  Can an RL-trained vision encoder surpass state-of-the-art vision models for MLLMs?

**3. Overall Core Idea and Design Philosophy**

**Overall Core Idea:**
The core idea is that RL (specifically Direct Preference Optimization, DPO) produces stronger and more localized visual representations compared to SFT, boosting the ability of the vision encoder for MLLM. The paper reframes the finding that RL reshapes visual representations into a simple recipe for building strong vision encoders for MLLMs, termed Preference-Instructed Vision OpTimization (PIVOT). PIVOT positions RL training not just as a post-training step for the MLLM, but as an auxiliary training process to evolve the vision encoder itself.

**Design Philosophy:**
The design philosophy is to decouple the analysis of the vision encoder from the overwhelming dominance of the LLM backbone. By isolating the vision encoder after training the MLLM with different strategies (SFT vs. RL) and evaluating it on classic vision tasks (ImageNet classification, segmentation), the paper seeks to prove that the choice of post-training strategy fundamentally alters the "seeing" capability of the model. Furthermore, the philosophy is efficiency: PIVOT demonstrates that one can unlock the potential of existing vision models using an LLM head and RL, requiring less than 1% of the computational cost of standard vision pretraining, rather than relying on scaling up the vision model parameters.

**4. Core Innovation Points**

1.  **Systematic Comparison of SFT vs. RL in MLLMs:** The paper establishes a controlled training setup to compare SFT and DPO across various model scales. It reveals that DPO achieves superior performance compared to SFT, particularly on tasks that require deep visual comprehension rather than those primarily relying on the LLM’s knowledge (Finding 2).
2.  **In-depth Analysis of the Vision Encoder in MLLMs:** The paper conducts the first in-depth analysis of the vision encoder in MLLMs through diverse experiments (ImageNet classification, segmentation, gradient visualization). It proves that MLLM post-training rewrites visual representations, with RL driving stronger representation than SFT (Finding 3).
3.  **Gradient Visualization of Localization:** Through Grad-CAM visualization, the paper demonstrates that the gradient signals from DPO align more strongly with question-relevant regions than those from SFT. SFT signals tend to be scattered, while DPO signals are precisely focused on semantically relevant regions (Finding 4).
4.  **Preference-Instructed Vision OpTimization (PIVOT):** The paper reframes RL training as an auxiliary training process for the vision encoder. PIVOT involves training a vision encoder with an LLM head using DPO, then detaching the encoder for use in MLLMs. This allows a vision model trained with PIVOT to outperform substantially larger models and even subsequent-generation encoders with less than 1% compute cost (Finding 6).
5.  **Discovery of LLM Backward Signal Benefit:** The paper finds that the vision encoder benefits from a larger LLM, which provides more informative backward signals for visual representation within an MLLM (Finding 5).
6.  **Importance of Vision Encoder Capacity:** The paper quantifies that increasing the capacity of the vision encoder in MLLMs is particularly important for tasks requiring fine-grained visual understanding, a factor often overshadowed by LLM scaling (Finding 1).

**5. Overview of the Overall Technical Solution**

The overall technical solution is divided into three main phases:
1.  **Analysis of MLLMs under different training strategies:** The paper compares the performance of MLLMs trained with SFT versus DPO across different model scales (varying vision encoder size and language model size) on 16 VQA benchmarks.
2.  **Analysis of the Vision Encoder:** After training MLLMs with SFT or DPO, the vision encoder is detached from the LLM and evaluated on standalone vision tasks (ImageNet classification and semantic segmentation). This disentangles the impact on visual representations. Additionally, gradient visualization (Grad-CAM) is used to trace how optimization signals propagate to the vision encoder.
3.  **PIVOT Implementation and Evaluation:** The findings are consolidated into the PIVOT recipe. A vision encoder is attached to an LLM, optimized through pre-training and post-training with DPO, and then detached. This "PIVOT-enhanced encoder" is then integrated into a new MLLM setup (with a new LLM and projector) and evaluated on VQA benchmarks to assess its efficacy.

**6. Detailed Module Design**

**6.1 MLLM Architecture Module**
The MLLM architecture integrates an LLM with a vision encoder via a multimodal projector.
*   **Vision Encoder:** SigLIP2 (variants B/16, L/16, So/16, g/16) at 384px.
*   **Language Model (LLM):** Qwen2.5 (variants 0.5B, 1.5B, 3B, 7B).
*   **Multimodal Projector:** A 2-layer MLP.

**6.2 Training Procedure Module**
The MLLM development process consists of two stages:
*   **Stage 1 Pre-training:**
    1.  **Projector-only training:** Align visual and language embedding spaces using the LAION/CC/SBU-558K dataset.
    2.  **End-to-end pre-training:** Train all model parameters on diverse VL datasets (LLaVA-OneVision-3.2M), including VQA, vision-grounded dialogue, and image captioning, to establish a base MLLM.
*   **Stage 2 Post-training:**
    Full-parameter update of the base model according to SFT or DPO objectives. The post-training dataset is a 20K subset of the MMPR dataset.

**6.3 Evaluation Mechanisms Module**
*   **Linear Probing for ImageNet Classification:** Features are extracted from the visual components (vision encoder or encoder-projector combined). A logistic regression model with L2 regularization is trained on a 50k random subset of ImageNet data to measure accuracy.
*   **Segmentation Probing:** The vision encoder is frozen, and a two-layer MLP is attached and trained as a patch-level classifier on the ADE20K dataset. Evaluation uses patch-level recall.
*   **Gradient Visualization (Grad-CAM):** The loss for a specific sample is computed, and a backward pass is performed. Gradients with respect to the feature activations of the vision encoder are obtained, and the gradient magnitude of each token is measured and visualized.
*   **Representation Alignment Metric:** Adopted from Huh et al. (2024) to evaluate representation similarity between models trained on different modalities.

**6.4 PIVOT Module**
PIVOT (Preference-Instructed Vision OpTimization) is defined as a training regime where a vision model is trained with an LLM-head using DPO.
*   **Process:**
    1.  Start with a vision encoder (e.g., CLIP, SigLIP1).
    2.  Attach it to an LLM (e.g., Qwen2.5-1.5B).
    3.  Optimize through Stage 1 (3M samples) and Stage 2 (20K preference pairs) with DPO.
    4.  Detach the vision encoder from the LLM. Freeze its weights. This is the "PIVOT-enhanced encoder".
    5.  Combine this frozen encoder with a new Qwen2.5-1.5B and a new projector.
    6.  Optimize the projector and LLM on Cambrian’s 737K dataset (Stage 3).

**7. All Mathematical Formulas and Symbol Definitions**

**Symbol Definitions:**
*   $X_{PT}$: The post-training dataset, defined as $X_{PT} = \{x_0, x_1, . . . , x_T\}$.
*   $x_i$: An element of the dataset, representing an image, a query, and the corresponding chosen and rejected responses. $x_i = \{I_i, q_i, y^c_i, y^r_i\}$.
*   $I_i$: The image.
*   $q_i$: The query.
*   $y^c_i$: The chosen response.
*   $y^r_i$: The rejected response.
*   $\pi_\theta$: The MLLM (policy model).
*   $\pi_{ref}$: The reference model (typically the initial model before preference alignment).
*   $\beta$: The temperature controlling the strength of preference alignment.
*   $\sigma$: The sigmoid function.

**Optimization Objectives:**
The optimization objectives using the dataset $X_{PT}$ are defined as follows:

1.  **SFT Loss:**
    $$L_{SFT} = -E_{i \sim X_{PT}} \log \pi_\theta(y^c_i | I_i, q_i)$$

2.  **DPO Loss:**
    $$L_{DPO} = -E_{i \sim X_{PT}} \log \sigma \left( \beta \left( \log \frac{\pi_\theta(y^c_i |I_i,q_i)}{\pi_{ref}(y^c_i |I_i,q_i)} - \log \frac{\pi_\theta(y^r_i |I_i,q_i)}{\pi_{ref}(y^r_i |I_i,q_i)} \right) \right)$$

**8. Algorithm Pseudocode**

The paper does not provide explicit algorithm pseudocode blocks, but describes the **PIVOT-enhanced vision model evaluation** procedure and the **training procedure** textually. The steps are extracted as follows:

**Procedure: PIVOT Training and Evaluation**
1.  **Input:** Vision Encoder $E_{vision}$ (e.g., SigLIP2-So/16), LLM Head $M_{LLM}$ (e.g., Qwen2.5-1.5B), Pre-training Data $D_{pre}$ (3.2M), Post-training Data $D_{post}$ (20K), Evaluation Data $D_{eval}$ (737K).
2.  **Stage 1: Pre-training**
    *   Construct MLLM: $F(x) = M_{LLM}(Projector(E_{vision}(x)))$.
    *   Optimize $F$ on $D_{pre}$ using SFT objective.
3.  **Stage 2: Post-training (PIVOT)**
    *   Optimize $F$ on $D_{post}$ using DPO objective (Equation 1).
4.  **Extraction:**
    *   Detach $E_{vision}$ from $F$. Denote as $E_{PIVOT}$.
    *   Freeze weights of $E_{PIVOT}$.
5.  **Stage 3: Evaluation in New MLLM**
    *   Construct New MLLM: $F'(x) = M_{LLM\_new}(Projector_{new}(E_{PIVOT}(x)))$.
    *   Optimize Projector only on LAION/CC/SBU-558K.
    *   Optimize $Projector_{new}$ and $M_{LLM\_new}$ on $D_{eval}$ (Cambrian 737K).
6.  **Output:** Evaluate $F'$ on 16 VQA benchmarks.