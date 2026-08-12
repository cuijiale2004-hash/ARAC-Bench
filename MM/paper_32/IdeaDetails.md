Based on the provided academic paper, the complete and standard Research Idea and the full set of implementation plans are extracted as follows:

### 1. Research Background and Existing Pain Points

**Research Background:**
Reinforcement learning with verifiable rewards (RLVR) has recently catalyzed a wave of "MLLM-r1" approaches that bring RL to vision language models. Prior to RL, employing a pre-training or warm-up phase (termed "cold start") can significantly improve the readability, stability, and final performance of RL training. Currently, the most commonly used cold start strategy is supervised fine-tuning (SFT), where the model is first fine-tuned on a set of high-quality reasoning data to provide a better initial policy for the subsequent RL phase. The common understanding behind SFT-based cold start is that reasoning abilities, reasoning format, and other learning objectives can be jointly learned during the cold start phase.

**Existing Pain Points:**
1.  **Generalization Degradation:** The SFT-based joint learning paradigm may largely affect the model's generalization capability. SFT adopts the reasoning paradigm intertwined with task solution and output format, which may induce instruction-style overfitting, weakens out-of-distribution (OOD) generalization, and consequently degrades subsequent RL performance.
2.  **Lack of Quantification Metrics:** There is a lack of metrics to quantify the generalization capability of a model under different cold start training methods and to evaluate how this capability coordinates with subsequent RL.
3.  **Data Generation Gap:** The prohibitive cost of human annotation has motivated the use of synthetic data via distillation from larger teacher models. However, when the capability gap between the teacher model and the student model is too large, it can lead to a decline in model performance due to distribution divergence.
4.  **Constraints between SFT and RL:** Existing paradigms like DeepSeek-R1-Zero still have limitations regarding reliance on SFT cold start and the constraints between SFT and subsequent RL, leaving room for further improvement in decoupling learning objectives.

### 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:**
To address the limitations of SFT-based joint learning, we consider an alternative learning paradigm that separates the learning process into hierarchical stages. The intuition is that the cold start phase should focus more on shallow learning to avoid prematurely getting stuck in in-distribution problem solving, while subsequent RL focuses on the deep-level learning of a solution to boost overall performance. The selection of pre-training methods in cold start needs to better support subsequent RL, both in terms of generalization and by having separate objectives to facilitate better final results. Furthermore, generating cold start data should align with the model's intrinsic capability distribution rather than relying on larger, dissimilar external models.

**Scientific Questions:**
1.  How to quantify and improve the model's generalization capability during the cold start phase to work in concert with subsequent RL?
2.  How to decouple the learning of shallow objectives (format, style, structure) from deep logical reasoning skills?
3.  How to generate high-quality preference data for cold start training without relying on larger external teacher models or manual annotation?
4.  How does a preference-based cold start (DPO) compare to SFT in terms of generalization, exploration capability, and training stability for downstream RL?

### 3. Overall Core Idea and Design Philosophy

**Overall Core Idea:**
The core idea is Decoupling Multimodal Learning. Instead of using SFT which jointly learns reasoning content and format (leading to overfitting), we separate the learning objectives. The cold start phase focuses on shallow, transferable surface-form criteria (format, structure, style) using preference-based training, while the subsequent RL phase focuses on deep reasoning results. We employ self-distillation to generate preference data, avoiding reliance on larger teachers.

**Design Philosophy:**
1.  **Hierarchical Decoupling:** Separate the learning process such that the cold start handles format alignment (shallow learning) and RL handles solution reasoning (deep learning). This prevents the model from prematurely getting stuck on in-distribution problem solving and allows credit assignment during RL to be more accurately attributed to reasoning quality rather than structural compliance.
2.  **Preference over Supervision:** Preference-based training methods (e.g., DPO) generalize better than SFT-based methods. DPO maximizes the margin between chosen and rejected responses, cultivating a smoother and more robust probability distribution that enhances the model’s generalization performance, unlike SFT which creates a sharply peaked distribution that limits generalization.
3.  **Self-Distillation:** Generating preference data via self-distillation (using the model's own exploratory policy) is more effective than using data from powerful external teacher models. Preference data closely aligned with the model's intrinsic capability distribution is more effective for alignment than guidance from a more capable but dissimilar external model.

### 4. Core Innovation Points

1.  **SPECS Framework:** A three-stage cold start strategy that generates preference data through self-distillation, uses DPO for cold start training, and separates training objectives so that the model first aligns with output formats, providing a stronger starting point for RL.
2.  **Generalization Factor (GF):** A novel metric proposed to evaluate a model's generalization capability under different cold start training methods by comparing its performance on in-distribution (ID) and out-of-distribution (OOD) tasks using the F$\beta$-score logic.
3.  **Decoupling Learning:** Revealing the importance of decoupling learning between the cold-start and RL phases. This separation improves exploration and reduces the risk of the model getting stuck on in-distribution solutions, leading to stable training and a higher performance ceiling.
4.  **Self-Distilled Preference Data:** A mechanism to generate preference data where both chosen and rejected responses contain the correct final answer but differ in format. Chosen responses are filtered for reasoning consistency, while rejected responses undergo "Rejected Response Pollution" to ensure a clear learning signal for format.
5.  **DPO-based Cold Start Superiority:** Empirical proof that a DPO cold start gives the model stronger generalization ability, greater exploration capability (higher Rollout Branching Factor), and more stable training dynamics compared to SFT-based cold start, achieving consistent performance gains over strong baselines (e.g., +4.1% MEGA-BENCH, +12.2% MATHVISTA).

### 5. Overview of the Overall Technical Solution

The overall technical solution is the SPECS framework, a three-stage training optimization strategy:

1.  **Stage 1: Self-Distillation for Preference Data Generation.** An initial policy ($\pi_{GRPO-zero}$) is created by applying a brief RL fine-tuning on the base model. This policy generates responses alongside the base model. Responses are filtered for consistency (reasoning matching answer) to form "Chosen" samples. "Rejected" samples are created by taking correct answers and applying format pollution. These form self-distilled preference pairs focused on format.
2.  **Stage 2: DPO-based Pre-Alignment for Cold-Start.** The base VLM is pre-aligned using Direct Preference Optimization (DPO) on the self-distilled dataset. A hybrid loss function combining DPO and SFT loss is used to ensure the model learns the directional preference signal without drifting from the core distribution of high-quality text.
3.  **Stage 3: Final GRPO Fine-Tuning.** The pre-aligned cold-start model undergoes final RL fine-tuning using the GRPO algorithm. Guided by a composite reward function (format + accuracy), the model focuses computational resources on enhancing complex reasoning capabilities, as it has already mastered the output format in Stage 2.

### 6. Detailed Module Design

**Module 1: Self-Distillation for Preference Data Generation**
*   **Objective:** To cultivate a preliminary "seed model" with enhanced reasoning capabilities and leverage it to autonomously generate a high-quality preference dataset.
*   **Mechanism:**
    1.  **Initial Policy Creation ($\pi_{GRPO-zero}$):** A standard base VLM often lacks the capability to generate outputs of sufficient reasoning ability. To address this, a brief, initial phase of RL fine-tuning (GRPO) is conducted on the base model. This creates an initial policy, $\pi_{GRPO-zero}$, which is more adept at exploring the solution space.
    2.  **Response Generation:** Two models, the exploratory $\pi_{GRPO-zero}$ and $\pi_{base}$, are prompted with specific format instructions (e.g.,  \`... <answer>...</answer>\`) to create a dataset containing pairs of responses that are both correct in their final answer, but differ in reasoning paradigm and answer format.
    3.  **Chosen Response Filtration:** For the chosen response ($y_i^+$), an LLM (e.g., Gemini-2.5 flash) is used as an evaluator to assess whether the reasoning path in the $\pi_{GRPO-zero}$ response aligns correctly with its final answer. Only responses where reasoning and answer are consistent are retained.
    4.  **Rejected Response Pollution:** For the rejected response ($y_i^-$), responses containing the correct answer but deviating from the required format are selected. To ensure a clear learning signal and avoid incidental correct formatting, one of five types of format corruption is randomly applied:
        *   Remove all tags ( \`<think\>, \`, \`<answer\>, \`</answer\>`).
        *   Remove the \`<answer\>` and \`</answer\>` tags.
        *   Remove the \` \` and \` \` tags.
        *   Remove the \`<answer\>` and \`</answer\>` tags and move the closing \` \` tag to the end of the response.
        *   Replace the \`<answer\>` tags with the string "Answer:" and remove \`</answer\>` tags.
    5.  **Preference Pair Construction:** Chosen and rejected responses are paired. Both contain the correct final answer, facilitating decoupled learning that separates reasoning paradigms/formats from core logical reasoning ability.

**Module 2: DPO-based Pre-Alignment for Cold-Start**
*   **Objective:** To leverage the self-distilled preference dataset to pre-align the base VLM, yielding a "cold-start" model that serves as a significantly improved starting point for final RL fine-tuning.
*   **Mechanism:**
    *   **Direct Preference Optimization (DPO):** DPO directly optimizes the language model on preference data without an explicit reward model. It increases the relative probability of the chosen response over the rejected one.
    *   **Hybrid Loss:** To augment the process, an SFT loss computed on the "chosen" samples is incorporated as regularization. This ensures that while the model learns the directional preference signal from DPO, it does not drift far from the core distribution of high-quality text embodied by the chosen responses. The combined loss function is $\mathcal{L}_{hybrid} = \mathcal{L}_{DPO} + \lambda \mathcal{L}_{SFT}$.

**Module 3: Final GRPO Fine-Tuning**
*   **Objective:** To achieve peak performance by fine-tuning the pre-aligned cold-start model, focusing computational resources on enhancing complex reasoning capabilities.
*   **Mechanism:**
    *   **Initialization:** Uses the cold-start model from Stage 2 as the initialization point. The pre-alignment ensures the model has already mastered the output format, so it does not expend resources on basic structural compliance.
    *   **Algorithm:** Employs the GRPO algorithm.
    *   **Reward Function:** Guided by a composite reward function combining format and accuracy components: $R_{total}(o, q) = R_{format}(o) + R_{acc}(o, q)$.
        *   **Format Reward ($R_{format}$):** Assigns a fixed value of 0.5 for structurally correct outputs.
        *   **Accuracy Reward ($R_{acc}$):** Provides a binary signal (1.0 for correct, 0 otherwise). Determined by a hybrid mechanism based on question type $T(q)$: Rule-based ($R_{rule}$) for Multiple-Choice/Numerical, and LLM-based ($R_{llm}$) for Short-Answer.

### 7. All Mathematical Formulas and Symbol Definitions

**Generalization Factor (GF) Definitions:**
*   $\psi(f_n, P) \in \mathbb{R}$: Evaluation function measuring the performance of model $f$ on distribution $P$ with sample size $n$.
*   $\Psi_{ID}(n) = \psi(f_n, P_{train})$: In-Distribution Performance, evaluated on a hold-out set from the same distribution $P_{train}$.
*   $\Psi_{OOD}(n) = \mathbb{E}_{Q \sim \alpha}[\psi(f_n, Q)]$: Out-of-Distribution Performance, the weighted average performance across a set of $m$ distinct OOD distributions $Q = \{Q_1, \dots, Q_m\}$, with weights defined by distribution $\alpha$.
*   $G_{ID}(n) = \Psi_{ID}(n) - \Psi_{ID}(0)$: ID performance gain over baseline model $f_0$.
*   $G_{OOD}(n) = \Psi_{OOD}(n) - \Psi_{OOD}(0)$: OOD performance gain over baseline model $f_0$.
*   $\Gamma(n)$: Generalization Factor, defined as the F$\beta$-score with respect to OOD and ID performance gains:
    $$\Gamma(n) = \frac{(1 + \beta^2) G_{ID}(n) G_{OOD}(n)}{\beta^2 \cdot G_{ID}(n) + G_{OOD}(n)}$$
    Where weighting coefficient $\beta$ is generally set to 2 to reflect the importance of OOD performance gain.

**DPO Loss Definition:**
*   $\mathcal{L}_{DPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]$
    *   $\pi_\theta$: The policy being optimized.
    *   $\pi_{ref}$: The reference policy (the initial base model).
    *   $\beta$: A temperature parameter.
    *   $(x, y_w, y_l)$: A triplet of prompt, chosen response, and rejected response from dataset $\mathcal{D}$.

**Hybrid Loss Definition:**
*   $\mathcal{L}_{hybrid} = \mathcal{L}_{DPO} + \lambda \mathcal{L}_{SFT}$
    *   $\mathcal{L}_{SFT}$: The conventional negative log-likelihood loss on the chosen responses, defined as $\mathcal{L}_{SFT}(\pi_\theta) = \mathbb{E}_{(x, y_c) \sim \mathcal{D}} [- \log \pi_\theta (y_c|x)]$.
    *   $\lambda$: A weighting coefficient to balance the two objectives (set to 1 in implementation).
    *   Full Expression:
        $$\mathcal{L}_{hybrid} = -\mathbb{E}_{(x, y_c, y_r) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_c|x)}{\pi_{ref}(y_c|x)} - \beta \log \frac{\pi_\theta(y_r|x)}{\pi_{ref}(y_r|x)} \right) \right] + \lambda \mathbb{E}_{(x, y_c) \sim \mathcal{D}}[- \log \pi_\theta(y_c|x)]$$

**Reward Function Definition:**
*   $R_{total}(o, q) = R_{format}(o) + R_{acc}(o, q)$
*   $R_{format}(o) = 0.5$ (for structurally correct outputs).
*   $R_{acc}(o, q) = \begin{cases} R_{rule}(o, q) & \text{if } T(q) \in \{\text{Multiple-Choice, Numerical}\} \\ R_{llm}(o, q) & \text{if } T(q) = \text{Short-Answer} \end{cases}$
    *   $R_{rule}$: Rule-based assessment for objective types.
    *   $R_{llm}$: LLM-based assessment (GPT-4o) for subjective types.
    *   $R_{acc}$ provides 1.0 for correct answer and 0 otherwise.

### 8. Algorithm Pseudocode

*(Note: The original text does not provide a formal pseudocode block. The logical flow based on the text's description of the stages and steps is extracted as follows:)*

**Algorithm: SPECS Framework**
**Input:** Base Model $\pi_{base}$, Training Data $D_{train}$, Format Instructions, LLM Evaluator.
**Output:** Final Optimized Model $\pi_{final}$.

**Stage 1: Self-Distillation for Preference Data Generation**
1.  **Initialize:** Perform brief GRPO fine-tuning on $\pi_{base}$ using $D_{train}$ to obtain exploratory policy $\pi_{GRPO-zero}$.
2.  **Generate Responses:** Prompt $\pi_{GRPO-zero}$ and $\pi_{base}$ with format instructions for questions in $D_{train}$.
3.  **Filtration (Chosen):** Use LLM Evaluator to verify consistency between reasoning path and final answer in $\pi_{GRPO-zero}$ responses. Retain consistent responses as Chosen Candidates $Y_{chosen}$.
4.  **Pollution (Rejected):** For responses with correct answers but incorrect format (or generated by $\pi_{base}$), apply one of 5 random format corruption rules (e.g., removing tags) to create Rejected Candidates $Y_{rejected}$.
5.  **Construct Pairs:** Form Preference Dataset $\mathcal{D}_{pref} = \{(x_i, y_i^+, y_i^-)\}$ where $y_i^+ \in Y_{chosen}$ and $y_i^- \in Y_{rejected}$.

**Stage 2: DPO-based Pre-Alignment for Cold-Start**
6.  **Initialize:** Set $\pi_{ref} \leftarrow \pi_{base}$.
7.  **Train:** Optimize $\pi_\theta$ using the hybrid loss on $\mathcal{D}_{pref}$:
    $\mathcal{L}_{hybrid} = \mathcal{L}_{DPO}(\pi_\theta; \pi_{ref}) + \lambda \mathcal{L}_{SFT}(\pi_\theta)$
8.  **Output:** Warmup Model $\pi_{warmup} \leftarrow \pi_\theta$.

**Stage 3: Final GRPO Fine-Tuning**
9.  **Initialize:** Set initial policy for RL as $\pi_{warmup}$.
10. **Train:** Optimize policy using GRPO with composite reward $R_{total}(o, q) = R_{format}(o) + R_{acc}(o, q)$.
11. **Return** $\pi_{final}$.