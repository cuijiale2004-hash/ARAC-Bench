**1. Research Background and Existing Pain Points**

Recent advances in multimodal large language models (MLLMs) have significantly improved video captioning, question answering, and long-form video understanding across various domains. However, their progress poses a critical challenge: how to evaluate their outputs with reliability, interpretability, and at scale? Existing evaluation methods face severe limitations:

1.  **Failure of Traditional Metrics:** Commonly used metrics such as BLEU, ROUGE, and BERTScore fail to capture the nuances of human judgment. They struggle to capture semantic fidelity, contextual grounding, or task-specific reasoning. Moreover, in open-ended tasks where multiple valid answers exist, simple reference overlap can be misleading.
2.  **Cost of Human Evaluation:** Human evaluation is often considered the gold standard, but it is expensive, slow to scale, and suffers from inter-annotator variability.
3.  **Underexploration of LLM-as-a-Judge for Video:** While LLM-as-a-Judge has improved evaluation in text generation and more recently in vision-language tasks via MLLM-as-a-Judge, applying MLLM-as-a-judge to video understanding remains relatively unexplored. This is largely due to the temporal and multimodal complexity of the video.
4.  **Lack of Large-Scale Evaluation Resources:** The field lacks comprehensive datasets with human preference signals or standardized benchmarks for verifying alignment with human judgments. Existing work either relies on proprietary models (lacking transparency and reproducibility) or on small open-source MLLMs in zero-shot settings (falling short of human-level reliability).
5.  **Lack of Principled Evaluation Criteria:** Current (M)LLM-as-a-judge methods depend either on generic rubrics, which are often vague and brittle, or on manually authored rubrics, which cannot scale across tasks.

**2. Core Research Motivation and Scientific Questions**

**Core Research Motivation:** To address the gap in video understanding evaluation by creating a scalable, automatic, and reliable MLLM-as-a-Judge framework that eliminates the need for costly human annotation while yielding robust training data and standardized evaluation suites that align with human judgment.

**Scientific Questions:**
1. How to synthesize large-scale, fine-grained training data for video evaluators without human annotation?
2. How to enforce quality control and self-correction in synthetic data generation for evaluation tasks?
3. Can smaller, fine-tuned models on bootstrapped data match or outperform much larger general-purpose models in evaluation accuracy and alignment with human ratings?
4. How to generate instance-specific, interpretable evaluation criteria (rubrics) automatically at test time for video understanding tasks?

**3. Overall Core Idea and Design Philosophy**

The overall core idea is to introduce a bootstrapping framework to train video understanding evaluators, building on the interplay between a generator and an evaluator. The design philosophy draws inspiration from self-refinement approaches, where self-consistency and self-verification enhance LLM performance, and models adapt through verbal feedback.

The framework consists of two key pillars:
1.  **Automated Data Generation:** Automatically generating training data by producing candidate responses across a 1–5 rating scale, validating them with an evaluator model, and refining cases where predicted ratings diverge from expectations.
2.  **Rubric-Driven Evaluation:** Enabling judge models to generate instance-specific rubrics at test time, ensuring fine-grained evaluation that is both interpretable and anchored in explicit evaluation guidelines tailored to the specific instruction and video content.

The framework has two stages: (1) iterative bootstrapping to construct large-scale, fine-grained training data, and (2) fine-tuning judge models which are evaluated under both pointwise and pairwise settings.

**4. Core Innovation Points**

1.  **First Bootstrapped Framework for Video Evaluators:** Introduction of VideoJudge, the first bootstrapped framework for training scalable MLLM-based evaluators across diverse video understanding tasks, eliminating the need for costly human annotation.
2.  **Instance-Specific Rubric Generation at Inference:** Train judge models that can not only assign ratings but also generate high-quality, instance-specific rubrics at inference time, enabling interpretable and grounded evaluation.
3.  **Small Models Matching Large Models:** Demonstration that fine-tuned small models (3B and 7B) on the bootstrapped data can match or outperform much larger models (up to 72B) in accuracy and alignment with human-specified ratings.
4.  **Comprehensive Artifacts:** Provision of a suite of trained pointwise and pairwise judge models, meta-evaluation benchmarks, bootstrapped datasets, and other artifacts to support reproducible research in video understanding evaluation.

**5. Overview of the Overall Technical Solution**

The overall technical solution involves a generator–evaluator pipeline that jointly synthesizes data and enforces quality control. 
1.  **Seed Data:** Sourced from three large-scale video instruction–response datasets (VideoInstruct-100K, VCG-Plus-112K, and VideoChat2-IT), merged and de-duplicated.
2.  **Context Generation:** Dense video descriptions are generated using strong vision-language models to provide richer grounding.
3.  **Iterative Bootstrapping:** For each instruction-video pair with a gold response (Rating 5), a generator produces candidate responses for Ratings 1 to 4. An evaluator scores them. If the evaluator's rating deviates from the target, the generator refines the response using the evaluator's feedback.
4.  **Training:** The bootstrapped dataset (pointwise and pairwise) is used to fine-tune evaluator models autoregressively to predict ratings with explanations, and optionally generate rubrics.

**6. Detailed Module Design**

**6.1 Seed Data Preparation Module**
*   **Sources:** VideoInstruct-100K, VCG-Plus-112K, and VideoChat2-IT. For VideoChat2-IT (multi-turn), only the first human–assistant exchange is retained.
*   **De-duplication:** The three corpora are merged and de-duplicated at the instruction level for each video using a MINHASHLSH index (128 permutations, Jaccard threshold 0.9).
*   **Sampling:** 25K examples are randomly sampled from the de-duplicated pool, resulting in a corpus of triplets $(v, x, y)$, where $v$ denotes the video, $x$ the instruction, and $y$ the gold-standard response.

**6.2 Dense Description Generation Module**
*   **Mechanism:** Dense video descriptions $\tilde{v}$ are generated using Qwen2.5-VL-32B-Instruct and GPT-4o-mini (frames provided as sequence of images, 2 FPS, max 180 frames). Used as semantic context for both the generator and evaluator to reduce the need for repeated inference over raw video.

**6.3 Initial Generation Module**
*   **Mechanism:** For each instruction–video pair $(x, \tilde{v})$ with gold response $y^*$, a generator model $G$ produces $N-1$ candidate responses (where $N=5$), each intended to correspond to a rating $r \in \{1, \dots, N-1\}$. The gold response $y^*$ is included as the highest-rated response with rating $N$.

**6.4 Feedback Module**
*   **Mechanism:** Each candidate response $y^{(r)}_t$ is evaluated by an evaluator model $E$, which assigns a rating $\hat{r}$ and provides reasoning $f^{(r)}_t$. The deviation between the intended rating $r$ and the evaluator’s rating $\hat{r}$ is computed to determine acceptance or refinement. Candidates for which the deviation is $\le \alpha$ are accepted directly.

**6.5 Refinement Module**
*   **Mechanism:** For candidates with a rating deviation $> \alpha$, the generator is prompted again using the evaluator’s feedback to improve the response. This iterative refinement continues until the candidate meets the acceptance criterion or a maximum of $T$ iterations is reached.

**6.6 Model Training Module**
*   **Dataset Definition:** $D = \{(v_i, x_i, y_i, t_i)\}^M_{i=1}$, where $v_i$ denotes the video, $x_i$ the instruction, $y_i$ a candidate response (or a response pair in the pairwise setting), and $t_i$ the associated target annotation.
*   **Pointwise Setting:** The model produces intermediate reasoning within `<thinking></thinking>` followed by a scalar rating in `<score></score>`.
*   **Rubric Setting:** The model can optionally generate task-specific rubrics in `<rubric></rubric>` before reasoning and evaluation.
*   **Pairwise Setting:** The model outputs its decision within `<answer></answer>` based on a pair of candidate responses. To avoid positional bias, the order of responses is randomized during training and evaluation.
*   **Hyperparameters:** Full fine-tuning in BF16 precision with a maximum sequence length of 128K tokens, fps rate of 1 with maximum number of frames 60 for training and 180 during evaluation. Trained for 2 epochs with a batch size of 16. Learning rate $2 \times 10^{-7}$ with cosine decay, warmup ratio of 0.03, weight decay of 0, gradient clipping at 1.

**7. All Mathematical Formulas and Symbol Definitions**

**Symbol Definitions:**
*   $v$: Video
*   $x$: Instruction
*   $y$ (or $y^*$): Gold-standard response
*   $\tilde{v}$: Dense video description
*   $y^{(r)}_t$: Candidate response intended for rating $r$ at iteration $t$
*   $r$: Intended rating ($1 \dots N-1$)
*   $N$: Max rating ($N=5$)
*   $G$: Generator model
*   $E$: Evaluator model
*   $p_{gen}, p_{eval}, p_{ref}$: Prompts for generation, evaluation, and refinement respectively
*   $\hat{r}$: Evaluator's predicted rating
*   $f^{(r)}_t$: Evaluator's reasoning/feedback for rating $r$ at iteration $t$
*   $\Delta^{(r)}_t$: Deviation between intended rating and evaluator's rating
*   $\alpha$: Acceptance threshold
*   $T$: Maximum number of refinement iterations
*   $D$: Bootstrapped dataset
*   $M$: Number of examples in dataset
*   $t_i$: Target annotation (rating or preference label) for example $i$
*   $t_{i,j}$: The $j$-th token of $t_i$
*   $\theta$: Model parameters
*   $\mathcal{L}(\theta)$: Negative log-likelihood loss function

**Mathematical Formulas:**

Initial Generation:
$$y^{(r)}_0 = G(p_{gen}\|\tilde{v}\|x\|y^*, r)$$

Feedback / Evaluation:
$$\hat{r}, f^{(r)}_t = E(p_{eval}\|\tilde{v}\|x\|y^*\|y^{(r)}_t)$$

Deviation Calculation:
$$\Delta^{(r)}_t = |r - \hat{r}|$$

Refinement:
$$y^{(r)}_{t+1} = G(p_{ref}\|\tilde{v}\|x\|y^*\|y^{(r)}_t\|f^{(r)}_t, r)$$

Training Loss Function:
$$\mathcal{L}(\theta) = - \frac{1}{M} \sum_{i=1}^M \sum_{j=1}^{|t_i|} \log P_\theta(t_{i,j} | t_{i,<j}, v_i, x_i, y_i)$$

**8. Algorithm Pseudocode**

**Algorithm 1: Bootstrapping Training Data with Self-Refinement**
Input: Video $v$, video description $\tilde{v}$, instruction $x$, gold response $y^*$, generator $G$, evaluator $E$, threshold $\alpha$, max iterations $T$
Output: Bootstrapped dataset $D$

Initialize $D \leftarrow \{(v, x, y^*, N)\}$ // gold response with max rating

for $r \in \{1, \dots, N-1\}$ do
$\quad$ $y^{(r)}_0 \leftarrow G(p_{gen}\|\tilde{v}\|x\|y^*, r)$ // initial generation
$\quad$ for $t \in \{0, \dots, T-1\}$ do
$\quad\quad$ // feedback and refinement
$\quad\quad$ $\hat{r}, f^{(r)}_t \leftarrow E(p_{eval}\|\tilde{v}\|x\|y^*\|y^{(r)}_t)$
$\quad\quad$ if $|r - \hat{r}| \le \alpha$ then
$\quad\quad\quad$ $D \leftarrow D \cup \{(v, x, y^{(r)}_t, r)\}$
$\quad\quad\quad$ break
$\quad\quad$ else
$\quad\quad\quad$ $y^{(r)}_{t+1} \leftarrow G(p_{ref}\|\tilde{v}\|x\|y^*\|y^{(r)}_t\|f^{(r)}_t, r)$
$\quad\quad$ end if
$\quad$ end for
end for

return $D$