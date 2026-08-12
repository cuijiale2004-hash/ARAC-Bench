**1. Research Background and Existing Pain Points**
Reinforcement Learning with Verifiable Rewards (RLVR) has proven effective for enabling large language models to elicit complex reasoning on tasks with clear verifiable outcomes, particularly in domains like math and code where reward models can be replaced by scoring functions or test cases that automatically verify correctness. However, extending RLVR to unstructured, real-world reasoning is challenging because such tasks lack easily verifiable answers. A common workaround is to use preference-based reward models, but they tend to overfit superficial artifacts (e.g. response length, formatting quirks, annotator biases) and require large volumes of pairwise comparisons. Instance-specific rubrics have recently emerged for nuanced evaluation in expert domains to capture multi-criteria judgments, yet their potential as reward signals for on-policy post-training remains underexplored. Existing generative reward models that learn to evaluate reasoning or final outputs with interpretable scores, and models that use internal confidence estimates as a proxy for reward, still lack a general-purpose approach for specifying reliable reward signals, particularly in tasks without a single ground truth where both subjective and objective criteria must be considered.

**2. Core Research Motivation and Scientific Questions**
The core research motivation is to address the gap between the simplicity of verifiable rewards and the expressiveness of preference rankings, which often come with human artifacts and operational overhead. The work explores a paradigm shift that introduces a middle ground by using structured criteria or rubrics as the core reward mechanism. The primary scientific questions are: How can RLVR be extended beyond verifiable domains to real-world reasoning tasks that depend on nuanced, multi-criteria judgments rather than binary correctness? How can instance-specific rubrics, previously used only for evaluation, be converted into reusable reward functions for on-policy reinforcement learning to provide interpretable and automatable supervision? How do rubric-based reward signals impact alignment with human preferences and performance variance across different judge scales?

**3. Overall Core Idea and Design Philosophy**
The overall core idea is Rubrics as Rewards (RaR), a framework for on-policy Reinforcement Learning that uses structured criteria or rubrics as the core reward mechanism. Rather than using rubrics only for evaluation, RaR treats them as checklist-style supervision that produces reward signals for on-policy RL. Each rubric is composed of modular, interpretable subgoals that provide automatable feedback aligned with expert intent. By decomposing "what makes a good response" into tangible, human-interpretable criteria, rubrics offer a middle ground between binary correctness signals and coarse preference rankings. The design philosophy is that once generated, rubrics provide interpretable and automatable supervision that can be applied consistently across new rollouts, offering a scalable and transparent alternative to opaque reward modeling in on-policy learning. This closes the rubric-to-learning loop and improves performance on both rubric-guided evaluations and tasks with verifiable answers.

**4. Core Innovation Points**
1. Introduce Rubrics as Rewards (RaR), an on-policy reinforcement learning framework that uses checklist-style rubrics for multi-criteria supervision in reasoning and real-world domains.
2. Synthesize instance-specific rubrics for medicine and science and release the corresponding training sets, RaR-Medicine and RaR-Science.
3. RaR-trained models consistently outperform strong baselines and yield a stable, generalizable training signal, with gains on both rubric-scored and verifiable multiple-choice evaluation settings.
4. Rubric-based rewards provide stable supervision across judge sizes, helping smaller models align effectively with human preferences and maintaining robust evaluation performance from small to large judges, thus reducing performance variance across judge scales.

**5. Overview of the Overall Technical Solution**
The overall technical solution consists of two main phases: Rubric Generation and GRPO Training. 
In the Rubric Generation phase, prompt-specific, self-contained rubric criteria are synthesized using a strong LLM (such as OpenAI’s o3-mini or GPT-4o) guided by four core design principles, with reference answers serving as proxies for expert supervision. Each rubric contains 7–20 items with categorical labels (Essential, Important, Optional, Pitfall) and numerical weights.
In the GRPO Training phase, these rubrics are used to prompt an LLM judge (e.g., gpt-4o-mini) for reward estimation. The framework investigates two complementary approaches for combining rubric feedback into scalar rewards: Explicit Aggregation (independently evaluating each criterion and computing a normalized weighted sum) and Implicit Aggregation (passing all rubric criteria to the judge to produce a single scalar reward holistically). The computed rewards then drive policy optimization via the GRPO (Group Relative Policy Optimization) on-policy learning loop.

**6. Detailed Module Design**
**6.1 Rubric Generation Module**
This module specifies criteria for high-quality responses and provides human-interpretable supervision based on four desiderata:
- Grounded in Expert Guidance: Rubrics should reflect domain expertise by capturing the essential facts, reasoning steps, and conclusions necessary for correctness. This grounding comes from human experts or their high-quality proxies (reference answers).
- Comprehensive Coverage: Rubrics should span multiple dimensions of response quality, including factual accuracy, logical coherence, completeness, style, and safety. Negative criteria (pitfalls) help identify frequent or high-risk errors.
- Criterion Importance: Rubrics should reflect that some dimensions are more critical than others (e.g., factual correctness outweighs stylistic clarity). Weights are assigned through categorical tags (Essential, Important, Optional, Pitfall) or explicit numeric values.
- Self-Contained Evaluation: Each rubric item should be independently actionable, allowing human annotators or automated judges to assess it in isolation without requiring external context or domain-specific knowledge.

Creation Process: Given the scarcity of human-annotated rubric datasets, LLMs generate instance-specific rubrics from golden reference answers at scale. For each prompt, an LLM generates a rubric of 7–20 self-contained items. Each item is assigned both a numeric and a categorical weight reflecting its relative importance. Categorical labels used are Essential, Important, Optional, and Pitfall. Pitfall criteria are phrased in positive form (e.g., “The response avoids misinformation”), so satisfying them contributes positively to the score.

**6.2 Reward Aggregation Module**
This module converts rubric feedback into scalar rewards for RL, investigating two complementary approaches:
- Explicit Aggregation: Each criterion is independently evaluated using an LLM-as-judge, and the final normalized reward is computed as a weighted sum. In experiments, numerical weights are manually assigned based on generated categorical labels: {"Essential": 1.0, "Important": 0.7, "Optional": 0.3, "Pitfall": 0.9}.
- Implicit Aggregation: All rubric criteria along with categorical weights are passed to an LLM-as-judge, delegating the aggregation to the model itself to produce a single scalar reward (a single Likert rating 1–10, normalized to [0,1]). This avoids the need for hand-tuned weights.

**6.3 Policy Training Module**
All experiments are conducted using on-policy reinforcement learning with the GRPO algorithm. The training pipeline consists of:
- Response Generation: For each prompt q, sample k = 16 responses from the current policy, using a context length of 3584 and a sampling temperature of 1.0.
- Reward Computation with Rubrics: Use gpt-4o-mini as the judge model to assign rewards to the sampled responses based on chosen aggregation strategies.
- Policy Update: The policy weights are updated using GRPO based on the computed rewards. Base policy is Qwen2.5-7B, trained with a batch size of 96, learning rate of $5 \times 10^{-6}$, and a constant schedule with 10% linear warmup on 8 NVIDIA H100 GPUs.

**6.4 Baseline Modules**
- OFF-THE-SHELF: Evaluates performance on Qwen2.5-7B and Qwen2.5-7B-Instruct without training.
- DIRECT-LIKERT: An LLM-as-judge provides a direct assessment for each response–prompt pair on a 1–10 Likert scale, normalized to [0, 1], used directly as the reward signal.
- REFERENCE-LIKERT: An LLM-as-judge compares the generated response against a reference answer and assigns a 1–10 Likert score, normalized to [0, 1].
- RaR-PREDEFINED: Uses a fixed set of generic rubrics for all prompts (e.g. response is concise, contains correct information) with Explicit Aggregation and uniformly weighted criteria.

**7. All Mathematical Formulas and Symbol Definitions**
Let $x$ denote an input prompt and $\hat{y} \sim \pi_\theta(\cdot | x)$ be a sampled response from a model parameterized by $\theta$.
Each prompt $x$ is associated with a set of $k$ rubric items $\{(w_j, c_j)\}_{j=1}^k$, where:
- $w_j \in \mathbb{R}$ denotes the weight of criterion $j$.
- $c_j: (x, \hat{y}) \mapsto \{0, 1\}$ is a binary correctness function that indicates whether the response $\hat{y}$ satisfies that criterion given the prompt.

Explicit Aggregation Formula:
$$r(x, \hat{y}) = \frac{\sum_{j=1}^k w_j \cdot c_j(x, \hat{y})}{\sum_{j=1}^k w_j}$$

Implicit Aggregation Formula:
$$r_{implicit}(x, \hat{y}) = f_\phi(x, \hat{y}, \{d_j\}_{j=1}^k)$$
Here, $f_\phi$ denotes an LLM-based judge that takes the prompt $x$, the response $\hat{y}$, and the set of rubric criteria $\{d_j\}$ as input.

Remark 1 (Rubrics as Rewards subsumes RLVR): The RLVR setting is a special case of rubric-based rewards defined in Equation 1, where $k = 1$, $w_1 = 1$, and $c_1(x, \hat{y})$ reduces to a single verifiable correctness function that compares the model output $\hat{y}$ against the known correct answer $y$. Formally:
$$r_{RLVR}(x, \hat{y}) = match(y, \hat{y})$$
where $match(y, \hat{y}) \in \{0, 1\}$ indicates whether the response satisfies the verifiable correctness condition.

Reference-Likert Reward Definition:
$$R_{ref}(q, x) = Norm(LikertScore(q, x, x^*))$$
where $x^*$ denotes the reference answer.

**8. Algorithm Pseudocode**
The paper does not contain explicit algorithm pseudocode blocks. The logical training pipeline is defined in Section 4.2 as follows:
1. Response Generation: For each prompt q, sample k = 16 responses from the current policy $\pi_\theta$, using a context length of 3584 and a sampling temperature of 1.0.
2. Reward Computation with Rubrics: Use gpt-4o-mini as the judge model to assign rewards $R_q$ to the sampled responses. Experiment with various reward computation and aggregations strategies (Explicit Aggregation or Implicit Aggregation).
3. Policy Update: The policy weights are updated using GRPO based on the computed rewards.