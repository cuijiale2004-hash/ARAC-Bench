Principle 1: Quantification and Distributional Diagnosis of Intrinsic Evaluator Biases in Multimodal Agent Verification

**Definition:**  
Research on MLLM-based verifiers for agent behavior must prioritize the systematic quantification of intrinsic evaluator biases over reliance on coarse aggregate metrics. Agreement bias and analogous failure modes frequently manifest as highly skewed output distributions—such as pathological concentrations at favorable Likert labels or near-maximal confidence scores—rather than as uniform random errors. Consequently, rigorous evaluation requires decomposing verifier performance into true positive and true negative rates, measuring distributional skewness and directional bias relative to human or oracle judgments, and stress-testing across diverse scoring templates and known bias-mitigation techniques. This perspective is essential because a verifier can display superficially strong accuracy while systematically failing at its core function of detecting flawed behavior, thereby exposing downstream pipelines to silent failures. Moreover, establishing whether such biases originate from fundamental model tendencies—such as RLHF-induced sycophancy or knowledge-retrieval bottlenecks—rather than from ephemeral prompting artifacts is necessary to ensure interventions address root causes rather than symptoms. Without this granular, distributional diagnosis, the field risks legitimizing evaluators that actively reinforce incorrect agent behavior through unchecked false-positive validation.

**Core Evaluation Criteria:**
- **Fine-Grained Metric Decomposition**: Does the work report disaggregated statistics such as TPR, TNR, bias, and distributional skewness, rather than relying solely on aggregate accuracy?
- **Cross-Template Robustness**: Are biases shown to persist (or be mitigated) across a diverse set of prompt designs, scoring templates, and label-granularity configurations?
- **Distributional Analysis**: Does the study explicitly analyze the shape and skew of model output distributions to reveal systematic over-validation or under-validation tendencies?
- **Root-Cause Differentiation**: Does the research distinguish between intrinsic model limitations (e.g., training artifacts) and transient prompting effects as sources of bias?

---

Principle 2: Downstream Efficacy and Non-Disruptiveness of Verifier Interventions in Agent Improvement Pipelines

**Definition:**  
The practical value of an MLLM verifier must be ultimately validated through its causal impact on agent performance in iterative improvement pipelines, including self-refinement, online supervision, and trajectory curation. A verifier may achieve high offline classification accuracy yet yield negligible or negative net effects if its false negatives withhold critical corrective feedback or its false positives entrench brittle, non-generalizable behaviors. Therefore, evaluation frameworks must incorporate end-to-end task-completion metrics that compare agent success rates with and without verifier interventions, alongside fine-grained confusion-matrix transitions that expose whether so-called correct interventions are actually non-disruptive. Particular attention must be paid to the leniency-strictness trade-off: overly strict verification can derail valid creative solutions, while overly lenient verification fails precisely when flawed behavior demands correction. By grounding assessment in downstream utility rather than static trajectory classification, this principle ensures that verifier research delivers measurable agent improvement rather than illusory metric gains.

**Core Evaluation Criteria:**
- **Net Benefit in Downstream Tasks**: Are improvements demonstrated in concrete applications such as self-improvement loops, online supervision, or filtered behavior cloning, using task success rates as the primary metric?
- **Intervention Disruptiveness Analysis**: Does the work disaggregate outcomes by confusion-matrix categories (e.g., true negatives leading to recovery, false negatives causing unnecessary failures) to verify net positive impact?
- **Strictness-Leniency Trade-off Assessment**: Is the balance between false positives and false negatives analyzed in the context of its practical consequences for agent exploration and task completion?
- **Baseline Isolation**: Does the evaluation clearly compare verifier-augmented agents against both unaided baselines and naive verifier implementations to isolate the incremental value of the proposed method?

---

Principle 3: Benchmark Fidelity and Agent Strength Calibration for Verification Assessment

**Definition:**  
The reliability of verifier evaluation hinges fundamentally on the fidelity of the experimental testbed, encompassing trajectory difficulty, environment correctness, and oracle alignment with human intent. Evaluations that rely on weak agents producing trivial failures, buggy environments generating artificial error signals, or misaligned oracle scripts can severely misrepresent verifier capabilities—either inflating performance or masking systematic biases. High-quality research must therefore employ strong, diverse agent policies capable of generating realistic, challenging trajectories; maintain rigorously debugged environments with proper state isolation and reset mechanisms; and validate oracle evaluators against human annotations or corrected ground truth. Furthermore, the experimental design should explicitly account for the capability gap between agent and verifier, as biases such as agreement bias often intensify when verifiers are weaker or less capable than the agents they judge. This principle demands that reviewers treat benchmark quality and agent strength not as implementation details, but as determinants of whether claimed verifier improvements represent genuine advances or artifacts of an undemanding testbed.

**Core Evaluation Criteria:**
- **Agent Capability and Diversity**: Does the study utilize strong, varied agent baselines that produce nontrivial trajectories, avoiding artificially easy verification tasks dominated by obvious failures?
- **Oracle and Environment Quality**: Are oracle evaluators validated against human judgments, and are environment bugs, state leakage, and reset failures systematically addressed?
- **Verifier-Agent Gap Analysis**: Does the work examine how the relative capability of verifier and agent affects verification difficulty and bias severity?
- **Trajectory Length and Complexity Scaling**: Are results reported across varying trajectory lengths and task complexities to ensure findings generalize beyond short or simplified episodes?

---

Principle 4: Mechanistic Robustness and Cross-Configuration Generalization of Prompting-Based Verification

**Definition:**  
Proposed lightweight interventions for improving MLLM verification—such as multi-step prompting, self-conditioning, or prior elicitation—must demonstrate robustness across a broad spectrum of experimental configurations rather than relying on a single favorable prompt or model. The effectiveness of such methods should persist across diverse model families, scales, scoring templates, and ablations of internal components, including variations in prior diversity, temperature, noise injection, and reasoning-mode toggles. Researchers are further expected to situate their contributions against simpler alternatives, including direct prompt variations, confidence calibration schemes like Platt scaling, and tool-augmented baselines, to rule out gains that stem merely from idiosyncratic prompt engineering. This principle is critical because verifier research is uniquely susceptible to brittle heuristics that exploit specific model idiosyncrasies; durable methods must instead provide mechanistic insight into how an intervention modulates the model's generation distribution. Demonstrating cross-configuration stability ensures that a method yields transferable knowledge about verifier design rather than an ephemeral trick.

**Core Evaluation Criteria:**
- **Cross-Template and Cross-Model Validation**: Is the method evaluated across multiple prompt templates, scoring scales, and model families to establish generalizability?
- **Ablative Mechanistic Analysis**: Are key components systematically ablated (e.g., prior diversity, noise, step decoupling) to isolate the source of improvement?
- **Comparison to Simpler Baselines**: Does the work compare against direct prompting variants, statistical calibration, and tool-based grounding to establish superiority over trivial alternatives?
- **Mechanistic Insight**: Does the paper provide evidence that the intervention alters an underlying generative behavior (e.g., output distribution skew) rather than exploiting surface-level prompt formatting?

---

Principle 5: Durability Assessment and Orthogonal Composability in Open-Ended Verification

**Definition:**  
Research advancing MLLM verification must assess whether identified biases are durable properties of current training paradigms and whether proposed mitigations remain effective as agents and environments grow more complex. This necessitates cross-model studies spanning different vintages, sizes, and training recipes—including reasoning models—to determine if biases are invariantly intrinsic or likely to vanish with future capabilities. Scalability to longer trajectories and more demanding multimodal reasoning is equally vital, as context-window pressure and extended interaction horizons may reintroduce or amplify biases that are muted in short-horizon settings. Finally, the field must evaluate how verification interventions compose with orthogonal capability modules, such as specialist perception models or symbolic checkers, recognizing that lightweight prompting alone cannot compensate for fundamental vision-language limitations. By confronting these durability and composability questions, verifier research avoids proposing temporary patches for transient flaws and instead contributes lasting design principles for increasingly autonomous and capable agent systems.

**Core Evaluation Criteria:**
- **Intrinsic vs. Transient Bias Classification**: Does the work provide evidence that biases persist across model generations and scales, suggesting intrinsic limitations rather than temporary weaknesses?
- **Long-Horizon Scalability**: Are verification methods tested on longer trajectories and more complex task structures to assess robustness to increased context and interaction length?
- **Orthogonal Composability**: Does the research discuss or evaluate integration with specialist modules (e.g., visual experts, symbolic verifiers) to address non-bias capability gaps?
- **Forward-Looking Durability**: Does the paper explicitly consider whether the proposed method is a durable design principle or a temporary heuristic likely to be obviated by future model improvements?