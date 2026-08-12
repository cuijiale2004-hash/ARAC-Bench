**Principle 1: Mechanistic Diagnosis of Visual Representation Reshaping Under Preference Optimization vs Supervised Fine-Tuning in Multimodal LLMs**

**Definition:**  
The field of multimodal LLMs has historically been LLM-centric, often treating vision encoders as static components whose capabilities are inherited from pre-training. This principle requires that research investigating post-training paradigms must provide deep, mechanistic analysis of how strategies such as RL and SFT fundamentally alter the underlying visual representations, rather than merely reporting downstream task scores. Reviewers should evaluate whether the work disentangles the vision encoder from the language backbone to assess standalone representational quality through vision-specific diagnostics. The analysis must transcend black-box accuracy metrics and employ tools such as linear probing, segmentation probing, gradient visualization, and cross-modal alignment metrics. Such rigor is essential because distinct training objectives may yield similar VQA scores while producing radically different visual features with divergent generalization properties. Ultimately, the work must convincingly demonstrate *how* and *why* a training strategy reshapes visual perception.

**Core Evaluation Criteria:**
- **Independent Encoder Evaluation**: Does the work isolate the vision encoder (and/or projector) from the LLM to assess pure visual representational changes via vision-only tasks like ImageNet classification?
- **Multi-Granularity Evidence**: Are diverse analysis tools employed, including global classification accuracy, dense prediction tasks like segmentation, and fine-grained gradient localization via Grad-CAM?
- **Causal Mechanistic Explanation**: Does the study establish a causal or mechanistic link between the training objective's intrinsic properties (e.g., the contrastive nature of DPO gradients) and the observed representational changes?
- **Scaling Consistency**: Are representational effects validated across varying scales of vision encoders, LLMs, and training data to ensure the phenomenon is not a scale-specific artifact?

---

**Principle 2: Controlled Comparative Rigor and Data-Fairness Validation Across Post-Training Objectives in Vision-Language Models**

**Definition:**  
Empirical claims that one post-training objective outperforms another are fragile when confounded by mismatched data formats, implicit data advantages, or unequal computational budgets. This principle mandates rigorous control over experimental variables when comparing strategies such as preference optimization and supervised fine-tuning in MLLMs. Because algorithms like DPO naturally consume paired responses while SFT uses single targets, reviewers must scrutinize whether authors account for data-scale disparities, distribution mismatches, and response-format biases. The work should validate its conclusions through data-scaling curves, distribution-shift experiments (e.g., SFT-friendly versus RL-friendly data), and fixed-budget comparisons. Only by systematically eliminating these confounders can the community attribute performance differences to intrinsic algorithmic properties rather than experimental artifacts.

**Core Evaluation Criteria:**
- **Budget Parity**: Are training comparisons conducted under strictly matched computational budgets, data scales, and model update regimes?
- **Data Distribution Robustness**: Does the study test whether findings hold across different data types and distributions, including those that may inherently favor one objective over the other?
- **Scaling Trajectory Analysis**: Are scaling experiments provided to verify whether performance gaps widen, shrink, or invert as training data or model size increases?
- **Confounder Disclosure**: Does the work transparently acknowledge structural biases (e.g., paired versus single-sample data, access to negative examples) and attempt to mitigate or bound them?

---

**Principle 3: Cross-Algorithm Generalizability Verification for Reinforcement Learning Effects on Visual Perception**

**Definition:**  
The multimodal RL landscape encompasses diverse algorithmic families—from offline preference optimization like DPO to online methods like PPO and group-relative schemes like GRPO—each with unique data requirements, reward structures, and optimization dynamics. This principle requires that findings about RL's superiority or representational impact not be anchored to a single algorithmic variant. Reviewers must evaluate whether the work validates its core claims across multiple RL paradigms and discusses how algorithm-specific mechanics (e.g., dependence on external reward models, requirements for verifiable signals, reference model formulation) influence the results. Such cross-algorithm validation is crucial to distinguish universal properties of RL-based visual learning from idiosyncrasies of a specific implementation or codebase.

**Core Evaluation Criteria:**
- **Algorithmic Breadth**: Are multiple distinct RL families (e.g., offline preference optimization, online policy gradient, group-relative methods) compared against equivalent SFT baselines?
- **Confounder Isolation**: Does the study address algorithm-specific confounding factors, such as reward model quality in PPO or the availability of verifiable tasks in GRPO?
- **Consistent Effect Direction**: Do the core phenomena (e.g., improved visual localization, stronger representations) replicate with the same directional effect across different algorithms?
- **Scope Limitations**: Does the work honestly discuss which RL paradigms were excluded and why, and whether findings might fail to generalize to them?

---

**Principle 4: Efficiency-Performance Tradeoff and Downstream Utility of RL-Evolved Vision Encoders in Production MLLM Pipelines**

**Definition:**  
When analytical insights are operationalized into training recipes (e.g., PIVOT), their scientific and practical value hinges on demonstrable utility within resource-constrained MLLM development pipelines. This principle assesses whether a proposed vision encoder evolution strategy delivers compelling efficiency-performance tradeoffs compared to standard vision pre-training or naive model scaling. Reviewers should examine whether the method achieves superior downstream performance with a fraction of the compute typically required for vision pre-training, and whether gains persist when the encoder is frozen and transferred to new LLM backbones. Comparisons must include strong contemporary baselines—such as larger encoders, next-generation architectures, or multi-encoder ensembles—to rule out trivial gains that merely reflect additional parameter scaling rather than representational quality.

**Core Evaluation Criteria:**
- **Compute Efficiency**: Does the proposed regime demonstrate a significantly better performance-to-FLOP ratio than standard vision pre-training or heavy fine-tuning?
- **Cross-Backbone Generalization**: Are gains maintained when the evolved encoder is integrated into different LLM architectures or training configurations without further encoder updates?
- **Strong Baseline Supremacy**: Does the method outperform not just its original checkpoint, but also larger, newer, or ensemble-based encoders?
- **Deployment Transparency**: Are training costs (e.g., GPU hours, data scale, wall-clock time) reported clearly to enable practical assessment by the community?

---

**Principle 5: Multidimensional Safety and Capability Retention Assessment in Multimodal Post-Training Alignment**

**Definition:**  
Improvements on standard academic VQA benchmarks do not guarantee safe, robust, or generalizable multimodal behavior. This principle demands holistic evaluation of post-training strategies across failure modes that matter in real-world deployment, including object hallucination, susceptibility to visual illusions, robustness to natural adversaries, and retention of core language capabilities. Reviewers must verify that purported alignment benefits in the visual modality do not come at the cost of degraded text-only performance, increased hallucination frequency, or narrowed task coverage. The evaluation should span generative tasks such as detailed captioning, preference-based alignment benchmarks, and real-world scenario tests to ensure that the training strategy genuinely enhances the model's perceptual reliability without causing catastrophic forgetting or modality interference.

**Core Evaluation Criteria:**
- **Hallucination and Safety Metrics**: Does the evaluation include dedicated benchmarks for object hallucination, visual illusion, and factual consistency?
- **Robustness and Real-World Transfer**: Are models tested on adversarial, naturalistic, or high-resolution real-world benchmarks beyond standard academic VQA suites?
- **Modality Interference Check**: Does the study verify preservation of text-only capabilities (e.g., language understanding, commonsense reasoning) after multimodal post-training?
- **Generative and Preference Alignment**: Is performance assessed on open-ended generation tasks (e.g., dense captioning) and human-preference alignment metrics?