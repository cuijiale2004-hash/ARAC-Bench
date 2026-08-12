**Principle 1: Anti-Gaming Validation and Human-Proxy Alignment of Verifiable Rewards in Open-Ended Vision-Language Generation**

**Definition:**  
In open-ended multimodal generation tasks such as image captioning, the design of a verifiable reward function—often derived from a downstream proxy task like visual question answering—must be rigorously validated to ensure it does not incentivize reward hacking or metric exploitation. A high-quality caption should not merely maximize proxy scores by injecting task-specific cues, verbose artifacts, or unnatural phrasing tailored to the verifier; instead, it must preserve human-aligned properties including fluency, succinctness, balanced detail coverage, and factual grounding. The burden of proof lies on the authors to disentangle genuine quality improvements from superficial proxy optimization through multi-dimensional evidence, such as double-blind human evaluations, automated caption-specific metrics, and qualitative failure analysis. Reviewers must scrutinize whether the reported gains reflect broadly useful descriptions or merely exploitation of the decoupled verifier's blind spots. Without such validation, claims of superior generation quality remain suspect, as the model may have learned to game the metric rather than acquire robust visual understanding. This principle is foundational because the core premise of RLVR in subjective domains rests entirely on the trustworthiness and human-alignment of the reward signal.

**Core Evaluation Criteria:**
- **Anti-Gaming Evidence**: Does the work provide direct empirical evidence—beyond downstream accuracy—that the generator does not exploit the verifier, for instance by producing VQA-biased cues, repetitive patterns, or length-tuned outputs that mislead the proxy?
- **Proxy-Human Correlation**: Are quantitative correlations established between the proxy reward and independent human judgments or established caption-quality benchmarks across multiple dimensions (detail, accuracy, fluency)?
- **Mechanistic Attribution**: Does the analysis explicitly distinguish performance gains attributable to genuine informativeness from those attributable to output formatting or exploitation of verifier blind spots?

---

**Principle 2: Fairness and Structural Bias Control in Decoupled Caption-Quality Evaluation Protocols**

**Definition:**  
Research that employs decoupled evaluation frameworks—where generated captions are assessed by feeding them into a fixed, vision-free question-answering module—must carefully control for structural biases inherent to the protocol. Because the proposed generator may be explicitly reinforcement-learned to optimize performance within this exact disentangled pipeline, fair comparison demands that baselines be evaluated under identical pipeline conditions or supplemented with independent, end-to-end benchmarking. The experimental design should transparently justify the decoupled setup as a probe for caption informativeness rather than conflating it with real-world downstream system performance. Furthermore, the work should demonstrate that observed gains are not artifacts of the evaluation framework by validating them through standard pretraining paradigms and diverse downstream benchmarks. This principle ensures that the scientific contribution reflects true advances in generation quality rather than evaluation-format overfitting. Ultimately, reviewers should demand evidence that the method improves model capabilities generally, not just scores within a bespoke, favorable evaluation structure.

**Core Evaluation Criteria:**
- **Protocol Fairness**: Does the evaluation avoid giving the proposed method an inherent structural advantage, and are comparisons conducted under strictly matched pipeline conditions (e.g., identical LLM answerers, identical prompt formats)?
- **Independent Benchmarking**: Are findings corroborated by complementary, standard evaluations—such as pretraining on synthetic captions and testing on broad VQA benchmarks—that are decoupled from the training-time proxy metric?
- **Disentanglement Justification**: Is the rationale for using a disjoint caption→QA setup clearly articulated, with explicit warnings against interpreting it as equivalent to end-to-end model capability?

---

**Principle 3: Quantitative Hallucination and Factuality Verification for Synthetic Multimodal Annotations**

**Definition:**  
When a method claims to improve the accuracy and reliability of synthetic image captions or multimodal annotations, it must provide quantitative evidence specifically targeting hallucination reduction and factuality preservation. General downstream performance gains are insufficient proxies, as they can mask the presence of confabulated details that happen not to penalize the specific benchmark in use. The work should report dedicated hallucination metrics—such as POPE, CHAIR, or specialized factuality scores—at both the generator level and, when applicable, for models pretrained on the synthetic corpus. Additionally, the analysis should investigate whether the generation process introduces systematic biases toward certain error modes, such as object hallucination or attribute misassignment. Adherence to this principle is essential for establishing trust in synthetic data pipelines intended for large-scale model pretraining. Reviewers should treat the absence of such targeted metrics as a critical weakness in any work that positions itself as advancing factual multimodal generation.

**Core Evaluation Criteria:**
- **Dedicated Hallucination Metrics**: Are specific, recognized hallucination or factuality benchmarks reported (e.g., POPE, ALOHa, CHAIR), rather than inferring factuality solely from task accuracy?
- **Multi-Level Evaluation**: Is hallucination analyzed both for the caption generator itself and for downstream models trained on the synthetic dataset, ensuring fidelity propagates through the pretraining pipeline?
- **Error Mode Analysis**: Does the work qualitatively and quantitatively characterize the types of hallucinations that persist, offering insight into the limitations of the proposed reward?

---

**Principle 4: Robustness and Sensitivity Analysis of Verifier Models and Filtering Protocols in RLVR Pipelines**

**Definition:**  
Training frameworks that rely on external verifiers—such as LLM-based QA judges or LVLM-based data filters—must demonstrate robustness to the choice, scale, and quality of these verifier components. The research should empirically establish whether the method is brittle to weaker or smaller verifier models, whether performance saturates beyond a certain scale, and how sensitive training is to the strictness of data curation protocols. Moreover, because verifier-centric pipelines introduce additional computational stages, the work must provide transparent cost-benefit analyses, measuring wall-clock overhead, throughput, and resource consumption relative to standard training baselines. This principle prevents the community from adopting methods that are implicitly tied to proprietary, oversized judges or hyper-specific filtering thresholds that cannot be replicated. A thorough sensitivity analysis also reveals whether the method extracts genuine signal from noisy or weakly grounded supervision, which is crucial for real-world deployment. Reviewers must therefore verify that the pipeline's efficacy holds across a realistic range of verifier capabilities and computational budgets.

**Core Evaluation Criteria:**
- **Verifier Scale Sensitivity**: Are systematic ablations presented across verifier model sizes and families (e.g., sub-1B to 30B+ parameters) to identify capability thresholds and saturation points?
- **Filtering Protocol Robustness**: Does the method maintain meaningful gains when trained with imperfectly filtered or weakly grounded training signals, and is the impact of strict versus loose filtering quantified?
- **Computational Efficiency**: Is the temporal and hardware overhead of the verification stage explicitly reported (e.g., seconds per RL step, GPU-hours, throughput) and shown to be marginal or at least justified?

---

**Principle 5: Scaling Behavior and Cross-Domain Generalization of Sparse Supervision Signals in Multimodal Data Synthesis**

**Definition:**  
Methods that leverage sparse, verifiable supervision—such as a handful of multiple-choice questions per image—to drive large-scale multimodal data synthesis must provide rigorous evidence of scaling laws, data efficiency, and cross-domain transfer. The work should demonstrate that increasing the volume of sparse supervision or synthetic captions yields monotonic, predictable improvements rather than diminishing returns or collapse. It should also validate that models trained on domain-restricted supervision generalize to visually distinct domains, confirming that the learned capability is not narrowly memorized. Finally, the efficiency of sparse signals should be explicitly ablated to establish that dense human-like annotations are not required to unlock significant model potential. This principle is critical for justifying the practical deployment of RLVR-based annotation pipelines at industrial scale. Without evidence of broad generalization and favorable scaling, a method risks being dismissed as a narrow, overfitted trick rather than a scalable scientific advance.

**Core Evaluation Criteria:**
- **Scaling Law Evidence**: Does performance improve reliably as the amount of QA training data or synthetic captions increases (e.g., from 1M to 5M), with clear trends rather than isolated points?
- **Cross-Domain Generalization**: Are models trained on a restricted image domain shown to improve on out-of-domain benchmarks, proving that caption-quality gains transfer beyond the training distribution?
- **Sparsity and Efficiency**: Is the sufficiency of sparse supervision explicitly tested (e.g., one versus three QA pairs per image), and are gains replicated across diverse base architectures (different LLMs and visual encoders)?