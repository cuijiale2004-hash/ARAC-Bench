Principle 1: Cross-Domain Synergy and Principled Composition of Embodied Procedural Training Data

**Definition:**
For datasets assembled from heterogeneous embodied scenarios spanning physical robotics, navigation, digital interfaces, and structured assembly, evaluation must assess whether the domain combination is architected around a unified theoretical objective rather than convenience. The principle examines whether cross-domain training yields transferable procedural abstractions or merely aggregates isolated, domain-specific skills. It demands evidence that diverse sources collectively teach an abstract "procedural grammar" applicable to both digital and physical embodiment, rather than shallow visual correlations. This perspective is crucial because opportunistic data aggregation can inflate scale without improving generalization, whereas principled composition can enable emergent cross-domain reasoning essential for deployable embodied AI.

**Core Evaluation Criteria:**
- **Principled Domain Selection**: Are the data domains justified by a unified capability goal (e.g., general procedural reasoning across physical and digital embodiment), or do they appear loosely connected?
- **Cross-Domain Ablation Evidence**: Does the work provide controlled ablations showing that mixed-domain training consistently outperforms single-domain training?
- **Emergent Capability Analysis**: Is there evidence that models learn abstract procedural structure rather than domain-specific shortcuts or visual correlations?

---

Principle 2: Active Cross-Modal Validation via Adversarial Negative Sampling in Temporal Reasoning

**Definition:**
In temporal and procedural understanding, models must be evaluated on their ability to actively validate and reject logically inconsistent sequences rather than passively recognize positive patterns. This principle centers on whether training paradigms incorporate adversarial negative samples that compel cross-modal verification between visual sequences and textual descriptions, thereby preventing over-reliance on language priors or positional heuristics. It demands evidence that models transition from surface-level pattern matching to robust causal reasoning about action sequences. The inclusion of challenging negatives is particularly vital in multimodal settings where text biases often dominate visual evidence, and without explicit rejection training, models may appear competent while lacking genuine procedural comprehension.

**Core Evaluation Criteria:**
- **Negative Sample Rigor**: Are negative samples genuinely challenging and semantically plausible, requiring fine-grained cross-modal validation rather than trivial rejection?
- **Mechanism Validation**: Does the work demonstrate through ablation that negative samples materially improve performance, and that gains stem from enhanced reasoning rather than format tuning?
- **Task Complementarity**: Are multiple temporal reasoning tasks (e.g., ordering, prediction, review) designed to jointly cultivate comprehensive procedural understanding rather than isolated skills?

---

Principle 3: RL Training Stability and Format-Reward Overfitting Avoidance in Multimodal Fine-Tuning

**Definition:**
When reinforcement learning is applied to multimodal fine-tuning for procedural reasoning tasks, evaluation must rigorously distinguish genuine reasoning acquisition from superficial reward hacking, particularly format overfitting. This principle demands empirical evidence that RL training achieves stable convergence without collapse or oscillation, and that performance gains significantly exceed those attainable through supervised fine-tuning alone. It further requires isolating the contribution of core task rewards from auxiliary format-specific rewards to prove that capabilities stem from logical reasoning rather than output structure mimicry. This scrutiny is essential because RL in multimodal contexts is susceptible to shortcut learning where models exploit reward sparsity or prompt structure, producing brittle behaviors that fail to generalize.

**Core Evaluation Criteria:**
- **RL vs. Supervised Baseline**: Is there a direct comparison showing RL significantly outperforms SFT under controlled, identical conditions?
- **Training Stability Evidence**: Are reward curves, gradient norms, or variance across seeds reported to demonstrate stable convergence without collapse or oscillation?
- **Format Reward Ablation**: Does the work isolate the contribution of core task rewards from format-specific rewards to prove reasoning is not driven by output structure mimicry?

---

Principle 4: Out-of-Distribution Generalization and Benchmark Diversity for Temporal Understanding Claims

**Definition:**
Claims of enhanced temporal and procedural understanding require rigorous out-of-distribution validation beyond custom, in-house test sets that mirror training distributions. This principle mandates evaluation on diverse, established external benchmarks featuring distinct formats, task framings, and underlying logic to ensure that gains reflect transferable skill acquisition rather than memorization or distribution-specific overfitting. It also requires careful disaggregation of sub-task performance to verify that specialization aligns with targeted capabilities and to identify potential regressions in unrelated dimensions. Such comprehensive validation is indispensable because small, manually curated test sets—while useful for measuring direct training efficacy—can be insufficient to certify broad generalization, a critical criterion for datasets claiming to advance foundational reasoning.

**Core Evaluation Criteria:**
- **External Benchmark Diversity**: Are results reported on multiple independent benchmarks with different data distributions and task formats?
- **In-Distribution vs. OOD Gap Analysis**: Does the work analyze performance gaps between proprietary test sets and public benchmarks to detect overfitting?
- **Sub-Task Disaggregation**: Are improvements disaggregated by sub-task to verify specialization aligns with claimed capabilities and to identify regressions in unrelated reasoning dimensions?

---

Principle 5: Upstream Model Bias Quantification and Human Verification in Automated Dataset Curation

**Definition:**
Automated dataset construction pipelines that leverage large upstream models for filtering, annotation, and quality control risk inheriting and amplifying systematic biases at scale. This principle evaluates whether the work explicitly quantifies and mitigates these risks through rigorous human verification on representative samples, detailed error taxonomy analysis, and strict de-duplication protocols. It requires demonstrating that automated filters achieve high precision and recall across failure modes such as temporal misordering, description hallucination, and visual incoherence. This scrutiny is fundamental because unverified automation can propagate hidden errors and modality biases into training data, undermining the validity of downstream performance claims and threatening the dataset's integrity as a community resource.

**Core Evaluation Criteria:**
- **Human Verification Scale and Rigor**: Is human validation performed on a statistically meaningful random sample with reported inter-annotator agreement or cross-review mechanisms?
- **Error Taxonomy and Bias Analysis**: Are failure modes of the automated pipeline categorized (e.g., temporal misordering, description hallucination) and quantified?
- **Leakage Prevention Protocol**: Are de-duplication strategies explicitly detailed at both source and content levels to prevent train-test overlap and semantic contamination?