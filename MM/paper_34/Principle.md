**Principle 1: Rigorous Quality Auditing and Leakage/Bias Control for Auto-Generated Fine-Grained Pretraining Data**

**Definition:**  
In the absence of large-scale human-annotated fine-grained datasets, vision encoder pretraining increasingly relies on auto-generated region-level annotations produced by strong proprietary or open-source MLLMs. However, this pipeline introduces critical risks regarding annotation noise, semantic misalignment, and circular dependency if the same model family is later used for evaluation or downstream adaptation. Reviewers must assess whether authors conduct multi-dimensional quality audits—combining automatic metrics such as QA correspondence, relevance, and visual dependence with human spot-checking—to validate the pretraining corpus. Equally important is the need to identify and mitigate potential data leakage or stylistic bias arising from distributional overlap between data generators and downstream consumers. Without such scrutiny, reported gains may reflect dataset artifacts rather than genuine representational improvements. This principle is foundational because the quality ceiling of the resulting vision encoder is bounded by the fidelity of its pretraining signals. Consequently, community trust in large-scale auto-generated datasets hinges on transparent, reproducible auditing protocols.

**Core Evaluation Criteria:**
- **Multi-Level Quality Verification**: Does the work present both automatic (model-based) and human evaluations of auto-generated annotations, with explicit scoring rubrics and distributional analyses?
- **Leakage and Bias Mitigation**: Are potential biases from using the same model family for annotation generation and downstream evaluation explicitly discussed and empirically controlled (e.g., via cross-family validation)?
- **Filtering and Curation Transparency**: Are the filtering criteria (e.g., aspect ratio, bounding box area, image resolution) and data sources systematically documented, and is the impact of filtering on dataset composition analyzed?
- **Dataset Scale vs. Quality Trade-off**: Does the work justify the chosen scale by demonstrating that annotation quality is maintained at scale, rather than assuming monotonic gains from more data?

---

**Principle 2: Cross-Architecture Transferability and Distributional Independence of Visual Representations**

**Definition:**  
A core value proposition of a general-purpose vision encoder is its ability to produce visual representations that transfer robustly across diverse LLM architectures, tokenization schemes, and pretraining distributions. If performance gains vanish or diminish significantly when paired with an LLM from a different family than used during pretraining, the encoder may be learning stylistic or distributional shortcuts rather than generic visual concepts. Reviewers therefore expect rigorous cross-architecture transfer experiments, where the vision encoder is paired with fundamentally different LLMs without extensive re-tuning of the pretraining recipe. This evaluation is crucial because the multimodal ecosystem depends on modular components that can be swapped and upgraded independently. An encoder that only works well within a closed ecosystem provides limited scientific or practical utility. Moreover, transferability tests serve as a guardrail against implicit overfitting to the pretraining LLM's semantic space or output style.

**Core Evaluation Criteria:**
- **Cross-Family LLM Evaluation**: Are results reported on LLMs with distinct architectures, pretraining data, and tokenizers (not merely different sizes within the same family)?
- **Disentanglement of Gains**: Does the analysis demonstrate that improvements are attributable to the visual encoder's representational quality rather than compatibility with a specific LLM's generation style or feature space?
- **Specialist vs. Generalist Comparisons**: When claiming generalization, does the work compare against both task-specific specialists and generalist encoders to show breadth without sacrificing depth?
- **Robustness to Adaptation Paradigms**: Is the encoder tested under varying adaptation protocols (e.g., frozen vs. tuned vision backbone) to verify that gains are not conditional on a specific optimization setup?

---

**Principle 3: Task-Specific Efficacy versus General Capability Trade-off and Avoidance of Overclaiming**

**Definition:**  
Enhancing fine-grained perception in vision encoders often involves architectural or pretraining trade-offs that may not uniformly benefit all downstream tasks. For instance, prioritizing bounding-box regression and OCR grounding can marginally reduce performance on holistic reasoning benchmarks that depend more on global scene understanding or LLM-internal knowledge. Reviewers must evaluate whether authors honestly characterize these trade-offs—disaggregating results by task category such as fine-grained localization, OCR, VQA, and reasoning rather than masking weaknesses behind aggregated averages. Furthermore, claims of state-of-the-art performance must be supported by comparisons against the most relevant and strongest baselines, including recent specialist models and generalist vision encoders. Overclaiming by narrowly selecting comparison points or omitting failure modes undermines scientific credibility. This principle demands that papers define the operational envelope of their contribution: what tasks are improved, what tasks are neutral or degraded, and why.

**Core Evaluation Criteria:**
- **Disaggregated Performance Reporting**: Are results broken down by capability category rather than presented only as a single averaged metric that obscures task-specific regressions?
- **Baseline Completeness and Fairness**: Does the comparison include strong contemporary generalist encoders and, where relevant, task-specific specialists under comparable training and inference settings?
- **Honest Limitation Disclosure**: Are failure cases, performance regressions, and out-of-domain degradation explicitly reported and analyzed with qualitative examples?
- **Scalability Analysis**: Does the work examine whether task-specific gains hold, diminish, or invert as the LLM backbone scales, avoiding cherry-picked model sizes?

---

**Principle 4: Experimental Rigor in Multi-Stage Pretraining Design and Computational Efficiency Justification**

**Definition:**  
Modern vision encoder pretraining for MLLMs increasingly employs multi-stage pipelines (e.g., pretraining the encoder with a frozen LLM, then adapting the LLM with a frozen encoder) and auxiliary objectives such as self-distillation. The scientific validity of such pipelines depends on ablating the necessity, ordering, and interaction of these stages. Reviewers expect evidence that each stage and hyperparameter contributes measurably to the final outcome, and that seemingly arbitrary choices such as freezing the vision encoder during adaptation are justified by empirical efficiency metrics rather than ad hoc assumptions. Justifications must extend beyond single-GPU FLOP counts to encompass realistic distributed-training overheads such as gradient synchronization latency, peak GPU memory, and wall-clock time. Without this rigor, the community cannot assess whether reported performance stems from a genuine methodological advance or from increased computational budget and engineering complexity.

**Core Evaluation Criteria:**
- **Stage Necessity and Ordering Ablations**: Are alternative pipeline configurations (e.g., reversed stage order, merged end-to-end training, omitting distillation) systematically compared to validate the proposed design?
- **Hyperparameter Sensitivity**: Are key hyperparameters (e.g., distillation coefficients, loss balancing weights) ablated over a meaningful range, with clear selection criteria?
- **Realistic Efficiency Metrics**: Are computational claims supported by measurements of wall-clock time, peak memory, and multi-node communication overhead, not just theoretical FLOPs or parameter counts?
- **Cost-Benefit Analysis of Freezing vs. Tuning**: Is the decision to freeze or tune specific modules justified by a quantitative performance-efficiency frontier under realistic training scales?

---

**Principle 5: Qualitative and Mechanistic Evidence for Fine-Grained Localization Claims**

**Definition:**  
Quantitative benchmarks alone are insufficient to validate claims of enhanced fine-grained perception, as aggregate metrics can obscure whether a model truly utilizes localized visual features or relies on linguistic priors and global context. Reviewers therefore expect detailed qualitative analyses, including attention-map visualizations on complex scenes such as multi-object, text-heavy, or cluttered environments, side-by-side success and failure cases against strong baselines, and diagnostic studies of localization behavior. Such evidence must be methodologically sound: normalization procedures should be consistent across compared models, attention maps should be filtered to reduce noise, and failure modes should be categorized (e.g., small objects, overlapping instances, coordinate representation limitations). This principle ensures that purported advances in fine-grained understanding are grounded in mechanistic insight rather than statistical happenstance.

**Core Evaluation Criteria:**
- **Complex-Scene Visualization**: Are attention maps and qualitative examples drawn from challenging scenarios (dense text, multiple overlapping objects, cluttered backgrounds) rather than trivial single-object images?
- **Normalized and Controlled Comparisons**: Are visualization protocols (e.g., attention thresholding, normalization, color scaling) held constant across all compared methods to prevent artifactual differences?
- **Failure Case Taxonomy**: Does the work provide a structured analysis of failure modes with concrete examples, and propose plausible explanations linked to architectural or representational limitations?
- **Correlation between Attention and Performance**: Is there evidence that improved localization behavior (e.g., tighter attention on queried regions) correlates with quantitative gains on fine-grained benchmarks?