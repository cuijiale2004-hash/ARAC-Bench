**Principle 1: Theoretical Grounding and Decomposition Justification for Non-Differentiable Ranking Reward Design in LLM RL**

**Definition:**  
In listwise reinforcement learning for ranking tasks, reviewers must scrutinize whether the proposed reward structure is a principled mathematical decomposition of the target non-differentiable metric—such as breaking Spearman correlation into listwise, pairwise, and pointwise sub-objectives—rather than an ad-hoc aggregation of heuristic signals. It is crucial that normalization constants, binary thresholds, and weighting schemes are either derived from data statistics or formally justified as invariances that preserve the optimal policy. Furthermore, the work must rigorously justify the shift from differentiable surrogates (e.g., Pearson correlation loss) to RL by demonstrating that optimizing the reasoning process itself yields superior alignment with the final ranking objective beyond mere scalar regression. This perspective prevents reward hacking and ensures that the RL paradigm is chosen for genuine structural reasons rather than novelty. Without such grounding, the method risks being indistinguishable from empirical tuning masked as algorithmic innovation.

**Core Evaluation Criteria:**
- **Mathematical Decomposition**: Is the reward function explicitly derived from or proven equivalent to components of the target metric (e.g., Spearman decomposition), with each term's purpose explained?
- **Justification over Differentiable Baselines**: Does the work compare against strong regression or contrastive baselines with differentiable ranking proxies under equal data and compute to isolate the value of RL-based reasoning alignment?
- **Absence of Arbitrary Design**: Are hyperparameters and constants (e.g., max difference, binary thresholds) grounded in dataset properties or theoretical necessity rather than intuition alone?
- **Theoretical Consistency**: Does the reward shaping preserve the optimal policy, and is there analysis showing why the shaping does not introduce spurious local optima or reward hacking?

---


**Principle 2: Cross-Domain and Cross-Scale Generalization Beyond Task-Specific Annotation Artifacts**

**Definition:**  
Research proposing novel ranking mechanisms for LLMs must demonstrate that the core innovation is not tightly coupled to the idiosyncrasies of a single benchmark, such as a specific Likert scale range, predetermined adjacent pair structures, or cross-sample batching conventions. Reviewers should evaluate whether the method generalizes empirically to disparate label scales (e.g., continuous 0–100), alternative data organizations (e.g., per-input candidate sets versus parallel slices), and distinct semantic domains. This requires out-of-domain validation on tasks that share the overarching ranking objective but differ fundamentally in structure and annotation protocol. Without such evidence, claims of a generalizable framework remain weak, and the work risks being classified as dataset-specific engineering rather than a transferable algorithmic contribution. Consequently, the paper must either provide broad empirical validation or explicitly delimit its scope to avoid misleading the community about the method's portability.

**Core Evaluation Criteria:**
- **Label Scale Independence**: Is the method validated on at least one task with a substantially different label distribution or scale, proving it is not reliant on task-specific normalization constants?
- **Structural Flexibility**: Does the work empirically test or theoretically justify the mechanism's adaptability to alternative list constructions (e.g., per-query local ranking vs. global cross-sample slicing)?
- **Out-of-Domain Benchmarking**: Are there experiments on a separate domain (e.g., translation quality estimation for a semantic similarity paper) that isolate the mechanism from benchmark-specific artifacts?
- **Scope Delimitation**: If generalization is limited, does the paper explicitly bound its applicability scope and avoid overstating universality in the abstract and conclusions?

---


**Principle 3: Disentanglement of Method Efficacy from Backbone Scale and Proprietary Reasoning Baselines**

**Definition:**  
When RL frameworks are applied to LLMs for ranking tasks, it is essential to isolate whether performance improvements originate from the proposed algorithmic design or merely from the increased capacity of generative backbones relative to traditional discriminative models. Reviewers must demand parameter-parity experiments using compact models (e.g., sub-1B parameters) compared against similarly sized discriminative baselines, as well as scaling analyses across model sizes to confirm consistent gains. Concurrently, the work must be benchmarked against state-of-the-art proprietary reasoning models (e.g., GPT-4o, DeepSeek-R1) to establish that specialized RL alignment can outperform general-purpose few-shot reasoning on fine-grained quantization tasks. This dual validation ensures that the contribution is a true algorithmic advance rather than an artifact of scale or backbone architecture. Absent such evidence, the perceived superiority of the proposed framework may simply reflect the well-known scaling laws of autoregressive LLMs rather than novel training methodology.

**Core Evaluation Criteria:**
- **Parameter-Parity Comparisons**: Does the work evaluate small-scale generative models (e.g., 0.6B) against comparable discriminative SOTA to prove method efficacy independent of scale?
- **Scaling Law Consistency**: Do results show monotonic improvement across multiple backbone sizes, confirming stable optimization without scale-dependent collapse?
- **Proprietary Model Headroom**: Are comparisons included against leading proprietary reasoning models under few-shot/CoT settings to quantify the gap and demonstrate specialization value?
- **Attribution Clarity**: Does the analysis explicitly decompose gains into backbone capability versus alignment/training methodology?

---


**Principle 4: Robustness Validation Against Label Noise and Dataset Shift in Fine-Grained Semantic Ranking**

**Definition:**  
Fine-grained semantic annotation tasks are inherently susceptible to label noise and subjective bias, which can mislead optimization and inflate performance metrics artificially. Therefore, reviewers should require evaluation on independently re-annotated or cleaned datasets to verify that improvements are robust and not byproducts of overfitting to noisy or inconsistent labels. The work should also assess stability under distribution shifts, such as changes in annotator guidelines or condition descriptions. In addition, the analysis should demonstrate whether the model's error distribution shifts toward fewer severe deviations on cleaned data, confirming genuine reasoning refinement rather than noise exploitation. This principle ensures that the reported gains reflect genuine advances in conditional reasoning and ranking alignment rather than spurious correlations with artifact-ridden training signals.

**Core Evaluation Criteria:**
- **Re-Annotated Evaluation**: Is the method evaluated on a recently cleaned or re-annotated version of the benchmark to ensure gains persist after label correction?
- **Noise Sensitivity Analysis**: Does the work provide empirical or theoretical analysis of how label noise affects the RL reward signal and whether the curriculum mitigates noisy credit assignment?
- **Stable Performance Delta**: Do the relative improvements over baselines remain consistent across original and cleaned datasets, ruling out noise exploitation?
- **Qualitative Error Analysis**: Are failure modes analyzed to show reduction in high-error predictions rather than mere aggregate metric inflation?

---


**Principle 5: Granularity and Empirical Validation of Progressive Curriculum Stages for Listwise Credit Assignment**

**Definition:**  
Listwise RL objectives are notoriously difficult to optimize directly due to sparse, coarse-grained signals that hamper credit assignment across multiple completions. A progressive curriculum—from pointwise foundations to hybrid listwise refinement—must be rigorously validated as a necessity rather than a convenience. Reviewers must examine whether each stage and each reward component is isolated through ablations to prove its marginal contribution, and whether the proposed credit assignment mechanism (e.g., parallel slice ranking) demonstrably outperforms naive batch-level rewards. Additionally, the work should justify hyperparameters governing the curriculum transition, such as generation multiplicity and slice size, with sensitivity analyses that link these choices to exploration diversity versus signal stability. Ultimately, the curriculum design must be shown to prevent training collapse and enable monotonic progression toward fine-grained ranking proficiency, not merely incremental metric gains.

**Core Evaluation Criteria:**
- **Stage Necessity Ablations**: Are both curriculum stages and their constituent rewards (pointwise, pairwise, listwise) individually ablated to prove that neither stage alone suffices?
- **Credit Assignment Granularity**: Is there evidence that the proposed fine-grained mechanism (e.g., per-completion rewards within parallel slices) outperforms coarse batch-level ranking signals?
- **Hyperparameter Sensitivity**: Are key architectural choices (e.g., slice size N, generation multiplicity G) systematically explored with performance curves, and are trade-offs (exploration vs. compute) explicitly discussed?
- **Convergence Stability**: Does the work demonstrate that the curriculum prevents training collapse or oscillation compared to direct listwise RL initialization?