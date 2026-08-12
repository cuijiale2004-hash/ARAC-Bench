**Principle 1: Systematic Environment Exploration and State-Space Coverage Assessment for Grounded Synthetic Task Generation in Interactive UI Agents**

**Definition:**  
The work must demonstrate that its exploration mechanism systematically uncovers the functional and state-space structure of interactive environments (e.g., mobile or desktop GUI applications) rather than performing shallow or random interaction. In synthetic task generation for agents, the exploration phase serves as the foundation for all downstream task feasibility and coverage; if the explorer fails to discover relevant UI states, functionalities, or entity relationships, the task generator will inevitably hallucinate instructions or produce tasks that are misaligned with actual environment dynamics. Reviewers expect explicit analysis—qualitative or quantitative—of how exploration trajectories cover novel states, whether exploration is conditioned on memory to avoid redundancy, and how the derived environment context grounds the task generator in observable reality rather than static LLM priors.

**Core Evaluation Criteria:**
- **Exploration Systematicity**: Does the work provide evidence that the explorer discovers diverse functionalities and UI states across multiple turns, and does it employ memory or summarization mechanisms to mitigate repetitive looping?
- **Coverage Justification**: Are there proxy metrics or analyses (e.g., unique control coverage, functional category diversity, trajectory state entropy) demonstrating broad environment coverage, even in the absence of ground-truth state-space enumerations?
- **Grounding Feasibility**: Do generated tasks exhibit significantly higher execution feasibility when conditioned on exploration trajectories compared to no-exploration or limited-context baselines, proving that exploration reduces hallucination?
- **Ablative Isolation**: Are there controlled ablations isolating the contribution of exhaustive exploration from task guideline prompts, showing that exploration itself provides non-trivial gains in downstream agent training?

---

**Principle 2: Reliability Validation and Bias Mitigation of MLLM-Based Verifiers Serving as Data Filters and Reward Models for Synthetic Agent Trajectories**

**Definition:**  
When an automated pipeline uses a multimodal large language model (MLLM) as a verifier to filter synthetic demonstrations for supervised fine-tuning (SFT) or to provide binary rewards for reinforcement learning (RL), the scientific validity of the entire pipeline hinges on the verifier’s reliability. Unvalidated verifiers introduce a single point of failure: false positives pollute the training set with incorrect trajectories, while false negatives discard useful data; optimistic bias in verifiers is particularly damaging for long-horizon tasks where partial completion is often mistaken for success. Reviewers demand rigorous human-consistency audits, precision/recall breakdowns, cross-model agreement studies, and analyses of whether the verifier shares stylistic or architectural biases with the executor or policy model.

**Core Evaluation Criteria:**
- **Human-Consensus Benchmarking**: Does the work report accuracy, precision, and recall of the MLLM verifier against a human-audited gold-standard dataset with explicit inter-annotator agreement protocols (e.g., majority voting among multiple human labelers)?
- **Failure Mode Analysis**: Are the verifier’s systematic failure patterns documented—such as optimism bias on long-horizon tasks, insensitivity to subtle UI state errors, or over-reliance on action descriptions rather than visual outcomes?
- **Model Independence and Bias Control**: Is the verifier distinct from the executor or policy model (or if shared, is the bias quantified), and are multi-judge consistency checks performed (e.g., comparing proprietary and open-source MLLM verdicts)?
- **Downstream Impact Correlation**: Does the work establish a causal or correlational link between verifier quality and downstream policy performance, demonstrating that verifier errors materially affect training outcomes?

---

**Principle 3: Alignment to Natural Task Distributions and Diversity Enforcement via Guideline-Driven Synthesis for Interactive Agent Training Data**

**Definition:**  
Synthetic task generation must produce not merely a large volume of instructions, but a distribution of tasks that is diverse across functional categories and aligned with real-world or benchmark-relevant usage patterns. In interactive agent domains, task guideline prompts are a critical lever for steering generators toward broad coverage (e.g., create, edit, delete, retrieve, compose); however, without empirical validation, guideline-driven diversity can remain a design claim rather than a demonstrated property. Reviewers evaluate whether generated tasks cover the same categorical spectrum as human-annotated benchmark tasks, whether certain capabilities are systematically underrepresented, and whether the synthesis process maintains linguistic and functional naturalism.

**Core Evaluation Criteria:**
- **Distributional Comparison**: Does the work quantitatively compare the category-level distribution of generated tasks against established human-annotated benchmarks (e.g., via automated classifiers and visualization), rather than merely asserting coverage?
- **Guideline Efficacy**: Are task guidelines ablated to demonstrate that they are necessary for diversity and downstream performance, rather than having most gains stem from prompt engineering alone?
- **Category-Level Feasibility**: Is task execution success analyzed per functional category to reveal coverage gaps (e.g., failure on cross-app navigation, complex UI interactions, or multi-hop reasoning) that indicate insufficient synthesis diversity?
- **Naturalism and Realism**: Is there evidence that generated instructions reflect realistic user requests, and are tasks filtered to exclude physically or contextually infeasible operations (e.g., requiring real-world camera capture in a simulated environment)?

---

**Principle 4: Empirical Cost-Scaling Analysis and Dataset Size Sensitivity for Automated Synthetic Demonstration Pipelines in Multimodal Agent Post-Training**

**Definition:**  
Claims of scalability in synthetic data generation must be supported by empirical measurements of computational cost, API expenditure, wall-clock time, and performance scaling as a function of dataset size. In agent training, a pipeline’s practical utility depends on whether marginal improvements in agent capability justify the linear or super-linear costs of generating additional demonstrations. Reviewers expect transparent accounting of token consumption, MLLM inference calls per task, exploration overhead, and the existence (or absence) of performance saturation curves. Without this analysis, a method risks being a small-scale proof-of-concept rather than a scalable alternative to human annotation.

**Core Evaluation Criteria:**
- **Per-Unit Cost Transparency**: Are the monetary and computational costs broken down by stage (exploration, task generation, execution, verification) with explicit token counts, API call frequencies, and per-trajectory estimates?
- **Scaling Curves**: Does the work present downstream agent performance as a function of training dataset size (e.g., subsampling experiments), demonstrating that the pipeline benefits from scale rather than saturating immediately?
- **Throughput and Bottleneck Identification**: Are wall-clock training times and throughput limitations (e.g., HTTP API latency for verifiers during RL) reported, and are cost-efficient alternatives (e.g., open-source models for verification) evaluated?
- **Cost-Utility Tradeoff**: Is the method positioned against human annotation or alternative synthetic pipelines in terms of cost-to-performance ratios, justifying its practical adoption?

---

**Principle 5: Evaluation Isolation and Cross-Environment Generalization Guarantees for Environment-Grounded Synthetic Task Datasets**

**Definition:**  
Because synthetic task generation occurs within the same simulator or application ecosystem used for downstream evaluation, the risk of task-level or state-level distribution leakage is severe. A scientifically rigorous study must enforce strict isolation between training and evaluation environments, tasks, and initial states. In UI agent research, this requires holdout application splits, deduplication of semantically similar instructions, and ideally zero-shot generalization experiments to unseen environments. Reviewers treat failure to demonstrate such isolation as a fundamental threat to validity, as agents may simply memorize training tasks rather than acquire generalizable interactive capabilities.

**Core Evaluation Criteria:**
- **Explicit Holdout Splits**: Does the work enforce app-level or environment-level holdouts where no tasks from test applications appear during training, and are zero-shot results reported separately from in-distribution results?
- **Task Deduplication**: Are semantic deduplication methods (e.g., sentence-embedding similarity thresholds) applied between generated training tasks and benchmark evaluation tasks to prevent near-identical leakage?
- **State Isolation**: Is there evidence that evaluation initial states (e.g., specific screenshots or environment configurations) are excluded from the exploration and training trajectory sets?
- **Generalization Granularity**: Are cross-task and cross-app generalization results compared against an upper bound of training directly on the test environment, allowing reviewers to assess the true transfer capability acquired from synthetic data?