**Principle 1: Controlled Ablation and Fair Comparison in Multi-Phase LLM Agent Training Pipelines**

**Definition:**  
In LLM agent research, it is common to combine supervised fine-tuning (SFT) cold-start phases with subsequent reinforcement learning (RL). However, this two-stage design introduces a significant risk of conflating the benefits of initialization with the true contributions of the RL algorithm. Reviewers must demand rigorous isolation of each phase to ensure that reported gains are attributable to the proposed method rather than to pre-training on high-quality teacher data. A fair comparison requires that baseline methods be evaluated under equivalent conditions regarding data exposure, model initialization, and computational budget. Furthermore, authors should explicitly ablate the cold-start phase by training a baseline that receives identical initialization but utilizes standard RL objectives, thereby quantifying the marginal value added by novel process-level rewards. Without such controlled experiments, claims of algorithmic superiority remain ambiguous and potentially misleading. This principle ensures that the field distinguishes genuine algorithmic innovation from implementation engineering. It also prevents hidden inductive biases introduced by format pre-training or teacher-model knowledge distillation from being mistaken for emergent RL capabilities. Ultimately, transparency in the training pipeline is essential for reproducible and trustworthy progress in agent learning.

**Core Evaluation Criteria:**
- **Phase-Isolation Ablations:** Does the work include explicit ablations that separate the cold-start/SFT contribution from the RL contribution, such as a baseline with identical initialization but standard RL rewards?
- **Baseline Parity:** Are competing methods granted equivalent access to initialization data, teacher annotations, and training compute to ensure fair comparison?
- **Marginal Gain Attribution:** Do the authors quantify the exact performance delta contributed by the novel RL component versus the initialization phase alone?
- **Inductive Bias Disclosure:** Does the work analyze whether format pre-training or tag-syntax learning accounts for a substantial portion of the observed gains?

---

**Principle 2: Algorithmic Transparency and Exact Formalization of Programmatic Process Rewards**

**Definition:**  
Dense, process-level reward shaping is a powerful tool for guiding long-horizon LLM agents, but its scientific value depends entirely on precise specification and reproducibility. Reviewers must insist that programmatic reward rules are defined with algorithmic exactitude—ideally through pseudocode or formal state-transition predicates—rather than through ambiguous natural language descriptions. Vague characterizations such as "rewarding reflection after failures" are insufficient because they conceal critical implementation details about state detection, temporal scope, and failure classification. Furthermore, true verifiability requires that rewards be computed from objective environment signals (e.g., state novelty, action validity) rather than from opaque LLM-as-a-judge models, which introduce cost, latency, and unpredictability. Exact formalization enables independent reproduction, facilitates fair comparison, and allows the community to identify hidden assumptions that might compromise validity. It also serves as a primary defense against reward hacking by making the optimization target explicit and auditable. Without this transparency, dense reward frameworks risk becoming irreproducible black boxes that cannot be built upon or rigorously evaluated.

**Core Evaluation Criteria:**
- **Precise Rule Specification:** Are all programmatic reward rules specified with sufficient detail (e.g., pseudocode, exact logical predicates, Appendix algorithms) to enable exact reproduction without guesswork?
- **Environment-Derived Verifiability:** Do the reward signals rely solely on objective, environment-provided feedback rather than subjective or model-based evaluation?
- **Reproducibility Artifacts:** Does the submission provide complete definitions of reward hyperparameters, state-transition checks, and edge-case handling?
- **Auditability of Reward Logic:** Can a third party independently verify the correctness of a reward assignment given a trajectory, and are the rules deterministic?

---

**Principle 3: Intrinsic Safeguards and Empirical Validation Against Reward Hacking in Dense Reward Shaping**

**Definition:**  
Custom dense reward designs in RL are inherently vulnerable to reward hacking, where agents exploit loopholes in the reward specification to accrue high returns without achieving the intended cognitive or task objectives. For long-horizon LLM agents, this risk is amplified because linguistic outputs offer numerous surface-level shortcuts—such as spamming meta-reasoning tags or manipulating format—that can satisfy heuristic reward functions without genuine problem-solving. Therefore, reviewers must evaluate whether the proposed reward structure incorporates intrinsic mechanism-design safeguards, such as group-relative normalization, per-tag zero-sum advantage estimation, or state-conditioned verification, that structurally disincentivize gaming. Equally important is extrinsic empirical validation: authors must demonstrate that policy improvements coincide with measurable gains in reasoning quality, such as reduced invalid action rates, shorter trajectories, or lower repetitive-loop frequency. Final task success alone is an inadequate proxy because a hacked policy might still achieve sporadic successes through brittle exploitation. Comprehensive anti-hacking analysis requires both theoretical argumentation about the reward mechanism and quantitative behavioral diagnostics that distinguish robust reasoning from exploitation.

**Core Evaluation Criteria:**
- **Structural Safeguards:** Does the reward mechanism include intrinsic properties (e.g., group-relative advantage, tag-group normalization, or state-based verification) that prevent gaming by tag spamming or format manipulation?
- **Behavioral Quality Metrics:** Beyond aggregate success rates, does the work report fine-grained process metrics (e.g., invalid/repetitive action rates, trajectory length, exploration coverage) to verify genuine cognitive improvement?
- **Failure Mode Analysis:** Does the study explicitly test for or report known hacking patterns, and does it show that the agent's internal reasoning quality improves rather than degrades?
- **Causal Link to Performance:** Is a causal or correlational argument established between the process-level reward design and the observed efficiency gains, rather than relying solely on end-task accuracy?

---

**Principle 4: Emergence Validation of Meta-Cognitive Behaviors Beyond Cold-Start Teacher Imitation**

**Definition:**  
When LLM agents are initialized via supervised fine-tuning on teacher-model annotations, there is a fundamental risk that apparent reasoning capabilities are merely memorized stylistic patterns rather than genuinely learned functional behaviors. Reviewers must therefore scrutinize whether meta-cognitive behaviors—such as reflection, replanning, or exploratory search—emerge through RL optimization or are simply imitated from the cold-start demonstrations. Strong evidence for emergence requires showing that the agent substantially exceeds the performance ceiling of the initialization phase alone, which typically achieves near-random success rates. Additionally, replacing the teacher model with a significantly weaker annotator should yield comparable final performance if the behaviors are truly reinforced by environment-driven rewards rather than distilled from teacher knowledge. Qualitative trajectory analyses should identify novel reasoning strategies, such as adaptive error recovery or context-specific replanning, that were absent from the SFT dataset. This principle ensures that the field credits RL with developing robust agency rather than attributing to it the mere refinement of pre-existing imitation policies. Distinguishing emergence from mimicry is essential for claims of learned reasoning.

**Core Evaluation Criteria:**
- **Imitation Ceiling Analysis:** Does the work report the performance of the cold-start SFT policy in isolation, and does the final RL policy substantially surpass this ceiling?
- **Teacher Model Robustness:** Are experiments conducted with alternative, weaker teacher models for initialization, demonstrating that final performance is insensitive to teacher quality?
- **Qualitative Emergence Evidence:** Are case studies or trajectory inspections provided to reveal novel behaviors (e.g., emergent reflection loops, creative exploration) not present in the SFT data?
- **Behavioral Distinction:** Does the analysis explicitly contrast SFT-generated behaviors with RL-generated behaviors to highlight functional differences beyond syntactic similarity?

---

**Principle 5: Theoretical Grounding and Cross-Environment Generalization of Meta-Reasoning Taxonomies**

**Definition:**  
Meta-reasoning reward frameworks must resist the temptation of ad-hoc benchmark engineering by grounding their taxonomies in established cognitive theory and validating them across diverse environments. A collection of heuristic tags tailored to a single benchmark risks overfitting to idiosyncratic state-transition dynamics and lacks scientific generalizability. Reviewers should expect authors to justify their choice of meta-reasoning dimensions—such as planning, monitoring, exploration, and reflection—through reference to metacognitive literature or cognitive architectures, establishing that these categories represent universal problem-solving functions rather than dataset-specific patches. Cross-environment validation on structurally distinct benchmarks (e.g., embodied household tasks versus text-based scientific experiments) is necessary to demonstrate that the reward taxonomy transfers without task-specific retuning. Furthermore, fine-grained ablations of individual meta-reasoning dimensions across multiple environments confirm that each component contributes robustly and consistently. This principle elevates process-level reward design from a bag of tricks to a systematic science of agent cognition, ensuring that innovations apply broadly across the landscape of long-horizon agent tasks.

**Core Evaluation Criteria:**
- **Cognitive/Theoretical Basis:** Is the meta-reasoning taxonomy explicitly linked to established theory (e.g., metacognition, cognitive regulation) rather than being presented as an arbitrary set of heuristics?
- **Cross-Environment Validation:** Is the framework evaluated on multiple, structurally distinct benchmarks to demonstrate transferability of the process-reward taxonomy?
- **Component Ablation Across Domains:** Are individual meta-reasoning tags or rewards ablated, and do these ablations show consistent directional effects across different task environments?
- **Task-Agnostic Reward Rules:** Are the programmatic reward rules designed to generalize using environment-agnostic signals (e.g., novelty, validity, outcome correlation) rather than benchmark-specific predicates?