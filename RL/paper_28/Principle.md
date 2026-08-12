**Principle 1: Systematic Bias Mitigation Verification for Positional, Verbosity, and Self-Preference in RL-Trained Tool-Augmented Judges**

**Definition:**  
In tool-augmented judge training via reinforcement learning, reviewers must rigorously evaluate whether the optimization process amplifies or mitigates known cognitive biases inherent in LLM evaluators. The assessment should examine positional bias through bidirectional A-B/B-A testing and averaging, verbosity bias via accuracy stratification across response-length disparities, and self-preference bias by ensuring evaluation benchmarks exclude responses from the backbone or affiliated models. This perspective is crucial because RL can inadvertently reward superficial heuristics rather than genuine reasoning quality, and tool integration does not automatically eliminate these artifacts. High-quality research must provide pre- and post-training bias metrics, demonstrating that gains in accuracy do not come at the cost of increased reliance on positional or length artifacts. Furthermore, the evaluation must distinguish between performance improvements on biased subsets versus unbiased subsets to confirm robust calibration.

**Core Evaluation Criteria:**
- **Pre/Post Bias Metrics**: Does the work quantify positional, verbosity, and self-preference biases before and after RL training, with explicit mitigation strategies?
- **Bidirectional Robustness**: Are pairwise evaluations conducted in both orderings (A-B and B-A) with discrepancy analysis, rather than relying on single random ordering?
- **Bias-Accuracy Trade-off Analysis**: Does the work verify that accuracy gains persist when evaluated on bias-controlled or bias-balanced subsets?
- **Backbone Independence**: Are evaluation responses drawn from models temporally and architecturally distinct from the judge's backbone to prevent self-preference contamination?

---

**Principle 2: Controlled Disentanglement of End-to-End RL Gains from Prompt-Based Tool Use and Supervised Fine-Tuning Baselines**

**Definition:**  
A central challenge in this subfield is isolating the marginal contribution of end-to-end RL training from simpler interventions such as prompt-based tool invocation, format specification, or supervised fine-tuning on tool-use trajectories. Reviewers must assess whether the experimental design includes fair baselines that consume identical training data and tool interfaces, and whether ablations explicitly separate SFT-only performance from SFT-plus-RL performance. This is vital because naive tool-augmented baselines may fail simply due to incorrect formatting or lack of exposure to tool-use data, rather than a genuine absence of tool reasoning capability. The principle ensures that claimed RL benefits reflect actual policy optimization of reasoning and tool-invocation strategies, rather than artifacts of data curation or prompt engineering. Without this disentanglement, the community cannot determine whether the proposed pipeline is necessary or merely a composite of existing techniques.

**Core Evaluation Criteria:**
- **SFT-Only Ablation**: Is there a tool-augmented judge baseline trained with supervised fine-tuning alone on the same data to isolate the incremental effect of RL?
- **Equivalent Data Exposure**: Are baseline models fine-tuned with comparable quantities and quality of tool-integrated data, or merely prompted at inference time?
- **Format vs. Reasoning Decomposition**: Does the work decompose error types (e.g., format errors, syntax errors, runtime errors) to prove gains stem from reasoning rather than format compliance?
- **Protocol Alignment**: For cited baseline results, is there evidence that identical evaluation protocols, preprocessing, and decontamination standards were strictly applied?

---

**Principle 3: Triangular Generalization Validation Across Verifiable Domains, Model Architectures, and Heterogeneous Tool Ecosystems**

**Definition:**  
Tool-integrated judge frameworks are often at risk of being overfit to specific task types (e.g., mathematical reasoning), single model families (e.g., Qwen), or a particular tool implementation (e.g., Python execution). This principle demands evidence that the training recipe generalizes across the verifiability spectrum—from precisely checkable tasks like code execution to subjective domains like safety and helpfulness—across diverse backbone architectures, and ideally toward multi-tool scenarios. The rationale is that a robust framework should function as a general methodology rather than a narrow, domain-specific recipe. Reviewers must scrutinize whether observed performance gains are concentrated in "low-hanging fruit" verifiable tasks or extend to nuanced, non-verifiable judgments. Additionally, cross-architecture validation prevents conflating framework efficacy with idiosyncratic inductive biases of a specific model series.

**Core Evaluation Criteria:**
- **Verifiable-Non-verifiable Balance**: Does performance improve or at least not degrade on non-verifiable tasks (chat, safety) when training emphasizes tool-use domains?
- **Cross-Architecture Transfer**: Are experiments conducted on at least one distinct model family to verify that the training recipe is portable and not backbone-dependent?
- **Tool Extensibility**: Does the framework design theoretically or empirically support extension to heterogeneous tools (search, SQL, symbolic solvers) beyond the primary tool?
- **Scaling with Complexity**: Do gains persist or increase on harder tasks (e.g., GPQA, competitive programming) rather than plateauing on easy or saturated benchmarks?

---

**Principle 4: Structural and Mechanistic Explanation of Tool-RL Efficacy Specific to Judgment and Verification Functions**

**Definition:**  
Unlike policy models that generate creative content, judges must verify, compare, and score existing outputs—a function that imposes distinct computational and epistemic demands. This principle evaluates whether the work provides mechanistic or theoretical justification for why coupling tool use with RL is structurally advantageous for evaluators, rather than simply demonstrating empirical improvement. The focus should be on explaining the verification bottleneck: judges often need to simulate or validate correctness, which is computationally harder than generation for small models, making external executors particularly valuable. Additionally, the work should articulate why iterative RL converges reliably in judgment tasks (e.g., deterministic binary rewards) and how the reward design shapes emergent tool-invocation strategies. Research lacking this depth risks being perceived as an arbitrary concatenation of existing techniques without domain-specific insight.

**Core Evaluation Criteria:**
- **Judge-Specific Motivation**: Does the work clearly articulate why verification is structurally harder than generation for the target model scale, motivating tool integration beyond generic "tools help reasoning" claims?
- **Convergence Analysis**: Is there empirical or theoretical analysis of why iterative RL stabilizes (e.g., rejection sampling filtering, deterministic reward signals) rather than diverging or reward hacking?
- **Emergent Strategy Explanation**: Does the work explain how the model learns *when* to invoke tools versus rely on internal knowledge, and what role reward shaping plays?
- **Differentiation from Policy TIR**: Is there explicit analysis of how tool use in judges differs fundamentally from tool use in generator policies?

---

**Principle 5: Rigorous Statistical Validation and Multi-Seed Variance Reporting for Comparative Claims in Iterative Judge Training**

**Definition:**  
In iterative judge training, especially when comparing resource-intensive paradigms like distillation-free RL versus teacher-distilled RL, small performance differences can be statistically fragile or an artifact of random initialization and decoding stochasticity. This principle mandates that comparative claims—particularly those asserting equivalence or superiority of training paradigms—be supported by variance estimates, confidence intervals, or significance tests across multiple random seeds or stochastic decoding settings. It is insufficient to report single-run point estimates, as the community cannot assess whether a 1-2% margin reflects a genuine methodological advantage or statistical noise. Moreover, the evaluation must transparently distinguish between deterministic greedy decoding and stochastic sampling regimes where variance is meaningful. Adherence to this standard ensures that decisions about whether distillation is truly unnecessary are grounded in reproducible evidence rather than anecdotal performance snapshots.

**Core Evaluation Criteria:**
- **Variance Quantification**: Are results reported with standard deviations, confidence intervals, or error bars derived from multiple random seeds or stochastic runs?
- **Significance Testing**: For key comparative claims (e.g., zero-distillation ≥ distilled), are p-values, bootstrap intervals, or paired t-tests provided?
- **Decoding Transparency**: Does the work clearly state whether evaluations use deterministic (temperature=0) or stochastic decoding, and justify the chosen statistical methodology accordingly?
- **Cost-Accuracy Trade-off Rigor**: When claiming a training paradigm is "viable" or "efficient," is there quantified analysis of compute cost, wall-clock time, and teacher API expenses alongside performance metrics?