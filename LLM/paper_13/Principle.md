Principle 1: Unsupervised Discovery and Theoretical Justification of Lottery-Sample Subsets for RLVR

**Definition:**  
The identification of critical training instances without ground-truth annotations represents a pivotal challenge in scaling Reinforcement Learning with Verifiable Reward (RLVR) for large language models, as complete dataset annotation imposes prohibitive costs and computational inefficiency. This principle demands that researchers articulate and validate a lottery-sample hypothesis—namely, that a tiny subset of the training data can match full-dataset performance—through both empirical demonstration and theoretical argumentation. The evaluation focuses on whether the proposed selection mechanism operates in a strictly unsupervised manner prior to annotation, and whether the resulting subset achieves comparable policy optimization outcomes with dramatically reduced annotation budgets. Furthermore, the work must demonstrate that the selected samples genuinely drive reasoning improvement rather than merely exploiting spurious correlations or dataset biases. Ultimately, this criterion distinguishes methods that enable annotation-efficient RLVR from standard active-learning approaches that rely on labeled data or surrogate losses.

**Core Evaluation Criteria:**
- **Lottery Hypothesis Validation**: Does the work empirically demonstrate that training on the selected subset achieves comparable performance to full-dataset training with minimal annotation (e.g., <0.5% of data), and is this validated across multiple independent trials?
- **Unsupervised Integrity**: Does the selection mechanism operate entirely without ground-truth answers, reward signals, or loss gradients during the discovery phase, clearly distinguishing it from supervised core-set or data-valuation methods?
- **Annotation Efficiency**: Are annotation budgets tested at scales that demonstrate practical cost reduction, and is the performance gap to full-dataset training quantified across multiple budgets (e.g., 4 vs. 8 instances)?
- **Policy Optimization Alignment**: Does the work verify that selected instances improve the RLVR policy specifically, rather than merely improving supervised fine-tuning or generic classification metrics?

---

Principle 2: Verifiability of Epsilon-Approximation Bounds and Gradient Proximity in Subset Selection

**Definition:**  
Theoretical guarantees in data subset selection for LLM policy optimization must transcend heuristic conjectures by establishing verifiable connections between the selection criterion and the optimization landscape. This principle scrutinizes whether the proposed method's approximation bounds—such as ε-approximate lottery-sample assumptions—are empirically measurable rather than purely abstract, and whether the selection score correlates with gradient proximity to the full-dataset optimum. Reviewers must assess if the analysis introduces novel theoretical constructs appropriate to the RLVR setting, such as ergodic Markov properties and mixing time characterizations, or merely repackages standard optimization assumptions. Crucially, the work should avoid relying solely on final performance improvements to infer that the selected subset satisfies the theoretical approximation conditions. A rigorous submission must either bound the actual gradient error or provide empirical diagnostics that validate the tightness of the theoretical approximation in practice.

**Core Evaluation Criteria:**
- **Empirical Verifiability of Assumptions**: Does the work measure or bound the actual ε (gradient approximation error) empirically, or does the bound remain purely theoretical and unobservable?
- **Gradient Proximity Justification**: Is there a rigorous, demonstrated connection between the sample selection score (e.g., conformal set size) and gradient similarity to the full-dataset optimum, supported by decomposition or diagnostic analysis?
- **Theoretical Novelty in RLVR Context**: Do the proofs introduce domain-specific constructs (e.g., ergodic MDP mixing time, GRPO generalization error) beyond standard convex optimization assumptions like PL conditions and L-smoothness?
- **Avoidance of Circular Reasoning**: Does the work avoid inferring ε-approximation validity solely from final task accuracy, instead providing independent evidence that the selected subset preserves the optimization trajectory?

---

Principle 3: Disentanglement of Reasoning Volatility from Linguistic Noise in Answer Space Evaluation

**Definition:**  
In evaluating uncertainty-based sample selection for reasoning models, it is essential to disentangle meaningful logical uncertainty from superficial linguistic stochasticity inherent in autoregressive language generation. This principle requires that volatility measures—whether procedural or outcome-based—reflect genuine reasoning complexity and multi-step inference fragility rather than stylistic variation, decoding temperature effects, or token-level unpredictability. The evaluation must specifically examine whether the method remains informative in constrained answer spaces, such as multiple-choice or discrete-output tasks, where outcome volatility may artificially collapse and fail to discriminate difficult instances. Additionally, the work should provide empirical evidence across harder scientific or mathematical domains where linguistic fluency and logical correctness diverge significantly. Without such disentanglement, selection mechanisms risk prioritizing linguistically noisy but logically trivial samples, undermining the data-efficiency objective.

**Core Evaluation Criteria:**
- **Separation of Noise and Reasoning**: Does the method include mechanisms to distinguish linguistic stochasticity (e.g., stylistic decoding variation) from substantive reasoning uncertainty (e.g., logical step fragility)?
- **Small Answer Space Validity**: Are experiments conducted on discrete or multiple-choice tasks where outcome volatility may be artificially compressed, and does the method adapt to remain discriminative (e.g., by removing choices during volatility computation)?
- **Hard Domain Robustness**: Does the evaluation extend beyond standard math benchmarks to harder scientific domains (e.g., astronomy, solid-state chemistry) where linguistic noise and reasoning difficulty are more likely to decouple?
- **Ablation of Volatility Components**: Are procedural and outcome volatility shown to capture distinct, non-overlapping aspects of uncertainty, with ablations demonstrating their complementary contributions?

---

Principle 4: Robustness and Sensitivity Analysis of Conformal Calibration and Clustering Configurations

**Definition:**  
Conformal prediction serves as a powerful uncertainty-quantification tool for LLM reasoning, yet its practical deployment in data-selection pipelines demands rigorous characterization of robustness to design choices and edge-case behaviors. This principle evaluates whether the method maintains stable performance across varying calibration set sizes, supports transfer across distinct datasets or distributions, and avoids degenerate behaviors such as prediction-set collapse in tiny answer spaces. The clustering mechanism used to enforce diversity must be fully specified and ablated, including embedding choices, distance metrics, and sensitivity to the number of clusters relative to the annotation budget. Furthermore, the dynamic behavior of conformal sets—how their cardinality adapts to question difficulty—must be analyzed to ensure it meaningfully guides sample selection rather than introducing arbitrary thresholds. Robustness in these dimensions determines whether the conformal framework provides a reliable foundation for unsupervised critical-instance discovery.

**Core Evaluation Criteria:**
- **Calibration Set Sensitivity**: Is the method's performance characterized across varying calibration set sizes (e.g., 256 to 1024), and is transfer across heterogeneous datasets (e.g., MMLU to BigMath) demonstrated?
- **Clustering Configuration Ablations**: Are alternative clustering strategies (number of clusters, instances per cluster) systematically evaluated to isolate diversity effects from pure uncertainty effects?
- **Conformal Set Behavior Analysis**: Does the work analyze whether prediction sets collapse or become uninformative in tiny answer spaces, and are dynamic set sizes shown to correlate with question difficulty?
- **Statistical Guarantee Practicality**: Are the exchangeability and coverage assumptions of conformal prediction explicitly stated and justified for the LLM reasoning setting, including calibration cost accounting?

---

Principle 5: Comprehensive Baseline Coverage Across Model Scales and Reasoning Domains for Data Selection

**Definition:**  
A convincing contribution in reasoning-specific data selection must be evaluated against a comprehensive battery of baselines that spans generic active learning, reasoning-specific heuristics, and intrinsic model likelihoods across diverse model scales and problem domains. This principle mandates comparisons not only with standard active-learning strategies such as entropy sampling and BADGE, but also with RLVR-specific selection criteria like self-consistency filtering, entropy-weighted sampling, and log-probability scoring. The experimental protocol must validate generalization across model architectures and scales—from small distilled models to larger instruction-tuned variants—and across reasoning domains ranging from competition mathematics to undergraduate science problems. Moreover, the work should investigate whether the selection strategy supports iterative or active-loop refinement, where the evolving policy re-selects training instances across rounds. Only through such multidimensional validation can a method claim broad applicability in LLM reasoning pipelines.

**Core Evaluation Criteria:**
- **Baseline Comprehensiveness**: Does the evaluation include both generic active-learning baselines (BADGE, EntSampling) and reasoning-specific strategies (SCF, EWS, log-probability scoring)?
- **Cross-Scale Model Validation**: Are results reported across multiple model scales (e.g., 1.5B, 7B, 8B parameters) to demonstrate scalability and architecture independence?
- **Cross-Domain Generalization**: Does the experimental suite span diverse reasoning domains beyond mathematics, such as physics, chemistry, or astronomy, to validate domain-agnostic selection?
- **Active-Loop and Iterative Capability**: Does the work investigate whether the selection criterion remains effective in iterative or active-loop settings where the policy evolves and re-selects instances over multiple rounds?