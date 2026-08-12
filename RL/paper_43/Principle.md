**Principle 1: Theoretical Validity and Training Stability of Partition-Function Approximations in Last-Token Self-Rewarding**

**Definition:**  
In methods that derive self-rewarding signals from last-token probability distributions under KL-regularized RL objectives, reviewers must scrutinize whether key theoretical approximations—such as treating the partition function as unity, collapsing reference-model log-probabilities to a pre-computed constant, or expanding the optimal policy form—remain valid and numerically stable throughout policy optimization. These approximations are foundational because they convert a complex joint RL objective into a tractable MSE loss; if they break down as the policy drifts from the reference, the entire self-rewarding mechanism becomes unreliable. A rigorous evaluation demands both formal justification for the simplifications and empirical tracking of approximation errors across training iterations. Furthermore, reviewers assess whether minor derivation errors undermine the theoretical claims or can be corrected without altering the method's core conclusion. The stability of these approximations directly determines whether the self-rewarding score retains its semantic meaning as a proxy for verifier rewards after extensive policy updates. Ultimately, the work must demonstrate that the approximations are not merely convenient assumptions for initialization but hold robustly across the non-stationary distribution induced by RL training.

**Core Evaluation Criteria:**
- **Empirical Stability of Constants**: Does the work track the dynamics of reference log-probabilities or partition-function values across training steps to verify that treating them as constants does not introduce bias or variance as the policy updates?
- **Mathematical Rigor**: Are derivations free of algebraic errors (e.g., correct handling of one-hot reward terms in partition sums), and are the conditions under which approximations hold explicitly stated?
- **Sensitivity Analysis**: Does the paper ablate the impact of replacing the exact reference term with its approximated constant on both reasoning accuracy and self-verification performance?
- **Robustness to Policy Drift**: Is there evidence that the approximations remain valid when the policy model diverges substantially from the reference model, or under different hyper-parameter regimes (e.g., KL coefficient)?

---

**Principle 2: Domain Generalization and Cross-Task Transferability Beyond Mathematical Reasoning Benchmarks**

**Definition:**  
For joint reasoning-verification training methods, the evaluation must extend well beyond the primary training domain—typically mathematical reasoning—to demonstrate that the learned self-rewarding capability is not an artifact of narrow task-specific optimization. Reviewers expect evidence of generalization to diverse domains such as general science QA, code generation, and open-ended instruction following, as well as zero-shot transfer to out-of-distribution benchmarks. The core concern is whether the self-rewarding signal captures universal markers of solution quality or merely overfits to verifiable math formats, which directly determines the method's practical utility in real-world deployment where verifiable rewards are sparse or unavailable. When cross-domain evaluation is performed, reviewers look for nuanced analysis of why performance gaps exist, such as base model capability limitations or noisy verifier signals, rather than superficial reporting of accuracy metrics. The principle further requires that inference-time scaling benefits, such as weighted majority voting gains, be shown to persist across domains rather than vanishing outside the training distribution. A method that claims to unify reasoning and verification must prove that its self-evaluation mechanism is domain-agnostic enough to support continual model improvement in heterogeneous environments.

**Core Evaluation Criteria:**
- **Breadth of Evaluation Domains**: Are experiments conducted on non-math tasks (e.g., code synthesis, commonsense reasoning, multimodal QA) to verify cross-domain applicability?
- **Out-of-Distribution Generalization**: Does the paper report zero-shot or few-shot performance on held-out benchmarks after training on a specific domain, and analyze whether self-verification F1 remains stable under distribution shift?
- **Mechanism Analysis for Domain Gaps**: When generalization is limited, does the work provide a mechanistic explanation (e.g., weaker base reasoning capability in non-math domains, noisy external verifiers) rather than presenting superficial accuracy numbers?
- **Inference-Time Scaling Across Domains**: Does the method improve weighted majority voting or Best-of-N selection consistently across diverse domains, or are gains confined to math tasks?

---

**Principle 3: Calibration Quality and Discriminative Validity of Auxiliary Self-Rewarding Signals for Solution Ranking**

**Definition:**  
The utility of a learned self-rewarding score hinges not merely on its ability to predict correctness, but on its calibration as a probabilistic or ordinal confidence metric that can reliably rank candidate solutions and drive test-time scaling strategies. Reviewers assess whether the score distributions for correct and incorrect solutions are well-separated and whether the scores are properly bounded, interpretable, and free from length bias or arbitrary thresholding artifacts. High F1 or accuracy alone is insufficient; the signal must behave like a meaningful confidence measure with low Expected Calibration Error (ECE) and high AUROC to support downstream applications such as weighted voting and early stopping. The principle demands that the relationship between self-rewarding scores and empirical correctness probabilities be explicitly characterized, ensuring that a score of 0.5 or any chosen threshold corresponds to a genuinely uncertain prediction rather than an arbitrary cutoff. Additionally, reviewers examine whether the scoring mechanism inadvertently encodes superficial heuristics such as response length or formatting rather than deep semantic correctness. Proper calibration validation separates methods that produce genuine self-evaluative signals from those that merely learn spurious correlations with verifier labels.

**Core Evaluation Criteria:**
- **Beyond-Accuracy Metrics**: Does the evaluation include calibration metrics such as AUROC, ECE, and Spearman correlation with response length, rather than relying solely on F1 or accuracy?
- **Score Distribution Analysis**: Are the distributions of self-rewarding scores for correct versus incorrect solutions visualized and analyzed, and is the boundedness or unboundedness of scores explicitly justified?
- **Threshold Rationality**: If a classification threshold (e.g., 0.5) is used, is its choice theoretically grounded given the score's definition (e.g., scaled log-probability), and does the paper demonstrate that scores naturally concentrate in interpretable ranges after optimization?
- **Length-Bias Control**: Does the work investigate and mitigate potential biases where response length artificially inflates or deflates the self-rewarding score, ensuring that the signal reflects solution quality rather than verbosity?

---

**Principle 4: Component Attribution through Ablation and Fair Comparative Evaluation against Externally Trained Verifiers**

**Definition:**  
Because joint optimization frameworks often introduce multiple auxiliary techniques simultaneously—such as class-balanced re-weighting, separate warm-up phases, self-rewarding-based advantage integration, and reference-model simplification—reviewers require meticulous ablation studies to attribute observed gains to specific design choices. Furthermore, when comparing against external reward models or alternative self-verification paradigms, the baselines must be trained on identical data distributions and model backbones to ensure fairness. The principle emphasizes that aggregate performance improvements cannot be accepted at face value; each component must justify its inclusion by demonstrating an isolated, positive contribution to either reasoning quality, verification accuracy, or training stability. Reviewers also expect comparisons against structurally related loss formulations, such as BCE or SFT objectives, to validate that the chosen MSE framework is not merely equivalent to simpler alternatives but offers distinct advantages in optimization dynamics. Without such granular attribution, it remains impossible to determine whether the reported gains stem from algorithmic novelty or from the cumulative effect of standard engineering tricks. Fair and comprehensive ablation practices are therefore essential for establishing the scientific value of the proposed method over its individual parts and prior art.

**Core Evaluation Criteria:**
- **Systematic Component Ablations**: Are all key technical components (advantage integration, loss re-weighting, warm-up stages, reference simplification) individually ablated to quantify their marginal contributions?
- **Fair Baseline Construction**: Are external verifiers or prior self-verification methods (e.g., ReVIsE) compared using the same backbone architecture, training data, and computational budget?
- **Alternative Loss Formulations**: Does the paper compare the chosen MSE loss against structurally related alternatives (e.g., BCE/SFT losses) under controlled settings to justify the design choice?
- **Training Dynamics Transparency**: Are training curves (rewards, self-verification F1, loss terms) reported in full to reveal instability, collapse, or saturation caused by specific components?

---

**Principle 5: Quantitative Validation of Near-Zero Overhead Claims in Inference-Time Self-Verification and Test-Time Scaling**

**Definition:**  
A central claim of lightweight self-rewarding methods is their near-zero computational overhead relative to standard generation and external verification, and reviewers evaluate this claim by requiring precise measurements of inference-time cost. Such measurements must include additional token count, latency per sample, and KV-cache utilization, alongside explicit training-time overhead quantification. The principle holds that efficiency gains must be benchmarked against concrete baselines, including sequential solution-and-verification generation and separately trained reward models, rather than asserted qualitatively. Without empirical cost validation, the practical advantage of the method over existing joint training or external-verifier paradigms remains speculative and potentially misleading. Reviewers further examine whether the method achieves superior accuracy per unit inference compute during test-time scaling procedures like weighted majority voting or Best-of-N selection. Ultimately, the burden of proof lies on the authors to demonstrate that the efficiency-accuracy Pareto frontier is genuinely shifted, not that accuracy is purchased through hidden computational expenditure.

**Core Evaluation Criteria:**
- **Token-Level Cost Quantification**: Does the paper report the exact additional number of tokens or forward passes required per sample during inference compared to sequential self-verification and standard generation?
- **Latency and Throughput Benchmarks**: Are wall-clock latency measurements or throughput comparisons provided to validate claims of minimal overhead, accounting for KV-cache reuse?
- **Training Overhead Analysis**: Is the added training cost of the MSE loss term quantified in terms of GPU hours, memory footprint, or convergence speed relative to vanilla RLVR?
- **End-to-End Scaling Efficiency**: Does the work demonstrate that the method achieves superior performance per unit inference compute (e.g., accuracy vs. total generated tokens) during test-time scaling procedures like weighted majority voting or Best-of-N?