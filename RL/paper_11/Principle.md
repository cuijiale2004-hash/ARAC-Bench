**Principle 1: Rigorous End-to-End Bias Quantification in Asynchronous Group-Level Importance Sampling Estimators**

**Definition:**  
In decentralized reinforcement learning for large language models, importance sampling estimators must operate under severe policy staleness caused by network latency, which exponentially inflates variance. The field increasingly embraces intentionally biased estimators—such as group-level expectation weighting—that sacrifice unbiasedness for variance reduction. However, reviewers must rigorously evaluate whether the introduced bias remains bounded and does not overwhelm the optimization objective. A high-quality submission must provide both theoretical upper bounds on the bias (e.g., via L2-norm ratios or distributional statistics) and empirical measurements of end-to-end bias relative to standard unbiased estimators. It should also characterize the mean squared error trade-off across realistic ranges of KL divergence and group sizes, demonstrating that the variance reduction dominates any bias penalty. Without such quantification, claims of "stable bias" are merely heuristic and cannot distinguish principled algorithmic advances from ad-hoc variance suppression.

**Core Evaluation Criteria:**
- **End-to-end bias measurement**: Does the work quantify bias relative to standard importance sampling using actual training batches, rather than relying solely on asymptotic arguments?
- **Theoretical bias bound**: Are explicit bounds provided (e.g., L2-norm ratio, Rényi divergence) that remain tight under realistic LLM policy concentration?
- **MSE characterization**: Is the mean squared error decomposed and analyzed across relevant operational ranges of KL divergence and group sizes?
- **Comparative ablation**: Does the work show that the biased estimator achieves net optimization benefit over unbiased alternatives under identical staleness conditions?

---

**Principle 2: Theoretical Consistency Under Non-Normalized Empirical Sampling in Long-Tail LLM Output Distributions**

**Definition:**  
Standard theoretical tools in reinforcement learning assume normalized probability distributions over discrete action spaces, but modern LLM policies employ top-k and top-p sampling that produces empirical probability vectors summing to much greater than one when samples are drawn with replacement. This creates a critical gap between theoretical guarantees and practical implementation that reviewers must scrutinize carefully. A rigorous submission must explicitly clarify whether its proofs concern the underlying policy distribution or the empirical sampled distribution, and justify how the two relate in the proof construction. Furthermore, because LLM output distributions are highly concentrated and long-tailed over massive vocabularies, theoretical constants—such as those bounding variance or bias—must be shown to be non-vacuous under these realistic conditions. Failure to address this distinction undermines confidence that the algorithm will behave as predicted when deployed with actual model sampling strategies. Theoretical claims must therefore be validated against the empirical distributional properties observed in real language model generation.

**Core Evaluation Criteria:**
- **Distribution clarity**: Does the work explicitly distinguish between the underlying normalized policy and empirical sampling with replacement in its theoretical analysis?
- **Non-vacuous constants**: Are theoretical constants (e.g., variance bound offsets) estimated using real model distributions and shown to be reasonable rather than catastrophically loose?
- **Scalability with vocabulary/sequence length**: Does the analysis address how constants and guarantees behave as sequence length increases or output distributions sharpen toward one-hot vectors?
- **Empirical validation of assumptions**: Are distributional assumptions (e.g., bounds on squared L2-norms) verified with measurements from actual LLM checkpoints during training?

---

**Principle 3: Comprehensive Baseline Fairness Including Enhanced Variants and System-Algorithm Separation**

**Definition:**  
Evaluating novel asynchronous policy optimization algorithms requires far more than comparing against canonical but unenhanced baselines such as vanilla GRPO or GSPO. Reviewers must assess whether the authors have conducted a fair contest by including strengthened variants of competing methods—such as trust-region filtering, defensive sampling, or adaptive clipping—as well as recent specialized asynchronous algorithms like CISPO, TOPR, or truncated importance sampling schemes. Additionally, it is crucial to separate algorithmic contributions from system-level optimizations; comparing a new algorithm against an entire system framework conflates orthogonal improvements and obscures true methodological progress. Fairness also demands that all methods operate under identical staleness constraints and comparable hyperparameter search budgets. Only through such comprehensive and carefully controlled comparisons can the community determine whether reported gains stem from a genuinely superior algorithmic mechanism or from unfair experimental advantage.

**Core Evaluation Criteria:**
- **Enhanced baseline inclusion**: Are strengthened variants of direct competitors (e.g., GSPO with trust region, GRPO with stronger clipping) evaluated under identical latency conditions?
- **Recent asynchronous methods**: Does the comparison include contemporary asynchronous policy optimization algorithms rather than only synchronous or classic baselines?
- **System vs. algorithm separation**: Does the work avoid conflating system-level throughput optimizations with algorithmic stability improvements?
- **Hyperparameter parity**: Are all baselines afforded comparable tuning effort under the same maximum delay and staleness constraints?

---

**Principle 4: Hyperparameter Robustness Validation Under Stochastic Network Delay and Resource Heterogeneity**

**Definition:**  
Algorithms targeting heterogeneous reinforcement learning environments must demonstrate that their stability is robust across the configuration space, not merely achievable at a single tuned point. High network latency and computational heterogeneity introduce non-stationary optimization dynamics that interact unpredictably with hyperparameters such as group size, KL regularization coefficients, and sampling temperatures. Reviewers should therefore expect thorough sensitivity analyses that sweep these hyperparameters under diverse delay distributions—such as lognormal, Weibull, and exponential—reflecting realistic wide-area network conditions. A method that collapses under slightly suboptimal KL coefficients or specific delay profiles reveals fragile engineering rather than a principled solution. Consequently, high-quality research in this subfield must map the stability landscape across hyperparameters and latency regimes, providing practical guidance for deployment in unpredictable decentralized environments.

**Core Evaluation Criteria:**
- **Group size ablation**: Is the method's stability evaluated across multiple group sizes under fixed delay conditions to confirm statistical regularization benefits?
- **Delay distribution sweep**: Are experiments conducted with different stochastic delay models rather than a single simplistic latency simulation?
- **Sampling and regularization grid**: Does the work report sensitivity to temperature, top-k/top-p, and KL coefficients specifically in delayed settings?
- **Stability generalization**: Does the method maintain performance across a plausible range of configurations, or is stability confined to a narrow hyperparameter tube?

---

**Principle 5: Mechanistic Empirical Validation of Variance Reduction Through Gradient Dynamics and Failure Diagnosis**

**Definition:**  
Theoretical guarantees of variance reduction in importance weights are insufficient on their own; reviewers must verify that these mathematical properties translate into tangible improvements in training dynamics and model behavior under latency stress. This requires empirical evidence linking lower importance-weight variance to smoother gradient norms, more stable reward curves, and avoidance of catastrophic failure modes such as repetitive token loops, incoherent reasoning chains, or sudden reward collapse. Submissions should include mechanistic analyses—such as gradient comparisons across tokens or sequences and case studies of model outputs under high delay—that illuminate how the proposed method stabilizes learning. Additionally, theoretical bound constants should be measured against actual training distributions to confirm they are empirically meaningful rather than vacuously large. By demanding this connection between abstract variance bounds and observable optimization behavior, reviewers ensure that proposed methods address the real symptom of policy staleness rather than optimizing a surrogate metric.

**Core Evaluation Criteria:**
- **Gradient variance dynamics**: Are gradient norms plotted and correlated with importance weight variance over the course of training?
- **Failure mode catalog**: Does the work diagnose and document specific pathological behaviors (repetition, collapse, gibberish) induced by high latency in baseline methods?
- **Case study depth**: Are qualitative examples provided that trace how the proposed mechanism alters token-level or sequence-level updates compared to baselines?
- **Bound constant measurement**: Are the empirical values of theoretical constants reported from training logs to demonstrate that bounds are non-vacuous in practice?