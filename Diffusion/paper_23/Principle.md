
**Principle 1: Tractable Surrogate Objective Design and Tightness Verification for Intractable Log-Likelihood in Diffusion Language Model Reinforcement Learning**

**Definition:**  
Research on reinforcement learning for diffusion language models must address the fundamental intractability of log-likelihood computation, which prevents direct application of standard policy gradient estimators. The proposed surrogate objectives—whether lower bounds, upper bounds, or mixtures thereof—must be mathematically derived from first principles and their tightness relative to the true likelihood must be analyzed. Since prior work relies exclusively on the ELBO, which is only valid for non-negative rewards, new methods must demonstrate that their upper or mixed bounds correctly handle negative advantage traces without introducing excessive bias. The Monte Carlo estimation of these bounds introduces practical challenges regarding sample efficiency and variance, which must be theoretically characterized and empirically validated. Furthermore, the relationship between bound tightness and model behavior—such as whether a loose upper bound still provides useful gradient signals for penalizing poor outputs—should be carefully examined. Ultimately, the validity of the surrogate objective determines whether the RL algorithm is optimizing a meaningful proxy for the true expected reward.

**Core Evaluation Criteria:**
- **Mathematical Soundness**: Are the upper/lower bounds correctly derived from variational principles for the specific masked diffusion formulation, and are constant terms properly handled in gradient computation?
- **Tightness Characterization**: Is the tightness of the bounds analyzed under relevant conditions (e.g., order-consistency), and are limitations honestly reported rather than obscured?
- **Monte Carlo Efficiency**: Does the work analyze how the number of samples affects estimator variance, optimization stability, and end-task performance?
- **Bias-Variance Tradeoff**: Is the mixture objective theoretically justified with proofs of variance reduction, and are gradient norm dynamics empirically validated across training steps?

---

**Principle 2: Rigorous Evaluation Protocol and Checkpoint Selection to Prevent Test-Set Overfitting in Iterative RL Fine-Tuning**

**Definition:**  
Iterative reinforcement learning fine-tuning creates unique evaluation risks because models are continuously updated and evaluated against fixed test sets, enabling implicit overfitting through checkpoint selection and hyperparameter tuning. A rigorous evaluation protocol must ensure that the reported results reflect genuine generalization rather than cherry-picking the best moment during training. This requires checkpoint selection based on a held-out validation set that is distinct from the final test set, or transparent reporting of both best-checkpoint and final-checkpoint performance. Hyperparameters specific to the proposed method must not be tuned against the test set, as this creates an unfair advantage over baseline methods. The evaluation protocol should be applied consistently across all compared methods, including baseline algorithms, to control for inherent variance in RL training dynamics. Without such methodological safeguards, apparent performance gains may simply reflect test-set contamination rather than true algorithmic advances.

**Core Evaluation Criteria:**
- **Validation-Based Selection**: Are checkpoints selected using a held-out validation set rather than test-set accuracy maximization across training steps?
- **Hyperparameter Independence**: Are method-specific hyperparameters (e.g., mixture coefficients, bound tightness parameters) tuned without access to test-set performance?
- **Protocol Uniformity**: Is the identical evaluation procedure applied to all baselines to ensure fair comparison given RL training variance?
- **Final-Step Transparency**: Are final-checkpoint results reported alongside best-checkpoint results to demonstrate robustness to training dynamics?

---

**Principle 3: Comprehensive Baseline Coverage Across Diverse RL Paradigms for Diffusion Language Models**

**Definition:**  
The landscape of RL algorithms for diffusion language models encompasses diverse methodological families, including policy gradient methods with one-step or Monte Carlo likelihood estimation, preference optimization approaches adapted from DPO, and trajectory-level algorithms that accumulate gradients across denoising steps. A thorough empirical evaluation must situate the proposed method against representatives from each relevant paradigm, not merely against the most accessible or recently published baselines. This includes reimplementing or faithfully comparing against trajectory-level methods, score-function estimators, and variance-reduced preference optimization when they represent the state of the art. Controlled ablations are necessary to disentangle the contribution of the core algorithmic innovation from auxiliary techniques such as masking strategies or inference configurations. Fair comparison also requires documenting and controlling for differences in experimental conditions, including fine-tuning scope, reward function definitions, and generation hyperparameters. Only through comprehensive baseline coverage can reviewers assess whether observed improvements are attributable to the claimed technical contribution or to confounding experimental factors.

**Core Evaluation Criteria:**
- **Paradigm Diversity**: Does the comparison include trajectory-level methods, preference optimization, and alternative likelihood estimators (e.g., score-matching, pathwise surrogates)?
- **State-of-the-Art Inclusion**: Are the strongest existing methods in the subfield included, even if reimplementation is required?
- **Implementation Fidelity**: Are baseline differences in fine-tuning type, reward functions, and generation setup documented and controlled?
- **Component Isolation**: Do ablations clearly separate the core algorithmic contribution from masking strategies, inference configurations, and other auxiliary techniques?

---

**Principle 4: Theoretical Grounding and Stability Justification for Simplified Policy Gradient Formulations**

**Definition:**  
Standard RL algorithms for language models rely on stabilization mechanisms—such as PPO clipping, KL divergence regularization, and importance sampling ratio corrections—to prevent policy collapse and control off-policy bias during iterative updates. When proposing a simplified policy gradient formulation for diffusion language models, researchers must explicitly justify the omission of these mechanisms, particularly because the intractable log-likelihood complicates their direct application. The underlying decision process should be formally specified, including how partially masked sequences constitute states and how the reverse denoising process defines actions and transitions. If multiple gradient updates are performed per rollout, the resulting off-policyness must be analyzed, and the absence of importance sampling corrections must be theoretically or empirically defended. Known biases in advantage estimation, such as those introduced by group-relative baselines, should be acknowledged and ideally mitigated. Without this level of theoretical and empirical scrutiny, claims of stable training rest on weak foundations.

**Core Evaluation Criteria:**
- **MDP Formalization**: Is the Markov decision process formally defined with clear states, actions, and transition dynamics for the diffusion setting?
- **Stabilization Justification**: Is the omission of clipping, KL regularization, or trust regions explained by fundamental intractability or merely empirical sufficiency?
- **Off-Policy Analysis**: Are the off-policy effects of multiple updates per rollout analyzed, and are missing importance sampling corrections justified?
- **Baseline Bias Mitigation**: Are known biases in advantage estimation (e.g., GRPO baseline dependence on sampled actions) acknowledged and addressed?

---

**Principle 5: Cross-Domain Empirical Validation and Robustness Across Task Categories and Inference Strategies**

**Definition:**  
Empirical validation of diffusion LLM RL methods should extend beyond a narrow domain such as mathematical reasoning to demonstrate broad applicability across coding, logical reasoning, and potentially multi-turn or tool-use scenarios. Since the base model's capabilities and the reward landscape differ substantially across task types, domain-specific success does not guarantee general algorithmic efficacy. Robustness must also be verified under diverse inference strategies—including varying decoding block sizes, confidence-based versus random unmasking, and full-sequence generation—to ensure the policy does not overfit to a specific inference-time configuration. Beyond final accuracy, training dynamics such as reward convergence curves, gradient norm stability, and effective generation length trajectories provide critical diagnostic information about optimization behavior. Computational efficiency, measured by wall-clock time per update and token throughput relative to baselines, is essential for assessing practical scalability. Comprehensive empirical scope ensures that the method represents a genuine advance for diffusion language model RL rather than an overfitted solution to a restricted benchmark suite.

**Core Evaluation Criteria:**
- **Task Category Breadth**: Are experiments conducted across reasoning, coding, and other domains to verify general applicability beyond a single task type?
- **Inference Robustness**: Is the method evaluated under diverse decoding strategies to rule out inference-specific overfitting?
- **Training Diagnostic Metrics**: Are reward curves, gradient norms, and effective length dynamics reported to diagnose optimization stability?
- **Computational Cost Analysis**: Is wall-clock time and throughput compared against baselines under equal resource budgets?