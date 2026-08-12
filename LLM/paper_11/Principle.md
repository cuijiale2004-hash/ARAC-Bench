**Principle 1: Theoretical Rigor and Gradient Equivalence Proofs for Reward Distribution Matching in Long-Horizon LLM Reasoning**

**Definition:**  
The work must provide a mathematically rigorous derivation of the distribution-matching objective—such as via reverse KL divergence, energy-based modeling, or GFlowNet trajectory balance—and clearly articulate why this formulation fundamentally mitigates mode collapse in LLM reasoning compared to conventional reward-maximizing RL. The theoretical claims should be instantiated with formal proofs (e.g., gradient equivalence between KL minimization and trajectory balance) and explicitly connected to the discrete, long-horizon nature of chain-of-thought generation. This principle ensures that the shift from reward maximization to distribution matching is not merely an empirical trick but a principled algorithmic advance, and that the theoretical framework accounts for practical approximations like learnable partition functions without resorting to post-hoc rationalization.

**Core Evaluation Criteria:**
- Does the paper provide formal proofs linking the proposed objective to established frameworks such as GFlowNet trajectory balance or maximum entropy RL?
- Are the failure modes of reward maximization (mode collapse, reduced reasoning diversity) mechanistically explained in the specific context of LLM CoT reasoning?
- Is the theoretical analysis tailored to the discrete sequential setting, explicitly acknowledging challenges like intractable partition functions and long trajectories?
- Does the theoretical justification avoid circular reasoning or post-hoc fitting to empirical observations?

---

**Principle 2: Necessity and Sensitivity Validation of Learnable Partition Functions and Adaptive Normalization Modules**

**Definition:**  
Because distribution matching requires converting scalar rewards into normalized densities, the necessity of a learnable partition function and associated normalization mechanisms must be rigorously established. Reviewers scrutinize whether these components are algorithmically essential or if simpler approximations—such as constant Z, batch averaging, or complete removal—suffice, and whether design choices like MLP depth are sensitivity-tested. This principle guards against unnecessary architectural complexity masquerading as innovation, ensuring that the partition function serves a distinct theoretical role that cannot be replicated by standard entropy regularization or KL penalties. A robust submission must provide ablation evidence that Z is irreducible and that its adaptive nature is critical for matching the reward distribution across varying prompts.

**Core Evaluation Criteria:**
- Are ablations conducted showing significant performance degradation when the partition function is removed, held constant, or replaced by trivial baselines?
- Is the sensitivity of the method to Z's architecture (e.g., MLP depth, capacity) thoroughly evaluated to justify the specific design choice?
- Do the authors demonstrate that the partition function serves a distinct role that cannot be replicated by standard entropy regularization, KL penalties, or importance sampling alone?
- Is the interaction between the partition function and long-horizon training stability explicitly analyzed, rather than only reporting final accuracy metrics?

---

**Principle 3: Disentanglement of Algorithmic Gains from Response Length and Test-Time Compute Exploitation**

**Definition:**  
A critical evaluation standard is whether performance improvements stem from the algorithm's intrinsic ability to explore better reasoning paths or merely from generating longer chain-of-thought sequences that exploit test-time compute. Since distribution-matching methods may inherently encourage longer rollouts, the work must control for response length and token budget when comparing against reward-maximizing baselines. In long-CoT reasoning, longer responses often correlate with higher accuracy, making it impossible to determine true algorithmic contribution without explicit controls. Therefore, the work must analyze the length-accuracy relationship and, where possible, provide token-matched comparisons or length-normalized metrics to verify that gains are not purely artifacts of expanded inference compute.

**Core Evaluation Criteria:**
- Are comparisons performed under matched token budgets or length constraints to isolate algorithmic contributions from length exploitation?
- Is the correlation between generation length and accuracy quantified, and do advantages persist when length is normalized or constrained?
- Do training curves simultaneously track both accuracy and response length to reveal whether outperformance coincides with length divergence?
- Does the diversity analysis distinguish between substantive strategic variety and superficial length expansion or lexical variation?

---

**Principle 4: Stability Diagnostics and Catastrophic Failure Mode Analysis for Trajectory-Balance Objectives with Extended CoT**

**Definition:**  
Applying trajectory-balance or sequence-level objectives to 8K+ token reasoning introduces severe instabilities—such as gradient explosion, length collapse, and sampling mismatch—that are not present in short-trajectory GFlowNet applications. The work must provide transparent diagnostic evidence, including gradient norm traces, length evolution curves, and reward variance statistics, demonstrating that its stabilization techniques are both necessary and sufficient. Unlike short-trajectory tasks, LLM reasoning exhibits high variance in sequence length and sparse credit assignment, causing standard sequence-level gradients to scale poorly. Reviewers therefore demand direct evidence of instability without the proposed fixes, ensuring the method is a robust, scalable paradigm rather than a fragile engineering workaround held together by opaque hyperparameters.

**Core Evaluation Criteria:**
- Are training dynamics comprehensively logged, including gradient norms, response lengths, and reward distribution statistics over training steps?
- Do ablations explicitly demonstrate catastrophic training failure—such as gradient spikes, length explosion, or collapse—when stabilization components are removed?
- Is the importance sampling strategy justified as necessary for off-policy correction, with ablations comparing on-policy versus off-policy or standard PPO-style variants?
- Are stability claims validated across multiple model scales and maximum response lengths to ensure scalability?

---

**Principle 5: Cross-Domain Generalization and Qualitative Validity of Diversity Beyond Narrow Reasoning Benchmarks**

**Definition:**  
While math and code benchmarks are standard for reasoning RL, reviewers expect evidence that distribution-matching benefits generalize to broader knowledge domains—such as MMLU or GPQA—and that increased diversity does not degrade performance on conventional tasks. The work must demonstrate that exploration is productive, producing valid and qualitatively distinct solution strategies, rather than degenerate or task-specific memorization. Distribution-matching methods explicitly optimize for diversity; however, if this diversity manifests as invalid trajectories or irrelevant explorations, the method fails to advance robust reasoning. Cross-domain evaluation and human-validated diversity assessments serve as litmus tests for overfitting and confirm that the method achieves a Pareto improvement across both specialized and general knowledge benchmarks.

**Core Evaluation Criteria:**
- Are results reported on out-of-domain benchmarks to verify generalization beyond the training task distribution?
- Does diversity evaluation include human or expert judgment confirming that multiple solution paths are qualitatively distinct and valid, not just lexically varied?
- Is there evidence that diversity gains do not come at the cost of accuracy on standard benchmarks compared to strong reward-maximizing baselines?
- Are diversity metrics validated against automated judges with human agreement studies to ensure that perceived gains are reliable and reproducible?