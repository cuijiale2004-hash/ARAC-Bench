
**Principle 1: Theoretical Justification and Gradient Recovery Analysis for Inpainting-Based Guided Exploration in Zero-Advantage Regimes of Diffusion LLMs**

**Definition:**  
The work must provide rigorous theoretical analysis that connects the unique architectural capability of masked diffusion LLMs—specifically inpainting via bidirectional attention—to concrete optimization benefits in reinforcement learning. This includes formalizing how partial ground-truth injection restores non-degenerate gradients in zero-advantage regimes where standard group-based policy optimization yields zero learning signals. The analysis should establish provable policy improvement guarantees under mixture-policy interpretations and quantify the distributional shift induced by hint injection through KL divergence bounds or trust-region constraints. Reviewers must assess whether the theoretical claims are novel and specific to the inpainting mechanism, rather than generic RL results repackaged, and whether they explain empirical phenomena such as the superiority of partial over full hint injection. The theoretical framework should ideally bridge architectural inductive biases with optimization dynamics, distinguishing the contribution from both standard supervised fine-tuning and conventional policy gradient methods.

**Core Evaluation Criteria:**
- **Gradient Recovery Rigor**: Does the work formally prove that the proposed mechanism restores non-zero gradients in all-wrong, zero-advantage scenarios, and does the analysis specify how gradient magnitude scales with injection parameters?
- **KL Control and Stability Bounds**: Are there provable bounds on the distributional shift caused by partial ground-truth injection, and do they explain the empirical stability of training?
- **Architectural Specificity**: Is the theoretical analysis fundamentally tied to the inpainting/bidirectional capabilities of diffusion LLMs, or could it apply generically to any model with external hint injection?
- **Distinction from Baselines**: Does the theory clearly differentiate why partial hint injection outperforms full supervision or pure RL, beyond empirical observation?

---

**Principle 2: On-Policy Validity Verification and Controlled Distribution Shift Assessment in Partial Ground-Truth Injection RL Frameworks**

**Definition:**  
The work must rigorously justify why introducing partial ground-truth tokens into an ostensibly on-policy reinforcement learning framework does not violate the foundational assumptions of the underlying policy gradient algorithm. This requires both empirical diagnostics—such as importance-ratio clipping statistics, token-level entropy measurements, and KL divergence from the reference policy—and theoretical arguments showing that the off-policy deviation is bounded and beneficial. Reviewers should evaluate whether the method maintains near-on-policy behavior in practice, whether the majority of generated tokens remain policy-sampled, and whether safeguards like entropy-based gradient filtering are theoretically motivated rather than post-hoc patches. The justification must address why the deviation from strict on-policy sampling is necessary and controlled, reframing the issue as guided exploration rather than invalid optimization.

**Core Evaluation Criteria:**
- **Empirical On-Policy Diagnostics**: Are clipping rates, KL divergences, and token-generation statistics reported to verify that hint-injected trajectories remain close to the policy distribution?
- **Theoretical Off-Policy Bounds**: Does the work provide theoretical guarantees that the distribution shift scales linearly with injection ratio or is otherwise bounded?
- **Mechanism Necessity**: Is the partial injection design justified as the minimal sufficient intervention, or could simpler alternatives achieve the same controlled shift?
- **Filtering Rationale**: Are techniques like entropy-based gradient filtering theoretically grounded in the on-policy analysis, or presented as purely empirical heuristics?

---

**Principle 3: Attribution Isolation via Compute-Matched Baselines and Comparative Analysis Against Distillation and Pure Resampling Methods**

**Definition:**  
The work must convincingly isolate the performance gains attributable to the core guided exploration mechanism from confounding factors such as increased sampling budget, additional compute, or implicit supervised fine-tuning effects. This requires designing compute-matched or sample-matched baselines that expend equivalent resources without the proposed mechanism, as well as comparing against direct distillation and sequential SFT-then-RL pipelines using the same ground-truth traces. Reviewers should assess whether the experimental design allows causal attribution of improvements to the inpainting-based guidance rather than to the mere availability of additional correct trajectories or increased sampling volume. The analysis should demonstrate that partial hint injection provides directional exploration signals that pure resampling or full SFT cannot replicate, and that the method's gains persist when total compute is held constant.

**Core Evaluation Criteria:**
- **Compute-Matched Baselines**: Does the work include baselines that match the total number of sampled trajectories or computational budget without hint injection?
- **Distillation Comparisons**: Are comparisons provided against applying SFT on the same reasoning traces, both alone and followed by standard RL?
- **Causal Mechanism Evidence**: Do ablation studies show that the injection mechanism itself (not just access to ground-truth data) drives performance improvements?
- **Resource Fairness**: Is the comparison fair in terms of training steps, model size, and data access, avoiding artificial handicaps for baseline methods?

---

**Principle 4: Robustness Evaluation to Imperfect Reasoning Traces and Sensitivity Analysis of Trace Quality in Guided Policy Optimization**

**Definition:**  
The work must evaluate whether the proposed guided exploration method remains effective when the available reasoning traces are imperfect, noisy, or partially corrupted, rather than assuming access to pristine ground-truth supervision. This is critical because real-world applicability depends on the method's ability to leverage imperfect or automatically generated hints without catastrophic failure. Reviewers should assess whether the authors conduct systematic corruption experiments introducing realistic noise types—such as numerical errors, operator swaps, logical inconsistencies, and filler tokens—and report performance degradation curves across corruption levels. The analysis should explain which design elements provide robustness, such as correctness verification, partial injection limiting exposure to errors, or entropy-based filtering of unreliable tokens. Furthermore, the work should clarify the boundary conditions under which trace quality becomes too degraded for the method to remain beneficial.

**Core Evaluation Criteria:**
- **Systematic Noise Injection**: Are experiments conducted with diverse, realistic corruption types at varying rates, and do results show graceful degradation?
- **Robustness Mechanism Attribution**: Does the analysis identify which components (e.g., partial injection, correctness gating, entropy filtering) confer robustness to noisy hints?
- **Failure Mode Analysis**: Are failure cases documented where corrupted traces mislead the policy, and are there diagnostics showing how incorrect hints are filtered or neutralized?
- **Practical Applicability Bounds**: Does the work clearly state the minimum trace quality required and discuss scenarios where the method should not be applied?

---

**Principle 5: Cross-Paradigm Positioning and Generalization Scope Assessment of Diffusion-Specific Guided Exploration Relative to Autoregressive and General RL Approaches**

**Definition:**  
The work must clearly position its contribution within the broader landscape of reinforcement learning for language models, explicitly addressing how architecture-specific methods for diffusion LLMs relate to concurrent guided exploration techniques in autoregressive models and general RL domains. Reviewers should evaluate whether the authors adequately discuss transferability of insights to non-diffusion architectures, compare against contemporaneous zero-advantage mitigation methods, and acknowledge limitations in scope rather than overstating generality. The work should articulate which components are fundamentally tied to bidirectional inpainting capabilities and which principles might inspire cross-architecture research. Additionally, the discussion should situate the method relative to alternative strategies such as advantage shaping, prefix sampling, and test-time guidance, clarifying whether the proposed approach is orthogonal, complementary, or strictly superior in specific regimes.

**Core Evaluation Criteria:**
- **Related Work Breadth**: Does the work discuss concurrent guided exploration methods in both diffusion and autoregressive LLM literature, including zero-advantage mitigation techniques?
- **Transferability Analysis**: Are there explicit arguments about which insights or mechanisms could transfer to autoregressive models or other diffusion-based systems?
- **Scope Limitations**: Does the work honestly acknowledge that the method requires diffusion-specific inpainting and high-quality traces, avoiding overgeneralized claims?
- **Complementary vs. Competitive Positioning**: Is the relationship to alternative approaches (e.g., advantage shaping, resampling, prefix-based methods) clearly articulated as orthogonal or synergistic?