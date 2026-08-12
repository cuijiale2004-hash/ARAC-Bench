**Principle 1: Causal Attribution and Mechanistic Grounding of Robustness via Optimizer-Grade-Specific Landscape Dynamics**

**Definition:**  
Research in this subfield must move beyond correlational observations that noisier optimizers yield more robust unlearning and instead establish a mechanistic, causal link between the fidelity of descending information, termed optimizer grade, and the resulting robustness properties. The work should demonstrate that robustness arises from systematic, optimizer-induced structural properties—such as convergence to flatter basins, randomized smoothing effects, or denoising projections—rather than from generic stochasticity that could be injected arbitrarily. This requires articulating why a specific downgrade in gradient information, whether through zeroth-order estimation or sign-based compression, produces distinct loss-landscape geometry compared to full-precision first-order methods. Furthermore, the work must clarify how this geometry directly resists post-unlearning weight perturbations including quantization and relearning attacks. The explanation should leverage established optimization principles, such as the equivalence between zeroth-order updates and randomized smoothing, to provide a rigorous rather than intuitive account of the observed phenomena.

**Core Evaluation Criteria:**
- **Structured vs. Arbitrary Noise Distinction**: Does the work explicitly differentiate the structured, optimizer-grade-dependent noise from arbitrary noise injection, and provide evidence that the specific noise structure (not just variance) drives robustness?
- **Landscape Mechanism Validation**: Does it empirically verify that downgraded optimizers converge to qualitatively different basins (e.g., via linear mode connectivity, Hessian analysis, or gradient norm trajectories) rather than merely quantitatively different loss values?
- **Theoretical/Conceptual Anchoring**: Are the robustness gains connected to established principles (e.g., randomized smoothing equivalence for zeroth-order methods, projection properties of quantized gradients) to avoid black-box reasoning?

---

**Principle 2: Rigorous Evaluation of Unlearning-Retention Trade-offs Under Weight-Space Perturbations**

**Definition:**  
Because unlearning inherently involves a tension between forgetting target knowledge and retaining general utility, any claim of robustness must be evaluated under the constraint that utility preservation is maintained both before and after perturbation. The work must demonstrate that robustness gains do not come at the cost of catastrophic retention collapse, and that comparisons across optimizers are fair—ideally calibrated to comparable retention levels using techniques such as model merging or controlled hyperparameter search. Evaluations should span multiple weight-space threat models, including quantization, relearning attacks, and downstream fine-tuning, rather than relying on isolated perturbations that may not reflect realistic deployment scenarios. It is essential to track both forgetting metrics and utility metrics after attacks to ensure that robustness reflects genuine stability of the desired unlearned state rather than mere model degradation or knowledge destruction. Finally, the work should explicitly justify its choice of threat model scope, distinguishing between weight-space robustness and input-space adversarial attacks to prevent overstated claims.

**Core Evaluation Criteria:**
- **Retention-Calibrated Comparisons**: Are optimizers compared at comparable unlearning-retention trade-off points? If direct comparison is confounded by different retention baselines, does the work employ calibration techniques to ensure fairness?
- **Multi-Perturbation Robustness**: Does the evaluation cover diverse weight-space perturbations (quantization, relearning on forget/retain sets, irrelevant fine-tuning) to substantiate broad robustness rather than specialization to a single attack vector?
- **Post-Perturbation Utility Tracking**: Does the work track not only forgetting metrics after attacks but also utility/retention metrics, ensuring that robustness reflects stability of the desired unlearned state rather than indiscriminate model damage?

---

**Principle 3: Cross-Benchmark and Cross-Objective Generalization of Optimizer-Grade Robustness Claims**

**Definition:**  
Since LLM unlearning encompasses diverse scenarios—memorized content removal, harmful knowledge erasure, and fictitious concept deletion—claims about optimizer-grade effects must generalize across multiple benchmarks and unlearning formulations to avoid idiosyncratic conclusions. Findings derived from a single objective, such as NPO, or a single dataset, such as MUSE, risk being artifacts of that specific configuration rather than reflecting universal optimizer behavior. The work should validate that the optimizer-grade-robustness relationship holds across distinct unlearning objectives, including gradient-difference, preference-optimization, and representation-misdirection methods, as well as across domain-specific benchmarks. Each optimizer must be evaluated with its best-performing hyperparameter configuration via controlled search to avoid artificially disadvantaged baselines that could confound grade-based conclusions. Moreover, the study should examine whether the observed phenomena persist across different model scales or architectures to establish the broader applicability of the proposed insights.

**Core Evaluation Criteria:**
- **Multi-Objective Validation**: Are the claims replicated across fundamentally different unlearning objectives (e.g., GradDiff, NPO, RMU) to rule out objective-specific artifacts?
- **Benchmark Diversity**: Does the evaluation span multiple standard benchmarks (e.g., MUSE for memorization, WMDP for safety, TOFU for structured forgetting) representing different unlearning modalities?
- **Hyperparameter Fairness and Best-Practice Controls**: Is each optimizer evaluated with its best-performing hyperparameter configuration (via controlled search), avoiding artificially disadvantaged baselines that could confound grade-based conclusions?

---

**Principle 4: Principled Design and Systematic Ablation of Hybrid Optimizer Architectures**

**Definition:**  
When proposing hybrid or composite optimizers that combine different grades of information, such as alternating first-order and zeroth-order updates, the work must provide a principled rationale for the integration strategy rather than presenting an ad-hoc empirical mixture. The design should be grounded in an explicit conceptual framework, whether drawn from game theory, dynamical systems, or optimization theory, that explains why alternation yields synergistic benefits unattainable by either standalone component. Furthermore, the specific scheduling choices, including alternation frequency, asymmetric step allocations, and termination conditions, must be systematically ablated to demonstrate that the proposed configuration achieves an optimal or near-optimal robustness-efficacy trade-off. The work must also demonstrate that the hybrid outperforms both its individual constituents and naive mixing strategies, confirming that the integration is genuinely synergistic rather than merely averaging their respective behaviors. Finally, the sensitivity of the hybrid design to its hyperparameters should be characterized to ensure that its advantages are stable across reasonable operating regimes.

**Core Evaluation Criteria:**
- **Conceptual Framework for Hybridization**: Is the hybrid design motivated by a clear conceptual model (e.g., leader-follower bilevel optimization, zero-sum game dynamics, smoothing-precision trade-offs) that explains why alternating yields benefits unattainable by either component alone?
- **Schedule Ablation Rigor**: Does the work include systematic ablations over switching intervals, asymmetric step allocations (e.g., more FO than ZO or vice versa), and initialization/termination protocols?
- **Non-Triviality of Combination**: Does the work demonstrate that the hybrid outperforms both standalone components and naive mixing strategies, confirming that the integration is synergistic rather than merely averaging?

---

**Principle 5: Empirical Evaluation of Computational Efficiency and Practical Deployment Viability**

**Definition:**  
Downgraded optimizers, particularly zeroth-order methods, are often assumed to be computationally prohibitive due to multiple forward evaluations or inexact gradient estimates, so research claiming practical utility must empirically validate efficiency claims. The work should report wall-clock time comparisons between downgraded, hybrid, and standard first-order optimizers under identical hardware and unlearning configurations to ensure fair assessment. Memory footprint, iteration counts, and per-step costs must be analyzed within the specific context of short-horizon unlearning, where updates start from a pretrained initialization and require only moderate adjustments. The trade-off between robustness improvement and computational overhead should be explicitly quantified, identifying scenarios where methods remain practical versus those where costs become prohibitive relative to gains. Finally, the work should compare against efficiency-aware robustness baselines to demonstrate that the proposed approach is not only effective but also competitive in resource-constrained deployment settings.

**Core Evaluation Criteria:**
- **Wall-Clock Time and Throughput Comparisons**: Does the work report actual runtime comparisons between downgraded, hybrid, and standard optimizers under identical hardware and unlearning configurations?
- **Cost-Benefit Analysis**: Is the trade-off between robustness improvement and computational cost explicitly quantified? Are there scenarios where the overhead makes an alternative method preferable?
- **Scalability Justification**: Does the work leverage the short-horizon nature of unlearning (moderate updates from a pretrained initialization) to justify the practicality of methods that might be infeasible for full pretraining?