**Principle 1: Theoretical Grounding for Spectral-Jacobian Conditioning in Attention Weight Initialization**

**Definition:**  
Research proposing initialization schemes for Transformer attention layers must provide a theoretically coherent justification that links the spectral structure of initial weights to optimization dynamics. This requires going beyond deriving algebraic bounds or matrix identities; the work must explain why controlling quantities such as the Jacobian condition number should lead to faster convergence or more stable training. Establishing this connection typically involves leveraging established theoretical frameworks such as Neural Tangent Kernel theory, dynamical isometry, or gradient flow analysis. Intermediate mathematical results—such as Kronecker-product Jacobian formulations—should be clearly positioned as methodological stepping stones rather than standalone contributions. Reviewers expect that the theoretical narrative distinguishes between novel conceptual insights and standard matrix calculus derivations. The absence of such a mechanistic explanation risks reducing the contribution to a heuristic observation, even when empirical gains are present.

**Core Evaluation Criteria:**
- **Non-Trivial Theoretical Bridge**: Does the work explain why the spectral property improves optimization (e.g., via NTK conditioning, convergence rate bounds) rather than merely deriving that the property can be bounded?
- **Scope of Analytical Setting**: Is the theoretical analysis restricted to vanilla attention without normalization, and if so, is this limitation clearly acknowledged as a simplifying assumption rather than a practical claim?
- **Positioning of Intermediate Results**: Are standard matrix calculus identities clearly distinguished from the paper’s core conceptual contribution, avoiding the misattribution of known formulas as novel theory?
- **Predictive vs. Post-Hoc Rationalization**: Does the theory predict the empirical phenomena observed, or does it only provide a loose narrative after the fact?

---

**Principle 2: Compatibility with Modern Normalization and Stabilization Mechanisms in Transformer Attention**

**Definition:**  
Initialization methods for modern Transformers must be evaluated within the context of contemporary architectures that ubiquitously employ stabilization mechanisms such as LayerNorm, RMSNorm, and QKNorm. Reviewers expect that proposals are not validated solely on stripped-down theoretical models lacking these components, but are instead tested on standard implementations where normalization layers remain active. A critical conceptual requirement is the clear articulation of why normalization and spectral conditioning are orthogonal: normalization primarily constrains the magnitude of activations and gradients, whereas conditioning governs the singular-value structure of the Jacobian. This distinction must be supported by controlled experiments demonstrating that the initialization provides benefits both with and without normalization, rather than simply substituting for it. The method should integrate seamlessly into existing training pipelines without requiring the removal or modification of standard architectural stabilizers.

**Core Evaluation Criteria:**
- **Preservation of Native Architectures**: Are all experiments conducted on standard architectures with their original normalization layers intact, without removing them to suit the theoretical setting?
- **Conceptual Distinction from Normalization**: Does the work explicitly articulate the difference between magnitude normalization and spectral conditioning, with illustrative examples or controlled experiments?
- **Empirical Complementarity**: Does the method yield consistent improvements both in the presence and absence of normalization, demonstrating it is complementary rather than a substitute?
- **Pipeline Integration**: Is the initialization shown to be a drop-in replacement that does not require modifying the architecture, optimizer, or training schedule?

---

**Principle 3: Cross-Modal and Cross-Scale Empirical Validation of Conditioning-Based Initialization**

**Definition:**  
Because initialization exerts its influence at the earliest stage of training, its benefits must be shown to persist across model scales ranging from small task-specific transformers to large-scale models with hundreds of millions or billions of parameters. Reviewers expect comprehensive empirical coverage spanning multiple modalities—such as vision classification, object detection, instance segmentation, language modeling, and long-range sequence tasks—to rule out domain-specific artifacts. The evaluation should include metrics beyond final test accuracy, such as convergence speed, training loss trajectories, and computational efficiency, to demonstrate that the initialization genuinely accelerates optimization rather than merely perturbing the final solution. Evidence at larger scales is particularly important because initialization effects may be washed out by massive datasets, extensive training budgets, or aggressive optimization algorithms. The work should explicitly discuss any resource constraints that limit large-scale evaluation and interpret results cautiously when extrapolating.

**Core Evaluation Criteria:**
- **Scale Coverage**: Does the evaluation span small-scale tasks, medium-scale benchmarks, and larger models, with explicit discussion of scalability trends and resource limitations?
- **Cross-Modal Breadth**: Are results reported across diverse modalities and task types (e.g., vision classification/detection/segmentation, language modeling, sequence modeling) to demonstrate architecture-agnostic benefits?
- **Training Dynamic Metrics**: Beyond final accuracy or perplexity, does the work report convergence speed, epochs-to-target, loss curves, or compute savings attributable to the initialization?
- **Robustness to Overfitting Regimes**: At very large scales where overfitting may dominate, does the work report whether the initialization maintains relative advantages or discuss why effects might diminish?

---

**Principle 4: Component-Wise Ablation and Disentanglement of Spectral Initialization Inductive Biases**

**Definition:**  
Rigorous evaluation of an initialization scheme requires systematic ablation studies that isolate the contribution of each modified component within the attention mechanism. Since attention comprises query, key, value, and output projections, reviewers expect experiments that initialize only subsets of these matrices to validate theoretical predictions about which components are essential for the claimed effect. Furthermore, the proposed bias must be disentangled from competing initialization paradigms—such as mimetic initialization or weight selection—through hybrid experiments that test for complementarity or redundancy. It is equally important to establish that the method is strictly additive: baseline models should be compared using their originally tuned hyperparameters, without re-tuning learning rates, warmup schedules, or weight decay specifically for the proposed method. This ensures that reported gains are attributable to the initialization’s inductive bias rather than incidental hyperparameter interactions.

**Core Evaluation Criteria:**
- **Component Isolation**: Are ablations provided for initializing subsets of parameters (e.g., only W_Q/W_K, only W_V, excluding W_O) to verify which components drive the effect?
- **Hybridization Controls**: Does the work test combinations with competing initialization schemes to assess whether biases are overlapping or complementary?
- **Hyperparameter Fairness**: Is it confirmed that baseline methods use their originally tuned configurations, and that the proposed method does not require re-tuning to outperform baselines?
- **Additive Benefit Verification**: Is the initialization demonstrated as strictly additive—applied on top of existing pipelines without architectural modification—ensuring gains are not confounded by structural changes?

---

**Principle 5: Bound-Gap Verification Between Jacobian Condition Bounds and Empirical Training Dynamics**

**Definition:**  
When an initialization strategy is motivated by optimizing an upper bound on a spectral quantity—such as the condition number of the attention Jacobian—reviewers demand rigorous empirical verification that the bound reduction translates into tangible optimization benefits. The work must track the actual spectral quantity during training to confirm that the bound serves as a reliable proxy, rather than merely being mathematically convenient. This verification should be accompanied by direct evidence of improved training dynamics, including smooth loss curves, stable gradient norms, and reduced training instability metrics. The authors must transparently acknowledge the looseness of the bound and discuss whether a tighter characterization might yield further improvements or reveal hidden limitations. Treating bound optimization as equivalent to objective optimization without such empirical and epistemic diligence is considered a significant methodological weakness.

**Core Evaluation Criteria:**
- **Empirical Proxy Validation**: Does the work track the actual condition number or relevant spectral metrics during training to verify that the bound reduction correlates with real improvement?
- **Optimization Stability Evidence**: Are training dynamics explicitly visualized or quantified (loss curves, gradient norm trajectories, smoothness metrics) to link spectral conditioning to stable optimization?
- **Limitation Transparency**: Does the work candidly discuss the gap between the bound and the true condition number, and whether tighter bounds might yield further improvements?
- **Causal Attribution**: Does the analysis avoid conflating "improved bound" with "improved optimization," establishing instead a causal chain from bound to spectrum to training outcomes?