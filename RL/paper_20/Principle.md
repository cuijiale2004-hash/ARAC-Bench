**Principle 1: Contamination-Free Hyperparameter Search and Seed-Robustness in Auxiliary RL Objective Evaluation**

**Definition:**  
The evaluation of auxiliary regularization objectives in reinforcement learning is particularly susceptible to selection bias due to the high variance inherent in long-horizon episodic returns and the sensitivity of deep RL algorithms to hyperparameters. To ensure that reported gains reflect genuine algorithmic improvements rather than artifacts of experimental tuning, it is essential to maintain strict separation between the random seeds and data used for hyperparameter search and those used for final performance reporting. Running an insufficient number of seeds—especially in pixel-based or sparse-reward domains—risks concluding significance from noisy estimates. Furthermore, when multiple auxiliary configurations are compared against a baseline, the search process itself constitutes a form of multiple hypothesis testing that can inflate false-positive rates unless controlled. Therefore, rigorous review demands explicit disclosure of search budgets, disjoint evaluation protocols, and statistical tests that account for seed dependency and environmental variance. This principle ensures that auxiliary losses are judged on their robust merit, not on cherry-picked configurations.

**Core Evaluation Criteria:**
- **Disjoint Seed Protocol**: Does the work explicitly separate hyperparameter search seeds from final evaluation seeds, and report how many configurations were tested per environment?
- **Statistical Power and Variance Reporting**: Are a sufficient number of independent training runs (e.g., ≥10) reported per environment, including confidence intervals or standard errors, to rule out chance improvements?
- **Controlled Search Budget**: Is the baseline subjected to an identical tuning budget and procedure as the proposed method, preventing asymmetric optimization effort from confounding the comparison?
- **Cross-Domain Validation**: Are experiments conducted across diverse environment classes (e.g., high-dimensional pixel tasks and discrete symbolic tasks) to demonstrate that gains are not domain-specific artifacts?

---

**Principle 2: Theoretical Grounding and Mechanistic Interpretability of Continuous-Time Latent Flow Constraints in RL**

**Definition:**  
Proposing a continuous-time regularization mechanism such as an ODE flow for latent MDP trajectories introduces non-trivial inductive biases that must be justified beyond intuitive analogy. Reviewers should expect at least a preliminary formal or mechanistic analysis linking the latent geometry imposed by the flow—such as Lipschitz continuity, non-intersection, or smoothness—to the optimization dynamics of the underlying RL algorithm, whether through variance reduction, improved credit assignment, or reduced approximation error. Without such grounding, the method risks being evaluated as a black-box hack whose performance gains may stem from unrelated side effects rather than the claimed alignment of dynamics. Moreover, because the ODE model co-evolves with the policy during training, its non-stationarity and potential to introduce representational bias (e.g., collapsing latent diversity) must be addressed. A satisfactory submission should therefore disentangle "performance improvement" from "mechanism validity" by providing causal or ablative evidence that manipulating the flow directly influences the proposed intermediary variables.

**Core Evaluation Criteria:**
- **Formal-Mechanistic Link**: Does the paper provide analytical arguments or simplified proofs connecting ODE flow properties (e.g., uniqueness, smoothness) to policy gradient variance, value function bias, or exploration?
- **Non-Stationarity and Collapse Analysis**: Are the implications of training an ODE on policy-generated, non-stationary trajectories discussed, with empirical checks for latent collapse or homogenization?
- **Causal Ablation**: Is there evidence that directly optimizing the auxiliary loss causes the intermediate phenomenon (e.g., smoother latent paths) which in turn causes performance gains, rather than correlating them post-hoc?
- **Assumption Transparency**: Are critical mathematical assumptions (e.g., action-agnostic dynamics, Lipschitzness of the ODE vector field) explicitly stated, and their violation in MDP settings acknowledged?

---

**Principle 3: Empirical and Conceptual Distinctiveness from Contrastive, Predictive, and Bisimulation Representation Objectives**

**Definition:**  
The landscape of auxiliary representation learning in RL is densely populated with methods that exploit temporal structure, including contrastive learning, self-predictive coding, bisimulation metrics, and inverse dynamics modeling. A novel regularization technique must therefore establish its conceptual niche—whether through the type of inductive bias (e.g., global path consistency versus local vicinity clustering) or through empirically demonstrable differences in learned representations. It is insufficient to claim novelty based solely on architectural differences (e.g., ODE versus MLP) if the resulting latent space and downstream policy behavior are indistinguishable from those produced by existing methods. The review process should prioritize papers that either directly compare against strong, recent baselines from these families or rigorously argue why the proposed mechanism targets a fundamentally different property of the MDP. Ultimately, significance is determined not by the originality of the components but by whether the combination yields unique representational or behavioral advantages.

**Core Evaluation Criteria:**
- **Direct Baseline Comparison**: Does the work include head-to-head experiments with recent representative methods (e.g., TACO, SPR, CURL, or bisimulation-based objectives) under matched conditions?
- **Conceptual Differentiation**: Does the paper clearly articulate how the proposed objective differs from prior approaches in terms of what structure it imposes (e.g., continuous flow paths vs. pairwise similarity) and why that matters for RL?
- **Latent Property Uniqueness**: Is there evidence—qualitative or quantitative—that the learned latent space exhibits properties (e.g., global trajectory predictability from any predecessor, diffeomorphic structure) that competing methods do not?
- **Boundary Condition Honesty**: Does the paper discuss scenarios or environments where the method offers no advantage over existing auxiliary objectives, establishing realistic scope rather than overstating generality?

---

**Principle 4: Informativeness and Structural Validity of Latent Trajectory Geometry Metrics in RL Representation Learning**

**Definition:**  
Geometric descriptors such as latent path length, net displacement, and acceleration energy are frequently invoked to argue that a regularization technique produces smooth, well-structured representations. However, these metrics are susceptible to trivial manipulation—for instance, isotropically scaling the latent space can artificially reduce path lengths and displacements without improving semantic organization or task performance. Consequently, reviewers must demand that such metrics be validated as informative: they should correlate with functional representational quality, resist naive shortcuts, and ideally be accompanied by complementary evidence such as probing accuracy or visualization. The mere observation that an auxiliary loss reduces path length is not sufficient; one must show that this reduction reflects a meaningful alignment with environment dynamics rather than a constrained but uninformative embedding norm. Robust evaluation therefore treats geometric metrics as hypotheses to be tested rather than self-evident proofs of representational superiority.

**Core Evaluation Criteria:**
- **Anti-Gaming Checks**: Are the metrics shown to be robust against trivial transformations (e.g., scaling), or is there an ablation demonstrating that directly minimizing latent norms degrades performance, confirming the metric is emergent?
- **Functional Correlation**: Do lower path roughness or displacement values correlate with improved downstream task performance, state predictability, or generalization in a way that is statistically consistent across environments?
- **Complementary Evidence**: Are geometric statistics supplemented with qualitative analyses (e.g., trajectory visualizations in reduced dimensions) or probing tasks that verify semantic structure is preserved?
- **Comparative Metric Analysis**: Are the same metrics reported for strong baseline methods (including those that shape representations differently) to calibrate whether the observed values are exceptional or commonplace?

---

**Principle 5: Validity of Action-Agnostic ODE Flow Modeling Under Non-Stationary Policy Dynamics in MDPs**

**Definition:**  
Modeling MDP latent trajectories as flows of an autonomous ODE assumes that the future latent state is determined exclusively by the present latent state and a time index, deliberately omitting the action that actually drives the transition. This action-agnostic abstraction is theoretically questionable because MDP transition dynamics are inherently action-conditional, and the distribution of trajectories used to train the ODE shifts as the policy improves, creating a moving-target problem. Furthermore, the uniqueness property of ODE solutions—whereby distinct flow paths cannot intersect—may conflict with MDP semantics in environments containing bottleneck states or looping behaviors that legitimately revisit the same state from different histories. Reviewers must assess whether the submission confronts these modeling discrepancies head-on, either by justifying the approximation through empirical robustness or by analyzing its failure modes. A method that ignores the non-stationarity and structural mismatch between continuous flows and discrete, action-driven MDPs risks being brittle or misattributing its empirical successes.

**Core Evaluation Criteria:**
- **Action-Conditioning Justification**: Does the paper explicitly justify why actions are excluded from the ODE model, and discuss whether incorporating them would alter learning dynamics or stability?
- **Non-Stationarity Robustness**: Is there empirical analysis (e.g., tracking ODE loss or latent alignment over training epochs) showing that the flow model adapts effectively despite the non-stationarity induced by policy updates?
- **Structural Mismatch Analysis**: Are pathological cases such as bottleneck states, trajectory intersections, or loops investigated to verify that ODE non-intersection does not harm representation or generalization?
- **Empirical Safety Evidence**: Does the work provide domain-spanning experiments (e.g., gridworlds with forced crossings and high-dimensional pixel tasks) demonstrating that the action-agnostic flow assumption remains benign across qualitatively different state-transition graphs?