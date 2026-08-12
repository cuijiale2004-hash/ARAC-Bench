**Principle 1: Theoretical Characterization of Objective Recovery and Convergence Guarantees Under Off-Policy Data in Quasimetric Goal-Conditioned RL**

**Definition:**  
In offline goal-conditioned reinforcement learning, methods that integrate multistep Monte Carlo returns with quasimetric distance architectures must rigorously characterize what their joint training objective actually recovers—whether it approximates the optimal value function, the behavior value function, or a novel intermediate distance measure. The quasimetric inductive bias fundamentally alters the optimization landscape by enforcing geometric constraints such as the triangle inequality, which means standard off-policy convergence results may not directly apply. Reviewers should demand formal statements of convergence conditions, explicit assumptions regarding tabular versus function approximation settings, and clarity on how the path relaxation or projection operator interacts with biased multistep backups. Without such theoretical grounding, empirical gains risk being attributed to algorithmic novelty rather than to implicit regularization or dataset-specific artifacts. Furthermore, the work must discuss sensitivity to off-policy data distributions and approximation errors, especially when combining globally biased multistep returns with locally optimal TD-style constraints. Ultimately, a clear theoretical narrative separates principled algorithmic advances from heuristic engineering.

**Core Evaluation Criteria:**
- Does the paper formally state what value or distance function the objective converges to (optimal, behavioral, or other) and under what assumptions?
- Are convergence guarantees explicitly conditioned on the quasimetric architecture, and is the interaction between multistep backup bias and the path relaxation operator analyzed?
- Does the work discuss limitations of the theory, such as the gap between tabular guarantees and function approximation, or sensitivity to off-policy data coverage?
- Is there a clear distinction between what is theoretically guaranteed versus what is empirically observed?

---

**Principle 2: Disentanglement of Architectural Inductive Biases from Multistep Algorithmic Innovations in Distance-Based GCRL**

**Definition:**  
When a method couples a specialized neural architecture—such as quasimetric residual networks—with an algorithmic mechanism like multistep waypoint regression, it is essential to isolate the contribution of each component to the observed performance. The field must understand whether improvements in long-horizon goal-reaching arise primarily from the distance structure itself, from global value propagation through multistep returns, or from a synergetic interaction that is non-obvious a priori. Reviewers should expect controlled ablations that apply the proposed algorithmic innovation to standard non-quasimetric critics and, conversely, that test alternative distance parameterizations without the multistep mechanism. Quantitative evidence should demonstrate performance drops when individual modules are removed or replaced, and qualitative visualization of learned distances can reveal whether the geometry respects the desired shortest-path structure. Such disentanglement prevents overclaiming novelty and ensures the community can correctly attribute success to the correct design decision. It also clarifies whether the architectural inductive bias is merely sufficient or actually necessary for the method to function.

**Core Evaluation Criteria:**
- Does the work include ablations applying multistep returns to non-quasimetric architectures (e.g., standard MLP critics) to isolate the architectural contribution?
- Are individual components systematically removed or replaced (e.g., action invariance loss, waypoint sampling distribution, quasimetric constraint) with performance quantified?
- Is there qualitative or quantitative evidence (e.g., distance heatmaps, policy gradient norms) showing how the learned geometry changes due to the multistep mechanism?
- Does the paper avoid conflating architectural sufficiency with algorithmic novelty by explicitly stating which design choices are necessary versus helpful?

---

**Principle 3: Comprehensive Baseline Coverage Against Quasimetric, Contrastive, and Hierarchical Horizon-Reduction Methods**

**Definition:**  
The offline GCRL literature has seen rapid proliferation of methods that vary subtly in their use of quasimetric representations, contrastive temporal distance learning, policy extraction mechanisms, and horizon-reduction strategies. To fairly position a new contribution, reviewers must insist on thorough, good-faith comparisons against the most closely related contemporaneous work, particularly prior quasimetric learners and recent flat or hierarchical approaches that target identical long-horizon challenges. An evaluation that omits strong baselines—or that implements them with mismatched policy extraction mechanisms or insufficient hyperparameter tuning—risks presenting an inflated picture of state-of-the-art performance. Comparisons should span both simulated benchmarks and real-world domains where applicable, using identical data modalities and observation spaces to ensure fairness. Beyond aggregate metrics, the paper should provide implementation details sufficient for independent reproduction of each baseline result. This standard ensures that claimed advances reflect genuine algorithmic progress rather than experimental asymmetries.

**Core Evaluation Criteria:**
- Does the evaluation include all major closely related quasimetric methods (e.g., QRL, CMD, TMD) and recent flat/hierarchical horizon-reduction baselines (e.g., SAW, HIQL) under fair conditions?
- Are policy extraction mechanisms controlled across methods (e.g., comparing DDPG+BC vs AWR consistently) to prevent confounding due to extraction strategy?
- Are both state-based and pixel-based environments covered, and do real-world experiments use identical modalities (e.g., pure image conditioning without language planners)?
- Are implementation details, hyperparameter tuning ranges, and random seeds reported transparently enough to reproduce baseline comparisons?

---

**Principle 4: Empirical Validation of Long-Horizon Stitching and Compositional Generalization in High-Dimensional Visual Robotics**

**Definition:**  
A central promise of distance-based GCRL is the ability to stitch together previously seen sub-trajectories and generalize to task horizons far exceeding those present in the offline dataset. Reviewers must therefore scrutinize whether the experimental protocol explicitly tests these capabilities through carefully designed stitch datasets, colossal or long-horizon evaluation mazes, and multi-stage manipulation tasks requiring temporal composition. Strong claims of scalability should be backed by diverse domain coverage—including both state and high-dimensional visual observations—and by real-world robotic validation that avoids reliance on external high-level planners or language-conditioned hierarchical structures. Fine-grained analysis of task progress, qualitative rollout videos, and failure mode categorization are necessary to establish that success is robust and not an artifact of narrow task distributions. Without such rigorous empirical demonstration, claims of compositional generalization and horizon scaling remain unsubstantiated.

**Core Evaluation Criteria:**
- Are there explicit stitch-dataset or long-horizon evaluations where test trajectories require composition or horizons significantly longer (e.g., 10×) than training data?
- Does the evaluation span multiple domains (locomotion, manipulation) and observation types (proprioceptive, visual) to demonstrate broad generalization?
- Are real-world robotic results reported with end-to-end policies, and is there fine-grained task-progress analysis rather than binary success alone?
- Does the paper analyze failure modes qualitatively (e.g., via rollout visualizations) to verify that stitching behavior is emergent rather than memorized?

---

**Principle 5: Rigor of Presentation, Reproducibility, and Transparent Reporting of Hyperparameter Costs and Heuristic Design Choices**

**Definition:**  
Technical soundness alone does not guarantee impact if a paper is presented opaquely or if its computational and hyperparameter demands render it impractical for broader adoption. Reviewers must evaluate whether the manuscript clearly articulates its key insight to a general RL audience, reports all critical hyperparameters and tuning procedures transparently, and acknowledges computational costs and heuristic design choices. This includes cleanly distinguishing between components with rigorous theoretical motivation—such as action invariance objectives—and those introduced as empirical heuristics, like geometric waypoint sampling or per-environment behavior-cloning coefficients. The writing should be free of significant notation errors, broken citations, and misleading figure captions, all of which undermine reproducibility and credibility. Additionally, sensitivity analyses and training resource requirements must be reported so that readers can assess the method's feasibility for their own settings. Overall, clarity and honesty about practical limitations are essential markers of high-quality research.

**Core Evaluation Criteria:**
- Is the core technical insight explained intuitively and early, accessible to readers outside the immediate sub-subfield?
- Are hyperparameter search costs, tuning ranges (e.g., for BC coefficients, waypoint discounts), and computational budgets reported transparently?
- Does the paper clearly label theoretically motivated design choices versus empirical heuristics, and discuss limitations of the latter?
- Is the manuscript substantially free of errors in notation, citations, figure captions, and mathematical indexing that would impede reproduction?