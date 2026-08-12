**Principle 1: Necessity and Theoretical Justification of Structural Inductive Biases Beyond Raw Transformer Expressivity in Offline Return-Conditioned Sequence Modeling**

**Definition:**  
In offline return-conditioned supervised learning, Transformers are often assumed to be universally expressive sequence approximators. However, reviewers consistently demand justification for why explicit architectural constraints—such as dual-feature fusion branches—are necessary despite this theoretical capacity. The core issue is that theoretical universality does not guarantee practical learnability from fixed, suboptimal datasets. A standard Transformer may implicitly overfit to historical correlations because it lacks domain-specific priors for Markovian decision processes. Therefore, high-quality research must rigorously argue that the proposed structural bias corrects a specific, empirically identifiable failure mode of pure sequence modeling. This principle evaluates whether authors move beyond end-to-end accuracy to demonstrate that their inductive bias is causally responsible for mitigating trajectory replication. It ensures that architectural novelty is grounded in diagnosed representational or optimization deficiencies rather than post-hoc performance attribution.

**Core Evaluation Criteria:**
- **Inductive Bias Necessity**: Does the work provide a rigorous argument (empirical or theoretical) for why standard Transformer architectures fail to discover the desired behavior implicitly, rather than assuming they cannot?
- **Causal Mechanistic Link**: Does it establish a causal link between the proposed structural bias and the specific failure mode (e.g., overfitting to suboptimal trajectories), rather than relying solely on end-to-end performance gains?
- **Differentiation from Universal Approximation**: Does the work explicitly address the theoretical expressivity of the base architecture and justify why explicit modularization is required for practical learning dynamics in offline RL?
- **Avoidance of Post-Hoc Rationalization**: Are the architectural decisions motivated a priori by identifiable limitations of pure sequence modeling, supported by diagnostic experiments or case studies?

---

**Principle 2: Mechanistic Validity and Non-Collapsing Behavior of State-Dependent Adaptive Fusion Weights for Global-Local Feature Integration in Offline RL**

**Definition:**  
Adaptive fusion mechanisms promise to dynamically balance global historical context against local immediate signals, but reviewers require evidence that these learned coefficients operate as intended rather than collapsing to trivial static solutions. The concern is that a learned weighting module may secretly converge to a near-constant value, effectively reducing the hybrid architecture to a single-mode baseline with unnecessary complexity. High-quality work must therefore transparently inspect the distribution of adaptive weights across states and link their variation to interpretable environmental properties, such as return consistency or data suboptimality. This principle scrutinizes whether the mechanism exhibits genuine, state-dependent modulation or merely provides an illusion of adaptivity. It also demands analysis of pathological cases where the signal used to drive adaptation becomes unreliable, such as encoder-induced aliasing in pixel-based tasks. Ultimately, mechanistic validity ensures that the "adaptive" claim is supported by internal, verifiable behavior rather than black-box output improvements.

**Core Evaluation Criteria:**
- **State-Dependent Variation**: Does the work provide quantitative or visual evidence that the adaptive weights vary meaningfully across states rather than collapsing to a constant or trivial solution?
- **Behavioral Transparency**: Is the relationship between the adaptive signal (e.g., estimated value discrepancy) and the resulting feature weighting clearly explained and validated?
- **Failure Mode Analysis**: Does the work analyze cases where the adaptive mechanism might fail or misweight features (e.g., visual aliasing in POMDPs), and are these limitations disclosed?
- **Anti-Collapse Verification**: Are ablations or gradient analyses conducted to verify that the mechanism does not degrade into a static bias toward one feature branch?

---

**Principle 3: Automated Cross-Domain Generalization and Explicit Limitation Characterization in Hybrid Architectures for MDPs and POMDPs**

**Definition:**  
Hybrid architectures must demonstrate that their automated balancing of feature streams generalizes across fundamentally different decision-making regimes without manual reconfiguration for each domain. A method that achieves strong performance on fully observable locomotion tasks but requires hand-tuned coefficients for partially observable visual control fails to deliver on the promise of a unified framework. Reviewers evaluate whether the proposed approach seamlessly transitions between Markovian settings, where local features should dominate, and history-dependent settings, where global context is essential. This principle requires explicit characterization not only of success domains but also of failure modes and scope limitations. By demanding cross-domain consistency and transparent disclosure of manual interventions, it prevents overgeneralized claims of adaptability and ensures that the method's automation is robust to environmental diversity.

**Core Evaluation Criteria:**
- **Cross-Domain Automation**: Does the method automatically adapt its global-local balance across diverse environmental properties (fully observable vs. partially observable, dense vs. sparse rewards) without manual per-domain intervention?
- **Fair Baseline Isolation**: Are comparisons against pure global and pure local baselines conducted under identical training pipelines to isolate the benefit of hybridization?
- **Limitation Disclosure**: Are domains where the adaptive mechanism fails or requires manual tuning explicitly characterized, with plausible explanations (e.g., encoder-induced state aliasing)?
- **Generalization Consistency**: Does the method maintain stable relative improvements across qualitatively distinct task families (locomotion, navigation, manipulation, pixel control)?

---

**Principle 4: Isolated Evaluation of Trajectory Stitching Efficacy and Suboptimal Trajectory Avoidance in Return-Conditioned Supervised Learning**

**Definition:**  
Trajectory stitching—the ability to recombine suboptimal trajectory segments into superior policies—is a central objective of offline RL, yet it is often obscured by aggregate benchmark scores alone. Reviewers must assess whether experimental designs explicitly isolate stitching capability from generic supervised learning improvements or favorable data distributions. This requires evaluating methods on datasets where optimal policies necessarily differ from any single behavioral trajectory, paired with ablations that control for context length and sequence coverage. The principle further demands analysis of how the method handles mixed-quality data, specifically whether it avoids replicating suboptimal historical patterns while still leveraging valuable long-horizon structure. Without such diagnostic rigor, claims of improved stitching remain empirically unsubstantiated and confounded by simpler confounding factors.

**Core Evaluation Criteria:**
- **Explicit Stitching Diagnostics**: Are experiments designed to explicitly measure trajectory stitching, such as evaluating performance on datasets requiring composition of suboptimal segments rather than only reporting aggregate returns?
- **Controlled Ablation of Sequence Length**: Does the work systematically vary context length to disentangle the benefits of fusion from simple sequence truncation or expansion?
- **Suboptimal Data Utilization**: Does the analysis distinguish between improved filtering of suboptimal data and genuine synthesis of novel trajectories exceeding behavioral policy performance?
- **Benchmark Coverage**: Is evaluation conducted across benchmarks with varying trajectory quality mixtures to validate robustness to suboptimal data?

---

**Principle 5: Complexity-Efficiency Justification and Reproducibility Standards for Multi-Objective Multi-Network Offline Reinforcement Learning Frameworks**

**Definition:**  
Advances in offline RL increasingly rely on multi-component architectures integrating sequence modeling, value estimation, and adaptive routing, which raises legitimate concerns about complexity inflation relative to practical gains. Reviewers must weigh whether incremental performance improvements justify significant increases in network count, loss functions, and training time. This principle mandates transparent reporting of computational overhead—including wall-clock training duration and memory consumption—relative to strong, comparably implemented baselines. Furthermore, it emphasizes strict reproducibility standards, requiring detailed hyperparameter specifications, pseudocode, and consistent evaluation protocols to ensure that reported gains are not artifacts of unfair tuning or small-sample variance. By enforcing a high bar for efficiency and replicability, this criterion distinguishes substantiated engineering contributions from opaque architectural accretions.

**Core Evaluation Criteria:**
- **Performance-Complexity Justification**: Does the performance improvement over simpler baselines justify the introduction of additional networks, loss functions, and training stages?
- **Computational Transparency**: Are training time and memory overheads reported relative to strong baselines under identical hardware and software conditions?
- **Reproducibility Standards**: Does the paper provide sufficient implementation details (pseudocode, hyperparameter tables, codebase) to enable exact reproduction and extension?
- **Fair Comparison Protocol**: Are baseline results obtained via consistent evaluation protocols (e.g., number of evaluation trajectories, seeds) to avoid artifacts of small-sample variance?