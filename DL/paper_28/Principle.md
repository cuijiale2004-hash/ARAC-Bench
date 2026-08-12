**Principle 1: Precise Formulation of End-to-End Inductive Mapping Versus Hyperparameter Search Automation**

**Definition:**  
In meta-learning for data visualization, the research must clearly articulate whether the contribution is a learned parametric mapping that directly produces low-dimensional embeddings from raw high-dimensional datasets, or merely an automated hyperparameter selection pipeline that still relies on iterative optimization. This distinction is foundational because an inductive model must eliminate per-dataset retraining and hyperparameter tuning at inference, whereas a search-then-embed approach retains the computational and tuning overhead of traditional methods. The work should formally define the learned function, specify how it handles variable input dimensions and sample sizes, and incorporate invariances to rotation, translation, and scaling directly into the problem formulation and loss design. Authors must explicitly demonstrate that no oracle-level per-dataset optimization, search, or label-dependent validation is required during deployment on unseen data. Without this clarity, the contribution risks being mischaracterized as a hyperparameter optimization wrapper rather than a true end-to-end visualization model.

**Core Evaluation Criteria:**
- Is the task rigorously defined as a single function mapping arbitrary datasets to embeddings without iterative per-dataset optimization?
- Does the work explicitly distinguish itself from hyperparameter search or Bayesian optimization-based tuning frameworks?
- Are visualization invariances (translation, rotation, scale) formally incorporated into the problem definition and loss design rather than treated as post-processing?
- Is the inference pipeline demonstrated to require zero hyperparameter tuning on unseen datasets?

---

**Principle 2: Rigorous Delineation of Generalization Boundaries and Training Data Diversity Requirements**

**Definition:**  
The value of an inductive visualization model hinges on understanding exactly when it succeeds and fails across structural, domain, and dimensional shifts. High-quality research must move beyond demonstrating transfer between similar classification datasets to systematically probing the boundaries of generalization, such as failure on unseen manifold topologies or asymmetric cross-dimensional transfer. The work should disentangle the effects of training sample size, domain diversity, and geometric structural diversity through controlled ablations, establishing that structural diversity—not merely domain coverage—is the dominant factor governing cross-dataset performance. Furthermore, explicit qualitative and quantitative failure analyses on out-of-distribution geometric structures, noise regimes, and overlapping clusters are essential to characterize the model's reliability and deployment limits. Only by mapping these boundaries can practitioners determine whether the model is suitable for their specific data geometry.

**Core Evaluation Criteria:**
- Does the work provide controlled ablations separating the impact of training structural diversity, domain diversity, and sample size on generalization?
- Are explicit failure modes documented, including unseen manifold types, high-noise regimes, and overlapping cluster structures?
- Does the work test cross-domain transfer across modalities with fundamentally different feature semantics and dimensionalities?
- Is there analysis of asymmetric dimensionality shifts and their impact on embedding stability?

---

**Principle 3: Comprehensive Neighborhood Preservation Assessment Beyond Label-Dependent Clustering Metrics**

**Definition:**  
Relying exclusively on clustering-centric metrics such as NMI or Silhouette Coefficient provides an incomplete and potentially biased picture of visualization quality, particularly for manifold-structured or unlabeled data. A rigorous evaluation framework for inductive visualization must include local neighborhood preservation metrics—such as Trustworthiness, Continuity, KNN overlap, Jaccard similarity, and triplet preservation—alongside global distance correlations and rank-based fidelity measures. Comparisons should encompass both oracle-optimal baselines (to establish approximation quality) and default-hyperparameter baselines (to establish practical utility), ensuring the model is assessed against realistic deployment scenarios. The selection of metrics must be justified in relation to the geometric properties of the test data, distinguishing between cluster-centric and manifold-centric visualization objectives. Without such a multi-scale assessment, claims of universal visualization quality remain unsubstantiated and may reflect overfitting to labeled cluster structures.

**Core Evaluation Criteria:**
- Does the evaluation suite include local structure preservation metrics (Trustworthiness, Continuity, KNN accuracy/overlap) in addition to clustering metrics?
- Are both oracle-optimal and default-hyperparameter variants of traditional methods compared to demonstrate real-world utility?
- Are distance rank correlations and triplet preservation used to assess global and local fidelity?
- Is metric selection justified by the geometric nature of test datasets (clusters vs. continuous manifolds)?

---

**Principle 4: Empirical and Theoretical Justification of Variable-Dimension Graph Embedding Architecture**

**Definition:**  
Because inductive visualization models must ingest datasets with varying sample sizes and feature dimensionalities, architectural decisions—such as multi-scale graph construction, positional encoding strategies, sign-ambiguity resolution, and affine-invariant loss functions—cannot be treated as arbitrary implementation details. High-quality work must provide systematic ablation studies isolating the contribution of each component, complemented by theoretical analysis that explains why the architecture is suitable for the visualization task. This includes bounding the model's stability under input perturbations and justifying how the design handles the one-to-many mapping problem induced by visualization invariances. The rationale for graph-based encoding of variable-dimension data must be clearly linked to the elimination of fixed-input-dimension constraints that plague standard parametric methods. Without such justification, the model risks being an opaque black box whose performance is accidentally tied to the training distribution rather than robustly engineered for cross-dataset generalization.

**Core Evaluation Criteria:**
- Are ablation studies provided for core components: number of graph scales, bandwidth parameters, PE type/dimension, sign-ambiguity handling, and loss formulation?
- Does the theoretical analysis include robustness or stability bounds that link input perturbations to embedding drift?
- Is the affine-invariant loss design empirically validated against naive coordinate-regression alternatives?
- Are architectural choices motivated by the specific challenges of variable-dimension, permutation-invariant dataset processing?

---

**Principle 5: Empirical Scalability Validation and Computational Feasibility for Large Dataset Inference**

**Definition:**  
An inductive visualization model that is constrained to training subset sizes imposes severe practical limitations unless explicit, validated mechanisms are provided for scaling to larger datasets at inference time. The work must demonstrate empirical feasibility—through wall-clock timing, memory profiling, and real-world large-scale experiments—rather than relying solely on asymptotic complexity arguments. This includes validating batch-based or anchor-based inference strategies that maintain embedding consistency across chunks, and comparing end-to-end latency against both iterative methods and parametric baselines. For a visualization method to be practically useful, it must convincingly show that its training-time constraints do not preclude deployment on realistically sized datasets encountered in production scientific and engineering workflows. Scalability claims must be backed by concrete evidence that the inference pipeline can handle datasets orders of magnitude larger than the training graphs without catastrophic degradation of neighborhood fidelity.

**Core Evaluation Criteria:**
- Are empirical running time and memory comparisons reported against traditional iterative methods and parametric deep baselines?
- Does the work demonstrate successful inference on datasets significantly larger than the maximum training subset size?
- Are specific scalability mechanisms (e.g., batch-based extension with anchor calibration, graph sparsification) described, justified, and empirically validated?
- Is the trade-off between inference speed and embedding fidelity quantified across different dataset scales?