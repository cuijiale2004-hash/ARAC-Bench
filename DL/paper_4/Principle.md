**Principle 1: End-to-End Generality and Cross-Domain Extensibility of Differentiable Graph Lifting Across Hypergraphs, Simplicial, and Cell Complexes**

**Definition:**  
A primary criterion for evaluating learnable lifting frameworks is whether they provide a unified, end-to-end differentiable mechanism that genuinely generalizes across disparate topological domains—including hypergraphs, simplicial complexes, cell complexes, and combinatorial complexes—rather than offering a domain-specific patch. Because the choice of topological domain and lifting procedure is highly data-dependent, a general framework must clearly articulate why prior domain-specific solutions (e.g., modules limited to cell complexes) cannot be trivially extended, and it should demonstrate applicability beyond standard attributed graphs, for instance to point clouds or higher-dimensional simplicial input. The principle demands that the proposed method not merely combine existing operators, but present a coherent recipe whose domain-dependent ingredients are minimal and well-isolated, ensuring that task-adaptive topology learning is portable rather than fragmented.

**Core Evaluation Criteria:**
- **Unified Domain Coverage**: Does the framework present a single, coherent formulation for all claimed target domains, with only minimal, well-defined domain-specific steps (e.g., candidate elicitation)?
- **Distinctiveness from Domain-Specific Predecessors**: Are the technical barriers to extending prior learnable lifting methods to other domains (e.g., hypergraphs) clearly explained, and does the new framework overcome them (e.g., via embarrassingly parallel vs. sequential sampling)?
- **Empirical Extensibility**: Does the evaluation include evidence beyond standard graph benchmarks, such as node classification on point clouds or discussion of applicability to higher-dimensional complexes?

---

**Principle 2: Computational Tractability and Scalability-Aware Design for Higher-Order Candidate Cell Elicitation in Learnable Topological Lifting**

**Definition:**  
Learnable lifting introduces additional computational stages—such as embedding-based nearest-neighbor search, cycle-basis computation, or powerset enumeration—that can incur significant time and memory overhead relative to static liftings. Reviewers consistently flagged whether these costs are modest enough to justify deployment, particularly for cell complexes where cycle-basis extraction scales cubically with the number of nodes. A rigorous evaluation must therefore explicitly characterize complexity, report wall-clock runtime and peak memory against static baselines, and demonstrate concrete mechanisms (e.g., sparsity regularization, adaptive neighborhood size bounds, deterministic thresholding) that control combinatorial growth. Without such analysis, claims of scalability remain unsubstantiated, and the method risks being confined to small molecular benchmarks.

**Core Evaluation Criteria:**
- **Explicit Complexity Characterization**: Does the paper provide formal or empirical analyses of time and memory complexity for each domain, contrasting learnable lifting with static counterparts (e.g., k-hop, cycle lifting)?
- **Scalability Safeguards**: Are there explicit algorithmic or regularization mechanisms to limit candidate-set explosion, such as bounding maximum neighbor counts, thresholding acceptance probabilities, or approximate cycle enumeration?
- **Empirical Overhead Justification**: Do runtime and memory experiments demonstrate that overhead is moderate (e.g., within a small multiplicative factor) and commensurate with performance improvements across datasets of varying scale?

---

**Principle 3: Comprehensive Fair Benchmarking of Learnable Lifting Against Static Topological, Differentiable Rewiring, and Vanilla GNN Baselines**

**Definition:**  
The topological deep learning literature currently lacks extensive, rigorous, and fair comparisons between higher-order message-passing networks and standard GNNs. Consequently, any proposal for learnable lifting must be evaluated on the breadth and fairness of its empirical benchmark. This includes comparing against not only the static lifting procedures native to TNNs (e.g., cycle, k-hop, kernel) but also against strong vanilla GNNs (GCN, GIN, GAT) under identical hyperparameter tuning and data-split protocols, as well as against related learnable structure methods such as differentiable pooling, probabilistic graph rewiring, and prior domain-specific learnable liftings. Furthermore, reviewers expect clear explanations when baselines underperform or exhibit high variance, and a candid discussion of the conditions (e.g., homophilic vs. heterophilic, graph-level vs. node-level) under which the added complexity of learned topology is warranted.

**Core Evaluation Criteria:**
- **Baseline Diversity**: Does the evaluation encompass (i) static TNN liftings, (ii) strong vanilla GNNs with comparable tuning, (iii) other learnable topology methods (e.g., DCM, IPR-MPNN, DiffPool), and (iv) relevant TDA-based approaches where appropriate?
- **Protocol Fairness**: Are data splits, model selection procedures, seeds, and hyperparameter grids held constant across all methods, with explicit disclosure of any deviations?
- **Failure Explanation and Scope Delimitation**: Does the paper explain anomalous baseline results (e.g., kernel lifting failures) and clearly state the regimes (dataset properties, task types) where learnable lifting offers decisive advantages over simpler alternatives?

---

**Principle 4: Theoretical Grounding and Mechanistic Attribution of Task-Adaptive Topology Benefits Beyond Empirical Accuracy Gains**

**Definition:**  
Because learnable lifting introduces stochastic discrete decisions into the model pipeline, empirical accuracy alone is insufficient; reviewers demand theoretical or deeply mechanistic justification for why task-adaptive topology construction improves learning. This includes analyzing the expressive power of the learned lifting relative to static alternatives, characterizing the stability and variance induced by Bernoulli or Gumbel-Softmax sampling, and distinguishing genuine topological benefits—such as mitigating over-squashing by creating shortcut higher-order cells—from generic increases in model capacity. The principle requires authors to move beyond correlational claims (e.g., “better accuracy implies better lifting”) and establish causal or mathematical links between the learned structure and information-flow properties of the downstream network.

**Core Evaluation Criteria:**
- **Expressiveness and Capacity Analysis**: Does the work discuss whether the learned lifting strictly expands the hypothesis space relative to static liftings or 1-WL GNNs, and are there theoretical intuitions or proofs provided?
- **Stability of Stochastic Topology**: Are the variance and stability of the sampling process analyzed, and are variance-reduction techniques (e.g., deterministic thresholds, annealing, straight-through estimators) ablated and justified?
- **Mechanistic Causal Claims**: Does the paper distinguish performance gains from stability gains, and does it provide qualitative or theoretical evidence (e.g., bottleneck reduction, express-lane cells) linking learned topology to specific information-flow improvements rather than relying solely on end-task metrics?

---

**Principle 5: Robustness and Sensitivity Control of Stochastic Learnable Lifting to Backbone Architectures, Hyperparameters, and Sampling Variance**

**Definition:**  
A learnable lifting framework is inherently coupled to the quality of its backbone node embeddings, the architecture of its acceptance-probability networks, and the stochasticity of its sampling stage. Therefore, its evaluation must rigorously assess robustness across these dimensions. Reviewers specifically questioned sensitivity to the choice of backbone GNN (e.g., GPS vs. GIN), the aggregation function used to pool node embeddings into cell representations, and the impact of hyperparameters such as maximum neighborhood size and acceptance thresholds. High-quality research in this area should demonstrate that performance is not an artifact of a single fortuitous backbone or hyperparameter setting, and should compare deterministic against stochastic variants to isolate the benefit of learned adaptive sampling from the noise it introduces.

**Core Evaluation Criteria:**
- **Backbone Architecture Sensitivity**: Are results reported across multiple backbone embedding generators (e.g., GIN, GPS, Graph Transformers) to verify that lifting improvements are not contingent on a single, highly expressive encoder?
- **Stochastic vs. Deterministic Ablation**: Does the paper include a direct comparison between fully stochastic sampling and deterministic thresholding to quantify the trade-off between adaptivity and variance?
- **Hyperparameter Stability**: Are ablation studies or sensitivity analyses provided for key additional hyperparameters (e.g., k_max, regularization strengths, MLP depths), and is the search space justified as modest rather than prohibitively large?
- **Consistency and Variance Reporting**: Are results reported with standard deviations over multiple seeds, and is the variance attributable to sampling noise distinguished from backbone initialization noise?