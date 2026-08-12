**Principle 1: Principled Design Justification and Component Ablation in MCTS-Embedded RLVR Training for LLM Reasoning**

**Definition:**  
When integrating Monte Carlo Tree Search into the RLVR training loop, reviewers expect rigorous justification for each algorithmic choice—such as global frontier selection, entropy-based negative trajectory selection, and asymmetric q-value updates—to avoid the appearance of ad hoc engineering. The methodology must be decomposed into independently testable components, with ablation studies demonstrating that no partial configuration achieves the full performance of the complete system. Authors must ground hyperparameters like search depth or expansion length in empirical data distributions or theoretical trade-offs rather than arbitrary constants. This principle ensures that gains are attributed to deliberate algorithmic innovations rather than accidental interactions, uncontrolled hyperparameter choices, or black-box reasoning that conflates stabilization with exploration.

**Core Evaluation Criteria:**
- **Grounding of Design Choices**: Are key hyperparameters (e.g., maximum search depth, expansion width) derived from task-specific data statistics or principled trade-off analyses, rather than presented as unmotivated constants?
- **Comprehensive Component Ablation**: Does the work isolate the marginal contribution of each module (frontier selection, credit assignment rule, replay buffer) to show that partial variants fail to match the full system?
- **Sensitivity and Robustness Analysis**: Are critical coefficients (e.g., frontier priority weights, filtering thresholds, clipping bounds) tested across a range of values to demonstrate that the chosen setting is stable and near-optimal?
- **Causal Attribution Clarity**: Does the work explicitly distinguish whether performance gains originate from improved exploration mechanisms, credit assignment refinements, or auxiliary stabilization heuristics?

---

**Principle 2: Wall-Clock Efficiency and GPU-Bound Overhead Analysis of Training-Time Tree Search in LLM Reasoning**

**Definition:**  
Because LLM training is fundamentally GPU-bound, embedding tree search into the RLVR loop invites intense scrutiny about whether algorithmic benefits are negated by structural overhead. Reviewers demand evidence that MCTS bookkeeping—such as frontier maintenance, node visitation tracking, and q-value backpropagation—consumes negligible wall-clock time relative to policy forward passes. The evaluation must demonstrate compute-normalized superiority over extended training baselines, proving that efficiency gains arise from better exploration quality rather than systems-level optimizations. A rigorous analysis should include asymptotic complexity arguments, empirical profiling of CPU versus GPU time, and throughput assessments under realistic batching constraints. Without such evidence, claims of improved efficiency remain unconvincing.

**Core Evaluation Criteria:**
- **Fine-Grained Computation Profiling**: Does the work provide a per-step time breakdown separating GPU inference from CPU-side tree operations, showing that search maintenance contributes minimally to total cost?
- **Budget-Matched Comparisons**: Are results compared against extended-training or search-free baselines under identical compute budgets, demonstrating superior accuracy per GPU hour rather than per training step?
- **Throughput and Scalability Assessment**: Does the analysis address structural bottlenecks such as incremental prefix prefill costs, sequential node dependencies, and reduced batching efficiency inherent to tree expansion?
- **Asymptotic and Empirical Scalability**: Is there evidence that tree-operation overhead remains negligible as search depth, branching width, or problem difficulty increase?

---

**Principle 3: Stability and Edge-Case Robustness of Fine-Grained Credit Assignment Across Tree-Structured Reasoning Paths**

**Definition:**  
Training-time MCTS introduces hierarchical, node-level credit assignment that must remain stable even under extreme conditions—such as trees where all trajectories are incorrect, severe q-value outliers, or incomplete reasoning chains. Reviewers evaluate whether asymmetric backup rules, soft clipping, and normalization strategies robustly separate positive and negative value regions without collapsing or inflating response lengths. The mechanism must be transparent enough to distinguish credit-assignment quality from exploration coverage. Furthermore, the work must demonstrate that its credit propagation does not silently rely on always having successful traces available, but instead provides stable learning signals in borderline failure scenarios.

**Core Evaluation Criteria:**
- **Mechanistic Clarity of Backup Rules**: Is there a clear explanation of how terminal rewards propagate to intermediate nodes, including sign-based separation of correct versus incorrect subtrees and handling of incomplete paths?
- **Robustness to All-Negative Rollouts**: Does the method remain stable when no correct solution exists in a given search, ensuring negative signals suppress high-risk prefixes without destructive gradient updates or policy collapse?
- **Empirical Stability Evidence**: Are q-value distributions, clipping saturation rates, gradient norms, and response-length trajectories reported to confirm the absence of mode collapse or runaway variance?
- **Orthogonality to Exploration Gains**: Does the work verify that credit-assignment components (e.g., mean-only normalization, constrained updates) are necessary even when frontier selection provides the dominant performance gain?

---

**Principle 4: Adaptive Curriculum and Anti-Forgetting Design in Progressive Difficulty Filtering for Search-Driven RLVR**

**Definition:**  
A training-time search framework must intelligently allocate computation toward problems that remain unsolved while preserving previously learned capabilities. Reviewers assess the design of replay buffers, solution caching, and hard-example filtering thresholds to determine whether the curriculum adapts effectively as the model improves. The mechanism should exhibit a monotonic shift from unsolved to cached problems and prevent catastrophic forgetting of solved instances without introducing excessive scheduling complexity that obscures the core contribution. The rationale for filtering criteria—whether fixed or adaptive—must be transparently linked to training dynamics and verified empirically across rounds.

**Core Evaluation Criteria:**
- **Filtering Rationale and Threshold Justification**: Is there a clear motivation for the difficulty-filtering threshold or metric that defines the active training set, and is it shown to be effective without requiring opaque adaptive scheduling?
- **Evolution of Cached versus Unsolved Problems**: Does the work quantify how the ratio of cached solutions to unsolved problems changes across training rounds, validating progressive concentration on tail cases?
- **Anti-Forgetting Evidence**: Is there empirical demonstration that replay-buffer caching prevents performance degradation on previously solved problem distributions or earlier training iterations?
- **Attribution Purity**: Does the adaptive curriculum design avoid introducing confounding hyperparameters that make it impossible to attribute gains specifically to search-based exploration?

---

**Principle 5: Compute-Normalized Benchmarking and Breadth-vs-Depth Scaling Evaluation in Saturated RLVR Regimes**

**Definition:**  
When evaluating search-augmented RLVR on frontier models, reviewers prioritize compute-normalized metrics over raw accuracy because gains at saturation are naturally small and difficult to obtain. The work must convincingly establish a paradigm shift from "depth scaling" (more training steps) to "breadth scaling" (structured exploration per step) by showing that brute-force extended training plateaus while algorithmic exploration yields superior GPU-hour efficiency. This requires cross-benchmark generalization analysis, characterization of problem types that most benefit from tree-based exploration, and explicit isolation of MCTS-specific contributions from adaptive training strategies alone. Claims of a new scaling direction must be supported by rigorous efficiency accounting rather than marginal accuracy deltas in isolation.

**Core Evaluation Criteria:**
- **Compute-Equivalent Plateau Analysis**: Does the work demonstrate that extended training baselines plateau despite massively increased GPU hours, while the proposed method achieves higher accuracy with substantially fewer resources?
- **Benchmark-Specific Gain Characterization**: Are improvements analyzed relative to base-model headroom across diverse tasks, clarifying whether gains stem from amplifying latent capabilities versus compensating for domain shifts?
- **Cross-Benchmark Consistency**: Do gains generalize across independent reasoning benchmarks to rule out dataset-specific overfitting or evaluation artifacts?
- **Paradigm Distinction and Isolation**: Is the distinction between depth scaling and breadth scaling articulated clearly, with empirical evidence that search-based exploration provides unique benefits not achievable by smarter filtering or adaptive sampling alone?