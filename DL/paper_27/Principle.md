**Principle 1: Representational Sufficiency of Higher-Order Hypergraph Interaction Modeling Over Pairwise Approximation in Dense Multi-Agent Coordination**

**Definition:**  
In multi-agent coordination problems such as MAPF, optimal solutions often require simultaneous reasoning about groups of interacting agents, yet prevailing graph neural networks are restricted to pairwise message passing. This principle evaluates whether a submission rigorously establishes that hypergraph-based higher-order interactions provide a necessary and principled representational advantage over pairwise alternatives. Reviewers expect authors to move beyond intuitive motivation and identify concrete failure modes of pairwise modeling—such as attention dilution in dense neighborhoods or the combinatorial explosion of depth needed to approximate group constraints via stacked pairwise layers. The work should provide theoretical or rigorous empirical justification, such as expressivity arguments, informal proofs of attention entropy dynamics, or complexity analyses, showing that hyperedges natively encode coordination patterns which graphs cannot efficiently approximate. Ultimately, the evaluation centers on whether the hypergraph structure is mechanistically responsible for performance gains rather than serving as an opaque engineering embellishment.

**Core Evaluation Criteria:**
- **Clarity of Mechanistic Limitation**: Does the work explicitly diagnose why pairwise models fail for the target task, such as softmax dilution over dense neighborhoods or exponential growth in required network depth to emulate group interactions?
- **Theoretical or Expressivity Justification**: Is there formal or rigorous informal analysis (e.g., attention-dilution proofs, Weisfeiler-Lehman-style hierarchies, or complexity arguments) establishing a representational gap between hypergraphs and graphs?
- **Empirical Isolation of Representational Effects**: Are gains demonstrated to stem from the hypergraph structure itself under controlled conditions, rather than from confounding factors like increased model capacity, auxiliary modules, or data-scale differences?
- **Alignment with Task Combinatorial Structure**: Does the hypergraph design reflect domain-specific group-coupled constraints (e.g., simultaneous vertex occupancy in MAPF) rather than applying generic clustering heuristics without justification?

---

**Principle 2: Principled Justification and Computational Scalability of Dynamic Hypergraph Construction in Large-Scale Multi-Agent Systems**

**Definition:**  
Unlike standard graphs where edges arise naturally from pairwise proximity, hypergraphs require explicit construction algorithms whose design choices profoundly affect model behavior and runtime. This principle assesses whether the proposed hypergraph generation strategy is algorithmically principled, computationally scalable, and semantically aligned with the coordination task. Reviewers scrutinize the rationale behind selecting specific grouping heuristics—whether spatial clustering, Voronoi partitioning, or shortest-path cliques—and demand evidence that these choices capture meaningful agent coalitions rather than arbitrary geometric partitions. A thorough submission must characterize the asymptotic and wall-clock costs of hypergraph construction as agent counts and map sizes grow, demonstrating that this overhead does not erase the inference-time advantages of a compact model. Furthermore, the work should analyze sensitivity to hyperparameters such as cluster counts, soft-boundary thresholds, and communication radii, providing practitioners with guidance on how to instantiate the method on new domains.

**Core Evaluation Criteria:**
- **Algorithmic Justification**: Are the chosen hypergraph generation strategies motivated by properties of the task domain (e.g., collision corridors, joint path dependencies) rather than presented as generic off-the-shelf clustering procedures?
- **Scalability and Complexity Analysis**: Does the paper provide formal complexity bounds and empirical runtime measurements showing that construction cost remains negligible or manageable relative to model inference at scale?
- **Robustness to Design Choices**: Are multiple generation strategies compared across benchmarks, and is guidance provided for selecting among them? Are critical hyperparameters like overlap ratios and cluster counts systematically ablated?
- **Dynamic Adaptation Efficiency**: Does the construction process support efficient updates under dynamic agent configurations without requiring full per-timestep recomputation that would prohibit online deployment?

---

**Principle 3: Rigorous Disentanglement of Core Architectural Inductive Biases from Auxiliary Training Regimes and Inference Enhancements**

**Definition:**  
State-of-the-art results in imitation learning for MAPF often emerge from compound pipelines that combine architecture, large-scale expert data, online fine-tuning, post-training quality optimization, and inference-time safety wrappers like collision shielding or temperature sampling. This principle demands that authors rigorously isolate the marginal contribution of the proposed architectural inductive bias—here, hypergraph attention—from these confounding factors. Reviewers expect controlled ablations where the core architecture is evaluated under identical training data scales, optimization schedules, and post-processing regimes as baseline methods. Claims regarding superior data efficiency or parameter efficiency must be validated by training baseline architectures on the same reduced datasets and comparing like-for-like, rather than contrasting a heavily tuned small model against an undertrained or differently post-processed large model. Only through such disentanglement can the community adjudicate whether the advance is representational or merely an artifact of engineering orchestration.

**Core Evaluation Criteria:**
- **Controlled Architectural Ablations**: Does the ablation study cleanly isolate the hypergraph component by holding constant all training procedures, data scales, and inference augmentations when comparing against graph-based baselines?
- **Fairness of Baseline Modifications**: When applying improvements such as collision shielding or temperature sampling to existing methods, are their individual effects ablated and reported across complete metric suites?
- **Data and Parameter Parity Controls**: Are efficiency claims supported by training baseline models on identical data budgets and parameter scales, ensuring comparisons isolate architectural representational power?
- **Pipeline Decomposition**: Is the final performance decomposed into contributions from the core architecture, observation encoding, post-training fine-tuning, and inference-time sampling to prevent over-attribution to any single factor?

---

**Principle 4: Depth of Mechanistic Interpretability and Diagnostic Failure Mode Analysis Beyond Aggregate Performance Metrics**

**Definition:**  
Because learned MAPF policies are heuristic approximations to an NP-hard problem, aggregate success rates and sum-of-costs metrics alone are insufficient for establishing scientific understanding. This principle evaluates whether authors provide mechanistic evidence that their model actually exploits higher-order structures for coordination, using tools such as attention visualizations, normalized entropy statistics, Shapley-value influence quantification, or qualitative trajectory inspections. Equally critical is a diagnostic analysis of failure modes—deadlocks, livelocks, timeouts, and collision cascades—particularly in high-density regimes where the method approaches its performance frontier. Reviewers look for systematic quantitative evidence that claimed phenomena, such as attention dilution in pairwise models, occur reliably across real benchmark instances rather than only in hand-crafted toy scenarios. Such transparency builds trust in the method's scope and points toward concrete avenues for improvement.

**Core Evaluation Criteria:**
- **Mechanistic Evidence of Higher-Order Utilization**: Does the paper analyze internal representations (e.g., attention entropy, Shapley values, hyperedge activation patterns) to verify the model leverages group-level reasoning rather than spurious local cues?
- **Systematic Phenomenon Validation**: Are claims such as attention dilution supported by quantitative statistics aggregated over large benchmark samples, rather than anecdotal hand-crafted examples alone?
- **Failure Mode Taxonomy**: Are failure cases categorized by type (e.g., deadlock, livelock, maximum timestep violation) and frequency in dense or challenging scenarios, clarifying the method's operational limitations?
- **Interpretability and Causal Attribution**: Do the interpretability analyses establish a causal link between the hypergraph structure and specific desirable coordination behaviors, avoiding post-hoc rationalization?

---

**Principle 5: Generalization Capacity and Out-of-Distribution Robustness Across Environment Morphologies and Task Variants**

**Definition:**  
Multi-agent pathfinding methods are frequently benchmarked on specific grid-world categories that may encourage overfitting to obstacle densities, spatial scales, or interaction patterns. This principle assesses whether the evaluation protocol spans a diverse spectrum of environment morphologies—including sparse and dense mazes, rooms, warehouses, and city maps—and whether authors characterize performance scaling as agent density increases toward the task's combinatorial limits. Reviewers particularly value evidence of out-of-distribution robustness, such as transfer to unseen map sizes, dynamic obstacles, or significantly larger agent teams than those encountered during training. A strong submission does not merely report high average performance on standard benchmarks but explicitly discusses where the inductive bias transfers, where it breaks down, and whether the hypergraph construction mechanism is sufficiently generic to accommodate novel spatial structures without redesign.

**Core Evaluation Criteria:**
- **Benchmark Diversity and Density Scaling**: Are experiments conducted across multiple distinct map categories with varying obstacle densities, and are performance trends analyzed as agent counts scale to high-density regimes?
- **Out-of-Distribution Evaluation**: Does the work test or discuss generalization to unseen map types, non-grid environments, dynamic obstacles, or agent counts outside the training distribution?
- **Transferability of Inductive Bias**: Is there evidence or principled argument that the hypergraph construction and attention mechanisms generalize beyond the specific benchmark suite to other multi-agent coordination domains?
- **Transparency on Scope Limitations**: Does the paper candidly report scenarios or environment classes where performance degrades, rather than masking limitations through aggregated metrics?