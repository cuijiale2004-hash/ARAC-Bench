**Principle 1: Scalability Analysis and Subset-Size Sensitivity in Heterogeneous Multi-Agent LLM Pools**

**Definition:**  
The work must rigorously evaluate how the proposed orchestration framework behaves as the agent pool grows beyond small fixed sets to tens or hundreds of heterogeneous models. Reviewers expect analyses of whether performance scales monotonically with pool size, how the subset selection parameter is determined or adapted, and whether efficiency gains persist when agents vary widely in capability, domain specificity, and model scale. It is crucial to distinguish between homogeneous replication of a single capable model and heterogeneous composition of diverse specialists, because scaling dynamics differ fundamentally between these regimes. The evaluation should demonstrate that selective activation remains beneficial rather than degenerating into noise or redundant computation as the pool expands.

**Core Evaluation Criteria:**
- **Large-Pool Generalization**: Does the work provide empirical or analytical evidence for behavior with large agent pools, or does it explicitly justify and scope the evaluation to smaller pools?
- **Subset Selection Justification**: Is the choice of subset size principled, adaptive to query complexity, or at least sensitivity-analyzed across varying pool sizes and task types?
- **Heterogeneity vs. Homogeneity Disentanglement**: Are experiments designed to separate the effects of adding redundant copies of one model from adding diverse specialists, and does the work explain non-monotonic scaling?
- **Computational Budget Scaling**: Is the inference cost analyzed as a function of total pool size, showing that the subset-selection overhead does not overwhelm the savings from reduced active agents?

---

**Principle 2: Robustness and Failure-Mode Analysis of Metadata-Driven Node Sampling and Dynamic Edge Construction**

**Definition:**  
Since graph-based multi-agent frameworks often rely on lightweight metadata or peer-evaluation heuristics to dynamically construct communication graphs, the work must demonstrate resilience to imperfect agent descriptions and suboptimal peer-ranking outcomes. Reviewers expect explicit stress tests that simulate real-world conditions where agent metadata is noisy, incomplete, or outdated, as well as an analysis of how graph topology degrades or recovers under such perturbations. The edge construction procedure—whether fully connected, thresholded, or meta-LLM generated—must be unambiguously defined, because ambiguity becomes a critical design liability when moving beyond small graphs. Cataloging concrete failure modes arising from incorrect node sampling or edge mis-weighting is essential to establish reliability.

**Core Evaluation Criteria:**
- **Noise Injection Studies**: Are there controlled experiments that degrade model-card quality or inject irrelevant agents to measure performance decay and recovery?
- **Edge Construction Clarity**: Is the edge formation procedure explicitly defined, and is its sensitivity to ranking errors or incomplete metadata analyzed?
- **Failure Case Characterization**: Does the work catalog concrete failure modes and their root causes, and propose actionable mitigations?
- **Threshold Sensitivity**: Are hyperparameters such as relevance thresholds ablated to show how graph sparsity affects robustness and accuracy?

---

**Principle 3: Comprehensive and Fair Baseline Coverage Across Graph-Based, Workflow-Based, and Router-Based Orchestration Paradigms**

**Definition:**  
A graph-based multi-agent method must be evaluated against a representative spectrum of competing paradigms, including static graph structures, optimizable agent networks, automated workflow generators, and router-based ensemble selectors. Comparisons must control for agent identity, number, and prompt budget to ensure that observed gains stem from the coordination mechanism rather than confounding factors. The work should also clarify its relationship to training-free versus training-based approaches, as this distinction drastically changes deployment cost and black-box compatibility. Reviewers scrutinize whether the proposed method outperforms not only generic multi-agent baselines but also structurally similar graph-based collaborators.

**Core Evaluation Criteria:**
- **Paradigm Diversity**: Does the evaluation include graph-based static or optimizable baselines, workflow systems, router ensembles, and standard multi-agent baselines?
- **Controlled Fairness**: Are comparisons conducted with matched agent sets, identical prompt formats, and comparable token or decoding budgets?
- **Training-Free vs. Training-Based Distinction**: Is the overhead of precomputation, optimization, or dedicated routing model training explicitly contrasted with the proposed test-time approach?
- **Efficiency-Accuracy Trade-off**: Is performance improvement evaluated alongside computational cost to demonstrate practical utility beyond raw accuracy?

---

**Principle 4: Mechanistic Justification and Ablation of Directional Communication Protocols in Test-Time Agent Collaboration**

**Definition:**  
The proposed communication protocol—particularly directional or bidirectional message passing—requires rigorous ablation and conceptual justification beyond end-to-end performance gains. Reviewers assess whether the ordering of information flow is theoretically or empirically motivated, whether each directional component contributes uniquely, and what happens when the flow is inverted or truncated. Since these systems operate at test time without gradient-based optimization, empirical demonstration of the causal impact of each communication step is essential. The work must distinguish structured collaboration that preserves expert influence from spurious aggregation effects that could be replicated by simpler pooling.

**Core Evaluation Criteria:**
- **Directional Ablation**: Are the distinct contributions of each message-passing direction isolated and quantified, including the effect of reversing the intended flow?
- **Ordering Rationale**: Is there a clear explanation for why the specific communication ordering avoids noise amplification and preserves source-node authority?
- **Stepwise Stability**: Does the work empirically demonstrate stability across varying graph sizes, relevance thresholds, and task types without degenerate feedback loops?
- **Alternative Mechanism Comparison**: Is the bidirectional graph mechanism compared against simpler variants such as single-round pooling or unidirectional broadcast to prove its added value?

---

**Principle 5: Benchmark Scope and Task-Environment Alignment for Agentic Multi-Agent Collaboration Claims**

**Definition:**  
When a framework is positioned as a multi-agent agentic system, the choice of benchmarks must match the claimed capabilities. If the work relies solely on static question-answering or reasoning benchmarks, reviewers scrutinize whether the term agent is justified beyond simple LLM ensembling. Effective evaluation should extend to tasks requiring tool use, environment interaction, multi-turn decision-making, or dynamic role specialization, or the paper should explicitly limit its scope to test-time model orchestration. This principle ensures that methodological contributions are assessed against appropriately demanding environments that exercise true coordination, not just consensus aggregation.

**Core Evaluation Criteria:**
- **Task Diversity**: Does the evaluation span interactive or environment-grounded tasks in addition to static QA, or is the scope honestly delimited?
- **Agent Definition Consistency**: Does the paper clarify what constitutes an agent and ensure that benchmark choices are consistent with that definition?
- **Generalization Claim Scope**: Are claims about multi-agent collaboration scoped to the demonstrated setting, or are implications for broader agentic settings appropriately qualified?
- **Failure Mode Benchmarking**: Are failure cases drawn from diverse task types to reveal whether breakdowns stem from reasoning errors, communication failures, or environment misalignment?