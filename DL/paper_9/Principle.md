**Principle 1: Compositional Novelty Justification and Ablative Isolation of Historical Activation Aggregation in Graph Readout**

**Definition:**  
Graph readout mechanisms frequently recombine well-studied primitives such as self-attention, residual connections, and positional encodings. Because these components are individually mature, reviewers demand that authors rigorously demonstrate how their specific configuration produces emergent capabilities that existing pooling architectures cannot replicate. The evaluation centers on whether the paper identifies a genuine functional gap—such as the need to decouple temporal aggregation across depth from spatial aggregation across nodes—and proves that this gap is closed by the proposed composition rather than by any single ingredient. Authors must provide targeted ablations that isolate the synergistic effect of the full design against degenerate variants (e.g., uniform averaging, random coefficient assignment, or removal of one stage). Furthermore, the work should explicitly position itself against the closest structural cousins to preempt reviewer skepticism about whether the contribution is merely a repurposing of standard layers. The principle ensures that compositional innovation is evaluated by its functional added value, not by the novelty of its atomic operations.

**Core Evaluation Criteria:**
- **Functional Gap Identification**: Does the paper clearly articulate a capability that existing pooling/readout methods lack (e.g., dynamic trajectory filtering over depth, signed reweighting of historical layers) and demonstrate that the proposed composition uniquely addresses it?
- **Ablative Discrimination**: Are there systematic ablations showing that the complete two-stage (or multi-stage) design outperforms degenerate variants such as uniform averaging, randomized attention weights, or independent application of the same components?
- **Positioning Against Structural Cousins**: Does the work explicitly differentiate itself from closely related methods (e.g., JKNet, GMT, DiffPool) in terms of how intermediate activations are reused, rather than claiming novelty solely from applying attention to historical states?
- **Beyond-Component Narrative**: Does the submission avoid framing its contribution as “using attention over layers” and instead explain why the specific parameterization (e.g., signed normalization, query-from-final-layer design) yields qualitatively different readout behavior?

---

**Principle 2: Formal Mechanistic Grounding of Signed Depth-Trajectory Filtering for Over-Smoothing Mitigation in Deep GNNs**

**Definition:**  
When a pooling method claims to mitigate deep GNN pathologies such as over-smoothing by reusing intermediate activations, it must supply more than intuitive narratives or empirically observed correlations. Reviewers expect a formal or tightly argued mechanistic account that binds specific architectural choices—such as signed normalization, layer-wise gating, or query-key designs—to the preservation of discriminative signals across depth. The analysis should yield falsifiable, non-obvious predictions; for instance, proving that a readout retains node distinguishability provided at least one pre-saturated layer receives non-zero weight, or demonstrating that the aggregation implements a learnable finite-impulse-response filter over the depth trajectory. Diagnostic experiments (e.g., feature-distance tracking across layers, depth sweeps beyond 32 layers, and attention-weight trajectories during training) must corroborate the formal account. This principle guards against generic “attention as filter” arguments by demanding that the theoretical framework be specific enough to explain why the proposed method outperforms alternative aggregation schemes that also operate on historical layers.

**Core Evaluation Criteria:**
- **Non-Obvious Formal Claims**: Does the work present theorems, propositions, or rigorous derivations that are specific to the proposed method’s design (e.g., signed, sum-to-one coefficients enabling high-pass filtering over depth trajectories) rather than restating general properties of attention?
- **Causal Diagnostic Evidence**: Are there empirical diagnostics—such as inter-node feature distances at varying depths, learned coefficient visualizations, or controlled depth sweeps—that validate the mechanistic link between the method’s operations and the mitigation of over-smoothing?
- **Rejection of Generic Rationalization**: Does the paper avoid post-hoc reasoning where any performance improvement is attributed to “using more layers,” and instead establish a precise causal chain from trajectory filtering to preserved representational diversity?
- **Design-Motivation Alignment**: Is there a clear feedback loop where the theoretical analysis directly motivates the architectural choices (e.g., why signed normalization is necessary instead of softmax), and where ablations confirm that deviating from these choices undermines the theoretical guarantee?

---

**Principle 3: Memory-Compute Transparency and Practical Scalability Delimitation for Multi-Layer Activation Caching**

**Definition:**  
Architectures that cache or attend to full historical activation tensors across many message-passing layers introduce tangible memory and compute bottlenecks, particularly when node-wise attention scales quadratically with graph size. A rigorous submission must therefore present an unambiguous complexity analysis that separates the costs of activation storage (e.g., O(NLD)), layer-wise mixing, and spatial attention (e.g., O(N²D)), and it must avoid conflating asymptotic notation with practical wall-clock savings. Authors should report empirical training and inference runtimes across varying depths and graph sizes, and they must honestly delimit the intended operational regime—whether small molecular graphs, medium social networks, or million-node industrial graphs. If efficiency claims are made (e.g., frozen-backbone post-processing), the source of savings should be precisely identified, such as eliminating backpropagation through the backbone rather than reducing asymptotic complexity. Finally, concrete strategies for extending the method to larger graphs (sparse attention, hierarchical coarsening, trajectory truncation) should be articulated so that scalability limitations are transparent rather than obscured.

**Core Evaluation Criteria:**
- **Explicit Complexity Decomposition**: Does the paper clearly report the space complexity of caching historical activations and the time complexity of each attention stage, avoiding misleading claims that conflate reduced depth-coefficients with sub-quadratic node scaling?
- **Empirical Runtime and Memory Scaling**: Are training and inference runtimes reported for multiple depths and graph sizes, with scaling plots or tables that allow reviewers to assess practical feasibility on their target hardware?
- **Honest Scope Delimitation**: Does the work explicitly state the intended graph regime (e.g., small-to-medium graphs with tens to thousands of nodes) and acknowledge that industrial-scale graphs (millions of nodes) require additional techniques such as neighbor sampling or sparse attention?
- **Precise Efficiency Claims**: When modes such as frozen-backbone training are proposed, is the advantage accurately characterized (e.g., removing gradient storage through L message-passing layers) rather than misrepresented as an asymptotic improvement?

---

**Principle 4: Cross-Domain Benchmarking Rigor and Diagnostic Verification of Anomalous Performance Outliers in Graph Pooling**

**Definition:**  
Novel graph pooling methods are expected to demonstrate broad generalization across benchmark families (e.g., TU, OGB molecular, link prediction, node classification) and against a comprehensive, up-to-date set of baselines spanning hierarchical, attention-based, and kernel methods. Reviewers scrutinize whether the experimental protocol includes recent competitors and whether hyperparameters are selected fairly via validation performance using reported search ranges. Unusually large performance deviations—such as a 16-point accuracy improvement on a single dataset—must be subject to intensive diagnostic scrutiny, including component ablations, checks for data leakage, and analysis of variance across seeds. The submission should also provide complete training configurations and, ideally, executable code to reproduce the core results table. This principle ensures that empirical gains are robust, reproducible, and not artifacts of narrow benchmarking or incomplete comparisons.

**Core Evaluation Criteria:**
- **Breadth and Recency of Baselines**: Does the evaluation span multiple benchmark families with diverse graph properties, and does it include recent pooling/readout competitors rather than only classical sum/mean/max or outdated methods?
- **Fair Hyperparameter Protocol**: Are hyperparameter search ranges, selection criteria (e.g., validation-based), and baseline configurations fully documented, following official implementations or published settings where possible?
- **Outlier Analysis**: Are anomalously large gains on specific datasets investigated through targeted ablations, statistical variance reports, and explicit arguments ruling out implementation artifacts or data leakage?
- **Reproducibility Artifacts**: Is source code provided with clear instructions (e.g., a single command) to reproduce the main results table, ensuring that the community can independently verify the reported numbers?

---

**Principle 5: Interpretability and Functional Characterization of Learnable Depth-Filters Over GNN Layer Trajectories**

**Definition:**  
Because layer-wise attention over historical activations functions as a dynamic reweighting of depth-specific representations, reviewers expect authors to characterize what the model has learned functionally—interpreting the coefficients as low-pass, high-pass, or band-pass filters over the GNN computation trajectory. The paper should visualize learned attention distributions across layers for multiple datasets and training phases, verifying that the weights respond to depth-dependent phenomena such as over-smoothing rather than collapsing to uniform or trivial solutions. Interpretability should extend to the node-wise stage as well, with illustrative cases (e.g., barbell graphs, critical-node dependencies) showing how spatial attention complements the depth filter. This principle demands that historical aggregation be presented not as an opaque black-box mixture, but as an interpretable, task-adaptive trajectory filter whose behavior can be validated against the theoretical motivations provided in the paper.

**Core Evaluation Criteria:**
- **Depth-Trajectory Visualization**: Does the paper visualize learned layer-wise weights across depths for multiple datasets and backbones, showing non-trivial patterns (e.g., up-weighting early layers in deep networks) rather than uniform or random distributions?
- **Functional Filter Narrative**: Is there a conceptual framework (e.g., finite-impulse-response filter, low-pass/high-pass over depth) that explains what the layer aggregation is doing, and is this framework validated by the visualized weights?
- **Node-Wise Interpretability**: Are there case studies or toy constructions (e.g., bridge/barbell graphs) demonstrating that the node-wise attention stage captures non-uniform spatial importance that mean pooling cannot approximate?
- **Behavioral Validation**: Do the interpretability findings align with the theoretical claims (e.g., early-layer weights increase as backbone depth grows and over-smoothing intensifies), confirming that the model behaves according to its motivated design rather than learning spurious correlations?