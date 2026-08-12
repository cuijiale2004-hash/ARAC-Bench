**Principle 1: Structural Complexity Assessment via Reasoning Tree Topology for Curriculum Learning in RLVR**

**Definition:**  
In RLVR training for reasoning tasks, difficulty metrics must transcend path-based proxies (e.g., final accuracy, uncertainty, or gradient norms) and instead explicitly model the topological structure of the reasoning tree. The central premise is that the distribution and concentration of errors across tree nodes—rather than the mere correctness of leaf nodes—determines true learning difficulty and credit assignment efficiency. A query with low initial accuracy but highly concentrated errors in a few critical nodes may be far more learnable than a query with higher accuracy but diffusely scattered errors. Consequently, evaluation must focus on whether the proposed structural metric mechanistically captures this distinction and provides a principled proxy for expected accuracy gain under limited policy refinement budgets.

**Core Evaluation Criteria:**
- **Distinctiveness from Path-Based Proxies**: Does the metric explicitly leverage parent-child branching relationships and subtree correctness distributions, rather than reducing reasoning to a flat sequence or scalar outcome?
- **Predictive Validity for Learning Dynamics**: Is there empirical evidence (e.g., learning curve analysis, correlation with early-training gains) that the structural metric better predicts which samples yield steep learning progress than accuracy-based or uncertainty-based alternatives?
- **Mechanistic Interpretability of Node-Editing**: Does the work provide a formal or intuitive framework linking the tree metric to the RLVR optimization process, such as explaining how concentrated errors enable efficient credit assignment while diffuse errors create noisy gradient signals?

---

**Principle 2: Offline Approximation Fidelity and Computational-Performance Trade-off Management in Tree Construction**

**Definition:**  
Since exact reasoning trees are combinatorially intractable, practical methods must approximate them via bounded sampling parameters (branching factor, depth, and token interval). The quality of the entire scheduling pipeline hinges on whether this approximation preserves the critical decision points of the true reasoning process without imposing prohibitive offline computational burdens relative to the RL training budget. Reviewers consistently raised concerns about whether coarser approximations oversimplify reasoning and whether the cost of building larger trees is justified. Therefore, research in this area must rigorously characterize the sensitivity of downstream performance to approximation granularity and transparently report the cost-performance trade-off.

**Core Evaluation Criteria:**
- **Granularity Appropriateness and Sensitivity**: Are ablation studies provided for key approximation parameters (e.g., token interval, branching factor, depth)? Is there evidence that excessively coarse intervals fail to capture critical branching points, and that performance stabilizes within a principled range?
- **Cost Transparency and Scalability**: Does the work quantitatively report offline tree construction cost (time and memory) relative to standard RL training epochs? Is the overhead characterized as a fixed preprocessing cost or a per-epoch burden, and is it shown to be manageable for standard academic hardware?
- **Principled Parameter Selection Guidance**: Does the work offer heuristic or empirical guidance for choosing approximation parameters based on dataset properties (e.g., average solution length) or model scale, rather than treating them as opaque tuning knobs?

---

**Principle 3: Empirical Stability, Fair Baseline Alignment, and Cross-Domain Generalization in RLVR Scheduling**

**Definition:**  
Data scheduling methods in RLVR are highly sensitive to training conditions, including dataset composition, evaluation randomness (especially at high sampling temperatures), and model architecture. To ensure that reported gains are attributable to the scheduling algorithm rather than confounding factors, reviewers demanded strict experimental hygiene: identical training corpora for all baselines, variance quantification across random seeds, and evaluation metrics that mitigate instability (e.g., averaging over multiple generations). Furthermore, because a scheduling principle risks being an artifact of a specific model-task pairing, its validity must be tested across diverse model families and reasoning domains.

**Core Evaluation Criteria:**
- **Evaluation Stability and Variance Reporting**: Are final results reported with robust averaging protocols and accompanied by variance statistics across multiple independent runs? Is the experimental setup designed to reduce evaluation randomness or at least account for it transparently?
- **Baseline Fairness and Reproducibility**: Are all baseline methods compared under the exact same training data, model checkpoint, and RL hyperparameters? If baseline results are drawn from prior literature, are they reproduced on the identical training set to eliminate data-regime confounders?
- **Model-Agnostic and Cross-Domain Validation**: Does the method demonstrate consistent improvements across multiple model architectures and reasoning domains (e.g., mathematical reasoning and code generation), or is its efficacy confined to a single specialized setting?

---

**Principle 4: Curriculum Directionality, Temporal Adaptation, and Mechanistic Justification of Scheduling Design**

**Definition:**  
A tree-based curriculum schedule is not merely a ranking function; it embodies causal claims about the optimal temporal ordering of training samples. The field must evaluate whether the proposed directionality—typically from structurally simple to complex—is empirically necessary or merely heuristic. This requires direct ablations of alternative orderings (e.g., reverse or random schedules) to expose failure modes such as reward signal starvation in early training. Additionally, because model competence evolves, the field must assess whether difficulty metrics should remain static or be dynamically recomputed, weighing the marginal performance benefit against computational cost. Finally, the underlying node-editing motivation must be validated beyond intuition through mechanistic evidence or formal frameworks.

**Core Evaluation Criteria:**
- **Directionality Causality and Failure Mode Analysis**: Is the assumed easy-to-hard progression explicitly validated against reverse-order or anti-curriculum baselines? Does the work analyze and explain the mechanistic cause of failure for suboptimal orderings (e.g., lack of positive reward signals when starting with structurally complex samples)?
- **Static versus Dynamic Metric Validity**: Is the choice between a static (pre-computed) and dynamic (periodically updated) difficulty metric empirically justified? Does the work demonstrate that dynamic updates provide negligible marginal gains relative to their substantial recomputation cost?
- **Theoretical and Mechanistic Grounding**: Does the work provide a formal framework (e.g., optimizing expected accuracy gain under a node-editing budget) or mechanistic evidence (e.g., tracking corrective node metrics during training) to substantiate the claim that RLVR operates as reasoning tree optimization, rather than relying solely on post-hoc empirical correlation?

---

**Principle 5: Intellectual Positioning and Explicit Differentiation from Existing Curriculum Learning and Tree-Search Paradigms**

**Definition:**  
Research that introduces tree-structured difficulty metrics for RLVR must rigorously situate its contributions within the broader landscape of curriculum learning, active learning, and tree-search-based reasoning methods. The novelty of a structural metric is diminished if its relationship to existing scalar proxies—such as accuracy gradients, semantic entropy, or uncertainty—and to algorithmic relatives—such as Monte Carlo Tree Search or process reward models—is left ambiguous. A thorough evaluation demands that authors articulate not only what their metric measures differently, but also why existing methods fail to approximate this structural information and how the proposed framework could interface with complementary techniques like explicit confidence modeling or value estimation at tree nodes.

**Core Evaluation Criteria:**
- **Comparative Analysis with Existing Proxies**: Does the work provide a nuanced comparison with existing difficulty estimators (e.g., accuracy-based curricula, gradient-based prioritization, uncertainty methods) that explains their structural blindness rather than merely listing them?
- **Connection to Tree-Search and Process Supervision Literature**: Does the work discuss its relationship to Monte Carlo Tree Search, process reward models, or step-level verification? Are potential synergies or distinctions (e.g., offline static trees versus online search) explicitly analyzed?
- **Scope of Applicability and Limitations**: Does the work clearly state the assumptions required for its structural metric (e.g., verifiable final rewards, tree-structured solution spaces) and acknowledge domains where the reasoning tree abstraction may break down or require non-trivial adaptation?