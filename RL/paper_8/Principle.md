**Principle 1: Policy-Level Information Gain Maximization for Query Selection in Offline Preference-Based RL**

**Definition:**  
In offline preference-based reinforcement learning, the efficiency of query selection is determined not merely by the volume of preference data collected, but by the policy-relevance of the information each query provides. This principle evaluates whether a proposed exploration strategy targets the reduction of uncertainty in the optimal policy or value function, rather than indiscriminately reducing reward model uncertainty across the entire state space. Since the ultimate objective is policy optimization, queries that discriminate between candidate optimal policies are fundamentally more efficient than those that refine reward estimates in low-return regions. The principle demands both theoretical justification—such as bounds via the Eluder dimension of the value function class or information ratio—and empirical demonstration that the query strategy achieves steep performance gains with minimal queries compared to reward-disagreement or random selection. It also requires clear articulation of why policy-level information gain is superior to reward-level information gain in the offline setting where exploration is constrained to a fixed dataset.

**Core Evaluation Criteria:**
- **Policy vs. Reward Information Targeting**: Does the query mechanism explicitly select trajectory pairs that maximize disagreement among candidate optimal policies or value functions, rather than merely maximizing reward model entropy or disagreement?
- **Theoretical Sample Efficiency**: Are there formal guarantees (e.g., via Eluder dimension, information ratio, or diameter reduction of the policy uncertainty set) showing that the query complexity scales with the complexity of the optimal value function class rather than the reward class?
- **Empirical Query Efficiency**: Does the method demonstrate rapid performance saturation with extremely small query budgets (e.g., 10–20 queries) against baselines like random querying and reward-disagreement methods?
- **In-Dataset Feasibility**: Is the exploration strategy executable without access to an online simulator, i.e., does it select from a fixed offline dataset rather than requiring new environment interactions?

---

**Principle 2: Uncertainty-Aware Pessimistic Reward Utilization and Overoptimization Mitigation in Offline PbRL**

**Definition:**  
Offline PbRL is particularly vulnerable to overoptimization because learned reward functions extrapolate optimistically in regions poorly covered by preference data, leading to severe value overestimation. This principle assesses whether the proposed method provides a principled, empirically validated mechanism to counteract this phenomenon beyond naive reward penalty or fixed pessimism. The evaluation centers on whether the uncertainty proxy (e.g., ensemble variance) is rigorously connected to overestimation risk, whether the mitigation strategy (e.g., variance-based discount scheduling) is theoretically grounded in offline RL pessimism, and whether it is robust to noisy or inconsistent human annotations. Reviewers expect clear ablations comparing the proposed pessimism mechanism against alternatives such as worst-case ensemble minimization, reward penalization, or fixed small discount factors. Furthermore, the principle requires that the heuristic components of the method (e.g., static top-m% thresholds) are justified or ablated against adaptive continuous variants.

**Core Evaluation Criteria:**
- **Mechanistic Clarity of Uncertainty-Overestimation Link**: Does the work clearly explain why the chosen uncertainty proxy (e.g., value ensemble variance) correlates with overestimation, and avoid conflating variance with direct causation?
- **Comparative Pessimism Ablations**: Are alternative pessimistic strategies (e.g., worst-case optimization over ensembles, direct reward penalties, fixed discount reduction) fairly compared to demonstrate the superiority of the proposed mechanism?
- **Robustness to Annotation Noise**: Does the evaluation include non-synthetic, noisy preference sources (e.g., real human annotators) to validate that the pessimism mechanism prevents overoptimization under realistic label noise?
- **Threshold/Scheduling Rationality**: If using heuristic thresholds (e.g., top-m% variance), are they ablated across ranges, and are adaptive or continuous alternatives investigated to justify the design choice?

---

**Principle 3: Cross-Paradigm Baseline Rigor and Multimodal Evaluation Scope in Offline PbRL**

**Definition:**  
The offline PbRL landscape comprises multiple methodological paradigms, including two-stage reward-learning pipelines, one-stage direct policy optimization frameworks that bypass explicit reward modeling, and listwise or sequential query enhancement techniques. This principle demands that new algorithms are evaluated against representatives from all relevant paradigms under identical query budgets and offline data conditions, rather than solely against prior two-stage methods. Additionally, the experimental scope must extend beyond vector-based state spaces to validate generalization to high-dimensional visual inputs and, crucially, to genuine human feedback rather than synthetic preference oracles. The principle exists because algorithmic advantages observed under synthetic teachers and low-dimensional states often fail to transfer to pixel-based domains and noisy human annotators, which represent the core practical motivations for offline PbRL. Comprehensive evaluation must therefore demonstrate that performance gains are not artifacts of restricted environmental or feedback modalities.

**Core Evaluation Criteria:**
- **One-Stage Framework Comparison**: Are direct preference-based policy optimization methods (e.g., IPL, CPL, DPPO) included as baselines to test whether explicit reward learning provides any advantage?
- **Enhanced Query Mechanism Baselines**: Are advanced query-sampling methods such as listwise ranking or sequential ranked lists (e.g., LiRE) compared against the proposed exploration strategy?
- **Multimodal State Evaluation**: Does the experimental evaluation encompass both vector-state and pixel-based environments (e.g., Atari) to establish generalization across observation modalities?
- **Real Human Feedback Validation**: Are experiments conducted with actual human annotators to confirm robustness to realistic preference noise and cognitive biases absent from synthetic oracles?

---

**Principle 4: Computational Scalability and Plasticity Preservation in Iterative Ensemble-Based Offline PbRL**

**Definition:**  
Iterative offline PbRL algorithms that alternate between reward learning, value estimation, and query selection impose significant computational burdens and risk neural network plasticity or capacity loss during repeated re-initialization or fine-tuning. This principle evaluates whether the proposed method acknowledges and mitigates these practical constraints, particularly when relying on ensemble models for uncertainty quantification. The assessment focuses on whether the authors provide empirical evidence that lightweight uncertainty approximations can substitute for full deep ensembles without catastrophic performance degradation, and whether computational cost analyses demonstrate sub-linear or marginal scaling with ensemble size. Furthermore, the work should address known pathologies of iterative deep RL training—such as plasticity loss or capacity saturation—either through architectural choices or explicit discussion of limitations. The principle also requires that the sensitivity of the algorithm to the fidelity of uncertainty estimates is theoretically or empirically characterized.

**Core Evaluation Criteria:**
- **Lightweight Uncertainty Alternatives**: Does the work empirically validate that computationally efficient alternatives to full ensembles (e.g., last-layer ensembles, dropout) produce uncertainty estimates of sufficient fidelity for query selection and variance-based scheduling?
- **Computational Cost Transparency**: Are training time and resource scaling reported as a function of ensemble size and iterative rounds, with evidence of efficiency gains from shared-backbone architectures?
- **Plasticity and Capacity Awareness**: Does the paper discuss or empirically monitor plasticity loss and capacity saturation risks arising from iterative retraining of reward and value networks on sequentially arriving preference data?
- **Architectural Flexibility**: Is the core algorithm framed as compatible with varying uncertainty estimators, and is sensitivity to the fidelity of the uncertainty measure theoretically or empirically characterized?

---

**Principle 5: Theoretical-Practical Alignment of Exploration-Pessimism Tradeoffs in Offline PbRL**

**Definition:**  
Offline PbRL methods often present a theoretical algorithm that is simplified or modified to yield a practical implementation, creating a gap that must be rigorously justified. This principle examines whether the theoretical analysis meaningfully informs the practical algorithm's design, and whether deviations such as threshold-based discount scheduling or pure offline query selection are explicitly acknowledged and motivated. The evaluation requires that complexity measures used in bounds are connected to the actual functional approximators used in experiments, and that finite-sample guarantees reflect realistic offline data coverage coefficients rather than assuming uniform exploration. Without this alignment, theoretical contributions risk being disjoint from empirical claims, undermining the credibility of both. The principle therefore insists that every major simplification from theory to practice is accompanied by either a formal justification or an empirical validation that the simplification preserves the theoretical behavior.

**Core Evaluation Criteria:**
- **Gap Articulation**: Are differences between the theoretical algorithm and practical implementation (e.g., idealized confidence sets vs. ensemble heuristics, online vs. offline query trajectories) clearly enumerated and justified?
- **Complexity Measure Relevance**: Is the theoretical complexity measure (e.g., Eluder dimension of ΔR or V*) connected to the neural function classes used in the empirical algorithm, and are finite-class assumptions extended to infinite classes via covering numbers where appropriate?
- **Offline Data Coverage Integration**: Do the theoretical bounds incorporate offline-specific quantities such as Bellman shift coefficients or coverage coefficients, rather than assuming uniform data coverage?
- **Pessimism Mechanism Theoretical Grounding**: Is the practical pessimism mechanism (e.g., discount scheduling) supported by offline RL theory (e.g., smaller discount factors yielding pessimistic value estimates), or is it presented as an unmotivated heuristic?