**Principle 1: Structural Fairness and Workflow Equivalence in Single-Agent versus Multi-Agent RL Evaluations**

**Definition:**  
Evaluating multi-agent RL algorithms demands rigorous structural alignment between single-agent and multi-agent baselines to prevent conflating algorithmic improvements with inherent workflow advantages. Reviewers heavily scrutinized whether reported performance gaps arose from closed-loop environmental feedback, multi-turn interaction privileges, or uneven per-decision inference budgets rather than from the proposed AT-GRPO optimizer itself. A principled comparison must therefore hold constant the observation space, reward signals, termination conditions, and sampling budgets across paradigms. Furthermore, researchers must conduct controlled ablations—such as multi-turn single-agent variants—to isolate the marginal contribution of multi-agent coordination from mere turn repetition. Without such alignment, headline gains risk being artifacts of task reformulation rather than genuine advances in credit assignment or policy optimization.

**Core Evaluation Criteria:**
- **Environmental Parity:** Are observation spaces, reward functions, and termination criteria strictly identical across single-agent and multi-agent settings?
- **Inference Budget Alignment:** Is the sampling budget (e.g., K candidates per decision) held constant so that gains are not attributable to increased search depth for the proposed method alone?
- **Multi-Turn Controls:** Are multi-turn single-agent ablations performed to demonstrate that extra interaction turns alone do not close the performance gap?
- **Workflow Transparency:** Is the experimental setup described with sufficient granularity to distinguish open-loop generation from closed-loop execution with environmental feedback?

---

**Principle 2: Disentanglement of Algorithmic Contributions from Dense Reward Engineering and Oracle Signals**

**Definition:**  
In LLM agent RL, elaborate dense reward shaping can provide oracle-level guidance that simplifies exploration and obscures the true efficacy of the underlying algorithm. Multiple reviewers challenged whether near-perfect planning accuracy stemmed from AT-GRPO or from shortest-path and deadlock heuristics embedded in the reward function. High-quality research must therefore systematically ablate dense rewards against sparse, outcome-only signals to verify that the algorithm remains effective when deprived of intermediate oracles. This disentanglement ensures that reported gains reflect genuine learning of collaborative reasoning rather than optimization of hand-crafted reward landscapes. Ultimately, the burden of proof lies on the authors to show that their method solves tasks even under minimal, verifiable feedback.

**Core Evaluation Criteria:**
- **Sparse Reward Robustness:** Does the method maintain strong performance when evaluated with strictly outcome-based binary rewards without intermediate heuristics?
- **Reward Ablation Depth:** Are dense rewards ablated across all domains, and do baselines share identical reward regimes to prevent handicapping?
- **Oracle Avoidance:** Is there evidence that agents learn generalizable policies rather than memorizing pre-computed heuristics (e.g., via strict train/test environment separation)?
- **Attribution Clarity:** Does the analysis explicitly separate the contribution of reward design from algorithmic design through controlled comparisons?

---

**Principle 3: Structural Validity of Group-wise Advantage Estimation under Heterogeneous Multi-Agent Prompts**

**Definition:**  
Group-relative policy optimization methods such as GRPO fundamentally assume that all candidates within a comparison group share an identical prompt. In multi-agent systems, this assumption collapses because prompts diverge across roles and accumulate divergent turn histories, rendering naive parallel sampling structurally invalid. The review process demanded rigorous justification for alternative grouping strategies—specifically agent-and-turn-wise grouping within a tree-structured rollout—to restore fair advantage estimation. Research in this area must demonstrate that its grouping mechanism ensures prompt homogeneity within each group and empirically validate that naive grouping destabilizes training. Without this structural validity, advantage estimates become biased and the variance-reduction benefit of group-relative methods is lost.

**Core Evaluation Criteria:**
- **Group Homogeneity Guarantee:** Does the proposed method ensure that all K candidates in a group share identical role, turn index, and interaction history?
- **Naive Baseline Failure:** Is there an explicit ablation showing that standard GRPO grouping applied to MAS (e.g., parallel sampling) degrades performance due to heterogeneous state aggregation?
- **Sampling Justification:** Is the tree-structured or alternative sampling mechanism justified against simpler designs through both empirical and theoretical arguments?
- **Empirical Validation:** Does the evidence show that heterogeneous grouping causes incorrect baseline aggregation and optimization instability?

---

**Principle 4: Theoretical Rigor for Bias-Variance Trade-offs and Optimality in Tree-Structured Multi-Agent Rollouts**

**Definition:**  
Introducing tree-structured sampling and greedy per-turn selection into multi-agent RL creates novel estimator properties that demand theoretical scrutiny. The meta-review explicitly noted the absence of formal bias-variance analysis for the tree sampling procedure and criticized reliance on a strong monotonicity assumption to justify greedy transitions. While empirical comparisons against MAS+GRPO can demonstrate efficacy, they cannot substitute for analytical characterization of estimator behavior as turn horizon T and branching factor K scale. Strong contributions should provide formal proofs or, at minimum, rigorous sensitivity studies under stated assumptions, while transparently acknowledging theoretical limitations that bound the method's generality. This rigor distinguishes heuristic adaptations from principled algorithmic advances.

**Core Evaluation Criteria:**
- **Estimator Analysis:** Is there formal or high-confidence empirical analysis of how bias and variance in advantage estimates scale with turn horizon T and branching factor K?
- **Optimality Grounding:** Are greedy selection rules formally linked to Bellman optimality under explicitly stated assumptions (e.g., monotonicity of verifiable rewards with respect to optimal Q-values)?
- **Sensitivity Studies:** Does the work systematically vary T and K to study training stability and performance boundaries?
- **Assumption Transparency:** Are limiting assumptions clearly disclosed, and are their practical implications for non-monotonic or stochastic rewards discussed?

---

**Principle 5: Empirical and Analytical Scalability Assessment for Multi-Policy On-Policy Training Beyond Two Agents**

**Definition:**  
A complete MAS RL contribution must address scalability along both algorithmic and systems dimensions as the number of agents N and turn horizon T grow. Reviewers questioned whether AT-GRPO would generalize beyond dyadic settings, noting that initial experiments were limited to N=2 with modest empirical scaling evidence. The work must provide formal complexity analysis (time, memory, communication) for maintaining agent-and-turn-wise groupings and executing concurrent on-policy updates for multiple distinct policies. Furthermore, empirical validation should extend beyond minimal two-agent configurations to demonstrate that coordination gains persist without hitting computational or optimization bottlenecks. Systems-level innovations must also be accompanied by reproducible engineering details regarding asynchronous batching and trajectory routing to enable independent verification.

**Core Evaluation Criteria:**
- **Complexity Analysis:** Are computational and memory overheads formally characterized with respect to N, T, and K?
- **Multi-Agent Scaling:** Is performance validated empirically with agent counts exceeding two (e.g., N ≥ 3 or N ≥ 5) and diverse interaction topologies?
- **System Correctness:** Does the training system preserve on-policy data routing and gradient isolation when concurrently updating shared or specialized policies?
- **Resource Transparency:** Are wall-clock latency, throughput, and resource implications reported for realistic hardware configurations?