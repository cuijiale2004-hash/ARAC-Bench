**Principle 1: Rigorous Inference Cost Normalization and Token Budget Fairness in Multi-Agent Test-Time Orchestration Comparisons**

**Definition:**  
In test-time multi-agent LLM orchestration, raw task accuracy is an insufficient evaluation metric unless accompanied by rigorous cost normalization. Reviewers demand that proposed coordination mechanisms be evaluated against single-model, parallel, and alternative multi-agent baselines under strictly matched inference budgets, including output token limits, turn counts, and context window sizes. Without such normalization, apparent performance gains may simply reflect increased computational expenditure rather than genuine algorithmic advantage. Furthermore, token efficiency and cost-effectiveness must be explicitly quantified, especially when comparing multi-turn protocols against single-turn or parallel test-time scaling strategies. This principle ensures that the field prioritizes sustainable, deployable coordination methods over those that achieve marginal accuracy improvements through prohibitively expensive inference. Authors must report actual output token counts and API-call costs to enable a fair assessment of practical utility.

**Core Evaluation Criteria:**
- **Actual token usage and computational cost reporting** for all compared methods, including per-task output token averages and total API call counts.
- **Fair budget matching** between multi-turn coordination systems and single-model baselines via equivalent context windows, self-reflection turns, or parallel sampling widths.
- **Inclusion of parallel test-time scaling baselines** (e.g., majority voting, best-of-N) with explicit scoping of their applicability limitations across open-ended versus multiple-choice tasks.
- **Quantified cost-effectiveness metrics**, such as accuracy gain per thousand tokens or relative error reduction normalized by inference budget.

---

**Principle 2: Demonstrated Necessity of Learned Coordination Over Prompted Frontier Models and Static Task-to-Agent Heuristics**

**Definition:**  
A core evaluative standard for learned LLM coordinators is proving that the learned component is strictly necessary compared to plausible non-learned alternatives. Simply outperforming individual agents is inadequate; authors must demonstrate that a trained lightweight coordinator surpasses a prompted frontier model given the same coordination privileges, and that adaptive agent selection outperforms static task-to-model heuristics. This addresses the fundamental question of whether coordination is a genuinely learnable function that benefits from compact representation learning, or merely a prompting problem solvable by larger models. Isolating the marginal contribution of learned parameters requires ablations that disable agent selection while preserving other protocol elements, such as role assignment. The principle ensures that architectural novelty translates into irreducible empirical advantages rather than convenient scaffolding around existing capabilities.

**Core Evaluation Criteria:**
- **Direct comparison against prompted frontier LLMs** acting as coordinators with identical action spaces, agent pools, and turn budgets.
- **Ablations isolating adaptive agent selection** from role assignment (e.g., fixing the system to a single agent while retaining learned roles versus the full adaptive system).
- **Comparison against static heuristic routing and random selection** to demonstrate that learned adaptivity outperforms uninformed or rule-based delegation.
- **Evidence that gains derive from learned hidden-state representations** rather than from the hand-designed interaction protocol alone.

---

**Principle 3: Architectural Extensibility and Progressive Role Ablation Rigor in Fixed-Taxonomy Multi-Turn Delegation**

**Definition:**  
For role-based multi-turn delegation frameworks, reviewers scrutinize whether the predefined role taxonomy represents robust, abstract workflow stages or a brittle, hard-coded constraint. A critical concern is architectural flexibility: adding new capabilities such as tool use or code execution should not require modifying the coordinator's output dimension or retraining the entire system from scratch. Consequently, authors must provide progressive ablations that remove, merge, or simplify roles to empirically validate the necessity of the full protocol. The distinction between roles as meta-categories of reasoning versus fixed functional silos must be clearly articulated, with evidence that new skills can be introduced through agent-pool expansion and prompting alone. This principle guards against over-engineered coordination schemas that sacrifice extensibility for marginal short-term gains.

**Core Evaluation Criteria:**
- **Progressive role ablations** including two-role variants, no-role variants, and merged configurations (e.g., thinker-worker merged) to validate the full taxonomy.
- **Explicit conceptual framing of roles as abstract workflow stages** rather than immutable capability silos, with zero-shot or few-shot evidence for accommodating new functionalities.
- **Analysis of functional contribution per role**, linking role removal to degradation in turn efficiency, solution quality, or reasoning depth.
- **Examination of output architecture constraints** to verify whether skill expansion is possible without retraining or changing the coordinator's logit dimensionality.

---

**Principle 4: Budget-Constrained Optimizer Selection and Objective Landscape Characterization for High-Dimensional Tiny Coordinator Heads**

**Definition:**  
When optimizing extremely lightweight coordination heads under high-dimensional, noisy, and budget-constrained conditions, the choice of learning algorithm requires rigorous justification beyond empirical success. Reviewers expect comparisons against standard alternatives—including reinforcement learning, random search, and supervised fine-tuning with oracle labels—under matched evaluation budgets to substantiate optimizer selection. Theoretical or empirical characterization of the objective landscape, such as block-ε-separability or weak parameter coupling, is essential to explain why gradient-based or imitation methods may fail. Particular attention must be paid to the scalability of label generation, as oracle-action construction for multi-turn trajectories grows exponentially, rendering supervised approaches impractical despite their single-step competitiveness. This principle ensures that optimization claims are grounded in the unique geometry and cost structure of test-time coordination.

**Core Evaluation Criteria:**
- **Head-to-head optimizer comparison under matched atomic evaluation budgets**, including derivative-free evolutionary strategies, REINFORCE-style RL, random search, and SFT with oracle labels.
- **Empirical or theoretical characterization of objective geometry** (e.g., block-separability, parameter coupling) motivating the chosen optimizer over alternatives.
- **Scalability analysis of label-generation cost**, demonstrating the impracticality of imitation learning in multi-turn settings due to exponential oracle query growth.
- **Evidence of non-degenerate policy learning**, showing that the optimizer produces meaningful agent/role selection distributions rather than uniform or collapsed policies.

---

**Principle 5: Empirical Validation of Hidden-State Separability as Representational Prerequisite for Lightweight Logit-Based Routing**

**Definition:**  
When a tiny linear head makes routing decisions from the hidden states of a small language model, reviewers require evidence that the representation space contains sufficient structure to support such extreme parameter efficiency. The burden of proof lies on establishing a quantitative link between the separability of tasks or agents in the SLM's hidden-state manifold and the coordinator's downstream routing accuracy. This involves not only descriptive visualizations but controlled synthetic experiments that manipulate separability independently of task confounders, alongside linear probe accuracies on real hidden states. Authors must also justify architectural choices such as penultimate-token extraction by showing their representational superiority over alternatives. This principle prevents unwarranted assumptions that arbitrary hidden states are automatically adequate for lightweight coordination.

**Core Evaluation Criteria:**
- **Quantitative separability analysis of SLM hidden states** using linear probes (e.g., SVM accuracy) and dimensionality reduction to establish task-aware clustering.
- **Controlled synthetic experiments** that modulate inter-class separability while fixing dimensionality and covariance, correlating separability indices with head classification accuracy.
- **Task-level correlation** between representation-space separability and the magnitude of coordinator performance gains over baselines.
- **Ablation on representation extraction points** (e.g., last token versus penultimate token) linking representational quality directly to routing performance.