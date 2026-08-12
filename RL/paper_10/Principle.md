Principle 1: Topology Scalability and Structural Generality Beyond Dyadic Interaction Patterns

**Definition:**  
The work must rigorously demonstrate that the proposed collaboration optimization framework extends beyond simple two-agent dyads to genuinely complex multi-agent settings, as claims about "multi-agent" collaboration carry specific structural implications that dyadic interactions cannot substitute. Reviewers in this subfield expect empirical validation on fixed topologies involving three or more agents—such as multi-turn debate, dynamic role assignment, or hierarchical reasoning chains—to ensure that coordination policies do not collapse under increased communication paths and credit assignment ambiguity. Furthermore, the method should ideally be topology-agnostic, meaning it can be plugged into diverse pre-existing interaction protocols without requiring fundamental redesign of the underlying communication structure. This criterion is essential because many dyadic helper-model setups can be reframed as "multi-agent" systems without actually addressing the combinatorial coordination challenges inherent in larger agent collectives. Authors must therefore avoid overstating generality when the empirical scope is limited to pairwise Actor-Collaborator configurations. A strong submission in this area clearly separates the contribution of interaction-structure optimization from the raw expressiveness of the chosen topology itself.

**Core Evaluation Criteria:**
- **Dyadic vs. Multi-Agent Coverage**: Does the evaluation include topologies with more than two interacting agents, or is the "multi-agent" framing unsupported by experiments beyond pairwise setups?
- **Topology-Agnostic Integration**: Can the method be applied to diverse fixed topologies (e.g., debate, role assignment, critic-actor loops) without altering their native communication protocols?
- **Structural Claim Justification**: Does the paper explicitly acknowledge the limitations of its evaluated topologies and avoid conflating helper-model dyads with general multi-agent collaboration?

---

Principle 2: Cross-Actor Transferability and Black-Box API Deployment Compatibility

**Definition:**  
A central value proposition of training lightweight Collaborator agents is real-world deployability, which hinges on whether the learned collaboration policy captures generalizable meta-strategies rather than overfitting to the idiosyncrasies of a specific frozen Actor. Reviewers scrutinize whether the Collaborator trained with one Actor model (e.g., a 3B parameter LLM) retains effectiveness when paired with a different Actor (e.g., an 8B model or a closed-weight API endpoint), because actor-specific memorization would require costly retraining with every model upgrade or swap. The principle also encompasses the practical scenario where the Actor is a powerful black-box API model whose weights are inaccessible, rendering direct fine-tuning impossible but collaborative enhancement via a local trainable module highly desirable. Consequently, the research must empirically validate robustness under extreme capability asymmetry between Collaborator and Actor. Without such evidence, claims regarding scalability and deployment efficiency remain theoretically speculative and practically unsubstantiated.

**Core Evaluation Criteria:**
- **Cross-Model Generalization**: Are there explicit transfer experiments where a Collaborator trained with one Actor is evaluated with a different Actor family or scale?
- **Black-Box Actor Compatibility**: Does the method support enhancing API-based or closed-source Actors without requiring gradient access, parameter modification, or white-box assumptions?
- **Capability-Gap Robustness**: Is performance maintained or analyzed when the Collaborator is substantially smaller or less capable than the Actor (e.g., 3B local model paired with frontier-scale API model)?

---

Principle 3: Controlled Experimental Rigor and Fair Baseline Alignment in Multi-Agent Evaluation

**Definition:**  
LLM-based multi-agent systems are notoriously sensitive to prompt phrasing, sampling temperature, model assignment, and inference budgets, making meticulous experimental control a non-negotiable standard for evaluation in this subfield. Reviewers expect transparent justification for any variation in model roles across tables, rigorous hyperparameter parity for re-implemented baselines, and statistical validation through multi-seed runs with reported variance—especially when benchmark slices are small and point estimates may reflect noise. Additionally, efficiency claims such as "superior scalability" or "lower inference cost" must be grounded in fair budget comparisons, whether through fixed-token, wall-clock, or complete-solution accounting protocols that do not artificially handicap iterative baselines. Inadequate control in any of these dimensions undermines the causal attribution of performance gains to the proposed algorithm rather than to incidental experimental choices. Ultimately, the experimental design must leave no ambiguity about whether observed differences stem from methodological innovation or from unequal resource allocation and implementation fidelity.

**Core Evaluation Criteria:**
- **Experimental Consistency and Clarity**: Are Actor/Collaborator model assignments held constant within comparisons, with explicit rationale provided for any cross-table variations?
- **Statistical Robustness**: Are results accompanied by confidence intervals, standard deviations, or significance tests across multiple random seeds?
- **Baseline Implementation Parity**: Do re-implemented baselines share identical sampling strategies, stopping criteria, prompt templates, and computational budgets?
- **Budget-Aware Efficiency Claims**: Are cost and scalability claims supported by equal-budget or rigorous token-accounting comparisons rather than extrapolated cost estimates?

---

Principle 4: Mechanistic Analysis of Collaboration Efficacy and Failure Mode Characterization

**Definition:**  
Aggregate accuracy improvements are insufficient to validate a collaboration optimization framework; reviewers demand mechanistic insight into *why* and *when* collaborative intervention succeeds or fails. This requires explicit analysis of performance degradation cases—such as untrained or poorly optimized hints misleading the Actor, or critique-based topologies yielding diminishing returns—to disentangle the marginal value of policy optimization from the intrinsic suitability of the interaction topology. The research should also diagnose training dynamics, including sample-scale sensitivity, overfitting, and early-stopping necessity, to demonstrate that the Collaborator learns robust meta-collaboration skills rather than spurious task-specific heuristics. Qualitative case studies illustrating the nature of learned hints or critiques are highly valued, as they distinguish strategic guidance from solution memorization or shallow prompt templating. A submission that merely reports final metrics without interrogating failure modes, training instabilities, or the interpretable structure of collaboration signals is considered incomplete in this evaluation dimension.

**Core Evaluation Criteria:**
- **Failure Case and Drop Analysis**: Are instances of degraded performance analyzed to identify causal factors (e.g., task difficulty, topology mismatch, negative transfer)?
- **Training Dynamic Robustness**: Are phenomena such as overfitting, performance oscillation, or sample-scale sensitivity investigated, and are mitigation strategies (e.g., early stopping, regularization) proposed?
- **Topology vs. Optimization Disentanglement**: Does the work isolate and compare the raw impact of the collaboration topology against the incremental gain from RL-based optimization?
- **Qualitative Interpretability**: Are concrete examples provided to reveal what the Collaborator learns, enabling reviewers to assess whether the policy captures generalizable guidance or superficial patterns?

---

Principle 5: Distinct Methodological Positioning Against Self-Refinement and Prompt Optimization Paradigms

**Definition:**  
The landscape of LLM collaboration research is densely populated with adjacent paradigms—including iterative self-refinement loops, standalone critic training, automated prompt optimization, and architectural topology search—making precise differentiation a critical evaluation criterion. Reviewers expect authors to articulate a clear, non-obvious design principle, such as functional decoupling of execution and collaboration, system-level reward optimization under fixed structures, or reinforcement learning of interaction policies, that moves beyond standard helper-model or reflection pipelines. This positioning must be substantiated through comprehensive related-work discussions that directly contrast the proposed framework with the closest existing methods both conceptually and, where possible, empirically. Vague framing or omission of highly similar prior work (e.g., conflating a trained critic with existing reflection agents) is treated as a serious weakness because it obscures the true incremental contribution and misleads readers about the state of progress in the field. A high-quality submission in this area defines its unique niche unambiguously and honestly acknowledges its relationship to precursor techniques.

**Core Evaluation Criteria:**
- **Novelty Articulation**: Is the core contribution framed as a generalizable collaboration optimization framework with a distinct theoretical or algorithmic basis, rather than an incremental variant of helper-model prompting?
- **Related-Work Differentiation**: Are closely related methods (e.g., self-reflection, critic training, prompt search, topology optimization) thoroughly cited and explicitly contrasted in terms of objective scope and methodology?
- **Framing Accuracy**: Does the paper avoid overstating the complexity of its "multi-agent" setup when the empirical contribution is primarily a two-model collaborative dyad, and does it honestly acknowledge limitations relative to prior art?