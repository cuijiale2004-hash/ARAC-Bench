**Principle 1: Theoretical Justification and Mechanistic Interpretability of Intrinsic Exploration Bonuses in Verifiable-Reward LLM Training**

**Definition:**  
Research proposing intrinsic exploration bonuses for LLM reasoning must provide rigorous theoretical justification that connects the proposed bonus to established exploration principles while accounting for the unique characteristics of language generation spaces. Unlike continuous control or game-playing environments, LLM reasoning involves discrete, high-dimensional token sequences where traditional state-visitation counts are ill-defined, necessitating theoretical analysis that justifies proxy signals—such as perplexity or value variance—as valid novelty measures. The principle demands that authors not only prove convergence properties or regret bounds where possible, but also mechanistically explain why the bonus penalizes specific failure modes, such as overconfident errors or repetitive reasoning patterns, rather than offering opaque performance gains. This is crucial because unprincipled bonuses can inadvertently incentivize linguistic noise or semantically vacuous diversity that does not improve reasoning coverage. A strong theoretical foundation enables reviewers to distinguish between ad-hoc heuristic injection and principled exploration augmentation that is transferable across RLVR algorithms.

**Core Evaluation Criteria:**
- **Connection to Classical Theory**: Does the work formally link the proposed bonus (e.g., perplexity, value variance) to established RL exploration frameworks (e.g., count-based UCB, prediction-error curiosity) under specified assumptions such as linear MDPs?
- **Mechanistic Clarity**: Does the analysis explicitly identify which failure modes the bonus targets (e.g., calibration collapse, entropy collapse, mode collapse) and provide causal reasoning rather than purely correlational observations?
- **Validity of Assumptions**: Are the theoretical assumptions reasonable for the LLM setting, or are they so restrictive that they undermine practical relevance for large-scale autoregressive models?
- **Distinction from Entropy Regularization**: Does the theory clearly differentiate the proposed bonus from naive entropy bonuses or KL regularization, demonstrating additive value beyond policy stochasticity?

---

**Principle 2: Adaptive Exploration-Exploitation Scheduling and Reward Hacking Prevention in High-Dimensional Discrete Action Spaces**

**Definition:**  
Effective exploration in LLM reasoning requires careful scheduling mechanisms that modulate bonus intensity over the course of training, given that unconstrained intrinsic rewards in discrete language spaces readily incentivize high-perplexity gibberish or syntactically degraded outputs. This principle evaluates whether the proposed method includes disciplined annealing strategies—such as staircase decay, linear decay, or performance-gated reduction—that deliberately transition the policy from broad state-action coverage to refined exploitation of high-reward reasoning paths. It further scrutinizes the robustness of clipping mechanisms and scaling factors that constrain the intrinsic signal to a fraction of the verifiable task reward, thereby preventing reward hacking where the optimizer prioritizes curiosity over correctness. The evaluation extends to whether the scheduling is validated across multiple decay schemes and shown to be necessary rather than incidental. In LLM RLVR, where final answers are verifiable but intermediate reasoning is not, such scheduling is essential to ensure exploration translates into discoverable correct solutions rather than merely diversifying incorrect paths.

**Core Evaluation Criteria:**
- **Annealing Necessity and Design**: Are multiple bonus decay schedules compared, and is there empirical evidence that decay is necessary for stable convergence (e.g., no-decay baselines fail or fluctuate)?
- **Anti-Hacking Constraints**: Does the method employ clipping ratios or scaling factors that rigorously bound the exploration bonus relative to the task reward, with sensitivity analysis showing robustness to these hyperparameters?
- **Phase Transition Analysis**: Does the work analyze training dynamics to demonstrate an initial exploration phase followed by stable exploitation, evidenced by learning curves, accuracy trajectories, or entropy profiles?
- **Quality-Preserving Exploration**: Is there evidence that exploration improves or maintains response coherence and correctness rates, rather than increasing output diversity through degraded or nonsensical language?

---

**Principle 3: Empirical Verification of Anti-Collapse Dynamics and State-Action Coverage in LLM Reasoning Policies**

**Definition:**  
Claims of improved exploration must be supported by direct empirical evidence of training dynamics, specifically demonstrating mitigation of entropy collapse, premature convergence, or policy oscillation during RLVR training. This principle requires that authors move beyond aggregate final-task accuracy and provide diagnostic analyses—such as token-level entropy trajectories, response uniqueness metrics, or value-head disagreement distributions over time—that reveal how the policy's behavior space evolves. In the context of LLM reasoning, coverage should be assessed not merely by surface-level n-gram diversity but by the model's ability to discover distinct valid solution paths and recover from local optima. The principle also demands cross-dataset validation, demonstrating that exploration bonuses respond appropriately to data novelty (e.g., higher disagreement on out-of-domain prompts). Without such diagnostics, performance gains could be attributed to generic optimization stabilization rather than genuine expansion of the reasoning state space.

**Core Evaluation Criteria:**
- **Entropy and Diversity Trajectories**: Are policy entropy and response diversity tracked throughout training, showing that the method avoids entropy collapse compared to standard RLVR baselines?
- **Novelty Sensitivity**: Is there evidence that curiosity signals correlate with data novelty (e.g., higher bonuses or critic disagreement for out-of-domain or rarely seen prompt types)?
- **Failure-Mode Diagnostics**: Does the work analyze failure cases to show reduced behavioral looping, reduced spurious mode collapse, or recovery from local reward optima?
- **Disentanglement of Exploration from Optimization**: Are the gains shown to stem from expanded coverage rather than merely reduced variance in policy gradients or improved baseline estimation?

---

**Principle 4: Calibration Preservation and Uncertainty-Aware Confidence Alignment in RL-Finetuned Language Models**

**Definition:**  
A critical but often overlooked aspect of RLVR training is the calibration of model confidence: policies should maintain high confidence when correct and appropriate uncertainty when incorrect. This principle evaluates whether exploration methods actively preserve or improve the alignment between a model's internal confidence (e.g., perplexity, probability margins) and its actual correctness, preventing the "calibration collapse" phenomenon where RL training progressively decouples confidence from accuracy. Proposed bonuses—particularly those derived from actor perplexity—must be assessed for their dual role in promoting exploration while serving as a calibration signal that penalizes overconfident errors. The evaluation should require stratified analysis of confidence metrics across correct and incorrect responses over training time, demonstrating sustained separation between the two classes. This is especially important in reasoning LLMs because well-calibrated confidence enables downstream inference-time strategies such as self-consistency voting, rejection sampling, and best-of-N selection.

**Core Evaluation Criteria:**
- **Calibration Monitoring**: Is the relationship between model confidence (e.g., perplexity, log-probability) and correctness tracked across training steps for both proposed and baseline methods?
- **Overconfident Error Penalization**: Does the method demonstrably reduce the prevalence of high-confidence incorrect predictions relative to baselines?
- **Correct-Response Confidence Dynamics**: Among correct answers, does the method maintain or increase confidence without collapsing diversity into a single high-confidence mode?
- **Utility for Inference-Time Strategies**: Does the improved calibration translate into better performance for inference-time selection methods (e.g., Pass@k gains, self-certainty best-of-N)?

---

**Principle 5: Architectural Minimality and Computational Scalability of Auxiliary Exploration Modules in LLM Actor-Critic Training**

**Definition:**  
Exploration methods for LLM reasoning must respect the substantial computational constraints of training large models, avoiding auxiliary modules that introduce prohibitive memory or latency overheads relative to the base RLVR pipeline. This principle evaluates whether proposed architectural modifications—such as multi-head critics, auxiliary prediction networks, or embedding-based hash tables—are justified by performance gains that outweigh their resource costs. It favors designs that reuse existing model components (e.g., leveraging the actor's own perplexity or sharing backbone parameters across critic heads) over those requiring separate forward passes, external state representations, or additional large networks. The evaluation should include concrete measurements of memory footprint, throughput, and wall-clock time, alongside ablations demonstrating that the chosen minimal design is sufficient and that increasing complexity yields diminishing returns. In the LLM context, where training already demands significant resources, architectural efficiency determines whether an exploration method is a practical advancement or merely a theoretical curiosity infeasible at scale.

**Core Evaluation Criteria:**
- **Overhead Quantification**: Are memory usage and runtime explicitly reported and shown to be negligible or modest relative to the base training framework (e.g., single-digit megabyte increases or sub-percent runtime differences)?
- **Design Parsimony**: Are auxiliary components kept minimal (e.g., shared backbones, no extra inference networks), and are ablations provided to justify specific architectural choices (e.g., optimal number of heads)?
- **Cost-Benefit Tradeoff**: Does the performance improvement justify any additional computational cost, and is there analysis showing diminishing returns with increased module complexity?
- **Framework Compatibility**: Is the method implementable with minor modifications to standard LLM RL frameworks (e.g., VERL, TRL), without requiring fundamentally different training infrastructures or data pipelines?