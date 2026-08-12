Principle 1: Causal Verification Through Targeted Intervention on Internal Activation Patterns During RL Fine-Tuning

**Definition:**  
Research investigating how RL fine-tuning reshapes LLM internal circuitry must move beyond correlational observations by establishing rigorous causal links between circuit-level changes and downstream capability improvements. It is insufficient to show that activation intensity or diversity merely co-varies with performance gains; reviewers expect targeted interventions—such as manipulating sampling temperatures, clamping activation magnitudes, or ablating specific edges—that directly perturb the hypothesized mechanisms and measure consequent performance shifts. This principle demands that authors disentangle whether observed circuit properties are functional drivers of RL effectiveness or epiphenomenal byproducts of training dynamics. Strong submissions should include temporal analyses demonstrating that internal changes precede or tightly coincide with capability improvements, alongside appropriate controls for confounders like response length or compute budget. Ultimately, causal verification elevates the work from descriptive phenomenology to mechanistic science.

**Core Evaluation Criteria:**
- **Intervention Design**: Does the work include experiments that actively manipulate training conditions or internal states to test causal hypotheses about circuit changes?
- **Temporal Causal Alignment**: Are internal metric shifts shown to precede or coincide with performance improvements, rather than merely co-occurring post-hoc?
- **Confound Control**: Are alternative explanations ruled out, such as changes being artifacts of output length, vocabulary distribution shifts, or increased compute?
- **Directional Causality Evidence**: Does suppressing the observed internal pattern degrade performance, or enhancing it improve performance, relative to appropriate baselines?

---

Principle 2: Theoretical Grounding Connecting RL Optimization Dynamics to Circuit-Level Reconfiguration Mechanisms

**Definition:**  
Papers in this subfield must provide explanatory frameworks that connect observed circuit-level transformations to the mathematical and algorithmic foundations of RL objectives, rather than stopping at empirical description. Authors should articulate how specific components—such as on-policy sampling distributions, policy gradient variance, advantage estimation, or reward signal propagation—mechanistically drive changes in edge activation patterns. This requires situating empirical findings within a unified view of post-training gradients, demonstrating how differences between SFT, online RL, and preference-based methods arise from their distinct data sources and gradient coefficients. The absence of such theoretical scaffolding leaves findings vulnerable to the critique of being "black-box" observations lacking mechanistic grounding. Reviewers expect testable hypotheses derived from RL theory that predict when and why dormant pathways activate or why entropy increases in edge-weight distributions.

**Core Evaluation Criteria:**
- **RL Theory Integration**: Does the work connect observed circuit metrics to formal properties of RL objectives (e.g., policy gradient variance, exploration bonuses)?
- **Unified Gradient Perspective**: Is there a framework explaining how SFT, online RL, and preference-based methods differ mechanistically in their circuit impact?
- **Testable Hypotheses**: Are the explanations formulated as falsifiable predictions about when specific circuit patterns should emerge or disappear?
- **Component Ablation**: Are specific RL components (reward model, KL divergence, sampling temperature) isolated to test their contribution to circuit reconfiguration?

---

Principle 3: Cross-Scale and Cross-Family Robustness of Circuit Dynamics Under RL Post-Training

**Definition:**  
Claims regarding universal RL-induced circuit modifications must be validated across multiple model scales and architectural families to rule out scale-dependent or architecture-specific artifacts. A study limited to a single parameter scale (e.g., 7B) risks conflating generic RL effects with emergent behaviors particular to that capacity regime, while narrow architectural focus may reflect inductive biases rather than fundamental mechanisms. Reviewers expect explicit discussion of computational constraints limiting scale coverage, alongside evidence from smaller or larger models when feasible, or transparent acknowledgment of this limitation as a boundary condition. Cross-family validation—spanning diverse attention mechanisms, normalization schemes, or mixture-of-experts designs—is essential to establish whether observed activation dynamics generalize or are idiosyncratic to specific model lineages. Robust submissions should disaggregate findings by scale and architecture, identifying invariant patterns versus family-specific anomalies.

**Core Evaluation Criteria:**
- **Multi-Scale Validation**: Are findings replicated across at least two significantly different parameter scales (e.g., 3B and 7B)?
- **Architectural Diversity**: Does the study cover multiple model families with distinct design choices, or clearly discuss architectural limitations?
- **Scale-Effect Disaggregation**: Does the work explicitly examine whether circuit effects strengthen, weaken, or qualitatively change with model scale?
- **Resource Transparency**: Are computational constraints acknowledged and their impact on scale coverage honestly discussed?

---

Principle 4: Domain Generalization and Task-Agnostic Consistency of RL-Induced Internal Pathway Rewiring

**Definition:**  
Findings derived from circuit analysis must be scrutinized for domain specificity, particularly when experiments are confined to structured reasoning tasks like mathematical problem-solving with verifiable rewards. Mathematical reasoning offers clean reward signals and convergent solution paths, but these properties may artificially simplify circuit dynamics compared to open-ended domains such as creative writing, dialogue, or code generation where valid outputs are highly heterogeneous. Reviewers expect authors to either empirically validate findings across diverse task types or rigorously justify domain restriction by analyzing how reward structure (sparse/verifiable vs. dense/subjective) and output space topology influence interpretability. Without such checks, claims about "general RL effects" risk overgeneralization from a narrow task distribution. High-quality work explicitly scopes its conclusions, discusses domain-transfer hypotheses, and acknowledges whether observed pathway diversification stems from task-specific reasoning demands or universal RL optimization pressures.

**Core Evaluation Criteria:**
- **Multi-Domain Testing or Scoped Claims**: Are claims validated beyond mathematical reasoning, or are conclusions explicitly scoped to structured reasoning domains?
- **Reward Structure Analysis**: Does the work analyze how verifiable vs. subjective rewards affect the interpretability and structure of internal circuits?
- **Output-Space Control**: Are analyses controlled for domain-induced output heterogeneity that could confound edge attribution metrics?
- **Transfer Hypotheses**: Does the paper articulate theoretically grounded expectations for how findings should or should not transfer to open-ended generation tasks?

---

Principle 5: Comprehensive Algorithmic Coverage Distinguishing Online RL, Offline Preference Optimization, and Reward Signal Sources

**Definition:**  
To attribute circuit-level effects correctly to training methodology, studies must provide comprehensive coverage of the algorithmic design space, distinguishing between online policy-gradient methods, offline preference optimization, and hybrid approaches. Conflating algorithmic categories—such as treating RLVR as an algorithm rather than a reward source, or failing to contrast DPO with PPO/GRPO—undermines the ability to isolate whether findings are driven by on-policy exploration dynamics, reward signal provenance, or preference-based constraints. Reviewers expect systematic comparisons across orthogonal dimensions: optimization algorithm (online vs. offline), reward type (learned model vs. verifiable rule), and sampling strategy (on-policy evolution vs. static dataset). This methodological cartography is essential for determining whether observed increases in activation intensity and diversity are unique to online RL's stochastic support expansion or shared by all post-training paradigms. Incomplete coverage leaves critical confounds unaddressed and weakens the paper's capacity to make principled distinctions between mechanistic signatures of different training regimes.

**Core Evaluation Criteria:**
- **Algorithmic Taxonomy Clarity**: Does the work correctly distinguish between optimization algorithms (PPO, GRPO, DPO), reward sources (learned RM, rule-based), and training modes (online/offline)?
- **Complete Methodological Coverage**: Are all relevant categories represented, or is missing coverage justified and acknowledged?
- **Orthogonal Comparison Design**: Are comparisons structured to isolate the effects of algorithmic category from reward signal type?
- **Mechanistic Distinction**: Does the analysis successfully attribute specific circuit signatures (e.g., intensity vs. diversity) to distinct algorithmic components?