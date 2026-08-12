Principle 1: Prescriptive Theory and Explicit Optimization Bounds for Mechanistic Hyperparameter Analysis in Private Transfer Learning

**Definition:**
Research in differentially private transfer learning must advance beyond correlational or post-hoc empirical observations to develop prescriptive theoretical frameworks that directly inform hyperparameter selection. The theoretical contributions should establish tight, explicit connections between mechanistic phenomena—such as gradient distribution shifts induced by clipping—and optimization guarantees, ensuring that abstract results translate into actionable guidance for practitioners. Assumptions must be clearly stated, internally consistent, and positioned in direct proximity to the theorems they support, avoiding ambiguous fixed-point formulations or implicit boundary conditions. Furthermore, the work should demonstrate that minimizing proposed theoretical proxies, such as per-step mean squared error, directly improves concrete optimization bounds on loss reduction, thereby validating the practical relevance of the theory. Without this prescriptive bridge, theoretical claims risk remaining explanatory storytelling rather than rigorous engineering principles for private model training.

**Core Evaluation Criteria:**
- **Prescriptive vs. Explanatory Value**: Does the theoretical framework yield explicit, actionable formulas or algorithms for hyperparameter selection, or does it merely rationalize empirical observations post hoc?
- **Explicit Assumption Structure**: Are all assumptions required by the theorems clearly enumerated, positioned adjacent to the results, and logically consistent with the empirical setup?
- **Mechanism-to-Optimization Bridge**: Is there a rigorous, quantitative link between the proposed mechanistic insight (e.g., gradient re-weighting, MSE minimization) and a concrete bound on optimization progress or convergence?
- **Mathematical Rigor and Ambiguity Resolution**: Does the theory resolve fixed-point ambiguities, spell out boundary cases, and avoid vague or "hand-wavy" derivations that practitioners cannot implement?

---

Principle 2: Cross-Architectural and Cross-Modal External Validity Verification for DP Fine-Tuning Mechanisms

**Definition:**
Empirical claims regarding optimal hyperparameters in private fine-tuning derive their scientific value from the breadth of experimental conditions under which they are validated, as narrow experimental scopes severely limit external validity. A high-quality study must systematically demonstrate that its identified mechanisms—such as the interaction between clipping bounds and gradient norms—generalize across diverse parameter-efficient fine-tuning strategies, backbone architectures, and data modalities. This includes contrasting methods like FiLM, LoRA, and full fine-tuning, as well as spanning vision transformers, convolutional networks, and language models where feasible. The experimental design should also incorporate relevant baselines, such as automatic clipping or alternative private optimizers, to situate the contributions within the current methodological landscape. Ultimately, the burden of proof lies on the authors to show that their hyperparameter principles are not artifacts of a specific experimental niche but reflect fundamental properties of private gradient-based learning.

**Core Evaluation Criteria:**
- **Architectural and Modal Diversity**: Are the core claims validated across multiple backbone architectures (e.g., ViT, ResNet, BERT-family) and, where relevant, across data modalities (vision, language)?
- **Fine-Tuning Paradigm Coverage**: Does the experimental scope include diverse fine-tuning strategies such as parameter-efficient methods (LoRA, FiLM) and full fine-tuning to ensure mechanisms are not paradigm-specific artifacts?
- **Baseline Comprehensiveness**: Are relevant alternative methods (e.g., automatic clipping, DP-SGD variants, linear probing) fairly compared under identical privacy and compute budgets?
- **External Validity Justification**: Does the paper explicitly discuss limitations regarding generalizability to unseen settings, or does it overclaim universality based on a narrow experimental niche?

---

Principle 3: Temporal Dynamics Verification and Anti-Confounding Analysis of Gradient Statistics in DP Training

**Definition:**
Mechanistic claims about differentially private optimization require dynamic, temporally-resolved empirical verification rather than reliance on static snapshots taken at initialization or convergence. Reviewers must assess whether the work rigorously tracks how key quantities—such as per-example gradient norms, clipping-induced re-weighting, or class-level signal retention—evolve throughout the training trajectory. This temporal analysis is essential to disentangle genuine mechanistic drivers from transient optimization artifacts or confounding factors, such as learning rate schedules and weight decay regimes, which can independently alter gradient statistics near the end of training. The work should further substantiate its interpretations with granular, per-class or per-example evidence that directly links the proposed mechanism to performance disparities. By insisting on dynamic verification, the field avoids accepting superficially plausible but ultimately unfounded narratives about how privacy noise and clipping interact with model training.

**Core Evaluation Criteria:**
- **Temporal Tracking of Key Statistics**: Does the work report time-resolved trajectories of gradient norms (or equivalent quantities) across training iterations rather than relying solely on end-state or post-hoc snapshots?
- **Confounding Factor Control**: Are alternative explanations for observed gradient dynamics—such as learning rate schedules, weight decay, or optimizer state—explicitly ruled out through controlled ablation or theoretical exclusion?
- **Granular Mechanistic Evidence**: Is the proposed mechanism supported by disaggregated evidence (e.g., per-class retained weights, per-example gradient analyses) that directly ties the mechanism to performance variations?
- **Failure Mode Characterization**: Does the study explicitly investigate and report edge cases or failure modes (e.g., full fine-tuning regimes where mechanisms break down) rather than selectively presenting confirming results?

---

Principle 4: Privacy-Cost Accounting and Practitioner-Ready Workflow Design for DP Hyperparameter Selection

**Definition:**
Studies investigating hyperparameter selection for differentially private learning bear a special obligation to confront the privacy economics of their own methodology, as the tuning process itself can consume significant portions of the privacy budget. High-quality research must explicitly quantify, bound, or candidly discuss the privacy cost associated with grid searches, adaptive procedures, or any data-dependent hyperparameter selection, and should articulate how practitioners can implement the findings under strict privacy constraints. This includes distinguishing clearly between purely explanatory mechanistic analysis and deployable, privacy-compliant workflows that minimize additional privacy expenditure. The work should ideally integrate with established private hyperparameter optimization techniques, such as differentially private bandit algorithms or selection-from-candidates protocols, to preserve end-to-end privacy guarantees. Research that ignores the privacy cost of tuning risks presenting theoretically elegant but practically infeasible solutions for real-world private deployments.

**Core Evaluation Criteria:**
- **Privacy Cost Quantification**: Does the work explicitly quantify, bound, or candidly analyze the privacy budget consumed by its own hyperparameter search or tuning protocol?
- **Deployable Workflow Design**: Are the tuning recommendations presented as practical, privacy-compliant procedures (e.g., using DP bandits, random search with private selection) rather than as idealized, non-private grid searches?
- **Distinction Between Analysis and Deployment**: Is there a clear separation between explanatory mechanistic findings and prescriptive deployment guidance, ensuring practitioners understand what can be safely implemented under privacy constraints?
- **Integration with Private HPO Literature**: Does the work engage with and build upon existing differentially private hyperparameter optimization methods, rather than ignoring the privacy overhead of tuning?

---

Principle 5: Formal Operationalization of Learning-Problem Difficulty and Its Causal Link to Private Optimization

**Definition:**
The construct of "learning-problem difficulty" or analogous task-complexity concepts must be formally operationalized using measurable, pre-defined quantities—such as the privacy budget epsilon, model capacity, dataset size, or gradient norm statistics—rather than invoked informally or defined circularly through the hyperparameters under investigation. A rigorous study must establish an independent, validated characterization of difficulty and then systematically demonstrate how this quantity causally modulates the optimal hyperparameter regime, for instance through controlled ablations across privacy levels and backbone capabilities. This operationalization prevents post-hoc rationalization where any observed hyperparameter sensitivity is retrospectively attributed to unmeasured "hardness," thereby rendering the framework unfalsifiable. The definition should be sufficiently general to apply across modalities and architectures while remaining concrete enough to guide experimental design. Clear, formal difficulty metrics are indispensable for transforming intuitive notions of task hardness into reproducible scientific variables within private machine learning research.

**Core Evaluation Criteria:**
- **Formal Definition Independence**: Is the central notion of "difficulty" or "hardness" defined formally using quantities independent of the hyperparameters under study (e.g., ε, model size, dataset statistics), avoiding circular definitions?
- **Causal Ablation Structure**: Does the study manipulate difficulty factors (privacy budget, backbone capability, data scarcity) in a controlled manner to establish causal, not merely correlational, links to optimal hyperparameters?
- **Operational Measurability**: Can the proposed difficulty metric be computed or estimated from observable training data and model properties without requiring oracle knowledge or post-hoc labeling?
- **Cross-Setting Consistency**: Is the difficulty metric validated to produce consistent rankings or predictions across different architectures, modalities, and privacy regimes, ensuring it captures a general property of the learning problem?