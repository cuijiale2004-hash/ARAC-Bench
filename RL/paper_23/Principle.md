**Principle 1: Clarity of Cognitive Motivation and Distinct Mechanistic Roles of Simulation Modules Within Joint Policy-Environment Reasoning Architectures**

**Definition:**  
In long-horizon agent research that integrates mental simulation into reasoning, the architectural design must be conceptually crisp regarding which components handle environment dynamics modeling versus action selection, and why offline distillation and online reinforcement are synergistically staged. Reviewers must assess whether the authors clearly articulate the cognitive inspiration—such as vicarious trial and error—and rigorously map each module to its mechanistic function, avoiding conflation of the policy, world model, and search aggregation procedures. This clarity is crucial because opaque pipelines hinder reproducibility, invite conceptual confusion between environment simulation and policy optimization, and obscure whether performance gains stem from improved reasoning structure or merely from additional computation. Without such clarity, it is impossible to determine whether the agent has learned a faithful internal world model or has simply memorized surface-level action heuristics.

**Core Evaluation Criteria:**
- **Conceptual Distinctness**: Does the work explicitly define the boundary (or intentional unification) between the environment model, policy, and search/aggregation components, and justify the architectural necessity of each stage?
- **Cognitive Grounding**: Is the human-cognition motivation translated into concrete algorithmic decisions rather than used as loose thematic inspiration?
- **Pipeline Justification**: Are the transitions between data construction, offline distillation, and online RL rigorously motivated, with each module's additive value explained?
- **Avoidance of Modular Obfuscation**: Does the work prevent conflation of the training-time search algorithm with the learned inference-time simulation capability?

---

**Principle 2: Systematic Ablation of Tree-Search Data Construction, Distillation Targets, and RL Refinement Signals in Simulation-Based Agent Training**

**Definition:**  
Research that combines offline imitation learning from search-derived data with online RL for simulation enhancement must provide exhaustive ablations that isolate the contribution of each constituent module—including the search algorithm, the trajectory types used for distillation, the intermediate-state feedback mechanisms, and the RL advantage formulations. Reviewers must evaluate whether the authors establish causal links between specific design choices and performance outcomes, rather than presenting only end-to-end results. This principle is essential because multi-stage pipelines compound variance and cost; without granular ablations and statistical dispersion metrics, it is impossible to distinguish genuine methodological advances from artifacts of confounded compute budgets or random seed variation. Furthermore, sensitivity analysis regarding the search heuristic is necessary to prove that the agent’s simulation ability generalizes beyond a specific tree-search bias.

**Core Evaluation Criteria:**
- **Granular Module Ablation**: Are individual components systematically removed or altered to quantify their isolated impact on both simulation quality and task success?
- **Statistical Robustness**: Are results reported with variance metrics to ensure observed differences between ablated conditions are statistically meaningful?
- **Search vs. Learning Disentanglement**: Does the work decouple the choice of training-time search heuristic from the agent’s learned inference-time simulation ability?
- **Trajectory-Type Analysis**: Are the different rollout and refinement trajectories individually evaluated to validate their necessity in the RL objective?

---

**Principle 3: End-to-End Computational Cost Transparency and Training-Inference Efficiency Trade-off Analysis in Long-Horizon Agent Simulation Learning**

**Definition:**  
For methods that introduce additional training stages—such as search-tree expansion, rollout value estimation, and simulation refinement—to imbue agents with world-modeling capabilities, reviewers must demand comprehensive accounting of computational expenditures relative to both performance gains and baseline methods. This principle is vital because simulation-augmented pipelines inherently carry heavy compute footprints involving data-generation tokens, training FLOPs, wall-clock time per stage, and inference token overhead. Claims of efficiency or amortized benefits cannot be accepted without quantitative evidence across diverse environment complexities, from synthetic grid-worlds to realistic GUI navigation. Transparent cost reporting enables the community to judge whether the simulation ability is purchased at a reasonable price and whether the method can practically scale beyond lightweight benchmarks.

**Core Evaluation Criteria:**
- **Stage-Wise Cost Breakdown**: Are training costs explicitly decomposed by stage and compared against baselines with matched compute budgets?
- **Inference Efficiency**: Is the inference-time token cost and latency reported relative to strong baselines, demonstrating whether learned simulation reduces reliance on external search?
- **Scalability Evidence**: Does the evaluation include at least one realistic, high-complexity environment to substantiate claims that the method scales beyond simplified domains?
- **Amortization Justification**: Do the authors demonstrate that higher one-time training costs are justified by sustained inference efficiency or final performance deltas?

---

**Principle 4: Quantitative Validation of Internal World-Model Simulation Fidelity and Its Predictive Correlation with Long-Horizon Task Success**

**Definition:**  
A central claim of simulation-based agent research is that the policy has acquired an accurate internal world model; therefore, reviewers must require direct evidence that the agent’s generated state predictions faithfully reflect true environment dynamics, and that improvements in this simulation fidelity causally drive task success rather than merely co-occurring with it. This principle mandates dedicated metrics—such as simulation scores judged against ground-truth states, correlation coefficients between prediction accuracy and success rates, and qualitative trace analysis—to prevent black-box reasoning where performance gains are attributed to simulation without proof. Without such validation, a method risks conflating better action selection with spurious reasoning patterns that do not represent genuine environment understanding.

**Core Evaluation Criteria:**
- **Direct Simulation Metrics**: Are there explicit, automated metrics measuring the correctness of imagined future states independent of final reward?
- **Correlation Analysis**: Do the authors establish a statistical relationship between simulation accuracy and task performance across models or training stages?
- **Qualitative Trace Inspection**: Are example reasoning traces provided and analyzed to verify that the model utilizes coherent, faithful simulation?
- **Avoidance of Performance-Only Inference**: Does the work refrain from inferring improved simulation solely from end-task success, instead isolating simulation quality as an intermediate variable?

---

**Principle 5: Comprehensive Generalization Assessment Across Synthetic-to-Realistic Modalities and In-to-Out-of-Distribution Task Horizons**

**Definition:**  
Methods that teach agents to simulate must demonstrate that the acquired simulation ability transfers beyond narrow, in-distribution training conditions to diverse modalities, varying task types, and structurally novel planning horizons. Reviewers must evaluate whether the experimental suite spans a meaningful spectrum from controlled synthetic environments to realistic, high-dimensional domains, and whether explicit out-of-distribution testing is conducted. This principle is critical because internal world models can easily overfit to specific environment dynamics or action vocabularies; robust simulation-based reasoning should manifest as consistent gains on held-out task categories and visually complex settings. Only through such diversity can one conclude that the agent has learned transferable predictive principles rather than dataset-specific shortcuts.

**Core Evaluation Criteria:**
- **Benchmark Spectrum**: Does the evaluation cover both lightweight, interpretable environments and at least one realistic, high-stakes domain?
- **Out-of-Distribution Testing**: Are there explicit held-out test sets with novel task types or configurations to assess simulation generalization?
- **Modality Transfer**: Does the method demonstrate efficacy across different input modalities, such as text-based states and pixel-based screenshots?
- **Planning Horizon Consistency**: Do the performance gains persist across tasks requiring varying depths of lookahead?