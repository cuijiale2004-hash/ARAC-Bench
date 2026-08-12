**Principle 1: Identification and Mitigation of Objective Misalignment Between Supervised Imitation and Reinforcement Learning Readiness**

**Definition:**  
The researcher must demonstrate that standard supervised fine-tuning objectives—particularly cross-entropy minimization and validation accuracy maximization—are fundamentally misaligned with the goal of preparing a base model for subsequent reinforcement learning. The work should establish that optimizing solely for demonstration imitation can lead to premature distributional forgetting of base model capabilities before traditional overfitting is observed, thereby degrading downstream RL potential. The evaluation focuses on whether the paper identifies reliable proxy metrics (e.g., response diversity, entropy, self-BLEU) that better predict post-RL performance than conventional SFT metrics, and whether the proposed training objective actively reconciles this misalignment rather than merely improving surface-level SFT results. This principle is crucial because modern LLM post-training pipelines universally depend on the SFT-to-RL handoff, yet the criteria for optimal initialization remain under-theorized.

**Core Evaluation Criteria:**
- **Misalignment Demonstration**: Does the work provide rigorous empirical evidence that the best SFT checkpoint (by standard metrics) is suboptimal for RL, and clearly distinguish between “SFT performance” and “RL readiness”?
- **Proxy Validity**: Are the proposed alternative metrics rigorously validated as early-stopping criteria across multiple settings, showing stronger and more consistent correlation with post-RL performance than accuracy or loss?
- **Causal Establishment**: Does the work establish a causal or mechanistic link between preserving base model distribution characteristics and improved RL outcomes, rather than relying solely on correlational improvements in final accuracy?

---

**Principle 2: Cross-Model-Family and Cross-Domain Robustness Validation for Post-Training Initialization Strategies**

**Definition:**  
Because RL preparation strategies may interact differently with distinct base model priors, architectures, and task structures, the work must demonstrate robustness across heterogeneous experimental conditions rather than relying on a single model-task dyad. This includes validation on multiple model families with diverse pre-training distributions, as well as evaluation across qualitatively different downstream tasks (e.g., mathematical reasoning, general reasoning, code generation, or instruction following). The principle ensures that findings represent generalizable insights into cold-start design rather than family-specific engineering artifacts. Reviewers place significant weight on whether the method maintains consistent trends when transferred to unfamiliar base distributions and non-mathematical domains.

**Core Evaluation Criteria:**
- **Model Family Diversity**: Are experiments conducted on multiple, distinct base model families (e.g., Qwen, Llama, Gemma) rather than minor variants within a single family?
- **Task Domain Breadth**: Does the evaluation span multiple domains beyond a single narrow task (e.g., extending from math to general reasoning, code, or alignment tasks), or does it remain confined to one subject area?
- **Consistency and Limitation Reporting**: Do the proposed methods consistently outperform baselines across all tested configurations, and are failure modes or domain-dependent weaknesses honestly reported rather than obscured?

---

**Principle 3: Mechanistic Interpretability and Root-Cause Attribution of Base Model Distribution Forgetting During Lightweight SFT**

**Definition:**  
The work must move beyond phenomenological observation of “distributional forgetting” to provide deeper mechanistic insights into how and why the base model’s distribution degrades during lightweight SFT. This includes analyzing the dynamics of representation drift, identifying whether forgetting occurs uniformly or is concentrated at specific layers, token positions, or knowledge types, and distinguishing between beneficial adaptation (acquiring new reasoning patterns) and harmful forgetting (loss of base knowledge). The principle demands that the paper connect distributional drift metrics to RL-specific phenomena such as exploration capacity and policy entropy. Without such mechanistic depth, the work risks being treated as a shallow engineering observation rather than a foundational contribution to post-training science.

**Core Evaluation Criteria:**
- **Depth of Mechanistic Analysis**: Does the work investigate underlying causes (e.g., overfitting to small demonstration sets, layer-wise representation collapse, or attention pattern entrenchment) rather than merely documenting the existence of forgetting?
- **Granularity of Attribution**: Are analyses provided at fine granularity (token-level, layer-level, or subsequence-level) to pinpoint precisely where distribution preservation matters most for RL preparation?
- **Link to RL Dynamics**: Does the work explicitly connect distributional diversity or its absence to RL-specific behaviors such as exploration collapse, reduced policy entropy, or impaired credit assignment?

---

**Principle 4: Design Rigor and Systematic Ablation of Adaptive Objective Functions for Distribution-Preserving Cold-Start**

**Definition:**  
When proposing novel adaptive loss functions or re-weighting schemes for cold-start training, the work must provide rigorous justification for its specific functional form and thoroughly ablate each design choice against simpler alternatives. Reviewers assess whether the proposed objective is principled or ad-hoc by examining theoretical motivation, comparisons against structurally simpler baselines (e.g., linear weighting, threshold-based masking, constant denominators, or naive early stopping), and sensitivity to key hyperparameters. This principle ensures that observed performance gains are attributable to the intended mechanism rather than accidental engineering choices, and it prevents overfitting the loss function to a narrow experimental setting. The decomposition of sequence-level diversity into token-level and subsequence-level components should be empirically validated, not merely asserted.

**Core Evaluation Criteria:**
- **Theoretical and Empirical Motivation**: Is the functional form (e.g., softmax-based weighting, prefix-confidence denominator) strongly motivated by analytical decomposition or empirical insight, rather than selected opaquely by trial-and-error?
- **Alternative Ablation**: Are simpler alternative designs systematically compared, including removal of key terms (e.g., prefix log-probability denominator) and structurally simpler weighting schemes?
- **Hyperparameter Sensitivity**: Is the method’s performance shown to be robust within a reasonable hyperparameter range, with extreme values shown to degrade performance predictably?
- **Complexity Transparency**: Are memory and computational overheads of the new objective explicitly analyzed relative to standard cross-entropy training?

---

**Principle 5: Computational Cost-Benefit Justification and Overhead Control in Lightweight Pre-RL Training Phases**

**Definition:**  
Because cold-start SFT is explicitly positioned as a lightweight, resource-efficient preparatory phase for computationally expensive RL training, the work must rigorously justify any additional computational cost introduced by proposed methods. This principle evaluates whether overhead (in training time, GPU memory, or auxiliary forward passes) is accurately quantified, whether it remains negligible relative to the total post-training budget, and whether the performance improvements genuinely warrant the cost. The evaluation prioritizes practical deployability in standard industrial post-training pipelines, where cold-start efficiency determines iteration speed. A method that consumes disproportionate resources during the cold-start phase undermines its own premise of lightweight preparation, regardless of downstream accuracy gains.

**Core Evaluation Criteria:**
- **Overhead Quantification**: Are the additional computational costs (e.g., wall-clock time increase, memory overhead, extra forward passes) explicitly measured and reported in absolute terms?
- **Budget Contextualization**: Is the overhead presented in context of the total post-training cost (e.g., cold-start phase time versus RL phase time), demonstrating that it does not erase the method’s practical benefits?
- **Efficiency-Performance Trade-off**: Does the work demonstrate that the performance gains cannot be achieved by simpler, cheaper alternatives (e.g., naive early stopping or reduced training epochs) without comparable or greater resource expenditure?
- **Scalability Assessment**: Is there discussion or evidence of how the overhead scales with model size, sequence length, or dataset size, ensuring feasibility beyond the current experimental scale?