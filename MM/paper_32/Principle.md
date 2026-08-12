**Principle 1: Necessity Justification and Marginal Contribution Isolation for Compound Cold-Start Stages Preceding Multimodal RLVR**

**Definition:**  
In multimodal reinforcement learning with verifiable rewards, cold-start pipelines often compose multiple sequential stages such as initial RL warming, self-distilled preference generation, and preference-based pre-alignment. This principle demands that each constituent stage be strictly justified through ablation studies that isolate its marginal contribution, rather than relying on intuitive motivation alone. Reviewers must assess whether the removal or substitution of any stage—such as generating preference data from a base model instead of an RL-warmed model—causes consistent performance degradation. Furthermore, the compound pipeline must demonstrate superiority over simpler baselines that receive an equivalent total computational budget, ensuring that observed gains arise from methodological design rather than confounding by additional training steps or parameters. This standard prevents "over-engineered" pipelines where complexity masquerades as innovation and ensures that every added stage genuinely improves the final policy initialization.

**Core Evaluation Criteria:**
- **Stage-wise Ablation Rigor:** Are all pipeline stages individually ablated (e.g., base-model vs. RL-warmed distillation, DPO vs. SFT pre-alignment) under fixed downstream conditions?
- **Compute-Controlled Comparison:** Does the full pipeline outperform a baseline trained with equivalent total RL steps and aggregate data volume?
- **Motivational Clarity:** Is the theoretical or empirical motivation for each stage explicitly linked to final RL performance, not just intermediate accuracy metrics?
- **Additivity Check:** Does combining stages yield strictly additive or supra-linear gains relative to the best single-stage baseline?

---

**Principle 2: Compute-Equivalent Baseline Control and Full-Cost Transparency in Self-Distilled Preference RL Pipelines**

**Definition:**  
The proliferation of resource-intensive multimodal RL workflows necessitates rigorous accounting of total training costs, including data generation, API-based filtering, and multi-stage GPU computation. This principle requires that proposed methods report transparent cost breakdowns—such as wall-clock hours per stage, API expenses, and aggregate FLOPs—and compare against baselines normalized for equivalent compute expenditure. A critical evaluation criterion is whether the method outperforms a pure RL baseline trained for the same total number of steps as the sum of all RL stages in the proposed pipeline, thereby disentangling algorithmic efficacy from mere compute scaling. Cost transparency extends to data efficiency, requiring authors to justify the scale and selection of synthetic datasets relative to end-task performance. Ultimately, this ensures that reported advances are practically viable and not artifacts of uncontrolled resource inflation.

**Core Evaluation Criteria:**
- **Transparent Cost Reporting:** Are GPU-hours, API costs, and data volumes reported per stage and in aggregate?
- **Normalized Baseline Comparison:** Is the primary baseline a pure RL run matching the total step count of all RL stages combined?
- **Resource Efficiency:** Is the method shown to be cost-competitive with concurrent approaches on either total budget or performance-per-dollar metrics?
- **Data Scaling Justification:** Is the size of synthetic preference datasets justified by ablations or scaling curves?

---

**Principle 3: Mathematical Grounding and Cross-Architecture Predictive Validation of Cold-Start Generalization Metrics**

**Definition:**  
Novel diagnostic metrics introduced to quantify cold-start quality—such as generalization factors balancing in-distribution and out-of-distribution gains—must be grounded in clear mathematical formulation and theoretically interpretable design choices. This principle emphasizes that such metrics cannot remain dataset- or model-specific artifacts; they require validation across diverse backbone architectures to establish broad predictive validity for downstream RL success. Reviewers should examine whether the metric's hyperparameters are principled, whether its correlation with final task performance is statistically established, and whether it remains stable under varying task distributions. The metric should also be dimensionless and normalized to permit fair comparison across experimental settings. Strong adherence to this principle elevates a heuristic observation into a reliable scientific instrument for the community.

**Core Evaluation Criteria:**
- **Mathematical Clarity:** Is the metric formula unambiguous, with all variables and hyperparameters (e.g., F-beta weights) theoretically justified?
- **Cross-Architecture Validation:** Is the metric validated on at least two distinct model families to establish generality beyond a single backbone?
- **Predictive Correlation:** Is quantitative evidence provided linking the metric to final downstream RL performance (e.g., correlation coefficients)?
- **Robustness to Distribution Shift:** Does the metric behave predictably under varying in-distribution and out-of-distribution task compositions?

---

**Principle 4: Mechanistic Ablation Depth for Preference Data Construction and Format-Reasoning Objective Decoupling**

**Definition:**  
Research advocating preference-based cold start or decoupled learning objectives must move beyond end-to-end accuracy comparisons to provide granular, mechanistic ablations of data construction and loss formulation choices. This includes dissecting whether preference pairs are self-distilled or teacher-distilled, whether learning objectives are decoupled (e.g., format-only) or coupled (mixing format and correctness), and whether auxiliary losses (e.g., hybrid SFT-DPO) materially affect training dynamics. Reviewers must evaluate whether authors analyze underlying behavioral mechanisms—such as reward margin evolution, policy loss stability, rollout diversity, or format adherence—that explain why specific configurations yield superior RL initialization. Surface-level performance tables are insufficient without causal insight into how data structure and objective design shape the policy landscape before RL begins. This depth distinguishes rigorous algorithmic research from opportunistic hyperparameter tuning.

**Core Evaluation Criteria:**
- **Data Source Ablation:** Are self-distilled, teacher-distilled, and unfiltered baselines compared under identical training recipes?
- **Decoupling Integrity:** Is the effect of decoupled objectives (e.g., format-only preference pairs) isolated from coupled alternatives mixing multiple learning signals?
- **Training Dynamics Analysis:** Are policy loss curves, reward margins, or exploration diversity metrics analyzed to explain stability and convergence behavior?
- **Mechanistic Explanation:** Do authors provide causal hypotheses (e.g., objective consistency between DPO and GRPO) supported by empirical evidence rather than post-hoc speculation?

---

**Principle 5: Cross-Domain Evaluation Rigor and Backbone-Independent Validation in Multimodal Reasoning RL**

**Definition:**  
Claims of advancing multimodal reasoning through novel training paradigms must be supported by evaluation breadth that transcends narrow mathematical reasoning benchmarks to include perceptual, scientific, and general-knowledge tasks. This principle requires that performance gains be contextualized with statistical rigor—reporting variance, significance tests, or confidence intervals—to distinguish true improvements from evaluation noise inherent in small accuracy differences. Additionally, findings should be replicated across multiple model families or scales to rule out backbone-specific artifacts and confirm the generality of the training paradigm. Reviewers must scrutinize whether the benchmark suite collectively stresses distinct cognitive capabilities (e.g., logical deduction, visual grounding, factual recall) rather than rewarding narrow domain overfitting. Only through such comprehensive and statistically grounded evaluation can a method claim broad applicability in multimodal foundation model development.

**Core Evaluation Criteria:**
- **Benchmark Diversity:** Does the evaluation span mathematical, scientific, perceptual, and general-domain multimodal tasks rather than a single domain?
- **Statistical Significance:** Are p-values, variance across seeds, or confidence intervals reported for key accuracy claims?
- **Backbone Generality:** Are core findings replicated on multiple vision-language architectures to rule out model-family-specific artifacts?
- **Capability Disaggregation:** Do results distinguish between improvements in deep reasoning, memorization, and superficial format compliance?