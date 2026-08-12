Principle 1: Theoretical Rigor and Epistemic Validity of Information-Theoretic Metrics for Representational Alignment in Model Composition

**Definition:**
In research proposing novel metrics to guide neural network stitching or model composition, theoretical justification must be precise, mathematically sound, and scrupulously avoid overstatement. Reviewers emphasized that KL divergence possesses a rigorous information-theoretic interpretation—quantifying the difference between probability distributions—which does not automatically imply direct representational similarity in feature space. A principled submission must either provide formal proof that its chosen metric captures claimed dual properties, such as both representational structure and functional alignment, or carefully circumscribe its claims to empirically observed phenomena. Without such grounding, the central contribution risks being perceived as post-hoc rationalization rather than a scientifically valid advance. This principle is especially critical in efficient deployment research, where proxy metrics dictate model selection; poorly theorized proxies can lead to unpredictable failures across architectures or downstream tasks. Furthermore, the work must distinguish clearly between performance improvement and stability or selection-quality enhancement, establishing causal rather than merely correlational links.

**Core Evaluation Criteria:**
- **Accuracy of Metric Interpretation:** Does the paper accurately characterize what the metric measures (e.g., distributional divergence versus feature geometry), avoiding technically imprecise claims?
- **Substantiation of Uniqueness or Superiority:** Are claims that a metric "uniquely satisfies" objectives supported by formal proof or appropriately scoped to empirically tested alternatives?
- **Distinction Between Similarity Types:** Does the work clearly differentiate and justify how the metric captures representational versus functional similarity, rather than conflating them?
- **Causal vs. Correlational Reasoning:** Does the analysis establish a mechanistic causal link between the metric and stitching success, or does it merely observe correlation?

---

Principle 2: Comparative Fairness and Baseline Completeness for Accuracy-Efficiency Tradeoff Mechanisms

**Definition:**
When introducing a mechanism for optimizing accuracy-efficiency tradeoffs, empirical evaluation must extend beyond the immediate predecessor method to include dominant alternative paradigms such as model cascades, early-exit networks, and dynamic inference systems. Reviewers identified the absence of cascade baselines as a critical methodological gap because cascades represent a standard, training-free approach that addresses an identical deployment problem. A thorough evaluation should report both average-case and worst-case computational profiles, clearly delineating structural conditions under which stitching is preferable—such as deterministic latency guarantees versus dynamic routing uncertainty. The comparison must also ensure that baseline methods are themselves optimally tuned for their own operating points, preventing an unfairly skewed Pareto front. This principle ensures that claims of practical utility are stress-tested against the full spectrum of real-world deployment strategies rather than a narrowly curated benchmark set. Without such comprehensive comparison, the practical significance of marginal AUC improvements remains ambiguous and potentially misleading to practitioners seeking deployment solutions.

**Core Evaluation Criteria:**
- **Inclusion of Standard Paradigms:** Are model cascades or comparable dominant baselines (e.g., early-exit, dynamic routing) rigorously evaluated under identical accuracy-efficiency objectives?
- **Worst- and Average-Case Analysis:** Does the comparison report deterministic versus stochastic cost profiles, including worst-case latency and average FLOPs?
- **Structural Preference Justification:** Does the paper analytically explain when and why stitching is structurally preferable to additive-cost alternatives?
- **Fairness of Comparison:** Are baselines tuned to optimize their own Pareto fronts (e.g., confidence thresholds for cascades) before comparison?

---

Principle 3: Cross-Modal and Cross-Task Generalizability Validation for Model Reuse and Composition Methods

**Definition:**
Research claiming to automate model composition across diverse "model families" or to provide generalizable deployment solutions must supply empirical validation that transcends a single task, dataset, or modality. Reviewers consistently demanded expansion beyond supervised image classification to dense prediction tasks—such as semantic segmentation—and to architecturally distant domains including autoregressive large language models. This expectation stems from the reality that deployment ecosystems span vision, language, and multimodal applications; a stitching framework that functions only on ImageNet-classification checkpoints has severely bounded practical relevance. Demonstrating efficacy on hierarchical vision transformers, residual convolutional networks, and decoder-only LLMs is necessary to prove that the similarity metric and selection algorithm are not overfitted to a narrow architectural inductive bias. Moreover, the work must honestly delimit its scope if broad generalization is not shown, rather than implying universal applicability through vague framing. Cross-domain validation is therefore not merely an additive experimental bonus but a core requirement for establishing the method's true impact and audience.

**Core Evaluation Criteria:**
- **Multi-Task Evidence:** Does the work include experiments on qualitatively different task types (e.g., classification, semantic segmentation, next-token generation)?
- **Multi-Modal or Cross-Architecture Scope:** Are results reported across architectural families (convolutional, transformer, decoder-only LLM) rather than a single homogeneous family?
- **Scope Honesty:** If generalization is not demonstrated, does the paper limit its claims appropriately rather than implying universal applicability?
- **Backbone Diversity:** Does the evaluation include self-supervised or instruction-tuned backbones, or only supervised checkpoints?

---

Principle 4: Reproducible Experimental Protocols and End-to-End Computational Cost Transparency in Model Stitching Pipelines

**Definition:**
Transparency regarding implementation details and total computational cost is essential for any method that claims to improve efficiency through automated model composition. Reviewers raised serious reproducibility concerns centered on missing fine-tuning protocols, unclear convergence criteria, and undocumented data splits, all of which prevent independent verification of reported AUC gains. The submission must fully specify training hyperparameters, probe architectures, stitching layer initialization, and the complete algorithmic pipeline, ideally including pseudocode or reference implementations. Beyond protocol clarity, the true resource overhead of the proposed approach—including probe training, similarity computation, candidate selection, and stitched-network fine-tuning—must be quantified and compared against heuristic baselines in wall-clock time or GPU-days. This accounting is particularly important when the method introduces additional search stages that, while individually cheap, may alter the total cost-benefit calculus relative to training-free alternatives. Reproducible and transparent reporting transforms an interesting proof-of-concept into a credible tool that the community can adopt, extend, and fairly benchmark.

**Core Evaluation Criteria:**
- **Fine-Tuning Fidelity:** Are training protocols (epochs, learning rates, convergence criteria, data splits) fully disclosed to enable exact reproduction?
- **End-to-End Cost Reporting:** Is the total resource overhead of the proposed pipeline—including proxy training, search, and stitch fine-tuning—quantified and compared against baselines?
- **Identifiability of Artifacts:** Does the work clarify whether compared methods produce identical models when selections overlap, ensuring observed gains stem from selection quality rather than training variance?
- **Algorithmic Accessibility:** Are key algorithms (e.g., probe architectures, selection procedures) presented with sufficient detail for independent implementation?

---

Principle 5: Sensitivity and Robustness Characterization of Similarity-Driven Configuration Selection Algorithms

**Definition:**
A similarity-driven selection algorithm must be accompanied by rigorous ablation and sensitivity analyses that characterize its behavior under varying design choices and inherent metric properties. Reviewers noted that critical hyperparameters—such as FLOP bucket granularity and selection thresholds—were underexplored, as were intrinsic characteristics of KL divergence including its asymmetry and sensitivity to calibration or temperature. The work should systematically vary these factors and demonstrate that the method's conclusions remain stable, thereby proving that reported gains are not artifacts of a fortuitous parameter setting. Additionally, the predictive validity of the similarity proxy must be explicitly validated through rank-correlation statistics linking pre-selection scores to post-fine-tuning task performance. Finally, failure modes—such as marginal gains in cross-architecture stitches—must be diagnosed via quantitative diagnostics like similarity heatmaps rather than being silently omitted. This depth of analysis distinguishes a robust, trustworthy method from one that relies on opaque, black-box selection criteria.

**Core Evaluation Criteria:**
- **Hyperparameter Stability:** Are key design choices (e.g., selection threshold, bucket granularity) ablated over a meaningful range, showing stable performance?
- **Metric Property Analysis:** Does the paper investigate inherent characteristics of the chosen metric (e.g., asymmetry, temperature/calibration sensitivity) and mitigate or explain their impact?
- **Proxy-to-Performance Validation:** Is the predictive power of the selection metric validated via rank correlation (e.g., Spearman/Pearson between similarity score and final accuracy drop)?
- **Failure Mode Diagnosis:** Are cases of small or marginal gains diagnosed to distinguish between inherent method limitations and suboptimal heuristic recovery?