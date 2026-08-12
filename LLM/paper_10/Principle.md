**Principle 1: Mechanistic Explainability and Qualitative Validation of Token-Level Filtering Decisions in Supervised Fine-Tuning**

**Definition:**  
For token-level filtering methods that claim explainability, reviewers expect substantive evidence extending beyond aggregate accuracy improvements. The core challenge lies in demonstrating that the proposed scoring dimensions—such as reasoning importance, knowledge novelty, and task relevance—actually capture mechanistically distinct properties of individual tokens rather than redundant or spurious statistical signals. Because token-level optimization operates at a granularity below human sentence-level perception, qualitative validation through concrete examples and visualizations becomes essential to establish trust in the filtering decisions. Furthermore, the decomposition of token value into multiple attributes must be theoretically and empirically justified as non-overlapping and complementary, ensuring that each dimension contributes unique explanatory power. Without such mechanistic interpretability, token filtering risks becoming a black-box procedure where performance gains cannot be attributed to meaningful data curation. The field therefore demands that authors confront ambiguous cases where metrics might conflict, showing how the framework resolves such tensions rather than ignoring edge cases.

**Core Evaluation Criteria:**
- **Qualitative Evidence**: Does the work provide concrete visualizations and examples showing exactly which tokens are filtered and why, across diverse domains (math, code, medicine)?
- **Semantic Alignment**: Do the filtering decisions align with human intuition about token importance, or do they reveal non-obvious but valid patterns that can be mechanistically justified?
- **Attribute Decomposition Rigor**: Is the multi-attribute decomposition theoretically motivated and empirically verified to be complementary rather than redundant?
- **Handling of Ambiguous Cases**: Does the work explicitly address edge cases where individual attributes might misidentify valuable tokens (e.g., low PCP indicating noise versus novel knowledge)?

---

**Principle 2: Robustness and Generalizability of Statistical Threshold Selection for Multi-Dimensional Token Scoring**

**Definition:**  
Multi-attribute token filtering frameworks inevitably depend on thresholding mechanisms to convert continuous scores into binary masking decisions, making the robustness of these thresholds a critical evaluation axis. The review process heavily scrutinizes whether threshold selection represents a principled statistical choice—such as quantile-based cutoffs or clustering-derived boundaries—or merely an ad-hoc heuristic tuned to specific experimental distributions. Because token score distributions shift across model families, scales, and downstream tasks, a method must demonstrate that its thresholding strategy maintains efficacy without task-specific re-optimization, thereby proving generalizability rather than overfitting. Reviewers particularly value analyses of interaction effects between thresholds across multiple dimensions, as complementarity between attributes can compensate for individual threshold imprecision, whereas highly sensitive thresholds indicate methodological brittleness. A robust framework should tolerate small perturbations in threshold values without catastrophic performance degradation, suggesting that the method operates in a stable regime rather than at a knife-edge optimum.

**Core Evaluation Criteria:**
- **Sensitivity Analysis Breadth**: Does the work test threshold perturbations across a meaningful range (not just narrow intervals like ±3%) and report performance variance systematically?
- **Cross-Scenario Stability**: Are thresholds derived from distribution-aware statistics (e.g., IQR, Multi-Otsu) that transfer across datasets, or do they rely on fixed magic numbers?
- **Interaction and Complementarity Analysis**: Are the overlaps and interactions between multi-dimensional thresholds quantified to demonstrate true complementarity?
- **Hyperparameter Brittleness**: Does small threshold variation cause large performance swings (e.g., >5% accuracy drops), indicating an unstable and ungeneralizable design?

---

**Principle 3: Theoretical Grounding and Convergence Guarantees for Gradient Alignment in Token Masking Methods**

**Definition:**  
Token-level gradient masking fundamentally alters the optimization trajectory during fine-tuning, necessitating rigorous theoretical analysis that connects filtering mechanisms to convergence properties and gradient geometry. Reviewers expect formal derivations that demonstrate how masking noisy tokens improves the alignment between the empirical training gradient and the ideal gradient derived from a clean data distribution, ideally expressed within a preconditioned optimization framework such as natural gradient or general SPD geometries. Such theoretical grounding must be assumption-transparent, explicitly stating conditions like local smoothness, mixture model structures for noise, and teacher-forcing invariance, so that readers can assess the scope of validity. The theory should distinguish between performance gains stemming from reduced variance versus improved alignment, ensuring that the method is understood as a principled optimization intervention rather than an empirical trick. Crucially, theoretical claims must be accompanied by empirical ablations that validate specific predictions, such as the negligible alignment cost of removing high-confidence tokens or the complementarity of multi-dimensional filtering.

**Core Evaluation Criteria:**
- **Formal Alignment Guarantees**: Does the theory prove that filtering improves alignment between the training gradient and the ideal gradient direction under explicitly stated assumptions?
- **Assumption Transparency**: Are assumptions (e.g., MAR within components, local smoothness, bounded selection bias) clearly articulated and empirically plausible?
- **Empirical-Theoretical Consistency**: Do ablation studies validate theoretical predictions (e.g., that high-confidence KN tokens contribute negligibly to alignment)?
- **Distinction from Black-Box Performance Gains**: Does the work clearly separate "stability enhancement" from raw "performance improvement" and establish a causal link between them?

---

**Principle 4: Cross-Task, Cross-Model, and Cross-Scale Empirical Rigor for Token-Level Data Optimization Frameworks**

**Definition:**  
Comprehensive empirical validation is essential for token-level dataset optimization methods to avoid the suspicion of being narrow artifacts tailored to specific model-task combinations. Reviewers demand broad coverage across diverse downstream tasks—such as mathematical reasoning, code generation, and domain-specific knowledge—because token distribution characteristics and noise profiles differ substantially between reasoning-heavy and fact-heavy applications. Equally important is evaluation across multiple model families and scales, ranging from small distilled models to large foundation models, and across both full-parameter and parameter-efficient fine-tuning regimes like LoRA. The inclusion of competitive and recent baselines, particularly other token-level methods and strong sample-level filtering approaches, is necessary to situate the contribution within the current state of the art. Beyond raw accuracy metrics, reviewers expect statistical rigor in the form of variance estimates, confidence intervals, or significance tests, especially given the computational constraints that may limit the number of repeated runs.

**Core Evaluation Criteria:**
- **Task Diversity**: Are experiments conducted on multiple task types with fundamentally different output structures and reasoning demands (e.g., math, code, medicine, finance)?
- **Model Family and Scale Coverage**: Does the evaluation span different architectures and scales (e.g., 1B to 14B+, both full fine-tuning and LoRA)?
- **Baseline Competitiveness**: Are recent and strong token-level baselines included (e.g., Token Cleaning, SLM/Rho), not just naive fine-tuning or sample-level methods?
- **Statistical Rigor**: Are variance, confidence intervals, or significance tests reported to ensure improvements are reproducible and not cherry-picked from best runs?

---

**Principle 5: Inference Cost Efficiency and Practical Feasibility of Token Scoring Compared to Reference-Model Baselines**

**Definition:**  
The practical adoption of token-level filtering methods hinges critically on their computational efficiency relative to the performance improvements they deliver, particularly in industrial settings where dataset sizes and model scales create severe resource constraints. Reviewers scrutinize whether the token scoring pipeline requires training auxiliary reference models—which incurs substantial training-level overhead—or can be accomplished via inference-only passes, since this distinction dramatically impacts scalability and cost-effectiveness. A thorough evaluation must provide explicit measurements of GPU memory consumption, time usage, and computational complexity, ideally broken down by scoring phase and compared directly against baseline methods under standardized conditions. Furthermore, authors should analyze how these costs scale with sequence length, vocabulary coverage, and dataset volume, rather than providing single-point estimates that obscure scalability bottlenecks. The total computational budget, encompassing both preprocessing and fine-tuning, must be justified by the magnitude of downstream performance gains; methods that double preprocessing time for marginal improvements face skepticism.

**Core Evaluation Criteria:**
- **Cost Transparency**: Is there explicit analysis of GPU memory, wall-clock time, and computational complexity for the scoring phase across different model sizes?
- **Comparative Efficiency**: How does the total cost (scoring + fine-tuning) compare to baselines requiring auxiliary model training (e.g., SLM, Token Cleaning) or multiple inference rounds?
- **Scalability Analysis**: Does the cost analysis address scaling with respect to sequence length, dataset size, and model parameter count?
- **Practical Feasibility**: Does the method remain viable for large-scale deployment, and are there proposed mitigations (e.g., distillation) for reducing preprocessing overhead?