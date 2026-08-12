**Principle 1: Rigorous Hyperparameter Equity and Fair Baseline Calibration in Training-Free Visual Token Compression**

**Definition:**  
In the domain of training-free visual token enhancement for Video LLMs, it is imperative that any proposed compression or restructuring method be evaluated against prior arts under strictly controlled conditions to ensure that observed gains are attributable to algorithmic innovation rather than experimental advantage. Reviewers must scrutinize whether authors perform hyperparameter searches for competing baselines (e.g., grid numbers, retention ratios), enforce identical token budgets across methods, and align backbone configurations. This principle is crucial because training-free methods often combine similar atomic operations—such as gridding, pooling, and sampling—in marginally different ways, making uncontrolled comparisons highly susceptible to inflating apparent contributions. A fair benchmarking protocol prevents false positives in performance gains and establishes a credible foundation for claiming practical applicability across diverse multimodal LLMs. Without such equity, the community cannot disentangle true representational improvements from artifacts of suboptimal baseline tuning or mismatched computational allotments.

**Core Evaluation Criteria:**
- **Hyperparameter Optimization for Baselines:** Did the authors conduct a systematic search over critical baseline hyperparameters and report the best-performing configuration, rather than using default or suboptimal settings?
- **Strict Token Budget Parity:** Is the number of retained visual tokens held constant across all compared methods to ensure that performance differences stem from representation quality rather than information quantity?
- **Controlled Backbone and Frame Settings:** Are identical vision encoders, LLM backbones, input frame counts, and evaluation protocols used across the proposed approach and all baselines?
- **Distinction from Prior Atomic Operations:** Does the work clearly articulate how the integration or hierarchical extension of existing operations (e.g., moving from single-level image gridding to multi-level token gridding) constitutes a non-trivial methodological advance?

---

**Principle 2: Granular Computational Decomposition and Pareto-Efficiency Validation for Plug-and-Play Token Modules**

**Definition:**  
For plug-and-play modules designed to reduce visual token overhead, reviewers must demand a fine-grained dissection of computational costs, including isolated latency and memory footprints of individual sub-modules and end-to-end comparisons against contemporary token-reduction techniques. It is insufficient to claim efficiency solely by comparing against an uncompressed baseline; the method must demonstrate its position on the accuracy-efficiency Pareto frontier relative to specialized competitors. This principle is vital because training-free enhancements are frequently motivated by real-time deployment needs, and opaque efficiency claims undermine their practical value. A thorough breakdown reassures the community that preprocessing introduces negligible overhead and that speedups are not offset by hidden computational costs. Ultimately, module-level accountability ensures that plug-and-play methods are genuinely deployable without disrupting existing inference pipelines.

**Core Evaluation Criteria:**
- **Micro-Benchmarked Module Latency:** Are the individual latencies of each component separately measured and reported as a percentage of total end-to-end inference time?
- **Comparative Pareto Analysis:** Does the work benchmark both accuracy and latency against state-of-the-art token reduction methods under identical hardware and compression budgets?
- **Scalability of Overhead:** Is the overhead trend analyzed as input frame count increases, confirming that the module cost remains negligible or vanishing relative to the vision encoder and autoregressive LLM?
- **Memory Footprint Transparency:** Is peak GPU memory usage reported alongside latency to validate claims of computational efficiency?

---

**Principle 3: Theoretical and Mechanistic Grounding of Heuristic Token-Saliency Assumptions in Spatial Downsampling**

**Definition:**  
When training-free methods rely on empirical heuristics—such as the correlation between token L2 norms and semantic saliency, or the choice to overwrite rather than append summary tokens—reviewers must require both theoretical justification and empirical validation of these design choices. This principle addresses the risk of "black-box" reasoning where performance improvements are asserted without causal explanations for why the heuristic functions in the target domain. In Video LLMs, the interaction between visual tokens and the LLM's attention mechanism must be articulated, for instance, by linking token norms to softmax logit magnitudes or causal masking structures. Mechanistic clarity enables the community to assess whether a design is principled or accidental, and whether it will generalize across different vision encoders or video genres. Without such grounding, the method remains an uninterpretable engineering trick rather than a reliable scientific contribution.

**Core Evaluation Criteria:**
- **Theoretical Rationale for Heuristics:** Does the paper provide a domain-specific theoretical explanation for why the chosen heuristic operates effectively within transformer-based visual-language architectures?
- **Design Choice Justification:** Are critical implementation decisions—such as overwriting tokens versus appending, or pyramid versus single-level gridding—motivated by causal modeling constraints or fairness requirements rather than mere convenience?
- **Alternative Indicator Comparison:** Does the work empirically compare the chosen heuristic against alternative unsupervised saliency indicators to demonstrate its superiority?
- **Interaction Between Modules:** Is the interaction between sequential modules (e.g., how temporal gridding affects the norm distribution input to spatial pooling) analyzed and explained?

---

**Principle 4: Cross-Scale Robustness Verification Across Temporal Horizons, Backbones, and Compression Budgets**

**Definition:**  
A training-free token enhancement method must demonstrate robust performance not merely on aggregate benchmark scores, but across diverse temporal scales, multiple model backbones, and varying compression ratios. This principle is essential because Video LLMs are deployed in heterogeneous scenarios—from brief action recognition to long-form temporal reasoning—and a method that excels only on a single benchmark or backbone offers limited generalizable insight. Reviewers should expect consistent gains across general and long-video understanding tasks, as well as ablations showing sensitivity to architectural hyperparameters such as pyramid depth and segment lengths. Furthermore, performance must be validated under aggressive token budgets where compression artifacts are most severe, ensuring the method preserves critical spatiotemporal structures when resources are constrained. Cross-scale validation separates robust architectural improvements from dataset-specific overfitting.

**Core Evaluation Criteria:**
- **Temporal Scale Generalization:** Are results reported and analyzed across diverse video durations to verify effectiveness for both short-term dynamics and long-range dependencies?
- **Backbone and Architecture Portability:** Does the method demonstrate consistent improvements across multiple Video LLM families without model-specific modifications?
- **Compression Budget Resilience:** Are experiments conducted under multiple token retention ratios, and does the method maintain or improve its relative advantage under higher compression?
- **Hyperparameter Sensitivity Analysis:** Are key hyperparameters (e.g., temperature coefficients, norm orders, pyramid levels) systematically ablated to demonstrate stable and interpretable performance trends?

---

**Principle 5: Explicit Trade-off Analysis Between Foreground Saliency Amplification and Background Context Preservation**

**Definition:**  
Methods that selectively amplify high-saliency regions through norm-based or attention-aware pooling must explicitly analyze the trade-off between foreground preservation and background context suppression. This principle is critical because video understanding tasks—particularly relational reasoning and holistic scene comprehension—often depend on subtle background cues that may exhibit low token norms. Reviewers should require fine-grained performance breakdowns by question category and empirical evidence that background compression does not degrade tasks requiring contextual awareness. The spatial pooling mechanism should be shown to compress rather than discard background information, for instance by operating within local windows rather than globally. Without this analysis, a method risks being silently detrimental to complex reasoning tasks despite aggregate score improvements, undermining its reliability as a plug-and-play enhancement.

**Core Evaluation Criteria:**
- **Task-Type Granular Breakdown:** Does the paper provide disaggregated results by question category (e.g., relational reasoning, perception, temporal ordering) to detect where saliency-based compression helps or harms?
- **Background Preservation Mechanism:** Is it demonstrated that the pooling strategy retains representative background features via local operations rather than globally suppressing low-norm regions?
- **Empirical Trade-off Acknowledgment:** Does the work honestly identify scenarios where the method underperforms relative to baselines, and attribute them to the loss of low-norm contextual information?
- **Qualitative and Failure Case Analysis:** Are qualitative examples or failure modes presented to validate that critical background interactions are not erased by the downsampling process?