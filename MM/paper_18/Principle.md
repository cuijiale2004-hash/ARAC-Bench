Principle 1: Qualitative Interpretability and Diagnostic Error Attribution in Dual-Path Spatiotemporal Token Compression

**Definition:**  
In training-free video token compression for multimodal large language models, the opacity of selection mechanisms poses a significant barrier to trust and iterative improvement. Reviewers consistently demand qualitative visualizations that explicitly reveal which tokens are preserved or discarded by each path of a dual-path framework, as purely quantitative metrics cannot expose whether static backgrounds or dynamic events are being correctly identified. A rigorous evaluation must require authors to establish clear error attribution schemes that classify failures into distinct categories—such as missed events due to conservative difference thresholds, inefficient redundancy compression exhausting the token budget, or inherent reasoning limitations of the backbone model. This diagnostic transparency is essential because it distinguishes between information loss caused by the compression strategy and errors stemming from the LLM's own hallucination or insufficient reasoning capability. Without such granular analysis, it is impossible to determine whether performance improvements result from better denoising or from genuine preservation of semantically critical tokens. Furthermore, explicit boundary analysis must identify scenarios where the method is ill-equipped, such as fine-grained OCR versus global event reasoning, ensuring that claims of general efficacy are appropriately qualified. Ultimately, interpretability and error attribution serve as the foundation for establishing causal relationships between compression design choices and downstream task performance.

**Core Evaluation Criteria:**
- **Token-level visualization requirements**: Are qualitative figures provided showing representative tokens (e.g., community centroids) and event tokens (e.g., temporal difference triggers) overlaid on source video?
- **Failure mode taxonomy**: Does the work systematically categorize failure cases into module-specific errors (e.g., DETS misses vs. SRTS over-retention) versus backbone model errors?
- **Boundary condition specification**: Are the operational limits explicitly defined, such as performance degradation on fine-grained static details versus long-form temporal reasoning?
- **Causal reasoning validation**: Does the analysis avoid conflating correlation (overall accuracy) with causation (specific token retention causing correct answers)?

---

Principle 2: Distribution Alignment and Semantic Integrity Preservation in Training-Free Sparse Inference

**Definition:**  
Training-free compression methods introduce a fundamental risk of distribution shift, where models pretrained on dense visual token sequences must infer from severely pruned inputs at test time. This misalignment between dense training and sparse inference can degrade model behavior unless the compression strategy explicitly preserves the semantic skeleton and temporal coherence that the LLM expects. Evaluators must assess whether authors quantify this misalignment penalty through controlled experiments, such as supervised fine-tuning on compressed tokens to isolate the gap between training-free and distribution-aligned performance. The core concern is whether the selected token subset maintains semantic density comparable to the original sequence, rather than merely reducing token count through random or uniform pruning. A principled method should demonstrate that retained cluster centers and high-difference nodes collectively reconstruct the essential information distribution required by frozen pretrained weights. Reviewers place significant weight on evidence that modern LVLMs' inherent robustness to variable sequence lengths is being leveraged effectively rather than exploited as a crutch for poor compression design. Additionally, the phenomenon where compressed inputs occasionally exceed full-video performance must be rigorously analyzed as a denoising effect rather than reported as an unexamined curiosity. Ensuring semantic integrity is therefore paramount for validating that efficiency gains do not come at the cost of catastrophic information loss.

**Core Evaluation Criteria:**
- **Misalignment quantification**: Are experiments conducted fine-tuning the backbone on compressed tokens to measure the exact penalty of dense-train/sparse-test mismatch?
- **Semantic skeleton argument**: Does the work theoretically and empirically justify that retained tokens preserve the complete semantic narrative (static representatives + dynamic events)?
- **Denoising vs. information loss analysis**: When compressed performance exceeds full-video baselines, is this attributed to noise reduction with supporting evidence rather than anecdotal reporting?
- **Continuity preservation**: Does the method maintain temporal and positional indices to ensure structural alignment with the LLM's pretrained expectations?

---

Principle 3: Algorithmic Scalability and Complexity-Justified Graph Design for Long-Context Video Inputs

**Definition:**  
For video token compression to be practical in real-world multimodal systems, the compression algorithm itself must remain computationally feasible as input scales to hundreds of frames or ultra-high resolutions. Reviewers emphasize that graph-based methods must provide rigorous complexity analysis demonstrating linear or near-linear scaling relative to the quadratic cost of self-attention, particularly when processing tens of thousands of tokens. The choice of specific algorithms—such as connected components versus modularity-optimizing methods like Louvain—must be justified through empirical trade-off studies measuring inference time, GPU memory, and accuracy simultaneously. A method claiming efficiency cannot ignore the overhead of its own graph construction and traversal, especially for long-range temporal edges or dense spatial connections. Furthermore, scalability validation must extend beyond theoretical big-O notation to include empirical tests at extended sequence lengths, such as 128 frames or beyond, to confirm real-world viability. The framework must also articulate its upper limits regarding input resolution and duration, addressing whether memory constraints in the visual encoder become the bottleneck before the compression stage. Ultimately, algorithmic efficiency must be scrutinized as a first-class contribution, not merely assumed as a byproduct of token reduction.

**Core Evaluation Criteria:**
- **Theoretical complexity rigor**: Is the complete computational complexity of graph construction, community detection, and difference analysis formally analyzed and compared against the O(N²d) attention baseline?
- **Empirical scalability validation**: Are experiments conducted at varying sequence lengths (e.g., 64 vs. 128 frames) to demonstrate maintained or improved efficiency gains as inputs grow?
- **Algorithm selection justification**: Are choices like connected components over Louvain methodologically defended with empirical measurements of speed, memory, and accuracy trade-offs?
- **Hardware-aware bottleneck analysis**: Does the work identify the practical upper limits of token count, resolution, and duration where the method remains viable before encoder memory constraints dominate?

---

Principle 4: Cross-Architecture Baseline Consistency and Granular Ablation of Spatiotemporal Components

**Definition:**  
Rigorous evaluation of video token compression requires validation across multiple backbone architectures and meticulous attention to experimental conditions that could confound performance comparisons. Reviewers consistently flag discrepancies in baseline performance that arise from unequal frame budgets, evaluation protocols, or implementation details, demanding that authors reconcile such gaps through controlled extended experiments. Beyond mere aggregate scores, a high-quality study must decompose its spatiotemporal graph framework into isolated spatial, temporal, and joint components to prove that unified modeling genuinely outperforms isolated alternatives. This includes systematic ablation of similarity thresholds, difference thresholds, and retention ratios to demonstrate robustness across hyperparameter choices. Cross-model validation on architecturally distinct systems—such as LLaVA-Video, NVILA, and Qwen2.5-VL—is essential for establishing generalizability rather than overfitting to a specific visual encoder or alignment module. Furthermore, when baseline reports from official technical documents diverge from reproduced results, authors must transparently explain protocol differences rather than dismiss them. The depth and honesty of ablation studies and baseline alignments directly reflect the methodological maturity of the work.

**Core Evaluation Criteria:**
- **Baseline alignment and protocol transparency**: Are reported baseline discrepancies explicitly explained and validated through matching frame budgets, input resolutions, and evaluation settings?
- **Modular ablation depth**: Are spatial-only, temporal-only, and spatiotemporal-joint similarity strategies separately evaluated, followed by the additive contribution of the difference module?
- **Hyperparameter robustness**: Is the sensitivity of key thresholds (e.g., similarity and difference cutoffs) analyzed across a meaningful range with performance curves?
- **Cross-model generalization**: Does the work validate performance and efficiency claims on at least three distinct base architectures with varying visual encoders and LLM backbones?

---

Principle 5: Query-Agnostic Design and Dynamic Content Adaptability in Static-Dynamic Video Compression

**Definition:**  
A critical design dimension in video token compression for MLLMs is whether the method functions as a general-purpose visual compressor or requires query-conditioned reprocessing, with significant implications for multi-turn dialogue efficiency. Reviewers evaluate whether query-agnostic compression preserves sufficient information to answer diverse potential questions without requiring repeated full-sequence scanning for each new prompt. This necessitates explicit analysis of how the method adapts implicitly to varying video dynamics, such as allocating more tokens to abrupt events during action sequences while compressing stable backgrounds during static scenes. The framework must also address edge cases like slowly changing scenes, where adjacent-frame differences are minimal but cumulative drift is semantically significant, requiring mechanisms such as dynamic anchor updates. Authors must justify why a fixed unified pruning strategy is sufficient across different video types, or alternatively, demonstrate adaptive weighting between similarity-based and difference-based paths. Furthermore, the trade-off between global event reasoning and fine-grained spatial perception must be characterized, acknowledging that aggressive redundancy compression may destroy high-frequency static details like text or small objects. A principled evaluation therefore demands evidence that the compression strategy intelligently balances static and dynamic information preservation without task-specific reconfiguration.

**Core Evaluation Criteria:**
- **Query-agnostic sufficiency**: Is the method justified as a general-purpose compressor, and does it demonstrate retained performance across diverse query types without per-query reprocessing?
- **Implicit dynamic weighting**: Does the framework adapt token allocation between static and dynamic content based on video statistics (e.g., automatic expansion of event tokens during high activity), and is this behavior measured?
- **Slow-drift handling**: Are slowly changing scenes explicitly addressed with mechanisms beyond adjacent-frame comparison, such as dynamic anchor maintenance or cumulative difference tracking?
- **Static-dynamic trade-off characterization**: Does the work honestly delineate performance boundaries between global temporal reasoning tasks and fine-grained spatial detail tasks (e.g., OCR)?