**Principle 1: Cross-Paradigm Baseline Completeness and Fair Comparative Evaluation for Visual Token Compression**

**Definition:**  
In the field of multimodal large language model efficiency, token compression methods span diverse technical paradigms including clustering-based merging, pruning-merge hybrids, training-free selection, and supervised fine-tuning-based aggregation. A rigorous evaluation must situate the proposed method against representative state-of-the-art approaches from each paradigm under identical compression ratios and model configurations. This principle ensures that claimed improvements are not artifacts of an impoverished baseline landscape or unfair experimental conditions, but rather reflect genuine advances in preserving semantic and positional fidelity during compression. Because different paradigms optimize distinct objectives—such as similarity-based merging versus importance-based pruning—omitting key categories can falsely inflate reported gains. Fairness further demands that comparisons control for backbone architecture, model scale, input resolution, and whether the method operates in a training-free or fine-tuned regime. Without such comprehensive and controlled benchmarking, reviewers cannot disentangle the intrinsic contribution of the proposed operator from confounding factors introduced by weaker or mismatched baselines.

**Core Evaluation Criteria:**
- **Baseline Paradigm Coverage**: Does the work compare against training-free baselines (e.g., ToMe, PACT, TokenCarve, FastV, VisionZip) and, where applicable, SFT-based methods (e.g., Chat-UniVi) under the same reduction ratio?
- **Fairness of Experimental Conditions**: Are comparisons conducted with matched model checkpoints, input resolutions, and fine-tuning data, avoiding conflation of the proposed operator’s benefits with those of additional training or architectural changes?
- **Ablation Rigor**: Is there ablation against the strongest available dense baseline and the native compression method without the proposed enhancement to isolate the operator’s marginal contribution?
- **Cross-Paradigm Generality**: Does the evaluation cover both pure merging and prune-merge hybrid strategies to validate the claimed generality across fundamentally different compression paradigms?

---

**Principle 2: Task-Specific Validation on Spatial, Layout, and Temporal Grounding Under Aggressive Visual Token Compression**

**Definition:**  
The primary motivation for positional preservation in token compression is to maintain fine-grained spatial, layout, and temporal structure that generic visual question answering benchmarks often fail to probe. Evaluation restricted to general VQA or holistic accuracy metrics cannot validate whether compressed tokens retain meaningful positional cues necessary for OCR, document understanding, chart reasoning, and video temporal ordering. This principle demands targeted benchmarking on position-sensitive tasks where the loss of spatial or temporal coherence directly degrades performance, thereby providing discriminative evidence that the method preserves structurally critical information. Aggressive compression inherently risks collapsing fine-grained spatial relationships; therefore, benchmarks like DocVQA, OCRBench, and ChartQA serve as rigorous stress tests for layout fidelity. Similarly, video temporal reasoning benchmarks are essential to verify that multi-frame positional continuity survives spatiotemporal token reduction. Only by demonstrating disproportionate gains on these structurally demanding tasks—relative to generic recognition benchmarks—can authors substantiate that their mechanism genuinely preserves positional integrity rather than merely recovering coarse semantics.

**Core Evaluation Criteria:**
- **Layout and OCR Benchmarking**: Are layout-sensitive and OCR-intensive benchmarks (e.g., DocVQA, OCRBench, ChartQA, TextVQA) included alongside general benchmarks (e.g., MMBench) to isolate positional preservation benefits?
- **Temporal Reasoning Evaluation**: For video inputs, are temporal reasoning benchmarks (e.g., VideoMME, NeXT-QA) evaluated under spatiotemporal compression to verify continuity preservation across frames?
- **Correlation with Positional Complexity**: Does the analysis demonstrate performance deltas that correlate with positional complexity, showing larger gains on tasks requiring fine-grained spatial grounding versus generic semantic recognition?
- **Failure Case Analysis**: Are failure cases analyzed specifically on layout and temporal tasks to identify the limits of positional recovery under extreme compression ratios?

---

**Principle 3: Cross-Architecture Validity of Positional Correspondence and Encoding Scheme Compatibility**

**Definition:**  
Visual token compression operates after complex vision encoder processing, such as ViT patch merging, tile-based encoding, and cross-attention pooling, which may disrupt direct pixel-to-token spatial correspondence. A positional preservation method must demonstrate validity across diverse positional encoding schemes—including 1D RoPE, 2D RoPE, 3D M-RoPE, ALiBi, and learned absolute embeddings—and across model families such as QwenVL, LLaVA, and InternVL. This principle evaluates whether the method's assumptions about positional IDs remain meaningful after encoder transformation and whether its benefits generalize beyond a single architecture or encoding scheme. Many vision encoders reshape or pool spatial grids before the language model, so claims about preserving spatial structure must be validated under these transformed representations rather than idealized pixel grids. Furthermore, model scale can interact with compression artifacts, making evaluation across scales critical to rule out capacity-dependent recovery effects. Without cross-architecture validation, a method risks being an overfit solution to a specific backbone's positional embedding format or encoder topology.

**Core Evaluation Criteria:**
- **Backbone Diversity**: Does the work validate positional preservation on multiple model families with different encoder designs, including those with tile-based or thumbnail-based preprocessing?
- **Model Scale Scalability**: Is the method tested across model scales (e.g., 0.5B, 3B, 7B+) to rule out size-specific artifacts and confirm scalability of the proposed mechanism?
- **Positional Encoding Scope**: Does the paper explicitly address compatibility with non-RoPE positional schemes, or clearly scope its applicability to rotary-based embeddings with a justified theoretical rationale?
- **Correspondence Validation**: Is the assumption that compressed tokens retain meaningful positional embeddings after vision encoding empirically or theoretically validated, rather than merely asserted as a pixel-level mapping?

---

**Principle 4: Mechanistic Interpretability and Frequency-Aware Theoretical Justification for Positional Embedding Redistribution**

**Definition:**  
Parameter-free operators that redistribute existing embedding dimensions require rigorous theoretical grounding to avoid being dismissed as empirical tricks or unexplained engineering tweaks. In multimodal LLMs utilizing frequency-decomposed positional encodings such as M-RoPE, the manner in which dimensions are partitioned and assigned to spatial or temporal axes directly impacts long-range modeling and local spatial precision. This principle demands that the proposed redistribution strategy—such as the choice of group number K—be justified by the structural properties of the underlying positional encoding, for instance via greatest common divisor analysis of frequency bands. Authors must provide mechanistic evidence, including attention entropy, variance metrics, and retained positional ID ratios, demonstrating that the operator actually increases positional information density rather than merely improving end-task accuracy through incidental correlation. Ablation studies should trace interpretable trends across multiple hyperparameter values, linking specific frequency band allocations to observable changes in spatial or temporal attention behavior. Establishing this causal chain from embedding redistribution to internal attention dynamics and finally to downstream performance is essential to distinguish principled mechanism design from opaque performance hacking.

**Core Evaluation Criteria:**
- **Theoretical Grounding of Hyperparameters**: Is the hyperparameter selection (e.g., K) derived from principled analysis of the positional encoding's frequency decomposition, rather than arbitrary tuning or grid search?
- **Mechanistic Evidence**: Does the work provide mechanistic evidence—such as attention entropy, attention variance, and retained positional ID ratios—demonstrating that the operator preserves positional information density rather than merely improving end-task accuracy?
- **Interpretable Ablations**: Are ablation studies conducted across multiple K values or grouping strategies with interpretable trends, linking specific frequency band allocations to spatial or temporal modeling quality?
- **Causal Chain Clarity**: Does the work avoid conflating end-task performance gains with positional preservation, instead establishing a causal chain from embedding redistribution to attention behavior to downstream accuracy?

---

**Principle 5: End-to-End Efficiency Profiling and Plug-and-Play Reproducibility for Parameter-Free Compression Operators**

**Definition:**  
Claims of "plug-and-play" and "parameter-free" operation carry specific methodological obligations that extend beyond accuracy improvements. The proposed operator must be shown to integrate without architectural modification, without introducing significant computational overhead, and with clear delineation of when auxiliary training is required by the host compression method versus the operator itself. Reviewers must evaluate whether efficiency gains are quantified beyond token count reduction, including FLOPs, memory footprint, and inference latency, because sequence length alone does not fully determine hardware-level efficiency. Clarity in experimental setup is equally critical: authors must explicitly state whether reported results rely on supervised fine-tuning of the entire model, partial fine-tuning, or purely inference-time application to pre-trained checkpoints. Conflating the operator's contribution with the host method's training requirements can misleadingly inflate both the apparent generality and the practical utility of the proposed technique. Consequently, rigorous reproducibility demands fair comparisons where the operator is validated on official pre-trained models under identical training and inference conditions as the baselines.

**Core Evaluation Criteria:**
- **Quantitative Efficiency Metrics**: Does the paper quantitatively report end-to-end inference efficiency (FLOPs reduction in attention modules, peak GPU memory, latency/throughput) rather than inferring efficiency solely from sequence length reduction?
- **Training Requirement Delineation**: Is the distinction between the operator's training requirements and the host compression method's requirements clearly articulated (e.g., training-free for PACT/ToMe but requiring SFT for Chat-UniVi)?
- **Plug-and-Play Validation**: Are plug-and-play claims validated by applying the operator to official pre-trained models without any additional training, and are these results fairly compared against baselines under identical conditions?
- **Experimental Reproducibility**: Is the experimental setup fully reproducible, with clear specification of which components are fine-tuned, which checkpoints are used, and how compression ratios are calculated across all compared methods?