**Principle 1: Mechanistic Attribution of Cross-Level Feature Collapse Under Compression in MLLM Vision Encoders**

**Definition:**  
This principle evaluates whether the work provides a genuine, mechanistic explanation of how compression artifacts propagate through the vision encoder and disproportionately degrade specific feature levels. Rather than merely describing performance drops at the output layer, the research must dissect the internal information flow—typically characterized as a multi-stage process (e.g., preliminary screening, local extraction, global integration)—and precisely identify which stage or cross-level synthesis is most vulnerable. The critical requirement is to distinguish compression-specific failure modes from well-known generic properties of Vision Transformers (such as the evolution from local to global attention), establishing a causal chain between low-level distortion, disrupted cross-level feature synthesis, and downstream task degradation.

**Core Evaluation Criteria:**
- **Distinction from Generic ViT Properties**: Does the analysis clearly differentiate known vision transformer behaviors from compression-induced vulnerabilities, or does it merely re-characterize existing observations in a new context?
- **Multi-Modal Evidence**: Is the claimed vulnerable stage supported by diverse quantitative and qualitative evidence, such as attention distance curves, PCA feature evolution, token similarity metrics, and information flow visualizations?
- **Causal Task Linkage**: Are specific downstream MLLM task failures (e.g., counting, fine-grained spatial reasoning) mechanistically linked to the identified collapse of cross-level features, rather than inferred solely from end-to-end accuracy changes?
- **Localization Precision**: Does the study precisely localize the failure point within the layer hierarchy (e.g., early Stage 3) and explain why adjacent stages (low-level or coarse high-level) remain comparatively robust?

---

**Principle 2: Semantic Guidance Generalization Across Heterogeneous Vision Encoders and Base Codecs**

**Definition:**  
This principle assesses whether the proposed semantic guidance or importance allocation mechanism is validated as a generalizable prior across diverse MLLM architectures, rather than being an implementation-specific artifact tied to a single encoder family (e.g., CLIP). Since production MLLMs employ heterogeneous vision backbones—ranging from CLIP and SigLIP to InternViT, with varying token designs and attention structures—a credible method must demonstrate codec-agnostic integration and justify the use of off-the-shelf guidance sources when matched-encoder guidance is feasible. The work must resolve architectural dependencies and prove that its bit-allocation logic transfers across base codecs and encoder types without redesign.

**Core Evaluation Criteria:**
- **Cross-Encoder Validation**: Is the guidance mechanism tested on structurally distinct vision encoders (e.g., those with vs. without a [CLS] token, different patch granularities, or divergent attention patterns)?
- **Matched vs. Unmatched Guidance Ablation**: Does the study compare matched-encoder guidance against off-the-shelf proxy guidance (e.g., CLIP attention for SigLIP-based MLLMs) to justify the design choice and quantify the performance gap?
- **Codec-Agnostic Robustness**: Is the framework shown to integrate with multiple base codecs (e.g., ELIC, DCAE, traditional codecs) and consistently yield bitrate savings without retraining the core codec?
- **Dependency Resolution**: Are assumptions about the guidance source explicitly stated, and are failure modes or limitations when transferring across encoder families discussed?

---

**Principle 3: Comprehensive Multi-Level Benchmarking Spanning Low-Level Detail and High-Level Semantics**

**Definition:**  
This principle evaluates whether the experimental validation covers the full spectrum of MLLM perceptual requirements, from low-level structural fidelity (e.g., OCR, attribute recognition, spatial localization) to high-level semantic abstraction (e.g., scene understanding, commonsense reasoning), with explicit attention to cross-level tasks that integrate both. A codec optimized solely for high-level machine perception may destroy fine-grained details, while a human-centric codec may waste bits preserving visually imperceptible pixels that MLLMs ignore. The evaluation must demonstrate balanced performance across task categories and avoid cherry-picking benchmarks where the method excels while masking failures in critical low-level or cross-level capabilities.

**Core Evaluation Criteria:**
- **Task Spectrum Coverage**: Does the benchmark suite include low-level, high-level, and cross-level tasks, ensuring that the codec does not sacrifice fine-grained detail for semantic abstraction or vice versa?
- **Baseline Diversity**: Are comparisons made against both human-centric codecs (which preserve low-level fidelity) and narrow machine-centric codecs (which target high-level features) to expose trade-offs?
- **Consistency Across Categories**: Does the method avoid erratic, task-specific performance collapses (e.g., excellent on scene understanding but catastrophic on counting or text recognition)?
- **Feature-Level Alignment Analysis**: Is there explicit analysis mapping each task category to the feature level it relies upon, and how compression distortion at that level affects performance?

---

**Principle 4: Rigor of Bitrate-Control Protocol and Rate-Distortion Comparison Standards**

**Definition:**  
This principle examines the experimental rigor with which bitrate alignment is controlled when comparing compression methods. Since different compression algorithms produce variable bitrates for a fixed quality parameter, and since auxiliary metadata (e.g., guidance maps) incur overhead, fair comparison demands transparent protocol choices. The research must clarify whether comparisons use fixed quality factors (accepting per-image bitrate variance), fixed target bitrates, or full rate-distortion curves. Furthermore, the bitrate cost of all auxiliary components must be rigorously accounted for in the total rate budget to prevent misleading efficiency claims.

**Core Evaluation Criteria:**
- **Protocol Transparency**: Does the paper explicitly state whether fixed quality parameters or fixed bitrates are used, and is the rationale for this choice justified given the analysis goals?
- **Fair Point-wise Comparison**: When comparing methods at a specific operating point, are bitrates aligned to within a negligible margin, and are quality-parameter sweeps used to generate comparable curves?
- **Metadata Overhead Accounting**: Is the bitrate cost of guidance maps, adapters, or other auxiliary structures fully included in the reported total bitrate and BD-rate calculations?
- **Standardized BD-Rate Methodology**: Are BD-rate savings computed against strong, relevant anchors using standard practices, and do they reflect performance across the entire rate spectrum rather than a single point?

---

**Principle 5: Hierarchical Scalability and Computational Efficiency for High-Resolution Visual Inputs**

**Definition:**  
This principle evaluates whether the proposed compression paradigm scales effectively to high-resolution images and video sequences without introducing prohibitive computational overhead or metadata bloat. As MLLMs increasingly process high-resolution inputs via patch-based pipelines, a guidance mechanism designed for low-resolution fixed grids may become too coarse or too expensive. The method must justify its strategy for handling resolution scaling—such as hierarchical local-global fusion—with ablations on guidance granularity, and it must demonstrate that encoding/decoding latency and auxiliary memory costs remain negligible relative to the base codec. For video, the approach should be compatible with frame-wise or sparse sampling strategies common in video MLLMs.

**Core Evaluation Criteria:**
- **High-Resolution Strategy Validation**: Is there a clearly explained mechanism (e.g., local patch guidance combined with downsampled global attention) to maintain semantic precision at resolutions significantly larger than the training size?
- **Granularity vs. Overhead Tradeoff**: Are guidance map resolutions systematically ablated to justify the chosen granularity against the bitrate and computational overhead of finer maps?
- **Computational Bound**: Are encoding and decoding times reported for high-resolution inputs, demonstrating that auxiliary components (e.g., shallow attention extraction, lightweight adapter) add marginal latency?
- **Video and Long-Sequence Compatibility**: Is the method validated or logically extended to video MLLM scenarios, and does it align with frame-sampling strategies without requiring dense temporal modeling?