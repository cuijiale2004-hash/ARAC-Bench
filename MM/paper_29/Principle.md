**Principle 1: Mechanistic Attribution and Causal Justification of Generative-to-Discriminative Architectural Adaptations in MLLM Embedding Learning**

**Definition:**  
When adapting autoregressive, decoder-only multimodal large language models (MLLMs) to discriminative embedding-based retrieval, reviewers expect authors to move beyond empirical performance tables and provide mechanistic explanations for why specific architectural modifications remedy the inherent mismatch between next-token prediction objectives and holistic sequence encoding. The work must establish a causal chain from design choice to representation quality, demonstrating how each modification alters information flow or mitigates biases—such as recency bias in last-token pooling or calculation bias introduced by instruction tokens—rather than treating findings as isolated black-box observations. This principle recognizes that porting techniques from pure LLMs to multimodal retrieval contexts is non-trivial and requires domain-specific causal validation.

**Core Evaluation Criteria:**
- **Causal Mechanism Clarity:** Does the work clearly articulate the mechanistic rationale (e.g., attention-map analysis, bias mitigation theory) linking architectural changes to retrieval performance, rather than merely reporting accuracy deltas?
- **Empirical Disentanglement:** Are architectural choices disentangled from training data effects through controlled ablations that isolate the impact of bidirectional attention, pooling strategy, and instruction handling under identical data regimes?
- **Rejection of Black-Box Reporting:** Does the analysis avoid purely correlational claims and instead validate that performance gains stem from the hypothesized mechanism (e.g., masking removes redundant instruction signal already encoded via self-attention) rather than confounding factors?

---

**Principle 2: Curriculum Design and Stage-Wise Interdependence in Progressive Transition for Cross-Modal Alignment**

**Definition:**  
For MLLM-based universal retrievers, the transition from generative pretraining to discriminative retrieval requires rigorous evaluation of curriculum learning strategies that progressively increase task complexity. Reviewers assess whether staged training—such as text-only retrieval followed by cross-modal alignment and finally instruction-tuned multimodal retrieval—is motivated by identifiable capability gaps, including the misalignment between causal pretraining attention and bidirectional retrieval encoding. The evaluation examines whether each stage produces transferable representations, whether gains are attributable to sequencing strategy rather than mere data accumulation, and whether the curriculum prevents catastrophic forgetting of pretrained capabilities during modality expansion.

**Core Evaluation Criteria:**
- **Stage Necessity and Transfer:** Is each curriculum stage justified by a specific capability deficit (e.g., establishing a semantic text encoder anchor before visual alignment), and does evidence show that earlier stages enable effective learning in later stages?
- **Data-Quantity Disentanglement:** Are gains demonstrated to result from curriculum sequencing rather than simply seeing more data, through ablations or held-out comparisons that isolate strategic ordering from volume effects?
- **Cross-Modal Stability:** Does the progressive strategy maintain training stability when transitioning from unimodal to complex multimodal instruction tasks, with metrics tracking intermediate representation quality and forgetting of prior capabilities?

---

**Principle 3: Hyperparameter Interaction and Noise-Aware Training Stability in Large-Batch Contrastive Learning for MLLMs**

**Definition:**  
Contrastive learning of MLLM embeddings operates under distinct optimization dynamics compared to smaller encoders, requiring careful coordination of batch size, learning rate scaling laws, temperature scheduling, and hard negative mining strategies. Reviewers scrutinize whether the work accounts for interdependencies among these factors—such as the necessity of learnable temperature parameters at large batch sizes or the collapse-inducing effects of unfiltered hard negatives in noisy multimodal datasets—and whether training stability is demonstrated through sensitivity analyses and convergence diagnostics. This principle demands that training recipes be presented as principled, interdependent systems rather than loosely assembled heuristic tricks ported from other domains.

**Core Evaluation Criteria:**
- **Scaling Law Validation:** Are batch size and learning rate scaling rules empirically validated under the specific MLLM embedding regime, with clear evidence of saturation points and instability zones?
- **Temperature and Convergence Rigor:** Is the temperature parameter analyzed as a dynamic, learnable component with documented convergence behavior across training stages, and does it adapt sensibly to batch noise and hard-negative introduction?
- **Hard Negative Quality Control:** Does the work explicitly address false-negative noise through robust filtering mechanisms with justified thresholds, and is the sensitivity of retrieval performance to filtering criteria and mining depth quantified rather than assumed?

---

**Principle 4: Cross-Scale Generalization and Backbone-Agnostic Validation of Retrieval Recipes**

**Definition:**  
Because retrieval findings in MLLMs may exhibit scale-dependent behaviors—particularly regarding instruction following and representation granularity—reviewers expect claims about universal training recipes to be validated across multiple model capacities and, ideally, across architectural families. The principle assesses whether performance gains are intrinsic to the proposed methodological adaptations or artifacts of a specific backbone scale, and whether smaller models can achieve competitive or superior performance relative to larger baselines when equipped with the proposed recipe. Findings derived from a single parameter size are treated as provisional unless their generalizability is theoretically grounded or empirically probed.

**Core Evaluation Criteria:**
- **Multi-Scale Evidence:** Are experiments conducted on at least two distinct parameter scales (e.g., 4B and 7B) to verify that findings are not unique to a single model capacity or instruction-following capability level?
- **Backbone Transfer:** Does the training recipe demonstrate consistent improvements when applied to different MLLM families or generations, indicating that the adaptation principles are architecture-agnostic?
- **Scaling Trend Interpretation:** Does the work honestly acknowledge current scale restrictions and discuss how findings are expected to extrapolate to significantly larger models, avoiding overgeneralization from limited evidence?

---

**Principle 5: Computational Feasibility and Efficiency-Aware Distillation for Recall-Then-Rerank Compression**

**Definition:**  
Distilling a cascaded recall-then-rerank pipeline into a single MLLM embedding model must be evaluated not only by end-task accuracy but also by the computational tractability of the distillation process itself. Reviewers expect rigorous theoretical or empirical characterization of training complexity reduction—such as transforming intractable quadratic teacher-student alignment into linear or near-linear operations—and validation that the distilled model preserves the reranker’s discriminative power without incurring prohibitive latency or memory costs. This principle distinguishes theoretically grounded efficiency innovations that make large-scale MLLM distillation practically feasible from naïve approximations that sacrifice reproducibility or scalability.

**Core Evaluation Criteria:**
- **Complexity Reduction Rigor:** Is the computational complexity of the proposed distillation method formally analyzed (e.g., Big-O reduction from quadratic to linear), and are theoretical speedups validated by proxy empirical measurements or by demonstrating feasible training completion where baselines fail?
- **Feature Diversity Preservation:** Does the distillation strategy maintain or increase feature diversity relative to baselines, and is this explicitly linked to the principled selection of hard-negative subsets rather than full-batch similarity matrices?
- **End-to-End Efficiency-Performance Trade-off:** Is the final single-model system shown to achieve comparable or superior performance to the cascaded teacher pipeline with demonstrably lower inference overhead, and is this trade-off quantified in absolute latency or resource terms?