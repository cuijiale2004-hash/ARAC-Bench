**Principle 1: Comprehensive End-to-End Latency and Hardware-Aware Efficiency Validation for Training-Free MLLM Acceleration**

**Definition:**  
Training-free acceleration methods must be evaluated through realistic, deployment-aligned efficiency metrics rather than isolated theoretical measures such as FLOPs reduction or ambiguously labeled CUDA time. Reviewers scrutinize whether latency improvements are validated via end-to-end wall-clock time, prefill latency, per-step decoding time, and memory footprint on clearly specified hardware, while disaggregating contributions from the vision encoder and LLM decoder. A method must transparently report whether auxiliary computations—such as token clustering, cross-modal similarity calculation, or adaptive weighting—are executed on GPU or CPU and whether they introduce hidden overhead that negates theoretical speedups. Furthermore, robust studies should present component-wise latency breakdowns to verify that plug-and-play modules achieve net acceleration when integrated into the full inference pipeline. Without such comprehensive hardware-aware profiling, claims of practical efficiency gains remain insufficiently grounded for real-world MLLM deployment.

**Core Evaluation Criteria:**
- **Multi-dimensional latency profiling**: Does the work report end-to-end inference time, prefill latency, decoding latency, and throughput on clearly specified hardware?
- **Overhead transparency**: Are auxiliary computations (clustering, similarity matrices, adaptive weighting) profiled individually and disclosed as GPU or CPU operations?
- **Component-wise speedup validation**: Is the net acceleration of each plug-in module isolated to ensure that combined modules do not mask hidden bottlenecks?
- **Memory and cache analysis**: Does the evaluation include KV-cache reduction and peak memory usage alongside computational metrics?

---

**Principle 2: Cross-Stage Pipeline Coherence and Stage-Wise Trade-off Ablation in Unified Vision Token Compression**

**Definition:**  
For unified frameworks that compress visual tokens across multiple pipeline stages—such as both the vision encoder and the LLM decoder—reviewers demand rigorous evidence that the multi-stage integration is architecturally coherent and empirically superior to isolated stage-wise pruning. This requires explicit ablation of insertion layers, token retention ratios allocated to each stage, and performance-efficiency Pareto fronts when the compression budget is reallocated between stages. Authors must demonstrate that unification yields complementary benefits rather than simply concatenating two independent methods, and they should justify why specific layers are chosen for pruning based on empirical sensitivity analyses. The configuration details, including whether modules are inserted shallow or deep, must be thoroughly disclosed to ensure reproducibility and to prevent opaque black-box designs. Ultimately, the work must prove that cross-stage coordination avoids error propagation and achieves a better accuracy-efficiency trade-off than single-stage alternatives.

**Core Evaluation Criteria:**
- **Stage-isolated ablation**: Are compression modules evaluated independently in the vision encoder, independently in the LLM decoder, and jointly, with performance-efficiency trade-offs reported for each?
- **Insertion layer sensitivity**: Does the study empirically justify the chosen insertion layers and analyze performance degradation when pruning occurs at deeper or shallower layers?
- **Token budget allocation analysis**: Is there a systematic study of how retention ratios are split between pipeline stages (e.g., encoder vs. decoder) and whether the allocation is adaptive or fixed?
- **Unified vs. isolated superiority**: Does the work fairly compare the unified framework against competitive single-stage baselines under identical overall compression ratios to prove synergy?

---

**Principle 3: Robustness and Prompt-Sensitivity Analysis for Text-Guided Visual Token Reduction Mechanisms**

**Definition:**  
When textual cues guide visual token selection, merging, or complementation, the method must be rigorously evaluated for robustness across diverse textual conditions, including the absence of prompts, noisy or irrelevant queries, and varying task-specific instructions per image. Reviewers expect analysis of whether text-dependency introduces hallucinations, disrupts multi-round conversational consistency, or forces expensive per-prompt recomputation that undermines efficiency gains. A well-designed method should gracefully degrade to text-agnostic behavior when textual signals are unreliable, leveraging visual redundancy to preserve semantic integrity through unsupervised clustering rather than erroneous text-driven selection. Furthermore, the computational and memory implications of conditioning on text must be quantified, particularly when the same image is processed with different prompts in multi-turn or batched settings. The core criterion is whether text guidance genuinely enhances cross-modal alignment or merely creates fragile, context-dependent performance spikes.

**Core Evaluation Criteria:**
- **Text-agnostic fallback evaluation**: Does the method maintain acceptable performance when textual prompts are entirely absent, relying solely on visual signals?
- **Noise and relevance robustness**: Is the method tested under irrelevant, misleading, or adversarial textual prompts to verify it avoids hallucinations and catastrophic misalignment?
- **Prompt-variability overhead**: When the same image is queried with different prompts, does the method require full visual reprocessing, and is this latency cost quantified?
- **Comparative safety against direct guidance**: Is the two-stage or merged-token design contrasted with direct text-guided pruning methods that may over-weight textual attention?

---

**Principle 4: Mechanistic Justification and Causal Grounding of Heuristic Token Importance Proxies in LLM Decoding**

**Definition:**  
Heuristic proxies for token importance—such as the attention distribution of the first generated token, [CLS] token attention, or local affinity scores—require deep mechanistic justification beyond post-hoc empirical correlations or convenience-driven arguments. Reviewers assess whether authors establish a causal link between the chosen proxy and semantic significance through layer-wise attention visualizations, redundancy analyses, or controlled ablations that isolate the proxy's contribution to final performance. Justifying a proxy solely on the grounds of computational efficiency, such as compressing tokens after the first decoding step to reduce subsequent costs, is insufficient if the proxy lacks empirical grounding for representing global or local information faithfully. The work must clearly distinguish between performance improvements arising from the proxy's semantic validity versus incidental architectural interactions, and it should explore alternative ensemble strategies to demonstrate robustness. High-quality research in this domain should strive to demystify why a particular importance indicator works, transforming opaque heuristics into interpretable, reliable selection mechanisms.

**Core Evaluation Criteria:**
- **Causal mechanistic evidence**: Does the work provide attention visualizations, layer-wise redundancy curves, or feature analyses showing the proxy correlates with semantically important regions?
- **Ablation of proxy components**: Are global importance, local affinity, and ensemble strategies isolated to demonstrate that each contributes non-redundant signal?
- **Rejection of efficiency-only justification**: Does the paper ground its proxy choice in empirical semantic validity rather than solely arguing from computational convenience?
- **Ensemble robustness**: Are alternative aggregation strategies (fixed weights, geometric mean, maximum) compared to prove the chosen method is robustly superior?

---

**Principle 5: Instance-Level Failure Diagnosis and Knowledge Boundary Drift Assessment Beyond Aggregate Benchmarks**

**Definition:**  
Aggregate benchmark accuracy is insufficient for validating compression methods because token pruning can induce knowledge boundary drift, wherein individual instances flip from correct to incorrect answers without affecting average scores substantially. Reviewers expect granular diagnostic analyses that track instance-level performance changes, visualize retained and discarded tokens via attention maps or patch masks, and explicitly characterize failure modes such as OCR degradation, object hallucination, or loss of fine-grained spatial details. The work must identify which semantic categories or visual contexts are most vulnerable to compression artifacts and whether complementary modules can recover critical information that dominant-token selection alone would discard. Qualitative case studies should be paired with quantitative stability metrics, such as the percentage of flipped answers or consistency rates across compression ratios, to prove reliability beyond mean accuracy. Ultimately, a rigorous evaluation demonstrates that the method preserves not only population-level performance but also instance-level trustworthiness required for real-world deployment.

**Core Evaluation Criteria:**
- **Instance-level flip tracking**: Does the evaluation report the rate of answer flips (correct→incorrect) induced by compression relative to the uncompressed baseline?
- **Qualitative token visualization**: Are attention maps, retained patch visualizations, or discard masks provided to diagnose what visual information is preserved or lost?
- **Failure mode taxonomy**: Does the work categorize compression-induced errors (e.g., text-region loss, object merging, color misidentification) across benchmark tasks?
- **Boundary drift mitigation**: If the method includes complement modules, is there explicit evidence that they recover critical tokens and reduce instance-level divergence compared to pruning alone?