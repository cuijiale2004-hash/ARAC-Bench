**Principle 1: Discrimination Between Incremental Engineering and Fundamental Architectural Innovation in AR-to-Block-Diffusion LLM Adaptation**

**Definition:**  
In the subfield of adapting pretrained autoregressive LLMs into diffusion-style decoders, reviewers must rigorously evaluate whether proposed contributions represent genuine architectural or algorithmic innovations versus skillful combinations of existing components. This principle addresses the critical challenge of distinguishing between novel training recipes, attention mechanisms, or decoding strategies that unlock new capabilities, and incremental engineering that merely assembles prior techniques without resolving new bottlenecks. The evaluation must assess whether the work introduces orthogonal innovations beyond prior iterations and concurrent works, whether ablation studies isolate the contribution of each component, and whether the claimed improvements stem from the proposed method itself rather than from architectural choices that any block-wise formulation would inherently enjoy. Given the rapid proliferation of concurrent work in this space, reviewers must verify that the contribution is not simply a repackaging of existing ideas under a new system.

**Core Evaluation Criteria:**
- **Component Orthogonality**: Does the work clearly decompose its contributions and demonstrate that each proposed component (e.g., complementary masking, hierarchical caching, sub-block decoding) addresses a distinct bottleneck not solved by prior work such as Fast-dLLM v1 or Block Diffusion?
- **Ablation Rigor**: Are comprehensive ablation studies provided that isolate the impact of each proposed technique, rather than only comparing against coarse end-to-end baselines?
- **Distinction from Concurrent/Prior Work**: Does the work establish clear technical distinctions from closely related concurrent works (e.g., SDAR) and prior versions, supported by both conceptual analysis and empirical comparisons?
- **Attribution of Gains**: Are performance and efficiency gains properly attributed to the proposed method rather than to inherent properties of the target architecture (e.g., block-wise vs. full-attention diffusion)?

---

**Principle 2: Training-Inference Block Granularity Co-Design and Hierarchical Cache Necessity in Block Diffusion LLMs**

**Definition:**  
For block diffusion language models, a core evaluation criterion is whether the proposed method maintains consistency between training-time block structures and inference-time decoding granularity, and whether it provides principled mechanisms to handle mismatches without retraining. Block-wise diffusion introduces a fundamental tension: training assumes fixed block sizes, but practical deployment benefits from flexible, fine-grained decoding. Reviewers must assess whether the work explicitly addresses this mismatch through mechanisms like sub-block decoding, whether the hierarchical caching architecture is co-designed with the decoding schedule to support variable granularity, and whether the necessity of multi-level caching (block-level exact cache and sub-block-level approximate cache) is empirically validated rather than assumed. This principle ensures that diffusion LLMs are evaluated not just on throughput metrics but on their practical deployability across diverse sequence lengths and hardware constraints.

**Core Evaluation Criteria:**
- **Mismatch Resolution**: Does the work explicitly identify and resolve the training-inference block size mismatch, and are the proposed mechanisms (e.g., sub-block decoding) shown to be necessary through ablations comparing against naive block-mismatch inference?
- **Cache Necessity and Design**: Is the hierarchical caching mechanism justified with both theoretical reasoning and empirical evidence? Does it outperform simpler alternatives (e.g., block-only cache, full recomputation, or DualCache applied without sub-block structure)?
- **Decoding Schedule Clarity**: Is the sub-block decoding schedule (e.g., left-to-right, confidence-based) clearly specified, and does it preserve bidirectional attention benefits while maintaining autoregressive consistency across blocks?
- **Scalability Validation**: Does the method demonstrate stable latency and throughput scaling across varying context lengths (e.g., 1K to 8K tokens) and batch sizes, particularly under compute-bound regimes?

---

**Principle 3: Attribution Analysis of Data Efficiency Claims in Low-Budget Post-Training Adaptation to Diffusion LLMs**

**Definition:**  
A central claim in AR-to-diffusion adaptation research is the dramatic reduction in required training data compared to training diffusion models from scratch. Reviewers must critically evaluate whether reported data efficiency stems from the proposed training recipe (e.g., complementary masking, token shift, padding strategies) or from inherent properties of the target architecture (e.g., block-wise attention being more AR-compatible than full-attention diffusion). This principle demands that authors disentangle architectural advantages from algorithmic innovations through controlled ablations, and that they validate adaptation quality under realistic data budgets. The evaluation should ensure that data efficiency claims are not artifacts of comparing against mismatched baselines (e.g., full-sequence diffusion retraining vs. block diffusion fine-tuning) and that the adaptation preserves pretrained model capabilities without catastrophic forgetting or domain-specific degradation.

**Core Evaluation Criteria:**
- **Disentanglement of Factors**: Are ablation studies provided that separate the contribution of the block-wise architecture from the proposed training recipe components (masking strategy, label shifting, padding schemes)?
- **Fair Baseline Comparison**: Are comparisons against prior diffusion models controlled for architectural differences, or does the work acknowledge when efficiency gains derive from structural choices rather than novel algorithms?
- **Adaptation Fidelity**: Does the adapted model preserve the original AR model's performance across diverse benchmarks, and are domain-specific trade-offs (e.g., math vs. code performance changes) thoroughly analyzed rather than masked by aggregate scores?
- **Data Budget and Quality Analysis**: Is the adaptation validated with data budgets that are practically meaningful, and is the quality of training data (e.g., instruction-tuning vs. pretraining corpora) justified and its impact analyzed?

---

**Principle 4: Task-Specific Performance Trade-off Characterization Under Block-Wise Bidirectional Decoding**

**Definition:**  
Block diffusion models introduce bidirectional context modeling within blocks, which can fundamentally alter the model's inductive biases compared to strict left-to-right autoregressive generation. Reviewers must evaluate whether the work provides mechanistic insights into which task types benefit from or are harmed by block-wise decoding, and whether these trade-offs are characterized beyond aggregate metrics. This principle requires authors to explain structural reasons for performance discrepancies (e.g., code benefiting from block boundaries and string completion, while math suffers from premature answer leakage through bidirectional attention or shortcut bias), and to validate these hypotheses with empirical evidence. The evaluation ensures that block diffusion methods are assessed holistically rather than solely on average scores, and that practitioners understand deployment risks for specific applications requiring strict logical chains or precise step-wise derivation.

**Core Evaluation Criteria:**
- **Trade-off Identification**: Does the work report performance changes across diverse task categories (code, math, reasoning, knowledge) rather than relying solely on aggregate scores that might hide systematic degradation?
- **Mechanistic Explanation**: Are hypotheses provided for why certain tasks improve or degrade, supported by qualitative analysis or references to structural priors (e.g., indentation boundaries in code vs. symbol-level logical chains in math)?
- **Robustness Across Scales**: Are observed trade-offs consistent across model scales, and do they indicate systematic biases inherent to the decoding paradigm rather than random fluctuations?
- **Mitigation and Communication**: Does the work propose or analyze potential mitigations for identified weaknesses, or at minimum clearly communicate deployment recommendations based on task type?

---

**Principle 5: Cross-Architecture Generalization and Production Deployability of Diffusion LLM Inference Acceleration**

**Definition:**  
For inference acceleration research on large language models, theoretical speedups must be validated under realistic deployment conditions, including varying batch sizes, context lengths, hardware platforms, and base model architectures. Reviewers must assess whether the proposed system is evaluated beyond a single model family or scale, whether implementation complexity is justified by practical gains, and whether the method integrates cleanly with existing serving frameworks. This principle ensures that efficiency claims are reproducible in production environments and that the method's benefits are not confined to narrow, favorable configurations. Cross-architecture validation (e.g., Qwen vs. LLaMA) is particularly important to confirm that acceleration techniques are not exploiting architecture-specific artifacts, while long-context and large-batch evaluations verify that caching mechanisms remain effective under memory and compute pressure.

**Core Evaluation Criteria:**
- **Hardware and Batch Scalability**: Are throughput and latency reported across multiple batch sizes and GPU types (e.g., A100, H100), and do speedups hold under both memory-bound and compute-bound regimes?
- **Architecture Generalization**: Is the method validated on multiple pretrained AR architectures beyond the primary model family, demonstrating that gains are not architecture-specific?
- **Integration Feasibility**: Does the work discuss implementation complexity, compatibility with standard KV cache abstractions, and integration with production inference frameworks (e.g., vLLM, SGLang)?
- **Long-Context Stability**: Does the method maintain efficiency advantages as context length increases, and are caching mechanisms validated to avoid approximation errors that compound over long sequences?