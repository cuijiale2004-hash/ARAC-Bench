**Principle 1: Inter-Head Similarity Generalization and Cross-Architecture Empirical Grounding for Long-Context Proxy Attention**

**Definition:**  
In sparse attention methods that leverage cross-head redundancy to reduce estimation cost, the central empirical premise—that multiple heads share token-focus rankings while differing only in sparsity magnitude—must be rigorously validated beyond a single model family or synthetic retrieval setting. This principle is crucial because LLM attention heads may specialize by function (e.g., positional, syntactic, semantic) in short contexts, and assuming universal similarity without architecture-specific ablation can lead to flawed proxy estimation in models with extreme GQA ratios or diverse pre-training. High-quality research must demonstrate that the observed similarity is not an artifact of attention sinks, shared positional biases, or task-specific degeneracy, but rather a robust phenomenon across layers, context lengths, and realistic long-context benchmarks. Without such grounding, proxy-based methods risk performance collapse when transferred to new model scales or downstream tasks that elicit heterogeneous attention patterns. The reviewer must therefore assess whether the authors provide systematic evidence (e.g., overlap metrics, cumulative probability curves, sink-ablation analyses) that justifies the proxy assumption across diverse operational regimes.

**Core Evaluation Criteria:**
- **Cross-Architecture Validation:** Does the work empirically verify head-similarity across multiple model families (e.g., LLaMA, Qwen) with varying GQA configurations, or does it rely on a single architecture?
- **Task-Type Generalization:** Is the similarity phenomenon demonstrated on both synthetic retrieval tasks and realistic long-context understanding benchmarks (e.g., InfiniteBench, LongBench-v2), or is evidence limited to needle-in-haystack settings?
- **Confounder Control:** Are attention sinks and other positional biases explicitly accounted for and ruled out as the primary driver of observed overlap? Are oracle analyses provided to isolate the contribution of shared token rankings?
- **Layer-wise and Depth Analysis:** Does the study disaggregate similarity metrics by layer depth, acknowledging that lower layers may exhibit lower recall or higher diversity than deeper layers?

---

**Principle 2: Granularity Fidelity and Implementation Consistency in Token-Level Sparse Attention Estimation**

**Definition:**  
When a sparse attention method claims to achieve fine-grained or token-level importance estimation, it is essential to evaluate whether the actual implementation fulfills this claim or instead relies on block-wise or strided approximations that reintroduce coarse-grained information loss. This distinction matters because the field's primary bottleneck is the trade-off between estimation precision (which preserves high-attention outliers) and hardware-aligned block sparsity (which enables kernel efficiency). A principled method should transparently articulate how token-level insights are operationalized within block-retrieval constraints, rather than using token-level rhetoric to describe fundamentally block-sparse execution. Reviewers must scrutinize whether the stride size, block size, and pooling operations undermine the purported granularity advantage, and whether ablations demonstrate genuine sensitivity to these implementation choices. The goal is to distinguish methodological advances that structurally improve estimation fidelity from incremental optimizations that merely soften the coarse-grained bottleneck.

**Core Evaluation Criteria:**
- **Claim-Implementation Alignment:** Does the method achieve actual token-level score computation, or does it employ strides and blocks that conflate token-level insights with conventional block-sparse retrieval? Are these choices justified?
- **Granularity-Performance Causality:** Do ablation studies (e.g., stride=1 versus stride>1) explicitly link finer-grained estimation to improved task performance, or is the relationship inferred from overall speedup numbers?
- **Hardware-Efficiency Justification:** If block or stride constraints are necessary for deployment, does the paper quantify the granularity loss versus a pure token-level oracle and explain why the chosen trade-off is Pareto-optimal?
- **Baseline Granularity Comparison:** Are comparisons made against sequence-dimension compression baselines at equivalent block granularities to isolate the benefit of head-dimension compression?

---

**Principle 3: Adaptive Budget Allocation and Out-of-the-Box Deployability in Multi-Head Sparse Attention**

**Definition:**  
Dynamic sparse attention methods must balance the need for per-head flexibility (accommodating heterogeneous sparsity across heads) against the practical requirement that hyperparameters generalize without expensive per-model grid search. This principle evaluates whether the proposed budget allocation mechanism is principled—deriving sparsity thresholds from input-aware statistics such as cumulative probability distributions—or merely a post-hoc search over fixed global thresholds. In production LLM systems, operators cannot tune sparsity rates for every model scale or context length; therefore, the method's robustness to a default hyperparameter configuration is a critical indicator of real-world utility. Reviewers should assess whether the authors characterize sensitivity to budget thresholds, provide a no-tune recommended setting, and demonstrate cross-model transferability of these hyperparameters. The absence of such analysis suggests a method that is laboratory-effective but deployment-fragile.

**Core Evaluation Criteria:**
- **Principled Adaptivity:** Is the per-head budget derived from an online, input-dependent criterion (e.g., cumulative probability mass) rather than a fixed global Top-K or manually tuned threshold?
- **Hyperparameter Sensitivity:** Does the paper report performance variance across a range of sparsity thresholds and minimum budgets, showing a stable operating region?
- **Zero-Shot Transferability:** Can the method run with a single recommended hyperparameter set (e.g., γ=0.95, min budget=2048) across different model families and context lengths without task-specific re-tuning?
- **Failure Mode Analysis:** Does the work identify regimes (e.g., extremely high sparsity >90%) where dynamic allocation breaks down, and explain why static allocation fails comparatively?

---

**Principle 4: End-to-End System Efficiency and Stage Coverage in Long-Context Sparse Attention Acceleration**

**Definition:**  
For inference-efficiency research, kernel-level speedup or theoretical sparsity ratios are insufficient; the ultimate criterion is wall-clock end-to-end latency reduction (e.g., Time-to-First-Token) under realistic deployment conditions, including memory constraints and heterogeneous batch sizes. This principle demands that sparse attention work report not only attention-kernel acceleration but also prefilling speedup on large-parameter models (e.g., 70B), account for estimation overhead, and honestly scope the method's applicability (e.g., prefill-only versus unified prefill-decode). Furthermore, because modern LLM serving is memory-bound, any additional states introduced by proxy heads or dynamic masks must be analyzed for KV-cache or activation-memory overhead. Reviewers must verify that speedup claims are reproducible on standard hardware, include baseline comparisons against optimized dense kernels (e.g., FlashAttention), and discuss whether the method's benefits diminish at smaller scales or in decode-heavy workloads.

**Core Evaluation Criteria:**
- **Wall-Clock TTFT Reporting:** Are end-to-end prefilling latency and speedup reported across multiple model scales (e.g., 1.5B to 70B) and sequence lengths, not solely kernel-level micro-benchmarks?
- **Memory Overhead Transparency:** Does the paper analyze storage costs for proxy key-value tensors, dynamic masks, or auxiliary buffers, particularly in memory-constrained scenarios?
- **Stage Coverage Honesty:** Is the limitation to the prefill stage clearly stated, and is there a credible analysis of why decode-stage extension is non-trivial or left as future work?
- **Baseline Competitiveness:** Are speedups compared against hardware-optimized dense attention (FlashAttention) and modern sparse alternatives under identical GQA/tensor-parallel configurations?

---

**Principle 5: Conceptual Distinctiveness and Technical Differentiation from Concurrent Head-Sharing Approaches**

**Definition:**  
In a rapidly evolving landscape where multiple works simultaneously identify inter-head redundancy (e.g., mask sharing, clustering, or proxy estimation), it is imperative that new methods precisely delineate their technical contribution relative to concurrent or preceding approaches. This principle ensures that the community can attribute specific algorithmic advances—such as pooled proxy heads versus exact mask replication, or dynamic per-head budgets versus uniform sharing—to the correct work. Reviewers should evaluate whether the authors situate their method within the nuanced design space of head-dimension reduction (pooling, clustering, learning) and sequence-dimension reduction, clarifying which components are novel and which are shared primitives. Failure to do so conflates independent contributions and undermines scientific progress. The rebuttal and related-work discussions should demonstrate a granular understanding of how the proposed mechanism alters the information flow (e.g., shared ranking plus distinct budgets) compared to alternatives.

**Core Evaluation Criteria:**
- **Mechanism Differentiation:** Does the paper clearly distinguish its approach from related methods (e.g., SharePrefill) in terms of exact mask sharing versus proxy ranking, static clustering versus online pooling, and uniform versus dynamic budget allocation?
- **Contribution Attribution:** Are the novel components (e.g., max-pooled proxy head, block-aware cumulative-probability budget) explicitly isolated and validated via ablation, rather than presenting an undifferentiated bundle of techniques?
- **Relationship to Sequence-Dimension Compression:** Does the work articulate why compressing along the head dimension is fundamentally different from (and complementary or superior to) compressing along the sequence dimension, beyond superficial speedup claims?
- **Intellectual Honesty in Concurrent Work:** Does the paper acknowledge temporal precedence and conceptual overlap with contemporaneous work, and use precise technical language to resolve apparent similarities?