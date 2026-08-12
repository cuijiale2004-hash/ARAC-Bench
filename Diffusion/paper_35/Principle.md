### Principle 1: Theoretical Justification and Empirical Grounding of Stability Criteria in Adaptive Token Locking Mechanisms

**Definition:**
For inference acceleration methods that rely on dynamically skipping computation for "converged" or "stable" tokens (e.g., via KL-divergence thresholds), reviewers must assess whether the stability criterion is supported by both rigorous theoretical bounds and empirical validation. It is insufficient to merely propose a heuristic threshold; the work must demonstrate that the chosen metric (e.g., local posterior stability) mathematically bounds the terminal generation error under stated assumptions. Crucially, because theoretical assumptions (such as geometric tail contraction or Lipschitz continuity) may not strictly hold in deep neural networks, the authors must provide empirical evidence showing that these assumptions are approximately satisfied in practice, thereby bridging the gap between idealized theory and actual model behavior.

**Core Evaluation Criteria:**
-   **Existence of Error Bounds:** Does the paper provide a closed-form bound linking the local locking criterion (e.g., step-wise KL) to the final output quality deviation, rather than relying solely on intuitive justification?
-   **Validation of Theoretical Assumptions:** Are key assumptions (e.g., monotonic decay of divergence, smoothness of logits) explicitly tested and visualized on real models and data, or are they treated as unverified axioms?
-   **Clarification of Theory’s Role:** Is the theoretical analysis clearly positioned as a design rationale/justification rather than a precise numerical predictor of empirical error?
-   **Transparency of Limitations:** Are the gaps between the theoretical model and practical implementation (e.g., temperature scaling effects, discrete vs. continuous posteriors) explicitly acknowledged and discussed?

---

### Principle 2: Comprehensive Assessment of Algorithmic FLOPs Reduction Versus Actual Wall-Clock Speedup in Sparse Attention Systems

**Definition:**
In evaluating compute-skipping methods for diffusion LMs, reviewers must rigorously distinguish between theoretical algorithmic complexity reductions (FLOPs) and realized wall-clock latency improvements (Tokens Per Second). Methods that introduce dynamic sparsity or irregular memory access patterns often face significant system-level overheads (e.g., cache misses, kernel launch latency, low GPU occupancy) that negate FLOP savings, especially in low-batch or short-sequence regimes. A high-quality submission must transparently analyze this "FLOPs-to-Time" gap, identify specific bottlenecks preventing linear scaling, and discuss potential hardware-specific optimizations required to realize theoretical gains in practical deployment scenarios.

**Core Evaluation Criteria:**
-   **Dual-Metric Reporting:** Does the paper report both algorithmic FLOPs and end-to-end wall-clock throughput across diverse batch sizes and sequence lengths?
-   **Bottleneck Analysis:** Is there a detailed discussion explaining why FLOP reduction does not translate proportionally to speedup (e.g., irregular memory access, index management overhead)?
-   **Regime-Specific Evaluation:** Are performance characteristics evaluated separately for compute-bound (large batch/long context) and memory-bound (small batch/short context) regimes?
-   **Pathway to Optimization:** Does the work outline concrete, feasible system-level optimizations (e.g., fused kernels, cache compaction) needed to close the efficiency gap?
-   **Honesty in Claims:** Are efficiency claims scoped appropriately to avoid overpromising practical speedups based solely on arithmetic operation counts?

---

### Principle 3: Orthogonality Verification and Composability with Existing Temporal and Reuse-Based Acceleration Strategies

**Definition:**
Since inference acceleration for diffusion LMs operates along multiple orthogonal axes (temporal step reduction, inter-step state reuse, and intra-step spatial sparsification), reviewers must evaluate whether a proposed method genuinely complements existing approaches rather than competing directly with them. The evaluation should verify that the new method shrinks the problem space for other accelerators (e.g., reducing the active token set for selection-based methods) and demonstrates additive benefits when combined. Mere comparison against baselines is insufficient; the work must establish structural compatibility and show that joint application yields superior efficiency-quality trade-offs compared to any single component alone.

**Core Evaluation Criteria:**
-   **Axis Differentiation:** Is the method clearly distinguished from temporal (step-count) and reuse (KV-cache) approaches in terms of what it optimizes?
-   **Composability Evidence:** Are experiments conducted combining the proposed method with at least one representative orthogonal baseline to demonstrate additive speedups?
-   **Non-Destructive Integration:** Does the combination maintain or improve generation quality compared to individual components, or does it introduce compounding errors?
-   **Clear Positioning:** Does the related work and introduction accurately position the contribution as complementary rather than substitutive?
-   **Scalability of Combination:** Is the combined approach evaluated under realistic settings where both components would typically be deployed together?

---

### Principle 4: Rigorous Validation of Generation Quality Preservation Across Diverse Complexity Regimes and Strict Functional Benchmarks

**Definition:**
When assessing approximation-based decoding methods, reviewers must ensure that quality preservation is validated beyond simple language modeling perplexity or open-ended instruction following. Approximation errors that are benign in creative text generation (e.g., synonym substitution) can be catastrophic in functional tasks like code generation or logical reasoning. Therefore, evaluation must include strict, execution-based benchmarks (e.g., HumanEval pass@1) and cover edge cases such as short-sequence generation where relative approximation impact is higher. The absence of degradation in high-stakes functional tasks provides stronger evidence of safety than similarity in soft metrics on open-ended text.

**Core Evaluation Criteria:**
-   **Functional Correctness Testing:** Is the method evaluated on execution-based benchmarks (e.g., code generation, math solving) where minor local errors cause task failure?
-   **Short-Sequence Sensitivity:** Is performance explicitly tested on short generation lengths where locking/approximation ratios are highest and quality degradation risk is greatest?
-   **Hyperparameter Tunability:** For regimes showing degradation, is it demonstrated that adjusting the approximation threshold can recover quality while retaining some efficiency?
-   **Qualitative Error Analysis:** Are failure modes characterized (e.g., localized phrasing changes vs. global coherence collapse) through case studies?
-   **Metric Diversity:** Does the evaluation combine soft metrics (PPL, LLM-as-judge) with hard metrics (pass rates, exact match) to capture different aspects of quality?

---

### Principle 5: Semantic Validity of Distributional Stability Metrics and Robustness to Premature Commitment in Iterative Generation

**Definition:**
Reviewers must critically examine whether the mathematical definition of "token convergence" used for compute skipping aligns with true semantic determinacy. A posterior distribution may be numerically stable (low KL divergence) while still oscillating between semantically distinct alternatives, leading to premature locking and irreversible generation errors. High-quality research must address this "distributional vs. semantic stability" gap, either through theoretical arguments about the unmasking policy (e.g., confidence-based scheduling ensuring only peaked distributions are locked) or empirical evidence showing that locked tokens rarely correspond to unresolved semantic ambiguities. The work should also discuss mechanisms (e.g., optional unlocking) to mitigate risks of premature commitment.

**Core Evaluation Criteria:**
-   **Stability-Semantics Alignment:** Does the paper explicitly discuss and justify why distributional stability implies semantic safety in the context of the specific sampling scheduler used?
-   **Failure Mode Investigation:** Has the author investigated whether locking occurs during genuine semantic uncertainty (e.g., contradictory token choices) versus benign paraphrase variation?
-   **Safety Mechanisms:** Are secondary safeguards (e.g., confidence gating, minimum step requirements) implemented and ablated to prevent locking during high-uncertainty phases?
-   **Reversibility Consideration:** Is there discussion or experimentation regarding "unlocking" mechanisms to correct premature locking decisions?
