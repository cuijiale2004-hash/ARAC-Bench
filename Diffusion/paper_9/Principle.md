### Principle 1: Synergistic Compatibility Verification Between Low-Bit Quantization and Unstructured/Structured Sparsity in LLM Weights

**Definition:**  
This principle evaluates whether the proposed method genuinely addresses the non-trivial interference that arises when applying quantization and sparsity simultaneously to Large Language Models. Reviewers must assess if the work identifies and resolves the specific conflict where removing weights via pruning alters the statistical distribution required for accurate low-bit quantization, or conversely, where quantization noise renders sparse masks suboptimal. The evaluation focuses on whether the joint optimization is theoretically grounded and empirically proven to be superior to simply applying two independent compression techniques sequentially, ensuring that the combined effect does not lead to catastrophic accuracy degradation in sensitive LLM layers.

**Core Evaluation Criteria:**
-   **Identification of Coupling Effects**: Does the paper explicitly analyze how sparsity affects quantization parameters (e.g., scale/zero-point calibration) and vice versa, rather than treating them as orthogonal operations?
-   **Joint vs. Sequential Performance Gap**: Is there a clear empirical demonstration that the joint training/optimization strategy outperforms a naive pipeline of "prune then quantize" or "quantize then prune"?
-   **Distribution-Aware Adaptation**: Does the method include mechanisms to adaptively adjust quantization grids or pruning masks based on the evolving weight distribution during the joint compression process?
-   **Layer-Specific Sensitivity Analysis**: Are critical LLM components (e.g., attention projections, MLP down-projections) analyzed separately to determine if the synergy holds universally or requires layer-wise customization?

---

### Principle 2: Rigorous End-to-End Hardware Efficiency Validation Beyond Theoretical FLOPs Reduction

**Definition:**  
In the context of LLM compression, theoretical metrics like FLOPs reduction or parameter count are often poor proxies for actual inference speedup due to memory bandwidth bottlenecks and kernel inefficiencies. This principle mandates that reviewers scrutinize whether the claimed efficiency gains translate to measurable wall-clock latency improvements and throughput increases on real hardware. It requires verifying that the specific combination of quantization bit-width and sparsity pattern is supported by existing compilers and kernels, or that custom kernels provided are robust and fairly benchmarked against optimized baselines, preventing "paper-only" speedups that fail in deployment.

**Core Evaluation Criteria:**
-   **Real-Hardware Latency Measurement**: Are end-to-end generation latencies (ms/token) and throughput (tokens/s) reported on standard GPUs/NPUs, rather than just theoretical FLOPs or MACs?
-   **Kernel Support and Overhead Accounting**: Does the work account for dequantization/de-sparsification overheads? If custom kernels are used, are they compared against highly optimized vendor libraries (e.g., cuBLAS, TensorRT-LLM)?
-   **Memory Footprint vs. Bandwidth Trade-off**: Is the reduction in memory access volume explicitly measured and correlated with speedup, acknowledging that LLM inference is typically memory-bound?
-   **Fair Baseline Comparison**: Are comparisons made against state-of-the-art *efficient* baselines (e.g., W4A16, SparseGPT) running on the same hardware stack, rather than unoptimized dense models?

---

### Principle 3: Preservation of Generative Coherence and Perplexity-Accuracy Alignment Under Extreme Compression Regimes

**Definition:**  
Evaluating compressed LLMs solely on perplexity (PPL) can be misleading, as PPL may remain stable while generative quality collapses (e.g., repetition loops, loss of instruction following). This principle requires reviewers to verify that the compression method maintains functional utility across diverse downstream tasks, not just language modeling scores. It emphasizes the need for a multi-dimensional evaluation protocol that correlates PPL with zero-shot/few-shot benchmark performance and qualitative generation stability, ensuring that the "QuantSparse" approach does not introduce subtle artifacts that render the model unusable despite acceptable PPL numbers.

**Core Evaluation Criteria:**
-   **Multi-Benchmark Consistency**: Does performance on PPL align with results on reasoning, coding, and instruction-following benchmarks (e.g., MMLU, HumanEval, IFEval)?
-   **Generation Stability Testing**: Are there specific evaluations for degeneration phenomena common in compressed models, such as repetitive n-grams or semantic drift in long-context generation?
-   **Compression Ratio vs. Utility Curve**: Is the trade-off between compression aggressiveness and task performance clearly mapped, identifying the "cliff" where utility breaks down?
-   **Calibration Data Dependency**: Is the sensitivity of the final model quality to the choice of calibration dataset thoroughly investigated and disclosed?

---

### Principle 4: Methodological Generalizability Across Diverse LLM Architectures and Scaling Laws

**Definition:**  
Given the rapid evolution of LLM architectures (Dense, MoE, GQA, MLA), a compression technique validated only on a single legacy model (e.g., LLaMA-2-7B) lacks sufficient impact. This principle assesses whether the proposed QuantSparse methodology is fundamentally transferable or merely overfitted to specific architectural traits. Reviewers must look for evidence of scalability testing across different model sizes and structural variants to ensure the principles derived are robust enough to apply to next-generation models, distinguishing universal compression insights from architecture-specific hacks.

**Core Evaluation Criteria:**
-   **Architectural Diversity**: Has the method been tested on at least two distinct modern architectures (e.g., Dense Transformer vs. Mixture-of-Experts) to prove generality?
-   **Scaling Behavior Analysis**: Does the paper demonstrate how compression efficacy changes with model scale (e.g., 7B vs. 70B)? Do larger models tolerate the joint compression better or worse?
-   **Hyperparameter Transferability**: Can optimal hyperparameters (sparsity ratio, quantization bits) found on small models be reliably transferred to larger ones, or does extensive re-tuning occur?
-   **Component-Specific Robustness**: For newer architectures (e.g., those with grouped query attention or sliding window attention), are these novel components handled correctly without ad-hoc modifications?

---

### Principle 5: Reproducibility and Transparency of Joint Compression Training Dynamics

**Definition:**  
Joint quantization and sparsity methods often involve complex training schedules, gradient estimators (e.g., STE variants), and auxiliary losses that are difficult to reproduce. This principle demands rigorous transparency regarding the optimization trajectory and computational cost of the compression process itself. Reviewers should evaluate whether the authors provide sufficient detail to replicate the exact compression outcome and whether the "cost of compression" (training hours, GPU memory peak) is reasonable relative to the gains, ensuring the method is practically accessible to the research community and not reliant on undisclosed engineering tricks.

**Core Evaluation Criteria:**
-   **Optimization Trajectory Disclosure**: Are convergence curves for both the main task loss and any auxiliary compression losses provided to demonstrate stable joint training?
-   **Computational Cost Reporting**: Is the total GPU-hours and peak memory required for the compression phase explicitly stated and justified?
-   **Gradient Estimator Specification**: Are the specific implementations of straight-through estimators or gradient correction terms detailed sufficiently for reproduction?
-   **Ablation of Training Components**: Are key training design choices (e.g., mask update frequency, quantization-aware fine-tuning duration) ablated to prove their necessity and guide future practitioners?