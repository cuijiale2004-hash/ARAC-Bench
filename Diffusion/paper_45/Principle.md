**Principle 1: Milestone Validation and Comprehensive Speed-Quality Frontier Assessment for Faster-than-AR Claims**

**Definition:**  
For research claiming to achieve "faster-than-autoregressive" inference in diffusion large language models, the evaluation must transcend single-point comparisons and rigorously characterize the entire throughput-accuracy Pareto frontier. The community treats "first open-source dLLM to surpass AR speed" as a significant milestone, but this claim is only credible if demonstrated across diverse benchmarks (reasoning, coding, generation), multiple model scales, and against properly matched AR baselines with identical generation-length protocols. Reviewers expect explicit trade-off curves—not isolated operating points—showing that the method maintains comparable quality at substantially higher throughput, and that speedups are not artifacts of early termination, shorter output lengths, or unfair hyperparameter tuning.

**Core Evaluation Criteria:**
- **Frontier Completeness**: Does the work present accuracy-throughput trade-off curves for the proposed method and all key baselines (both AR LLMs and prior dLLM accelerators) under unified experimental settings?
- **Baseline Fairness**: Are generation lengths, sampling temperatures, and evaluation protocols strictly aligned with baseline protocols? Are different max-length settings (e.g., 256 vs. 512) clearly justified and not used to inflate speedups?
- **Milestone Robustness**: Is the "faster-than-AR" claim validated across multiple benchmarks and model families, or limited to a single favorable task? Does the work report both absolute throughput (TPS) and relative speedup with clear denominators?
- **Quality Preservation**: Does the method maintain statistically significant performance parity with unaccelerated dLLMs and competitive AR models, or does acceleration come with hidden accuracy degradation?

---

**Principle 2: Architectural Adaptation Rationale and Exact KV Cache Mechanism Design**

**Definition:**  
A central challenge in dLLM acceleration is the incompatibility between bidirectional attention (inherent to diffusion models) and standard KV caching (essential for AR-speed inference). Research proposing block-wise or causal attention modifications must provide a mechanistic justification for why naive adaptations fail and why the proposed attention structure succeeds. Reviewers scrutinize whether the KV cache is "exact" (not approximate), whether it is applied only to fully decoded blocks, and whether the causal constraint is maintained without leaking future information. The architectural change must be shown to be necessary—not merely a convenient engineering choice—and its interaction with the diffusion denoising process must be theoretically grounded.

**Core Evaluation Criteria:**
- **Mechanism Clarity**: Is it explicitly explained why standard bidirectional dLLMs cannot use exact KV cache, and how the proposed block-wise causal attention resolves this without approximation?
- **Causal Integrity**: Does the design guarantee that future-block predictions do not access information from incomplete predecessor blocks in ways that violate causality? Is KV caching restricted to fully unmasked blocks?
- **Architectural Necessity**: Are ablations provided comparing the full method against cache-only variants (serial block decoding without inter-block parallelism) to isolate the marginal gain of each architectural component?
- **Information Flow**: Is the attention mask structure clearly visualized and explained, particularly the distinction between intra-block bidirectional attention and inter-block causal constraints?

---

**Principle 3: Training Paradigm Justification: Asymmetric Distillation Necessity, Cost, and Scaling Pathways**

**Definition:**  
When acceleration relies on distilling a pretrained bidirectional teacher into a causally restricted student, reviewers demand rigorous justification for why standard (symmetric) distillation or training from scratch is insufficient. The "asymmetric" nature of the distillation—where the teacher sees all noisy blocks globally while the student sees only a causal prefix—must be defended as essential for enabling the target inference paradigm, not merely as a training-speed convenience. The community evaluates whether the distillation cost is reported transparently (GPU-hours, data volume, convergence time), whether the student is capped by teacher quality, and whether the authors acknowledge and address the limitation of teacher dependency. A credible work must articulate a pathway to stronger variants beyond the teacher's ceiling.

**Core Evaluation Criteria:**
- **Asymmetry Necessity**: Does the work provide empirical or theoretical evidence that standard distillation (where teacher and student share the same attention structure) cannot alter the student's attention mechanism to be causal? Is the asymmetric objective shown to be critical rather than incidental?
- **Cost Transparency**: Are training resources reported precisely (e.g., "12 hours on 8× A100 GPUs using 9,000 samples")? Is the distillation process described as rapid and efficient compared to pre-training from scratch?
- **Teacher Dependency**: Does the work acknowledge that performance is bounded by the teacher model? Are there discussions or preliminary experiments on training-from-scratch alternatives or scaling beyond the teacher?
- **Distillation Fidelity**: Does the student preserve the original model's capabilities while gaining the new causal/block-wise generation capability? Are there ablations showing performance degradation if the asymmetric component is removed?

---

**Principle 4: Pipelined Parallel Decoding: Inter-Block Parallelism, Hyperparameter Stability, and Practical Inference Complexity**

**Definition:**  
The inference algorithm—particularly pipelined parallel decoding with dynamic block management and dual-state activation—is a core technical contribution that reviewers evaluate for both elegance and practicality. The method must demonstrate that inter-block parallelism (decoding future blocks before predecessors finish) yields substantial speedups beyond what KV caching alone provides. However, reviewers also scrutinize the introduced inference complexity: new hyperparameters (block size, addition threshold, activation threshold, confidence threshold), their sensitivity, and whether a "default" configuration works robustly across tasks. A method with fragile tuning requirements or opaque interacting knobs faces skepticism regarding real-world deployability.

**Core Evaluation Criteria:**
- **Parallelism Gain Isolation**: Are ablations provided separating the speedup attributable to KV cache reuse from the additional gain of pipelined inter-block parallelism? (e.g., "Cache-only" vs. "Cache + Parallel" comparisons)
- **Hyperparameter Sensitivity**: Is the sensitivity of speed and accuracy to key thresholds (τ_add, τ_act, τ_conf, block size) systematically studied? Is a strong default configuration identified and shown to generalize across benchmarks?
- **Inference Complexity**: Is the decoding algorithm described with sufficient detail (e.g., pseudocode) to be reproducible? Are the trade-offs between aggressive decoding (higher throughput) and conservative decoding (higher accuracy) clearly characterized?
- **Dynamic Behavior**: Is the block-scheduling mechanism (when to add a new block, when to activate aggressive decoding) justified intuitively and empirically? Does it handle variable output lengths and early termination gracefully?

---

**Principle 5: Experimental Rigor, Statistical Significance, and Reproducibility in Throughput Benchmarking**

**Definition:**  
In the dLLM acceleration subfield, experimental rigor is paramount because throughput measurements are highly sensitive to hardware, implementation details, generation length, and sampling randomness. Reviewers expect deterministic or well-controlled evaluation protocols, clear reporting of variance, and explicit alignment with baseline implementations. Claims of 10×–50× speedups are met with heightened scrutiny: reviewers check for statistical significance, error bars on stochastic baselines, consistent hardware platforms, and unambiguous table/figure formatting. Any discrepancy between text and figures (e.g., conflicting TPS or accuracy numbers) severely undermines credibility. The work must also release sufficient implementation details or code to enable independent verification.

**Core Evaluation Criteria:**
- **Statistical Robustness**: For stochastic baselines, are multiple random seeds used and variance reported? For deterministic methods, is the protocol described clearly enough to eliminate hidden randomness?
- **Numerical Consistency**: Are throughput and accuracy numbers consistent across the abstract, main text, tables, and figures? Are discrepancies (e.g., different experimental settings producing different numbers) explicitly explained and justified?
- **Hardware and Implementation Fairness**: Are all methods benchmarked on the same hardware with comparable optimization levels? Is it disclosed whether custom CUDA kernels, batching strategies, or other implementation-level optimizations contribute to speedups?
- **Reproducibility**: Are hyperparameters for every benchmark reported in appendices? Is the code promised or released? Are the training data sources and preprocessing pipelines documented transparently?

---