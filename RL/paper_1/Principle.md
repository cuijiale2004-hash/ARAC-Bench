**Principle 1: Diagnostic Identification and Robust Validation of Experience Value Proxies in RLVR for Reasoning Models**

**Definition:**  
In RLVR for large reasoning models, naive experience replay is insufficient; the community must evaluate whether a paper provides a principled, empirically grounded answer to what constitutes a valuable historical trajectory. This principle demands diagnostic studies that isolate experience characteristics (e.g., question difficulty via rollout correctness, trajectory certainty via entropy) and rigorously validate that these proxies correlate with reasoning quality. It further requires robustness analyses showing that selection heuristics are not brittle artifacts of a specific threshold or dataset. Without such foundational analysis, replay strategies risk being opaque heuristics that fail to generalize across tasks or model scales. The core emphasis is on moving beyond uniform replay toward targeted, justifiable experience valuation.

**Core Evaluation Criteria:**
- **Diagnostic Depth**: Does the work conduct controlled experiments (e.g., bucket-masked training) to isolate which experience properties yield the strongest learning signals, rather than assuming all historical rollouts are equally valuable?
- **Proxy Validation**: Are the proposed value proxies (e.g., medium-difficulty questions, low-entropy trajectories) validated against external references (e.g., a stronger model serving as a reasoning judge) or linked to theoretical objectives?
- **Robustness to Design Choices**: Does the paper provide sensitivity analyses (e.g., varying Gaussian width or mean for difficulty sampling, comparing symmetric vs. hard-biased focus) demonstrating that performance is not contingent on a single arbitrary hyperparameter?

---

**Principle 2: Theoretical and Empirical Stability Assurance in Mixed-Policy Off-Policy RLVR Optimization**

**Definition:**  
Integrating off-policy replay into on-policy RLVR introduces distribution shift and variance that can destabilize training, particularly for large language models prone to entropy collapse or reward hacking. This principle evaluates whether the methodological design includes theoretically grounded corrections (e.g., importance sampling, variance bounds) and empirical safeguards (e.g., policy shaping, delayed activation) to maintain exploration while exploiting past successes. The focus is on ensuring that replay improves stability rather than inducing pathological dynamics such as gradient explosion or policy oscillation. A rigorous evaluation must verify that the mixed-policy objective does not sacrifice the exploration necessary for discovering novel reasoning paths. Ultimately, the work must demonstrate that off-policy experience exploitation is controlled and theoretically sound.

**Core Evaluation Criteria:**
- **Off-Policy Correction Rigor**: Does the method provide unbiased or bounded-variance gradient estimates for replayed trajectories? Are theoretical guarantees (e.g., importance-weighted unbiasedness, variance decomposition) provided and empirically consistent?
- **Exploration Preservation**: Is there empirical evidence (e.g., sustained entropy curves, controlled policy gradient norms) that the method avoids premature convergence or collapse when replaying low-entropy trajectories?
- **Stability Across Model Capacities**: Does the method prevent training failure on both strong and weak base models, especially in regimes where pure on-policy RLVR collapses?

---

**Principle 3: Comprehensive Ablation, Baseline Positioning, and Sample-Efficiency Validation in Experience-Based RLVR**

**Definition:**  
Experience replay research must isolate the unique contribution of experience management from confounding factors such as additional compute, policy shaping, or mere data duplication. This principle demands rigorous experimental design through systematic ablations of each replay component (selection, shaping, importance weighting, replay ratio) and fair positioning against contemporary replay-enhanced baselines. Crucially, because replay's historical promise is sample efficiency, the work must demonstrate learning-curve improvements or data-utilization metrics, not just final-step performance gains. A method that only improves final accuracy without showing improved convergence dynamics may be benefiting from implicit regularization rather than true experience exploitation. Reviewers should scrutinize whether the experimental protocol distinguishes between "more data" and "better data management."

**Core Evaluation Criteria:**
- **Component Ablation**: Are all critical design choices ablated (e.g., question sampling, trajectory selection, policy shaping, IS correction, replay ratio ρ) to establish causal contribution rather than correlated improvement?
- **Baseline Fairness**: Is the method compared against relevant replay baselines (e.g., RePO, RLEP, ARPO) under identical data and training configurations, with clear differentiation of mechanistic focus?
- **Sample Efficiency Evidence**: Does the work present learning curves, buffer evolution analyses, or data-consumption metrics proving that replay yields faster convergence or superior data utilization, rather than only marginal final-score improvements?

---

**Principle 4: Principled Extension and Generalization of Replay Mechanisms to Non-Verifiable Continuous Reward Domains**

**Definition:**  
RLVR methods often rely on binary, verifiable rewards, limiting their applicability to open-ended domains. This principle assesses whether an experience replay framework is inherently constrained to verifiable tasks or can be extended to continuous, preference-based, or sparse-reward settings. High-quality research should propose a principled mapping from the original value proxies to alternative reward structures (e.g., using reward variance as a difficulty proxy) and provide preliminary empirical validation outside deterministic math or code benchmarks. The ability to generalize beyond verifiable domains signals whether the replay logic captures universal learning dynamics or merely task-specific engineering. Without such extension, the method's impact may be siloed to narrow reasoning benchmarks. A strong contribution must at minimum outline a theoretically coherent adaptation path for non-binary rewards.

**Core Evaluation Criteria:**
- **Transferability of Value Proxies**: Does the paper redefine its core selection criteria (e.g., difficulty, quality) without relying on exact answer correctness, enabling extension to standard RLHF or dialogue?
- **Empirical Generalization**: Are experiments conducted on non-verifiable tasks (e.g., open-ended instruction following, chat) with appropriate difficulty or quality proxies?
- **Domain-Independent Formulation**: Is the replay framework formulated such that the bucketing and selection logic remains valid under continuous or stochastic rewards, rather than requiring hard-coded correctness thresholds?

---

**Principle 5: Computational Profiling and System-Level Scalability Analysis of Experience Replay Architectures**

**Definition:**  
Replay buffers inherently introduce memory and computation overheads that can become prohibitive at LLM scale. This principle evaluates whether the paper transparently profiles the system-level costs of experience management (storage, entropy recalculation, bucketing) and demonstrates that these overheads are sublinear, batched, or otherwise manageable relative to the training pipeline. The performance gains must be justified against realistic throughput and memory footprints across different model scales. A method with unbounded per-step complexity is unlikely to be adopted in large-scale training systems regardless of its accuracy benefits. Therefore, algorithmic efficiency and hardware-aware profiling are critical for practical impact. Reviewers must verify that the replay mechanism is engineered for scalability rather than theoretical elegance alone.

**Core Evaluation Criteria:**
- **Overhead Quantification**: Are memory footprint (e.g., GB of RAM) and per-step latency reported across model sizes, with breakdowns of buffer storage versus entropy computation costs?
- **Algorithmic Efficiency**: Is the selection mechanism designed to avoid full-buffer scans (e.g., per-batch entropy recomputation for a small subset of trajectories), ensuring the overhead does not scale linearly with buffer size?
- **Cost-Benefit Justification**: Does the performance improvement (e.g., accuracy gain, training stability) justify the additional computational cost, and is this trade-off explicitly discussed for realistic deployment scenarios?