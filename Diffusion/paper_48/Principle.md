
### Principle 1: Theoretical Justification and Empirical Validation of Noise Schedule Adaptation Mechanisms in Diffusion Probabilistic Models

**Definition:**
For diffusion models that propose adaptive or non-standard noise schedules (e.g., modifying $\beta_t$, SNR, or timestep sampling) to address specific data characteristics or training instabilities, reviewers must assess whether the adaptation is grounded in rigorous theoretical analysis rather than heuristic tuning. This principle evaluates whether the authors explicitly derive the relationship between the proposed noise schedule modification and the underlying variational bound or score matching objective, and whether they provide empirical evidence that the adaptation specifically targets the identified deficiency (e.g., low-frequency signal loss, high-frequency artifact generation, or convergence stagnation). Crucially, the work must distinguish between improvements arising from better alignment with data distribution versus incidental benefits from altered training dynamics or effective batch size changes.

**Core Evaluation Criteria:**
-   **Theoretical Grounding:** Does the paper provide a mathematical derivation or formal argument linking the noise schedule adaptation to an optimization objective (e.g., ELBO, Fisher divergence), rather than presenting it solely as an engineering trick?
-   **Causal Attribution of Improvement:** Do ablation studies isolate the effect of the noise schedule itself from confounding factors such as learning rate adjustments, architecture changes, or total training compute?
-   **Data-Specific Motivation:** Is there clear evidence (e.g., spectral analysis, loss landscape visualization) demonstrating *why* the standard schedule fails for the target domain and how the proposed schedule rectifies this specific mismatch?
-   **Sensitivity and Robustness:** Has the method been tested across different dataset scales, resolutions, and model architectures to ensure the adaptive mechanism is not overfitted to a narrow set of hyperparameters or data statistics?

---

### Principle 2: Comprehensive Multi-Dimensional Assessment of Perceptual Quality Versus Distributional Fidelity in Adaptive Denoising Processes

**Definition:**
In evaluating noise-adaptive diffusion methods, reviewers must scrutinize the evaluation protocol to ensure it captures the inherent trade-off between sample diversity (distributional fidelity) and perceptual quality (sharpness/realism). Adaptive noise strategies often implicitly shift the model’s operating point along this Pareto frontier; therefore, relying on a single metric (e.g., only FID or only LPIPS) is insufficient and potentially misleading. The review must verify that the authors employ a balanced suite of metrics including distribution-based measures (FID, IS, MMD), perceptual metrics (LPIPS, DISTS), and precision/recall decomposition, alongside qualitative analysis that specifically checks for mode collapse, oversaturation, or structural artifacts that quantitative metrics might miss. Furthermore, the choice of reference statistics and preprocessing pipelines must be consistent and justified.

**Core Evaluation Criteria:**
-   **Metric Complementarity:** Does the evaluation include both distribution-matching metrics (FID/KID) and perceptual quality metrics, and do the authors discuss cases where these metrics diverge?
-   **Precision-Recall Analysis:** Are Improved Precision and Recall reported to disentangle gains in sample quality from losses in coverage/diversity, particularly when adaptive noise prioritizes certain frequency bands or timesteps?
-   **Qualitative Failure Mode Inspection:** Beyond cherry-picked best samples, does the paper present uncurated grids or failure case analyses to reveal potential artifacts introduced by the adaptive mechanism (e.g., color shifts, texture repetition, semantic inconsistency)?
-   **Evaluation Protocol Consistency:** Are all baselines evaluated using identical preprocessing, resizing, quantization, and number of samples to ensure fair comparison, especially given that adaptive methods may be sensitive to evaluation-time discretization?

---

### Principle 3: Rigorous Disentanglement of Training-Time Adaptation Benefits From Inference-Time Sampling Strategy Effects

**Definition:**
A critical source of ambiguity in noise-adaptive diffusion research is the conflation of benefits derived from modified training objectives/schedules versus those achievable through post-hoc inference adjustments (e.g., DDIM rescheduling, classifier-free guidance scaling, or stochastic solvers). Reviewers must demand explicit disentanglement experiments that apply the proposed training adaptation while keeping inference fixed at standard settings, and conversely, apply advanced inference strategies to baseline-trained models. This principle ensures that claimed contributions are attributable to genuine learning improvements rather than merely exploiting known sampling heuristics. If the method requires co-designed inference procedures, the combined complexity and computational overhead must be transparently reported and justified against simpler alternatives.

**Core Evaluation Criteria:**
-   **Orthogonal Ablation Design:** Are there controlled experiments comparing [Baseline Train + Standard Sample], [Proposed Train + Standard Sample], [Baseline Train + Advanced Sample], and [Proposed Train + Advanced Sample] to isolate training vs. inference effects?
-   **Computational Overhead Transparency:** If the adaptive method necessitates specialized samplers or additional forward passes during generation, is the wall-clock time and memory cost explicitly compared against standard approaches at matched performance levels?
-   **Compatibility Verification:** Has the proposed training adaptation been validated with multiple ODE/SDE solvers (Euler, Heun, DPM-Solver) to confirm it improves the learned score function universally rather than being coupled to a specific integrator?
-   **Guidance Interaction Analysis:** For conditional generation, has the interaction between the adaptive noise schedule and classifier-free guidance scale been characterized, as adaptive training can significantly alter optimal guidance strengths?

---

### Principle 4: Scalability Validation and Computational Efficiency Trade-off Analysis for Noise-Adaptive Training Regimes

**Definition:**
Noise-adaptive mechanisms often introduce additional computational overhead per step (e.g., dynamic schedule computation, auxiliary networks, per-sample weighting) or require longer training horizons to converge due to modified gradient landscapes. Reviewers must evaluate whether the claimed quality improvements justify the efficiency costs, particularly in the context of large-scale LLM-aligned or high-resolution diffusion training where marginal gains matter enormously. This principle demands reporting of normalized metrics (e.g., FID vs. GPU-hours, not just epochs) and validation that the method’s relative advantage persists—or ideally increases—with scale. Methods that show promise only at toy scales but degrade or become impractical at production scales should be flagged, as scalability is a defining characteristic of impactful diffusion research.

**Core Evaluation Criteria:**
-   **Compute-Normalized Reporting:** Are results presented as performance-versus-compute curves (GPU-hours or FLOPs) rather than solely epoch-based comparisons, accounting for any per-step overhead of the adaptive mechanism?
-   **Scale Transfer Evidence:** Has the method been validated at multiple model sizes or dataset scales, with analysis of whether the benefit-to-cost ratio improves, plateaus, or deteriorates as scale increases?
-   **Convergence Dynamics Characterization:** Does the paper analyze how the adaptive noise schedule affects training stability and convergence speed, including potential warmup requirements or learning rate schedule dependencies?
-   **Memory and Throughput Impact:** For methods involving dynamic computation graphs or auxiliary components, are peak memory usage and samples-per-second throughput reported to assess practical deployability?

---

### Principle 5: Reproducibility Assurance and Open-Source Completeness for Stochastic Noise Schedule Optimization Methods

**Definition:**
Given the high sensitivity of diffusion training to random seeds, initialization, and subtle implementation details in noise scheduling, reviewers must enforce stringent reproducibility standards for adaptive noise methods. This extends beyond code availability to include complete specification of stochastic elements (seed handling, RNG state management for dynamic schedules), exact hyperparameter configurations for all baselines, and precomputed reference statistics used in evaluation. Since adaptive methods often involve non-deterministic components (e.g., importance sampling of timesteps, stochastic schedule perturbations), the review must verify that variance across runs is reported and that the method’s superiority holds statistically, not just on a single lucky seed. Lack of reproducibility infrastructure should be treated as a fundamental flaw regardless of claimed performance.

**Core Evaluation Criteria:**
-   **Statistical Significance Reporting:** Are mean and standard deviation reported across multiple independent runs (≥3 seeds), with appropriate statistical tests confirming that differences exceed run-to-run variance?
-   **Complete Configuration Disclosure:** Are all hyperparameters—including those for baselines—fully specified in the paper or supplementary material, with confirmation that baselines were re-tuned fairly rather than using suboptimal literature values?
-   **Stochastic Component Documentation:** For adaptive/stochastic schedules, is the exact random number generation strategy documented, including how seeds interact with distributed training and data loading to ensure deterministic replay?
-   **Artifact and Metric Code Availability:** Is the evaluation code (including preprocessing, metric computation, and reference statistic generation) released alongside training code to prevent metric implementation discrepancies that commonly plague diffusion benchmarks?