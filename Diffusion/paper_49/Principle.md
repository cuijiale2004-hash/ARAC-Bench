### Principle 1: Theoretical Grounding of Stability Interventions in Diffusion Model Loss Landscapes and Gradient Flow

**Definition:**
Reviewers must assess whether the proposed stabilization method is derived from a rigorous mathematical analysis of the specific instability mechanisms inherent to diffusion models, such as gradient norm explosion at low noise levels, loss landscape curvature singularities, or signal-to-noise ratio degradation. It is insufficient to propose heuristic modifications (e.g., arbitrary clipping or weighting) without linking them to the underlying differential equations or optimization dynamics governing the denoising process. The work must demonstrate that the intervention directly mitigates a theoretically identified pathology rather than merely acting as a generic regularizer. Furthermore, the theoretical claims must be consistent with the empirical behavior observed across different noise schedules and model architectures.

**Core Evaluation Criteria:**
-   **Causal Link to Instability Source:** Does the paper explicitly derive or cite the mathematical origin of the instability (e.g., Hessian eigenvalue spectrum, gradient variance bounds) and show how the proposed method alters this specific quantity?
-   **Consistency with Diffusion Theory:** Is the stabilization technique compatible with the probabilistic interpretation of diffusion models (e.g., preserving the ELBO lower bound or maintaining valid score matching)?
-   **Predictive Power of Theory:** Does the theoretical framework predict *when* and *where* instability will occur, allowing for targeted intervention rather than blanket application?
-   **Distinction from Generic Regularization:** Is the method specifically tailored to diffusion dynamics, or could it be trivially applied to any neural network training without modification?

---

### Principle 2: Disentanglement of Sampling Quality Improvements from Optimization Stability Gains in Generative Modeling

**Definition:**
A critical evaluation standard is the strict separation of "training stability" (optimization convergence, gradient health) from "generation quality" (FID, CLIP scores). Reviewers must verify that improvements in sample metrics are not conflated with stabilization; a method might produce better images simply by changing the effective noise schedule or data weighting, unrelated to solving numerical instabilities. The authors must provide direct metrics of optimization health, such as gradient norm trajectories, loss variance over time, or convergence speed to a fixed threshold, independent of downstream generation benchmarks. This distinction ensures that the contribution is genuinely about robust training dynamics rather than accidental hyperparameter tuning or metric gaming.

**Core Evaluation Criteria:**
-   **Direct Stability Metrics:** Are optimization-centric metrics (gradient norms, loss smoothness, update step sizes) reported alongside or prior to generation quality metrics?
-   **Controlled Ablation for Metric Decoupling:** Do ablation studies isolate the effect of the stability mechanism from other factors known to improve FID (e.g., increased capacity, longer training, EMA tuning)?
-   **Failure Mode Characterization:** Does the paper quantify the frequency or severity of training failures (e.g., NaN rates, divergence events) reduced by the method, rather than just average performance?
-   **Baseline Fairness regarding Generation:** When comparing generation quality, are baselines matched for computational budget and convergence status to ensure gains aren't due to implicit regularization effects?

---

### Principle 3: Robustness Validation Across Noise Schedules, Architectures, and Dataset Scales for Diffusion Stabilizers

**Definition:**
Stabilization methods for diffusion models must be evaluated for their generalizability across the diverse design space of modern generative modeling. A technique that only works under a specific variance-preserving schedule or for UNet architectures but fails for flow-matching or DiT (Diffusion Transformer) backbones lacks sufficient utility. Reviewers should demand evidence that the stability benefits persist when key configuration variables—such as noise parameterization ($\sigma$-space vs. $t$-space), model size, and dataset complexity—are altered. This principle guards against overfitting the stabilization method to a narrow experimental setup and ensures it addresses fundamental properties of the diffusion process.

**Core Evaluation Criteria:**
-   **Schedule Invariance:** Is the method tested across multiple noise schedules (e.g., linear, cosine, sigmoid, discrete-time) to prove it isn't schedule-specific?
-   **Architectural Transferability:** Has the method been validated on at least two distinct backbone types (e.g., UNet and Transformer-based) or parameterizations (VP vs. VE)?
-   **Scale Consistency:** Do stability benefits hold or improve as model parameters and dataset size increase, or do they degrade due to scaling laws?
-   **Sensitivity to Hyperparameters:** Is the method's effectiveness robust to reasonable variations in its own hyperparameters, or does it require precise retuning for each new setting?

---

### Principle 4: Computational Overhead and Implementation Complexity Trade-offs in Stable Diffusion Training Protocols

**Definition:**
In the context of large-scale diffusion model training, any stabilization mechanism must be evaluated through the lens of practical feasibility and computational efficiency. Reviewers must scrutinize whether the stability gains come at an unacceptable cost in terms of memory footprint, wall-clock time per iteration, or implementation complexity that hinders adoption. A method that stabilizes training but doubles memory usage or requires custom CUDA kernels may be less valuable than a slightly less effective but plug-and-play alternative. The evaluation should include explicit profiling of overhead relative to standard baselines and discuss the trade-off curve between stability margin and resource consumption.

**Core Evaluation Criteria:**
-   **Explicit Overhead Reporting:** Are memory usage and training throughput (samples/sec) explicitly measured and compared against vanilla training?
-   **Implementation Accessibility:** Can the method be implemented with minimal code changes using standard deep learning frameworks, or does it require specialized infrastructure?
-   **Stability-Efficiency Pareto Frontier:** Does the paper discuss or visualize the trade-off between the degree of stabilization achieved and the computational cost incurred?
-   **Scalability of Overhead:** Does the computational penalty scale linearly, sub-linearly, or super-linearly with model/data size?

---

### Principle 5: Reproducibility and Diagnostic Transparency of Instability Phenomena in Diffusion Model Research

**Definition:**
Given the notorious sensitivity of diffusion model training, reviewers must demand exceptional transparency regarding the reproduction of instability phenomena and the verification of fixes. Papers claiming to solve stability issues must provide sufficient detail to reproduce both the failure mode and the recovery, including exact random seeds, initialization schemes, and numerical precision settings. Vague descriptions like "training sometimes diverges" are unacceptable; the work must characterize the instability deterministically or statistically with clear diagnostic protocols. This principle ensures that the community can validate the problem-solution pair and prevents the publication of non-replicable "fixes" for idiosyncratic bugs.

**Core Evaluation Criteria:**
-   **Reproducible Failure Induction:** Can reviewers reliably trigger the described instability using the provided code/configs to verify the problem exists?
-   **Diagnostic Tooling Provision:** Does the paper release monitoring scripts or visualization tools that allow others to detect the specific instability in their own setups?
-   **Numerical Precision Specification:** Are floating-point formats (FP16/BF16/FP32) and mixed-precision strategies explicitly documented, as these critically affect diffusion stability?
-   **Statistical Significance of Stability Claims:** Are stability improvements demonstrated over multiple runs with different seeds, rather than single-run anecdotes?