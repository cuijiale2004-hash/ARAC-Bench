**Principle 1: Theoretical Rigor and Statistical Validity of Markov Chain Monte Carlo Approximation in Diffusion-Based Distribution Estimation**

**Definition:**  
This principle evaluates whether the reuse of intermediate reverse-diffusion states as samples for estimating conditional distributions is theoretically grounded in established statistical methodology. Reviewers consistently challenged the validity of approximating $p(x|y)$ from correlated, partially-denoised Markov-chain samples, questioning whether strided sampling truly yields near-independent draws and whether intermediate-state estimation introduces fundamental bias. A rigorous submission must therefore frame its sampling strategy within classical MCMC theory—including burn-in and thinning operations—and provide empirical quantification of approximation error. The principle ensures that distributional knowledge distillation does not rest on heuristic sample reuse but on demonstrably reliable statistical estimation, which is essential for reviewer confidence in the method's scientific soundness.

**Core Evaluation Criteria:**
- **MCMC Theoretical Framing:** Does the work rigorously connect the reverse diffusion trajectory to standard MCMC principles (e.g., chain thinning, burn-in phases, and convergence to a target stationary distribution), or does it merely invoke sampling heuristics without theoretical justification?
- **Correlation Reduction Evidence:** Is empirical evidence provided (e.g., Autocorrelation Function analysis) demonstrating that the proposed thinning interval effectively mitigates temporal correlation among intermediate samples?
- **Bias Quantification:** Does the work systematically measure approximation bias against a high-fidelity baseline (e.g., fully denoised independent samples) using rigorous metrics such as KL divergence, and are the reported values sufficiently small to justify the trade-off?
- **Bias-Variance Trade-off Analysis:** Is the performance variation with respect to thinning interval $k$ analyzed in a controlled manner that decouples sample correlation from sample size, ruling out spurious efficiency claims?

---

**Principle 2: Necessity and Distinctiveness of Bidirectional Mutual Enhancement Mechanisms in Unified Perception-Generation Frameworks**

**Definition:**  
This principle assesses whether the proposed unified framework establishes a genuine bidirectional feedback loop in which visual perception and image generation mutually improve one another, as distinct from simpler unidirectional knowledge transfer or data-augmentation baselines. Reviewers raised sharp concerns about whether the architectural complexity—specifically semantic-driven modulation and gradient alignment—is necessary, noting that existing unified models (e.g., Diff-2-in-1) achieve comparable unification through far simpler self-training loops. A defensible submission must therefore demonstrate mechanistically and empirically that its bidirectional design produces synergies unattainable by unidirectional alternatives. The principle guards against "unnecessary innovation" by demanding clear evidence that the cost of added complexity is repaid by unique mutual enhancement rather than incremental, separable gains.

**Core Evaluation Criteria:**
- **Baseline Distinction:** Does the work provide detailed, fair comparisons against simpler unified frameworks under identical experimental conditions, explicitly articulating the functional limitations of unidirectional approaches that the bidirectional design overcomes?
- **Mutual Enhancement Evidence:** Is there empirical proof that both tasks improve simultaneously because of the feedback loop, including analyses that rule out scenarios where generation gains come at the expense of perception performance (or vice versa)?
- **Component Necessity:** Are systematic ablations conducted to verify that each directional pathway (perception $\rightarrow$ generation and generation $\rightarrow$ perception) is individually indispensable, with removal of either component causing significant degradation?
- **Mechanistic Clarity:** Does the work concretely explain how high-level semantic representations inform the generative sampling process and how distributional knowledge specifically refines discriminative decision boundaries beyond what one-hot supervision provides?

---

**Principle 3: Gradient Conflict Resolution and Optimization Stability in Joint Discriminative-Generative Training**

**Definition:**  
This principle examines the effectiveness, necessity, and robustness of strategies designed to reconcile conflicting optimization objectives between discriminative perception losses and generative reconstruction losses within a single shared architecture. Because joint training inherently risks gradient interference—where updates beneficial to one task distort the other—reviewers scrutinized whether the proposed gradient alignment mechanism truly mitigates such conflicts or merely adds computational overhead. The principle requires evidence that the alignment strategy preserves orthogonal (non-conflicting) gradient components while damping conflicting ones, and that training remains stable across diverse hyperparameters, datasets, and architectural backbones. Without such verification, claims of "balanced joint optimization" remain unsubstantiated.

**Core Evaluation Criteria:**
- **Conflict Mitigation Efficacy:** Does the work demonstrate quantitatively that omitting gradient alignment causes measurable performance degradation on one or both tasks relative to independently optimized baselines?
- **Gradient Direction Analysis:** Is there quantitative characterization of gradient geometry (e.g., cosine similarity between task gradients, decomposition into parallel/orthogonal components) showing precisely how conflicts are identified and resolved?
- **Hyperparameter Robustness:** Are task-weighting coefficients and damping factors systematically ablated across a meaningful range to demonstrate that performance is stable rather than dependent on brittle, dataset-specific tuning?
- **Stability Across Settings:** Does the work verify the absence of mode collapse, training divergence, or instability when scaling to different backbones (e.g., CNN-based LDM vs. ViT-based DiT) and across multiple diverse benchmarks?

---

**Principle 4: Computational Overhead Justification and Unified Training Efficiency Assessment**

**Definition:**  
This principle evaluates whether the performance gains of a unified perception-generation model justify its additional computational costs in training and inference, and whether efficiency claims are assessed against appropriately matched baselines. Reviewers challenged the "nearly zero additional cost" assertion, noting that iterative Monte Carlo sampling and gradient alignment could introduce substantial overhead. A credible submission must therefore transparently report training time (GPU-hours), inference throughput (FPS), and memory footprint relative to both specialist perception-only models and comparable unified frameworks. Crucially, efficiency must be evaluated against models with equivalent capabilities rather than against stripped-down single-task baselines, and the work must honestly acknowledge latency constraints that may limit deployment in resource-sensitive scenarios.

**Core Evaluation Criteria:**
- **Explicit Cost Quantification:** Are training time, inference speed, and parameter counts explicitly benchmarked against both perception-only specialist models and standard image-generation training pipelines?
- **Fair Baseline Comparison:** Is the efficiency evaluation framed against models offering comparable unified functionality, avoiding misleading comparisons with models that sacrifice either perception or generation capability entirely?
- **Marginal Cost-Benefit Analysis:** Does the work clearly articulate the trade-off between incremental computational expense and performance gains, demonstrating that overhead is modest relative to the dual-task improvements achieved?
- **Deployment Realism:** Are limitations in latency-sensitive or large-scale deployment scenarios honestly discussed, with concrete directions for future efficiency optimization (e.g., distillation, efficient sampling) provided?

---

**Principle 5: Cross-Task Generalization, Robustness, and Extensibility of Unified Representations**

**Definition:**  
This principle assesses whether the unified model learns representations that generalize robustly across diverse visual understanding tasks—such as classification, semantic segmentation, object detection, and depth estimation—while maintaining effectiveness under distribution shift, input corruption, and open-vocabulary settings. Reviewers questioned whether the framework could extend beyond its core experimental tasks to OCR, image captioning, and text-conditioned generation, and whether representations remained reliable when fed noisy latent inputs rather than clean images. The principle demands that claims of modality-agnostic design and broad applicability be substantiated with empirical evidence across multiple task types, out-of-distribution benchmarks, and input perturbations, ensuring that the unified architecture does not overfit to narrow experimental conditions.

**Core Evaluation Criteria:**
- **Task Breadth and Generalization:** Does the work validate performance across multiple distinct perception tasks (fine-grained classification, segmentation, detection, depth estimation) and demonstrate strong out-of-distribution generalization (e.g., on ObjectNet) rather than isolated in-distribution improvements?
- **Input Robustness:** Is the model evaluated on corrupted or noised inputs (e.g., latents at various diffusion timesteps) to verify that learned representations are robust and not artifactually dependent on clean, noise-free inputs?
- **Open-Vocabulary and Multimodal Extensibility:** Are experiments conducted on open-vocabulary segmentation and detection, and does the work provide concrete architectural analysis or preliminary evidence supporting extension to text-conditioned or cross-modal scenarios?
- **Calibration and Reliability:** Does the joint training paradigm improve prediction calibration (e.g., lower Expected Calibration Error) compared to discriminative-only training, indicating that unified representations yield more reliable confidence estimates?