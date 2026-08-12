### Principle 1: Theoretical Rigor and Mathematical Validity of Score Function Approximation in Diffusion-Based Inverse Problem Solving

**Definition:**
This principle evaluates whether the proposed method provides a mathematically sound derivation for approximating the posterior score function when integrating a pre-trained diffusion prior with an external measurement model. Reviewers must assess if the approximation of $\nabla_{x_t} \log p(y|x_t)$ is theoretically justified rather than heuristically motivated, particularly regarding the linearization assumptions or Taylor expansions used to bridge the latent space of the prior and the observation space. It is crucial to verify that the method correctly handles the noise schedule alignment between the pre-trained prior and the inverse problem formulation, ensuring that the guidance signal remains valid across all diffusion timesteps. Without this theoretical grounding, empirical improvements may be attributed to hyperparameter tuning rather than genuine algorithmic correctness.

**Core Evaluation Criteria:**
-   **Derivation Soundness:** Is the mathematical approximation of the conditional score clearly derived with explicit statements of assumptions (e.g., local linearity, Gaussian likelihood), and are the conditions under which these assumptions hold rigorously discussed?
-   **Noise Schedule Consistency:** Does the method properly account for discrepancies between the training noise schedule of the pre-trained prior and the inference schedule required for the specific inverse task?
-   **Gradient Propagation Validity:** Are the gradients computed through the measurement model numerically stable and theoretically consistent with the denoising trajectory, avoiding issues like gradient explosion or vanishing guidance at critical timesteps?
-   **Comparison with Exact Posterior Sampling:** Does the work benchmark against or theoretically relate to methods that compute exact (or more accurate) posterior scores to quantify the error introduced by the proposed approximation?

---

### Principle 2: Comprehensive Empirical Validation Across Diverse Ill-Posed Inverse Tasks and Degradation Severities

**Definition:**
For research proposing a general-purpose diffusion prior framework, it is insufficient to validate performance on a single restoration task or a narrow range of degradation parameters. This principle mandates that the evaluation covers a representative spectrum of ill-posed problems (e.g., super-resolution, deblurring, inpainting, compressed sensing) with varying degrees of ill-posedness to test the robustness of the prior integration. The assessment must determine whether the method’s performance gains are consistent across different physical measurement models or if they are artifacts of a specific task setup. Furthermore, the evaluation should include both synthetic benchmarks with ground truth and, where possible, real-world degraded images to assess practical applicability beyond controlled laboratory settings.

**Core Evaluation Criteria:**
-   **Task Diversity Coverage:** Does the experimental section include at least three distinct types of inverse problems with fundamentally different forward operators (e.g., spatial vs. frequency domain degradations)?
-   **Degradation Severity Sweep:** Are results reported across a continuous or sufficiently granular range of degradation levels (e.g., multiple blur kernel sizes, noise variances, or sampling rates) rather than just standard benchmark settings?
-   **Real-World Generalization:** Is there evidence that the method performs reliably on real-captured data where the forward model may be imperfect or unknown, distinguishing it from methods that overfit to synthetic degradation models?
-   **Metric Complementarity:** Are both pixel-level metrics (PSNR/SSIM) and perceptual/generative quality metrics (LPIPS, FID, CLIP-IQA) reported to capture the trade-off between fidelity and perceptual realism inherent in generative priors?

---

### Principle 3: Computational Efficiency Analysis and Practical Feasibility of Guidance-Based Posterior Sampling

**Definition:**
Since diffusion-based inverse solvers often require iterative gradient computations through the measurement model at each denoising step, computational cost is a first-class evaluation criterion alongside reconstruction quality. This principle requires authors to provide a transparent analysis of the runtime overhead, memory footprint, and number of function evaluations (NFEs) introduced by their guidance mechanism compared to baseline posterior sampling methods. Reviewers must assess whether the claimed quality improvements come at a prohibitive computational cost that renders the method impractical for real applications. The evaluation should also consider whether the method introduces additional trainable parameters or requires retraining, as training-free adaptation is often a key selling point of diffusion prior approaches.

**Core Evaluation Criteria:**
-   **Runtime and NFE Reporting:** Are wall-clock time, GPU memory usage, and total NFEs explicitly reported and compared against relevant baselines under identical hardware conditions?
-   **Quality-Efficiency Trade-off Curves:** Does the paper present Pareto-frontier analyses showing reconstruction quality as a function of computational budget, rather than only reporting peak performance at maximum cost?
-   **Training-Free vs. Training-Required Clarity:** Is it clearly stated whether the method requires any fine-tuning or additional training, and if so, is the associated training cost justified by the inference-time benefits?
-   **Scalability Assessment:** Does the computational complexity scale reasonably with image resolution and batch size, or does it exhibit super-linear growth that limits applicability to high-resolution scenarios?

---

### Principle 4: Robustness to Measurement Model Mismatch and Prior-Domain Distribution Shifts

**Definition:**
A critical failure mode in applying pre-trained diffusion priors to inverse problems is the sensitivity to mismatches between the assumed forward model and the true degradation process, as well as distribution shifts between the prior’s training data and the target domain. This principle evaluates whether the authors have systematically tested the method’s resilience to such mismatches, including inaccurate blur kernels, misestimated noise levels, or domain gaps (e.g., natural image prior applied to medical imaging). High-quality work should demonstrate graceful degradation rather than catastrophic failure when assumptions are violated, and ideally provide mechanisms for adapting to or detecting such mismatches. This distinguishes robust methodologies from those that merely exploit benchmark-specific priors.

**Core Evaluation Criteria:**
-   **Model Mismatch Stress Testing:** Are experiments conducted with intentionally perturbed or misspecified forward models (e.g., wrong kernel size, incorrect noise variance) to evaluate sensitivity?
-   **Cross-Domain Transfer Evaluation:** If applicable, is the method tested on target domains that differ from the prior’s training distribution, with clear quantification of performance degradation?
-   **Adaptation Mechanisms:** Does the method include any built-in robustness features (e.g., adaptive step sizes, uncertainty-aware guidance, or online calibration) to handle mismatch, or is it entirely reliant on perfect model knowledge?
-   **Failure Mode Characterization:** Are typical failure cases under mismatch conditions analyzed and explained mechanistically, rather than being omitted or dismissed as out-of-scope?

---

### Principle 5: Ablation Integrity and Causal Attribution of Performance Gains to Proposed Technical Innovations

**Definition:**
Given the complexity of combining diffusion priors with inverse problem solvers, it is essential to verify that reported improvements are causally attributable to the novel technical contributions rather than confounding factors such as superior baselines, favorable hyperparameter choices, or implementation details. This principle demands rigorous ablation studies that isolate each proposed component (e.g., specific score approximation, timestep scheduling, guidance weighting scheme) and demonstrate its individual contribution. Reviewers should scrutinize whether the baselines are implemented fairly with equivalent computational budgets and whether the ablations cover plausible alternative designs. Without such causal verification, the paper’s contribution remains ambiguous and potentially non-reproducible.

**Core Evaluation Criteria:**
-   **Component-Wise Isolation:** Does the ablation study systematically remove or replace each proposed innovation to confirm its necessity and sufficiency for the observed gains?
-   **Baseline Fairness Verification:** Are all baselines run with optimized hyperparameters and equivalent computational resources, with explicit confirmation that no baseline was handicapped by suboptimal configuration?
-   **Alternative Design Comparison:** Are reasonable alternative formulations of the key technical ideas tested (e.g., different approximation orders, alternative guidance weights) to justify the specific design choices made?
-   **Reproducibility Transparency:** Are all critical implementation details (hyperparameters, random seeds, preprocessing steps) fully disclosed to enable independent verification of the ablation claims?