**Principle 1: Cross-Architectural Validity of Input-Output Feature Correlation Assumptions in Diffusion Acceleration**

**Definition:**  
For methods that leverage input-output feature relationships to accelerate diffusion models, a critical evaluation criterion is whether the fundamental correlation assumption—that changes in input features reliably predict changes in output features—generalizes beyond the primary experimental architecture. Reviewers must assess whether the claimed correlation holds across diverse architectural families, including both transformer-based diffusion models and convolutional U-Net backbones, as well as across standard and distilled few-step models that exhibit fundamentally different feature trajectories. The evaluation should examine whether formal theoretical justifications, such as local linearity assumptions or proofs of directional consistency, are provided and whether these assumptions are empirically validated across multiple datasets, generative modalities, and timestep regimes. Furthermore, the work should demonstrate that the correlation remains stable under varying layer depths, feature magnitudes, and acceleration intervals, rather than being an artifact of a specific model configuration. Without such cross-architectural validation, the method risks being perceived as an overfitted engineering trick rather than a generalizable scientific insight.

**Core Evaluation Criteria:**
- Does the work empirically validate the input-output correlation across multiple architectural families (e.g., DiT, U-Net) and model types (standard vs. distilled)?
- Are formal theoretical conditions (e.g., local linearity, directional consistency) explicitly stated, and are they supported by empirical evidence beyond a single model or dataset?
- Does the analysis examine correlation stability across different layer depths, timesteps, and feature magnitudes?
- Is the correlation analysis extended to diverse generative modalities (image, text-to-image, video) to establish broad applicability?

---

**Principle 2: Failure Boundary Characterization and Graceful Degradation Under Extreme Acceleration**

**Definition:**  
Acceleration methods must be evaluated not only under typical operating conditions but also at their computational limits, where the interval between full computations becomes large and feature predictions are most stressed. Reviewers should expect explicit characterization of when and why the method fails, including scenarios with extremely low numbers of full computations, highly irregular feature trajectories, or weak input-output correlations. The work must demonstrate whether performance degradation is graceful or catastrophic under these stress conditions and provide quantitative comparisons with baselines at identical aggressive acceleration settings. Crucially, the analysis should identify specific failure modes—such as local linearity breakdown, error accumulation in deep layers, or timestep-dependent approximation breakdown—and explain their root causes. A method that only reports favorable operating points without exposing its limitations lacks the scientific rigor required for trustworthy deployment.

**Core Evaluation Criteria:**
- Does the work include explicit failure case analysis and ablation studies under extremely low NFC or high acceleration ratios?
- Is the degradation pattern characterized as graceful versus catastrophic, with quantitative comparisons to baselines under identical stress conditions?
- Are specific failure modes identified (e.g., breakdown of linearity assumptions, error accumulation) with mechanistic explanations?
- Does the analysis reveal under what conditions (architectures, timesteps, layers) the input-output correlation weakens and how the method behaves in these regimes?

---

**Principle 3: Controllability and Sensitivity of Dynamic Quality-Efficiency Trade-off Mechanisms**

**Definition:**  
For adaptive caching strategies that employ dynamic scheduling based on prediction error thresholds or similar control parameters, the evaluation must rigorously assess whether the quality-efficiency trade-off is intuitive, controllable, and reproducible across different models and tasks. The work should demonstrate a systematic approach to tuning critical hyperparameters—such as error accumulation thresholds—rather than relying on ad hoc selection, and should characterize the relationship between these parameters and the resulting computational budget. Reviewers must examine whether the trade-off curve is smooth and predictable, whether the same control logic transfers across different diffusion models without extensive re-tuning, and whether the method maintains stable NFC counts across different samples. Additionally, the work should visualize or tabulate the quality-efficiency Pareto frontier to demonstrate that the method provides meaningful control over the inference budget. Without such controllability analysis, the practical deployability of the acceleration method remains uncertain.

**Core Evaluation Criteria:**
- Is the tuning process for trade-off parameters (e.g., τ) systematic and reproducible, with clear procedures for matching target computational budgets?
- Does the work characterize the sensitivity of generation quality to the control parameter and visualize the quality-efficiency trade-off curve?
- Do the same scheduling principles and parameter values generalize across different models, datasets, and tasks without extensive re-tuning?
- Is the stability of the computational budget (e.g., NFC consistency across samples) demonstrated and analyzed?

---

**Principle 4: Comprehensive Accountability of Auxiliary Overheads in Real-World Inference**

**Definition:**  
Beyond theoretical FLOP reductions, reviewers must critically evaluate whether the proposed acceleration method introduces hidden computational, memory, or latency costs that diminish real-world deployment benefits. This includes analyzing the overhead of auxiliary operations such as input feature computation, prediction error estimation, and additional memory buffering for intermediate features, and comparing wall-clock time improvements against theoretical speedups. The evaluation should assess scalability concerns, including whether overhead grows with model depth, resolution, or sequence length, and whether memory costs are comparable to or exceed those of existing caching methods. Furthermore, the work should investigate pragmatic approximations—such as computing error estimates for only a subset of modules or layers—to mitigate overhead without substantially degrading quality. A method that achieves impressive FLOP reductions but incurs prohibitive memory costs or wall-clock latency penalties fails to deliver practical value.

**Core Evaluation Criteria:**
- Does the work report actual wall-clock latency and peak memory usage in addition to theoretical FLOP counts?
- Are all auxiliary overheads (input feature computation, error estimation, additional caching memory) explicitly quantified and analyzed?
- Does the overhead analysis scale with model size, depth, and input resolution, and are mitigation strategies evaluated?
- Are practical approximations (e.g., subset module error computation) explored to balance overhead reduction with quality preservation?

---

**Principle 5: Multi-Dimensional Quality Assessment and Metric Reliability Verification**

**Definition:**  
Since feature caching methods can differentially impact perceptual fidelity, semantic alignment, and temporal consistency, the evaluation must employ a diverse battery of metrics spanning distortion-based, perceptual, and semantic dimensions. Reviewers should scrutinize whether the work relies on single-metric conclusions and whether it addresses known limitations of popular metrics—such as CLIP scores reflecting coarse semantic alignment rather than detailed visual correctness or perceptual quality. The work must provide qualitative visual evidence that corroborates quantitative findings, particularly when different metrics yield conflicting rankings or when certain metrics exhibit counter-intuitive behaviors under caching approximations. Additionally, for cross-modal generation tasks, the evaluation should verify that acceleration preserves not only low-level visual quality but also high-level semantic coherence and text-image alignment. A comprehensive quality assessment prevents misleading conclusions driven by metric artifacts and ensures that acceleration does not compromise the user-facing generation experience.

**Core Evaluation Criteria:**
- Does the evaluation include diverse metrics covering distortion (FID, sFID), perceptual similarity (LPIPS, PSNR, SSIM), and semantic alignment (CLIP, ImageReward)?
- Does the work explicitly discuss metric limitations and resolve conflicts where different metrics yield contradictory rankings?
- Is qualitative visual evidence provided to validate quantitative results, especially for failure cases or metric outliers?
- For multi-modal generation, does the assessment verify preservation of both fine-grained visual details and high-level semantic coherence?