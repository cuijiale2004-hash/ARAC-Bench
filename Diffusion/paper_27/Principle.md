**Principle 1: Necessity Verification for Learned Sample and Frequency Meta-Weighting Against Simplified Static Heuristics in Diffusion PTQ**

**Definition:**  
The work must rigorously demonstrate that the proposed learned adaptive weighting mechanisms—whether for calibration samples or frequency components—provide substantial and consistent improvements over simpler static or heuristic alternatives. In PTQ for diffusion models, where computational efficiency is paramount, reviewers must assess whether the additional optimization complexity (e.g., bi-level meta-learning, gradient-based weight updates) is justified by empirically superior performance, or whether naive weighting schemes (e.g., linearly increasing frequency emphasis, uniform sample weighting) could achieve comparable results. The principle demands that authors explicitly quantify the performance gap between learned and heuristic approaches through controlled ablation studies, ensuring that the learned components are not merely over-parameterized variants of trivial baselines. Furthermore, the justification must extend beyond aggregate metrics to explain why the learning process captures dynamics that static rules cannot, such as dataset-specific sample importance or non-linear frequency evolution patterns. Finally, reviewers should verify that the overhead of learning is commensurate with the gains, ensuring the method remains practical for post-training deployment scenarios.

**Core Evaluation Criteria:**
- **Ablation against Static Heuristics**: Does the work include direct comparisons with carefully designed static baselines (e.g., fixed frequency weighting schedules, uniform sample weighting) under identical quantization settings? Are the performance differences statistically significant across multiple datasets and bit-widths?
- **Cost-Benefit Analysis**: Does the work transparently report the trade-off between the performance gains from learned weighting and the associated computational overhead (e.g., additional training iterations, memory for weight parameters)?
- **Mechanistic Insight**: Does the work provide visual or analytical evidence (e.g., learned weight visualizations, calibration dataset analysis) explaining why static heuristics fail to capture the necessary adaptation, thereby justifying the need for learning?
- **Quantified Performance Gap**: Are the absolute and relative improvements of the learned approach over static heuristics reported in standardized metrics (e.g., FID, sFID) to demonstrate that the complexity is warranted?

---

**Principle 2: Comprehensive Benchmarking Rigor Against Contemporary PTQ Methods with Efficiency Metrics**

**Definition:**  
Research in diffusion model PTQ must be evaluated against the full spectrum of relevant state-of-the-art methods, including the most recent advances in the field, under standardized experimental conditions. Beyond generation quality metrics (e.g., FID, sFID), reviewers must scrutinize whether the work provides thorough efficiency analyses—including memory usage, inference latency, and calibration time—to establish practical applicability. The principle emphasizes that omitting key competitors (especially those published concurrently or in adjacent venues) or failing to report hardware efficiency metrics significantly undermines the empirical validity and real-world utility of the proposed method. Comparisons must be fair and reproducible, utilizing identical model checkpoints, sampling configurations, and evaluation protocols to isolate the true contribution of the proposed technique. Additionally, the work should clarify the relationship between the proposed method and orthogonal approaches, explaining whether they are competitors, complements, or incompatible due to fundamental design differences.

**Core Evaluation Criteria:**
- **Completeness of Baseline Coverage**: Does the work compare against all major recent PTQ methods for diffusion models (e.g., methods from top-tier venues in the past 1–2 years)? If certain methods are excluded, are the reasons clearly justified (e.g., orthogonal design goals, incompatible quantization schemes)?
- **Standardized Evaluation Protocol**: Are experiments conducted under consistent settings (e.g., same model architectures, datasets, sampling steps, evaluation metrics) to ensure fair comparison? Are results reproduced from official code where possible?
- **Hardware Efficiency Disclosure**: Does the work report memory footprint reduction, inference speedup relative to full-precision models, and any additional overhead introduced by the proposed method? Are these metrics sufficient to assess deployment feasibility on resource-constrained devices?
- **Clarification of Method Relationships**: Does the work explicitly position itself against recent methods with similar or overlapping goals, distinguishing whether they are direct competitors or complementary techniques that could be combined?

---

**Principle 3: Theoretical Grounding and Convergence Analysis for Bi-Level Meta-Learning in Quantization Optimization**

**Definition:**  
When proposing optimization frameworks involving bi-level or meta-learning formulations for PTQ, the work must provide adequate theoretical justification for the convergence properties and optimality of the learned parameters. Reviewers should evaluate whether authors acknowledge the inherent limitations of non-convex bi-level optimization, establish realistic assumptions under which convergence holds, and connect their formulation to established theoretical results in meta-learning or bilevel optimization literature. The principle requires that empirical effectiveness be complemented by theoretical reasoning that explains why the optimization procedure is expected to yield meaningful quantization weights rather than arbitrary local optima. Authors should explicitly state what type of optimality is achieved—whether global, local, or approximate—and under what conditions, avoiding vague claims about "optimal" weighting without qualification. Moreover, the stability and sensitivity of the bi-level optimization to hyperparameters such as learning rates and iteration counts should be analyzed to ensure reliable reproducibility across different experimental settings.

**Core Evaluation Criteria:**
- **Convergence Justification**: Does the work explicitly discuss the convergence behavior of the bi-level optimization procedure? Are assumptions (e.g., Lipschitz continuity, bounded gradients) stated clearly, and are they connected to established convergence theorems in meta-learning?
- **Optimality Characterization**: Does the work clarify what notion of optimality is achieved (e.g., local vs. global optimum)? Is there analysis or empirical validation showing that the learned weights generalize beyond the calibration set to the validation distribution?
- **Avoidance of Black-Box Claims**: Does the work avoid presenting meta-learning as an uninterpretable solution without explaining the optimization dynamics? Are the roles of inner-loop and outer-loop objectives clearly articulated and justified?
- **Hyperparameter Sensitivity**: Does the work analyze how sensitive the bi-level optimization is to key hyperparameters (e.g., learning rates for sample/frequency weights, number of inner steps), and does it provide guidance for stable training?

---

**Principle 4: Empirical Validation and Visual Corroboration of Frequency-Timestep Dynamics as Quantization Priors**

**Definition:**  
For works that leverage domain-specific insights—such as the evolution of frequency components across diffusion timesteps—to guide quantization, reviewers must assess whether the claimed physical or algorithmic properties are rigorously validated. The principle demands that authors not merely assert frequency characteristics but provide empirical evidence (e.g., visualization of calibration data at different timesteps, analysis of learned frequency weights, ablation of frequency decomposition methods) demonstrating that these properties genuinely influence quantization behavior. Furthermore, the choice of frequency transformation (e.g., DWT vs. FFT) and the independence of frequency subbands must be justified with both qualitative and quantitative support. Reviewers should examine whether the frequency prior is exploited through a principled loss design or merely added as an auxiliary term without clear interaction with the quantization objective. Ultimately, the work must demonstrate that the frequency-aware design captures timestep-dependent phenomena that are both reproducible across datasets and structurally meaningful for the denoising process.

**Core Evaluation Criteria:**
- **Motivational Evidence**: Does the work provide visual or statistical evidence from calibration data supporting the claim that different timesteps exhibit distinct frequency characteristics? Are these visualizations intuitive and representative?
- **Component Necessity**: Are ablation studies conducted to isolate the contribution of frequency weighting from sample weighting? Does combining both components yield synergistic improvements?
- **Frequency Decomposition Justification**: Is the choice of frequency transform (e.g., Haar DWT) justified against alternatives (e.g., FFT)? Does the work analyze whether individual frequency subbands (e.g., LH, HL, HH) contribute independently meaningful information, or could they be merged without performance loss?
- **Regularization Efficacy**: If regularization terms are used to enforce frequency-weight patterns (e.g., encouraging high-frequency emphasis at later timesteps), is their impact quantified through ablation?
- **Cross-Dataset Consistency**: Does the frequency evolution pattern hold consistently across diverse datasets and resolutions, validating it as a general property of diffusion denoising rather than a dataset-specific artifact?

---

**Principle 5: Cross-Modality Generalization and Domain-Agnostic Adaptability of Sample-Adaptive PTQ Strategies**

**Definition:**  
A critical criterion for evaluating PTQ methods—especially those claiming model-agnostic or domain-agnostic properties—is their potential to generalize beyond the primary experimental domain (e.g., unconditional image generation). Reviewers must assess whether authors discuss, analyze, or empirically validate the extension of their method to other modalities (e.g., text-to-image, video, audio diffusion) or to different model architectures and noise schedules. The principle requires that claims of generalizability be supported by theoretical reasoning about why the core mechanism (e.g., sample informativeness, frequency evolution) should transfer across domains, ideally accompanied by preliminary empirical evidence or clear adaptation protocols. Even when full cross-modality experiments are infeasible, authors must provide concrete technical roadmaps for adaptation, such as how frequency transforms would be modified for temporal data or how sample weighting would apply to non-image modalities. Finally, reviewers should evaluate whether the method maintains consistent performance across diverse architectures and dataset scales, or whether its efficacy is narrowly tied to specific experimental conditions.

**Core Evaluation Criteria:**
- **Modality Extension Analysis**: Does the work discuss how the proposed method would adapt to other diffusion modalities (e.g., video with 3D DWT, audio with STFT)? Are the domain-specific adaptations clearly specified?
- **Architecture and Schedule Robustness**: Does the work validate performance across different diffusion architectures (e.g., DDPM, LDM) and noise schedules? Is the method schedule-agnostic, and is this property explained?
- **Generalization Claims vs. Evidence**: Are claims of domain-agnostic behavior backed by theoretical arguments (e.g., sample informativeness is universal) or limited to speculation? Does the work acknowledge limitations in generalization scope?
- **Scalability Assessment**: Does the work analyze whether the method scales to larger models (e.g., >100M parameters) and larger datasets without prohibitive overhead?
- **Adaptation Roadmap**: Does the work provide a concrete technical pathway for extending the method to new modalities, including necessary modifications to frequency transforms or sampling strategies?