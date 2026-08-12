### Principle 1: Construction Validity and Ecological Representativeness of Mobile-Specific Video Generation Benchmarks

**Definition:**  
For research proposing new benchmarks for mobile video generation, reviewers must rigorously assess whether the dataset construction methodology genuinely reflects the unique constraints and usage patterns of mobile environments, rather than merely downsampling desktop/web datasets. This includes evaluating the diversity of mobile-captured content (e.g., vertical aspect ratios, varying lighting conditions, handheld camera motion), the authenticity of user prompts derived from real-world mobile applications, and the alignment between benchmark tasks and actual downstream mobile use cases. The principle emphasizes that a valid benchmark must serve as a reliable proxy for real-world mobile performance, ensuring that improvements on the benchmark translate to tangible user experience enhancements on edge devices.

**Core Evaluation Criteria:**
-   **Data Source Authenticity**: Are the videos and prompts sourced directly from mobile platforms or synthesized to accurately mimic mobile-specific artifacts (e.g., compression, sensor noise, orientation)? Is there clear documentation distinguishing mobile-native data from adapted web data?
-   **Task Alignment with Mobile UX**: Do the evaluation tasks (e.g., text-to-video, video editing, super-resolution) correspond to high-frequency mobile user behaviors? Are latency, battery consumption, or thermal constraints integrated into the benchmark design alongside quality metrics?
-   **Distributional Coverage**: Does the benchmark cover the long-tail distribution of mobile content (e.g., low-resource languages, niche cultural contexts, diverse hardware capabilities) to prevent evaluation bias toward high-end device scenarios?
-   **Reproducibility and Licensing**: Is the dataset publicly available with clear licensing for academic and industrial use? Are preprocessing pipelines and annotation protocols fully documented to ensure fair comparison across future studies?

---

### Principle 2: Correlation Rigor Between Automated Metrics and Human Perception in Mobile Video Quality Assessment

**Definition:**  
In mobile video generation research, automated metrics (e.g., FVD, CLIP-Score, LPIPS) are often used as proxies for human judgment due to scalability needs. However, reviewers must demand rigorous statistical validation demonstrating that these metrics correlate strongly with human perception *specifically within the mobile viewing context*. Mobile screens differ significantly from desktop monitors in size, resolution, color gamut, and viewing distance, which can alter perceptual priorities (e.g., temporal consistency may matter more than fine spatial detail). A principle violation occurs when papers assume metric validity based on desktop-centric literature without re-validating on mobile-displayed content or mobile-user populations.

**Core Evaluation Criteria:**
-   **Mobile-Specific Correlation Studies**: Does the paper report Pearson/Spearman correlations between proposed/used metrics and human ratings collected via mobile devices or calibrated to mobile viewing conditions? Are confidence intervals and p-values provided?
-   **Metric Sensitivity to Mobile Artifacts**: Do the metrics reliably detect mobile-relevant degradations (e.g., bitrate-induced blocking, frame drops during network throttling, color banding on OLED screens) that generic metrics might overlook?
-   **Human Evaluation Protocol Transparency**: Is the human study design (e.g., rater recruitment, display calibration, rating scale, inter-rater reliability) fully disclosed? Were raters representative of target mobile users rather than expert annotators using professional monitors?
-   **Ablation of Metric Components**: If a composite metric is proposed, are ablations provided to justify the weighting of sub-components in the mobile context? Is there evidence that removing or adjusting components degrades correlation with human judgment?

---

### Principle 3: Holistic Efficiency-Quality Trade-off Analysis Under Realistic Mobile Hardware Constraints

**Definition:**  
Mobile video generation research cannot be evaluated solely on output quality; it must be assessed through the lens of practical deployability under strict resource budgets. Reviewers should require comprehensive profiling of computational cost (FLOPs, latency), memory footprint, energy consumption, and thermal behavior across a representative range of mobile chipsets (not just flagship SoCs). Crucially, the trade-off analysis must demonstrate Pareto optimality: does the method achieve meaningful quality gains without violating real-time or battery-life thresholds acceptable to mobile users? Papers that report efficiency only in abstract terms (e.g., "parameter reduction") without empirical validation on actual mobile hardware fail this principle.

**Core Evaluation Criteria:**
-   **On-Device Empirical Profiling**: Are latency, memory, and power measurements taken on physical mobile devices (with model/version specified) rather than simulated or extrapolated from server GPUs? Is the testing environment controlled (e.g., thermal state, background processes)?
-   **Pareto Frontier Visualization**: Does the paper present clear efficiency-quality curves comparing the proposed method against baselines? Are operating points justified relative to mobile usability thresholds (e.g., <500ms latency for interactive editing, <2W sustained power for battery life)?
-   **Hardware Diversity Testing**: Is performance evaluated across multiple tiers of mobile hardware (entry-level, mid-range, flagship) to assess generalizability? Are degradation patterns on lower-tier devices analyzed and mitigated?
-   **End-to-End System Integration**: Does the evaluation account for full pipeline overhead (e.g., preprocessing, postprocessing, I/O bottlenecks) rather than isolated model inference? Are software optimizations (e.g., quantization, compiler flags) applied consistently across all compared methods?

---

### Principle 4: Robustness and Generalization of Video Generation Models Across Heterogeneous Mobile Input Conditions

**Definition:**  
Mobile environments introduce extreme variability in input conditions—unstable network streams, diverse camera sensors, inconsistent lighting, and user-generated prompts with grammatical errors or ambiguous intent. Reviewers must evaluate whether proposed models maintain stable performance across this heterogeneity, rather than excelling only on curated, clean inputs. This principle demands stress-testing under realistic noise distributions and out-of-distribution shifts common in mobile deployments. A model that achieves SOTA on benchmark test sets but fails catastrophically on typical mobile inputs lacks practical validity, regardless of its theoretical novelty.

**Core Evaluation Criteria:**
-   **Input Perturbation Robustness**: Has the model been tested against common mobile input corruptions (e.g., low-bitrate video, overexposed/underexposed frames, misspelled prompts, mixed-language queries)? Are degradation curves reported as corruption severity increases?
-   **Cross-Domain Generalization**: Does performance hold when evaluated on unseen mobile content domains (e.g., switching from social media clips to surveillance footage or telehealth videos)? Is domain adaptation required, and if so, at what cost?
-   **Failure Mode Characterization**: Are systematic failure modes under mobile conditions identified and categorized (e.g., temporal flickering in low-light, semantic drift in short prompts)? Are mitigation strategies proposed and validated?
-   **Adaptive Mechanism Validation**: If the model includes adaptive components (e.g., dynamic resolution scaling, prompt rewriting), are their triggers and efficacy validated across diverse input conditions? Do they introduce new failure modes or latency spikes?

---

### Principle 5: Methodological Transparency and Reproducibility of Mobile Video Generation Research

**Definition:**  
Given the rapid evolution and commercial sensitivity of mobile AI, reproducibility is frequently compromised by undisclosed implementation details, proprietary datasets, or unshared code. Reviewers must enforce stringent transparency standards to ensure that claimed advances are verifiable and buildable upon. This includes requiring open-source release of models, training scripts, evaluation code, and (where legally permissible) datasets. When full openness is impossible due to IP constraints, papers must provide sufficient technical detail (e.g., hyperparameters, architecture diagrams, data statistics) to enable faithful reimplementation. Vague descriptions like "standard training procedure" or "proprietary mobile dataset" are unacceptable without justification and alternative validation pathways.

**Core Evaluation Criteria:**
-   **Code and Model Availability**: Are training/inference code, pretrained weights, and evaluation scripts publicly released under an open license? If not, is a detailed reproducibility checklist completed with justification for each missing component?
-   **Training Recipe Completeness**: Are all critical hyperparameters (learning rate schedules, batch sizes, optimizer settings, data augmentation pipelines, loss weights) explicitly listed? Is compute infrastructure (GPU type/count, training duration) disclosed to enable cost-aware replication?
-   **Dataset Documentation**: Even if raw data cannot be shared, are comprehensive statistics (size, distribution, collection methodology, preprocessing steps, license restrictions) provided? Are synthetic data generation procedures fully specified if used?
-   **Evaluation Script Standardization**: Is the exact evaluation protocol (metric implementations, sampling seeds, post-processing steps) version-controlled and shared? Are known pitfalls in metric computation (e.g., FVD implementation differences) addressed and documented?