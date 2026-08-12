**Principle 1: Novelty Attribution and Incremental Contribution Validation in Hybrid Flow Matching-GAN Audio Vocoders**

**Definition:**  
In the rapidly evolving landscape of neural audio generation, hybrid architectures that combine flow-based pre-training with adversarial fine-tuning have become increasingly crowded. Reviewers demand that new work in this space precisely identifies its technical delta over the closest existing hybrid systems, such as concurrent FM+GAN vocoders. It is insufficient to present aggregate improvements; the contribution must be decomposed into individually validated components with causal links to performance gains. This principle ensures that incremental advances are properly attributed and that the community can assess whether the assembly of techniques constitutes a meaningful system-level innovation or a marginal reconfiguration. Ablation studies must isolate each proposed modification within the full pipeline, comparing against strong baselines that share the same hybrid backbone. Ultimately, this criterion guards against ambiguous positioning and ensures that claimed novelty is empirically defensible rather than rhetorical.

**Core Evaluation Criteria:**
- **Component-wise Ablation against Hybrid Baselines**: Are individual technical contributions (e.g., pre-training objective, loss scaling) ablated while holding the FM+GAN framework constant to prove non-redundancy?
- **Direct Comparison with Nearest Hybrids**: Does the work perform head-to-head evaluation against the most relevant concurrent hybrid systems using identical datasets, metrics, and publicly available checkpoints?
- **Causal Pipeline Analysis**: Does the paper establish that improvements in the pre-training stage causally benefit the final few-step GAN output, rather than merely correlating with it?
- **Attribution Clarity**: Is the novelty precisely scoped to architectural, objective, or systemic contributions, avoiding conflation with generic benefits of hybrid training?

---

**Principle 2: Comprehensive Quality-Efficiency Pareto Evaluation and Perceptual Validation in Few-Step Neural Audio Generation**

**Definition:**  
Few-step audio generation is fundamentally evaluated on the quality-efficiency Pareto frontier, where inference speed, computational steps, and perceptual fidelity are inseparable. A method that achieves high fidelity at the cost of many steps, or extreme speed with unacceptable quality, fails to advance the practical state of the art. Reviewers therefore require exhaustive benchmarking that spans multiple step configurations, hardware platforms, and both objective and subjective evaluation protocols. This includes direct comparison against the full spectrum of acceleration techniques, including consistency distillation, shortcut models, and highly optimized pure GANs, under standardized conditions. Perceptual metrics such as MOS and SMOS are essential to validate that objective gains translate to human-perceptual improvements, particularly since spectral-domain metrics can be misleading. Furthermore, downstream integration tests—such as TTS vocoding—are necessary to confirm robustness when conditioning is imperfect. This principle ensures that empirical claims about "favorable trade-offs" are substantiated across the complete operational envelope.

**Core Evaluation Criteria:**
- **Multi-Step Configuration Coverage**: Are results reported for a range of inference steps (e.g., 1, 2, 4, 10+) to characterize the scalability of quality with computational budget?
- **Perceptual and Objective Metric Balance**: Does the evaluation include subjective or learned perceptual scores (MOS, SMOS, UTMOS) alongside spectral/objective metrics to validate human-perceptual gains?
- **Hardware-Agnostic Speed Benchmarking**: Is inference latency reported on both CPU and GPU with standardized batch sizes and segment lengths, enabling fair comparison across the quality-efficiency Pareto frontier?
- **Downstream Robustness Testing**: Is the vocoder validated within an end-to-end system (e.g., TTS) using realistic, potentially noisy conditioning to confirm practical utility?

---

**Principle 3: Theoretical Soundness and Domain-Specific Objective Adaptation Rigor in Continuous Audio Generative Models**

**Definition:**  
Audio data exhibits distinct structural properties—such as silent regions, harmonic periodicity, and perceptually critical low-energy spectral components—that motivate domain-specific modifications to continuous generative frameworks like Flow Matching or diffusion. When researchers reformulate training objectives, for instance by replacing velocity prediction with endpoint estimation or by introducing spectral energy-adaptive losses, they must provide rigorous mathematical justification and prove equivalence or superiority relative to standard formulations. Reviewers scrutinize these adaptations to ensure they are not merely superficial changes or repackaged versions of existing techniques, such as per-frame energy scaling repurposed as time-frequency scaling. The work must demonstrate through controlled ablations that each adaptation specifically addresses an audio generation pathology, supported by both qualitative visualizations and quantitative metrics. Clear differentiation from closely related prior adaptations is mandatory, including conceptual dissection and empirical head-to-head comparisons. This principle enforces methodological rigor and prevents conflation of generic generative tweaks with genuine audio-domain innovations.

**Core Evaluation Criteria:**
- **Mathematical Rigor of Reformulations**: Are modifications to the generative objective (e.g., ODE reformulation, prediction target change) accompanied by derivations, equivalence proofs, or convergence analyses?
- **Audio-Specific Motivation and Evidence**: Is it demonstrated through qualitative and quantitative analysis that the adaptation targets unique audio pathologies (e.g., silent regions, spectral energy sparsity)?
- **Differentiation from Prior Domain Adaptations**: Does the work explicitly contrast its adaptations with the most similar existing techniques conceptually and empirically (e.g., per-frame vs. time-frequency scaling)?
- **Isolated Ablation of Adaptations**: Are audio-specific modifications ablated independently within the full pipeline to verify their specific contribution beyond generic generative improvements?

---

**Principle 4: Conservative Claim Calibration and Paradigm-Aware Positioning in Competitive Neural Audio Synthesis Research**

**Definition:**  
The neural vocoding field encompasses diverse competing paradigms—pure GANs, large-scale diffusion models, consistency distillation, and hybrid FM+GAN systems—each occupying different regions of the speed-quality-data Pareto surface. Papers must avoid overstated universal superiority claims, such as declaring "state-of-the-art" status while ignoring competitors that dominate specific axes like raw inference speed, absolute fidelity, or large-scale generalization. Reviewers expect conservative claim calibration that transparently acknowledges dataset disparities, architectural capacity differences, and computational budget asymmetries between the proposed method and its baselines. The scope of contribution should be framed as a validated, well-characterized system improvement within a clearly defined operational regime, rather than a fundamental paradigm shift. This includes explicitly discussing limitations, such as performance degradation at low bandwidths or speed gaps against ultra-lightweight models. By maintaining disciplined positioning, authors preserve credibility and provide an honest map of the method's practical utility.

**Core Evaluation Criteria:**
- **Avoidance of Overstated Universal Claims**: Does the paper refrain from blanket "state-of-the-art" declarations when competitors excel on specific evaluation axes?
- **Transparent Confounder Disclosure**: Are disparities in training data scale, model size, or compute budget explicitly discussed when comparing against stronger baselines?
- **Bounded Scope of Contribution**: Is the method's advantage clearly framed within a specific operational regime (e.g., few-step high-fidelity vocoding) rather than presented as a universal paradigm shift?
- **Explicit Limitation Acknowledgment**: Does the paper honestly report failure modes or suboptimal regimes (e.g., low bandwidth, specific conditioning types)?

---

**Principle 5: Reproducibility Transparency and Multi-Stage Training Cost Accountability in Practical Audio Generation Systems**

**Definition:**  
Multi-stage training pipelines, while potentially beneficial, introduce reproducibility barriers and engineering complexity that must be justified by substantial end-to-end gains. Reviewers assess whether authors provide comprehensive implementation details, public code, pretrained checkpoints, and precise computational cost accounting for each stage of the pipeline. Training cost transparency—reporting wall-clock time, GPU hours, and iteration counts—is essential to determine whether a two-stage approach is practically preferable to simpler single-stage alternatives. The evaluation must further extend to real-world deployment considerations, such as robustness to conditioning variations generated by imperfect upstream models in TTS pipelines. Without such transparency, the community cannot assess the true cost-benefit ratio of the proposed system. This principle ensures that research advances are not merely theoretical constructs but reproducible, deployable technologies with clearly articulated engineering trade-offs.

**Core Evaluation Criteria:**
- **Public Artifact Availability**: Are source code, model configurations, and pretrained checkpoints released to enable independent reproduction?
- **Stage-wise Cost Accountability**: Are training iterations, GPU hours, and wall-clock time reported separately for each stage and compared against end-to-end baseline costs?
- **Pipeline Complexity Justification**: Is the performance improvement of a multi-stage pipeline sufficiently large to justify its increased engineering and reproducibility burden over simpler alternatives?
- **Robustness to Realistic Conditioning**: Does the evaluation include tests on imperfect or varied conditioning (e.g., log-Mel vs. Mel, noisy TTS outputs) to validate deployment readiness?