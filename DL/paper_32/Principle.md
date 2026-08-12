**Principle 1: Direct Mechanistic Evidence for Complex-Domain Modeling Superiority Beyond End-to-End Performance Metrics**

**Definition:**  
When proposing complex-valued neural networks (CVNNs) for tasks where real-valued signals are the ultimate target (e.g., waveform generation via inverse STFT), reviewers expect direct, mechanistic evidence that joint real-imaginary processing captures structural dependencies better than real-valued factorization. End-to-end quality improvements alone are insufficient because they may stem from confounding factors such as increased effective capacity, different initialization landscapes, or hybrid discriminator interactions. Authors must provide controlled experiments—such as synthetic complex-distribution fitting, representation coherence analysis, or ablations with matched memory and capacity—that isolate the representational benefit of native complex arithmetic. The evaluation must explicitly address whether unconstrained complex hidden layers are advantageous compared to enforcing physical constraints (e.g., STFT consistency) or using amplitude-phase decoupled predictors, ensuring that claims of “intrinsic structure capture” are grounded in causal evidence rather than inferred from aggregate metrics.

**Core Evaluation Criteria:**
- **Controlled Distribution Fitting**: Does the work include controlled synthetic experiments (e.g., JSD on magnitude/phase) showing CVNNs better fit coupled real-imaginary distributions than memory-matched real-valued baselines?
- **Representation Analysis**: Are internal representations analyzed (e.g., Grad-CAM, activation statistics) to demonstrate that complex layers exploit cross-component dependencies rather than learning decoupled mappings?
- **Capacity-Controlled Ablations**: Are baselines matched for memory footprint and parameter count to rule out the hypothesis that gains come simply from doubled float storage?
- **Physical Constraint Awareness**: Does the work reconcile the use of unconstrained complex arithmetic with the physical requirement of time-frequency consistency for real-valued outputs?

---

**Principle 2: Comprehensive Baseline Coverage Encompassing Amplitude-Phase Vocoders and Memory-Matched Parameter Controls**

**Definition:**  
In iSTFT-based neural vocoder research, it is insufficient to compare solely against real-valued baselines that process spectrogram channels independently. The evaluation must also include amplitude-phase predictive vocoders (e.g., APNet, APNet2, FreeV), which explicitly model magnitude and phase relationships through separate but coupled objectives, as they represent a strong alternative to complex-valued joint modeling. Furthermore, because complex-valued parameters effectively store two real floats per parameter, fair comparisons require memory-matched or parameter-doubled real-valued baselines to disentangle architectural innovations from raw capacity increases. Reviewers assess whether the proposed method maintains advantages across both objective metrics (UTMOS, PESQ, MR-STFT) and subjective evaluations when these competitive baselines and controls are included under identical training configurations.

**Core Evaluation Criteria:**
- **Amplitude-Phase Baseline Inclusion**: Are explicit amplitude-phase prediction methods compared under identical data and training configurations?
- **Memory-Matched Fairness**: Are real-valued baselines scaled to match the memory footprint of complex-valued parameters (e.g., widened baselines with comparable memory) to isolate the modeling contribution from storage effects?
- **Cross-Dataset Generalization**: Does the evaluation span multiple datasets (e.g., LibriTTS for in-distribution, MUSDB18-HQ for out-of-distribution) and downstream pipelines (e.g., TTS integration)?
- **Metric Diversity**: Is quality assessed through both subjective (MOS, CMOS, SMOS) and objective (spectral, periodicity, voicing) measures to avoid overfitting to a single criterion?

---

**Principle 3: Disentangled Attribution of Quality Gains in Hybrid Multi-Discriminator Adversarial Frameworks**

**Definition:**  
Modern vocoders frequently combine multiple discriminators operating in different signal domains (e.g., time-domain Multi-Period Discriminator and spectral-domain multi-resolution discriminators). When introducing a novel discriminator variant—such as a complex multi-resolution discriminator (cMRD)—reviewers require rigorous ablation to isolate its contribution from co-existing powerful discriminators. If a strong real-valued MPD remains active across all configurations, observed quality gains cannot be confidently attributed to the new complex-domain feedback. Authors must evaluate each discriminator component in isolation (e.g., MPD-only, cMRD-only, MRD-only) and in combination, using identical generator architectures, to establish whether the novel component provides complementary supervision or is merely redundant.

**Core Evaluation Criteria:**
- **Isolated Discriminator Ablations**: Are generator-discriminator combinations systematically tested (e.g., real generator + real discriminator, complex generator + real discriminator, etc.) with the novel discriminator used alone and paired with existing ones?
- **Complementarity Evidence**: Does the novel discriminator improve metrics when used by itself, and does it provide additive gains when combined with the standard MPD?
- **Qualitative Feedback Analysis**: Are visualization tools (e.g., Grad-CAM) used to demonstrate that the complex discriminator provides more structured spectral feedback compared to real-valued counterparts?
- **Avoidance of Confounding**: Is the MPD held constant or removed in ablations to prevent it from masking or dominating the effects of the proposed complex discriminator?

---

**Principle 4: Conceptual and Empirical Validation of Domain-Specific Structural Constraints Like Phase Quantization**

**Definition:**  
Novel architectural inductive biases introduced for complex-valued networks—such as phase quantization (PQ)—must be supported by both theoretical or heuristic motivation and rigorous empirical ablation. Because complex nonlinearities are underexplored, reviewers expect clear justification for why a specific structural constraint is needed at a particular network location (e.g., regularizing initial phase formation when converting real inputs to complex features). The evaluation must include placement ablations (e.g., PQ at different layers), architectural variants (e.g., replacing the first complex layer with a real layer producing two channels), and sensitivity analyses (e.g., varying quantization levels) to demonstrate that the component is a deliberate design choice rather than an ad-hoc add-on whose effects are weak or inconsistent.

**Core Evaluation Criteria:**
- **Theoretical/Heuristic Motivation**: Is there a clear explanation for why the constraint addresses a specific pathology (e.g., unconstrained phase drift during early feature formation)?
- **Placement and Sensitivity Ablations**: Are the effects of the component tested at different network positions and across hyperparameter values (e.g., number of quantization levels)?
- **Architectural Variant Controls**: Is the component tested against alternatives that achieve similar representational capacity without the proposed constraint (e.g., real Conv1D outputting two channels vs. complex Conv1D + PQ)?
- **Empirical Impact Clarity**: Do ablations show consistent, meaningful effects on perceptual and objective metrics, or are improvements marginal and noisy across seeds?

---

**Principle 5: Rigorous Fair Accounting for Memory Footprint and Latency in Complex-Valued Efficiency Claims**

**Definition:**  
Efficiency claims for complex-valued implementations—such as block-matrix computation schemes—must be evaluated against appropriately matched baselines with careful attention to hardware and framework realities. Because complex data types double memory usage and because deep-learning frameworks may offer limited optimization for complex operations, authors must report not just relative training-time reductions but also inference latency, memory consumption, and backward-graph complexity under identical experimental conditions. Reviewers expect comparisons against both native complex implementations and established arithmetic optimizations (e.g., Gauss multiplication trick), along with numerical consistency checks to verify that computational reformulations do not alter training dynamics or model fidelity.

**Core Evaluation Criteria:**
- **Multi-Way Implementation Comparison**: Is the proposed scheme compared against native complex operations and known optimizations in both forward and backward pass times?
- **Memory and Latency Transparency**: Are inference latency, GPU memory, and parameter storage reported honestly, including acknowledgments of framework limitations affecting complex-valued operations?
- **Numerical Equivalence Verification**: Are forward outputs and gradient magnitudes checked for consistency between the optimized and standard implementations?
- **Backward-Pass Emphasis**: Does the analysis focus on backward-pass savings (which dominate training cost) and tie them to actual end-to-end training time reductions rather than forward throughput alone?