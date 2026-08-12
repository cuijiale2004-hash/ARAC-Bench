**Principle 1: Exactness Verification and Expressivity-Efficiency Trade-off in Sparse Hard-Constrained Parameterizations for High-Dimensional Fields**

**Definition:**  
For research proposing neural architectures with exact mathematical constraints (e.g., divergence-free, curl-free, or conservative fields) in high-dimensional spaces, it is crucial to evaluate whether the practical implementation preserves exactness under sparse or low-rank approximations. The theoretical guarantee must not be compromised by computational simplifications, and the work should formally prove that the structural property holds for the deployed parameterization regardless of truncation levels. Furthermore, the expressive power of the constrained architecture must be empirically characterized, as hard constraints inherently limit the hypothesis space. Reviewers should expect numerical verification—such as direct measurement of the constrained quantity across inputs—to confirm that the learned model adheres to the theoretical specification. Finally, the trade-off between constraint fidelity and model capacity must be explicitly analyzed to ensure the method remains competitive with less constrained alternatives.

**Core Evaluation Criteria:**
- **Exactness Under Practical Approximation:** Does the sparse or truncated parameterization formally guarantee the exact mathematical property (e.g., zero divergence) for any level of approximation, or does it rely on asymptotic or soft constraints?
- **Expressivity Quantification:** Are ablation studies provided to characterize how the approximation parameter (e.g., number of basis terms) affects representational capacity and downstream performance?
- **Numerical Constraint Verification:** Is the structural property empirically verified on the learned model across the input domain using reliable numerical probes (e.g., Monte Carlo divergence estimation)?
- **Theoretical-Experimental Consistency:** Does the paper reconcile the theoretical representer theorem with the practical sparse instantiation, clarifying what is lost or preserved?

---

**Principle 2: Computational Cost Transparency and Practical Scalability in Constraint-by-Construction Architectures**

**Definition:**  
Architectural constraints enforced by construction often introduce significant computational overhead, such as additional forward passes, automatic differentiation steps, or structured matrix computations. Reviewers must assess whether the paper provides transparent reporting of training and inference costs relative to unconstrained baselines using identical backbone networks and hardware. Claims of "resource efficiency" require rigorous justification, particularly when the method incurs multiplicative slowdowns compared to standard feedforward models. The work should present ablation studies over key controlling hyperparameters (e.g., the number of basis terms) to map the cost-performance Pareto frontier. Ultimately, the practical utility of the approach depends on whether the gains in accuracy, stability, or theoretical purity justify the increased computational budget.

**Core Evaluation Criteria:**
- **Transparent Runtime Analysis:** Are training and inference times reported and compared against unconstrained or soft-constrained baselines implemented with identical backbone architectures?
- **Cost-Performance Justification:** Does the work explicitly discuss whether accuracy or stability gains justify the multiplicative computational overhead (e.g., additional forward/backward passes)?
- **Scalability Ablations:** Are architectural hyperparameters that dominate cost (e.g., basis count, block size) systematically ablated to reveal the accuracy-speed Pareto frontier?
- **Resource Claim Scrutiny:** Are statements such as "resource-efficient" quantitatively justified relative to the unconstrained case, not merely in absolute terms?

---

**Principle 3: Empirical Scope and Real-World Generalization Beyond Idealized Synthetic Benchmarks**

**Definition:**  
Methods developed for inverse problems such as denoising must ultimately demonstrate utility beyond idealized synthetic benchmarks (e.g., grayscale additive white Gaussian noise) to be considered practically relevant. Reviewers should evaluate whether the experimental protocol encompasses diverse data modalities (e.g., color imagery), spatially varying corruption, and real-world sensor noise. Claims regarding "blind" or "unknown parameter" settings demand particularly strict scrutiny: the paper must show that the model was trained with genuinely random or unseen parameter values and not merely tested on slightly different fixed values. Demonstrations on real-world datasets—or clear empirical evidence of transferability via domain adaptation or variance stabilization frameworks—are necessary to validate applicability. Without such breadth, the work risks being perceived as a narrow theoretical exercise with limited external validity.

**Core Evaluation Criteria:**
- **Modality and Noise Diversity:** Does the evaluation extend beyond simplified synthetic settings (e.g., grayscale AWGN) to color imagery, spatially varying noise, or non-Gaussian corruptions?
- **Blind-Setting Rigor:** Are claims of handling "unknown" parameters supported by training protocols with genuinely random parameter values, rather than fixed values omitted from the loss?
- **Real-World Demonstration:** Is applicability demonstrated on practical sensor data or real-world datasets, or is transferability shown via downstream frameworks (e.g., variance-stabilizing transforms)?
- **Domain Limitation Acknowledgment:** Does the paper candidly discuss the limitations of the chosen experimental domain and outline concrete steps for broader validation?

---

**Principle 4: Comprehensive Cross-Paradigm Baseline Coverage and Fair Evaluation Protocol**

**Definition:**  
A thorough empirical evaluation must situate the proposed method within the full landscape of competing approaches, including both structurally similar techniques and conceptually distinct state-of-the-art paradigms. For constraint-based methods, this means comparing not only against other hard-constrained or divergence-based estimators but also against soft-constrained, non-SURE, and supervised baselines under fair conditions (shared architectures, training sets, and evaluation protocols). The comparison should extend beyond final PSNR or accuracy metrics to include training stability, sensitivity to hyperparameters, and convergence behavior. Fairness also requires accurate characterization of prior methods: claims about competing approaches (e.g., their ability to handle unknown noise levels) must be factually correct and up to date. Only through comprehensive, cross-paradigm benchmarking can reviewers assess whether the proposed constraint provides a genuine advance or merely incremental variation.

**Core Evaluation Criteria:**
- **Cross-Paradigm Coverage:** Does the evaluation include both structurally related methods (e.g., same-constraint family) and conceptually distinct state-of-the-art approaches (e.g., non-SURE self-supervised, supervised)?
- **Fair Implementation Standards:** Are shared network backbones, training datasets, and hyperparameter search strategies used to isolate the effect of the proposed architectural constraint?
- **Stability and Sensitivity Metrics:** Does the comparison include training stability signals (e.g., loss curves, gradient norms) and sensitivity to nuisance hyperparameters (e.g., Monte Carlo step sizes)?
- **Accurate Prior Art Characterization:** Are competing methods described correctly and compared with their latest variants, avoiding misrepresentation of their capabilities (e.g., blind vs. non-blind settings)?

---

**Principle 5: Precision in Problem Setting Definition and Theoretical-Experimental Claim Alignment**

**Definition:**  
Precise terminology and rigorous alignment between theoretical claims and experimental validation are essential for trustworthy research in self-supervised inverse problems. Reviewers should verify that terms such as "unknown noise level," "blind denoising," and "exact constraint" are operationally defined and not used interchangeably to inflate contributions. When a theoretical advantage is claimed—for instance, that eliminating a Monte Carlo term reduces gradient variance and improves stability—the paper must isolate this mechanism experimentally rather than relying solely on final performance metrics. Additionally, if hard constraints introduce an inductive bias that shifts the solution away from the unconstrained optimum, the work should quantify or bound this bias theoretically. This ensures that readers can distinguish between performance improvements stemming from the core innovation versus incidental implementation choices.

**Core Evaluation Criteria:**
- **Operational Definition Clarity:** Are critical terms such as "blind," "unknown noise level," or "exact constraint" precisely defined and distinguished from weaker or related concepts?
- **Mechanistic Isolation of Claims:** When a theoretical benefit is claimed (e.g., reduced gradient variance by removing Monte Carlo estimation), is it experimentally isolated from general performance improvements?
- **Bias Quantification:** Does the work quantify or bound the inductive bias introduced by the hard constraint relative to the unconstrained theoretical optimum?
- **Causal Reasoning Avoidance:** Does the paper avoid inferring method effectiveness solely from final metrics without establishing a causal link between the constraint and the observed behavior?