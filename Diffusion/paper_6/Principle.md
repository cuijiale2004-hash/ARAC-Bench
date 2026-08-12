
**Principle 1: Exploitation of Internal Attention-Based Uncertainty Signals for Training-Free Noise Selection in Video Diffusion Generation**

**Definition:**
The work must demonstrate a principled departure from externally designed noise priors—such as frequency filtering, inter-frame smoothing, or heuristic rescheduling—toward internally derived model signals that identify inherently preferable noise seeds. The central claim is that the diffusion model itself contains actionable knowledge about which initial noise configurations will yield higher-quality generations, and that this knowledge can be extracted from intermediate representations rather than imposed through auxiliary priors. The evaluation should assess whether the proposed uncertainty formulation is theoretically grounded in the generative context (e.g., adapting epistemic uncertainty estimation from classification to attention space), whether the chosen internal signal (e.g., attention map entropy disagreement) is demonstrably more informative than external heuristics, and whether the method effectively leverages the model's own confidence and consistency indicators to guide seed selection without requiring retraining or fine-tuning.

**Core Evaluation Criteria:**
- **Theoretical Grounding**: Is the internal uncertainty measure (e.g., BALD-style entropy disagreement) rigorously adapted to the generative setting with appropriate justification, rather than applied ad hoc from discriminative domains?
- **Signal Validity**: Does the proposed metric exhibit strong negative correlation with established quality dimensions, verified through quantitative correlation analysis and scatter-plot visualization across diverse prompts and noise seeds?
- **Paradigm Differentiation**: Does the work clearly articulate limitations of external priors (e.g., high computational cost, lack of model awareness) and empirically demonstrate that internal signals provide superior or complementary guidance?
- **Attention Space Justification**: Is the choice of attention maps over other internal representations (activations, gradients, latent states) theoretically and empirically motivated, with analysis of different attention types (cross, self, temporal) where applicable?

---

**Principle 2: Statistical Significance Verification and Hyperparameter Sensitivity Analysis for Marginal Gains in Video Diffusion**

**Definition:**
In the regime of high-performance video diffusion models where inference-time optimizations often yield modest absolute improvements, rigorous statistical validation is essential to distinguish genuine methodological advances from random variation. This principle demands that empirical claims be supported by confidence interval reporting, multiple independent runs, and appropriate significance testing to establish robustness across prompts and seeds. Furthermore, because training-free methods frequently rely on several critical hyperparameters—such as stochastic masking probabilities, ensemble sizes, or layer truncation thresholds—the work must provide comprehensive sensitivity analyses that demonstrate performance stability across reasonable parameter ranges and justify default configurations through empirical trade-off analysis rather than arbitrary selection.

**Core Evaluation Criteria:**
- **Confidence Interval Reporting**: Are 95% confidence intervals (or equivalent statistical bounds) reported for primary metrics across multiple independent experimental runs with random subsets?
- **Significance Testing**: Does the work include statistical tests validating that improvements are significant and not attributable to sampling variance, particularly for small absolute gains?
- **Hyperparameter Ablation**: Are all key hyperparameters (e.g., Bernoulli masking probability, ensemble size, truncation depth) systematically ablated to show robustness and justify optimal choices?
- **Cross-Dimensional Consistency**: Do improvements manifest consistently across quality, semantic, and motion dimensions without bias toward specific metric subsets that might inflate perceived performance?

---

**Principle 3: Cross-Architecture Generalization and Robustness Validation Across Scales and Post-RL Video Diffusion Backbones**

**Definition:**
A noise selection method must demonstrate broad applicability across the diverse landscape of modern video diffusion architectures, including U-Net-based models, MMDiT transformers, and DiT-based systems, spanning parameter scales from several billion to tens of billions. Beyond base model evaluation, the method must prove robust under post-training enhancement regimes—such as RLHF, DPO, and iterative preference optimization—that fundamentally alter the model's noise-response characteristics and may diminish or amplify the effectiveness of seed-selection strategies. This principle evaluates whether the approach is truly architecture-agnostic and plug-and-play, or whether its efficacy is confined to specific model families, scales, or training paradigms. Fair comparison demands identical inference configurations for each backbone and explicit acknowledgment of any architecture-specific tuning.

**Core Evaluation Criteria:**
- **Architectural Diversity**: Does the method generalize across fundamentally different architectural families (e.g., U-Net, MMDiT, DiT) with consistent directional improvements?
- **Scale Robustness**: Are experiments conducted on both lightweight open-source models and large-capacity state-of-the-art systems (e.g., 14B+ parameters)?
- **Post-Training Compatibility**: Does the method maintain effectiveness on models refined via RL, DPO, IPO, or other preference alignment techniques, which may change attention dynamics?
- **Configuration Fairness**: Are baseline comparisons conducted under each backbone's default or equivalently optimized inference settings, avoiding cherry-picking of favorable hyperparameters?

---

**Principle 4: Mechanistic Interpretability and Predictive Validation of Noise Quality Metrics via Internal Attention Dynamics**

**Definition:**
Beyond aggregate performance metrics, the work must establish a transparent, verifiable causal link between the proposed quality indicator and actual perceptual outcomes through controlled experiments and mechanistic analysis. This principle requires that the metric's predictive power be validated not merely by end-to-end performance gains, but by demonstrating that it captures meaningful internal dynamics—such as attention stability, latent trajectory smoothness, or temporal consistency—that explain *why* certain noise seeds produce superior results. Critical validation includes reversal experiments (selecting worst-scored seeds to confirm degradation), qualitative comparisons of samples from the same prompt with different metric values, and quantitative correlation studies linking the metric to specific quality dimensions. The absence of such analysis risks reducing the method to a black-box oracle whose successes cannot be distinguished from coincidental correlations.

**Core Evaluation Criteria:**
- **Score-Quality Correlation**: Are Pearson or Spearman correlations computed between the proposed metric and automated quality dimensions, with scatter plots visualizing the relationship across diverse prompts?
- **Reversal Validation**: Does selecting seeds with the highest uncertainty scores (reverse criterion) consistently degrade performance, confirming the metric's directional validity?
- **Mechanistic Explanation**: Does the work analyze internal behaviors (e.g., attention map stability, latent trajectory variation) to explain the mechanism by which selected seeds improve motion, composition, or artifact reduction?
- **Qualitative Discrimination**: Are generated samples from identical prompts with high vs. low metric scores visually compared to demonstrate the metric's practical discriminative power?

---

**Principle 5: Inference-Time Efficiency Optimization and Fair Computation Budget Comparison in Training-Free Video Enhancement**

**Definition:**
For training-free inference-time methods, practical viability depends critically on maintaining minimal computational overhead while delivering consistent quality improvements. This principle evaluates whether efficiency claims are validated through rigorous ablation of approximation techniques—such as stochastic masking, layer truncation, or single-step evaluation—and whether comparisons against baselines are conducted under fair neural function evaluation (NFE) budgets. The assessment must verify that performance gains originate from superior noise selection rather than simply increased computation (e.g., additional denoising steps or larger noise pools), and that the overhead remains acceptable for real-world deployment. Key considerations include absolute and relative inference time increases, memory footprint analysis, and identification of optimal operating points that maximize quality per unit computation.

**Core Evaluation Criteria:**
- **Approximation Ablation**: Are efficiency approximations (e.g., Bernoulli masking, cumulative layer truncation, reduced ensemble size) individually validated to confirm they preserve predictive accuracy?
- **Budget Fairness**: Does the work compare against baselines given equivalent computation budgets (e.g., matching NFE counts) to isolate the true benefit of selective initialization from raw computational scaling?
- **Overhead Quantification**: Is inference overhead reported in absolute time and relative percentage across all evaluated backbones, with explicit memory usage characterization?
- **Efficiency-Accuracy Trade-off**: Does the work identify and justify optimal configurations (e.g., default ensemble size, masking probability) that maximize quality gains per unit computational cost?
