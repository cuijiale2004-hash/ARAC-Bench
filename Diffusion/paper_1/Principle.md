**Principle 1: Theoretical Unification and Local Accuracy Guarantees in Generalized Learned ODE Solvers for Diffusion Models**

**Definition:**  
For learned ODE solvers that aim to unify existing sampling methods through learnable parameterization, the work must demonstrate that the proposed framework rigorously subsumes established solvers as special cases while preserving numerical accuracy guarantees. In the context of diffusion and flow-matching models, where the choice of prediction type (noise, data, velocity) and integration domain (logarithmic, linear) leads to divergent discrete-time behaviors, a generalized solver must prove that its learnable interpolations do not degrade the local truncation error order. The theoretical contribution should explicitly address why the unification requires specific architectural choices—such as dual prediction or log-linear domain transforms—and whether these choices are mathematically necessary or merely convenient reparameterizations. Without such rigor, the framework risks being viewed as an arbitrary concatenation of heuristics rather than a principled advancement.

**Core Evaluation Criteria:**
- **Subsumption and Special Cases**: Does the paper rigorously prove that fixing learnable parameters to specific values recovers existing solvers (e.g., noise prediction, data prediction, velocity prediction)? Are the derivations complete and correct?
- **Accuracy Preservation**: Does the learned parameterization maintain the local truncation error order (e.g., second-order accuracy for predictor-corrector schemes)? Are formal proofs provided for the predictor and corrector stages?
- **Necessity of Design Choices**: Does the work justify why dual prediction (or other novel components) is mathematically necessary for deriving the unified integral form, rather than being a redundant reparameterization of a single prediction model?
- **Discretization-Aware Analysis**: Does the paper analyze how continuous-time equivalences break down under discrete-time updates and explain how the proposed framework addresses these discrepancies?

---

**Principle 2: Training Objective Design and Proxy-Objective Alignment for Learned Sampling Parameters**

**Definition:**  
For learned solvers that optimize sampling parameters through differentiable objectives, the choice of training paradigm—whether trajectory regression, feature regression, soft-label classification, or hard-label classification—fundamentally determines the solver's alignment with perceptual quality metrics such as FID and CLIP score. Unlike regression-based methods that require expensive high-NFE teacher trajectories and often underperform in the few-step regime, classification-based objectives using frozen pretrained classifiers offer a teacher-free alternative. However, this introduces the risk of proxy-objective leakage, where optimization biases generated samples toward patterns that maximize classifier confidence rather than true generative quality. A rigorous evaluation must examine how the selected objective correlates with FID, how classifier accuracy affects sample diversity and precision/recall trade-offs, and whether the method remains viable for unconditional generation where class labels are unavailable.

**Core Evaluation Criteria:**
- **Objective Rationale and Ablation**: Does the paper provide a clear motivation for choosing classification over regression? Are all relevant paradigms (trajectory, sample, feature regression; soft-label and hard-label classification) systematically compared under identical settings?
- **Classifier Selection and Bias Analysis**: Is the relationship between classifier architecture, accuracy, and resulting FID thoroughly analyzed? Does the work identify whether very high or very low classifier accuracy harms sample diversity, and is this analyzed through precision/recall decomposition?
- **Proxy-Objective Alignment**: Does the work establish empirical correlation between the training loss (e.g., cross-entropy) and generative quality metrics (FID/CLIP)? Are safeguards against classifier-friendly pattern bias discussed?
- **Applicability to Unconditional Generation**: If the method relies on hard-label classification, does the paper address its limitation for unconditional generation and propose viable alternatives (e.g., soft-label variants or likelihood-based objectives)?

---

**Principle 3: Cross-Architecture Generalization and Comprehensive Baseline Coverage in Few-Step Sampling**

**Definition:**  
Learned ODE solvers are inherently tied to the backbone architecture, noise schedule, and guidance scale, making it essential to validate their efficacy across diverse model families—including both diffusion and flow-matching backbones—and across conditional and unconditional generation tasks. A high-quality study must demonstrate consistent improvements not only against classical dedicated solvers (e.g., DPM-Solver++, DDIM) but also against contemporary learning-based solvers that represent the true state of the art. Because different backbones employ distinct prediction types and operate in different latent spaces, the solver's adaptability and the fairness of comparisons become critical. The evaluation should cover the full low-NFE regime (typically 3–9 steps) and include both quantitative metrics and qualitative visual evidence to support claims of universal improvement.

**Core Evaluation Criteria:**
- **Backbone Diversity**: Are results reported across multiple backbone types (e.g., diffusion-based class-conditional models, flow-matching text-to-image models)? Does the solver handle noise-prediction, velocity-prediction, and data-prediction backbones effectively?
- **Baseline Comprehensiveness**: Does the evaluation include both classical numerical solvers and recent learning-based competitors? Are omissions of relevant methods adequately justified if they cannot be fairly compared?
- **Metric Consistency**: Are standard metrics (FID, CLIP score) computed with consistent protocols (sample size, dataset statistics, guidance scales) across all methods? Are results reported for the full range of NFE values claimed?
- **Qualitative Validation**: Do visual comparisons support quantitative claims, especially in extreme low-NFE settings (e.g., NFE=3) where metric improvements may not fully capture perceptual quality differences?

---

**Principle 4: Parameter Efficiency, Overfitting Robustness, and Practical Deployability of Learned Solvers**

**Definition:**  
The introduction of per-step learnable parameters—often numbering in the tens or hundreds depending on the NFE—creates a significant risk of overfitting to specific backbones, guidance scales, or NFE targets. A credible learned solver must demonstrate that its parameter budget is justified through ablation studies showing performance degradation when parameters are fixed, shared, or removed. Furthermore, practical deployment demands that solvers avoid requiring separate training runs for every possible NFE or backbone configuration. The work should therefore investigate parameter sharing strategies (e.g., global parameters across steps), interpolation schemes for unseen NFEs, and the sensitivity of performance to the total number of learnable degrees of freedom. The computational overhead of the solver itself, beyond backbone evaluations, must also be negligible to ensure real-world utility.

**Core Evaluation Criteria:**
- **Ablation of Parameter Budget**: Are systematic ablations provided for fixing individual parameters (e.g., γ, τ, κ) to their special-case values or sharing parameters across steps? Does the fully learnable configuration significantly outperform restricted variants?
- **Overfitting and Brittleness Assessment**: Does the work analyze whether the solver overfits to specific guidance scales, backbone architectures, or NFE values? Is there evidence of generalization when parameters are transferred across these dimensions?
- **NFE Interpolation and Transfer**: Can parameters trained at one NFE be interpolated or transferred to nearby NFEs without catastrophic performance loss? Are the interpolation formulas and empirical gaps quantified?
- **Training Cost and Practicality**: Is the training overhead reasonable (e.g., no need for high-NFE teacher trajectories)? Is the solver's runtime overhead during inference negligible compared to backbone evaluations?

---

**Principle 5: Methodological Clarity and Intuitive Exposition of Prediction-Domain Unification**

**Definition:**  
Generalized solvers that interpolate among prediction types and integration domains operate in a mathematically complex space where subtle discretization effects determine sampling quality. For such work to have lasting impact, it must provide intuitive explanations—complemented by rigorous derivations—that allow readers to understand why and when specific prediction-domain combinations outperform others. The paper should clearly articulate the geometric or probabilistic meaning of its learnable parameters, explain the observed discretization discrepancies between noise and data predictions through accessible visualizations, and ensure that the narrative flow from problem motivation to technical solution is logically coherent. Without such clarity, even technically sound contributions risk being underappreciated, misapplied, or rejected due to reviewer uncertainty about the core mechanism.

**Core Evaluation Criteria:**
- **Intuitive Parameter Interpretation**: Does the paper explain what the learnable parameters represent geometrically or dynamically? Are visualizations of learned parameter trajectories provided and meaningfully interpreted?
- **Discretization Discrepancy Explanation**: Is the breakdown of continuous-time equivalence under discrete updates explained intuitively (e.g., via geometric illustrations of Euler updates) rather than purely algebraically?
- **Narrative Coherence**: Is the paper structure logically organized so that the necessity of each technical section follows naturally from the preceding one? Are figures explicitly referenced and discussed in the main text?
- **Accessibility of Training Procedure**: Is the classification-based training procedure described with sufficient detail (e.g., via algorithmic pseudocode) to enable reproduction? Are the roles of the decoder, classifier, and loss function clearly delineated?