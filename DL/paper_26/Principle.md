**Principle 1: Theoretical Rigor and Assumption Validity in Extending Convex Optimization Bounds to Deep Learning and LLM Training Dynamics**

**Definition:**  
Research that bridges convex optimization theory and deep learning must be evaluated on whether its theoretical foundations are legitimately extended to non-convex, large-scale settings rather than relying on formalistic analogy. Reviewers must scrutinize whether core assumptions—such as uniform gradient bounds, convexity, or star-convexity—are explicitly justified, relaxed, or shown to hold empirically along the optimization trajectory rather than globally. The theoretical claims should offer novel schedular-specific or architecture-aware insights that go beyond direct citation of classical SGD rates, and the derivation steps must be clearly connected to the empirical phenomena being explained. It is crucial that the theoretical framework distinguishes between what is proven under idealized conditions and what is hypothesized for deep learning practice. Ultimately, the principle demands that theoretical additions provide meaningful, falsifiable constraints on loss dynamics and learning rate scaling.

**Core Evaluation Criteria:**
- **Assumption Justification:** Are convexity-like or bounded-gradient assumptions validated for the optimization trajectory, or are they imposed globally without empirical support?
- **Novelty Beyond Classical Results:** Do the theoretical contributions extend beyond standard convex SGD analysis to address schedule-specific or last-iterate behaviors in deep learning?
- **Theoretical-Empirical Consistency:** Do the derived bounds align quantitatively with observed loss curves across multiple optimizers and architectures?

---

**Principle 2: Comprehensive Cross-Horizon, Cross-Size, and Cross-Modality Empirical Validation with Hyperparameter Ablation**

**Definition:**  
Empirical scaling laws in large model training derive their value from the breadth and robustness of the experimental evidence supporting extrapolation claims. A rigorous evaluation must require validation across substantial ranges of training horizons and model sizes—ideally demonstrating out-of-distribution predictive power—while simultaneously probing sensitivity to hyperparameters that are often treated as nuisance variables. This includes systematic ablation over optimization hyperparameters such as weight decay, momentum, gradient clipping, batch size, and random seeds, as well as extension across diverse modalities like vision, language, and multi-modal architectures. The experimental design should distinguish between interpolation within observed ranges and genuine extrapolation to unseen scales. Without such comprehensive stress-testing, scaling risks being an overfitted artifact of a narrow experimental setup.

**Core Evaluation Criteria:**
- **Extrapolation Breadth:** Does the work provide evidence of predictive accuracy across large multipliers of training horizons and model sizes?
- **Hyperparameter Ablation:** Are key hyperparameters systematically varied to verify that the scaling law is robust and not an artifact of fixed settings?
- **Cross-Modal and Cross-Architectural Validation:** Is the phenomenon replicated across different tasks, model families, and optimizers?

---

**Principle 3: Actionability and Reference-Learning-Rate Transferability in Data-Driven Scaling Laws for Large Models**

**Definition:**  
For scaling laws to transcend academic curiosity and impact practice, they must offer actionable, end-to-end procedures for predicting and controlling training dynamics without requiring exhaustive re-tuning at every scale. Reviewers should evaluate whether the proposed method enables practitioners to determine critical hyperparameters—such as the reference learning rate—from small-scale pilot experiments and transfer them reliably to large-scale deployments. The data-driven fitting procedure must be transparent about what needs to be measured, how coefficients are estimated, and how sensitive predictions are to the choice of reference points. Furthermore, the work should articulate a clear protocol for applying the qualifying exam or scaling rule in realistic training pipelines. Practical utility is ultimately measured by the reduction in computational cost and human tuning effort achieved by the law.

**Core Evaluation Criteria:**
- **Small-to-Large Transferability:** Can the fitted coefficients or reference learning rates determined from small models/short runs predict large-scale behavior without re-fitting?
- **Operational Clarity:** Is the step-by-step procedure for applying the scaling law sufficiently detailed for practitioners to implement?
- **Predictive Scope:** Does the law jointly predict loss dynamics and optimal learning rates, and does it handle schedule families beyond a single default?

---

**Principle 4: Reproducibility and Transparency of Code, Configurations, Logs, and Fitting Methodologies**

**Definition:**  
Reproducibility is a non-negotiable pillar of empirical research in large-scale deep learning, particularly when claims rest on nuanced regression fits and carefully curated training configurations. Reviewers must insist on the release of complete source code, training logs, random seeds, and detailed hyperparameter specifications, not merely high-level descriptions or references to public codebases. The data-driven nature of scaling law estimation demands that the fitting procedures, dataset filtering, and model configuration files be accessible for independent verification. Transparency also extends to reporting negative results, fit quality metrics, and variance across runs. Only with full reproducibility can the community validate extrapolation claims and build upon the work.

**Core Evaluation Criteria:**
- **Artifact Completeness:** Are code, logs, seeds, and configuration files released to enable exact replication of all figures and tables?
- **Methodological Transparency:** Are the regression methods, data preprocessing, and fitting protocols described with sufficient detail to reproduce the scaling law derivation?
- **Disclosure of Variance:** Are error bars, fit statistics, and sensitivity to random seeds reported across experiments?

---

**Principle 5: Diagnostic Clarity and Explicit Characterization of Convex-Like Regime Boundaries and Breakdown Modes**

**Definition:**  
Any empirical law that claims to characterize training dynamics must explicitly define its domain of validity and provide diagnostics for when the underlying assumptions cease to hold. In the context of convex-like behavior, reviewers should expect thorough analysis of boundary conditions—such as the onset of overfitting, divergence between training and test loss, or transitions in curvature regimes—rather than reporting only average-case success. The work should ideally propose empirical diagnostics (e.g., online goodness-of-fit monitoring) or theoretical indicators that signal departure from the predicted scaling regime. Acknowledging and characterizing failure modes not only safeguards against overgeneralization but also deepens scientific understanding of the optimization process.

**Core Evaluation Criteria:**
- **Failure Mode Analysis:** Does the work explicitly identify scenarios where the scaling law breaks down, such as overfitting or non-qualified learning rate schedules?
- **Regime Diagnostics:** Are methods provided to detect the onset or breakdown of the convex-like regime during training?
- **Distinction of Loss Types:** Is the applicability to training versus test loss carefully delineated, with empirical evidence for both?