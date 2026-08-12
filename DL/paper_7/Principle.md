**Principle 1: Precision and Conceptual Rigor in Applying Statistical Physics Analogies to Stochastic Optimization Dynamics**

**Definition:**  
Work at the intersection of statistical physics and neural network optimization must translate physical concepts with high precision to avoid fundamental category errors. The notions of entropic force, free energy minimization, and effective temperature must be clearly distinguished from their energetic and dynamical counterparts, particularly because SGD operates as a non-equilibrium, discrete-time stochastic process rather than a thermodynamic ensemble. Reviewers expect authors to avoid conflating thermodynamic unfavorability with dynamical impossibility, instead using precise language such as "effectively forbidden" to describe low-probability transitions. The mapping between physical quantities and algorithmic hyperparameters—such as identifying an effective temperature proportional to learning rate divided by batch size—must be explicitly justified and qualified with respect to its regime of validity. Finally, the work must demonstrate that any apparent paradoxes, such as optimization climbing loss landscapes, are properly understood as free energy minimization rather than violations of energy descent.

**Core Evaluation Criteria:**
- **Physical Conceptual Fidelity:** Are statistical physics terms used in ways consistent with established definitions (e.g., distinguishing entropic confinement from energetic/dynamical confinement)?
- **Clarity of Effective Dynamics:** Does the work explicitly define the stochastic process approximations and their regimes of validity (assumptions on noise color, effective temperature scalings, and timescale separations)?
- **Avoidance of Overreaching Framing:** Does the manuscript avoid claiming that entropy "causes" dynamics to climb loss in ways that contradict free energy minimization principles, and does it refrain from overstating localization phenomena as purely entropic?

---

**Principle 2: Quantitative Disentanglement of Entropic and Energetic Contributions to Optimization Barriers**

**Definition:**  
A defining characteristic of high-quality research in this domain is the rigorous separation of curvature-driven entropic effects from gradient-driven energetic effects along low-loss manifold structures. It is insufficient to demonstrate that curvature rises away from minima; the work must establish that these variations produce effective forces capable of biasing optimization dynamics independently of loss gradients, especially in flat regions where energetic barriers are negligible. This requires explicit identification of experimental regimes and quantitative metrics that compare the magnitude of entropic drift against loss-gradient drift, rather than relying on post-hoc interpretation of final performance. The research should employ controlled experimental designs—such as projected dynamics on minimum energy paths, systematic variation of noise scale via batch size and learning rate, and initialization studies—that isolate curvature as the causal variable. Without such disentanglement, claims about entropic confinement remain phenomenological and cannot distinguish true mechanistic effects from correlated but epiphenomenal landscape features.

**Core Evaluation Criteria:**
- **Regime Identification:** Does the work identify and experimentally demonstrate regimes where energetic forces are negligible relative to entropic forces (e.g., late-stage training, projected dynamics on flat paths)?
- **Relative Strength Quantification:** Is there explicit measurement or bounding of the ratio between entropic drift and loss-gradient drift, rather than inferring entropic dominance solely from final performance or qualitative observations?
- **Causal Isolation:** Are experiments designed to isolate curvature effects (e.g., projected SGD on MEPs, controlled noise injection, varying batch size/learning rate as temperature proxies) while controlling for confounding variables?

---

**Principle 3: Cross-Architectural, Cross-Dataset, and Cross-Optimizer Robustness for Landscape Geometry Claims**

**Definition:**  
Because geometric claims about loss landscapes are often implicitly universal, reviewers demand robust experimental validation that transcends narrow architectural or dataset-specific idiosyncrasies. The burden of proof requires systematic demonstration across diverse network families, data distributions, and training configurations to rule out phenomena that arise from a single model-dataset pair. High-quality studies extend their analysis to multiple architectures varying in depth and width, validate findings on datasets of different complexity, and examine sensitivity to optimizer choice and hyperparameter settings. This broad scope is essential because connectivity properties, curvature spectra, and implicit regularization strengths can vary dramatically across these dimensions. Ultimately, the work must provide evidence that observed barriers, bumps, or connectivity patterns are structural features of the optimization problem rather than artifacts of a particular experimental setup.

**Core Evaluation Criteria:**
- **Architectural Diversity:** Are experiments conducted across multiple architectures of varying depth and width (e.g., ResNet-20, Wide ResNet-16-4, ResNet-110)?
- **Dataset Scaling:** Is behavior validated beyond a single benchmark (e.g., extending from CIFAR-10 to CIFAR-100 or other domains)?
- **Optimizer and Hyperparameter Sensitivity:** Does the work investigate how entropic/barrier phenomena vary with optimizer choice (SGD vs. Adam/momentum) and hyperparameters (learning rate, batch size), and does it report scaling laws or sensitivity analyses?
- **Ablation Completeness:** Are key experimental components ablated (e.g., different splitting epochs in linear mode connectivity, varying path construction methods)?

---

**Principle 4: Methodological Awareness and Validation of Path-Sampling Biases in Landscape Connectivity Analysis**

**Definition:**  
The empirical study of landscape connectivity depends critically on algorithmic path-sampling methods—such as nudged elastic band algorithms and linear interpolation—that introduce specific geometric and dynamical biases. Reviewers expect authors to acknowledge that these algorithms may favor paths with particular length, curvature, or coupling properties, and to validate that observed phenomena persist across independent methodological approaches. Furthermore, constrained optimization protocols like projected SGD, while useful for isolation, artificially restrict dynamics and may not reflect the behavior of unconstrained training trajectories. High-quality work carefully controls for estimation artifacts in curvature measurement, such as gradient-norm contamination near non-stationary path points, and addresses the non-uniform metric parameterization inherent to adaptive path discretization. The distinction between phenomena observed under constrained versus natural dynamics must be clearly articulated, with explicit discussion of how methodological choices limit the scope of inferential claims.

**Core Evaluation Criteria:**
- **Bias Acknowledgment and Discussion:** Does the work explicitly discuss biases introduced by path-finding algorithms (e.g., spring energy minimization, pivot insertion density) or constrained projected dynamics?
- **Cross-Method Consistency:** Are curvature/barrier phenomena reproduced using independent path construction methods (e.g., comparing nonlinear minimum energy paths with linear mode connectivity)?
- **Control of Artifacts:** Does the work control for artifacts such as gradient-norm contamination in Hessian eigenvalue estimation near path endpoints, non-uniform metric parameterization along paths, and projection-induced distortions in constrained optimization?
- **Natural vs. Constrained Dynamics:** Does the work clearly distinguish between phenomena observed under artificially constrained dynamics versus unconstrained training, and discuss the limitations of such constraints?

---

**Principle 5: Theoretical Tractability and Explanatory Power of Reduced Models for Landscape Phenomena**

**Definition:**  
Toy models and analytical derivations serve as essential tools for distilling high-dimensional optimization dynamics into interpretable mechanisms, but they carry significant explanatory obligations. Reviewers expect reduced models to be derived with mathematical rigor, including explicit assumptions about timescale separation, noise characteristics, and dimensionality reduction, rather than presenting opaque or abbreviated derivations. The model must do more than qualitatively reproduce an empirical phenomenon; it should yield precise, falsifiable predictions regarding scaling behavior, force magnitudes, or phase transitions that can be tested in full-scale networks. Authors must clearly map toy-model variables to concrete quantities in deep learning, such as identifying soft modes with low-curvature directions and effective temperature with minibatch noise scale. When such connections are tenuous or the derivation is disconnected from the experimental system, the theoretical contribution is viewed as illustrative rather than explanatory, substantially diminishing its value to the field.

**Core Evaluation Criteria:**
- **Derivation Clarity and Completeness:** Are analytical steps complete and well-justified (e.g., adiabatic elimination of fast variables, effective potential derivation), with assumptions explicitly stated?
- **Predictive Power:** Does the toy model make falsifiable predictions about real networks (e.g., scaling of entropic force with temperature, dependence on curvature derivative) that are subsequently tested?
- **Conceptual Illumination:** Does the reduced model genuinely clarify the mechanism in real networks, or does it merely provide a parallel example? Is the mapping from toy variables to network parameters (soft/stiff modes, effective temperature) clearly established and justified?