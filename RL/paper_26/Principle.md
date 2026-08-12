**Principle 1: Reliability and Calibration of Learned Dynamics-Value Extrapolation for Selective OOD Classification in Offline RL**

**Definition:**  
For methods that classify OOD actions using learned world models and value estimators, the evaluation must rigorously assess whether the classification remains trustworthy in out-of-support regions. Since offline RL agents cannot interact with the environment, any dynamics-model-based or value-comparison-based discrimination on OOD inputs constitutes an unverified extrapolation that risks systematic bias amplification or self-reinforcing errors. A rigorous study must empirically demonstrate that the classifier does not collapse, misclassify, or amplify errors during policy optimization, and must provide quantitative evidence rather than relying solely on final performance improvements. Furthermore, the work should address potential circular dependencies between detection and evaluation modules, ensuring that pretrained and frozen models do not create iterative feedback loops that amplify approximation errors. The principle emphasizes that selective regularization is only as reliable as the weakest link in its extrapolation chain, making dynamics-model validation and uncertainty gating essential review criteria.

**Core Evaluation Criteria:**
- **Quantitative OOD Detection Metrics:** Does the work report standard detection metrics (AUROC, precision/recall, F1) on held-out or synthetic OOD data with known labels?
- **Dynamics Model Sensitivity:** Are there ablations or sensitivity analyses showing performance degradation (or stability) when the dynamics model has varying accuracy or when early-stage checkpoints are used?
- **Uncertainty Quantification & Gating:** Does the method incorporate and validate uncertainty estimation (e.g., ensemble variance) to gate unreliable predictions, and does the gate meaningfully filter ambiguous cases?
- **Visualization and Correlation Analysis:** Are qualitative visualizations (e.g., Q-value distributions for different action classes, state-action heatmaps) and quantitative correlations (e.g., between reconstruction error and true/log-likelihood) provided to justify the classifier's signal?

---

**Principle 2: Theoretical Non-Triviality and Asymptotic Error Characterization Under Model and Detection Approximation**

**Definition:**  
Theoretical contributions in offline RL value regularization must move beyond elementary contraction mapping properties or trivial boundedness statements to provide meaningful, non-vacuous performance guarantees. Specifically, for methods involving approximate OOD detectors and learned dynamics models, the theory should explicitly bound the performance gap relative to the optimal policy as a function of detection misclassification error, dynamics approximation error, and function approximation error. Such asymptotic or finite-sample analyses are essential to establish that the method is principled rather than heuristic, and to justify the design choices under realistic conditions where auxiliary models are imperfect. Reviewers should look for proofs that decompose error into interpretable terms with clear physical meaning, avoiding bounds that merely restate global minima or maxima without linking to the method's selective mechanisms. Ultimately, the theory must support the claim that the algorithm approaches optimal behavior as its constituent approximations improve.

**Core Evaluation Criteria:**
- **Beyond Standard Contraction:** Does the theoretical analysis provide more than a proof that the Bellman operator is a γ-contraction and admits a fixed point?
- **Explicit Error Decomposition:** Are the bounds decomposed into interpretable error terms (e.g., dynamics error, detection error, function approximation error) with clear meaning regarding model mismatch and detection failure?
- **Asymptotic Optimality:** Does the theory show that as auxiliary model errors vanish, the learned policy approaches the optimal policy (or a well-defined reference policy)?
- **Tightness and Practical Interpretability:** Do the bounds avoid trivialities (e.g., bounds that merely state Q-values lie between global min and max without relating to the method's specific mechanism), and do they offer insight into hyperparameter choices (e.g., compensation weights)?

---

**Principle 3: Decoupled Validation of OOD Detection and Disentanglement from End-to-End Performance Gains**

**Definition:**  
Research proposing new OOD detection mechanisms for offline RL must empirically disentangle the detection capability itself from downstream policy optimization gains. It is insufficient to infer detector quality solely from improved task returns, as performance improvements may stem from confounding factors such as additional regularization, model capacity, or implicit policy constraints. High-quality work should validate the detector in isolation using synthetic OOD probes, distributional shift benchmarks, or correlation with likelihood estimates to prove that the detection signal is well-calibrated and discriminative. Furthermore, the method must demonstrate generalization beyond narrow task domains to establish that the detection mechanism is robust across diverse data distributions and action spaces. This principle demands that detection innovation be backed by standalone metrics rather than end-to-end performance alone.

**Core Evaluation Criteria:**
- **Isolated Detector Benchmarking:** Are there controlled experiments where the detector is evaluated independently of the RL loop (e.g., synthetic Gaussian-noise OOD, standard OOD detection benchmarks)?
- **Correlation with Ground-Truth Likelihood:** Does the detection score correlate with true or approximate negative log-likelihood (or other density proxies) to validate it as a distributional distance metric?
- **Domain Generalization:** Is the detector validated across diverse environments or dataset types (e.g., locomotion vs. manipulation vs. navigation) to ensure it is not overfitted to a specific action/space structure?
- **Ablation of Detection vs. Regularization:** Are there ablations that isolate the contribution of the detection signal from the selective regularization strategy (e.g., uniform penalty with the same detector vs. selective treatment)?

---

**Principle 4: Practical Deployability and Hyperparameter Robustness Under Dataset Distribution Heterogeneity**

**Definition:**  
Offline RL methods intended for real-world deployment must balance algorithmic sophistication with practical feasibility, requiring careful scrutiny of architectural complexity and hyperparameter sensitivity. Reviewers evaluate whether the framework introduces excessive auxiliary models, prohibitive computational overhead, or brittle threshold tuning that requires environment-specific grid search. High-quality research should demonstrate that hyperparameters can be set based on principled, observable dataset characteristics rather than opaque per-task tuning, and that the method remains stable across heterogeneous dataset qualities. The work must clearly report computational costs and justify architectural choices, controlling for confounding advantages such as additional critic networks or ensemble sizes. Ultimately, practical deployability hinges on showing robust, predictable behavior under varying conditions without exhaustive manual reconfiguration.

**Core Evaluation Criteria:**
- **Architectural Parsimony and Compute Cost:** Is the computational overhead of auxiliary models (pretraining time, inference cost, number of forward passes) clearly reported and justified relative to the gains? Are the models frozen during RL training?
- **Hyperparameter Adaptability:** Can key thresholds (e.g., OOD percentiles) and coefficients (penalty, compensation) be determined from dataset statistics (e.g., IQR/Median of reconstruction errors) rather than exhaustive per-task search?
- **Sensitivity and Stability:** Are sensitivity analyses provided for key hyperparameters (thresholds, penalty coefficient, compensation coefficient, number of samples), showing stable performance across a reasonable range?
- **Fair Comparison Control:** Does the work control for architectural advantages (e.g., number of critics, model capacity) to ensure performance gains derive from the core algorithmic innovation rather than increased compute or parameters?

---

**Principle 5: Nuanced Treatment of Out-of-Distribution Actions Beyond Uniform Pessimism or Binary ID/OOD Classification**

**Definition:**  
A central conceptual advance in modern offline RL is the shift from uniform pessimism toward fine-grained, outcome-aware regularization that distinguishes harmful extrapolations from beneficial explorations. Reviewers assess whether the proposed selective mechanism is theoretically grounded and empirically validated, moving beyond heuristic bonus or penalty designs. The work must demonstrate that its selective treatment is formally integrated into the learning objective in a way that preserves convergence guarantees, rather than being an ad-hoc adjustment. It should also provide empirical evidence that the classification of OOD actions into beneficial and detrimental categories aligns with dataset characteristics and produces measurable behavioral differences during training. Strong contributions in this space show improved performance on suboptimal datasets where extrapolation is valuable, while maintaining safety on narrow, near-optimal data.

**Core Evaluation Criteria:**
- **Conceptual Distinctness:** Does the method clearly differentiate itself from uniform penalization (e.g., CQL, uniform support penalties) and from binary ID/OOD constraints (e.g., standard behavior cloning or filtering)?
- **Theoretical Integration:** Is the selective regularization mechanism (bonus/penalty) formally incorporated into the Bellman backup or policy objective with convergence guarantees, rather than being an ad-hoc post-hoc adjustment?
- **Empirical Evidence of Selectivity:** Does the work report the empirical proportions of beneficial vs. detrimental OOD actions during training, and do these proportions align with dataset characteristics (e.g., broader support leading to more beneficial OOD)?
- **Performance Pattern Consistency:** Are the strongest gains observed on datasets where selective exploration is most needed (e.g., medium, medium-replay, random), with stable or conservative behavior on narrow expert datasets?