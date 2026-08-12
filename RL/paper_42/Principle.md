**Principle 1: Theoretical Validity of Performance Guarantees Under Undiscounted Finite-Horizon LLM Reasoning Dynamics**

**Definition:**  
In modern LLM reasoning pipelines, policy gradient methods such as GRPO effectively operate under undiscounted, finite-horizon regimes with per-trajectory normalization, which diverges sharply from the classical infinite-horizon discounted MDP assumptions inherited from mainstream reinforcement learning theory. Reviewers must assess whether monotonic improvement guarantees and trust-region bounds are derived under assumptions that genuinely match the LLM training regime, or whether authors import standard RL frameworks that yield vacuous constants as the effective discount factor approaches unity. This principle demands explicit theoretical adaptation to finite-horizon dynamics, including performance difference lemmas and state-distribution shift bounds that scale polynomially in the reasoning horizon length. It further requires authors to transparently discuss the empirical unverifiability of global smoothness preconditions at billion-parameter scale.

**Core Evaluation Criteria:**
- **Regime Alignment:** Are the core theorems proved for the undiscounted finite-horizon setting used in practice, or are they uncritically transplanted from incompatible infinite-horizon frameworks?
- **Constant Tightness:** Do theoretical constants (e.g., the bound constant $C$) scale reasonably with the effective horizon (e.g., $\mathcal{O}(T^2)$), rather than blowing up via terms like $1/(1-\gamma)^2$ that become vacuous in GRPO-style training?
- **Finite-Horizon Adaptation:** Does the paper provide explicit finite-horizon analogues of key lemmas (performance difference, state-distribution shift), or merely assert that they follow trivially from prior work?
- **Assumption Verifiability:** Is there an honest discussion of which theoretical preconditions (e.g., uniform Hessian bounds, bounded step norms) are logically necessary for guarantees but computationally intractable to verify in LLM-scale training?

---

**Principle 2: Empirical Fidelity of Last-Layer Curvature Surrogates to True Optimization Dynamics**

**Definition:**  
Methods that approximate second-order geometry via last-layer or otherwise tractable surrogates must demonstrate that these cheap diagnostics faithfully track the true underlying optimization dynamics. In the LLM setting, where full Hessian or Fisher computation is infeasible, reviewers must evaluate whether proposed curvature estimates (e.g., directional Fisher $m_F$, Hessian $m_H$) correlate with, bound, or at least monotonically rank the ground-truth quantities such as actual KL policy shifts and objective changes. Without such evidence, the mechanism risks becoming a post-hoc rationalization for heuristic data selection rather than a principled trust-region implementation. This principle requires empirical validation that the surrogate's ranking of samples by prospective instability aligns with the true ranking of samples by their destabilizing effect on the policy distribution.

**Core Evaluation Criteria:**
- **Correlation Evidence:** Are rank correlations (e.g., Spearman) reported between surrogate estimates and measured true policy shifts (e.g., forward KL divergence) across training steps and algorithms?
- **Calibration Analysis:** Does the work assess whether thresholds applied to surrogates translate into practical bounds on true quantities, or does it merely assert that monotonic correlation is sufficient without examining the calibration factor?
- **Mechanism Isolation:** Are ablations conducted to show that masking based on curvature is superior to random masking, gradient-norm clipping, or other simpler heuristics for identifying unstable samples?
- **Dynamic Visualization:** Do training logs visually demonstrate that the method induces well-behaved, bounded policy shifts (e.g., suppressed KL spikes) as predicted by the curvature model, rather than only showing improved downstream accuracy?

---

**Principle 3: Computational Tractability and Scalability of Trust-Region Mechanisms Relative to LLM Scale and Baseline Stabilizers**

**Definition:**  
Any proposed stabilization mechanism for LLM policy gradients must be evaluated not only against standard first-order baselines but also against prior trust-region and second-order methods—such as TRPO with conjugate gradients or K-FAC—in terms of wall-clock time, memory footprint, and numerical feasibility. The principle treats scalability to billion-parameter autoregressive models as a first-class constraint: methods requiring iterative matrix solvers, full Fisher materialization, or persistent reference-policy storage may be theoretically elegant yet practically prohibitive. Reviewers must assess whether the new method’s overhead is quantified as negligible (e.g., sub-five-percent step-time increase, sub-gigabyte volatile memory) and whether it successfully replaces existing stabilization components rather than stacking additional complexity atop an already intricate pipeline.

**Core Evaluation Criteria:**
- **Overhead Quantification:** Is wall-clock time and GPU memory overhead reported with granular breakdowns (e.g., curvature estimation, masking, moment updates) relative to a standard training iteration?
- **Comparison with Prior Trust-Region Methods:** Does the paper explicitly contrast its scalability with TRPO-style conjugate-gradient and line-search procedures, or with natural gradient approximations, showing mathematically why they fail at LLM scale?
- **Replacement versus Addition:** Does the method obviate the need for other stabilization mechanisms (e.g., KL regularization, ratio clipping) and their associated costs (reference-model memory, biased gradient estimates), or does it introduce new hyperparameters without removing old ones?
- **Structural Approximation Efficiency:** Are complexity-reducing approximations (e.g., vocabulary sparsity, last-layer restriction) justified with rigorous complexity analysis comparing $\mathcal{O}(Kd)$ directional computations against $\mathcal{O}((Kd)^2)$ full matrix operations?

---

**Principle 4: Breadth of Empirical Evaluation Across Model Families, Task Domains, and Properly Tuned Baselines**

**Definition:**  
Claims of dramatic sample efficiency improvements (e.g., an order of magnitude reduction in training completions) require rigorous substantiation across diverse model families, training datasets, and baseline configurations tuned to their own optimal stable hyperparameters. This principle reflects the concern that aggressive hyperparameter regimes can induce catastrophic collapse in baseline methods while the proposed method survives, creating a misleading comparison unless stable baseline trajectories are also reported. Single-model, single-dataset evaluations severely limit generalizability and may reflect idiosyncratic optimization landscapes rather than robust algorithmic advances. Reviewers must verify that efficiency gains persist against both aggressive and properly tuned conservative baselines, and that evaluation spans multiple reasoning benchmarks beyond the training distribution.

**Core Evaluation Criteria:**
- **Baseline Fairness:** Are headline comparisons made against properly tuned stable baselines at their best hyperparameters, with claims carefully disambiguated between “prevents collapse” and “improves efficiency over stable training”?
- **Model Family Diversity:** Is the method validated on more than one architecture or model family, or are cross-family generalizability claims deferred to future work without justification?
- **Dataset and Benchmark Breadth:** Does training span multiple datasets, or does evaluation at minimum cover diverse downstream benchmarks (mathematical reasoning, STEM, competition problems) to rule out memorization or distribution-specific overfitting?
- **Statistical Rigor:** Are results reported over multiple random seeds with measures of variance, and are sample efficiency metrics defined unambiguously (e.g., completions versus accepted tokens versus wall-clock time)?

---

**Principle 5: Sensitivity and Robustness of Data-Selection Thresholds to Optimizer Dynamics and Training Regimes**

**Definition:**  
Curvature-aware data selection introduces operational thresholds (e.g., $\delta_H$, $\delta_F$) that govern which samples are masked during gradient aggregation. The principle demands thorough analysis of how these thresholds interact with the optimizer’s internal state (e.g., Adam momentum and second-moment estimates), the aggressiveness of the learning regime (high versus low learning rates), and the resulting temporal distribution of rejected tokens. Reviewers must assess whether threshold selection is presented as a principled proxy for theoretical constants or as an opaque manual tuning exercise, whether sensitivity sweeps are reported or convincingly argued to be infeasible, and whether the method maintains low rejection rates throughout training without requiring problem-specific retuning. The fidelity of the planned update step—whether modeled via SGD or Adam—must also be justified, as mismatch can distort the predicted curvature and undermine the selection mechanism.

**Core Evaluation Criteria:**
- **Threshold Sensitivity:** Are ablations or sweeps provided for key masking thresholds across orders of magnitude, demonstrating a stable operating range rather than a brittle optimum?
- **Optimizer Interaction:** Does the work analyze how modeling the update with SGD versus Adam affects threshold selection and training stability, particularly regarding momentum-induced update directions?
- **Rejection Rate Dynamics:** Are token-level or sample-level rejection rates reported over the full training run, confirming that intervention remains minimal (e.g., single-digit percentages) and does not asymptotically censor the data stream?
- **Theoretical-Empirical Bridge:** Are thresholds explicitly framed as practical, tunable surrogates for theoretical constants (e.g., smoothness bounds, trust-region radii), with transparent acknowledgement of the gap between assumed and actual values?
- **Regime Adaptability:** Is the method tested under both aggressive (high learning rate, small batch) and conservative regimes to show that thresholds need not be retuned from scratch when the optimization landscape changes?