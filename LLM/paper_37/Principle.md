**Principle 1: Theoretical Surrogate Validity and Mechanistic Transferability from Linear ICL to Non-Linear LLM CAT**

**Definition:**  
Research that employs simplified theoretical surrogates—such as linear self-attention with embedding modules trained on in-context linear regression—to explain complex real-world phenomena must rigorously justify the surrogate's representational adequacy. The linearized model abstracts away critical features of large-scale LLMs, including deep non-linear transformations, multi-layer interactions, and discrete token-space dynamics, so the burden of proof lies in demonstrating which mechanistic insights transfer and which do not. A high-quality study should explicitly state its modeling assumptions, provide empirical or theoretical bridging arguments (e.g., citing prior work that shows linear attention shares optimization properties with non-linear variants), and validate derived hypotheses on full-scale LLMs to confirm external validity. Simply observing that a theory-inspired algorithm improves empirical performance is necessary but not sufficient; the paper should also delineate the boundaries of the approximation. Without this reflective analysis, reviewers cannot assess whether the theoretical conclusions are causal explanations or artifacts of the simplified setting.

**Core Evaluation Criteria:**
- **Explicit Assumption Enumeration**: Are all key assumptions (e.g., symmetric initialization, Gaussian inputs, linearity) clearly listed, motivated, and cross-referenced to prior theoretical literature?
- **Empirical External Validation**: Does the paper validate that insights derived from the linear surrogate (e.g., singular-value regularization) consistently improve real LLMs across multiple model families and attacks?
- **Surrogate-to-Target Correspondence**: Is there a clear mapping drawn between each component of the surrogate (e.g., ICL input points) and the target system (e.g., token sequences, embedding layers)?
- **Limitation Disclosure**: Does the work candidly discuss discrepancies between the surrogate and real LLMs (e.g., inability to capture discrete token-space search dynamics) and identify which conclusions are speculative versus proven?

---

**Principle 2: Attack Model Alignment and Perturbation Space Consistency in Robustness Guarantees**

**Definition:**  
A robustness guarantee is only as practically meaningful as the threat model under which it is derived. Theoretical analyses often rely on bounded-norm perturbations applied to embeddings or suffixes of in-context inputs, whereas real-world jailbreak attacks like GCG synthesize entirely new unbounded token sequences and prompt-level attacks like PAIR rewrite the full input. Consequently, a rigorous evaluation must scrutinize whether the proven attack model faithfully reflects the discrete, potentially unbounded, and semantic-preserving manipulations that attackers actually employ. The paper is expected to explicitly articulate any discrepancies—such as the difference between perturbing the last M tokens versus adding arbitrary suffixes—and discuss how the theoretical insights might or might not generalize to these stronger threat models. Failure to address this alignment risks overstating the practical security provided by the defense.

**Core Evaluation Criteria:**
- **Threat Model Fidelity**: Does the theoretical attack model (e.g., bounded ℓ₂ embedding perturbations, suffix length constraints) structurally resemble the actual attacks used in empirical evaluation (e.g., GCG, PAIR)?
- **Discrepancy Analysis**: Does the paper contain a dedicated discussion of mismatches between the theoretical perturbation space and real attack capabilities, including unbounded token addition and full-prompt rewriting?
- **Qualified Generalization Claims**: Are claims about jailbreak robustness appropriately scoped to the proven attack model, avoiding implicit over-generalization to all token-space adversaries?
- **Empirical Verification Under Mismatched Threats**: Even if theory uses a restricted model, do experiments include attacks (e.g., prompt-level rewrites) that violate those restrictions to stress-test generalization?

---

**Principle 3: Tightness and Non-Vacuousness of Robust Generalization Bounds**

**Definition:**  
When a paper derives a robust generalization bound to explain an empirical phenomenon, the bound must be shown to be non-vacuous and sufficiently tight to yield actionable insights. A bound that is orders of magnitude looser than trivial guarantees, or one whose terms are inestimable in practice, provides little scientific value. The evaluation should therefore demand either theoretical evidence of tightness (e.g., matching lower bounds, explicit constants, or dependence on salient problem parameters) or empirical calibration (e.g., comparing bound values against measured robust risk across varying perturbation radii or model configurations). Furthermore, if the analysis proceeds via a surrogate upper bound on the training objective, the quality of that surrogate—specifically, how closely it tracks the original adversarial loss—must itself be quantified or at least argued rigorously. This ensures that optimizing the surrogate is a reliable proxy for improving true robustness.

**Core Evaluation Criteria:**
- **Empirical Calibration**: Are the predicted bound trends (e.g., negative correlation with perturbation radius, dependence on singular values) quantitatively compared against observed empirical robustness curves?
- **Surrogate Loss Fidelity**: Is there analysis of how tight the surrogate objective is relative to the original adversarial training loss, and does minimizing the surrogate demonstrably minimize the original?
- **Actionability Verification**: Does direct optimization of the bound’s proxy (e.g., singular-value concentration) empirically translate into measurable robustness gains, confirming the bound is not merely an analytical artifact?
- **Asymptotic Behavior**: Does the paper analyze how the bound behaves in relevant limits (e.g., large perturbation radius, high-dimensional embeddings) and whether those limiting behaviors match empirical observations?

---

**Principle 4: Principled Derivation and Optimization Alignment of Theory-Inspired Regularization**

**Definition:**  
Translating a theoretical bound into a practical training objective requires more than heuristic intuition; the regularization term must be demonstrably linked to the mechanism identified in the theory. If the bound suggests that robustness depends on the singular values of the embedding matrix being neither too large nor too small, then the proposed algorithmic penalty should ideally correspond to minimizing that bound or a rigorously derived proxy thereof. The review process must verify that the chosen regularization (e.g., variance of singular values) is not an ad-hoc design but is instead justified as optimizing the theoretical quantity of interest, or at least that alternatives are systematically compared. Additionally, because adversarial training landscapes are notoriously sensitive to hyperparameters, the analysis must include ablations on the regularization coefficient and perturbation radius to disentangle genuine methodological improvements from incidental tuning effects.

**Core Evaluation Criteria:**
- **Theoretical-to-Algorithmic Traceability**: Can the reader trace the regularization term directly back to a specific term or condition in the robust generalization bound?
- **Comparative Regularization Ablation**: Does the paper experimentally compare the proposed regularization against structurally simpler alternatives (e.g., max-min singular-value difference, direct spectral norm constraints)?
- **Hyperparameter Sensitivity Analysis**: Are the regularization strength and perturbation radius ablated to show that improvements are robust to these choices and not artifacts of a narrow, hand-tuned setting?
- **Computational Overhead Transparency**: Is the computational cost of the regularization term reported relative to the baseline, ensuring practicality is not sacrificed for marginal theoretical gains?

---

**Principle 5: Comprehensive and Nuanced Evaluation of Robustness-Utility Tradeoffs**

**Definition:**  
Evaluating an adversarial defense solely through isolated robustness and utility tables is insufficient for judging whether a method achieves a favorable tradeoff. A rigorous study must characterize the entire Pareto frontier by varying control parameters—such as the embedding perturbation radius or regularization coefficient—and plotting robustness against utility on ROC-style curves. This enables direct comparison of methods at equivalent operating points and reveals whether gains in one metric come at a disproportionate cost in the other. Moreover, because security guarantees require lower-bound reasoning, the evaluation should report worst-case metrics (e.g., success rate if any trial succeeds, ensemble success across all attack families) alongside averages. Only through this multi-dimensional, visually intuitive, and conservatively measured analysis can reviewers ascertain whether a proposed method truly advances the state of the art in safe and usable LLM deployment.

**Core Evaluation Criteria:**
- **Pareto Frontier Visualization**: Does the paper provide ROC-style or scatter plots mapping utility (e.g., LC-WinRate) against robustness (e.g., ASR) across varying hyperparameters or training configurations?
- **Worst-Case Robustness Metrics**: Are lower-bound metrics reported, such as maximum-over-trials attack success rate (1@5 ASR) and ensemble attack success across the full attack suite?
- **Controlled Comparison Fairness**: When comparing methods (e.g., CAT vs. ER-CAT), are they evaluated under identical conditions (same epochs, learning rate schedules, base implementations), and is any deviation explicitly justified to prevent confounding?
- **Sensitivity to Operating Points**: Does the analysis vary key hyperparameters (e.g., loss cut-offs, perturbation radii) to show how the operating point moves along the robustness-utility frontier?