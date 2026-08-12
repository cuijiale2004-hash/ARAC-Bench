## Principle 1: Explicit Threat Modeling and Attack-Agnostic Defense Design for Jailbreak/Relearning-Robust LLM Unlearning

**Definition:**
In robust LLM unlearning, any defense claim is only meaningful relative to a precisely specified adversary. Reviewers must first verify that the paper formalizes, for each attack family, the adversary's goal, knowledge (architecture, forget data, training procedure), access pattern (black-box query vs. white-box weight access), and computational budget. In this paper, the absence of a formal threat model was a major criticism; the rebuttal's clarification that jailbreak adversaries are black-box while relearning adversaries hold white-box weight access materially strengthened the work's credibility. A second requirement is attack-agnosticism: a defense whose auxiliary components are trained on attack-specific artifacts (e.g., real jailbreak prompts) implicitly assumes knowledge of the attacker and is impractical, whereas training only on held-out forget/retain splits with generic perturbations is the appropriate standard. A third requirement is awareness of the well-known adversarial-ML phenomenon in which defending against one attack increases vulnerability to others; strong work investigates such trade-offs explicitly rather than ignoring them. This principle is crucial because, without it, robustness claims in unlearning are unfalsifiable and easily overclaimed.

**Core Evaluation Criteria:**
- **Completeness of the threat model:** Are goals, knowledge, access patterns, and budgets formally stated for *each* attack type (e.g., black-box jailbreak vs. white-box relearning)?
- **Attack-agnostic construction:** Does the defense avoid any dependence on attacker-specific prompts, outputs, or algorithms during training of its components (e.g., probes, regularizers)?
- **Cross-attack trade-off analysis:** Does the work examine whether robustness gains against one attack family introduce new vulnerabilities elsewhere?
- **Realism and conservativeness:** Are adversary budgets bounded yet non-trivial, such that failure under them is damning and success is meaningful?

---

## Principle 2: Synergy Verification and Causal Attribution in Compositional Dual-Space Smoothness Designs for LLM Unlearning

**Definition:**
This is the decisive novelty criterion for frameworks that compose multiple robustness mechanisms under a unified formulation. When a paper claims a joint design—for instance, "dual-space smoothness" coupling representation-space and parameter-space objectives—reviewers must determine whether the composition yields synergistic, super-additive benefits or is merely a well-engineered combination of independently known techniques (sharpness-aware minimization, gradient orthogonalization, adversarially trained probes). The meta-review identified exactly this as the outstanding concern: each component was individually motivated and effective, yet the work did not rigorously establish that jointly enforcing both objectives outperforms applying them independently, particularly since the two target attacks operate under different threat models. High-quality work supplies either a formal argument (e.g., composing local Lipschitz bounds via the chain rule so the minimal successful perturbation grows in both spaces simultaneously) or controlled experiments isolating interactions (e.g., ablating the orthogonalized gradient separately for each smoothness component). Without such evidence, the conceptual framing must be discounted even when empirical performance is strong—precisely why this paper was accepted only at poster level despite solid engineering. This principle separates genuinely novel frameworks from "A+B" recombinations of existing ideas.

**Core Evaluation Criteria:**
- **Super-additivity evidence:** Does the work show that the joint objective yields more than the sum of independently applied components, rather than assuming it?
- **Mechanistic linkage:** Is there a formal or geometric argument connecting the components (e.g., sensitivity bounds composing across representation and parameter spaces)?
- **Interaction-isolating ablations:** Beyond removing whole modules, are the *interactions* between components (e.g., gradient decoupling applied per smoothness term) separately ablated for causal attribution?
- **Calibrated framing:** Are novelty claims ("unified," "dual-space") proportionate to the evidence actually provided?

---

## Principle 3: Multi-Objective Balance Auditing Across Forgetting Effectiveness, Utility Preservation, Privacy Leakage, and Over-Refusal in LLM Unlearning

**Definition:**
LLM unlearning is inherently multi-objective, and the field's central difficulty is the trade-off among unlearning effectiveness, post-unlearning utility, privacy leakage, and side effects such as over-refusal. Reviewers should require joint reporting of all axes and principled aggregation (e.g., a normalized geometric-mean composite score) instead of cherry-picked single metrics. Critically, strong performance on one axis can be an *artifact* of failure on another: a model suffering catastrophic utility collapse trivially exhibits low attack success rates and low knowledge memorization, so robustness numbers must be interpreted jointly with utility. High-quality work diagnoses these confounds explicitly—as this paper did when attributing collapsed baselines' "good" memorization scores to incoherent generation rather than to robustness. Reviewers should also probe whether baseline failures are fundamental or mere hyperparameter artifacts: requiring full re-engineering of baselines exceeds standard expectations, but authors should evidence that failures persist under tuned settings or officially released checkpoints. Finally, side effects such as elevated over-refusal on benign safety-adjacent prompts (e.g., XStest) must be measured and honestly discussed, not hidden.

**Core Evaluation Criteria:**
- **Joint multi-metric evaluation:** Are forgetting, utility, and privacy reported together with transparent normalization/aggregation rather than selectively?
- **Confound diagnosis:** Does the analysis distinguish genuine robustness gains from artifacts of utility collapse or over-forgetting?
- **Baseline failure attribution:** Are baseline collapses shown to persist under tuned or officially released configurations, with a mechanistic explanation rather than surface observation?
- **Side-effect auditing:** Are over-refusal, privacy leakage (membership inference), and degradation on benign-adjacent content explicitly measured?

---

## Principle 4: Multi-Axis Adversarial Evaluation Over Attack Steps, Forget-Subset Sizes, Seeds, and Attack Families in Robust Unlearning

**Definition:**
Robustness claims in LLM unlearning are credible only when the evaluation varies along several independent axes: attack strength (relearning steps), attack resources (size of the forget subset available to the adversary), stochasticity (multiple seeds producing different attack subsets), and attack family (prefill, optimization-based adaptive prompts, multi-turn dialogue). This paper's initial submission varied relearning only by step count; reviewer pressure produced 25%/50% subset-size sweeps with distinct seeds, which materially strengthened the conclusions—this should be the expected standard, not an optional addition. Evaluation should further span distinct unlearning scenarios (continuous-text copyright removal versus conversational/behavioral unlearning) and at least two model families to exclude setting-specific effects. Baselines must include recent, competitive robustness-oriented defenses (e.g., representation-misdirection and latent-adversarial-training methods), not only classical gradient-ascent or preference-optimization approaches. Bounded adversary budgets are acceptable when conservative, but the paper must acknowledge that stronger adaptive attackers remain untested. This principle ensures robustness results are not artifacts of one weak or narrow attack configuration.

**Core Evaluation Criteria:**
- **Multi-axis sweeps:** Are relearning attacks varied over steps × subset sizes × seeds, with the actual sampled content differing across settings?
- **Attack-family coverage:** Are multiple distinct jailbreak families (prefill, adaptive/AutoDAN-style, multi-turn) evaluated under consistent protocols and judged outputs?
- **Currency of baselines:** Are recent robustness-focused baselines included and compared under identical inputs and evaluation pipelines?
- **Cross-scenario consistency:** Do conclusions hold across text types (continuous vs. conversational) and across at least two base model families?

---

## Principle 5: Mechanistic Margin Quantification and Approximation Validity for Smoothness-Based Robustness Claims in LLM Unlearning

**Definition:**
Beyond end-task metrics, reviewers should demand direct, quantitative evidence that a proposed regularizer induces the geometric change it advertises. For smoothness-based unlearning defenses, this means measuring representation-space margins (e.g., median and low-quantile signed distances to a probe's decision hyperplane, including tail analysis) rather than asserting robustness intuitively, and providing loss-landscape or gradient-norm evidence for parameter-space flatness. Approximation assumptions must be validated in the actual operating regime: first-order Taylor/FGSM-style linearizations are standard practice (as in SAM), but reviewers should expect either theoretical error bounds (e.g., O(dε²) under Lipschitz-gradient assumptions) or empirical measurements of approximation error under the precise perturbation radii used in experiments—this paper's measured ~0.53% average error is the kind of evidence that converts a contested assumption into a supported one. Mathematical notation must be complete and consistent; undefined symbols and shifting definitions were treated here as a major flaw because they prevent independent verification of the mechanism. Finally, causal attribution must proceed link by link (regularizer → geometric change → robustness outcome), not by black-box inference from final task performance alone.

**Core Evaluation Criteria:**
- **Direct mechanism quantification:** Are margins, curvature, or gradient statistics explicitly measured and compared against baselines, including tail behavior rather than only averages?
- **Approximation validity:** Is the error of first-order/linearized inner maximizations bounded theoretically or measured empirically under the hyperparameters actually used?
- **Notational rigor:** Are all symbols, functions, and loss components formally defined and consistently used, enabling independent reproduction of the derivations?
- **Causal chain completeness:** Does the work demonstrate each link from the proposed objective to the claimed robustness, rather than inferring mechanism from aggregate scores?
