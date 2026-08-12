## Principle 1: Stage-Wise Component Isolation Ablations Against Trivial Compute-Scaling Controls in Multi-Step Rollout-Reuse Update Rules for On-Policy LLM Reasoning RL

**Definition:**
When a method restructures the standard one-shot on-policy update into multiple coordinated stages (e.g., a fast inner trajectory of K updates on the same batch, a reposition/interpolation step, and a slow correction), it inherently confounds two explanations for its gains: "the proposed mechanism works" versus "we simply performed more gradient steps per rollout batch." Reviewers in this case explicitly demanded controls such as GRPO-K (K inner updates without reposition or slow correction), Stage-II-only, and Stage-III-only variants, and the absence of these controls was a primary driver of a reject score. High-quality work must therefore demonstrate that each claimed component is causally necessary—for example, that removing the reposition step leads to progressive degradation or outright training collapse as K grows and off-policy drift accumulates. Crucially, such ablations must hold the total gradient-update budget fixed, so improvements cannot be attributed to extra per-step computation alone. This principle is highly discriminative in this subfield because many proposed "stabilizers" reduce, upon scrutiny, to compute amplification. It separates genuine algorithmic insight from trivially reusing data more times.

**Core Evaluation Criteria:**
- **Completeness of stage isolation**: Does the work ablate each stage individually and in combination, including the critical "K inner updates only" control that isolates the drift-control and correction components?
- **Fixed compute budget in ablations**: Are comparisons designed so that gains cannot be explained merely by performing more gradient updates per batch?
- **Stress-test evidence of necessity**: Does removing the key mechanism (e.g., interpolation) cause measurable degradation or collapse, particularly in aggressive regimes (large K), rather than only in benign regimes?
- **Interaction analysis**: Are interactions between components and hyperparameters (e.g., α × K) mapped, rather than varying one knob at a time?
- **Replication across settings**: Is the component-necessity evidence shown on more than one model, dataset, or task to rule out setting-specific artifacts?

---

## Principle 2: Matched-Compute Efficiency Verification via Wall-Clock/FLOPs-Equalized Curves and Multi-Target Fixed-Accuracy Comparisons in Rollout-Dominated Reasoning RL Pipelines

**Definition:**
In GRPO-style training for LLM reasoning, rollout generation dominates per-step cost (~70% of wall-clock), while the policy update is a minority share; hence any method that multiplies update compute (K inner backward passes) to save rollouts is making an explicit tradeoff that must be audited rather than asserted. Reviewers required comparisons at matched wall-clock or FLOPs, at multiple fixed accuracy targets, and rejected reliance on a single, potentially cherry-picked endpoint such as "the baseline's best accuracy." Strong work decomposes per-step costs transparently (rollout vs. update vs. overhead), quantifies the actual per-step multiplier (e.g., ~1.2–1.4×), and reports auxiliary resources such as GPU memory. Efficiency claims should be expressed as full accuracy-versus-time curves spanning a wide range of budgets, plus speedups measured at several accuracy thresholds (e.g., 3–6.6× across four targets). This principle distinguishes substantive sample-efficiency contributions from accounting illusions, and in this case the rebuttal's matched-time tables were decisive in converting borderline reviewers.

**Core Evaluation Criteria:**
- **Equalized-budget comparisons**: Are accuracy-versus-wall-clock (or FLOPs) curves provided across a broad range of time budgets, showing the method dominates at essentially all budgets rather than selected points?
- **Multi-target speedup measurement**: Is time-to-accuracy reported for several fixed accuracy levels, avoiding dependence on a single favorable target?
- **Honest overhead accounting**: Is the increase in per-step compute from extra updates explicitly acknowledged, decomposed, and bounded, with justification for operating in a modest-overhead regime (e.g., small K)?
- **Full resource reporting**: Are memory and engineering overheads profiled and disclosed alongside time and rollout counts?
- **Tradeoff framing**: Are efficiency claims framed as an audited tradeoff (extra update compute exchanged for rollout savings) rather than an unqualified "free lunch"?

---

## Principle 3: Mechanistic Attribution of Stability Gains and Evidence-Bound Interpretation of Training Dynamics (Off-Policy Drift, Entropy, Response Length) in GRPO-Family Optimization Enhancements

**Definition:**
Methods claiming to "stabilize" on-policy reasoning RL must substantiate the mechanism, not merely the outcome: reviewers asked why an interpolation-based reposition should be preferred to the standard KL-penalty constraint, whether gains merely stem from handling zero-advantage batches, and whether lower entropy reflects "more efficient exploration" or premature convergence to a deterministic policy. High-quality work answers these questions with direct, causal experiments: comparing the proposed drift-control mechanism against conventional alternatives under cumulative drift (where KL regularization acts only locally and cannot reset accumulated mismatch—empirically collapsing at large K), composing the method with algorithms like DAPO that already remove specific variance sources to demonstrate orthogonality, and supporting dynamic interpretations with convergent evidence (reward curves, response-length regulation, diversity/accuracy trends). Convenient narratives that reinterpret any observed dynamic as beneficial—without ruling out alternative explanations—are penalized. This principle separates papers that understand *why* their method works from those that only show *that* it works.

**Core Evaluation Criteria:**
- **Mechanism-vs-alternative experiments**: Is the proposed control mechanism (e.g., interpolation reposition) empirically compared against standard constraints (KL penalty, clipping) in the identical multi-step reuse setting, including failure regimes?
- **Disentanglement of confounded variance sources**: Are experiments provided (e.g., applying the method atop DAPO) showing gains persist when other known instability sources, such as zero-advantage batches, are already mitigated?
- **Training-dynamics diagnostics**: Do reward, entropy, response-length, and accuracy trajectories jointly support the stability claim, including collapse/degradation analysis in ablated variants?
- **Balanced interpretation of dynamics**: Are alternative explanations for observed signals (e.g., entropy drop as exploitation-vs.-exploration) explicitly considered and adjudicated with evidence?
- **Causal chain, not correlation**: Is a coherent causal narrative established from mechanism → training dynamics → final metrics, rather than inferring stability solely from improved final accuracy?

---

## Principle 4: Claim–Evidence Alignment in Theoretical Framing and Empirical Reporting for Update-Rule Innovations in LLM Reasoning RL

**Definition:**
This subfield frequently accompanies algorithmic proposals with informal optimization analyses (descent-lemma bounds, local quadratic models, curvature-based intuitions), and reviewers hold the framing of such analyses to a strict standard of honesty. In this case, the meta-review explicitly flagged that "the contribution on theoretical analysis was over-claimed": reviewers demanded that all quantities in the stated bound (the constant c, the residual F(K, α)) be precisely defined, that assumptions (L-smoothness, step-size bounds, clipping/KL handling, noise correlation, behavior under negative curvature) be stated, and—failing that—that the section be explicitly relabeled from "theoretical intuition" to "intuition," which the authors ultimately did. The same alignment standard applies to empirical reporting: inconsistencies between narrative claims and table values (e.g., +1.80/+0.8 claimed vs. +0.83/+2.57 actual), duplicated references, and ambiguous terminology ("fast"/"slow" naming) were all raised and materially affected scores. Simplified analytic models must also acknowledge their regime of validity and their limited fidelity to real LLM training dynamics. Claim–evidence alignment is a trust criterion: once reviewers find over-claimed theory or sloppy numbers, all other results are discounted.

**Core Evaluation Criteria:**
- **Precision of formal statements**: Are all symbols, constants, and residual terms in any bound or lemma defined, with assumptions and validity regimes (step size, curvature sign, smoothness, noise structure) made explicit?
- **Honest labeling of analytic strength**: Is heuristic reasoning clearly marked as intuition rather than guarantee, with no over-claiming of theoretical contributions in the abstract, contributions list, or section headings?
- **Theory–empirics consistency**: Do the analysis's qualitative predictions (e.g., F(K, α) grows with K and quadratically in α) actually match the observed hyperparameter ablations?
- **Numerical and referential fidelity**: Are all in-text numbers consistent with tables, and is the reference list curated without duplicate or misleading entries?
- **Terminological clarity**: Are potentially ambiguous names and notations defined precisely enough to prevent misreading of the method's properties?

---

## Principle 5: Cross-Domain, Cross-Algorithm, and Cross-Scale Generalization with Hyperparameter Robustness for Plug-Compatible Optimization Layers in Reasoning RL

**Definition:**
A method marketed as a plug-and-play, orthogonal layer above GRPO-family algorithms assumes a correspondingly broad burden of proof: the meta-review's leading concern was precisely that "the generalization of the proposed approach should be verified by experiments," and the rebuttal's additions—coding-task training evaluated on LiveCodeBench, composition with DAPO, and cross-domain hyperparameter ablations—were central to acceptance. Evaluation confined to mathematical reasoning with a single base algorithm is the field's default setting for fair comparison, but it is insufficient to substantiate orthogonality claims; reviewers expect at least one additional domain (code, logic), one additional base algorithm (DAPO, GSPO), and coverage of multiple model scales (1.5B–7B) and dataset scales (24K–105K). Equally important is hyperparameter robustness: the sensitivity landscape over the method's knobs (inner steps K, reposition factor α, entropy-trigger schedules) must be mapped, and evidence must show that a single default (e.g., K=3, α=0.8) transfers across domains within a robust operating region rather than requiring delicate per-task tuning. Adaptive components, such as entropy-triggered schedule switches, require their own ablation and cross-domain validation. Finally, honest scoping—acknowledging, for instance, that multi-turn agentic settings remain future work—strengthens rather than weakens credibility.

**Core Evaluation Criteria:**
- **Domain generality**: Is the method validated beyond the primary benchmark family (e.g., coding in addition to math), with consistent trends across domains?
- **Algorithm generality**: Is the claimed orthogonality demonstrated by composition with at least one additional GRPO-family variant (DAPO, GSPO, etc.), showing gains do not depend on the base algorithm's internals?
- **Scale generality**: Are results reported across multiple model sizes and training-data scales/distributions?
- **Hyperparameter robustness and transferability**: Are sensitivity maps for key hyperparameters provided across domains, with evidence of a shared, robust default region and documented interaction effects (e.g., larger K requiring smaller α)?
- **Honest scoping and adaptive-component validation**: Are scope boundaries (single-turn vs. agentic multi-turn) stated explicitly, and are scheduling/adaptive mechanisms ablated and justified rather than assumed beneficial?