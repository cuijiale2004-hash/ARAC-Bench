## Principle 1: Theoretical and Mechanistic Grounding of Hybrid On-/Off-Policy Update Mixing in Memory-Augmented LLM-RL

**Definition:**
When a paper proposes to stochastically interleave on-policy and off-policy updates—for instance, to distill memory-conditioned ("tips-augmented") behavior back into the base parametric policy—reviewers require more than end-task performance as evidence. The Area Chair explicitly flagged as an *outstanding* weakness that "the core mechanism's theoretical grounding is light" and "the random update mixing... remains a heuristic choice without theoretical justification." A rigorous submission must (a) provide a mechanistic account of *why* the mixture resolves the stability–efficiency trade-off (on-policy for stable learning, off-policy for reward-guided knowledge internalization), (b) specify the importance-sampling correction with mathematical precision under mismatched conditioning contexts (numerator without tips, denominator with tips), and (c) justify the mixing schedule itself, or at minimum demonstrate why a random per-batch schedule is preferable to adaptive or scheduled alternatives. In this subfield, where "architectural combination" papers are common, the presence or absence of such grounding is a primary discriminator between a conceptual contribution and an engineering cocktail.

**Core Evaluation Criteria:**
- **Correctness and transparency of the off-policy correction:** Is the importance-sampling ratio explicitly derived for *every* rollout/update mode combination (e.g., a summary table of current vs. old log-prob conditions), as Reviewer c3bg demanded? Ambiguity here is a red flag for implementation errors.
- **Mechanistic, not post-hoc, rationale:** Does the paper explain *why* distilling tip-conditioned trajectories into the tip-free policy improves exploration and generalization, rather than merely asserting complementarity after observing gains?
- **Justification of the stochastic mixing design:** Is the choice of random mixing (vs. learned, scheduled, or adaptive switching) defended theoretically or empirically, rather than presented as an untested heuristic?
- **Explicit treatment of off-policy instability:** Does the work identify the failure mode (e.g., low-probability tokens amplifying gradients through unbounded likelihood ratios) and validate the stabilization mechanism (e.g., token masking) across multiple seeds, not a single run?

---

## Principle 2: Causal, Beyond-Anecdote Evidence Linking Self-Generated Memory to Enhanced Exploration Behavior

**Definition:**
The central claim of this subfield is that self-generated textual memory ("tips") induces *better exploration*, not merely better scores. The meta-review explicitly states that even after rebuttal, "a deeper explanation of why the memory-prompting leads to better exploration (beyond example-based claims) is still missing." High-quality work must therefore establish the full causal chain—memory content → avoidance of repeated mistakes / broadened action selection → discovery of novel states → improved returns—with evidence at each link. Selected qualitative transcripts (side-by-side rollouts with/without tips) are necessary but insufficient. Reviewers should demand quantitative behavioral signatures of exploration: policy entropy trajectories, state-visitation coverage, rates of repeated failure, and action-diversity metrics. Furthermore, the semantic status of the memory itself must be analyzed: are tips transferable reasoning strategies or task-specific heuristics, and is tip quality itself improving over training? Without this, one cannot distinguish genuine exploratory learning from pattern memorization.

**Core Evaluation Criteria:**
- **Controlled causal comparison:** Are memory-augmented and memory-free rollouts compared under otherwise identical training conditions, isolating memory as the treatment variable?
- **Quantitative exploration metrics:** Does the paper report entropy, novel-state discovery rates, or coverage measures—not only final return—to substantiate the "exploration" claim (e.g., the entropy analysis in Figure 7)?
- **Semantic analysis of generated memory:** Is there a systematic characterization of what kinds of tips help, whether they generalize across tasks, and whether tip generation itself should be (or is) directly optimized?
- **Failure-mode illustration:** Are concrete cases shown where memory-free agents repeat mistakes while memory-augmented agents escape them, establishing the mechanism rather than the correlation?

---

## Principle 3: Completeness of Component Ablation and Hyperparameter Sensitivity Analysis for Mixing Probabilities and Intrinsic Reward Design

**Definition:**
Frameworks in this area introduce multiple interacting new components—memory rollout probability *p*, off-policy update probability *q*, intrinsic reward coefficients and similarity thresholds, token-masking thresholds. All three reviewers independently demanded ablations on exactly these, and the rebuttal's addition of Appendix F (sweeps of *p*, *q*, and intrinsic reward variants) was pivotal in converting borderline scores (6s) into acceptance and raising c3bg's score from 8 to 10. The standard is threefold: (a) **necessity**—removing each component must demonstrably degrade performance (e.g., intrinsic-reward removal causes collapse); (b) **robustness**—the method should tolerate reasonable perturbations of scale and mechanism (e.g., 0.5×/2× reward scaling, substitution with an RND bonus) with effects confined to learning dynamics rather than final performance; (c) **safe operating ranges**—the paper must identify robust hyperparameter intervals (e.g., p ∈ [0.6, 0.75], q ∈ [0.15, 0.5]) rather than silently relying on a single, unexplained default. Single-point heuristic choices without sweeps are a hallmark of low-rigor work in this subfield.

**Core Evaluation Criteria:**
- **Mode-combination ablations:** Is each hybrid mode (on-policy without memory, on-policy with memory, off-policy distillation) individually removed to demonstrate complementarity (e.g., Figure 9)?
- **Systematic sweep of mixing probabilities:** Are *p* and *q* varied across wide ranges, with identification of degradation at extremes and a justified default that prioritizes cross-task robustness over task-specific optima?
- **Intrinsic reward audit:** Is the auxiliary reward tested for necessity (removal), robustness (scaling), and mechanism-dependence (e.g., RND substitution), with effects reported on **final performance**, not only intermediate metrics like entropy?
- **Generalizability risk of reward shaping:** Does the paper acknowledge domain-dependence risks of naive similarity-based novelty signals (e.g., the static-noise-TV pathology from Pathak et al.) when transferring to new environments?

---

## Principle 4: Transparency and Fairness of the Evaluation Protocol, Including Test-Time Memory Conditions, Baseline Provenance, Statistical Reporting, and Compute Matching

**Definition:**
Reviewers in this subfield are highly sensitive to evaluation-protocol ambiguity, because memory-augmented agents can report numbers under fundamentally different conditions (with vs. without memory at test time, varying memory-bank sizes). Reviewer JKue's confusion over which rollout mode produced Table 1's numbers—and the authors' clarification that EMPO² is evaluated *without* memory at test time—illustrates how easily misleading comparisons arise. Additionally, RL methods are "notorious for subtle implementation differences," so baselines copied from prior papers (Naive, Reflexion, GRPO, GiGPO scores imported from Feng et al.) require explicit justification, standardized base models/test sets, and re-implementation where settings mismatch (as done for Retrospex). Statistical rigor is mandatory: missing error bars (Figures 1B, 8; Table 2) and single-seed stability claims (Figure 6) were repeatedly challenged and had to be remedied. Finally, because memory mechanisms add rollout overhead (~19%) and induce longer generations, fair comparison demands performance-versus-compute (GPU-hours) curves, not only performance-versus-steps.

**Core Evaluation Criteria:**
- **Unambiguous test-time protocol:** Does every reported number clearly state whether memory was used at evaluation, the buffer size, and the retrieval configuration—especially when comparing against episodic-memory baselines like Reflexion that *do* use memory at test time?
- **Statistical completeness:** Are multi-seed means with standard deviations reported for all main tables and figures, including OOD adaptation curves and stability claims?
- **Baseline provenance and standardization:** When reusing published numbers, is equivalence of base model, heuristics (e.g., Retrospex's teleport/focus hacks), hyperparameters, and test splits verified—or are mismatched baselines re-run with tuned configurations?
- **Compute-matched fairness:** Is the additional cost of tip generation/retrieval/storage and longer responses quantified, and is a time- or FLOPs-normalized comparison provided to show efficiency gains are not an artifact of larger compute budgets?

---

## Principle 5: Generality of Claims Across Base Models, Model Scales, and Task Domains—Guarding Against Model-Specific and Benchmark-Specific Gains

**Definition:**
A recurring and *unresolved* concern in this paper's assessment is external validity: "All results use Qwen2.5-7B-Instruct. Gains could be partly due to model-specific biases" (meta-review). Memory-and-reflection mechanisms depend heavily on the base model's latent summarization and self-correction abilities, so single-model evidence cannot separate algorithmic contribution from pretraining priors. Similarly, evaluation confined to two text-based benchmarks (ScienceWorld, WebShop) leaves open whether the framework transfers to single-turn reasoning (math/coding, as c3bg urged), multimodal, or robotic domains. Reviewers should calibrate score ceilings to the breadth of evidence: strong papers either (a) demonstrate cross-family and cross-scale consistency (e.g., Llama, Mistral, larger models), (b) show convincing out-of-distribution adaptation under proper controls, or (c) explicitly scope claims and articulate why the chosen benchmarks are sufficient and non-toy. Claims of a "general framework" resting on one model and two benchmarks warrant discounting.

**Core Evaluation Criteria:**
- **Cross-model validation:** Are results replicated across at least two model families or scales, or is the single-model limitation candidly acknowledged with claims appropriately narrowed?
- **Domain diversity relative to claim breadth:** Do the benchmarks exercise the claimed capability (hard exploration, long horizons—e.g., ScienceWorld's 25 action types / ~30-step episodes vs. WebShop's 3 / ~10), and is differential performance across domains explained mechanistically?
- **OOD adaptation protocol:** For cross-task transfer claims (e.g., Biology→Electricity→Chemistry adaptation), are initial-performance differences between compared checkpoints accounted for, and are results averaged over seeds with variance reported?
- **Disentangling priors from algorithm:** Does the analysis address whether gains stem from the method versus the base model's pretrained reflective abilities (e.g., via weaker-model controls or capability-matched baselines)?
