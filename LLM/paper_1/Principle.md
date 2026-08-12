## Principle 1: Pipeline-Global Objective Alignment — Judging Pre-Training Design Choices by Post-SFT Outcomes Rather Than Intermediate Stage Metrics

**Definition:**
Modern LLM construction is a staged pipeline (pre-training → mid-training → SFT/alignment), and design decisions at early stages are conventionally validated using early-stage metrics such as validation loss or zero-shot accuracy. This paper demonstrates that such stage-wise greedy optimization can be actively misleading: decay-based schedules win on pre-training metrics yet lose after SFT, a clear "performance inversion" across stages. High-quality research in this area must therefore treat the final pipeline artifact—the post-training model—as the evaluation target, and must honestly surface cases where intermediate-stage winners become downstream losers. This perspective is critical because the field increasingly recognizes that pre-training quality does not translate linearly into fine-tuned quality. Work that demonstrates gains only at the stage where the intervention is applied, without tracking downstream consequences, provides weak evidence for practical value. Reviewers should reward studies that explicitly formalize the multi-stage objective and design experiments around the end-to-end goal.

**Core Evaluation Criteria:**
- **Final-stage measurement**: Does the work evaluate the complete pipeline output (post-SFT model quality on downstream benchmarks) rather than only pre-training loss or zero-shot metrics?
- **Detection of stage-metric inversion**: Does it explicitly compare method rankings across training stages and report instances where the intermediate-stage winner loses downstream?
- **Honest trade-off disclosure**: Are intermediate-stage regressions (e.g., higher validation loss under WSO) transparently reported alongside final-stage gains, rather than hidden?
- **Formal problem framing**: Is the multi-stage selection problem clearly articulated (joint pipeline optimization vs. stage-wise greedy selection), clarifying why the study targets the terminal objective?

---

## Principle 2: Confound-Controlled Recipe Comparison with Symmetric Downstream Hyperparameter Optimization for Every Variant

**Definition:**
Empirical comparisons of training recipes (LR schedules, optimizers, data strategies) are only credible if the studied factor is cleanly isolated—identical architecture, data, token budget, and optimizer settings—so that observed differences are attributable to the recipe itself. Equally important, and frequently neglected, is that downstream adaptation must be tuned equally thoroughly for every compared variant; otherwise results may reflect unequal tuning effort rather than the recipe. This paper exemplifies the standard: it applies an identical eight-point SFT learning-rate sweep to every pre-trained checkpoint and selects the best per model using a justified criterion (AlpacaEval), explicitly noting that selecting by validation loss would not yield better downstream performance. This symmetry prevents any scheduler from receiving a selective advantage, which is essential given the modest effect sizes typical of such studies. Reviewers should treat recipe comparisons lacking per-variant downstream tuning as fundamentally weak evidence.

**Core Evaluation Criteria:**
- **Variable isolation**: Are architecture, dataset, compute budget, and optimizer held constant so that only the studied schedule varies across conditions?
- **Symmetric downstream tuning**: Is the identical SFT hyperparameter sweep applied to all variants, with independent best-config selection per variant?
- **Justified selection criterion**: Is the downstream model-selection metric principled, disclosed, and defended against alternatives?
- **Verifiable transparency**: Are absolute per-task results and the selected hyperparameters for every condition reported (e.g., in appendices) so fairness can be audited?

---

## Principle 3: Generality Verification of Training-Dynamics Claims Across Parameter Scales, Token Budgets, and Multi-Stage Pipeline Topologies

**Definition:**
Findings about LLM training dynamics are notorious for failing to transfer across model scale or data regime, so robust claims require deliberate variation along these axes. This paper validates its central claim across 1B and 8B models, across standard and heavily over-trained budgets (2T tokens, ~100× Chinchilla-optimal), and across both two-stage and three-stage pipelines that include a mid-training phase with interacting schedule choices (α_pre × α_mid). This design guards against the risk that the phenomenon is an artifact of one scale, one budget, or one pipeline structure. Reviewers should check whether the qualitative trend replicates in every regime tested and whether exceptions or shrinking margins are surfaced rather than suppressed. Single-scale or single-regime evidence supporting universal training-recipe recommendations should be substantially discounted, as scaling behavior is a first-order confound in this subfield.

**Core Evaluation Criteria:**
- **Multi-scale evidence**: Are at least two meaningfully different parameter scales examined, with the qualitative trend holding at each?
- **Budget robustness**: Is the claim tested under both standard and over-trained token regimes where training dynamics may shift?
- **Pipeline-topology generality**: Does the claim survive the insertion of additional stages (mid-training), including interactions between stage-wise schedule choices?
- **Consistency and exception reporting**: Are outcomes shown for all regimes, including configurations where the effect weakens, rather than only favorable subsets?

---

## Principle 4: Mechanistic Attribution via Loss-Landscape Geometry Using Scalable Curvature Estimation and Calibrated Correlation Claims

**Definition:**
Beyond demonstrating an empirical effect, high-quality work explains *why* it occurs by linking the intervention to measurable optimization-geometry properties. This paper attributes WSO's downstream advantage to the preservation of flatter minima, operationalized as Hessian-trace sharpness tracked throughout training on both pre-training and SFT data distributions. Because exact Hessian computation is infeasible at billion-parameter scale, it employs Hutchinson's stochastic trace estimator with Hessian-vector products—an established scalable technique—and discloses its full configuration. It then substantiates the mechanism with a quantitative correlation (Pearson r ≈ −0.71) between pre-training sharpness and SFT performance across all schedules, while explicitly acknowledging the limited sample size and refraining from strong causal claims. This combination—literature-grounded hypothesis, valid at-scale measurement, quantitative supporting evidence, and calibrated language—is the standard reviewers should demand for mechanistic explanations in training-dynamics research.

**Core Evaluation Criteria:**
- **Grounded mechanistic hypothesis**: Is the explanatory variable (flatness → adaptability under distribution shift) motivated by prior optimization and transfer-learning theory?
- **Measurement validity at scale**: Is the curvature estimator appropriate for billion-parameter models, with sampling procedure and computation details fully disclosed?
- **Quantitative mechanism–outcome link**: Is there measured statistical evidence (e.g., correlation across conditions) connecting the proposed mechanism to the downstream outcome?
- **Calibrated causal language**: Does the work distinguish correlation from causation and acknowledge small sample sizes or plausible alternative explanations?

---

## Principle 5: Evidentiary Breadth, Practical Actionability, and Reproducibility Required to Overturn Entrenched Training Conventions

**Definition:**
Claims contradicting near-universal practice—here, LR decay as the de facto standard—carry a heavy evidentiary burden: they must defeat the incumbent across its main families (Cosine, Linear, WSD) and its critical hyperparameter settings (decay to 0% vs. 10% of peak LR), under equal compute. The paper meets this bar by sweeping four schedulers across multiple minimum-LR factors and evaluating on a broad downstream suite spanning instruction following, truthfulness, knowledge, mathematics, reading comprehension, and reasoning. It further converts the finding into actionable guidance—train and release WSO checkpoints to maximize downstream adaptability—addressing a real decision point for practitioners. Reproducibility strengthens the challenge to convention: public datasets, standard Llama-3 architectures, complete hyperparameter tables, and per-task absolute results enable independent verification. Reviewers should reward convention-challenging work only when breadth of evidence, practical utility, transparent reporting, and candid limitation statements (e.g., only SFT studied, not RLHF or preference tuning) are all present.

**Core Evaluation Criteria:**
- **Coverage of incumbent variants**: Are all major incumbent schedule families and their key hyperparameters (decay depth) benchmarked under matched budgets?
- **Downstream evaluation breadth**: Does evaluation span multiple capability dimensions using standard benchmarks with disclosed protocols (shot counts, judge models, reference baselines)?
- **Actionable practitioner guidance**: Does the work translate findings into concrete recommendations for training and model-release strategy?
- **Reproducibility and scope honesty**: Are data, architectures, hyperparameters, and full results publicly detailed, with explicit acknowledgment of scope limits (single post-training stage, model-size ceiling)?