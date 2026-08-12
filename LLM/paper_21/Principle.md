**Principle 1: Scope Boundary Generalization and In-Scope Detection for Reasoning-Guided LLM Unlearning**

**Definition:**  
In reasoning-guided LLM unlearning, the fundamental objective is to remove not merely specific training instances but entire equivalence classes of knowledge—what is termed the unlearning scope. Reviewers consistently emphasize that prior gradient-ascent methods fail because they lack explicit scope specification, leading to either incomplete removal or excessive forgetting. A high-quality method must therefore demonstrate that its reasoning traces capture the underlying conceptual boundaries of the forbidden knowledge, enabling the model to recognize paraphrased, translated, or indirectly formulated in-scope queries. Without such scope-aware generalization, the unlearned model merely memorizes surface-level refusal templates, which collapse under linguistic variation and fail to distinguish benign out-of-scope content. Consequently, evidence of cross-lingual and paraphrase robustness is essential to validate true conceptual unlearning rather than superficial data suppression. The ability to maintain precise scope discrimination directly determines whether the method solves Problem 2 (scope unlearning) or only Problem 1 (data unlearning).

**Core Evaluation Criteria:**
- **Cross-lingual and paraphrase generalization:** Does the unlearned model refuse semantically equivalent in-scope queries expressed in different languages or rephrased forms, rather than only literal training examples?
- **Out-of-scope preservation:** Does the model maintain high accuracy and specificity on benign queries that are superficially similar but conceptually outside the unlearning scope, avoiding over-refusal?
- **Ablative evidence for reasoning:** Do ablation studies show that removing reasoning traces causes the model to collapse into rigid, template-based refusals with degraded specificity, proving reasoning is causal for boundary detection?

---

**Principle 2: Mitigation of Circular Evaluation and Proxy Overfitting in Target-Generated Unlearning Frameworks**

**Definition:**  
When unlearning targets and evaluation protocols are synthesized by external reasoning LLMs, the unlearned model risks learning the stylistic biases of the generator or receiving artificially inflated scores from a cognate evaluator. This creates two interlocking threats: proxy overfitting, where performance stems from mimicking the target model's idiosyncrasies, and circular evaluation, where the judge favors outputs structurally similar to its own training distribution. To establish methodological credibility, a study must disentangle the target-generation model, the evaluation model, and the unlearned model family, ensuring they are architecturally and distributionally distinct. Empirical validation should report low variance across multiple target generators and evaluators, demonstrating that improvements are intrinsic to the unlearning design rather than artifacts of model homophily. Furthermore, the authors should explicitly discuss whether the generator and evaluator share training corpora or architectures with the unlearned backbone. Without such cross-model validation and transparency, claims of superior unlearning quality remain confounded by shared inductive biases and cannot be trusted as generalizable contributions.

**Core Evaluation Criteria:**
- **Architectural and distributional disjointness:** Are the target-generation LLM, evaluation LLM, and unlearned backbone drawn from distinct model families or training pipelines?
- **Cross-generator performance variance:** Does the method maintain stable unlearning and retention scores when targets are produced by qualitatively different reasoning models?
- **Cross-evaluator consistency:** Do evaluation scores remain consistent when different judge LLMs are used, confirming that improvements are not artifacts of evaluator bias?

---

**Principle 3: Ethical Safety Control and Prevention of Harmful Knowledge Leakage in Reasoning-Trace Generation**

**Definition:**  
Reasoning-based unlearning targets are intended to explain why a query falls within the unlearning scope, yet this very process can inadvertently reintroduce harmful, copyrighted, or private knowledge into the supervision signal. If the reasoning trace elaborates on the sensitive content it seeks to suppress, the method violates the safety intent of unlearning and may leak restricted information into the training pipeline. Therefore, the evaluation of such methods must scrutinize the target-generation pipeline for explicit safeguards, including system-prompt restrictions, content filters, redaction protocols, and manual spot-checks that verify traces contain only scope-boundary analysis. Reviewers must demand transparency regarding generation hyperparameters, filtering rules, and seed settings to ensure reproducibility and auditability. The absence of such protective measures undermines the claim of "explainable" unlearning, transforming it into a liability. Ultimately, a reasoning-guided unlearning method is only ethically viable if it can demonstrate that its explanatory traces are semantically safe and do not function as Trojan horses for the very knowledge they claim to erase.

**Core Evaluation Criteria:**
- **Absence of sensitive content in traces:** Do manual inspections or automated filters confirm that reasoning traces describe scope boundaries without reproducing harmful, private, or copyrighted details?
- **Pipeline transparency:** Are the full target-generation parameters—prompt templates, temperature, top-p, seed, filtering rules, and redaction steps—documented for reproducibility and safety auditing?
- **Ethical risk disclosure:** Does the paper explicitly discuss potential leakage risks and justify why the chosen generation protocol is unlikely to reintroduce undesirable knowledge?

---

**Principle 4: Comprehensive Multi-Metric Validation and Hyperparameter Sensitivity for Utility-Preserving Unlearning**

**Definition:**  
A rigorous unlearning study cannot rely solely on a single proprietary evaluation framework, such as LLM-as-a-Judge, without validating its conclusions against established community benchmarks and external capability tests. Reviewers expect convergence between novel metrics and standard metrics—such as WMDP accuracy, MMLU retention, and ES scores—to confirm that observed trade-offs are real rather than artifacts of a particular judging protocol. Additionally, because unlearning methods balance knowledge removal against general utility, the sensitivity of balancing hyperparameters must be characterized through systematic sweeps, with failure modes identified at extreme values. The method should also be stress-tested on out-of-domain general capabilities like mathematical reasoning to ensure that retention scores reflect broad competence rather than narrow benchmark fitting. Authors must further demonstrate that their evaluation protocol is robust to superficial perturbations such as answer reordering. Only through this multi-layered, cross-validated evaluation can authors substantiate claims of achieving an optimal unlearning-retention frontier.

**Core Evaluation Criteria:**
- **Convergence across metric paradigms:** Do the reported improvements hold under both the proposed evaluation framework and standard community metrics such as WMDP accuracy, ES score, or MMLU retention?
- **General utility stress-testing:** Are external capability benchmarks (e.g., GSM8K for mathematical reasoning, broad MMLU slices) reported to verify that retention reflects general competence rather than benchmark overfitting?
- **Hyperparameter sensitivity and failure modes:** Is a sensitivity sweep provided for key trade-off parameters, with clear identification of regimes leading to collapse, incomplete erasure, or excessive forgetting?

---

**Principle 5: Robustness Against Jailbreak, Relearning Attacks, and Dynamic Scope Evolution in Unlearned Models**

**Definition:**  
Unlearning is practically worthless if forgotten knowledge can be trivially recovered through jailbreak prompts, cross-lingual restatement, or brief relearning fine-tuning. Therefore, reviewers must require evidence that the method withstands adaptive attacks, with explicit baseline comparisons under identical threat models showing relative resilience. Beyond static adversarial settings, the method should also demonstrate dynamic adaptability—such as adjusting to expanded unlearning scopes—without requiring complete retraining or suffering catastrophic forgetting of previously retained knowledge. This property of scope controllability distinguishes genuine reasoning-based boundary learning from rigid template memorization. The inclusion of continual unlearning simulations, even preliminary ones, significantly strengthens the claim that the method learns mutable knowledge boundaries rather than fixed refusal patterns. A comprehensive robustness evaluation thus serves as the final arbiter of whether the unlearned model has internalized durable, scope-aware refusal behavior.

**Core Evaluation Criteria:**
- **Adversarial baseline comparisons:** Are jailbreak, cross-lingual, and relearning attacks evaluated with direct side-by-side comparisons to baselines under identical conditions?
- **Scope evolution adaptability:** Can the method adjust to expanded or modified unlearning targets (e.g., broader semantic categories) without full retraining, and does it avoid catastrophic interference with retained knowledge?
- **Durability of refusal behavior:** Does the model maintain coherent, reasoning-based refusals after attack attempts, rather than reverting to uncontrolled outputs or trivial template refusals?