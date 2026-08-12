**Principle 1: Causal Validation Through Controlled Intervention of Emergent Reasoning Behaviors**

**Definition:**  
When a paper claims that an emergent, spontaneous behavior in large reasoning models plays a functional cognitive role, reviewers must demand evidence that goes beyond observational correlation. The work should include explicit interventional experiments—such as controlled insertion, suppression, or swapping of the target behavior while holding problem difficulty, prefix context, and decoding parameters constant—to establish a cause-and-effect relationship. Furthermore, the study should predict and verify boundary conditions, such as null effects in models lacking requisite reasoning priors, to rule out confounding by general model capability. Simply showing that a behavior co-occurs with correct answers is insufficient; the behavior must be shown to actively shape downstream reasoning. The causal narrative should also address reverse-causation alternatives, ensuring that correct reasoning does not merely produce the behavior as a byproduct.

**Core Evaluation Criteria:**
- **Interventional Design**: Does the work include direct manipulation (e.g., echo insertion into failing traces, echo removal from correct traces) with matched controls?
- **Confound Isolation**: Are problem statements, context length, and decoding budgets strictly equated across conditions?
- **Boundary Condition Testing**: Does the study verify predicted null or diminished effects in base models or other control groups where the mechanism should not apply?
- **Alternative Explanation Rejection**: Does it explicitly rule out competing hypotheses such as confirmation bias, length artifacts, or generic deliberation effects?

---

**Principle 2: Token-Granular Mechanistic Attribution and Cross-Layer Information Flow Tracing**

**Definition:**  
Mechanistic explanations of reasoning phenomena must transcend aggregate layer-wise or sequence-level statistics to be persuasive. Reviewers should expect fine-grained evidence that identifies which specific tokens, attention heads, or hidden states mediate the hypothesized effect, and how information propagates across transformer layers. This includes moving from scalar attention scores to token-level significance testing, word-level attribution heatmaps, and directed information-flow routes that trace backward from answer tokens to source tokens through the network. Such granularity is essential to distinguish genuine functional anchoring from superficial positional bias or diffuse attention increases. Without localized evidence, claims of "mechanistic explanation" risk remaining descriptive correlations that could arise from multiple confounding architectural factors.

**Core Evaluation Criteria:**
- **Sub-token/Token-level Resolution**: Are analyses conducted at the token or word level rather than solely via pooled metrics?
- **Information Flow Routing**: Does the work trace cross-layer propagation paths (e.g., using attention-edge backtracking) to demonstrate how the behavior acts as an information hub?
- **Specificity of Attention Targets**: Is there evidence that attention concentrates on semantically critical tokens (e.g., numerical entities) rather than uniform or template-driven boosting?
- **Positional Bias Control**: Does the work include controls (e.g., fixed-length ablations, distance-stratified tests) to ensure the effect is not a byproduct of token proximity?

---

**Principle 3: Strict Paired-Baseline Isolation of Data-Centric Variables in Supervised Fine-Tuning**

**Definition:**  
When evaluating training-based interventions that instill or suppress a specific generation pattern, the baseline comparison must be constructed from strictly paired data originating from the same teacher model and prompt distribution, differing only in the presence or absence of the target variable. This principle guards against conflating the target manipulation with differences in overall chain-of-thought quality, teacher capability, or prompt style that could independently drive performance changes. All training hyperparameters—such as optimizer, learning rate, batch size, and sequence length—must be identical across experimental and baseline runs. Any length differences introduced by the manipulation must be explicitly measured and, where possible, controlled for in the analysis. Without such rigorous isolation, observed gains may reflect data quality rather than the intended mechanistic intervention.

**Core Evaluation Criteria:**
- **Shared Teacher Corpus**: Do the experimental and baseline training sets derive from the same verified pool of teacher traces?
- **Minimal Editing Protocol**: Is the target feature (e.g., echo prefix) inserted or removed via controlled teacher editing with instructions to preserve all other reasoning steps and final answers?
- **Hyperparameter Lock**: Are fine-tuning configurations explicitly matched, ensuring any performance gap is attributable to the data difference?
- **Length and Format Parity**: Does the study report and control for sequence length differences that arise from the manipulation?

---

**Principle 4: Statistical Significance Testing and Effect Size Quantification for Attention-Based Claims**

**Definition:**  
Quantitative claims regarding attention-based mechanisms require rigorous statistical validation beyond visual inspection or mean differences. Reviewers should expect formal hypothesis testing—such as Welch’s t-test or Kolmogorov-Smirnov tests—to assess whether observed attention or likelihood gaps are statistically significant. These tests must be complemented by effect-size metrics like Cohen’s d or AUC to gauge the practical magnitude and discriminative power of the proposed mechanism. This is particularly critical in interpretability research where raw attention weights can fluctuate across layers, making absolute differences potentially misleading without normalization. The study should also report robustness checks using standardized scores to rule out layer-specific scale artifacts or outlier-driven effects.

**Core Evaluation Criteria:**
- **Formal Hypothesis Testing**: Are p-values or other statistical tests reported for key attention or likelihood gaps?
- **Effect Size Reporting**: Are Cohen’s d, AUC, or similar metrics provided to quantify the discriminative power of the proposed mechanism?
- **Normalization and Robustness**: Does the work include normalized scores (e.g., Z-scored attention, standardized mean difference) to rule out layer-specific scale artifacts?
- **Multiple Comparison Awareness**: When testing many tokens or layers, is there acknowledgement or correction for multiple comparisons?

---

**Principle 5: Cross-Model Architectural Consistency and Distributional Generalization of Mechanisms**

**Definition:**  
A proposed cognitive mechanism in reasoning models should not be treated as established if it is demonstrated only within a single model family or benchmark. Reviewers must evaluate whether the phenomenon and its downstream benefits replicate across diverse architectures, model scales, and task distributions that vary in difficulty and format. Mechanistic signatures—such as characteristic mid-layer attention peaks or specific information-flow patterns—should be shown to generalize across systems to confirm they reflect fundamental reasoning strategies rather than dataset-specific spurious correlations. The evaluation should span multiple benchmarks within or beyond the training domain to assess distributional robustness. If a mechanism is claimed to be universal, its absence or reversal in certain architectures should be theoretically motivated and empirically checked.

**Core Evaluation Criteria:**
- **Multi-Architecture Validation**: Are experiments conducted across at least two distinct model families or training paradigms?
- **Distributional Robustness**: Does evaluation span multiple benchmarks that differ in topic, difficulty, or answer format beyond the training distribution?
- **Mechanistic Consistency**: Do attention or activation signatures (e.g., layer-wise peaks) replicate across models?
- **Avoidance of Single-Dataset Claims**: Is the core analysis (especially mechanistic probing) validated beyond a single benchmark (e.g., GSM8K)?