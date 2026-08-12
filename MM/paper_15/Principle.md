**Principle 1: Circularity Avoidance and Independent Human-Annotation Validation in Bootstrapped Video Judge Training**

**Definition:**  
In bootstrapped training pipelines for multimodal judges, where synthetic training data is produced by an iterative generator-evaluator loop, there exists a fundamental risk of circular evaluation if meta-evaluation benchmarks are constructed using the same pipeline. This principle demands that any claim of superior evaluation capability be rigorously validated on fully independent, human-annotated benchmarks drawn from non-overlapping distributions. The research must demonstrate that the trained judge is not merely distilling the evaluator's preferences, but rather learns generalizable evaluation criteria. Empirically, the student model should consistently outperform its zero-shot teacher evaluator on human-labeled data, establishing that performance gains reflect genuine judgment capability rather than distribution fitting. Without such independent validation, claims of surpassing larger models remain methodologically ungrounded.

**Core Evaluation Criteria:**
- **Independent Benchmark Coverage**: Does the evaluation include substantial human-annotated benchmarks (e.g., VATEX-Eval, LongVideoBench, VideoAutoArena) that were created entirely separately from the training pipeline?
- **Student-Teacher Separation**: Does the fine-tuned judge demonstrably exceed the zero-shot evaluator model that supervised its training, particularly on held-out human judgments?
- **Non-Overlapping Distributions**: Is there explicit evidence that training and evaluation videos/instructions come from disjoint sources, preventing data leakage?

---

**Principle 2: Generalization Across Video Duration and Temporal Reasoning Complexity for Multimodal Judges**

**Definition:**  
Video understanding tasks span a vast range of temporal scales and reasoning demands, from brief clip captioning to long-form temporal causal reasoning over hundreds of seconds. A video judge must therefore exhibit robust generalization beyond the short-form, simple-QA distribution typical of many instruction-tuning datasets. Evaluating such a judge requires dedicated assessment on long-duration videos that demand cross-frame event tracking, sequential understanding, and temporal localization. Furthermore, the judge must prove it leverages visual-temporal signals rather than relying on textual shortcuts or language priors. Performance should be validated across diverse video domains, task formats, and temporal granularities to ensure the model's evaluation capability is tied to genuine video comprehension rather than superficial pattern matching.

**Core Evaluation Criteria:**
- **Long-Form Temporal Reasoning**: Are there explicit evaluations on long-duration video benchmarks requiring understanding of event sequences and temporal relationships?
- **Cross-Task and Cross-Domain Robustness**: Does performance hold across varied task types (QA, captioning, instruction following) and video characteristics (duration, genre, complexity)?
- **Modal-Signal Dependency**: Does the model degrade significantly when visual input is reduced or replaced, confirming reliance on video content rather than text-only heuristics?

---

**Principle 3: Bias Propagation Control and Supervisory Quality Assurance in Generator-Evaluator Feedback Loops**

**Definition:**  
Bootstrapped supervision pipelines inherently rely on imperfect generator and evaluator models that may encode idiosyncratic biases, inconsistent rating standards, or factual errors. Without explicit safeguards, these flaws can amplify across iterative refinement loops, causing the final judge to inherit and potentially magnify its teachers' weaknesses. Research in this domain must therefore demonstrate rigorous bias-mitigation mechanisms, such as filtering samples where evaluator ratings diverge from intended targets, aggregating seed data from heterogeneous sources to avoid stylistic overfitting, and enforcing quality thresholds during data synthesis. Critically, the work must provide empirical evidence that the final judge transcends the capability ceiling of its synthetic supervisors. This includes showing performance gains on human-annotated benchmarks where the original evaluator performed poorly, proving that the model extracts generalizable evaluation principles rather than replicating biased heuristics.

**Core Evaluation Criteria:**
- **Explicit Mitigation Mechanisms**: Are there documented steps to reject mismatched samples, diversify data sources, or correct for evaluator drift during bootstrapping?
- **Ceiling Breaking**: Does the final model surpass its own evaluator and generator on independent human benchmarks, indicating it learns beyond supervisory limitations?
- **Synthetic Label Validation**: Is the quality of bootstrapped labels verified through both automatic consistency checks and human verification (e.g., inter-annotator agreement on generated preference pairs)?

---

**Principle 4: Calibration Rigor and Granular Failure Mode Taxonomy in Human-Aligned Video Evaluation**

**Definition:**  
High correlation with human preferences is a necessary but insufficient condition for a trustworthy video judge; the model must also be well-calibrated across the entire rating spectrum and exhibit interpretable, fine-grained error patterns. Particular scrutiny should fall on mid-range quality discrimination—such as distinguishing between ratings of 3 and 4 or 4 and 5—where nuanced judgment is most critical and human disagreement is highest. A rigorous study must move beyond aggregate correlation metrics to provide a detailed taxonomy of failure modes, quantifying issues like systematic overestimation bias, tolerance of vague or hedging language, temporal reasoning failures, and missed subtle factual errors. This analysis should directly inform understanding of the model's limitations and point toward specific improvements in training data or architecture.

**Core Evaluation Criteria:**
- **Calibration Across Ratings**: Are calibration metrics such as Expected Calibration Error (ECE), RMSE, and exact-agreement rates reported and analyzed per rating level?
- **Granular Error Taxonomy**: Does the work provide a quantitative breakdown of failure types (e.g., vagueness, hallucination, temporal misalignment, positive bias) with supporting examples?
- **Mid-Range Discrimination**: Is there explicit analysis of performance on ambiguous, adjacent-quality pairs and middle-range ratings where fine-grained discernment is required?

---

**Principle 5: Distinctiveness from Prior Image/Generation Judges and Video-Specific Benchmarking Rigor**

**Definition:**  
Research proposing MLLM-as-a-Judge for video understanding must clearly demarcate its contributions from adjacent but distinct lines of work, including judges for image understanding (e.g., Prometheus-Vision) and evaluators for video generation quality (e.g., VideoScore). Distinctions should span the target domain (understanding textual responses conditioned on video versus assessing synthesized video fidelity), training methodology (iterative bootstrapping versus human annotation or direct distillation from proprietary models), and the provision of instance-specific, interpretable rubrics. Furthermore, the work must adhere to rigorous benchmarking standards by comparing against appropriate baselines at similar capability levels, transparently reporting why certain candidate models were excluded, and validating on video-specific meta-evaluation suites rather than repurposing image or text benchmarks. Failure to establish these boundaries undermines both the novelty claim and the practical utility of the proposed judge.

**Core Evaluation Criteria:**
- **Scope and Task Clarity**: Is the evaluation target (video understanding responses) clearly differentiated from video generation and image understanding evaluation?
- **Methodological Novelty**: Is the bootstrapping approach, rubric generation mechanism, or training paradigm sufficiently differentiated from prior distillation or annotation-based strategies?
- **Baseline Rigor and Transparency**: Are baselines fairly selected, comprehensively reported, and accompanied by transparent justification for any exclusions based on instruction-following failures?