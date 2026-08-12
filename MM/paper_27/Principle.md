**Principle 1: Diagnostic Attribution and Controlled Validation of Internal Generation-Understanding Capability Gaps in Unified MLLMs**

**Definition:**  
In unified MLLMs, claims regarding internal capability imbalances must be supported by metrics that quantify internal consistency directly, rather than relying solely on external estimators that may introduce systematic bias. The evaluation must rigorously control for confounding variables, particularly task difficulty, which can artificially suppress or inflate apparent non-unification scores. Researchers must further disentangle the root cause of observed misalignments—specifically determining whether the gap arises from weak generation or from misunderstanding by the ostensibly stronger branch—through cross-verification with human annotators and multiple independent strong judges. This diagnostic rigor ensures that the identified phenomenon is both real and correctly attributed before it is leveraged as a foundation for downstream methodological contributions. Without such controlled validation, self-improvement frameworks risk optimizing for spurious or misdiagnosed discrepancies.

**Core Evaluation Criteria:**
- **Phenomenon Isolation**: Does the work control for task difficulty and other confounders when measuring the internal gap, and does it validate the metric across diverse models and tasks?
- **Root Cause Diagnosis**: Is there evidence (e.g., weak generation scores, human verification) attributing the gap primarily to one branch rather than assuming it?
- **Metric Independence**: Are the proposed internal metrics designed to minimize bias from external evaluators, and are they validated against human judgments?

---

**Principle 2: Multi-Judge Reliability Auditing and Generalization Verification for Self-Referential Evaluation Signals**

**Definition:**  
When a model employs its own internal signals or a single external surrogate as both judge and training signal, the reliability and impartiality of that signal become central to scientific validity. Reviewers must scrutinize whether the competence of the evaluator is validated across task complexities, ideally through cross-checking with multiple state-of-the-art multimodal judges and human-in-the-loop verification on challenging subsets. Furthermore, improvements demonstrated on bespoke, internally defined metrics must be accompanied by corresponding gains on diverse public benchmarks to rule out overfitting to the idiosyncratic preferences or failure modes of a specific judge. The work should explicitly disambiguate whether metric sensitivity derives from genuine capability enhancement or from statistical artifacts such as tie-breaking on small denominators. This principle safeguards against circular reasoning in self-referential improvement loops.

**Core Evaluation Criteria:**
- **Judge Robustness**: Are multiple strong external judges and/or human annotators used to verify the reliability of the evaluation signal across task difficulties?
- **Benchmark Generalization**: Do the claimed improvements on custom internal metrics translate into statistically meaningful gains on diverse, standard understanding and generation benchmarks?
- **Bias Control**: Does the work explicitly rule out overfitting to the judge's distribution or exploit tie-breaking mechanics that inflate metric sensitivity without real capability gains?

---

**Principle 3: Mechanistic Empirical Grounding of Shared-Parameter Learning Dynamics in Multimodal Co-Improvement**

**Definition:**  
Theoretical explanations for co-improvement phenomena in unified models—such as aligned learning dynamics driven by shared empirical Neural Tangent Kernels—must be substantiated by empirical evidence that directly connects abstract formalism to concrete observables. It is insufficient to observe that shared parameters yield correlated improvements; the work must provide measurable proxies, such as post-training data similarity analyses, eNTK norm comparisons, or fine-grained error mode tracking, that validate the hypothesized mechanistic pathway. The analysis should additionally interrogate how architectural coupling strength modulates the effect, comparing tightly shared backbones against loosely coupled or decoupled designs to establish whether the mechanism is universal or architecture-dependent. Reviewers should demand that theoretical propositions yield non-obvious, falsifiable predictions rather than merely formalizing intuitively expected outcomes.

**Core Evaluation Criteria:**
- **Theory-Empirics Alignment**: Does the work provide empirical measurements (e.g., data similarity, eNTK proxies) that corroborate the proposed theoretical mechanism for co-improvement?
- **Architectural Scope**: Are the theoretical conclusions tested across different architectural paradigms (shared backbone, decoupled components, mixture-of-experts) to establish generality?
- **Distinguishing Novelty**: Does the theory yield non-obvious, testable predictions (e.g., false-positive correction dominance) rather than formalizing already-expected correlations?

---

**Principle 4: Cross-Architectural and Cross-Scale Robustness of Closed-Loop Self-Improvement Phenomena**

**Definition:**  
Self-improvement frameworks operating without external reward signals must demonstrate that their core phenomena are robust across model scales, training algorithms, and architectural families. Reviewers should expect replication of both generation gains and understanding co-improvement across distinct configurations, coupled with ablation studies on sensitive design choices such as candidate generation counts and selection thresholds. The method's practical significance must be assessed not only against weaker baselines but also against external-reward methods, clarifying whether internal-signal approaches provide a complementary free bonus or suffer from inherent performance ceilings. Additionally, extensions such as curriculum learning must be justified as non-obvious in the coupled unified-model setting, with empirical evidence that the prerequisite co-improvement effect prevents degradation from misjudged hard samples.

**Core Evaluation Criteria:**
- **Scale and Architecture Diversity**: Are experiments conducted across multiple scales and at least two distinct architectural families?
- **Hyperparameter Stability**: Are key hyperparameters (e.g., candidate count, confidence thresholds) ablated to show stability of the self-improvement signal?
- **Phenomenon Consistency**: Is the co-improvement effect demonstrated consistently on both custom metrics and standard benchmarks, or is it isolated to narrow task-model combinations?

---

**Principle 5: Data Integrity Assurance and Explicit Contamination Control in Post-Training Benchmark Evaluation**

**Definition:**  
For studies that construct post-training datasets from benchmark prompt pools, absolute separation between training and evaluation data is a foundational requirement for valid performance claims. Reviewers must verify that authors explicitly document official data splits, report quantitative overlap checks such as string-level matching between post-training candidates and evaluation prompts, and clarify any reuse of benchmark-derived materials. Even inadvertent contamination can invalidate reported gains, making transparency in data construction protocols essential for reproducibility. The absence of such verification procedures casts doubt on key experimental results regardless of the novelty of the proposed algorithm. Independent auditors should be able to reconstruct the data pipeline and confirm its integrity from the manuscript alone.

**Core Evaluation Criteria:**
- **Split Transparency**: Does the paper explicitly state which benchmark splits were used for post-training data construction versus evaluation?
- **Contamination Verification**: Are quantitative checks (e.g., string-level matching) reported to confirm zero overlap between post-training prompts and evaluation prompts?
- **Protocol Documentation**: Is the data construction pipeline sufficiently detailed to allow independent auditors to verify integrity?