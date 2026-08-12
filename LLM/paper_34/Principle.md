**Principle 1: Theoretical Grounding of Energy-Based Chain-Rule Violations as Hallucination Indicators in Autoregressive LLMs**

**Definition:**  
This principle evaluates whether the proposed energy-based reinterpretation of the LLM softmax layer and the derived “spilled energy” metric are anchored in rigorous probabilistic theory rather than empirical heuristics. The autoregressive factorization implies that certain energy quantities across consecutive timesteps should cancel; the work must demonstrate that deviations from this cancellation are mechanistically linked to model errors rather than merely correlated. The evaluation examines the clarity, correctness, and completeness of the EBM derivation, its invariance to implementation details such as temperature scaling, and whether the theoretical narrative convincingly explains why chain-rule violations manifest as hallucinations. Furthermore, the work should clarify whether spilled energy measures a fundamental inconsistency in the model's probability assignment or simply reparameterizes existing uncertainty signals. A strong submission must also address potential confounders, such as alignment-induced calibration shifts, and show that the theoretical claims hold across both pretrained and instruction-tuned model variants.

**Core Evaluation Criteria:**
- **Mathematical Rigor**: Is the derivation from softmax probabilities to energy terms and the cancellation property formally correct and free of notational ambiguity? Are partition function equivalences proven?
- **Causal Explanation vs. Correlation**: Does the paper provide a principled argument for why chain-rule violations cause or reliably indicate hallucinations, beyond showing histogram separation? Does it address cases where high spilled energy might not correspond to factual error (false positives)?
- **Sensitivity to Model Variants**: Is the theoretical claim robust across pretrained and instruction-tuned variants, or are modifications needed to account for alignment-induced calibration shifts?
- **Distinction from Baselines**: Is the method clearly differentiated from simpler logit-confidence heuristics via this theoretical framework?

---

**Principle 2: Reliability and Failure Analysis of Exact Answer Token Localization in Localized Hallucination Detection**

**Definition:**  
This principle assesses the robustness and practicality of the prerequisite step that localizes the exact answer span within generated text. Since the detection signal is concentrated in specific tokens, the validity of the entire detection pipeline hinges on accurate localization. Reviewers scrutinized whether this step is brittle, requires auxiliary models, introduces latency, or fails on malformed outputs. A rigorous evaluation must quantify extraction success rates, characterize failure modes such as hallucinated brief answers, and analyze the sensitivity of downstream detection to localization errors. The work should also justify the overhead of this step relative to the overall detection cost and discuss whether the localization dependency constrains deployment in real-world, open-ended generation scenarios. Ultimately, the method's utility depends on whether the localization step can be implemented reliably without undermining the training-free advantage of the detector.

**Core Evaluation Criteria:**
- **Localization Methodology Clarity**: Is the exact algorithm for identifying answer tokens fully specified, including prompts for auxiliary LLMs, heuristics for closed-set tasks, and verification steps?
- **Quantified Reliability**: Are success rates reported across all benchmarks and model families? Is there analysis of cases where extraction fails or returns incorrect spans?
- **Overhead and Latency**: Is the computational and latency cost of the extraction step explicitly measured and compared against the detection cost itself?
- **Error Propagation**: Does the paper analyze how localization failures (e.g., wrong span, missing answer) affect hallucination detection metrics? Is the detection pipeline robust to such errors?

---

**Principle 3: Cross-Dataset Generalization and Robustness of Training-Free Detectors Against Trained Probe Classifiers**

**Definition:**  
This principle focuses on evaluating whether a training-free detector truly generalizes across diverse tasks and datasets, which is its primary advantage over dataset-specific trained probes. The evaluation must move beyond in-distribution validation and rigorously test transfer performance. Key is the experimental design that contrasts the detector's zero-shot cross-dataset behavior against probes trained on one dataset and tested on another, using confusion-matrix or similar analyses to reveal where and why generalization succeeds or fails. The work should demonstrate that its signal is universal across heterogeneous task distributions rather than exploiting spurious dataset-specific correlations. Additionally, claims regarding cross-model generalization must be supported by experiments across multiple families, sizes, and training regimes to validate broad deployability.

**Core Evaluation Criteria:**
- **Cross-Dataset Experimental Design**: Is performance reported under strict cross-dataset transfer, where training-free methods are tested on held-out datasets and probe baselines are explicitly trained on different datasets?
- **Generalization Metrics**: Are comprehensive transfer matrices (e.g., train-on-X-test-on-Y heatmaps) provided to visualize off-diagonal performance?
- **Multi-Model Validation**: Are claims about cross-LLM generalization supported by experiments across multiple model families (e.g., LLaMA, Mistral, Gemma), sizes, and training regimes (base vs. instruct)?
- **Comparative Advantage Justification**: Does the work quantify the performance gap between training-free detectors and OOD probe classifiers to justify the practical value of avoiding training?

---

**Principle 4: Scope of Applicability Beyond Short-Answer Factual Errors to Long-Form and Distributed Hallucinations**

**Definition:**  
This principle examines whether the detection method is evaluated beyond localized, short-answer factual mistakes to encompass complex, distributed, or subtle hallucinations typical of long-form generation. Many internal-consistency signals are noisy on non-semantic tokens such as punctuation or struggle with reasoning traces where errors are not confined to a single noun phrase. The evaluation should assess performance on diverse error types, characterize false positives on high-entropy functional tokens, and honestly report limitations regarding long-form outputs. A comprehensive study must ablate the necessity of exact-answer pooling versus full-sequence measurement to quantify how much detection relies on semantic localization. The work should also clarify whether the energy signal can detect distributed factual errors, self-contradictions, or reasoning breakdowns that do not map to a discrete answer window. Without such analysis, claims of broad hallucination detection risk being overstated and limited to narrow, extractive question-answering settings.

**Core Evaluation Criteria:**
- **Task Diversity**: Does the evaluation include long-form reasoning, multi-hop QA, and generations where errors are distributed across phrases rather than localized?
- **False Positive Characterization**: Are false positive rates analyzed for non-informative tokens (punctuation, stop words) and semantically plausible but factually incorrect outputs?
- **Signal Localization Analysis**: Does the paper ablate whether detection relies on exact-answer pooling vs. full-sequence measurement, quantifying the performance drop when localization is removed?
- **Limitation Disclosure**: Are the method's boundaries explicitly stated (e.g., inability to detect self-contradictions, reliance on discrete answer windows) without overstating universal applicability?

---

**Principle 5: Computational Overhead Characterization and Fair Baseline Inclusion for Real-Time Detection Feasibility**

**Definition:**  
This principle evaluates the practical feasibility of deploying the detector by rigorously characterizing its computational overhead and comparing it against relevant baselines under fair conditions. Hallucination detection methods occupy a spectrum of cost: logit confidence is essentially free, trained probes require upfront data collection, and semantic entropy requires multiple sampled generations. The work must clearly report latency, number of forward passes, and auxiliary model dependencies, and include baselines such as P(True) and, where feasible, Semantic Entropy, while justifying any exclusions. Furthermore, the evaluation should analyze the efficiency-performance trade-off, demonstrating that accuracy gains justify any additional computation or extraction steps. Scalability must also be addressed, including discussion of distillation or optimization strategies to reduce overhead for real-time or resource-constrained applications.

**Core Evaluation Criteria:**
- **Cost Transparency**: Is the inference cost explicitly broken down (e.g., per-query forward passes, number of energy computations, auxiliary LLM calls for extraction)?
- **Baseline Comprehensiveness**: Does the evaluation include logit confidence, P(True), trained probes under matched conditions, and discuss Semantic Entropy (even if excluded for cost, with justification)?
- **Efficiency-Performance Trade-off**: Is there a clear analysis of the trade-off between detection accuracy (AuROC) and computational budget, demonstrating that gains justify overhead?
- **Scalability Analysis**: Are results reported across model scales and is there discussion of distillation or optimization strategies to reduce overhead for real-time applications?