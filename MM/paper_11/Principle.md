Principle 1: Theoretical Grounding and Mechanistic Justification of Explainability-Derived Token Relevance for Compression

**Definition:**  
When explainability methods are used to derive token importance scores for compression, reviewers must assess whether the mathematical formulation is rigorously justified and whether the relevance scores faithfully reflect causal contributions of visual tokens to instruction following. This includes analyzing gradient behavior, attribution faithfulness under ambiguous instructions, and the validity of assumptions that attention maps correlate with task relevance. Without such grounding, compression may discard critical tokens based on spurious correlations, undermining both model performance and trustworthiness. The principle further demands that authors distinguish between post-hoc explanatory utility and actionable compression signals, ensuring that explainability is not merely invoked as a heuristic but as a theoretically defensible criterion for token selection. In the MLLM domain, where visual tokens are high-dimensional and tasks are instruction-driven, mechanistic justification prevents the method from collapsing under distribution shift or adversarial prompting.

**Core Evaluation Criteria:**
- **Mathematical Rigor:** Is the relevance propagation equation derived or interpreted with clear analysis of properties such as gradient behavior, non-linearity effects, and faithfulness under multi-factor instructions?
- **Causal Validity:** Does the work establish that removing low-relevance tokens causes minimal performance degradation, versus merely correlating relevance scores with model outputs?
- **Sensitivity Analysis:** Are the explainability assumptions validated under diverse instruction types (ambiguous, compositional, fine-grained) rather than only coarse-grained recognition tasks?
- **Distinction from Black-Box Proxy:** Does the work avoid treating explainability scores as oracle labels without interrogating their limitations or failure modes?

---

Principle 2: Generalization Robustness and Data Efficiency of Lightweight Surrogate Models Predicting Token Importance

**Definition:**  
For methods that train lightweight networks to approximate expensive explainability signals, reviewers must evaluate whether the surrogate model generalizes beyond the limited training distribution to open-world multimodal inputs. This principle is critical because the utility of input-stage compression hinges on the surrogate's ability to robustly predict task-specific token relevance across varying image resolutions, video lengths, and semantic categories without requiring scale-matched training data. The review should scrutinize the inductive biases, structural motivations for the input-output mapping, and empirical evidence of out-of-distribution generalization. A surrogate that overfits to narrow training statistics would negate the practical benefits of pre-LLM compression by introducing unpredictable failure modes during deployment. Consequently, strong work in this area must provide both theoretical arguments for learnability and exhaustive empirical validation on held-out domains, task types, and input scales.

**Core Evaluation Criteria:**
- **Training-Test Discrepancy:** Is the surrogate evaluated on benchmarks and data domains entirely disjoint from its training set?
- **Scale Generalization:** Does the model generalize to inference inputs with significantly more tokens (larger images, longer videos) than seen during training?
- **Theoretical Motivation:** Is there a principled justification for why the mapping from shallow activation signals (e.g., early-layer attention) to global relevance is learnable with limited capacity and data?
- **Failure Mode Analysis:** Are there analyses of when and why the lightweight model fails, particularly on challenging sub-tasks such as fine-grained OCR or compositional reasoning?

---

Principle 3: Feasibility Validation and Comparative Advantage of Input-Stage Versus Intermediate-Layer Compression Paradigms

**Definition:**  
Research proposing token compression prior to the LLM must rigorously validate that this paradigm does not inherently sacrifice information essential for vision-language alignment, given prior consensus that shallow-layer tokens are indispensable. Reviewers should assess whether the work establishes a clear comparative advantage over intermediate-layer compression methods in terms of architectural flexibility, inference efficiency, and preservation of task-critical visual signals. This principle ensures that paradigm-shifting claims are backed by dispositive empirical and theoretical evidence rather than isolated observations on a narrow benchmark set. Specifically, the evaluation must demonstrate that input-stage compression yields net reductions in prefill latency and memory footprint while maintaining performance parity with deeper-layer alternatives. Without such cross-paradigm scrutiny, purported innovations risk being artifacts of incompatible evaluation protocols or undisclosed architectural dependencies. Ultimately, the burden of proof lies in showing that early pruning preserves the very semantic signals that intermediate-layer methods are designed to retain.

**Core Evaluation Criteria:**
- **Architectural Independence:** Does the method enable deployment without modifying the target MLLM's architecture or inference pipeline?
- **Efficiency Decomposition:** Are efficiency gains (FLOPs, prefill time, KV-cache memory) disaggregated and shown to stem from the input-stage design rather than confounding factors?
- **Performance Parity Under Compression:** Does the input-stage approach match or exceed intermediate-layer baselines at comparable compression ratios across diverse retention rates?
- **Justification for Paradigm Shift:** Does the work provide evidence challenging the assumption that shallow-layer tokens must be fully retained for lossless compression?

---

Principle 4: Comprehensive Benchmark Coverage and Robustness Under Aggressive Token Retention Regimes

**Definition:**  
For token compression research, experimental validation must span heterogeneous benchmarks encompassing coarse-grained recognition, fine-grained reasoning, OCR, chart understanding, document analysis, and video comprehension. Reviewers must evaluate whether the method maintains robustness under aggressive compression where task-irrelevant redundancy is scarce and model performance is most vulnerable. This principle prevents overestimation of effectiveness on low-redundancy tasks and ensures claims of negligible performance loss are genuinely stress-tested across retention regimes. Aggressive retention ratios expose whether the relevance signal can distinguish truly essential tokens from marginally informative ones, a capability that mild compression regimes often mask. Comprehensive coverage also guards against benchmark-specific overfitting, particularly in multimodal settings where visual and linguistic demands vary drastically across domains. High-quality research should therefore present degradation curves, cross-task variance analyses, and explicit disclosure of high-sensitivity task categories.

**Core Evaluation Criteria:**
- **Task Diversity:** Are evaluations conducted on benchmarks requiring fine-grained visual reasoning, text-rich OCR, compositional QA, and temporal video understanding?
- **Compression Granularity:** Are results reported across a spectrum of retention ratios to assess performance degradation curves under increasingly aggressive pruning?
- **Cross-Model Consistency:** Is the method validated on multiple MLLM architectures with divergent visual encoding strategies?
- **Failure Case Disclosure:** Does the work transparently report task categories or benchmarks where compression causes significant degradation?

---

Principle 5: Architectural Parsimony and Cost-Benefit Justification of Surrogate Model Overhead

**Definition:**  
When a compression method introduces an auxiliary predictor, reviewers must scrutinize whether the surrogate's architectural choices are empirically justified against the efficiency budget. The principle demands that the overhead of the surrogate does not erode the net gains from token pruning, and that design choices are validated through controlled ablations balancing accuracy against computational cost. Inefficient or over-parameterized auxiliary networks can shift the bottleneck from LLM attention to compressor inference, defeating the purpose of acceleration. Therefore, authors must transparently report end-to-end latency, FLOPs, and memory accounting for the auxiliary module alongside the main model. Furthermore, the selection of input representations for the surrogate should be grounded in structural correspondence to the target relevance distribution rather than convenience. Only through rigorous cost-benefit analysis can the community judge whether the surrogate is a practical deployment component or an academic artifact.

**Core Evaluation Criteria:**
- **Ablation Completeness:** Are key architectural hyperparameters (network depth, input layer selection, channel dimensions) systematically ablated with performance-efficiency trade-offs reported?
- **Net Efficiency Accounting:** Does the work report end-to-end inference costs including the surrogate's forward pass, demonstrating net savings versus baselines?
- **Input Representation Justification:** Is the choice of input signal (e.g., shallow attention maps versus raw visual embeddings versus deep-layer features) justified by both empirical performance and theoretical alignment with the target output?
- **Overhead Proportionality:** Is the surrogate's parameter count and computational footprint orders of magnitude smaller than the LLM's attention and FFN costs, ensuring practical deployability?