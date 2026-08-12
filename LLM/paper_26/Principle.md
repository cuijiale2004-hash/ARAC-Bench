**Principle 1: Functional Fidelity and Mechanistic Rigor of Bio-Inspired Associative Memory Circuits in Transformer-Based Continual Learning**

**Definition:**  
In the LLM domain, proposals claiming biological inspiration for memory or retrieval mechanisms must be evaluated on whether they offer genuine algorithmic novelty beyond interpretable analogy. A hippocampal metaphor, for instance, must translate into computationally distinct operations—such as iterative attractor dynamics or pattern completion in latent space—that cannot be reduced to standard key-value attention with residual updates. Reviewers should assess whether the biological mapping is functionally necessary (i.e., removing the bio-inspired structure eliminates a uniquely enabled capability) or merely a narrative overlay on conventional PEFT techniques. Furthermore, the work must avoid overclaiming biomimicry; given that neuroscience remains incomplete, functional abstraction is acceptable only when accompanied by clear, minimal computational correspondences to the cited biological process. Ultimately, this criterion distinguishes research that leverages biological insight to unlock new optimization or representational behaviors from work that repackages existing methods with neuroscientific branding.

**Core Evaluation Criteria:**
- **Necessity test:** Does the bio-inspired component enable a capability (e.g., iterative self-correction of queries, multi-step pattern completion) that is theoretically or empirically impossible with standard single-step retrieval or plain residual connections?
- **Mapping specificity:** Are biological structures (e.g., CA3 auto-association, DG pattern separation) mapped to explicit, differentiable computational steps rather than described only at a conceptual level?
- **Ablation disentanglement:** Do ablation studies isolate the bio-inspired mechanism’s contribution from confounding factors such as increased parameter count, additional layers, or generic regularization terms?

---

**Principle 2: Theoretical Grounding of Iterative Latent Retrieval as Implicit Higher-Order Optimization in PEFT Dynamics**

**Definition:**  
For methods employing iterative or recurrent latent retrieval in LLMs, theoretical analyses—such as Krylov-subspace expansions or implicit Hessian preconditioning—must do more than provide post-hoc mathematical legitimacy. They should generate actionable, falsifiable predictions about hyperparameter regimes (e.g., optimal iteration depth, temperature scaling), convergence behavior, and gradient stability that are empirically validated. The theory must specifically explain why recurrence improves continual learning performance, for instance by demonstrating curvature correction that aligns task gradients or by characterizing attractor convergence that stabilizes old-task memory reactivation. Reviewers must be wary of theories that are mathematically correct but optimization-generic, failing to distinguish the proposed method from any other iterative smoothing technique. This principle demands that theoretical claims serve as a guiding framework for design choices rather than decorative justification.

**Core Evaluation Criteria:**
- **Actionable predictions:** Does the theory yield concrete guidelines (e.g., iteration counts of 2–4, spectral normalization requirements) that are experimentally verified, or is it purely descriptive?
- **CL-specific relevance:** Does the preconditioning or convergence analysis address continual-learning pathologies (e.g., gradient interference across tasks, loss landscape distortion from sequential updates) rather than generic optimization acceleration?
- **Distinctiveness check:** Can the theoretical result be trivially applied to standard iterative methods (e.g., unrolled gradient descent, deep equilibrium models) without modification, or does it hinge on the specific retrieval structure proposed?

---

**Principle 3: Multi-Protocol and Multi-Metric Rigor in Quantifying Catastrophic Forgetting Beyond Aggregate Accuracy**

**Definition:**  
Rigorous evaluation of catastrophic forgetting in LLM continual learning requires going beyond final accuracy or average accuracy to report explicit forgetting metrics such as Forgetting Measure (FM), Backward Transfer (BWT), and per-task accuracy degradation curves. The field must demand evaluation across diverse protocols—class-incremental learning (task-agnostic), task-incremental learning (task-aware), and online or streaming scenarios—to ensure that performance gains are not artifacts of a single, favorable experimental setup. Furthermore, analysis should include qualitative or quantitative diagnostics of when and why forgetting occurs, such as confusion patterns across task boundaries or representation drift measurements. This principle ensures that claims of “mitigating forgetting” are backed by multidimensional evidence rather than inferred solely from aggregate accuracy improvements, which can obscure severe performance collapse on early tasks.

**Core Evaluation Criteria:**
- **Forgetting quantification:** Are FM, BWT, and/or per-task forgetting curves reported in addition to final accuracy and average accuracy?
- **Protocol diversity:** Does the evaluation span multiple continual learning settings (CIL, TIL, domain-incremental, online) to demonstrate robustness to task-boundary assumptions?
- **Diagnostic depth:** Does the paper analyze failure modes—such as which tasks are most forgotten, whether forgetting correlates with task similarity, or how representation geometry evolves—rather than only reporting summary statistics?

---

**Principle 4: Cross-Architectural Transferability and PEFT-Modality Generalization in Continual Learning Systems**

**Definition:**  
Parameter-efficient continual learning methods for LLMs must demonstrate generalization across multiple backbone architectures (e.g., supervised, self-supervised, or sharpness-aware pretraining paradigms) and PEFT modalities (prompt tuning, adapters, LoRA, prefix tuning) to rule out overfitting to a specific model family or parameterization. A method designed exclusively for soft prompt pools has limited applicability if it cannot be instantiated for adapter-based or low-rank continual learners. Reviewers should evaluate whether the proposed retrieval or memory mechanism relies on generic Transformer latent-space properties—such as layer-wise hidden states—rather than idiosyncratic features of a particular backbone or prompt representation. This principle ensures that contributions are structural and portable, advancing the broader PEFT-CL toolkit rather than producing narrow, benchmark-specific solutions.

**Core Evaluation Criteria:**
- **Backbone diversity:** Are experiments conducted on multiple pretrained models with distinct training paradigms to verify feature-space agnosticism?
- **PEFT modality coverage:** Is the framework instantiated across different efficient fine-tuning families, or does the method explicitly acknowledge and justify its restriction to specific module types?
- **Structural vs. incidental gains:** Does the analysis demonstrate that performance stems from the proposed retrieval mechanism rather than from interactions with a specific backbone’s pretraining objective or a particular PEFT module’s capacity?

---

**Principle 5: End-to-End Resource Accountability in Memory-Augmented Efficient Fine-Tuning for Large Language Models**

**Definition:**  
Claims of computational efficiency in memory-augmented LLM fine-tuning—such as reduced FLOPs, halved training costs, or elimination of auxiliary forward passes—must be substantiated with holistic resource accounting that includes peak GPU memory usage, wall-clock time, and hidden overheads like sequence-length expansion from concatenated retrieval vectors or truncated backpropagation memory. Efficiency comparisons must control for baseline equivalence, ensuring that cost reductions are not achieved by comparing against unnecessarily expensive baselines (e.g., methods using double forward passes when a single-pass variant exists). Additionally, reviewers should assess whether efficiency gains are scalable: iterative latent mechanisms may be cheap at small scale but become memory-bound as sequence length or retrieval depth grows. This principle enforces that “efficiency” is treated as an empirically verified, multi-dimensional property rather than a marketing claim derived from a single metric.

**Core Evaluation Criteria:**
- **Holistic cost metrics:** Are training FLOPs, peak activation memory, and actual GPU memory footprint reported alongside accuracy? Are overheads from iterative loops (e.g., concatenated token sequences, key-value cache expansion) explicitly quantified?
- **Fair baseline comparison:** Is the efficiency gain calculated against comparably equipped baselines under identical hardware and batch-size constraints, with careful accounting for removed versus added operations?
- **Scalability analysis:** Do experiments or theoretical complexity analysis indicate how costs scale with retrieval depth, task sequence length, or model dimension, identifying potential bottlenecks before deployment?