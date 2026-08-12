**Principle 1: Disentanglement of Framework Contributions from Modality-Specific Generator Priors and Extraneous Compute in Multimodal Prompt Optimization**

**Definition:**  
This principle requires authors to rigorously isolate the performance gains attributable to the proposed optimization methodology from confounding factors such as the inherent capabilities of powerful multimodal generators (e.g., proprietary image or video synthesis models) and the additional computational budget allocated to multimodal candidate generation. Reviewers must assess whether the framework remains effective when instantiated with lightweight, open-source generators rather than state-of-the-art proprietary ones, ensuring that improvements are not merely artifacts of stronger generator priors. Furthermore, controlled ablations must separate the contribution of simply adding a non-textual modality from the contribution of the joint optimization procedure itself, for instance by comparing full joint optimization against fixed-image-plus-optimized-text and text-only-optimized variants. The work should also demonstrate that sequential or independent optimization of modalities performs inferiorly to the proposed joint strategy, confirming that the gains arise from cross-modal co-optimization rather than from modular stacking. Additionally, the evaluation must account for whether performance improvements scale proportionally with increased computational investment, or whether the method delivers superior returns on investment compared to simpler baselines given the same resource envelope. Without such disentanglement, claims of methodological novelty are undermined because the observed gains may stem from tooling or resource scaling rather than algorithmic innovation.

**Core Evaluation Criteria:**
- **Generator-Agnostic Validation:** Are experiments conducted with diverse modality-specific generators, including small open-source models, showing consistent gains independent of generator strength?
- **Controlled Modality Ablation:** Does the work provide ablations isolating the optimization algorithm from the modality itself (e.g., fixed non-textual prompt + text optimization, text-only with same search procedure)?
- **Joint vs. Sequential Comparison:** Is there empirical evidence that joint optimization outperforms naive sequential or independent updates of textual and non-textual components?
- **Cost-Normalized Attribution:** Are performance gains analyzed under fixed real-world budgets to ensure that improvements are not simply purchased with additional compute or API calls?

---

**Principle 2: Robustness and Recoverability of Cross-Modal Semantic Alignment Under Joint Prompt Space Exploration**

**Definition:**  
Multimodal prompt optimization operates in a combinatorially expanded search space where textual and non-textual components must remain semantically consistent throughout iterative updates; without explicit safeguards, joint exploration risks catastrophic semantic drift, where one modality evolves to contradict the other. Because the search space grows combinatorially with modality addition, even minor misalignments between text and visual or molecular signals can compound over iterations, leading to degenerate prompt populations that appear promising under superficial metrics but fail to guide the MLLM reliably. This principle evaluates whether the proposed exploration mechanism actively preserves cross-modal alignment during optimization and whether it can recover when candidate prompts temporarily diverge or degrade. Reviewers should examine whether alignment is maintained through unified feedback signals, joint update rules, or explicit consistency constraints rather than through implicit assumptions. The evaluation must go beyond a single alignment metric (e.g., DSG) to include qualitative traces of optimization trajectories, failure-case analyses, and comparisons against independent update strategies that lack alignment-preserving mechanisms. Ultimately, the framework must demonstrate that its alignment strategy is not merely a post-hoc justification but a causal factor in performance improvement, verified through ablations where alignment mechanisms are removed or weakened.

**Core Evaluation Criteria:**
- **Mechanism Transparency:** Is the alignment-preserving mechanism explicitly defined, and does it use a unified feedback signal or joint update rule rather than decoupled modality-specific updates?
- **Empirical Alignment Tracking:** Are quantitative alignment scores (ideally multiple complementary metrics) reported across optimization iterations to detect drift?
- **Recovery Evidence:** Does the work provide empirical or theoretical evidence that the method recovers from semantic drift, such as trajectories showing temporary degradation followed by correction?
- **Ablative Necessity:** Are there ablation studies showing degraded performance when alignment-preserving components are removed or replaced with independent updates?

---

**Principle 3: Real-World Monetary and Computational Budget Accountability Including Generation Overheads in Multimodal Optimization Deployment**

**Definition:**  
Unlike text-only prompt optimization, multimodal optimization incurs substantial hidden costs from modality-specific generation, longer context windows for multimodal inputs, and increased GPU memory usage, which can easily offset savings from efficient candidate selection. This principle demands that efficiency claims be grounded in real-world deployment costs—monetized API expenses, wall-clock time, and memory footprints—rather than abstract evaluation counts or iteration numbers alone. Authors must present cost-constrained analyses where total budgets (including generation and evaluation) are fixed, demonstrating that the method outperforms text-only baselines not just in an unconstrained setting but when both operate under identical financial or computational caps. The analysis should explicitly account for the asymmetry between text generation and multimodal generation costs, and it should validate that the method remains practical when researchers lack access to high-end proprietary infrastructure. Without such accountability, efficiency claims risk being misleading or non-reproducible in resource-limited environments.

**Core Evaluation Criteria:**
- **Monetized Cost Comparison:** Does the paper report total monetary costs (e.g., API dollars) comparing multimodal and text-only methods under fixed budgets?
- **Generation Overhead Inclusion:** Are the costs of multimodal candidate generation (time, memory, API calls) explicitly included in efficiency analyses rather than omitted or treated as free?
- **Budget-Constrained Performance:** Are there experiments showing performance at low, medium, and high fixed budgets, proving superiority even when total spend is restricted?
- **Accessibility Validation:** Does the method demonstrate viability using low-cost, open-source components rather than relying exclusively on expensive proprietary generators?

---

**Principle 4: Coverage Breadth Across High-Cardinality Classification, Standard MLLM Benchmarks, and Complex Multimodal Reasoning Tasks**

**Definition:**  
A multimodal prompt optimization framework must demonstrate broad applicability beyond narrow, fine-grained classification tasks with artificially reduced label spaces, as such settings may overstate generalization by simplifying the discrimination problem. This principle requires evaluation on standard foundational multimodal benchmarks (e.g., MME, MMBench) that assess core perceptual capabilities, temporally complex video benchmarks that test dynamic alignment, and cross-modal reasoning benchmarks (e.g., MathVista, ScienceQA, Geometry3K) that require integrative logical inference. High-cardinality classification tasks with many categories and open-ended question-answering scenarios are essential to validate that optimized prompts capture generalizable, task-level semantics rather than instance-specific or class-specific shortcuts. Artificially constraining the label space risks allowing the optimizer to exploit dataset-specific shortcuts rather than learning robust, transferable multimodal representations that generalize to the full complexity of real-world tasks. The benchmark suite should span diverse modalities beyond static images—such as video, molecules, and audio—to confirm modality-agnostic optimization. Narrow task-specific datasets are acceptable only as part of a broader portfolio, not as the sole evidentiary basis for universal adaptability claims.

**Core Evaluation Criteria:**
- **Foundational Benchmark Coverage:** Are widely recognized static multimodal benchmarks (e.g., MME, MMBench) included to validate core perceptual improvements?
- **Complex Reasoning Tasks:** Does the evaluation encompass mathematical, scientific, or logical reasoning benchmarks requiring multimodal integration (e.g., MathVista, Geometry3K, ScienceQA)?
- **High-Cardinality and Open-Ended Validation:** Are there tasks with large numbers of classes (e.g., >30) or open-ended QA that prevent memorization of small label sets?
- **Temporal/Dynamic Modality Testing:** For video or temporal modalities, are there evaluations beyond short clips, testing long-sequence or dynamic alignment capabilities?

---

**Principle 5: Conceptual Distinctness from Soft-Prompt Tuning and Mechanistic Interpretability of Modality Interaction Effects**

**Definition:**  
The work must rigorously differentiate multimodal prompt optimization—defined as the discovery of discrete, reusable text-and-non-text prompt pairs for task-level performance—from superficially related paradigms such as continuous soft-prompt tuning (e.g., MaPLe), instance-specific query-time augmentation (e.g., MM-CoT with bounding boxes), and standard multimodal conditioning where multimodal inputs vary per example. The transition from optimizing discrete text strings to optimizing discrete multimodal pairs introduces fundamentally different sparsity and alignment challenges that must be explicitly characterized rather than assumed away as a natural extension. Authors must provide formal definitions of key constructs (e.g., parent prompts, child prompts, exploration operators) to prevent conceptual vagueness, and they must explain why the shift from text-only to multimodal optimization introduces unique technical challenges beyond a trivial expansion of the search space. Equally important, the work should furnish mechanistic evidence—via hidden-state visualizations, failure-case resolution analyses, and qualitative comparisons—showing that multimodal prompts induce genuinely new representational behaviors in MLLMs rather than redundant textual paraphrases. Performance tables alone are insufficient; reviewers must see proof that non-textual components provide complementary, irreducible information that text cannot encode.

**Core Evaluation Criteria:**
- **Paradigm Differentiation:** Is there a precise definition distinguishing the work from soft-prompt learning, per-instance visual augmentation, and per-query multimodal conditioning?
- **Formalization Rigor:** Are core algorithmic concepts (e.g., parent prompt, operator definitions, prior inheritance rules) formally specified with mathematical or algorithmic clarity?
- **Mechanistic Evidence:** Does the paper provide internal representations (e.g., hidden-state PCA), failure-resolution rates, or qualitative traces demonstrating that multimodal prompts shift model reasoning in ways text-only prompts cannot?
- **Complementarity Proof:** Is there evidence that the non-textual modality contributes information orthogonal to text, rather than acting as a redundant or decorative signal?