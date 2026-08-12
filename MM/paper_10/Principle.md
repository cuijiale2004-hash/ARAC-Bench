**Principle 1: Causal Mechanistic Attribution of Teacher-Student Capability Gaps in Multimodal Knowledge Distillation**

**Definition:**  
In MLLM distillation, identifying a bottleneck through observational analysis (e.g., attention map visualization) is insufficient for high-quality research. Reviewers expect authors to establish a rigorous causal link between the proposed intermediate alignment mechanism and downstream task performance, particularly when claiming that a specific misalignment (e.g., visual attention divergence) is the root cause of failed capability transfer. This requires moving beyond correlation by designing counterfactual experiments—such as attention mixing, component isolation, or cross-model attention swapping—that directly manipulate the hypothesized variable while holding other training dynamics constant. The principle demands that the work rule out confounding explanations, such as implicit regularization effects, text-token attention spillover, or generic feature-space alignment, which might otherwise explain the observed performance shifts. Ultimately, causal attribution ensures that the proposed module addresses a genuine mechanistic limitation rather than a superficial training artifact.

**Core Evaluation Criteria:**
- **Rigor of Causal Intervention:** Does the work include counterfactual experiments (e.g., mixing teacher, random, or third-party attention maps into the student) that directly manipulate the proposed mechanism to verify its causal impact on performance?
- **Confounding Factor Control:** Are alternative explanations rigorously ruled out, such as demonstrating that gains do not stem from text-attention alignment, logit-level distillation, or other implicit regularization effects?
- **Complementary Mechanism Isolation:** Is the proposed attention alignment shown to provide complementary benefits beyond vision-language feature-space alignment, for instance through module ablations or staged insertion tests?
- **Theoretical Grounding:** Does the analysis provide a principled explanation for why the identified mechanism specifically obstructs the target capability (e.g., compositional reasoning) rather than generic performance, supported by both empirical evidence and conceptual argumentation?

---

**Principle 2: Cross-Architecture and Cross-Scale Generalization Validation for MLLM Distillation Frameworks**

**Definition:**  
Because MLLMs exhibit substantial architectural heterogeneity—ranging from LLaVA-style projection-based designs to native-resolution ViT and DeepStack configurations—claims about universal distillation bottlenecks must be validated across diverse model families and scales. A study that identifies a bottleneck (e.g., visual attention misalignment) solely within a single model lineage risks conflating a fundamental transfer limitation with an architecture-specific training pathology. High-quality research must therefore demonstrate that the phenomenon persists and the remedy remains effective when applied to distinct vision encoders, LLM backbones, and tokenizer vocabularies, as well as across varying teacher-student capacity gaps. This principle also requires honest acknowledgment of tokenizer-lock limitations in logit-based distillation while providing empirical evidence for cross-architecture viability. Such validation is essential to distinguish domain-general scientific contributions from narrow, engineering-only improvements.

**Core Evaluation Criteria:**
- **Architectural Diversity:** Are experiments conducted on multiple MLLM families (e.g., LLaVA-style, Qwen-VL, native-resolution ViT architectures) to verify that the bottleneck is not an artifact of a single design choice?
- **Scale Robustness:** Does the evaluation span varying parameter scales for both teacher and student (e.g., sub-2B to 8B+) to demonstrate consistent distillation efficacy across model sizes?
- **Cross-Lineage Feasibility:** Where applicable, is there evidence of effectiveness across different tokenizer vocabularies or model lineages, or is the limitation clearly articulated as a shared community constraint rather than a method-specific flaw?
- **Phenomenon Persistence:** Does the work demonstrate that the identified bottleneck (e.g., attention misalignment) and its resulting performance gap recur under fundamentally different architectural conditions?

---

**Principle 3: Disentangled Evaluation of Visual Recognition, Perception, and Grounding Competencies**

**Definition:**  
MLLM distillation research must explicitly treat visual recognition, compositional perception, and spatial grounding as distinct cognitive competencies that are differentially affected by alignment strategies. Existing KD methods often optimize for visual recognition benchmarks (e.g., VQA), which primarily test object identification, while overlooking compositional reasoning (e.g., attribute binding, relational discrimination) that requires fine-grained perceptual integration. A rigorous evaluation framework therefore necessitates targeted benchmarks for each competency, alongside an explicit analysis of trade-offs: for instance, attention alignment may boost compositional reasoning while yielding only marginal gains on OCR-heavy VQA tasks due to differing underlying mechanisms. The work must explain why specific alignment objectives (feature-level vs. attention-level) selectively impact distinct competencies and must verify that improvements reflect genuine perceptual gains rather than dataset-specific statistical biases. Without this disentanglement, reviewers cannot assess whether the method advances holistic visual understanding or merely redistributes performance across task types.

**Core Evaluation Criteria:**
- **Competency-Specific Benchmarking:** Does the work evaluate on clearly distinct benchmarks for recognition (standard VQA), compositional reasoning (e.g., SugarCrepe, Winoground), and grounding (e.g., RefCOCO), rather than conflating them under a single "vision" metric?
- **Trade-off Transparency:** Is there explicit analysis of whether improving one competency (e.g., CR via attention alignment) degrades or stagnates another (e.g., TextVQA or OCR tasks), with null or negative results explained rather than obscured?
- **Mechanistic Explanation for Selective Gains:** Does the work provide a principled explanation for why the proposed method affects different competencies differently, such as distinguishing feature alignment (critical for recognition) from attention alignment (critical for perception)?
- **Diagnostic Validation:** Are fine-grained diagnostic tools (e.g., attention similarity stratification, hallucination benchmarks) used to verify that improvements reflect genuine perceptual gains rather than superficial statistical biases or language priors?

---

**Principle 4: Training Efficiency, Computational Cost, and Practical Deployability of Multi-Stage Distillation**

**Definition:**  
The introduction of multi-stage distillation pipelines, auxiliary loss terms, and intermediate alignment modules carries significant practical implications for training time, memory overhead, and reproducibility. In the context of MLLM KD, where baseline methods already span hundreds of GPU hours, high-quality research must quantitatively justify its training recipe by reporting wall-clock time, algorithmic complexity, and resource consumption relative to both standard supervised fine-tuning and comparable KD frameworks. This principle demands not only a complexity analysis of novel components (e.g., attention distillation losses versus correlation-matrix calculations) but also ablation studies validating that each stage is necessary and ordered correctly. Furthermore, the work should address deployability under resource-constrained settings—for example, by analyzing offline teacher pre-computation or gradient-checkpointing strategies—to ensure the method is accessible to researchers without industrial-scale compute. A method that achieves modest gains at disproportionate training costs faces scrutiny regardless of its theoretical elegance.

**Core Evaluation Criteria:**
- **Quantitative Cost Reporting:** Does the work report total training hours, GPU requirements, and memory overhead, comparing against both standard SFT and comparable KD baselines under identical hardware conditions?
- **Algorithmic Complexity Analysis:** Is the computational complexity of the novel components (e.g., attention loss calculation) formally analyzed and contrasted with the complexity of prior baseline methods?
- **Stage Necessity and Ordering:** Are ablation studies provided that validate the necessity of each training stage (e.g., permuting or removing stages) and confirm that the multi-stage design is not redundant?
- **Resource-Constrained Adaptability:** Does the work discuss applicability under limited compute (e.g., offline teacher pre-computation, gradient checkpointing) or provide evidence that the method remains practical outside of large-cluster environments?

---

**Principle 5: Intrinsic Student Capability Acquisition versus Teacher Over-Mimicry in Attention Transfer**

**Definition:**  
Distilling intermediate attention representations from a teacher to a student MLLM carries an inherent risk of over-mimicry, where the student memorizes teacher-specific focus patterns rather than developing robust, generalizable visual reasoning. High-quality research must demonstrate that the student acquires intrinsic perceptual capabilities—evidenced by strong performance on samples where its attention distribution diverges from the teacher’s—rather than collapsing into brittle, point-to-point imitation. This requires careful analysis of the alignment loss design: for example, cosine-similarity objectives that preserve vector directionality may foster relative importance learning, whereas MSE or KL-divergence losses may enforce rigid magnitude matching that harms generalization. The principle further requires empirical tests of student autonomy, such as evaluating performance across stratified attention-similarity intervals, testing frozen-attention ablations, or assessing robustness under adversarial visual perturbations. Without such evidence, a distillation method may be perceived as creating a frail copy of the teacher rather than cultivating an independently capable efficient model.

**Core Evaluation Criteria:**
- **Autonomy Under Divergence:** Does the work analyze performance on samples where student-teacher attention similarity is low, verifying that the student reasons correctly even when not closely mimicking the teacher’s focus?
- **Loss Function Design Rationale:** Is the choice of alignment loss (e.g., cosine similarity versus MSE or KL divergence) justified in terms of preserving student autonomy and fostering directional alignment rather than rigid point-to-point mimicry?
- **Robustness to Over-Mimicry:** Are experiments conducted to test whether strict attention alignment harms robustness (e.g., freezing attention blocks, adversarial perturbations), and do ablations show that adaptive student updates outperform frozen mimicry?
- **Attention Distribution Analysis:** Is the distribution of teacher-student attention similarities analyzed across the dataset to demonstrate that the student does not collapse to identical attention patterns but maintains diverse, task-appropriate focus capabilities?