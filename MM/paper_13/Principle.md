**Principle 1: Task-Conflict Quantification and Unified Representation Efficacy in Joint Multimodal Understanding-Generation Models**

**Definition:**  
This principle evaluates whether a proposed unified multimodal architecture genuinely mitigates the inherent tension between visual understanding and generation, or merely combines them superficially. The work must provide empirical evidence that joint training does not severely degrade either capability relative to single-task specialists. It should analyze whether continuous and discrete visual representations originate from a homogeneous semantic space and whether the LLM processes them without catastrophic interference. Crucially, the evaluation demands explicit comparison against both pure-discrete and dual-encoder baselines to isolate the benefits of hybrid unification. The principle also requires examination of whether task interference decreases as model scale increases, validating the architectural choice beyond small-scale prototypes.

**Core Evaluation Criteria:**
- **Single-Task Baseline Comparison**: Does the paper compare unified training against understanding-only and generation-only variants to quantify the degree of task conflict?
- **Homogeneous Semantic Space Verification**: Is there evidence that shared encoders reduce cross-modality friction compared to heterogeneous token streams from separate pathways?
- **Cross-Scale Stability**: Does the work demonstrate that task interference remains minimal when the model scales, or does the unified benefit vanish at larger capacities?
- **Isolation of Unified Effects**: Are performance gains clearly attributable to the unified representation design rather than independent improvements in data scale or training compute?

---

**Principle 2: Comprehensive Ablation and Scaling Validation for Hybrid Vision Tokenization Architectures**

**Definition:**  
This principle assesses the thoroughness of empirical validation for hybrid tokenization strategies that produce both continuous and discrete visual representations. Given the complexity of unifying understanding and generation, the research must systematically ablate tokenizer design choices, training recipes, and model scales. It should justify architectural decisions—such as using latent-space alignment versus pixel-level reconstruction for tokenizer pre-training—through controlled experiments. Scaling analyses must be sufficiently granular to reveal trends, or explicitly justify any gaps in model size ladders. Furthermore, the training data mixture and stage-wise curricula must be documented and justified, as these heavily influence the success of joint optimization.

**Core Evaluation Criteria:**
- **Tokenizer Ablation Completeness**: Are pure-discrete, dual-encoder, and hybrid strategies fairly compared under identical training and inference conditions?
- **Scaling Granularity and Justification**: Does the scaling study include intermediate model sizes to reveal smooth scaling laws, or does it provide a compelling rationale for non-continuous ladders?
- **Architectural Decision Rationale**: Are key design choices—such as trainable versus frozen vision backbones, or adapter configurations—validated through ablation rather than asserted?
- **Training Recipe Transparency**: Is the multi-stage data mixture (e.g., proportions of text, image-to-text, and text-to-image data) documented and its impact on dual-capability learning analyzed?

---

**Principle 3: Novelty Articulation and Technical Differentiation in Crowded Unified Multimodal Literature**

**Definition:**  
This principle examines whether the paper clearly distinguishes its contributions from closely related unified multimodal models with similar high-level motivations, such as combining continuous and discrete tokens. In a rapidly evolving field, simply integrating existing components is insufficient; the work must identify specific technical advances—such as training strategies, architectural simplifications, or scaling insights—that enable prior conceptual ideas to become competitive systems. Reviewers expect precise differentiation from contemporaneous methods that share conceptual overlap, such as those employing dual-tokenizer or mixture-of-transformer paradigms. The evaluation focuses on whether the paper's novelty claims are supported by ablations that isolate the proposed component's impact from confounding factors like increased compute or data. Without such differentiation, even strong empirical results may be attributed to engineering scale rather than genuine methodological innovation.

**Core Evaluation Criteria:**
- **Prior Work Positioning**: Does the paper explicitly contrast its method with architecturally similar approaches beyond superficial citation, clarifying technical distinctions?
- **Component Isolation**: Are ablations designed to show that the proposed hybrid design specifically causes the observed gains, separate from data quantity or model size effects?
- **Concept-to-System Advancement**: Does the work explain why previously explored high-level ideas failed to produce competitive results, and how the current approach overcomes those barriers?
- **Claim Proportionality**: Are novelty claims carefully scoped to actual technical contributions, avoiding overstatement of incremental architectural changes?

---

**Principle 4: Mechanistic Insight and Representational Analysis Beyond Pure Performance Metrics**

**Definition:**  
This principle demands that unified multimodal research move beyond reporting benchmark scores to explain why and how architectural unification affects learning. Reviewers increasingly expect discussions of learned representations, qualitative failure modes, and causal analyses of where unification benefits or hurts capabilities. The work should examine representation geometry, analyze cross-task interference mechanisms, or provide qualitative examples that reveal model behavior invisible in aggregate statistics. Specifically, it must avoid using improved final performance alone to infer that unification is successful, instead establishing clear links between architectural choices and representational properties. Deep interpretability is especially critical when claiming minimal task conflict, as quantitative parity with single-task models does not automatically explain mechanistic compatibility.

**Core Evaluation Criteria:**
- **Explainability Depth**: Does the paper analyze why hybrid tokens reduce task interference mechanistically, rather than offering only descriptive summaries of results?
- **Qualitative Evidence**: Are there illustrative examples—such as text-rich understanding precision, generation fidelity, or interleaved task behavior—that validate claims invisible in numeric scores?
- **Benefit-Hurt Analysis**: Does the work explicitly discuss scenarios where unified representation helps (e.g., fine-grained spatial reasoning) versus potential limitations or trade-offs?
- **Avoidance of Black-Box Inference**: Does the study refrain from concluding architectural superiority based solely on end-task accuracy, instead providing intermediate analyses (e.g., feature space alignment, gradient conflicts)?

---

**Principle 5: Fair Comparative Benchmarking Against Specialist and Unified Baselines Across Dual Capabilities**

**Definition:**  
This principle evaluates whether the paper conducts rigorous and fair comparisons against both specialist models—understanding-only or generation-only—and unified counterparts across the full spectrum of capabilities. Claims of state-of-the-art unified performance or competitiveness with specialists require matched evaluation protocols, comparable model scales, and diverse benchmarks, particularly text-rich understanding tasks that are sensitive to tokenization strategy. The comparison must avoid cherry-picking and should cover emergent capabilities like instruction following and world-knowledge grounding. Human evaluations of generation quality must follow transparent protocols with adequate sample sizes and rater blinding to support qualitative superiority claims. Ultimately, the work must demonstrate that its unified approach does not merely achieve acceptable trade-offs, but delivers genuine synergies or at least non-destructive integration across both modalities.

**Core Evaluation Criteria:**
- **Benchmark Breadth and Sensitivity**: Are both understanding (especially text-rich, document, and chart OCR tasks) and generation (automated and human evaluation) covered comprehensively?
- **Baseline Parity**: Are comparisons made against appropriately sized specialist and unified models with similar training data scope and architectural families?
- **Human Evaluation Rigor**: Is the assessment protocol for generation quality described with sufficient detail— including rater counts, blinding procedures, and scoring dimensions—to be reproducible?
- **Emergent Capability Coverage**: Does the evaluation extend beyond standard benchmarks to test unified emergent behaviors, such as multimodal instruction following, visual editing, or world-knowledge-aware generation?