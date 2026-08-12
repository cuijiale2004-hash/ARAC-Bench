**Principle 1: Disentanglement of Method Efficacy from Training Data Scale and Backbone Capability in Graph-Augmented Retrieval**

**Definition:**  
In graph-augmented retrieval-augmented generation systems, reported performance gains must be rigorously isolated from confounding variables such as larger training corpora, more powerful base LLMs, or divergent fine-tuning scales. Reviewers must evaluate whether authors conduct controlled experiments that match baselines on training set size, hold backbone architectures constant across comparisons, and explicitly demonstrate data-efficiency advantages. This principle ensures that improvements in reasoning accuracy are causally attributable to the proposed structural mechanisms—such as dynamic graph assembly, topology-aware planning, or denoising via triple extraction—rather than scale-driven memorization or base-model capability inflation. Without such disentanglement, the community risks conflating engineering resource advantages with genuine algorithmic innovation. The meta-review specifically flagged this concern, requiring authors to prove their gains persisted when RPG was trained on five times as much data.

**Core Evaluation Criteria:**
- **Controlled Data-Scale Comparisons**: Does the work compare against baselines trained on equivalent or larger data volumes, or demonstrate superior performance with significantly less training data?
- **Backbone Consistency**: Are comparisons performed across identical LLM backbones (e.g., LLaMA-2/3, Claude-3.5/4.5) to ensure fair evaluation?
- **Mechanism Isolation via Ablation**: Are there ablations that remove graph-specific components to quantify their isolated contribution beyond scale effects?
- **Efficiency-Performance Trade-off**: Does the work analyze whether performance gains justify any additional computational cost relative to simpler baselines?

---

**Principle 2: Architectural and Paradigmatic Distinction from Static Graph Indexing and Iterative Text Retrieval**

**Definition:**  
Research proposing dynamic knowledge graph construction must clearly demarcate its technical contributions from existing static corpus-level graph methods and iterative text-based retrieval systems. Reviewers evaluate whether the work establishes a unique position in the design space—such as online question-specific graph assembly versus offline global indexing, or trainable graph neural encoders versus serialized textual triplets. The motivation should articulate why flat text accumulation suffers from context rot or reasoning brittleness, and how structured topology explicitly remedies these failures. This principle prevents incremental prompting strategies from being repackaged as architectural contributions. The reviews highlighted that prior works like KnowTrace, SG-Prompt, and ERA-CoT operate in adjacent areas, so authors must prove their method introduces trainable architectures or active retrieval loops that meaningfully expand the capability frontier rather than repackaging existing ideas.

**Core Evaluation Criteria:**
- **Positioning in Design Space**: Does the paper clearly map its method against static vs. dynamic, global vs. query-specific, and trainable vs. prompting-based paradigms?
- **Technical Differentiation**: Are the core architectural choices fundamentally distinct from prior art, or are they minor syntactic variations of existing prompting or indexing schemes?
- **Comparative Baseline Coverage**: Does the evaluation include state-of-the-art representatives from both static graph-RAG and iterative text-RAG lineages?
- **Motivated Theoretical/Empirical Claim**: Does the work provide evidence or mechanistic intuition for why unstructured context fails and why graph topology succeeds?

---

**Principle 3: Causal Interpretability of Dynamic Knowledge Graph Properties on Multi-Step Reasoning**

**Definition:**  
Beyond aggregate accuracy metrics, high-quality research must investigate the causal relationship between the properties of constructed knowledge graphs and downstream reasoning performance across diverse task types. Reviewers expect analyses that correlate graph structural characteristics—such as connectivity, noise level, or coverage of reasoning chains—with task complexity, explicitly distinguishing which reasoning modalities benefit most from structuring. The work should avoid treating the graph as a black-box performance booster; instead, it must demonstrate how triple-based denoising improves compositional reasoning, or how topology-guided planning aids multi-hop synthesis. Qualitative case studies should expose correct reasoning traces and trace failures to identifiable graph construction defects. The meta-review noted the initial absence of in-depth graph quality analysis, underscoring that rigorous evaluation requires linking graph properties to reasoning outcomes rather than merely reporting end-to-end accuracy.

**Core Evaluation Criteria:**
- **Structure-Performance Correlation**: Does the work analyze how graph properties (density, completeness, triple quality) correlate with performance on tasks of varying reasoning complexity?
- **Reasoning-Modality Breakdown**: Is there explicit analysis of which reasoning types (multi-hop, long-form synthesis, closed-set classification) benefit most from graph structuring?
- **Qualitative Error Tracing**: Are failure cases traced to specific graph construction defects rather than generic model hallucinations?
- **Representation Ablation**: Does the work directly compare structured triples against raw retrieved text while controlling for information content to empirically validate denoising?

---

**Principle 4: Benchmark Modernity and Cross-Domain Generalization in Dynamic Graph RAG Systems**

**Definition:**  
Proposed methods must be evaluated on a representative suite of contemporary benchmarks that span open-domain question answering, multi-hop reasoning, long-form generation, and emerging domain-specific graph reasoning tasks, using both current-generation open-source and proprietary LLMs. Reviewers assess whether the evaluation protocol reflects the present state of the field—avoiding outdated datasets that no longer discriminate among modern methods—and whether results generalize across diverse knowledge domains and model families. This principle demands that authors justify their benchmark selection, test on recent challenging suites such as GraphRAG-Bench when relevant, and acknowledge any scope limitations. Failure to validate on modern benchmarks or backbones undermines claims of general robustness and risks attributing success to benchmark-specific artifacts. One reviewer explicitly flagged the use of outdated benchmarks and recommended newer GraphRAG benchmarks, while another criticized the use of older backbones, emphasizing that robust claims require validation on modern model generations.

**Core Evaluation Criteria:**
- **Benchmark Currency**: Does the evaluation incorporate recent, challenging benchmarks alongside established ones, including domain-specific graph reasoning suites?
- **Cross-Architecture Validation**: Are experiments conducted across multiple model generations and families to demonstrate orthogonality of gains to specific backbone capabilities?
- **Domain Diversity**: Does the evaluation span short-form QA, multi-hop reasoning, long-form generation, and structured domain tasks?
- **Limitation Acknowledgment**: Does the paper explicitly discuss benchmark gaps and scope limitations rather than overstating generalizability?

---

**Principle 5: Component-Level Ablation and Sensitivity Analysis of Retrieval-Planning-Structuring Pipelines**

**Definition:**  
Research on iterative graph-based RAG must provide comprehensive ablation studies that dissect the contributions of individual pipeline stages, including text-to-triple extraction, graph neural encoding, iterative planning versus single-pass retrieval, and multitask learning strategies. Reviewers evaluate whether authors systematically vary operational parameters—such as graph information abundance, training data volume, and extractor model capacity—to establish the robustness and sensitivity of the proposed framework. This principle ensures that performance gains are durable across configuration changes and not brittle artifacts of a specific experimental setup. The rebuttal highlighted this by adding ablations for text-to-triple conversion, graph encoding, LoRA, and multitask training, as well as scaling analyses for training data volume and graph completeness. High-quality work should proactively include such analyses to demonstrate that each component is necessary and that the system remains stable under realistic operational conditions.

**Core Evaluation Criteria:**
- **Component Ablation**: Are all key modules (planning, triple extraction, graph encoding, multitask training) individually ablated in both training and inference phases?
- **Sensitivity to Graph Quality**: Does the work analyze performance as a function of triple extraction quality, graph completeness, or information abundance?
- **Training Efficiency Analysis**: Is there analysis of how performance scales with training data volume and whether strong results are achievable with limited supervision?
- **Hyperparameter and Design Robustness**: Are key architectural choices systematically studied to confirm that gains are not hyperparameter-specific?