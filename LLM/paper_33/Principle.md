**Principle 1: Mechanism-Aware Diagnosis of Information Distraction and Coverage-Distraction Trade-off in Long-Context Retrieval**

**Definition:**  
In the field of retrieval-augmented generation for large language models, it is insufficient to frame the research as a binary choice between dense retrieval and full long-context processing. Reviewers expect authors to provide a mechanistic diagnosis of why retrieved passages fail, specifically identifying the "distraction" phenomenon—where individually relevant passages collectively interfere with reasoning and degrade output quality. A high-quality submission must empirically characterize the coverage-distraction trade-off, demonstrating that performance follows a non-monotonic (often inverted-U) trajectory as more passages are added. The work should distinguish between failures due to missing information (low coverage) and failures due to cognitive interference (high distraction), and provide causal evidence linking specific passage combinations to answer correctness degradation. This perspective is crucial because it shifts the focus from heuristic retrieval tuning to principled, model-aware context selection.

**Core Evaluation Criteria:**
- **Depth of Mechanistic Analysis**: Does the work explain why additional passages degrade performance beyond simple irrelevance, identifying combinatorial interference effects among retrieved passages?
- **Empirical Characterization of Trade-off**: Are inverted-U curves or equivalent evidence provided to show the non-monotonic relationship between retrieval scope and performance?
- **Separation of Failure Modes**: Does the study clearly distinguish distraction-induced errors from recall-induced errors (e.g., via controlled experiments with gold vs. distracting passages)?
- **Causal Grounding**: Is there evidence that manipulates retrieval set composition to establish a causal link between distraction and degraded LLM outputs, rather than relying solely on aggregate accuracy metrics?

---

**Principle 2: Interpretability and Search Space Abstraction in Contiguity-Constrained Adaptive Retrieval Strategy Design**

**Definition:**  
When proposing learning-based adaptive retrievers, authors must rigorously justify the structural constraints imposed on the retrieval policy. In particular, band-based or contiguous-interval retrieval represents a significant abstraction of the combinatorial subset-selection problem, and reviewers evaluate whether this constraint is theoretically or empirically warranted. A credible submission should demonstrate that the chosen abstraction (e.g., selecting a quantile band rather than arbitrary subsets) improves sample efficiency, training stability, or generalization, while remaining expressive enough to capture useful context. Furthermore, because these retrievers often operate as opaque learned modules, reviewers demand behavioral interpretability through case studies, clustering analyses, and controlled experiments (e.g., sliding-window sweeps) that reveal how retrieval decisions relate to properties of the similarity distribution. This principle ensures that adaptive retrieval is treated as a scientific design choice rather than an unmotivated architectural convenience.

**Core Evaluation Criteria:**
- **Structural Justification**: Is the contiguity constraint or other structural abstraction motivated by empirical analysis (e.g., local coherence of semantic similarity) or theoretical arguments about search space complexity?
- **Search Space Tractability**: Does the method avoid combinatorial explosion in the action space, and is there evidence that the abstraction leads to better convergence or generalization than unconstrained alternatives?
- **Behavioral Interpretability**: Are concrete case studies, visualizations, or clustering analyses provided to explain when and why the retriever narrows or expands its selection band?
- **Validation via Controlled Sweeps**: Are experiments such as fixed-width sliding-window analyses performed to validate that performance peaks align with the learned retrieval bands?

---

**Principle 3: Target-LLM Capability Calibration and Cross-Architecture Robustness in Adaptive Retrieval**

**Definition:**  
Adaptive retrieval strategies must be evaluated for their sensitivity to the long-context reasoning capabilities of the specific LLM they serve, rather than assuming a universal optimal retrieval width. Reviewers expect that a robust method will not apply a uniform retrieval policy across all models; instead, it should internalize the target model's capacity, retrieving narrower bands for distraction-prone, limited-capacity models and wider bands for state-of-the-art models with stronger noise resilience. The evaluation therefore centers on whether the retrieval policy demonstrably adapts to model scale, architecture, and training regimen, and whether improvements hold consistently across this spectrum. Cross-architecture robustness is especially important because retrieval modules are often positioned as portable bridges between embedders and LLMs; if performance gains collapse when switching from a 7B to a 70B model, or from an open to a closed API, the practical utility of the method is severely limited. Finally, authors should analyze whether retrieval behavior correlates with known indicators of model capability to prove that the policy is truly capability-aware.

**Core Evaluation Criteria:**
- **Model-Aware Adaptation**: Does the retrieval strategy quantifiably adjust its passage selection width based on the target LLM's long-context capability (e.g., using fewer tokens for weaker models)?
- **Cross-Architecture Consistency**: Are improvements maintained across a diverse set of LLM families, sizes, and training paradigms (open vs. closed source)?
- **Sensitivity Analysis**: Does the paper analyze how retrieval behavior changes as a function of LLM capacity, rather than reporting only aggregate averages?
- **Avoidance of One-Size-Fits-All**: Does the method explicitly reject or outperform fixed-k retrieval heuristics that ignore model-specific distraction thresholds?

---

**Principle 4: Comprehensive End-to-End Cost-Benefit Analysis for Learning-Based Lightweight Retrieval Modules**

**Definition:**  
Learning-based retrieval methods that require reward signals from a target LLM introduce non-trivial training and inference costs, and reviewers treat the omission of these costs as a serious weakness that undermines claims of practical efficiency. A complete evaluation must transparently report the total computational budget, including the number of LLM forward passes required for policy-gradient reward estimation, wall-clock training time, GPU consumption, and per-example inference latency across all pipeline stages. Beyond static reporting, reviewers expect cumulative cost analyses that illustrate break-even points: for instance, how many inference queries are required before the token savings and accuracy gains of the learned retriever outweigh its upfront training expense. The analysis should also benchmark against efficiency-oriented baselines such as cache-augmented generation, and convert costs into standardized units to enable fair comparison across disparate hardware and API pricing regimes. Without such transparency, assertions of efficiency remain unverified and the method's deployment viability cannot be assessed.

**Core Evaluation Criteria:**
- **Training Cost Transparency**: Are the total number of LLM API calls, GPU hours, and wall-clock training time explicitly reported?
- **Inference Overhead Benchmarking**: Is the retriever module's latency measured end-to-end, including embedding computation, retrieval selection, and LLM generation, relative to rerankers and long-context baselines?
- **Cumulative Cost-Benefit Analysis**: Are break-even plots provided showing total cost (training + inference) against performance across varying deployment scales?
- **Fair Efficiency Comparisons**: Are appropriate efficiency baselines (e.g., prompt caching, cache-augmented generation) included, and are costs normalized to common units like USD?

---

**Principle 5: Cross-Embedder and Cross-Corpus Generalization Robustness in Similarity-Only Retrieval Policies**

**Definition:**  
Retrieval policies that operate exclusively on similarity scores—without access to raw text—are vulnerable to overfitting the statistical peculiarities of a specific corpus or embedding model, making generalization a primary concern for reviewers. Reviewers therefore scrutinize whether the learned policy captures generalizable semantic patterns or merely memorizes corpus-specific similarity-shape distributions that fail to transfer. High-quality research must provide evidence of robustness across embedding spaces, requiring analysis of scale and rank alignment between embedders to explain transferability. Additionally, zero-shot transfer must be evaluated across corpora with divergent semantic domains, and the paper must explain success or failure through task-structural alignment rather than treating positive results as coincidental. Finally, architectural choices specific to score-only processing must be ablated against simpler alternatives to prove their necessity for distributional robustness rather than overfitting capacity.

**Core Evaluation Criteria:**
- **Embedder Generalization**: Is cross-embedder transfer analyzed, with explicit metrics for scale and rank alignment between source and target embedding models?
- **Corpus Distribution Robustness**: Are zero-shot evaluations conducted on corpora with different semantic profiles, accompanied by similarity-distribution analysis?
- **Task-Structure Alignment**: Is zero-shot transfer explained and validated through task structural similarity (e.g., matching multi-hop training to multi-hop evaluation), with misalignment experiments to rule out coincidence?
- **Architectural Necessity**: Are key design choices (e.g., periodic embeddings, sequence models vs. MLPs) ablated to demonstrate that complexity is required for robustness rather than overfitting?