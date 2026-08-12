
**Principle 1: Neuroscientific and Empirical Justification for Cross-Species Pretraining in Neural Foundation Models**

**Definition:**  
In neural foundation model research for brain-computer interfaces, leveraging data across species, brain regions, or tasks requires rigorous justification beyond simple data augmentation. Reviewers expect authors to articulate the biological or mechanistic rationale for why disparate neural recordings (e.g., monkey motor cortex and human speech cortex) can share a common latent representation, particularly when sensor properties like Utah array spiking patterns are the primary commonality. The work must provide empirical evidence—through ablation studies, scaling curves varying data proportions, and comparisons against within-species or within-task baselines—that cross-species pretraining yields transferable representations rather than merely increasing sample size. Authors should also address whether standard data augmentation on single-species data could achieve similar gains, thereby isolating the unique value of cross-species transfer. This principle ensures that cross-dataset aggregation is grounded in both neuroscience reasoning and statistically sound machine learning practice, preventing unjustified data pooling that could mask true generalization failures.

**Core Evaluation Criteria:**
- **Mechanistic Rationale**: Does the work explain why neural computations from different species or tasks should share latent structure (e.g., common probe physiology, shared spiking dynamics) rather than assuming generic transfer?
- **Ablative Evidence**: Are there controlled experiments isolating the contribution of cross-species data from simple data scaling, including human-only and monkey-only pretraining comparisons?
- **Scaling Analysis**: Does the work provide scaling curves showing performance as a function of cross-species data proportion, demonstrating diminishing or increasing returns?
- **Separation from Augmentation**: Does the work distinguish cross-species pretraining benefits from standard data augmentation (e.g., jittering, masking) applied to single-species data?

---

**Principle 2: Architectural Rationale and Fair Comparative Evaluation of End-to-End LLM Decoding in Low-Data Regimes**

**Definition:**  
When proposing end-to-end differentiable architectures that integrate large language models with neural encoders—especially in domains like intracortical BCI where labeled data are extremely scarce—authors must rigorously justify why this approach is preferable to established cascaded pipelines. Reviewers evaluate whether the authors honestly acknowledge current performance gaps, control for decoder differences (e.g., beam search in cascaded vs. greedy decoding in end-to-end), and articulate forward-looking arguments based on scalability, future data availability, or compatibility with higher-level cortical representations. The principle demands that comparisons be fair not only in model capacity but also in inference strategy, ensuring that end-to-end models are not disadvantaged by suboptimal decoding algorithms. Furthermore, authors should address computational overheads such as latency and memory constraints relevant to real-time clinical deployment. By requiring transparent accounting of both present limitations and future potential, this criterion distinguishes visionary architectural proposals from premature replacements of mature, optimized systems.

**Core Evaluation Criteria:**
- **Honest Performance Accounting**: Does the work transparently report where end-to-end methods currently lag behind cascaded baselines and explain why this gap is expected to close?
- **Controlled Comparisons**: Are LLM differences (size, pretraining data) controlled when comparing encoders, and are decoding strategies (beam search, rescoring) matched or their disparities explicitly discussed?
- **Scalability Argument**: Is there a compelling argument that end-to-end architectures will outperform cascaded ones as training data grows or as recordings move to higher-level cortical areas?
- **Efficiency and Deployability**: Are training costs, inference latency, and on-device feasibility reported and contextualized against clinical requirements?

---

**Principle 3: Modality-Specific Inductive Bias Selection and Cross-Modal Alignment Design for Neural-to-Language Translation**

**Definition:**  
In multimodal systems that connect continuous neural time-series to language model embedding spaces, the choice of LLM modality (audio, text, vision, or omni) and the design of alignment mechanisms must be tightly motivated by the properties of the neural data. Reviewers assess whether authors justify why a specific LLM modality provides superior inductive biases—for instance, whether audio-LLMs better capture phonetic and prosodic structure encoded in motor cortex compared to text-only LLMs. Furthermore, the work should empirically validate its alignment approach through ablations of loss weightings, projector architectures, and prompt engineering strategies. Representational similarity analyses or embedding visualizations should demonstrate genuine structural correspondence between neural and linguistic embeddings, confirming that alignment is not merely superficial. This principle ensures that multimodal integration is driven by the intrinsic structure of neural signals rather than by convenience or trendiness in model selection.

**Core Evaluation Criteria:**
- **Modality Choice Justification**: Is the selection of audio, text, or multimodal LLM grounded in the nature of the recorded neural signals (e.g., continuous temporal structure, phoneme-level encoding)?
- **Alignment Mechanism Ablation**: Are the contrastive loss weighting, projector architecture, and prompting strategy systematically ablated to validate their individual contributions?
- **Geometric Validation**: Does the work provide representational similarity analysis or embedding visualizations showing that aligned neural embeddings share geometric structure with the chosen LLM modality space?
- **Cross-Task Generalization**: Does the alignment enable meaningful generalization across related neural tasks (e.g., attempted vs. imagined speech) through shared semantic structure?

---

**Principle 4: Data Leakage Control, Overfitting Mitigation, and Metric Appropriateness in Small-Data Neural Decoding Benchmarks**

**Definition:**  
Neural decoding research inherently operates in small-data regimes where a few participants and limited labeled sentences constitute entire benchmarks. Consequently, reviewers place stringent demands on experimental hygiene: pretraining datasets must not leak test-subject neural patterns into representation learning in ways that inflate generalization estimates, even when labels are withheld. Authors must justify training durations with appropriate regularization and validation-based checkpoint selection, and select evaluation metrics that reflect the clinical or scientific goal. While word error rate is standard for high-fidelity invasive recordings, supplementary analyses such as phoneme error patterns, semantic similarity, or confusion matrices are often necessary to fully characterize decoder behavior. This principle guards against overfitting and inflated claims by enforcing rigorous data handling, training protocols, and multidimensional evaluation.

**Core Evaluation Criteria:**
- **Leakage Safeguards**: Are pretraining procedures scrutinized for subject overlap with test sets, with ablations excluding fine-tuning data to validate true generalization?
- **Overfitting Controls**: Does the work justify extensive training via data augmentation, regularization, and rigorous validation-based early stopping or checkpoint selection?
- **Metric Completeness**: Beyond aggregate WER, does the work provide phoneme-level error analyses, confusion matrices, or semantic similarity metrics suited to the decoding fidelity?
- **Benchmark Integrity**: Are results reported on standardized, held-out competition sets with proper cross-validation, and are claims of state-of-the-art supported by fair leaderboard comparisons?

---

**Principle 5: Differentiation of Integrative Engineering from Fundamental Empirical Contributions in Foundation Model-Based BCIs**

**Definition:**  
As neural decoding systems increasingly assemble existing machine learning components—self-supervised masked autoencoders, transformer encoders, low-rank adaptation, and off-the-shelf LLMs—reviewers scrutinize whether the work offers fundamental empirical insights or merely repackages established techniques. High-quality research in this space must clearly articulate what new capabilities emerge from the integration that individual components do not provide, such as cross-task embedding alignment or record-setting benchmark performance. Authors should frame their novelty accurately, distinguishing architectural innovations from application-specific empirical discoveries, and benchmark against the strongest relevant baselines in both the BCI and broader representation learning literature. The contribution should ideally provide reusable resources, such as pretrained encoders or open-source models, that justify the engineering effort by enabling future research. This principle ensures that integrative work is valued for its emergent scientific or practical impact rather than for the novelty of its individual building blocks.

**Core Evaluation Criteria:**
- **Novelty Framing**: Does the work accurately distinguish between the novelty of individual components (e.g., MAE, transformers) and the novelty of their integration or empirical findings?
- **Emergent Capabilities**: Are there demonstrated capabilities—such as cross-task generalization or modality alignment—that arise specifically from the proposed combination of methods?
- **Baseline Rigor**: Does the work compare against the strongest existing methods, including recent cross-species pretraining approaches and end-to-end neural decoders?
- **Community Utility**: Does the contribution provide reusable resources (e.g., pretrained encoders, open-source models) that justify the engineering effort by enabling future research?