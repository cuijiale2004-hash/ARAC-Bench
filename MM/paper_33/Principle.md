Principle 1: Dynamic Per-Image Visual Vocabulary Construction and Semantic-Patch Alignment in MLLM Decoding

**Definition:**  
The construction of visual vocabularies in MLLMs must move beyond static, dataset-level codebooks or textual coordinate serialization. This principle demands that researchers dynamically expand the LLM's embedding table using visual patch embeddings derived exclusively from the current query image, thereby guaranteeing that every predicted visual reference token corresponds to a real, spatially grounded image region. Such dynamic expansion eliminates the risk of hallucinating visual tokens absent from the input and preserves fine-grained positional information that global codebooks typically discard. The projected visual prototypes must reside in a feature space naturally interpretable by the pretrained LLM, ensuring semantic continuity between textual and visual modalities. Reviewers should assess whether the method establishes a unique, deterministic mapping between predicted output tokens and input image patches, rather than ambiguous nearest-neighbor assignments. Finally, the approach should demonstrate that dynamic vocabulary expansion introduces negligible computational overhead relative to the backbone, preserving the inference efficiency expected of modern MLLMs.

**Core Evaluation Criteria:**
- **Avoidance of hallucinated visual tokens**: Does the method guarantee that every predicted visual token corresponds to an actual patch in the query image, rather than retrieving entries from a pretrained, dataset-level codebook that may be absent from the current input?
- **Feature-space alignment**: Are the dynamically inserted visual prototypes projected into the LLM's native semantic space such that autoregressive prediction over visual indices is as stable as textual next-token prediction?
- **Deterministic patch-to-token correspondence**: Is the mapping between an output visual reference token index and its source image patch unique and spatially preserved, preventing confusion between similar-looking objects or repetitive patterns?
- **Computational overhead control**: Does dynamic vocabulary expansion introduce less than marginal cost in memory and latency relative to the backbone LLM, quantified across varying image resolutions?

---

Principle 2: Unified Autoregressive Interleaving of Textual and Visual Reference Tokens for Dense Grounding

**Definition:**  
A unified multimodal paradigm must enable the LLM to autoregressively generate sequences where textual tokens and visual reference tokens are freely interleaved without intermediate formatting or parsing stages. This principle evaluates whether the proposed method replaces brittle coordinate-string generation with a native token-level interface that supports dense prediction tasks such as segmentation and pixel-level grounding. The output space should be structurally homogeneous, allowing the same autoregressive head to produce both semantic labels and spatial references within a single decoding pass. Reviewers must verify that the interleaving mechanism does not disrupt the LLM's core language modeling capabilities or introduce cross-modal interference during training. Furthermore, the paradigm should generalize across heterogeneous vision tasks—detection, segmentation, referring expression comprehension, and grounded captioning—without requiring task-specific output formats or post-hoc regex extraction.

**Core Evaluation Criteria:**
- **Native interleaving capability**: Can the model autoregressively emit textual labels and visual reference tokens within the same decoding stream without format parsers, JSON templates, or digit-by-digit coordinate serialization?
- **Support for dense prediction**: Does the unified format extend beyond axis-aligned bounding boxes to pixel-accurate masks and arbitrary polygonal regions without architectural modifications?
- **Preservation of linguistic fluency**: Does the insertion of visual tokens into the output vocabulary degrade the LLM's language modeling performance or cause modality interference in open-ended generation?
- **Task-agnostic output homogeneity**: Is the same decoding head and prompting strategy used across detection, segmentation, and grounded captioning, or does the method resort to task-specific formatting rules?

---

Principle 3: Robust Sparse Supervision and Training Stability under Dynamic Visual Token Vocabularies

**Definition:**  
Training MLLMs to predict visual tokens demands robust supervisory strategies because dense foreground-token supervision often causes the decoder to overfit to trivial spatial aggregations or redundant patch clusters. This principle requires explicit justification for the sampling strategy used to select ground-truth visual tokens, such as random sparse sampling versus exhaustive foreground masking or boundary-aware heuristics. The training objective must remain stable despite a dynamically changing target vocabulary across forward passes, necessitating carefully designed loss functions like masked cross-entropy that suppress unsampled tokens without corrupting gradient flow. Reviewers should examine whether ablation studies clearly isolate the contribution of the sampling strategy from other architectural changes and demonstrate performance collapse under dense supervision. Additionally, the method must show that sparse sampling improves generalization by forcing the model to infer complete object extents from partial patch references rather than memorizing fixed spatial layouts.

**Core Evaluation Criteria:**
- **Sampling strategy justification**: Is the choice of sparse ground-truth token selection rigorously ablated, with clear evidence that dense supervision harms generalization?
- **Stability under dynamic targets**: Does the training loss remain effective when the set of target visual indices changes every forward pass, and are unsupervised tokens properly masked to avoid penalizing valid but unselected patches?
- **Generalization from partial references**: Does the model learn to infer complete object geometry from a small subset of reference tokens, rather than memorizing fixed spatial layouts that fail when inference tokens differ?
- **Prevention of decoder collapse**: Is there evidence that the decoder does not simply learn to output trivial bounding boxes or average masks when given sparse supervision, verified through qualitative failure analysis?

---

Principle 4: Lightweight Decoding and Cross-Task Generalization for Unified Visual Perception Outputs

**Definition:**  
The translation from predicted visual reference tokens to structured visual outputs must be executed by a lightweight, parameter-efficient decoder that does not negate the computational benefits of sparse token prediction. This principle evaluates whether the decoder architecture—often based on attention-based aggregation—can simultaneously produce bounding boxes, segmentation masks, and confidence scores from grouped visual token hidden states without task-specific branches. Reviewers must scrutinize whether joint multi-task training yields positive transfer across perception tasks or whether task interference degrades individual performance. The system should also preserve scalability to high-resolution inputs, ensuring that the number of visual tokens scales gracefully with image size without quadratic blowups in the LLM forward pass. Finally, the work must provide rigorous efficiency benchmarks quantifying memory overhead, inference latency, and sequence-length reduction relative to coordinate-based textual generation, proving that visual-token prediction is not only more accurate but also practically deployable.

**Core Evaluation Criteria:**
- **Decoder parameter efficiency**: Is the visual-to-output decoder sufficiently lightweight that it does not dominate computational cost or overfit to training distributions?
- **Positive multi-task transfer**: Does joint training across detection, segmentation, and grounding tasks consistently improve or maintain per-task performance relative to individually fine-tuned specialists?
- **Scalability to high-resolution inputs**: Does token count and memory consumption scale linearly or sub-quadratically with image resolution, and are native-resolution inputs supported without forced resizing?
- **Inference efficiency gains**: Are quantitative benchmarks provided showing reduced autoregressive steps, lower latency, or smaller memory footprint compared to textual coordinate generation at equivalent or higher accuracy?