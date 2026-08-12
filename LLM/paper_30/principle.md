
**Principle 1: Cross-Model Internal Representation Alignment and Projectability in Heterogeneous LLM Communication**

**Definition:**  
When proposing communication via internal representations (KV-Cache, hidden states) across different LLM architectures or sizes, the work must rigorously demonstrate that representations are alignable and projectable. This involves explicit token-level and layer-level alignment strategies that reconcile divergent tokenizers, sequence lengths, and model depths. It further requires empirical evidence—such as oracle experiments, representation visualizations, or reconstruction losses—showing that source representations can be mapped into the target model's latent space without catastrophic semantic distortion. Because different models instantiate fundamentally different geometric manifolds in their activation spaces, naive projection risks collapsing task-relevant structure or introducing incompatible positional or attention biases. Consequently, the community must demand justification for why a specific internal representation (e.g., KV-Cache over hidden states) is the optimal communication medium, grounded in compatibility with existing inference engines and the model's native training objective. This principle ensures that cross-model communication claims rest on mechanistic feasibility rather than on post-hoc accuracy correlations alone.

**Core Evaluation Criteria:**
- **Rigor of Alignment Strategy**: Are tokenization differences handled robustly (e.g., chat-template alignment, one-to-many mapping heuristics)? Is the layer-alignment scheme (e.g., terminal, depth-normalized) justified and empirically validated?
- **Empirical Evidence of Convertibility**: Does the work provide oracle experiments (e.g., cross-model cache transformation, enrichment oracles, t-SNE/MSE analysis) proving that source representations are meaningfully transformable into the target space?
- **Analysis of Architectural Barriers**: Does the work discuss limitations when hidden dimensions, attention-head counts, or layer depths differ significantly, and explain how these architectural mismatches are mitigated?
- **Medium Justification**: Is there a principled argument for why the chosen internal representation (e.g., KV-Cache) is superior to alternatives such as hidden states, grounded in system compatibility or semantic encoding properties?

---

**Principle 2: Cost-Benefit Trade-off and Scalability Analysis of Pairwise vs. Universal Communication Modules**

**Definition:**  
Research introducing learned adapters or fusers for inter-model communication must provide transparent, reproducible quantification of training costs—including GPU hours, peak memory, wall-clock time, and convergence steps—and contrast these against tangible inference benefits such as end-to-end latency reduction and accuracy gains. A communication paradigm that requires expensive pairwise training for every sharer-receiver combination faces severe practical barriers in real-world deployments with heterogeneous model zoos. Therefore, the work must articulate a credible scalability pathway beyond the pairwise setting, analyzing whether the design inherently demands O(N²) modules or can achieve O(N) complexity via universal projectors, shared semantic spaces, or additive residual composition. Preliminary evidence for many-to-one or many-to-many communication, even if exploratory, is essential to establish that the paradigm is not permanently confined to dyadic interaction. Reviewers should treat the absence of training-inference trade-off analysis and scaling discussion as a critical weakness, because a method whose overhead dwarfs its savings cannot be adopted in production multi-agent systems. Ultimately, the burden is on authors to prove that the one-time training cost is amortized across downstream tasks and diverse model combinations.

**Core Evaluation Criteria:**
- **Training Cost Transparency**: Are GPU hours, peak memory, wall-clock time, and convergence dynamics reported per model pair? Is the cost explicitly contrasted against per-query inference savings?
- **Scalability Pathway**: Does the work discuss the complexity of extending to N models? Are preliminary results or architectural prototypes provided for many-to-one or many-to-many settings?
- **Justification of Pairwise Specialization**: If pair-specific training is required, is the rationale empirically justified by gains that significantly exceed reusable or universal adapter approaches?
- **Additive Composition**: Does the work investigate whether multiple sharers can be composed without retraining (e.g., residual summation), and is interference between multiple sources quantified?

---

**Principle 3: Baseline Fairness and Benchmark Suitability for Multi-LLM Communication Evaluation**

**Definition:**  
Evaluations of novel communication paradigms must employ baselines that are equivalently empowered, controlling for confounding factors such as additional supervised fine-tuning data, prompt sophistication, and parameter count. If the proposed method is trained on a large general-purpose corpus, text-to-text baselines should also be fine-tuned on that same corpus, and prompting strategies should be optimized rather than reduced to simplistic templates that artificially deflate competitor performance. Beyond data parity, benchmark selection must reflect realistic multi-agent workloads; reliance solely on single-turn, single-answer question-answering tasks risks conflating communication quality with static knowledge retrieval. The field should require demonstrations on collaborative workflows—such as multi-step reasoning, tool-use orchestration, or agentic planning—where sustained semantic coordination is necessary. Without such controls, reviewers cannot disentangle gains arising from the communication medium itself from gains attributable to incidental training effects or underpowered baselines. Fair comparison and task realism are therefore indispensable for establishing a true scientific contribution.

**Core Evaluation Criteria:**
- **Controlled Baseline Comparison**: Are text-based baselines fine-tuned on the identical dataset used for the communication module? Are prompts and decoding strategies optimized to maximize knowledge transfer for all compared methods?
- **Task Diversity and Realism**: Does the evaluation include multi-agent or multi-step collaborative workflows, or does it rely exclusively on static single-turn benchmarks that do not stress-test semantic coordination?
- **Isolation of Medium Benefit**: Does the work disentangle performance gains arising from the novel communication medium versus gains from additional trainable parameters, extra training data, or fine-tuning effects?
- **Data Contamination Avoidance**: When using auxiliary training sets (e.g., for fusers), are evaluation benchmarks strictly held out, and is overlap between training and evaluation data explicitly checked?

---

**Principle 4: Mechanistic Evidence of Semantic Transfer and Information Content Preservation**

**Definition:**  
Claims that internal representations enable richer semantic communication than text must be supported by mechanistic evidence beyond aggregate accuracy metrics, which alone cannot distinguish representation enrichment from spurious correlation or additional trainable capacity. High-quality work should analyze the intrinsic dimensionality of fused representations—using metrics such as effective rank, singular-value entropy, or activation covariance—to demonstrate that the transferred signal is genuinely informative and non-redundant. Qualitative inspections, including concrete examples of knowledge transfer across model boundaries and per-category performance breakdowns, are necessary to illuminate what types of information survive projection and where they improve decision boundaries. Furthermore, authors must avoid black-box reasoning that infers successful communication purely from improved downstream metrics; instead, they should establish a causal chain linking representation-space modification to specific behavioral changes in the receiver. This standard is particularly important when the communication medium is implicit and uninterpretable, as with KV-Cache fusion, where surface-level text outputs may mask internal representation distortions. Mechanistic transparency elevates the contribution from an engineering trick to a scientifically grounded communication primitive.

**Core Evaluation Criteria:**
- **Representation-Level Evidence**: Are metrics like effective rank, singular-value distribution, or activation entropy used to demonstrate non-redundant information injection? Are visualizations of representation-space alignment provided?
- **Qualitative and Granular Analysis**: Are concrete examples provided showing successful cross-model knowledge transfer (e.g., knowledge-cutoff bridging)? Is there per-category or per-task breakdown identifying where the method helps or hurts?
- **Causal Attribution**: Does the work establish a causal link between representation enrichment and downstream performance, rather than relying solely on end-accuracy correlations?
- **Avoidance of Black-Box Reasoning**: Does the study distinguish between "performance improvement" and "semantic transfer quality," ensuring that gains are not merely artifacts of increased model capacity or data exposure?

---

**Principle 5: Robustness Analysis and Failure Mode Characterization in Representation-Based Communication**

**Definition:**  
Because representation-based communication bypasses explicit natural-language checks, it introduces distinct failure modes—most notably the risk that a weak or misaligned sharer will inject noisy, biased, or factually incorrect representations that a receiver cannot easily filter, leading to irreversible cache contamination and degraded outputs. Rigorous research must explicitly characterize these failure modes through both quantitative degradation analysis and concrete qualitative examples, identifying capacity-mismatch thresholds below which communication becomes harmful. The efficacy of proposed mitigation mechanisms—such as learnable gating, dynamic weighting, or residual scaling—must be validated via ablation studies that show selective suppression of harmful inputs rather than uniform acceptance. Additionally, the behavior of these selective mechanisms should be analyzed across training regimes and model scales to ensure they generalize beyond the specific dyad studied. Reviewers should scrutinize whether the work acknowledges and investigates cases where the method underperforms text-based alternatives, as silent omission of negative results undermines scientific credibility. A comprehensive robustness profile is essential before deploying such opaque communication channels in safety-critical multi-agent systems.

**Core Evaluation Criteria:**
- **Explicit Failure Taxonomy**: Does the work identify, categorize, and quantify specific failure cases (e.g., incorrect knowledge injection, representation collapse, misleading weak-to-strong transfer) with concrete examples?
- **Mitigation Mechanism Efficacy**: Are learned gates, dynamic weights, or filtering mechanisms ablated to prove they suppress harmful sharer inputs? Is their behavior analyzed across layers and training regimes?
- **Capacity-Mismatch Sensitivity**: Does the work analyze how relative model capacities (strong sharer vs. weak receiver, or vice versa) affect communication efficacy, and identify thresholds where adding a model degrades performance?
- **Out-of-Distribution and Scaling Robustness**: Are diminishing returns or failures evaluated as model scales increase? Does the method maintain benefits across diverse sequence lengths and input distributions?