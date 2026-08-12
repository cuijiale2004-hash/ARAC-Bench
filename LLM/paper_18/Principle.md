**Principle 1: Mechanism Clarity and Theoretical Grounding in Hybrid Linear-Softmax Attention State-Transition Architectures**

**Definition:**  
The evaluation must ascertain whether the proposed hybrid attention architecture is described with sufficient mechanistic clarity to permit replication and theoretical scrutiny. Reviewers expect a rigorous explanation of how the softmax-attention branch handles local, intra-step token dependencies while the linear-attention branch compresses cross-step history into a compact state matrix, avoiding conflation of these distinct information flows. The theoretical motivation for adopting linear attention—particularly its gradient-descent interpretation as test-time training and its compatibility with existing Transformer interfaces—must be articulated rather than assumed. Authors should provide visual schematics of attention matrices, pseudocode, or diagrams illustrating both training and inference procedures to reduce cognitive load. Furthermore, the work must clearly distinguish between “performance improvement” and “mechanistic efficiency,” establishing a causal link between the state-transition formulation and observed gains rather than relying on black-box empirical success.

**Core Evaluation Criteria:**
- **Architectural Coherence**: Is the division of labor between the local softmax-attention submodule and the global linear-attention submodule rigorously justified, with clear definitions of what constitutes the “reasoning state” and how it evolves?
- **Theoretical Motivation**: Does the work leverage the test-time-training interpretation of linear attention (or equivalent theoretical properties) to justify why state matrices can preserve reasoning-critical information, rather than treating linear attention as merely a computational shortcut?
- **Expository Completeness**: Are schematics, pseudocode, or attention-matrix visualizations included in the main text to clarify how tokens retrieve historical information from the state and how the gating mechanism modulates this retrieval?
- **Causal Attribution**: Does the paper avoid inferring mechanism validity solely from end-task accuracy, instead tracing efficiency gains explicitly to complexity reduction and information-flow design?

---

**Principle 2: Cross-Domain Generalization and Robustness of CoT Segmentation and Reasoning-State Clustering**

**Definition:**  
This principle assesses whether the step-level segmentation and reasoning-state abstraction generalize robustly across diverse tasks and reasoning modalities. The segmentation strategy must preserve semantic independence among steps to prevent information loss when the KV cache is cleared between transitions, and the clustering or annotation of thinking patterns must minimize type conflicts that could lead to overfitting or underfitting. Reviewers expect evidence that the framework is not exploiting domain-specific linguistic artifacts—such as math-specific transitional tokens—by validating on scientific and code benchmarks despite training exclusively on mathematical data. The granularity of state abstraction (e.g., cluster count, state dimension) should be justified, with analyses showing balanced distributions of reasoning patterns and controlled risk of degenerate clustering. Ultimately, the framework must demonstrate task-agnostic controllability through special tokens or monitoring mechanisms without requiring manual redesign for each new domain.

**Core Evaluation Criteria:**
- **Semantic Independence of Segmentation**: Does the segmentation method ensure that each reasoning step maintains low linguistic dependency on previous steps, so that KV-cache eviction does not destroy necessary context?
- **Clustering Robustness**: Is the thinking-pattern clustering (e.g., K-means with balanced cluster sizes) validated to avoid overfitting or underfitting to specific reasoning types, and is the cluster granularity justified empirically?
- **Domain Portability**: Does the framework generalize beyond the training domain (e.g., from math to scientific reasoning and code generation) without domain-specific architectural modifications?
- **Controllability and Interpretability of Patterns**: Can the framework monitor and guide reasoning processes via special tokens across diverse tasks, or is its controllability limited to narrow reasoning structures?

---

**Principle 3: Comprehensive Baseline Coverage and Empirical Verification of Orthogonality to Length-Reduction Methods**

**Definition:**  
A thorough evaluation demands comparison against a comprehensive suite of recent baselines spanning reinforcement learning, supervised fine-tuning, prompting, and cache-compression paradigms, rather than a narrow selection of weak or outdated methods. Because the paper explicitly claims that its approach is orthogonal to length-reduction techniques, reviewers require empirical integration experiments—combining the proposed method with representative RL-based and SFT-based shortcuts—to verify compatibility and isolate additive gains. The baseline selection must reflect the current state of the art, including methods that control reasoning length via format rewards, model merging, or token budgets. Authors must clearly disaggregate whether reported efficiency improvements stem from faster per-token generation, reduced memory overhead, or merely shorter output length, ensuring fair metrics are used across paradigms. Finally, the rebuttal and revision process should demonstrate responsiveness to missing baseline criticisms by incorporating these experiments into the main body rather than relegating them to appendices.

**Core Evaluation Criteria:**
- **State-of-the-Art Baseline Inclusion**: Are the latest RL-based, SFT-based, and prompt-based efficient-reasoning methods included, rather than only older KV-cache compression or simple prompt-control techniques?
- **Empirical Orthogonality Tests**: Does the paper provide experiments that combine the proposed method with length-reduction baselines (e.g., L1-Max, model merging, anytime reasoning) to prove that per-token acceleration is additive to token-count reduction?
- **Fair Metric Disaggregation**: Are efficiency metrics (latency, memory, FLOPs) reported separately from output-length metrics, ensuring that gains are not conflated with CoT truncation?
- **Responsiveness to Reviewer Critiques**: When baseline gaps are identified during review, are the corresponding experiments moved into the main text to strengthen the paper’s empirical completeness?

---

**Principle 4: Statistical Rigor of Evaluation Protocols and Joint Sensitivity Analysis for Efficiency-Performance Trade-offs**

**Definition:**  
Reviewers require strict adherence to statistically reliable evaluation protocols, including multiple independent runs with temperature-based sampling and reporting of pass-at-k metrics, rather than relying solely on greedy decoding or single-run results that may mask high variance on difficult benchmarks. A rigorous study must include joint and marginal sensitivity analyses of key hyperparameters—such as distillation loss weights and state-correction magnitudes—via heatmaps or contour plots to establish reproducible operating ranges and avoid ad hoc selections. Data efficiency should be explicitly analyzed, demonstrating how model performance scales with training set size and whether the method remains viable when annotation resources are constrained. Experimental coverage must extend beyond the primary training domain to test generalization, using standardized benchmarks across mathematics, scientific reasoning, and code generation. Additionally, training and inference costs must be reported in absolute terms (wall-clock time, GPU hours, memory footprint) to contextualize efficiency claims against practical deployment constraints.

**Core Evaluation Criteria:**
- **Reliable Inference Protocols**: Are results reported with standard sampling strategies (e.g., temperature = 0.6, top-k = 0.95) and multiple runs (e.g., 32 samples) alongside greedy decoding to ensure statistical reliability on competitive benchmarks?
- **Hyperparameter Transparency**: Is a joint or marginal sensitivity analysis (e.g., heatmaps, ablation curves) provided for critical hyperparameters such as distillation weight and reasoning-direction correction magnitude?
- **Data-Efficiency Characterization**: Does the paper analyze performance as a function of training data quantity, and does it justify the scale of supervision relative to RL-based alternatives?
- **Absolute Cost Reporting**: Are training time (GPU hours), inference latency, and peak memory usage reported in absolute units to substantiate claims of linear-time and constant-memory complexity?

---

**Principle 5: Capability Preservation and Interpretability under Attention Complexity Reduction in Long-Horizon Reasoning**

**Definition:**  
This principle examines whether the framework genuinely preserves or enhances reasoning capability while reducing computational complexity, ensuring that efficiency gains do not derive from hidden trade-offs that degrade output quality. Authors must empirically validate the theoretical complexity reductions—linear attention cost and near-constant KV-cache memory—across varying context lengths, particularly at extreme scales such as 32K tokens, with controlled latency and memory measurements. The work should diagnose and quantify failure modes specific to long-horizon reasoning, such as repetitive generation or overthinking, and attribute mitigation to the proposed state-based correction strategy rather than to unrelated decoding artifacts. Interpretability analyses, including gating-score trajectories within reasoning steps and state-derived process-reward estimates, are expected to illuminate how historical information is retrieved and how noisy steps are detected. Ultimately, the framework must demonstrate that it supports test-time scaling by maintaining full CoT length and interpretability, avoiding the reasoning-capacity losses associated with methods that aggressively compress or truncate thought chains.

**Core Evaluation Criteria:**
- **Validated Complexity Reduction**: Is the linear-time computational complexity and near-constant KV-cache memory overhead demonstrated empirically across a range of CoT lengths, including very long contexts (> 4K, up to 32K)?
- **Reasoning Quality Assurance**: Does the method maintain or improve accuracy relative to the base model, proving that efficiency gains are not purchased through degradation of reasoning depth or test-time scaling capacity?
- **Failure-Mode Diagnosis**: Are pathological behaviors such as repetitive generation (RC rate) and overthinking explicitly measured, and is their reduction causally linked to the state-based reasoning strategy rather than to external decoding tricks?
- **State Interpretability**: Does the paper provide analyses (e.g., gating scores by token position, process-reward estimation from state transitions) that make the reasoning-state dynamics interpretable and verifiable?