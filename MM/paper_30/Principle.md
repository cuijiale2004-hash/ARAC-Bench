**Principle 1: Non-Trivial Adaptation and Theoretical Positioning of Classical Vision Self-Supervised Pretext Tasks for Modern MLLM Post-Training**

**Definition:**  
In evaluating research that repurposes classical computer vision self-supervised learning (SSL) pretext tasks—such as jigsaw puzzles, masked reconstruction, or contrastive learning—for modern multimodal large language model (MLLM) post-training, reviewers must assess whether the adaptation is merely a superficial transplantation or a methodologically grounded contribution. The field has evolved from convolutional networks with specialized decoders to decoder-only transformers producing discrete text tokens, meaning that reward design, task formulation, and optimization paradigms (e.g., RLVR versus SFT) require non-trivial reconsideration. A strong submission must clearly articulate why the classical pretext task needed to be reformulated for the MLLM context, how the new formulation avoids architectural modifications or auxiliary generative modules, and why this specific task is superior to or complementary with dominant text-centric post-training paradigms. Furthermore, the work must be positioned with surgical precision against highly related concurrent and prior art, including reconstruction-based MLLM methods, other jigsaw-based rule-based RL approaches, and emerging "thinking-with-images" methods, clarifying distinctions in supervision signal, target capabilities, and scope.

**Core Evaluation Criteria:**
- **Positioning Against Reconstruction Paradigms**: Does the work clearly distinguish ordering-based signals from dense pixel reconstruction or contrastive learning, and justify why avoiding generative modules is beneficial or sufficient?
- **Differentiation from Concurrent Jigsaw-Based Methods**: Is there explicit comparison with closely related works in terms of task complexity, solvability, and downstream transfer?
- **Claim Validity and Historical Accuracy**: Are claims about the dominance of text-centric post-training or the historical performance of jigsaw SSL properly nuanced and factually accurate?
- **Theoretical Motivation for Adaptation**: Does the paper explain why naive application fails and what specific innovations (e.g., graded rewards, permutation space design) were necessary to make the adaptation successful?

---

**Principle 2: Cross-Architecture Generalization and Multi-Modal Transfer in Vision-Centric Reinforcement Learning Post-Training**

**Definition:**  
Vision-centric post-training methods carry the risk of being overfit to a single base model's idiosyncrasies or a specific modality's data distribution. To be considered a robust contribution, a method must demonstrate that its self-supervised visual task yields consistent improvements across diverse model families and transfers across visual modalities (e.g., from image to 3D spatial reasoning). This principle evaluates whether the authors have moved beyond a proof-of-concept on one flagship model to establish their framework as a general post-training paradigm. Evidence should include experiments on at least one additional high-performing base model with converging architectural standards, multi-task joint training across modalities, and cross-domain evaluations where a model trained on one modality's jigsaw task improves on another's benchmarks. Such validation is essential to rule out dataset contamination, memorization, or architecture-specific reward hacking.

**Core Evaluation Criteria:**
- **Multi-Base-Model Validation**: Are results reported on at least two distinct high-capability MLLMs to confirm architecture-agnostic benefits?
- **Multi-Task and Cross-Modal Compatibility**: Does the work demonstrate successful joint training across modalities (e.g., image + 3D) without catastrophic forgetting or interference?
- **Cross-Domain Transfer Evidence**: Is there explicit evaluation showing that skills learned from a jigsaw task in one domain (e.g., 2D image ordering) transfer to benchmarks in another domain (e.g., 3D spatial understanding)?
- **Benchmark Diversity**: Does the evaluation span a comprehensive suite of benchmarks covering distinct visual skills rather than cherry-picking tasks where improvements are largest?

---

**Principle 3: Mechanistic Decomposition of Visual Sub-Skills and Causal Attribution to Downstream Perceptual Capabilities**

**Definition:**  
A critical weakness in many post-training papers is the reliance on end-to-end benchmark scores as the sole evidence of improved visual understanding. Reviewers must demand mechanistic insight into what specific perceptual sub-skills the self-supervised task cultivates and how those sub-skills causally improve downstream performance. This requires moving beyond aggregate accuracy metrics to provide qualitative reasoning traces (e.g., `<think>` outputs), skill-to-benchmark mapping, and controlled ablations that vary pretext task difficulty. The work should explicitly decompose the target capability into constituent skills—such as fine-grained grounding, spatial relational reasoning, temporal causality, or depth estimation—and provide evidence that improvements on specific benchmarks align with the skills exercised by the pretext task. Without this, claims of "enhanced intrinsic visual understanding" remain empirically unsubstantiated.

**Core Evaluation Criteria:**
- **Qualitative Reasoning Trace Analysis**: Are representative thinking traces provided and analyzed to show the model utilizes genuine visual analysis rather than linguistic heuristics or memorization?
- **Skill-to-Benchmark Attribution**: Does the paper map specific pretext task demands (e.g., inferring spatial layout from patches) to specific downstream benchmark improvements (e.g., VSR for spatial reasoning, DA-2K for depth)?
- **RL versus SFT Generalization Analysis**: Is there evidence distinguishing memorization of the pretext task (SFT) from acquisition of transferable visual skills (RL), ideally showing RL's superior cross-task transfer?
- **Controlled Difficulty Correlation**: Do experiments varying task difficulty (e.g., 2×2 versus 3×3 grids) demonstrate a dose-response relationship between pretext task complexity and downstream gains?

---

**Principle 4: Optimal Difficulty Calibration and Information-Theoretic Design Rationale for Visual Ordering Tasks**

**Definition:**  
The design of visual self-supervised ordering tasks involves delicate trade-offs between task difficulty, information sufficiency per element, and pedagogical value. Reviewers must evaluate whether authors have analytically and empirically established the feasible operating regime of their pretext task, rather than arbitrarily selecting configurations. This includes analyzing why certain difficulty levels fail (e.g., 4×4 image grids where tiles become semantically ambiguous due to information scarcity), how training data properties (resolution, scene complexity, temporal continuity) constrain valid task configurations, and why the chosen reward structure is necessary for learnability. A rigorous submission should provide design heuristics for selecting post-training data and task parameters, justify the permutation space size relative to visual element granularity, and analyze failure modes when the task becomes ill-posed. This principle ensures that improvements stem from sound task design rather than accidental data leakage or trivial pattern matching.

**Core Evaluation Criteria:**
- **Difficulty Sweet Spot Identification**: Is there empirical investigation of multiple task complexities (e.g., grid sizes, clip counts) with analysis of why overly easy or hard configurations underperform?
- **Data Selection Rationale**: Does the paper justify the choice of post-training datasets based on resolution, structural diversity, and temporal characteristics, rather than using them merely because they are standard?
- **Reward Engineering Justification**: Is the reward design (e.g., graded partial credit versus binary) motivated by learnability challenges and supported by ablation studies?
- **Failure Mode and Ill-Posedness Analysis**: Are there explicit analyses of when and why the task fails (e.g., uniform sky patches causing semantic ambiguity) and proposed remedies?

---

**Principle 5: Cost-Benefit Justification and Comparative Efficiency Analysis of RL versus SFT in Visual Post-Training**

**Definition:**  
Reinforcement learning post-training is computationally expensive compared to supervised fine-tuning, often requiring an order of magnitude more GPU-hours due to on-policy rollouts and reward estimation. Reviewers must evaluate whether the substantial computational premium of RL is justified by proportionally superior generalization or final performance, or whether similar gains could be achieved more efficiently. A credible submission must provide transparent cost breakdowns for all modalities, compare RL against SFT not merely on final accuracy but on cross-task transfer robustness, and demonstrate that key hyperparameters (e.g., reward discount factors) do not require prohibitively expensive tuning. Furthermore, the work should contextualize costs relative to the base model's scale and discuss scalability to larger models or longer training regimes. Without this analysis, claims of practical utility remain unconvincing to the community.

**Core Evaluation Criteria:**
- **Transparent Resource Reporting**: Are exact training costs reported (e.g., GPU-hours per modality) for both RL and any SFT baselines?
- **RL versus SFT Generalization Trade-off**: Does the work explicitly demonstrate that RL's higher cost purchases superior generalization or stability that SFT cannot match at any compute budget?
- **Hyperparameter Sensitivity and Robustness**: Are key hyperparameters (e.g., partial-credit discount γ) ablated to show the method is robust and does not require expensive grid searches?
- **Scalability Discussion**: Does the paper discuss whether the cost structure remains acceptable for larger models or longer videos, and identify bottlenecks?