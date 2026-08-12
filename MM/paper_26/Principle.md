Principle 1: Strict Verification of Training-Free Claims Against Hidden Data-Dependent Calibration or Head Selection Heuristics

**Definition:**  
In the subfield of training-free adaptation of large multimodal models for dense visual tasks, reviewers must rigorously scrutinize whether the proposed method is strictly free from any data-driven calibration, dataset-specific head discovery, or model-specific heuristic tuning. A method’s claim of being “training-free” can be substantially weakened if it implicitly relies on benchmark samples to identify attention heads, spatial exclusion rules, or normalization statistics that are not computed on-the-fly during inference. This principle ensures that the boundary between truly zero-shot inference and lightly-supervised engineering is clearly delineated, preserving the scientific integrity and reproducibility of the training-free paradigm.

**Core Evaluation Criteria:**
- **Absence of Dataset-Dependent Calibration:** Does the method require any dataset-specific sampling (e.g., using 1,000 image-text pairs to discover localization heads) or manual inspection of attention patterns prior to inference?
- **Inference-Time-Only Computation:** Are all aggregation weights, normalization factors, or selection criteria computed dynamically during the forward pass without gradient updates, learned parameters, or offline optimization?
- **Clear Distinction from Prior Art:** Does the paper explicitly contrast its strictly inference-time design against prior approaches that use data-dependent head selection, and does it demonstrate superior cross-model portability as a direct consequence?

---

Principle 2: Comprehensive Baseline Inclusion with Native Model Capabilities and Fair Backbone Alignment

**Definition:**  
When evaluating a new training-free pipeline that leverages existing foundation models, it is essential to establish the strongest possible simple baseline: the native capabilities of the underlying MLLM itself. Reviewers must assess whether the paper fairly isolates the incremental contribution of the proposed method from the inherent power of the backbone by comparing against the MLLM’s own grounding or pointing outputs when coupled with identical refinement tools. Without this comparison, it is impossible to determine whether a complex multi-stage pipeline adds genuine value or merely repackages a pre-existing model ability.

**Core Evaluation Criteria:**
- **Native Grounding Evaluation:** Does the paper evaluate the MLLM’s native spatial grounding capability (e.g., direct keypoint or bounding box generation) combined with the same segmentation refinement pipeline (e.g., SAM2) used by the proposed method?
- **Controlled Backbone Comparisons:** Are competing methods evaluated using the exact same MLLM checkpoints (e.g., identical parameter counts and architecture versions) to prevent confounding improvements from backbone differences?
- **Fair Adaptation of Existing Methods:** Does the paper attempt reasonable adaptations of prior art (e.g., extending an image-domain method to video with comparable prompting strategies) rather than comparing against handicapped or unmodified implementations?

---

Principle 3: Cross-Architecture Generalization without Model-Specific Attention Heuristics

**Definition:**  
A core criterion for training-free methods that probe internal model representations is their ability to generalize across diverse MLLM families without architecture-specific hacks. Reviewers must evaluate whether the method relies on attention patterns unique to a single model family—such as head-selection rules tied to specific spatial entropy profiles or exclusion heuristics for attention sinks that only appear in certain architectures—or whether it is founded on general mechanisms applicable to any transformer-based multimodal model. This principle distinguishes transferable scientific insights from brittle, overfitted engineering.

**Core Evaluation Criteria:**
- **Multi-Family Validation:** Are experimental results reported across at least three distinct MLLM architectures (e.g., LLaVA-based, InternVL, Qwen-VL) to demonstrate broad applicability?
- **Absence of Model-Specific Rules:** Does the method avoid dependencies on model-specific spatial priors (e.g., “exclude bottom-row attention”) or fixed head subsets discovered for individual models?
- **Zero-Shot Portability:** Does the method maintain stable performance when applied to a new MLLM without manual retuning of hyperparameters, and does it outperform heuristics that fail to transfer across architectures?

---

Principle 4: Mechanistic Justification and Causal Ablation of Attention Refinement Components

**Definition:**  
Proposed attention manipulation mechanisms—such as contrastive fusion or multi-scale complementary fusion—must be supported by a mechanistic account of why raw attention maps fail and how each refinement step remediates a specific pathology (e.g., visual attention sinks, low-resolution temporal sparsity, or cross-modal token imbalance). Reviewers should expect ablation studies that not only quantify performance drops but also provide causal explanations for counter-intuitive observations, such as why removing a single module degrades performance below a seemingly comparable baseline. This ensures that the design is principled rather than an accidental combination of simple operations.

**Core Evaluation Criteria:**
- **Failure Mode Diagnosis:** Does the paper clearly diagnose specific pathologies in raw attention rollout (e.g., spurious background activations, sink tokens, or reasoning-irrelevant hotspots) that motivate the proposed fusions?
- **Component-Level Causality:** Do ablations isolate each fusion mechanism and explain, both quantitatively and qualitatively, why its removal causes a specific performance change, including seemingly paradoxical results?
- **Architectural Difference Analysis:** Does the paper explain performance gaps relative to prior work through mechanistic differences (e.g., using all attention heads versus a selected subset) rather than attributing them solely to better hyperparameters?

---

Principle 5: Stress-Testing Robustness under Prompt Variations, Ambiguous Expressions, and Adversarial Visual Scenarios

**Definition:**  
Training-free methods that rely on textual prompting and attention extraction are inherently vulnerable to prompt wording, linguistic ambiguity, and challenging visual layouts. Reviewers must evaluate whether the method is stress-tested beyond standard benchmark averages through systematic prompt ablations, non-explicit referring expressions that require complex reasoning, multi-object discrimination scenarios, and fine-grained part-level references. This principle prevents acceptance of methods whose apparent strength is merely an artifact of prompt overfitting or cherry-picked evaluation conditions.

**Core Evaluation Criteria:**
- **Prompt Sensitivity Analysis:** Are ablation studies conducted with multiple independent prompt templates for each sub-module (object-focused, background-focused, category choice) to demonstrate insensitivity to exact phrasing?
- **Reasoning and Implicit Query Handling:** Does the method handle queries where the target object is not explicitly named but must be inferred through temporal or commonsense reasoning, and are these cases evaluated quantitatively?
- **Adversarial and Fine-Grained Scenarios:** Is performance reported on adversarial subsets (e.g., scenes with multiple similar objects) and qualitatively demonstrated for part-level segmentation, ensuring the method does not collapse under ambiguity or fine-grained descriptions?