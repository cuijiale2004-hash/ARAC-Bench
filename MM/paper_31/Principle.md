**Principle 1: Adaptive Attack Resilience and Robustness Validation Against Patch-Distributed and Optimization-Based Adversaries in Vision-Language Safety Defenses**

**Definition:**  
For test-time safety alignment methods in vision-language models, robustness must be evaluated beyond static benchmark attacks to include adaptive adversaries specifically designed to circumvent the proposed defense mechanism. This principle recognizes that inference-time detectors and calibrators present a distinct attack surface; consequently, an attacker may distribute harmful semantics across multiple image patches or directly optimize perturbations against the detection metric. The evaluation must stress-test whether the defense maintains efficacy when the adversary has full knowledge of the detection algorithm, thresholding mechanism, and calibration strategy. This perspective is crucial because safety claims based solely on standard benchmarks are insufficient for high-risk deployment scenarios where attackers actively evolve their strategies.

**Core Evaluation Criteria:**
- **Adaptive Attack Evaluation:** Does the work evaluate against attacks explicitly designed to bypass its detection module, such as patch-distributed perturbations or gradient-based optimization targeting the defense's specific distance metrics?
- **Failure Mode Analysis:** Are there clear empirical or theoretical analyses of conditions under which the defense fails, including semantic dilution across patches and fine-grained texture embedding?
- **Comparative Robustness:** Is there a fair comparison with baseline defenses under identical adaptive threat models, demonstrating relative rather than absolute robustness?
- **Detection Bound Characterization:** Does the work provide bounds or empirical distributions showing the separation between adversarial and benign inputs under attack?

---

**Principle 2: Mechanistic Explainability of Safety-Utility Trade-offs in Input Sanitization and Attention Calibration**

**Definition:**  
Test-time safety frameworks that modify visual inputs or internal attention distributions must provide rigorous mechanistic explanations for why general model utility is preserved or enhanced, rather than merely reporting metric improvements. This principle demands that researchers disentangle the causal effects of input masking and activation calibration from coincidental performance fluctuations, particularly when benign inputs are processed alongside harmful ones. In the multimodal setting, visual information degradation and cross-modal attention drift are primary risks; therefore, the evaluation must examine how the defense distinguishes between malicious and benign semantic regions and how calibration affects token-level generation dynamics. Establishing this causal linkage is essential to ensure that safety gains do not stem from random perturbations or task-specific overfitting.

**Core Evaluation Criteria:**
- **Benign Input Analysis:** Is there empirical evidence (e.g., false positive rates, qualitative visualizations) demonstrating that benign, semantically rich regions are not erroneously sanitized?
- **Attention Dynamic Tracing:** Does the work analyze how safety calibration modifies cross-modal attention patterns across decoding layers, and whether this restores or distorts visual grounding?
- **Causal Ablation Design:** Are ablation studies structured to isolate the utility impact of detection versus calibration modules, ruling out confounding factors?
- **Theoretical Justification:** Is there a theoretical or qualitative model explaining why utility improvements arise (e.g., noise reduction, restored visual signal in deep layers)?

---

**Principle 3: Hyperparameter Stability and Cross-Architecture Generalization of Training-Free Detection Metrics**

**Definition:**  
Inference-time safety methods that rely on external vision-language encoders and manually configured thresholds must demonstrate robust performance across diverse hyperparameter settings and backbone architectures to avoid brittle, overfitted solutions. This principle emphasizes that a reliable training-free defense should not depend on a narrow operating point or a specific encoder's latent biases; rather, it should maintain consistent safety-utility profiles when thresholds, amplification coefficients, or backbone models are varied. Given the prevalence of CLIP-family encoders with known biases and distributional idiosyncrasies, the evaluation must explicitly verify that the core mechanism transfers across representational spaces. This ensures the contribution is a generalizable algorithmic insight rather than an encoder-specific artifact.

**Core Evaluation Criteria:**
- **Hyperparameter Sensitivity Sweep:** Are key parameters (e.g., detection thresholds, attention amplification factors) systematically varied to show performance variance bounds?
- **Multi-Backbone Validation:** Does the evaluation span multiple visual encoders (e.g., ResNet-based CLIP, ViT-based CLIP, SigLIP) to establish backbone independence?
- **Separation Quality Metrics:** Is there quantitative evidence (e.g., distributional divergence between safe/unsafe patches) that the core metric maintains discriminative power across architectures?
- **Threshold Interpretability:** Does the work provide theoretical or empirical guidance on how to set critical thresholds without dataset-specific tuning?

---

**Principle 4: White-Box Internal Access Requirements versus Black-Box Deployment Feasibility in Safety Systems**

**Definition:**  
For test-time safety alignment, the evaluation must rigorously delineate which components require white-box model access (e.g., attention map manipulation) and which remain functional in black-box, proprietary deployment environments. This principle is critical because safety defenses in production are often deployed by model providers with full access, yet partial functionality may be needed for API-level filtering or third-party auditing. The evaluation should not obscure deployability limitations; instead, it must transparently assess the degree to which the method's safety guarantees degrade when internal activations are inaccessible. Furthermore, the scope of applicability should be honestly constrained to specific modalities without overclaiming universal multimodal generalizability.

**Core Evaluation Criteria:**
- **Access Delineation:** Does the work clearly classify each component as white-box or black-box and explain the practical implications for deployment?
- **Black-Box Empirical Validation:** Are the black-box-compatible components tested on proprietary models (e.g., Gemini, GPT-4V) via API-only access?
- **Deployability Limitation Discussion:** Is there an honest analysis of scenarios where white-box requirements prevent deployment (e.g., consumer-facing proprietary APIs)?
- **Scope Boundary Clarity:** Are claims about "multimodal" applicability restricted to the actually evaluated modality, with explicit discussion of transfer challenges to audio/video?

---

**Principle 5: Scalability to Production-Scale Models and Complementarity with Existing Alignment Pipelines**

**Definition:**  
Test-time safety mechanisms must demonstrate that their protective effects scale to large-capacity production models and compose additively with pre-existing safety fine-tuning or reinforcement learning alignment, rather than conflicting with or substituting for them. This principle addresses the real-world constraint that deployed vision-language models increasingly exceed 70B parameters and often already undergo post-hoc safety training; an inference-time defense that only works on small open-source checkpoints or supersedes SFT gains has limited practical longevity. The evaluation must verify consistent safety-utility trade-offs across scales and prove that the method provides incremental value when layered atop aligned base models.

**Core Evaluation Criteria:**
- **Large-Model Validation:** Does the evaluation include high-capacity models (e.g., 30B+ parameters) or state-of-the-art proprietary APIs to validate scalability?
- **Compatibility with SFT/RLHF:** Is there experimental evidence that the method provides additive safety improvements when applied to already fine-tuned models without catastrophic utility collapse?
- **Inference Overhead Scaling:** Does the computational cost analysis remain favorable as model size increases, ensuring practical deployment feasibility?
- **Cross-Model Consistency:** Are safety gains and utility preservation consistent across diverse architectural families (e.g., LLaVA, InternVL, proprietary)?