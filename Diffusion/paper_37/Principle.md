### Principle 1: Mechanistic Attribution of Generative Efficiency to Semantic Discriminability in Unified Latent Spaces

**Definition:**
When evaluating latent diffusion models that replace traditional VAEs with semantic-rich representations (e.g., self-supervised features), reviewers must assess whether the authors provide a causal mechanistic explanation linking the geometric structure of the latent space to generative training/sampling efficiency. It is insufficient to merely claim that "semantic features are better"; the work must demonstrate *why* properties like inter-class separation or intra-class compactness reduce optimization difficulty or discretization error during flow matching/diffusion. This principle ensures that performance gains are attributed to fundamental representational improvements rather than incidental hyperparameter tuning or architectural capacity increases, distinguishing principled representation engineering from empirical trial-and-error.

**Core Evaluation Criteria:**
-   **Geometric-Dynamic Linkage:** Does the paper explicitly connect latent space geometry (e.g., t-SNE clustering, linear probe accuracy) to diffusion dynamics (e.g., velocity field consistency, sampling trajectory straightness)? Is there evidence (such as toy examples or gradient norm analysis) showing that semantic separation simplifies the learning target?
-   **Disentanglement of Factors:** Are the benefits of the semantic backbone isolated from other confounding variables? For instance, does the comparison control for model capacity, training compute, and decoder architecture to prove that the *latent structure itself* drives efficiency?
-   **Quantitative Semantic Metrics:** Beyond visualization, are there quantitative metrics (e.g., linear probing accuracy, cluster purity scores) used to validate the "semantic discriminability" claim? Do these metrics correlate with generation metrics (FID/IS) across different ablation settings?
-   **Theoretical Grounding:** Is the intuition supported by relevant theory (e.g., manifold hypothesis, optimal transport paths) or rigorous empirical proxies that explain why high-dimensional semantic spaces do not suffer from the curse of dimensionality in this specific context?

---

### Principle 2: Rigorous Preservation Verification of Downstream Utility in Hybrid Generative-Perceptive Representations

**Definition:**
For methods proposing a "unified" feature space that serves both generation and perception/understanding, it is mandatory to quantitatively verify that the generative adaptation process (e.g., adding residual branches, fine-tuning decoders) does not degrade the original semantic capabilities of the foundation model. Reviewers must scrutinize whether the "unified" claim is substantiated by comprehensive downstream benchmarks, not just reconstruction fidelity. This criterion prevents the acceptance of methods that achieve high-quality generation at the cost of destroying the very semantic structure they claim to leverage, ensuring the proposed representation truly bridges the gap between discriminative and generative paradigms without compromising either.

**Core Evaluation Criteria:**
-   **Comprehensive Downstream Probing:** Are standard perception benchmarks (classification, segmentation, depth estimation) evaluated using the *exact* frozen or adapted features used for generation? Is performance compared directly against the original foundation model baseline to confirm no regression?
-   **Structural Integrity Analysis:** Beyond task performance, is the geometric structure of the original feature space preserved? Are analyses like t-SNE, PCA, or nearest-neighbor retrieval provided to show that the relative topology of the semantic manifold remains intact after introducing generative components?
-   **Ablation of Adaptation Components:** Is the impact of each generative adaptation (e.g., residual encoder, distribution alignment) on downstream utility explicitly measured? If a component improves generation but harms perception, is this trade-off clearly acknowledged and justified?
-   **Zero-Shot Transfer Capability:** Does the unified representation support zero-shot or few-shot transfer to tasks beyond those seen during generative training (e.g., editing, retrieval)? This tests the robustness and generality of the preserved semantics.

---

### Principle 3: System-Level Computational Fairness and Tokenizer Overhead Transparency in Latent Diffusion Benchmarking

**Definition:**
Evaluations claiming superior training or inference efficiency in latent diffusion models must account for the *total* system cost, including the often-overlooked tokenizer/autoencoder overhead. Reviewers should reject comparisons that only report diffusion backbone metrics while ignoring the computational footprint of the encoder/decoder, especially when replacing lightweight VAEs with heavier vision foundation models. This principle mandates transparent reporting of parameter counts, FLOPs, latency, and throughput for all components to ensure that claimed efficiency gains are real at the system level and not artifacts of selective metric reporting or unfair baseline configurations.

**Core Evaluation Criteria:**
-   **Full-Stack Metric Reporting:** Does the paper provide a complete breakdown of parameters, GFLOPs, encoding/decoding latency, and throughput for both the tokenizer and the diffusion model? Are these metrics compared side-by-side with baselines under identical hardware and batch-size conditions?
-   **Fair Baseline Alignment:** Are baselines matched for total computational budget or model capacity? If the proposed method uses a larger backbone, are comparisons made against scaled-up VAE baselines or adjusted to reflect equivalent compute?
-   **Inference Pipeline Reality:** Do inference speed claims consider the end-to-end pipeline (encoding + diffusion + decoding)? Is the impact of high-dimensional latents on diffusion step cost accurately reflected, rather than just citing fewer sampling steps?
-   **Reproducibility of Efficiency Claims:** Are missing baseline numbers (e.g., specific epoch checkpoints with CFG) reproduced and reported to prevent straw-man comparisons? Is the code/config for efficiency measurement publicly available or sufficiently detailed?

---

### Principle 4: Architectural Minimality and Design Rationale Justification in Residual-Augmented Semantic Encoders

**Definition:**
When augmenting frozen semantic encoders with trainable modules (e.g., residual branches) to recover lost perceptual details, reviewers must evaluate whether the design is minimal, well-justified, and robust to architectural choices. The principle demands evidence that the added complexity is necessary and sufficient, and that performance is not brittlely dependent on specific hyperparameters or architectures. This guards against over-engineered solutions where simple alternatives might suffice, and ensures that the proposed augmentation genuinely complements rather than interferes with the base semantic representation.

**Core Evaluation Criteria:**
-   **Necessity Ablation:** Is there a clear comparison between the augmented system and the base semantic encoder alone? Does the augmentation provide significant gains in reconstruction/generation that justify its added complexity?
-   **Architecture Sensitivity Analysis:** Have alternative designs for the augmentation module been tested (e.g., different backbones, LoRA vs. full finetuning, varying capacities)? Is the chosen design shown to be optimal or at least robust compared to plausible alternatives?
-   **Interaction Mechanism Clarity:** Is the mechanism for combining semantic and residual features (e.g., concatenation, addition, gating) theoretically or empirically justified? Are potential issues like distribution mismatch addressed (e.g., via normalization or alignment losses)?
-   **Training Stability and Hyperparameter Robustness:** Is the joint training of the augmentation module and decoder stable? Are sensitivity analyses provided for key hyperparameters (e.g., loss weights, alignment strength) to demonstrate that the method does not require fragile tuning?

---

### Principle 5: Multidimensional Scalability and Generalization Validation Beyond Standard Resolution Benchmarks

**Definition:**
For latent diffusion frameworks relying on pretrained foundations, reviewers must assess scalability and generalization beyond the primary experimental resolution (typically 256×256). Since foundation models may have inherent resolution biases or constraints, it is critical to verify that the proposed method extends effectively to higher resolutions, larger datasets, and diverse generative tasks (e.g., video, editing) without architectural overhaul. This principle ensures the work represents a scalable paradigm shift rather than a niche trick optimized for a specific benchmark setting, addressing concerns about real-world applicability and long-term relevance.

**Core Evaluation Criteria:**
-   **High-Resolution Empirical Evidence:** Are experiments conducted at resolutions significantly above the training baseline (e.g., 512×512, 1024×1024)? Is visual quality and metric performance maintained or improved, or does degradation occur?
-   **Architectural Adaptation Requirements:** Does scaling require major architectural changes (e.g., new positional embeddings, retraining from scratch) or can it be achieved via simple finetuning/interpolation? Is the scalability path clearly documented and validated?
-   **Task Generalization:** Is the framework tested on tasks beyond unconditional/class-conditional image generation (e.g., text-to-image, video synthesis, inpainting/editing)? Does the unified representation facilitate these downstream applications without extensive task-specific redesign?
-   **Foundation Model Dependency Analysis:** Is the method's scalability bounded by the underlying foundation model's limitations? Are potential bottlenecks (e.g., patch size, context length) discussed and mitigated?