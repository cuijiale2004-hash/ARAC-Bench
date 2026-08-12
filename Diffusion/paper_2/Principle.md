**Principle 1: Rigor of Training-Free Claims and Disentangled Attribution of Performance Contributions in Plug-and-Play Diffusion Guidance**

**Definition:**  
In plug-and-play frameworks that adapt pre-trained latent diffusion models to ultra-high-definition image restoration, the boundary between training-free core mechanisms and optional trainable auxiliary modules must be articulated with absolute clarity. Reviewers must evaluate whether the primary performance gains are genuinely attributable to the training-free guidance component or to supplementary fine-tuned modules—such as VAE decoder fine-tuning—and whether the authors provide transparent, quantitative ablations that isolate these contributions. This principle is critical because ambiguous "training-free" claims can misrepresent the practical burden of adaptation, and the scientific value of the guidance mechanism itself must be disentangled from task-specific parameter updates. Furthermore, reviewers should assess whether any fine-tuned components are justified as lightweight, task-level auxiliaries that generalize across backbones, rather than hidden training requirements that undermine the plug-and-play premise.

**Core Evaluation Criteria:**
- **Transparency of Training Scope**: Does the paper explicitly delineate which components are strictly training-free and which require fine-tuning, using unambiguous terminology and scope definitions?
- **Disentangled Ablation Evidence**: Are there quantitative ablations demonstrating the isolated contribution of the training-free module versus the optional module, proving that the core mechanism alone achieves meaningful gains?
- **Nature of Optional Modules**: Are fine-tuned components justified as task-level (not model-specific) auxiliaries, and is evidence provided that they can be reused across different diffusion backbones without retraining?
- **Validity of Core Claims**: Does the training-free mechanism provide substantial improvements independently, or does performance collapse when the auxiliary module is removed?

---

**Principle 2: Fundamental Distinction and Mechanistic Enforcement of Restoration-Specific Constraints Versus Generation-Oriented Diffusion Adaptation**

**Definition:**  
When adapting diffusion priors originally designed for high-resolution image generation to the UHD restoration domain, reviewers must rigorously assess whether the method explicitly recognizes and enforces the fundamental differences between generation and restoration tasks. Unlike generation, where perceptual plausibility and global coherence are sufficient, restoration demands strict input-output fidelity, selective injection of reference information without overriding valid input details, and absolute suppression of hallucinated content. The evaluation must verify that proposed guidance mechanisms—such as frequency-domain phase injection or feature-level attention blending—are specifically engineered to preserve the geometry and semantics of the degraded input rather than merely synthesizing visually coherent outputs. This principle ensures that restoration-oriented designs are not conflated with generation-oriented baselines, and that the method’s constraints actively prevent the introduction of unrealistic textures or structural alterations.

**Core Evaluation Criteria:**
- **Explicit Theoretical Distinction**: Does the paper clearly articulate why generation-oriented methods (e.g., MultiDiffusion, DemoFusion, PixelSmith) are fundamentally insufficient for restoration due to their lack of strict input-output fidelity constraints?
- **Input-Output Fidelity Mechanisms**: Are there specific architectural or algorithmic constraints ensuring that the output remains precisely faithful to the degraded input in both structure and semantics?
- **Hallucination Suppression**: Does the method include explicit mechanisms to prevent the introduction of textures, colors, or structures not present in the original degraded image?
- **Baseline Adaptation Clarity**: If generation-oriented diffusion methods are used as baselines, are they evaluated under restoration-appropriate metrics that penalize content deviation and structural inconsistency?

---

**Principle 3: Comprehensive Comparative Evaluation Against Task-Specific, Super-Resolution, and Training-Free Diffusion Adaptation Baselines**

**Definition:**  
For UHD-IR methods leveraging diffusion priors, the experimental evaluation must be sufficiently comprehensive to establish competitiveness across multiple methodological paradigms and degradation scenarios. This requires comparison not only with task-specific UHD restoration networks but also with recent super-resolution frameworks and training-free diffusion adaptation strategies, all evaluated under fair and identical settings. Reviewers must verify that the proposed guidance mechanism is tested across diverse degradation types—such as low-light enhancement, dehazing, deblurring, and deraining—to demonstrate broad applicability rather than narrow task-specific optimization. Furthermore, cross-backbone validation is essential to confirm that the plug-and-play framework generalizes across different latent diffusion architectures, ensuring that reported gains are attributable to the guidance mechanism rather than a particular backbone’s inherent strengths.

**Core Evaluation Criteria:**
- **Coverage of Methodological Categories**: Does the evaluation include comparisons with (1) end-to-end UHD restoration networks, (2) LDM-based super-resolution methods, and (3) recent training-free diffusion adaptation strategies?
- **Fair Experimental Protocol**: Are comparisons performed under identical diffusion backbones, resolutions, and inference settings to isolate the contribution of the proposed guidance mechanism?
- **Multi-Degradation Validation**: Are experiments conducted on multiple degradation scenarios beyond a single task to demonstrate generalization and robustness?
- **Quantitative and Qualitative Rigor**: Do the results include both full-reference metrics (PSNR, SSIM) and perceptual/no-reference metrics (LPIPS, DISTS, MUSIQ), accompanied by visual evidence highlighting failure modes of competing methods?

---

**Principle 4: Quantitative Computational Efficiency Assessment and Practical Deployment Feasibility for Ultra-High-Resolution Inference**

**Definition:**  
Given the inherent computational intensity of iterative diffusion sampling on 4K and 8K images, reviewers must rigorously evaluate whether the proposed method provides a favorable and quantified cost-performance trade-off. This requires detailed reporting of incremental overhead—including FLOPs, peak GPU memory consumption, and inference latency—relative to standard patch-based diffusion inference, rather than vague acknowledgments of computational cost. The principle further demands an honest discussion of practical deployment scenarios, distinguishing between methods suitable for real-time edge deployment and those targeting offline, quality-centric workflows such as film remastering, digital cultural heritage preservation, or large-scale remote sensing analysis. Without such contextualization, claims of practical applicability remain unsubstantiated, and the method’s true utility in resource-constrained or resource-abundant environments cannot be assessed.

**Core Evaluation Criteria:**
- **Incremental Overhead Quantification**: Does the paper report precise computational metrics (FLOPs, VRAM, inference time) for the proposed method versus the base patch-based pipeline, isolating the cost of the guidance mechanism itself?
- **Cost-Performance Trade-off**: Is the performance gain commensurate with the additional computational cost, and how does this trade-off compare to cascaded, multi-pass, or full-resolution attention alternatives?
- **Scalability to Extreme Resolutions**: Does the method maintain manageable resource consumption as resolution increases to 8K, or does it introduce memory-quadratic operations that render it impractical on consumer GPUs?
- **Practical Applicability Discussion**: Are realistic deployment scenarios identified, with clear acknowledgment of whether the method targets offline quality-critical applications versus real-time or edge-side systems?

---

**Principle 5: Architectural Generalization Beyond U-Net Backbones and Hyperparameter Robustness Across Diverse Restoration Tasks**

**Definition:**  
As the field of diffusion modeling rapidly transitions from U-Net-based latent diffusion models to Diffusion Transformers (DiTs), reviewers must assess whether the proposed guidance principles are inherently tied to specific architectures or can be conceptually extended to emerging frameworks. This principle evaluates whether the core mechanisms—such as frequency-domain latent manipulation or global-local feature blending—are formulated in an architecture-agnostic manner, and whether the authors provide a concrete conceptual pathway for adaptation to transformer-based diffusion models. Additionally, because plug-and-play methods are intended for broad adoption without cumbersome per-task tuning, the sensitivity of key hyperparameters must be thoroughly analyzed to ensure stable performance across diverse degradation types and backbone models. A method requiring extensive parameter tuning for each new task fundamentally contradicts the plug-and-play objective and limits its practical utility.

**Core Evaluation Criteria:**
- **Architecture-Agnostic Design**: Are the core guidance principles formulated in a way that extends beyond U-Net self-attention structures to token-based transformer blocks without violating fundamental computational constraints?
- **Conceptual Extension Evidence**: Does the paper provide a concrete conceptual pathway (e.g., modified attention block formulations) for adapting the method to DiT architectures, even if full empirical validation is deferred to future work?
- **Hyperparameter Sensitivity Analysis**: Are systematic sensitivity studies provided for key guidance parameters, demonstrating broad stable regions and predictable performance curves?
- **Cross-Task Default Generalization**: Do the default hyperparameters generalize consistently across multiple degradation types, datasets, and backbone models without requiring task-specific re-tuning?