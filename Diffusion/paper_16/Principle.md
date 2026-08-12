
**Principle 1: Cross-Domain, Cross-Resolution, and Cross-Task Generalization Assessment for Unsupervised Representation Alignment in Diffusion Transformers**

**Definition:**  
For methods claiming to replace external visual encoders with self-supervised alignment to avoid distribution mismatch and computational overhead, the evaluation must extend far beyond the standard ImageNet 256×256 benchmark. Reviewers must assess whether the method maintains efficacy across higher resolutions (e.g., 512×512), diverse domains (e.g., medical imaging, vertical-specific datasets), and different task modalities (e.g., class-conditional, text-to-image). The core risk is that unsupervised alignment may exploit dataset-specific statistical regularities that do not transfer, or that its advantages may diminish when external encoders are fine-tuned for in-distribution data. A thorough evaluation should demonstrate that the claimed benefits—particularly the avoidance of out-of-distribution issues and pre-training costs—are realized in practice rather than being purely theoretical. Without such breadth, claims about generalizability and robustness remain unsubstantiated.

**Core Evaluation Criteria:**
- **Resolution Scalability**: Does the work provide high-resolution experimental results (e.g., 512×512) with comparable training efficiency gains, or does performance degrade significantly?
- **Domain Diversity**: Are experiments conducted on domain-specific datasets where external encoders face distribution mismatch (e.g., medical, satellite, or specialized industrial imagery)?
- **Task Modality Breadth**: Does the evaluation span multiple generation paradigms (class-conditional, text-to-image, potentially video) to validate architecture-agnostic claims?
- **OOD Robustness Evidence**: Is there explicit analysis showing failure modes or performance gaps when moving from in-distribution to out-of-distribution data, rather than assuming inherent robustness?

---

**Principle 2: Disentanglement of Core Alignment Mechanism from Batch-Size Effects and Sampling Overhead Artifacts**

**Definition:**  
When a method introduces multiple noise sampling paths or increased per-sample computation (e.g., dual-path sampling), reviewers must carefully evaluate whether reported improvements stem from the proposed alignment objective or from confounding factors such as increased effective batch size, additional gradient updates, or architectural modifications. The method must be rigorously ablated to isolate the alignment loss from the sampling procedure itself, and compared against baselines with matched computational budgets. Furthermore, if the method builds upon a modified architecture (e.g., decoupled encoder-decoder), the contribution of the alignment mechanism must be separated from the base architectural improvements. Failure to disentangle these factors leads to inflated claims about the alignment method's intrinsic value.

**Core Evaluation Criteria:**
- **Sampling vs. Alignment Isolation**: Are ablation studies provided that compare dual-path sampling alone, condition alignment alone, and their combination against the baseline?
- **Batch-Size Control**: Does the work explicitly compare against baselines with enlarged batch sizes or increased training iterations to rule out the possibility that gains come simply from more gradient information per step?
- **Architectural Fairness**: If architectural modifications are used, are ablations conducted to show that alignment improvements persist when applied to standard architectures without those modifications?
- **Compute-Matched Comparisons**: Are all critical comparisons performed under matched training budgets (wall-clock time, GPU hours, or total samples seen)?

---

**Principle 3: Mechanistic and Theoretical Grounding for How Self-Alignment Objectives Accelerate Diffusion Training**

**Definition:**  
Beyond demonstrating improved FID scores or faster convergence curves, work in this subfield must provide a compelling analytical or empirical account of why self-alignment between noisy views facilitates learning. This includes explaining what semantic properties are preserved through alignment (e.g., low-frequency structure, class-discriminative features), which network layers benefit most, and how alignment interacts with the denoising objective. The work should ideally connect the alignment loss to training dynamics—such as gradient variance, representation collapse prevention, or curriculum-like learning effects across noise levels. Without such mechanistic insight, the method risks being treated as an opaque engineering trick rather than a principled contribution to representation learning in generative models.

**Core Evaluation Criteria:**
- **Representation Quality Analysis**: Does the work analyze the learned representations (e.g., via linear probing, CK-NN accuracy, or feature visualization) to demonstrate that alignment produces more semantically meaningful features?
- **Layer-Specific Impact**: Is there analysis of which layers benefit from alignment, and does this align with theoretical expectations (e.g., early layers capturing low-frequency semantics)?
- **Convergence Mechanism Explanation**: Does the paper explain the mechanism by which alignment accelerates convergence (e.g., reduced variance in gradient estimates, better conditioning of the optimization landscape, faster semantic feature emergence)?
- **Timestep and Noise-Level Analysis**: Are experiments provided showing how alignment behaves across different noise levels and timestep selection strategies, with interpretable trends?

---

**Principle 4: Rigorous Differentiation and Positioning Within the Self-Supervised and Contrastive Diffusion Learning Landscape**

**Definition:**  
The subfield of unsupervised representation enhancement for diffusion transformers has become densely populated with related techniques—including masked image modeling, contrastive flow matching, dispersive losses, and various self-alignment variants. A high-quality submission must precisely articulate its relationship to these neighboring methods, clearly stating the technical distinctions in objective design, sampling strategy, and architectural assumptions. Reviewers should evaluate whether the authors have comprehensively surveyed and experimentally compared against the most relevant contemporaneous work, or whether critical related papers have been omitted. The comparison must go beyond citation and include empirical head-to-head results under matched conditions, as well as conceptual discussion of trade-offs in simplicity, efficiency, and generality.

**Core Evaluation Criteria:**
- **Comprehensive Related Work Coverage**: Does the paper cite and discuss all major contemporaneous self-supervised diffusion methods (e.g., SD-DiT, SRA, MaskDiT, contrastive flow matching) and explain the technical distinctions?
- **Head-to-Head Experimental Comparison**: Are direct empirical comparisons provided against the closest related methods under identical training configurations, rather than relying on reported numbers from different papers?
- **Design Choice Justification**: Are key design choices (e.g., cosine similarity vs. MSE, specific layer alignment, timestep selection) justified in contrast to alternatives used in related work?
- **Generality Claims Validation**: If the method claims to be simpler or more general, is this supported by evidence showing easier integration across architectures and fewer hyperparameters?

---

**Principle 5: Validity and Transparency of Training Efficiency and Inference Acceleration Claims**

**Definition:**  
Claims regarding training speedup, inference acceleration, or computational savings are among the most impactful but also most easily misleading aspects of diffusion model research. Reviewers must scrutinize whether "speedup" refers to reduced training iterations, fewer sampling steps, wall-clock time, or some combination thereof, and whether the comparison baseline is fair. For instance, reducing sampling steps may not translate to proportional wall-clock savings if per-step computation increases; similarly, training convergence in fewer epochs must account for any additional per-epoch overhead (e.g., from multiple forward passes or alignment computations). The work must provide clear memory and throughput analysis, and avoid framing that conflates algorithmic improvements with implementation-level optimizations.

**Core Evaluation Criteria:**
- **Clear Speedup Definition**: Does the paper explicitly define what dimension of speed is being improved (training iterations, wall-clock time, sampling steps, memory footprint) and provide corresponding measurements?
- **Wall-Clock Verification**: Are training and inference speedups verified with actual wall-clock time measurements, not just epoch or step counts?
- **Overhead Accounting**: Does the analysis account for additional computational overhead introduced by the method (e.g., extra forward passes, alignment loss computation, increased memory usage)?
- **Baseline Fairness**: Are speedup claims compared against properly tuned baselines with equivalent engineering optimizations, rather than naive or under-tuned implementations?