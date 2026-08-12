**Principle 1: Theoretical Validity and Statistical Guarantees of Detector-Dependent Guidance Watermarking**

**Definition:**  
The method fundamentally relies on pretrained differentiable watermark decoders to compute guidance gradients, meaning its theoretical guarantees are inherently coupled to the decoder's feature-space properties. Reviewers raise critical concerns about whether the detector's output distribution is isotropic and unbiased, as biased or correlated features can invalidate statistical guarantees such as p-values for false-alarm control. The work must rigorously analyze how the decoder's robustness and isotropy bound the guidance method's performance, and whether the watermark signal becomes content-dependent rather than a fixed content-unaware perturbation. Furthermore, if the detector is biased, the paper should demonstrate valid correction mechanisms—such as whitening transformations—that restore statistical soundness without retraining. This principle is essential because, without establishing the detector's statistical properties, claims about security, detectability, and robustness lack theoretical foundation. The evaluation should empirically validate these guarantees across diverse decoders and image distributions, ensuring the guidance mechanism does not introduce unforeseen statistical artifacts.

**Core Evaluation Criteria:**
- **Detector Property Analysis**: Does the work explicitly analyze and mitigate dependence on decoder properties such as feature-space isotropy, bias, and robustness? Are biased detectors corrected via validated whitening or comparable techniques?
- **Statistical Soundness**: Are false-alarm rates and p-values theoretically derived and empirically validated across large-scale non-watermarked image sets? Do reported probabilities hold up to the claimed detection thresholds?
- **Content-Dependency vs. Fixed Perturbations**: Is it rigorously demonstrated whether the embedded watermark is content-dependent or merely a fixed perturbation, and are the security implications of this distinction explored?
- **Inheritance vs. Enhancement**: Does the paper clearly separate performance gains inherited from the underlying detector from those genuinely introduced by the guidance mechanism?

---

**Principle 2: Comprehensive Robustness Verification Against Adversarial and Geometric Attacks Beyond Standard Augmentations**

**Definition:**  
A recurring criticism across reviews is that the robustness evaluation is initially limited to standard image augmentations such as JPEG compression, brightness adjustment, and central cropping, which inadequately tests the method's claimed advantages. The subfield demands rigorous stress-testing against adversarial attacks—including VAE purification, regeneration via re-noising, and averaging attacks—that specifically target the watermarking scheme rather than generic image processing. Moreover, geometric transformations like rotation and Gaussian blur must be evaluated, as seed-based methods often fail on these while guidance-based approaches might inherit decoder robustness. The method's core claim is enhancing robustness without retraining the detector, which can only be substantiated by demonstrating improved performance on attacks never seen during the detector's original training. Gradient aggregation across augmentations using algorithms like PCGrad must be justified and shown to genuinely improve robustness beyond simple averaging. This principle ensures that the method's practical security is validated against realistic, hostile conditions rather than benign distortions.

**Core Evaluation Criteria:**
- **Adversarial Attack Coverage**: Does the evaluation include targeted adversarial attacks such as VAE purification, regeneration, and averaging attacks under gray-box or white-box settings?
- **Geometric and Unknown Transformations**: Are geometric attacks (e.g., rotation, Gaussian blur) and transformations outside the detector's training augmentations systematically evaluated?
- **Gradient Aggregation Justification**: Is the choice of gradient aggregation strategy (e.g., PCGrad) ablated and shown to outperform naive averaging when gradients from different augmentations conflict?
- **Retraining-Free Robustness Claims**: Is there empirical evidence that the guidance mechanism improves robustness against novel attacks without any fine-tuning or retraining of the original decoder?

---

**Principle 3: Computational Cost-Benefit Trade-offs and Practical Deployment Feasibility of Guidance-Based Embedding**

**Definition:**  
Multiple reviewers identify computational burden as a primary limitation, noting that deriving guidance gradients through complete denoising operations at every noise level drastically exceeds the time required for standard image generation. The feasibility of the method in production environments depends on demonstrating effective simplifications—such as activating guidance only during the last diffusion steps or approximating gradients via identity transforms—without catastrophic loss of detectability or image quality. The paper must transparently report the trade-offs between guidance strength, clipping thresholds, and the number of guided steps, showing how each parameter affects the quality-detectability frontier. Importantly, the method claims faster detection compared to seed-based schemes, but this advantage must be weighed against the increased embedding cost to present an honest accounting of total system latency. Ablation studies should independently isolate the contribution of each simplification strategy to observed performance changes. Without rigorous cost-benefit analysis, the method risks being dismissed as theoretically interesting yet practically infeasible for large-scale image production.

**Core Evaluation Criteria:**
- **Quantified Computational Overhead**: Is the embedding cost reported in absolute terms (e.g., additional diffusion steps, wall-clock time) and compared against standard generation and baseline watermarking methods?
- **Ablation of Fast Guidance Strategies**: Are simplifications such as step reduction and gradient approximation ablated both independently and jointly, with their respective impacts on bit accuracy, PSNR, and detectability clearly quantified?
- **Quality-Detectability-Steps Trade-off**: Does the paper map the Pareto frontier between image quality (PSNR, LPIPS, CLIP), watermark detectability, and the number of guidance steps?
- **End-to-End System Cost**: Is the total cost (embedding plus detection) compared fairly against seed-based methods that require reverse diffusion for detection, and are claims of "instantaneous" detection contextualized with embedding overhead?

---

**Principle 4: Fair Baseline Alignment and Consistent Metric Reporting Across In-Generation and Post-Hoc Watermarking Schemes**

**Definition:**  
Reviewers challenge the fairness of claiming the method is training-free when it requires a well-trained watermark detector, unlike seed-based alternatives such as Tree-Rings or Gaussian-Shading that need no external neural network. The evaluation must carefully align comparison metrics—reporting bit accuracy alongside capacity, matching watermark lengths, and using consistent detection assumptions—to prevent misleading conclusions about superiority. There is scrutiny over whether the method's gains stem from the guidance mechanism itself or merely from leveraging a stronger underlying detector than those used by baselines. Additionally, detection procedures differ fundamentally: post-hoc and guidance methods use instantaneous detector inference, while seed-based methods require costly reverse diffusion, and these asymmetries must be explicitly acknowledged in utility comparisons. The paper should avoid conflating detector performance with guidance contribution by isolating each factor through controlled ablations. Fair baseline alignment also extends to reporting end-to-end costs and ensuring that all methods are evaluated on identical attack suites and image distributions.

**Core Evaluation Criteria:**
- **Prerequisite Fairness**: Are comparisons honest about prerequisites? Does the paper acknowledge that "retraining-free" refers only to the diffusion model while still requiring a pretrained detector, unlike truly training-free seed-based methods?
- **Metric Consistency**: Are metrics aligned with community standards (e.g., bit accuracy, capacity, p-values) and reported transparently so that methods with different watermark lengths can be fairly compared?
- **Detector vs. Guidance Attribution**: Are controlled experiments included to isolate the contribution of the guidance mechanism from the underlying detector's inherent performance?
- **Symmetric Evaluation Conditions**: Are all methods evaluated under identical attack suites, image distributions, and detection assumptions, with detection-time costs explicitly factored into overall utility?

---

**Principle 5: Clarity of Motivation and System-Level Presentation for Guidance-Driven In-Generation Watermarking Methods**

**Definition:**  
Several reviewers express confusion about the paper's motivation, noting that the introduction fails to explicitly justify why guidance-based in-generation watermarking is necessary given existing post-hoc, VAE-based, and seed-based alternatives. The presentation must provide clear system-level diagrams that illustrate the entire watermarking pipeline—from latent initialization through guidance application to detection—and visually contrast these with baseline paradigms. Formula-heavy descriptions should be complemented by intuitive explanations of how detection and decoding are performed, including how raw detector outputs are converted into binary messages or detection scores. The writing must explicitly distinguish whether the method embeds semantic, content-dependent watermarks or merely adds fixed perturbations, as this distinction carries profound security implications against removal attacks. Motivation should be anchored in concrete limitations of prior art, such as the spectral concentration of VAE-based signals or the geometric fragility of seed-based patterns. Without this clarity, reviewers struggle to assess the incremental contribution and practical integration of the proposed method into existing watermarking ecosystems.

**Core Evaluation Criteria:**
- **Explicit Motivation**: Is the motivation clearly anchored in specific, demonstrated limitations of prior art (e.g., spectral bias in VAE-based methods, geometric fragility of seed-based methods, detection cost asymmetries)?
- **System-Level Visualization**: Are comprehensive pipeline diagrams provided that show embedding, detection, and explicit comparisons to post-hoc and seed-based workflows?
- **Detection and Decoding Transparency**: Are the detection and decoding procedures clearly explained, including how the detector's raw logits are converted into binary messages or detection scores?
- **Conceptual Distinction**: Does the text clearly differentiate the proposed guidance approach from VAE-based fine-tuning and seed-based latent manipulation, explaining the unique positioning of in-generation guidance?