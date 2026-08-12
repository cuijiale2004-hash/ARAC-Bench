
**Principle 1: Theoretical Tightness and Empirical Verifiability of Dimensionality-Independent Estimation Bounds in Multi-Subspace Latent Diffusion**

**Definition:**  
Reviewers must assess whether the derived estimation error bounds fundamentally escape the curse of ambient dimensionality through verifiable structural assumptions—specifically, the union of low-rank subspaces and mixture-of-Gaussian (MoG) latent distributions—rather than introducing hidden dependencies on the raw data dimension $D$. The bounds should explicitly reveal how sample complexity scales with intrinsic parameters such as the number of subspaces $K$, latent dimensions $d_k$, and number of modes $n_k$. Critically, the theoretical claims must be accompanied by empirical phenomena that reflect these scaling laws, ensuring the theory provides a predictive, mechanistic explanation for the observed sample efficiency of diffusion models rather than serving as a post-hoc rationalization.

**Core Evaluation Criteria:**
- **Explicit Dependence on Intrinsic Dimensions**: Does the bound rigorously replace the ambient dimension $D$ with terms involving $\sum n_k$ and $\sum n_k d_k$, and is this dependence tight up to polynomial or logarithmic factors?
- **Empirical Consistency with Theory**: Does the paper provide empirical evidence (e.g., loss curves scaling with dataset size, controlled parameter-count comparisons) consistent with the $1/\sqrt{n}$ regime predicted by the theory?
- **Fairness of Theoretical Comparison**: Are comparisons to prior minimax rates (e.g., $n^{-1/D}$ or $n^{-2/d}$) and competing frameworks (e.g., MoLRG) technically accurate, with clear articulation of what structural assumptions enable the improvement?
- **Bounded-Support Justification**: Are the necessary regularity assumptions (e.g., bounded data support, Lipschitz constants) justified as natural for image manifolds rather than being artificially restrictive?

---

**Principle 2: Structural Consistency Between Theoretical Latent Modeling and Empirical Neural Architecture Design**

**Definition:**  
A central evaluation criterion is the degree of alignment between the theoretical abstraction—such as linear orthonormal encoders $A_k$, exact MoG score functions, and soft responsibility weights—and the actual experimental implementation, which may employ nonlinear VAEs, deep score networks, and hard clustering. Reviewers scrutinize whether deviations are explicitly acknowledged and theoretically justified, and whether the theoretical insights legitimately guide architectural innovations like expert-specific VAEs or Mixture-of-Experts (MoE) score networks. The bridge between theory and practice must not be ad hoc; the empirical design should be presented as a principled relaxation or practical instantiation of the theoretical model, with ablations isolating the contribution of each structural component.

**Core Evaluation Criteria:**
- **Explicit Theoretical-Empirical Mapping**: Does the paper clearly state which theoretical components are implemented faithfully (e.g., latent MoG score) and which are relaxed (e.g., nonlinear VAE replacing linear $A_k$), with justification for why the relaxation preserves the theoretical insight?
- **Necessity of Structure via Ablations**: Do ablation studies demonstrate that the theoretically motivated structure—such as expert-specific VAEs versus a single unified VAE, or MoG parameterization versus Gaussian—is necessary to achieve reported performance?
- **Naturalness of MoE Induction**: Is the MoE structure derived naturally from the latent distribution’s posterior responsibilities, or is it artificially imposed? Does the closed-form score under MoLR-MoG modeling rigorously justify the network architecture?
- **Controlled Baseline Comparisons**: Does the paper compare against baselines that share identical structural backbones (e.g., the same VAE encoder/decoder) to ensure that performance differences are attributable to the latent score parameterization rather than confounding architectural choices?

---

**Principle 3: Robustness of Optimization Guarantees Under Realistic Latent Distribution Overlap**

**Definition:**  
The optimization analysis must extend beyond idealized regimes—such as perfectly separated Gaussian modes or symmetric two-component mixtures—to address realistic scenarios where latent modes overlap significantly. Reviewers evaluate whether the authors quantify how overlap degrades the strong convexity of the score-matching objective and whether convergence guarantees degrade gracefully rather than collapsing entirely. This requires explicit definitions of overlap metrics (e.g., pairwise overlap factor $\xi_{i,j}$, maximum expected overlap $\epsilon_{\text{overlap}}$) and rigorous perturbation analyses (e.g., Weyl’s inequality, Hessian perturbation bounds) that establish positive definiteness of the Hessian under non-negligible mode confusion. The principle demands that theoretical convenience does not override empirical realism, and that the analysis identifies explicit thresholds where guarantees fail.

**Core Evaluation Criteria:**
- **Quantitative Overlap Analysis**: Does the analysis provide explicit, computable bounds on the minimum eigenvalue of the Hessian as a function of the overlap factor $\epsilon_{\text{overlap}}$?
- **Empirical Validation of Overlap Regime**: Are the overlap assumptions validated on realistic latent distributions (e.g., reporting measured $\xi_{i,j} < 0.001$ for ImageNet VAE latents) to confirm that the theoretical regime is actually encountered in practice?
- **Graceful Degradation and Failure Thresholds**: Does the paper derive a condition number $\kappa_{\text{eff}}$ that degrades smoothly with overlap, and does it explicitly state the threshold $\epsilon_{\text{overlap}} < \alpha/C'$ beyond which local strong convexity is lost?
- **Extension Beyond Hard Assignment**: Does the analysis avoid relying exclusively on hard assignment approximations ($r_k \approx 1$) by bounding the perturbation introduced by soft, overlapping responsibilities?

---

**Principle 4: Cross-Scale Empirical Validation of Parameter Efficiency and Generative Fidelity**

**Definition:**  
Claims of parameter efficiency (e.g., $10\times$ fewer parameters) and performance comparable to standard deep architectures must be substantiated across multiple dataset scales and complexity regimes, from simple benchmarks (MNIST, CIFAR-10) to high-resolution generation (ImageNet 256). Reviewers assess whether the evaluation protocol includes quantitative metrics beyond qualitative visual inspection—such as CLIP scores, FID, or LPIPS—and whether the proposed lightweight score network maintains fidelity as latent manifold complexity increases. The scalability of the approach, particularly how the number of experts $K$ and modes $n_k$ must grow with dataset diversity, is critical for judging whether the efficiency gains are architectural artifacts or principled properties of the modeling framework.

**Core Evaluation Criteria:**
- **Progressive Scale Validation**: Are experiments conducted on progressively complex datasets (low-resolution $\to$ high-resolution, single-class $\to$ multi-class) to demonstrate that the approach scales beyond toy domains?
- **Quantitative Metrics Beyond Visuals**: Does the evaluation include quantitative generative metrics (e.g., CLIP score for text-to-image alignment, FID, or perceptual similarity) alongside qualitative sample grids?
- **Honest Cost Accounting**: Are the full pipeline costs—including $K$ VAEs or LoRA adapters, routing overhead, and expert storage—reported and compared honestly against unified baselines, rather than counting only the score network parameters?
- **Scaling Laws for Experts and Modes**: Does the paper analyze how performance and required model capacity scale with $K$ and $n_k$, and do the efficiency gains persist as these structural parameters increase?

---

**Principle 5: Theoretical Necessity and Non-Incrementality of Multi-Modal Latent Extensions Beyond Uni-Modal Gaussian Priors**

**Definition:**  
Reviewers must evaluate whether the shift from prior uni-modal Gaussian latent assumptions (e.g., MoLRG) to multi-modal mixture-of-Gaussian latents (MoLR-MoG) is theoretically necessary and empirically consequential, rather than an incremental capacity increase. This involves assessing whether the prior Gaussian assumption fundamentally limits representational capacity—manifesting as blurry samples, inability to capture intra-class multi-modality, or linear score functions—and whether the MoG extension naturally induces meaningful architectural innovations (e.g., nonlinear score functions, input-dependent gating) rather than merely adding parameters. The positioning relative to prior work must demonstrate that the multi-modal extension unlocks new theoretical capabilities (e.g., escaping dimensionality via mode-specific structure) that were inaccessible under Gaussian latents.

**Core Evaluation Criteria:**
- **Demonstrated Failure of Gaussian Baselines**: Does the paper rigorously attribute the limitations of Gaussian latent baselines (e.g., blurry generation, higher loss floors) to their uni-modality and linear score structure, rather than to insufficient tuning or capacity?
- **Natural Emergence of Architecture**: Is the MoE structure shown to emerge organically from the posterior responsibilities of the latent MoG, with gating weights $g_k(x)$ and expert outputs $E_k(x)$ corresponding to derivable statistical quantities?
- **Fair Attribution of Gains**: Does the comparison with prior subspace or MoLRG work control for confounding factors (e.g., identical VAE quality, training budget, inference protocol) to ensure improvements stem from multi-modal latent modeling?
- **Unique Theoretical Enablers**: Are the estimation and optimization guarantees (e.g., local strong convexity, $1/\sqrt{n}$ bounds) uniquely enabled by the MoG structure, or would they apply generically to any latent mixture model without additional insight?