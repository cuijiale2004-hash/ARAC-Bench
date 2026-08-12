**Principle 1: Formal Theoretical Linkage Between Posterior Covariance Eigen-Spectra and Distribution Shift in Diffusion Models**

**Definition:**
The work must establish rigorous, mathematically sound connections between the proposed detection statistic—specifically the eigenvalue spectrum of the posterior covariance induced by a diffusion denoiser—and the fundamental quantity of interest, namely distribution mismatch or KL divergence. It is insufficient to merely observe empirical "inflation" of eigenvalues on OOD inputs; the paper must formally demonstrate why this spectral signature is a consistent and reliable marker of distribution shift, explicitly linking posterior covariance to excess denoising error and clarifying why scalar summaries such as MSE or total trace collapse into uninformative isotropic noise while top-K eigenvalues preserve discriminative anisotropic structure. The theoretical claims must be reconciled with empirical observations, particularly when theory predicts behavior in expectation but experiments reveal sample-wise trends. Without this formal grounding, the method risks being perceived as a heuristic aggregation of Jacobians rather than a principled uncertainty quantifier.

**Core Evaluation Criteria:**
- **Formal Rigor:** Are the derivations connecting KL divergence, denoising MSE, and posterior covariance trace formally correct and clearly stated? Are assumptions such as MMSE optimality and positive semi-definiteness justified or appropriately qualified for learned denoisers?
- **Theoretical-Empirical Consistency:** Does the work explain why MSE fails catastrophically (e.g., AUROC < 0.5) in some settings despite Proposition 1 predicting excess denoising error? Is the distinction between expectation-level guarantees and per-sample ordering clearly articulated?
- **Spectral Justification:** Is there a principled argument for why top-K eigenvalues outperform the full trace or scalar score norms, grounded in the geometry of high-noise regimes where isotropic noise dominates?
- **Clarity of Assumptions:** Does the work address whether the learned denoiser's Jacobian is symmetric positive semi-definite, and if not, how the spectral estimation remains valid under approximation?

---

**Principle 2: Comprehensive Multi-Scale Benchmark Coverage from Low-Resolution to Large-Scale High-Resolution OOD Detection**

**Definition:**
Because diffusion models are celebrated for their ability to model complex, high-resolution data distributions, an OOD detection method built upon them must be validated across a comprehensive spectrum of dataset scales, resolutions, and semantic distances. Evaluation restricted solely to small, low-resolution benchmarks such as 32×32 CIFAR variants is insufficient to establish practical utility or to demonstrate that the spectral signal generalizes beyond toy settings. The work must demonstrate effectiveness on large-scale datasets such as ImageNet-1K, standard OOD detection suites including LSUN, iSUN, Textures, and Places365, and challenging near-OOD settings where in-distribution and out-of-distribution samples share low-level statistics. Furthermore, high-resolution experiments (e.g., 256×256) are essential to verify that the covariance inflation signal remains stable and discriminative as dimensionality and visual complexity increase, mirroring the real-world deployment scenarios of diffusion models.

**Core Evaluation Criteria:**
- **Scale Diversity:** Does the evaluation include large-scale datasets (e.g., ImageNet-1K) and high-resolution images (e.g., 256×256 faces or microscopy data), or is it limited to small canonical benchmarks?
- **Benchmark Completeness:** Are standard OOD datasets (SVHN, LSUN, iSUN, Textures, Places365) covered, particularly when CIFAR-10/100 serve as in-distribution data?
- **Near-OOD Robustness:** Does the method maintain discriminative power in semantically similar near-OOD scenarios (e.g., CIFAR-10 vs. CIFAR-100, CIFAR-100 vs. TinyImageNet) where likelihood and score-norm baselines often fail?
- **Baseline Contemporary Relevance:** Are comparisons made against state-of-the-art diffusion-based baselines (e.g., DiffPath, DDPM-OOD, LMD) and recent generative foundation model approaches?

---

**Principle 3: Computational Tractability and Scalability of Jacobian-Free Spectral Estimation Algorithms**

**Definition:**
Methods that probe the geometric structure of diffusion denoisers through Jacobian or Hessian information face an inherent scalability challenge in high-dimensional image spaces, where explicit matrix operations can be prohibitively expensive. A proposed method is not practically viable unless it avoids prohibitive computational costs such as dense Jacobian materialization or O(n³) matrix decompositions. The work must demonstrate that spectral estimation is achieved through efficient, scalable approximations—such as Jacobian-free power iteration combined with thin QR orthogonalization—using only forward evaluations of the denoiser. The evaluation must provide concrete empirical evidence of computational cost, including wall-clock runtime and number of function evaluations (NFE), benchmarked against alternative detectors to prove the method is tractable at scale and does not introduce an unacceptable inference-time overhead relative to simpler baselines.

**Core Evaluation Criteria:**
- **Algorithmic Efficiency:** Does the method avoid forming the full Jacobian matrix and avoid O(n³) dense QR decompositions? Is the complexity explicitly characterized in terms of forward passes, timesteps, and iteration count?
- **Empirical Cost Analysis:** Are runtime (seconds per image) and NFE comparisons reported against key baselines (e.g., DiffPath, LMD, DDPM-OOD)?
- **Scalability to High Resolution:** Does the method remain feasible when applied to 256×256 or higher-resolution inputs, or does the cost become prohibitive as spatial dimensions grow?
- **Numerical Stability:** Is the sensitivity of spectral estimates to the finite-difference step size analyzed, and is the method shown to be stable across a reasonable range of this parameter?

---

**Principle 4: Hyperparameter Robustness and OOD-Sample-Independent Deployment Validity**

**Definition:**
A critical litmus test for real-world OOD detection is whether the method can be deployed effectively without any prior knowledge of the OOD distribution. If hyperparameters such as the number of timesteps, eigenvalues, noise realizations, or aggregation rules must be tuned on an ID-OOD validation split, the method's practical applicability is severely compromised because the nature, severity, and semantic content of the distribution shift are inherently unknown at deployment time. The work must therefore demonstrate that a single, fixed universal hyperparameter configuration achieves strong performance across diverse datasets without any OOD-dependent validation. Furthermore, the sensitivity of performance to individual hyperparameters must be thoroughly ablated to prove the method is robust rather than fragile, and the paper must be transparent about whether its main results derive from tuned or default settings to avoid misleading the reader about real-world deployability.

**Core Evaluation Criteria:**
- **Fixed-Configuration Generalization:** Are results reported using a single default hyperparameter setting across all experiments, and does this configuration still outperform or match tuned baselines?
- **OOD-Independent Validation:** Is it explicitly clarified whether hyperparameters were selected using OOD validation data, and are the main results based on OOD-free settings?
- **Sensitivity Analysis:** Are ablation studies provided for key hyperparameters (timesteps T, eigenvalue count K, repetitions I, finite-difference step c), showing performance variation is minimal (e.g., <3% AUROC variation across timesteps)?
- **Transparency in Reporting:** Does the paper clearly distinguish between tuned results and default-configuration results, avoiding misleading presentation of numbers obtained with OOD-dependent tuning?

---

**Principle 5: Methodological Distinction and Diagnostic Interpretability Within Diffusion-Based OOD Paradigms**

**Definition:**
The literature on diffusion-model OOD detection is crowded with conceptually related approaches—including reconstruction-based scores, trajectory-path metrics, likelihood-geometry analyses, and non-generative reconstruction methods such as PCA and kernel PCA. A new contribution must clearly articulate its unique conceptual position: it must explain why posterior covariance eigen-spectra capture a fundamentally different and more robust signal than reconstruction fidelity (DDPM-OOD), diffusion-path geometry (DiffPath), or likelihood-volume effects (Kamkari et al.). Beyond methodological positioning, the work must provide deep diagnostic analyses that explain when and why the method succeeds or fails along the diffusion trajectory, why competing baselines collapse under certain model-dataset combinations, and how the spectral behavior of InD versus OOD inputs evolves across noise levels. This diagnostic depth transforms the paper from a purely empirical report into an interpretable scientific contribution.

**Core Evaluation Criteria:**
- **Conceptual Differentiation:** Does the work clearly distinguish itself from reconstruction-based, trajectory-based, and likelihood-geometry baselines, explaining why the proposed spectral signal is more robust or theoretically grounded?
- **Baseline Failure Diagnosis:** Are analyses provided for why strong baselines (e.g., DiffPath) underperform in the paper's experimental setting, rather than simply reporting lower AUROC numbers?
- **Timestep-Dependent Mechanistic Analysis:** Does the work include dense analyses of AUROC and eigenvalue trajectories across diffusion timesteps, revealing where OOD separation emerges and why certain noise levels are more informative?
- **Visual and Qualitative Evidence:** Are uncertainty maps, eigenvalue distributions, and denoised outputs visualized to provide intuitive, interpretable evidence of covariance inflation on OOD inputs?