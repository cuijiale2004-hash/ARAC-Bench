**Principle 1: Theoretical Justification and Distinguishability of Prior Dependence versus Posterior Independence in Sequential Factor Disentanglement**

**Definition:**
In diffusion-based sequential disentanglement, a fundamental design choice is whether static and dynamic factors are modeled as independent or dependent, and where in the probabilistic pipeline each assumption is enforced. This principle evaluates whether the work rigorously distinguishes between the generative prior—where dependence may be necessary to capture realistic correlations (e.g., a specific person influencing plausible motions)—and the inference posterior—where independence is enforced to achieve factor separation. The work must provide formal or mechanistic justification for why this specific factorization is theoretically preferable over fully independent or fully dependent alternatives. Furthermore, it should empirically validate that the prior dependence genuinely improves generation quality (e.g., via FVD or swap coherence) without compromising the disentanglement achieved by the posterior independence, and avoid conflating overall performance gains with the validity of the probabilistic model itself.

**Core Evaluation Criteria:**
- **Clarity of Prior/Posterior Distinction:** Does the work explicitly separate the generative prior from the inference posterior, and rigorously justify the independence or dependence assumptions imposed on each?
- **Theoretical Grounding:** Is there a formal or intuitive mechanistic explanation for why dependent prior modeling improves expressiveness or realism, while independent posterior modeling ensures disentanglement?
- **Empirical Validation of Causal Claims:** Are ablation studies provided that isolate the effect of prior dependence (e.g., comparing independent vs. dependent latent DDIM priors) on both generation quality and disentanglement metrics?
- **Avoidance of Post-Hoc Rationalization:** Does the work establish a causal link between the probabilistic formulation and observed disentanglement, rather than inferring mechanism validity solely from improved downstream metrics?

---

**Principle 2: Inductive Bias Design and Empirical Verification of Single-Loss Diffusion Disentanglement Mechanisms**

**Definition:**
A central claim in diffusion-based disentanglement is that a single standard loss term can replace the complex multi-term objective functions (e.g., reconstruction, KL, contrastive, and mutual information losses) typical of VAE/GAN-based methods. This principle evaluates whether the work precisely defines what "simplicity" means—distinguishing architectural complexity from optimization complexity—and clearly identifies the inductive biases that enable disentanglement without auxiliary losses. These biases may include architectural constraints such as sharing a single static vector across all temporal steps and bottlenecking the dynamic latent dimensionality. The evaluation must verify through controlled ablations that removing these biases degrades disentanglement, and must quantify the practical benefits of a single-loss formulation, such as the elimination of gradient conflicts and the reduction of hyperparameter tuning burden relative to multi-term baselines.

**Core Evaluation Criteria:**
- **Precision of Simplicity Claims:** Are claims about "simplicity" explicitly defined in terms of optimization complexity (number of loss terms, hyperparameter count) rather than architectural minimalism, and do they account for all training stages including pre-trained components?
- **Inductive Bias Identification and Verification:** Does the work explicitly state and empirically validate the inductive biases (e.g., temporally shared static factors, low-dimensional dynamic vectors) that substitute for explicit disentanglement regularizers?
- **Gradient Conflict and Optimization Analysis:** Is there quantitative evidence (e.g., per-batch gradient conflict counts) that the single-loss formulation avoids the optimization interference inherent in multi-objective sequential disentanglement methods?
- **Ablation Completeness:** Are systematic ablations conducted to demonstrate that removing or relaxing key architectural constraints (e.g., non-shared static vectors, enlarged dynamic capacity) directly degrades disentanglement quality?

---

**Principle 3: Modality-Agnostic Generalization and Zero-Shot Cross-Domain Transfer in Real-World Sequential Disentanglement**

**Definition:**
For methods claiming modality-agnostic sequential disentanglement, it is insufficient to demonstrate strong results on a single domain. This principle assesses whether the method genuinely avoids reliance on domain-specific inductive biases—such as spatial-temporal consistency priors in video, spectral structure in audio, or seasonal patterns in time series—and whether it can adapt to diverse modalities with only minor, principled architectural adjustments. Beyond cross-modal evaluation, the principle demands rigorous zero-shot cross-dataset transfer experiments to verify that the learned static/dynamic factorization captures universal representational structure rather than overfitting to dataset-specific statistics. The evaluation must also critically examine the role of pre-trained components (e.g., VQ-VAE tokenizers) in enabling generalization, and isolate whether transfer capability stems from the proposed disentanglement framework or from the pre-trained representation space.

**Core Evaluation Criteria:**
- **Cross-Modal Validation:** Does the method demonstrate effective disentanglement on at least three distinct modalities (e.g., video, audio, time series) with minimal and well-documented backbone changes?
- **Domain-Specific Assumption Audit:** Does the work explicitly identify and avoid reliance on modality-specific priors (e.g., facial landmarks, temporal smoothness constraints, or spectral inductive biases)?
- **Zero-Shot Transfer Rigor:** Are zero-shot cross-dataset swap experiments conducted to verify generalization, and is the contribution of pre-trained encoders (e.g., VQ-VAE) to transfer performance properly isolated via controlled ablations?
- **Dataset Diversity and Real-World Complexity:** Does the evaluation extend beyond toy datasets to challenging, high-resolution real-world benchmarks that exhibit significant intra-domain variability?

---

**Principle 4: Evaluation Protocol Rigor and Metric Adaptation for High-Resolution Temporal Disentanglement Benchmarking**

**Definition:**
Sequential disentanglement requires fundamentally different evaluation paradigms than static image disentanglement, yet the field lacks established protocols for real-world, high-resolution temporal data. This principle evaluates whether the work critically examines the applicability of traditional static disentanglement metrics (e.g., SAP, modularity, beta-VAE metric) to sequential settings, where dynamics are often high-dimensional and continuous rather than categorical. It further assesses whether the proposed evaluation metrics—such as Average Keypoint Distance (AKD) for motion preservation and Average Euclidean Distance (AED) for identity preservation—are theoretically justified for temporal factorization, and whether benchmarking protocols ensure fair comparison through fixed evaluation pairs and equitable hyperparameter search budgets for baselines. Finally, the principle examines whether the work analyzes metric limitations and failure modes, such as judge classifier bias, rather than reporting aggregate scores alone.

**Core Evaluation Criteria:**
- **Metric Appropriateness:** Does the work critically evaluate whether traditional static disentanglement metrics are applicable to sequential data, and rigorously justify the selected metrics for temporal static/dynamic separation?
- **Protocol Novelty and Fairness:** Are evaluation protocols designed specifically for high-resolution real-world sequential data, with controls for fair comparison (e.g., predefined swap pairs, extensive baseline hyperparameter searches, identical hardware)?
- **Qualitative-Quantitative Consistency:** Do quantitative metrics align with qualitative results, and are failure cases analyzed to reveal metric limitations (e.g., classifier bias toward identity when judging dynamics)?
- **Benchmarking Scope:** Does the evaluation comprehensively cover reconstruction fidelity, conditional/unconditional swapping, and generative quality across multiple datasets?

---

**Principle 5: Computational Efficiency Transparency and Quality-Cost Trade-off Analysis in Diffusion-Based Disentanglement**

**Definition:**
Diffusion models are inherently more computationally intensive than VAE or GAN counterparts, making transparent cost reporting essential for assessing practical viability. This principle evaluates whether the work comprehensively reports training and inference costs—including wall-clock time, peak GPU memory, number of function evaluations (NFEs), and the computational burden of all auxiliary components (e.g., pre-trained VQ-VAE encoders, separately trained latent DDIM priors). It further demands that efficiency claims be benchmarked against relevant state-of-the-art baselines under identical hardware conditions, and that the trade-off between improved sample quality/disentanglement and increased computational overhead is explicitly discussed. The evaluation must verify that efficiency gains in one aspect (e.g., reduced hyperparameter tuning) are not offset by hidden costs in another (e.g., lengthy pre-training or high-resolution latent diffusion decoding).

**Core Evaluation Criteria:**
- **Cost Transparency:** Are training time, peak memory, inference latency, and NFEs reported for all components, including pre-trained encoders and separately trained prior models?
- **Baseline Comparability:** Are computational costs compared against state-of-the-art baselines under fair conditions (same hardware, comparable training epochs, equivalent hyperparameter search budgets)?
- **Trade-off Justification:** Does the work explicitly analyze the quality-cost trade-off, and justify why diffusion-based modeling is preferable despite potentially higher per-sample inference costs?
- **Scalability Evidence:** Is there empirical evidence that the method scales to high-resolution data without disproportionate increases in training time or memory footprint?