**Principle 1: Theoretical Justification and Avoidance of Trivial Collapse in Self-Consistency-Based Guidance Weight Learning**

**Definition:**  
Research proposing learned guidance weights must provide rigorous theoretical and empirical justification for why the optimization objective yields meaningful, non-zero guidance schedules. Since standard regression losses—such as guided score matching or L2 denoising objectives applied to forward-process samples—can theoretically converge to zero guidance when the base diffusion model is sufficiently well-trained, the work must clearly explain why its proposed proxy (e.g., self-consistency via Maximum Mean Discrepancy) avoids this collapse. The justification should articulate the relationship between the tractable proxy objective and the ideal but intractable target (e.g., marginal consistency), distinguish between forward-process and reverse-process sampling regimes, and explain the variance properties of the resulting gradient estimator.

**Core Evaluation Criteria:**
- **Non-Collapsing Property**: Does the work prove or convincingly argue why the proposed objective does not collapse to trivial zero-guidance weights, and are alternative formulations (e.g., guided score matching, backward KL, trajectory-based regression) systematically shown to fail or underperform?
- **Connection to Ideal Target**: Is the relationship between the proxy objective (self-consistency) and the theoretically grounded target (marginal consistency) clearly explained, including why the stronger condition is a necessary and sufficient substitute?
- **Variance and Stability**: Is the gradient variance of the objective analyzed, and is the proposed method empirically demonstrated to be more stable than direct optimization of the ideal target?
- **Theoretical Rigor**: Are the conditions under which minimizing the loss improves conditional sampling formalized, even if complete theoretical bounds remain open problems?

---

**Principle 2: Fair Comparison with Existing Pre-trained Models and Strong Heuristic Guidance Schedules**

**Definition:**  
Claims of improved guidance must be validated against a comprehensive and competitive baseline suite. This includes not only trivial baselines (unguided generation, constant guidance) but also strong heuristic scheduling methods (e.g., limited interval guidance, clamp-linear schedules) that have been extensively tuned. Crucially, reviewers expect experiments to be conducted on publicly available, community-standard pre-trained diffusion models (e.g., EDM, EDM-2, Stable Diffusion, MicroDiffusion) rather than exclusively on privately trained models, to ensure that observed gains generalize to models practitioners actually use. The evaluation must also assess the trade-off between distributional alignment metrics (e.g., FID) and perceptual alignment metrics (e.g., CLIP score, Inception Score).

**Core Evaluation Criteria:**
- **Pre-trained Model Evaluation**: Does the work demonstrate improvements on publicly available pre-trained models, or at least explicitly commit to and justify the use of privately trained models?
- **Strong Heuristic Baselines**: Are comparisons made against state-of-the-art hand-crafted guidance schedulers with extensive hyperparameter sweeps, rather than only constant or naive interval baselines?
- **Metric Trade-off Analysis**: Does the method improve distributional metrics (FID) without catastrophic degradation of perceptual or text-alignment metrics (CLIP score), and is this trade-off explicitly analyzed?
- **Fairness of Comparison**: Are baselines and the proposed method evaluated under identical conditions (sampling steps, model checkpoints, inference budget)?

---

**Principle 3: Computational Overhead and Practical Feasibility of Auxiliary Guidance Network Deployment**

**Definition:**  
Introducing an auxiliary neural network to predict guidance weights adds both training and inference overhead. Reviewers evaluate whether the computational and engineering costs are justified by the performance margin, and whether more efficient alternatives were considered. The work must explicitly quantify the guidance network's parameter count, training time, memory footprint, and inference latency relative to the base diffusion model. Furthermore, the work should discuss whether the learned guidance can be distilled into the base model or implemented via parameter-efficient fine-tuning (PEFT) to avoid maintaining a separate network at deployment time.

**Core Evaluation Criteria:**
- **Cost Quantification**: Are the training and inference costs of the guidance network explicitly reported as absolute values and as fractions of the base model's cost?
- **Overhead Justification**: Is the performance improvement sufficiently large to justify the additional training step and per-sample MLP evaluation?
- **Efficiency Alternatives**: Does the work discuss or experiment with more efficient deployment strategies (e.g., LoRA adaptation of the base denoiser, distillation, or lookup-table caching)?
- **Scalability**: Is the guidance network demonstrated to be lightweight and scalable across different model sizes (e.g., from U-Net to billion-parameter transformers)?

---

**Principle 4: Semantic Interpretability and Per-Conditioning Variability of Learned Time-Dependent Guidance Weights**

**Definition:**  
For guidance weights that are functions of the conditioning signal, the work must demonstrate that the learned schedules vary in a semantically meaningful way across different conditions (classes, text prompts) rather than merely learning a global, conditioning-agnostic schedule with minor perturbations. The analysis should reveal whether specific semantic categories systematically demand stronger or weaker guidance, and whether the network effectively leverages the conditioning embeddings (e.g., CLIP, T5) to modulate its output. Practitioners should be able to inspect and interpret how guidance evolves over time for any given condition.

**Core Evaluation Criteria:**
- **Per-Conditioning Variation**: Is there quantitative evidence (e.g., top/median/bottom-K analysis) showing that guidance weights differ substantially across conditions, and do ablations confirm that conditioning-aware weights outperform conditioning-agnostic ones?
- **Semantic Correlation**: Do the learned schedules correlate with semantic properties of the conditioning (e.g., complexity, rarity, prompt length), or is the variation attributed to random training artifacts?
- **Visual Inspection**: Are clear visualizations provided to show how guidance weights evolve over timesteps for individual conditions?
- **Embedding Utilization**: Are ablations performed over different conditioning inputs (e.g., CLIP-only vs. T5-only vs. both) to verify that the network uses the conditioning signal effectively?

---

**Principle 5: Hyperparameter Robustness and Stability of MMD-Based Guidance Weight Learning Objectives**

**Definition:**  
The proposed method typically introduces several new hyperparameters governing the training of the guidance network, such as the time-step gap δ, minimum time S_min, MMD kernel parameters (β, λ), and the number of particles m. Reviewers expect thorough ablation studies demonstrating that the method is robust across a reasonable range of these hyperparameters and that it does not require brittle, dataset-specific tuning to achieve good performance. The method should be shown to be more stable than simpler alternative objectives (e.g., pure L2 losses), which may be highly sensitive to these choices.

**Core Evaluation Criteria:**
- **Comprehensive Ablations**: Are systematic ablations provided for all major hyperparameters (δ, S_min, β, λ, m, network depth/width)?
- **Stability Range**: Does performance remain reasonable when hyperparameters are varied, or does it collapse (e.g., to zero guidance or extremely high FID) in certain regimes?
- **Comparative Robustness**: Is the proposed objective empirically shown to be less sensitive to hyperparameters than alternative formulations (e.g., L2 self-consistency or trajectory-based losses)?
- **Design Justification**: Are the final hyperparameter choices supported by empirical evidence (e.g., grid search with metric reporting) rather than arbitrary selection?