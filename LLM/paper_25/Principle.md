**Principle 1: Theoretical Soundness of Score-Generator Unification in Self-Adversarial Flow Distillation**

**Definition:**  
In few-step generative modeling, unifying multiple roles—such as score estimator, generator, and fake-score network—into a single model demands rigorous theoretical justification. Reviewers scrutinize whether the hybrid training objective preserves the mathematical properties required for each role, particularly whether the model remains a valid score function when simultaneously optimized for flow matching, consistency-style, and adversarial distillation losses. The validity of gradient derivations, such as reverse KL via score-based gradients, hinges on the model's capacity to estimate instantaneous velocity fields and their relationship to log-probability gradients. Any ambiguity in whether the model conditions on single versus dual timesteps, or whether it can truly represent the score of the data distribution, becomes a central point of contention. Authors must demonstrate that the unified architecture does not break the theoretical foundations of the underlying probability flow ODE. This principle ensures that empirical performance gains are not achieved through theoretically inconsistent manipulations that could mislead future work.

**Core Evaluation Criteria:**
- **Validity of Dual-Role Claims**: Does the paper rigorously prove or convincingly argue that the unified model remains a valid score estimator despite being trained with mixed objectives (e.g., flow matching, rectification, and adversarial losses)?
- **Correctness of Gradient Derivations**: Are key theoretical derivations (e.g., converting distribution matching to velocity matching, computing log-probability gradients) mathematically sound and clearly tied to the model's actual architecture and conditioning inputs?
- **Architectural Fidelity**: Does the implementation faithfully support the theoretical claims (e.g., dual timestep conditioning for any-step transitions) rather than silently deviating from the stated assumptions?
- **Resolution of Theoretical Disputes**: In cases of reviewer disagreement, can the authors resolve concerns through clarifications, minimal examples, or citations, or do fundamental flaws remain unresolved?

---

**Principle 2: Fairness and Contamination Control in Few-Step Generative Benchmarking**

**Definition:**  
Few-step methods are highly sensitive to training data composition, and even minor overlaps with evaluation prompts can inflate benchmark scores artificially. Reviewers expect strict controls against data contamination, such as filtering benchmark-like prompts from training corpora and reporting results on purely self-distilled or held-out datasets. Comparisons against baselines must be conducted under identical experimental conditions—same base model, parameter budget, training data, and inference configuration—to ensure that reported gains stem from algorithmic innovation rather than dataset or compute advantages. The evaluation should extend beyond a single headline metric to include multiple benchmarks that assess different failure modes, such as compositional binding, counting, and spatial reasoning. Authors must also disclose whether results rely on proprietary data, LLM-rewritten prompts, or architectural modifications that are unavailable to baseline methods. This principle guards against benchmark overfitting and ensures that claimed improvements generalize to real-world deployment settings.

**Core Evaluation Criteria:**
- **Data Leakage Prevention**: Does the work explicitly verify that training data does not contain benchmark prompts or near-duplicates, and are ablations provided on clean, self-distilled datasets to validate score stability?
- **Controlled Baseline Comparisons**: Are baseline methods evaluated under the same model scale, training budget, data regime, and hardware constraints (e.g., full-parameter vs. LoRA, identical batch sizes)?
- **Multi-Benchmark Validation**: Does the evaluation span multiple complementary benchmarks (e.g., GenEval, DPG-Bench, WISE) to prevent gaming of a single metric, and are trade-offs between metrics discussed?
- **Transparency of External Advantages**: Are auxiliary advantages such as proprietary training data, prompt rewriting, or architecture modifications clearly disclosed and their contributions isolated?

---

**Principle 3: Memory Scalability and Auxiliary-Free Training on Billion-Parameter Models**

**Definition:**  
A primary motivation for novel few-step frameworks is the practical barrier of scaling distillation to ultra-large models (e.g., 20B parameters), where maintaining separate generator, teacher, and fake-score networks often causes out-of-memory failures. Reviewers evaluate whether the proposed method genuinely eliminates auxiliary trainable or frozen models, thereby reducing GPU memory overhead and simplifying the training pipeline. The distinction between theoretical algorithmic simplicity and practical engineering feasibility is critical: methods that still require multiple LoRA adapters or fixed teacher backbones under the hood must not overstate their memory benefits. Full-parameter training at scale serves as the ultimate stress test, demonstrating that the method avoids the stability issues and memory bottlenecks that plague adversarial or multi-model distillation. Authors should quantify memory reduction relative to strong baselines and demonstrate convergence without architectural crutches like frozen stabilizers or specialized normalization layers.

**Core Evaluation Criteria:**
- **Elimination of Auxiliary Networks**: Does the method truly unify all components (generator, real score, fake score) into one trainable model, or does it rely on hidden frozen teachers, separate discriminators, or auxiliary LoRA modules that increase memory?
- **Scalability to Full-Parameter Training**: Are results demonstrated on billion-parameter models with all parameters trainable, and do baseline methods fail or require compromises (e.g., smaller batches, gradient checkpointing) under the same regime?
- **Quantified Memory and Compute Overhead**: Does the paper report concrete GPU memory usage, training throughput, and wall-clock time versus comparable methods (e.g., DMD, SiD, VSD, sCM) under standardized settings?
- **Training Stability Without Crutches**: Does the method converge stably at scale without relying on frozen backbone "teachers," special normalization layers, or alternating optimization phases that add hidden complexity?

---

**Principle 4: Mechanistic Justification and Prior Work Distinction for Non-Standard Design Choices**

**Definition:**  
Novel few-step methods often extend or modify established frameworks—such as introducing negative time intervals, twin trajectories, or self-adversarial losses—and reviewers demand clear evidence that these choices are both conceptually distinct from prior art and empirically necessary. Vague claims of novelty are insufficient when related work has explored similar time-interval expansions or distribution matching; authors must pinpoint the exact mechanistic difference (e.g., comparison versus separation, rectification versus stabilization). Furthermore, every non-obvious design decision—such as using a negative versus positive time interval for fake data, or avoiding a simple boolean condition—should be supported by ablations showing performance degradation when that choice is altered. The method's narrative should explicitly map each component to a specific problem it solves (e.g., generating "hard" negatives for contrastive learning) rather than presenting a black-box recipe. This principle ensures that contributions are built on a deep understanding of why the method works, not merely that it works.

**Core Evaluation Criteria:**
- **Clear Differentiation from Related Work**: Does the paper explicitly contrast its mechanism with similar prior techniques (e.g., FACM's stabilization vs. twin-trajectory rectification, sCM's JVP handling vs. unified modeling) rather than omitting discussion?
- **Ablations of Key Design Choices**: Are alternative formulations systematically tested (e.g., positive time intervals instead of negative, boolean fake/real conditioning, varying consistency approximation orders) to prove necessity?
- **Mechanistic Explainability**: Does the work explain what specific failure mode each component addresses (e.g., insufficient contrast between real and fake trajectories, trajectory curvature) using both theory and empirical failure-case analysis?
- **Novelty Beyond Loss Combination**: If the method combines existing loss terms, does it demonstrate that the specific architectural integration—not merely the combination itself—yields unique scalability or stability properties?

---

**Principle 5: Diversity Preservation and Mode Collapse Detection in Low-NFE Generation**

**Definition:**  
Aggressive distillation to one or two function evaluations often trades sampling diversity for speed, leading to mode collapse where models produce nearly identical outputs across different noise seeds. Reviewers scrutinize whether high benchmark scores reflect genuine generative capability or exploit evaluation protocols that average multiple samples per prompt, thereby masking low diversity. Quantitative perceptual metrics such as LPIPS, alongside qualitative visualizations across multiple random seeds, are essential to verify that the model maintains output diversity comparable to multi-step baselines. Authors should also examine whether their method introduces instability when transferred across domains (e.g., from text-to-image to class-conditional image generation) or suffers from training-data-specific overfitting. Detecting and reporting mode collapse is particularly important when comparing against adversarial methods or strong baselines like consistency distillation, which are known to degrade in diversity. This principle ensures that few-step acceleration does not come at the unacceptable cost of reduced generative coverage or memorization of narrow output modes.

**Core Evaluation Criteria:**
- **Diversity Quantification**: Does the paper report explicit diversity metrics (e.g., average pairwise LPIPS across samples per prompt) and compare them against collapsed baselines and original multi-step models?
- **Benchmark Gaming Detection**: Are results analyzed for exploitability (e.g., generating identical images to inflate averaged benchmark scores), and are such behaviors diagnosed and corrected?
- **Cross-Domain Stability**: Does the method maintain reasonable performance and diversity across different data domains (e.g., ImageNet, text-to-image, image editing) without severe degradation or loss conflicts?
- **Visual Fidelity at Low NFE**: Are uncurated, high-resolution visualizations provided for 1-NFE and 2-NFE outputs across varied prompts to substantiate both quality and diversity claims beyond scalar metrics?