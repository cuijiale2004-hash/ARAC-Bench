**Principle 1: Scalability Verification and Memory-Efficient Linear Projection Efficacy for Deep Weight Space Flow Matching**

**Definition:**  
Deep weight space generative models must provide convincing evidence that they can scale to high-dimensional parameter spaces characteristic of modern neural networks without relying on lossy nonlinear autoencoders or prohibitive memory footprints. The reviewers expect explicit demonstration of scaling beyond trivial model sizes—ideally to hundreds of millions of parameters or more—using techniques such as incremental or dual PCA, and a transparent discussion of absolute VRAM requirements, wall-clock training time, and generation throughput. It is insufficient to claim scalability in principle; the work must empirically validate that linear projections preserve the structure and performance of the generated weights through rigorous ablations comparing full-rank, projected, and reconstructed models. Furthermore, the method should articulate a credible path toward billion-parameter scales, addressing whether the chosen dimensionality reduction remains computationally tractable and whether the flow matching backbone itself becomes a bottleneck. This perspective is crucial because the primary barrier to adoption of weight-space generative models is their perceived inability to handle contemporary large-scale architectures efficiently.

**Core Evaluation Criteria:**
- **Empirical Scaling Evidence:** Does the work present results for models exceeding tens of millions of parameters and explicitly discuss memory limits and wall-clock times?
- **Projection Fidelity:** Are ablations provided showing that PCA-based compression and reconstruction preserve accuracy and diversity relative to full-rank training and generation?
- **Hardware Transparency:** Are absolute VRAM usage, micro-batch sizes, and GPU types reported to allow reproducible assessment of computational tractability?
- **Path to Extreme Scale:** Does the paper provide a technically grounded argument for scaling to billion-parameter models, or honestly acknowledge hard limits?

---

**Principle 2: Capacity-Dependent Quantification of Permutation Canonicalization Costs and Benefits Across Architectures**

**Definition:**  
Because neural network weight spaces exhibit complex permutation symmetries that fragment the loss landscape, generative methods must carefully evaluate whether canonicalization strategies—such as Git Re-Basin or TransFusion—genuinely improve generation quality or merely add computational overhead. The evaluation should move beyond qualitative arguments and provide quantitative, architecture-specific evidence showing how canonicalization affects final test accuracy, training stability, and variance across generated ensembles, particularly conditional on flow model capacity. Reviewers note that for modest-dimensional weights or high-capacity flow models, canonicalization may offer diminishing returns, whereas for very large transformers it can be essential; this nuanced, scale-dependent behavior must be documented. The work should also report the computational cost of canonicalization relative to overall training and generation time, enabling a clear cost-benefit analysis. Establishing when and why symmetry breaking is necessary ensures that future methods do not adopt expensive alignment procedures indiscriminately.

**Core Evaluation Criteria:**
- **Quantitative Accuracy Impact:** Are there controlled experiments measuring the effect of canonicalization on final generated model accuracy across varying flow model capacities?
- **Architecture-Specific Analysis:** Does the evaluation separately report canonicalization benefits for different classes of architectures, noting where it becomes essential versus redundant?
- **Computational Cost Accounting:** Is the time and memory overhead of permutation alignment explicitly quantified relative to flow matching training and sampling?
- **Capacity-Dependent Behavior:** Does the work show whether high-capacity flow models can implicitly learn around symmetries, rendering explicit canonicalization unnecessary for smaller scales?

---

**Principle 3: Multimetric Assessment of Generated Weight Diversity, Distributional Fidelity, and Training Set Memorization**

**Definition:**  
A generative model of neural network weights must demonstrate that it produces novel parameter configurations rather than memorizing or trivially perturbing training instances, especially when trained on small collections of independently initialized terminal models. Reviewers demand multifaceted diversity metrics—such as L2 distance, cosine similarity, Wasserstein distance, Jensen-Shannon divergence, and nearest-neighbor analysis—computed not only for small MLPs on simple datasets but also for large convolutional and attention-based architectures. The work should explicitly test whether the best generated networks are near-exact copies of training set members and whether behavioral metrics like prediction disagreement align with geometric distances in weight space. Additionally, the sensitivity of diversity to design choices like source distribution variance must be characterized to rule out mode collapse. This principle is vital because the utility of weight generation hinges on producing diverse, high-performing ensembles for uncertainty quantification and model soups, which collapses if the generator simply interpolates memorized points.

**Core Evaluation Criteria:**
- **Geometric Distance Metrics:** Does the paper report L2, cosine similarity, Wasserstein, and nearest-neighbor distances between generated and training weights, supplementing behavioral overlap measures?
- **Scale of Diversity Evaluation:** Are diversity metrics reported for large-scale architectures rather than only trivially small networks?
- **Memorization Audits:** Is there an explicit test checking whether generated weights are near-exact copies of training set members?
- **Source Distribution Sensitivity:** Are the effects of source distribution variance on diversity and mode collapse systematically ablated?

---

**Principle 4: Fair Cross-Benchmark Evaluation and Computational Efficiency Auditing for Complete Weight Generation Methods**

**Definition:**  
Rigorous experimental practice in weight generation requires broad benchmarking across multiple datasets and architectures, alongside scrupulously fair comparisons with state-of-the-art methods that account for differences in generation scope and post-processing requirements. Reviewers emphasize the importance of distinguishing between methods that generate complete weights versus partial weights, those that require fine-tuning versus those that are inference-ready, and those conditioned on rich metadata versus unconditional models. Efficiency must be audited comprehensively, reporting per-model training time, generation latency, and memory consumption on standardized hardware, with ablations isolating the contribution of components like PCA. Transfer learning experiments must include strong baselines such as original pretrained models, random initialization, and relevant SOTA methods under identical fine-tuning protocols. This evaluative discipline prevents inflated claims and ensures that reported advances in speed or accuracy reflect genuine methodological progress rather than incomparable experimental setups.

**Core Evaluation Criteria:**
- **Benchmark Breadth:** Does the evaluation span multiple datasets and modalities (vision, tabular, language) and diverse architecture families?
- **Fair SOTA Comparison:** Are comparisons clearly delineated by complete vs. partial weight generation, fine-tuning requirements, and conditioning information?
- **Efficiency Benchmarking:** Are training and generation times reported per model with standardized hardware, including breakdowns of PCA, canonicalization, and inference?
- **Transfer Baseline Rigor:** Do transfer learning experiments include baselines of original pretrained models, random initialization, and appropriate SOTA methods under matched protocols?

---

**Principle 5: Cross-Dataset Transferability and Conditional Controllability of Neural Network Weight Distributions**

**Definition:**  
Weight generation research should ultimately demonstrate functional utility beyond replicating in-distribution training performance, including effective transfer to unseen datasets, adaptation to mismatched output dimensions, and conditional controllability over architectural or dataset attributes. Reviewers expect evidence that generated weights serve as strong priors for transfer learning—ideally matching or exceeding original pretrained models when fine-tuned on new domains—and that the method handles practical challenges like varying numbers of target classes. The exploration of conditional generation, whether by dataset embedding, class label, or architecture specification, should be evaluated for its ability to produce coherent weights outside the training mixture, even if preliminary. Extending evaluation beyond classification to regression or natural language tasks further establishes generalizability. This principle ensures that weight-space generative models are positioned as practical tools for rapid model deployment and ensemble construction rather than narrow scientific curiosities.

**Core Evaluation Criteria:**
- **Transfer Learning Efficacy:** Does the work evaluate zero-shot and fine-tuned transfer to unseen datasets, comparing generated weights against original pretrained weights and random initialization?
- **Output Mismatch Handling:** Is there a clear protocol for adapting generated models to tasks with different numbers of output classes or regression targets?
- **Conditional Generation Feasibility:** Are results presented for generating weights conditioned on dataset identity, class label, or architectural specification?
- **Cross-Domain Applicability:** Does the evaluation extend beyond image classification to other modalities to demonstrate broad functional utility?