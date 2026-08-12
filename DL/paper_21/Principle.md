**Principle 1: Conceptual Validity of Deep Representation Learning Claims in Gradient-Free Analytic Federated Architectures**

**Definition:**  
This principle evaluates whether a gradient-free analytic method genuinely achieves deep representation learning or merely performs non-linear classification atop frozen backbone features. In the reviewed work, multiple reviewers challenged the representation learning claim because the backbone remains frozen and all transformations are confined to fixed feature spaces. A rigorous assessment must determine if the proposed residual blocks extract novel semantic information or simply enhance linear separability of pre-extracted features. The evaluation should reference canonical definitions of representation learning, including automated feature extraction and utility for downstream predictors. Furthermore, reviewers demanded intermediate feature analyses—such as separability metrics, layer-wise visualizations, and discriminability scores—to substantiate that deeper layers yield progressively more abstract representations. The principle also scrutinizes whether analogies to gradient-based architectures (e.g., MLPs with residual connections) are structurally valid or mask fundamental limitations imposed by the analytic, closed-form paradigm. Ultimately, the work must clearly delineate the boundary between representation learning and feature reweighting to justify its architectural depth.

**Core Evaluation Criteria:**
- **Alignment with Canonical Definitions:** Does the method satisfy established criteria for representation learning (automated non-linear feature extraction and demonstrable utility for downstream prediction), or does it conflate feature reweighting with semantic feature learning?
- **Distinction from Fixed-Feature Classification:** Is there evidence that the model transforms features beyond simple linear or non-linear remapping of frozen backbone outputs, particularly when the backbone discards task-relevant information?
- **Intermediate Feature Separability:** Are quantitative analyses (e.g., Compactness-Separation-Ratio, Inverse Fisher Score, Discriminative Measure) provided to verify that features become increasingly separable across analytic layers on both training and test sets?
- **Architectural Analogy Validity:** Are claims drawing parallels to gradient-based deep networks (such as residual MLPs) theoretically and empirically justified, given the absence of backpropagation and end-to-end optimization?

---

**Principle 2: Methodological Fairness and Exhaustive Benchmarking Across IID and Non-IID Federated Regimes**

**Definition:**  
This principle ensures that empirical evaluations fairly isolate a method’s intrinsic advantages from artificial confounds inherent in the experimental protocol. Reviewers noted that comparing an analytic method with innate heterogeneity invariance against gradient-based baselines solely under severe Non-IID conditions obscures whether gains stem from the analytic formulation or from bypassing heterogeneity effects. A scientifically rigorous study must therefore include both highly heterogeneous and idealized IID partitions, alongside centralized analytic baselines, to disentangle efficiency gains from robustness artifacts. Additionally, architectural fairness demands that baseline methods be evaluated with comparably deepened classification heads or stronger backbones to match the proposed model’s capacity. The evaluation must also incorporate recent state-of-the-art competitors rather than relying exclusively on older benchmarks, ensuring the experimental landscape reflects contemporary progress. Without such controlled comparisons, claims of superiority in accuracy or efficiency remain methodologically unconvincing.

**Core Evaluation Criteria:**
- **IID and Centralized Baseline Inclusion:** Are results reported under completely IID data distributions and against a centralized analytic solution to verify that improvements arise from representation depth rather than immunity to Non-IID effects?
- **Architectural Capacity Matching:** Are gradient-based baselines evaluated with similarly deepened trainable layers (e.g., 10–20 layers) or stronger backbones (e.g., ResNet-34, ViT) to ensure the comparison reflects model capacity rather than protocol bias?
- **Contemporary Competitor Coverage:** Does the benchmark include recent advances (e.g., FedAWA, FedLPA) in addition to classical methods, and are hyperparameters tuned fairly for all methods?
- **Isolation of Heterogeneity Invariance Effects:** Does the experimental design allow reviewers to distinguish performance gains due to analytic efficiency from those due to the method’s inherent avoidance of gradient-based heterogeneity degradation?

---

**Principle 3: Theoretical Soundness and Generalization Bounds for Layer-Wise Sandwiched Least-Squares Optimization**

**Definition:**  
This principle assesses the completeness and practical relevance of theoretical guarantees offered for layer-wise closed-form optimization in distributed settings. The review process revealed concerns about unstated assumptions—such as shared random seeds, regularization hyperparameters, finite-precision arithmetic, and the behavior of the theory under imperfect backbone features. A robust theoretical framework must explicitly articulate its prerequisites and demonstrate that core results, including heterogeneity invariance and monotonic empirical risk reduction, extend beyond idealized conditions. Moreover, reviewers required evidence that monotonically decreasing training loss translates into improved generalization bounds, for instance via VC-dimension or Rademacher complexity analyses that remain constant across layers. The principle further demands examination of whether theoretical claims hold when backbone features suffer domain shift, noise, or weak correlation with labels, as these scenarios define the realistic operating regime of federated learning. Finally, architectural innovations must be causally linked to theoretical properties; for example, residual skip connections should be proven essential for the non-increasing risk guarantee rather than treated as heuristic add-ons.

**Core Evaluation Criteria:**
- **Explicitness of Theoretical Assumptions:** Are all prerequisites—such as synchronized random projections, regularization terms, and infinite-precision arithmetic—clearly stated, and is their impact on the final solution analyzed?
- **Generalization Beyond Empirical Risk:** Does the theory extend from monotonic training loss to rigorous generalization error bounds (e.g., via Rademacher complexity or VC-dimension) under fixed classifier complexity?
- **Robustness to Imperfect Backbone Features:** Do invariance and convergence results remain valid when backbone features are noisy, mismatched with client data, or contain low mutual information with labels?
- **Causal Architectural-Theoretical Linkage:** Are key structural components (e.g., residual skip connections) proven necessary for the theoretical guarantees, such that their removal invalidates the core theorems?

---

**Principle 4: Empirical Verification of Wall-Clock Efficiency and Scalability to Large Pretrained Backbones**

**Definition:**  
This principle mandates rigorous empirical quantification of computational costs and validation of scalability claims across diverse backbone architectures. Reviewers challenged the asserted efficiency advantages by pointing to quadratic parameter complexity relative to linear parameter-efficient fine-tuning alternatives and the absence of wall-clock runtime, FLOP, or GPU-hour metrics. To satisfy this criterion, a work must provide precise profiling of total wall-clock time, per-client latency, and communication overhead, comparing them directly against gradient-based and one-shot federated baselines under identical hardware conditions. Furthermore, scalability cannot be extrapolated from small CNN backbones alone; experiments must explicitly evaluate large-scale models such as Vision Transformers to confirm that analytic layers retain negligible marginal cost and stable accuracy gains. The evaluation should also contextualize the trade-off between intermediate feature dimensions and computational burden, clarifying that efficiency stems from the gradient-free protocol rather than merely from reduced parameter counts. Ultimately, efficiency claims require reproducible evidence that the method sustains orders-of-magnitude speedups without sacrificing model utility.

**Core Evaluation Criteria:**
- **Quantitative Efficiency Profiling:** Are wall-clock runtime, FLOPs, GPU-hours, and per-layer communication costs reported and compared against the most efficient gradient-based and one-shot baselines?
- **Large Backbone Scalability:** Are experiments conducted on large-scale architectures (e.g., ViT-B-16) to verify that analytic layers provide stable accuracy improvements with marginal runtime increases across different backbone capacities?
- **Parameter Complexity Contextualization:** Is the method’s asymptotic complexity (e.g., quadratic dependence on feature dimensions) honestly disclosed and compared against linear PEFT alternatives under realistic hidden-dimension regimes?
- **Reproducible Cost-Accuracy Trade-offs:** Does the study demonstrate that performance gains are achieved with negligible time overhead (e.g., sub-second per client per layer) and that deeper layers do not trigger numerical instability or prohibitive costs?

---

**Principle 5: Robustness Validation Under Practical Federated Deployment Constraints**

**Definition:**  
This principle examines whether a federated learning method maintains performance and reliability under the messy, asynchronous, and resource-constrained conditions of real-world deployment. The review highlighted that theoretical invariance properties may degrade when clients drop out partially or participate inconsistently across aggregation rounds, necessitating empirical stress tests with participation rates as low as fifty percent. Additionally, because practical clients often supply noisy or inconsistently labeled data—particularly in cross-silo medical or IoT applications—the method must be evaluated under controlled label corruption and compared against cross-entropy baselines for noise sensitivity. Privacy protections, while essential, should not nullify efficiency gains; therefore, the overhead of integrating secure aggregation or encryption protocols must be quantified and shown to remain lightweight. The principle also requires honest disclosure of limitations regarding backbone quality, specifically whether the method can compensate for domain shift or whether its performance collapses when pre-trained features are misaligned with local distributions. Robustness under these practical constraints separates theoretically elegant methods from deployable federated systems.

**Core Evaluation Criteria:**
- **Partial Participation Resilience:** Are experiments conducted with varying participation rates (e.g., 50%–90%) and inconsistent client subsets across within-layer aggregation rounds to verify model stability?
- **Noise and Corruption Robustness:** Is performance evaluated under systematic label noise (e.g., 10%–50% random flips) to demonstrate superior robustness compared to gradient-based cross-entropy optimization?
- **Privacy-Efficiency Overhead Analysis:** Are the computational and communication costs of privacy-enhancing protocols (e.g., secure aggregation, secret sharing) quantified and shown to preserve the method’s efficiency advantage?
- **Domain Shift and Backbone Mismatch Evaluation:** Does the study test scenarios where the pre-trained backbone has never seen the clients’ task data, and are limitations regarding uncorrelated or misaligned features transparently discussed rather than obscured?