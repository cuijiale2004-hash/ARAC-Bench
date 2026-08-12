**Principle 1: Domain-Specific Stability-Plasticity Trade-off Formulation for Continual Generative Recommendation**

**Definition:**  
In continual recommendation, the stability-plasticity dilemma diverges fundamentally from traditional continual learning paradigms in computer vision or standard NLP. The objective is not to preserve performance on past tasks, but to retain enduring user preferences that remain predictive while aggressively overwriting outdated interests that may actively harm current predictions. A high-quality study must therefore articulate a domain-specific conceptual model of preference evolution, explicitly distinguishing between transient noise, durable long-term interests, and emerging trends. The evaluation must move beyond aggregate accuracy metrics to demonstrate that the method achieves this nuanced balance, as naive stability mechanisms can enforce rigid memorization of obsolete behaviors. This perspective is crucial because recommendation systems operate in inherently non-stationary environments where user interests drift, and the utility of past knowledge is conditional on its relevance to future predictions rather than its historical accuracy.

**Core Evaluation Criteria:**
- **Conceptual Clarity of Preference Dynamics**: Does the work explicitly define what "stability" and "plasticity" mean in the recommendation context, and how they differ from task-incremental learning in other domains?
- **Realism of Temporal Setup**: Does the experimental design reflect realistic deployment scenarios, including quantified distribution shift between chronological blocks and periodic updates rather than artificial user-disjoint splits?
- **Disentangled Evaluation**: Are there targeted analyses distinguishing retention of useful long-term preferences from harmful outdated ones, such as dormant-user analysis and new-user adaptation tests?
- **Appropriate Forgetting Metrics**: Does the work use domain-appropriate forgetting measures that account for natural preference drift, rather than blindly applying classification-style forgetting ratios that penalize necessary adaptation?

---

**Principle 2: Theoretical Mechanism Explanation for Direction-Wise Parameter-Efficient Adaptation in Low-Rank Subspaces**

**Definition:**  
When proposing parameter-efficient continual adaptation methods—particularly those operating within low-rank adapter subspaces—it is insufficient to demonstrate empirical gains alone. The research must provide a mechanistic understanding of how the regularization or update mechanism modulates individual directions of parameter change, ideally linking mathematical formulations to semantic interpretations such as eigenvectors of gradient covariance representing latent preference axes. Such theoretical grounding enables reviewers to assess whether the method's behavior is principled or merely heuristic, and whether the stability-plasticity balance emerges from data-aware geometric properties rather than opaque hyperparameter tuning. This is especially critical in LoRA-based continual learning, where the restricted subspace makes the geometry of updates a first-order determinant of interference and retention.

**Core Evaluation Criteria:**
- **Geometric Interpretability**: Does the theory provide a closed-form or analytically tractable characterization of how updates interpolate between old and new states in the adapter subspace?
- **Data-Awareness**: Does the analysis demonstrate that regularization strength adapts intrinsically to the signal strength along different parameter directions, for example via eigenvalues of the tangent-feature covariance?
- **Semantic Mapping**: Are theoretical constructs such as principal directions and eigenvalues mapped to intuitive operational concepts, such as specific preference axes or short-term versus long-term interests?
- **Connection to Instantiation**: Is the practical algorithmic instantiation, such as a KL proximal or L2 proximal, rigorously derived from or justified by the general theoretical framework rather than introduced ad hoc?

---

**Principle 3: Comprehensive Baseline Coverage and Cross-Architectural Validation for Continual PEFT Methods**

**Definition:**  
The proliferation of parameter-efficient fine-tuning and continual learning techniques demands that new methods be evaluated against a sufficiently diverse and state-of-the-art set of baselines, spanning not only different adapter aggregation strategies but also alternative PEFT paradigms and backbone architectures. In LLM-based recommendation, this includes comparing against prompt-based tuning, full-parameter fine-tuning, cumulative LoRA variants with and without inheritance, orthogonal subspace methods, and attentional mixture approaches. Furthermore, robustness must be validated across multiple generative recommender backbones to ensure that gains are not artifacts of a specific tokenizer or model structure. Without such comprehensive coverage, claims of superiority remain fragile and may reflect implementation gaps rather than genuine methodological advances.

**Core Evaluation Criteria:**
- **PEFT Diversity**: Are alternative parameter-efficient methods, such as prompt tuning and full fine-tuning as an upper or lower bound, included and justified in the context of generative recommendation?
- **Continual Strategy Breadth**: Does the evaluation cover the full spectrum of continual LoRA strategies, including single evolving, cumulative summation, orthogonal projection, attentional mixture, and interpolation-based approaches?
- **Backbone Generalization**: Is the method tested on multiple LLM-based recommender architectures to verify that improvements are not backbone-specific?
- **Training Paradigm Comparisons**: Are comparisons made against standard industry practices such as periodic full retraining, and are the relative merits clearly articulated?

---

**Principle 4: Multi-Pattern Temporal Dynamics Validation and Domain-Appropriate Forgetting Quantification**

**Definition:**  
User preference drift is not monolithic; it manifests as linear evolution, cyclical re-engagement, sudden interest shifts, and dormant-returning patterns. A rigorous continual recommendation study must disaggregate its evaluation to probe each of these dynamics separately, rather than relying solely on global metrics averaged over all users and time stages. Additionally, because recommendation is inherently about predicting future behavior rather than classifying static categories, conventional catastrophic forgetting metrics from computer vision are often misleading. Instead, the evaluation must measure whether the model can selectively discard obsolete trends while recalling long-term preferences when they become relevant again. This requires explicit quantification of distribution shift and targeted user-group analyses that serve as proxies for distinct drift patterns.

**Core Evaluation Criteria:**
- **Drift Pattern Disaggregation**: Does the evaluation separately analyze performance on continuous users, dormant-returning users, and newly emerged users?
- **Distribution Shift Quantification**: Is the magnitude and nature of temporal drift explicitly measured, for example via domain discrimination AUC drift scores, to validate the realism of the experimental setup?
- **Selective Retention versus Rigid Forgetting**: Does the work distinguish between harmful catastrophic forgetting and beneficial selective forgetting of outdated preferences?
- **Cross-Domain Temporal Generalization**: Are experiments conducted across multiple domains with different semantic richness and interaction patterns, such as e-commerce product reviews and location-based check-ins?

---

**Principle 5: Storage and Computational Efficiency Analysis for Incremental Adapter Deployment**

**Definition:**  
A central motivation for using parameter-efficient fine-tuning in continual learning is the promise of efficient deployment, both in terms of memory and storage overhead and computational cost per update. In streaming or periodically updated recommender systems, methods that accumulate adapters linearly with the number of time stages impose prohibitive storage and inference costs, undermining their practical viability. Consequently, high-quality research must provide explicit complexity analysis and empirical measurements of training and inference overhead, comparing against both cumulative and single-adapter baselines. This analysis is crucial for distinguishing theoretically elegant methods from those that are actually deployable at scale in production environments where latency and memory constraints are paramount.

**Core Evaluation Criteria:**
- **Storage Complexity Analysis**: Is the storage cost characterized asymptotically, such as O(1) versus O(T), and validated empirically relative to the number of continual stages?
- **Training Overhead Measurement**: Are training time and per-iteration computational costs reported, including the overhead of regularization terms?
- **Inference Efficiency**: Does the method introduce additional latency during autoregressive generation, such as prompt retrieval or adapter aggregation, that could negate parameter efficiency?
- **Scalability Projections**: Does the work discuss how efficiency trade-offs evolve as the number of stages, users, or adapter ranks increases?