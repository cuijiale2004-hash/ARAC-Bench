**Principle 1: Theoretical Guarantees and Empirical Optimization Convergence for Non-Split Multimodal Concept Dictionaries**

**Definition:**  
In multimodal interpretability research using dictionary learning, it is crucial to distinguish between the mathematical existence of aligned decompositions and the practical ability of stochastic optimization to discover them. The field requires rigorous theoretical motivation showing that poor alignment—such as split dictionaries—is an optimization bias rather than a fundamental representational limit imposed by the linear representation hypothesis. However, existence proofs alone are insufficient for establishing a usable method; reviewers must assess whether authors empirically validate that their training objective actually induces convergence toward these theoretically favorable dictionaries in highly non-convex landscapes. Furthermore, the work must establish a causal relationship between the proposed inductive biases—such as group-sparse regularization or cross-modal masking—and the empirical emergence of multimodal concepts. Without this bridge, the method remains heuristic, and the community cannot determine whether observed improvements stem from the intended mechanism or incidental training effects. Consequently, reviewers should demand both formal arguments about solution existence and experimental evidence that the optimization trajectory reliably finds multimodal rather than split dictionaries. This includes analyzing whether the loss landscape favors joint support and whether initialization or architectural choices are necessary for convergence.

**Core Evaluation Criteria:**
- **Existence vs. Constructibility**: Does the work clearly separate theoretical existence results from empirical evidence that the proposed training procedure discovers such solutions under gradient-based optimization?
- **Causal Link to Inductive Bias**: Is there evidence that the specific proposed regularization is responsible for escaping split-dictionary solutions, rather than any generic training modification?
- **Landscape Characterization**: Does the paper provide any analysis—empirical or formal—of why non-convex training converges to aligned dictionaries instead of local optima?
- **Theoretical-Empirical Consistency**: Do the qualitative and quantitative results actually reflect the theoretical predictions (e.g., improved joint support) rather than merely improved reconstruction?

---

**Principle 2: Cross-Architecture Generalization and Hyperparameter Robustness in Multimodal Sparse Autoencoder Training**

**Definition:**  
Because multimodal embedding spaces vary significantly across foundation model families, claims about dictionary learning improvements must be validated across diverse pretrained encoders and modality pairs. A method that corrects split dictionaries in a single embedding space may fail to generalize if the phenomenon is tied to idiosyncratic geometric properties of that space. Therefore, reviewers must evaluate whether reported gains hold across fundamentally different encoders or are merely artifacts of a specific backbone or dataset. Additionally, sparse autoencoders are notoriously sensitive to expansion ratios, sparsity constraints, and regularization strengths. High-quality research must demonstrate that performance rankings and qualitative insights remain stable under varying configurations rather than relying on a single favorable tuning. This principle demands systematic sweeps or sensitivity analyses that situate the contribution within a realistic range of training conditions. By enforcing cross-architecture and cross-hyperparameter validation, the field avoids overfitting review standards to narrow experimental settings and ensures methodological advances are broadly applicable.

**Core Evaluation Criteria:**
- **Architectural Breadth**: Are experiments conducted across multiple independent multimodal encoders or modality pairs to verify that the phenomenon and remedy generalize beyond one embedding space?
- **Hyperparameter Sensitivity Analysis**: Does the paper provide systematic sweeps or stability analyses for key hyperparameters such as expansion factor, sparsity level, regularization strength, and masking probability?
- **Dataset Diversity**: Are results replicated across different data distributions beyond a single curated corpus, demonstrating robustness to domain shifts?
- **Consistency of Performance Ranking**: Do the relative improvements of the proposed method remain stable across configurations, or do they rely on a narrow optimal regime?

---

**Principle 3: Principled Ablation and Mechanistic Intuition for Group-Sparsity Induced Joint Support and Dead Neuron Reduction**

**Definition:**  
Novel regularization techniques in multimodal sparse autoencoders require granular ablation studies that isolate the contribution of each architectural or loss component—such as group-sparse regularization versus cross-modal random masking—from the overall performance gain. Without explicit dissection, reviewers cannot determine whether a full method's success is driven by one critical ingredient or by their interaction. Beyond aggregate metrics, high-quality research must provide mechanistic insight into how structured sparsity alters training dynamics, specifically explaining why it reduces dead neurons and encourages joint support across modalities. This involves analyzing the geometry of learned dictionaries, the evolution of latent activation patterns during training, or gradient flow characteristics induced by the group norm. Reviewers should treat methods lacking such ablations and intuition as potentially opaque engineering tricks rather than principled scientific advances. The goal is to move beyond showing that a method works to explaining how and why it reshapes the learned feature space. Consequently, papers should be evaluated on their ability to attribute empirical phenomena—such as reduced dead neurons or increased multimodal co-activation—to specific, testable mechanisms.

**Core Evaluation Criteria:**
- **Component Isolation**: Are there explicit comparisons between the full method, intermediate variants (e.g., group-sparse without masking), standard baselines, and prior architectures?
- **Mechanistic Explanation**: Does the work explain why group-sparsity reduces dead neurons and increases multimodal co-activation via analysis of sparsity patterns, dictionary geometry, or gradient dynamics?
- **Training Dynamics Insight**: Is there any investigation into how the loss shapes the evolution of latent support structures during training rather than only reporting final metrics?
- **Intuitive Accessibility**: Can the authors explain the method in terms that connect the mathematical formulation to an intuitive concept of joint support or shared concepts?

---

**Principle 4: Discriminative Metric Design and Validation for Single-Neuron Multimodal Monosemanticity Scoring**

**Definition:**  
Evaluating multimodal dictionaries requires metrics that precisely capture whether individual neurons encode semantically coherent concepts across modalities, as opposed to merely measuring aggregate embedding alignment or post-hoc pairwise correlations between unimodal neurons. Reviewers must scrutinize whether proposed evaluation measures are clearly differentiated from prior art, both in their mathematical formulation and in the empirical quantities they capture. A valid metric should operate at the single-concept level, assessing whether one neuron co-activates for semantically similar inputs from different modalities without requiring explicit matching of separate unimodal features. Furthermore, its utility must be validated by correlating scores with downstream interpretability tasks such as concept naming, steering, or spurious correlation detection. If a metric cannot distinguish true multimodal concepts from accidentally correlated unimodal activations, it fails its primary purpose. Therefore, reviewers should demand rigorous comparisons to existing alternatives and evidence that the new measure better reflects the intended notion of cross-modal semantic coherence. This ensures that claims of improved alignment are grounded in meaningful, interpretable quantities rather than abstract statistical artifacts.

**Core Evaluation Criteria:**
- **Differentiation from Prior Metrics**: Does the paper rigorously contrast its metric against existing alternatives, clarifying what unique quantity is measured (e.g., single-neuron co-activation versus neuron-pair bridging)?
- **Interpretability Correlation**: Is there evidence that higher scores on the proposed metric correspond to improved human-interpretable concept naming, steering, or control?
- **Compositional Validity**: Does the metric properly disentangle monosemanticity within a modality from cross-modal alignment without conflating the two?
- **Actionability**: Does the metric provide a practical tool for evaluating future methods, and is its computation reproducible without proprietary resources?

---

**Principle 5: Trade-off Analysis Between Cross-Modal Alignment and Modality-Specific Concept Diversity Preservation**

**Definition:**  
Forcing alignment across modalities may inadvertently suppress modality-unique features—such as visual textures lacking linguistic equivalents or acoustic timbre without text descriptions—thereby degrading the overall richness of the learned dictionary. High-quality research must demonstrate awareness of this fundamental tension and provide empirical or theoretical analysis showing that multimodal regularization does not catastrophically reduce unimodal concept diversity. Reviewers should look for analyses of per-modality activation distributions, concept coverage, or feature diversity metrics that reveal whether modality-specific information is preserved while shared concepts are enhanced. An effective method should expose tunable parameters, such as the regularization coefficient or masking probability, that allow practitioners to modulate the degree of alignment versus specificity. Without such trade-off analysis, a method might appear successful simply because it collapses all representations into a small set of generic shared concepts, obscuring fine-grained modality-specific semantics. Consequently, papers must be evaluated on their ability to balance cross-modal coherence with representational completeness, ensuring that alignment is achieved at the cost of specificity only when deliberately chosen. This principle safeguards against over-aligned but semantically impoverished dictionaries that would be impractical for detailed interpretability and control.

**Core Evaluation Criteria:**
- **Preservation of Unimodal Features**: Does the work analyze whether neurons still encode meaningful modality-specific concepts via per-modality activation statistics or qualitative examples?
- **Explicit Trade-off Quantification**: Is there any measurement of how alignment pressure affects the diversity of learned features, such as comparing concept coverage or activation entropy?
- **Controllability of Balance**: Does the method expose tunable parameters that modulate the degree of alignment versus specificity, and are their effects characterized?
- **Avoidance of Collapse**: Is there evidence that improved cross-modal metrics do not come at the cost of reduced within-modality semantic richness or increased representational homogenization?