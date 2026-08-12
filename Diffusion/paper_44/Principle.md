**Principle 1: Topological Fidelity and Robustness of Non-Learned Global Structure Extraction in Graph Diffusion Models**

**Definition:**  
This principle evaluates whether global information extraction methods—particularly non-learned clustering algorithms such as K-means and spectral clustering applied to node embeddings or geometric coordinates—genuinely capture topology-aware global structure or merely provide coarse, heuristic approximations. It examines whether authors rigorously justify the choice of fixed clustering over end-to-end learnable hierarchical pooling, analyze sensitivity to clustering algorithms and hyperparameters such as the number of clusters, and demonstrate robustness across diverse graph domains including generic graphs and molecular structures. The principle further assesses whether comparisons with modern deep clustering or differentiable pooling methods are provided, and whether the global extraction approach generalizes to unseen graph types without manual algorithm selection. A method that relies on brittle or domain-specific clustering without justification or ablation is considered inadequately validated.

**Core Evaluation Criteria:**
- **Justification for Non-Learned Clustering:** Does the work provide a compelling rationale for using fixed clustering methods over trainable hierarchical pooling or deep clustering approaches in modern GNNs?
- **Sensitivity Analysis:** Are comprehensive experiments conducted across multiple clustering algorithms (e.g., K-means, spectral, GMM, Louvain, deep clustering) and varying numbers of clusters to demonstrate robustness?
- **Cross-Domain Generalization:** Does the proposed global extraction method generalize across different graph types (generic vs. molecular), or does it require hand-tuned algorithm selection for each domain?
- **Comparison with Learnable Alternatives:** Are quantitative comparisons provided against end-to-end hierarchical methods (e.g., DiffPool, Dink-Net) to justify the efficiency-effectiveness trade-off of non-learned extraction?

---

**Principle 2: Theoretical Substantiation of Joint Distribution Factorization via Alternating Conditional Processes**

**Definition:**  
This principle evaluates whether the decomposition of joint distributions into alternating conditional processes (e.g., global-to-local and local-to-global) is theoretically grounded beyond standard conditional probability identities. It examines whether authors provide formal analysis or strong empirical evidence that the proposed factorization scheme meaningfully models the joint distribution, whether stability and convergence properties are analyzed, and whether the alternating mechanism is justified mechanistically rather than purely intuitively. The principle also assesses whether ablation studies convincingly demonstrate superiority over simultaneous updates or unidirectional conditioning, and whether the probabilistic interpretation is positioned appropriately—as a rigorous guarantee versus an efficient factorization heuristic. Mere restatement of probability chain rules without substantive theoretical or empirical validation is insufficient.

**Core Evaluation Criteria:**
- **Substantive Theoretical Justification:** Does the work provide theoretical evidence that the proposed alternating factorization is valid and non-trivial, or does it rely solely on standard conditional-probability identities that hold for any joint distribution?
- **Stability and Convergence Analysis:** Are formal guarantees or detailed empirical analyses provided for the stability and convergence of the alternating mechanism compared to simultaneous updates?
- **Ablation of Conditioning Directions:** Are ablation studies included comparing alternating conditioning, simultaneous conditioning, global-to-local only, and local-to-global only to isolate the contribution of each component?
- **Clear Positioning of Claims:** Does the paper honestly position its theoretical contributions (e.g., as strict guarantees vs. efficient conditional factorizations), avoiding overstated claims about probabilistic rigor?

---

**Principle 3: Chemical and Structural Validity with Failure Mode Analysis in Graph Generation Benchmarks**

**Definition:**  
This principle assesses whether generated graphs satisfy domain-specific structural constraints, with particular emphasis on chemical validity for molecular generation. For molecular graphs, it evaluates whether the model reliably produces chemically valid structures (correct valence, bond types, functional groups) and whether validity metrics are competitive with strong autoregressive and one-shot baselines. For generic graphs, it examines validity, uniqueness, and novelty (V.U.N.) scores. The principle requires authors to analyze failure cases, categorize invalid outputs, and distinguish between limitations inherent to the latent diffusion paradigm (e.g., autoencoder reconstruction errors) and fundamental architectural incapacities. A significant validity gap versus strong baselines without adequate failure analysis undermines claims of effective local constraint capture.

**Core Evaluation Criteria:**
- **Validity Competitiveness:** Are validity rates on standard molecular benchmarks (e.g., ZINC250k, QM9) competitive with strong baselines, or are there significant gaps indicating failure to capture local chemical rules?
- **Failure Case Analysis:** Does the work examine and categorize failure modes (e.g., Kekulization errors, valency violations, invalid substructures) rather than reporting only aggregate metrics?
- **Paradigm Contextualization:** Does the paper appropriately contextualize validity limitations within the latent diffusion framework (e.g., edge reconstruction errors from autoencoders) versus claiming universal superiority?
- **Post-Processing Transparency:** Are post-processing corrections (e.g., RDKit sanitization) explicitly disclosed, and are their implications for fair comparison discussed?

---

**Principle 4: Comprehensive Benchmark Coverage and Scalability Validation Across Graph Sizes and Domains**

**Definition:**  
This principle assesses whether experimental validation spans a sufficiently diverse range of benchmarks—including small generic graphs, standard molecular datasets, and large-scale graphs such as proteins—to substantiate claims of generalizability and scalability. It examines whether the method maintains consistent performance across varying graph sizes and topological characteristics, whether baseline comparisons are fair and comprehensive, and whether the experimental scope aligns with the paper's stated applicability. The principle also evaluates whether conditional generation scenarios and domain extensions (e.g., from 2D to 3D molecular structures) are explored to demonstrate broad utility. Limited evaluation to small or homogeneous datasets weakens generalizability claims.

**Core Evaluation Criteria:**
- **Benchmark Diversity:** Does the evaluation cover small generic graphs, mainstream molecular datasets, and larger-scale graphs (e.g., proteins with 100–500 nodes) to validate scalability?
- **Performance Consistency:** Does the method maintain stable performance across different graph sizes, or does quality degrade significantly for larger or more complex topologies?
- **Fair Baseline Comparisons:** Are baselines selected appropriately for each task, and are comparisons conducted under identical evaluation protocols (train/test splits, metrics, preprocessing)?
- **Extensions and Conditional Scenarios:** Are extensions to conditional generation, 3D structure generation, or property-conditioned tasks included to demonstrate broad applicability beyond unconditional generation?

---

**Principle 5: Rigorous Complexity Analysis and Empirical Efficiency Validation for Multi-Branch Diffusion Architectures**

**Definition:**  
This principle evaluates whether computational efficiency claims for architectures incorporating multiple diffusion branches, dual conditioning mechanisms, and alternating update steps are substantiated by both theoretical complexity analysis and empirical measurements of sampling time and memory consumption. It examines whether the overhead from additional networks, clustering procedures, and conditioning operations is quantified relative to standard latent diffusion baselines, and whether efficiency advantages (if claimed) are validated across different graph sizes and datasets. The principle also assesses whether the trade-off between generation quality and computational cost is explicitly characterized. Claims of efficiency must be supported by actual timing and memory benchmarks, not just asymptotic arguments.

**Core Evaluation Criteria:**
- **Theoretical Complexity Analysis:** Does the paper provide rigorous complexity analysis accounting for all components, including clustering, dual denoising networks, and conditioning mechanisms?
- **Empirical Time and Memory Benchmarks:** Are actual sampling times and memory usages reported and compared against comparable baselines under identical hardware and software conditions?
- **Scalability with Graph Size:** Does the efficiency analysis validate how computational costs scale with the number of nodes and clusters, particularly for larger graphs?
- **Quality-Efficiency Trade-off:** Is the relationship between computational overhead and generation quality explicitly characterized, demonstrating that additional costs yield proportional or superior improvements?