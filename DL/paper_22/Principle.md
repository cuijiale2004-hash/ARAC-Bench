**Principle 1: Necessity of Explicit Higher-Dimensional Geometric Element Modeling Beyond Vertex-Edge Graphs**

**Definition:**  
Research in mesh-based neural simulation often conflates augmenting vertex features with geometric descriptors and architecturally modeling higher-dimensional topological entities. This principle demands that work explicitly disentangle the contribution of precomputed geometric features (e.g., volume, surface area) from the structural necessity of instantiating graph nodes for cells and facets. Reviewers must assess whether authors provide an ablation that isolates explicit higher-dimensional node representations from mere feature concatenation onto vertex nodes, since implicit learning of geometric patterns through vertex message passing may catastrophically fail on sparse meshes. The core issue is whether the proposed architecture captures spatial relationships and physical interactions that are fundamentally inaccessible to vertex-edge graphs, regardless of how rich the per-node features become. Establishing this necessity requires mechanistic justification grounded in geometric approximation theory—specifically, why discretely sampled vertices cannot reconstruct local physical fields or contact boundaries under coarse discretization. Ultimately, this principle ensures that claimed advances reflect genuine representational innovation rather than incremental feature engineering.

**Core Evaluation Criteria:**
- **Feature-versus-Topology Disentanglement**: Does the work include an ablation that tests geometric features averaged onto vertices without explicit cell/facet nodes (e.g., a “Model C” variant), proving that topology modeling itself—not just added descriptors—is essential?
- **Mechanistic Justification**: Does the paper explain theoretically or empirically why vertex-only sampling is insufficient for capturing local integral kernels or contact boundaries under sparse discretization?
- **Avoidance of Implicit Black-Box Claims**: Does the work refrain from claiming “geometry awareness” solely because geometric scalars are fed into a standard GNN, and instead demonstrate that information propagation through higher-dimensional edges is required?

---

**Principle 2: Computational Efficiency and Scalability of Geometry-Aware Feature Extraction**

**Definition:**  
Geometry-aware architectures incur non-trivial overhead from computing volumetric and surface descriptors at each timestep, which reviewers must rigorously weigh against accuracy gains. This principle requires authors to quantify the computational cost of geometric feature extraction and higher-dimensional message passing relative to total inference time, and to do so across diverse mesh types such as tetrahedral and hexahedral discretizations. Crucially, the evaluation must demonstrate that performance improvements are not artifacts of increased parameter counts or uncontrolled inference budgets, but rather stem from efficient exploitation of geometric structure. Authors should address scalability by analyzing how runtime grows with mesh resolution and by proposing approximation strategies—such as reduced quadrature schemes—for expensive element types. Without such analysis, claims of practical utility for engineering applications remain unsubstantiated, as the method may become prohibitive for the fine meshes often required in industrial simulation pipelines.

**Core Evaluation Criteria:**
- **Overhead Quantification**: Does the work report the precise runtime percentage consumed by geometric feature computation (e.g., cell volume, facet area) separately from message passing, across datasets with varying mesh density?
- **Controlled Comparative Efficiency**: Are all models compared under matched parameter budgets, identical hyperparameters, and similar inference-time budgets to ensure that gains are not purchased with disproportionate compute?
- **Scalability and Approximation Strategy**: Does the analysis characterize cost growth with mesh fineness, and are approximation schemes (e.g., Gauss quadrature for hexahedra) provided to mitigate bottlenecks without catastrophic accuracy loss?

---

**Principle 3: Physical Fidelity of Contact and Volumetric Representation Under Sparse Mesh Discretization**

**Definition:**  
A primary motivation for incorporating higher-dimensional geometric elements is to preserve physical correctness when mesh discretization is coarse, where vertex-based methods suffer from contact misdetection and inaccurate integral approximations. This principle evaluates whether the proposed method yields physically plausible boundary representations—specifically, whether facet-based contact detection reduces spurious contact pairs and captures true surface-to-surface interactions that node-based methods miss under identical or comparable probing radii. Furthermore, the method must demonstrate stable propagation of internal physical quantities, such as stress and plastic strain, without the interpenetration or mesh distortion common in sparse-mesh GNN simulators. Reviewers should look for quantitative evidence of valid contact pair ratios, qualitative visualizations of failure modes, and ablation studies linking geometric representational capacity to physical quantity accuracy. The principle ultimately distinguishes methods that achieve lower error metrics through genuine geometric fidelity from those that exploit dataset biases or smoother loss landscapes.

**Core Evaluation Criteria:**
- **Contact Representation Quality**: Does the work quantify valid contact pair detection rates (facet-based versus node-based) and demonstrate that surface-to-surface contact avoids the spurious pairing and missed collisions endemic to point-based detection under sparse meshes?
- **Volumetric Quantity Accuracy**: Does the method show improved prediction of internal physical fields (e.g., stress, strain, PEEQ) that depend on volumetric integration, particularly on the coarsest tested meshes?
- **Qualitative Plausibility**: Are rollout visualizations and error maps provided to confirm absence of interpenetration, element inversion, or unrealistic deformation modes in long-horizon sparse-mesh rollouts?

---

**Principle 4: Controlled Fairness in Cross-Representation Baseline Comparisons for Mesh-Based Simulators**

**Definition:**  
Fair comparison across methods with differing geometric representations is notoriously difficult in physical simulation, as surface-based, volume-based, and hierarchical models operate on fundamentally different graph structures. This principle mandates that authors control for parameter budgets, hyperparameters, and inference-time constraints when comparing against node-based baselines, ensuring that performance differences arise from representational design rather than capacity mismatches. For contact modeling, detection radii must be calibrated so that facet-based and point-based methods evaluate comparable candidate sets, preventing unfair advantages from either inflated contact graphs or missed interactions. Additionally, baseline selection must respect domain scope: methods designed for rigid-body contact or Eulerian fluids should not be evaluated against deformable solid simulators without appropriate adaptation. By enforcing representation-aware fairness, reviewers can confidently attribute empirical gains to the proposed geometric encoding rather than experimental confounders.

**Core Evaluation Criteria:**
- **Matched Experimental Conditions**: Are all models trained with identical hyperparameters and evaluated under matched parameter counts and inference-time budgets?
- **Representation-Aware Contact Calibration**: For contact modeling, are probing radii or detection thresholds adjusted so that node-based and facet-based methods detect comparable numbers of contact candidates, ensuring neither method is advantaged by graph density?
- **Scope-Appropriate Baseline Selection**: Does the work clearly categorize baselines by geometric representational capacity (node-only, face-only, hierarchical) and avoid unfair comparisons with methods designed for fundamentally different physical regimes?

---

**Principle 5: Cross-Regime Generalizability and Extensibility to Non-Volumetric Geometries**

**Definition:**  
While a method may excel on volumetric solids, its scientific impact depends on whether the underlying geometric abstraction generalizes to other mesh element types, geometric shapes, and physical formulations. This principle requires authors to articulate clear adaptation pathways for extending their framework to thin-shell structures, surface-based geometries, or Eulerian formulations, identifying which geometric features require redefinition and which architectural components remain invariant. Evaluation should span diverse mesh types—tetrahedral, hexahedral, and mixed—along with varied geometries such as slender bodies, hollow shells, and plates, to validate that the approach is not overfitted to a narrow class of shapes. Furthermore, since local geometric operators inherently lack global receptive fields, authors must discuss composability with standard long-range interaction mechanisms (e.g., graph pooling, hierarchical clustering) even if such extensions are deferred to future work. A method that convincingly demonstrates broad geometric applicability and modular extensibility offers significantly higher value to the simulation community than one restricted to a single well-characterized setting.

**Core Evaluation Criteria:**
- **Geometric and Mesh-Type Diversity**: Does the evaluation span multiple element types (tetrahedral, hexahedral, mixed) and geometric configurations (slender, hollow, plate-like), or is it limited to a single shape class?
- **Adaptation Pathway Clarity**: Does the work explicitly describe what geometry-aware modifications are required to extend the method to thin-shells without volumetric cells or to Eulerian fixed-mesh formulations?
- **Composability with Long-Range Mechanisms**: Does the paper discuss how the proposed local operator can integrate with pooling or hierarchical strategies to capture long-range interactions, acknowledging this as a necessary extension rather than an ignored limitation?