# 1. Research Background and Existing Pain Points
**Research Background:** 
Flexible deformation and contact of solids are prevalent across a wide range of real-world scenarios, ranging from industrial manufacturing, aeronautical engineering, to nuclear materials. Accurate modeling of these behaviors and their evolution significantly enhances the understanding of these scenarios. Deep learning (DL) has rapidly emerged as a powerful tool for efficient physical simulation, showing great potential in molecular dynamics, fluid simulations, and structural solid deformations. Among these DL-based solvers, graph neural networks (GNNs) have demonstrated superior performance in the domain of solid deformation, due to their ability to handle dynamic unstructured meshes and perform nonlinear regression on graphs. To handle irregular solution domains, existing GNN-based methods input unstructured meshes, where the geometry is discretized into multiple connected simple cells using a regular polyhedron. GNNs abstract mesh vertices into graph nodes to capture internal physical interactions, using edges defined by mesh connectivity. Inter-object contact is typically detected via an interaction radius. Mesh edges serve as pathways for propagating physical quantities, making the mesh structure and its associated graph a central representation of the physical system.

**Existing Pain Points:**
1. Existing GNNs commonly represent meshes with graphs built solely from vertices and edges. These approaches tend to overlook higher-dimensional spatial features, e.g., 2D facets and 3D cells, from the original geometry. 
2. It is challenging for node-based GNNs to accurately capture boundary representations and volumetric characteristics, though this information is critically important for modeling contact interactions and internal physical quantity propagation, particularly under sparse mesh discretization.
3. The accuracy of GNNs deteriorates on sparse meshes, which are commonly used for computation efficiency concerns in practice. As the distance between the points in GNN differs from the actual distance between surfaces, and this deviation worsens under coarser mesh configurations, contact information may be missing without an appropriate detection radius. Increasing the radius may help, but with the cost of computations, there are still no guarantees of complete and accurate contact information.
4. GNNs model internal propagation in approximating integral kernels, which is based on positional vectors and physical variables. However, with coarse meshes, nodes may not be sufficient to adequately sample neighborhood regions, hindering the accurate construction of characteristics of the localized physical field.

# 2. Core Research Motivation and Scientific Questions
**Core Research Motivation:**
The limitations of current methods are mainly due to node-based modeling by only using the vertices, which rely on topological representations that often lose critical spatial features required for physical accuracy. Crucially, in addition to the vertices, the mesh contains a much more comprehensive set of high-dimensional geometric elements, that is, 2D facets and 3D cells. High-dimensional elements like 2D facets and 3D cells could be incorporated to enable the model to better characterize geometric structures within 3D continuous space. With this key incorporation, graph-based contact modeling can explicitly capture boundaries and contact as face-to-face geometries, making approaches more suitable for precise simulations. Additionally, even though in the case of coarse mesh discretization, the model might lead to inaccurate integral approximations based on discretely sampled vertices, volumetric cells could compensate and maintain stable computations by retaining geometric continuity. Furthermore, explicit geometric features can be incorporated into the model to alleviate the burden of implicitly learning geometric patterns.

**Scientific Questions:**
How to explicitly model geometric mesh elements of higher dimension to achieve a more accurate and natural physical simulation? How to establish learnable mappings among 3D cells, 2D facets, and vertices, enabling flexible mutual transformations? How to comprehensively capture boundary representations and volumetric characteristics that are critically important for modeling contact interactions and internal physical quantity propagation under sparse mesh discretization?

# 3. Overall Core Idea and Design Philosophy
**Overall Core Idea:**
Introduce MAVEN, a mesh-aware volumetric encoding network for simulating 3D flexible deformation, which explicitly models geometric mesh elements of higher dimension to achieve a more accurate and natural physical simulation. MAVEN establishes learnable mappings among 3D cells, 2D facets, and vertices, enabling flexible mutual transformations. Explicit geometric features are incorporated into the model to alleviate the burden of implicitly learning geometric patterns.

**Design Philosophy:**
To fully exploit high-dimensional geometric elements in mesh-based neural networks, design a novel framework within an encoder–processor–decoder architecture that explicitly embeds cell and facet elements into the model, thereby enhancing performance under sparse mesh conditions. MAVEN constructs explicit nodes for each geometric element in the mesh, including vertices, facets, and cells. During each processing step, the vertex information is encoded into higher-dimensional elements using learnable position-aware aggregators. The internal interactions and external loads are then handled at the cell level through facet nodes, allowing information to propagate through the mesh via its higher-dimensional structures (cells, facets). This cell-facet co-design enables comprehensive geometric modeling beyond node-based approaches. In MAVEN, cells and facets focus on different types of features. The facet is a pivotal hub for information exchange, where external forces, object contacts, and the propagation of internal physical quantities converge. The cells fully characterize the geometric and volumetric properties of the 3D continuum domains. The adoption of volumetric features of adjacent cells as propagation coefficients for vertices significantly enhances the performance.

# 4. Core Innovation Points (List all innovations in the original text completely)
1. Propose a paradigm that explicitly incorporates high-dimensional geometric elements into the simulation of 3D solid systems. This approach enables the model to capture precise geometric information and maintain stability under sparse discretization conditions.
2. Design MAVEN, a model based on mesh-aware volumetric encoding that captures high-dimensional geometries by explicitly modeling both cell and facet elements. 
3. MAVEN facilitates data transformation between elements through carefully designed transformation functions, and propagates information over a cell-facet graph using a modified two-stage message passing process.
4. Explicit geometric features are incorporated into the model to alleviate the burden of implicitly learning geometric patterns, computing representations of cells and facets from high-dimensional geometric properties such as volume, total surface area, area, and perimeter.
5. Construct aggregation coefficients from each vertex to the element based on the local coordinate system of the element using a position-sensitive geometric aggregator inspired by the shape function in numerical solvers, avoiding severe homogenization of features between vertices and overlooking relative geometric relationships.
6. Establish contact connections directly between the interacting facets instead of constructing edges between vertices, applying a simplified Bounding Volume Hierarchy algorithm to detect all pairs of faces within a collision radius.

# 5. Overview of the Overall Technical Solution
The overall architecture of MAVEN follows the widely adopted encoder-processor-decoder framework. Unlike node-based models, MAVEN additionally models each element in the cell set and facet set as individual nodes participating in message passing, thus enhancing the ability to capture high-dimensional geometric information. 
First, MAVEN performs feature extraction for all node types. The cell and facet nodes are initialized with their geometric characteristics, such as 3D volume, surface area, as well as 2D area and perimeter, while the vertex nodes retain their physical quantities as input features. Facet-to-facet features and external force features are also encoded.
Subsequently, a stack of processors is employed to model physical interactions within and across solid objects. In each processor, the cell and facet nodes update their geometric representations from the nearby vertex nodes via a position-sensitive geometric aggregator. A modified two-stage message passing is then applied to propagate information during a cell-facet graph constructed by the relative geometric relationships between elements. 
Finally, a geometric disaggregator operation distributes the aggregated features back to the vertex nodes, generating smooth intermediate results. The final processor output is subsequently mapped back to the original domain to produce the predicted results.

# 6. Detailed Module Design (If any, complete mechanisms of each layer/sub-module)
**6.1 Problem Definition**
The evolution of a Lagrangian system is initiated by an initial material domain $\Omega_0$, together with a field function $U(0, x)$ that defines the initial physical quantities at each material point $x$. At each time $t \in [0, T]$, the focus is the current physical state $U(t, x)$ of every point $x \in \Omega_0$. The initial material domain $\Omega_0$ is partitioned into a set of regular tetrahedra (or hexahedra) cells $C$, which collectively form the mesh. The collection of vertices and surface facets defines the set of vertices $V$ and the set of facets $F$ of the mesh. Physical field information $u^t_i$ at time $t$ is stored at each vertex $v_i \in V$. The simulation trajectory originates from an initial physical field $u^0$ governed by a discretization mesh $\mathcal{M} = \{C,F ,V\}$. The input of variable-time external forces $\{f^0, f^{\Delta t}, ..., f^{K\Delta t}\}$ drives the progression of the dynamic state, generating the sequence of physical evolution $\{u^0,u^{\Delta t}...,u^{K\Delta t}\}$. The objective is to predict next physical state $u^{(k+1)\Delta t}$ from a history of previous states, considering $\{u^0,u^{(k-1)\Delta t},u^{k\Delta t}, f^{k\Delta t}\}$ as input states.

**6.2 Encoder: Geometry-Informed Feature Encoding**
The MAVEN encoder constructs feature representations for the cell, facet, and vertex nodes while also processing external force conditions and inter-facet contact relationships.
- **Vertex node encoder:** For a given vertex node $v_i \in V$ and its associated physical field quantities $u^t_{v_i}$, apply standard GNN practices to encode quantities into a high-dimensional latent space to derive the vertex node feature $h^0_{v_i}$.
- **Cell and facet node encoder:** Since cells and facets do not possess direct input features, compute their representations from high-dimensional geometric properties. Use volume and total surface area as initial geometric features for each cell, while area and perimeter are used to characterize facet nodes. Incorporate the initial geometric attributes of each element to enhance the model’s awareness of high-dimensional geometric variations over time. Let $\Omega(c_i),\Sigma(c_i)$ be the volume and surface area of cell $c_i$, and $\alpha(f_i), \lambda(f_i)$ be the area and perimeter of facet $f_i$. Generate cell and facet features $h_{c_i}$ and $h_{f_i}$. All $h^0_{v_i}$, $h_{c_i}$, and $h_{f_i}$ are projected to the same latent dimension, which is set to 128 in practice.
- **Facet-to-facet features:** MAVEN establishes contact connections directly between the interacting facets. Apply a simplified Bounding Volume Hierarchy algorithm to detect all pairs of faces within a collision radius $r$. For two contacting facets $f_s$ and $f_r$, the translation equivariant vectors of each face edge include: (1) the relative displacement between the center points $d^F_{rs} = p_r - p_s$ on the two faces, (2) the spanning vectors from the vertices of each face to the center of the other face $d^F_{v_i} = x^s_i - p_s, d^F_{r_i} = x^r_i - p_r$, and (3) the normal unit vectors of the face of the sender and receiver faces $n_s, n_r$. Generate face-to-face features $h_{f_s \rightarrow f_r}$.
- **External force features:** External forces are defined as scripted motions for specific surface vertices, where their next-step positions $x^{t+1}$ are explicitly provided. Define the external force feature for each node $h^S_{v_i}$ as its relative displacement to the next time step, and introduce a flag to indicate whether the node is scripted. Define scripted features on each facet $h^S_{f_i}$ by concatenating motion-related features of all its associated vertex nodes. To ensure translational and permutation invariance, sort the vertices of each facet by their distances to the facet centroid, thereby enforcing a unique representation.

**6.3 Processor**
All features extracted by the encoders are fed into a processor module composed of L stacked layers. In the $l$-th layer, the processor takes the vertex features of the previous layer $h^{l-1}_V$ and applies a geometric aggregator to incorporate the vertex information into the facet and cell nodes to generate geometric features $h^l_C$ and $h^l_F$. Two-stage message passing is used to propagate physical information across high-dimensional elements, where messages are first exchanged on facets and subsequently to cells. Finally, an inverse disaggregation decoder maps the updated cell-level features back to the vertex nodes for residual updates, producing spatially smooth features $h^{l+1}_V$ over domain.
- **Geometric Aggregator:** Update the features of each element by aggregating the features of all vertices that it contains. Inspired by the shape function in numerical solvers, MAVEN constructs aggregation coefficients from each vertex to the element based on the local coordinate system of the element. These coefficients are shared across all processor layers. Let $\vec{d}_{c_i,v_j}$ denote the position vector from the center of the cell $c_i$ to vertices $v_j \in c_i$, similar to $\vec{d}_{f_k,v_l}$. Based on the local coordinate system, employ an MLP to generate a centered normalized vertex aggregation weight $a_{c_i,v_j} \in \mathbb{R}$ for each cell. Similarly, coefficients $a_{f_i,v_0}, \dots, a_{f_i,v_{M-1}}$ for each facet $f_i$ can be derived, where $K$ and $M$ denote the number of associated vertices for each cell and facet, respectively. The coefficients obtained are then utilized to perform weighted aggregation for element feature update. Also sort the vertices within each cell to ensure permutation invariance.
- **Message passing on cell-facet graph:** After extracting features, construct a bipartite cell-facet graph $G = (V_G = \{C,F\}, E_G)$ to explicitly capture the topological and geometric relationships between volumetric elements and their boundary interfaces. $E_G$ contains all pairs $(c_i, f_j)$ for $f_j \in c_i$. MAVEN performs two distinct message passing steps, each dedicated to the facet and cell nodes, respectively.
  - **First stage:** Information is aggregated through facet nodes, which serve as 'edges', bridging not only adjacent cells but also mediating interactions between external forces or contacts and the internal dynamics of the object. Inter-object contact interactions are represented on the facet level, where information from all face-to-face edges is aggregated. To incorporate context from adjacent cells, similarly employ a learnable aggregation scheme with $a_{f_i,c_j}$.
  - **Second stage:** Each cell aggregates information from its associated facets. A similar strategy is adopted, employing symmetric aggregation coefficients $a_{c_i,f_j} = a_{f_i,c_j}$ to combine messages from multiple surfaces.
- **Geometric Disaggregator:** Finally, the high-dimensional geometric information encoded in each cell is retransmitted to its associated vertex nodes using the same symmetric aggregation coefficients $a_{v_i,c_j} = a_{c_j,v_i}$. A residual connection is applied to update the vertex features. This disaggregation at the vertex level facilitates a boundary-aware averaging of cell-level information, contributing to globally smooth predictions across the domain. The next layer uses $h^{l+1}_V$ as the input vertex features.

**6.4 Decoder and Updater**
The model decodes the features of vertices $h^L_V$ using an MLP decoder to predict the velocity $\dot{\hat{x}}^{t+1}$ and the physical quantities $\hat{c}^{t+1}$ for the next state. The next position $\hat{x}^{t+1}$ is estimated by first-order integration. The rollout trajectory can be obtained by applying the simulator to the last prediction iteratively.

# 7. All Mathematical Formulas and Symbol Definitions
**Vertex node encoder:**
$h^0_{v_i} = \mathcal{A}_V(u^t_{v_i})$

**Cell and facet node encoder:**
$h_{c_i} = \mathcal{A}_C(\Omega(c^t_i),\Sigma(c^t_i),\Omega(c^0_i),\Sigma(c^0_i)), \quad h_{f_i} = \mathcal{A}_F(\alpha(f^t_i), \lambda(f^t_i), \alpha(f^0_i), \lambda(f^0_i))$

**Facet-to-facet features:**
$h_{f_s \rightarrow f_r} = \mathcal{A}_{F \leftrightarrow F}([d^F_{rs}, [d^F_{s_j}]_{s_j \in f_s}, [d^F_{r_j}]_{r_j \in f_r}, n_s, n_r])$

**External force features:**
$h^S_{v_i} = \begin{cases} [1, x^{t+1}_{v_i} - x^t_{v_i}], & \text{if } v_i \text{ is scripted} \\ 0, & \text{if } v_i \text{ is not scripted} \end{cases}$
$h^S_{f_i} = \mathcal{A}_S(\text{concat}_{v_j \in f_i}(h^S_{v_j}))$

**Geometric Aggregator coefficients:**
$a_{c_i, v_0}, \dots, a_{c_i, v_{K-1}} = \text{MLP}(\text{concat}_{v \in \{v_0, \dots, v_{K-1}\}}(\vec{d}_{c_i, v})) \quad \{v_0, \dots, v_{K-1}\} \text{ represents } c_i$
$h^l_{c_i} = \mathcal{A}^l_{V \rightarrow C}(h_{c_i}, \sum_{v \in \{v_0, \dots, v_{K-1}\}} a_{c_i, v} h^l_v), \quad h^l_{f_i} = \mathcal{A}^l_{V \rightarrow F}(h_{f_i}, \sum_{v \in \{v_0, \dots, v_{M-1}\}} a_{f_i, v} h^l_v)$

**Message passing on cell-facet graph:**
First stage:
$h^{F \rightarrow F, l}_{f_i} = \sum_{f_r \sim f_i} \mathcal{A}_{F \rightarrow F}(h_{f_s \rightarrow f_r}, h^l_{f_s}), \quad h^{\rightarrow F, l}_{f_i} = \mathcal{A}^l_{\rightarrow F}(h^S_{f_i}, h^{F \rightarrow F, l}_{f_i}, h^l_{f_i}, \sum_{(c_j, f_i) \in E_G} a_{f_i, c_j} h^l_{c_j})$
Second stage:
$h^{\rightarrow C, l}_{c_i} = \mathcal{A}^l_{\rightarrow C}(h^l_{c_i}, \sum_{(c_i, f_j) \in E_G} a_{c_i, f_j} h^{\rightarrow F, l}_{f_j})$

**Geometric Disaggregator:**
$h^{\rightarrow V, l}_{v_i} = \mathcal{A}^l_{\rightarrow V}(h^l_{v_i}, \sum_{v_i \in c_j} a_{v_i, c_j} h^{\rightarrow C, l}_{c_j}), \quad h^{l+1}_{v_i} = h^l_{v_i} + h^{\rightarrow V, l}_{v_i} + \text{FFN}(h^l_{v_i} + h^{\rightarrow V, l}_{v_i})$

**Decoder and Updater:**
$\hat{x}^{t+1} = \dot{\hat{x}}^{t+1} + x^t$

**Training Loss:**
$L = \frac{1}{|V|} \|x^{t+1} - \hat{x}^{t+1}\|^2 + \frac{1}{|V|} \|c^{t+1} - \hat{c}^{t+1}\|^2$

**Symbol Definitions:**
- $\Omega_0$: Initial material domain
- $U(t, x)$: Field function defining initial physical quantities at material point $x$
- $C, F, V$: Set of cells, facets, and vertices of the mesh
- $\mathcal{M}$: Discretization mesh $\{C, F, V\}$
- $u^t_i$: Physical field information at time $t$ at vertex $v_i$
- $\mathcal{A}$: Feature fusion operator (implemented via MLP)
- $h^0_{v_i}, h_{c_i}, h_{f_i}$: Vertex node feature, cell node feature, facet node feature
- $\Omega(c_i), \Sigma(c_i)$: Volume and surface area of cell $c_i$
- $\alpha(f_i), \lambda(f_i)$: Area and perimeter of facet $f_i$
- $d^F_{rs}$: Relative displacement between center points of two faces
- $d^F_{s_j}, d^F_{r_j}$: Spanning vectors from vertices to center of other face
- $n_s, n_r$: Normal unit vectors of sender and receiver faces
- $h_{f_s \rightarrow f_r}$: Face-to-face features
- $h^S_{v_i}, h^S_{f_i}$: External force features for vertex and facet
- $\vec{d}_{c_i,v_j}$: Position vector from center of cell $c_i$ to vertex $v_j$
- $a_{c_i, v_j}$: Centered normalized vertex aggregation weight for cell
- $a_{f_i, v_j}$: Centered normalized vertex aggregation weight for facet
- $K, M$: Number of associated vertices for each cell and facet
- $G=(V_G, E_G)$: Bipartite cell-facet graph
- FFN($\cdot$): Feed-forward network used in the Transformer

# 8. Algorithm Pseudocode (If any, paste the pseudocode from the paper exactly as it is)
The paper does not provide explicit algorithm pseudocode. The operational process and logical iterative steps are fully described in the Detailed Module Design section and mathematical formulas above.