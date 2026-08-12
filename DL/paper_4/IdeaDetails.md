### 1. Research Background and Existing Pain Points

**Research Background:**
Topological neural networks (TNNs) have recently emerged as a prominent class of models for learning on topological domains, such as hypergraphs and simplicial complexes, representing the new frontier for relational learning. Akin to graph neural networks (GNNs), typical TNNs employ message-passing layers where each element of the input updates its representation based on its topological neighbors. TNNs generalize convolution-like operations on graphs to higher-order relational objects. The primary application of TNNs has been to enhance the capabilities of graph-based models, particularly in terms of expressivity. In this context, the input graphs must first be transformed to the domain on which a TNN operates — a process known as lifting. Lifting methods explore graph connectivity and features to create higher-order relational structures. For instance, clique lifting produces a simplicial complex by leveraging cliques, while cycle lifting detects cycles to create a cell complex. 

**Existing Pain Points:**
1. **Unsupervised and Task-Agnostic Nature:** Despite the high impact of the lifting operation on TNNs, most lifting methods are not supervised and thus not informed by the task at hand, which may lead to suboptimal architectures. To date, differentiable lifting has only been explored in the limited context of cell complexes.
2. **Data-Dependent Impact:** The optimal choice of topological domain and lifting procedure for each task is non-obvious, and its impact on performance is highly data-dependent. Lifting to different domains can lead to disparate performances, and performances of liftings to the same domain vary greatly.
3. **Static and Rigid Structures:** Current static liftings (like connectivity-based or feature-based ones) are predefined and cannot adapt the topological structure based on the learning signal, limiting their effectiveness across diverse downstream tasks.

### 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:**
To circumvent the issue of suboptimal static and unsupervised graph liftings, there is a need for a general framework that can learn graph liftings to various topological domains (hypergraphs, cellular complexes, and simplicial complexes) in an end-to-end fashion. By making the lifting process task-informed and differentiable, the model can adaptively identify and parameterize distributions over candidate higher-order cells, thereby constructing optimal topological representations for downstream learning tasks.

**Scientific Questions:**
How can we design a general, differentiable lifting framework that is applicable to various topological domains, allows for end-to-end learning, and leverages learned vertex-level latent representations to probabilistically sample and include candidate higher-order cells of adaptive sizes while respecting the hierarchical constraints of the target domains?

### 3. Overall Core Idea and Design Philosophy

The core idea is ∂lift (DiffLift), a general framework for learning graph lifting functions. The design philosophy centers on a probabilistic approach to sample candidate cells of adaptive sizes. Specifically, ∂lift parameterizes distributions over cells using node embeddings derived from arbitrary graph models (e.g., GNNs or Graph Transformers). For each candidate cell, it computes an embedding and uses a multilayer perceptron (MLP) to estimate the probability of accepting or rejecting the cell (whether it should be included in the output structure). To model the typical hierarchical structure of topological objects, ∂lift proposes an iterative sampling procedure where cells are generated in increasing order of dimensionality: samples of dimension $i$ are used to inform the sampling of dimension $(i+1)$-cells. The model is trained end-to-end using the straight-through estimator to propagate gradients through the sampling steps.

### 4. Core Innovation Points

1. **General Differentiable Lifting Framework:** ∂lift is the most general form of learnable lifting proposed so far, applicable to various domains including hypergraphs, simplicial complexes, cell complexes, and combinatorial complexes, unlike prior work limited to cell complexes only.
2. **Probabilistic Sampling with Adaptive Sizes:** The framework leverages learned vertex-level latent representations to parameterize distributions over candidate higher-order cells. It allows for adaptive cell/hyperedge sizes by sampling neighborhood sizes ($k_v$) according to a probability distribution parameterized by node embeddings.
3. **Iterative Hierarchical Sampling Procedure:** To respect the hierarchical structure of topological objects, ∂lift generates cells iteratively in increasing order of dimensionality. Lower-dimensional samples inform the generation of higher-dimensional candidates (e.g., using cycle bases of dimension $D-1$ to form dimension $D$ cells).
4. **End-to-End Learnability with Straight-Through Estimator:** The entire lifting process is learned in an end-to-end fashion, utilizing the straight-through estimator to propagate gradients through the Bernoulli sampling of cell acceptances, seamlessly integrating into standard TNN pipelines without requiring auxiliary reward-based losses.
5. **Broad Empirical Superiority:** ∂lift consistently outperforms unsupervised lifting methods across multiple benchmarks and TNN architectures, yielding performance gains of up to 45% over static liftings.

### 5. Overview of the Overall Technical Solution

The overall technical solution follows a pipeline: Input Graph $\rightarrow$ GNN $\rightarrow$ Hierarchical Cell Sampler $\rightarrow$ TNN. 
1. An attributed graph is fed into an arbitrary GNN to compute node embeddings.
2. The (Hierarchical) Cell Sampler uses these embeddings to select cells/hyperedges. Cell-level embeddings run through MLPs responsible for returning acceptance probabilities.
3. For hierarchical domains (e.g., cell complexes), cells are generated in increasing dimensionality.
4. From the accepted cells, a relational topological object is formed.
5. This lifted object is sent to an off-the-shelf TNN for graph/node-level predictions.

### 6. Detailed Module Design

#### 6.1 Background Definitions and Neighborhood Structures
*   **Graphs and hypergraphs:** Undirected graph $G = (V,E)$. Neighbors $N_G(v) = \{u \in V : \{v, u\} \in E\}$. Hypergraph on $V$ is $(V,K)$, $K \subseteq 2^V \setminus \emptyset$.
*   **Simplicial complexes:** Abstract simplicial complex (ASC) over $V$ is a set $K$ of subsets of $V$ closed under taking subsets. Dimension of simplex is cardinality minus 1. Boundary relation $\tau \prec \sigma$ iff $\tau \subset \sigma$ and no $\delta$ such that $\tau \subset \delta \subset \sigma$.
*   **Cell complexes:** Regular cell complex with partition $\{X_\sigma\}_{\sigma \in P_X}$ satisfying specific topological conditions, imposing a poset structure $\tau \le \sigma \iff X_\tau \subseteq X_\sigma$. Boundary relation $\sigma \prec \tau$ iff $\sigma < \tau$ and no cell $\delta$ such that $\sigma < \delta < \tau$.
*   **Combinatorial complexes (CC):** Tuple $(V,K, rk)$ where $rk: K \to \mathbb{Z}_{\ge 0}$ s.t. (1) For all $v \in V, \{v\} \in K$; (2) For all $\sigma, \sigma' \in K, \sigma \subseteq \sigma' \implies rk(\sigma) \le rk(\sigma')$.
*   **Neighborhood structures:**
    *   Boundary and co-boundary: $N_B(\sigma) = \{\tau : \tau \prec \sigma\}$ and $N_C(\sigma) = \{\tau : \sigma \prec \tau\}$
    *   Upper/lower adjacency: $N_\uparrow(\sigma)=\{\tau : \exists \delta \text{ st } \tau \prec \delta, \sigma \prec \delta\}$ and $N_\downarrow(\sigma)=\{\tau : \exists \delta \text{ st } \delta \prec \tau, \delta \prec \sigma\}$
*   **Topological neural networks (TNNs):** Message-passing TNN recursively computes:
$$m^\ell_{i,\sigma} = \begin{cases} \{\{\phi^{\ell,i}(h^\ell_\tau, h^\ell_\sigma, h^\ell_\delta) : \tau \in N_i(\sigma), \delta \in N_B(\sigma, \tau)\}\}, & \text{if } N_i = N_\downarrow \\ \{\{\phi^{\ell,i}(h^\ell_\tau, h^\ell_\sigma, h^\ell_\delta) : \tau \in N_i(\sigma), \delta \in N_C(\sigma, \tau)\}\}, & \text{if } N_i = N_\uparrow \\ \{\{\phi^{\ell,i}(h^\ell_\tau, h^\ell_\sigma) : \tau \in N_i(\sigma)\}\}, & \text{otherwise.} \end{cases}$$
$$h^{\ell+1}_\sigma = \phi\left( h^\ell_\sigma, \underset{i}{\otimes} \text{Agg}^\ell\left( m^\ell_{i,\sigma} \right) \right)$$

#### 6.2 General Formulation of $\partial$lift
Lifting consists of determining which higher-order cells should be added to an input graph $G$, satisfying domain constraints.
*   **Step 1:** Compute node embeddings using arbitrary GNN. Set $D = 1$.
*   **Step 2:** Elicit candidate cells $\mathcal{C} \subseteq 2^V$ of dimension $D$. Compute cell embedding $z_\mathcal{C} = \oplus_{v \in \mathcal{C}} z_v$.
*   **Step 3:** Apply neural network $\phi$ (e.g., MLP) defining acceptance probability $\phi(z_\mathcal{C})$. Draw sample $y_\mathcal{C} \sim Ber(\phi(z_\mathcal{C}))$.
*   **Step 4:** If $D = D_{max}$ halt; else $D \leftarrow D + 1$.

#### 6.3 Graph-to-Hypergraph Lifting Module
*   **Step 2 (Elicit candidates):** For each node $v$, define candidate hyperedge $C(v)$ using $k_v$ nearest neighbors in embedding space:
$$C(v) = \{S \subset V : |S| = k_v \text{ and } w \notin S \implies dist(z_w, z_v) \ge \max_{u \in S} dist(z_u, z_v)\}$$
Sample $k_v$ according to $\pi_v \propto \exp \circ MLP(z_v)$ and draw $k_v \sim Categorical(\pi_v)$.
*   **Step 3 (Accept/reject):** 
$$b_v \sim Ber(\Psi(\{\{ z_u : u \in C(v)\}\}))$$
*   **Feature lifting:**
$$x_{C(v)} = \frac{1}{k_v} \sum_{u \in C(v)} x_u, \quad \forall v \text{ such that } b_v = 1$$

#### 6.4 Graph-to-Cell-Complex Lifting Module
**Case D = 1: Learning edges**
*   **Step 2:** Sample $k_v$ and define $C(v) \subseteq V \setminus \{v\}$ containing $k_v$ nearest neighbors.
*   **Step 3:** $b_{v,v'} \sim Ber (\Psi(\{\{z_v, z_{v'}\}\}))$.
*   **Resulting complex:**
$$K_1 = V \cup E \cup \{\{v, v'\} : v \in V, v' \in C(v) \text{ with } b_{v,v'} = 1\}$$

**Case D $\ge$ 2: Learning D-cells**
*   **Step 2:** Let $K_{D-1}$ be complex at end of iteration $D-1$. Select a basis for $(D-1)$-set of cycles $Z_{D-1}(K_{D-1})$ in $K_{D-1}$ to serve as candidate cells.
*   **Step 3:** Let $\mathcal{C}$ be candidate $(D-1)$-cycles. Acceptance uses DeepSet model over $\{\{z_v\}\}_{v \in C}$.
$$K_D = K_{D-1} \cup \{C \in \mathcal{C} : b_C = 1\}$$

#### 6.5 Graph-to-Simplicial-Complex Lifting Module
**Case D $\ge$ 2: Learning D-simplices**
*   **Step 2:** Run static D-clique lifting on $K_1$. Filter out elements where lower-order cliques are not in $K_{D-1}$:
$$\mathcal{C} = \{ C \in \mathcal{C}' | S \in K_{D-1} \text{ for all } S \subset C \}$$
*   **Step 3:** Apply DeepSet over embeddings $\{\{z_v\}\}_{v \in C}$, sample Bernoulli variables.
$$K_D = K_{D-1} \cup \{C \in \mathcal{C} : b_C = 1\}$$

#### 6.6 Graph-to-Combinatorial-Complex Lifting Module
Run $\partial$lift to either cell or simplicial complexes $K$. Then employ (in parallel) $\partial$lift to a hypergraph $H$ where edges/hyperedges have rank 1. Prune sampled hyperedges to include only those that are not supersets of any cell of rank greater than 1 in $K$.
$$\text{Resulting CC} = \{h \in H : \not\exists \sigma \in K \text{ s.t. } \sigma \subseteq h\} \cup K$$

### 7. All Mathematical Formulas and Symbol Definitions

*   $G = (V,E)$: Undirected graph.
*   $N_G(v)$: Neighbors of node $v$.
*   $(V,K)$: Hypergraph, where $K \subseteq 2^V \setminus \emptyset$.
*   $\tau \prec \sigma$: Boundary relation ( $\tau$ on boundary of $\sigma$).
*   $N_B(\sigma) = \{\tau : \tau \prec \sigma\}$: Boundary neighborhood.
*   $N_C(\sigma) = \{\tau : \sigma \prec \tau\}$: Co-boundary neighborhood.
*   $N_\uparrow(\sigma)=\{\tau : \exists \delta \text{ st } \tau \prec \delta, \sigma \prec \delta\}$: Upper adjacency.
*   $N_\downarrow(\sigma)=\{\tau : \exists \delta \text{ st } \delta \prec \tau, \delta \prec \sigma\}$: Lower adjacency.
*   $rk: K \to \mathbb{Z}_{\ge 0}$: Ranking function for Combinatorial Complexes.
*   $h^\ell_\sigma$: Embedding of $\sigma$ at layer $\ell$.
*   $m^\ell_{i,\sigma}$: Message computed at layer $\ell$ for neighborhood $i$ and cell $\sigma$.
*   $C(v)$: Candidate hyperedge/cell for node $v$.
*   $k_v$: Sampled neighborhood size for node $v$.
*   $\pi_v \propto \exp \circ MLP(z_v)$: Probability vector for sampling $k_v$.
*   $b_v, b_{v,v'}, b_C$: Bernoulli variables for acceptance.
*   $\Psi(\cdot)$: Learned order-invariant function mapping multisets to $(0,1)$.
*   $K_D$: Cell complex at dimension $D$.
*   $C_n(K)$: Space of n-chains with $\mathbb{Z}/2\mathbb{Z}$-vector space structure.
*   $\partial_n: C_n(K) \to C_{n-1}(K)$: Boundary linear map.
*   $Z_n(K) = \ker(\partial_n)$: n-cycles of $K$.
*   **Chain operations**:
$$c + c' = \sum_{\sigma} (\epsilon_\sigma + \epsilon'_\sigma)\sigma$$
$$\lambda c = \sum_{\sigma} (\lambda \epsilon_\sigma)\sigma$$
*   **Boundary of simplex**:
$$\partial_n \sigma = \sum_{\tau \subset \sigma : |\tau| = |\sigma| - 1} \tau$$
*   **Boundary operator extension**:
$$\partial_n c = \partial_n \sum_{\sigma \in K(n)} \epsilon_\sigma \sigma = \sum_{\sigma \in K(n)} \epsilon_\sigma \partial_n \sigma$$
*   **Cycle definition**:
$$Z_n(K) = \{c \in C_n(K) : \partial_n c = 0\}$$
*   **Clique lifting**:
$$\text{lift}_{\text{clique}, k}(G) = V(G) \cup_{i=2}^k \text{Cl}_i(G)$$
*   **Cycle lifting**:
$$\text{lift}_{\text{cycle}}(G) = V(G) \cup E(G) \cup \{V(b) : b \in B(G)\}$$
*   **k-hop lifting**:
$$N_k[v] = \{u \in V(G) : dist(u, v) \le k\}$$
$$\text{lift}_{\text{k-hop}}(G) = \{N_k[v] : v \in V(G)\}$$

### 8. Algorithm Pseudocode

**∂lift: General recipe for differentiable graph liftings**

**Input:** Attributed graph $G = (V,E, x)$, target domain $\mathcal{T}$, and maximum dimension $D_{max}$.

**Step 1:** Compute node embeddings. Use an arbitrary GNN to compute a vector representation (embedding) $z_v$ for each node $v \in V$. This GNN component can be either a pre-trained model or learned end-to-end. Set the current domain dimension to $D = 1$.

**Step 2:** Elicit candidate cells. Given the node embeddings $\{z_v\}_{v \in V}$, define a set of candidate cells $\mathcal{C} \subseteq 2^V$ of dimension $D$. For each cell $C \in \mathcal{C}$, compute an embedding $z_C = \oplus_{v \in C} z_v$, where $\oplus$ is an arbitrary permutation-invariant aggregation function. Note that the exact procedure for defining candidate cells depends on the target domain $\mathcal{T}$, as candidates must respect possible hierarchical constraints.

**Step 3:** Accept/reject candidate cells. Apply a neural network $\phi$ (e.g., an MLP) that defines an acceptance probability $\phi(z_C)$ for each candidate cell $C$. Finally, draw a sample $y_C$ from a Bernoulli distribution with parameter $\phi(z_C)$ indicating whether cell $C$ is accepted or not. The resulting domain is then given by $V \cup E \cup \{C \in \mathcal{C} : y_C = 1 \text{ with } y_C \sim Ber(\phi(z_C))\}$.

**Step 4:** Termination check. If $D = D_{max}$, halt; otherwise, $D \leftarrow D + 1$ and return to Step 2.