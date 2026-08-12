### 1. Research Background and Existing Pain Points

In graph representation learning, capturing a graph’s natural symmetries (i.e., isomorphism invariance) is essential for learning and generalization. Existing approaches to achieving invariance and expressivity face fundamental limitations:

1.  **Limitations of Message-Passing Neural Networks (MPNNs):** MPNNs achieve architectural invariance by iteratively aggregating neighbor embeddings. However, they are provably equivalent in expressive power to the 1-dimensional Weisfeiler–Leman test (1-WL), and suffer from oversmoothing and oversquashing, making them fundamentally limited in expressive power.
2.  **Limitations of Random Walk Neural Networks (RWNNs):** RWNNs sample walks as input to powerful sequence models, overcoming MPNN limitations but incurring potentially prohibitive sampling costs when training on large datasets.
3.  **Limitations of Sequence-Based Canonicalization:** Existing graph canonicalization approaches map each graph to a unique representative sequence, allowing expressive, non-invariant models to operate on a fixed, invariant input. However, they suffer from two critical pain points:
    *   **Distance Distortion (Stretch and Contraction):** Flattening a graph into a single sequence distorts graph distances. For example, in an $n$-node star graph $S_n$, each leaf node has distance 1 to the center node in the graph, but leaf nodes in the sequence necessarily have distance $O(n)$ to the center node (stretch). Moreover, while leaves have distance 2 to each other in $S_n$, certain leaves have distance 1 in the sequence (contraction).
    *   **Expressive Limitations:** The canonical graph neural network relying on a single sequence is only as expressive as its node labeler. Even when the downstream sequence model is universal, information lost at the labeling stage cannot be recovered, discarding the benefits of using powerful downstream models.

### 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:** To design a graph canonicalization method that preserves graph distances significantly better than sequences, increases expressivity beyond the initial node labeler by using multiple canonical representatives, and maintains invariance (probabilistic or deterministic) without incurring the expensive sampling costs of random walk methods.

**Scientific Questions:**
1.  How can we formally quantify the distance distortion introduced by sequence-based canonicalizations, and what alternative representations can minimize this distortion?
2.  How can we overcome the expressivity ceiling of single-sequence canonical GNNs, which are bounded by their node labelers, without sacrificing isomorphism invariance?
3.  How can we construct a canonical representation that efficiently covers the entire graph structure (specifically all edges) on sparse graphs to guarantee high expressivity and universality?

### 3. Overall Core Idea and Design Philosophy

The core idea is to replace the single canonical sequence with a **canonical spanning tree cover**. Instead of flattening the graph into a sequence, the graph is represented by a set of canonical spanning trees. Tree covers better preserve graph distances in comparison to sequences, and on sparse graphs, the cover recovers all edges with a logarithmic number of trees in the graph size. Each tree in the cover is processed with an expressive recurrent tree encoder augmented with message passing on residual edges. Aggregating over the cover yields an invariant representation. The design philosophy leverages coverage-aware edge label refinement to ensure the tree cover collectively captures the full graph structure, thus bypassing the expressivity limits of the initial node labeler while maintaining strict or probabilistic invariance.

### 4. Core Innovation Points

1.  **Theoretical Characterization of Sequence Canonicalization Limitations:** Formal quantification that sequence canonicalization distorts graph structure (stretch/contraction) bounded by graph bandwidth, and the proof that single-sequence canonical GNNs are only as expressive as their node labeler.
2.  **Canonical Tree Cover Neural Networks (CTNNs):** A new canonicalization method that constructs a canonical spanning tree cover via minimum spanning tree extraction and coverage-aware edge label refinement, replacing single sequences with a set of trees.
3.  **Coverage-Aware Edge Label Refinement:** An iterative mechanism that updates edge weights by penalizing edges already selected in previous trees, ensuring that the union of spanning trees covers all edges with a logarithmic number of trees on sparse graphs.
4.  **TreeMPNN Architecture:** A hybrid encoder that processes each canonical tree with a recurrent tree encoder (Tree-LSTM) while applying message passing on the residual non-tree edges to capture local connectivity missed by the individual spanning tree.
5.  **Theoretical Guarantees for CTNNs:** Proofs that CTNNs produce invariant graph representations (probabilistic or deterministic), preserve graph distance information with lower expected distortion, exceed the expressivity of sequence-based canonical GNNs and MPNNs, and achieve universality on invariant graph functions when paired with canonical node labelers and universal tree encoders.

### 5. Overview of the Overall Technical Solution

The CTNN pipeline operates as follows:
1.  **Initialization:** Assign an isomorphism-invariant node label $\pi_V$ (e.g., degree) to all nodes. Initialize edge weights based on the sum of node labels.
2.  **Canonical Tree Cover Construction:** Iteratively construct $K$ spanning trees. In each iteration $k$, extract a Minimum Spanning Tree (MST) using the current edge weights and random tie-breaking. Select a canonical root (tree center). Update the edge weights by adding a penalty $\tau$ to edges included in the current tree to bias future trees toward uncovered edges.
3.  **Tree Encoding with Residual Message Passing:** For each extracted tree $T^{(k)}$, compute node representations using a TreeMPNN. This involves a bidirectional recurrent tree encoder (Tree-LSTM) operating on the tree structure, combined with an MPNN (e.g., GIN) operating on the residual graph (edges not in $T^{(k)}$).
4.  **Cover Aggregation:** Aggregate the representations across all $K$ trees using a permutation-invariant aggregator (e.g., SUM) to produce the final invariant graph representation.

### 6. Detailed Module Design

**Module 1: Node Labeling and Edge Weight Initialization**
Given a graph $G=(V,E,X)$, a node labeler $\pi_V: V \to \mathbb{R}$ is applied. The initial edge weights $\pi_E^{(0)}$ are set to the negative sum of the endpoint node labels to prioritize edges incident to high-degree nodes.
$$\pi_E^{(0)}(e) = -(\pi_V(e_u) + \pi_V(e_v))$$

**Module 2: Coverage-Aware MST Extraction and Root Selection**
For iteration $k \in \{0, \dots, K-1\}$, an MST is extracted using Kruskal's algorithm with lexicographic keys $(w_e^{(k)}, \zeta(e))$, where $\zeta(e)$ are i.i.d. continuous tie-breakers. The root of the tree $r^{(k)}$ is chosen as the tree center via a two-BFS routine: one BFS finds an endpoint of a longest path, a second BFS from that endpoint finds the opposite endpoint, and the center(s) of this path serve as the root.

**Module 3: Edge Weight Update (Penalization)**
To promote edge coverage across the set, the weights are updated by penalizing edges used in the last tree $T^{(k)}$ with hyperparameter $\tau$:
$$\pi_E^{(k+1)}(e) = \pi_E^{(k)}(e) + \tau 1\{e \in T^{(k)}\}$$

**Module 4: TreeMPNN Encoder**
For each tree $T^{(k)}$, the residual graph is defined as $G \setminus T^{(k)} := (V, E \setminus E(T^{(k)}))$. Let $f_{tree}$ be a recurrent tree encoder (Bidirectional Child-Sum Tree-LSTM) and $f_{MPNN}$ an MPNN. For each node $i \in V$:
$$f_{TreeMPNN}(T^{(k)})_i = f_{tree}(T^{(k)})_i + f_{MPNN}(G \setminus T^{(k)})_i$$
The bidirectional Tree-LSTM consists of a bottom-up pass (leaves to root) and a top-down pass (root to leaves).
*   **Bottom-up pass:** Aggregates child states using scatter_add for the child-sum aggregator. Edgewise forget gates and summed transformed cell contributions are computed. Node-level gates (input, output, update) are applied.
*   **Top-down pass:** Propagates from root to leaves. Each node reads its parent’s state via index_select and applies batched gating. The output feature per node is the concatenation of the bottom-up and top-down hidden states.

**Module 5: Cover Aggregation**
The final graph representation is obtained by aggregating across the set of trees with a permutation-invariant operator $f_{agg}$:
$$f_{CTNN}(G) := f_{agg}(\{f_{TreeMPNN}(T^{(k)}) : T^{(k)} = C_{tree}(G, \pi_E^{(k)}), k = 0, \dots, K-1\})$$

### 7. All Mathematical Formulas and Symbol Definitions

**Notation:**
*   $G = (V, E, X)$: Undirected graph with $n = |V|$ nodes, $m = |E|$ edges, and node features $X \in \mathbb{R}^{n \times d}$.
*   $x_v$: The $v$-th row of $X$.
*   $N(v) = \{u \in V : (u, v) \in E\}$: Neighborhood of $v$.
*   $deg(v) = |N(v)|$: Degree of $v$.
*   $d_G(u, v)$: Shortest path distance in $G$.
*   $T = (V, E, X, r)$: Rooted tree with root $r \in V$.
*   $p(v)$: Unique parent of non-root node $v$.
*   $C(v) = \{u \in V : p(u) = v\}$: Children of $v$.

**MPNN Update:**
$$f_{MPNN}(G)_i = f_{agg}(\{x_j : j \in \hat{N}(i)\})$$
where $\hat{N}(i)$ is the self-loop augmented neighborhood and $f_{agg}$ is permutation-invariant.

**Preorder on Expressivity:**
$$f_2 \preceq f_1 \iff \forall G, H : f_1(G) = f_1(H) \Rightarrow f_2(G) = f_2(H)$$
$f_2 \prec f_1$ if strict, $f_1 \simeq f_2$ if both hold.

**General Sequence-Based Canonical GNN:**
$$f_{CanSeq}(G) = f_{seq}(C_{seq}(G, \pi_V))$$

**Recurrent Sequence Model:**
$$h_t = \Phi(h_{t-1}, x_t), \text{ for } t = 1, \ldots, T$$

**Recurrent Tree Model:**
$$h_v = f_{agg}(\{\Phi(h_c, x_v) | c \in C(v)\}) \text{ for } \ell = L, \ldots, 0 \text{ and all } v \text{ with } d_T(v, r) = \ell$$

**Distortion Definition:**
A mapping $f: (X, d_X) \to (Y, d_Y)$ has distortion $D \ge 1$ if $\exists r > 0$ such that $\forall x, y \in X$:
$$r d_X(x, y) \le d_Y(f(x), f(y)) \le D r d_X(x, y)$$

**Graph Bandwidth:**
$$\phi(G) = \min_\sigma \max_{(u,v) \in E} |\sigma(u) - \sigma(v)|$$

**Sequence Distortion Bound:**
$$\phi(G) \le D_{seq}$$

**Expressive Limitation Proposition:**
$$f_{CanSeq} \simeq \pi_V$$

**Edge Weight Initialization:**
$$\pi_E^{(0)}(e) = -(\pi_V(e_u) + \pi_V(e_v))$$

**Edge Weight Update:**
$$\pi_E^{(k+1)}(e) = \pi_E^{(k)}(e) + \tau 1\{e \in T^{(k)}\}$$

**TreeMPNN Update:**
$$f_{TreeMPNN}(T^{(k)})_i = f_{tree}(T^{(k)})_i + f_{MPNN}(G \setminus T^{(k)})_i$$

**CTNN Aggregation:**
$$f_{CTNN}(G) := f_{agg}(\{f_{TreeMPNN}(T^{(k)}) : T^{(k)} = C_{tree}(G, \pi_E^{(k)}), k = 0, \ldots, K-1\})$$

**Probabilistic Invariance:**
$$f_{CTNN}(G) \stackrel{d}{=} f_{CTNN}(g \cdot G) \text{ for all } g \in S_n$$
$$\Phi(G) := \mathbb{E}[f_{CTNN}(G)] \text{ is an invariant function satisfying } \Phi(G) = \Phi(g \cdot G)$$

**Expected Distortion Definition:**
$$r d_X(x, y) \le \mathbb{E}_{\rho \sim \mu}[\rho(x, y)] \le D r d_X(x, y)$$

**UST Expected Distortion Bound:**
$$D_{UST} = \max_{u,v} \mathbb{E}[d_T(u, v)] / d_G(u, v), \quad \mathbb{E}[d_T(u, v)] \le H(u, v) + H(v, u) / 2$$

**Logarithmic Spanning-Tree Cover Lemma:**
If $K \ge \Upsilon(G) \ln m$ iterations, the union of the MSTs covers all edges:
$$\bigcup_{k=0}^{K-1} E(T^{(k)}) = E$$

**Expressivity Guarantee:**
$$f_{MPNN} \prec f_{CanTree}$$

### 8. Algorithm Pseudocode

**Algorithm 1: BUILDCANONICALTREECOVER: iterative MST cover with root selection**
Input: Graph $G = (V,E)$; node labeler $\pi_V : V \to \mathbb{R}$; iterations $K$; step $\tau > 0$; tiny $\epsilon > 0$
Output: Tree cover $\mathcal{T} = \{T^{(k)}\}_{k=0}^{K'-1}$ and roots $\mathcal{R} = \{r^{(k)}\}_{k=0}^{K'-1}$
(Initialization)
foreach $e = \{u, v\} \in E$ do
    $w_e^{(0)} \leftarrow -(\pi_V(u) + \pi_V(v))$ /* base edge weights $\pi_E^{(0)}$ */
Draw i.i.d. tie–breakers $\zeta : E \to (0, 1)$ (fixed across rounds)
$S_0 \leftarrow \emptyset$ /* covered edges so far */
$\mathcal{T} \leftarrow \emptyset, \mathcal{R} \leftarrow \emptyset$
for $k = 0$ to $K - 1$ do
    // 1) Minimum spanning tree with lexicographic keys
    $T^{(k)} \leftarrow \text{KRUSKALMST}(G, e \mapsto (w_e^{(k)}, \zeta(e)))$
    $\mathcal{T} \leftarrow \mathcal{T} \cup \{T^{(k)}\}$; $S_{k+1} \leftarrow S_k \cup E(T^{(k)})$
    // 2) Canonical root: tree center via two BFS passes
    $r^{(k)} \leftarrow \text{TREECENTER}(T^{(k)}, \pi_V)$
    $\mathcal{R} \leftarrow \mathcal{R} \cup \{r^{(k)}\}$
    // 3) Edge-weight update (penalize edges just used)
    foreach $e \in E$ do
        if $e \in E(T^{(k)})$ then $w_e^{(k+1)} \leftarrow w_e^{(k)} + \tau$
        else $w_e^{(k+1)} \leftarrow w_e^{(k)}$
    if $|S_{k+1}| = |E|$ then break
return $(\mathcal{T}, \mathcal{R})$

**Algorithm 2: KRUSKALMST with exchangeable tie–breakers**
Input: Undirected graph $G = (V,E)$; base edge weights $w : E \to \mathbb{R}$; tie–breakers $\zeta : E \to (0, 1)$ i.i.d.
Output: A spanning tree $T$ of $G$
$T \leftarrow \emptyset$; initialize disjoint–set $D$ with MAKESET($v$) for all $v \in V$
if $E = \emptyset$ then return $T$
// I.i.d. continuous tie–breakers $\zeta$ make keys distinct w.p. 1.
Sort edges $E$ by nondecreasing key $k(e) := (w(e), \zeta(e))$
// Union–find tracks connected components of the partial forest.
for $e = \{u, v\}$ in $E$ in the above order do
    if FINDD($u$) $\neq$ FINDD($v$) then
        $T \leftarrow T \cup \{e\}$; UNIOND($u, v$)
    if $|T| = |V| - 1$ then return $T$
return $T$

**Algorithm 3: TREECENTER: root selection by two BFS passes**
Input: Tree $T = (V_T, E_T)$; tie–breaker ranking on vertices (e.g., $(\pi_V, \text{ID})$)
Output: Root $r \in V_T$ (a center of $T$)
Pick canonical start $s \in V_T$ (minimizing the tie–breaker);
$u \leftarrow \text{BFS\_FARTHEST}(T, s)$; $v \leftarrow \text{BFS\_FARTHEST}(T, u)$;
$P \leftarrow$ unique path from $u$ to $v$ in $T$;
if $|P|$ odd then $r \leftarrow$ middle vertex of $P$
else $r \leftarrow$ the nearer of the two middle vertices under the tie–breaker
return $r$

**Algorithm 4: BITREELSTMFORWARD: Bidirectional child–sum Tree–LSTM forward pass**
Input : $x \in \mathbb{R}^{N \times D}$ node features; rooted tree $T = (V, E_T, r)$ in COO form with (row, col) = (parent, child); arrays parent[$v$], depth[$v$] $\in \{0, \ldots, L\}$
Output : $h \in \mathbb{R}^{N \times 2H}$ : concat. of bottom–up and top–down hidden states
Parameters: bottom–up $W_{iou} \in \mathbb{R}^{D \times 3H}$, $U_{iou} \in \mathbb{R}^{H \times 3H}$, $W_f \in \mathbb{R}^{D \times H}$, $U_f \in \mathbb{R}^{H \times H}$; top–down $W_{iou}^\downarrow, U_{iou}^\downarrow, W_f^\downarrow, U_f^\downarrow$ of matching shapes.
Init: $h^\uparrow \leftarrow 0_{N \times H}$, $c^\uparrow \leftarrow 0_{N \times H}$, $h^\downarrow \leftarrow 0_{N \times H}$, $c^\downarrow \leftarrow 0_{N \times H}$.
Bucket nodes by depth: $V_\ell \leftarrow \{v \in V : \text{depth}[v] = \ell\}$ for $\ell = 0, \ldots, L$.
/* Bottom–up pass (children $\to$ parent): process parents from deepest to root. */
for $\ell = L$ to $0$ do
    $P \leftarrow V_\ell$ // parents at depth $\ell$
    $E_\ell \leftarrow \{(u \leftarrow v) \in E_T : u \in P\}$ // edges with parent at depth $\ell$
    // Aggregate child states with scatter_add (child–sum $f_{agg} = \sum$)
    $h_{\text{sum}} \leftarrow 0_{N \times H}$;
    $h_{\text{sum}}[u] += \sum_{(u \leftarrow v) \in E_\ell} h^\uparrow[v]$
    // Per–edge forgets and summed transformed cell contributions
    $f_{uv} \leftarrow \sigma(W_f x[u] + U_f h^\uparrow[v])$ for $(u \leftarrow v) \in E_\ell$
    $\tilde{c} \leftarrow 0_{N \times H}$;
    $\tilde{c}[u] += \sum_{(u \leftarrow v) \in E_\ell} f_{uv} \odot c^\uparrow[v]$
    // Node–level gates and updates for all $u \in P$
    for $u \in P$ do
        $[i, o, \tilde{u}] \leftarrow \text{split3}(W_{iou} x[u] + U_{iou} h_{\text{sum}}[u])$;
        $i \leftarrow \sigma(i), o \leftarrow \sigma(o), \tilde{u} \leftarrow \tanh(\tilde{u})$;
        $c^\uparrow[u] \leftarrow i \odot \tilde{u} + \tilde{c}[u]$;
        $h^\uparrow[u] \leftarrow o \odot \tanh(c^\uparrow[u])$
/* Top–down pass (parent $\to$ children): propagate from root to leaves. */
for $\ell = 1$ to $L$ do
    $V \leftarrow V_\ell$ // children at depth $\ell$
    for $v \in V$ do
        $p \leftarrow \text{parent}[v]$ // unique parent
        $[i, o, \tilde{u}] \leftarrow \text{split3}(W_{iou}^\downarrow x[v] + U_{iou}^\downarrow h^\downarrow[p])$;
        $i \leftarrow \sigma(i), o \leftarrow \sigma(o), \tilde{u} \leftarrow \tanh(\tilde{u})$;
        $f \leftarrow \sigma(W_f^\downarrow x[v] + U_f^\downarrow h^\downarrow[p])$;
        $c^\downarrow[v] \leftarrow i \odot \tilde{u} + f \odot c^\downarrow[p]$;
        $h^\downarrow[v] \leftarrow o \odot \tanh(c^\downarrow[v])$
return $h \leftarrow \text{concat}(h^\uparrow, h^\downarrow)$ // $[N, 2H]$