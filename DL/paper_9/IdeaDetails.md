## 1. Research Background and Existing Pain Points

**Research Background:**
Graph Neural Networks (GNNs) have demonstrated remarkable success in various domains such as social networks, molecular chemistry, and more. A crucial component of GNNs is the pooling procedure, in which the node features calculated by the model are combined to form an informative final descriptor to be used for the downstream task. In GNNs, layers capture multiple scales: early layers model local neighborhoods and motifs, while deeper layers encode global patterns (communities, long-range dependencies, topological roles), mirroring CNNs where shallow layers detect edges/textures and deeper layers capture object semantics. 

**Existing Pain Points:**
1. Previous graph pooling schemes rely on the last GNN layer features as an input to the pooling or classifier layers, potentially under-utilizing important activations of previous layers produced during the forward pass of the model (historical graph activations).
2. This gap is particularly pronounced in cases where a node’s representation can shift significantly over the course of many graph neural layers, and worsened by graph-specific challenges such as over-smoothing in deep architectures (where node representations become indistinguishable) and over-squashing.
3. Greater depth can overwrite early information and cause over-smoothing, making node representations indistinguishable. Standard pooling operations fail to preserve multi-scale features at readout, limiting their ability to capture long-range dependencies and hierarchical patterns.
4. Specialized methods like state space and autoregressive moving average models on graphs often focus on improving stability during training, without explicitly modeling the internal trajectory of node features across layers.
5. Standard softmax-based attention is restricted to non-negative convex combinations and cannot realize subtractive (high-pass) filtering, preventing the model from expressing additive or subtractive relationships between layers akin to finite-difference approximations in dynamical systems.

## 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:**
We argue that a GNN’s computation path and the sequence of node features through layers can be a valuable signal. By reflecting on this trajectory, models can better understand which transformations were beneficial and refine their final predictions accordingly. Computational history is a powerful, general inductive bias. By modeling the evolution of node representations across layers, the model can leverage both the activation history of nodes and the graph structure to refine features used for final prediction, preserving information across propagation depths without clustering or node dropping.

**Scientific Questions:**
1. How to leverage the full trajectory of node embeddings across layers rather than relying solely on the final GNN layer features?
2. How to disentangle and model the two critical axes of GNN behavior: the evolution of node embeddings through layers, and their spatial interactions across the graph?
3. How to enable the architecture to approximate low/high pass filters over the layer trajectory to express subtractive relationships between layers, overcoming the limitations of standard softmax-based attention?

## 3. Overall Core Idea and Design Philosophy

**Overall Core Idea:**
Introduce HISTOGRAPH, a self-reflective architectural paradigm that enables GNNs to reason about their historical graph activations. HISTOGRAPH introduces a two-stage self-attention mechanism that disentangles and models two critical axes of GNN behavior: the evolution of node embeddings through layers, and their spatial interactions across the graph.

**Design Philosophy:**
1. **Historical Pooling:** At readout, accumulate node histories across layers, creating a global shortcut at aggregation that revisits and integrates multi-hop features into the final representation, unlike prior models that apply residuals only within node updates or via hierarchical coarsening.
2. **Disentangled Two-Stage Attention:** The layer-wise module treats each node’s layer representations as a sequence and learns to attend to the most informative representation, while the node-wise module aggregates global context to form richer, context-aware outputs.
3. **Signed Normalization over Softmax:** Apply a normalization that permits signed contributions (division by sum rather than softmax). This allows the model to express additive or subtractive relationships between layers, akin to finite-difference approximations in dynamical systems, realizing general Finite Impulse Response (FIR) filters.
4. **Single Readout Spatial Attention:** Apply node-wise self-attention exactly once at the readout stage—not at every GNN depth—omitting spatial positional encodings to preserve permutation invariance. This ensures it does not contribute to over-smoothing during the GNN forward pass and incurs only a single $O(N^2D)$ cost.

## 4. Core Innovation Points

1. **Self-Reflective Architectural Paradigm:** Introduce a self-reflective architectural paradigm for GNNs that leverages the full trajectory of node embeddings across layers (historical graph activations).
2. **Two-Stage Self-Attention Mechanism:** Propose HISTOGRAPH, a two-stage self-attention mechanism that disentangles the layer-wise node embeddings evolution and spatial aggregation of node features.
3. **Signed Normalization for Adaptive Trajectory Filtering:** Use a signed normalization (division by sum of raw scores) instead of softmax, which permits negative weights enabling the architecture to approximate low/high pass filters over the layer trajectory and express subtractive relationships between layers.
4. **Readout-Only Spatial Attention:** Apply multi-head self-attention across nodes exactly once at the readout stage to model spatial dependencies, which prevents over-smoothing during the forward pass and reduces computational depth coefficient from $L$ to $1$.
5. **Post-Processing Capability:** HISTOGRAPH can be employed as a lightweight post-processing tool (frozen backbone auxiliary head) to further enhance the performance of models trained with standard graph pooling layers, yielding substantial gains with minimal overhead.

## 5. Overview of the Overall Technical Solution

1. Given input node features $F$ and adjacency $A$, a backbone GNN produces historical graph activations $X_1, .., X_{L-1}$.
2. The historical graph activations are projected to a common hidden dimension $D$ and augmented with fixed sinusoidal positional encodings to encode layer ordering.
3. **Stage 1 (Layer-wise Attention):** The Layer-wise attention module uses the final-layer embedding as a query to attend over all historical states while averaging across nodes, yielding per-node aggregated embeddings $H$.
4. **Stage 2 (Node-wise Attention):** A Node-wise self-attention module refines $H$ by modeling interactions across nodes, producing $Z$, then averaged if graph embeddings $G$ is wanted.

## 6. Detailed Module Design

**Input Projection and Per-Layer Positional Encoding:**
Project the historical activations to a common hidden dimension $D$ using a linear transformation. To encode layer ordering, add fixed sinusoidal positional encodings as in Vaswani et al. (2017). The positional encoding is broadcast across the node dimension and added element-wise to obtain layer-aware features.

**Layer-wise Attention:**
View each node through its sequence of historical activations and use attention to learn which activations are most relevant. Use only the last-layer embedding as the query to attend over all historical states. Apply scaled dot-product attention and average across nodes, obtaining a layer weighting scheme. Rather than softmax, which enforces non-negative weights and suppresses negative differences, apply a normalization that permits signed contributions. This allows the model to express additive or subtractive relationships between layers, akin to finite-difference approximations in dynamical systems. The cross-layer pooled node embeddings are computed as a weighted sum.

**Node-wise Attention:**
To obtain a graph-level representation, apply multi-head self-attention across nodes exactly once at the readout stage—not at every GNN depth—omitting spatial positional encodings to preserve permutation invariance. Optionally followed by residual connections and LayerNorm. Averaging over nodes yields the graph representation which feeds the final prediction head.

**Frozen Backbone Efficiency:**
With a pretrained, frozen message-passing backbone, train only the HISTOGRAPH head. Cache the $N \times L \times D$ activations per graph in one forward pass and skip gradients through the backbone, removing $O(L(ED + ND^2))$ work. The backward pass applies only to the head, $O(N(L + N)D)$, substantially reducing memory and training time.

**Properties of HISTOGRAPH:**
1. **Mitigating Over-smoothing:** As node embeddings tend to become indistinguishable after a certain depth, HISTOGRAPH aggregates representations across layers using a weighted combination. Attention weights that place nonzero mass on early layers let the final embedding retain discriminative early representations, countering over-smoothing so node embeddings remain distinguishable.
2. **Adaptive Trajectory Filter:** HISTOGRAPH's layer-wise attention acts as a filter over the sequence of representations. The signed normalization permits negative weights, enabling subtractive relationships analogous to an FIR filter with signed coefficients. This allows low-pass (uniform average), high-pass (first difference), or general FIR filtering. Standard softmax cannot realize subtractive filtering.

## 7. All Mathematical Formulas and Symbol Definitions

**Symbol Definitions:**
- $F \in \mathbb{R}^{N \times D_{in}}$: Raw input node features.
- $N$: Number of nodes in the batch.
- $D_{in}$: Input feature dimensionality.
- $X^{(0)} = \text{Emb}_{in}(F)$: Initial representation, where $\text{Emb}_{in}$ is a linear layer.
- $X^{(l)} = \text{GNN}^{(l)}(X^{(l-1)})$: Recursive representation for layer $l = 1, \ldots, L-1$.
- $\mathcal{X} = [X^{(0)}, X^{(1)}, \ldots, X^{(L-1)}] \in \mathbb{R}^{N \times L \times D}$: Historical graph activations.
- $W^{(l)}: \mathbb{R}^{D_l} \rightarrow \mathbb{R}^D$: Per-layer linear projections.
- $\mathcal{X}' = \text{Emb}_{hist}(\mathcal{X}) \in \mathbb{R}^{N \times L \times D}$: Projected historical activations.
- $P \in \mathbb{R}^{L \times D}$: Positional encodings.
- $\tilde{\mathcal{X}} \in \mathbb{R}^{N \times L \times D}$: Layer-aware features.
- $Q, K, V$: Query, Key, Value for attention.
- $c \in \mathbb{R}^{1 \times L}$: Layer weighting scheme (attention logits).
- $\alpha_l$: Signed normalization coefficients.
- $H \in \mathbb{R}^{N \times D}$: Cross-layer pooled node embeddings.
- $Z \in \mathbb{R}^{N \times D}$: Node-wise self-attention output.
- $G \in \mathbb{R}^D$: Graph-level representation.

**Formulas:**
Input Projection and Per-Layer Positional Encoding:
$$ \mathcal{X}' = \text{Emb}_{hist}(\mathcal{X}) \in \mathbb{R}^{N \times L \times D} $$
$$ P_{l,2k} = \sin\left(\frac{l}{10000^{2k/D}}\right), \quad P_{l,2k+1} = \cos\left(\frac{l}{10000^{2k/D}}\right) $$
for $0 \le l < L, 0 \le k < D/2$.
$$ \tilde{\mathcal{X}}_{v,l} = \mathcal{X}'_{v,l} + P_l $$

Layer-wise Attention and Node-wise Attention:
$$ Q = \tilde{\mathcal{X}}_{L-1} W^Q \in \mathbb{R}^{N \times 1 \times D}, \quad K = \tilde{\mathcal{X}} W^K \in \mathbb{R}^{N \times L \times D}, \quad V = \tilde{\mathcal{X}} \in \mathbb{R}^{N \times L \times D} $$
$$ c = \text{Average}\left(\frac{QK^\top}{\sqrt{D}}\right) \in \mathbb{R}^{1 \times L} $$
$$ \alpha_l = \frac{c_l}{\sum_{l'=0}^{L-1} c_{l'}} $$
$$ H = \sum_{l=0}^{L-1} \alpha_l \cdot \tilde{\mathcal{X}}_l = \sum_{l=0}^{L-1} \frac{c_l}{\sum_{l'=0}^{L-1} c_{l'}} \cdot \tilde{\mathcal{X}}_l \in \mathbb{R}^{N \times D} $$

Graph-level Representation:
$$ Z = \text{MHSA}(H, H, H) \in \mathbb{R}^{N \times D} $$
$$ G = \text{Average}(Z) \in \mathbb{R}^D $$

Computational Complexity:
$$ O(NLD + N^2D) = O(N(L+N)D) $$

Over-smoothing Definition:
$$ \|x_u^{(l_1)} - x_v^{(l_2)}\| \to 0 \quad \text{for all } u, v \text{ and layers } l_1, l_2 \ge L_0 $$
$$ h_u = \sum_{l=0}^{L-1} \alpha_l x_u^{(l)}, \quad \text{with } \sum_l \alpha_l = 1 $$
$$ \|h_u - h_v\| > 0 $$

Proposition 1 Proof Formulas:
$$ h_u - h_v = \sum_{l=0}^{L-1} \alpha_l \left( x_u^{(l)} - x_v^{(l)} \right) $$
$$ h_u - h_v = \sum_{l=0}^{L_0} \alpha_l \left( x_u^{(l)} - x_v^{(l)} \right) + \sum_{l=L_0+1}^{L-1} \alpha_l \left( x_u^{(l)} - x_v^{(l)} \right) $$
$$ \|x_u^{(l)} - x_v^{(l)}\| \approx 0 \quad \text{for all } l > L_0 $$
$$ h_u - h_v \approx \sum_{l=0}^{L_0} \alpha_l \left( x_u^{(l)} - x_v^{(l)} \right) $$
$$ \|x_u^{(l')} - x_v^{(l')}\| \neq 0 $$
$$ \|h_u - h_v\| \approx \left\| \alpha_{l'} \left( x_u^{(l')} - x_v^{(l')} \right) + \sum_{l \neq l'} \alpha_l \left( x_u^{(l)} - x_v^{(l)} \right) \right\| > 0 $$

## 8. Algorithm Pseudocode

**Algorithm 1 HISTOGRAPH Forward Pass**

**Input:** $\mathcal{X} \in \mathbb{R}^{N \times L \times D_{in}}$
**Output:** Graph-level representation $Y \in \mathbb{R}^D$

1: $\mathcal{X}' \leftarrow \text{Emb}_{hist}(\mathcal{X})$ \hspace{1cm} \triangleright Linear projection to $D$ dimensions
2: $\tilde{\mathcal{X}} \leftarrow \mathcal{X}' + P$ \hspace{1cm} \triangleright Add sinusoidal positional encoding
3: $Q \leftarrow W^Q \tilde{\mathcal{X}}_{L-1}$ \hspace{1cm} \triangleright Query: last-layer embedding
4: $K \leftarrow W^K \tilde{\mathcal{X}}$, $V \leftarrow \tilde{\mathcal{X}}$ \hspace{1cm} \triangleright Key and Value: all layers
5: $c \leftarrow \frac{QK^\top}{\sqrt{D}}$ \hspace{1cm} \triangleright Dot-product attention logits
6: $c \leftarrow \text{Average}(c)$ \hspace{1cm} \triangleright Average across nodes
7: $\alpha_t \leftarrow \frac{c_t}{\sum_{t'} c_{t'}}$ \hspace{1cm} \triangleright Normalize over time
8: $H \leftarrow \sum_{t=0}^{L-1} \alpha_t \tilde{\mathcal{X}}_t$ \hspace{1cm} \triangleright Layer-wise aggregation
9: $Z \leftarrow \text{MHSA}(H, H, H)$ \hspace{1cm} \triangleright Node-wise self-attention
10: **return** $Y = \text{Average}(Z)$ \hspace{1cm} \triangleright Average across nodes