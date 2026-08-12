# Research Idea and Implementation Plan Extraction
**Paper Title:** PPE: Positional Preservation Embedding for Token Compression in Multimodal Large Language Models

## 1. Research Background and Existing Pain Points

**Research Background:**
Multimodal Large Language Models (MLLMs) have achieved remarkable success across a range of vision-language understanding tasks. A common paradigm involves encoding images or video frames into dense visual tokens, which are then fed into the language model for joint understanding. However, this dense representation is often highly redundant, leading to inefficiencies in computation and inference. To address this, visual token compression methods have been explored to merge similar tokens to reduce visual sequence length while preserving semantic information, thereby accelerating inference and lowering memory usage.

**Existing Pain Points:**
Despite the efficiency, existing compression methods often disrupt the spatial and temporal structure of visual inputs, limiting their applicability in layout-sensitive tasks such as counting, temporal grounding, and sequential understanding. Specifically:
1.  **Disruption of Spatiotemporal Structure:** Existing token merging methods reduce sequence length but frequently disrupt spatial layouts and temporal continuity by disregarding positional relationships. This distortion affects intra-frame layouts and inter-frame temporal relations.
2.  **Loss of Fine-Grained Cues:** Clustering-based methods like Chat-UniVi mainly assign randomized ID values to the clustered visual tokens, discarding fine-grained spatial or temporal cues.
3.  **Insufficient and Imprecise Positions:** Recent methods like PACT attempt to preserve layouts during compression but remain constrained by insufficient and imprecise positions, as they only retain the ID of the cluster center for the clustered visual tokens. This single-position representation leads to the loss of detailed layouts.
4.  **Incompatibility with High Compression Ratios:** Existing positional embedding methods that preserve only one position for a single token lead to the loss of detailed layouts, which is particularly detrimental when aiming for higher compression ratios or applying cascade compression across multiple transformer layers.

## 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:**
1.  **Preserving Spatial and Temporal Positions:** The primary motivation is to preserve spatial and temporal positions during similar token merging. Since visual scene layouts are crucial for MLLM reasoning, it is essential to ensure that most of the visual scene layouts are still accessible to MLLM at high compression rates.
2.  **Leveraging Feature Sharing for Positions:** Similar token embeddings can share their feature embeddings during token merging. This idea can be extended to a multi-position formulation that better reflects the internal diversity of a token cluster, containing more positional information of original input visual tokens.
3.  **Supporting Cascade Compression:** Different transformer layers capture increasingly abstract representations with higher similarity. Merging tokens in a multi-stage manner enables higher compression ratios without collapsing shallow semantics prematurely. A method is needed that preserves fine-grained spatiotemporal layouts across multiple compression stages.

**Scientific Questions:**
How can we design a positional encoding strategy that explicitly retains different spatiotemporal layouts into one compressed token, preserves positional cues without adding parameters, and supports progressive token compression across multiple layers?

## 3. Overall Core Idea and Design Philosophy

**Overall Core Idea:**
The paper proposes Positional Preservation Embedding (PPE), a novel encoding operator that explicitly introduces the disentangled encoding of 3D positions in the token dimension. PPE enables each compressed token to encapsulate different positions from multiple original tokens by splitting the embedding dimension into chunks to prefill multiple position IDs.

**Design Philosophy:**
1.  **Dimension Independence of RoPE:** The design is motivated by the principle behind RoPE, in which the rotation of position embeddings is independent at the dimension. As a result, the position ID $m$ can be totally different on the dimension $D$ to represent different positions at the same time.
2.  **Extension of M-RoPE:** Inspired by M-RoPE, which partitions the embedding dimension into several groups to store spatiotemporal positions, PPE merges different positions to one merged token by splitting into more groups ($K$ groups).
3.  **Multi-Position Formulation:** Rather than assigning a single position ID (e.g., cluster center) to a merged token, PPE selects the top-$K$ IDs per cluster which are scored by the distance from the cluster center. This ensures that the internal diversity of a token cluster is captured, preserving the vision layout more completely.
4.  **Parameter-Free and Plug-and-Play:** PPE is decoupled from the merging algorithm and operates solely on position IDs. It is parameter-free and can be seamlessly integrated into existing token merging methods without any additional computational costs.

## 4. Core Innovation Points

1.  **Identification of Critical Limitation:** The paper identifies a critical limitation in existing visual token merging methods—namely, the neglect of spatial structure preservation and temporal coherence—which leads to distortion of intra-frame layouts and disruption of inter-frame temporal relations.
2.  **Positional Preservation Embedding (PPE):** A novel, plug-and-play approach that explicitly preserves spatiotemporal integrity during token merging. It effectively addresses both spatial and temporal challenges by enabling each single token to represent multiple positions, thus preserving the vision layout more completely than methods preserving only one position.
3.  **Cascade Compression with PPE:** The paper shows that PPE can be applied in a cascade compression manner within multiple transformer layers of the MLLM. This progressive token compression strategy preserves fine-grained spatial structure across compression stages, enabling substantial compression with minimal performance degradation.
4.  **Parameter-Free and Generic Operator:** PPE is parameter-free and can be used in a plug-and-play fashion for easy implanting into existing visual token compression methods without any additional computational costs or adjustments.

## 5. Overview of the Overall Technical Solution

The overall technical solution involves modifying the positional ID assignment during visual token compression. The process is as follows:
1.  **Visual Token Clustering:** Visual tokens are clustered based on similarity using algorithms like DPC-KNN. The compressed token embeddings are obtained by averaging the token embeddings in each cluster group.
2.  **Top-K Position ID Selection:** For each cluster group, instead of selecting one representative ID, the top-$K$ IDs per cluster are selected based on a scoring mechanism (distance from the cluster center). If the cluster size is smaller than $K$, high-weight tokens are repeated to fill the slots.
3.  **Position ID Dimension Splitting (PPE):** The embedding dimension $D$ is split into $K$ chunks. The selected $K$ position IDs are assigned to these chunks. In the 1D RoPE manner, the dimension is cut into $K$ equal parts. In the 3D M-RoPE manner, the dimensions corresponding to temporal ($D_1$), height ($D_2$), and width ($D_3$) are each evenly cut into $K$ groups.
4.  **PPE Rotation Application:** The rotation operation is applied to the compressed token using the new PPE ID structure, following the standard RoPE formulation but with the split dimension-specific IDs.
5.  **Cascade Compression Integration:** Token compression with PPE is applied not only prior to feeding tokens into the LLM but also within selected LLM layers (e.g., layers 11, 23, and 35). During repeated merges, the number of preserved position IDs reduces gradually.

## 6. Detailed Module Design

### 6.1 Visual Token Compression Module (Base)
This module utilizes a lightweight, parameter-free clustering algorithm (DPC-KNN) to mitigate computational overhead.
*   **Mechanism:** Visual token embeddings are clustered into groups based on token similarity.
*   **Embedding Aggregation:** The compressed token embedding is defined as the average of the token embeddings in the group.

### 6.2 PPE Position ID Assignment Module
This module handles the selection and assignment of multiple position IDs to a single compressed token.
*   **Scoring Mechanism:** Tokens within a cluster are scored by the distance from the cluster center. The score is higher if the token is closer to the cluster center.
*   **Top-K Selection:** The top-$K$ IDs per cluster are selected. $K$ is a fixed hyper-parameter representing the maximum capacity of PPE.
*   **Slot Filling:** If the number of tokens in a cluster is less than $K$, high-weight tokens are repeated to fill the slots.

### 6.3 Dimension Splitting Module
This module splits the embedding dimensions to accommodate multiple position IDs.
*   **1D Manner:** The dimension $D$ is evenly cut into $K$ groups.
*   **3D Manner:** For 3D M-RoPE with sections $D_1, D_2, D_3$, each section is evenly cut into $K$ groups. $K$ is set to the greatest common divisor (GCD) of the M-RoPE sections to ensure each dimension is evenly cut.

### 6.4 PPE Rotation Module
This module computes the attention rotation using the PPE IDs.
*   **Mechanism:** It follows the RoPE formula but uses the dimension-specific PPE IDs ($\hat{m}_d$) instead of a single position ID $m$.

### 6.5 Cascade Compression Module
This module integrates compression within the LLM layers.
*   **Layer-wise Insertion:** Visual token clustering is applied within selected LLM layers (e.g., layers 11, 23, 35).
*   **Progressive ID Reduction:** During repeated merges, the number of preserved position IDs reduces gradually (e.g., retaining only four IDs when two previously merged tokens are merged again).

## 7. All Mathematical Formulas and Symbol Definitions

**Formula 1: RoPE Rotation Operation**
$$RoPE(z_d, m) = e^{im\theta_d}z_d, d = 1...D$$
*   $z \in \mathbb{R}^D$: Token vector in multi-head attention.
*   $m$: Position ID of $z$.
*   $\theta_d$: Frequency parameter.
*   $D$: Embedding dimension.

**Formula 2: M-RoPE Rotation Operation**
$$M\text{-}RoPE(z_d, m_d) = e^{im_d\theta_d}z_d, d = 1...D$$
*   $m \in \mathbb{Z}^D$: Pre-filled 2D or 3D visual position IDs.
*   $m_d$: Position ID for dimension $d$.

**Formula 3: 3D Visual Position ID Definition**
$$m_d = \begin{cases} t, & d = 1 ... D_1 \\ h, & d = D_1 + 1 ... D_1 + D_2 \\ w, & d = D_1 + D_2 + 1 ... D_1 + D_2 + D_3 \end{cases}$$
*   $(t, h, w)$: 3D visual token position (temporal, height, width).
*   $D_1, D_2, D_3$: Human-crafted integers controlling the size of M-RoPE sections, satisfying $D_1 + D_2 + D_3 = D$.

**Formula 4: Compressed Token Embedding**
$$z'_j = \frac{1}{|C_j|} \sum_{i \in C_j} z_i, j = 1...M$$
*   $\{z_i \in \mathbb{R}^D\}_{i=1}^N$: Original visual token embeddings.
*   $C_j$: The set of token indices in cluster group $j$.
*   $M$: Number of clustered groups.
*   $z'_j$: Compressed token embedding.

**Formula 5: PPE ID in 1D Manner**
$$\hat{m}_d = m_{k,d}, d = (k - 1)\frac{D}{K} + 1 ... k\frac{D}{K}$$
*   $\hat{m}$: Merged PPE ID.
*   $K$: Fixed hyper-parameter representing the maximum capacity of PPE (number of preserved IDs). $K$ is always divisible by dimension $D$.
*   $\{m_k\}^K$: Set of different position IDs in 1D RoPE manner before merging.

**Formula 6: PPE ID in 3D Manner**
$$\hat{m}^{3D}_d = m^{3D}_{k,d}, d = \begin{cases} (k - 1)\frac{D_1}{K} + 1 ... k\frac{D_1}{K}, \\ (k - 1)\frac{D_2}{K} + D_1 + 1 ... k\frac{D_2}{K} + D_1, \\ (k - 1)\frac{D_3}{K} + D_1 + D_2 + 1 ... k\frac{D_3}{K} + D_1 + D_2. \end{cases}$$
*   $D_1, D_2, D_3$: M-RoPE sections.
*   $K$: Set to the greatest common divisor of the M-RoPE sections.

**Formula 7: PPE Rotation of Vector $z$**
$$PPE(z_d, \hat{m}_d) = e^{i\hat{m}_d\theta_d}z_d, d = 1...D$$
*   $\hat{m}_d$: The merged PPE ID for dimension $d$.

## 8. Algorithm Pseudocode

The paper does not provide an explicit algorithm pseudocode block. The algorithmic logic is described in the text as follows:
1.  Cluster $N$ visual tokens into $M$ groups using DPC-KNN.
2.  Compute compressed embedding $z'_j$ by averaging embeddings in group $C_j$.
3.  For each group $C_j$, select top-$K$ tokens with highest scores (closest to cluster center).
4.  If $|C_j| < K$, repeat high-weight tokens to fill $K$ slots.
5.  Assign the position IDs of these top-$K$ tokens to the split dimensions of the merged token according to Eq. 5 (1D) or Eq. 6 (3D).
6.  Apply PPE rotation (Eq. 7) during attention computation.
7.  (For Cascade) Repeat clustering and PPE assignment in selected LLM layers.