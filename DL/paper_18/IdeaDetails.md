## 1. Research Background and Existing Pain Points

**Research Background:** Large Language Models (LLMs) have shown outstanding performance in long context tasks and are widely applied in fields such as code generation and mathematical reasoning. However, transformer-based LLMs face efficiency limitations when processing long texts with millions of tokens. The quadratic complexity of the attention mechanism poses a significant challenge to the inference of models. Recently, numerous studies have worked on refining the attention module to increase the training and inference efficiency. Sparse attention methods mitigate the latency issues of the attention mechanism by skipping certain computations. By leveraging the inherent sparsity of attention score distributions, these methods can reduce computational costs while preserving model performance.

**Existing Pain Points:** While existing dynamic block sparse attention methods have proven effective in long-context scenarios, their performance at high sparsity rates is still limited by their coarse-grained importance estimation. To evaluate the importance of each block, these methods typically compress along the sequence dimension to approximate their attention scores (e.g., heuristic "vertical and slash" patterns or pooling methods). This coarse-grained compression can cause a few high-scoring positions to be overlooked, which in turn impacts the final performance. A token-level dot-product calculation would solve this issue but is not feasible, as its time complexity scales identically to full attention. Static methods use heuristic fixed templates to achieve sparse computation, making it difficult to maintain model performance with diverse attention patterns.

## 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:** To achieve a fine-grained and efficient trade-off in block importance estimation by exploring compression of the attention head dimension instead of the sequence dimension. A viable approach for both accuracy and efficiency in token-level dot-product computations is to compress along the head number dimension, allowing a few key heads to serve as proxies for estimating the importance of all heads. However, the feasibility of this approach hinges on the assumption that attention heads exhibit a certain degree of consistency in their scores.

**Scientific Questions:** 
1. Do attention heads exhibit consistency in their token focus such that a few representative heads can approximate the behavior of all heads?
2. How do the differences in sparsity among various attention heads affect the design of sparse attention masks, and how can these differences be dynamically accommodated?
3. Can compressing the head number dimension provide a more fine-grained and accurate block importance estimation than compressing the sequence dimension, while maintaining low computational cost?

Based on observational studies on the behaviors of multiple heads in a long-context setting (Llama model using 8K tokens), two key findings answer these questions:
*   **Attention heads exhibit consistency in token focus:** In long texts, multiple attention heads, particularly in deeper layers, focus on a large intersection of tokens. The consistency between the cumulative probability curve based on a shared ranking and the actual score curve for specific heads serves as the basis for the ProxyAttn algorithm.
*   **Attention heads exhibit variability in sparsity:** The primary variation between heads is in their sparsity rather than the tokens they focus on. Given the same ranking, all heads assign high attention to the leading tokens, but some heads are very sparse and only attend to the initial portion, while others are dense and maintain significant scores for later tokens.

## 3. Overall Core Idea and Design Philosophy

The core idea is to propose ProxyAttn, a training-free sparse attention algorithm that achieves more precise block estimation by compressing the dimension of attention heads. Based on the observation of the similarity among multiple attention heads, the method uses the scores of pooled representative heads to approximate the scores for all heads. To account for the varying sparsity among heads, the design incorporates a block-aware dynamic budget estimation method. By combining the scores from representative proxy heads with multi-head dynamic budgets, the method achieves a more fine-grained block importance evaluation at low computational cost.

**Design Philosophy:** Current works perform coarse-grained block estimation with sequence dimension compression, which loses fine-grained token-level importance. ProxyAttn shifts the compression dimension to the head number. It leverages the consistency of token focus among heads to share scores within a group (using a proxy head), and leverages the variability of sparsity to assign dynamic budgets per head. This allows for fine-grained importance estimation (token-level dot-products via proxy heads) while ensuring efficiency (computing only a fraction of the heads) and flexibility (diverse masks for diverse sparsity levels).

## 4. Core Innovation Points

1.  **Proxy Head Score Estimation:** Introduction of ProxyAttn, a training-free sparse attention estimation method that leverages the attention scores from a small number of proxy heads to efficiently estimate the scores for all heads. By compressing the head number dimension instead of the sequence dimension, it achieves more accurate importance measures without the quadratic cost of full token-level computation.
2.  **Block-Aware Dynamic Budget Allocation:** Considering the diverse sparsity among heads, the proposal of a block-aware budget allocation method. Coupled with the unified importance score, this method can provide different head sparsity budgets online, which in turn yields diverse sparse attention masks tailored to individual heads' density characteristics.
3.  **Fine-Grained Importance Evaluation:** By combining unified proxy scores with multi-head dynamic budgets, ProxyAttn achieves a fine-grained block importance evaluation that avoids the information loss inherent in sequence-dimension compression methods.
4.  **Efficiency Optimization via Stride and Max Pooling:** The integration of strided dropout on qk pairs and max pooling ensures that the estimation overhead is minimal (roughly one percent of full attention for a 7B model) while preserving the accuracy of token-level dot-product representations.

## 5. Overview of the Overall Technical Solution

The overall technical solution of ProxyAttn consists of two main modules: Unified Score Estimation and Dynamic Budget Allocation. 
First, the attention heads are partitioned into distinct groups. Within each group, a designated proxy head is formed by average pooling the queries and keys of all heads in the group. The attention scores for this proxy head are computed, and max pooling is applied to obtain token-level importance scores, which are then shared across all heads in the group. To reduce computational cost further, a stride mechanism drops certain qk pairs.
Second, since heads vary in sparsity, a dynamic budget is estimated for each head individually online. This uses the queries from the last block and average pooling to approximate the minimum budget required for the query of the final block to achieve a cumulative probability threshold $\gamma$. 
Finally, the unified block importance scores are combined with the per-head dynamic budgets to select the Top-K blocks, generating diverse sparse attention masks for each head. The actual attention computation is then performed using an efficient block sparse attention kernel based on these masks.

## 6. Detailed Module Design

### 6.1 Unified Score Estimation Module
In order to obtain the token-level dot-product scores for each block in the attention map, compression is applied along the head num dimension based on the consistency of token attention across heads.
*   **Head Grouping:** The attention heads are partitioned into distinct groups $G_g$.
*   **Proxy Head Construction:** For the representative head queries $Q_g$ and keys $K_g$, the same average pooling as for the sequence is heuristically applied. This is a form of compression along the head number dimension.
*   **Score Computation:** A single token-level score will be shared for all heads belonging to group $g$. Combined with max-pooling, it enables fine-grained block importance estimation at the cost of computing only $|G|$ proxy heads.
*   **Stride Reduction:** To further reduce the computational cost of the estimation operation, certain qk pairs are dropped by stride, keeping only the first token in each stride window. Driven by the local similarity inherent in attention mechanisms, this operation reduces the computational cost without significant loss of accuracy. For a model with $n$ heads, the theoretical computational cost of using $g$ representative proxy heads for estimation is expected to be $\frac{g}{n} \cdot \text{stride}^2$ of full multi-head attention.
*   **GQA Alignment:** For models using Group-Query Attention (GQA), the group granularity will be aligned with the keys, ensuring that all queries belonging to the same group of keys are also within that same group.

### 6.2 Dynamic Budget Allocation Module
After obtaining importance scores, a common practice is to obtain a sparse block mask using Top-K selection. However, directly applying this would result in identical masks for all heads in the same group, impacting the maximum achievable sparsity rate.
*   **Sparsity Estimation:** Inspired by prior work, the queries from the last block are used to estimate the sparsity of different heads. The budget required for a given head is approximated as the minimum budget for the query of the final block to achieve a cumulative probability of $\gamma$.
*   **Block-Level Aggregation:** Given the significant discrepancy between token-level computation and the actual block budget due to attention sparsity, average pooling is employed to ensure that the budget estimation is performed at the block level.
*   **Mask Generation:** After obtaining the budget $b_i$ for each head, the top $b_i$ blocks with the highest unified scores are selected to form the sparse attention masks. The final sparse attention mask $S^i_{M \times N}$ for head $i$ is generated by checking if the block index falls into the Top-K indices of the importance score matrix $A_i$.

## 7. All Mathematical Formulas and Symbol Definitions

**Formula 1: Unified Score Estimation**
The attention score approximation for all heads in a group using a proxy head.
$$A_i = \text{maxpool}\left(\text{softmax}\left(\frac{Q_g K_g^T}{\sqrt{d_k}}\right)\right), \quad i \in G_g$$
*   $Q_g, K_g \in \mathbb{R}^{N \times d_k}$: Representative head queries and keys.
*   $A_i \in \mathbb{R}^{N_b \times N_b}$: The block-level importance score matrix for head $i$.
*   $G_g$: The set of heads belonging to group $g$.
*   $d_k$: The dimension of the key vectors.

**Formula 2: Proxy Head Pooling**
The construction of representative head queries and keys via average pooling along the head dimension.
$$Q_g = \frac{1}{|G|} \sum_{i \in G} Q_i$$
$$K_g = \frac{1}{|G|} \sum_{i \in G} K_i$$
*   $|G|$: The number of heads in the group.
*   $Q_i, K_i$: The query and key states for head $i$.

**Formula 3: Sparse Attention Mask Generation**
The formulation of the final sparse attention mask based on dynamic budgets.
$$S^i_{mn} = \begin{cases} 1, & \text{if } n \in \text{TopK}(A_i[m], K=b_iN).\text{index} \\ 0, & \text{otherwise} \end{cases}$$
*   $S^i_{mn}$: The sparse attention mask value at block row $m$ and block column $n$ for head $i$.
*   $M, N$: The block-level sequence length.
*   $A_i$: The previously estimated block-level importance of head $i$.
*   $b_i$: The estimated budget ratio for head $i$.
*   $N$: The total number of blocks.

**Formula 4: Token Overlap Score (Observational Study)**
The calculation of overlap scores between heads to validate consistency, particularly after removing sink tokens.
$$\text{token index} = \text{topk}(A[i], k = 1024).\text{indices}$$
$$\text{Score}[i, j] = \text{sum}(A[j, \text{token index}])$$
*   $A$: Attention scores for a given token with shape (head num, seq len).
*   $k$: The number of top tokens to consider (1024).

## 8. Algorithm Pseudocode

**Algorithm 1: Block-aware Budget Estimation for Head i**

**Input:** Query states $Q_i$, key states $K_i$, cumulative probability threshold $\gamma$
**Output:** Estimated budget $b_i$

1: $\hat{A}_i \leftarrow \text{softmax}\left(Q_i K_i^\top / \sqrt{d_k}\right)$ $\triangleright$ Compute approximate attention scores
2: $a_i \leftarrow \text{avgpool}(\hat{A}_i)$ $\triangleright$ Aggregate into block-level scores
3: $a_i \leftarrow \text{sort}(a_i) / \text{sum}(a_i)$ $\triangleright$ Normalize and sort block-level scores
4: $b_i \leftarrow \min\left\{k \mid \sum_{j=0}^k a_i[j] \geq \gamma \right\} / |a_i|$ $\triangleright$ Determine budget ratio for selected head
5: **return** $b_i$