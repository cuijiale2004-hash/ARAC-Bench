1. **Research Background and Existing Pain Points**
Transformer decoders have achieved strong results across tasks, but the memory required for the KV cache becomes prohibitive at long sequence lengths. Practical deployment for real-world applications requires LLM inference to achieve low latency and high throughput, which becomes particularly challenging with increasingly long contexts, primarily due to the linear memory overhead of the key-value (KV) caches in autoregressive generation. To mitigate this, various methods have been proposed to reduce KV caches. Within the layer, Group-Query Attention (GQA) and Multi-Query Attention (MQA) reduce redundancy by sharing one head’s key and value across multiple query heads. MLA uses low-rank joint compression for key-value caches. Across the layer, CLA shares the KV cache between adjacent layers, while YOCO reuses the intermediate layer cache for the top-half layers. Although cross-layer sharing techniques also reduce memory overhead, they consistently underperform within-layer methods, which motivates the need for a more effective cross-layer sharing strategy.

2. **Core Research Motivation and Scientific Questions**
To understand the root cause of why cross-layer KV Cache sharing (e.g., YOCO, CLA) typically underperforms within-layer methods like GQA, the authors investigate the information flow of keys and values of the top-layers. The preliminary experiment reveals a clear distribution: values are predominantly derived from the bottom layer, while keys draw more information from both bottom and middle layers. Most approaches often overlook the distinct roles of keys and values: keys drive relevance scoring, while values supply contextual content. Treating KV as a single unit for caching neglects this functional disparity. Moreover, sharing KV states indiscriminately across layers risks undermining each layer’s specific role in hierarchical processing, potentially introducing irrelevant context and hindering its unique function. The core scientific question is how to leverage this asymmetric information flow principle to design a cross-layer KV cache sharing strategy that reduces memory while achieving performance superior to standard Transformer decoders.

3. **Overall Core Idea and Design Philosophy**
The overall core idea is based on a previously overlooked asymmetric key-value sharing principle: for top-half layers, values are most effectively reconstructed by the bottom layer, whereas keys draw their more critical information from the bottom and middle layer. Based on this insight, the design philosophy proposes FusedKV, which reconstructs the top-half layer caches by performing a dimension-wise, weighted fusion of caches from two highly informative source layers: the bottom and middle layer. This fusion operates directly on post-RoPE keys, preserving relative positional information without the computational cost of re-applying rotary embeddings. To further improve efficiency, FusedKV-Lite is proposed, where top-layer KV caches are directly derived from the bottom-layer values and the middle-layer keys.

4. **Core Innovation Points**
*   Identification of a key-value asymmetry in cache reconstruction: values are most effectively reconstructed by the bottom layer, whereas keys draw more critical information from the bottom and middle layer.
*   Proposal of FusedKV, a memory-efficient architecture that reconstructs the top-half layer caches via a learnable weighted fusion of caches from the bottom and middle layers, striking an effective balance between representational power and memory traffic costs.
*   Proposal of FusedKV-Lite, an I/O-efficient variant that uses direct asymmetric sharing where keys are directly reused from the middle layer and values from the bottom layer, decreasing I/O overhead by one-third with only slight degradation.
*   Demonstration of RoPE compatibility with learnable weights: establishing that RoPE is compatible with learnable weights when the weights are symmetric, and showing that fusing post-RoPE key caches preserves relative positional encoding without the computational overhead of RoPE recomputation.
*   Efficient Triton implementation of FusedKV and a systematic evaluation showing that the methods can reduce KV cache memory by 50% while achieving lower perplexity than a standard Transformer decoder.
*   Discovery of gradient flow enhancement: FusedKV and FusedKV-Lite consistently exhibit a markedly larger gradient L2 norm compared to baselines, particularly within the shallower layers, suggesting these layers learn more effectively and rapidly.

5. **Overview of the Overall Technical Solution**
The overall technical solution partitions the set of $L$ decoder layers, $\mathcal{L} = \{1, . . . , L\}$, into two disjoint subsets: Storage Layers ($\mathcal{L}_S$), for which the KV caches are explicitly stored in memory during generation, and Reconstruction Layers ($\mathcal{L}_R$), whose KV caches are not stored but are recomputed on-demand via a reconstruction function. For any reconstruction layer $i \in \mathcal{L}_R$, its Key, $K^i \in \mathbb{R}^{s\times d}$, and Value, $V^i \in \mathbb{R}^{s\times d}$, are generated by a parameterized reconstruction function $F_i$ using the KV caches from a subset of the storage layers. The architecture of the reconstruction function $F$ determines the complexity and effectiveness. Two primary categories are explored: Direct Cache Reuse and Weighted Fusion of Caches. FusedKV uses weighted fusion from two source layers (bottom and middle), while FusedKV-Lite uses direct asymmetric reuse (middle keys, bottom values).

6. **Detailed Module Design**
*   **Overall Framework Definition**: Partition layers into Storage Layers ($\mathcal{L}_S$) and Reconstruction Layers ($\mathcal{L}_R$). The source layer mapping function $\Phi(i)$ specifies a set of source storage layer indices for reconstruction layer $i$.
*   **Direct Cache Reuse Module**: The most straightforward approach where the KV cache is directly reused from source layers without transformation. The reconstruction function $F_i$ acts as a selector, and the source layer mapping $\Phi(i)$ typically contains a single index.
    *   YOCO: Partitions layers into two contiguous blocks. All reconstruction layers access the KV cache from the final storage layer, defined by the constant mapping $\phi(i) = L/2$.
    *   CLA: Employs an interleaved partitioning scheme, designating odd-numbered layers for storage and even-numbered layers for reconstruction. Each reconstruction layer accesses the KV cache from its immediately preceding storage layer, with the mapping $\phi(i) = i-1$.
*   **Weighted Fusion of Caches Module**: Computes the reconstructed KV cache as a weighted linear combination of the caches from multiple source layers. The learnable weight can be scalars, vectors, or matrices, and are broadcast to match the shape of the Key and Value caches for the Hadamard product.
*   **FusedKV Module**: Reconstructs the cache by computing a learnable weighted fusion of caches from two highly informative source layers: the bottom (layer 1) and the middle (layer $n$). It allows each reconstruction layer to synthesize its cache from both the low-level, foundational features of the initial layer and the more abstract, contextual representations from the final source layer.
*   **FusedKV-Lite Module**: Reconstructs the keys by directly reusing the cache from the last source layer $n$, and the values from the bottom layer. By reusing only a single source key and value cache, FusedKV-Lite avoids the additional I/O overhead from fusion, thereby maintaining efficiency on par with the vanilla model, making it highly efficient for I/O-bound inference scenarios.
*   **RoPE Compatibility Module**: Analyzes the attention score with a learnable weight vector applied to RoPE-transformed keys. To ensure that learnable weights do not disrupt the relative positional encoding property of RoPE, identity within each 2D weight pair must be enforced (symmetric weights). With this symmetry constraint, a weighted fusion of multiple RoPE-transformed key vectors maintains relative positional encoding. This allows storage layers to retain their original post-RoPE KV caches, avoiding re-computing RoPE at inference time.
*   **Training and Initialization Strategies Module**:
    *   *Iterative Fusion with Auxiliary Caches*: Introduces an iterative fusion process for reconstruction layers, maintaining one-layer caches iteratively updated. For the first reconstruction layer ($i = n + 1$), it is identical to standard FusedKV. For subsequent layers ($i > n + 1$), the key cache is formed by fusing the reconstructed key from the previous layer with the cache from the last storage layer, and similarly for the value cache. All learnable d-dimensional weight vectors are drawn from a standard normal distribution, $\mathcal{N}(0, 1)$.
    *   *Standard FusedKV with Normal Initialization*: Each d-dimensional weight vector for every layer is independently sampled from $\mathcal{N}(0, 1)$.
    *   *Standard FusedKV with Equivalent Initialization*: Initializes the standard FusedKV by recursively computing its weights to match the output of the iterative method at initialization. Auxiliary weights are sampled from $\mathcal{N}(0, 1)$, and the final model weights are deterministically computed.
*   **Inference Module**: Implemented as a Triton-based attention kernel. Benchmarking includes attention throughput, Time to First Token (TTFT), and Time Per Output Token (TPOT). FusedKV-Lite maintains a comparable cache I/O to achieve identical throughput to MHA. FusedKV and FusedKV-Lite skip cache prefilling in current layers, halving the overall prefilling latency. In compute-bound settings, the cache I/O overhead in FusedKV is effectively hidden by the computational workloads.

7. **All Mathematical Formulas and Symbol Definitions**
*   $\mathcal{L} = \{1, . . . , L\}$: Set of $L$ decoder layers.
*   $\mathcal{L}_S$: Storage Layers.
*   $\mathcal{L}_R$: Reconstruction Layers.
*   $K^i \in \mathbb{R}^{s\times d}$: Key cache of layer $i$.
*   $V^i \in \mathbb{R}^{s\times d}$: Value cache of layer $i$.
*   $F_i$: Parameterized reconstruction function for layer $i$.
*   $\Phi(i)$: Source Layer Mapping Function specifying a set of source storage layer indices for reconstruction layer $i$.
*   $\theta_i$: Trainable parameters associated with the i-th reconstruction layer.
*   Equation 1 (Framework Definition): $(K^i, V^i) = F_i(\{(K^j, V^j)|j \in \Phi(i)\}; \theta_i)$
*   $\phi(i) \in \mathcal{L}_S$: Selector mapping for direct cache reuse.
*   Equation 2 (Direct Cache Reuse): $(K^i, V^i) = (K^{\phi(i)}, V^{\phi(i)})$, where $\phi(i) \in \mathcal{L}_S$.
*   $a_{ij}, b_{ij}$: Learnable weights for Key and Value fusion, which can be scalars ($\mathbb{R}$), vectors ($\mathbb{R}^d$), or matrices ($\mathbb{R}^{s\times d}$).
*   $\odot$: Hadamard product.
*   Equation 3 (Weighted Fusion of Caches): $K^i = \sum_{j \in \Phi(i)} a_{ij} \odot K^j , V^i = \sum_{j \in \Phi(i)} b_{ij} \odot V^j$
*   $n$: Middle layer index.
*   Equation 4 (FusedKV Key): $K^i = a_{i,1} \odot K^1 + a_{i,n} \odot K^n, i > n$
*   Equation 5 (FusedKV Value): $V^i = b_{i,1} \odot V^1 + b_{i,n} \odot V^n, i > n$
*   Equation 6 (FusedKV-Lite): $K^i = K^n, V^i = V^1, i > n$
*   $w_j = [w_{2j}, w_{2j+1}]^T$: Learnable weight vector in the j-th 2D subspace.
*   Equation 7 (RoPE Attention Score Decomposition): 
    $A_j = \frac{w_{2j} + w_{2j+1}}{2} [(q_{m,2j}k_{n,2j} + q_{m,2j+1}k_{n,2j+1}) \cos((m-n)\theta_j) + (q_{m,2j}k_{n,2j+1} - q_{m,2j+1}k_{n,2j}) \sin((m-n)\theta_j)] + \frac{w_{2j} - w_{2j+1}}{2} [(q_{m,2j}k_{n,2j} - q_{m,2j+1}k_{n,2j+1}) \cos((m+n)\theta_j) - (q_{m,2j}k_{n,2j+1} + q_{m,2j+1}k_{n,2j}) \sin((m+n)\theta_j)]$
*   Symmetry constraint: $w_{2j} = w_{2j+1}$
*   $\tilde{k}_s = \sum_{i=1}^N (w_n^i \odot \tilde{k}_n^i)$: Fused key vector from N different storage layers.
*   Equation 8 (Weighted Fusion Preserves RoPE): $\tilde{q}_m^T \tilde{k}_s = \tilde{q}_m^T (\sum_{i=1}^N (w_n^i \odot \tilde{k}_n^i)) = \sum_{i=1}^N \tilde{q}_m^T(w_n^i \odot \tilde{k}_n^i)$
*   Iterative Fusion with Auxiliary Caches Update Rules:
    $K^{n+1} = a_{n,1} \odot K^1 + a_{n+1,n} \odot K^n$
    $V^{n+1} = b_{n+1,1} \odot V^1 + b_{n+1,n} \odot V^n$
    $K^i = a_{i,i-1} \odot K^{i-1} + a_{i,n} \odot K^n, i > n+1$
    $V^i = b_{i,i-1} \odot V^{i-1} + b_{i,1} \odot V^1, i > n+1$
*   Equivalent Initialization Deterministic Computations:
    For key weights:
    $a_{i,1} = \begin{cases} a'_{n+1,1} & i = n+1 \\ a'_{i,i-1} \odot a_{i-1,1} & i > n+1 \end{cases}$
    $a_{i,n} = \begin{cases} a'_{n+1,n} & i = n+1 \\ a'_{i,i-1} \odot a_{i-1,n} + a'_{i,n} & i > n+1 \end{cases}$
    For value weights:
    $b_{i,1} = \begin{cases} b'_{n+1,1} & i = n+1 \\ b'_{i,i-1} \odot b_{i-1,1} + b'_{i,1} & i > n+1 \end{cases}$
    $b_{i,n} = \begin{cases} b'_{n+1,n} & i = n+1 \\ b'_{i,i-1} \odot b_{i-1,n} & i > n+1 \end{cases}$
*   Detailed Derivation of RoPE Compatibility (from Appendix A.4):
    $q_{m,j} = \begin{pmatrix} \cos(m\theta_j) & -\sin(m\theta_j) \\ \sin(m\theta_j) & \cos(m\theta_j) \end{pmatrix} \begin{pmatrix} q_{m,2j} \\ q_{m,2j+1} \end{pmatrix} = \begin{pmatrix} q_{m,2j} \cos(m\theta_j) - q_{m,2j+1} \sin(m\theta_j) \\ q_{m,2j} \sin(m\theta_j) + q_{m,2j+1} \cos(m\theta_j) \end{pmatrix}$
    $k_{n,j} = \begin{pmatrix} \cos(n\theta_j) & -\sin(n\theta_j) \\ \sin(n\theta_j) & \cos(n\theta_j) \end{pmatrix} \begin{pmatrix} k_{n,2j} \\ k_{n,2j+1} \end{pmatrix} = \begin{pmatrix} k_{n,2j} \cos(n\theta_j) - k_{n,2j+1} \sin(n\theta_j) \\ k_{n,2j} \sin(n\theta_j) + k_{n,2j+1} \cos(n\theta_j) \end{pmatrix}$
    Expansion of the dot product:
    $A_j = (q_{m,2j} \cos(m\theta_j) - q_{m,2j+1} \sin(m\theta_j)) \cdot w_{2j}(k_{n,2j} \cos(n\theta_j) - k_{n,2j+1} \sin(n\theta_j)) + (q_{m,2j} \sin(m\theta_j) + q_{m,2j+1} \cos(m\theta_j)) \cdot w_{2j+1}(k_{n,2j} \sin(n\theta_j) + k_{n,2j+1} \cos(n\theta_j))$
    $= q_{m,2j}k_{n,2j} (w_{2j} \cos(m\theta_j) \cos(n\theta_j) + w_{2j+1} \sin(m\theta_j) \sin(n\theta_j)) + q_{m,2j+1}k_{n,2j+1} (w_{2j} \sin(m\theta_j) \sin(n\theta_j) + w_{2j+1} \cos(m\theta_j) \cos(n\theta_j)) + q_{m,2j}k_{n,2j+1} (w_{2j+1} \sin(m\theta_j) \cos(n\theta_j) - w_{2j} \cos(m\theta_j) \sin(n\theta_j)) + q_{m,2j+1}k_{n,2j} (w_{2j+1} \cos(m\theta_j) \sin(n\theta_j) - w_{2j} \sin(m\theta_j) \cos(n\theta_j))$
    Using identities:
    $w_{2j} \cos(m\theta_j) \cos(n\theta_j) + w_{2j+1} \sin(m\theta_j) \sin(n\theta_j) = \frac{w_{2j} + w_{2j+1}}{2} \cos((m-n)\theta_j) + \frac{w_{2j} - w_{2j+1}}{2} \cos((m+n)\theta_j)$
    $w_{2j} \sin(m\theta_j) \sin(n\theta_j) + w_{2j+1} \cos(m\theta_j) \cos(n\theta_j) = \frac{w_{2j} + w_{2j+1}}{2} \cos((m-n)\theta_j) - \frac{w_{2j} - w_{2j+1}}{2} \cos((m+n)\theta_j)$
    $w_{2j+1} \sin(m\theta_j) \cos(n\theta_j) - w_{2j} \cos(m\theta_j) \sin(n\theta_j) = \frac{w_{2j+1} + w_{2j}}{2} \sin((m-n)\theta_j) + \frac{w_{2j+1} - w_{2j}}{2} \sin((m+n)\theta_j)$
    $w_{2j+1} \cos(m\theta_j) \sin(n\theta_j) - w_{2j} \sin(m\theta_j) \cos(n\theta_j) = \frac{-w_{2j+1} + w_{2j}}{2} \sin((m-n)\theta_j) + \frac{w_{2j+1} - w_{2j}}{2} \sin((m+n)\theta_j)$

8. **Algorithm Pseudocode**
The paper does not provide explicit algorithm pseudocode boxes, but details the algorithmic process for Reconstruction Function and Training/Initialization Strategies, which are transcribed exactly as follows:

**Reconstruction Function Process:**
For any reconstruction layer $i \in \mathcal{L}_R$:
1. Identify the set of source storage layer indices $\Phi(i)$.
2. Retrieve the KV caches $\{(K^j, V^j)|j \in \Phi(i)\}$.
3. Apply the parameterized reconstruction function $F_i$ with trainable parameters $\theta_i$:
   $(K^i, V^i) = F_i(\{(K^j, V^j)|j \in \Phi(i)\}; \theta_i)$

**Direct Cache Reuse Strategy:**
$(K^i, V^i) = (K^{\phi(i)}, V^{\phi(i)})$, where $\phi(i) \in \mathcal{L}_S$

**Weighted Fusion of Caches Strategy:**
$K^i = \sum_{j \in \Phi(i)} a_{ij} \odot K^j , V^i = \sum_{j \in \Phi(i)} b_{ij} \odot V^j$

**FusedKV Process:**
For layers $i > n$:
$K^i = a_{i,1} \odot K^1 + a_{i,n} \odot K^n$
$V^i = b_{i,1} \odot V^1 + b_{i,n} \odot V^n$

**FusedKV-Lite Process:**
For layers $i > n$:
$K^i = K^n$
$V^i = V^1$

**Iterative Fusion with Auxiliary Caches Process:**
1. For the first reconstruction layer ($i = n + 1$):
   $K^{n+1} = a_{n,1} \odot K^1 + a_{n+1,n} \odot K^n$
   $V^{n+1} = b_{n+1,1} \odot V^1 + b_{n+1,n} \odot V^n$
2. For all subsequent reconstruction layers ($i > n + 1$):
   $K^i = a_{i,i-1} \odot K^{i-1} + a_{i,n} \odot K^n$
   $V^i = b_{i,i-1} \odot V^{i-1} + b_{i,1} \odot V^1$
3. Initialization: Draw all learnable d-dimensional weight vectors ($a, b$) from $\mathcal{N}(0, 1)$.

**Standard FusedKV with Equivalent Initialization Process:**
1. Sample auxiliary weights from $\mathcal{N}(0, 1)$ for all reconstruction layers $i > n$:
   $a'_{i,i-1}, a'_{i,n}, b'_{i,i-1}, b'_{i,1} \sim \mathcal{N}(0, 1)$
2. Compute final model weights for key weights deterministically:
   $a_{i,1} = \begin{cases} a'_{n+1,1} & i = n+1 \\ a'_{i,i-1} \odot a_{i-1,1} & i > n+1 \end{cases}$
   $a_{i,n} = \begin{cases} a'_{n+1,n} & i = n+1 \\ a'_{i,i-1} \odot a_{i-1,n} + a'_{i,n} & i > n+1 \end{cases}$
3. Compute final model weights for value weights deterministically:
   $b_{i,1} = \begin{cases} b'_{n+1,1} & i = n+1 \\ b'_{i,i-1} \odot b_{i-1,1} + b'_{i,1} & i > n+1 \end{cases}$
   $b_{i,n} = \begin{cases} b'_{n+1,n} & i = n+1 \\ b'_{i,i-1} \odot b_{i-1,n} & i > n+1 \end{cases}$
4. Treat the computed ($a, b$) as independent and learnable parameters.