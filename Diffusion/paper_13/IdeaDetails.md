# Research Idea and Implementation Plan Extraction

## 1. Research Background and Existing Pain Points
**Research Background:**
In Diffusion Transformer (DiT) models, particularly for video generation, attention latency is a major bottleneck due to the long sequence length and the quadratic complexity. Among the operations in Transformers, attention is the only one with quadratic computation complexity, while others mostly scale linearly with the sequence length $N$. In DiT models, especially for video generation, the sequence length typically ranges from 10K to 100K, making attention the primary computational bottleneck. Reducing the cost of attention is therefore critical for improving the efficiency of DiT models. Existing efficient attention methods for DiTs fall into two main categories: (1) numerous sparse attention methods, which compute only a subset of attention scores, and (2) a few linear attention methods, which reformulate the operation to achieve $O(N)$ complexity.

**Existing Pain Points:**
Despite recent progress, both approaches face challenges in substantially reducing attention computation:
*   **(L1) Linear attention methods often fail in practice, especially on video diffusion models.** Existing work on linear attention in diffusion is rare and primarily limited to image generation. Experiments show that when applied to diffusion models, particularly video generation, linear attention severely degrades video quality.
*   **(L2) Sparse attention methods rarely achieve very high sparsity and require a considerable fraction of the full complexity of attention.** In practice, they typically reach only 40–60% sparsity for sequence length below 50K. Although some recent works report sparsity of 80–85%, such results are obtained on very long sequences (e.g., 100K–300K), where achieving high sparsity is easier. Skipping the smallest weights (e.g., 45%) introduces a relative L1 error of less than 3% compared to the full attention output. In contrast, retaining only the largest 8.1% of weights (sparsity = 92%) leads to a sharp increase in error, reaching about 33%. This explains why existing sparse attention methods struggle to achieve a sparsity beyond 90%. The intermediate values between $1/(100N)$ and $1/N$ present a dilemma: omitting them introduces significant accuracy loss, yet computing them with full attention causes a great decrease in sparsity.

## 2. Core Research Motivation and Scientific Questions
**Core Research Motivation:**
We find that attention weights in diffusion transformers can be decomposed into two matrices: a small fraction of large weights with high rank and a large fraction of the remaining weights with extremely low rank. This explains why sparse attention or linear attention alone cannot achieve satisfactory results and naturally suggests applying sparse acceleration to the first part and low-rank acceleration to the second. The full attention weights can be decoupled into two parts: (1) a small subset (< 10%) with rank comparable to full attention, and (2) a large subset (> 90%) with very low rank. Since the methods for accelerating attention focus mainly on sparsity or low-rank structure, this suggests a natural and elegant strategy: apply sparse attention to the first part and low-rank approximation to the second. Previous failures of linear attention are largely due to the high rank of full attention weights, while linear attention is restricted to a rank at most $d$. After removing the top values in the attention weights $P$, the remaining matrix becomes extremely low-rank.

**Scientific Questions:**
How to effectively fuse sparse attention and linear attention within a unified trainable framework to handle the different rank characteristics of attention weights, thereby achieving very high sparsity (e.g., 95%) without degrading video generation quality?

## 3. Overall Core Idea and Design Philosophy
**Overall Core Idea:**
We propose SLA (Sparse-Linear Attention), a trainable attention method that fuses sparse and linear attention to accelerate diffusion models. SLA classifies attention weights into critical, marginal, and negligible, applying $O(N^2)$ attention to critical weights, $O(N)$ attention to marginal weights, and skipping negligible ones. Specifically, attention weights are partitioned into blocks and dynamically classified into three categories: critical, marginal, and negligible. Critical blocks are computed exactly using FlashAttention, negligible blocks are skipped, and, unlike existing methods, marginal blocks are processed with linear attention.

**Design Philosophy:**
Linear attention in SLA does not approximate the output corresponding to marginal attention weights, but serves as a learnable compensation that enhances the effectiveness of sparse attention. This is because linear attention alone struggles to approximate the output of full attention. Therefore, we need to fine-tune the parameters of the target model, enabling it to adapt to the use of linear attention. This design allows sparsity to increase dramatically (e.g., 70%→95%) while maintaining accuracy. Since linear attention is computationally negligible, costing less than 0.5% of full attention in video generation models, SLA is several times faster than sparse attention alone.

## 4. Core Innovation Points
1.  **Observation of Attention Weight Decomposition:** We find that the attention weights in diffusion models can be perfectly decomposed into two parts: a highly sparse matrix with high rank and a dense matrix with very low rank.
2.  **First Fusion of Sparse and Linear Attention:** We propose the first attention method that effectively fuses sparse attention and linear attention.
3.  **High Sparsity with Quality Preservation:** Our method achieves about 95% attention sparsity, corresponding to approximately a 20× reduction in attention computation, while maintaining video generation quality.
4.  **Efficient GPU Kernel Implementation:** We implement efficient GPU kernels for SLA, which combines forward and backward computations into a single GPU kernel, yielding significant end-to-end acceleration.

## 5. Overview of the Overall Technical Solution
SLA first predicts a compressed attention weights matrix $P_c$ by computing $P_c = \text{Softmax}(\text{pool}(Q)\text{pool}(K)^\top/\sqrt{d})$. For each element of $P_c$, it is classified into three types and recorded in a compressed mask $M_c$: the top $k_h$\% positions are marked as critical (labeled 1), the bottom $k_l$\% positions as negligible (labeled -1), and the remaining positions as marginal (labeled 0).

Guided by the mask $M_c$, different methods are applied:
*   For $M_c[i, j] = 1$ (critical), sparse FlashAttention is used to compute the sparse attention output $O^s$.
*   For $M_c[i, j] = 0$ (marginal), linear attention is used to compute the linear attention output $O^l$.
*   For $M_c[i, j] = -1$ (negligible), computation is skipped.

Finally, the overall attention output of SLA is defined as $O = O^s + \text{Proj}(O^l)$, where $\text{Proj}$ is a learnable linear transformation $R^d \to R^d$. To apply SLA to a diffusion model, we simply replace the original attention with SLA and fine-tune the model for a few steps on a dataset consistent with the pretraining data.

## 6. Detailed Module Design
### Attention Weight Prediction
SLA first predicts a compressed attention weights matrix $P_c \in R^{N/b_q \times N/b_{kv}}$:
$P_c = \text{Softmax}(\text{pool}(Q)\text{pool}(K)^\top/\sqrt{d})$
where $\text{pool}(\cdot)$ is a mean pooling operator along the token dimension.

### Mask Classification
For each element of $P_c$, classify it into three types and record the results in a compressed mask $M_c \in R^{N/b_q \times N/b_{kv}}$. Specifically, the top $k_h$\% positions are marked as critical (labeled 1), the bottom $k_l$\% positions as negligible (labeled -1), and the remaining positions as marginal (labeled 0).

### Sparse Attention Component
Guided by the mask $M_c$, sparse FlashAttention is used to compute the sparse attention output. For each $Q$ block $Q_i$, iterate over all $K, V$ blocks $K_j, V_j$ with $j = 0, \ldots, N/b_{kv}$. Whenever $M_c[i, j] = 1$, perform:
$S_{ij} = Q_i K_j^\top / \sqrt{d}$
$P_{ij} = \text{OnlineSoftmax}(S_{ij})$
$O_i^s = O_i^s + P_{ij} V_j$
The initial value of each $O_i^s$ is set to zero. The final output of the sparse attention component is $O^s$.

### Linear Attention Component
Inspired by the idea of low-rank approximation, replace the low-rank component $P \odot (1 - M)$ with linear attention. The entries of 0 in $M_c$ determine the blocks processed by linear attention. For each query block $Q_i$, compute the corresponding linear attention output:
$H_i = \sum_{j:M_c[i,j]=0} \phi(K_j)^\top V_j$
$Z_i = \sum_{j:M_c[i,j]=0} \text{rowsum}(\phi(K_j)^\top)$
$O_i^l = \frac{\phi(Q_i) H_i}{\phi(Q_i) Z_i}$
Here, $\phi(\cdot)$ denotes the activation function (softmax in the optimal configuration), and $H_i \in R^{d \times d}, Z_i \in R^{d \times 1}$ are intermediate results. The final output of this component is $O^l$.

### Output Fusion
The overall attention output of SLA is defined as:
$O = O^s + \text{Proj}(O^l)$
where $\text{Proj}$ is a learnable linear transformation $R^d \to R^d$.

### Backward Pass
**Sparse attention gradients:** The output gradient $dO^s$ is backpropagated to compute $dQ, dK$, and $dV$.
**Linear attention gradients:** The gradient $dO^l$ yields $dQ_\phi, dK_\phi, dV$ through the chain rule. $dK_j^\phi$ and $dV_j$ are obtained by aggregating $dH_i$ and $dZ_i$. Similar to the forward pass, each $dH_i$ and $dZ_i$ is precomputed so that the remaining computation reduces to simple matrix additions.

### Additional Efficiency Optimization
*   **Lookup table:** When $M_c$ is highly sparse (> 90%), preprocess the nonzero positions of each row and column and store them in a lookup table to reduce memory traffic.
*   **Pre-aggregation for linear attention:** Precompute the row/column sums $\sum_j h_j$ and $\sum_j z_j$, and then subtract the contributions corresponding to $M_c[i, j] \neq 0$.
*   **Method of Four Russians:** When the number of blocks with $M_c[i, j] = 0$ is around 50%, group $h_j$ and $z_j$ into segments of $g$ consecutive blocks and precompute all $2^g$ possible subset sums.

## 7. All Mathematical Formulas and Symbol Definitions
**Standard Attention:**
$S = QK^\top/\sqrt{d}$
$P = \text{Softmax}(S)$
$O = PV$

**Block Sparse Attention:**
$P \leftarrow P \odot M$, where $\odot$ is the element-wise product and $M \in \{0, 1\}^{N \times N}$ is the mask.

**Linear Attention:**
$H = \phi(K)^\top V$
$Z = \text{rowsum}(\phi(K)^\top) \in R^{d \times 1}$
$O = \frac{\phi(Q)H}{\phi(Q)Z}$

**Equation 1: Attention Weight Decomposition**
$P = \underbrace{P \odot M}_{\text{sparse component}} + \underbrace{P \odot (1 - M)}_{\text{low-rank component}}$

**Equation 2: Compressed Attention Weights**
$P_c = \text{Softmax}(\text{pool}(Q)\text{pool}(K)^\top/\sqrt{d})$

**Equation 3: Mask Classification**
$M_c[i, j] = \begin{cases} 1 & \text{(top } k_h\%) \\ -1 & \text{(bottom } k_l\%) \\ 0 & \text{(otherwise)} \end{cases}$

**Equation 4: Sparse Attention in SLA**
$S_{ij} = Q_i K_j^\top / \sqrt{d}$
$P_{ij} = \text{OnlineSoftmax}(S_{ij})$
$O_i^s = O_i^s + P_{ij} V_j$

**Equation 5: Linear Attention in SLA**
$H_i = \sum_{j:M_c[i,j]=0} \phi(K_j)^\top V_j$
$Z_i = \sum_{j:M_c[i,j]=0} \text{rowsum}(\phi(K_j)^\top)$
$O_i^l = \frac{\phi(Q_i) H_i}{\phi(Q_i) Z_i}$

**Equation 6: Overall Attention Output**
$O = O^s + \text{Proj}(O^l)$

**Equation 7: Sparse Attention Gradients**
$dP_{ij} = dO_{ij}^s V_j^\top$
$D_i^s = \text{rowsum}(dO_i^s \odot O_i^s)$
$dS_{ij} = P_{ij} \odot (dP_{ij} - D_i^s)$
$dQ_i = dS_{ij} K_j$
$dK_j = dS_{ij}^\top Q_i$
$dV_j = P_{ij}^\top dO_i^s$

**Equation 8: Linear Attention Gradients**
$dH_i = \left(\frac{Q_i^\phi}{Q_i^\phi Z_i}\right)^\top dO_i^l$
$D_i^l = \text{rowsum}(dO_i^l \odot O_i^l)$
$dZ_i = -\left(\frac{Q_i^\phi}{Q_i^\phi Z_i}\right)^\top D_i^l$
$dQ_i^\phi = \frac{dO_i^l(H_i)^\top - D_i^l Z_i^\top}{Q_i^\phi Z_i}$
$dK_j^\phi = V_j(dH_i)^\top + (dZ_i)^\top$
$dV_j = K_j^\phi dH_i$

**Symbol Definitions:**
*   $Q, K, V \in R^{N \times d}$: Queries, keys, and values.
*   $N$: Sequence length.
*   $d$: Feature dimension.
*   $S$: Score matrix.
*   $P$: Attention weights.
*   $M \in \{0, 1\}^{N \times N}$: Element-wise sparse mask.
*   $\phi(\cdot)$: Feature map / activation function for linear attention (e.g., softmax, ELU+1, ReLU).
*   $Q_i \in R^{b_q \times d}, K_j, V_j \in R^{b_{kv} \times d}$: Blocks of Q, K, V.
*   $P_c, M_c$: Compressed attention weights and compressed mask.
*   $\text{pool}(\cdot)$: Mean pooling operator along the token dimension.
*   $\text{Proj}$: Learnable linear transformation $R^d \to R^d$.
*   $O^s, O^l$: Outputs of sparse and linear attention components.
*   $k_h\%$: Percentage of critical positions (top).
*   $k_l\%$: Percentage of negligible positions (bottom).

## 8. Algorithm Pseudocode

**Algorithm 1: Forward pass of SLA.**
1: Input: Matrices $Q, K, V, Q_\phi, K_\phi \in R^{N \times d}$, block sizes $b_q, b_{kv}$, hyper-parameters $k_h, k_l$.
2: Divide $Q, Q_\phi$ to $T_m = N/b_q$ blocks $\{Q_i\}$ and $\{Q_\phi^i\}$;
3: Divide $K, V, K_\phi$ to $T_n = N/b_{kv}$ blocks $\{K_i\}, \{V_i\}$ and $\{K_\phi^i\}$;
4: $h = \{h_j\} = \{(K_\phi^j)^\top V_j\}$; $z = \{z_j\} = \{\text{rowsum}((K_\phi^j)^\top)\}$; // Precompute for linear attention
5: $P_c = \text{Softmax}(\text{pool}(Q)\text{pool}(K)^\top/\sqrt{d})$; Initialize $M_c = 0$;
6: $M_c[i, j] = 1$ if $P_c[i, j] \in \text{TopK}(P_c[i, :], k_h)$; $M_c[i, j] = 0$ if $P_c[i, j] \in \text{BottomK}(P_c[i, :], k_l)$;
7: for $i = 1$ to $T_m$ do
8:   for $j = 1$ to $T_n$ do
9:     if $M_c[i, j] = 1$ then
10:       $S_{ij} = Q_i K_j^\top / \sqrt{d}$; $m_{ij} = \max(m_{i,j-1}, \text{rowmax}(S_{ij}))$; $P_{ij} = \exp(S_{ij} - m_{ij})$;
11:       $l_{ij} = e^{m_{i,j-1} - m_{ij}} l_{i,j-1} + \text{rowsum}(P_{ij})$; $O_{ij}^s = \text{diag}(e^{m_{i,j-1} - m_{ij}}) O_{i,j-1}^s + P_{ij} V_j$;
12:     else if $M_c[i, j] = 0$ then
13:       $H_i \leftarrow H_i + h_j$; $Z_i \leftarrow Z_i + z_j$;
14:     end if
15:   end for
16:   $O_i^s = \text{diag}(l_i^{T_n})^{-1} O_{i,T_n}^s$; $O_i^l = Q_\phi^i H_i / (Q_\phi^i Z_i)$; $L_i = m_{i,T_n} + \log(l_{i,T_n})$;
17: end for
18: return $O^s = \{O_i^s\}, O^l = \{O_i^l\}$;

**Algorithm 2: Backward pass of SLA.**
1: Input: $Q, K, V, Q_\phi, K_\phi, M_c, \{L_i\}, \{H_i\}, \{Z_i\}, O^s, O^l$ from the forward, $dO^s, dO^l \in R^{N \times d}$.
2: $D^s = \text{rowsum}(dO^s \odot O^s)$, $D^l = \text{rowsum}(dO^l \odot O^l)$, divide $D^s, D^l$ into $T_m$ blocks $\{D_i^s\}, \{D_i^l\}$;
3: for $i = 1$ to $T_m$ do
4:   $dH_i = (Q_\phi^i / (Q_\phi^i Z_i))^\top dO_i^l$; $dZ_i = -(Q_\phi^i / (Q_\phi^i Z_i))^\top D_i^l$;
5:   $dQ_\phi^i = (dO_i^l(H_i)^\top - D_i^l Z_i^\top) / (Q_\phi^i Z_i)$;
6: end for
7: for $j = 1$ to $T_n$ do
8:   Initialize $dH = 0, dZ = 0$;
9:   for $i = 1$ to $T_m$ do
10:    if $M_c[i, j] = 1$ then
11:      $S_{ij} = Q_i K_j^\top / \sqrt{d}$; $P_{ij} = \exp(S_{ij} - L_i)$; $dV_j \leftarrow dV_j + P_{ij}^\top dO_i^s$; $dP_{ij} = dO_{ij}^s V_j^\top$;
12:      $dS_{ij} = P_{ij} \odot (dP_{ij} - D_i^s)$; $dQ_i \leftarrow dQ_i + dS_{ij} K_j$; $dK_j \leftarrow dK_j + dS_{ij}^\top Q_i$;
13:    else if $M_c[i, j] = 0$ then
14:      $dH \leftarrow dH + dH_i$; $dZ \leftarrow dZ + dZ_i$;
15:    end if
16:  end for
17:  $dK_\phi^j = V_j(dH)^\top + (dZ)^\top$; $dV_j = K_\phi^j dH$;
18: end for
19: return $dQ = \{dQ_i\}, dK = \{dK_i\}, dV = \{dV_i\}, dQ_\phi = \{dQ_\phi^i\}, dK_\phi = \{dK_\phi^i\}$;