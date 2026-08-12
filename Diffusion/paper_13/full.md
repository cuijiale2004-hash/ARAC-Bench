## ABSTRACT

In Diffusion Transformer (DiT) models, particularly for video generation, attention latency is a major bottleneck due to the long sequence length and the quadratic complexity. Interestingly, we find that attention weights can be decoupled into two matrices: a small fraction of large weights with high rank and the remaining weights with very low rank. This naturally suggests applying sparse acceleration to the first part and low-rank acceleration to the second. Based on this finding, we propose SLA (Sparse-Linear Attention), a trainable attention method that fuses sparse and linear attention to accelerate diffusion models. SLA classifies attention weights into critical, marginal, and negligible, applying (N<sup>2</sup>) attention to critical weights, (N) attention to marginal weights, and skipping negligible ones. SLA combines these computations into a single GPU kernel and supports both forward and backward passes. With only a few fine-tuning steps using SLA, DiT models achieve about 20 reduction in attention computation, resulting in significant acceleration without loss of generation quality. Experiments show that SLA reduces attention computation by about 95% without degrading end-to-end generation quality, outperforming baseline methods. In addition, we implement an efficient GPU kernel for SLA, which yields a 13.7 speedup in attention computation and a 2.2 end-to-end speedup in video generation on Wan2.1-1.3B. The code is available at https://github.com/thu-ml/SLA.

## 1 INTRODUCTION

Among the operations in Transformers, attention (Vaswani et al., 2017) is the only one with quadratic computation complexity, while others mostly scale linearly with the sequence length N. In Diffusion Transformer (DiT) models (Peebles & Xie, 2022), especially for video generation, attention becomes the primary computational bottleneck, as the sequence length typically ranges from 10K to 100K. Reducing the cost of attention is therefore critical for improving the efficiency of DiT models. Existing efficient attention methods for DiTs fall into two main categories: (1) numerous sparse attention methods (Li et al., 2025; Zhang et al., 2025b; Xi et al., 2025; Yang et al., 2025a; Zhang et al., 2025e; Wu et al., 2025; Shen et al., 2025; Hassani et al., 2023; Liu et al., 2025), which compute only a subset of attention scores, and (2) a few linear attention methods (Xie et al., 2024; Zhu et al., 2025), which reformulate the operation to achieve (N) complexity.

Limitation. Despite recent progress, both approaches face challenges in substantially reducing attention computation: (L1) Linear attention methods often fail in practice, especially on video diffusion models. Existing work on linear attention in diffusion is rare and primarily limited to image generation. Our experiments show that when applied to diffusion models, particularly video generation, linear attention severely degrades video quality. (L2) Sparse attention methods rarely achieve very high sparsity and require a considerable fraction of the full complexity of attention. In practice, they typically reach only 40–60% sparsity for sequence length below 50K. Although some recent works (Yang et al., 2025a; Li et al., 2025) report sparsity of 80–85%, such results are obtained on very long sequences (e.g., 100K–300K), where achieving high sparsity is easier.

Key Observation. We find that attention weights in diffusion transformers can be decomposed into two matrices: a small fraction of large weights with high rank and a large fraction of the remaining weights with extremely low rank. This explains why sparse attention or linear attention alone cannot achieve satisfactory results and naturally suggests applying sparse acceleration to the first part and low-rank acceleration to the second.

Our Method. Based on the observation above, we propose SLA, a trainable hybrid sparse and linear attention for DiT models. Specifically, attention weights are partitioned into blocks and dynamically classified into three categories: critical, marginal, and negligible. Critical blocks are computed exactly using FlashAttention, negligible blocks are skipped, and, unlike existing methods, marginal blocks are processed with linear attention. This design allows sparsity to increase dramatically (e.g., 70% 95%) while maintaining accuracy. Since linear attention is computationally negligible, costing less than 0.5% of full attention in video generation models, SLA is several times faster than sparse attention alone. Furthermore, we implement efficient forward and backward passes for SLA. With a few steps of fine-tuning, SLA significantly reduces the computation complexity and latency of attention while preserving the quality of the generation results.

Result. SLA reduces attention computation by about 95% without degrading video generation quality, even at a moderate sequence length of 30K, which is the sequence length in Wan2.1-1.3B. In addition, our implementation achieves a 13.7 speedup in the attention kernel and a 2.2 endto-end acceleration for video generation, where the attention time becomes almost negligible. SLA consistently surpasses baselines in both generation quality and efficiency.

Contribution. We summarize our main contributions as follows. (1) We find that the attention weights in diffusion models can be perfectly decomposed into two parts: a highly sparse matrix with high rank and a dense matrix with very low rank. (2) We propose the first attention method that effectively fuses sparse attention and linear attention. (3) Our method achieves about 95% attention sparsity, corresponding to approximately a 20× reduction in attention computation, while maintaining video generation quality. (4) We implement efficient GPU kernels for SLA.

## 2 PRELIMINARY

## 2.1 BLOCK SPARSE ATTENTION

Given queries, keys, and values $Q , K , V \in \mathbb { R } ^ { N \times d }$ , the standard attention computes the score matrix $S = Q K ^ { \top } / \sqrt { d }$ and the attention weights $P = \mathrm { S o f t m a x } ( S )$ to obtain the output $O = P V$ . This is inefficient for large N as it requires $\bar { \mathcal { O } } ( N ^ { 2 } d )$ operations. The idea of sparse attention is to reduce computation by applying a mask $M \in \dot { \{ 0 , 1 \} } ^ { N \times N }$ to the attention weights: $P  P \odot M$ , where is the element-wise product. A common strategy is to choose a threshold τ and set $M _ { i j } = 1$ if $P _ { i j } > \tau$ . For entries with $M _ { i j } = 0$ , the multiplications $Q _ { i } K _ { j } ^ { \top }$ and $P _ { i j } V _ { j }$ can be skipped, where $Q _ { i } = Q [ i , : ] , K _ { j } = K [ j , : ] , V _ { j } = V [ j , : ]$

However, element-wise sparse attention is inefficient on modern GPUs. Practical implementations such as FlashAttention (Dao, 2023) operate at the block level. Specifically, the sparse FlashAttention first partitions $Q , K , V , S , P , M$ into blocks $\{ \mathbf { Q } _ { i } \} , \{ \mathbf { K } _ { j } \} , \{ \mathbf { V } _ { j } \} , \{ \mathbf { S } _ { i j } \} , \{ \mathbf { P } _ { i j } \} , \{ \mathbf { M } _ { i j } \}$ , where $\mathbf { Q } _ { i } \in$ $\mathbb { R } ^ { b _ { q } \times d } , { \bf K } _ { j } , { \bf V } _ { j } \in \mathbb { R } ^ { b _ { k v } \times d }$ , and $\mathbf { S } _ { i j } , \mathbf { P } _ { i j } , \mathbf { M } _ { i j } \in \mathbb { R } ^ { b _ { q } \times \bar { b } _ { k v } }$ . Each block mask $\mathbf { M } _ { i j }$ is fully filled with either 0 or 1, and we skip the computations of $\mathbf { Q } _ { i } \mathbf { K } _ { j } ^ { \top }$ and $\mathbf { P } _ { i j } \mathbf { V } _ { j } \mathrm { i f } \mathbf { M } _ { i j } [ : , : ] = \bar { 0 }$

## 2.2 LINEAR ATTENTION

Linear attention methods reduce the complexity of standard attention from $\mathcal { O } ( N ^ { 2 } d )$ to $\mathcal { O } ( N d ^ { 2 } )$ . A key idea is to decouple the softmax operation by introducing a feature map $\phi ( \cdot )$ applied to Q and K. Specifically, it replaces the attention weights in standard attention with $\frac { \phi ( Q ) \phi ( K ) ^ { \top } } { \operatorname { r o w s u m } ( \phi ( Q ) \phi ( K ) ^ { \top } }$ . This reformulation enables reordering of the matrix multiplications: instead of explicitly computing the attention weights, it first computes $\phi ( K ) ^ { \top } V$ , and then applies this intermediate result to $\phi ( Q )$

$$
H = \phi (K) ^ {\top} V, \quad Z = \operatorname{rowsum} (\phi (K) ^ {\top}) \in \mathbb {R} ^ {d \times 1}, \quad O = \frac {\phi (Q) H}{\phi (Q) Z}.
$$

![](images/917f0bb87726251890ddafa3035403b8375e5479a7df9fff85ec3c462975a615.jpg)

![](images/5f40889d0b63f71149d128c2e2350a8ada64f69e574109dc1b0540e3f89bd806.jpg)  
Full Attention

![](images/b7222400068b79e1f47507ec31e576c00287cfadccf297f459958945f93fff84.jpg)

The mapping ϕ( ) is usually an activation function (e.g., ELU + 1 or ReLU (Clevert et al., 2016; Xavier et al., 2011)). This formulation avoids explicitly constructing the N N matrices S, P and achieves linear computational complexity.

## 3 MOTIVATION AND ANLYSIS

![](images/ccb3f14055af08c72bb0b80eca4bb181f9702ebbbcfca2ed2a3959f29b4fe115.jpg)  
Frequency of Attention Weights

![](images/975cb881a6b65c80d784444f42d519755aa61dc4ac002aca281c60af0b9b3eb9.jpg)

![](images/3bc2eeb7a28f2e45888a1ca3c1b1e666c07ad31632c8a4ed39d6cf344d75b77e.jpg)  
Figure 1: The left figure shows a typical distribution of attention weights sampled from the Wan2.1 model. The right figure shows the accuracy of sparse attention with different sparsity.

## 3.1 MOTIVATION OF SLA

Due to the softmax operator, the attention weights P lie in [0, 1] with each row summing to 1. Furthermore, because of the exponential scaling in softmax, only a small fraction of entries in P are relatively large, while the vast majority are close to zero. Figure 1 (left) shows the typical distribution of attention weights P sampled from the Wan2.1 model (Wan et al., 2025). We highlight two key observations: (1) Only about 8.1% of the weights are larger than the average value 1/N. (2) A considerable proportion of weights are extremely small. In our case, approximately 45% fall below 1/(100N). As shown in Figure 1 (right), skipping these smallest 45% of weights in sparse attention (i.e., setting the corresponding entries in M to 0) introduces a relative L1 error of less than 3% compared to the full attention output. In contrast, retaining only the largest 8.1% of weights (sparsity = 92%) leads to a sharp increase in error, reaching about 33%. This explains why existing sparse attention methods struggle to achieve a sparsity beyond 90%.

The intermediate values between 1/(100N) and 1/N (the yellow column in Figure 1) present a dilemma: omitting them introduces significant accuracy loss, yet computing them with full attention causes a great decrease in sparsity. Fortunately, these values are far less critical than the largest ones. This finding motivates us to categorize the attention weights into three types: critical, marginal, and negligible. For critical weights, we use sparse FlashAttention to compute the output as they dominate the attention distribution; For negligible weights, we skip the computation; For marginal weights, we employ a linear attention method to reduce the computational complexity to (Nd<sup>2</sup>) and enhance the performance of sparse attention.

![](images/40e94ee8c1e1a1999899622c955d4d24f18a47e5ab8156ea67b62c20ee978a07.jpg)  
(Sparsity = 0%)

![](images/6924b98a015e0e19bd1222fe7a4818441be3923c4d581ea77d7edd2c6384c373.jpg)  
(Sparsity = 100%)

![](images/77f6d0c0ae6f273e2d2325df906b24ac300ae2da6384341622ac1ff0df6965d6.jpg)  
(Sparsity = 90%)

![](images/939a60e66564c6110c9b384664fd32e0ecb140dacadb8c8463fe992c2bc8fcd6.jpg)  
(Sparsity = 95%)  
Figure 2: Video generation examples on Wan2.1 fine-tuned with full attention, linear attention, sparse attention, and SLA. SLA could achieve a high sparsity of 95% and lossless video quality.

Empirical results. In Figure 2, we present some videos generated by Wan2.1 fine-tuned with different attention methods: using only linear attention, sparse attention with 90% sparsity, and SLA with 95% sparsity. Note that the computational complexity of SLA at 95% sparsity is nearly half that of 90% sparse attention, since the cost of linear attention is almost negligible. For example, in the Wan2.1 model, linear attention accounts for less than 0.5% of the cost of full attention. These empirical results show that SLA significantly outperforms the other two methods in video quality.

## 3.2 SEPARATING ATTENTION WEIGHTS: SPARSE FEW, LOW-RANK MANY

![](images/4b0e3be45163fab97dfd6150764ba6259ae6c32e19acd13924aedc573ebadc01.jpg)  
Figure 3: Decomposition of attention weights. We sample attention weights from the Wan2.1 model: the left figure shows the full weights, the middle the top 8%, and the right the bottom 92%.

Observation. As shown in Figure 3, full attention weights can be decoupled into two parts: (1) a small subset $( < 1 0 \% )$ with rank comparable to full attention, and (2) a large subset $( > 9 0 \% )$ with very low rank. Since the methods for accelerating attention focus mainly on sparsity or low-rank structure, this suggests a natural and elegant strategy: apply sparse attention to the first part and low-rank approximation to the second.

Previous failures of linear attention are largely due to the high rank of full attention weights (Fan et al., 2025), while linear attention is restricted to a rank at most d. Figure 3 (left) illustrates this with a typical example using the notion of stable rank (Rudelson & Vershynin, 2006). We observe that after removing the top values in the attention weights $P ,$ , the remaining matrix becomes extremely low-rank. This motivates the decomposition of $P$ using the sparse mask M:

$$
P = \underbrace {P \odot M} _ {\text { sparse   component }} + \underbrace {P \odot (1 - M)} _ {\text { low - rank   component }}.\tag{1}
$$

Since linear attention is essentially a low-rank version of attention, we are provided with a possibility to replace the low-rank component $P \odot ( 1 - M )$ with linear attention.

## 4 SLA

SLA effectively integrates sparse and linear attention within a unified framework, allowing them to complement each other. In particular, we fuse both attention into a single efficient GPU kernel. In this section, we introduce the sparse and linear attention components of SLA.

SLA first predicts a compressed attention weights matrix $P _ { c } \in \mathbb { R } ^ { N / b _ { q } \times N / b _ { k v } } ;$

$$
P _ {c} = \operatorname{Softmax} (\operatorname{pool} (Q) \operatorname{pool} (K) ^ {\top} / \sqrt {d}).\tag{2}
$$

where pool( ) is a mean pooling operator along the token dimension. For each element of $P _ { c } ,$ we classify it into three types and record the results in a compressed mask $M _ { c } \in \mathbb R ^ { N / b _ { q } \times N / b _ { k v } }$ Specifically, the top ${ k _ { h } } \%$ positions are marked as critical (labeled 1), the bottom $k _ { l } \%$ positions as negligible (labeled 1), and the remaining positions as marginal (labeled 0). Formally,

$$
M _ {c} [ i, j ] = \{1 (\text { top } k _ {h} \%), - 1 (\text { bottom } k _ {l} \%), 0 (\text { otherwise }) \}.\tag{3}
$$

We apply different methods according to $M _ { c }$

## 4.1 SPARSE ATTENTION IN SLA

Guided by the mask $M _ { c } ,$ , sparse FlashAttention is used to compute the sparse attention output. For each Q block $\mathbf { Q } _ { i }$ , we iterate over all $K , V$ blocks $\mathbf { K } _ { j } , \mathbf { V } _ { j }$ with $j ~ = ~ 0 , \ldots , N / b _ { k \ i }$ . Whenever $M _ { c } [ i , j ] = 1$ , we perform:

$$
\mathbf {S} _ {i j} = \mathbf {Q} _ {i} \mathbf {K} _ {j} ^ {\top} / \sqrt {d}, \quad \mathbf {P} _ {i j} = \text { OnlineSoftmax } (\mathbf {S} _ {i j}), \quad \mathbf {O} _ {i} ^ {s} = \mathbf {O} _ {i} ^ {s} + \mathbf {P} _ {i j} \mathbf {V} _ {j}.\tag{4}
$$

Here, OnlineSoftmax( ) operator (Milakov & Gimelshein, 2018) computes the softmax of a matrix in a block-wise manner (see lines 10-11 of Algorithm 1 for implementation). The initial value of each $\mathbf { O } _ { i } ^ { s }$ is set to zero. Algorithm 1 describes the forward computation of the sparse attention component, and we denote the final output of the sparse attention component $O ^ { s }$

## 4.2 LINEAR ATTENTION IN SLA

Inspired by the idea of low-rank approximation, we replace the low-rank component $P \odot ( 1 - M )$ in Equation 1 with linear attention introduced in Section 2.2 as

$$
\frac {\phi (Q) \phi (K) ^ {\top}}{\operatorname{rowsum} (\phi (Q) \phi (K) ^ {\top})} \odot (1 - M).
$$

Specifically, the entries of 0 in M<sub>c</sub> determine the blocks processed by linear attention. For each query block $\mathbf { Q } _ { i }$ , we compute the corresponding linear attention output:

$$
\mathbf {H} _ {i} = \sum_ {j: M _ {c} [ i, j ] = 0} \phi (\mathbf {K} _ {j}) ^ {\top} \mathbf {V} _ {j}, \quad \mathbf {Z} _ {i} = \sum_ {j: M _ {c} [ i, j ] = 0} \operatorname{rowsum} (\phi (\mathbf {K} _ {j}) ^ {\top}), \quad \mathbf {O} _ {i} ^ {l} = \frac {\phi (\mathbf {Q} _ {i}) \mathbf {H} _ {i}}{\phi (\mathbf {Q} _ {i}) \mathbf {Z} _ {i}}.\tag{5}
$$

Here, as mentioned in Section 2.2, ϕ( ) denotes the activation function, and $\mathbf { H } _ { i } \in \mathbb { R } ^ { d \times d } , \mathbf { Z } _ { i } \in \mathbb { R } ^ { d \times 1 }$ are intermediate results similar to H and $Z .$ Algorithm 1 describes the forward pass of the linear attention component, and the final output of this component is denoted as $O ^ { l }$

Finally, the overall attention output of SLA is defined as:

$$
O = O ^ {s} + \operatorname{Proj} (O ^ {l}).\tag{6}
$$

where Proj is a learnable linear transformation $\mathbb { R } ^ { d }  \mathbb { R } ^ { d }$ . Applying this projection to $O ^ { l }$ helps reduce the distribution mismatch between softmax and linear attention. Its computational cost is $\mathcal { O } ( N d ^ { 2 } )$ , the same as computing $O ^ { l }$ and negligible compared with the $\mathcal { O } ( N ^ { 2 } d )$ cost of full attention. Specifically, $\mathcal { O } ( N d ^ { 2 } ) = \dot { 0 } . 0 0 \dot { 4 } \times \mathcal { O } ( N ^ { 2 } d )$ , when $N \overset { \cdot } { = } 3 2 K , d = 1 2 8$ in the Wan2.1-1.3B. In this case, 95% sparsity in the sparse attention component means 94.7% attention complexity reduction.

Insight. Linear attention in SLA does not approximate the output corresponding to marginal attention weights, but serves as a learnable compensation that enhances the effectiveness of sparse attention. This is because linear attention alone struggles to approximate the output of full attention (Choromanski et al., 2020; Qin et al., 2022). Therefore, we need to fine-tune the parameters of the target model, enabling it to adapt to the use of linear attention.

![](images/b7bbca54380b21ffb8aaf1a63750a8e5d9567cac099da71c8f30d492ca68a0e3.jpg)  
Figure 4: Overview of SLA. The left figure illustrates the high-level idea: attention weights are classified into three categories and assigned to computations of different complexity. The right figure shows the detailed forward algorithm of SLA using the predicted compressed attention weights.

## 5 FINE-TUNING USING SLA

To apply SLA to a diffusion model, we can simply replace the original attention with SLA and finetune the model for a few steps on a dataset consistent with the pretraining data. In this section, we describe the forward and backward passes of SLA. Moreover, we detail some additional efficiency optimization for SLA in Appendix A.6.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1: Forward pass of SLA.

1: Input: Matrices $Q, K, V, Q^{\phi}, K^{\phi} \in \mathbb{R}^{N \times d}$, block sizes $b_{q}, b_{kv}$, hyper-parameters $k_{h}, k_{l}$.

2: Divide $Q, Q^{\phi}$ to $T_{m} = N/b_{q}$ blocks $\{\mathbf{Q}_{i}\}$ and $\{\mathbf{Q}_{i}^{\phi}\}$;

3: Divide $K, V, K^{\phi}$ to $T_{n} = N/b_{kv}$ blocks $\{\mathbf{K}_{i}\}, \{\mathbf{V}_{i}\}$ and $\{\mathbf{K}_{i}^{\phi}\}$;

4: $h = \{h_{j}\} = \{(\mathbf{K}_{j}^{\phi})^{\top} \mathbf{V}_{j}\}; z = \{z_{j}\} = \{\text{rowsum}((\mathbf{K}_{j}^{\phi})^{\top})\}$; // Precompute for linear attention

5: $P_{c} = \text{Softmax}(\text{pool}(Q)\text{pool}(K)^{\top}/\sqrt{d})$; Initialize $M_{c} = 0$;

6: $M_{c}[i,j] = 1$ if $P_{c}[i,j] \in \text{TopK}(P_{c}[i,:], k_{h})$; $M_{c}[i,j] = 0$ if $P_{c}[i,j] \in \text{BottomK}(P_{c}[i,:], k_{l})$;

7: for $i = 1$ to $T_{m}$ do

8: for $j = 1$ to $T_{n}$ do

9: if $M_{c}[i,j] = 1$ then

10: $\mathbf{S}_{ij} = \mathbf{Q}_{i} \mathbf{K}_{j}^{\top}/\sqrt{d}$; $m_{ij} = \max(m_{i,j-1}, \text{rowmax}(\mathbf{S}_{ij}))$; $\mathbf{P}_{ij} = \exp(\mathbf{S}_{ij} - m_{ij})$;

11: $l_{ij} = e^{m_{i,j-1}-m_{ij}} l_{i,j-1} + \text{rowsum}(\mathbf{P}_{ij})$; $\mathbf{O}_{ij}^{s} = \text{diag}(e^{m_{i,j-1}-m_{ij}}) \mathbf{O}_{i,j-1}^{s} + \mathbf{P}_{ij} \mathbf{V}_{j}$;

12: else if $M_{c}[i,j] = 0$ then

13: $\mathbf{H}_{i} \leftarrow \mathbf{H}_{i} + h_{j}$; $\mathbf{Z}_{i} \leftarrow \mathbf{Z}_{i} + z_{j}$;

14: end if

15: end for

16: $\mathbf{O}_{i}^{s} = \text{diag}(l_{i}^{T_{n}})^{-1} \mathbf{O}_{i,T_{n}}^{s}$; $\mathbf{O}_{i}^{l} = \mathbf{Q}_{i}^{\phi} \mathbf{H}_{i}/(\mathbf{Q}_{i}^{\phi} \mathbf{Z}_{i})$; $\mathbf{L}_{i} = m_{i,T_{n}} + \log(l_{i,T_{n}})$;

17: end for

18: return $O^{s} = \{\mathbf{O}_{i}^{s}\}$, $O^{l} = \{\mathbf{O}_{i}^{l}\}$;
</div>

## 5.1 FORWARD PASS

The formulation of the forward computation was introduced in Section 4. The complete algorithm of the forward pass of SLA is presented in Algorithm 1. It’s worth noting that we precompute $h _ { j } = \phi ( \mathbf { K } _ { j } ) ^ { \top } \bar { \mathbf { V } _ { j } }$ and $z _ { j } = \mathrm { r o w s u m } ( \phi ( \mathbf { K } _ { j } ) ^ { \top } )$ for each pair $( K _ { j } , V _ { j } )$ (Line 4 in Algorithm 1). This design ensures that, when $M _ { c } [ i , j ] = 0$ , the corresponding operation only involves a single matrix addition (Line 13 in Algorithm 1), thereby improving efficiency. To simplify the notation, we denote $Q ^ { \phi } = \phi ( Q )$ and $K ^ { \phi } = { \phi } ( K )$ in the following.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 2: Backward pass of SLA.

1: Input: Q, K, V,  $Q^{\phi}$ ,  $K^{\phi}$ ,  $M_{c}$ ,  $\{L_{i}\}$ ,  $\{H_{i}\}$ ,  $\{Z_{i}\}$ ,  $O^{s}$ ,  $O^{l}$  from the forward,  $dO^{s}$ ,  $dO^{l} \in R^{N \times d}$ .

2:  $D^{s} = \text{rowsum}(dO^{s} \odot O^{s})$ ,  $D^{l} = \text{rowsum}(dO^{l} \odot O^{l})$ , divide  $D^{s}$ ,  $D^{l}$  into  $T_{m}$  blocks  $\{D_{i}^{s}\}$ ,  $\{D_{i}^{l}\}$ ;

3: for i = 1 to  $T_{m}$  do

4:  $\mathbf{dH}_{i} = (\mathbf{Q}_{i}^{\phi}/(\mathbf{Q}_{i}^{\phi}\mathbf{Z}_{i}))^{\top}\mathbf{dO}_{i}^{l}$ ;  $\mathbf{dZ}_{i} = -(\mathbf{Q}_{i}^{\phi}/(\mathbf{Q}_{i}^{\phi}\mathbf{Z}_{i}))^{\top}D_{i}^{l}$ ;

5:  $\mathbf{dQ}_{i}^{\phi} = (\mathbf{dO}_{i}^{l}(\mathbf{H}_{i})^{\top} - \mathbf{D}_{i}^{l}\mathbf{Z}_{i}^{\top})/(\mathbf{Q}_{i}^{\phi}\mathbf{Z}_{i})$ ;

6: end for

7: for j = 1 to  $T_{n}$  do

8: Initialize dH = 0, dZ = 0;

9: for i = 1 to  $T_{m}$  do

10: if  $M_{c}[i,j] = 1$  then

11:  $S_{ij} = Q_{i}K_{j}^{\top}/\sqrt{d}$ ;  $P_{ij} = \exp(S_{ij} - L_{i})$ ;  $dV_{j} \leftarrow dV_{j} + P_{ij}^{\top}dO_{i}^{s}$ ;  $dP_{ij} = dO_{ij}^{s}V_{j}^{\top}$ ;

12:  $dS_{ij} = P_{ij} \odot (dP_{ij} - D_{i}^{s})$ ;  $dQ_{i} \leftarrow dQ_{i} + dS_{ij}K_{j}$ ;  $dK_{j} \leftarrow dK_{j} + dS_{ij}^{\top}Q_{i}$ ;

13: else if  $M_{c}[i,j] = 0$  then

14: dH  $\leftarrow dH + dH_{i}$ ; dZ  $\leftarrow dZ + dZ_{i}$ ;

15: end if

16: end for

17:  $dK_{j}^{\phi} = V_{j}(dH)^{\top} + (dZ)^{\top}$ ;  $dV_{j} = K_{j}^{\phi}dH$ ;

18: end for

19: return  $dQ = \{dQ_{i}\}, dK = \{dK_{i}\}, dV = \{dV_{i}\}, dQ^{\phi} = \{dQ_{i}^{\phi}\}, dK^{\phi} = \{dK_{i}^{\phi}\}$ ;
</div>

## 5.2 BACKWARD PASS

The backward pass computes gradients for both the sparse and linear components, which are also fused into a single GPU kernel for efficiency.

Gradient notation. The prefix d or d is used to denote gradients, $\mathrm { e . g . , } d O ^ { s } , d O ^ { l }$ are the gradients of $O ^ { s } , O ^ { l }$ with respect to some loss function ℓ, respectively.

Sparse attention gradients. The output gradient $d O ^ { s }$ is backpropagated to compute $d Q , d K$ , and $d V$ , following the same derivation as in FlashAttention (Dao, 2023). Given $d O ^ { s }$ , the backward pass is carried out as follows:

$$
\begin{array}{r} \mathbf {d P} _ {i j} = \mathbf {d O} _ {i j} ^ {s} \mathbf {V} _ {j} ^ {\top}, \quad \mathbf {D} _ {i} ^ {s} = \mathrm{rowsum} (\mathbf {d O} _ {i} ^ {s} \odot \mathbf {O} _ {i} ^ {s}), \quad \mathbf {d S} _ {i j} = \mathbf {P} _ {i j} \odot (\mathbf {d P} _ {i j} - \mathbf {D} _ {i} ^ {s}), \\ \mathbf {d Q} _ {i} = \mathbf {d S} _ {i j} \mathbf {K} _ {j}, \quad \mathbf {d K} _ {j} = \mathbf {d S} _ {i j} ^ {\top} \mathbf {Q} _ {i}, \quad \mathbf {d V} _ {j} = \mathbf {P} _ {i j} ^ {\top} \mathbf {d O} _ {i} ^ {s}. \end{array}\tag{7}
$$

Here, we consider $\mathbf { D } _ { i } ^ { s } \in \mathbb { R } ^ { b _ { q } \times 1 }$ as a column vector.

Linear attention gradients. The gradient $d O ^ { l }$ yields $d Q ^ { \phi } , d K ^ { \phi } , d V$ through the chain rule:

$$
\begin{array}{r l} & {\mathbf {d H} _ {i} = \left(\frac {\mathbf {Q} _ {i} ^ {\phi}}{\mathbf {Q} _ {i} ^ {\phi} \mathbf {Z} _ {i}}\right) ^ {\top} \mathbf {d O} _ {i} ^ {l}, \quad \mathbf {D} _ {i} ^ {l} = \mathrm{rowsum} (\mathbf {d O} _ {i} ^ {l} \odot \mathbf {O} _ {i} ^ {l}), \quad \mathbf {d Z} _ {i} = - \left(\frac {\mathbf {Q} _ {i} ^ {\phi}}{\mathbf {Q} _ {i} ^ {\phi} \mathbf {Z} _ {i}}\right) ^ {\top} \mathbf {D} _ {i} ^ {l}} \\ & {\mathbf {d Q} _ {i} ^ {\phi} = \frac {(\mathbf {d O} _ {i} ^ {l} (\mathbf {H} _ {i}) ^ {\top} - \mathbf {D} _ {i} ^ {l} \mathbf {Z} _ {i} ^ {\top})}{\mathbf {Q} _ {i} ^ {\phi} \mathbf {Z} _ {i}}, \quad \mathbf {d K} _ {j} ^ {\phi} = \mathbf {V} _ {j} (\mathbf {d H} _ {i}) ^ {\top} + (\mathbf {d Z} _ {i}) ^ {\top}, \quad \mathbf {d V} _ {j} = \mathbf {K} _ {j} ^ {\phi} \mathbf {d H} _ {i}} \end{array}\tag{8}
$$

Here, $\mathbf { d K } _ { i } ^ { \phi }$ and $\mathbf { d V } _ { j }$ are obtained by aggregating $\mathbf { d H } _ { i }$ and $\mathbf { d } \mathbf { { Z } } _ { i }$ . Similar to the forward pass, each $\mathbf { d H } _ { i }$ and dZ<sub>i</sub> is precomputed so that the remaining computation reduces to simple matrix additions. The detailed algorithm is provided in Algorithm 2.

## 6 EXPERIMENT

## 6.1 SETUP

Model and Datasets. We use the Wan2.1-1.3B model (Wan et al., 2025) for video generation experiments in the main text and LightningDiT (Yao et al., 2025) for image generation experiments in the Appendix A.2. We also conduct experiments on a private MM-DiT (Esser et al., 2024) model in the Appendix A.4. For video experiments, we use a private dataset collected from websites such as Pexels (Pexels) and Common Crawl (Common Crawl), consisting of 20,000 5-second videos at 480p resolution for fine-tuning. For image experiments, following LightningDiT (Yao et al., 2025), we use the ImageNet (Deng et al., 2009) dataset at a resolution of $5 1 2 \times 5 1 2$

Baselines. We compare SLA with state-of-the-art sparse attention methods applicable to diffusion models, including (1) VSA (Zhang et al., 2025e), (2) VMoBa (Wu et al., 2025), and (3) the training-free SparseAttn (Zhang et al., 2025b) (Sparge-F) and (4) a trainable implementation of SpargeAttn (Sparge-T). For VSA and VMoBa, we use their official implementations, while for ${ \tt S p a r s e { - T } }$ , we implement the method ourselves because there is no official implementation. In addition, we design several baselines for ablation studies: (5) Linear Only, which applies only linear attention; $( 6 ) \ \mathrm { S p a r s e } \ \mathrm { O n 1 y }$ , which applies only the sparse attention component of SLA; and (7) L+S, which directly sums the attention outputs of the Linear Only and Sparse Only.

Metrics. For video quality, following Zhang et al. (2024a); Yang et al. (2025b), we use four evaluation dimensions of VBench (Zhang et al., 2024a): Imaging Quality (IQ), Overall Consistency (OC), Aesthetic Quality (AQ), Subject Consistency (SC). We also use the Vision Reward (VR) (Xu et al., 2024) for human preference evaluation, Aesthetic Video Quality (VA), and Techniqual Video Quality (VT) (Liu et al., 2023). For image quality, following Yao et al. (2025), we use FID. For attention computation complexity, we use FLOPs (floating point of operations). For attention efficiency, we use FLOPS (floating-point operations per second) for attention kernel efficiency. Specifically, FLOPS here is $\mathcal { O } ( \mathrm { f u l l a t t e n t i o n } ) / \dot { t } .$ , where $\bar { \mathcal { O } } ( \cdot )$ denotes the operation count and t the attention latency. We use seconds for end-to-end generation latency.

Hyper-parameters. We use a training batch size of 64 and fine-tune the Wan2.1 model for 2000 steps. For the activation function $\phi ,$ we use softmax according to our ablation experiments. $k _ { h }$ % is 5% and $k _ { l } \%$ is 10%. For block size, we use $b _ { q } = b _ { k v } = 6 4$ . The hyper-parameters for image generation tasks are detailed in Appendix A.2.

## 6.2 EFFECTIVENESS

Table 1 compares the video generation quality and efficiency of SLA with baseline methods on Wan2.1-1.3B, fine-tuned separately with SLA, Full Attention, and each baseline. SLA delivers about a 19.3 efficiency gain while maintaining video quality comparable to Full Attention. Moreover, compared with the baselines, SLA consistently achieves higher quality even under greater sparsity. For example, 95% (1-5%) sparsity in SLA is actually about 3 more efficient than 85% (1-15%) while still producing better video quality.

Table 1: Quality and efficiency comparison of SLA and other baseline methods.

<table><tr><td rowspan="2">Method</td><td colspan="7">Quality</td><td colspan="2">Efficiency</td></tr><tr><td>VA ↑</td><td>VT ↑</td><td>IQ ↑</td><td>OC ↑</td><td>AQ ↑</td><td>SC ↑</td><td>VR ↑</td><td>FLOPs ↓</td><td>Sparsity ↑</td></tr><tr><td>Full Attention</td><td>76.78</td><td>82.88</td><td>62.5</td><td>23.3</td><td>56.1</td><td>93.0</td><td>0.059</td><td>52.75T</td><td>0%</td></tr><tr><td>Sparge-F</td><td>0.002</td><td>0.026</td><td>26.0</td><td>4.6</td><td>35.7</td><td>85.1</td><td>-0.216</td><td>7.91T</td><td>85%</td></tr><tr><td>Sparge-T</td><td>73.83</td><td>77.87</td><td>61.9</td><td>22.7</td><td>55.4</td><td>93.1</td><td>0.014</td><td>7.38T</td><td>84%</td></tr><tr><td>VMoBa</td><td>32.33</td><td>35.79</td><td>58.0</td><td>18.8</td><td>46.2</td><td>89.9</td><td>-0.175</td><td>7.91T</td><td>85%</td></tr><tr><td>VSA</td><td>55.37</td><td>64.61</td><td>60.6</td><td>22.4</td><td>51.9</td><td>83.6</td><td>-0.069</td><td>5.92T</td><td>89%</td></tr><tr><td>SLA</td><td>76.96</td><td>83.92</td><td>62.2</td><td>23.6</td><td>55.9</td><td>93.1</td><td>0.048</td><td>2.74T</td><td>95%</td></tr></table>

Table 2: Ablation results for SLA.

<table><tr><td rowspan="2">Method</td><td colspan="7">Quality</td><td colspan="2">Efficiency</td></tr><tr><td>VA↑</td><td>VT↑</td><td>IQ↑</td><td>OC↑</td><td>AQ↑</td><td>SC↑</td><td>VR↑</td><td>FLOPs ↓</td><td>Sparsity ↑</td></tr><tr><td>Full Attention</td><td>76.78</td><td>82.88</td><td>62.5</td><td>23.3</td><td>56.1</td><td>93.0</td><td>0.059</td><td>52.75T</td><td>0%</td></tr><tr><td>Linear Only</td><td>0.042</td><td>0.099</td><td>39.5</td><td>3.6</td><td>28.8</td><td>90.7</td><td>-0.213</td><td>0.10T</td><td>100%</td></tr><tr><td>Sparse Only</td><td>64.00</td><td>70.50</td><td>57.2</td><td>21.8</td><td>51.7</td><td>88.7</td><td>-0.073</td><td>7.91T</td><td>85%</td></tr><tr><td>L+S</td><td>29.65</td><td>41.15</td><td>58.6</td><td>18.8</td><td>45.3</td><td>87.1</td><td>-0.105</td><td>5.37T</td><td>90%</td></tr><tr><td>SLA (softmax)</td><td>76.96</td><td>83.92</td><td>62.2</td><td>23.6</td><td>55.9</td><td>93.1</td><td>0.048</td><td>2.73T</td><td>95%</td></tr><tr><td>SLA (elu+1)</td><td>75.50</td><td>81.01</td><td>62.8</td><td>23.5</td><td>55.3</td><td>92.9</td><td>0.034</td><td>2.74T</td><td>95%</td></tr><tr><td>SLA (hedgehog)</td><td>74.59</td><td>82.62</td><td>61.9</td><td>22.5</td><td>54.3</td><td>93.2</td><td>0.035</td><td>3.11T</td><td>95%</td></tr><tr><td>SLA (Top 5%)</td><td>76.96</td><td>83.92</td><td>62.2</td><td>23.6</td><td>55.9</td><td>93.1</td><td>0.048</td><td>2.73T</td><td>95%</td></tr><tr><td>SLA (Top 10%)</td><td>75.29</td><td>82.20</td><td>62.5</td><td>22.6</td><td>55.8</td><td>93.5</td><td>0.057</td><td>5.38T</td><td>90%</td></tr><tr><td>SLA (Top 20%)</td><td>75.81</td><td>83.82</td><td>62.7</td><td>22.4</td><td>54.5</td><td>92.6</td><td>0.059</td><td>10.65T</td><td>80%</td></tr></table>

Table 3: Quality of SLA in the zero-shot and limited finetuning settings.

<table><tr><td rowspan="2">Method</td><td colspan="7">Quality</td></tr><tr><td>VA ↑</td><td>VT ↑</td><td>IQ ↑</td><td>OC ↑</td><td>AQ ↑</td><td>SC ↑</td><td>VR ↑</td></tr><tr><td>Full Attention</td><td>76.78</td><td>82.88</td><td>62.5</td><td>23.3</td><td>56.1</td><td>93.0</td><td>0.059</td></tr><tr><td>SLA (0 step)</td><td>41.11</td><td>51.79</td><td>58.3</td><td>21.4</td><td>46.7</td><td>81.0</td><td>-0.1105</td></tr><tr><td>SLA (250 steps)</td><td>64.46</td><td>78.06</td><td>59.0</td><td>22.8</td><td>55.7</td><td>88.5</td><td>-0.0244</td></tr><tr><td>SLA (1000 steps)</td><td>74.58</td><td>80.09</td><td>61.8</td><td>23.7</td><td>56.1</td><td>92.3</td><td>0.0429</td></tr><tr><td>SLA (2000 steps)</td><td>76.96</td><td>83.92</td><td>62.2</td><td>23.6</td><td>55.9</td><td>93.1</td><td>0.0483</td></tr></table>

## 6.3 EFFICIENCY

Figure 6 compares the kernel speed and end-to-end latency of SLA on Wan2.1–1.3B with an RTX5090. Note that even VSA in 89% sparsity and VMoBa in 85% sparsity, their generation quality is already worse than SLA, so higher sparsity settings (e.g., 95%) are not quality-matched comparisons. In the forward pass, SLA achieves a 13.7 speedup over FlashAttention2 and is 1.93 faster than VSA with 95% sparsity and 3.36 faster than VMoBa with 95% sparsity. In the backward pass, it delivers a 6.8 speedup over FlashAttention2, still outperforming VSA and VMoBa. For end-toend video generation, SLA reduces attention latency from 97s to 11s (8.8 reduction), resulting in a 2.2 end-to-end speedup. For fine-tuning overhead, we train Wan2.1-1.3B for only 2,000 steps with a batch size of 64, which is less than 0.1% of the cost of pretraining (typically $\mathrm { 1 0 ^ { 5 } { - } 1 0 ^ { 6 } }$ steps with a batch size of $1 0 ^ { 3 } – 1 0 ^ { 4 } )$ (Wan et al., 2025). The finetuning of SLA requires approximately 9 hours on 8 NVIDIA H200 GPUs.

![](images/29d8500087ec942913d5e7f7bb420425af7fa9de853e573f0a1c02c9732fdf02.jpg)  
Figure 5: Video examples using Wan2.1 fine-tuned with SLA and baselines. For Linear Only, Sparse Only, Sparge-F, VSA, and VMoBa, only a single frame per prompt is shown, as their video quality is not sufficient. The full visible comparison is in Figure 7 in Appendix A.1.

![](images/4a789b6dccb926f72911c54eacded074e9cc6ee24dfe6440b5dae373dfb68f2b.jpg)  
(a) Attention GPU Kernel speed comparison on RTX5090.

![](images/f8945bab60e1938162039d6a2de75b77e96bd03619f75aceeb591599c8ebcc04.jpg)  
(b) End-to-end video generation latency comparison.  
Figure 6: Attention kernel speed and end-to-end generation latency of SLA and baselines on Wan2.1- 1.3B with RTX5090. FlashAttn refers to FlashAttn2, the fastest available version on RTX5090.

In Appendix A.7, we compare the efficiency of SLA and FlashAttention on more GPUs, while in Appendix A.9, we explore LoRA (Hu et al., 2022) as a more efficient finetuning paradigm.

## 6.4 ABLATION STUDY

Fusing sparse and linear attention. To evaluate the effectiveness of SLA in integrating sparse and linear attention, we compare SLA with Sparse Only, Linear Only, and S+L on Wan2.1 in terms of end-to-end generation quality and efficiency. As shown in Table 2, SLA achieves the best generation quality and is more efficient than Sparse Only and S+L, confirming the effectiveness of our fusion strategy.

Activation function in linear attention. To study the effect of the activation function ϕ in the linear attention component of SLA, we evaluate softmax, elu+1, and hedgehog. Results in Table 2 show that softmax generally provides better generation quality and efficiency.

Impact of parameter $k _ { h } .$ . We vary $k _ { h }$ from 5% to 20% and report the results in Table 2. We find that $k _ { h } = 5 \%$ already yields generation quality close to that of full attention. Since $k _ { h } = 5 \%$ saves about half and a quarter of the computation compared with $k _ { h } = 1 0 \%$ and $k _ { h } = 2 0 \%$ , it offers the best trade-off between efficiency and quality.

Impact of parameters $k _ { l } , b _ { q }$ and $b _ { k v }$ . These parameters appear to have a smaller influence compared with $k _ { h }$ . We conduct ablation experiments on the LightningDiT model, and the results are reported in Appendix A.3.

Zero-shot and limited finetuning. Table 3 presents the performance of SLA under zero-shot (0- step finetuning) and limited finetuning budgets. The quality of videos generated by SLA steadily improves as finetuning progresses. After only 1K finetuning steps, the quality is already close to that of full attention, and 2K steps yield the best results.

## 6.5 VISIBLE EXAMPLES

Figure 5 and Figure 7 show video examples from Wan2.1-1.3B fine-tuned using SLA and baselines. SLA produces videos comparable to full attention even at 95% sparsity, while other methods exhibit noticeable distortions even at sparsity levels below 90%.

## 7 RELATED WORK

As sequence lengths in generative models (e.g., language and video) grow, the quadratic cost of attention becomes a key bottleneck. Many studies aim to improve efficiency in two main directions: sparse and linear attention. Most sparse attention methods (Xiao et al., 2024b;a; Jiang et al., 2024; Gao et al., 2024; Fu et al., 2024; Xi et al., 2025; Zhang et al., 2025b; Ribar et al., 2023; Yang et al., 2025a; Xi et al., 2026) speed up inference without training by masking computation at test time. Some (Zhang et al., 2025e; Wu et al., 2025; Zhang et al., 2026b;c) add sparsity during training, enabling higher sparsity. Linear attention methods (Wang et al., 2020; Choromanski et al., 2020; Katharopoulos et al., 2020; Qin et al., 2024; Yang et al., 2024; Sun et al., 2023) are mainly studied in language models. For DiT, SANA (Xie et al., 2024) and Dig (Zhu et al., 2025) show linear attention works for image generation pre-training, but in video generation, existing methods cannot rely on it alone for lossless quality. Another direction is hardware-efficient attention (Dao et al., 2022; Dao, 2023; Shah et ${ \mathrm { a l . , } }$ 2024; Zhang et al., 2024c;b; 2025c;a), which optimizes GPU execution through tiling, kernel fusion, and quantization.

## 8 CONCLUSION

We propose SLA, a trainable attention that unifies sparse and linear attention to accelerate Diffusion Transformers. SLA assigns computation according to importance: it computes $\mathcal { O } ( N ^ { 2 } )$ attention for critical weights, (N) attention for marginal weights, and skips negligible computations. This design enables substantial reductions in attention cost while preserving effectiveness. Experiments show that just a few fine-tuning steps enable SLA to accelerate models effectively. Specifically, SLA achieves about 20 reduction in attention computation, along with a 13.7 GPU kernel speedup and a 2.2 end-to-end speedup on Wan2.1-1.3B, all without degrading the quality of video generation.

## ETHICS STATEMENT

This work proposes a method for improving the efficiency of Diffusion Transformers. The study does not involve human subjects, personally identifiable information, or sensitive data. We believe the proposed method does not raise ethical concerns beyond standard considerations in efficient model design.

## ACKNOWLEDGEMENTS

This work was supported by the Fundamental and Interdisciplinary Disciplines Breakthrough Plan of the Ministry of Education of China (No. JYB2025XDXM101); the NSFC Projects (Nos. 62550004, 92270001, 62376131). J.Z is also supported by the XPlorer Prize.

## REFERENCES

V. L. Arlazarov, E. A. Dinic, M. A. Kronod, and I. A. Faradzev. On economical construction of the transitive closure of an oriented graph. Soviet Mathematics Doklady, 11:1209–1210, 1970.

Krzysztof Marcin Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Quincy Davis, Afroz Mohiuddin, Lukasz Kaiser,