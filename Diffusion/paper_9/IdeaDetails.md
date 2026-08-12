# Research Idea and Implementation Plan Extraction

## 1. Research Background and Existing Pain Points
Diffusion transformers exhibit remarkable video generation capability, yet their prohibitive computational and memory costs hinder practical deployment. State-of-the-art models such as Wan2.1-14B still demand extraordinary computational resources: generating a single high-resolution video clip can consume more than 20GB of GPU memory and take nearly one hour of inference time. Such prohibitive memory and latency requirements fundamentally limit the deployment of diffusion-based video generation models in real-world applications, especially under resource-constrained scenarios.

Model quantization and attention sparsification have emerged as two promising directions for compression and acceleration. Quantization reduces memory footprint and computation by representing weights and activations in compact integer formats, while attention sparsification prunes redundant computations by removing negligible attention scores. However, pushing either technique to the extreme inevitably causes severe degradation. For instance, binary quantization collapses representational capacity, while aggressive sparsification discards crucial context information. Although quantization and sparsification are fundamentally orthogonal, naively integrating these techniques results in severe performance degradation. The sparsity-induced information loss exacerbates quantization noise, leading to an amplified attention shift. While sparsification removes low-magnitude attention weights, quantization introduces systematic perturbations to the remaining attention products. These two effects reinforce each other, producing compounded distortions in attention distributions and severely impairing fine-grained dependency modeling in video generation.

## 2. Core Research Motivation and Scientific Questions
The core research motivation is to address the amplified attention shift problem that occurs when naively integrating model quantization and attention sparsification, which severely damages video generation quality. 

The scientific questions are: Why does naive integration fail, and how can we synergistically combine these two orthogonal compression techniques for compounded efficiency gains while maintaining complementary benefits? The failure is attributed to the compounded shift where the parallel error caused by quantization and sparse attention leads to $\Delta_{total} = \Delta_{sparse} + \Delta_{quant} + O(\|\epsilon\|_F \cdot \|M\|_0)$. Furthermore, under quantization, the first-order residual of sparse attention varies with time due to quantization noise, violating the temporal invariance assumption used in prior works. How can we calibrate the attention shift during quantization and recover information lost due to sparsity during inference?

## 3. Overall Core Idea and Design Philosophy
The overall core idea is to propose QuantSparse, a unified compression framework that synergistically integrates model quantization and attention sparsification. The design philosophy is to break the traditional trade-off between efficiency and performance by approaching a Pareto frontier. Specifically, QuantSparse introduces two novel techniques: 1) Multi-Scale Salient Attention Distillation (MSAD), which leverages both global structural guidance and local salient supervision to mitigate quantization-induced bias during the calibration phase; and 2) Second-Order Sparse Attention Reparameterization (SSAR), which exploits the temporal stability of second-order residuals to efficiently recover information lost under sparsity during the inference phase.

## 4. Core Innovation Points
1. We provide formal analysis of the amplified attention shift problem, showing that naive integration of quantization and sparsification severely damages video generation quality.
2. We propose QuantSparse, a unified compression framework that seamlessly combines model quantization and attention sparsification, breaking the traditional trade-off between efficiency and performance.
3. We introduce Multi-Scale Salient Attention Distillation (MSAD) for robust attention alignment, balancing global and local supervision in a memory-efficient manner.
4. We introduce Second-Order Sparse Attention Reparameterization (SSAR) for temporally stable correction for efficient yet accurate approximation of full-attention outputs.

## 5. Overview of the Overall Technical Solution
The QuantSparse framework consists of two components: MSAD for attention distillation during calibration and SSAR for dynamic attention reparameterization during inference. 

During the calibration phase, we apply two parallel attention distillation branches for efficient and robust attention alignment. The global guidance branch downsamples Q and K via average pooling to capture coarse structural topology. The local guidance branch selects top-k salient queries based on token saliency distribution to focus high-resolution supervision on a small set of salient tokens. The distillation object aligns the quantized attention with its FP counterpart.

During the inference phase, we apply an accurate attention approximation using temporally stable second-order residual. We define the first-order residual as the difference between full-attention and sparse attention. Since the first-order residual varies with quantization noise over time, we propose the second-order residual (the difference between adjacent first-order residuals), which exhibits significantly higher temporal stability. We further project the second-order residual onto its most stable subspace via SVD, enabling a lightweight yet accurate correction mechanism that restores fine-grained attention outputs at negligible computational overhead.

## 6. Detailed Module Design
### 6.1 Multi-Scale Salient Attention Distillation (MSAD)
This module addresses the amplified attention shift during PTQ calibration. A straightforward mitigation strategy is to perform attention distillation, but storing full attention matrices incurs O(L^2) memory and compute overhead. MSAD employs two complementary guidance mechanisms:

**Global Guidance:** This approach exploits the intrinsic locality of video data. To efficiently capture global attention patterns, we downsample Q and K via average pooling with stride s, producing low-resolution features. The global distillation is computed using MSE between the FP and quantized global attention.

**Local Guidance:** While global guidance ensures structural fidelity, it fails to capture fine-grained details. We observe that attention saliency is highly skewed: only a small subset of tokens dominates the attention mass. We define the token saliency as the aggregate attention received by token j. We exploit this by selecting the top-k queries from the FP model and computing high-resolution attention only for these salient queries. Local distillation focuses supervision on high-impact regions at minimal cost.

**Integration and Optimization:** We combine both guidance terms into a unified distillation object and optimize the quantization parameters over the calibration dataset to minimize the distillation loss, aligning the quantized attention with its FP counterpart.

### 6.2 Second-Order Sparse Attention Reparameterization (SSAR)
This module addresses the intrinsic sparsity loss during inference. We define the deviation at denoising timestep t as the first-order residual. Prior work assumes residuals are invariant across timesteps, allowing caching of a reference residual. However, quantization noise modifies the first-order residual, violating the temporal invariance assumption. 

**Second-Order Residual:** Although the quantized first-order residual is unstable, the second-order residual (the difference between adjacent first-order residuals) exhibits significantly higher temporal stability. This stability arises because quantization noise follows a slow-varying stochastic process in diffusion process: adjacent timesteps share similar distributions. Leveraging this property, we propose second-order sparse attention reparameterization, which caches the sum of the first-order and second-order residuals.

**SVD Projection:** We further reduce the temporal variance of the second-order residual by projecting it onto its most stable subspace. Empirically, the top-r principal components from the singular value decomposition (SVD) of the second-order residual capture the dominant, temporally stable patterns. We project residuals onto the top-r extracted stable components, suppressing temporal variance and yielding accurate full-attention approximation results. We apply sparse attention for inference with a fixed cache-refreshing interval for full-attention calculation.

## 7. All Mathematical Formulas and Symbol Definitions

**Quantization Mapping:**
$X_Q = \text{clip}(\lfloor \frac{X}{s} \rceil + z, 0, 2^b - 1), Q(X) = s \cdot (X_Q - z)$
where X is a floating-point tensor, $X_Q$ is the discrete representation, s is scale, z is zero-point, and b is bit-width.

**PTQ Calibration Loss:**
$L_{quant} = \min_{s,z} \sum_{X_i \in D_{cal}} \|X_i - Q(X_Q^i; s, z)\|_2^2$

**Sparse Attention:**
$\text{SparseAttention}(Q,K,V;M) = \text{softmax}(\frac{QK^\top}{\sqrt{d_k}} \odot M)V$
where $M \in \{0, 1\}^{L\times L}$ is the mask, L is sequence length, and $\odot$ denotes element-wise multiplication.

**Proposition 3.1 (Amplified Attention Shift):**
$\hat{Q} = Q(X)Q(W_q)^\top, \hat{K} = Q(X)Q(W_k)^\top, \hat{Q}\hat{K}^\top = QK^\top + \epsilon, \text{where } \|\epsilon\|_F \leq \delta$
$\Delta_{total} = \Delta_{sparse} + \Delta_{quant} + O(\|\epsilon\|_F \cdot \|M\|_0)$

**Global Guidance Distillation:**
$A_{global} = \text{softmax}(\frac{\tilde{Q}\tilde{K}^\top}{\sqrt{d_k}}), L_{global} = \text{MSE}(A_{global}^{FP} \| A_{global}^{quant})$
where $\tilde{Q}, \tilde{K} \in \mathbb{R}^{\tilde{L}\times d_k}$ are downsampled features with $\tilde{L} = L/s^2$.

**Token Saliency Definition:**
$A = \text{softmax}(QK^\top/\sqrt{d_k}) \in \mathbb{R}^{h,L,L}, s_j = \sum_{h,i} A_{h,i,j}$
where h denotes the attention head, i denotes the key token index.

**Local Guidance Distillation:**
$A_{local} = \text{softmax}(\frac{Q_{I,:}K^\top}{\sqrt{d_k}}), L_{local} = \text{MSE}(A_{local}^{FP} \| A_{local}^{quant})$
where $I = \{j | s_j \text{ is top-k}\}$ and $Q_{I,:} \in \mathbb{R}^{k\times d_k}$.

**Unified Distillation Object:**
$L_{distill} = L_{quant} + \lambda_{global}L_{global} + \lambda_{local}L_{local}$

**First-Order Residual Invariance Assumption:**
$\Delta(t') \approx \Delta(t) \forall t, t'$

**First-Order Sparse Attention Reparameterization:**
$A_{full}^{(t)} - A_{sparse}^{(t)} \approx A_{full}^{(t_{ref})} - A_{sparse}^{(t_{ref})} = \Delta(t_{ref}) \Rightarrow \hat{A}^{(t)} = A_{sparse}^{(t)} + \underbrace{\Delta(t_{ref})}_{\text{cached}}$

**Proposition 3.2 (Quantized Residual Variance):**
$\Delta_{quant}^{(t)} = A_{full}^{(t)} - A_{s,q}^{(t)} = \Delta^{(t)} + \epsilon^{(t)} + O(\|\epsilon^{(t)}\|_F \cdot \|M\|_0), \Rightarrow \Delta_{quant}^{(t')} \neq \Delta_{quant}^{(t)}, \text{for } t' \neq t.$

**Proposition 3.3 (Second-Order Residual Stability):**
$E_t[\|\hat{\Delta}_{quant}^{(t)} - \hat{\Delta}_{quant}^{(t')}\|_F] \leq E_t[\|\Delta_{quant}^{(t)} - \Delta_{quant}^{(t')}\|_F] \text{for } |t - t'| \leq \tau.$
where $\hat{\Delta}_{quant}^{(t)} := \Delta_{quant}^{(t)} - \Delta_{quant}^{(t-1)}$ is the second-order residual.

**Second-Order Sparse Attention Reparameterization:**
$(A_{full}^{(t)} - A_{s,q}^{(t)}) - (A_{full}^{(t_{ref})} - A_{s,q}^{(t_{ref})}) \approx (A_{full}^{(t_{ref})} - A_{s,q}^{(t_{ref})}) - (A_{full}^{(t'_{ref})} - A_{s,q}^{(t'_{ref})}) = \hat{\Delta}_{quant}^{(t_{ref})},$
$\Rightarrow \tilde{A}^{(t)} = A_{s,q}^{(t)} + \underbrace{\Delta_{quant}^{(t_{ref})} + \hat{\Delta}_{quant}^{(t_{ref})}}_{\text{cached}}.$

**Theorem 3.4 (Error Bound):**
$\underbrace{E_t[\|A_{full}^{(t)} - \tilde{A}_{s,q}^{(t)}\|_F]}_{\text{second-order}} \leq \underbrace{E_t[\|A_{full}^{(t)} - \hat{A}_{s,q}^{(t)}\|_F]}_{\text{first-order}} \text{for } |t - t'| \leq \tau.$

**SVD Projection:**
$\text{SVD}(\hat{\Delta}_{quant}) = S U V^\top, \tilde{\Delta}_{quant} := S_{:,:r}U_{:r,:r}V^\top_{:,:r},$
$\tilde{A}^{(t)} = A_{s,q}^{(t)} + \underbrace{\Delta_{quant}^{(t_{ref})} + \tilde{\Delta}_{quant}^{t_{ref}}}_{\text{cached}}.$

**Proof of Theorem 3.4:**
For $\tilde{A}_{s,q}^{(t)}$, we have:
$(A_{full}^{(t)} - A_{s,q}^{(t)}) - (A_{full}^{(t_{ref})} - A_{s,q}^{(t_{ref})}) = \hat{\Delta}_t^{quant} \Rightarrow A_{full}^{(t)} = A_{s,q}^{(t)} + (A_{full}^{(t_{ref})} - A_{s,q}^{(t_{ref})}) + \hat{\Delta}_t^{quant} = A_{s,q}^{(t)} + \Delta_{quant}^{(t_{ref})} + \hat{\Delta}_t^{quant}.$

Given this, we further have:
$A_{full}^{(t)} - \tilde{A}_{s,q}^{(t)} = (A_{s,q}^{(t)} + \Delta_{quant}^{(t_{ref})} + \hat{\Delta}^{(t)}_{quant}) - \tilde{A}_{s,q}^{(t)} = (A_{s,q}^{(t)} + \Delta_{quant}^{(t_{ref})} + \hat{\Delta}^{(t)}_{quant}) - (A_{s,q}^{(t)} + \Delta_{quant}^{(t_{ref})} + \hat{\Delta}^{(t_{ref})}_{quant}) = \hat{\Delta}^{(t)}_{quant} - \hat{\Delta}^{(t_{ref})}_{quant}.$

Similarly, for $\hat{A}_{s,q}^{(t)}$, we also have:
$A_{full}^{(t)} - \hat{A}_{s,q}^{(t)} = (A_{s,q}^{(t)} + \Delta_{quant}^{(t)}) - \hat{A}_{s,q}^{(t)} = (A_{s,q}^{(t)} + \Delta_{quant}^{(t)}) - (A_{s,q}^{(t)} + \Delta_{quant}^{(t_{ref})}) = \Delta_{quant}^{(t)} - \Delta_{quant}^{(t_{ref})}.$

Based on Proposition 3.3, we have:
$\underbrace{E_t[\|A_{full}^{(t)} - \tilde{A}_{s,q}^{(t)}\|_F]}_{\text{second-order}} = E_t[\|\hat{\Delta}^{(t)}_{quant} - \hat{\Delta}^{(t_{ref})}_{quant}\|_F] \leq E_t[\|\Delta_{quant}^{(t)} - \Delta_{quant}^{(t_{ref})}\|_F] = \underbrace{E_t[\|A_{full}^{(t)} - \hat{A}_{s,q}^{(t)}\|_F]}_{\text{first-order}}.$

## 8. Algorithm Pseudocode

**Algorithm 1 QuantSparse: Calibration to Inference Pipeline**
Require: Pre-trained video diffusion transformer M (FP16), calibration dataset D_cal, target bit-width (W/A), denoising steps T, cache interval $\tau$
Ensure: Quantized-sparse model M_QS, generated video Y

1: Calibration Phase:
2: Initialize quantization parameters {s, z} for weights (W) and activations (A)
3: Input $X \in D_{cal}$ to M
4: Compute token saliency $s_j$ using Eq. 7 for FP model M
5: Select top-k salient tokens $I = \{j | s_j \text{ is top-k}\}$
6: Global Guidance Distillation:
7: Calculate $L_{global}$ using Eq. 6
8: Local Guidance Distillation:
9: Calculate $L_{local}$ using Eq. 8
10: Optimize quantization parameters using Eq. 9 with $L_{global}$ and $L_{local}$
11: Obtain quantized model $M_{quant}$ with optimized {s, z}
12: Inference Phase:
13: Load $M_{quant}$ and input prompt P.
14: Input P into $M_{quant}$ and initialize cached residuals $\{\Delta_{quant}^{(t_{ref})}, \hat{\Delta}_{quant}^{(t_{ref})}\}$
15: for t in T
16: Compute quantized sparse attention:
$A_{s,q}^{(t)} = \text{SparseAttention}(Q_{quant}, K_{quant}, V_{quant}; M)$
17: if $t - t_{ref} \leq \tau$
18: Reuse cached residuals: $\Delta_{curr} = \Delta_{quant}^{(t_{ref})} + \hat{\Delta}_{quant}^{(t_{ref})}$
19: else
20: Update $t_{ref} = t$, recompute and cache residuals
21: endif
22: Refine attention using Eq. 16
23: endfor
24: Generate video Y return Y