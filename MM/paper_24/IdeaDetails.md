### 1. Research Background and Existing Pain Points

Recent advancements in Multimodal Large Language Models (MLLMs) have underscored that their performance is fundamentally tied to the quality of their visual perception. In a typical MLLM architecture, a vision encoder processes visual signals and projects them into the embedding space of a Large Language Model (LLM) for subsequent reasoning. Consequently, the richness of these visual features is crucial for the model to achieve a comprehensive and fine-grained understanding of the input.

To enhance fine-grained perception, scaling the resolution of visual inputs has emerged as a direct and effective strategy. However, processing entire high-resolution images uniformly is computationally prohibitive. While recent methods leverage a Region-of-Interest (RoI) mechanism to focus on salient areas, they typically present a difficult trade-off: training-based approaches depend on large-scale annotated datasets, a process that is both data-intensive and computationally demanding, often requiring a complete prefilling stage for the initial low-resolution image. Conversely, training-free methods that utilize the model’s internal attention are computationally inefficient and less accurate, as they typically require either complex, multi-pass operations during the prefill stage or reliance on the inherently sequential and slow auto-regressive decoding stage. Furthermore, leveraging the internal attention of MLLMs for RoI identification often yields noisy signals—exhibiting erroneously high attention in background areas (sink tokens) and incomplete activation across the true foreground object. Previous studies reveal that using such raw, noisy attention maps for dense supervision yields sub-optimal results.

### 2. Core Research Motivation and Scientific Questions

The core research motivation stems from the insight that while an MLLM’s internal attention provides a strong RoI signal, it is too noisy for direct supervision. Existing methodologies for identifying these regions present notable limitations regarding data dependency, computational efficiency, and noise. The scientific questions this work aims to answer are:
1. Can we leverage the localization capabilities inherent in the middle layers of MLLMs to build a more efficient RoI predictor in both inference and data?
2. How can we transform the noisy attention maps from the MLLM’s middle layers into high-quality pseudo-RoI labels by explicitly denoising the signal and resolving ambiguity?
3. How can we predict the RoI in a single forward pass using features from the MLLM’s middle layers, decoupling RoI identification from the auto-regressive generation and avoiding costly multi-pass operations?
4. How can we extract high-resolution and fine-grained perceptual cues directly from MLLMs themselves via self-distillation without external models or annotated data?

### 3. Overall Core Idea and Design Philosophy

The overall core idea is to propose an efficient, annotation-free Self-Distilled Region Proposal Network (SD-RPN) that resolves the trade-off between data dependency and computational efficiency. The design philosophy is built around a pipeline that transforms the noisy attention maps from the MLLM’s middle layers into high-quality pseudo-RoI labels by explicitly denoising the signal and resolving ambiguity. These labels are then used to train a Region Proposal Network (RPN) that learns a more precise localization. The RPN is designed to be highly efficient, predicting the RoI in a single forward pass using features from the MLLM’s middle layers, thereby decoupling RoI identification from the auto-regressive generation process and avoiding costly multi-pass operations. The entire framework is trained end-to-end, distilling the localization knowledge from the model’s own response-guided attention into the efficient RPN, enabling a dynamic two-stage inference process where the model first predicts salient regions and then analyzes high-resolution crops to generate its final response.

### 4. Core Innovation Points

1. **Robust Pipeline for Denoising Internal Attention Maps**: Introduction of a robust pipeline to denoise the internal attention maps of an MLLM, generating high-quality pseudo-labels for supervision.
2. **Annotation-Free Self-Distillation Framework**: Proposal of a novel, annotation-free self-distillation framework that trains an RPN to predict RoIs by leveraging the MLLM’s intrinsic localization knowledge.
3. **Highly Efficient Single Forward Pass RoI Prediction**: The RPN is highly efficient, predicting the RoI in a single forward pass using features from the MLLM’s middle layers, decoupling localization from the slow, auto-regressive generation process.
4. **Explicit Denoising and Ambiguity Resolution in Pseudo-Label Generation**: The pseudo-label generation pipeline explicitly denoises the map by removing sink tokens based on feature norms and employs a selective classification strategy that assigns discrete labels based on confidence thresholds and a minimal bounding box around high-attention tokens to resolve the ambiguity inherent in the original noisy map.

### 5. Overview of the Overall Technical Solution

The overall technical solution integrates a Self-Distilled Region Proposal Network (SD-RPN) into the MLLM architecture. The SD-RPN consists of R transformer blocks built upon the first B frozen MLLM layers, which serve as the backbone. The framework operates through a training stage and a two-stage inference process. 

During the training stage (Teacher), the full MLLM generates a response auto-regressively, and the internal response-to-image attention maps are extracted from a pre-selected middle layer. These maps undergo the pseudo-label generation pipeline: first denoising by removing sink tokens, and then assigning foreground, background, or ignored labels based on confidence thresholds and a minimal bounding box. During the training stage (Student), the RPN performs a forward pass on the original input to predict a dense RoI map, which is supervised by the generated pseudo-labels via a masked binary cross-entropy loss.

During the two-stage inference process, the first stage involves predicting the dense RoI map using only the frozen backbone and the trained RPN in a single forward pass. This map is then post-processed (reshaped, smoothed, and binarized) to produce a clean binary foreground mask. In the second stage, the predicted RoI is used to extract fine-grained visual features via either Box Upscaling (processing distinct connected-component regions independently) or Masked Upscaling (computing a single all-encompassing bounding box). The high-resolution visual tokens are inserted into the sequence immediately following the original visual tokens, and the LLM performs auto-regressive decoding on this augmented context to generate the final answer.

### 6. Detailed Module Design

**Preliminaries / Standard MLLM Architecture**: The standard architecture comprises a vision encoder $E_v$, a vision-language projector $P$, and an LLM $L$. Given an image-text pair $(x_v, x_t)$, $E_v$ extracts visual features, which are transformed by $P$ into visual embeddings $H^0_v = P(E_v(x_v))$. Input text $x_t$ yields text embeddings $H^0_{sys}$ and $H^0_{user}$. The MLLM generates responses auto-regressively during the decoding stage. From the cross-modal attention in layer $l$, a RoI map $M^l_{RoI} \in \mathbb{R}^{H \times W}$ is derived, representing the averaged importance of each visual token across textual tokens.

**Pseudo-Label Generation for RoI**: This module transforms noisy RoI maps into sparse and reliable supervisory signals.
1. **Remove Sink Tokens**: Sink tokens attract substantial attention despite being semantically irrelevant, identified by the high L2-norm of their corresponding feature vectors $H_v$. To mitigate them, the initial RoI map $M_{RoI}$ is denoised by suppressing sink tokens identified via a predefined norm threshold $\tau_{norm}$.
2. **Label Assignment**: To address the obscure foreground-background margin and incomplete activation, a selective binary classification is implemented that assigns foreground or background only to high-confidence tokens, leaving ambiguous tokens to be ignored. A minimal bounding box $B_{fg}$ encloses identified foreground tokens. Tokens inside this box that are not classified as foreground are explicitly ignored.

**RoI Prediction via Self-Distillation**: 
1. **Predicting the RoI Map**: The RPN's weights are initialized using layers $B$ to $B+R$ of the pretrained MLLM. For a conversation with $n$ turns, the sequence of hidden states $H$ is collected from the second last layer of the RPN. The hidden state corresponding to the final token of each user question $H_u^{(i)}[-1]$ is isolated as query vectors. These query vectors $H_{RoI}$ and token hidden states $H_v$ are projected into the query and key space using linear layers ($LP_q$ and $LP_k$) from the RPN's final attention block. The predicted dense RoI map $\hat{M}_{RoI}$ is computed via matrix multiplication of these query and key matrices.
2. **Training via Self-Distillation**: The RPN is trained entirely through self-distillation to predict the pseudo-label map $\bar{M}_{RoI}$ by minimizing a binary cross-entropy loss. Using the MLLM’s self-predicted responses yields superior performance due to the principle of representational consistency, creating a more consistent and attainable distillation target.

**Two-Stage Inference with RoI**:
1. **Stage 1**: Predict $\hat{M}_{RoI}$ and post-process it to produce a clean binary foreground mask $B$. The dense map is reshaped into a 2D map $\gamma$, smoothed with a Gaussian filter $G$, and then binarized using a fixed threshold $\tau$.
2. **Stage 2**: Extract fine-grained visual features using upscaling strategies:
   - **Box Upscaling**: Process each salient region independently. Identify distinct connected-component regions $\{R_i\}_{i=1}^k$, compute minimal bounding boxes $b_i$, crop sub-images, and encode them to produce high-resolution embeddings.
   - **Masked Upscaling**: Compute a single, all-encompassing bounding box $b_{all}$ that encloses the union of all connected foreground regions. Crop a single sub-image and use a masking operation to select foreground features.

### 7. All Mathematical Formulas and Symbol Definitions

- **Visual Embeddings**: 
  $H^0_v = P(E_v(x_v))$

- **RoI Map from Attention**:
  $M^l_{RoI} = \sum_{i=1}^{N_t} A^l_i / N_t$, $A = \text{softmax}(Q^l_t (K^l_v)^\top / \sqrt{d})$
  where $Q^l_t \in \mathbb{R}^{N_t \times d}$ and $K^l_v \in \mathbb{R}^{(H \times W) \times d}$ are the query and key matrices from the $N_t$ response tokens and $H \times W$ visual tokens.

- **Denoised Map (Sink Token Removal)**:
  $(M'_{RoI})_j = \begin{cases} 0 & \text{if } \|(H_v)_j\|_2 > \tau_{norm} \\ (M_{RoI})_j & \text{otherwise} \end{cases}$

- **Pseudo-Label Map (Label Assignment)**:
  $(\bar{M}_{RoI})_j = \begin{cases} 1 & \text{if token } j \in S_{fg}, \\ 0 & \text{if token } j \in S_{bg}, \\ -1 & \text{otherwise (ignored)}, \end{cases}$
  where the foreground set is $S_{fg} = \{j \mid a_j \ge \tau_{fg} a_{max}\}$ and the background set is $S_{bg} = \{j \mid j \notin B_{fg} \text{ and } a_j \le \tau_{bg} a_{max}\}$.

- **Hidden States Collection**:
  $H = [H_{sys}, H_v, H_u^{(1)}, H_r^{(1)}, \dots, H_u^{(n)}, H_r^{(n)}]$

- **Query Vectors**:
  $H_{RoI} = \text{concat}(H_u^{(1)}[-1], \dots, H_u^{(n)}[-1])$, where $H_{RoI} \in \mathbb{R}^{n \times d}$

- **Projections for RoI Map**:
  $Q_{RoI} = LP_q(\text{Norm}(H_{RoI}))$, $K_v = LP_k(\text{Norm}(H_v))$

- **Predicted Dense RoI Map**:
  $\hat{M}_{RoI} = Q_{RoI} K_v^\top$

- **Training Loss**:
  $\mathcal{L}_{BCE}(\hat{M}_{RoI}, \bar{M}_{RoI})$

- **Binary Foreground Mask**:
  $B(x, y) = \begin{cases} 1, & \text{if } G(\gamma(\hat{M}_{RoI}))(x, y) > \tau, \\ 0, & \text{otherwise}, \end{cases}$

- **Box Upscaling**:
  $b_i = \text{bbox}(R_i)$, $H^0_{vbox} = \{P(E_v(x_v^i))\}_{i=1}^k$

- **Masked Upscaling**:
  $b_{all} = \text{bbox}(\bigcup_{i=1}^k R_i)$, $H^0_{vmask} = P(B \circ E_v(x_v^{all}))$

- **Theoretical Proof Formulas (Appendix F)**:
  Let $X$ denote token-level features, $Y \in \{0, 1\}$ is the latent FG indicator, $A \in [0, 1]$ is the noisy proxy, $\eta(x) := \Pr(Y = 1 \mid X = x)$.
  Population squared loss: $R(h) = \mathbb{E}[(h(X) - A)^2]$.
  Optimal predictor: $h^\star(x) = \mathbb{E}[A \mid X = x]$.
  Under standard noise models:
  $\mathbb{E}[A \mid X = x] = \begin{cases} (1 - \rho_1 - \rho_0) \eta(x) + \rho_0, & \text{class-conditional noise (CCN)}, \\ \mu_0 + (\mu_1 - \mu_0) \eta(x), & \text{additive activation model}. \end{cases}$
  Noise decomposition: $\epsilon = A - \mathbb{E}[A \mid X]$.
  MSE of optimal predictor: $\mathbb{E}[(h^\star(X) - h^\star(X))^2] = 0$.
  MSE of raw attention: $\mathbb{E}[(A - h^\star(X))^2] = \mathbb{E}[\text{Var}(A \mid X)]$.
  Inequality: $\mathbb{E}[(h^\star(X) - h^\star(X))^2] < \mathbb{E}[(A - h^\star(X))^2]$.
  Classification view (symmetric CCN): $\Pr(A = 1 \mid X = x) = (1 - 2\rho) \eta(x) + \rho$.

### 8. Algorithm Pseudocode

**Algorithm 1: Self-Distilled RPN: training and helper routines**

**Require:** Full MLLM L with L layers; frozen depth B; RPN depth R; pseudo-label layer l; dataset D; thresholds ($\tau_{norm}$, $\tau_{fg}$, $\tau_{bg}$); optimizer for trainable RPN parameters
**Ensure:** Trained RPN on top of the frozen backbone

1: Initialization: Initialize RPN layers B+1:B+R from L; freeze layers 1:B; initialize optimizer.
2: for each $(x_v, x_t) \in D$ do
3: $\quad$ ▷ // Teacher: generate pseudo-labels from the full MLLM
4: $Y_{pred} \leftarrow \text{Decode}(L, x_v, x_t)$ ▷ Auto-regressive response
5: $x_t^{full} \leftarrow \text{Concat}(x_t, Y_{pred})$
6: $H_v^l, Q_r^l, K_v^l \leftarrow \text{Features}(L, x_v, x_t^{full}, l)$
7: $A \leftarrow \text{Softmax}(Q_r^l (K_v^l)^\top / \sqrt{d})$ ▷ Softmax over visual tokens
8: $M_{RoI} \leftarrow \text{MeanAcrossHeadsAndResp}(A)$
9: $M'_{RoI} \leftarrow \text{RemoveSinkTokens}(M_{RoI}, H_v^l, \tau_{norm})$
10: $\bar{M}_{RoI} \leftarrow \text{AssignLabels}(M'_{RoI}, \tau_{fg}, \tau_{bg})$ ▷ Values in {-1, 0, 1}
11: $\text{mask} \leftarrow (\bar{M}_{RoI} \neq -1)$
12: $\quad$ ▷ // Student: forward through frozen 1:B and trainable B+1:B+R
13: $H_v, H_{RoI} \leftarrow \text{Features}(L_{RPN}, x_v, x_t, B+R-1)$
14: $Q_{RoI} \leftarrow LP_q(\text{Norm}(H_{RoI}))$, $K_v \leftarrow LP_k(\text{Norm}(H_v))$
15: $\hat{M}_{RoI} \leftarrow Q_{RoI} K_v^\top$ ▷ Logits for dense RoI map
16: $\text{loss} \leftarrow \text{BCEWithLogits}(\hat{M}_{RoI}[\text{mask}], \bar{M}_{RoI}[\text{mask}])$
17: Update(optimizer, loss)

**Helper routines**

1: function REMOVESINKTOKENS(M, $H_v$, $\tau_{norm}$)
2: $\quad$ return M with entries set to 0 where $\|H_{v,j}\|_2 > \tau_{norm}$
3: function ASSIGNLABELS(M, $\tau_{fg}$, $\tau_{bg}$)
4: $a_{max} \leftarrow \max_j M_j$
5: $S_{fg} \leftarrow \{j : M_j \ge \tau_{fg} a_{max}\}$
6: $B_{fg} \leftarrow \text{MinEnclosingBox}(S_{fg})$
7: $S_{bg} \leftarrow \{j \notin B_{fg} : M_j \le \tau_{bg} a_{max}\}$
8: Build $\bar{M}$ with $\bar{M}_j=1$ for $j \in S_{fg}$, $\bar{M}_j=0$ for $j \in S_{bg}$, and $-1$ otherwise
9: return $\bar{M}$