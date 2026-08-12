# Research Idea and Implementation Plan Extraction

## 1. Research Background and Existing Pain Points

Recent advances in Multimodal Large Language Models (MLLMs) have revolutionized multimodal understanding, delivering breakthroughs in image captioning, cross-modal retrieval, and video reasoning. In these models, the quadratic complexity of self-attention mechanisms poses strict limitations on the input token length. However, video understanding inherently requires dense spatiotemporal analysis across thousands of visual tokens to capture precise motion dynamics and scene evolution. Therefore, it is a pivotal challenge to strengthen visual token representations for better video understanding within the bounds of token length and computational limitations.

As highlighted in the research, existing MLLMs, such as the LLaVA family, typically employ straightforward 2D pooling or interpolation to compress visual tokens into an appropriate shape. While these approaches are computationally lightweight, they often overlook the intricate spatiotemporal interactions inherent in visual tokens, leading to suboptimal performance. Recent works like SF-LLaVA and TS-LLaVA have introduced advanced training-free techniques to adapt Image LLMs for video understanding, showing significant improvements on Image LLMs. However, over the past year, the rapid evolution of Video LLMs has resulted in remarkable advancements in video understanding, rendering mere optimizations on Image LLMs insufficient. There is an urgent need for training-free visual token enhancement strategies specifically for Video LLMs.

Furthermore, regarding spatial token processing, Video LLMs often rely on uniform 2D pooling or bilinear interpolation to downsample visual tokens in the spatial dimension. These methods treat all spatial tokens equally, disregarding the inherent heterogeneity in their information content. A typical image frame is dominated by background regions, while semantically salient objects occupy only a small fraction of the spatial grid. By applying equal-weighted downsampling, these critical regions (rich in visual information) are inadequately prioritized, resulting in substantial information loss and token redundancy. This limitation underscores the need for a more refined approach to spatial pooling that dynamically weighs tokens based on their semantic importance.

## 2. Core Research Motivation and Scientific Questions

The core research motivation is to bridge the gap between efficient token compression and the preservation of intricate spatiotemporal interactions in Video LLMs. Real-world video content frequently encompasses a spectrum of temporal granularities, ranging from rapid micro-motions (e.g., hand gestures) to gradual displacements (e.g., pedestrian walking), which demands a multi-scale temporal modeling strategy. Existing Video LLMs implicitly assume that temporal dynamics in videos are uniform across scales, which is a flawed assumption. 

Additionally, there is a strong motivation to address the information loss caused by uniform spatial pooling. By systematically exploring the correlation between token norms and semantic richness, the research seeks to establish token norm as a reliable metric for assessing regional information saliency.

The scientific questions addressed are: How to design a training-free visual token enhancement method specifically for Video LLMs that strategically refines visual tokens across both temporal and spatial dimensions? How to capture multi-grained spatiotemporal interactions without introducing additional trainable parameters? How to preserve high-information visual regions and adaptively compress low-energy backgrounds during spatial pooling to maximize the preservation of semantically meaningful visual details?

## 3. Overall Core Idea and Design Philosophy

The overall core idea is to propose ST-GridPool, a novel training-free visual token enhancement method tailored for Video LLMs, which strategically refines visual tokens across both temporal and spatial dimensions while maintaining computational efficiency. The design philosophy is centered on being training-free, plug-and-play, and requiring no costly retraining or architectural modifications, thereby offering an efficient enhancement for existing Video LLMs like LLaVA-Video.

The method integrates two key components: 
1) Pyramid Temporal Gridding (PTG), which captures multi-grained spatiotemporal interactions through hierarchical temporal gridding; 
2) Norm-based Spatial Pooling (NSP), which preserves high-information visual regions by leveraging the correlation between token norms and semantic richness. 

Through the joint operation of these two components, ST-GridPool effectively enhances visual token representations in a training-free manner, achieving robust video understanding while maintaining computational efficiency.

## 4. Core Innovation Points

1. **First Training-Free Visual Token Enhancement for Video LLMs**: The paper proposes the first training-free visual token enhancement method specifically designed for Video LLMs. By optimizing the visual token compression process, the approach significantly improves video understanding performance while maintaining computational efficiency, eliminating the need for costly fine-tuning or architecture modifications.

2. **Pyramid Temporal Gridding (PTG)**: Introduces a hierarchical gridding strategy that captures multi-grained spatiotemporal interactions across varying temporal lengths. It partitions the video sequence into multiple layers corresponding to distinct temporal segment lengths, enabling multi-grained spatiotemporal feature extraction, capturing both short-term dynamics and long-term context without introducing additional trainable parameters.

3. **Norm-based Spatial Pooling (NSP)**: Leverages the positive correlation between token norms and semantic importance to effectively preserve high-value visual information during token compression. It systematically establishes token norm as a reliable metric for assessing regional information saliency and proposes a norm-based 2D dynamic pooling approach that preserves high-norm regions while adaptively compressing low-energy backgrounds.

4. **Consistent and Significant Performance Improvements**: Extensive experiments on 6 video understanding datasets using widely adopted Video LLMs demonstrate that ST-GridPool consistently enhances performance of Video LLMs. It significantly enhances long-term video understanding, enabling the method to surpass existing 7B models on long-form benchmarks.

5. **Dual Optimization in Computational Efficiency**: The approach achieves a dual optimization by substantially reducing both inference latency and peak GPU memory usage as input frame counts increase, proving its value as a highly efficient enhancement solution robust under strict token budgets.

## 5. Overview of the Overall Technical Solution

Given an input video $V$ with $R$ raw frames, conventional Video LLMs typically select $N \ll R$ frames $I_1, I_2, \cdots, I_N$ via fixed-interval uniform sampling. The selected frames are extracted by the vision tower $\Phi(\cdot)$, to generate frame tokens $T_1, T_2, \cdots, T_N \in \mathbb{R}^{H \times W \times d}$ where $H \times W$ is the spatial dimension, $d$ is the feature dimension. 

The proposed ST-GridPool method takes the token sequence $T_1, \cdots, T_N$ as input and outputs the pooled token $T^\downarrow_1, \cdots, T^\downarrow_N$. It consists of two main components applied sequentially:
1. **Pyramid Temporal Gridding**: Applies a hierarchical gridding strategy over the temporal dimension. It grids and updates frame tokens from different segments of varying lengths to generate summary tokens that capture multi-grained spatiotemporal feature extraction.
2. **Norm-based Spatial Pooling**: Designs a norm-based 2D weighted pooling mechanism that prioritizes high-norm regions during spatial pooling to preserve rich visual semantics, adapting the weights based on the Lp norm of visual tokens within sliding windows.

The resulting downsampled visual tokens are then fed into the language model to generate the final response.

## 6. Detailed Module Design

### Module 1: Pyramid Temporal Gridding (PTG)
Video LLMs often adopt straightforward approaches by uniformly sampling and concatenating token sequences, implicitly assuming that temporal dynamics in videos are uniform across scales. PTG is a hierarchical module that partitions the video sequence into multiple layers, each corresponding to distinct temporal segment lengths. PTG generates summary tokens at the end of each segment, enriching the temporal representation while maintaining computational efficiency.

Given the original visual token sequence $T_1, T_2, \cdots, T_N \in \mathbb{R}^{H \times W \times d}$, PTG consists of $L$ levels, each corresponding to a specific segment length. For the $l$-th level ($l = 1, 2, ..., L$), the segment length is defined as $K_l = K \cdot 2^{l-1}$, where $K$ is the base length of Level-1. The input visual tokens are divided into $N_l$ segments, where $N_l = \lceil N/K_l \rceil$. The start frame index for the $j$-th segment in the $l$-th level is calculated. Each segment spans a specific frame range. For example, for an input token sequence with the length $N = 32$, the base length $K = 8$, and number of levels $L = 3$, the input sequence is divided into 3 layers: Level-1 divides the sequence into 4 segments of 8 frames; Level-2 into 2 segments of 16 frames; Level-3 into 1 segment of 32 frames.

For the $j$-th segment at the $l$-th layer, a summary token is generated to capture the temporal dynamics. First, $m \times n$ frames are uniformly sampled from the segment using specific sampling indices. The corresponding token grids of these frames are spatially concatenated to form an intermediate token grid $G_{l,j}$. Since each $T_{t_{l,j}+k}$ has a resolution of $H \times W$, the resolution of $G_{l,j}$ becomes $mH \times nW$. Subsequently, bilinear interpolation is applied to resize $G_{l,j}$ back to the original resolution $H \times W$, resulting in the final segment-end token grid. The last frame of the segment is then updated with the generated summary token. By processing all segments across layers, the token sequence is updated to incorporate multi-scale temporal dynamics.

### Module 2: Norm-based Spatial Pooling (NSP)
To address the limitation of uniform 2D pooling that treats all spatial tokens equally, NSP dynamically prioritizes regions based on their information saliency. To identify an effective indicator of regional saliency, experiments were conducted on the validation set of the HKU-IS salient object detection dataset. Visualizations of the area of the top 50% visual tokens in terms of L2 norm alongside the salient object ground truth clearly demonstrate that regions corresponding to salient objects consistently exhibit high token norms, while redundant background regions are associated with low token norms. Quantitative analysis of the norm distributions confirms a significant discrepancy in token norms between these two regions, establishing token norm as a reliable metric for assessing regional information saliency.

NSP is a dynamic pooling mechanism that leverages visual token norms to assign adaptive weights to each spatial location during the pooling process. The input to NSP is the visual token sequence $T_1, T_2, \cdots, T_N \in \mathbb{R}^{H \times W \times d}$, derived from the PTG module. The pooling operation employs a kernel size of $(k_H, k_W)$ and a stride of $(s_H, s_W)$. For an input visual token $T_i$, let $t$ denote the sliding window corresponding to the $h$-th row and $w$-th column of the output pooled token. The elements of the sliding window are defined. For each visual feature within the sliding window, its Lp norm is calculated. This norm is then normalized into a weight using the softmax function, ensuring that the weights sum to one across the window. Finally, the pooling result for each sliding window is obtained as a weighted summation of the visual features. 

By computing these weights based on the L2 norm of visual tokens, NSP selectively amplifies the representation of high-importance regions, such as salient objects, while diminishing the influence of low-importance backgrounds. This weighting strategy ensures that semantically rich visual information is prioritized and preserved.

## 7. All Mathematical Formulas and Symbol Definitions

**Formula 1: Start frame index for the $j$-th segment in the $l$-th level**
$$t_{l,j} = (j - 1) \cdot K_l, j = 1, 2, ..., N_l$$
Where:
- $l$: The level index, $l = 1, 2, ..., L$.
- $K_l$: The segment length for the $l$-th level, defined as $K_l = K \cdot 2^{l-1}$.
- $K$: The base length of Level-1.
- $N_l$: The number of segments in the $l$-th level, $N_l = \lceil N/K_l \rceil$.
- $j$: The segment index within the level.
- $t_{l,j}$: The start frame index for the $j$-th segment in the $l$-th level.

**Formula 2: Sampling indices for summary token generation**
$$\{t_{l,j} + k \cdot \lfloor K_l/(m \cdot n) \rfloor\}_{k=0}^{m \cdot n - 1}$$
Where:
- $k$: The sampling index variable ranging from $0$ to $m \cdot n - 1$.
- $m, n$: The number of frames uniformly sampled along the height and width dimensions for spatial concatenation.

**Formula 3: Spatial concatenation to form intermediate token grid $G_{l,j}$**
$$
G_{l,j} =
\begin{bmatrix}
T_{t_{l,j}+0} & T_{t_{l,j}+1} & \cdots & T_{t_{l,j}+m-1} \\
T_{t_{l,j}+m} & T_{t_{l,j}+m+1} & \cdots & T_{t_{l,j}+2m-1} \\
... & ... & \ddots & ... \\
T_{t_{l,j}+(n-1)m} & T_{t_{l,j}+(n-1)m+1} & \cdots & T_{t_{l,j}+mn-1}
\end{bmatrix}
$$
Where:
- $G_{l,j}$: The intermediate token grid for the $j$-th segment at the $l$-th layer.
- $T_{t_{l,j}+k}$: The visual token at the specific frame index, with resolution $H \times W$. The resolution of $G_{l,j}$ is $mH \times nW$.

**Formula 4: Summary token update rule**
$$T_{t_{l,j}+K_l-1} \xleftarrow{update} Interp(G_{l,j})$$
Where:
- $T_{t_{l,j}+K_l-1}$: The last frame token of the segment.
- $Interp(G_{l,j})$: The result of applying bilinear interpolation to resize $G_{l,j}$ back to the original resolution $H \times W$, resulting in the final segment-end token grid $\in \mathbb{R}^{H \times W \times d}$.

**Formula 5: Sliding window elements for Spatial Pooling**
$$t_{m,n} = T_i(h \cdot s_H + m, w \cdot s_W + n)$$
Where:
- $T_i$: The input visual token for the $i$-th frame.
- $t_{m,n}$: The element of the sliding window at local coordinates $m, n$.
- $h, w$: The spatial indices of the output feature map $T^\downarrow_i$.
- $k_H, k_W$: The kernel size of the pooling operation.
- $s_H, s_W$: The stride of the pooling operation.
- $0 \le m < k_H$, $0 \le n < k_W$.

**Formula 6: Weight calculation using softmax normalization**
$$\alpha_{m,n} = \frac{\exp(\beta \|t_{m,n}\|_p)}{\sum_{i=0}^{k_H-1} \sum_{j=0}^{k_W-1} \exp(\beta \|t_{i,j}\|_p)}$$
Where:
- $\alpha_{m,n}$: The normalized weight for the visual feature $t_{m,n}$.
- $\|t_{m,n}\|_p$: The Lp norm of the visual feature $t_{m,n}$.
- $\beta$: A temperature parameter that controls the sharpness of the weight distribution.
- $p$: The norm order (empirically set to $p=2$, L2-norm).

**Formula 7: Weighted summation for pooling result**
$$T^\downarrow_i(h,w) = \sum_{m=0}^{k_H-1} \sum_{n=0}^{k_W-1} \alpha_{m,n} \cdot t_{m,n}$$
Where:
- $T^\downarrow_i(h,w)$: The pooling result for the sliding window corresponding to the $h$-th row and $w$-th column of the output pooled token.

## 8. Algorithm Pseudocode

The original paper does not provide explicit algorithm pseudocode blocks. The algorithmic flow is detailed within the method descriptions and Figure 2. Based strictly on the text, the step-by-step algorithm logic is extracted as follows:

**Algorithm: ST-GridPool**
**Input:** Original visual token sequence $T_1, T_2, \cdots, T_N \in \mathbb{R}^{H \times W \times d}$, Number of levels $L$, Base length $K$, Sampling grid size $m \times n$, Pooling kernel $(k_H, k_W)$, Pooling stride $(s_H, s_W)$, Temperature $\beta$, Norm order $p$.

**Phase 1: Pyramid Temporal Gridding (PTG)**
1. For level $l = 1$ to $L$:
2. $\quad$ Calculate segment length $K_l = K \cdot 2^{l-1}$
3. $\quad$ Calculate number of segments $N_l = \lceil N/K_l \rceil$
4. $\quad$ For segment $j = 1$ to $N_l$:
5. $\quad\quad$ Calculate start frame index $t_{l,j} = (j - 1) \cdot K_l$
6. $\quad\quad$ Determine frame range $\{t_{l,j}, t_{l,j} + 1, ..., \min(t_{l,j} + K_l - 1, N - 1)\}$
7. $\quad\quad$ Uniformly sample $m \cdot n$ frames using indices $\{t_{l,j} + k \cdot \lfloor K_l/(m \cdot n) \rfloor\}_{k=0}^{m \cdot n - 1}$
8. $\quad\quad$ Spatially concatenate token grids of sampled frames to form intermediate token grid $G_{l,j}$ of resolution $mH \times nW$
9. $\quad\quad$ Apply bilinear interpolation to resize $G_{l,j}$ to original resolution $H \times W$, yielding $Interp(G_{l,j}) \in \mathbb{R}^{H \times W \times d}$
10. $\quad\quad$ Update the last frame token: $T_{t_{l,j}+K_l-1} \leftarrow Interp(G_{l,j})$
11. Output updated token sequence incorporating multi-scale temporal dynamics.

**Phase 2: Norm-based Spatial Pooling (NSP)**
12. For each input visual token $T_i$ from the updated sequence:
13. $\quad$ For output spatial index $h, w$:
14. $\quad\quad$ Extract sliding window elements $t_{m,n} = T_i(h \cdot s_H + m, w \cdot s_W + n)$ for $0 \le m < k_H, 0 \le n < k_W$
15. $\quad\quad$ For each element $t_{m,n}$ in the window:
16. $\quad\quad\quad$ Calculate Lp norm $\|t_{m,n}\|_p$
17. $\quad\quad\quad$ Calculate weight $\alpha_{m,n} = \frac{\exp(\beta \|t_{m,n}\|_p)}{\sum_{i=0}^{k_H-1} \sum_{j=0}^{k_W-1} \exp(\beta \|t_{i,j}\|_p)}$
18. $\quad\quad$ Compute output pooled token $T^\downarrow_i(h,w) = \sum_{m=0}^{k_H-1} \sum_{n=0}^{k_W-1} \alpha_{m,n} \cdot t_{m,n}$
19. Return final pooled token sequence $T^\downarrow_1, T^\downarrow_2, \cdots, T^\downarrow_N$