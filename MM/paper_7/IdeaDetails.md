**1. Research Background and Existing Pain Points**

**Research Background:**
With the recent advancements in large language models (LLMs), significant efforts have been devoted to extending their impressive reasoning and interaction capabilities to vision-language tasks, resulting in Multimodal Large Language Models (MLLMs). Current MLLMs typically integrate visual signals as sequential tokens, which are processed by an LLM to enable visual perception of the world. The typical image- and video-based MLLMs utilize an MLP to project visual information encoded by a Vision Transformer (ViT) into a space interpretable by LLMs, improving performance on visual-language tasks through visual instruction tuning. However, this paradigm requires a large number of visual tokens to represent visual information, particularly with high-resolution images and long-context video inputs. The extensive use of visual tokens, which dominate the input sequence of LLMs, substantially increases the computational complexity and cost associated with inference in MLLMs. The quadratic complexity inherent in Transformer networks scales with the sequence length of input tokens, posing significant challenges and hindering the practical deployment of MLLMs in real-world applications.

**Existing Pain Points:**
Existing token reduction methods typically focus on isolated pipeline components and often neglect textual alignment, leading to performance degradation. Specifically:
1. **Isolated Pipeline Optimization:** Recent studies focus on accelerating inference by reducing visual tokens while preserving essential information, but they tend to focus primarily on specific individual components of the MLLM framework. For instance, FasterVLM and VisionZip perform global dominant visual token selection after vision encoding (operating before sending vision tokens to the LLM), whereas FastV and SparseVLM prune tokens based on attention weights during LLM decoding. These approaches neglect the optimization of the entire MLLM pipeline.
2. **Neglect of Textual Alignment:** Most existing text-agnostic approaches, like Pyramid-Drop, frequently overlook the necessity of aligning visual token selection with textual information. This oversight can result in the loss of textual context, which is essential for accurate LLM decoding, ultimately leading to substantial degradation in performance. VScan adopts a two-stage pruning approach but overlooks the essential role of the text query in aiding visual token selection during the vision encoding stage and directly uses the attention distribution between all visual tokens and the final instruction token for pruning during the LLM decoding stage, causing the potential loss of crucial text-related visual tokens.
3. **Hallucinations from Excessive Text Emphasis:** While CrossGET and Turbo directly leverage text-visual attention to aid token selection, they place excessive emphasis on text tokens, which can lead to hallucinations and disrupt multi-round interactions.

**2. Core Research Motivation and Scientific Questions**

**Core Research Motivation:**
The core motivation is to tackle the high computational costs of MLLMs caused by excessive visual tokens while preserving essential information and textual alignment. There is a strong need to optimize the entire MLLM pipeline (both vision encoding and LLM decoding stages) rather than isolated components, ensuring both visual integrity and cross-modal alignment. By integrating textual context into the visual token reduction process, the method aims to enhance the implicit alignment between visual and textual representations, thereby improving the overall efficiency and performance of the pruned MLLM.

**Scientific Questions:**
1. How to effectively reduce visual tokens across the entire MLLM forward propagation (both vision encoding and LLM decoding stages) in a training-free manner?
2. How to preserve both global semantics and local spatial continuity during visual token selection to maintain visual integrity?
3. How to align visual token reduction with textual instructions to prevent the loss of crucial text-related visual details without causing hallucinations or disrupting multi-round interactions?

**3. Overall Core Idea and Design Philosophy**

The overall core idea is to propose VisionTrim, a unified framework for training-free MLLM acceleration that integrates two effective plug-and-play modules: the Dominant Vision Token Selection (DVTS) module and the Text-Guided Vision Complement (TGVC) module. The design philosophy centers on comprehensively considering the entire forward propagation of the MLLM rather than a specific part. 

Firstly, within the DVTS module, the design considers both global semantics and local spatial continuity to filter visual tokens that convey essential visual information. Beyond utilizing [CLS] token’s attention scores for global semantic importance, the Local Token Affinity Measurement (LTAM) algorithm is developed to simultaneously capture feature similarity and spatial proximity among visual tokens, ensuring critical visual details are retained while reducing redundancy. 

Secondly, in the TGVC module, textual information is leveraged to guide the clustering and merging of pruned visual tokens relevant to the input text instructions. These tokens are then employed to complement the dominant visual tokens from the DVTS module. By integrating textual context into the visual token reduction process, the method enhances the implicit alignment between visual and textual representations.

Both DVTS and TGVC are designed as plug-and-play modules that can be seamlessly integrated between any two layers of either the vision encoder or the LLM, enabling a multi-stage pruning strategy that refines the visual representation while ensuring both computational efficiency and effective cross-modal alignment.

**4. Core Innovation Points (List all innovations in the original text completely)**

1. **Unified Framework for Entire Pipeline:** Introduction of VisionTrim, a unified framework for vision token compression that enables training-free MLLM acceleration, optimizing the entire MLLM pipeline rather than isolated stages.
2. **Dual Plug-and-Play Modules:** Presentation of two effective plug-and-play modules, DVTS and TGVC, designed to accelerate the forward processes of both the vision encoder and the LLM backbone, seamlessly integrable between any two layers.
3. **Local Token Affinity Measurement (LTAM):** Development of the LTAM algorithm within the DVTS module, utilizing a dual-kernel affinity mechanism to simultaneously capture feature similarity and positional proximity for preserving local spatial continuity.
4. **Adaptive Variance-based Weighting Scheme:** Introduction of an adaptive variance-based weighting mechanism in DVTS to integrate global and local importance scores, automatically prioritizing more reliable signals based on their consistency to ensure robust token selection.
5. **Text-Guided Vision Complement (TGVC):** Design of the TGVC module that leverages textual context to identify clustering centers among discarded tokens, assigns remaining tokens via text-guided similarity, and aggregates clusters to complement dominant tokens, thereby preventing the loss of crucial text-related visual elements.
6. **Multi-Stage Pruning Strategy:** Proposal of a multi-stage pruning strategy applying DVTS and TGVC at both the vision encoding stage (using [CLS] token attention) and the LLM decoding stage (using the first generated token's attention and cross-modal attention), dynamically adjusting token retention while preserving cross-modal alignment.

**5. Overview of the Overall Technical Solution**

The approach comprehensively considers the entire pipeline of MLLM, comprising two key components that simultaneously accelerate the vision encoder and LLM forward processes. The first component, the Dominant Vision Token Selection (DVTS) module, meticulously filters tokens to preserve vital visual information, focusing particularly on their significance for global semantics and local spatial continuity. It integrates global semantic importance derived from [CLS] token attention and local spatial continuity captured by the LTAM algorithm using adaptive variance-based weighting to select top-K informative tokens. The second component, the Text-Guided Vision Complement (TGVC) module, leverages textual context to guide the clustering and merging of discarded visual tokens relevant to the input text instructions. It calculates text-visual similarity to identify top-R clustering centers, assigns remaining tokens to clusters based on text-guided similarity, and aggregates them to form R vision complement tokens. These complement tokens are concatenated with the dominant tokens to form the complete visual representation. 

The solution implements a Multi-Stage Pruning Strategy:
1) **Vision Encoding Stage:** Before LLM processing, DVTS and TGVC reduce the initial visual token sequence.
2) **LLM Decoding Stage:** DVTS and TGVC are integrated between any two transformer layers during LLM decoding. Instead of the [CLS] token, the attention distribution of the first generated token is used for global semantic significance. Cross-modal attention scores between visual and textual tokens are computed for TGVC. This multi-stage application refines the visual representation, ensuring computational efficiency and effective cross-modal alignment.

**6. Detailed Module Design (Complete mechanisms of each layer/sub-module)**

**6.1 Dominant Vision Token Selection (DVTS)**
The DVTS module preserves visual integrity during compression by incorporating both global semantic significance and local spatial continuity.
*   **Global Semantic Importance:** Motivated by previous methods, the [CLS] token’s attention distribution across all image tokens serves as a natural measure of global semantic significance. The attention weights are extracted from the penultimate layer of the CLIP-based vision encoder. The self-attention computation for the [CLS] token is expressed to yield the global importance score for each visual token by averaging the attention scores across all attention heads. The computed global scores are then normalized to yield a probability distribution over all visual tokens.
*   **Local Spatial Continuity:** The Local Token Affinity Measurement (LTAM) algorithm effectively captures the local spatial continuity of visual tokens. LTAM utilizes a dual-kernel affinity mechanism to simultaneously account for feature similarity and positional proximity. For a token at a specific position, its local importance is determined by computing the affinity with neighboring tokens within a local kernel. The affinity kernel is defined as a weighted combination of a feature-based term (computed using feature vectors and standard deviations) and a position-based term (computed using spatial coordinates and standard deviations). The local importance of the token is then computed by averaging the affinity scores over all neighboring tokens and converting to a probability distribution.
*   **Adaptive Variance-based Weighting:** To integrate global and local importance scores, an adaptive variance-based weighting mechanism is presented. The final importance score is a weighted sum of the normalized global score and the local score, where the weight alpha is determined by the ratio of the variance of the local scores to the sum of the variances of global and local scores. This adaptive weighting scheme automatically prioritizes more reliable signals based on their consistency. The final importance scores are used to select the top-K informative tokens from the complete set.

**6.2 Text-Guided Vision Complement (TGVC)**
Selected dominant tokens may not fully reflect their relevance to the input instructions. The TGVC module utilizes text instructions to complement the selected dominant vision tokens.
*   **Clustering Centers:** Given the remaining visual tokens after dominant token selection, their similarity with the text features is calculated to identify potential clustering centers. Token-level importance scores are obtained by averaging the similarity scores across all text tokens. The top-R tokens are then selected as clustering centers.
*   **Token Assignment:** For each remaining token not chosen as a center, its assignment score for each clustering center is computed using text-guided similarity. Specifically, similarity scores are calculated from the visual token to text, and from text to the clustering center. The assignment score is the product of these two similarities. Each token is assigned to the clustering center with the highest assignment score.
*   **Cluster Aggregation:** For each cluster centered at a specific center, the assigned tokens are aggregated through weighted averaging based on their text-guided similarities. The center token is added to the weighted sum of assigned tokens, normalized by the sum of assignment scores. This process is repeated for T iterations to refine the clusters. The final vision complement tokens are then concatenated with the dominant tokens to form the complete visual representation.

**6.3 Multi-Stage Pruning Strategy**
*   **Vision Encoding Stage:** Before LLM processing, DVTS and TGVC reduce the initial visual token sequence to a more compact representation.
*   **LLM Decoding Stage:** DVTS and TGVC can be integrated between any two transformer layers during LLM decoding, enabling dynamic token pruning while preserving cross-modal alignment. Instead of using the [CLS] token, the attention distribution of the first generated token is leveraged as a natural measure of the global semantic significance over all image tokens. Global semantic scores for DVTS and cross-modal attention scores between visual and textual tokens for TGVC are computed using the hidden states of the first generated token, visual tokens, and textual tokens at the specific layer. The average cross-modal attention score for each visual token is derived. Using these scores along with the local spatial affinity scores from the LTAM mechanism, the top-K tokens are selected in DVTS and then top-R token complement is performed in TGVC.

**7. All Mathematical Formulas and Symbol Definitions (Replicate exactly as in the original text)**

**Self-attention for [CLS] token:**
$Q_{[CLS]} = W_Q X_{[CLS]}^{L-1}, K_i = W_K X_i^{L-1}$
$A_{[CLS],i}^{L-1} = \text{softmax}(Q_{[CLS]} K_i^T / \sqrt{d_k}), i \in [1, N]$
Where $X_{[CLS]}^{L-1}$ and $X_i^{L-1}$ denote the hidden states of the [CLS] token and the i-th visual token at the (L − 1)-th layer, respectively. $W_Q$ and $W_K$ are learnable projection matrices, and $d_k$ is the dimension of key vector. N represents the total number of visual tokens.

**Global importance score:**
$S_i^g = \frac{1}{H} \sum_{h=1}^H A_{[CLS],i,h}^{L-1}, i \in [1, N]$
Where H is the number of attention heads.
$\hat{S}_i^g = \exp(S_i^g) / \sum_{j=1}^N \exp(S_j^g)$

**LTAM affinity kernels:**
$\kappa_{feat}^{xy,uv} = - \left( \frac{\|F_{xy} - F_{uv}\|}{w_1 \sigma_f} \right)^2, \kappa_{pos}^{xy,uv} = - \left( \frac{\|P_{xy} - P_{uv}\|}{w_2 \sigma_p} \right)^2$
$\kappa_{xy,uv}^* = \kappa_{feat}^{xy,hw} + w_3 \kappa_{pos}^{xy,hw}$
Where $F_{xy} \in \mathbb{R}^d$ and $P_{xy} \in \mathbb{R}^2$ denote the feature vector and spatial coordinates of the token at (x, y), respectively. $\sigma_f$ and $\sigma_p$ represent the standard deviations of the feature and positional differences. The pair (h,w) is sampled from the neighborhood set $\mathcal{N}(x, y)$, and $w_1, w_2$, and $w_3$ are balancing parameters.

**Adaptive weighting:**
$S_i = \alpha \hat{S}_i^g + (1-\alpha) S_i^l, \text{where } \alpha = \sigma_l^2 / (\sigma_g^2 + \sigma_l^2)$
Where $\sigma_g^2$ and $\sigma_l^2$ denote the variances of the global and local importance scores, respectively. The final importance scores $\{S_i\}_{i=1}^N$ are used to select the top-K informative tokens $V_{dom} \in \mathbb{R}^{K \times d}$ from the complete set $V \in \mathbb{R}^{N \times d}$.

**TGVC Clustering Centers:**
$S_{t2v} = \text{softmax}(T V_r^T / \sqrt{d}) \in \mathbb{R}^{L \times (N-K)}$
$s = \frac{1}{L} \sum_{i=1}^L S_{t2v}^i$
Where $V_r \in \mathbb{R}^{(N-K) \times d}$ are the remaining visual tokens, $T \in \mathbb{R}^{L \times d}$ are the text features. The top-R tokens are selected as clustering centers $C = \{c_1, ..., c_R\}$.

**TGVC Token Assignment:**
$S_{v2t}^i = \text{softmax}(v_i^T T / \sqrt{d}), S_{t2c}^j = \text{softmax}(T c_j^T / \sqrt{d})$
$a_{ij} = S_{v2t}^i S_{t2c}^j$
$\text{cluster}(v_i) = \arg\max_j a_{ij}$

**TGVC Cluster Aggregation:**
$v_j^{com} = c_j + \frac{\sum_{v_i \in \text{cluster}(j)} a_{ij} v_i}{\sum_{v_k \in \text{cluster}(j)} a_{kj}}$

**Final Token Concatenation:**
$V_{final} = [V_{dom}; V_{com}] \in \mathbb{R}^{(K+R) \times d}$

**LLM Decoding Stage Scores:**
$S^g = \text{softmax}(H_{gen}^l H_v^l / \sqrt{D}) \in \mathbb{R}^{1 \times N_v}$
$A = \text{softmax}(H_v^l H_t^l / \sqrt{D}) \in \mathbb{R}^{N_v \times N_t}$
$\alpha_i = \frac{1}{N_t} \sum_{j=1}^{N_t} A_{i,j}$
Where $H_{gen}^l \in \mathbb{R}^{1 \times D}$, $H_v^l \in \mathbb{R}^{N_v \times D}$ and $H_t^l \in \mathbb{R}^{N_t \times D}$ represent the first generated token, visual tokens, and textual tokens at layer l, respectively. $\alpha_i$ denotes the average cross-modal attention score for the i-th visual token.

**Computational Budget Estimation:**
$A = \text{Attention}(Q,K) = \text{Softmax}(Q \cdot K^T / \sqrt{d_k})$
$\text{FLOPs}_{0:K-1} = K \times (4nd^2 + 2n^2d + 2ndm)$
Where n denotes the token number, d is the hidden state size, and m is the intermediate size of FFN.

**Theoretical Computation Reduction:**
$F = 1 - \frac{8\gamma N d^2 + 4(\gamma N)^2 d + 6\gamma N dm}{8N d^2 + 4N^2 d + 6N dm}$
Where N denotes the initial number of visual tokens, $K + R$ represents the number of tokens retained, and $\gamma = (K+R)/N$ is the token reduction rate.

**8. Algorithm Pseudocode (If any, paste the pseudocode from the paper exactly as it is)**

The original paper does not provide explicit algorithm pseudocode blocks. The step-by-step mechanisms and iterative processes are fully detailed in the module design section and mathematical formulas above.