## ABSTRACT

Diffusion Large Language Models (dLLMs) have emerged as a promising alternative to autoregressive (AR) LLMs for text generation, with the potential to decode multiple tokens in a single iteration. However, none of the existing opensource dLLMs have achieved superior inference speed over AR LLMs of similar size. This paper breaks this barrier based on a simple and effective strategy named discrete diffusion forcing (D2F). D2F equips dLLMs with two key capabilities: (1) block-wise autoregressive generation to enable KV cache utilization; (2) prediction of following tokens without requiring completion of prior blocks for inter-block parallel decoding. In this way, the vanilla dLLMs are refurbished into an AR-diffusion hybrid paradigm for efficient inference. D2F can be implemented with an asymmetric distillation process based on pre-trained dLLMs to achieve rapid convergence. We further propose a pipelined parallel decoding algorithm, which enables a trade-off between efficiency and efficacy. Empirically, D2F dLLMs achieve more than 2.5× inference speed than LLaMA3 and Qwen2.5 on GSM8K. Compared to vanilla dLLMs such as LLaDA and Dream, D2F delivers over 50× acceleration while preserving comparable output quality. Moreover, it boosts DiffuCoder by up to 11.8× speedup without sacrificing performance. Code is available at https://github.com/SJTU-DENG-Lab/ Discrete-Diffusion-Forcing.

## 1 INTRODUCTION

Large Language Models (LLMs) have maintained a dominant position in text generation for a long time (Achiam et al., 2023; Touvron et al., 2023a; Yang et al., 2025; Grattafiori et al., 2024). Recently,

![](images/4ec11303f2a3f8eddfc486d5715854b18b4d1c408ce1b1322c47ad158a4feeaa.jpg)  
Figure 2: Throughput vs. performance trade-off. As shown, D2F achieves a more favorable trade-off compared to vanilla dLLMs. Refer to Section 4.3 for the details of the hyperparameters τ<sub>add</sub> and τ<sub>conf</sub>.

Diffusion Large Language Models (dLLMs) have emerged as a promising alternative to LLMs (Ye et al., 2025; Nie et al., 2025; Zhu et al., 2025), acknowledged by their potential to generate multiple text tokens in parallel. For example, closed-source dLLMs such as Gemini Diffusion (Google DeepMind, 2025), Mercury (Inception et al., 2025), and Seed Diffusion (Song et al., 2025) can yield thousands of tokens per second, 5-10 times faster than autoregressive (AR) LLMs of similar size.

However, the speed merits of dLLMs have not been demonstrated within the open-source community. Approaches to bridge the gap include designing KV cache strategies (Arriola et al., 2025; Liu et al., 2025; Ma et al., 2025) and improving parallel sampling algorithms (Wu et al., 2025; Wei et al., 2025; Hu et al., 2025). For instance, Block Diffusion (Arriola et al., 2025) turns dLLMs into a block-wise sequential generation paradigm to leverage KV cache. Yet, it precludes the interblock parallelism, a crucial factor for efficient inference. Fast-dLLM (Wu et al., 2025) also adopts a block-wise generation order to facilitate the reuse of states of generated tokens and incorporates confidence-based remasking for parallel decoding. Nonetheless, the cached states can be biased after subsequent tokens are decoded due to the bidirectional nature of the involved attention.

This paper achieves the first breakthrough in accelerating dLLMs to a faster-than-AR regime. Conceptually, we aim to embrace block-wise sequential generation to facilitate KV cache utilization, yet reject the dilemma that the decoding of subsequent blocks must wait for preceding blocks to be fully denoised. This implies a novel AR-diffusion hybrid paradigm, which, however, cannot be realized through naive teacher-forcing training. This is because teacher-forcing requires complete preceding information to predict subsequent content. Interestingly, this insight shares the spirit with the diffusion forcing (DF) (Chen et al., 2024a) technique developed specifically for continuous-space diffusion models. In this sense, this paper forms an extension of DF to discrete data, giving rise to the discrete diffusion forcing (D2F) method for dLLM acceleration.

Concretely, D2F dLLMs learn to denoise a sequence of token blocks with monotonically increasing mask ratios in parallel. Naturally, preceding blocks can finish before subsequent ones, allowing their KV states to be cached for subsequent computations. Note that we constrain the attention to be block-wise causal to ensure the KV cache remains accurate. For training efficiency, we distill D2F dLLMs from existing bidirectional attention dLLMs using an asymmetric distillation loss. In inference, we design a pipelined parallel decoding algorithm which enables inter-block parallelism and offers a decent trade-off between inference efficiency and performance (see Figure 2).

Distilled on the Bespoke-Stratos-17k (Bespoke Labs, 2025) for 12 hours with 8 NVIDIA A100- SXM4-40GB GPUs, D2F can accelerate LLaDA-Instruct-8B (Nie et al., 2025) and Dream-Base-7B (Ye et al., 2025) for over 50× without degradation on mathematical and programming benchmarks, including GSM8K (Cobbe et al., 2021), MATH (Hendrycks et al., 2021), HumanEval (Chen et al., 2021) and MBPP (Austin et al., 2021b). More importantly, D2F-Dream-Base-7B is up to 2.5× faster than LLaMA3-Instruct-8B (Grattafiori et al., 2024) on GSM8K and 1.6× faster on HumanEval, establishing the first open-source dLLMs that outrun AR ones. Moreover, D2F can accelerate DiffuCoder (Gong et al., 2025) by up to 11.8× without sacrificing performance.

![](images/fe77c1e17ba26c02d566723dfdfe4d384c4bcdd0318d848c92264f536ebeb1d2.jpg)  
Figure 3: An overview of discrete diffusion forcing (D2F). During training, the answer is divided into blocks with progressively increasing masking ratios. D2F dLLM is specified with a block-wise causal attention mask, and trained to mimic the prediction of a pre-trained bidirectional dLLM conditioned on partially denoised preceding tokens. This enables inter-block parallel decoding with KV cache compatibility during inference.

In summary, this work makes the following key contributions:

• Discrete Diffusion Forcing (D2F), a novel framework adapting diffusion forcing to the discrete domain. Unlike prior work, D2F is specifically designed to unlock inter-block parallelism for inference acceleration.

• An asymmetric distillation strategy that efficiently refurbishes pre-trained bidirectional dLLMs into block-wise causal models, avoiding the high cost of training from scratch.

• A pipelined parallel decoding algorithm which leverages accurate KV caching to decode future blocks before prior ones are fully generated, ensuring high throughput.

• The first faster-than-AR dLLMs. Our models achieve up to 2.5× speedup over LLaMA3 (Grattafiori et al., 2024) and Qwen2.5 (Yang et al., 2024a) on GSM8K, establishing a new speed benchmark for open-source dLLMs.

## 2 RELATED WORK

Diffusion Large Language Models (dLLMs). The landscape of language generation has long been defined by autoregressive models (Achiam et al., 2023; Guo et al., 2025; Liu et al., 2024). These models, renowned for their high-quality output, are inherently limited by a sequential, tokenby-token decoding process. To overcome this latency bottleneck, dLLMs have emerged (Ye et al., 2025; Nie et al., 2025; Google DeepMind, 2025; Inception et al., 2025; Zhu et al., 2025; Gong et al., 2025). Instead of generating text sequentially, dLLMs operate by iteratively denoising a fully masked sequence, a process that enables the parallel prediction of all tokens at once. This approach, which draws from non-autoregressive principles, utilizes a bidirectional attention mechanism to achieve a more holistic understanding of context. Recent large-scale dLLMs, such as LLaDA (Nie et al., 2025), trained from scratch, and Dream (Ye et al., 2025), initialized from pre-trained AR weights, have demonstrated outstanding performance competitive with leading ARMs, establishing dLLMs as a powerful alternative paradigm for high-quality, parallelizable text generation.

Acceleration of dLLMs. dLLMs suffer from slower inference than autoregressive models due to incompatibility with standard KV cache and limited parallelization. Existing acceleration methods fall into two categories. First, caching-based approaches (Liu et al., 2025; Wu et al., 2025; Ma et al., 2025) develop approximate schemes to reuse computations for static sequence parts, as standard KV cache is incompatible with bidirectional attention. Second, sampling optimization methods reduce decoding steps through confidence-aware strategies (Wu et al., 2025), auxiliary model guid ance (Israel et al., 2025), or adaptive decoding speeds (Wei et al., 2025). However, these methods achieve limited speedups due to inherent approximations and auxiliary model overhead, often failing to match the efficiency of comparable AR (Achiam et al., 2023; Touvron et al., 2023a; Yang et al., 2025; Touvron et al., 2023b; Grattafiori et al., 2024) models. Our approach fundamentally restructures generation into a block-autoregressive framework that is compatible with standard KV cache, while enabling prediction of subsequent tokens without completing previous blocks.

AR-diffusion hybrid models. Recent studies have explored incorporating the speed advantages of autoregressive models into diffusion-based frameworks, particularly in tasks such as video generation (Yin et al., 2025; Po et al., 2025; Sun et al., 2025; Huang et al., 2025). For the video generation task, it is common to model the temporal dependencies between frames using an autoregressive approach, while applying denoising within each frame independently. This hybrid architecture leverages the acceleration benefits of KV cache enabled by the autoregressive modeling paradigm, while preserving the high generation quality of diffusion-based methods.

## 3 PRELIMINARY: DIFFUSION LARGE LANGUAGE MODELS (DLLMS)

Diffusion models, originally developed for continuous data, have achieved state-of-the-art results in image synthesis (Esser et al., 2024; Chen et al., 2024b; Podell et al., 2023; Labs et al., 2025) and video generation (Peng et al., 2025; Zheng et al., 2024; Yang et al., 2024b; Bao et al., 2024). In recent years, advances in the theory of discrete diffusion (Austin et al., 2021a; Shi et al., 2024; Lou et al., 2023; Campbell et al., 2022) have facilitated the emergence of large-scale dLLMs (Nie et al., 2025; Ye et al., 2025) for text generation tasks, which have demonstrated performance competitive with their AR counterparts, while offering the potential for parallel generation.

The majority of successful dLLMs operate under a masked diffusion paradigm (Nie et al., 2025). This process begins with a forward process, where an original text sequence of L tokens, $Y ^ { 0 } =$ $\{ y _ { 1 } ^ { 0 } , \dotsc , y _ { L } ^ { 0 } \}$ }, is progressively corrupted into a noisy sequence $Y ^ { t }$ over a continuous time schedule $t \in [ 0 , 1 ]$ . This corruption is routinely achieved by replacing original tokens independently with a special [MASK] token. Typically, we can define the conditional distribution as:

$$
q (Y ^ {t} | Y ^ {0}) = \prod_ {i = 1} ^ {L} q (y _ {i} ^ {t} | y _ {i} ^ {0}), \quad \text { where } \quad q (y _ {i} ^ {t} | y _ {i} ^ {0}) = \left\{ \begin{array}{l l} 1 - t, & \text { if } y _ {i} ^ {t} = y _ {i} ^ {0} \\ t, & \text { if } y _ {i} ^ {t} = [ \text { MASK } ] \end{array} \right..\tag{1}
$$

Consequently, $Y ^ { 1 }$ represents a fully masked sequence. dLLMs are designed to learn a parameterized model $\mathsf { \bar { p } } _ { \theta } ( Y ^ { \bar { 0 } } | Y ^ { t } )$ to reverse the forward process, hence enabling denoising from the fully masked sequence ${ \dot { Y } } ^ { 1 }$ to language samples $Y ^ { 0 }$ . This formulation allows the model $p _ { \theta }$ to predict all mask tokens in the sequence simultaneously at each inference step, forming the basis of dLLMs’ theoretical potential to surpass the AR paradigm for the high-speed, parallel generation.

However, practical implementations of dLLMs face severe inference efficiency bottlenecks. First, the use of bidirectional attention mechanisms fundamentally conflicts with KV cache, leading to significant redundant computation across denoising steps. Second, the reliance on a conditional independence assumption for parallel decoding makes it hard to generate interdependent tokens (Song & Zhou, 2025), so more iterative steps are required for high-quality outputs. While prior works attempt to mitigate these issues, for instance, by introducing block-wise sequential generation to enable KV cache (Arriola et al., 2025) or implementing approximate KV cache (Wu et al., 2025; Liu et al., 2025), they fail to simultaneously achieve both precise KV cache and efficient parallel decoding. Consequently, no open-source dLLMs has yet matched the inference speed of AR models.

## 4 METHOD

The paper derives discrete diffusion forcing (D2F) to accelerate the inference of dLLMs to a fasterthan-AR regime. Refer to Figures 3 and 4 for an overview of the training and inference of D2F.

## 4.1 DISCRETE DIFFUSION FORCING

D2F endows dLLMs with two critical capabilities: (1) block-level AR generation, which enables efficient standard KV cache and significantly reduces computational redundancy; and (2) inter-block parallel decoding, where the model is trained to predict future blocks from incomplete, partially reconstructed predecessors to maximize the number of decoded tokens per inference step. This naturally gives rise to an AR-diffusion hybrid modeling paradigm, but naive teacher-forcing training, as employed in block diffusion (Arriola et al., 2025), cannot lead to the second capability. To address this, D2F trains the model to perform conditional denoising of the current block based on a partially denoised prefix, enabling more coherent and efficient sequential generation.

```txt
dLLMs a new that Semi-activated Block
dLLMs are a new type of that Fully-activated Block
dLLMs are a new type of that mix Cached Block
dLLMs are a new type of AI that mix ideas from tech They
dLLMs are a new type of AI that mix key ideas from modern tech They LLM
dLLMs are a new type of AI that mix key ideas from modern tech . They blend LLM
dLLMs are a new type of AI that mix key ideas from modern tech . They blend diffusion LLM to ...
```  
Figure 4: Visualization of the pipelined parallel decoding of D2F dLLMs. As shown, a pipeline of blocks are decoded in parallel. A new block is dynamically added when the completion ratio of the last block exceeds a threshold $\begin{array} { r } { \tau _ { a d d } ~ ( = ~ \frac { 1 } { 3 } } \end{array}$ here). The new block is initially semi-activated and will transition to a fully-activated state when its predecessor reaches the completion threshold $\begin{array} { r } { \tau _ { a c t } ~ ( = ~ \frac { 5 } { 6 } } \end{array}$ here). The fully-activated blocks differ from semi-activated ones in that they would decode multiple tokens in each step more aggressively.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 Asymmetric Distillation for D2F
Require: Pre-trained dLLM  $p_{\phi^{-}}$ ; D2F dLLM  $p_{\theta}$ ; block size k; training dataset D.
1: while training do
2: Sample a sequence Y from D.
3: Divide Y into N blocks  $\{Y_{B_{1}},\ldots,Y_{B_{N}}\}$ , each of size k.
4: Sample a monotonic noise schedule  $\{t_{1},\ldots,t_{N}\}$  where  $t_{1}&lt;\cdots&lt;t_{N}$ .
5: For each  $i\in\{1,\cdots,N\}$ , corrupt  $Y_{B_{i}}$  to  $Y_{B_{i}}^{t_{i}}$  using Eq. 1.
6: Predict distributions for each block  $Y_{B_{i}}^{0}$  with:
7: Teacher (global view):  $p_{\phi^{-}}(Y_{B_{i}}^{0}|Y_{B_{1}}^{t_{1}},\ldots,Y_{B_{N}}^{t_{N}})$ 
8: Student (causal view):  $p_{\theta}(Y_{B_{i}}^{0}|Y_{B_{1}}^{t_{1}},\ldots,Y_{B_{i}}^{t_{i}})$ 
9: Update  $p_{\theta}$  based on the loss  $L_{D2F}$  defined in Eq. 3.
</div>

Concretely, D2F partitions a clean sequence $Y ^ { 0 }$ into N blocks of size k. Let $B _ { i } : = \{ ( i { - } 1 ) { * } k , \ldots , i { * }$ $k - 1 \}$ denote the token indices in the i-th block and $Y _ { B _ { i } }$ denote the corresponding subsequence. In the forward process, D2F applies a monotonically increasing noise schedule $( t _ { 1 } < t _ { 2 } < \cdots < t _ { N } )$ to the N blocks, i.e., $Y ^ { t } = \hat { \{ } Y _ { B _ { 1 } } ^ { t _ { 1 } } , . . . , Y _ { B _ { N } } ^ { t _ { N } } \}$ . Namely, the earlier blocks in $Y ^ { t }$ are progressively less masked (i.e., more complete), while later blocks remain increasingly masked (i.e., more uncertain). For the reverse process, D2F trains a θ-parameterized model to characterize:

$$
p _ {\theta} (Y ^ {0} | Y ^ {t}) = \prod_ {i = 1} ^ {N} p _ {\theta} (Y _ {B _ {i}} ^ {0} | Y _ {B _ {1}} ^ {t _ {1}}, \ldots , Y _ {B _ {i}} ^ {t _ {i}}).\tag{2}
$$

Intuitively, the learned model can first complete the decoding of preceding blocks while simultaneously advancing the denoising of subsequent ones, which effectively enables inter-block parallel decoding. By preserving a causal attention structure across blocks—wherein intra-block attention remains bidirectional—we can cache the KV states of already decoded blocks for exact reuse, thereby reducing redundant computations and improving inference efficiency.

Connection to diffusion forcing (Chen et al., 2024a). From a high-level perspective, our approach bears a strong conceptual resemblance to diffusion forcing (DF) (Chen et al., 2024a), originally developed for continuous-space diffusion models and notably applied in video generation (Yin et al., 2025). Both methods involve predicting the tokens of the next block conditioned on a noisy, incomplete premise, and our framework can be viewed as an extension of DF to discrete sequences. Such a principled extension motivates our naming of discrete diffusion forcing.

## 4.2 ASYMMETRIC DISTILLATION

Noting that training a dLLM with billions of parameters from scratch can be costly (Nie et al., 2025; Ye et al., 2025), we propose to distill a D2F dLLM from a pre-trained vanilla dLLM available in the open-source community. Let $p _ { \phi }$ − denote the standard bidirectional teacher dLLM and $p _ { \theta }$ the student D2F dLLM (θ is initialized as $\phi ^ { - } )$ . The distillation minimizes the following loss

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 2 Pipelined Parallel Decoding for D2F Inference
Require: D2F model $p_{\theta}$; thresholds $\tau_{\text{add}}, \tau_{\text{act}}, \tau_{\text{conf}}$.
1: Initialize $Y = \{Y_{B_1}\}$ with a block of mask tokens.
2: while generation is not complete do
3: if the ratio of decoded tokens in $Y_{B_{-1}}$ exceeds $\tau_{\text{add}}$ and $&lt;|\text{EOS}|&gt;$ not in $Y$ then
4: Append a new fully masked block with semi-activated state to $Y$.
5: for the active block $Y_{B_i}$ in $Y$ do
6: if the ratio of decoded tokens in $Y_{B_{i-1}}$ exceeds $\tau_{\text{act}}$ then
7: Set $Y_{B_i}$ to be fully-activated
8: Forward pass of $Y$ with D2F dLLM $p_{\theta}$ using cached KV
9: for the active block $Y_{B_i}$ in $Y$ do
10: Let $J_i$ record the set of token positions in $B_i$ with $&gt;\tau_{\text{conf}}$ prediction confidence
11: if $Y_{B_i}$ is fully-activated and $J_i$ is $\emptyset$ then
12: Add the token position with the highest confidence to $J_i$
13: Sample tokens with positions in $J_i$ and remask other positions
14: Update KV cache for completed blocks
</div>

$$
\mathcal {L} _ {\mathrm{D2F}} = \mathbb {E} _ {t _ {1} <   \dots <   t _ {N}} \left[ \sum_ {i = 1} ^ {N} D _ {\mathrm{KL}} \left(p _ {\theta} (Y _ {B _ {i}} ^ {0} | Y _ {B _ {1}} ^ {t _ {1}}, \ldots , Y _ {B _ {i}} ^ {t _ {i}}) \| p _ {\phi^ {-}} (Y _ {B _ {i}} ^ {0} | Y _ {B _ {1}} ^ {t _ {1}}, \ldots , Y _ {B _ {N}} ^ {t _ {N}})\right) \right],\tag{3}
$$

where $D _ { \mathrm { K I } }$ represents the KL divergence aggregated over the mask tokens. As shown, the distillation is asymmetric—the teacher $p _ { \phi ^ { - } }$ predicts for each block $Y _ { B _ { i } } ^ { 0 }$ with a global view of all noisy blocks while the student $p _ { \theta }$ learns to approximate using only a causally restricted view. In this way, the mask prediction capabilities of existing dLLMs can be embedded into a new D2F dLLM. This objective also connects to CausVid (Yin et al., 2025), which distills existing holistic diffusion video generators into streaming ones. As for model architecture, the student differs from the teacher solely in attention masks—it uses the block-wise causal attention instead of a bidirectional one.

We summarize the whole algorithmic process in Algorithm 1.

## 4.3 PIPELINED PARALLEL DECODING

As illustrated in Figure 4, we introduce a pipelined parallel decoding algorithm for the inference of D2F dLLMs. Specifically, we maintain a sliding window of active blocks and dynamically append a new fully-masked block when the decoding progress of the last block exceeds a threshold $\tau _ { \mathrm { a d d } } .$ . The dynamic strategy significantly reduces the per-step computational cost compared to maintaining a full sequence of massive blocks throughout inference.

Observing that aggressive decoding of a newly added block can degrade performance, we incorporate a dual-state decoding mechanism. Concretely, the newly added block is initialized in a semiactivated state to enable conservative parallel decoding, and will become fully activated when its predecessor has finished $\tau _ { \mathrm { a c t } }$ of the decoding—i.e., sufficient contextual information has been accumulated to support aggressive decoding of the latter block. Following Fast-dLLM (Wu et al., 2025), semi-activated blocks admit tokens with confidence above a threshold $\tau _ { \mathrm { c o n f } }$ , while fully activated blocks additionally enforce the selection of the most confident token when no such token exists.

The synergy between the dynamic block management and dual-state mechanism conjoins per-step efficiency and inter-block parallelism. It is interesting to note that our approach also shares conceptual similarities with prior work in video generation (Teng et al., 2025). More algorithmic details are provided in Algorithm 2 and a hyperparameter analysis is in Table 3.

## 5 EXPERIMENTS

This section details the experimental setup and presents the results of D2F dLLMs.

<table><tr><td>Test Set</td><td>Method</td><td>TPS ↑</td><td>Latency (s) ↓</td><td>Gen. Length</td><td>Score ↑</td></tr><tr><td rowspan="4">GSM8K4-shot</td><td>LLaDA-Instruct</td><td>7.2 (1.0x)</td><td>32.3 (1.0x)</td><td>231</td><td>77.4</td></tr><tr><td>dLLM-Cache</td><td>20.1 (2.8x)</td><td>11.5 (2.8x)</td><td>231</td><td>77.5</td></tr><tr><td>Fast-dLLM (Prefix-Cache)</td><td>33.3 (4.6x)</td><td>7.0 (4.6x)</td><td>232</td><td>77.8</td></tr><tr><td>Fast-dLLM (Dual-Cache)</td><td>35.2 (4.9x)</td><td>6.6 (4.9x)</td><td>232</td><td>78.9</td></tr><tr><td></td><td>D2F-LLaDA</td><td>52.5 (7.3x)</td><td>2.8 (11.5x)</td><td>144</td><td>77.3</td></tr><tr><td rowspan="4">MBPP3-shot</td><td>LLaDA-Instruct</td><td>0.9 (1.0x)</td><td>71.4 (1.0x)</td><td>65</td><td>39.0</td></tr><tr><td>dLLM-Cache</td><td>2.3 (2.6x)</td><td>28.3 (2.5x)</td><td>66</td><td>37.0</td></tr><tr><td>Fast-dLLM (Prefix-Cache)</td><td>13.0 (14.4x)</td><td>4.9 (14.6x)</td><td>64</td><td>37.6</td></tr><tr><td>Fast-dLLM (Dual-Cache)</td><td>15.3 (17.0x)</td><td>3.8 (18.8x)</td><td>58</td><td>36.4</td></tr><tr><td></td><td>D2F-LLaDA</td><td>47.6 (52.9x)</td><td>1.4 (51.0x)</td><td>68</td><td>38.0</td></tr><tr><td rowspan="4">HumanEval0-shot</td><td>LLaDA-Instruct</td><td>2.8 (1.0x)</td><td>38.8 (1.0x)</td><td>107</td><td>36.0</td></tr><tr><td>dLLM-Cache</td><td>4.5 (1.6x)</td><td>23.3 (1.7x)</td><td>104</td><td>39.0</td></tr><tr><td>Fast-dLLM (Prefix-Cache)</td><td>13.7 (4.9x)</td><td>7.4 (5.2x)</td><td>102</td><td>38.4</td></tr><tr><td>Fast-dLLM (Dual-Cache)</td><td>19.2 (6.9x)</td><td>5.2 (7.5x)</td><td>100</td><td>35.4</td></tr><tr><td></td><td>D2F-LLaDA</td><td>81.6 (29.1x)</td><td>1.6 (24.3x)</td><td>133</td><td>40.2</td></tr><tr><td rowspan="4">Math4-shot</td><td>LLaDA-Instruct</td><td>21.1 (1.0x)</td><td>11.5 (1.0x)</td><td>243</td><td>23.7</td></tr><tr><td>dLLM-Cache</td><td>26.9 (1.3x)</td><td>9.1 (1.3x)</td><td>246</td><td>23.2</td></tr><tr><td>Fast-dLLM (Prefix-Cache)</td><td>47.7 (2.3x)</td><td>5.2 (2.2x)</td><td>246</td><td>22.4</td></tr><tr><td>Fast-dLLM (Dual-Cache)</td><td>42.5 (2.0x)</td><td>5.8 (2.0x)</td><td>246</td><td>22.4</td></tr><tr><td></td><td>D2F-LLaDA</td><td>90.2 (4.3x)</td><td>4.3 (2.7x)</td><td>384</td><td>29.1</td></tr></table>

Table 1: Performance comparison of various acceleration methods on LLaDA-Instruct-8B. Speedup ratios are shown in (green). All baseline methods use the default sampling configuration from the original LLaDA implementation. See Appendix D for detailed hyperparameters.

## 5.1 EXPERIMENT SETTINGS

We perform evaluation on two representative dLLMs: LLaDA-Instruct-8B (Nie et al., 2025) and Dream-Base-7B (Ye et al., 2025)—backbones shared by prior acceleration work—for fair comparison. For our distillation-based training, we utilize a dataset derived from the Bespoke-Stratos-17k benchmark (Bespoke Labs, 2025). Specifically, we use two publicly available collections sourced from the HuggingFace Hub , where a third party had previously generated responses to the problems from Bespoke-Stratos-17k using the Qwen2.5-7B model (Yang et al., 2024a). These collections are pre-filtered to a maximum length of 600 tokens. The q proj, v proj, k proj, and o proj modules of D2F dLLMs are tuned with the LoRA (Hu et al., 2022) technique, using the rank of 32, the scaling factor of 32, and the dropout rate of 0.1. During training, We truncate or pad all sequences to a final length of 1024 tokens, and employ a block size of 16. We applied block-wise monotonically increasing mask ratios, with a maximum threshold of 0.7 and a minimum threshold of 0.2, inspired by Block Diffusion (Arriola et al., 2025). During inference, unless otherwise specified, $\tau _ { \mathrm { c o n f } }$ is set to 0.9, $\tau _ { \mathrm { a d d } }$ is set to 0.1, and $\tau _ { \mathrm { a c t } }$ is set to 0.95. The models are trained for 12 hours using an AdamW optimizer with a constant learning rate of $1 0 ^ { - 5 }$ . All training and inference are conducted on a setup comprising 8 NVIDIA A100-SXM4-40GB GPUs. Detailed hyperparameter configurations for inference on each benchmark are provided in the Appendix D. In addition, we conduct experiments on DiffuCoder (Gong et al., 2025), and report the results in Appendix E.

## 5.2 MAIN RESULTS

Benchmarks. Following established conventions, performance evaluation of D2F is conducted across mathematical reasoning and code generation benchmarks, including GSM8K (Cobbe et al., 2021), GSM8K-CoT (Chain-of-Thought reasoning variant of GSM8K), Math (Hendrycks et al., 2021), HumanEval (Chen et al., 2021), and MBPP (Austin et al., 2021b).

Baselines. Comprehensive comparisons are established between D2F and state-of-the-art acceleration strategies, including Fast-dLLM (Wu et al., 2025) and dLLM-Cache (Liu et al., 2025), implemented on LLaDA-Instruct-8B (Nie et al., 2025) and Dream-Base-7B (Ye et al., 2025) architectures.

<table><tr><td>Test Set</td><td>Method</td><td>TPS ↑</td><td>Latency (s) ↓</td><td>Gen. Length</td><td>Score ↑</td></tr><tr><td rowspan="4">GSM8K-CoT8-shot</td><td>Dream-Base</td><td>9.5 (1.0x)</td><td>26.8 (1.0x)</td><td>255</td><td>75.0</td></tr><tr><td>dLLM-Cache</td><td>26.0 (2.7x)</td><td>9.8 (2.7x)</td><td>255</td><td>72.0</td></tr><tr><td>Fast-dLLM (Prefix-Cache)</td><td>50.3 (5.3x)</td><td>5.1 (5.3x)</td><td>255</td><td>76.6</td></tr><tr><td>Fast-dLLM (Dual-Cache)</td><td>49.8 (5.2x)</td><td>5.1 (5.3x)</td><td>255</td><td>75.0</td></tr><tr><td></td><td>D2F-Dream</td><td>91.2 (9.6x)</td><td>2.8 (9.6x)</td><td>256</td><td>77.6</td></tr><tr><td rowspan="4">MBPP3-shot</td><td>Dream-Base</td><td>10.4 (1.0x)</td><td>24.6 (1.0x)</td><td>256</td><td>56.2</td></tr><tr><td>dLLM-Cache</td><td>25.5 (2.5x)</td><td>10.0 (2.5x)</td><td>256</td><td>52.6</td></tr><tr><td>Fast-dLLM (Prefix-Cache)</td><td>71.6 (6.9x)</td><td>3.6 (6.8x)</td><td>256</td><td>56.4</td></tr><tr><td>Fast-dLLM (Dual-Cache)</td><td>73.2 (7.0x)</td><td>3.5 (7.0x)</td><td>256</td><td>51.0</td></tr><tr><td></td><td>D2F-Dream</td><td>105 (10.1x)</td><td>2.3 (10.7x)</td><td>240</td><td>55.2</td></tr><tr><td rowspan="4">HumanEval0-shot</td><td>Dream-Base</td><td>20.2 (1.0x)</td><td>12.6 (1.0x)</td><td>255</td><td>54.3</td></tr><tr><td>dLLM-Cache</td><td>23.2 (1.1x)</td><td>11.0 (1.1x)</td><td>255</td><td>55.5</td></tr><tr><td>Fast-dLLM (Prefix-Cache)</td><td>62.4 (3.1x)</td><td>4.1 (3.1x)</td><td>255</td><td>54.3</td></tr><tr><td>Fast-dLLM (Dual-Cache)</td><td>60.0 (3.0x)</td><td>4.3 (2.9x)</td><td>255</td><td>53.0</td></tr><tr><td></td><td>D2F-Dream</td><td>73.2 (3.6x)</td><td>3.1 (4.1x)</td><td>227</td><td>54.3</td></tr><tr><td rowspan="4">Math4-shot</td><td>Dream-Base</td><td>9.9 (1.0x)</td><td>25.8 (1.0x)</td><td>256</td><td>35.8</td></tr><tr><td>dLLM-Cache</td><td>12.7 (1.3x)</td><td>20.2 (1.3x)</td><td>256</td><td>34.5</td></tr><tr><td>Fast-dLLM (Prefix-Cache)</td><td>65.6 (6.6x)</td><td>3.9 (6.6x)</td><td>256</td><td>37.6</td></tr><tr><td>Fast-dLLM (Dual-Cache)</td><td>67.0 (6.8x)</td><td>3.8 (6.8x)</td><td>256</td><td>37.1</td></tr><tr><td></td><td>D2F-Dream</td><td>98.8 (10.0x)</td><td>2.6 (9.9x)</td><td>256</td><td>35.4</td></tr></table>

Table 2: Performance comparison of various acceleration methods on Dream-Base-7B. Speedup ratios relative to the baseline are shown in (green). The max generation length is set to 256. See Appendix D for detailed hyperparameters.

Additional benchmarking against leading auto-regressive (AR) LLMs of comparable scale, specifically LLaMA3-Instruct-8B (Grattafiori et al., 2024) and Qwen2.5-Base-7B (Yang et al., 2024a), demonstrates the efficacy of D2F in achieving faster-than-AR inference speeds.

Quantitative Results. As shown in Figure 1, D2F represents the first open-source dLLMs to surpass state-of-the-art AR LLMs in inference speed. The maximum generation length is set to 512 for all methods to ensure fairness. Concretely, D2F-Dream-Base-7B achieves a throughput of 119.9 tokens/s on GSM8K. This constitutes a 2.5× speedup over LLaMA3-Instruct-8B (48.0 tokens/s) and a 2.3× speedup over Qwen2.5-Base-7B (52.7 tokens/s).

D2F also significantly accelerates baseline dLLMs while maintaining equivalent average performance. As shown in Table 1, D2F-LLaDA-Instruct-8B achieves a 52.9× speedup (47.6 tokens/s vs. baseline 0.9 tokens/s) with minimal performance difference (38.0 vs. baseline 39.0), and consistently outperforms prior dLLM acceleration techniques. This owes to D2F’s early termination via <EOS> detection on Instruct models, avoiding unnecessary generation. On MATH, it reaches 90.2 tokens/s, a 2.1× speedup over Fast-dLLM’s Dual-Cache (42.5 tokens/s). For fairness, the baseline hyperparameters follow the settings in (Nie et al., 2025). Similarly, as shown in Table 2, D2F-Dream-Base-7B attains 91.2 tokens/s on GSM8K-CoT, corresponding to a 9.6× speedup over Dream-Base-7B (9.5 tokens/s) with slight performance improvement, and a 1.8× speedup over FastdLLM (49.8 tokens/s). Since Dream-Base struggles to generate the stop token correctly, we set a unified maximum length of 256 for all methods in this comparison, following the setting in dLLMcache (Liu et al., 2025). We vary max lengths (512 vs. 256) to strictly align with respective baseline protocols for fair comparison. These results demonstrate that D2F not only surpasses existing acceleration methods but also enables dLLMs to exceed AR LLMs in throughput, substantially enhancing their practical applicability. More detailed hyperparameter settings are provided in Appendix D.

## 5.3 ABLATIONS AND ANALYSIS

This section presents ablation studies to dissect the contributions of key components within D2F. All ablation experiments are conducted on the D2F-Dream-Base-7B model unless otherwise specified.

![](images/1d9237016f653b3fc2ebc44135d6f9fc208cbed30ed542bf967548678bd7a0a6.jpg)

![](images/885b005548038a21049c4b4acfbbfe5ba2c15fa4a8a2766bde69fa320af67bdc.jpg)  
Figure 5: Ablation study on the block size during inference. The block size for inference is tested with integer multiples of the training block size (16). All experiments were conducted with a maximum length of 512, $\tau _ { \mathrm { c o n f } } = 0 . 9 ,$ $\tau _ { \mathrm { a d d } } = 0 . 1$ , and $\tau _ { \mathrm { a c t } } = 0 . 9 5$

<table><tr><td> $\tau_{\text{act}} = \tau_{\text{conf}}$ </td><td> $\tau_{\text{add}}$ </td><td>TPS ↑</td><td>Score ↑</td></tr><tr><td rowspan="4">0.95</td><td>0.95</td><td>105.2</td><td>76.9</td></tr><tr><td>0.7</td><td>107.2</td><td>77.2</td></tr><tr><td>0.5</td><td>106.3</td><td>77.3</td></tr><tr><td>0.1</td><td>104.0</td><td>77.7</td></tr><tr><td rowspan="4">0.90</td><td>0.90</td><td>124.5</td><td>74.7</td></tr><tr><td>0.7</td><td>126.2</td><td>75.7</td></tr><tr><td>0.5</td><td>124.7</td><td>76.2</td></tr><tr><td>0.1</td><td>122.1</td><td>76.4</td></tr><tr><td rowspan="4">0.85</td><td>0.85</td><td>136.8</td><td>72.6</td></tr><tr><td>0.7</td><td>139.0</td><td>74.2</td></tr><tr><td>0.5</td><td>138.5</td><td>74.0</td></tr><tr><td>0.1</td><td>135.4</td><td>75.0</td></tr></table>

Table 3: Ablation of inference hyperparameters on GSM8K-4-shot.

<table><tr><td>Benchmark</td><td> $\tau_{act}$ </td><td>Model</td><td>TPS ↑</td><td>Score ↑</td></tr><tr><td rowspan="6">MBPP 3-shot</td><td rowspan="2">0.95</td><td>random</td><td>147.2</td><td>49.6</td></tr><tr><td>D2F</td><td>171.2</td><td>54.6</td></tr><tr><td rowspan="2">0.90</td><td>random</td><td>148.5</td><td>48.6</td></tr><tr><td>D2F</td><td>175.5</td><td>53.2</td></tr><tr><td rowspan="2">0.85</td><td>random</td><td>150.6</td><td>47.6</td></tr><tr><td>D2F</td><td>177.5</td><td>52.6</td></tr><tr><td rowspan="6">GSM8k 4-shot</td><td rowspan="2">0.95</td><td>random</td><td>114.5</td><td>77.1</td></tr><tr><td>D2F</td><td>119.9</td><td>77.2</td></tr><tr><td rowspan="2">0.90</td><td>random</td><td>116.7</td><td>76.5</td></tr><tr><td>D2F</td><td>123.5</td><td>76.4</td></tr><tr><td rowspan="2">0.85</td><td>random</td><td>118.8</td><td>75.9</td></tr><tr><td>D2F</td><td>124.8</td><td>75.4</td></tr></table>

Table 4: Ablation of the noise scheduling on MBPP-3-shot and GSM8k-4-shot. Comparison of D2F and random noise schedules.

Throughput-performance trade-off. Figure 2 illustrates the throughput-performance trade-off for different methods on GSM8K and MBPP. D2F-Dream-Base-7B employs a fixed $\tau _ { \mathrm { a d d } } = 0 . 1$ and a block size of 32, with unified thresholds $\tau _ { \mathrm { a c t } } = \tau _ { \mathrm { c o n f } }$ being varied. Results demonstrate that D2F-Dream-Base-7B achieves significantly higher efficiency than baselines. For instance, on GSM8K, D2F-Dream-Base-7B attains 150.9 tokens/sec with a score of 71.2, achieving 3.1× the throughput of LLaMA3-Instruct-8B (48.0 tokens/sec) while exceeding its score (70.1). In contrast, Dream-Base-7B exhibits substantial performance degradation at higher throughput: reducing sampling steps from 512 to 128 causes its GSM8K score to drop from 71.4 to 42.8. This demonstrates the superior capability of D2F-Dream-Base-7B in maintaining performance during accelerated inference.

Effect of block size for inference. Figure 5 demonstrates the influence of block size on inference performance and speed. Overall, increasing block size consistently reduces throughput while yielding an initial performance improvement followed by deterioration. Optimal block size selection moderately reduces throughput but maximizes performance—for example, a block size of 48 achieves the peak GSM8K score of 77.5, surpassing smaller block sizes such as 16.

Ablation on inference hyperparameters. Table 3 studies $\tau _ { \mathrm { c o n f } } , \tau _ { \mathrm { a c t } } .$ , and $\tau _ { \mathrm { a d d } }$ . When $\tau _ { \mathrm { a d d } } = \tau _ { \mathrm { a c t } } ,$ new blocks are immediately activated (single-state). Our dual-state design $( \tau _ { \mathrm { a d d } } < \tau _ { \mathrm { a c t } } )$ introduces a conservative initial state and consistently performs better. For example, with $\tau _ { \mathrm { a c t } } = 0 . 8 5$ , lowering $\tau _ { \mathrm { a d d } }$ from 0.85 to 0.7 increases the score from 72.6 to 74.2 and throughput from 136.8 to 139.0 TPS; further reduction yields additional score gains with only slight throughput loss.

## 6 CONCLUSION

In this work, we introduce Discrete Diffusion Forcing (D2F), a novel training paradigm for dLLMs. D2F employs a generation scheme that conditions on partially predicted tokens from previous blocks to predict the next block, thereby supporting KV cache and enabling parallel generation across multiple blocks, resulting in significantly faster inference. Empirically, extensive experiments demonstrate that D2F achieves the milestone of being the first dLLM to support faster-than-AR inference.

## ETHICS STATEMENT

This work does not involve human subjects, animal experiments, or sensitive data. Therefore, it does not raise any ethical concerns.

## REPRODUCIBILITY STATEMENT

To promote reproducibility, we plan to release the source code for both training and inference of the proposed method in the future. The released code and accompanying instructions will enable other researchers to reproduce our results and build upon this work.

## ACKNOWLEDGEMENTS

This work was supported by Shanghai Key Technology R&D Program “New Generation of Information Technology” (No. 25511103700), NSF of China (Nos. 62306176, 92470118), Natural Science Foundation of Shanghai (No. 23ZR1428700), CCF-ALIMAMA TECH Kangaroo Fund (NO. CCF-ALIMAMA OF 2025010), and Ant Group.