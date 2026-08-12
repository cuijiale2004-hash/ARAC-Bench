## ABSTRACT

The utilization of large language models (LLMs) has become increasingly widespread, and has attracted considerable attention. Although Discrete Diffusion Large Language Models (dLLMs) mitigate the inference latency inherent in autoregressive LLM decoding, their computational overhead remains substantial. To address this challenge, we propose Hierarchy-dLLM, a hierarchical decoding framework inspired by the divide-and-conquer principle. Our method recursively partitions masked spans into smaller sub-decoding areas and decodes tokens according to their confidence, which substantially increases the number of tokens generated per forward pass and improves information utilization. Extensive experiments conducted on multiple benchmarks demonstrate that HierarchydLLM achieves accuracy comparable to or even surpassing existing baselines. Meanwhile, it is up to 17× faster than vanilla decoding and about 1.5× faster than the Fast-dLLM. These results establish hierarchical decoding as a practical solution for efficient dLLMs inference. The implementation is available at https://github.com/inclusionAI/dInfer.

## 1 INTRODUCTION

Although autoregressive (AR) large language models (Radford & Narasimhan, 2018) currently dominate the field, Diffusion large language models (dLLMs) (Yu et al., 2025a) are gaining momentum within the research community due to their unique potential for parallel decoding. In AR decoding, tokens are generated sequentially, which constrains efficiency and limits opportunities for parallelization. In contrast, dLLMs reconstruct linguistic sequences through iterative denoising with bidirectional attention, enabling simultaneous refinement of multiple tokens and thus parallel decoding (Li et al., 2022). Such a property not only improves scalability but also opens new research directions for developing more efficient and flexible decoding strategies.

In practice, however, comparable performance has yet to be observed in the open-source community, despite several commercial closed-source dLLMs claiming impressive throughput (Google Deep-Mind, 2025; Khanna et al., 2025; Song et al., 2025b). A key reason lies in the architectural trade-off of dLLMs: by adopting bidirectional attention, their per-step computation is significantly slower than that of autoregressive models of similar size. To compensate, dLLMs must achieve substantial gains from parallel decoding. However, representative open-source models such as LLaDA (Zhu et al., 2025) and Dream (HKUNLP, 2025) default to greedy decoding, generating only one token per step. This approach makes their efficiency fall short of AR models, underscoring why accelerating parallel decoding has become a central research focus for dLLMs.

Yet, attempts to scale up parallel decoding face intrinsic difficulties, often referred to as the curse of parallel decoding (Wu et al., 2025). This curse arises because tokens predicted within the same decoding step should satisfy a conditional independence assumption; otherwise, forcing them to be generated simultaneously can lead to substantial performance degradation. For example, given the sentence “In the classroom, Alice arranged pens, papers, and books neatly on her desk before the teacher began the lesson”, parallel prediction may produce incoherent outputs such as “pens, pens, and pens”, illustrating how naive parallel decoding can undermine semantic consistency.

While existing studies primarily focus on confidence-based criteria to determine which tokens should be decoded at each step, such approaches commonly ignore how the spatial distribution of undecoded positions affects the decoding process. To better understand this factor, we conducted a preliminary study and observed that when undecoded tokens are sparsely scattered across the sequence, one-pass decoding produces a token distribution that closely matches step-by-step generation. In contrast, when undecoded tokens form consecutive spans, the resulting distribution exhibits substantial divergence from step-wise generation, leading to pronounced distributional drift. These observations highlight the necessity of incorporating spatial structure into decoding strategies for diffusion-based language models.

To address this challenge, we introduce Hierarchy-dLLM, a novel hierarchical parallel decoding framework inspired by the divide-and-conquer paradigm. Rather than treating all masked positions equally, Hierarchy-dLLM dynamically partitions masked tokens into independent subordinate decoding areas according to the positions of decoded tokens. The proposed decoding strategy is executed independently across individual decoding areas, which allows multiple areas to be decoded in parallel and leads to a significant improvement in overall decoding efficiency.

## Our main contributions can be summarized as follows:

1. We performed a comprehensive analysis of the dLLM decoding mechanism. We found that preserving a sparse layout of undecoded tokens within the sequence can effectively reduce distributional drift, thus improving the stability and accuracy of parallel decoding.

2. We propose Hierarchy-dLLM, to the best of our knowledge the first position-based decoding framework for diffusion-based large language models, which systematically leverages divide-and-conquer principles to enhance parallel decoding.

3. We conduct extensive experiments demonstrating that Hierarchy-dLLM achieves superior trade-offs between inference speed and generation quality compared with existing opensource dLLM baselines, running up to 17× faster than vanilla decoding and 1.5× faster than Fast-dLLM.

![](images/02c5372ef888b911ba493f599fe04b6ffaeae4d081bd66c0fcea3f788b8e1aaa.jpg)  
(a) Pre-study

![](images/503893f97dedf30188cc9f767a383de8930b8b2d9e59fe33b257343e50562549.jpg)  
(b) HumanEval

![](images/e79b40daed9824eaf0cbac4d048841da66873f7ef5ae9b47345e2a0f54a7c8f2.jpg)  
(c) GSM8k  
Figure 1: Average KL-Divergence of Masked Tokens Over Number of Tokens. (a) shows two masking methods used in our study: Consecutive Mask, where tokens are masked as a contiguous block, and Sparse Mask, where masked tokens are scattered across multiple positions. (b) and (c) report the average KL divergence $( \mathrm { m e a n } \pm \mathrm { S E M } )$ of masked tokens under these two strategies on HumanEval and GSM8k official answers, respectively. The results indicate that Consecutive Mask generally yields a larger KL divergence than Sparse Mask, suggesting that scattered masking provides more stable token-level predictions across decoding steps.

## 2 PRELIMINARY STUDY

## 2.1 DLLM INFERENCE PROCESS

Within the framework of dLLMs, the current mainstream instantiation is the Masked Diffusion Model (MDM) (Shi et al., 2025). We therefore focus our discussion on MDMs in this subsection. Unlike traditional autoregressive models (ARMs) that rely on the chain rule for left-to-right prediction, MDMs construct probabilistic models via masked token prediction, thereby naturally supporting bidirectional context modeling and alleviating several limitations of ARMs such as reversal reasoning difficulties and temporal distribution shifts.

Problem setting. Let V denote a fixed vocabulary. We define a sequence $X = ( x ^ { 1 } , x ^ { 2 } , \dots , x ^ { L } )$ where each element $x ^ { i } \in \nu$ represents a token drawn from the vocabulary, and $L \in \mathbb { N }$ denotes the length of the sequence. MDMs define aforward masking process that progressively replaces tokens with a special mask symbol [MASK]. At time $t \in [ 0 , 1 ]$ , the noisy sequence $X _ { t }$ is sampled as

$$
q _ {t | 0} (X _ {t} | X _ {0}) = \prod_ {i = 1} ^ {L} q _ {t | 0} (x _ {t} ^ {i} | x _ {0} ^ {i}), \quad q _ {t | 0} (x _ {t} ^ {i} | x _ {0} ^ {i}) = \left\{ \begin{array}{l l} 1 - t, & x _ {t} ^ {i} = x _ {0} ^ {i}, \\ t, & x _ {t} ^ {i} = M. \end{array} \right.\tag{1}
$$

As $t \to 1$ , the sequence becomes fully masked.

Reverse process. The reverse process recovers the original data distribution by iteratively predicting masked tokens:

$$
q _ {s | t} (X _ {s} | X _ {t}) = \prod_ {i = 1} ^ {L} q _ {s | t} (x _ {s} ^ {i} | X _ {t}),\tag{2}
$$

where

$$
q _ {s | t} (x _ {s} ^ {i} | X _ {t}) = \left\{ \begin{array}{l l} 1, & x _ {t} ^ {i} \neq M,   x _ {s} ^ {i} = x _ {t} ^ {i}, \\ \frac {s}{t}, & x _ {t} ^ {i} = M,   x _ {s} ^ {i} = M, \\ \frac {t - s}{t}   q _ {0 | t} (x _ {s} ^ {i} | X _ {t}), & x _ {t} ^ {i} = M,   x _ {s} ^ {i} \neq M, \\ 0, & \text {otherwise.} \end{array} \right.\tag{3}
$$

Decoding process. During generation, MDMs start from a fully masked sequence $( t = 1 )$ and gradually denoise toward $t = 0 ,$ . Let $p _ { 0 }$ denote the prompt, let $r _ { t }$ denote a fully masked sequence, and let $c _ { i }$ denote the confidence score of the i-th masked token in $r _ { t }$ . Then, the start state of decoding process can be denoted as $X _ { t } ~ = ~ c o n c a t ( p _ { 0 } , r _ { t } )$ . At each step, the model assigns a predictive distribution over the true values of selected masked tokens:

$$
x _ {s} ^ {i} = \arg \max p _ {\theta} (X _ {s} \mid X _ {t}),\tag{4}
$$

and a proportion $\frac { s } { t }$ of the tokens remain masked according to their confidence, such that only one token is decoded in each step when $\frac { s } { t }$ is scheduled accordingly, and the reverse process remains consistent with the forward process. Importantly, $\frac { s } { t }$ is a tunable parameter that controls the tradeoff between speed and fidelity: smaller values correspond to more tokens being decoded at once (fewer steps, higher efficiency), whereas larger values yield fewer tokens per step (more steps, better generation quality).

Semi-Autoregressive Diffusion Decoding. To further enhance quality and controllability, Semi-Autoregressive Diffusion Decoding (SADD) has been introduced. The idea is to divide the sequence into multiple blocks and generate them sequentially from left to right. Within each block, however, the MDM reverse process (with random or low-confidence remasking) is applied in parallel. This hybrid approach combines the global consistency of diffusion with the sequential structure of autoregression, yielding better performance on complex reasoning and dialogue tasks. This hybrid strategy has been employed in recent dLLMs such as LLaDA (Nie et al., 2025a) and MMaDA (Yang et al., 2025a).

## 2.2 PARALLEL DECODING ANALYSIS

DLLMs are designed to exploit their parallel decoding ability to accelerate the decoding process of LLMs, but most open-source dLLMs fall short of expectations because of their incompatibility between parallelism and accuracy. During decoding, the sampling procedure defined in Equation 4 produces only the marginal distribution for each token, $p ( x _ { s } ^ { i } \mid \mathbf { x } _ { t } )$ , for $i = \{ 1 , \ldots , L \}$ . However, parallel decoding requires access to the joint distribution over multiple tokens to be decoded simultaneously: $p ( x _ { j } ^ { 1 } , x _ { j } ^ { 2 } , \cdot \cdot \cdot , x _ { j } ^ { k } \mid \mathbf { X } _ { t } )$ , where k denotes the number of tokens generated in one parallel decoding step. This discrepancy gives rise to a methodological challenge, namely that parallel decoding must approximate the joint distribution using only the available marginals $p \big ( x _ { i } ^ { i } \mid \bar { \mathbf { X } } _ { t } \big )$ Designing effective approximation strategies for bridging this gap constitutes a central problem in the development of parallel decoding algorithms.

To gain empirical insights into this theoretical inconsistency, we investigate how the positional distribution of previously decoded tokens affects the decoding process. In principle, the consistency of different decoding strategies can be quantitatively assessed by the Kullback–Leibler divergence $\mathrm { K L } ( p _ { \mathrm { s t e p w i s e } } ( \mathbf { x } ) \parallel p _ { \mathrm { o n e - p a s s } } ( \mathbf { x } ) )$ ). However, directly computing the standard KL divergence over multistep generation is computationally intractable. Therefore, we employ an approximation where n denotes the number of masked tokens, and their step-by-step generation is regarded as the ground truth. Specifically, given the logits $z ^ { \mathrm { s t e p } }$ from stepwise decoding, we take the most probable token index

$$
i ^ {*} = \arg \max _ {v \in \mathcal {V}} z _ {v} ^ {\mathrm{step}},\tag{5}
$$

and approximate the ground-truth distribution as one-hot, i.e., $p _ { \mathrm { s t e p w i s e } } ( v ) \approx \mathbf { 1 } _ { [ v = i ^ { * } ] }$ . For the onepass prediction, we compute

$$
p _ {\text { one - pass }} (v) = \operatorname{softmax} (z ^ {\text { once }}) _ {v}.\tag{6}
$$

Under this approximation, the KL divergence reduces to the negative log-likelihood of the one-pass model at $i ^ { * } { } ;$ :

$$
\mathrm{KL} _ {\text { approx }} = - \sum_ {j = 1} ^ {n} \log p _ {\text { one - pass }} (i _ {j} ^ {*}),\tag{7}
$$

where $i _ { j } ^ { * }$ is the argmax token in the j-th masked position determined by stepwise decoding. This formulation can be interpreted as a cross-entropy surrogate of the KL divergence, where one-hot targets make the comparison tractable and are standard in language modeling (Goldberg, 2017).

We evaluate the proposed approximation on two representative benchmarks: HumanEval (Chen et al., 2021), which focuses on code generation, and GSM8k (Cobbe et al., 2021), which targets mathematical reasoning. The experimental results are presented in Figure 1. On both benchmarks, the approximate KL divergence under the consecutive masking strategy increases steadily as the number of masked tokens grows, with a markedly faster growth rate compared to the sparse masking strategy. In contrast, sparse masking maintains consistently low divergence across decoding steps. This observation suggests that sparse masking allows dLLMs to make better use of bidirectional self-attention. Specifically, by leaving unmasked anchor positions interleaved throughout the input, sparse masking enables the model to attend to reliable contextual signals from both left and right neighborhoods of each masked position, thereby improving the robustness and consistency of parallel decoding.

These results suggest that the sparse mask offers substantial advantages in mitigating distributional shift and maintaining decoding stability. This empirical robustness is consistent with our preliminary observation: when most tokens in a sequence are already decoded, the undecoded tokens approximate the ground truth more closely if they are sparsely scattered across the text rather than being continuously clustered. Motivated by this finding, we hypothesize that if undecoded tokens can be structurally organized to mimic such sparse distributions through an appropriate decoding strategy, it becomes possible to accelerate the generation process while preserving or even improving decoding accuracy.

## 3 METHODOLOGY

Building on our preliminary study, we find that sparse masking—where undecoded tokens remain sparsely scattered—suppresses distributional shift and stabilizes decoding. This effect arises because sparsity enables more effective use of bidirectional attention, guiding predictions toward the ground truth. Motivated by this observation, we propose a divide-and-conquer framework that partitions undecoded sequences into smaller subproblems, allowing parallel resolution that both accelerates generation and reduces bias.

![](images/84c47f8ac30aebc4c03b01e9851707c550a073873b0ab929d705e2dc800ba988.jpg)  
Figure 2: Illustration of Hierarchy-dLLM. The decoding process starts by initializing a decoding area, then decoding tokens based on confidence, and partitioning sub-areas according to the decoded tokens. Steps 2 and 3 are iteratively applied within each sub-decoding area, enabling efficient multitoken decoding while preserving accuracy.

## 3.1 DIVIDE-AND-CONQUER DECODING STRUCTURE

To achieve efficient and stable text generation, we design the model with a divide-and-conquer decoding structure, which progressively resolves masked spans through an iterative process of initialization, decoding, and subdivision. This design seeks to balance decoding efficiency with generation accuracy by breaking down the complex decoding task into smaller, well-structured units. The hierarchical organization prevents the model from predicting overly dependent tokens in the same step, converting the difficulty of parallel decoding into a series of more tractable sub-problems.

Initialization stage. Before decoding begins in each block, the block is represented as a contiguous span of masked tokens with a predefined length l. This masked span serves as the initial subdecoding areas, providing a structured starting point for the generation process.

Decoding stage. Each sub-decoding area is processed independently and in parallel, following the decoding strategy introduced in Section 3.2. The objective is to maximize decoding efficiency while trying to ensure that each block yields at least one valid token. By decoding in parallel across multiple sub-areas, the model inherently possesses significant structural potential for acceleration.

Subdivision stage. Tokens generated in the previous step act as anchors to partition the remaining masked regions. Every contiguous span of undecoded tokens is split into smaller, independent subdecoding areas, which are then processed in the next iteration. This recursive partitioning gradually reduces the decoding problem to smaller segments, simplifying the generation task.

The decoding and subdivision stages are repeated iteratively until no masked tokens remain in any decoding block. Through this iterative refinement process, the model incrementally resolves all masked positions while maintaining stability and coherence in the generated sequence. Overall, the divide-and-conquer decoding structure provides a principled framework that achieves O(log n)-level acceleration under ideal conditions while fully exploiting the rich contextual information inherent in the bidirectional attention mechanism of dLLMs. This design not only ensures substantial efficiency gains but also preserves decoding accuracy, thereby offering a scalable and reliable foundation for subsequent stages of our model.

## 3.2 DECODING STRATEGIES WITHIN SUB-DECODING AREAS

The greatest challenge in the divide-and-conquer structure lies in how to decode effectively within each sub-decoding area. Our objective is to decode as many tokens as possible at each step while minimizing the risk of introducing errors that propagate through later iterations. To formalize this, let $p _ { \theta } ( x _ { s } ^ { i } \mid \bar { X } _ { t } )$ denote the model’s posterior probability of predicting token $x _ { s } ^ { i }$ at position i in the s-th decoding step, given the current corrupted sequence $X _ { t }$ . We then define the confidence score for position i as

$$
c ^ {i} = \max _ {v \in \mathcal {V}} p _ {\theta} (x _ {s} ^ {i} = v \mid X _ {t}).\tag{8}
$$

A natural starting point is to decode tokens only when they are sufficiently reliable. Let A denote a sub-decoding area. Concretely, whenever $c ^ { i }$ surpasses a high threshold $\tau _ { \mathrm { h i g h } } .$

$$
x _ {s} ^ {i} = \arg \max _ {v \in \mathcal {V}} p _ {\theta} (x _ {s} ^ {i} = v \mid X _ {t}), \quad \text { if } \quad c ^ {i} \geq \tau_ {\text { high }}, \quad i \in \mathcal {A},\tag{9}
$$

the corresponding token is committed to the sequence. This simple rule favors semantic stability, since only high-confidence tokens are introduced, and it also allows multiple positions within A to be decoded in parallel when the evidence is strong.

Yet in practice, some sub-decoding areas may not contain any tokens above the high threshold, as the underlying probability distribution over the vocabulary can be relatively flat in these regions, leaving all candidate tokens with comparable but low confidence scores. If we decode nothing in such cases, progress slows dramatically; if we force a decision, performance can suffer. To mitigate this trade-off, we relax the condition: when no position meets equation 9, we still allow one token to be decoded, namely the most confident candidate in ${ \mathcal { A } } ,$ , provided that its confidence exceeds a lower threshold $\tau _ { \mathrm { l o w } }$

$$
x _ {s} ^ {i ^ {*}} = \arg \max _ {v \in \mathcal {V}} p _ {\theta} (x _ {s} ^ {i ^ {*}} = v \mid X _ {t}), \quad \text {if} \quad i ^ {*} = \arg \max _ {i \in \mathcal {A}} c ^ {i}, \quad c ^ {i ^ {*}} \geq \tau_ {\mathrm{low}}.\tag{10}
$$

This adaptive rule ensures that each area contributes meaningfully while preventing the premature incorporation of extremely unreliable predictions.

These two conditions, however, may occasionally leave an iteration without any decoded tokens. This typically happens in more challenging cases, where the model cannot confidently commit to a prediction, so that the probability distribution over the vocabulary is relatively flat across positions and all candidate tokens fall below the relaxed threshold $\tau _ { \mathrm { l o w } }$ . To avoid stalling, we enforce steady progress by always decoding the globally most confident position when necessary:

$$
x _ {s} ^ {i ^ {\dagger}} = \arg \max _ {v \in \mathcal {V}} p _ {\theta} (x _ {s} ^ {i ^ {\dagger}} = v \mid X _ {t}), \quad \text { if } \quad i ^ {\dagger} = \arg \max _ {i \in \{1, \ldots , L \}} c ^ {i}.\tag{11}
$$

This fallback guarantees that every step yields at least one decoded token.

Finally, as decoding advances, early predictions can become inconsistent with the evolving context, reflected by a noticeable confidence drop. To adaptively correct such cases, we introduce a remasking step: before repartitioning into sub-decoding areas, all decoded tokens are checked, and any token with $c ^ { i } < \tau _ { \mathrm { r e m a s k } }$ is replaced by the mask symbol [MASK],

$$
x _ {s} ^ {i} = [ M A S K ] \quad \text { if } \quad c ^ {i} <   \tau_ {\text { remask }}.\tag{12}
$$

This prevents error accumulation and helps maintain global coherence throughout the sequence.

Taken together, our decoding strategy begins with a strict high-threshold rule, then gradually relaxes through a controlled low-threshold selection, incorporates a fallback to guarantee steady progress, and finally applies a remasking step to revise unreliable predictions. In following this progressively relaxed procedure, the strategy adheres to a best-effort principle, since it encourages decoding whenever trustworthy evidence is available while postponing or correcting low-confidence tokens, thereby balancing efficiency, token-level reliability, and contextual coherence under the bidirectional attention mechanism of dLLMs.

## 4 EXPERIMENTS

## 4.1 EXPERIMENT SETTINGS

We implement the proposed Hierarchy-dLLM framework on three open-source models: llada-instruct-8B, llada-1.5-8B, and Dream-7B, and evaluate it on four widely used benchmarks: GSM8K and MATH500 (Lightman et al., 2023) for mathematical reasoning, HumanEval and MBPP (Austin et al., 2021) for code generation, and IF-Eval (Zhou et al., 2023) for instruction following, with few-shot settings adopted in accordance with Nie et al. (2025b) and Zhu et al. (2025). To provide a comprehensive assessment of performance and efficiency, we compare Hierarchy-dLLM against vanilla autoregressive decoding, the parallel decoding scheme of FastdLLM (Wu et al., 2025), and a more complex multi-stage decoding method of WINO (Hong et al., 2025). All experiments are run on a single NVIDIA H20 GPU. Unless otherwise specified, the block size is set to 32 and the generation length to 512. For hyperparameter tuning, we conduct a grid search where $\tau _ { \mathrm { h i g h } }$ ranges from 0.78 to 0.88, τ ranges from 0.3 to 0.5, and $\tau _ { \mathrm { r e m a s k } }$ is chosen in 0.01, 0.3 and 0.35. The exact settings and implementation details are available in our released code. We report performance using Pass@1 accuracy, and efficiency is measured with tokens per forward call (TPF) and throughput per second (TPS). Note that TPS excludes the eos token, and for consistency, we also exclude the eos token in TPF. All evaluations are conducted with the lm-eval (Gao et al., 2024) library to ensure consistency and reproducibility.

Table 1: Main results of Hierarchy-dLLM on the LLaDA-Instruct-8B model and LLaDA-1.5- 8B model across five benchmarks. We report task performance (Accuracy Score) and decoding efficiency. Efficiency is measured by TPS (throughput per second), reflecting practical throughput, and TPF (tokens per forward call), indicating how many tokens are decoded per model invocation. Values in parentheses denote the relative performance change compared to the baseline and the speedup factor with respect to decoding efficiency. The best performance and highest TPS/TPF are highlighted in bold.

<table><tr><td rowspan="3">Task</td><td rowspan="3">Method</td><td colspan="3">LLaDA-Instruct-8B</td><td colspan="3">LLaDA-1.5-8B</td></tr><tr><td>Performance</td><td colspan="2">Speed</td><td>Performance</td><td colspan="2">Speed</td></tr><tr><td>Score ↑</td><td>TPF ↑</td><td>TPS ↑</td><td>Score ↑</td><td>TPF ↑</td><td>TPS ↑</td></tr><tr><td rowspan="4">GSM8K</td><td>Vanilla</td><td>77.26</td><td>0.52</td><td>2.35</td><td>83.17</td><td>0.65</td><td>1.99</td></tr><tr><td>Fast-dLLM</td><td>77.86(0.6+)</td><td>2.85(5.48×)</td><td>12.80(5.45×)</td><td>83.32(0.15+)</td><td>3.10(4.77×)</td><td>9.54(4.79×)</td></tr><tr><td>WINO</td><td>68.75(8.51-)</td><td>3.24(6.23×)</td><td>10.95(4.66×)</td><td>81.25(1.92-)</td><td>3.97(6.10×)</td><td>8.00(4.02×)</td></tr><tr><td>Hierarchy-dLLM</td><td>77.18(0.08-)</td><td>3.79(7.29×)</td><td>19.41(8.26×)</td><td>83.70(0.53+)</td><td>4.25(6.54×)</td><td>14.83(7.45×)</td></tr><tr><td rowspan="4">Math500</td><td>Vanilla</td><td>41.20</td><td>0.83</td><td>7.87</td><td>39.80</td><td>0.84</td><td>7.96</td></tr><tr><td>Fast-dLLM</td><td>40.60(0.6-)</td><td>2.71(3.27×)</td><td>24.99(3.17×)</td><td>39.40(0.4-)</td><td>2.79(3.32×)</td><td>25.81(3.24×)</td></tr><tr><td>WINO</td><td>37.40(3.8-)</td><td>3.78(4.55×)</td><td>30.73(3.90×)</td><td>41.00(1.2+)</td><td>3.75(4.46×)</td><td>30.48(3.83×)</td></tr><tr><td>Hierarchy-dLLM</td><td>41.60(0.4+)</td><td>3.53(4.25×)</td><td>37.34(4.74×)</td><td>41.60(1.8+)</td><td>3.99(4.75×)</td><td>42.25(5.31×)</td></tr><tr><td rowspan="4">HumanEval</td><td>Vanilla</td><td>43.90</td><td>0.93</td><td>8.56</td><td>43.29</td><td>0.93</td><td>8.56</td></tr><tr><td>Fast-dLLM</td><td>43.90(0+)</td><td>2.94(3.16×)</td><td>27.12(3.17×)</td><td>42.07(1.22-)</td><td>2.97(3.19×)</td><td>27.45(3.21×)</td></tr><tr><td>WINO</td><td>43.90(0+)</td><td>3.80(4.08×)</td><td>28.15(3.29×)</td><td>42.68(0.61-)</td><td>3.84(4.12×)</td><td>28.53(3.33×)</td></tr><tr><td>Hierarchy-dLLM</td><td>44.51(0.61+)</td><td>3.93(4.23×)</td><td>41.52(4.85×)</td><td>45.12(1.83+)</td><td>4.20(4.52×)</td><td>44.18(5.16×)</td></tr><tr><td rowspan="4">MBPP</td><td>Vanilla</td><td>37.60</td><td>0.13</td><td>0.65</td><td>40.20</td><td>0.16</td><td>0.80</td></tr><tr><td>Fast-dLLM</td><td>37.60(0+)</td><td>1.73(10.81×)</td><td>7.26(11.17×)</td><td>40.40(0.2+)</td><td>1.73(10.81×)</td><td>8.40(10.5×)</td></tr><tr><td>WINO</td><td>38.40(0.8+)</td><td>1.27(9.76×)</td><td>4.56(7.02×)</td><td>34.80(5.4-)</td><td>1.37(8.56×)</td><td>4.92(6.15×)</td></tr><tr><td>Hierarchy-dLLM</td><td>37.60(0+)</td><td>2.03(15.62×)</td><td>11.20(17.23×)</td><td>40.40(0.2+)</td><td>2.29(14.31×)</td><td>12.70(15.88×)</td></tr><tr><td rowspan="4">IF-Eval</td><td>Vanilla</td><td>57.67</td><td>0.59</td><td>6.28</td><td>58.78</td><td>0.59</td><td>6.29</td></tr><tr><td>Fast-dLLM</td><td>57.30(0.37-)</td><td>1.45(2.46×)</td><td>15.46(2.46×)</td><td>58.23(0.55+)</td><td>1.38(2.34×)</td><td>14.43(2.29×)</td></tr><tr><td>WINO</td><td>55.64(2.03-)</td><td>1.64(2.78×)</td><td>14.33(2.28×)</td><td>58.60(0.18-)</td><td>1.59(2.69×)</td><td>13.95(2.22×)</td></tr><tr><td>Hierarchy-dLLM</td><td>57.12(0.45-)</td><td>1.82(3.08×)</td><td>22.14(3.53×)</td><td>59.15(0.37+)</td><td>1.74(2.95×)</td><td>21.21(3.37×)</td></tr></table>

## 4.2 MAIN RESULTS

Across five benchmarks and three models, Hierarchy-dLLM consistently delivers strong accuracy while achieving the highest decoding efficiency.

Based on the experimental results on LLaDA-1.5-8B and LLaDA-Instruct-8B in 1, Hierarchy-dLLM achieves the best of both worlds—higher accuracy than both vanilla and Fast-dLLM baselines while providing the fastest decoding. The method shows the most notable gains on mathematical reasoning tasks of GSM8K and Math500, where it not only accelerates throughput by up to 17× over baselines but also improves accuracy by about 1 point, indicating its ability to mitigate error accumulation in long reasoning chains. On code generation tasks of HumanEval and MBPP, Hierarchy-dLLM maintains or slightly improves accuracy while substantially increasing speed, with TPF gains up to 10×, underscoring its suitability for deterministic token generation.

On the Dream-7B model(see Table 2), we observe that Hierarchy-dLLM still brings comparable speedup gains, achieving large improvements in both TPF and TPS across all four benchmarks while maintaining stable performance. This confirms that the hierarchical mechanism consistently enhances decoding efficiency even on models trained with different architectures. However, compared to LLaDA, the absolute performance of Dream-7B with Hierarchy-dLLM achieves comparable speedups while its accuracy drop is no larger than, and sometimes smaller than, Fast-dLLM. We attribute this to Dream originating from an autoregressive base model, which provides weaker inherent support for parallel decoding and thus limits quality preservation compared to models with stronger parallelism.

Overall, Hierarchy-dLLM offers a unified acceleration framework that simultaneously improves or preserves task performance while substantially reducing inference cost across diverse settings.

Table 2: Main results of Hierarchy-dLLM on the Dream-7B model across four benchmarks.

<table><tr><td rowspan="2">Task</td><td rowspan="2">Method</td><td>Performance</td><td colspan="2">Speed</td></tr><tr><td>Score ↑</td><td>TPF ↑</td><td>TPS ↑</td></tr><tr><td rowspan="3">GSM8K</td><td>Vanilla</td><td>75.8</td><td>1.00</td><td>4.76</td></tr><tr><td>Fast-dLLM</td><td>73.8(2-)</td><td>2.08(2.08×)</td><td>9.30(1.95×)</td></tr><tr><td>Hierarchy-dLLM</td><td>73.2(2.6-)</td><td>2.61(2.61×)</td><td>12.39(2.60×)</td></tr><tr><td rowspan="3">Math500</td><td>Vanilla</td><td>17.8</td><td>1.00</td><td>10.41</td></tr><tr><td>Fast-dLLM</td><td>12.0(5.8-)</td><td>2.32(2.32×)</td><td>21.45(2.06×)</td></tr><tr><td>Hierarchy-dLLM</td><td>16.0(1.8-)</td><td>3.17(3.17×)</td><td>32.54(3.13×)</td></tr><tr><td rowspan="3">HumanEval</td><td>Vanilla</td><td>54.9</td><td>1.00</td><td>9.69</td></tr><tr><td>Fast-dLLM</td><td>50.6(4.3-)</td><td>2.08(2.08×)</td><td>18.12(1.87×)</td></tr><tr><td>Hierarchy-dLLM</td><td>51.8(3.1-)</td><td>2.61(2.61×)</td><td>25.32(2.61×)</td></tr><tr><td rowspan="3">MBPP</td><td>Vanilla</td><td>56.8</td><td>1.0</td><td>5.97</td></tr><tr><td>Fast-dLLM</td><td>54.6(2.2-)</td><td>4.35(4.35×)</td><td>23.57(3.96×)</td></tr><tr><td>Hierarchy-dLLM</td><td>54.4(2.4-)</td><td>5.06(5.06×)</td><td>29.89(5.01×)</td></tr></table>

## 4.3 ABLATION STUDY AND ANALYSIS

Unless otherwise specified, all ablation studies are conducted on the GSM8k dataset using the LLaDA-1.5-8B model. The generation length is fixed to 512 and the block length to 32, with all other hyperparameters kept identical to those in Section 4.1.

Impact of different Generation Length. To investigate the impact of block length, we fix the generation length to 512 and evaluate the performance of the baseline model and Hierarchy-dLLM with block sizes of 16, 32, and 64. As shown in Table 3, while the performance of both methods remains stable across different block sizes, Hierarchy-dLLM consistently achieves significantly higher TPF and TPS compared to vanilla decoding. Moreover, its advantage becomes more pronounced as the block length increases, demonstrating that the hierarchical design effectively sustains high throughput under larger decoding blocks.

Impact of different Block Length. As reported in Table 4, the TPS of dLLMs decreases as the generation length grows, which can be attributed to the bidirectional self-attention mechanism: longer sequences require more computation per decoding step. Although Hierarchy-dLLM is also affected, the slowdown is considerably mitigated compared to vanilla decoding. Consequently, the speedup ratio of Hierarchy-dLLM increases with generation length, while its performance exhibits a similar upward trend to the baseline, demonstrating stable efficiency and effectiveness under longer decoding scenarios.

Effects of adjusting the threshold and low-threshold hyperparameters. The effect of threshold and low threshold settings on performance and speed is examined when remasking is disabled. Figure 3a reports the score and TPS as the high threshold $\tau _ { \mathrm { h i g h } }$ varies, with low threshold $\tau _ { \mathrm { l o w } }$ fixed at 0.3. Figure 3b shows the counterpart results when $\tau _ { \mathrm { l o w } }$ varies with $\tau _ { \mathrm { h i g h } }$ fixed at 0.82. Across both settings, the performance of Hierarchy-dLLM remains relatively stable at a high level, indicating robustness to threshold choices. By contrast, TPS is more sensitive, which means increasing τ<sub>high</sub> leads to a noticeable decline in efficiency, while changes in $\tau _ { \mathrm { l o w } }$ only cause minor variations in TPS. These findings suggest that selecting a moderately small $\tau _ { \mathrm { h i g h } }$ is crucial to balancing accuracy and efficiency, whereas $\tau _ { \mathrm { l o w } }$ has a negligible impact, reflecting that the model is resilient to uncertainty pruning in low-confidence regions.

Table 3: Performance and Speed on GSM8K with Different Generation Lengths.

<table><tr><td rowspan="2">Gen Length</td><td rowspan="2">Method</td><td>Performance</td><td colspan="2">Speed</td></tr><tr><td>Score ↑</td><td>TPF ↑</td><td>TPS ↑</td></tr><tr><td rowspan="2">256</td><td>LLaDA-1.5-8B</td><td>82.29</td><td>0.97</td><td>4.13</td></tr><tr><td>Hierarchy-dLLM</td><td>81.34</td><td>4.38 (4.52×)</td><td>18.53 (4.49×)</td></tr><tr><td rowspan="2">512</td><td>LLaDA-1.5-8B</td><td>83.17</td><td>0.65</td><td>1.99</td></tr><tr><td>Hierarchy-dLLM</td><td>83.70</td><td>4.25 (6.54×)</td><td>14.83 (6.45×)</td></tr><tr><td rowspan="2">1024</td><td>LLaDA-1.5-8B</td><td>84.38</td><td>0.26</td><td>0.76</td></tr><tr><td>Hierarchy-dLLM</td><td>84.31</td><td>3.09 (11.88×)</td><td>9.06 (11.92×)</td></tr></table>

Table 4: Performance and Speed on GSM8K with Different Block Lengths

<table><tr><td rowspan="2">Block Length</td><td rowspan="2">Method</td><td>Performance</td><td colspan="2">Speed</td></tr><tr><td>Score ↑</td><td>TPF ↑</td><td>TPS ↑</td></tr><tr><td rowspan="2">16</td><td>LLaDA-1.5-8B</td><td>83.40</td><td>0.69</td><td>2.22</td></tr><tr><td>Hierarchy-dLLM</td><td>81.58</td><td> $3.52(5.10 \times)$ </td><td> $12.89(5.81 \times)$ </td></tr><tr><td rowspan="2">32</td><td>LLaDA-1.5-8B</td><td>83.17</td><td>0.66</td><td>2.13</td></tr><tr><td>Hierarchy-dLLM</td><td>83.70</td><td> $4.25 (6.44 \times)$ </td><td> $14.83 (6.96 \times)$ </td></tr><tr><td rowspan="2">64</td><td>LLaDA-1.5-8B</td><td>83.85</td><td>0.64</td><td>1.98</td></tr><tr><td>Hierarchy-dLLM</td><td>81.35</td><td> $4.69 (7.23 \times)$ </td><td> $17.18 (8.68 \times)$ </td></tr></table>

![](images/d79da78fad524d83b872f3fe6bbb7d741584beff191c9323021f6d81c8219f81.jpg)  
(a) Score and TPS vs. $\pi _ { \mathrm { h i g h } }$

![](images/f1d7eecda64812a97c099e7c8967396bd59fc48672a5a937a7058ebba322d6cb.jpg)  
(b) Score and TPS vs. $\boldsymbol { \mathcal { n } } _ { \mathrm { { o w } } }$

![](images/a1585baf5ab8cd116cb26dc2f98bb181097638cdf1f4359ab18d0b0dd8cb32ba.jpg)  
(c) Score vs. TPF(with eos)  
Figure 3: Joint Analysis of Accuracy and Efficiency On GSM8K (a) Score and TPS with varying τ<sub>high</sub> while fixing $\tau _ { \mathrm { l o w } } = 0 . 3 .$ . (b) Score and TPS with varying $\tau _ { \mathrm { l o w } }$ while fixing $\tau _ { \mathrm { h i g h } } = 0 . 8 2 .$ (c) Comparison of score versus TPF between vanilla parallel decoding and Hierarchy-dLLM.

Comparison with naive parallel sampling. We further compare Hierarchy-dLLM with a vanilla parallel decoding strategy where a fixed number of tokens, including the eos token, are sampled in each step so that the total token count divided by sampling steps matches the intended parallel factor. As shown in Figure 3c, increasing the number of tokens per step rapidly degrades the performance of vanilla parallel decoding despite the speed gain, leading to a poor speed–accuracy trade-off. In contrast, Hierarchy-dLLM maintains consistently high performance even under high TPF, while preserving substantial speed improvements. This demonstrates that Hierarchy-dLLM achieves a more favorable balance between efficiency and accuracy compared to naive parallel decoding.

## 5 RELATED WORKS

## 5.1 DIFFUSION LARGE LANGUAGE MODELS

Diffusion modeling for language has recently gained significant attention. Early work extended score matching to discrete spaces with score entropy to build Score Entropy Discrete Diffusion models (Lou et al., 2024). To simplify and generalize this paradigm, masked diffusion formulations reinterpreted objectives as weighted cross-entropy losses, leading to improved training and flexibility (Shi et al., 2025). Scaling studies further demonstrated that Masked Diffusion Models follow comparable scaling laws to autoregressive methods and enable efficient compute–quality trade-offs (Nie et al., 2025a).

At larger scale, diffusion has been positioned as a credible foundation model alternative: LLaDA shows diffusion-based LLMs trained with pretraining and fine-tuning can rival strong AR models like LLaMA and even surpass GPT-4o on specific tasks (Nie et al., 2025b); follow-up work LLaDA2.0 (Bie et al., 2025) further scales this line via AR-to-dLLM conversion up to 100B parameters with block-level diffusion training, and LLaDA2.1 (Bie et al., 2026) improves the speed–quality trade-off by combining Mask-to-Token diffusion with Token-to-Token editing and RL-based alignment for dLLMs.

Extensions to robotics and multimodality include LLaDA-VLA (Wen et al., 2025), which adapts diffusion VLMs for vision-language-action policy learning, and MMaDA (Yang et al., 2025a), which introduces a unified multimodal diffusion architecture with shared reasoning and generation abilities. Complementary to the above, DREAM (HKUNLP, 2025) explores discrete reasoning in autoregressive models to improve controllability and reasoning quality.

## 5.2 ACCELERATION TECHNIQUES FOR DLLMS

Recent work on efficient inference for dLLMs has rapidly expanded along two main axes: (i) parallel decoding strategies, and (ii) caching techniques that reduce the cost of bidirectional attention and KV storage. A number of system-level frameworks further integrate these algorithmic advances into unified inference stacks.

Parallel Decoding Strategies To accelerate dLLM inference, researchers leverage token confidence. Fast-dLLM (Wu et al., 2025) proposes a training-free, confidence-aware parallel decoding for dLLMs. CreditDecoding (Wang et al., 2025) utilizes “Trace Credits” based on historical logits to accelerate the convergence of underconfident tokens. LocalLeap (Kong et al., 2025) employs local determinism propagation around high-confidence anchors to reduce decoding steps. Dimple (Yu et al., 2025b) applies a confident decoding strategy dynamically adjusting token generation counts to significantly reduce iterations. Several approaches introduce verification steps to ensure quality while accelerating inference. WINO (Hong et al., 2025) introduces a revocable decoding mechanism that aggressively drafts tokens and uses bidirectional context to re-mask and refine errors. Adaptive Parallel Decoding (Israel et al., 2025) inverts standard speculation by mixing dLLM marginals with a small autoregressive model to dynamically adjust sample sizes. Other works focus on sampling dynamics and model structure. SlowFast Sampling (Wei et al., 2025) alternates between exploratory and accelerated decoding stages based on certainty and convergence principles. Addressing fixedlength limitations, dLLM-Var (Yang et al., 2025b) enables native variable generation lengths by training the model to predict [EOS] tokens.

KV Cache Optimization Given the bidirectional attention costs in dLLMs, efficient cache management is critical. Fast-dLLM (Wu et al., 2025) introduces a block-wise approximate KV cache tailored for bidirectional models, enabling cache reuse with negligible performance drop alongside confidence-aware decoding. DLLM-Cache (Liu et al., 2025) proposes a training-free framework that combines long-interval prompt caching with partial response updates guided by feature similarity. To address memory constraints, Sparse-dLLM (Song et al., 2025a) leverages persistent cross-layer sparsity to dynamically evict unimportant tokens while retaining pivotal ones, maintaining performance with reduced memory usage. D<sup>2</sup>Cache (Jiang et al., 2025) reuses KV states for stable tokens via a dual adaptive cache. MaskKV (Huang et al., 2025) leverages mask-token–guided scoring and adaptive budgeting for fine-grained KV compression.

## 6 CONCLUSION

In this work, we introduced Hierarchy-dLLM, a hierarchical decoding framework that applies the divide-and-conquer principle to accelerate large language model inference. By recursively partitioning masked spans into smaller sub-decoding areas and decoding tokens according to confidence, our method effectively increases the number of tokens generated per step, thereby improving information utilization. Experiments on multiple benchmarks show that Hierarchy-dLLM maintains comparable or even better accuracy than existing approaches, while achieving up to 17× speedup over vanilla decoding and about 1.5× faster than Fast-dLLM. These results demonstrate that hierarchical, divide-and-conquer decoding provides a practical and scalable solution for efficient diffusion-based generation. While our current framework is entirely training-free, an exciting future direction is to perform post-training adaptations so that the model distribution better aligns with hierarchical decoding, potentially further enhancing both efficiency and effectiveness.