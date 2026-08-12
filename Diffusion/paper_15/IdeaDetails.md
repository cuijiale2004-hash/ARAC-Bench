### 1. Research Background and Existing Pain Points

The utilization of large language models (LLMs) has become increasingly widespread and has attracted considerable attention. Although autoregressive (AR) large language models currently dominate the field, Diffusion large language models (dLLMs) are gaining momentum within the research community due to their unique potential for parallel decoding. In AR decoding, tokens are generated sequentially, which constrains efficiency and limits opportunities for parallelization. In contrast, dLLMs reconstruct linguistic sequences through iterative denoising with bidirectional attention, enabling simultaneous refinement of multiple tokens and thus parallel decoding. Such a property not only improves scalability but also opens new research directions for developing more efficient and flexible decoding strategies.

In practice, however, comparable performance has yet to be observed in the open-source community. A key reason lies in the architectural trade-off of dLLMs: by adopting bidirectional attention, their per-step computation is significantly slower than that of autoregressive models of similar size. To compensate, dLLMs must achieve substantial gains from parallel decoding. However, representative open-source models such as LLaDA and Dream default to greedy decoding, generating only one token per step. This approach makes their efficiency fall short of AR models, underscoring why accelerating parallel decoding has become a central research focus for dLLMs.

Yet, attempts to scale up parallel decoding face intrinsic difficulties, often referred to as the curse of parallel decoding. This curse arises because tokens predicted within the same decoding step should satisfy a conditional independence assumption; otherwise, forcing them to be generated simultaneously can lead to substantial performance degradation. For example, given the sentence “In the classroom, Alice arranged pens, papers, and books neatly on her desk before the teacher began the lesson”, parallel prediction may produce incoherent outputs such as “pens, pens, and pens”, illustrating how naive parallel decoding can undermine semantic consistency. While existing studies primarily focus on confidence-based criteria to determine which tokens should be decoded at each step, such approaches commonly ignore how the spatial distribution of undecoded positions affects the decoding process. When undecoded tokens form consecutive spans, the resulting distribution exhibits substantial divergence from step-wise generation, leading to pronounced distributional drift.

### 2. Core Research Motivation and Scientific Questions

The core research motivation stems from the necessity to accelerate the inference process of dLLMs while preserving or even surpassing generation accuracy. Existing dLLMs suffer from substantial computational overhead due to bidirectional attention, and naive parallel decoding degrades performance due to the curse of parallel decoding and distributional drift. 

The scientific questions addressed are: How can the spatial distribution of undecoded tokens be leveraged to mitigate distributional drift in parallel decoding? How can a divide-and-conquer paradigm be systematically integrated into dLLM decoding to break down the complex joint distribution approximation into tractable sub-problems? Can a position-based decoding framework that recursively partitions masked spans into smaller sub-decoding areas increase the number of tokens generated per forward pass without sacrificing accuracy?

### 3. Overall Core Idea and Design Philosophy

The overall core idea is Hierarchy-dLLM, a novel hierarchical parallel decoding framework inspired by the divide-and-conquer paradigm. Rather than treating all masked positions equally, Hierarchy-dLLM dynamically partitions masked tokens into independent subordinate decoding areas according to the positions of decoded tokens. 

The design philosophy is grounded in the observation that preserving a sparse layout of undecoded tokens within the sequence can effectively reduce distributional drift, thus improving the stability and accuracy of parallel decoding. Sparse masking allows dLLMs to make better use of bidirectional self-attention; by leaving unmasked anchor positions interleaved throughout the input, sparse masking enables the model to attend to reliable contextual signals from both left and right neighborhoods of each masked position. Motivated by this, Hierarchy-dLLM structurally organizes undecoded tokens to mimic sparse distributions through recursive partitioning, converting the difficulty of parallel decoding into a series of more tractable sub-problems. This allows multiple areas to be decoded in parallel, leading to a significant improvement in overall decoding efficiency while balancing token-level reliability and contextual coherence.

### 4. Core Innovation Points

1. Comprehensive analysis of the dLLM decoding mechanism: The study found that preserving a sparse layout of undecoded tokens within the sequence can effectively reduce distributional drift, thus improving the stability and accuracy of parallel decoding. It was shown that Consecutive Mask generally yields a larger KL divergence than Sparse Mask.
2. Proposal of Hierarchy-dLLM: To the best of our knowledge, the first position-based decoding framework for diffusion-based large language models, which systematically leverages divide-and-conquer principles to enhance parallel decoding.
3. Extensive experiments demonstrating superior trade-offs: Hierarchy-dLLM achieves superior trade-offs between inference speed and generation quality compared with existing open-source dLLM baselines, running up to 17× faster than vanilla decoding and 1.5× faster than Fast-dLLM.

### 5. Overview of the Overall Technical Solution

The overall technical solution is a divide-and-conquer decoding structure that progressively resolves masked spans through an iterative process of initialization, decoding, and subdivision. This design seeks to balance decoding efficiency with generation accuracy by breaking down the complex decoding task into smaller, well-structured units. The hierarchical organization prevents the model from predicting overly dependent tokens in the same step. The structure operates in three stages:
1. **Initialization stage**: Before decoding begins in each block, the block is represented as a contiguous span of masked tokens with a predefined length, serving as the initial sub-decoding areas.
2. **Decoding stage**: Each sub-decoding area is processed independently and in parallel, maximizing efficiency while ensuring each block yields at least one valid token based on confidence thresholds.
3. **Subdivision stage**: Tokens generated in the previous step act as anchors to partition the remaining masked regions. Every contiguous span of undecoded tokens is split into smaller, independent sub-decoding areas for the next iteration.
This process is repeated iteratively until no masked tokens remain, achieving O(log n)-level acceleration under ideal conditions.

### 6. Detailed Module Design

**6.1 DLLM Inference Process Module (Masked Diffusion Model)**
Unlike traditional ARMs that rely on the chain rule, MDMs construct probabilistic models via masked token prediction, supporting bidirectional context modeling. The forward masking process progressively replaces tokens with a special mask symbol. The reverse process recovers the original data distribution by iteratively predicting masked tokens. During generation, MDMs start from a fully masked sequence and gradually denoise. A proportion of the tokens remain masked according to their confidence, controlling the trade-off between speed and fidelity. Semi-Autoregressive Diffusion Decoding (SADD) is also incorporated, dividing the sequence into blocks generated sequentially left-to-right, while within each block, the MDM reverse process is applied in parallel.

**6.2 Parallel Decoding Analysis Module**
Parallel decoding must approximate the joint distribution over multiple tokens using only the available marginals. To assess consistency, an approximate KL divergence is computed where step-by-step generation is regarded as the ground truth. The ground-truth distribution is approximated as one-hot based on the stepwise logits, and the one-pass prediction is computed via softmax. The KL divergence reduces to the negative log-likelihood of the one-pass model at the stepwise argmax token index. Empirical evaluation shows that sparse masking maintains consistently low divergence across decoding steps compared to consecutive masking.

**6.3 Decoding Strategies within Sub-Decoding Areas Module**
The objective is to decode as many tokens as possible at each step while minimizing error propagation.
- **High-threshold rule**: Tokens are committed when their confidence score surpasses a high threshold $\tau_{high}$.
- **Low-threshold rule**: When no position meets the high threshold, the most confident candidate in the area is decoded, provided its confidence exceeds a lower threshold $\tau_{low}$.
- **Global fallback**: If no tokens are decoded by the above rules, the globally most confident position is decoded to guarantee steady progress.
- **Remasking step**: Before repartitioning, all decoded tokens with confidence $c_i < \tau_{remask}$ are replaced by the mask symbol to prevent error accumulation.

### 7. All Mathematical Formulas and Symbol Definitions

**Symbol Definitions:**
- $V$: Fixed vocabulary.
- $X = (x_1, x_2, \ldots, x_L)$: Sequence of length $L$ where $x_i \in V$.
- $M$: Special mask symbol.
- $t \in [0, 1]$: Time parameter.
- $X_t$: Noisy sequence at time $t$.
- $p_0$: Prompt.
- $r_t$: Fully masked sequence.
- $c_i$: Confidence score of the i-th masked token.
- $s_t$: Proportion of tokens that remain masked.
- $A$: Sub-decoding area.
- $\tau_{high}$: High confidence threshold.
- $\tau_{low}$: Low confidence threshold.
- $\tau_{remask}$: Remasking confidence threshold.

**Formulas:**
Forward masking process:
$$q_{t|0}(X_t|X_0) = \prod_{i=1}^L q_{t|0}(x_t^i|x_0^i), \quad q_{t|0}(x_t^i|x_0^i) = \begin{cases} 1-t, & x_t^i = x_0^i \\ t, & x_t^i = M \end{cases}$$

Reverse process:
$$q_{s|t}(X_s|X_t) = \prod_{i=1}^L q_{s|t}(x_i^s|X_t)$$

Marginal reverse transition:
$$q_{s|t}(x_i^s|X_t) = \begin{cases} 1, & x_t^i \neq M, x_i^s = x_t^i \\ \frac{s}{t}, & x_t^i = M, x_i^s = M \\ \frac{t-s}{t} q_{0|t}(x_i^s|X_t), & x_t^i = M, x_i^s \neq M \\ 0, & \text{otherwise} \end{cases}$$

Decoding step:
$$x_i^s = \arg\max p_\theta(X_s | X_t)$$

Stepwise argmax token index:
$$i^* = \arg\max_{v \in V} z_{step, v}$$

One-pass prediction:
$$p_{one-pass}(v) = \text{softmax}(z_{once})_v$$

Approximate KL divergence:
$$KL_{approx} = -\sum_{j=1}^n \log p_{one-pass}(i_j^*)$$

Confidence score:
$$c_i = \max_{v \in V} p_\theta(x_i^s = v | X_t)$$

High-threshold decoding:
$$x_i^s = \arg\max_{v \in V} p_\theta(x_i^s = v | X_t), \quad \text{if } c_i \geq \tau_{high}, i \in A$$

Low-threshold decoding:
$$x_{i^*}^s = \arg\max_{v \in V} p_\theta(x_{i^*}^s = v | X_t), \quad \text{if } i^* = \arg\max_{i \in A} c_i, c_{i^*} \geq \tau_{low}$$

Global fallback decoding:
$$x_{i^\dagger}^s = \arg\max_{v \in V} p_\theta(x_{i^\dagger}^s = v | X_t), \quad \text{if } i^\dagger = \arg\max_{i \in \{1,\ldots,L\}} c_i$$

Remasking step:
$$x_i^s = \langle \text{mask} \rangle \quad \text{if } c_i < \tau_{remask}$$

Unified decoding rule:
$$x_i^s = \begin{cases} \arg\max_{v \in V} p_\theta(x_i^s = v | X_t), & \text{if } c_i \geq \tau_{high}, i \in A \\ \arg\max_{v \in V} p_\theta(x_{i^*}^s = v | X_t), & \text{if } i^* = \arg\max_{j \in A} c_j, c_{i^*} \geq \tau_{low} \\ \arg\max_{v \in V} p_\theta(x_{i^\dagger}^s = v | X_t), & \text{if } i^\dagger = \arg\max_{j \in \{1,\ldots,L\}} c_j \\ \langle \text{mask} \rangle, & \text{otherwise} \end{cases}$$

### 8. Algorithm Pseudocode

**Algorithm 1: Hierarchy-dLLM**
Input: Prompt: $p$; block length: $L$; number of blocks: $N$; thresholds: $\tau_{low}, \tau_{high}, \tau_{remask}$
Output: Decoded sequence

Initialize sequence $x \leftarrow p \parallel \langle \text{mask} \rangle^{L \times N}$;
for $i \leftarrow 1$ to $N$ do
    // Select the i-th block of $L$ masks
    Let $B \leftarrow$ positions of masks $[(i-1)L+1, iL]$ in $x$;
    while not all tokens in $B$ decoded do
        // Remask step (only within block $B$)
        foreach token $x_j \in B$ do
            if $\text{conf}(x_j) < \tau_{remask}$ then
                set $x_j \leftarrow \langle \text{mask} \rangle$;
        Divide block $B$ into sub-decoding areas;
        $D \leftarrow \emptyset$;
        // decoded tokens in this iteration
        foreach sub-decoding area $A \subseteq B$ do
            $t^* \leftarrow \arg\max_{t \in A} \text{conf}(t)$;
            $c^* \leftarrow \text{conf}(t^*)$;
            if $c^* \geq \tau_{high}$ then
                decode $t^*$; add to $D$;
            if $t^*$ is maximal in $A$ and $c^* \geq \tau_{low}$ then
                decode $t^*$; add to $D$;
        if $D = \emptyset$ then
            decode token with highest confidence across the entire block $B$;
return decoded sequence;