# 1. Research Background and Existing Pain Points
**Research Background:**
Large Language Models (LLMs) have become foundational in numerous applications, yet their deployment is often hindered by inference latency. The predominant autoregressive framework generates text one token at a time, imposing a sequential constraint that limits speed and scalability. This has driven interest toward parallel decoding strategies. Diffusion-based Large Language Models (dLLMs) offer a promising alternative by eliminating sequential dependencies. Formulating text generation as a parallel denoising process, they can predict entire sequences or generate text block-wise (i.e., semi-autoregressively). 

**Existing Pain Points:**
While dLLMs can generate multiple tokens in parallel, the resulting throughput gains are undermined by a disproportionate increase in computation, posing a key bottleneck to their widespread adoption. At each step, predictions for all future (suffix) tokens are computed, though only a small fraction are retained. Through systematic empirical analysis, the paper reveals that suffix tokens act as a non-semantic information reservoir, or scratchpad. While this scratchpad collects contextual signals across Transformer layers to guide the generation of the current block, it is highly inefficient. Most suffix tokens are redundant and low-entropy, a problem that worsens with distance as their attention scores decay sharply. This redundancy not only creates significant computational overhead but can also degrade generation fidelity. Uncontrolled filling of all future suffix tokens with low-entropy, repetitive content distracts the model's attention and wastes storage and computation, making it neither sustainable nor scalable.

# 2. Core Research Motivation and Scientific Questions
**Core Research Motivation:**
The core motivation is to address the high computational overhead caused by full suffix attention in dLLMs—a key bottleneck resulting from redundant computation. By understanding the role of suffix tokens as a "scratchpad," the authors aim to restrict attention to a structured subset of suffix tokens, preserving fidelity while eliminating redundancy. The motivation extends to proving that sparsity is an inherent property of suffix attention, allowing for training-free inference strategies that can discover "winning" suffix tokens on the fly without sacrificing generation quality.

**Scientific Questions:**
1. How can the inherent inefficiency of full suffix attention in dLLMs be mitigated without degrading generation accuracy?
2. What is the exact mechanism by which suffix tokens contribute to the denoising process in dLLMs?
3. Can a sparse subset of suffix tokens suffice for high-quality generation, and how can this subset be determined a priori without computing exact attention scores?
4. How does the distance-decay pattern of suffix attention inform an efficient and proactive dropout strategy that is orthogonal to existing dLLM optimizations?

# 3. Overall Core Idea and Design Philosophy
**Overall Core Idea:**
The paper proposes the Diffusion Scratchpad (DPad), a training-free method that restricts attention to a structured subset of suffix tokens. DPad integrates two strategies: (i) a sliding window, which maintains a fixed-length suffix window, and (ii) distance-decay dropout, which deterministically removes distant suffix tokens before attention computation. 

**Design Philosophy:**
The design philosophy is likened to a real scratchpad used when writing a book. For the current chapter (i.e., block), the writer (dLLM) devotes the most attention, revising it multiple times, akin to denoising within a diffusion block. The upcoming chapter receives focused drafts for consistency, while much later chapters contain only brief outlines. The "writer" should not fill all future chapters (all suffix tokens) with low-entropy, repetitive content merely to satisfy fixed sequence length constraints. DPad leverages the inherent sparsity of suffix attention, predetermining a distance-decay sparse pattern for suffix tokens prior to model execution. This orthogonal design ensures compatibility with existing optimizations like parallel decoding and prefix caching, eliminating substantial redundancy and compounding efficiency gains.

# 4. Core Innovation Points
1. **Identification and Formalization of the Scratchpad Mechanism:** The paper identifies and formalizes the Scratchpad mechanism in dLLMs, demonstrating that suffix tokens act as a dynamic, cross-layer reservoir (termed an Attention Connection) that guides the denoising process.
2. **Revelation of Suffix Token Properties:** Through systematic empirical analysis, the paper is the first to reveal three key properties of suffix tokens: inherent sparsity, a distance-decay pattern, and position insensitivity, which expose a major yet overlooked inefficiency in dLLMs.
3. **Diffusion Lottery Tickets (DLT) Hypothesis:** The paper introduces the Diffusion Lottery Tickets hypothesis for dLLM inference, positing that a sparse set of "winning" suffix tokens suffices for high-quality generation, and frames DPad as a training-free method to discover such tickets on the fly.
4. **Training-Free Distance-Decay Dropout (DPad):** The paper proposes DPad, a training-free method that applies distance-decay dropout to suffix tokens. This orthogonal design eliminates substantial redundancy and compounds with existing optimizations, achieving up to 61.39× speedup while preserving accuracy.

# 5. Overview of the Overall Technical Solution
The overall technical solution, DPad, is a training-free inference strategy that restricts suffix attention to a small number of near-suffix tokens. It uses two suffix drop strategies: a sliding window and distance-decay dropout. For the sliding window, inspired by prefix KV-cache optimizations, the suffix window has a fixed length and moves forward along with the current block, retaining only a limited number of nearby suffix tokens. Unlike vanilla dLLMs where suffix computation increases with sequence length, this design keeps the cost bounded.

For distance-decay dropout, suffix tokens are removed according to their distance from the current block: the farther they are, the higher the dropout ratio, until all tokens beyond the window are omitted. Unlike conventional attention-score pruning, which first computes attention scores and then prunes based on magnitude, DPad predetermines a distance-decay sparse pattern for suffix tokens prior to model execution and eliminates them at the beginning of each step. Both mechanisms are efficiently realized through a Gaussian sampling process. Furthermore, a minor adjustment to Rotary Positional Embeddings (RoPE) ensures correct positional encoding under suffix dropout, and an early termination mechanism halts generation upon detecting an end-of-sequence token.

# 6. Detailed Module Design
**Scratchpad Mechanism:**
The token sequence is partitioned into three contiguous segments: Prefix indices $[0, c-1]$, current indices $[c, s-1]$, and suffix indices $[s, L-1]$. The corresponding attention matrix is divided into $3 \times 3$ blocks. Blocks 7 and 8 collect information from the prefix and current into the suffix at layer $n$, while Block 6 feeds this stored information back to the current block at layer $(n+1)$. This path enables the current block to reuse the information collected in the suffix at the previous layer, where the suffix serves as temporary memory to assist the ongoing denoising process. The role of suffix tokens resembles a residual connection specialized for attention, termed an "Attention Connection." The model compresses high-dimensional signals from both the prefix and the current tokens into the suffix, which is then re-injected into the current block at the subsequent layer.

**Suffix Dropout Strategies:**
- **Sliding Window:** A fixed-length window retains only a bounded number of near-suffix tokens. The suffix dropout window is decoupled from the overall sequence length, reducing suffix-related complexity by one dimension.
- **Distance-Decay Dropout:** Progressively prunes distant suffix tokens. Implemented via a Gaussian sampling process, which simultaneously enforces a bounded window and distance-dependent decay. As a result, suffix attention only needs to focus on nearby tokens.

**RoPE Adjustment:**
Suffix dropout requires a minor adjustment to RoPE to maintain correct positional information. Rather than using re-indexed positions after dropout, a mapping function $f(i_k)$ retrieves the original absolute position of the $k$-th preserved token. This adjustment requires only a lightweight remapping inside the attention module.

**Early Termination:**
An early termination technique halts generation upon detecting an end-of-sequence (<eos>) token. This addresses the computational redundancy inherent in fixed-length generation models that continue generating redundant tokens to fill the token budget.

# 7. All Mathematical Formulas and Symbol Definitions
**Forward Masking Process and Reverse Unmasking Process Loss:**
$$L(\theta) = -E_{t,x_0,x_t} \left[ \frac{1}{t} \sum_{i:x_i^t=<M>} \log p_\theta(x_0^i|x_t) \right]$$

**Inference Probability Distribution and Confidence Scores:**
$$\hat{y}_i = \arg\max_{v \in V} p_\theta(y_i = v|y_{s-1}) \quad \text{and} \quad c_i = p_\theta(y_i = \hat{y}_i|y_{s-1})$$

**Global Attention Scores:**
$$A^{(n)} = \text{Softmax}\left(\frac{Q^{(n)}(K^{(n)})^\top}{\sqrt{d_k}}\right) \in \mathbb{R}^{L \times L}$$

**Suffix to Prefix/Current Attention Scores:**
$$A^{(n)}_{S,P} = A^{(n)}[s : L, 0 : c], \quad A^{(n)}_{S,C} = A^{(n)}[s : L, c : s]$$

**Suffix Hidden Representations After Attention:**
$$H^{(n)}_S = A^{(n)}_{S,P}V^{(n)}_P + A^{(n)}_{S,C}V^{(n)}_C + A^{(n)}_{S,S}V^{(n)}_S$$

**Current-to-Suffix Attention at Next Layer:**
$$A^{(n+1)}_{C,S} = A^{(n+1)}[c : s, s : L]$$

**Retention Probability for Suffix Token at Distance $d$:**
$$P(d) = a \cdot \frac{1}{\sigma\sqrt{2\pi}} \exp\left[-\frac{1}{2}\left(\frac{k\sigma}{W} \cdot d - \mu_\sigma\right)^2\right], \quad 0 < d \leq W$$
where $\mu_\sigma = 0$, $\sigma = 1$, $W$ is the effective window size, $k$ maps the window size $W$ to $k\sigma$, $a$ scales the overall sampling magnitude vertically, and the suffix boundary is the center of the Gaussian distribution.

**Modified RoPE Angle:**
$$\theta'_{i_k} = f(i_k) \cdot \Delta$$

**Modified RoPE Application:**
$$\text{RoPE}'(x_{i_k}, i_k) = \text{RoPE}(x_{i_k}, f(i_k))$$

**Revised Learning Objective for SFT (Discussion):**
$$L_{DPad}(\theta) = -E_{x_0,t,M} \left[ \frac{1}{t} \sum_{i \in C} \log p_\theta\left(x_0^i | x_{(I_R \cup I_P \cup M)}\right) \right]$$
where $M$ denotes a subset of masked suffix tokens sampled via distance-decay dropout at each training step, $I_R$ represents the prompt tokens, $I_P$ represents the generated prefix tokens, $\theta$ represents the model parameters, $t \in [0, 1]$ is the degree of masking of samples from the forward masking process, and the loss is computed over the current block tokens $C$.

# 8. Algorithm Pseudocode
**Algorithm 1 DPad Suffix-Dropout Inference Step**
Input: Indices of tokens $\in x$: $P$ (prefix), $C$ (current block), $S$ (suffix); window $W$; decay params $(k, a)$; model $p_\theta$
Output: model output for this step
1: /*select a near-suffix window */
2: $S_W \leftarrow$ the first $W$ indices of $S$
3: /* Select suffix indices using distance-decay sampling */
4: $K \leftarrow \{j \in S_W \mid u_j \leq P(d_j)\}$ where $d_j=j - \max(C)- 1$ (dist to current) , $u_j \sim \text{Uniform}(0, 1)$, and $P(d_j) = \min\{1, a(2\pi)^{-1/2} \exp(-\frac{1}{2} (\frac{k}{W} d_j)^2)\}$
5: scratchpad $\leftarrow x[K]$
6: $x' \leftarrow \text{concat}(x[P], x[C], \text{scratchpad})$
7: output $\leftarrow p_\theta(x')$
8: return output