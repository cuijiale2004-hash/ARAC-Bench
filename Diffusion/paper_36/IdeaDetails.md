**1. Research Background and Existing Pain Points**

Autoregressive (AR) large language models (LLMs) have achieved remarkable performance across a wide range of natural language tasks by modeling next-token prediction. However, AR models suffer from inherent inefficiencies: since tokens are generated one-by-one in a strict left-to-right order, they cannot exploit full parallelism during decoding, which limits inference efficiency. 

Diffusion-based language models (dLLMs) offer a promising alternative by allowing multiple tokens or entire blocks of tokens to be predicted or refined jointly, achieving much higher decoding parallelism in principle. Nevertheless, in practice, they come with significant drawbacks: they often cannot use KV cache effectively due to bidirectional attention, their inference latency often exceeds that of AR models, and many require fixed sequence lengths or have restricted flexibility in generation length. Some works employ approximate KV cache mechanisms to reuse computation, such as the DualCache in Fast-dLLM. However, this does not fundamentally resolve incompatibility of dLLMs with KV cache, since such approximate caches are not equivalent to the original computation. 

Block diffusion language models (such as BD3-LMs) have been proposed to interpolate between purely autoregressive and diffusion regimes by generating tokens in blocks, performing diffusion within each block, while conditioning on previous blocks autoregressively. This achieves flexible sequence length and KV caching between blocks. However, BD3-LMs have so far only been validated on relatively small-scale models and conventional LM metrics, rather than modern large-scale LLM settings. Their practical applicability to state-of-the-art LLMs remains unclear, especially in terms of maintaining high-quality text generation and robust scaling behavior. Furthermore, full-attention diffusion models such as Dream require on the order of 500B tokens for fine-tuning, making adaptation highly costly.

**2. Core Research Motivation and Scientific Questions**

The core research motivation is to bridge the autoregressive and diffusion paradigms to achieve both high generation quality and high inference efficiency, enabling the practical deployment of fast and accurate LLMs. The scientific questions addressed are: How to efficiently adapt pretrained AR models into block-diffusion frameworks without full retraining from scratch? How to design a block-wise attention mechanism that is compatible with AR training objectives while enabling block-wise bidirectional context modeling? How to resolve the inference latency of dLLMs and enable effective KV caching while maintaining parallel generation within blocks?

**3. Overall Core Idea and Design Philosophy**

The overall core idea is Fast-dLLM v2, a carefully designed block diffusion language model that efficiently adapts pretrained AR models into dLLMs for parallel text generation. The design philosophy centers on an AR-friendly block-wise attention structure that is closer to the original AR models, making the adaptation process inherently more compatible and data-efficient. It introduces a novel training recipe that combines a block diffusion mechanism with a complementary attention mask, enabling blockwise bidirectional context modeling without sacrificing AR training objectives. To enhance inference speed, it designs a hierarchical caching mechanism (a block-level cache and a sub-block cache) coupled with a parallel decoding pipeline to maximize decoding efficiency.

**4. Core Innovation Points (List all innovations in the original text completely)**

1. Identify the AR-friendly nature of the block-wise attention design and leverage it to present a post-training strategy for adapting pretrained AR models into block-diffusion frameworks, requiring only affordable fine-tuning rather than full retraining. Specifically, Fast-dLLM v2 achieves lossless adaptation with just ~1B tokens, compared to ~500B tokens required by Dream.
2. Introduce a novel training recipe that combines a block diffusion mechanism with a complementary attention mask, enabling block-wise bidirectional context modeling while simultaneously preserving the original AR training objectives and predictive performance.
3. Introduce an inference strategy that combines a hierarchical caching mechanism with block-wise parallel decoding. This design enables effective reuse of context across blocks and accelerates token generation within each block, yielding substantially faster inference than prior diffusion-based methods.
4. Conduct comprehensive large-scale experiments on models up to 7B parameters and diverse tasks, showing that Fast-dLLM v2 achieves up to 2.5× speedup over standard AR decoding while maintaining comparable generation quality.

**5. Overview of the Overall Technical Solution**

The overall technical solution builds a block-wise diffusion training pipeline on top of pretrained Qwen2.5-Instruct models (1.5B and 7B). Fine-tuning is conducted as supervised fine-tuning (SFT) on instruction-tuning data, where each training batch is constructed using a blockwise diffusion setup. It introduces partial token masking within each block together with a complementary masking strategy to ensure that every token is trained in both visible and masked contexts. 

Sequences are padded and packed into a block-aligned long token stream, divided into non-overlapping blocks of size D. For each block, a binary mask is randomly sampled, and the training sample is duplicated into two views with masks m and m̄ = 1−m. A token shift strategy is adopted where prediction of a masked token at position i uses the logit from its preceding position (i − 1). The model is trained using a masked-token-only cross-entropy loss with a specialized block-wise attention mask.

At inference time, Fast-dLLM v2 employs a block-wise decoding strategy that balances the autoregressive nature of LLMs with the parallelism afforded by diffusion-based decoding. Generation proceeds one block at a time: previously decoded blocks are cached and reused as clean prefix context, while the current block undergoes parallel masked token refinement using a confidence-aware parallel decoding strategy. DualCache is integrated for sub-block reuse to further reduce redundant computation.

**6. Detailed Module Design (If any, complete mechanisms of each layer/sub-module)**

**6.1 Block-wise organization**
Given a set of tokenized samples, each sequence is padded to a length that is an integer multiple of the block size D by appending <MASK> tokens as needed. These padding tokens are ignored in the loss computation and do not contribute to gradient updates. After this alignment step, the padded sequences are packed by concatenating them into a long token stream and splitting it into training sequences of a fixed context length L. Each packed sequence is naturally divided into B = L/D non-overlapping blocks of size D, already aligned by construction.

**6.2 Masked token prediction with complementary views**
For each block, a binary mask m ∈ {0, 1}D is randomly sampled, where mj = 1 means position j is replaced with a learned <MASK> embedding. To ensure all tokens receive both masked and unmasked supervision across training, a complementary masking strategy is used: each training sample is duplicated into two views with masks m and m̄ = 1−m. These two views are placed together in the same batch, so the model can jointly see masked and unmasked contexts across the views.

**6.3 Token shift for prediction**
To preserve the pretrained AR model’s representation quality, a shifted-label strategy is adopted: prediction of a token at masked position i uses the logit from its preceding position (i − 1). Concretely, if xi is masked, the model uses the hidden state at i − 1 to predict xi, consistent with the next-token prediction mechanism in causal language models. This allows dLLM to maintain AR-like temporal representations while supporting intra-block diffusion.

**6.4 Block-wise attention masking**
For each training sample, the noised sequence xt and its corresponding clean sequence x0 are concatenated along the sequence dimension, resulting in a total length of 2L. The attention mask A ∈ {0, 1}2L×2L is then applied to control both causal and bidirectional connections. The overall attention mask can be decomposed into four sub-masks:
Mfull = [[MBD, MOBC], [0, MBC]]
- MBD (Block-diagonal mask): Provides bidirectional self-attention among tokens within the same block in the noised sequence xt.
- MOBC (Offset block-causal mask): Allows each noised token in xt to attend to tokens from previous blocks in the clean sequence x0, preserving inter-block causal conditioning.
- MBC (Block-causal mask): Enables each token in the clean sequence x0 to attend to all previous and current block positions, facilitating autoregressive-like progression.

**6.5 Inference Pipeline**
- *Block-wise autoregressive decoding with caching*: Since each block is decoded in a causal order, left-to-right semantics across blocks are naturally preserved. After decoding each block, its unmasked tokens are cached as read-only context for future blocks. This design enables block-level Key-Value (KV) cache reuse and significantly reduces redundant computation. The attention mask at inference time allows each block to attend bidirectionally within itself, while attending causally to preceding blocks.
- *Parallel refinement within each block*: To accelerate generation within a block, the confidence-aware parallel decoding strategy is adopted. Iteratively refine masked tokens in the current block based on their model confidence of predicted tokens. Tokens exceeding a confidence threshold are decoded and unmasked in parallel, while uncertain positions remain masked for future refinement.
- *DualCache for sub-block reuse*: DualCache maintains both prefix and suffix KV caches for partially decoded blocks, enabling efficient recomputation as additional tokens are revealed. This hierarchical caching not only rules out expensive recomputation, but also supports the iterative, selective decoding pattern used in confidence-aware refinement.
- *Batch decoding with padding*: To support batch generation of sequences with varying target lengths, each sequence is right-padded with <MASK> tokens to make their total lengths divisible by the block size D. The sequences are then grouped and decoded block-by-block.

**7. All Mathematical Formulas and Symbol Definitions (If any, replicate exactly as in the original text)**

- $x = \{x_1, x_2, \ldots, x_L\}$: Token sequence of length L.
- $P_\theta(x_i | x_{<i})$: Conditional distribution modeled by autoregressive models.
- $x_t$: Corrupted sequence at time t, where each token in x0 is masked independently with probability t.
- $p_\theta(x_0 | x_t)$: Reverse model predicting the original tokens given the noised input.
- Standard MDM Training Loss:
$L(\theta) = -\mathbb{E}_{t, x_0, x_t} \left[ \frac{1}{t} \sum_{i=1}^L 1[x_i^t = \text{<MASK>}] \log p_\theta(x_0^i | x_t) \right]$
- $m \in \{0, 1\}^D$: Binary mask for a block of size D.
- $\bar{m} = 1 - m$: Complementary mask.
- Block Diffusion Training Objective:
$L_{\text{block}}(\theta) = -\mathbb{E}_{x, m} \left[ \sum_{i=1}^L 1[x_i^t = \text{<MASK>}] \log p_\theta(x_0^i | x_{<i}, x_{\text{block}(i)}) \right]$
where $x_{\text{block}(i)}$ denotes all tokens in the block containing position i (including masked/unmasked), and $x_{<i}$ are clean tokens from earlier blocks.
- Complementary Masking Loss Sum:
$- \left[ \sum_{i=1}^L 1[x_i^t = \text{<MASK>}] \log p_\theta(x_0^i | x_{<i}, x_{\text{block}(i)}) \right] + \left[ \sum_{i=1}^L 1[x_i^{1-t} = \text{<MASK>}] \log p_\theta(x_0^i | x_{<i}, x_{\text{block}(i)}) \right]$
- Attention Mask Definitions:
$M_{\text{full}} = \begin{bmatrix} M_{BD} & M_{OBC} \\ 0 & M_{BC} \end{bmatrix}$
$[M_{BD}]_{ij} = \begin{cases} 1 & \text{if } i, j \text{ belong to the same block} \\ 0 & \text{otherwise} \end{cases}$
$[M_{OBC}]_{ij} = \begin{cases} 1 & \text{if } j \text{ is in a block before } i \\ 0 & \text{otherwise} \end{cases}$
$[M_{BC}]_{ij} = \begin{cases} 1 & \text{if } j \text{ is in the same or an earlier block as } i \\ 0 & \text{otherwise} \end{cases}$

**8. Algorithm Pseudocode (If any, paste the pseudocode from the paper exactly as it is)**

The original paper does not provide a formal algorithm pseudocode block.