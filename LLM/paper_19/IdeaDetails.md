## 1. Research Background and Existing Pain Points

**Research Background:**
Large Language Models (LLMs) have become ubiquitous across diverse applications, from conversational chatbots and personal assistants to specialized systems handling question answering, document summarization, and machine translation. Despite their impressive capabilities, LLMs suffer from hallucination issues and knowledge limitations, particularly when dealing with domain-specific or up-to-date information. Retrieval-Augmented Generation (RAG) has emerged as a powerful paradigm to bridge this gap. By incorporating external knowledge bases, such as Wikipedia or domain-specific corpora, RAG retrieves relevant contextual information to enrich user queries. This approach has demonstrated remarkable success in improving generation quality, while enabling general-purpose LLMs to tackle specialized domain problems without costly fine-tuning.

Autoregressive Transformers execute inference in two distinct phases. In the prefill phase, the model processes the entire input sequence, performing self-attention across all tokens and materializing per-layer KV caches. In the subsequent decode phase, tokens are generated step by step while attending to this cached state. By reusing the stored projections of preceding tokens, the KV cache eliminates redundant recomputation of the prefix and enables efficient autoregressive generation. RAG extends this pipeline by incorporating external evidence. A retriever encodes the user query, searches a corpus, and returns the top-k passages. The generator concatenates the query and retrieved passages, tokenizes the combined sequence, and applies the same prefill–decode process. 

**Existing Pain Points:**
1. **Substantial Computational Overhead from Extended Input Sequences:** The injection of retrieved text chunks substantially increases the length of input prompts, leading to higher computation and memory requirements during the LLM inference. While a raw user query typically contains fewer than 200 tokens, augmenting it with retrieved context can push the sequence length beyond 2,000 tokens, leading to more than a 10x increase in computational and memory overhead. This dramatic expansion significantly degrades Time-To-First-Token (TTFT) and system throughput, ultimately compromising user experience. As a result, prefill dominates serving latency, raising TTFT and reducing throughput under load. The fundamental bottleneck is thus the cost of full prefill and KV materialization over long contexts.

2. **Cross-Query Context Overlap and Redundant Processing:** Current RAG systems exhibit the inefficiency of redundant processing of frequently retrieved text chunks across multiple queries. Identical text chunks from the external knowledge base are repeatedly retrieved across multiple user queries, and a small fraction of text chunks dominate the retrieval requests. There are power-law distributions in text chunks popularity, where the most frequently accessed 10% of text chunks satisfy 80% of all questions under top-1 retrieval. This skewed access pattern indicates substantial redundant computation during LLM inference, as the same contextual information is processed repeatedly for different user queries.

3. **Over-Allocation of Context and Uniform Deep Retrieval:** Current RAG systems exhibit the inefficiency of uniform deep retrieval that over-provisions context regardless of query complexity. Although LLMs consistently benefit from expanded contextual information, accuracy improvements follow a pattern of diminishing marginal utility with additional retrieved text chunks. Over 60% of queries require only minimal context, whereas only approximately 3% need top-8 retrieval. This distribution highlights a critical inefficiency: static deep retrieval incurs unnecessary computational costs on simple queries while potentially degrading accuracy through contextual noise.

4. **Limitations of Current Caching Systems:** Caching represents a promising solution to address computational redundancy by reusing previously computed representations (KV cache). However, prefix caching (such as vLLM, SGLang, and RAGCache) requires exact sequence matching, leading to poor hit rates with longer contexts and positional variations. Independent chunk caching approaches attempt more flexible strategies; for instance, PromptCache achieves higher efficiency through independent chunk caching but sacrifices accuracy by ignoring cross-chunk attention. CacheBlend partially restores cross-chunk attention via selective recomputation, yet applies uniform recomputation ratios across all chunks without considering the heterogeneous attention characteristics across different chunks. Furthermore, all prior work assumes static top-k retrieval, missing query-adaptive optimization opportunities. General LLM inference systems (vLLM, Orca, DistServe, FlexGen) reduce phase contention and memory pressure but treat prompts as monolithic sequences, leaving them ill-suited to RAG’s retrieval-induced redundancy.

## 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:**
The key objective is to achieve the best of both worlds: harnessing RAG’s quality improvements while preserving computational efficiency. The research is motivated by the fundamental optimization challenge in the RAG system: How can we achieve both computational efficiency and performance gains simultaneously? It is motivated by the need to address the two major observed inefficiencies: the power-law distribution of context reuse across queries, where 10% of chunks satisfy 80% of retrieval requests, and the over-allocation of context, where 60% of queries require only minimal retrieval. The motivation is to eliminate cross-request computational redundancy in overlapping contexts while preventing unnecessary context augmentation for simple queries.

**Scientific Questions:**
How can we reuse previously computed representations (KV cache) flexibly without requiring exact sequence matching, while preserving cross-chunk dependencies and maintaining generation fidelity? How can we dynamically determine the optimal retrieval depth for each individual user query to avoid over-allocation of context and eliminate redundant computation for simple queries?

## 3. Overall Core Idea and Design Philosophy

**Overall Core Idea:**
Present AdaCache, an adaptive caching framework that addresses both computational redundancy and contextual over-provisioning in RAG systems through dual optimization strategies: cache-aware partial recomputation and adaptive context augmentation. 

**Design Philosophy:**
The design philosophy is based on profiling attention patterns to understand inter-chunk dependencies in RAG contexts. Since attention patterns reveal that only a subset of chunks serves as effective prefixes (exhibiting localized patterns in early layers and attention sink phenomena in deeper layers), the framework establishes a hierarchical cache that systematically balances cache utilization efficiency against generation quality. It leverages the observation that sparsity patterns exhibit layer-wise consistency, enabling efficient selective recomputation. Furthermore, it adopts a philosophy of confidence-guided, per-query adaptive expansion, using lightweight confidence estimation to incrementally expand prompts. This design reduces redundant computation, lowers long-context overhead, and significantly improves both TTFT and throughput while preserving generation quality.

## 4. Core Innovation Points

1. **Cache-aware Partial Recomputation Mechanism:** Profiles attention patterns to construct selective cache variants, enabling flexible reuse while preserving cross-chunk dependencies. It identifies attention-critical tokens through first-layer analysis and applies the same selection across all layers, adapting the recomputation ratio based on available cache matches.

2. **Adaptive Context Augmentation (ACA) Strategy:** Dynamically determines optimal retrieval depth via lightweight confidence estimation, avoiding unnecessary overhead on simple queries. It incrementally expands prompts by adding one text chunk at a time, evaluating confidence after each addition, and terminating when sufficient confidence is achieved.

3. **Hierarchical Cache Architecture:** Establishes a three-tier cache hierarchy (Hard Prefix Cache, Soft Prefix Cache, Independent Cache) that systematically balances cache utilization efficiency against generation quality based on inter-chunk attention patterns. It employs progressively relaxed matching constraints and adaptive token recomputation that responds to varying prefix match conditions (exact, partial, or absent) to achieve an optimal efficiency-accuracy trade-off.

4. **Composite Confidence Metric for Early Termination:** Introduces a composite confidence metric combining average KL divergence across the last few layers and output entropy to robustly evaluate internal reasoning consistency and predictive certainty for incremental context augmentation. This dual-metric balances stability with predictive certainty, providing a more robust confidence estimate.

5. **Strategic Caching within Incremental Augmentation:** During the incremental augmentation loop, only the KV states for the newly added chunk are recomputed, storing them in the hard prefix cache space for reuse in subsequent context augmentation. This ensures that all previously processed chunks maintain cache hits, dramatically reducing computational overhead without introducing additional retrieval overhead.

## 5. Overview of the Overall Technical Solution

AdaCache consists of two complementary modules. Cache-aware selective recomputation maintains three hierarchical cache spaces: (1) Hard prefix cache requires exact prefix matching and stores KV cache for all chunks along the matching path (solid boxes), enabling direct reuse without recomputation; (2) Soft prefix cache matches only effective prefix chunks requiring partial recomputation at ratio $\alpha$, where solid boxes represent cached entries while dashed boxes indicate prefix dependencies without storage; (3) Independent cache performs chunk-level matching with higher recomputation ratio $\beta$. The top attention diagram shows selective recomputation where a fraction of tokens (blue solid blocks) are recomputed while the remaining tokens reuse cached KV states. 

Adaptive context augmentation incrementally expands prompts by adding one text chunk at a time, evaluating confidence after each addition using a composite metric combining average KL divergence across the last few layers and output entropy, terminating when sufficient confidence is achieved or maximum context is reached.

## 6. Detailed Module Design

### Module 1: Cache-aware Selective Recomputation

**Attention Analysis:**
Segment the augmented prompt into discrete chunks: [system prompt, text chunk 1, ..., text chunk k, query]. Aggregate attention weights of each layer into chunk granularity. Early layers (e.g., 1-18) show localized patterns where each chunk primarily attends to its predecessor, while deeper layers (e.g., 19-36) exhibit attention sink phenomena, with certain chunks capturing most attention from subsequent chunks. This pattern reveals that only a subset of chunks serves as effective prefixes, enabling joint caching of partial prefix sequences to restore cross-chunk dependencies lost in independent chunk caching.

**Hierarchical Cache:**
Based on the observed attention patterns, a three-tier cache hierarchy systematically balances cache utilization efficiency against generation quality:
- **Hard Prefix Cache:** Requires exact prefix sequence matching, making it the most restrictive but accuracy-preserving tier. Due to the causal attention mask in autoregressive inference, exact prefix matches guarantee computational equivalence to full recomputation, thereby preserving perfect generation quality when cache hits occur. However, this strict requirement significantly constrains cache utilization.
- **Soft Prefix Cache:** Relaxes the matching constraint to effective prefix matching, where only the sink chunk or predecessor chunk needs to match for cache reuse. This design leverages the attention analysis findings: since attention primarily flows from these key chunks, partial prefix matching can maintain most cross-chunk dependencies. Successful soft matching requires recomputing $\alpha$ fraction of tokens to restore global cross-chunk attention. Note that soft prefix chunks serve only as cache key identifiers without maintaining separate cache entries.
- **Independent Cache:** Provides the fallback mechanism when prefix matching fails entirely. Individual text chunks are precomputed and stored independently without prefix dependencies. It maximizes cache hit rates but poses the greatest risk for accuracy preservation, as cross-chunk attention dependencies must be reconstructed during LLM inference.

**Cache Reusing and Recomputation:**
Only a subset of tokens within each chunk exhibit significant cross-chunk attention, leading to substantial KV states deviations compared to those in the independent cache. Critically, this sparsity pattern exhibits layer-wise consistency: tokens with the highest KV deviations in one layer are likely to have the highest deviations in subsequent layers. This insight enables efficient selective recomputation by identifying attention-critical tokens through first-layer analysis and applying the same selection across all layers. Recomputation candidates are determined by analyzing cross-chunk attention ratios in the initial layer, selecting tokens with the highest proportion of cross-chunk attention weights.

Rather than a uniform recomputation across all chunks in context, the recomputation ratio is adapted based on available cache matches. The KV states of each chunk may have multiple cached variants stored under different prefix contexts. Cache spaces are retrieved in a hierarchical order with progressively relaxed matching constraints. First, query the hard prefix cache for exact matches, where any chunks along the matched prefix path can be directly reused without recomputation. When exact matching fails, examine the soft prefix cache for effective prefix alignment. In the first half of model layers, effective prefixes correspond to predecessor chunks, while in the second half, they correspond to both sink chunks and predecessor chunks. Sink chunk positions are identified by analyzing attention matrices at transition layers: chunks before the sink chunk require predecessor-based matching, while chunks after the sink chunk use both sink chunks and predecessor chunks as their effective prefixes. Including predecessor chunks prevents cumulative errors that would arise from inconsistency with the predecessor-based matching used in the first half of layers. If no cached KV states exist in the soft prefix cache space, turn to the independent cache with recomputation ratio $\beta$ ($\beta > \alpha$) to reconstruct discarded cross-chunk dependencies. This cache reuse approach achieves an optimal efficiency-accuracy trade-off through adaptive token recomputation that responds to varying prefix match conditions: exact, partial, or absent.

### Module 2: Adaptive Context Augmentation

**Incremental Augmentation Strategy:**
Rather than concatenating all top-k retrieved text chunks into the user prompt simultaneously, an incremental augmentation strategy progressively incorporates one chunk at a time until reaching the k-th chunk or achieving sufficient confidence. While this approach necessitates multiple forward passes for the same query, it eliminates redundant context computation through strategic caching. At each iteration, only recompute the KV states for the newly added chunk, storing them in the hard prefix cache space for reuse in subsequent context augmentation. This ensures that all previously processed chunks maintain cache hits, dramatically reducing computational overhead. Importantly, ACA does not introduce any additional retrieval overhead. The top-k retrieval step is executed once per query, and the augmentation loop then operates solely within the prefill phase using the already-retrieved text chunks.

**Confidence Evaluation:**
To decide whether augmentation should terminate, employ a composite confidence metric combining two complementary uncertainty measures. First, compute the average KL divergence between the logits of the last $l$ layers and the final layer, capturing internal reasoning consistency. If the model can accurately infer the answer from the current context, its logit distribution should converge early across layers. Second, calculate the entropy of the final token distribution, reflecting output uncertainty. Normalize both the average KL divergence and entropy to [0, 1], then compute a weighted confidence score, with weights determined through optimization on the validation set. This dual-metric balances stability with predictive certainty, providing a more robust confidence estimate. Notably, the confidence metric is computationally lightweight. In practice, AdaCache computes logits only for the last 4 layers and for the final token rather than the full context, which keeps the overhead negligible at less than 1% of the prefill cost.

**Computational Savings:**
ACA reduces computational and memory demands by avoiding excessive context allocation for simple queries. Given k retrieved text chunks of length $l_c$ tokens each, a query of length $l_q$ tokens, and early termination at step t, ACA processes at most $t \cdot (l_c + l_q)$ tokens. It yields substantial savings in computation and memory compared to static context augmentation, which requires processing $k \cdot l_c + l_q$ tokens.

## 7. All Mathematical Formulas and Symbol Definitions

- **Entropy:**
  $$H = -\sum p_i \log p_i$$

- **Average KL Divergence:**
  $$KL = 1/k \sum D_{KL}(L_i || L_j)$$

- **Confidence Score:**
  $$conf \leftarrow \lambda \cdot \hat{KL}(O_{1..l-1}, O_l) + (1-\lambda) \cdot \hat{H}(O_l)$$

- **Recomputation Ratio for Soft Prefix Cache:**
  $$\rho \leftarrow \alpha$$

- **Recomputation Ratio for Independent Cache:**
  $$\rho \leftarrow \beta$$

- **Constraint Condition for Recomputation Ratios:**
  $$\beta > \alpha$$

- **Maximum Tokens Processed by ACA:**
  $$t \cdot (l_c + l_q)$$

- **Tokens Processed by Static Context Augmentation:**
  $$k \cdot l_c + l_q$$

**Symbol Definitions:**
- $k$: Number of retrieved text chunks.
- $l_c$: Length of each retrieved text chunk in tokens.
- $l_q$: Length of the query in tokens.
- $t$: Early termination step in adaptive context augmentation.
- $l$: Number of last layers used for confidence evaluation.
- $\lambda$: Weight parameter for the composite confidence score, determined through optimization on the validation set.
- $\alpha$: Lower recomputation ratio for Soft Prefix Cache.
- $\beta$: Higher recomputation ratio for Independent Cache.
- $\tau$: Confidence threshold for terminating context augmentation.
- $O_{1..l-1}$: Logits of the last $l-1$ layers.
- $O_l$: Logits of the final layer.
- $\hat{KL}$: Normalized average KL divergence.
- $\hat{H}$: Normalized entropy.

## 8. Algorithm Pseudocode

**Algorithm 1 Adaptive Context Augmentation**

**Require:** System prompt s, query q, retrieved text chunks [c1, . . . , ck], confidence threshold $\tau$
1: for i = 1 to k do
2: &nbsp;&nbsp;&nbsp;&nbsp;p ← s + [c1, . . . , ci] + q &nbsp;&nbsp;&nbsp;&nbsp;▷ Construct context-augmented prompt
3: &nbsp;&nbsp;&nbsp;&nbsp;for j = 1 to i−1 do
4: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Retrieve hash([c1, . . . , cj ]) in Hard Prefix Cache &nbsp;&nbsp;&nbsp;&nbsp;▷ Reuse KV states w/o recomputation
5: &nbsp;&nbsp;&nbsp;&nbsp;end for
6: &nbsp;&nbsp;&nbsp;&nbsp;if hash(effective prefix([c1, . . . , ci])) $\in$ Soft Prefix Cache then
7: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\rho \leftarrow \alpha$ &nbsp;&nbsp;&nbsp;&nbsp;▷ Lower recomputation ratio for Soft Prefix Cache
8: &nbsp;&nbsp;&nbsp;&nbsp;else
9: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\rho \leftarrow \beta$ &nbsp;&nbsp;&nbsp;&nbsp;▷ Higher recomputation ratio for Independent Cache
10: &nbsp;&nbsp;&nbsp;&nbsp;end if
11: &nbsp;&nbsp;&nbsp;&nbsp;T ← $\rho$ fraction of tokens in ci &nbsp;&nbsp;&nbsp;&nbsp;▷ Selected tokens with the highest cross-chunk attention
12: &nbsp;&nbsp;&nbsp;&nbsp;O ← Prefill(T) &nbsp;&nbsp;&nbsp;&nbsp;▷ Generate logits for last l layers
13: &nbsp;&nbsp;&nbsp;&nbsp;conf ← $\lambda \cdot \hat{KL}(O_{1..l-1}, O_l) + (1-\lambda) \cdot \hat{H}(O_l)$ &nbsp;&nbsp;&nbsp;&nbsp;▷ Evaluate the confidence
14: &nbsp;&nbsp;&nbsp;&nbsp;if conf > $\tau$ then
15: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Execute decoding &nbsp;&nbsp;&nbsp;&nbsp;▷ Stop context augmentation
16: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;break
17: &nbsp;&nbsp;&nbsp;&nbsp;end if
18: &nbsp;&nbsp;&nbsp;&nbsp;Update KV states of ci with hash([c1, . . . , ci]) in Hard Prefix Cache
19: end for