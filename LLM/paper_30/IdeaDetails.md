**1. Research Background and Existing Pain Points**

**Research Background:**
With the rapid progress of Large Language Models (LLMs), they are now applied across increasingly diverse domains and tasks. To meet versatile demands, LLMs are trained with distinct focuses, such as coding, mathematics, visual understanding, and edge computing. General-purpose LLMs can also simulate specialized capabilities through prompt engineering, enabling flexible role adaptation across downstream applications. Leveraging the diversity of LLMs, many multi-LLM systems are proposed to further enhance overall performance and efficiency. In collaborative multi-LLM systems, LLMs are assigned distinct roles and proactively exchange text messages. Mirroring human collaboration, these systems accumulate partial understandings or sub-solutions from different agents via verbal communication. By contrast, routing-based multi-LLM inference systems rely on passive context inheritance rather than active message exchange. These systems coordinate models of varying parameter sizes or reasoning depths for more dynamic and efficient responses. Downstream models inherit the context from preceding models in multi-round conversations, then generate follow-up responses to the new questions based on their own understanding of the conversation history.

**Existing Pain Points:**
Current text-to-text (T2T) interfaces restrict information exchange among LLMs, particularly when conveying rich or diverse semantic interpretations of a shared context. These limitations arise from several inherent constraints of T2T communication:
1. **Information Bottleneck:** As a low-bandwidth medium, text introduces an information bottleneck. The high-dimensional internal representations must be repeatedly compressed into linear strings and then decompressed by the receiver LLM. When models differ in knowledge or assigned roles, some signals may be irrecoverable (e.g., interpreting `<p>` as a section marker).
2. **Ambiguity of Natural Language:** Natural language is inherently ambiguous, with idioms, underspecified references, and vague expressions. Although recent agent protocols aim to standardize text messages, rigid templates remain insufficient for flexible, open-domain collaboration.
3. **High Latency:** T2T communication incurs noticeable latency. Every exchange requires exhaustive, token-by-token decoding of contextual explanations in sequence.

**2. Core Research Motivation and Scientific Questions**

**Core Research Motivation:**
Motivated by the limitations of text communication, this work explores using KV-Cache as the medium for LLM communication. KV-Cache is a naturally richer representation than text. It also enables fully parallel communication through direct projection, avoiding the slow sequential decoding in text exchanges. Oracle experiments show that: (1) enriching KV-Cache under the same context length increases accuracy, (2) KV-Cache is convertible between LLMs, and (3) different LLMs encode distinct semantic understandings and contextual knowledge of the same input, reflecting their complementary strengths.

**Scientific Questions:**
Can LLMs communicate beyond text? Specifically: (1) Benefit: Can a model’s capabilities be improved through KV-Cache semantic enrichment without extending sequence length? (2) Convertibility: Can the KV-Cache of one model be effectively utilized by another model?

**3. Overall Core Idea and Design Philosophy**

**Overall Core Idea:**
The paper proposes Cache-to-Cache (C2C), a new paradigm for richer and faster multi-LLM communication. C2C uses a neural network to project and fuse the source model’s KV-cache with that of the target model to enable direct semantic transfer. 

**Design Philosophy:**
C2C operates under a residual integration principle to enhance the Receiver’s KV-Cache without destructive overwriting of its information. It directly projects the KV-Cache from a source model (Sharer) into the space of a target model (Receiver) and merges them through a neural cache fuser. A learnable gating mechanism selects the target layers that benefit from cache communication. Compared with text communication, C2C utilizes the deep, specialized semantics from both models, while avoiding explicit intermediate text generation.

**4. Core Innovation Points**

1. **New Communication Paradigm:** Proposes Cache-to-Cache (C2C), a new paradigm for direct semantic communication between LLMs using KV-Cache, moving beyond the information bottleneck, ambiguity, and latency of the text-to-text (T2T) interface.
2. **Oracle Validation of KV-Cache Semantics:** Demonstrates through oracle experiments that enriching KV-Cache semantics improves response quality without increasing cache size, and that KV-Cache is convertible and carries complementary semantics across different LLMs.
3. **Cache Fuser with Residual Integration:** Designs a neural cache fuser comprising a projection module, a feature fusion layer, a dynamic weighting module, and a learnable gating mechanism (Gumbel-sigmoid with temperature annealing) to project and fuse heterogeneous KV-Caches while preserving the Receiver's original information via residual connection.
4. **Model Alignment Strategies:** Introduces token alignment (maximal-coverage selection for one-to-many tokenizer mappings) and layer alignment (terminal alignment strategy aligning final layers first in reverse order) to enable cross-family and cross-size KV-Cache sharing.
5. **Efficiency and Performance Gains:** Achieves significant accuracy improvements over individual models and T2T, while delivering an average 2.5× speedup in latency by replacing sequential text decoding with parallel cache fusion.

**5. Overview of the Overall Technical Solution**

In LLM communication scenarios, the LLM that provides contextual understanding or knowledge is defined as the Sharer, and the one that utilizes it is the Receiver. The C2C paradigm contains a set of key/value cache fusers $F$ and a layer mapping strategy $G$. 

During the prefill stage, both the Sharer and Receiver models encode the input context to produce their respective KV-Caches. The fuser $F_n$ takes the $n$th layer cache of the Receiver Model $C_n(X)$ and the corresponding $G(n)$th layer cache of the Sharer Model $C^S_{G(n)}(X)$ and generates the corresponding fused cache with a residual connection.

During the decoding stage, with the current token $y_i$ and caches from the input and the generated prefix, the Receiver model predicts the next token conditioned on the fused KV-Cache rather than its own original cache, enabling direct semantic transfer without explicit intermediate text generation.

**6. Detailed Module Design**

**6.1. Oracle Experiments for Cache-to-Cache Communication**

**6.1.1. Cache Enrichment Oracle**
To validate the benefit of cache enrichment, three setups are evaluated:
1. **Direct:** Prefill on $X$ only and decode with $C(X)$.
2. **Few-shot:** Prefill on $E \oplus X$ and decode with $C(E \oplus X)$ (longer cache).
3. **Oracle:** Prefill on $E \oplus X$ but discard the exemplar segment and keep only the question-aligned slice $C^*(X) = C_{[|E|:|E|+|X|]}(E \oplus X)$, so that decoding uses a question-length cache with no extra tokens. Comparing Direct and Oracle isolates the effect of cache enrichment. Substantial variation exists across layers: selectively applying cache enrichment to the top-performing layers yields slightly higher accuracy than enriching all layers, while targeting the worst-performing layers leads to accuracy decline.

**6.1.2. Cache Transformation Oracle**
To verify KV-Cache convertibility, a 3-layer MLP is trained to map the KV-Cache from a source LLM to a target LLM. T-SNE visualizations reveal that after transformation, the mapped KV-Cache lies inside the target model’s representation space, demonstrating convertibility.

**6.2. Cache Fuser Structure**
The fuser is designed under a residual integration principle and contains three key modules:
1. **Projection Module:** Concatenates the Receiver’s KV-Cache with the Sharer’s KV-Cache, then processes the concatenated features through a projection layer followed by a feature fusion layer.
2. **Dynamic Weighting Module:** Applies an input-aware head modulation layer to dynamically reweight the projected information.
3. **Learnable Gate:** Introduces a trainable per-layer gate value that decides whether to inject the Sharer’s context. The gate applies a Gumbel-sigmoid with temperature annealing to smoothly transition from differentiable during training to binary at inference.

**6.3. Model Alignment**
Fusing KV-Caches across model families and sizes requires alignment at two levels:
1. **Token Alignment:** Different tokenizers may produce slightly varied token sequences. Alignment is done by decoding each Receiver token into its string form and re-encoding it using the Sharer’s tokenizer. Template sections are aligned by simple length padding. For message sections, if re-encoding produces multiple source model tokens (one-to-many case), maximal-coverage selection is applied: decode each candidate token, compute its string length, and select the longest to preserve maximal surface correspondence.
2. **Layer Alignment:** Adopts a terminal alignment strategy: the final layers of both models are aligned first, then the penultimate layers, and so on in reverse order until reaching the shallower model’s first layer. This prioritizes alignment between the deeper layers, which typically capture higher-level semantic representations.

**6.4. Training Scheme**
During training, both the Sharer and Receiver models are frozen, training only the C2C module for KV-Cache fusion. Standard next-token prediction loss on the Receiver’s response predictions is employed, similar to supervised fine-tuning (SFT). The Receiver predicts responses conditioned on fused KV-Cache rather than its own. The training procedure consists of three stages:
1. **Forward:** Both models encode the input context to produce their respective KV-Caches.
2. **Fusion:** The C2C module fuses both KV-Caches and replaces the Receiver’s cache.
3. **Supervision:** The Receiver prefills the response using the fused cache, and gradients backpropagate through C2C to minimize prediction loss.

**7. All Mathematical Formulas and Symbol Definitions**

**LLM Inference Preliminaries:**
*   $X[0:n] = [x_0, \ldots, x_{n-1}]$: Input token sequence.
*   $C(X[0:n]) = [c_0, \ldots, c_{n-1}] \in \mathbb{R}^{n \times d}$: Per-token KV-Cache after prefill. $d$ denotes the KV dimensionality flattened from all layers into a single vector per token.
*   $\oplus$: Sequence-wise concatenation.
*   Decoding next token prediction:
    $$y_{i+1} = P(y_i | C(X) \oplus C(Y[0:i]))$$
*   Cache update during decoding: $C(Y[0:i+1]) = C(Y[0:i]) \oplus C(y_i)$

**Cache Enrichment Oracle:**
*   $E$: Exemplars.
*   $X$: Question.
*   $| \cdot |$: Sequence length.
*   Oracle question-aligned slice cache:
    $$C^*(X) = C_{[|E|:|E|+|X|]}(E \oplus X)$$

**C2C Design:**
*   $F$: Set of key/value cache fusers.
*   $G$: Layer mapping strategy.
*   $C_n(X)$: $n$th layer cache of the Receiver Model.
*   $C^S_{G(n)}(X)$: $G(n)$th layer cache of the Sharer Model.
*   Fused cache with residual connection:
    $$C^F = \{C_n(X) + F_n(C_n(X), C^S_{G(n)}(X))\}_{n=1}^N$$
*   Decoding with fused cache:
    $$y_{i+1} = P(y_i | C^F(X) \oplus C(Y[0:i]))$$

**Depth-Normalized Alignment (Alternative strategy explored):**
*   $L_{\min}$: Number of layers in the model with fewer layers (anchor).
*   $L_{\max}$: Number of layers in the other model.
*   Layer alignment mapping:
    $$j^\star = \arg\min_j \left| \frac{i}{L_{\min}-1} - \frac{j}{L_{\max}-1} \right|$$

**8. Algorithm Pseudocode**

The paper provides a detailed code snippet for the Cache Transformation Oracle loss calculation (MSE loss) in Appendix A.3.2:

```python
source_prefilled = source_model.forward(prefill_input_id)
target_prefilled = target_model.forward(prefill_intput_id)
source_k, source_v = source_prefilled.past_key_values[-1]
target_k, target_v = target_prefilled.past_key_values[-1]
project_k = k_projector(source_k)
project_v = v_projector(source_v)
mseloss(torch.dstack([project_k, project_v]),
        torch.dstack([target_k, target_v]))
```