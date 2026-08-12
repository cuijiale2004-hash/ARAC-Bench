# 1. Research Background and Existing Pain Points

**Research Background:** 
Large Language Models (LLMs) have made significant progress in many real-world applications, including chatbots, search engines, and coding assistants. While fine-tuning LLMs has been demonstrated to be a promising way to improve performance on downstream tasks, it incurs substantial computational and storage costs. To alleviate this, Parameter-Efficient Fine-Tuning (PEFT) methods such as LoRA update only a small subset of parameters while keeping the base model frozen. However, a fundamental limitation is that PEFT’s adapted parameters are dependent on the base model and thus cannot be transferred across different models. This limitation is increasingly critical given the rapid release of new LLMs and the growing diversity of available models.

**Existing Pain Points:**
1. **Dependency of LoRA on Base Models:** LoRA adapters are tied to the frozen backbone they were trained on, making them difficult to transfer to other base models.
2. **Data Dependency of Knowledge Distillation (KD):** Knowledge distillation transfers knowledge by aligning target output distributions with those of the source. However, KD is inherently data-dependent, typically requiring access to training data from target downstream tasks, which is often unavailable or costly to obtain.
3. **Complexity and Overhead of TransLoRA:** TransLoRA attempts to transplant the knowledge of LoRA adapters across models by generating synthetic data. While effective to some extent, this approach requires training an additional discriminator model to filter high-quality synthetic data, resulting in a relatively heavy pipeline with extra complexity and computational overhead.
4. **Neglect of Transfer Process Design:** TransLoRA primarily emphasizes the role of synthetic data, paying less attention to how the knowledge transfer process itself should be designed. It focuses on transferring full outputs and relies entirely on synthetic data, whereas the underlying philosophy of selective knowledge extraction at a fine-grained level is overlooked.

# 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:**
The core motivation is to selectively convey task-relevant information from the source model’s LoRA by using token-level signals to guide the transfer process, rather than relying on the entire token sequence. This approach aims to provide a fine-grained yet lightweight alternative that enables efficient transplantation of LoRA adapters across diverse models for deployment, without requiring access to the original training corpus or training additional discriminator models.

**Scientific Questions:**
1. How can we identify and selectively transfer task-relevant knowledge embedded in a source model's LoRA adapter without relying on the entire token sequence?
2. How can we filter synthetic data effectively for training a target LoRA adapter without introducing the extra burden of training and maintaining a separate discriminator model?
3. How can we capture the knowledge discrepancy introduced by the LoRA adapter at the token level?
4. How can we align token-level signals and masks when the source and target models use different tokenizers, ensuring reliable cross-tokenizer transfer?

# 3. Overall Core Idea and Design Philosophy

**Overall Core Idea:**
The paper proposes TITOK, a new framework that enables effective LoRA Transplantation through Token-level knowledge transfer. Specifically, TITOK captures task-relevant information through a token-wise contrastive excess between a source model with and without LoRA. This excess highlights informative tokens and enables selective filtering of synthetic data, all without additional models or overhead.

**Design Philosophy:**
The design philosophy is rooted in the observation that not all tokens contribute equally to model training and knowledge transfer. The contrast between the source backbone (amateur) and its LoRA adapter (expert) yields token-level judgments that identify where task knowledge is injected. Tokens where the backbone model is uncertain but the LoRA-enhanced model assigns high confidence contain concentrated task-specific knowledge. By focusing training only on these prioritized, highly informative tokens via a token-wise contrastive excess score, the framework achieves a more focused, fine-grained, and efficient knowledge transfer. Furthermore, by using the internal behavior of the model itself (log-likelihood ratio) as an attribution signal, the need for external discriminator models is eliminated.

# 4. Core Innovation Points

1. **Novel Framework for LoRA Transplantation:** Proposal of TITOK, a framework that enables LoRA Transplantation through Token-level knowledge transfer, focusing on token-level selective transfer rather than indiscriminately relying on the entire token sequence or full outputs.
2. **Token-wise Contrastive Excess Score:** Introduction of a lightweight metric that utilizes the source model and its LoRA adapter to perform two complementary roles (amateur and expert) to compute a token-level excess score. This captures the knowledge discrepancy introduced by the LoRA adapter and identifies informative tokens without requiring extra models or additional training overhead.
3. **Two-level Filtering Mechanism:** Design of a two-stage training objective that first performs sample filtering based on the mean excess score to remove uninformative examples, and subsequently performs token selection within samples based on the excess score ranking to train the target model only on the most informative tokens.
4. **Excess Score Alignment Algorithm:** Design of a simple yet robust tokenizer alignment algorithm that propagates token masks from the source token sequence to the target token sequence using dual pointers and specific mapping rules, addressing the tokenizer mismatch problem between source and target models.
5. **Source Expert-driven Synthetic Data Generation:** Utilizing the source expert model (source backbone + LoRA) to synthesize both queries and labels, unlike TransLoRA which uses the untuned target model for queries, thereby providing more coherent supervision for effective transfer.

# 5. Overview of the Overall Technical Solution

The overall technical solution of TITOK consists of three main components and an alignment mechanism:
1. **Synthetic Data Generation:** Starting from a small set of seed prompts, the source expert model (source base model + LoRA) generates a synthetic dataset consisting of query-label pairs.
2. **Excess Score Computation:** A token-wise contrastive excess filtering mechanism compares the expert (source backbone + LoRA) against its base backbone (source backbone only) to compute token-level excess scores. These scores quantify the knowledge discrepancy introduced by the LoRA adapter.
3. **Target Model Training with Filtering:** Using the computed excess scores, TITOK first performs sample filtering by retaining samples with the highest mean excess scores. It then performs token selection within these samples, retaining only the top k% of tokens ranked by their excess scores. The target model's LoRA adapter is trained using a masked loss applied only to these selected tokens.
4. **Excess Score Alignment:** In cases where the source and target models have different tokenizers, a dual-pointer alignment algorithm is applied to propagate the token masks from the source to the target sequence, ensuring consistent supervision across tokenizers before the top-k% token selection and training steps.

# 6. Detailed Module Design

**Module 1: Synthetic Data Generation via LLM Prompting with Few-Shot Data**
Let $M_s$ denote the source backbone LLM and $A_s$ its LoRA adapter on the target downstream task, which forms the source expert model $M_s + A_s$. The target model is denoted by $M_t$. TITOK constructs a synthetic dataset $D_s$. The source expert model $M_s + A_s$ synthesizes both the query $q$ and the label $y$ within a prompting-based data synthesis framework. Given a few-shot data of the downstream task, it first generates a query $q$, and then produces the corresponding label $y$ conditioned on $q$. To encourage diversity, ROUGE-L filtering together with deduplication is applied to all tasks, except for exceptional cases where such filtering is infeasible. The resulting synthetic dataset consists of query–label pairs.

**Module 2: Token-wise Contrastive Excess Score from Source Model with LoRA**
To filter imperfect synthetic data without an additional discriminator, the framework uses the source model and its LoRA adapter to perform two complementary roles: (1) the amateur role ($M_s$) and (2) the expert role ($M_s + A_s$). The difference between the two roles provides an implicit supervision signal where the task information is encoded. The excess score quantifies the knowledge discrepancy introduced by the LoRA adapter, identifying tokens where the adapter provides a decisive contribution. Intuitively, if the backbone model is uncertain about predicting a token but the LoRA-enhanced model assigns it with high confidence, that token obtains a large excess score, implying it contains task-specific knowledge.
Theoretically, the token-wise contrastive excess score is connected to a token-level log-likelihood ratio (LLR) between the source expert and the backbone. By the Neyman–Pearson lemma, the likelihood ratio is the optimal statistic for identifying differences between two models. High-LLR tokens are exactly the regions where the adapter changes the predictive distribution (where task knowledge is injected). From the perspective of standard knowledge distillation, high-LLR tokens correspond to maximal teacher–student divergence, larger gradients, and higher information contribution to the student.

**Module 3: Target Model Training with Sample Filtering and Token Selection**
This module involves two stages:
*   **First stage: Sample filtering.** The synthetic dataset $D_s$ is filtered at the sample level to remove less informative examples. For each synthetic sample, the mean of the excess scores across the tokens in the label is computed, and only $M$ samples with the highest values are retained.
*   **Second stage: Token selection.** The newly initialized LoRA adapter $A_t$ for target model $M_t$ does not learn from all tokens within the retained samples. Instead, it focuses only on those prioritized by the excess scores. It selects the top $k\%$ of tokens ranked by their excess scores using an indicator function. The training objective for $A_t$ is defined using a masked loss that only considers the selected tokens.

**Module 4: Excess Score Alignment Across Different Tokenizers**
In cases of transfer between models with different tokenizers, a direct mapping of token-level signals is not possible. To address this, a dual-pointer alignment procedure is implemented. The algorithm maintains two pointers, one for the source tokens and one for the target tokens. At each step, the source pointer advances by one token, accumulating a decoded segment, while the target pointer incrementally extends its own segment until the normalized texts match. Once a match is found, the corresponding spans are recorded as an alignment. After alignment, masking rules are applied to propagate the source binary mask scores to the target tokens:
1.  **One-to-One:** binary mask score is directly copied.
2.  **One-to-Many:** score is replicated across all aligned target tokens.
3.  **Many-to-One:** averaged scores of multiple source tokens are assigned to the target token.
4.  **Many-to-Many:** the averaged score of the source tokens is assigned to the corresponding aligned target tokens.
This process yields fractional mask scores that capture the relative importance of each target token. Finally, a top-k% selection step retains the most confident target tokens, producing a final binary selection mask.

# 7. All Mathematical Formulas and Symbol Definitions

**Symbol Definitions:**
*   $M_s$: Source backbone LLM.
*   $A_s$: LoRA adapter on target downstream task for source model.
*   $M_s + A_s$: Source expert model.
*   $M_t$: Target backbone LLM.
*   $A_t$: LoRA adapter for target model to be trained.
*   $D_s$: Synthetic dataset.
*   $q$: Synthesized query.
*   $y$: Synthesized response / label, where $y = [y_1, ..., y_L]$.
*   $y_i$: A token of $y$ corresponding to the synthesized query $q$.
*   $S(y_i)$: Token-wise contrastive excess score.
*   $L_a(y_i)$: Amateur loss on token $y_i$.
*   $L_e(y_i)$: Expert loss on token $y_i$.
*   $\bar{S}_j$: Mean excess score for sample $j$.
*   $D_f$: Filtered synthetic dataset containing top $M$ samples.
*   $I_{k\%}(y_i)$: Indicator for token selection (1 if in top $k\%$, 0 otherwise).
*   $L_{TiTok}$: Training objective for target LoRA adapter $A_t$.
*   $L_t(y_i)$: Negative log-likelihood loss assigned by $M_t + A_t$ on token $y_i$.

**Mathematical Formulas:**
1.  Synthetic dataset construction:
    $$D_s = \{(q_j, y_j)\}_{j=1}^N$$

2.  Token-wise contrastive excess score:
    $$S(y_i) = L_e(y_i) - L_a(y_i)$$

3.  Amateur and expert losses on token $y_i$:
    $$L_a(y_i) = \log P_{M_s}(y_i | q, y_{<i}), \quad L_e(y_i) = \log P_{M_s+A_s}(y_i | q, y_{<i})$$

4.  Mean excess score for sample $j$:
    $$\bar{S}_j = \frac{1}{|y_j|} \sum_{y_i \in y_j} S(y_i)$$

5.  Filtered dataset selection (Top $M$ samples by $\bar{S}_j$):
    $$D_f = Top_M\{(q_j, y_j) \in D_s : \bar{S}_j\}$$

6.  Token selection indicator (Top $k\%$ of tokens):
    $$I_{k\%}(y_i) = \begin{cases} 1, & \text{if } rank_{y_j}(S(y_i)) \le \lfloor k\% \cdot |y_j| \rfloor, \\ 0, & \text{otherwise} \end{cases}$$
    where $|y_j|$ denotes the number of tokens in $y_j$, and $rank_{y_j}(S(y_i))$ indicates the rank of $S(y_i)$ among the tokens of that response.

7.  Training objective for target LoRA adapter:
    $$L_{TiTok} = \sum_{(q_j, y_j) \in D_f} \sum_{y_i \in y_j} I_{k\%}(y_i) \cdot L_t(y_i)$$

# 8. Algorithm Pseudocode

**Algorithm 1: TITOK: Transplanting LoRA through Token-Level Knowledge**
Input: source expert $M_s + A_s$, target $M_t$, parameters $N, M, k\%$
Output: trained target LoRA $A_t$

1. Construct synthetic dataset $D_s = \{(q_j, y_j)\}_{j=1}^N$ with $M_s + A_s$
2. for $(q_j, y_j) \in D_s$ do
       Compute token excess scores $S(y_i) = L_e(y_i) - L_a(y_i)$
       Calculate mean score $\bar{S}_j = \frac{1}{|y_j|} \sum_{y_i \in y_j} S(y_i)$
3. Select top-$M$ samples by $\bar{S}_j$ to form $D_f$
4. for $(q_j, y_j) \in D_f$ do
       Rank tokens by $S(y_i)$ and keep top-$k\%$, represented by mask $I_{k\%}(y_i)$
5. if $tokenizer(M_s) \neq tokenizer(M_t)$ then
       Align masks $I^{(s)}_{k\%}(y_i) \rightarrow I^{(t)}_{k\%}(y_i)$
6. Train $A_t$ on $M_t$ with masked loss $L_{TiTok} = \sum_{(q_j, y_j) \in D_f} \sum_{y_i \in y_j} I_{k\%}(y_i) \cdot L_t(y_i)$
return $A_t$