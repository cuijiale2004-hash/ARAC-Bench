1. **Research Background and Existing Pain Points**

Despite the remarkable progress of Large Language Models (LLMs), they continue to exhibit factual errors, and their knowledge remains limited to the static dataset on which they were trained. To address these limitations, Retrieval-Augmented Generation (RAG) has been proposed, enabling models to ground their outputs in external, up-to-date information, thereby enhancing both accuracy and relevance in knowledge-intensive tasks. In practice, RAG retrieves a small set of the most relevant passages from an external corpus to ground the LLM’s generation process. Recent advancements have substantially increased the context length of LLMs, with some models now supporting inputs of up to 128K tokens or more (e.g., GPT-4o, Gemini 2.5, Qwen 2.5). This capability offers an alternative strategy for grounding model outputs: supplying the full document context directly to the model, rather than relying on RAG to supply only a subset of them. With sufficient capacity, LLMs can selectively attend to salient information while disregarding irrelevant content, thereby reducing reliance on explicit retrieval mechanisms. Indeed, empirical evidence from emerging benchmarks indicates that providing LLMs with full long-contexts frequently outperforms RAG-based approaches.

Nevertheless, this emerging alternative strategy has notable limitations and pain points: (i) it is token-inefficient to handle large and potentially redundant contexts; (ii) it exacerbates the ‘lost in the middle’ phenomenon, where the model struggles to recall information presented in the middle of a long input sequence; and (iii) under limited model capacity, it amplifies distraction, ultimately degrading LLM output quality.

Furthermore, both open and closed-source LLMs can still fail to answer questions even when the gold passage is retrieved, due to interference from additionally retrieved passages (i.e., distracting passages). The columns in Figure 2 illustrate retrieval strategies commonly adopted in practice and their associated pain points: (1) gold passages that lead to correct answer when individually retrieved; (2) retrieving all passages cumulatively from top to bottom, which corresponds to the conventional top-k similarity-based retrieval approach. Top-k retrieval is susceptible to distracting passages. Even when individually correct passages are included, their joint presence with distracting passages can result in incorrect answers. Note that retrieving all available passages is effectively equivalent to the long-context approach setting, which is likewise prone to errors. (3) retrieving all gold passages first followed by retrieving additional passages, which corresponds to reranking the retrieved top-k passages to prioritize relevance. Reranking approaches that place highly relevant passages at the front of the retrieved passages still fail when distracting passages are present, as their inclusion can override or obscure the signal from the relevant ones. (4) retrieving gold passages cumulatively from top to bottom, which resembles the successful outcome of the hybrid strategy: first selecting the top-k passages by similarity and then applying a relevance-based top-n selection to retain only the gold passages. Even when the hybrid strategy, which combines similarity-based top-k retrieval with relevance-based top-n selection, successfully retains only the gold passages, their joint retrieval can still lead to incorrect answers. Counterintuitively, even when all passages individually lead to the correct answer, their collective inclusion can complicate the reasoning process and ultimately cause the model to generate incorrect outputs. Based on these observations, distracting passages are defined as passages that misguide the LLM in generating the correct answer, irrespective of whether they lead to correct answer when individually retrieved. Moreover, the differing outcomes demonstrate that retrieval effectiveness is strongly tied to the capacity of the underlying LLM.

2. **Core Research Motivation and Scientific Questions**

These findings underscore the inherent difficulty of reliably retrieving passages that yield correct answers, motivating the need for retrieval strategies that explicitly account for distracting effects in relation to model capacity. The core research motivation is the need for approaches that integrate the advantages of both paradigms—approaching the performance of long-context approaches while maintaining the token efficiency of RAG.

The scientific questions addressed are:
1. How to retrieve passages to minimize distraction, given that the optimal strategy depends not only on the capacity of the target LLM, but also on the combinatorial interactions among the retrieved passages?
2. How to avoid the prohibitively expensive cost of employing a high-capacity LLM to exhaustively read and align every passage to the query, which scales with the number of passages and is infeasible in practice, especially in the long-context setting?
3. How to learn a distraction-aware retrieval strategy that balances information coverage and distraction in accordance with the LLM's long-context capability, relying solely on the similarity distribution without accessing textual information?

3. **Overall Core Idea and Design Philosophy**

We introduce LDAR (Learning Distraction-Aware Retrieval), an adaptive retriever that learns to select passages to minimize potential interference from distracting passages in accordance with the capacity of the LLM, thereby achieving significantly better performance and lower token usage compared to long-context approaches.

Design Philosophy 1: Lightweight Adaptive Retriever. We deliberately restrict πθ from accessing textual information to ensure our approach scales to large-scale retrieval scenarios. Moreover, our approach avoids the expense of fine-tuning the large pretrained LLM and embedding model by training only a lightweight neural network to reduce distraction, while keeping the larger components fixed.

Design Philosophy 2: Band-based Retrieval Strategy. Our retriever πθ operates on the cosine similarity distribution and selects a dynamic set of passages from a contiguous quantile interval qL, qU ⊂ [0, 1]. When the retriever πθ determines that information coverage should be prioritized at the risk of increased distraction, it retrieves passages from a wide quantile interval. On the contrary, if the risk of having distraction is higher, πθ retrieves passages from a narrow quantile interval, minimizing the risk of having distraction.

Design Philosophy 3: Attention Pooling and Transformer Encoding. Since the number of passages associated with each query may differ, the dimensionality of s ∈ RN is not fixed and varies across queries. To accommodate this variability, we employ a bidirectional self-attention Transformer that maps the token embedding of each similarity score si to a contextualized representation. An attention-pooling layer then aggregates these token-level representations into a global summary vector.

Design Philosophy 4: Abstraction via Band Retrieval. Intuitively, allowing the adaptive retriever πθ to select passages via independent Bernoulli sampling for each candidate is also a valid strategy. However, the Bernoulli-based variant of LDAR fails to identify a balanced trade-off between the RAG and long-context approach. This limitation arises from its need to explore the entire combinatorial subset selection space, which impedes generalization and ultimately causes convergence to a local optimum (corresponding to the long-context approach in this case). In contrast, band-based retrieval reduces the effective search space from combinatorial subset selection to a low-dimensional and smooth control space, yielding a temporally abstract retrieval strategy that enables more sample-efficient credit assignment and promotes better exploration.

4. **Core Innovation Points**

1. Unlike previous heuristic-based methods, we propose a learning-based retrieval strategy framework that adaptively balances information coverage and distraction in accordance with the capacity of the LLM, achieving better performance with significantly reduced token usage compared to the long-context approach.
2. We empirically demonstrate that retrieving passages in bands (i.e., selecting from contiguous ranges along the similarity-ranked list) is critical for learning a distraction-aware retrieval strategy. The banded retrieval strategy provides a form of abstraction that improves generalization and prevents the retriever from converging to suboptimal solutions.
3. The LDAR framework adaptively retrieves fewer passages for open-source models compared to closed-source models on the same task, indicating that our framework aligns retrieval strategies with the long-context capability of the underlying LLM.
4. We validate our approach across diverse LLM architectures (both open and closed-source) and six knowledge-intensive benchmarks, demonstrating both the effectiveness and robustness of the proposed retrieval strategy.
5. We demonstrate that including a penalty term proportional to the number of retrieved tokens during training often pushes the model away from the optimal region because the objective is not a linear trade-off but rather identifying the peak of an inverted-U relationship. Thus, focusing on letting LDAR learn to retrieve passages near the performance peak without an explicit cost term naturally avoids the high-distraction region.

5. **Overview of the Overall Technical Solution**

RAG employs a pretrained embedding model fϕ that maps a query q and passages {pi}Ni=1 into a shared vector space Rd, retrieving the top-k passages ranked by semantic similarity. Let si denote the similarity (e.g., dot product or cosine similarity) between the query and the i-th passage. Retrieved passages R with higher similarity scores are semantically closer to the query, as the embedding space is trained via contrastive objectives.

Our lightweight retriever πθ learns to select passages to minimize potential interference from distracting passages, solely based on the similarity between the query and passages. The adaptive retriever πθ is provided with a cosine similarity vector s ∈ RN between the query and the N passages, computed by a pretrained embedding model fϕ. Since the number of passages associated with each query may differ, the dimensionality of s ∈ RN is not fixed and varies across queries.

We employ periodic embedding layer for encoding numeric features, which projects batch of raw scalar inputs s ∈ RB×N×1 to the embedding dimension RB×N×d, followed by layer normalization. The embedded tokens are then processed by a self-attention Transformer model and outputs h ∈ RB×N×D. A learned linear token scorer projects each Transformer output to a scalar, which is normalized with a softmax to obtain attention weights w ∈ RB×N. We then form a global summary by attention pooling over the token dimension N: zb,d = ∑Nn=1 wb,nhb,n,d, and pass z ∈ RB×D through a small MLP head to obtain g.

From g, four linear heads produce the parameters of two Beta distributions: (αL, βL) for the lower quantile and (α∆, β∆) for the band width q∆ to ensure that lower quantile qL to be smaller than upper quantile qU. In addition, we apply softplus to ensure all parameters of Beta distributions are strictly positive. The lower and upper quantiles qL and qU are then sampled from these distributions respectively, and the resulting band {qL, qU} ⊂ [0, 1] is used to select the passages from the similarity distribution.

The main goal of πθ is to retrieve a set of passages that maximizes the likelihood of the pretrained LLM producing the correct answer to a given query. To this end, we formulate the objective as maximizing the prediction accuracy of the LLM conditioned on the passage set retrieved by πθ. By applying the likelihood ratio gradient with log-derivative trick, we can update θ at k-th gradient update step.

We train LDAR on 4 NVIDIA RTX 3090 GPUs for 32K context length settings and 1 NVIDIA RTX PRO 6000 GPU for 128K context length settings. The hyperparameter settings used for LDAR are: Batch Size 32; Embedding Dimension 256; Transformer Hidden Dimension 256; Baseline EMA coefficient 0.5; # Transformer Layer 2; # Transformer Head 4; Optimizer Adam(β = [0.9, 0.999], ϵ = 1e-8); learning rate γ 3e-4.

6. **Detailed Module Design**

- **Module 1: Similarity Computation.** Pretrained embedding model fϕ computes cosine similarity. Equation (2): si := \frac{f_\phi(q)^\top f_\phi(p_i)}{\|f_\phi(q)\| \|f_\phi(p_i)\|}, i = 1, ..., N.

- **Module 2: Periodic Embedding Layer.** We employ periodic embedding layer for encoding numeric features, which projects batch of raw scalar inputs s ∈ RB×N×1 to the embedding dimension RB×N×d, followed by layer normalization.

- **Module 3: Transformer Encoder.** Bidirectional self-attention Transformer processes the token embedding of each similarity score si to a contextualized representation h ∈ RB×N×D.

- **Module 4: Attention Pooling.** A learned linear token scorer projects each Transformer output to a scalar, which is normalized with a softmax to obtain attention weights w ∈ RB×N. We then form a global summary by attention pooling over the token dimension N: zb,d = ∑Nn=1 wb,nhb,n,d.

- **Module 5: Output Heads and Sampling.** Pass z ∈ RB×D through a small MLP head to obtain g. From g, four linear heads produce the parameters of two Beta distributions: (αL, βL) for the lower quantile and (α∆, β∆) for the band width q∆. We apply softplus to ensure all parameters of Beta distributions are strictly positive. The lower and upper quantiles qL and qU are then sampled from these distributions respectively.

- **Module 6: Passage Selection.** The resulting band {qL, qU} ⊂ [0, 1] is used to select the passages from the similarity distribution. ℓ← max(1, ⌊N · qL⌉), u← max(ℓ, ⌊N · qU⌉). σ ← argsort(s) s.t. sσ(1) ≤ sσ(2) ≤ · · · ≤ sσ(N). Rm ← {pσ(ℓ), pσ(ℓ+1), . . . , pσ(u)}.

- **Module 7: Optimization.** The main goal of πθ is to retrieve a set of passages that maximizes the likelihood of the pretrained LLM producing the correct answer to a given query. Equation (3): \max_\theta J(\theta) = \mathbb{E}_{(q,P,y)\sim D, R\sim\pi_\theta(\cdot|s)} [r_\psi(q,R, y)], where r_\psi(q,R, y) := \mathbb{1}_{corr} (F_\psi(q,R), y). By applying the likelihood ratio gradient with log-derivative trick, we can update θ at k-th gradient update step as Equation (4): θk+1 = θk + γ · rψ(q,R, y) · ∇θk log πθk(·|s), where γ denotes the step size. Through this optimization, πθ learns a distraction-aware retrieval strategy that reduces the likelihood of distracting passages interfering with the prediction LLM.

- **Module 8: Cost-Regularization Avoidance.** Introducing a cost term proportional to the number of retrieved tokens caused the optimization to collapse into a local optimum, where the LDAR strategy retrieved too few passages and suffered a drop in accuracy. Because the objective is not a linear trade-off but rather identifying the peak of this inverted-U, we found penalizing token usage during training often pushes the model away from the optimal region. Therefore, instead of imposing an explicit cost term, we focused on letting LDAR learn to retrieve passages near the performance peak of the inverted-U.

7. **All Mathematical Formulas and Symbol Definitions**

- fϕ: Pretrained embedding model.
- q: Query.
- {pi}Ni=1: Passages.
- Rd: Shared vector space.
- si: Similarity (e.g., dot product or cosine similarity) between the query and the i-th passage.
- σ: Sorting permutation such that sσ(1) ≤ sσ(2) ≤ . . . ≤ sσ(N).
- R: Retrieved passages.
- Equation (1): ∃σ s.t. sσ(1) ≤ sσ(2) ≤ . . . ≤ sσ(N), R = {pσ(N−k+1), . . . , pσ(N)}.
- πθ: Lightweight adaptive retriever.
- s: Cosine similarity vector between the query and the N passages.
- qL, qU: Lower and upper quantiles defining the contiguous quantile interval used for retrieval.
- B: Batch size.
- d: Embedding dimension.
- D: Transformer hidden dimension.
- w: Attention weights.
- z: Global summary vector obtained by attention pooling.
- g: Output of the MLP head.
- αL, βL: Parameters of the Beta distribution for the lower quantile.
- α∆, β∆: Parameters of the Beta distribution for the band width q∆.
- D: Dataset with each instance comprising a query q, a candidate passage pool P, and a ground-truth answer y.
- R: Set of passages retrieved by πθ given the similarity scores s.
- Fψ: Pretrained LLM.
- rψ(q,R, y): Reward function evaluating whether the output of the LLM matches the ground-truth answer.
- 1corr: Indicator function that evaluates whether the output of the LLM Fψ(q,R) matches ground-truth answer y.
- θk: Parameters of the adaptive retriever at the k-th gradient update step.
- γ: Step size.
- Equation (2): si := \frac{f_\phi(q)^\top f_\phi(p_i)}{\|f_\phi(q)\| \|f_\phi(p_i)\|}, i = 1, . . . , N.
- Equation (3): \max_\theta J(\theta) = \mathbb{E}_{(q,P,y)\sim D,R\sim\pi_\theta(\cdot|s)} [r_\psi(q,R, y)], where r_\psi(q,R, y) := \mathbb{1}_{corr} (F_\psi(q,R), y).
- Equation (4): θk+1 = θk + γ · rψ(q,R, y) · ∇θk log πθk(·|s).

8. **Algorithm Pseudocode**

Algorithm 1 Distraction-Aware Adaptive Retrieval
Require: Instances D = {(qm, Pm, ym)}Mm=1, Embedding model fϕ, Adaptive Retriever πθ
1: for m-th query qm and passages {pm,i}Ni=1 in D do
2: si ← fϕ(qm)⊤fϕ(pm,i) / (∥fϕ(qm)∥ ∥fϕ(pm,i)∥) for i = 1, · · · , N
3: (qL, qU ) ∼ πθ(· | s)
4: ℓ← max(1, ⌊N · qL⌉)
5: u← max(ℓ, ⌊N · qU⌉)
6: σ ← argsort(s) s.t. sσ(1) ≤ sσ(2) ≤ · · · ≤ sσ(N)
7: Rm ← {pσ(ℓ), pσ(ℓ+1), . . . , pσ(u)}
8: end for
9: return {Rm}Mm=1