### 1. Research Background and Existing Pain Points

**Research Background:**
Large language models (LLMs) have achieved strong performance on knowledge-intensive tasks driven by billions of parameters and large-scale pretraining corpora. However, their parametric knowledge remains static, making them ill-suited for evolving world knowledge or emerging domains. To address this, two dominant approaches exist: (1) Retrieval-Augmented Generation (RAG), which supplements LLMs with in-context retrieved passages, and (2) Parametric Knowledge Adaptation (PKA), which directly updates model parameters with new domain knowledge. Recently, Parametric RAG (PRAG) has emerged as a promising framework, extending RAG by translating retrieved passages into parameter updates via a “hypernetwork”, thereby mitigating inefficiency and noise sensitivity inherent to standard RAG.

**Existing Pain Points:**
1. **RAG Limitations:** RAG faces challenging issues: (1) knowledge conflict between parametric and retrieved information, (2) inference inefficiency from processing long retrieval-heavy contexts, and (3) noise sensitivity, where irrelevant or erroneous passages degrade performance.
2. **PRAG Limitations in Multi-hop Settings:** Existing PRAG methods remain limited to single-pass retrieval, falling short of the multi-hop RAG setting that requires iterative retrieval and reasoning. In multi-hop RAG, a complex query is decomposed into sub-questions, each requiring iterative retrieval and sub-answer generation, such that retrieved passages are incrementally provided. Existing PRAG fails to continuously internalize retrieved passages across hops without necessitating the retraining or rebuilding of a hypernetwork originally designed for single-hop RAG.

### 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:**
To extend PRAG to multi-hop settings where the internalization of retrieved passages must continuously progress across hops, enabling LLMs to effectively accumulate external knowledge through iterative retrievals without retraining the hypernetwork. This extension represents an important milestone toward reasoning-enhanced RAG frameworks.

**Scientific Questions:**
How to effectively extend PRAG to multi-hop settings—where the internalization of retrieved passages must continuously progress across hops—without necessitating the retraining or rebuilding of a hypernetwork originally designed for single-hop RAG?

### 3. Overall Core Idea and Design Philosophy

**Overall Core Idea:**
Propose MergePRAG (Orthogonal Merging of Passage-experts for Multi-hop PRAG), a generalized framework that scales PRAG to multi-hop RAG. At each stage, retrieved passages are translated into expert parameters by a hypernetwork and merged with the previously accumulated experts through a continual merging mechanism, thus enabling effective accumulation of knowledge across iterative retrievals. 

**Design Philosophy:**
Reuse a single passage-level hypernetwork across hops, avoiding the redesign or retraining of additional hypernetworks to support multi-hop RAG. To achieve effective continual merging, the design employs orthogonal merging using the Gram–Schmidt process to minimize conflicts between newly introduced and existing experts, and critical-layer parameterization to efficiently encode in-context passages while keeping the remaining layers frozen to reduce computation and stabilize reasoning.

### 4. Core Innovation Points

1. **First Generalized PRAG Framework for Multi-hop RAG:** Introduction of MergePRAG, which extends the PRAG framework to the multi-hop RAG setting, serving as a critical stepping stone toward reasoning-enhanced RAG systems.
2. **Continual Merging Mechanism:** Proposal of a continual merging mechanism that sequentially integrates retrieved passages into LLM parameters, inducing the mapping from accumulated passages to parameter space by reusing the single passage-level hypernetwork.
3. **Orthogonal Merging via Gram–Schmidt Process:** Introduction of orthogonal merging to minimize conflicts between "passage experts" by eliminating redundant information and promoting complementary, non-overlapping representations, ensuring that new parameters only add orthogonal components with respect to previously merged parameters.
4. **Critical-Layer Parameterization:** A module that updates only the preselected critical layer to efficiently encode in-context passages, while keeping the remaining layers frozen to reduce computation and stabilize reasoning, motivated by locate-and-edit methods.

### 5. Overview of the Overall Technical Solution

MergePRAG tackles multi-hop QA by decomposing a complex query into sub-questions. For each timestep $t$:
1. A sub-question $sq_t$ is generated from the reasoning chain $C_{t-1}$ using the sub-question generator $M_{sq}$.
2. The retriever returns top-ranked passages $SP_t \subseteq R(sqt)$.
3. Given $SP_t = [p_i]_{i=1}^m$, each passage is parameterized by the hypernetwork to produce $\{H_\phi(p_i)\}_{i=1}^m$, which are combined into $H_\phi(SP_t)$ via the inner-merging mechanism.
4. Orthogonal continual merging updates the accumulated parameters $F(SP_{1:t-1})$ with $H_\phi(SP_t)$ to obtain $F(SP_{1:t})$.
5. The merged expert $F(SP_{1:t})$ is injected into the base LLM $M_{\theta_0}$ at the critical layer $l^*$ to generate the sub-answer.
This process repeats until no further sub-questions are produced, after which the final answer is generated using the fully passage-injected model.

### 6. Detailed Module Design

**1. Multi-hop RAG Formulation:**
Two language models and a retrieval module are defined: $M_{\theta_0}$ (base LLM for sub-answer generation), $M_{sq}$ (smaller LLM for sub-question generation), and $R$ (retriever). At step $t$, given $C_{t-1}$, the next sub-question and sub-answer are obtained, and the context is updated.

**2. MergePRAG Core Mechanisms:**
- **Sequence-merging ($Merge_{seq}$):** A recursive operation that combines the previously accumulated parameters $F(SP_{1:t-1})$ with the new passage-specific parameters $H_\phi(SP_t)$.
- **MergePRAG+:** Integrates RAG and PRAG in a complementary manner, using both injected parameters and in-context passages.
- **Inner-merging ($Merge_{inner}$):** Induces $H(SP)$ from individual passage parameters when $|SP| > 1$.

**3. Hypernetwork-based Key–Value Memory Parameterization for $H_\phi$:**
The hypernetwork generates $k$ key and value vectors for each passage, serving as a “compressed” passage-specific memory. The passage-specific memory is inserted into the feed-forward network (FFN) at the critical layer $l^*$ via a memory attention mechanism. The original FFN output is used as the query to attend to the passage-specific key-value memory, forming a passage-specific FFN expert, which is then injected into the original FFN layer.

**4. Orthogonal Continual Merging Mechanism ($Merge$):**
Operates on each parameter (key or value memory matrices) independently. To form a merged expert without overwriting previously acquired knowledge, the Gram–Schmidt orthogonalization procedure computes the projection matrix onto the subspace spanned by the previously merged parameters. The new parameter is then merged by adding only its orthogonal component with respect to this subspace.

**5. Hypernetwork Architecture: Sequence-to-Memory:**
Given a passage as a token sequence, the hypernetwork computes a passage embedding via attentive pooling over token-level embeddings. The resulting passage embedding is passed through a two-layer MLP, whose output is transformed by linear projections to generate the passage-specific memory (key and value matrices).

**6. Critical-Layer Parameterization:**
The hypernetwork is applied only to a single critical layer $l^*$, rather than across all layers. The critical layer is identified by layer-wise scanning experiments measuring the change in perplexity after injecting the corresponding passage vectors. Early-to-middle layers typically contribute most substantially.

**7. Training Objective:**
- **Hypernetwork $H_\phi$:** Trained by minimizing the cross-entropy loss on a dataset of question-passage-answer triples.
- **Subquestion generator $M_{sq}$:** Trained via a cold-start stage using an autoregressive objective on a constructed dataset of decomposed reasoning chains.

### 7. All Mathematical Formulas and Symbol Definitions

**Symbol Definitions:**
- $M_{\theta_0}$: General-purpose LLM for sub-answer generation (base LM)
- $M_{sq}$: Sub-question generator based on a smaller LLM
- $R$: Retriever
- $q$: Original complex query
- $C_{t-1}$: Accumulated context so far
- $sq_t$: Sub-question at step $t$
- $sa_t$: Sub-answer at step $t$
- $SP_t$: Retrieved passages at step $t$
- $H_\phi$: Hypernetwork mapping a retrieved passage $p$ to passage-specific LoRA parameters
- $\theta_p$: Passage-specific LoRA parameters, $\theta_p = H_\phi(p)$
- $\oplus$: Parameter-injection operation
- $F$: Mapping from the sequence $SP_{1:t}$ to the parameter space
- $Merge_{seq}$: Sequence merging operation
- $Merge_{inner}$: Inner merging operation
- $K_p, V_p$: Key and value matrices for passage-specific memory, $\in \mathbb{R}^{k \times d_{out}}$
- $E_{H_\phi(p)}$: Passage-specific FFN expert
- $MLP_{\theta_0}$: Original FFN module at layer $l^*$
- $W_i$: Key or value memory matrices from $H_\phi(SP_i)$, $W_i \in \mathbb{R}^{k \times d_{out}}$
- $W_F^{t-1}$: Merged parameter obtained from $\{W_i\}_{i=1}^{t-1}$
- $P_{t-1}$: Projection matrix onto the subspace spanned by $W_F^{t-1}$
- $Emd(p)$: Sentence embedding obtained via attentive pooling
- $h_b$: Latent representation from MLPhyp
- $W_K, W_V$: Linear projection tensors, $\in \mathbb{R}^{K \times d \times d_{hid}}$
- $b_K, b_V$: Bias terms, $\in \mathbb{R}^{K \times d}$
- $D_H$: Training dataset for hypernetwork, $\{(q_i, p_i, a_i)\}_{i=1}^N$
- $D_{sq}$: Training dataset for sub-question generator, $\{(q^{(j)}, y^{(j)})\}_{j=1}^M$
- $l^*$: Critical layer for parameterization

**Mathematical Formulas:**
- Sub-question and sub-answer generation:
  $sq_t = M_{sq}(C_{t-1}), sa_t = M_{\theta_0}(sq_t, SP_t), SP_t \subseteq R(sq_t)$
- Context update:
  $C_t = [C_{t-1}, sq_t, sa_t]$
- Final answer generation:
  $a = M(C_{T-1}, q)$, where $sq_T = M_{sq}(C_{T-1}) = \langle EOS \rangle$
- PRAG parameter injection:
  $\theta' = \theta_0 \oplus H_\phi(p)$
- PRAG answer generation:
  $a = M_{\theta_0 \oplus H_\phi(p)}(q)$
- Sequence-merging:
  $F(SP_{1:t}) = Merge_{seq}(F(SP_{1:t-1}), H_\phi(SP_t))$
- MergePRAG sub-answer generation:
  $sa_t = M_{\theta_0 \oplus F(SP_{1:t})}(sq_t)$
- MergePRAG final answer generation:
  $a = M_{\theta_0 \oplus F(SP_{1:T})}(q)$
- MergePRAG+ formulation:
  $sa_t = M_{\theta_0 \oplus F(SP_{1:t})}(SP_t, sq_t), t < T$
  $a = M_{\theta_0 \oplus F(SP_{1:T})}(C_T, q), t = T$
- Inner-merging:
  $H([p_i]_{i=1}^m) = Merge_{inner}(H_\phi(p_1), \dots, H_\phi(p_m)) = Merge_{inner}(H([p_i]_{i=1}^{m-1}), H(p_m))$
- Passage-specific memory generation:
  $H_\phi(p) = \{K_p, V_p\}$
- Passage-specific FFN expert:
  $E_{H_\phi(p)}(x) = Attention(MLP_{\theta_0}(x), K_p, V_p)$
  $Attention(q, K_p, V_p) = softmax(\frac{q K_p^\top}{\sqrt{d_{out}}}) V_p$
- Injection into original FFN layer:
  $MLP_{\theta_0 \oplus H_\phi(p)}(x) = MLP_{\theta_0}(x) + E_{H_\phi(p)}(x)$
- Orthogonal merging projection matrix:
  $P_{t-1} = W_F^{t-1} ((W_F^{t-1})^\top W_F^{t-1})^{-1} (W_F^{t-1})^\top$
- Orthogonal continual merging update:
  $W_F^t = W_F^{t-1} + (I - P_{t-1}) W_t$
- Hypernetwork passage embedding via attentive pooling:
  $Emd(p) = h = softmax(w_a^\top X_{emd}^\top) X_{emd} \in \mathbb{R}^d$
- Hypernetwork latent representation:
  $h_b = MLPhyp(Emd(p)) = ReLU(V' LN (ReLU(W' Emd(p))))$
- Hypernetwork linear projection to key and value:
  $K_p = W_K h_b + b_K, V_p = W_V h_b + b_V$
- Hypernetwork training loss:
  $L_{CE}(\phi) = -\sum_{(q,p,a) \in D_H} \log P_{M_{\theta_0 \oplus H_\phi(p)}}(a | q)$
- Sub-question generator target sequence:
  $y^{(j)} = [ sq_1^{(j)}, sa_1^{(j)}, sq_2^{(j)}, sa_2^{(j)}, \dots, sq_{n_j}^{(j)}, sa_{n_j}^{(j)}, \langle EOS \rangle ]$
- Sub-question generator training loss:
  $L(M_{sq}) = -\sum_{j=1}^M \sum_{t=1}^{|y^{(j)}|} \log P_{M_{sq}}(y_t^{(j)} | q^{(j)}, y_{<t}^{(j)})$

### 8. Algorithm Pseudocode

**Algorithm 1: Multi-hop Inference with MergePRAG**
```
Require: Original question q, sub-question generator Msq , base LLM Mθ0 , hypernetwork Hϕ, retriever R
Ensure: Final answer a
1: Initialize merged expert F ← ∅
2: Initialize reasoning chain C ← ∅
3: while next sub-question exists do
4:     Generate sub-question: sqi ← Msq(q, C)
5:     Retrieve passages: SPi ← R(sqi)
6:     Parameterize passages: Hϕ(SPi)← Mergeinner({Hϕ(p) | p ∈ SPi}) (Eq. 6)
7:     if i > 1 then
8:         Orthogonal continual merge: F ← Mergeseq(F ,Hϕ(SPi)) (Sec. 3.2.2)
9:     else
10:        Initialize expert: F ← Hϕ(SPi)
11:    end if
12:    Inject F into the base LLM:Mθ0⊕F
13:    Generate sub-answer:
14:    sai ← { Mθ0⊕F (sqi), (MergePRAG)
             Mθ0⊕F (SPi, sqi), (MergePRAG+) }
15:    Append (sqi, sai) to reasoning chain C
16: end while
17: Generate final answer: a←Mθ0⊕F (C, q) (MergePRAG / MergePRAG+)
18: return a
```

**Algorithm 2: Multi-hop Inference with MultihopRAG**
```
Require: Original question q, sub-question generator Msq , base LLM Mθ0 , retriever R
Ensure: Final answer a← ∅
1: Initialize reasoning chain C ← ∅
2: while next sub-question exists do
3:     Generate sub-question: sqi ← Msq(q, C)
4:     Retrieve passages: SPi ← R(sqi)
5:     Generate sub-answer: sai ← Mθ0(SPi, sqi)
6:     Append (sqi, sai) to reasoning chain C
7: end while
8: Generate final answer: a←Mθ0(C, q)
9: return a
```