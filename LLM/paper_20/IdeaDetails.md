# 1. Research Background and Existing Pain Points

**Research Background:** Large language models (LLMs) have achieved impressive performance on knowledge-intensive tasks, yet they often struggle with multi-step reasoning due to the absence of precise, logically organized information. Retrieval-augmented generation (RAG) approaches provide LLMs with additional context from retrieved passages but often face hallucination challenges, where generated content deviates from retrieved information. Recent interpretability analyses have suggested that LLMs attempt to chain facts across context internally, and failures in these implicit reasoning chains correlate with hallucinations. These findings reinforce the need for explicitly structured intermediate knowledge to guide reasoning. Furthermore, recent efforts have integrated knowledge graphs (KGs) with LLMs to provide compact relational representations that support more interpretable reasoning.

**Existing Pain Points:**
1. **Unstructured Retrieval Context:** RAG methods lack explicit organization among retrieved passages, which limits their effectiveness, leading to brittle reasoning pathways and forcing the model to implicitly bridge logical gaps.
2. **High Cost of Static Global KGs:** Existing KG-based approaches typically rely on static, corpus-wide graphs. Global KGs are costly to build and maintain; indexing a corpus like Wikipedia 2018 can require millions of LLM calls and cost tens of thousands to millions of USD.
3. **Noise and Contradictions in Global KGs:** Global graphs often blend evidence from many documents, leading to ambiguous or even contradictory relations (e.g., biomedical KGs containing both positive and negative associations between a drug and a disease reflecting conflicting studies, or global KGs encoding multiple irrelevant aspects of an entity without clarifying relevance to the query).
4. **Inefficiency of Offline Indexing:** Static corpus-level graph methods require building and maintaining large knowledge graphs offline, involving heavy clustering, embedding computation, and days of GPU/CPU compute, which introduces prohibitive upfront investment requirements.

# 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:** The limitations of both traditional RAG and static KG-based approaches highlight the critical need for knowledge graphs that are constructed on demand, tailored to the query, and structured to support reasoning. Interpretability studies highlight that explicit, structured intermediate reasoning can mitigate failures in implicit reasoning chains. A question-specific KG built from targeted documents resolves conflicts by grounding relations in a coherent, query-relevant context, avoiding both the inefficiency of offline indexing and the noise of global graphs.

**Scientific Questions:**
- How can a framework dynamically construct question-specific knowledge graphs that capture only task-relevant information during inference?
- How can a model jointly plan retrieval strategies and generate answers over evolving knowledge structures to enable precise and robust reasoning?
- How can structured knowledge representation be integrated with LLMs to avoid the prohibitive costs of offline corpus-wide indexing while improving multi-step reasoning accuracy?

# 3. Overall Core Idea and Design Philosophy

**Overall Core Idea:** The paper proposes Retrieval-And-Structuring (RAS), a framework that dynamically constructs and reasons over question-specific knowledge graphs through iterative retrieval and structured knowledge building. RAS interleaves targeted retrieval planning with incremental graph construction, enabling models to assemble and reason over evolving knowledge structures tailored to each query.

**Design Philosophy:**
- **Dynamic vs. Static:** RAS dynamically constructs question-specific graphs tailored to each question rather than relying on static, corpus-wide graphs.
- **Iterative Structuring:** RAS iteratively plans and fills knowledge gaps at inference, converting retrieved content into a structured, evolving graph aligned with the query.
- **Unified Modeling:** RAS employs a unified graph structure-aware model (Graph LLM) that jointly plans retrieval and generates answers over the evolving knowledge graph using parameter-efficient fine-tuning (LoRA).
- **On-Demand Reasoning:** The framework only extracts and structures information relevant to the current reasoning trajectory, ensuring denser, task-relevant context and eliminating unnecessary offline processing.

# 4. Core Innovation Points

1. **Dynamic Question-Specific KG Construction:** RAS dynamically constructs question-specific knowledge graphs through iterative retrieval and structuring, departing from static, global KGs that are costly to build and maintain. This design avoids the noise of global graphs and eliminates offline indexing costs.
2. **Unified Graph Structure-Aware Model:** A unified model (Graph LLM) is designed to jointly plan retrieval and generate answers over evolving knowledge graphs. This structure-aware model uses a Graph Neural Network (GNN) to encode and project the evolving KG, allowing the LLM to attend to structured knowledge.
3. **Iterative Knowledge Enrichment Loop:** The framework implements an iterative loop where planning identifies knowledge gaps, retrieval-and-structuring fills them by extracting triples and merging them into the evolving KG. This allows the model to systematically assemble evidence tailored to the reasoning trajectory.
4. **Structure-Aware Multitask Learning:** A multitask training setup that unifies knowledge-aware planning and knowledge-augmented answering under a standard next-token prediction objective. The model is trained to perform both action planning (generating subqueries or stopping criteria) and answer generation based on the accumulated graph context.
5. **Cost-Efficiency and Robustness:** By constructing graphs dynamically at inference time, RAS eliminates the prohibitive offline indexing costs associated with methods like GraphRAG (estimated $85k–$500k for Wikipedia scale) and achieves substantial performance gains over strong RAG baselines.

# 5. Overview of the Overall Technical Solution

The RAS process unfolds in three main stages:
1. **Planning:** The model strategically determines retrieval needs and generates focused sub-queries based on the current knowledge state (accumulated graph and history).
2. **Text Retrieval and Structuring:** The system retrieves passages based on sub-queries, extracts factual triples using a text-to-triples model, and merges them into an evolving question-specific knowledge graph that expands iteratively with reasoning needs.
3. **Answering:** The accumulated structured knowledge is leveraged to generate the final output.

The framework uses a single LLM backbone enhanced with a Graph Neural Network (GNN) encoder and LoRA adapters, trained via structure-aware multitask learning to handle both planning and answering tasks.

# 6. Detailed Module Design

**Key Definitions:**
- **Main question (Q):** The original task input.
- **Subquery (qi):** A focused retrieval query generated at iteration $i$ to obtain supporting evidence.
- **Retrieved text (ti):** A set of the top-k documents retrieved by $q_i$.
- **Text-to-triples model (ft2t):** Converts retrieved text into triples $g_i$, structured as subject-predicate-object facts.
- **Evolving question-specific KG (GQ):** Represents organized evidence related to Q, incrementally accumulated.
- **Model (M):** The LLM used for planning and answering.
- **Plan (pi):** The decision at each step: `[SUBQ]` (continue retrieval), `[SUFFICIENT]` (stop retrieval), or `[NO_RETRIEVAL]` (answer directly).

**Module 1: Knowledge-Aware Planning**
This module initiates and controls the retrieval-and-structuring process by dynamically assessing the current knowledge state.
- **Initial Planning:** Given input query $Q$, the model $M$ generates an initial plan $p_0$. If $M$ assesses the query cannot be answered with its own knowledge, it outputs `[SUBQ] q_0 = Q` to start the iteration. If $M$ determines $Q$ can be answered directly, it outputs `[NO_RETRIEVAL]` and terminates planning.
- **Iterative Planning:** At iteration $i > 0$, given the accumulated knowledge $G_i$ and subquery-triples history, the model updates the plan. If the model determines more information is needed, it outputs `[SUBQ] q_{i+1}` to generate a new subquery designed to fill specific gaps. If the accumulated knowledge $G_i$ is deemed sufficient, it outputs `[SUFFICIENT]` and proceeds to answering.

**Module 2: Text Retrieval and Structuring**
Once `[SUBQ]` is detected, this module retrieves text and transforms it into structured knowledge, progressively merged into the question-specific graph $G_Q$.
- **Text Retrieval:** A standard dense retriever is used to retrieve the top-k semantically relevant passages $t_i$ from the corpus $C$ for subquery $q_i$.
- **Text-to-Triples Conversion:** A text-to-triples model $f_{t2t}$ extracts essential factual information from $t_i$, generating structured triples in the format (subject, predicate, object).
- **Iterative Knowledge Enrichment:** The extracted triples $g_i$ are converted into a graph structure $g'_i = (V_i, E_i)$ where nodes are unique entities and edges are predicates. Semantic embeddings are obtained via Sentence-BERT for nodes and edges. The structured graph $g'_i$ is then merged into the evolving KG $G_Q = (V_Q, E_Q)$ to progressively enrich the question-related knowledge.

**Module 3: Knowledge-Augmented Answering**
When answering is triggered, the model $M$ generates an answer $A$ conditioned on the knowledge graph $G_Q$ and subquery chain.
- If no retrieval is needed ($p_0 = [NO\_RETRIEVAL]$), the answer is generated directly.
- Otherwise, after iterative enrichment concludes with `[SUFFICIENT]` or maximum iteration is reached, the answer is generated using the encoded KG (via GNN) and subquery chain. $M$ attends to knowledge in $G_Q$ and subquery chain to generate accurate, coherent answers.

**Module 4: Structure-Aware Multitask Learning**
The framework is trained through a multitask setup that unifies knowledge-aware planning and knowledge-augmented answering under a standard next-token prediction objective. The model $M$ is based on Graph LLM, adapted with a Graph Transformer encoder and LoRA.
- **Planning Task:** Input includes current encoded $G_Q$, planning instruction $INST_{Plan}$, subquery-triples history, and $Q$. Output is the next plan $p_{i+1}$.
- **Answering Task:** Input includes final encoded $G_Q$, answering instruction $INST_{Ans}$, subquery-triples history, and $Q$. Output is the final answer $A$.

# 7. All Mathematical Formulas and Symbol Definitions

**Symbol Definitions:**
- $Q$: Main question (original task input).
- $q_i$: Subquery at iteration $i$.
- $t_i$: Retrieved text (top-k documents for $q_i$).
- $f_{t2t}$: Text-to-triples conversion model.
- $g_i$: Extracted triples from $t_i$.
- $G_Q$: Evolving question-specific knowledge graph for $Q$.
- $G_i$: Accumulated knowledge at iteration $i$.
- $M$: The language model.
- $p_i$: Plan at step $i$.
- $INST_{Plan}$: Planning instruction.
- $INST_{Ans}$: Answering instruction.
- $C$: Corpus for retrieval.
- $k$: Number of top documents to retrieve.
- $g'_i$: Graph structure corresponding to triples $g_i$, $g'_i = (V_i, E_i)$.
- $V_i, E_i$: Sets of nodes and edges in $g'_i$.
- $V_Q, E_Q$: Sets of nodes and edges in $G_Q$.
- $A$: Generated answer.

**Mathematical Formulas:**
- Initial Planning:
  $p_0 \leftarrow M(\emptyset;INST_{Plan}; \emptyset;Q)$
- Iterative Planning:
  $p_{i+1} \leftarrow M(GNN(G_i);INST_{Plan}; [q_0, g_0, ..., q_i, g_i];Q)$
- Text Retrieval:
  $t_i \leftarrow Retrieval(q_i, C, k)$
- Text-to-Triples Conversion:
  $g_i \leftarrow ft2t(ti) = [(s_0, r_0, o_0), ..., (s_{|g_i|}, r_{|g_i|}, o_{|g_i|})]$
- Node and Edge Embedding:
  $emb(v) \leftarrow encode(v), \forall v \in V_i; emb(e) \leftarrow encode(e), \forall e \in E_i$
- Iterative Knowledge Enrichment:
  $G_Q \leftarrow G_Q \cup g'_i$
- Answering (No Retrieval):
  $A \leftarrow M(\emptyset;INST_{Ans}; \emptyset;Q)$
- Answering (With Retrieval):
  $A \leftarrow M(GNN(G_Q);INST_{Ans}; [q_0, g_0, ..., q_i, g_i];Q)$
- Evaluation Metric - Golden Match:
  $match(p,G) = \begin{cases} 1 & \text{if } \exists g \in G : norm(g) \subseteq norm(p) \\ 0 & \text{otherwise} \end{cases}$
- Evaluation Metric - Precision:
  $precision = \frac{|tokens(p) \cap tokens(g)|}{|tokens(p)|}$
- Evaluation Metric - Recall:
  $recall = \frac{|tokens(p) \cap tokens(g)|}{|tokens(g)|}$
- Evaluation Metric - F1 Score:
  $F1(p, g) = \frac{2 \cdot precision \cdot recall}{precision + recall}$
- Evaluation Metric - Accuracy:
  $accuracy = \frac{|\{i : norm(p_i) = norm(g_i)\}|}{N} \times 100$

# 8. Algorithm Pseudocode

```text
Algorithm 1 HotpotQA-SUBQ Sample Labeling
Require: D: Filtered HotpotQA dataset
Require: M: Base LLM (LLaMA-2-7B)
Require: ft2t: Text-to-triple conversion model
Ensure: Tplan: Training data for Planning
Ensure: Tans: Training data for Answering
1: Initialize Tplan, Tans ← {}, {}
2: for all (Q, {d0, ..., dn}, A) ∈ D do
3:   Â ← M(Q) ←↩ Direct answer attempt
4:   if Â = A then
5:     Tplan ← Tplan ∪ {(Q,[NO_RETRIEVAL])}
6:     Tans ← Tans ∪ {(Q,A)}
7:     continue
8:   end if
9:   subq0, ..., subqn ← GenerateSubqueries(Q, {d0, ..., dn})
10:  G0, ..., Gn ← ∅ ←↩ Initialize graph contexts
11:  for i ← 0 to n do
12:    gi ← ft2t(di) ←↩ Convert text to triples
13:    if i < n then
14:      input ← FormatInput({(subqj , gj)}i j=0, Q)
15:      Tplan ← Tplan ∪ {(input,[SUBQ] subqi+1)}
16:    else
17:      input ← FormatInput({(subqj , gj)}n j=0, Q)
18:      Tplan ← Tplan ∪ {(input,[SUFFICIENT])}
19:      Tans ← Tans ∪ {(input, A)}
20:    end if
21:    Gi ← Gi−1 ∪ gi ←↩ Accumulate graph context
22:  end for
23: end for
24: return Tplan, Tans
```