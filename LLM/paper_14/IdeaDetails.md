1. **Research Background and Existing Pain Points**
Large Language Models (LLMs) have demonstrated remarkable reasoning capabilities in a wide range of complex natural language processing tasks. However, LLMs remain prone to hallucinations and factual errors in real-world applications due to their reliance on implicit parametric knowledge. Knowledge graphs (KGs), as large-scale structured external sources of factual knowledge, offer explicit, interpretable relational structures which can ground LLM reasoning, providing a natural complement to the limitations of LLMs. Recent LLM⊗KG approaches can be categorized into two paradigms. The first leverages step-wise graph exploration, where LLMs iteratively perform entity–relation walks to progressively construct reasoning paths. The second generates global reasoning plans where questions are decomposed into sub-objectives and the KG is queried along the planned path to obtain external information. Though demonstrating notable improvements, these methods often struggle with complex logical queries that involve conjunctions or multiple constraints. The existing pain points are:
- **Search Space Truncation Bias due to Linear Reasoning Paths:** Current methods construct reasoning paths primarily along linear entity–relation steps, iteratively expanding from one entity to its neighbors. To control the combinatorial explosion of graph exploration, they prune candidate entities at each step (e.g., using top-k selection). While efficiency, this strategy often eliminates correct entities prematurely. The limited planning capability of existing methods fundamentally biases the search space and limits reasoning performance.
- **Entity Error Amplification from Faulty Entities and Relations:** LLM-generated reasoning paths may introduce spurious or weakly related entities and relations during KG exploration. Existing methods typically follow a retrieve-and-answer paradigm, where the LLM heavily relies on the retrieved evidence to produce the final answer. This reliance amplifies errors. When a spurious or weakly related entity is introduced, the subsequent steps will propagate and accumulate this error. Furthermore, LLMs often assumes the retrieved evidences from KG to be correct and sufficient, even when the external information only partially address the query. This over-reliance prevents the model from self-correcting, leading to faulty answers.

2. **Core Research Motivation and Scientific Questions**
- **Core Research Motivation:** To alleviate hallucinations and factual errors in LLMs by tightly integrating structured explicit guidance with parametric LLM reasoning, specifically addressing the fundamental limitations of search space truncation bias and error amplification when handling multi-hop reasoning and complex logical queries.
- **Scientific Questions:** How to move beyond linear entity-relation expansions to handle complex logical operations (such as conjunctions, superlatives, comparatives, and compositions) without prematurely pruning correct candidates during iterative exploration? How to mitigate error amplification and over-reliance on retrieved KG evidence when the retrieved evidence is spurious, weakly related, or insufficient for the query?

3. **Overall Core Idea and Design Philosophy**
The overall core idea is to propose a Plan–Answer–Refine framework (PARoG), a hybrid reasoning paradigm that tightly integrates structured explicit guidance with parametric LLM reasoning. The design philosophy introduces two key technical contributions. First, to address Search Space Truncation Bias, PARoG leverages SPARQL queries as structured references to supervise planning and trains the planning module to generate flexible, compositional reasoning paths that allow complex logical operations over sub-queries (e.g. conjunctions, compositions, superlatives and comparatives). Instead of searching sequentially and expanding linearly, the model can generate conjunctive sub-objectives and reason over the combination of sub-objectives, which mitigates search space truncation bias. Second, to address Error Amplification, rather than committing to retrieved entities in a one-shot retrieve-and-answer paradigm, PARoG first produces a tentative answer using its parametric knowledge and then explicitly refines it by referring to the retrieved KG entities. This refinement step overrides earlier faulty evidences, preventing error amplification due to spurious entities or weakly related relations.

4. **Core Innovation Points**
- Proposing to leverage SPARQL as structured references to supervise planning to train the model to generate compositional reasoning paths, which enables the model to handle complex logical reasoning including conjunction, composition, superlative and comparative queries.
- Proposing a plan-answer-refine framework, where the agent first attempts to answer then explicitly refines the results using retrieved evidence. This step reduces error propagation caused by faulty entities or relations involved in the reasoning paths.
- Introducing a novel framework PARoG by combining the proposed techniques and evaluating the performance on multiple real-world KGQA datasets. The experimental results demonstrate significant improvements over state-of-the-art approaches, achieving especially superior accuracy on multi-hop and logically complex queries.
- Using a relatively small model (e.g., Llama-3.1-8B) to generate reasoning paths with SPARQL-guided supervision, yet its performance can surpass larger planning LLMs (e.g., ChatGPT or Deepseek-R1), demonstrating that structured symbolic guidance can enhance LLM reasoning.

5. **Overview of the Overall Technical Solution**
PARoG comprises 2 major stages: SPARQL-Guided Structured Planning and Plan-Answer-Refine Paradigm.
- **Stage 1: SPARQL-Guided Structured Planning:** This stage trains LLMs to generate compositional planning of sub-objective paths based on SPARQL-guided supervision for knowledge graph exploration. It utilizes a SPARQL-to-Planning pipeline consisting of Source Data Collection and Semantic Consistency Mapping to construct training data, followed by Model Training with supervised fine-tuning.
- **Stage 2: Plan-Answer-Refine Paradigm:** This stage iteratively completes a sub-objective with parametric knowledge and then corrects the answer using external KG evidence to mitigate errors and inconsistencies. It consists of three steps: Answering (generating a tentative answer using parametric reasoning), Exploration (iteratively exploring the KG to retrieve relevant relations and entities), and Self-Refinement (re-evaluating and refining the tentative answer against retrieved evidences).

6. **Detailed Module Design**
- **SPARQL-Guided Structured Planning Module:**
  - **SPARQL-Guided Supervision:** SPARQL inherently supports complex queries that involve logical operations. The module considers the following operation types: Conjunctions (finding entities that satisfy multiple constraints simultaneously), Compositions (expressing queries where the output of one relation serves as the input to another), Comparatives (retrieving entities based on relative attributes), and Superlatives (selecting the best entity according to a ranking predicate).
  - **SPARQL-to-Planning:**
    - **Source Data Collection:** Select diverse ⟨Question, SPARQL⟩ pairs of multi-hop queries from public KGQA training datasets including WebQSP, CWQ, and GrailQA.
    - **Semantic Consistency Mapping:** Decompose the SPARQL queries into sub-operations and then translate each atomic operation into a fluent natural language question as single sub-objective of the reasoning plan. Rephrase the decomposed sub-objective sequence back to natural language queries to maintain the semantic consistency. Use the rephrased natural language queries and the generated sub-objectives as the training data.
  - **Model Training:** Employ a relatively small but powerful open-source model Llama-3.1-8B as the foundation backbone. The training objective follows the standard autoregressive language modeling loss. With supervised training, the model learns to map complex natural language questions into sequences of structured sub-questions which mirror SPARQL compositional logic.

- **Plan-Answer-Refine Paradigm Module:**
  - **Answering:** Let O denotes the reasoning plan generated by the planning module. For each sub-objective oi ∈ O generated by the planning module, the initial step is leveraging a LLM M to generate a tentative answer.
  - **Exploration:** The KG exploration process starts with a set of n0 topic entities E0. For the i-th iteration (i > 1), first obtain the current set of K reasoning paths Pi−1 after previous i − 1 iterations. Continue to extend the reasoning paths forward based on the current triples. Begin with all relations connected to the tail entities in Ei−1, and employ the LLM to filter out irrelevant relations. In this step, the entire reasoning plan O is also provided to the LLM so that the model maintains awareness of the global reasoning objective. Given the tail entities and filtered relations, the missing entities are obtained using predefined SPARQL query templates such as (e, r, ?) or (?, r, e). Leverage the model to further calculate the relevance between the retrieved entities and the current sub-objective oi and the question Q. The most relevant entities from a large set of candidates are reserved to update the reasoning path set, denoted by Pi.
  - **Self-Refinement:** After each KG exploration iteration, PARoG explicitly re-evaluate the tentative answer against the retrieved evidences. If inconsistencies or supplementary are detected, PARoG refines the result by adjusting entities or relations, effectively correcting errors from earlier steps. Explicitly ask the LLM to judge whether the retrieved knowledge aligns with the question; if it does not, the generated tentative answer is directly used instead. After each round of self-refinement, PARoG is leveraged to determine whether the current result ai is sufficient to answer the overall question Q. If the answer is ”yes”, stop searching and use ai as the final answer to avoid over-exploration. Otherwise, PARoG continues iterative searches until PARoG finds enough knowledge or reach the maximum number of iterations.

7. **All Mathematical Formulas and Symbol Definitions**
- **Knowledge Graph:** A knowledge graph (KG) is composed of a large set of fact triples, represented as a graph G = {⟨e, r, e′⟩ | e, e′ ∈ E , r ∈ R}, where E and R denote the sets of entities and relations respectively.
- **Model Training Loss:**
  argminθ L(θ) = − ∑i=1^H ∑j=1^Th logPθ(oi,j | ohi,<j ,x)
  where H and Th deba the total number of sub-objectives and the token number of a single sub-objective respectively, and oi = {oi,1, · · · , oi,Th} is the h-th sub-objective.
- **Answering:**
  âi =M(Q, oi, IA)
  where IA denotes a predefined instruction template and Q is the input question.
- **Exploration Variables:**
  E0 = {e01, e02, . . . , e0n0}
  Pi−1 = pi−11 , · · · , pi−1K
  pi−1k = [(ei−1,1s,k , ri−1,1k , ei−1,1o,k ), · · · , (ets,k, ri−1tk , ei−1,to,k ), · · · , (eTks,k, ri−1,Tkk , eTko,k)]
  where Tk < i, t indexes the elements, ei−1,ts,k and ei−1,to,k denote the subject and object entities respectively, and ri−1,tk is a relation linking them.
  Ei−1 = {ei−11 , ei−12 , · · · ei−1nk}
  Ri−1 = {ri−11 , ri−12 , · · · ri−1nk}
  RKinit = {riinit,1, riinit,2, . . . , riinit,n}
- **Self-Refinement:**
  ai =M(Pi, oi, âi, IR)
  where Pi denotes the set of retrieved triples in the current iteration, and IR is the instruction prompt.

8. **Algorithm Pseudocode**
**Algorithm 1 PARoG.**
Require: Question Q, Knowledge Graph G, LLMM, Planning module PLAN(·), instruction templates IA, IR, initial topic entity set E0 (size n0), max iterations Tmax
Ensure: Final answer a
1: O ← PLAN(Q) ▷ Generate sub-objectives with the SPARQL-supervised planner
2: A ← ∅
3: for each sub-objective oi ∈ O do
4: âi ←M(Q, oi, IA) ▷ Tentative answer by parametric reasoning
5: P0 ← {[ ]} ▷ Initialize reasoning paths
6: E0 ← {e01, . . . , e0n0}
7: for t = 1 to Tmax do
8: Et−1 ← TAILENTITIES(Pt−1)
9: Rinit t ← NEIGHBORRELATIONS(G, Et−1)
10: Rt ←M(Q, oi,O,Rinit t ) ▷ Filter relations with global plan awareness
11: Ecandt ← SPARQLQUERY(G, Et−1,Rt)
12: SCORE(e)←M(Q, oi, e), ∀e ∈ Ecandt
13: Et ← SELECTRELEVANT(Ecandt , SCORE) ▷ Select a variable number of most relevant entities
14: Pt ← EXTENDPATHS(Pt−1, Et,Rt)
15: ai ←M(Pt, oi, âi, IR) ▷ Self-refine tentative answer
16: if ALIGN(Pt, Q) = false then
17: ai ← âi ▷ Fallback if retrieved knowledge is irrelevant
18: end if
19: if SUFFICIENT(ai, Q) = true then
20: break
21: end if
22: end for
23: A ← A∪ {ai}
24: end for
25: a← AGGREGATE(A,O) ▷ Combine refined sub-answers according to O to form the final answer.
26: return a