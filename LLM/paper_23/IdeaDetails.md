**1. Research Background and Existing Pain Points**

**Research Background:**
With an ever-growing zoo of LLMs and benchmarks, the need to orchestrate multiple models for improved task performance has never been more pressing. As the ecosystem of Large Language Models (LLMs) rapidly diversifies, researchers are increasingly overwhelmed—not just by the sheer number of models available, but also by the complexity of evaluating and combining them effectively at test time to solve complex tasks. While frameworks like Mixture-of-Agents (MoA) attempt to coordinate LLMs to enhance overall model performance through the Mixture-of-Experts (MoE) framework by aggregating responses from multiple LLM agents, appending them to the original query and feeding the enriched input to the next layer, they still face several limitations that hinder their scalability and effectiveness.

**Existing Pain Points:**
1. **Which agents? (Lack of effective agent selection mechanism):** As shown in the scaling law of diverse LLM agents, selecting a relevant subset from a large pool of agents handling complex and diverse queries is a crucial challenge. The current MoA approach lacks an effective agent selection mechanism, instead forwarding queries to all available agents, leading to excessive computational costs. This not only risks multi-agent system explosion but also introduces noise from irrelevant agents, highlighting the need for a more efficient and practical agent selection strategy.
2. **How do they communicate? (Ineffective intra-agent communication):** Once agents are selected, facilitating effective communication among them becomes a pivotal challenge. MoA adopts a many-to-one aggregation scheme, but this approach has notable drawbacks: it requires collecting responses from all available agents and treating them as a single chunk, which fails to capture fine-grained interactions—such as one-to-one communication between individual agent pairs. Moreover, since different agents have varying strengths and relevance depending on the query, treating all agent messages equally can hinder consensus. Instead, adaptively weighting responses based on their relevance is crucial for improving decision-making.
3. **How to integrate? (Inefficient response integration):** Finally, when making the final decision, MoA aggregates responses by concatenating tokens from all agents. However, this approach incurs a huge computational cost, with a complexity of O(LNd), where L is the number of communication layers, N is the number of agents, and d is the token length per agent. Given that token usage is directly tied to cost, this method becomes prohibitively expensive. Moreover, not all agents contribute equally valuable responses—some domain-specialized agents produce significantly higher-quality outputs than others. A scalable integration mechanism that prioritizes more reliable agents while reducing the impact of less relevant ones is essential for cost-efficient and effective decision-making.

**2. Core Research Motivation and Scientific Questions**

**Core Research Motivation:**
To address the challenges of agent selection, communication, and integration in existing multi-agent LLM systems, there is a fundamental need to rethink multi-agent collaboration. Modeling agents as nodes and their relevance-based relationships as edges can enable structured message passing, allowing for the selective activation of only relevant agents while capturing inter-agent interactions and finalizing decisions efficiently via graph-based pooling. This positions a graph-based structure as a strong candidate for navigating the challenges of the ever-growing LLM zoo.

**Scientific Questions:**
(Q) Given the diversity of available LLMs, how can we design an effective playground where agents interact synergistically—leveraging strengths, compensating for weaknesses, and improving decision-making through efficient collaboration?

**3. Overall Core Idea and Design Philosophy**

**Overall Core Idea:**
We propose Graph-of-Agents (GoA), a new graph-based framework for modeling multi-agent LLM communication. GoA fundamentally rethinks multi-agent collaboration by modeling agents as nodes and their relevance-based relationships as edges, enabling structured message passing. This graph-based design allows for the selective activation of only relevant agents (as a subgraph of entire pool) while capturing inter-agent interactions through message passing and finalizing decisions via graph-based pooling.

**Design Philosophy:**
GoA follows a structured process to answer the three core questions:
❶ **Which agents? → Node Sampling:** GoA begins by selecting relevant agents based on available metadata (e.g., domain, task) from model cards. This information is provided to a meta-LLM, simply a general-domain LLM, which identifies the most relevant agents given the query.
❷ **How do they communicate? → Edge Sampling & Message Passing:** Once nodes (agents) are selected, we obtain their initial responses and ask each agent to rank others, capturing the significance of each agent’s output. Based on these rankings, we construct directed edges in two perspectives: (i) Source-to-Target: Higher-ranked agents propagate their information to lower-ranked agents; (ii) Target-to-Source: After lower-ranked agents update their responses, the refined information is passed back to the higher-ranked agents.
❸ **How to integrate? → Graph Pooling:** With refined responses, GoA applies max or mean pooling, akin to graph pooling, to adaptively aggregate the outputs of multiple agents. 

Unlike traditional multi-agent learning methods that require model fine-tuning or additional training, GoA operates purely through the prompt interface. This design ensures compatibility with black-box LLM APIs while maintaining high adaptability across diverse domains during test-time inference.

**4. Core Innovation Points**

1. **Identification of Key Challenges:** We identify key challenges in current multi-agent LLM systems: selecting which agents to sample, facilitating effective communication, and integrating responses efficiently.
2. **Graph-based Formulation of Multi-Agent Collaboration:** We formulate multi-agent collaboration as a graph-based framework and introduce GoA, incorporating node sampling, edge sampling, message passing, and graph pooling. This enables construction of a scalable multi-agent LLM ecosystem, while enhancing inter-agent communication.
3. **Node Sampling via Model Cards:** GoA begins with node sampling, selecting only the most relevant agents by leveraging model cards that summarize each model’s domain, task specialization, and other characteristics, effectively filtering out unnecessary agents and preventing agent explosion.
4. **Edge Sampling and Relevance-aware Bidirectional Message Passing:** We construct edges between the selected agents by evaluating their responses against one another to determine relevance ordering. Directed message passing is then performed from highly relevant agents (source) to less relevant ones (target) to enhance their responses, followed by reverse message passing (target-to-source) to refine the original responses of the more relevant agents, capturing fine-grained 1-to-1 communication.
5. **Graph-based Pooling for Scalable Integration:** The updated responses are aggregated via graph-based pooling (e.g., max or mean pooling) to produce a single, unified answer, minimizing the computational cost associated with token stacking.
6. **Theoretical Generalization of MoA:** GoA serves as a more flexible and extensible generalization of multi-agent communication frameworks, mathematically reducing to the MoA framework under specific conditions.

**5. Overview of the Overall Technical Solution**

The overall technical solution of GoA approaches multi-agent LLMs through the lens of a graph framework. Given a query spanning diverse domains, GoA produces an answer through the following pipeline:
1. **Node Sampling:** Each agent is mapped to a model card containing domain and task information. The Meta-LLM, a general-purpose LLM, takes the query Q and the model cards as input and selects the most relevant agents (Top-k), forming an adaptive multi-agent framework.
2. **Edge Sampling:** After collecting initial responses from selected agents, each agent evaluates the relevance of others (excluding itself) to generate a normalized score matrix. Edges are established in Source-to-Target and Target-to-Source directions, while low-relevance nodes are pruned using a threshold τ=0.05.
3. **Message Passing:** GoA first passes messages from source to target nodes, allowing lower-ranked agents to refine responses, then reverses the flow to update source nodes.
4. **Graph Pooling:** With updated responses structured as a graph, GoA outputs the final prediction via max or mean pooling.

**6. Detailed Module Design**

**6.1 Node Sampling Module**
Given a query Q, the goal is to select a subset of agents most relevant to the task. This module leverages publicly available model cards from Hugging Face, which contain useful metadata such as the dataset the LLM was trained on, its specialized domain, model size. Using this information, each agent’s model card is summarized into three key categories: (1) The domain of the LLM, (2) The specialized task, and (3) The model size and special features. Once the summarized model card is obtained, we prompt the Meta-LLM to determine which agents are most likely to generate the most effective response given the query. This approach effectively filters out unnecessary agents, preventing agent explosion while maintaining relevant agents for handling the query.

*Prompt Implementation Details:*
- **Model Card Extraction:** The prompt extracts domain, task specialization, parameter size, and special features from the README file.
- **Initial Response Generation:** Each sampled model generates an initial response in a structured JSON format: `{"reasoning": "<brief reasoning>", "answer": "<answer format hint>", "confidence level": "<a float between 0.0 and 1.0>"}`. A confidence level is introduced to filter out models with limited capability; in such cases, they are replaced with a general-domain model.
- **Node Sampling Prompt:** Selects the top-k models based on model descriptions. Selection criteria (in priority order): 1. Domain match, 2. Task specialization, 3. Include at least one generalist model if applicable, 4. Prefer larger models only when the size gap is significant. Output exactly top-k comma-separated indices.

**6.2 Edge Sampling Module**
Once the subset of selected agents Vs is obtained, we prompt each agent to generate its initial response to the query Q. To model inter-agent relevance, we construct a score matrix in which each agent scores the responses of all other agents (excluding its own to reduce self-bias) based on alignment with Q. These scores are normalized such that each agent distributes a total score of 1.0 across the remaining S−1 agents. We then compute a relevance score for each agent by summing the scores it receives from others. The relevance scores are then used to rank agents and determine their communication roles (e.g., source vs. target). To avoid including weak or noisy responders—particularly in cases where model cards may lack detailed information—we introduce a threshold hyperparameter τ. Agents with relevance scores below τ are pruned from the communication graph, ensuring scalability and better task-fit in the constructed structure. Using the remaining high-relevance agents, we define a weighted directed adjacency matrix to govern message passing.

*Prompt Implementation Details:*
- **Edge Sampling Prompt:** Score the N model responses to the question. Assign a score to each response based on correctness, coherence, and relevance. Scores must sum to exactly 1.0. Output only a comma-separated list of N scores.

**6.3 Message Passing Module**
Given the edge information and weighted adjacency matrix, we proceed with message passing. To incorporate the significance of each agent, GoA performs message passing in two steps:

1. **Source-to-Target:** For highly influential nodes, it is crucial to maintain their initial strength rather than being influenced by less significant or noisy nodes. Conversely, less significant nodes benefit from receiving messages (i.e., more relevant responses to the query) from stronger nodes. Thus, we first propagate information from source nodes (higher-ranked agents) to target nodes (lower-ranked agents), allowing the latter to refine their responses based on more confident initial answers. With these updated responses, we now proceed with the Target-to-Source step.
*Prompt Implementation Details (Source-to-Target):* "Refine your answer by considering other models’ responses. Integrate useful insights from these responses to improve your answer. Be critical — some information may be incorrect." Source responses are annotated with relevance labels derived from normalized edge scores: high relevance (w > 0.7), moderate relevance (0.4 < w ≤ 0.7), or low relevance (w ≤ 0.4).

2. **Target-to-Source:** Since the previous step only updates target nodes, we also allow source nodes to refine their responses based on the improved outputs of their neighbors, rather than the initial responses from target nodes. This enables source nodes to incorporate the consensus of their neighboring agents, indirectly influenced by the original source node, leading to further refinement.
*Prompt Implementation Details (Target-to-Source):* "Other models refined their answers after seeing yours. Use their improvements to finalize your response. Write your final response, incorporating valuable refinements. Be critical — some information may be incorrect."

**6.4 Graph Pooling Module**
To address response integration while minimizing the computational cost associated with token stacking, we formalize response integration through graph pooling. We propose two pooling strategies: 
❶ **Max-Pooling:** Relies on the most influential node (i.e., the agent with the highest number of incoming edges, indicating a higher relevance score). Prioritizes the response of the most significant agent.
❷ **Mean-Pooling:** Balances contributions by considering responses from all selected agents but on a reduced scale, unlike MoA, which involves all available agents. Incorporates responses from all selected agents, requiring an additional forward pass through the Meta-LLM. The averaging is performed in a weighted manner using the relevance scores assigned during edge sampling.
*Prompt Implementation Details (Graph-Pooling):* "Synthesize these model responses into one final answer. Produce an accurate, coherent answer integrating the best insights. Be critical — some information may be incorrect."

**6.5 Generalization to MoA**
We demonstrate that GoA generalizes the existing MoA framework. The message-passing and response-updating procedure of MoA can be formulated as Equation 7. Comparing this to Equation 4, we establish Proposition 1: Graph-of-Agents (GoA) reduces to MoA when the node sampling parameter k equals the total number of agents N, the adjacency matrix is fully connected with all edge weights set to 1, i.e., A ∈ RN×N = 1, and a self-loop is included with the initial query (Q) at each layer, ultimately aggregated via mean pooling.

**7. All Mathematical Formulas and Symbol Definitions**

**Symbol Definitions:**
- N: Number of LLM agents in the system.
- Q: The given query.
- A: The optimized response / Adjacency matrix (context-dependent in original text, distinguished below).
- G = (V, E): Directed graph representing the multi-agent system.
- V = {v1, . . . , vN}: The set of agent nodes.
- E ⊆ V × V: The directed edges.
- S: The number of selected relevant agents (subset size).
- Vs ⊆ V: The selected subset of agents.
- R = {v1(Q), . . . , vS(Q)}: The response set from selected agents.
- Scorei→j: The score assigned by agent i to agent j.
- Sj: The relevance score for agent j.
- τ: Threshold hyperparameter for pruning weak or noisy responders.
- Nj = {i | (i → j) ∈ E}: The set of neighbors of agent j.
- A ∈ RS×S: Weighted directed adjacency matrix governing message passing.
- Aji: How much influence agent i exerts on agent j when passing messages.
- Rsortedi: Sorted responses, ranked from highly relevant to less relevant based on S.
- R′j: Updated response for target node j.
- R′′i: Refined response for source node i.
- vj(·): The forward pass of LLM j.
- ∥: Concatenation of neighboring responses.

**Mathematical Formulas:**
- **Node Sampling (Eq. 1):** 
Vs = Meta-LLM(Top-k|Q,Model Cards)

- **Relevance Score (Eq. 2):** 
Sj = ∑_{i=1, i≠j}^{S} Scorei→j

- **Adjacency Matrix (Eq. 3):** 
Aji = Si / ∑_{k∈Nj} Sk, where Nj = {i | (i → j) ∈ E}

- **Source-to-Target Message Passing (Eq. 4):** 
R′j = vj(∥_{i<j≤S} Aij Rsortedi), where i < j ≤ S

- **Target-to-Source Message Passing (Eq. 5):** 
R′′i = vi(∥_{i<j≤S} Aji R′j), where i < j ≤ S

- **Graph Pooling (Eq. 6):** 
A = { R′′max-source if max-pooling; Meta-LLM(Average|R′′) if mean-pooling }

- **MoA Formulation (Eq. 7):** 
R′i = vi(∥_{j=1}^{N} Rj +Q)

**8. Algorithm Pseudocode**

The paper does not provide explicit algorithm pseudocode blocks; the operational logic is embedded in the methodology and pipeline descriptions. Based on the strict extraction rules, the operational flow is represented structurally as follows:

**Input:** Query Q, Pool of N LLM Agents with Model Cards, Meta-LLM, Threshold τ, Top-k parameter.
**Output:** Final Answer A.

1. **Node Sampling:**
   - Summarize model cards for all N agents into Domain, Task, Size.
   - Vs = Meta-LLM(Top-k | Q, Model Cards)
   - Collect initial responses R = {v1(Q), . . . , vS(Q)} for agents in Vs.

2. **Edge Sampling:**
   - For each agent i in Vs, prompt to score all other agents j (Scorei→j) based on alignment with Q. Normalize scores per agent to sum to 1.0.
   - Compute relevance score Sj = ∑_{i=1, i≠j}^{S} Scorei→j for all j.
   - Prune agents with Sj < τ.
   - Construct weighted directed adjacency matrix A where Aji = Si / ∑_{k∈Nj} Sk.
   - Rank agents by S to define Source nodes (high rank) and Target nodes (low rank).

3. **Message Passing:**
   - **Source-to-Target:**
     For each target node j:
       R′j = vj(∥_{i<j≤S} Aij Rsortedi)
   - **Target-to-Source:**
     For each source node i:
       R′′i = vi(∥_{i<j≤S} Aji R′j)

4. **Graph Pooling:**
   - IF Max-Pooling:
       A = R′′max-source (Response of the agent with the highest number of source edges)
   - ELSE IF Mean-Pooling:
       A = Meta-LLM(Average | R′′)

5. **Return** A