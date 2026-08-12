1. Research Background and Existing Pain Points
Multi-agent coordination is a fundamental topic in multi-agent systems, with applications including warehouse automation, autonomous driving, traffic management, and manufacturing process control. These settings involve complex interactions and dependencies among agents. Multi-Agent Path Finding (MAPF) is a representative multi-agent coordination problem, where multiple agents are required to navigate to their respective goals without collisions. Optimally solving MAPF is known to be NP-hard, leading to the adoption of learning-based approaches to alleviate the online computational burden. Prevailing approaches, such as Graph Neural Networks (GNNs) and transformers, are typically constrained to pairwise message passing between agents. However, this limitation leads to suboptimal behaviors and critical issues:
1. Attention Dilution in Dense Scenarios: In multi-agent systems, agent interactions tend to be uneven. Methods like MAGAT use an attentional mechanism where attention scores are computed pairwise and normalized over the neighborhood via softmax. In dense scenarios, each agent will have a dense neighborhood with only a few truly relevant agents. Since attention scores are first computed pairwise and then normalized, the presence of many irrelevant agents dilutes the attention scores of the relevant agents.
2. Inability to Capture Group Interactions: The interactions in GNNs are inherently pairwise, while multi-agent problems like MAPF tend to be fundamentally group planning problems. Optimality and completeness can generally only be achieved when modelling the full joint state space of all agents, not just two. Models can assuage this by using multiple GNN layers where deeper layers can theoretically capture group interactions, but without an explicit bias for these group interactions, the model likely struggles to learn them.

2. Core Research Motivation and Scientific Questions
The core research motivation stems from the insufficiency of pairwise interactions to fully capture highly-coupled multi-agent dynamics, where interactions may involve multiple agents simultaneously. Higher-order representational structures, such as hypergraphs, can naturally model group interactions. Recent works show hypergraphs can effectively model small-scale multi-agent problems and simple group interactions. 
The scientific question posed is: "Can hypergraphs scale beyond simple group settings to capture the dynamics of complex, highly-coupled multi-agent tasks?"
The work provides a positive answer by focusing on MAPF, arguing that appropriate inductive biases (like hypergraphs) are often more critical than training data size or sheer parameter count for multi-agent problems.

3. Overall Core Idea and Design Philosophy
The overall core idea is to address the representational bottleneck of pairwise message passing by introducing HMAGAT (Hypergraph Multi-Agent Attention Network), a novel architecture that leverages attentional mechanisms over directed hypergraphs to explicitly capture group dynamics. The design philosophy centers on modelling the group interaction of multiple agents influencing a single agent's decision. This is achieved by designing directed hypergraphs with a singleton head and a multi-node tail. By replacing GNN layers with Hypergraph Neural Network (HGNN) layers, the architecture explicitly biases the model towards group interactions, thereby mitigating the attention dilution inherent in pairwise normalization.

4. Core Innovation Points
1. HMAGAT Architecture: A novel imitation learning framework for MAPF, leveraging a hypergraph attention network to better model higher-order group interactions.
2. Dynamic Hypergraph Generation Strategies: Proposing hypergraph generation strategies that dynamically construct directed hypergraphs, including Lloyd-based, k-means-based, and Shortest Distance-based strategies with overlapping "soft" borders.
3. Mitigation of Attention Dilution: Theoretical and empirical demonstration that hypergraph representations mitigate the attention dilution inherent in GNNs. By grouping irrelevant nodes into the tail of a single hyperedge rather than treating them as individual neighbors, HGNNs prevent the dilution of attention scores to relevant nodes.
4. Comprehensive Training Pipeline: A pipeline incorporating an Online Expert with quality improvement, a Post-Training phase, and a Reinforcement Learning-based Temperature Sampling module to dynamically adjust the softmax temperature for each agent.

5. Overview of the Overall Technical Solution
The overall technical solution follows a pipeline consisting of:
1. Expert Data Collection: Using the POGEMA toolkit to generate 21K instances (20% random obstacles, 80% maze-like). Expert trajectories are generated using lacam3 with a staged timeout strategy [1, 5, 15, 60]s.
2. Hypergraph Generation: For each timestep, extracting a directed hypergraph representation based on spatial proximity and coloring or distance metrics.
3. Model Training: Training the HGNN-based model on collected expert trajectories using cross-entropy loss.
4. Post-Training: Applying a post-training phase to improve solution quality by training on a 1:3 ratio of quality improvement instances to pre-collected instances.
5. Temperature Sampling: Training an RL module to dynamically adjust the softmax temperature $\tau \in [0.5, 1.0]$ for each agent.

The model architecture consists of a CNN encoder, followed by HGNN layers, and an MLP decoder. The local observation tensor $o_i$ for agent $i$ at configuration $Q$ is centered around its current position $Q[i]$, with shape $4 \times (2R_{obs} + 1) \times (2R_{obs} + 1)$. The four channels consist of an obstacle map, an agent map, a projection of the goal direction, and a normalized cost-to-go map. For each location $v \in V$ within FOV, the normalized cost-to-go value is computed as $(dist(v, g_i) - dist(Q[i], g_i))/(2R_{obs})$, while setting it to 1 for obstacles.

6. Detailed Module Design
1. Communication Hypergraph: Designed as a directed hypergraph with singleton heads and multi-node tails. Hyperedge features $\omega_{je}$ for each $e \in H$ and $j \in T(e)$ are included. This is a three-dimensional vector consisting of the relative positional co-ordinates and the Manhattan distance between the agent and the hyperedge centre (position of head agent or centroid of head agents).
2. Hypergraph Attention Network (HGNN): Designed in a message-passing manner, with messages going from the tail nodes to the hyperedges to the head nodes. It uses attentional mechanisms to accommodate dynamic and flexible hypergraph structures.
3. Online Expert: On-demand dataset aggregation is applied. After achieving 80% success rate, if the model's solution length exceeds $\delta_{buf} \times$ expert trajectory length (where $\delta_{buf} = 1.2$), instances are extracted every $h=16$ steps. These are solved using lacam3 with timeouts of [1, 2, 10]s, adding trajectories that are $\delta_{buf} \times$ shorter. The total number of quality improvement expert-calls is limited to 30 per online expert phase.
4. Post-Training: Follows the same training procedure but with the quality improvement online expert called for all 500 instances and training consisting of a 1:3 ratio of quality improvement instances to pre-collected instances.
5. Temperature Sampling Module: An actor-critic setup. The actor takes the log-odds predicted by the model, the number of agents in FOV, the number of obstacles in FOV, and the distance and relative positional vector to its goal. It uses a 2-layer MLP with ReLU activations, outputting a scalar $\tau \in [0.5, 1.0]$ via a sigmoid activation and scaling. The critic uses a 2-layer MLP with ReLU and outputs a scalar value by averaging outputs for all agents. Reward is +1 to each agent if all agents reach goals, else -1. Trained using PPO with clip ratio 0.2, discount factor 0.99, GAE $\lambda$ 0.95, learning rate $3\times 10^{-4}$, and batch size 64.
6. Hypergraph Generation Strategies:
   - Lloyd Hypergraphs: Uses Lloyd's algorithm for balanced Voronoi partitioning. Discards half of the least populous colours and reassigns regions to neighboring colours to introduce "soft" borders (allowing multiple colours per vertex).
   - k-means Hypergraphs: Diffuses colours from $k$ randomly sampled vertices over $T$ iterations where each vertex updates its vector to the mean of its neighbours. Applies k-means clustering and "soft boundary operation". Complexity is $O(k|V|)$.
   - Shortest Distance-based Hypergraphs: Non-colouring-based. Two agents $j$ and $k$ are in the tail of a hyperedge with head $i$ if agent $i$ can encounter one agent while visiting the other, formalized using shortest path distances.

7. All Mathematical Formulas and Symbol Definitions
- Directed Hypergraph: $H = (V, E)$. Each hyperedge $e \in E$ is an ordered pair $(T(e), H(e))$, where $T(e) \subseteq V$ is the tail and $H(e) \subseteq V$ is the head. $\Gamma(i) = \{e \in E | i \in T(e) \cup H(e)\}$.
- Normalized cost-to-go: $(dist(v, g_i) - dist(Q[i], g_i))/(2R_{obs})$
- Hyperedge feature: $w_{je} = \phi(\omega_{je})$ processed via MLP $\phi$.
- HGNN Layer Update:
  $x^{(l+1)}_i = \sigma \left( W^{(l)}_R x^{(l)}_i + \sum_{e \in \Gamma(i) \wedge i \in H(e)} \alpha^{(l)}_{ie} (W^{(l)}_h h^{(l)}_e) \right)$
  $\alpha^{(l)}_{ie} = softmax \left[ LeakyReLU \left( (x^{(l)}_i)^\top (\Theta^{(l)}_h h^{(l)}_e) \right) \right]$
  $h^{(l)}_e = \sum_{j \in T(e)} \alpha^{(l)}_{ej} (W^{(l)}_n x^{(l)}_j + W^{(l)}_e w_{je})$ (hyperedge representation)
  $\alpha^{(l)}_{ej} = softmax \left[ LeakyReLU \left( ( \sum_{i \in H(e)} x^{(l)}_i / |H(e)| )^\top (\Theta^{(l)}_n x^{(l)}_j + \Theta^{(l)}_e w_{je}) \right) \right]$
  Where $W^{(l)}_{\{R,n,e,h\}}$ and $\Theta^{(l)}_{\{n,e,h\}}$ are learnable weights, and $\sigma$ is a non-linearity.
- Attention score sum for node $i$ to agent $j$ in layer $l$: $a^{(l)}_{i,j} = \sum_{e \in \Gamma(i) \wedge i \in H(e) \wedge j \in T(e)} \alpha^{(l)}_{ie} \alpha^{(l)}_{ej}$
- Radar Plot Metrics:
  Sol. Quality $= \begin{cases} SoC_{best}/SoC & \text{if solution found} \\ 0 & \text{if no solution found} \end{cases}$
  Scalability $= \begin{cases} runtime(\#agents) / (\#agents \cdot Min. \#agents / runtime(Min. \#agents)) & \text{if solved} \\ 0 & \text{if timed out} \end{cases}$
- Normalized entropy of attention scores: $H = - \sum_{i \in V} \frac{1}{|V|} \left( \sum_{j \in N_i} \alpha_{ij} \log \alpha_{ij} / \log |N_i| \right)$
- Attention Dilution Proof for GNNs: $\alpha'_{ij} = \frac{exp(a'_{ij})}{\sum_{k \in N'_i} exp(a'_{ik})} = \frac{exp(a_{ij})}{\sum_{k \in N_i} exp(a_{ik}) + \sum_{k \in A'} exp(a'_{ik})} < \alpha_{ij}$
- Attention Mitigation Proof for HGNNs: $\alpha'_{ij} = \frac{exp(a'_{ij})}{\sum_{k \in \{e'_\sigma\} \cup E_{rest}} exp(a'_{ik})} = \frac{exp(a_{ij})}{\sum_{k \in \{e_\sigma\} \cup E_{rest}} exp(a_{ik})} = \alpha_{ij}$

8. Algorithm Pseudocode
Algorithm 1 COLOURINGBASEDHYPERGRAPHS
input: colours C, colouring R, agents A, positions Q
1: E ← ∅
2: for v ∈ A, c ∈ C do
3: T ← {u ∈ A | ∥Q[u]−Q[v]∥ ≤ Rcomm ∧ (Q[u], c) ∈ R}
4: if T ̸= ∅ then E ← E ∪ {(T ∪ {v}, {v})}
5: return (A, E) ▷ directed hypergraph

Algorithm 2 SHORTESTDISTANCEBASEDHYPERGRAPHS
input: grid G = (V,E), agents A, agent positions Q
params: communication radius Rcomm, dynamic slack variable ϵ
1: E ← ∅ ▷ set of hyperedges
2: for v ∈ A do
3: H ← {v} ▷ head of hyperedge
4: AC ← {u ∈ A | ∥Q[u]−Q[v]∥ ≤ Rcomm} \ {v} ▷ communicating agents
5: EC ← ∅ ▷ initialize candidate edges
6: for u ∈ AC do
7: for w ∈ AC \ {u} do
8: if dist(Q[v],Q[u]) + dist(Q[u],Q[w]) ≤ max(dist(Q[v],Q[u]), dist(Q[v],Q[w])) + ϵ then
9: EC ← EC ∪ {(u,w), (w, u)}
10: C ← GETCLIQUES(AC , EC )
11: for T ∈ C do
12: E ← E ∪ {(T ∪ {v}, H)}
13: return E

Algorithm 3 GRIDCOLOURING
input: grid G = (V,E)
params: number of initial colours k′, number of final colours k, number of iterations n
1: R′, C′ ← LLOYDSONGRAPH(G, k′, n) ▷ Rigid colouring from Lloyd’s
2: C ← k-most populous colours in R′ ▷ We use k = k′/2
3: R← {(v, c) ∈ R′ | c ∈ C} ▷ Discard least populous colourings
4: U ← V \ {v | ∃c ∈ C.(v, c) ∈ R} ▷ Initial uncoloured vertices
5: while ∃v ∈ V . !∃c ∈ C . (v, c) ∈ R do ▷ While there are uncoloured vertices
6: Rold ← R
7: for u ∈ U do
8: R← R ∪ {(u, c) ∈ V × C | ∃v ∈ neigh(u) . (v, c) ∈ Rold} ▷ Take neighbours’ colours
9: return R,C ▷ Less rigid colouring

Algorithm 4 KMEANSCOLOURING
input: grid G = (V,E)
params: number of initial colours k′, number of final colours k, number of iterations n
1: X ← 0|V |×k′ ▷ Initial colour vectors
2: X ← Randomly assign k′ vertices to unique colours (i.e. set X[v, c] := 1) ▷ Random seeding
3: for i ∈ [1, n] do ▷ Diffuse colours
4: Xold ← X
5: for v ∈ V do
6: X[v]← ∑u∈neigh(v) Xold[u]
7: X[v]← X[v]/∥X[v]∥1 ▷ Mean of neighbours’ colours
8: R′, C′ ← KMEANS(X, k′, n) ▷ Rigid colouring from k-means
9: C ← k-most populous colours in R′ ▷ We use k = k′/2
10: R← {(v, c) ∈ R′ | c ∈ C} ▷ Discard least populous colourings
11: U ← V \ {v | ∃c ∈ C.(v, c) ∈ R} ▷ Initial uncoloured vertices
12: while ∃v ∈ V . !∃c ∈ C . (v, c) ∈ R do ▷ While there are uncoloured vertices
13: Rold ← R
14: for u ∈ U do
15: R← R ∪ {(u, c) ∈ V × C | ∃v ∈ neigh(u) . (v, c) ∈ Rold} ▷ Take neighbours’ colours
16: return R,C ▷ Less rigid colouring