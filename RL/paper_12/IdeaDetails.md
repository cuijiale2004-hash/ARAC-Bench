## 1. Research Background and Existing Pain Points

In the pursuit of Artificial General Intelligence (AGI), autonomous AI agents represent a critical milestone, with “Deep Research” emerging as a core paradigm for achieving more generalized capabilities. By leveraging external tools like search engines and web browsers, these agents can autonomously conduct systematic and in-depth analyses to tackle complex, multi-step research tasks through dynamic reasoning and iterative information retrieval. Despite recent advancements across the research community, a considerable performance gap still persists between open-source solutions and proprietary systems (e.g., OpenAI DeepResearch), leading to a bottleneck in democratizing powerful research capabilities.

This performance disparity primarily stems from fundamental challenges in two of the most critical stages for developing powerful agents: **Data** and **Training**.

*   **Data Pain Point: Insufficient diversity and monolithic definitions of uncertainty.** Information-seeking relies on the agent’s ability to leverage existing information and logical relationships to infer or acquire new, reliable knowledge. If the training data lacks a sufficiently broad and complex range of logical structures, the model will struggle to generalize to novel and intricate problems. Existing methodologies often rely on a narrow set of uncertainty definitions, such as obfuscation. Furthermore, recent advancements in data construction typically initiate from a simple “seed” question and progressively expand the graph by employing external tools (an "easy-to-hard" or iterative expansion strategy). A significant drawback is its inherent tendency to produce predominantly tree-like or acyclic logical structures. This approach inherently struggles to capture or generate scenarios involving complex cyclic relationships, feedback loops, or intricate interdependencies that are common in real-world knowledge graphs. A wider variety of uncertainty types is needed to elicit more diverse and sophisticated reasoning behaviors from the base model.
*   **Training Pain Point: Lack of scalable reinforcement learning (RL) training environment.** Creating a scalable and robust RL training environment for agentic systems poses a significant challenge, which typically demands massive rollouts, each potentially involving numerous tool calls. The high cost and engineering complexity of high-concurrency requests to external APIs can lead to practical issues like tool latency, API failures, and inconsistent outputs. These issues would contaminate the training data, degrade the model’s learned policies (often causing "destructive updates"), and severely hinder iteration of RL training algorithms.

## 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:** To significantly advance the capabilities of open-source web agents and bridge the chasm to proprietary agents by overcoming the fundamental challenges in data diversity and RL environmental stability. This is achieved through a complete post-training pipeline encompassing topological data synthesis (SailorFog-QA-V2) and environmental stabilization strategies (Symbiotic Dual-Environment framework).

**Scientific Questions:**
*   **(RQ1) Topological Generalization:** Does training on data derived from dense, cyclic knowledge graphs (SailorFog-QA-V2) yield stronger reasoning capabilities compared to standard linear or tree-like data expansion used by baselines?
*   **(RQ2) Environmental Stability:** Does decoupling training into a dual-environment (Simulation-to-Real) framework effectively mitigate the instability of real-world RL and lead to better convergence?

## 3. Overall Core Idea and Design Philosophy

The overall core idea revolves around two main strategies:
1.  **Topological Strategy for Reasoning:** To resolve the insufficient diversity inherent in linear or tree-like training data, the method constructs densely interconnected knowledge graphs. By specifically sampling trajectories that require resolving cyclic dependencies and traversing critical “cut vertices”, the agent is forced to learn abstract graph-search patterns. This mechanism ensures the model acquires robust reasoning capabilities that generalize beyond simple information retrieval.
2.  **Stabilization Strategy for RL:** To mitigate the “signal-to-noise” ratio issue caused by real-world volatility, the method proposes a Symbiotic Dual-Environment framework. By utilizing a high-fidelity simulator as an algorithmic “Wind Tunnel”, policy learning is decoupled from environmental stochasticity. This allows for high-frequency, stable policy iteration, ensuring that the agent establishes a robust policy before adapting to the noisy, managed real-world interface.

**Design Philosophy:** The choice of the ReAct framework is a deliberate one, rooted in its simplicity and alignment with “The Bitter Lesson”, which posits that general methods leveraging scalable computation ultimately outperform approaches that rely on complex, human-engineered knowledge and intricate designs. Frameworks that require extensive, specialized prompt engineering or possess rigid operational structures risk becoming obsolete as the intrinsic capabilities of models scale.

## 4. Core Innovation Points

1.  **SailorFog-QA-V2 Dataset Construction:** An enhanced dataset built upon a densely interconnected knowledge graph that introduces a wide variety of uncertainties beyond simple obfuscation, fostering more sophisticated reasoning. Unlike conventional “easy-to-hard” expansion, this method actively seeks out and establishes dense connections between nodes, intentionally creating cyclic structures to ensure the resulting graph is a richly interconnected web, reflecting the complex, non-linear nature of real-world knowledge.
2.  **Expanded Definitions of Uncertainty:** Beyond simple obfuscation, the framework introduces a wider array of defined uncertainties: (1) **Semantic Ambiguity**, which deliberately under-specifies key entities; (2) **Distractor Noise**, which injects plausible but factually incorrect distractors into the context; and (3) **Structural Constraints**, which identify critical paths (e.g., bridges or cut vertices) via non-isomorphic node analysis and generate questions requiring traversal of these specific “cut” edges.
3.  **Symbiotic Dual-Environment RL Framework:** A dual-environment approach combining a high-fidelity simulator (based on an offline Wikipedia database) for rapid, low-cost algorithmic iteration (“Wind Tunnel”) with a robust, managed real-world environment for stable final policy training. This integrated within a symbiotic data-policy feedback loop decouples policy learning from environmental stochasticity.
4.  **Tailored Agentic RL Algorithm:** A modified GRPO algorithm employing a token-level policy gradient loss, a leave-one-out strategy for variance reduction, and a conservative strategy for negative samples. This strategy selectively masks the loss for trajectories that exceed the length limit without yielding a final answer (ambiguous learning signals), while assigning negative rewards to trajectories containing explicit format errors or invalid tool calls to actively learn to avoid such behaviors, effectively preventing "format collapse".

## 5. Overview of the Overall Technical Solution

The overall technical solution is a complete post-training pipeline encompassing data construction, Supervised Fine-Tuning (SFT), and Reinforcement Learning (RL). The agent is built upon the ReAct framework and the Qwen3-30B-A3B base model.
1.  **Data Construction:** Generation of the SailorFog-QA-V2 dataset via dense graph construction, random-walk subgraph extraction, and QA generation with enhanced uncertainties.
2.  **SFT Cold Start:** Training the base model on synthetic data from SailorFog-QA-V2 using rejection sampling to provide a robust initial policy before RL.
3.  **Agentic RL:** Training the agent in a dual-environment setup. The high-fidelity simulated environment is used for high-frequency hyperparameter tuning and reward shaping validation. The managed real-world environment (with unified tool execution interface) is used for the final production training run.
4.  **Data Curation Loop:** A dynamic, automated data synthesis and filtering pipeline that synthesizes and filters high-quality data based on training dynamics, integrated into a symbiotic feedback loop with the RL training.

## 6. Detailed Module Design

### 6.1 Agentic Framework (ReAct)
The agent adopts the ReAct framework as the foundation:
$$H_T = (\tau_0, a_0, o_0, \ldots, \tau_i, a_i, o_i, \ldots, \tau_T, a_T)$$
where $\tau_i, a_i, o_i$ represent thought, action, and observation in the i-th iteration, respectively. At step t, the agent’s thought $\tau_t$ and subsequent action $a_t$ are sampled from a policy $\pi$ conditioned on the complete preceding context, defined as $\pi(a_t, \tau_t|H_{t-1})$.
The action space is composed of four primary tools: **search**, **visit**, **Google Scholar**, and **Python interpreter**, along with the terminal action **final answer**.

### 6.2 SailorFog-QA-V2 Data Construction
#### 6.2.1 Graph Construction
Starts with a seed entity and leverages web tools to discover related entities and extract their corresponding information. To achieve comprehensive topological coverage overcoming the limitations of acyclic graphs, the method actively seeks out and establishes more dense connections between nodes, intentionally creating cyclic structures. The system preserves more complete procedural information (e.g., specific search queries used, source URLs) and computes and stores various statistical features for each entity.

#### 6.2.2 Subgraph Extraction
Due to the combinatorial explosion of exhaustive enumeration in the dense V2 graph, a random-walk based approach is adopted for subgraph extraction. This strategy efficiently gathers a sufficient quantity of non-isomorphic (verified by the Weisfeiler-Leman algorithm), connected subgraphs that collectively represent the full spectrum of structural complexities without the prohibitive cost of a brute-force search.

#### 6.2.3 QA Generation
Instead of feeding the subgraph end-to-end to an LLM, the system first analyzes how many non-isomorphic nodes exist in a given topology, so that the QA focus can be evenly distributed across all orbit nodes (nodes occupying different structural roles).
Uncertainties are injected as follows:
*   **Semantic Ambiguity:** Deliberately under-specify key entities (e.g., “the mathematician who generalized this result” instead of the specific name) or dates. Forces the agent to reason over the graph structure to disambiguate entities via context.
*   **Distractor Noise:** Inject plausible but factually incorrect distractors into the context (e.g., a publication year of a related paper by the same author). Requires the agent to perform active cross-verification and contradiction detection.
*   **Structural Constraints:** By analyzing non-isomorphic nodes, identify critical paths (e.g., bridges or cut vertices). Generate questions that require traversing these specific “cut” edges, ensuring the agent engages in global graph exploration rather than local greedy search.

### 6.3 SFT Cold Start
The SFT dataset is constructed entirely from synthetic data derived from the SailorFog-QA-V2 generator. Training trajectories are produced by high-performing, open-source models solving the generated QA tasks, with quality maintained via rejection sampling. The agent is built upon the Qwen3-30B-A3B-Thinking-2507 foundation model, with the context length deliberately extended to 128k.

### 6.4 Agentic Reinforcement Learning
#### 6.4.1 Simulated Environment
A fully controllable simulated environment is constructed utilizing an offline Wikipedia database and a corresponding suite of simulated web tools. The SailorFog-QA-V2 generation pipeline is adapted to operate exclusively on this offline corpus, synthesizing a dedicated, structurally complex training and testing dataset aligned with the simulation’s capabilities. This simulator serves as a “wind tunnel” for algorithmic stability, where algorithmic improvements that stabilize learning in the simulator consistently translate to stability in the real world. A staged training strategy is adopted: high-frequency hyperparameter tuning and reward shaping validation are performed in the simulator, and the final production training run is executed in the managed real-world environment.

#### 6.4.2 Real Environment
To ensure stability against the inherent volatility of external APIs, a robust, unified tool execution interface is architected. This interface includes a scheduling and management layer that orchestrates tool execution and incorporates sophisticated concurrency handling and fault-tolerance mechanisms. These include QPS constraints, result caching, automated timeout-and-retry protocols, service degradation strategies for non-critical failures, and seamless switching to backup data sources. This design abstracts tool invocation into a deterministic and stable interface, insulating the training loop from real-world stochasticity.

#### 6.4.3 Data Curation
A data optimization loop guided by real-time training dynamics is implemented via a fully automated data synthesis and filtering pipeline that dynamically curates the training set. This closed loop between data generation and model training ensures training stability and yields substantial performance gains by continually feeding the model with the most informative trajectories.

#### 6.4.4 RL Algorithm Modifications
The RL algorithm is a tailored adaptation of GRPO. It employs a strictly on-policy training regimen. The training objective is optimized using a token-level policy gradient loss. To reduce variance in the advantage estimation, a leave-one-out strategy is adopted. Furthermore, a conservative strategy for negative samples is employed to mitigate “format collapse”: Selectively mask the loss for trajectories that exceed the length limit without yielding a final answer (ambiguous signals), while assigning negative rewards to trajectories containing explicit format errors or invalid tool calls. Larger batch and group sizes are leveraged instead of dynamic sampling to maintain smaller variance and provide adequate supervision.

## 7. All Mathematical Formulas and Symbol Definitions

**Equation (1): ReAct History**
$$H_T = (\tau_0, a_0, o_0, \ldots, \tau_i, a_i, o_i, \ldots, \tau_T, a_T)$$
*   $\tau_i$: thought in the i-th iteration
*   $a_i$: action in the i-th iteration
*   $o_i$: observation in the i-th iteration
*   $\pi(a_t, \tau_t|H_{t-1})$: policy at step t conditioned on complete preceding context

**Equation (2): GRPO Objective Function**
$$J(\theta) = \mathbb{E}_{(q,y)\sim D,\{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(\cdot|context)}\left[ \frac{3\sum_{i=1}^G |o_i|}{\sum_{i=1}^G \sum_{t=1}^{|o_i|} \min\left( r_{i,t}(\theta)\hat{A}_{i,t}, \text{clip}(r_{i,t}(\theta), 1-\epsilon_{low}, 1+\epsilon_{high})\hat{A}_{i,t} \right)} \right]$$
*(Note: The fraction structure in the original text is represented by the line breaks `3∑G\ni=1 |oi|` over `G∑\ni=1\n|oi|∑\nt=1`. The term `3∑` is preserved exactly as per the source document OCR.)*
*   $(q, y)$: question-answer pair
*   $r_{i,t}(\theta)$: importance sampling ratio
*   $\hat{A}_{i,t}$: estimator of the advantage at time step t

**Equation (3): Importance Sampling Ratio and Advantage Estimator**
$$r_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t} | \text{context})}{\pi_{\theta_{old}}(o_{i,t} | \text{context})}$$
$$\hat{A}_{i,t} = R_i - \text{mean}(\{R_i\}_{i=1}^G)$$
*   $R_i$: Reward for trajectory $i$
*   $G$: Group size

**Equation (4): Pass@1 Metric**
$$\text{pass@1} = \frac{3}{n} \sum_{i=1}^n p_i$$
*   $n$: number of responses
*   $p_i$: correctness of the i-th response

## 8. Algorithm Pseudocode

The paper does not provide explicit structured algorithm pseudocode blocks (e.g., `Algorithm 1`). However, it details the algorithmic logic and flowchart structure (Figure 2) as follows:

**Logic Flow of the Reinforcement Learning Framework:**
1.  **Input:** Question-Answer pairs $(q, y) \sim D$.
2.  **Rollout Generation:**
    *   Sample trajectories $\{o_i\}_{i=1}^G$ using the current old policy $\pi_{\theta_{old}}(\cdot|context)$.
    *   Actions are executed via **Async Call** to either the **Simulated Environment** or **Real Environment**.
3.  **Observation & Reward:**
    *   Collect Observations via **Async Rollout Service**.
    *   Compute Rewards via **Reward Service**.
4.  **Data Processing (Preservation & Filtering):**
    *   **Preservation:** Store trajectory data.
    *   **Update & Utilization:** Apply conservative strategy: Mask loss for length-exceeded trajectories without final answer; assign negative rewards for format errors.
5.  **Policy Update:**
    *   Compute Advantage $\hat{A}_{i,t}$ using Leave-One-Out strategy.
    *   Calculate token-level policy gradient loss using Equation (2).
    *   Update policy $\pi_\theta$.
6.  **Data Curation Loop:** Dynamically update training data distribution based on **Automatic Synthetic Data** pipeline reflecting current training dynamics.