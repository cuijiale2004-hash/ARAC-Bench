### 1. Research Background and Existing Pain Points

**Research Background:**
With the growing adoption of large language model (LLM) agents in persistent real-world roles, they naturally encounter continuous streams of tasks and interactions. The rapid advancement of LLMs has significantly accelerated the development of interactive LLM agents, which are crucial in tackling complex real-world tasks that require multi-turn interactions with environments. These agents have demonstrated great potential across diverse scenarios, including web browsing, computer use, and scientific discovery. As these agents are increasingly deployed in persistent, long-running roles, they naturally encounter a continuous stream of tasks and interactions. Recent efforts on agent memory have primarily focused on storing past interactions for reuse.

**Existing Pain Points:**
A critical limitation persists: agents largely fail to learn from this accumulated experience. By approaching each new task in isolation, they are doomed to:
1.  Repeat similar errors observed in the past.
2.  Discard valuable insights gained from related problems.
3.  Lack self-evolving capabilities that make the agent system more capable over time.
Existing memory approaches are often limited to leveraging raw trajectories (e.g., Synapse) or successful routines (i.e., workflows, procedures) (e.g., AWM). They suffer from two fundamental drawbacks:
1.  They lack the ability to distill higher-level, transferable reasoning patterns.
2.  By over-emphasizing successful experiences, they leave the valuable lessons from an agent’s own failures largely underexplored.
Consequently, existing memory designs often remain limited to passive record-keeping rather than providing actionable, generalizable guidance for future decisions.

### 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:**
The necessity of building memory-aware agent systems that could learn from their past experiences to enable self-evolving capabilities. There is a need to bridge the gap where existing memory designs are passive and lack actionable guidance, specifically by abstracting experiences into reusable reasoning units that generalize not only from successful cases but also by learning from failures to provide richer guidance for test-time learning. Furthermore, there is motivation to study experience scaling to establish a powerful synergy between memory and test-time scaling, positioning memory-driven experience scaling as a new scaling dimension for agents.

**Scientific Questions:**
1.  How to extract and preserve useful memory from past trajectories, particularly distilling high-level, transferable reasoning patterns from both successes and failures?
2.  How to effectively leverage such memory for future queries to avoid redundantly re-discovering already successful strategies or repeating past mistakes?
3.  How to scale experience through depth by tackling each single task with more exploration, and how to leverage the contrastive signals from abundant exploration to synthesize higher-quality memory, creating a synergy between memory and test-time scaling?

### 3. Overall Core Idea and Design Philosophy

**Overall Core Idea:**
REASONINGBANK is a novel memory framework that distills generalizable reasoning strategies from an agent’s self-judged successful and failed experiences. At test time, an agent retrieves relevant memories from REASONINGBANK to inform its interaction and then integrates new learnings back, enabling it to become more capable over time. Building on this powerful experience learner, the work further introduces memory-aware test-time scaling (MATTS), which accelerates and diversifies this learning process by scaling up the agent’s interaction experience. By allocating more compute to each task, the agent generates abundant, diverse experiences that provide rich contrastive signals for synthesizing higher-quality memory. The better memory in turn guides more effective scaling, establishing a powerful synergy between memory and test-time scaling.

**Design Philosophy:**
1.  **Distillation over Raw Storage:** Abstracting away low-level execution details while preserving transferrable reasoning patterns and strategies into structured memory items, rather than storing raw trajectories or success-only workflows.
2.  **Learning from Failure:** Utilizing counterfactual signals and pitfalls from failed trajectories as negative signals to sharpen guardrails, enabling agents to learn from their own failures.
3.  **Memory-Driven Experience Scaling:** Positioning memory-driven experience as a new scaling dimension. High-quality memory steers the scaled exploration toward more promising paths, while the rich experiences generated forge even stronger memories.
4.  **Contrastive Curation:** Leveraging inherent contrastive signals that arise from abundant explorations on the same problem (via parallel self-contrast and sequential self-refinement) to curate more reliable and higher-quality memory.

### 4. Core Innovation Points

1.  **REASONINGBANK Framework:** A novel memory framework that distills generalizable reasoning strategies from both successful and failed experiences, beyond prior work that primarily stores raw trajectories or success-only routines.
2.  **MATTS (Memory-Aware Test-Time Scaling):** A novel integration of test-time scaling with the memory framework that establishes a powerful, bidirectional synergy between memory and test-time scaling, introducing memory-driven experience as a new scaling dimension.
3.  **Closed-Loop Memory Process:** A closed-loop process where the agent leverages past experiences (Memory Retrieval), constructs new memory from current tasks (Memory Extraction), and continually updates its memory (Memory Consolidation), enabling sustained evolvement in test-time learning scenarios.
4.  **Failure Utilization:** The ability to distill reasoning patterns from failed trajectories, supplying counterfactual signals and pitfalls that act as negative signals, which transforms failures into constructive signals rather than noise.
5.  **Contrastive Scaling Strategies:** The design of two complementary instantiations for MATTS—Parallel Scaling (using self-contrast across multiple trajectories to identify consistent patterns and filter spurious solutions) and Sequential Scaling (using self-refinement within a single trajectory to use intermediate notes as valuable signals for memory).

### 5. Overview of the Overall Technical Solution

The overall technical solution focuses on the test-time learning paradigm where a sequence of task queries $Q = \{q_1, q_2, ..., q_N\}$ arrives in a streaming fashion without ground truth. The agent policy $\pi_L(\cdot|M,A)$ is equipped with a memory module M (REASONINGBANK).
The solution proceeds as follows:
1.  **Agent Configuration:** The agent interacts with the environment via the policy $\pi_L$, generating a trajectory of $(o_{0:t}, a_{0:t})$ for $t$ steps.
2.  **Memory Retrieval:** For a new query, the agent queries REASONINGBANK using embedding-based similarity search to identify the top-k relevant experiences. Retrieved items are injected into the agent’s system instruction.
3.  **Interaction:** The agent interacts with the environment to complete the task.
4.  **Memory Extraction:** After completion, proxy correctness signals are obtained using an LLM-as-a-judge. Based on success/failure signals, different extraction strategies are applied: successful experiences contribute validated strategies, while failed ones supply counterfactual pitfalls. Experiences are distilled into structured memory items (Title, Description, Content).
5.  **Memory Consolidation:** New items are incorporated into REASONINGBANK with a simple addition operation.
6.  **MATTS Integration:** To scale experience, MATTS is applied:
    *   **Parallel Scaling:** Generate multiple trajectories for the same query. Compare and contrast (self-contrast) across trajectories to curate reliable memory.
    *   **Sequential Scaling:** Iteratively refine reasoning within a single trajectory (self-refinement). Use intermediate notes as signals for memory.
7.  **Synergy:** High-quality memory guides scaling toward more promising rollouts, while diverse rollouts enrich memory with valuable contrastive signals, creating a virtuous cycle.

### 6. Detailed Module Design

**6.1 Agent Configuration Module**
*   The agent policy is parameterized by the backbone LLM L, conditioned on a memory module M, and the action space A, denoted as $\pi_L$ for short.
*   The transition function of the environment is defined as $T (s_{t+1}|s_t, a_t)$ where $s_t$ is the state and $a_t$ is the action.
*   The agent generates a trajectory of $(o_{0:t}, a_{0:t})$ for $t$ steps, where observation $o_t$ is from the current state $s_t$.
*   The agent needs to generate an action $a_{t+1} \in A$ via $\pi_L(o_{0:t}, a_{0:t};M,A) \rightarrow a_{t+1}$.
*   The memory module M contributes relevant memories as additional system instruction for $\pi_L$.

**6.2 REASONINGBANK Memory Schema Module**
Memory items are designed and induced from past experiences as structured knowledge units that abstract away low-level execution details while preserving transferrable reasoning patterns and strategies. Each memory item specifies three components:
*   **Title:** Serves as a concise identifier summarizing the core strategy or reasoning pattern.
*   **Description:** Provides a brief one-sentence summary of the memory item.
*   **Content:** Records the distilled reasoning steps, decision rationales, or operational insights extracted from past experiences.

**6.3 Memory Retrieval Module**
*   The agent queries REASONINGBANK with the current query context to identify the top-k relevant experiences and their corresponding memory items using embedding-based similarity search.
*   Embedding is performed using gemini-embedding-001. Similarity search is conducted over the memory pool using cosine distance.
*   The selected memory items of the top-k most similar experiences (default k = 1) are concatenated into the agent’s system prompt with a simple formatting template (each item represented by its title and content) and instruction:
    *   "Below are some memory items that I accumulated from past interaction from the environment that may be helpful to solve the task. You can use it when you feel it’s relevant. In each step, please first explicitly discuss if you want to use each memory item or not, and then take action."

**6.4 Memory Extraction Module**
*   **LLM-as-a-Judge for Correctness Signals:** An LLM-based binary classifier is adopted to obtain proxy correctness signals without ground-truth reference. The classifier is prompted with the trajectory and the given user query, and asked to output a categorical judgment (Success or Failure).
*   **Extraction Strategy:**
    *   **Successful Trajectories:** The instruction emphasizes analyzing why the trajectory led to success and summarizing transferable reasoning strategies. Successes provide validated strategies.
    *   **Failed Trajectories:** The instruction requires reflecting on the causes of failure and articulating lessons or preventive strategies. Failures supply counterfactual pitfalls that act as negative signals.
*   **Extraction Pipeline:** An LLM-based extraction pipeline converts raw trajectories into structured memory items (Title, Description, Content). At most 3 memory items could be extracted for each trajectory. The backbone LLM of the extractor is set to the same as the agent system with temperature 1.0. The output format is constrained to ensure concise, non-redundant, and generalizable insights.

**6.5 Memory Consolidation Module**
*   After finishing each new query, the trajectory is processed by the extraction pipeline to produce new memory items, which are appended into the memory pool.
*   A minimal consolidation strategy is adopted: newly generated items are directly added without additional pruning.
*   REASONINGBANK is maintained in a JSON format, and each entry consists of a task query, the original trajectory, and the corresponding memory items. The embedding is pre-computed for each given query and stored in another JSON file.

**6.6 MATTS (Memory-Aware Test-Time Scaling) Module**
*   **Scaling Factor $k$:** Denoting the number of trajectories for parallel scaling and refinement steps for sequential scaling.
*   **Parallel Scaling:** Generates multiple trajectories for the same query under the guidance of retrieved memory items. Uses self-contrast reasoning across different trajectories. The model is instructed to directly compare and contrast trajectories to identify patterns that lead to success and mistakes that cause failure. This identifies consistent reasoning patterns while filtering out spurious solutions.
*   **Sequential Scaling:** Iteratively refines reasoning within a single trajectory after the initial completion (self-refinement). The intermediate notes generated in self-refinement are used as valuable signals for memory, capturing reasoning attempts, corrections, and insights that may not appear in the final solution.
*   **Best-of-N Calculation:** Given the task query and N trajectories, an LLM (same backbone as agent) selects the best answer from the N trajectories based on criteria including Progress Toward Goal, Trajectory Efficiency, Loop Detection, Error Severity and Stability, and Overall Trajectory Quality.

### 7. All Mathematical Formulas and Symbol Definitions

*   $\pi_L(\cdot|M,A)$: Agent policy parameterized by the backbone LLM L, conditioned on a memory module M, and the action space A.
*   $T (s_{t+1}|s_t, a_t)$: Transition function of the environment, where $s_t$ is the state and $a_t$ is the action selected by $\pi_L$ at time $t$.
*   $(o_{0:t}, a_{0:t})$: Trajectory of observations and actions for $t$ steps, where observation $o_t$ is from the current state $s_t$.
*   $a_{t+1} \in A$: The action generated by the agent via $\pi_L(o_{0:t}, a_{0:t};M,A) \rightarrow a_{t+1}$.
*   $Q = \{q_1, q_2, ..., q_N\}$: A sequence of task queries arriving in a streaming fashion.
*   $SR = \frac{1}{N}\sum_{i=1}^N isSuccess(q_i)$: Success Rate, where $isSuccess(q_i)$ is the binary function that returns 1 if task $q_i$ is successful and 0 otherwise.
*   $AS = \frac{1}{N}\sum_{i=1}^N Steps(q_i)$: Average Steps, calculated as the total number of steps taken in the trajectory when solving task $q_i$ divided by the total number of tasks.

### 8. Algorithm Pseudocode

No explicit algorithm pseudocode block is provided in the original text.