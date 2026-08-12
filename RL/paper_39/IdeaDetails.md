**1. Research Background and Existing Pain Points**
Large Language Models (LLMs) and Vision-Language Models (VLMs) have dramatically expanded the scope and depth of artificial intelligence capabilities. Among all scenarios, the emergence of LLM-based GUI (graphical user interface) agents, capable of independently perceiving, reasoning, and executing complex tasks on user devices, has aroused particular interest. Given that desktops remain central to intelligence-intensive tasks, developing computer use agents is crucial for fundamentally transforming human-computer interactions and elevating AI capabilities. Despite previous attempts to develop computer use agents, enabling them to operate autonomously over extended periods in real-world scenarios remains a significant challenge due to the following existing pain points:
*   **Inherent Mismatch between Agents and Human-Centric GUIs:** GUIs are inherently designed for human interaction, making the simulation of human actions by GUI agents a non-trivial and cumbersome endeavor. Existing GUI agents face challenges due to their reliance on human-like interactions, while API-based control offers efficiency but introduces implementation complexity and security restrictions.
*   **Limitations of Behavior Cloning (BC):** Current mainstream approaches of behavior cloning, including manual annotation and model distillation, are limited in scalability and effectiveness. Manual annotation, while precise, is prohibitively labor-intensive for complex tasks. Model distillation, on the other hand, is constrained by the performance of the teacher models, limiting overall capability. Both methods typically exhibit poor generalization and limited error recovery abilities.
*   **Challenges in Scaling Reinforcement Learning (RL):** Although RL has shown potential for desktop automation tasks, its practical application remains restricted due to computational complexity and methodological challenges. Complex environments, slow convergence, and known inefficiencies in RL training (such as environmental inefficiency, instability during extended training, entropy collapse, and rising KL divergence) severely limit its large-scale adoption in training computer use agents. Existing RL frameworks rely on synchronous training paradigms, resulting in training inefficiencies. Furthermore, limited environment scalability restricts large-scale training, as current virtual machine setups are CPU-intensive, unstable under high concurrency, suffer from network bottlenecks, and lack native distributed support.

**2. Core Research Motivation and Scientific Questions**
The core research motivation is to advance desktop-level planning, reasoning, and device operation by proposing an end-to-end algorithmic framework designed to address the inherent biases of human-oriented operational paradigms, overcome the critical bottleneck of limited environment scalability that restricts RL-based computer use agent training, and counteract stagnation and convergence issues (specifically entropy collapse and rising KL divergence) in extended RL training. The scientific questions addressed are: How can agents transcend the inherent biases of human-oriented operational paradigms to leverage a more machine-oriented approach for device interaction? How can we build a scalable and stable distributed training infrastructure to support large-scale online RL with thousands of parallel environments? How can we systematically address the challenges of entropy collapse and KL divergence accumulation in extended RL training to ensure sustained performance improvements?

**3. Overall Core Idea and Design Philosophy**
The overall core idea is to propose COMPUTERRL, an end-to-end algorithmic framework that integrates API-based and GUI-based actions with scalable RL training. The design philosophy centers on a shift from human-centric to machine-oriented interaction by introducing an API-GUI paradigm that unifies programmatic API calls and direct GUI interaction. It emphasizes scalability through a distributed RL infrastructure capable of orchestrating thousands of parallel virtual desktop environments based on Docker and gRPC protocols, fully compatible with AgentBench. Furthermore, it introduces a novel training methodology, Entropulse, which strategically alternates between RL and SFT phases to maintain exploratory capacity and ensure continuous performance gains, thereby solving the entropy collapse problem in extended training.

**4. Core Innovation Points**
*   **API-GUI Interaction Paradigm:** A new interaction paradigm that unifies human-like GUI interactions with efficient API invocation. It introduces a large-scale, automatically constructed API ecosystem that enables the agent to transcend the inherent biases of human-oriented operational paradigms, leveraging a more machine-oriented approach for device interaction. This addresses the inherent mismatch between human-designed interfaces and artificial agent capabilities, achieving superior operational efficiency and generalization performance.
*   **Large-Scale Distributed RL Infrastructure:** The establishment of a large-scale, distributed RL infrastructure for computer use agents by reconstructing virtual machine clusters. It achieves unprecedented scalability with thousands of parallel environments and seamless AgentBench compatibility, overcoming the critical bottleneck that has limited RL-based computer use agent training to scale experiments.
*   **Entropulse Training Methodology:** A novel training methodology that systematically addresses the challenges of entropy collapse and KL divergence accumulation in extended RL training through strategic alternation between RL and SFT phases. It restores collapsed entropy through targeted SFT training on diverse rollout data, enabling extended training and sustained performance improvements.
*   **Step-Level Group Relative Policy Optimization (Step-Level GRPO):** The extension of the GRPO algorithm to the step-level, making it more suitable for agent RL training. It groups all steps from the same task and computes the advantage for each step, treating each prompt-response pair as an independent training instance with rewards based on the final trajectory outcome to facilitate effective policy optimization.
*   **Full-Asynchronous RL Framework:** The use of a fully asynchronous RL framework (AgentRL) that decouples training from rollout, addressing the inefficiencies of synchronous training paradigms through resource partitioning, dynamic batch sizing, modular component isolation, and off-policy bias mitigation.

**5. Overview of the Overall Technical Solution**
The COMPUTERRL framework features the API-GUI paradigm, a scalable Ubuntu desktop environment for parallelism, and a fully asynchronous RL framework for efficient training. The overall technical solution includes three stages for training: (1) Behavior Cloning for cold start with trajectories collected from general LLMs; (2) RL with step-level GRPO using verifiable, rule-based rewards; (3) Entropulse, which alternates RL with SFT on correct rollouts to restore entropy and sustain learning. The API-GUI paradigm utilizes an LLM-based automated workflow for API development. The infrastructure utilizes a stable Ubuntu environment for large-scale parallelism through standardized decoupled interfaces, lightweight VM deployment, distributed multi-node clustering, and web-based visualization. The RL framework ensures efficient training through resource partitioning, dynamic batch sizing, modular component isolation, and off-policy bias mitigation.

**6. Detailed Module Design**
*   **General API-GUI Paradigm:**
    *   **LLM-based Automated Workflow for API Development:** Users provide exemplar tasks, and the system autonomously generates API code and test cases through three stages:
        1.  **Requirement Analysis:** The LLM analyzes task instances, extracts essential functionalities, and compares them against existing API interfaces to identify gaps. New interfaces are automatically generated for uncovered functionalities, focusing on general-purpose functions to minimize complexity.
        2.  **API Implementation:** The workflow iterates over each interface definition, implementing API functionalities using designated Python libraries. Error-handling mechanisms and logging are implemented.
        3.  **Test Case Generation:** Verifies API correctness by checking: (1) runtime error-free invocation and (2) correct results across parameter inputs; failed APIs receive error feedback for autonomous correction.
    *   **Action Space Formulation:** Dynamically detects the currently active application to infer potentially relevant APIs, reducing the number of available APIs. Uses Python classes and descriptive docstrings to delineate each operation type, provided to the agent via the system prompt for interaction through Python function calls. The agent's output format is standardized to interleave reasoning and action execution.
*   **Stable Ubuntu Environment for Large-Scale Parallelism:**
    *   **Standardized, Decoupled Interface:** Refactors the environment via AgentBench API, providing a unified interface that decouples environment execution from the computational back-end and enables flexible resource management.
    *   **Lightweight VM Deployment:** Uses qemu-in-docker to deploy containerized Ubuntu VMs with streamlined images that reduce network issues and optimize resource usage, significantly lowering per-instance CPU consumption.
    *   **Distributed Multi-Node Clustering:** Employs gRPC-based communication to link CPU nodes into a distributed cluster with centralized resource allocation and orchestration.
    *   **Web-based Visualization and Monitoring:** A web interface provides real-time visualization of environment statuses, agent states, and resource allocation.
*   **Full-Asynchronous RL Framework for Efficient Training:**
    *   **Resource Partitioning:** Data collection runs on dedicated resources while the trainer streams data from the replay engine, preventing mutual blocking.
    *   **Dynamic Batch Sizing:** The trainer processes incoming data with flexible batch sizes, reducing idle time and improving efficiency.
    *   **Modular Component Isolation:** Actor, reference, and critic modules run independently with dedicated resources, utilizing PyTorch distributed groups and NCCL for efficient parameter sharing.
    *   **Off-policy Bias Mitigation:** Limits the replay buffer size and syncs trajectories after each update, ensuring trajectories remain close to the latest policy.
*   **Behavior Cloning Setup:**
    *   **Trajectory Collection with Multiple LLMs:**
        1.  **Initial Sampling:** Closed-source LLMs sample several trajectories per task independently. Records both interaction trajectories and evaluation function outputs.
        2.  **Outcome Stratification:** Categorizes tasks into Fully Solved (acc = 100%), Partially Solved (0 < acc < 100%), and Unsolved (acc = 0%).
        3.  **Task-Oriented Augmentation with Stratified Sampling:** For partially solved tasks, conducts SFT on the backbone model using initial trajectories, then uses the fine-tuned model to sample additional trajectories. For unsolved tasks, builds a model pool of high-performing models and randomly selects one to determine each action, leveraging inter-model variance.
    *   Aggregates and filters collected data, retaining only successful trajectories (180k+ correct steps) for supervised fine-tuning.
*   **Reinforcement Learning with Verifiable Rewards:**
    *   **Step-Level Group Relative Policy Optimization:** Extends GRPO to the step-level. For each task, the policy samples G trajectories. All steps from the same task are grouped, and the advantage is computed for each step.
    *   **Reward Design:** Employs a rule-based verification function. Successfully solved trajectories receive a reward of 1 for every correctly formatted action that contributes to the solution; failed trajectories or improperly formatted actions receive a reward of 0. Treats each prompt-response pair as an independent training instance with rewards based on the final trajectory outcome.
*   **Entropulse for Scaling RL Training:**
    *   Aggregates and retains all successful rollout trajectories during initial RL training. Randomly selects successful trajectories per unique task to construct a new SFT training set with the attributes: High quality, Diversity, and Computational efficiency.
    *   Conducts SFT on this dataset, which produces notable shifts in policy behavior (increased entropy relative to the original policy, indicating enhanced exploration).
    *   Conducts a second round of RL training based on the enhanced exploration capability to achieve significant performance improvements.

**7. All Mathematical Formulas and Symbol Definitions**
*   **Step-Level Group Relative Policy Optimization (Step-Level GRPO) Loss Function:**
    $$J_{StepGRPO}(\theta) = E_{T \sim P(T), \{\{o_{i,j}\}_{j=1}^{L_i}\}_{i=1}^G \sim \pi_{\theta_{old}}} \left[ \frac{1}{\sum_{i=1}^G L_i} \sum_{i=1}^G \sum_{j=1}^{L_i} \left( \min \left( \frac{\pi_\theta(o_{i,j} | q_{i,j})}{\pi_{\theta_{old}}(o_{i,j} | q_{i,j})} A_{i,j}, \text{clip} \left( \frac{\pi_\theta(o_{i,j} | q_{i,j})}{\pi_{\theta_{old}}(o_{i,j} | q_{i,j})}, 1-\epsilon, 1+\epsilon \right) A_{i,j} \right) - \beta D_{KL}(\pi_\theta \| \pi_{ref}) \right) \right]$$
*   **Advantage Function:**
    $$A_{i,j} = \frac{r_{i,j} - \text{mean}(R)}{\text{std}(R)}, \quad R = \{r_{u,v} | u = 1, \ldots, G, v = 1, \ldots, L_u\}$$
*   **Symbol Definitions:**
    *   $\tau$: A task.
    *   $\pi_\theta$: The policy model.
    *   $\pi_{\theta_{old}}$: The old policy model used for sampling.
    *   $\pi_{ref}$: The reference policy model.
    *   $G$: The number of sampled trajectories per task.
    *   $T_1, T_2, \ldots, T_G$: The sampled trajectories.
    *   $L_i$: The length (number of steps) of the $i$-th trajectory.
    *   $o_{i,j}$: The $j$-th step-level action in the $i$-th trajectory.
    *   $q_{i,j}$: The prompt/context corresponding to action $o_{i,j}$.
    *   $A_{i,j}$: The advantage of the $j$-th step in the $i$-th trajectory.
    *   $r_{i,j}$: The reward for the $j$-th step in the $i$-th trajectory.
    *   $R$: The set of all rewards across all trajectories and steps for the task.
    *   $\epsilon$: The clipping ratio parameter.
    *   $\beta$: The coefficient for the KL divergence penalty.
    *   $D_{KL}$: The Kullback-Leibler divergence.

**8. Algorithm Pseudocode**
The paper describes the COMPUTERRL training approach through a three-stage process logic rather than a formal pseudocode block. The iterative process is detailed as follows:
*   **Step 1: Behavior Cloning for Cold Start**
    *   Collect extensive tasks with corresponding evaluation functions and augment to construct an 8k-task dataset.
    *   Perform Initial Sampling using closed-source LLMs to sample trajectories per task.
    *   Perform Outcome Stratification: categorize tasks into Fully Solved, Partially Solved, and Unsolved.
    *   Perform Task-Oriented Augmentation with Stratified Sampling: For Partially Solved tasks, SFT backbone model on initial trajectories, then sample additional trajectories. For Unsolved tasks, build model pool of high-performing models, randomly select one for each action.
    *   Aggregate and filter interaction data, retaining only successful trajectories (180k+ correct steps).
    *   Employ them for supervised fine-tuning of the model to get the Init SFT Model.
*   **Step 2: First RL Phase & Rollout data Collection**
    *   Train the Init SFT Model using RL with step-level GRPO using verifiable, rule-based rewards.
    *   During RL training, aggregate and retain all successful rollout trajectories from various policies at different training steps.
    *   Process this dataset by randomly selecting successful trajectories per unique task to construct a new SFT training set.
*   **Step 3: Entropulse and Second RL Phase**
    *   Conduct SFT on the constructed dataset (from Step 2) to restore entropy and enhance exploration.
    *   Perform a second round of RL training (RL Phase 2) using step-level GRPO based on the enhanced exploration capability to achieve final performance improvements.