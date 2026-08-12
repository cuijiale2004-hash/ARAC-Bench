Based on the provided academic paper "DON’T JUST FINE-TUNE THE AGENT, TUNE THE ENVIRONMENT", here is the complete and standard extraction of the research idea and implementation plan:

### 1. Research Background and Existing Pain Points
The development of Large Language Model (LLM) agents capable of complex, multi-turn tool use faces significant bottlenecks compared to single-turn tasks. The transition to multi-turn scenarios introduces three key challenges:
*   **Data Scarcity (C1):** High-quality multi-turn tool-use datasets are exceedingly scarce due to the labor-intensive nature of human annotation and validation. For example, the BFCL V3 multi-turn dataset contains only 800 samples, severely limiting the effectiveness of traditional data-driven approaches.
*   **Complex Environment (C2):** Multi-turn scenarios require agents to navigate interconnected tool ecosystems spanning multiple domains. Agents must orchestrate cross-domain API calls and understand dependencies.
*   **Long Interaction Chain (C3):** Success demands consistent performance across all turns; a single failure leads to complete task failure.
Existing training paradigms suffer from distinct pain points:
*   **Supervised Fine-Tuning (SFT):** While constructing synthetic trajectories for SFT mitigates data scarcity, agents trained on static, synthetic traces fail to generalize to real-world scenarios (performance collapse in OOD) due to overfitting.
*   **Reinforcement Learning (RL):** RL offers a natural alternative for generalization through online interaction, but it suffers from a critical "cold-start" problem where unproficient agents cannot explore vast action spaces effectively. Furthermore, long interaction chains make training notoriously unstable and prone to performance collapse (gradient explosion).

### 2. Core Research Motivation and Scientific Questions
The central scientific question of this research is: **How can we train a high-quality agent for complex, multi-turn tool use under extreme data scarcity, ensuring both generalization and stability?**
The core motivation is to address the intersection of data scarcity, generalization, and learning stability through a paradigm shift. Instead of relying on supervised fine-tuning on static trajectories (imitation), the research aims to cultivate both generalization and stability through dynamic, environment-based exploration, learning directly from problem instances without relying on pre-collected expert trajectories.

### 3. Overall Core Idea and Design Philosophy
The overall core idea is **ENVIRONMENT TUNING**, a novel training paradigm that shifts the focus from trajectory imitation to environment-driven exploration. The design philosophy is to orchestrate learning through a structured environment and feedback system that transforms sparse, uninformative signals into rich, dense, and actionable learning opportunities. It enables an agent to acquire sophisticated, multi-step behaviors from scratch, proving that robust learning is achievable even in the complete absence of expert demonstrations by tuning the environment rather than just the agent.

### 4. Core Innovation Points
*   **A novel learning paradigm for data-scarce environments:** Shifting from trajectory-based imitation to environment-based exploration, enabling agents to learn multi-turn tool-use capabilities directly from problem instances without expert demonstrations.
*   **A practical structured curriculum:** A four-stage curriculum that progressively builds skills, transitioning from syntactic correctness to full complexity, utilizing a validation-and stability-based transition rule.
*   **Actionable Environment Augmentation:** Modifying the environment’s feedback to provide pedagogical hints that directly inform the agent about dependency relationships and operational rules, turning failed trajectories into constructive learning opportunities.
*   **Fine-Grained Progress Rewards:** Replacing sparse, binary outcomes with a continuous, turn-by-turn measure of task completion to provide a dense and informative learning signal.

### 5. Overview of the Overall Technical Solution
The problem is modeled as a Partially Observable Markov Decision Process (POMDP). An episode corresponds to a complete user task composed of a sequence of user requests. The agent generates actions (Tool Calls or Final Answers) and receives observations. The objective is to learn policy parameters $\theta$ that maximize the expected terminal reward $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta, P}[R_T]$.

The solution utilizes an adapted version of the Group-Relative Policy Optimization (GRPO) algorithm (adapted PPO) enhanced with a decoupled clipping mechanism and a KL-divergence penalty. The training process follows a four-stage curriculum:
1.  **Stage 1:** Mastering syntactic and schematic correctness using task-agnostic rewards.
2.  **Stage 2:** Basic learning on the "Base" data split using Progress Reward and Actionable Environment Augmentation.
3.  **Stage 3:** Advanced learning on the full dataset (Missing Parameters, Missing Functions, Long-Context) using Progress Reward and Augmentation.
4.  **Stage 4:** Alignment with the evaluation environment using Progress Reward but disabling Actionable Environment Augmentation.

Stage transitions occur only when the validation accuracy has plateaued and its gradient norm is stable.

### 6. Detailed Module Design
**Module 1: Structured Curriculum RL Training**
*   **Stage 1 (Syntax):** Isolates foundational skills by deconstructing actions into structural integrity counters: $C_{correct}$ (correct tool/valid args), $C_{error}$ (correct tool/invalid args), $C_{format}$ (format violations).
*   **Stage 2 (Basic Learning):** Focuses on foundational multi-turn capabilities on the Base split. Introduces Progress Reward and Actionable Environment Augmentation for guided learning.
*   **Stage 3 (Advanced Learning):** Introduces the full training dataset (Missing Parameters, Missing Functions, Long-Context). Continues using Progress Reward and Augmentation to handle ambiguity and noisy contexts.
*   **Stage 4 (Alignment):** Disables Actionable Environment Augmentation to align with true evaluation conditions, forcing the agent to generalize its policies and rely on internal reasoning for standard error messages.

**Module 2: Actionable Environment Augmentation**
*   **Mechanism:**
    *   **Discovering inter-tool dependencies via guided discovery:** Embeds dependencies within the environment’s feedback instead of pre-constructed graphs. Example: If an agent books a flight without an airport code, the augmented feedback provides "Invalid airport code[s]:...", suggesting another tool is needed.
    *   **Revealing internal tool constraints with pedagogical hints:** Provides actionable hints that reveal a tool’s specific internal rules and constraints, moving beyond diagnostic errors. Example: Instead of "FileNotFoundError", it returns "Paths are not allowed. Specify only file/directory name...".
*   **Implementation (LLM-driven workflow):**
    *   **Augmentation Generator:** Analyzes error trajectories and produces enhanced feedback. It determines if augmentation is needed (`need_augmentation`), the error category (e.g., `dependency_missing`, `parameter_constraint`), and the augmentation type (e.g., `clarify_constraint`, `suggest_prerequisite`). It follows strict guidelines: clarifying root causes, providing directional guidance without revealing solutions, and maintaining exploration space.
    *   **Constitutional Judge:** Validates augmented feedback against strict criteria to prevent solution leakage and over-constraining. It checks:
        1.  `solution_leakage`: Must be false (no specific solutions).
        2.  `exploration_constraint`: Must be false (not over-constrained).
        3.  `actionability`: Must be true (provides actionable guidance).
        4.  `hint_quality`: Must be true (appropriate specificity).
        Approval logic: `overall_approved = true` IF AND ONLY IF all checks pass.

**Module 3: Fine-Grained Progress Reward**
*   Provides a dense, turn-by-turn learning signal.
*   **Environment State Evaluation:** Evaluates function calls that modify the environment by comparing the current environment state against the ground-truth state.
*   **Execution Result Evaluation:** Evaluates functions whose outcomes are primarily communicated through return values by inspecting the function’s return value against the expected ground-truth result.
*   A turn is successful only if both state and execution are correct.

### 7. All Mathematical Formulas and Symbol Definitions
*   **Stage 1 Counters:**
    *   $C_{correct}$: Number of tool calls with a correct tool name and valid arguments.
    *   $C_{error}$: Number of calls with a correct tool name but invalid arguments.
    *   $C_{format}$: Number of turns violating the required XML-like format.
*   **Stage 1 Rewards:**
    *   $R_{format} = (N - C_{format}) / N$
    *   $R_{tool} = C_{correct} / (C_{correct} + C_{error})$
    *   $R_{Stage1} = I_{tool} \cdot (R_{format} + R_{tool})$
    (Where $N$ is the total number of turns in the episode, and $I_{tool}$ is an indicator that is 1 if the agent attempts at least one tool call and 0 otherwise.)
*   **Progress Reward:**
    *   $r_t = r_t^{state} \cdot r_t^{exec}$
    *   $R_P = \frac{1}{T} \sum_{t=1}^T r_t = \frac{1}{T} \sum_{t=1}^T (r_t^{state} \cdot r_t^{exec})$
    (Where $r_t^{state}$ and $r_t^{exec}$ are binary scores for state and execution result evaluation.)
*   **Loss Function (Adapted PPO/GRPO):**
    *   $L(\theta) = -\mathbb{E}_t [\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon_{low}, 1+\epsilon_{high})\hat{A}_t)] + \beta D_{KL}(\pi_\theta \| \pi_{ref})$
    (Where $r_t(\theta) = \pi_\theta(a_t|o_t) / \pi_{\theta_{old}}(a_t|o_t)$, $\epsilon_{low} = 0.2$, $\epsilon_{high} = 0.28$, $\beta = 0.1$.)
*   **Advantage Estimate (GRPO):**
    *   $\hat{A}_t(\tau) = \frac{R(\tau) - \mu_G}{\sigma_G + \epsilon_A}$
    (Where $R(\tau)$ is the terminal reward of trajectory $\tau$, $\mu_G$ and $\sigma_G$ are the mean and standard deviation of rewards across group $G$, and $\epsilon_A$ is a small constant.)
*   **Objective Function:**
    *   $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta, P}[R_T]$

### 8. Algorithm Pseudocode
The paper describes the process via a structured curriculum and a two-agent augmentation workflow. The algorithmic flow is as follows:

**Algorithm: Environment Tuning Curriculum Training**
1.  **Input:** Base Model $\pi_{ref}$, Training Dataset $D$ (Base, Missing Func, Missing Param, Long Context), Environment $E$.
2.  **Initialize:** $\pi_\theta \leftarrow \pi_{ref}$
3.  **For Stage $s \in \{1, 2, 3, 4\}$ do:**
4.  $\quad$ **If Stage $s == 1$:**
5.  $\quad\quad$ **Set** Data Split $\leftarrow$ Base
6.  $\quad\quad$ **Set** Reward $R \leftarrow R_{Stage1} = I_{tool} \cdot (R_{format} + R_{tool})$
7.  $\quad$ **Else If Stage $s == 2$:**
8.  $\quad\quad$ **Set** Data Split $\leftarrow$ Base
9.  $\quad\quad$ **Set** Reward $R \leftarrow Progress Reward (R_P)$
10. $\quad\quad$ **Set** Environment Mode $\leftarrow$ Augmented Feedback
11. $\quad$ **Else If Stage $s == 3$:**
12. $\quad\quad$ **Set** Data Split $\leftarrow$ Full Dataset (Base, Missing Func, Missing Param, Long Context)
13. $\quad\quad$ **Set** Reward $R \leftarrow Progress Reward (R_P)$
14. $\quad\quad$ **Set** Environment Mode $\leftarrow$ Augmented Feedback
15. $\quad$ **Else If Stage $s == 4$:**
16. $\quad\quad$ **Set** Data Split $\leftarrow$ Full Dataset
17. $\quad\quad$ **Set** Reward $R \leftarrow Progress Reward (R_P)$
18. $\quad\quad$ **Set** Environment Mode $\leftarrow$ Standard Feedback (No Augmentation)
19. $\quad$ **While** Validation Accuracy not plateaued OR Gradient Norm not stable:
20. $\quad\quad$ Sample batch $B$ from Data Split
21. $\quad\quad$ For each prompt $p \in B$, generate trajectories $\tau$ using $\pi_\theta$ in Environment $E$
22. $\quad\quad$ Calculate Advantage $\hat{A}_t(\tau) = \frac{R(\tau) - \mu_G}{\sigma_G + \epsilon_A}$
23. $\quad\quad$ Update $\pi_\theta$ by minimizing $L(\theta)$
24. $\quad$ **End While**
25. **End For**
26. **Return** $\pi_\theta$

**Algorithm: Actionable Environment Augmentation Workflow**
1.  **Input:** Error Trajectory, Original Feedback, Tool Name, Ground Truth
2.  **Augmentation Generator:**
3.  $\quad$ Analyze Error Category (`dependency_missing`, `parameter_constraint`, etc.)
4.  $\quad$ Determine Augmentation Type (`clarify_constraint`, `suggest_prerequisite`, etc.)
5.  $\quad$ Generate `augmented_feedback` (Clarifies root cause, directional guidance, maintains exploration space)
6.  **Constitutional Judge:**
7.  $\quad$ Check `solution_leakage` (Must be false)
8.  $\quad$ Check `exploration_constraint` (Must be false)
9.  $\quad$ Check `actionability` (Must be true)
10. $\quad$ Check `hint_quality` (Must be true)
11. $\quad$ **If** all checks pass $\rightarrow$ `overall_approved = true`
12. $\quad$ **Else** $\rightarrow$ Reject and retry generation (up to 3 attempts)
13. **Return** Approved `augmented_feedback`