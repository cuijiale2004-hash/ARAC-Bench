1. **Research Background and Existing Pain Points**
Large language model (LLM)-based AI agents have demonstrated remarkable capabilities across diverse tasks such as web navigation and function calling. However, they often fail to generalize when deployed in novel agentic environments. These failures reflect a fundamental mismatch between the agent’s pre-trained knowledge and the specific syntax and dynamics of the deployed environment. Although both mismatches stem from a lack of prior knowledge, they require different adaptation mechanisms. There are two distinct failure modes:
*   **Syntactic Mismatch**: The LLM agent’s prior knowledge does not align with environment-specific information, such as observation structure and the environment’s unique syntax. This causes parsing and context-understanding issues. For example, a pre-trained LLM agent might try `click("Search")` or `target destination`, while a new site exposes `Go` and `dest field`, producing invalid actions and immediate failures.
*   **Semantic Mismatch**: The agent lacks an accurate, environment-specific causal model of state transitions, so it cannot predict the consequences of actions. For example, the agent may expect `click("Go")` to show flight results, but the site instead opens a date-confirmation pop-up; without knowing this transition, the agent cannot form the correct multi-step plan and fails.

Current approaches to this generalization gap are ill-suited for rapid adaptation in novel environments. Many methods require human-annotated or LLM-annotated demonstrations, which can be resource-intensive or rely on the LLM’s prior knowledge about the environment. Explicit world modeling approaches involve a heavyweight pipeline of extensive data collection and fine-tuning a separate, specialized model that is computationally expensive and struggles to generalize without being retrained. Both paradigms present significant overhead, highlighting the need for more efficient strategies that can ground an agent using only the information available at test time.

2. **Core Research Motivation and Scientific Questions**
*   **Core Research Motivation**: To address the two failure modes (syntactic and semantic mismatches) and close the generalization gap, there is a need for annotation-free strategies that operate at deployment time. The motivation is to leverage unsupervised, test-time interaction to adapt agents to novel environment formats and dynamics without supervised trajectories or expensive fine-tuning.
*   **Scientific Questions**: How can we design efficient test-time adaptation strategies that mitigate the systematic mismatch between the agent’s pre-trained knowledge and the deployed environment’s specific syntax and dynamics? How can we parametrically adapt the model’s output distribution to align with local syntactic patterns using only the current context as a self-supervisory signal? How can we proactively discover the causal dynamics of an environment before encountering user tasks to construct an in-context world model using natural language?

3. **Overall Core Idea and Design Philosophy**
The overall core idea is to propose two distinct, annotation-free strategies for adapting LLM agents by leveraging environment-specific information from interaction available during deployment.
1.  **Online Syntactic Alignment (SA)**: A lightweight module that learns a small per-interaction bias (an adaptation vector applied to late features) to quickly align an agent’s output distribution with environment-specific syntax. Observing environment-specific strings lets the adapter steer action generation toward the site’s actual element names.
2.  **Dynamics Grounding (DG)**: A short, persona-guided exploration phase that iteratively collects in-context rules about state transitions and supplies those rules as context for downstream planning, equipping the agent with an in-context world model.

**Design Philosophy**:
*   **Operational Distinction**: Distinguish syntactic and semantic mismatches operationally because they require different adaptation mechanisms (parameteric distribution bias vs. in-context causal rules).
*   **Deployment-Time Constraints**: Operate strictly within realistic deployment scenarios defined by three constraints: (1) No annotated trajectories or offline data; (2) Online, streaming adaptation; (3) Permitted test-time interaction.
*   **Lightweight and Efficient**: SA is highly efficient for real-time use (adding minimal latency), while DG is a one-time, deployment-time investment amortized over all subsequent tasks, avoiding the costly training phase of traditional world models.

4. **Core Innovation Points**
1.  **Problem Formalization of Dual Mismatches**: Identification and formalization of two distinct failure modes—syntactic misunderstanding of environment-specific components and semantic misunderstanding of state-transition dynamics—which require distinct adaptation mechanisms.
2.  **Online Syntactic Alignment via Adaptation Vector**: Introduction of a single lightweight adaptation vector $\delta$ acting as an additive bias to the final hidden representations. This vector is updated online using the language modeling loss of the current input context as a self-supervisory signal and is reset per episode to ensure precise, context-dependent alignment without interference across tasks.
3.  **Dynamics Grounding via Persona-Driven Exploration**: Proposal of a deployment-time pipeline that constructs an in-context world model using natural language. It employs a persona-driven exploration phase to systematically probe and learn the environment’s causal dynamics before task execution, replacing the need for a separate, trained dynamics model.
4.  **Annotation-Free and Efficient Adaptation**: Both strategies require only test-time observations and no trajectory annotations or expensive fine-tuning. SA adds only a 3% latency overhead with a single update step, while DG avoids the costly training and inference overhead of learned world models by using a streamlined exploration and filtering process.
5.  **Self-Improvement Capability**: The dynamics grounding approach enables self-improvement, where using the same model for exploration and dynamics extraction performs as well as using a stronger model, demonstrating robustness to the choice of exploration policy.

5. **Overview of the Overall Technical Solution**
The technical solution is formulated around a test-time LLM input $I$ constructed as:
$I = [p; o; \{a\}_{i=1}^{T-1}]$
where $p$ denotes the task instruction, $o$ denotes the current environment observation, and $\{a\}_{i=1}^{T-1}$ denotes the sequence of past actions taken by the agent up to the current step. For environments with short observations (e.g., tool-use), the model input is constructed using all previous interactions.

The overall solution consists of two parallel strategies:
*   **Syntactic Alignment (SA)**: Modifies the model's internal logits by applying an adaptation vector to the final hidden layer. This vector is iteratively updated via gradient descent on the cross-entropy loss of the current input sequence, biasing the output distribution towards tokens observed in the current environment context.
*   **Dynamics Grounding (DG)**: Modifies the input context by augmenting it with a set of discovered environmental dynamics $E_{\text{clean}}$. This set is generated via a four-step pipeline: (1) Persona/Exploratory Goal Synthesis, (2) Exploration and on-the-fly Dynamics Extraction, (3) Filtering and Consolidation, and (4) Context Augmentation during task execution.

6. **Detailed Module Design**

**6.1 Syntactic Alignment (SA) Module**
*   **Objective**: Parametrically adapt the model’s output distribution at test time by treating the current context as a self-supervisory signal to align with local syntactic patterns.
*   **Mechanism**:
    1.  **Initialization**: At the episode’s start, the adaptation vector is initialized as a zero vector, $\delta \leftarrow 0$.
    2.  **Bias Application**: The adapted logits are computed by adding the adaptation vector $\delta \in \mathbb{R}^d$ (where $d$ is the hidden dimension) to the final hidden representations $H \in \mathbb{R}^{n \times d}$ before the final projection layer.
    3.  **Adaptation Update**: At each turn, $\delta$ is updated by performing one gradient descent step to minimize the language modeling loss (cross-entropy loss) of the current input context. Because environment-specific strings are present in the input context, the gradient update to $\delta$ will shift its values to favor their tokens.
    4.  **Correction**: In the subsequent generation step, the adapted logits assign a higher probability to the syntactically correct action.
    5.  **Reset**: To prevent catastrophic forgetting and ensure adaptation is specific to the current environment, $\delta$ is reset to zero at the beginning of each new episode.

**6.2 Dynamics Grounding (DG) Module**
*   **Objective**: Proactively discover the causal dynamics of an environment before the agent encounters any user tasks during deployment, constructing an in-context world model using natural language.
*   **Four-Step Pipeline**:
    1.  **Persona/Exploratory Goal Synthesis**: An LLM is prompted with a high-level description of the environment to generate a set of $N$ diverse, exploratory "personas" or "goals". Personas frame open-ended tasks designed to probe for non-obvious interactions, guiding the agent to explore complex, multi-step interactions that a naive exploration strategy might miss.
    2.  **Exploration and on-the-fly Dynamics Extraction**: For each persona, an LLM agent interacts with the environment, instructed to take novel actions to maximize discovery of new state transitions. After each action, the transition $(observation, action, new\_observation)$ is summarized into a concise human-readable rule $e$. The list of generated rules $\{e\}_{i=1}^T$ is appended to the agent to encourage taking actions that have not been taken before.
    3.  **Filtering and Consolidation**: The set of all extracted dynamics $E = \{e\}_{i=1}^M$ is passed to a reasoning model to filter out trivial or repetitive rules, producing a final, clean set of dynamics $E_{\text{clean}}$.
        *   *Removal Criteria*: Trivial dynamics (e.g., "scrolling reveals content"), simple expansion/collapse of text without new interface elements, navigation within the same page without substantial change, and repetitive entries.
        *   *Retention Criteria*: Unexpected state changes, actions resulting in no change when expected, appearance of new controls/filters, navigation to a new page, and changes that enable new actions.
    4.  **Context Augmentation**: During task execution at test time, the input context $I$ is augmented with discovered dynamics: $I' = [I; E_{\text{clean}}]$. This explicit knowledge guides the agent to make more informed and transition-aware decisions.

7. **All Mathematical Formulas and Symbol Definitions**

*   **Input Formulation**:
    $I = [p; o; \{a\}_{i=1}^{T-1}]$
    Where:
    *   $I$: Test-time LLM input.
    *   $p$: Task instruction (rules, task description).
    *   $o$: Current environment observation (accessibility tree, function call response).
    *   $\{a\}_{i=1}^{T-1}$: Sequence of past actions taken by the LLM-based agent up to the current step.

*   **Adapted Logits Computation**:
    $\text{logits}' = (H + \delta)W_{\text{LM}}^T$
    Where:
    *   $\text{logits}'$: The adapted logits.
    *   $H \in \mathbb{R}^{n \times d}$: The final hidden representations of the language model.
    *   $\delta \in \mathbb{R}^d$: The adaptation vector (additive bias), where $d$ is the hidden dimension of the language model.
    *   $W_{\text{LM}} \in \mathbb{R}^{|V| \times d}$: The model’s output projection matrix.
    *   $|V|$: The vocabulary size.

*   **Adaptation Vector Update Rule**:
    $\delta_{\text{new}} \leftarrow \delta_{\text{old}} - \eta \nabla_{\delta} L_{\text{CE}} (f_{\theta,\delta}(I_{1:n-1}), I_{2:n})$
    Where:
    *   $\delta_{\text{new}}$: The updated adaptation vector.
    *   $\delta_{\text{old}}$: The previous adaptation vector.
    *   $\eta$: The learning rate.
    *   $f_{\theta,\delta}$: The LLM parameterized by its fixed weights $\theta$ and the adaptable vector $\delta$.
    *   $I_{1:n-1}$: The input subsequence consisting of tokens 1 through $n-1$.
    *   $I_{2:n}$: The corresponding next-token targets obtained by shifting the sequence by one position.
    *   $L_{\text{CE}}$: The cross-entropy loss, computed by predicting each token in $I_{2:n}$ from its preceding context $I_{1:n-1}$.

*   **Context Augmentation with Dynamics**:
    $I' = [I; E_{\text{clean}}]$
    Where:
    *   $I'$: The augmented input context for task execution.
    *   $I$: The original input context.
    *   $E_{\text{clean}}$: The final, clean set of environment dynamics extracted and filtered during the exploration phase.

8. **Algorithm Pseudocode**

**Algorithm 1: Syntactic Alignment (SA)**
```
Input: LLM parameterized by fixed weights theta, output projection W_LM, learning rate eta
For each episode:
    Initialize adaptation vector delta <- 0
    For each step t = 1, 2, ... T:
        Construct input context I = [p; o; {a}_{i=1}^{t-1}]
        Compute final hidden representations H
        Compute adapted logits: logits' = (H + delta) * W_LM^T
        Generate action a_t based on logits'
        Execute action a_t in environment, receive new observation o
        Compute gradient update:
            delta_new <- delta_old - eta * grad_delta L_CE(f_{theta,delta}(I_{1:n-1}), I_{2:n})
        Update delta <- delta_new
```

**Algorithm 2: Dynamics Grounding (DG)**
```
Input: Environment description, Number of personas N
// Phase 1: Persona Synthesis
Generate N diverse exploratory personas or goals based on environment description.

// Phase 2: Exploration and On-the-fly Dynamics Extraction
Initialize dynamics list E = []
For each persona:
    Initialize exploration agent
    While exploration budget not exhausted:
        Take novel action a to maximize discovery of new state transitions
        Observe transition (initial_state, action, new_state)
        Extract concise human-readable rule e from transition
        Append e to E
        Append {e} to agent context to encourage unseen actions

// Phase 3: Filtering and Consolidation
Filter E to remove trivial (e.g., scrolling, pagination) and repetitive rules.
Output cleaned dynamics list E_clean.

// Phase 4: Task Execution
For each task instruction p:
    Construct original input I = [p; o; {a}_{i=1}^{T-1}]
    Augment input context: I' = [I; E_clean]
    Execute task using LLM agent with augmented context I'
```