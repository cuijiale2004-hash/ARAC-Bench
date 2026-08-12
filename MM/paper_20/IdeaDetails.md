## 1. Research Background and Existing Pain Points

**Research Background:**
Several breakthroughs in artificial intelligence can be viewed through the lens of search guided by verifiers—functions assigning rewards to agent behavior aligned with desired criteria. Notable examples include seminal work in Go and Chess, where search and learning are guided by 0/1 rewards tied to the game’s final outcome, and recent advancements in large reasoning models (LRMs), leveraging formal verifiers in code and math. However, while domains such as math, code, and games benefit from relatively well-defined criteria to evaluate agent behavior, this clarity diminishes in open-ended settings (e.g., computer use, web navigation). Evaluation in such scenarios often requires nuanced criteria and reasoning over possibly long sequences of multimodal inputs. Although humans can often recognize satisfactory outcomes, formalizing this intuition into precise and scalable rules remains a challenge. Multimodal Large Language Models (MLLMs) emerge as a promising solution to bridge this gap, given vast world knowledge, human-preference alignment, and large context windows, holding the potential to serve as general-purpose verifiers.

**Existing Pain Points:**
There is a critical limitation in using MLLMs as verifiers: a strong tendency for MLLMs to over-validate agent behavior—a phenomenon termed **agreement bias**. MLLM verifiers validate flawed behavior and even generate chains-of-thought (CoT) to rationalize incorrect judgments. This bias limits MLLMs’ ability to fulfill a core function of a verifier—identifying flawed behavior and providing feedback to improve performance—posing risks to methods that rely on MLLM-based evaluations. Specifically, this bias is pervasive across models (including LRMs), resilient to test-time scaling techniques (CoT, majority voting, Set-of-Marks), and holds for both weak and strong agents. It is exacerbated when verification is treated as a binary success-or-failure classification and increases with the capability gap between agents and verifiers. Furthermore, this bias occurs despite MLLMs exhibiting human-aligned priors about desired behavior, suggesting a bottleneck in knowledge extraction—potentially rooted in fundamental limitations in pretraining and RLHF.

## 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:**
The core motivation is to mitigate the agreement bias in MLLMs when they act as verifiers for agent behavior in open-ended tasks. Current MLLM verifiers fail precisely when most needed—when agent behavior is flawed and requires correction. This failure degrades the quality of MLLM-based rewards and negatively impacts downstream applications such as benchmarking agent performance, behavior cloning, self-improvement, and online supervision. Since modifying pretraining corpora or retraining large models is costly and unclear, there is a strong need for a lightweight, effective inference-time method that leverages MLLMs' existing knowledge and alignment to produce accurate, balanced verification.

**Scientific Questions:**
1. How to effectively extract and utilize the human-aligned priors about desired behavior that MLLMs already possess, rather than having them rationalize flawed trajectories?
2. How to design a verification mechanism that modulates (un)conditional generation to prevent the model from anchoring its judgment on the provided trajectory data, thereby mitigating agreement bias?
3. How does improving the accuracy of failure detection (True Negative Rate) in MLLM verifiers impact downstream applications like self-improvement and online supervision?

## 3. Overall Core Idea and Design Philosophy

**Overall Core Idea:**
The paper introduces **Self-Grounded Verification (SGV)**, a lightweight method that harnesses MLLMs’ own sampling mechanisms by modulating (un)conditional generation to better leverage their knowledge, alignment, and reasoning. SGV operates in two steps:
1.  **First Step (Prior Extraction):** The MLLM is elicited to generate broad priors about desired behavior, conditioned only on the necessary context to frame the task (independent of the data under evaluation).
2.  **Second Step (Grounded Verification):** Conditioned on the self-generated priors from the first step, the MLLM reasons over and evaluates a candidate trajectory.

**Design Philosophy:**
The philosophy is that MLLMs hold the necessary knowledge for verification but fail to apply it when conditioned directly on the trajectory due to agreement bias (a tendency to conflate truthfulness with the information presented in the context, potentially rooted in RLHF limitations). By conditioning only on essential information in the first step, the model is encouraged to explore its probability distribution freely, extracting knowledge pertinent to the task and independent of the data under evaluation. In the second step, the MLLM evaluates a candidate trajectory by sampling from a conditional distribution induced by its own priors, leading to more balanced distributions and more accurate verification. This approach treats the MLLM as a source of domain knowledge first, and then as a verifier, grounding the latter in the former.

## 4. Core Innovation Points

1.  **Identification of Agreement Bias:** The paper systematically identifies and characterizes "agreement bias" in MLLM verifiers—a pervasive tendency to over-validate agent behavior and rationalize flawed trajectories. This bias is shown to be resilient to major test-time scaling techniques (CoT, majority voting) and persists across distinct model families and sizes.
2.  **Self-Grounded Verification (SGV) Method:** A novel two-step inference method that modulates (un)conditional generation to mitigate agreement bias. By separating prior generation from trajectory evaluation, SGV prevents the model from being anchored by the trajectory under evaluation.
3.  **Guidance on Verifier Evaluation Metrics:** The paper elucidates the importance of fine-grained metrics for evaluating verifiers (e.g., TNR, TPR, Bias, dSkew) rather than aggregate metrics like Accuracy, demonstrating that high Accuracy can mask a 50% failure detection rate, which harms downstream applications.
4.  **Downstream Application Improvements:** Demonstrating that mitigating agreement bias via SGV significantly improves downstream applications: Self-Improvement (Reflexion) and Online Supervision. SGV enables non-reasoning models to match reasoning counterparts and sets a new state-of-the-art on VisualWebArena.
5.  **Updated Benchmark Release:** Release of an updated VisualWebArena featuring more human-aligned evaluators, high-fidelity environment parallelism, runtime speedups, and VisualWebArena-Lite.

## 5. Overview of the Overall Technical Solution

The overall technical solution involves redefining the verification process from a single-step conditional generation to a two-step self-grounded generation process.

**Traditional Verifier Setup:**
An agent executes a policy $\pi$ to complete a task $q$, producing a trajectory $\tau_{r:t}$. A standard MLLM verifier $r_{MLLM}$ is typically implemented by prompting the MLLM with the task $q$, trajectory $\tau_t$, and context $C$ (rubrics, instructions), and mapping the output $y$ to a reward:
$r_{MLLM}(q, \tau_t, C) = h\left(\prod_{i=1}^n P(y_i | y_{<i}, q, \tau_t, C)\right)$
This setup suffers from agreement bias.

**SGV Technical Solution:**
1.  **First Step:** Elicit broad priors $\hat{k}_q$ about successful completion of task $q$, conditioned only on the data needed to frame the task (e.g., initial screenshot $s_{0:t}$), excluding the trajectory $\tau_t$.
2.  **Second Step:** Evaluate the trajectory $\tau_t$ conditioned on the self-generated priors $\hat{k}_q$.

This solution applies across offline evaluation, Reflexion (Self-Improvement), and Online Supervision. In Reflexion, the SGV verifier evaluates the trajectory and provides feedback for the agent to reflect upon. In Online Supervision, the verifier monitors progress and provides natural language feedback to steer the agent (e.g., triggering backtracking).

## 6. Detailed Module Design

**Module 1: Agent and Environment Setup**
*   **Agents:** Policy $\pi$ maps history $s_{r:t}$ and task $q$ to action $a_t$.
*   **Trajectory:** Defined as $\tau_{r:t} \equiv (s_r, a_r, \ldots, s_t, a_t)$.
*   **Multimodal Verifier:** Defined as a function $r: T \times Q \to \mathbb{R} \times (V \cup \{\emptyset\})$ mapping trajectory-task pairs to rewards (real-valued score + optional multimodal outputs).

**Module 2: Self-Grounded Verification (SGV) Mechanism**
SGV operates in two distinct phases:
*   **Phase 1: Unconditional Prior Generation.**
    *   Input: Task objective $q$, Context $C$, Initial states $s_{0:t}$ (e.g., initial screenshot) to frame the task.
    *   Output: Priors $\hat{k}_q$ describing how tasks like this are typically accomplished.
    *   Mechanism: The MLLM samples completions independently of the agent's trajectory.
    *   Formula: $\hat{k}_q = g\left(\prod_{i=1}^n P(y_i | y_{<i}, s_{0:t}, C, q)\right)$, where $g$ is a selection function.
*   **Phase 2: Conditional Trajectory Evaluation.**
    *   Input: Task $q$, Trajectory $\tau_t$, Context $C$, and the Priors $\hat{k}_q$ from Phase 1.
    *   Output: Evaluation (SUCCESS/PARTIAL SUCCESS/FAILURE) and Feedback.
    *   Mechanism: The MLLM evaluates the trajectory grounded by the priors generated in the first step.
    *   Formula: $r_{SGV}(\tau_t, C, q) = h\left(\prod_{i=1}^n P(y_i | y_{<i}, q, \tau_t, C, \hat{k}_q)\right)$, where $h$ maps outputs to rewards.

**Module 3: Downstream Application Integration**
*   **Self-Improvement (Reflexion):** After an episode, the SGV verifier evaluates the trajectory. If flawed, feedback is generated. The agent reflects on this feedback, saving it to memory for subsequent attempts.
*   **Online Supervision:** The SGV verifier monitors the agent's progress (e.g., at every 5 steps or upon STOP actions). If it deems the trajectory off-track (FAILURE), it generates feedback appended to the agent's context to trigger backtracking or replanning.

## 7. All Mathematical Formulas and Symbol Definitions

**Symbol Definitions:**
*   $q \in Q$: The task to be completed.
*   $s_r, \ldots, s_t$: Sequence of environment states.
*   $a_t$: Action executed by the agent at time $t$.
*   $\pi$: Agent policy.
*   $\tau_{r:t} \equiv (s_r, a_r, \ldots, s_t, a_t)$: Trajectory consisting of states and actions.
*   $r$: Multimodal verifier function, $r: T \times Q \to \mathbb{R} \times (V \cup \{\emptyset\})$.
*   $C$: Context provided to the verifier (rubrics, instructions).
*   $y_1, \ldots, y_n$: MLLM outputs.
*   $h$: Function mapping MLLM outputs to rewards.
*   $g$: Function selecting a completion in the first step of SGV.
*   $\hat{k}_q$: Broad priors about desired behavior generated in the first step.
*   $s_{0:t}$: States needed to define the task (e.g., initial screenshot).

**Formulas:**
*   **Agent Action Selection:**
    $$a_t = \pi(s_{r:t}, q)$$
*   **Standard MLLM Verifier:**
    $$r_{MLLM}(q, \tau_t, C) = h\left(\prod_{i=1}^n P(y_i | y_{<i}, q, \tau_t, C)\right)$$
*   **SGV First Step (Prior Generation):**
    $$\hat{k}_q = g\left(\prod_{i=1}^n P(y_i | y_{<i}, s_{0:t}, C, q)\right)$$
*   **SGV Second Step (Grounded Verification):**
    $$r_{SGV}(\tau_t, C, q) = h\left(\prod_{i=1}^n P(y_i | y_{<i}, q, \tau_t, C, \hat{k}_q)\right)$$
*   **Verification Evaluation Metrics:**
    *   **Bias:**
        $$\text{bias} = \frac{1}{n}\sum_i E[d_i]$$
        where $d_i = \hat{r}_i - r^*_i$ ($\hat{r}_i$ is MLLM reward, $r^*_i$ is oracle/human reward).
    *   **Distance Skewness:**
        $$\text{dSkew} = 1 - \frac{\sum_{ij}\|d_i - d_j\|}{\sum_{ij}\|d_i + d_j\|}$$
    *   **True Positive/Negative Rate:**
        $$T?R(c) = \frac{\sum_i \mathbf{1}(\hat{r}_i = c \land r^*_i = c)}{\sum_i \mathbf{1}(r^*_i = c)} \approx \hat{P}(\hat{r}_i = c | r^*_i = c), \quad c \in \{0, 1\}$$
    *   **Accuracy:**
        $$ACC = (1 - SR) \cdot TNR + SR \cdot TPR$$
        where $SR$ is the success rate.

## 8. Algorithm Pseudocode

The paper describes the algorithmic flow through the two steps of SGV. Below is the extraction of the process:

**Algorithm: Self-Grounded Verification (SGV)**

**Input:** Task objective $q$, Initial context states $s_{0:t}$, Verifier context $C$, Agent trajectory $\tau_{r:t}$.

**Output:** Reward $r$ (Score and Feedback).

**Step 1: Generate Priors**
1.  Construct First Step Prompt containing $q$, $s_{0:t}$, and $C$.
2.  Sample from MLLM distribution to generate broad priors about task completion:
    $$\hat{k}_q = g\left(\prod_{i=1}^n P(y_i | y_{<i}, s_{0:t}, C, q)\right)$$
3.  (Note: The prompt instructs the model to provide a "Description of how tasks such as this are typically accomplished", independent of the trajectory $\tau_t$.)

**Step 2: Evaluate Trajectory**
1.  Construct Second Step Prompt containing $q$, $\tau_t$, $C$, and the generated priors $\hat{k}_q$.
2.  Sample from MLLM distribution to evaluate the trajectory grounded on priors:
    $$r_{SGV}(\tau_t, C, q) = h\left(\prod_{i=1}^n P(y_i | y_{<i}, q, \tau_t, C, \hat{k}_q)\right)$$
3.  Extract Reasoning, Evaluation (SUCCESS/PARTIAL SUCCESS/FAILURE), and Feedback from the MLLM output using function $h$.
4.  Return reward $r$.