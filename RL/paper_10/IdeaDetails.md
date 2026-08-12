1. Research Background and Existing Pain Points
Research Background: Multi-Agent Systems (MAS) offer a powerful paradigm for solving complex problems through distributed reasoning and collaboration. Drawing inspiration from human collaborative intelligence, MAS enables agents to collaborate and take on specialized roles, unlocking new frontiers in artificial intelligence for addressing challenges beyond the reach of individual agents. Significant progress has been made in developing individual agent capabilities, such as reasoning, memory, and tool use. 

Existing Pain Points:
1. Instability of Manual Orchestration: Current approaches often rely on manual orchestration of agent workflows, which is a labor-intensive process requiring deep domain expertise and meticulous prompt engineering. Because LLM-based agents are inherently sensitive to prompt variations, this approach introduces cascading instability: minor performance fluctuations in one agent can easily propagate and amplify instability across the entire system.
2. Infeasibility of Automated Topology Search: The second category employs automated methods to search for optimal interaction topologies. However, due to the combinatorial explosion of possible topologies, such search strategies quickly become infeasible at scale and often fail to discover truly effective collaborative structures.
3. Integration Challenge: While individual agent capabilities have advanced, integrating these components into a cohesive multi-agent system remains a formidable task, hindered by the challenge of optimizing interactions among agents effectively and robustly.

2. Core Research Motivation and Scientific Questions
Core Research Motivation: Rather than manually designing complex agent topologies or searching the vast and fragile space of interaction structures, a more robust path lies in optimizing the collaborative dynamics within a well-designed topology. This shifts the focus from structural search to behavioral optimization within a given structure.

Scientific Questions: The core research question is reframed from "What is the optimal agent topology?" to "Given an effective topology, how can we train agents to cooperate more effectively to maximize system performance?"

3. Overall Core Idea and Design Philosophy
Overall Core Idea: AgentPO is a novel framework that directly optimizes agent collaboration. AgentPO employs reinforcement learning to train a specialized Collaborator agent, which refines its interaction policy to enhance overall system performance within a fixed multi-agent topology. 

Design Philosophy: The architecture is founded on a functional decoupling of collaboration and execution, implemented through two distinct roles: Collaborator Agent and Actor Agent. The Actor, which can be a state-of-the-art model, focuses on task execution, while the Collaborator learns how to help it succeed. Crucially, only the small Collaborator is trained. Using a simple success-or-fail reward signal, it learns to provide effective hints, critiques, or suggestions to its Actor partner. This approach boosts the team’s overall efficiency and adaptability without the heavy computational cost of architectural search or fine-tuning powerful but ultra-large Actor models, offering a scalable and resource-efficient path to building cooperative MAS.

4. Core Innovation Points
1. Framework Innovation: Proposes AgentPO, a novel framework that focuses on optimizing collaboration among agents through reinforcement learning, shifting the paradigm from topology search to collaborative dynamics optimization.
2. Hypothesis Validation: Presents empirical results on complex reasoning tasks, showing that AgentPO achieves significantly enhanced performance, validating the hypothesis that optimizing collaborative dynamics is an effective strategy for improving system-level capabilities.
3. Scalable Methodology: Offers a practical and scalable methodology for enhancing multi-agent systems: by training small, specialized Collaborator within a well-designed topology, AgentPO achieves superior performance at a significantly lower cost (only 500 training samples and 7.8% of EvoAgent’s inference cost).
4. Functional Decoupling: Introduces a functional decoupling of collaboration and execution (Collaborator + Actor), allowing efficient optimization of a lightweight model to guide a frozen, powerful model (or API-based model) without altering foundational capabilities.
5. Hybrid System Enhancement: Demonstrates that a compact, cost-efficient local model can effectively co-pilot a large, inflexible model through targeted guidance, enabling local Collaborators to enhance API-based powerful Actors.

5. Overview of the Overall Technical Solution
The system formalizes the multi-agent collaborative problem-solving task as an optimization problem over a problem distribution. The AgentPO framework operates by creating a partnership between two agent types: a Collaborator Agent (governed by a learnable policy) and an Actor Agent (governed by a fixed policy). For any given problem, the interaction proceeds sequentially. First, the Collaborator Agent generates an auxiliary signal conditioned on a problem-derived context determined by the interaction topology. This signal is then used to form an enriched context for the Actor Agent, which in turn produces the final solution. Finally, the solution is evaluated against the ground-truth to produce a binary reward. This scalar reward provides the learning signal to update the Collaborator's policy via the Group Relative Policy Optimization (GRPO) algorithm. The framework supports two distinct collaboration topologies: the Feed-forward Mode (Hint-Actor) and the Feedback Mode (Critic-Actor).

6. Detailed Module Design
6.1 Agent Topology Modules
The interaction protocol between the Collaborator and Actor is defined by agent topology, specifying the input context for the Collaborator’s policy and the mechanism for integrating the Collaborator’s output signal into the actor’s context.

Feed-forward Mode (Hint-Actor): In this configuration, the Collaborator acts as a Hint Agent, providing proactive guidance. Conditioned solely on the problem, it generates a hint. This hint is then prepended to the problem query to form an augmented context for the Actor Agent, which then produces the final solution. The resulting reward trains the Hint Agent to generate maximally effective guidance. This topology embodies a feed-forward, proactive model of collaboration.
- Hint Agent Prompt: <Instruction> Rewrite the question below to make it easier to understand. <Problem> {{problem}}
- Actor Agent Prompt: <Instruction> Please reason step by step, and put your final answer within \boxed{}. <Problem> {{problem (Hint: hint)}}

Feedback Mode (Critic-Actor): Here, the Collaborator assumes the role of a Critic Agent, facilitating an iterative refinement loop. The sequence begins with the Actor generating an initial draft solution. This draft, along with the problem, forms the context for the Critic to produce a critique. Finally, the Actor conditions on the complete history—the problem, its initial attempt, and the critique, to generate a refined solution. The reward, computed from the refined solution, trains the Critic to provide feedback that most effectively steers the refinement process toward a correct solution. This topology models a reflective, feedback-driven form of collaboration.
- Critic Agent Prompt: <Instruction> Given a question and its current solution, analyze the solution and provide concise, specific feedback identifying any errors, logical gaps, or missing justifications. Do not rewrite the solution, only highlight issues. <Problem> {{problem}} <Current solution> {{current solution}}
- Actor Agent Prompt: <Instruction> You have an opportunity to improve your solution. Please review Current Solution and Comment carefully. Correct errors and fill justification gaps if any. <Problem> {{problem}} <Current solution> {{current solution}} <Comment> {{comment}}

6.2 Agent Role Definitions
- Hint: Provides proactive suggestion or plan to guide the Actor.
- Critic: Evaluates the Actor’s output and offers feedback.
- Actor: The primary reasoning agent that produces the final answer.
- Collaborator: A trainable auxiliary agent (e.g., Hint or Critic).
- Debater: Participates in multi-agent debate by generating arguments.
- Expert: A specialized agent assigned to a specific subtask.

6.3 Experimental Setup Modules
Datasets: For training, the MATH dataset is used, focusing on problems from difficulty levels 3 to 5. For evaluation, five mathematical reasoning datasets are used: (1) AIME24: 30 high-school olympiad-level problems; (2) AMC: 83 intermediate-difficulty problems; (3) MATH500: 500 randomly sampled problems from MATH; (4) MinervaMath: 272 multi-step reasoning problems; (5) OlympiadBench: 675 high-difficulty mathematics problems.

Models and Baselines: Models include Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct, Qwen2.5-3B-Instruct, Qwen2.5-7B-Instruct, Qwen2.5-Math-7B, and Qwen-Plus. Qwen2.5-3B-Instruct and Qwen2.5-7B-Instruct are employed as the Collaborator, while the remaining LLMs serve as the Actor. Baselines include: (1) CoT: Zero-shot Chain-of-Thought prompting; (2) Self-Consistency: CoT with self-consistency; (3) Self-Refine: Iterative critique and refine; (4) Multi-Agent Debate: Agents debate and aggregate peer feedback; (5) Step-back Abstraction: Reasoning about core principles first; (6) Quality-Diversity: Generates and ensembles diverse solution candidates; (7) Role Assignment: Assigns distinct roles to foundation models; (8) EvoAgent: Self-evolving agent through self-planning, self-control, and self-reflection.

Implementation Details: Reinforcement learning training uses the verl framework. The clipping threshold is set to $\varepsilon = 0.2$. During training, 16 rollouts per prompt are sampled at a temperature of 1.0, with a maximum response length of 2048 tokens. The global batch size is 16, with a per-GPU mini-batch size of 4 and a learning rate of $1 \times 10^{-6}$. For inference, the vLLM library is used, with temperature set to 0.0 and top-p to 1.0. Verification functions from Math-Verify are incorporated. Experiments are conducted on 1 compute node with 4 NVIDIA A40 40GB GPUs.

7. All Mathematical Formulas and Symbol Definitions
Problem Formulation:
- $\mathcal{D}$: Problem distribution where each instance is a tuple $(q, y)$ comprising a problem description $q$ and its ground-truth solution $y$.
- $\theta^*$: Optimal parameters for the multi-agent system.
- $R$: Reward function.
- Objective:
  $$\theta^* = \arg\max_{\theta} \mathbb{E}_{(q,y)\sim\mathcal{D}} [R(\hat{y}, y)] \quad \text{(1)}$$

Framework Formulations:
- $\pi_\theta$: Collaborator's learnable policy with parameters $\theta$.
- $\pi_\phi$: Actor's fixed policy with frozen parameters $\phi$.
- $c_\theta(q)$: Context for the Collaborator, determined by the interaction topology.
- $z$: Auxiliary signal generated by the Collaborator.
  $$z \sim \pi_\theta(\cdot | c_\theta(q)) \quad \text{(2)}$$
- $\hat{y}$: Final solution produced by the Actor.
  $$\hat{y} \sim \pi_\phi(\cdot | q, z) \quad \text{(3)}$$
- $R(ŷ, y)$: Binary reward function based on exact match.
  $$R(\hat{y}, y) = \mathbb{I}(\hat{y} = y) \quad \text{(4)}$$
  where $\mathbb{I}(\cdot)$ is the indicator function.

Feed-forward Mode (Hint-Actor):
- $h \sim \pi_\theta(\cdot | q)$: Hint generation.
- $y \sim \pi_\phi(\cdot | [q;h])$: Actor solution with hint.

Feedback Mode (Critic-Actor):
- $y_{init} \sim \pi_\phi(\cdot | q)$: Initial draft.
- $c \sim \pi_\theta(\cdot | [q; y_{init}])$: Critique generation.
- $y_{ref} \sim \pi_\phi(\cdot | [q; y_{init}; c])$: Refined solution.

GRPO Objective (Module Optimization):
- $J_{GRPO}(\theta)$: GRPO objective function.
  $$J_{GRPO}(\theta) = \mathbb{E}_{q\sim D,\{o_i\}_{i=1}^G\sim\pi_{\theta_{old}}(\cdot|q)} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left( \min \left( r_{i,t}(\theta)\hat{A}_{i,t}, \text{clip}(r_{i,t}(\theta), 1-\varepsilon, 1+\varepsilon) \hat{A}_{i,t} \right) - \beta D_{KL}(\pi_\theta\|\pi_{ref}) \right) \right] \quad \text{(5)}$$
- $r_{i,t}(\theta)$: Probability ratio for token $o_{i,t}$.
  $$r_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|q,o_{i,<t})}$$
- $\hat{A}_{i,t}$: Advantage estimated from intra-group reward comparisons.
- $D_{KL}(\pi_\theta\|\pi_{ref})$: KL divergence term regularizing policy updates relative to a reference policy $\pi_{ref}$.
- $G$: Number of rollouts in a group.
- $\varepsilon$: Clipping threshold.
- $\beta$: KL penalty coefficient.

8. Algorithm Pseudocode
Algorithm 1 AgentPO Algorithm
1: Input: Collaborator policy $\pi_\theta$, fixed Worker policy $\pi_\phi$, problem distribution $D$.
2: Input: Collaboration Topology $T \in \{\text{Hint-Actor, Critic-Actor}\}$.
3: Initialize: Collaborator parameters $\theta$.
4: for each training step do
5:   Sample a problem $(q, y^*)$ from $D$.
6:   if $T$ is Hint-Actor then
7:     Define Collaborator context: $c_\theta \leftarrow q$.
8:     Generate hint (collaborative signal): $z \sim \pi_{\theta_{old}}(\cdot | c_\theta)$.
9:     Worker generates final solution: $y_{final} \sim \pi_\phi(\cdot | [q; z])$.
10:  else if $T$ is Critic-Actor then
11:    Worker generates initial solution: $y_{init} \sim \pi_\phi(\cdot | q)$.
12:    Define Collaborator context: $c_\theta \leftarrow [q; y_{init}]$.
13:    Generate critique (collaborative signal): $z \sim \pi_{\theta_{old}}(\cdot | c_\theta)$.
14:    Worker generates final solution: $y_{final} \sim \pi_\phi(\cdot | [q; y_{init}; z])$.
15:  end if
16:  Compute reward: $R \leftarrow \mathbb{I}(y_{final} = y^*)$.
17:  Store trajectory $(c_\theta, z, R)$ in an experience buffer $B$.
18:  Update policy $\pi_\theta$ using the GRPO objective on data from $B$.
19: end for
20: Return: Optimized Collaborator $\pi_\theta$.