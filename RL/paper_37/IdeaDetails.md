1. Research Background and Existing Pain Points
Scalable and realistic simulation of multi-agent traffic behavior is critical for advancing autonomous driving technologies. Traditional simulators that simply replay logged data lack reactive capability, while rule-based methods, such as the Intelligent Driver Model (IDM), depend on handcrafted heuristics and fail to capture the diversity and realism of human behavior. Recent research has reframed realistic traffic simulation as a distribution matching problem, aiming to align the behavior distributions of simulated agents with those observed in real-world data. 

Existing pain points include:
(1) Existing data-driven simulators predominantly rely on supervised learning to align simulated distributions with real-world driving scenarios. A persistent challenge lies in the distributional shift that arises between training and testing, which often undermines model generalization in unseen environments.
(2) Early learning-based simulators largely adopted architectures designed for motion forecasting. However, simulation fundamentally differs from trajectory prediction: it requires (i) a closed-loop setup, (ii) consistent scene-level multi-agent futures, and (iii) recovery of the underlying behavior distribution rather than simple trajectory imitation. These distinctions often cause motion predictors to perform poorly when adapted to simulation.
(3) Although autoregressive approaches treat agent behaviors as discrete tokens under the Next-Token Prediction (NTP) paradigm to generate interactive and realistic multi-agent behaviors through minimizing cross-entropy loss, they are prone to encountering the covariate shift issue where small prediction errors accumulate during closed-loop simulation rollouts.
(4) A fundamental limitation persists: the training objectives of current imitative or Behavior Cloning (BC) models are not explicitly aligned with ultimate behavior-characterizing goals of simulators, such as reducing collisions or off-road rates. These outcome metrics are often scalar, sparse, and non-differentiable, making them unsuitable as direct trainable loss functions for gradient-based optimization. Relying solely on BC or Supervised Fine-Tuning (SFT) is insufficient for realistic simulation agents.

2. Core Research Motivation and Scientific Questions
Core Research Motivation: To bridge the gap between the training objectives of current imitative/BC models and the ultimate behavior-characterizing evaluation metrics of simulators. The work draws inspiration from cutting-edge Large Reasoning Models (LRMs) such as OpenAI-o1 and DeepSeek-R1, which leverage Reinforcement Learning from Human Feedback (RLHF) or Reinforcement Fine-Tuning (RFT) to align model behaviors with specific preferences.
Scientific Questions: 
(1) How to effectively apply an R1-style reinforcement fine-tuning paradigm directly to the domain of multi-agent traffic simulation to align NTP model behaviors with designated target metrics?
(2) How to design a policy optimization algorithm that overcomes the sampling bias and unreliable advantage estimation present in methods like Group Relative Policy Optimization (GRPO), which relies on average rewards across multiple sampled rollouts?
(3) How to prevent catastrophic forgetting of the original behavioral distribution learned during pretraining and SFT when applying RFT, which risks deviating the model from real-world logged data distributions?

3. Overall Core Idea and Design Philosophy
Overall Core Idea: Introduce SMART-R1, a novel R1-style reinforcement fine-tuning paradigm tailored for next-token prediction models to better align agent behavior with human preferences and evaluation metrics. The approach combines SFT and RFT to maximize performance gains.
Design Philosophy: 
(1) Metric-oriented alignment: Instead of relying solely on behavior cloning losses that are misaligned with sparse, non-differentiable outcome metrics, explicitly optimize the policy towards targeted evaluation metrics using reinforcement fine-tuning.
(2) Exploiting predictable reward expectations: Given the relatively predictable reward expectations in the traffic simulation task, avoid the sampling bias of GRPO by proposing a Metric-oriented Policy Optimization (MPO) strategy that uses an empirical threshold as prior knowledge to guide policy updates more efficiently and effectively.
(3) Balancing metric optimization and distribution preservation: Address catastrophic forgetting caused by RFT by introducing an iterative "SFT-RFT-SFT" post-training scheme. This strategy alternates between SFT and RFT to strike a balance between optimizing for metric-driven objectives and preserving generalization from real-world data, ensuring the model remains faithful to the logged data distribution while improving metric scores.

4. Core Innovation Points
(1) Introduce SMART-R1, the first R1-style post-training paradigm for multi-agent traffic simulation, which combines SFT and RFT to better align simulated behaviors with human preferences and evaluation metrics.
(2) Develop a simple yet effective Metric-oriented Policy Optimization (MPO) strategy for RFT to reinforce the model towards targeted evaluation metrics, which exploits prior knowledge of relatively predictable reward expectations to avoid the sampling bias of GRPO.
(3) Propose an iterative "SFT-RFT-SFT" post-fine-tuning pipeline that balances optimization for metric-specific objectives with preservation of the behavioral distribution learned during SFT, mitigating catastrophic forgetting and further enhancing overall simulation realism.
(4) Achieve state-of-the-art performance on the 2025 Waymo Open Sim Agents Challenge, validating the effectiveness of the simple yet powerful R1-style training framework in enhancing foundation models for traffic simulation.

5. Overview of the Overall Technical Solution
The overall architecture follows a multi-stage training paradigm consisting of Behavior Cloning (BC) pretraining, closed-loop SFT, and RFT.
Given a real-world scenario, continuous agent trajectories and map polylines are first discretized into agent motion tokens and map tokens using the K-disks clustering algorithm. These tokens are then processed by a stack of attention layers: a temporal self-attention layer to capture sequential dependencies in agent motions, a map-to-agent cross-attention layer to incorporate map-relevant information, and an agent-to-agent self-attention layer to model interactions among traffic participants for generating joint multi-agent behaviors. The outputs from these modules produce the next-token logits. 
In the BC pretraining stage, the model is optimized via token-level classification under the standard NTP paradigm. The resulting pretrained base model is subsequently refined through closed-loop SFT, where Closest Among Top-K (CAT-K) rollouts are employed to mitigate the covariate shift inherent in BC. 
Finally, in the RFT stage, the proposed MPO algorithm aligns the policy with target evaluation metrics, further improving overall simulation realism. 
The post-training pipeline is structured as "SFT-RFT-SFT": starting from a pretrained base model, first apply CAT-K rollouts for closed-loop SFT (16 epochs) to mitigate covariate shift and stabilize the policy; then employ MPO for RFT to explicitly align with evaluation metrics; and finally conduct an additional round of closed-loop SFT (16 epochs) to restore fidelity to the logged data distribution, thereby enhancing simulation performance.

6. Detailed Module Design
(1) Tokenization Module: The driving context is preprocessed by discretizing and clustering continuous agent trajectories into a motion token vocabulary comprising template motion tokens using the K-disks clustering algorithm. Continuous trajectories and map polylines are converted into agent motion tokens and map tokens.
(2) Attention Stack Module: 
- Temporal self-attention layer: Captures sequential dependencies in agent motions.
- Map-to-agent cross-attention layer: Incorporates map-relevant information.
- Agent-to-agent self-attention layer: Models interactions among traffic participants for generating joint multi-agent behaviors.
(3) Closed-Loop SFT Module (CAT-K): After obtaining the pre-trained foundation model, the policy is autoregressively unrolled to generate tokenized trajectories, and recovery motion indices are constructed to stay close to the ground truth. CAT-K rollouts are performed to obtain a sequence of motion tokens and identify the motion index that is closest to the ground-truth trajectories. The model is optimized using cross-entropy loss.
(4) Metric-oriented Policy Optimization (MPO) Module: The behavior simulation problem is formulated as a Markov Decision Process (MDP), where each tokenized motion corresponds to an element in the action space. The policy deterministically transitions each agent to its next state by selecting the associated motion token. The reward model is defined by the Realism Meta metric. For each traffic scenario, complete trajectories for all agents are autoregressively rolled out, and the resulting simulation is evaluated using the official Realism Meta metric to assign a scalar reward score. The Generalized Advantage Estimation (GAE) is computed as the reward minus an empirical threshold. A per-token KL divergence penalty between the updated policy and a reference model is incorporated using an unbiased estimator. The policy is updated by minimizing the MPO loss function.

7. All Mathematical Formulas and Symbol Definitions
- Equation 1: Joint distribution of agent behaviors
$P (S_{1:\tau} |S_0, C) = \prod_{t=1}^{\tau} \pi_\theta(S_t|S_{<t}, C)$
Where $\pi_\theta(S_t|S_{<t}, C)$ is the policy; $S_t = \{s_i^t\}_{i=1}^N$ denotes the state of $N$ agents in the scene at time step $t$; $S_{<t} = \{s_i^{<t}\}_{i=1}^N$ is the historical states; $C$ represents the contextual information such as features from High-Definition (HD) maps; $\theta$ denotes the trainable model parameters; $\tau$ is the time horizon.

- Equation 2: Next-token prediction policy
$\pi_\theta(S_t|S_{<t}, C) = \prod_{i=1}^N \pi_\theta(k_i^t|S^*_{<t}, C^*)$
Where $V = \{m_k\}_{k=1}^{|V|}$ is the motion token vocabulary comprising $|V|$ template motion tokens $m_k$; $X^*$ denotes the tokenized format of the inputs $X$; $k_i^t$ is the index of the motion token vocabulary at time step $t$ for agent $i$.

- Equation 3: Generalized Advantage Estimation
$A = r - \alpha$
Where $A$ is the simplified Generalized Advantage Estimation (GAE); $r$ is the reward function computed for the simulation outcomes using the Realism Meta metric; $\alpha$ is an empirical threshold used to normalize the reward signal and facilitate relative evaluation for efficient policy optimization.

- Equation 4: Unbiased estimation of per-token KL divergence
$D_{KL}[\pi_\theta||\pi_{\theta_{ref}}] = \frac{\pi_{\theta_{ref}}}{\pi_\theta} - \log \frac{\pi_\theta}{\pi_{\theta_{ref}}} - 1$
Where $D_{KL}[\pi_\theta||\pi_{\theta_{ref}}]$ is the per-token KL divergence penalty between the updated policy $\pi_\theta$ and a reference model $\pi_{\theta_{ref}}$.

- Equation 5: Training objective for MPO
$L_{MPO} = -(\frac{\pi_\theta}{\pi_\theta}A - \beta D_{KL}[\pi_\theta||\pi_{\theta_{ref}}])$
Where $\pi_\theta$ is a no-gradient copy of the policy (as described in the text, reflecting the original paper's notation); $\beta$ is a tunable coefficient balancing the advantage term against the KL penalty.

- Equation 6: Base model training objective
$L_{base} = -\sum_{t=1}^{\tau} \sum_{i=1}^N \log (\pi_\theta(\tilde{k}_i^t|S^*_{<t}, C^*))$
Where $\tilde{k}$ is the corresponding ground-truth index.

8. Algorithm Pseudocode
Algorithm 1 Metric-oriented Policy Optimization (MPO)
Input: Initial policy $\pi_{\theta_{init}}$.
Output: Optimized RFT policy $\pi_{\theta_{rft}}$.
1: Initialize the reference model $\pi_{\theta_{ref}} \leftarrow \pi_{\theta_{init}}$.
2: Initialize the policy model $\pi_\theta \leftarrow \pi_{\theta_{init}}$.
3: for iteration = 1, 2, . . . do
4:     Roll out multi-agent trajectories in the scene via $\pi_\theta$.
5:     Compute the reward function $r$ for the simulation outcomes using the Realism Meta metric.
6:     Compute the simplified generalized advantage estimation $A$ via Equation 3.
7:     Compute the unbiased estimation of per-token KL divergence $D_{KL}[\pi_\theta||\pi_{\theta_{ref}}]$ via Equation 4.
8:     Update policy $\pi_\theta$ by minimizing $L_{MPO}$ via Equation 5.
9: end for