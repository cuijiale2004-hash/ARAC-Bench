1. Research Background and Existing Pain Points
Offline reinforcement learning (Offline RL) aims to learn effective policies from fixed datasets without further interaction with the environment. This setting is important in real-world domains such as robotics, logistics, and operations research, where environment access is limited and data collection is expensive or risky. A central challenge in offline RL is distributional shift: when a learned policy queries state-action pairs outside the dataset support, value extrapolation can cause severe overestimation and degenerate performance.
Contemporary methods primarily employ policy constraints or value regularization to address this challenge. Policy-constraint methods are largely limited by the behavior policies used to collect the offline data and exhibit a trade-off between generalization and safe adherence to the constraint. Recent value-regularization methods aim to provide conservative estimates that impose softer penalties on out-of-distribution actions. Nevertheless, the optimality of the learned value function is not guaranteed when the static dataset is limited and potentially biased.
Furthermore, existing approaches often rely on a shared global Q-function. In many RL control tasks, the state space can be naturally divided into multiple subtasks. A shared global Q-function is inadequate for capturing the compositional structure of such tasks. Knowledge learned from one subtask may not be fully transferable to another. Geometrically similar state clusters may correspond to semantically different behaviors and exhibit distinct long-horizon returns. When offline data are insufficient and exploration is unavailable, this property is not naturally captured by an offline value-learning algorithm that fits a shared global value function.

2. Core Research Motivation and Scientific Questions
The core research motivation is to address the limitations of global value approximation in offline RL by enabling flexible local adaptation of value estimation within context windows. The scientific questions explored are: How can value learning in offline RL be cast as a contextual inference problem? How can local Q-functions be adaptively inferred from small retrieved sets of transitions without explicit subtask labels or predefined subtask structure? Can linear attention be utilized as a tool for in-context value learning to provide strong performance and generalization benefits for robust and compositional value estimation in offline RL?

3. Overall Core Idea and Design Philosophy
The overall core idea is to cast value learning in offline reinforcement learning as a contextual inference problem, thereby enabling local Q-function approximation through in-context learning. The design philosophy introduces In-context Compositional Q-Learning (ICQL), a general offline RL framework that leverages the in-context learning capability of linear Transformers to infer local Q-functions from small retrieved sets of transitions. Rather than fitting a global approximator of the value function, ICQL leverages the compositional nature and local structure of the task to learn a family of value functions, thereby enabling flexible local adaptation of value estimation within context windows without requiring additional supervision.

4. Core Innovation Points
• Introduces the first offline RL framework ICQL that formulates Q-learning as a contextual inference problem, leveraging in-context learning with linear Transformers to adaptively infer local Q-functions without requiring explicit subtask labels or predefined subtask structure.
• Provides a theoretical analysis showing that ICQL achieves bounded approximation error under two assumptions—linear approximability of the local Q-function and accurate inference of the weights from retrieved context—and proves that the greedy policy with respect to the inferred Q-function is near-optimal.
• Improves performance in offline settings through in-context local approximation, demonstrating effectiveness in both offline Q-learning and offline actor-critic frameworks. Achieves significant performance improvements (up to 16.4% on kitchen tasks, 8.8% on MuJoCo, and 6.3% on Adroit) and yields more accurate value estimation than strong baselines.
• Conducts extensive ablation studies to isolate the contributions of in-context learning and localized value inference, investigating the impact of different retrieval strategies, including similarity metrics and context selection criteria, on overall performance and stability.

5. Overview of the Overall Technical Solution
The overall technical solution defines local Q-functions based on a local neighborhood associated with each state. It uses a retrieval mechanism to select the top-k most similar transitions from the offline dataset to form a local context set. It constructs an input "prompt" matrix from the retrieved transitions and the query state-action pair. An L-layer linear Transformer is used to perform in-context TD learning, computing a context-dependent Q-function approximation. The framework is trained by performing value iteration via expectile regression and policy extraction via advantage-weighted regression, following the IQL paradigm. Theoretical analysis is provided to bound the approximation error and prove the near-optimality of the greedy policy.

6. Detailed Module Design
• Local Q-functions (Definition 3.1): Defines nearby transitions within a radius d (and d̄ for next states) around a query state s. It assumes the existence of an optimal uniform local weight vector w∗_s such that the local Q-function can be approximated linearly using a feature function ϕ.
• Retrieval Methods (Definition 3.2): Introduces State-Similar Retrieval, which retrieves k transitions with the smallest l2-distance between the retrieved state and the query state. Random Retrieval selects transitions uniformly at random. State-Similar-with-High-Rewards Retrieval filters similar-state candidates by selecting those with higher rewards.
• In-Context Compositional Q-Learning (Definition 3.3): Defines a context-dependent weight function w_s inferred through in-context learning. It constructs a prompt matrix Z_0 and passes it through an L-layer linear Transformer with specific weight matrices P_ℓ and G_ℓ. The output yields a context-dependent Q-function approximation. The linear attention layers iteratively update the weights using a SARSA-like rule.
• Training Procedure: The critic loss is defined using expectile regression on the local Q-function approximation. The policy is optimized via advantage-weighted regression using an advantage defined by local value estimation.
• Theoretical Analysis (Assumptions 3.1 & 3.3, Theorem 3.5): Provides bounded feature norms (Assumption 3.1) and set coverage conditions (Assumption 3.3). Proves that the performance gap between the optimal policy and the greedy policy with respect to the learned local Q-function is bounded.
• Variants for TD3+BC: Extends ICQL to the TD3+BC framework by replacing the global Q-function with the local estimator, using MSE for the critic loss and adding a behavior cloning regularization term to the actor loss.
• Network Architectures: The in-context critic consists of a 3-layer MLP feature extractor (256 hidden units, Tanh in the last layer, ReLU in others, layer normalization, output dimension 64, dropout 0.1) and a linear Transformer (20 layers, gradient normalization with max L2 norm of 10). The policy network is a 2-layer MLP for ICQL-IQL and a 3-layer MLP for ICQL-TD3+BC.

7. All Mathematical Formulas and Symbol Definitions
Definition 3.1:
(s̄, ā, r̄, s̄′, ā′) ∈ { (s_i, a_i, r_i, s′_i, a′_i) ∈ D | ∥s_i − s∥^2_2 ≤ d^2 and ∥s′_i − s_i∥^2_2 ≤ d̄^2 } ≜ Ω^(d,d̄)_s
Q̂_Ω^(d,d̄)_s(s̄, ā) ≜ w∗_s^⊤ ϕ(s̄, ā), ∀(s̄, ā, r̄, s̄′, ā′) ∈ Ω^(d,d̄)_s
|Q_Ω^(d,d̄)_s(s̄, ā) − w∗_s^⊤ ϕ(s̄, ā)| ≤ ε^approx_s, ∀(s̄, ā, r̄, s̄′, ā′) ∈ Ω^(d,d̄)_s

Definition 3.2:
Ω_s_query ≜ { (s_i, a_i, r_i, s′_i, a′_i) ∈ D | s_i ∈ arg top-k {−∥s_query − s_i∥^2_2} }
d^s_query_k ≜ max_{(s_i,a_i,r_i,s′_i,a′_i)∈Ω_s_query} {∥s_query − s_i∥_2}

Definition 3.3:
Prompt matrix Z_0:
Z_0 = [ ϕ_0 · · · ϕ_{N−1} ϕ_query; γϕ′_0 · · · γϕ′_{N−1} 0; r_0 · · · r_{N−1} 0 ]
where ϕ_i ≜ ϕ(s_i, a_i), ϕ′_i ≜ ϕ(s′_i, a′_i), and ϕ_query ≜ ϕ(s_query, a_query).

Layer weight matrices:
P_ℓ ≜ [ 0_{2d×2d} 0_{2d×1}; 0_{1×2d} 1 ]
G_ℓ ≜ [ −C^T_ℓ C^T_ℓ 0_{d×1}; 0_{d×d} 0_{d×d} 0_{d×1}; 0_{1×d} 0_{1×d} 0 ]

Linear Attention:
LinAttn(Z; P, G) ≜ PZM(Z^⊤GZ)

Context-dependent Q-function approximation:
Q̂(s_query, a_query | Ω^{d_k}_s_query) = w^L_s_query(Ω^{d_k}_s_query)^⊤ ϕ(s_query, a_query)

Weight update rule:
w^{l+1}_s_query(Ω^{d_k}_s_query) = w^l_s_query(Ω^{d_k}_s_query) + α (r + γ w_s_query(Ω^{d_k}_s_query)^⊤ ϕ(s′, a′) − w^l_s_query(Ω^{d_k}_s_query)^⊤ ϕ(s, a)) ϕ(s, a)

Critic Loss:
L_critic = E_{(s,a,r,s′)∼D} [ ρ_τ ( Q̂(s, a | Ω^{d_k}_s) − y ) ]
where y = r + γ V (s′ | Ω^{d_k}_{s′}), V (s′ | Ω^{d_k}_{s′}) = E_{a′∼π} [ Q̂(s′, a′ | Ω^{d_k}_{s′}) ]

Policy Loss:
L_policy = E_{s∼D} [ E_{a∼π} [ exp(β · (Q̂(s, a | Ω^{d_k}_s) − V (s | Ω^{d_k}_s))) log π(a|s) ] ]

Assumption 3.1:
∥ϕ(s, a)∥ ≤ B_ϕ

Assumption 3.3 (Set Coverage):
κ_s_query ≜ |Ω^{d_k}_s_query ∩ Ω∗_s_query| / |Ω∗_s_query| ≥ σ

Theorem 3.5 (Policy Performance Gap):
J(π∗) − J(π) ≤ 2/(1−γ) E_{s∼d^π} [ ε^approx_s(1 + B_ϕ) + C B_ϕ sqrt( (d + log(1/δ)) / (σ |Ω^{d_k}_s|) ) ]

ICQL-TD3+BC Critic Loss:
L^{TD3+BC}_critic = E_{(s,a,r,s′)∼D} [( Q̂(s, a | Ω^{d_k}_s) − y )^2]
where y = r + γ min_{i=1,2} Q̂^{(i)}_{target}(s′, π(s′) | Ω^{d_k}_s)

ICQL-TD3+BC Actor Loss:
L^{TD3+BC}_actor = −E_{s∼D} [ Q̂(s, π(s) | Ω^{d_k}_s) ] + α · E_{(s,a)∼D} [ ∥π(s) − a∥^2 ]

8. Algorithm Pseudocode
Algorithm 1 In-context Q-Learning (ICQL)
1: Input: Offline dataset D, the number of retrieved transitions k, feature dimension d.
2: Initialize: Linear transformer TFQ_θ with parameters θ, feature extractor ϕ.
3: Sample a trajectory {(s_i, a_i, r_i)}^{T−1}_{i=0} ∼ D.
4: For each query state s_i, retrieve k states s^0_i, · · · , s^{k−1}_i using the state-similar retrieval method defined in Definition 3.2 and extract the corresponding transitions {(s^j_i, a^j_i, r^j_i, s′^j_i, a′^j_i)}^{k−1}_{j=0}.
5: //In-context Q value estimation.
6: for t = 0, . . . , T − 1 do
7: Construct the input prompt matrix Z_t by Equation (5).
8: Q̂_t ← TFQ_θ (Z_t)[2d+ 1, k + 1] by Equation (18).
9: end for
10: Update the parameters θ, ϕ using Equation (9) and Equation (10).