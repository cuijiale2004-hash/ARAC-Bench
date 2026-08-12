**1. Research Background and Existing Pain Points**

Offline reinforcement learning (RL) is a data-driven paradigm that learns exclusively from a fixed dataset of previously collected experiences. This paradigm can be used in scenarios where online interaction is impractical, either because data collection is expensive or dangerous. However, the inability to interact with the environment brings some challenges, like out-of-distribution (OOD) problems. Previous works on offline RL generally address this problem by compelling the learned policy to be close to the behavior policy or constraining the value function to assign low values to OOD actions. In recent years, with the success of Transformer in CV and NLP, the architecture has also been introduced to offline RL to model trajectories. Based on Transformer, the offline RL problem is viewed as a return-conditioned supervised learning (RCSL) task whose core idea is to learn the return-conditional distribution of actions in each state. By framing RL as a supervised learning problem, RCSL improves both data efficiency and training stability, demonstrating superior performance in offline settings.

However, existing RCSL methods, by treating RL as a pure sequence modeling problem, suffer from a fundamental limitation: they tend to overfit to historical sub-trajectories and thus hinder effective trajectory stitching. A sequence generation model is usually required to reliably reproduce trajectories found within the training dataset, but this objective is inherently misaligned with the goal of RL, which is to discover novel policies that may outperform any single trajectory in the dataset by combining the best segments from multiple different trajectories. The policy can overfit to the specific, often suboptimal, actions found within those historical contexts. Consequently, even when conditioned on a high target return, the model struggles to synthesize a correspondingly high-quality action sequence, which fundamentally limits its ability to perform effective trajectory stitching and outperform the behavioral policy. Furthermore, during training, the RTG often fails to accurately represent the true value of a given state-action pair. The mismatch between the target RTG and the optimal RTG can further degrade the performance of RCSL methods, preventing convergence to the theoretical optimum.

**2. Core Research Motivation and Scientific Questions**

Some prior works have acknowledged the challenges associated with long-sequence modeling in the context of RL. However, it remains unclear when sequence modeling specifically hinders trajectory stitching in practice. Existing approaches employ various methods to shorten the dependency on long sequences, but these prior works either lack the flexibility to effectively balance global and local information or require extensive hyperparameter tuning to specific datasets to achieve optimal performance.

The core research motivation is to address the trajectory stitching challenges inherent in sequence modeling by introducing single-step inputs, while not entirely discarding sequence modeling which remains essential in high-dimensional, pixel-based environments where temporal context is necessary to compensate for missing information. The scientific questions are: How to balance sequential and immediate decision information in RCSL? How to overcome the trajectory stitching limitations of conventional sequence modeling while still leveraging valuable long-term information? How to eliminate the need for brittle hyperparameter tuning across different datasets? How to address the known limitation that RTG signals may not reliably reflect action values?

To explain the limitation of sequence modeling, the paper considers a simple task. A sequence modeling method is capable of finding the optimal sub-trajectory “ABCD” in the initial stages. However, after reaching state ‘D’, the context information can easily mislead the model, causing it to follow a trajectory from the training data towards state ‘K’, thereby missing the optimal trajectory: “ABCDE”. In contrast, a single-step modeling method based on the Markov property is unaffected by historical sub-trajectories and can directly select the action that yields the highest return. The results of the simple experiment highlight the inherent difficulty of achieving effective trajectory stitching solely through sequence modeling. Even if with the introduction of the Q-function to guide policy optimization (as in QT), the model remains susceptible to the influence of prior sub-trajectories. This is hypothesized to be due to the design of QT’s policy objective, which seeks to simultaneously optimize action values while constraining the policy to close to the behavior distribution, preventing the policy optimization term from fully counteracting the adverse effects introduced by the behavior cloning term.

**3. Overall Core Idea and Design Philosophy**

The core idea is to learn a state-dependent fusion weight that adaptively balances two distinct feature streams: (1) a global sequential modeling stream, which uses a standard self-attention module to capture long-term context, and (2) a local immediate modeling stream, which employs a simple MLP to extract features from the current state, respecting the Markov property. This dual-feature design introduces a structural bias that helps the model prioritize high-quality local decisions while not ignoring valuable long-term information, eliminating the need for brittle hyperparameter tuning across different datasets. Furthermore, to address the known limitation that RTG signals may not reliably reflect action values, the RCSL framework is augmented with a Q-learning module to enable explicit policy improvement through value-based guidance. The resulting algorithm is the Q-Augmented Dual-Feature Fusion Decision Transformer (QDFFDT).

**4. Core Innovation Points**

1) Identify and demonstrate how pure sequence modeling hinders the trajectory stitching ability of the RCSL paradigm through a simple case study where sequence modeling is misled by context information to follow a suboptimal trajectory, while a single-step method can find the optimal path.
2) Propose QDFFDT, an adaptive method that integrates global sequential features with local immediate features, providing the necessary inductive bias for offline RL. The dual-feature fusion mechanism explicitly separates and then adaptively combines global, history-aware sequence features with local, immediate Markovian features.
3) A learnable fusion mechanism that introduces a structural bias prioritizing single-step dynamics while still leveraging long-term context, improving generalization without the need for extensive hyperparameter tuning. This is achieved by learning a state-dependent fusion weight using an Alpha network and expectile regression to identify suboptimal data.
4) Incorporate the Q-learning module into the RCSL framework to enable explicit policy improvement through value-based guidance, addressing the limitation that RTG signals may not reliably reflect action values. The policy loss is defined as a weighted sum of the DFFDT behavior cloning loss and the Q-value guided improvement term.

**5. Overview of the Overall Technical Solution**

The proposed algorithm is the Q-Augmented Dual-Feature Fusion Decision Transformer (QDFFDT). It consists of two main components: Dual-Feature Fusion Decision Transformer (DFFDT) and Q-Augmented RCSL Optimization.

For the DFFDT component, it adaptively integrates sequential and immediate contextual features using a learnable weighting mechanism. To evaluate the degree of alignment between the training returns and the optimal return, it employs expectile regression to approximate the maximum RTG for a given state using a separate value function Vψ(s). This value function is trained to approximate the empirical upper bound of RTG values in the dataset. When Vψ(s) significantly exceeds an observed RTG, it indicates the subsequent trajectory is suboptimal, which prompts the model to reduce reliance on sequence features and prioritize Markovian information through an Alpha network. The Alpha network produces αω(s) ∈ (0, 1), which is used to compute the final fusion coefficient α̂(s) = αmin + (1−αmin) · αω(s).

The architecture takes the trajectory representation τt = (R̂t−K+1, st−K+1, ..., R̂t−1, st−1, R̂t, st) as input (excluding action tokens). Each RTG-state pair is encoded into a local representation hloct (via a lightweight MLP) and a global representation hglot (via the self-attention mechanism). These two feature streams are then fused through the learned coefficient α̂. Both hloct and hglot are normalized using Layer Normalization (LN) to align their feature scales. The fused representation is finally projected into the action space.

For the Q-Augmented RCSL Optimization component, it develops five networks: two Q-networks Qϕ1, Qϕ2, two target networks Qϕ′1, Qϕ′2, and one target policy network πθ′. The Q-function is updated using either a 1-step or n-step Bellman equation. The policy loss is defined as a weighted sum of the DFFDT behavior cloning loss and the Q-value guided improvement term. The Q-function is normalized to mitigate scale mismatch across offline datasets and prevent gradient imbalance.

**6. Detailed Module Design**

- **V Network**: Built as a 3-layer MLP with ReLU activation functions and 256 hidden units in each layer. It is trained to approximate the empirical upper bound of RTG values in the dataset using expectile regression. The state-value function Vψ(s) is trained to approximate the empirical upper bound of RTG values in the dataset. Notably, this application of expectile regression differs from that in IQL, which uses it to approximate the optimal Q-value. The purpose of Vψ(s) is thus to provide a reliable signal for identifying suboptimal data. Specifically, when Vψ(s) significantly exceeds an observed RTG R̂(s), it indicates the subsequent trajectory is suboptimal.
- **Alpha Network**: Built as a 2-layer MLP with a ReLU activation and a Sigmoid output layer, using 256 hidden units. It learns the adaptive fusion weight αω(s) ∈ (0, 1). To encourage strong reliance on sequence modeling at the beginning of training, it is initialized with zero weights and a large positive bias (e.g., 5) in the final linear layer, ensuring αω(s) ≈ 1 initially. As training progresses, the loss Lα(ω) penalizes αω(s) when Vψ(s) > R̂, driving αω(s) (and thus α̂(s)) downwards. Conversely, when Vψ(s) ≤ R̂, the loss term vanishes, allowing αω(s) and α̂(s) to remain high, thereby preserving the contribution from sequence modeling for high-quality trajectories. The final fusion coefficient is computed as α̂(s) = αmin + (1−αmin) · αω(s), where αmin is a predefined minimum sequential weight.
- **Conditional MLP Policy (Local Feature Stream)**: Built as an RTG-conditioned MLP branch consisting of a 3-layer MLP with GeLU activation functions and 256 hidden units in each layer. It extracts features from the current state, respecting the Markov property, to compute the local representation hloct. A single-step model selects ã = argmax_{ãi} R̃_i(s̃, ãi), making decisions solely based on RTGs and enabling more flexible trajectory stitching by directly choosing actions associated with higher desired returns.
- **Conditional Transformer Policy (Global Feature Stream)**: Built as an RTG-conditioned Transformer branch following the architecture of DT. It uses a standard self-attention module to capture long-term context and compute the global representation hglot. A sequence model, conditioned on preceding sub-trajectories, tends to choose the next action from within a trajectory in the dataset that closely resembles the observed history, which often prioritizes the continuation of previously observed sub-trajectories.
- **Dual-Feature Fusion Block**:
  - Input: Trajectory representation τt = (R̂t−K+1, st−K+1, ..., R̂t−1, st−1, R̂t, st), excluding action tokens.
  - Local Representation: hloct computed by the Conditional MLP Policy.
  - Global Representation: hglot computed by the Conditional Transformer Policy via the self-attention mechanism.
  - Normalization: Both hloct and hglot are normalized using Layer Normalization (LN) to align their feature scales to enable effective fusion.
  - Fusion: The two feature streams are fused through the learned coefficient α̂ as follows: h = (1− α̂(st)) · hloct + α̂(st) · hglot. The model adaptively modulates the influence of sequential features based on whether the training RTGs for s̃ align with the optimal return.
  - Projection: The fused representation h is finally projected into the action space.
- **Q Networks**: Two Q networks with the same architecture, each consisting of a 4-layer MLP with Mish activation functions and 256 hidden units in each layer. Used for Q-value function learning and policy improvement. The Q-function is normalized to mitigate scale mismatch across offline datasets and prevent gradient imbalance between the two learning objectives.

**7. All Mathematical Formulas and Symbol Definitions**

- Expectile of a random variable X: 
  argmin_{mσ} E_{x∼X} [L^σ_2 (x−mσ)]
  where L^σ_2 (u) = |σ − 1(u < 0)|u^2, and σ ∈ (0, 1).
- V network loss:
  LV (ψ) = E_{(s,R̂)∼D}[L^σ_2 (R̂− Vψ(s))]
- Alpha network loss:
  Lα(ω) = E_{(s,R̂)∼D} [ αω(s) · max(Vψ(s)− R̂, 0) / T ]
  where T is a temperature hyperparameter, and αω(s) ∈ (0, 1).
- Final fusion coefficient:
  α̂(s) = αmin + (1−αmin) · αω(s)
  where αmin is a predefined minimum sequential weight.
- Trajectory representation:
  τt = (R̂t−K+1, st−K+1, ..., R̂t−1, st−1, R̂t, st)
- Feature fusion:
  h = (1− α̂(st)) · hloct + α̂(st) · hglot
- DFFDT training loss (Mean Squared Error):
  LDFFDT (θ) = E_{(τt,at−K+1:t)∼D} [ 1/K ∑_{i=t−K+1}^t (ai − πθ(τt)i)^2 ]
  where ai denotes the ground-truth action and πθ(τt)i is the i-th action predicted by the policy network.
- Q-function loss:
  E_{(st−K+1:t+1,at−K+1:t,rt−K+1:t)∼D} ∑_{m=t−K+1}^t ∥Q̂m − Qϕi(sm, am)∥^2
  where
  Q̂m = { ∑_{j=m}^t γ^{j−m}rj + γ^{t+1−m} min_{i=1,2} Qϕ′_i (st+1, ât+1), (n-step)
         { rm + γ min_{i=1,2} Qϕ′_i (sm+1, âm+1), (1-step)
  where γ is the discount factor, and target actions âm+1 and ât+1 are sampled from the target policy πθ′.
- Policy loss:
  Lπ(θ) = λ · LDFFDT (θ) + LQ(θ) = λ · LDFFDT (θ)− E_{τt∼D} E_{si∼τt} Qϕ(si, πθ(τt)i)
  where λ is a weighting coefficient. The Q-function is normalized.

**8. Algorithm Pseudocode**

Algorithm 1 QDFFDT: Q-Augmented Dual-Feature Fusion Decision Transformer
Initialize V network Vψ and Alpha network αω .
# Train the V network
for each iteration do
Sample transition mini-batch B = {(R̂j , sj)} ∼ D.
Update Vψ by Equation 2.
end for.
# Train the Alpha network
for each iteration do
Sample transition mini-batch B = {(R̂j , sj)} ∼ D.
Update αω by Equation 3.
end for.
Initialize policy network πθ and critic networks Qϕ1, Qϕ2.
Initialize target networks πθ′ , Qϕ′1, Qϕ′2.
# Train the QDFFDT
for each iteration do
Sample sequence transition mini-batch B = {(R̂j , sj , rj)t+Kj=t } ∼ D.
# Q-value function learning
Sample target actions â via one of two methods:
(n-step): Sample ât+K ∼ πθ′(ât+K |R̂t+1:t+K , st+1:t+K)
(1-step): For each step m ∈ [1,K], sample ât+1:t+i ∼ πθ′(ât+1:t+i|R̂t+1:t+i, st+1:t+i)
Update Qϕ1 and Qϕ2 by Equation 7.
# Policy learning
for i = 0 to K − 1 do
Sample at:t+i ∼ πθ′(at:t+i|R̂t:t+i, st:t+i).
end for.
Update policy by minimizing Equation 8.
# Update target networks
θ′ = ρθ′ + (1− ρ)θ, ϕ′i = ρϕ′i + (1− ρ)ϕi for i = {1, 2}.
end for.
# Inference with QDFFDT
Given multiple target return-to-go choice R̂1:m0 and initial state s0.
repeat
Sample multiple actions with different return-to-go ãit = πθ(ãit|R̂it−K+1:t, st−K+1:t) for i = 1, . . . ,m.
Compute Q value with candidate state-action pair (st, ãit) for i = 1, . . . ,m.
Sample the action at from action set {ãit}mi=1 with the max Q value by
argmax_{ãit} min_{i=1,2} Qϕ′_i (st, ãit).
Execute the action at and collect the reward rt and next state st+1.
Update current return-to-go R̂it+1 = R̂it − rt for i = 1, . . . ,m.
until Done is true.