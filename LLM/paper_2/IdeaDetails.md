1. Research Background and Existing Pain Points
Large Language Models (LLMs) have recently emerged as powerful agents capable of reasoning, planning, and interacting with external environments. When combined with reinforcement learning (RL), such agents can adapt their behavior based on experience and feedback, enabling them to go beyond static prompting or supervised fine-tuning. This paradigm has driven recent progress in areas such as interactive decision-making, tool use, and embodied AI. However, a key limitation of current LLM-based agents lies in their reliance on exploiting prior knowledge rather than engaging in systematic exploration. While RL frameworks emphasize balancing exploration and exploitation, many LLM-agent systems primarily leverage pretrained knowledge and conduct only limited search within familiar distributions. As a result, these agents often struggle in environments where progress depends on discovering novel states or actively acquiring new information, rather than reusing what is already known. To address this challenge, recent research has incorporated external memory modules into LLMs as a form of long-term memory. This enables models to leverage past experiences to correct failed attempts, thereby improving decision-making in subsequent trials without requiring parameter updates. However, the performance of such methods tends to saturate quickly, since collecting experiences with static parameters cannot fully capture the diversity needed for continuous improvement. Without intrinsic exploration capability, online RL struggles to optimize effectively. As illustrated in the ScienceWorld environment, when an agent is given the mission "turn on the red light bulb" and the object is not present in the current observation, the agent follows the instruction literally, attempts to focus on the red light bulb, and fails. Ideally, when an agent fails to reach its goal, it should analyze the reasons for failure and broaden its action space to discover successful strategies. Yet in representative online RL algorithms like GRPO, prior trajectory rollouts provide no continuity beyond a scalar reward signal, thereby restricting exploration and ultimately limiting learning.

2. Core Research Motivation and Scientific Questions
The core research motivation is to present a unified framework that enables LLM agents to learn more effectively through broader exploration by jointly updating their parametric policy parameters with RL and their non-parametric memory module through interaction. Crucially, the non-parametric updates not only complement but also enhance the efficiency of parametric learning, thereby enabling more effective exploration and adaptation. This dual-update paradigm serves as a bridge between parameter-level optimization and memory-augmented reasoning. While memory is utilized during learning, moving toward more generalizable intelligence requires reducing dependence on external memory and instead embedding its benefits directly into the model’s parameters. The scientific question is how to efficiently train agents in online RL through trial and error, without any prior embedding of the environment’s rules, solving the key challenge that without intrinsic exploration capability, online RL struggles to optimize effectively. The research aims to build an algorithm that incorporates two modes in the rollout phase—depending on whether memory is used—and two modes in the update phase—on-policy and off-policy learning—thereby enabling agents to leverage memory when available while remaining robust in its absence.

3. Overall Core Idea and Design Philosophy
The overall core idea is Exploratory Memory-Augmented On- and Off-Policy Optimization (EMPO2), a novel algorithm aimed at tackling the exploration challenges in online RL. The design philosophy operates on the principle that non-parametric updates can encourage exploration, bootstrapping parametric updates. EMPO2 leverages memory for exploration and combines on- and off-policy updates to make LLMs perform well with memory while also ensuring robustness without it. During rollout, actions can be generated either through prompting without memory or memory-augmented prompting conditioned on tips retrieved from memory. In the update phase, rollouts with memory-augmented prompting are used in two ways: on-policy, where tips are retained and the update is performed with the original prompt, and off-policy, where tips are removed during update. Notably, tips are generated not by a separate model but by the policy itself, which is continually updated during training. The off-policy update functions as a form of reward-guided knowledge distillation. Trajectories sampled under the tips-conditioned policy act as teacher demonstrations, while the student policy is updated to reproduce those trajectories in proportion to their advantage. High-reward trajectories are reinforced, while low-reward trajectories are suppressed, resulting in selective distillation that emphasizes beneficial behaviors. In this way, tips serve as an intermediate scaffolding mechanism that improves exploration and trajectory quality, while the reward signal ensures that only advantageous behaviors are ultimately retained. Consequently, the final policy learns to internalize the benefits of tip conditioning without requiring tips at inference time.

4. Core Innovation Points
1. A novel hybrid RL framework (EMPO2) that integrates non-parametric memory updates into both on- and off-policy learning, yielding substantially higher sample efficiency and tackling the exploration bottleneck in online RL for LLM agents.
2. A self-generated memory mechanism where the agent reviews past rollouts, generates self-guidance tips, and stores them in memory. These tips help the agent avoid repeated mistakes and explore new strategies, maintaining continuity across rollouts and promoting exploration with all data and guidance generated autonomously by the agent.
3. An off-policy update mode via reward-guided knowledge distillation. Actions are sampled from the tips-conditioned policy (teacher), but the log-probabilities for the new policy are recomputed without the tips (student). This internalizes the benefits of tip conditioning into the model parameters without requiring tips at inference time.
4. A masking mechanism to stabilize off-policy training. It suppresses the advantage term for tokens whose probability under the current policy falls below a threshold, preventing low-probability tokens from destabilizing training by amplifying gradient magnitudes through unbounded likelihood ratios.
5. An intrinsic reward mechanism based on state novelty to encourage exploration. A memory list stores distinct states, and for each new state, its cosine similarity with existing entries is computed. If the similarity falls below a threshold, the state is added to memory and assigned a reward based on the number of similar past states, maintaining policy entropy.

5. Overview of the Overall Technical Solution
The overall technical solution is the EMPO2 algorithm, which operates in two modes for both rollout phase and update phase. During the rollout phase, the agent samples between two modes: (1) Prompting Without Memory, where the policy generates actions conditioned only on the current state and task with probability 1-p; and (2) Memory-Augmented Prompting, where a retrieval operator selects tips most relevant to the current state, and the policy conditions its action on both the state and the retrieved tips with probability p. During the update phase, trajectories generated under rollout mode (1) are directly used for standard on-policy updates. Trajectories generated under rollout mode (2) follow one of two update modes chosen at random: Mode (a) On-Policy Updates, selected with probability 1-q, where the same prompt with tips is used; and Mode (b) Off-Policy Updates, selected with probability q, where the stored log-probabilities conditioned on tips are replaced with the log-probabilities assigned by the same policy when conditioned only on the state and task. This off-policy mode acts as reward-guided knowledge distillation. Additionally, a masking mechanism is applied to the loss function to suppress the advantage term for low-probability tokens, and an intrinsic reward based on state novelty is introduced to further encourage exploration.

6. Detailed Module Design
6.1 Self-Generated Memory and Tips Module
A memory buffer M = {tip1, tip2, ...} stores reflective tips generated by the policy during trajectory reflection. When an episode i of task u terminates at timestep t, the policy takes the final state st together with a tip-generation prompt as input and produces a tip, where tipi ~ πθ(st, u, tip-generation prompt). These tips are generated not by a separate model but by the policy πθ itself, which is continually updated during training. This module strengthens the linkage between rollouts and promotes exploration.

6.2 Rollout Modes Module
During rollouts, the agent samples between the two modes, selecting one mode at each step: mode (2) with memory rollout probability p and mode (1) with probability 1−p.
(1) Prompting Without Memory: For each task u, at each timestep t, the policy πθ generates actions conditioned only on the current state st and the task u: at+1 ~ πθ(· | st, u).
(2) Memory-Augmented Prompting: For each task u, at each timestep t, a retrieval operator Retr(ot; M) ⊆ M selects tips most relevant to the current state st, e.g., via similarity search in the embedding space. The retrieved set is denoted as tipst. In memory-augmented prompting, the policy conditions its action on both st and tipst: at+1 ~ πθ(· | st, u, tipst). The number of retrieved tips is limited to 10.

6.3 Update Modes Module
Trajectories generated under rollout mode (1) are directly used for updates, whereas those generated under rollout mode (2)—memory-augmented prompting—follow one of two update modes chosen at random during the update phase. Mode (b) is selected with off-policy update probability q, and mode (a) with probability 1−q.
(a) On-Policy Updates: On-policy update uses the same prompt as in the rollout, and the importance sampling ratio ρθ(a(i)t) in eq.1 becomes ρθ(a(i)t) = πθ(a(i)t | s(i)t, u, tipst) / πθold(a(i)t | s(i)t, u, tipst).
(b) Off-Policy Updates: In this mode, the stored log-probabilities ℓtipst = log πθ(at | st, u, tipst) are replaced with the log-probabilities assigned by the same policy πθ when conditioned only on (st, u), namely ℓno-tipst = log πθ(at | st, u). In this formulation, the advantage update is performed based on how natural the action appears under the distribution without tips. This construction is interpreted as a form of reward-guided knowledge distillation. Trajectories sampled under the tips-conditioned policy act as teacher demonstrations, while the student policy πθ(· | s, u) is updated to reproduce those trajectories in proportion to their advantage. High-reward trajectories (Ât > 0) are reinforced, while low-reward trajectories (Ât < 0) are suppressed, resulting in selective distillation that emphasizes beneficial behaviors.

6.4 Stabilizing Off-Policy Training Module
Off-policy training is prone to instability and may collapse. Prior work shows that low-probability tokens destabilize training by amplifying gradient magnitudes through unbounded likelihood ratios. Motivated by this, EMPO2 introduces a masking mechanism that suppresses the advantage term for tokens whose probability under πθ falls below a threshold δ. The loss in Eq. 1 is modified as shown in Eq. 2.

6.5 Intrinsic Rewards for Exploration Module
Inspired by prior work on exploration-targeted online RL, EMPO2 introduces an intrinsic reward based on the novelty of the current state. A memory list stores distinct states, and for each new state, its cosine similarity with existing entries is computed. If the similarity falls below a threshold, the state is added to memory and assigned a reward. The intrinsic reward is defined as rintrinsic = 1/n, where n denotes the number of similar past states. This mechanism encourages the agent to explore novel states even when no extrinsic reward is provided by the environment and maintains policy entropy.

7. All Mathematical Formulas and Symbol Definitions
7.1 Trajectory Definition
A rollout trajectory is defined as the sequence of states, actions, and rewards: τ = (u, a1, r1, s1, a2, r2, ..., sT).

7.2 Return and Relative Advantage
Given a task u, the policy πθ generates N rollout trajectories {τ (1), ..., τ (N)}. Each trajectory receives a return {R(1), ..., R(N)}, defined as the sum of rewards along the trajectory: R(i) = ∑T t=1 r(i)t. For each action a(i)t taken in trajectory τ (i), its relative advantage is defined as:
A(a(i)t) = (R(i)− (1/N)∑N j=1 R(j)) / σ(R)
where actions from trajectories with higher-than-average reward obtain positive advantage, while those from lower-performing ones obtain negative advantage.

7.3 GRPO Loss Function (Equation 1)
The GRPO loss is:
E u∼p(U) {τ(i)}N i=1∼πθold [ 1/(NT) ∑N i=1 ∑T t=1 min(ρθ(a(i)t)A(a(i)t), clip(ρθ(a(i)t), 1−ϵ, 1+ϵ)A(a(i)t))] − β DKL(πθ(·|u) ∥ πref(·|u))
where ρθ(a(i)t) = πθ(a(i)t | s(i)t, u) / πθold(a(i)t | s(i)t, u), with β ≥ 0 controlling the regularization strength toward a reference policy πref.

7.4 Importance Sampling Ratios
Regular On-Policy: Current Log Prob = log πθ(at | st, u), Old Log Prob = log πθold(at | st, u).
On-Policy w/ Tips: Current Log Prob = log πθ(at | st, u, tipst), Old Log Prob = log πθold(at | st, u, tipst).
Off-Policy: Current Log Prob = log πθ(at | st, u), Old Log Prob = log πθold(at | st, u, tipst).

7.5 Masked EMPO2 Loss Function (Equation 2)
The loss in Eq. 1 is modified as:
E u∼p(U) {τ(i)}∼πθold [ 1/(NT) ∑N i=1 ∑T t=1 min(ρ(i,t)θ A(a(i)t), clip(ρ(i,t)θ, 1−ϵ, 1+ϵ)A(a(i)t)) · 1(πθ(a(i)t | s(i)t, u)≥δ)] − β DKL(πθ(·|u) ∥ πref(·|u))
where 1(·) is the indicator function that suppresses the advantage term for tokens whose probability under πθ falls below a threshold δ.

7.6 Intrinsic Reward
rintrinsic = 1/n, where n denotes the number of similar past states.

8. Algorithm Pseudocode
Algorithm 1 EMPO2: Exploratory Memory-Augmented On- and Off-Policy Optimization
1: Inputs: Initial policy πθold , memory buffer M, task distribution p(U), group size N , batch size B, max episode length T
2: for each training iteration do
3:   {// Multi-step rollout}
4:   Sample B tasks u ∼ p(U) and initialize N identical environments (total B × N )
5:   Sample mrollout ∼ {Prompting Without Memory : p, Memory-Augmented Prompting : 1− p}
6:   Initialize state s(i)0 ← u(i) for i = 0, . . . , B ×N − 1
7:   for t = 0 to T − 1 do
8:     for i = 0 to B ×N − 1 do
9:       if mrollout = Memory-Augmented Prompting then
10:        tipst ← Retr(s(i)t ;M)
11:        Sample a(i)t ∼ πoldθ (· | s(i)t , tipst, u(i))
12:      else
13:        Sample a(i)t ∼ πoldθ (· | s(i)t , u(i))
14:      end if
15:      Execute a(i)t , observe r(i)t , s(i)t+1
16:    end for
17:  end for
18:  for i = 0 to B ×N − 1 do
19:    Sample tips ∼ πoldθ (· | s(i), u(i), tip-generation prompt)
20:    Append tips to M
21:  end for
22:  {// Policy update}
23:  if mrollout = Memory-Augmented Prompting then
24:    Sample mupdate ∼ {On-Policy : q, Off-Policy : 1− q}
25:    if mupdate = Off-Policy then
26:      for i = 0 to B ×N − 1 do
27:        log πθold(a | s(i)t , tipst, u(i))← log πθold(a | s(i)t , u(i))
28:      end for
29:    end if
30:  end if
31:  Update policy θ using the loss function in Eq. 2.
32: end for