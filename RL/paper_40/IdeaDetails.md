1. Research Background and Existing Pain Points
Large Language Models (LLMs) can significantly improve their reasoning capabilities by interacting with external tools, a paradigm known as Tool-Integrated Reasoning (TIR). Training LLMs from pretrained models using only outcome-based rewards, a paradigm known as Zero Reinforcement Learning (Zero RL), represents a fundamental step towards autonomous agents and holds the potential to unlock more general problem-solving capabilities and foster the emergence of diverse, novel reasoning patterns. A critical frontier for realizing this paradigm’s potential is enabling LLMs to perform multi-turn TIR, where models iteratively reason, generate code, execute it, and use the output for informed reasoning in subsequent turns. This directly addresses inherent LLM limitations such as poor computational accuracy and knowledge cutoffs by allowing them to leverage external tools like a Python interpreter. 

However, extending TIR to multi-turn scenarios using Reinforcement Learning is often hindered by training instability and performance collapse. One common solution is to “cold start” the model with distilled TIR trajectories to enhance stability. However, this SFT-based approach fundamentally violates the core philosophy of Zero RL, constraining the model to human-annotated patterns and thus preventing the discovery of novel reasoning strategies. The existing pain points in naive multi-turn TIR training include: 1) Frequent instability and gradient explosion issues during RL training, which prevent models from learning effective multi-turn strategies; 2) Performance collapse where naive multi-turn training not only suffers from unstable dynamics and catastrophic gradient norm explosions but also fails to match the performance of the baseline without TIR; 3) Misaligned credit assignment, where a sparse, terminal reward for a trajectory that fails in its final turns unfairly penalizes correct, high-probability reasoning in early turns, causing the policy to collapse toward safer, single-turn generations.

2. Core Research Motivation and Scientific Questions
The core research motivation is to realize the full potential of Zero RL for multi-turn TIR by directly addressing the frequent instability and gradient explosion issues without relying on a “cold-start” SFT phase. The authors aim to identify the root cause of this instability and develop a general, plug-and-play solution that stabilizes training, enables significant performance gains, and allows models to discover diverse and sophisticated reasoning patterns such as self-correction and cross-validation. 

The scientific questions addressed are: What is the root cause of training instability and gradient explosions in multi-turn TIR under the Zero RL setting? How do low-probability tokens emerge from distributional drift and compromise Zero-RL training through gradient explosion and misaligned credit assignment? How can we effectively identify problematic trajectories and block the harmful, high-magnitude gradients to stabilize learning dynamics while ensuring successful early turns are not penalized for a later collapse?

3. Overall Core Idea and Design Philosophy
The overall core idea lies in identifying the root cause of instability in multi-turn TIR: the emergence and accumulation of extremely low-probability tokens stemming from a distributional drift. When external tool feedback is used as model input in multi-turn TIR, such input may deviate significantly from the LLM’s pretrained data distribution. Although the tool feedback itself is masked when computing the policy loss, the model’s subsequent generations inherit this distributional shift, leading to increased stochasticity and the sampling of low-probability tokens. This issue compounds in the multi-turn loop, as these low-probability tokens are fed back as input, exacerbating the distributional shift in subsequent turns and culminating in collapsed, nonsensical responses.

The design philosophy is to target a specific, robust heuristic indicator of these pathological trajectories: "void turns." A void turn is defined as an LLM response that contains neither a complete code block nor a final answer. Void turns are often symptoms of distributional drift, where high generation stochasticity leads to a premature end-of-sequence token. Because void turns are rare in successful trajectories but indicative of pathological ones, they serve as a powerful heuristic for identifying problematic trajectories. SimpleTIR’s philosophy is to filter out trajectories containing void turns from the policy loss computation. This single step simultaneously prevents large gradients from low-probability tokens from backpropagating and corrects misaligned credit assignment by ensuring successful early turns are not penalized for a later collapse, all without relying on supervised fine-tuning data.

4. Core Innovation Points
1. Identification of the root cause of multi-turn TIR training instability: The paper identifies that the instability is primarily caused by a distributional drift from external tool feedback, leading to the generation of extremely low-probability tokens. Tool feedback originating from an external interpreter can deviate significantly from the LLM’s pretrained data distribution. Conditioned on such out-of-distribution input, the model’s subsequent generations drift away from pretrained patterns, becoming highly stochastic and assigning anomalously low probabilities to selected tokens, which compounds over successive turns.
2. Theoretical analysis of gradient explosion caused by low-probability tokens: The paper provides a formal analysis of the policy gradient with respect to the pre-softmax logits. Proposition 1 reveals that the gradient norm is highly sensitive to two factors exacerbated by low-probability tokens: the unclipped importance ratio (which explodes when the old policy assigns a very low probability) and a sustained high gradient norm probability-dependent term.
3. Identification of the "void turn" phenomenon as a robust filtering criterion: The paper observes that the accumulation of low-probability tokens and high generation stochasticity frequently results in a "void turn"—an LLM response that contains neither a complete code block nor a final answer. Void turns serve as a powerful heuristic indicator for identifying problematic trajectories with low-probability tokens and distributional drift.
4. The SimpleTIR trajectory filtering algorithm: A plug-and-play algorithm that identifies and filters out trajectories containing void turns when performing the policy update. SimpleTIR effectively blocks the harmful, high-magnitude gradients associated with problematic low-probability sequences, directly addressing the gradient explosion issue, and corrects misaligned credit assignment.
5. Emergence of diverse reasoning behaviors without SFT constraints: By avoiding the constraints of supervised fine-tuning, SimpleTIR encourages the model to discover diverse and sophisticated reasoning patterns, such as self-correction, cross-validation, and progressive reasoning, which are constrained in cold-start SFT methods.

5. Overview of the Overall Technical Solution
The overall technical solution models the multi-turn TIR process as a Hierarchical Markov Decision Process (HMDP), which separates decision-making into two levels: a high-level policy governing the sequence of conversational turns and a low-level policy for generating tokens within each turn. A single, unified policy is trained using Group Relative Policy Optimization (GRPO) with feedback token masking to ensure correct credit assignment by excluding feedback tokens from the gradient computation. 

During training, trajectories are sampled. The training instability caused by distributional drift from external tool feedback leads to low-probability tokens and gradient explosions. SimpleTIR stabilizes this by inspecting each sampled trajectory for "void turns." If any turn contains neither a complete code block nor a final answer, the entire trajectory is masked from the policy loss, removing it from the batch before the GRPO update. To further enhance training stability and efficiency, the solution adopts several key practices: avoiding out-of-distribution special tokens by not using chat templates and instead prepending tool outputs with "Code Execution Result:"; prepending every LLM-generated code block with a `final_answer` function to allow early termination; and strictly halting LLM generation after a complete code block is produced to append true, external tool feedback.

6. Detailed Module Design
Module 1: Hierarchical MDP Formulation for Multi-Turn TIR
End-to-end training of multi-turn TIR agents with RL is modeled as a Hierarchical Markov Decision Process, separating decision-making into two levels.
- A full interaction trajectory is a sequence $o = (q, l_0, f_0, \ldots, l_{K-1}, f_{K-1})$, where $l_k$ is the model’s generated response at turn $k$ and $f_k$ is the subsequent tool feedback.
- High-level MDP $M_H = \langle S_H, A_H, T_H, R_H, \gamma_H \rangle$: Operates at the turn level to govern the overall strategy. It includes the state $S_k = (q, l_0, f_0, \ldots, l_{k-1}, f_{k-1})$ representing the complete conversation history before the current turn, the action $A_k$ indicating an option or high-level sub-policy, the transition $T_H: S_{k+1} = S_k \circ (l_k, f_k)$, and the terminal reward assigned to the final trajectory based on its overall success.
- Low-level MDP $M_L = \langle S_L, A_L, T_L, R_L, \gamma_L \rangle$: Operates at the token level to execute the chosen high-level action. It includes the state $s_t = S_k \circ (a_1, \ldots, a_{t-1})$ indicating the sequence of tokens generated so far within the current turn $k$, the action $a_t \in A_L$ which is a single token selected from the model’s vocabulary, the transition $T_L: s_{t+1} = s_t \circ a_t$, and the reward $R_L = 0$. The low-level policy receives no intrinsic reward. The symbol $\circ$ denotes concatenation, and the discount factor is set as $\gamma = \gamma_H = \gamma_L = 1$. A single, unified policy $\pi_\theta(a_t|s_t)$ is trained to implicitly solve this two-level problem.

Module 2: Joint Policy Optimization and Feedback Masking
The unified policy $\pi_\theta$ is trained using Group Relative Policy Optimization (GRPO). GRPO circumvents the need for a learned value function by calculating the advantage based on the relative performance within a group of $G$ trajectories sampled from the same prompt. The advantage for trajectory $o_i$ is $\hat{A}_i = r_i - \text{mean}(\{r_j\}_{j=1}^G)$, where $r_i$ is the terminal reward from the high-level MDP. A critical adaptation for the TIR setting is feedback token masking, where the policy is only responsible for generating response tokens, not the environment-provided feedback tokens. The loss is accumulated only over the timesteps corresponding to the agent’s actions, using a binary mask $m_{i,t}$ that is 1 if the token at step $t$ belongs to any response $l_k$ and 0 otherwise.

Module 3: Void Turn Identification and Trajectory Filtering
This module stabilizes training by identifying and filtering out trajectories containing void turns. A void turn is defined as an LLM response that contains neither a complete code block nor a final answer. These turns are symptoms of distributional drift and high generation stochasticity leading to a premature end-of-sequence token. During the policy update, for each sampled trajectory, the module inspects its turns. If any turn contains neither a complete code block nor a final answer, it is labeled a void turn. The module then masks the policy loss for the entire trajectory, removing it from the batch before the GRPO update. This prevents large gradients from low-probability tokens from backpropagating and corrects misaligned credit assignment by ensuring successful early turns are not penalized for a later collapse.

Module 4: Implementation Details
To enhance training stability and efficiency, several key practices are adopted: 1) To avoid out-of-distribution special tokens when using base models, chat templates are not used. Instead, tool outputs are prepended with a simple prefix, “Code Execution Result:”. 2) To provide a shortcut for simple tasks and improve sample efficiency, every LLM-generated code block is prepended with a `final_answer` function, allowing the model to terminate and answer within a single turn if possible. 3) To prevent the model from hallucinating tool outputs, LLM generation is strictly halted after a complete code block is produced and the true, external tool feedback is always appended before the next turn begins.

7. All Mathematical Formulas and Symbol Definitions
- Trajectory sequence: $o = (q, l_0, f_0, \ldots, l_{K-1}, f_{K-1})$, where $l_k$ is the model’s generated response at turn $k$ and $f_k$ is the subsequent tool feedback.
- High-level MDP: $M_H = \langle S_H, A_H, T_H, R_H, \gamma_H \rangle$.
- High-level state: $S_k = (q, l_0, f_0, \ldots, l_{k-1}, f_{k-1})$.
- High-level action: $A_k$ indicating an option, or high-level sub-policy.
- High-level transition: $T_H: S_{k+1} = S_k \circ (l_k, f_k)$, where $\circ$ denotes concatenation.
- Low-level MDP: $M_L = \langle S_L, A_L, T_L, R_L, \gamma_L \rangle$.
- Low-level state: $s_t = S_k \circ (a_1, \ldots, a_{t-1})$.
- Low-level action: $a_t \in A_L$ which is a single token.
- Low-level transition: $T_L: s_{t+1} = s_t \circ a_t$.
- Low-level reward: $R_L = 0$.
- Discount factor: $\gamma = \gamma_H = \gamma_L = 1$.
- Unified policy: $\pi_\theta(a_t|s_t)$.
- GRPO Advantage: $\hat{A}_i = r_i - \text{mean}(\{r_j\}_{j=1}^G)$, where $r_i$ is the terminal reward.
- Feedback mask: $m_{i,t}$ is 1 if the token at step $t$ belongs to any response $l_k$ and 0 otherwise.
- Training objective $J_{TIR}(\theta)$:
$$J_{TIR}(\theta) = \mathbb{E}_{q \sim q_0, \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(\cdot|q)} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{\sum_t m_{i,t}} \sum_{t=1}^{|o_i|} m_{i,t} \cdot L_{CLIP}(\theta, i, t) \right]$$
- Clipped surrogate objective: $L_{CLIP}(\theta, i, t) = \min\left(\rho_{i,t}(\theta)\hat{A}_i, \text{clip}(\rho_{i,t}(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_i\right)$.
- Importance sampling ratio: $\rho_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t}|o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|o_{i,<t})}$.
- Proposition 1: Consider a token $c$ at timestep $t$ of a trajectory $o_i$. The L2 norm of the policy gradient with respect to the logits $z_t$ is:
$$\|\nabla_{z_t} J_{TIR}\|_2 = \frac{m_{i,t}}{\sum_j m_{i,j}} \cdot \rho_{i,t}(\theta) \cdot g_{i,t} \cdot |\hat{A}_i| \cdot \sqrt{1 - 2P(c) + \sum_{j \in A} P(j)^2}$$
where $m_{i,t}$ is the feedback mask, $\rho_{i,t}(\theta)$ is the importance ratio, $|\hat{A}_i|$ is the absolute advantage, $P$ is the policy’s probability distribution $\pi_\theta(\cdot|o_{i,<t})$, and $g_{i,t}$ is a gating function active when the PPO update is not clipped.

8. Algorithm Pseudocode
Based on the procedural description in Section 3.3 and Figure 4, the SimpleTIR algorithm is executed as follows:

```
Algorithm: SimpleTIR Training Step
Input: Old policy π_θ_old, current policy π_θ, batch of prompts Q, group size G
Output: Updated policy parameters θ

1: for each prompt q in Q do
2:     Sample a group of G trajectories {o_1, o_2, ..., o_G} from π_θ_old(·|q)
3:     for each trajectory o_i = (q, l_0, f_0, ..., l_{K-1}, f_{K-1}) do
4:         Calculate terminal reward r_i based on overall success
5:         Initialize trajectory mask flag is_valid = True
6:         for each turn k from 0 to K-1 do
7:             Inspect the model's generated response l_k
8:             if l_k contains neither a complete code block nor a final answer then
9:                 Label l_k as a "void turn"
10:                Set is_valid = False
11:                break
12:            end if
13:        end for
14:        if is_valid == False then
15:            Mask the policy loss for the entire trajectory o_i 
            (set m_{i,t} = 0 for all t in o_i)
16:        end if
17:    end for
18: end for
19: Compute GRPO advantage Â_i = r_i - mean({r_j}_{j=1}^G) for all valid trajectories
20: Compute the training objective J_TIR(θ) using only valid, unmasked trajectories
21: Update policy parameters θ by optimizing J_TIR(θ)
22: return θ
```