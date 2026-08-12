1. Research Background and Existing Pain Points

**Research Background:**
Large Language Models (LLMs) rely on two critical post-tuning paradigms to enhance performance: Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL). SFT is an off-policy paradigm that adjusts the policy $\pi_\theta$ to mimic a static dataset of expert demonstrations $D_{SFT} = \{(x_i, y_i^*)\}_{i=1}^N$. It relies on high-quality expert trajectories to effectively mimic response patterns. RL is an on-policy paradigm that optimizes the policy $\pi_\theta$ by maximizing expected reward $R(\tau)$ from generated trajectories $\tau = (x, y^*)$, encouraging LLMs to actively explore for better generalization through learning from direct feedback.

**Existing Pain Points:**
1.  **Limitations of SFT:** SFT is sensitive to the quality and quantity of expert data. It may struggle to generalize beyond mere memorization and is vulnerable to exposure bias, as the model trained exclusively on ground-truth expert data struggles to navigate self-generated contexts encountered during inference.
2.  **Limitations of RL:** RL exploration can be inefficient, leading to policy degradation caused by entropy collapse or over-exploitation of suboptimal strategies.
3.  **Suboptimal SFT-then-RL Paradigm:** The sequential SFT-then-RL paradigm does not consistently outperform pure RL. Training on expert data that significantly diverges from the model's established patterns leads to a **"shift-readapt-overfit" progression**, consisting of three distinct phases:
    *   **Policy Shift:** The performance initially declines since the model is forced to follow off-policy expert demonstrations whose response patterns are significantly different, disrupting established patterns and causing performance drop, exacerbated by exposure bias.
    *   **Readapt:** As SFT continues, the model policy $\pi_\theta$ integrates the expert's response patterns, mitigating exposure bias and allowing performance to rise.
    *   **Overfit:** Extended training on limited expert data leads to overfitting, resulting in generalization decline, significant loss of output diversity, and restricted exploratory capacity crucial for subsequent RL optimization.
4.  **Fragility of Binary Transition:** The SFT-then-RL paradigm demands careful timing for the transition, and the rigid binary switch (from $\mu=1$ to $\mu=0$) is fragile when expert data’s response patterns significantly diverge from the model's established response patterns.

2. Core Research Motivation and Scientific Questions

**Core Research Motivation:**
The fundamental motivation is to address the suboptimal performance and fragility of the sequential SFT-then-RL paradigm. When employing a separate SFT process to integrate off-policy expert knowledge into models with established policies, the off-policy data can disrupt established response patterns. The goal is to harmonize off-policy expert data with on-policy exploration effectively, ensuring training stability and preventing overfitting and disruption.

**Scientific Questions:**
1.  How to unify SFT and RL to avoid the "shift-readapt-overfit" progression caused by separate training stages?
2.  How to control the influence of off-policy expert data at both the holistic level (preventing global disruption) and the granular level (preventing token-level disruption and entropy collapse)?
3.  How to design a weighting mechanism that allows the model to selectively absorb expert knowledge as guidance for exploration rather than merely as a target for imitation?

3. Overall Core Idea and Design Philosophy

**Overall Core Idea:**
The core idea is to unify SFT and RL through the lens of off-policy versus on-policy learning. SFT is reframed not as a separate tuning stage, but as a dynamically weighted auxiliary objective within the on-policy RL process. This is achieved through a dual-control mechanism: a global coefficient $\mu$ for holistic guidance and a fine-grained weighting function $\phi(\cdot)$ for granular stability.

**Design Philosophy:**
1.  **Off-policy as Auxiliary:** Instead of forcing the model to imitate the expert first, expert data is used as a concurrent auxiliary loss to the on-policy RL objective.
2.  **Dynamic Transition:** A global coefficient $\mu$ is decayed over time to create a smooth transition from off-policy imitation to on-policy optimization, avoiding the rigid switch.
3.  **Selective Absorption (Uncertainty-based Weighting):** The model should selectively absorb expert knowledge. A token-wise weighting function is designed to focus on the "learning sweet spot"—tokens where the policy is most uncertain. It down-weights tokens that are highly probable (to prevent entropy collapse) or extremely improbable (to avoid disruption).

4. Core Innovation Points

1.  **Identification of the "Shift-Readapt-Overfit" Progression:** A systematic and in-depth analysis of the training dynamics when employing a separate SFT process on models with established policies, revealing how off-policy data disrupts established response patterns through a three-stage progression.
2.  **Unified Framework (CHORD):** A novel framework that unifies SFT and RL via a dynamically weighted auxiliary loss, reframing SFT as part of the on-policy RL process rather than a separate stage.
3.  **Global Control Coefficient $\mu$:** A global coefficient $\mu$ with a decay schedule is designed to holistically guide the transition from off-policy imitation to on-policy optimization, bridging the distributional gap and replacing the rigid binary switch of SFT-then-RL.
4.  **Token-wise Fine-grained Weighting $\phi(\cdot)$:** A token-wise weighting function $\phi(p_t) = p_t(1-p_t)$ is introduced to enhance stability. It down-weights tokens at both ends of the probability spectrum (highly probable to prevent entropy collapse, extremely improbable to avoid disruption), creating a "learning sweet spot" focused on tokens where the policy is most uncertain.

5. Overview of the Overall Technical Solution

The proposed framework, CHORD (Controllable Harmonization of On- and Off-Policy Reinforcement Learning via Dynamic Weighting), unifies SFT and RL via a dynamically weighted auxiliary loss.
1.  **Data Source:** The framework operates on two data sources: an RL dataset (prompts) for on-policy rollouts and an SFT dataset (prompts with expert trajectories) for off-policy learning.
2.  **Dual-Control Mechanism:**
    *   **Global Level ($\mu$):** The training begins with a large $\mu$ value, encouraging the model to learn more from off-policy expert data. As training progresses, $\mu$ gradually decays to a smaller value, shifting the training focus towards on-policy exploration.
    *   **Granular Level ($\phi$):** The SFT loss is modified by a token-wise weighting function $\phi(\cdot)$ that stabilizes the integration of off-policy data.
3.  **Loss Function:** The final objective function combines the GRPO loss (on-policy) and the modified SFT loss (off-policy) using the coefficient $\mu$.
4.  **Learning Dynamics:** This design promotes a harmonious integration of learning from both off-policy expert demonstrations and the model’s on-policy exploration, mitigating the "shift-readapt-overfit" progression.

6. Detailed Module Design

**Module 1: On-Policy Reinforcement Learning (GRPO)**
This module optimizes the policy $\pi_\theta$ using Group Relative Policy Optimization (GRPO). It samples $K$ responses $\{\tau_1, \ldots, \tau_K\}$ from a policy $\pi_{sample}$ for a given prompt $x$. Each response is evaluated with a reward function $R(\tau_k)$. The advantage $A_k$ for each response is computed by group normalization. The policy is updated to maximize a PPO-style clipped surrogate objective.
*   **Advantage Calculation:** $A_k = \frac{R(\tau_k)-\mu_R}{\sigma_R+\epsilon_z}$, where $\mu_R$ and $\sigma_R$ are the mean and standard deviation of rewards within the group, and $\epsilon_z$ is a small constant.
*   **IS Ratio:** $r_{i,k,t}(\theta) \triangleq \frac{\pi_\theta(\tau_{i,k,t}|x,\tau_{i,k,<t})}{\pi_{sample}(\tau_{i,k,t}|x,\tau_{i,k,<t})}$.

**Module 2: Global Coefficient $\mu$ (CHORD-$\mu$)**
This module controls the overall influence of off-policy expert data. It combines the RL and SFT losses into a hybrid loss function using a hyperparameter $\mu \in [0, 1]$.
*   **Hybrid Loss:** $L_{Hybrid}(\theta) = (1-\mu)L_{GRPO}(\theta) + \mu L_{SFT}(\theta)$.
*   **Decay Schedule:** To manage the "shift-readapt-overfit" progression, a decay schedule for $\mu$ is applied. The training begins with a large $\mu$ value (e.g., 0.9) and gradually decays to a smaller value (e.g., 0.05), creating a smooth transition from off-policy supervision to on-policy exploration.

**Module 3: Token-wise Weighting $\phi(\cdot)$ (CHORD-$\phi$)**
This module enhances the stability of off-policy learning by differentiating tokens based on their generation probabilities $\pi_\theta(y_t^*|x, y_{<t}^*)$.
*   **Problem with IS:** Importance Sampling (IS) prevents disruptive shifts by down-weighting low-probability tokens but aggressively reinforces existing high-probability tokens, causing entropy collapse (overconfidence) and limiting exploration.
*   **Solution:** A weighting function $\phi(y_t^*; \pi_\theta)$ is proposed to down-weight learning signals for tokens at both ends of the probability spectrum.
*   **Mechanism:** The weight is defined as $\phi(y_t^*; \pi_\theta) = p_t(1-p_t)$, where $p_t = \pi_\theta(y_t^*|x, y_{<t}^*)$. This forms a parabolic curve peaking at $p_t = 0.5$ and decaying to zero as $p_t \to 0$ or $1$.
*   **Rationale:** From an information-theoretic perspective, $p_t(1-p_t)$ measures the policy's uncertainty for the binary event of generating token $y_t^*$. This biases learning towards tokens where the policy is most uncertain ("learning sweet spot"), preventing disruption from extremely unlikely tokens and entropy collapse from highly likely ones.

**Module 4: Final Objective Function**
The final objective function of CHORD applies the global coefficient $\mu$ and the fine-grained weighting function $\phi(\cdot)$. The standard SFT loss in the hybrid function is replaced by the modified SFT loss $L_{SFT-\phi}$.

7. All Mathematical Formulas and Symbol Definitions

*   **Symbol Definitions:**
    *   $\pi_\theta$: Policy parameterized by $\theta$.
    *   $D_{SFT} = \{(x_i, y_i^*)\}_{i=1}^N$: Static dataset of N expert demonstrations.
    *   $x_i$: Prompt.
    *   $y_i^* = (y_{i,1}^*, \ldots, y_{i,|y_i^*|}^*)$: Expert response with $|y_i^*|$ tokens.
    *   $B$: Mini-batch size for SFT.
    *   $\hat{B}$: Number of prompts in the mini-batch for RL.
    *   $K$: Number of responses sampled per prompt.
    *   $\tau_k$: Generated trajectory.
    *   $R(\tau_k)$: Reward function.
    *   $\epsilon$: Clipping hyper-parameter.
    *   $A_k$: Advantage for each response.
    *   $\mu_R, \sigma_R$: Mean and standard deviation of rewards within the group.
    *   $\epsilon_z$: Small constant for stability.
    *   $\mu \in [0, 1]$: Hyperparameter governing the trade-off between SFT and RL.
    *   $sg(\cdot)$: Stop-gradient operator.
    *   $p_t$: Policy probability for expert token, $p_t = \pi_\theta(y_t^*|x, y_{<t}^*)$.

*   **Formulas:**
    *   **Equation 1 (SFT Loss):**
        $$L_{SFT}(\theta) = - \frac{1}{\sum_{i=1}^B |y_i^*|} \sum_{i=1}^B \sum_{t=1}^{|y_i^*|} \log \pi_\theta(y_{i,t}^*|x_i, y_{i,<t}^*)$$

    *   **Equation 2 (GRPO Loss):**
        $$L_{GRPO}(\theta) = - \frac{1}{\sum_{i=1}^{\hat{B}} \sum_{k=1}^K |\tau_{i,k}|} \sum_{i=1}^{\hat{B}} \sum_{k=1}^K \sum_{t=1}^{|\tau_{i,k}|} \min (r_{i,k,t}(\theta)A_{i,k}, \text{clip}(r_{i,k,t}(\theta), 1-\epsilon, 1+\epsilon)A_{i,k})$$
        where $r_{i,k,t}(\theta) \triangleq \frac{\pi_\theta(\tau_{i,k,t}|x,\tau_{i,k,<t})}{\pi_{sample}(\tau_{i,k,t}|x,\tau_{i,k,<t})}$ and $A_k = \frac{R(\tau_k)-\mu_R}{\sigma_R+\epsilon_z}$.

    *   **Equation 3 (Hybrid Loss):**
        $$L_{Hybrid}(\theta) = (1-\mu)L_{GRPO}(\theta) + \mu L_{SFT}(\theta)$$

    *   **Equation 4 (SFT Loss with Importance Sampling):**
        $$L_{SFT-IS}(\theta) = \mathbb{E}_{(x,y^*)\sim D_{SFT}} \left[ -\sum_{t=1}^{|y^*|} \text{sg}\left(\frac{\pi_\theta(y_t^*|x,y_{<t}^*)}{\pi_{sample}(y_t^*|x,y_{<t}^*)}\right) \cdot \log \pi_\theta(y_t^*|x,y_{<t}^*) \right]$$

    *   **Equation 5 (Token-wise Weighting Function):**
        $$\phi(y_t^*; \pi_\theta) = p_t(1-p_t)$$
        where $p_t = \pi_\theta(y_t^*|x, y_{<t}^*)$.

    *   **Equation 6 (SFT Loss with $\phi$ Weighting):**
        $$L_{SFT-\phi}(\theta) = -\mathbb{E}_{(x,y^*)\sim D_{SFT}} \left[ \sum_{t=1}^{|y^*|} \phi(y_t^*;\pi_\theta) \cdot \log \pi_\theta(y_t^*|x,y_{<t}^*) \right]$$

8. Algorithm Pseudocode

The original text does not provide a dedicated pseudocode block, but the training logic is described in Section 3 and Figure 3. The training procedure is summarized as follows based on the logical sequence:

1.  **Input:** Policy model $\pi_\theta$, SFT dataset $D_{SFT}$, RL prompt dataset $D_{RL}$, max steps $T$, decay schedule for $\mu$.
2.  **Initialize:** Model parameters $\theta$.
3.  **For** step $= 1$ to $T$ **do**:
    *   **On-Policy Rollout:** Sample a mini-batch of prompts from $D_{RL}$. For each prompt, generate $K$ responses using $\pi_{sample}$.
    *   **Reward Calculation:** Compute rewards $R(\tau_k)$ for each response $\tau_k$.
    *   **Advantage Calculation:** Compute advantages $A_k$ based on group rewards.
    *   **GRPO Loss:** Compute $L_{GRPO}(\theta)$ using Equation 2.
    *   **Expert Sampling:** Sample a mini-batch of expert data $(x, y^*)$ from $D_{SFT}$.
    *   **Token-wise Weighting:** For each token $y_t^*$ in the expert data, compute probability $p_t = \pi_\theta(y_t^*|x, y_{<t}^*)$ and weight $\phi(y_t^*; \pi_\theta) = p_t(1-p_t)$.
    *   **Modified SFT Loss:** Compute $L_{SFT-\phi}(\theta)$ using Equation 6.
    *   **Coefficient Update:** Update $\mu$ based on the decay schedule.
    *   **Hybrid Loss:** Compute total loss $L(\theta) = (1-\mu)L_{GRPO}(\theta) + \mu L_{SFT-\phi}(\theta)$.
    *   **Update:** Update policy parameters $\theta$ using gradient descent on $L(\theta)$.
4.  **End For**