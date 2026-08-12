# Research Idea and Implementation Plan Extraction

## 1. Research Background and Existing Pain Points

**Research Background:**
Reinforcement Learning with Verifiable Rewards (RLVR) has recently emerged as a core paradigm for enhancing the reasoning capabilities of Large Language Models (LLMs). By rewarding reasoning paths based on the consistency between final outcomes and ground-truth answers through a deterministic verifier, RLVR incentivizes LLMs to produce more deliberate reasoning chains while effectively mitigating the risk of reward hacking. Despite its effectiveness in training, a fundamental limitation of standard RLVR is its inability to continue providing verification signals for model outputs in scenarios where ground truth answers are unavailable, such as during test-time inference.

**Existing Pain Points:**
1. **Lack of Verification Signals at Test Time:** Standard RLVR relies on ground-truth answers to compute rewards, which are absent during test-time inference. This prevents the model from evaluating the quality of its own generated solutions when it matters most for inference-time scaling.
2. **Inefficiency of External Verifiers:** The standard approach to address the lack of test-time verification involves training an external verifier (such as Outcome-supervised Reward Models, Process-supervised Reward Models, or Generative Verifiers). However, training external verifiers requires additional training on a separate LLM during or after reinforcement learning, introducing significant computational overhead and complexity.
3. **Inefficiency of Joint Optimization Approaches:** Recent works exploring the joint optimization of reasoning and self-verification capabilities within a single policy model require the LLM to sequentially generate solutions and self-verifications using two separate prompt templates. This practice doubles the inference cost per sample and significantly reduces generation efficiency.
4. **Length Bias of Implicit Reward:** A straightforward alternative is to utilize the implicit reward (the log-probability ratio between the policy and reference models) to rank different generations at test time. However, this approach has a critical drawback: the absolute value of the implicit reward is length-biased. Since the absolute value increases proportionally with the response length, and incorrect solutions are usually longer than correct solutions in reasoning tasks, the implicit reward is unreliable in evaluating solution correctness. Furthermore, directly aligning the implicit reward with the true reasoning reward during training degrades the policy model’s generation ability due to a fundamental gap (the partition function term) between the RLVR solution and reward modeling.

## 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:**
The core motivation is to endow LLMs with the ability for autonomous and continual self-improvement by jointly optimizing reasoning and self-verification capabilities within the RLVR framework, while eliminating the inefficiency inherent in prior approaches. The goal is to achieve this unification at nearly zero additional cost, allowing the model to both produce high-quality reasoning paths and evaluate its own outputs effectively at test time without requiring separate inference steps or external models.

**Scientific Questions:**
How can a model's assessment of its own solution be efficiently captured and optimized directly within the generation process? Specifically, can the verification signal be derived from the next-token probability distribution of the final token of the generated sequence, thereby replacing explicit and costly self-verification generation steps with a remarkably simple and efficient formulation?

## 3. Overall Core Idea and Design Philosophy

The overall core idea is that a model’s assessment of its own solution can be captured in the last token’s predicted probability distribution. Theoretically, the closed-form solution to the RL objective of self-verification training can be approximately reduced to a remarkably simple form: the true reasoning reward of a solution is equal to its last-token self-rewarding score, which is computed as the difference between the policy model’s next-token log-probability assigned to any pre-specified token at the solution’s last token and a pre-calculated constant, scaled by the KL coefficient.

Based on this insight, the design philosophy replaces the explicit RL optimization for self-verification with a simple Mean Squared Error (MSE) loss. The model is trained to align its last-token self-rewarding score with the true reasoning reward from the verifier. This MSE objective is added directly to the standard RLVR loss, allowing for seamless joint optimization of both the reasoning and self-rewarding capabilities of the policy model. By deriving the self-rewarding signal directly from the next-token probability distribution of the final token of the generated sequence, the method achieves a highly efficient unification of generation and self-verification, incurring only the minimal extra cost of at most one additional token inference.

## 4. Core Innovation Points

1. **Theoretical Derivation of Last-Token Self-Rewarding:** The paper theoretically reveals that the closed-form solution to the RL objective of self-verification can be approximately reduced to the last-token self-rewarding score, showing that the true reasoning reward is equal to the log-probability ratio of the policy model to the reference model for a pre-specified special token at the last response token, scaled by the KL coefficient.
2. **LaSeR Algorithm via MSE Loss:** Proposes LaSeR, an algorithm that simply augments the original RLVR loss with an MSE loss that aligns the last-token self-rewarding scores with the verifier-based reasoning rewards, jointly optimizing reasoning and self-rewarding capabilities without requiring separate verification generation steps.
3. **Simplification of Reference Log-Probability:** Observes that for a randomly chosen special token, its predicted log-probability under the reference model is practically a constant, small value across all problems and solutions. This enables the simplification of the self-rewarding score into a form that depends only on the policy model’s outputs and a pre-calculated constant ($c_{ref}$), eliminating the need for forwarding through the reference model and reducing computational cost by half.
4. **Integration of Verifier-based and Self-Rewarding-based Advantages:** The optimized self-rewarding scores serve as auxiliary reward signals in training through the integration of verifier-based and self-rewarding-based advantages, which helps mitigate the issue of misjudgments by rule-based verifiers and produces more fine-grained rewards.
5. **MSE Loss over SFT/BCE Loss:** The choice of MSE loss over SFT/BCE loss is a critical innovation. The SFT loss drives the probability of the special token toward 1 for correct solutions, causing strong interference with reasoning optimization. In contrast, the MSE loss drives the probability toward a controllable, much smaller target ($\exp(1/\beta_v) \cdot \pi_{ref}(z_c|x,y)$), exerting negligible influence on the original RLVR optimization while effectively training the self-rewarding capability.

## 5. Overview of the Overall Technical Solution

The overall technical solution begins by formulating the RL objective of verification, where the model produces a verification to identify the correctness of a candidate solution. By analyzing the partition function of the closed-form solution to this objective, the paper demonstrates that the partition function can be naturally discarded under their formulation, leading to the conclusion that the true reasoning reward is equal to the scaled log-probability ratio of the policy model to the reference model at a pre-specified token $z_c$.

Based on this theoretical foundation, the method constructs a last-token self-rewarding score, simplified by replacing the reference log-probability with a pre-calculated constant $c_{ref}$. An MSE loss is then formulated between this self-rewarding score and the verifier-based reasoning reward. To handle sample imbalance, a class-level loss re-weighting strategy is applied to the MSE loss.

During training, the method employs a separate warm-up strategy: first warming up the reasoning capability using standard RLVR, then warming up the self-rewarding capability using the MSE loss, and finally integrating the verifier-based and self-rewarding-based advantages to provide fine-grained learning signals. At inference time, the model generates a solution and computes the last-token self-rewarding score with at most one additional token inference. This score is clipped to the interval [0,1] and used to determine the self-verification outcome or perform weighted majority voting for inference-time scaling.

## 6. Detailed Module Design

**Self-Verification Formulation Module:**
This module defines the RL objective of verification. Given a problem $x$ and a candidate solution $y$, the model is required to produce a verification $z \sim \pi_\theta(\cdot|x,y)$. The verification reward measures the consistency between the true correctness of $y$ and the verification result of $z$. The label space is simplified to two single tokens $z_c$ (e.g., "Yes") and $z_i$ (e.g., "No"). By analyzing the partition function $Z(x,y)$ of the closed-form solution, it is shown that because $\pi_{ref}(z|x,y) \approx 0$ for $z \in \{z_c, z_i\}$ and under an appropriate choice of $\beta_v$, $\log Z(x,y) \approx 0$. This allows the optimal solution to be reduced to the form where the true verification reward equals the scaled log-probability ratio between the policy and reference models at $z_c$.

**Last-Token Self-Rewarding Loss Module:**
This module optimizes the model’s verification capability by directly adding an MSE loss into the original RLVR loss. The loss aligns the last-token self-rewarding score (defined as $\beta_v \log \pi_\theta(z_c|x,y) - \beta_v c_{ref}$) with the true reasoning reward $r_v(x,y)$. To prevent the score from being biased toward the class with more samples, a class-level loss re-weighting strategy is applied, using dynamic re-weighting factors calculated from the numbers of correct ($N_c$) and incorrect ($N_i$) solutions in the current batch.

**Advantage Integration Module:**
This module facilitates the training process by integrating verifier-based and self-rewarding-based advantages. The self-rewarding scores are used to compute a separate set of advantages, which are then linearly combined with the verifier-based advantages using a mixing coefficient $\tau$. To stabilize training, a filtering strategy sets $\tau = 0$ whenever the standard deviation of self-rewarding scores within a group falls below a threshold $T$.

**Warm-Up Strategy Module:**
During the initial phase of training, only the last-token self-rewarding score is optimized without integrating self-rewarding-based advantages. When training from base models, a two-stage warm-up is used: first, standard RLVR is performed without the self-rewarding loss to warm up reasoning capability; second, the self-rewarding capability is warmed up before advantage integration.

**Inference Module:**
During testing, after the model generates a solution, the last-token self-rewarding score is computed based on the policy model’s log-probability and the constant $c_{ref}$. The score is clipped into the interval [0,1]. The comparison between this score and 0.5 determines the self-verification outcome, or the score itself is used to perform weighted majority voting for inference-time scaling.

## 7. All Mathematical Formulas and Symbol Definitions

**Symbol Definitions:**
- $\pi_\theta$: Target policy model to be optimized.
- $\pi_{ref}$: Reference model from which $\pi_\theta$ is initialized.
- $D$: Query set.
- $x$: Input problem.
- $y$: Generated response.
- $r(x,y)$: Reward function.
- $D_{KL}$: Kullback–Leibler (KL) divergence loss.
- $r_v$: Deterministic verifier reward (binary {0,1}).
- $a$: Final extracted answer.
- $a^*$: Ground-truth answer.
- $A_t$: Advantage function.
- $y_t$: Token at step $t$.
- $Z(x)$: Partition function for the RL objective.
- $\pi_g$: Generator to solve the problem.
- $\hat{r}(x,y,z)$: Verification reward.
- $z$: Verification token.
- $z_c$: Pre-specified token indicating correct verification (e.g., "Yes" or unused special token).
- $z_i$: Pre-specified token indicating incorrect verification (e.g., "No").
- $\beta_v$: KL coefficient for verification.
- $r_s$: Last-token self-rewarding score.
- $c_{ref}$: Pre-calculated constant representing the mean value of $\log \pi_{ref}(z_c|x,y)$.
- $\alpha$: Loss balancing coefficient.
- $N_c$: Number of correct solutions in a batch.
- $N_i$: Number of incorrect solutions in a batch.
- $w_c$: Re-weighting factor for correct solutions.
- $w_i$: Re-weighting factor for incorrect solutions.
- $\tau$: Mixing coefficient for advantage integration.
- $T$: Threshold for filtering advantage integration.

**Mathematical Formulas:**

1. Standard RL Objective:
$$O_{\pi_\theta} = \max_{\pi_\theta} \mathbb{E}_{x \sim D, y \sim \pi_\theta(\cdot|x)} \left[ r(x,y) - \beta D_{KL}(\pi_\theta || \pi_{ref}) \right]$$

2. Deterministic Verifier Reward:
$$r_v(x,y) = \mathbb{I}\{a \equiv a^*\} = \begin{cases} 1 & \text{if } a \text{ is semantically equivalent to } a^*, \\ 0 & \text{otherwise.} \end{cases}$$

3. Policy Gradient:
$$\nabla_\theta O_{\pi_\theta} = \mathbb{E}_{x \sim D, y \sim \pi_\theta(\cdot|x)} \left[ \sum_{t=1}^T A_t \nabla_\theta \log \pi_\theta(y_t|x, y_{<t}) \right]$$

4. GRPO Advantage:
$$A_t^i = \frac{r_v^i - \text{mean}(r_v^1, \cdots, r_v^K)}{\text{std}(r_v^1, \cdots, r_v^K)}, \quad r_v^i = r_v(x, y^i)$$

5. Implicit Reward:
$$r_v(x,y) = \beta \log[\pi_\theta(y|x)/\pi_{ref}(y|x)] + \beta \log Z(x)$$
where $Z(x) = \sum_y \pi_{ref}(y|x) \exp(\frac{1}{\beta} r_v(x,y))$.

6. RL Objective of Verification:
$$V_{\pi_\theta} = \max_{\pi_\theta} \mathbb{E}_{x \sim D, y \sim \pi_g(\cdot|x), z \sim \pi_\theta(\cdot|x,y)} \left[ \hat{r}(x,y,z) - \beta_v D_{KL}(\pi_\theta || \pi_{ref}) \right]$$

7. Verification Reward:
$$\hat{r}(x,y,z) = \begin{cases} 1 & (z = z_c \text{ and } r_v(x,y) = 1) \text{ or } (z = z_i \text{ and } r_v(x,y) = 0) \\ 0 & \text{otherwise} \end{cases}$$

8. Close-form solution to Verification Objective:
$$\hat{r}(x,y,z) = \beta_v \log \frac{\pi_\theta(z|x,y)}{\pi_{ref}(z|x,y)} + \beta_v \log Z(x,y), \quad Z(x,y) = \sum_z \pi_{ref}(z|x,y) \exp(\frac{1}{\beta_v} \hat{r}(x,y,z))$$

9. Approximation of Partition Function:
$$Z(x,y) = \sum_z \pi_{ref}(z|x,y) \exp(\frac{1}{\beta_v} \hat{r}(x,y,z)) = \sum_{z \notin \{z_c, z_i\}} \pi_{ref}(z|x,y) \exp(\frac{1}{\beta_v} \hat{r}(x,y,z)) + \pi_{ref}(z_c|x,y) \exp(\frac{1}{\beta_v} \mathbb{I}_{r_v(x,y)=1}) + \pi_{ref}(z_i|x,y) \exp(\frac{1}{\beta_v} (1 - \mathbb{I}_{r_v(x,y)=1})) \approx (1 - 0 - 0) \exp(0) + 0 + 0 = 1 \Longrightarrow \log Z(x,y) \approx 0$$

10. Reduced Optimal Solution:
$$\hat{r}(x,y,z) = \beta_v \log[\pi_\theta(z|x,y)/\pi_{ref}(z|x,y)]$$

11. True Verification Reward:
$$\hat{r}(x,y, z_c) = r_v(x,y) = \beta_v \log[\pi_\theta(z_c|x,y)/\pi_{ref}(z_c|x,y)]$$

12. Self-Rewarding MSE Loss:
$$\mathcal{L} = \mathbb{E}_{x \sim D, y \sim \pi_g(\cdot|x)} \left( \beta_v \log[\pi_\theta(z_c|x,y)/\pi_{ref}(z_c|x,y)] - r_v(x,y) \right)^2$$

13. Joint Optimization Objective (LaSeR):
$$S_{\pi_\theta} = \max_{\pi_\theta} \mathbb{E}_{x \sim D, y \sim \pi_\theta(\cdot|x)} \left\{ r_v(x,y) - \beta D_{KL}(\pi_\theta || \pi_{ref}) - \alpha \left[ \beta_v \log \frac{\pi_\theta(z_c|x,y)}{\pi_{ref}(z_c|x,y)} - r_v(x,y) \right]^2 \right\}$$
where $r_s = \beta_v \log \frac{\pi_\theta(z_c|x,y)}{\pi_{ref}(z_c|x,y)}$ is the last-token self-rewarding score.

14. Self-Rewarding Loss Re-Weighting:
$$l = \frac{1}{N_c + N_i} \sum_x \sum_y [w_c \mathbb{I}_{\{r_v(x,y)=1\}} + w_i \mathbb{I}_{\{r_v(x,y)=0\}}] [\beta_v \log \pi_\theta(z_c|x,y) - \beta_v c_{ref} - r_v(x,y)]^2$$
where $w_c = \frac{N_c + N_i}{2 \times N_c}$ and $w_i = \frac{N_c + N_i}{2 \times N_i}$.

15. Advantage Integration:
$$\hat{A}_t^i = (1 - \tau) \frac{r_v^i - \text{mean}(r_v^1, \cdots, r_v^K)}{\text{std}(r_v^1, \cdots, r_v^K)} + \tau \frac{r_s^i - \text{mean}(r_s^1, \cdots, r_s^K)}{\text{std}(r_s^1, \cdots, r_s^K)}$$
where $r_v^i = r_v(x,y^i)$ and $r_s^i = \beta_v \log \pi_\theta(z_c|x,y^i) - \beta_v c_{ref}$.

16. SFT/BCE Loss (for comparison):
$$\mathcal{L}_{SFT} = -\mathbb{E}_{x \sim D, y \sim \pi_g(\cdot|x)} [r_v(x,y) \cdot \log \pi_\theta(z_c|x,y) + (1 - r_v(x,y)) \cdot \log \pi_\theta(z_i|x,y)]$$

## 8. Algorithm Pseudocode

**Algorithm 1: LaSeR: Reinforcement Learning with Last-Token Self-Rewarding**

**Input:** Initial policy model $\pi_\theta$, prompts $D$, verifier $r_v$, warm-up hyper-parameters $w_r$ and $w_{sr}$, coefficient $\beta_v$, pre-specified token $z_c$, pre-calculated $c_{ref} = \mathbb{E}_{(x,y)}[\log \pi_{ref}(z_c|x,y)]$

1. **for** Step $s = 1, \cdots, S$ **do**
2. $\quad$ Set $\pi_{old} \leftarrow \pi_\theta$;
3. $\quad$ Sample batch prompts $D_s$ from $D$;
4. $\quad$ Generate solutions $\{y_i\}_{i=1}^K$ for each $x \in D_s$;
5. $\quad$ Calculate verifier-based rewards and advantages (e.g., Eq. (4)), calculate RL loss;
6. $\quad$ **If** $s \ge w_r$, calculate last-token self-rewarding loss based on Eq. (14) and add it to RL loss;
7. $\quad$ **If** $s \ge w_{sr}$, calculate self-rewarding-based advantages and perform advantage integration based on Eq. (15);
8. $\quad$ Update the policy model $\pi_\theta$ using any RL algorithm with integrated loss and advantages;
9. **end**

**Output:** $\pi_\theta$