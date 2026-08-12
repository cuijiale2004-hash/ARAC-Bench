# Research Idea and Implementation Plan Extraction

## 1. Research Background and Existing Pain Points

**Research Background:**
Large language models (LLMs), empowered by scaling laws and vast pretraining corpora encompassing virtually all publicly available text, have demonstrated impressive language understanding and reasoning abilities. Among these abilities, LLMs are believed to acquire latent safety understanding, i.e., intrinsic self-awareness of safety that distinguishes harmful from benign content, given the widespread presence of safety-relevant knowledge in their training data. Empirical evidence supports this intuition: at the prompt level, advanced LLMs can detect their own unsafe generations; at the representation level, they exhibit distinct internal activation patterns for benign inputs, harmful queries, and jailbreak attacks.

**Existing Pain Points:**
Despite the inherent safety understanding and continued efforts toward safety alignment, current models still exhibit critical safety vulnerabilities:
1.  **Vulnerability to Attacks:** They remain susceptible to jailbreak attacks and can be manipulated into revealing harmful content.
2.  **Over-refusal:** They often over-refuse benign or legitimate prompts.
3.  **Utility Degradation:** They frequently suffer degradation in general utility following safety alignment tuning.
4.  **Shallow Alignment:** Current post-training safety alignment methods fail to fully leverage the model's safety self-awareness. Supervised fine-tuning (SFT) and preference-based approaches (RLHF, DPO) often induce shallow alignment, in which models tend to memorize specific trigger patterns to refuse, known as refusal shortcuts, rather than reasoning through the underlying safety principles.
5.  **Intensive Supervision:** Reasoning-based alignment techniques typically depend on intensive supervision or complex reward signals derived from handcrafted safety specifications, limiting scalability and generalization. They fail to fully leverage the model's intrinsic safety self-awareness.

## 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:**
Achieving deep safety alignment requires incentivizing the model's latent safety awareness, moving beyond both superficial refusal strategies and heavy reliance on safety reasoning examples. The motivation is to explore the potential of LLMs to develop safety reasoning capabilities without relying on any supervised safety-specific reasoning data, instead focusing on safety incentivization and self-evolution.

**Scientific Questions:**
1.  Can structured reasoning alone elicit latent safety capabilities, and can reinforcement learning with verifiable rewards further incentivize this latent safety awareness without any supervised safety-specific reasoning data?
2.  How can we balance the dual objectives of reliably refusing harmful inputs and preserving utility on benign ones, breaking the inherent tension and safety-utility trade-off?
3.  To what extent can a model move beyond shallow alignment (memorizing specific refusal prefixes) to promote deeper safety reasoning that actively monitors and corrects potential harmful trajectories?

## 3. Overall Core Idea and Design Philosophy

**Overall Core Idea:**
AlphaAlign is a simple yet effective pure reinforcement learning (RL) framework with verifiable safety reward designed to incentivize latent safety awareness through proactive safety reasoning. It employs a dual-reward system: a verifiable safety reward encourages correctly formatted and explicitly justified refusals for harmful queries while penalizing over-refusals, and a normalized helpfulness reward guides high-quality responses to benign inputs.

**Design Philosophy:**
The framework is designed to incentivize the model's safety self-awareness rather than externally imposing priors (e.g., handcrafted safety policy). It uses structured prompting to enforce explicit safety reasoning derived solely from the model's own safety awareness. It adapts Reinforcement Learning with Verifiable Rewards (RLVR) to safety alignment by treating safety as a verifiable property of model outputs. Since prioritizing safety alone can lead to degraded utility, it introduces a normalized helpfulness reward to guide the model in generating high-quality responses to benign queries through relative score reshaping, ensuring that verifiable rewards and preference feedback are sufficient to drive effective alignment without the need for supervised safety rationales.

## 4. Core Innovation Points

1.  **Simplicity and Data Efficiency:** AlphaAlign demonstrates strong safety alignment performance with minimal supervision and training cost. It requires only binary safety labels (indicating whether a prompt is harmful), and fewer than 200 RL steps are sufficient to yield substantial improvements, suggesting that the model’s internal safety understanding can be incentivized rather than externally imposed via distillation.
2.  **Breaking the Safety-Utility Trade-off:** In contrast to conventional refusal pattern safety alignment methods that inevitably degrade instruction-following ability, AlphaAlign enhances refusal on harmful prompts, reduces overrefusal, and increases robustness to unseen jailbreaks while maintaining or even improving the model’s instruction-following, mathematical reasoning, and general task completion abilities.
3.  **Deep Alignment via Proactive Safety Reasoning:** Unlike prior methods that often induce shallow refusal alignment, AlphaAlign enables a deep safety alignment that proactively generates safety reasoning, evidenced by an increased presence of safety-relevant tokens and reduced reliance on generic refusal patterns. It remains safety-aware of its own generation process, actively monitoring and correcting potential harmful trajectories even when early-token cues are perturbed.
4.  **AlphaAlign-Zero Formulation:** A verifiable safety training paradigm that elicits the model’s intrinsic ability to recognize and reason about harmful queries, facilitated by structured prompting and a verifiable safety reward, proving that RL can draw out latent safety self-awareness without safety-specific supervised data.
5.  **Normalized Helpfulness Reward Design:** A balanced helpfulness reward design for benign queries that normalizes raw helpfulness scores within a batch of rollouts, applies a threshold to ensure non-negative utility rewards for non-refusal responses, and assigns zero reward to refusals. This prevents over-refusal while rewarding high-quality non-refusal responses in proportion to their relative helpfulness.

## 5. Overview of the Overall Technical Solution

The framework prompts the model to reason about the safety of a query before giving the final answer. Its behavior is aligned through two complementary rewards: a verifiable safety reward that enforces correct refusals of harmful queries and correct acceptance of benign ones, and a normalized helpfulness reward that promotes high-quality responses while preventing over-refusal.

The technical solution involves:
1.  **Task Formulation:** Extending safety alignment to reasoning-based safety alignment, where the output includes both safety reasoning and a final answer. This is further generalized to a reinforcement learning paradigm with verifiable rewards.
2.  **AlphaAlign-Zero:** Utilizing a structured output format (safety reasoning tags + answer tags) and a verifiable safety reward function based on a refusal verifier (checking for refusal patterns) and a format verifier (checking for reasoning structure).
3.  **AlphaAlign (Dual-Objective):** Introducing a helpfulness reward model for benign inputs. The raw helpfulness scores are normalized across rollouts and thresholded. Refusals on benign queries receive zero reward to discourage over-refusal.
4.  **Reinforcement Learning Algorithm:** Optimizing the model parameters using Proximal Policy Optimization (PPO) with Generalized Advantage Estimation (GAE) under the dual-objective reward formulation.

## 6. Detailed Module Design

### 6.1 Prompting and Output Structure Module
The prompt template enforces a structured output format consisting of an explicit safety reasoning component $s$ followed by a final answer $y$. The reasoning is derived solely from the model’s own safety awareness rather than relying on externally imposed priors.
*   **Template Structure:**
    "You should FIRST carefully evaluate the safety implications of this question through reasoning. Your safety reasoning MUST be enclosed within `<safety_reasoning>` and `</safety_reasoning>` tags. Your answer MUST be enclosed within `<answer> </answer>` tags. If the question is deemed unsafe, respond like `\boxed{Sorry, I can’t comply}` within `<answer> </answer>` tags, which will be extracted automatically by the `\boxed{}` tag. If the question is safe, then the content within `<answer> </answer>` tags will be shown to users. {{prompt}}"

### 6.2 Verifier Module
AlphaAlign-Zero incorporates a verifiable safety reward that provides explicit supervision based on ground-truth safety labels, utilizing two verifiers:
*   **Refusal Verifier ($V_r$):** Determines whether the final answer $y$ constitutes a refusal response. $V_r$ works by comparing $y$ against a set of predefined refusal patterns (e.g., "Sorry, I can’t comply", "I’m sorry", "I am sorry", "I apologize", "As an AI", "As a language model", "As an assistant", "I cannot", "I can’t", "I am unable to", "I’m unable to", "I am not able to", "I’m not able to", "I will not", "No, I will not", "I am not", "I’m not going to do that", "I’\m sorry", "“I’m sorry", "‘I’m sorry"), which were observed in the model’s initial responses. Its output is then compared with the ground-truth harmfulness label of the input to compute the safety reward.
*   **Format Verifier ($V_f$):** Checks whether the output includes explicit safety reasoning before the final answer and follows the required safety reasoning structure. It is implemented using a regular expression: `r"<safety_reasoning>.*</safety_reasoning>.*<answer>.*</answer>"`.

### 6.3 Reward Function Module
The reward function is divided into a verifiable safety reward and a normalized helpfulness reward.

*   **Verifiable Safety Reward ($R_s$):** For harmful queries ($x \in X_h$), the reward incentivizes the format verifier and the refusal verifier positively. For benign queries ($x \in X_b$), it incentivizes the format verifier but penalizes the refusal verifier to avoid over-refusal.
*   **Normalized Helpfulness Reward ($R_h$):** To mitigate the utility degradation from discrimination-focused optimization, a helpfulness reward model $R_r$ is introduced for benign inputs. Given a benign input $x_b$ and a set of rollouts $\{o_1, \dots, o_n\}$, raw helpfulness scores $r = \{r_1, r_2, \dots, r_n\}$ are computed. The scores are normalized as $\tilde{r}_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$. A threshold $\max(\tilde{r}_i, 0)$ is applied to ensure non-negative utility rewards for non-refusal responses. Refusals on benign queries ($V_r(y_i) = 1$) receive a reward of 0.
*   **Final Reward Function ($R$):** Harmful queries rely solely on the safety reward, while benign queries additionally benefit from the helpfulness reward.

### 6.4 Reinforcement Learning Optimization Module
The model is optimized using Proximal Policy Optimization (PPO).
*   For each input, the model generates a set of candidate responses.
*   Rewards are attached to the final token of each output.
*   Advantages are estimated using Generalized Advantage Estimation (GAE).
*   The policy $\pi_\theta$ is updated by minimizing the clipped PPO loss.
*   The value function $V_\phi$ is jointly optimized by minimizing the squared error between predicted values and empirical returns.

## 7. All Mathematical Formulas and Symbol Definitions

**Symbol Definitions:**
*   $x$: Input prompt, where $x \in X$.
*   $X_h \subset X$: Harmful inputs.
*   $X_b \subset X$: Benign inputs.
*   $\pi_\theta$: Safety-aligned LLM with parameters $\theta$.
*   $y$: Final answer generated by the model, where $y \in Y_r$ (refusal) or $y \in Y_c$ (compliant).
*   $o$: Complete output of the model, $o = (s, y)$.
*   $s$: Safety reasoning component.
*   $V_r$: Refusal verifier function.
*   $V_f$: Format verifier function.
*   $r_s$: Safety reward.
*   $r_f$: Format reward.
*   $r_a$: Refusal reward coefficient (used in $R_s$).
*   $D$: Distribution of prompts.
*   $R_r$: Helpfulness reward model.
*   $r_i$: Raw helpfulness score for rollout $i$, $r_i = R_r(x_b, y_i)$.
*   $r$: Set of raw helpfulness scores $\{r_1, r_2, \dots, r_n\}$.
*   $\tilde{r}_i$: Normalized helpfulness score.
*   $R_s$: Overall safety reward.
*   $R_h$: Overall helpfulness reward.
*   $R$: Final reward function.
*   $J(\theta)$: Expected reward objective.
*   $J_{PPO}(\theta)$: PPO loss objective.
*   $\hat{A}_t$: Estimated advantage.
*   $\epsilon$: Clipping threshold for PPO.
*   $w$: Token sequence for a target keyword $(t_1, t_2, \dots, t_m)$.
*   $N$: Maximum number of new tokens in the generation window.
*   $G$: Greedy token path $(g_0, g_1, \dots, g_{N-1})$.
*   $G(k)$: Prefix of $k$ greedily generated tokens $(g_0, g_1, \dots, g_{k-1})$.

**Mathematical Formulas:**

1.  **Safety Alignment Objective:**
    $$
    y =
    \begin{cases}
    y_r \in Y_r, & \text{if } x \in X_h, \\
    y_c \in Y_c, & \text{if } x \in X_b,
    \end{cases}
    $$

2.  **Reasoning-Based Safety Alignment Output:**
    $$
    o = \pi_\theta(x) = (s, y)
    $$

3.  **Refusal Verifier Function:**
    $$
    V_r(y) =
    \begin{cases}
    1, & \text{if } y \in Y_r, \\
    0, & \text{if } y \in Y_c,
    \end{cases}
    $$

4.  **Reinforcement Learning Objective:**
    $$
    J(\theta) = \mathbb{E}_{x \sim D} \mathbb{E}_{o \sim \pi_\theta(\cdot|x)}
    \left[
    r_s + r_f
    \right]
    $$

5.  **Verifiable Safety Reward ($R_s$):**
    $$
    R_s(x, o_i) =
    \begin{cases}
    r_f V_f(o_i) + r_a V_r(y_i), & x \in X_h \\
    r_f V_f(o_i) - r_a V_r(y_i), & x \in X_b
    \end{cases}
    $$
    (Note: The paper uses $r_a$ in the formula for the refusal reward coefficient).

6.  **Normalized Helpfulness Reward ($R_h$):**
    $$
    R_h(x_b, o_i, \{o_1, \dots, o_n\}) =
    \begin{cases}
    \max(\tilde{r}_i, 0), & \text{if } V_r(y_i) = 0, \\
    0, & \text{if } V_r(y_i) = 1,
    \end{cases}
    $$
    where $\tilde{r}_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$.

7.  **Final Reward Function ($R$):**
    $$
    R(x, o_i, \{o_1, o_2, \dots, o_n\}) =
    \begin{cases}
    R_s(x, o_i), & x \in X_h, \\
    R_s(x, o_i) + R_h(x, o_i, \{o_1, o_2, \dots, o_n\}), & x \in X_b,
    \end{cases}
    $$

8.  **PPO Loss Function:**
    $$
    J_{PPO}(\theta) = \mathbb{E}_{t, s_t, a_t \sim \pi_{\theta_{\text{old}}}}
    \left[
    \min
    \left(
    \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
    \hat{A}_t, \text{clip}
    \left(
    \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
    , 1 - \epsilon, 1 + \epsilon
    \right)
    \hat{A}_t
    \right)
    \right]
    $$

9.  **Cumulative Keyword Adoption Score (CKAS):**
    $$
    \text{CKAS}(w, x, N) = \sum_{k=0}^{N-m} P(w|x, G(k))
    $$

## 8. Algorithm Pseudocode

The paper describes the training process and logic flow without a formal pseudocode block. Based on the text and figures, the algorithmic process is extracted as follows:

**Algorithm: AlphaAlign Training Process**

**Input:** Pretrained/Instruct-tuned LLM $\pi_\theta$, Prompt dataset $D$ with binary safety labels, Helpfulness reward model $R_r$, Refusal verifier $V_r$, Format verifier $V_f$, PPO clipping threshold $\epsilon$.

1.  **Initialize:** Actor model $\pi_\theta$, Critic model $V_\phi$.
2.  **For each training iteration:**
3.  $\quad$ **Sample** a batch of prompts $x \sim D$.
4.  $\quad$ **For each prompt $x$:**
5.  $\quad\quad$ **Generate** a set of rollouts $\{o_1, \dots, o_n\}$ using current policy $\pi_\theta$, where $o_i = (s_i, y_i)$.
6.  $\quad\quad$ **Evaluate Format:** Compute $V_f(o_i)$ for each rollout using regex matching.
7.  $\quad\quad$ **Evaluate Refusal:** Compute $V_r(y_i)$ for each rollout by matching against predefined refusal patterns.
8.  $\quad\quad$ **Compute Safety Reward:** Calculate $R_s(x, o_i)$ based on $V_f$ and $V_r$ (Eq. 5).
9.  $\quad\quad$ **If $x \in X_b$ (Benign):**
10. $\quad\quad\quad$ **Compute Raw Helpfulness:** $r_i = R_r(x, y_i)$ for all rollouts.
11. $\quad\quad\quad$ **Normalize Scores:** $\tilde{r}_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$.
12. $\quad\quad\quad$ **Compute Helpfulness Reward:** $R_h(x, o_i, \{o_1, \dots, o_n\})$ based on $\max(\tilde{r}_i, 0)$ and $V_r(y_i)$ (Eq. 6).
13. $\quad\quad$ **Else If $x \in X_h$ (Harmful):**
14. $\quad\quad\quad$ Set $R_h(x, o_i, \dots) = 0$.
15. $\quad\quad$ **Compute Final Reward:** $R(x, o_i, \dots) = R_s(x, o_i) + R_h(x, o_i, \dots)$ (Eq. 7).
16. $\quad$ **Estimate Advantage:** Compute $\hat{A}_t$ using GAE based on rewards $R$ and value function $V_\phi$.
17. $\quad$ **Update Parameters:**
18. $\quad\quad$ Update $\theta$ by minimizing $J_{PPO}(\theta)$ (Eq. 8).
19. $\quad\quad$ Update $\phi$ by minimizing squared error between $V_\phi$ and empirical returns.
20. **End For**

**Output:** Aligned model $\pi_\theta$.