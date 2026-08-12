**1. Research Background and Existing Pain Points**

**Research Background:** Reinforcement Learning with Verifiable Rewards (RLVR) is a leading paradigm for enhancing the reasoning capabilities of Large Language Models (LLMs), directly optimizing models based on the correctness of their final answers. Despite the development of advanced RLVR algorithms like GRPO and DAPO, critical challenges persist in the training process.

**Existing Pain Points:** 
1. **Poor Exploration and Premature Convergence:** Current RLVR methods often explore poorly, leading to premature convergence and entropy collapse. This reveals a critical flaw where the training process is heavily biased towards exploitation, causing models to converge prematurely instead of sufficiently exploring their environment for better solutions.
2. **Poorly Calibrated Policies:** These methods tend to produce poorly calibrated policies that remain confident in their generations regardless of correctness. Early in standard training, correct responses have lower perplexity (higher confidence) than incorrect ones, but as training progresses, this gap shrinks—a phenomenon termed "calibration collapse."
3. **Limitations of Simple Exploration Heuristics:** Simple heuristics such as $\epsilon$-greedy policies and entropy bonuses either inject randomness into the environment or encourage the policy to be more stochastic. Directly applying these approaches often demonstrates debatable effectiveness in complex environments like Deep RL and LLM-based reasoning.
4. **Challenges with Principled Exploration Strategies:** Count-based and prediction-based strategies require training auxiliary modules and effective state representations. This is particularly challenging for LLMs, where efficiently representing a reasoning path into a fixed-size embedding remains an open problem. For instance, using the SimHash technique maps embeddings into discrete hash codes, but most responses collapse into a small number of hash grids, creating a highly concentrated count distribution that undermines count-based exploration.

**2. Core Research Motivation and Scientific Questions**

**Core Research Motivation:** An LLM, having been trained on vast reasoning corpora, develops a sophisticated internal model of what constitutes a familiar versus a novel reasoning pattern. This parallels early childhood development, where learning is not driven by an external summary and count of experiences, but is instead propelled by an intrinsic curiosity to explore novel situations. Therefore, there is a need to develop an exploration paradigm guided by the model’s intrinsic curiosity, moving beyond methods that rely on explicit counts or response embeddings.

**Scientific Questions:** How can intrinsic curiosity be formalized and leveraged to guide exploration efficiently and scalably in RLVR? How can the model’s uncertainty about its own actions (actor curiosity) and its long-term value estimates (critic curiosity) be quantified and integrated as exploration bonuses? How can overconfident errors be penalized while promoting diversity among correct responses? How does the critic's uncertainty relate to classical count-based exploration bonuses in reinforcement learning theory?

**3. Overall Core Idea and Design Philosophy**

**Overall Core Idea:** The paper introduces Curiosity-Driven Exploration (CDE), a systematic framework that leverages the model’s intrinsic sense of curiosity to guide exploration. CDE formalizes curiosity with signals from both the actor and the critic. For the actor, the perplexity (PPL) over its generated response serves as the curiosity measure. For the critic, the variance of its posterior value distribution (approximated via a multi-head, bootstrapped structure) measures curiosity. Both signals serve as exploration bonuses within the RLVR framework, shaping the reward and advantage functions to effectively guide exploration.

**Design Philosophy:** The design philosophy centers on utilizing the model's own internal uncertainty as a guide. For the actor, a response surprising to the actor (low probability) resides in an underexplored region. For the critic, sparse data regions result in higher uncertainty. To ensure stability, the bonuses are integrated using an adaptive clipping mechanism and dynamic weight annealing, allowing aggressive exploration in early stages and a gradual shift to exploitation as the policy converges.

**4. Core Innovation Points**

1. **Curiosity-Driven Exploration Framework (CDE):** A novel framework that leverages intrinsic curiosity signals from both the actor (perplexity) and the critic (multi-head variance) to guide exploration in RLVR, moving beyond external counts or problematic response embeddings.
2. **Actor Curiosity via PPL Bonus with Adaptive Clipping:** Introducing the logarithm of perplexity as a response-level curiosity bonus. It is integrated via an adaptive clipping mechanism $\omega \min(|r|/\kappa, \alpha B_{\text{actor}})$ that constrains the bonus to remain a fraction of the original reward, preventing reward hacking and over-exploration.
3. **Critic Curiosity via Multi-Head Bootstrapped Architecture:** Approximating the posterior distribution of value estimates using a multi-head critic with bootstrapping. The standard deviation across the $K$ heads serves as the critic curiosity bonus, reflecting higher-level uncertainty about the long-term value of a reasoning path.
4. **Theoretical Proof of Calibration Effect (Theorem 3.1):** Establishing that the actor's PPL bonus intrinsically penalizes overconfident errors while encouraging diversity among correct responses. Among correct responses, those with higher PPL receive a larger relative probability increase; among incorrect responses, those with lower PPL receive a larger relative probability decrease.
5. **Theoretical Connection to Count-Based Exploration (Theorem 3.2):** Proving that under a Linear MDP assumption, the standard deviation across multi-head critics is a consistent estimator of the classical pseudo-count exploration bonus $\sqrt{\phi^\top_{n,t} \Lambda^{-1}_{n,t} \phi_{n,t}}$, grounding the approach in established RL principles.
6. **Identification of Calibration Collapse:** Discovering and analyzing the "calibration collapse" phenomenon under standard GRPO, where the model's confidence progressively decouples from its correctness, and demonstrating that the PPL bonus mitigates this miscalibration.

**5. Overview of the Overall Technical Solution**

The overall technical solution integrates the CDE framework into standard RLVR algorithms (GRPO and PPO). 
- **For GRPO (Critic-free):** The PPL bonus is calculated for each generated response. The total response-level reward is modified by adding the PPL bonus, constrained by an adaptive clipping mechanism. The bonus weight $\omega$ follows a staircase decay schedule to balance exploration and exploitation.
- **For PPO (Actor-Critic):** The single-head critic is replaced with a multi-head architecture sharing a common LLM backbone. The standard deviation of the value estimates from these $K$ heads forms the critic bonus. This bonus is added to the modified PPO advantage estimate, again using the adaptive clipping mechanism. The critic heads are updated using bootstrapped subsets of the collected trajectories. The solution enables the policy to initially explore broadly to expand state-action coverage and then shift effectively to exploitation.

**6. Detailed Module Design**

**Actor Curiosity Module:**
- **Mechanism:** Measures actor curiosity as the actor’s uncertainty about its own actions. A response with low probability under the current policy indicates an underexplored region.
- **PPL Bonus Calculation:** The curiosity bonus is the negative average log-probability of the generated response.
- **Adaptive Clipping:** The total response-level reward combines the original reward and the PPL bonus. The clipping ratio $\kappa$ governs the maximum size of the bonus relative to the original reward. The bonus scaling factor $\alpha$ normalizes the bonus. The bonus weight $\omega$ is typically set with a staircase annealing schedule.
- **Calibration Effect:** The PPL bonus penalizes confident mistakes (incorrect responses with low PPL) and encourages novel correct responses (correct responses with high PPL).

**Critic Curiosity Module:**
- **Multi-Head Architecture:** Approximates the posterior distribution of value estimates using a multi-head critic where $K$ critics share a common LLM backbone.
- **Bootstrap Approximation:** Each head is trained on a resampled subset of the collected trajectories (sampled with replacement). The hyperparameter $\zeta \in (0, 1]$ controls the size of this subset. A smaller $\zeta$ increases head diversity, while a larger $\zeta$ improves sample efficiency.
- **Bonus Calculation:** The curiosity signal is the standard deviation across the $K$ value heads. It encourages exploration by assigning a higher bonus to actions leading to uncertain or less-visited regions.
- **Advantage Integration:** The advantage calculation consists of a modified PPO advantage estimate (using the mean of the $K$ value heads) plus the adaptively clipped critic bonus.
- **Critic Update:** The critic heads are updated by minimizing the bootstrap loss on their respective data subsets.

**7. All Mathematical Formulas and Symbol Definitions**

**GRPO Advantage and Objective:**
$A_i = (r_i - \text{mean}(r_1, \ldots, r_G)) / \text{std}(r_1, \ldots, r_G)$
$J_{\text{GRPO}}(\theta) = \mathbb{E}_{q \sim \mathcal{D}, \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min(\tilde{r}_{i,t} A_i, \text{clip}(\tilde{r}_{i,t}, 1 - \varepsilon, 1 + \varepsilon) A_i) \right]$
where $\tilde{r}_{i,t} = \pi_{\theta}(o_{i,t} | q, o_{i,<t}) / \pi_{\theta_{\text{old}}}(o_{i,t} | q, o_{i,<t})$.

**PPO Advantage and Objective:**
$A_t = \sum_{l=t}^{|o|} (\gamma\lambda)^{l-t} \delta_l$, where $\delta_l = r_l + \gamma V_\phi(q, o_{\leq l+1}) - V_\phi(q, o_{\leq l})$.
$J_{\text{PPO}}(\theta) = \mathbb{E}_{q \sim \mathcal{D}, o \sim \pi_{\theta_{\text{old}}}} \left[ \frac{1}{|o|} \sum_{t=1}^{|o|} \min(\tilde{r}_t A_t, \text{clip}(\tilde{r}_t, 1 - \varepsilon, 1 + \varepsilon) A_t) \right]$.

**Actor Curiosity (PPL Bonus):**
$B_{\text{actor}}(q, o) = -\frac{1}{T} \sum_{t=1}^T \log \pi(o_t | o_{<t}, q)$
$\tilde{r}(q, o) = r(q, o) + \omega \min\left(\frac{|r(q, o)|}{\kappa}, \alpha B_{\text{actor}}(q, o)\right)$

**Critic Curiosity (Multi-Head Critic Bonus):**
$\tilde{A}_{i,t} = \sum_{l=t}^{|o_i|} (\gamma\lambda)^{l-t} \left[ \underbrace{\tilde{\delta}_{i,l}}_{\tilde{A}_{i,t}} + \omega \min\left( \frac{|\tilde{A}_{i,t}|}{\kappa}, \alpha B_{\text{critic}}(q, o_{i, \leq t+1}) \right) \right]$
$\tilde{\delta}_{i,l} = r_{i,l} + \gamma \frac{1}{K} \sum_{j=1}^K \hat{V}_j(q, o_{i, \leq l+1}) - \frac{1}{K} \sum_{j=1}^K \hat{V}_j(q, o_{i, \leq l})$
$B_{\text{critic}}(q, o_{i, \leq t+1}) = \text{std}\left( \left\{ \hat{V}_j(q, o_{i, \leq t+1}) \right\}_{1 \leq j \leq K} \right)$

**Critic Update Loss:**
$L_\phi = \frac{1}{\zeta K |D|} \sum_{j=1}^K \sum_{(q, o, r) \in D_j} \left( \hat{V}_j(q, o) - r \right)^2$

**Theorem 3.1 Proof Formulations:**
Define $\tilde{r}_h(q, o) = r(q, o) + b_h(q, o)$ where $b_h(q, o) = \omega \min(\kappa |r(q, o)|, -\frac{\alpha}{|o|} \log \pi_h(o|q))$.
Consider single step policy optimization:
$\pi_{h+1}(\cdot|q) = \arg\max_\pi \left[ \sum_o \pi(o|q) \tilde{r}_h(q, o) - \frac{1}{\eta} \text{KL}(\pi(\cdot|q) \| \pi_h(\cdot|q)) \right]$
Closed-form solution: $\pi_{h+1}(o|q) = \frac{\pi_h(o|q) \exp(\eta \tilde{r}_h(q, o))}{\sum_{o'} \pi_h(o'|q) \exp(\eta \tilde{r}_h(q, o'))}$.
Define $Z(q) = \sum_{o'} \pi_h(o'|q) \exp(\eta \tilde{r}_h(q, o'))$.
$\log \pi_{h+1}(o|q) = \log \pi_h(o|q) + \eta \tilde{r}_h(q, o) - \log(Z(q))$.
Define $\Delta_h(o|q) = \log \pi_{h+1}(o|q) - \log \pi_t(o|q)$.
For two correct responses $o_1^+$ and $o_2^+$ with $-\frac{\alpha}{|o_1^+|} \log \pi_h(o_1^+|q) \geq -\frac{\alpha}{|o_2^+|} \log \pi_h(o_2^+|q)$:
$\Delta_h(o_1^+|q) - \Delta_h(o_2^+|q) = \tilde{r}_h(q, o_1^+) - \tilde{r}_h(q, o_2^+) = b_h(q, o_1^+) - b_h(q, o_2^+) = \min(\kappa, -\frac{\alpha}{|o_1^+|} \log \pi_h(o_1^+|q)) - \min(\kappa, -\frac{\alpha}{|o_2^+|} \log \pi_h(o_2^+|q)) \geq 0$.

**Theorem 3.2 Proof Formulations (Linear MDP):**
Assumption G.1 (Linear MDP): $R(s, a) = \phi(s, a)^\top \theta$, $P(s'|s, a) = \phi(s, a)^\top \psi(s')$.
Ridge estimator using all data: $\hat{w}_{n,t} = \arg\min_w \sum_{i=1}^n (G_{i,t} - \phi_{i,t}^\top w)^2 + \zeta \lambda \|w\|^2$.
Bonus term: $b^{\text{cnt}}_t(\phi) = \sqrt{\phi^\top \Lambda^{-1}_{n,t} \phi}$, where $\Lambda_{n,t} = \lambda I + \sum_{i=1}^n \phi_{i,t} \phi_{i,t}^\top$.
Bootstrap multi-head bonus: $b^{\text{boot}}_{t,K}(\phi) = \text{std}\left( \left\{ \phi^\top \hat{w}^{(k)}_{n,t} \right\}_{1 \leq k \leq K} \right)$.
Limit consistency: $b^{\text{boot}}_{t,K}(\phi) \xrightarrow[K \to \infty, n \to \infty]{p} \beta \sqrt{\phi^\top \Lambda^{-1}_{n,t} \phi}$.
Conditional variance: $\text{Var}\left( \phi^\top \hat{w}^{(k)}_{n,t} | D_n \right) \xrightarrow{p} \frac{1 - \zeta}{\zeta} \sigma^2 \phi^\top \Lambda^{-1}_{n,t} \phi$.

**SimHash Count-based Bonus:**
$\tilde{r}(q, o) = r(q, o) + \frac{\beta}{\sqrt{n(b)}}$

**8. Algorithm Pseudocode**

```text
Algorithm 1: Count-based exploration for RLVR through SimHash
Inputs :Policy πθ; aggregator g(·); random projection matrix A ∈ Rk×d; hash counts n[·] ← 0;
exploration coefficients β.

for each training iteration do
    Sample prompts q and generate responses o ∼ πθ(· | q)
    for each (q, o) in the batch do
        Obtain last-layer token states {hi}^{|q|+|o|}_{i=1}
        htraj ← g({hi}) // e.g., select the last state h|q|+|o|
        ϕ ← sign(Ahtraj) // Compute k-bit hash code
        b ← bucket(ϕ) // Convert hash code to integer index
        n[b] ← n[b] + 1
        r̃(q, o) ← r(q, o) + β/√n[b] // Shape the reward with the bonus
    Update πθ using the shaped rewards r̃(q, o).
```