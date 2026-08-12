**1. Research Background and Existing Pain Points**

**Research Background:** Reinforcement learning (RL) with group relative policy optimization (GRPO) has become a widely adopted approach for enhancing the reasoning capabilities of multimodal large language models (MLLMs). GRPO advances long-chain reasoning by combining rewards with group-relative advantage estimation, significantly boosting both alignment and reasoning efficiency without a traditional critic model. However, directly applying GRPO to MLLMs faces critical challenges due to the heterogeneous nature of text and visual modalities.

**Existing Pain Points:** GRPO and its extensions often suffer from two critical issues:
1. **Reward Sparsity:** When the base multimodal model has limited capability or the problem is difficult, only a few reasoning paths obtain positive rewards, especially in early training. 
2. **Advantage Vanishing:** Since relative advantages are computed within response groups, overly hard or easy problems lead to all-correct or all-wrong outputs, yielding zero relative advantages. As training proceeds, this issue worsens, reducing optimization efficiency and sample utilization.

Existing solutions fall into three categories, each with limitations:
- **Sample enhancement and expansion:** Enlarges the problem space by adding prompts to difficult instances or generating diverse text and image variants. However, it may exacerbate advantage vanishing due to poor control over the difficulty distribution and introduce hidden biases.
- **Selective sample utilization:** Prioritizes highly effective samples or disregards those with low learning contribution. However, this may limit exposure to complex problems or reduce data diversity, failing to fully leverage the value of all data.
- **Indirect reward design:** Provides finer-grained feedback signals to mitigate reward sparsity. However, these rewards may not perfectly align with ultimate task objectives, potentially guiding the model toward suboptimal directions due to misalignment between reasoning and the final outcome.

**2. Core Research Motivation and Scientific Questions**

**Core Research Motivation:** Although existing methods can mitigate certain instances of advantage vanishing, they all overlook a critical issue: as training progresses, problem difficulty continuously decreases, advantage vanishing intensifies, and model training efficiency steadily declines. Existing approaches cannot guarantee stable reward variance within each group regardless of problem difficulty.

**Scientific Question:** For a given problem, how can we ensure that the within-group reward distribution of responses exhibits enough variance to yield clear optimization signals for each response?

**3. Overall Core Idea and Design Philosophy**

**Overall Core Idea:** To address the challenge of reward sparsity and advantage vanishing, we propose DIVA-GRPO, a difficulty-adaptive variant augmentation advantage method that dynamically adjusts the difficulty distribution of variants for each problem from a global perspective. 

**Design Philosophy:** The core philosophy is to dynamically assess problem difficulty, adaptively sample semantically consistent variants of different difficulties, and enhance the diversity of rewards within the problem and its variant space. By computing difficulty-weighted local (for the original problem) and global (for the problem and its variants) group advantages, and introducing reward-range-based rescaling, the framework mitigates reward sparsity and vanishing advantage issues, minimizes data waste, and improves training stability.

**4. Core Innovation Points**

1. Propose DIVA-GRPO, based on dynamic difficulty assessment, using difficulty-adaptive variant generation and advantage sharing to alleviate reward sparsity and advantage vanishing.
2. Introduce joint estimation of local and global advantages, combined with batch z-score normalization and difficulty-weighted scaling, to improve training stability.
3. Present a reward-range-based advantage rescaling method, effectively preventing unreasonable advantage inflation and accelerating convergence.
4. Conduct experiments on six mainstream multimodal reasoning benchmarks, demonstrating the effectiveness and superiority of DIVA-GRPO.

**5. Overview of the Overall Technical Solution**

The overall technical solution consists of four main stages:
1. **Difficulty Assessment of Samples:** Dynamically assign a difficulty level to each problem based on historical rollout outcomes. The difficulty is recalibrated at every training epoch to adapt as the model improves.
2. **Difficulty-Adaptive Variant Generation:** Based on the assessed difficulty, generate semantically consistent variants. Simpler problems receive text and image perturbations; moderately difficult ones receive semantically equivalent textual versions; difficult problems incorporate partial reasoning guidance (think steps).
3. **Difficulty-Weighted and Normalized Advantage Balancing:** Compute both local (individual problem) and global (multiple variants derived from the original problem) advantages. Apply batch z-score normalization to make local and global signals comparable, and introduce difficulty-weighted scaling to adaptively rescale advantages according to each problem’s dynamic difficulty coefficient.
4. **Reward-Range-Based Advantage Rescaling:** Scale advantages based on the actual variability in rewards within a rollout group to prevent minor reward differences from being exaggerated and to provide a more reliable optimization signal.

**6. Detailed Module Design**

**Module 1: Difficulty Assessment of Samples**
A prerequisite for implementing adaptive strategies is to assign an appropriate difficulty level to each problem. Problem difficulty is dynamically assessed in line with the evolving capability of the model based on historical rollout outcomes: if most rollouts for a problem are correct, it is regarded as relatively easy; otherwise, it is considered relatively difficult. This assessment is recalibrated at every training epoch. We define problem difficulty within the range $[D_{min}, D_{max}]$, initialized at the midpoint $D_{mid}$. Let each original problem contain $m$ variants, and let each variant be rolled out $k$ times. The empirical accuracy is denoted as $\alpha$. The difficulty is then updated according to the update rule, ensuring continuous calibration of sampling strategies.

**Module 2: Difficulty-Adaptive Variant Generation**
Let each original problem be denoted as $q = (I_q, T_q)$, where $I_q$ is the associated image and $T_q$ is the textual description and question. After evaluating the difficulty $D_q \in [D_{min}, D_{max}]$ of each problem, we generate a set of difficulty-specific consistent variants $V_q = \{q^{(1)}, q^{(2)}, \ldots \}$ for training, ensuring that each variant preserves the original answer $y_q^*$ while selectively adjusting its difficulty.
- **Simpler problems ($D_q < D_{mid}$):** Variants are generated by perturbing both the text and the image: $q^{(i)} = (I_q^{(i)}, T_q^{(i)})$. Textual variants $T_q^{(i)}$ are created by rephrasing, restructuring, or introducing slight linguistic perturbations. Image variants $I_q^{(i)}$ are generated through perturbation functions such as rotation, Gaussian noise, salt-and-pepper noise, speckle noise, and blurring. Alternatively, the textual information $T_q$ can be embedded within the images in the variant set $V_q$ and provided as images to convey the problem information.
- **Moderate problems ($D_q \approx D_{mid}$):** Variants are generated by creating semantically equivalent textual versions: $V_q = \{(I_q, T_q^{(i)}) \mid T_q^{(i)} \text{ is a paraphrase of } T_q\}$. These variants maintain the original difficulty while diversifying expression.
- **Difficult problems ($D_q > D_{mid}$):** Variants incorporate partial reasoning guidance: $q^{(i)} = (I_q, T_q \oplus R_q^{(i)})$, where $R_q^{(i)}$ is a sequence of intermediate reasoning steps generated and verified by a closed-source model. For more difficult problems, additional reasoning steps are provided as hints.

**Module 3: Difficulty-Weighted and Normalized Advantage Balancing**
For each $q$, a set of variants $V_q = \{q^{(1)}, \ldots, q^{(N)}\}$ is constructed. Two types of advantages are defined: Local advantage and Global advantage.
- **Local advantage:** $A^{local}(y_i^{(i)})$, computed for each problem using the standard GRPO formula.
- **Global advantage:** $A^{global}(y_i^{(j)})$, computed across all responses in the variant set.
To ensure both contribute effectively while accounting for varying problem difficulty, a two-step balancing strategy is applied:
1. **Normalization (Local-Global imbalance fix):** Apply batch-level z-score normalization separately to ensure both signals remain comparable and contribute fully to training.
2. **Difficulty-weighted scaling:** Let $\{D_q^{(i)}\}_{i=1}^N$ denote the difficulty coefficients of the $N$ variants in a problem group $V_q$, and let $\bar{D}_q = \frac{1}{N}\sum_{i=1}^N D_q^{(i)}$ be the group-wise mean difficulty. The rescaled advantage is computed using the exponential difficulty-weighted scaling formula. Intuitively, when a variant is harder than the group average, correct answers are amplified while incorrect ones are softened; when a variant is easier, correct answers are down-weighted and incorrect ones are penalized more heavily.

**Module 4: Reward-Range-Based Advantage Rescaling (RRB-Rescaling)**
If rewards are tightly clustered, standard z-score normalization may exaggerate minor differences, producing misleading optimization signals. To address this, we propose reward-range-based advantage rescaling. Let $R_q = \{r_1, r_2, \ldots, r_k\}$ denote the rewards of all rollouts for a problem $q$, with the maximum possible reward range $R_{max}$. The rescaled advantage is computed using the reward range $\Delta r_q$, ensuring that the magnitude of the advantage reflects the actual variability in rewards.

**7. All Mathematical Formulas and Symbol Definitions**

**GRPO Formulas:**
- Advantage: $A(y_i) = \frac{r(y_i) - \mu_r}{\sigma_r + \epsilon}$
- Mean reward: $\mu_r = \frac{1}{k}\sum_{j=1}^k r(y_j)$
- Standard deviation of reward: $\sigma_r = \sqrt{\frac{1}{k}\sum_{j=1}^k (r(y_j) - \mu_r)^2}$
- Policy gradient: $\nabla_\theta L(\theta) = \mathbb{E}_{q \sim Q, y \sim \pi_\theta}[A(y)\nabla_\theta \log \pi_\theta(y|q)]$

**Difficulty Assessment Formulas:**
- Empirical accuracy: $\alpha = \frac{1}{mk}\sum_{i=1}^m \sum_{j=1}^k I[y_{i,j} \text{ is correct}]$
- Difficulty update rule: $D_{new} = \text{clip}(D_{old} + \eta \cdot (0.5 - \alpha), D_{min}, D_{max})$

**Advantage Balancing Formulas:**
- Global advantage: $A^{global}(y_i^{(j)}) = \frac{r(y_i^{(j)}) - \mu_q}{\sigma_q + \epsilon}$
- Batch-level z-score normalization: $\tilde{A}^{local}(y) = \frac{A^{local}(y) - \mu_{local}}{\sigma_{local} + \epsilon}$, $\tilde{A}^{global}(y) = \frac{A^{global}(y) - \mu_{global}}{\sigma_{global} + \epsilon}$
- Difficulty-weighted scaling: $\hat{A}(y_i | q^{(i)}) = \exp\left(k \cdot (D_q^{(i)} - \bar{D}_q) \cdot \text{sgn}(\tilde{A}(y_i))\right) \cdot \tilde{A}(y_i)$

**RRB-Rescaling Formulas:**
- Reward range: $\Delta r_q = (\max(R_q) - \min(R_q)) / R_{max}$
- Rescaled advantage: $\hat{A}_{range}(y_i) = \Delta r_q \cdot \tilde{A}(y_i)$

**Theoretical Derivation Propositions and Proof Formulas:**
**Theorem B.1 (Gradient Variance Control).** Let $g(\theta)$ be the stochastic policy gradient estimator $g(\theta) = \sum_{t=1}^T A_t \nabla_\theta \log \pi_\theta(a_t|s_t)$. Suppose $\mathbb{E}[g(\theta)] = \nabla J(\theta)$ is unbiased and $\text{Var}[g(\theta)] < \infty$. Then for step size $\eta > 0$:
$\mathbb{E}[\|\theta_{t+1} - \theta^*\|^2] \le \mathbb{E}[\|\theta_t - \theta^*\|^2] - \eta\|\nabla J(\theta_t)\|^2 + \eta^2 \text{Var}[\tilde{g}(\theta_t)]$

**Corollary B.2 (Advantage Normalization and Difficulty-Weighted Balancing).** Consider the adjusted gradient estimator $g̃(\theta) = \sum_{i=1}^k \hat{A}(y_i)\nabla_\theta \log \pi_\theta(y_i)$, where the adjusted advantage is defined as $\hat{A}(y_i) = \exp(k \cdot (D_i - \bar{D}) \cdot \text{sgn}(\tilde{A}(y_i))) \tilde{A}(y_i)$. 
1. Unbiasedness: Normalization enforces zero-mean scaling, so $\mathbb{E}[g̃(\theta)] = \nabla J(\theta)$.
2. Variance Reduction: $\text{Var}[g̃(\theta)] \le \text{Var}[g(\theta)]$.

**Binary Reward Setup:**
Consider a single batch of $n$ rollouts. Rewards take only two values $\{0, R_{max}\}$. Let $k$ be the number of rollouts with reward $R_{max}$ and set $\mu := \frac{k}{n} \in [0, 1]$.
Advantages: $A_i = \frac{r_i - \bar{r}}{\sigma_r}$, where $\bar{r} = \mu R_{max}$, $\sigma_r = \sqrt{\text{Var}(r)} = R_{max}\sqrt{\mu(1-\mu)}$.
Update direction: $\Delta \theta \propto \frac{1}{n}\sum_{i=1}^n A_i g_i$.
Reference unit vector $v \in \mathbb{R}^d$. Define $s^+ := v^\top\mathbb{E}[g \mid r = R_{max}]$, $s^- := v^\top\mathbb{E}[g \mid r = 0]$.

**Lemma C.1 (advantages for binary rewards).** $A^+ := \frac{R_{max} - \bar{r}}{\sigma_r} = \sqrt{\frac{1-\mu}{\mu}}$, $A^- := \frac{0 - \bar{r}}{\sigma_r} = -\sqrt{\frac{\mu}{1-\mu}}$.

**Lemma C.2 (batch update projection onto v).** $v^\top\Delta\theta \propto \mu A^+ s^+ + (1-\mu)A^- s^- = \sqrt{\mu(1-\mu)}(s^+ - s^-)$. Hence, $|v^\top\Delta\theta| \propto \sqrt{\mu(1-\mu)}|s^+ - s^-|$.

**Theorem C.3 (optimality of $\mu = 1/2$).** For any fixed $s^+, s^- \in \mathbb{R}$, the absolute projected update magnitude $F(\mu) := \sqrt{\mu(1-\mu)}|s^+ - s^-|$ over $\mu \in [0, 1]$ attains its maximum at $\mu = 1/2$. 

**Corollary C.4 (Case A: opposite-class gradients).** Assume $s^- = -s^+$. Then $v^\top\Delta\theta \propto \sqrt{\mu(1-\mu)}(s^+ - s^-) = 2\sqrt{\mu(1-\mu)}s^+$, and hence $|v^\top\Delta\theta| \propto 2|s^+|\sqrt{\mu(1-\mu)}$. Maximised at $\mu = 1/2$.

**Corollary C.5 (Case B: orthogonal-class gradients).** Suppose $s^- = 0$, while $s^+ \neq 0$. Then $v^\top\Delta\theta \propto \sqrt{\mu(1-\mu)}s^+$, so $|v^\top\Delta\theta| \propto |s^+|\sqrt{\mu(1-\mu)}$, maximised at $\mu = 1/2$. If the mean gradient of the positive class lies along $v$ with norm $\|g^+\| = \alpha$ and the mean gradient of the negative class is orthogonal with norm $\|g^-\| = \beta$, and if $\alpha\sqrt{\mu(1-\mu)} = \beta\sqrt{\mu(1-\mu)}$ (i.e., $\alpha = \beta$), the angle $\phi$ between the update and $v$ satisfies $\cos\phi = 1/\sqrt{2}$, and the absolute magnitude scales as $\sqrt{\mu(1-\mu)}$.

**8. Algorithm Pseudocode**

The paper does not contain a formal algorithm pseudocode block. The workflow is described through Figure 2 and the detailed module design in Section 3.