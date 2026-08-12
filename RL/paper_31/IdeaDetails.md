**1. Research Background and Existing Pain Points**

Offline-to-online reinforcement learning (O2O RL) aims to improve the performance of offline pretrained agents through online fine-tuning. Since the performance of pretrained agents is heavily limited by the quality and state-action space coverage of their offline datasets, O2O RL proposes to fine-tune pretrained agents in online environments to achieve notable performance improvement within just a few online environmental interaction steps. Recently, several O2O RL methods highlight that pretrained agents confront a significant state-action distribution shift between offline and online data during the fine-tuning process. This shift causes pretrained agents to inaccurately estimate Q-values in online data and misguide update directions, resulting in slow performance improvement and instability during online fine-tuning. Therefore, existing methods employ various techniques to improve the Q-value estimation, such as incorporating a conservative penalty into the estimated Q-values, employing ensemble models to enable conservative and stable Q-value estimation, or increasing the update frequency of models to facilitate accurate Q-value estimation. 

However, the existing pain points are that these methods fail to account for the full distribution of Q-value biases. Based on a comprehensive study, the paper reveals that Q-value estimation bias (i.e., Q bias) in these methods often follows a heavy-tailed distribution during online fine-tuning. Such biases induce high estimation variance and hinder performance improvement. In practice, the heavy-tailed Q bias significantly affects both Q-value estimation and policy updates in O2O RL. These heavy-tailed biases lead to extreme overestimation of Q-values and misguide agents to choose poor actions, causing large performance degradation during online fine-tuning. Meanwhile, high Q biases can lead to large fluctuations and even collapse of Q-value estimation. Furthermore, existing O2O RL methods often rely on the l2 loss function for Q-value updates. This reliance may cause heavy-tailed Q biases to disproportionately influence the Q-value loss and dominate the update gradients, further destabilizing Q-value updates.

**2. Core Research Motivation and Scientific Questions**

The core research motivation stems from the unexplored phenomenon of heavy-tailed Q-value estimation bias in O2O RL. Previous research mainly focuses on reducing the mean or variance of Q bias, neglecting its underlying distributional characteristics. The heavy-tailed nature of Q bias is inherently caused by the distribution shift problem and persists throughout online fine-tuning, challenging the widely adopted assumptions of finite variance or Gaussian modeling for such biases in prior research, thereby questioning their broad applicability. The scientific questions are: How does the Q-value estimation bias distribute during online fine-tuning in O2O RL? Why do existing methods struggle to mitigate the impact of heavy-tailed Q biases on training stability and performance? How can we explicitly model the heavy-tailed Q bias and incorporate it into the Q-value estimation process to reduce estimation variance and re-center the bias for stable and efficient fine-tuning?

**3. Overall Core Idea and Design Philosophy**

The overall core idea is to explicitly model and mitigate heavy-tailed Q-value biases by transferring their heavy-tailed nature into a parameterized Laplace-distributed noise. Since the Laplace distribution is well-suited for modeling heavy-tailed data, LAROO introduces this parameterized noise to adaptively capture heavy-tailed characteristics. By combining estimated Q-values with the Laplace-distributed noise to approximate the true Q-values and minimizing the Kullback-Leibler (KL) divergence between the Laplace likelihoods, LAROO transfers the heavy-tailed nature of biases into the noise distribution, effectively reducing the high estimation variance. The design philosophy relies on the fact that the resulting negative log-likelihood under a Laplace likelihood grows linearly rather than quadratically with the Q bias, down-weighting outliers and driving optimization using the central mass of the Q bias. Additionally, LAROO employs conservative ensemble-based estimates to re-center Q-value biases, shifting their mean toward zero. By jointly leveraging the Laplace-based noise model and the ensemble models, LAROO promotes the heavy-tailed Q-value biases into a standardized form, thereby significantly enhancing training stability and performance without the need for exploration constraints under distribution shifts.

**4. Core Innovation Points**

1. To the best of our knowledge, for the first time, the paper reveals that the Q-value estimation bias in existing O2O methods often follows a heavy-tailed distribution during online fine-tuning, which leads to instability and inefficiencies in performance improvement in O2O RL.
2. The paper proposes LAROO, a Laplace-based robust offline-to-online RL approach that introduces Laplace-distributed noise to alleviate heavy-tailed biases and uses conservative ensemble-based estimates to re-center bias, improving training stability and performance.
3. The paper provides a theoretical analysis demonstrating that LAROO reduces single-step estimation bias and estimation variance compared to the conventional l2 loss function used in Q-value updates.
4. Extensive experiments demonstrate that LAROO achieves significant performance improvement, outperforming several state-of-the-art baselines across various tasks, and showing high compatibility as a plug-in method to further enhance the performance of existing methods.

**5. Overview of the Overall Technical Solution**

The overall technical solution, LAROO, is built upon three key components:
1. Adaptive Laplace-distributed noise modeling: LAROO introduces a parameterized Laplace-distributed noise that adaptively captures heavy-tailed characteristics, providing a principled way to represent such heavy-tailed distributions. This is based on the assumption that the approximation noise in both the target Q-value and the estimated Q-value follows Laplace distributions.
2. Variance reduction via a learning process that incorporates noise: By combining estimated Q-values with the Laplace-distributed noise to approximate the true Q-values, LAROO transfers the heavy-tailed nature of biases into the noise distribution. This is achieved by training the Q-network to minimize the KL divergence between the Laplace likelihoods, leading to a robust Q-value loss function that effectively reduces the high estimation variance of Q-values.
3. Bias Re-centering: For stability, LAROO further employs conservative ensemble-based estimates to re-center Q-value biases, shifting their mean toward zero. By taking the minimum over a random subset of Q-function estimators, it explicitly reduces highly positive biases.
Together, these components promote the heavy-tailed Q-value biases into a standardized form, improving training stability and performance.

**6. Detailed Module Design**

- **Adaptive Laplace-distributed noise modeling module**: Based on the assumption that the approximation noises in target and estimated Q-values follow independent Laplace distributions with the same mean $\mu$ but potentially different scale parameters $b_1, b_2$, the module derives the Laplace-based likelihood of the true Q-value. To update the Q-value network, it minimizes the Kullback-Leibler (KL) divergence between the likelihood $p(Q(s_i, a_i) | \mathcal{T} Q_{\theta_t}(s_i, a_i))$ and $q(Q(s_i, a_i) | Q_{\theta_t}(s_i, a_i))$. The KL divergence can be transformed into a closed-form robust loss function $D_b(x)$. To effectively capture the heavy-tailed nature, the scale parameter $b$ must adapt to the changing distribution of biases. The module uses the MSE-best biased estimator (MBBE) to robustly estimate the variance of TD-errors within a batch, serving as a statistical surrogate to approximate the variance of Q-bias, and then computes the scale parameter $b$.

- **Conservative ensemble-based estimates module**: To mitigate the large-magnitude Q biases in the heavy tails that persist under standard training, this module adopts ensemble models known for effectively reducing estimation bias. It computes the target Q-value by taking the minimum over a random subset of size $M$ from an ensemble of size $K$ of Q-function estimators. This strategy reduces highly positive biases and re-centers the Q bias distribution because different Q-functions are approximately independent, and the probability of selecting the most overoptimistic head is reduced. The ensemble model is integrated with the robust function $D_b(x)$ to derive the final loss function for robust Q-value estimation, transforming the poorly-behaved, heavy-tailed Q-bias distribution into a more standardized form during Q-value updates.

**7. All Mathematical Formulas and Symbol Definitions**

- MDP Definition: $M = \langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$, where $\mathcal{S}$ is the state space, $\mathcal{A}$ is the action space, $P : \mathcal{S} \times \mathcal{A} \rightarrow \Delta(\mathcal{S})$ is the transition function, $R : \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$ is the reward function and $\gamma \in [0, 1)$ is the discount factor.
- Expected return: $\mathbb{E}_\pi[\sum_{t=0}^\infty \gamma^t r_t]$, where $r_t = R(s_t, a_t)$.
- Q-value network update (Policy evaluation): $J_\theta(Q_\theta) := \mathbb{E}_{(s,a,r,s') \sim \mathcal{B}} \left[ \left( Q_\theta(s, a) - \mathcal{T} Q_\theta(s, a) \right)^2 \right]$
- Q-value update target: $\mathcal{T} Q_\theta(s, a) = R(s, a) + \gamma \mathbb{E}_{s' \sim T(\cdot|s,a)} \left[ \max_{a'} Q_{\hat{\theta}}(s', a') \right]$
- TD error: $\delta_\theta(s, a) = Q_\theta(s, a) - \mathcal{T} Q_\theta(s, a)$
- Kurtosis metric: $\kappa = \frac{\frac{1}{n}\sum_{i=1}^n(x_i-\bar{x})^4}{\left(\frac{1}{n}\sum_{i=1}^n(x_i - \bar{x})^2\right)^2}$
- Heavy-tailed definition (moment-based): $X$ is called heavy-tailed if $\mathbb{E}\left[\frac{(X - \mu_X)^4}{\sigma_X^4}\right] > 3$
- Assumption 4.1: For each $(s, a)$ pair, $Q(s, a)$ is the true but unknown Q-value, $\varepsilon_{\hat{\theta}}$ and $\varepsilon_\theta$ are independent random variables denoting the approximation noise in target Q-value and estimated Q-value respectively. The approximation noise is independent from the $(s, a)$ and $\theta$. Then:
  $\mathcal{T} Q_\theta(s, a) = R(s, a) + \gamma \mathbb{E}_{s' \sim T(\cdot|s,a)} \left[ \max_{a'} Q_{\hat{\theta}}(s', a') \right]$
  $Q(s, a) = \mathcal{T} Q_\theta(s, a) + \varepsilon_{\hat{\theta}}$
  $Q(s, a) = Q_\theta(s, a) + \varepsilon_\theta$
- Q-value bias definition: $\text{Bias}(Q_\theta(s, a)) = \mathbb{E}[Q_\theta(s, a)] - Q(s, a) = -\mathbb{E}[\varepsilon_\theta]$
- Assumption 4.2 (Laplace-based noise): $\varepsilon_{\hat{\theta}}$ and $\varepsilon_\theta$ follow Laplace distributions with the same mean $\mu$, i.e., $\varepsilon_{\hat{\theta}} \sim \text{Laplace}(\mu, b_1)$, $\varepsilon_\theta \sim \text{Laplace}(\mu, b_2)$, $\mu \in \mathbb{R}, b_1, b_2 \in \mathbb{R}^+$.
- Laplace-based likelihood:
  $q\left(Q(s, a) \mid Q_\theta(s, a); \theta\right) = \frac{1}{2b_2} \exp\left(-\frac{|Q(s, a) - Q_\theta(s, a) - \mu|}{b_2}\right)$
  $p\left(Q(s, a) \mid \mathcal{T} Q_\theta(s, a); \hat{\theta}\right) = \frac{1}{2b_1} \exp\left(-\frac{|Q(s, a) - \mathcal{T} Q_\theta(s, a) - \mu|}{b_1}\right)$
- Q-network update via KL divergence minimization: $\theta_{t+1} = \arg\min_\theta \sum_{i=0}^N D_{KL}\left( p(Q(s_i, a_i) \mid \mathcal{T} Q_{\theta_t}(s_i, a_i)) \| q(Q(s_i, a_i) \mid Q_{\theta_t}(s_i, a_i)) \right)$
- KL divergence closed form: $D_{KL}\left( p(Q(s_i, a_i) \mid \mathcal{T} Q_\theta(s_i, a_i)) \| q(Q(s_i, a_i) \mid Q_\theta(s_i, a_i)) \right) = \frac{b_1 \exp\left(-\frac{|\mathcal{T}Q_\theta(s_i,a_i)-Q_\theta(s_i,a_i)|}{b_1}\right)}{b_2} + \frac{|\mathcal{T} Q_\theta(s_i, a_i) - Q_\theta(s_i, a_i)|}{b_2} + \log \frac{b_2}{b_1} - 1$
- Robust Q-value loss function: $D_b(x) = \exp\left(-\frac{|x|}{b}\right) + \frac{|x|}{b} - 1$
- Derivatives of $D_b(x)$:
  $D'_b(x) = \frac{\text{sgn}(x)}{b} \left( 1 - \exp\left(-\frac{|x|}{b}\right) \right)$
  $D''_b(x) = \frac{1}{b^2} \exp\left(-\frac{|x|}{b}\right)$
- Proof of differentiability at $x=0$:
  $D'_b(0) = \lim_{x \to 0} \frac{b \exp\left(-\frac{|x|}{b}\right) + |x| - b}{bx} = \lim_{x \to 0} \frac{b \left( \exp\left(-\frac{|x|}{b}\right) - 1 \right)}{bx} + \lim_{x \to 0} \frac{|x|}{bx}$. Right limit is $\lim_{x \to 0^+} D'_b(0) = 0$; Left limit is $\lim_{x \to 0^-} D'_b(0) = 0$.
  $D''_b(0) = \lim_{x \to 0} \frac{\text{sgn}(x) \left( 1 - \exp\left(-\frac{|x|}{b}\right) \right)}{bx}$. Right limit is $\frac{1}{b^2}$; Left limit is $\frac{1}{b^2}$.
- MBBE of variance (Lemma 4.3): $s^2$ is the unbiased sample variance $s^2 = \frac{1}{n-1}\sum_{i=1}^n(X_i - \bar{X})^2$, $n$ is the sample size and $\kappa$ is the population kurtosis. Then, MSE-best biased estimator $s^2_{\omega^*}$ is: $s^2_{\omega^*} = \left( \frac{\kappa}{n} + \frac{n+1}{n-1} \right)^{-1} s^2$
- Scale parameter $b$ from TD-errors: $b = s^*_{\omega} / \sqrt{2}$
- Ensemble loss function: $\mathcal{L}_b(\theta_t) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{B}} \left[ \frac{1}{K} \sum_{k=1}^K D_b \left( Q^{(k)}_{\theta_t}(s, a) - y_{\min}(s, a, r, s') \right) \right]$
- Target Q-value with ensemble: $y_{\min}(s, a, r, s') = r + \gamma \max_{a' \in \mathcal{A}} \left[ \min_{1 \leq k \leq M} Q^{(k)}_{\hat{\theta}_t}(s', a') \right]$
- l2 update parameter: $\theta^{\text{new}} = \theta + \beta (y - Q_\theta(s, a)) \nabla_\theta Q_\theta(s, a)$
- l2 post-update Q-value: $Q^{\text{new}}_\theta(s, a) \approx Q_\theta(s, a) + \beta (y - Q_\theta(s, a)) \|\nabla_\theta Q_\theta(s, a)\|_2^2$
- Ideal post-update Q-value: $Q^{\text{id}}_\theta(s, a) \approx Q_\theta(s, a) + \beta (\tilde{y} - Q_\theta(s, a)) \|\nabla_\theta Q_\theta(s, a)\|_2^2$
- l2 single-step estimation bias: $\Delta_{l2}(s, a) = \mathbb{E}_{\varepsilon_{\hat{\theta}}} \left[ Q^{\text{new}}_\theta(s, a) - Q^{\text{id}}_\theta(s, a) \right] \approx \beta \left( \mathbb{E}_{\varepsilon_{\hat{\theta}}}[y] - \tilde{y} \right) \|\nabla_\theta Q_\theta(s, a)\|_2^2$
- Theorem 4.4: When $b > 1$, the single-step estimation bias of post-update Q-value $Q^{\text{new}}_\theta$ with function $D_b(x)$ is smaller than that with l2 loss function, i.e., $\Delta_{D_b} < \Delta_{l2}$.
- Proof of Theorem B.2: 
  $Q^{\text{id}}_\theta(s, a) \approx Q_\theta(s, a) + \beta D'_b(\tilde{y} - Q_\theta(s, a)) \|\nabla_\theta Q_\theta(s, a)\|_2^2$
  $Q^{\text{new}}_\theta(s, a) \approx Q_\theta(s, a) + \beta D'_b(y - Q_\theta(s, a)) \|\nabla_\theta Q_\theta(s, a)\|_2^2$
  $\Delta_{D_b}(s, a) = \beta \mathbb{E}_{\varepsilon_{\hat{\theta}}} \left[ D'_b(y - Q_\theta(s, a)) - D'_b(\tilde{y} - Q_\theta(s, a)) \right] \|\nabla_\theta Q_\theta(s, a)\|_2^2$
  Let $z = y - Q_\theta(s, a)$, $\tilde{z} = \tilde{y} - Q_\theta(s, a)$, $f(x) = D'_b(x) - x$. 
  $\Delta_{D_b} - \Delta_{l2} = \beta \mathbb{E}_{\varepsilon_{\hat{\theta}}} [f(z) - f(\tilde{z})] \|\nabla_\theta Q_\theta(s, a)\|_2^2$.
  For $\mathbb{E}_{\varepsilon_{\hat{\theta}}}[y] - \tilde{y} > 0$, $\mathbb{E}_{\varepsilon_{\hat{\theta}}}[z] - \tilde{z} > 0$. When $b > 1$, $f'(x) = D''_b(x) - 1 = \frac{1}{b^2} \exp(-\frac{|x|}{b}) - 1 < 0$, implying $f(x)$ is monotonically decreasing. So $\Delta_{D_b} - \Delta_{l2} < 0$.
- Lemma B.3: Assume three random variables $X, Y, Z$. $Y$ is independent from $X$ and $Z$. $X$ and $Z$ are not independent. Then:
  $\text{Var}(X + Y Z) = \text{Var}(X) + \text{Var}(Y)\text{Var}(Z) + \text{Var}(Y)(\mathbb{E}[Z])^2 + \text{Var}(Z)(\mathbb{E}[Y])^2 + 2\mathbb{E}[Y]\text{Cov}(X,Z)$
- Proof of Lemma B.3: $\text{Var}(YZ) = \text{Var}(Y)\text{Var}(Z) + \text{Var}(Y)(\mathbb{E}[Z])^2 + \text{Var}(Z)(\mathbb{E}[Y])^2$.
  $\text{Cov}(X,YZ) = \mathbb{E}[Y](\mathbb{E}[XZ] - \mathbb{E}[X]\mathbb{E}[Z]) = \mathbb{E}[Y]\text{Cov}(X,Z)$.
  $\text{Var}(X + YZ) = \text{Var}(X) + \text{Var}(YZ) + 2\text{Cov}(X,YZ)$.
- Theorem 4.5 (B.4): With Assumption 4.1 and 4.2, when $b > 1$, the variance of post-update Q-value $Q^{\text{new}}_\theta$ with $D_b(x)$ function is smaller than that with l2 loss function.
- Proof of Theorem B.4:
  $\text{Var}_{l2}(Q) = \text{Var}(Q_\theta(s, a)) + \beta^2 \text{Var}(\varepsilon_\theta - \varepsilon_{\hat{\theta}}) \text{Var}(\nabla) + \beta^2 \text{Var}(\varepsilon_\theta - \varepsilon_{\hat{\theta}}) (\mathbb{E}[\nabla])^2$
  $\text{Var}_{D_b}(Q) = \text{Var}(Q_\theta(s, a)) + \beta^2 \text{Var}(D'_b(\varepsilon_\theta - \varepsilon_{\hat{\theta}})) \text{Var}(\nabla) + \beta^2 \text{Var}(D'_b(\varepsilon_\theta - \varepsilon_{\hat{\theta}})) (\mathbb{E}[\nabla])^2$
  $\text{Var}(\varepsilon_\theta - \varepsilon_{\hat{\theta}}) = 4b^2$. $|D'_b(x)| < 1/b$, so $\text{Var}(D'_b(\varepsilon_\theta - \varepsilon_{\hat{\theta}})) < 1/b^2$.
  $\text{Var}_{D_b}(Q) - \text{Var}_{l2}(Q) = \beta^2 (\text{Var}(D'_b(\varepsilon_\theta - \varepsilon_{\hat{\theta}})) - \text{Var}(\varepsilon_\theta - \varepsilon_{\hat{\theta}})) (\text{Var}(\nabla) + (\mathbb{E}[\nabla])^2) < (1/b^2 - 4b^2)(\text{Var}(\nabla) + (\mathbb{E}[\nabla])^2) < 0$.
- Asymmetric Laplace distribution PDF:
  $f(x \mid \mu, b_1, b_2) = \begin{cases} \frac{1}{b_1 + b_2} \exp\left(\frac{x-\mu}{b_2}\right), & x < \mu \\ \frac{1}{b_1 + b_2} \exp\left(-\frac{x-\mu}{b_1}\right), & x \geq \mu \end{cases}$
- Asymmetric Laplace CDF:
  $F(x \mid \mu, b_1, b_2) = \begin{cases} \frac{b_2}{b_1 + b_2} \exp\left(\frac{x-\mu}{b_2}\right), & x < \mu \\ 1 - \frac{b_1}{b_1 + b_2} \exp\left(-\frac{x-\mu}{b_1}\right), & x \geq \mu \end{cases}$
- Inverse CDF:
  $x = F^{-1}(u \mid \mu, b_1, b_2) = \begin{cases} \mu + b_2 \log\left(\frac{u(b_1 + b_2)}{b_2}\right), & u < \frac{b_2}{b_1 + b_2} \\ \mu - b_1 \log\left(\frac{(1-u)(b_1 + b_2)}{b_1}\right), & u \geq \frac{b_2}{b_1 + b_2} \end{cases}$
- Reparameterized KL divergence: $KL(q \| p) = \mathbb{E}_{u \sim U(0,1)} [\log q(T(u;\mu_q, b_1, b_2)) - \log p(T(u;\mu_q, b_1, b_2))]$

**8. Algorithm Pseudocode**

Algorithm 1 LAROO
1: Input: Offline dataset $\mathcal{D}^\text{off}$, offline RL algorithm $\mathcal{F}^\text{offline}$, value networks $\{Q_{\theta_j}\}_{j=1}^N$, a policy network $\pi_\psi$. Online replay buffer $\mathcal{D}^\text{on}$, online steps $T$.
2: Offline Phase: Training $\mathcal{F}^\text{offline}$ using $\mathcal{D}^\text{off}$
3: Online Phase: Remove the constraints in $\mathcal{F}^\text{offline}$, named online algorithm $\mathcal{F}^\text{online}$
4: for $i = 1$ to $T$ do
5:     Collect an online sample $\tau = (s, a, r, s')$ with $\pi_\psi$ in online environment, update online replay buffer $\mathcal{D}^\text{on}$
6:     Sample a training batch $\mathcal{B}$ from $\mathcal{D}^\text{off} \cup \mathcal{D}^\text{on}$
7:     Update the parameter of Laplace-distributed noise, with Equation (7)
8:     Update each Q-function $Q_{\theta_i}$ with the loss function (8) according to $\mathcal{F}^\text{online}$
9:     Update policy $\pi_\psi$ according to $\mathcal{F}^\text{online}$
10: end for