# Research Idea and Implementation Plan Extraction

## 1. Research Background and Existing Pain Points
Masked diffusion models (MDMs) have emerged as a promising alternative architecture to autoregressive models (ARMs) for large language models and multi-modal models. They operate by masking input tokens at a randomly sampled masking rate and learning to reconstruct them, potentially addressing ARM limitations such as lack of parallelism, exposure bias, and reversal curse. However, the noise-based training paradigm of MDMs inherently suffers from much higher training variance than ARMs, leading to noisier gradient estimates and unstable optimization. This phenomenon manifests as the training divergence from ARMs in two aspects:
1.  **Across-run variability:** Even under identical training setups, MDMs can converge to very different solutions due to erratic updates driven by fluctuating losses. By contrast, ARMs converge reliably to the same solution across runs.
2.  **Within-run sub-optimality:** Although pretrained MDMs and ARMs show comparable abilities at initialization, a sharp divergence occurs after post-training (fine-tuning), where ARMs always outperform MDMs by wide margins. The variance gap makes MDMs use supervision resources less efficiently as noisier updates slow convergence.

Currently, there has been no theoretical explanation or systematic solution for this variance gap. Existing mitigation strategies remain isolated and ad-hoc, lacking a unified theoretical framework. For instance, Arriola et al. (2025) relies on coarse candidate intervals for noise schedules, while others like Zhu et al. (2025) are limited to preference optimization.

## 2. Core Research Motivation and Scientific Questions
The core research motivation is to develop a systematic explanation and theoretically grounded solutions for the high training variance of MDMs. The paper revisits the basic definitions to answer the central scientific question: **Can we construct an alternative estimator that remains unbiased but achieves lower variance, providing a more stable basis for optimization?** Specifically, the work seeks to identify the distinct sources of variance in the MDM training objective (which ARMs do not suffer from) and design Pareto-optimal strategies to minimize these specific noise sources without introducing bias.

## 3. Overall Core Idea and Design Philosophy
The overall core idea is to derive a systematic decomposition of MDM training variance into three distinct sources and design targeted, theoretically grounded variance-reduction techniques for each source while maintaining unbiasedness. The design philosophy centers on importance sampling and control variates. Instead of using the standard uniform masking rate sampler, the work seeks the Pareto-optimal sampling distribution that minimizes the total variance of the loss estimator. Simultaneously, it introduces complementary masking patterns to create negatively correlated samples, effectively hedging against the randomness of which tokens are masked. This dual approach ensures that the estimator remains unbiased while drastically reducing the gradient noise, bridging the training stability gap with ARMs.

## 4. Core Innovation Points
1.  **First Systematic Variance Decomposition:** Derivation of the MDM training variance into three sources: A (Masking Pattern Noise), B (Masking Rate Noise), and C (Data Noise), cleanly explaining the fundamental training gap with ARMs (which are only affected by C).
2.  **P-POTS (Parametric-Pareto Optimal Timestep Sampling):** A Pareto-optimal unbiased t-sampler that minimizes A, B, and C by sampling harder t values more often with appropriately smaller update steps. It introduces a parametric model (EPR) to fit the optimal distribution.
3.  **MIRROR (Variance reduction with MIRRORed masks):** A method targeting A by using negatively correlated samples. It constructs two complementary noised inputs from the same data and masking rate, reducing masking pattern noise by at least a half.
4.  **ISAD (Importance Sampling on Answer Delimiters):** Shifts masking toward answer delimiters to target A, reweighting token-level losses to keep the estimator unbiased.
5.  **SyRM (Syntax-and-Response Mask):** Modifies the eligibility set for masking to include both response and syntax tokens, reducing masking-pattern noise A at the cost of a small, bounded bias (optimum shift), which overall decreases the mean-squared error.
6.  **StraTS (Stratified T-Sampling):** Replaces i.i.d. sampling of masking rates with stratified sampling to reduce B via inter-stratum variance reduction.
7.  **Bin-Wise EMA Control Variate:** Reduces B by maintaining an exponential moving average of losses within bins of t, using them as control variates to cancel baseline-related noise.

## 5. Overview of the Overall Technical Solution
The standard MDM training algorithm samples clean data, samples a masking rate $t \sim U[0,1]$, constructs a masked input, computes the sample loss, and updates parameters. This estimator is unbiased but high variance. The proposed overall technical solution replaces this standard pipeline with:
1.  **Variance-Aware Sampling:** Replacing the uniform masking rate sampler $t \sim U[0,1]$ with a data-fitted non-uniform sampler $t \sim p^*(t)$ derived from the variance decomposition, and reweighting the loss by $1/p(t)$ to maintain unbiasedness (P-POTS).
2.  **Complementary Masking:** Constructing two masked samples complementarily for the same data and $t$ and averaging their losses, leveraging negative correlation to reduce pattern noise (MIRROR).
3.  **Specific Structural Adjustments:** Adapting the eligibility sets for masking (SyRM) or shifting masking probabilities towards specific tokens (ISAD) for structured data.
4.  **Stratification and Control Variates:** Using stratified sampling for $t$ (StraTS) and bin-wise EMA as control variates (EMA) to reduce mask rate noise.

## 6. Detailed Module Design

### 6.1 P-POTS (Parametric-Pareto Optimal Timestep Sampling)
**Mechanism:** P-POTS replaces the masking rate sampler $t \sim U[0,1]$ with a data-fitted non-uniform one $p(t)$ to reduce $A+B+C$. 
**Pre-training Estimation:** Before training, for each of $\{x_0^{(i)}\}_{i=1}^a$, draw $b$ values $\{t_j\}_{j=1}^b$ and for each $(x_0^{(i)}, t_j)$, draw $c$ masked samples $\{x_t^{(i,j,k)}\}_{k=1}^c$ with losses $\ell_{i,j,k} := l_\theta(x_0^{(i)}, t_j, x_t^{(i,j,k)})$. Compute:
$\hat{g}_j = \frac{1}{ac} \sum_{i=1}^a \sum_{k=1}^c \ell_{i,j,k}, \quad \hat{v}_j = \frac{1}{a} \sum_{i=1}^a \frac{1}{c-1} \sum_{k=1}^c (\ell_{i,j,k} - \frac{1}{c} \sum_{k'=1}^c \ell_{i,j,k'})^2, \quad \hat{p}_j = \frac{\sqrt{\hat{g}_j^2 + \hat{v}_j}}{\sum_{j=1}^b \sqrt{\hat{g}_j^2 + \hat{v}_j}}$
**Parametric Fitting (EPR Model):** To capture the structure of the scatter $\{\hat{p}_j\}$, the EPR model is defined as:
$p_{\text{EPR}}(t) = \sqrt{at^r + b(1-t)^q + A^2 \exp(2\kappa t^m)}$, with $a, b, A, \kappa > 0, r, q \ge 0, m > 1$
with $g_{\text{EPR}}(t) = A \exp(\kappa t^m)$ and $v_{\text{EPR}}(t) = b(1-t)^q + at^r$. It is fitted via KL divergence. Parameters are set as $b=70, a=15, c=15$.
**Training:** Sample $t \sim p^*(t)$ and reweight the loss by $1/p^*(t)$.

### 6.2 MIRROR (Variance reduction with MIRRORed masks)
**Mechanism:** MIRROR targets A by creating two masked samples complementarily from the same $x_0$ and $t$.
**Steps:**
1. Sample clean data $x_0$.
2. Sample masking rate $t \sim U[0,1]$.
3. For each eligible token $i$, sample $U_i \sim U[0,1]$, and generate two noisy samples:
   - $x_t^1$ by masking $x_0(i)$ if $U_i < t$
   - $x_t^2$ by masking $x_0(i)$ if $U_i > 1-t$
4. Compute $l_j = -\frac{1}{t} \sum_{i=1}^P \mathbb{1}[x_t^j(i) = \texttt{[MASK]}] \log p_\theta(x_0(i) | x_t^j)$ for $j \in \{1, 2\}$
5. Average $\bar{l} = \frac{1}{2}(l_1 + l_2)$ across batch for backpropagation.
**Variance Reduction:** Since $l_1, l_2$ are identically distributed with correlation $\rho \le 0$, $Var(\bar{l}) = \frac{\sigma^2}{2}(1+\rho) \le Var(l_1)$.

### 6.3 SyRM (Syntax-and-Response Mask)
**Mechanism:** Modifies the eligibility set for masking to $R \cup C$ (response tokens $\cup$ syntax tokens). 
**Theoretical Basis (Theorem 1):** Let $L = \frac{1}{Pt} \sum_{i=1}^P M_i Y_i$. $\text{Var}[L|t] = \frac{1}{P^2 t} \sum_{i=1}^P \sigma_i^2 + \frac{1-t}{P^2 t} \sum_{i=1}^P \mu_i^2 + \frac{2}{P^2} \sum_{1 \le i < j \le P} \rho_{ij}$.
Under assumptions A1-A5 (homogeneity within groups, stable syntax tokens, stronger reasoning coupling for responses), Theorem 3 proves $A(S_C) < A(S_R)$.
**Optimum Shift:** Theorem 4 states $L_{\text{SyRM}} = \alpha L_{\text{resp}} + (1-\alpha) L_{\text{coord}}$. Theorem 5 proves optimum-shift bias is inevitable if $\nabla_\theta J_{\text{coord}}(\theta_{\text{resp}}^*) \neq 0$. Theorem 6 bounds the shift: $\|\theta_{\text{SyRM}}^* - \theta_{\text{resp}}^*\| \le \frac{1-\alpha}{\lambda_{\min}(H_{\text{SyRM}})} \|\nabla J_{\text{coord}}(\theta_{\text{resp}}^*)\|$.

### 6.4 ISAD (Importance Sampling on Answer Delimiters)
**Mechanism:** Shift masking toward answer delimiters using $q_j(t) = \begin{cases} t, & \text{if token } j \notin \text{rare ids} \\ \min(1, t+\Delta), & \text{if token } j \in \text{rare ids} \end{cases}$, with $\Delta \in (0,1)$. Token-level losses are reweighted by $1/q_j(t)$.

### 6.5 StraTS (Stratified T-Sampling)
**Mechanism:** Partition $[0,1]$ into $k$ equal strata $I_j = [\frac{j-1}{k}, \frac{j}{k})$. Sample one $t$ per stratum.
**Variance Reduction:** $Var_{\text{strat}}[\hat{\mu} | x_0] = \frac{1}{n}\sigma_t^2 - \frac{1}{k}\sigma_b^2$. Reduction is $\frac{1}{k}\sigma_b^2$. Optimal strata $n_{\text{opt}} \propto \sqrt{B}$ (batch size).

### 6.6 Bin-Wise EMA Control Variate
**Mechanism:** Partition $t \in [0,1]$ into $m$ bins. Maintain EMA stats $\mu_L(j), \mu_H(j), M_{LH}(j), M_{HH}(j)$ and baseline $\tilde{b}_j$.
**Update Rule:**
$c_j = \frac{M_{LH}(j) - \mu_L(j)\mu_H(j)}{M_{HH}(j) - \mu_H(j)^2 + \epsilon}$
$\ell_i^{\text{adj}} = \ell_i - c_j \cdot \tilde{b}_j$
Update stats with learning rate $\eta$.
**Hyperparameter Selection:** $b^* \propto m^{1/5}$. $m \times (\frac{1}{\eta} \times \text{batch size per GPU}) \approx 0.1 \times \text{train data size}$.

## 7. All Mathematical Formulas and Symbol Definitions

**MDM Training Objective:**
$\mathcal{L}_{\text{MDM}} = \mathbb{E}_{x\sim q(x), t\sim \mathcal{U}[0,1], x_t\sim q(\cdot|x,t)} [l_\theta(x, t, x_t)]$
$l_\theta = \frac{\alpha'_t}{1-\alpha_t} \sum_{\ell=1}^L \log \langle x^\ell_\theta(x^{1:L}_t, t), x^\ell \rangle$

**Variance Decomposition:**
$\text{Var}_{x_0,t,x_t}(l_\theta) = \underbrace{\mathbb{E}_{x_0,t}[\text{Var}_{x_t}(l_\theta | x_0, t)]}_{A} + \underbrace{\mathbb{E}_{x_0}[\text{Var}_t(g_\theta(x_0, t) | x_0)]}_{B} + \underbrace{\text{Var}_{x_0}(\mathbb{E}_t[g_\theta(x_0, t)])}_{C}$
where $g_\theta(x_0, t) = \mathbb{E}_{x_t}[l_\theta | x_0, t]$.

**Importance Sampling Unbiasedness:**
$\mathbb{E}_{t\sim p, x_t}[\frac{1}{p(t)} l_\theta(x_0, t, x_t)] = \int_0^1 \frac{p(t)}{p(t)} g(t) dt = \int_0^1 g(t) dt = \mathbb{E}_{t\sim \mathcal{U}, x_t}(l_\theta)$

**Variance with Importance Sampling:**
$A+B+C = \text{Var}_{x_0,t,x_t}(\frac{1}{p(t)} l_\theta(x_0, t, x_t)) = \int_0^1 \frac{g(t)^2+v(t)}{p(t)} dt - (\int_0^1 g(t) dt)^2$

**Optimal Sampler (P-POTS):**
$p^*(t) = \frac{\sqrt{g(t)^2+v(t)}}{\int_0^1 \sqrt{g(s)^2+v(s)} ds} \propto \sqrt{g(t)^2+v(t)}$

**EPR Model:**
$p_{\text{EPR}}(t) = \sqrt{at^r + b(1-t)^q + A^2 \exp(2\kappa t^m)}, \quad a, b, A, \kappa > 0, r, q \ge 0, m > 1$
$g_{\text{EPR}}(t) = A \exp(\kappa t^m)$
$v_{\text{EPR}}(t) = b(1-t)^q + at^r$

**MIRROR Variance:**
$\text{Var}(\bar{l}) = \text{Var}(\frac{1}{2}(l_1 + l_2)) = \frac{1}{4}(\text{Var}(l_1) + \text{Var}(l_2) + 2\text{Cov}(l_1, l_2)) = \frac{2\sigma^2}{4}(1+\rho) = \frac{\sigma^2}{2}(1+\rho)$
Coverage: $\Pr(i \in M_1 \cup M_2) = \Pr(U_i < t) + \Pr(U_i > 1-t) = \min(1, 2t)$

**SyRM Variance (Theorem 1):**
$\text{Var}[L|t] = \frac{1}{P^2 t} \sum_{i=1}^P \sigma_i^2 + \frac{1-t}{P^2 t} \sum_{i=1}^P \mu_i^2 + \frac{2}{P^2} \sum_{1 \le i < j \le P} \rho_{ij}$

**ISAD Distribution:**
$q_j(t) = \begin{cases} t, & \text{if token } j \notin \text{rare ids} \\ \min(1, t+\Delta), & \text{if token } j \in \text{rare ids} \end{cases}$

**StraTS Variance:**
$\text{Var}_{\text{strat}}[\hat{\mu} | x_0] = \frac{1}{n}\sigma_t^2 - \frac{1}{k}\sigma_b^2$

**EMA Control Variate:**
$\ell_i^{\text{adj}} = \ell_i - c_j \cdot \tilde{b}_j$
$c_j = \frac{M_{LH}(j) - \mu_L(j)\mu_H(j)}{M_{HH}(j) - \mu_H(j)^2 + \epsilon}$

**UniGRPO Objective:**
$J_{\text{UniGRPO}}(\theta) = \mathbb{E}_{(q,a)\sim D, \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot|q), \{p_i \in [0,1]\}_{i=1}^G} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left( \min(r'_{i,t}(\theta) \hat{A}_{i,t}, \text{clip}(r'_{i,t}(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_{i,t}) - \beta D_{KL}(\pi'_\theta \| \pi'_{\text{ref}}) \right) \right]$

## 8. Algorithm Pseudocode

**Standard MDM Training:**
1. Sample clean data $x_0$ from the data distribution.
2. Sample a masking rate $t \sim U[0,1]$.
3. For each eligible token $i$, sample $U_i \sim U[0,1]$, and construct $x_t$ by replacing $x_0(i)$ with [MASK] if $U_i < t$.
4. Compute the sample loss $l_\theta(x_0, t, x_t) = -\frac{1}{Pt} \sum_{i=1}^P \mathbb{1}[x_t(i) = \texttt{[MASK]}] \log p_\theta(x_0(i) | x_t)$.
5. Average $l_\theta(x_0, t, x_t)$ across the batch to obtain $\hat{l}_\theta$ and update $\theta$ via backpropagation.

**MIRROR Training:**
1. Sample clean data $x_0$ from the data distribution.
2. Sample a masking rate $t \sim U[0,1]$.
3. After sampling $x_0$ and $t$, for each eligible token $i$, sample $U_i \sim U[0,1]$, and generate two noisy samples:
   - $x_t^1$ by masking $x_0(i)$ if $U_i < t$
   - $x_t^2$ by masking $x_0(i)$ if $U_i > 1-t$
4. Compute $l_j = -\frac{1}{t} \sum_{i=1}^P \mathbb{1}[x_t^j(i) = \texttt{[MASK]}] \log p_\theta(x_0(i) | x_t^j)$ for $j \in \{1, 2\}$
5. Average $\bar{l} = \frac{1}{2}(l_1 + l_2)$ across batch for backpropagation.

**Algorithm 1: UniGRPO Policy Gradient Optimization**
Require: Reference model $\pi_{\text{ref}}$, prompt distribution $\mathcal{D}$, number of completions per prompt $G$, number of inner updates $\mu$, diffusion steps $T$
1: Initialize policy $\pi_\theta \leftarrow \pi_{\text{ref}}$
2: while not converged do
3:  $\pi_{\text{old}} \leftarrow \pi_\theta$
4:  Sample a prompt $q \sim \mathcal{D}$
5:  Sample $G$ completions $o_i \sim \pi_{\text{old}}(\cdot | q)$ for $i \in [G]$
6:  For each $o_i$, compute reward $r_i$ and advantage $\hat{A}_i^{(k)}(\pi_{\text{old}})$
7:  Sample a starting timestep $t_0 \sim U(0, T-1)$
8:  Generate $\mu-1$ uniformly spaced timesteps $t_1, \dots, t_{\mu-1}$ from $[t_0, T]$
9:  for gradient update iterations $n = 1$ to $\mu$ do
10:   if $n = 1$ then
11:    Sample a starting mask ratio $r_1 \sim U(0,1)$ and compute initial timestep $t_1 = \lfloor r_1 T \rfloor$
12:   else
13:    Compute $t_n = \lfloor \frac{n-1}{\mu-1}(T-t_1) + t_1 \rfloor$
14:   end if
15:   Construct input $(q, \text{masked } o_i)$ using timestep $t_n$ (with $q$ always unmasked)
16:   For $\pi_\theta, \pi_{\text{old}}, \pi_{\text{ref}}$, estimate log-probabilities of masked tokens in $o_i$ at $t_n$
17:   Compute UniGRPO objective 24 and update $\pi_\theta$ via gradient descent
18:  end for
19: end while
20: return $\pi_\theta$