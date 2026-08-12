# Research Idea and Implementation Plan Extraction

## 1. Research Background and Existing Pain Points

**Research Background:**
Diffusion large language models (dLLMs) are emerging as an efficient alternative to autoregressive models due to their ability to decode multiple tokens in parallel. Masked Diffusion Language Models (MDLMs) operate in discrete space, employing a fixed noising process that progressively corrupts text data by replacing tokens with a special [mask] token, while a neural network is trained to learn the reverse denoising process. MDLMs optimize an Evidence Lower Bound (ELBO) of the log-likelihood, which has been widely adopted by subsequent large-scale dLLMs. Aligning these models with human preferences or task-specific rewards typically requires a post-training stage of reinforcement learning (RL), treating the model as a policy that generates a response to a prompt.

**Existing Pain Points:**
A principal challenge in applying RL to dLLMs is that the log-likelihood of dLLMs, $\log \pi_\theta(x|c)$, is computationally intractable, which precludes the direct application of standard policy gradient methods. To circumvent this, prior works (such as D1, UniGRPO, LLaDA-1.5) adapt standard RL algorithms by using the ELBO or a one-step estimation as a surrogate for the true likelihood. While straightforward, this one-sided approximation introduces a critical flaw: the ELBO is only a lower bound on the true log-likelihood ($\text{ELBO} \le \log \pi_\theta$). Consequently, minimizing the ELBO for negatively-rewarded samples does not guarantee a reduction in the true log-likelihood. This prevents the model from effectively learning from negative feedback and is incompatible with advanced RL algorithms that use relative or negative rewards, biasing the final policy.

## 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:**
To resolve the bias introduced by using the ELBO as a surrogate for the intractable log-likelihood in RL for dLLMs, particularly the inability to penalize negative reward traces effectively. The motivation is to construct a valid optimization objective that properly decreases the likelihood of bad outputs while increasing the likelihood of good outputs, without requiring the intractable true likelihood.

**Scientific Questions:**
How can we compute a more robust and less biased policy gradient for masked diffusion language models when the true log-likelihood is intractable? Can we leverage tractable upper and lower bounds of the evidence to "sandwich" the intractable log-likelihood, ensuring that maximizing a lower bound increases likelihood for positive rewards, and minimizing an upper bound decreases likelihood for negative rewards?

## 3. Overall Core Idea and Design Philosophy

**Overall Core Idea:**
The core idea is the "Sandwiched Policy Gradient" (SPG), which leverages both an upper and a lower bound of the true log-likelihood to create a valid policy optimization objective. For sequences with positive advantages (high rewards relative to the group), the method maximizes a tractable lower bound (ELBO) on the log-likelihood. For sequences with negative advantages (low rewards), it minimizes a tractable evidence upper bound (EUBO) on the log-likelihood. 

**Design Philosophy:**
Since $L_{ELBO} \le \log \pi_\theta \le L_{EUBO}$, maximizing the ELBO guarantees an increase in the true log-likelihood (or a valid lower bound), while minimizing the EUBO guarantees a decrease in the true log-likelihood (since the true likelihood is bounded above by the EUBO). By combining these based on the sign of the advantage, the resulting SPG objective acts as a true lower bound for the original group relative policy optimization objective, thereby serving as a valid proxy for optimizing the true expected reward without the bias introduced by solely using ELBO.

## 4. Core Innovation Points

1.  **Sandwiched Policy Gradient Algorithm:** A novel policy gradient algorithm for dLLMs that reduces bias by optimizing sandwiched variational bounds based on reward. It maximizes ELBO for positive advantage traces and minimizes EUBO for negative advantage traces.
2.  **Tractable Evidence Upper Bound (EUBO):** Derivation of a tractable upper bound for the log-likelihood of masked diffusion models based on the Rényi variational bound, enabling the valid penalization of negative reward sequences.
3.  **Block-Wise Masking Strategy:** A novel Monte Carlo estimation strategy that aligns the data distribution during policy rollout and optimization. Instead of random masking, it selects a random block for masking, keeping earlier blocks clean and later blocks fully masked, which improves the stability of the objective's estimation.
4.  **Mixture of Upper and Lower Bounds:** A practical mixture objective combining EUBO and ELBO ($\tilde{L}_{Mix} = \omega \tilde{L}_{EUBO} + (1-\omega)L_{ELBO}$) for negative advantage traces, which harnesses the strong correction signal of EUBO and the stable estimation of ELBO, theoretically proven to strictly reduce gradient variance.

## 5. Overview of the Overall Technical Solution

The overall technical solution follows a group relative policy optimization framework. For a given prompt $c$, a group of $g$ responses $\{x_j\}_{j=1}^g$ is generated from the policy $\pi_\theta$. The advantage $A_j(c,x_j) := R(c,x_j) - \frac{1}{g}\sum_{\hat{\jmath}=1}^g R(c,x_{\hat{\jmath}})$ is computed for each response. 

The policy gradient objective is transformed into an advantage-weighted log-likelihood objective. To handle the intractability of $\log \pi_\theta(x_j|c)$, the objective is modified into the SPG objective: for samples with positive advantages ($A_j \ge 0$), the ELBO is maximized; for samples with negative advantages ($A_j < 0$), the EUBO (or a mixture of EUBO and ELBO) is minimized. 

The bounds are estimated via Monte Carlo sampling using a block-wise masking strategy. For each generated sequence $x_j$, random timesteps are sampled, and partially masked samples are generated. The ELBO and EUBO are computed based on the model's predictions at the masked positions, weighted by time-dependent factors. The model parameters $\theta$ are then updated using the gradient of the sandwiched objective.

## 6. Detailed Module Design

### 6.1 Masked Diffusion Language Model (MDLM) Module
MDLMs define a forward noising process that corrupts clean text $x_{1:n}$ into $z_t \equiv z_{t,1:n}$ over continuous timestep $t \in [0,1]$ by replacing tokens with a [mask] token. The forward transition kernel is defined per token. The reverse process is parameterized by a policy $\pi_\theta$ which predicts the original tokens from the corrupted version. The model is pre-trained by maximizing the ELBO of the log-likelihood.

### 6.2 SPG Objective Formulation Module
This module constructs the sandwiched optimization objective. The standard group objective $J^{group}(\theta)$ requires $\log \pi_\theta(x_j|c)$. SPG replaces this with a sandwiched objective $J_{SPG}(\theta)$. For positive advantages, it uses the ELBO (which is tight for maximization). For negative advantages, it uses the EUBO (which is tight for minimization). Since $L_{ELBO} \le \log \pi_\theta \le L_{EUBO}$, maximizing the ELBO and minimizing the EUBO guarantees that $J_{SPG}(\theta) \le J^{group}(\theta)$, making it a valid lower bound proxy.

### 6.3 Evidence Upper Bound (EUBO) Derivation Module
To penalize negative rewards, an EUBO is derived based on the Rényi variational bound. The bound utilizes a hyperparameter $\beta \ge 1$ to control tightness. The derivation starts from the discrete-time Markov chain and generalizes to sequences of categorical variables. The constant term $C(T)$ is independent of $\theta$ and drops out during gradient computation. The continuous-time version is obtained by taking the limit $T \to \infty$.

### 6.4 Block-Wise Masking Strategy Module
To estimate the bounds via Monte Carlo, perturbed samples $z_t$ are generated. Standard random masking does not align with the semi-autoregressive unmasking strategy used during generation (where blocks are unmasked sequentially). The block-wise masking strategy divides the sequence into blocks of size $B$. A random block is selected as the current generation block. Earlier blocks are kept clean, the selected block has tokens randomly masked, and later blocks are fully masked. Additionally, prompt and clean blocks are lightly perturbed by randomly masking tokens with probability $p_{mask}=0.15$.

### 6.5 Mixture Objective Module
Monte Carlo estimation of EUBO can be biased (due to the log outside the expectation) and requires many samples. The mixture objective combines EUBO and ELBO for negative traces: $\tilde{L}_{Mix}(x|c;\theta) := \omega \cdot \tilde{L}_{EUBO}(x|c;\theta) + (1-\omega) \cdot L_{ELBO}(x|c;\theta)$. The EUBO provides a strong correction signal (sharpening decisions via $\beta$-power), while ELBO provides stable estimation. Theoretically, the gradient of the mixture objective realizes a confidence-aware weighting where uncertain tokens have smaller weights and confident tokens are upweighted, strictly reducing coordinate-wise gradient variance.

## 7. All Mathematical Formulas and Symbol Definitions

**Symbol Definitions:**
-   $x$: Scalar
-   $\mathbf{x}$: Vector
-   $x_{1:n}$: Sequence
-   $[k]$: Set $\{1, \dots, k\}$
-   $\text{Cat}(x|p)$: Categorical distribution over $x$ with probabilities $p$
-   $\mathcal{U}[a,b]$: Uniform distribution in $[a,b]$
-   $i \in [n]$: Position of the token
-   $j \in [g]$: Sequence in a group of rollouts
-   $t$: Diffusion timestep (discrete $t \in [T]$, continuous $t \in [0,1]$)
-   $z_t$: Corrupted sequence at timestep $t$
-   $\mathbf{m}$: One-hot representation of the [mask] token
-   $\alpha_t$: Noise schedule (strictly decreasing, $\alpha_0=1, \alpha_1=0$)
-   $\pi_\theta$: Policy (neural network)
-   $R(c,x)$: Reward function
-   $A_j(c,x_j)$: Advantage function
-   $\beta$: Hyperparameter for EUBO tightness ($\beta \ge 1$)
-   $\omega$: Blend coefficient for mixture objective ($0 \le \omega \le 1$)

**Mathematical Formulas:**

1.  **Forward Transition Kernel:**
    $$q_{t|0}(z_{t,i}|x_i) = \text{Cat}\left(z_{t,i} | \alpha_t x_i + (1-\alpha_t)\mathbf{m}\right)$$

2.  **Reverse Process Parameterization:**
    $$p_\theta(z_s|z_t) = q(z_s|z_t, x=\pi_\theta(\cdot|z_t)) = \begin{cases} \text{Cat}(z_s; z_t), & z_t \neq \mathbf{m}, \\ \text{Cat}\left(z_s; (1-\alpha_s)\mathbf{m} + \frac{(\alpha_s-\alpha_t)\pi_\theta(\cdot|z_t)}{1-\alpha_t}\right), & z_t = \mathbf{m}. \end{cases}$$

3.  **ELBO Objective:**
    $$L_{ELBO}(x;\theta) = \mathbb{E}_{t,z_t}\left[\sum_{i=1}^n w(t) \cdot \mathbb{1}(z_{t,i}=\mathbf{m}) \cdot \log \pi_\theta(x_i|z_t)\right]$$
    where $w(t) = \alpha'_t / (\alpha_t - 1)$.

4.  **Policy Gradient Estimator:**
    $$\nabla_\theta J(\theta) = \mathbb{E}_{x \sim \pi_\theta(\cdot|c)}\left[R(c,x)\nabla_\theta \log \pi_\theta(x|c)\right]$$

5.  **Group Relative Objective:**
    $$J^{group}(\theta) = \mathbb{E}_{c,\{x_j\}\sim\pi_{sg[\theta]}}\left[\frac{1}{g}\sum_{j=1}^g A_j(x_j,c) \log \pi_\theta(x_j|c)\right]$$

6.  **Sandwiched Policy Gradient Objective:**
    $$J_{SPG}(\theta) = \mathbb{E}\left[\frac{1}{g}\sum_{j=1}^g \left(\mathbb{1}_{A_j \ge 0} \cdot A_j L_{ELBO}(x_j|c;\theta) + \mathbb{1}_{A_j < 0} \cdot A_j L_{EUBO}(x_j|c;\theta)\right)\right]$$

7.  **Evidence Upper Bound (Discrete Time):**
    $$L_{EUBO}(x_{1:n};\theta) = \frac{1}{\beta}\sum_{i=1}^n \log \sum_{t=1}^{T-1} \mathbb{E}_{z_{t+1}}\left[\frac{\alpha_t - \alpha_{t+1}}{1-\alpha_{t+1}} \cdot \mathbb{1}(z_{t+1,i}=\mathbf{m}) \cdot \pi_\theta^\beta(x_i|z_{t+1})\right] + C(T)$$
    where $C(T) := \mathbb{1}(\beta < n) \cdot \frac{1}{\beta}\log\mathbb{E}_{z_{1:T}\sim q(\cdot|x)}\left[q(z_{1:T}|x)^{-n}\right]$ is a constant independent of $\theta$.

8.  **Evidence Upper Bound (Continuous Time, Corollary 1):**
    $$\tilde{L}_{EUBO}(x_{1:n};\theta) = \frac{1}{\beta}\sum_{i=1}^n \log \mathbb{E}_{t,z_t}\left[w(t) \cdot \mathbb{1}(z_{t,i}=\mathbf{m}) \cdot \pi_\theta^\beta(x_i|z_t)\right]$$

9.  **Mixture Objective:**
    $$\tilde{L}_{Mix}(x|c;\theta) := \omega \cdot \tilde{L}_{EUBO}(x|c;\theta) + (1-\omega) \cdot L_{ELBO}(x|c;\theta)$$

10. **Proposition 1 (Gradient of Mixture):**
    Fix a coordinate $k$ and let $\rho_\beta := w(t,z_t)\pi_\theta^\beta(x_i|z_t,c)/\mathbb{E}[w(t,z_t)\pi_\theta^\beta(x_i|z_t,c)]$, where $w(t,z_t) := w(t)\mathbb{1}(z_t=\mathbf{m})$. The gradient of the mixture objective is:
    $$g_{\omega,k} = ((1-\omega)w(t,z_t) + \omega\rho_\beta) \partial_{\theta_k}\log \pi_\theta(x|z_t,c)$$
    If $\text{Var}((\rho_\beta - w(t,z_t))\partial_{\theta_k}\log \pi_\theta(x|z_t,c)) > 0$, then $\text{Var}[g_{\omega,k}]$ is a strictly convex quadratic in $\omega$ and admits a unique minimizer $\omega_k^\star$. Moreover,
    $$\text{Var}[g_{\omega_k^\star,k}] < \min\{\text{Var}[g_{0,k}], \text{Var}[g_{1,k}]\}$$

## 8. Algorithm Pseudocode

```algorithm
Algorithm 1 SPG: Sandwiched Policy Gradient for Masked dLLMs
Require: prompt distribution D, number of completions per prompt g, number of inner updates µ, forward process q, number of Monte Carlo samples m, initial policy π0, learning rate ϵ.

1: Initialize πθ ← π0
2: while not converged do
3:   Sample a prompt c ∼ D, then g completions {xj ∼ πθ(· | c)}gj=1
4:   ∀j ∈ [g], compute reward R(c,xj) and advantage Aj(xj , c)
5:   for gradient update iterations {1, . . . , µ} do
6:     ∀j ∈ [g], generate m perturbed samples {zj tτ }m τ=1 ∼ q(· | xj) via block-wise masking (Section 3.3).
7:     Compute the sandwiched policy gradient ∇JSPG(θ) where:
       JSPG(θ) = E [ 1 g g∑j=1 (1Aj≥0 ·AjLELBO(x j | c;θ) + 1Aj<0 ·AjL̃EUBO(x j | c;θ)) ],
8:     and LELBO, L̃EUBO are estimated from {zj tτ }m τ=1, using Equation 2 and 7.
9:     Perform gradient update: θ ← θ + ϵ∇JSPG(θ)
10: return πθ
```