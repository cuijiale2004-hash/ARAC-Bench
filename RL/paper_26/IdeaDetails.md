**1. Research Background and Existing Pain Points**

**Research Background:** Offline reinforcement learning (RL) enables learning policies exclusively from static datasets, eliminating the need for online interaction. This is crucial for real-world domains where exploration is risky or costly, such as robotics, healthcare, and autonomous systems. However, applying standard off-policy RL algorithms to offline data causes distribution shift. When the learned policy generates out-of-distribution (OOD) actions—those deviating substantially from the training data—value functions extrapolate erroneously, leading to severe value overestimation and catastrophic performance degradation.

**Existing Pain Points:** Existing approaches to mitigate this issue fall into two categories:
1.  **Policy Constraint Methods:** These enforce the learned policy to remain close to the behavior policy using variational auto-encoders (VAEs). However, VAEs struggle to capture multi-modal nature of real-world behaviors, often collapsing diverse action distributions into suboptimal averaged outputs within low-density regions.
2.  **Value Regularization Methods:** These learn conservative Q-functions that penalize OOD actions. Their effectiveness relies on the underlying OOD identification mechanism, which is limited by the representation capacity of models characterizing data distribution. Crucially, they apply **uniform penalties** across the entire out-of-support region without considering valuable explorations that could enhance policy performance, thereby suppressing beneficial exploration.
3.  **Uncertainty-based Methods:** Model ensembles and MC dropout often conflate epistemic and aleatoric uncertainty, leading to erroneous OOD identification.
4.  **Recent Adaptive Methods:** Approaches like DoRL-VC employ VAE-based detectors but inherit strong Gaussian assumptions regarding the behavior policy, fundamentally limiting their ability to reliably identify OOD samples.

**2. Core Research Motivation and Scientific Questions**

**Core Research Motivation:** There is a critical need to move beyond uniform penalization of OOD actions in offline RL. Not all OOD actions are detrimental; some lie just outside the behavioral support but have the potential to improve policy performance. The core motivation is to establish a mechanism that can accurately identify OOD actions without restrictive assumptions (like Gaussianity) and selectively regularize them—suppressing hazardous actions while encouraging promising explorations.

**Scientific Questions:**
1.  How can we reliably detect OOD actions in offline RL without relying on strong distributional assumptions (e.g., Gaussian) or unreliable likelihood estimates?
2.  How can we differentiate between "beneficial" OOD actions (those that improve performance while staying within state distribution) and "detrimental" OOD actions (those causing state distribution shift or value degradation)?
3.  How can we design a theoretical framework that ensures convergence (contraction mapping) and bounded value estimation while allowing for exploration beyond the behavioral support?

**3. Overall Core Idea and Design Philosophy**

**Overall Core Idea:** The paper proposes DOSER (Diffusion-based OOD Detection and SElective Regularization), a framework that advances OOD handling through two key innovations: diffusion-based OOD detection and adaptive selective regularization. 

**Design Philosophy:** 
1.  **Expressive Modeling:** Use diffusion models instead of VAEs to capture multi-modal behavior policies and state distributions, avoiding Gaussian assumptions.
2.  **Reconstruction as Indicator:** Leverage single-step denoising reconstruction error as a theoretically grounded, likelihood-free metric for OOD detection.
3.  **Selective Regularization:** Integrate a learned dynamics model to evaluate the predicted outcomes of OOD actions. Instead of binary ID/OOD classification, distinguish beneficial OOD actions from detrimental ones. Penalize detrimental actions while providing value compensation (bonus) for beneficial ones to encourage exploration toward high-potential regions.

**4. Core Innovation Points**

1.  **Diffusion-based OOD Detection:** A novel approach using reconstruction error from pretrained diffusion models (for behavior policy and state distribution) as a reliable OOD indicator. This provides a likelihood-free surrogate for distributional alignment and naturally captures multi-modal distributions without restrictive assumptions.
2.  **Adaptive OOD Action Classification:** A fine-grained mechanism that goes beyond binary classification. By evaluating predicted transitions via a dynamics model, it classifies OOD actions into beneficial sets (leading to in-distribution states with higher value) and detrimental sets (leading to OOD states or lower value).
3.  **Selective Regularization via Dual Strategy:** A dual regularization strategy that adaptively adjusts treatment of OOD actions. It penalizes detrimental OOD actions to mitigate overestimation while applying a value-difference bonus ($\delta V$) to beneficial OOD actions to compensate for extrapolation errors and guide policy improvement.
4.  **Theoretical Guarantees:** Theoretical proof that DOSER is a $\gamma$-contraction admitting a unique fixed point with bounded value estimates, along with asymptotic performance guarantees relative to the optimal policy under model approximation and OOD detection errors.

**5. Overview of the Overall Technical Solution**

The DOSER framework operates through the following pipeline:
1.  **Model Pretraining:** Train two diffusion models: a conditional one for the behavior policy $\pi_\beta(a|s)$ and an unconditional one for the state distribution $d_0(s)$. Concurrently, train a dynamics model $P_\psi(s'|s,a)$ via supervised regression.
2.  **OOD Detection:** For a state-action pair encountered during policy optimization, perturb the action with noise and compute the L2 reconstruction error using the pretrained diffusion models. Compare the error against thresholds ($\tau_a, \tau_s$) to determine if the action or resulting state is OOD.
3.  **OOD Action Classification:** If an action is detected as OOD, use the dynamics model to predict the next state $s'_\pi$. Compare this state against the training distribution (state OOD detection) and its value $V(s'_\pi)$ against the value of the predicted next state of the optimal in-distribution action $V(s'_{id})$. Classify as detrimental if OOD state or lower value; otherwise, beneficial.
4.  **Policy Optimization:** Update the Q-function using a modified Bellman target that applies $Q_{min}$ penalty to detrimental OOD actions and a value-compensated target ($\eta(Q(s, a^*_{id}) + \delta V)$) to beneficial OOD actions. Update the policy using maximum entropy regularization.

**6. Detailed Module Design**

**6.1 Diffusion-based Behavior and State Modeling**
The foundation establishes two diffusion models to capture the underlying distributions.
*   **Behavior Policy Modeling:** A conditional diffusion model learns the empirical behavior policy distribution $\hat{\pi}_\beta(a|s)$ by training a denoising network $\varepsilon_{\theta_a}(a_t, \sigma_t, s)$ to reconstruct the original action $a_0$.
*   **State Distribution Modeling:** A diffusion model captures the state distribution $d_0(s)$. The denoising network $\varepsilon_{\theta_s}(s_t, \sigma_t)$ recovers original states $s_0$ from noisy versions.

**6.2 OOD Detection via Reconstruction Error**
The detection mechanism leverages denoising capabilities to identify OOD samples.
*   **OOD Score Calculation:** Perturb the action $a_0$ as $a_t = a_0 + \sigma_t \varepsilon$. The OOD score is the L2 reconstruction error between the original action and its denoised counterpart. Analogously for states.
*   **Indicator Functions:** Formal binary indicators based on thresholds $\tau_a$ and $\tau_s$.
*   **Advantages:** 1) Likelihood-free surrogate; 2) Captures multi-modal distributions; 3) Efficient (single forward pass).

**6.3 Adaptive OOD Action Classification**
This module distinguishes between beneficial and detrimental OOD actions.
*   **Outcome Assessment:** For an OOD action $a_{ood}$, predict subsequent state $s'_\pi$. Assess if $s'_\pi$ is OOD (using state reconstruction error) and if $V(s'_\pi) \ge V(s'_{id})$.
*   **Classification Rule:** Defines Beneficial set $A^+_{ood}$ and Detrimental set $A^-_{ood}$.
*   **Selective Regularization:** Detrimental actions are penalized. Beneficial actions receive an adaptive bonus $\delta V = V(s'_\pi) - V(s'_{id})$.

**6.4 Practical Implementation**
*   **Value Learning:** Perform expectile regression to train the value network (similar to IQL).
*   **Dynamics Model:** Trained via supervised regression on offline dataset quadruples.
*   **Policy Learning:** Optimize actor network with maximum entropy regularization.

**7. All Mathematical Formulas and Symbol Definitions**

**Diffusion Process:**
Forward SDE:
$d x_t = f(x_t, t) d t+ g(t) d w_t$ (1)
Reverse Probability-Flow ODE:
$d x_t = -\dot{\sigma}(t) \sigma(t) \nabla_{x_t} \log p_t(x_t) d t$ (2)
Denoising Loss:
$\mathcal{L}(\theta) = \mathbb{E}_{\sigma_t, x_0 \sim p(x_0), \varepsilon \sim \mathcal{N}(0, I)} \left[ \lambda(\sigma_t) \| x_0 - \varepsilon_\theta(x_t, \sigma_t) \|_2^2 \right]$ (3)

**Behavior and State Modeling:**
Behavior Policy Loss:
$\mathcal{L}(\theta_a) = \mathbb{E}_{\sigma_t, (s, a_0) \sim \mathcal{D}, \varepsilon \sim \mathcal{N}(0, I)} \left[ \lambda(\sigma_t) \| a_0 - \varepsilon_{\theta_a}(a_t, \sigma_t, s) \|_2^2 \right]$ (4)
State Distribution Loss:
$\mathcal{L}(\theta_s) = \mathbb{E}_{\sigma_t, s \sim \mathcal{D}, \varepsilon \sim \mathcal{N}(0, I)} \left[ \lambda(\sigma_t) \| s_0 - \varepsilon_{\theta_s}(s_t, \sigma_t) \|_2^2 \right]$ (5)

**OOD Detection:**
Action Reconstruction Error:
$E_a(s, a_0) = \| a_0 - \varepsilon_{\theta_a}(a_t, \sigma_t, s) \|_2$ (6)
State Reconstruction Error:
$E_s(s_0) = \| s_0 - \varepsilon_{\theta_s}(s_t, \sigma_t) \|_2$ (7)
OOD Indicators:
$\mathbb{I}_{ood}(a_0) = \{E_a(s, a_0) > \tau_a\}, \mathbb{I}_{ood}(s_0) = \{E_s(s_0) > \tau_s\}$ (8)

**OOD Action Classification:**
$A^+_{ood} := \{ a \in \mathcal{A} \mid E_s(s'_\pi) \le \tau_s \land V(s'_\pi) \ge V(s'_{id}) \}$
$A^-_{ood} := \{ a \in \mathcal{A} \mid E_s(s'_\pi) > \tau_s \lor V(s'_\pi) < V(s'_{id}) \}$ (9)

**Critic Learning:**
$\mathcal{L}(\theta) = \mathbb{E}_{(s, a, s') \sim \mathcal{D}} \left[ \left( Q_\theta(s, a) - \left( R(s, a) + \gamma \mathbb{E}_{a' \sim \pi_\beta(\cdot|s')} [Q_{\theta'}(s', a')] \right) \right)^2 \right]$
$+ \beta \mathbb{E}_{s \sim \mathcal{D}, a \sim \pi_\phi(\cdot|s)} \left[ \mathbb{I}(a \in A^-_{ood}) \cdot (Q_\theta(s, a) - Q_{min})^2 \right]$
$+ \lambda \mathbb{E}_{s \sim \mathcal{D}, a \sim \pi_\phi(\cdot|s)} \left[ \mathbb{I}(a \in A^+_{ood}) \cdot (Q_\theta(s, a) - \eta (Q_{\theta'}(s, a^*_{id}) + \delta V))^2 \right]$ (10)
Optimal ID Action Approximation:
$\hat{a}^*_{id} = \arg \max_{a_i \sim \hat{\pi}_\beta(\cdot|s)} Q(s, a_i) \text{ for } i = 1, \ldots, N$ (11)

**Practical Implementation Losses:**
Value Learning:
$\mathcal{L}(\theta) = \mathbb{E}_{(s, r, s') \sim \mathcal{D}} [L^\tau_2(r + \gamma V_{\theta'}(s') - V_\theta(s)]$ (12)
Dynamics Model:
$\mathcal{L}(\psi) = \mathbb{E}_{(s, a, s') \sim \mathcal{D}} \| p_\psi(\cdot|s, a) - s' \|_2^2$ (13)
Policy Learning:
$\mathcal{L}(\phi) = \mathbb{E}_{s \sim \mathcal{D}, a \sim \pi_\phi(s)} [\alpha \log \pi_\phi(\cdot|s) - Q_\theta(s, a)]$ (14)

**Theoretical Derivations:**
Definition 2: $\mathcal{T}_{In} Q(s, a) := R(s, a) + \gamma \mathbb{E}_{s' \sim P(\cdot|s, a), a' \sim \hat{\pi}_\beta(\cdot|s')} [Q(s', a')]$ (15)
Definition 3: $\mathcal{T}_{DOSER} Q(s, a) = \begin{cases} \mathcal{T}_{In} Q(s, a) & \text{if } E_a(s, a) \le \tau_a \\ Q_{adj}(s, a) & \text{otherwise} \end{cases}$ (16)
$Q_{adj}(s, a) = \begin{cases} Q_{min} & \text{if } a \in A^-_{ood} \\ \eta (Q(s, a^*_{id}) + \delta V) & \text{if } a \in A^+_{ood} \end{cases}$ (17)

Theorem 1: $\| \mathcal{T}_{DOSER} Q_1 - \mathcal{T}_{DOSER} Q_2 \|_\infty \le \gamma \| Q_1 - Q_2 \|_\infty$ (18)
Proof Logic: 
1) In-distribution: $\| \mathcal{T}_{DOSER} Q_1 - \mathcal{T}_{DOSER} Q_2 \|_\infty \le \gamma \| Q_1 - Q_2 \|_\infty$.
2) Detrimental: $\| Q_{min} - Q_{min} \|_\infty = 0 \le \gamma \| Q_1 - Q_2 \|_\infty$.
3) Beneficial: $\| \eta(Q_1(s, a^*_{id,1}) + \delta V) - \eta(Q_2(s, a^*_{id,2}) + \delta V) \|_\infty \le \eta \| Q_1 - Q_2 \|_\infty \le \gamma \| Q_1 - Q_2 \|_\infty$.

Theorem 2: $Q_{min} \le Q^\pi_{DOSER}(s, a) \le Q^\pi_{In}(s, a^*_{id}) + \eta \delta V$ (19)
Proof Logic:
Lower bound: For in-distribution $Q \ge R_{min} + \gamma Q_{min} = Q_{min}$. For detrimental $Q=Q_{min}$. For beneficial $Q \ge \eta Q_{min} \ge Q_{min}$.
Upper bound: For in-distribution $Q^\pi_{In}(s, a) \le Q^\pi_{In}(s, a^*_{id}) + \eta \delta V$. For detrimental $Q_{min} \le Q^\pi_{In}(s, a^*_{id}) + \eta \delta V$. For beneficial $\eta(Q^\pi_{In}(s, a^*_{id}) + \delta V) \le Q^\pi_{In}(s, a^*_{id}) + \eta \delta V$.

Theorem 3: $\| \hat{Q} - Q^{\pi_{ref}} \|_\infty \le \frac{\gamma}{1-\gamma} (Q_{max}(C_1 \varepsilon_{dyn} + C_2 \varepsilon_{det}) + \eta \delta V)$ (20)

Theorem 4: $|J(\pi^*) - J(\hat{\pi})| \le \delta_f + \frac{C L_P R_{max}}{1-\gamma} (C_1 \varepsilon_{dyn} + C_2 \varepsilon_{det})$ (21)

**8. Algorithm Pseudocode**

**Algorithm 1 Diffusion-Based OOD Detection with selective regularization (DOSER)**
Initialize Q-network $Q_\theta$, V-network $V_\theta$, diffusion behavior model $\varepsilon_{\theta_a}$, diffusion state model $\varepsilon_{\theta_s}$, policy network $\pi_\phi$, dynamics model $p_\psi$, and target networks $Q_{\theta'}$, $V_{\theta'}$, $\pi_{\phi'}$

\# Model Pretraining
Pretraining dynamics model $p_\psi$ by minimizing (13)
Pretraining diffusion models $\varepsilon_{\theta_a}$ and $\varepsilon_{\theta_s}$ by minimizing (4) and (5)
Calculate OOD detection thresholds $\tau_a$ and $\tau_s$ based on in-sample reconstruction error

for each iteration do
    Sample transition minibatch $\{(s, a, r, s')\}$ from $\mathcal{D}$
    
    \# Critic Learning
    Generate action $a^\pi \sim \pi_\phi(s)$ and predict the next state $s'^\pi = p_\psi(s, a^\pi)$
    Select the best ID action $a^*_{id}$ and predict the next state $s'_{id} = p_\psi(s, a^*_{id})$
    Calculate the reconstruction errors of policy action and next state by (6) and (7)
    Calculate the adaptive bonus $\delta V = V_\theta(s'^\pi) - V_\theta(s'_{id})$
    Update $Q_\theta$ and $V_\theta$ by minimizing (10) and (12)
    
    \# Actor Learning
    Update $\pi_\phi$ by minimizing (14)
    
    \# Target Network Update
    $\theta' \leftarrow \rho \theta + (1-\rho)\theta'$, $\phi' \leftarrow \rho \phi + (1-\rho)\phi'$
end for