### 1. Research Background and Existing Pain Points

**Research Background:** Combining diverse foundation models is a promising direction for enhancing AI capabilities. Currently, two primary approaches exist for model fusion: micro-level fusion in parameter space and macro-level fusion in data-flow space.

**Existing Pain Points:**
1.  **Inefficiency of Scaling Laws:** Scaling model size, training tokens, and compute according to empirical scaling laws is resource-intensive and uncertain in yielding sustained returns.
2.  **Limitations of Micro-level Model Merging:** Weight-merging approaches are limited by mismatched architectures and the closed-source nature of many high-performing models. They require access to model weights and compatibility, excluding the strongest available closed-source models (e.g., GPT-5, Claude-4, Gemini-2.5-pro).
3.  **Limitations of Macro-level Fusion:** Existing macro-level methods (like Mixture of Agents, Multi-Agent Debate, or routing) rely on expensive multi-model inference or static, human-designed collaboration patterns (scaffolding). Simple or heuristic-based routing approaches cannot easily achieve sophisticated organizational decisions and sometimes degrade performance below random baselines.
4.  **Central Challenge for Coordinators:** The central challenge for a macro-level coordinator is acquiring a rich contextual understanding of a given query to make effective decisions. Existing works underexplore whether sufficient semantic signal can be extracted efficiently from internal representations of small language models for this purpose.

### 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:** To fuse the complementary strengths of multiple state-of-the-art models from diverse providers without modifying their weights (test-time model composition via coordination), leveraging prior data and training investments to deliver performance improvements without retraining individual models.

**Scientific Questions:**
1.  Can contextual representations from a small language model (SLM) contain sufficient semantic signal for a lightweight head to coordinate multiple LLMs effectively?
2.  How can the representation-to-coordination mapping be optimized efficiently given the challenges of high dimensionality, weak parameter correlations, and high per-step cost?
3.  Can a coordinator offload complex skill acquisition to diverse LLMs while maintaining extreme parameter efficiency (under 20K learnable parameters)?

### 3. Overall Core Idea and Design Philosophy

**Overall Core Idea:** TRINITY is a lightweight and adaptive framework for coordinating multiple diverse LLMs. It uses a coordinator comprising a compact language model (SLM) and a lightweight head (~10K parameters), optimized with an evolutionary strategy (sep-CMA-ES). TRINITY processes queries over multiple turns, assigning one of three roles (Thinker, Worker, Verifier) to a selected LLM at each turn.

**Design Philosophy:**
1.  **Efficient Contextualization:** Rich contextual signals for coordination can be efficiently extracted from the hidden states of a compact language model, specifically the penultimate token's hidden state.
2.  **Role-based Offloading:** The coordinator need not be as capable as the underlying agents; it orchestrates them by assigning distinct roles (Thinker, Worker, Verifier) to offload complex skill acquisition from the coordinator itself.
3.  **Evolutionary Optimization:** In regimes of high dimensionality, weak parameter correlations, and budget constraints, derivative-free Covariance Matrix Adaptation Evolution Strategy (CMA-ES) with diagonal covariance (sep-CMA-ES) is superior to RL and random search due to the problem's block-ε-separability.

### 4. Core Innovation Points

1.  **Lightweight and Effective Coordination Mechanism:** Rich contextual signals from the hidden states of an SLM are sufficient for a tiny head (total learnable parameters under 20K) to coordinate multiple diverse LLMs, a previously underexplored approach to model composition.
2.  **Highly Efficient Training Methodology:** Theoretical and empirical evidence demonstrates that under challenging, budget-constrained conditions, sep-CMA-ES is a superior optimization choice over RL, imitation learning, and random search, leveraging the problem's block-ε-separability.
3.  **Tri-Role Coordination Protocol:** A novel multi-turn protocol assigning three distinct roles—Thinker (strategize), Worker (execute), and Verifier (evaluate)—effectively offloads complex skill acquisition from the coordinator and structures the collaboration.
4.  **State-of-the-Art Performance and Generalization:** TRINITY sets a new record on LiveCodeBench (86.2%) and outperforms existing methods on a wide range of benchmarks while robustly zero-shot generalizing to unseen tasks and developing emergent, task-aware coordination strategies.
5.  **Efficient Parametrization:** A singular value fine-tuning approach adapted for a small set of the backbone’s layers, keeping orthogonal matrices fixed and only learning singular value scales, yielding representational benefits with extreme parameter efficiency.

### 5. Overview of the Overall Technical Solution

TRINITY consists of a **Coordinator** (SLM + Lightweight Head) and a **Pool of LLMs**. The process is a multi-turn interaction protocol:
1.  **Input:** The user query is passed to the Coordinator.
2.  **Coordinator Action:** The Coordinator (SLM) processes the full conversation transcript and passes the hidden state of the penultimate token to the Lightweight Head.
3.  **Selection:** The Head outputs logits to select an Agent (LLM) and a Role (Thinker, Worker, or Verifier).
4.  **Message Processing:** A role-specific prompt is injected before the request is sent to the chosen LLM.
5.  **LLM Response:** The selected LLM generates a response (message).
6.  **Transcript Update:** The response is post-processed and appended to the transcript.
7.  **Termination:** The process halts when the Verifier is selected and accepts the current response as the final answer, or when a fixed-turn budget is exhausted.

### 6. Detailed Module Design

**6.1 Coordinator Parametrization**
*   **Backbone (SLM):** A pre-trained SLM (Qwen3-0.6B) is used as the backbone. It maps interaction states $s$ to a representation state $h(s)$.
*   **Singular Value Fine-tuning:** For a selected subset of the SLM’s weight matrices, a singular value decomposition is performed. Only the singular value scales are learned; orthogonal matrices are fixed.
*   **Lightweight Head:** Appended directly after the SLM’s final hidden layer. It takes the hidden state $h$ corresponding to the penultimate output token as input.
    *   **Output Structure:** Projects hidden state to $L + 3$ logits ($L$ for LLM selection, 3 for Role selection).
    *   **Head Variants:**
        *   *Linear Head:* Direct affine mapping $z = Wh$.
        *   *Low-rank Head:* Factorized bottleneck with nonlinearity $u = ELU(Uh)$, $z = Vu \cdot \sigma$.
        *   *Sparse Head:* Learnable dimension-selection $z = W(h \odot \alpha)$.
        *   *Block-diagonal Head:* Structured projection with disjoint blocks $W = diag(W_1, \dots, W_B)$.

**6.2 Tri-Role Coordination**
*   **Interaction State:** Transcript after $k-1$ turns is $C_{k-1} = (Q, O_1, \dots, O_{k-1})$.
*   **Roles:**
    *   **Thinker (T):** Strategizes. Returns meta-level guidance, plans, decompositions, or critiques.
    *   **Worker (W):** Executes. Produces actionable content (derivation, code, result).
    *   **Verifier (V):** Evaluates. Checks correctness/completeness. Outputs judgment $u_k \in \{\text{ACCEPT, REVISE}\}$ and optional diagnosis $\delta_k$.
*   **Termination:** $\tau = \min\{ k \le K : R_k = V \text{ and } u_k = \text{ACCEPT} \}$, with $\tau = K$ if no acceptance. Final answer is $O_\tau$.

**6.3 Learning with Evolutionary Strategy (sep-CMA-ES)**
*   **Algorithm:** Diagonal Covariance Matrix Adaptation Evolution Strategy (sep-CMA-ES).
*   **Mechanism:** Iteratively improves a central "parent" policy by sampling a population of perturbed parameter vectors, evaluating each to obtain a fitness score, and recombining via fitness-weighted averaging.
*   **Theoretical Basis:** The optimization objective exhibits strong block-ε-separability, favoring diagonal methods where informative signal is concentrated within blocks and inter-block interference is negligible.

### 7. All Mathematical Formulas and Symbol Definitions

**7.1 Problem Formulation**
Let $S$ be the set of interaction states. SLM maps $s$ to $h(s) \in H \subset \mathbb{R}^d$. Coordination head with parameters $\theta \in P \subset \mathbb{R}^n$ takes $h(s)$ and outputs logits over finite action set $A$:
$$f_\theta : H \to \mathbb{R}^{|A|}, \pi_\theta(a | s) \propto \exp\left(f_\theta(h(s))_a\right), a \in A$$

Trajectory $\tau = (s_0, a_0, \dots, s_T)$ with horizon $T \le B_{\text{turn}}$. Terminal reward $R(\tau) \in \{0, 1\}$. Optimization objective:
$$J(\theta) := \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$$

**7.2 Head Architectures**
*   **Linear:**
    $$z = Wh, W \in \mathbb{R}^{n_a \times d_h}$$
*   **Low-rank:**
    $$u = ELU(Uh), ELU(x) = \begin{cases} x, x \ge 0 \\ \alpha(e^x - 1), x < 0 \end{cases}, \alpha = 0.1$$
    $$z = Vu \cdot \sigma$$
    Initialization:
    $$U \sim \mathcal{U}\left(-\sqrt{\frac{6}{d_h+r}}, \sqrt{\frac{6}{d_h+r}}\right), V \sim \mathcal{U}\left(-\sqrt{\frac{18}{r+n_a}}, \sqrt{\frac{18}{r+n_a}}\right)$$
*   **Sparse:**
    $$z = W(h \odot \alpha), W \in \mathbb{R}^{n_a \times d_h}$$
    Target active dimensions: $k = \max\left(1, \lfloor d_h \cdot (1 - \text{sigmoid}(\rho))\rfloor\right)$.
    Differentiable mask:
    $$\tilde{s} = (s + \epsilon)/\tau, \epsilon \sim \text{Gumbel}(0, 1)$$
    $$\alpha_{\text{soft}} = \text{TopK}_{\text{soft}}(\tilde{s}, k), \alpha = \frac{\alpha_{\text{soft}} \cdot k}{\sum_{i=1}^{d_h} \alpha_{\text{soft},i}}$$
*   **Block-diagonal:**
    $$W = \begin{bmatrix} W_1 & 0 & \cdots & 0 \\ 0 & W_2 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & W_B \end{bmatrix}, z = \begin{bmatrix} W_1h_1 \\ W_2h_2 \\ \vdots \\ W_Bh_B \end{bmatrix}$$

**7.3 Theoretical Analysis of Sep-CMA-ES (Appendix A.1)**
*   **Notations:** $g(\theta) := -J(\theta)$, $H(\theta) := \nabla^2 g(\theta)$. Diagonal scaling $D_t = \text{diag}(\sqrt{s_{1,t}}, \dots, \sqrt{s_{n,t}}) \succ 0$, $y = m_t + \sigma_t D_t z$. Whitened chart: $x = D_t^{-1}(y - m_t)$.
*   **Definition 1 (Hessian-based block-ε separability):** Exists scaling $S = \text{diag}(s_1, \dots, s_n) \succ 0$ such that $H_S(\theta) := S^{1/2}H(\theta)S^{1/2}$ is uniformly nearly block-diagonal. With $D(\theta) := \text{diag}(H_S(\theta))$, one of:
    $$\sup_{\theta \in D} \left\| D(\theta)^{-1/2} \text{offinter}\left(H_S(\theta)\right) D(\theta)^{-1/2} \right\|_2 \le \varepsilon_H \quad \text{(B1)}$$
    $$\sup_{\theta \in D} \max_{i \in B_p, j \in B_q, p \ne q} \frac{|[H_S(\theta)]_{ij}|}{\sqrt{[H_S(\theta)]_{ii}[H_S(\theta)]_{jj}}} \le \varepsilon_H \quad \text{(B2)}$$
    $$\sup_{\theta \in D} \max_{i \in B_p} \sum_{j \in B_q, q \ne p} \frac{|[H_S(\theta)]_{ij}|}{[H_S(\theta)]_{ii}} \le \varepsilon_H (< 1) \quad \text{(B3)}$$
*   **Assumption 1 (Diagonal comparability):** Exists $c_{\text{cmp}}, C_{\text{cmp}} > 0$ such that $\forall t, i$: $c_{\text{cmp}} \le \frac{s_i}{s_{i,t}} \le C_{\text{cmp}}$.
*   **Definition 2 (Metric–alignment factor):** $\chi(u, D) := \frac{(u^\top D u)^2}{u^\top D^2 u} \in \left[\frac{1}{\kappa_D}, 1\right]$.
*   **Assumption 2 (Local linear score):** $J(m_t + \sigma D_t z) = \frac{1}{2} + \gamma \sigma \langle u_t, z \rangle + \xi_t(z)$, $|\xi_t(z)| \le (L_{\text{curv}} + c_H \varepsilon_H)\sigma^2 \|z\|^2$.
*   **Definition 3 (Rank attenuation):** Let $Z^\star_\parallel := \min_{1 \le k \le N} \langle u_t, z^{(k)} \rangle$ and $x^- := \min\{x, 0\}$.
    $$\tilde{\rho}^2_{RS} := \frac{\mathbb{E}[(Z^\parallel_{sel})^2 -]}{\mathbb{E}[(Z^\star_\parallel)^2 -]} \in [0, 1], \quad \tilde{\rho}^2_{CMA} := \frac{\mathbb{E}\left[\left\langle u_t, \sum_{j=1}^\mu w_j z_{j:\lambda}^{(\hat{q}m_{CMA})}\right\rangle^2\right]}{\mathbb{E}\left[\left\langle u_t, \sum_{j=1}^\mu w_j z_{j:\lambda}^{(-Z_\parallel)}\right\rangle^2\right]} \in [0, 1]$$
*   **Assumption 3 (Metric–alignment comparability):** Exists $C_\chi \ge 1$: $\frac{1}{C_\chi} \le \frac{\chi(u_t, D_t)/\kappa_D(t)}{\chi(u_0, D_0)/\kappa_D(0)} \le C_\chi$.
*   **RS Misorder Bound:** $\Pr[\hat{f}(m_t + \sigma D_t z_1) \le \hat{f}(m_t + \sigma D_t z_2) \text{ but } J(\dots) > J(\dots)] \le C e^{-c m \Delta^2} + \Pr(\text{curv} > \frac{\Delta}{2})$.
*   **CMA Per-iteration Gain:** Let $\kappa_{\mu,\lambda} := \frac{\alpha^2_{\mu,\lambda}}{\beta_{\mu,\lambda}} = \Theta(1/n)$ and $\bar{\kappa}_{\mu,\lambda} := n \kappa_{\mu,\lambda} = \Theta(1)$.
    $$\frac{\mathbb{E}[r^2_t - r^2_{t+1}]}{r^2_t} \ge \chi(u_t, D_t) \cdot \frac{1}{\kappa_D(t)} \kappa_{\mu,\lambda} \tilde{\rho}^2_{CMA} - C \varepsilon_H \quad \text{(Eq 2)}$$
*   **Proposition 1:** Fix $T \in [2, 60]$. CMA budget $B_{\text{env}} = m_{\text{CMA}} \lambda T$. If $\tilde{\rho}_{\text{CMA}}/\tilde{\rho}_{\text{RS}} \ge \eta$:
    $$\frac{\text{CMA gain in } J}{\text{RS gain in } J} \gtrsim \frac{\bar{\kappa}_{\mu,\lambda}}{2} \cdot \frac{T}{\ln\left(\max\{e, \lfloor(m_{\text{CMA}} \lambda / m_{\text{RS}})T\rfloor\}\right)} \cdot \eta^2 - \frac{C}{\ln\left(\max\{e, \lfloor(m_{\text{CMA}} \lambda / m_{\text{RS}})T\rfloor\}\right)}$$
*   **Proposition 2:** Under Def 1, Assump 1, 2, 3, and $\tilde{\rho}^2_{CMA} = 1 - O(\varepsilon_H)$, sep-CMA-ES achieves per-iteration contraction:
    $$\frac{\bar{\kappa}_{\mu,\lambda}}{n} (1 - O(\varepsilon_H))$$
    i.e., $\mathbb{E}[r^2_T] \lesssim \exp(-c' T / n) r^2_0$.

### 8. Algorithm Pseudocode

The paper does not provide a formal pseudocode block, but the algorithmic flow of TRINITY is detailed in Section 3.2 and Figure 1 as follows:

**Algorithm Flow: TRINITY**
1.  **Input:** User Query $Q$, Maximum turns $K$, LLM Pool $\mathcal{M}$.
2.  Initialize Transcript $C_0 = (Q)$.
3.  **For** $k = 1, \dots, K$:
    a.  **Coordinator Inference:** Pass $C_{k-1}$ through SLM to obtain hidden state $h$.
    b.  **Head Prediction:** Feed $h$ to Lightweight Head to select Agent $A_k \in \mathcal{M}$ and Role $R_k \in \{T, W, V\}$.
    c.  **Message Processing:** Prepare role-specific prompt based on $C_{k-1}$ and $R_k$.
    d.  **Agent Query:** Query $A_k$ with prompted input to obtain message $M_k$.
    e.  **Post-processing:** Lightly post-process $M_k$ into $O_k$. Update transcript $C_k = C_{k-1} \cup \{O_k\}$.
    f.  **Termination Check:** If $R_k = V$ (Verifier) and judgment $u_k = \text{ACCEPT}$:
        *   Return $O_k$ as final answer.
4.  **Return** $O_K$ (Output at turn limit).