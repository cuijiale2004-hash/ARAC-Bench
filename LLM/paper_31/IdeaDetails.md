### 1. Research Background and Existing Pain Points

**Research Background:**
Large language models (LLMs) are increasingly utilized for recommendation by framing the task as sequence generation: given a user’s interaction history, the model autoregressively generates the next item tokens. In practice, the LLM is fine-tuned on user histories paired with their next interactions to align with the recommendation objective. However, real-world interaction data are continuously collected and evolve over time: new users and items appear, and user preferences drift. Periodic retraining from scratch on both historical and new data is highly inefficient, making continual learning (i.e., updating the model effectively with new data) a natural and appealing solution.

**Existing Pain Points:**
1.  **Unique Stability-Plasticity Challenge in Recommendation:** While continual learning generally requires balancing stability (retaining past knowledge) and plasticity (adapting to new knowledge), recommendation presents a unique interpretation. Unlike other domains (e.g., computer vision) where tasks are disjoint and the goal is preserving performance on previous tasks, continual recommendation aims to accurately capture *evolving* user preferences to predict future interests. Stability refers to preserving long-term user preferences that remain predictive, while plasticity is required to *overwrite* outdated preferences. Outdated preferences can even harm performance if current user interests have shifted significantly.
2.  **Limitations of Single Evolving LoRA:** A simple intuitive approach is maintaining a single evolving LoRA: sequentially fine-tuning one adapter initialized from the previous stage. While this provides strong plasticity and partial preservation via parameter inheritance, it inevitably overwrites useful past knowledge during fine-tuning, leading to forgetting.
3.  **Limitations of Cumulative LoRA:** Cumulative LoRA methods (widely used in vision) use the sum of the new trainable adapter and all frozen past adapters. This design enhances stability by reusing prior adapters and works well when tasks are independent. However, in recommendation where user preferences evolve, frozen adapters entangle outdated and relevant preferences, making them hard to disentangle. This design implicitly assumes task independence, which is ill-suited for recommendation. Additionally, cumulative LoRA incurs growing storage costs and struggles to reflect relative importance during aggregation.

### 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:**
The core motivation is to address the distinctive stability-plasticity trade-off in continual recommendation systems based on LLMs. The goal is not merely to preserve past performance (as in standard continual learning) but to selectively retain long-term predictive preferences while actively adapting to or overwriting outdated preferences as user interests drift. The motivation stems from the empirical finding that cumulative LoRA, while effective in user-disjoint settings, underperforms in natural chronological settings where preferences evolve, whereas single evolving LoRA suffers from forgetting.

**Scientific Questions:**
How to design a continual adaptation method for LoRA in recommendation that avoids the rigidity of cumulative adapters (which entangle outdated signals) and the forgetting of single evolving adapters, while providing a mechanism that flexibly balances adaptation (plasticity) and preservation (stability) based on the support of current data?

### 3. Overall Core Idea and Design Philosophy

**Overall Core Idea:**
The paper proposes PESO (Proximally rEgularized Single evolving lOra), a continual adaptation method for LoRA in recommendation. PESO maintains a single evolving LoRA adapter and regularizes each update by keeping the current adapter close to the previous one using a proximal term. The balance is achieved through the natural competition between the data-fitting loss (which pulls toward the optimal state for the new data) and the proximal term (which pulls toward the previous state), allowing the model to decide what to adapt or retain based on data support.

**Design Philosophy:**
1.  **Avoid multiple adapters:** Using multiple adapters (like cumulative LoRA) implicitly assumes task independence, which is harmful in recommendation where stages interfere and preferences evolve.
2.  **Preserve past knowledge supportively:** Preserve past knowledge in a way that supports understanding of current user behavior, rather than rigidly reusing frozen past adapters that may contain outdated signals.

### 4. Core Innovation Points

1.  **Analysis of Distinctive Stability-Plasticity Challenge:** The paper identifies and empirically demonstrates that cumulative LoRA, while effective in simulated user-disjoint settings, underperforms in the natural case where user preferences evolve across time stages, due to the entanglement of outdated and relevant preferences.
2.  **PESO Method:** Proposes PESO, a single evolving LoRA with a proximal regularizer that anchors each update to the previous state. This allows flexible balancing of stability and plasticity through the competition between the loss and the proximal term.
3.  **Theoretical Proof of Data-Aware Guidance:** Theoretically shows that the proximal design provides data-aware, direction-wise guidance in the LoRA subspace. Specifically, it interpolates between the new optimum and the previous adapter based on the generalized eigenvalues of the data covariance and the proximal metric.
4.  **Per-Module Softmax-KL Instantiation:** Instantiates the general proximal framework with a per-module softmax–KL divergence, which preserves internal module structure (interpreted as a p-weighted variance) rather than treating all parameters equally (like L2), providing a more nuanced stability mechanism.

### 5. Overview of the Overall Technical Solution

The overall technical solution is PESO, which operates within the LoRA subspace of an LLM-based recommender.
1.  **Data Structure:** Interaction data is divided into chronological blocks $D_1, \ldots, D_T$. The model is pretrained on $D_1$ and fine-tuned sequentially on $D_2, \ldots, D_T$.
2.  **Parameterization:** PESO maintains a single LoRA adapter $v_t$ (concatenated parameters of $A$ and $B$ matrices) for all modules.
3.  **Initialization:** At each stage $t$, $v_t$ is initialized from the previous state $v_{t-1}$.
4.  **Optimization:** The model is fine-tuned using a combined loss: the standard autoregressive cross-entropy loss on the new data $D_t$, plus a proximal regularizer that penalizes deviation from $v_{t-1}$.
5.  **Regularizer Choice:** The proximal regularizer is instantiated using a block-wise Softmax-KL divergence, which acts as a p-weighted variance penalty on parameter changes within each module, preserving the internal structure of the LoRA factors.
6.  **Theoretical Effect:** This setup ensures that along directions where the new data provides strong support, the update moves toward the new optimum; along directions with weak support, the update stays close to the previous state.

### 6. Detailed Module Design

**1. Data and Task Definition Module:**
*   **Input:** At stage $t$, the data consists of user interaction sequences $E_t = \{(x_{u,1}, \ldots, x_{u,N_u})\}_{u \in U_t}$.
*   **Formatting:** Data is presented as prefix-target pairs $D_t = \{(x_u, y_u)\}$ where $x_u$ is the history and $y_u$ is the next item (represented by semantic IDs).
*   **Base Loss:** The stage-$t$ training loss is the autoregressive cross-entropy: $L_{ce}^{D_t} = \mathbb{E}_{(x,y)\sim D_t}[-\log p_\theta(y | x)]$.

**2. LoRA Parameterization Module:**
*   **Structure:** Freezes pretrained LLM weight $W_0$ and adds a trainable low-rank update $\Delta W = BA$.
*   **Single Evolving Logic:** $W_t = W_0 + B_t A_t$, with initialization $B_t \leftarrow B_{t-1}, A_t \leftarrow A_{t-1}$ for $t \geq 2$.
*   **Grouping:** LoRA parameters are partitioned into groups $g \in \{1, \ldots, G\}$ corresponding to flattened parameters of $A$ or $B$ for one LoRA-injected module (e.g., attention projections q/k/v/o). The full parameter vector is $v_t$.

**3. Proximal Regularization Module (General Framework):**
*   **Objective:** $L_t = L_{ce}^{D_t} + \frac{\lambda}{2} \sum_{g=1}^G \| v_t^{(g)} - v_{t-1}^{(g)} \|_{H_{t-1}^{(g)}}^2$, with $v_t \leftarrow v_{t-1}$ at init.
*   **Metric:** $H_{t-1}^{(g)} \succeq 0$ is a symmetric PSD metric fixed during stage $t$, which can be precomputed at $v_{t-1}^{(g)}$.

**4. Softmax-KL Proximal Instantiation Module:**
*   **Objective:** Replaces the quadratic norm with a Softmax-KL divergence applied per group.
*   **Formula:** $L_t = L_{ce}^{D_t} + \lambda \sum_{g=1}^G D_{KL}(\text{softmax}(v_t^{(g)}) \| \text{softmax}(v_{t-1}^{(g)}))$.
*   **Mechanism:** This penalizes reshuffling within each module’s LoRA factor and deviations for coordinates with higher prior mass. It locally reduces to a quadratic form with metric $H_{t-1}^{(g)} = \text{diag}(p^{(g)}) - p^{(g)}(p^{(g)})^\top$ where $p^{(g)} = \text{softmax}(v_{t-1}^{(g)})$.

**5. Theoretical Analysis Module (Direction-wise Guidance):**
*   **Linearization:** Uses tangent features $\Phi(x) = U^\top \nabla_\theta s(\theta_0, x)$ to approximate the loss.
*   **Second-Moment Matrix:** $\Sigma_t = \mathbb{E}_{x \sim D_t}[\Phi(x)\Phi(x)^\top]$ captures data support.
*   **Interpolation:** The solution $\hat{v}_t$ interpolates between $v_t^*$ (new optimum) and $v_{t-1}$ (previous adapter) along generalized eigenvectors of $(\Sigma_t, H_{t-1})$. The weight towards $v_t^*$ is determined by the ratio of data support ($\rho_k$) to regularization ($\lambda$).

### 7. All Mathematical Formulas and Symbol Definitions

**Notations and Data:**
*   $U_t$: Set of active users at stage $t$.
*   $I_t$: Item set.
*   $E_t = \{(x_{u,1}, \ldots, x_{u,N_u})\}_{u \in U_t}$: Collection of user interaction sequences.
*   $D_t = \{(x_u, y_u) : u \in U_t, N_u \geq 2\}, x_u = (x_{u,1}, \ldots, x_{u,N_u-1}), y_u = x_{u,N_u}$: Stage-$t$ data as prefix-target pairs.
*   $y = (y_1, \ldots, y_M)$: Semantic-ID token sequence of the target item.

**Base Loss Function:**
$$L_{ce}^{D_t} = \mathbb{E}_{(x,y)\sim D_t}[-\log p_\theta(y | x)], \quad p_\theta(y | x) = \prod_{m=1}^M p_\theta(y_m | x, y_{<m})$$

**LoRA Definition:**
$$\Delta W = BA, \quad A \in \mathbb{R}^{r \times d_{in}}, \quad B \in \mathbb{R}^{d_{out} \times r}, \quad r \ll \min(d_{in}, d_{out})$$

**Single Evolving LoRA:**
$$W_t = W_0 + B_t A_t, \quad B_t \leftarrow B_{t-1}, \quad A_t \leftarrow A_{t-1}, \quad t \geq 2$$

**Cumulative LoRA:**
$$W_t = W_0 + \sum_{i=1}^{t-1} \alpha_i \hat{B}_i \hat{A}_i + B_t A_t, \quad t \geq 2$$
Where $\hat{B}_i = B_i/\|B_i\|_F$ and $\hat{A}_i = A_i/\|A_i\|_F$.

**General Proximal Framework (PESO):**
$$L_t = L_{ce}^{D_t} + \frac{\lambda}{2} \sum_{g=1}^G \| v_t^{(g)} - v_{t-1}^{(g)} \|_{H_{t-1}^{(g)}}^2, \quad v_t \leftarrow v_{t-1} \text{ at init}$$
Where $\|z\|_H^2 := z^\top H z$, $\lambda > 0$, and $H_{t-1}^{(g)} \succeq 0$.

**Theoretical Setup:**
*   $\theta_0$: Parameter vector after training on first data block.
*   $\theta(v) = \theta_0 + Uv$ with $U \in \mathbb{R}^{d \times m}$.
*   Linearization: $s(\theta_0 + Uv, x) \approx s(\theta_0, x) + \Phi(x)^\top v$
*   Tangent features: $\Phi(x) := U^\top \nabla_\theta s(\theta_0, x) \in \mathbb{R}^m$
*   Quadratic approximation: $L_{D_t}(v) \approx \frac{1}{2} (v - v_t^*)^\top \Sigma_t (v - v_t^*), \quad \Sigma_t = \mathbb{E}_{x \sim D_t}[\Phi(x)\Phi(x)^\top] \succeq 0$

**Proposition 1 (Generalized–eigen interpolation):**
$$L_t(v) = \frac{1}{2} (v - v_t^*)^\top \Sigma_t (v - v_t^*) + \frac{\lambda}{2} (v - v_{t-1})^\top H_{t-1} (v - v_{t-1})$$
Let $\{(q_k, \rho_k)\}_{k=1}^r$ be generalized eigenpairs of $(\Sigma_t, H_{t-1})$ on $\text{range}(H_{t-1})$ (i.e., $\Sigma_t q_k = \rho_k H_{t-1} q_k$), normalized by $q_i^\top H_{t-1} q_j = \delta_{ij}$.
$$\langle \hat{v}_t, q_k \rangle_{H_{t-1}} = \frac{\rho_k}{\rho_k + \lambda} \langle v_t^*, q_k \rangle_{H_{t-1}} + \frac{\lambda}{\rho_k + \lambda} \langle v_{t-1}, q_k \rangle_{H_{t-1}}, \quad k = 1, \ldots, r$$

**Corollary 2 (L2 special case):**
Taking $H_{t-1} = I$. If $\Sigma_t q_k = \sigma_k^2 q_k$ with $\{q_k\}$ orthonormal:
$$\langle \hat{v}_t, q_k \rangle = \frac{\sigma_k^2}{\sigma_k^2 + \lambda} \langle v_t^*, q_k \rangle + \frac{\lambda}{\sigma_k^2 + \lambda} \langle v_{t-1}, q_k \rangle, \quad k = 1, \cdots, m$$

**Softmax-KL Instantiation:**
$$L_t = L_{ce}^{D_t} + \lambda \sum_{g=1}^G D_{KL}(\text{softmax}(v_t^{(g)}) \| \text{softmax}(v_{t-1}^{(g)})), \quad v_t \leftarrow v_{t-1} \text{ at init}$$

**Proposition 3 (Per-module softmax–KL is locally quadratic):**
Let $p^{(g)} = \text{softmax}(v_{t-1}^{(g)})$ and $\Delta^{(g)} = v_t^{(g)} - v_{t-1}^{(g)}$.
$$K_{blk}(v_t, v_{t-1}) = \frac{\lambda}{2} \sum_{g=1}^G (\Delta^{(g)})^\top (\text{diag}(p^{(g)}) - p^{(g)}(p^{(g)})^\top) \Delta^{(g)} + o(\|\Delta\|^2)$$
$$= \frac{\lambda}{2} \Delta^\top \text{blkdiag}(H_{t-1}^{(1)}, \ldots, H_{t-1}^{(G)}) \Delta + o(\|\Delta\|^2)$$
With $H_{t-1}^{(g)} = \text{diag}(p^{(g)}) - p^{(g)}(p^{(g)})^\top \succeq 0$.

**Corollary 4 (Softmax–KL equals p-weighted variance):**
$$K_{blk}(v_t, v_{t-1}) = \frac{\lambda}{2} \sum_{g=1}^G \text{Var}_{p^{(g)}}(\Delta^{(g)})$$
Where $\text{Var}_{p^{(g)}}(\Delta^{(g)}) = \sum_{i \in g} p_i^{(g)} (\Delta_i^{(g)} - \mu^{(g)})^2$ and $\mu^{(g)} = \sum_{i \in g} p_i^{(g)} \Delta_i^{(g)}$.

**Conceptual Model (Appendix A):**
$$P_t(y | x_{u}^{t-1}) \approx \alpha_t P_{t-1}(y | x_{u}^{t-1}) + (1-\alpha_t) Q_t(y | x_{u}^{t-1})$$
Where $P_{t-1}$ captures stability, $Q_t$ captures plasticity, and $\alpha_t \in [0, 1]$.

**Detailed Proof Formulas (Appendix B):**
*   **Eq 25:** $\nabla_v L_t(v) = \Sigma_t (v - v_t^*) + \lambda H_{t-1} (v - v_{t-1})$
*   **Eq 26:** $(\Sigma_t + \lambda H_{t-1}) \hat{v}_t = \Sigma_t v_t^* + \lambda H_{t-1} v_{t-1}$
*   **Eq 28:** $\hat{v}_t = (\Sigma_t + \lambda H_{t-1})^{-1} (\Sigma_t v_t^* + \lambda H_{t-1} v_{t-1})$
*   **Proposition 5 (Local quadratic form of softmax-KL):**
    $K(\Delta) := D_{KL}(\text{softmax}(v_{t-1} + \Delta) \| \text{softmax}(v_{t-1}))$
    $K(\Delta) = \frac{1}{2} \Delta^\top (\text{diag}(p) - p p^\top) \Delta + o(\|\Delta\|^2)$
    $= \frac{1}{2} \sum_{i=1}^n p_i (\Delta_i - \mu)^2 + o(\|\Delta\|^2)$

### 8. Algorithm Pseudocode

The paper does not provide an explicit algorithm pseudocode block in the text. However, the implementation process is described as follows based on the method description:

**Iterative Process of PESO:**
1.  **Input:** Pretrained LLM with weights $W_0$; Data sequence $D_1, \ldots, D_T$; LoRA rank $r$; Regularization strength $\lambda$.
2.  **Pretraining (Stage $t=1$):** Fine-tune LoRA matrices $A_1, B_1$ on $D_1$ using loss $L_{ce}^{D_1}$. Freeze $W_0$. Store $v_1$ (concatenated parameters).
3.  **For Stage $t = 2$ to $T$:**
    a.  **Initialize:** Set current LoRA parameters $v_t \leftarrow v_{t-1}$.
    b.  **Compute Proximal Metric:** For each module group $g$, compute $p^{(g)} = \text{softmax}(v_{t-1}^{(g)})$.
    c.  **Optimize:** Fine-tune $v_t$ on $D_t$ by minimizing the loss:
        $$L_t = L_{ce}^{D_t} + \lambda \sum_{g=1}^G D_{KL}(\text{softmax}(v_t^{(g)}) \| \text{softmax}(v_{t-1}^{(g)}))$$
    d.  **Update State:** Store $v_t$ as the new previous state for the next stage.
4.  **Inference:** At any stage $t$, use $W_0 + B_t A_t$ for autoregressive generation of recommendations.