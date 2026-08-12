### 1. Research Background and Existing Pain Points
**Research Background:** Data-driven models trained by empirical risk minimization (ERM) are often prone to shortcut learning, indiscriminately exploiting any correlation, including spurious cues unrelated to the true causal mechanisms. This pattern is widely documented across modalities and tasks, where models rely on shallow artifacts instead of robust features. The rise of Transformers and Large Language Models (LLMs) sharpens this tension: these systems can sometimes rely on shallow artifacts, yet they also produce answers that appear strikingly logical and robust in certain scenarios. Recent theory offers partial clues for why Transformers sometimes appear causal, but does not yet answer the cause only generalization question. On stylized in-context tasks, Transformers trained on Markov Chain sequences can recover parent sets and estimate transition probabilities in-context, but they rely on designed ICL setups and do not show when spurious information is suppressed at both train and test time. In parallel, large margin analyses show that gradient descent (GD) pushes query–key parameters toward max-margin separators, but this does not characterize how such separation filters out spurious features or yields cause only risk guarantees.

**Existing Pain Points:** 
1. Models trained by ERM are vulnerable to superficial shortcuts and spurious correlations, leading to brittle generalization under shift.
2. Existing theoretical analyses on in-context learning rely on designed setups rather than generic pipelines with spurious features, and do not address when spurious information is suppressed.
3. Prior large margin analyses establish that separation can emerge during training, yet they do not identify which side of the margin corresponds to causal versus spurious directions, nor when separation suffices for cause only generalization.
4. Strong causal correlation in the data alone does not guarantee cause only prediction for generic estimators (e.g., population least squares retains a constant fraction of a spurious feature even under a dominance gap).

### 2. Core Research Motivation and Scientific Questions
**Core Research Motivation:** To uncover and analyze a new training phenomenon called Correlation Crowding-Out (CCO), which demonstrates that when a causal feature is strongly correlated with the target, the implicit regularization of gradient descent in Transformers can actively suppress spurious features and drive the model toward a cause-only predictor without explicit invariance penalties or multi-environment training.

**Scientific Questions:** When and through what mechanism can Transformer training produce predictors that rely on causes while ignoring spurious effects?

### 3. Overall Core Idea and Design Philosophy
**Overall Core Idea:** The core idea is the formalization of Correlation Crowding-Out (CCO). CCO posits that if there exists a dominant causal feature whose association with the target exceeds that of any competing spurious feature by a uniform gap, gradient descent drives the Transformer to progressively suppress spurious features and converge to a predictor that relies almost exclusively on the causal feature. 

**Design Philosophy:** The problem is viewed through data correlation strength and the implicit regularization of gradient descent. CCO is formalized as the mirror image of shortcut learning. Intuitively, if a causal feature explains the target with overwhelming strength, the model has little incentive to rely on weaker spurious cues. The mechanism relies on the coupling of two phases: "Occupation" (rapid growth of weights aligned with the dominant causal direction) and "Crowding-out" (attention logits align with separation directions favoring the causal branch, suppressing descendants).

### 4. Core Innovation Points
1. **Introduction and Formalization of CCO:** A new training phenomenon, Correlation Crowding-Out (CCO), is identified and formalized, showing that standard training alone can induce cause only prediction under suitable conditions.
2. **Mechanism Elucidation:** The mechanism of CCO is elucidated through two coupled phases: "Occupation" (early rise of dominant causal weights) and "Crowding-out" (attention selection suppressing spurious features).
3. **Dominant-Coordinate Condition:** A theoretical condition characterizing when CCO arises is introduced, quantifying how strong the causal correlation must be relative to spurious ones.
4. **Theoretical Convergence and Generalization Guarantees:** The paper provides convergence guarantees for the optimization trajectory (Theorem 1) and generalization guarantees for test risk (Theorem 2), showing the predictor essentially relies on causes while filtering out effects.
5. **Distinction from Generic Estimators:** It is proven that CCO is not a trivial corollary of data dominance (as population linear regression retains spurious features), but relies on the implicit regularization induced by Transformers and GD.

### 5. Overview of the Overall Technical Solution
The technical solution involves analyzing a simplified two-layer Transformer module trained on data generated by a causal chain $x \rightarrow y \rightarrow z$. The data generative process assumes a sparse quadratic signal for $y$ in $x$, and a Lipschitz dependent $z$. The model architecture adopts a two-key attention mechanism with fixed positional encodings and a squared-parameter (quadratic) feed-forward head. The training dynamics are analyzed via Gradient Descent. The theoretical analysis divides the training into three stages: Stage 1 (Occupation), Stage 2 (Crowding-out), and Stage 3 (Recovery). Under the Dominant-coordinate condition, it is proven that the squared-parameter head rapidly amplifies the dominant causal coordinate, the attention mechanism aligns with the max-margin separator to squeeze out the descendant branch, and finally, the head recovers the sparse ground truth, leading to cause-only generalization.

### 6. Detailed Module Design
**Data Generative Process Module:** Considers the causal chain $x \rightarrow y \rightarrow z$, where $x, z \in \mathbb{R}^d$ and $y \in \mathbb{R}$. The response $y$ is a sparse quadratic signal in $x$ with noise $\epsilon$, and the descendant $z$ depends on $y$ via a Lipschitz function and additive noise $\xi$.

**Positional Encoding Module:** Augments inputs with fixed positional encodings $s_1, s_2 \in \mathbb{R}^M$:
$$ \tilde{x}_i = \begin{bmatrix} s_1 \\ x_i \end{bmatrix}, \quad \tilde{z}_i = \begin{bmatrix} s_2 \\ z_i \end{bmatrix} \in \mathbb{R}^{M+d} $$

**Two-Key Attention Module:** Parameterizes the query as the gating vector $q_t := \tilde{v}_t \in \mathbb{R}^{M+d}$, keys as $k_x^i := \tilde{x}_i$ and $k_z^i := \tilde{z}_i$, and values as $v_x^i := \tilde{x}_i$ and $v_z^i := \tilde{z}_i$. Logits and weights are defined as:
$$ \ell_{x,i}^t = (q_t)^\top k_x^i, \quad \ell_{z,i}^t = (q_t)^\top k_z^i $$
$$ \alpha_{x,i}^t = \frac{e^{\ell_{x,i}^t}}{e^{\ell_{x,i}^t} + e^{\ell_{z,i}^t}}, \quad \alpha_{z,i}^t = 1 - \alpha_{x,i}^t $$
By softmax translation invariance: $\alpha_{x,i}^t = \sigma((\tilde{v}_t)^\top(\tilde{x}_i - \tilde{z}_i)) =: p_i^t$. The attention output is:
$$ \hat{h}_{i,t} = p_i^t \tilde{x}_i + (1 - p_i^t) \tilde{z}_i $$

**Squared-parameter Head Module:** Predicts with a quadratic parameterization feed-forward layer:
$$ \hat{y}_{i,t} = (\hat{h}_{i,t})^\top (\tilde{w}_t)^{\odot 2} = \sum_{j=1}^{M+d} (\tilde{w}_t^j)^2 \hat{h}_{i,t}^j $$

**Loss Function Module:** Squared loss:
$$ L_n(\tilde{w}, \tilde{v}) = \frac{1}{2n} \sum_{i=1}^n (\hat{y}_i - y_i)^2 $$

**Training Dynamics Modules:**
*   **Stage 1 (Occupation):** Within the Transformer’s feed-forward sublayer, the weight vector that aligns with the dominant causal coordinate in $x$ grows rapidly to a stable magnitude, while weights in other directions remain small.
*   **Stage 2 (Crowding-out):** The Transformer’s attention mechanism gradually shifts its query-key alignment toward the max-margin separator between the transformed causal and spurious features. Attention weights concentrate almost entirely on the causal x branch, gating out the spurious z branch.
*   **Stage 3 (Recovery):** After the descendant z is nearly excluded, the squared-parameter FFN recovers the sparse ground truth weights.

### 7. All Mathematical Formulas and Symbol Definitions
**Data Generative Process:**
$y = x^\top (w^*)^{\odot 2} + \epsilon$, with noise $\epsilon \perp x$, $E[\epsilon] = 0$, $\text{Var}(\epsilon) = \sigma^2$.
$z = f(y) + \xi$, $\xi \perp y$, where $f$ is an L-Lipschitz function.
$H := E[xx^\top] = \begin{bmatrix} a & 0 \\ 0 & I_{d-1} \end{bmatrix}$, $E[x+ z] = \zeta$, $\text{Var}(x+ z) = \Sigma$.
Almost surely $\sup_{1\le j\le d} |x_j| \le B_x$, $\sup_{2\le j\le d} |x_j| \le B'_x$, $|\epsilon| \le B_\epsilon$, $\sup_{1\le j\le d} |\xi_j| \le B_\xi$, $\sup_{1\le j\le d} |z_{ij}| \le \|f(0)\|_\infty + L (rB_x + B_\epsilon) + B_\xi := B_z$.
Ground truth $w^*$ is sparse and binary: $w_j^* \in \{0, 1\}$, $w_1^* = 1$, $|\text{supp}(w^*)| \le r$.

**Cross-Moments and Signals:**
$s_j := E[(x^\top (w^*)^{\odot 2}) (x_j + z_j)]$
$\mu_j := E[\epsilon(x_j + z_j)]$
$s^{eff}_j := s_j + \mu_j$
$m_j := E[(x_j + z_j)^2] = \Sigma_{jj} + \zeta_j^2$
$m_{kj} := E[(x_k + z_k)(x_j + z_j)] = \Sigma_{kj} + \zeta_k \zeta_j$

**Dominant-Coordinate Conditions:**
*Condition 1:* $s^{eff}_1 > \frac{2m_1}{15} + \max_{j>1} \left( \frac{4|s^{eff}_j| + m_{1j}}{8} \right)$
*Condition 2:* There exist constant $\tau_1, \tau_2 > 0$ such that for every sample $i = 1, \ldots, n$:
(i) $|x_{i1} - z_{i1}| \ge \tau_1$
(ii) $\text{sgn}(x_{i1} - z_{i1}) = \text{sgn}(x_{i1})$
(iii) $\frac{3}{4}|x_{i1}| \ge (r - 1)B'_x + B_\epsilon + \tau_2$

**Theorems:**
*Theorem 1 (CCO’s Mechanism):* Under Conditions 1 & 2, with specific initialization and stepsize schedule, the squared-parameter head satisfies $|w^{T^*}_i - w_i^*| \lesssim \sigma \sqrt{\frac{\log d}{n}}$ for $i \in \text{supp}(w^*)$, and $|w^{T^*}_i - w_i^*| \lesssim \frac{1}{d}$ for $i \notin \text{supp}(w^*)$. The gate iterate $\tilde{v}_t = \hat{u} \log t + \rho_t$ ($\hat{u}$ is max-margin solution), yielding $p_i^{T^*} \ge 1 - \frac{1}{d^2}$.

*Theorem 2 (Generalization of CCO):* For an independent test triple $(x, y, z)$, there exists an event $\Omega$ with $Pr(\Omega) \ge 1 - \frac{8\sqrt{\|s\|_2^2 + d(B_x+B_z)^2}}{\|s\|_2\sqrt{n}} - \frac{\sqrt{2\ln(2d^2)}}{n}$, such that conditioned on $\Omega$, $p^{T^*} \ge 1 - \frac{1}{d^2}$ and $E[|L - \frac{\sigma^2}{2}| | \Omega] \lesssim \frac{r \sigma^2 \log d}{n}$.

*Corollary 1 (Robust generalization):* Under shifted $z' = f'(y) + \xi'$, the risk bound $E[|L(x,y,z') - \frac{\sigma^2}{2}| | \Omega] \lesssim \frac{r \sigma^2 \log d}{n}$ holds.

### 8. Algorithm Pseudocode
**Algorithm 1 GD on the two-key attention model**
1: **Input:** $\{(x_i, y_i, z_i)\}_{i=1}^n$, encodings $s_1, s_2$, stepsizes $\{\eta_t\}, \{\beta_t\}$, initialization scale $\alpha$, iterations $T$.
2: **Positional Encoding:** $\tilde{x}_i = \begin{bmatrix} s_1 \\ x_i \end{bmatrix}, \tilde{z}_i = \begin{bmatrix} s_2 \\ z_i \end{bmatrix}$.
3: **Init:** $\tilde{w}_0 = \begin{bmatrix} 0 \\ \alpha I_d \end{bmatrix}, \tilde{v}_0 = 0_{M+d}$.
4: **for** $t = 0, 1, \ldots, T - 1$ **do**
5: **for** $i = 1$ **to** $n$ **do**
6: $p_i^t \leftarrow \sigma((\tilde{v}_t)^\top(\tilde{x}_i - \tilde{z}_i))$, $\hat{y}_{i,t} \leftarrow (p_i^t \tilde{x}_i + (1 - p_i^t)\tilde{z}_i)^\top (\tilde{w}_t)^{\odot 2}$, $r_i^t \leftarrow \hat{y}_{i,t} - y_i$
7: $\tilde{w}_{t+1} \leftarrow \tilde{w}_t - \frac{\eta_t}{n} \sum_{i=1}^n r_i^t (p_i^t \tilde{x}_i + (1 - p_i^t)\tilde{z}_i) \odot \tilde{w}_t$
8: $\tilde{v}_{t+1} \leftarrow \tilde{v}_t - \frac{\beta_t}{n} \sum_{i=1}^n r_i^t p_i^t (1 - p_i^t) (\tilde{x}_i - \tilde{z}_i)^\top (\tilde{w}_t)^{\odot 2} (\tilde{x}_i - \tilde{z}_i)$
9: **Return:** $(\tilde{w}_{t+1}, \tilde{v}_{t+1})$.