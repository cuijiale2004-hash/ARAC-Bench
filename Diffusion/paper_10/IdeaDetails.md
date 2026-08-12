### 1. Research Background and Existing Pain Points

**Research Background:**
Multi-objective optimization (MOO) is fundamental in numerous scientific and engineering disciplines, where decision-makers often face the challenge of optimizing conflicting objectives simultaneously. The primary aim is to identify the Pareto front: a set of non-dominated solutions where improving one objective would deteriorate at least one other.

**Existing Pain Points:**
1.  **Scalability Limitations:** Traditional methods for approximating the Pareto front, including evolutionary algorithms, scalarization techniques, and multiple-gradient descent (MGD) combined with multi-start techniques, may struggle with scalability, especially in high-dimensional or resource-constrained settings.
2.  **Lack of Broader Applicability:** Domain-specific MOO algorithms (e.g., offline MOO, Bayesian MOO, federated learning) have been developed to leverage domain knowledge for improved efficiency in targeted settings, but this comes at the expense of broader applicability.
3.  **Diversity vs. Convergence Trade-off:** Employing a classical multi-start approach with MGD does not inherently promote diversity among solutions. Existing generative modeling approaches for MOO (like ParetoFlow or PGD-MOO) either rely on separate preference classifiers or lack explicit diversity-promoting strategies integrated within the generative process, which can lead to mode collapse or insufficient coverage of the Pareto front.

### 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:**
There is a critical need for MOO methods capable of efficiently adapting to large-scale, high-dimensional, and computationally expensive problem settings. Developing such universal approaches would streamline the optimization process and broaden the applicability of MOO techniques across different domains.

**Scientific Questions:**
How can we design a generative framework based on diffusion models that not only converges to Pareto optimal solutions but also ensures comprehensive and well-distributed coverage of the Pareto front? How can gradient-based guidance be effectively integrated into a conditional diffusion process to guarantee objective improvement while mitigating mode collapse through diversity mechanisms?

### 3. Overall Core Idea and Design Philosophy

**Overall Core Idea:**
The paper introduces SPREAD (Sampling-based Pareto front Refinement via Efficient Adaptive Diffusion), a novel diffusion-driven generative framework designed to tackle multi-objective optimization. SPREAD leverages the strengths of diffusion models to iteratively generate and refine candidate solutions, guiding them towards Pareto optimality.

**Design Philosophy:**
SPREAD applies a conditional diffusion modeling approach, where a conditional DDPM is trained on points sampled from the input space, allowing the model to effectively learn the underlying structure of the objective functions and steer the generation process toward promising regions. To further enhance convergence towards Pareto optimality, SPREAD incorporates an adaptive guidance mechanism inspired by the multiple gradient descent algorithm, dynamically guiding the sampling process to regions likely to contain optimal solutions. Furthermore, to promote diversity among the generated solutions and ensure an approximation of the entire Pareto front, SPREAD utilizes a Gaussian RBF repulsion mechanism that discourages clustering, mitigates mode collapse, and encourages exploration in the objective space.

### 4. Core Innovation Points

1.  **Novel Diffusion-Based Generative Framework:** A novel diffusion-based generative framework for MOO that effectively approximates the Pareto front.
2.  **Novel Conditioning Approach:** A novel conditioning approach along with an adaptive guidance mechanism inspired by MGD to improve convergence towards Pareto optimal solutions. Specifically, the training labels are shifted by a strictly positive vector $\Xi$, which provides a theoretical guarantee (Theorem 1) that conditioned sampling yields points that dominate the initialization.
3.  **Diversity-Promoting Strategy:** Implementation of a diversity-promoting strategy (Gaussian RBF repulsion) to ensure a comprehensive and well-distributed set of solutions, integrated directly into the guidance update alongside the MGD direction.
4.  **Validation on Challenging MOO Tasks:** Validation on challenging MOO tasks, including offline and Bayesian settings, demonstrating the method's effectiveness and generalizability across different problem domains.
5.  **Theoretical Guarantees:** Theoretical guarantees establishing that the conditioned sampling produces dominant solutions (Theorem 1) and that the combined guidance direction serves as a common descent direction for all objectives under specific scaling (Theorem 2).

### 5. Overview of the Overall Technical Solution

SPREAD operates in two main phases: Training and Sampling.

**Training Phase:**
The framework adopts a Transformer-based noise-prediction network, DiT-MOO (Diffusion Transformer adapted for MOO). It is trained on a dataset $X$ sampled from the decision space via Latin hypercube sampling. The model learns to predict the noise $\hat{\epsilon}_\theta$ conditioned on a shifted objective vector $C = F(X) + \Xi$, where $\Xi \in (0, \infty)^m$.

**Sampling Phase:**
Starting from random noise $X_T$, the sampling process iteratively denoises the points over $T$ steps. At each reverse diffusion step $t$, the standard denoising update is augmented with a guided update. This guidance combines two components:
1.  **MGD-inspired Alignment:** A main direction $h_t^i$ derived by solving an optimization sub-problem that maximizes alignment with the Multiple Gradient Descent (MGD) direction $g_t^i$ while minimizing a Gaussian RBF repulsion function in the objective space to promote diversity.
2.  **Random Perturbation:** A random perturbation $\delta_t$ scaled by an adaptive parameter $\gamma_t^i$ determined based on Theorem 2 to ensure the combined direction is a common descent direction for all objectives.

The final set of approximate solutions is obtained by selecting the top-$n$ non-dominated points from the union of all timesteps using crowding distance.

### 6. Detailed Module Design

**6.1. Data Sampling and Training (Algorithm 2)**
*   **Sampling:** $N$ points $\{x_i\}_{i=1}^N = X \subset \mathcal{X}$ are sampled using Latin hypercube sampling.
*   **Forward Diffusion:** Noising the data $X_t = \sqrt{\bar{\alpha}_t}X + \sqrt{1-\bar{\alpha}_t}\epsilon$, where $\epsilon \sim \mathcal{N}(0, I)$.
*   **Conditioning:** The condition is defined as $C = F(X_t) + \Xi$, where $\Xi$ is a vector with strictly positive entries. $\Xi = \omega \cdot \mathbf{1}_m$.
*   **Loss Function:** The model is trained to minimize the loss $L_s(\theta) = \mathbb{E}_{x_0, \epsilon, t, c} [\|\epsilon - \hat{\epsilon}_\theta(x_t, t, c)\|^2]$.

**6.2. DiT-MOO Architecture**
The noise-prediction network is based on the Diffusion Transformer (DiT).
*   **Input Projection:** Noisy points $X_t \in \mathbb{R}^{n \times d}$ are projected to $(n, 1, e)$.
*   **Embeddings:** Timestep $t$ and Condition $C$ are embedded into $(n, 1, e)$.
*   **DiT Block:** Contains Layer Norm, Multi-Head Self-Attention (MHSA), and Multi-Head Cross-Attention (MHCA). In MHCA, the Query is from $Z_t$ (layer-normalized features from $X_t$) and Key/Value are from $B_t$ (concatenation of condition and time embeddings).
    *   $Q^{(i)} = Z_t W_Q^{(i)}$, $K^{(i)} = B_t W_K^{(i)}$, $V^{(i)} = B_t W_V^{(i)}$.
    *   Attention weights: $A^{(i)} = \text{softmax}(Q^{(i)}(K^{(i)})^\top / \sqrt{d_k})$.
    *   Head output: $O^{(i)} = A^{(i)}V^{(i)}$.
    *   Concatenated and projected: $\text{MHCA}(Z_t, B_t) = (\text{concat}_{i=1}^h O^{(i)}) W_O$.
*   **Output Projection:** Back to $(n, d)$ as predicted noise.

**6.3. MGD Direction Computation**
For each point $x$, the MGD direction $g(x)$ is computed to find a common descent direction for all objectives:
1.  Solve for optimal weights: $\lambda^* = \arg \min_{\lambda \in \Delta_m} \|\sum_{j=1}^m \lambda_j \nabla f_j(x)\|^2$, where $\Delta_m$ is the standard simplex.
2.  Aggregated gradient: $g(x) = \sum_{j=1}^m \lambda_j^* \nabla f_j(x)$.

**6.4. Main Direction Computation ($h_t$)**
The main directions $(h_t^i)_{i=1}^n$ are obtained by balancing alignment with MGD directions and minimizing diversity repulsion. They are the solution to the following sub-problem:
$$(h_t^i)_{i=1}^n = \arg \min_{(u_i)_{i=1}^n} \left\{ - \frac{1}{n} \sum_{i=1}^n \langle g_t^i, u_i \rangle + \nu_t \Gamma_t(F(X'_t - \eta_t((u_i)_{i=1}^n + \gamma_t^T \delta_t))) \right\}$$
This is solved by performing a fixed number of gradient descent steps.

**6.5. Gaussian RBF Repulsion Function**
To promote diversity, a repulsion function $\Gamma_t(Y_t)$ is used based on Gaussian RBF:
$$\Gamma_t(Y_t) = \frac{2}{n(n-1)} \sum_{1 \le i < j \le n} \exp\left(-\frac{\|y_t^i - y_t^j\|^2}{2\sigma^2}\right)$$

**6.6. Adaptive Perturbation Scaling ($\gamma_t$)**
A random perturbation $\delta_t$ is added to the main direction. The scaling parameter $\gamma_t^i$ is computed to ensure descent:
$$\gamma_t^i = \begin{cases} \rho \min_{j: b_{i,j} < 0} \left( -\frac{a_{i,j}}{b_{i,j}} \right), & 0 < \rho < 1, \text{ if any } b_{i,j} < 0 \\ \zeta, & \zeta > 0, \text{ otherwise} \end{cases}$$
where $a_{i,j} = \langle \nabla f_j(x_t'^i), h_t^i \rangle$ and $b_{i,j} = \langle \nabla f_j(x_t'^i), \delta_t \rangle$.

**6.7. Sampling Update Mechanism (Algorithm 1)**
At each sampling step $t$:
1.  **Denoising Step:**
    $$X'_t \leftarrow \frac{1}{\sqrt{1-\beta_t}} \left( X_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \hat{\epsilon}_\theta(X_t, t, C) \right) + \sqrt{\beta_t} z$$
2.  **Guidance Update:**
    $$X_{t-1} \leftarrow X'_t - \eta_t \tilde{h}_t(X'_t)$$
    where $\tilde{h}_t(X'_t) = (h_t^i)_{i=1}^n + \gamma_t^T \delta_t$.
3.  **Step Size ($\eta_t$):** Determined using Armijo backtracking line search to ensure sufficient decrease.
4.  **Selection:** $P_{t-1}$ is the top-$n$ non-dominated points from $X_{t-1} \cup P_t$ using crowding distance.

### 7. All Mathematical Formulas and Symbol Definitions

*   **Forward Diffusion Process:**
    $$q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$$
*   **Training Loss:**
    $$L_s(\theta) = \mathbb{E}_{x_0, \epsilon, t, c} [\|\epsilon - \hat{\epsilon}_\theta(x_t, t, c)\|^2]$$
*   **Noisy Data:**
    $$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, \quad \bar{\alpha}_t = \prod_{i=1}^t (1-\beta_i)$$
*   **Reverse Diffusion Process (Standard):**
    $$x_{t-1} = \frac{1}{\sqrt{1-\beta_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \hat{\epsilon}_\theta(x_t, t, c) \right) + \sqrt{\beta_t} z$$
*   **Multi-Objective Optimization Problem:**
    $$\min_{x \in \mathcal{X}} F(x) = (f_1(x), \dots, f_m(x))$$
*   **MGD Optimization Problem:**
    $$\lambda^* = \arg \min_{\lambda \in \Delta_m} \left\| \sum_{j=1}^m \lambda_j \nabla f_j(x) \right\|^2$$
*   **Aggregated Gradient:**
    $$g(x) = \sum_{j=1}^m \lambda_j^* \nabla f_j(x)$$
*   **Conditioning Vector (Training):**
    $$C = F(X) + \Xi, \quad \Xi \in (0, \infty)^m$$
*   **Total Variation Distance:**
    $$TV(Q_\theta(\cdot|c), P_{X|C=c}) = \sup_A |Q_\theta(A|c) - P_{X|C=c}(A)| \le \tau$$
*   **Dominance Probability (Theorem 1):**
    $$P(x_0 \prec x_T) \ge 1 - \tau$$
*   **Sampling Update Equations:**
    $$X'_t \leftarrow \frac{1}{\sqrt{1-\beta_t}} \left( X_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \hat{\epsilon}_\theta(X_t, t, C) \right) + \sqrt{\beta_t} z$$
    $$X_{t-1} \leftarrow X'_t - \eta_t \tilde{h}_t(X'_t)$$
*   **Guidance Directions:**
    $$\tilde{h}_t(X'_t) = (\tilde{h}_t^i)_{i=1}^n = (h_t^i)_{i=1}^n + \gamma_t^T \delta_t$$
*   **Alignment Objective:**
    $$\frac{1}{n} \sum_{i=1}^n \langle g_t^i, h_t^i \rangle$$
*   **Gaussian RBF Repulsion Function:**
    $$\Gamma_t(Y_t) = \frac{2}{n(n-1)} \sum_{1 \le i < j \le n} \exp\left(-\frac{\|y_t^i - y_t^j\|^2}{2\sigma^2}\right)$$
*   **Sub-problem for Main Directions:**
    $$(h_t^i)_{i=1}^n = \arg \min_{(u_i)_{i=1}^n} \left\{ - \frac{1}{n} \sum_{i=1}^n \langle g_t^i, u_i \rangle + \nu_t \Gamma_t(F(X'_t - \eta_t((u_i)_{i=1}^n + \gamma_t^T \delta_t))) \right\}$$
*   **Perturbation Scaling (Theorem 2):**
    $$\gamma_t^i = \begin{cases} \rho \min_{j: b_{i,j} < 0} \left( -\frac{a_{i,j}}{b_{i,j}} \right), & 0 < \rho < 1, \text{ if any } b_{i,j} < 0 \\ \zeta, & \zeta > 0, \text{ otherwise} \end{cases}$$
    where $a_{i,j} = \langle \nabla f_j(x_t'^i), h_t^i \rangle$ and $b_{i,j} = \langle \nabla f_j(x_t'^i), \delta_t \rangle$.
*   **Crowding Distance:**
    $$CD(x_i) = \sum_{j=1}^m d_i^j, \quad d_i^j = \frac{f_j(x_{i+1}) - f_j(x_{i-1})}{\max_k f_j(x_k) - \min_k f_j(x_k)}$$

### 8. Algorithm Pseudocode

**Algorithm 1: SPREAD (Online Setting)**
Input: DiT-MOO architecture (untrained model), a multi-objective optimization problem (MOP).
Parameter: epochs E, timesteps T, sample size n.
Output: approximate pareto optimal points $P_0$.
1: DiT-MOO training via Algorithm 2.
2: Initialize n random points $X_T = \{x_T^i\}_{i=1}^n \subset \mathcal{X}$
3: $P_T \leftarrow X_T$
4: for $t = T$ to 1 do
5:  $(g_t^i)_{i=1}^n \leftarrow$ Get the MGD directions via Section 3.3.
6:  $(h_t^i)_{i=1}^n \leftarrow$ Get the main directions via equation 13.
7:  $(\tilde{h}_t^i)_{i=1}^n \leftarrow$ Get the guidance directions via equation 10.
8:  $X_{t-1} \leftarrow$ Get the denoised points via equation 9.
9:  $P_{t-1} \leftarrow$ Use crowding distance (Appendix A.5) to get the top-n non-dominated points from $X_{t-1} \cup P_t$.
10: end for
Return: $P_0$

**Algorithm 2: Training (Online Setting)**
Input: DiT-MOO as the noise prediction network $\hat{\epsilon}_\theta(\cdot)$, a multi-objective optimization problem (MOP).
Parameter: epochs E, timesteps T.
Output: a trained noise prediction network $\hat{\epsilon}_\theta(\cdot)$.
1: Sample N points $\{x_i\}_{i=1}^N = X \subset \mathcal{X}$ using Latin hypercube sampling (Appendix A.5).
2: $\{\beta_t\}_{t=1}^T \leftarrow$ Get the variances via a cosine schedule (Appendix A.5).
3: for epoch = 1 to E do
4:  $t \leftarrow$ Uniform$(\{1, \dots, T\})$
5:  $X_t \leftarrow \sqrt{\bar{\alpha}_t}X + \sqrt{1-\bar{\alpha}_t}\epsilon$, with $\epsilon \sim \mathcal{N}(0, I)$, and $\bar{\alpha}_t \leftarrow \prod_{i=1}^t(1-\beta_i)$.
6:  $C \leftarrow F(X_t) + \Xi$, with $\Xi \in (0, \infty)^m$ an arbitrary vector with strictly positive entries.
7:  Take gradient descent step on $\nabla_\theta \|\epsilon - \hat{\epsilon}_\theta(X_t, t, C)\|^2$.
8: end for
Return: $\hat{\epsilon}_\theta(\cdot)$