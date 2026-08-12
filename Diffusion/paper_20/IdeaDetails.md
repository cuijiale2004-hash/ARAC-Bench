**1. Research Background and Existing Pain Points**

**Research Background**: To model discrete sequences such as DNA, proteins, and language using diffusion, practitioners must choose between three major methods: diffusion in discrete space, Gaussian diffusion in Euclidean space, or diffusion on the simplex. Despite their shared goal of generating high-quality sequences conditioned on desired properties, these models have disparate algorithms, theoretical structures, and tradeoffs: discrete diffusion has the most natural domain, Gaussian diffusion has more mature algorithms, and diffusion on the simplex in principle combines the strengths of the other two but in practice suffers from numerically unstable stochastic processes.

**Existing Pain Points**:
*   **Lack of Unification**: There is little theoretical infrastructure to compare these models, leaving practitioners with little tacit knowledge when selecting or designing a model. Previous theories have only considered connections in special cases.
*   **Incomparable Likelihoods**: Despite models from the three frameworks achieving similar likelihood values, it is believed that the "continuous-space likelihood is not directly comparable with discrete-space likelihood." The Gaussian ELBO is infinity due to a singularity as $t \to 0$, requiring a minimum $t_{min}$.
*   **Incomparable Hyperparameters**: Forward processes are specified by hyperparameters with vastly different interpretations (a rate matrix $L$ vs. an embedding function $emb$). It is unclear how to qualitatively compare the assumptions embedded into each set of hyperparameters across models.
*   **Instability of Simplicial Diffusion**: Simplicial diffusion suffers from numerical instability; sampling noisy $x_t$ involves costly and approximate simulation from a stochastic differential equation (SDE), and loss estimation involves calculations that become very large and cause numerical issues at very small $t$.
*   **Inflexible Deployment**: Practitioners must commit to a specific domain (discrete, Gaussian, or simplicial) before training, restricting their access to downstream algorithms and forcing a choice between tradeoffs.

**2. Core Research Motivation and Scientific Questions**

**Core Research Motivation**: Ideally, we could see each of the three major discrete diffusion models as instances of the same underlying framework, and enable practitioners to switch between models for downstream applications. The motivation is to build a theory unifying all three methods of discrete diffusion to explain their differences, solve their instabilities, and relieve practitioners of balancing model trade-offs by enabling a single model to operate across domains.

**Scientific Questions**:
*   How can discrete, Gaussian, and simplicial diffusion be formally unified as different parameterizations of the same underlying process?
*   How can the likelihoods and hyperparameters of these models be formally connected and compared?
*   How can the numerical instability of simplicial diffusion be solved using the unification theory?
*   Is it possible to train a single model that can perform diffusion in any of the three domains at test time?

**3. Overall Core Idea and Design Philosophy**

**Overall Core Idea**: The paper unifies discrete, Gaussian, and simplicial diffusion as different parameterizations and limits of the Wright-Fisher (WF) population genetics model. The core idea is to represent each dimension of a sequence with $\zeta$ copies to get a sequence of sequences, where each letter in each sequence evolves by a mutation matrix $L$.

**Design Philosophy**:
*   **Unification via Population Size**: When the population size $\zeta=1$, the process corresponds to discrete diffusion. As $\zeta \to \infty$ with zero reproduction rates, the trajectories converge quickly to the stationary distribution $\pi$ and behave like Gaussians near $\pi$ (Gaussian diffusion). As $\zeta \to \infty$ with non-zero reproduction rates, the limit corresponds to simplicial diffusion (Wright-Fisher diffusion).
*   **Leveraging Genetics Literature**: The unification allows leveraging decades of mathematical genetics literature (specifically Wright-Fisher diffusion) to unlock stable simplicial diffusion.
*   **Sufficient Statistics for Universality**: By using the sufficient-statistic parameterization (SSP), the diffusion model becomes independent of the specific diffusion process or time $t$, allowing a single neural network to learn the data distribution across all three modalities.

**4. Core Innovation Points**

1.  **Formal Unification of Three Diffusion Methods**: The paper proves that all three methods are instances of the Wright-Fisher model. Discrete diffusion corresponds to the WF model with a population size of 1, and simplicial and Gaussian diffusion correspond to large population limits with and without reproduction, respectively.
2.  **Likelihood and Hyperparameter Comparison**: The theory formally connects the likelihoods and hyperparameters of these models. It shows that the "hollow parameterization" is crucial for removing the singularity in the Gaussian ELBO and making likelihoods comparable. It also derives that the embedding `emb` in Gaussian diffusion is determined by the slowest-decaying directions (dominant eigenspace) of the mutation rate matrix $L$.
3.  **Stable Simplicial Diffusion**: The paper applies theory from mathematical genetics to solve the instability of simplicial diffusion. It replaces costly SDE simulation with exact sampling using Dirichlet marginals and an ancestral process algorithm. It also stabilizes the loss computation using Griffiths' approximation for small $t$.
4.  **Unified Diffusion Models via SSP**: The paper introduces the Sufficient Statistic Parameterization (SSP), allowing one to train a single neural network that can perform diffusion on all three domains (discrete, Gaussian, simplicial) at test time, removing the necessity for the practitioner to choose a particular model before training.
5.  **Time-Invariance Explanation**: The SSP explains and extends the noted "time-invariance" of masking diffusion to every diffusion model.

**5. Overview of the Overall Technical Solution**

The technical solution starts by defining a $\zeta$-population discrete diffusion model. Each dimension of a sequence is represented by $\zeta$ copies. The mutation process is defined by a rate matrix $L$, and the reproduction process (for simplicial diffusion) adds a generation swap at rate $\zeta$. 

For the **Gaussian Limit**, the paper proves that as $\zeta \to \infty$ with only mutation, the trajectories converge to the stationary distribution $\pi$ and behave like Gaussians near $\pi$ due to the central limit theorem. By zooming into the neighborhood of $\pi$ and projecting onto the first eigenspace of $L$, the process converges to Gaussian diffusion in Euclidean space.

For the **Simplicial Limit**, as $\zeta \to \infty$ with reproduction, the limit is the continuous Wright-Fisher diffusion process. To address instability, the solution leverages "Exact simulation of the Wright-Fisher diffusion" by sampling $x_t$ from a Dirichlet distribution conditional on the number of ancestors $m$ (sampled via an alternating series algorithm or a low-$t$ Gaussian approximation). The loss is stabilized by replacing the neural network derivative with a score-matching style loss derived from the ELBO limit, using specific scaling and metrics, and bounding the loss at low $t$.

For the **Unified Model**, the solution defines the Sufficient Statistic Parameterization (SSP) where the network input is the evidence $\phi(x_{t,d}, t)$ rather than raw $x_t$. This $\phi$ contains all relevant information about the diffusion process and $t$, leaving a regression task that is invariant to the diffusion process and time, allowing a single network $F^d_\theta$ to be trained across modalities by alternating the ELBO minimization.

**6. Detailed Module Design**

**6.1 Unification of Discrete and Gaussian Diffusion Module**
*   **Representation**: Represent a sequence dimension with $\zeta$ copies. $x_0$ is a sequence of sequences (e.g., $AAAA|CCCC$). Each letter evolves by mutation matrix $L$.
*   **Gaussian Limit Mechanism**: As $\zeta \to \infty$, by the law of large numbers, $x_t$ approaches the stationary distribution $\pi$. The difference $x_t - \pi$ is decomposed into "signal" ($x_0^T e^{\tau_t^\zeta L} - \pi$) and "noise" ($x_t - x_0^T e^{\tau_t^\zeta L}$). The noise is approximately Gaussian with scale $\zeta^{-1/2}$. The signal is dominated by the slowest-decaying eigenspace $P_1$ of $L$.
*   **Hollow Parameterization**: To fix the singularity in the Gaussian ELBO limit, the network output is weighted by the evidence: $q_\theta(x_0 | x_t, t) \propto p(x_t | x_0, t) q_\theta(x_0 | x_t^{-d}, t)$. This removes the divergence because $p(x_t | x_0, t)$ "automatically handles" deciding when $x_0$ is obvious.

**6.2 Unification of Simplicial Diffusion Module**
*   **Wright-Fisher Model**: Add reproduction to the population. The population is swapped with a new generation at rate $\zeta$. At each generation, $\zeta$ "children" pick a parent uniformly at random. Between generations, individuals mutate according to $L$.
*   **Limit**: As $\zeta \to \infty$, the stochastic process converges to Wright-Fisher diffusion on the simplex. Unlike the mutation-only case, paths travel throughout the simplex.
*   **ELBO Limit**: The discrete diffusion ELBO converges to a score-matching style objective involving the difference between score functions $s(x_t | x_0, t) - s(x_t | \tilde{x}_0, t)$.

**6.3 Stable Simplicial Diffusion Sampling Module**
*   **Exact Sampling**: Instead of simulating SDEs, sample $x_t$ exactly from a Dirichlet distribution: $x_t \sim \text{Dirichlet}(\psi\pi + m x_0)$, where $m$ is the number of ancestors sampled from a distribution $A(\psi, \tau_t)$.
*   **Low-t Approximation**: For small $\tau_t$, the infinite series for $A(\psi, \tau_t)$ becomes unstable. Replace with Griffiths' Gaussian approximation: $m = \max(0, \lfloor Z + 0.5 \rfloor)$ where $Z \sim \mathcal{N}(\mu, \sigma^2)$.

**6.4 Stable Loss Computation Module**
*   **Score Calculation**: Compute $\vec{s}(v | x_0, t) = \nabla \log p(x_t | x_0, t)$ using infinite series $G_\psi$ and $F_\psi$ involving hypergeometric functions.
*   **Low-t Loss Bound**: For $\tau_t < 0.05$, bounding the loss using a saddle-point approximation of the ancestral process expectation, avoiding large derivatives.

**6.5 Sufficient Statistic Parameterization (SSP) Module**
*   **Definition**: $\phi(x_{t,d}, t)_b \propto p(x_{t,d} | t, x_{0,d}=b)$.
*   **Mechanism**: The network predicts $q_\theta(x_0^d | x_t^{-d}, t) = F^d_\theta(\phi(x_{t,1}, t), \dots, \phi(x_{t,D}, t))$. Since $\phi$ contains all relevant information about the diffusion process and $t$, $F^d$ is invariant to the choice of diffusion domain and time. Training alternates between discrete, Gaussian, and simplicial ELBO minimization.

**7. All Mathematical Formulas and Symbol Definitions**

*   $p(x_0)$: Distribution over a discrete space of size $B$.
*   $q(x_1)$: Easy-to-sample starting distribution.
*   $\tau_t$: Time dilation function $\tau: [0,1] \to [0, \infty)$.
*   $\dot{\tau}_t$: Derivative of $\tau_t$.
*   $D(\lambda_1 || \lambda_2)$: KL divergence between two Poisson distributions $\lambda_1 \log \frac{\lambda_1}{\lambda_2} - \lambda_1 + \lambda_2$.
*   $L$: Mutation rate matrix for discrete diffusion.
*   $emb(x_0)$: Embedding function for Gaussian diffusion into $\mathbb{R}^r$.
*   $\zeta$: Population size / number of copies.
*   $x_t$: Noisy state.
*   $\pi$: Stationary distribution of $L$.
*   $-\lambda_1 > -\lambda_2 > \dots$: Eigenvalues of $L$.
*   $P_1$: Projection onto the left eigenspace corresponding to $\lambda_1$.
*   $\tau_t^\zeta = \frac{1}{2}\log(\zeta e^{-2\tau_t} - \zeta + 1)$: Time dilation for $\zeta$-discrete diffusion.
*   $\mathbf{x}_t^\zeta = \sqrt{\zeta - (\zeta - 1)e^{-2\tau_t}} (\mathbf{x}_t - \pi) / \sqrt{\pi}$: Rescaled variable for $\zeta$-discrete diffusion.
*   $Q_1 = j_1(\tilde{Q}_1 \tilde{Q}_1^T)^{-1/2} \tilde{Q}_1$: Embedding into $\mathbb{R}^{\text{rank}(P_1)}$.
*   $\tilde{Q}_1 = \text{diag}(\pi)^{-1/2} P_1 \text{diag}(\pi)^{1/2}$: Symmetrized projection.
*   **Theorem 4.1 (Gaussian Limit)**: When $\zeta \to \infty$, paths $Q_1 \mathbf{x}_t^\zeta$ converge in distribution to paths from Gaussian diffusion with time dilation $\tau_t$ and embedding $emb(x_0) = Q_1(\mathbf{x}_0 / \sqrt{\pi})$.
*   **Wright-Fisher SDE**: $d\mathbf{z}_t = \frac{\psi}{2}(\pi - \mathbf{z}_t)dt + (\text{diag}(\sqrt{\mathbf{z}_t}) - \sqrt{\mathbf{z}_t}\sqrt{\mathbf{z}_t}^T) d\mathbf{W}_t$.
*   $A(\psi, \tau_t)$: Distribution for number of ancestors.
*   **Score function**: $\vec{s}(\mathbf{v} | x_0, t) = \nabla \log p(\mathbf{x}_t | x_0, t)|_{\mathbf{x}_t = \mathbf{v}}$.
*   $\vec{s}(\mathbf{v} | x_0, t) = \vec{c}(\mathbf{v}) + \mathbf{x}_0 w(x_0, \mathbf{v})$ (Eqn 2), where $\vec{c}(\mathbf{v}) = (\psi\pi - 1)/\mathbf{v}$.
*   $w(x_0, \mathbf{v}) = \frac{e^{-\psi\tau_t/2}(\psi+1)}{\pi(x_0)} \frac{F_\psi(\tau_t, x_0, \mathbf{v})}{G_\psi(\tau_t, x_0, \mathbf{v})}$.
*   $G_\psi(\tau, x_0, \mathbf{x}_t)$ and $F_\psi(\tau, x_0, \mathbf{x}_t)$: Infinite series defined with coefficients $a_k$ and $b_k$ involving hypergeometric functions $\mathstrut_2 F_1$.
*   **Low-t Loss Bound**: $L \leq 2 \dot{\tau}_t v_{x_0}^{-1} (\tilde{E}_{1,p}[m_t])^2$ (Eqn 3).
*   **Saddle point approx**: $\tilde{E}_{1,p}[m_t] \approx \frac{- (\mu - (\psi-1)) + \sqrt{(\mu + (\psi-1))^2 + 4(1-p)\psi\sigma^2}}{2}$ (Eqn 4).
*   **Sufficient Statistic**: $\phi(\mathbf{x}_{t,d}, t)_b \propto p(\mathbf{x}_{t,d} | t, x_{0,d} = b)$.
*   $p(x_0^d | x_t^{-d}, t) = F^d(\phi(\mathbf{x}_{t,1}, t), \dots, \phi(\mathbf{x}_{t,D}, t))$.

**8. Algorithm Pseudocode**

**Algorithm 1 ELBO for discrete diffusion**
1: Sample $t \sim \text{Unif}(0, 1)$
2: Sample noisy $x_t$:
3: Sample $x_t \sim \text{Categorical}(\mathbf{x}_0^T e^{\tau_t L})$
4: Predict de-noised $x_0$:
5: Predict $\tilde{x}_0 = q_\theta(x_0 | x_t, t)$
6: Estimate ELBO:
7: $\hat{w}(b) = (\mathbf{b}^T e^{\tau_t L}) \circ (1 / \mathbf{b}^T e^{\tau_t L})^T$
8: $L = \sum_{b \neq x_t} L_{b \to x_t} \dot{\tau}_t D(\hat{w}(x_0) \circ x_t || \hat{w}(\tilde{x}_0) \circ x_t)$

**Algorithm 2 ELBO for Gaussian diffusion**
1: Sample $t \sim \text{Unif}(0, 1)$
2: Sample noisy $x_t$:
3: Set $x_t = e^{-\tau_t} emb(x_0) + \sqrt{1 - e^{-2\tau_t}} \mathcal{N}(0, I)$
4: Predict de-noised $x_0$:
5: Predict $\tilde{x}_0 = q_\theta(x_0 | x_t, t)$
6: Estimate ELBO:
7: $L = \dot{\tau}_t \frac{e^{-2\tau_t}}{(1 - e^{-2\tau_t})^2} \|emb(x_0) - emb(\tilde{x}_0)\|^2$

**Algorithm 3 ELBO for $\zeta$ discrete diffusion**
1: Sample $t \sim \text{Unif}(0, 1)$
2: Sample noisy $x_t$:
3: Sample $\mathbf{x}_t \sim \text{Multinomial}(\zeta, \mathbf{x}_0^T e^{\tau_t L}) / \zeta$
4: Predict de-noised $x_0$:
5: Predict $\tilde{x}_0 = q_\theta(x_0 | \mathbf{x}_t, t)$
6: Estimate ELBO:
7: $\hat{w}(b) = (\mathbf{b}^T e^{\tau_t L}) \circ (1 / \mathbf{b}^T e^{\tau_t L})^T$
8: $L = \sum_{b \neq b'} L_{b \to b'} \dot{\tau}_t \zeta \mathbf{x}_{t, b'} D(\hat{w}(x_0)_{b, b'} || \hat{w}(\tilde{x}_0)_{b, b'})$

**Algorithm 4 ELBO for simplicial diffusion**
1: Sample $t \sim \text{Unif}(0, 1)$
2: Sample noisy $x_t$:
3: Sample $m \sim A(\psi, \tau_t)$ with Alg. 5; if $\tau_t < 0.05$, use Alg. 6
4: Sample $\mathbf{x}_t \sim \text{Dirichlet}(\psi\pi + m\mathbf{x}_0)$.
5: Predict de-noised $x_0$:
6: Predict $\tilde{x}_0 = q_\theta(x_0 | x_t, t)$
7: Estimate ELBO:
8: Compute $\vec{s}(\mathbf{x}_t | x_0, t) = \nabla_{x_t} \log p(x_t | x_0, t)$ with Eqn 2
9: $L = \frac{\dot{\tau}_t}{2} \|\vec{s}(\mathbf{x}_t | x_0, t) - \vec{s}(\mathbf{x}_t | \tilde{x}_0, t)\|^2_{\text{diag}(\mathbf{x}_t) - \mathbf{x}_t \mathbf{x}_t^T}$ (this is an ELBO); if $\tau_t < 0.05$, use Eqn 3

**Algorithm 5 Exact sampling from ancestral process $A(\psi, \tau_t)$**
1: Define coefficients: $c^{\psi}_{km} = \frac{(2k+\psi-1)(\psi)_{(k-1)}}{m!(k-m)!} e^{-k(k+\psi-1)\tau_t / 2}$ for $k \geq m$
2: Sample $U \sim \text{Uniform}[0, 1]$
3: Initialize $M \leftarrow 0$
4: Initialize an empty vector $\mathbf{k} = ()$
5: while True do
6:  Find $k_M \geq M$ such that $c^{\psi}_{(k_M+1)M} < c^{\psi}_{k_M M}$
7:  Make $k_M$ even: $k_M \leftarrow 2\lfloor k_M / 2 \rfloor$
8:  Update lower bound: $S_- \leftarrow S_- + \sum_{k=M}^{k_M+1} (-1)^{k-M} c^{\psi}_{kM}$
9:  Update upper bound: $S_+ \leftarrow S_+ + \sum_{k=M}^{k_M} (-1)^{k-M} c^{\psi}_{kM}$
10: Update $\mathbf{k} = (k_0, \dots, k_{M-1}, k_M)$
11: while $S_- < U < S_+$ do
12:  Update lower bound: $S_- \leftarrow S_- + \sum_{m=0}^{M} (c^{\psi}_{(k_m+2)m} - c^{\psi}_{(k_m+3)m})$
13:  Update upper bound: $S_+ \leftarrow S_- + \sum_{m=0}^{M} (-c^{\psi}_{(k_m+1)m} + c^{\psi}_{(k_m+2)m})$
14:  Update $\mathbf{k} = \mathbf{k} + (2, \dots, 2)$
15: end while
16: if $S_- > U$ then
17:  return $m = M$
18: else if $S_+ < U$ then
19:  $M \leftarrow M + 1$
20: end if
21: end while

**Algorithm 6 Sampling from ancestral process $A(\psi, \tau_t)$ - Low t approximation**
1: Set $\beta \leftarrow \frac{1}{2} (\psi - 1) \tau_t$
2: if $\beta \neq 0$ then
3:  Set $\eta \leftarrow \beta / (e^\beta - 1)$
4:  Set $\mu \leftarrow \frac{2\eta}{\tau_t}$
5:  Set $\sigma^2 \leftarrow \frac{2\eta}{\tau_t} \left( \frac{(\eta + \beta)^2 - 1 + \frac{\eta}{\eta+\beta} - 2\eta}{\beta-2} \right)$
6: else
7:  Set $\mu \leftarrow \frac{2}{\tau_t}$
8:  Set $\sigma^2 \leftarrow \frac{2}{3\tau_t}$
9: end if
10: Sample $Z \sim \mathcal{N}(\mu, \sigma^2)$
11: return $m = \max(0, \lfloor Z + 0.5 \rfloor)$