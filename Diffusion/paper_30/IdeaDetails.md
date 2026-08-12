**1. Research Background and Existing Pain Points**
Diffusion- and flow-based models have revolutionized generative modeling but rely on resource-intensive training, making it impractical to retrain from scratch for every new setting. This renders lightweight adaptation of pre-trained models to target distributions at inference time both compelling and essential. Existing methods face significant pain points:
- **Guidance-based techniques** (e.g., classifier, classifier-free guidance) inject new information into the drift term. While simple and effective for many tasks, these methods are often heuristic and introduce uncontrolled bias, a significant drawback for scientific applications where sampling accuracy is paramount.
- **Extra-training methods** (e.g., fine-tuning, learning within stochastic control frameworks) shift the computational burden back to retraining, forfeiting the efficiency of inference-time approaches.
- **Particle-based corrections** (Sequential Monte Carlo - SMC) correct for the bias of guidance by simulating the target dynamics with weighted particles. Despite strong theoretical grounding, these methods face a critical practical bottleneck: weight degeneracy. As simulation progresses, the weights of a few particles grow exponentially while the rest decay, causing the effective sample size (ESS) to collapse. Mitigating this requires increasing the number of particles (raising computational cost) or using fewer particles (resulting in instability and degraded sample quality).

**2. Core Research Motivation and Scientific Questions**
The core research motivation is to resolve the inherent instability (weight degeneracy) of particle-based methods without sacrificing mathematical rigor or inference-time efficiency. The central scientific question is: How to proactively steer the inference dynamics of diffusion models on the fly to stabilize particle weights and prevent collapse, while maintaining a lightweight, training-free framework that avoids the computational burden of solving high-dimensional PDEs or retraining?

**3. Overall Core Idea and Design Philosophy**
The overall core idea is to exploit a fundamental degree of freedom in the Feynman-Kac-type Fokker-Planck equation between the drift and particle potential. Instead of passively reweighting particles, DriftLite proactively steers them by "offloading" the problematic parts of the potential (which cause weight variance) into a new, corrective drift term. This mechanism absorbs sources of weight variation, preventing the weight collapse common in passive reweighting schemes. The design philosophy ensures the control is computed on-the-fly by restricting the search for the control drift to a finite-dimensional subspace, transforming the complex PDE problem into solving a small linear system per step, thus ensuring lightweight and scalable overhead.

**4. Core Innovation Points**
- Formulated and exploited a fundamental degree of freedom in the Feynman-Kac-type Fokker-Planck equation, establishing a principled trade-off between the particle drift and reweighting potential, and exploited it directly to actively minimize particle weight variance.
- Introduced DriftLite, a lightweight and training-free framework that computes a control drift on-the-fly to stabilize the sampling dynamics, derived directly from the principle of variance reduction.
- Derived two practical instantiations, Variance-Controlling Guidance (VCG) and Energy-Controlling Guidance (ECG), which are computationally efficient and require solving only a small additional linear system at each time step.
- Conducted extensive experiments on challenging benchmarks (high-dimensional GMMs, molecular particle systems, large-scale protein-ligand co-folding), demonstrating that DriftLite substantially reduces weight variance, stabilizes ESS, and improves final sample quality over current baselines.

**5. Overview of the Overall Technical Solution**
The solution starts with a pre-trained diffusion or flow-matching base model defining a forward/backward process. For inference-time scaling tasks (Annealing or Reward-Tilting) targeting a new distribution $q_T$, a modified backward process is defined along a path of distributions $(q_t)_{t\in[0,T]}$. While pure guidance injects information into the drift, it is biased. The exact Guidance-Based Dynamics (Proposition 2.1) adds a reweighting potential $g_t$ to the Fokker-Planck equation. Simulating this with SMC suffers from weight degeneracy due to the variance of $g_t$. DriftLite identifies a degree of freedom (Proposition 3.1) allowing the introduction of a control drift $b_t$ and a residual potential $\phi_t = g_t + h_t$. The goal is to choose $b_t$ to minimize the variance of $\phi_t$. The optimal control $b_t^*$ eliminates variance completely (Proposition 3.2) but requires solving an intractable PDE. Instead, DriftLite uses practical approximations (VCG and ECG) by restricting $b_t$ to a linear combination of basis functions (Ansatz 3.3 and 3.4), reducing the problem to solving a small linear system at each step. The controlled dynamics are then simulated using a weighted particle method with the controlled drift $\tilde{v}_t + b_t$ and residual potential $\phi_t$.

**6. Detailed Module Design**
- **Base Diffusion Model Module**: Defines the forward process $(x_s)_{s\in[0,T]}$ and backward process $(\overleftarrow{x}_t)_{t\in[0,T]}$ with drifts $u_s$ and $v_t$, marginals $p_s$ and $\overleftarrow{p}_t$, and diffusion coefficients $U_s$ and $V_t$. The backward drift $v_t$ is related to the forward drift and score function.
- **Inference-Time Scaling Module**: Targets distribution $q_T(x) \propto \overleftarrow{p}_T(x)^\gamma \exp(r(x)) = p_0(x)^\gamma \exp(r(x))$ for Annealing ($\gamma$) or Reward-Tilting ($r(x)$). Defines a path $q_t(x) \propto \overleftarrow{p}_t(x)^\gamma \exp(r_t(x))$ interpolating from noise to target.
- **Guidance-Based Dynamics Module**: Modifies the drift to $\tilde{v}_t$ using pure guidance. Derives the exact Feynman-Kac-type Fokker-Planck equation containing the drift $\tilde{v}_t$, diffusion, and a reweighting potential $g_t(x)$.
- **Weighted Particle Method Module (G-SMC)**: Simulates the dynamics using particles $\{x_t^{(i)}, w_t^{(i)}\}$ evolving under $\tilde{v}_t$ with weight updates based on the centered potential $\hat{g}_t$.
- **Drift Control Module (DriftLite)**:
  - *Degree of Freedom Mapping*: Introduces control drift $b_t$, defining residual potential $\phi_t = g_t + h_t$ where $h_t$ depends on $b_t$, the score $\nabla \log \overleftarrow{p}_t$, and $\nabla r_t$.
  - *Optimal Control Formulation*: Defines the optimal curl-free control $b_t^* = \nabla A_t^*$ solving a Poisson equation.
  - *VCG Sub-module*: Uses Ansatz 3.3 for linear control drift. Solves a linear system $A_t\theta_t = c_t$ to minimize the variance of $\phi_t$.
  - *ECG Sub-module*: Uses Ansatz 3.4 for linear control potential. Solves a linear system $A_t\theta_t = c_t$ to variationally minimize the energy functional $\mathcal{E}_t[A_t]$.
  - *Implementation Sub-module*: Selects basis functions (e.g., $\nabla r_t, \nabla \log \overleftarrow{p}_t, \overleftarrow{u}_t$ for VCG). Simulates the controlled SDE with effective drift $\tilde{v}_t + b_t$ and effective potential $\phi_t$.

**7. All Mathematical Formulas and Symbol Definitions**
- Forward SDE and Fokker-Planck:
  $dx_s = u_s(x_s)ds + U_s dw_s$ (SDE), 
  $\partial_s p_s(x) = -\nabla \cdot [p_s(x)u_s(x)] + \frac{U_s^2}{2} \Delta p_s(x)$ (FP)
- Backward SDE and Fokker-Planck ($t = T-s$, $\overleftarrow{*}_t = *_{T-t}$):
  $d\overleftarrow{x}_t = v_t(\overleftarrow{x}_t)dt + V_t dw_t$ (SDE), 
  $\partial_t \overleftarrow{p}_t(x) = -\nabla \cdot [\overleftarrow{p}_t(x)v_t(x)] + \frac{V_t^2}{2} \Delta \overleftarrow{p}_t(x)$ (FP)
- Backward drift relation ($u_s(x_s) = -F_s x_s$):
  $v_t(x) = -\overleftarrow{u}_t(x) + \frac{\overleftarrow{U}_t^2 + V_t^2}{2} \nabla \log \overleftarrow{p}_t(x)$
- Target distribution:
  $q_T(x) \propto \overleftarrow{p}_T(x)^\gamma \exp(r(x)) = p_0(x)^\gamma \exp(r(x))$
- Distribution path:
  $q_t(x) \propto \overleftarrow{p}_t(x)^\gamma \exp(r_t(x))$
- Pure Guidance Fokker-Planck:
  $\partial_t q_t(x) = -\nabla \cdot [\tilde{v}_t(x)q_t(x)] + \frac{V_t^2}{2} \Delta q_t(x)$
- Modified drift $\tilde{v}_t$:
  $\tilde{v}_t(x) = -\overleftarrow{u}_t(x) + \frac{\overleftarrow{U}_t^2 + V_t^2}{2} (\gamma \nabla \log \overleftarrow{p}_t(x) + \nabla r_t(x))$
- Guidance-Based Dynamics (Proposition 2.1):
  $\partial_t q_t(x) = -\nabla \cdot [\tilde{v}_t(x)q_t(x)] + \frac{V_t^2}{2} \Delta q_t(x) + q_t(x)g_t(x)$
  where $g_t(x) = G_t(x) - \mathbb{E}_{q_t}[G_t(\cdot)]$
  $G_t = \dot{r}_t - (1-\gamma)\nabla \cdot \overleftarrow{u}_t + \frac{\overleftarrow{U}_t^2}{2} (\Delta r_t - \gamma(1-\gamma)\|\nabla \log \overleftarrow{p}_t\|^2) + \nabla r_t^\top (-\overleftarrow{u}_t + \gamma \overleftarrow{U}_t^2 \nabla \log \overleftarrow{p}_t + \frac{\overleftarrow{U}_t^2}{2} \nabla r_t)$
- Weighted Particle Method (G-SMC):
  $\begin{cases} dx_t^{(i)} = \tilde{v}_t(x_t^{(i)})dt + V_t dw_t^{(i)}, & i \in [N] \\ d \log w_t^{(i)} = \hat{g}_t(x_t^{(i)}) := G_t(x_t^{(i)}) - \sum_{j=1}^N w_t^{(j)} G_t(x_t^{(j)}), & i \in [N] \end{cases}$
- Degree of Freedom (Proposition 3.1):
  $\partial_t q_t(x) = -\nabla \cdot [(\tilde{v}_t(x) + b_t(x)) q_t(x)] + \frac{V_t^2}{2} \Delta q_t(x) + q_t(x)\phi_t(x)$
  $\phi_t(x) = g_t(x) + h_t(x; b_t(x))$
  $h_t(x; b_t) = (\gamma \nabla \log \overleftarrow{p}_t(x) + \nabla r_t(x)) \cdot b_t(x) + \nabla \cdot b_t(x)$
- Optimal Control (Proposition 3.2):
  $b_t^*(x) = \nabla A_t^*(x)$
  $\nabla \cdot (q_t(x)\nabla A_t^*(x)) = -q_t(x)g_t(x)$
- VCG Objective (Ansatz 3.3):
  $b_t(x) = \sum_{i=1}^n \theta_t^i s_i(x)$
  $\min_{b_t} \text{Var}_{x\sim q_t}[\phi_t(x)] = \text{Var}_{x\sim q_t}[g_t(x) + h_t(x; b_t)]$
  Linear system $A_t\theta_t = c_t$, where $A_{ij} = \mathbb{E}_{q_t}[h_t^i h_t^j]$ and $c_i = -\mathbb{E}_{q_t}[g_t h_t^i]$
- ECG Objective (Ansatz 3.4):
  $A_t(x) = \sum_{i=1}^n \theta_t^i s_t^i(x)$, $b_t(x) = \nabla A_t(x) = \sum_{i=1}^n \theta_t^i \nabla s_t^i(x)$
  $\min_{A_t} \mathcal{E}_t[A_t] = \int (\frac{1}{2} q_t(x)\|\nabla A_t(x)\|^2 - q_t(x)g_t(x)A_t(x)) dx$
  Linear system $A_t\theta_t = c_t$, where $A_{ij} = \mathbb{E}_{q_t}[\nabla s_t^i{}^\top \nabla s_t^j]$ and $c_i = \mathbb{E}_{q_t}[g_t s_t^i]$
- Basis functions:
  VCG: $s_1(x) = \nabla r_t(x)$, $s_2(x) = \nabla \log \overleftarrow{p}_t(x)$, $s_3(x) = \overleftarrow{u}_t(x)$
  ECG: $s_1(x) = r_t(x)$, $s_2(x) = \log \overleftarrow{p}_t(x)$, $s_3(x) = \overleftarrow{U}_t(x)$

**8. Algorithm Pseudocode**
```text
Algorithm 1: DriftLite-VCG/ECG-SMC Implementation
Input: Guidance drift path ṽt from (2.4), uncentered potential path Gt from Prop. 2.1, time steps {tk}Mk=0, reward r(x), schedule βt, basis functions, number of particles N , ESS threshold τ .
1 Initialize particles x(i)0 ∼ ←p0, log-weights ℓ(i)0 ← 0, and weights w(i)0 ← 1N for i = 1, . . . , N ;
2 for k ← 0 to M − 1 do
3   Set ∆tk ← tk+1 − tk and form estimates of Atk and ctk using {(x(i)tk , w(i)tk )}i∈[N ];
4   Solve Atkθtk = ctk to obtain the control drift btk(·);
5   vefftk (·)← ṽtk(·) + btk(·), Hefftk (·)← Gtk(·) + htk(·; btk);
6   ĝefftk (x(i)tk )← Hefftk (x(i)tk )− ∑Nj=1 w(j)tk Hefftk (x(j)tk );
7   ℓ(i)tk+1 ← ℓ(i)tk + ĝefftk (x(i)tk )∆tk, wtk+1 ← softmax(ℓtk+1);
8   x(i)tk+1 ← x(i)tk + vefftk (x(i)tk )∆tk + Vtk √∆tk z(i), where z(i) ∼ N (0, I);
9   if ESS(wtk+1) < τ or periodically then
10    Resample {x(i)tk+1}i∈[N ] according to {w(i)tk+1}i∈[N ] and reset ℓ(i)tk+1 ← 0, w(i)tk+1 ← 1N for all i;
Output: Final samples {x(i)T , w(i)T }i∈[N ] from the last completed pass.

Algorithm 2: Iterative Refinement for Inference-Time Scaling (cumulative drift/potential updates)
Input: Guidance drift path ṽt from (2.4), uncentered potential path Gt from Prop. 2.1, time steps {tk}Mk=0, reward r(x), schedule βt, basis functions, number of refinement rounds K, number of particles N , ESS threshold τ .
1 Initialize effective drift ṽefft (·)← ṽt(·), and effective uncentered potential Hefft (·)← Gt(·);
2 for j ← 1 to K do
3   Initialize particles x(i)t0 ∼ ←p0, log-weights ℓ(i)t0 ← 0, and weights w(i)t0 ← 1N for i = 1, . . . , N ;
4   for k ← 0 to M − 1 do
5     Set ∆tk ← tk+1 − tk and form estimates of Atk and ctk using {(x(i)tk , w(i)tk )}i∈[N ];
6     Solve Atkθtk = ctk to obtain the control drift btk(·);
7     ṽefftk (·)← ṽefftk (·) + btk(·), Hefftk (·)← Hefftk (·) + htk(·; btk);
8     ĝefftk (x(i)tk )← Hefftk (x(i)tk )− ∑Nm=1 w(m)tk Hefftk (x(m)tk );
9     ℓ(i)tk+1 ← ℓ(i)tk + ĝefftk (x(i)tk )∆tk, wtk+1 ← softmax(ℓtk+1);
10    x(i)tk+1 ← x(i)tk + ṽefftk (x(i)tk )∆tk + Vtk √∆tk z(i), where z(i) ∼ N (0, I);
11    if ESS(wtk+1) < τ or periodically then
12      Resample {x(i)tk+1}i∈[N ] according to {w(i)tk+1}i∈[N ];
13      Reset ℓ(i)tk+1 ← 0, w(i)tk+1 ← 1N for all i;
Output: Final samples {(x(i)T , w(i)T )}i∈[N ] from the K-th refinement round.
```