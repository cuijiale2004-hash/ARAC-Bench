1. Research Background and Existing Pain Points
The research background of this paper lies in the domain of discrete diffusion models, which are a powerful class of generative models demonstrating strong performance across many discrete domains such as natural language text, molecules, and images. For efficiency, discrete diffusion typically parameterizes the generative (reverse) process with factorized distributions. This factorization is crucial because it allows all elements to be sampled in parallel, enabling a fully parallel sampling procedure instead of the slow sequential inference required by autoregressive (AR) models. 

However, significant existing pain points limit the practical application of conventional discrete diffusion models:
1. Factorized Parameterization Mismatch: The factorized parameterization greatly limits the model’s flexibility. In general, the target distribution for the reverse process (the marginalization of the posterior) is not factorized. This difference is especially notable when the number of steps T is small, creating a severe mismatch between the complicated target distribution and the limited parameterization of the generative process.
2. Computational Expense of Long Sampling: Because of the mismatch described above, direct training of a few-step conventional diffusion model fails. The model struggles to learn the target process in a small number of steps, thereby necessitating a long, computationally and time-expensive sampling procedure. In practice, the number of steps T required for discrete diffusion models can be comparable to the sequence length D, which undermines the speed advantage over autoregressive models.
3. Absence of Generative ODE: The fundamental difference between continuous and discrete diffusion is the existence of a generative ordinary differential equation (ODE) in the continuous case. Such an ODE defines continuous trajectories that uniquely map noise to data, enabling few-step generation or distillation. In discrete space, continuous deterministic trajectories that would uniquely map noise to data do not exist, rendering these advanced continuous acceleration techniques inapplicable.

2. Core Research Motivation and Scientific Questions
The core research motivation is to reduce the gap between the target distribution and the model approximation to enable few-step generation in discrete diffusion models. Since it is unclear how to make the reverse process more flexible without compromising the efficiency of parallel sampling, the authors propose making the forward (noising) process more flexible. By adapting the forward process, the target distribution induced by it can be adapted to the constraints of the generative process, allowing the generative process to remain factorized while matching the target defined by the noising process. The paper is inspired by motivational examples (like a mixture of two isotropic Gaussians or a discrete random walk) which demonstrate that some forward processes allow the generation of complex data in a small number of steps using factorized conditional distributions.

The primary scientific questions addressed are:
1. How can we construct a discrete diffusion model with only a few generative steps while maintaining the efficiency of factorized parallel sampling?
2. How should the forward process be structured and parameterized so that the target distribution it induces becomes factorized and easily matchable by a factorized reverse process?
3. How can the parameters of this flexible forward process be learned efficiently from data in an end-to-end manner without introducing inference overhead?

3. Overall Core Idea and Design Philosophy
The overall core idea is Forward-Learned Discrete Diffusion (FLDD), a framework for discrete diffusion models with a highly flexible parameterization of the forward process. Rather than fixing a Markovian forward chain as in conventional diffusion, FLDD adopts a non-Markovian formulation with learnable marginal and posterior distributions. The design philosophy is to align the forward and reverse processes: since the generative process is factorized by design for efficiency, the forward process is encouraged to produce a factorized target distribution that the reverse process can effectively match. The way information is destroyed at each individual coordinate depends on the entire datapoint and not just the time step or the coordinate itself. This flexibility allows the model to learn intermediate factorized structures that facilitate few-step generation. All parameters are trained end-to-end under the standard variational objective, preserving important properties of conventional diffusion models while introducing no inference overhead.

4. Core Innovation Points
1. Introduction of FLDD generalizing discrete diffusion models by introducing a learnable forward (noising) process, enabling better alignment of the forward and reverse processes with a small number of steps.
2. Non-Markovian formulation of the forward process with learnable marginal and posterior distributions, allowing the trajectory of each coordinate to have a complex, non-linear dependence on the entire datapoint.
3. Efficient parameterization of the forward process using Maximum Coupling for the posterior distributions, ensuring consistency with marginals while minimizing the amount of probability mass that must be moved.
4. An efficient, end-to-end, simulation-free training procedure that optimizes the standard variational bound, utilizing a relaxed warm-up strategy with Concrete distributions followed by unbiased optimization using the REINFORCE method.
5. The approach does not change the generative process and thus introduces no inference overhead, preserving the parallel sampling capability of factorized distributions.

5. Overview of the Overall Technical Solution
The overall technical solution begins by reformulating the forward process from its Markovian definition to a non-Markovian formulation where the forward-process distributions (marginals and posteriors) are learnable. The forward marginals are parameterized as factorized distributions conditioned on the entire data point, and the forward posteriors are parameterized using a non-parametric technique called Maximum Coupling to define a valid transport plan that minimizes mass movement. The model is trained end-to-end by minimizing the variational bound on the log-likelihood of the reverse process. Because the forward marginals involve discrete samples, the reparameterization trick cannot be applied directly. Thus, the optimization employs a relaxed warm-up phase using Concrete distributions with decreasing temperature, followed by the REINFORCE method for unbiased gradient estimation. During inference, the sampling procedure utilizes the standard factorized reverse process, requiring no forward process computation and thereby adding no computational overhead.

6. Detailed Module Design
1. Forward Marginals Module: The forward marginal distributions are proposed to use the same factorized form as the generative model. This allows efficient sampling of latent variables during training. Unlike conventional discrete diffusion, each coordinate's distribution depends on the entire data sample, not just the specific coordinate. Appropriate reparameterization is applied to enforce the boundary conditions where the distribution at step 0 equals the data point and the distribution at step T equals the prior.
2. Forward Posteriors Module (Maximum Coupling): To ensure the posterior distributions are consistent with the marginals and define a valid transport plan, Maximum Coupling is utilized. For constructing a transport plan between two 1-dimensional categorical distributions, Maximum Coupling attempts to minimize the amount of mass that must be moved. If the probability mass at a bin at the earlier step is greater than or equal to that at the later step, the state is kept the same. Otherwise, the excess mass from the later step is redistributed to bins with a deficit. This procedure is applied independently to each coordinate.
3. Optimization Module: The model is trained by minimizing the variational bound with respect to both the reverse and forward parameters. Because the samples from the forward marginals are discrete, gradients with respect to the forward parameters are difficult to compute. The solution involves:
   - Relaxed Warm-Up: Training starts with a continuous relaxation of the categorical distribution using the Concrete distribution with a temperature that exponentially decreases from 1 to 10^-3 over 10^4 to 10^5 steps. The posterior for a relaxed sample is constructed by combining the posteriors for discrete samples according to the components of the relaxed sample. This allows the use of the reparameterization trick.
   - Unbiased Optimization: After the warm-up, the model switches to the REINFORCE method to construct an unbiased gradient estimator for the discrete sampling process, allowing end-to-end training.
4. Generative Process Module: The generative process uses the standard factorized parameterization. It starts from a sample drawn from the prior distribution and proceeds backward in time, sampling from factorized categorical distributions at each step. It is emphasized that parameterizations where the model samples an approximate data point and resamples an intermediate step are not applicable, as the distribution of the data point given the latent variable will generally not be factorizable.

7. All Mathematical Formulas and Symbol Definitions
Autoregressive Model Density:
pθ(x) = ∏_{i=1}^{D} pθ(x_i | x_{<i})
where x_i represents coordinate number i of x, x_{<i} represents a prefix of first i coordinates, D is sequence length.

Forward Process:
q(z_{0:T} | x) = q(z_0 | x) ∏_{t=1}^{T} q(z_t | z_s)
where s = t - 1.

Reverse Process:
pθ(z_{0:T}, x) = p(x | z_0) p(z_T) ∏_{t=1}^{T} q(z_s | z_t)

Variational Bound Objective:
- log pθ(x) ≤ E_{u(t)q(z_t|x)} [ D_{KL}( q(z_s|z_t,x) ∥ pθ(z_s|z_t) ) ]_{Ldiff} + E_{q(z_0|x)} [ - log p(x|z_0) ]_{Lrec} + D_{KL}( q(z_T|x) | p(z_T) )_{Lprior}
where u(t) is a uniform discrete distribution over time steps 1, ..., T.

Target Distribution Marginalization:
q(z_s | z_t) = ∫ q(x | z_t) q(z_s | z_t, x) dx = E_{q(x|z_t)} [ q(z_s | z_t, x) ]

Factorized Reverse Process Parameterization:
pθ(z_s | z_t) = ∏_{i=1}^{D} pθ(z_s^i | z_t), where pθ(z_s^i | z_t) = Cat( z_s^i; v_θ^i(z_t, t) )

Non-Markovian Forward Process Formulation:
q(z_{0:T} | x) = q(z_T | x) ∏_{t=1}^{T} q(z_s | z_t, x)

Factorized Forward Marginals Parameterization:
qφ(z_t | x) = ∏_{i=1}^{D} qφ(z_t^i | x), where qφ(z_t^i | x) = Cat( z_t^i; u_φ^i(x, t) )
Boundary conditions: qφ(z_0 | x) = δ(z_0 - x) and qφ(z_T | x) = p(z_T)

Maximum Coupling Posterior Parameterization:
q(z_s | z_t) = Cat( z_s; u_{s|t} ), where 
u_{s|t}^j = { min(u_s^k, u_t^k) / u_t^k , z_t = k = j;
             min(0, u_s^k - u_t^k) / u_s^k * m_{s|t}^j , z_t = k ≠ j }
and m_{s|t} = min(0, u_s - u_t) / ∥min(0, u_s - u_t)∥

Gradients of the Diffusion Term:
∇_φ L_{diff} = ∇_φ E_{u(t)q_φ(z_t|x)} [ D_{KL}( q_φ(z_s|z_t,x) ∥ pθ(z_s|z_t) ) ]

REINFORCE Gradient Estimator:
∇_φ L_{diff} = E_{u(t)q_φ(z_t|x)} [ ∇_φ ( q_φ(z_t|x) ⌊q_φ(z_t|x)⌋_{sg} D_{KL}( q_φ(z_s|z_t,x) ∥ pθ(z_s|z_t) ) ) ]
where ⌊·⌋_{sg} is a stop-grad operation.

Relaxed Posterior for Concrete Distribution:
q̄_{τ,φ}(z_s^i | z̄_t^i, x) = ∑_{k=1}^{K} z̄_{t,k}^i q_φ(z_s^i | z_t^i = k, x)
where z̄_{t,k}^i denotes the component corresponding to discrete value k of coordinate i in the relaxed sample z̄_t.

8. Algorithm Pseudocode
Algorithm 1 Training procedure with REINFORCE method.
Require: Dataset, T – time-steps, u_φ(x, t) – forward parameterization, v_θ(z_t, t) – reverse parameterization
for learning iterations do
  x ∼ Dataset, t ∼ u(t), s = t - 1      ▷ Sampling data point x and two consecutive time steps
  u_s = u_φ(x, s), u_t = u_φ(x, t)      ▷ Parameters of q_φ(z_s|x) and q_φ(z_t|x) (Equation 9)
  z_t ∼ Cat(z_t; u_t)                   ▷ Sampling z_t from categorical distribution with parameters u_t
  Construct u_{s|t} by u_s, u_t and x    ▷ Calculating parameters of q_φ(z_s|z_t,x) (Section 3.2)
  v_{s|t} = v_θ(z_t, t)                 ▷ Parameters of the reverse process pθ(z_s|z_t) (Equation 7)
  L = ∑_{i=1}^{D} u_{s|t}^i log ( u_{s|t}^i / v_{s|t}^i )  ▷ Objective function is KL divergence (Equation 5)
  Gradient step on θ and φ w.r.t. ⟨u_t, z_t⟩ ⌊⟨u_t, z_t⟩⌋_{sg} L  ▷ REINFORCE functional (Section 3.3)
end for

Algorithm 2 Sampling procedure.
Require: T – time-steps, v_θ(z_t, t) – reverse parameterisation
z_T ∼ Cat(z_T; v_T)                     ▷ Sampling z_T from prior distribution p(z_T)
for t ∈ T, ..., 1 do
  v_{s|t} = v_θ(z_t, t)                 ▷ Parameters of the reverse process pθ(z_s|z_t) (Equation 7)
  z_s ∼ Cat(z_s; v_{s|t})               ▷ Sampling z_t from categorical distribution with parameters v_{s|t}
end for