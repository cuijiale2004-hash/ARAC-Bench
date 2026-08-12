**Principle 1: Variational Consistency and Theoretical Validity of Non-Markovian Learnable Forward Processes in Discrete Diffusion**

**Definition:**  
For discrete diffusion frameworks that jointly learn the forward noising process, the non-Markovian parameterization must preserve a valid variational lower bound on the data likelihood. The learnable forward marginals and posteriors must satisfy critical consistency constraints—namely, that the marginals integrate correctly across time steps, that boundary conditions hold exactly ($q(\mathbf{z}_0|\mathbf{x}) = \delta(\mathbf{z}_0 - \mathbf{x})$ and $q(\mathbf{z}_T|\mathbf{x}) \approx p(\mathbf{z}_T)$), and that the induced target distribution $q(\mathbf{z}_s|\mathbf{z}_t)$ is structurally compatible with the factorized reverse parameterization. This principle ensures that "end-to-end" training is not merely a slogan but corresponds to a principled, simulation-free optimization of a valid ELBO where forward and reverse parameters co-adapt.

**Core Evaluation Criteria:**
- **Marginal and Boundary Validity:** Does the learned forward process satisfy the marginal consistency equation and enforce exact boundary conditions through a principled reparameterization rather than soft constraints?
- **Objective Integrity:** Is the training objective explicitly derived as a valid variational bound, and does the joint optimization of $\theta$ and $\varphi$ follow from this derivation rather than heuristic alternating updates?
- **Target-Reverse Alignment:** Does the forward process actually induce a target distribution that the factorized reverse sampler can approximate, or does the theoretical formulation leave a persistent mismatch?
- **Clarity of Probabilistic Assumptions:** Are the independence assumptions (e.g., coordinate-wise Maximum Coupling) and their implications for expressivity clearly stated and justified?

---

**Principle 2: Comprehensive Computational Budget Analysis for Few-Step Discrete Generation Claims**

**Definition:**  
Claims of accelerated sampling via few-step generation must be evaluated within a rigorous computational accounting framework that encompasses both training and inference costs. Simply reducing the number of reverse diffusion steps is insufficient if the learned forward process introduces disproportionate training overhead (e.g., doubling per-iteration cost) or increases memory footprint. Reviewers must assess whether the paper reports total training time, inference FLOPs, GPU-hours, and parameter counts, and whether quality improvements persist when normalized for total compute expenditure. The evaluation should distinguish between "fewer steps" as an inference-time convenience and genuine efficiency gains under a fixed total budget.

**Core Evaluation Criteria:**
- **Training Cost Transparency:** Are per-iteration and total training costs quantified relative to conventional discrete diffusion that uses fixed forward processes?
- **Quality-vs-Budget Curves:** Does the paper report sample quality as a function of total computational budget (training + inference), rather than only as a function of step count $T$?
- **Inference Overhead Assessment:** Is the forward process completely excluded from the generative procedure, and is any residual overhead (e.g., auxiliary network evaluations) explicitly measured?
- **Architectural Efficiency:** Is the forward parameterization designed with computational efficiency in mind, or does the paper use symmetrically sized networks for forward and reverse processes without justification?

---

**Principle 3: Empirical Rigor on High-Dimensional Real-World Discrete Domains Beyond Toy Benchmarks**

**Definition:**  
Methods proposing general frameworks for discrete few-step generation must demonstrate effectiveness on realistic, high-dimensional discrete data—such as natural language text, molecular graphs, or complex structured outputs—and cannot rely primarily on low-dimensional synthetic problems (e.g., 2D Gaussian mixtures) or heavily simplified image benchmarks (e.g., binarized MNIST) as principal evidence. Reviewers must evaluate whether the chosen benchmarks represent genuine challenges for the method, whether comparisons against strong baselines are fair and current, and whether the authors detect and transparently report benchmark saturation where metric differences fall within noise margins.

**Core Evaluation Criteria:**
- **Domain Complexity:** Are the primary experiments conducted on high-dimensional discrete domains where the few-step challenge is non-trivial, or do toy experiments dominate the empirical narrative?
- **Baseline Competitiveness:** Are comparisons made against strong, relevant baselines (e.g., SEDD, LLaDA, specialized text diffusion models) with comparable architectural capacity and training regimes?
- **Benchmark Saturation Detection:** Does the paper acknowledge when performance differences across methods are marginal (e.g., molecular generation metrics near ceiling effects) and provide qualitative evidence (generated samples, visualizations) to supplement quantitative scores?
- **Failure Mode Analysis:** Are poor-performing cases (e.g., high perplexity on text generation) analyzed mechanistically rather than dismissed, with hypotheses linking failure to forward process limitations or data distribution mismatch?

---

**Principle 4: Clear Distinction and Compatibility with Existing Discrete Diffusion Paradigms**

**Definition:**  
Proposed frameworks must be explicitly positioned within the landscape of discrete generative modeling, distinguishing their contributions from closely related paradigms such as mask diffusion models (MDM), star-shaped diffusion, continuous-space distillation, and latent discrete diffusion. Reviewers must assess whether the authors articulate what is fundamentally novel versus incrementally adapted, whether they discuss structural compatibility with complementary acceleration techniques (e.g., classifier-free guidance, self-conditioning, latent encoding), and whether they honestly acknowledge scenarios where specialized existing methods outperform their general framework.

**Core Evaluation Criteria:**
- **Related Work Precision:** Is the relationship to structurally similar methods (e.g., Star-shaped DDPM, SEDD, LLaDA) precisely characterized in terms of forward process non-Markovianity, marginal factorization, and reverse process requirements?
- **Framework vs. Method Clarity:** Does the paper present itself as a general, domain-agnostic framework (compatible with future task-specific enhancements) or as a fully tuned system, and is this distinction maintained consistently?
- **Compatibility Analysis:** Is the method's compatibility with existing techniques (distillation, latent modeling, advanced guidance) discussed, or are these presented as incompatible alternatives without justification?
- **Honest Limitation Disclosure:** Are cases where specialized models achieve superior performance acknowledged, and are the trade-offs between generality and peak performance clearly explained?

---

**Principle 5: Optimization Stability and Gradient Estimation Rigor for Discrete Latent Forward Processes**

**Definition:**  
Learning the parameters of a discrete forward process requires gradient estimation through discrete sampling operations, which introduces well-known challenges of high variance and instability. Reviewers must evaluate whether the paper systematically addresses the choice of gradient estimator (e.g., REINFORCE, Gumbel-Softmax, REBAR, RELAX, Rao-Blackwellization), whether variance reduction techniques and control variates are compared, and whether the training procedure—including continuous relaxation warm-up phases and temperature annealing schedules—is robust and principled across diverse discrete domains.

**Core Evaluation Criteria:**
- **Estimator Justification:** Are the chosen gradient estimators for the discrete forward process justified, and are alternative standard estimators experimentally compared or theoretically discussed?
- **Variance Reduction:** Are control variates, straight-through estimators, or other variance reduction techniques evaluated, and is their impact on convergence stability quantified?
- **Warm-Up Principledness:** Is the transition from continuous relaxation (e.g., Concrete distribution) to discrete REINFORCE training theoretically grounded, and is the annealing schedule validated?
- **Cross-Domain Stability:** Does the optimization remain stable across different discrete modalities (text, molecules, images), or does training success depend on domain-specific tuning that undermines claims of generality?