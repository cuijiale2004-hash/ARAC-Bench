## ABSTRACT

Discrete diffusion models are a powerful class of generative models with strong performance across many domains. For efficiency, however, discrete diffusion typically parameterizes the generative (reverse) process with factorized distributions, which makes it difficult for the model to learn the target process in a small number of steps and necessitates a long, computationally expensive sampling procedure. To reduce the gap between the target and model distributions and enable few-step generation, we propose Forward-Learned Discrete Diffusion (FLDD), which introduces discrete diffusion with a learnable forward (noising) process. Rather than fixing a Markovian forward chain, we adopt a non-Markovian formulation with learnable marginal and posterior distributions. This allows the generative process to remain factorized while matching the target defined by the noising process. We train all parameters end-to-end under the standard variational objective. Experiments on various benchmarks show that, for a given number of sampling steps, our approach produces a higher quality samples than conventional discrete diffusion models using the same reverse parameterization.

## 1 INTRODUCTION

In the last years, diffusion models have demonstrated strong performance across many continuous (Hoogeboom et al., 2024) and discrete (Lou et al.) domains . Recent work has shown that distillation approaches and advanced training techniques allow learning a few-step (Salimans et al., 2024), or sometimes even a single-step, generative (Xu et al., 2025) procedure in the continuous domain. However, discrete diffusion still requires a significant number of consecutive generative steps (often comparable to the data dimensionality), which makes inference computationally and time expensive.

The fundamental difference between continuous and discrete diffusion is the existence of a generative ordinary differential equation (ODE) in the continuous case. Such an ODE defines continuous trajectories that connect the noise and data distributions. Therefore, by learning the trajectories of an ODE, we can sample noise and directly map it to a data point. In discrete space, however, such continuous deterministic trajectories that would uniquely map noise to data do not exist, so these techniques are not applicable. At the same time, direct training of a few-step conventional diffusion model fails due to a mismatch between the complicated target distribution and the limited parameterization of the generative process. However, recent work on more flexible forward processes in diffusion models demonstrates that learning the noising process can improve generative performance in both continuous (Bartosh et al., 2024) and discrete Shi et al. settings.

Inspired by these results, we propose Forward-Learned Discrete Diffusion (FLDD) – a framework for discrete diffusion models with a highly flexible parameterization of the forward process. We show how to parameterize the forward process so that a training iteration remains computationally efficient, while the way we destroy information at each individual coordinate depends on the entire datapoint and not just the time step, as in conventional diffusion. Moreover, we show how to train this model efficiently in an end-to-end manner by minimizing the variational bound on the model’s likelihood. The flexibility of FLDD enables better alignment of the two processes with a small number of steps, leading to improved generative performance.

We summarize our contributions as follows:

• We introduce Forward-Learned Discrete Diffusion (FLDD), generalizing discrete diffusion models and providing an efficient parameterization of the model.

• We provide an efficient, end-to-end, simulation-free training procedure.

• We demonstrate the robustness of our method on several discrete domains, including images, molecules, and natural language text.

## 2 BACKGROUND

## 2.1 AUTOREGRESSIVE MODELS

We aim to model a distribution over discrete sequences $q ( \mathbf { x } )$ , where $\mathbf { x } \in \{ 1 , \ldots , K \} ^ { D }$ , K is the number of discrete values, and D is the length of the sequence x. The standard approach for this is the autoregressive (AR) model, where each next token is predicted sequentially, conditioned on all previous ones. Formally, the density of the AR model can be written as:

$$
p _ {\theta} (\mathbf {x}) = \prod_ {i = 1} ^ {D} p _ {\theta} (\mathbf {x} ^ {i} | \mathbf {x} ^ {<   i}),\tag{1}
$$

where $\mathbf { x } ^ { i }$ represent coordinate number i of $\mathbf { x }$ and $\mathbf { x } ^ { < i }$ represents a prefix of first i coordinates.

An important drawback of the AR approach is its slow sequential inference. To sample a sequence of D elements, one must sample D times in sequence from the conditional distribution $p _ { \theta } ( \mathbf { x } ^ { i } | \mathbf { x } ^ { < i } )$

## 2.2 DIFFUSION MODELS

In contrast, diffusion models (DMs) provide a fully parallel sampling procedure. A DM consists of two main components: the forward and reverse processes. The forward, or noising, process is a Markov process that takes a data point x and progressively corrupts it, producing a sequence of latent variables $\mathbf { z } _ { 0 } , \mathbf { z } _ { 1 } , \ldots , \mathbf { z } _ { T } ;$

$$
q (\mathbf {z} _ {0: T} | \mathbf {x}) = q (\mathbf {z} _ {0} | \mathbf {x}) \prod_ {t = 1} ^ {T} q (\mathbf {z} _ {t} | \mathbf {z} _ {s}),\tag{2}
$$

where $s = t - 1$

The reverse process is also a Markov chain. It starts from a sample $\mathbf { z } _ { T }$ and proceeds backward in time, inverting the corruptions applied by the forward process:

$$
p _ {\theta} (\mathbf {z} _ {0: T}, \mathbf {x}) = p (\mathbf {x} | \mathbf {z} _ {0}) p (\mathbf {z} _ {T}) \prod_ {t = 1} ^ {T} q (\mathbf {z} _ {s} | \mathbf {z} _ {t}),\tag{3}
$$

where $p ( \mathbf { x } | \mathbf { z } _ { 0 } )$ is usually a simple, non-parametric distribution.

The training objective of diffusion models is a variational bound on the log-likelihood of the reverse process Ho et al. (2020):

$$
\begin{array}{r l} & {- \log p _ {\theta} (\mathbf {x}) \leq \underbrace {\mathbb {E} _ {u (t) q (\mathbf {z} _ {t} | \mathbf {x})} \left[ \mathrm{D} _ {\mathrm{KL}} \Big (q (\mathbf {z} _ {s} | \mathbf {z} _ {t} , \mathbf {x}) \| p _ {\theta} (\mathbf {z} _ {s} | \mathbf {z} _ {t}) \Big) \right]} _ {\mathcal {L} _ {\mathrm{diff}}} +} \\ & {\quad \underbrace {\mathbb {E} _ {q (\mathbf {z} _ {0} | \mathbf {x})} \Big [ - \log p (\mathbf {x} | \mathbf {z} _ {0}) \Big ]} _ {\mathcal {L} _ {\mathrm{rec}}} + \underbrace {\mathrm{D} _ {\mathrm{KL}} \Big (q (\mathbf {z} _ {T} | \mathbf {x}) | p (\mathbf {z} _ {T}) \Big)} _ {\mathcal {L} _ {\mathrm{prior}}},} \end{array}\tag{4}
$$

(5)

where $u ( t )$ is a uniform discrete distribution over the time steps $1 , \ldots , T$ . The three terms correspond to the reconstruction loss ${ \mathcal { L } } _ { \mathrm { r e c } }$ , the diffusion term ${ \mathcal { L } } _ { \mathrm { d i f f } }$ , and the prior term $\mathcal { L } _ { \mathrm { p r i o r } }$

Typically, the forward process is designed such that $\mathbf { z } _ { 0 } \approx \mathbf { x }$ and $q ( \mathbf { z } _ { T } | \mathbf { x } ) \approx p ( \mathbf { z } _ { T } )$ , where $p ( \mathbf { z } _ { T } )$ is a known prior. In practice, the reconstruction and prior terms are often negligible, and attention is usually focused on the diffusion term ${ \mathcal { L } } _ { \mathrm { d i f f } }$

Although a single step in DMs, following $p _ { \theta } ( \mathbf { z } _ { s } | \mathbf { z } _ { t } )$ , may have a computational cost comparable to sampling from $p _ { \theta } ( \mathbf { x } )$ in the AR approach, DMs allow this process to be parallelized, with all coordinates updated simultaneously at each step. Thus, instead of $D$ sequential steps in AR models, DMs require $\dot { T }$ sequential steps.

## 2.3 LIMITATIONS OF DIFFUSION MODELS

Despite this appealing formulation, in practice the number of steps $T$ required for DMs can be comparable to sequence length $D ,$ undermining the speed advantage. To see why DMs perform poorly when T is small, we need to examine what they actually learn.

It can be shown that, for a given forward process, the target distribution for the reverse process is the marginalization of the posterior:

$$
q (\mathbf {z} _ {s} | \mathbf {z} _ {t}) = \int q (\mathbf {x} | \mathbf {z} _ {t}) q (\mathbf {z} _ {s} | \mathbf {z} _ {t}, \mathbf {x}) d \mathbf {x} = \mathbb {E} _ {q (\mathbf {x} | \mathbf {z} _ {t})} \Big [ q (\mathbf {z} _ {s} | \mathbf {z} _ {t}, \mathbf {x}) \Big ].\tag{6}
$$

Thus, the forward process implicitly defines the target for the reverse process. If the reverse model $p _ { \theta } ( \mathbf { z } _ { s } | \mathbf { z } _ { t } )$ is not expressive enough to match the target distribution $q ( \mathbf { z } _ { s } | \mathbf { z } _ { t } )$ , it will fail to accurately approximate the data distribution $q ( \mathbf { x } )$

At the same time, efficient sampling from $p _ { \theta } ( \mathbf { z } _ { s } | \mathbf { z } _ { t } )$ is essential for DM performance. The most straightforward way to enable fast, parallel sampling is to parameterize the reverse process as factorized:

$$
p _ {\theta} (\mathbf {z} _ {s} | \mathbf {z} _ {t}) = \prod_ {i = 1} ^ {D} p _ {\theta} (\mathbf {z} _ {s} ^ {i} | \mathbf {z} _ {t}), \quad \text { where } \quad p _ {\theta} (\mathbf {z} _ {s} ^ {i} | \mathbf {z} _ {t}) = \operatorname{Cat} \Bigl (\mathbf {z} _ {s} ^ {i}; \mathbf {v} _ {\theta} ^ {i} (\mathbf {z} _ {t}, t) \Bigr),\tag{7}
$$

where $\mathbf { z } _ { s } ^ { i }$ denotes the i-th coordinate of the discrete sequence $\mathbf { z } _ { s }$

This factorization allows all elements to be sampled in parallel, but it also greatly limits the model’s flexibility. However, in general, the target distribution $q ( \mathbf { z } _ { s } | \mathbf { z } _ { t } )$ in Equation (6) is not factorized. This difference is especially notable when the number of steps T is small.

Importantly, the factorized parameterization in Equation (7) does not imply that each coordinate is treated independently. At step s, the distribution of each $\mathbf { z } _ { s } ^ { i }$ depends on all coordinates of the previous state $\mathbf { \bar { z } } _ { t } = [ \mathbf { z } _ { t } ^ { 1 } , \dots , \mathbf { z } _ { t } ^ { D } ]$ , not only on $ { \mathbf { z } } _ { t } ^ { i }$ . Thus, even though sampling is performed through factorized distributions, the resulting distribution $p _ { \theta } ( \mathbf { x } )$ can still be complex and non-factorized.

## 3 FORWARD-LEARNED DISCRETE DIFFUSION

To construct a discrete DM with only a few generative steps, we need to reduce the gap between the target distribution $q ( \mathbf { z } _ { s } | \mathbf { z } _ { t } )$ (Equation (6)) and the model approximation $p _ { \theta } ( \mathbf { z } _ { s } | \mathbf { z } _ { t } )$ (Equation (7)). It is unclear how to make the reverse process more flexible without compromising efficiency. Instead, we propose to make the forward process more flexible. Then, the target distribution induced by the forward process (Equation (6)) can be adapted to the constraints of the generative process and enable few-step generation. To build intuition for the desired structure of the model, let us consider two motivational examples.

First, consider a mixture of two isotropic Gaussian distributions in D-dimensional space: $q ( \mathbf { x } ) =$ $w \mathcal { N } ( \mathbf { x } ; \mu _ { 1 } , I ) + ( 1 - w ) \mathcal { N } ( \mathbf { x } ; \mu _ { 2 } , I )$ . In the general case, we cannot sample from this distribution in a single step by drawing a sample from a factorized distribution, since the coordinates are correlated. However, we can do it in two steps: first sample the index of the Gaussian, and then sample from the chosen Gaussian. In this case, both samples come from factorized conditional distributions.

A less trivial example is a discrete random walk. Consider a process that starts at $\mathbf { x } ^ { 1 } = 0$ and takes steps (either $+ 1 \mathrm { o r } - 1 ) \colon \mathbf { x } ^ { i + 1 } = \mathbf { x } ^ { i } \pm 1$ . This process generates sequences of D elements and can be naturally modeled in D steps with an autoregressive approach. Interestingly, it can also be modeled in just two steps. First, sample $D - 1$ independent ±1 values, which define the individual steps. Then, by taking prefix sums, we can construct the full random-walk sequence.

These examples demonstrate that, in contrast to conventional discrete diffusion with uniform-noise injection or absorption (masking), some forward processes allow generation of complex data in a small number of steps. The question is: what should this process be? Unfortunately, in the general case, it is unclear how the forward process should look. This question is especially challenging because the structure of the forward process must depend on the underlying data distribution. Therefore, in this work we propose learning it from data.

In this section, we present Forward-Learned Discrete Diffusion (FLDD) – a discrete diffusion framework that enables end-to-end training of both the forward and reverse processes, which leads to better likelihood estimation and allows a significant reduction in the number of generative steps. At the same time, our approach does not change the reverse process and, as a result, the generative procedure, so it introduces no inference overhead. FLDD also preserves important properties of conventional diffusion models, such as the variational objective and a simulation-free optimization procedure.

## 3.1 LEARNABLE FORWARD PROCESS

Since we want to modify the target (Equation (6)), we must change the forward process that induces it. Let us begin by generalizing the forward process.

The forward process is only required during training. From the objective in Equation (5), we observe that the forward process must satisfy two conditions: efficient sampling from the marginals $q ( \mathbf { z } _ { t } | \mathbf { x } )$ and tractable posteriors $q ( \mathbf { z } _ { s } | \mathbf { z } _ { t } , \mathbf { x } )$ for evaluating the KL divergence. With this in mind, we can reformulate the forward process from its Markovian definition in Equation (2) to the following non-Markovian form:

$$
q (\mathbf {z} _ {0: T} | \mathbf {x}) = q (\mathbf {z} _ {T} | \mathbf {x}) \prod_ {t = 1} ^ {T} q (\mathbf {z} _ {s} | \mathbf {z} _ {t}, \mathbf {x}).\tag{8}
$$

This non-Markovian formulation enables a more flexible and efficient parameterization of the forward process. Specifically, we suggest making the forward-process distributions $q _ { \varphi } ( \mathbf { z } _ { t } | \mathbf { x } )$ and $q _ { \varphi } ( \mathbf { z } _ { s } | \mathbf { z } _ { t } , \mathbf { x } )$ learnable. If the parameterization is sufficiently flexible, we can expect to find parameters $\varphi$ such that the resulting target distribution $q _ { \varphi } ( \mathbf { z } _ { s } | \mathbf { z } _ { t } )$ (Equation (6)) is factorized.

To learn the parameters $\varphi ,$ we optimize the same variational objective as in Equation (5). In this setup, just as the reverse process adapts to the forward process, the forward process also adapts to the reverse one. Since the generative process is factorized by design (Equation (7)), the forward process is encouraged to produce a factorized target distribution that the reverse process can effectively match.

## 3.2 PARAMETERIZATION OF THE FORWARD PROCESS

Forward Marginals. For the forward marginal distributions $q _ { \varphi } ( \mathbf { z } _ { t } | \mathbf { x } )$ , we propose using the same form as the generative model (Equation (7)) and parameterizing them as factorized distributions:

$$
q _ {\varphi} (\mathbf {z} _ {t} | \mathbf {x}) = \prod_ {i = 1} ^ {D} q _ {\varphi} (\mathbf {z} _ {t} ^ {i} | \mathbf {x}), \quad \text { where } \quad q _ {\varphi} (\mathbf {z} _ {t} ^ {i} | \mathbf {x}) = \operatorname{Cat} \Bigl (\mathbf {z} _ {t} ^ {i}; \mathbf {u} _ {\varphi} ^ {i} (\mathbf {x}, t) \Bigr).\tag{9}
$$

This parameterization allows us to sample $\mathbf { z } _ { t }$ given x efficiently during training. However, unlike in conventional discrete diffusion models, each $ { \mathbf { z } } _ { t } ^ { i }$ may come from a complex distribution that depends on the entire data sample $\mathbf { x } = \mathbf { x } ^ { 1 } , \ldots , \mathbf { x } ^ { D }$ , not just the i-th element $\mathbf { x } ^ { i { \bar { \mathbf { \alpha } } } }$

In addition, we must enforce the boundary conditions $q _ { \varphi } ( \mathbf { z } _ { 0 } | \mathbf { x } ) = \delta ( \mathbf { z } _ { 0 } - \mathbf { x } )$ and $q _ { \varphi } ( \mathbf { z } _ { T } | \mathbf { x } ) = p ( \mathbf { z } _ { T } )$ Both properties can be ensured through an appropriate reparameterization of $\mathbf { u } _ { \varphi }$

Forward Posteriors. In addition to being tractable, the posterior distribution $q _ { \varphi } ( \mathbf { z } _ { s } | \mathbf { z } _ { t } , \mathbf { x } )$ must be consistent with the marginals $q _ { \varphi } ( \mathbf { z } _ { t } | \mathbf { x } )$

$$
q _ {\varphi} (\mathbf {z} _ {s} | \mathbf {x}) = \int q _ {\varphi} (\mathbf {z} _ {t} | \mathbf {x}) q _ {\varphi} (\mathbf {z} _ {s} | \mathbf {z} _ {t}, \mathbf {x}) d \mathbf {z} _ {t}.\tag{10}
$$

n other words, the posteriors $q _ { \varphi } ( \mathbf { z } _ { s } | \mathbf { z } _ { t } , \mathbf { x } )$ must define a valid transport plan for the probability mass across consecutive time steps. We propose using a simple non-parametric technique, Maximum Coupling, for posterior parameterization.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 Training procedure with REINFORCE method.

Require: Dataset, T - time-steps,  $u_{\varphi}(\mathbf{x}, t)$  – forward parameterization,  $\mathbf{v}_{\theta}(\mathbf{z}_{t}, t)$  – reverse parameterization

for learning iterations do

    x ~ Dataset,  $t \sim u(t)$ , s = t - 1 ▷ Sampling data point x and two consecutive time steps
     $\mathbf{u}_{s} = u_{\varphi}(\mathbf{x}, s)$ ,  $\mathbf{u}_{t} = u_{\varphi}(\mathbf{x}, t)$  ▷ Parameters of  $q_{\varphi}(\mathbf{z}_{s}|\mathbf{x})$  and  $q_{\varphi}(\mathbf{z}_{t}|\mathbf{x})$  (Equation (9))
     $z_{t} \sim \text{Cat}(z_{t}; \mathbf{u}_{t})$  ▷ Sampling  $z_{t}$  from categorical distribution with parameters  $u_{t}$ 
    Construct  $u_{s|t}$  by  $u_{s}$ ,  $u_{t}$  and x ▷ Calculating parameters of  $q_{\varphi}(\mathbf{z}_{s}|\mathbf{z}_{t}, \mathbf{x})$  (Section 3.2)
    $v_{s|t} = v_{\theta}(\mathbf{z}_{t}, t)$  ▷ Parameters of the reverse process  $p_{\theta}(\mathbf{z}_{s}|\mathbf{z}_{t})$  (Equation (7))
    $L = \sum_{i=1}^{D} u_{s|t}^{i} \log \frac{u_{s|t}^{i}}{v_{s|t}^{i}}$  ▷ Objective function is KL divergence (Equation (5))
    Gradient step on  $\theta$  and  $\varphi$  w.r.t.  $\frac{\langle u_{t}, z_{t} \rangle}{[\langle u_{t}, z_{t} \rangle]_{sg}} L$  ▷ REINFORCE functional (Section 3.3)

end for
</div>

Suppose we want to construct a transport plan between two 1-dimensional categorical distributions with parameters ${ \bf u } _ { s } , { \bf u } _ { t } \in \Delta ^ { K - 1 }$ . Intuitively, when moving probability mass from $\mathbf { u } _ { t }$ to $\mathbf { u } _ { s }$ , Maximum Coupling attempts to minimize the amount of mass that must be moved. Specifically, for $\mathbf { z } _ { t } = k .$ , if $\mathbf { \bar { u } } _ { s } ^ { k } \geq \mathbf { u } _ { t } ^ { k }$ , we keep ${ \bf z } _ { s } = { \bf z } _ { t }$ . Otherwise, if $\mathbf { u } _ { s } ^ { k } < \mathbf { u } _ { t } ^ { k }$ , we redistribute the excess mass $\mathbf { u } _ { t } ^ { k } - \mathbf { u } _ { s } ^ { k }$ from bin k to bins with a deficit.

Formally, we can write it as:

$$
\begin{array}{r l} q (\mathbf {z} _ {s} | \mathbf {z} _ {t}) = \mathrm{Cat} \big (\mathbf {z} _ {s}; \mathbf {u} _ {s | t} \big), & \mathrm{where} \quad \mathbf {u} _ {s | t} ^ {j} = \left\{ \begin{array}{l l} \frac {\min (\mathbf {u} _ {s} ^ {k} , \mathbf {u} _ {t} ^ {k})}{\mathbf {u} _ {t} ^ {k}}, & z _ {t} = k = j, \\ \frac {\min (0 , \mathbf {u} _ {s} ^ {k} - \mathbf {u} _ {t} ^ {k})}{\mathbf {u} _ {s} ^ {k}} \mathbf {m} _ {s | t} ^ {j}, & z _ {t} = k \neq j. \end{array} \right. \\ \mathrm{and} & \mathbf {m} _ {s | t} = \frac {\min (0 , \mathbf {u} _ {s} - \mathbf {u} _ {t})}{\| \min (0 , \mathbf {u} _ {s} - \mathbf {u} _ {t}) \|}. \end{array}\tag{11}
$$

Here, $\mathbf { m } _ { s \mid t }$ represents the parameters of a categorical distribution corresponding to the deficit of probability mass at timestep s compared to timestep t. We redistribute the extra probability mass from $\mathbf { u } _ { t }$ to $\mathbf { u } _ { s }$ according to $\mathbf { m } _ { s \mid t }$ . Importantly, the computation of the posterior parameters requires only simple vector operations, so it can be performed efficiently.

To construct the full D-dimensional distribution $q _ { \varphi } ( \mathbf { z } _ { s } | \mathbf { z } _ { t } , \mathbf { x } )$ , we apply the Maximum Coupling procedure independently to each coordinate, using the parameters ${ \bf u } _ { s } ~ = ~ { \bf u } _ { \varphi } ( { \bf x } , s )$ and $\begin{array} { r l } { \mathbf { u } _ { t } } & { { } = } \end{array}$ $\mathbf { u } _ { \varphi } ( \mathbf { x } , t )$ (Equation (9)).

We emphasize that, with this parameterization, the distributions of forward trajectories for each coordinate are conditionally independent given the data point x. However, in contrast to conventional discrete diffusion, the trajectory of each coordinate ${ \bf z } _ { t } ^ { \imath }$ may have a complex, non-linear dependence on the entire datapoint x, not just on $\mathbf { x } ^ { i }$ . Moreover, marginal distributions can still be complex, and the unconditional distribution of forward trajectories remains expressive.

The parameterization of the forward process described here is not unique. For instance, we may trade off sampling efficiency of the marginals for additional flexibility by making them partially autoregressive. For the posterior, we could use element-wise optimal transport with respect to a chosen metric or introduce dependencies between coordinates or additional learnable parameters. We believe there may be better ways to parameterize the forward process, but we leave this question for future research.

## 3.3 OPTIMIZATION

As mentioned in Section 3.1, we train the model by minimizing the same variational objective as in conventional diffusion (Equation (5)) with respect to the parameters θ and $\varphi .$ . While the objective itself is fully tractable, its gradients with respect to $\varphi$ are difficult to compute. Let us take a closer look at the gradients of the diffusion term:

$$
\nabla_ {\varphi} \mathcal {L} _ {\mathrm{diff}} = \nabla_ {\varphi} \mathbb {E} _ {u (t) q _ {\varphi} (\mathbf {z} _ {t} | \mathbf {x})} \left[ \mathrm{D} _ {\mathrm{KL}} \Big (q _ {\varphi} (\mathbf {z} _ {s} | \mathbf {z} _ {t}, \mathbf {x}) \| p _ {\theta} (\mathbf {z} _ {s} | \mathbf {z} _ {t}) \Big) \right].\tag{12}
$$

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 2 Sampling procedure.

Require: T - time-steps,  $\mathbf{v}_{\theta}(\mathbf{z}_{t}, t)$  – reverse parameterisation
 $z_{T} \sim \text{Cat}(\mathbf{z}_{T}; \mathbf{v}_{T})$  ▷ Sampling  $z_{T}$  from prior distribution  $p(\mathbf{z}_{T})$ 
for  $t \in T, \ldots, 1$  do
    $\mathbf{v}_{s|t} = \mathbf{v}_{\theta}(\mathbf{z}_{t}, t)$  ▷ Parameters of the reverse process  $p_{\theta}(\mathbf{z}_{s}|\mathbf{z}_{t})$  (Equation (7))
    $z_{s} \sim \text{Cat}(\mathbf{z}_{s}; \mathbf{v}_{s|t})$  ▷ Sampling  $z_{t}$  from categorical distribution with parameters  $v_{s|t}$ 
end for
</div>

The expression inside the expectation is fully differentiable. However, the distribution $q _ { \varphi } ( \mathbf { z } _ { t } | \mathbf { x } )$ and the samples $\mathbf { z } _ { t }$ are discrete, which prevents us from using the reparameterization trick to efficiently estimate gradients.

Unbiased Optimization. The standard approach for computing gradients of this form is the REIN-FORCE method Williams (1992). We can rewrite the gradients of the diffusion loss as follows:

$$
\nabla_ {\varphi} \mathcal {L} _ {\mathrm{diff}} = \mathbb {E} _ {u (t) q _ {\varphi} (\mathbf {z} _ {t} | \mathbf {x})} \left[ \nabla_ {\varphi} \left(\frac {q _ {\varphi} (\mathbf {z} _ {t} | \mathbf {x})}{\lfloor q _ {\varphi} (\mathbf {z} _ {t} | \mathbf {x}) \rfloor_ {\mathrm{sg}}} \mathrm{D} _ {\mathrm{KL}} \Big (q _ {\varphi} (\mathbf {z} _ {s} | \mathbf {z} _ {t}, \mathbf {x}) \| p _ {\theta} (\mathbf {z} _ {s} | \mathbf {z} _ {t}) \Big)\right) \right],\tag{13}
$$

where $\lfloor \cdot \rfloor _ { \mathrm { s g } }$ is a stop-grad operation.

In this way, we can use the Monte Carlo method to build an unbiased gradient estimator and train the model end-to-end. Importantly, as in conventional diffusion, we optimize a variational bound on the model’s likelihood. We summarize the training algorithm in Algorithm 1.

Relaxed Warm-Up. Unfortunately, REINFORCE is known to produce high-variance gradients, so training a model from scratch with REINFORCE alone often leads to instability. To obtain a better initialization, we adopt a different strategy.

We begin training with a continuous relaxation of the categorical distribution $q _ { \varphi } ( \mathbf { z } _ { t } | \mathbf { x } )$ using the Concrete distribution Jang et al. (2016); Maddison et al. (2016). Specifically, instead of drawing hard samples $\mathbf { z } _ { t }$ according to Equation (9), we use the same parameters $\mathbf { u } _ { \varphi } ( \mathbf { x } , t )$ to define a Concrete distribution $\bar { q } _ { \tau , \varphi } ( \bar { \mathbf { z } } _ { t } | \mathbf { x } ) \approx q _ { \varphi } ( \mathbf { z } _ { t } | \mathbf { x } )$ with temperature $\tau .$ . To construct a posterior for a relaxed sample $\bar { \mathbf { z } } _ { t } ,$ we combine the posteriors for discrete samples $\mathbf { z } _ { t }$ according to the components of $\bar { \mathbf { z } } _ { t }$ :

$$
\bar {q} _ {\tau , \varphi} (\mathbf {z} _ {s} ^ {i} | \bar {\mathbf {z}} _ {t} ^ {i}, \mathbf {x}) = \sum_ {k = 1} ^ {K} \bar {\mathbf {z}} _ {t} ^ {i, k} q _ {\varphi} (\mathbf {z} _ {s} ^ {i} | \mathbf {z} _ {t} ^ {i} = k, \mathbf {x}),\tag{14}
$$

where $\bar { \mathbf { z } } _ { t } ^ { i , k }$ denotes the component corresponding to discrete value k of coordinate i in the relaxed sample $\bar { \mathbf { z } } _ { t }$ . Thanks to the simple structure of the posterior distribution (Section 3.2), this weighted average does not add computational overhead.

Relaxed samples allow us to use the reparameterization trick for gradient estimation. Training starts with a temperature $\tau = 1$ , which we exponentially decrease to $\tau = 1 0 ^ { - 3 }$ over $1 0 ^ { 4 }$ to $1 0 ^ { \overline { { 5 } } }$ optimization steps. After this warm-up phase, we switch to training the model with the REINFORCE approach described above.

## 3.4 GENERATIVE PROCESS

We use the standard parameterization of the generative process as discussed in Section 2.2 (Equation (7)). Importantly, the sampling procedure does not require the forward process, so we do not add any computational overhead for inference. We summarize the generative procedure in Algorithm 2.

It is important to note that another common parameterization, where the model samples an approximate data point xˆ and then resamples an intermediate step $q _ { \varphi } ( \mathbf { z } _ { s } | \mathbf { z } _ { t } , \hat { \mathbf { x } } )$ , is not applicable in our formulation. Although it may be possible to make $q _ { \varphi } ( \mathbf { z } _ { s } | \mathbf { z } _ { t } )$ (Equation (6)) factorized, the distribution $q _ { \varphi } ( \mathbf { x } | \mathbf { z } _ { t } )$ will generally not be factorizable. As a result, the reverse process cannot learn it accurately.

![](images/a36f3b973cb7bcd2838a971a586fbcab92fd95f50126d9d47ffa237dcbd6b6e8.jpg)  
Figure 1: Training data distribution and learned dynamics for a mixture of two Gaussians.

![](images/8c03013e68399b42bc3fdff94dd75d282c219119aea7779a086960177647bd22.jpg)

![](images/543542406fb2415feb8cfaad7da0aff5ac4f973781afa78c0968ee20ebf84632.jpg)  
(b) Binarized MNIST generation with learned forward process in 4 steps.  
Figure 2: Learned dynamics for Binarized MNIST dataset. Generative process starts from prior distribution $p ( z _ { T } )$ at time steps $t = T$ and goes backwards in time.

## 4 EXPERIMENTS

In this section, we provide experiments on both synthetic and real-world data. We present FLDD as a general framework for reducing the number of sampling steps. Therefore, the goal of this section is to demonstrate that FLDD indeed improves sampling efficiency across different settings. Our objective is not to outperform all other approaches in every domain. Instead, FLDD is compatible with existing extensions. We also assume there is potential room for optimizations and better design choices in these domains.

In all our experiments, we use the same neural network architectures and hyperparameters to parameterize both the generative process $p _ { \theta } ( \mathbf { z } _ { s } | \mathbf { z } _ { t } )$ and the forward marginals $q _ { \varphi } ( \mathbf { x } | \mathbf { z } _ { t } )$ . We emphasize that, while this doubles the number of learnable parameters, it does not affect the capacity of the generative process. Additional details on parameterization, training, and evaluation are provided in Section A.

## 4.1 TOY DATA

First, we consider a toy setting with $\mathbf { x } \in \{ 1 , \ldots , 5 0 \} ^ { 2 }$ . We train a diffusion model to generate samples in just two steps. In Figure 1, we show results for a mixture of two Gaussian distributions. As we discussed in Section 3, each individual Gaussian can be generated in a single step, since the coordinates are independent. However, generating the mixture requires at least two steps. In this case, the model learns first to generate an intermediate factorized structure and then to produce the final data.

Table 1: Results on ROCStories dataset.

<table><tr><td></td><td>MAUVE ↑</td><td>PPL ↓</td><td>Div ↑</td></tr><tr><td>GPT2 (Radford et al., 2019)</td><td>0.789</td><td>20.5</td><td>0.252</td></tr><tr><td>GPT Neo (Black et al., 2021)</td><td>0.720</td><td>19.9</td><td>0.258</td></tr><tr><td>AR-Diffusion (Wu et al., 2023)</td><td>0.066</td><td>41.8</td><td>0.101</td></tr><tr><td>DiffuSeq (Gong et al., 2022)</td><td>0.086</td><td>50.5</td><td>0.124</td></tr><tr><td>SeqDiffuSeq (Yuan et al., 2022)</td><td>0.103</td><td>29.3</td><td>0.137</td></tr><tr><td>TESS (Mahabadi et al., 2023)</td><td>0.061</td><td>22.4</td><td>0.163</td></tr><tr><td>SEDD (Lou et al., 2023)</td><td>0.598</td><td>70.8</td><td>0.336</td></tr><tr><td>LD4LG (Lovelace et al., 2023)</td><td>0.716</td><td>30.6</td><td>0.331</td></tr><tr><td>COSMOS (Meshchaninov et al., 2025)</td><td>0.940</td><td>26.3</td><td>0.346</td></tr><tr><td>FLDD, T = 100</td><td>0.538</td><td>55.2</td><td>0.280</td></tr><tr><td>FLDD, T = 10</td><td>0.511</td><td>60.5</td><td>0.285</td></tr></table>

Table 2: Molecular generation results.

<table><tr><td rowspan="2"></td><td colspan="3">QM9</td><td colspan="3">ZINC250k</td></tr><tr><td>Valid ↑</td><td>Unique ↑</td><td>FCD ↓</td><td>Valid ↑</td><td>Unique ↑</td><td>FCD ↓</td></tr><tr><td>MoFlow (Zang &amp; Wang, 2020)</td><td>91.36</td><td>98.65</td><td>4.467</td><td>63.11</td><td>99.99</td><td>20.931</td></tr><tr><td>EDP-GNN (Niu et al., 2020)</td><td>47.52</td><td>99.25</td><td>2.680</td><td>82.97</td><td>99.79</td><td>16.737</td></tr><tr><td>GraphEBM (Liu et al., 2021)</td><td>8.22</td><td>97.90</td><td>6.143</td><td>5.29</td><td>98.79</td><td>35.471</td></tr><tr><td>GDSS (Jo et al., 2022)</td><td>95.72</td><td>98.46</td><td>2.900</td><td>97.01</td><td>99.64</td><td>14.656</td></tr><tr><td>Digress Vignac et al. (2022)</td><td>99.00</td><td>96.20</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Flow Matching (Lipman et al., 2023)</td><td>94.10</td><td>98.20</td><td>5.155</td><td>94.01</td><td>96.68</td><td>18.764</td></tr><tr><td>Dirichlet FM (Stark et al., 2024)</td><td>99.10</td><td>98.15</td><td>0.888</td><td>97.52</td><td>99.20</td><td>14.222</td></tr><tr><td>CatFlow (Eijkelboom et al., 2024)</td><td>99.81</td><td>99.95</td><td>0.441</td><td>99.21</td><td>100.00</td><td>13.211</td></tr><tr><td>LO-ARM (Wang et al., 2025)</td><td>99.85</td><td>98.85</td><td>0.240</td><td>96.70</td><td>100.00</td><td>3.229</td></tr><tr><td>FLDD, T = 100</td><td>99.67</td><td>98.93</td><td>0.328</td><td>97.79</td><td>100.00</td><td>8.487</td></tr><tr><td>FLDD, T = 10</td><td>99.08</td><td>99.95</td><td>0.385</td><td>96.77</td><td>100.00</td><td>10.414</td></tr></table>

## 4.2 REAL WORLD DATA

We evaluate FLDD on the text dataset ROCStories (Mostafazadeh et al., 2016) and two molecular generation benchmarks: QM9 (Ramakrishnan et al., 2014) and ZINC250k (Irwin et al., 2012). The results are summarized in Table 1 and Table 1. For each problem, we evaluate FLDD in two settings: first, with T = 100, which is comparable to what other models use; and second, with a significantly reduced T = 10 steps.

In all cases, FLDD demonstrates results comparable to other baselines with T = 100. Some baselines may outperform FLDD, but we attribute this to suboptimal parameterization and hyperparameters that we do not tune. As we discuss in Section 3.4, it is important for FLDD to parameterize the distribution $q _ { \varphi } ( \mathbf { z } _ { s } | \mathbf { z } _ { t } )$ directly, which is known to be suboptimal in conventional diffusion. We believe there are ways to reparameterize the generative process efficiently while preserving the same properties. However, we leave this for future research.

At the same time, FLDD shows only a slight drop in performance with T = 10 steps, whereas other diffusion models do not generate realistic-looking data with such a small number of steps. This demonstrates that FLDD allows a significantly better trade-off between sample quality and inference speed.

## 4.3 LEARNABLE MASK DIFFUSION MODELS

A popular variation of discrete diffusion is mask discrete models (MDM), where the forward process randomly replaces tokens from the sequence x with masks, and the reverse process replaces mask tokens with tokens from the vocabulary. For MDM, it is clear that if there is correlation between coordinates and the number of generative steps T is significantly smaller than the sequence length D, the generative model will fail, since it will have to unmask tokens independently without taking correlations into account. In the general case, this problem is unavoidable in conventional MDM. However, with FLDD, we can improve performance by learning to unmask at each step only tokens that are less correlated (conditionally on current observations).

To demonstrate how this works, we restrict the forward process to masking. Specifically, $q _ { \varphi } ( \mathbf { z } _ { t } | \mathbf { x } )$ defines only the probability of masking each token conditionally on the full data point x. Then, we expect the model to learn masking schedules such that the generative process corresponds to unmasking less correlated tokens.

We conduct an experiment on the MNIST dataset Salakhutdinov & Hinton (2009). Figure 2 shows images generated by a conventional MDM and by FLDD with learnable masking schedules. Since an MDM learns to unmask pixels uniformly, it struggles to generate natural-looking images. In contrast, our model learns more convenient intermediate distributions and produces more realistic samples. We emphasize that the generative processes in both models share the same parameterization.

## 5 RELATED WORK

Diffusion models Ho et al. (2020) have recently been adapted to discrete spaces for language, graphs, and molecules. Text diffusion includes sequence-to-sequence formulations Gong et al. (2022); Yuan et al. (2022), hybrid or autoregressively-conditioned variants Wu et al. (2023), simplex- or ratiobased objectives Mahabadi et al. (2023); Lou et al. (2023), and masked/absorbing-state processes with improved schedules and theory Shi et al.. For molecular and graph data, flow-/score-based and diffusion-style approaches continue to advance sampling quality and validity Lipman et al. (2023); Stark et al. (2024); Eijkelboom et al. (2024); Jo et al. (2022); Vignac et al. (2022); Wang et al. (2025). Reducing the number of reverse steps has been pursued via distillation and consistency-style accelerations in the continuous domain Salimans et al. (2024); Xu et al. (2025), but discrete models often retain long sampling chains due to factorized reverse parameterizations. Our work targets this limitation by learning a forward process that induces a reverse target distribution compatible with the same efficient, factorized generative family, thereby enabling few-step sampling without changing the sampler itself.

Most diffusion methods fix the forward (noising) dynamics, which implicitly defines the target that the reverse model must match. Recent work in the continuous setting shows that learning the forward process can tighten likelihood bounds and improve generation Bartosh et al. (2024); Shi et al.. We build on this thread but in the discrete regime: we introduce a non-Markovian yet tractable parameterization of the forward marginals and posteriors, and train it end-to-end together with the standard factorized reverse model. This design preserves the usual variational training objective and the parallel, stepwise sampler, while adapting the training target to better match the model’s inductive biases.

## 6 CONCLUSION

We presented Forward-Learned Discrete Diffusion (FLDD), a framework that learns the forward (noising) process—via a non-Markovian factorization of forward marginals and tractable posteriors—while keeping the standard factorized reverse sampler unchanged and preserving the usual variational training objective. This alignment between a learned target and the fixed reverse family enables few-step generation without inference overhead.

Empirically, FLDD matches strong baselines when T=100 and maintains competitive quality even at T=10 across text (ROCStories) and molecule benchmarks (QM9, ZINC250k), and improves masked-diffusion generation on images by learning data-aware masking schedules—all with the same reverse parameterization. This yields a better quality–latency trade-off than conventional discrete diffusion.

Limitations and future work include exploring richer forward parameterizations, reducing reliance on REINFORCE via lower-variance estimators. We also note the extra parameters of the forward network (though not affecting reverse capacity) and leave broader design optimizations to future research.

## REFERENCES

Grigory Bartosh, Dmitry P Vetrov, and Christian Andersson Naesseth. Neural flow diffusion models: Learnable forward process for improved diffusion modelling. Advances in Neural Information Processing Systems, 37:73952–73985, 2024.

Sid Black, Leo Gao, Phil Wang, Connor Leahy, and Stella Biderman. Gpt-neo: Large scale autoregressive language modeling with mesh-tensorflow, march 2021. URL https://doi. org/10.5281/zenodo, 5297715(5):3, 2021.

Floor Eijkelboom, Grigory Bartosh, Christian Andersson Naesseth, Max Welling, and Jan-Willem van de Meent. Variational flow matching for graph generation. Advances in Neural Information Processing Systems, 37:11735–11764, 2024.

Shansan Gong, Mukai Li, Jiangtao Feng, Zhiyong Wu, and LingPeng Kong. Diffuseq: Sequence to sequence text generation with diffusion models. arXiv preprint arXiv:2210.08933, 2022.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Emiel Hoogeboom, Thomas Mensink, Jonathan Heek, Kay Lamerigts, Ruiqi Gao, and Tim Salimans. Simpler diffusion (sid2): 1.5 fid on imagenet512 with pixel-space diffusion. arXiv preprint arXiv:2410.19324, 2024.

John J Irwin, Teague Sterling, Michael M Mysinger, Erin S Bolstad, and Ryan G Coleman. Zinc: a free tool to discover chemistry for biology. Journal of chemical information and modeling, 52 (7):1757–1768, 2012.

Eric Jang, Shixiang Gu, and Ben Poole. Categorical reparameterization with gumbel-softmax. arXiv preprint arXiv:1611.01144, 2016.

Jaehyeong Jo, Seul Lee, and Sung Ju Hwang. Score-based Generative Modeling of Graphs via the System of Stochastic Differential Equations. In Proceedings ofthe 39th International Conference on Machine Learning, pp. 10362–10383. PMLR, June 2022.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=PqvMRDCJT9t.

Meng Liu, Keqiang Yan, Bora Oztekin, and Shuiwang Ji. GraphEBM: Molecular graph generation with energy-based models. In Energy Based Models Workshop - ICLR 2021, 2021. URL https: //openreview.net/forum?id=Gc51PtL\_zYw.

Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion modeling by estimating the ratios of the data distribution, 2024. URL https://arxiv. org/abs/2310.16834.

Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion modeling by estimating the ratios of the data distribution. arXiv preprint arXiv:2310.16834, 2023.

Justin Lovelace, Varsha Kishore, Chao Wan, Eliot Shekhtman, and Kilian Q Weinberger. Latent diffusion for language generation. Advances in Neural Information Processing Systems, 36:56998– 57025, 2023.

Youzhi Luo, Keqiang Yan, and Shuiwang Ji. Graphdf: A discrete flow model for molecular graph generation. In International conference on machine learning, pp. 7192–7203. PMLR, 2021.

Chris J Maddison, Andriy Mnih, and Yee Whye Teh. The concrete distribution: A continuous relaxation of discrete random variables. arXiv preprint arXiv:1611.00712, 2016.

Rabeeh Karimi Mahabadi, Hamish Ivison, Jaesung Tae, James Henderson, Iz Beltagy, Matthew E Peters, and Arman Cohan. Tess: Text-to-text self-conditioned simplex diffusion. arXiv preprint arXiv:2305.08379, 2023.