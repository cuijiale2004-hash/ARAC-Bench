## ABSTRACT

Diffusion models have demonstrated remarkable empirical success in recent years and are considered one of the state-of-the-art generative models in modern AI. These models consist of a forward process, which gradually diffuses the data distribution to a noise distribution spanning the whole space, and a backward process, which inverts this transformation to recover the data distribution from noise. Most of the existing literature assumes that the underlying space is Euclidean. However, in many practical applications, the data are constrained to lie on a submanifold of Euclidean space. Addressing this setting, De Bortoli et al. (2022) introduced Riemannian diffusion models and proved that using an exponentially small step size yields a small sampling error in the Wasserstein distance, provided the data distribution is smooth and strictly positive, and the score estimate is $L _ { \infty } { \mathrm { - a c c u r a t e } }$ . In this paper, we greatly strengthen this theory by establishing that, under $L _ { 2 } .$ -accurate score estimate, a polynomially small stepsize suffices to guarantee small sampling error in the total variation distance, without requiring smoothness or positivity of the data distribution. Our analysis only requires mild and standard curvature assumptions on the underlying manifold. The main ingredients in our analysis are Li-Yau estimate for the log-gradient of heat kernel, and Minakshisundaram-Pleijel parametrix expansion of the perturbed heat equation. Our approach opens the door to a sharper analysis of diffusion models on non-Euclidean spaces.

## 1 INTRODUCTION

Initially introduced by Sohl-Dickstein et al. (2015) and later advanced by Song & Ermon (2019); Ho et al. (2020); Dhariwal & Nichol (2021), diffusion model has become one of the bedrocks in generative modeling across a variety of application domains such as vision, video, speech, and many others. On a high level, diffusion models generate samples from a target distribution by operating on two stochastic processes:

1. A forward process

$$
X _ {0} \xrightarrow {\text { add   noise }} X _ {1} \xrightarrow {\text { add   noise }} \dots \xrightarrow {\text { add   noise }} X _ {T},
$$

where $X _ { 0 }$ is sampled from the target distribution $p _ { 0 }$ in $\mathbb { R } ^ { d }$ , and $X _ { T }$ resembles pure noise.

2. A reverse process

$$
Y _ {T} \xrightarrow {\text { d   e   n   o   i   s   e }} Y _ {T - 1} \xrightarrow {\text { d   e   n   o   i   s   e }} \dots \xrightarrow {\text { d   e   n   o   i   s   e }} Y _ {0},
$$

where $Y _ { T }$ starts from pure noise, and gradually removes the noise, so that at $Y _ { 0 } .$ , we recover a new sample from a distribution close to $p _ { 0 }$

The reverse process is built to recover the target data distribution by step-wise reversing the forward process, with a goal of matching the probabilities $Y _ { t } \approx X _ { t }$ in distribution for $t \in \{ T , \ldots , 1 \}$ . Leveraging the theory of backward stochastic differential equations (SDE) (Anderson, 1982; Haussmann & Pardoux, 1986), this can be formally achieved as soon as the as long as the score function, i.e., the log-gradient of the the marginal density of the forward process, becomes available, which can be estimated via score matching (Hyvarinen & Dayan¨ , 2005).

Tremendous recent progresses have been made in understanding the convergence of diffusion models in the Euclidean space, e.g. Lee et al. (2023); Chen et al. (2023); Benton et al. (2024); Li et al. (2024), which establish near-tight polynomial iteration complexities of discrete-time samplers under $L _ { 2 }$ -accurate score estimates and mild assumptions of the data distribution. These convergence results provide strong justifications to the empirical success of diffusion models for generating from complex multi-modal distributions.

Many scientific domains, however, are intrinsically non-Euclidean; examples include orientations on $\mathrm { S O ( 3 ) }$ , directions on spheres, toroidal angles, articulated poses, and symmetric positive definite (SPD) matrices are naturally modeled on Riemannian manifolds (Piggott & Solo, 2016; Muniz et al., 2022). Recently, there has been an increasing interest in effectively sampling from distributions supported on manifolds and providing theoretical guarantees (Girolami & Calderhead, 2011; Gatmiry & Vempala, 2022; Li & Erdogdu, 2023; Guan et al., 2025). Although sampling on manifolds has been studied extensively (Cheng et al., 2023), extending diffusion models to manifolds requires careful treatments to incorporate the manifold constraints into both the time-inhomogeneous forward and reverse processes, with selected attempts in De Bortoli et al. (2022); Huang et al. (2022); Lou et al. (2023); Liu et al. (2023); Fishman et al. (2023).

One notable development is De Bortoli et al. (2022), who introduced Riemannian Score-Based Generative Models (RSGMs) with convergence guarantees in the Wasserstein distance. Specifically, they established a time-reversal diffusion process for geometric Brownian motion on manifolds, which can be similarly learned via score matching (Hyvarinen & Dayan¨ , 2005). While groundbreaking, their convergence bound suffers from a few caveats: 1) it requires an exponentially small stepsize, leading to a possibly exponential iteration complexity in some of the manifold parameters; 2) it requires $L _ { \infty }$ -accurate score estimates, which are impractical in deep learning; and 3) the data distribution is required to be smooth and strictly positive on compact manifolds. This naturally raises the following questions:

Can we achieve polynomial iteration complexity for manifold diffusion models using $L _ { 2 }$ -accurate score estimates under mild data assumptions?

## 1.1 OUR CONTRIBUTION

We provide a discrete-time analysis of the RSGM sampler in De Bortoli et al. (2022), assuming L -accurate score estimates. Under mild geometric conditions of the manifold without assuming smooth or strictly positive data densities, we establish that polynomial stepsizes suffice for accurate sampling on manifolds in total variation (TV). More precisely, for some $\varepsilon > 0$ , the TV error between the distribution of the output $Y _ { 0 }$ and $p _ { \delta }$ , that is, an approximation to $p _ { 0 }$ with early-stopping time $\delta > 0$ , obeys

$$
\operatorname{TV} \left(p _ {\delta}, \operatorname{Law} \left(Y _ {0}\right)\right) \lesssim \varepsilon + \varepsilon_ {\text { score }} + \sqrt {h T} \operatorname{poly} \left(d, \delta^ {- 1}\right),
$$

as long as the horizon satisfies $T \gtrsim \lambda _ { 1 } ^ { - 1 } ( d \log d + \log ( d / \varepsilon ) )$ . Here, d is the dimension of $\mathcal { M } .$ $\lambda _ { 1 }$ is the spectral gap of $- \Delta _ { M } , h > 0$ is the stepsize, and $\varepsilon _ { \mathsf { s c o r e } }$ is the $L _ { 2 }$ score estimation error.<sup>1</sup> This bound suggests that a polynomial stepsize is sufficient: take $T \asymp \lambda ^ { - 1 } ( d \log d + \log ( d / \varepsilon ) )$ ) and $\begin{array} { r } { h = \frac { \varepsilon ^ { 2 } } { \mathrm { p o l y } ( d , \delta ^ { - 1 } ) T } } \end{array}$ , then the TV error is bounded by $\varepsilon + \varepsilon _ { \mathsf { s c o r e } }$ after an iteration complexity of

$$
N = T / h \asymp \mathrm{poly} (d, \delta^ {- 1}) / (\lambda_ {1} \varepsilon) ^ {2}.
$$

This conveys a much more benign message about the efficiency of Riemannian diffusion models, compared with the iteration complexity in De Bortoli et al. (2022) that scales exponentially with the dimension $d ,$ under relaxed assumptions on both the data distribution and the score estimates.

Techniques. Our proof highlights three ingredients: (i) high-probability Li-Yau gradient bounds for the manifold heat kernel together with early stopping to control $\| \nabla \log p _ { t } \|$ without assuming positivity/smoothness of $p _ { 0 } ; ( \mathrm { i i } )$ a localization scheme that “freezes” drifts across nearby tangent spaces but preserves continuous Brownian motion (BM), to separate the effects of discretizing scores and BM; and (iii) a quantitative estimates for Minakshisundaram–Pleijel parametrix that controls one-step deviations between the manifold heat flow and its discretized proxy. These components allow us to handle the discretization errors sharply to avoid exponential dependence.

<table><tr><td>Work</td><td>Structure</td><td>Metric</td><td>Iteration complexity</td><td>Data distribution</td></tr><tr><td>Benton et al. (2024)</td><td>Euclidean</td><td>TV</td><td> $\widetilde{O}(d/\varepsilon^{2})$ </td><td>bounded moment</td></tr><tr><td>Li et al. (2024)</td><td>Euclidean</td><td>TV</td><td> $\widetilde{O}(\text{poly}(d)/\varepsilon)$ </td><td>bounded support</td></tr><tr><td>Li &amp; Yan (2025)</td><td>Euclidean</td><td>TV</td><td> $\widetilde{O}(d/\varepsilon)$ </td><td>bounded moment</td></tr><tr><td>De Bortoli et al. (2022)</td><td>Manifold</td><td> $W_{p}$ </td><td> $\widetilde{O}\left(\exp(O(d))/\varepsilon^{-1/\lambda_{1}}\right)^{a}$ </td><td>smooth, strictly positive</td></tr><tr><td>This work</td><td>Manifold</td><td>TV</td><td> $\widetilde{O}\left(\frac{\text{poly}(d)}{\lambda_{1}^{2}\varepsilon^{2}}\right)$ </td><td>None (early stopping)</td></tr></table>

<sup>a</sup>The original $W _ { p }$ error in De Bortoli et al. (2022) is stated in the form of $\widetilde { O } ( C \mathrm { e } ^ { - \lambda _ { 1 } T } + \mathrm { e } ^ { T } \sqrt { h } )$ , where C is defined in Proposition C.6 therein, which in turn is specified by Urakawa (2006, Proposition 2.6) as the supremum of $t ^ { \hat { d } / 2 } H ( t , x , y )$ , where H is the heat kernel on the manifold. In general, the best estimate for this is due to Li-Yau (Li & Yau, 1986), which gives $C \leq \mathrm { e } ^ { O ( d ) }$ . To achieve ε-error, we must set $T =$ $\lambda _ { 1 } ^ { - 1 } ( \Omega ( d ) + \log \varepsilon ^ { - 1 } )$ and $h = \mathrm { e } ^ { - 2 \dot { T } } \varepsilon ^ { 2 }$ , then the iteration complexity T/h has the claimed form.  
Table 1: Comparison of the current theoretical guarantees on diffusion probabilistic models on Euclidean spaces and manifolds. Here, $\lambda _ { 1 } > 0$ is the spectral gap of the Laplace-Beltrami operator.

## 1.2 RELATED WORKS

Non-asymptotic convergence for Euclidean diffusion models. Early convergence analyses of diffusion models require $L _ { \infty }$ -accurate score estimates (De Bortoli et al., 2021). For stochastic samplers such as DDPM (Ho et al., 2020), early bounds under Lipschitz/smoothness assumptions of the data distribution admit an ${ \cal O } ( T ^ { - \frac { 1 } { 2 } } )$ iteration complexity in the total variation distance assuming L -accurate score estimates (Chen et al., 2023), with subsequent analyses relaxing the Lipschitz assumption yet retaining the same complexity (Lee et al., 2023; Benton et al., 2024; Li et al., 2024). More recently, Li & Yan (2025) has improved the iteration complexity to $\widetilde O ( T ^ { - 1 } )$ . For deterministic samplers, Chen et al. (2023) established polynomial convergence with exact scores, and Li et al. (2024) established a convergence rate of $\dot { O ( } T ^ { - 1 } )$ under $L _ { 2 } .$ -accurate scores. See Beyler & Bach (2025); Liang et al. (2024); Li & Jiao (2024) for additional analyses that established convergence in the Wasserstein distance and improved discrete-time rates. Several works (Li & Yan, 2024; Liang et al., 2025; Huang et al., 2024; Potaptchik et al., 2024) also developed non-asymptotic convergence rates of diffusion models under the manifold hypothesis, suggesting diffusion models are adaptive to low-dimensional structures. This line of work should not be confused with ours, where the diffusion process is designed specifically to be constrained on the manifold.

Sampling on Riemannian manifold. Cheng et al. (2022; 2023) analyzed the geometric Euler–Maruyama (EM) discretization for time-homogeneous SDEs, and proved a polynomial complexity guarantee under dissipative-distant geometric assumptions on the manifold. See also Bharath et al. (2025) for follow-ups. Guan et al. (2025) proposed a Riemannian proximal sampler with convergence guarantees under the log-Sobolev inequality. Various sampling algorithms are also studied for a related problem known as sampling from constrained spaces (Srinivasan et al., 2024; Ahn & Chewi, 2021). Nonetheless, convergence analyses of Riemannian diffusion models under general data distributions remain highly limited, with De Bortoli et al. (2022) being the only prior work with non-asymptotic convergence rates.

## 2 BACKGROUNDS

## 2.1 DIFFUSION MODELS ON EUCLIDEAN SPACE

We briefly recall diffusion processes on $\mathbb { R } ^ { d }$ . Let $( W _ { t } ) _ { t \geq 0 }$ be a standard Brownian motion in $\mathbb { R } ^ { d }$

Forward SDE and Fokker–Planck. Given a drift term $b _ { t } : \mathbb { R } ^ { d }  \mathbb { R } ^ { d }$ , the forward process $( X _ { t } ) _ { t \in [ 0 , T ] }$ solves the Ito SDEˆ

$$
\mathrm{d} X _ {t} = b _ {t} (X _ {t}) \mathrm{d} t + \mathrm{d} W _ {t}, \quad X _ {0} \sim p _ {0}.
$$

Let $p _ { t }$ denote the law of $X _ { t } ,$ , then $p _ { t }$ satisfies the Fokker–Planck equation $\begin{array} { r } { \partial _ { t } p _ { t } = - \nabla ( b _ { t } p _ { t } ) + \frac { 1 } { 2 } \Delta p _ { t } } \end{array}$ In the driftless setting where $b _ { t } \equiv 0$ , the marginal $p _ { t } = p _ { 0 } * \varphi _ { t }$ is a Gaussian smoothing of $p _ { 0 }$ with kernel $\varphi _ { t } ( z ) = ( 2 \pi t \bar { ) } ^ { - d / 2 } \exp \big ( - \| z \| ^ { 2 } / ( 2 t \bar { ) } \big )$

Score and reverse process. The score of the forward process at time t is defined as $s _ { t } ( x ) : =$ $\nabla \log p _ { t } ( x )$ . The time-reversal identity (Anderson, 1982) yields a reverse-time process $( Y _ { t } ) _ { t \in [ 0 , T ] }$ whose marginals match those of $( X _ { t } ) _ { t \in [ 0 , T ] }$ which solves the reverse-time SDE (note that t flows from T to 0):

$$
d Y _ {\tau} = \left[ - b _ {\tau} (Y _ {t}) + s _ {\tau} (Y _ {t}) \right] \mathrm{d} t + \mathrm{d} W _ {t}, \quad \tau = T - t, \quad Y _ {T} \sim p _ {X _ {T}}.
$$

Discretization with approximate score. Let $0 = t _ { 0 } < t _ { 1 } < \cdot \cdot \cdot < t _ { N } = T ,$ , where $t _ { i } - t _ { i - 1 } = : h$ be a time grid. In practice, exact score function is often unavailable. Instead, we use an approximation $\widehat { s } _ { t }$ trained via score matching (Hyvarinen & Dayan¨ , 2005). The Euler–Maruyama discretization of the reverse-time SDE above is given by

$$
y _ {k - 1} = y _ {k} + h \left[ - b _ {t _ {k}} (y _ {k}) + \widehat {s} _ {t _ {k}} (y _ {k}) \right] + \sqrt {h} g _ {k}, \quad g _ {k} \sim \mathcal {N} (0, I _ {d}).
$$

In our driftless setting, this reduces to

$$
y _ {k - 1} = y _ {k} + h \widehat {s} _ {t _ {k}} (y _ {k}) + \sqrt {h} g _ {k}, \quad g _ {k} \sim \mathcal {N} (0, I _ {d}).
$$

## 2.2 GEOMETRY AND NOTATION

We assume some familiarity with Riemannian geometry, and make use of standard notation. Please refer to Jost (2017); Petersen (2006) for a more in-depth treatment. In particular, we use $\alpha , \beta , \xi , \zeta ,$ etc., to index coordinate representation of tensors, and assume Einstein’s summation convention. Let $( \mathcal { M } , g )$ be a connected, compact d-dimensional Riemannian manifold, with geodesic distance $\rho ( \cdot , \cdot )$ and volume measure $\mu .$ . We assume $\mu ( \mathcal { M } ) = 1$ . The Levi–Civita connection is denoted by $\nabla .$ , and the Laplace–Beltrami operator by

$$
\Delta_ {\mathcal {M}} f := \nabla_ {\alpha} \nabla^ {\alpha} f.
$$

We use $T _ { x } { \mathcal { M } }$ for the tangent space at $x$ and use ex $\mathrm { p } _ { x } : T _ { x } \mathcal { M } \to \mathcal { M }$ for the exponential map and $\log _ { x }$ for its local inverse on the normal neighborhood of x. Furthermore, we define geodesic diameter of $( \mathcal { M } , g )$ is

$$
\operatorname{Diam} (\mathcal {M}) := \sup _ {x, y \in \mathcal {M}} \rho (x, y),
$$

where $\rho ( \cdot , \cdot )$ is the geodesic distance induced by $g .$ We further denote Rm as the Riemannian curvature tensor. Geodesic ball centered at x with radius r is denoted $B _ { x } ( r )$

We use the total variation (TV) and the Kullback-Leibler (KL) distance to measure the discrepancy between two distributions $p , q \colon$

$$
\mathsf {T V} (p, q) = \int_ {\mathcal {M}} \left| \mathrm{d} p - \mathrm{d} q \right|, \quad \mathsf {K L} (p \| q) = \int_ {\mathcal {M}} \left(\log \frac {\mathrm{d} p}{\mathrm{d} q}\right) \mathrm{d} p.
$$

## 2.3 HEAT FLOW, BROWNIAN MOTION, AND DIFFUSION ON $\mathcal { M }$

We also recall the setup for SDE and diffusion processes on Riemannian manifolds introduced in De Bortoli et al. (2022); Cheng et al. (2023). Let $( W _ { t } ) _ { t \geq 0 }$ be a standard Brownian motion in $\mathbb { R } ^ { d }$ and $U _ { x } : \mathbb { R } ^ { d }  T _ { x } \mathcal { M }$ any orthonormal frame at $x .$ . The Geometric Brownian motion solves

$$
\mathrm{d} X _ {t} = U _ {X _ {t}} \circ \mathrm{d} W _ {t},
$$

where ◦ denotes Stratonovich integral, and its transition density $p _ { t } ( x , y )$ with respect to $\mu$ solves the heat equation

$$
\partial_ {t} p _ {t} (\cdot , y) = \frac {1}{2} \Delta_ {\mathcal {M}} p _ {t} (\cdot , y).
$$

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 Riemannian Score-Based Generative Models (RSGM)

1: Manifold  $(\mathcal{M}, g)$ ; score  $\widehat{s}_{t}(x)$ ; early stopping time  $\delta &gt; 0$ ; reverse time grid  $\delta = t_{0} &lt; t_{1} &lt; \cdots &lt; t_{N} = T$ ; step size  $h = t_{k} - t_{k-1}$ ; initial  $x_{N} \sim \mu$  (uniform distribution);

2: for  $k \in \{N, \ldots, 1, 0\}$  do

3: Choose an orthonormal frame  $U_{k}$  at  $Y_{k}$ , which is a linear map from  $R^{d}$  to  $T_{Y_{k}} M$ .

4:  $\xi_{k} \sim \mathcal{N}(0, I_{d})$  in  $R^{d}$ ;  $G_{k} \leftarrow U_{k} \xi_{k} \in T_{Y_{k}} M$ .

5:  $b_{k} \leftarrow \widehat{s}_{t_{k}}(Y_{k}) \in T_{Y_{k}} M$ 

6:  $\Delta_{k} \leftarrow hb_{k} + \sqrt{h} G_{k} \in T_{Y_{k}} M$ 

7: if  $\| \Delta_{k} \| \leq h^{1/4}$  then

8:  $Y_{k-1} \leftarrow \exp_{Y_{k}}(\Delta_{k})$ 

9: else

10:  $Y_{k-1} \sim \mu$ 

11: return  $Y_{0}$
</div>

Equivalently, Brownian motion can be defined abstractly as the solution to the martingale problem for the operator $\scriptstyle { \frac { 1 } { 2 } } \Delta _ { { \mathcal { M } } }$ . Concretely, for any $f \in C ^ { \infty } ( [ 0 , \mathbf { \bar { \infty } } ) \times \mathcal { M } )$ , the process

$$
M _ {t} ^ {f} := f (t, X _ {t}) - f (0, X _ {0}) - \int_ {0} ^ {t} \left(\partial_ {s} + \frac {1}{2} \Delta_ {\mathcal {M}}\right) f (s, X _ {s}) \mathrm{d} s
$$

is a martingale with respect to the natural filtration of X. More generally, a forward diffusion process with drift is given by

$$
\mathrm{d} X _ {t} = b _ {t} (X _ {t}) \mathrm{d} t + U _ {X _ {t}} \circ \mathrm{d} W _ {t},
$$

with Fokker–Planck equation $\begin{array} { r } { \partial _ { t } p _ { t } = - \nabla ( b _ { t } p _ { t } ) + \frac { 1 } { 2 } \Delta _ { \mathcal { M } } p _ { t } } \end{array}$ . Note that in this setting, the following process is a martingale for smooth $f \colon$

$$
M _ {t} ^ {f} := f (t, X _ {t}) - f (0, X _ {0}) - \int_ {0} ^ {t} \left(\partial_ {s} f + \langle b _ {t}, \nabla f \rangle + \frac {1}{2} \Delta_ {\mathcal {M}} f\right) (s, X _ {s}) \mathrm{d} s.\tag{1}
$$

Let $p _ { t }$ denote the density of $X _ { t }$ w.r.t. $\mu ,$ and define the score $s _ { t } : = \nabla$ log $p _ { t }$ . The time-reversal identity on manifolds yields a reverse SDE:

$$
\mathrm{d} \widetilde {X} _ {\tau} = (- b _ {t} (\widetilde {X} _ {\tau}) + \nabla \log p _ {\tau} (\widetilde {X} _ {\tau})) \mathrm{d} t + U _ {\widetilde {X} _ {\tau}} \circ \mathrm{d} W _ {t}, \quad \tau = T - t, \quad \tilde {X} _ {T} \sim p _ {X _ {T}}.
$$

In practice, the score $\nabla \log p _ { t }$ is approximated by a trained neural network $\widehat { s } _ { t } ( x )$

Last but not least, note that on compact manifold $\mathfrak { s } , - \Delta _ { \mathcal { M } }$ admits a spectral gap $\lambda _ { 1 } > 0$ . Any initial distribution mixes to the uniform distribution $\mu$ along the heat flow with rate $\mathrm { ~ \ i ~ e ^ { - \lambda _ { 1 } } } t$

## 3 MAIN RESULT

In this section, for completeness, we first introduce the RSGM algorithm in De Bortoli et al. (2022). Then, we offer our polynomial convergence guarantee in Theorem 1. For simplicity, we use a driftless forward process:

$$
\mathrm{d} X _ {t} = U _ {X _ {t}} \circ \mathrm{d} W _ {t}, \quad X _ {0} \sim p _ {0}.
$$

The time-reversal identity yields the reverse-time SDE

$$
\mathrm{d} Y _ {t} = \nabla \log p _ {\tau} (Y _ {\tau}) \mathrm{d} t + U _ {Y _ {\tau}} \circ \mathrm{d} W _ {t}, \quad \tau = T - t, \quad Y _ {T} \sim p _ {T}.\tag{2}
$$

In Algorithm 1, we provide an outline of discretized reverse-time SDE on Riemannian manifold, modified from De Bortoli et al. (2022). In each reverse step $k \in \{ N , \ldots , 1 , 0 \}$ , we select an orthonormal frame $U _ { k }$ at $y _ { k }$ , then sample Gaussian noise $\xi _ { k }$ and lift it to the tangent space $T _ { y _ { k } } { \mathcal { M } }$ using the orthonormal frame, obtaining $G _ { k } \in T _ { y _ { k } } { \mathcal { M } }$ . Afterwards, we propose a tangent update $\Delta _ { k } = h \widehat { s } _ { t _ { k } } ( y _ { k } ) + \sqrt { h } G _ { k }$ and the project to the manifold using the exponential map. To prevent the update from exitting the injective radius, we perform a rejection sampling step that rejects exceedingly large update. The algorithm terminates at $k = 0$ and returns the final iterate $y _ { 0 }$ . In this way, we ensure every update is well-defined in normal coordinates during the algorithm.

Before presenting the main theorem, we formalize the assumptions needed for the convergence guarantee.

Assumption 1 (Regularity). Let $( \mathcal { M } , g )$ be a connected, compact d-dimensional Riemannian manifold. We assume thefollowing conditions on $\mathcal { M } .$

(A1) Positive injectivity radius: there exists some constant $K \geq 1$ such that the injective radius $\geq 1 / K$

(A2) Uniform curvature bounds: for the same constant K (which can be enlarged if necessary), we have

$$
\max \left\{\operatorname{Diam} (\mathcal {M}), \| \mathrm{Rm} \| _ {L ^ {\infty}}, \| \nabla \mathrm{Rm} \| _ {L ^ {\infty}}, \| \nabla^ {2} \mathrm{Rm} \| _ {L ^ {\infty}} \right\} \leq K.
$$

(A3) Regularity of score estimates: there exists a polynomial $\mathrm { p o l y } ( d , K )$ , such that

$$
\| \widehat {s} _ {t _ {k}} (x) \| \leq \operatorname{poly} (d, K) \left(\| \nabla \log p _ {t _ {k}} (x) \| + t _ {k} ^ {- 1}\right), \quad \forall x \in \mathcal {M}.
$$

In Assumption 1, we made the standard “bounded geometry” assumption; similar assumptions also occur in Cheng et al. (2022); De Bortoli et al. (2022). A positive injective radius ensures that we have sufficient room to operate on the tangent spaces as a proxy of operating on manifolds, since for every $x \in \mathcal { M }$ , the exponential map $\mathrm { e x p } _ { x }$ is a diffeomorphism on the geodesic ball within injective radius. Bounds on Riemannian tensors rule out pathological cases, which helps to control the error propagation along the reverse diffusion. Lastly, compactness ensures a positive spectral gap of $\Delta _ { { \scriptscriptstyle M } }$ with $\bar { \lambda _ { 1 } } > 0$ , which is necessary to guarantee that the forward process mixes. The mild assumption (A3) on the score estimates avoids excessively large drifts in diffusion, and can be implemented easily in practice by clipping. In addition to the above, we also need a standard assumption on the score estimation error (Chen et al., 2023).

Assumption 2 (Score estimation error). There exists $\varepsilon _ { \mathsf { s c o r e } } > 0$ such that

$$
\sum_ {k = 1} ^ {N} (t _ {k} - t _ {k - 1}) \mathbb {E} \| \widehat {s} _ {t _ {k}} (Y _ {t _ {k}}) - \nabla \log p _ {t _ {k}} (Y _ {t _ {k}}) \| ^ {2} \leq \varepsilon_ {\mathrm{score}} ^ {2}.
$$

With the above assumptions, we are now ready to present our main convergence guarantee for RSGM, as outlined in the following TV-accuracy bound.

Theorem 1. Assume Assumptions 1 and 2 hold. There exists some universal constant $C , C ^ { \prime } > 0$ such that thefollowing holds. $\begin{array} { r } { I f T \geq \frac { C } { \lambda _ { 1 } } ( d \log ( K d ) + K { + } \log ( \frac { N } { \varepsilon } ) ) } \end{array}$ , then the output Y<sub>0</sub> ofAlgorithm 1 obeys

$$
\mathsf {T V} \left(p _ {\delta}, \operatorname{Law} \left(Y _ {0}\right)\right) \leq \varepsilon + C ^ {\prime} \varepsilon_ {\text { score }} + \sqrt {h T} \operatorname{poly} (d, K, \delta^ {- 1}),
$$

where h is the discretization step size, $\lambda _ { 1 } > 0$ is the mixing rate ofthe geometric Brownian Motion on $\mathcal { M } , i . e .$ , the smallest eigenvalue o $f - \Delta _ { \mathcal { M } }$ in $L ^ { 2 } ( \mu )$ ).

A few remarks are in order.

Iteration complexity. The error bound decomposes cleanly into three terms: ε results from mixing of the heat semigroup at the spectral gap $\lambda _ { 1 } , \varepsilon _ { \mathsf { s c o r e } }$ captures error from imperfect score estimation, and $\sqrt { h T } \operatorname { p o l y } ( d , K , \delta ^ { - 1 } )$ is the discretization error controlled by the step size and curvature. Consequently, choosing $T \asymp \lambda _ { 1 } ^ { - 1 } ( d \log d + \log ( d / \varepsilon ) )$ and $\begin{array} { r } { h = \frac { \varepsilon ^ { 2 } } { \mathrm { p o l y } ( d , K , \delta ^ { - 1 } ) T } } \end{array}$ , then the TV error is bounded by $\varepsilon + \varepsilon _ { \mathsf { s c o r e } }$ after polynomially many iterations

$$
N = T / h \asymp \frac {\operatorname{poly} (d , K , \delta^ {- 1})}{(\lambda_ {1} \varepsilon) ^ {2}}.
$$

Compared to prior convergence rates in the Wasserstein metric (De Bortoli et al., 2022), which require exponential complexity, we achieve polynomial convergence of Riemannian diffusion models for the first time. Nonetheless, we emphasize that TV and Wasserstein distances are incomparable with each other in general, and our guarantee complements prior Wasserstein results (De Bortoli et al., 2022) by ensuring distributional closeness in a different notion with a much smaller number of iterations.

Possible improvements. We note that the bound established in Theorem 1 holds under very mild geometric assumptions, requiring only constraints on the injective radius and Riemannian curvature. The purpose of this study is to demonstrate that, in the manifold setting, the exponential blow-up in $T$ can be avoided and polynomial complexity can be achieved. To keep the exposition as simple as possible and to clearly highlight the key ideas, we have not attempted to optimize the current bound on the degree of the polynomial. Potential approaches for sharper bounds include: (i) a better design of discretization schedule, possibly adaptive to the manifold geoemetry, and a more careful computation of discretization error, such as those in Li & Jiao (2024), Benton et al. (2024) (notably, the dependence on $\delta$ might be improved to poly-logarithmic in this way); (ii) a tailored analysis for TV error that does not rely on Pinsker’s inequality, like those in Li & Yan (2025), may also be extended to manifolds; (iii) a tighter version of our Minakshisundaram-Pleijel parametrix bound. We leave these improvements as future work.

## 4 PROOF OUTLINE

Throughout the proof, we assume that

$$
h \leq \frac {1}{\operatorname{poly} (d , K , \delta^ {- 1})},\tag{3}
$$

since otherwise the bound in Theorem 1 would be trivial (recall that TV distance is always bounded by 2). We start by recalling the sequence considered in RSGM. Let $( Y _ { k } ) _ { k \in \{ 0 , \ldots , N \} }$ be given by $Y _ { N } \sim \mu$ and for any $k \in \{ 0 , \ldots , N - 1 \}$

$$
Y _ {k - 1} = \left\{ \begin{array}{l l} \exp_ {Y _ {k}} \left[ h \widehat {s} _ {t _ {k}} (Y _ {k}) + \sqrt {h} G _ {k} \right], & \| h \widehat {s} _ {t _ {k}} (Y _ {k}) + \sqrt {h}   G _ {k} \| \leq h ^ {1 / 4}, \\ \text { drawn   from } \mu , & \text { otherwise }. \end{array} \right.
$$

This defines a sequence of probability transition kernels $\widehat { \sf K } _ { t _ { k } , t _ { k - 1 } }$ . For simplicity, we denote this by $\widehat { \sf K } _ { k }$ . Let $q _ { k }$ be the law of $Y _ { k }$ . We have

$$
q _ {0} = q _ {N} \widehat {\mathsf {K}} _ {N} \widehat {\mathsf {K}} _ {N - 1} \dots \widehat {\mathsf {K}} _ {1}.
$$

Similarly, the probability transition kernel from time $t _ { k }$ to $t _ { k - 1 }$ in (2) is denoted by $\mathsf { K } _ { t _ { k } , t _ { k - 1 } }$ or $\mathsf { K } _ { k }$ in short. We have

$$
p _ {0} = p _ {N} \mathsf {K} _ {N} \mathsf {K} _ {N - 1} \dots \mathsf {K} _ {1}.
$$

Our goal would be to bound ${ \sf T V } ( p _ { 0 } , q _ { 0 } )$ as in Theorem 1, by decomposing the total error into four components:

$$
\text {(initialization error)} + \text {(score error)} + \text {(drift discretization error)} + \text {(BM simulation error)}.
$$

More concretely:

• Initialization error arises from initializing $Y _ { N }$ with $\mu$ instead of the true marginal $p _ { N }$ ;

• Score error arises from imperfect score estimation;

• Drift discretization error arises from approximating the continuous-time drift $\widehat { s } _ { t } ( Y _ { t } )$ by its “time-frozen” counterpart $\widehat { s } _ { t _ { k } } ( Y _ { t _ { k } } )$ ;

• Brownian motion (BM) simulation error is a distinctive feature of the manifold setting. Unlike in Euclidean space — where the transition kernel of Brownian motion over $[ t _ { k } , t _ { k - 1 } ]$ is exactly Gaussian with variance $\left( t _ { k } - t _ { k - 1 } \right)$ — the transition kernel of manifoldvalued Brownian motion cannot be simulated exactly by any discrete-time process, even after time discretization. This inherent inexactness gives rise to this final error term.

The first two components are relatively easier to bound using well-established tools: mixing rate bounds of heat flow (Urakawa, 2006) and Girsanov transform (Chen et al., 2023). For the drift discretization error, recent techniques developed in the Euclidean setting (Benton et al., 2024) can also be adapted with modifications that account for the manifold curvature. However, the last component — the Brownian motion simulation error — represents the core challenge in the manifold setting, which fundamentally denies a direct extension of Euclidean analysis.

Step I. Constructing auxiliary kernels via localization. In view of this, we first introduce an intermediate random process that separates the drift discretization error from the BM simulation error. Constructing such a process, however, involves additional technicality. In particular, the frozen drift $\widehat { s } _ { t _ { k } } ( Y _ { t _ { k } } )$ is a vector in the tangent space $T _ { Y _ { k } } { \mathcal { M } }$ , and is therefore only well-defined at the fixed point $Y _ { k }$ . This poses a compatibility issue: as Brownian motion evolves continuously on the manifold, it immediately departs from $Y _ { k } ,$ rendering the frozen drift ill-defined. Careful geometric considerations are thus required to reconcile the piecewise-constant drift approximation with the intrinsic curvature of the manifold.

In our analysis, this is handled using localization by the construction of an auxiliary sequence of transition kernels $\mathsf { K } _ { k } ^ { \mathsf { a u x } }$ . These kernels do not appear in the algorithm itself; they serve solely as an analytical tool to facilitate the proof. These kernels expose the behavior of the time-reverse SDE (2) when the estimated score $\widehat { s } _ { t }$ is frozen to be a constant vector field in between discretization steps, meanwhile keeping the continuous Brownian motion.

Let $\eta : [ 0 , \infty )  [ 0 , 1 ]$ be a smooth cutoff function, i.e., η is decreasing, $\eta | _ { [ 0 , 1 ] } \equiv 1$ and $\eta | _ { [ 4 , \infty ) } \equiv 0$ Such a function can be chosen such that $| \eta ^ { \prime } | + | \eta ^ { \prime \prime } | + | \eta ^ { \prime \prime \prime } | \leq 1 0 0$ . Recall that the injective radius of M is lower bounded by $1 / K$ , and the curvature is upper bounded by K. Define

$$
\omega := \frac {c _ {\omega}}{K d ^ {4}}, \qquad \eta_ {\omega} (r) = \eta \left(\frac {4 r ^ {2}}{\omega^ {2}}\right), \quad r \geq 0,\tag{4}
$$

where $c _ { \omega } > 0$ is a small universal constant. We have $\eta _ { \omega } | _ { [ 0 , \frac { \omega } { 2 } ] } \equiv 1$ and $\eta _ { \omega } | _ { [ \omega , \infty ) } \equiv 0$ . For $t > 0$ 2 $x , y \in { \mathcal { M } }$ , define the following vector field on M:

$$
\mathcal {S} _ {t, x} (y) = (\mathrm{dexp} _ {x}) _ {\log_ {x} y} \left(\eta_ {\omega} (\rho (x, y)) \cdot \widehat {s} _ {t} (x)\right) \in T _ {y} \mathcal {M}.
$$

Intuitively speaking, $\mathcal { S } _ { t , x } ( \cdot )$ is the “constant” velocity field $\widehat { s } _ { t } ( x )$ in normal coordinates, which represents our idea of freezing the drift term for a time period. The d $\mathrm { e x p } _ { x }$ in the formula is responsible for identifying $T _ { y } \mathcal { M }$ with $\mathit { \check { T } } _ { x } \mathcal { M } . \mathit { \Omega } ^ { 2 }$ On the other hand, the cut-off function $\eta _ { \omega }$ is necessary to keep all our discussions restricted to the injective radius, so as to avoid pathologies of cut locus.

With this in mind, we are ready to define $\mathsf { K } _ { k } ^ { \mathsf { a u x } }$ as the transition kernel from time $t _ { k }$ to $t _ { k - 1 }$ of the reverse-time SDE

$$
\mathrm{d} Y _ {\tau} = \mathscr {S} _ {t _ {k}, Y _ {t _ {k}}} (Y _ {\tau}) \mathrm{d} t + U _ {Y _ {\tau}} \circ \mathrm{d} W _ {t}, \quad \tau = T - t, \quad \tau \in [ t _ {k - 1}, t _ {k} ],\tag{5}
$$

and in addition,

$$
p _ {k} ^ {\mathsf {a u x}} = p _ {N} \mathsf {K} _ {N} ^ {\mathsf {a u x}} \mathsf {K} _ {N - 1} ^ {\mathsf {a u x}} \dots \mathsf {K} _ {k + 1} ^ {\mathsf {a u x}}, \quad k = N, N - 1, \dots , 0.
$$

Step II. Decomposing different sources of error. We now decompose

$$
\mathsf {T V} \left(p _ {0}, q _ {0}\right) \leq \mathsf {T V} \left(p _ {0}, p _ {0} ^ {\text {aux}}\right) + \mathsf {T V} \left(p _ {0} ^ {\text {aux}}, q _ {0}\right) \leq \sqrt {2 \mathrm{KL} \left(p _ {0} \| p _ {0} ^ {\text {aux}}\right)} + \mathsf {T V} \left(p _ {0} ^ {\text {aux}}, q _ {0}\right),
$$

where the last inequality used Pinsker’s inequality. To control ${ \mathsf { K L } } ( p _ { 0 } \parallel p _ { 0 } ^ { \mathsf { a u x } } )$ , we further introduce the counterpart of $\mathcal { S } _ { t , x }$ using the exact score function $\nabla \log p _ { t } \mathbf { : }$

$$
\mathcal {S} _ {t, x} ^ {\star} (y) = (\mathrm{d} \exp_ {x}) _ {\log_ {x} y} \left(\eta_ {\omega} (\rho (x, y)) \cdot \nabla \log p _ {t} (x)\right) \in T _ {y} \mathcal {M}.
$$

We apply Girsanov’s theorem (Hsu, 2002) to compare (5) with (2), in a way that is standard in recent literature (Chen et al., 2023; De Bortoli et al., 2022). Denote the path law of the solution of (2) by $\mathsf { L a w } ( Y )$ , and the path law of the solution of (5) by $\mathsf { L a w } ( Y ^ { \mathsf { a u x } } )$ . Girsanov’s theorem asserts that the KL divergence $\mathsf { K L } ( \mathsf { L a w } ( Y ) \parallel \mathsf { L a w } ( Y ^ { \mathsf { a u x } } ) )$ is upper bounded by the expectation of the squared norm of the difference between the drift terms in the two $\mathrm { S D E s } . ^ { 3 }$ More concretely,

$$
\mathrm{KL} \big (\mathsf {L a w} (Y) \| \mathsf {L a w} (Y ^ {\mathsf {a u x}}) \big) \leq \sum_ {k = 1} ^ {N} \int_ {t _ {k - 1}} ^ {t _ {k}} \mathbb {E} \left\| \nabla \log p _ {t} (Y _ {t}) - \mathscr {S} _ {t _ {k}, Y _ {t _ {k}}} (Y _ {t}) \right\| ^ {2} d t.
$$

Since $p _ { 0 }$ and $p _ { \mathrm { 0 } } ^ { \mathsf { a u x } }$ are marginals of $\mathsf { L a w } ( Y )$ and $\mathsf { L a w } ( Y ^ { \mathsf { a u x } } )$ respectively at time $t = t _ { 0 }$ , by postprocessing inequality, we have

$$
\begin{array}{l} \mathsf {K L} (p _ {0} \parallel p _ {0} ^ {\mathsf {a u x}}) \leq \sum_ {k = 1} ^ {N} \int_ {t _ {k - 1}} ^ {t _ {k}} \mathbb {E} \left\| \nabla \log p _ {t} (Y _ {t}) - \mathcal {S} _ {t _ {k}, Y _ {t _ {k}}} (Y _ {t}) \right\| ^ {2} \mathrm{d} t \\ \leq 2 \underbrace {\sum_ {k = 1} ^ {N} \int_ {t _ {k - 1}} ^ {t _ {k}} \mathbb {E} \| \nabla \log p _ {t} (Y _ {t}) - \mathcal {S} _ {t _ {k} , Y _ {t _ {k}}} ^ {\star} (Y _ {t}) \| ^ {2} \mathrm{d} t} _ {\text {drift discretization}} + 2 \underbrace {\sum_ {k = 1} ^ {N} \int_ {t _ {k - 1}} ^ {t _ {k}} \mathbb {E} \| \mathcal {S} _ {t _ {k} , Y _ {t _ {k}}} (Y _ {t}) - \mathcal {S} _ {t _ {k} , Y _ {t _ {k}}} ^ {\star} (Y _ {t}) \| ^ {2} \mathrm{d} t} _ {\text {score matching}}. \end{array}\tag{6}
$$

It remains to decompose $\mathsf { T V } ( p _ { 0 } ^ { \mathsf { a u x } } , q _ { 0 } )$ . To isolate the initialization error, we introduce

$$
q _ {0} ^ {\star} = p _ {N} \widehat {\mathsf {K}} _ {N} \widehat {\mathsf {K}} _ {N - 1} \dots \widehat {\mathsf {K}} _ {1}.
$$

By triangle inequality and post-processing inequality, we have

$$
\mathsf {T V} (p _ {0} ^ {\mathsf {a u x}}, q _ {0}) \leq \mathsf {T V} (p _ {0} ^ {\mathsf {a u x}}, q _ {0} ^ {\star}) + \mathsf {T V} (q _ {0} ^ {\star}, q _ {0}) \leq \underbrace {\mathsf {T V} (p _ {0} ^ {\mathsf {a u x}} , q _ {0} ^ {\star})} _ {\text {BM simulation}} + \underbrace {\mathsf {T V} (p _ {N} , q _ {N})} _ {\text {initialization}}.
$$

Step III. Controlling initialization and score errors. By our design, $q _ { N } = \mu ,$ and $\mathsf { T V } ( p _ { N } , q _ { N } ) = \mathsf { T V } ( p _ { N } , \mu )$ . This is known as the mixing rate of heat flow in total variation norm, and has well-established bounds, $\mathrm { e . g . }$ ., Urakawa (2006). The score-matching error, on the other hand, can be controlled with an analysis on the distortion on the Riemannian metric in normal coordinates. We compile the bounds into the following lemma.

Lemma 1. There exists a universal constant $C > 0 ,$ , such that whenever $T \geq 1$ , we have

$$
\mathsf {T V} (p _ {N}, q _ {N}) \leq \mathrm{e} ^ {C (K + d \log d)} \mathrm{e} ^ {- \frac {\lambda_ {1}}{2} (T - \frac {1}{2})},
$$

and

$$
\sum_ {k = 1} ^ {N} \int_ {t _ {k - 1}} ^ {t _ {k}} \mathbb {E} \| \mathcal {S} _ {t _ {k}, Y _ {t _ {k}}} (Y _ {t}) - \mathcal {S} _ {t _ {k}, Y _ {t _ {k}}} ^ {\star} (Y _ {t}) \| ^ {2} \mathrm{d} t \leq 2 \varepsilon_ {\text { score }} ^ {2}.
$$

Step IV. Controlling drift discretization error with Ito/Stratonovich calculus and Li-Yau esti-ˆ mates. The drift discretization error defined in (6) has a similar form to the discretization error for the Euclidean setting (Benton et al., 2024), though additional complications arise due to nonconstant $\mathcal { S } _ { t _ { k } , Y _ { t _ { k } } } ^ { \star }$ . The idea is to study the time derivative of E∥∇ log $p _ { t } ( \dot { \boldsymbol { Y } } _ { t } ) - \mathcal { S } _ { t _ { k } , Y _ { t _ { k } } } ^ { \star } ( \boldsymbol { Y } _ { t } ) \| ^ { 2 }$ , which in view of $\partial _ { \tau }$ log $p _ { t } = - \frac { 1 } { 2 } \Delta _ { \mathcal { M } } p _ { t }$ (negative sign due to reverse time) involves space derivatives of log p up to third order. Fortunately, after applying Ito/Stratonovich calculus to simplify the ex-ˆ pression, a key property in the proof of the Euclidean setting carries over: third-order derivatives of $\log { p _ { t } }$ cancel out. The remaining first and second-order derivatives can be controlled by Li-Yau estimates on the log-gradient of the heat kernel. We obtain

Lemma 2. Under the assumptions in Theorem $^ { l , }$ there is a universal constant $C > 0$ such that

$$
\sum_ {k = 1} ^ {N} \int_ {t _ {k - 1}} ^ {t _ {k}} \mathbb {E} \| \nabla \log p _ {t} (Y _ {t}) - \mathcal {S} _ {t _ {k}, Y _ {t _ {k}}} ^ {\star} (Y _ {t}) \| ^ {2} \mathrm{d} t \leq \frac {C d ^ {6} K ^ {8}}{\delta^ {3}} h ^ {2} N.
$$

Step V. Controlling BM simulation error using parametrix estimates. Our approach is inspired by the following consequence of post-processing inequality and Pinsker’s inequality:

$$
\mathsf {T V} (p _ {0} ^ {\mathsf {a u x}}, q _ {0} ^ {\star}) \leq \sqrt {2 \mathsf {K L} (p _ {0} ^ {\mathsf {a u x}} \parallel q _ {0} ^ {\star})} \leq \sqrt {2 \sum_ {k = 1} ^ {N} \mathsf {K L} (p _ {k} ^ {\mathsf {a u x}} \mathsf {K} _ {k} ^ {\mathsf {a u x}} \parallel p _ {k} ^ {\mathsf {a u x}} \widehat {\mathsf {K}} _ {k})}.
$$

This leads us to compare the kernel $\mathsf { K } _ { k } ^ { \mathsf { a u x } }$ and $\widehat { \sf K } _ { k }$ . In normal coordinates, the Fokker-Planck equation shows that these two are the solutions of the heat equations with the Euclidean Laplacian and with the manifold Laplace-Beltrami operator. We utilize the Minakshisundaram-Pleijel parametrix theory (Berline et al., 2003) in geometric analysis for this comparison, and establish a quantitative bound in polynomially small radius and polynomially short time (cf. Lemma 20).

## 5 CONCLUSION

We developed a discrete-time theory for Riemannian diffusion models showing that a polynomial stepsize suffices for TV-accurate sampling under mild geometric conditions. In particular, our results show that choosing a stepsize polynomially small in manifold parameters achieves any prescribed TV target without exponential blow-ups in dimension or curvature. This complements prior Wasserstein-type guarantees which require exponentially many steps. Several important future directions remain open.

• Sharper bounds. For simplicity, we did not attempt to establish sharp bounds for the error terms in our analysis, and it is likely that the degree of the polynomial in the bound could be improved significantly by refining our analysis, and some of the polynomial dependencies can be improved to logarithmic ones (Benton et al., 2024; Li & Jiao, 2024).

• Analysis of deterministic samplers. We focused on DDPM-style stochastic samplers in our analysis. For practical purpose, it is also tempting to develop an analogous theory for DDIM-style deterministic samplers (Song et al., 2021; Li et al., 2024).

• Conditional sampling. Our theory was for unconditional diffusion models. Applications like solving inverse problems require conditional sampling, which calls for both new algorithm design and new theoretical analysis (Xu & Chi, 2024).

## 6 ACKNOWLEDGMENT

The work of X. Xu and Y. Chi is supported in part by Air Force Office of Scientific Research under FA9550-25-1-0060, and by National Science Foundation under ECCS-2126634/2537078. Z. Zhang, Y. Nakahira, and G. Qu are supported in part by NSF CAREER Award 2339112, NSF Award 2512805, NSF Award 2442948, and the Pennsylvania Infrastructure Technology Alliance. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the United States Air Force.

## REFERENCES

Kwangjun Ahn and Sinho Chewi. Efficient constrained sampling via the mirror-Langevin algorithm. Advances in Neural Information Processing Systems, 34:28405–28418, 2021.

Brian D.O. Anderson. Reverse-time diffusion equation models. Stochastic Processes and their Applications, 12(3):313–326, 1982. ISSN 0304-4149. doi: https://doi.org/10.1016/0304-4149(82) 90051-5.

Joe Benton, Valentin De Bortoli, Arnaud Doucet, and George Deligiannidis. Nearly \$d\$-linear convergence bounds for diffusion models via stochastic localization. In The Twelfth International Conference on Learning Representations, 2024.

Nicole Berline, Ezra Getzler, and Michele Vergne. Heat kernels and Dirac operators. Springer Science & Business Media, 2003.

Eliot Beyler and Francis Bach. Convergence of deterministic and stochastic diffusion-model samplers: A simple analysis in Wasserstein distance. arXiv preprint arXiv:2508.03210, 2025.

Karthik Bharath, Alexander Lewis, Akash Sharma, and Michael V Tretyakov. Sampling and estimation on manifolds using the langevin diffusion. In Advances in Neural Information Processing Systems, 2025.

Sitan Chen, Sinho Chewi, Jerry Li, Yuanzhi Li, Adil Salim, and Anru R. Zhang. Sampling is as easy as learning the score: theory for diffusion models with minimal data assumptions. arXiv preprint arXiv:2209.11215, 2023.

Xiang Cheng, Jingzhao Zhang, and Suvrit Sra. Efficient sampling on Riemannian manifolds via Langevin MCMC. Advances in Neural Information Processing Systems, 35:5995–6006, 2022.