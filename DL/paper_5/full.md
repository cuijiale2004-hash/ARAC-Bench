## ABSTRACT

We introduce a resource-efficient neural network architecture with zero divergence by design, adapted for high-dimensional problems. Our method is directly applicable to image denoising, for which divergence-free estimators are particularly well-suited for self-supervised learning, in accordance with Stein’s unbiased risk estimation theory. Comparisons of our parameterization on popular denoising datasets demonstrate that it retains sufficient expressivity to remain competitive with other divergence-based approaches, while outperforming its counterparts when the noise level is unknown and varies across the training data.

## 1 INTRODUCTION

The divergence is a scalar quantity that measures the rate at which a vector field “flows out” of an infinitesimal region of space. Formally, for a weakly differentiable function $f : \mathbb { R } ^ { n }  \mathbb { R } ^ { n }$ , the divergence at point $\pmb { y } \in \mathbb { R } ^ { n }$ is defined as the trace of the Jacobian matrix $J _ { f } ( { \pmb y } ) \mathrm { : }$

$$
\operatorname{div} f (\boldsymbol {y}) \triangleq \operatorname{tr} \left(\boldsymbol {J} _ {f} (\boldsymbol {y})\right) = \sum_ {i = 1} ^ {n} \frac {\partial f _ {i}}{\partial y _ {i}} (\boldsymbol {y}).\tag{1}
$$

In the special case where the divergence is zero everywhere, the vector field is said to be divergencefree or solenoidal, indicating an incompressible flow. One of the most famous example is without doubt the magnetic field, which, according to Maxwell’s equations, has zero divergence (Maxwell, 1873). Learning divergence-free vector fields is of particular interest at the interface of physics and machine learning (Richter-Powell et al., 2022; Raissi et al., 2017a), as such fields naturally emerge in systems governed by fundamental conservation laws. Parameterizations for learning often exploit the fact that, in $\mathbb { R } ^ { 3 }$ , the curl of any vector field is divergence-free (Morita, 2001), or, more generally, draw on its multidimensional extension via differential forms (Cartan, 1899; Richter-Powell et al., 2022). As long as the target functions remain low-dimensional, training can be performed efficiently with the help of an automatic differentiation engine (Paszke et al., 2019) that powers the computation of partial derivatives. However, scaling challenges arise quickly as the dimensionality increases (Richter-Powell et al., 2022).

In this paper, we establish a representer theorem for divergence-free vector fields, based on structured combinations of conservative fields. Building on this result, and incorporating sparsity constraints, we show how this representation can inform neural network parameterizations that remain resource-efficient, thereby ensuring computational tractability in high dimension. With application to image denoising, for which divergence-free estimators are particularly well-suited for selfsupervised learning, in accordance with Stein’s unbiased risk estimation theory Stein (1981), we propose a methodology to construct low-overhead network architectures that have zero divergence by design and which are adapted to image processing tasks. We demonstrate their competitiveness in comparison to other divergence-based approaches (Batson & Royer, 2019; Tachella et al., 2025a; Soltanayev & Chun, 2018) for the removal of Gaussian noise without clean data.

In summary, the contributions of our work are as follows:

1. The establishment of a representer theorem for divergence-free fields, on which we build to construct neural network architectures that have zero divergence by design.

2. A theoretical framework for analyzing self-supervised image denoising methods grounded in the principle of constant divergence.

3. The demonstration of the competitiveness of our approach in comparison with other constraint-based approaches, particularly when the noise level is unknown and varies across the training data.

## 2 RELATED WORK

Divergence-free networks are particularly studied within physics-informed machine learning and related scientific modeling tasks, which integrate physical laws into the training of neural networks to solve partial differential equations. Notably, enforcing incompressibility constraints is often important—especially in fluid dynamics, where velocity fields are required to be divergence-free.

A common approach employs soft constraints by adding penalty terms to the loss function that encourage the predicted fields to be divergence-free (Raissi et al., 2017b; Mao et al., 2020; Jin et al., 2021). Although such penalty-based methods are straightforward to implement, they do not guarantee strict satisfaction of the incompressibility condition, and residual divergence can remain in some cases, particularly for complex or high-dimensional problems.

To overcome these limitations, recent works have explored hard constraints that enforce divergencefree properties by construction through network architecture or parameterization. For example, in Raissi et al. (2017a), a stream function formulation is used in 2D to represent the velocity field as derivatives of a scalar network output, which is analytically divergence-free. Extending this idea to the multidimensional case, Richter-Powell et al. (2022) designed networks that directly encode conservation laws—including divergence-free constraints—thereby allowing modeling of flow fields and advected quantities without explicit divergence penalties. Nonetheless, scaling these models proves challenging due to their heavy reliance on automatic differentiation. For example, the vectorfield parameterization proposed by Richter-Powell et al. (2022) requires computing a Jacobian matrix, which becomes intractable as the dimension grows.

## 3 CONSTRAINT-BASED APPROACHES FOR SELF-SUPERVISED DENOISING

We focus on denoising problems under the assumption of additive white Gaussian noise (AWGN):

$$
\boldsymbol {y} = \boldsymbol {x} + \sigma \epsilon ,\tag{2}
$$

where $\boldsymbol { y } \in \mathbb { R } ^ { n }$ is the noisy observation, x $\in \mathbb { R } ^ { n }$ is the underlying noise-free signal drawn from a distribution $p _ { \pmb { x } } , \pmb { \epsilon } \sim \mathcal { N } ( \mathbf { 0 } , \pmb { I } _ { n } )$ is a standard Gaussian noise vector, and $\sigma > 0$ is the noise level, drawn from a distribution $p _ { \sigma }$ . Provided that a sufficiently large dataset of noise-free signals x is available, problem (2) is traditionally tackled in a supervised manner by solving:

$$
\arg \min _ {f} \mathbb {E} _ {\boldsymbol {x}, \boldsymbol {y}} \| f (\boldsymbol {y}) - \boldsymbol {x} \| _ {2} ^ {2},\tag{3}
$$

that is, by finding the minimum mean square error (MMSE) estimator, which we denote $f ^ { \mathrm { M M S E } }$ Interestingly, $f ^ { \mathrm { M M S E } }$ admits a closed-form expression whenever the noise level σ is the same for all noisy observations y $( i . e . , p _ { \sigma }$ is a Dirac) which is given by Tweedie’s formula (Efron, 2011): $f ^ { \mathrm { M M S E } } ( \pmb { y } ) = \pmb { y } + \sigma ^ { 2 } \overset {  } { \nabla } \log p _ { \pmb { y } } ( \pmb { \bar { y } } )$ . In this latter expression, the optimal estimator f<sub>MMSE</sub> depends solely on the score of the distribution of the noisy data $\nabla \log p _ { \mathbf { \pmb { y } } } ( \pmb { y } )$ . Accordingly, this formulation indicates that Gaussian denoising may be effectively addressed even in the absence of ground-truth data x for training.

## 3.1 STEIN’S UNBIASED RISK ESTIMATE

Among all the methods proposed in the literature for tackling self-supervised denoising, the approaches grounded in Stein’s Unbiased Risk Estimate (SURE) (Stein, 1981) hold a prominent place. This latter establishes a remarkable identity involving the divergence operator:

$$
\mathbb {E} _ {\boldsymbol {x}, \boldsymbol {y}} \| f (\boldsymbol {y}) - \boldsymbol {x} \| _ {2} ^ {2} = \mathbb {E} _ {\boldsymbol {y}, \sigma} \left[ - n \sigma^ {2} + \| f (\boldsymbol {y}) - \boldsymbol {y} \| _ {2} ^ {2} + 2 \sigma^ {2} \operatorname{div} f (\boldsymbol {y}) \right],\tag{4}
$$

provided that f belongs to $\mathcal { L } ^ { 1 }$ , the space of weakly differentiable functions, and under the assumption that $\mathbb { E } _ { \pmb { y } | \pmb { x } } | f _ { i } ( \pmb { y } ) |$ is bounded. This result is particularly powerful, as it reveals that the mean square error can be reformulated to depend solely on noisy observations and noise levels.

When the noise level σ is known: Assuming that σ is known for every noisy training signal y, equation (4) allows (3) to be rewritten in the following equivalent self-supervised form:

$$
\arg \min _ {f} \mathbb {E} _ {\boldsymbol {y}, \sigma} \left[ \| f (\boldsymbol {y}) - \boldsymbol {y} \| _ {2} ^ {2} + 2 \sigma^ {2} \operatorname{div} f (\boldsymbol {y}) \right],\tag{5}
$$

where the divergence term acts as a form of regularization, preventing the model from simply learning the identity. Many traditional image denoisers—whose divergence admits a closed-form expression—are in fact rooted in SURE (Blu & Luisier, 2007; Van De Ville & Kocher, 2009; Wang & Morel, 2013), even if this connection is not made explicit in some cases (Dabov et al., 2007; Lebrun et al., 2013), as shown by Herbreteau & Kervrann (2025). However, when the estimator f is considerably more complex, such as a deep neural network, its divergence is generally intractable to compute exactly. A popular method is then to employ a Monte Carlo approximation of the divergence, grounded in the following result (Ramani et al., 2008):

$$
\operatorname{div} f (\boldsymbol {y}) = \lim _ {\tau \rightarrow 0} \mathbb {E} _ {\boldsymbol {h} \sim \mathcal {N} (\boldsymbol {0}, \boldsymbol {I})} \left[ \boldsymbol {h} ^ {\top} \frac {f (\boldsymbol {y} + \tau \boldsymbol {h}) - f (\boldsymbol {y})}{\tau} \right],\tag{6}
$$

provided that f admits a well-defined second-order Taylor expansion (if not, this is still valid in the weak sense provided that f is tempered, which is the case for networks with piecewise differentiable activation functions as shown by Soltanayev & Chun (2018)). In practice, a single realization h from the standard normal distribution $\mathcal { N } ( \mathbf { 0 } , \pmb { I } )$ is used for approximating the divergence and τ is chosen as a small constant. In total, only two evaluations of the function f are necessary to estimate its divergence with this method. In a deep learning setting, Soltanayev & Chun (2018); Chen et al. (2022) leveraged this Monte Carlo approximation in combination to the SURE loss (5) to train neural networks on datasets composed only of noisy observations y, leading to the MC-SURE approach. While they achieved performance close to that of the MMSE estimator, a slight gap remains, partly due to approximation errors in the divergence term.

When the noise level σ is unknown: In more realistic settings, one typically has access only to a dataset composed of noisy signals, with no information about the underlying noise level σ for each observation y. Although several methods have been proposed for ad hoc noise estimation (Chen et al., 2015; Pyatykh et al., 2013; Foi et al., 2008), a recent line of research (Tachella et al., 2025a; Batson & Royer, 2019; Krull et al., 2019) seeks to circumvent the need for estimating σ altogether. This is achieved by restricting the estimator $f$ to belong to a constrained set such that

$$
\exists \lambda \in \mathbb {R}, \forall f \in \mathcal {S}, \quad \mathbb {E} _ {\boldsymbol {y}, \sigma} \left[ \sigma^ {2} \operatorname{div} f (\boldsymbol {y}) \right] = \lambda ,\tag{7}
$$

thereby effectively removing this term from the optimization objective (5). The final objective then consists in minimizing the measurement consistency under constraint:

$$
\arg \min _ {f} \mathbb {E} _ {\boldsymbol {y}} \| f (\boldsymbol {y}) - \boldsymbol {y} \| _ {2} ^ {2} \quad \text { s.t. } \quad f \in \mathcal {S}  .\tag{8}
$$

In what follows, we describe two distinct types of constraints proposed in the literature for tackling self-supervised denoising with unknown noise levels, propose an alternative constraint set and then study their shared properties.

Remark In addition to the constraint-based approaches studied in this paper, we also mention, for completeness, the approaches that directly utilize the score function log $p _ { \pmb { y } } ( \pmb { y } )$ in Tweedie’s formula, as proposed in Kim & Ye (2021); Kim et al. (2022); Xie et al. (2023), all of which depend on the estimation technique introduced by Lim et al. (2020). Moreover, Noise2Noise-like (Lehtinen et al., 2018) data augmentation techniques were also proposed (Pang et al., 2021; Huang et al., 2021; Wang et al., 2022; Mansour & Heckel, 2023) as an alternative to SURE.

## 3.2 BLIND-SPOT ESTIMATORS

A radical way to achieve (7) is to impose that each component function $f _ { i }$ does not depend on y<sub>i</sub>. Under this constraint, $f$ is trivially divergence-free by construction since $\begin{array} { r } { \frac { \partial f _ { i } } { \partial y _ { i } } = 0 . } \end{array}$ . This idea dates back to Efron (2004) and lies at the core of the Noise2Self approach (Batson & Royer, 2019) and its variants (Krull et al., 2019; Laine et al., 2019), in which a so-called “blind-spot” network architecture is employed. From a broader perspective, this constraint can be generalized by restricting f to the space

![](images/eb0867392511d36aafe81f2b1fb3dabb5e29f56fafa99caab5054e96d339229a.jpg)  
Figure 1: Balancing expressive power in constraint-based self-supervised Gaussian denoising. When the noise model is fully specified, SURE yields the most expressive estimators, whose performance matches theoretically that of supervised methods. As assumptions on the noise are gradually relaxed, however, the learned estimators must reduce their expressivity to avoid overfitting the noise. In this work, we introduce divergence-constant estimators, a family that preserves much of the expressivity lost in blind-spot designs, yet still allows training even when the noise level σ is unknown and varies across the data.

$$
\mathcal {S} _ {\mathrm{BS}} ^ {c} = \left\{f \in \mathcal {L} ^ {1} (\mathbb {R} ^ {n}, \mathbb {R} ^ {n}): \forall \boldsymbol {y} \in \mathbb {R} ^ {n}, \frac {\partial f _ {i}}{\partial y _ {i}} (\boldsymbol {y}) = c \right\},\tag{9}
$$

which still satisfies (7) as long as $c \in \mathbb { R }$ is a constant fixed in advance. In the case $c = 0$ , the solution of (8) under blind-spot constraints is given by $f _ { i } ^ { \mathrm { B S } } ( { \pmb y } ) = \mathbb { E } \{ y _ { i } | { \pmb y } _ { - i } \} = \mathbb { E } \{ x _ { i } | { \pmb y } _ { - i } \}$ , where ${ \mathbf { \delta } } _ { \mathbf { \pmb { y } } - i }$ refers to the vector obtained by excluding the ith entry.

The strength of blind-spot constraints lies actually in their versatility: they can handle a wide range of noise types, specifically those that are zero-mean and spatially independent, of which (2) is a prime example, without precisely knowing the noise distribution. However, this flexibility comes at a significant performance cost. A blind-spot architecture is indeed inherently less expressive than a classic one, especially since $y _ { i }$ is usually highly informative about $x _ { i } ,$ , and tends to introduce checkerboard artifacts (Hock et al., 2022).¨

## 3.3 UNSURE ESTIMATORS

In the particular case where every noisy observation y is corrupted by the same noise level $\sigma ( i . e . , p _ { o }$ is a Dirac), Tachella et al. (2025a) proposed a softened version of the constraint (9) imposing only that the estimator has zero expected divergence, that is, $\mathbb { E } _ { y }$ div $f ( \pmb { y } ) = 0$ . This relaxation retains the validity of (7) while offering increased expressivity, since any blind-spot estimator automatically satisfies the zero–expected-divergence condition. Extending it to the constant case, this alternative constraint forces f to belong to the space

$$
\mathcal {S} _ {\mathrm{CED}} ^ {c} = \left\{f \in \mathcal {L} ^ {1} (\mathbb {R} ^ {n}, \mathbb {R} ^ {n}): \mathbb {E} _ {\boldsymbol {y}} \operatorname{div} f (\boldsymbol {y}) = n c \right\}.\tag{10}
$$

In the case $c = 0$ , a closed-form solution of the optimization problem (8) was established, namely $\begin{array} { r } { f ^ { \mathrm { Z E D } } ( \pmb { y } ) = \pmb { y } + \hat { \eta } \nabla \log p _ { \pmb { y } } ( \pmb { y } ) , \mathrm { w i t h } \hat { \eta } = ( \mathbb { E } _ { \pmb { y } } \frac { 1 } { n } \lVert \dot { \nabla } \log p _ { \pmb { y } } ( \pmb { y } ) \rVert _ { 2 } ^ { 2 } ) ^ { - 1 } } \end{array}$

An important difference with the blind-spot approach lies in how the optimization is carried out in practice. Indeed, unlike blind-spot approaches—where the constraint is enforced directly through the design of f—the so-called UNSURE approach seeks a saddle point of the Lagrangian by formulating the problem as a min–max optimization which is solved by alternating gradient-descent-ascent (Arrow et al., 1958; Platt & Barr, 1987). However, such an optimization method comes with several caveats. First, constraint satisfaction is not guaranteed in practice; only the penalty term associated with violations is minimized. Second, the outcome is highly sensitive to the choice of learning-rate pair for gradient-descent-ascent, which controls the trade-off between the objective and the constraint, and an inappropriate choice can lead to instabilities or oscillatory dynamics during training (Platt & Barr, 1987; Gallego-Posada et al., 2022). Finally, in this setting, the divergence term is estimated via a Monte Carlo approximation based on (6) using a limited number of samples, which can further degrade the accuracy of the optimization.

## 3.4 PROPOSED ALTERNATIVE: CENSURE ESTIMATORS

In this work, we propose to study the set of weakly differentiable vector fields on $\mathbb { R } ^ { n }$ with constant (normalized) divergence $c \in \mathbb { R }$ , denoted by

$$
\mathcal {S} _ {\mathrm{DC}} ^ {c} = \left\{f \in \mathcal {L} ^ {1} (\mathbb {R} ^ {n}, \mathbb {R} ^ {n}): \forall \boldsymbol {y} \in \mathbb {R} ^ {n}, \operatorname{div} f (\boldsymbol {y}) = n c \right\},\tag{11}
$$

de facto introducing an intermediate constraint set lying between the blind-spot constraint set and the much looser expected divergence constraint one: $S _ { \mathrm { B S } } ^ { c } \subset S _ { \mathrm { D C } } ^ { c } \subset S _ { \mathrm { C E D } } ^ { c }$ . We emphasize that all inclusions are strict, with in particular the possibility for $f \in S _ { \mathrm { D C } } ^ { 0 }$ to have each of its component function $f _ { i }$ to depend on $y _ { i } ,$ , which is excluded for a function in $ { \widetilde { S } } _ { \mathrm { B S } } ^ { \mathrm { 0 } }$ (see Fig. 6 in Appendix). We postpone the description of the way we construct in practice such divergence-constant estimators to the next section. Let us simply note that, similarly to existing alternatives, divergence-constant mappings are of particular interest in self-supervised denoising since the condition (7) is always satisfied, regardless of whether the noise level σ is constant across observations y or varies from one observation to another. This is a major difference with the UNSURE approach which cannot, a priori, be used when the noise level varies across the noisy observations. Indeed, in full generality, $\scriptstyle \mathbf { \bar { E } } _ { y , \sigma } [ \sigma ^ { 2 }$ div $f ( \pmb { y } ) ] \neq \mathbb { E } _ { \sigma } [ \sigma ^ { 2 } ] \mathbb { E } _ { \pmb { y } } [ \operatorname { d i v } f ( \pmb { y } ) ]$ because $\sigma ^ { 2 }$ is not independent of div $f ( \boldsymbol { y } )$ , since $\textbf {  { y } }$ itself depends on σ. Figure 1 illustrates the expressivity trade-off that emerges in self-supervised denoising. Our proposed approach, termed CENSURE (Concealed and Erratic Noise level with Stein’s Unbiased Risk Estimate), bridges the gap between constant expected divergence estimators and the more restrictive blind-spot ones.

## 3.5 PROPERTIES SHARED BY ALL CONSTRAINT SETS

For conciseness, $S ^ { c }$ denotes one of the following sets: ${ \cal S } _ { \mathrm { B S } } ^ { c } , { \cal S } _ { \mathrm { C E D } } ^ { c }$ or $ { \boldsymbol { S } } _ { \mathrm { D C } } ^ { c }$ . This paragraph should be read by selecting one of these three sets consistently, without mixing them. As a preliminarily observation, we notice that the constraint set $S ^ { c }$ admits an affine space structure, based at c id, where id refers to the identity map on $\mathbb { R } ^ { n }$ . This statement is formalized in the following lemma (all the proofs of this paper are given in Appendix C).

Lemma 1. $S ^ { 0 }$ is a linear space and $S ^ { c }$ is an affine space with $S ^ { c } = c \mathrm { i d } + S ^ { 0 }$

A direct consequence (see Proposition 1) is that the optimal denoiser within each class can be written as an affine combination of the identity function and the minimizer in $ { \boldsymbol { S } } ^ { 0 }$ of the measurement consistency term. Thus, it is sufficient to restrict the search to estimators in $ { \boldsymbol { S } } ^ { 0 }$ , since any optimal denoiser in $S ^ { c }$ can be recovered from an optimal denoiser in $ { \boldsymbol { S } } ^ { 0 }$

Proposition 1. The solution $o f ( \delta )$ for the constraint set $S ^ { c }$ can be expressed as

$$
\underset {f \in \mathcal {S} ^ {c}} {\arg \min} \mathbb {E} _ {\boldsymbol {y}} \| f (\boldsymbol {y}) - \boldsymbol {y} \| _ {2} ^ {2} = c \operatorname{id} + (1 - c) \underset {f \in \mathcal {S} ^ {0}} {\arg \min} \mathbb {E} _ {\boldsymbol {y}} \| f (\boldsymbol {y}) - \boldsymbol {y} \| _ {2} ^ {2}.\tag{12}
$$

An immediate question that arises at this point is: How to choose the constant c to achieve the best denoising? Proposition 2 provides a theoretical characterization of the optimal constant $c ^ { * }$

Proposition 2 (Optimal constant). In the AWGN setting (see equation 2), $i f S ^ { c } \in \{ S _ { \mathrm { B S } } ^ { c } , S _ { \mathrm { D C } } ^ { c } \}$ or if $S ^ { c } \overset { = } { = } S _ { \mathrm { C E D } } ^ { c }$ and $p _ { \sigma }$ is a Dirac then

$$
c ^ {*} = \underset {c \in \mathbb {R}} {\arg \min} \underset {f \in \mathcal {S} ^ {c}} {\min} \mathbb {E} _ {\boldsymbol {x}, \boldsymbol {y}} \| f (\boldsymbol {y}) - \boldsymbol {x} \| _ {2} ^ {2} = 1 - \frac {n \mathbb {E} _ {\sigma} [ \sigma^ {2} ]}{\min _ {f \in \mathcal {S} ^ {0}} \mathbb {E} _ {\boldsymbol {y}} \| f (\boldsymbol {y}) - \boldsymbol {y} \| _ {2} ^ {2}} \in [ 0, 1 ].\tag{13}
$$

Interestingly, the optimal constant $c ^ { * }$ lies in [0, 1]. As a consequence, the affine combination in Prop. 1 is in fact a convex combination in the optimal case. Naturally, $c ^ { * }$ depends on the knowledge of $\mathbb { E } _ { \sigma } [ \sigma ^ { 2 } ]$ , which may be unknown in some settings. This explains why the arbitrary choice $c = 0$ is made in practice (Batson & Royer, 2019; Tachella et al., 2025a). Unless otherwise stated, we adopt the same choice for CENSURE.

Finally, using the fact that $S _ { \mathrm { D C } } ^ { c } \subset S _ { \mathrm { C E D } } ^ { c } ,$ we can derive a lower bound on the reconstruction error for divergence-constant estimators in the case where σ is constant.

Proposition 3 (A lower bound). In the AWGN setting with constant noise level σ (see equation 2),

$$
\begin{array}{r l} & {\underset {f \in \mathcal {S} _ {\mathrm{DC}} ^ {c}} {\min} \mathbb {E} _ {\boldsymbol {x}, \boldsymbol {y}} \frac {1}{n} \| f (\boldsymbol {y}) - \boldsymbol {x} \| _ {2} ^ {2} \geq \underset {f \in \mathcal {S} _ {\mathrm{CED}} ^ {c}} {\min} \mathbb {E} _ {\boldsymbol {x}, \boldsymbol {y}} \frac {1}{n} \| f (\boldsymbol {y}) - \boldsymbol {x} \| _ {2} ^ {2}} \\ & {\qquad = \mathrm{MMSE} + \frac {\sigma^ {2}}{1 - \frac {\mathrm{MMSE}}{\sigma^ {2}}} \left(c - \frac {\mathrm{MMSE}}{\sigma^ {2}}\right) ^ {2} \geq \mathrm{MMSE},} \end{array}\tag{14}
$$

where $\begin{array} { r } { \mathrm { M M S E } = \mathbb { E } _ { \pmb { x } , \pmb { y } } \frac { 1 } { n } \| \mathbb { E } ( \pmb { x } | \pmb { y } ) - \pmb { x } \| _ { 2 } ^ { 2 } } \end{array}$ is the minimum mean square error.

## 4 DESIGN OF DIVERGENCE-FREE NEURAL NETWORKS

We now present our proposed methodology for constructing divergence-free network architectures.

## 4.1 REPRESENTING DIVERGENCE-FREE VECTOR FIELDS

Lemma 2 offers a straightforward method for generating divergence-free vector fields and highlights the key role played by skew-symmetric matrices in ensuring zero divergence.

Lemma ${ \mathfrak { 2 } } \left( { \mathrm { A } } \right.$ simple divergence-free vector field). Let $\psi : \mathbb { R } ^ { n } $ R be a smooth scalarfield and let $A \in \mathbb { R } ^ { n \times n }$ be a skew-symmetric matrix, $i . e . \ { \pmb A } ^ { \top } = - { \pmb A }$ . The vectorfield A ψ is divergence-free.

Nevertheless, this construction does not capture all divergence-free vector fields, except for the case $n = 2$ . In fact, fully representing such fields typically requires combining multiple expressions of this form, as formalized in the following representer theorem.

Theorem 1 (A universal representation of divergence-free fields). Let $f : \mathbb { R } ^ { n }  \mathbb { R } ^ { n }$ be a smooth vector field and let $\{ A _ { 1 } , \dotsc , A _ { K } \} \in { \mathbb { R } } ^ { n \times n }$ be a basis of the space of real skew-symmetric $n \times n$ matrices, with $n \geq 2 .$ f is divergence-free ifand only ifthere exist smooth scalarfields $\psi _ { 1 } , \dots , \psi _ { K } :$ $\mathbb { R } ^ { n } \to \mathbb { R }$ such that

$$
f = \sum_ {k = 1} ^ {K} \boldsymbol {A} _ {k} \nabla \psi_ {k}.\tag{15}
$$

The proof is an extension of the work of Richter-Powell et al. (2022) which builds on the classical Hodge decomposition theorem (Morita, 2001; Berger, 2003). Note that the space of real skewsymmetric $n \times n$ matrices is of dimension $K = { \binom { n } { 2 } }$ , hence the above construction uses a number of scalar fields that scales quadratically with the dimension n.

Application for $n = 3$ Consider the following basis of real skew-symmetric $3 \times 3$ matrices:

$$
(\boldsymbol {A} _ {1}, \boldsymbol {A} _ {2}, \boldsymbol {A} _ {3}) = \left(\left( \begin{array}{c c c} 0 & 0 & 0 \\ 0 & 0 & 1 \\ 0 & - 1 & 0 \end{array} \right), \left( \begin{array}{c c c} 0 & 0 & - 1 \\ 0 & 0 & 0 \\ 1 & 0 & 0 \end{array} \right), \left( \begin{array}{c c c} 0 & 1 & 0 \\ - 1 & 0 & 0 \\ 0 & 0 & 0 \end{array} \right)\right)  ,\tag{16}
$$

and let $f$ be a smooth divergence-free vector field. According to Theorem 1, there exist $\psi _ { 1 } , \psi _ { 2 } , \psi _ { 3 } :$ $\mathbb { R } ^ { 3 } \to \dot { \mathbb { R } }$ , such that

$$
f = \sum_ {k = 1} ^ {3} \boldsymbol {A} _ {k} \nabla \psi_ {k} = \left( \begin{array}{c c} \frac {\partial \psi_ {3}}{\partial y _ {2}} & - \frac {\partial \psi_ {2}}{\partial y _ {3}} \\ \frac {\partial \psi_ {1}}{\partial y _ {3}} & - \frac {\partial \psi_ {3}}{\partial y _ {1}} \\ \frac {\partial \psi_ {2}}{\partial y _ {1}} & - \frac {\partial \psi_ {1}}{\partial y _ {2}} \end{array} \right) \triangleq \nabla \times \binom{\psi_ {1}}{\psi_ {2}}.\tag{17}
$$

In other words, there exists a vector field $\psi = ( \psi _ { 1 } , \psi _ { 2 } , \psi _ { 3 } )$ such that $f$ is its curl. This is a wellknown result in the literature (Morita, 2001).

## 4.2 PROPOSED ARCHITECTURE

Our goal is to construct a parameterized function $f ,$ under the form of a neural network, that is divergence-free by design and whose architecture is tailored for image processing tasks, in particular denoising. To this end, we build on the representer Theorem 1, which suggests defining f as a structured combination of conservative fields $\bar { \nabla } \psi _ { k }$ . However, as previously noted, the number of terms in this representation, namely K in Theorem 1, is on the order of $n ^ { 2 }$ , which becomes prohibitive as soon as we work with images. This scalability issue was previously highlighted by Richter-Powell et al. (2022). To keep computations tractable, we propose to constraint $f$ to be represented using a sparse combination of conservative fields, which we assume retains sufficient representational fidelity. This deliberate simplification ultimately consists in substituting K with $K ^ { \prime } \ll K$ (typically $K ^ { \prime } = 8 )$ . More precisely, we build f under the form

$$
f = \sum_ {k = 1} ^ {K ^ {\prime}} \boldsymbol {A} _ {k} \nabla \psi_ {k},\tag{18}
$$

where $\psi _ { 1 } , \ldots , \psi _ { K ^ { \prime } } : \mathbb { R } ^ { n } \to \mathbb { R }$ are parameterized via a single shared neural network and $\left\{ A _ { 1 } , \ldots , A _ { K ^ { \prime } } \right\} \ \in \ \mathbb { R } ^ { n \times n }$ are (sparse) skew-symmetric matrices, also parameterized. Note that even though we retain fewer terms in (18) than required, the resulting function remains exactly divergence-free, as ensured by Lemma 1 and 2 (divergence-free functions are closed under addition). The parameter $K ^ { \prime }$ only affects expressivity: when $\check { K ^ { \prime } } = 0 , f = 0$ which is trivially divergence-free but not expressive at all; conversely, $K ^ { \prime } = K$ yields maximal expressivity, as stated in Theorem 1, but becomes computationally intractable. We now detail the construction of both types of parameterization.

## 4.2.1 DESIGN OF THE SKEW-SYMMETRIC MATRICES

For the sake of computational efficiency, the skew-symmetric matrices $\pmb { A } _ { k }$ in (18) are chosen to be sparse matrices with shared parameters as follows:

$$
\boldsymbol {A} _ {k} = \boldsymbol {P} _ {k} ^ {\top} \frac {\boldsymbol {\Theta} - \boldsymbol {\Theta} ^ {\top}}{2} \boldsymbol {P} _ {k},\tag{19}
$$

where $\Theta$ is a shared parameterized repeated-block diagonal matrix and where each $P _ { k } \in \mathbb { R } ^ { n \times n }$ is a different and fixed permutation matrix (typically a rotation or shift matrix). Note that the matrices $\pmb { A } _ { k }$ are guaranteed to be skew-symmetric by design thanks to the following equality of sets, valid for any permutation matrix $P _ { k } \colon \{ A \in \mathbb { R } ^ { n \times n } : A ^ { \top } = - A \} = \{ P _ { k } ^ { \top } \frac { A - A ^ { \top } } { 2 } P _ { k } : A \in \mathbb { R } ^ { n \times n } \}$

## 4.2.2 DESIGN OF THE SCALAR FIELDS

The idea of designing neural networks to represent exact conservative fields, $i . e .$ of the form ψ, has already been explored in works targeting energy based models or plug-and-play methods (Salimans & Ho, 2021; Hurault et al., 2022a). They all point out the critical choice of the architecture for the scalar potential function ψ in order to achieve good performance in practice. In particular, as experimentally observed, modeling $\psi$ as a standard feedforward network, such as the ones used for classification, severely degrades performance. Instead, it is recommended to incorporate an architecture tailored to the target application directly into the design of $\psi$ . This is why, we propose to consider parameterized scalar fields of the form

$$
\psi_ {\pmb {\theta}, \pmb {B} _ {k}}: \pmb {y} \in \mathbb {R} ^ {n} \mapsto \frac {1}{2} \left(\| \pmb {B} _ {k} \pmb {y} \| _ {2} ^ {2} - \| \pmb {B} _ {k} \pmb {y} - D _ {\pmb {\theta}} (\pmb {y}) \| _ {2} ^ {2}\right),\tag{20}
$$

where $B _ { k } \in \mathbb { R } ^ { n \times n }$ and $D _ { \pmb { \theta } } : \mathbb { R } ^ { n }  \mathbb { R } ^ { n }$ is a neural network specific to image processing, typically a U-Net (Ronneberger et al., 2015). Please note that the neural network parameters θ are shared for all scalar fields. The specific form of the scalar fields in (20) is strongly inspired by Hurault et al. (2022a), with the addition of the $\scriptstyle B _ { k }$ matrices introduced in our formulation. It is justified by the fact that

$$
\nabla \psi_ {\boldsymbol {\theta}, \boldsymbol {B} _ {k}} (\boldsymbol {y}) = \boldsymbol {B} _ {k} ^ {\top} D _ {\boldsymbol {\theta}} (\boldsymbol {y}) + \mathbf {J} _ {D _ {\boldsymbol {\theta}}} (\boldsymbol {y}) ^ {\top} \left(\boldsymbol {B} _ {k} \boldsymbol {y} - D _ {\boldsymbol {\theta}} (\boldsymbol {y})\right),\tag{21}
$$

for which the first term is known to be effective for learning denoising functions. Beyond introducing diversity for the scalar potential functions, the inclusion of the matrix $\scriptstyle B _ { k }$ in (20) has the effect of replacing the term $D _ { \pmb \theta } ( \pmb y )$ by $B _ { k } ^ { \top } D _ { \theta } ( \pmb { y } )$ , with the hope that this (transposed) matrix could counterbalance the potentially negative effect of multiplication by a skew-symmetric matrix $\pmb { A } _ { k }$ afterwards. In practice, expression (21) is computed by differentiating (20) with respect to the input y using an automatic differentiation engine (Paszke et al., 2019), which avoids computing the full Jacobian.

Finally, the matrices $\scriptstyle B _ { k }$ in (20) are parameterized analogously to (19) via a shared repeated-block diagonal matrix $\Theta ^ { \prime } \in \mathbb { R } ^ { n \times n }$ , in accordance with

$$
\boldsymbol {B} _ {k} = \boldsymbol {P} _ {k} ^ {\top} \boldsymbol {\Theta} ^ {\prime} \boldsymbol {P} _ {k}.\tag{22}
$$

Table 1: The PSNR (dB) results of self-supervised methods on color datasets corrupted by synthetic white Gaussian noise. Training was conducted with unknown non-constant $\sigma \in [ 0 , 7 5 ]$ (a single model to handle all noise levels). The best and second best results in each category are highlighted in red and blue colors, respectively. Non-constraint-based methods are denoted with \*.

<table><tr><td></td><td>Dataset</td><td colspan="5">Kodak24</td><td colspan="5">CBSD68</td></tr><tr><td></td><td>Noise level σ</td><td>15</td><td>/</td><td>25</td><td>/</td><td>50</td><td>/</td><td>75</td><td>15</td><td>/</td><td>25</td></tr><tr><td>supervised</td><td>DRUNet light</td><td>35.18</td><td>/</td><td>32.78</td><td>/</td><td>29.77</td><td>/</td><td>28.14</td><td>34.22</td><td>/</td><td>31.61</td></tr><tr><td rowspan="6">self-supervised</td><td>Noise2Score*</td><td>34.69</td><td>/</td><td>32.02</td><td>/</td><td>28.17</td><td>/</td><td>22.91</td><td>33.82</td><td>/</td><td>31.00</td></tr><tr><td>Neighbor2Neighbor*</td><td>34.72</td><td>/</td><td>32.37</td><td>/</td><td>29.42</td><td>/</td><td>27.82</td><td>33.82</td><td>/</td><td>31.29</td></tr><tr><td>UNSURE ( $\tau = 10^{-2}$ )</td><td>29.48</td><td>/</td><td>22.03</td><td>/</td><td>15.58</td><td>/</td><td>12.56</td><td>29.32</td><td>/</td><td>22.15</td></tr><tr><td>UNSURE ( $\tau = 10^{-4}$ )</td><td>26.89</td><td>/</td><td>22.01</td><td>/</td><td>15.93</td><td>/</td><td>12.82</td><td>26.83</td><td>/</td><td>22.10</td></tr><tr><td>Noise2Self</td><td>34.08</td><td>/</td><td>31.90</td><td>/</td><td>29.07</td><td>/</td><td>27.49</td><td>33.06</td><td>/</td><td>30.70</td></tr><tr><td>CENSURE (ours)</td><td>34.21</td><td>/</td><td>32.05</td><td>/</td><td>29.24</td><td>/</td><td>27.67</td><td>33.17</td><td>/</td><td>30.83</td></tr></table>

Please note that the fixed permutation matrices $P _ { k }$ are the same as in (19). Ultimately, the learnable parameters for the proposed parametrization of (18) are $\{ \pmb \theta , \Theta , \Theta ^ { \prime } \}$ and their number is only slightly greater than that of $D _ { \theta }$ since Θ and $\Theta ^ { \prime }$ are sparse, which supports the practicality of our proposed parameterization.

## 5 EXPERIMENTAL RESULTS

We demonstrate the effectiveness of our proposed methodology to construct divergence-free networks, termed CENSURE (Concealed and Erratic Noise level with Stein’s Unbiased Risk Estimate), in the case of self-supervised image denoising under the assumption of additive white Gaussian noise and compare its competitiveness with related state-of-the-art approaches that rely on (4) such as MC-SURE (Soltanayev & Chun, 2018) and its constraint-based variants, namely Noise2Self (Batson & Royer, 2019) and UNSURE (Tachella et al., 2025a). We also compare with Neighbor2Neighbor (Huang et al., 2021) and Noise2Score (Kim & Ye, 2021) (in the unknown–noise-level setting, its extension (Kim et al., 2022) was considered, still referred to as Noise2Score) as non-constraint-based baselines. For a fair comparison, we used a common backbone architecture for all methods and trained all models ourselves. Note that approaches that rely on a Monte Carlo approximation (6) of the divergence involve an additional hyperparameter τ. In our experiments, we considered two values: $\tau = 1 0 ^ { - 2 }$ , as recommended for vectors with entries in [0, 1] by Tachella et al. (2025b), and $\tau = 1 0 ^ { - 4 }$ , a value we selected based on test-set performance and which may be viewed as an oracle hyperparameter. Performance of CENSURE and other methods are assessed in terms of PSNR values. Details about training, datasets and implementations can be found in Appendix B. Source code is available at the following repository: https://github.com/sherbret/divergence free nn.

## 5.1 UNKNOWN NON-CONSTANT NOISE LEVEL

We first evaluate the self-supervised constraint-based methods for denoising under the assumption that the noise level σ is unknown and varies across the training data. This scenario is the most realistic in practice because real-world sensors rarely operate at a fixed signal-to-noise ratio, and the corruption level can change over time, across devices, or even between consecutive acquisitions. According to Figure 1, our approach is, in principle, the most suitable in this setting. For each method, we learn a single model to handle all noise levels. Specifically, training is conducted using only noisy images synthetically corrupted with Gaussian noise and random $\sigma \sim \mathcal { U } ( [ 0 , 7 5 ] )$ ). Please note that the values of σ (assumed unknown) are not used in the loss function or during inference/test time. Table 1 reports the PSNR results on two benchmark datasets for color image denoising, with representative qualitative examples shown in Figure 2. Results for grayscale images are presented in Table 3 in Appendix, along with corresponding qualitative comparisons in Figure 7. We can observe that CENSURE outperforms its constraint-based counterparts, which is in accordance with the theory. In particular, UNSURE fails when σ varies, mainly because it tends to overfit the noise. Indeed, contrary to the case where σ is constant, $\mathbb { E } _ { \pmb { y } , \sigma } [ \sigma ^ { 2 } \mathrm { d i v } ^ { \circ } f ( \pmb { y } ) ] \neq \mathbb { E } _ { \sigma } [ \sigma ^ { 2 } ] \mathbb { E } _ { \pmb { y } } [ \mathrm { d i v } f ( \pmb { y } ) ]$ in full generality since $\sigma ^ { 2 }$ and div $f ( \boldsymbol { y } )$ are statistically dependent. Therefore, condition (7) is not satisfied for UNSURE. Finally, although Neighbor2Neighbor appears to be the best-performing self-supervised method overall, it is inherently limited to natural images, as it relies on the core assumption that two noise-free neighboring pixels share similar values most of the time. In contrast, constraint-based approaches are more general for denoising problems.

![](images/eee52016d7eb876b72e3be806c8340e5980516efc9c744c85424b207ff2b9d7b.jpg)  
Figure 2: Qualitative blind image denoising results. All models were trained on a dataset with unknown random $\sigma \in [ 0 , 7 5 ]$ (a single model to handle all noise levels). Best viewed by zooming.

## 5.2 CONSTANT NOISE LEVEL

For the sake of completeness, we also assess the performance of self-supervised constraint-based methods in the less realistic setting where the noise level σ is constant (whether known or unknown) across the training data. As shown in Figure 1, this scenario favors the use of estimators that are more expressive than CENSURE such as UNSURE or MC-SURE. Table 4 in Appendix D reports a quantitative comparison on two test datasets for grayscale images. Contrary to the previous scenario, a separate model is learned for each noise level. The results are organized into two categories: methods that require the noise level σ during training or inference (known σ setting), and those that are completely agnostic to it (unknown σ setting), including CENSURE, UNSURE and Noise2Self. Importantly, Propositions 1 and 2 show that divergence-free denoisers, either everywhere or in expectation, can be converted into noise-level–aware ones (marked with symbol in Table 4) without additional training. Specifically, for a divergence-free estimator f, the quantity min $f { \in } S ^ { 0 } \mathbb { E } _ { \pmb { y } } { \| } f ( \pmb { y } ) - \pmb { y } { \| } _ { 2 } ^ { 2 }$ in Proposition 2 can be approximated using a single realization $\| f ( \pmb { y } ) - \pmb { y } \| _ { 2 } ^ { 2 }$ , where y denotes the noisy input image (averaging this term over more image realizations did not lead to further improvements).

As expected, UNSURE achieves the best performance among divergence-based methods when the noise level σ is constant and unknown, followed by CENSURE and then Noise2Self. This ordering is consistent with the fact that $S _ { \mathrm { B S } } ^ { 0 } \subset S _ { \mathrm { D C } } ^ { 0 } \subset S _ { \mathrm { C E D } } ^ { \mathrm { 0 } } \mathrm { : }$ the fewer constraints imposed on the search space of the estimator, the more expressive it becomes, as shown by Figure 1. Note that a slight performance gap is observed for UNSURE depending on the choice of the hyperparameter τ .

When the noise level σ is assumed known, all divergence-free methods naturally benefit from this additional information, showing PSNR gains in line with theoretical expectations (see Subsection 3.5), except for UNSURE with $\tau \stackrel { - } { = } 1 0 ^ { - 2 }$ . This hyperparameter choice is also suboptimal for MC-SURE, whose performance falls short of expectations, allowing CENSURE to outperform it in most cases. Indeed, according to (4), a denoiser trained with the SURE loss should achieve performance comparable to its supervised counterpart. The observed underperformance can largely be attributed to stability issues during training. As illustrated in Figure 4 in Appendix, the training curves of MC-SURE and UNSURE exhibit pronounced fluctuations, in contrast to the much smoother trajectories of the other methods, including ours. This phenomenon was already described in Soltanayev & Chun (2018) and considering the oracle hyperparameter $\tau = 1 0 ^ { - 4 }$ (chosen based on test-set performance) instead solved this issue in our case. To the best of our knowledge, there is no systematic way to set this value; it likely depends on the Lipschitz constant of the network, which is itself difficult to estimate in advance Pintore & Despres (2024). This highlights the strength of our approach, as CEN-´ SURE remains completely independent of this choice by naturally enforcing strict zero divergence. Finally, when the oracle value $\tau = 1 0 ^ { - 4 }$ is used, MC-SURE becomes the best-performing method, achieving results close to the supervised model, in agreement with theoretical analysis. Qualitative examples illustrating the performance of self-supervised methods in the constant-noise regime are presented in Figure 8 in Appendix.

## 6 CONCLUSION

We presented an original approach for constraining neural networks to be divergence-free by design. Our proposed parameterization is grounded in a representer theorem for divergence-free vector fields, which characterizes them as structured combinations of conservative fields. Leveraging this theoretical foundation and incorporating sparsity constraints, we derived parameterizations for neural networks that are both resource-efficient and scalable to high-dimensional settings. The practical relevance of our approach is illustrated in the context of self-supervised image denoising, where we demonstrated that that these models achieve competitive performance compared to existing constraint-based methods, especially when the noise level is unknown and varies across the training data. Beyond denoising, our results suggest that our divergence-free parameterization may hold promise for a wider range of high-dimensional learning tasks, in particular in physics-informed machine learning, opening new avenues for future research.

## ACKNOWLEDGEMENTS

The authors acknowledge the support of the Research Group IASIS of the CNRS Informatics (Project DIVIN). Etienne Meunier was partially supported by the Research Programme “PPR Ocean & Climate” through a postdoctoral scholarship. This work was granted access to the HPC resources of IDRIS under the allocation 2025-AD011015932R1 made by GENCI. We thank Cedric Herzet for´ fruitful discussions.

## REFERENCES

Eirikur Agustsson and Radu Timofte. NTIRE 2017 Challenge on single image super-resolution: Dataset and study. In Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pp. 1122–1131, 2017.

Kenneth Joseph Arrow, Leonid Hurwicz, Hirofumi Uzawa, Hollis Burnley Chenery, Selmer Johnson, and Samuel Karlin. Studies in linear and non-linear programming, volume 2. Stanford University Press Stanford, 1958.

Joshua Batson and Loic Royer. Noise2Self: Blind denoising by self-supervision. In International Conference on Machine Learning (ICML), volume 97, pp. 524–533, 2019.

Marcel Berger. A panoramic view of Riemannian geometry. Springer, 2003.

D.P. Bertsekas. Nonlinear Programming. Athena Scientific optimization and computation series. Athena Scientific, 1995. ISBN 9781886529144.

Thierry Blu and Florian Luisier. The SURE-LET approach to image denoising. IEEE Transactions on Image Processing, 16(11):2778–2786, 2007.