## ABSTRACT

Out-of-distribution (OOD) detection is critical for the safe deployment of machine learning systems in safety-sensitive domains. Diffusion models have recently emerged as powerful generative models, capable of capturing complex data distributions through iterative denoising. Building on this progress, recent work has explored their potential for OOD detection. We propose EigenScore, a new OOD detection method that leverages the eigenvalue spectrum of the posterior covariance induced by a diffusion model. We argue that posterior covariance provides a consistent signal of distribution shift, leading to larger trace and leading eigenvalues on OOD inputs, yielding a clear spectral signature. We further provide analysis explicitly linking posterior covariance to distribution mismatch, establishing it as a reliable signal for OOD detection. To ensure tractability, we adopt a Jacobian-free subspace iteration method to estimate the leading eigenvalues using only forward evaluations of the denoiser. Empirically, EigenScore achieves state-of-the-art performance, with up to 2% AUROC improvement over the best baseline. Notably, it remains robust in near-OOD settings such as CIFAR-10 vs CIFAR-100, where existing diffusion-based methods often fail.

## 1 INTRODUCTION

Most machine learning systems assume that test data matches the training distribution, but distribution shift or out-of-distribution (OOD) data can severely degrade performance in safety-critical domains such as medical imaging and autonomous driving (Yang et al., 2024). OOD inputs may stem from sensor noise, semantic differences, or acquisition changes, leading to unreliable predictions (Zhang et al., 2023). To address this, many OOD detection methods have been proposed, ranging from supervised approaches that require labeled OOD data to unsupervised approaches that rely only on in-distribution (InD) training data (Graham et al., 2023).

Existing OOD detection methods can be broadly categorized into four families: (i) uncertaintybased methods, which rely on signals such as softmax confidence (Hendrycks & Gimpel, 2017), ensemble variance (Choi et al., 2018), or Bayesian inference (Wang & Aitchison, 2021; Charpentier et al., 2020) to identify anomalous inputs; (ii) distance-based methods (Regmi et al., 2024), which compare test embeddings to in-distribution features, commonly via Mahalanobis distance (Colombo et al., 2022; Lee et al., 2018); (iii) density-based methods (Huang et al., 2022), including flow- and energy-based models (Kumar et al., 2023; Liu et al., 2020), which attempt to estimate likelihoods but have been shown to assign spuriously high likelihoods to OOD data (Nalisnick et al., 2019a); and (iv) representation-learning methods (Wang et al., 2022), including self-supervised and contrastive techniques (Seifi et al., 2024; Hendrycks et al., 2019; Tack et al., 2020), which improve robustness by explicitly shaping feature spaces (Also see reviews in Koh & et. al (2021); Salehi et al. (2022)).

Diffusion models (DMs) (Ho et al., 2020; Song et al., 2020) have emerged as state-of-the-art generative models, achieving high-quality samples across diverse domains. Their success has spurred both architectural advances (Vahdat et al., 2021; Dhariwal & Nichol, 2021; Rombach et al., 2022; Karras et al., 2022) and applications beyond generation, such as imaging inverse problems and medical tasks (Chung et al., 2023; Adib et al., 2023) (see also recent reviews (Daras et al., 2024; Croitoru et al., 2023; Li et al., 2023; Kazerouni et al., 2023)). Crucially, diffusion models are especially relevant for OOD detection because their iterative denoising process does not simply produce samples, but also provides access to score functions that explicitly characterize the data distribution. Early work exploited this property through likelihood- or reconstruction-based scores (Graham et al., 2023; Gao et al., 2023; He et al., 2025). More recent studies have explored structural aspects of the diffusion trajectory, such as score geometry and intermediate representations (Heng et al., 2024; Graham et al., 2023; Liu et al., 2023; Choi et al., 2023). These developments highlight both the promise of diffusion-based OOD methods and the need for principled approaches that move beyond heuristic scoring rules.

Building on recent work in diffusion-based OOD detection, we introduce EigenScore, an unsupervised, feature-based framework for identifying distribution shift. Unlike reconstruction-based methods that measure input–output similarity (Graham et al., 2023) or trajectory-based methods that analyze diffusion-path geometry (Heng et al., 2024), EigenScore leverages the covariance structure of the denoising process to capture uncertainty signals. In particular, DDPM-OOD (Graham et al., 2023) detects distribution shift by measuring reconstruction fidelity, using perceptual or pixel-space discrepancies between the input and its denoised reconstruction as an OOD score. While effective in some regimes, this strategy implicitly assumes that distribution shift manifests primarily through degraded reconstruction quality. In contrast, EigenScore does not depend on reconstruction error magnitude or perceptual similarity; instead, it probes the posterior uncertainty of the denoiser by analyzing the eigenvalue spectrum of the posterior covariance. As shown in Section 3.1, even when OOD inputs admit visually plausible reconstructions—as is common in near-OOD settings—the posterior covariance inflates in structured directions, yielding a consistent and theoretically grounded signal of distribution shift.

Recent work by Kamkari et al. (2024) provides a geometric explanation of the likelihood-based OOD detection paradox by analyzing the local geometry of generative models. Their approach studies the Jacobian of the generative mapping and shows that likelihood scores conflate radial distance to the data manifold with tangential volume effects, leading to misleading OOD behavior. While their method also involves spectral quantities through singular values of a Jacobian-related matrix, the motivation and signal differ fundamentally from ours. EigenScore does not operate on likelihoods or generative mappings; instead, it analyzes the posterior covariance of the diffusion denoiser, capturing predictive uncertainty rather than volume distortion. As a result, EigenScore yields a stable uncertainty-based signal even in regimes where likelihood geometry is known to fail, such as near-OOD settings.

By explicitly linking posterior covariance, estimated from the denoiser’s Jacobian, to distribution mismatch, EigenScore provides an interpretable uncertainty-based signal while remaining practical at scale through a Jacobian-free eigenvalue estimation algorithm. Our theoretical analysis and empirical results demonstrate that applying an in-distribution diffusion model to OOD samples leads to systematic inflation of posterior covariance, enabling stable and discriminative OOD detection. Our contributions are threefold:

• We introduce EigenScore, an unsupervised, feature-based framework for OOD detection in diffusion models. EigenScore leverages the posterior covariance of the denoising process to characterize distribution shift.

• We provide supporting analysis establishing a direct connection between denoising uncertainty from posterior covariance and distribution mismatch, thereby explaining why EigenScore reliably separates InD from OOD samples.

• We conduct extensive experiments on standard OOD benchmarks (CIFAR-10 (C10), CIFAR-100 (C100), SVHN, CelebA, TinyImageNet), showing that EigenScore achieves average state-of-the-art performance and remains notably robust in challenging near-OOD scenarios.

## 2 BACKGROUND

Diffusion Models. Diffusion models learn to generate samples by simulating a gradual denoising process. During training, a clean sample x $\sim p ( { \pmb x } )$ is perturbed by Gaussian noise across timesteps $t = 1 , \cdots , T$ , producing noisy states $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ through the forward Markov chain $p ( \pmb { x } _ { t } | \pmb { x } ) = \mathcal { N } ( \pmb { x } , \sigma _ { t } ^ { 2 } \pmb { \bar { I } } )$ which allows for direct sampling via ${ \pmb x } _ { t } = { \pmb x } + { \pmb z }$ , where $\smash { z \sim \mathcal { N } ( 0 , \sigma _ { t } ^ { 2 } I ) }$

The reverse process is approximated by a denoising network $\ D _ { \theta } ( \pmb { x } _ { t } , t )$ , trained to predict either the clean signal or the injected noise. A standard training objective is mean squared error (MSE):

$$
\mathcal {L} _ {\mathrm{MSE}} (\mathrm{D} _ {\theta}) = \mathbb {E} _ {\boldsymbol {x}, \boldsymbol {x} _ {t}, t} \left[ \left\| \boldsymbol {x} - \mathrm{D} _ {\theta} (\boldsymbol {x} _ {t}, t) \right\| _ {2} ^ {2} \right].\tag{1}
$$

Once trained, the model generates new samples by iteratively denoising from Gaussian noise at $t = T$ back to $t = 0$ . Importantly, Tweedie’s formula (Robbins, 1956; Miyasawa, 1961) connects Gaussian denoising with score estimation, linking the posterior mean to the gradient of the log-density as

$$
\mathsf {D} _ {p} (\boldsymbol {x} _ {t}) = \mathbb {E} _ {p} [ \boldsymbol {x} | \boldsymbol {x} _ {t} ] = \boldsymbol {x} _ {t} + \sigma_ {t} ^ {2} \nabla \log p (\boldsymbol {x} _ {t}),\tag{2}
$$

where $\mathsf { D } _ { p } ( \pmb { x } _ { t } )$ denotes an MMSE estimator trained on samples from distribution $p .$ Here, the gradient is with respect to $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ , and $p ( { \pmb x } _ { t } )$ ) denotes the marginal distribution noisy image

$$
p (\pmb {x} _ {t}) = \int p (\pmb {x} _ {t} | \pmb {x}) p (\pmb {x}) \mathrm{d} \pmb {x} = \int G _ {\sigma_ {t}} (\pmb {x} _ {t} - \pmb {x}) p (\pmb {x}) \mathrm{d} \pmb {x},\tag{3}
$$

where $G _ { \sigma _ { t } }$ denotes the Gaussian density function with standard deviation $\sigma _ { t } \geq 0$ (Vincent, 2011; Raphan & Simoncelli, 2011). This relationship implies that denoising does more than produce samples—it provides access to score functions and posterior statistics of the underlying distribution. In the context of OOD detection, this observation motivates our use of the denoiser’s covariance structure as a principled signal of distribution shift. Diffusion models admit several formulations (e.g., variance-preserving, variance-exploding, and SDE-based), but all share the key property of learning the score function $\nabla _ { \pmb { x } _ { t } }$ log $p ( { \pmb x } _ { t } )$ to guide denoising (Ho et al., 2020; Song et al., 2020; Song & Ermon, 2019; Yang et al., 2023).

Unsupervised OOD detection. Unsupervised OOD detection aims to determine whether a given sample x originates from the same distribution as the training data, using only unlabeled InD samples $\pmb { x } _ { 1 } , \cdot \cdot \cdot , \pmb { x } _ { n } \sim p ( \pmb { x } )$ . The goal is to learn a detector that assigns an OOD score to each input, where higher scores indicate a greater likelihood that x was drawn from a different distribution, such as the OOD density q(x) (Graham et al., 2023; Heng et al., 2024).

Likelihood-based methods. These methods use generative models including VAEs, flows, diffusion models to estimate sample likelihoods, under the assumption that OOD data should receive lower likelihoods (Salimans et al., 2017; Kingma & Dhariwal, 2018; Morningstar et al., 2021; Ding et al., 2025). However, it has been shown that generative models often assign high likelihoods to OOD inputs (Choi et al., 2018; Nalisnick et al., 2019a; Kirichenko et al., 2020). To mitigate this, refined scores have been proposed, including likelihood ratios (Ren et al., 2019), compression corrections (Serrà et al., 2019), WAIC ensembles (Choi et al., 2018), and typicality tests (Nalisnick et al., 2019b). Diffusion-based variants further extend this idea by analyzing statistics across the denoising trajectory (Heng et al., 2024; Livernoche et al., 2024).

Reconstruction-based methods. Another line of work assumes that InD samples reconstruct well, whereas OOD samples do not. Early examples include autoencoders (Zhou & Paffenroth, 2017) and GANs (Schlegl et al., 2019). More recently, diffusion models have been exploited for their strong reconstruction fidelity, leading to perceptual quality scores (Graham et al., 2023), projection regret (Choi et al., 2023), and masked inpainting like LMD (Liu et al., 2023). OOD sample detection can alse be via subspace reconstruction of features or gradients, using PCA (Guan et al., 2023), kernel PCA (Fang et al., 2024), or gradient projections such as GradOrth (Behpour et al., 2023).

Feature-based methods. These approaches distinguish InD from OOD by leveraging learned representations, such as Mahalanobis distance in latent space (Denouden et al., 2018), unsupervised contrastive features (Hendrycks et al., 2019; Bergman & Hoshen, 2020; Tack et al., 2020), or encoder features from invertible models (Ahmadian & Lindsten, 2021). Pretrained feature extractors have also proven effective (Xiao et al., 2020).

![](images/3e65179667d3c4c57db9009f38b4b606ee7c9e17769810087f9b0a4620a725a8.jpg)  
Figure 1: We compare negative log-likelihood (NLL), score norm $\sqrt { \sum _ { t } \| \epsilon _ { \theta } ( \pmb { x } _ { t } , t ) \| _ { 2 } ^ { 2 } }$ , score derivative norm $\begin{array} { r } { \sqrt { \sum _ { t } \| \partial _ { t } \epsilon _ { \theta } ( \pmb { x } _ { t } , t ) \| _ { 2 } ^ { 2 } } } \end{array}$ , and the eigenvalue sum (ours) $\textstyle \sum _ { t , k } \lambda _ { k } ^ { t } ( { \pmb x } _ { t } )$ as OOD detection statistics. Top row: near OOD taskfor C10 (InD) vs. C100, NLL and score-based metricsfail to separate distributions, showing substantial overlap. Bottom row: for C10 (InD) vs. SVHN (OOD), the ordering of metrics inverts—score and derivative norms assign lower values to OOD than InD, making thresholds unreliable. In both settings, our eigenvalue-based metric achieves clear separation and consistently assigns higher scores to OOD samples.

Complementing these approaches, we introduce a new perspective based on posterior covariance in diffusion models, which provides a principled feature for quantifying distribution shift.

## 3 DIFFUSION MODEL FOR OUT-OF-DISTRIBUTION DETECTION

EigenScore is a novel OOD detection method that exploits covariance structure of the denoising diffusion process. Our key insight is that when a diffusion model trained on InD data is applied to OOD inputs, the variance of its denoising predictions inflates, leaving a characteristic signature in the eigenvalue spectrum of the score Jacobian.

To motivate EigenScore, we first revisit why commonly used diffusion-based OOD metrics—likelihood, score norm, and score derivatives—are unreliable (Sec. 3.1). We then show, both theoretically and empirically, that posterior covariance offers a consistent marker of distribution shift (Sec. 3.2), before formalizing EigenScore and its efficient computation (Sec. 3.3).

## 3.1 WHY LIKELIHOOD AND SCORE DYNAMICS ARE INSUFFICIENT

Since diffusion models are trained via a variational lower bound (ELBO), likelihood-based scores such as negative log-likelihood (NLL) are natural candidates for OOD detection. However, likelihood does not necessarily align with semantic structure: diffusion models often emphasize low-level statistics while ignoring higher-level semantics, making NLL unreliable (Nalisnick et al., 2019b; Serrà et al., 2019). Empirically, NLL can even assign higher likelihoods to OOD samples than to InD ones (Heng et al., 2024). As shown in Fig. 1, NLL is not a reliable metric for separating InD from OOD samples.

Beyond likelihood, diffusion-based OOD metrics have also used the score function $\epsilon _ { \theta } ( \pmb { x } _ { t } , t )$ and its temporal derivative $\partial _ { t } \epsilon _ { \theta } ( { \pmb x } _ { t } , t )$ as statistics (Heng et al., 2024). Their norms provide some empirical separation, but they remain unstable. In near-OOD settings (C10 vs. C100), the distributions overlap substantially (Fig.1, top row). In some settings (C10 vs. SVHN), the ordering can invert, with OOD samples receiving lower scores than InD (Fig.1, bottom row). These limitations motivate shifting from scalarized scores toward a covariance-based perspective, where the structure of denoiser variability itself provides a more principled signal of distribution shift

## 3.2 UNCERTAINTY AS A SIGNAL OF DISTRIBUTION SHIFT

We formalize why denoising uncertainty yields a principled OOD signal. Let $p ( { \pmb x } )$ denote InD and $q ( { \pmb x } )$ an OOD distribution. Under Gaussian corruption with variance $\sigma _ { t } ^ { 2 }$ , the KL divergence admits the score-based representation (Song et al., 2021; Kadkhodaie et al., 2024; Shoushtari et al., 2025;

![](images/293370d0826844e843b78c6234e681965b4c3adb37a73e367e30574b4a3c357f.jpg)  
Figure 2: Denoised outputs (left), corresponding uncertainty maps (first principle component) (middle), and violin plots ofthe three largest eigenvaluesfor CelebA dataset (right). Top: clean CelebA image and its noisy variantsfor varying t. Middle: InD model (trained on CelebA) applied to CelebA inputs. Bottom: OOD model (trained on C100) applied to the same inputs. InD models yield sharp reconstructions and localized uncertainty with smaller leading eigenvalues, whereas OOD models produce blurrier outputs, diffuse uncertainty, and inflated eigenvalues—highlighting the eigenvalue spectrum as an indicator ofdistribution shift

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 OOD Detection with EigenScore — Train/Validation (left) and Test (right)

Train/Validation: Select time-steps, aggregation, and compute Z-score stats

Require: Trained DM  $D_{p}$ , train set  $X_{train}$ , K number of eigenvalues, I number of repetition,  $L_{train} = []$ 

1: for  $x \in X_{train}$  do

2: Compute M(x) via Eq. (10)

3: Append M(x) to  $L_{train}$ 

4: end for

5: Compute  $\mu_{t}, \sigma_{t}$  across  $L_{train}$ 

6: Use validation set to tune T and aggregation method (mean/median/none)

7: return ( $T^{*}$ , agg*,  $\{\mu_{t}, \sigma_{t}\}_{t=1}^{T^{*}}$ )

Test: Compute EigenScore

Require: Trained DM  $D_{p}$ , test set  $X_{test}$ , number of eigenvalues K, number of repetitions I, ( $T^{*}$ , agg*,  $\{\mu_{t}, \sigma_{t}\}_{t=1}^{T^{*}}$ ),  $L_{test} = []$ 

1: for  $x \in X_{test}$  do

2: Compute M(x) using  $T^{*}$  and agg*

3:  $z_{t}(x) = \frac{\overline{m}_{t}(x) - \mu_{t}}{\sigma_{t}}$  for  $t = 1, \ldots, T^{*}$ 

4:  $S_{\theta}(x) = \sum_{t=1}^{T^{*}} z_{t}(x)$ 

5: Append  $S_{\theta}(x)$  to  $L_{test}$ 

6: end for

7: return  $L_{test}$ $\triangleright$  OOD scores for all test samples
</div>

$$
\mathrm{D} _ {\mathrm{KL}} (p \parallel q) = \int_ {0} ^ {T} \mathbb {E} _ {\boldsymbol {x}, \boldsymbol {x} _ {t}} \left[ \left\| \nabla \log p (\boldsymbol {x} _ {t}) - \nabla \log q (\boldsymbol {x} _ {t}) \right\| _ {2} ^ {2} \right] \sigma_ {t} \mathrm{d} t.\tag{4}
$$

where $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ is generated by the forward diffusion applied to x. Building on prior analyses of KL divergence in diffusion processes (Shoushtari et al., 2025; Heng et al., 2024), we restate the divergence in terms of denoising error (a derivation is given in App. A.1 for completeness).

Proposition 1. Let $p _ { t }$ and $q _ { t }$ denote the noisy marginals ofInD and OOD distributions generated by theforward diffusion process $( E q . \ ( 3 ) )$ . For MMSE denoisers $\mathsf { D } _ { p } ( \pmb { x } _ { t } ) = \mathbb { E } _ { p } [ \pmb { x } | \pmb { x } _ { t } ]$ and $\mathsf { D } _ { q } ( \pmb { x } _ { t } ) =$ $\dot { \mathbb { E } } _ { q } [ \pmb { x } | \dot { \pmb { x } } _ { t } ]$

$$
\mathrm{D} _ {\mathrm{KL}} (p \parallel q) = \int_ {0} ^ {T} \left[ M S E (\mathrm{D} _ {q}, t) - M S E (\mathrm{D} _ {p}, t) \right] \sigma_ {t} ^ {- 3} \mathrm{d} t
$$

$$
M S E (\mathsf {D} _ {p}, t) = \mathbb {E} \big [ \| \boldsymbol {x} - \mathsf {D} _ {p} (\boldsymbol {x} _ {t}) \| _ {2} ^ {2} \big ]
$$

This proposition, adapted from earlier derivations, shows that KL divergence—and thus distribution shift—can be viewed as the accumulation of excess denoising error incurred when contrasting the optimal MMSE denoiser under q with that under p. Thus, we have $\mathbf { M S E } ( \mathsf { D } _ { p } ; q , t ) \ge \mathbf { M S E } ( \mathsf { D } _ { q } ; q , t )$ for each t and the denoising error of a single InD denoiser $\mathsf { D } _ { p }$ is larger in expectation on OOD inputs, yielding a practical detection signal without access to q.

For the MMSE denoiser $\mathsf { D } _ { p }$ , the mean-squared error admits a law-of-total-variance decomposition (proof in App. A.3):

$$
\operatorname{MSE} \left(\mathrm{D} _ {p}, t\right) = \mathbb {E} \left[ \| \boldsymbol {x} - \mathrm{D} _ {p} \left(\boldsymbol {x} _ {t}\right) \| _ {2} ^ {2} \right] = \mathbb {E} _ {\boldsymbol {x} _ {t}} \left[ \operatorname{tr} \left(\operatorname{Cov} _ {p} [ \boldsymbol {x} \mid \boldsymbol {x} _ {t} ]\right) \right].\tag{5}
$$

Thus, denoising error equals the total posterior variance—the trace of the conditional covariance—averaged over noisy observations at noise level t. Intuitively, $\mathrm { C o v } _ { p } [ \pmb { x } \ | \ \pmb { x } _ { t } ]$ quantifies the spread of plausible clean signals consistent with $\mathbf { \Delta } \mathbf { x } _ { t } .$ . When $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ is informative (e.g., InD or low noise), this spread is small and the MSE remains low; when $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ lies off-manifold, the spread inflates and the MSE increases. In this sense, denoising error corresponds directly to the model’s predictive uncertainty, providing a principled basis for OOD detection and motivating our focus on the spectral structure of posterior covariance in Section 3.3.

While Eq.4 relates distribution shift to differences in score norms, such metrics do not preserve ordering: depending on the dataset and noise scale, OOD samples may appear either larger or smaller than InD ones. By contrast, the MSE-based formulation in Prop.1 guarantees that OOD denoising error is systematically larger in expectation than InD error. This ensures a consistent separation with preserved ordering, whereas score-norm methods such as DiffPath (Heng et al., 2024) capture only relative differences without indicating direction.

## 3.3 EIGENVALUE-BASED UNCERTAINTY ESTIMATION

Section 3.2 established that denoising error equals the total posterior variance and that this variance inflates under distribution shift. We now make this connection operational by expressing posterior covariance through the Jacobian of the MMSE denoiser and analyzing its eigen-spectrum. For the MMSE denoiser $\mathbf { \bar { D } } _ { p } ( \mathbf { \boldsymbol { x } } _ { t } ) = \mathbb { E } [ \mathbf { \boldsymbol { x } } | \mathbf { \boldsymbol { x } } _ { t } ]$ , the posterior covariance admits the Miyasawa identity (Miyasawa, 1961):

$$
\mathrm{Cov} _ {p} [ \pmb {x} | \pmb {x} _ {t} ] = \sigma_ {t} ^ {2} \left(\pmb {I} + \sigma_ {t} ^ {2} \nabla^ {2} \log p (\pmb {x} _ {t})\right) = \sigma_ {t} ^ {2} \nabla \mathsf {D} _ {p} (\pmb {x} _ {t}),\tag{6}
$$

so that

$$
\mathrm{MSE} (\mathrm{D} _ {p}, t) = \mathbb {E} _ {\boldsymbol {x} _ {t}} \left[ \mathrm{tr} \left(\mathrm{Cov} _ {p} [ \boldsymbol {x} | \boldsymbol {x} _ {t} ]\right) \right] = \sigma_ {t} ^ {2} \mathbb {E} _ {\boldsymbol {x} _ {t}} \left[ \mathrm{tr} \left(\nabla \mathrm{D} _ {p} (\boldsymbol {x} _ {t})\right) \right],\tag{7}
$$

Thus, the Jacobian trace measures how much posterior uncertainty remains after observing $\mathbf { \nabla } _ { \mathbf { x } _ { t } : \mathbf { \nabla } }$ small traces indicate concentrated beliefs (InD), while large traces signal inflated uncertainty (OOD). Derivations are provided in Appendix (A.2).

Writing $\Sigma _ { t } ( { \pmb x } _ { t } ) : = \mathrm { C o v } _ { p } [ { \pmb x } | { \pmb x } _ { t } ] = \sigma _ { t } ^ { 2 } \nabla \mathsf { D } _ { p } ( { \pmb x } _ { t } )$ , the covariance is symmetric positive semi-definite and admits the eigen-decomposition $\Sigma _ { t } ( \bar { \boldsymbol { x } } _ { t } ) = U _ { t } \mathrm { d i a g } ( \lambda _ { 1 } ^ { t } , \cdot \cdot \cdot , \lambda _ { n } ^ { t } ) U _ { t } ^ { \top }$ , with nonnegative eigenvalues $\lambda _ { k } ^ { t }$ . Since ${ \mathrm { \bf { \bar { t r } } } } ( \Sigma _ { t } ) = \mathbf { \bar { \Sigma } } \Sigma _ { k } \lambda _ { k } ^ { t }$ , the MSE corresponds to the sum of eigenvalues—the total uncertainty across all principal directions at noise level t:

$$
\operatorname{MSE} \left(\mathrm{D} _ {p}, t\right) = \mathbb {E} _ {\boldsymbol {x} _ {t}} \left[ \operatorname{tr} \left(\operatorname{Cov} _ {p} [ \boldsymbol {x} | \boldsymbol {x} _ {t} ]\right) \right] = \mathbb {E} _ {\boldsymbol {x} _ {t}} \left[ \operatorname{tr} \left(\boldsymbol {\Sigma} _ {t} (\boldsymbol {x} _ {t})\right) \right] = \mathbb {E} _ {\boldsymbol {x} _ {t}} \left[ \sum_ {k = 1} ^ {n} \lambda_ {k} ^ {t} (\boldsymbol {x} _ {t}) \right].\tag{8}
$$

For InD samples, the spectrum is compact with smaller leading eigenvalues, reflecting structured denoising aligned with the training data. For OOD samples, uncertainty spreads across multiple eigen directions, inflating both the spectrum and the trace. This spectral inflation provides a quantitative signal of distribution shift. Figure 2 illustrates this effect: OOD samples exhibit consistently higher uncertainty, confirming the link between spectral inflation and distribution mismatch.

## 3.4 EIGENSCORE AS AN OOD METRIC

While Prop.1 links distribution shift to excess denoising error, it requires two denoisers, whereas in practice we have only a single InD model. Combining Prop.1 with Eq. 7 implies that OOD inputs exhibit inflated posterior covariance, which we can read from its eigenvalues. For each input x and diffusion step t, we compute

$$
m _ {t} (\boldsymbol {x}) = \sum_ {k = 1} ^ {K} \lambda_ {k} ^ {t} (\boldsymbol {x} _ {t}), \quad \boldsymbol {x} _ {t} = \boldsymbol {x} + \sigma_ {t} \boldsymbol {z}, \quad \boldsymbol {z} \sim \mathcal {N} (0, \boldsymbol {I}),\tag{9}
$$

the sum of the top–K eigenvalues across I noise realizations. These are aggregated (e.g., by mean, median, or full set) into $\overline { { m } } _ { t } ( { \pmb x } )$ , yielding the EigenScore feature vector

$$
\mathbf {M} (\boldsymbol {x}) = \left[ \overline {{m}} _ {1} (\boldsymbol {x}), \dots , \overline {{m}} _ {T} (\boldsymbol {x}) \right] ^ {\mathsf {T}}.\tag{10}
$$

We normalize each coordinate using $\mathrm { _ { Z } }$ -scores with statistics $\left( \mu _ { t } , \sigma _ { t } \right)$ computed on training set $z _ { t } ( \pmb { x } ) ~ = ~ ( \overline { { m } } _ { t } ( \pmb { x } ) - \mu _ { t } ) / \sigma _ { t }$ . The final OOD score is the sum of normalized coordinates, with the number of timesteps T tuned for efficiency on validation data.

Direct Jacobian evaluation is costly. Following (Manor & Michaeli, 2024), we estimate the top–K eigenvalues using a Jacobian-free subspace iteration. The method approximates Jacobian–vector products with finite differences of the denoiser $\pmb { v } ^ { + } \approx ( \mathsf { D } ( \pmb { x } _ { t } + c \pmb { v } ) - \mathsf { D } ( \bar { \pmb { x } } _ { t } - c \pmb { v } ) ) / 2 c$ , where $c \ll 1$ is the linear approximation constant, v is the current principle component, and $v ^ { + }$ in the next principle component. We then orthogonalizes directions via QR. After a few iterations, eigenvalues are obtained as

$$
\lambda_ {k} ^ {t} (\pmb {x} _ {t}) \approx \frac {\sigma_ {t} ^ {2}}{2 c} \left\| \mathsf {D} (\pmb {x} _ {t} + c \pmb {v} ^ {k}) - \mathsf {D} (\pmb {x} _ {t} - c \pmb {v} ^ {k}) \right\| _ {2}.\tag{11}
$$

Full pseudo-code is given in App. B.1.

## 3.5 EIGENSCORE VS. MSE: CAPTURING STRUCTURE INSTEAD OF COLLAPSE

Eq. (4) and Prop. 1 show that distribution shift manifests as excess denoising error, measurable through the posterior covariance. However, scalar metrics such as MSE or score norms collapse this uncertainty into a single number, discarding how variance is distributed across directions. At high noise levels, this collapse is particularly problematic: many small eigenvalues are dominated by isotropic noise, obscuring class-specific structure. The following lemma formalizes this effect.

Lemma 1. $L e t p ( \mathbf { x } _ { t } ) = p * N ( 0 , \sigma _ { t } ^ { 2 } I )$ denotes the noisy marginal in Eq. (3) and let $\begin{array} { r l } { \pmb { \Sigma } _ { t } ( \pmb { x } _ { t } ) } & { { } = } \end{array}$ $\sigma _ { t } ^ { 2 } \bigl ( I + \sigma _ { t } ^ { 2 } \nabla ^ { 2 } \log p ( { \pmb x } _ { t } ) \bigr )$ from Eq. (6). As $\sigma _ { t } \to \infty , \| \nabla ^ { 2 } \log p ( \pmb { x } _ { t } ) \| \to 0$ uniformly on compact sets. Hence

$$
\Sigma_ {t} (\boldsymbol {x} _ {t}) = \sigma_ {t} ^ {2} \boldsymbol {I} + o (\sigma_ {t} ^ {2}),
$$

so all eigenvalues satisfy $\lambda _ { k } ^ { t } ( { \pmb x } _ { t } ) = \sigma _ { t } ^ { 2 } + o ( \sigma _ { t } ^ { 2 } )$

Lemma 1 implies that the spectrum flattens under heavy Gaussian smoothing: all directions approach the same variance $\sigma _ { t } ^ { 2 } .$ , so low-variance components lose discriminative information. Consequently, MSE and score norms aggregate mostly isotropic noise, rather than meaningful structure (proof in App. A.4). To retain the informative structure, we focus on the dominant modes. Ky Fan’s theorem guarantees that the top-K eigenvalues capture the maximal variance among all K-dimensional projections:

Proposition 2 (Ky Fan’s theorem (Fan, 1950)). Let $\Sigma _ { t } ( \pmb { x } _ { t } ) \succeq 0$ have eigenvalues $\lambda _ { 1 } ^ { t } \geq \cdots \geq \lambda _ { n } ^ { t }$ with eigenvectors forming $U _ { t } = [ \pmb { u } _ { 1 } ^ { t } , \dots , \pmb { u } _ { K } ^ { t } ] .$ . For any $K \in \left\{ 1 , \ldots , n \right\}$

$$
\max _ {\boldsymbol {V} \in \mathbb {R} ^ {n \times K}: \boldsymbol {V} ^ {\top} \boldsymbol {V} = \boldsymbol {I} _ {K}} \operatorname{tr} \left(\boldsymbol {V} ^ {\top} \boldsymbol {\Sigma} _ {t} (\boldsymbol {x} _ {t}) \boldsymbol {V}\right) = \sum_ {k = 1} ^ {K} \lambda_ {k},
$$

and a maximizer is $V ^ { \star } = [ { \pmb u } _ { 1 } ^ { t } , \ldots , { \pmb u } _ { K } ^ { t } ] .$

By retaining only the top-K eigenvalues, EigenScore preserves the most informative uncertainty directions while discarding noise-dominated components. This explains its consistent advantage over MSE and score-based metrics. The choice of K is not dictated by theory, but reflects a practical trade-off between capturing discriminative information and computational efficiency (see App. A.5 for proof). Reconstruction-based diffusion methods such as DDPM-OOD (Graham et al., 2023) can be interpreted as relying on scalarized denoising error, either in pixel space or perceptual feature space. As with MSE, these approaches collapse uncertainty into a single value, discarding how variance is distributed across directions. EigenScore differs fundamentally by retaining the dominant spectral modes of posterior covariance, which remain discriminative even when reconstruction error alone becomes unreliable. We further discuss the implications of using learned denoisers, whose Jacobians need not be strictly SPD, and their relation to spectral truncation in Appendix B.2.

## 4 EXPERIMENTS

We now evaluate the effectiveness of EigenScore for OOD detection. Specifically, we benchmark EigenScore across a suite of pairwise OOD detection tasks and compare its performance against state-of-the-art baselines.

Datasets. We evaluate OOD detection on standard image benchmarks commonly used with diffusion models: C10 (Krizhevsky, 2009), SVHN (Netzer et al., 2011), C100 (Krizhevsky, 2009), and CelebA (Liu et al., 2015a;b). For near-OOD tasks (Yang et al., 2022), we additionally include TinyImageNet (Le & Yang, 2015). Further details can be found in Appendix (B.5)

Table 1: Main OOD detection results (AUROC). Comparison of EigenScore with likelihood-based, reconstruction-based, and diffusion-based baselines across multiple InD–OOD dataset pairings (CelebA, C10, C100, SVHN). Best and second best are highlighted. Note that EigenScore achieves the best average performance and is either best or second best in most settings.

<table><tr><td rowspan="2">InDOOD</td><td colspan="3">CelebA vs.</td><td colspan="3">C10 vs.</td><td colspan="3">C100 vs.</td><td colspan="3">SVHN vs.</td><td rowspan="2">Avg</td></tr><tr><td>C10</td><td>C100</td><td>SVHN</td><td>C100</td><td>SVHN</td><td>CelebA</td><td>C10</td><td>CelebA</td><td>SVHN</td><td>C10</td><td>C100</td><td>CelebA</td></tr><tr><td>DoS</td><td>0.630</td><td>0.615</td><td>0.808</td><td>0.504</td><td>0.752</td><td>0.456</td><td>0.491</td><td>0.520</td><td>0.777</td><td>0.911</td><td>0.904</td><td>0.956</td><td>0.693</td></tr><tr><td>TT</td><td>0.676</td><td>0.655</td><td>0.773</td><td>0.558</td><td>0.714</td><td>0.469</td><td>0.538</td><td>0.464</td><td>0.648</td><td>0.957</td><td>0.961</td><td>0.994</td><td>0.701</td></tr><tr><td>WAIC</td><td>0.589</td><td>0.569</td><td>0.793</td><td>0.476</td><td>0.760</td><td>0.469</td><td>0.502</td><td>0.530</td><td>0.782</td><td>0.978</td><td>0.974</td><td>0.955</td><td>0.698</td></tr><tr><td colspan="14">Diffusion-based</td></tr><tr><td>NLL</td><td>0.507</td><td>0.671</td><td>0.753</td><td>0.558</td><td>0.545</td><td>0.599</td><td>0.480</td><td>0.484</td><td>0.481</td><td>0.635</td><td>0.660</td><td>0.636</td><td>0.584</td></tr><tr><td>IC</td><td>0.510</td><td>0.673</td><td>0.755</td><td>0.552</td><td>0.540</td><td>0.583</td><td>0.460</td><td>0.466</td><td>0.469</td><td>0.625</td><td>0.653</td><td>0.625</td><td>0.576</td></tr><tr><td>DDPM-OOD</td><td>0.922</td><td>0.928</td><td>0.992</td><td>0.618</td><td>0.944</td><td>0.642</td><td>0.462</td><td>0.496</td><td>0.870</td><td>0.963</td><td>0.972</td><td>0.996</td><td>0.817</td></tr><tr><td>LMD</td><td>0.886</td><td>0.848</td><td>0.950</td><td>0.601</td><td>0.821</td><td>0.834</td><td>0.569</td><td>0.595</td><td>0.748</td><td>0.780</td><td>0.749</td><td>0.872</td><td>0.771</td></tr><tr><td>DiffPath (CelebA)</td><td>1.000</td><td>1.000</td><td>0.964</td><td>0.554</td><td>0.729</td><td>0.885</td><td>0.475</td><td>0.887</td><td>0.724</td><td>0.919</td><td>0.941</td><td>0.328</td><td>0.784</td></tr><tr><td>DiffPathV2 (CelebA)</td><td>1.000</td><td>0.995</td><td>0.969</td><td>0.535</td><td>0.812</td><td>0.862</td><td>0.483</td><td>0.513</td><td>0.724</td><td>0.969</td><td>0.975</td><td>0.883</td><td>0.810</td></tr><tr><td>EigenScore (Ours)</td><td>0.965</td><td>0.944</td><td>0.888</td><td>0.880</td><td>0.810</td><td>0.873</td><td>0.642</td><td>0.427</td><td>0.661</td><td>0.992</td><td>0.982</td><td>0.994</td><td>0.838</td></tr></table>

Baselines. We compare our method against several generative baselines for OOD detection, including Improved CD (Du et al., 2021), DoS (Morningstar et al., 2021), TT (Nalisnick et al., 2019b), WAIC (Choi et al., 2018), NLL, IC, DDPM-OOD (Graham et al., 2023), LMD (Liu et al., 2023), DiffPathV2 (Abdi et al., 2025), and DiffPath (Heng et al., 2024). Details regarding the baselines can be found in Appendix (B.4).

Table 2: Near-OOD detection results (AUROC). We evaluate on semantically related datasets, including C10 vs. C100 and TinyImageNet, which are particularly challenging due to shared low-level statistics between InD and OOD samples. The best and second best methods are highlighted. EigenScore achieves the best average performance across both tasks, with a clear margin over prior diffusion-based approaches.

<table><tr><td rowspan="2">InDOOD</td><td colspan="2">C10 vs.</td><td colspan="2">C100 vs.</td><td rowspan="2">Avg</td></tr><tr><td>C100</td><td>TinyImageNet</td><td>C10</td><td>TinyImageNet</td></tr><tr><td>DDPM-OOD</td><td>0.618</td><td>0.570</td><td>0.462</td><td>0.457</td><td>0.527</td></tr><tr><td>LMD</td><td>0.601</td><td>0.592</td><td>0.569</td><td>0.558</td><td>0.580</td></tr><tr><td>DiffPath</td><td>0.554</td><td>0.993</td><td>0.475</td><td>0.995</td><td>0.754</td></tr><tr><td>EigenScore</td><td>0.884</td><td>0.973</td><td>0.652</td><td>0.888</td><td>0.849</td></tr></table>

## 4.1 MAIN RESULTS

Table 1 reports AUROC results across all dataset pairs. EigenScore achieves the highest average performance across all settings and is best or second best on nearly every dataset pair. Its advantage is most pronounced in the challenging near–OOD regime (CIFAR-10 vs. CIFAR-100), where likelihood based scores and trajectory metrics often fail. For example, EigenScore improves AUROC by up to 2% over the best diffusion-based baseline, consistent with our theoretical claim that retaining leading eigenvalues preserves discriminative structure.

On the near-OOD task (Yang et al., 2022) (C10 vs. C100/TinyImageNet), EigenScore continues to deliver strong separation, achieving higher average AUROC than other diffusion-based metrics in Table 2. In particular, while baselines such as NLL, IC, and WAIC (C10 vs. C100) struggle to distinguish the closely related distributions, EigenScore consistently maintains reliable performance, showing its robustness even under challenging near-OOD settings. Notably, EigenScore outperforms DDPM-OOD, where reconstruction quality remains high but posterior uncertainty inflates.

## 4.2 ABLATIONS

Number of timesteps. The parameter T determines how many points along the diffusion trajectory contribute to the score. Larger T includes more noise levels but increases computation and eventually saturates, since high noise merely lifts all eigenvalues uniformly (Lemma 1) without adding discriminative power. As shown in Table 3, even a small budget (e.g., T=5) achieves nearly the same AUROC as larger T, with only marginal improvements from denser schedules. This confirms that a compact subset of timesteps captures most of the useful information, balancing accuracy and efficiency.

Table 3: Ablation on timesteps T, repetitions I, and number of eigenvalues K. AUROC performance of EigenScore across InD–OOD dataset pairs.

<table><tr><td rowspan="2">Timesteps</td><td colspan="3">CelebA vs.</td><td colspan="3">C10 vs.</td><td colspan="3">C100 vs.</td><td colspan="3">SVHN vs.</td><td rowspan="2">Avg</td></tr><tr><td>C10</td><td>C100</td><td>SVHN</td><td>C100</td><td>SVHN</td><td>CelebA</td><td>C10</td><td>CelebA</td><td>SVHN</td><td>C10</td><td>C100</td><td>CelebA</td></tr><tr><td>5</td><td>0.964</td><td>0.943</td><td>0.893</td><td>0.869</td><td>0.778</td><td>0.866</td><td>0.635</td><td>0.448</td><td>0.648</td><td>0.990</td><td>0.977</td><td>0.995</td><td>0.834</td></tr><tr><td>7</td><td>0.963</td><td>0.942</td><td>0.878</td><td>0.848</td><td>0.703</td><td>0.862</td><td>0.619</td><td>0.500</td><td>0.603</td><td>0.988</td><td>0.975</td><td>0.996</td><td>0.823</td></tr><tr><td>10</td><td>0.952</td><td>0.931</td><td>0.850</td><td>0.819</td><td>0.627</td><td>0.867</td><td>0.578</td><td>0.589</td><td>0.521</td><td>0.987</td><td>0.975</td><td>0.998</td><td>0.808</td></tr><tr><td rowspan="2">Repetitions I</td><td colspan="3">CelebA vs.</td><td colspan="3">C10 vs.</td><td colspan="3">C100 vs.</td><td colspan="3">SVHN vs.</td><td rowspan="2">Avg</td></tr><tr><td>C10</td><td>C100</td><td>SVHN</td><td>C100</td><td>SVHN</td><td>CelebA</td><td>C10</td><td>CelebA</td><td>SVHN</td><td>C10</td><td>C100</td><td>CelebA</td></tr><tr><td>5</td><td>0.959</td><td>0.940</td><td>0.885</td><td>0.863</td><td>0.773</td><td>0.857</td><td>0.635</td><td>0.455</td><td>0.649</td><td>0.990</td><td>0.977</td><td>0.995</td><td>0.832</td></tr><tr><td>10</td><td>0.962</td><td>0.942</td><td>0.889</td><td>0.867</td><td>0.776</td><td>0.864</td><td>0.635</td><td>0.450</td><td>0.650</td><td>0.990</td><td>0.978</td><td>0.995</td><td>0.833</td></tr><tr><td>15</td><td>0.962</td><td>0.942</td><td>0.891</td><td>0.869</td><td>0.778</td><td>0.866</td><td>0.635</td><td>0.447</td><td>0.649</td><td>0.990</td><td>0.977</td><td>0.994</td><td>0.833</td></tr><tr><td>20</td><td>0.964</td><td>0.943</td><td>0.893</td><td>0.869</td><td>0.778</td><td>0.866</td><td>0.635</td><td>0.448</td><td>0.648</td><td>0.990</td><td>0.977</td><td>0.995</td><td>0.834</td></tr><tr><td rowspan="2">Eigenvalues K</td><td colspan="3">CelebA vs.</td><td colspan="3">C10 vs.</td><td colspan="3">C100 vs.</td><td colspan="3">SVHN vs.</td><td rowspan="2">Avg</td></tr><tr><td>C10</td><td>C100</td><td>SVHN</td><td>C100</td><td>SVHN</td><td>CelebA</td><td>C10</td><td>CelebA</td><td>SVHN</td><td>C10</td><td>C100</td><td>CelebA</td></tr><tr><td>1</td><td>0.968</td><td>0.950</td><td>0.945</td><td>0.871</td><td>0.803</td><td>0.830</td><td>0.639</td><td>0.432</td><td>0.713</td><td>0.983</td><td>0.966</td><td>0.983</td><td>0.840</td></tr><tr><td>2</td><td>0.967</td><td>0.946</td><td>0.919</td><td>0.872</td><td>0.793</td><td>0.852</td><td>0.636</td><td>0.439</td><td>0.679</td><td>0.987</td><td>0.972</td><td>0.991</td><td>0.838</td></tr><tr><td>3</td><td>0.964</td><td>0.943</td><td>0.893</td><td>0.869</td><td>0.778</td><td>0.866</td><td>0.635</td><td>0.448</td><td>0.648</td><td>0.990</td><td>0.977</td><td>0.995</td><td>0.834</td></tr></table>

Table 4: Comparison ofMSE vs. EigenScore. Average AUROC across all InD–OOD dataset pairs. EigenScore consistently outperforms MSE by leveraging spectral structure of the uncertainty.

<table><tr><td rowspan="2">Method</td><td colspan="3">CelebA vs.</td><td colspan="3">C10 vs.</td><td colspan="3">C100 vs.</td><td colspan="3">SVHN vs.</td><td rowspan="2">Avg</td></tr><tr><td>C10</td><td>C100</td><td>SVHN</td><td>C100</td><td>SVHN</td><td>CelebA</td><td>C10</td><td>CelebA</td><td>SVHN</td><td>C10</td><td>C100</td><td>CelebA</td></tr><tr><td>MSE</td><td>0.804</td><td>0.783</td><td>0.220</td><td>0.629</td><td>0.184</td><td>0.841</td><td>0.552</td><td>0.675</td><td>0.147</td><td>0.994</td><td>0.994</td><td>1.000</td><td>0.652</td></tr><tr><td>EigenScore</td><td>0.964</td><td>0.943</td><td>0.893</td><td>0.869</td><td>0.778</td><td>0.866</td><td>0.635</td><td>0.448</td><td>0.648</td><td>0.990</td><td>0.977</td><td>0.995</td><td>0.834</td></tr></table>

Number of repetitions. The parameter I sets how many noise draws are averaged per timestep. Larger I reduces variance in the estimated eigenvalues and stabilizes OOD scores, but also increases runtime. As shown in Table 3, small values (e.g., I=5) are already sufficient, with only marginal gains beyond I=15. The results are reported with timesteps t ∈ {100, 150, 200, 250, 300}, mean aggregation, and K=3 eigenvalues.

Number of eigenvalues. The parameter K determines how many leading eigenvalues are aggregated at each timestep. Table 3 shows that K=1 achieves the best average performance, though in some settings K=3 yields slightly higher AUROC. This indicates that the bulk of discriminative information resides in the first few modes, consistent with Lemma 1. The results are reported with the same timestep schedule, mean aggregation, and I=20 repetitions.

![](images/8e3c2e7fd131155cdf3b4d7ed8c8286273db564ed1476ecbfb273b3387fdfd95.jpg)  
Figure 3: Ablation on eigenvalue informativeness across t. Performance declines with increasing noise, consistent with Lem. 1, while $\lambda _ { 1 }$ retains the strongest OOD signal compared to $\lambda _ { 2 }$ and $\lambda _ { 3 }$ supporting Prop. 2.

EigenScore vs. MSE. EigenScore consistently outperforms MSE by retaining the spectral structure of posterior covariance rather than collapsing uncertainty into a single scalar. As shown in Table 4, this leads to higher AUROC across diverse dataset pairs. All experiments here use 20 repetitions, timesteps t ∈ {100, 150, 200, 250, 300}, and mean aggregation.

To further examine Lemma 1 and Prop. 2, we analyze the informativeness of individual eigenvalues across noise levels. Lemma 1 predicts that at larger t, eigenvalues converge toward $\sigma _ { t } ^ { 2 }$ , diminishing their discriminative value. Figure 3 confirms this: AUROC with $\lambda _ { 1 }$ decreases gradually as t increases, while $\lambda _ { 2 }$ and $\lambda _ { 3 }$ deteriorate more sharply. Prop. 2 further implies that the leading eigenvalues capture the most informative variance. Consistently, $\lambda _ { 1 }$ provides the strongest OOD signal, followed by $\lambda _ { 2 }$ and then $\lambda _ { 3 }$ . These results highlight why focusing on dominant modes, as done in EigenScore, yields more stable and informative detection than scalarized measures such as MSE.

## 5 CONCLUSION

We introduced EigenScore, a principled OOD detection method for diffusion models that leverages the spectral structure of denoising uncertainty. By linking KL divergence to excess denoising error and showing that posterior covariance inflation consistently signals distribution shift, EigenScore offers both theoretical justification and strong empirical performance. Across diverse benchmarks, EigenScore consistently outperformed likelihood-based and score-norm methods, with particularly robust gains in near-OOD settings where traditional approaches fail. Ablation studies further confirmed that most discriminative information lies in the leading eigenvalues at moderate noise levels, validating our spectral perspective.

## LIMITATIONS

Despite its strengths, our approach has several limitations. First, EigenScore only leverages a subset of the information available in diffusion models: we use the eigenvalues of the posterior covariance but discard eigenvector structure, which may contain additional discriminative cues. Moreover, we compute features at a limited set of timesteps rather than across the full diffusion trajectory, potentially overlooking temporal dynamics of uncertainty. Second, our framework focuses on the magnitude of eigenvalues, but does not explicitly exploit their rate of change across eigenvalues, which itself may differ systematically between InD and OOD inputs and could serve as a detection signal.

## REPRODUCIBILITY STATEMENT

We have taken several steps to ensure the reproducibility of our results. All datasets used in our experiments (CIFAR-10, CIFAR-100, SVHN, CelebA, TinyImageNet) are publicly available. We provide detailed descriptions of our training and evaluation protocols, including the diffusion model architecture, noise schedules, hyperparameters, and aggregation strategies. Experimental results are averaged across multiple random seeds, and we report the effect of varying key parameters (number of timesteps, repetitions, and eigenvalues) in Section 4.2. To further facilitate reproducibility, we have released the full anonymous source code and scripts for running experiments in supplement B.5.

## LLMS USAGE STATEMENT

During the preparation of this manuscript, we made limited use of large language models (LLMs), specifically OpenAI’s ChatGPT, to assist with language refinement and organization of some sections. All technical content, equations, derivations, and experimental design were developed entirely by the authors. The LLM was not used for ideation of methods, data analysis, or generation of results.

## ACKNOWLEDGMENTS

This paper is partially based on work supported by the NSF under CAREER Awards CCF-2043134, 2046293 and Grants CCF-2504613, 2504614.