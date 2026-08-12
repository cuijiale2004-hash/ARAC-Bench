## ABSTRACT

Classifier-Free Guidance (CFG) is a widely used technique for conditional generation and improving sample quality in continuous diffusion models, and its extensions to discrete diffusion has recently started to be investigated. In order to improve the algorithms in a principled way, this paper starts by analyzing the exact effect of CFG in the context of a low-dimensional masked diffusion model, with a special emphasis on the guidance schedule. Our analysis shows that high guidance early in sampling (when inputs are heavily masked) harms generation quality, while late-stage guidance improves it. These findings provide a theoretical explanation for empirical observations in recent studies on guidance schedules. The analysis also reveals an imperfection of the current CFG implementations. These implementations can unintentionally cause imbalanced transitions, such as unmasking too rapidly during the early stages of generation, which degrades the quality of the resulting samples. To address this, we draw insight from the analysis and propose a novel classifier-free guidance mechanism. Intuitively, our method smooths the transport between the data distribution and the initial (masked) distribution, resulting in improved sample quality. Remarkably, our method is achievable via a simple one-line code change. Experiments on conditional image and text generation empirically confirm the efficacy of our method.

![](images/bf84251cf54fa7588a7e1fee166626763aede7abb7ce79298f98447c3772e912.jpg)  
Figure 1: We proposed an improved guidance mechanism through column normalization. Our method produces sharper images while being more stable to the guidance strength. Notably, it requires only a minor code modification.

## 1 INTRODUCTION

Continuous-state diffusion models (Ho et al., 2020; Song et al., 2021) have proven effective in both unconditional and conditional generation tasks, such as generating data from natural language prompts. Prominent examples include text-to-image and text-to-video models like Stable Diffusion, Sora, and others (Rombach et al., 2022; Esser et al., 2024; Brooks et al., 2024). More recently, progress in discrete diffusion modeling (Campbell et al., 2022; Lou et al., 2023; Huang et al., 2023; Gruver et al., 2023; Ou et al., 2024; Shi et al., 2024; Sahoo et al., 2024) has extended the applicability of diffusion-based generation to new domains, including molecular design, protein synthesis, and languages.

Despite their success, these models often produce outputs that lack fine detail or strong alignment with conditioning inputs (e.g., text prompts). A widely adopted technique to address this issue is classifier-free guidance (CFG) (Ho & Salimans, 2021), which improves fidelity but typically at the cost of reduced sample diversity (Karras et al., 2024).

A growing body of work has sought to understand the theoretical foundations of CFG in diffusion models (Chidambaram et al., 2024; Pavasovic et al., 2025; Bradley & Nakkiran, 2024; Ye et al., 2025), while others have developed improved guidance algorithms (Karras et al., 2024; Li et al., 2024). Classifier-free guidance has also been adapted to discrete diffusion models (Nisonoff et al., 2024; Schiff et al., 2024), yielding promising empirical gains.

Among these improvements, dynamic guidance schedules—where guidance strength varies over the generation trajectory—have shown especially effective. Strategies such as guidance intervals (Kynkäänniemi et al., 2024) and gradually increasing schedules (Xi et al., 2024) can significantly enhance sample quality and are increasingly adopted in practice (Hoogeboom et al., 2024; Yu et al., 2024; Karras et al., 2024). However, such scheduling techniques remain exclusive to the continuous setting.

While recent adaptations of CFG to discrete diffusion have improved empirical performance, defining and optimizing effective guidance strategies in discrete spaces remains a fundamentally challenging and open research problem.

In our work we aim to better understand the mechanisms by which guidance affects the sampling process in discrete diffusion. Specifically, we aim to answer the following questions:

\- How does the guidance schedule affect the distribution of the generated samples?

\- Is it possible to characterize properties of good guidance schedules?

To do so, we start by deriving explicit formulas for the sampled distribution under varying guidance schedules in 1 and 2 dimensions. Our analysis not only reveals flaws in current CFG implementations, but also leads to effective design principles for effective guidance schedules in masked diffusion. Our contributions can be summarized as:

\- We identify a key flaw in existing discrete guidance mechanisms that complicates simulation, and provide a theoretical explanation of its cause.

\- To address the flaw, we propose a novel classifier-free guidance mechanism based on a simple yet principled column normalization of the rate matrix. This change is theoretically justified, easy to implement (pseudocode in Sec.1), and compares favorably to existing approaches in practice.

\- The first theoretical justifications to characterize guidance schedules and the mechanisms by which they improve sample generation

```python
def normalized_guidance_euler_transition(
    x, c, t, dt, w
):
uncond = model(x, cond=None)
cond = model(x, cond=c)
logits = w * cond + (1 - w) * uncond

p_theta = logits.softmax(dim=-1)

s, s_bar = sigma(t), sigma_bar(t)
change = dt * s * (1 - exp(-s_bar))
return sample(delta(x) + change * p_theta)

def other_guidance_euler_transition(
    x, c, t, dt, w
):
uncond = model(x, cond=None)
cond = model(x, cond=c)
logits = w * cond + (1 - w) * uncond

p_theta = logits.exp()

s, s_bar = sigma(t), sigma_bar(t)
change = dt * s * (1 - exp(-s_bar))
return sample(delta(x) + change * p_theta)
```  
Listing 1: Our guidance in the special case of masked diffusion using Euler transitions. Our method is a simple one line change but clearly motivated by theory  
Listing 2: Unlocking/Simple guidance for the special case of masked diffusion using Euler transitions.

## 2 PRELIMINARIES

This paper considers a vocabulary of size M and state space $S = \{1, 2, \ldots, M\}^{d}$ , with each element being a sequence of tokens. The number of tokens d will also be referred to as the dimension. Each probability distribution on S is represented as a vector in $R^{M^{d}}$ whose entries sum to one.

## 2.1 INTRODUCTION TO DISCRETE DIFFUSION VIA CTMC

Given an initial distribution $p \in R^{M^{d}}$ , discrete diffusion is defined by considering a rate matrix $R_{t} \in R^{M^{d} \times M^{d}}$ and defining a continuous time Markov chain (CTMC):

$$
\frac {\mathrm{d} p _ {t}}{\mathrm{d} t} = R _ {t} p _ {t}, \quad p _ {0} = p.\tag{1}
$$

we pick $R_{t}$ such that when $t \to \infty$ , $p_{t}$ converges to a simple distribution. Additionally, $R_{t}$ must satisfy that its non-diagonal entries are non-negative and each column must add up to zero. The time reversal of this process corresponds to a different CTMC given by:

$$
\frac {\mathrm{d} p _ {T - t}}{\mathrm{d} t} = \overline {{R}} _ {T - t} p _ {T - t}.\tag{2}
$$

This process is considered as the time reversal since it has the same law as (1) for all values of $t$ and the reverse transition matrix can be found through the following identities:

$$
\overline {{R}} _ {t} (y, x) = R _ {t} (x, y) \cdot \frac {p _ {t} (y)}{p _ {t} (x)}, \qquad \overline {{R}} _ {t} (x, x) = - \sum_ {y \neq x} \overline {{R}} _ {t} (y, x).\tag{3}
$$

The ratios $\frac{p_{t}(y)}{p_{t}(x)}$ are called the concrete score and they enable sampling through Euler schemes, $\tau$ -leaping (Lou et al., 2023) or higher order methods (Ren et al., 2025).

Masked Discrete Diffusion is a special case of diffusion where a clean sequence $x_{0}$ is gradually corrupted over time by randomly masking some of its entries. Typically, the forward process is chosen such that at time t = 0, the data is completely unmasked, and at t = T the data is completely masked. Formally, the distribution of each token can be written in a simple form:

$$
p _ {t} (x _ {t} ^ {i} | x _ {0}) = \left\{ \begin{array}{l l} x _ {0} ^ {i} & \quad \text { with   probability } e ^ {- \overline {{\sigma}} _ {t}} \\ M & \quad \text { with   probability } 1 - e ^ {- \overline {{\sigma}} _ {t}} \end{array} \right.
$$

Where $\overline{\sigma}_{t}$ is an increasing function that defines the unmasking schedule. The forward dynamics are defined such that tokens transition only from a clean state to a masked state, remaining masked thereafter. Generation is achieved by starting from a fully masked state and iteratively unmasking tokens until a clean sequence is recovered by following Equation (2).

Masked diffusion enjoys a simple and structured design, which has enabled its successful scaling to large practical tasks (Nie et al., 2025; Xie et al., 2025; Ou et al., 2024; Sahoo et al., 2024; Shi et al., 2024; Campbell et al., 2022). For this reason, we adopt it as the primary setting for our analysis.

## 2.2 CLASSIFIER-FREE GUIDANCE

Classifier-free guidance (CFG) (Ho & Salimans, 2021) was introduced to improve conditional diffusion models, like generating images from class labels or text. Models often failed to capture fine details, which led to less accurate and misaligned samples (Karras et al., 2024).

CFG tackles this by comparing predictions with and without conditioning, and biasing generation toward the conditional signal. Formally, the method defines a reweighted distribution:

$$
p ^ {(w)} (x | y) \propto p ^ {w} (x | y) p ^ {1 - w} (x)
$$

Where w is called the guidance strength. Setting w = 1 recovers the usual conditional distribution $p(x|y)$ while w = 0 corresponds to unconditional sampling. The crucial insight is that by setting w > 1 it is possible to emphasize the conditional part, effectively pulling the generation closer to satisfying the required condition. CFG is now a standard tool in conditional diffusion models, more controllable generations across tasks such as text-to-image synthesis.

While the original formulation contrasted the conditional model against its unconditional counterpart, later works recognized that this can be extended by replacing the unconditional distribution with other distributions. For example, Karras et al. (2024) used a weaker conditional model as the guiding distribution. This view has led to the understanding that the essence of guidance lies in balancing a target distribution p with a guiding distribution q.

$$
p ^ {(w)} (x) \propto p ^ {w} (x) q ^ {1 - w} (x)\tag{4}
$$

This view highlights that the unconditional model is simply one possible choice of q. By carefully selecting q recent works (Karras et al., 2024; Li et al., 2024; Rojas et al., 2025) have proposed novel guidance strategies that further improve sample quality and control.

## 2.3 GUIDANCE FOR DISCRETE DIFFUSION MODELS

In parallel to advances in continuous domains, discrete diffusion models have emerged as powerful generative models, enabling diffusion-based approaches on modalities that were previously out of reach—most notably, text. Improving the fidelity and controllability of these models is crucial, and guidance offers a natural path forward. Extending classifier-free guidance to the discrete setting has therefore become an active line of research with two main approaches having been proposed, which we describe below, followed by a discussion in Section 3.3 comparing them to our method.

Unlocking Guidance (Nisonoff et al., 2024) introduced the first classifier-free guidance mechanisms for discrete diffusion models. Inspired by the continuous case, they constructed a guided backwards transition by interpolating between two transition matrices in equation 2, yielding

$$
\overline {{R}} _ {t} ^ {(w)} (y, x) = R _ {t} (x, y) \cdot \Big (\frac {p _ {t} (y)}{p _ {t} (x)} \Big) ^ {w} \Big (\frac {q _ {t} (y)}{q _ {t} (x)} \Big) ^ {1 - w}, \qquad \overline {{R}} _ {t} ^ {(w)} (x, x) = - \sum_ {y \neq x} \overline {{R}} _ {t} ^ {(w)} (y, x),\tag{5}
$$

where $p_{t}, q_{t}$ follows the forward CTMC (1). Here $p_{0} = p$ is the distribution that we want to generate from and q serves as the guiding distribution. $^{1}$ . Notice how the products mimic those present in equation 4. A useful way to interpret this is by introducing the notion of the tilted distribution:

$$
p ^ {(w)} (x) = Z _ {w} ^ {- 1} p ^ {w} (x) \cdot q ^ {1 - w} (x), \qquad Z _ {w} = \sum_ {y \in S} p ^ {w} (x) \cdot q ^ {1 - w} (x).
$$

The generation process follows the dynamics induced by the guided transition matrix substituted in equation 2. Nisonoff et al. (2024) showed that guidance in the discrete setting serves a role analogous to its continuous counterpart—steering the model toward more faithful conditional samples—thus providing an important step toward improving the quality of discrete diffusion generations.

Simple Guidance. Concurrently, Schiff et al. (2024) proposed an alternative formulation of classifier-free guidance for discrete diffusion. Rather than interpolating the rate matrices as in

Nisonoff et al. (2024), they directly interpolate the transition probabilities themselves. Specifically, when transitioning from time t to time s < t, the following transition was proposed:

$$
p _ {\text { simple }} ^ {(w)} (z _ {s} | z _ {t}) \propto p ^ {w} (z _ {s} | z _ {t}) p ^ {1 - w} q (z _ {s} | z _ {t}).\tag{6}
$$

As before, increasing w biases towards the target distribution p. Although the construction appears different, in the limit $s \rightarrow t$ the transitions coincide with those of Nisonoff et al. (2024). In practice, however, a finite number of steps is used, and the resulting methods are distinct. To implement these transitions, one can use equation (2) together with a suitable numerical integration scheme.

## 2.4 DYNAMIC GUIDANCE SCHEDULES

In our work we will consider dynamic guidance schedules, i.e. making w a function of time. Such schedules have become more popular in practice. For instance, guidance interval (Kynkäänniemi et al., 2024) only applies guidance on a segment of the generation process. Doing so produces a boost in the performance of diffusion models. However, existing work on dynamic guidance schedules (Kynkäänniemi et al., 2024; Xi et al., 2024) has been limited to a continuous (state-space) diffusion models. It remains unclear whether such schedules are also effective in discrete state diffusion—a question that serves as the main focus of our investigation.

Specifically, this work will consider $w : [0, T] \to R$ , i.e. guidance strength as a function of time, referred to as the guidance schedule. The schedule induces a generative process given by:

$$
\frac {\mathrm{d} p _ {T - t}}{\mathrm{d} t} = \overline {{R}} _ {T - t} ^ {(w _ {T - t})} p _ {T - t}\tag{7}
$$

Understanding which schedules result in the best generation is of crucial importance to further improve the sample accuracy of discrete diffusion models.

## 3 METHODOLOGY

We begin by analyzing the guided process in the simplest case of a single token in Section 3.1, which already reveals a key limitation of existing guidance. We then introduce our proposed remedy in Section 3.2 via column normalization. Afterwards, we analyze the effect of guidance schedules on two tokens in Section 3.4. Finally, we present experimental results of our methods in Section 4.

## 3.1 IDENTIFYING AN ISSUE IN THE GUIDANCE OF DISCRETE DIFFUSION

We start by studying guidance for d = 1 where an exact analysis is possible. The following result characterizes the distribution at time t under constant guidance: Theorem 3.1. (Informal) Along the dynamics of equation (7), starting from a fully masked state, the distribution at time t is given by:

$$
p _ {t} = \left(1 - \left(\frac {1 - e ^ {- \overline {{\sigma}} _ {t}}}{1 - e ^ {- \overline {{\sigma}} _ {T}}}\right) ^ {Z _ {w}}\right) \cdot p ^ {(w)}\tag{8}
$$

We present a full proof, as well as a more general result for varying guidance schedules in Theorem B.1. This shows that for d = 1 the guided process exactly recovers the tilted distribution, with the unmasking speed controlled by the factor in front of $p^{(w)}$ . Although low-dimensional, this result already reveals important properties of the guided backwards process.

Crucially, the partition function $Z_{w}$ appears in the ex-

![](images/0b0382884a3fdf5afd756a8acdb3b364ef8c89491c5d558072c34cbb5a6ed6ec.jpg)  
Figure 2: We plot the unmasking rates as a function of time under guidance. Faster unmasking ( $Z_{w} > 1$ ) leads to worse numerical solvers, demonstrating an issue in the existing guidance mechanism.

ponent of the rate term, meaning that even small changes in w can result in fast changes in the sampling rate. Figure 2 shows the percentage of tokens that remain masked as a function of time $p_{t}(M)$ for different values of $Z_{w}$ . Applying guidance can significantly accelerate unmasking rates. While this can lead to faster generation, it may also introduce stiffness (Rathinam et al., 2003) and inefficiencies if not properly controlled.

## 3.2 IMPROVED GUIDANCE MECHANISMS FOR DISCRETE DIFFUSION VIA COLUMN NORMALIZATION

In order to alleviate the unintentional fast unmasking rates, we propose a simple yet effective change to the guidance mechanism. We begin by isolating the source of the issue by rewriting by decomposing the rate matrix in (2) into a jump rate and a jump distribution:

$$
\overline {{R}} _ {t} (y, x) = r _ {t, p} (x) \underbrace {\left(\frac {1}{r _ {t , p} (x)} \frac {p _ {t} (y)}{p _ {t} (x)} R _ {t} (x , y)\right)} _ {p _ {t} (y | x)}, \qquad r _ {t, p} (x) = \sum_ {y \neq x} \overline {{R}} _ {t} (y, x).\tag{9}
$$

Under this decomposition, $r_{t,p}(x)$ governs how often jumps occur from state $x$ , while $p_t(y \mid x)$ determines where the process jumps. Using this decomposition the guided rate matrix in (5) becomes:

$$
\overline {{R}} _ {t} ^ {(w)} (y, x) = \underbrace {r _ {t , p} ^ {w} (x) r _ {t , p} ^ {1 - w} (x) \mathcal {Z} _ {w}} _ {\text { rate }} \quad \underbrace {\mathcal {Z} _ {w} ^ {- 1} p _ {t} ^ {w} (y | x) q ^ {1 - w} (y | x)} _ {\text { distribution }}\tag{10}
$$

Where $\mathcal{Z}_{w} = \sum_{y \neq x} p_{t}^{w}(y|x) q^{1-w}(y|x)$ is the normalizing constant of the guided jump measure. Crucially, (10) reveals that the $Z_{w}$ does not merely affect the jump distribution, but instead rescales the overall jump rate. This is unintentional and leads to disproportionately fast transitions.

To make this effect explicit, we consider the masked diffusion setting and explicitly write the transition rates between a masked state M a non-masked state:

Lemma 3.1. The transition rates between a masked state and an unnormalized state are given by:

$$
\bar {R} _ {t} ^ {(w)} (y, M) = R _ {t} (x, y) \frac {e ^ {- \bar {\sigma} _ {t}}}{1 - e ^ {- \bar {\sigma} _ {t}}} Z _ {w} p ^ {(w)} (y)
$$

The appearance of $Z_{w}$ as a multiplicative factor in the transition rate confirms the source of the observed pathology: guidance amplifies how often unmasking occurs, rather than only affecting which token is selected. Notably, when w = 1 (the purely conditional setting), $Z_{w} = 1$ and the issue disappears, explaining why this effect is absent in standard conditional diffusion.

Normalized Guidance. The above analysis indicates that the normalizing constant $Z_{w}$ should not influence the jump rate. To correct this, we explicitly decouple rate and distribution by normalizing the guided rate matrix column-wise:

$$
\overline {{R}} _ {t} ^ {(w)} (y, x) = \underbrace {r _ {t , p} ^ {w} (x) r _ {t , p} ^ {1 - w} (x)} _ {\text { rate }} \quad \underbrace {\mathcal {Z} _ {w} ^ {- 1} p _ {t} ^ {w} (y | x) q ^ {1 - w} (y | x)} _ {\text { distribution }}\tag{11}
$$

In the case of masked diffusion this normalization admits a very simple form:

$$
\overline {{R}} _ {\mathrm{nor}, t} ^ {(w)} (\hat {\mathbf {x}}, \mathbf {x}) = \frac {R _ {t} (\mathbf {x} , \hat {\mathbf {x}}) e ^ {- \overline {{\sigma}} _ {t}}}{1 - e ^ {- \overline {{\sigma}} _ {t}}} \operatorname{Softmax} (w \log p _ {0} (\hat {\mathbf {x}} ^ {i} | \mathbf {x} ^ {\mathrm{UM}}) + (1 - w) \log q _ {0} (\hat {\mathbf {x}} ^ {i} | \mathbf {x} ^ {\mathrm{UM}})).\tag{12}
$$

The normalization introduced in $(11)$ and $(12)$ has the effect of smoothing the transport between the starting distribution and the data distribution. This simple change stabilizes the sampling process and allows for a cleaner theory. Notably, this can be done with a simple one line change to the code as presented in the pseudocode in 1. We further elaborate on the experimental benefits on Section 4.

## 3.3 COMPARISON OF GUIDANCE MECHANISMS

We now clarify the distinctions between the various classifier-free guidance mechanisms. While some differences between our method and that of Nisonoff et al. (2024) were already discussed, we further highlight how our formulation also differs from the approach of Schiff et al. (2024). To better understand these differences, we begin by comparing the unlocking guidance mechanism of Nisonoff et al. (2024) with the simple guidance proposed by Schiff et al. (2024). For this analysis, we keep the guidance strength fixed throughout. Notice that: $p(x_{s}|x_{t}) = \exp \left(\int_{s}^{t}\overline{R}_{\tau}^{(w)}\mathrm{d}\tau\right)p_{t}$ . Therefore, if $p_{t}$ denotes the law of $x_{t}$ , we can write the transition probabilities for each method:

$$
p _ {\mathrm{unlocking}} (x _ {s} | x _ {t}) = \exp \Big (\int_ {s} ^ {t} \overline {{R}} _ {\tau} ^ {w} (\cdot | c) \overline {{R}} _ {\tau} ^ {1 - w} (\cdot) \mathrm{d} \tau \Big) p _ {t},
$$

$$
p _ {\text { simple }} (x _ {s} | x _ {t}) = Z _ {\text { simple }} \left(\exp \left(\int_ {s} ^ {t} \overline {{R}} _ {\tau} (\cdot | c) \mathrm{d} \tau\right) p _ {t}\right) ^ {w} \left(\exp \left(\int_ {s} ^ {t} \overline {{R}} _ {\tau} (\cdot) \mathrm{d} \tau\right) p _ {t}\right) ^ {1 - w}.
$$

where $Z_{simple}$ is a normalizing constant. Now we look at the w-dependence inside the exponential. For $\log p_{unlocking}$ , the w-dependence is exponential as it appears in the exponent of the rate matrices, while for $\log p_{simple}$ , the w-dependence is linear. Therefore, the transitions induced by the unlocking guidance method get much more aggressive when w increases. On the other hand, our normalization (depending on w) normalizes the columns so that it maintains the smoothness of the transition when w increases. Therefore, our method approximates the convergence rates of the original process.

## 3.4 ANALYSIS OF GUIDANCE SCHEDULES IN 2D

Having addressed the existing issue we switch our focus to the analysis of guidance schedules in the case of two tokens. Although the analysis can be extended to higher-dimensions, the complexity of the problem grows exponentially with the dimension, leading to increasingly intricate expressions and reduced interpretability. This low-dimensional analysis already reveals the underlying mechanisms that define a good guidance schedule, and its impacts in high-dimensions are remarkable.

We start by stating our main theorem, in a simple to understand case that is used in practice. This simplification doesn't result in loss of generality, but significantly increases the interpretability of the results. We present a more general version in Theorem C.1.

Corollary 3.1. Consider a time partition $0 = t_{0} < t_{1} < t_{2} < t_{3} = T$ with guidance $w_{i}$ in the interval $[t_{i}, t_{i+1})$ . With $\overline{\sigma} = -\log(1 - \delta t)$ and $p_{T}(M, M) = 1$ . Then the sampled distribution follows the following formula:

$$
\begin{array}{r} p _ {t _ {0}} (i, j) = \left(\frac {t _ {3} - t _ {2}}{t _ {3}}\right) ^ {2} p ^ {(w _ {2})} (i, j) + \left(\frac {t _ {2} - t _ {1}}{t _ {3}}\right) ^ {2} p ^ {(w _ {1})} (i, j) + \left(\frac {t _ {1} - t _ {0}}{t _ {3}}\right) ^ {2} p ^ {(w _ {0})} (i, j) \\ + \frac {(t _ {3} - t _ {2}) (t _ {2} - t _ {1})}{t _ {3} ^ {2}} p ^ {(w _ {1}, w _ {2})} (i, j) + \frac {(t _ {3} - t _ {2}) (t _ {1} - t _ {0})}{t _ {3} ^ {2}} p ^ {(w _ {0}, w _ {2})} (i, j) + \frac {(t _ {2} - t _ {1}) (t _ {1} - t _ {0})}{t _ {3} ^ {2}} p ^ {(w _ {0}, w _ {1})} (i, j), \end{array}
$$

where $p^{(w,\gamma)}(i,j) = p^{(w)}(i,j|X_1 = i)p^{(\gamma)}(X_1 = i) + p^{(w)}(i,j|X_2 = j)p^{(\gamma)}(X_2 = j)$ , notice that this is not exactly a probability distribution as it is not normalized, but we will refer to it as one.

This theorem states that guidance schedules induce an interpolation of different distributions, which depend only on the guidance strengths and that the portion assigned to each one depends on the time parameters. We analyze the role of each component separately.

![](images/c72850a8ff404b9b77b42a99df8590f3aa120f69d381f6eac0f74c1bccf84057.jpg)

![](images/13a413913ed2d2248dcfb71ba8e29ee984bcb37474057341fc376c17deeb5105.jpg)  
Figure 5: Notice that when $\omega < \gamma$ the combined distribution doesn't bias the leftmost mode, making this setting less efficient for guidance.

The role of guidance weights: We study a toy example in 2D with 4 clusters, two of which intersect in the middle (see Appendix D for visualizations). Figure 3 shows that increasing w leads to concentration of mass in one of the modes. Similarly, Figure 5 shows that $p^{(w,\gamma)}$ strongly resembles the tilted distribution of w. Practically, this means that the combined distribution will be more similar to the guidance applied at the end of the generation! Therefore, effective schedules have

higher guidance at the final and middle phases of the generation while keeping early guidance small.

![](images/6f7a0fcc685dd19c3eb81da4c69c14ece0335c3184a390229914da7bd1294623.jpg)

![](images/e1df412d890ff6d053f92c708d88b61e7ed26943012c4427df5f525ba1302131.jpg)

![](images/c34846f98ac9265ad58ff664a385ff63479244e09d38c48c4ebd61c20e2b8479.jpg)  
Figure 3: Tilted distributions for varying values of w. Large w concentrates mass on one mode.

![](images/a16e0333b30d14d33760984149bfc0dfe2c6656861089ebb2b4a97fd0b8ceeab.jpg)

![](images/6651f688f7044d861efff1683f7b9908cda532ecd7db6e34fcde6866d68ffff5.jpg)

![](images/13bdcbe86a8e44deb88a09e89a4da02bfb9c0795e317b00705491d3461d307e6.jpg)  
Figure 4: Evolution of the coefficients in Corollary 3.1 for different values of $t_{2}$ , with $t_{1} \leq t_{2}$ . For moderate $t_{2}$ , no single coefficient dominates, yielding a balanced target distribution.

The role of the time parameters: As observed in corollary 3.1, the time parameters set the proportion of each distribution that will contribute towards the final output. As observed in Figures 3.5, biasing just one of the distributions usually results in oversampling from a certain area. A good schedule is one that appropriately balances the contribution of each distribution.

We fix several values of $t_{2}$ and plot the coefficients as a function of $t_{1}$ in Figure 4. When $t_{2} = 1$ . we only have two intervals, and the curves change quickly; this implies that finding the right balance requires more careful tuning. On the other hand when $t_{2} = .75$ , many values of $t_{1}$ result in balanced combinations of all distributions, which ensures that we sample in a balanced way.

Which schedules perform best? Our theoretical analysis provides several insights into the design of effective guidance schedules. As discussed earlier, schedules that apply stronger guidance during the middle and later stages of the sampling process, while keeping early guidance small, tend to perform better. These selections seem to be the most critical, as they govern which distributions are mixed. Moreover, our theory predicts that using all three intervals (early, middle, and late) in the schedule facilitates easier tuning and yields more balanced output distributions. Based on these principles, we evaluate (according to our theory) various guidance schedules for discrete diffusion in Table 1, and we validate these predictions empirically in Section 4.2.

Table 1: Comparison of several guidance schedules.

<table><tr><td></td><td>Low G. Beg</td><td>High G. Mid</td><td>High G. End</td><td># Params Tune</td><td>Difficulty to Tune</td></tr><tr><td>Constant</td><td>×</td><td>√</td><td>√</td><td>1</td><td>High</td></tr><tr><td>Interval</td><td>√</td><td>√</td><td>×</td><td>3</td><td>Low</td></tr><tr><td>Increasing</td><td>√</td><td>√</td><td>√</td><td>1</td><td>Low</td></tr><tr><td>Decreasing</td><td>×</td><td>√</td><td>×</td><td>1</td><td>Low</td></tr></table>

## 4 NUMERICAL RESULTS

In this section, we test whether low-dimensional theoretical insights extend to high-dimensional image and text domains. Section 4.1 studies the effect of normalization, while Section 4.2 examines different guidance schedules. Additional details and samples are provided in Appendix G.

## 4.1 EFFECT OF NORMALIZATION

Recall that our theory predicted that failing to normalize complicates the simulation, so normalization should improve results in practice, which we confirm below.

Testing on Imagenet: We assess MaskGIT on the ImageNet dataset (Deng et al., 2009) and evaluate FID on ImageNet-256 using 50K samples, following standard practices. For our method and for the Unlocking Guidance baseline (Nisonoff et al., 2024), we use the $\tau$ -leaping sampler. For Simple Guidance (Schiff et al., 2024), we interpolate Euler transitions. For all methods, we use 50 steps. Figure 6a shows FID as a function of guidance strength using a constant schedule. Our experiments demonstrate that failing to normalize can substantially degrade sample quality as suggested by our theory.

![](images/d69cfe7dbf9548d391684fc7ce6990f53d42166ec1fabf7879198e117e19445f.jpg)  
(a) Comparison of different guidance mechanisms.

![](images/64045ae08acb19b3f287e8cf21ec8320b586289ffad0bb4320d3a58fd897c66d.jpg)  
(b) Right Interval vs Ramp-Up schedules.

![](images/527228dfb823e97c365a270e370f91f25b3983392ada824f1eceba4b5cc3ece8.jpg)  
(c) Left Interval vs Ramp-Down schedules  
Figure 6: Evaluation of different guidance mechanisms and schedules on Imagenet

The effect on diversity-quality tradeoff: To assess the effect of guidance on the fidelity-diversity tradeoff, Fig. 7 shows the precision-recall curves. Relative to the no-guidance baseline (w = 1), both Unlocking Guidance and Simple Guidance exhibit reduced precision as guidance increases, indicating a tradeoff between fidelity and diversity. In contrast, our method improves precision while maintaining comparable recall for moderate guidance strengths, before all methods degrade at large guidance.

![](images/5e709740f7f45ee06d7c8b576de954229e772f462041ea57a122d49984d269f4.jpg)

Testing on text-to-image: We evaluate our method on the GenEval benchmark (Ghosh et al., 2023) using Meissonic (Bai et al., 2024) as well as Show-O Xie et al. (2024) which is a mixed model leveraging discrete diffusion for image generation and auto-regressive for text generation. This benchmark provides a comprehensive measure of both prompt alignment and perceptual image quality. Figure 9 compares generations with and without normalization. Red regions indicate prompts where normalization

Figure 7: Only our method simultaneously improves fidelity and diversity. We plot the Precision/recall curves for different values of w. The dashed lines indicate the no-guidance baseline.

improved the score. Overall, we observe consistent gains: normalization enhances prompt adherence and yields images that better match the target distribution.

Testing on text generation: To assess the effectiveness of normalization in the text generation domain, we evaluated using LLaDA-8B-Instruct (Nie et al., 2025) on the MATH-500 dataset, generating up to 256 tokens. We sample autoregressively in blocks of 32 tokens using a simple Euler sampler with 32 denoising steps per block, resulting in a total of 256 steps for the full generation.

Figure 8 presents the results of such an experiment. The results clearly show that normalization consistently improves performance across all guidance strengths. We note that the results are not directly comparable to those reported in the LLaDA paper; we use a simple Euler sampler without remasking to better isolate the effect of guidance and normalization in a simple setting.

![](images/79791b481b1213ff2b0810f59305647a1adf4d3760f4ea8f11f9e84bf6bd506c.jpg)  
Figure 8: MATH-500 performance for LLada-8B-Instruct under a simple sampler without remasking to isolate the effect of the guidance mechanism. Normalization always yields better results.

Empirical effect of normalization: All our empirical findings demonstrate that normalization is a helpful step in improving the simulation of classifier-free guidance for discrete diffusion. This aligns with our low-dimensional theoretical analysis in Section 3.1, demonstrating that low-dimensional studies can have a significant impact in high dimensions.

![](images/f53c132813ca9e1567ea8c366cbbfc9e4b9129f321d4e9b1aa345af1e544825d.jpg)

![](images/1c4b4f9c56b135b229eefb2b5d18017e4a8e827160deeeb76b83bd92f2070e00.jpg)  
Figure 9: GenEval with and without normalization using Show-o (Top) and Meissonic (Bottom). Red denotes improved performance from normalization. Normalization improves prompt adherence and quality.  
Table 2: Description of guidance schedules.

## 4.2 STUDY OF GUIDANCE SCHEDULES

Previously, our theory predicted that increasing schedules improve discrete diffusion while decreasing ones degrade generation. We test this theory on Imagenet-256 with 10K samples. For precise formulas for the schedules, see Table 2. When testing increasing schedules (Ramp-Up and Right Interval)

<table><tr><td>Schedule</td><td>Formula w(t)</td></tr><tr><td>Left Interval</td><td>w · 1[0,l](t)</td></tr><tr><td>Right Interval</td><td>w · 1[r,1](t)</td></tr><tr><td>Ramp-Up</td><td>min (w, w · 1-t/1-r)</td></tr><tr><td>Ramp-Down</td><td>min (w, w · t/ℓ)</td></tr></table>

in 6b, we observe that both schedules can significantly improve the results. Furthermore, the Right Interval schedule exhibits a convex trend with respect to r, while the Ramp-Up schedule is monotone in r, and reaches a lower FID value, indicating that a gradual, linear increase in guidance outperforms abrupt alternatives. When testing the decreasing schedules (Left interval and Ramp-Down), we observe that they consistently damage the generation as seen in Figure 6c. Overall, our experiments confirm our theory that increasing schedules are most effective for masked diffusion.

## 5 CONCLUSIONS

In this work, we introduced a framework for analyzing guidance schedules in masked diffusion. Our analysis led to a novel approach for classifier-free guidance in the discrete setting. We validate the effectiveness of our method through experiments and show that guidance applied near t = T is harmful to the generation quality while near t = 0 can improve the it. This insights enabled us to identify effective scheduling strategies. Our theoretical insights align closely with empirical observations, bridging the gap between theory and practice.

Limitations and Future work. While our framework provides a principled and tractable approach to CFG in discrete diffusion, our theoretical analysis is currently limited to masked diffusion in low-dimensional settings. Although the method is applicable to more complex real-world settings, our current theoretical study does not cover such regimes. Promising directions include extending the framework to other forms of discrete diffusion, such as uniform diffusion, scaling to higher dimensions, and analyzing the role of score estimation error in the guidance dynamics.