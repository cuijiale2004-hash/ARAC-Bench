**Principle 1: Rigorous Justification of Steady-State Reachability and Local Homogeneity in Anomalous SGD Diffusion on Degenerate Landscapes**

**Definition:**  
The theoretical framework models late-stage SGD as a fractional Fokker-Planck diffusion on a porous, fractal-like loss landscape, which fundamentally relies on the existence of an approximate steady state and local validity of geometric relations such as the Alexander-Orbach relation. Reviewers must scrutinize whether the authors provide rigorous theoretical or empirical justification for these assumptions, as standard neural loss landscapes are highly non-homogeneous and SGD iterates may not converge to exact equilibria. A credible theory must explicitly address the distinction between transient early-training super-diffusive behavior and asymptotic sub-diffusive dynamics, ensuring that the latter dominates the stationary distribution analysis. Furthermore, the work should discuss the regime of validity—such as small learning rates, large batch sizes, or proximity to degenerate saddles—and what theoretical predictions fail if these conditions are violated. Without such justification, the connection between SGD and the tempered Bayesian posterior remains formally fragile. The principle demands that assumptions are treated as claims requiring validation rather than background postulates.

**Core Evaluation Criteria:**
- **Steady-State Existence:** Does the paper rigorously argue that SGD attains an approximate steady state under the studied conditions, and discuss non-equilibrium or divergent regimes?
- **Local Homogeneity:** Are geometric relations like the Alexander-Orbach relation justified for locally multifractal, non-homogeneous neural loss surfaces?
- **Transient vs. Asymptotic:** Is there a clear theoretical or empirical separation of early super-diffusion from late sub-diffusion, with proof that transients do not corrupt stationary properties?
- **Robustness Discussion:** Does the work analyze sensitivity to assumption violations and bound the resulting approximation error?

---

**Principle 2: Empirical Generalization and Statistical Rigor in Validating Fractal Diffusion Predictions Across Scales and Optimizers**

**Definition:**  
Theoretical predictions linking the local learning coefficient, spectral dimension, and weight displacement scaling must be validated with statistical rigor across a spectrum of model scales and optimizer classes. Reviewers should demand empirical evidence that extends far beyond toy datasets or small-scale benchmarks, encompassing modern architectures such as language models with tens of millions of parameters and vision models trained on ImageNet-scale data. Crucially, the evaluation must not rely solely on visually suggestive scatter plots or qualitative trends; it requires quantitative statistical metrics, confidence intervals, ablation studies over hyperparameters, and explicit testing on adaptive optimizers like Adam. The principle ensures that claimed scaling laws and bounds are reproducible phenomena rather than artifacts of a narrow experimental setup. Ultimately, the breadth and statistical depth of the empirical protocol determine whether the theory is a broadly applicable scientific account or a boutique observation.

**Core Evaluation Criteria:**
- **Scale and Architecture Diversity:** Are experiments conducted on sufficiently large and varied models (e.g., LLMs, deep CNNs) to support general claims about deep learning dynamics?
- **Optimizer Generality:** Does the empirical evaluation explicitly test whether predictions hold for both non-adaptive (SGD) and adaptive (Adam) optimizers, and identify where the theory breaks down?
- **Statistical Rigor:** Are claims supported by quantitative metrics (correlation coefficients, R² values, distributional distances) and multiple random seeds, rather than qualitative visuals alone?
- **Hyperparameter Ablation:** Are systematic ablations performed for learning rate, batch size, weight decay, and coarse-graining scale ξ to demonstrate robustness?

---

**Principle 3: Quantitative Verification of Tempered Bayesian Posterior Equivalence for SGD Stationary Distributions**

**Definition:**  
A headline claim of this research genre is that the stationary distribution of SGD corresponds to a tempered Bayesian posterior modulated by local accessibility constraints encoded in the loss landscape geometry. Reviewers must insist on direct, quantitative empirical comparisons between the empirical SGD stationary distribution and the true or SGLD-approximated Bayesian posterior, using rigorous statistical distances such as KL divergence, Jensen-Shannon divergence, or Wasserstein distance. Qualitative visual alignment or improved downstream task performance are insufficient proxies for distributional equivalence. The evaluation must also test the specific mechanism of tempering—demonstrating how the effective diffusion coefficient and the characteristic length scale ξ quantitatively reshape posterior concentration—and show sensitivity to these geometric hyperparameters. This principle ensures that the "almost Bayesian" characterization is a falsifiable empirical statement rather than a metaphorical interpretation.

**Core Evaluation Criteria:**
- **Distributional Metrics:** Are SGD and Bayesian posterior samples compared using rigorous statistical distances (KL, JS, Wasserstein) rather than qualitative inspection?
- **Tempering Mechanism:** Does the paper empirically validate that the proposed tempering transformation (via \(D_{\xi}\)) predicts the observed concentration of SGD solutions?
- **Scale Sensitivity:** Is the choice and impact of the characteristic length scale ξ explicitly tested and reported?
- **Causal Distinction:** Does the work rule out alternative explanations for the observed distributional shape, establishing that it arises specifically from LLC-mediated accessibility constraints?

---

**Principle 4: Conceptual Clarity and Notational Consistency in Introducing Singular Learning Theory and Fractal Dimensions**

**Definition:**  
Research at the intersection of singular learning theory, stochastic processes, and neural network optimization employs specialized constructs—including the local learning coefficient, spectral dimension, walk dimension, and fractional derivatives—that may be unfamiliar to the broader machine learning theory community. Reviewers must evaluate whether these concepts are introduced with sufficient pedagogical care, whether notation is internally consistent across sections, and whether definitions are cleanly separated from theorems and corollaries. In particular, the distinction between the geometric dimension of the loss landscape (LLC as a local mass dimension) and the dynamical dimension experienced by the optimizer (spectral dimension) must be articulated with precision to avoid conflating static structure with diffusion behavior. A failure in clarity renders sophisticated mathematical machinery opaque, preventing proper peer assessment and limiting community impact. The paper should ideally enable a non-expert reviewer to understand the logical flow from landscape geometry to Bayesian posterior.

**Core Evaluation Criteria:**
- **Pedagogical Accessibility:** Are SLT concepts and fractal dimensions explained with enough background for reviewers without specialized training in algebraic geometry or anomalous diffusion?
- **Notational Rigor:** Is notation consistent (e.g., function arguments, dependencies on \(p\), \(r\), \(\varepsilon\); walk vs. walker dimension), and are definitions distinguished from derived results?
- **Conceptual Distinction:** Is the difference between geometric properties (LLC) and dynamical properties (spectral dimension, walk dimension) clearly explained and motivated?
- **Motivation and Context:** Are introduced quantities justified by specific failures of prior simplifying assumptions (e.g., non-degenerate quadratic minima)?

---

**Principle 5: Actionable Insights and Practical Utility of Landscape Geometry Diagnostics for Optimization and Model Selection**

**Definition:**  
Beyond theoretical characterization, the work must translate its geometric insights into actionable guidance for deep learning practitioners and demonstrate that tracking landscape diagnostics offers value beyond conventional metrics. Reviewers should assess whether the authors propose concrete applications—such as learning rate scheduling, optimizer design, transfer learning protocols, or robust model selection—that are grounded in the theory and supported by empirical evidence. If computing the local learning coefficient or spectral dimension is computationally expensive, the paper should propose efficient, scalable proxies (e.g., the walk dimension estimated from displacement scaling) and honestly discuss feasibility for industry-scale networks. The practical significance principle also demands that authors situate their diagnostics against existing baselines, explaining why a practitioner should prefer tracking LLC or spectral dimension over simpler alternatives like gradient norms or Hessian eigenvalues. Without this translational step, the theory risks remaining an abstract curiosity rather than a tool for scientific progress in large-scale model training.

**Core Evaluation Criteria:**
- **Practical Applications:** Does the paper propose concrete, empirically grounded applications (e.g., scheduler design, fine-tuning guidance, optimizer evaluation)?
- **Diagnostic Value Proposition:** Is there a clear argument for why LLC, spectral dimension, or walk dimension provide information not captured by standard training diagnostics?
- **Computational Feasibility:** Are the costs of estimating these quantities acknowledged, and are scalable proxies proposed for large models?
- **Generality of Recommendations:** Are practical insights validated across multiple tasks, architectures, or optimizers, or are they confined to narrow settings?