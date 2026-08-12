
**Principle 1: Theoretical Characterization of Inner-Loop Estimation Noise and Its Impact on Outer-Loop Bayesian Optimization Convergence**

**Definition:**  
When interleaving data selection with sequential black-box optimization for LLM training, it is crucial to theoretically quantify how the inner-loop data selection estimator shapes the observation noise distribution and consequentially affects the outer-loop regret bounds. In this subfield, simply combining Bayesian optimization with off-the-shelf data selection methods is insufficient; the research must establish a principled, mechanistic link between the quality of the inner-loop estimator and the convergence behavior of the outer-loop optimizer. This includes formalizing how the estimator reduces noise variance or tail heaviness, deriving regret bounds that explicitly incorporate these statistical properties, and demonstrating that the interleaving yields provably faster convergence than either component alone. Such theoretical rigor distinguishes a novel algorithmic contribution from a naive engineering pipeline.

**Core Evaluation Criteria:**
- **Novelty of Interaction**: Does the work formally link the error distribution of the data selection estimator (e.g., truncated exponential) to the Bayesian optimization observation noise model, rather than treating the two components as independent black boxes?
- **Regret Bound Dependence**: Are the derived regret bounds explicitly parameterized by inner-loop estimator properties (e.g., sampling size, truncation constants), clearly showing how data selection quality propagates into optimization convergence?
- **Avoidance of Naïve Composition**: Does the manuscript convincingly argue why the interleaving itself constitutes the technical contribution, and does it rule out trivial baselines where standard BO and data selection are applied sequentially without co-design?

---

**Principle 2: Computational Cost Transparency and Scalability Validation across LLM Training Paradigms**

**Definition:**  
Iterative algorithms that require multiple full LLM training runs and expensive data selection score computations must provide rigorous complexity analysis, memory profiling, and empirical validation beyond parameter-efficient fine-tuning. Because each outer-loop iteration may involve fine-tuning or pre-training an LLM, reviewers must assess whether the authors clearly articulate the computational and storage overheads of all algorithmic components, including the cubic cost of Gaussian Process matrix inversion and the quadratic or linear costs of influence function computation. Furthermore, claims about practical applicability must be supported by evidence across training paradigms—such as full-parameter fine-tuning on multi-billion parameter models—or by credible amortization and approximation strategies that show a feasible path to pre-training scale.

**Core Evaluation Criteria:**
- **Explicit Complexity Analysis**: Are the computation and memory overheads of every major subroutine explicitly characterized (e.g., BO kernel matrix operations, influence function gradient/Hessian costs, storage of best adapters)?
- **Scaling Empirics**: Does the work validate feasibility on full-parameter fine-tuning or larger model scales, or does it convincingly justify scalability via surrogates, pre-computation, or cheaper inner-loop alternatives?
- **Cost-Performance Tradeoffs**: Are alternative inner-loop selectors with varying compute budgets systematically compared, demonstrating that practitioners can flexibly navigate the accuracy-efficiency Pareto frontier?

---

**Principle 3: Robustness and Signal Reliability of Data Selection under Unseen Task Distribution Mismatch**

**Definition:**  
In settings where the downstream evaluation task is a black box and its data distribution is entirely unseen, the reliability of data selection signals must be justified under severe distribution shift. Conventional influence functions or gradient-based selection methods assume access to validation sets drawn from the target task, which is violated in this problem setting. Consequently, reviewers must evaluate whether the authors adequately explain why their chosen signal—computed solely on training-domain data—remains stable and informative when the true evaluation distribution is unknown. This requires addressing known failure modes of these signals (e.g., the brittleness of influence functions in deep learning), articulating the specific regime where they remain useful (such as filtering universally low-quality or nonsensical data), and empirically verifying that the signal does not collapse under out-of-domain evaluation conditions.

**Core Evaluation Criteria:**
- **Justification Under Mismatch**: Does the work rigorously justify why the data selection signal remains reliable despite being computed without any evaluation task data, labels, or gradients?
- **Failure Mode Acknowledgment**: Are known limitations of the selection signal explicitly discussed, and are mitigations or operational boundaries (e.g., using influence functions primarily for down-weighting noisy data rather than precise attribution) clearly stated?
- **OOD Stress Testing**: Does the empirical validation include challenging out-of-domain settings where the evaluation domain is explicitly withheld from the training mixture, proving that the signal aids generalization rather than mere memorization of an accessible target domain?

---

**Principle 4: Experimental Rigor in Feedback-Driven Data Mixture Optimization across Diverse Task Types**

**Definition:**  
Because the optimization target is a noisy, coarse feedback signal from unseen tasks, experimental design must rigorously validate performance across both in-domain and out-of-domain tasks, control for confounders such as total training tokens or epochs, and disentangle the contributions of the global optimizer from the local data selector. Reviewers must ascertain whether the authors compare against strong baselines that exploit fine-grained task information (e.g., DoReMi, LESS) as well as naive approaches (e.g., uniform mixing, random search), and whether ablation studies isolate the value added by interleaving versus using either BO or data selection in isolation. Additionally, because Bayesian optimization can exhibit surprising sample efficiency on simple landscapes, the experiments must convincingly demonstrate that performance gains are robust across task types and not artifacts of an overly simple objective surface.

**Core Evaluation Criteria:**
- **Strong Baseline Comparison**: Are comparisons made against both informed baselines that use task-specific information and uninformed baselines that mirror the practical constraints of the setting?
- **Component Ablation**: Do ablation studies clearly disentangle the contributions of the outer-loop BO, the inner-loop data selection, and their specific interaction?
- **Landscape and Iteration Analysis**: Does the work provide qualitative or quantitative analysis of the optimization landscape and per-iteration mixtures to rule out lucky initialization or trivial exploitation of a single dominant domain?

---

**Principle 5: Clarity of Problem Formulation and Distinction from Standard Domain Adaptation or Data Selection Settings**

**Definition:**  
Research targeting unseen, black-box evaluation tasks must precisely define the information asymmetry—specifically what is unknown versus what coarse feedback is available—and rigorously distinguish the setting from conventional domain adaptation, domain generalization, or standard data mixing. Reviewers must evaluate whether the authors clearly articulate why existing methods that require evaluation gradients, labels, or distributional assumptions are inapplicable, and whether real-world scenarios (e.g., encrypted user conversations, third-party model marketplaces) genuinely necessitate this restrictive feedback-only formulation. A well-posed problem in this subfield should leave no ambiguity about whether the evaluation task data is partially observable, whether feedback is aggregated at the model level, and how noise enters the observation channel.

**Core Evaluation Criteria:**
- **Formal Setting Definition**: Is the problem formally defined regarding the absence of evaluation data, gradients, or token-level losses, and the exclusive availability of coarse, potentially noisy, aggregate feedback?
- **Distinction from Prior Settings**: Does the work clearly contrast its assumptions with those of domain adaptation, domain generalization, conventional data mixing, and standard data selection to establish a unique niche?
- **Real-World Necessity**: Are concrete, realistic motivations provided that explain why the feedback-only formulation is unavoidable in practice, rather than being an artificially restrictive academic variant of a standard problem?