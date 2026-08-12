Principle 1: Trans-Dimensional Architecture Learning and Higher-Order Interaction Discovery in Bayesian Functional ANOVA Neural Networks

**Definition:**  
In Bayesian neural network approaches to functional ANOVA decomposition, a critical evaluation criterion is whether the method transcends a straightforward Bayesian reformulation of existing architectures by enabling joint learning of both network parameters and the combinatorial structure of interaction components. The functional ANOVA framework suffers from exponential growth in component counts as interaction order increases, so principled trans-dimensional inference—such as reversible-jump MCMC with stepwise proposals—must be demonstrated to actively discover meaningful higher-order terms rather than merely selecting among pre-specified candidates. This principle assesses whether the architectural learning mechanism is technically sophisticated, empirically validated, and causally linked to performance improvements on datasets where fixed-order models fail due to computational or representational limitations.

**Core Evaluation Criteria:**
- **Distinction from Fixed-Architecture Baselines:** Does the method outperform approaches that exhaustively enumerate components up to a pre-specified order (e.g., ANOVA-T2PNN), and are performance gains explicitly attributed to discovered higher-order interactions rather than Bayesian averaging alone?
- **Mechanistic Rigor of the Sampler:** Is the trans-dimensional proposal mechanism (e.g., stepwise node addition guided by feature importance) rigorously derived, and do ablation studies demonstrate its necessity for escaping low-order local modes?
- **Causal Attribution of Architectural Discovery:** Are discovered architectures inspected qualitatively (e.g., component importance rankings) and shown to correspond to semantically meaningful variable groups?

---

Principle 2: Comprehensive Uncertainty Quantification and Calibration Evaluation for Interpretable Bayesian Neural Networks

**Definition:**  
For Bayesian interpretable models, predictive accuracy is insufficient; the evaluation must rigorously assess uncertainty quality using both proper scoring rules and calibration metrics against strong, widely adopted baselines. Reviewers expect comparisons with deep ensembles, standard Bayesian models (BART, mBNN), and—where applicable—scalable stochastic gradient MCMC methods, using metrics such as CRPS, ECE, and NLL across regression and classification tasks. The principle emphasizes that claims of superior uncertainty must be substantiated in both in-distribution calibration and out-of-distribution detection, ensuring that interpretability does not come at the cost of poorly calibrated predictions.

**Core Evaluation Criteria:**
- **Baseline Completeness:** Are state-of-the-art uncertainty quantification baselines included, such as deep ensembles and black-box Bayesian models, with hyperparameters tuned via cross-validation?
- **Metric Diversity:** Are multiple standard metrics reported (e.g., CRPS/ECE and NLL), covering both calibration and sharpness, on a diverse suite of real datasets?
- **Out-of-Distribution Robustness:** Does the model exhibit appropriate uncertainty expansion on OOD data, and is this evaluated with quantitative metrics (e.g., AUROC for OOD detection)?

---

Principle 3: Computational Scalability and Memory Feasibility in Bayesian ANOVA Networks with Order-Adaptive Interactions

**Definition:**  
A method claiming to alleviate the combinatorial explosion of higher-order functional ANOVA components must provide convincing empirical evidence of computational efficiency and memory scalability as input dimension and interaction order grow. This includes runtime comparisons showing scaling advantages over exhaustive enumeration methods, validation on high-dimensional regimes where fixed-order competitors are infeasible (e.g., p=500 with fourth-order interactions), and evidence that practical approximations such as mini-batch training do not substantially degrade posterior quality. Without such evidence, claims of scalability remain theoretical and the method's practical utility is questionable.

**Core Evaluation Criteria:**
- **Runtime Scaling Analysis:** Are empirical runtimes reported as a function of input dimension p and maximum interaction order, demonstrating clear advantages over exhaustive methods like ANOVA-T2PNN?
- **High-Dimensional Regime Validation:** Are experiments conducted on datasets with large p (e.g., genomics with p >> n or p in the hundreds) where discovering higher-order interactions is computationally prohibitive for baseline methods?
- **Approximation Quality:** If mini-batch or stochastic approximations are used, do ablations show negligible degradation in predictive performance, uncertainty metrics, and component selection accuracy?

---

Principle 4: Cross-Domain Applicability and Representation-Level Integration in Interpretable Bayesian ANOVA Models

**Definition:**  
Interpretable machine learning research must demonstrate substantive practical value beyond conventional UCI-style tabular datasets, particularly by addressing complex, high-stakes domains such as genomics or perceptual tasks via concept bottleneck pipelines. Reviewers assess whether the proposed method integrates effectively with modern representation learning (e.g., CNN-derived concepts), produces domain-relevant interpretations (e.g., specific gene-gene interactions), and maintains competitive predictive performance when operating on structured latent features rather than raw inputs. A narrow focus on small tabular benchmarks risks limiting conclusions about real-world applicability.

**Core Evaluation Criteria:**
- **Cross-Domain Evaluation:** Does the empirical evaluation span diverse data modalities, including at least one high-dimensional scientific dataset (e.g., genomics) and one perceptual task with learned representations?
- **Interpretability Utility:** Are discovered components inspected for domain relevance, and do they provide actionable insights (e.g., identifying higher-order gene interactions or concept relationships)?
- **Integration with Representation Learning:** When applied to non-tabular data, is the method shown to work synergistically with backbone models (e.g., CBMs), rather than degrading performance by operating on flattened raw features?

---

Principle 5: Component-Wise Posterior Consistency and Identifiability Guarantees in Bayesian ANOVA Decomposition

**Definition:**  
In functional ANOVA models, theoretical soundness requires more than global function approximation; it necessitates guarantees that the posterior distribution concentrates around the true individual components (main effects and interactions) under the model's identifiability constraints. Because ANOVA decomposition relies on structural assumptions such as sum-to-zero conditions to ensure unique component recovery, reviewers must evaluate whether proofs of posterior consistency explicitly address component-level recovery, account for the exponential family likelihood, and respect the neural network function class's complexity. The absence of such guarantees weakens the interpretability claims, as the estimated components may not correspond to a well-defined statistical target.

**Core Evaluation Criteria:**
- **Component-Level Consistency:** Does the theoretical analysis prove posterior consistency for individual ANOVA components f_S, not merely for the aggregate regression function f?
- **Identifiability Enforcement:** Are the constraints ensuring unique decomposition (e.g., sum-to-zero conditions) explicitly incorporated into the prior and posterior analysis?
- **Regime Appropriateness:** Do the regularity conditions, sieve constructions, and prior specifications align with the neural network basis function class and the exponential family likelihood structure used in the model?