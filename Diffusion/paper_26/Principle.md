**Principle 1: Empirical Validation and Extrapolation Reliability of Scaling Law Predictions in Discrete Diffusion Language Models**

**Definition:**  
Scaling laws are a cornerstone of modern LLM research, yet their predictive power depends critically on whether trends observed in small-scale experiments hold when extrapolated to orders-of-magnitude larger compute budgets. In the discrete diffusion setting, this concern is amplified by the possibility that different noise types (masked, uniform, hybrid) may exhibit distinct convergence behaviors or phase transitions at scale that are not captured by power-law fits over limited ranges. This principle evaluates whether authors treat extrapolation as a hypothesis to be tested rather than a foregone conclusion, and whether they invest substantial computational resources to validate their fitted laws at scales far exceeding the training data used for fitting. It further assesses whether the methodology for deriving these laws is robust to the choice of fitting technique and whether confidence intervals or sensitivity analyses are provided to quantify uncertainty.

**Core Evaluation Criteria:**
- **Large-Scale Empirical Verification:** Does the work validate its scaling law predictions with training runs that are significantly larger (e.g., 5×–50×) than the maximum scale used for fitting? Are the observed losses reported alongside the predicted values?
- **Methodological Reliability of Extrapolation:** Are the scaling laws derived using established, reliable techniques (e.g., iso-FLOP profiles) rather than fragile parametric fits? Are the authors transparent about why a particular fitting approach was chosen?
- **Regime-Specific Analysis:** Does the work clearly distinguish between compute-bound and data-bound scaling regimes, and does it verify predictions in both settings independently?
- **Uncertainty Quantification:** Are confidence intervals, bootstrapped errors, or sensitivity analyses provided for the fitted exponents, and are limitations of the extrapolation explicitly acknowledged?

---

**Principle 2: Hyperparameter Scaling Dynamics and Compute-Optimal Training Configuration in Large-Scale Diffusion Pre-training**

**Definition:**  
In scaling law research, it is common to fix batch size and learning rate to constants across experiments, but this assumption can severely distort the compute-optimal frontier if these hyperparameters have strong scale-dependent optima. For discrete diffusion models, the interaction between noise type, model size, and training dynamics may produce hyperparameter scaling behaviors that differ substantially from autoregressive baselines. This principle evaluates whether authors treat batch size, learning rate, and annealing schedules as first-class scaling dimensions that must be systematically swept and characterized, rather than as fixed nuisances. It also assesses whether claimed invariances—such as the independence of optimal batch size from model size or noise type—are backed by sufficient statistical evidence across a broad enough range of scales.

**Core Evaluation Criteria:**
- **Systematic Hyperparameter Sweeps:** Are batch size and learning rate swept across a meaningful range for each model size and training horizon, or are they fixed to potentially suboptimal constants?
- **Scaling Laws for Optimal Hyperparameters:** Are the optimal batch size and learning rate characterized as explicit functions of training tokens or model scale (e.g., via power-law fits), and are the fitted exponents reported with goodness-of-fit metrics?
- **Treatment of Annealing/Cooldown:** Is learning rate annealing analyzed as a distinct training phase? Does the work justify whether scaling laws estimated without annealing remain valid when annealing is applied, and is the performance delta quantified?
- **Statistical Support for Invariance Claims:** When authors claim that optimal hyperparameters are independent of model size or noise type, is this supported by overlapping confidence intervals across grouped subsets, or is it an underpowered generalization?

---

**Principle 3: Downstream Performance Correlation and Practical Utility Verification of Diffusion Pre-training Objectives**

**Definition:**  
Pre-training loss (e.g., ELBO or negative log-likelihood) is the standard metric for scaling law estimation, but it is ultimately a surrogate for the downstream capabilities that determine a model's practical utility. For diffusion language models, the relationship between pre-training ELBO and downstream benchmark performance is complicated by the non-autoregressive nature of the model and the sensitivity of inference to sampling algorithms (e.g., ancestral vs. adaptive decoding). This principle evaluates whether authors move beyond loss-curve analysis to demonstrate that their scaling findings translate into meaningful improvements on standard NLP benchmarks, and whether they analyze how different noise types or sampling strategies affect task-specific performance. It also examines whether the evaluation protocol is fair and comprehensive enough to support claims about competitiveness with autoregressive models.

**Core Evaluation Criteria:**
- **Standard Benchmark Coverage:** Does the work evaluate on a diverse set of established NLP benchmarks (e.g., ARC, PIQA, BoolQ, GSM8k) to validate practical utility, or does it rely solely on pre-training loss?
- **Correlation Analysis:** Is the relationship between pre-training loss and downstream accuracy explicitly analyzed? Are there task-type-specific patterns (e.g., reasoning-heavy vs. knowledge-heavy tasks) that differ across noise types?
- **Inference Protocol Rigor:** Are downstream evaluations conducted with well-defined, reproducible sampling procedures (including the number of denoising steps and any adaptive heuristics), and are these procedures justified and generalizable across noise types?
- **Comparative Competitiveness:** Does the work provide sufficient evidence to support claims that diffusion models can match or exceed autoregressive models at scale, including appropriate caveats about dataset and tokenizer differences?

---

**Principle 4: Theoretical Framing and Prior Art Integration in Discrete Diffusion Process Design and SNR Parameterization**

**Definition:**  
The discrete diffusion landscape is evolving rapidly, with concurrent work on masked diffusion simplifications, hybrid-noise formulations, and diffusion forcing. A rigorous contribution in this space must clearly articulate its theoretical motivation—such as reparameterizing the diffusion process by signal-to-noise ratio (SNR) rather than time—and rigorously connect it to existing frameworks like GIDD or continuous-state diffusion theory. This principle evaluates whether authors provide sufficient background for their design choices, define notation precisely, and contextualize their theoretical claims against the most relevant prior work. It also assesses whether the work acknowledges and addresses any apparent tensions between its empirical findings and existing theoretical results (e.g., arguments about the inherent difficulty of uniform diffusion versus masked diffusion).

**Core Evaluation Criteria:**
- **Clarity of Theoretical Motivation:** Are key concepts (e.g., isotropic vs. anisotropic denoising, SNR parameterization, diffusion forcing) formally defined and motivated, or are they introduced opaquely with assumed prior knowledge?
- **Comprehensive Prior Work Contextualization:** Does the work cite and discuss the most relevant recent advances in discrete diffusion, including simplified masked diffusion objectives, hybrid-noise variants, and theoretical analyses of task difficulty? Are missing citations addressed?
- **Rigor of Mathematical Presentation:** Is the derivation of the objective function or ELBO clear and self-contained? Are all terms in key equations defined, and is the connection to continuous diffusion theory articulated?
- **Resolution of Theoretical Tensions:** If the work's empirical conclusions appear to contradict existing theoretical arguments (e.g., regarding the comparative difficulty of uniform vs. masked diffusion), does it provide a reconciling explanation rather than ignoring the discrepancy?

---

**Principle 5: Statistical Rigor and Methodological Robustness in Scaling Law Estimation and Phenomenological Fitting**

**Definition:**  
Scaling law research often walks a fine line between phenomenological description and mechanistic explanation. Claims about optimal scaling coefficients, critical batch sizes, or the functional relationship between training variables must be supported by appropriate statistical methodology rather than visual inspection or underpowered fits. This principle evaluates whether the authors' quantitative claims are backed by robust statistical analysis, whether they test alternative hypotheses (e.g., power laws with vs. without an irreducible loss term), and whether they avoid overstating the implications of their fits. It also examines whether the authors clearly distinguish between observed empirical patterns and their underlying causal mechanisms, and whether they retract or weaken claims when the data demands it.

**Core Evaluation Criteria:**
- **Appropriate Fitting Methodology:** Is the scaling law estimation technique well-suited to the data (e.g., iso-FLOP profiles for compute-optimal frontiers), and are alternative approaches considered and rejected with justification?
- **Statistical Validation of Claims:** Are quantitative claims supported by formal statistical analysis (e.g., confidence intervals, R² values, normal approximations) rather than informal visual comparisons? Are ablation studies analyzed with appropriate rigor?
- **Handling of Irreducible Loss and Asymptotics:** Does the work correctly account for non-zero irreducible loss in its hypothesis space, and does it avoid extrapolating power-law fits into regimes where they are known to break down?
- **Intellectual Honesty in Claim Revision:** When updated methodology or additional data weakens an initial claim (e.g., shifting from "uniform diffusion is clearly superior" to "all noise types converge comparably"), does the work transparently report this revision and update its framing accordingly?