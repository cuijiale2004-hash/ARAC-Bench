**Principle 1: Theoretical Consistency and Empirical Verifiability of Conditional Diffusion Dominance Guarantees**

**Definition:**  
In diffusion-based generative frameworks for multi-objective optimization, theoretical claims often rely on idealized assumptions such as bounded total-variation distance between learned and true conditional distributions, or exact descent directions under restrictive parameter regimes. Reviewers must assess whether authors clearly acknowledge the gap between theoretical assumptions and practical implementation—particularly when key hyperparameters deviate from idealized values in experiments. The principle demands that authors either provide tractable bounds under relaxed assumptions or supply rigorous empirical validation (e.g., dominance visualizations, convergence diagnostics) that substantiates the practical reliability of theoretical guarantees. This criterion is crucial because unconditional acceptance of asymptotic or idealized claims can mask fundamental brittleness in real-world deployment. Furthermore, the distinction between provable guarantees under idealized conditions and observed behavior under practical settings must be explicitly discussed to ensure scientific honesty.

**Core Evaluation Criteria:**
- **Assumption Transparency**: Does the paper explicitly state which assumptions are unverifiable in practice and justify why they are standard or unavoidable?
- **Empirical Validation**: Are theoretical dominance or convergence claims supported by targeted experiments that isolate the diffusion component from guidance and repulsion?
- **Relaxed Regime Analysis**: When theory requires idealized settings (e.g., zero repulsion), does the work provide approximate arguments or sensitivity analyses for realistic nonzero values?

---

**Principle 2: Isolation and Attribution of Performance Drivers Across Diffusion, Guidance, and Conditioning Mechanisms**

**Definition:**  
Hybrid architectures that combine conditional diffusion with gradient-based guidance, repulsion terms, and novel conditioning strategies risk obscuring which component actually drives Pareto improvement. Reviewers must demand systematic ablations that disentangle the generative prior from the optimization machinery—specifically comparing against pure gradient-based baselines (e.g., MGD+RBF without diffusion), guidance-free diffusion, and alternative conditioning formulations (e.g., explicit versus implicit shift vectors). This principle ensures that reported gains are not merely the result of stronger baseline gradient updates repackaged within a diffusion framework, and that the added computational cost of generative modeling is scientifically justified by measurable, unique contributions. Without such isolation, it remains impossible to determine whether the diffusion model provides meaningful benefit beyond simpler gradient-based refinement. Consequently, each architectural innovation must be accompanied by targeted experiments that isolate its marginal contribution to convergence and diversity.

**Core Evaluation Criteria:**
- **Baseline Isolation**: Does the work include a no-diffusion baseline that matches the guidance and repulsion components to quantify the marginal value of the generative model?
- **Guidance Ablation**: Is there a guidance-free diffusion variant to isolate the contribution of adaptive gradient directions from the raw denoising process?
- **Conditioning Disambiguation**: Are alternative conditioning strategies (e.g., explicit versus implicit shift encoding) empirically compared to validate the chosen mechanism?

---

**Principle 3: Asymptotic and Empirical Scalability Across Decision-Space Dimensionality, Sample Count, and Objective Dimension**

**Definition:**  
Diffusion-based optimization methods carry significant computational overhead from iterative denoising, inner-loop gradient optimization, and pairwise repulsion calculations. Reviewers must evaluate whether authors provide formal asymptotic complexity analysis (time and memory) in terms of the number of samples, decision variables, objectives, and diffusion steps, alongside empirical runtime comparisons against lightweight gradient or evolutionary baselines. Furthermore, scalability must be validated beyond toy problem sizes—particularly for decision-space dimensionality and many-objective settings where repulsion costs grow quadratically or hypervolume computation becomes prohibitive. A method that performs well on low-dimensional benchmarks but scales poorly to higher dimensions or larger sample counts offers limited practical utility. Therefore, both theoretical complexity bounds and empirical wall-clock measurements across varying problem scales are essential for assessing real-world viability.

**Core Evaluation Criteria:**
- **Formal Complexity**: Is a rigorous big-O analysis provided for the full sampling procedure, identifying dominant terms (e.g., repulsion versus denoising)?
- **Empirical Runtime**: Are wall-clock measurements reported across varying sample counts and decision dimensions, with training amortized appropriately?
- **Many-Objective Scaling**: Does the evaluation extend to high-dimensional objective spaces (e.g., m ≥ 10) using appropriate approximate metrics, demonstrating that the method does not collapse or become computationally intractable?

---

**Principle 4: Hyperparameter Transparency, Default Justification, and Sensitivity Robustness in Expensive Optimization Regimes**

**Definition:**  
Generative optimization frameworks introduce numerous hyperparameters—repulsion weights, perturbation scales, step sizes, and shift magnitudes—whose defaults are often chosen opaquely or omitted entirely. In expensive multi-objective settings (offline or Bayesian), extensive hyperparameter tuning is infeasible, making transparent reporting and robustness essential. Reviewers must assess whether default values are explicitly stated, whether their selection is principled or merely heuristic, and whether sensitivity analyses demonstrate stable relative performance across a wide range of values. The principle also requires that authors distinguish between settings (online, offline, Bayesian) when specifying defaults, as optimal values may vary dramatically across regimes. Without such transparency, reproducibility suffers and practitioners cannot determine whether observed advantages require extensive hidden tuning or are genuinely robust.

**Core Evaluation Criteria:**
- **Default Disclosure**: Are all critical hyperparameters (e.g., repulsion strength, perturbation scaling factor) reported with their exact default values per setting?
- **Selection Rationale**: Does the paper explain how defaults were derived (e.g., preliminary checks, theoretical bounds) rather than presenting them as arbitrary?
- **Sensitivity Evidence**: Are ablation studies or landscape analyses provided showing that performance ordering against baselines remains stable under hyperparameter variation?

---

**Principle 5: Calibrated Cross-Setting Positioning and Transferability Across Problem Domains in Generative MOO**

**Definition:**  
Methods that claim applicability across online, offline, and Bayesian multi-objective optimization must be evaluated against setting-specific state-of-the-art baselines with claims calibrated to each regime's constraints. Reviewers must scrutinize whether performance gains in one setting (e.g., offline with abundant data) are inappropriately generalized to others (e.g., online with cheap evaluations or Bayesian with scarce data). Additionally, because training a separate diffusion model per problem is computationally burdensome, the principle demands investigation of cross-problem transferability—training on related problems and evaluating on held-out tasks—to establish whether the method learns generalizable structure or merely memorizes problem-specific features. Overclaiming universal superiority while excelling only in narrow settings undermines scientific credibility and misleads the community about practical applicability. Thus, rigorous scope delineation and cross-domain transfer experiments are necessary to validate the generality of the proposed framework.

**Core Evaluation Criteria:**
- **Setting-Specific Benchmarking**: Are comparisons fair and state-of-the-art within each claimed setting, with claims scoped appropriately to regimes where the method excels?
- **Claim Calibration**: Does the paper avoid overstating universal superiority when advantages are concentrated in specific settings (e.g., offline-friendly but marginal in online)?
- **Transfer Validation**: Are experiments provided demonstrating generalization or transfer learning across related problems, quantifying the trade-off between training cost and performance retention?