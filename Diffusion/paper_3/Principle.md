**Principle 1: Fairness of Computational Budget Matching and Deployment-Context Justification in Test-Time Error Correction for Efficient Diffusion Models**

**Definition:**  
For test-time methods that introduce additional computation to mitigate errors in deployed efficient diffusion models, the evaluation must rigorously address whether observed performance gains stem from methodological innovation or merely from increased computational budget. This requires explicit comparison against compute-matched baselines—such as increased sampling steps or naive iterative refinement—under identical deployment constraints. The justification for additional compute must account for real-world deployment realities, including frozen model schedules, retraining infeasibility, and quantization recalibration costs, rather than treating compute as an abstract fungible resource. Furthermore, the work should demonstrate that the proposed correction mechanism utilizes computational budget more efficiently than naive scaling, for instance by achieving comparable or superior quality with significantly less overhead. Finally, the evaluation should include ablations varying the scope of correction application to demonstrate flexible, runtime-adjustable trade-offs between quality and latency.

**Core Evaluation Criteria:**
- **Compute-Matched Baseline Inclusion:** Are baselines provided that match the total inference compute of the proposed method (e.g., doubled sampling steps without the correction module), and do results show the proposed method still outperforms these baselines?
- **Deployment-Context Argumentation:** Does the work clearly articulate why naive compute scaling is infeasible or inferior in the target deployment scenario (e.g., retraining costs, schedule immutability, quantization parameter recalibration)?
- **Efficiency-Adjusted Quality Metrics:** Does the work report performance per unit of additional compute and demonstrate superior efficiency compared to naive scaling?
- **Ablation on Applied Scope:** Are experiments provided showing the method's effectiveness when applied to only a subset of timesteps, demonstrating flexible runtime trade-offs?

---

**Principle 2: Rigor of Theoretical Derivation and Direct Empirical Verification for Exponential Error Propagation Claims in Quantized and Cached Diffusion Models**

**Definition:**  
Research claiming to mitigate error accumulation in efficient diffusion models must provide both rigorous theoretical analysis and direct empirical verification of asserted error propagation dynamics. Theoretical claims regarding error growth rates—such as exponential versus linear accumulation—cannot rely solely on intermediate mathematical constructs like spectral norms of propagation matrices exceeding unity, but must establish clear connections to end-to-end error behavior over complete sampling trajectories. Empirical validation must move beyond proxy metrics such as FID to directly measure accumulated error magnitudes across diffusion timesteps, demonstrating rapid amplification from early to late stages. Such verification must span multiple datasets and model architectures to rule out setting-specific artifacts. The work should also clearly distinguish approximation errors introduced by efficiency techniques from other error sources like discretization error or model under-training, ensuring the analysis targets the correct problem domain.

**Core Evaluation Criteria:**
- **End-to-End Theoretical Rigor:** Does the theoretical analysis rigorously connect local error amplification conditions to global end-to-end error bounds, avoiding gaps between intermediate constructs and final claims?
- **Direct Empirical Error Measurement:** Are there direct measurements of accumulated error magnitude over timesteps showing rapid amplification across diverse settings?
- **Cross-Model/Dataset Verification:** Is the error propagation phenomenon empirically validated across multiple model architectures and datasets of varying complexity?
- **Distinction of Error Sources:** Does the work clearly distinguish between approximation errors from efficiency techniques and other error sources?

---

**Principle 3: Cross-Architecture, Cross-Sampler, and Cross-Efficiency-Technique Generalization Validation for Post-Deployment Diffusion Model Enhancement**

**Definition:**  
Test-time enhancement methods for efficient diffusion models must demonstrate broad applicability beyond a single model family, sampling algorithm, or efficiency technique to establish generalizability. Given the diversity of diffusion architectures—including CNN-based U-Nets and Transformer-based models—validation limited to one architecture raises concerns about hidden structural assumptions. Similarly, the method must show compatibility with multiple sampling procedures such as DDIM, DPM-Solver variants, and flow-based ODE solvers, rather than relying on idiosyncrasies of a specific sampler. The evaluation should also span diverse structural efficiency techniques including weight quantization, activation quantization, feature caching, and hybrid approaches, as each introduces distinct error profiles. Finally, experiments must cover contemporary fast-sampling regimes with low step counts, as these reflect real-world deployment conditions where efficiency techniques are most critical.

**Core Evaluation Criteria:**
- **Architecture Diversity:** Are experiments conducted on both CNN-based and Transformer-based diffusion architectures showing consistent error correction behavior?
- **Sampler Compatibility:** Does the work validate applicability beyond the primary sampler to other popular samplers or demonstrate theoretical compatibility with different sampling paradigms?
- **Efficiency Technique Breadth:** Is the method evaluated across multiple distinct efficiency techniques rather than a single technique?
- **Low-Step Regime Validation:** Are results reported for contemporary fast-sampling regimes that reflect real-world deployment conditions?

---

**Principle 4: Comprehensive Pareto Trade-off Characterization and Practical Deployability Assessment of Test-Time Compute-Quality Controllers for Efficient Generative Models**

**Definition:**  
For test-time methods offering flexible trade-offs between computational overhead and generation quality, the evaluation must comprehensively characterize the compute-quality Pareto frontier across the full operational design space. This requires systematically varying the intensity of the correction mechanism—such as iteration count and percentage of timesteps corrected—alongside base efficiency parameters like quantization bit-width and caching interval. The analysis must transcend isolated point comparisons to demonstrate that the method provides superior quality for any given compute budget relative to appropriate baselines. Runtime overhead must be explicitly quantified as a percentage of baseline inference time under different application scopes, enabling practical deployment decisions. Additionally, the work should demonstrate dynamic runtime adjustability of the compute-quality trade-off without requiring model retraining or architectural modification.

**Core Evaluation Criteria:**
- **Pareto Frontier Analysis:** Are comprehensive compute-performance trade-off curves provided across varying sampling steps, quantization configurations, and caching intervals?
- **Granular Overhead Reporting:** Is the time overhead reported for different application scopes as a percentage of baseline inference time?
- **Dynamic Trade-off Demonstration:** Does the work demonstrate that users can flexibly adjust the compute budget at runtime without retraining?
- **Real-World Cost Integration:** Does the analysis account for total cost of ownership, including whether avoiding retraining offsets inference-time overhead?

---

**Principle 5: Theoretical Distinctiveness and Convergence Guarantee Beyond Naive Iterative Refinement in Fixed-Point-Based Test-Time Diffusion Correction**

**Definition:**  
When a test-time refinement method structurally resembles iterative procedures such as repeated denoising function calls, the research must establish clear theoretical and empirical distinctions from naive iterative baselines. Simply reporting improved metrics is insufficient; the work must demonstrate that its specific formulation provides unique benefits—including convergence guarantees, error suppression properties, or superior compute efficiency—that arbitrary repetition cannot achieve. This requires rigorous theoretical grounding, such as fixed-point convergence analysis via contraction mapping theorems, proving that the iterative procedure converges to a unique bounded solution. Empirically, the method must demonstrate superiority over structurally similar but theoretically unmotivated alternatives, such as naively increasing iteration counts or sampling steps. Finally, the work should provide mechanistic explanations of why the correction formulation addresses the root cause of error accumulation rather than merely averaging or diluting errors across timesteps.

**Core Evaluation Criteria:**
- **Theoretical Distinctiveness:** Does the method provide a rigorous theoretical framework that distinguishes it from naive repeated sampling or simple step-doubling?
- **Convergence and Boundedness Guarantees:** Are there formal proofs that the iterative procedure converges and that corrected error remains bounded?
- **Empirical Superiority over Naive Baselines:** Are there experiments showing the proposed method achieves better quality than naive compute-matched alternatives at the same or lower cost?
- **Mechanistic Explanation:** Does the work explain why the specific correction formulation addresses the root cause of error accumulation?