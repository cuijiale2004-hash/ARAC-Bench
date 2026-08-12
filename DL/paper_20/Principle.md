**Principle 1: Mechanistic Justification and Empirical Verification of Latent Fixed-Point Convergence in Recurrent-Depth Neural Simulators**

**Definition:**  
In recurrent-depth neural simulators that expose test-time compute-accuracy trade-offs through iterative latent refinement, the claim that repeated application of a recurrent block contracts toward a fixed point cannot be taken for granted. Reviewers must demand empirical or theoretical evidence that latent vectors stabilize as the number of iterations increases, since this underpins the predictability and monotonicity of the accuracy-cost curve. Furthermore, the convergence behavior should ideally reflect the underlying physical dynamics—such as requiring more iterations during shock formation or soliton interactions—to demonstrate that the model has acquired a physically meaningful inductive bias rather than performing arbitrary iterative deepening. Without such verification, the framework risks being indistinguishable from generic iterative procedures that lack guarantees of stable improvement with additional compute. The evaluation should therefore scrutinize whether the authors establish a causal link between iterative depth and solution fidelity, supported by latent-space trajectory analysis and explicit convergence criteria. This principle is crucial because fixed-point convergence separates principled scientific solvers from unstable black-box iterative schemes.

**Core Evaluation Criteria:**
- **Convergence Evidence**: Does the work provide empirical measurements (e.g., Euclidean distance between latent states and an approximate fixed point) demonstrating practical convergence, and are tolerance thresholds explicitly defined?
- **Physical Alignment**: Does the convergence pattern correlate with physical regimes (e.g., slower convergence during nonlinear transients), or is convergence uniform and physics-agnostic?
- **Stability of Iterative Rollout**: Does the model avoid oscillation or divergence at high iteration counts, and is this verified across multiple PDE benchmarks with varying dynamical complexity?

---

**Principle 2: Rigorous Differentiation from Deep Equilibrium and Diffusion-Based Adaptive Compute Paradigms**

**Definition:**  
Neural simulators with test-time compute control occupy a crowded design space alongside Deep Equilibrium Models (DEQs) and diffusion-based refiners, which also iterate a function to improve output quality. A recurrent-depth simulator must therefore be rigorously distinguished from these alternatives in terms of forward-pass mechanics (truncated unrolling versus root-finding or denoising), backward-pass training (truncated backpropagation-through-depth versus implicit differentiation or phantom gradients), and inference-time behavior (anytime prediction versus mandatory completion of a fixed schedule). Reviewers should evaluate whether the authors articulate why their training distribution over recurrent iterations yields superior generalization to out-of-distribution compute budgets compared to DEQs trained with fixed-point solvers, or why their latent trajectory avoids the early plateauing and parameter sensitivity observed in diffusion-based PDE refiners. This positioning is essential because superficial algorithmic similarities—such as iterative function application—can misleadingly suggest incrementalism when the actual contribution lies in enabling smooth, predictable, and architecture-agnostic accuracy-cost curves. The evaluation must assess whether the comparison is fair, covering identical data regimes, parameter budgets, and inference-time compute ranges.

**Core Evaluation Criteria:**
- **Forward/Backward Distinction**: Is there a clear technical comparison of how the forward and backward passes differ from DEQs and diffusion models, beyond high-level intuition?
- **Anytime vs. Fixed-Schedule Inference**: Does the method support true anytime prediction (stopping at arbitrary iteration counts with usable results), and is this contrasted against diffusion models that require a complete denoising chain or DEQs that depend on convergence tolerances?
- **OOD Compute Generalization**: Does the method maintain performance across a continuous range of test-time iteration values, including those unseen during training, and is this compared against baseline behavior where additional compute fails to improve accuracy?

---

**Principle 3: Exhaustive Ablation of Truncated Backpropagation, Recurrent Iteration Distributions, and Latent Initialization Choices**

**Definition:**  
The effectiveness of a recurrent-depth framework hinges on several coupled training design choices: the backpropagation window that caps memory, the distribution governing sampled recurrent depths, and the initial latent distribution. Reviewers must insist on systematic ablations that isolate the impact of each choice, because unjustified defaults can mask training instabilities or conflate the benefits of depth with those of increased memory or compute. For instance, truncated backpropagation-through-depth must be justified against gradient checkpointing not only in terms of memory but also wall-clock time and gradient fidelity, especially when batch items draw vastly different iteration counts. Similarly, the mean and variance of the recurrent iteration distribution must be shown to balance low-compute and high-compute inference performance rather than overfitting to a narrow compute regime. Ablation studies should demonstrate that performance saturates gracefully with the backpropagation window and that the initial latent distribution's impact diminishes as iterations increase, confirming the robustness of the iterative contraction mechanism.

**Core Evaluation Criteria:**
- **TBPTD Justification**: Is truncated backpropagation-through-depth justified against gradient checkpointing in terms of memory, FLOPs, batch consistency, and empirical performance saturation?
- **Distribution Parameter Sensitivity**: Is the training distribution parameter controlling expected depth ablated, and does the paper demonstrate how it shifts the accuracy-cost Pareto frontier across inference budgets?
- **Initial Latent Robustness**: Does the work ablate multiple initializations (e.g., standard normal, zero, conditioning, learnable) and show that final performance is largely decoupled from initialization given sufficient recurrent depth?

---

**Principle 4: Architecture-Agnostic Validation and Cross-Backbone Scaling in High-Dimensional Scientific Machine Learning**

**Definition:**  
A central claim of recurrent-depth simulators is architecture agnosticism—the ability to wrap arbitrary encoder-recurrent-decoder primitives without redesigning the training loop. Reviewers must evaluate whether this generality is substantiated across fundamentally different inductive biases, from spectral operators to attention mechanisms, and whether the accuracy-cost control property transfers consistently. Crucially, validation must extend to high-dimensional, memory-intensive regimes where the memory savings of weight sharing and truncated backpropagation become non-negligible. The evaluation should verify that the plug-and-play property does not come at the cost of architectural incompatibility or hidden tailoring, and that efficiency claims hold across backbones rather than being idiosyncratic to a single model class. This principle ensures that the framework is a genuine scientific computing tool rather than a niche modification applicable only to low-dimensional or single-architecture settings.

**Core Evaluation Criteria:**
- **Cross-Architecture Fidelity**: Are multiple distinct backbones (e.g., spectral, convolutional, attention-based) evaluated on diverse physical domains, and does the recurrent-depth mechanism preserve or improve physical fidelity in each?
- **High-Dimensional Scalability**: Are experiments conducted on high-dimensional benchmarks where memory constraints are meaningful, and do the reported savings (parameters, training memory) justify the design in realistic scientific computing scenarios?
- **Drop-In Compatibility**: Does the paper demonstrate that existing architectures can be adapted with minimal code changes, and are the modifications limited to wrapping existing blocks rather than requiring deep structural restructuring?

---

**Principle 5: Long-Horizon Physical Fidelity and Stability Assessment in Chaotic and Multi-Regime Dynamical Systems**

**Definition:**  
For neural simulators, aggregate error metrics such as one-step mean-squared error or even trajectory L2 error can obscure catastrophic failures in physical consistency, particularly over long rollouts or in chaotic systems where small perturbations grow exponentially. Reviewers must therefore demand evidence of physical faithfulness: the preservation of sharp shocks, soliton phases, and turbulent structures even in low-compute settings, and the absence of unphysical artifacts or error blow-up over time. In chaotic regimes, correlation-horizon metrics should supplement L2 error to quantify stability, while visual evidence must be accompanied by quantitative error scales, ground-truth comparisons, and shared colorbars to enable meaningful qualitative assessment. The principle is vital because the ultimate utility of a test-time controllable simulator depends on whether reduced compute produces merely noisier versions of correct physics or fundamentally incorrect dynamics. Without such multifaceted evaluation, claims of "physically faithful" simulation remain unsubstantiated.

**Core Evaluation Criteria:**
- **Physical Structure Preservation**: Does the model reproduce key physical features (shocks, solitons, conservation properties) at low compute budgets, and is this verified with side-by-side ground-truth visualizations and absolute error maps?
- **Chaotic Stability Metrics**: For chaotic systems, are correlation horizons (average and worst-case) reported across a sweep of thresholds, and do they improve monotonically with additional compute?
- **Visualization Rigor**: Are visualizations provided with ground truth, shared colorbars, quantitative scales, and baseline comparisons, enabling reviewers to assess physical fidelity rather than just aggregated numerical error?