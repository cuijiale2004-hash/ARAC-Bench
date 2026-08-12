**Principle 1: Layer-wise Credit Attribution and Mechanistic Interpretability in Feedback-Regulated Equilibrium Propagation Networks**

**Definition:**  
In biologically plausible learning frameworks such as Equilibrium Propagation, architectural modifications that accelerate convergence—like feedback regulation—can inadvertently attenuate error signals to early layers, reducing deep networks to mere reservoir systems where only readout layers learn. Reviewers consistently demand conclusive evidence that the proposed method achieves genuine layer-wise credit assignment rather than trivial output-layer fitting. A rigorous submission must therefore dissect the depth of gradient propagation through controlled ablations (e.g., freezing lower layers) and quantify gradient alignment with backpropagation across the hierarchy. Furthermore, the dynamical origins of improved stability must be mechanistically explained via measurable quantities such as finite-time Lyapunov exponents or spectral radius, establishing a causal chain from network dynamics to learning outcomes. Without this level of mechanistic scrutiny, claims of scalable bio-plausible training remain vulnerable to the criticism that deep layers are functionally inert.

**Core Evaluation Criteria:**
- **Depth of Error Signal Propagation**: Does the paper experimentally verify that early hidden layers receive meaningful gradients (e.g., via layer-freezing ablations, per-layer error analysis, or gradient cosine similarity with backpropagation)?
- **Dynamical Systems Grounding**: Are convergence improvements mechanistically linked to quantifiable dynamical properties (e.g., spectral radius, finite-time maximum Lyapunov exponent) rather than phenomenologically described?
- **Rejection of Trivial Mechanisms**: Does the work explicitly exclude alternative explanations such as reservoir-like behavior, where only the final layer or readout head drives performance?

---

**Principle 2: Empirical Rigor and Baseline Legitimacy in Biologically Plausible Local Learning Benchmarks**

**Definition:**  
Empirical claims regarding speed, accuracy, and stability must be supported by statistically robust experiments with appropriate baselines and task complexities. Because EP occupies a niche between standard backpropagation and recurrent backpropagation, baseline selection requires careful justification; comparing EP-trained RNNs to feedforward BP networks can be misleading if the recurrent dynamics are not properly accounted for. Furthermore, experiments must avoid overparameterization artifacts—where excessive depth or width renders lower-layer learning unnecessary—and should include repeated trials with reported variance to ensure reproducibility. Controlled ablations must isolate the contribution of each architectural component, while dataset choices should reflect meaningful complexity gradients that test the limits of the proposed method. Only through such empirical discipline can reviewers trust that observed gains are intrinsic to the bio-plausible mechanism rather than artifacts of experimental design.

**Core Evaluation Criteria:**
- **Baseline Appropriateness**: Are comparisons fair and justified—e.g., comparing recurrent EP architectures against recurrent backpropagation or feedback alignment rather than feedforward backpropagation where dynamics differ fundamentally?
- **Statistical Robustness and Ablation Completeness**: Are experiments repeated over multiple seeds with variance reported, and are key hyperparameters (feedback scaling, iteration counts, residual topology) systematically ablated and interpreted causally?
- **Controlled Capacity-to-Task Matching**: Does the experimental design avoid conclusions drawn from heavily overparameterized networks on trivial tasks (e.g., very deep nets on MNIST) where lower-layer learning may be unnecessary?

---

**Principle 3: Hardware Noise Resilience and Biological Plausibility Validation for Neuromorphic Deployment**

**Definition:**  
Claims of biological plausibility and suitability for neuromorphic or physical neural networks carry specific experimental obligations. Biological and analog hardware substrates are inherently noisy, exhibiting weight imprecision, state variability, and quantization errors. Therefore, a method targeting these domains must demonstrate robustness to realistic noise profiles—such as Gaussian perturbations on weights and time-varying state noise—rather than assuming idealized digital precision. Additionally, the work must situate its biological framing within established neuroscience literature and acknowledge prior art in weak-feedback bio-plausible learning to avoid superficial novelty claims. The absence of such validation risks reducing hardware-oriented claims to speculation, undermining both scientific credibility and practical utility.

**Core Evaluation Criteria:**
- **Noise and Imperfection Robustness**: Does the paper quantify performance under realistic hardware noise profiles—such as weight quantization, additive weight noise, and time-varying state noise—with noise magnitude normalized to signal dynamic range?
- **Neuromorphic Applicability Justification**: Are claims of hardware friendliness supported by concrete evidence regarding local computation, avoidance of explicit activation derivatives, and measured energy or wall-clock time advantages?
- **Integration with Bio-Plausible Literature**: Does the work adequately contextualize itself within established biologically plausible learning paradigms (e.g., local representation alignment, dendritic cortical microcircuits) and acknowledge prior uses of weak feedback?

---

**Principle 4: Theoretical Transparency and Approximation Fidelity in Non-Energy-Based Equilibrium Dynamics**

**Definition:**  
When extending equilibrium propagation beyond strict energy-based models to vector-field or asymmetric recurrent dynamics, the theoretical framework must be presented with precision and honesty. Approximations—such as the equivalence between EP and BP under infinitesimal weak feedback, or the lack of guaranteed convergence to energy stationary points—must be explicitly stated and their implications discussed. Algorithmic descriptions should avoid unnecessary notational confusion or obfuscating complexity that masks simple underlying mechanics. Theoretical derivations should clarify whether the dynamics truly implement canonical EP or an approximate variant, and how this affects correctness guarantees. Transparency in these matters is essential because technical ambiguities directly erode reviewer confidence in the method's soundness.

**Core Evaluation Criteria:**
- **Explicit Approximation Disclosure**: Does the manuscript clearly state theoretical deviations from canonical energy-based EP (e.g., asymmetric vector-field dynamics) and the resulting loss of convergence guarantees or exact gradient equivalence?
- **Notational and Conceptual Clarity**: Is the algorithmic presentation free of overloaded symbols, contradictory figures or equations, and unnecessary complexity that obscures the underlying mechanics?
- **Theoretical or Empirical Verification of Equivalence**: Are claimed equivalences to backpropagation in limiting regimes (e.g., weak feedback) supported by derivation or by empirical gradient-alignment analysis?

---

**Principle 5: Scalability and Cross-Modal Generalization Beyond Static Inputs in Convergent RNN Learning**

**Definition:**  
A central criterion for practical EP is whether improvements hold as networks scale in depth, width, and task complexity. Proposals that demonstrate compelling results on simple static datasets like MNIST must address whether the method generalizes to deeper convolutional architectures, more complex vision benchmarks, and—critically—temporal or sequential modalities that exploit the recurrent nature of RNNs. The evaluation should scrutinize whether the convergence acceleration and stability gains are maintained under these scaling pressures, and whether performance gaps relative to backpropagation are honestly characterized rather than obscured. Clear articulation of current limitations—such as the restriction to static-input settings—helps set realistic expectations and defines meaningful pathways for future work. Ultimately, scalability and honest scope delineation separate niche methodological curiosities from genuinely impactful contributions to brain-inspired computing.

**Core Evaluation Criteria:**
- **Architectural Scalability**: Are results reported across a substantial depth range (e.g., 2 to 20+ layers), and do residual or skip-connection remedies demonstrably recover performance in deep regimes on sufficiently complex tasks?
- **Task Complexity and Modality Scope**: Does the paper address applicability beyond simple static-image benchmarks, or provide a candid justification for limiting evaluation to static inputs given the recurrent nature of the model?
- **Honest Gap Analysis**: Are performance deficits on more challenging datasets (e.g., CIFAR-10) openly reported and interpreted, with clear boundaries drawn between currently achievable and aspirational application domains?