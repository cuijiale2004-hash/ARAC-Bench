**Principle 1: Real-Device Energy and Memory Footprint Validation Against Theoretical Proxies in On-Device SNN FSCIL**

**Definition:**  
In on-device FSCIL research using SNNs, the primary motivation for adopting spiking architectures is their potential for ultra-low-power inference and compatibility with neuromorphic hardware. Therefore, efficiency claims must be validated through actual hardware measurements rather than purely theoretical proxies such as SynOps or FLOPs, which can misrepresent real-world energy and latency profiles. Reviewers must scrutinize whether the paper reports real-device energy consumption, training and inference runtime, and peak memory usage on resource-constrained platforms such as Jetson Orin AGX or neuromorphic chips. The distinction between theoretical energy estimates and empirical measurements should be clearly articulated, and comparisons must specify whether they are against ANN or SNN baselines. Furthermore, the experimental setup should demonstrate that the method remains practical under strict memory budgets typical of edge devices.

**Core Evaluation Criteria:**
- **Real-device measurement**: Does the paper report actual energy, latency, or power measured on physical edge hardware rather than relying exclusively on theoretical calculations?
- **Clarity of comparison context**: Are energy efficiency claims accompanied by explicit baseline definitions (ANN vs. SNN) and standardized measurement protocols?
- **Memory and parameter footprint**: Is the peak memory consumption and parameter size reported and compared against both ANN-based PEFT and SNN alternatives to verify on-device feasibility?
- **Training vs. inference efficiency**: Does the evaluation distinguish between the efficiency of base-session training and incremental-session adaptation, given that FSCIL involves distinct computational phases?

---

**Principle 2: Cross-Paradigm Baseline Fairness and ANN-SNN Comparative Rigor for Edge Incremental Learning**

**Definition:**  
Because SNN-based FSCIL operates at the intersection of neuromorphic computing and continual learning, reviewers must evaluate whether the manuscript establishes its value proposition relative to both SNN-specific and ANN-based state-of-the-art methods. A common pitfall is comparing only against converted or weak SNN baselines while omitting strong parameter-efficient ANN methods that may achieve comparable accuracy. The review should assess whether backbone architectures, pretraining regimes, and parameter counts are matched fairly across paradigms. Additionally, within the SNN literature, the method must be differentiated from closely related techniques such as soft subnetworks, lottery ticket pruning, and surrogate gradient training. Only through rigorous cross-paradigm and intra-paradigm comparisons can the paper claim genuine advancement in the on-device FSCIL landscape.

**Core Evaluation Criteria:**
- **ANN baseline inclusion**: Does the paper compare against modern ANN-based parameter-efficient FSCIL methods (e.g., prompt tuning, adapters) under comparable backbone and dataset conditions?
- **SNN intra-paradigm comparisons**: Are alternative SNN training strategies, such as surrogate gradient methods and other dynamic threshold mechanisms, empirically evaluated alongside the proposed approach?
- **Novelty differentiation**: Does the work clearly articulate how its sparsity-aware dynamics differ from soft-subnetwork or subnet-based continual learning approaches, beyond superficial similarities?
- **Fairness of experimental conditions**: Are comparisons controlled for backbone capacity, pretraining data, and incremental session protocols to prevent confounding factors?

---

**Principle 3: Mechanistic Clarity and Differentiation of Sparsity-Aware Neuronal Dynamics from Parameter-Freezing Subnetworks**

**Definition:**  
A central technical contribution in SNN-based FSCIL is the use of sparsity-aware neuronal dynamics to mitigate catastrophic forgetting by stabilizing base-class representations while allowing plasticity for novel classes. Reviewers must demand mechanistic clarity regarding how sparsity is defined, measured, and enforced at the neuron level, and how dynamic threshold regulation functionally preserves synaptic traces encoding prior knowledge. The explanation must transcend heuristic descriptions and establish a causal link between firing-rate stability and knowledge retention. Furthermore, the work must explicitly differentiate its neuronal division mechanism from structurally similar approaches such as parameter freezing or soft subnetworks. Empirical validation should include visualization of firing rates, ablation of stable versus adaptive neurons, and evidence that base-class performance degradation is minimized across incremental sessions.

**Core Evaluation Criteria:**
- **Mechanistic explainability**: Is the sparsity-aware threshold regulation mechanism explained with precise mathematical formulations and intuitive biological or functional analogies?
- **Distinction from related methods**: Does the paper explicitly contrast its neuron-level dynamic thresholds with soft-subnetwork, lottery ticket, or parameter-isolation approaches in terms of trainable parameters and computational overhead?
- **Empirical evidence of stability**: Are firing-rate traces, ablation studies, and base-class retention metrics provided to verify that sparsity dynamics genuinely alleviate catastrophic forgetting?
- **Sensitivity and robustness**: Does the work analyze the sensitivity of the adaptive ratio and threshold hyperparameters, and demonstrate robustness across varying network depths and architectures?

---

**Principle 4: Theoretical Convergence and Empirical Competitiveness of Gradient-Free Spike Optimization in Incremental Training**

**Definition:**  
Training deep SNNs requires handling the non-differentiability of spike activation functions, and the choice of gradient approximation strategy fundamentally impacts learning stability and final performance. When a paper proposes gradient-free optimization such as zeroth-order methods for on-device FSCIL, reviewers must evaluate both the theoretical rigor—convergence bounds, error decomposition, and sample complexity—and the empirical competitiveness against established surrogate gradient techniques. The justification must address why standard surrogate gradients are insufficient in the specific FSCIL setting, particularly regarding gradient vanishing in saturation regions. Error analysis should decompose approximation and sampling errors, and empirical comparisons must cover multiple surrogate variants to ensure the conclusion is not an artifact of a poorly tuned baseline.

**Core Evaluation Criteria:**
- **Theoretical convergence guarantees**: Does the paper provide formal convergence bounds for the zeroth-order estimator in the non-convex SNN optimization landscape?
- **Error decomposition**: Are the sources of gradient approximation error (e.g., Gaussian smoothing bias and sample-average variance) formally analyzed and bounded?
- **Empirical comparison breadth**: Is the proposed optimizer compared against a diverse set of surrogate gradient functions under identical FSCIL protocols to establish consistent superiority?
- **Computational justification**: Does the paper quantify the overhead introduced by zeroth-order sampling and demonstrate that it remains negligible relative to the overall on-device training budget?

---

**Principle 5: Bias Mitigation and Computational Efficiency of Training-Free Prototype Projection in Few-Shot Incremental Sessions**

**Definition:**  
In few-shot incremental sessions, class prototypes derived from scarce samples are inherently biased and prone to overfitting, while adaptation mechanisms must remain computationally lightweight to preserve the edge-deployment advantages of SNNs. Reviewers must therefore assess whether the proposed prototype update strategy—such as orthogonal subspace projection—effectively enhances discriminability without introducing prohibitive matrix operations or base-class bias. The evaluation should include comparisons against alternative projection and feature-space regularization methods commonly used in FSCIL. Crucially, the paper must analyze whether the adaptation remains training-free or introduces minimal overhead, as complex computations such as large matrix inversions may conflict with neuromorphic deployment constraints. Finally, the balance between base-class retention and novel-class accuracy should be quantified through harmonic mean metrics and misclassification analysis.

**Core Evaluation Criteria:**
- **Bias and discriminability analysis**: Does the paper analyze whether the prototype projection introduces bias toward base classes, and does it report harmonic mean or base/novel accuracy disaggregation?
- **Comparative projection evaluation**: Are alternative projection techniques (e.g., weight space rotation, subspace regularization) compared in terms of both accuracy and runtime cost?
- **Computational overhead control**: Is the incremental update cost quantified, and is it shown to be negligible relative to feature extraction, thereby maintaining on-device feasibility?
- **Training-free adaptation**: Does the adaptation mechanism avoid introducing additional trainable parameters during incremental sessions, preserving the parameter-efficiency required for edge continual learning?