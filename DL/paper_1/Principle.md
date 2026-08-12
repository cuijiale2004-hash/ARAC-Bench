**Principle 1: Distinction Between Biological Fractional Neuron Modeling and Generalizable End-to-End Trainable f-SNN Computational Frameworks**

**Definition:**  
Reviewers must assess whether the work rigorously demarcates itself from prior fractional-order neuron research that focuses on biological plausibility, signal approximation, or fixed-weight shallow dynamical analysis. The pivotal question is whether the proposed method constitutes a strict generalization of integer-order SNNs into a trainable deep-learning paradigm compatible with modern architectures (CNN, Transformer, GNN), rather than a narrow application of fractional calculus to biological modeling. High-quality contributions must demonstrate that fractional dynamics are embedded as a computational engine within the neuron itself, not merely used as an optimization tool or biological mimicry.

**Core Evaluation Criteria:**
- **Positioning Clarity:** Does the work explicitly differentiate its trainable, architecture-agnostic framework from prior biological f-LIF modeling and fractional Hopfield networks?
- **Generalization Formalism:** Is it formally shown that setting α = 1 exactly recovers standard integer-order SNNs, making the framework a strict superset rather than an unrelated alternative?
- **Architectural Integration:** Does the work demonstrate seamless substitution of integer-order neurons with fractional variants across diverse deep architectures without requiring structural redesign?

---

**Principle 2: Theoretical Depth in Expressivity, Irreducibility, and Robustness of Fractional-Order Spiking Dynamics**

**Definition:**  
Beyond qualitative descriptions of long-term memory via Mittag-Leffler functions, research in this subfield must provide rigorous mathematical guarantees that establish fundamental capabilities of fractional neurons unattainable by finite integer-order ensembles. This includes formal analysis of how fractional dynamics propagate through the spiking nonlinearity to affect output spike trains, and quantified robustness advantages. Theoretical contributions should not be restricted to single-neuron, constant-input regimes but should ideally address dynamic inputs and deep-network stability.

**Core Evaluation Criteria:**
- **Irreducibility Guarantees:** Are there formal proofs that f-SNN dynamics cannot be exactly reproduced by any finite linear combination of classical LIF neurons, with explicit error decay rates?
- **Spike-Level Expressivity:** Does the analysis extend beyond membrane potential to prove that spike-train outputs encode temporal information inaccessible to integer-order ensembles, confirming the advantage is not "washed out" by thresholding?
- **Robustness Bounds:** Are there concrete theorems on perturbation accumulation (e.g., sub-linear vs. linear growth) and spike-timing sensitivity under input corruption?
- **Dynamic Input Coverage:** Does theoretical analysis address time-varying/dynamic inputs, or is it honestly acknowledged as restricted to constant-input regimes?

---

**Principle 3: Large-Scale Empirical Benchmarking, Computational Cost Transparency, and Energy Efficiency Rigor**

**Definition:**  
Fractional-order recursions introduce nonlocal memory kernels with inherent computational overhead. Reviewers must evaluate whether the work provides transparent, comprehensive empirical benchmarks on standard and large-scale datasets, including wall-clock training/inference speed, peak memory consumption, and granular energy accounting. Energy analysis must move beyond synaptic operations to include neuron-intrinsic fractional update costs, ensuring claims of efficiency are grounded in complete hardware-aware estimates.

**Core Evaluation Criteria:**
- **Dataset Scale Coverage:** Are results reported on large-scale benchmarks (e.g., ImageNet) beyond small neuromorphic or graph datasets?
- **Speed and Memory Benchmarks:** Are there explicit comparisons of training/inference FPS and peak GPU memory against standard SNN baselines under identical architectures?
- **Memory Optimization:** Are practical approximations (e.g., truncated history windows, fractional adjoint methods) evaluated for their speed-memory-accuracy trade-offs?
- **Energy Granularity:** Does energy analysis include both synaptic operations and neuron-intrinsic fractional update costs, with layer-wise or total-network breakdowns?

---

**Principle 4: Characterization, Interpretability, and Training Stability of the Fractional Order Hyperparameter α**

**Definition:**  
The fractional order α is the central degree of freedom governing memory decay. High-quality research must rigorously characterize its sensitivity, task-dependent interpretability, and trainability. This includes assessing whether α can be learned end-to-end, how its optimal value correlates with data temporal sparsity or long-range dependencies, and whether the authors transparently report numerical instabilities—such as gradient explosion near α → 0 due to Gamma-function singularities—and propose concrete mitigation strategies.

**Core Evaluation Criteria:**
- **Sensitivity and Ablation:** Are systematic ablations provided across diverse datasets to reveal how accuracy varies with α?
- **Interpretability:** Is there a clear empirical or theoretical narrative linking smaller α (stronger history dependence) to sparser or longer-horizon datasets?
- **Learnable α Feasibility:** Does the work experimentally validate joint optimization of α with synaptic weights, and honestly report failure modes or stability issues in extreme regimes?
- **Automation Pathways:** Are concrete future directions (e.g., distributed fractional operators) proposed to overcome stability barriers and eliminate reliance on manual grid search?

---

**Principle 5: Cross-Domain Generalization, Robustness Validation, and Fair Comparison with Advanced Integer-Order Variants**

**Definition:**  
The practical value of an f-SNN framework depends on its consistent performance gains and robustness advantages across diverse modalities (neuromorphic vision, static images, graph data, recurrent architectures) and its standing relative to modern advanced integer-order spiking neurons (ALIF, GLIF, CLIF, PSN). Reviewers should evaluate the breadth of validation, the multi-dimensionality of robustness tests, and whether baseline comparisons are fair and contemporary.

**Core Evaluation Criteria:**
- **Cross-Domain Consistency:** Does the method demonstrate stable improvements across event-based, static, and graph-structured tasks, as well as recurrent architectures?
- **Multi-Dimensional Robustness:** Are robustness tests comprehensive (noise injection, occlusion, temporal jitter/truncation, frame discard, structural edge dropping) with quantitative degradation curves?
- **Baseline Fairness and Modernity:** Are comparisons conducted under identical architectural backbones? Are advanced integer-order variants discussed or evaluated as baselines rather than only basic LIF/IF?
- **Implementation Transparency:** Does the work clarify implementation differences between SNN toolboxes (e.g., reset semantics, input decay scaling) that could confound cross-framework results?