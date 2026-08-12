**Principle 1: Formal Theoretical Separation Between Next-Token Prediction and RL-Driven Length Generalization in Autoregressive Models**

**Definition:**  
This principle evaluates whether research provides rigorous theoretical analysis that formally separates pure next-token prediction from reinforcement learning augmented training in autoregressive language models, specifically regarding the emergence of length generalization when long reasoning chains are rare. The assessment centers on whether the theory identifies mechanistic optimization drivers—such as RL's effective up-sampling of rare long demonstrations through reward-weighted losses and the resulting length calibration dynamics—and proves non-trivial sample-complexity gaps between the two training paradigms. Crucially, reviewers must examine whether simplified theoretical architectures (e.g., linear autoregressive models) are explicitly justified as capturing essential phenomenology independent of full transformer complexity, with clear discussion of what the simplification preserves and omits. The work should derive testable quantitative predictions, such as critical thresholds for the proportion of long sequences required for greedy decoding to generalize, and prove convergence rates for RL rounds. Finally, the principle demands that authors address how their theoretical results might extend to non-convex, non-linear architectures or honestly acknowledge the limitations of their simplified framework.

**Core Evaluation Criteria:**
- **Formal Separation Rigor:** Does the work prove a concrete sample-complexity, optimization, or statistical separation between next-token prediction and RL-augmented training?
- **Mechanistic Identification:** Are the optimization pressures driving length increase and generalization explicitly identified (e.g., reward-weighted upsampling, length calibration) rather than merely observed?
- **Simplification Justification:** Is the use of simplified models robustly defended, with honest discussion of limitations regarding non-linear transformers and what the abstraction omits?
- **Threshold Predictivity:** Does the theory derive specific, empirically validated quantitative predictions such as critical mixture proportions or RL round complexity?
- **Architectural Generality:** Does the analysis explicitly address whether core results are architecture-independent or provide a plausible path to extending results to transformer-based LLMs?

---

**Principle 2: Empirical Validation and Cross-Domain Generalization from Synthetic Parity to Real-World LLM Mathematical Reasoning**

**Definition:**  
This principle assesses the breadth and rigor of empirical validation required to support theoretical claims about RL post-training in large language models, particularly the generalization of phenomena from synthetic tasks to realistic settings. Reviewers should evaluate whether experiments span diverse task complexities—from controlled synthetic problems like parity to real-world mathematical reasoning benchmarks such as GSM8K and MATH—and whether they include both models trained from scratch and pre-trained foundation models fine-tuned with standard pipelines. The evaluation must verify that core theoretical predictions, including critical mixture thresholds, RL-induced length increases, and sample-efficiency advantages over supervised fine-tuning alone, are consistently replicated across different architectures and scales. Attention should be paid to whether the experimental design systematically varies key parameters like the proportion of long chain-of-thought data, controls for confounding factors such as pre-training checkpoint maturity, and reports sufficient implementation details for reproducibility. Ultimately, the principle distinguishes high-quality work that bridges theory and practice from studies that remain confined to narrow, unrealistic experimental settings.

**Core Evaluation Criteria:**
- **Cross-Task Consistency:** Are theoretical predictions validated across multiple task types (synthetic parity, arithmetic, grade-school/hard mathematics) with diverse computational depth?
- **Model Scale and Architecture Diversity:** Do experiments include both small models trained from scratch and instruction-tuned foundation models to test generality?
- **Controlled Parameter Variation:** Are key parameters such as mixture proportions, RL sampling temperatures, and pre-training checkpoint timing systematically ablated?
- **Reproducibility Transparency:** Are implementation details, hyperparameters, random seeds, and computational costs reported with sufficient granularity?
- **Phenomenon Isolation:** Does the experimental design isolate RL-specific gains from confounders such as continued pre-training, pre-existing knowledge in base models, or evaluation artifacts?

---

**Principle 3: Robustness to Noisy and Partial Chain-of-Thought Data in Theoretical and Empirical Analyses of RL Post-Training**

**Definition:**  
This principle evaluates whether analyses of RL post-training account for realistic data imperfections that violate idealized assumptions of perfectly correct and complete chain-of-thought demonstrations. High-quality research must investigate how noise—whether isolated to final answer tokens or propagated through entire reasoning chains—and heterogeneous reasoning path lengths affect the learning dynamics and critical thresholds identified in clean settings. Reviewers should examine whether authors experimentally or theoretically characterize phase transition shifts when training mixtures include partial or medium-length chains alongside short and long sequences. The work should further demonstrate whether RL post-training possesses robustness properties, such as the ability to erase pre-training noise or overcome initially low correlations with the target, and whether these robustness claims are accompanied by sample-complexity characterizations. This principle ensures that theoretical insights are not artifacts of sanitized data environments but remain relevant to the messy, web-scale distributions encountered in practice.

**Core Evaluation Criteria:**
- **Noise Model Coverage:** Does the work analyze multiple realistic noise types (e.g., final-token corruption, snowballing reasoning-chain errors) and their impact on learning?
- **Partial Chain Dynamics:** Are settings with heterogeneous reasoning lengths (short, medium, long chains) explored to test the limits of the simplified mixture theory?
- **Threshold Shift Characterization:** Does the analysis quantify how noise and partial data alter critical phase transitions or sample-complexity requirements?
- **RL Recovery Evidence:** Is there empirical or theoretical evidence that RL can overcome pre-training noise, and are the associated overheads characterized?
- **Realism Discussion:** Do the authors honestly discuss which idealized assumptions are likely violated in practice and how this affects their conclusions?

---

**Principle 4: Causal Mechanistic Attribution of Length Increase and Sample-Efficiency Gains During RL Beyond Observational Correlation**

**Definition:**  
This principle distinguishes research that provides causal, mechanistic explanations for RL-induced training phenomena from work that merely documents observational correlations between response length increase and performance gains. Reviewers must assess whether the authors establish a formal, optimization-theoretic link between RL algorithms and emergent behaviors—such as proving that length increase arises from asymmetric learning difficulty between short and long sequences rather than from representational capacity alone. The evaluation focuses on whether the work demonstrates novelty beyond prior empirical observations by deriving the dynamics of effective data re-weighting during RL and predicting quantifiable outcomes like exponential convergence of mixture proportions across rounds. Furthermore, the principle requires that researchers disentangle genuine learning mechanisms from surface-level performance improvements, ensuring that claims about RL's benefits are rooted in identifiable training dynamics rather than black-box model capability. Such mechanistic depth is essential for transforming anecdotal training recipes into reproducible scientific understanding.

**Core Evaluation Criteria:**
- **Causal vs. Correlational Reasoning:** Does the work establish a causal chain from RL optimization dynamics to length increase and generalization, avoiding post-hoc rationalization?
- **Optimization-Theoretic Depth:** Are training dynamics derived formally (e.g., via convergence proofs or fixed-point analysis) rather than inferred solely from end-point accuracy?
- **Distinction from Representation Arguments:** Is the RL learning advantage disentangled from the known representational benefits of longer contexts (e.g., circuit depth)?
- **Novel Predictive Power:** Does the theory generate non-obvious, testable predictions (e.g., exponential convergence of effective mixture weights, length calibration during SFT)?
- **Observational Novelty:** Is the work clearly distinguished from prior empirical observations by explaining the underlying mechanism rather than rediscovering surface phenomena?

---

**Principle 5: Derivation of Actionable Algorithmic Guidance for RL Reward Design and Training Pipelines**

**Definition:**  
This principle evaluates whether theoretical insights into RL post-training are translated into concrete, implementable recommendations for designing training pipelines and reward functions in large language models. Reviewers should assess whether the authors derive specific prescriptions from their analysis—such as optimal timing for length penalties, strategies for mixing short and long supervised fine-tuning data, or criteria for selecting pre-training checkpoints before RL—that are validated through controlled experiments. The work must move beyond descriptive analysis to demonstrate how understanding optimization dynamics can inform practical decisions, such as when penalizing response length improves inference efficiency without sacrificing accuracy. Furthermore, the principle examines whether the authors honestly scope the applicability of their guidance to specific RL paradigms (e.g., outcome-based versus process-based rewards) and acknowledge where theoretical predictions require further empirical testing at production scale. High-quality research in this area closes the loop between theory and engineering practice.

**Core Evaluation Criteria:**
- **Concrete Prescriptions:** Does the work derive specific, actionable recommendations from theoretical analysis (e.g., when to apply length penalties, how to schedule curriculum)?
- **Experimental Validation:** Are the practical suggestions tested empirically with ablation studies or controlled comparisons?
- **Efficiency Trade-offs:** Does the guidance address practical resource constraints such as inference-time compute, sample efficiency, or GPU-hour budgets?
- **Paradigm Scoping:** Are limitations regarding reward types (end-to-end versus chain-of-thought verification) and RL algorithms explicitly discussed?
- **Industry Relevance:** Do the authors connect their findings to prevailing LLM training practices in a way that practitioners can act upon?