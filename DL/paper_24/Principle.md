**Principle 1: Preservation of Exact BPTT Gradient Fidelity and Mathematical Equivalence in Memory-Efficient SNN Training**

**Definition:**  
In BPTT-based SNN training, preserving exact gradient fidelity is paramount because many existing memory-saving approaches—such as online learning, BPTT-to-BP, and reversible networks—introduce approximation errors, architectural constraints, or gradient bias that degrade temporal learning capabilities. A core evaluation criterion is therefore whether the proposed method achieves strict mathematical equivalence to standard BPTT, ensuring that memory reduction stems solely from engineering optimizations rather than algorithmic simplifications. Reviewers must assess whether the authors formally prove or rigorously demonstrate that every component, including gradient checkpointing, spike compression, and spatio-temporal partitioning, leaves the backward computational semantics intact. Furthermore, the work must empirically validate this equivalence through full-training convergence experiments across multiple architectures and datasets, showing no systematic accuracy loss. Any observed numerical deviations should be explicitly traced to backend-specific floating-point behaviors rather than methodological bias. Finally, the work should be situated clearly against approximate alternatives to highlight the practical value of lossless training in scaling SNNs to deeper architectures and longer time horizons.

**Core Evaluation Criteria:**
- **Verification of Lossless Property**: Does the work formally establish or rigorously demonstrate that gradient checkpointing, spike compression, and all partitioning strategies preserve the exact BPTT backward pass without algorithmic approximation?
- **Empirical Accuracy Validation**: Are full-training convergence results reported across diverse architectures and tasks, confirming no systematic accuracy degradation relative to standard BPTT baselines?
- **Distinction from Approximate Alternatives**: Does the paper clearly contrast its lossless approach with online learning, BPTT-to-BP, and reversible networks in terms of gradient fidelity, architectural constraints, and task applicability?
- **Controlled Numerical Analysis**: Are minor numerical discrepancies explicitly attributed to non-associative floating-point accumulation or backend differences, with evidence that they are bounded and non-systematic?

---

**Principle 2: Fine-Grained Decomposition and Scaling Analysis of Memory-Computation Trade-offs in Spatio-Temporal SNN Training**

**Definition:**  
The spatio-temporal dynamics of SNNs create highly non-uniform memory and computation loads across layers and time steps, making aggregate metrics such as overall speedup or peak memory reduction insufficient for rigorous evaluation. Reviewers expect a fine-grained decomposition that reveals how memory evolves during forward and backward passes, identifies critical bottleneck segments, and quantifies the overhead of each introduced operation. This includes detailed runtime profiling at the layer or segment level to disentangle the costs of gradient checkpointing recomputation, spike compression and decompression, and segment restoration. Additionally, the scaling behavior of these overheads must be characterized as a function of key variables such as the number of checkpointed segments, temporal partitioning factors, and network depth. Such decomposition not only strengthens the argument for scalability but also exposes nonlinear interactions between spatial and temporal partitioning that hidden aggregate statistics would otherwise obscure. Ultimately, this level of analytical depth distinguishes robust systems engineering from superficial benchmarking.

**Core Evaluation Criteria:**
- **Layer-wise and Stage-wise Runtime Profiling**: Does the work provide per-layer or per-segment forward and backward time costs, separately quantifying recomputation, compression, and decompression overheads?
- **Scaling Behavior Analysis**: Is the time overhead characterized as a function of the number of GC segments and temporal partitioning factors, revealing linear or nonlinear scaling trends?
- **Peak Memory Attribution**: Does the analysis trace the global peak memory to specific critical layers or segments, and explain how optimization shifts the bottleneck across the network?
- **Cost-Benefit of Individual Operations**: Are the computational costs of spike compression and greedy restoration shown to be negligible or well-controlled relative to total training time?

---

**Principle 3: Component Isolation and Synergy Validation for Multi-Stage Memory Optimization Pipelines in SNNs**

**Definition:**  
Modern memory optimization pipelines for SNNs typically integrate multiple complementary techniques—layer-wise gradient checkpointing, lossless spike compression, spatial partitioning, temporal partitioning, and greedy segment restoration—raising the risk of black-box reasoning where overall improvement is attributed indiscriminately to the entire system. A rigorous review must insist on isolated ablations that quantify the individual contribution of each component to both memory reduction and computational throughput. Beyond isolation, the evaluation should examine whether combining components yields genuine synergy, such as spike compression creating memory headroom that enables more aggressive spatio-temporal partitioning than would be possible otherwise. Reviewers should also scrutinize the sensitivity of the pipeline to optimization levels and partitioning thresholds, mapping the full Pareto frontier of memory versus speed trade-offs. This component-level transparency is essential to establish that the proposed multi-stage strategy is a deliberately engineered solution rather than an accidental accumulation of generic tricks. It also ensures that subsequent researchers can build upon verified causal insights rather than correlative observations.

**Core Evaluation Criteria:**
- **Isolated Ablation of Each Component**: Are the independent effects of gradient checkpointing, spike compression, spatial partitioning, temporal partitioning, and greedy restoration separately reported?
- **Synergy Quantification**: Does the work demonstrate that combined components produce super-additive benefits, such as compression enabling partitioning strategies that require lower memory floors?
- **Sensitivity to Hyperparameters**: Is the sensitivity to optimization levels and partitioning rules analyzed, showing how memory and throughput vary across the configuration space?
- **Avoidance of Black-Box Reasoning**: Does the paper refrain from attributing all gains to the full pipeline indiscriminately, and instead establish causal links between specific design choices and observed metrics?

---

**Principle 4: Explicit Scope Delimitation and Practical Automation for Layer-wise SNN Memory Optimization**

**Definition:**  
Because memory optimization methods for SNNs operate under distinct training paradigms—layer-wise versus step-wise or strict online updates—the scope of applicability must be explicitly delimited and honestly justified from the outset. Reviewers must evaluate whether the authors clearly identify which training regimes are supported, explain why the chosen regime represents the mainstream bottleneck for modern medium- and large-scale SNNs, and acknowledge limitations in unsupported settings without defensive obfuscation. Equally important is the degree of practical automation: an ideal pipeline should function as an optimization pass that automatically restructures computation graphs with minimal user intervention, exposing only high-level knobs such as an optimization level. When manual choices remain necessary—such as user-specified spatial or temporal partitioning rules—the paper must transparently position them as current limitations or configurable extensions rather than hidden implementation burdens. Finally, the work should demonstrate broad compatibility across neuron models, architectures, and complementary techniques like automatic mixed precision to substantiate claims of general applicability.

**Core Evaluation Criteria:**
- **Clear Training Paradigm Scope**: Does the paper explicitly state whether it targets layer-wise, step-wise, or online training, and honestly discuss limitations in unsupported regimes?
- **Automation and User Effort**: Is the pipeline presented as an automatic optimization pass with minimal user intervention, and are manual configurations clearly documented as limitations?
- **Framework Integration and Reproducibility**: Is there a concrete commitment to open-source release and integration into mainstream SNN frameworks, alongside sufficient implementation detail for reproduction?
- **Compatibility and Generality**: Is the method validated across diverse neuron models, network architectures, and tasks, and shown to be compatible with other memory-saving techniques such as AMP or low-memory optimizers?

---

**Principle 5: Hardware Accessibility and Scalability Impact on Democratizing Large-Scale SNN Research**

**Definition:**  
Beyond algorithmic novelty, a defining hallmark of high-impact memory optimization research is its ability to tangibly lower hardware barriers and democratize access to large-scale SNN training. Reviewers should assess whether the paper provides concrete case studies demonstrating previously infeasible scenarios—such as training large video transformers on commodity GPUs, increasing batch size by orders of magnitude, or extending temporal resolution without memory overflow. These scalability claims must be contextualized against the actual hardware constraints commonly faced by the research community, including the prevalence of memory-limited GPUs. The work should also articulate a clear community impact narrative, explaining how reducing peak memory enables broader participation from resource-constrained research groups and facilitates the scaling of SNNs toward deeper architectures and longer sequences. Without such evidence, memory savings remain abstract engineering achievements rather than transformative practical contributions.

**Core Evaluation Criteria:**
- **Tangible Scalability Gains**: Are case studies provided that demonstrate previously infeasible training regimes, such as larger batch sizes, longer time steps, or large models on commodity hardware?
- **Hardware Barrier Reduction**: Does the paper quantify memory savings in terms of accessibility, such as enabling training on widely available GPUs rather than high-end server hardware?
- **Community Impact Argument**: Does the work explicitly discuss how democratizing memory-efficient training benefits researchers with limited computational resources?
- **Comparison with Hardware Constraints**: Are the reported gains contextualized against the actual memory limitations of hardware commonly used in the community?