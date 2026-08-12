**Principle 1: System-Level Engineering Innovation vs. Algorithmic Novelty in Scaling RL for Real-World Computer Control Agents**

**Definition:**  
Reviewers must evaluate whether the work presents principled, non-trivial engineering innovations that overcome fundamental bottlenecks inherent in scaling reinforcement learning to real desktop operating systems, rather than merely applying existing RL algorithms with more compute. In agentic RL, stable orchestration of thousands of heterogeneous, crash-prone virtual environments introduces research-grade challenges—such as asynchronous experience collection, fault tolerance, and adaptive environment management—that are distinct from generic distributed systems. The evaluation should assess whether the authors identify these as core scientific bottlenecks, propose architecturally principled solutions, and position the contribution appropriately as an engineering/systems advance rather than a theoretical algorithmic breakthrough.

**Core Evaluation Criteria:**
- **Identification of Fundamental Bottlenecks**: Does the work clearly articulate why real-world desktop RL scaling is qualitatively harder than simulated or small-scale RL (e.g., VM crashes, application unresponsiveness, network latency under concurrency)?
- **Principled Systems Design**: Are the infrastructure solutions (e.g., decoupled AgentBench APIs, lightweight QEMU-in-Docker VMs, gRPC-based clustering) architecturally motivated, or are they brute-force operational patches?
- **Contribution Positioning**: Does the paper honestly situate its value as engineering-driven innovation (analogous to GPT-3’s scaling demonstration) rather than obscuring the lack of algorithmic novelty?
- **Necessity Ablations**: Does the paper provide evidence that the distributed infrastructure and interaction paradigm are necessary conditions for achieving the reported results, not merely sufficient?

---

**Principle 2: Quantitative Attribution and Disentanglement of Gains Between RL Framework and Pre-trained Base Model Capabilities**

**Definition:**  
When strong pre-trained vision-language or language models serve as the initialization for agent training, reviewers must rigorously evaluate whether the paper isolates the marginal contribution of the RL training framework from the base model’s inherent prior knowledge. In computer-use agent research, foundation models may already possess substantial GUI grounding and reasoning capabilities; therefore, claims of framework superiority require controlled decomposition. The evaluation demands explicit comparisons that separate base-model performance, offline distillation gains, and the incremental value of online environment-interactive RL.

**Core Evaluation Criteria:**
- **Base Model Baseline**: Is the zero-shot or SFT-only performance of the exact base model reported to establish a clear lower bound?
- **Scale-Stratified Comparisons**: Does the paper compare against larger pre-trained models (e.g., 72B parameters) to demonstrate that the smaller RL-tuned model achieves superior performance through methodological innovation rather than scale?
- **Component Ablations**: Are systematic ablations provided for each framework component (e.g., API-GUI paradigm, Entropulse, distributed training) while holding the base model constant?
- **Offline vs. Online RL Attribution**: Does the work distinguish gains from offline methods (DPO, BC) versus genuine online environment-interactive RL to justify the infrastructure investment?

---

**Principle 3: Comprehensive Reproducibility and Resource Transparency for Thousand-Scale Distributed Desktop RL Infrastructure**

**Definition:**  
For research claiming scalability to thousands of parallel desktop environments, complete disclosure of computational infrastructure, training duration, FLOPs, cluster topology, and virtualization parameters is essential for scientific assessment. Reviewers must evaluate whether the community can judge feasibility, attempt partial reproduction, and compare resource efficiency. Omissions of hardware specifications, cost estimates, or cluster configuration are considered significant flaws for systems-oriented RL work, as they prevent assessment of whether results are accessible only to well-resourced labs or represent efficient, generalizable engineering.

**Core Evaluation Criteria:**
- **Hardware Specification Completeness**: Are detailed specs provided for all clusters (training GPUs, environment CPUs, NUMA topology, memory hierarchy, interconnects)?
- **Cost and Duration Transparency**: Are training FLOPs, GPU-hours, wall-clock time per stage (SFT, RL, Entropulse), and total monetary cost explicitly reported?
- **Scalability Trade-off Analysis**: Does the paper analyze key hyperparameter bottlenecks (e.g., responses per prompt vs. sampling latency, batch size vs. gradient variance) and report optimal configurations?
- **Minimum Viable Configuration**: Is the minimum cluster size or resource requirement for running the pipeline disclosed to assess reproducibility outside of industrial labs?
- **Artifact Availability**: Is source code released with sufficient documentation to replicate the infrastructure, not just the algorithm?

---

**Principle 4: Statistical Robustness and Multi-Run Validation of Training Stability in Extended Large-Scale RL**

**Definition:**  
Given the inherent instability of long-horizon RL training—including entropy collapse, gradient variance, and policy degradation—reviewers demand rigorous statistical validation beyond single-run training curves. Claims about training stability, convergence, and sustained performance gains must be supported by repeated experiments, confidence intervals, and validation across diverse initialization conditions. A single training trace is insufficient to substantiate that an intervention (such as a hybrid RL/SFT strategy) robustly mitigates collapse rather than benefiting from a favorable random seed.

**Core Evaluation Criteria:**
- **Multi-Seed or Multi-Run Variance**: Are training curves and final metrics reported with error bars, confidence intervals, or standard deviations across multiple runs?
- **Cross-Model Replication**: Are key stability claims replicated across different base model architectures or sizes to demonstrate generalizability of the training dynamic?
- **Intermediate Dynamics Monitoring**: Beyond final task accuracy, does the paper report time-series metrics such as policy entropy, KL divergence, gradient norms, and reward variance throughout training?
- **Statistical Significance**: Where performance differences are claimed, is appropriate statistical testing performed to ensure differences are not due to noise?
- **Failure Case Analysis**: Does the paper analyze and report failure modes or instability instances that occurred across runs, rather than only reporting the best or average outcome?

---

**Principle 5: Quantitative Mechanistic Validation of Entropy Collapse Mitigation and Exploration Diversity in Hybrid RL/SFT Regimes**

**Definition:**  
For training strategies designed to counteract entropy collapse during extended LLM RL—such as cyclically alternating RL with supervised fine-tuning (SFT)—reviewers require evidence that goes beyond improved final task performance. The evaluation must assess whether the method mechanistically restores exploration capacity, increases behavioral diversity, and sustains learning progress, rather than merely resetting the policy to a high-entropy state without lasting benefit. Claims about "enhanced diversity" or "restored exploration" must be grounded in behavioral metrics, not inferred from aggregate reward curves alone.

**Core Evaluation Criteria:**
- **Direct Behavioral Metrics**: Does the paper quantify exploration using policy entropy, action-space coverage, trajectory variance, or Best-of-N diversity metrics before and after the intervention?
- **Qualitative Response Pattern Analysis**: Are changes in agent behavior (e.g., response length, dialogue turns, reasoning depth) analyzed to show that restored entropy leads to novel solution strategies rather than degenerate repetition?
- **Comparative Mechanistic Analysis**: Does the work compare against existing entropy management techniques (e.g., adaptive clipping, token-level interventions, maximum entropy RL) in terms of computational overhead, convergence speed, and final performance?
- **Causal Intervention Evidence**: Do ablation studies show that removing the entropy-recovery component causes clear performance degradation or premature convergence, establishing a causal link?
- **Longitudinal Learning Progress**: Is there evidence that the intervention enables sustained learning progress over hundreds of additional update steps, rather than a temporary performance fluctuation?