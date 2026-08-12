**Principle 1: Fairness and Controllability of Parameter and Compute Budget in Modular Memory Augmentation**

**Definition:**  
When a method introduces lightweight auxiliary modules (e.g., LoRA adapters) to augment a frozen backbone LLM with memory or reasoning capabilities, it is imperative to disentangle architectural innovation from raw capacity expansion. Reviewers in this subfield treat uncontrolled parameter budget and unreported inference overhead as critical threats to validity, because any additional trainable weights or forward passes can artificially inflate performance. Consequently, high-quality research must demonstrate that gains persist under strict parameter parity and that inference latency remains bounded and characterized, ensuring the community can attribute success to the proposed mechanism rather than to hidden compute or capacity subsidies.

**Core Evaluation Criteria:**
- **Parameter-Matched Baseline Inclusion:** Does the work compare against baselines that introduce a comparable number of extra trainable parameters (e.g., LoRA-based SFT or RL with matched rank), rather than comparing adapter-enhanced systems against vanilla full-parameter tuning without budget adjustment?
- **Inference Cost Quantification:** Are the inference-time overheads (FLOPs, wall-clock latency, number of forward passes) formally analyzed and empirically reported relative to the vanilla backbone, including both amortized and worst-case scenarios?
- **Sparse Activation Validation:** If the method relies on conditional invocation (e.g., an RL trigger), is the empirical activation frequency reported and theoretically bounded, proving the system does not degenerate into dense auxiliary computation?
- **Necessity vs. Capacity Argument:** Do ablations or controlled experiments show that the specific functional form of the module (e.g., generative reconstruction) outperforms a same-budget baseline that simply adds equivalent capacity (e.g., pause tokens, larger rank, or ensembling)?

---

**Principle 2: Mechanistic Transparency of Latent Token Injection and State Manipulation**

**Definition:**  
Research that manipulates the internal hidden-state trajectory of a frozen LLM by injecting latent tokens must provide exhaustive mechanistic disclosure. Because latent interventions operate below the surface of human-readable text, ambiguity regarding insertion depth (embedding layer versus intermediate hidden states), interaction with the KV cache, attention-mask modification, and cross-component state bridging severely undermines reproducibility. This principle demands that authors formally specify the computational graph of intervention, justify design choices (e.g., hidden-state versus token-space inputs), and transparently describe staged training protocols when co-dependent components (such as a trigger and a weaver) cannot be trained jointly from scratch.

**Core Evaluation Criteria:**
- **Layer and Position Specificity:** Are the exact insertion points (e.g., input-embedding layer, specific transformer depths) and concatenation rules formally defined with precise notation, not described only via high-level diagrams?
- **KV Cache and Attention Compatibility:** Does the work explain how latent tokens integrate with existing key-value caches and causal attention masks without corrupting autoregressive properties or requiring non-standard cache handling?
- **Decoupled Training Protocol Clarity:** When components are mutually dependent (e.g., a trigger requires a trained weaver and vice versa), is the bootstrapping or alternating optimization procedure fully detailed in the main text, including stability justifications?
- **State-Space Bridging Justification:** Are mappings between the output spaces of auxiliary modules and the backbone’s embedding or hidden-state space explicitly defined, and is the choice to operate in latent space rather than token space rigorously motivated?

---

**Principle 3: Rigorous Differentiation from Latent Steering and Continuous Reasoning Paradigms**

**Definition:**  
As the boundary between latent memory, latent Chain-of-Thought, and latent policy steering continues to blur, submissions bear the burden of establishing sharp conceptual and empirical differentiation. A method that inserts latent vectors during generation risks being perceived as an incremental variant of continuous reasoning unless it demonstrates memory-specific properties: cross-task persistence, continual learning without catastrophic forgetting, and context-contingent recall. This principle requires authors to situate their work within the latent-computation taxonomy through direct comparisons and to validate that the proposed system functions as a memory architecture (retaining, reconstructing, and transferring experience) rather than merely as an internal reasoning accelerator.

**Core Evaluation Criteria:**
- **Taxonomic Positioning:** Does the work provide explicit conceptual or tabular comparisons against latent CoT, speculative decoding, and latent steering methods on dimensions such as triggering granularity, backbone modification, multi-turn retention, and cross-task generalization?
- **Memory-Specific Capabilities:** Are cross-domain generalization, continual learning, and multi-episode retention empirically tested, rather than relying solely on single-task, single-episode accuracy improvements?
- **Functional Specialization Evidence:** Is there empirical support (e.g., intervention studies) that latent tokens serve distinct memory roles (e.g., planning, procedural, working memory), rather than functioning as undifferentiated continuous thoughts?
- **Novelty Beyond Compression:** Does the work quantitatively demonstrate that generated latent tokens are geometrically or semantically distinct from simple compressions of prior trajectories or re-encodings of current context?

---

**Principle 4: Causal Validation of Emergent Cognitive Functionality in Learned Latent Structures**

**Definition:**  
Claims that learned latent tokens spontaneously evolve human-like cognitive faculties—such as planning memory, procedural memory, or working memory—are held to a high evidentiary standard. The community treats post-hoc cognitive analogies with skepticism unless they are supported by causal intervention studies, targeted ablations, and falsifiable hypotheses about token function. This principle insists that interpretability analyses move beyond descriptive clustering or t-SNE visualization; authors must demonstrate that ablating specific latent structures selectively degrades corresponding agent capabilities, thereby establishing functional roles rather than spurious correlates. Furthermore, anthropomorphic framing must be carefully calibrated to the strength of the evidence.

**Core Evaluation Criteria:**
- **Intervention-Based Causality:** Are there experiments where specific latent clusters or tokens are removed or masked to measure targeted degradation in specific failure modes (e.g., planning errors versus formatting mistakes)?
- **Distinction from Surface Patterns:** Does the analysis rule out the possibility that clusters merely encode shallow syntactic templates, domain identifiers, or position biases?
- **Quantitative Separation Metrics:** Are latent memory representations shown to be structurally distant from both the current reasoning context and past trajectory embeddings, supporting the claim of generative reconstruction rather than retrieval?
- **Evidentiary Calibration of Analogies:** Does the framing avoid overstated biological or cognitive metaphors, restricting claims to the strictly demonstrated computational functions?

---

**Principle 5: Scalability Analysis and Pathological Behavior Characterization in RL-Triggered Memory Systems**

**Definition:**  
Dynamic memory systems that employ reinforcement learning to control when memory is invoked must be evaluated as deployable systems, not merely as proof-of-concept models on short-horizon benchmarks. This principle demands theoretical or empirical characterization of how inference latency scales with task horizon, the existence and mitigation of pathological trigger behaviors (such as over-firing under distribution shift or under-firing during critical reasoning junctures), and the finite capacity limits of parametric memory stores. Without such analysis, it remains unclear whether the architecture can realistically extend to long-horizon, multi-hop agentic tasks or whether it will succumb to silent failures, runaway compute costs, or gradual knowledge saturation.

**Core Evaluation Criteria:**
- **Horizon-Aware Complexity Analysis:** Is inference overhead analyzed asymptotically or empirically as a function of output sequence length and multi-turn interaction depth, rather than reported only on fixed-length benchmarks?
- **Trigger Pathology Documentation:** Are edge cases examined where the RL-trained trigger may over-activate (wasting compute and disrupting reasoning) or under-activate (omitting necessary memory), particularly under domain shift?
- **Capacity Saturation and Forgetting Curves:** Does the work empirically probe memory capacity limits through extended continual-learning experiments, and does it propose mechanisms (e.g., dynamic expansion, MoE recruiting) for graceful scaling beyond fixed-size adapters?
- **Budget-Constrained Robustness:** Is performance evaluated under explicit inference budgets or activation-rate constraints to verify that gains persist when memory invocation is deliberately limited?