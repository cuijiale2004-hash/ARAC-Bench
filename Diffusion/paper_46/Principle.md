### Principle 1: Theoretical Validity and Constraint Compliance of Non-Monotonic Noise Schedules in Self-Correcting Discrete Diffusion Processes

**Definition:**
In mask-based diffusion language models (DLMs) that introduce self-corrective remasking mechanisms, reviewers must rigorously evaluate whether the proposed method maintains the fundamental theoretical guarantee of diffusion models: the monotonic decrease of noise levels during the reverse generation process. Since remasking inherently reintroduces noise (mask tokens) into previously generated sequences, the work must provide explicit mathematical constraints or scheduling protocols ensuring that the number of newly remasked tokens never exceeds the number of tokens scheduled to be unmasked at any given step. This principle is critical because violating this constraint breaks the convergence guarantees of the diffusion process, potentially leading to non-terminating generation loops or degenerate output distributions, regardless of empirical performance gains.

**Core Evaluation Criteria:**
*   **Mathematical Guarantee of Monotonicity:** Does the paper explicitly define and prove an inequality (e.g., relating incorrect token ratios to mask ratios) that ensures the total count of mask tokens strictly decreases or remains bounded at every diffusion step, even when active remasking occurs?
*   **Noise Schedule Compatibility:** Is the remasking probability schedule designed to be compatible with standard discrete diffusion forward processes, or does it require ad-hoc modifications that lack theoretical grounding?
*   **Termination Assurance:** Is there a mechanism preventing "remasking oscillation" where tokens are repeatedly masked and unmasked without convergence? Reviewers should look for quantitative evidence (e.g., remask frequency distributions) showing that oscillatory behavior is rare and controlled.
*   **Handling of Edge Cases:** How does the method handle scenarios where the model identifies more errors than can be safely remasked under the monotonicity constraint? Is there a prioritization strategy that preserves the diffusion trajectory's validity?

---

### Principle 2: Disentanglement of Performance Gains from Computational Artifacts via Matched-Compute Ablations in Two-Stage DLM Training

**Definition:**
For DLM research employing multi-stage training pipelines (e.g., Supervised Fine-Tuning followed by Reinforcement Learning), reviewers must demand rigorous "matched-compute" ablations to distinguish genuine algorithmic innovation from mere scaling effects. Since adding an RL stage or a secondary confidence-prediction stream inevitably increases total compute budget, observed improvements could simply stem from additional training time or parameter count rather than the proposed mechanism itself. This principle mandates that authors compare their full method against baselines trained with identical computational resources (GPU-hours/FLOPs) to validate that the specific design choices (e.g., remasking policy, dual-stream architecture) are the true causal factors for performance enhancement.

**Core Evaluation Criteria:**
*   **Matched-Compute Baselines:** Does the paper include comparisons where baseline models (e.g., vanilla SFT, standard RL without remasking) are trained for equivalent GPU-hours as the proposed method’s additional stages?
*   **Parameter-Efficiency Analysis:** If a new architectural component (e.g., Unmasking Policy Stream) is added, is its contribution isolated from the mere increase in parameter count? Are there ablations comparing against simpler alternatives (e.g., single head vs. dual stream) with matched capacity?
*   **Convergence Rate Normalization:** When claiming faster convergence or superior sample efficiency, are learning curves plotted against wall-clock time or total FLOPs rather than just optimization steps?
*   **Isolation of Multi-Task Benefits:** For methods combining multiple objectives (e.g., token prediction + confidence scoring), is the benefit of the joint objective disentangled from the benefit of extended training duration?

---

### Principle 3: Behavioral Interpretability and Causal Attribution of Learned Confidence Signals in Iterative Token Refinement

**Definition:**
When a model introduces auxiliary prediction heads (such as per-token confidence scores) to guide generation dynamics, reviewers must assess whether these signals are genuinely informative and causally linked to correction behaviors, rather than being spurious correlations or redundant outputs. It is insufficient to show only final task accuracy; the work must demonstrate that the learned confidence scores accurately reflect token quality and that low-confidence predictions are systematically associated with subsequent successful corrections. This principle ensures that the "self-reflective" capability is a learned, interpretable skill rather than a stochastic artifact, which is essential for trusting and debugging complex generative systems.

**Core Evaluation Criteria:**
*   **Calibration and Correlation:** Is there quantitative evidence showing that predicted confidence scores correlate with actual token correctness? Do tokens eventually remasked exhibit statistically lower initial confidence scores?
*   **Causal Link to Correction:** Does the analysis demonstrate that remasking low-confidence tokens leads to higher-quality replacements compared to random or heuristic remasking strategies?
*   **Behavioral Dynamics Visualization:** Are there visualizations of the generation trajectory showing *when* and *why* remasking occurs (e.g., correcting calculation errors, refining terminology)? Do these examples align with the claimed capabilities?
*   **Ablation of Signal Utility:** What happens if the confidence signal is ablated, randomized, or replaced with a heuristic? Does performance degrade significantly, confirming the signal's necessity?

---

### Principle 4: Rigorous Head-to-Head Benchmarking Against Inference-Time and Training-Time Editing Baselines Under Identical Settings

**Definition:**
In the rapidly evolving subfield of editable DLMs, claims of state-of-the-art performance must be validated against both inference-time correction methods (e.g., predictor-corrector samplers) and training-time editing approaches (e.g., seed diffusion, edit flows) under strictly controlled experimental conditions. Reviewers should reject comparisons where baselines are evaluated using different base models, datasets, or decoding budgets. This principle addresses the common pitfall where proposed methods appear superior simply due to unfair advantages in implementation or evaluation setup, ensuring that reported gains reflect true methodological advances in self-correction capability.

**Core Evaluation Criteria:**
*   **Identical Checkpoint and Data:** Are all compared methods (including reproduced baselines) trained on the exact same base model weights and datasets?
*   **Decoding Budget Parity:** Are inference-time baselines given equivalent computational budgets (e.g., number of sampling steps, latency constraints) to ensure fair comparison?
*   **Reimplementation Transparency:** For baselines without public code, do authors provide reimplementation details and validation that their reproduction matches original reported performance?
*   **Comprehensive Metric Coverage:** Does the evaluation span diverse domains (math, code, open-ended generation) to test the generality of the correction mechanism, rather than cherry-picking benchmarks where the method excels?

---

### Principle 5: Practical Deployment Viability Assessment Through Quality-Latency Pareto Frontiers and Engineering Overhead Analysis

**Definition:**
For DLM research proposing architectural modifications like dual-stream transformers or iterative refinement loops, reviewers must evaluate practical deployment viability beyond peak benchmark scores. The introduction of self-correction mechanisms often entails trade-offs in inference speed, memory footprint, and engineering complexity. High-quality research should characterize these trade-offs explicitly through quality-latency Pareto frontiers and overhead analyses, demonstrating that the method offers a favorable operating point for real-world applications. This principle prevents the acceptance of theoretically interesting but practically unusable methods that achieve marginal gains at prohibitive computational costs.

**Core Evaluation Criteria:**
*   **Quality-Latency Trade-off Curves:** Does the paper present Pareto frontiers showing accuracy versus throughput (tokens/sec) across different decoding configurations? Does the method dominate baselines on this frontier?
*   **Overhead Quantification:** Are the additional parameters, memory usage, and FLOPs of new components (e.g., UPS) explicitly reported relative to the base model?
*   **Engineering Complexity Assessment:** Is the integration cost discussed (e.g., lines of code, training pipeline complexity)? Is the method compatible with existing acceleration techniques (e.g., block-causal attention, KV caching)?
*   **Scalability Evidence:** Is there evidence or discussion regarding how the overhead scales with model size or sequence length? Does the method remain efficient for variable-length generation?