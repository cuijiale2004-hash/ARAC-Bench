**Principle 1: Causal Mechanism Diagnosis of Training Instability Induced by Tool Feedback Distributional Drift in Multi-Turn RL**

**Definition:**  
Multi-turn tool-integrated reinforcement learning introduces unique instability modes that differ fundamentally from single-turn reasoning or generic RL training, primarily because external tool feedback can inject out-of-distribution tokens into the agent's context window. A high-quality study must do more than merely observe gradient spikes or final performance collapse; it must rigorously trace the causal chain from tool-feedback distributional drift to low-probability token generation and subsequent policy gradient norm explosion. This requires carefully isolating the effect of iterative tool feedback loops from other confounders such as reward sparsity, inherent exploration noise, or model initialization. Authors should provide theoretical or mechanistic evidence demonstrating that the identified root cause directly produces the observed symptom across multiple turns. Such diagnostic depth is essential because superficial fixes might suppress symptoms without resolving the underlying compounding error dynamics inherent to long-horizon agentic interaction. Reviewers should expect clear evidence that the proposed mechanism explains why instability worsens with turn depth and why it is specific to the tool-use setting rather than a generic RL pathology.

**Core Evaluation Criteria:**
- **Isolation of Causal Factors:** Does the work isolate tool-feedback distributional drift from generic RL confounders (entropy collapse, reward sparsity, initialization variance) through controlled comparisons such as single-turn versus multi-turn ablations?
- **Theoretical Grounding:** Is there a formal or mechanistic analysis linking out-of-distribution tool outputs to low-probability tokens and quantifying their contribution to gradient norm explosion?
- **Empirical Corroboration:** Are token probability traces, gradient norms, and failure modes presented together to establish a reproducible causal chain rather than a post-hoc narrative?
- **Dose-Response Relationship:** Does the evidence show that instability worsens predictably with turn depth or distributional shift severity, supporting the compounding-error hypothesis?

---

**Principle 2: Design Rationality and Generality of Trajectory Filtering Heuristics for Compounded Error Mitigation**

**Definition:**  
Because the space of possible trajectories in multi-turn TIR is vast and high-dimensional, trajectory filtering inevitably relies on heuristic indicators to identify pathological samples efficiently. A filtering rule such as "void turns" may effectively proxy for collapsed generation in code-interpreter tasks, yet its robustness to format drift, alternative tool types, and diverse task domains remains an open empirical question that must be addressed. The principle demands that authors justify why their chosen heuristic is the most efficient and effective proxy for the diagnosed failure mode, rather than an ad-hoc convention tied to a specific prompt format or sandbox implementation. Furthermore, the evaluation must rigorously quantify the costs of false positives, which discard useful trajectories, and false negatives, which retain harmful ones, under varying output conventions. Authors should test the heuristic's limits when output formats change or when alternative tools such as search engines or multimodal processors are introduced. Without this generality analysis, a filtering method risks being non-transferable to broader agentic environments beyond the initial experimental setup.

**Core Evaluation Criteria:**
- **Mechanistic Justification:** Is the filtering heuristic explicitly linked to the diagnosed pathology, with ablations showing that alternative heuristics (low-probability tokens, high importance ratios) are inferior or non-transferable?
- **Cross-Domain Validation:** Has the heuristic been tested beyond the primary task domain to other tool types (search, APIs) or modalities, demonstrating transferability rather than task-specific overfitting?
- **Error Rate Quantification:** Are false positive and false negative rates reported, with analysis of edge cases such as partial code, format variations, or valid turns lacking explicit final answers?
- **Robustness to Format Drift:** Does the work evaluate sensitivity to prompt templates, answer formats, and tool-output conventions that might break the heuristic in deployment?

---

**Principle 3: Credit Assignment Integrity and Sample Efficiency in Trajectory-Level Intervention for Long-Horizon RL**

**Definition:**  
In long-horizon RL, discarding entire trajectories to block harmful gradients creates an inherent tension between training stability and correct credit assignment across multiple turns. A principled filtering method must clarify whether excluded trajectories are removed from policy gradients only, retained for advantage estimation, or dropped entirely from the batch, because each choice carries distinct theoretical implications for estimator bias and variance. Full trajectory exclusion can prevent gradient explosion but may unfairly suppress valid multi-turn exploration if early correct turns are implicitly penalized by late failures, whereas partial masking can introduce misaligned credit assignment that pushes the policy toward single-turn shortcuts. The field therefore requires explicit analysis of how filtering shapes the effective reward distribution and whether it induces a systematic bias against complex, multi-step reasoning strategies. Equally important is the empirical measurement of sample efficiency: authors should report filter rates, wall-clock overhead, and the necessity of oversampling to maintain training throughput. Ultimately, the intervention must be shown to stabilize training without collapsing the behavioral diversity needed for sophisticated tool use.

**Core Evaluation Criteria:**
- **Theoretical Clarity of Intervention:** Does the work precisely specify which components of the RL update are affected by filtering (policy loss versus advantage estimation) and justify this choice in terms of bias and variance?
- **Credit Assignment Analysis:** Is there evidence that the chosen filtering granularity (full trajectory versus suffix-only versus down-weighting) avoids misaligned credit assignment that would suppress multi-turn reasoning?
- **Sample Efficiency Metrics:** Are the filter rate, trajectory discard percentage, and impact on effective batch size reported across training, along with any necessary oversampling or wall-clock overhead?
- **Behavioral Bias Check:** Does the work verify that filtering does not systematically bias the policy toward fewer turns or simpler reasoning patterns, for instance by tracking average turn count and response length?

---

**Principle 4: Rigor of Stability Metrics and Long-Horizon Training Dynamics Validation**

**Definition:**  
Claims of stabilized training in multi-turn TIR cannot be validated solely through final benchmark accuracy or short-horizon ablations, because instability may manifest differently across training phases or may temporarily plateau before later catastrophic collapse. Comprehensive validation requires continuously tracking gradient norms, minimum token probabilities, reward curves, and turn-count distributions across the complete training trajectory and ideally across multiple random seeds. Reviewers should look for quantitative correlations between proposed instability indicators and measured gradient spikes, rather than relying on a handful of anecdotal case studies or plots restricted to early training steps. Ablation studies must also demonstrate that baseline methods not only exhibit early instability but fail to recover in the long run, thereby establishing that the proposed intervention addresses a persistent failure mode rather than a transient fluctuation. This level of dynamic scrutiny ensures that stability claims are grounded in the temporal realities of long-horizon RL training rather than in selectively chosen snapshots.

**Core Evaluation Criteria:**
- **Full-Horizon Monitoring:** Are training dynamics reported for the complete training duration, including gradient norms, token probability statistics, and reward curves across multiple random seeds?
- **Quantitative Correlation:** Is there statistical analysis correlating proposed instability indicators with gradient spikes across independent runs, rather than reliance on single-case visualizations?
- **Irreversibility Evidence:** Do ablation studies demonstrate that baseline instability leads to permanent suboptimal convergence or numerical collapse, rather than assuming early spikes imply long-run failure?
- **Comparative Baseline Rigor:** Are existing stabilization techniques evaluated under identical multi-turn settings to confirm that instability is not trivially solvable by standard methods?

---

**Principle 5: Emergent Capability Preservation and Behavioral Diversity Analysis in Zero-RL Tool-Integrated Agents**

**Definition:**  
A central promise of Zero RL for tool-integrated reasoning is the emergence of sophisticated, self-discovered behaviors such as progressive reasoning, cross-validation, and autonomous error correction, which supervised fine-tuning might artificially constrain or fail to elicit. Therefore, stability interventions must be evaluated not only for their technical effect on gradient dynamics but also for their impact on the diversity and richness of the agent's reasoning repertoire over the course of training. High-quality research should move beyond aggregate accuracy metrics to systematically catalog emergent behaviors using automated labeling, structured taxonomies, or rigorous case-study protocols, comparing the resulting behavioral distribution against both unstable baselines and SFT-cold-started models. It is crucial to verify that trajectory filtering or other stabilizing constraints do not inadvertently prune complex multi-turn strategies, thereby trading stability for behavioral impoverishment or mode collapse toward simplistic single-turn solutions. This principle recognizes that in agentic LLM research, the ultimate goal is not merely a stable learning process but one that preserves and enhances the model's capacity for autonomous, adaptive problem-solving through external tools.

**Core Evaluation Criteria:**
- **Systematic Behavior Cataloging:** Does the work use automated labeling or structured annotation to quantify emergent reasoning patterns beyond anecdotal qualitative examples?
- **Comparative Behavioral Analysis:** Are emergent behaviors compared against SFT-based or instruction-tuned baselines to isolate the unique contribution of stable Zero RL training?
- **Diversity Preservation Metrics:** Does the evaluation track whether stability interventions maintain response diversity, average turn depth, and reasoning pattern richness as training progresses?
- **Functional Utility of Emergence:** Is there evidence that emergent behaviors actively contribute to performance gains rather than being spurious byproducts of the training process?