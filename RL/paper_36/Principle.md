Principle 1: Mechanistic Diagnosis of Shift-Readapt-Overfit Progression in Off-Policy Expert Data Integration

**Definition:**
In unified or sequential SFT-RL post-training, integrating off-policy expert demonstrations into a policy model with established response patterns can induce complex training dynamics. A critical evaluation perspective is whether the work provides a deep, mechanistic diagnosis of how off-policy data disrupts the current policy, beyond merely reporting final performance metrics. This includes identifying distinct phases such as initial policy shift (capability disruption due to distributional mismatch), readaptation (recovery through pattern assimilation), and eventual overfitting (loss of diversity and generalization on static expert data). Such diagnosis is essential because it reveals whether the proposed method addresses root causes—such as exposure bias, pattern overwriting, and premature entropy collapse—or merely masks symptoms. Reviewers must assess whether the empirical evidence (e.g., accuracy curves, entropy trajectories, gradient norms) robustly supports the claimed progression. Without this level of analysis, purported improvements in hybrid training may rely on fortuitous hyperparameter choices rather than principled intervention. Ultimately, the field requires cumulative understanding of failure modes in off-policy/on-policy hybridization to advance beyond trial-and-error method design.

**Core Evaluation Criteria:**
- **Phase Identification and Evidence**: Does the work clearly demarcate and empirically validate distinct training phases (shift, readapt, overfit) through multiple metrics, or does it only present aggregate final scores?
- **Causal Attribution vs. Correlation**: Does the analysis establish causal links between off-policy data properties (e.g., distributional divergence from the policy) and observed disruptions, supported by controlled experiments varying expert-policy similarity?
- **Generalizability of Observed Dynamics**: Are the diagnosed dynamics shown across multiple task types (e.g., reasoning vs. tool-use) and model scales, ensuring the phenomenon is not an artifact of a single experimental configuration?

---

Principle 2: Dual-Control Granularity for Balancing Global Loss Scheduling and Token-Level Uncertainty Weighting

**Definition:**
When unifying SFT and RL into a single objective, the granularity at which off-policy expert influence is modulated determines both training stability and the quality of knowledge transfer. A high-quality contribution in this subfield must move beyond coarse-grained, sample-level or stage-level mixing (e.g., binary SFT-then-RL switches or simple dataset interpolation) toward finer-grained control mechanisms that respect the heterogeneity of expert trajectories. The central evaluation question is whether the proposed weighting scheme—whether at the token, phrase, or sample level—is principled, interpretable, and computationally practical. Specifically, token-level dynamic weighting functions should be grounded in policy-state-dependent quantities (e.g., per-token probability, entropy, or importance sampling ratios) that reflect the model's current uncertainty about the expert demonstration. Reviewers should scrutinize whether the design avoids "black-box" heuristics and whether the authors thoroughly explore alternative instantiations to justify their specific choice. The ability to selectively down-weight highly probable tokens (preventing entropy collapse) and highly improbable tokens (preventing policy disruption) represents a critical benchmark for fine-grained control. This principle ensures that hybrid objectives do not indiscriminately force imitation but instead create a stable "learning sweet spot" that evolves with the policy.

**Core Evaluation Criteria:**
- **Granularity Justification**: Does the method justify why its chosen granularity (token, step, or sample) is necessary, and does it demonstrate failure modes of coarser alternatives (e.g., fixed global coefficients alone)?
- **Design Space Exploration**: Are alternative weighting functions or scheduling strategies systematically compared, including ablations that isolate the contribution of the fine-grained component from the global schedule?
- **Theoretical/Interpretable Grounding**: Is the weighting function derived from or justified by an interpretable quantity (e.g., policy uncertainty, variance, or information gain), and does the paper explain why it avoids entropy collapse and policy disruption simultaneously?

---

Principle 3: Robustness to Policy-Expert Distributional Shift Across Expert Strengths and Model Architectures

**Definition:**
The efficacy of unified SFT-RL methods is tightly coupled to the distributional gap between the policy model's established behavior and the off-policy expert's response patterns. A robust method must not be validated solely under ideal conditions—such as a powerful expert with minimal stylistic divergence or a single model family—but must be stress-tested across varying degrees of pattern mismatch. This principle demands evaluation across experts of differing capabilities (e.g., frontier-level vs. moderately strong models) and, crucially, differing stylistic alignment with the policy model. Furthermore, the method should demonstrate architecture-agnostic benefits, spanning dense transformers and alternative structures like Mixture-of-Experts (MoE), as well as varying policy capacities from small to large scales. The underlying concern is that many hybrid methods fail when expert data introduces severe distributional shift or when the policy model lacks the capacity to absorb expert patterns natively. Therefore, reviewers must evaluate whether the experimental design intentionally varies the policy-expert compatibility and model topology to verify generalizable stability.

**Core Evaluation Criteria:**
- **Expert Source Diversity**: Does the evaluation include experts that induce both large and small pattern shifts relative to the policy model, and are the resulting performance differences analyzed rather than averaged away?
- **Architecture and Scale Coverage**: Are experiments conducted across multiple model families (e.g., Qwen, LLaMA, Phi) and scales, including edge cases like weaker base models or MoE architectures where imitation-exploration trade-offs are acute?
- **Stability Under Mismatch**: Does the method maintain training stability (e.g., monotonic reward improvement, controlled entropy) when expert data is stylistically distant, or does it collapse or overfit in high-shift regimes?

---

Principle 4: Task-Specific Selective Pattern Absorption and Response Style Preservation in Unified Post-Training

**Definition:**
A unified SFT-RL objective should not treat expert data as a uniform target for indiscriminate imitation; rather, it must enable the policy to selectively absorb expert reasoning patterns that are beneficial while preserving its own effective, task-adaptive response styles. Different downstream tasks impose divergent structural requirements on model outputs—for instance, mathematical reasoning may benefit from verbose, step-by-step chains of thought, whereas tool-use tasks demand concise, efficient action sequences. Reviewers must assess whether the proposed method demonstrates task-specific selectivity, verified through both quantitative response-pattern metrics (e.g., generation length, lexical diversity, entropy) and qualitative case studies. The key risk in hybrid training is "pattern overwriting," where the policy model loses its intrinsic conciseness or structure by wholesale adoption of the expert's verbose meta-commentary. High-quality research in this subfield must therefore prove that its mechanism promotes a hybrid reasoning style—one that strategically integrates expert verification steps or reasoning structures without destroying the policy's native efficiency. This evaluation perspective prevents methods from being judged solely by accuracy on narrow benchmarks and ensures broader behavioral alignment.

**Core Evaluation Criteria:**
- **Response Pattern Analysis**: Does the work report task-specific changes in response length, formatting, and verbosity, comparing against pure RL, pure SFT, and sequential baselines to detect indiscriminate mimicry?
- **Qualitative Evidence of Hybrid Styles**: Are case studies or error analyses provided to demonstrate that the model produces novel hybrid behaviors (e.g., structured yet concise reasoning) rather than reverting to pure expert or pure policy modes?
- **Selective Absorption Mechanism**: Does the method explicitly link its technical design (e.g., token-wise down-weighting of extreme probabilities) to the observed preservation of intrinsic styles, establishing a mechanistic connection between objective and behavior?

---

Principle 5: Generalization Beyond Verifiable Rewards and Sensitivity Analysis of Dynamic Loss Coefficient Scheduling

**Definition:**
While mathematical reasoning and code generation offer convenient verifiable rewards, unified SFT-RL methods must ultimately prove effective in broader post-training scenarios where deterministic correctness checks are unavailable, such as open-ended medical question answering or creative writing. This principle demands that reviewers evaluate whether the proposed hybrid framework generalizes to non-verifiable domains through alternative reward signals (e.g., model-based judging, rubric-based rewards) without relying on sparse binary correctness. Concurrently, because the balance between off-policy SFT and on-policy RL is governed by dynamic loss coefficients or schedules, rigorous sensitivity analysis is paramount. Reviewers should expect thorough ablations comparing fixed weights, decay schedules, adaptive reward-aware schedules, and warm-up strategies, alongside an analysis of how these schedules interact with fine-grained token-level weighting. A method whose performance is brittle to the exact annealing trajectory of the global coefficient is less practically useful than one that maintains robustness across scheduling variants. This dual requirement—breadth of domain and depth of scheduling analysis—ensures that unified methods are both generalizable and deployable.

**Core Evaluation Criteria:**
- **Non-Verifiable Domain Validation**: Are experiments conducted on tasks without automatic verifiable rewards, and does the method demonstrate stable convergence and final performance gains in these settings?
- **Schedule Ablation and Interaction Effects**: Does the paper systematically compare multiple scheduling strategies for the global loss coefficient (fixed, decay, adaptive, warmup), and does it analyze how these interact with token-level weighting mechanisms?
- **Hyperparameter Sensitivity and Practicality**: Is the method's performance shown to be stable across a reasonable range of scheduling hyperparameters, or does it require narrow, task-specific tuning that would hinder real-world adoption?