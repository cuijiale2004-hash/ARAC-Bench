**Principle 1: Rigorous Empirical Validation of Spatial Masking Hypotheses Underlying Parallel Decoding Strategies**

**Definition:**  
For training-free parallel decoding methods in diffusion LLMs, the foundational hypothesis—typically that spatially sparse masking patterns reduce distributional drift and stabilize multi-token prediction—must be validated through carefully designed preliminary studies. Reviewers expect that the empirical motivation is not merely illustrative but constitutes rigorous evidence that the proposed structural insight genuinely improves prediction consistency. This includes close scrutiny of the ground-truth construction, the choice of divergence metrics, and whether the study design avoids circular reasoning or metric-specific artifacts. A weak or methodologically flawed foundational study undermines the entire contribution, regardless of downstream algorithmic performance, because it calls into question whether the algorithm is solving a real problem or merely capitalizing on a measurement illusion.

**Core Evaluation Criteria:**
- **Ground-Truth Rigor:** Is the ground truth for the preliminary study constructed independently of the model's own sequential outputs, avoiding circular validation where the model predicts itself?
- **Metric Appropriateness:** Are alternative metrics or direct accuracy comparisons used to confirm that the observed phenomenon is not an artifact of a specific surrogate loss (e.g., cross-entropy KL approximation)?
- **Generalizability of Phenomenon:** Does the pre-study demonstrate that the spatial masking advantage persists across diverse task types (code, math, open-ended generation) and scales meaningfully with the number of masked tokens?
- **Causal Clarity:** Does the analysis clearly establish a causal link between spatial sparsity and reduced distributional drift, rather than merely correlating the two?

---

**Principle 2: Comprehensive Speed-Accuracy Trade-off Analysis with Algorithmic Overhead Transparency**

**Definition:**  
A central evaluation criterion for inference acceleration methods is whether they achieve favorable speed-accuracy trade-offs under realistic deployment conditions. Reviewers distinguish sharply between theoretical parallel capacity (tokens per forward pass, TPF) and practical wall-clock throughput (tokens per second, TPS), demanding full transparency regarding non-negligible algorithmic overheads such as dynamic sub-decoding area management, index manipulation, confidence-score sorting, and CPU-side control logic. The method must demonstrate that wall-clock speedups are not offset by hidden computational costs or unacceptable accuracy degradation, particularly on complex reasoning and code generation tasks where token dependencies are strong and errors propagate easily.

**Core Evaluation Criteria:**
- **Wall-Clock Efficiency:** Are both TPF and TPS reported, and does the method demonstrate practical throughput gains rather than only theoretical parallel factors?
- **Overhead Accountability:** Is the non-model overhead (e.g., hierarchical selection logic, remasking operations, sub-area partitioning) explicitly quantified and shown to be negligible relative to the dominant model forward-pass cost?
- **Accuracy Preservation:** Does the method maintain comparable or improved accuracy relative to strong baselines, and are performance drops honestly characterized as acceptable trade-offs rather than ignored or obscured?
- **Scalability with Sequence Length:** Does the speedup ratio remain stable or improve as generation length increases, confirming efficiency gains hold under extended decoding scenarios?

---

**Principle 3: Hyperparameter Robustness and Cross-Task Generalization of Training-Free Decoding Configurations**

**Definition:**  
Training-free decoding strategies frequently introduce multiple threshold-based heuristics to govern parallel token commitment, raising immediate concerns about sensitivity to hyperparameters and the practical burden of per-task grid search. Reviewers evaluate whether the method can deploy with a single robust default configuration across diverse benchmarks without expensive tuning, and whether sensitivity analyses cover a sufficiently broad range to claim stability. A method requiring fragile, task-specific calibration is considered to have limited practical value, even if it achieves strong peak performance under tuned conditions. The community prioritizes plug-and-play inference strategies that preserve the training-free promise of zero adaptation cost.

**Core Evaluation Criteria:**
- **Default Configuration Viability:** Can a single fixed set of hyperparameters achieve competitive performance across all evaluated benchmarks, or does the method rely on per-task tuning to mask underlying instability?
- **Sensitivity Analysis Breadth:** Are ablation studies conducted over a wide enough range of key thresholds to demonstrate stability, and do they reveal a flat operating region where performance is insensitive to exact values?
- **Practical Deployability:** Does the paper provide principled heuristics, automatic mechanisms, or at least robust defaults for setting thresholds, rather than leaving them as opaque tuning knobs?
- **Tuning Cost Transparency:** Is the performance gap between default and task-tuned configurations explicitly quantified to inform practitioners about the accuracy or speed cost of avoiding grid search?

---

**Principle 4: Breadth of Baseline Coverage and Diversity of Evaluation Benchmarks for Decoding Acceleration**

**Definition:**  
Claims of superiority in decoding acceleration must be substantiated against a comprehensive and contemporary set of baselines, including recent training-free parallel samplers, revocable decoding frameworks, and multi-stage decoding methods. Reviewers expect comparisons beyond a single primary baseline to methods that represent the current state of the art. Furthermore, evaluation should span multiple domains—including mathematical reasoning, code generation, and open-ended instruction following—to verify that the method's advantages are not confined to narrow task types where diffusion models happen to excel. A paper that omits strong contemporaneous baselines or benchmarks is viewed as making unsubstantiated claims of generality.

**Core Evaluation Criteria:**
- **Baseline Comprehensiveness:** Does the evaluation include at least one other recent, sophisticated training-free baseline in addition to the primary comparison method?
- **Fair Comparison Conditions:** Are all baselines evaluated under identical experimental conditions (same model checkpoints, evaluation framework, generation length, and hardware) to ensure equitable assessment?
- **Cross-Domain Benchmarking:** Are results reported on benchmarks outside the method's apparent "comfort zone" (e.g., non-code/math tasks like instruction following) to validate generalization?
- **State-of-the-Art Awareness:** Does the related work and experimental section acknowledge and compare against contemporaneous parallel decoding, caching, and variable-length generation techniques?

---

**Principle 5: Structural Novelty and Theoretical Grounding of Hierarchical Divide-and-Conquer Decoding**

**Definition:**  
Proposed decoding frameworks must clearly articulate their structural and algorithmic distinction from prior confidence-based parallel decoding paradigms. Reviewers scrutinize whether hierarchical or divide-and-conquer designs represent genuine architectural innovations—such as adaptive sub-decoding areas, recursive partitioning, and interlocking correction mechanisms—or are merely incremental modifications of existing thresholding rules. Additionally, theoretical claims regarding asymptotic acceleration (e.g., logarithmic decoding depth) require explicit formal assumptions and careful caveats distinguishing ideal-case upper bounds from practical runtime guarantees. Without this clarity, the contribution risks being dismissed as a heuristic wrapper around well-known confidence truncation.

**Core Evaluation Criteria:**
- **Architectural Distinction:** Does the method introduce a qualitatively new decoding structure (e.g., dynamic hierarchical partitioning) that is fundamentally unavailable to flat confidence-thresholding schemes?
- **Theoretical Rigor:** Are complexity claims accompanied by formal assumptions, and is the distinction between idealized upper bounds and empirical runtime behavior clearly communicated to avoid overstatement?
- **Mechanism Interlocking:** Does the algorithm present a cohesive, interlocking system of mechanisms rather than a loose collection of independent heuristics that could be trivially decoupled?
- **Novelty Justification:** Does the paper effectively counter claims of limited novelty by demonstrating synergistic gains that exceed what simple threshold adjustments or masking rule changes could achieve?