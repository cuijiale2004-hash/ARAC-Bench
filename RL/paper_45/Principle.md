**Principle 1: Attribution Analysis of RL-Driven Emergence vs. Teacher-Forced Imitation in Parallel Reasoning**

**Definition:**  
This principle evaluates whether the claimed parallel thinking capability arises from genuine reinforcement learning exploration or merely from imitating teacher demonstrations provided during supervised fine-tuning. In the context of training non-native reasoning behaviors, reviewers must assess whether the cold-start data is intentionally stripped of complex reasoning to isolate RL’s contribution, or whether the method inadvertently relies on distilled teacher traces that pre-shape the model’s behavior. The scientific validity of the work hinges on establishing that RL induces novel behaviors beyond the teacher’s upper bound, rather than performing SFT-by-another-name. This requires clear articulation of what the cold-start stage teaches—structural format versus reasoning logic—evidence that RL drives generalization to harder tasks, and an honest discussion of dependencies on teacher-generated data.

**Core Evaluation Criteria:**
- **Isolation of RL Contribution**: Does the work clearly demonstrate that performance gains on hard tasks emerge from RL exploration rather than from reasoning knowledge injected via SFT cold-start or teacher distillation?
- **Cold-Start Design Justification**: Is the choice to use format-only SFT (as opposed to reasoning-heavy SFT) methodologically justified as a means to study emergence, or does it reflect a data pipeline limitation mischaracterized as a feature?
- **Distinction from Imitation Baselines**: Does the paper rigorously compare against strong SFT-only or distillation-based baselines under comparable model scales and data conditions to empirically prove RL’s added value?
- **Transparency of Data Dependencies**: Are the sources, costs, and roles of teacher-generated data fully disclosed, and does the work avoid claiming “pure RL” when SFT data may contain implicit reasoning supervision?

---

**Principle 2: Functional Authenticity Verification Beyond Syntactic Format Compliance in Parallel Thinking**

**Definition:**  
This principle assesses whether the learned “parallel thinking” behavior represents genuine multi-path exploration and reasoning, or merely surface-level compliance with syntactic templates such as inserting control tags without semantic diversity. High parallel ratios or format adherence do not constitute evidence of functional parallelism; reviewers must demand metrics that verify path diversity, mutual information between branches, and the adaptive utility of the parallel structure. The work must distinguish between structural mimicry and cognitively meaningful parallel reasoning, particularly when rewards are partially tied to format production. Without such verification, claims about the model’s internalized reasoning capability remain unsubstantiated, and the method risks being evaluated as a template-engineering trick rather than a reasoning advance.

**Core Evaluation Criteria:**
- **Semantic Path Diversity**: Does the work provide quantitative evidence (e.g., low BLEU scores, high divergence, or cosine similarity of path embeddings) that parallel branches contain genuinely distinct reasoning rather than duplicated or trivially varied content?
- **Behavioral Evolution Analysis**: Does the work track how parallel thinking is used across training stages—such as shifting from early exploration to late verification—using mechanistic probes rather than relying solely on aggregate accuracy metrics?
- **Format-Function Decoupling**: Are experiments designed to show that performance gains come from the functional utility of parallel thinking, not just from the additional test-time compute or token budget enabled by the structure?
- **Training-Inference Consistency**: For methods that modify architectures or attention masks during training, does the work verify that the learned behavior remains coherent and consistent under inference-time generation dynamics?

---

**Principle 3: Regime-Aligned Baseline Selection and Fair Comparison in Non-Sequential Reasoning Evaluation**

**Definition:**  
This principle mandates that empirical comparisons must control for model capacity, data scale, starting checkpoint, and inference compute when evaluating novel reasoning paradigms against sequential baselines. Comparing a small base model trained with a new method against a large instruction-tuned model—or one trained with massive proprietary post-training data—is scientifically invalid for assessing the method’s intrinsic utility. Reviewers must scrutinize whether baselines share the same backbone, RL algorithm, and data budget, and whether sequential baselines are enhanced to their maximum reasonable strength under the same experimental regime. The goal is to isolate the effect of the reasoning paradigm itself from confounding factors like scale, data quality, or pre-existing instruction-following capabilities.

**Core Evaluation Criteria:**
- **Model Scale and Initialization Control**: Are comparisons made against baselines using the same base model size and initialization (e.g., base vs. base), rather than against heavily post-trained instruct models or product-level systems?
- **Sequential Baseline Enhancement**: Does the paper compare against the strongest possible sequential RL baselines, including those trained with equivalent data curricula, rather than naive out-of-the-box RL?
- **Inference Compute Parity**: When comparing against test-time parallel methods (e.g., Self-Consistency, Tree of Thoughts), does the analysis account for total token usage, latency, or sampling budget to ensure fair cost-performance trade-offs?
- **Data Budget Transparency**: Are the amounts and sources of training data (SFT plus RL) clearly reported, and are baseline methods granted comparable data resources and training time?

---

**Principle 4: Methodological Necessity and Rigor of Cold-Start Curriculum for Non-Native Reasoning Topologies**

**Definition:**  
This principle evaluates whether the progressive curriculum—typically involving SFT on easy tasks followed by RL on hard tasks—is necessary, well-motivated, and rigorously ablated for training reasoning behaviors absent in pre-training. Since LLMs lack innate parallel thinking priors, a cold-start stage is often essential, but its design must be justified as more than a patch for poor RL stability. Reviewers should assess whether each curriculum stage has a distinct, validated purpose—such as format acquisition versus capability generalization—and whether removing stages causes predictable, reported degradation. The principle also demands investigation into whether architectural modifications, such as explicit path isolation, require fundamentally different curricula than autoregressive variants due to distribution shift or objective misalignment.

**Core Evaluation Criteria:**
- **Stage Necessity via Ablation**: Are ablation studies provided showing the performance degradation when removing the SFT cold-start, the easy-task RL stage, or the curriculum progression?
- **Curriculum-Architecture Interaction**: Does the work investigate whether the curriculum design must change when architectural inductive biases are introduced (e.g., does path isolation require skipping easy-task RL)?
- **Reward Schedule Justification**: Is the reward design (e.g., alternating accuracy and structure rewards) principled and robustly ablated, or does it appear ad-hoc and hyperparameter-sensitive?
- **Generalization from Easy to Hard**: Does the work provide evidence that skills learned on the easy task actually transfer and generalize to the hard task distribution, rather than merely preventing initial training collapse?

---

**Principle 5: Cross-Domain Transfer and Model-Scale Scalability of RL-Induced Parallel Thinking Behaviors**

**Definition:**  
This principle examines whether the induced reasoning behavior is an emergent, generalizable strategy or a dataset-specific artifact optimized for a narrow domain such as mathematics. Because mathematical reasoning offers clean verifiable rewards and highly structured problems, it serves as a natural testbed, but claims about “general reasoning” require substantive evidence beyond math benchmarks. Reviewers must evaluate whether the method is tested on qualitatively different domains—such as commonsense QA, planning, or open-ended writing—and whether larger models exhibit the same behavioral patterns and scaling trends. For a novel reasoning paradigm to be impactful, it must demonstrate consistent scaling properties and domain transferability that suggest it is a fundamental emergent capability rather than a brittle, task-specific optimization trick.

**Core Evaluation Criteria:**
- **Non-Math Domain Validation**: Does the work evaluate zero-shot or fine-tuned transfer to non-mathematical tasks with distinct reasoning structures (e.g., commonsense reasoning, multi-hop QA)?
- **Model Scale Robustness**: Are experiments conducted on multiple model scales to verify that the parallel thinking behavior emerges consistently and that performance trends hold or improve with scale?
- **Behavioral Consistency Across Settings**: Does the key phenomenological finding (e.g., exploration-to-verification shift, mid-training scaffold effect) replicate across different model sizes, domains, and random initializations?
- **Claim Calibration**: Does the paper appropriately scope its claims (e.g., stating “general mathematical reasoning” rather than “general real-world reasoning”) when cross-domain evidence is preliminary or absent?