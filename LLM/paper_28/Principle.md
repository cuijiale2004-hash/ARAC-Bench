**Principle 1: Fairness and Comprehensiveness of Baseline Configurations in Multi-Hop Parametric Knowledge Integration**

**Definition:**  
In parametric RAG systems that extend single-hop retrieval into iterative multi-hop reasoning, reviewers must scrutinize whether the proposed method is evaluated against a sufficiently comprehensive and fair set of baselines. This includes ensuring that auxiliary components shared across pipelines—such as sub-question decomposition modules, retrievers, and backbone language models—are controlled or explicitly matched, so that performance differences are attributable to the core parametric integration mechanism rather than external architectural or procedural factors. The principle demands that comparisons cover standard in-context RAG variants, existing parametric adaptation methods such as PRAG and DyPRAG, and contemporary reasoning-enhanced retrieval systems, while respecting reproducibility constraints and clearly distinguishing between pure parametric, combined, and fine-tuned variants. Furthermore, authors should explicitly justify any omitted baseline entries due to incompatible backbones or unreleased training configurations, ensuring that experimental conclusions are not artificially inflated by the absence of strong competitors. Without such rigorous baseline controls, claims regarding the superiority of parametric knowledge integration over in-context retrieval remain unconvincing.

**Core Evaluation Criteria:**
- **Controlled Pipeline Components:** Are shared modules (e.g., sub-question generators, retrievers, decomposition strategies) held constant across baselines to isolate the contribution of the parametric mechanism itself?
- **Comprehensive Baseline Coverage:** Does the evaluation include standard RAG, state-of-the-art multi-hop RAG, existing parametric methods, and relevant reasoning-enhanced systems under comparable experimental settings?
- **Reproducibility and Fairness:** Are missing baseline entries clearly justified (e.g., incompatible backbones or unreleased checkpoints), and do reported comparisons avoid cherry-picking weak or incomparable configurations?
- **Variant Clarity:** Are different configurations of the proposed method (e.g., parametric-only, combined parametric-plus-in-context, fine-tuned variants) clearly separated and labeled to prevent conflation of distinct operational modes?

---

**Principle 2: Mechanistic Justification and Comparative Ablation of Orthogonal Interference Mitigation in Continual Expert Merging**

**Definition:**  
When a method proposes to sequentially merge multiple parameter experts—such as passage-specific memory vectors generated across multi-hop retrieval steps—reviewers must evaluate whether the work provides concrete mechanistic evidence that knowledge interference or representational conflict actually occurs during non-orthogonal accumulation. The assessment should go beyond end-to-end accuracy gains to examine whether non-orthogonal alternatives, including arithmetic mean, additive, concatenation, or task-vector merging strategies, suffer from performance degradation due to representational collision, and whether the proposed orthogonal or projected merging strategy specifically preserves complementary information across hops. This principle is crucial because parametric multi-hop systems fundamentally rely on the stable accumulation of external knowledge without overwriting previously injected facts; without empirical evidence that merging conflicts degrade performance, a complex orthogonalization procedure risks being an unnecessary embellishment. Reviewers should verify that ablations isolate the merging operation from the hypernetwork parameterization, ensuring that observed gains are causally linked to the merging mechanism itself rather than to the quality of individual passage experts. Finally, the work should demonstrate stability as the number of retrieved passages or reasoning hops increases, confirming that the merging mechanism scales effectively in long-horizon reasoning chains.

**Core Evaluation Criteria:**
- **Empirical Evidence of Conflict:** Does the work quantify or illustrate the severity and frequency of knowledge conflicts when multiple passage experts are combined without orthogonalization?
- **Ablation Against Non-Orthogonal Merging:** Are there systematic comparisons with alternative merging strategies (additive, arithmetic mean, TIES, concatenation) that isolate the merging mechanism itself?
- **Isolation from Parameterization Effects:** Is the merging strategy evaluated independently of the hypernetwork architecture, ensuring that observed gains stem from the merging operation rather than the parameterization quality?
- **Stability Across Hops:** Does the analysis demonstrate that the merging mechanism maintains performance stability as the number of retrieved passages or reasoning hops increases?

---

**Principle 3: Cross-Architectural Empirical Validation and Theoretical Grounding of Locate-Then-Inject Critical Layer Selection**

**Definition:**  
For methods that restrict parametric knowledge injection to a single "critical layer" rather than distributing updates across all transformer layers, reviewers must assess whether the layer selection is empirically validated across multiple model architectures and supported by a coherent theoretical framework such as the locate-then-edit paradigm or functional layer specialization hypotheses. Because different transformer families often exhibit divergent sensitivity patterns—where early, middle, or late layers vary in their capacity to integrate external facts—a universal layer assignment is rarely tenable and must be treated as model-dependent. The principle requires that authors provide systematic layer-scanning experiments for each evaluated backbone, justify the chosen sensitivity metric such as perplexity reduction with a clear link to knowledge integration efficacy, and explain why the scanning procedure is computationally practical relative to full fine-tuning. Authors should also discuss whether the identified critical layers are expected to transfer across model scales or downstream tasks, and must acknowledge limitations when generalizing beyond the evaluated question-answering settings. Without cross-architectural validation and theoretical grounding, critical-layer parameterization can appear arbitrary or overfit to a specific model family's representational idiosyncrasies.

**Core Evaluation Criteria:**
- **Multi-Architecture Scanning Evidence:** Are layer-sensitivity analyses provided for more than one backbone (e.g., LLaMA, Qwen) to demonstrate architectural dependence of optimal injection points?
- **Metric Justification:** Is the choice of sensitivity indicator (e.g., perplexity, causal mediation effects) theoretically motivated and explained in terms of external knowledge integration?
- **Locate-Then-Inject Consistency:** Does the method clearly position itself within established locate-then-edit or locate-then-inject frameworks, and is the scanning procedure computationally lightweight relative to full fine-tuning?
- **Generalization Discussion:** Does the work discuss whether critical layers are expected to transfer across model scales or tasks, and does it acknowledge limitations in settings beyond the evaluated benchmarks?

---

**Principle 4: Disentangled Efficiency-Performance Trade-off Analysis in Hypernetwork-Based Iterative Parametric Augmentation**

**Definition:**  
In systems that augment inference with hypernetwork-based parameter generation, iterative sub-question decomposition, and sequential expert merging, reviewers must demand a transparent decomposition of computational costs and end-to-end latency relative to standard retrieval-augmented pipelines. The principle recognizes that multi-hop parametric methods introduce additional forward passes for passage-vector generation, sub-question generation, and merging operations, which could potentially offset their accuracy benefits if not carefully benchmarked. A rigorous evaluation should therefore break down latency by component, compare throughput against both standard RAG and chain-of-thought reasoning baselines under identical hardware conditions, and analyze how the relative overhead scales with base model dimension. Furthermore, the analysis must quantify the memory footprint of injected parameters and demonstrate that the hypernetwork remains lightweight and practical as the backbone model scales. This ensures that claimed efficiency advantages are not conflated with accuracy improvements and that the method's real-world deployability is validated alongside its effectiveness.

**Core Evaluation Criteria:**
- **Decomposed Latency Analysis:** Is the inference cost broken down by component (hypernetwork forward pass, sub-question generation, merging operation, base LM generation)?
- **Comparative Efficiency Benchmarking:** Are end-to-end latency and throughput compared against standard RAG, RAG-CoT, and other iterative reasoning systems under identical hardware conditions?
- **Scaling Behavior:** Does the analysis address how hypernetwork and merging overhead scales with base model dimension or layer count, and whether the relative cost diminishes for larger models?
- **Memory and Throughput Impact:** Is the additional memory footprint of injected parameters and merged experts quantified, and does the method maintain practical batch throughput?