**Principle 1: Architectural Necessity and Mechanistic Attribution in Region-Level Feature Extraction**

**Definition:**  
In region-level MLLM research, it is insufficient to demonstrate superior end-task performance alone; reviewers must assess whether the proposed architectural modifications constitute a necessary advancement over simpler, well-established alternatives. This principle emphasizes the need for authors to isolate the causal impact of specific design choices—such as unpooled feature replay versus RoI-pooling, direct cropping, or cross-attention—through controlled ablations that hold data, training regimes, and base models constant. The core objective is to prevent "architectural over-engineering" by ensuring that each module provides a non-trivial, empirically quantifiable benefit that cannot be replicated by trivial baselines or prior recipes. Furthermore, the work must articulate the mechanistic rationale for why the chosen design avoids information bottlenecks inherent in simpler approaches. Ultimately, this principle discriminates between incremental embellishments and genuinely necessary architectural innovations by demanding proof that complex components outperform minimal baselines by a substantial margin.

**Core Evaluation Criteria:**
- **Minimal Baseline Supremacy**: Does the work compare against the simplest plausible alternative (e.g., directly cropping and resizing the local region) and demonstrate a substantial, statistically meaningful gap?
- **Component Isolation**: Are controlled ablations provided that isolate the contribution of each novel module (e.g., removing the RoI-Aligned Feature Replay module while keeping all other variables fixed)?
- **Mechanistic Clarity**: Does the paper clearly explain the information flow (e.g., preservation versus aggregation) and why the proposed method avoids information bottlenecks that prior methods suffer from?
- **Efficiency Trade-off Analysis**: Is the computational or memory overhead of the proposed architecture justified by the performance gains relative to simpler baselines?

---

**Principle 2: Rigorous Validation of Global-Local Context Fusion and Misbinding Avoidance**

**Definition:**  
A central challenge in region-level understanding is balancing fine-grained local detail with holistic global context. This principle requires that authors provide compelling evidence that their model genuinely integrates global context rather than either ignoring it or suffering from "context misbinding" (incorrectly associating a region with irrelevant global cues). Reviewers evaluate whether the work moves beyond asserting global-context awareness to empirically proving it through targeted interventions, task-specific attribution, and diagnostic experiments that disentangle local feature quality from global reasoning capability. Authors must demonstrate that performance gains on context-dependent tasks are specifically attributable to successful fusion mechanisms rather than to spurious correlations or improved local extraction alone. Failure to validate this fusion can lead to models that perform well on isolated descriptions but fail on compositional reasoning requiring scene-level semantics.

**Core Evaluation Criteria:**
- **Context Binding Diagnostics**: Are there targeted experiments that disrupt global-context integration (e.g., removing prompt-aware masks from global features) to show performance collapse, thereby proving the model relies on correct context?
- **Task-Specific Attribution**: Does the work explicitly identify which sub-tasks (e.g., non-entity recognition, spatial position reasoning) fundamentally require global context, and demonstrate that performance gains on those tasks are specifically attributable to global-local fusion?
- **Disentanglement of Local vs. Global Gains**: Does the evaluation distinguish between improvements stemming purely from better local feature extraction versus improvements from global-context reasoning (e.g., by holding local performance constant on local-only benchmarks while improving on context-dependent benchmarks)?
- **Qualitative Failure Analysis**: Are failure cases presented that reveal whether errors stem from local detail loss, global context neglect, or misbinding?

---

**Principle 3: Benchmark Stability, Judge Validity, and Evaluation Hygiene for Region Reasoning**

**Definition:**  
The introduction of new benchmarks for compositional region reasoning demands rigorous meta-evaluation of the evaluation protocol itself. This principle mandates that authors demonstrate their benchmark is not only challenging but also statistically stable, resistant to evaluator bias, and diagnostically informative. Given the prevalence of LLM-as-a-Judge methodologies in fine-grained vision-language evaluation, authors must prove that their conclusions are robust across different judge models, subsampling regimes, and ideally human validation. Additionally, strict data hygiene must be maintained to ensure no overlap or leakage exists between training corpora and benchmark images, particularly when benchmark data is sourced from widely used pre-training datasets. Without this meta-evaluation, benchmark claims risk being dismissed as artifacts of small sample sizes, single-judge stylistic preferences, or contaminated test sets.

**Core Evaluation Criteria:**
- **Statistical Stability**: Is the benchmark's ranking order stable under subsampling (e.g., 50% or 25% of data) to ensure conclusions are not artifacts of small sample sizes?
- **Cross-Judge Consistency**: Do model rankings remain consistent when evaluated by diverse LLM judges (e.g., GPT-4o, Gemini, o3) with different stylistic preferences, or are results tethered to a single evaluator's bias?
- **Data Hygiene Transparency**: Are data leakage checks, deduplication procedures, and the provenance of benchmark images (especially out-of-distribution sources) clearly documented to ensure fair evaluation?
- **Human Validation Preference**: Where LLM judges are used, is there either human validation or a clear justification for why LLM-based evaluation is sufficient for the specific task?

---

**Principle 4: Generalization Across Visual Prompt Formats and Preservation of Generalist Capabilities**

**Definition:**  
Region-level MLLMs must demonstrate that their capabilities reflect a fundamental advancement in dense scene understanding rather than overfitting to a specific input modality or task formulation. This principle evaluates whether the method generalizes across diverse region-specification formats (masks, bounding boxes, points, outlines) and whether specialized training on region tasks catastrophically degrades general multimodal competencies. The goal is to distinguish between narrow task-specific engineering and broadly applicable visual reasoning that transfers to adjacent domains. Authors should therefore validate their models on standard general-purpose multimodal benchmarks and demonstrate zero-shot transfer to related tasks such as video understanding. Furthermore, if specialization induces performance drops on general tasks, the work should present effective mitigation strategies to confirm that the trade-off is manageable rather than fundamental.

**Core Evaluation Criteria:**
- **Cross-Format Robustness**: Does the work evaluate performance across multiple visual prompt formats to rule out "input bias" (i.e., the possibility that baseline models fail merely because they cannot interpret masks)?
- **General Benchmark Retention**: Are results reported on standard general-purpose multimodal benchmarks (e.g., MMVP, RealWorldQA, MMStar) to verify that specialization does not induce catastrophic forgetting of general VQA abilities?
- **Zero-Shot Transfer**: Does the model demonstrate transferable reasoning to adjacent domains (e.g., zero-shot video understanding) without in-domain training, indicating the learned representations are task-agnostic?
- **Mitigation of Forgetting**: If general capabilities degrade, does the work provide effective and standard mitigation strategies (e.g., data mixing) to recover general performance while retaining specialized skills?

---

**Principle 5: Precise Novelty Positioning and Differentiation from Prior Global-Local Architectures**

**Definition:**  
In a mature subfield where multiple works have explored global-local feature fusion for region understanding, novelty must be precisely scoped and empirically defended. This principle requires authors to identify the exact technical axis along which their method advances beyond closest architectural cousins and to provide minimal-pair ablations that isolate this specific axis. Reviewers assess whether the paper avoids overclaiming by honestly acknowledging similarities in high-level recipes while rigorously proving the material impact of low-level mechanistic differences. Authors must clearly articulate why a specific design choice—such as prompt-aware encoding or unpooled token replay—functionally outperforms the aggregation strategies used in prior work. Ultimately, this principle ensures that contributions are evaluated based on honest differentiation and controlled empirical evidence rather than on vague narratives of holistic superiority.

**Core Evaluation Criteria:**
- **Minimal-Pair Ablation**: Is there a direct, fair comparison where the proposed method is compared against a baseline that replicates the core design of the most similar prior work (e.g., GPT4RoI-style RoI-pooling), holding all other factors (LLM, data, training) constant?
- **Technical Distinction Clarity**: Does the paper clearly articulate the mechanistic difference (e.g., "preservation and selection" versus "aggregation and compression") and explain why this difference matters functionally?
- **Narrative Accuracy**: Does the related work and introduction avoid strawman arguments or overstated claims about prior methods (e.g., claiming prior works "neglect global context" when they in fact incorporate it via pooling), and instead frame the contribution as solving a specific trade-off?
- **Empirical Gap Validation**: Do the ablation results show a large, consistent performance gap on benchmarks specifically designed to stress the claimed difference (e.g., fine-grained detail benchmarks for unpooled features)?