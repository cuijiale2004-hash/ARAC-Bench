Principle 1: Comprehensive Baseline Coverage and Controlled Comparison Protocols for Event-Based Multimodal Learning

**Definition:**  
In the emerging domain of event-based multimodal large language models, fair and exhaustive benchmarking is essential because the field lacks standardized evaluation protocols and publicly accessible baselines. Reviewers must assess whether a work adequately compares its proposed method against direct architectural competitors—both event-native models and strong frame-based adaptations—under controlled conditions such as matching parameter scales, fine-tuning protocols, and input representations. When direct competitors are not open-sourced, authors must provide rigorously justified internal baselines that replicate standard architectures rather than implicitly favorable variants. Furthermore, comparisons against RGB-based or video-based MLLMs must explicitly clarify whether these models are evaluated zero-shot or after modality-specific fine-tuning to avoid misleading performance impressions. A failure to satisfy these standards undermines claims of state-of-the-art performance and prevents the community from accurately gauging incremental progress. The absence of such benchmarking also limits reproducibility and stalls the establishment of shared progress bars in nascent fields.

**Core Evaluation Criteria:**
- **Coverage of Direct Competitors**: Does the work benchmark against all relevant event-based MLLMs that are publicly available, and does it explicitly acknowledge and justify omissions when models are closed-source?
- **Internal Baseline Justification**: Are internally constructed baselines (e.g., removing proposed modules to create a "zero" variant) designed following documented, standard architectural recipes rather than intentionally weak configurations?
- **Controlled Cross-Modality Comparison**: When comparing against frame-based or video-based MLLMs, does the work clearly distinguish between zero-shot transfer and fine-tuned adaptation, and does it match model scales and compute budgets?
- **Scale and Protocol Matching**: Are throughput and accuracy comparisons performed at identical parameter scales and training configurations to ensure that gains stem from methodological innovations rather than capacity disparities?

---

Principle 2: Alignment between Claimed Raw Event Stream Processing and Actual Representation Encoding Paradigms

**Definition:**  
A persistent tension in event-based vision research lies in the gap between methodological claims of handling "raw," asynchronous event streams and the practical use of dense, image-like proxy representations for compatibility with existing vision encoders. Reviewers must scrutinize whether the paper’s sparsification or efficiency modules truly operate on native event structures—such as point clouds, graphs, or spike trains—or instead process converted image-based tokens while borrowing event-derived metadata. This distinction matters because operating on proxy representations may inherit frame-based inductive biases that negate the theoretical advantages of event cameras, such as high temporal resolution and sparse dynamic sensing. The work must transparently disclose any representation conversion pipelines and justify why native asynchronous encoders were infeasible, particularly regarding dataset scale and pretraining limitations. Ultimately, principled progress in event-based MLLMs requires clarity on whether efficiency gains are achieved within an event-native computational graph or merely by sparsifying converted dense tensors. Misalignment between claims and implementations can misdirect future research toward incremental frame-based adaptations rather than fundamental event-native architectures.

**Core Evaluation Criteria:**
- **Transparency of Representation Conversion**: Does the paper explicitly describe the pipeline from raw asynchronous events to the final token format fed into the MLLM, including any image-like intermediate representations?
- **Native vs. Proxy Modality Fidelity**: Do the proposed spatiotemporal modules operate on event-native features (density, spike times, polarity) before or independently of image-based encoding, or do they merely post-process dense visual tokens?
- **Justification for Proxy Usage**: If native event representations are abandoned in favor of image-like proxies, does the work provide a rigorous argument based on current encoder limitations, dataset scarcity, or alignment challenges with LLM backbones?
- **Consistency of Terminology**: Does the paper avoid conflating "raw event stream processing" with "processing of event-derived dense frames," and does it clearly distinguish its scope as efficiency optimization within existing dense paradigms versus novel event-native architectures?

---

Principle 3: Component-Level Ablation Rigor and Incremental Validation for Multi-Stage Spatiotemporal Sparsification

**Definition:**  
Event-based MLLMs frequently introduce complex, multi-stage modules—such as cascaded temporal merging followed by semantic aggregation—to exploit spatiotemporal sparsity. Reviewers must demand granular ablation evidence that isolates the contribution of each internal stage, rather than validating only the composite block. A two-stage design must be justified against simpler single-stage alternatives to demonstrate that its complexity is mechanistically necessary and not merely an incremental stacking of existing techniques. Authors should quantify how each stage affects both computational efficiency and task accuracy across diverse event understanding benchmarks. Without such dissection, the community cannot determine whether observed gains arise from a specific mechanism or from an opaque accumulation of heuristics. Furthermore, sequential dependencies between stages must be carefully unpacked to ensure that reported ablations are not methodologically confounded by irreversible data transformations. This level of rigor is crucial for transforming isolated engineering successes into generalizable principles for token sparsification in multimodal systems.

**Core Evaluation Criteria:**
- **Stage-Wise Isolation**: Does the ablation study evaluate each stage of a multi-stage module independently (where feasible) to isolate its additive contribution to accuracy and efficiency?
- **Comparison with Simplified Alternatives**: Are complex multi-stage designs compared against principled single-stage baselines that merge analogous information using simpler heuristics or direct averaging?
- **Joint and Separate Impact Quantification**: Does the work report throughput and accuracy metrics for temporal-only, spatial-only, and combined sparsification to demonstrate orthogonality or synergy between components?
- **Mechanistic Justification**: Are the sequential dependencies between stages explained with empirical evidence that collapsing them degrades performance or violates representational constraints?

---

Principle 4: Contextualized Throughput Metric Definition and Fair Efficiency Evaluation in Sparse Token Architectures

**Definition:**  
Efficiency claims in sparse token architectures are highly sensitive to the choice of baseline, context length, and metric normalization. Reviewers must evaluate whether reported speedups are measured against a fair internal baseline operating under identical conditions or against unavailable external state-of-the-art models using incompatible configurations. In event-based vision, throughput must be contextualized by the temporal span or bin capacity supported, because raw throughput can be artificially inflated by truncating long sequences. Authors should report context-normalized efficiency metrics that reflect the effective computational cost of processing long-duration event streams without losing motion dynamics. Additionally, hardware and deployment settings must be standardized across compared methods to ensure reproducible benchmarking. Without such contextualization, efficiency claims risk becoming marketing artifacts that obscure the true performance-efficiency trade-offs governing real-world deployment of event-based MLLMs.

**Core Evaluation Criteria:**
- **Baseline Fairness for Speedup Claims**: Is the throughput improvement calculated against an internal baseline that differs only by the proposed sparsification mechanism, and is this baseline architecturally sound and reproducible?
- **Context-Length Normalization**: Does the work report throughput alongside the maximum supported event bins or temporal context, and does it explicitly contrast long-context capability with competitors rather than relying solely on raw tokens-per-second?
- **Standardized Deployment Conditions**: Are all throughput measurements conducted on the same hardware and software stack, with explicit reporting of batch size, precision, and sequence length?
- **Performance-Efficiency Trade-off Analysis**: Does the paper quantify the accuracy degradation (if any) associated with each level of throughput gain, and does it identify optimal operating points rather than reporting only the highest speedup?

---

Principle 5: Transparency and Verification Rigor in Automated Large-Scale Instruction Dataset Generation for Event Vision

**Definition:**  
The construction of large-scale instruction-following datasets has become a cornerstone contribution in event-based MLLM research, yet the automated nature of their generation introduces significant risks of label noise, modality misalignment, and quality drift. Reviewers must scrutinize whether the annotation pipeline, particularly when relying on powerful RGB-trained VLMs or LLMs, is designed to respect the unique spatiotemporal characteristics of event data rather than importing static, color-dependent biases. The work must provide detailed documentation of quality-control mechanisms, including automated metadata validation, cross-model consistency checks, and human inspection protocols, rather than vague assurances of manual checking. Furthermore, the source of supervisory signals must be explicit, as annotations derived from clean RGB frames may not generalize to the degradation scenarios where event cameras excel. A rigorous dataset paper should also demonstrate that the generated instructions cover diverse temporal scales and scene types proportional to real-world event camera deployment. Without such transparency, datasets risk becoming opaque black boxes that hinder reproducibility and inflate performance through superficially large but low-quality training corpora.

**Core Evaluation Criteria:**
- **Annotation Source and Modality Alignment**: Does the paper clearly state whether text annotations are generated from paired RGB video, raw event representations, or hybrid sources, and are static or color biases explicitly filtered out?
- **Automated Quality Control Documentation**: Is the automated filtering and validation pipeline described with sufficient detail—including prompt design, lightweight model cross-checking, and metadata extraction—to permit replication?
- **Human Verification Protocol**: Are the scale, sampling strategy, and criteria for human inspection explicitly reported, rather than mentioned as an afterthought?
- **Temporal and Task Diversity Justification**: Does the dataset demonstrate balanced coverage across short, medium, and long event durations and across diverse task types, and is this distribution justified by downstream evaluation needs?