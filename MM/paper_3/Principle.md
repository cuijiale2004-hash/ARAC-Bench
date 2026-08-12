**Principle 1: Bootstrapping Paradigm Design for Generative MLLM-Driven Weakly-Supervised Object Counting**

**Definition:**  
This principle evaluates whether a work successfully repurposes generative multimodal large language models for numeric vision tasks under weak supervision, specifically addressing the fundamental modality gap between continuous visual features and discrete text-based count tokens. It assesses whether the method exploits the MLLM's inherent reasoning capabilities—such as multi-round dialogue and relative ranking—instead of simply appending a regression head or mapping to a density map. The principle further examines whether the supervision signal is creatively utilized given that only image-level counts are available, and whether the resulting system is competitive with fully-supervised discriminative approaches that rely on costly point-level annotations. Finally, it demands clarity on how the bootstrapping strategies collectively elevate the base model's quantitative awareness beyond its pretrained sparse-scene bias. Works that merely fine-tune MLLMs for direct count regression without addressing these structural challenges fail to meet this standard.

**Core Evaluation Criteria:**
- **Modality Gap Mitigation**: Does the work explicitly address the difficulty of regressing exact counts in text token space, and does it propose alternative training objectives (e.g., range judgment, ranking) that are better aligned with generative MLLM capabilities?
- **Weak Supervision Fidelity**: Is the method strictly limited to image-level count labels during training, and does it avoid leaking point-level or bounding-box information into the pipeline?
- **Competitiveness with Fully-Supervised Baselines**: Does the proposed framework match or surpass state-of-the-art fully-supervised methods on standard benchmarks, thereby justifying the reduced annotation cost?
- **Paradigm Distinctiveness**: Is the approach clearly differentiated from discriminative VLM-based counting methods that rely on contrastive alignment or specialized counting heads?

---

**Principle 2: Interpretability of Counting Mechanisms via Attention and Progressive Behavioral Analysis**

**Definition:**  
Given the black-box nature of large multimodal models, this principle mandates that works provide mechanistic evidence explaining how visual attention aligns with object regions during count generation. Reviewers must assess whether the paper demonstrates progressive behavioral refinement across model variants—such as zero-shot, base fine-tuned, and final enhanced versions—to attribute improvements to specific design choices rather than generic fine-tuning. The principle requires quantitative and qualitative failure mode analyses that go beyond aggregate error metrics, linking performance degradation to concrete visual phenomena such as extreme density, occlusion, or small object size. Furthermore, it expects visualization of internal attention maps or response patterns to verify that the model focuses on relevant object regions rather than background textures or spurious correlations when producing count-related tokens. Ultimately, the goal is to ensure that empirical gains are traceable to genuine mechanistic improvements in spatial and numerical reasoning.

**Core Evaluation Criteria:**
- **Attention Alignment**: Does the work visualize attention weights between count-related text tokens and image patches to verify that the model grounds its predictions in actual object regions?
- **Progressive Refinement Evidence**: Is there a systematic comparison across intermediate model variants showing how each proposed strategy incrementally improves visual grounding and reduces counting error?
- **Failure Mode Characterization**: Are failure cases categorized and quantitatively correlated with scene characteristics (e.g., object size, density, occlusion level) to reveal fundamental limitations?
- **Avoidance of Black-Box Claims**: Does the paper refrain from inferring method effectiveness solely from final metrics without demonstrating intermediate reasoning or behavioral changes?

---

**Principle 3: Computational Efficiency and Scalability Assessment for MLLM-Based Vision Systems**

**Definition:**  
Because MLLMs are inherently resource-intensive, this principle requires rigorous quantification of both training and inference costs—including FLOPs, frames per second, memory footprint, and wall-clock training time—relative to weakly- and fully-supervised baselines. Works must demonstrate cost-aware design decisions, such as parameter-efficient fine-tuning strategies, that mitigate computational overhead without catastrophic performance drops. The principle also demands validation across multiple model families and scales to ensure that the method is not an artifact of a specific backbone architecture. Finally, the work should candidly discuss deployment trade-offs and practical application scenarios, acknowledging where MLLM-based counting is currently suitable and where model compression or distillation may be necessary for edge deployment. Ignoring these practical constraints undermines the real-world utility of the proposed framework.

**Core Evaluation Criteria:**
- **Cost Transparency**: Are training time, inference speed, GPU memory usage, and computational complexity explicitly reported and fairly compared against relevant baselines under identical hardware?
- **Efficiency Mitigation Strategies**: Does the work employ parameter-efficient tuning (e.g., LoRA), quantization, or other techniques to reduce the burden of fine-tuning massive multimodal backbones?
- **Backbone Scalability**: Is the method validated across diverse MLLM families and parameter scales to confirm generalizability of the bootstrapping strategies?
- **Deployment Realism**: Does the paper discuss practical deployment constraints and target scenarios honestly, rather than ignoring the computational gap between MLLMs and lightweight counting networks?

---

**Principle 4: Cross-Domain Generalization and Robustness to Visual Domain Shifts**

**Definition:**  
Object counting methods must be evaluated beyond their training distribution to ensure they are not overfitted to a specific dataset's density or object taxonomy. This principle assesses whether the proposed method maintains performance across diverse visual domains, such as crowd counting, aerial vehicle enumeration, and everyday object scenes, under varying conditions of occlusion and visual ambiguity. It specifically examines whether test-time enhancement strategies—like local-global aggregation—remain effective across datasets or are merely heuristic adjustments tuned for a single benchmark. The principle also requires disaggregated analysis of performance across density regimes, from sparse scenes to extremely dense scenes containing hundreds of instances, to verify robustness under severe distributional shift. Without such cross-domain validation, claims of class-agnostic generalization remain unsubstantiated.

**Core Evaluation Criteria:**
- **Cross-Dataset Validation**: Are experiments conducted on multiple benchmarks representing distinct visual domains, and are models trained on one dataset evaluated zero-shot or fine-tuned on others?
- **Density Regime Analysis**: Is performance reported separately for sparse, dense, and extremely dense subsets to verify that gains are not averaged away by easy cases?
- **Occlusion and Ambiguity Handling**: Does the method demonstrate robustness to heavily occluded objects, small object sizes, and visually ambiguous categories?
- **Heuristic Stability Across Domains**: Are test-time strategies (e.g., grid partitioning thresholds, local aggregation rules) validated across datasets to ensure they are not domain-specific artifacts?

---

**Principle 5: Rigor of Ablation and Differentiation from Conceptually Similar Methods**

**Definition:**  
This principle evaluates the depth of experimental ablation and the clarity of intellectual differentiation from prior work that employs similar conceptual primitives, such as ranking-based training or divide-and-conquer inference. A high-quality study must systematically isolate each proposed component, test alternative implementations and hyperparameters, and demonstrate that the final combination is non-obvious and empirically superior to simpler variants. It must also explicitly contrast with contemporary methods—such as VLM-based ranking or detection-assisted partitioning—to establish that the proposed strategies are not trivial recombinations of existing ideas. The principle further expects sensitivity analyses for key hyperparameters to prove that performance gains are robust to design choices rather than the result of narrow tuning. Superficial ablations or vague positioning against related work indicate insufficient experimental rigor and novelty.

**Core Evaluation Criteria:**
- **Component Isolation**: Is each core strategy ablated individually and in combination to establish its marginal contribution and complementarity?
- **Design Choice Justification**: Are key hyperparameters (e.g., grid size, number of ranking samples, dialogue stopping thresholds) systematically swept and empirically justified?
- **Related Work Differentiation**: Does the work clearly distinguish its methodology from conceptually similar approaches, explaining why alternatives (e.g., contrastive ranking, detection-based partitioning) are inferior or incompatible?
- **Ablation of Plausible Alternatives**: Are inferior but reasonable variants (e.g., cross-category ranking, single-round dialogue, external segmentation masks) tested to validate that the proposed design decisions are indeed optimal?