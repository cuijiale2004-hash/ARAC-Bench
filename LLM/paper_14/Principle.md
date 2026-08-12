Principle 1: Rigorous Verification of Technical Novelty and Mechanistic Differentiation in Structured KG-LLM Planning Pipelines

**Definition:**  
In the subfield of KG-augmented LLM reasoning, numerous frameworks share superficially similar pipeline architectures, such as iterative graph exploration or plan-answer-refine loops. Reviewers therefore critically evaluate whether claimed innovations constitute genuine mechanistic advances or merely repackage existing paradigms with minor variations. This principle demands that authors provide explicit causal explanations of how each proposed component alters the reasoning trajectory compared to the closest baselines, rather than relying solely on end-to-end accuracy improvements. High-quality work must include targeted ablations that isolate the effect of individual stages and demonstrate that structural similarities to prior methods mask substantive differences in error propagation, search dynamics, or correction behavior. The evaluation further requires that authors anticipate and directly address comparisons with the most related concurrent work, clearly articulating the technical boundary between inspiration and incremental adaptation.

**Core Evaluation Criteria:**
- **Explicit Mechanistic Distinction**: Does the paper clearly differentiate its method from structurally similar prior work (e.g., SPARQL-grounded decomposition versus heuristic decomposition) and explain how this changes the reasoning dynamics?
- **Isolated Ablation of Innovations**: Are individual components ablated within the full pipeline to prove their independent contribution beyond the overall framework?
- **Error-Dynamic Evidence**: Does the work demonstrate altered failure modes (e.g., reduced truncation rates, changed correction behavior) rather than only reporting final accuracy gains?

---

Principle 2: Quality Assurance and Scalability Assessment of Symbolic-Supervised Training Data for KG Reasoning

**Definition:**  
Frameworks that train specialized modules for KG reasoning increasingly rely on automatically generated corpora derived from symbolic queries or LLM teachers. Reviewers treat the quality, semantic fidelity, and scalability of this supervision signal as a first-class evaluation criterion because the downstream planner's reliability is fundamentally bounded by the data it ingests. This principle requires evidence of rigorous data validation, including estimated error rates, semantic equivalence checks between source symbolic forms and generated natural language plans, and robustness to noisy or ambiguous inputs. Authors must also demonstrate that the data pipeline is not tethered to a single domain or expensive proprietary teacher model, but can scale to diverse schemas and resource settings. Without such quality assurance, claims of efficient small-model planning risk being undermined by hidden annotation artifacts or distribution-specific memorization.

**Core Evaluation Criteria:**
- **Quantified Data Quality Metrics**: Are error rates, semantic equivalence rates, or consistency checks reported for the generated training corpora?
- **Teacher-Model Dependency Analysis**: Does the work analyze reliance on the teacher model (e.g., GPT-4o) used for data generation and propose paths to reduce dependence on expensive proprietary models?
- **Robustness to Input Noise**: Is the planner validated on noisy, ambiguous, or out-of-distribution symbolic inputs to ensure it does not merely memorize benchmark-specific patterns?

---

Principle 3: Faithfulness Auditing and Causal Tracing of Self-Refinement and Error-Correction Mechanisms

**Definition:**  
For systems incorporating self-refinement or retrieval-based error correction, reviewers require rigorous evidence that refinement steps genuinely repair reasoning errors rather than selectively reinforcing fortunate initial guesses or introducing new hallucinations. This principle emphasizes the need for execution-grounded verification, where refinement decisions are traceable to concrete KG triples rather than opaque LLM re-generation. Authors must quantify the correction rate on initially incorrect tentative answers, report rates of erroneous overrides of correct initial answers, and analyze cases where retrieved evidence is irrelevant or misleading. The evaluation distinguishes superficial performance gains from true stability improvements by establishing a causal link between the refinement mechanism and measurable changes in failure modes. Ultimately, a trustworthy refinement module must demonstrate that it reduces error amplification under both strong and weak base LLM configurations.

**Core Evaluation Criteria:**
- **Correction-Rate Quantification**: Does the paper explicitly report the percentage of initially wrong tentative answers that are corrected, and conversely, the rate at which correct answers are erroneously overridden?
- **Execution-Grounded Constraints**: Are refinement decisions constrained by verifiable KG triples rather than relying on free-form LLM regeneration without external validation?
- **Behavior Under Varying Base LLM Capabilities**: Is refinement robustness validated across weaker and stronger base LLMs to ensure it does not depend on high-quality initial guesses?

---

Principle 4: Cross-Schema Generalization and Robustness Evaluation Under KG Incompleteness and Domain Shift

**Definition:**  
Because KGQA benchmarks are often anchored to specific knowledge graphs such as Freebase, reviewers prioritize evidence that methods learn transferable reasoning patterns rather than overfitting to benchmark-specific schemas or entity vocabularies. This principle demands validation across heterogeneous KGs with distinct relation granularities, naming conventions, and topological structures to ensure that planning and refinement mechanisms are schema-agnostic. Authors must also evaluate robustness under KG incompleteness, where gold relational paths are absent, requiring the system to either gracefully degrade or effectively leverage parametric knowledge without hallucination. Furthermore, the evaluation scrutinizes whether performance depends on the LLM possessing strong domain-specific prior knowledge, which may not generalize to truly novel or low-resource domains. Cross-domain experiments and explicit incompleteness stress tests are essential to distinguish robust compositional reasoning from benchmark-specific engineering.

**Core Evaluation Criteria:**
- **Multi-KG Schema Validation**: Are experiments conducted on KGs with different organizations, relation granularities, and naming conventions (e.g., Freebase versus Wikidata)?
- **Incompleteness Stress Testing**: Does the work evaluate behavior when KG triples are missing or when gold paths are absent, requiring bridging of KG gaps?
- **Disentanglement from Parametric Priors**: Is performance analyzed under settings where the base LLM lacks domain-specific prior knowledge to ensure gains stem from the KG-reasoning framework itself?

---

Principle 5: Transparency and Reproducibility of Multi-Stage Reasoning Pipelines and Decision Boundaries

**Definition:**  
Complex KGQA pipelines involve multiple interacting stages—planning, entity-relation pruning, tentative answering, and refinement—each with implicit decision boundaries that strongly influence reproducibility and fair comparison. Reviewers evaluate whether authors fully disclose implementation details including exact prompt templates, stopping criteria, model assignments per stage, and fallback behaviors when KG evidence conflicts with parametric answers. This principle requires that presentation clarity match technical complexity, ensuring that figures, pseudocode, and notation faithfully represent the described methodology without obscuring the core insight. Transparency about why and how the pipeline transitions between stages enables the community to assess whether observed gains stem from algorithmic innovation or undocumented engineering optimizations. Reproducibility is further supported by comprehensive error analyses that break down failures into actionable categories beyond aggregate accuracy metrics.

**Core Evaluation Criteria:**
- **Per-Stage Implementation Disclosure**: Are the specific models, prompts, stopping criteria, and filtering logic used at each pipeline stage fully specified?
- **Consistency Between Description and Artifacts**: Do figures, algorithms, and textual descriptions align accurately without inconsistencies or notational obfuscation?
- **Granular Error Taxonomy**: Does the paper provide qualitative and quantitative failure case analysis that classifies errors into actionable categories (e.g., relation selection errors, missing-entity retrieval, KG incompleteness)?