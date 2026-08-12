**Principle 1: Causal Validation of Proactive Multi-Evidence Visual Reasoning Bottlenecks in Long-Form Human-Centric Video Benchmarks**

**Definition:**  
For benchmarks targeting complex video reasoning in human-centric scenes, authors must move beyond correlational error analyses to provide causal evidence that identifies specific reasoning bottlenecks. Because models must integrate multiple disparate visual evidence pieces and proactively extract implicit cues beyond query text, it is crucial to design controlled interventions—such as graded hinting, oracle evidence supply, or query rewriting—that isolate whether performance limitations stem from perceptual search, evidence selection, or higher-order reasoning. This ensures that claimed “bottlenecks,” such as proactive evidence extraction, are not inferred solely from aggregate performance gaps or manual error inspection, but are supported by experimental manipulations that selectively relieve specific constraints and quantify marginal gains.

**Core Evaluation Criteria:**
- **Controlled Intervention Design**: Does the work include graded ablations (e.g., relation-aware hints vs. logic-aware hints vs. vague proactive evidence descriptions) that independently manipulate evidence accessibility while holding the core reasoning task constant?
- **Causal Interpretation Strength**: Are claims about failure mechanisms explicitly scoped to the intervention results, avoiding overgeneralization from correlational patterns? Is it clear that performance gains track the specific manipulation (e.g., +10–13% only when proactive evidence is hinted) rather than generic prompting improvements?
- **Isolation of Sub-capabilities**: Does the analysis disentangle “seeing” (perception), “finding” (evidence search), and “reasoning” (integration/inference) through targeted upper-bound or oracle experiments?
- **Reproducibility of Manipulation**: Is the intervention protocol precisely defined so that the proactive/referred distinction is maintained consistently without collapsing into text-guided retrieval shortcuts?

---

**Principle 2: Robustness and Longitudinal Stability of LLM-Based Judging in Open-Ended Multimodal Reasoning Evaluation**

**Definition:**  
In open-ended video reasoning benchmarks where human evaluation is infeasible at scale, the adoption of LLM-as-a-judge introduces risks of metric drift, self-bias, and domain-dependent sensitivity. A rigorous evaluation framework must demonstrate that LLM judges are reliable proxies for human judgment across diverse reasoning types and domains, provide uncertainty quantification for reported scores, and establish mechanisms for longitudinal stability—such as human-adjudicated gold sets—to ensure comparability as judge models evolve and future models are released.

**Core Evaluation Criteria:**
- **Multi-Judge Consistency and Human Alignment**: Does the work report agreement across multiple LLM judges (not just one) and quantify Pearson/Win-loss correlation against human annotations, broken down by reasoning type and domain?
- **Uncertainty Quantification**: Are confidence intervals, standard deviations across repeated runs, or variance estimates reported for key metrics to establish score stability?
- **Absence of Self-Bias**: Is statistical testing conducted to rule out favoritism toward same-family models (e.g., an OpenAI judge favoring OpenAI models)?
- **Longitudinal Calibration Plan**: Does the work commit to releasing a human-adjudicated gold subset to enable future recalibration of automated judges and track metric drift over time?

---

**Principle 3: Diagnostic Granularity and Non-Redundancy of Fine-Grained Human-Centric Perception/Comprehension Taxonomies**

**Definition:**  
Fine-grained perception and comprehension benchmarks in human-centric vision must prove that their taxonomic decomposition provides diagnostic value beyond simply increasing task counts. Because prior benchmarks aggregate performance into coarse categories (e.g., “action” or “relation”) that obscure heterogeneous failure modes, new taxonomies must empirically demonstrate that neighboring subtasks within the same high-level dimension exhibit divergent model rankings, differential sensitivity to prompting strategies (e.g., Chain-of-Thought), and distinct error profiles. This confirms that the granularity captures non-redundant capabilities rather than trivial subdivisions, and that collapsing these subtasks would hide systematic weaknesses.

**Core Evaluation Criteria:**
- **Heterogeneous Subtask Response**: Does the work show that model rankings or relative performance shifts change across fine-grained subtasks (e.g., self-contact vs. human-object contact) within the same coarse dimension?
- **Conditional Strategy Effectiveness**: Is there evidence that intervention effects (e.g., CoT gains) vary systematically across subtask granularity, indicating that task difficulty and visual demand are genuinely differentiated?
- **Coverage Justification**: Does the benchmark explicitly map its fine-grained tasks (e.g., gaze estimation, body orientation, long-term procedure understanding) against prior human-centric benchmarks to justify that the taxonomy isolates previously collapsed or absent capabilities?
- **Complementarity to Reasoning Benchmarks**: Is the relationship between fine-grained perception/comprehension and higher-level reasoning clearly articulated, showing how the former provides necessary diagnostic context for the latter?

---

**Principle 4: Controlled Equivalence and Transparency of Evaluation Configurations Across Heterogeneous Multimodal Model Families**

**Definition:**  
When evaluating a diverse suite of MLLMs spanning proprietary APIs and open-source architectures with variable context windows, inference efficiencies, and reasoning modes, strict fairness demands transparent documentation of evaluation configurations. The primary comparative results must be generated under uniform conditions (e.g., consistent prompting templates, standardized frame sampling policies within task levels), while enhanced settings (e.g., maximum frame budgets, Best-of-N sampling, test-time reasoning) must be quarantined to ablation studies. Additionally, computational costs (wall-clock time, GPU hours, memory) must be reported to enable reproducibility and practical adoption assessments.

**Core Evaluation Criteria:**
- **Primary Leaderboard Uniformity**: Are the main results obtained with identical evaluation protocols (prompts, decoding parameters, frame sampling) across all models, or are model-specific optimizations explicitly segregated?
- **Frame and Context Budget Fairness**: For perception/comprehension versus reasoning tasks with differing visual complexity, are frame sampling strategies justified by task requirements and applied consistently within each tier, with per-model optimal configurations only reported as supplementary ablations?
- **Compute Transparency**: Are wall-clock runtimes, GPU hours, and hardware configurations documented for representative model scales?
- **Separation of Test-Time Scaling**: Are techniques like Best-of-N, self-refine, or specialized reasoning models clearly distinguished from base-model evaluations and not commingled in primary rankings?

---

**Principle 5: Rigorous Annotation Protocols for Proactive Evidence Definition and Human Baseline Validity in Complex Video QA**

**Definition:**  
The validity of a human-centric video reasoning benchmark hinges on rigorous annotation protocols that objectively define and verify the core cognitive demands—specifically the distinction between proactive and referred visual evidence. Annotations must be reproducible (via weighted scoring or LLM-assisted decomposition with human refinement), grounded in domain expertise (with annotator qualifications specified), and protected against shortcut behaviors (through blind filtering, meta-review, and controlled human baselines). Human baseline protocols must approximate realistic usage without conferring unfair advantages (e.g., prohibiting video/image search while allowing text translation) and must avoid data leakage by using evaluators separate from dataset constructors.

**Core Evaluation Criteria:**
- **Objective Quantification of Evidence Types**: Is the proactive/referred distinction operationalized through an explicit, reproducible protocol (e.g., weighted “proactivity degree” scores, multi-stage human verification) rather than subjective intuition?
- **Annotator Qualification and Sourcing**: Are domain experts recruited with relevant backgrounds for specialized domains, and is the total number, sourcing platform, and qualification process disclosed?
- **Anti-Shortcut Safeguards**: Are questions filtered to remove text-only solvability, and is meta-review used to enforce that every accepted question requires multiple visual evidence pieces including at least one proactive cue?
- **Human Baseline Validity**: Is the human evaluation protocol described in sufficient detail (e.g., text-only open-book allowances, prohibition of reverse image/video search, separation of evaluators from annotators) to support claims of human vs. model gaps?
- **Data Contamination Mitigation**: Are safeguards described (test/validation splits for sourced data, recent web videos, novel question formulation) to minimize overlap with pretraining corpora?