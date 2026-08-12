
## Principle 1: Specification Rigor of Non-Parameter Bayesian Prompt-Distribution Updates in LLM Multi-Agent Code Generation Frameworks

**Definition:**
When an LLM-agent system claims to perform "Bayesian updating" without gradient-based training, reviewers must demand an unambiguous specification of the probabilistic object actually being updated. In this subfield, the update target is typically the sampling distribution over prompt components (test cases, sample codes), not the LLM's internal parameters — a distinction that confused multiple reviewers here ("what weights are being updated?") and required explicit rebuttal clarification. The pseudo-code must be mathematically self-consistent: every symbol introduced must be initialized, used in subsequent steps, and connected to the update equations (the unused λ in Algorithm 1 was a substantive flaw, not a cosmetic one). Prior, likelihood, and posterior terms must each be given explicit functional forms rather than intuitive prose. This principle is crucial because "Bayesian" terminology is frequently used decoratively in agent-framework papers, and only precise mechanistic disclosure allows reviewers to distinguish genuine probabilistic inference from heuristic re-weighting dressed in Bayesian language. Works failing this criterion project an illusion of principled design; works passing it substantially raise confidence in soundness, reproducibility, and the validity of claimed uncertainty reduction.

**Core Evaluation Criteria:**
- **Unambiguous Update Target**: Does the work explicitly state what is being updated (prompt-component distributions vs. model parameters vs. test-case difficulty scores), and is this distinction maintained consistently across text, equations, and pseudo-code?
- **Mathematical Completeness**: Are prior, likelihood, posterior, and any surrogate/acquisition terms each given explicit functional forms, with all algorithmic symbols defined, initialized, and actually consumed by later steps?
- **Substantive vs. Decorative Probabilistic Machinery**: Is the Bayesian/GP component load-bearing — i.e., would ablating it degrade performance — or is it terminological ornamentation over a deterministic scoring heuristic?
- **Notation and Presentation Discipline**: Are figure legends, loop semantics, equation references, and variable names free of errors that materially impede understanding of the algorithm's control flow?

---

## Principle 2: Validity, Discriminative Power, and Anti-Leakage Guarantees of LLM-Generated Test Cases in Adversarial Co-Evolution Loops

**Definition:**
In adversarial multi-agent code generation, the same class of LLM produces both the solutions and the tests that judge them, creating a circular-validation risk: hallucinated or trivial ("dummy") tests can certify flawed code, and errors compound through the pipeline. Reviewers must therefore examine whether the framework includes an explicit, execution-grounded mechanism for scoring test-case quality — here, a "hardness" score measuring each test's ability to discriminate high-quality from low-quality code via the pass/fail score differential. It must be demonstrated, not assumed, that the adversarial dynamics filter out dummy tests and converge toward informative edge cases rather than degenerating into always-pass or always-fail equilibria. When domain experts supply prior knowledge (physical formulas, modeling advice), the work must additionally show that such advice is restricted to conceptual knowledge and does not leak ground-truth implementations, preserving evaluation fairness. This principle is vital for AI4S because scientific correctness typically exceeds what standard unit tests can verify, and the evaluator's reliability bounds the entire system's reliability. The strongest evidence is an ablation showing that removing adversarial test evolution causes measurable degradation in later iterations.

**Core Evaluation Criteria:**
- **Anti-Dummy-Test Mechanism**: Is there a quantitative, execution-based criterion for test-case quality (e.g., discriminative power between passing and failing code populations), rather than reliance on correlated LLM self-assessment?
- **Convergence of Co-Evolution**: Do experiments show monotonic robustness improvement from iterative adversarial refinement, with a clear divergence between with/without-adversarial-test ablations as iterations proceed?
- **Expert-Knowledge Leakage Control**: Is domain advice explicitly filtered to exclude solution-level code, and are leakage-prevention protocols disclosed — especially in case studies where expert advice is highly specific?
- **Grounding in External Execution**: Are evaluation signals derived from actually executing code against tests rather than from LLM judgments, thereby decorrelating solver and evaluator hallucinations?

---

## Principle 3: Standardized, Attributable, and Multi-Granular Experimental Comparison Across Backbone Models, Benchmarks, and Agentic Baselines

**Definition:**
Model-agnostic agent frameworks derive their credibility from demonstrating that observed gains stem from the framework's design rather than from discrepancies in underlying base models. Reviewers must verify that all compared methods — including commercial IDE tools used in case studies — are evaluated under a standardized, explicitly disclosed base model, since backbone choice has a first-order effect on downstream performance (a key reviewer complaint here was the initially missing base-model disclosure for Cursor/Windsurf comparisons). Because no single benchmark establishes generality, the experimental matrix should span multiple granularity levels: general programming benchmarks, domain-specific scientific benchmarks, complex multi-step agentic workflows, and real-world cross-disciplinary case studies. Where a key experiment employs only a zero-shot baseline, the work should justify that experiment's purpose (e.g., studying scalability across model sizes) and compensate with rich agentic comparisons elsewhere. Reviewers should also check that metrics suit the scientific setting — for instance, Valid Execution Rate may better reflect reliability than raw pass@1. The overarching question is whether performance differences are interpretable and causally attributable to the proposed mechanism rather than to confounding experimental choices.

**Core Evaluation Criteria:**
- **Base-Model Standardization**: Are backbone models identical and disclosed across all compared systems (including case-study competitors), so framework effects are isolated from model effects?
- **Benchmark Stratification**: Does evaluation cover general code tasks, scientific code tasks, multi-step agentic workflows, and real-world case studies, with each experiment's role in the overall argument clearly articulated?
- **Baseline Sufficiency and Fairness**: Are state-of-the-art agentic baselines (not merely prompting baselines) included under matched settings, and are thin-baseline experiments justified and complemented by broader comparisons?
- **Metric Suitability for Scientific Reliability**: Do reported metrics capture execution validity and domain-constraint adherence — not only functional pass rates — reflecting the high-stakes nature of scientific code?

---

## Principle 4: Quantified Cost–Reliability Trade-off and Surrogate-Assisted Efficiency in Iterative Multi-Round Scientific Code Synthesis

**Definition:**
Iterative adversarial-and-Bayesian refinement inherently consumes more tokens and execution time than one-shot generation; a credible submission must quantify this cost rather than merely acknowledge it. Reviewers should look for iteration–performance curves showing where gains saturate (here, convergence around the fourth–fifth iteration), explicit reporting of budget growth relative to reliability gains (e.g., execution validity rising from ~40% to ~90%), and an application-level argument for why correctness outweighs marginal token cost in scientific domains where failed simulations are far costlier than extra inference calls. Equally important is the efficiency machinery: the Gaussian-Process surrogate that predicts candidate-code performance from embedding similarity before expensive execution must be fully specified — embedding model, kernel, acquisition function — and its filtering benefit evidenced. A framework achieving reliability only through unbounded brute-force iteration, without surrogate guidance or convergence analysis, offers limited practical value. Conversely, a well-argued cost–reliability frontier signals engineering maturity and honest limitation accounting, which the area chair explicitly credited in this case.

**Core Evaluation Criteria:**
- **Convergence Characterization**: Are iteration-wise performance curves provided on both general and scientific benchmarks, identifying a practical operating point balancing cost and accuracy?
- **Explicit Cost Accounting**: Does the work disclose token/computation overhead versus one-shot baselines and justify it via domain-specific reasoning about the cost of scientific errors?
- **Surrogate Transparency and Utility**: Are surrogate components (code embedding model, kernel choice, acquisition function such as UCB) fully specified, with evidence that surrogate-guided selection reduces execution cost without sacrificing quality?
- **Honest Limitation Discussion**: Are scalability implications (long pipelines, resource-intensive ML/DL evaluation, training-dynamics variance) acknowledged with concrete mitigation directions?

---

## Principle 5: Empirical Substantiation of Non-Expert Accessibility and Domain-Knowledge Fidelity in Low-Code AI4S Platforms

**Definition:**
A central claimed impact of low-code AI4S platforms is democratization: enabling domain scientists without prompt-engineering or coding expertise to obtain reliable, domain-faithful code. This claim must be tested empirically rather than asserted rhetorically. The strongest evidence is a controlled "with/without expert knowledge" comparison showing that the framework's internal planning and prompt-structuring modules narrow the performance gap caused by vague user inputs — ideally, a non-expert user with the framework should outperform an expert-crafted prompt on a raw baseline model, as demonstrated here. Reviewers should further require real-world, cross-disciplinary case studies (e.g., beach-profile prediction, brain-MRI segmentation) demonstrating qualitative incorporation of domain constraints — theoretical models, physically derived features, appropriate architectures and loss functions — alongside quantitative outcomes against tools practitioners actually use. Transparency of module internals (structured planning prompts, sample generated test cases, complete generated code) is essential for reproducibility and for judging whether the "refinement" is substantive rather than superficial paraphrasing. Works validating accessibility only anecdotally, or failing to show domain-law adherence in outputs, fall short of the AI4S promise.

**Core Evaluation Criteria:**
- **Controlled Prompt-Quality Robustness Test**: Is there a systematic vague-vs-expert prompt comparison quantifying how much the framework closes the gap across a spectrum of backbone models?
- **Domain-Knowledge Fidelity**: Do case studies verify that generated artifacts genuinely incorporate required domain constraints (physical formulas, theory-derived features, suitable losses), rather than merely producing runnable code?
- **Comparative Practitioner Relevance**: Are comparisons drawn against realistic practitioner tools (e.g., AI coding IDEs) under fair backbone settings, reporting both quality and efficiency outcomes?
- **Module-Level Transparency and Reproducibility**: Are internal prompts, generated test-case examples, full code outputs, and refinement logs disclosed so the human–AI collaboration mechanism can be inspected and replicated?