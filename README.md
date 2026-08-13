# ARAC-Bench

**Benchmarking Auto-Research's Alignment and Completeness on End-to-End Researchs**

ARAC-Bench is a researcher-mimicking benchmark for evaluating autonomous research systems. Instead of judging only the final answer, ARAC-Bench measures how closely a system follows rigorous human research methodology across three independently evaluated stages: **Proposal**, **Experiment**, and **Synthesis**.

The benchmark combines stage-specific evaluation with **Academic Cognition Skills (ACS)**, a structured collection of research principles derived from papers, reviews, rebuttals, and discussions at major AI conferences. This design turns otherwise implicit reviewer expectations into traceable and quantifiable criteria.

## Highlights

- **Process-oriented evaluation:** assesses research methodology rather than final-answer similarity alone.
- **Three-stage diagnosis:** isolates Proposal, Experiment, and Synthesis capabilities under controlled conditions.
- **Stage-aware rubrics:** uses ACS to evaluate logical completeness, methodological reasoning, and causal analysis.
- **Traceable scoring:** links deductions to missing cognition skills, missing implementation modules, or broken causal reasoning.
- **Human validation:** the paper reports an average correlation of **0.8141** with rankings from Ph.D.-level evaluators.
- **Broad coverage:** ACS spans five major AI themes and 121 sub-themes.

## Benchmark Overview

| Stage | Points | What is evaluated |
| --- | ---: | --- |
| Proposal | 40 | Literature coverage, proposal quality, logical self-consistency, and benchmark selection |
| Experiment | 35 | Functional implementation, module completeness, code robustness, and hyperparameter choices |
| Synthesis | 25 | Structural completeness, factual fidelity, mechanism analysis, counterfactual reasoning, and conclusion boundaries |
| **Total** | **100** | Alignment and completeness across the end-to-end research process |

### Proposal Stage (40 points)

The system receives a domain background and an abstract inspiration with key implementation clues removed. It must conduct a literature survey and develop a proposal within a constrained time budget.

| Component | Points | Scoring basis |
| --- | ---: | --- |
| Related Work | 5 | Recall against the references cited by the Gold Reference |
| Proposal Details | 25 | ACS coverage and deductive self-consistency |
| Benchmark Selection | 10 | Selection rationale and successful dataset or environment access |

For Proposal Details, the evaluator retrieves the five most relevant ACS entries. Each entry is scored at `0`, `1.5`, or `3` points. The remaining 10 points assess consistency among notation, formulas, pseudocode, descriptions, and claims.

### Experiment Stage (35 points)

The system receives only the standard method description from the Gold Reference and must independently produce a runnable implementation.

| Component | Points | Scoring basis |
| --- | ---: | --- |
| Code Implementation | 30 | Functional correctness, module completeness, and robustness |
| Hyperparameter Setting | 5 | Agreement with domain priors and the Gold Reference, without a search budget |

The paper describes a standard module library containing **2,869 functional units**. A missing required module receives zero credit for that module.

### Synthesis Stage (25 points)

The system receives the complete result and ablation tables from the Gold Reference and produces an analytical research narrative.

| Component | Points | Scoring basis |
| --- | ---: | --- |
| Basic Analysis | 10 | Coverage and structure of the Introduction, Related Work, and Preliminaries |
| Methodological Deep Analysis | 15 | Mechanism depth, counterfactual reasoning, and conclusion extrapolation boundaries |

## Dataset Scope

The benchmark described in the paper is based on:

- structured annotations of **200 papers accepted at ICLR 2026**;
- ACS derived from **7,000 accepted papers** from NeurIPS, ICLR, and ICML in 2025 and 2026, together with associated review discussions and rebuttals;
- five major AI themes: Large Language Models, Multimodal Learning, Diffusion Models, Reinforcement Learning, and Deep Learning;
- 121 sub-themes and 2,869 standard implementation modules; 
- stage-level results for 11 autonomous research frameworks and research tools evaluated with a shared base model and evaluation protocol.

## Getting Started

1. Clone the repository.
2. Read the data-specific documentation and license terms.
3. Validate checksums and file counts using the release manifest, if provided.
4. Use the included scripts or schemas to load the benchmark artifacts.

Use the clone URL shown on the repository's GitHub page. Exact installation and evaluation commands will depend on the scripts included in the final release. Add tested commands here before publication; do not publish commands that have not been verified on a clean environment.

## Evaluation Protocol

For comparisons intended to follow the paper:

1. Evaluate only one research stage at a time.
2. Supply ground-truth outputs from earlier stages as fixed inputs to later stages.
3. Apply the same base model, tools, ACS library, resource limits, and prompts across compared systems.
4. Enforce the paper's literature-access cutoff at mid-2025 to prevent retrieval of the 2026 Gold References or their later citation networks.
5. Score the final stabilized output from each stage, including any self-corrections made within the allowed budget.
6. Report component scores as well as the total score so that failures remain diagnosable.

The Gold References and annotations may expose benchmark answers. Keep evaluation-only artifacts separated from public development inputs when measuring systems that could train on or retrieve this repository.

## Reported Results

The paper evaluates 11 systems under a shared Kimi-K2.6 base model. The strongest reported system obtains **67.9/100**, leaving a substantial gap from complete alignment with the benchmark's human-research criteria. These numbers should not be compared with runs that use different models, search cutoffs, prompts, tools, budgets, or benchmark versions.

## Intended Use

ARAC-Bench is intended for:

- evaluating autonomous research systems at distinct stages of the research workflow;
- diagnosing weaknesses in literature review, research design, implementation, and scientific synthesis;
- comparing systems under controlled model and resource settings; and
- studying process-level alignment between AI research agents and human research methodology.

ARAC-Bench is not intended to certify that a system can conduct safe, correct, or trustworthy research without human oversight. A benchmark score is not a substitute for expert review, replication, safety assessment, or domain-specific validation.

## Limitations

- The Gold References are drawn from selected AI conferences and do not represent all disciplines or research cultures.
- LLM-based scoring can inherit model-specific biases and may change with evaluator versions or prompts.
- The Experiment stage intentionally emphasizes implementation fidelity and parameter intuition rather than the full range of experimental-design and resource trade-offs considered by human researchers.
- Public release may create contamination risk for future evaluations.
- Conference metadata, reviews, rebuttals, paper text, and third-party code may be subject to separate licenses or platform terms.

