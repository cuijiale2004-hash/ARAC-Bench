## Principle 1: Honest Functional Analogy vs. Numerical Approximation When Mapping Formal RL Quantities (Advantage, Value Baseline, UCB Bonus) onto LLM Reasoning Components

**Definition:**
A hallmark design pattern in this subfield is replacing scalar RL quantities—advantages A(s,a), value baselines V(s), exploration bonuses—with LLM-generated *semantic counterparts* (textual advantage annotations, frontier trajectories as reference points). Reviewers and the AC explicitly identified the rigor of this mapping as the paper's most consequential and most fragile aspect ("the connection between the formal RL terms and their semantic counterparts, while currently a bit hand wavy, is very interesting"). High-quality work must clearly state whether it claims *numerical approximation* (a strong, usually unsupportable claim for text-generating mechanisms) or *functional analogy / theoretical motivation* (a defensible claim requiring empirical substantiation). Any formal proofs must genuinely correspond to the implemented procedure; when they do not (e.g., a variance-reduction proof about explicit averaging, while the method performs LLM comparison without averaging), the theory must be honestly reframed as design motivation. Crucially, the claimed functional role must be validated empirically—e.g., ablating the semantic advantage module against single-trajectory reflection, and sweeping the multi-path parameter n to show the predicted benefit of multi-trajectory comparison. This principle distinguishes conceptually deep work from work that merely decorates prompts with RL vocabulary.

**Core Evaluation Criteria:**
- **Precision of claim type:** Does the paper explicitly distinguish "approximation" from "analogy/motivation," and avoid language that misleadingly implies formal equivalence between textual outputs and scalar RL quantities?
- **Integrity of theoretical support:** Do formal results actually model the deployed mechanism, or are they clearly demarcated as motivating principles (e.g., relabeling "theoretical analysis" as "theoretical motivation")?
- **Empirical substantiation of the functional role:** Do ablations demonstrate the semantic component delivers its claimed RL-inspired benefit (e.g., MAR vs. Reflexion; n=1 vs. n=3 confirming variance-reduction-like gains)?
- **Consistency of narrative across main text and appendix:** Are mechanism descriptions (e.g., how "advantage identification" is actually performed via prompting) transparent in the main text rather than buried in appendices?

---

## Principle 2: Terminological Discipline When Repurposing Established Concepts such as "World Model" for Textual Experiential Memory in LLM Agents

**Definition:**
Borrowing loaded terminology from adjacent literatures creates strong reader expectations; in this case, "world model" canonically denotes a learned predictive dynamics model P(s′|s,a) in model-based RL, whereas the work builds structured, retrospective textual representations of experience. Three of four reviewers and the AC flagged this as confusing, and the AC explicitly recommended dropping or better justifying the term. Strong work in this subfield handles repurposed terms in one of three defensible ways: (a) grounds the broadened usage in credible, cited literature (e.g., surveys distinguishing implicit-representation world models from dynamics simulators; state-abstraction theory); (b) adopts a clarifying modifier early and consistently (e.g., "textual world model"); or (c) removes terminology that adds packaging but not conceptual content. The discriminative question is whether the term illuminates the mechanism's actual function or merely inflates the conceptual framing. Reviewers also assess whether the authors respond to terminological pushback with substantive positioning arguments rather than stubbornness.

**Core Evaluation Criteria:**
- **Alignment or explicit justification:** Is the term used consistently with its established technical meaning, or is the deviation justified with specific literature support, stated early in the paper?
- **Early and precise definition:** Are repurposed terms operationally defined (what the module stores, computes, and outputs) before being load-bearing in the method description?
- **Net conceptual value:** Does the terminology clarify the design (e.g., dual-scale memory fulfilling a world-model-like functional role in decision-making) or obscure the absence of the expected machinery (e.g., no transition prediction)?
- **Responsiveness to clarification:** Do the authors adopt disambiguating language (e.g., "textual world model") when confusion is identified?

---

## Principle 3: Backbone-LLM Robustness and Capability-Scaling Validation of Prompt-Driven Exploration Mechanisms

**Definition:**
Because every component of such frameworks (frontier analysis, state selection, advantage reflection, action policy) is mediated by an LLM, reviewers must determine whether the proposed mechanism is a *generalizable principle* or an artifact of one model's idiosyncrasies. Three reviewers independently raised this concern, and the requested remedy became a model rebuttal: targeted multi-model evaluation even on a subset of games, scaling to a stronger backbone to show the mechanism *benefits* from better reasoning, and—most distinctively—component-level analysis tracing how backbone capability affects the quality of intermediate artifacts (e.g., the specificity of generated potential-value targets in the global memory) and how those quality gaps propagate to downstream exploration performance. Equally important is prompt generality: using one fixed set of zero-shot prompts across all games and models rebuts the "hand-crafted prompt engineering" criticism. Sensitivity to LLM-interfacing design choices (trajectory truncation length, context degradation over long horizons) should also be addressed.

**Core Evaluation Criteria:**
- **Multi-backbone evaluation:** Is the framework tested with at least two model capability tiers (even on a game subset), demonstrating that improvements stem from the mechanism rather than a specific checkpoint?
- **Component-level quality analysis:** Does the work isolate and inspect how backbone capability changes the *intermediate representations* (e.g., achieved-value vs. potential-value inference), not just final scores?
- **Prompt generality and minimality:** Are identical, few, zero-shot prompts used across all environments and models, with prompt count and design disclosed?
- **Scaling behavior:** Does the method show monotonic benefit from stronger backbones (ideally surpassing compute-heavy baselines), and are failure cases of weaker backbones mechanistically explained?

---

## Principle 4: Joint Accounting of Environment Sample Efficiency and LLM Inference Cost in Exploration-Agent Evaluation

**Definition:**
Headline claims of "100–800× fewer environment interactions" are this subfield's central selling point, but reviewers explicitly criticized one-sided efficiency analysis that obscures total LLM calls. Since LLM-agent methods trade environment samples for massive inference compute, credible evaluation requires a *two-axis* efficiency ledger: environment steps (vs. RL/MCTS baselines) AND token consumption / API cost per run (vs. LLM baselines under identical step budgets). The rebuttal's cost tables—reporting input/output tokens and dollar costs for every LLM method under two backbone pricing tiers, and showing relative efficiency (e.g., ~40% fewer tokens than the strongest in-context baseline) alongside superior performance—exemplify the expected standard. Cost scaling when upgrading backbones (5–6× price increase) should also be disclosed so practitioners can assess the performance-per-dollar frontier. Work that reports only environment steps while hiding inference volume should be penalized as incomplete.

**Core Evaluation Criteria:**
- **Dual-axis reporting:** Are environment steps AND LLM token/API costs both quantified and tabulated?
- **Fair intra-category comparison:** Are costs compared across all LLM-based baselines under matched environment budgets and backbones?
- **Cost-performance frontier analysis:** Is the efficiency claim re-verified under stronger, pricier models rather than assumed invariant?
- **No obscured compute:** Does the paper avoid framing that celebrates sample efficiency while omitting the inference budget that purchased it?

---

## Principle 5: Hard-Exploration Benchmark Rigor — Justified Environment Selection, Fairly Strengthened Baselines, Complete Ablations, Statistical Care, and Failure-Mode Attribution

**Definition:**
Evaluation in this subfield is credible only if the testbed genuinely stresses hard exploration and the comparison is maximally fair. Reviewers demanded: (1) principled justification of the benchmark suite (combinatorial state-action scale, absence of explicit instructions, hidden causal dependencies, established comparability with prior RL/MCTS work) over simpler controlled alternatives; (2) baseline implementations strengthened to their best achievable form (e.g., the hybrid free-generation-plus-valid-action-soft-constraint interface that lifted ReAct/Reflexion from near-zero to competitive)—which paradoxically *strengthens* the paper's claims; (3) ablations isolating every module plus their synergy, and hyperparameter sweeps exposing design trade-offs (e.g., n controlling local-learning depth vs. global state-selection frequency); (4) statistical and reproducibility rigor (adequate seeds, standard-error vs. standard-deviation choices, disclosed determinism assumptions for action-replay state restoration, data-contamination checks, full prompt/code release); and (5) qualitative failure-mode attribution (e.g., conservative state selection loops; multi-step dependency reasoning failures) that explains *why* the system stalls, not merely that it does.

**Core Evaluation Criteria:**
- **Benchmark justification:** Is environment selection defended on complexity, instruction-freeness, and community comparability grounds, with explicit reasoning about excluded alternatives?
- **Baseline fairness and strength:** Are all baselines given the same action interface, budget, and backbone—and ideally implemented at their strongest, with headroom limits (e.g., 50-step episode sufficiency) justified?
- **Ablation and sensitivity completeness:** Is each component ablated individually and jointly, and are key hyperparameters swept to reveal global/local trade-offs and sweet spots?
- **Statistical and reproducibility hygiene:** Multiple seeds with appropriate error metrics, contamination analysis, disclosed environmental assumptions (determinism), and full release of prompts/code?
- **Mechanistic failure analysis:** Are qualitative failure cases attributed to specific mechanism deficiencies (selection bias, dependency-reasoning gaps) rather than reported as unexplained variance?