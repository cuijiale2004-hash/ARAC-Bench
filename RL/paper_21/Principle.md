**Principle 1: Structured Decomposition and Granularity of Reward Signals Beyond Binary Verifiability in LLM Post-Training**

**Definition:**  
In domains where binary correctness is unavailable, the quality of post-training depends critically on whether reward signals capture nuanced, multi-dimensional judgments rather than coarse scalar preferences. This principle evaluates whether the proposed reward mechanism decomposes response quality into structured, interpretable, and instance-specific criteria—such as checklist-style rubrics—rather than relying on holistic Likert scores or pairwise preferences. It asks whether the criteria are grounded in domain expertise or high-quality proxies, whether the aggregation strategy into scalar rewards is principled, and whether the work provides evidence that the structural decomposition itself drives gains. The crucial distinction here is between genuine structured supervision and superficial prompt engineering disguised as reward design. Reviewers must assess whether the rubric construction process is reproducible, whether criteria cover essential versus optional dimensions, and whether the work avoids conflating the format of evaluation with the signal used for policy optimization. Ultimately, this principle ensures that reward signals are sufficiently granular and reliable to guide policy learning in complex, real-world reasoning tasks.

**Core Evaluation Criteria:**
- **Granularity and Decomposition:** Does the reward signal decompose quality into structured, instance-specific criteria rather than relying on a single holistic scalar? Are the criteria interpretable and actionable?
- **Grounding and Authenticity:** Are rubrics grounded in expert knowledge or high-quality proxies (e.g., reference answers)? Is there ablation comparing synthetic versus human-authored rubrics?
- **Aggregation Rationale:** Is the choice between explicit (weighted sum) and implicit (judge-mediated holistic) aggregation empirically and theoretically justified? Are limitations of static weights acknowledged?
- **Structural Signal vs. Prompt Engineering:** Does the work provide evidence that gains stem from the rubric structure itself rather than minor prompt modifications or increased context length?

---

**Principle 2: Statistical Rigor and Baseline Fairness in LLM-as-Judge Reinforcement Learning Experiments**

**Definition:**  
When reinforcement learning relies on LLM-based judges for reward computation, evaluation is susceptible to high variance, small-sample artifacts, and subtle implementation biases that can exaggerate or mask true performance differences. This principle demands rigorous statistical validation, including sufficient independent runs, properly reported confidence intervals, and careful control of evaluation protocols across compared methods. It emphasizes that all baselines must share identical training data distributions, judge models, prompt budgets, and checkpoint selection procedures to ensure fair comparison. Reviewers should scrutinize whether observed improvements survive increased statistical power and whether the authors transparently disclose their validation and checkpointing strategies. The principle also requires sensitivity analyses to demonstrate that results are not fragile to random initialization, judge model drift, or minor prompt variations. Without such rigor, claims of superiority in LLM-as-judge RL remain unconvincing and potentially misleading.

**Core Evaluation Criteria:**
- **Statistical Power and Variance:** Are experiments conducted with sufficient independent runs and sample sizes to produce tight confidence intervals? Do claimed improvements survive higher-powered evaluation?
- **Baseline Fairness and Control:** Are all methods trained on identical data with identical judge models and prompt budgets? Are baseline implementations tuned fairly rather than deliberately weakened?
- **Checkpoint and Validation Transparency:** Is checkpoint selection pre-specified and consistent across methods? Are training convergence curves or validation metrics provided?
- **Robustness to Judge Variability:** Are results replicated across multiple judge scales or model families to ensure the effect is not idiosyncratic to a specific judge?

---

**Principle 3: Cross-Domain Generalization and Evaluation-Format Robustness of Structured Reward Methods**

**Definition:**  
The practical utility of a structured reward method is determined not only by its performance on in-domain, rubric-based evaluations but also by its robustness across diverse task formats and domains. This principle evaluates whether a reward design trained with instance-specific structured criteria transfers effectively to qualitatively different evaluation settings—such as shifting from free-form medical dialogue to multiple-choice science questions—without overfitting to the training-time rubric structure. It requires evidence that the policy learns generalizable reasoning competencies rather than exploiting idiosyncrasies of the training reward signal. Reviewers must assess whether the paper scopes its claims appropriately, acknowledging limitations in highly subjective, creative, or agentic domains where rubric design is non-trivial. The principle also asks whether the work demonstrates format-independent benefits or merely optimizes for the specific benchmark on which it was trained.

**Core Evaluation Criteria:**
- **Evaluation-Format Transfer:** Does the method demonstrate robust performance when evaluated on output formats distinct from training (e.g., rubric-trained policies tested on multiple-choice or dialogue tasks)?
- **Domain Breadth and Scope:** Is evaluation conducted across substantively different domains? Does the paper avoid overgeneralizing from narrow, domain-specific benchmarks?
- **Resistance to Rubric Overfitting:** Is there evidence that the policy learns generalizable capabilities rather than exploiting specific phrasing or idiosyncrasies of the training rubrics?
- **Explicit Limitations:** Does the paper clearly scope its claims and acknowledge settings where the structured reward design may fail (e.g., highly creative, subjective, or multi-turn agentic tasks)?

---

**Principle 4: Computational Efficiency and Scalability Trade-offs in Instance-Specific Structured Reward Pipelines**

**Definition:**  
Instance-specific structured rewards introduce unique computational overhead relative to simple scalar rewards, including rubric generation costs, increased prompt lengths, and potential additional inference calls during training. This principle requires that papers explicitly quantify these overheads—distinguishing between one-time preprocessing expenses and per-rollout inference costs—and honestly compare them against the computational budgets of strong baselines. Reviewers should evaluate whether performance gains meaningfully justify the increased resource consumption or whether the method remains prohibitively expensive for large-scale deployment. The principle also asks whether the authors propose or evaluate optimizations, such as caching, model distillation, or batching, that could mitigate costs. A high-quality contribution must demonstrate that its structured reward pipeline is not only effective but also practically scalable beyond small-scale academic benchmarks.

**Core Evaluation Criteria:**
- **Overhead Quantification:** Are the additional computational costs—such as rubric generation tokens, increased prompt length, or extra judge calls—explicitly measured and reported relative to strong baselines?
- **Cost-Benefit Justification:** Do the empirical gains justify the increased resource expenditure? Is there analysis demonstrating practicality at scale?
- **Preprocessing versus Online Costs:** Does the paper clearly distinguish between one-time rubric synthesis costs and per-training-step inference overhead? Are both analyzed?
- **Scalability and Optimization:** Does the work discuss or evaluate strategies to reduce overhead, such as caching, distillation, or batched rubric evaluation, for large-scale deployment?

---

**Principle 5: Methodological Scope and Distinction Between Reward Engineering and Novel RL Algorithm Design**

**Definition:**  
Contributions at the intersection of reward design and reinforcement learning for LLMs often blur the line between novel training algorithms and carefully engineered reward functions within existing frameworks. This principle evaluates whether a paper clearly positions its contribution on this spectrum, avoiding overstatement of novelty by reframing prompt-based reward restructuring as a fundamentally new RL method. It demands that the work rigorously isolate the effect of its reward design within a standard, fixed RL pipeline—such as GRPO or PPO—without confounding algorithmic changes. Reviewers should assess whether the paper provides mechanistic or theoretical insight into why the reward structure improves learning dynamics, such as reduced variance or better alignment gradients, or whether it offers only an empirical recipe. The principle also requires precise articulation of the relationship to prior paradigms like RLVR and RLHF, ensuring that the distinction between rubric-conditioned evaluation and rubric-conditioned training is maintained.

**Core Evaluation Criteria:**
- **Novelty Positioning:** Does the paper clearly distinguish whether its contribution is a new RL algorithm, a new reward engineering scheme, or an empirical recipe? Are claims proportionate to the technical novelty?
- **Controlled Algorithmic Baseline:** Is the reward design evaluated within a fixed, standard RL framework with all hyperparameters held constant, isolating the reward effect?
- **Theoretical or Mechanistic Insight:** Does the work provide principled analysis of why the reward structure improves learning dynamics (e.g., variance reduction, credit assignment, alignment gradients)?
- **Disciplined Related-Work Positioning:** Does the paper precisely differentiate its contribution from RLVR, preference-based RLHF, and LLM-as-judge baselines, avoiding conflation of evaluation rubrics with training rewards?