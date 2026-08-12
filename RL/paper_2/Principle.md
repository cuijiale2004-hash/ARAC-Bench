**Principle 1: Mechanism Validation for Deep Safety Alignment Beyond Surface-Level Refusal Patterns**

**Definition:**  
Reviewers must critically assess whether claimed "deep alignment" reflects genuinely internalized safety awareness or merely superficial refusal heuristics memorized during training. In the context of large language model safety, it is insufficient to demonstrate reduced attack success rates alone; the evaluation must probe whether the model actively monitors and corrects its own generation trajectory when early-token cues are adversarially perturbed. This principle is crucial because many alignment methods optimize for easily measurable refusal keywords at the start of a response, creating brittle defenses that collapse under prefix-forcing or jailbreak attacks. Establishing deep alignment requires evidence that safety reasoning persists throughout the decoding process and remains causally connected to the final output, even when the model is prevented from emitting its typical refusal prefix.

**Core Evaluation Criteria:**
- **Adversarial Stress Testing for Alignment Depth:** Are evaluation protocols employed that perturb early-generation cues (e.g., prefilling attacks, forced-compliant prefixes) to test whether safety mechanisms persist beyond superficial token patterns?
- **Reasoning-Action Consistency:** Does the work provide evidence that explicit safety reasoning remains internally consistent with final outputs under adversarial conditions, rather than relying solely on aggregate refusal rates?
- **Causal Distinction from Shallow Baselines:** Is there explicit comparison with supervised fine-tuning or direct refusal methods demonstrating robustness to distribution shifts that bypass memorized prefixes?
- **Mechanistic Interpretability Evidence:** Are metrics or case studies presented that trace safety-related activations or token probabilities throughout the generation process to validate intrinsic awareness?

---

**Principle 2: Disentanglement of RL Contribution from Structured Reasoning Templates in Safety Alignment**

**Definition:**  
When a method employs mandatory structured reasoning templates (e.g., forcing explicit safety deliberation tags before an answer), reviewers must rigorously isolate the marginal contribution of reinforcement learning optimization from the elicitation effect of the template itself. Large language models often possess latent safety knowledge that can be partially unlocked through prompting or scaffolding alone; without proper ablation, performance gains may be incorrectly attributed to the RL algorithm rather than the output structure. This principle ensures that scientific credit is assigned correctly and that the community understands whether the proposed training framework provides reliable, consistent exploitation of latent capabilities beyond what zero-shot or few-shot templating achieves.

**Core Evaluation Criteria:**
- **Prompting-Only Baselines:** Are ablation studies included that quantify performance gains attributable solely to structured output templates or reasoning scaffolds without RL optimization?
- **RL Consistency Under Low-Sampling:** Does the work demonstrate that RL enables reliable safety performance in practical single-sample inference, beyond high-pass@k scenarios where templating alone might suffice?
- **Training Dynamics Analysis:** Are stepwise learning curves reported to show continued improvement from policy gradient updates after any initial elicitation effects plateau?
- **Robustness to Inference-Time Template Variation:** Does the method maintain safety performance when the structured reasoning scaffold is partially or fully removed during deployment?

---

**Principle 3: Rigor of Dual-Objective Reward Balancing for Safety-Utility Pareto Frontiers**

**Definition:**  
Methods that jointly optimize safety and utility through heterogeneous reward signals must be evaluated on the stability and empirical validity of their reward design. The safety-utility trade-off is a central challenge in alignment; naive combinations of refusal rewards and helpfulness scores often collapse into pathological behaviors such as over-refusal or reward hacking. Reviewers must scrutinize whether normalization, thresholding, or clipping mechanisms provably stabilize joint optimization, and whether the reported results represent genuine Pareto improvements rather than cherry-picked snapshots. This principle is essential to prevent methods that achieve safety gains only by silently degrading general capabilities or by becoming excessively conservative on benign inputs.

**Core Evaluation Criteria:**
- **Reward Signal Stability and Scaling:** Is the stability of heterogeneous reward signals empirically validated, including the impact of normalization or variance reduction mechanisms on training dynamics?
- **Multi-Dimensional Utility Preservation:** Are general capability metrics (reasoning, knowledge, instruction following) reported alongside safety metrics to detect covert utility degradation?
- **Nuanced Over-Refusal Evaluation:** Is over-refusal quantified on challenging benign datasets requiring fine-grained discrimination, rather than inferred solely from aggregate helpfulness scores?
- **Component Ablation and Interaction:** Are systematic ablations conducted removing each reward component to verify necessity and identify dominance or collapse modes in joint optimization?

---

**Principle 4: Empirical Scrutiny of Computational Cost and Inference Overhead in Lightweight RL Alignment**

**Definition:**  
Claims of "simplicity," "data efficiency," or "minimal training cost" in reinforcement learning-based safety alignment require rigorous quantification of both training expenditure and inference overhead. Because RL methods typically require multiple rollouts per sample, they can incur substantially higher GPU-hour costs than supervised alternatives, even when the step count is low. Similarly, mandatory reasoning steps at inference time may introduce token and latency overheads that affect deployability. This principle ensures that efficiency claims are holistic and transparent: reviewers must evaluate whether the method’s practical resource demands are proportionate to its safety benefits and whether overhead remains acceptable across both benign and adversarial query distributions.

**Core Evaluation Criteria:**
- **Absolute Training Cost Transparency:** Are computational costs reported in standardized units (e.g., GPU hours, rollout counts) and compared against supervised or distillation baselines under matched conditions?
- **Inference Overhead Quantification:** Is the latency and token-count impact of mandatory reasoning or formatting steps measured across both benign and adversarial workloads?
- **Cost-Effectiveness Contextualization:** Does the work relate computational expenditure to the magnitude and robustness of safety improvements, avoiding selective efficiency claims?
- **Scalability Verification Across Model Sizes:** Are efficiency claims validated across scales to rule out artifacts unique to small-model training regimes?

---

**Principle 5: Validity and Robustness of Verifiable Reward Signals Under Distribution Shift**

**Definition:**  
The integrity of reinforcement learning with verifiable rewards (RLVR) depends entirely on the accuracy, coverage, and robustness of the automated verifiers used to compute rewards. In safety alignment, rule-based or model-based verifiers can possess blind spots; an RL policy will aggressively exploit these blind spots, leading to reward hacking, format exploitation, or collapse into a narrow set of verifier-satisfying patterns. Reviewers must therefore evaluate whether verifier accuracy is audited on diverse refusal styles and adversarial outputs, and whether final evaluation relies on independent external judges rather than the same signals used for optimization. This principle is critical to ensure that observed gains reflect genuine safety improvement rather than overfitting to brittle, internally consistent verification logic.

**Core Evaluation Criteria:**
- **Verifier Accuracy and Coverage:** Is the accuracy of automated verifiers reported on diverse, held-out output distributions including edge-case refusals and adversarially generated text?
- **Independence of Evaluation Classifiers:** Does final evaluation rely on external safety judges independent of the training-time reward verifiers to avoid circular validation?
- **Diversity Preservation of Aligned Behaviors:** Does the work analyze whether policy optimization collapses output diversity to exploit verifier shortcuts, or maintains robust linguistic variation?
- **Reward Hacking Detection:** Are analyses or probes included to detect whether high rewards are achieved through format or keyword exploitation rather than genuine task completion?