1. Research Background and Existing Pain Points
Open-ended generation tasks require outputs to satisfy diverse and often implicit task-specific evaluation rubrics. The sheer number of relevant rubrics leads to prohibitively high verification costs and incomplete assessments of a response, making reinforcement learning (RL) post-training with rubric-based rewards difficult to scale. This problem is exacerbated by the fact that often the best way to combine these rubrics into one single reward is also highly prompt-specific. Existing approaches face severe pain points:
1. Enumerative Verification: While enumerating the evaluation criteria and optimizing their aggregate is more faithful to the underlying rubrics, this strategy assumes the evaluation set can be exhaustively listed, which is unrealistic for complex tasks. Even when such enumeration is feasible, iterating over the entire set is computationally prohibitive, turning optimization into an intractable verification bottleneck. Furthermore, exhaustive fact extraction does not necessarily yield a cleaner reward signal; it can introduce overlapping or near-duplicate claims, each verified independently and contributing separately to the aggregated score, thereby overweighting a single underlying statement and amplifying noise.
2. Reward Models and RLHF: Training a single proxy reward model from offline human preference data is efficient, but this optimization is hard because the learned proxy is only as good as its coverage of the preference dataset. When the generator explores beyond this support, the proxy can misalign, often necessitating additional constraints like KL regularization to avoid collapse. These constraints stabilize training but limit exploration, making it difficult to scale to highly free-form generation tasks. This often leads to reward hacking, where the model overfits to the reward model and drafts without real quality gains.
3. Static LLM-as-a-Judge: LLMs prompted to serve as a judge or generative verifiers produce fixed judgments or explanations but do not learn adaptively from the generator’s evolving behaviors. They use the judge or verifier only as a static evaluator, failing to allocate verification effort where it is most informative.

2. Core Research Motivation and Scientific Questions
Core Research Motivation: To address the challenges of scaling RL post-training to free-form generation tasks where exhaustive rubric verification is infeasible and optimal reward design is unknown. The motivation is to provide rewards while avoiding the scalability limits of enumerative verification and the misalignment of static reward models. By recasting verification of a generator response as a dynamic process guided by a learned critic, verification becomes adaptive and adversarial, tailored to the generator’s current weaknesses.
Core Scientific Questions:
1. How can we train LLMs on free-form generation tasks with multiple (even uncountable) rubrics without exhaustively enumerating and verifying every rubric?
2. How can we dynamically combine task-specific rubrics into a reward signal that is prompt-specific, adversarially chosen, and always on-policy, while mitigating reward hacking?
3. How can we transform LLM-as-a-judge from a static scoring module into an active, on-policy agent that adaptively identifies failure modes and allocates verification effort where it is most informative?

3. Overall Core Idea and Design Philosophy
Overall Core Idea: Formulate the problem of training LLM generators on free-form tasks as an adversarial game between a generator and a critic. The critic is a learned model that proposes a rubric where the generator’s output is likely to fail, and an external validator verifies this. Both models are trained jointly. The critic is rewarded when it correctly pinpoints a rubric that the generator fails (verified by an external validator), while the generator is rewarded when the critic is unable to do so.
Design Philosophy: The core philosophy is dynamic rubric verification. Instead of collapsing all rubrics into a single scalar or a fixed multi-objective vector, RLAC learns a critic that dynamically proposes a verifiable rubric for each instance and grounds its supervision through an external validator. This yields a reward signal that is still scalar for RL optimization, but derived from an objectively checkable criterion rather than a static, unverified proxy, offering better alignment and reliability in open-ended tasks. It transforms LLM-as-a-judge from a static scoring module into an active, on-policy agent within an adversarial training loop.

4. Core Innovation Points
1. A novel post-training paradigm (RLAC) that frames free-form LLM optimization as an adversarial min-max game between a generator and a learned critic, with an external validator providing ground-truth feedback.
2. Elimination of the need for exhaustive rubric enumeration or manual reward design while providing adaptive learning signals that prevent reward hacking.
3. Introduction of a dynamic, adversarial feedback loop that yields prompt-specific, verifiable, and scalable supervision for free-form generation tasks.
4. Reformulation of the generator's objective to incorporate a critic policy that selects the worst-case criterion, bypassing the need to enumerate all criteria over the rubric set.
5. Joint training of both the generator and critic using DPO objectives with binary preference signals, where the critic adaptively identifies failure modes, the validator provides ground-truth feedback, and both are updated to improve over time.
6. Transformation of LLM-as-a-judge from a static scoring module into an active, on-policy agent that dynamically allocates verification effort where it is most informative based on the generator's evolving behavior.

5. Overview of the Overall Technical Solution
The overall technical solution, Reinforcement Learning with Adversarial Critic (RLAC), employs three task-agnostic components: a Generator, a Critic, and a Validator. 
The generator is an LLM fine-tuned to produce an output for an instruction. RLAC samples multiple response generations from the generator for each instruction. The critic is a pre-trained LLM that RLAC fine-tunes; for each instruction and a query generation output, the critic is prompted to generate a natural language output representing a rubric through auto-regressive decoding. The rubric along with the instruction and the generation are sent to the external validator to obtain a binary reward signal. 
The validator is an external tool or process that can verify whether a generated response satisfies a rubric provided as input. Neither the generator nor the critic ever observes the reference solution or any intermediate traces. 
At each training step, instructions are sampled and the generator produces K candidate outputs. For each instruction-output pair, the adversarial critic proposes a criterion, which is checked by the validator to yield a binary reward. This online feedback provides signals for both the generator and the critic. Outputs with reward 1 are treated as positives and those with reward 0 as negatives, and the generator is updated using the DPO objective. Similarly, for each instruction-output pair, N criteria are sampled from the critic. Criteria rejected by the validator (invalid or satisfied by the generator) are treated as negatives, while valid, unsatisfied ones are positives. The critic is then updated with its own DPO objective. In this way, evaluation and improvement are unified: the critic adaptively identifies failure modes, the validator provides ground-truth feedback, and both generator and critic are jointly updated to improve over time.

6. Detailed Module Design
Generator: The generator $\pi_g$ is an LLM that is fine-tuned to produce an output $a \in A$ for an instruction $s \in S$. RLAC samples multiple response generations from $\pi_g$ for each instruction $s$. We train $\pi_g$ to maximize the probability of producing outputs that satisfy all task-specific rubrics. The prompt for the generator includes specific formatting based on the domain (e.g., asking for exactly four sentences for biography generation).

Critic: The critic $\pi_c$ is a pre-trained LLM that RLAC fine-tunes. For each instruction $s$ and a query generation output $a$, the critic is prompted to generate a natural language output representing a rubric $c$ through auto-regressive decoding. The rubric $c$ along with the instruction $s$ and the generation $a$ are then sent to the external validator to obtain a reward signal $R(s, a, c) \in \{0, 1\}$. The prompt for the adversarial critic instructs it to identify the most clearly verifiable factual error (for text) or a failing test case (for code).

Validator: The validator is an external tool or process that can verify whether a generated response satisfies a rubric provided as input. Neither the generator nor the critic ever observes the reference solution or any intermediate traces.
- Factual Text Generation Validator: Follows a strict validation process. In the first stage, it uses textual entailment checking to verify that the proposed fact genuinely appears in the specified sentence. In the second stage, for proposals passing authenticity checks, it reuses FactScore’s atomic fact verification component, which queries the Wikipedia knowledge base to provide binary verification of individual factual claims.
- Code Generation Validator: Constructs reliable verification anchors by using Qwen2.5-Coder-7B-Instruct or GPT-4o to generate reference solutions. It filters these solutions using original test cases, retaining only those that pass. Given a critic-generated test case, it first executes it on the reference solution to check that the test case is valid and to obtain the expected output, and then executes it on the model-generated code to obtain the actual output. The validation reward is 1 if the outputs match and 0 otherwise, treating execution failures as detected errors.

Hyperparameters K and N: K controls how many candidate outputs are sampled for each prompt. A larger K increases candidate diversity and raises the probability that, for the same instruction, at least one candidate passes all critic checks while another fails at least one. The hyperparameter N specifies how many criteria or testcases the critic proposes for each candidate. A larger N expands the critic’s search space and enables it to discover more potential failure modes, but reduces the number of candidates that pass all checks and increases verification costs. For factual text generation, K=10 and N=4. For code generation, K=8 and N=2.

7. All Mathematical Formulas and Symbol Definitions
$S$: Distribution over prompts or instructions.
$s$: A specific instruction/prompt, $s \in S$.
$\pi_g(a | s)$: Generator LLM policy producing output $a \in A$ for instruction $s$.
$A$: The action space (set of all possible textual outputs).
$C(s)$: Set of rubrics associated with instruction $s$.
$c$: A specific rubric, $c \in C(s)$.
$R(s, a, c)$: Binary verification or reward function, returns 1 if output $a$ satisfies rubric $c$ on instruction $s$, and returns 0 otherwise.
$\pi^*_g$: Optimal generator policy.

Equation 1 (Objective to satisfy all rubrics):
$\pi^*_g := \text{argmax}_\pi \mathbb{E}_{s \sim S} \left[ \mathbb{E}_{a \sim \pi(\cdot|s)} \left[ \prod_{c \in C(s)} R(s, a, c) \right] \right]$

Equation 2 (Rewriting requirement as minimum over all rubrics):
$1\{R(s, a, c) = 1, \forall c \in C(s)\} = \min_{c \in C(s)} 1\{R(s, a, c) = 1\}$

Equation 3 (Substituting Equation 2 into Equation 1):
$\pi^*_g = \text{argmax}_\pi \mathbb{E}_{s \sim S} \left[ \mathbb{E}_{a \sim \pi(\cdot|s)} \left[ \min_{c \in C(s)} R(s, a, c) \right] \right]$

$\pi_c$: Critic policy, modeled as a stochastic policy that takes an instruction–generation pair $(s,a)$ as input and outputs a rubric $c \in C(s)$.

Equation 4 (Equivalent min-max form with critic):
$\pi_g = \text{argmax}_\pi \min_{\pi_c} \mathbb{E}_{s \sim S} \left[ \mathbb{E}_{a \sim \pi(\cdot|s)} \mathbb{E}_{c \sim \pi_c(\cdot|s,a)} [R(s, a, c)] \right]$

$\pi^g_{ref}$: Reference generator policy for DPO.
$a^+$: Positive generator output (reward 1).
$a^-$: Negative generator output (reward 0).
$\beta$: DPO regularization coefficient.
$\sigma$: Sigmoid function.

Equation 5 (Generator DPO loss):
$\mathcal{L}(\pi_g; \pi^g_{ref}) = -\mathbb{E}_{s, (a^+, a^-)} \left[ \log \sigma \left( \beta \log \frac{\pi_g(a^+|s)}{\pi^g_{ref}(a^+|s)} - \beta \log \frac{\pi_g(a^-|s)}{\pi^g_{ref}(a^-|s)} \right) \right]$

$\pi^c_{ref}$: Reference critic policy for DPO.
$c^+$: Positive critic criterion (valid, unsatisfied by generator).
$c^-$: Negative critic criterion (invalid or satisfied by generator).

Equation 6 (Critic DPO loss):
$\mathcal{L}(\pi_c; \pi^c_{ref}) = -\mathbb{E}_{s,a, (c^+, c^-)} \left[ \log \sigma \left( \beta \log \frac{\pi_c(c^+|s, a)}{\pi^c_{ref}(c^+|s, a)} - \beta \log \frac{\pi_c(c^-|s, a)}{\pi^c_{ref}(c^-|s, a)} \right) \right]$

8. Algorithm Pseudocode
Algorithm 1 RLAC
1: Initialize parameters $\pi_g, \pi_c, \pi^g_{ref}, \pi^c_{ref}$
2: for each iteration do
3:  ## Policy Evaluation for Generator $\pi_g$ .
4:  for each instruction s do
5:   Generate K generations $a_1, ..., a_K \sim \pi_g(\cdot|s)$
6:   Sample a criterion from the adversarial critic for each generation $c_i \sim \pi_c(\cdot|s, a_i)$.
7:   Construct a generator dataset $D^g_s = \{(s, a_i, R(s, a_i, c_i))\}^K_{i=1}$
8:  ## Policy Evaluation for Critic $\pi_c$. ▷ Optional
9:  for each instruction s, output a do
10:  Generate N criteria $c_1, ..., c_N \sim \pi_c(\cdot|s, a)$
11:  Construct a critic dataset $D^c_{(s,a)} = \{(s, a,R(s, a, c_j))\}^N_{j=1}$
12: ## Policy Improvement for Generator $\pi_g$ .
13: $\pi^{new}_g \leftarrow \pi_g$
14: for each update step do
15:  $\pi^{new}_g \leftarrow \pi^{new}_g - \nabla \mathcal{L}(\pi^{new}_g, \pi^g_{ref})$ ▷ Equation 5
16: $\pi^g_{ref} \leftarrow \pi_g$
17: ## Policy Improvement for Critic $\pi_c$. ▷ Optional
18: $\pi^{new}_c \leftarrow \pi_c$
19: for each update step do
20:  $\pi^{new}_c \leftarrow \pi^{new}_c - \nabla \mathcal{L}(\pi^{new}_c, \pi^c_{ref})$ ▷ Equation 6
21: $\pi^c_{ref} \leftarrow \pi_c$