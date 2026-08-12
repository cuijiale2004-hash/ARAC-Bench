1. Research Background and Existing Pain Points
The ascent of Large Language Model (LLM)-powered agents marks a paradigm shift across diverse domains. Pivotal to this success is the concept of agent memory, which enables LLM agents to learn progressively from environmental interactions, internalize experience, simulate human-like cognitive iteration, and progressively enhance problem-solving competence. The memory serving as this self-evolving engine typically manifests in two dominant paradigms, both of which exhibit significant limitations:
(I) Parametric Memory: This paradigm internalizes experiences by directly updating agents’ parameters (e.g., via SFT, GRPO, DPO). While this approach can yield substantial performance gains, its reliance on parameter modification inevitably entails catastrophic forgetting, i.e., the erosion of general knowledge.
(II) Retrieval-based Memory: This paradigm externalizes past experiences into a structured database, such as (i) raw trajectories, (ii) high-level experiences, and (iii) condensed skills like reusable APIs or MCP boxes. Although this non-invasive approach circumvents catastrophic forgetting, its efficacy is fundamentally tethered to context engineering. It adheres to a rigid execution pipeline, providing retrieved context to the agent without achieving the fluid, seamless integration characteristic of truly internalized memory. 
(III) Latent Memory (Existing): Existing approaches either use the key-value (KV) cache to maintain dynamic memory set (confined to addressing long-context issues) or latent token embeddings to store agent experiences (which still rely on invasive LLM parameter updates). More fundamentally, these methods diverge from human cognition in two critical dimensions: they lack the seamless interleaving of reasoning and memory, a process where thought and memory dynamically reshape one another, and remain largely retrieval-based, fetching memories by embedding similarity rather than generatively reconstructing them into novel, coherent insights.

2. Core Research Motivation and Scientific Questions
Given the deficiencies of existing memory paradigms, latent memory offers a compelling alternative, leveraging latent states as a machine-native, high-density medium for memory. Human reasoning and recollection operate in a continuous interplay, whereas most agent memory frameworks retrieve information only once at task initiation and append it to the query in a coarse and static manner. This leads to the pivotal research question that motivates this work: How can we architect agent memory as a dynamic cognitive faculty, capable of fluid, reconstructive processes that interweave seamlessly with reasoning?

3. Overall Core Idea and Design Philosophy
To address the challenge, MemGen is introduced as a dynamic and generative memory framework designed to endow any LLM agent with a more human-esque cognitive faculty. At its core, MemGen continuously monitors an agent’s cognitive state, enabling it to dynamically invoke a generative process that synthesizes a bespoke latent memory at any critical juncture during its reasoning process. The design philosophy is that memory should participate dynamically in the reasoning process, elevating reasoning from a linear unfolding to a recursive dialogue with memory. All experiential information is exclusively internalized into the parameters of the memory weaver, leaving the core reasoner entirely unmodified, thereby inherently mitigating catastrophic forgetting. Memory is not a verbatim restatement of prior content but a selective reconstruction, filtered and integrated through the weaver, akin to the memory consolidation process in the human brain.

4. Core Innovation Points
- Dynamic Interleaving of Reasoning and Memory: MemGen enables memory to participate dynamically in the reasoning process through an iterative cycle of generation, monitoring, invocation, weaving, and reintegration, transcending the rigid, static retrieval of existing paradigms.
- Metacognitive Memory Trigger: A reinforcement learning-trained memory trigger acts as a metacognitive monitor to discern the opportune moments for explicit memory invocation, balancing the need for critical latent memories with computational efficiency and reasoning integrity.
- Generative Memory Weaver: A memory weaver takes the agent’s current state as a stimulus to draw upon relevant implicit parametric memory (potentially augmented with externally retrieved information) and reconstructs this synthesis into a succinct, machine-native latent memory sequence, rather than merely retrieving verbatim past states.
- Non-invasive Parameter Update Mechanism: By freezing the core reasoner and updating only the lightweight memory trigger and weaver modules, MemGen inherently mitigates catastrophic forgetting when exposed to new data while equipping agents with a generative memory deeply integrated with reasoning.
- Emergent Human-like Memory Hierarchy: Without explicit supervision, MemGen spontaneously evolves distinct human-like memory faculties, including planning memory, procedural memory, and working memory, suggesting an emergent trajectory toward more naturalistic forms of machine cognition.

5. Overview of the Overall Technical Solution
MemGen consists of two synergistic components: a reinforcement learning-trained memory trigger and a memory weaver. The reasoning process in an agent equipped with MemGen unfolds autoregressively, driven by a frozen core LLM, the reasoner. For a given state, the reasoner generates the action token-by-token. MemGen continuously monitors this token-by-token generation process and performs on-demand memory insertion. At each token-generation step, the memory trigger monitors the reasoner’s internal cognitive state (hidden states) to determine if a memory invocation is necessary. If the decision is to SKIP, the reasoner proceeds with standard autoregressive generation. If the decision is to INVOKE, the reasoning process is momentarily paused. This summons the memory weaver, which takes the same cognitive state as a stimulus to perform a generative act of recollection, synthesizing a bespoke, machine-native latent memory sequence. This latent memory is woven seamlessly into the reasoner’s ongoing dynamics: its hidden states are prepended to the current context, upon which the reasoner resumes generation conditioned on this enriched context. The training procedure first trains the memory weaver using a random inserter as a surrogate for the trigger, and then freezes the trained weaver to train the trigger.

6. Detailed Module Design
- Reasoner (πθ): The frozen core LLM that generates action sequences autoregressively. It receives the environment state, previously generated tokens, and the inserted latent memory. It remains completely unmodified during training to preserve general capabilities.
- Memory Trigger (Ttrigger): Instantiated as a lightweight LoRA adapter attached to the reasoner. It receives the sequence of all hidden states and outputs an action probability. To avoid excessive computational overhead, a sentence-granularity activation strategy is adopted, where the trigger acts only when the current token falls in a delimiter token set (e.g., commas, periods). The trigger is trained via reinforcement learning with a reward-adaptive penalty to encourage sparse yet strategically critical memory invocation, maximizing task reward while maintaining computational efficiency.
- Memory Weaver (Wweaver): Instantiated as another LoRA adapter attached to the reasoner. Given the incoming hook (current hidden states), the weaver outputs a latent memory matrix. The synthesized latent memory is first projected through a linear layer to align it with the reasoner’s token-embedding space, and is then prepended to the current hidden states of the reasoner. The training of the weaver proceeds over a batch of past trajectories, internalizing experiential knowledge solely into the weaver. It is agnostic to optimization strategies and compatible with diverse LLM backbones, trainable under SFT or RL-based objectives (like GRPO) to optimize the generation process of latent memory to maximize downstream reward. Gradients from the reward are propagated solely to the weaver's parameters.
- Integration with Retrieval-based Memory: Although the memory generation primarily draws on the weaver’s parametric knowledge, it can be combined with external memory sources. When triggered, any retrieval-based system can provide textual memory, which is encoded into a sequence of embeddings, merged with the reasoner's hidden states, and fed into the weaver to produce latent memory, allowing the weaver to integrate internal knowledge and external information.

7. All Mathematical Formulas and Symbol Definitions
- Notation Definitions:
E: Environment
πθ: LLM agent parameterized by θ
x: Given task
τ: High-level trajectory (s0, a0, s1, a1, . . . , sT)
st: State of the environment at step t
at: High-level action taken by the agent at step t
zt,j: The j-th token in action at
H: History of past experiences H = {(xi, τi)}_{i=1}^N
M: Memory system
mt: Memory representation at timestep t
fM: Function for the generation of memory mt
πθ (Reasoner): Frozen core LLM
Ht,<j: Sequence of hidden state vectors (ht,1, . . . ,ht,j−1)
Ttrigger: Memory trigger module
pj: Invocation probability
dj: Binary decision (INVOKE/SKIP)
Wweaver: Memory weaver module
Mt: Latent memory matrix at step t ∈ R^{K×dmodel}
K: Fixed length of the latent memory sequence
D: Delimiter token set
ϕ: Trainable parameters of memory trigger
p̄: Mean activation probability across high-reward trajectories
θ′: Trainable LoRA parameters of memory weaver
Mext: External memory database
R(·): Retrieval function
Ct: Set of P retrieved textual snippets
Et: Encoded sequence of retrieved information embeddings ∈ R^{Lc×dmodel}
m̄i: Mean embedding of latent memory sequence Mi ∈ R^{dmodel}
C: Set of N discrete clusters
Sk(m̄new): Set of top-k nearest neighbors to m̄new

- Formulas:
Token generation conditioned on state and previous tokens:
zt,j ∼ πθ(· | st, zt,<j). (1)

Joint optimization objective for policy and memory system:
max_{θ,M} E_{x∼D, τ∼πθ,M} [R(τ)], (2)

Memory generation function:
mt = fM(st,H,m<t), (3)

Invocation probability computed by trigger:
pj = σ (Ttrigger(ht,1, . . . ,ht,j−1)) , (4)

Latent memory synthesis by weaver:
Mt := [mt,1,mt,2, · · · ,mt,K ] = Wweaver(Ht,<j), (5)

Token generation conditioned on state, previous tokens, and latent memory:
zt,j ∼ πθ(· | st, zt,<j ,Mt). (6)

Sentence-granularity activation strategy for trigger:
dj = Bernoulli(pj), pj =
\begin{cases} 
0 & \text{if } z_j / \in D,\\
T_{trigger}(H_{t,<j}) & \text{if } z_j \in D,
\end{cases} (7)

Trigger training objective with reward-adaptive penalty:
max_ϕ E_{τi∼πθ,d̃∼T^ϕ_trigger} [R(τi)− λ \sum_{i,j} max(0, d̃_{i,j} − p̄)], (8)

Computation of mean activation probability p̄:
p̄ = \frac{1}{|H_{high}|} \sum_{i\in H_{high}} \frac{1}{|\tau_i|} \sum_j d̃_{i,j} , H_{high} = \{i : R(τ_i) \geq median_k(R(τ_k))\}, (9)

Weaver training objective to maximize expected reward:
max_{θlora} E_{(x_i,\tau_i)\sim H} E_{\tau\sim\Pi^{W\theta', T}_\theta(\cdot|x_i)} [R(x_i, \tau)], (10)

SFT Loss function for weaver:
L_{SFT}(\theta') = -E_{(x_i,\tau^*_i)\sim H} \left[\sum_{t=0}^{T_i-1} \sum_{j=1}^{L_t} \log \pi_\theta(z^*_{i,t,j} | s_{i,t}, z^*_{i,t,<j}, M_{i,t,j})\right], (11)

Memory synthesis in SFT:
M_{i,t,j} = W_{\theta'}(H_{i,t,<j}). (12)

Gradient update for weaver in SFT:
\theta' \leftarrow \theta' - \eta \nabla_{\theta'} L_{SFT}(\theta'), (13)

Group-relative baseline for GRPO:
\bar{R}(G_i) = \frac{1}{K} \sum_{k=1}^K R(\tau_{i,k}). (14)

Advantage computation in GRPO:
A(\tau_{i,k}) = R(\tau_{i,k}) - \bar{R}(G_i). (15)

GRPO objective for weaver:
J_{GRPO}(\theta') = E_{x_i\sim H, G_i\sim\Pi^{W_{\theta'}, T}_\theta} \left[ \frac{1}{K} \sum_{k=1}^K A(\tau_{i,k}) \log \Pi^{W_{\theta'}, T}_\theta(\tau_{i,k} | x_i) - \beta KL(\Pi^{W_{\theta'}, T}_\theta(\cdot | x_i) \| \Pi_{ref}(\cdot | x_i)) \right], (16)

Trigger sampling action:
d_{t,j} \sim \pi_\phi(d | H_{t,<j}) := Bernoulli(p_{t,j}), p_{t,j} = \sigma(T^\phi_{trigger}(H_{t,<j})), (17)

High-reward subset for trigger penalty:
H_{high} = \{i : R(\tau_i) \geq median_k R(\tau_k)\}, (18)

Reference activation level for trigger penalty:
\bar{p} = \frac{1}{|H_{high}|} \sum_{i\in H_{high}} \frac{1}{|I_i|} \sum_{(t,j)\in I_i} d_{t,j}, (19)

Trigger training objective with sparsity penalty:
max_\phi E_{\tau_i\sim\pi_\theta, \tilde{d}_i\sim T^\phi_{trigger}} \left[ R(\tau_i) - \lambda \sum_{(t,j)\in I_i} max(0, d_{t,j} - \bar{p}) \right], (20)

Query decoding for external retrieval integration:
q_{t,j} = Decode(z_{t,<j}). (21)

Retrieval process from external memory:
C_t = R(q_{t,j} ; M_{ext}), (22)

Weaver invocation with external retrieval integration:
M_t = W_{weaver}([H_{t,<j} ; E_t]), (23)

Mean embedding of latent memory sequence:
\bar{m}_i = \frac{1}{K} \sum_{l=1}^K m_{i,l}. (24)

Mean embedding of new latent memory sequence during inference:
\bar{m}_{new} = \frac{1}{K} \sum_{l=1}^K m_{t,l} (25)

Condition for filtering target cluster during intervention:
\mu_j \in S_k(\bar{m}_{new}). (26)

8. Algorithm Pseudocode
The paper does not provide explicit algorithm pseudocode in standard format, but the algorithmic flow and iterative processes are detailed as follows based on the training and inference mechanisms:

Training Procedure:
1. Initialize reasoner πθ (frozen), memory weaver Wweaver (parameters θ′), and memory trigger Ttrigger (parameters ϕ).
2. Stage 1: Train Memory Weaver
   - Use a random inserter as a lightweight surrogate for the trigger.
   - For each batch of past trajectories H = {(xi, τi)}_{i=1}^N:
     - Randomly insert latent memory tokens at delimiter positions during rollout.
     - If using SFT:
       - Compute loss L_{SFT}(θ′) via Equation (11).
       - Update θ′ ← θ′ − η∇_{θ′} L_{SFT}(θ′) (Equation 13).
     - If using GRPO:
       - Generate a group of K trajectories for each task.
       - Compute advantage A(τ_{i,k}) via Equations (14) and (15).
       - Maximize J_{GRPO}(θ′) via Equation (16) updating only θ′.
3. Stage 2: Train Memory Trigger
   - Freeze the trained memory weaver Wweaver.
   - For each batch of trajectories:
     - Generate trajectories under policy πθ, Wweaver, and current Ttrigger.
     - At each decoding step where zt,j ∈ D, sample action d_{t,j} via Equation (17).
     - If invoked, pause and synthesize Mt using Wweaver.
     - Compute reward R(τi) and reference activation p̄ via Equations (18) and (19).
     - Update ϕ by maximizing the objective in Equation (20) using a GRPO-style surrogate.

Inference Procedure:
1. For a given state st, the frozen reasoner πθ generates tokens autoregressively.
2. At each token-generation step j:
   - If zt,j ∈ D:
     - Compute invocation probability pj and decision dj using Equations (4) and (7).
   - If dj = SKIP:
     - Generate next token via Equation (1).
   - If dj = INVOKE:
     - Pause reasoning.
     - Synthesize latent memory Mt using Equation (5) (and Equation 23 if using external retrieval).
     - Prepend Mt to the current hidden states.
     - Resume generation via Equation (6).
3. Continue until the action sequence is completed and executed in the environment.