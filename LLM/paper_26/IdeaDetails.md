1. Research Background and Existing Pain Points
Deep neural networks excel under independent and identically distributed (i.i.d.) data, yet in continual learning (CL) settings where tasks arrive sequentially and distributions shift, they often suffer catastrophic forgetting: performance on earlier tasks degrades sharply when learning new ones. Classical approaches such as replay, regularization, and structural isolation often require large-scale fine-tuning of the entire network, leading to high computational and storage costs as models grow. Moreover, many such methods are largely model-agnostic and do not leverage inductive biases of prevalent architectures such as Transformers. 

Parameter-efficient fine-tuning (PEFT) mitigates these costs by inserting a small number of trainable modules (e.g., adapters, LoRA modules, or prompts) into an otherwise frozen backbone, substantially reducing training overhead. Recent PEFT-CL methods further maintain a “parameter/prompt pool” and, at inference time, retrieve and activate a subset of submodules using the sample representation as a query. However, this single-step retrieval can under-activate old-task memories and often requires a full backbone forward pass to extract high-level features for the query, introducing additional latency. Furthermore, studies have shown that catastrophic forgetting primarily stems from the difficulty of reactivating old memories; although parameter-efficient fine-tuning can mitigate forgetting while keeping most model parameters frozen, it still falls short in fully reawakening knowledge of prior tasks.

2. Core Research Motivation and Scientific Questions
In contrast to artificial networks, humans can efficiently retrieve and flexibly integrate existing experiences when learning new tasks, thereby maintaining stable performance on earlier ones. During cognition, the hippocampal EC–DG–CA3–CA1 circuit engages in multiple rounds of associative recall, and its pattern-separation and memory-completion mechanisms excel at activating historical information. Concretely, sparse cues can trigger multi-round recall via the hippocampal EC–DG–CA3–CA1 circuit, enabling pattern separation and completion without repeatedly reconstructing high-level semantic representations. This circuit serves as a core pathway for memory formation and retrieval in the brain: information flows from the entorhinal cortex (EC), through the dentate gyrus (DG) for pattern separation, is completed via auto-associative recurrence in CA3, and finally integrated in CA1 into a coherent memory trace.

The core scientific question is how to design a continual learning mechanism that mimics this biological multi-round associative recall and integration, embedding a query–retrieve–feedback loop within each Transformer layer to deepen memory activation without extra backbone passes, and how to theoretically justify and optimize this iterative retrieval process.

3. Overall Core Idea and Design Philosophy
Inspired by the pattern separation, association, and integration mechanisms of the EC–DG–CA3–CA1 circuit, the overall core idea is Latent Deliberation, a layer-internal, differentiable, iterative retrieval mechanism. At each Transformer layer, the standard forward pass is extended into a controllable dynamic process, modeled as an iterative associative loop. 

The design philosophy unifies all PEFT modules into a shared retrieval pool and performs iterative key–value lookups and one-shot fusion at each Transformer layer, mimicking EC–DG–CA3–CA1 hippocampal loops to dynamically activate and integrate past-task knowledge. Existing prompt-pool continual learning methods are distilled into a single key–value formulation, clarifying their shared trade-offs and the limits of one-shot retrieval. HippoTune provides a differentiable deepening of retrieval depth to increase expressiveness and precision in memory access. Operating entirely in latent space avoids repeated construction of high-level features, and exposes practical budget controls via the maximum iteration count, a convergence threshold, and top-k sparsity.

4. Core Innovation Points
• A unified retrieval perspective for PEFT-CL. We distill existing prompt-pool continual learning methods into a single key–value formulation, clarifying their shared trade-offs and the limits of one-shot retrieval.
• Latent Deliberation: hippocampal-inspired iterative retrieval. Drawing on the EC–DG–CA3–CA1 circuit, we embed a lightweight, multi-step soft lookup–and–feedback process within each Transformer layer, deepening memory activation without extra backbone passes.
• Krylov-subspace preconditioning theory. We prove that our finite-step loop implements a polynomial approximation to the inverse Hessian, acting as an implicit second-order preconditioner. We also derive convergence and stability criteria to guide iteration count, temperature, and regularization.
• Strong performance at low compute cost. On three vision benchmarks, HippoTune delivers substantial accuracy gains over one-shot PEFT-CL baselines while using only about half the training FLOPs, demonstrating efficiency under tight resource constraints.

5. Overview of the Overall Technical Solution
The overall technical solution unifies all PEFT modules into a shared retrieval pool and performs iterative key–value lookups and one-shot fusion at each Transformer layer, mimicking EC–DG–CA3–CA1 hippocampal loops to dynamically activate and integrate past-task knowledge. The model is trained end-to-end with classification, orthogonality, and entropy losses, using truncated BPTT to align training and inference budgets. At each Transformer layer, the previous layer's hidden state serves as the initial query; the model performs a few rounds of soft key–value retrieval, projects the retrieved signals back into the query, and updates it iteratively until convergence or a preset iteration limit. Finally, it fuses the per-iteration outputs to realize multi-level completion and integration of prior-task knowledge.

6. Detailed Module Design
6.1 PEFT-CL Framework Formalization
All m parameter-efficient modules are collected into a single pool V = {θ(1), . . . , θ(m)}, indexed by a learnable key matrix K = [k(1), . . . , k(m)]⊤ ∈ Rm×d. Each θ(i) parameterizes a small PEFT block that takes a layer hidden state as input and produces a residual update. Using ϕ(x; θ(i)) as an abstract notation to unify these different PEFT modules, where ϕ(·; θ(i)) denotes the forward mapping of the i-th PEFT module applied to the hidden state x. Given a frozen-backbone hidden state x ∈ Rd, routing scores are computed. Each module then emits a residual ∆h(i) = ϕ(x; θ(i)) ∈ Rd, which are stacked as ∆H = [∆h(1), . . . ,∆h(m)]⊤. Starting from the current backbone state h = x, it is conceptually updated by mixing all module outputs with the routing weights.

6.2 Latent Deliberation
At each Transformer layer, the standard forward pass is extended into a controllable dynamic process, modeled as an iterative associative loop.
• Initial Query Seeding (EC): The hidden state from the previous layer h(l−1) ∈ Rd is treated as the initial query: q(1) = h(l−1). Each layer maintains a learnable key matrix K(l) ∈ RM×d and value matrix V (l) ∈ RM×dv to capture old-task subspaces. This is inspired by the EC–DG structure in the hippocampus, where K(l) acts as a guide to fixed-point memory.
• Pattern separation (DG) and Retrieval: At step t, the query q(t) retrieves from memory via soft key-value lookup, with temperature T > 0 modulating retrieval sharpness. Optionally, Top-k filtering can be applied to retain only the most relevant memory slots.
• Recurrent Retrieval (CA3): The query is updated by incorporating the retrieved memory. P (l) is a layer-specific linear transformation, and α ∈ [0, 1] controls the blending. This can be seen as a minimal abstraction of the recurrent CA3 circuit performing memory completion and state integration. The loop terminates when either ∥v(t) − v(t−1)∥2 < ε or t = Tmax.
• Memory Integration Fusion (CA1): To avoid repeated forward passes after each retrieval iteration, a one-shot fusion strategy integrates all retrieved vectors within the latent space. The retrieval vector v(t) obtained at each iteration t is concatenated along the feature dimension to form an aggregated retrieval vector. Next, the output of the (l−1)-th layer, denoted as h(l−1), is combined with the concatenated retrieval vector Vcat and fed into the l-th layer’s ViT block to produce the output of layer l. This one-shot fusion operation corresponds to the CA1 region in the hippocampal circuit.

6.3 End-to-End Training Objective
A unified loss is designed to jointly optimize task performance, retrieval sparsity, and module disentanglement: L = Lcls + λorth Lorth + λent Lent .
• Classification Loss Lcls: Cross-entropy loss supervising downstream performance.
• Orthogonality Regularization Lorth: Encourages keys K(l) to be orthogonal, reducing memory interference.
• Entropy Regularization Lent: Controls the entropy of retrieval weights S(t), balancing sharpness and robustness.
During training, Truncated Backpropagation Through Time (BPTT) is adopted, propagating gradients only through the final steps of the retrieval loop.

6.4 Theoretical Analysis
A single-layer “recurrence” is abstracted as gradient descent on a smooth potential function ϕ(q). The outer loss depends only on the final state q(Tmax). Proposition 1 establishes that near a fixed point, multi-step iteration implements a Krylov subspace polynomial approximation to the inverse Hessian, yielding an implicit second-order preconditioner for gradient propagation. The detailed proof involves: Step 1: Recursion on sensitivities, differentiating the inner recurrence w.r.t. θ; Step 2: Chain rule for the outer loss; Step 3: Convergence to the Neumann–series inverse; Step 4: Finite-step interpretation. Practical guidelines include ensuring convergence by spectrally normalizing H or choosing a sufficiently small η, choosing Tmax ≈ 2–4 to balance second-order preconditioning with computational budget, and early stopping when ∥q(t+1) − q(t)∥ falls below a threshold.

7. All Mathematical Formulas and Symbol Definitions
Problem Definition:
min_Θ ∑_{k=1}^t L^{(k)}_{cls}(f(x; Θ))  (1)
where L^{(k)}_{cls} denotes the cross-entropy classification loss for task k.

Routing Scores:
s = xK^⊤ / τ , g = softmax(s) ∈ ∆^{m−1}  (2)

Conceptual Update:
h ← h + g^⊤ ∆H  (3)

Retrieval Step:
S^{(t)} = softmax(q^{(t)} K^{(l)⊤} / T) , v^{(t)} = S^{(t)} V^{(l)}  (4)

Query Update:
q^{(t+1)} = α q^{(t)} + (1− α)P^{(l)}(v^{(t)})  (5)

Concatenation:
V_{cat} = [ v^{(1)} ∥ v^{(2)} ∥ · · · ∥ v^{(T)} ] ∈ R^{T d_v}  (6)

Layer Output:
h^{(l)} = ViT^{(l)}([ h^{(l−1)} ∥ V_{cat} ])  (7)

Total Loss:
L = L_{cls} + λ_{orth} L_{orth} + λ_{ent} L_{ent}  (8)

Classification Loss:
L_{cls} = − (1/N) ∑_{i=1}^N ∑_{c=1}^C y_{i,c} log p_{i,c}  (9)

Orthogonality Regularization:
L_{orth} = ∑_l ∥ K^{(l)⊤} K^{(l)} − I ∥_F^2  (10)

Entropy Regularization:
L_{ent} = − ∑_l ∑_{t=1}^T ∑_i S^{(t)}_i log S^{(t)}_i  (11)

Abstract Gradient Descent:
q^{(t+1)} = q^{(t)} − η∇ϕ(q^{(t)}), t = 1, 2, . . . , T_{max} − 1  (12)

Outer Loss Gradient:
g_{out} = ∂L / ∂q^{(T_{max})}  (13)

Jacobians:
J = I − ηH, P = ∂(step)/∂θ |_{q⋆}, b_θ = P^⊤ g_{out}  (14)

Leading term of the gradient:
∇_θ L_{T_{max}} = ∑_{k=0}^{T_{max}−1} (J^⊤)^k b_θ = K_{T_{max}}(H) b_θ  (15)

Krylov series operator:
K_{T_{max}}(H) = ∑_{k=0}^{T_{max}−1} (I − ηH^⊤)^k  (16)

Neumann series convergence:
K_{T_{max}}(H)→ (η H^⊤)^{−1}, ⟹ ∇_θ L_∞ ≈ H^{−1} b_θ  (17)

Instantiation of ϕ for Prefix Tuning:
ϕ_{Prefix}(x; θ^{(i)}) = softmax( x W^{(i)}_q P^{(i)⊤}_k ) P^{(i)}_v  (18)

Instantiation of ϕ for Adapter:
ϕ_{Adapter}(x; θ^{(i)}) = ReLU( x W^{(i)}_{down} ) W^{(i)}_{up}  (19)

Instantiation of ϕ for LoRA:
ϕ_{LoRA}(x; θ^{(i)}) = x W^{(i)}_{down} W^{(i)}_{up}  (20)

8. Algorithm Pseudocode
Algorithm 1 Latent Deliberation: Iterative Retrieval and Integration
Require: Backbone depth L, max iterations T_{max}, tolerance ε, temperature T , blend factor α, (optional) Top-k
1: for l = 1 to L do
2:   Input: previous hidden state h^{(l−1)} ∈ R^d
3:   Initialize query: q^{(1)} ← h^{(l−1)}
4:   Initialize empty list V ← []
5:   for t = 1 to T_{max} do
6:     Compute retrieval weights: S^{(t)} ← softmax( q^{(t)} (K^{(l)})^⊤ / T )
7:     if Top-k enabled then
8:       Keep only top-k entries of S^{(t)}, zero out others
9:     end if
10:    Retrieve memory: v^{(t)} ← S^{(t)} V^{(l)}
11:    Append v^{(t)} to V
12:    Update query: q^{(t+1)} ← α q^{(t)} + (1− α)P^{(l)}(v^{(t)})
13:    if t > 1 and ∥v^{(t)} − v^{(t−1)}∥_2 < ε then
14:      break
15:    end if
16:  end for
17:  One-shot fusion: V_{cat} ← concat(V) ∈ R^{T_{max} d_v}
18:  Compute layer output: h^{(l)} ← ViT^{(l)}([ h^{(l−1)} ∥ V_{cat} ])
19: end for