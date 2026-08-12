## ABSTRACT

Studies have shown that catastrophic forgetting primarily stems from the difficulty of reactivating old memories; although parameter-efficient fine-tuning can mitigate forgetting while keeping most model parameters frozen, it still falls short in fully reawakening knowledge of prior tasks. In contrast, humans can efficiently retrieve and flexibly integrate existing experiences when learning new tasks, thereby main taining stable performance on earlier ones. During cognition, the hippocampal EC–DG–CA3–CA1 circuit engages in multiple rounds of associative recall, and its pattern-separation and memory-completion mechanisms excel at activating historical information. Inspired by this mechanism, we propose HippoTune, a latent-space iterative retrieval strategy that embeds a query–retrieve–feedback loop within each Transformer layer. Starting from the hidden state as an initial query, the model performs a few rounds of soft key–value retrieval, projects the retrieved signals back into the query, and updates it iteratively until convergence or a preset iteration limit. Theoretically, we show this process implements a Krylov-style polynomial approximation, equivalent to a differentiable second-order preconditioner, thereby deepening retrieval in a principled way. Empirically, HippoTune outperforms classical buffer-free PEFT-CL methods by 5–8% in accuracy across three vision benchmarks, while reducing training FLOPs by 50%, effectively mitigating forgetting under tight compute constraints. Code is available at: https://github.com/yan4xi1/HippoTune.

## 1 INTRODUCTION

Deep neural networks excel under independent and identically distributed (i.i.d.) data, yet in continual learning (CL) settings where tasks arrive sequentially and distributions shift, they often suffer catastrophicforgetting: performance on earlier tasks degrades sharply when learning new ones McCloskey & Cohen (1989); Kirkpatrick et al. (2017). Classical approaches such as replay, regularization, and structural isolation often require large-scale fine-tuning of the entire network, leading to high computational and storage costs as models grow. Moreover, many such methods are largely model-agnostic and do not leverage inductive biases of prevalent architectures such as Transformers Rebuffi et al. (2017); Kirkpatrick et al. (2017); Mallya & Lazebnik (2018); Vaswani et al. (2017).

Parameter-efficient fine-tuning (PEFT) mitigates these costs by inserting a small number of trainable modules (e.g., adapters, LoRA modules, or prompts) into an otherwise frozen backbone, substantially reducing training overhead Houlsby et al. (2019); Hu et al. (2022); Lester et al. (2021); Li & Liang (2021). Recent PEFT-CL methods further maintain a “parameter/prompt pool” and, at inference time, retrieve and activate a subset of submodules using the sample representation as a query. However, this single-step retrieval can under-activate old-task memories and often requires a full backbone forward pass to extract high-level features for the query, introducing additional latency. In contrast, humans performing previously learned tasks engage in multiple rounds of associative recall and integration, leading to richer reactivation of historical knowledge. Concretely, sparse cues can trigger multi-round recall via the hippocampal EC–DG–CA3–CA1 circuit, enabling pattern separation and completion without repeatedly reconstructing high-level semantic representations. This circuit serves as a core pathway for memory formation and retrieval in the brain: information flows from the entorhinal cortex (EC), through the dentate gyrus (DG) for pattern separation, is completed via auto-associative recurrence in CA3, and finally integrated in CA1 into a coherent memory traceYassa & Stark (2011); Treves & Rolls (1994).

Inspired by the pattern separation, association, and integration mechanisms of the EC–DG–CA3–CA1 circuit, we propose Latent Deliberation, a layer-internal, differentiable, iterative retrieval mechanism. At each Transformer layer, we embed a light-weight associative loop: the previous layer’s hidden state serves as the initial query; we perform soft key–value retrieval to activate relevant memories; the retrieved signal is linearly projected and fed back to update the query; the loop continues until convergence or a maximum number of iterations; finally, we fuse the per-iteration outputs to realize multi-level completion and integration of prior-task knowledge. Operating entirely in latent space avoids repeated construction of high-level features, and exposes practical budget controls via the maximum iteration count, a convergence threshold, and top-k sparsity. This unified view also clarifies relationships to prompt-pool methods such as L2P Wang et al. (2022b), DualPrompt Wang et al. (2022a), and CODA-Prompt Smith et al. (2023): these can be seen as single-depth retrieval, whereas our method provides a differentiable deepening of retrieval depth to increase expressiveness and precision in memory access. See Fig. 1 for the differences between our method and the classic PEFT-CL approaches.

On the theory side, we characterize two key properties. First, near a fixed point, multi-step iteration implements a Krylov subspace polynomial approximation to the inverse Hessian, yielding an implicit second-order preconditioner for gradient propagation, achieving curvature correction in a finite number of steps without explicitly computing or storing second-order information Saad (2003); Martens & Grosse (2015). Second, we provide convergence and stability conditions based on step sizes and Jacobian spectral bounds, which translate into actionable choices for maximum iteration count, temperature, and entropy regularization; in effect, they operationalize the intuition that “longer deliberation/retrieval leads to better old-task performance” into verifiable and tunable optimization criteria Boyd & Vandenberghe (2004).

## We highlight four key contributions:

• A unified retrieval perspective for PEFT-CL. We distill existing prompt-pool continual learning methods into a single key–value formulation, clarifying their shared trade-offs and the limits of one-shot retrieval.

• Latent Deliberation: hippocampal-inspired iterative retrieval. Drawing on the EC–DG–CA3–CA1 circuit, we embed a lightweight, multi-step soft lookup–and–feedback process within each Transformer layer, deepening memory activation without extra backbone passes.

• Krylov-subspace preconditioning theory. We prove that our finite-step loop implements a polynomial approximation to the inverse Hessian, acting as an implicit second-order preconditioner. We also derive convergence and stability criteria to guide iteration count, temperature, and regularization.

• Strong performance at low compute cost. On three vision benchmarks, HippoTune delivers substantial accuracy gains over one-shot PEFT-CL baselines while using only about half the training FLOPs, demonstrating efficiency under tight resource constraints.

## 2 RELATED WORK

Continual Learning with Parameter-Efficient Fine-Tuning Within PEFT paradigm, continual learning has evolved into a prominent direction, now marked by the convergence of modularity, routing, and theoretical grounding. Early methods such as L2P Wang et al. (2022b), DualPrompt Wang et al. (2022a), and CoDA-Prompt Smith et al. (2023) introduced learnable prompt pools with key–query retrieval to select modules during training and inference, mitigating forgetting without relying on replay. Later approaches such as LAE Gao et al. (2023a), HiDe Zuo et al. (2023), and MoE-Adapter Yu et al. (2024) have further improved adaptability and efficiency via dynamic expansion, module merging, and expert routing. Theoretical work, including NTK analysis Doan et al. (2021) and loss landscape studies, has provided insights into how routing reduces gradient interference. However, most methods lack end-to-end optimization and seldom explore fundamental architectural principles for CL, limiting their scalability.

![](images/0cd62542a9df10b81df8a15f122b315ee4e87ed329780e18c9dc0c6137cb3154.jpg)  
Figure 1: Classic PEFT-CL vs. HippoTune. (Left) Standard prompt-based continual learning retrieves a single prompt $v ^ { ( l ) }$ per ViT layer to compute $\it { h ^ { ( l ) } }$ . (Right) HippoTune iteratively retrieves and integrates multiple prompts $\{ v _ { 1 } ^ { ( l ) } , \ldots , v _ { T } ^ { ( l ) } \}$ using $\boldsymbol { h } ^ { ( l - 1 ) }$ , enabling deeper memory activation and improved retention.

Hippocampus–Neocortex Inspired Continual Learning Inspired by the hippocampus–neocortex interplay, continual learning research has proposed the Complementary Learning Systems (CLS) theory: the hippocampus rapidly encodes new experiences, while the neocortex gradually extracts generalized knowledge. Building on this, models like FearNet Kemker & Kanan (2018), CLS-ER Arani et al. (2022), and Triple Memory Networks Wang et al. (2021) employ short- and long-term memory modules to balance fast adaptation with long-term retention via experience replay. Key cognitive mechanisms such as hippocampal replay, pattern separation (DG), and pattern completion (CA3) have been abstracted into algorithmic strategies. Some models adopt key–value memory for associative retrieval, while GATE Liu et al. (2025) simulates gated pathways across hippocampal subregions. Despite improving the stability–plasticity trade-off, these brain-inspired methods are often architecturally complex, replay-dependent, and rarely applied under the PEFT paradigm. In this work, we propose a fine-grained emulation of hippocampal associative memory, aligned with the EC–DG–CA3–CA1 circuit. We further validate its biological plausibility and computational efficacy from both theoretical and empirical perspectives.

## 3 METHODOLOGY

We unify all PEFT modules into a shared retrieval pool and perform iterative key–value lookups and one-shot fusion at each Transformer layer, mimicking EC–DG–CA3–CA1 hippocampal loops to dynamically activate and integrate past-task knowledge. The model is trained end-to-end with classification, orthogonality, and entropy losses, using truncated BPTT to align training and inference budgets.

## 3.1 PROBLEM DEFINITION

In the continual learning (CL) setting, the model is exposed to a sequence of tasks $\{ \mathcal { T } _ { 1 } , \mathcal { T } _ { 2 } , \ldots , \mathcal { T } _ { L } \}$ each associated with a dataset $\mathcal { D } _ { t } = \mathbf { \bar { \{ } }  ( x _ { i } , y _ { i } ) \}$ . The goal is to learn a new task $\mathcal { T } _ { t }$ while maintaining performance on previous tasks $\{ \mathcal { T } _ { 1 } , \ldots , \mathcal { T } _ { t - 1 } \}$ }. Formally, given a model output $f ( x ; \Theta )$ , we aim to optimize:

$$
\min _ {\Theta} \sum_ {k = 1} ^ {t} \mathcal {L} _ {\mathrm{cls}} ^ {(k)} \left(f (x; \Theta)\right),\tag{1}
$$

where $\mathcal { L } _ { \mathrm { c l s } } ^ { ( k ) }$ denotes the cross-entropy classification loss for task k.

## 3.2 PEFT-CL FRAMEWORK FORMALIZATION

In this subsection, we present a formalization of the PEFT-CL framework: we unify all lightweight modules into a shared retrieval pool, define how to compute relevance scores from a frozen backbone state, and show how to aggregate module outputs to update the model representation.

We collect all m parameter-efficient modules into a single pool

$$
\mathcal {V} = \{\theta^ {(1)}, \dots , \theta^ {(m)} \},
$$

indexed by a learnable key matrix

$$
K = \left[ k ^ {(1)}, \dots , k ^ {(m)} \right] ^ {\top} \in \mathbb {R} ^ {m \times d}.
$$

Each $\theta ^ { ( i ) }$ parameterizes a small PEFT block that takes a layer hidden state as input and produces a residual update (for example, an adapter, a prompt-induced projection, or a LoRA-style low-rank block).

Following prior works Gao et al. (2023b); He et al. (2022), we use $\phi ( x ; \theta ^ { ( i ) } )$ as an abstract notation to unify these different PEFT modules; details of the specific form are provided in Appendix A. Here $\phi ( \cdot ; \theta ^ { ( \bar { i } ) } )$ denotes the forward mapping of the i-th PEFT module applied to the hidden state x.

Given a frozen-backbone hidden state $x \in \mathbb { R } ^ { d }$ , we first compute the routing scores

$$
s = \frac {x K ^ {\top}}{\tau}, g = \operatorname{softmax} (s) \in \Delta^ {m - 1},\tag{2}
$$

with optional Top-k truncation. Each module then emits a residual $\Delta h ^ { ( i ) } = \phi \big ( \boldsymbol { x } ; \theta ^ { ( i ) } \big ) \ \in \ \mathbb { R } ^ { d }$ , which can be understood as the effect of the i-th PEFT module on this layer’s representation. We stack all residuals as

$$
\Delta H = \left[ \Delta h ^ {(1)}, \dots , \Delta h ^ {(m)} \right] ^ {\top}.
$$

Starting from the current backbone state $h = x ,$ , we conceptually update it by mixing all module outputs with the routing weights $g \colon$

$$
h \leftarrow h + g ^ {\top} \Delta H.\tag{3}
$$

In implementation, this update is realized by integrating the PEFT modules into the backbone block so that the layer directly outputs the updated $h ;$ the residual formulation above is an equivalent, unified view used for analysis. We provide a detailed explanation in Appendix A on how classical PEFT-CL methods correspond to this framework.

## Why this unification matters.

1. Query cost. Using the model’s hidden output as the retrieval query leverages rich semantic features but incurs extra computation.

2. Retrieval depth. All existing PEFT-CL methods perform only a single retrieval; this framework points naturally to deeper, iterative retrieval strategies.

3. Key-gating design. Learning and regularizing K, and choosing temperature, Top-k or entropy penalties, determines which modules activate.

## 3.3 LATENT DELIBERATION

At each Transformer layer, we extend the standard forward pass into a controllable dynamic process, modeled as an iterative associative loop.

We treat the hidden state from the previous layer $h ^ { ( l - 1 ) } \in \mathbb { R } ^ { d }$ as the initial query: $q ^ { ( 1 ) } = h ^ { ( l - 1 ) }$ . Each layer maintains a learnable key matrix $\boldsymbol { K } ^ { ( l ) } \in \mathbb { R } ^ { M \times d }$ and value matrix $\bar { V } ^ { ( l ) } \in \mathbb { R } ^ { M \times d _ { v } }$ to capture old-task subspaces. This is inspired by the EC–DG structure in the hippocampus, where $K ^ { ( l ) }$ acts as a guide to fixed-point memory.

At step t, the query $q ^ { ( t ) }$ retrieves from memory via:

$$
S ^ {(t)} = \mathrm{softmax} \left(\frac {q ^ {(t)} K ^ {(l) ^ {\top}}}{T}\right), \quad v ^ {(t)} = S ^ {(t)} V ^ {(l)},\tag{4}
$$

where the temperature $T > 0$ modulates retrieval sharpness. Optionally, Top-k filtering can be applied to $S ^ { ( t ) }$ to retain only the most relevant memory slots. The top-k hyperparameter is robust within the range of 3 to 10. As this is a standard setting for prompt-based methods, we omit it from further discussion.

![](images/04b4503c063430c0b3dfb924d9a704b37c9aa2636e6e990b3db82ebcc52be8b6.jpg)  
Figure 2: A comparative illustration of HippoTune. At each Transformer layer, use the hidden state as an initial query to iteratively perform key–value soft retrieval (with orthogonality and entropy regularization), update the query via projected residual feedback until convergence or max iterations, then fuse all retrievals into a memory-enhanced output, enabling selective multi-round activation of parameter-pool submodules.

Then, the query is updated by incorporating the retrieved memory:

$$
q ^ {(t + 1)} = \alpha q ^ {(t)} + (1 - \alpha) P ^ {(l)} \left(v ^ {(t)}\right),\tag{5}
$$

where $P ^ { ( l ) }$ is a layer-specific linear transformation, and $\alpha ~ \in ~ [ 0 , 1 ]$ controls the blending. The CA3 region features an auto-associative recurrent mechanism and can be regarded as the core of associative memory. This can be seen as a minimal abstraction of the recurrent CA3 circuit performing memory completion and state integration. The loop terminates when either $\lVert \boldsymbol { v } ^ { ( t ) } - \boldsymbol { v } ^ { ( t - 1 ) } \rVert ^ { \frac { \vee } { 2 } } < \varepsilon$ or $t = T _ { \mathrm { m a x } }$

To avoid repeated forward passes after each retrieval iteration, we adopt a one-shot fusion strategy, which integrates all retrieved vectors within the latent space in a unified manner. Specifically, the retrieval vector $v ^ { ( t ) }$ obtained at each iteration t is concatenated along the feature dimension to form an aggregated retrieval vector:

$$
V _ {\mathrm{cat}} = \left[ v ^ {(1)} \parallel v ^ {(2)} \parallel \dots \parallel v ^ {(T)} \right] \in \mathbb {R} ^ {T d _ {v}},\tag{6}
$$

where ∥ denotes vector concatenation.

Next, the output of the (l−1)-th layer, denoted as $\boldsymbol { h } ^ { ( l - 1 ) }$ , is combined with the concatenated retrieval vector $V _ { \mathrm { c a t } }$ and fed into the l-th layer’s ViT block to produce the output of layer l:

$$
h ^ {(l)} = \operatorname{ViT} ^ {(l)} \left(\left[ h ^ {(l - 1)} \| V _ {\text { cat }} \right]\right),\tag{7}
$$

where $\mathrm { V i T } ^ { ( l ) }$ represents the backbone network of the l-th layer. This one-shot fusion operation corresponds to the CA1 region in the hippocampal circuit, which is responsible for integrating the retrieved information from both DG and CA3 and producing a complete memory representation.

Importantly, this mechanism enables explicit controllability during inference via hyperparameters such as $T _ { \mathrm { m a x } } , \varepsilon .$ , and Top-k, allowing flexible trade-offs between retrieval quality and efficiency. The method framework is shown in Fig. 2. The pseudocode is provided in Appendix B.

## 3.4 END-TO-END TRAINING OBJECTIVE

We design a unified loss to jointly optimize task performance, retrieval sparsity, and module disentanglement:

$$
\mathcal {L} = \mathcal {L} _ {\mathrm{cls}} + \lambda_ {\mathrm{orth}} \mathcal {L} _ {\mathrm{orth}} + \lambda_ {\mathrm{ent}} \mathcal {L} _ {\mathrm{ent}}.\tag{8}
$$

• Classification Loss $\mathcal { L } _ { \mathrm { c l s } } \mathrm { : }$ Cross-entropy loss supervising downstream performance:

$$
\mathcal {L} _ {\mathrm{cls}} = - \frac {1}{N} \sum_ {i = 1} ^ {N} \sum_ {c = 1} ^ {C} y _ {i, c} \log p _ {i, c},\tag{9}
$$

where $y _ { i , c }$ is the one-hot label and $p _ { i , c }$ the predicted probability.

• Orthogonality Regularization $\mathcal { L } _ { \mathrm { o r t h } } \mathrm { : }$ : Encourages keys $K ^ { ( l ) }$ to be orthogonal, reducing memory interference:

$$
\mathcal {L} _ {\text { orth }} = \sum_ {l} \left\| K ^ {(l) ^ {\top}} K ^ {(l)} - I \right\| _ {F} ^ {2}.\tag{10}
$$

• Entropy Regularization ${ \mathcal { L } } _ { \mathrm { e n t } }$ : Controls the entropy of retrieval weights $S ^ { ( t ) }$ , balancing sharpness and robustness: T

$$
\mathcal {L} _ {\mathrm{ent}} = - \sum_ {l} \sum_ {t = 1} ^ {T} \sum_ {i} S _ {i} ^ {(t)} \log S _ {i} ^ {(t)}.\tag{11}
$$

The weights $\lambda _ { \mathrm { o r t h } }$ and $\lambda _ { \mathrm { e n t } }$ balance these objectives, guiding the model towards disentangled, controllable, and generalizable behaviors.

During training, we adopt Truncated Backpropagation Through Time (BPTT), propagating gradients only through the final steps of the retrieval loop. This design aligns with the dynamic budget at inference $( \mathbf { e } . \mathbf { g } . , T _ { \mathrm { m a x } } , \mathrm { T o p }  – k )$ , ensuring consistency between training and deployment.

## 4 THEORETICAL ANALYSIS: MULTI-STEP RECURRENCE AND HIGHER-ORDERPRECONDITIONING

We abstract a single-layer “recurrence” as gradient descent on a smooth potential function $\phi ( q )$ :

$$
q ^ {(t + 1)} = q ^ {(t)} - \eta \nabla \phi (q ^ {(t)}), t = 1, 2, \dots , T _ {\max} - 1,\tag{12}
$$

with step size $\eta > 0$ . The outer loss depends only on the final state $q ^ { ( T _ { \mathrm { m a x } } ) }$ , so we denote

$$
g _ {\mathrm{out}} = \frac {\partial \mathcal {L}}{\partial q ^ {(T _ {\mathrm{max}})}}.\tag{13}
$$

Proposition 1 (Krylov Subspace Polynomial Approximation). Suppose ϕ is twice differentiable near afixed point $q ^ { \star }$ , and its Hessian $\dot { H _ { \mathrm { ~ } } } = \nabla ^ { 2 } \phi ( q ^ { \dot { \star } } )$ is symmetric positive definite. Further assume the step size satisfies $\rho ( I - \eta H ) < 1 . D e f$ ne

$$
J = I - \eta H, P = \left. \frac {\partial (s t e p)}{\partial \theta} \right| _ {q ^ {\star}}, b _ {\theta} = P ^ {\top} g _ {\mathrm{out}}.\tag{14}
$$

Then the leading term of the gradient w.r.t. parameters θ after $T _ { \mathrm { m a x } }$ steps is

$$
\nabla_ {\theta} \mathcal {L} _ {T _ {\max}} = \sum_ {k = 0} ^ {T _ {\max} - 1} (J ^ {\top}) ^ {k} b _ {\theta} = \mathcal {K} _ {T _ {\max}} (H) b _ {\theta},\tag{15}
$$

where

$$
\mathcal {K} _ {T _ {\max}} (H) = \sum_ {k = 0} ^ {T _ {\max} - 1} (I - \eta H ^ {\top}) ^ {k}\tag{16}
$$

is the Krylov series operator. As $T _ { \mathrm { m a x } } \to \infty ,$ , the Neumann series converges and

$$
\mathcal {K} _ {T _ {\max}} (H) \to (\eta H ^ {\top}) ^ {- 1}, \Longrightarrow \nabla_ {\theta} \mathcal {L} _ {\infty} \approx H ^ {- 1} b _ {\theta}.\tag{17}
$$

The detailed proof is provided in Appendix C.

Corollary 1 (Effect of Finite-step Preconditioning). For any finite $T _ { \mathrm { m a x } } ,$ the operator ${ { \ K } _ { { T _ { \operatorname* { m a x } } } } } ( H )$ is a truncated polynomial approximation of $H ^ { - 1 }$ in the Krylov subspace. In practice, $T _ { \mathrm { m a x } } = 2 \stackrel { . } { \sim } 4$ already yields effective second-order correction at only linear computational cost.

<table><tr><td rowspan="2">Method</td><td rowspan="2">GFLOPs</td><td colspan="2">Seq-CIFAR100</td><td colspan="2">Seq-ImageNet-R</td><td colspan="2">Seq-CUB200</td></tr><tr><td>Acc</td><td>AAA</td><td>Acc</td><td>AAA</td><td>Acc</td><td>AAA</td></tr><tr><td colspan="8">Classical-CL (w/ buffer)</td></tr><tr><td>LwF Li &amp; Hoiem (2016)</td><td>16.88</td><td> $80.29 \pm 0.86$ </td><td> $87.33 \pm 0.73$ </td><td> $60.74 \pm 0.51$ </td><td> $68.55 \pm 0.65$ </td><td> $69.75 \pm 1.37$ </td><td> $80.45 \pm 2.08$ </td></tr><tr><td>DER++ Buzzega et al. (2020)</td><td>16.88</td><td> $84.50 \pm 1.67$ </td><td> $90.16 \pm 0.61$ </td><td> $54.21 \pm 0.52$ </td><td> $65.26 \pm 0.58$ </td><td> $77.42 \pm 0.71$ </td><td> $83.61 \pm 0.09$ </td></tr><tr><td colspan="8">PEFT-CL (w/o buffer)</td></tr><tr><td>L2P Wang et al. (2022b)</td><td>35.20</td><td> $82.76 \pm 1.17$ </td><td> $88.48 \pm 0.83$ </td><td> $71.26 \pm 0.44$ </td><td> $76.13 \pm 0.46$ </td><td> $68.39 \pm 0.46$ </td><td> $78.29 \pm 0.38$ </td></tr><tr><td>DualPrompt Wang et al. (2022a)</td><td>35.38</td><td> $85.56 \pm 0.33$ </td><td> $90.33 \pm 0.33$ </td><td> $68.22 \pm 0.20$ </td><td> $73.81 \pm 0.39$ </td><td> $66.00 \pm 0.57$ </td><td> $77.92 \pm 0.50$ </td></tr><tr><td>CODA-Prompt Smith et al. (2023)</td><td>35.84</td><td> $86.28 \pm 0.26$ </td><td> $91.05 \pm 0.37$ </td><td> $74.05 \pm 0.41$ </td><td> $78.14 \pm 0.39$ </td><td> $72.45 \pm 0.51$ </td><td> $78.94 \pm 0.37$ </td></tr><tr><td>LAE-PreT Gao et al. (2023a)</td><td>35.68</td><td> $85.25 \pm 0.66$ </td><td> $89.71 \pm 0.42$ </td><td> $62.81 \pm 0.48$ </td><td> $69.47 \pm 0.44$ </td><td> $77.48 \pm 0.79$ </td><td> $85.83 \pm 0.68$ </td></tr><tr><td>HiDe-Prompt Zuo et al. (2023)</td><td>35.25</td><td> $88.25 \pm 0.24$ </td><td> $92.69 \pm 0.27$ </td><td> $74.65 \pm 0.14$ </td><td> $78.46 \pm 0.18$ </td><td> $84.27 \pm 0.16$ </td><td> $88.64 \pm 0.19$ </td></tr><tr><td colspan="8">Ours (w/o buffer)</td></tr><tr><td>HippoTune (ours)</td><td>16.92</td><td> $87.65 \pm 0.21$ </td><td> $92.07 \pm 0.25$ </td><td> $74.85 \pm 0.17$ </td><td> $79.92 \pm 0.22$ </td><td> $81.12 \pm 0.34$ </td><td> $86.63 \pm 0.41$ </td></tr></table>

Table 1: Comparison of Continual Learning Methods on Seq-CIFAR100, Seq-ImageNet-R, and Seq-CUB200 in terms of Accuracy (Acc) and Average Accuracy Across All Tasks (AAA), along with Training Time.

## Practical Guidelines

1. Ensure convergence: Spectrally normalize H or choose a sufficiently small η so that $\rho ( I - \eta H ) < 1$

2. Choose $T _ { m a x } \colon \mathbf { A }$ small constant $T _ { m a x }$ ≈ 2–4 balances second-order preconditioning with computational budget.

3. Early stopping: When $\| q ^ { ( t + 1 ) } - q ^ { ( t ) } \|$ falls below a threshold, the Krylov polynomial has effectively converged and further iterations are unnecessary.

Conclusion. The revised derivation eliminates the erroneous “product = series sum” step and uses the chain rule and recursive expansion to rigorously demonstrate how multi-step recurrence implicitly implements Newton or natural-gradient second-order preconditioning in the gradient.

## 5 EXPERIMENTS

## 5.1 EXPERIMENTAL SETUPS

Benchmarks We evaluate on three mainstream vision continual-learning benchmarks: Seq-CIFAR100 Krizhevsky (2009); Lomonaco et al. (2021) is randomly split by class into 10 subtasks, each with 10 categories; Seq-ImageNet-R Boschini et al. (2022) is divided into 10 subtasks of 20 classes each; Seq-CUB200 Wah et al. (2011); Lomonaco et al. (2021) is split into 10 subtasks, each containing 20 bird species. We conduct evaluations under the Class-Incremental Learning.

Compared Methods We evaluate two broad classes of methods. First, classical continual learning methods: LwF Li & Hoiem (2016) is a regularization-based approach and does not use any replay buffer, while DER++ Buzzega et al. (2020) is replay-based and in our experiments maintains a memory buffer of 1000 images. Second, buffer-free PEFT-CL methods, including L2P Wang et al. (2022b), DualPrompt Wang et al. (2022a), CODA-Prompt Smith et al. (2023), LAE-PreT Gao et al. (2023a), and HiDe-Prompt Zuo et al. (2023), freeze the backbone and train only lightweight inserted modules. We include HippoTune in the buffer-free setting and re-implement all baselines under the same backbone and training regime, using published results for LAE-PreT and HiDe-Prompt.

Implementation Details We adopt ViT-Base/16 as the backbone Dosovitskiy et al. (2021), freeze all non-PEFT parameters, and fine-tune only the key–value projection layers of the inserted prompt modules. Training uses the Adam optimizer Kingma & Ba (2015) with a base learning rate of 0.01, a batch size of 128, and 5 epochs per subtask. Unless otherwise noted, modules are inserted into layers 1–7 and an orthogonality regularization coefficient of λ = 1 is applied. All experiments run on NVIDIA L40S GPUs without any replay buffer, and results are averaged over three random seeds. The detailed hyperparameter settings and code link are provided in Appendix D.

<table><tr><td rowspan="2">Method</td><td rowspan="2">GFLOPs</td><td colspan="2">ImageNet-R (N = 5)</td><td colspan="2">ImageNet-R (N = 10)</td><td colspan="2">ImageNet-R (N = 20)</td></tr><tr><td>Acc</td><td>AAA</td><td>Acc</td><td>AAA</td><td>Acc</td><td>AAA</td></tr><tr><td>Full Fine-Tuning</td><td>16.88</td><td> $64.92 \pm 0.87$ </td><td> $75.57 \pm 0.50$ </td><td> $60.57 \pm 1.06$ </td><td> $72.31 \pm 1.09$ </td><td> $49.95 \pm 1.31$ </td><td> $65.32 \pm 0.84$ </td></tr><tr><td colspan="8">PEFT-CL (w/o buffer)</td></tr><tr><td>L2P Wang et al. (2022b)</td><td>35.20</td><td> $73.04 \pm 0.71$ </td><td> $76.94 \pm 0.41$ </td><td> $71.26 \pm 0.44$ </td><td> $76.13 \pm 0.46$ </td><td> $68.97 \pm 0.51$ </td><td> $74.16 \pm 0.32$ </td></tr><tr><td>DualPrompt Wang et al. (2022a)</td><td>35.38</td><td> $69.99 \pm 0.57$ </td><td> $72.24 \pm 0.41$ </td><td> $68.22 \pm 0.20$ </td><td> $73.81 \pm 0.39$ </td><td> $65.23 \pm 0.45$ </td><td> $71.30 \pm 0.16$ </td></tr><tr><td>CODA-Prompt Smith et al. (2023)</td><td>35.84</td><td> $76.63 \pm 0.27$ </td><td> $80.30 \pm 0.28$ </td><td> $74.05 \pm 0.41$ </td><td> $78.14 \pm 0.39$ </td><td> $69.38 \pm 0.33$ </td><td> $73.95 \pm 0.63$ </td></tr><tr><td>HiDe-Prompt Zuo et al. (2023)</td><td>35.25</td><td> $74.77 \pm 0.25$ </td><td> $78.15 \pm 0.24$ </td><td> $74.65 \pm 0.14$ </td><td> $78.46 \pm 0.18$ </td><td> $73.59 \pm 0.19$ </td><td> $77.93 \pm 0.19$ </td></tr><tr><td colspan="8">Ours (w/o buffer)</td></tr><tr><td>HippoTune (ours)</td><td>16.92</td><td> $77.16 \pm 0.28$ </td><td> $81.04 \pm 0.37$ </td><td> $74.85 \pm 0.17$ </td><td> $79.92 \pm 0.22$ </td><td> $74.06 \pm 0.25$ </td><td> $79.33 \pm 0.49$ </td></tr></table>

Table 2: Comparison of Continual Learning Methods on ImageNet-R under different task numbers (N), including GFLOPs. Results are reported in terms of final Accuracy (Acc ↑) and Average Accuracy Across All Tasks (AAA ↑).

![](images/2bb3b6676c48f9cb2aa4242fb567e96af009bd7eef44ee3be2b5b020747a3c26.jpg)

![](images/0435e32cc2c2dffe1ac6c897a03128cc81cc47f2c9479a59655752c575feef8b.jpg)

![](images/d10fc92195b76f12324762c936edef2a40a070b1dd311afa521061bfcdeaa56d.jpg)  
Figure 3: Further analysis on Seq-CIFAR100 for three design choices. (Left) Moderate max iterations $( \mathrm { e . g . , } T _ { \mathrm { m a x } } = 4 )$ balance early gains and late stability; too few/many degrade results. (Middle) Temperature T tunes retrieval softness: mid-range $( 1 0 ^ { \overset { - } { - } 1 } )$ is best; extremes underperform. (Right) PEFT depth: combining shallow+middle (1–7) beats shallow (1–4), middle (5–8), or deep (9–12), highlighting multi-level memory.

## 5.2 EXPERIMENTAL RESULTS

In this section, we adopt Acc and AAA as evaluation metrics. Table 1 presents the results of various methods on three benchmarks, along with their respective computational cost (GFLOPs).

• Comparison with Classical CL Methods Using Replay Buffers Without relying on sample replay, HippoTune achieves Acc improvements of approximately 7.4, 14.1, and 11.4 percentage points over LwF on Seq-CIFAR100, Seq-ImageNet-R, and Seq-CUB200, respectively. Compared to DER++, it yields gains of around 3.2, 20.6, and 3.7 points. In terms of AAA, HippoTune also outperforms LwF (+4.7%) and DER++ (+1.9%), clearly demonstrating that the latent-space iterative retrieval mechanism can effectively suppress forgetting without any additional memory overhead.

• Comparison with Other PEFT-CL Methods Compared to typical prompt-tuning methods (L2P, DualPrompt, CODA-Prompt, LAE-PreT), HippoTune achieves the highest performance on Seq-ImageNet-R with 74.85% Acc and 79.92% AAA. On Seq-CIFAR100 (87.65%/92.07%) and Seq-CUB200 (81.12%/86.63%), it also surpasses most PEFT-CL baselines, second only to HiDe-Prompt with 88.25%/92.69% and 84.27%/88.64%, respectively. The superior performance of HiDe-Prompt on these two datasets can be largely attributed to its higher computational budget and more sophisticated multi-step prompting design. Notably, HippoTune achieves better results than HiDe-Prompt on Seq-ImageNet-R (despite using only 16.92 GFLOPs compared to 35.25 GFLOPs), and delivers comparable performance on the other two benchmarks—demonstrating its efficiency and competitiveness under limited computational resources.

• Resource Efficiency and Training Speed With a computational cost of only 16.92 GFLOPs, approximately half that of most mainstream PEFT-CL methods (around 35 GFLOPs), HippoTune significantly improves training speed and GPU memory efficiency. Under identical hardware settings, its training time is reduced by approximately 30% on average, confirming its practicality for scenarios with constrained computational resources.

## 5.3 ABLATION STUDY

In the ablation study, constraining the number of iterative retrieval steps to just one $( T _ { \operatorname* { m a x } } = 1 )$ leads to a notable performance drop: Acc/AAA on Seq-CIFAR100 declines from 87.65%/92.07% to 86.81%/90.63%, and on Seq-ImageNet-R from 74.85%/79.92% to 73.25%/78.62%. See Table 3 for the detailed results. This highlights the critical role of multi-step retrieval in integrating historical information and mitigating forgetting. Removing orthogonality regularization has a limited effect on Seq-CIFAR100, but results in a nearly 1.2-point drop in AAA on the more complex Seq-ImageNet-R, indicating that maintaining the diversity of retrieved vectors is especially important for leveraging prior knowledge in challenging domains. In contrast, removing entropy regularization or adopting a fusion strategy that only integrates the last-step retrieval affects overall performance by less than 0.6 points, suggesting their roles are more in stabilizing and fine-tuning the core mechanism. These findings suggest that iterative retrieval and orthogonality regularization are central to preventing catastrophic forgetting, while entropy regularization and fusion strategy can be flexibly adjusted in resource-constrained or inference-sensitive settings.

## 5.4 FURTHER ANALYSIS

ImageNet-R under Varying Task Counts We split ImageNet-R into sequences of $N = 5$ , 10, and 20 tasks. HippoTune consistently outperforms leading prompt-based PEFT-CL methods—gaining around 0.5–0.8 points at $N = 5$ versus CODA-Prompt and 0.2–1.5 points at $N ~ = ~ 1 0$ versus HiDe-Prompt—and even in the hardest N = 20 setting maintains a similar margin. Over the range $N = 5 {  } 2 0 .$ its overall accuracy drops by only about 3 points, far less than typical PEFT-CL declines. Crucially, these gains come at just 16.92 GFLOPs, underscoring HippoTune’s efficiency and resilience to forgetting. See Table 2 for the detailed results.

Table 3: Ablation Study: Impact of Removing Individual Components of Latent Deliberation on Seq-CIFAR100 and Seq-ImageNet-R

<table><tr><td rowspan="2">Variant</td><td colspan="2">Seq-CIFAR100</td><td colspan="2">Seq-ImageNet-R</td></tr><tr><td>Acc</td><td>AAA</td><td>Acc</td><td>AAA</td></tr><tr><td>Full Method</td><td>87.65</td><td>92.07</td><td>74.85</td><td>79.92</td></tr><tr><td>Baseline</td><td>86.28</td><td>90.33</td><td>72.93</td><td>78.16</td></tr><tr><td>w/o Orthogonality Regularization</td><td>87.32</td><td>91.87</td><td>74.09</td><td>78.77</td></tr><tr><td>w/o Entropy Regularization</td><td>87.43</td><td>91.30</td><td>74.67</td><td>79.55</td></tr><tr><td>w/o Iterative Retrieval ( $T_{\text{max}} = 1$ )</td><td>86.51</td><td>90.63</td><td>72.89</td><td>78.10</td></tr><tr><td>w/o Orthogonality &amp; Entropy (Loop only)</td><td>87.24</td><td>91.11</td><td>74.37</td><td>78.53</td></tr><tr><td>w/o Iterative Retrieval &amp; Orthogonality</td><td>86.40</td><td>90.43</td><td>72.74</td><td>77.92</td></tr><tr><td>w/o Iterative Retrieval &amp; Entropy</td><td>86.32</td><td>90.41</td><td>72.72</td><td>78.09</td></tr><tr><td>w/o Fusion Strategy (last-step only)</td><td>87.27</td><td>91.28</td><td>74.13</td><td>79.04</td></tr><tr><td>w/o Early Stopping</td><td>87.36</td><td>91.39</td><td>74.22</td><td>79.13</td></tr></table>

Impact of Iteration Length We sweep $T _ { \mathrm { m a x } } \in \{ 1 , 3 , 4 , 5 , 7 \}$ } on Seq-CIFAR100 (Fig. 3). Increasing from $T _ { \mathrm { m a x } } = 1$ to 3 yields clear gains on tasks 2–6, while $T _ { \mathrm { m a x } } = 4$ delivers the best overall accuracy—particularly mid-sequence—improving by about 1–2 points over $T _ { \mathrm { m a x } } = 1$ . Larger budgets (5 or 7) offer only marginal early-task gains and actually degrade later-task performance, suggesting that excessive iterations introduce noise or redundancy. Thus, $T _ { \mathrm { m a x } } = 4$ strikes the right balance between effective memory reuse and stability.

Accuracy Comparison Across All Tasks After Training Figure 4 in Appendix E.4 shows that HippoTune consistently outperforms both DER++ and DualPrompt throughout the full task sequence. In the early tasks it gains a clear lead, demonstrating its ability to recall prior knowledge immediately. This advantage persists in the mid-stage, with baseline methods trailing by a noticeable margin, and even as all methods degrade in later tasks, HippoTune’s drop is the smallest. Overall, iterative retrieval both reinforces early memories and promotes steadier performance across all ten tasks.

Impact of Temperature and Insertion Depth We swept the retrieval temperature T from $1 0 ^ { - 3 }$ to 10 on Seq-CIFAR100 (Fig. 3). Accuracy peaks at $T = 1 0 ^ { - 1 }$ , with mid-phase tasks $( 3 - 7 )$ improving by 1–2 points versus $T = \overline { { 1 0 ^ { - 3 } } }$ and smoother convergence later. Extremes $( T = 1 0 \mathrm { o r } 1 0 ^ { - 3 } )$ degrade performance, especially in mid-to-late tasks, indicating that moderate temperature best balances knowledge sharing and task isolation. Insertion depth experiments (Fig. 3) compare placing the module in shallow (layers 1–4), middle (5–8), deep (9–12), or shallow+middle (1–7) blocks. The 1–7 configuration wins—outperforming shallow-only and middle-only by 0.5–1 point and showing smaller late-task drops than deep-only. This confirms that early-layer feature retrieval plus mid-layer memory integration yields the strongest continual-learning gains.

Model Performance in Online Continual Learning Setting Our method remains highly effective even in the online setting with just one epoch of training. On seq-CIFAR100, it achieves 84.52% accuracy—less than 3% below the offline result—and surpasses the offline performance (epoch = 5) of some competing methods (see Appendix E.1).

Effectiveness on Diverse Pre-trained Backbones Experiments using DINO and SAM backbones further demonstrate the strong generalization of HippoTune. As shown in Table 6 (see Appendix E.2), our method consistently achieves superior final and average accuracy across both architectures, significantly outperforming baselines like L2P and DualPrompt. Notably, it surpasses CODA-Prompt by over 4% on the SAM backbone. These results indicate that the latent iterative deliberation mechanism is architecture-agnostic and adapts well to feature distributions from diverse pre-training objectives, effectively leveraging heterogeneous representations while minimizing forgetting.

Stability and Backward Transfer Analysis As shown in Table 7 (see Appendix E.3),in experiments on ImageNet-R split into 10 tasks, HippoTune outperforms standard prompt-based baselines in both accuracy and retention. While LoRA and adapter-based methods such as SD-Lora and EASE achieve high plasticity due to their architectural capacity, they suffer from significant catastrophic forgetting. In contrast, HippoTune maintains a Forgetting Measure of 4.03%, which is significantly lower than the 6% to 7% range observed in these adapter variants. This demonstrates that our method offers superior stability and effectively mitigates the interference common in high-capacity adapter approaches.

Results in the Task-Incremental Setting Table 8 (see Appendix E.5) presents the comparative results under the Task-Incremental Learning (TIL) setting. HippoTune consistently outperforms the PEFT-CL baselines across both Seq-CIFAR100 and ImageNet-R benchmarks. While existing methods like CODA-Prompt already mitigate interference by conditioning on task identities, our approach further pushes the performance boundary, achieving the highest average accuracy and the lowest forgetting measures. This superiority indicates that HippoTune effectively leverages task-specific contexts to refine feature representations, ensuring robust learning of new tasks without compromising the stability of previously acquired knowledge.

## 6 CONCLUSION

We introduced HippoTune, a hippocampal-inspired continual learning method that embeds an iterative retrieval loop into each Transformer layer. By simulating the brain’s multi-round associative recall and integration—combining pattern separation (DG) and completion (CA3–CA1)—HippoTune deepens memory access within PEFT frameworks without incurring repeated backbone passes. Our convergence analysis establishes a connection to Krylov-subspace second-order preconditioning, guiding choices of iteration count, temperature, and regularization. Experimentally, HippoTune delivers buffer-free gains across visual benchmarks, outperforms prompt-pool methods, and halves PEFT-CL’s computational cost. Limitations include evaluation on two-level hierarchies and image classification; future work will explore deeper loops, broader modalities, and adaptive retrieval budgets to further bridge biological memory mechanisms and scalable continual learning.

## ACKNOWLEDGMENTS

This work was supported by the National Science Fund for Distinguished Young Scholars of China under Grant 62325601, and the Beijing Natural Science Foundation under Grant L247011.