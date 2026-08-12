### 1. Research Background and Existing Pain Points

**Research Background:**
Discrete diffusion language models (DLMs) generate text by iteratively denoising a discrete sequence over $T$ steps. A widely used family of DLMs operates through masking and unmasking, known as Masked Diffusion Language Models (MDLMs). Unlike autoregressive (AR) decoding—whose per-step compute naturally grows by one query token via a KV-cache—standard diffusion-style sampling repeatedly recomputes self-attention and per-token feed-forward sublayers for all $N$ token positions at every step. 

**Existing Pain Points:**
The standard MDLM sampler recomputes attention scores and feed-forward sublayers for every token position even for tokens that have already been unmasked and considered to be stable. This leads to substantial waste in compute. The per-block cost is dominated by computing the attention scores $QK^\top$ and applying them to $V$, i.e., $O(N^2d)$ where $d$ is the model dimension. Prior work has mainly progressed along two axes: 1) **Temporal approaches** shrink the step count (requiring fewer refinement steps); 2) **Reuse approaches** reduce per-step compute by reusing or approximating K/V vectors and partially updating hidden features across steps. These choices chiefly either reduce the step count $T$ or amortize work across steps by reusing intermediate states; they do not alter the within-step spatial granularity. Each step still issues $N$ query rows, so the attention-dominated cost remains $O(N^2d)$ even in late iterations.

### 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:**
Beyond reducing the step count $T$ or reusing K/V across steps, there is a largely orthogonal axis: permanently and monotonically deactivating token positions as the sampling unfolds. When many unmasked tokens are essentially fixed, recomputing their attention and feed-forward blocks is wasteful. The motivation is to answer a different question from prior work: what to remove from compute permanently, instead of what to compute now, so the active position set contracts over time monotonically.

**Scientific Questions:**
How to permanently and monotonically stop step-wise computation for converged token positions in Masked Diffusion-LM decoding while ensuring that other positions can still attend to them, and what theoretical justification can guarantee that this locking mechanism bounds the deviation in final token probabilities?

### 3. Overall Core Idea and Design Philosophy

**Overall Core Idea:**
SURELOCK proposes that once a token's posterior has stabilized across steps (the "sure condition"), the method locks that position—thereafter skipping its query projection and feed-forward sublayers—while caching its attention keys and values so other positions can continue to attend to it. This reduces the dominant per-iteration computational cost from $O(N^2d)$ to $O(MNd)$ where $M$ is the number of unlocked token positions. In practice, $M$ decreases as the iteration progresses, yielding substantial savings.

**Design Philosophy:**
The locking rule is driven primarily by local KL divergence, justifying that monitoring only the local KL at the lock step suffices to bound the deviation in final token probabilities. Once a token position is locked, it is excluded from subsequent re-masking and step-wise compute. The design is complementary to existing Temporal and Reuse approaches, as they can operate on the progressively shrinking active positions induced by SURELOCK.

### 4. Core Innovation Points

1.  **Orthogonal Optimization Axis:** Pursuing the permanent and monotonic deactivation of token positions as sampling unfolds, changing the within-step spatial granularity instead of just reducing step counts or reusing K/V across steps.
2.  **Permanent Stopping of Step-Wise Compute with K/V Caching:** Permanently eliminating converged positions from per-step compute by bypassing their query projection and per-token FFN at all subsequent steps, while caching their K/V vectors so that active token positions can still attend to them.
3.  **Step-wise KL Divergence Locking Criterion:** Proposing a locking rule based on local KL divergence ($D_t^{(i)} \le \varepsilon$) combined with an optional confidence gate, ensuring tokens are only locked when their posteriors have stabilized.
4.  **Theoretical Justification for Locking Criterion:** Deriving a closed-form bound (Theorem 1) that justifies the use of a local KL threshold as a locking criterion by proving it upper-bounds the terminal error of the log-probability in closed form ($\delta = C_{tail}\sqrt{\varepsilon}$).

### 5. Overview of the Overall Technical Solution

The overall technical solution considers MDLMs that iteratively denoise a length-$N$ sequence over steps $t=1, \ldots, T$. At step $t$, an $L$-block Transformer yields token-wise logits $z_t^{(i)}$ and posteriors $p_t^{(i)} = \text{softmax}(z_t^{(i)})$ at token position $i$, together with hidden states $h_t^{(i)} \in \mathbb{R}^d$. Let $M_t \subseteq [N]$ denote the masked positions and $\bar{M}_t = [N] \setminus M_t$ the unmasked ones. 

SURELOCK permanently stops step-wise compute for stable positions:
1.  **Active and Locked Sets:** Define the locked set $L_t$ and active set $A_t := \bar{M}_t \setminus L_t$.
2.  **Compute Flow:** At step $t$, assemble queries $Q[A_t]$ and run a variable-length attention kernel against the full key/value tables $K_{all}$ and $V_{all}$, where $K_{all}[L_t], V_{all}[L_t]$ are read from cache. Apply FFN sublayers only to active positions $\in A_t$.
3.  **Locking Evaluation:** Score active positions using step-wise KL divergence $D_t^{(i)}$ and uncertainty $u_t^{(i)}$. Lock positions satisfying $D_t^{(i)} \le \varepsilon$ and the confidence gate condition.
4.  **Update:** Cache K/V vectors for newly locked positions, freeze their block inputs, and update the locked set.

### 6. Detailed Module Design

#### 6.1 Permanently Stopping Step-Wise Compute and Caching K/V
Once a token $i$ is deemed converged at step $t^\star$, it is permanently eliminated from per-step compute. The method caches its K/V vectors and bypasses its query projection and per-token FFN at all subsequent steps. 
Let $L_t$ be the set of indices locked by step $t$, and define the active set $A_t := \bar{M}_t \setminus L_t$ (unmasked and not yet locked). At step $t$, the queries $Q[A_t]$ are assembled and run against the full key/value tables $K_{all}$ and $V_{all}$ where $K_{all}[L_t], V_{all}[L_t]$ are read from cache. This yields attention outputs only for active positions $\in A_t$, and FFN sublayers are applied only to them. For locked indices $i \in L_t$, predictions are kept fixed: $p_t^{(i)} \leftarrow p_{t^\star}^{(i)}$, and their FFN computation is skipped. Locked positions continue to be attended by other tokens via cached K/V.

#### 6.2 Criterion for Locking: Step-wise KL Divergence
**Primary: Local KL divergence.** For position $i$ at step $t$, the one-step divergence is defined as:
$D_t^{(i)} \triangleq \text{KL}\left(p_t^{(i)} \| p_{t-1}^{(i)}\right)$.

**Optional: Confidence gate.** Let uncertainty $u_t^{(i)} = 1 - \max_{v \in \mathcal{V}} p_t^{(i)}(v)$ and $q_m(u_t)$ the empirical $m$-th percentile of $\{u_t^{(j)} : j \in A_t\}$. When enabled, the gate accepts token position $i$ as a locking candidate, if $u_t^{(i)} \le q_m(u_t)$, i.e., top-$m$% confidence among active tokens.

**Locking rule.** We lock token position $i$ at step $t^\star$, if:
$D_{t^\star}^{(i)} \le \varepsilon$ and, if the confidence gate is enabled, $u_{t^\star}^{(i)} \le q_m(u_{t^\star})$.

#### 6.3 Design Justification
The design justifies the use of a local KL threshold as a locking criterion by proving that it upper-bounds the terminal error of the log-probability in closed form.

**Standing assumptions for Theorem 1:**
Let $z$ the logit and $f(z) = \log \text{softmax}(z)$. Fix constants $L > 0$, $L_{sm} > 0$, and $\rho \in (0, 1)$. For any position $i$ and step $s$, the following hold:
(A1) **Locking semantics.** Once $i$ is locked at $t^\star$, it is excluded from subsequent re-masking.
(A2) **Geometric tail contraction.** $D_s^{(i)} \le \rho D_{s-1}^{(i)}$ for $s > t^\star$.
(A3) **One-step logit smoothness.** $\|z_s^{(i)} - z_{s-1}^{(i)}\|_2 \le L\sqrt{D_{s-1}^{(i)}}$.
(A4) **Log-softmax Lipschitzness.** $\|f(z) - f(z')\|_\infty \le L_{sm}\|z - z'\|_2$.

**Theorem 1 (Locking error bound and closed-form threshold).** Fix a position $i$ that is unlocked up to $t^\star$ and then locked. Under (A1)–(A4), for any terminal time $T > t^\star$,
$\|\log p_T^{(i)} - \log \hat{p}_T^{(i)}\|_\infty \le C_{tail}\sqrt{D_{t^\star}^{(i)}}, \quad C_{tail} := L_{sm} L / (1 - \sqrt{\rho})$.
In particular, if the locking test enforces $D_{t^\star}^{(i)} \le \varepsilon$, then the terminal log-probability error is at most $\delta = C_{tail}\sqrt{\varepsilon}$, so the closed-form threshold is $\varepsilon(\delta) = \delta^2 / C_{tail}^2$.

**Proof:**
By (A1), locking token position $i$ at $t^\star$ permanently stops the $i$’s step-wise compute, so for all $t \ge t^\star$ we have $\hat{p}_t^{(i)} = p_{t^\star}^{(i)}$ and $\log \hat{p}_T^{(i)} = \log p_{t^\star}^{(i)}$. Therefore
$\|\log p_T^{(i)} - \log \hat{p}_T^{(i)}\|_\infty = \|\log p_T^{(i)} - \log p_{t^\star}^{(i)}\|_\infty = \|f(z_T^{(i)}) - f(z_{t^\star}^{(i)})\|_\infty$.
By the triangle inequality and telescoping over $s = t^\star+1, \ldots, T$,
$\|z_T^{(i)} - z_{t^\star}^{(i)}\|_2 \le \sum_{s>t^\star}^T \|z_s^{(i)} - z_{s-1}^{(i)}\|_2$.
Applying (A3) term-wise yields
$\|z_T^{(i)} - z_{t^\star}^{(i)}\|_2 \le L \sum_{s=t^\star+1}^T \sqrt{D_{s-1}^{(i)}}$.
Under (A2), $D_{s-1}^{(i)} \le \rho^{s-1-t^\star} D_{t^\star}^{(i)}$, therefore $\sqrt{D_{s-1}^{(i)}} \le \rho^{(s-1-t^\star)/2} \sqrt{D_{t^\star}^{(i)}}$ and $\sum_{s>t^\star} \sqrt{D_{s-1}^{(i)}} \le \frac{1}{1-\sqrt{\rho}} \sqrt{D_{t^\star}^{(i)}}$.
Finally, by (A4): $\|\log p_T^{(i)} - \log p_{t^\star}^{(i)}\|_\infty \le L_{sm}\|z_T^{(i)} - z_{t^\star}^{(i)}\|_2 \le C_{tail}\sqrt{D_{t^\star}^{(i)}}$.

#### 6.4 Computational Complexity: Algorithmic FLOPs
Let $N_{gen}$ the number of positions reserved for generation at initial step, $N_{prompt}$ the number of tokens for the prompt, $N_b := \max_{b \in B}(N_{prompt}) + N_{gen}$ be the sequence length of batch $b$; $S$ the number of sampling steps; $B$ the batch size; $d$ the model dimension; $L$ the number of layers; $H$ the number of attention heads; $d_h := d/H$ the head size; $H_{kv}$ the number of K/V heads; $d_{ff}$ the FFN dimension; $T_b := BN_b$ the number of token positions; $A_{t,b}$ the active index set at $t$; and $M_{t,b} := |A_{t,b}|$. We count algorithmic FLOPs as follows:

**Baseline.** For a step $t$, algorithmic FLOPs per batch are constant across $t$:
$F_{base, b}^t = L \left( \underbrace{4BHN_b^2d_h}_{QK^\top+AV} + \underbrace{2BN_bd^2}_{Q} + \underbrace{2BN_bd^2}_{Out} + \underbrace{4BN_bdH_{kv}d_h}_{K,V} + \underbrace{6BN_bdd_{ff}}_{FFN} \right)$.

**SureLock.** Algorithmic FLOPs exhibit temporal dynamics since step-wise computation is only for active positions $\in A_{t,b}$:
$F_{prop, b}^t = \frac{|A_{t,b}|}{BN_b} F_{base, b}^t = \frac{M_{t,b}}{T_b} F_{base, b}^t$.

### 7. All Mathematical Formulas and Symbol Definitions

*   $N$: Sequence length
*   $T$: Total number of sampling steps
*   $t$: Current sampling step ($1, \ldots, T$)
*   $L$: Number of Transformer layers
*   $d$: Model dimension
*   $z_t^{(i)}$: Logits at position $i$ after step $t$
*   $p_t^{(i)}$: Posterior at position $i$ after step $t$ ($\text{softmax}(z_t^{(i)})$)
*   $h_t^{(i)} \in \mathbb{R}^d$: Hidden states at position $i$ after step $t$
*   $M_t \subseteq [N]$: Masked positions (prediction targets) at step $t$
*   $\bar{M}_t = [N] \setminus M_t$: Unmasked positions at step $t$
*   $L_t$: Set of indices locked by step $t$
*   $A_t := \bar{M}_t \setminus L_t$: Active set (unmasked and not yet locked) at step $t$
*   $t^\star$: The step at which position $i$ is locked
*   $D_t^{(i)}$: Step-wise KL divergence for position $i$ at step $t$. Defined as: $D_t^{(i)} \triangleq \text{KL}\left(p_t^{(i)} \| p_{t-1}^{(i)}\right)$
*   $\varepsilon$: Threshold for local KL divergence locking criterion
*   $u_t^{(i)}$: Uncertainty for position $i$ at step $t$. Defined as: $u_t^{(i)} = 1 - \max_{v \in \mathcal{V}} p_t^{(i)}(v)$
*   $q_m(u_t)$: Empirical $m$-th percentile of $\{u_t^{(j)} : j \in A_t\}$
*   $f(z)$: Log-softmax function ($f(z) = \log \text{softmax}(z)$)
*   $\rho$: Geometric tail contraction constant ($\rho \in (0, 1)$)
*   $L$: One-step logit smoothness constant ($L > 0$)
*   $L_{sm}$: Log-softmax Lipschitzness constant ($L_{sm} > 0$)
*   $C_{tail}$: Bounding constant defined as $C_{tail} := L_{sm} L / (1 - \sqrt{\rho})$
*   $\delta$: Upper bound on terminal log-probability error ($\delta = C_{tail}\sqrt{\varepsilon}$)
*   $\varepsilon(\delta)$: Closed-form threshold ($\varepsilon(\delta) = \delta^2 / C_{tail}^2$)
*   $N_{gen}$: Number of positions reserved for generation
*   $N_{prompt}$: Number of tokens for the prompt
*   $N_b$: Sequence length of batch $b$ ($\max_{b \in B}(N_{prompt}) + N_{gen}$)
*   $S$: Number of sampling steps
*   $B$: Batch size
*   $H$: Number of attention heads
*   $d_h$: Head size ($d/H$)
*   $H_{kv}$: Number of K/V heads
*   $d_{ff}$: FFN dimension
*   $T_b$: Total number of token positions ($BN_b$)
*   $A_{t,b}$: Active index set at step $t$ in batch $b$
*   $M_{t,b}$: Number of active positions ($|A_{t,b}|$)
*   $F_{base, b}^t$: Algorithmic FLOPs for baseline at step $t$ for batch $b$
*   $F_{prop, b}^t$: Algorithmic FLOPs for SureLock at step $t$ for batch $b$

### 8. Algorithm Pseudocode

**Algorithm 1 SureLock**
Require: sequence length $N$, total steps $T$, confidence threshold percentile $m$, KL threshold $\varepsilon$
Require: vocabulary set $\mathcal{V}$, number of layers $L$, an unmasking policy $\text{UpdateMask}(\cdot)$
1: State: boolean $\text{lock} \in \{0, 1\}^N$ (init: all 0), caches $C$ for $(k, v)$ (init: None)
2: State: previous posteriors $p_{t-1}$ (init: None), masked indices $M_t$ (init: $[N]$)
3: State: frozen block-input $\hat{x} \in \mathbb{R}^{N \times d}$ (init: prompt embeddings)
4: Notation: $X_t \in \mathbb{R}^{N \times d}$ denotes the model input at $t$ (from embeddings or previous step output)
5: for $t = 1$ to $T$ do $\triangleright$ one diffusion step
6: $A_t \leftarrow \{i \mid \text{lock}_i = 0\}$, $L_t \leftarrow \{i \mid \text{lock}_i = 1\}$ $\triangleright$ active and locked position at $t$
7: Initialize block input $x^{(1)}[A_t] \leftarrow X_t[A_t]$, $x^{(1)}[L_t] \leftarrow \hat{x}[L_t]$
8: Compute on active positions $i \in A_t$:
9: for $\ell = 1$ to $L$
10: $Q[A_t], K[A_t], V[A_t] \leftarrow \text{Proj}_{Q,K,V}(\text{LN}(x^{(l)}[A_t]))$ $\triangleright$ main projections
11: $K_{all}[A_t] \leftarrow K[A_t]$, $K_{all}[L_t] \leftarrow C.k[L_t]$; same for $V_{all}$ $\triangleright$ assemble
12: $h^{(l)}[A_t] \leftarrow \text{FFN}(\text{Attn}(Q[A_t], K_{all}, V_{all}))$ $\triangleright$ attention with FFN
13: $x^{(l+1)}[A_t] \leftarrow h^{(l)}[A_t]$, $x^{(l+1)}[L_t] \leftarrow x^{(l)}[L_t]$ $\triangleright$ locked rows pass through
14: end for
15: Obtain posteriors $p_t[A_t] \leftarrow \text{softmax}(\text{Proj}_{out}(x^{(L+1)}[A_t]))$
16: Unmasking: $M_t \leftarrow \text{UpdateMask}(p_t, M_t)$, $\bar{M}_t \leftarrow [N] \setminus M_t$ $\triangleright$ unmask with posterior
17: Score on active positions $i \in A_t$:
18: $u_t^{(i)} \leftarrow 1 - \max_{v \in \mathcal{V}} p_t^{(i)}(v)$ for $i \in A_t$ $\triangleright$ confidence value
19: $D_t^{(i)} \leftarrow \text{KL}(p_t^{(i)} \| p_{t-1}^{(i)})$ if $t > 1$ else $\infty$ $\triangleright$ step-wise KL
20: Locking candidates: $Z_t \leftarrow A_t \cap \bar{M}_t$ $\triangleright$ must be active & unmasked
21: $\theta_m \leftarrow \text{Percentile}(\{u_t^{(j)}\}_{j \in Z_t}, m)$ $\triangleright$ threshold over candidate positions
22: Lock:
23: $F_t \leftarrow \{ i \in Z_t \mid u_t^{(i)} \le \theta_m \land D_t^{(i)} \le \varepsilon \}$ $\triangleright$ locking evaluation
24: $\text{lock}[F_t] \leftarrow 1$; $C_\ell.\{K,V\}[F_t] \leftarrow \{K_\ell, V_\ell\}[F_t] \forall \ell$ $\triangleright$ update locked indices and cache
25: $\hat{x}[F_t] \leftarrow x^{(1)}[F_t]$ $\triangleright$ freeze block-input for future steps
26: end for