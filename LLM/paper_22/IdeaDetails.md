Based on the complete academic paper provided, here is the full and detailed extraction of the research idea and implementation plan, strictly following the rules without omission or rewriting.

### 1. Research Background and Existing Pain Points
**Research Background:**
Reasoning is one of the central capabilities of large language models (LLMs), allowing models to tackle complex tasks such as mathematics, science, and programming far beyond simple question answering. A key limitation of the dominant reasoning approach, explicit chain-of-thought (CoT), lies in the reliance on discrete tokens during inference. In standard CoT decoding, the model commits to a single token at each step, sampled from the predicted distribution. While effective and ensuring readability by verbalizing intermediate steps, this discrete process collapses the full probability distribution into a single trajectory, discarding uncertainty and eliminating many potentially useful reasoning paths. Recent work has explored an alternative reasoning technique, latent reasoning, where the model operates directly in a continuous hidden space instead of a discrete text space. Latent reasoning offers two key advantages over CoT: 1) higher representational bandwidth per step, since hidden vectors can encode richer information than single tokens; and 2) the ability to preserve multiple reasoning hypotheses implicitly, rather than collapsing them prematurely into one tokenized path. Training-free approaches like Soft-Thinking operate directly at inference time without incurring additional training costs, making them cost-effective and resource-friendly for deployment.

**Existing Pain Points:**
Although training-free latent reasoning eliminates the need for costly retraining, operating purely in the latent space presents significant challenges:
1. **Accuracy Degradation:** The model is not explicitly trained to perform long-horizon reasoning with latent inputs. As a result of distributional mismatches, when inference relies solely on latent trajectories, the process is less controlled and can easily drift off course. Instead of collapsing into a single path, the model tends to spread probability mass across many implicit reasoning paths. While this preserves multiple hypotheses, it also introduces persistent noise, slows convergence, and ultimately harms reasoning accuracy.
2. **Overthinking and Inefficiency:** The absence of explicit tokens does not necessarily ensure efficiency. In latent space, models may still suffer from repetitive or unnecessarily extended internal deliberations and continuation, essentially overthinking. This prolongs inference and over-consumes tokens, undermining the very efficiency that latent reasoning is meant to improve.

### 2. Core Research Motivation and Scientific Questions
**Core Research Motivation:**
Remaining in a single mode throughout reasoning is inherently suboptimal: explicit thinking provides readability but may discard useful information beyond chosen tokens, while latent thinking preserves richer signals but can drift into noise and reduce accuracy. The key insight is that reasoning should switch modes based on confidence. Latent reasoning enables exploration across multiple potential continuations when confidence is low, and explicit reasoning encourages convergence when confidence is high, striking a balance that supports broad exploration while maintaining accuracy. Furthermore, the absence of explicit tokens does not ensure efficiency; thus, a mechanism is needed to suppress overthinking by utilizing partial reasoning trajectories for early answering.

**Scientific Questions:**
1. How to dynamically alternate between explicit and latent reasoning in a training-free manner to balance exploration and exploitation, thereby exploiting the expressivity of latent thinking without sacrificing the stability of explicit thinking?
2. How to suppress overthinking in reasoning LLMs to improve token efficiency under limited budgets without sacrificing final accuracy?

### 3. Overall Core Idea and Design Philosophy
**Overall Core Idea:**
This paper introduces SWIREASONING (SWIR), a training-free framework for LLM reasoning that alternates between explicit and latent thinking, based on block-wise confidence inferred from entropy trends of next-token distributions, and suppresses overthinking by bounding the number of switches.

**Design Philosophy:**
When block-wise uncertainty decreases (confidence rises), the framework collapses to a single explicit path to consolidate progress. When uncertainty rises (confidence drops) and has persisted for a minimal dwell window, it expands into latent space to explore more alternatives. Complementing this mode switch, a switch count controller caps the number of transitions, thereby curbing overthinking while preserving prediction quality by providing early-answer checkpoints based on partial reasoning trajectories.

### 4. Core Innovation Points
1. **Dynamic Mode Switching Mechanism:** SWIREASONING dynamically alternates between explicit and latent thinking based on confidence signals estimated from entropy trends of next-token distributions, thereby exploiting the expressivity of latent thinking without sacrificing the stability of explicit thinking.
2. **Switch Count Control Mechanism:** A switch count control mechanism caps the number of transitions, enabling early answering based on partial reasoning trajectories at switch boundaries. This effectively suppresses overthinking and improves token efficiency under limited budgets.
3. **Asymmetric Dwell Window Design:** The mode switch criterion imposes an asymmetric dwell window where WL→E = 0 and WE→L is positive. Latent reasoning is divergent, so immediate exit upon confidence recovery (WL→E = 0) reduces the risks of introducing spurious signals. Explicit reasoning is convergent, so a positive dwell window (WE→L) ensures it has sufficient opportunity to stabilize and accumulate meaningful structure before switching back.
4. **Thinking-Related Signal Mixing:** To better align mode switches with the LLMs’ learned reasoning patterns, the framework blends the embeddings of thinking-related signal tokens (ühle and _VEC) when a switch occurs, biasing the first latent step toward "begin thinking" and the first explicit step toward "end thinking."

### 5. Overview of the Overall Technical Solution
SWIREASONING operates as a training-free framework that dynamically alternates between explicit and latent reasoning. The number of switches is regulated to suppress overthinking and improve token efficiency. The framework first tracks a reference entropy H̄ within each thinking block to reflect block-wise confidence. Rising confidence (Ht < H̄) triggers an explicit switch to consolidate progress along a single path, while sustained uncertainty (Ht > H̄) triggers a latent switch to re-explore in continuous space. Dwell windows are imposed to avoid oscillations, with an asymmetric design reflecting the distinct roles of the two modes. Additionally, thinking-related signal mixing is applied at switching instants to align with learned reasoning patterns. To suppress overthinking, a switch count controller caps the number of Latent→Explicit switches. Two triggers are defined based on the user-specified budget Cmax: a convergence trigger that forces the next token to be _VEC when the switch count is between 1/2 Cmax and Cmax, and a termination trigger that injects a concise answer prefix when the count exceeds Cmax. These triggers are implemented as short injection queues that overwrite future-generated tokens.

### 6. Detailed Module Design

**Module 1: Preliminaries (Explicit and Training-Free Latent Thinking)**
*   **Explicit Thinking:** Let V be a vocabulary and pθ(xt | x<t) an LLM over V with parameters θ. Given a question q, the model produces a reasoning trace r1:T ∈ V T followed by a final answer a1:U ∈ V U. The concatenated sequence is written as x1:(|q|+T+U) = [ q, r1:T , a1:U ]. At inference, decoding proceeds by repeatedly choosing a token xt from the predictive distribution pθ(· | x<t) according to a policy πt(·). While effective, this hard policy collapses the full distribution to a single discrete decision at each step.
*   **Training-Free Latent Thinking:** It replaces the hard policy πt(·) by a continuous surrogate that preserves distributional information. Let E ∈ R|V |×d denote the token embedding matrix with rows e(v)∈Rd. At step t, the model yields logits ℓt ∈ R|V | and pt = softmax(ℓt). It forms a soft embedding ẽt and feeds ẽt back to the model as the next input representation, rather than committing to an explicit token by πt(·). The convexity ensures ẽt lies in the embedding hull of E, retaining all first-order uncertainty in pt.

**Module 2: Dynamic Switch Between Explicit and Latent Thinking**
*   **Mode Switch Criterion:** The reasoning content between two consecutive switches is referred to as a thinking block, and its confidence is estimated by entropy Ht=−∑v pt[v] log pt[v]. Let H̄ denote the reference entropy of the current block, which is initialized at the first step of the block and refreshed when a mode switch happens.
    *   Latent→Explicit: (Ht < H̄) (confidence rises)
    *   Explicit→Latent: (Ht > H̄) (confidence drops)
*   **Switch Window Size:** To avoid oscillations, dwell windows are imposed upon the mode switch criterion. Formally, with mode variable mt ∈ {Explicit,Latent} and dwell step counter ∆t:
    mt+1 = { Explicit, mt = Latent ∧ (Ht < H̄) ∧ (∆t ≥WL→E), Latent, mt = Explicit ∧ (Ht > H̄) ∧ (∆t ≥WE→L), mt, otherwise. }
    H̄←Ht, ∆t←0 upon any switch. Otherwise, ∆t←∆t+1. In practice, WL→E = 0 while WE→L is positive.
*   **Thinking-Related Signal Mixing:** Let e_VEC and e_VEC denote the embeddings of ühle and _VEC. At the entrance to a latent thinking block, the first latent step t⋆ is biased toward “begin thinking”. At the exit to an explicit thinking block, the first explicit step t† is biased toward “end thinking” to encourage the model to close the latent phase and move on to answer production. The schedules for αt and βt are predefined with α0 and β0 as initial ratios.

**Module 3: Overthinking Suppression by Switch Count Control**
*   **Counter and Triggers:** Let Ct count completed Latent→Explicit switches up to step t. Given a user-specified budget Cmax, two triggers are defined:
    *   **Convergence trigger** (at 1/2 Cmax ≤ Ct ≤ Cmax on Latent→Explicit transitions): force the next token to be _VEC. This is to encourage rather than enforce the end of the thinking process.
    *   **Termination trigger** (at Ct > Cmax on a subsequent Latent → Explicit transition): inject a concise answer prefix sfinal, “_VEC\n\n The final answer is”, then allow at most B additional tokens for the final answer. This enforces immediate answer generation.
*   **Injection Queue:** Triggers are implemented as short injection queues that overwrite future-generated tokens. Let Qt be the per-sample injection queue. When a trigger fires, Qt is set accordingly. At the next step, if Qt ̸= ∅, xt is deterministically set to Qt.pop(). For the termination trigger, a budget counter bt=B is started and decremented each step, terminating decoding once bt=0.

### 7. All Mathematical Formulas and Symbol Definitions
*   **Concatenated Sequence:** x1:(|q|+T+U) = [ q, r1:T , a1:U ]
*   **Explicit Policy:** xt ∼ πt(·) with πt = { Greedy: argmaxv∈V pθ(v | x<t), Sampling: Top-k/Top-p with temperature τ. }
*   **Soft Embedding:** ẽt = ∑v∈V pt[v] e(v) ∈ Rd
*   **Entropy:** Ht=−∑v pt[v] log pt[v]
*   **Mode Switch Criterion:**
    Latent→Explicit : (Ht < H̄) (confidence rises)
    Explicit→Latent : (Ht > H̄) (confidence drops)
*   **Switch Window Size Logic:** mt+1 = { Explicit, mt = Latent ∧ (Ht < H̄) ∧ (∆t ≥WL→E), Latent, mt = Explicit ∧ (Ht > H̄) ∧ (∆t ≥WE→L), mt, otherwise. }
*   **Thinking-Related Signal Mixing:**
    Entrance to latent block: ẽt⋆ ← αt⋆ ·ẽt⋆ + (1− αt⋆)·e_VEC, αt⋆ ∈ [0, 1]
    Exit to explicit block: ẽt† ← βt† ·ẽt† + (1− βt†)·e_VEC, βt† ∈ [0, 1]
*   **Schedules for Mixing:**
    αt = α0 + (1 − α0) t/Tmax
    βt = β0 + (1 − β0) t/Tmax
*   **Token Efficiency Metrics:**
    Plain token efficiency: PEm(ℓ) = Accm(ℓ)/ℓ
    Token efficiency: Em(ℓ) = PEm(ℓ)/PE⋆CoT = (Accm(ℓ)/ℓ) / (Acc⋆CoT/ℓ⋆CoT)
    Average efficiency gain: E[∆Em] = ∫(Em(ℓ)− ECoT(ℓ)) dℓ / ∫ECoT(ℓ) dℓ

### 8. Algorithm Pseudocode
**Algorithm 1 SWIREASONING**
Input: Question x1:n, model M, max steps Tmax, coefficient α0, coefficient β0, dwell window WE→L, max switches Cmax, and answer budget B
Output: Answer y1:m
1 Init: Mode m0 ← Latent, switch counter C ← 0, injection queue Q← ∅, budget flag b← −1
2 for t = 1 to Tmax do
3 ℓt ←M(x1:t−1); pt ← softmax(ℓt); Ht ← −∑v pt[v] log pt[v] // Entropy
4 if Q ̸= ∅ then // Token injection (convergence/termination prefix)
5 xt ← Q.pop()
6 if b = 0 then
7 break
8 if b > 0 then
9 b← b− 1
10 continue
11 if t = 1 then
12 H̄ ← H1; ∆t← 0
13 if mt−1 = Latent and Ht < H̄ then // Mode switching (Sec. 3.3)
14 mt ← Explicit; H̄ ← Ht; ∆t← 0; C ← C + 1
15 else if mt−1 = Explicit and Ht > H̄ and ∆t ≥WE→L then
16 mt ← Latent; H̄ ← Ht; ∆t← 0
17 else
18 mt ← mt−1; ∆t← ∆t+ 1
19 if mt = Explicit and 1/2 Cmax ≤ C ≤ Cmax then // Switch count control (Sec. 3.4)
20 Q← [ ID[_VEC] ] // Convergence trigger
21 else if mt = Explicit and C > Cmax then
22 Q← [ ID[“_VEC\n\n The final answer is”] ]; b← B // Termination trigger
23 if mt = Explicit and ∆t > 0 then
24 xt ← argmaxv pt[v] or Sampling
25 else
26 ẽt ← ∑v pt[v]E[v]
27 if mt = Latent and ∆t = 0 then // Thinking-related signal mixing
28 αt = α0 + (1− α0) t/Tmax
29 ẽt ← αt ẽt + (1− αt) e_VEC
30 if mt = Explicit and ∆t = 0 then // Thinking-related signal mixing
31 βt = β0 + (1− β0) t/Tmax
32 ẽt ← βt ẽt + (1− βt) e_VEC
33 xt ← ẽt // Soft embeddings feed as inputs
34 if xt = <EOS> then
35 break
36 Extract answer y from xn+1:t
37 return y