### 1. Research Background and Existing Pain Points

**Research Background:**
Continuous-state diffusion models have proven effective in both unconditional and conditional generation tasks, such as generating data from natural language prompts. More recently, progress in discrete diffusion modeling has extended the applicability of diffusion-based generation to new domains, including molecular design, protein synthesis, and languages. A widely adopted technique to improve fidelity and alignment with conditioning inputs in these models is Classifier-Free Guidance (CFG). In continuous domains, dynamic guidance schedules—where guidance strength varies over the generation trajectory—have shown especially effective. Strategies such as guidance intervals and gradually increasing schedules can significantly enhance sample quality and are increasingly adopted in practice. Concurrently, extensions of CFG to discrete diffusion models have yielded promising empirical gains.

**Existing Pain Points:**
1. **Unintentional Imbalanced Transitions in Discrete CFG:** Current Classifier-Free Guidance implementations in discrete diffusion can unintentionally cause imbalanced transitions, such as unmasking too rapidly during the early stages of generation. This accelerated unmasking introduces stiffness and inefficiencies, degrading the quality of the resulting samples.
2. **Lack of Theoretical Understanding for Discrete Guidance Schedules:** Dynamic guidance schedules have been limited to continuous (state-space) diffusion models. It remains unclear whether such schedules are also effective in discrete state diffusion. Defining and optimizing effective guidance strategies in discrete spaces is a fundamentally challenging and open research problem. There is a lack of theoretical justification to characterize guidance schedules and the mechanisms by which they improve sample generation in the discrete setting.

### 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:**
To better understand the mechanisms by which guidance affects the sampling process in discrete diffusion in a principled way, aiming to improve the algorithms theoretically and practically. 

**Scientific Questions:**
1. How does the guidance schedule affect the distribution of the generated samples?
2. Is it possible to characterize properties of good guidance schedules?

### 3. Overall Core Idea and Design Philosophy

**Overall Core Idea:**
The paper starts by analyzing the exact effect of CFG in the context of a low-dimensional masked diffusion model, with a special emphasis on the guidance schedule. By deriving explicit formulas for the sampled distribution under varying guidance schedules in 1 and 2 dimensions, the analysis reveals an imperfection in current CFG implementations—unintentionally fast unmasking rates caused by the partition function scaling the jump rate. To address this, the paper draws insight from the analysis and proposes a novel classifier-free guidance mechanism based on column normalization of the rate matrix. Intuitively, the method smooths the transport between the data distribution and the initial (masked) distribution.

**Design Philosophy:**
1. **Low-Dimensional Theoretical Insights:** Utilize 1D and 2D exact analysis of masked diffusion to uncover fundamental flaws and properties of guidance schedules that scale to high dimensions.
2. **Decoupling Rate and Distribution:** Isolate the source of the unmasking pathology by decomposing the guided rate matrix into a jump rate and a jump distribution, explicitly decoupling them to prevent the normalizing constant from distorting the transition speed.
3. **Principled Normalization:** Apply column normalization to the rate matrix to stabilize the sampling process, ensuring the transport smoothness matches the original unguided process.

### 4. Core Innovation Points

1. **Identified a Key Flaw in Existing Discrete Guidance Mechanisms:** The paper identifies that existing discrete guidance mechanisms complicate simulation by causing unintentional fast unmasking rates. It provides a theoretical explanation of its cause, showing that the partition function $Z_w$ does not merely affect the jump distribution but rescales the overall jump rate, leading to disproportionately fast transitions.
2. **Proposed a Novel Classifier-Free Guidance Mechanism via Column Normalization:** To address the identified flaw, the paper proposes a normalized guidance mechanism that explicitly decouples rate and distribution by normalizing the guided rate matrix column-wise. This change is theoretically justified, easy to implement (a simple one-line code change), and compares favorably to existing approaches in practice.
3. **First Theoretical Justifications for Guidance Schedules in Discrete Diffusion:** The paper provides the first theoretical justifications to characterize guidance schedules and the mechanisms by which they improve sample generation. The analysis reveals that high guidance early in sampling harms generation quality, while late-stage guidance improves it.
4. **Characterization of Schedule Interpolation:** The 2D theoretical analysis proves that guidance schedules induce an interpolation of different tilted distributions, where the portion assigned to each depends on the time parameters. This theoretically predicts that increasing schedules (higher guidance at the final and middle phases, small early guidance) are the most effective.

### 5. Overview of the Overall Technical Solution

The overall technical solution begins by defining discrete diffusion via a Continuous Time Markov Chain (CTMC) governed by a rate matrix. In the context of masked diffusion, the forward process gradually corrupts a clean sequence by masking entries, while generation is achieved by reversing this process. Existing CFG methods (Unlocking and Simple Guidance) interpolate the reverse transitions but inadvertently introduce a dependency on the normalizing constant $Z_w$ in the jump rate.

To analyze this, the paper first examines the 1D case, deriving the exact distribution under constant guidance (Theorem 3.1) and showing the exponentiation of $Z_w$ accelerates unmasking. The rate matrix is then decomposed into a jump rate $r_{t,p}(x)$ and a jump distribution $p_t(y|x)$. It is shown that the standard guided rate matrix multiplies the rate by $Z_w$, which is identified as the source of the pathology. 

The proposed solution, Normalized Guidance, removes $Z_w$ from the rate component, leaving only the interpolated jump distribution. This column normalization smooths the transport between the initial and data distributions. 

Next, the paper extends the analysis to 2D with dynamic guidance schedules (Corollary 3.1). By deriving the exact sampled distribution under piece-wise constant guidance, it is shown that the final distribution is an interpolation of tilted distributions weighted by the duration of the time intervals. This leads to the theoretical conclusion that effective schedules apply stronger guidance during the middle and later stages, while keeping early guidance small. Finally, the method is validated empirically on conditional image and text generation tasks.

### 6. Detailed Module Design

**Module 1: Discrete Diffusion via CTMC**
Given an initial distribution $p \in \mathbb{R}^{M^d}$, discrete diffusion is defined by a rate matrix $R_t \in \mathbb{R}^{M^d \times M^d}$ and a CTMC forward process. The time reversal of this process corresponds to a different CTMC defined by the reverse transition matrix identities, which rely on the concrete score $p_t(y)/p_t(x)$.

**Module 2: Masked Discrete Diffusion**
A special case where tokens transition only from a clean state to a masked state $M$, remaining masked thereafter. The distribution of each token is defined by $p_t(x^i_t|x_0)$, and the scores decompose into a time-dependent term and a time-independent probability distribution.

**Module 3: Existing Guidance Mechanisms**
- **Unlocking Guidance:** Constructs a guided backwards transition by interpolating between two transition matrices using the tilted distribution $p^{(w)}(x) = Z^{-1}_w p^w(x) \cdot q^{1-w}(x)$.
- **Simple Guidance:** Directly interpolates the transition probabilities themselves, biasing towards the target distribution $p$ as $w$ increases.

**Module 4: Rate Matrix Decomposition**
To isolate the source of the unmasking issue, the rate matrix is decomposed into a jump rate and a jump distribution:
$$R_t(y, x) = r_{t,p}(x) \left( \frac{1}{r_{t,p}(x)} \frac{p_t(y)}{p_t(x)} R_t(x, y) \right) := r_{t,p}(x) p_t(y|x)$$
Under standard Unlocking Guidance, this becomes:
$$R^{(w)}_t(y, x) = r^w_{t,p}(x) r^{1-w}_{t,q}(x) Z_w \cdot Z^{-1}_w p^w_t(y|x) q^{1-w}_t(y|x)$$
The $Z_w$ term rescaling the overall jump rate is identified as the flaw causing disproportionately fast transitions.

**Module 5: Normalized Guidance Mechanism**
To correct the flaw, the proposed mechanism explicitly decouples rate and distribution by normalizing the guided rate matrix column-wise, removing $Z_w$ from the rate:
$$R^{(w)}_{nor,t}(y, x) = r^w_{t,p}(x) r^{1-w}_{t,q}(x) \cdot Z^{-1}_w p^w_t(y|x) q^{1-w}_t(y|x)$$
In the case of masked diffusion, this normalization admits a simple form using Softmax, which can be implemented as a one-line code change.

**Module 6: Dynamic Guidance Schedule Analysis (2D)**
Considers a time partition $0 = t_0 < t_1 < t_2 < t_3 = T$ with guidance $w_i$ in the interval $[t_i, t_{i+1})$. The derived formula shows that the sampled distribution follows an interpolation of different tilted distributions $p_{(w_i)}$ and combined distributions $p_{(w_i, w_j)}$, where the coefficients depend on the time parameters. The analysis of the coefficients and the role of guidance weights reveals that combined distributions more closely resemble the tilted distribution of the guidance applied at the end of generation, validating increasing schedules.

### 7. All Mathematical Formulas and Symbol Definitions

**Symbol Definitions:**
- $M$: Vocabulary size.
- $S = \{1, 2, \ldots, M\}^d$: State space with $d$ tokens (dimension).
- $p_t$: Distribution at time $t$.
- $R_t$: Rate matrix at time $t$.
- $w$: Guidance strength.
- $Z_w$: Normalizing constant of the guided jump measure, $Z_w = \sum_{y \neq x} p^w_t(y|x) q^{1-w}_t(y|x)$.
- $\sigma_t, \bar{\sigma}_t$: Unmasking schedule and its integral, $\bar{\sigma}_t = \int_0^t \sigma_s ds$.
- $M$: Masked token.

**Mathematical Formulas:**
Forward CTMC:
$$\frac{dp_t}{dt} = R_t p_t, \quad p_0 = p$$

Reverse CTMC:
$$\frac{dp_{T-t}}{dt} = R_{T-t} p_{T-t}$$

Reverse transition matrix identities:
$$R_t(y, x) = R_t(x, y) \cdot \frac{p_t(y)}{p_t(x)}, \quad R_t(x, x) = -\sum_{y \neq x} R_t(y, x)$$

Masked token distribution:
$$p_t(x^i_t|x_0) = \begin{cases} x^i_0 & \text{with probability } e^{-\sigma_t} \\ M & \text{with probability } 1- e^{-\sigma_t} \end{cases}$$

General guided distribution:
$$p^{(w)}(x) \propto p^w(x) q^{1-w}(x)$$

Unlocking Guidance rate matrix:
$$R^{(w)}_t(y, x) = R_t(x, y) \cdot \left(\frac{p_t(y)}{p_t(x)}\right)^w \left(\frac{q_t(y)}{q_t(x)}\right)^{1-w}, \quad R^{(w)}_t(x, x) = -\sum_{y \neq x} R^{(w)}_t(y, x)$$

Simple Guidance transition:
$$p^{(w)}_{simple}(z_s|z_t) \propto p^w(z_s|z_t) q^{1-w}(z_s|z_t)$$

Dynamic guidance schedule process:
$$\frac{dp_{T-t}}{dt} = R^{(w_{T-t})}_{T-t} p_{T-t}$$

Theorem 3.1 (Informal 1D Distribution):
$$p_t = \left(1 - \left(\frac{1-e^{-\sigma_t}}{1-e^{-\sigma_T}}\right)^{Z_w}\right) \cdot p^{(w)}$$

Rate Matrix Decomposition:
$$R_t(y, x) = \underbrace{r_{t,p}(x)}_{\text{rate}} \underbrace{\left(\frac{1}{r_{t,p}(x)} \frac{p_t(y)}{p_t(x)} R_t(x, y)\right)}_{p_t(y|x)}, \quad r_{t,p}(x) = \sum_{y \neq x} R_t(y, x)$$

Guided Rate Matrix (Original Decomposition):
$$R^{(w)}_t(y, x) = \underbrace{r^w_{t,p}(x) r^{1-w}_{t,q}(x) Z_w}_{\text{rate}} \underbrace{Z^{-1}_w p^w_t(y|x) q^{1-w}_t(y|x)}_{\text{distribution}}$$

Lemma 3.1 (Transition rates between masked and unnormalized states):
$$\bar{R}^{(w)}_t(y,M) = R_t(x, y) \frac{e^{-\bar{\sigma}_t}}{1-e^{-\bar{\sigma}_t}} Z_w p^{(w)}(y)$$

Normalized Guidance Rate Matrix:
$$R^{(w)}_{nor,t}(y, x) = \underbrace{r^w_{t,p}(x) r^{1-w}_{t,q}(x)}_{\text{rate}} \underbrace{Z^{-1}_w p^w_t(y|x) q^{1-w}_t(y|x)}_{\text{distribution}}$$

Normalized Guidance for Masked Diffusion:
$$R^{(w)}_{nor,t}(\hat{x},x) = R_t(x, \hat{x}) \frac{e^{-\sigma_t}}{1-e^{-\sigma_t}} \text{Softmax}(w \log p_0(\hat{x}_i|x_{UM}) + (1-w) \log q_0(\hat{x}_i|x_{UM}))$$

Corollary 3.1 (2D Schedule Distribution):
$$p_{t_0}(i, j) = \left(\frac{t_3-t_2}{t_3}\right)^2 p_{(w_2)}(i, j) + \left(\frac{t_2-t_1}{t_3}\right)^2 p_{(w_1)}(i, j) + \left(\frac{t_1-t_0}{t_3}\right)^2 p_{(w_0)}(i, j) + \frac{(t_3-t_2)(t_2-t_1)}{t_3^2} p_{(w_1,w_2)}(i, j) + \frac{(t_3-t_2)(t_1-t_0)}{t_3^2} p_{(w_0,w_2)}(i, j) + \frac{(t_2-t_1)(t_1-t_0)}{t_3^2} p_{(w_0,w_1)}(i, j)$$

Combined distribution definition:
$$p_{(w,\gamma)}(i, j) = p_{(w)}(i, j|X_1 = i)p_{(\gamma)}(X_1 = i) + p_{(w)}(i, j|X_2 = j)p_{(\gamma)}(X_2 = j)$$

Lemma A.1 (Score Decomposition):
$$\frac{p_t(\hat{x}_t)}{p_t(x_t)} = \frac{e^{-\sigma_t}}{1-e^{-\sigma_t}} p_0(\hat{x}^i_t|x_{UM})$$

Lemma B.1 (Matrix Exponential for 1D):
$$e^A = I + A \cdot \frac{e^{v_n}-1}{v_n}$$

Theorem B.1 (General 1D Distribution):
$$p_t = \left(A_1 \cdot \frac{1-e^A}{A}, \cdots, A_{M-1} \cdot \frac{1-e^A}{A}, e^A\right)^\intercal$$
$$A_i = \int_T^t \frac{\sigma_s e^{-\sigma_s}}{1-e^{-\sigma_s}} Z_{w_s} p_{z,w_s}(i)ds, \quad A = -\sum_{i=0}^{M-1} A_i = \int_T^t \frac{\sigma_s e^{-\sigma_s}}{1-e^{-\sigma_s}} Z_{w_s}ds$$

Corollary B.1 (Constant Guidance 1D):
$$p_s(i) = p_t(i) + p_s(M) \left(\frac{1-e^{-\sigma_t}}{1-e^{-\sigma_s}} - 1\right)^{Z_w} p^{(w)}(i) \quad \text{for } i \neq M$$
$$p_s(M) = \left(\frac{1-e^{-\sigma_t}}{1-e^{-\sigma_s}} - 1\right)^{Z_w} p_t(M)$$

Theorem B.2 (Piece-wise Constant 1D):
$$p_\delta = p_T + \sum_{i=0}^{k-1} p_{t_{i+1}}(M) \cdot \left(1 - \left(\frac{1-e^{-\sigma_{t_i}}}{1-e^{-\sigma_{t_{i+1}}}}\right)^{Z_{w_i}}\right) p_{(w_i)}$$
$$p_{t_i}(M) = p_{t_{i+1}}(M) \left(\frac{1-e^{-\sigma_{t_i}}}{1-e^{-\sigma_{t_{i+1}}}}\right)^{Z_{w_i}}$$

Lemma C.1 (Matrix Exponential for 2D):
$$\exp(\alpha A) = \begin{pmatrix} 1 & a(1-e^{-\alpha}) & b(1-e^{-\alpha}) & \frac{(ac+bd)(e^\alpha-1)^2 e^{-2\alpha}}{2} \\ 0 & e^{-\alpha} & 0 & c (e^\alpha - 1) e^{-2\alpha} \\ 0 & 0 & e^{-\alpha} & d (e^\alpha - 1) e^{-2\alpha} \\ 0 & 0 & 0 & e^{-2\alpha} \end{pmatrix}$$

Theorem C.1 (General 2D Distribution):
$$p_s(i, j) = \begin{cases} p_t(i, j) + (1- e^{-\alpha})^2 p^{(w)}(i, j) p_t(M,M) + (1- e^{-\alpha}) [p^{(w)}(X_2 = j | X_1 = i) p_t(i,M) + p^{(w)}(X_1 = i | X_2 = j) p_t(M, j)] & \text{if } i, j \neq M \\ e^{-\alpha} p_t(i,M) + e^{-2\alpha}(e^\alpha - 1) p^{(w)}(X_1 = i) p_t(M,M) & \text{if } i \neq M, j = M \\ e^{-\alpha} p_t(M, j) + e^{-2\alpha}(e^\alpha - 1) p^{(w)}(X_2 = j) p_t(M,M) & \text{if } i = M, j \neq M \\ e^{-2\alpha} p_t(M,M) & \text{if } i = j = M \end{cases}$$
(where $\alpha = \ln \left(\frac{1-e^{-\sigma_t}}{1-e^{-\sigma_s}}\right)$)

Corollary C.1 (Specific Schedule 2D):
$$p_s(i, j) = \begin{cases} p_t(i, j) + \left(\frac{t-s}{t}\right)^2 p^{(w)}(i, j) p_t(M,M) + \left(\frac{t-s}{t}\right) [p^{(w)}(X_2 = j | X_1 = i) p_t(i,M) + p^{(w)}(X_1 = i | X_2 = j) p_t(M, j)] & \text{if } i, j \neq M \\ \frac{s}{t} \cdot p_t(i,M) + \left(\frac{s}{t}\right)^2 \left(\frac{t-s}{s}\right) p^{(w)}(X_1 = i)p_t(M,M) & \text{if } i \neq M, j = M \\ \frac{s}{t} \cdot p_t(M, j) + \left(\frac{s}{t}\right)^2 \left(\frac{t-s}{s}\right) p^{(w)}(X_2 = j)p_t(M,M) & \text{if } i = M, j \neq M \\ \left(\frac{s}{t}\right)^2 p_t(M,M) & \text{if } i = j = M \end{cases}$$

### 8. Algorithm Pseudocode

**Listing 1: Normalized Guidance for Masked Diffusion using Euler Transitions**
```python
def normalized_guidance_euler_transition(
    x, c, t, dt, w
):
    uncond = model(x, cond=None)
    cond = model(x, cond=c)
    logits = w * cond + (1 - w) * uncond
    p_theta = logits.softmax(dim=-1)
    s, s_bar = sigma(t), sigma_bar(t)
    change = dt * s * (1 - exp(-s_bar))
    return sample(delta(x) + change * p_theta)
```

**Listing 2: Unlocking/Simple Guidance for Masked Diffusion using Euler Transitions**
```python
def other_guidance_euler_transition(
    x, c, t, dt, w
):
    uncond = model(x, cond=None)
    cond = model(x, cond=c)
    logits = w * cond + (1 - w) * uncond
    p_theta = logits.exp()
    s, s_bar = sigma(t), sigma_bar(t)
    change = dt * s * (1 - exp(-s_bar))
    return sample(delta(x) + change * p_theta)
```

**Listing 4: Normalized Guidance in the General Case using Euler Transitions**
```python
def normalized_guidance_euler_transition(
    x, c, t, dt, w
):
    # Get scores
    log_score_c = get_score(x, t, cond=c)
    log_score_u = get_score(x, t, cond=None)
    score_c = log_score_c.exp()
    score_u = log_score_u.exp()
    # Set diagonal terms to zero
    score_c.scatter_(-1, x[..., None], torch.zeros_like(score_c))
    score_u.scatter_(-1, x[..., None], torch.zeros_like(score_u))
    # Multiply matrix by edge to get the rates
    rate_c = edge * score_c
    rate_u = edge * score_u
    # Obtain the diagonal term
    total_rate_c = score_c.sum(dim=-1, keepdim=True)
    total_rate_u = score_u.sum(dim=-1, keepdim=True)
    # Get jump distributions
    jump_c = rate_c / total_rate_c
    jump_u = rate_u / total_rate_u
    # Get guided jump distribution
    jump_w = torch.softmax(w * jump_c + (1-w) * jump_u)
    total_jump_w = torch.exp(w * total_rate_c + (1-w) * total_rate_u)
    # Convert to the rate matrix
    rate_w = jump_w * total_jump_w
    rate_w.scatter_(-1, x[..., None], -total_jump_w)
    return sample(delta(x) + dt * sigma(t) * rate_w)
```