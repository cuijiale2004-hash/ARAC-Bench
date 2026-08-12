**1. Research Background and Existing Pain Points**
Unified Multimodal Large Language Models (MLLMs) have attracted growing attention for their capability to conduct both generation and understanding. However, an emerging consensus is that, despite being designed to unify both generation and understanding, they are not truly unified in performance, where understanding typically outperforms generation, revealing an internal generation–understanding gap. Prior works have discussed the internal gap in unified MLLMs, but their mitigation methods often rely on external reward models or additional supervised datasets, or focus solely on improving a single task (e.g., generation), without emphasizing generation–understanding alignment. Additionally, existing studies lack systematic quantification of such gap across multiple MLLMs and tasks, with conclusions often confined to single models or single tasks. Their measurements of internal gap rely on external models instead of measuring internal consistency, which potentially makes biased estimation by external evaluators.

**2. Core Research Motivation and Scientific Questions**
The core scientific question arises: Can the internal gap in MLLMs be leveraged as a free bonus, with the stronger branch guiding the weaker one to improve the model’s performance and mitigate non-unification? The motivation is to systematically investigate the generation–understanding gap in MLLMs through empirical validation, mitigation, mechanistic analysis, and improved method design, exploring the potential of mitigating MLLMs’ non-unification without any external signals.

**3. Overall Core Idea and Design Philosophy**
The overall core idea is to turn the internal gap into self-improvement by promoting the generation-understanding unification in MLLMs. The design philosophy proceeds as follows: 1) First, validate the generation–understanding gap across multiple MLLMs and tasks using a proposed internal consistency metric (non-unification score), confirming that the gap primarily stems from weak generation rather than misunderstanding. 2) Propose an internal gap-based self-improvement framework that aligns MLLMs by leveraging stronger understanding to guide the weaker generation. 3) Analyze the dynamic interplay (co-improvement effect) between generation and understanding during self-improvement using learning dynamic theory extended to multimodal settings. 4) Incorporate curriculum learning into the self-improvement process by gradually introducing harder samples that were initially excluded due to limited capabilities, thereby dynamically expanding post-training data.

**4. Core Innovation Points**
- First introduction of the non-unification score, an internal consistency metric to measure MLLMs’ internal gap. Extensive evaluations across diverse models and tasks confirm pervasive non-unification phenomenon, which is primarily caused by weak generation.
- Proposal of a simple yet effective internal gap-based self-improvement framework, which leverages stronger understanding capability to guide the weaker generation without any external signals. Standard pipelines (SFT and DPO) significantly boost generation and unification.
- Empirical identification of a co-improvement effect in self-improvement, where understanding better detects prompt-misaligned generations (false positives). Extension of learning dynamic theory to the MLLM setting, showing that the shared empirical neural tangent kernel (eNTK) between generation and understanding encourages aligned learning dynamics, thereby driving co-improvement.
- Inspiration of a curriculum-based self-improvement strategy: progressively strengthen understanding and generation to reuse underutilized samples, thereby expanding post-training data and boosting both performance and unification.

**5. Overview of the Overall Technical Solution**
The technical solution starts with the validation of the non-unification phenomenon using the non-unification score. It then implements an internal gap-based self-improvement framework where, given an image generation prompt, the MLLM produces candidate images, which are then scored by its own understanding branch. Images judged as aligned are selected as chosen samples, and those misaligned are rejected, forming preference data for DPO and supervision pairs for SFT. During this process, a co-improvement effect is observed and theoretically explained via shared eNTK in learning dynamics. Finally, a curriculum learning approach is incorporated: previously discarded prompts (where generation failed or understanding was inaccurate) are revisited by the progressively enhanced model to dynamically expand the post-training data, leading to further improvements.

**6. Detailed Module Design**
- **Non-unification Score Module**: Consider an MLLM πθ, a prompt y and the generated image x = πθgen(y). Form an image–question pair (x, q(y)), where q(y) := “Does this image describe y?”. This pair is processed by the understanding branch πθund(·), yielding a binary decision. The non-unification score is the proportion of decisions equal to 0.
- **Self-Improvement Module (Data Construction)**: Given an image generation prompt y, the MLLM πθ produces N candidate images {xi}Ni=1 = πθ(y). Each candidate xi is paired with the question q(y) and processed by understanding branch πθund. Images judged as aligned are labeled as chosen, while those judged as misaligned are labeled as rejected, forming preference data for DPO and supervision pairs for SFT.
- **Curriculum Learning Module**: As generation and understanding improve together, difficult samples that pre-trained MLLMs could not previously utilize can be incorporated later, forming an adaptive data expansion process. During curriculum replay, for each prompt y in the discard pool B, the model regenerates images and rescores them. If the understanding branch judges any as aligned, they are added to the post-training data and the prompt is removed from the discard pool.

**7. All Mathematical Formulas and Symbol Definitions**
- Non-unification score: 
  Non-unification score := E(x,y)I[πθund(x, q(y)) = 0]
- Weak Generation: 
  Weak Generation := P(πθund(x, q(y)) = πQwenund(x, q(y)) | πθund(x, q(y)) = 0)
- Human-evaluated Weak Generation: 
  Human-evaluated Weak Generation := P(πθund(x, q(y)) = Shuman(x, q(y)) | πθund(x, q(y)) = 0)
- Likelihood of sample (y0,x0):
  πθ(x0 | Y0) = ∏Mk=1 πθ(x0,k | y0,x0,<k) = ∏Mk=1 [softmax(z0und)]x0,k,k (Generation)
  πθ(y0 | X0) = ∏Lℓ=1 πθ(y0,ℓ | x0,y0,<ℓ) = ∏Lℓ=1 [softmax(z0gen)]y0,ℓ,ℓ (Understanding)
- One-step learning dynamics:
  ∆Gt(x0 | Y0) := log πθt+1(x0 | Y0)− log πθt(x0 | Y0) (Generation)
  ∆Ut(y0 | X0) := log πθt+1(y0 | X0)− log πθt(y0 | X0) (Understanding)
- Derivation linking Generation and Understanding:
  log πθ(y0 | x0) = log πθ(x0 | y0)− log πθ(x0) + C.
  ∆Ut(y0 | X0) = ∆Gt(x0 | Y0)−∆ log πt(x0).
- Lemma 1 (Learning Dynamics of Generation under SFT):
  ∆Gt(x0 | Y0) = −η ∑Mk=1 ∑Mr=1 (ex0,k − π0k)⊤Ktk,r(Y0,Yu)(πur − exu,r ) +O(η2)
- Lemma 2 (Learning Dynamics of Understanding under SFT):
  ∆Ut(y0 | X0) = −η ∑Mk=1 ∑Mr=1 ∑yi≠y0 wθt(yi | x0)( (ex0,k − π0k)⊤Ktk,r(Y0,Yu)− (ex0,k − πik)⊤Ktk,r(Yi,Yu) )(πur − exu,r ) +O(η2)
  where wθt(y | x0) := πθt (x0|y)/∑y′ πθt (x0|y′)
- DPO loss:
  LDPO(yu,x+u ,x−u ) = −E(yu,x+u ,x−u )[log σ(β log πθ(x+u | Y+u)/πref(x+u | Y+u) − β log πθ(x−u | Y−u)/πref(x−u | Y−u))]
- Lemma 3 (Learning Dynamics of Generation under DPO):
  ∆Gt(x0 | Y0) = −ηβσ(−α) ∑Mk=1 ∑Mr=1 (ex0,k − π0k)⊤ [Ktk,r(Y0,Y+u)(πu,+r − ex+u,r )−Ktk,r(Y0,Y−u)(πu,−r − ex−u,r )] +O(η2)
  where the margin α := β log πθ(x+u |Y+u)/πref (x+u |Y+u) − β log πθ(x−u |Y−u)/πref (x−u |Y−u)
- Lemma 4 (Learning Dynamics of Understanding under DPO):
  ∆Ut(y0 | X0) = −ηβσ(−α) ∑Mk=1 ∑Mr=1 ∑yi≠y0 wθt(yi | x0)( (ex0,k − π0k)⊤(Ktk,r(Y0,Y+u)(πu,+r − ex+u,r )−Ktk,r(Y0,Y−u)(πu,−r − ex−u,r )) − (ex0,k − πik)⊤(Ktk,r(Yi,Y+u)(πu,+r − ex+u,r )−Ktk,r(Yi,Y−u)(πu,−r − ex−u,r )) ) +O(η2)
- Win rate: 
  Win rate := ∑y I[πpreund(xpre, q(y)) ≠ πselfund(xpre, q(y)) ∧ πselfund(xpre, q(y)) = sQwen] / ∑y I[πpreund(xpre, q(y)) ≠ πselfund(xpre, q(y))]
- Symbol definitions:
  y0 = (y0,1, . . . , y0,L) Tokenized text prompt (index form), length L
  x0 = (x0,1, . . . , x0,M) Tokenized image (index form), length M
  U0 = [u0,1 · · ·u0,M ] Image token embedding matrix
  V0 = [v0,1 · · ·v0,L ] Text token embedding matrix
  X0 = [U0 | V0 ] Input to understanding branch
  Y0 = [V0 | U0 ] Input to generation branch
  Xu = [Uu | Vu] Post-training sample (image) for SFT/DPO updates
  Yu = [Vu | Uu] Post-training sample (prompt) for SFT/DPO updates
  πθ Unified MLLM parameterized by θ
  hθ(·) Logit network producing token-wise logits before softmax
  V Unified vocabulary size for text and image tokens
  ztk(S) Logits at position k on sequence S at epoch t
  πtk = softmax(ztk) Token distribution at position k
  ex One-hot vector corresponding to token x
  πθ(x0 | Y0) Generation likelihood of image tokens
  πθ(y0 | X0) Understanding likelihood of text tokens
  ∆Gt(x0 | Y0) One-step update of generation log-likelihood
  ∆Ut(y0 | X0) One-step update of understanding log-likelihood
  Ktk,r(Y0,Yu) Empirical NTK: (∇θtz0k)(∇θtzur)⊤

**8. Algorithm Pseudocode**
Algorithm 1: Self-Improvement (SFT)
Input: πθ , prompts P , image candidates N , epochs T
Data: DSFT←∅, discard pool B←∅
for y ∈ P do
    {xi}Ni=1←πθgen(y);
    si←πθund(xi, q(y))∈{0, 1};
    C←{xi : si = 1}; 
    if |C| = 0 then
        B←B ∪ {y}
    else
        DSFT←DSFT ∪ {(y,xchosen) |xchosen∈C}
for t = 1 to T do
    θ ← θ − η∇θLgen(θ;DSFT) ;

Algorithm 2: Curriculum Replay
Input: πθ , discard pool B, image candidates N , curriculum epochs Ecur
Data: DSFT (shared with Alg. 1)
for t ∈ Ecur do
    for y ∈ B do
        {x̃j}Nj=1←πθgen(y);
        s̃j←πθund(x̃j , q(y));
        C̃←{x̃j : s̃j = 1}; 
        if |C̃| > 0 then
            DSFT←DSFT ∪ {(y,x) |x∈C̃};
            remove y from B