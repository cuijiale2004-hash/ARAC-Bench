**1. Research Background and Existing Pain Points**
Deep learning has a highly non-convex and complex loss landscape, where the global minimum may not be unique, and there may be many local minima and saddle points where the optimization can be trapped. The optimization dynamics of deep learning are hard to analyze or control. Current popular scaling laws primarily focus on loss, predicting how loss changes as model sizes and training horizons change, assuming optimal learning rate. As a result, the learning rate is not explicitly presented in these laws. Other scaling laws of learning rate scale learning rate across training horizons as η∗peak = λT−α, where α varies significantly across studies (e.g., α = 0.125, α ∈ {0.32, 0.38, 0.42, 0.65, 0.70}, or α = 0 for horizon-unaware laws). Nevertheless, these existing laws may not predict loss. Furthermore, maximal update parameterization (muP) is a technique to transfer optimal learning rate across model size, but empirical evidence contradicts its ability to transfer across training horizons.

**2. Core Research Motivation and Scientific Questions**
The core research motivation is to examine the applicability of convexity and Lipschitz continuity in deep learning, in order to precisely control the loss dynamics via the learning rate schedules. The paper illustrates that deep learning quickly becomes weakly convex after a short period of training, and the loss is predictable by an upper bound on the last iterate, which further informs the scaling of the optimal learning rate. The scientific question is whether insights from convex optimization—specifically the convergence bounds for convex, non-smooth loss with bounded gradients—can be generalized to non-convex deep learning to simultaneously predict loss and optimal learning rate across training horizons and model sizes, thereby establishing a unified scaling law.

**3. Overall Core Idea and Design Philosophy**
The overall core idea is to study the scaling law of deep learning loss and learning rate through the lens of convex loss (non-smooth) and bounded gradient. The design philosophy involves establishing a series of generalizations from convex theory to deep learning in a progressive manner:
1. Starting from rigorous analysis restricted to SGD and convex loss, deriving non-asymptotic and asymptotic upper bounds on loss for both averaged and last iterates.
2. Identifying "qualified" learning rate schedules that achieve the optimal O(1/√T) convergence rates based on symbolic analysis of continuous integrals.
3. Generalizing these theoretical bounds to non-convex deep learning with general optimizers by replacing the physical theoretical constants (D, G, L*) with data-driven fitted quantities (D~, G~, L~∞).
4. Formulating a two-dimensional scaling law that uses data-driven linear regression to predict both loss and optimal learning rate across model sizes (N) and training horizons (T).

**4. Core Innovation Points**
1. Studying convex-like behaviors in deep learning for general model architectures, optimizers, and learning rate schedules, thereby establishing a non-asymptotic mapping from learning rate sequence to loss sequence.
2. Generalizing to an asymptotic upper bound of loss, achieving O(1/√T) convergence when (I) the peak learning rate is scaled by 1/√T and (II) the learning rate schedule is qualified.
3. Proposing a data-driven method to fit the asymptotic bound, establishing a scaling law across training horizons and model sizes.
4. Introducing a "qualifying exam" for learning rate schedules using symbolic analysis and continuous definite integrals, theoretically and empirically proving that linear, cosine, and warmup-stable-decay (WSD) schedules pass (achieving O(1/√T)), while constant and square-root inverse schedules fail.
5. Establishing Generalizations 1-4 which progressively relax convex assumptions and extend to any-iteration loss, last-iteration loss, optimal peak learning rate, and scaled peak learning rate for deep learning.
6. Presenting a two-dimensional scaling law that extrapolates optimal loss and learning rate as much as 80× across training horizons and 70× across model sizes.

**5. Overview of the Overall Technical Solution**
The overall technical solution follows a progressive generalization path from convex theory to deep learning:
- **Phase 1 (Convex SGD Non-asymptotic Bounds):** Establishes Condition 2.1 (convexity, bounded gradient) and derives the upper bound of the averaged iterate (Equation 2.3) and the last iterate (Equation 2.4) for SGD.
- **Phase 2 (Asymptotic Bounds and Qualified Schedules):** Derives the asymptotic loss bound at the last iteration for specific schedules (Theorem 1). Identifies that the optimal loss convergence rate is O(1/√T) and defines qualified schedules satisfying Equation (2.5). Establishes the qualifying exam (Condition 2.5) using definite integrals.
- **Phase 3 (Generalization to Deep Learning):** Proposes Generalization 1 (any-iteration loss bound by replacing D, G, L* with D~, G~, L~∞). Proposes Generalization 2 (last-iteration loss for qualified schedules). Proposes Generalization 3 (optimal peak learning rate asymptotics).
- **Phase 4 (Two-Dimensional Scaling Law):** Proposes Generalization 4 (scaled peak learning rate bound). Establishes the final scaling law (Equation 5.1) using data-driven linear regression to fit quantities Q~ and L~∞, enabling extrapolation across model sizes and horizons.

**6. Detailed Module Design**
**Module 1: Non-asymptotic Bound of SGD**
- **Condition 2.1:** Denoting a differentiable function as L and its gradient as ∇L, then L is convex if ∀(w,x), L(w)− L(x) ≤ (w − x)⊤∇L(w). Bounded in gradient if ∃G s.t. ∀w,E∥g(w)∥2 ≤ G2.
- **Remark 2.2:** Convexity can be replaced by star-convexity and iterate-wise convexity along the optimization path: L(wt)− L(w∗) ≤ (wt −w∗)⊤∇L(wt) where w∗ ∈ argminwL(w), and L(wt)− L(ws) ≤ (wt −ws)⊤∇L(wt).
- **Derivation:** Considering SGD wt+1 = wt − ηt+1g(wt), with gt := g(wt), in the parameter space: ∥wt+1 −w∗∥2 = ∥wt −w∗∥2 − 2ηt+1(wt −w∗)⊤gt + η2t+1∥gt∥2. For bounded gradient and convex loss, in expectation: E∥wt+1 −w∗∥2 ≤ E∥wt −w∗∥2 − 2ηt+1(Lt − L∗) + η2t+1G2. Telescoping sum gives: 0 ≤ E∥wτ −w∗∥2 ≤ ∥w0 −w∗∥2 − 2∑τ−1t=0 ηt+1(Lt − L∗) + ∑τ−1t=0 η2t+1G2.
- **Finding 2.3:** For constant learning rate η, the loss upper bound simplifies to L∗ + D2/(2Tη) + ηG2/2. The optimal η∗ = D/(√TG).

**Module 2: Asymptotic Loss Bound and Qualified Schedules**
- **Theorem 1:** Derives the upper bound of EL(wT), optimal loss bound, and optimal η∗peak(T) for 5 schedules: constant, square-root inverse, linear decaying, cosine decaying, and warmup-stable-decay (WSD).
- **Qualified Schedules:** Defines schedules taking the form of Equation (2.5), which achieve O(1/√T) convergence. Linear, cosine, and WSD are qualified; constant and sqrt inverse are not.
- **Condition 2.5 (Qualifying Exam):** A learning rate schedule function st(T ) ∈ [0, 1] is qualified, if with ηt(T ) := st(T )/√T, the following holds: D2/2 ∫T0 ηtdt + G2/2 ∫T0 (η2t ∫Tt ηkdk) dt = O(1/√T).
- **Corollary 2.4:** Any ηref ∈ R+ achieves asymptotically optimal convergence rate O(1/√T) with ηpeak = ηref/√T. The optimal η∗ref = q1/q2.

**Module 3: Deep Learning Generalizations**
- **Generalization 1 (Any-iteration loss):** Modifies Equation (2.4) by renouncing the physical meaning of D, G, L* and replacing them with fitted quantities D~, G~, L~∞.
- **Generalization 2 (Last-iteration loss):** Modifies Equation (2.5) for qualified schedules, renouncing exact forms of q~1 and q~2 to be data-driven.
- **Generalization 3 (Optimal peak learning rate):** Derived from Corollary 2.4 Part 2, yielding EL(wT) ∼ L~∞ + 2q~1q~2/√T.
- **Generalization 4 (Scaled peak learning rate):** Derived from Corollary 2.4 Part 1, yielding EL(wT) ∼ L~∞ + Q~(ηref)/√T. Enables linear regression of loss against 1/√T.

**Module 4: Data-Driven Fitting and Scaling Law**
- **Fitting Method:** Fits a non-negative linear regression y ∼ Xβ + L~∞ where y ∈ RT,β = [D~, G~]⊤,X ∈ RT×2, and specifically yτ = L(wτ), Xτ,1 = 1/(2∑τt=1 ηt), Xτ,2 = 1/2 (∑τt=1 η2t / ∑τt=1 ηt + ∑τ−1k=1 (ηk ∑τt=k+1 ηt ∑τt=k η2t / ∑τt=k ηt)).
- **Scaling Law:** Connects optimal learning rate and loss across model sizes N and horizons T.

**7. All Mathematical Formulas and Symbol Definitions**
*Symbol Definitions:*
- w: parameters; η: learning rate; g: mini-batch gradient with Eg(w) = ∇L; t: iteration, 0 ≤ t < T
- T: training horizon; ηpeak: peak learning rate; st(T): learning rate schedule function
- L: differentiable function (loss); ∇L: gradient; w∗: minimizer, w∗ ∈ argminwL(w)
- D := ∥w0 − w∗∥; L∗ = minw L(w); G: bounded gradient constant
- q1, q2: constants in the qualified schedule loss bound; Q(ηref) := q21/ηref + ηref q22
- L~∞: irreducible loss in deep learning, L(limτ→∞ wτ); D~, G~: fitted quantities replacing D, G
- q~1, q~2: fitted quantities replacing q1, q2; Q~(ηref): fitted function replacing Q(ηref)
- N: model size; Nsmall, Tsmall: smaller model size and shorter training horizon for transfer

*Mathematical Formulas:*
Condition 2.1:
L(w)− L(x) ≤ (w − x)⊤∇L(w) (2.1)
∃G s.t. ∀w,E∥g(w)∥2 ≤ G2 (2.2)

Remark 2.2:
L(wt)− L(w∗) ≤ (wt −w∗)⊤∇L(wt)
L(wt)− L(ws) ≤ (wt −ws)⊤∇L(wt)

Averaged iterate bound (Equation 2.3):
EL(w̄τ ) ≤ ∑τ−1t=0 ηt+1Lt / ∑τ−1t=0 ηt+1 ≤ L∗ + D2/(2∑τt=1 ηt) + G2∑τt=1 η2t/(2∑τt=1 ηt) := LSGD-ave({ηt}). (2.3)

Last iterate bound (Equation 2.4):
EL(wτ ) ≤ L∗ + D2/(2∑τt=1 ηt) + G2/2 (∑τt=1 η2t / ∑τt=1 ηt + ∑τ−1k=1 ηk∑τt=k+1 ηt ∑τt=k η2t/∑τt=k ηt) (2.4)

Qualified schedule bound (Equation 2.5):
EL(wT ) ≲ L∗ + q21/(Tηpeak) + ηpeakq22 := LSGD-last(ηpeak, T ) (2.5)

Corollary 2.4:
1. LSGD-last(ηpeak = ηref/√T , T )− L∗ ∼ Q(ηref)/√T = O(1/√T), where Q(ηref) := q21/ηref + ηrefq22.
2. The optimal η∗ref = argminηref Q = q1/q2, and LSGD-last(T )− L∗ ∼ 2q1q2/√T.

Condition 2.5 (Qualifying Exam):
D2/2 ∫T0 ηtdt + G2/2 ∫T0 (η2t ∫Tt ηkdk) dt = O(1/√T).

Theorem 1 Cases:
Case 1: Constant ηpeak. EL(wT ) ≲ L∗ + D2/(2Tηpeak) + ηpeakG2/2 lnT. Optimal: L∗ + DG√lnT/T, η∗peak(T ) = D/G √lnT · T.
Case 2: Square-root inverse ηpeak/√t. EL(wT ) ≲ L∗ + D2/(4√Tηpeak) + ηpeakG2 lnT/(4√T). Optimal: L∗ + DG√lnT/(4T), η∗peak(T ) = D/G √lnT.
Case 3: Linear decaying ηpeak (1− t/T ). EL(wT ) ≲ L∗ + D2/(Tηpeak) + ηpeak G2. Optimal: L∗ + 2DG√1/T, η∗peak(T ) = D/G √T.
Case 4: Cosine decaying ηpeak (1+cos(πt/T ))/2. EL(wT ) ≲ L∗ + D2/(Tηpeak) + ηpeak G2 · 1.061. Optimal: L∗ + 2DG√1.061/T, η∗peak(T ) = D/G √1.061T.
Case 5: Warmup-stable-decay. EL(wT ) ≲ L∗ + D2/((1+c)Tηpeak) + ηpeakG2 [1 + 1/2 ln((1+c)/(1−c))]. Optimal: L∗ + 2DG√(1+1/2 ln((1+c)/(1−c)))/((1+c)T), η∗peak(T ) = D/G √((1+c)(1+1/2 ln((1+c)/(1−c))))T.

Proof Derivations for Theorem 1:
Constant: ∫Tk ηtdt = (T − k)η, ∫Tk η2t dt = (T − k)η2. LSGD-last-constant(T )− L∗ = D2/(2Tη) + G2η/2 + G2/2 ∫T−10 ηk (∫Tk−1 η2t dt)/(∫Tk ηtdt ∫Tk−1 ηtdt) dk = D2/(2Tη) + G2η/2 + G2/2 ∫T−10 η/(T − k) dk = D2/(2Tη) + G2/2 η lnT.
Square-root inverse: ∫Tk = 2η(√T + 1 − √k + 1) and ∫Tk η2t = η2 ln(T+1/k+1). LSGD-last-sqrt-inv(T )− L∗ = D2/(4η(√T + 1− 1)) + η ln(T + 1)/(4(√T + 1− 1)) G2 + ηG2/8 ∫T−10 ln(T+1/k)/(√k + 1(√T + 1)− √k + 1)(√T + 1− √k)) dk. Let A = √T + 1. The integral ≈ 4 ln(T + 1) + 6.7726/√T + 1 + o(1/√T). Thus LSGD-last-sqrt-inv(T )− L∗ ≈ D2/(4η√T) + η ln(T)/(4√T) G2 + O(ln(T)/√T).
Linear decay: ∫Tk ηtdt = η(T − k)2/2T, ∫Tk η2t dt = η2(T − k)3/3T 2. LSGD-last-linear(T )− L∗ = D2/(ηT) + η2T/3/(ηT) G2 + G2/2 ∫T−10 ηk η2(T − k + 1)3/3T 2/(η(T − k)2/2T ∗ (T − k + 1)2/2T) dk = D2/(ηT) + ηG2/3 + 2ηG2/3T ∫T−10 (1 + 1/(T − k)) dk = D2/(ηT) + ηG2/3 (3 + 2 lnT/T − 2/T).
Cosine decay: A(k) := ∫Tk ηtdt = η(T − k/2 − T/2π sin(πk/T)). B(k) := ∫Tk η2t dt = η2(3(T − k)/8 − T/2π sin(πk/T) − T/16π sin(2πk/T)). Substituting gives D2/(Tη) + 3ηG2/8 + ηG2/4 ∫T−10 (1 + cos(πk/T)) [3(T−k+1)/8 − T/2π sin(π(k−1)/T) − T/16π sin(2π(k−1)/T)] / ((T−k)/2 − T/2π sin(πk/T)) (T−k+1)/2 − T/2π sin(π(k−1)/T)) dk ≈ 2.7443 + O(1/T). Thus LSGD-last-cosine(T )−L∗ ≈ D2/(Tη) + ηG2(1.061+O(1/T)).

Proof of Theorem 2 (Condition 2.5 Derivation):
For any sequence {xt}, if xT /ηT = 0, then
1/∑Tt=1 ηt ∑Tt=1 xt + ∑T−1k=1 (ηk ∑Tt=k+1 ηt / ∑Tt=k ηt) ∑Tt=k xt = ∑T−1t=1 (xt ∑Tk=t+1 ηk).
Denoting Ak = ηk ∑Tt=k+1 ηt / ∑Tt=k ηt = 1/∑Tt=k+1 ηt − 1/∑Tt=k ηt.
LHS = 1/∑Tt=1 ηt ∑Tt=1 xt + ∑T−1k=1 (Ak ∑T−1t=k xt) + ∑T−1k=1 AkxT
= 1/∑Tt=1 ηt ∑Tt=1 xt + ∑T−1t=1 (xt ∑tk=1 Ak) + xT (∑T−1k=1 Ak)
= ∑T−1t=1 xt (1/∑Tt=1 ηt + ∑tk=1 Ak) + xT (1/∑Tt=1 ηt + ∑T−1k=1 Ak) (A.7)
Since 1/∑Tt=1 ηt + ∑tk=1 Ak = 1/∑Ts=t+1 ηs, substitute back using xT /ηT = 0 to get ∑T−1t=1 (xt ∑Tk=t+1 ηk).
Continuous version for Condition 2.5: L∗ + D2/2 ∫T0 ηtdt + G2/2 ∫T0 (η2t ∫Tt ηkdk) dt.

Generalization 1 (Equation 3.1):
EL(wτ ) ≤ L̃∞ + D̃2/(2∑τt=1 ηt) + G̃2/2 (∑τt=1 η2t/∑τt=1 ηt + ∑τ−1k=1 ηk∑τt=k+1 ηt ∑τt=k η2t/∑τt=k ηt) (3.1)

Generalization 2 (Equation 4.1):
EL(wT ) ∼ L̃∞ + q̃21/(Tηpeak) + ηpeakq̃22 := LDL-last(ηpeak, T ) (4.1)

Generalization 3:
EL(wT ) ∼ L̃∞ + 2q̃1q̃2/√T

Generalization 4:
EL(wT ) ∼ L̃∞ + Q̃(ηref)/√T, ∀ηref

Scaling Law (Equation 5.1):
EL(N,T ) ∼ L̃∞(η∗ref;N) + Q̃(η∗ref;N)/√T
η∗peak(N,T ) ∼ η∗peak(Nsmall, Tsmall)/ √(T/Tsmall) (5.1)

**8. Algorithm Pseudocode**
There is no explicit algorithm pseudocode block provided in the original paper. The full data-driven fitting method is defined mathematically in the text as follows:
"We fit a non-negative linear regression y ∼ Xβ + L̃∞ where y ∈ RT,β = [D̃, G̃]⊤,X ∈ RT×2, and specifically yτ = L(wτ), Xτ,1 = 1/(2∑τt=1 ηt), Xτ,2 = 1/2 (∑τt=1 η2t/∑τt=1 ηt + ∑τ−1k=1 ηk∑τt=k+1 ηt ∑τt=k η2t/∑τt=k ηt)."