**1. Research Background and Existing Pain Points**

Spiking Neural Networks (SNNs) draw inspiration from biological neurons to enable brain-like computation, demonstrating effectiveness in processing temporal information with energy efficiency and biological realism. Traditional Artificial Neural Networks (ANNs) excel across many tasks but differ from real biological mechanisms and require far more compute than the human brain. This gap has motivated SNNs, which model neural activity more realistically by communicating through discrete spikes rather than continuous values. Their event-driven computation paradigm allows for significant energy savings, particularly when implemented on neuromorphic hardware. Additionally, SNNs naturally handle time as part of their processing, making them well-suited for tasks with time-series data or real-time interactions.

Despite these advantages, existing SNN models predominantly describe spiking neuronal membrane-potential dynamics using the widely adopted Integrate-and-Fire (IF) and Leaky Integrate-and-Fire (LIF) neurons, along with variants including nonlinear spike initiation, ternary spikes, adaptive membrane time constants, and threshold adaptation or learning. These models discretize first-order ordinary differential equations (ODEs) which contain only d/dt terms. A fundamental assumption in these models is a Markovian property in which the current state depends mainly on the immediate previous state. The existing pain point is that while this simplification enables computational tractability, it fundamentally limits the expressiveness of these networks. Neurophysiological research has demonstrated that real neurons display far more complex behaviors influenced by long-range correlations, fractal dendritic structures, and the interaction of multiple active membrane conductances. These dynamics cannot be adequately captured by integer-order models and suggest that non-Markovian dynamics play a significant role in biological neural computation.

**2. Core Research Motivation and Scientific Questions**

The core research motivation stems from empirical studies of real neurons which reveal long-range correlations and fractal dendritic structures, suggesting non-Markovian behavior better modeled by fractional-order ODEs (f-ODEs). Fractional calculus offers mathematical tools for modeling such dynamics better than standard first-order ODEs. In contrast to integer-order calculus, the fractional-order derivative dα/dtα, with non-integer α values, considers the entire history of a function, weighted by a power-law kernel. The fractional leaky integrate-and-fire (f-LIF) neuron dynamic can effectively explain spiking frequency adaptations observed in most biological neurons and has been shown to generate more reliable spike patterns than integer-order models when subjected to noisy input.

The scientific questions addressed are: How to formulate a generalized fractional-order SNN (f-SNN) framework that incorporates fractional-order dynamics into the neuronal membrane potential charging to capture non-Markovian characteristics and long-term dependencies? What are the fundamental theoretical distinctions between f-SNNs and traditional SNNs regarding memory behavior, irreducibility, and robustness? And how can this framework be seamlessly integrated into existing SNN architectures and trained effectively?

**3. Overall Core Idea and Design Philosophy**

The overall core idea is to introduce a generalized fractional-order SNN (f-SNN) framework, which incorporates fractional-order dynamics into the neuronal membrane potential charging. By replacing the first-order ODE neurons traditionally used in SNNs with fractional-order ODEs (f-ODEs), f-SNN naturally captures long-term dependencies that are beyond the capability of standard SNN models, leading to improved performance on tasks that require complex temporal processing. The design philosophy emphasizes that this framework is a more general framework which subsumes many traditional SNNs as special instances by setting α = 1. The framework is designed to be compatible with diverse architectures (CNN, Transformer, ResNet, MLP) and is supported by an open-source toolbox to facilitate adoption.

**4. Core Innovation Points**

1.  **Framework Formulation**: We propose an f-SNN framework that integrates f-ODEs into SNNs to naturally capture long-term dependencies using the fractional-order operator dα/dtα. This framework generalizes the traditional class of integer-order SNNs that use IF, LIF neuron dynamics, and their variants, subsuming them as a special case by setting α = 1.
2.  **Theoretical Distinctions**: We establish fundamental theoretical distinctions between f-SNNs and traditional SNNs, proving that fractional-order dynamics confer three key advantages: persistent memory through power-law relaxation, irreducibility to finite classical ensembles, and enhanced robustness to perturbations.
3.  **Compatibility and Integration**: We underscore the compatibility of f-SNN, emphasizing its ability to be seamlessly integrated to augment the performance of many existing SNNs by using non-integer α with various neural network architectures like convolutional neural networks (CNN), Transformer, ResNet, and multilayer perceptron (MLP).
4.  **Open-Source Toolbox**: We provide the community with an open-source, out-of-the-box toolbox, spikeDE, to support the f-SNN framework across diverse architectures and real-world tasks.

**5. Overview of the Overall Technical Solution**

The technical solution involves replacing the first-order derivative d/dt in traditional SNN neuron dynamics with the generalized Caputo fractional derivative Dα of order α ∈ (0, 1]. We present fractional extensions for IF and LIF neurons, defining f-IF and f-LIF dynamics. To make the time-varying solution compatible with neural network models, we discretize the fractional-order ODEs using the fractional Adams–Bashforth–Moulton (ABM) predictor discretization. This formulation makes the memory effect explicit by incorporating weighted contributions from all past function evaluations, reflecting the nonlocal nature of Dα. The discrete updates involve a history-convolution with a stationary power-law kernel. Spiking and reset mechanisms follow the same rules as integer-order cases, and training is achieved using the surrogate-gradient method. For efficiency, a short-memory approximation principle can be applied by truncating the history sum to a sliding window.

**6. Detailed Module Design**

*   **Neuron Dynamics Module**: The subthreshold dynamics are modeled using f-ODEs. For f-IF: τ DαU(t) = RIin(t). For f-LIF: τ DαU(t) = -U(t) + RIin(t).
*   **Discretization Module**: Utilizes the fractional Adams–Bashforth–Moulton (ABM) predictor to discretize the f-ODEs. Using time grid tk = kh, step size h > 0, and yk denoting the numerical approximation of y(tk), the update is: yk = y0 + (1/Γ(α)) * Σ_{j=0}^{k-1} μ_{j,k} f(tj, yj).
*   **Iteration Module (Charge-Spike-Reset)**:
    *   **Charge**: The membrane potential update incorporates a history-convolution.
        *   f-IF charge: Uk = U0 + Σ_{m=0}^{k-1} c^{(α)}_m X_{k-m}
        *   f-LIF charge: Uk = U0 + Σ_{m=0}^{k-1} c^{(α)}_m (-Uk-1-m + Xk-m)
    *   **Spike**: S_k = H(U_k - θ), where H(·) is the Heaviside step function.
    *   **Reset**:
        *   Soft reset: Uk ← Uk - θ Sk
        *   Hard reset: Uk ← (1 - Sk)Uk + Sk Ureset
*   **Surrogate Gradient Module**: In the backward pass, the hard spike H(U - θ) is replaced by a smooth surrogate for its derivative. A common choice is a threshold-shifted sigmoid: H(U - θ) ≈ σ(U) = 1 / (1 + e^{θ - U}).
*   **Short-Memory Principle Module**: For efficiency, the sum in the charge phase is truncated to Σ_{m=max(0, k-M)}^{k-1}, i.e., a sliding memory window of fixed width M.

**7. All Mathematical Formulas and Symbol Definitions**

*   **Caputo Fractional Derivative (Definition 1)**: For a function y(t) defined over an interval [0, T], its Caputo fractional derivative of order α ∈ (0, 1] is given by:
    Dαy(t) := (1 / Γ(1-α)) ∫_0^t (t-τ)^{-α} y'(τ) dτ
*   **Integer-order ODE**: dy(t)/dt = f(t, y(t))
*   **Fractional-order ODE**: Dαy(t) = f(t, y(t))
*   **IF neuron dynamics**: τ dU(t)/dt = RIin(t)
*   **LIF neuron dynamics**: τ dU(t)/dt = -U(t) + RIin(t)
*   **Spike generation**: S(t) = H(U(t−) − θ)
*   **Reset**:
    1) soft reset: U(t+) ← U(t−) − θ
    2) hard reset: U(t+) ← Ureset
*   **Forward Euler discretization**: y_{k+1} = y_k + h f(t_k, y_k)
*   **IF (discrete)**: U_{k+1} = U_k + (hR/τ) I_{in,k}
*   **LIF (discrete)**: U_{k+1} = (1 - h/τ) U_k + (hR/τ) I_{in,k}
*   **Integer-order SNN iterations**:
    *   IF charge: U_k = U_{k-1} + X_k
    *   LIF charge: U_k = β U_{k-1} + X_k (where β := 1 - 1/τ)
    *   spike: S_k = H(U_k − θ)
    *   reset: (soft) U_k ← U_k − θ S_k or (hard) U_k ← (1− S_k)U_k + S_k U_{reset}
*   **f-IF neuron dynamics**: τ DαU(t) = RIin(t)
*   **f-LIF neuron dynamics**: τ DαU(t) = -U(t) + RIin(t)
*   **Fractional ABM predictor discretization**: y_k = y_0 + (1/Γ(α)) Σ_{j=0}^{k-1} μ_{j,k} f(t_j, y_j)
*   **ABM weight coefficients**: μ_{j,k} = (h^α / α) [(k − j)^α − (k − 1 − j)^α]
*   **Fractional discrete updates**:
    *   f-IF (discrete): U_k = U_0 + (R / τΓ(α)) Σ_{j=0}^{k-1} μ_{j,k} I_{in,j}
    *   f-LIF (discrete): U_k = U_0 + (1 / τΓ(α)) Σ_{j=0}^{k-1} μ_{j,k} (−U_j + RI_{in,j})
*   **Coefficient definition**: c^{(α)}_m = 1 / (τ^α α Γ(α)) [(m+1)^α − m^α]
*   **f-SNN iterations**:
    *   f-IF charge: U_k = U_0 + Σ_{m=0}^{k-1} c^{(α)}_m X_{k-m}
    *   f-LIF charge: U_k = U_0 + Σ_{m=0}^{k-1} c^{(α)}_m (-U_{k-1-m} + X_{k-m})
    *   spike: S_k = H(U_k − θ)
    *   reset: (soft) U_k ← U_k − θ S_k or (hard) U_k ← (1− S_k)U_k + S_k U_{reset}
*   **Proposition 1 (Long-memory Behavior)**: Under constant current Iin(t) ≡ Ic:
    *   U_{LIF}(t) = RI_c + [U_0 − RI_c] e^{−t/τ}
    *   U_{f-LIF}(t) = RI_c + (U_0 − RI_c) E_α(−t^α/τ)
    *   For 0 < α < 1, E_α exhibits (i) initial stretched–exponential decay and (ii) a power-law tail: E_α(− t^α / τ^α) ∼ (τ^α / Γ(1−α)) t^{−α} as t→∞
*   **Theorem 1 (Robustness of f-SNN)**:
    *   Membrane Potential Robustness:
        *   ∆U_{f-IF}(t) = (Rϵ / τΓ(α+1)) t^α
        *   ∆U_{IF}(t) = (Rϵ / τ) t
    *   Spike Timing Sensitivity:
        *   |∆t^{f-IF}_s| ∝ ϵ · I_c^{−(1+1/α)}
        *   |∆t^{IF}_s| ∝ ϵ · I_c^{−2}
*   **Theorem 2 (Irreducibility of f-IF Dynamics to Finite Classic LIF Ensembles)**: There exist no finite integer W, weights {ϕ_i}^W_{i=1}, and leak factors {β_i}^W_{i=1} such that:
    Û_k = Σ_{i=1}^W ϕ_i U^{LIF(β_i)}_k ≡ U^{f-IF}_k ∀ k
    The impulse response error is O(k^{α−1}).
*   **Corollary 1 (Spike Train Irreducibility)**: For any finite integer W, weights {ϕ_i}^W_{i=1} ⊂ R, leak factors {β_i}^W_{i=1} ⊂ (0, 1), and any Boolean function f : {0, 1}^W → {0, 1}, there exists an input sequence {X_k}_{k≥0} and threshold θ > 0 such that:
    S^{f-IF}_k ≠ f(S^{LIF(β_1)}_k, . . . , S^{LIF(β_W)}_k) for some k.

**8. Algorithm Pseudocode**

Based on the paper's Equation (13) and the described fractional iterations, the algorithm pseudocode for the forward pass is as follows:

```
Input: Input sequence X_k for k = 0, ..., T-1
Parameter: Fractional order α ∈ (0, 1], Threshold θ, Memory window M
Initialize: U_0

For k = 1 to T:
    // 1. Charge Phase
    // Calculate coefficient c^{(α)}_m = 1 / (τ^α α Γ(α)) [(m+1)^α − m^α]
    // Option A: Full Memory
    If f-IF:
        sum_term = Σ_{m=0}^{k-1} c^{(α)}_m X_{k-m}
    Else If f-LIF:
        sum_term = Σ_{m=0}^{k-1} c^{(α)}_m (-U_{k-1-m} + X_{k-m})
    // Option B: Short Memory (approximation)
    // Truncate sum: m ranges from max(0, k-M) to k-1
    
    U_k = U_0 + sum_term

    // 2. Spike Phase
    If U_k >= θ:
        S_k = 1
    Else:
        S_k = 0

    // 3. Reset Phase
    If Soft Reset:
        U_k ← U_k - θ S_k
    Else If Hard Reset:
        U_k ← (1 - S_k)U_k + S_k U_{reset}

Output: Spike train S_1, ..., S_T
```

Based on the Toolbox training logic described in Appendix E:

```
Model Initialization:
model = SNNWrapper(NetworkBackbone, integrator="fdeint", interpolation_method="linear")
model._set_neuron_shapes(input_shape)

Training Loop:
Define data_time = torch.linspace(0, time_interval * (time_steps - 1), time_steps)
Define output_time = torch.linspace(0, time_interval * (T - 1), T)
options = {'step_size': step_size}

For batch (data, target) in train_loader:
    optimizer.zero_grad()
    output = model(data, data_time, output_time, method, options)
    loss = criterion(output.mean(0), target)
    loss.backward() // Uses surrogate gradient for spiking threshold
    optimizer.step()
```