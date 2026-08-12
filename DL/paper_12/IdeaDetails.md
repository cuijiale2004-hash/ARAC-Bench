### 1. Research Background and Existing Pain Points

**Research Background:**
The error backpropagation algorithm has been fundamental to the success of deep learning, yet its core mechanisms are widely considered biologically implausible. Key limitations include the requirement for global error signals—where synaptic updates depend on information transmitted across multiple layers, far beyond locally available signals—and the reliance on continuous-valued communication and gradients, in contrast to the brain’s use of discrete, event-driven signals. These discrepancies create a significant gap between conventional artificial neural networks (ANNs) and biological neural systems. Artificial intelligence does not need to replicate biology, but certain biological properties are worth emulating, such as the brain’s energy efficiency and its ability to perform robust and adaptive computation with sparse, noisy, and low-precision signals. These observations align with the development of neuromorphic systems, which address the limitations of conventional von Neumann architectures by co-locating memory and computation to reduce data movement, thereby enabling substantially lower energy consumption. Within this hardware setting, Spiking Neural Networks (SNNs) provide a natural computational model where information is represented by discrete spikes. One promising framework for training such systems is Predictive Coding (PC). PC theorizes that the brain functions as a prediction machine, continuously generating top-down predictions of sensory input while bottom-up signals convey only the residual prediction errors. PC relies on local learning rules, where synaptic updates depend only on the activity of adjacent pre- and post-synaptic neurons, making it highly compatible with the parallel and distributed organization of neuromorphic hardware.

**Existing Pain Points:**
Despite its theoretical alignment with neuromorphic principles, the standard formulation of PC faces practical computational challenges. To infer neural activities, PC networks typically perform an iterative settling process, requiring multiple forward and backward passes of information to converge for a single input. In standard implementations, this process relies on dense, floating-point message passing, resulting in a computational overhead that notably exceeds that of backpropagation. This reliance on dense, continuous communication during the iterative phase can offset the efficiency gains sought by deploying SNNs on event-driven hardware. Therefore, while PC offers a solution to the global transport problem of backpropagation, its standard formulation does not fully exploit the sparsity and efficiency of neuromorphic substrates. Furthermore, existing works integrating PC with SNNs (e.g., PC-SNN) still rely on the transmission of dense floating-point numbers during the training stage, or they adopt a hybrid approach where a separate, non-spiking linear classifier is trained post-hoc on rate-coded activities, meaning the reported accuracies do not reflect the performance of an end-to-end spiking system.

### 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:**
To address the communication overhead of standard predictive coding and bridge the gap between PC theory and neuromorphic hardware efficiency by creating a "spike-native" PC formulation that exploits sparsity and event-driven computation.

**Scientific Questions:**
How to reformulate the predictive coding framework for native implementation in spiking neural networks such that computation and message passing are event-driven, occurring only when necessary to correct prediction errors? How to replace dense, floating-point message passing with sparse, ternary spike-based communication while ensuring that the discrete spiking network closely approximates the dynamics of standard continuous predictive coding with high precision and reduced data movement?

### 3. Overall Core Idea and Design Philosophy

**Overall Core Idea:**
Introduce Difference Predictive Coding (DiffPC), an algorithm that reformulates the predictive coding framework for native implementation in SNNs. DiffPC replaces dense, floating-point message passing with sparse, event-driven ternary spikes. By employing spike-compatible state update rules and adaptive threshold schedules, DiffPC ensures that computation and message passing are event-driven, occurring when necessary to correct prediction errors. 

**Design Philosophy:**
The design philosophy centers on communication efficiency and biological alignment. Instead of broadcasting the full state, DiffPC transmits incremental state updates via sparse ternary spikes. It employs difference-based update rules that trigger communication only upon state changes to minimize redundancy. It utilizes a cyclic threshold scheduler designed to accelerate the convergence of discrete spiking states toward continuous PCN targets, ensuring that the network remains silent during steady states and emits spikes only on changes.

### 4. Core Innovation Points

1.  **Spike-Native Framework:** Propose Difference Predictive Coding (DiffPC), a spike-native framework based on novel update rules that transmit incremental state updates via sparse ternary spikes rather than broadcasting the full state, resulting in reduced communication costs.
2.  **Adaptive Threshold Scheduling:** Introduce an adaptive threshold scheduling mechanism (including cyclic and cyclic decay schedulers) that enables the discrete spiking network to closely approximate the dynamics of standard continuous predictive coding with fewer timesteps.
3.  **Empirical Validation of Efficiency and Accuracy:** Empirically validate DiffPC on fully connected and convolutional architectures, demonstrating that it matches the accuracy of standard predictive coding and matches or exceeds that of Backpropagation on MNIST, Fashion-MNIST, and CIFAR-10, while reducing the number of transmitted bits by more than two orders of magnitude compared to standard predictive coding baselines.
4.  **Spike-based Message Passing Protocol:** A spike-based message passing protocol that adapts sparse ternary communication specifically for predictive coding error propagation.
5.  **Difference-based Update Rule:** A difference-based update rule that triggers communication only upon state changes to minimize redundancy.

### 5. Overview of the Overall Technical Solution

To adapt the predictive coding framework for spiking neural networks, its floating-point computations and information transfer must be converted into discrete spikes. In DiffPC, all information transmitted during the learning steps takes the form of a sequence of ternary values (-1, 0, 1). 

Each unit maintains a target state $x_T$ and an actual state $x_A$. The target state $x_T$ represents the desired (target) activity, while the actual state $x_A$ attempts to follow $x_T$, aiming to minimize the difference between them. Their difference is reduced incrementally by steps proportional to an adaptive threshold $T_\theta$. The difference-based adjustments are communicated as spikes to subsequent layers, which integrate the incoming spiking information.

Two error variables, $e_T$ and $e_A$, function in the same way and represent the errors of the two states. $e_T$ is the target error and $e_A$ is the actual error that attempts to align with $e_T$. The error $e_A$ is adjusted by steps proportional to $T_\theta$, which are then transmitted to subsequent layers as spikes. Since the threshold $T_\theta$ determines the step size of these updates, its schedule is critical for convergence.

Before the iterative DiffPC process begins, the network undergoes a feed forward initialization phase where input spikes are propagated through the layers in a single pass without feedback error calculation. This rapidly establishes an initial estimate for the target activities $x_T$ and predictions $x_F$, reducing the number of subsequent iterative steps required for convergence.

### 6. Detailed Module Design

**1. Feed Forward Initialization Module:**
Input spikes are propagated through the layers in a single pass without feedback error calculation. This rapidly establishes an initial estimate for the target activities $x_T$ and predictions $x_F$. This phase effectively mimics a standard feedforward SNN inference step to prime the network state and can be implemented by utilizing graded spikes on the Loihi 2 chip.

**2. Update Forward Prediction Module:**
Forward prediction $x_F$ is updated using the incoming spike signal $s_A$ from the previous layer:
$x_F^{(t+1)} \leftarrow x_F^{(t)} + W_l s_{A}^{l-1} \cdot T_\theta(t-1)$

**3. Update Target Activity and Generate Spikes Module:**
The core of the DiffPC algorithm lies in iteratively updating the target activity vector $x_T$ for each layer and communicating changes via spikes. The process begins by adjusting $x_T$ to minimize prediction error, followed by generating spikes based on the discrepancy between this new target and the layer’s current state.

First, the target activity $x_T$ is updated based on both the local prediction error $e_T$ and the error propagated from the subsequent layer, which is accumulated in $e_B$. This update, performed only when the learning rate $\gamma(t)$ is active, is defined as:
$x_T \leftarrow x_T + \gamma(t) \cdot (-e_T + (x_T > 0) \odot e_B)$
This rule closely mirrors the standard PC update. The term $-e_T$ corrects for local prediction error, while the second term incorporates feedback from the next layer. The element-wise condition $(x_T > 0)$ serves as the derivative of the ReLU activation function, ensuring that updates are only applied to active neurons. During training, the target activities of the input and output layers are clamped to the provided data and labels, respectively. During inference, only the input layer is clamped.

Next, the algorithm generates spikes to communicate the necessary adjustments for bringing the layer’s actual state, $x_A$, in line with the newly updated target state, $x_T$. Instead of transmitting dense floating-point values, DiffPC sends sparse ternary spikes. A spike is generated only if the magnitude of the difference between the target and actual activity for a given neuron exceeds the adaptive threshold $T_\theta(t)$. The activity spike vector $s_A$ is computed as follows:
$s_A = \text{sign}(x_T - x_A) \odot (|x_T - x_A| > T_\theta(t))$
where $\odot$ denotes the Hadamard product. The $\text{sign}(\cdot)$ function determines the spike’s polarity (+1 or -1), while the comparison operator produces a binary mask, ensuring that spikes are only generated when the required update is significant. This event-driven mechanism ensures that communication is sparse.

**4. Spiking ReLU Module:**
To implement a non-linear transfer function similar to the Rectified Linear Unit (ReLU) in conventional neural networks, a masking operation effectively prevents the actual neural activity $x_A$ from becoming negative:
$s_A^+ \leftarrow s_A \odot (x_A + s_A \cdot T_\theta(t) > 0)$
The spiking ReLU ensures that only correction spikes in $s_A$ maintaining $x_A \geq 0$ are allowed, effectively implementing a ReLU-like activation function. A clipped ReLU activation function can be implemented in a similar manner by setting an additional constraint:
$s_A^+ \leftarrow s_A \odot (1 > x_A + s_A \cdot T_\theta(t) > 0)$

**5. Update Activation Module:**
The spiking ReLU produces a simple update step to update the actual activity $x_A$:
$x_A \leftarrow x_A + T_\theta(t) \cdot s_A^+$
The activation update operation adjusts $x_A$ towards $x_T$.

**6. Error Encoding in Spikes Module:**
The target error vector $\epsilon_l$ is computed. Similarly to the original PC, this error term received from the following layer serves as a measure of how well the current forward prediction matches the target activity. However, unlike standard PC, the update rule cannot be used directly since the error is encoded in the form of spikes. Instead, after computing the target error $e_T = \epsilon_l$, the DiffPC algorithm generates error spikes $s_e$ based on the difference between $e_T$ and the actual error $e_A$:
$s_e = \text{sign}(e_T - e_A) \odot (|e_T - e_A| > T_\theta(t))$
The spikes $s_e$ are then sent to the following layers as error signals.

**7. Accumulate Incoming Errors Module:**
Errors from the next layers are integrated into the network using the accumulated error vector $e_B$. This vector represents the sum of the incoming error signals weighted by the threshold $T_\theta(t)$:
$e_B \leftarrow e_B + T_\theta(t) \cdot e^{in}$
where $e^{in}$ denotes the incoming error signals from the next layer. The accumulated error vector $e_B$ helps refine the target activity $x_T$ by incorporating feedback from different layers of the network. The actual error $e_A$ is then updated using previously generated error spikes $s_e$:
$e_A \leftarrow e_A + T_\theta(t) \cdot s_e$

**8. Weight Update Mechanism Module:**
The update rule for a single sample, derived from minimizing free energy, is computed as $\Delta W_{ij} \propto e_{T,i} \cdot \phi(x_{T,j})$, where $e_{T,i}$ is the post-synaptic error state and $x_{T,j}$ is the pre-synaptic activity state. This computation is compatible with neuromorphic hardware like Loihi 2, which supports fixed-precision multiplication and accumulation of local variables.

### 7. All Mathematical Formulas and Symbol Definitions

**Spiking Neural Network Model (Integrate-and-Fire):**
$V_i(t+1) = V_i(t) - T_\theta s_i(V_i(t)) + \sum_j w_{ij} s_j(t), \quad V_i(0) = b_i$
$s_i(V_i(t)) := \begin{cases} 1 & \text{if } V_i(t) \geq T_\theta, \\ -1 & \text{if } V_i(t) \leq -T_\theta, \\ 0 & \text{otherwise} \end{cases}$

**Predictive Coding Energy Function and Updates:**
$x_{F,l} = W_l \phi(x_{T,l-1})$
$\epsilon_l = x_{T,l} - x_{F,l}$
$F = \sum_{l=1}^L \|\epsilon_l\|_2^2 = \sum_{l=1}^L (x_{T,l} - x_{F,l})^2$
$W_l^{(t+1)} = W_l^{(t)} - \alpha \frac{\partial F}{\partial W_l} = W_l^{(t)} + \alpha \epsilon_l \phi(x_{T,l-1})^\top$
$x_{T,l}^{(t+1)} = x_{T,l}^{(t)} - \gamma \frac{\partial F}{\partial x_{T,l}} = \begin{cases} x_{T,l}^{(t)} - \gamma (\epsilon_l - W_{l+1}^\top \epsilon_{l+1} \odot \phi'(x_{T,l})), & \text{for } l < L, \\ x_{T,l}^{(t)} - \gamma \epsilon_l, & \text{for } l = L \end{cases}$

**DiffPC Scheduler Formulas:**
Cyclic scheduler:
$T_\theta(t) = \frac{2^m}{2^{t \bmod n}}, \quad \gamma(t) = g(t \bmod n)$
$g(x) = \begin{cases} \gamma, & \text{if } x = 0 \\ 0, & \text{otherwise} \end{cases}$

Cyclic decay scheduler:
$T_\theta(t) = d(t \bmod T) \frac{2^m}{2^{t \bmod n}}, \quad \gamma(t) = g(t \bmod n)$
$d(t) = \left(1 - (1-a)^{\frac{t}{T}}\right)$

Constant decay scheduler:
$T_\theta(t) = d(t \bmod T) c, \quad \gamma(t) = g(t \bmod n)$

**DiffPC Algorithm Breakdown Formulas:**
Update forward prediction:
$x_F^{(t+1)} \leftarrow x_F^{(t)} + W_l s_{A}^{l-1} \cdot T_\theta(t-1)$
Generate activity spikes:
$s_A = \text{sign}(x_T - x_A) \odot (|x_T - x_A| > T_\theta(t))$
Spiking ReLU:
$s_A^+ \leftarrow s_A \odot (x_A + s_A \cdot T_\theta(t) > 0)$
Clipped Spiking ReLU:
$s_A^+ \leftarrow s_A \odot (1 > x_A + s_A \cdot T_\theta(t) > 0)$
Update activation:
$x_A \leftarrow x_A + T_\theta(t) \cdot s_A^+$
Generate error spikes:
$s_e = \text{sign}(e_T - e_A) \odot (|e_T - e_A| > T_\theta(t))$

**Theorem 4.1 (Convergence):**
Suppose that the target activity for layer $l$, denoted as $x_{T,l}$, satisfies $|x_{T,l} - x_{A,l}| < 2^{m+1}$ and $x_{T,l} > 0$. Then, after one $\gamma$-cycle of $n$ timesteps of a cyclic scheduler, the difference between the target activity and the actual activity $|x_{T,l} - x_{A,l}|$ is less than $2^{m+1-n}$.

**Symbol Definitions:**
$l$: Layer index, $l \in \{0, \dots, L\}$.
$W_l$: Synaptic weight matrix connecting layer $l-1$ to $l$.
$x_F$: Forward Prediction. The prediction generated by the previous layer.
$x_T$: Target Activity. The ideal state calculated to minimize prediction energy.
$\epsilon$: Prediction Error. The difference between target and prediction ($x_T - x_F$).
$\gamma(t)$: Inference learning rate at time $t$.
$x_A$: Actual Activity. The discrete state that tracks the shared target $x_T$, updated via spikes.
$s_A$: Activity Spikes. Ternary spikes $\{-1, 0, 1\}$ communicating changes in $x_A$.
$e_T$: Target Error. The local error variable (functionally equivalent to $\epsilon$ in this context).
$e_A$: Actual Error. The discrete state that tracks $e_T$, updated via spikes.
$s_e$: Error Spikes. Ternary spikes communicating changes in the error state.
$e_B$: Backward Error. The error signal accumulated from layer $l+1$.
$T_\theta(t)$: Adaptive firing threshold at time $t$.
$m$: Scheduler magnitude parameter (sets max threshold $2^m$).
$n$: Scheduler cycle length (periodicity of the steps).
$a$: Decay factor for the cyclic decay scheduler.

### 8. Algorithm Pseudocode

**Algorithm 1 DiffPC Algorithm for Spiking Neural Network Training**
Input: Spike signals $s_{in}$, $s_e$
Process parameters: threshold $T_\theta(t)$, learning rate $\gamma(t)$, weight lr $\alpha$
Initialize: $x_F$, $x_T$, $x_A$, $e_T$, $e_A$, $e_B$, $s_A$, $s_e$
1: Feed Forward Initialization: Propagate input to prime $x_F$
2: for each time step $t$ do
3: $\quad s_{in}^l \leftarrow W^l s_A^{l-1}$ \hspace{1cm} $\triangleright$ Receive spike input
4: $\quad x_F^l \leftarrow x_F^l + s_{in}^l \cdot T_\theta(t-1)$ \hspace{1cm} $\triangleright$ Update forward prediction
5: $\quad$ if $\gamma^l(t) > 0$ then
6: $\quad\quad e_T^l \leftarrow x_T^l - x_F^l$ \hspace{1cm} $\triangleright$ Compute target error
7: $\quad\quad x_T^l \leftarrow x_T^l + \gamma^l(t) \cdot (-e_T^l + (x_T^l > 0) \odot e_B^l)$ \hspace{1cm} $\triangleright$ Update Target Activity
8: $\quad\quad$ if $\gamma^l(t) > 0$ then
9: $\quad\quad\quad e_T^l \leftarrow x_T^l - x_F^l$ \hspace{1cm} $\triangleright$ Update target error
10: $\quad\quad s_A^l \leftarrow \text{sign}(x_T^l - x_A^l) \odot (|x_T^l - x_A^l| > T_\theta(t))$ \hspace{1cm} $\triangleright$ Generate spikes
11: $\quad\quad s_A^l \leftarrow s_A^l \odot (x_A^l + s_A^l \cdot T_\theta(t) > 0)$ \hspace{1cm} $\triangleright$ 'Spiking ReLU'
12: $\quad\quad x_A^l \leftarrow x_A^l + T_\theta(t) \cdot s_A^l$ \hspace{1cm} $\triangleright$ Update Actual Activity
13: $\quad\quad s_{out}.send(s_A^l)$ \hspace{1cm} $\triangleright$ Send state spikes
14: $\quad\quad$ $\triangleright$ Propagate Error: $\lhd$
15: $\quad\quad s_{e}.out.send(s_e^l)$ \hspace{1cm} $\triangleright$ Send error spikes
16: $\quad\quad e_{in}^l \leftarrow (W^{l+1})^\top s_e^{l+1}$ \hspace{1cm} $\triangleright$ Receive error input
17: $\quad\quad e_B^l \leftarrow e_B^l + T_\theta(t-1) \cdot e_{in}^l$ \hspace{1cm} $\triangleright$ Accumulate incoming errors
18: $\quad\quad s_e^l \leftarrow \text{sign}(e_T^l - e_A^l) \odot (|e_T^l - e_A^l| > T_\theta(t))$ \hspace{1cm} $\triangleright$ Generate error spikes
19: $\quad\quad e_A^l \leftarrow e_A^l + T_\theta(t) \cdot s_e^l$ \hspace{1cm} $\triangleright$ Update actual error
20: $\quad\quad W^l \leftarrow W^l + \alpha e_T^l \phi(x_{T}^{l-1})^\top$ \hspace{1cm} $\triangleright$ Update Weights