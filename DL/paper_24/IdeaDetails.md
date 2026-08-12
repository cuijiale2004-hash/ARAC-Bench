1. Research Background and Existing Pain Points
Deep spiking neural networks (SNNs) hold immense promise for low-power event-driven computing, transmitting information via discrete spikes rather than continuous activations. To train a deep SNN end-to-end, the temporal dimension is discretized into T time steps so that the SNN can be considered as a binary-activated recurrent neural network (RNN). Then, backpropagation through time (BPTT) is adopted to compute parameter updates, with surrogate gradient (SG) tackling the non-differentiable spike emission process. Despite its high accuracy and broad applicability, BPTT imposes intensive memory overhead. For an L-layer SNN unfolded over T time steps, BPTT requires O(LT) memory to store intermediate states, compared to O(L) for a structurally similar ANN. Consequently, SNN direct training is more likely to exceed the memory capacity of computational devices. The scaling of SNNs to deeper architectures and more time steps is thus severely hindered. A key feature of SNN direct training is that intermediate features (inputs and internal states) dominate memory usage, occupying over 96% of the memory at peak usage for SNNs, compared to about 77% for ANNs, because SNNs' T time steps scale intermediate feature sizes by O(T).

Existing memory-saving approaches compromise accuracy, training speed, or applicability. Online learning truncates temporal gradients and stores only the intermediate results at the current step. However, the gradient mismatch results in a severe performance drop on temporal tasks, and its step-wise running mode undermines its compatibility with widely adopted temporal parallelization techniques like parallel spiking neuron (PSN). BPTT-to-BP approximation trains an SNN by backpropagating through a static proxy based on firing rates, effectively removing the temporal gradient dimension. Despite its memory and time efficiency, BPTT-to-BP can hardly handle sequential data due to the neglect of temporal information, thus limiting its applicability. Reversible networks reconstruct intermediate features reversely during backward pass rather than storing them. It preserves BPTT-level accuracy, but imposes strict architectural constraints and significantly slows training. Furthermore, these methods require manual architectural modifications or training code rewrites, which are error-prone and cumbersome.

2. Core Research Motivation and Scientific Questions
The core research motivation is to address the prohibitive memory cost of BPTT-based SNN direct training that limits the scalability of SNNs, without compromising accuracy, training speed, or broad applicability. There is a critical need for a broadly applicable and user-friendly solution that improves the memory efficiency of SNN direct training while preserving training speed and performance. 

The scientific questions are: How to eliminate the dominant memory consumers (internal states and input spikes) in SNN training without introducing gradient bias? How to adaptively refine checkpoint placement based on profiling results to further optimize global peak memory usage and improve training speed? How to encapsulate the optimization strategies into an automatic pipeline that automatically restructures the computation flow before training with minimal user effort?

3. Overall Core Idea and Design Philosophy
The overall core idea is to propose a novel and broadly applicable pipeline for memory-efficient SNN training that preserves BPTT’s accuracy by integrating layer-wise gradient checkpointing with lossless spike compression. The design philosophy centers on trading computation for memory while maintaining mathematical equivalence to standard BPTT. Layer-wise gradient checkpointing is applied to eliminate internal state storage, and lossless spike compression is utilized to reduce the memory cost of per-layer input spikes. To further optimize peak memory usage, a multi-stage checkpoint adjustment strategy is introduced that adaptively refines checkpoint placement based on profiling results, including spatial segment partitioning, temporal segment partitioning, and greedy segment restoration. The entire process is wrapped in an optimization pass that automatically reconfigures the computation flow before training, requiring minimal user intervention.

4. Core Innovation Points
(1) Memory cost analysis: The paper analyzes the memory cost of SNN direct training and identifies input spikes and internal states as primary memory consumers.
(2) An automatic pipeline: The paper proposes a broadly applicable pipeline that integrates gradient checkpointing with spike compression for memory-efficient SNN training.
(3) Multi-stage checkpoint adjustment strategy: The paper introduces a multi-stage checkpoint adjustment strategy that adaptively refines checkpoint placement, including spatial segment partitioning, temporal segment partitioning, and greedy segment restoration, to optimize memory usage and improve training speed.
(4) Memory-efficient LIF kernel: The paper designs a memory-efficient LIF (MELIF) kernel that avoids storing output spikes by reconstructing them during the backward pass.

5. Overview of the Overall Technical Solution
The overall technical solution begins with a memory cost analysis of BPTT-based SNN direct training, formulating the upper bound for peak memory. To reduce this cost, layer-wise gradient checkpointing is applied to each layer, storing only the input and weight during the forward pass and reconstructing internal states during the backward pass. Furthermore, lossless input spike compression is employed, storing the compressed form of input spikes during the forward pass and decompressing them when needed in the backward pass. To further optimize global peak memory, the gradient checkpointing structure is adjusted through three strategies: spatial segment partitioning (splitting the critical segment along the layer dimension), temporal segment partitioning (splitting the critical segment along the time axis), and greedy segment restoration (reverting GC segments with local memory cost well below global peak to standard BPTT blocks to accelerate training). All these strategies are encapsulated into an automatic pipeline with different optimization levels (O1 to O4). Additionally, a memory-efficient LIF kernel is designed to avoid storing output spikes by reconstructing them during the backward pass.

6. Detailed Module Design
Layer-wise Gradient Checkpointing: In standard BPTT, all internal states must be stored, resulting in a memory cost of up to $\sum_l M_{\Omega_l}$. To reduce this cost, gradient checkpointing (GC) is applied to each layer $l \in \{1, \ldots, L\}$. During the forward pass on layer $l$, only the input $S_{l-1}$ and weight $W_l$ are stored. In the backward pass, internal states $\Omega_l$ are reconstructed through an extra local forward pass given $S_{l-1}$ and $W_l$. With GC, $\Omega_l$ is allocated and freed during layer $l$'s backward pass. Thus, at most one layer's internal states are stored in memory at any given time.

Lossless Input Spike Compression: Input spikes $S_{l-1}$ must be stored even if GC is applied. Instead of storing $S_{l-1}$ as floats during the forward pass, its compressed form $\tilde{S}_{l-1}$ is stored. $\tilde{S}_{l-1}$ is decompressed to $S_{l-1}$ when needed in the backward pass. The spike compressor must be lossless to ensure computational equivalence with standard BPTT. Bit representation uses 1 bit per binary value, achieving up to 32× compression over 32-bit floats. Alternatives include sparse representation and lossless bit stream compressors like Zstandard and asymmetric numeral systems (ANS). Bit representation is chosen by default as it is faster and more memory-saving than alternatives in most cases. To further accelerate compression and decompression, handcrafted Triton kernels are used. Compression is skipped for non-binary inputs (e.g., $S_0$). For non-binary integer activations (e.g., in SEW residual connections), they are compressed into 8-bit unsigned integers (uint8).

Adjusting Gradient Checkpointing Structure: After layer-wise GC and spike compression, the global peak memory $M_{peak}$ achieved at the critical layer far exceeds the local peaks elsewhere. Model trainability on specific devices depends only on this global peak. 
- Spatial Segment Partitioning: To reduce $M_{peak}$, the GC segment $l^*$ with the largest peak memory is identified, and a spatial checkpoint is inserted within it. In other words, $l^*$ is split along the layer dimension into two spatial subsegments $l_1^*$ and $l_2^*$. Since $M_{\Omega_{l^*}} > \max\{M_{\Omega_{l_1^*}}, M_{\Omega_{l_2^*}}\}$, a reduction of $\max_l M_{\Omega_l}$ is guaranteed. This process repeats until $M_{peak}$ cannot further decrease.
- Temporal Segment Partitioning: Temporal partitioning similarly finds the critical segment $l^*$ and splits it along the time axis into $k$ sequential temporal subsegments. Each temporal subsegment checkpoints both its inputs and initial hidden states to enable recomputation during the backward pass. Temporal partitioning is applied conservatively after spatial partitioning as a complementary strategy, since splitting segments along time disables temporal parallelism and limits temporal kernel fusion, resulting in restricted applicability and slower training. The procedure repeats until $M_{peak}$ cannot be further reduced.
- Greedy Segment Restoration: For GC segments whose local memory cost is well below $M_{peak}$, they can be safely reverted to standard BPTT blocks (i.e., storing all intermediate features) without increasing $M_{peak}$. Since GC segments require an extra forward pass for recomputation, restoring them accelerates training. Specifically, the forward time cost of each GC segment is profiled, and then the segments with the largest time cost are greedily restored. The change is kept only if $M_{peak}$ does not increase. This process terminates after all segments are considered.

Automatic Pipeline: To minimize user intervention, all the above strategies are wrapped into an automatic pipeline. Users can set the level parameter to specify the applied strategy set. At level O1, only layer-wise GC and spike compression are enabled; O2 additionally applies spatial segment partitioning; O3 further incorporates temporal partitioning; O4 additionally activates greedy segment restoration.

Memory-Efficient LIF Kernel: The BPTT formulation of LIF requires storing only $\{H_l[t]\}_{t=1}^T$ and $\{S_l[t]\}_{t=1}^T$ during the forward pass. The kernel further avoids storing $\{S_l[t]\}_{t=1}^T$ by reconstructing it during the LIF layer's backward pass through $S_l[t] = \Theta(H_l[t] - V_{th})$. In this way, the floating-point spikes can be dropped once their compression at the subsequent layer is done.

7. All Mathematical Formulas and Symbol Definitions
Discrete-time dynamics of a L-layer SNN composed of LIF neurons:
$X_l[t] = g_l(S_{l-1}[t]; W_l),$
$H_l[t] = \lambda V_l[t-1] + X_l[t],$
$S_l[t] = \Theta(H_l[t] - V_{th}),$
$V_l[t] = H_l[t](1 - S_l[t]).$
where $l \in \{1, \ldots, L\}$ is the layer index, and $t \in \{1, \ldots, T\}$ is the time step index. $X$ is the input current, $H$ and $V$ are the membrane potentials before and after spike emission, and $S$ is the output spike. $X_l[t]$ can be computed from the previous layer's output $S_{l-1}[t]$ via a linear transformation $g_l$ with weight $W_l$. $\lambda \in (0, 1)$ is the decay factor, $V_{th} > 0$ is the firing threshold, and $\Theta(x)$ is the Heaviside step function.

Upper bound for the peak memory of standard BPTT:
$M_{peak}^{BPTT} \le \sum_l (M_{W_l} + M_{G_l} + M_{\Lambda_l} + M_{S_{l-1}} + M_{\Omega_l}) + \max_l M_{R_l},$
where $M_{W_l}, M_{G_l}, M_{\Lambda_l}, M_{S_{l-1}}, M_{\Omega_l},$ and $M_{R_l}$ are the memory consumptions of the weights, gradients, optimizer states, inputs, internal states, and runtime variables at layer $l$, respectively.

Peak memory's upper bound with Layer-wise GC:
$M_{peak}^{GC} \le \sum_l (M_{W_l} + M_{G_l} + M_{\Lambda_l} + M_{S_{l-1}} + \mathbb{1}\mathbb{H}M_{\Omega_l}) + \max_l (M_{\Omega_l} + M_{R_l}).$

Peak memory's upper bound with GC and Spike Compression:
$M_{peak}^{GC+Comp} \le \sum_l (M_{W_l} + M_{G_l} + M_{\Lambda_l} + \mathbb{X}\mathbb{X}\mathbb{X}M_{S_{l-1}} + M_{\tilde{S}_{l-1}}) + \max_l (M_{\Omega_l} + M_{R_l}).$

BPTT formulation of LIF:
$\frac{\partial L}{\partial X_l[t]} = \left( \frac{\partial L}{\partial S_l[t]} - \frac{\partial L}{\partial V_l[t]_{H_l[t]}} \right) \Theta'_{sg}(H_l[t] - V_{th}) + \frac{\partial L}{\partial V_l[t]} (1 - S_l[t]),$
$\frac{\partial L}{\partial V_l[t-1]} = \lambda \frac{\partial L}{\partial X_l[t]},$
where $L$ is the loss and $\Theta'_{sg}$ is the surrogate gradient function.

Memory evolution sequence:
$[M_{start}^{FP_1}, M_{peak}^{FP_1}, M_{end}^{FP_1}, M_{start}^{FP_2}, M_{peak}^{FP_2}, M_{end}^{FP_2}, \ldots, M_{start}^{BP_2}, M_{peak}^{BP_2}, M_{end}^{BP_2}, M_{start}^{BP_1}, M_{peak}^{BP_1}, M_{end}^{BP_1}]$
The global peak memory is defined as $M_{peak} = \max(\{M_{peak}^{FP_l}\}_l \bigcup \{M_{peak}^{BP_l}\}_l)$.

Gradient accumulation without temporal partitioning:
$G = \sum_{t=1}^T \sum_{n=1}^N G_{t,n},$
Gradient accumulation with temporal partitioning ($k=2$):
$G^{(1)} = \sum_{t=1}^{T/2} \sum_{n=1}^N G_{t,n}, \quad G^{(2)} = \sum_{t=T/2+1}^T \sum_{n=1}^N G_{t,n},$
$G = G^{(1)} + G^{(2)}.$

8. Algorithm Pseudocode
Algorithm 1 One iteration of SNN training with layer-wise GC and spike compression.
Input: parameters $\{W_l\}_{l=1}^L$; network input $S_0$; compressor $C(\cdot)$; other hyperparameters.
Output: trained parameters $\{W_l\}_{l=1}^L$.
1: // forward pass
2: for $l = 1, 2, \ldots, L$ do
3:  $S_l \leftarrow \text{layer}_l(S_{l-1}; \text{autograd} = \text{False})$;
4:  if $S_{l-1}$ is binary then
5:   Compress: $\tilde{S}_{l-1} \leftarrow C(S_{l-1})$;
6:   Save $\tilde{S}_{l-1}$, and free $S_{l-1}$;
7:  else
8:   Save $S_{l-1}$;
9:  end if
10: end for
11: Compute the loss $L$ and the gradient $\frac{\partial L}{\partial S_L}$;
12: // backward pass
13: for $l = L, L-1, \ldots, 1$ do
14:  if $S_{l-1}$ is compressed then
15:   Decompress: $S_{l-1} \leftarrow C^{-1}(\tilde{S}_{l-1})$;
16:  end if
17:  $S_l \leftarrow \text{layer}_l(S_{l-1}; \text{autograd} = \text{True})$;
18:  Compute $\frac{\partial L}{\partial W_l}, \frac{\partial L}{\partial S_{l-1}}$ by BPTT;
19:  Free the saved tensors of layer $l$;
20: end for
21: Update the parameters $\{W_l\}_{l=1}^L$.

Algorithm 2 GC structure adjustment.
Input: A list of GC segments $\Psi = [\text{seg}_l]_{l=1}^L$.
Output: the adjusted GC segment list.
1: // spatial partitioning
2: while True do
3:  Find $l^* = \text{argmax}_l(M_{peak}^l)$;
4:  Spatially split: $\text{seg}_{l^*} \rightarrow \{\text{seg}_{l_1^*}, \text{seg}_{l_2^*}\}$
5:  if global $M_{peak}$ doesn't decrease then
6:   Revert the split; break;
7:  end if
8: end while
9: // temporal partitioning
10: while True do
11:  Find $l^* = \text{argmax}_l(M_{peak}^l)$;
12:  Temporally split: $\text{seg}_{l^*} \rightarrow \{\text{seg}_{l_i^*}\}_{i=1}^k$;
13:  if global $M_{peak}$ doesn't decrease then
14:   Revert the split; break;
15:  end if
16: end while
17: // greedy restoration
18: sort $\Psi$ descendingly by forward time cost;
19: for $\text{seg}_l$ in $\Psi$ do
20:  Restore $\text{seg}_l$ to a BPTT segment;
21:  if global $M_{peak}$ increases then
22:   Re-enable GC for $\text{seg}_l$;
23:  end if
24: end for