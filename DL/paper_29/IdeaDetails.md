### 1. Research Background and Existing Pain Points

**Research Background:**
Continuous learning of novel classes is crucial for edge devices to preserve data privacy and maintain reliable performance in dynamic environments. Intelligent edge devices sense dynamic contexts in their surrounding environment and make decisions based on sensor data. The high labeling cost and privacy constraints make it difficult to acquire sufficient data for continual batch learning. To this end, on-device Few-Shot Class-Incremental Learning (FSCIL) is introduced to incrementally learn from newly collected sensory data with scarce labeled samples while preserving existing knowledge. Spiking Neural Networks (SNNs) process spatiotemporal information efficiently, offering lower energy consumption, greater biological plausibility, and compatibility with neuromorphic hardware than ANNs, as neurons fire only upon event-driven activation, reducing unnecessary computation cost.

**Existing Pain Points:**
1. **Catastrophic forgetting and Overfitting:** On-device FSCIL faces two significant challenges. Catastrophic forgetting reflects the model’s difficulty learning new tasks while retaining information of originally learned tasks. Overfitting occurs when the model memorizes limited training samples, leading to poor generalization.
2. **Ineffectiveness under limited memory:** Recent advanced solutions tackle these challenges by parameter-efficient fine-tuning (PEFT), which fine-tunes parameters on top of a large frozen pre-trained model. Nonetheless, these approaches remain ineffective under limited memory budgets (i.e., 4–12 GB).
3. **Static and offline on-device SNNs:** Concurrent on-device SNNs concentrate on integrating SNN algorithms with specialized neuromorphic hardware, which shifts the consideration of the above problems to the later model. Additionally, most existing neuromorphic systems are still trained offline and remain static during development, presenting a heavy dependency on abundant labeled data.
4. **Resource constraints on edge devices:** Smart devices are severely constrained in terms of memory capacity and maximum performance, extremely scarce labeled samples, and the persistent need for model updates. Edge devices are incapable of accumulating comprehensive data on all classes due to limited storage capacity and must simultaneously perform low-power inference while continuously learning from real-time data streams.
5. **Surrogate gradient limitations in SNNs:** SNNs present challenges in terms of the non-differentiable activation function. Surrogate gradient (SG) methods deviate significantly from the true gradients and easily cause vanishing gradient problems.

### 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:**
To tackle the challenges of catastrophic forgetting, overfitting, and resource constraints in general on-device FSCIL scenarios, this work explores if the regulation of neuronal dynamics can help the trade-off between plasticity and stability in dealing with FSCIL tasks. Inspired by the regularized lottery ticket hypothesis and the interconnected nature of biological neural networks forming different subnetworks by synapse weights, the motivation is to explore sparsity-aware dynamic thresholds that shift networks to task-specific local contexts. Additionally, since prototypes of novel classes still tend to deviate to the base one causing bias, there is a need for a subspace projection scheme that represents each new class with an orthogonal subspace basis and projects onto the most similar base classes.

**Scientific Questions:**
1. Can the regulation of neuronal dynamics help the trade-off between plasticity and stability in dealing with FSCIL tasks?
2. How to expand the network structure via sparsity-aware dynamic thresholds so that most neurons remain stable to preserve base knowledge while others adapt to incremental learning?
3. How to solve the problem of non-differentiable spikes in backpropagation while avoiding the limited width and gradient vanishing issues of surrogate gradients?
4. How to mitigate the prototype bias caused by few-shot data in incremental learning by enhancing prototype discriminability without introducing heavy trainable modules?

### 3. Overall Core Idea and Design Philosophy

**Overall Core Idea:**
Propose an SNN-based method containing Sparsity-Aware neuronal dynamics and Fast Adaptive structure (SAFA-SNN) for on-device FSCIL. By threshold regulation, most neurons exhibit stable spikes and others exhibit adaptive spikes. As a result, synaptic traces that encode base-class knowledge are naturally preserved, thereby alleviating catastrophic forgetting. To cope with spike non-differentiability in backpropagation, employ a gradient-free technique, i.e., zeroth-order optimization. Moreover, class prototypes can limit overfitting on few-shot data but introduce bias. Enhance prototype discriminability by orthogonal subspace projection.

**Design Philosophy:**
The design philosophy centers on implementing efficient heterogeneous plasticity in SNNs via a dynamic threshold function. Following a homeostatic plasticity rule, stable neurons maintain firing rates close to the base-class training, preserving acquired knowledge, while adaptive neurons facilitate knowledge updating for novel classes. This creates a clear distinction between neurons encoding base classes and those encoding novel classes, implementing subnet regularization without additional trainable parameters. For gradient estimation, zeroth-order optimization accommodates the influence of all neighbors for a balanced depiction of local dynamics. For prototype bias, the philosophy is to project the prototypes onto the subspace spanned by base classes and blend the projected prototype with the original novel prototype through a convex combination, ensuring features with higher similarity contribute more to the prototype update.

### 4. Core Innovation Points

1. **Emphasis on practical on-device FSCIL challenges with SNNs:** The paper emphasizes practical challenges of on-device FSCIL, where both training and inference are performed on resource-constrained devices using energy-efficient, hardware-friendly SNNs, contrasting with ANN-based FSCIL that depends on large-scale server models.
2. **Sparsity-Aware neuronal dynamics:** Incorporating spikes for prediction, the paper proposes the Sparsity-Aware dynamics for SNN. By threshold regulation, most neurons exhibit stable spikes and others exhibit adaptive spikes, preserving base-class knowledge synaptic traces and minimizing updates to the majority of synapses.
3. **Zeroth-order optimization for non-differentiable spikes:** To solve the problem of non-differentiable spikes in back propagation, the paper adopts zeroth-order optimization within neuronal dynamics. This approximates true gradients using only function values, addressing stochastic optimization for the non-differentiable spike activation in a non-convex setting.
4. **Fast Adaptive Prototype Subspace Projection:** Enhancing the discriminability of prototypes by subspace projection in incremental learning. Projecting novel-class prototypes onto the orientation subspace spanned by base-class prototypes and blending them via convex combination mitigates the bias where prototypes learned from scarce samples deviate to the base classes.
5. **First SNN-based solution towards general on-device FSCIL:** To the best of the authors' knowledge, this is the first SNN-based solution towards general on-device FSCIL, validated with practical implementation on realistic devices (Jetson Orin AGX).

### 5. Overview of the Overall Technical Solution

The SAFA-SNN framework for on-device FSCIL comprises three main parts: a sparsity-aware dynamic mechanism to mitigate forgetting, the formulation of zeroth-order optimization for non-differentiable spikes, and a prototype subspace projection for fast adaptation on few-shot new classes.
- **Training abundant base data:** Neurons are divided into active and stable neurons by masks. Sparsity-aware neuronal dynamics are applied where thresholds of adaptive neurons change more freely than stable neurons. Backpropagation uses zeroth-order optimization.
- **Few-shot class incremental learning:** The backbone is frozen. The thresholds are updated dynamically based on sparsity differences. Prototypes for new classes are computed and then updated via orthogonal subspace projection using the subspace spanned by base classes, integrating base and new knowledge.

### 6. Detailed Module Design

**6.1 Leaky Integrate-and-Fire (LIF) Neuron Model**
The process of state updating in LIF neurons follows certain principles: charging, leakage, and firing. It mimics biological neurons. At time step t, when the temporary membrane potential rises and reaches the threshold, a spike is generated as 1, and the temporary voltage will be reset; otherwise, the spike remains 0.

**6.2 Sparsity-Aware Neuronal Dynamics**
The sparsity-aware neuronal dynamics implement efficient heterogeneous plasticity in SNNs by a dynamic threshold function built on the LIF neuron model.
- **Neuron Division:** Neurons in each channel are randomly designated as either stable or adaptive, using a channel-wise mask M. Most neurons remain stable to maintain consistent spike patterns, while others adapt to incremental learning.
- **Sparsity Definition:** Sparsity is defined as the channel-wise firing rate with spike output.
- **Threshold Regulation:** The threshold regulation factor ensures that thresholds of adaptive neurons change more freely than those of stable neurons. The threshold is dynamically updated by the difference between the sparsity of neurons in the current incremental task and base task 0. With dynamic thresholds, neurons maintain relatively stable spike patterns, which limits unnecessary synaptic updates, preserving synaptic traces encoding base-class knowledge.

**6.3 Zeroth-Order Optimization (ZOO)**
Multi-point estimate: ZOO approximates the full gradients through function value. It presents the gradient estimate with two-point directional derivative. Assessing whether perturbations induce a spike in the neuron allows for an approximation of the gradient. Through random permutation, the central finite difference approximation accommodates the influence of all neighbors, for a balanced depiction of local dynamics.
- **Convergence Analysis:** The non-differentiability renders the entire optimization problem non-convex. The convergence is evaluated using the first-order stationary condition via the squared gradient norm. More details on analysis of sources of error involve the discrepancy between the true gradient and the gradient of the Gaussian-smoothed surrogate, and the error introduced when estimating through a finite number of sample averages.
- **Loss Function:** The total loss function incorporates both the temporal error term (TET) and the MSE loss.

**6.4 Fast Adaptive Prototype Subspace Projection**
Class prototypes averaging extracted features often cause discrepancies with actual data distributions. To mitigate this bias, prototype orthogonal subspace projection updates novel-class prototypes in two steps:
1. Projecting novel-class prototypes onto the orientation subspace spanned by base-class prototypes. A new prototype representation is constructed to represent a generalized inverse of a covariance-like matrix in projection calculation. The normalization term is necessary because the generation process of the subspace basis is not guaranteed to be orthogonal to each other.
2. Blending the projected prototype with the original novel prototype through a convex combination. The trade-off coefficient balances the integration of base and novel knowledge. The orthogonal projection measures the alignment between novel and base classes, where a larger projection magnitude implies stronger cosine similarity. Consequently, features with higher similarity contribute more to the prototype update.

**6.5 Spiking Architecture and Energy Consumption Calculation**
The architectures transform popular ANN architectures into SNN equivalents, removing the need for floating-point multiplication. The theoretical energy consumption of an SNN layer is calculated using spike-based accumulate (AC) operations, while ANNs use floating-point multiply-and-accumulate (MAC) operations.

### 7. All Mathematical Formulas and Symbol Definitions

**LIF Neuron Model:**
U′t = γIt +Ut−1, It = f(x;θ) (1)
St = H(U′t −Uth) (2)
Ut = StUreset + (1− St)U′t (3)
Where H = 1Ut′>Uth, γ is the time constant, It is the spatial input calculated by applying function f with x as input and θ as learnable parameters, U′t is a temporary voltage at time step t, Uth is the threshold, St is the spike, and Ureset is the reset voltage.

**Backpropagation of SNNs:**
∂L/∂W = T∑t=1 ∂L/∂St ∂St/∂Ut ∂Ut/∂It ∂It/∂W (4)
Where W is the weight of input, and ∂St/∂Ut is non-differentiable.

**General Goal of FSCIL:**
E(xi,yi)∼Dtrain0∪···∪Dtrains [L(f(xi;Dtrains, θs−1), yi)] (5)
Where the algorithm f constructs a new model using current training data Dtrains and parameters θs−1 to minimize the loss L.

**Sparsity-Aware Neuronal Dynamics:**
M = {mc}, c = 1, 2, ..., C, where C is the channel number and mc = 1c≤⌊ηC⌋
r = 1/|Ω| ∑(b,t,n)∈Ω S(b, t, n)
A = β(1−M) + γM (6)
Where β and γ are hyper-parameters, and setting β > γ ensures that thresholds of adaptive neurons change more freely than those of stable neurons. M is the channel-wise mask, η ∈ (0, 1) is the adaptive ratio, and r is sparsity.
U′th = Uth +A(rc − rb) (7)
Where rc and rb are the sparsity of neurons in current incremental task and base task 0, respectively.

**Zeroth-Order Optimization:**
∇̂f(x) := ϕ(d)/µ Σbi=1[f(x+ δzi)− f(x− δzi)] (8)
Where b is the number of i.i.d. samples {zi}bi=0, d is a dimension-dependent factor.
g2(u; δ, z) = H(u+ δz)−H(u− δz)/2δ z = {0, |u| > δ |z| |z|/2δ , |u| < δ |z| (9)
Where z obeys a specific distribution p over b empirical samples and H is the heaviside function.
∂St/∂Ut := 1/b b∑i=1 g2(u; δ, zi) (10)

**Convergence analysis of ZOO:**
1/T T∑t=1 E[∥g(u)∥22] = O(δ2 + 1/b) (11)
E[g(uT )− g(u∗)] ≤ O(1/√T) (12)

**Loss Function:**
L = (1− λ) 1/T T∑t=1 LCE(ut,y) + λLMSE(ut,y) (13)
Where T denotes the number of time steps, and λ is a hyperparameter.

**Prototype Subspace Projection:**
B̃ = [Pb1/∥Pb1∥2, Pb2/∥Pb2∥2, . . . , PbB/∥PbB∥2]⊤ ∈ RB×D
C̃ = [Pn1/∥Pn1∥2, Pn2/∥Pn2∥2, . . . , PnN/∥PnN∥2]⊤ ∈ RN×D
G = B̃(B̃⊤B̃)−1B̃⊤ (14)
P̃proj = C̃G
P̃ = (1− α)C̃+ αP̃proj (15)
Where B̃ and C̃ are specific prototypes that belong to CB base classes and CN new classes, D is the feature dimension, G represents a generalized inverse of a covariance-like matrix, P̃proj are the projection vectors, and α is the trade-off coefficient.

**Appendix Formulas:**
It = BN(Conv(Xt;θ)) (16)
Ut = τUt−1 + It (17)
St = H(Ut −Uth) (18)
A = β(1−M) + γM (19)
U′th = Uth +A(rc − rb) (20)
E[∥∇̂g(x)−∇g(x)∥22] = O(δ2) +O(d/bδ2) (21)
E[f(xT )− f(x∗)] (22)
E[T∑t=1 ft(xt)−minx∈X T∑t=1 ft(x)] (23)
1/T T∑t=1 E[∥∇f(xt)∥22] (24)
PX(xt,∇f(xt), ηt) := 1/ηt [xt −ΠX(xt − ηt∇f(xt))] (25)
∫∞0 zα+1λ(z) dz < ∞ (26)
λ̃(z) = 1/c ∫∞|z| tαλ(t) dt (27)
Ez∼p[g2(u; z, δ)] = d/du Ez∼λ̃[h(u+ δz)] (28)
Ez∼p[g2(u; z, δ)] = ∫∞−∞ |z|/2δ 1/√2π exp(− z2/2) dz = 1/δ√2π exp(− u2/2δ2) (29)
c := −2δ2 ∫∞0 1/zα h′(zδ)dz < ∞, p(z) := − δ2/c zα h′(zδ), z > 0 (30)
cEz∼p[g2(u; z, δ)] = h(u) (31)
h′(u) = −k2 exp(−ku)(1− exp(−ku)) / (1 + exp(−ku))3 (32)
h(u) := d/du 1/(1 + exp(−ku)) = k exp(−ku) / (1 + exp(−ku))2 (33)
c = −2δ2 ∫∞0 h′(tδ)/t dt = 2δ2k2 ∫∞0 exp(−kδt)(1− exp(−kδt)) / t(1 + exp(−kδt))3 dt = δ2k2/a2 (34)
p(z) = −δ2/c h′(δz)/z = a2 exp(−kδz)(1− exp(−kδz)) / z(1 + exp(−kδz))3, z > 0 (35)
∥g(x)−∇Φ(x)∥ = ∥(∇F (x)−∇Φ(x)) + (g(x)−∇F (x))∥ ≤ ∥∇F (x)−∇Φ(x)∥+ ∥g(x)−∇F (x)∥ (36)
|f(x)− Φ(x)| = |ϵ(x)| ≤ ϵf , ∀x ∈ Rn (37)
∥∇F (x)−∇Φ(x)∥ ≤ √b Lδ + √b ϵf/δ (38)
∥∇F (x)−∇Φ(x)∥ ≤ nMδ2 + √b ϵf/δ (39)
Var(g(x)) = 1/N Eu∼N(0,I)[(f(x+ δu)− f(x− δu)/2δ)2 uuT] − 1/N ∇F (x)∇F (x)T (40)
κ(x) = 3/N (3∥∇Φ(x)∥2 + M2δ4/36 (n+ 2)(n+ 4)(n+ 6) + ϵ2f/δ2) (41)
S = SN ((BN(CONV(X)))) (42)
E(l) = EAC × SOPs(l) (43)
E(b) = EMAC × FLOPs(b) (44)
SOPs(l) = T × ζ × FLOPs(l) (45)

### 8. Algorithm Pseudocode

**Algorithm 1 Overall SAFA-SNN Algorithm**
Input: Dataset {Ds}Ss=0, membrane U, feature extractor ϕ
Output: Spike trains St, parameters θ
1: Initialize mask M to set η% adaptive neurons 1 and setting others 0 in each channel
2: Initialize threshold Uth with zeros
3: for session s = 0 to S do
4: Conduct state updating in LIF neuron models by Equation (1), (2), and (3)
5: if s == 0 then
6: Calculate the threshold regulation factor A by Equation (6)
7: Sample point zi ∼ N (0, 1), i = 1, 2, ..., b
8: if |x| < δ|zi| then
9: Estimate the gradient ĝi = mi · |zi|
10: end if
11: Estimate the non-differential ∂St/∂Ut by Equation (10)
12: Conduct base session optimization L and update ϕ(θ) by Loss in Equation (13)
13: else
14: Update threshold Uth by U′th = Uth +A(rc − rb)
15: Compute the projection subspace by Equation (14)
16: Conduct incremental learning with P̃ in Equation (15)
17: end if
18: end for