**1. Research Background and Existing Pain Points**
Backpropagation (BP) has been the driving force behind the success of artificial intelligence across a wide variety of tasks. Despite these triumphs, BP’s reliance on non-local error signals and weight transport lacks biological plausibility. The brain does not appear to implement the gradient computations performed by BP, in particular the explicit derivative of the activation function, which demands precise access to the rate of change in neuronal activities at specific operating points. Moreover, implementing BP in neuromorphic systems incurs enormous overhead. Equilibrium Propagation (EP) presents a compelling and hardware-friendly alternative. It leverages naturally settling dynamics in RNN for credit assignment, and eliminates the need for explicit activation derivatives. EP operates in two phases with nearly identical dynamics, and the synaptic adjustments depend only on local information. In EP, the output layer is softly nudged by the prediction error toward configurations that incrementally minimize the loss function, a regime termed weak supervision. A major drawback of existing implementations of EP is its notably slow training speed and instability. An RNN often requires dozens or even hundreds of iterations to reach a stable state. Previous attempts to optimize EP’s performance have led to markedly more complicated procedures.

**2. Core Research Motivation and Scientific Questions**
Brain-like intelligent systems need brain-like learning methods. Drawing inspiration from the topology and dynamics of the brain is a viable approach to advancing biologically plausible learning mechanisms and to promoting energy-efficient computing systems for AI. The core research motivation is to substantially improve the convergence properties of RNNs and the training speed of EP while achieving performance comparable to BP, drawing inspiration from the brain's structure and dynamics. The scientific questions addressed are: How to overcome the instability and high computational cost of EP? How to accelerate convergence in RNNs for EP? How to counteract the vanishing gradient problem that arises when feedback pathways are weak in deep RNNs within the EP framework?

**3. Overall Core Idea and Design Philosophy**
The overall core idea is to propose a biologically plausible Feedback-regulated REsidual recurrent neural network (FRE-RNN) and study its learning performance in the EP framework. The design philosophy is directly inspired by the structure and dynamics of the brain. Cortical areas in the brain feature dynamic regulation of feedforward and feedback connections; dynamically regulating the strength of feedback allows the brain to optimize information integration, ensuring efficient perception and decision-making. Feedback regulation enables rapid convergence by attenuating feedback signals and reducing the disturbance of feedback paths to feedforward paths. Furthermore, mammalian neocortices feature extensive lateral and feedback loops that interconnect disparate regions, forming a richly recursive network with short average path lengths. Residual connections with brain-inspired topologies reflect the long-range skip-layer projections observed in cortical circuits, which help alleviate the vanishing gradient problem by providing skip pathways that preserve gradient flow in deep RNNs where feedback pathways are weak.

**4. Core Innovation Points**
*   **Feedback Regulation for Acceleration:** By scaling down the feedback strength of RNNs, the robustness of EP is enhanced and the training and inference speed are accelerated by orders of magnitude because of the improved convergence properties.
*   **Residual Connections for Deep Networks:** To counteract the gradient vanishing problem caused by weak feedback, residual connections are introduced into the layered RNNs, enabling the training of deep networks that previously challenged EP and achieving performance closer to BP.
*   **Arbitrary Graph Topologies:** The feedback regulation and residual connections in RNNs of arbitrary graph topologies mirror the multi-scale recurrence in biological neural networks. This fosters EP’s biological plausibility and extends its applicability in brain-inspired computational hardware.
*   **Coordinated Plasticity:** Down-scaled feedback coordinates plasticity of different layers. The feedback scaling alone is able to regulate gradient strength of different layers, allowing a uniform learning rate, pointing to another possible mechanism to coordinate plasticity.
*   **Theoretical Equivalence:** At the infinitesimal inference limit (weak supervision and weak feedback), EP is equivalent to Local Representation Alignment (LRA) and BP, explaining why the dynamics of FRE-RNN resembles a feedforward neural network and achieves comparable performance.

**5. Overview of the Overall Technical Solution**
The technical solution separates the input and output layer from the recurrent network, allowing the output layer to adopt the SoftMax activation. The RNN states evolve for T discrete steps until they converge. Feedforward weights $W_i$ and feedback weights $B_i$ are rescaled by coefficients $\alpha_i$ and $\beta_i$ respectively. Scaling only the feedback strength $\beta_i$ is chosen as scaling forward weights distorts forward signal propagation. EP operates in two phases: a free (first) phase where $\beta_f = 0$ for $t_e$ iterations, and a weakly clamped (second) phase where $\beta_f = \beta_{f1}$ for $K$ iterations. The weights are updated with an STDP-compatible rule based on the offset of the stable point caused by error nudging. For deep RNNs (e.g., 10-hidden-layer), cross-layer residual links are added. For asymmetric connections, skip-layer connections between non-adjacent layers are introduced with $P=20\%$ probability, creating an RNN with arbitrary graph topologies (AGT). The approach also embeds convolutional architectures within the RNN by adding feedback connections symmetric with the feed-forward connections.

**6. Detailed Module Design**
*   **Layered Architecture of RNN:** The prototypical setting separates the input layer $s_0$ and output layer $s_p$ from the RNN (hidden layers $s_1, s_2$). The feedforward weights $W_i$ and feedback weights $B_i$ are rescaled by coefficients $\alpha_i$ and $\beta_i$ respectively. The dashed box encloses an RNN formed by layers $s_1$ and $s_2$ with feedforward and feedback pathways. $\beta_f$ is the nudging factor scaling the feedback strength of prediction error.
*   **Feedback Regulation Module:** Scaling down the feedback strength $\beta_i$ enhances robustness and accelerates training. Down-scaling feedback reduces the convergence time of RNN. In both symmetric and asymmetric feedback cases, down-scaling feedback connections tends to stabilize the network. For deeper networks, overly low feedback scaling jeopardizes performance due to vanishing gradients, requiring a trade-off in feedback scaling (e.g., $\beta_i = 0.1$).
*   **Residual Connection Module:** For a 10-hidden-layer RNN with symmetric connections, 3 long-range bidirectional connections bypass adjacent layers to reduce gradient decay. For RNN with asymmetric connections, skip-layer connections between non-adjacent layers are introduced with $P=20\%$ probability, creating an RNN with arbitrary graph topologies (AGT) where any pair of layers form connections stochastically.
*   **Convolutional Architecture Embedding:** An RNN embedded with convolutional architecture in its forward paths (2 convolution layers, 2 max-pooling layers and 1 fully connected layer). The forward convolutional structure follows CNN architecture. The CNN is transformed to an RNN by adding feedback connections symmetric with the feed-forward connections, utilizing transpose convolution and max-unpooling.

**7. All Mathematical Formulas and Symbol Definitions**
*   **Convergent RNN Dynamics:** $s[t+ 1] = F (x, s[t], \theta)$, where $F$ is the transition function, $s[t]$ is the network state at time step $t$, and $\theta$ denotes the parameters.
*   **Whole RNN Dynamics:** $s^{\beta_f}[t+ 1] = F (s^{\beta_f}[t], b) = \rho(W \cdot s^{\beta_f}[t] + b)$
*   **Combined Input/Bias:** $b = [W_0 \cdot s_0, \beta_f \cdot B_f \cdot e_p]$, where $s^{\beta_f} = [s^{\beta_f}_1 , s^{\beta_f}_2 ]$.
*   **Prediction and Error:** $s_p = \text{SoftMax}(W_f \cdot s_2)$, $e_p = s_{tar} - s_p$
*   **STDP-compatible Weight Update:** $\Delta W_i = ds_{i+1} \cdot (s^0_i)^\top$, $ds_{i+1} = s^{\beta_{f1}}_{i+1} - s^0_{i+1}$, where $ds_i$ is the offset of stable point caused by the error nudging.
*   **Output Weight Update:** $\Delta W_f = (s_{tar} - s^0_p) \cdot (s^0_2)^\top$
*   **Equivalence Proofs (BP):**
    *   Forward process: $s_1 = \rho(h_1), h_1 = W_0 \cdot s_0$; $s_2 = \rho(h_2), h_2 = W_1 \cdot s_1$; $s_p = h_p, h_p = W_f \cdot s_2$
    *   BP Loss: $L_{BP} = \frac{1}{2} (s_p-s_{tar})^2$
    *   BP Weight Update: $\Delta W_0 = -\frac{\partial L_{BP}}{\partial W_0} = -\rho'(h_1)\odot W^\top_1 \cdot (\rho'(h_2)\odot W^\top_f \cdot (s_p - s_{tar})) \cdot (s_0)^\top$
*   **Equivalence Proofs (LRA):**
    *   Forward process: $s^0_1 = \rho(h^0_1), h^0_1 = W_0 \cdot s_0$; $s^0_2 = \rho(h^0_2), h^0_2 = W_1 \cdot s^0_1$; $s^0_p = h^0_p, h^0_p = W_f \cdot s^0_2$
    *   Target representations: $s^{\beta_{LRA}}_2 = \rho(h^{\beta_{LRA}}_2 ), h^{\beta_{LRA}}_2 = W_1 \cdot s^0_1 + \beta_{LRA} \cdot B_f \cdot e_p$
    *   $s^{\beta_{LRA}}_1 = \rho(h^{\beta_{LRA}}_1 ), h^{\beta_{LRA}}_1 = W_1 \cdot s_0 + \beta_{LRA} \cdot B_1 \cdot e_2$, where $e_2 = s^{\beta_{LRA}}_2 - s^0_2$
    *   LRA Loss: $L_{LRA} = \sum_{i=1}^L k_i L_i(s^0_i , s^{\beta_{LRA}}_i ) = \sum_{i=1}^L \frac{1}{2}(s^0_i - s^{\beta_{LRA}}_i)^2$
    *   LRA Weight Update: $\Delta W_i = -\frac{\partial k_i L_i(s^0_{i+1}, s^{\beta_{LRA}}_{i+1})}{\partial W_i} = (s^{\beta_{LRA}}_{i+1} - s^0_{i+1})\odot f'(h^0_{i+1}) \cdot (s^0_i)^\top \approx (s^{\beta_{LRA}}_{i+1} - s^0_{i+1}) \cdot (s^0_i)^\top$
    *   Error expansion: $e_i = s^{\beta_{LRA}}_i - s^0_i = \rho(h^{\beta_{LRA}}_i)- \rho(h^0_i) = \rho(h^0_i + \beta_{LRA} \cdot B_i \cdot e_{i+1})- \rho(h^0_i) \approx \rho'(h^0_i)\odot (\beta_{LRA} \cdot B_i \cdot e_{i+1})$
*   **Equivalence Proofs (EP):**
    *   Network states evolution: $h^\beta_1 = W_0 \cdot s^\beta_0 + \beta_1 \cdot B_1 \cdot s^\beta_2$; $h^\beta_2 = W_1 \cdot s^\beta_1 + \beta_f \cdot B_f \cdot e_p$; $h^\beta_p = W_f \cdot s^\beta_2$; $s^\beta_1 , s^\beta_2 , s^\beta_p = \rho(h^\beta_1 ), \rho(h^\beta_2 ), h^\beta_p$
    *   Error of $s_2$ neurons: $ds_2 = [\rho(h^{\beta_f}_2 )]_{\beta_f\to 0} - [\rho(h^0_2)]_{\beta_f=0} \approx \rho'(h^0_2)\odot (\beta_f \cdot B_f \cdot e_p)$
    *   Error of $s_1$ neurons: $ds_1 = [\rho(h^{\beta_f}_1 )]_{\beta_f\to 0} - [\rho(h^0_1)]_{\beta_f=0} \approx \rho'(h^0_1)\odot (\beta_1 \cdot B_1 \cdot (\rho'(h^0_2)\odot (\beta_f \cdot B_f \cdot e_p)))$
    *   EP Weight Update: $\Delta W_0 = \frac{ds_1 \cdot (s^0_0)^\top}{\beta_1 \cdot \beta_f} = \rho'(h^0_1)\odot B_1 \cdot (\rho'(h^0_2)\odot B_f \cdot e_p) \cdot (s^0_0)^\top$
    *   With $B_i = W^\top_i$: $ds_1 = \beta_f \cdot \beta_1 \cdot \rho'(h^0_1)\odot W^\top_1 \cdot (\rho'(h^0_2)\odot W^\top_f \cdot -(s_p - s_{tar}))$
    *   Final EP update equivalent to BP: $\Delta W_0 = \frac{ds_1 \cdot (s^0_0)^\top}{\beta_1 \cdot \beta_f} = -\rho'(h^0_1)\odot W^\top_1 \cdot (\rho'(h^0_2)\odot W^\top_f \cdot (s_p - s_{tar})) \cdot (s^0_0)^\top$
    *   Dynamics under infinitesimal limit: $s^0_1 = \rho(h^\beta_1 ), h^0_1 = [W_0 \cdot s_0 + \beta_1 \cdot B_1 \cdot s^0_2]_{\beta_1\to 0} \approx W_0 \cdot s_0$; $s^0_2 = \rho(h^0_2), h^0_2 = [W_1 \cdot s^0_1 + \beta_f \cdot B_f \cdot e_p]_{\beta_1\to 0, \beta_f=0} \approx W_1 \cdot s^0_1$; $s^0_p = h^0_p, h^0_p = W_f \cdot s^0_2$.

**8. Algorithm Pseudocode**

**Algorithm 1 EP with Feedforward and Feedback Scaling**
Require: Input: (x, star)
Require: Parameters: $\theta = [W_0,W_1,W_f , B_f , B_1, \alpha_1, \beta_1, \beta_{f1}]$
Ensure: Updated parameters $\theta$
1: function FIRST-PHASE($\theta$, star)
2: $s_0 \leftarrow x$
3: for t = 1 to T do
4: $h_1 \leftarrow W_0 \cdot s_0 + \beta_1 \cdot B_1 \cdot s^0_2$
5: $h_2 \leftarrow \alpha_1 \cdot W_1 \cdot s^0_1$
6: $h_p \leftarrow W_f \cdot s^0_2$
7: $s^0_1, s^0_2, s^0_p \leftarrow \rho(h_1), \rho(h_2), \text{SoftMax}(h_p)$
8: end for
9: $\Lambda_1 \leftarrow [s^0_i ], i = 0, 1, 2, p$
10: return $\Lambda_1$
11: end function
12: function SECOND-PHASE($\theta,\Lambda_1$, star)
13: $s^{\beta_{f1}}_1 , s^{\beta_{f1}}_2 , s^{\beta_{f1}}_p \leftarrow s^0_1, s^0_2, s^0_p$
14: for t = 1 to K do
15: $e_p \leftarrow star - s^{\beta_{f1}}_p$
16: $h_1 \leftarrow W_0 \cdot s_0 + \beta_1 \cdot B_1 \cdot s^{\beta_{f1}}_2$
17: $h_2 \leftarrow \alpha_1 \cdot W_1 \cdot s^{\beta_{f1}}_1 + \beta_f \cdot B_f \cdot e_p$
18: $h_p \leftarrow W_f \cdot s^{\beta_{f1}}_2$
19: $s^{\beta_{f1}}_1 , s^{\beta_{f1}}_2 , s^{\beta_{f1}}_p \leftarrow \rho(h_1), \rho(h_2), \text{SoftMax}(h_p)$
20: end for
21: $ds_i \leftarrow s^{\beta_{f1}}_i - s^0_i , i = 1, 2$
22: $\Lambda_2 \leftarrow [ds_1, ds_2]$
23: return $\Lambda_2$
24: end function
25: function UPDATING-WEIGHTS($\theta,\Lambda_1,\Lambda_2$, star)
26: $\Delta W_i \leftarrow ds_{i+1} \cdot (s^0_i)^\top, i = 0, 1$
27: $\Delta W_f \leftarrow (star - s^0_p) \cdot (s^0_2)^\top$
28: end function

**Algorithm S1 Local Representation Alignment (LRA)**
Input: (x, star)
Parameter: $\theta = [W_0,W_1,W_2, B_2, B_1, \beta_{LRA}]$
Output: $\theta$
1: function FORWARD($\theta$, x)
2: $s_0 \leftarrow x$
3: $s^0_1 \leftarrow \rho(h_1), h_1 \leftarrow W_0 \cdot s_0$
4: $s^0_2 \leftarrow \rho(h_2), h_2 \leftarrow W_1 \cdot s^0_1$
5: $s^0_p \leftarrow W_f \cdot s^0_2$
6: $\Lambda_1 \leftarrow [s^0_i ], i = 0, 1, 2, p$
7: return $\Lambda_1$
8: end function
9: function FEEDBACK($\theta,\Lambda_1$, star)
10: $e_p \leftarrow star - s^0_p$
11: $s^{\beta_{LRA}}_2 \leftarrow \rho(h_2), h_2 \leftarrow W_1 \cdot s^0_1 + \beta_{LRA} \cdot B_f \cdot e_p$
12: $e_2 \leftarrow s^{\beta_{LRA}}_2 - s^0_2$
13: $s^{\beta_{LRA}}_1 \leftarrow \rho(h_1), h_1 \leftarrow W_0 \cdot s_0 + \beta_{LRA} \cdot B_1 \cdot e_2$
14: $e_1 \leftarrow s^{\beta_{LRA}}_1 - s^0_1$
15: $\Lambda_2 \leftarrow [e_1, e_2, e_p]$
16: return $\Lambda_2$
17: end function
18: function UPDATING-WEIGHTS($\theta,\Lambda_1,\Lambda_2$)
19: $\Delta W_i \leftarrow e_{i+1} \cdot (s^0_i)^T , i = 0, 1$
20: $\Delta W_f \leftarrow e_p \cdot (s^0_2)^T$
21: end function

**Algorithm S2 Two phases in EP training process for convolution architecture**
Input: Sample-label pairs (x, star)
Parameter: $\theta = [W_0,W_1,W_f , B_f , B_1, \alpha_1, \beta_1, \beta_{f1}]$
Output: $\theta$
1: function FIRST-PHASE($\theta$, star)
2: $s_0 \leftarrow x$
3: for t $\leftarrow$ 1 to T do
4: $h_1 \leftarrow \text{Conv0}(s_0) + \beta_1 \cdot \text{MaxUnpool1}(\text{ConvT1}(s^0_2))$
5: $h_2 \leftarrow \text{Conv1}(\text{MaxPool1}(s^0_1))$
6: $h_p \leftarrow W_f \cdot \text{Flatten}(\text{MaxPool2}(s^0_2))$
7: $s^0_1, s^0_2, s^0_p \leftarrow \rho(h_1), \rho(h_2),\text{SoftMax}(h_p)$
8: end for
9: $\Lambda_1 \leftarrow [s^0_i ], i = 0, 1, 2, p$
10: return $\Lambda_1$
11: end function
12: function SECOND-PHASE($\theta,\Lambda_1$, star)
13: $s_0, s^{\beta_{f1}}_1 , s^{\beta_{f1}}_2 , s^{\beta_{f1}}_p \leftarrow x, s^0_1, s^0_2, s^0_p$
14: for t $\leftarrow$ 1 to K do
15: $e_p \leftarrow star - s^{\beta_{f1}}_p$
16: $h_1 \leftarrow \text{Conv0}(s_0) + \beta_1 \cdot \text{MaxUnpool1}(\text{ConvT1}(s^{\beta_{f1}}_2 ))$
17: $h_2 \leftarrow \text{Conv1}(\text{MaxPool1}(s^{\beta_{f1}}_1 )) + \beta_f \cdot \text{MaxUnpool2}(\text{Unflatten}(W^\top_f e_p))$
18: $h_p \leftarrow W_f \cdot \text{Flatten}(\text{MaxPool2}(s^{\beta_{f1}}_2 ))$
19: $s^{\beta_{f1}}_1 , s^{\beta_{f1}}_2 , s^{\beta_{f1}}_p \leftarrow \rho(h_1), \rho(h_2),\text{SoftMax}(h_p)$
20: end for
21: end function

**Algorithm S3 EP with feedback scaling and residual connections (Figure 3b)**
Input: (x, star)
Parameter: $\theta = [W_0,W_i,W_f , B_f , B_i, \beta_i, \beta_{f1}]$
Output: $\theta$
1: function ITERATION($\theta,\Lambda_1$, star)
2: for t $\leftarrow$ 1 to K do
3: if Nudging Phase then
4: $\beta_f \leftarrow \beta_{f1}$
5: $s_0, s^{\beta_{f1}}_1 , s^{\beta_{f1}}_2 , s^{\beta_{f1}}_p \leftarrow x, s^0_1, s^0_2, s^0_p$
6: else
7: $\beta_f \leftarrow 0$
8: $s_0 \leftarrow x$
9: end if
10: $h_1 \leftarrow W_0 s_0 + \beta_1 B_1 s^{\beta_f}_2 + \beta_{4,1}B_{4,1}s^{\beta_f}_4$
11: $h_2 \leftarrow W_1 s^{\beta_f}_1 + \beta_2 B_2 s^{\beta_f}_3$
12: $h_3 \leftarrow W_2 s^{\beta_f}_2 + \beta_3 B_3 s^{\beta_f}_4$
13: $h_4 \leftarrow W_3 s^{\beta_f}_3 + \beta_{14} B_{14} s^{\beta_f}_5 +W_{1,4}s^{\beta_f}_1 + \beta_{7,4}B_{7,4}s^{\beta_f}_7$
14: $h_5 \leftarrow W_4 s^{\beta_f}_4 + \beta_5 B_5 s^{\beta_f}_6$
15: $h_6 \leftarrow W_5 s^{\beta_f}_5 + \beta_6 B_6 s^{\beta_f}_7$
16: $h_7 \leftarrow W_6 s^{\beta_f}_6 + \beta_7 B_7 s^{\beta_f}_8 +W_{4,7}s^{\beta_f}_4 + \beta_{10,7}B_{10,7}s^{\beta_f}_{10}$
17: $h_8 \leftarrow W_7 s^{\beta_f}_7 + \beta_8 B_8 s^{\beta_f}_9$
18: $h_9 \leftarrow W_8 s^{\beta_f}_8 + \beta_9 B_9 s^{\beta_f}_{10}$
19: $h_{10} \leftarrow W_9 s^{\beta_f}_9 + \beta_f B_f e^{\beta_f}_p +W_{7,10}s^{\beta_f}_7$
20: $h_p \leftarrow W_f s^{\beta_f}_{10}$
21: $s^{\beta_f}_i \leftarrow \rho(h_i), i = 0, 1, 2, \ldots , 10$
22: $s^{\beta_f}_p \leftarrow \text{SoftMax}(h_p)$
23: end for
24: end function