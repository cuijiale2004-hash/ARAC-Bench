## ABSTRACT

Brain-like intelligent systems need brain-like learning methods. Equilibrium Propagation (EP) is a biologically plausible learning framework with strong potential for brain-inspired computing hardware. However, existing implementations of EP suffer from instability and prohibitively high computational costs. Inspired by the structure and dynamics of the brain, we propose a biologically plausible Feedback-regulated REsidual recurrent neural network (FRE-RNN) and study its learning performance in the EP framework. Feedback regulation enables rapid convergence by attenuating feedback signals and reducing the disturbance of feedback paths to feedforward paths. The improvement in the convergence property reduces the computational cost and training time of EP by orders of magnitude, delivering performance on par with backpropagation (BP) in benchmark tasks. Meanwhile, residual connections with brain-inspired topologies help alleviate the vanishing gradient problem that arises when feedback pathways are weak in deep RNNs. Our approach substantially enhances the applicability and practicality of EP. The techniques developed here also offer guidance for implementing in-situ learning in physical neural networks.

## 1 INTRODUCTION

Backpropagation (BP) has been the driving force behind the success of artificial intelligence (AI) across a wide variety of tasks, ranging from image recognition to natural language processing (Rumelhart et al., 1986; Lecun, 1988; He et al., 2016; Vaswani et al., 2017). Despite these triumphs, BP’s reliance on non-local error signals and weight transport lacks biological plausibility (Journe et al., 2023; Ororbia, 2023). The brain does not appear to implement the gradient computa-´ tions performed by BP, in particular the explicit derivative of the activation function, which demands precise access to the rate of change in neuronal activities at specific operating points (Ororbia, 2023). Moreover, implementing BP in neuromorphic systems incurs enormous overhead (Kudithipudi et al., 2025). Drawing inspiration from the topology and dynamics of the brain is a viable approach to advancing biologically plausible learning mechanisms and to promoting energy-efficient computing systems for AI.

Equilibrium Propagation (EP) (Scellier & Bengio, 2017; Ernoult et al., 2019; Laborieux et al., 2021) presents a compelling and hardware-friendly alternative. It leverages naturally settling dynamics in RNN for credit assignment, and eliminates the need for explicit activation derivatives. EP operates in two phases with nearly identical dynamics, and the synaptic adjustments depend only on local information (Ackley et al., 1985; Movellan, 1991; Ernoult et al., 2020). In EP, the output layer is softly nudged by the prediction error toward configurations that incrementally minimize the loss function, a regime termed weak supervision (Millidge et al., 2023). A major drawback of EP is its notably slow training speed and instability. An RNN often requires dozens or even hundreds of iterations to reach a stable state (Scellier & Bengio, 2017). Previous attempts to optimize EP’s performance have led to markedly more complicated procedures (O’Connor et al., 2019; Laborieux & Zenke, 2024).

In this paper, we draw inspiration from the brain and propose a Feedback-regulated REsidual recurrent neural network (FRE-RNN). We substantially improve the convergence properties of the RNNs and training speed of EP while achieving performance comparable to BP. Our contributions are as follows:

• By scaling down the feedback strength of RNNs, we enhance the robustness of EP and accelerate the training and inference speed by orders of magnitude because of the improved convergence properties.

• To counteract the gradient vanishing problem caused by weak feedback, we introduce residual connections into the layered RNNs, enabling the training of deep networks that previously challenged EP and achieving performance closer to BP.

• The feedback regulation and residual connections in RNNs of arbitrary graph topologies mirror the multi-scale recurrence in biological neural networks. Our work fosters EP’s biological plausibility and extends its applicability in brain-inspired computational hardware.

## 2 BACKGROUND

## 2.1 CONVERGENT RNNS WITH STATIC INPUT

Consider an RNN as a dynamical system driven by a static input x:

$$
s [ t + 1 ] = F (x, s [ t ], \theta),\tag{1}
$$

where F is the transition function, s[t] is the network state at time step $t ( t = 0 , 1 , 2 , . . . , T )$ , and θ denotes the parameters. Assuming that the network state stabilizes in $T$ steps, the RNN reaches a stable point s[T]. Its convergence is typically guaranteed by either symmetric connections with asynchronous updates or by a sufficiently small spectral radius of asymmetric connections with synchronous updates (Hopfield, 1982; Yildiz et al., 2012; Liu et al., 2026). That said, other factors, e.g. activation function, also influence the dynamical properties of RNNs (Miller & Hardt, 2019).

## 2.2 SCALING ADJACENCY MATRIX TO TUNE NETWORK DYNAMICS

Scaling the spectral radius (SR) of the adjacency matrix, the largest eigenvalue of the weight matrix, is a common method to tune the dynamics of RNN (Bai et al., 2012; Nakajima et al., 2024; Liu et al., 2026). A SR less than one yields stable and convergent dynamics. In this case, injected signals tend to decay over time, which manifests as short-term memory. A SR exceeding one can give rise to expansive or even chaotic behavior in which small perturbations are amplified. By adjusting SR, one can bias the RNN toward convergent, oscillatory, or edge-of-chaos regimes, thereby tuning computational properties, such as convergence speed or long-term memory capacity. (Jaeger & Haas, 2004; Legenstein & Maass, 2007; Miller & Hardt, 2019).

## 2.3 EQUILIBRIUM PROPAGATION

Equilibrium propagation is a learning framework initially based on energy-based models. It proceeds in two phases: a free (first) phase and a weakly clamped (second) phase. For the first phase, the RNN converges to a steady state $s ^ { 0 }$ under the stimulation of input alone. In the clamped phase, the network is gently nudged by the prediction error and settles to a new stable state $s ^ { \hat { \beta } }$ . The weight update can be simplified to a contrastive learning compatible with spiking-time-dependent plasticity (STDP) (Scellier et al., 2018). EP has been further generalized to asymmetric RNNs governed by vector field dynamics (Scellier et al., 2018). Recent work shows that asymmetry in skew-symmetric Hopfield models can improve classification performance (Høier et al., 2024).

## 2.4 FEEDBACK REGULATION AND NETWORK STRUCTURE IN THE BRAIN

Cortical areas in the brain feature dynamic regulation of feedforward and feedback connections (Felleman & Van Essen, 1991; Mejias et al., 2016; Michalareas et al., 2016; Semedo et al., 2022; Fis¸ek et al., 2023; Wang et al., 2023). In the visual system, for instance, feedforward signals dominate immediately following the onset of external stimulus, whereas feedback signals become prominent during spontaneous activity. Dynamically regulating the strength of feedback allows the brain to optimize information integration, ensuring efficient perception and decision-making. In mammalian neocortices, information processing involves not only feedforward synaptic chains but also extensive lateral and feedback loops that interconnect disparate regions, forming a richly recursive network rather than a strictly layered structure. This topology implies short average path length between neurons and efficient information flow (Watts & Strogatz, 1998; Markov et al., 2013; Lynn & Bassett, 2019; Kulkarni & Bassett, 2025). In deep neural networks, residual connections reflect the long-range skip-layer projections observed in cortical circuits (Perich & Rajan, 2020; Holk & Mejias, 2024). They mitigate the vanishing gradient by providing skip pathways that preserve gradient (He et al., 2016).

## 3 ACCELERATING EP WITH BRAIN-INSPIRED NETWORK PROPERTIES

![](images/0f2299fad45f35d7126895ff0db6fb51f8ad03cdf4d044a9f1abcb96d4176f26.jpg)  
Figure 1: Illustration of feedback and feedforward regulation. (a) Layered architecture of RNN. The feedforward weights $W _ { i }$ and feedback weights $B _ { i }$ are rescaled by coefficients $\alpha _ { i }$ and $\beta _ { i }$ respectively. The dashed box encloses an RNN formed by layers $s _ { 1 }$ and $s _ { 2 }$ with feedforward and feedback pathways. $\beta _ { f }$ is the nudging factor, which essentially scales the feedback strength of prediction error. (b) Embedding convolutional architecture in RNN. Convolutional parameter (32,5,1,0) is written as (channels, kernels, stride, padding). Parameter (2) in (b) denotes max-pooling with stride 2. ConvT<sub>i</sub> represents transpose convolution, the inverse process of the convolution, and $P _ { i } ^ { - 1 }$ means max-unpooling (Ernoult et al., 2019). Model architectures and training process are given in Appendix D.

## 3.1 PROTOTYPICAL SETTING OF EQUILIBRIUM PROPAGATION

Unlike the prototypical setting of equilibrium propagation (P-EP) (Ernoult et al., 2019), we separate the input and output layer from the recurrent network (Figure 1a). This separation allows the output layer to adopt the SoftMax activation commonly used in feedforward networks, which facilitates performance comparison (Laborieux & Zenke, 2024). For clarity, the RNN (black dashed box in Figure 1a) shown here only contains two hidden layers $s _ { 1 }$ and $s _ { 2 } .$ , but the approach applies to deeper structures (see below). The states of the RNN evolve for $T$ discrete steps until they converge. The dynamics of the whole RNN can be formulated as:

$$
\begin{array}{c} s ^ {\beta_ {f}} [ t + 1 ] = F (s ^ {\beta_ {f}} [ t ], b) = \rho (W \cdot s ^ {\beta_ {f}} [ t ] + b), \\ b = [ W _ {0} \cdot s _ {0}, \quad \beta_ {f} \cdot B _ {f} \cdot e _ {p} ], \end{array}\tag{2}
$$

where $s ^ { \beta _ { f } } \left\lceil t \right\rceil$ is the state of the RNN at time $\mathbf { t } , \rho$ is the activation function, $W$ is the forward weight matrix of the RNN, and b combines the feedforward input and the error-nudging term. Note that $s ^ { \beta _ { f } } ~ = ~ [ s _ { 1 } ^ { \beta _ { f } } , s _ { 2 } ^ { \beta _ { f } } ]$ For each sample-label pair $( x , s _ { t a r } )$ , we run the free phase $( \beta _ { f } ~ = ~ 0 )$ for $t _ { e }$ iterations, obtain the prediction $s _ { p } = \operatorname { S o f t M a x } ( W _ { f } \cdot s _ { 2 } )$ , and compute the prediction error $e _ { p } =$ $s _ { t a r } - s _ { p }$ . During the clamped phase, the error nudges the RNN through the feedback weights $B _ { f }$ and scaling coefficient $\beta _ { f } = \bar { \beta } _ { f 1 } \bar { ( \beta _ { f 1 } = 0 . 1 }$ for layered architecture and $\beta _ { f 1 } = 0 . 2 5$ for convolutional architecture by default). The network evolves for K further iterations under clamping to another state. The weights $( W _ { 0 } , W _ { 1 } )$ are then updated with an STDP-compatible rule:

$$
\Delta W _ {i} = d s _ {i + 1} \cdot (s _ {i} ^ {0}) ^ {\top}, d s _ {i + 1} = s _ {i + 1} ^ {\beta_ {f 1}} - s _ {i + 1} ^ {0},\tag{3}
$$

where $d \boldsymbol { s } _ { i }$ is the offset of stable point caused by the error nudging (Scellier et al., 2018). Similarly, the final weight for output is updated:

$$
\Delta W _ {f} = (s _ {t a r} - s _ {p} ^ {0}) \cdot (s _ {2} ^ {0}) ^ {\top}.\tag{4}
$$

We also consider an RNN embedded with convolutional architecture in its forward paths (2 convolution layers, 2 max-pooling layers and 1 fully connected layer) shown in Figure 1b. The forward convolutional structure follows the architecture of existing convolutional neural networks (CNN) (Krizhevsky et al., 2012; Simonyan & Zisserman, 2015), in which a pooling layer is placed after the activation of the convolution layer. We transform the CNN to an RNN by adding feedback connections symmetric with the feed-forward connections (See Appendix D for the pseudocode and schematics).

## 3.2 FEEDBACK REGULATION IN LAYERED RNN FOR FAST CONVERGENCE

![](images/c46ba5e0edc63d67a6545f78b9110a9fd4ecd62969131348cd8bca27accb1186.jpg)

![](images/8869ba53f308412ba71e732235eec826db07ef5ae98d759c031bf2cf231370de.jpg)  
Figure 2: Convergence dynamics and speed versus feedback scaling $\beta _ { i }$ . All neurons in all hidden layers are indexed $( s _ { 1 } { : } 0 { - } 6 3 ; s _ { 2 } { : } 6 4 { - } 1 2 7 )$ . Colors indicate neuronal activity $^ { ( \mathrm { a } , \mathrm { c } , \mathrm { e } , \mathrm { g } ) }$ and changes in activity $^ { ( \mathrm { b , d , f , h } ) }$ . (a) The state evolution of RNN with symmetric weights and $\beta _ { i } = 0 . 1 ; ( \mathbf { b } )$ The one-step difference of neural states in (a). (c, d) Symmetric weights with $\beta _ { i } = 2 ; ( \mathbf { e } , \mathbf { f } )$ Asymmetric weights with $\beta _ { i } = 0 . 1 ; ( \mathrm { g }$ , h) Asymmetric weights with $\beta _ { i } = 4$ . In both symmetric and asymmetric feedback cases, down-scaling feedback connections tends to stabilize the network. See Figure 5d for the statistical robustness.

Although the SR can tune the RNN dynamics, scaling forward weights $W _ { i }$ distorts forward signal propagation, which is harmful to performance (see below). Therefore, we turn to another choice, namely, scaling only the feedback strength with $\beta _ { i }$ . This coefficient scales the gradients, in the same way as the nudging factor $\beta _ { f }$ . We consider both symmetric $( B _ { i } = ( W _ { i } ) ^ { \top } )$ and asymmetric $( B _ { i } \neq$ $( W _ { i } ) ^ { \top } )$ recurrent connections in the study, and compare the results with FNNs of the same size trained by BP (feedback connections removed) or feedback alignment (FA) (Lillicrap et al., 2016) that uses random weights $B _ { i } \neq ( W _ { i } ) ^ { \top }$ to feedback the error. Note that, after scaling, the overall weight matrix $W$ of a symmetric RNN is no longer strictly symmetric. Therefore, we started from the vector field setting of EP rather then the energy-based setting in the first place. The feedforward and feedback weights are multiplied by coefficients $\alpha _ { i }$ and $\beta _ { i }$ respectively. Figure 2a-d shows convergence speed for different $\beta _ { i }$ . With asymmetric weights, the network can converge to a fixed point (Figure 2e, f), exhibit cyclical oscillation (Figure 2g, h), or even become chaotic. The feedback weights stay fixed during training process, which differs from EP in vector field dynamics (Scellier et al., 2018). The pseudocode of learning procedure with a 2-hidden-layer RNN shown in Figure 1(a) is provided in Algorithm 1.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 EP with Feedforward and Feedback Scaling
Require: Input: $(x, s_{tar})$
Require: Parameters: $\theta = [W_0, W_1, W_f, B_f, B_1, \alpha_1, \beta_1, \beta_{f1}]$
Ensure: Updated parameters $\theta$
1: function FIRST-PHASE($\theta, s_{tar}$)
2:    $s_0 \leftarrow x$
3:    for t = 1 to T do
4:    $h_1 \leftarrow W_0 \cdot s_0 + \beta_1 \cdot B_1 \cdot s_2^0$
5:    $h_2 \leftarrow \alpha_1 \cdot W_1 \cdot s_1^0$
6:    $h_p \leftarrow W_f \cdot s_2^0$
7:    $s_1^0, s_2^0, s_p^0 \leftarrow \rho(h_1), \rho(h_2), \text{SoftMax}(h_p)$
8:    end for
9:    $\Lambda_1 \leftarrow [s_i^0], i = 0, 1, 2, p$
10:    return $\Lambda_1$
11: end function
12: function SECOND-PHASE($\theta, \Lambda_1, s_{tar}$)
13:    $s_1^{\beta_{f1}}, s_2^{\beta_{f1}}, s_p^{\beta_{f1}} \leftarrow s_1^0, s_2^0, s_p^0$
14:    for t = 1 to K do
15:    $e_p \leftarrow s_{tar} - s_p^{\beta_{f1}}$
16:    $h_1 \leftarrow W_0 \cdot s_0 + \beta_1 \cdot B_1 \cdot s_2^{\beta_{f1}}$
17:    $h_2 \leftarrow \alpha_1 \cdot W_1 \cdot s_1^{\beta_{f1}} + \beta_f \cdot B_f \cdot e_p$
18:    $h_p \leftarrow W_f \cdot s_2^{\beta_{f1}}$
19:    $s_1^{\beta_{f1}}, s_2^{\beta_{f1}}, s_p^{\beta_{f1}} \leftarrow \rho(h_1), \rho(h_2), \text{SoftMax}(h_p)$
20:    end for
21:    $ds_i \leftarrow s_i^{\beta_{f1}} - s_i^0, i = 1, 2$
22:    $\Lambda_2 \leftarrow [ds_1, ds_2]$
23:    return $\Lambda_2$
24: end function
25: function UPDATING-WEIGHTS($\theta, \Lambda_1, \Lambda_2, s_{tar}$)
26:    $\Delta W_i \leftarrow ds_{i+1} \cdot (s_i^0)^\top, i = 0, 1$
27:    $\Delta W_f \leftarrow (s_{tar} - s_p^0) \cdot (s_2^0)^\top$
28: end function
</div>

## 3.3 RESIDUAL CONNECTIONS TO AVOID VANISHING GRADIENTS

In our 10-hidden-layer RNN with symmetric connections, we add cross layer residual links $( { \mathrm { F i g } } .$ ure 3a-b) and carry out ablation study on their effects in performance. The three long-range bidirectional connections bypass adjacent layers to reduce gradient decay. For RNN with asymmetric connections, we introduce skip-layer connections between non-adjacent layers with $P = 2 0 \%$ probability, creating an RNN with arbitrary graph topologies (AGT) where any pair of layers form connections stochastically (Figure 3c) (Salvatori et al., 2022). (See Algorithm S3 in Appendix D for training detail)

## 4 EXPERIMENTS

We evaluated our RNN models on MNIST and CIFAR-10 datasets and compared the results with $\mathrm { P } \mathrm { \cdot }$ EP and BP. The MNIST dataset consists of 70,000 grayscale handwritten digit images (28×28 pixels) split into 60,000 training and 10,000 test samples. CIFAR-10 contains 60,000 RGB images (32×32 pixels) of 10 categories, divided into 50,000 training and 10,000 test samples. Pre-processing, network structures and additional training details are in Appendix D.

## 4.1 INFLUENCE OF FEEDFORWARD SCALING AND FEEDBACK SCALING

Figure 4 compares the effects of feedforward scaling $\alpha _ { i }$ and feedback scaling $\beta _ { i } .$ In general, relative small feedback scaling $( \beta _ { i } = 0 . 1 )$ yields high MNIST accuracy (Figure 4). In deeper RNNs, overly low feedback scaling $\beta _ { i }$ jeopardizes the performance, which we attribute to vanishing gradients (Figure 4c, right two columns). In contrast, down-scaling the feedforward weights degrades performance, as the inference signals are weakened through the layers (see rows of Figure 4a). However, up-scaling $\alpha _ { i }$ can also be detrimental, as this easily leads to saturation of neural state. The best performance of a 5-hidden-layer RNN is achieved without feedforward scaling $\alpha _ { i } = 1$ and a trade-off in feedback scaling at $\beta _ { i } = 0 . 1$ . These results suggest that balancing the feedforward and feedback strengths is critical for better performance, not only accuracy but also speed (see Table 1).

(a)  
![](images/f959715ac9c82729c057cc4f7f9ec953aaece70f01727f5fe674f4f68bf2aaa0.jpg)

![](images/8c6c8e98611b58eba1ead3a8d6c8ae1e4e73382119343b7a4aa67c4585d9bf35.jpg)

![](images/240ae04767bbf2520ba0993d28218672c23c4964f0722b1a5abf9d045dae3bdc.jpg)  
Figure 3: (a) A 10-hidden-layer RNN model with residual connections. The solid blue wires and the dashed orange wires represent forward and feedback residual connections respectively. The bidirectional connections are symmetric. (b) Adjacency matrix of (a). The blocks (green) other than the sub-diagonals indicate residual connections. (c) Adjacency matrix for an RNN with arbitrary graph topology.

(a)  
![](images/a9a76849dbe534fded4e8e29415017232a747b8e3b8a1b27ad0c005288d3d153.jpg)

(b)  
![](images/93e9c7f6b6f3cbcb7fdb8680413e75c77a9e176c4c72036ac976015ee79da736.jpg)

(c)  
![](images/97728ac384ae523a4414c2e870dbce79a6e30722dadec62568c3e9dd956e3cce.jpg)  
Figure 4: The influence of feedforward scaling $\alpha _ { i }$ and feedback scaling $\beta _ { i }$ on accuracy of MNIST classification. (a) 2 hidden layers; (b) 3 hidden layers; (c) 5 hidden layers. Each layer has 64 neurons. By default, $T = 1 0 \times N _ { H i d d e n L a y e r } , K = T / 2$ . Each result is averaged over five repetitive experiments.

![](images/3e3a4cc7ca6412ae9378d3073a3dbfa066970a7704f40c567eb020e4637a01cf.jpg)  
i

(b)  
![](images/be5a6c975fd4c5f21713bbe2f0ce1c213376963967dbc225e7658e587997019b.jpg)  
i

(c)  
![](images/9d8dc3e00f2d5736b8e8b4215051dc305c4fa506b6924d8b1e95d9716e0dc213.jpg)  
i

(d)  
![](images/a43f0f8430cb00d4f386a703a9ee8d5f7f9631045a453c15996c4f6345082793.jpg)  
i  
Figure 5: The test error, SR, FTMLE, and convergence time versus feedback scaling $\beta _ { i }$ . The results are obtained from a 3-hidden-layer (64 neurons per layer) model trained on MNIST dataset. Note that the network does not converge under certain conditions, resulting in missing value in d.

To further investigate the influence of feedback scaling $\beta _ { i } ,$ we plot the error, SR, finite time maximum Lyapunov exponent (FTMLE) (Shadden et al., 2005; Kanno & Uchida, 2014) and convergence time against feedback scaling coefficient before and after training of a 3-hidden-layer RNN on MNIST (Figure 5). It shows that larger feedback scaling $\beta _ { i }$ decreases accuracy (Figure 5a). As expected, SR is positively correlated to $\beta _ { i }$ (see Figure 5b), and large SR can lead to instability of an RNN indicated by the FTMLE shown in Figure 5c, which in turn explains the results in Figure 5a. In general, down-scaling the feedback $( \beta _ { i } < 1 )$ reduces the convergence time of RNN, which is favorable. Note that up-scaling of feedback $\beta _ { i } > 1$ can also decrease FTMLE and convergence time. However, this is attributed to the saturation of neural state, and will also lower the performance.

Additionally, one might suspect that the gradient signals in the lower layers are not fulfilling their intended role. In reservoir computing, where only the last layer is trained, the network can also reach high accuracy as long as the output dimension is large enough. However, this is unlikely in our case, as each layer in our network has only 64 neurons by default (other than the results in Table 1). To further confirm that the learning in lower layers is meaningful, we performed training with the weights of lower layers frozen—details of these experiments are included in Appendix C.5. The results clearly show that getting comparable results to BP requires effective training of lower layers.

![](images/e722ba276d0a5d74743b0229d50ba838e939e7c5c1bf655895a5ac58994ef0e1.jpg)

(b)  
![](images/8d5f55b6d200a9e586eff3af6abe6775679547e5add34190343b9d95ba9d45ff.jpg)

![](images/9b42390e06400e91fd00689e967816ca9c0fbdaa6e13b799936ed12aa4fb8eef.jpg)

(f)  
![](images/b8c0da77c021f8c73a4c5d71c892e9a568724faa35056e591e60dfc161c106c0.jpg)

![](images/527b70a6b8807233dcdd4634cd79edc3307713354a7277778929e7cc583206bf.jpg)

(g)  
![](images/51fc2a6375a35ba45f45e94555d731cc8c3d3e6b4ca0a80c204d69de74f7f8cf.jpg)

![](images/091fa0a58a6e4dc2f7cf7a04eaca828619a08334faaa25b6ac5e7211273c3ed3.jpg)

(h)  
![](images/ecf1ae81aa3c83373209a5b2d725b39604338eaeb7316ea33f65a022988dd89d.jpg)  
Figure 6: Test error with different hyperparameters. The curves of different T (10, 20, 50, 100) with 2 hidden layers (64 neurons per hidden layer) and (a) $\beta _ { i } = 0 . 0 1$ ; (b) $\beta _ { i } = 0 . 1 ; ( \mathrm { c } ) \beta _ { i } = 1 ;$ (d) $\beta _ { i } = 4 .$ . The curves of different $\beta _ { i }$ (0.001, 0.01, 0.1, 0.25, 1, 2, 4) with (e) 2 hidden layers; (f) 3 hidden layers; (g) 5 hidden layers; (h) 10 hidden layers. The shaded areas represent deviations of five repeated experiments. By default, $T = 1 0 \times \bar { N } _ { H i d d e n L a y e r } , K = T / 2$ . See Appendix A for more information.

## 4.2 DOWN-SCALING FEEDBACK LEADS TO FASTER CONVERGENCE

Figure 6a-d plots the error versus the number of epochs with different iteration steps T. Under the condition of $\beta _ { i } = 0 . 0 1$ (Figure 6a), the model with $T = 1 0$ and $K = 5$ works as well as the model with $T = 1 0 0$ and $K = 5 0 ,$ , suggesting possibility of speedup in training. Larger $\beta _ { i }$ requires more iterations to achieve a certain level of performance (See Figure 6b, c, d). Larger $\beta _ { i }$ means larger SR and FTMLE, thus requiring more iterations to settle the RNN as shown in Figure 2 and Figure 5(bd). Or even worse, the gradient signal is completely distorted. $\begin{array} { r } { \operatorname { A t } \beta _ { i } = 4 , } \end{array}$ , even T=100 fails to exceed 95% accuracy. Figure 6e-h shows that while shallow networks benefit from low $\beta _ { i }$ , deeper networks (3, 5 and 10 layers) lose accuracy. In all cases, training performance peaks at certain $\beta _ { i }$ dependent on the network depth. Additional results are provided in Table S1 in Appendix B.

Table 1: Comparison with P-EP and BP in accuracy and cost. The results of P-EP come from previous work (Ernoult et al., 2019). For BP results, we used a network with the same number of layers and number of nodes/channels. Each experiment is repeated five times, and the standard deviation is given. By default, $\beta _ { i } = 0 . 0 1$ in our results, the feedback weights are symmetric with the feedforward weights for P-EP and Ours, and the learning rate in all layers are the same except for Ours-DLR (different learning rate), which uses varying learning rates identical to that of P-EP. For 2HL (two hidden layers) and 3HL (three hidden layers), there are 512 nodes per hidden layer. See Appendix D for more details.

<table><tr><td>Architecture</td><td>Training approach</td><td>Testing (Training)</td><td>Epoch / Batch size -T/K</td><td>Wall Clock Time (HH:MM:SS)</td></tr><tr><td rowspan="3">2HL</td><td>P-EP (sigmoid-s)</td><td>98.05%±0.10% (99.86%)</td><td>50/20-100/20</td><td>1:56:-</td></tr><tr><td>Ours (tanh, Adam)</td><td>98.39%±0.04% (100.00%)</td><td>50/500-10/10</td><td>0:01:16</td></tr><tr><td>BP (tanh, Adam)</td><td>98.26%±0.06% (100.00%)</td><td>50/500-1/1</td><td>0:00:18</td></tr><tr><td rowspan="5">3HL</td><td>P-EP (sigmoid-s)</td><td>97.99%±0.18% (99.90%)</td><td>100/20-180/20</td><td>8:27:-</td></tr><tr><td>Ours-DLR (tanh)</td><td>97.65%±0.08% (98.93%)</td><td>100/20-18/10</td><td>1:01:14</td></tr><tr><td>Ours (tanh)</td><td>97.83%±0.13% (99.98%)</td><td>100/20-18/10</td><td>1:01:54</td></tr><tr><td>Ours (tanh, Adam)</td><td>98.36%±0.06% (100.00%)</td><td>50/500-18/10</td><td>0:02:11</td></tr><tr><td>BP (tanh, Adam)</td><td>98.36%±0.08% (100.00%)</td><td>50/500-1/1</td><td>0:00:24</td></tr><tr><td rowspan="3">Conv</td><td>P-EP (hard-sigmoid)</td><td>98.98%±0.04% (99.46%)</td><td>40/20-200/10</td><td>8:58:-</td></tr><tr><td>Ours (hard-sigmoid)</td><td>99.14%±0.02% (99.78%)</td><td>40/128-20/10</td><td>0:12:28</td></tr><tr><td>BP (hard-sigmoid)</td><td>98.93%±0.18% (99.43%)</td><td>40/128-1/1</td><td>0:01:01</td></tr></table>

Table 2: Comparison with BP and FA and ablation study of residual connection. For layered architecture, there are 64 nodes per hidden layer and we chose $T = 1 0 \times N _ { H i d d e n L a y e r } ,$ and $K = 5 \times N _ { H i d d e n L a y e r }$ , which guarantees saturation of accuracy at $\beta _ { i } = 0 . 1$ . For convolutional architectures, $\beta _ { i } = 0 . 0 1$ . By default, the Adam optimizer is used. Each experiment is repeated five times. See Appendix D for more training details.

<table><tr><td>Architecture -connections</td><td>Training approach</td><td>MNIST-Testing (Training)</td><td>CIFAR-10-Testing (Training)</td></tr><tr><td rowspan="2">5-symm</td><td>BP</td><td>97.69%±0.10% (100.00%)</td><td>49.23%±0.81% (56.72%)</td></tr><tr><td>Ours</td><td>97.64%±0.10% (99.98%)</td><td>50.72%±0.17% (57.02%)</td></tr><tr><td rowspan="2">5-asymm</td><td>FA</td><td>96.44%±0.10% (98.96%)</td><td>37.97%±2.18% (38.92%)</td></tr><tr><td>Ours</td><td>96.37%±0.11% (97.99%)</td><td>45.27%±0.73% (46.79%)</td></tr><tr><td rowspan="3">10-symm</td><td>BP</td><td>97.61%±0.04% (99.93%)</td><td>48.23%±1.26% (55.37%)</td></tr><tr><td>Ours</td><td>92.49%±0.32% (95.27%)</td><td>34.90%±0.38% (34.64%)</td></tr><tr><td>Ours-Residual</td><td>97.49%±0.05% (99.77%)</td><td>44.46%±0.51% (48.67%)</td></tr><tr><td rowspan="3">10-asymm</td><td>FA</td><td>94.52%±0.26% (95.54%)</td><td>30.16%±6.12% (30.20%)</td></tr><tr><td>Ours</td><td>87.37%±0.49% (87.95%)</td><td>30.37%±1.09% (29.97%)</td></tr><tr><td>Ours-AGT</td><td>96.87%±0.11% (99.45%)</td><td>30.94%±4.90% (31.36%)</td></tr><tr><td rowspan="2">20-symm</td><td>BP</td><td>97.48%±0.07% (99.74%)</td><td>47.35%±1.49% (54.59%)</td></tr><tr><td>Ours-Residual</td><td>95.95%±0.18% (98.20%)</td><td>43.61%±1.17% (44.26%)</td></tr><tr><td rowspan="2">Conv</td><td>BP</td><td>99.34%±0.04% (99.97%)</td><td>75.45%±0.46% (83.61%)</td></tr><tr><td>Ours</td><td>99.27%±0.07% (99.78%)</td><td>75.04%±0.51% (80.79%)</td></tr></table>

Table 1 compares our approach with P-EP, BP, and FA. Our model supersedes P-EP in training speed by at least one order of magnitude for both convolutional architecture and layered architecture. Importantly, our accuracy is comparable to BP and FA for the shallow architectures (5-hidden-layer and conv model, see also Table 2). In consideration of the improved stability (Figure 6) via feedback regulation, we anticipate that physical implementations of RNN can achieve performance on par with BP. Additionally, for layered architecture, we also adopt the same training parameters (learning rate, batch size and epochs) as P-EP, differing only in feedback scaling (‘ours-DLR’ in Table 1). The results present clear evidence of speedup, which mainly stems from the reduced number of iterations required for convergence.

## 4.3 DOWN-SCALED FEEDBACK COORDINATES PLASTICITY OF DIFFERENT LAYERS

It is hypothesized that the brain requires different plasticity in different areas due to their varying functional roles (Atallah et al., 2004; Lowet et al., 2020). The variability in plasticity can be realized explicitly by adjusting learning rates or implicitly by modulating the intensity of gradient. Previous work postulated that EP with weak feedback necessitates learning rates differing by orders of magnitude across layers (Scellier & Bengio, 2017). Here, we found that due to gradient differences across different layers induced by weak feedback, a 3-hidden-layer RNN at $\beta _ { i } = 0 . 0 1$ (Table 1, ‘ours (tanh)’) learns well with a uniform learning rate. This result suggests that the feedback scaling alone is able to regulate gradient strength of different layers, pointing to another possible mechanism to coordinate plasticity.

## 4.4 RESIDUAL CONNECTIONS OVERCOME THE GRADIENT VANISHING IN DEEP RNNS

Weak feedback exacerbates vanishing gradient in deeper layered RNN (Figures S5–S6 in Appendix B). Adding residual connections restores gradient flow (Figure S7 in Appendix B). As a result, a 10-hidden-layer network sees substantial performance gains (Table 2), 5% increase in accuracy for MNIST and 9% for CIFAR-10. Even 20-hidden-layer model can be trained. As shown in Table 2, without residual connections, an asymmetric RNN trained by EP falls short of FA in accuracy, but arbitrary residual links surpass the accuracy of FA for the MNIST classification (See ablation study on connection probability in Appendix B.). For more complex dataset CIFAR-10, the 10-hidden-layer asymmetric model with residual random feedback connections achieves accuracy nearly 14% below the symmetric model. A possible reason is that the gradient signal through multiple random fixed feedback connections becomes too distorted by error to coordinate the forward weight learning.

## 5 DISCUSSION

We have applied the feedback scaling to RNN to speed up the convergence and to accelerate training with EP with negligible overhead. To counteract the vanishing gradient in deep architectures, we have added residual connections to non-adjacent layers of deep RNNs, partly restoring classification performance. In principle, the residual connections make credit assignment pathways shorter (Veit et al., 2016). The training exhibits remarkable resilience to noise on weight and neural state. Our structural modification is compatible with other algorithmic speed-ups (Scellier et al., 2023), thereby expanding the design space for efficient EP implementations.

Recent work on credit assignment in brain-inspired networks, e.g. adjoint propagation (Liu et al., 2026), partitions a large network into local RNNs with random internal connections of low SR for fast convergence and dynamic resource allocation, yielding speed and accuracy similar to this work. This work, however, adopts the feedback scaling to solve the stability issue and accelerate convergence of EP.

Weak feedback is often considered in biologically plausible learning algorithms (Sacramento et al., 2018; Haider et al., 2021; Meulemans et al., 2021). It has been shown that contrastive Hebbian learning with weak feedback approximates backpropagation while converging quickly (Xie & Seung, 2003). More recently, local representation alignment (LRA) likewise employed weak feedback (Ororbia et al., 2023) and skip connections from the output to deep layers for efficient training. The EP framework also approximates BP (Scellier & Bengio, 2017; Millidge et al., 2023), but under the weak clamping condition (weak supervision) (Laborieux et al., 2021; Millidge et al., 2023). We have shown that, at the infinitesimal inference limit, namely weak supervision and weak feedback (Millidge et al., 2023), EP is equivalent to LRA and BP (Appendix C). In other words, the dynamics of FRE-RNN is more like the feedforward neural network due to its weak feedback.

However, there are still a few limitations to our approaches for large-scale neural networks that underpin artificial intelligence. For complex datasets like CIFAR-10, there exists a notable performance gap compared to BP, using deep fully connected neural networks. We attribute this gap to the inaccurate approximation to the true gradient as computed by BP (See Appendix C.4). Therefore, although EP can be extended to deep fully connected network (20-hidden-layers) and shallow CNNs, its applicability for deep CNN remains to be explored. For deep architectures with asymmetric connections, the accuracy decreases faster with increasing depth due to the inaccurate random error feedback. More in-depth investigation on residual connection topology is required to scale up the methodology to large scale deep architectures. Besides, the hyperparameters are optimized empirically. We find a feedback scaling in the range of 0.01-0.1 is favorable for shallow networks (less than 4 layers) and 0.1-0.25 for deeper architectures. Finding a general way to determine these parameters is still on-going. Additionally, existing research on EP converging naturally continues to focus primarily on static-input settings (Laborieux et al., 2021; Ernoult et al., 2019; Laborieux & Zenke, 2024). Extending naturally converging RNN trained by EP to sequence tasks remains a challenge.

From a neurobiological perspective, residual connections, particularly the randomly generated arbitrary graph topologies, yield cortex-like connectivity patterns in the brain. The feedback-regulated residual RNNs equip the biologically plausible learning framework, EP, with biologically plausible network architecture. Although it currently runs on GPUs, it can exploit the natural convergence of physical RNNs and facilitate efficient learning and inference on dedicated neuromorphic hardware.

## ACKNOWLEDGEMENTS

This work was supported by the National Key R&D Program of China (Grant No. 2024YFA1208804). Additional financial support from the University of Science and Technology of China and the Chinese Academy of Sciences is also gratefully acknowledged.

## CODE AVAILABILITY

The code used in this work is available at https://github.com/Zero0Hero/ FRE-RNN-EP.

## REPRODUCIBILITY STATEMENT

The code necessary to reproduce the main results is provided as Jupyter Notebooks in the Supplementary Materials. Researchers can directly run them to reproduce the results. Further details on data pre-processing and training process are available within the provided code and in Appendix D.

## THE USE OF LARGE LANGUAGE MODELS (LLMS)

In the preparation of this work, the authors used GPT-5 and DeepSeek solely for the purpose of polishing and improving the linguistic fluency and readability of the text. This includes tasks such as correcting grammar and rephrasing sentences. After using the model, the authors have reviewed and edited all content extensively and take full responsibility for all ideas, claims, and the final language presented in this paper.

## REFERENCES

David H. Ackley, Geoffrey E. Hinton, and Terrence J. Sejnowski. A learning algorithm for boltzmann machines. Cognitive Science, 9(1):147–169, 1985.

Hisham E. Atallah, Michael J. Frank, and Randall C. O’Reilly. Hippocampus, cortex, and basal ganglia: Insights from computational models of complementary learning systems. Neurobiology of Learning and Memory, 82(3):253–267, 2004. ISSN 1074-7427. doi: https://doi. org/10.1016/j.nlm.2004.06.004. URL https://www.sciencedirect.com/science/ article/pii/S1074742704000693.

Zhang Bai, D. J. Miller, and Wang Yue. Nonlinear system modeling with random matrices: Echo state networks revisited. IEEE Transactions on Neural Networks and Learning Systems, 23(1): 175–182, 2012. ISSN 2162-237X 2162-2388. doi: 10.1109/tnnls.2011.2178562.