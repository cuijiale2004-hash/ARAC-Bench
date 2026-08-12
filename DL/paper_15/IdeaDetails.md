**1. Research Background and Existing Pain Points**

**Research Background:**
Unsupervised learning has garnered significant attention in artificial intelligence for its capacity to extract meaningful representations from unlabeled data, substantially reducing dependence on extensive manual annotation. By revealing inherent structures and patterns in unlabeled data, this approach more accurately reflects natural human learning processes. Spiking Neural Networks (SNNs), with their characteristics of simulating brain functioning principles, constitute an ideal framework for unsupervised learning research. The temporal processing capability of SNNs stems from the intrinsic dynamics of spiking neurons, which serve as information carriers across timesteps. Standard Leaky Integrate-and-Fire (LIF) neurons accumulate membrane potential to retain temporal information and emit discrete spikes when the potential exceeds a threshold. 

**Existing Pain Points:**
1.  **Limited Long-Range Temporal Dependencies:** Current unsupervised learning research in SNNs has primarily concentrated on shallow architectures or synaptic plasticity-based methods. The challenges in extending these approaches to deep architectures, particularly when processing complex temporal data, predominantly arise from the limited capacity of current deep SNN models to effectively capture and leverage long-term temporal dependencies.
2.  **Inadequate Intrinsic Neuronal Dynamics:** This elementary integrate-and-fire mechanism proves inadequate for processing large-scale video data with complex temporal dependencies. SNNs typically preserve original temporal resolution, potentially resulting in feature instability without appropriate temporal aggregation. Consequently, intrinsic neuronal dynamics alone are insufficient for complex temporal information processing, necessitating the integration of explicit temporal modeling mechanisms.
3.  **Semantically Unstable Representations:** The limitations in modeling long-range dependencies and maintaining temporal feature consistency result in semantically unstable representations, thereby impeding the development of deep unsupervised SNNs for large-scale temporal video data.
4.  **Lack of Systematic Benchmarks:** There is a scarcity of unsupervised methods for SNNs and a lack of systematic investigations into self-supervised learning for deep spiking neural networks, leaving the effective application of these methods to SNNs an urgent problem requiring resolution. Conventional DVS datasets are small in scale, failing to meet the demands for large-scale temporal video data processing.

**2. Core Research Motivation and Scientific Questions**

**Core Research Motivation:**
Effective temporal modeling should enhance consistency among features extracted across different timesteps. Analysis shows that as models converge, semantic extraction capability improves significantly while feature distributions across timesteps become increasingly consistent. Ideally, high-consistency SNNs should extract stable high-level semantic features (action types, object categories) that remain invariant to temporal fluctuations. While directly constraining temporal consistency might seem intuitive, experiments reveal that such enforced consistency constraints actually impair performance. There is a need for a mechanism that naturally improves cross-temporal feature consistency without suppressing critical temporal dynamics.

**Scientific Questions:**
1.  How can deep unsupervised SNNs effectively capture and leverage long-range temporal dependencies in large-scale video data?
2.  How can temporal feature consistency be enhanced without resorting to forced consistency constraints that degrade feature extraction capabilities and suppress temporal dynamics?
3.  Can explicit temporal relationship modeling through prediction serve as an implicit regularizer to filter out unpredictable low-level noise while preserving semantically rich features?

**3. Overall Core Idea and Design Philosophy**

**Overall Core Idea:**
The paper proposes PredNext, which explicitly models temporal relationships through cross-view future Step Prediction and Clip Prediction. This plug-and-play module seamlessly integrates with diverse self-supervised objectives.

**Design Philosophy:**
Based on Predictive Coding theory, PredNext explicitly models temporal relationships through future representation prediction. This approach operates on the principle that semantically rich features should accurately predict their next semantical feature, whereas features capturing only low-level dynamics cannot generate effective predictions. Video data contains semantic content (long-range correlation) and low-level noise (exponential decay). By predicting future features, the optimization process naturally prioritizes encoding predictable semantic content while filtering unpredictable noise. Predictability serves as an implicit regularizer that filters out unpredictable noise, unlike forced consistency which indiscriminately suppresses all temporal variations. By explicitly modeling temporal relationships both within and between clips, features with higher semantic density better predict future representations while excluding low-level dynamic information, thus naturally improving cross-temporal feature consistency.

**4. Core Innovation Points**

1.  **Explicit Cross-View Temporal Prediction (PredNext):** Proposed PredNext, a plug-and-play module seamlessly integrable with diverse self-supervised learning frameworks, which explicitly models temporal relationships through cross-view future Step Prediction and Clip Prediction to enhance temporal feature consistency.
2.  **Step and Clip Prediction Mechanisms:** Designed two complementary mechanisms: Step Prediction (predicts representations at subsequent timesteps) and Clip Prediction (predicts features from future temporal clips), while cross-view prediction enhances feature discrimination by requiring the model to disregard view-specific noise.
3.  **Establishment of Standard Benchmarks:** First to establish standard benchmarks for SNN self-supervised learning on UCF101, HMDB51, and MiniKinetics, which are substantially larger than conventional DVS datasets, confirming that SNNs benefit significantly from data scale.
4.  **Analysis of Consistency vs. Prediction:** Demonstrated that superior feature extraction capability corresponds to higher temporal feature consistency, and explicitly enforcing consistency constraints degrades performance by suppressing critical temporal dynamics, whereas prediction-based consistency improves generalization.
5.  **SNN Adaptation of Self-Supervised Methods:** Adapted prevailing self-supervised methods (SimCLR, MoCo, SimSiam, BYOL, BarlowTwins) to SNN architectures to establish comparative baselines for deep unsupervised learning in SNNs.

**5. Overview of the Overall Technical Solution**

The overall technical solution involves integrating PredNext into existing self-supervised learning frameworks for SNNs. 
1.  **Baseline Adaptation:** Prevailing self-supervised methods (SimCLR, MoCo, SimSiam, BYOL, BarlowTwins) are adapted to SNN architectures using SEW ResNet as the backbone. Temporal features are aggregated following the SNN encoder.
2.  **PredNext Integration:** PredNext serves as an auxiliary module introducing temporal prediction as an auxiliary objective while preserving the original self-supervised paradigm. It consists of an SNN feature extractor and a nonlinear MLP projection head (F), alongside two temporal prediction heads (PT and PC).
3.  **Prediction Mechanism:** For augmented clips, representations are obtained. The Step Predictor PT maps current features to future timestep features, while the Clip Predictor PC maps current clip representations to future clip representations.
4.  **Cross-View Design:** Features from one view (pti, ci) predict future features of another view (zt+mj, z*j) to enhance discrimination. Stop-gradient is applied to the target features.
5.  **Loss Combination:** The final optimization objective combines the original self-supervised loss and the prediction loss, balanced by a weight coefficient α.

**6. Detailed Module Design**

**A. Self-Supervised Learning in SNNs (Base Modules)**
Given a clip x of length t sampled from dataset D. Through data augmentation H(x), two views x_i^t and x_j^t are obtained. These are processed through feature extractors and MLP projection heads to yield representations z_i^t and z_j^t. The time-averaged representation z_i = \sum_{t=1}^T z_i^t / T is computed as the final feature.
*   **SimCLR & MoCo:** Utilize the InfoNCE loss function. SimCLR utilizes in-batch samples as negative examples, whereas MoCo maintains a dynamic feature queue for negative samples with a momentum encoder.
*   **SimSiam & BYOL:** Employ a predictor network h that maps representations between views while minimizing their distance. BYOL employs a momentum encoder, while SimSiam utilizes a weight-shared siamese network with stop-gradient.
*   **BarlowTwins:** Minimizes feature redundancy using the cross-correlation matrix of batch-normalized features.

**B. PredNext Module**
PredNext comprises three main components:
1.  **Feature Extractor and Projection Head (F):** SNN feature extractor and a nonlinear MLP projection head, jointly denoted as F.
2.  **Step Predictor (PT):** Establishes mappings between current and future timestep features. It is a 2-layer MLP with batch normalization and dimensions matching the projection head output. It predicts features at subsequent timesteps.
3.  **Clip Predictor (PC):** Models relationships between current and future clip representations. It is a 2-layer MLP. It predicts features from future temporal clips.

**C. Cross-View Prediction Mechanism**
For augmented clips x_i^t and x_j^t, representations z_i^t = F(x_i^t) and z_j^t = F(x_j^t) are obtained. Predictions are generated through:
*   **Step Prediction:** p_i^t = PT(F(x_i^t)), p_j^t = PT(F(x_j^t))
*   **Clip Prediction:** c_i = PC(1/T \sum_t F(x_i^t)), c_j = PC(1/T \sum_t F(x_j^t))

Cross-view prediction is employed where features from one view (pti, ci) predict future features of another view (zt+mj, z*j), with stop-gradient applied to the target features. This enhances feature discrimination by requiring the model to disregard view-specific noise.

**7. All Mathematical Formulas and Symbol Definitions**

**Symbol Definitions:**
*   x: A clip of length t sampled from dataset D.
*   H(x): Data augmentation function.
*   x_i^t, x_j^t: Two augmented views of the clip.
*   z_i^t, z_j^t: Representations obtained from feature extractors and MLP projection heads.
*   z_i, z_j: Time-averaged representations.
*   sim(\cdot, \cdot): Cosine similarity.
*   \tau: Temperature parameter.
*   N: Batch size.
*   h: Predictor network in SimSiam/BYOL.
*   C: Cross-correlation matrix in BarlowTwins.
*   \lambda: Hyperparameter balancing objectives in BarlowTwins.
*   F: SNN feature extractor and projection head.
*   PT: Step Predictor.
*   PC: Clip Predictor.
*   p_i^t, p_j^t: Predicted features for Step Prediction.
*   c_i, c_j: Predicted features for Clip Prediction.
*   m: Prediction time step interval.
*   z*_i, z*_j: Temporally aggregated features of the subsequently sampled clip.
*   \alpha: Weight coefficient balancing self-supervised loss and prediction loss.
*   Q: Step Predictor’s loss function.
*   M: Clip Predictor’s loss function.
*   E_consistency: Feature consistency error.
*   f_i^t: Video i’s feature at time t.
*   \beta: Constraint intensity for forced consistency.

**Mathematical Formulas:**

1.  **SimCLR and MoCo Loss (InfoNCE):**
    L = - \log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^N \exp(\text{sim}(z_i, z_k)/\tau)}

2.  **SimSiam and BYOL Loss:**
    L = 1 - \frac{z_j}{\|z_j\|_2} \cdot \frac{h(z_i)}{\|h(z_i)\|_2}

3.  **BarlowTwins Loss:**
    L = \sum_i (1 - C_{ii})^2 + \lambda \sum_i \sum_{j \neq i} C_{ij}^2

4.  **Step Predictor’s Loss Function:**
    Q(p_i^t, z_j^{t+m}) = - \sum_t \frac{p_i^t}{|p_i^t|} \cdot \frac{z_j^{t+m}}{|z_j^{t+m}|}

5.  **Clip Predictor’s Loss Function:**
    M(c_i, z*_j) = - \frac{c_i}{|c_i|} \cdot \frac{z*_j}{|z*_j|}

6.  **PredNext Final Loss (Symmetric Design):**
    L_{pred} = \sum_t \left( \frac{1}{2} Q(p_i^t, z_j^{t+m}) + \frac{1}{2} Q(p_j^t, z_i^{t+m}) \right) + \frac{1}{2} M(c_i, z*_j) + \frac{1}{2} M(c_j, z*_i)

7.  **Total Optimization Objective:**
    L = (1 - \alpha) \cdot L_{ssl} + \alpha \cdot L_{pred}

8.  **Feature Consistency Error:**
    E_{consistency} = \frac{1}{N} \frac{1}{T(T-1)} \sum_{i=1}^N \sum_{t=1}^T \sum_{s=1, s \neq t}^T \left( 1 - \cos(f_i^t, f_i^s) \right)

9.  **Forced Consistency Loss:**
    L_{forced} = L_{ssl} + \beta \cdot E_{i,t,s}[1 - \cos(f_i^t, f_i^s)]

**8. Algorithm Pseudocode**

**Algorithm 1 PredNext Training Procedure**
Require: Dataset D, data augmentation function H, feature extractor and projection head F, temporal prediction head PT, PC, self-supervised loss function Lssl, weight coefficient α
Ensure: Trained feature extractor F
1: for each mini-batch do
2: // Get features from two augmented views
3: x_i = H(x), x_j = H(x)
4: z_i^t = F(x_i^t), z_j^t = F(x_j^t) for t = 1...T
5: // Compute original self-supervised loss
6: Lssl = self-supervised loss based on z_i and z_j
7: // Compute PredNext predicted features
8: p_i^t = PT(z_i^t), p_j^t = PT(z_j^t) for t = 1...T - 1
9: c_i = PC(z_i), c_j = PC(z_j)
10: // Compute PredNext loss
11: Lpred = 0.25 \cdot (\sum_t (Q(p_i^t, z_j^{t+m}) + Q(p_j^t, z_i^{t+m})) + M(c_i, z*_j) + M(c_j, z*_i))
12: // Compute total loss and update parameters
13: L = (1 - \alpha) \cdot Lssl + \alpha \cdot Lpred
14: Update parameters of F and PT, PC to minimize L
15: end for