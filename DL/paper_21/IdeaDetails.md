**1. Research Background and Existing Pain Points**
Federated Learning (FL) is a prominent distributed learning paradigm designed to break down data silos, allowing multiple clients to collaboratively train a global model while preserving data privacy. Traditional FL methods rely heavily on a gradient-based optimization paradigm, exemplified by FedAvg and its variants. These gradient-based techniques necessitate iterative optimization processes and are widely acknowledged to suffer from several major challenges:
1. Heterogeneity Issues: Data across clients are often Not Independently and Identically Distributed (Non-IID), severely impacting model performance and convergence.
2. Scalability Issues: As the number of clients increases to a large scale, FL systems experience substantial performance degradation.
3. Convergence Issues: Gradient-based methods may struggle to converge within limited aggregation rounds, particularly in challenging non-IID data or large-scale client scenarios.
4. Overhead Issues: The overall FL process incurs significant overhead from multi-epoch training on each client and multi-round model aggregation across clients.
These challenges are fundamentally rooted in the long-standing reliance on gradient-based updates, which are inherently sensitive and costly in distributed FL scenarios. Existing gradient-based methods can only superficially alleviate these issues rather than fundamentally address them. 

Recently, Analytic Federated Learning (AFL) has been proposed as a gradient-free technique to handle these issues by eliminating gradient-based updates via analytical (closed-form) solutions. By leveraging a frozen pre-trained backbone for feature extraction and building a single-layer linear model with an analytical solution, AFL achieves ideal invariance to data heterogeneity. However, AFL is fundamentally limited by its single-layer linear model. This analytic model can only learn a linear mapping from the frozen backbone’s features to the final output, fundamentally failing to perform representation learning. Consequently, due to the limited learning capacity of the linear analytic model, AFL is prone to underfitting, especially when the backbone itself is lightweight or the output features are insufficient in linear separability. Naive approaches to deepen the analytic model, such as using random projections or adding activation functions without skip connections, fail to effectively improve performance as the number of layers increases, as they remain simply linear mappings or struggle to learn appropriate representations.

**2. Core Research Motivation and Scientific Questions**
The core research motivation stems from the need to overcome the fundamental limitation of existing analytic-learning-based FL approaches, which lack representation learning capability due to their reliance on single-layer linear models. The central scientific question arises: Can we deepen AFL’s analytic model to enable its representation learning capabilities while simultaneously preserving its analytical solutions for invariance to data heterogeneity? Drawing inspiration from the great success of ResNet in gradient-based learning, the motivation is to adopt similar skip connections to boost the representation of the analytic layers, modeling the representation learning as a residual updating process. The challenge lies in effectively learning the residual blocks in a gradient-free manner within the framework of analytic learning, as Stochastic Gradient Descent (SGD) with backpropagation is precluded.

**3. Overall Core Idea and Design Philosophy**
The overall core idea is to propose a Deep Analytic Federated Learning approach, named DeepAFL, that achieves gradient-free representation learning while preserving ideal invariance to data heterogeneity in FL. The design philosophy involves constructing a deep residual analytic network layer by layer. Inherited from AFL, DeepAFL employs a pre-trained backbone for initial feature extraction and incorporates an activated random projection to form zero-layer features. To progressively learn richer representations, DeepAFL continuously refines the features layer by layer using gradient-free residual blocks. Specifically, the feature updating is modeled as Φt = Φt−1+gt(Φt−1), where gt(·) represents a nonlinear feature transformation. To instantiate the residual block gt(Φt−1) with analytical solutions possessing stochasticity, nonlinearity, and learnability, DeepAFL constructs a hidden random feature using a random projection layer with an activation function, and introduces a learnable transformation matrix to adjust and scale this representation. The optimal analytical solution for this transformation matrix is derived via "sandwiched least squares," enabling layer-wise refinement of features and classifiers in a distributed, gradient-free manner.

**4. Core Innovation Points**
1. Conceptually: DeepAFL is a novel approach that achieves gradient-free representation learning while preserving ideal invariance to data heterogeneity in FL, addressing the fundamental limitation of single-layer analytic models.
2. Technically: DeepAFL develops an efficient layer-wise protocol for learning deep analytic models via least squares (specifically, sandwiched least squares). Clients only conduct lightweight forward-propagation computations, allowing the server to aggregate the global models layer by layer without iterative gradient updates.
3. Theoretically: DeepAFL demonstrates two ideal properties: invariance to data heterogeneity and capability of representation learning. It is the first approach to achieve both of these ideal properties simultaneously, ensuring that global model weights are identical to centralized analytical solutions and that empirical risk monotonically non-increases with layer depth.
4. Experimentally: DeepAFL provides extensive evaluations showing superior performance with dual advantages in heterogeneity invariance and representation learning, outperforming state-of-the-art baselines by up to 5.68%–8.42% across three benchmark datasets while maintaining high efficiency.

**5. Overview of the Overall Technical Solution**
The overall technical solution considers a typical FL setting involving one server and K clients. The objective is to construct a deep residual analytic network comprising T layers for deriving the features ΦT and the corresponding global classifier WT in a distributed and gradient-free manner. 
The process begins with each client extracting features using a frozen pre-trained backbone, followed by an activated random projection to form zero-layer features Φ0. The network construction proceeds layer-wise, alternating between deriving the analytic classifier Wt and the residual block transformation matrix Ωt+1. For each layer t, the analytic classifier Wt is derived by minimizing the regularized mean squared error (MSE) loss, yielding a closed-form solution via least squares. To construct the residual block for the next layer, a hidden random feature Ft is generated using random projection and activation. The transformation matrix Ωt+1 is then optimized to minimize the empirical risk given the fixed classifier Wt, framed as a sandwiched least squares problem (a special case of generalized Sylvester matrix equations), which also yields a distinct closed-form solution. The features are updated via the residual connection Φt+1 = Φt + FtΩt+1. 
In the federated implementation, clients compute and upload local Auto-Correlation and Cross-Correlation matrices. The server aggregates these matrices to compute the global Wt and Ωt+1, which are then broadcast to clients for local feature updates, ensuring invariance to data heterogeneity.

**6. Detailed Module Design**
1. Pre-trained Backbone Module: Employs a pre-trained backbone with frozen parameters Θ for initial feature extraction from input images X to obtain X̃ ∈ RN×dX. X̃ = Backbone(X,Θ).
2. Zero-layer Feature Construction Module: Incorporates an activated random projection to boost the features’ representation, forming the zero-layer features Φ0 ∈ RN×dΦ. This increases the feature dimension to a suitable size, boosting linear separability. Φ0 = σ(X̃A).
3. Analytic Classifier Module (for each layer t): Constructs an analytic classifier Wt based on the obtained feature matrix Φt. The optimization objective uses the MSE loss with regularization, yielding an analytical solution via least squares. The global classifier in the FL setting is computed by aggregating local Feature Auto-Correlation Matrices and Label Cross-Correlation Matrices, ensuring equivalence to the centralized analytical solution regardless of data partition.
4. Residual Block Module (for each layer t): Designed to provide gradient-free representation boosting with properties of stochasticity, nonlinearity, and learnability. 
   - Hidden Random Feature: Constructed using a random projection matrix Bt for stochasticity and an activation function σ(·) for nonlinearity. Ft = σ(ΦtBt).
   - Learnable Transformation Matrix (Ωt+1): Provides learnability to adjust and scale the representation Ft. Optimized via the sandwiched least squares problem to minimize the empirical risk given the fixed classifier Wt. The analytical solution is derived using spectral decompositions of aggregated correlation matrices. 
   - Feature Updating: The next-layer features are computed recursively using the residual skip connection. Φt+1 = Φt + FtΩt+1.
5. Theoretical Properties Modules:
   - Invariance to Data Heterogeneity: Theoretical analysis proves that given fixed seeds for random projection matrices, the global model weights {Wt}Tt=0 and {Ωt}Tt=1 are invariant to any data partition, being identical to the centralized analytical solutions.
   - Representation Learning Capability: Theoretical analysis proves that the sequence of empirical risks (both regularized and non-regularized) keeps monotonically non-increasing with increasing layer depth, guaranteeing convergence to a limit.

**7. All Mathematical Formulas and Symbol Definitions**
Dk = {Xk,Yk}: Client k’s local dataset. Xk ∈ RNk×dH×dW×dC (local samples), Yk ∈ RNk×C (labels). Nk: number of local samples. dH × dW × dC: 3 dimensions of input images. C: number of output classes.
X̃ = Backbone(X,Θ) (1). Backbone(·): pre-trained backbone with frozen parameters Θ. X̃ ∈ RN×dX: extracted features.
Φ0 = σ(X̃A) (2). σ(·): activation function. A ∈ RdX×dΦ: random projection matrix. Φ0 ∈ RN×dΦ: zero-layer features. dΦ: feature dimension.
Φt = Φt−1 + gt(Φt−1),∀t ∈ [1, T ] (3). gt(·): residual block as a nonlinear feature transformation.
Wt = argmin_W ∥Y −ΦtW∥2F + λ∥W∥2F,∀t ∈ [0, T ] (4). λ: regularization parameter. ∥ · ∥2F: Frobenius norm.
Wt = (Φ⊤t Φt + λI)−1Φ⊤t Y,∀t ∈ [0, T ] (5). Analytical solution for Wt.
gt+1(Φt) = σ(ΦtBt)Ωt+1 = FtΩt+1,∀t ∈ [0, T ) (6). Bt ∈ RdΦ×dF: random projection matrix for stochasticity. Ωt+1 ∈ RdF×dΦ: trainable transformation matrix. Ft = σ(ΦtBt): hidden random feature. dF: hidden feature dimension.
Ωt+1 = argmin_Ω ∥Y − (Φt + FtΩ)Wt∥2F + γ∥Ω∥2F = argmin_Ω ∥Rt − FtΩWt∥2F + γ∥Ω∥2F,∀t ∈ [0, T ) (7). Rt = Y−(ΦtWt): residual of current layer’s classification. γ: regularization parameter.
Ωt+1 = Vt[(V⊤t F⊤t RtW⊤t Ut)⊘ (γ1+ diag(ΛFt )⊗ diag(ΛWt ))]U⊤t ,∀t ∈ [0, T ) (8). Vt, ΛFt, Ut, ΛWt: from spectral decompositions. ⊘: element-wise division. ⊗: outer product. 1: all-ones matrix.
Φt+1 = Φt + FtΩt+1,∀t ∈ [0, T ) (9). Recursive formula for next-layer features.
Φk0 = σ(Backbone(Xk,Θ)A) (10). Client k’s local zero-layer feature matrix.
Gkt = (Φkt)⊤Φkt , Hkt = (Φkt)⊤Yk (11). Gkt: local Feature Auto-Correlation Matrix. Hkt: local Label Cross-Correlation Matrix.
G1:Kt =∑Kk=1Gkt , H1:Kt =∑Kk=1Hkt (12). Aggregated Auto-Correlation and Cross-Correlation Matrices.
Wt = [(Φ1:Kt )⊤Φ1:Kt + λI]−1(Φ1:Kt )⊤Y1:K = (G1:Kt + λI)−1H1:Kt (13). Global classifier derivation.
Fkt = σ(ΦktBt), Rkt = Yk − (ΦktWt) (14). Fkt: hidden random features. Rkt: local residual matrix.
Πkt = (Fkt)⊤Fkt , Υkt = (Fkt)⊤Rkt (15). Πkt: local Hidden Auto-Correlation Matrix. Υkt: local Residual Cross-Correlation Matrix.
Π1:Kt =∑Kk=1Πkt , Υ1:Kt =∑Kk=1Υkt (16). Aggregated Hidden and Residual Cross-Correlation Matrices.
Ωt+1 = Vt[(V⊤t (F1:Kt )⊤R1:Kt W⊤t Ut)⊘ (γ1+ diag(ΛFt )⊗ diag(ΛWt ))]U⊤t = Vt[(V⊤t Υ1:Kt W⊤t Ut)⊘ (γ1+ diag(ΛFt )⊗ diag(ΛWt ))]U⊤t (17). Transformation matrix derivation.
Π1:Kt = (F1:Kt )⊤F1:Kt = VtΛFtV⊤t , WtW⊤t = UtΛWt U⊤t (18). Spectral decompositions.
Φkt+1 = Φkt + FktΩt+1 (19). Next-layer local feature update.
Lemma 1: W∗ = argmin_W ∥Y −ΦW∥2F + λ∥W∥2F yields W∗ = (Φ⊤Φ+ λI)−1Φ⊤Y.
Lemma 2: Ω∗ = argmin_Ω ∥R− FΩW∥2F + γ∥Ω∥2F yields Ω∗ = V[(V⊤F⊤RW⊤U)⊘ (γ1+ diag(ΛF)⊗ diag(ΛW))]U⊤.
Theorem 1 (Invariance to Data Heterogeneity): Given fixed seeds for A and {Bt}T−1t=0, {Wt}Tt=0 and {Ωt}Tt=1 derived from DeepAFL are invariant to any data partition P with different heterogeneity, being identical to the centralized analytical solutions on D.
Theorem 2 (Capability of Representation Learning): Let H(Φ,W) = ∥Y −ΦW∥2F. When γ=0 and λ=0, the sequence {H(Φt,Wt)}Tt=0 keeps monotonically non-increasing: H(Φt,Wt) ≥ H(Φt+1,Wt+1),∀t ∈ [0, T ). As T → ∞, the sequence converges to a limit H∗ ≤ H(Φ0,W0).
Theorem 3 (Regularized Capability of Representation Learning): Define regularized empirical risk G(t) = ∥Y −ΦtWt∥2F +λ ∥Wt∥2F + ∑ti=1 γ ∥Ωi∥2F. The sequence {G(t)}Tt=0 remains monotonically non-increasing: G(t) ≥ G(t+ 1),∀t ∈ [0, T ). As T → ∞, it converges to a limit G∗ ≤ G(0).

**8. Algorithm Pseudocode**

Algorithm 1 The Training Procedure of our Proposed DeepAFL
Input: The clients’ local datasets {D1,D2, · · · ,DK}.
Output: The transformation matrices {Ωt}Tt=1 and the analytic classifiers {Wt}Tt=0.
1: for each layer t ∈ {0, 1, 2, · · · , T} do
2: // (a) Local Computation for Analytic Classifier Construction (Client side)
3: for each client k ∈ {1, 2, · · · ,K} do
4: if constructing the first layer, i.e., t = 0 then
5: Extract and construct its local zero-layer feature matrix Φk0 via (10)
6: Compute its local Feature Auto-Correlation Matrix Gkt using Φkt via (11);
7: Compute its local Label Cross-Correlation Matrix Hkt using Φkt and Yk via (11);
8: Transmit {Gkt ,Hkt} to the server;
9: // (b) Information Aggregation for Analytic Classifier Construction (Server side)
10: Aggregate {Gkt}Kk=1 and {Hkt}Kk=1 to obtain G1:Kt and H1:Kt via (12);
11: Derive the global classifier Wt using the obtained G1:Kt and H1:Kt via (13);
12: Transmit the global classifier Wt to all clients;
13: // (c) Local Computation for Residual Block Construction (Client side)
14: for each client k ∈ {1, 2, · · · ,K} do
15: if layer requirement not satisfied, i.e., t < T then
16: Compute its local hidden random feature Fkt using Φkt via (14);
17: Compute its local residual matrix Rkt using Φkt and Wt via (14);
18: Compute its local Hidden Auto-Correlation Matrix Πkt using Fkt via (15);
19: Compute its local Residual Cross-Correlation Matrix Υkt using Fkt and Rkt via (15);
20: Transmit {Πkt ,Υkt} to the server;
21: // (d) Information Aggregation for Residual Block Construction (Server side)
22: if layer requirement not satisfied, i.e., t < T then
23: Aggregate {Πkt}Kk=1 and {Υkt}Kk=1 to obtain Π1:Kt and Υ1:Kt via (16);
24: Derive the transformation matrix Ωt+1 using Π1:Kt and Υ1:Kt via (17) and (18);
25: Transmit the transformation matrix Ωt+1 to all clients;
26: // (e) Feature Updating for Next Layer Construction (Client side)
27: if layer requirement not satisfied, i.e., t < T then
28: for each client k ∈ {1, 2, · · · ,K} do
29: Update its local feature matrix Φkt to obtain Φkt+1 via (19);
30: Return: The transformation matrices {Ωt}Tt=1 and the analytic classifiers {Wt}Tt=0.

Algorithm 2 The Inference Procedure of our Proposed DeepAFL
Input: Local sample xi, transformation matrices {Ωt}Tt=1, and analytic classifiers {Wt}Tt=0.
Output: Predicted label ŷi.
1: Extract and construct its local zero-layer feature Φi0 using xi via (10)
2: for each layer t ∈ {1, 2, · · · , T} do
3: Compute its local hidden random feature Fit via (14);
4: Update its local feature Φit−1 to obtain Φit using Ωt and Fit via (19);
5: Calculate the predicted score vector ŷi = ΦiTWT ;
6: Calculate the predicted label ŷi = argmax(ŷi);
7: Return: Predicted label ŷi.