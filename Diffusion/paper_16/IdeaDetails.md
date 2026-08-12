1. Research Background and Existing Pain Points
In recent years, denoising-based generative models have achieved remarkable progress in modeling complex data distributions. Such models are typically composed of stacking transformer blocks. Denoising-based generative models have been significantly advanced by representation-alignment (REPA) loss, which leverages pre-trained visual encoders to guide intermediate network features. REPA points out that aligning the intermediate representations of transformer blocks with the features extracted by high-performance visual encoders (e.g., CLIP, DINOv2, etc.) can significantly enhance the performance of generative models. Since the proposal of REPA, most methods in class-to-image generation tasks have been built upon this approach.
However, applying REPA to specific application scenarios may face the following critical challenges from the authors' perspective:
1. Out of distribution: If there is a significant discrepancy between the data distribution modeled by the generative model and the pre-training distribution of the large visual encoder, the features extracted by the visual encoder may not only fail to facilitate the training of the generative model but could also potentially ”mislead” it, resulting in performance degradation.
2. Additional computational costs: Both pre-training and fine-tuning large visual encoders for specific application scenarios incur additional computational costs. For instance, pre-training DINOv2 requires 1.1 billion model parameters, 1,500 training epochs, and 142 million images—far exceeding the computational resources needed to train DiT or SiT. Moreover, if the data distribution in a specific domain differs from the pre-training distribution, further fine-tuning of the visual encoder is necessary, which further increases the computational costs.
Recently, several works have incorporated unsupervised learning into generative model training to improve performance, which can be categorized into two types: introducing masked image modeling into the denoising process to enhance the contextual reasoning ability of generative models, such as MaskDiT and SD-DiT; and utilizing intermediate representations of generative models for contrastive learning (typically treating them as negative pairs) to improve training efficiency, such as Contrastive Flow Matching and Dispersive Loss. However, neither of these unsupervised approaches can provide accurate representation guidance for each image in the way REPA does, making it difficult for their performance to match that of REPA. Therefore, the existing pain points lie in the reliance on external visual encoders causing distribution mismatches and high computational costs, while current unsupervised alternatives fail to provide accurate per-image representation guidance.

2. Core Research Motivation and Scientific Questions
Xie et al. point out in REPA: “Limiting regularization to the first few layers further enhances generation performance. We hypothesize that this enables the remaining layers to concentrate on capturing high-frequency details, building on a strong representation.” Similarly, Wang et al. note in Decoupled Diffusion Transformer (DDT): “Current diffusion transformers are fundamentally constrained by their low-frequency semantic encoding capacity.” Therefore, the authors posit that the primary contribution of REPA lies in providing accurate and invariant representations derived from pure images to the first few transformer blocks when they extract semantic features from noisy images. As illustrated in the paper, REPA acts like a “data annotator” during training, supplying “labels” (i.e., effective representations) obtained from “ground truth”(i.e., pure images) for noisy images, which is similar to supervised learning. However, as discussed above, this “supervised learning” approach in REPA faces two challenges compared to unsupervised learning: “costliness of labeling” and “inaccurate labeling” issues.
The core research motivation is to utilize unsupervised learning to provide effective representation guidance for generative model training, much like REPA does but without the assumption of consistent data distribution and expensive additional computational costs. The scientific question is whether internal alignment of noisy images can also provide effective representation guidance for training of diffusion transformer without external supervision. The authors hypothesize that the features output by the visual encoder provide consistent and accurate conditioning for different noisy latents derived from the same pure image during training. The fact that different condition features of the same image converge toward the representation extracted by the visual encoder during training resembles clustering in unsupervised learning. This inspires the core scientific question: can we sample multiple condition features in a single training step and align them towards the cluster center—which corresponds to the representation extracted by the visual encoder in REPA—to achieve effective self-alignment without external supervision?

3. Overall Core Idea and Design Philosophy
Inspired by the observation that REPA primarily aids early layers in capturing robust semantics, the paper proposes an unsupervised alternative that avoids external visual encoder and the assumption of consistent data distribution. The overall core idea is to introduce DUal-Path condition Alignment (DUPA), a novel self-alignment framework, which independently noises an image multiple times and processes these noisy latents through decoupled diffusion transformer, then aligns the derived conditions—low-frequency semantic features extracted from each path.
The design philosophy is rooted in the analogy of clustering in unsupervised learning. Since REPA acts as a "data annotator" supplying "labels" from pure images, but faces "costliness" and "inaccuracy", DUPA replaces this external supervision with internal alignment. An image is independently noised multiple times during training, and uses Decoupled Diffusion Transformer to predict different denoising paths. In this way, the condition encoder can extract different conditions, which are low-frequency semantic features from different noisy images. Since these conditions originate from the same pure image, they should be similar, much like the representations obtained by large visual encoders in REPA. The paper proposes to align these different conditions derived from independently noised versions of a single image to furnish effective representation guidance for model training. By doing so, DUPA can get effective representations through internal alignment without needing external images, external parameters, or external computations, effectively solving the "costliness" and "inaccuracy" issues while maintaining the benefits of providing accurate and invariant low-frequency semantic guidance.

4. Core Innovation Points (List all innovations in the original text completely)
1. The paper points out that REPA may face issues of out of distribution and high computational costs, and hypothesizes that internal alignment of noisy images can also provide effective representation guidance for training of diffusion transformer without external supervision.
2. The paper introduces DUPA, a simple alignment for two noisy views of a single image without external supervision, which can be easily applied to other denoising-based generative models.
3. The proposed DUPA achieves a remarkable FID of 1.46 after only 400 training epochs, surpassing all evaluated methods that do not rely on external supervision. It also significantly narrows the performance gap with REPA (FID=1.42), a model trained for 800 epochs under the guidance of external visual encoders.
4. DUPA is model-agnostic and can be readily applied to any denoised-based generative models, showcasing its excellent scalability and generalizability.
5. DUPA accelerates training by 5× and inference by 10× compared to its base model DDT without external supervision, highlighting remarkable efficiency.
6. DUPA identifies that aligning the representations of different noised images can efficiently train diffusion transformer as REPA does, achieving FID performance comparable to that of REPA with only 400 training epochs, which means ≥ 3× faster convergence than current state-of-the-art methods that do not rely on supervision from an external visual encoder.

5. Overview of the Overall Technical Solution
The overall technical solution is built upon the foundation of Flow and Diffusion-based Models integrated with the Decoupled Diffusion Transformer (DDT) architecture. DUPA utilizes a dual-path sampling strategy where an image is independently noised multiple times, generating distinct noisy latents. These latents are processed through the DDT's dedicated condition encoder to extract semantic condition features (low-frequency features) and a velocity decoder to predict the velocity field. The framework then aligns the different conditions derived from independently noised versions of the same image using a condition alignment loss (LDUPA), combined with an averaged diffusion loss over the multiple sampling paths (Lvelocity). This allows the model to capture accurate and invariant low-frequency semantic information without relying on any external visual encoder or external data. The total loss function sums the condition alignment loss and diffusion loss to construct the loss function for model training. The entire framework is model-agnostic and can be readily applied to any denoising-based generative model.

6. Detailed Module Design (If any, complete mechanisms of each layer/sub-module)
*Decoupled Diffusion Transformer (DDT) Module:*
Decoupled Diffusion Transformer (DDT) introduces a novel encoder-decoder architecture to resolve the optimization dilemma in traditional diffusion transformers between low-frequency semantic encoding and high-frequency detail decoding. Specifically, DDT uses a dedicated condition encoder to extract semantic condition features zt = Encoder(xt, t, y) and a velocity decoder to predict the velocity field vt = Decoder(xt, t, zt). This encoder-decoder architecture significantly improves training efficiency while reducing FID.

*Dual-Path Sampling Module:*
For an input image x and its class label y, we conduct multiple samplings to get different noises ϵk and timestamps tk, thereby generating distinct noisy latents xtk = αtk ·x+ σtk · ϵk, 1 ≤ k ≤ K to be denoised, where K represents the number of independent sampling times. Then we use DDT to estimate the velocity for xtk:
ztk = Encoder(xtk , tk, y)
vtk = Decoder(xtk , tk, ztk)
Considering the overall performance and computational cost trade-off, we set K = 2. Multiple independent noise sampling of a single pure image are performed for two main reasons:
1. Training efficiency: It enables the training of different noised states of an image through a single training step. As discussed in the paper, this approach is more efficient compared to applying only a single noising operation.
2. Different conditions to align: Multiple independent noise sampling can obtain different velocity conditions for decoding velocities of distinct paths with the same “end point” via DDT. By aligning these conditions, DDT can encode more accurate low-frequency semantic information. Experiments show that independently resampling of both timestamp t and noise ϵ performs the best. We believe this provides more diverse noisy images, thereby enhancing the reliability of cluster centers of extracted condition representations.

*Condition Alignment Module:*
In REPA and DDT, the features extracted from pure images by state-of-the-art visual encoders are used to align the conditional features learned by DiT blocks from noisy latents. However, large visual encoders introduce additional training data and model parameters. We posit that the features output by the visual encoder provide consistent and accurate conditioning for different noisy latents derived from the same pure image during training. Similarly, we align any two conditions of {ztk} in the manner of REPA.
To align the conditions, a projector zϕ (a trainable MLP) is used to align the data dimensions of the conditions. It is crucial to avoid setting both the weights and biases to 0 when initializing projector zϕ. Otherwise, the condition used to align with will remain 0, leading to shortcut learning. In the experiments, the paper employs Kaiming initialization for the first layer of projector zϕ to preserve variance during forward propagation, while utilizing a reduced-gain Xavier initialization for subsequent layers to prevent gradient explosion or overfitting.
Regarding the condition encoder depth, the paper investigates the impact of the number of layers in the condition encoder. Similar to the conclusion in REPA, aligning the representations output by the first few layers can help the subsequent network predict high-frequency details. In the remaining experiments, the paper performs condition alignment at the 8th layer.

*Loss Function Integration:*
The paper modifies the original diffusion model’s loss to the average of diffusion losses over K-times samplings. Then it sums the condition alignment loss and diffusion loss to construct the loss function for model training. The hyperparameter λ controls the tradeoff between condition alignment and denoising. The alignment objective uses a similarity function, specifically cosine similarity, to align the projected conditions.

7. All Mathematical Formulas and Symbol Definitions (If any, replicate exactly as in the original text)
- Stochastic interpolants process: xt = αtx∗ + σtϵ
  - x∗ ∼ p(x) is data
  - ϵ ∼ N (0, I) is Gaussian noise
  - αt decreasing in time t
  - σt increasing in time t
- Probability flow ODE: ẋt = v(xt, t)
- Equivalent reverse SDE:
  dxt = v(xt, t)dt− 1/2 wts(xt, t)dt+ √wtdw̄t
- Velocity field:
  v(x, t) = α̇tE[x∗|xt = x] + σ̇tE[ϵ|xt = x]
- Velocity training objective:
  Lvelocity(θ) = Ex∗,ϵ,t [ ∥vθ(xt, t)− α̇tx∗ − σ̇tϵ∥2 ]
- DDT encoder-decoder:
  zt = Encoder(xt, t, y)
  vt = Decoder(xt, t, zt)
- REPA loss:
  LREPA(θ, ϕ) = −Ex∗,ϵ,t [ 1/N ∑N/n=1 sim(y∗[n], zϕ(zt[n])) ]
  - y∗ denotes the output of the visual encoder
  - zt represents the conditions extracted by DDT
  - zϕ is a trainable MLP used to align the data dimensions of y∗ and zt
  - θ and ϕ are the parameters of DDT and zϕ, respectively
  - N is the patch number
  - sim(·, ·) is a pre-defined similarity function
- Dual-Path Sampling generation:
  xtk = αtk ·x+ σtk · ϵk, 1 ≤ k ≤ K
  - K represents the number of independent sampling times
- DDT estimation for dual-path:
  ztk = Encoder(xtk , tk, y)
  vtk = Decoder(xtk , tk, ztk)
- DUPA Loss:
  LDUPA(θ, ϕ) := −Ex∗,{ϵk,tk}K/k=1 [ 2/(K(K − 1)) ∑1≤i<j≤K 1/N ∑N/n=1 sim(zϕ(zti[n]), zϕ(ztj[n])) ]
- Modified Velocity Loss:
  Lvelocity(θ) := Ex∗,{ϵk,tk}K/k=1 [ ∑K/k=1 ∥vθ(xtk , tk)− α̇tkx∗ − σ̇tkϵk∥2 ]
- Total Loss:
  L := Lvelocity + λLDUPA
  - λ is a hyperparameter that controls the tradeoff between condition alignment and denoising

8. Algorithm Pseudocode (If any, paste the pseudocode from the paper exactly as it is)
Algorithm 1 Dual-Path Condition Alignment Batch Step
1: Input: DDT vθ, batch of B flow examples F = {(x1, y1), . . . , (xB , yB)}, projector zϕ, learning rate β, sampling times K = 2 and hyperparameter λ = 0.5.
2: Output: Updated model parameters θ.
3: L(θ, ϕ) = 0
4: for i in range(B) do
5:   for j in range(K) do
6:     tj ∼ U(0, 1), ϵj ∼ N (0, I), xtj = αtjxi + σtj ϵj
7:     v̂j , zj = vθ(xtj , tj , yi), vj = α̇tjxi + σ̇tj ϵj
8:     zj = zϕ(zj)
9:     L(θ, ϕ)+ = ||v̂j − vj ||2
10:    for k in range(j) do
11:      L(θ, ϕ)− = 2λ/(K(K−1)) · sim(zk, zj)
12:    end for
13:  end for
14: end for
15: θ ← θ − β/B ∇θL(θ, ϕ), ϕ← ϕ− β/B ∇ϕL(θ, ϕ)