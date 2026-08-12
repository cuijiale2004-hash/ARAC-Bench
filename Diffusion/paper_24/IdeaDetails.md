### 1. Research Background and Existing Pain Points

**Research Background:**
Object detection is a fundamental task in computer vision that has gained significant attention for its wide use in real-world applications. It can be decomposed into two more primitive tasks: object localization and object classification. Traditional methods depend heavily on predefined bounding boxes, CPU-intensive selective search, and region proposal networks to propose candidate locations. Initial deep learning approaches require a set of predefined anchor boxes and heuristics to generate proposals, involving a one-to-many label assignment strategy where each ground-truth bounding box is matched with multiple points in the detector’s predictions. Despite their strong performance, these detectors depend heavily on manually designed components, such as predefined anchor boxes or non-maximum suppression (NMS) for postprocessing. Later, DEtection TRansformer (DETR) proposed an end-to-end training objective by solving the problem as a bipartite set matching between a set of predictions and ground truths. DETR architectures rely on object queries where each query matches exactly one object (one-to-one matching), simplifying the training objective and achieving state-of-the-art results. 

Denoising Diffusion Probabilistic Models (DDPM) were introduced as a probabilistic framework for image generation, achieving SOTA performance. The core idea is to model the process of gradually adding noise to a clean image, eventually reaching pure noise, and the model learns to reverse this process. This method has been successfully extended to a wide range of computer vision tasks, including image segmentation, classification, and anomaly detection. Beyond generative tasks, diffusion models have proven useful in discriminative settings, highlighting their potential in broader representation learning contexts. Recent studies have also started to view object detection through the lens of generative modeling, such as sequence generation (Pix2Seq) and denoising diffusion (DiffusionDet).

**Existing Pain Points:**
1. **Query Initialization Problem in DETR:** Although DETR simplified the training objective, it suffers from a query initialization problem. Since its queries are initialized as zero embedding vectors without explicit spatial priors, the model must learn query-to-object alignment entirely from scratch, leading to slow convergence and unstable training.
2. **Underexploration of Diffusion in Object Detection:** While diffusion models have seen significant success in generative and discriminative tasks, their application to object detection remains relatively underexplored. The progress in this area noticeably lags behind that in segmentation because segmentation tasks are naturally formulated in an image-to-image fashion, which aligns closely with the denoising and generative process of diffusion models. In contrast, object detection is inherently a set prediction problem, requiring the model to generate a discrete set of object candidates and assign them to corresponding ground truth objects. This difference introduces unique challenges for diffusion-based approaches, as the generation of unordered object queries and accurate localization is conceptually less straightforward than reconstructing pixel-wise maps.
3. **Limitations of Existing DETR Improvements:** While research has focused on improving query initialization (e.g., DAB-DETR, Deformable DETR) and designing more effective training objectives (e.g., DN-DETR, DINO), there is still room for improvement in providing strong starting points for the decoder and enhancing the training signal for precise object localization.

### 2. Core Research Motivation and Scientific Questions

**Core Research Motivation:**
Building on previous work that demonstrates better-designed queries with spatial priors and dynamic updates can significantly enhance convergence and detection performance, there is a strong motivation to integrate the powerful generative capabilities of denoising diffusion models with DETR-based detectors. By progressively denoising noisy reference points, a diffusion-based object detector can generate better-initialized queries that provide strong starting points for the decoder, effectively addressing limitations in query alignment and convergence. Simultaneously, the denoising process can serve as an improved training objective, guiding the model to recover precise object locations and class predictions. This dual advantage of enhanced query anchor initialization and a denoising-driven learning signal can achieve greater training stability and superior detection performance. Furthermore, because the diffusion space of reference points is low-dimensional, it allows for the design of an efficient inference scheme that avoids the heavy computational cost typically associated with diffusion models.

**Scientific Questions:**
1. How can object detection with detection transformers be represented and formulated as a conditional object query generation task using a denoising diffusion process?
2. How can DETR-based models be integrated with denoising diffusion training to generate object queries’ reference points from a prior gaussian distribution?
3. How can the inference efficiency of such a diffusion-based detection model be improved to ensure that the generative process adds only minimal computation overhead compared to the baseline?

### 3. Overall Core Idea and Design Philosophy

**Overall Core Idea:**
The core idea is to formulate object detection as a conditional object query generation task, conditioned on the image and a set of noisy reference points. The approach integrates DETR-based models with denoising diffusion training to generate object queries’ reference points from a prior gaussian distribution. Specifically, reference points are treated as low-dimensional latent diffusion variables and refined through an iterative denoising process to enhance object localization.

**Design Philosophy:**
The design philosophy revolves around adapting the denoising diffusion process to the DETR framework. During training, a diffusion-like training scheme is adopted where ground-truth object reference points are introduced with Gaussian noise, the level of which is controlled by a time step. The transformer decoder then denoises these noisy reference points based on the encoded image features. During inference, a lightweight sampling scheme is introduced that requires only multiple forward passes through the decoder. By using a deterministic DDIM sampler over a small number of timesteps, the model can efficiently generate object proposals from sampled noise. This ensures that the generative denoising process provides both a strong spatial prior for query initialization and a robust training signal, while maintaining computational efficiency.

### 4. Core Innovation Points

1. **Diffusion Denoising Process for Reference Points:** Object detection with detection transformers is represented as a diffusion denoising process by denoising queries’ reference points. This introduces diffusion-based query generation into transformer-based detection.
2. **Two Novel Model Variants:** Two models are introduced: DiffuDETR, built on top of the Deformable DETR decoder, and DiffuDINO, based on DINO’s decoder with contrastive denoising queries (CDNs).
3. **Lightweight Inference Sampling Scheme:** To improve inference efficiency, a lightweight sampling scheme is introduced that requires only multiple forward passes through the decoder. It is demonstrated that only three decoder evaluations are sufficient to achieve the best results while adding minimal computation overhead compared to the baseline.
4. **Superior Performance in Complex Scenes:** The method demonstrates consistent improvements across multiple backbones and datasets (COCO 2017, LVIS, V3Det), surpassing the performance of respective baselines with notable gains in complex and crowded scenes.
5. **Robustness to Initialization Noise:** The model is highly robust to initialization noise, exhibiting consistently low seed-to-seed variance across all decoder evaluation settings, maintaining stability in both sparse and dense scenes.

### 5. Overview of the Overall Technical Solution

The overall technical solution retains the encoder-decoder transformer architecture of Deformable DETR, comprising a backbone that extracts multi-scale image features, an encoder that employs multi-scale deformable attention within its transformer layers, a transformer decoder that applies cross-attention between initial object queries and the encoder’s image features, and an MLP head that decodes each object query into a class label and bounding-box coordinates. 

The key modification is introduced at the training scheme. A diffusion-like training is adopted where reference points are treated as low-dimensional latent diffusion variables and refined through an iterative denoising process. During training, a diffusion schedule with 100 timesteps (T = 100) is used. Ground-truth object reference points are introduced with Gaussian noise, where the noise level is controlled by a time step sampled uniformly. These noisy reference points, along with learnable content queries, are processed by the transformer decoder, which denoises them based on the encoded features to produce precise object localizations.

At inference time, object proposals are generated by applying a deterministic DDIM sampler over a small number of timesteps. For N object queries, K = 4 reference points per query are first sampled from a standard Gaussian at the final diffusion step. Iterative denoising is then performed for t = T, T − 1, . . . , 1. At each step, the noise residual is predicted via the transformer decoder and the reference points are updated according to the DDIM update rule. This efficient sampling scheme adds only a few extra computations using the decoder while the backbone and encoder are run once.

### 6. Detailed Module Design

**Transformer Encoder:**
The encoder is similar to that introduced in Deformable DETR, which extracts multi-scale visual features from the input image using deformable attention modules, effectively capturing rich contextual information at various spatial resolutions. These encoded features serve as the foundational representation for the detection pipeline.

**Transformer Decoder Layer:**
The transformer decoder layer takes three inputs: the encoded multi-scale features Oenc, the noisy reference points rt, and static learnable content queries. While the content queries encode semantic information about potential object classes, the noisy reference points provide spatial priors that guide the denoising process. The decoder iteratively refines these noisy queries conditioned on the encoded image features, effectively denoising and localizing objects. It integrates self-attention, multi-scale deformable attention, feed-forward networks, and time-step embeddings to refine queries across diffusion steps.

**Diffusion Training Process (Forward Process):**
The query denoising task is introduced as a diffusion process on the normalized object reference points r ∈ RN×4, representing N ground-truth objects in an image. Specifically, initial time step t ∼ U(0, 100) reference points rt are sampled from a posterior Noise distribution. A Gaussian distribution is chosen as the noise function for noising, as it empirically prevails as the most suitable distribution even in detection tasks, outperforming Beta and Sigmoid Gaussian distributions.

**Inference Process (Reverse Process):**
At inference, a deterministic DDIM sampler is applied over a small number of timesteps. Reference points are sampled from a standard Gaussian at the final diffusion step, and iterative denoising is performed. The number of decoder evaluations is ablated, and it is found that three decoder evaluations yield the best overall results, while further increasing to five or ten does not yield additional performance gains and increases computational costs. 

**Noise Scheduler Design:**
Different noise schedulers are evaluated: Cosine, Linear, and Square Root. The cosine scheduler achieves the best overall results because it tends to retain more of the original signal in later timesteps, providing a smoother distortion for reference points as time steps t < 100 while still completely distorting the reference points at t = 100. The linear scheduler performs competitively, while the square root scheduler shows the lowest performance.

**DiffuDETR Variant:**
Built upon Deformable DETR. It does not adopt the two-stage variant but instead incorporates the encoder loss to enhance representation learning. Exponential moving average (EMA) updates are applied to stabilize training and improve convergence under the diffusion denoising objective.

**DiffuDINO Variant:**
Built upon DINO's decoder with contrastive denoising queries (CDNs). It employs 300 CDN denoising queries. EMA updates are also applied to stabilize training.

### 7. All Mathematical Formulas and Symbol Definitions

**Forward diffusion process:**
q(xt|xt−1) = N (xt; √1− βtxt−1, βtI)
*Where:*
βt is the noise scheduler to control the mean of the added noise.
N is the normal distribution of the added noise.

**Sampling update equation:**
xt−1 = 1 √αt ( xt − √1− αt 1− ᾱt ) ϵθ(xt, t, y) + σt.z
*Where:*
αt = 1− βt
ᾱt = ∏T s=1 αs
βt is scheduled to control the mean of the noise added to the original image.
z ∼ N (0, I)
σt are used to control the stochasticity of sampling.

**Decoder query update rule:**
qn = FFN(MSDA(SA(qn−1) + t), rt, Oenc)
*Where:*
qn represents the nth layer queries.
t represents the timestep embedding.
Oenc is the output multi scale features of the encoder.
SA stands for self-attention.
MSDA stands for Multi-scale deformable attention that takes the noisy reference points and interpolates the sampled points on the encoded features.

**Posterior Noise distribution for reference points:**
q(rt|r) = f(rt; r, σ2I)
*Where:*
rt are noisy reference points at diffusion step t.
σ2 denotes the variance controlling the noise intensity.
r ∈ RN×4, representing N ground-truth objects in an image.
f stands for the forward process refers to the noise function used for noising conveniently chosen to be a normal distribution.

**DDIM update rule for inference:**
rt−1 = √ᾱt−1 rt − √1− ᾱt ϵ̂√ᾱt + √1− ᾱt−1 ϵ̂
*Where:*
r(i,k)T ∼ N (0, I), i = 1, . . . , N, k = 1, . . . , 4 (K = 4 reference points per query are sampled).
ϵ̂ = ϵθ(rt, t) is the predicted noise residual via the transformer decoder.

### 8. Algorithm Pseudocode

No explicit algorithm pseudocode is provided in the original paper.