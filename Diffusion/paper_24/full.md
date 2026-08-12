## ABSTRACT

In this paper, we present DiffuDETR, a novel approach that formulates object detection as a conditional object query generation task, conditioned on the image and a set of noisy reference points. We integrate DETR-based models with denoising diffusion training to generate object queries’ reference points from a prior gaussian distribution. We propose two variants: DiffuDETR, built on top of the Deformable DETR decoder, and DiffuDINO, based on DINO’s decoder with contrastive denoising queries (CDNs). To improve inference efficiency, we further introduce a lightweight sampling scheme that requires only multiple forward passes through the decoder. Our method demonstrates consistent improvements across multiple backbones and datasets, including COCO 2017, LVIS, and V3Det, surpassing the performance of their respective baselines, with notable gains in complex and crowded scenes. Using ResNet-50 backbone we observe a +1.0 in COCO-val, reaching 51.9 mAP on DiffuDINO compared to 50.9 mAP of the DINO. We also observe similar improvements on LVIS and V3DET datasets with +2.4 and +2.2 respectively. The project page and source code are available at https://mbadran2000.github.io/DiffuDETR.

## 1 INTRODUCTION

Object detection is a fundamental task in computer vision. It has gained much attention in recent years for its wide use in real-world applications. Object detection can be decomposed to two more primitive tasks: object localization and object classification. Traditional methods depend heavily on predefined bounding boxes (Liu et al., 2016), CPU-intensive selective search (Girshick et al., 2014), and region proposals networks (Girshick, 2015; Cai & Vasconcelos, 2018) to propose candidate locations. These methods, while effective, limit flexibility and generalizability in network training.

Initial deep learning approaches (Girshick, 2015; Cai & Vasconcelos, 2018) require a set of predefined anchor boxes and heuristics to generate proposals. This method involves a one-to-many label assignment strategy, wherein each ground-truth bounding box is matched with multiple points in the detector’s predictions. Despite their strong performance, these detectors

![](images/7bbc20cfad2e59fd8d7b966f86bd445e2a0d4824f783c0b39da677277edf459d.jpg)  
Figure 1: Performance of DiffuDINO against other DETR and CNN-based models using ResNet-50 (He et al., 2016) backbone.

depend heavily on several manually designed components, such as the predefined anchor boxes or non-maximum suppression for postprocessing.

Later, DEtection TRansformer (DETR) (Carion et al., 2020) proposed an end-to-end training objective by solving the problem as a bipartite set matching between a set of predictions and ground truths. These architectures rely on object queries, where each query matches exactly one object what is also known as one-to-one matching. This approach simplified the training objective and achieved SOTA results in both object detection. However, it suffers from query initialization problem.

Denoising Diffusion Probabilistic Models (DDPM) (Ho et al., 2020; Song et al., 2020) were introduced as a probabilistic framework for image generation, achieving state-of-the-art (SOTA) performance in this domain. The core idea of DDPM is to model the process of gradually adding noise to a clean image, eventually reaching pure noise. The model learns to reverse process: starting from noisy data, it attempts to recover the original clean image at each timestep. To generate new images, DDPM samples random noise from a learned distribution and sequentially denoises it through the reverse process, ultimately producing a high-quality image. Over time, this method has been successfully extended to a wide range of computer vision tasks, demonstrating its versatility and effectiveness.

Building on previous work, we propose DiffuDETR, a framework built upon DeformableDETR and DiffuDINO, built on DINO, which shares DETR’s backbone that extracts multi-scale image features, an encoder that employs multi-scale deformable attention within its transformer layers, a transformer decoder that applies cross-attention between initial object queries and the encoder’s image features, and an MLP head that decodes each object query into a class label and bounding-box coordinates. We propose a new query initialization technique that aligns with the objective of denoising diffusion models to sample from the normally distributed reference points. This method avoids the inconveniences of query reference points initialization in DETR variants. Figure 1 shows the convergence of our proposed DiffuDINO against other DETR-based models, while it required more epochs due to the slow nature of training diffusion models. The performance surpasses DINO after 50 epochs of training on COCO dataset, which tends to deteriorate after 36 epochs.

We summarize our contributions in the following:

1. We represent object detection with detection transformers as a diffusion denoising process by denoising queries’ reference points.

2. We introduce two models, DiffuDETR and DiffuDINO, built upon Deformable DETR and DINO, respectively.

3. We conduct extensive experiments with our models on multiple benchmark datasets and conduct extensive ablation studies to validate their effectiveness.

## 2 RELATED WORK

## 2.1 DIFFUSION MODELS

Denoising diffusion models have emerged as powerful generative frameworks in computer vision, demonstrating exceptional performance in tasks such as image generation (Austin et al., 2021; Avrahami et al., 2022), super-resolution (Gao et al., 2023), inpainting (Lugmayr et al., 2022), and editing. These models progressively learn to reverse a diffusion process that adds noise to data, enabling them to synthesize high-quality and diverse samples.

Beyond generative tasks, diffusion models have also proven useful in discriminative settings, including image segmentation (Amit et al., 2021; Baranchuk et al., 2021; Brempong et al., 2022), classification (Chen et al., 2023a), and anomaly detection (Wolleb et al., 2022). Their ability to learn strong latent representations highlights their potential in broader representation learning contexts, making them an increasingly popular choice across multiple vision domains (Croitoru et al., 2023).

While diffusion models have seen significant success in generative and discriminative tasks, their application to object detection remains relatively underexplored. Only a few works (Chen et al., 2023b) have investigated generative diffusion models for detection, and the progress in this area noticeably lags behind that in segmentation. This discrepancy arises because segmentation tasks are naturally formulated in an image-to-image fashion, which aligns closely with the denoising and generative process of diffusion models. In contrast, object detection is inherently a set prediction problem, requiring the model to generate a discrete set of object candidates and assign them to corresponding ground truth objects (Ren et al., 2015). This difference introduces unique challenges for diffusion-based approaches, as the generation of unordered object queries and accurate localization is conceptually less straightforward than reconstructing pixel-wise maps.

![](images/8bc716048c2a6fc87d9373f6080c7656dccccc491a0993bcd656f39d1ad32bcb.jpg)  
Figure 2: The DiffuDETR model framework involves a transformer encoder that encodes multiscale visual features from the backbone. Ground-truth object reference points are introduced with Gaussian noise, where the noise level is controlled by a time step. These noisy reference points, along with learnable content queries, are processed by the transformer decoder, which denoises them based on the encoded features to produce precise object localizations during inference.

## 2.2 DETECTION TRANSFORMER

DEtection TRansformer (DETR) (Carion et al., 2020) revolutionized object detection by reformulating it as a direct set prediction problem, eliminating the need for hand-crafted components such as anchor boxes, region proposals, or non-maximum suppression. At its core, DETR employs an encoder-decoder transformer architecture: the encoder processes image features extracted by a CNN backbone, while the decoder operates on a fixed set of learnable object queries. Each query acts as a placeholder for a potential object and interacts with encoded features through cross-attention. Training is performed end-to-end using a bipartite matching loss based on the Hungarian algorithm, which enforces one-to-one alignment between predictions and ground-truth objects. This design enables DETR to predict all objects in a single forward pass, making it both conceptually simple and architecture-agnostic.

Despite its conceptual clarity, DETR faces practical challenges. Since its queries are initialized as zero embedding vectors without explicit spatial priors, the model must learn query-to-object alignment entirely from scratch, leading to slow convergence and unstable training. These limitations have motivated two complementary directions of research: improving query initialization and designing more effective training objectives.

The first direction focuses on enhancing query representation and initialization. DAB-DETR (Dynamic Anchor Box DETR) (Liu et al., 2022) reformulates decoder queries explicitly as anchor box coordinates—center, width, and height—which are dynamically updated across decoder layers. By incorporating positional priors directly into the queries, DAB-DETR improves alignment with image features and refines localization progressively, resulting in faster convergence and higher accuracy. In parallel, Deformable DETR (Zhu et al., 2020) introduces Deformable Attention, which restricts attention to a sparse set of spatially relevant sampling points, allowing the model to focus on key object regions. Its two-stage variant further strengthens query content by generating region proposals in the encoder (topK proposals) and refining them in the decoder, making queries image-dependent. Together, these works demonstrate that better-designed queries with spatial priors and dynamic updates can significantly enhance convergence and detection performance.

The second line of research focuses on auxiliary training tasks that enhance optimization and stability. DN-DETR (Li et al., 2022) introduces a denoising auxiliary task, where noisy ground-truth boxes and labels are injected during training, and the model learns to reconstruct the correct targets. This additional supervision alleviates instability in bipartite matching and accelerates convergence. Building upon this idea, DINO (Zhang et al., 2022) incorporates a contrastive denoising (CDN) auxiliary task combined with mixed query selection. Reference points are sampled from high-confidence encoder outputs, while their content queries are initialized with learnable class embeddings, aligning better with the denoising objective. Furthermore, DINO introduces hard negatives to strengthen the auxiliary contrastive loss, yielding improved robustness and overall detection performance.

Building on these advances, we introduce DiffuDETR, a diffusion-based object detector that leverages a generative denoising process to produce object query anchors from noise. By progressively denoising noisy reference points, DiffuDETR generates better-initialized queries that provide strong starting points for the decoder, effectively addressing limitations in query alignment and convergence. At the same time, the denoising process serves as an improved training objective, guiding the model to recover precise object locations and class predictions. This dual advantage of enhanced query anchor initialization and a denoising-driven learning signal allows DiffuDETR to achieve greater training stability and superior detection performance compared to existing DETR variants.

Beyond these two research directions, several additional approaches have explored alternative pathways for improving DETR-based detectors. Many-to-one supervision strategies (Zhao et al., 2024; Ouyang-Zhang et al., 2022) relax the strict one-to-one matching constraint to provide richer supervisory signals. Other works introduce multi-route decoding to enhance information flow (Zhang et al., 2025) and improve optimization. Aside from architectural changes, loss-function refinements aim to better align classification and localization objectives (Cai et al., 2024). While these directions offer promising directions for advancing DETR-style models, they fall outside the scope of this paper, which focuses on diffusion-based query initialization and denoising-driven training.

## 2.3 OBJECT DETECTION AS GENERATION TASK

Recent studies have started to view object detection through the lens of generative modeling, moving beyond traditional discriminative formulations. In this perspective, detection is cast as the process of generating structured outputs (bounding boxes and class labels) conditioned on image features. Two notable directions are sequence generation and denoising diffusion.

Pix2Seq (Chen et al., 2021) is one of the first works to explore this paradigm, formulating detection as a sequence generation task. Bounding boxes and class labels are represented as discrete tokens, and an encoder–decoder architecture is trained to autoregressively generate these tokens conditioned on the image and previously generated outputs. By doing so, Pix2Seq eliminates the need for handcrafted components such as proposal generation or bounding box regression, offering a simple, generic formulation of detection.

![](images/5e5b609f6b80dcea7dd7a9e6ed0ac016d2af3f9b8765cb569e4e8fa4569be6f3.jpg)  
Figure 3: DiffuDETR’s decoder iteratively denoises noisy object reference points using multiscale encoded features, integrating self-attention, deformable attention, feed-forward networks, and time-step embeddings to refine queries across diffusion steps.

In contrast, DiffusionDet (Chen et al., 2023b) formulates detection as a denoising diffusion process. Built on the Sparse R-CNN decoder (Sun et al., 2021), DiffusionDet replaces fixed proposal boxes with noisy ones, which are progressively refined into accurate predictions.

During training, ground-truth boxes are diffused into random distributions, and the model learns to reverse this noising process. At inference, boxes sampled from a Gaussian distribution are iteratively denoised through multiple cascaded decoder stages, enabling progressive refinement and flexibility in the number of predictions.

While Pix2Seq treats detection as language modeling and DiffusionDet as denoising noisy proposal boxes, our work DiffuDETR continues along the diffusion direction but adapts it to DETR. Specifically, we formulate object detection as a denoising diffusion process that generates DETR’s object queries’ anchors directly from noise, introducing diffusion-based query generation into transformerbased detection.

## 3 METHOD

## 3.1 PRELIMINARIES

## 3.1.1 DETR

The DETR model Carion et al. (2020) uses an image feature extracted as a backbone for feature extraction, followed by a Transformer encoder that processes the feature map into sequences. The decoder, which utilizes learned object queries, predicts bounding boxes and object classes. The Hungarian Matcher is used to match predicted boxes with ground truth by minimizing a cost matrix based on class and bounding box errors. Deformable DETR (Zhu et al., 2020) introduced deformable attention, where each query generates reference points to facilitate more efficient computations. This approach allows the queries to leverage multi-scale features, improving object localization. Additionally, Deformable DETR introduced a 2-stage initialization method, where queries are initialized with the top-k encoder proposals.

## 3.1.2 DENOISING DIFFUSION PROBABILISTIC MODELS

In Denoising Diffusion Probabilistic Models (DDPM), the forward process refers to the gradual addition of noise to an image over a series of timesteps, which transforms a clean image $x _ { 0 }$ into a noisy vector $x _ { t }$ . This process is performed in a series of steps, each adding a small amount of Gaussian noise to the image, progressively degrading it. Mathematically, this forward diffusion process is defined as:

$$
q (x _ {t} | x _ {t - 1}) = \mathcal {N} (x _ {t}; \sqrt {1 - \beta_ {t}} x _ {t - 1}, \beta_ {t} I)\tag{1}
$$

where $\beta _ { t }$ is the noise scheduler to control the mean of the added noise, and $\mathcal { N }$ is the normal distribution of the added noise.

During sampling, we can generate samples by sampling a noisy image $x _ { T } \sim \mathcal { N } ( 0 , I )$ , we update the noisy image using the following equation:

$$
x _ {t - 1} = \frac {1}{\sqrt {\alpha_ {t}}} \left(x _ {t} - \frac {\sqrt {1 - \alpha_ {t}}}{1 - \bar {\alpha} _ {t}}\right) \epsilon_ {\theta} (x _ {t}, t, y) + \sigma_ {t}. z\tag{2}
$$

where $\alpha _ { t } = 1 - \beta _ { t }$ and $\begin{array} { r } { \bar { \alpha } _ { t } = \prod _ { s = 1 } ^ { T } \alpha _ { s } } \end{array}$ and $\beta _ { t }$ is scheduled to control the mean of the noise added to the original image. $z \sim \mathcal { N } ( 0 , I )$ and $\sigma _ { t }$ are used to control the stochasticity of sampling.

## 3.2 DIFFUDETR

Our models build upon the Deformable DETR framework, retaining the encoder-decoder transformer architecture with multi-scale deformable attention. DiffuDETR is built on Deformable-DETR and DiffuDINO adds the CDNs introduced in DINO (Zhang et al., 2022). The key modification is introduced at the training scheme. We adopt a diffusion-like training where reference points are treated as low-dimensional latent diffusion variables and refined through an iterative denoising process to enhance object localization. During training, we use a diffusion schedule with 100 timesteps $T = 1 0 0$ , which is significantly fewer than the typical 1000 steps $T = 1 0 0 0$ used in standard diffusion models. This reduction is made possible by the low dimensionality of the diffusion space, which allows efficient sampling during inference.

The complete architecture is illustrated in Figure 2. The encoder is similar to that introduced in Zhu et al. (2020), which extracts multi-scale visual features from the input image using deformable attention modules, effectively capturing rich contextual information at various spatial resolutions. These encoded features serve as the foundational representation for the detection pipeline. The transformer decoder layer shown in Figure 3 takes three inputs: the encoded multi-scale features $O _ { \mathrm { e n c } } ,$ the noisy reference points $r _ { t } ,$ , and static learnable content queries. While the content queries encode semantic information about potential object classes, the noisy reference points provide spatial priors that guide the denoising process. The decoder iteratively refines these noisy queries conditioned on the encoded image features, effectively denoising and localizing objects.

$$
q _ {n} = \mathrm{FFN} (\mathrm{MSDA} (\mathrm{SA} (q _ {n - 1}) + t), r _ {t}, O _ {e n c})\tag{3}
$$

DiffuDINO  
Ground Truth  
DiffuDETR  
![](images/eba501d450b09ecfaedf9c82adeaf78db08c131f795eb71fecc96008034d8ab6.jpg)  
Figure 4: Qualitative comparison between Deformable DETR, DiffuDETR, DINO, and DiffuDINO on COCO 2017 Validation set. Only predictions with confidence scores above 50% are shown.

where $q _ { n }$ represents the $n ^ { t h }$ layer queries, t represents the timestep embedding, $O _ { e n c }$ is the output multi scale features of the encoder. SA stands for self-attention, MSDA stands for Multi-scale deformable attention that takes the noisy reference points and interpolates the sampled points on the encoded features.

We introduce the query denoising task as simply a diffusion process on the normalized object reference points $r \in \mathbb { R } ^ { N \times \mathbf { \breve { 4 } } }$ , representing N ground-truth objects in an image. Specifically, we sample initial time step $t \sim \mathcal { U } ( 0 , 1 0 0 )$ reference points $r _ { t }$ from a posterior Noise distribution given by:

$$
q (r _ {t} | r) = f (r _ {t}; r, \sigma^ {2} I),\tag{4}
$$

where $r _ { t }$ are noisy reference points at diffusion step t, and $\sigma ^ { 2 }$ denotes the variance controlling the noise intensity. This formulation encourages the model to learn the conditional distribution of object locations. $f$ stands for the forward process refers to the noise function used for noising conveniently chosen to be a normal distribution; however, more options could suit different problems (Nachmani et al., 2021).

At inference time, we generate object proposals by applying a deterministic DDIM sampler over a small number of timesteps. Concretely, for N object queries, we first sample $K = 4$ reference points per query from a standard Gaussian at the final diffusion step:

$$
r _ {T} ^ {(i, k)} \sim \mathcal {N} (0, I), \quad i = 1, \ldots , N, \quad k = 1, \ldots , 4.
$$

We then perform iterative denoising for $t = T , T - 1 , \dots , 1$ . At each step, we predict the noise residual $\hat { \epsilon } = \epsilon _ { \theta } ( r _ { t } , t )$ via the transformer decoder and update the reference points according to the DDIM update rule (Song et al., 2020):

$$
r _ {t - 1} = \sqrt {\bar {\alpha} _ {t - 1}} \frac {r _ {t} - \sqrt {1 - \bar {\alpha} _ {t}} \hat {\epsilon}}{\sqrt {\bar {\alpha} _ {t}}} + \sqrt {1 - \bar {\alpha} _ {t - 1}} \hat {\epsilon},\tag{5}
$$

Different noise schedulers have been proposed for diffusion models (Chen, 2023), with performance varying across tasks. In our setting, inference requires only S decoding evaluations, where $S \ll T$ The outputs $\{ r _ { 0 } ^ { ( i , k ) } \}$ from these steps are taken as the final bounding boxes. This efficient sampling scheme adds only a few extra computations using the decoder while the backbone and encoder are run once, resulting in only a small increase in computation (GFLOPs) compared to DINO.

Table 1: Comparison of various object detectors on the COCO 2017 validation set using ResNet-50 and ResNet-101 backbones

<table><tr><td>Model</td><td>Epochs</td><td>AP</td><td> $AP_{50}$ </td><td> $AP_{75}$ </td><td> $AP_s$ </td><td> $AP_m$ </td><td> $AP_l$ </td></tr><tr><td colspan="8">ResNet-50 (He et al., 2016)</td></tr><tr><td>DETR-DC5 (Carion et al., 2020)</td><td>500</td><td>43.3</td><td>63.1</td><td>45.9</td><td>22.5</td><td>47.3</td><td>61.1</td></tr><tr><td>DN-Deformable DETR (Li et al., 2022)</td><td>50</td><td>48.6</td><td>67.4</td><td>52.7</td><td>31.0</td><td>52.0</td><td>63.7</td></tr><tr><td>MS-DETR (Zhao et al., 2024)</td><td>24</td><td>50.9</td><td>68.4</td><td>56.1</td><td>34.7</td><td>54.3</td><td>65.1</td></tr><tr><td>Salience DETR(Hou et al., 2024)</td><td>24</td><td>51.2</td><td>68.9</td><td>55.7</td><td>33.9</td><td>55.5</td><td>65.6</td></tr><tr><td>MR-DETR (Zhang et al., 2025)</td><td>24</td><td>51.4</td><td>69.0</td><td>56.2</td><td>34.9</td><td>54.8</td><td>66.0</td></tr><tr><td>Pix2Seq (Chen et al., 2021)</td><td>300</td><td>43.2</td><td>61.0</td><td>46.1</td><td>26.6</td><td>47.0</td><td>58.6</td></tr><tr><td>DiffusionDet (Chen et al., 2023b)</td><td>-</td><td>46.8</td><td>65.3</td><td>51.8</td><td>29.6</td><td>49.3</td><td>62.2</td></tr><tr><td>Deformable DETR (Zhu et al., 2020)</td><td>50</td><td>48.2</td><td>67.0</td><td>52.2</td><td>30.7</td><td>51.4</td><td>63.0</td></tr><tr><td>Align-DETR (Cai et al., 2024)</td><td>24</td><td>51.4</td><td>69.1</td><td>55.8</td><td>35.5</td><td>54.6</td><td>65.7</td></tr><tr><td>DINO (Zhang et al., 2022)</td><td>36</td><td>50.9</td><td>69.0</td><td>55.3</td><td>34.6</td><td>54.1</td><td>64.6</td></tr><tr><td>DiffuDETR (Ours)</td><td>50</td><td>50.2</td><td>66.8</td><td>55.2</td><td>33.3</td><td>53.9</td><td>65.8</td></tr><tr><td>DiffuAlignDETR (Ours)</td><td>24</td><td>51.9</td><td>69.2</td><td>56.4</td><td>34.9</td><td>55.6</td><td>66.2</td></tr><tr><td>DiffuDINO (Ours)</td><td>50</td><td>51.9</td><td>69.4</td><td>55.7</td><td>35.8</td><td>55.7</td><td>67.1</td></tr><tr><td colspan="8">ResNet-101 (He et al., 2016)</td></tr><tr><td>DETR-DC5 (Carion et al., 2020)</td><td>50</td><td>43.5</td><td>63.8</td><td>46.4</td><td>21.9</td><td>48.0</td><td>61.8</td></tr><tr><td>DAB-DETR-DC5 (Liu et al., 2022)</td><td>50</td><td>46.6</td><td>67.0</td><td>50.2</td><td>28.1</td><td>50.5</td><td>64.1</td></tr><tr><td>DN-DETR-DC5 (Li et al., 2022)</td><td>50</td><td>47.3</td><td>67.5</td><td>50.8</td><td>28.6</td><td>51.5</td><td>65.0</td></tr><tr><td>MR-DETR (Zhang et al., 2025)</td><td>12</td><td>51.4</td><td>68.6</td><td>55.7</td><td>34.3</td><td>55.1</td><td>66.7</td></tr><tr><td>Pix2Seq (Chen et al., 2021)</td><td>300</td><td>44.5</td><td>62.8</td><td>47.5</td><td>26.0</td><td>48.2</td><td>60.3</td></tr><tr><td>DiffusionDet (Chen et al., 2023b)</td><td>-</td><td>47.5</td><td>65.7</td><td>52.0</td><td>30.8</td><td>50.4</td><td>63.1</td></tr><tr><td>DINO (Zhang et al., 2022)</td><td>12</td><td>50.0</td><td>67.7</td><td>54.4</td><td>32.2</td><td>53.4</td><td>64.3</td></tr><tr><td>Align-DETR (Cai et al., 2024)</td><td>12</td><td>51.2</td><td>68.8</td><td>55.7</td><td>32.9</td><td>55.1</td><td>66.6</td></tr><tr><td>DiffuDINO (Ours)</td><td>12</td><td>51.2</td><td>68.6</td><td>55.8</td><td>33.2</td><td>55.6</td><td>67.2</td></tr><tr><td>DiffuAlignDETR (Ours)</td><td>12</td><td>51.7</td><td>69.3</td><td>56.1</td><td>34.0</td><td>55.6</td><td>67.0</td></tr></table>

## 4 EXPERIMENTS

## 4.1 SETUP

We evaluated DiffuDETR on multiple benchmark datasets to assess its performance across varying levels of object density, diversity, and scale.

COCO 2017. The COCO 2017 dataset (Lin et al., 2014) comprises 80 object categories, with 118,287 training images and 5,000 validation images. It serves as a standard benchmark for object detection, segmentation, and captioning tasks, featuring a diverse range of everyday scenes.

LVIS. The LVIS dataset (Gupta et al., 2019) includes 1,203 object categories, with 100,170 training images and 19,809 validation images. It is designed to address long-tail object distributions, featuring a large vocabulary of object categories with varying frequencies.

V3DET. The V3DET dataset (Wang et al., 2023) encompasses 13,204 object categories, with 183,354 training images and 29,821 validation images. It is a large-scale, richly annotated dataset featuring detection bounding box annotations for a vast number of object classes.

For evaluation, we adopt the standard metrics defined by each benchmark. On COCO, we report mean Average Precision (AP) averaged across IoU thresholds from 0.5 to 0.95, together with AP at IoU thresholds of 0.5 (AP ) and 0.75 (AP ), as well as scale-specific AP for small, medium, and large objects. On LVIS, we follow its official evaluation protocols, which extend the COCO-style AP by also reporting performance across categories with different frequencies: rare (AP<sub>r</sub>), common $( \mathrm { A P _ { c } ) }$ , and frequent (AP<sub>f</sub>). These standardized measures provide a comprehensive view of detector performance across imbalanced category distributions.

## 4.2 RESULTS

Table 1 compares our proposed models with existing detectors on the COCO 2017 validation set. Using ResNet-50, DiffuDETR improves upon its baseline Deformable DETR by achieving 50.2

Table 2: Comparison of DINO and DiffuDINO on the LVIS validation set using ResNet-50 and ResNet-101 backbones. All models are trained for 12 epochs.

<table><tr><td>Model</td><td>AP</td><td> $AP_{50}$ </td><td> $AP_{75}$ </td><td> $AP_s$ </td><td> $AP_m$ </td><td> $AP_l$ </td><td> $AP_r$ </td><td> $AP_c$ </td><td> $AP_f$ </td></tr><tr><td colspan="10">ResNet-50 (He et al., 2016)</td></tr><tr><td>DINO (Zhang et al., 2022)</td><td>26.5</td><td>35.9</td><td>27.8</td><td>20.0</td><td>35.2</td><td>40.9</td><td>9.2</td><td>24.6</td><td>36.2</td></tr><tr><td>DiffuDINO (Ours)</td><td>28.9</td><td>38.5</td><td>30.8</td><td>20.7</td><td>37.5</td><td>46.4</td><td>13.7</td><td>27.6</td><td>36.9</td></tr><tr><td colspan="10">ResNet-101 (He et al., 2016)</td></tr><tr><td>DINO (Zhang et al., 2022)</td><td>30.9</td><td>40.4</td><td>32.8</td><td>23.2</td><td>40.5</td><td>46.3</td><td>13.9</td><td>29.7</td><td>39.7</td></tr><tr><td>DiffuDINO (Ours)</td><td>32.5</td><td>42.4</td><td>34.8</td><td>23.5</td><td>43.4</td><td>49.7</td><td>13.5</td><td>32.0</td><td>41.5</td></tr></table>

Table 3: Comparison of DINO and DiffuDINO on the V3DET validation set using ResNet-50 and Swin-B backbones. All models are trained for 24 epochs.  
Table 4: DiffuDINO performance with different diffusion noise distributions on the COCO 2017 validation set.

<table><tr><td>Model</td><td>AP</td><td> $AP_{50}$ </td><td> $AP_{75}$ </td></tr><tr><td colspan="4">ResNet-50 (He et al., 2016)</td></tr><tr><td>DINO (Zhang et al., 2022)</td><td>33.5</td><td>37.7</td><td>35.0</td></tr><tr><td>DiffuDINO (Ours)</td><td>35.7</td><td>41.4</td><td>37.7</td></tr><tr><td colspan="4">Swin-B (Liu et al., 2021)</td></tr><tr><td>DINO (Zhang et al., 2022)</td><td>42.0</td><td>46.8</td><td>43.9</td></tr><tr><td>DiffuDINO (Ours)</td><td>50.3</td><td>56.6</td><td>52.9</td></tr></table>

<table><tr><td>Diffusion Noise</td><td>AP</td><td> $AP_{50}$ </td><td> $AP_{75}$ </td></tr><tr><td>Beta</td><td>49.5</td><td>66.7</td><td>53.8</td></tr><tr><td>Sigmoid Gaussian</td><td>50.4</td><td>68.0</td><td>54.7</td></tr><tr><td>Gaussian</td><td>51.9</td><td>69.5</td><td>56.3</td></tr></table>

AP compared to 48.2, while also outperforming DN-Deformable DETR (48.6 AP), showing that our diffusion-based denoising task provides a stronger training signal than DN-DETR denoising strategy. Similarly, DiffuDINO surpasses its baseline DINO, reaching 51.9 AP versus 50.9, with consistent improvements across $\mathrm { { \bf A P } _ { 5 0 } , \mathrm { { \bf A P } _ { 7 5 } , } }$ and scale-specific metrics. When compared with other generation-inspired approaches, both DiffuDETR and DiffuDINO significantly outperform DiffusionDet (46.8 AP) and Pix2Seq (43.2 AP), highlighting the advantages of integrating the diffusion denoising process with DETR-style query generation. With ResNet-101, DiffuDINO further improves over DINO (51.2 vs. 50.0 AP), demonstrating that our approach consistently strengthens DETR-based detectors across different backbones. We additionally introduce DiffuAlignDETR, built upon Align-DETR Cai et al. (2024), and observe that diffusion refinement provides a consistent boost under standard COCO training schedules. With a ResNet-50 backbone and the 2× schedule (24 epochs), Align-DETR achieves 51.4 AP, whereas DiffuAlignDETR improves this to 51.9 AP. Similarly, with ResNet-101 under the 1× schedule (12 epochs), DiffuAlignDETR reaches 51.7 AP compared to 51.2 AP for the baseline.

We show the results on LVIS validation set in Table 2. where we compare DiffuDINO with its baseline DINO under both ResNet-50 and ResNet-101 backbones, trained for 12 epochs. With ResNet-50, DiffuDINO achieves 28.9 AP, improving over DINO’s 26.5 by +2.4 AP. With ResNet-101, DiffuDINO continues to outperform DINO (32.5 vs. 30.9 AP), with improvements in medium and large objects as well as common and frequent categories.

We present results on the V3DET validation set with ResNet-50 and Swin-B backbones in Table 3, trained for 24 epochs. With ResNet-50, DiffuDINO surpasses its baseline DINO by +2.2 AP (35.7 vs. 33.5), alongside clear improvements in $\mathsf { A P _ { 5 0 } } \ ( + 3 . { \bar { 7 } } )$ and $\mathsf { A P } _ { 7 5 }$ (+2.7). The performance gap becomes more pronounced with the stronger Swin-B backbone, where DiffuDINO achieves 50.3 AP, outperforming DINO’s 42.0 by a large margin of +8.3 points.

Figure 4 visually compares detection results across four models: Deformable DETR, DiffuDETR, DINO, and DiffuDINO on samples from COCO 2017 Validation. The qualitative examples highlight the consistent improvements introduced by our diffusion-based models, particularly in crowded scenes with multiple overlapping objects. DiffuDETR exhibits more accurate and complete localization of instances compared to Deformable DETR, while DiffuDINO further refines predictions beyond DINO, reducing missed detections and improving boundary alignment. These visualizations support the quantitative results and confirm that the diffusion-based denoising process yields clearer and more precise predictions, especially in challenging, densely populated regions.

## 4.3 ABLATION STUDY

## 4.3.1 DIFFUSION NOISE DISTRIBUTIONS

Table 4 reports the results using different noise distributions. Among the tested, Gaussian distribution consistently achieves the best performance across all metrics, yielding 51.9 AP. In comparison, Beta noise distribution underperforms relative to Gaussian noise, with up to 2.4 AP lower. Sigmoid Gaussian shows competitive results. Our primary intuition for using the sigmoid Gaussian distribution is to avoid clipping values from the diffused Gaussian points (ensuring that reference points are valued between (0 and 1). However, it still lags behind Gaussian noise overall. This shows that empirically Gaussian distribution still prevails as the more suitable distribution even in detection tasks.

Table 5: DiffuDINO performance with different noise schedulers on the COCO 2017 validation set.  
![](images/9ebc0cbb21c266c8ade763b0cb1d5ed43ba7ee5d5d3a9ccccc51e0aabc68fab9.jpg)  
Figure 5: Different Schedulers noise retained plot against timesteps.

<table><tr><td>Scheduler</td><td>AP</td><td> $AP_{50}$ </td><td> $AP_{75}$ </td><td> $AP_s$ </td><td> $AP_m$ </td><td> $AP_l$ </td></tr><tr><td>Cosine</td><td>51.9</td><td>69.5</td><td>56.4</td><td>35.9</td><td>55.8</td><td>67.1</td></tr><tr><td>Linear</td><td>51.6</td><td>69.1</td><td>56.2</td><td>35.6</td><td>55.8</td><td>66.4</td></tr><tr><td>Square Root</td><td>51.4</td><td>68.9</td><td>56.0</td><td>35.3</td><td>55.4</td><td>66.0</td></tr></table>

Table 6: Comparison of DINO and DiffuDINO with different Decoder Evaluation on the COCO 2017 validation set. We report AP metrics along with FLOPs (in GigaFLOPs) and activations (in millions).

<table><tr><td>Model</td><td>D.E.</td><td>AP</td><td>FLOPs (G)</td><td>Activations (M)</td></tr><tr><td>DINO</td><td>1</td><td>50.9</td><td>244.5 ± 25.5</td><td>673.1 ± 66.5</td></tr><tr><td>DiffuDINO</td><td>1</td><td>51.6</td><td>244.5 ± 25.5</td><td>673.1 ± 66.5</td></tr><tr><td>DiffuDINO</td><td>3</td><td>51.9</td><td>285.2 ± 27.1</td><td>871.0 ± 72.7</td></tr><tr><td>DiffuDINO</td><td>5</td><td>51.8</td><td>326.0 ± 28.7</td><td>1068.9 ± 79.0</td></tr><tr><td>DiffuDINO</td><td>10</td><td>51.4</td><td>427.9 ± 32.7</td><td>1563.6 ± 94.7</td></tr></table>

## 4.3.2 DIFFUSION NOISE SCHEDULERS

We compare the impact of different noise schedulers on DiffuDINO performance in Table 5. The cosine scheduler achieves the best overall results, reaching 51.9 AP. The linear scheduler performs competitively, particularly on medium-sized objects where it matches cosine at 55.8 $\operatorname { A P } _ { m } .$ , but slightly trails behind in other metrics. The square root scheduler shows the lowest performance among the three, with a drop of around 0.5 AP compared to cosine. This is expected because cosine schedulers tend to retain more of the original signal in later timesteps as shown in Figure 5. These results indicate that cosine scheduling provides a smoother distortion for reference points as time steps t < 100 while still completely distorting the reference points at $t = 1 0 0$

## 4.3.3 NUMBER OF DECODER EVALUATIONS

Table 6 shows the effect of varying decoder evaluation on DiffuDINO performance compared to the DINO baseline on the COCO 2017 validation set. With a single decoder evaluation, DiffuDINO already surpasses DINO by +1.7 AP while maintaining identical computational cost in both FLOPs $( 2 4 4 . 5 \pm 2 \bar { 5 } . 5 \ : \mathrm { G } )$ and activations $( 6 7 3 \pm 6 6 \mathrm { { M } ) }$ . Increasing the number of decoder evaluations to three yields the best overall results, reaching 51.9 AP with improvements across all object scales. However, this comes with a moderate increase in computation to 285.2±27.1 GFLOPs and $8 7 1 \pm 7 3$ M activations. Further increasing the decoder evaluations to five or ten does not yield additional performance gains, with AP slightly declining while computational costs continue to grow significantly, reaching $4 2 8 \pm 3 3$ GFLOPs and 1564 ± 95 M activations at ten steps. This demonstrates that our approach enables effective sampling, with only three decoder evaluations being sufficient to achieve the best results while adding minimal computation overhead compared to the baseline.

Table 7: DiffuDINO performance with different numbers of decoder evaluations on COCO val2017, averaged over 5 random initializations. Results are reported as mean ± standard deviation.

<table><tr><td>D.E.</td><td>AP</td><td> $AP_{50}$ </td><td> $AP_{75}$ </td><td> $AP_s$ </td><td> $AP_m$ </td><td> $AP_l$ </td></tr><tr><td>1</td><td>51.68 ± 0.02</td><td>69.28 ± 0.02</td><td>55.89 ± 0.04</td><td>35.50 ± 0.09</td><td>55.58 ± 0.04</td><td>66.96 ± 0.10</td></tr><tr><td>3</td><td>51.95 ± 0.03</td><td>69.54 ± 0.02</td><td>56.32 ± 0.05</td><td>35.92 ± 0.08</td><td>55.81 ± 0.01</td><td>67.18 ± 0.05</td></tr><tr><td>5</td><td>51.83 ± 0.01</td><td>69.24 ± 0.03</td><td>56.21 ± 0.05</td><td>35.82 ± 0.06</td><td>55.71 ± 0.06</td><td>67.02 ± 0.05</td></tr><tr><td>10</td><td>51.49 ± 0.08</td><td>68.54 ± 0.09</td><td>56.02 ± 0.08</td><td>35.56 ± 0.12</td><td>55.46 ± 0.15</td><td>66.83 ± 0.04</td></tr></table>

Table 8: Comparison of DiffuDINO performance across varying numbers of decoder evaluations on COCO Dense and Sparse scene subsets. The first row of each section reports the baseline DINO results, followed by DiffuDINO results using different numbers of decoder evaluations. Metrics are reported as mean ± standard deviation over 5 random initializations.

<table><tr><td>D.E.</td><td>AP</td><td>AP50</td><td>AP75</td><td>APS</td><td>APM</td><td>APL</td></tr><tr><td colspan="7">COCO Sparse Scenes</td></tr><tr><td>DINO</td><td>57.00</td><td>73.37</td><td>62.63</td><td>36.14</td><td>55.95</td><td>59.26</td></tr><tr><td>1</td><td> $58.48 \pm 0.05$ </td><td> $74.68 \pm 0.031$ </td><td> $63.53 \pm 0.02$ </td><td> $37.78 \pm 0.10$ </td><td> $57.66 \pm 0.08$ </td><td> $68.34 \pm 0.13$ </td></tr><tr><td>3</td><td> $58.65 \pm 0.03$ </td><td> $74.75 \pm 0.03$ </td><td> $63.85 \pm 0.04$ </td><td> $38.07 \pm 0.19$ </td><td> $57.81 \pm 0.05$ </td><td> $68.55 \pm 0.06$ </td></tr><tr><td>5</td><td> $58.50 \pm 0.04$ </td><td> $74.52 \pm 0.05$ </td><td> $63.71 \pm 0.08$ </td><td> $37.72 \pm 0.15$ </td><td> $57.57 \pm 0.14$ </td><td> $68.46 \pm 0.04$ </td></tr><tr><td>10</td><td> $58.16 \pm 0.04$ </td><td> $73.94 \pm 0.03$ </td><td> $63.41 \pm 0.07$ </td><td> $37.36 \pm 0.14$ </td><td> $57.27 \pm 0.15$ </td><td> $68.10 \pm 0.04$ </td></tr><tr><td colspan="7">COCO Dense Scenes</td></tr><tr><td>DINO</td><td>43.72</td><td>62.29</td><td>47.81</td><td>33.98</td><td>51.95</td><td>66.63</td></tr><tr><td>1</td><td> $44.53 \pm 0.07$ </td><td> $63.24 \pm 0.08$ </td><td> $47.87 \pm 0.10$ </td><td> $34.45 \pm 0.11$ </td><td> $53.15 \pm 0.16$ </td><td> $61.48 \pm 0.16$ </td></tr><tr><td>3</td><td> $44.88 \pm 0.02$ </td><td> $63.65 \pm 0.03$ </td><td> $48.39 \pm 0.09$ </td><td> $34.96 \pm 0.09$ </td><td> $53.40 \pm 0.06$ </td><td> $61.60 \pm 0.06$ </td></tr><tr><td>5</td><td> $44.85 \pm 0.05$ </td><td> $63.36 \pm 0.09$ </td><td> $48.40 \pm 0.11$ </td><td> $34.92 \pm 0.07$ </td><td> $53.46 \pm 0.07$ </td><td> $61.61 \pm 0.07$ </td></tr><tr><td>10</td><td> $44.57 \pm 0.07$ </td><td> $62.69 \pm 0.10$ </td><td> $48.23 \pm 0.11$ </td><td> $34.79 \pm 0.15$ </td><td> $53.35 \pm 0.10$ </td><td> $61.31 \pm 0.09$ </td></tr></table>

## 4.4 SENSITIVITY TO INITIALIZATION NOISE

To assess the robustness of DiffuDINO with respect to initial noise, we evaluate the model across five independent runs with different random seeds and report the mean and standard deviation of AP and related metrics. As shown in Tables 7, DiffuDINO exhibits consistently low seed-to-seed variance across all decoder evaluation settings (1, 3, 5, and 10 steps). On COCO with a ResNet-50 backbone, the variation across seeds remains below ±0.2 AP, demonstrating that the model’s predictions are highly stable despite changes in initialization noise.

To further analyze stability under different scene complexities, we split the COCO validation set into a sparse scenes subset (images with 10 or fewer objects) and a dense scenes subset (images with more than 10 objects). DiffuDINO maintains the same level of robustness in both subsets. As shown in the Dense and Sparse results in Table 8, the standard deviation remains below ±0.2 AP across all metrics, even under large variations in object density. Moreover, DiffuDINO consistently improves over the baseline DINO in both crowded and sparse scenes, demonstrating that diffusionbased refinement not only enhances accuracy but also preserves stability across seeds.

## 5 CONCLUSION

In this work, we represented object detection with detection transformers as a diffusion denoising process by progressively denoising queries’ reference points. We introduced two models, DiffuDETR and DiffuDINO, built upon Deformable DETR and DINO, respectively. Our approach enables effective sampling at inference, where only three decoder evaluations are sufficient to achieve the best results while adding minimal computation overhead compared to the baseline. Extensive experiments on COCO, LVIS, and V3Det demonstrate that our method consistently improves over the baselines across all datasets. Furthermore, we show that Gaussian noise provides the most suitable training signal, consistently outperforming alternative noise distributions. In addition, we observe that a cosine scheduler achieves the best performance among the different noise scheduling strategies we tested. Moreover, our multi-seed analysis confirms that DiffuDINO is highly robust to initialization noise. Finally, we believe this work opens up new directions for integrating generative and autoregressive approaches into object detection, offering fresh perspectives beyond traditional discriminative formulations.