### 1. Research Background and Existing Pain Points
Broadly, there are two fundamental goals in the field of computer vision: visual understanding, which extracts meaningful cues from scenes, and image generation, which aims to create new visual contents. Visual understanding is typically solved through visual representation learning, i.e., transforming raw pixel data into features or embeddings that can capture high-level semantics in a discriminative manner, leading to strong performance in downstream tasks such as visual recognition and semantic segmentation. On the other hand, image generation relies on generative modeling and emphasizes the learning of underlying patterns and distributions within data, thereby enabling the synthesis of new samples that faithfully resemble the original one. Since visual understanding and synthesis have long been addressed with different paradigms, most existing work excels in either synthesizing realistic outputs or interpreting input data, but seldom does both on a unified basis. This brings several pain points and drawbacks:
① Representations learned in a discriminative manner for visual perception tasks often generalize poorly to unseen patterns and overlook fine-grained details. This stems from their narrow focus on decision boundaries between classes, rather than capturing the underlying data distribution like generative models.
② Modern generative models such as GANs or diffusion models exhibit a lower-level understanding of semantics due to the reliance on low-level reconstruction loss. As a result, they tend to underperform discriminative approaches in scene understanding tasks.
③ The divergence in technological protocols for image understanding and synthesis diffuses the research endeavors, and hinders innovations and insights achieved in one paradigm to enhance the other.
④ Prior diffusion-based work that applies diffusion models for perception often compromises the generation ability for visual understanding. Joint discriminative and generative learning research mostly focuses on one-direction enhancement. Existing unified image understanding and synthesis models utilizing diffusion models often frame perception tasks as the generation of colorful maps or text embeddings, which still falls in low-level reconstruction, lacking high-level modeling on semantics.

### 2. Core Research Motivation and Scientific Questions
The core research motivation stems from rethinking the perceived incompleteness in discriminative-based representation learning and generative modeling, seeking to bridge this gap by preserving both synthesis and understanding abilities within the same model. This idea is motivated by the observations that: i) diffusion models facilitate downstream visual perception tasks; ii) high-quality discriminative representations accelerate the generative learning of diffusion models. This reveals the potential commonality of representations learned via two paradigms, and forms the basis for devising a unified visual understanding and generation framework. 
The scientific question is how to reconcile the learning processes for both visual perception tasks and image generation within diffusion models while enabling mutual benefits, establishing a positive feedback loop where the unique strengths of two learning paradigms can be leveraged to enhance each other without sacrificing one for the other.

### 3. Overall Core Idea and Design Philosophy
The overall core idea is to present GENREP, a unified image understanding and synthesis model that jointly conducts discriminative learning and generative modeling in one training session. The design philosophy respects and harnesses the unique characteristics of both paradigms. Specifically:
First, to enhance visual understanding, GENREP leverages the distributional knowledge captured by generative modeling. By assuming the diffusion-based image generation can capture object distributions, it distills this knowledge to guide the discriminative learning for visual perception tasks.
Second, to enable image generation informed by visual understanding, a semantic-driven image generation process is established, where high-level semantics learned from perception tasks are used to inform and instruct the sampling stage (i.e., reverse diffusion) of image synthesis.
Third, to reconcile the learning process for both tasks, a gradient alignment strategy is proposed to symmetrically modify the optimization directions of perception and generation losses, harmonizing the learned representations for both tasks to deliver a single and cohesive model.

### 4. Core Innovation Points
1. **Generative Visual Perception Learning via MCMC Approximation**: By leveraging Monte Carlo approximation, GENREP distills distributional knowledge embedded in diffusion models to guide the discriminative learning. It approximates the conditional distribution $p(x|y)$ through the trajectory of a single reverse diffusion process using Markov Chain Monte Carlo (MCMC), adopting burn-in and thinning techniques to mitigate uninformative initial states and temporal correlation. The class-wise posterior probabilities $p(y|x)$ are retrieved via Bayes’ theorem, serving as a supplementary guidance (KL divergence regularizer) for the discriminative learning.
2. **Semantic-Driven Generation Learning**: To enhance image generation informed by visual understanding, a semantic-aware noise adjustment strategy is proposed. It leverages high-level semantics derived from perception tasks to dynamically modulate the noise parameters (mean and variance) during the reverse diffusion process, encouraging the generated images to faithfully reflect the desired content.
3. **Gradient Alignment Strategy for Weight Merge**: To reconcile the optimization of visual perception loss and image generation loss within a single model, a gradient alignment mechanism is introduced to address potential conflicts. It decomposes gradients into parallel and orthogonal components, selectively dampening conflicting parallel components based on an adaptive retention factor, while fully preserving non-conflicting orthogonal ones to deliver a unified model.
4. **Unified Architecture with Positive Feedback Loop**: GENREP builds a feedback loop to enable mutual boosts between generative and discriminative learning. Unlike prior work that sacrifices generation for perception or focuses on one-direction enhancement, GENREP preserves both abilities and achieves top-leading performance on both image understanding and generation benchmarks.

### 5. Overview of the Overall Technical Solution
The overall technical solution is built upon latent diffusion models (LDM) or Diffusion Transformers (DiT). Given an input sample $x$ and its corresponding textual class label $y$, $x$ is first encoded into the latent space. 
For the **generative visual perception learning** branch, the conditional distribution $p(x|y)$ is approximated using the intermediate outputs of a single reverse diffusion process treated as a Markov chain. With burn-in and thinning, the trajectory estimates $p(x|y)$, which is then converted to posterior $p(y|x)$ via Bayes' theorem. This posterior is used to distill distributional knowledge into the discriminative decoder $q(y|x)$ by minimizing the KL divergence, combining with the standard discriminative loss to form the perception loss.
For the **semantic-driven generation learning** branch, the intermediate output representation containing rich semantic cues from the perception branch ($x_{sem}$) is used to modulate the reverse diffusion process. It augments the mean prediction to steer the denoising trajectory towards the desired semantic target, and modulates the variance prediction to adaptively control the influence of semantic guidance. The generation loss combines the standard LDM reconstruction loss and a representation alignment loss.
For **gradient alignment**, during joint optimization, the gradients of perception and generation losses are decomposed into parallel and orthogonal components. An adaptive retention factor based on cosine similarity is used to dampen conflicting parallel gradients, while preserving orthogonal ones. The final update uses a weighted sum of symmetrically aligned gradients. The training is staged: first optimizing with perception loss to get a semantic denoiser, then copying it for unified optimization with aligned gradients while maintaining the semantic denoiser via momentum update.

### 6. Detailed Module Design
**6.1 Generative Visual Perception Learning Module**
This module aims to transfer distribution knowledge from diffusion models to discriminative visual perception. Assuming diffusion models capture visual distributions via generative modeling, the conditional distribution for sample $x$ (i.e., $p(x|y)$) is derived with class label $y$ as the conditional input. Since the exact computation of $p(x|y)$ is intractable, it is approximated following the principle of Markov Chain Monte Carlo (MCMC). During the reverse diffusion process, a sequence of intermediate states $x_T \to x_{T-1} \to \cdots \to x_0$ naturally constitutes a non-stationary Markov chain. The transition kernel $p_\theta(x_{t-1}|x_t)$ at each step is parameterized as a Gaussian distribution. However, leveraging the chain directly introduces challenges: i) initial states correspond to nearly pure noise, degrading approximation quality; ii) sequentially drawn samples are temporally correlated, conflicting with the independent assumption of Monte Carlo methods. 
To mitigate these, two MCMC techniques are adopted:
- **Burn-in**: Discard the first $m$ uninformative steps of the chain.
- **Thinning**: Reduce correlation by selecting samples at a fixed interval of the $k$-th step (empirically $k=2$ provides a good trade-off).
Following MCMC practice, the trajectory of a single reverse diffusion process estimates $p(x|y)$ by averaging Gaussian PDF values. The posterior distribution $p(y|x)$ is then computed by substituting into Bayes' theorem, assuming a uniform prior $p(y) = 1/|Y|$. This creates a non-informative prior allowing the posterior to be shaped primarily by the learned likelihood $p(x|y)$. To inform the discriminative perception process with distributional knowledge, the KL divergence between $p(y|x)$ (computed by generative modeling) and $q(y|x)$ (obtained by applying softmax to task-specific decoder logits) is minimized. This generative distillation loss bridges the gap between generative and discriminative frameworks as a regularizer, creating a soft posterior that faithfully reflects ambiguity for similar classes, unlike standard discriminative loss that encourages overconfident predictions.

**6.2 Semantic-Driven Generation Learning Module**
This module enhances image generation by leveraging high-level semantics learned through visual perception. Assuming a well-trained denoising network optimized for visual perception, the intermediate output representation containing rich semantic cues is denoted as $x_{sem}$. During the reverse diffusion process, the noise parameters (mean $\mu_\theta$ and variance $\sigma_\theta$) are dynamically modulated according to $x_{sem}$.
- **Mean Modulation**: The prediction for $\mu_\theta$ is augmented by adding a semantic correction function $f_{sem}^\mu$ to the baseline mean value. Conceptually, $\mu_\theta$ determines the primary denoising direction, and this adjustment steers the denoising trajectory towards the desired semantic target encoded in $x_{sem}$.
- **Variance Modulation**: The variance prediction network is augmented by scaling the baseline variance with a semantic scaling factor $f_{sem}^\sigma$. The variance $\sigma_\theta$ controls the uncertainty of reverse diffusion. A positive scaling factor increases variance, encouraging broader exploration towards the semantic target when the current sample is far. Conversely, a negative value reduces variance, promoting finer refinement near the target semantics.
During inference, the explicit semantic representation $x_{sem}$ in the modulation equations can be directly replaced with the current noisy sample $x_t$, as the enhanced denoising network has already learned to capture necessary semantic cues with the knowledge preserved in model weights.

**6.3 Gradient Alignment Module**
To reconcile the optimization of visual perception loss and image generation loss within a single model, a gradient alignment mechanism addresses potential conflicts by symmetrically modifying their respective gradients according to the severity of the conflict. Each gradient is decomposed into components parallel and orthogonal to the other gradient. The parallel components capture movements in the same or opposite gradient directions of two tasks, while the orthogonal components are gradient directions that do not affect the objective of the other task. The aligned gradients for both tasks are reconstructed by fully preserving the orthogonal components and dampening the parallel components by an adaptive retention factor $\alpha$. This factor is defined according to the cosine similarity between two original gradients using a scaled and shifted power function, ensuring $\alpha = 1$ when cosine similarity is 1 (no damping needed) and $\alpha$ decreases towards 0 as cosine similarity approaches -1 (maximum damping). The final gradients used for the model update are a weighted sum of the aligned gradients.

**6.4 Training Strategy**
GENREP is first optimized with solely task-specific perception loss, yielding a denoising network $\epsilon_\theta^{sem}$ which encodes high-level semantic cues into the intermediate output $x_t$ (resulting in $x_{sem}$). Subsequently, the image generation loss steps in. A new denoising network $\epsilon_\theta^{unified}$ copied from $\epsilon_\theta^{sem}$ is optimized where at each training step: Gradients for both perception and generation losses are computed using the same input image; Gradients are aligned according to the symmetric alignment equation to update weights of attention blocks in $\epsilon_\theta^{unified}$; Parameters of $\epsilon_\theta^{sem}$ are updated in a momentum manner to maintain stable semantic features $x_{sem}$ for image generation learning.

### 7. All Mathematical Formulas and Symbol Definitions
- $x$: Input sample
- $y$: Textual class label
- $E$: Encoder of pre-trained generator
- $\mathbf{x}$: Latent representation, $\mathbf{x} = E(x)$
- $\epsilon_\theta$: Denoising network
- $c_\theta(y)$: Encoded label condition
- $\hat{\mathbf{x}}$: Extracted features, $\hat{\mathbf{x}} = \epsilon_\theta(\mathbf{x}, 0, c_\theta(y))$
- $p(x|y)$: Conditional distribution for sample $x$ given class label $y$
- $p_\theta(x_{t-1}|x_t)$: Transition kernel at step $t$
$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_\theta(x_t, t)) \quad (1)$$
- $\mathcal{N}$: Gaussian distribution
- $\mu_\theta, \sigma_\theta$: Networks parameterized by $\theta$ to predict mean $\mu_t$ and variance $\sigma_t$
- $m$: Number of uninformative steps discarded in burn-in period
- $k$: Fixed interval step for thinning
- $T$: Total number of reverse diffusion steps after burn-in and thinning
- Approximation of $p(x|y)$:
$$p(x|y) \approx \frac{1}{T}\sum_{t=1}^T \mathcal{N}(x; \mu_{t,y}, \sigma_{t,y}) \quad (2)$$
- $p(y|x)$: Posterior distribution
- $p(y)$: Prior distribution, assumed uniformly distributed as $p(y) = 1/|Y|$
$$p(y|x) = \frac{p(y)p(x|y)}{\sum_{y'\in Y} p(y')p(x|y')} \quad (3)$$
- $q(y|x)$: Discriminative probability obtained by applying softmax to output logits $z$
- $\mathcal{L}_{distil}^{gen}$: KL divergence between $p(y|x)$ and $q(y|x)$
$$\mathcal{L}_{distil}^{gen} = D_{KL}(p||q) = \sum_{y\in Y} p(y|x) \log \frac{p(y|x)}{q(y|x)} \quad (4)$$
- $\mathcal{L}_{disc}$: Conventional discriminative loss for each perception task
- $\mathcal{L}_{percept}$: Final objective for perception
$$\mathcal{L}_{percept} = \mathcal{L}_{disc} + \mathcal{L}_{distil}^{gen} \quad (5)$$
- $x_{sem}$: Intermediate output representation containing rich semantic cues
- Augmented mean prediction:
$$\mu_\theta(x_t, t, x_{sem}) = \mu_\theta^{base}(x_t, t) + f_{sem}^\mu(x_t, x_{sem}) \quad (6)$$
- $\mu_\theta^{base}(x_t, t)$: Baseline mean value predicted by underlying diffusion model
- $f_{sem}^\mu$: Semantic correction function
$$f_{sem}^\mu(x_t, x_{sem}) = W_t^\mu \cdot \text{concat}(x_t, x_{sem}) \quad (7)$$
- $W_t^\mu$: Learned weight matrix
- Augmented variance prediction:
$$\sigma_\theta(x_t, t, x_{sem}) = \sigma_\theta^{base}(x_t, t) \cdot (1 + f_{sem}^\sigma(x_{sem})) \quad (8)$$
- $\sigma_\theta^{base}(x_t, t)$: Baseline variance predicted following improved DDPM
- $f_{sem}^\sigma$: Semantic scaling factor
$$f_{sem}^\sigma(x_{sem}) = \text{MLP}(x_{sem}) \quad (9)$$
- $\mathcal{L}_{LDM}$: Standard reconstruction loss for latent diffusion models
- $\mathcal{L}_{align}^{rep}$: Representation alignment loss minimizing cosine similarity between $\mathbf{x}$ and $x_{sem}$
- $\mathcal{L}_{genera}$: Overall training objective for generation
$$\mathcal{L}_{genera} = \mathcal{L}_{LDM} + \mathcal{L}_{align}^{rep} \quad (10)$$
- Gradient decomposition for generation loss:
$$\nabla \mathcal{L}_{genera}^{\parallel} = \frac{\nabla \mathcal{L}_{percept} \cdot \nabla \mathcal{L}_{genera}}{\|\nabla \mathcal{L}_{percept}\|^2} \nabla \mathcal{L}_{percept}, \quad \nabla \mathcal{L}_{genera}^{\perp} = \nabla \mathcal{L}_{genera} - \nabla \mathcal{L}_{genera}^{\parallel} \quad (11)$$
- Gradient decomposition for perception loss:
$$\nabla \mathcal{L}_{percept}^{\parallel} = \frac{\nabla \mathcal{L}_{percept} \cdot \nabla \mathcal{L}_{genera}}{\|\nabla \mathcal{L}_{genera}\|^2} \nabla \mathcal{L}_{genera}, \quad \nabla \mathcal{L}_{percept}^{\perp} = \nabla \mathcal{L}_{percept} - \nabla \mathcal{L}_{percept}^{\parallel} \quad (12)$$
- Aligned gradient for generation:
$$\nabla \mathcal{L}_{genera}^{aligned} = \nabla \mathcal{L}_{genera}^{\perp} + \alpha \nabla \mathcal{L}_{genera}^{\parallel} \quad (13)$$
- Aligned gradient for perception:
$$\nabla \mathcal{L}_{percept}^{aligned} = \nabla \mathcal{L}_{percept}^{\perp} + \alpha \nabla \mathcal{L}_{percept}^{\parallel} \quad (14)$$
- $\alpha$: Adaptive retention factor
- $\cos \text{sim}$: Cosine similarity
$$\cos \text{sim} = \frac{\nabla \mathcal{L}_{percept} \cdot \nabla \mathcal{L}_{genera}}{\|\nabla \mathcal{L}_{percept}\| \|\nabla \mathcal{L}_{genera}\|} \quad (15)$$
- Formulation for $\alpha$:
$$\alpha = ((\cos \text{sim} + 1)/2)^k$$
(Here $k=2$ is a hyperparameter controlling the sharpness of the damping)
- $\nabla_{symmetric}^{aligned}$: Final gradients used for model update
$$\nabla_{symmetric}^{aligned} = w_p \nabla \mathcal{L}_{percept}^{aligned} + w_g \nabla \mathcal{L}_{genera}^{aligned} \quad (16)$$
(Where $w_p = 0.7$ and $w_g = 0.3$ scale task weights)

### 8. Algorithm Pseudocode
```text
Algorithm 1 Generative Visual Perception Learning via Knowledge Distillation.
1: Hyperparameters:
2: T ← total diffusion steps
3: k ← 2 {Thinning interval}
4: m ← 50 {Burn-in steps}
5: Initialize models:
6: diffusion model ← PretrainedDiffusionModel()
7: task decoder ← TaskSpecificDecoder()
8: hot params ← diffusion model.attention blocks[:]
9: Freeze all parameters except attention blocks:
10: freeze all parameters(diffusion model)
11: unfreeze parameters(hot params)
12: for each (x, ytrue) in training data do
13:  Step 1: Reverse diffusion process
14:  xT ← sample noise(x)
15:  reverse samples ← ∅
16:  for t = T, T − 1, . . . , 1 do
17:      xt ← diffusion model.reverse step(xt, t, ytrue)
18:      if t < T −m and T mod k = 0 then
19:          reverse samples.append(xt)
20:      end if
21:  end for
22:  Step 2: Estimate p(x|y)
23:  µlist ← {(s.mean) | s ∈ reverse samples}
24:  σlist ← {(s.variance) | s ∈ reverse samples}
25:  p(x|y) ← 0
26:  for µ, σ ∈ (µlist, σlist) do
27:      p(x|y) ← p(x|y) + N (µ, σ) (Add Gaussian component)
28:  end for
29:  p(x|y) ← p(x|y)/(T//k)
30:  Step 3: Compute generative posterior p(y|x)
31:  prior ← 1/num classes
32:  logitsgen ← p(x | y).log prob(x) + log(prior)
33:  p(y | x) ← softmax(logitsgen)
34:  Step 4: Compute discriminative probability q(y|x)
35:  logitsdisc ← task decoder(x)
36:  q(y | x) ← softmax(logitsdisc)
37:  Step 5: Loss computation
38:  lossdisc ← cross entropy(q(y | x), ytrue)
39:  lossgen distil ← KL divergence(p(y | x), q(y | x))
40:  total loss ← lossdisc + lossgen distil
41:  Step 6: Backpropagation
42:  optimizer.zero grad()
43:  total loss.backward()
44:  optimizer.step(hot params, task decoder)
45: end for
```