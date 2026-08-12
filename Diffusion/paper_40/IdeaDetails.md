**1. Research Background and Existing Pain Points**

Speech tokenizers define the interface between continuous speech signals and discrete speech language models, directly shaping how speech is perceived, modeled, and generated. As speech language models continue to scale and unify understanding and generation, the quality and structure of their discrete speech representations become a critical bottleneck. An effective speech tokenizer is generally expected to balance three competing objectives: (1) extreme compression for scalable language modeling, (2) high-fidelity reconstruction for natural speech generation, and (3) semantic-rich representations for downstream speech understanding. However, existing speech tokenizers struggle to satisfy these objectives jointly, particularly in the low token-rate regime, where each discrete token is required to encode substantial information. 

Most prior approaches address this tension through heuristic compromises rather than principled solutions. The existing pain points are:
(1) To mitigate reconstruction degradation at low bitrates, many methods rely on residual vector quantization (RVQ) or increased frame rates, which directly inflate token budgets and undermine efficiency for language modeling.
(2) Tokenizers optimized primarily for acoustic fidelity often neglect linguistic structure, resulting in representations that are poorly suited for speech understanding.
(3) Some approaches rely on multi-stage training pipelines, where representation learning is decoupled from waveform reconstruction, requiring a separate second-stage token-to-waveform mapping and thus preventing end-to-end joint optimization.
(4) Under traditional acoustic reconstruction objectives, simply scaling training data or model size yields diminishing returns at low token rates. This limitation suggests a structural bottleneck imposed by vector quantization: when trained solely with deterministic reconstruction losses, aggressive compression forces the discrete latent space to collapse uncertainty, often prioritizing low-level signal details over linguistically meaningful structure. Such tokenizers yield suboptimal semantic representations for downstream speech understanding tasks.

**2. Core Research Motivation and Scientific Questions**

The core research motivation is to explore a speech tokenizer paradigm that simultaneously achieves extreme compression, high-quality reconstruction, and effective representations for speech language modeling. To overcome the structural bottleneck where deterministic reconstruction losses force discrete latent spaces to collapse uncertainty, a speech tokenizer requires a generative framework that can explicitly model the uncertainty induced by aggressive quantization, rather than enforcing a deterministic inverse mapping. Diffusion models offer a natural solution as they learn to reverse a stochastic corruption process. Furthermore, since tokenizers trained solely with reconstruction losses lack alignment with linguistic information, there is a need to directly impose semantic supervision on the quantized latent space.

The scientific questions addressed are: How to explicitly model the uncertainty induced by aggressive quantization in a speech tokenizer? How to ensure that discrete tokens preserve semantic-rich and linguistically meaningful structure while being optimized for high-fidelity audio reconstruction? How to achieve efficient decoding for diffusion-based tokenizers to make them practical for inference?

**3. Overall Core Idea and Design Philosophy**

The overall core idea is the Speech Diffusion Tokenizer (SiTok), a diffusion autoencoder that jointly learns semantic-rich representations through supervised learning and enables high-fidelity audio reconstruction with diffusion. The design philosophy centers on replacing traditional deterministic or adversarial reconstruction paradigms with a generative diffusion framework that models uncertainty, and explicitly regularizing the discrete latent space with semantic supervision. Instead of directly using raw waveform signals and adversarial training (which is unfavorable for scaling due to excessive sequence length and training instability), SiTok utilizes mel-spectrograms as both input and reconstruction target, leveraging a vocoder for waveform synthesis. Crucially, SiTok jointly optimizes quantization and reconstruction within a diffusion autoencoder, ensuring discrete codes are both highly compressive and explicitly aligned with the generative distribution of speech. It also introduces semantic regularization via an auxiliary CTC decoder to directly predict textual content from the quantized space.

**4. Core Innovation Points**

1. **Diffusion Autoencoder Framework**: SiTok utilizes mel-spectrograms as input and reconstruction targets and replaces adversarial training with a diffusion autoencoder, which facilitates more stable and scalable training. By learning to reverse the diffusion process, the model effectively captures the underlying data distribution, enabling robust recovery of the original signal from its quantized representation. It jointly optimizes quantization and reconstruction, preventing the suboptimal discrete codes that result from decoupled two-stage designs.
2. **Semantic Regularization via CTC Decoder**: Unlike approaches that employ representation alignment or semantic distillation to match latent representations with features from self-supervised models, SiTok directly imposes semantic supervision on the quantized latent space by introducing an auxiliary lightweight CTC decoder optimized with a CTC loss. This encourages discrete tokens to preserve semantic-rich and linguistically meaningful structure.
3. **Efficient Decoding via Shortcut Fine-tuning**: To address the computational inefficiency of iterative sampling in diffusion models, SiTok explores shortcut fine-tuning. The decoder is fine-tuned using a shortcut model objective conditioned on both time step $t$ and a desired step size $d$, allowing the model to learn a direct mapping from a noisy input to a significantly denoised output in a single forward pass, effectively reducing inference steps to 2 or 4 while maintaining high reconstruction quality.
4. **Reconstruction Refinement Strategies**: SiTok introduces two refinement strategies: (a) Decoder finetuning, where the encoder and VQ modules are frozen and only the decoder is trained further to specialize for high-fidelity synthesis; (b) Token Classifier-Free Guidance (Token CFG), where the decoder is trained to be conditionally dependent on discrete tokens by randomly dropping all input tokens with a 10% probability, allowing steering of the decoding process during inference by combining conditional and unconditional predictions.

**5. Overview of the Overall Technical Solution**

The overall technical solution of SiTok follows an autoencoder architecture with semantic regularization. Given an input mel-spectrogram $x$ (50 Hz, 128-bin), the training process is as follows:
1. **Downsampling**: The temporal resolution of $x$ is reduced by stacking every four consecutive frames, reducing the frame rate to 12.5 Hz for computational efficiency.
2. **Encoding**: The encoder $E_\theta$ maps the downsampled spectrogram $x$ to a sequence of latent features $z$.
3. **Quantization**: Each feature vector in the latent sequence $z$ is mapped to its closest entry in a discrete codebook $E = \{e_1, e_2, \ldots, e_K\}$, producing a sequence of discrete indices $q$ and quantized embeddings $z_q$.
4. **Diffusion Modeling**: The decoder $D_\phi$ is trained to reconstruct $x$ conditioned on the quantized representation $z_q$. Using a flow-matching objective, the decoder learns to predict a velocity field that transforms a noisy sample back to the original data.
5. **Semantic Regularization**: An auxiliary lightweight CTC decoder $D^{ctc}_\phi$ is trained on the quantized embeddings $z_q$ to predict the ground-truth text transcript $y$ using a CTC loss.
The total loss combines the diffusion reconstruction loss, the CTC loss, and the VQ loss. For waveform synthesis, a Vocos-based vocoder converts the reconstructed mel spectrograms back to audio waveforms at 24K Hz.

**6. Detailed Module Design**

*   **Encoder**: The encoder is composed of standard Llama-style causal Transformer blocks, incorporating RoPE positional encoding and the SiLU activation function. It is implemented with 16 causal Llama decoder layers. The default configuration sets the hidden size to 1536, the intermediate size to 4096, and the number of attention heads to 16. It maps the downsampled mel-spectrogram to continuous latent features.
*   **Vector Quantizer (VQ)**: The VQ module adopts a default configuration of 32 dimensions with a codebook of 65,536 entries. The codebook is updated using an exponential moving average (EMA). Each feature vector in the latent sequence is mapped to its closest entry in the discrete codebook.
*   **Diffusion Decoder**: The decoder is implemented by modifying the causal Llama decoder layers into a non-causal form with 16 layers. It incorporates the diffusion step embedding by replacing RMSNorm with an Adaptive RMSNorm variant. It takes the noisy mel-spectrogram $x_t$, time step $t$, and quantized embedding $z_q$ as input to predict the velocity field.
*   **CTC Decoder**: The auxiliary semantic decoder is composed of 4 causal Llama decoder layers. It takes the quantized embeddings $z_q$ as input and outputs logits for CTC loss computation against text transcripts.
*   **Light-weight Diffusion Head**: To accelerate inference, the decoder can be partitioned into a main body $D^{main}_\phi$ (first 12 layers) and a smaller head $D^{head}_\phi$ (last 4 layers). The main body is run once to produce a base representation $h_{base} = D^{main}_\phi(z_q)$, and only the lightweight head is executed iteratively across diffusion steps: $v_\phi(x_t, t, h_{base}) = D^{head}_\phi(x_t, t, h_{base})$.
*   **Token Classifier-Free Guidance (Token CFG)**: To enable CFG, the decoder is trained to be conditionally dependent on the discrete tokens by randomly dropping all input tokens with a 10% probability. During inference, the decoding process is steered by combining predictions from both conditional and unconditional passes.

**7. All Mathematical Formulas and Symbol Definitions**

*   **Encoder Mapping**:
    $z = E_\theta(x)$
*   **Vector Quantization**:
    $q = VQ(z;E)$
*   **Forward Diffusion**:
    $x_t = tx+ (1- t) \epsilon, where \epsilon \sim N (0, I) and t \sim U(0, 1)$
*   **Velocity Prediction**:
    $v_\phi(xt, t,zq) = D_\phi(xt, t,zq) \rightarrow x- \epsilon$
*   **Total Loss Function**:
    $Ltotal = Et,x,\epsilon [\|D_\phi(xt, t,zq)- (x- \epsilon)\|] + \lambda ctc CTC(D_\phi ctc(zq),y) + Lvq$
*   **Shortcut Fine-tuning Loss**:
    $LS = Ex0,x1,t,d [\|s_\phi(xt, t, 0)- (x1 - x0)\|22 + \|s_\phi(xt, t, 2d)- starget\|22 ]$
*   **Shortcut Target**:
    $starget = stopgrad( 1 2 s_\phi(xt, t, d) + 1 2 s_\phi(xt+d, t+ d, d) )$
*   **Shortcut Step Update**:
    $xt+d = xt + s_\phi(xt, t, d)d$

**8. Algorithm Pseudocode**

```python
class SiTok:
    def __init__(
        self,
        in_dim=128,
        hidden_size=1536,
        intermediate_size=4096,
        encoder_layers=16,
        decoder_layers=16,
        ctc_decoder_layers=4,
        num_heads=16,
        vq_emb_dim=16,
        downsample_factor=4,
        vocab_size=32100,
    ):
        # temporal stacking (reduce FPS to 12.5 Hz)
        self.stack_in = StackIn(downsample_factor)
        self.stack_out = StackOut(downsample_factor)
        # transformer encoder (Llama-style causal model)
        self.encoder = LlamaModel(
            hidden_size, intermediate_size,
            encoder_layers, num_heads,
            in_dim * downsample_factor)
        # vector quantizer (Binary Spherical Quantization)
        self.vq_in = Linear(hidden_size, vq_emb_dim)
        self.vq = BinarySphericalQuantizer(vq_emb_dim)
        self.vq_out = Linear(vq_emb_dim, hidden_size)
        # diffusion decoder (DiT-style transformer)
        self.decoder = DiT(
            hidden_size, intermediate_size,
            decoder_layers, num_heads,
            use_cond=True, use_diff_step=True)
        # CTC semantic decoder (Llama-style causal model)
        self.ctc_decoder = LlamaModel(
            hidden_size, intermediate_size,
            ctc_decoder_layers, num_heads,
            vocab_size)
        # ------------------------------------------------------------
        # forward: training-time outputs for loss computation
        # ------------------------------------------------------------
    def forward(self, x, x_mask):
        """SiTok forward for training losses."""
        # 1) stack + encode to continuous latents
        h = self.stack_in(x)
        h = self.encoder(h, x_mask)
        # 2) vector quantization to discrete speech tokens
        z = self.vq_in(h)
        z_q, vq_info = self.vq(z)
        cond = self.vq_out(z_q)
        # 3) forward diffusion (flow matching)
        t = sample_uniform() # t ˜ U(0, 1)
        eps = randn_like(x) # eps ˜ N(0, I)
        x_t = self.forward_diffuse(x, eps, t) # noisy mel
        x_t = self.stack_in(x_t)
        # 4) diffusion decoder predicts flow / velocity
        flow_pred = self.decoder(x_t, t, cond)
        # 5) CTC semantic logits (for semantic regularization)
        ctc_logits = self.ctc_decoder(cond)
        return {
            "x": x, # GT mel
            "noise": eps, # diffusion noise target
            "flow_pred": flow_pred, # for flow-matching loss
            "ctc_logits": ctc_logits, # for CTC loss
            "vq_loss": vq_info["commit"], # VQ commitment loss
        }
        # ------------------------------------------------------------
        # forward diffusion (flow matching target)
        # ------------------------------------------------------------
    def forward_diffuse(self, x, eps, t):
        """Apply forward diffusion to obtain a noisy sample x_t."""
        # x_t = (1 - alpha(t)) * eps + alpha(t) * x
        # In practice alpha(t) implements the flow-matching schedule.
        x_t = (1 - t) * eps + t * x
        return x_t
        # ------------------------------------------------------------
        # inference helpers
        # ------------------------------------------------------------
    def encode(self, x, mask):
        """Encode mel into quantized VQ embeddings / indices."""
        h = self.stack_in(x)
        h = self.encoder(h, mask)
        z = self.vq_in(h)
        z_q, indices = self.vq(z)
        return z_q, indices

    def decode(self, z_q, prompt=None, steps=N):
        """Reverse diffusion to generate mel from VQ embeddings."""
        cond = self.vq_out(z_q)
        mel = diffusion_reverse(self.decoder, cond, prompt, steps)
        return self.stack_out(mel)

for batch in dataloader:
    # ------------------------------------------------------------
    # 1) prepare mel features and masks
    # ------------------------------------------------------------
    x = mel_extractor(batch.speech) # [B, T, d]
    x_mask = batch.speech_mask
    text_ids = batch.text_ids # semantic supervision
    text_mask = batch.text_mask
    # ------------------------------------------------------------
    # 2) forward pass through SiTok
    # ------------------------------------------------------------
    out = sitok.forward(x, x_mask)
    x_gt = out["x"]
    noise = out["noise"]
    flow_pred = out["flow_pred"]
    ctc_logits = out["ctc_logits"]
    vq_loss = out["vq_loss"]
    # ------------------------------------------------------------
    # 3) diffusion (flow-matching) loss
    # ------------------------------------------------------------
    # target velocity v* = x - eps
    flow_gt = x_gt - noise
    diff_loss = L1(flow_pred, flow_gt)
    # ------------------------------------------------------------
    # 4) CTC semantic loss
    # ------------------------------------------------------------
    ctc_loss = CTC_Loss(ctc_logits, text_ids, text_mask)
    # ------------------------------------------------------------
    # 5) total loss
    # ------------------------------------------------------------
    total_loss = diff_loss + vq_loss + lambda_ctc * ctc_loss
    # ------------------------------------------------------------
    # 6) optimization
    # ------------------------------------------------------------
    optimizer.zero_grad()
    total_loss.backward()
    clip_gradients(sitok.parameters(), max_norm=0.5)
    optimizer.step()
```