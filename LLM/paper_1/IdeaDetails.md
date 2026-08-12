# Research Idea and Implementation Plan Extraction

## 1. Research Background and Existing Pain Points
Learning rate (LR) scheduling is one of the most critical yet operationally challenging aspects of large language model (LLM) pre-training. Conventionally, Cosine decay has been employed in numerous models (e.g., GPT-3, Chinchilla, Llama) to minimize pre-training loss. Recent studies have introduced Warmup-Stable-Decay (WSD), which keeps the LR constant through most of pre-training and decays it only briefly at the end to address the inflexibility of Cosine decay in continual pre-training paradigms. However, these prevailing practices suffer from the following pain points:
1. **Suboptimal Objective Optimization**: Existing LR schedulers (Cosine, Linear, WSD) are chosen to minimize loss at the pre-training stage, effectively optimizing pre-training metrics independently. However, the primary objective in real applications is to maximize performance after post-training (specifically Supervised Fine-Tuning, SFT), not merely pre-training performance.
2. **Performance Inversion**: Recent findings reveal that better performance after pre-training does not guarantee superior performance after SFT. A stronger pre-training model does not necessarily imply better downstream adaptability. Thus, decaying LRs to optimize pre-training metrics may compromise downstream adaptability.
3. **Sharp Minima Degradation**: Decay-based schedulers lead models into sharper minima of the loss landscape. Models in sharper regions demonstrate inferior adaptability to downstream tasks compared to models in flatter regions, as they experience more fluctuation in loss values during parameter updates.
4. **Inflexibility in Multi-stage Training**: Cosine decay requires heuristic tuning of the LR from the decayed value in continual pre-training scenarios, making it inflexible. Even with WSD, the decay phase confines models to narrow loss valleys, reducing their "loss potential" for subsequent training stages.

## 2. Core Research Motivation and Scientific Questions
**Core Research Motivation**: The motivation is to maximize the performance of the final model after the complete training pipeline (including SFT), rather than optimizing intermediate pre-training metrics. Since practitioners evaluate models at multiple stages and select the best-performing one as the starting point for the subsequent stage, a paradigm shift is needed from optimizing pre-training loss to optimizing downstream adaptability.

**Scientific Questions**:
1. Is LR decay, which is chosen based on pre-training performance, still the best choice when the model will undergo supervised fine-tuning?
2. How do different LR schedulers during pre-training influence the loss landscape geometry and the downstream adaptability of LLMs?
3. Can removing the LR decay phase entirely preserve flatter minima and enhance SFT performance across diverse training regimes (standard, mid-training, over-training)?

## 3. Overall Core Idea and Design Philosophy
The core idea is to adopt **Warmup-Stable-Only (WSO)**, a learning rate scheduling strategy that removes the decay phase from WSD and maintains a constant LR after warmup until the end of training. The design philosophy is rooted in the insight that preserving flatter regions in the loss landscape during pre-training enhances a model's adaptability to post-training tasks. By avoiding LR decay, WSO prevents the model from converging into sharp minima optimized specifically for the pre-training data distribution, thereby maintaining the flexibility required for effective parameter updates during SFT.

## 4. Core Innovation Points
1. **Systematic Demonstration of WSO Superiority**: The paper provides the systematic demonstration that WSO consistently outperforms decay-based schedulers (Cosine, Linear, WSD) on downstream tasks after SFT, with comprehensive evidence across 1B and 8B parameter models and diverse evaluation benchmarks, despite WSO potentially underperforming in pre-training metrics.
2. **Efficacy in Modern Training Regimes**: The paper shows that WSO similarly benefits mid-training and over-training scenarios. It achieves superior SFT performance compared to conventional decay-based schedulers, proving that decay at any stage consistently harms SFT performance.
3. **Loss Landscape Geometry Explanation**: Through loss landscape analysis, the paper reveals that WSO preserves flatter minima than decay-based schedulers. It establishes a negative correlation between the sharpness of pre-trained models and their subsequent SFT performance, explaining why models trained with WSO achieve better performance after SFT.
4. **Formalization of the Pipeline Optimization Gap**: The paper formalizes the discrepancy between optimizing intermediate stages (pre-training) and the final objective (post-training), introducing a search problem framework that defines the selection of models based on the final pipeline performance rather than intermediate metrics.

## 5. Overview of the Overall Technical Solution
The overall technical solution involves a rigorous empirical investigation comparing WSO against decay-based schedulers across three experimental setups, followed by a loss landscape analysis:
1. **Experiment 1 (Two-Stage)**: Pre-training followed by SFT. Models (1B and 8B) are pre-trained on FineWeb-Edu using WSO, WSD, Cosine, and Linear schedulers with varying minimum LR factors ($\alpha_{pre}$). Models are then fine-tuned using the Tulu-3 SFT mixture with a comprehensive LR sweep to find the best hyperparameters.
2. **Experiment 2 (Three-Stage)**: Pre-training, Mid-training, and Post-training. Following the OLMo 2 setup, models are pre-trained on olmo-mix-1124 and mid-trained on dolmino-mix-1124. Mid-training schedulers are parameterized with $\alpha_{mid}$ to test constant LR vs. decay in the mid-training stage.
3. **Experiment 3 (Over-Training)**: 1B models are trained on 2T tokens (approx. 100x Chinchilla-optimal) with and without mid-training to test the generality of WSO in over-training regimes.
4. **Loss Landscape Analysis**: Computing the sharpness (Hessian trace) of models throughout pre-training using Hutchinson's unbiased estimator to quantify the curvature of the loss landscape and correlate it with SFT performance.

## 6. Detailed Module Design
### 6.1 Task Definition Module
This module defines the formalization of model evaluation across stages. It introduces the function $Tasks(M)$ which returns the performance of a model $M$ on a set of pre-defined tasks for a target stage $s \in \{pre, post\}$. It defines the notation $M_2[M_1]$ to denote model $M_2$ trained with initialization $M_1$.
- **Typical Pipeline**: Select the best pre-trained model based on pre-training metrics, then select the best post-trained model based on post-training metrics.
- **Proposed Search Problem**: Jointly optimize the configuration of both pre-training and post-training to maximize the final post-training performance, bypassing the potential suboptimality of greedy stage-wise selection.

### 6.2 Learning Rate Scheduler Module
This module implements the specific LR scheduling formulations.
- **WSD (Warmup-Stable-Decay)**: Increases LR linearly during warmup, keeps it constant during the stable phase, and decays it linearly during the decay phase.
- **WSO (Warmup-Stable-Only)**: A variant of WSD where the decay phase is omitted. This corresponds to setting the minimum LR factor $\alpha_{pre} = 1.0$.
- **Cosine Scheduler**: Follows a cosine curve for decay after warmup.
- **Linear Scheduler**: Decays linearly after warmup.
- **Mid-Training Extension**: Extends the pre-training schedulers by defining the mid-training LR based on the final pre-training LR, controlled by a minimum LR factor $\alpha_{mid}$.

### 6.3 Loss Landscape Analysis Module
This module quantifies the sharpness of the loss landscape to explain the adaptability differences.
- **Sharpness Metric**: Uses the trace of the Hessian matrix as the sharpness measure, providing a scalar summary of curvature across all parameter dimensions.
- **Estimation Method**: Employs Hutchinson's unbiased estimator, which requires only Hessian-vector products computed via automatic differentiation, avoiding the prohibitive cost of computing the full Hessian for billion-parameter models.
- **Correlation Analysis**: Measures the Pearson correlation between the sharpness of pre-trained models and their subsequent SFT performance to validate the hypothesis that flatter minima enhance adaptability.

## 7. All Mathematical Formulas and Symbol Definitions
### Task Definition Formulas
Let $M$ be a model, $s \in \{pre, post\}$ be the training stage, $\mathcal{M}_{pre}$ and $\mathcal{M}_{post}$ be sets of models obtained through pre-training and post-training.
$M_2[M_1]$: Model $M_2$ trained with configuration and initialization $M_1$.
$M_{rand}$: Model with randomly initialized weights.

Typical training pipeline:
$$ \hat{M}_{pre} = \text{argmax}_{M_{pre} \in \mathcal{M}_{pre}} \{Tasks_{pre}(M_{pre}[M_{rand}])\} $$
$$ \hat{M}_{post} = \text{argmax}_{M_{post} \in \mathcal{M}_{post}} \{Tasks_{post}(M_{post}[\hat{M}_{pre}[M_{rand}]])\} $$

Proposed search problem for the final model:
$$ \hat{M}_{post} = \text{argmax}_{(M_{pre}, M_{post}) \in (\mathcal{M}_{pre}, \mathcal{M}_{post})} \{Tasks_{post}(M_{post}[M_{pre}[M_{rand}]])\} $$

Pipeline with Mid-training ($M_{mid}$):
$$ \hat{M}_{post} = \text{argmax}_{(M_{pre}, M_{mid}, M_{post}) \in (\mathcal{M}_{pre}, \mathcal{M}_{mid}, \mathcal{M}_{post})} \{Tasks_{post}(M_{post}[M_{mid}[M_{pre}[M_{rand}]]])\} $$

### Learning Rate Scheduler Formulas
$\eta_{Scheduler}(t, \alpha_{pre})$: LR at step $t$, where $Scheduler$ specifies the scheduler and $\alpha_{pre}$ controls the minimum LR factor.
$\eta_{max}$: Maximum LR.
$T_{pre}$: Total number of pre-training steps.
$T_{warmup}$: Number of warmup steps.
$T_{stable}$: Step at which the decay phase begins.

**WSD Schedule**:
$$ \eta_{WSD}(t, \alpha_{pre}) = \begin{cases} \eta_{max} \cdot \frac{t}{T_{warmup}} & t \le T_{warmup} \\ \eta_{max} & T_{warmup} < t \le T_{stable} \\ \eta_{max} \cdot \left( (1 - \alpha_{pre}) \cdot \frac{T_{pre} - t}{T_{pre} - T_{stable}} + \alpha_{pre} \right) & T_{stable} < t \le T_{pre} \end{cases} $$

**WSO Schedule** (Setting $\alpha_{pre} = 1.0$ in WSD):
$$ \eta_{WSO}(t, \alpha_{pre}) = \begin{cases} \eta_{max} \cdot \frac{t}{T_{warmup}} & t \le T_{warmup} \\ \eta_{max} & T_{warmup} < t \le T_{pre} \end{cases} $$

**Cosine Schedule**:
$$ \eta_{Cosine}(t, \alpha_{pre}) = \begin{cases} \eta_{max} \cdot \frac{t}{T_{warmup}} & t \le T_{warmup} \\ \eta_{max} \cdot \left( \alpha_{pre} + \frac{1 - \alpha_{pre}}{2} \left( 1 + \cos\left( \frac{t - T_{warmup}}{T_{pre} - T_{warmup}} \cdot \pi \right) \right) \right) & t > T_{warmup} \end{cases} $$

**Linear Schedule**:
$$ \eta_{Linear}(t, \alpha_{pre}) = \begin{cases} \eta_{max} \cdot \frac{t}{T_{warmup}} & t \le T_{warmup} \\ \eta_{max} \cdot \left( (1 - \alpha_{pre}) \cdot \frac{T_{pre} - t}{T_{pre} - T_{warmup}} + \alpha_{pre} \right) & t > T_{warmup} \end{cases} $$

**Mid-training LR Schedule**:
$T_{mid}$: Total number of mid-training steps.
$$ \eta_{Scheduler}(t, \alpha_{pre}, \alpha_{mid}) = \eta_{Scheduler}(T_{pre}, \alpha_{pre}) \cdot \left( (1 - \alpha_{mid}) \cdot \frac{T_{pre} + T_{mid} - t}{T_{mid}} + \alpha_{mid} \right) $$
for $t \in [T_{pre} + 1, T_{pre} + T_{mid}]$.

### Sharpness and Loss Landscape Formulas
$L(\theta_t; D)$: Loss function evaluated on dataset $D$ with model parameters $\theta_t \in \mathbb{R}^d$.
$H_L(\theta_t) \in \mathbb{R}^{d \times d}$: Hessian matrix of the loss with respect to parameters at $\theta_t$.

**Sharpness Definition**:
$$ Sharpness(\theta_t) = Tr(H_L(\theta_t)) = \sum_{i=1}^{d} \frac{\partial^2 L(\theta_t; D)}{\partial \theta_i^2} $$

**Hutchinson's Unbiased Estimator**:
$z_i$: Random vectors sampled from a Rademacher distribution (elements are $\pm 1$).
$N$: Number of samples.
$$ Tr(H) \approx \frac{1}{N} \sum_{i=1}^{N} z_i^T H z_i $$

## 8. Algorithm Pseudocode
The paper does not provide explicit algorithm pseudocode blocks. The algorithmic logic for the core methodology and sharpness computation is described as follows based on the text:

**Algorithm: WSO Pre-training and Sharpness Evaluation**
1. **Input**: Dataset $D$, Model parameters $\theta_0$, Max LR $\eta_{max}$, Warmup steps $T_{warmup}$, Total steps $T_{pre}$.
2. **Pre-training Loop**:
   FOR $t = 1$ to $T_{pre}$:
     IF $t \le T_{warmup}$:
       $\eta_t = \eta_{max} \cdot \frac{t}{T_{warmup}}$
     ELSE:
       $\eta_t = \eta_{max}$  // WSO: No decay phase
     Update $\theta_t$ using optimizer with LR $\eta_t$
3. **Sharpness Computation Loop** (evaluated every 4,000 steps):
   FOR each evaluation step $t$:
     Sample $N$ random vectors $z_i$ from Rademacher distribution
     FOR $i = 1$ to $N$:
       Compute Hessian-vector product $H z_i$ via automatic differentiation
       Compute $z_i^T H z_i$
     Estimate $Sharpness(\theta_t) \approx \frac{1}{N} \sum_{i=1}^{N} z_i^T H z_i$
4. **Output**: Pre-trained model $\theta_{T_{pre}}$, Sharpness trajectory.