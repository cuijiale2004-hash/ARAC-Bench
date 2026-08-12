# Research Idea and Implementation Plan Extraction

## 1. Research Background and Existing Pain Points
Diffusion models have recently achieved remarkable success in generative tasks, such as text-to-image generation and video generation. While early approaches relied on U-Net architectures, recent works have shifted toward diffusion transformers (DiTs), which demonstrate superior performance, particularly when model and data scales increase. However, such gains come with significant computational costs, since DiTs require performing expensive forward passes over numerous denoising timesteps, which hinders their practical application.

To address this, feature caching approaches have emerged, offering a promising solution for reducing redundant computation in DiTs. These methods are based on the observation that the intermediate features in DiTs are highly similar across adjacent timesteps. Specifically, they perform full computations at certain timesteps, cache the output features of computationally expensive modules (e.g., attention and MLP), and exploit the cached features at subsequent timesteps to reduce redundant computations.

Existing pain points include:
1. Early caching approaches typically reuse cached features directly without adaptation. While this improves efficiency, the discrepancy between cached and fully computed features accumulates over timesteps, which in turn degrades the generation quality.
2. Recent forecasting-based methods predict features through temporal extrapolation techniques, with an assumption that features evolve smoothly over timesteps. For instance, FasterCache and GOC perform a linear extrapolation technique based on features from the two most recent full-compute steps, while TaylorSeer leverages the Taylor expansion to estimate feature changes along timesteps. However, the magnitude of changes in the output features differs significantly across timesteps, leading to substantial prediction errors. Relying exclusively on temporal extrapolation still suffers from significant prediction errors, leading to performance degradation, especially when employing large temporal intervals between full computations (i.e., lower FLOPs).

## 2. Core Research Motivation and Scientific Questions
**Core Research Motivation:** Through a detailed analysis, it is found that 1) these errors in forecasting-based caching stem from the irregular magnitude of changes in the output features, and 2) an input feature of a module is strongly correlated with the corresponding output. The magnitudes of changes in the input and output features for the same module are highly correlated, suggesting that differences in input features can serve as effective predictors of the output variations. This strong correlation implies that the relationship between input and output features can be leveraged to enhance the accuracy of feature prediction.

**Scientific Questions:** How to address the significant prediction errors caused by the irregular magnitude of changes in output features across timesteps when using temporal extrapolation? How can the relationship between input and output features be utilized to estimate the magnitude of changes in the output features effectively and efficiently? Since directly measuring output errors requires full computations, how can we estimate the prediction errors of the output features to determine when predictions become unreliable and adaptively perform full computations?

## 3. Overall Core Idea and Design Philosophy
**Overall Core Idea:** Leverage the relationship between the input and output features of modules (e.g., attention and MLP), rather than relying solely on temporal extrapolation techniques, to enhance feature prediction and determine when to perform full computations.

**Design Philosophy:** Based on the observation that the magnitude of changes in the output features is highly correlated with that of the input features, and the ratio between the magnitudes of changes in output and input features is highly consistent across timesteps, the framework uses the input feature differences to estimate the magnitude of changes in output features (Relational Feature Estimation). Additionally, since errors in the output prediction are likely to be correlated with those in the input prediction due to the abrupt feature changes, the framework uses the error of the input prediction as an efficient proxy to estimate the accumulated errors in the output prediction, performing full computations only when the accumulated error is expected to be substantial (Relational Cache Scheduling).

## 4. Core Innovation Points
1. Propose Relational Feature Estimation (RFE), a forecasting method that leverages input feature variations to more accurately estimate output features by capturing irregular dynamics of feature changes.
2. Propose Relational Cache Scheduling (RCS), a dynamic strategy that performs full computations adaptively by estimating the prediction error of the output features from that of input features.
3. Discover the strong correlation and consistency of the ratio between input and output feature magnitudes across timesteps, theoretically supported by Proposition 1, which proves the ratio is approximately invariant under assumptions of locally linear mapping and constant direction of the difference vector.
4. Extensive experiments across various DiT models (class-conditional, text-to-image, text-to-video generation tasks) demonstrate that using RFE and RCS consistently outperforms existing caching methods in terms of both the generation quality and computational efficiency.

## 5. Overview of the Overall Technical Solution
RFC (Relational Feature Caching) is a novel framework that enhances feature prediction by leveraging the relationship between input and output features through two complementary components: RFE and RCS. 

Forecasting-based methods predict features through temporal extrapolation techniques with an assumption that features evolve smoothly over timesteps. FasterCache and GOC perform linear extrapolation, while TaylorSeer leverages Taylor expansion. However, the magnitude of changes in output features differs significantly across timesteps. RFC addresses this by not solely relying on temporal extrapolation. The RFE component specifically targets the magnitude of the feature change prediction. The Taylor expansion provides the direction of the change, which is assumed consistent, but its magnitude is irregular. RFE corrects this by scaling the direction vector with the magnitude estimated from the input features.

RCS is a dynamic scheduling strategy that determines when to perform full computations based on estimated output errors. Since directly measuring output errors requires full computations, it instead employs errors in the input prediction as an efficient proxy, leveraging the relationship between the input and output features. It tracks the accumulated relative L1 error of the input prediction for the first module and performs a full computation when the accumulated error exceeds a predefined threshold. This allows RFC to perform full computations more frequently at timesteps where the accumulated errors are large, reducing the cache error and improving the generation quality.

## 6. Detailed Module Design
**Relational feature estimation (RFE):**
RFE is proposed to estimate the magnitude of changes in output features by leveraging the differences in input features. To quantify the relationship between input and output magnitudes, the ratio between the magnitudes of changes in output and input features is defined. Empirical analysis shows that this ratio is highly consistent across timesteps with low relative standard deviation (RSD) values (typically around 2%). 

Theoretically, Proposition 1 states that if the mapping from input to output features is locally linear, and the direction of the difference vector remains constant for 1 ≤ k ≤ N, where N is an interval between full computations, then the ratio is approximately invariant w.r.t. k. The proof assumes the output feature is a locally linear transformation of the input feature, and derives that the change in output features over the temporal offset k equals the linear transformation matrix multiplied by the change in input features. Taking the L2-norm gives the ratio formula, which becomes invariant with the constant direction assumption. The two assumptions are commonly observed: small feature changes allow local linear approximation, and the direction of feature changes remains largely consistent.

Given the consistency of this ratio across timesteps, it can be approximated using the ratio computed between the two most recent full computations with the interval of N. Note that obtaining an input feature is efficient, since it only requires lightweight operations (e.g., LayerNorm, scaling, and shifting). Consequently, RFE refines the predicted magnitude of feature changes by replacing the magnitude of the Taylor expansion term with the estimated output change magnitude, while normalizing the Taylor expansion term to get the direction. The final prediction is the sum of the output feature at the last full computation and the product of the estimated magnitude and the normalized direction vector.

**Relational cache scheduling (RCS):**
RCS is a dynamic scheduling strategy that determines when to perform a full computation based on the estimated prediction error. Computing the error of the output prediction directly is not feasible during sampling, since it depends on the output feature, which requires a costly computation. Instead, RCS proposes to estimate this error using the error of the input prediction as a proxy. The intuition is that prediction errors in the output tend to increase when the feature changes abruptly, and such changes are highly correlated with the variations in the corresponding input. Therefore, errors in the output prediction are likely to be correlated with those in the input prediction. 

To support this, the relative L1 error of the output and input features are defined, and their trends closely align across timesteps. Based on this, RCS aims to estimate the accumulated errors in the output prediction during the sampling process by tracking the relative L1 error of the input prediction for the first module. Computing the prediction error from only the first module, rather than all modules, is sufficient because errors in early modules affect subsequent modules, suggesting that the prediction error of the first module can be a reliable indicator of the overall prediction error. RCS then performs a full computation when the accumulated error exceeds a predefined threshold τ. By adjusting τ, RCS controls the trade-off between the generation quality and efficiency. RCS performs full computations more frequently at timesteps where the accumulated errors are large. It naturally adapts to the generative process, where early stages form coarse structures with more stable feature dynamics (prediction error accumulates more slowly, allowing larger intervals), while later stages refine fine-grained details with more dynamic feature changes (requiring shorter intervals).

## 7. All Mathematical Formulas and Symbol Definitions
Forward process definition:
xt = √αt x0 + √(1−αt)ϵ, ϵ ∼ N(0, I)

Reverse process (DDIM):
xt−1 = √αt−1 ( xt − √(1−αt)ϵθ(xt, t) / √αt ) + √(1−αt−1)ϵθ(xt, t)

TaylorSeer m-th order prediction:
OlTaylor(t− k) = Ol(t) + ∑_{i=1}^m (k^i / i!) ∆i_N Ol(t) / N^i

Finite difference operator:
∆i_N Ol(t) = { Ol(t)−Ol(t+N), i = 1; ∆i−1_N Ol(t)−∆i−1_N Ol(t+N), i > 1 }

Ratio between output and input changes:
sk(t− k) = ||∆kO(t− k)||2 / ||∆kI(t− k)||2

Proposition 1 mapping:
O(t) = AI(t) + b

Output change derivation:
∆kO(t− k) = O(t− k)−O(t) = A(I(t− k)− I(t)) = A∆kI(t− k)

L2-norm of output change:
||∆kO(t− k)||2 = ||A∆kI(t− k)||2 = ||Auk(t− k)||2 ||∆kI(t− k)||2

Normalized direction:
uk(t− k) = ∆kI(t− k) / ||∆kI(t− k)||2

Ratio formula:
sk(t− k) = ||∆kO(t− k)||2 / ||∆kI(t− k)||2 = ||Auk(t− k)||2

Approximation of Eq (14):
||∆kO(t− k)||2 ≈ sN (t)||∆kI(t− k)||2

RFE prediction:
ORFE(t− k) = O(t) + (sN (t)||∆kI(t− k)||2)g( ∑_{i=1}^m (k^i / i!) ∆i_NO(t) / N^i )

L2 normalization:
g(x) = x / ||x||2

Output prediction error:
EO(t− k) = O(t− k)−ORFE(t− k)

Input prediction error:
EI(t− k) = I(t− k)− ITaylor(t− k)

Predicted input feature (Taylor expansion):
ITaylor(t− k) = I(t) + ∑_{i=1}^m (k^i / i!) ∆i_NI(t) / N^i

Relative L1 errors:
EO(t− k) = ||EO(t− k)||1 / ||O(t− k)||1, EI(t− k) = ||EI(t− k)||1 / ||I(t− k)||1

Accumulated error threshold for RCS:
∑_{j=1}^k EI(t− j) > τ

Linear extrapolation baseline:
Opred(t− k) = O(t) + (k/N) w(t)∆NO(t)

## 8. Algorithm Pseudocode
No algorithm pseudocode is provided in the original text.