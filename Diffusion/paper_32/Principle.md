### Principle 1: Validity of Temporal Alignment and Tokenization Strategies for Continuous Financial Signals in Discrete LLM Architectures

**Definition:**  
This principle evaluates whether the proposed method fundamentally resolves the inherent mismatch between discrete token-based LLM architectures and continuous, high-frequency financial time series data. Reviewers must assess if the tokenization or embedding strategy preserves critical temporal dependencies, trend information, and volatility patterns without introducing artifacts or information loss that renders the LLM's semantic reasoning capabilities irrelevant to the forecasting task. It is crucial to determine whether the adaptation treats financial data as a genuine sequential signal with unique statistical properties (e.g., non-stationarity, low signal-to-noise ratio) rather than merely forcing it into a text-like format that ignores domain-specific dynamics. The evaluation must distinguish between methods that leverage the LLM’s pre-trained knowledge of economic concepts versus those that simply use the transformer architecture as a generic sequence modeler.

**Core Evaluation Criteria:**
-   **Preservation of Temporal Dynamics:** Does the tokenization/embedding scheme explicitly maintain causal ordering and multi-scale temporal dependencies, or does it inadvertently destroy autocorrelation structures essential for financial forecasting?
-   **Semantic-Signal Integration:** Is there clear evidence that the LLM’s textual pre-training contributes to forecasting performance (e.g., through cross-modal alignment with news/sentiment), or would a randomly initialized transformer perform identically?
-   **Handling of Non-Stationarity:** Does the input representation include mechanisms (e.g., normalization, differencing, adaptive scaling) to handle distribution shifts common in financial markets, preventing the LLM from learning spurious correlations?
-   **Ablation of Representation Choices:** Are comprehensive ablations provided comparing different tokenization granularities, patch sizes, or embedding layers to justify the specific design choice over simpler baselines?

---

### Principle 2: Rigorous Prevention of Look-Ahead Bias and Data Leakage in Pretraining-Finetuning Pipelines for Financial Markets

**Definition:**  
Given the extreme sensitivity of financial data to temporal leakage, this principle mandates strict verification that no future information contaminates the training, validation, or testing phases, particularly when leveraging large-scale pretraining corpora or external knowledge bases. Reviewers must scrutinize whether the dataset splitting respects chronological order, whether feature engineering (e.g., normalization statistics, technical indicators) uses only past data, and whether any auxiliary text data (news, reports) is strictly timestamp-aligned. In the context of LLMs, special attention must be paid to whether the pretraining corpus contains post-hoc analyses or future-referencing content that could artificially inflate performance. Failure to demonstrate leak-proof experimental design invalidates all claimed predictive gains regardless of architectural novelty.

**Core Evaluation Criteria:**
-   **Chronological Integrity:** Is the train/validation/test split strictly time-based (never random), and are all preprocessing steps (scaling, imputation) fitted solely on training data?
-   **Auxiliary Data Synchronization:** If multimodal inputs (news, social media) are used, is there explicit verification that each text sample’s timestamp precedes the corresponding prediction target window?
-   **Pretraining Corpus Audit:** Has the authors verified that the LLM’s base knowledge or continued pretraining data does not contain retrospective market analyses or leaked labels relevant to the test period?
-   **Reproducibility of Leakage Checks:** Are code or detailed protocols provided allowing reviewers to independently verify the absence of look-ahead bias in the entire pipeline?

---

### Principle 3: Statistical Significance Testing and Economic Utility Validation Beyond Standard Point Forecast Metrics

**Definition:**  
In financial forecasting, marginal improvements in MSE/MAE are often statistically insignificant or economically meaningless due to market noise and transaction costs. This principle requires that evaluation extends beyond point forecast accuracy to include rigorous statistical tests (e.g., Diebold-Mariano, Hansen’s SPA) and realistic economic utility metrics (e.g., risk-adjusted returns, Sharpe ratio, maximum drawdown under transaction costs). Reviewers must reject claims of superiority based solely on average metric improvements without confidence intervals or significance testing. Furthermore, the evaluation should assess whether the model’s predictions translate to actionable trading signals or risk management insights, distinguishing between academic curve-fitting and genuine market-relevant predictive power.

**Core Evaluation Criteria:**
-   **Statistical Robustness:** Are pairwise comparison tests against strong baselines reported with p-values or confidence intervals, rather than just tabulated mean scores?
-   **Economic Realism:** Is performance evaluated using profit/loss metrics incorporating realistic slippage, fees, and position sizing constraints, not just directional accuracy?
-   **Risk-Aware Evaluation:** Are tail-risk metrics (VaR, CVaR, drawdown) included to ensure the model doesn’t achieve higher returns by taking uncompensated risks?
-   **Baseline Competitiveness:** Are comparisons made against domain-specific strong baselines (e.g., ARIMA-GARCH, LightGBM with financial features, specialized temporal models like PatchTST), not just generic NLP models?

---

### Principle 4: Interpretability and Causal Attribution of LLM Reasoning Processes in Noisy Financial Environments

**Definition:**  
For LLMs applied to finance, it is insufficient to treat the model as a black box; reviewers must evaluate whether the work provides credible evidence that the LLM’s outputs stem from meaningful reasoning about market dynamics rather than pattern matching on noise. This includes assessing the quality of chain-of-thought explanations, attention map interpretations, or probing analyses that link internal representations to known financial factors. Crucially, the interpretation must be validated against domain expertise or ground-truth causal relationships, not just plausibility. In a low-signal domain like finance, interpretability serves as a sanity check against spurious correlations and is essential for regulatory compliance and practitioner trust.

**Core Evaluation Criteria:**
-   **Faithfulness of Explanations:** Do generated rationales or attention patterns correlate with actual predictive drivers, verified via perturbation tests or expert assessment?
-   **Domain-Grounded Reasoning:** Does the LLM reference valid economic theories, market microstructure concepts, or macroeconomic indicators, rather than generating generic or hallucinated justifications?
-   **Consistency Across Regimes:** Does the model’s reasoning adapt appropriately to changing market conditions (bull/bear/crisis), or does it apply static narratives regardless of context?
-   **Quantitative Link to Performance:** Is there analysis showing that instances with coherent/valid reasoning correspond to more accurate forecasts, establishing a connection between interpretability and reliability?

---

### Principle 5: Comprehensive Benchmarking Against Domain-Specific Baselines and Robustness Across Market Regimes

**Definition:**  
The validity of an LLM-based financial forecasting method cannot be established through comparisons with general-purpose time series models alone; it must be benchmarked against established financial econometric models and state-of-the-art ML approaches specifically designed for market data. This principle evaluates whether the experimental design covers diverse market regimes (trending, mean-reverting, volatile, crisis periods) and multiple asset classes to demonstrate robustness. Reviewers should assess whether performance gains are consistent across these varying conditions or limited to specific favorable periods that may reflect overfitting. The principle also demands transparency regarding computational cost-efficiency trade-offs, as LLMs’ resource intensity must be justified by commensurate practical advantages over lighter alternatives.

**Core Evaluation Criteria:**
-   **Domain-Specific Baseline Coverage:** Does the study include comparisons with classical financial models (GARCH family, cointegration), gradient-boosted trees with financial features, and recent neural forecasting architectures (e.g., iTransformer, TimesNet)?
-   **Regime-Stratified Evaluation:** Is performance reported separately for distinct market states (e.g., high vs. low volatility, bull vs. bear markets) to reveal conditional strengths/weaknesses?
-   **Cross-Asset Generalization:** Has the method been tested on multiple instruments (stocks, indices, forex, commodities) with differing statistical properties?
-   **Efficiency-Performance Trade-off Analysis:** Is inference latency, memory footprint, or training cost discussed relative to performance gains, providing guidance on practical deployment viability?