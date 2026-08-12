**Principle 1: Theoretical Tractability and Architectural Fidelity in Mechanistic Analysis of Transformer Optimization Dynamics**

**Definition:**  
When studying how Transformer training dynamics induce specific representational biases—such as the suppression of spurious features—reviewers evaluate whether the theoretical analysis relies on overly simplified or non-representative architectural components (e.g., two-key attention blocks, staged parameter updates, squared-parameter heads) and whether the authors provide rigorous justification or empirical evidence that the identified mechanism transfers to standard architectures trained with joint gradient descent. This principle emphasizes that theoretical convenience must be carefully balanced with evidence of generalizability to realistic Transformer pipelines. It requires authors to articulate precisely which core inductive biases are preserved in their simplification and to validate that the phenomenon is not an artifact of the idealized protocol.

**Core Evaluation Criteria:**
- Does the work clearly articulate why specific architectural simplifications (e.g., decoupled gate/head updates, restricted attention patterns, or custom parameterizations) are mathematically necessary for the theoretical result?
- Is there empirical validation on standard architectures (e.g., multi-head self-attention, ViT, BERT) demonstrating that the core phenomenon persists under conventional joint training without staged schedules or freezing?
- Are the identified mechanistic ingredients (e.g., time-varying softmax competition, multiplicative heads) proven sufficient in the simplified model, and is their preservation in standard settings explicitly verified?
- Does the work avoid overclaiming that results derived under idealized training protocols apply to “standard training alone” without either theoretical argument or controlled empirical evidence?

---

**Principle 2: Causal Claim Rigor and Distinguishing Dominant Correlation from Interventional Robustness in Feature Selection**

**Definition:**  
In research claiming that models learn “cause-only” predictors, reviewers scrutinize whether the causal terminology is justified beyond the observation that the most strongly correlated feature dominates predictions. This involves assessing whether the work establishes true interventional or robustness properties—such as invariance under distribution shifts of spurious descendants, formal anti-causal analysis, or out-of-distribution guarantees—rather than merely formalizing correlation dominance. The principle demands terminological precision and insists that authors distinguish implicit bias toward statistically dominant signals from principled causal learning that remains stable when spurious pathways are perturbed.

**Core Evaluation Criteria:**
- Does the work formally define “causal” versus “spurious” features and justify why the dominant feature is causal rather than merely the strongest available correlating signal?
- Are out-of-distribution or intervention-style experiments provided to verify that the predictor relies on invariant mechanisms rather than dominant training correlations?
- Does the work address anti-causal settings or the statistical validity of descendant-based prediction, clarifying when and why gradient descent avoids z→y shortcuts?
- Is the causal terminology appropriately scoped (e.g., acknowledging the absence of true interventional causal identification) to avoid overinterpretation of correlation-driven dynamics?

---

**Principle 3: Experimental Design Validity and Baseline Comparisons for Spurious Feature Suppression in Vision and Language**

**Definition:**  
When evaluating empirical claims that specific architectures suppress spurious correlations, reviewers assess whether the experimental design successfully isolates the claimed mechanism from confounding factors. This includes controlling for low-level feature quality asymmetries (e.g., foreground versus background texture richness), selecting realistic and challenging bias strengths, providing fair baseline comparisons across diverse architectures (e.g., CNNs, MLPs, EfficientNets), and quantifying spuriousness explicitly. The principle ensures that observed attention shifts or accuracy gains reflect genuine optimization-driven causal preference rather than trivial correlation dominance, dataset artifacts, or architectural indifferences.

**Core Evaluation Criteria:**
- Are experiments designed to control for feature quality differences (e.g., using foreground-foreground setups or matched complexity) that could independently drive attention shifts?
- Is the spurious correlation strength sufficiently high to pose a meaningful challenge, and does the work characterize boundary conditions where suppression fails?
- Are fair comparisons against non-Transformer baselines conducted to establish whether the effect is architecture-specific or generic to any model facing a dominant correlation?
- Is spurious correlation explicitly quantified and manipulated (e.g., via controlled conditional sampling), and are quantitative metrics reported rather than relying solely on qualitative visualizations?

---

**Principle 4: Two-Phase Optimization Dynamics Characterization and Sensitivity Analysis of Dominance Conditions**

**Definition:**  
Research proposing multi-phase training dynamics—such as an early “occupation” phase followed by a “crowding-out” phase—must provide both rigorous theoretical characterization and granular empirical tracing of each phase, their coupling, and the precise conditions that trigger them. Reviewers evaluate whether the dynamics are derived under realistic optimization protocols, whether key conditions (e.g., uniform dominance gaps, per-sample sign-stability, margin requirements) are interpretable and robustly justified, and whether the boundary between successful feature selection and failure is empirically mapped. This principle demands that mechanistic claims about training trajectories be accompanied by ablation studies and sensitivity analyses.

**Core Evaluation Criteria:**
- Is each phase formally defined with convergence guarantees or clear empirical signatures (e.g., weight norm trajectories, attention entropy curves) that can be independently verified?
- Are the operative data conditions (e.g., dominant-coordinate conditions) interpretable, and is their sensitivity to violations (e.g., label noise, sign flips, mixed-strength spurious features) empirically assessed?
- Does the work identify critical thresholds (e.g., minimum dominance gap, maximum tolerable corruption rate) and validate them through controlled degradation experiments?
- Are the dynamics analyzed under joint, simultaneous optimization rather than staged freezing, and do ablations isolate the unique contribution of each proposed phase?

---

**Principle 5: Out-of-Distribution Robustness Guarantees and Test-Time Generalization Under Spurious Feature Shifts**

**Definition:**  
For work connecting training dynamics to causal or robust prediction, reviewers evaluate whether the learned predictor’s robustness is formally guaranteed and empirically verified under test-time distribution shifts—particularly perturbations that alter the spurious feature distribution (e.g., background changes, style shifts, or direct interventions on descendants). This principle demands that “cause-only” behavior be accompanied by high-probability generalization bounds and systematic out-of-distribution experiments demonstrating stable performance when spurious correlations change or break entirely. It insists on establishing a causal link between the training mechanism and test-time invariance.

**Core Evaluation Criteria:**
- Does the work provide formal generalization guarantees (e.g., test risk bounds) for the cause-only predictor, and are corollaries provided for robustness under specific shifts (e.g., changes to p(z|y))?
- Are empirical out-of-distribution evaluations conducted with varying test-time bias strengths or environmental shifts, and do they demonstrate the predicted phase transitions?
- Is the relationship between the training-dynamics mechanism (e.g., crowding-out) and test-time robustness causally established, or is robustness merely inferred from improved training accuracy?
- Does the work explicitly test the limits of robustness by probing scenarios where the dominance gap is eroded at test time, confirming that the mechanism generalizes only under the claimed conditions?