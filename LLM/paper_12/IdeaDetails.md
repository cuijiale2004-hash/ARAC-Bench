1. Research Background and Existing Pain Points
Research Background: Test-time compute allocation in large reasoning models (LRMs) is widely used and has applications in mathematical problem solving, code synthesis, and planning. Recent advancements in Large Language Models (LLMs) have demonstrated remarkable capabilities in complex reasoning tasks, often mediated by a process known as Chain-of-Thought (CoT) prompting. Inspired by the CoT paradigm, modern LRMs achieve strong performance on complex tasks by allocating significant test-time compute to think before answering. A common yet underexplored phenomenon in their reasoning traces is the tendency to begin by repeating the user’s prompt, a behavior termed the Echo of Prompt (EOP). 

Existing Pain Points: Recent work has addressed the test-time compute allocation problem by scaling self-consistency and parallel thinking, adding generic “thinking tokens” and prompting models to re-read the question before answering. Unfortunately, these approaches either inject task-agnostic tokens or mandate heuristics that do not explain—and often ignore—the spontaneous repetition that many LRMs exhibit at the head of their internal chains. While uncontrolled repetition is a known failure mode (the “repeat curse”), explicit instructions to re-read or “look-twice” are known to improve performance. However, the spontaneous Echo of Prompt that initiates a complex reasoning trace has remained largely unanalyzed. Current methods treat repetition as an instructed heuristic to re-align the model, failing to analyze the spontaneous emergence of echoes as an intrinsic, learned strategy.

2. Core Research Motivation and Scientific Questions
Core Research Motivation: This paper is driven by the need to analyze and harness the model’s tendency to restate the question, termed the Echo of Prompt (EOP), as a front-loaded, compute-shaping mechanism. The work aims to reframe the EOP from a superficial flaw into a functional strategy for cognitive self-alignment, offering new insights into how models learn to structure their own thought processes for complex reasoning.

Scientific Questions: The initial echo raises a critical question: Is it a superfluous artifact of the training process, or does it serve a functional role in reasoning? Furthermore, is there a positive relationship between the probabilistic cost of an echo and the model’s final reasoning accuracy? In other words, does spending probability on an echo lead to better performance?

3. Overall Core Idea and Design Philosophy
Overall Core Idea: The Echo of Prompt (EOP) serves as an intrinsic attention-refocusing mechanism, a learned strategy to ground subsequent reasoning steps in the salient details of the original query. It is not merely about re-reading the question, but about anchoring the subsequent reasoning process to a stable internal representation (the answer prefix), a mechanism that strongly correlates with correctness. EOP acts as a cognitive scaffold for higher-level reasoning.

Design Philosophy: The design philosophy bridges the gap between emergent phenomena and deliberate cognitive design. It moves beyond treating repetition as a heuristic to be added, and instead formalizes it probabilistically and mechanistically. By quantifying the probabilistic cost via rejection-based conditioning and identifying the attention refocusing mechanism in middle layers, the philosophy advocates for cultivating beneficial thought processes through both training-based (Echo-Distilled SFT) and training-free (Echoic Prompting) interventions that harness this emergent behavior.

4. Core Innovation Points
1. A Probabilistic Framework based on Rejection Sampling: The paper introduces a novel probabilistic framework to formalize the EOP by casting echo removal as rejection-based conditioning. It defines the Echo Likelihood Gap (∆L) as a computable proxy to quantify the probabilistic cost, providing the missing theoretical link that connects early repetition to likelihood gains and downstream accuracy.
2. A Mechanistic Explanation via Attention Analysis: The paper uncovers the underlying mechanism through attention analysis, showing that EOP serves to refocus the model’s attention. It identifies that EOP increases answer to answer-prefix attention in middle layers (layers 7–18), consistent with an attention refocusing mechanism, rather than simply increasing attention to the original question.
3. Echo-Distilled SFT (ED-SFT): A practical supervised fine-tuning method developed to instill an “echo-then-reason” pattern through supervised finetuning. It uses a teacher model and an MLP probe to construct distilled data that explicitly includes the echo, yielding significant performance gains that generalize across data distributions.
4. Echoic Prompting (EP): A training-free inference strategy developed to re-ground the model mid-trace without training. It re-introduces the prompt via a natural echo, outperforming strong baselines like Thinking Token based Test-time Scaling (TTTS) by using task-specific context rather than generic, task-agnostic stimuli.

5. Overview of the Overall Technical Solution
The overall technical solution proceeds in three major phases: Probabilistic and Mechanistic Analysis, Training-Based Implementation, and Training-Free Implementation.
Phase 1 (Analysis): The solution begins by formalizing the impact of prompt echoes using a probabilistic framework that treats the presence or absence of an echo as a probabilistic event. This defines a hypothetical echo-free model and measures the echo’s likelihood cost using the Echo Likelihood Gap (∆L) and Suffix-only Likelihood Gap (∆Lsuffix). Following this, a mechanistic explanation is developed through layer-wise attention analysis, comparing attention weights from answer tokens to question tokens versus answer prefix tokens. This identifies the middle-layer dominance where attention refocusing occurs.
Phase 2 (Training-Based Implementation): Based on the insight that echoes are beneficial, Echo-Distilled SFT (ED-SFT) is developed. It uses a high-capacity teacher model to generate verified reasoning traces. An MLP probe detects the absence of EOP, and the teacher model is prompted to minimally insert a short echo-style opening for those traces. Models are fine-tuned on this echo-augmented dataset versus a normal SFT dataset where echoes are removed.
Phase 3 (Training-Free Implementation): For inference-time enhancement, Echoic Prompting (EP) is designed. After the model produces an initial reasoning chain, a reminder phrase and the original question are appended to encourage the model to revisit the problem’s context and continue generating a more grounded response. This is compared against generic thinking token insertion baselines.

6. Detailed Module Design
Module 1: Probabilistic Cost Framework
This module formalizes the EOP probabilistically. Let x ∈ X be an input prompt and y ∈ Y be a generated output sequence. A base LRM parameterized by θ defines a conditional probability distribution πθ(y|x). A predicate (implemented by a trained MLP) partitions the output space Y into two disjoint sets: Y = Ytrim ∪ Yecho, where Ytrim is the set of all trimmed sequences deemed echo-free, and Yecho is its complement. An indicator function is defined. Assuming the model has a non-zero probability of producing at least one echo-free trace (Zx > 0), the target trimmed distribution τθ(y|x) is defined as the base distribution conditioned on the event that the output is echo-free. The denominator Zx is the partition function. To measure the effect without computing the intractable partition function, the Echo Likelihood Gap (∆L) is introduced as a sample-based alternative, comparing the average log-likelihood of a generated trace (yraw) against its trimmed counterpart (ytrim). To isolate the echo’s influence on subsequent reasoning steps, the Suffix-only Likelihood Gap (∆Lsuffix) is defined for a raw trace composed of an echo prefix e and a reasoning suffix s.

Module 2: Attention Analysis Mechanism
This module quantifies the attention dynamics to explain EOP's effectiveness. Let A(l) ∈ RT×T be the head-averaged attention matrix at layer l for a full sequence of T tokens. The average attention weight from a set of query tokens SQ to a set of key tokens SK is defined. For the answer→answer-prefix metric, SQ comprises the indices of all tokens in the generated reasoning trace, while SK contains the indices of the initial K tokens of that same trace (dynamically set to the per-sample echo length). For the answer→question metric, SK contains the indices of the original question tokens. The layer-wise analysis localizes the EOP effect to the model’s reasoning bottleneck, observing a Middle-Layer Dominance where the attention gap peaks in layers 7–18. The Differential Impact confirms that the echo acts as a distinct working memory anchor, as correct traces attend significantly more to the answer-prefix than to the original question. Layer-wise discriminability is quantified using AUC and Cohen’s d between Correct and Wrong groups.

Module 3: MLP Probe for Echo Detection
To operationalize the probabilistic framework and data distillation, a lightweight two-layer MLP probe is designed for binary classification of echo prefixes. 
Input Features: For each (question, think_content) pair, features are constructed by concatenating two sentence embeddings: the full question and the initial prefix of the think_content (first 32 word tokens), encoded using Qwen3-Embedding-0.6B and z-scored.
Architecture: A two-layer MLP with a 32-dimensional hidden layer and ReLU activation, mapping the concatenated embedding to a single logit.
Training: Trained using weighted binary cross-entropy on logits (BCEWithLogitsLoss), with the positive class weight set to the ratio of negative to positive samples. Optimized with AdamW (learning rate 10−4, weight decay 0.01, batch size 64) for up to 200 epochs with early stopping.
Usage: A calibrated threshold on the sigmoid output with a hysteresis scheme (initial threshold 0.6, drop threshold 0.15) ensures stable prefix detection.

Module 4: Echo-Distilled SFT (ED-SFT) Pipeline
This module instills the echo-then-reason pattern via fine-tuning.
Step 1: Construct a shared pool of high-quality teacher traces by querying gpt-oss-120B with a standard CoT prompt. Verify the final answer exactly matches the ground-truth solution; discard mismatches.
Step 2: For ED-SFT data, run the MLP probe. For traces flagged as missing EOP, call gpt-oss-120B with an edit instruction that minimally inserts a short echo-style opening (e.g., “Okay, let me see. The problem is asking: ...”) while preserving the subsequent reasoning and final answer. Traces with EOP are kept unchanged. Re-run the automatic checker and drop changed answers.
Step 3: For the normal-SFT baseline, start from the same verified traces. When the probe predicts EOP presence, prompt gpt-oss-120B to delete the echo prefix under a “do not change the reasoning or final answer” instruction. Discard examples with changed answers.
Step 4: Fine-tune target models (e.g., Qwen3-8B-Base, Qwen3-8B) using AdamW, identical learning-rate schedules, batch sizes, and maximum sequence lengths, differing only in the training corpus (ED-SFT vs. normal-SFT).

Module 5: Echoic Prompting (EP) Strategy
This is a training-free method for inference-time enhancement. After the model produces an initial reasoning chain, the strategy appends a reminder phrase such as "look back at the question again" followed by the original question itself. This intervention encourages the model to revisit the problem’s context and continue generating a more grounded response, re-grounding the model with task-specific context rather than generic thinking tokens.

7. All Mathematical Formulas and Symbol Definitions
Indicator function:
1y∈Ytrim = {1 if y is echo-free, 0 otherwise.}
(1)

Trimmed distribution:
τθ(y|x) = πθ(y|x)1y∈Ytrim / ∑y∈Ytrim πθ(y|x)
(2)

Partition function:
Zx = ∑y∈Ytrim πθ(y|x) = Ey∼πθ(·|x)[1y∈Ytrim ]
(3)

Average log-likelihood:
L(y) = (1/|y|) ∑t=1 to |y| log πθ(yt | x, y<t)
(4)

Echo Likelihood Gap:
∆L = L(yraw) − L(ytrim)

Suffix-only Likelihood Gap:
∆Lsuffix = L(s | x, e)− L(s | x)
(5)

Attention metric:
Attn(l)(SQ → SK) = (1/|SQ|) ∑i∈SQ ∑j∈SK A(l)ij
(6)

Cohen’s d:
d(l) = µ(A(l)C )− µ(A(l)W ) / s(l)p
(7)

Likelihood Decomposition linking ∆L to Correctness:
Let πθ(y | x) be the base model and τθ(y | x) = πθ(y | x)1y∈Ytrim/Zx the trimmed distribution with Zx > 0. For a raw trace yraw = [e, s] and its trimmed counterpart ytrim = s, define the per-token log-likelihood Lπ(y | x) = 1/|y| ∑t log πθ(yt | x, y<t), and the Echo Likelihood Gap ∆L = Lπ(yraw | x)− Lπ(ytrim | x).
Because log τθ(ytrim | x) = log πθ(ytrim | x)− logZx, we have, for n= |ytrim|,
Lτ (ytrim | x) = Lπ(ytrim | x)− c(x, n) = Lπ(yraw | x)−∆L − c(x, n),
where the “constant” shift is c(x, n) = (1/n) logZx. Taking conditional expectations with respect to the correctness label G ∈ {C,W} yields
E[Lτ | G] = E[Lπ(yraw | x) | G]− E[∆L | G]− E[c(x, n) | G]

Therefore,
E[Lτ | C]− E[Lτ | W] = (E[Lπ(yraw | x) | C]− E[Lπ(yraw | x) | W]) − (E[∆L | C]− E[∆L | W]) − (E[c(x, n) | C]− E[c(x, n) | W])

Logistic Regression:
logit(P (Y = 1)) = β0 + β1∆L+ β2Lecho 
(8)

8. Algorithm Pseudocode
The original paper does not contain explicit algorithm pseudocode blocks. However, the procedural logic for the proposed methods is detailed in the text and extracted in the Detailed Module Design section (specifically Modules 4 and 5).