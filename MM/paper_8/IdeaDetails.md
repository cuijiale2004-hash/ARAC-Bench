# 1. Research Background and Existing Pain Points

## Research Background
The aspiration for artificial general intelligence, fueled by the rapid progress of multimodal understanding, demands models to understand humans in diverse and complex scenarios, as humans manifests intelligence and embody the world. The rapid advancement of Multimodal Large Language Models (MLLMs) has shown remarkable potential in understanding diverse contexts. This progress fuels the aspiration toward artificial general intelligence, where a key prerequisite lies in the ability to understand humans in diverse, complex, and dynamic contexts, as human behavior inherently reflects intelligence as well as the complexities of the world. Human-centric visual understanding remains a fundamental challenge in artificial intelligence.

## Existing Pain Points
1.  **Limited Assessment Scope of Current Benchmarks**: Current MLLM benchmarks provide a limited assessment of human-centric capabilities. They either isolate narrow tasks, such as action or facial recognition, or, when adopting broader scopes, overlook intricate yet crucial aspects such as gaze and contact, while reporting only coarse-grained scores.
2.  **Lack of Probing Power**: Existing benchmarks lack the necessary probing power to rigorously evaluate MLLMs’ nuanced capabilities in human-centric scenarios, thus providing limited guidance for future research and applications.
3.  **Gap in Reasoning Assessment**: A more critical gap lies in the assessment of reasoning. Unlike humans, who naturally synthesize multiple visual cues in reasoning, current task-specific or fragmented benchmarks rarely challenge models to perform multi-evidence reasoning. Although some recent video benchmarks have featured intricate reasoning questions, they often cannot necessitate sophisticated visual evidence demand.
4.  **Overlooked Critical Reasoning Faculties**: Two critical reasoning faculties remain overlooked: 1) the ability of integrating multiple, disparate visual evidence to achieve coherent understanding, and 2) the ability of proactively seeking implicit visual cues.
5.  **Shortcuts in Existing Benchmarks**: General video benchmarks typically assess shallow comprehension with limited reasoning diversity and depth. Early works either struggle with question quality due to semi-automatic annotation or are confined to narrow source or task. While concurrent benchmarks require multi-frame reasoning, their task-specific designs limit the diversity of evidence and reasoning paths. Most importantly, their multiple-choice or certain question references often implicitly reveal the required visual evidence, failing to evaluate the crucial ability of proactive evidence seeking.
6.  **Single-Source Bias**: Existing benchmarks often suffer from single-source bias and lack diverse data from daily to professional scenes.

# 2. Core Research Motivation and Scientific Questions

## Core Research Motivation
The core motivation is to systematically investigate how well MLLMs understand humans across critical aspects of perception, comprehension, and reasoning in diverse human-centric visual understanding scenarios. There is a pressing need for a benchmark that provides a comprehensive and fine-grained taxonomy for probing human-centric capabilities across a diverse range of tasks and modalities, and a paradigm for video reasoning that integrates disparate visual evidence and proactively seeks implicit visual cues beyond what is explicitly prompted.

## Scientific Questions
1.  How well do current MLLMs understand humans across critical aspects of perception, comprehension, and reasoning in diverse, complex, and dynamic human-centric visual understanding scenarios?
2.  Can MLLMs integrate multiple, disparate visual evidence to achieve coherent understanding in complex human scenes?
3.  Can MLLMs proactively seek implicit visual cues beyond what is explicitly prompted in the question, rather than relying solely on referred evidence?
4.  Do advanced video understanding configurations (like visual context extraction or test-time computation scaling) truly address the challenges in complex video reasoning involving multiple and proactive evidence?

# 3. Overall Core Idea and Design Philosophy

## Overall Core Idea
The paper proposes HumanPCR, a comprehensive evaluation suite designed to meticulously probe the human-centric visual understanding of MLLMs. HumanPCR is structured along a hierarchical taxonomy, perception (Human-P), comprehension (Human-C), and reasoning (Human-R). Human-P and Human-C feature a large-scale dataset of over 6,000 image- and video-based QA pairs, assessing 34 tasks that span 9 dimensions from individual attributes to spatio-temporal dynamics. Human-R introduces a unique challenge through a manually curated, open-ended video reasoning benchmark sourced from 11 diverse human-related domains, which compels models to integrate multiple, disparate visual evidence and proactively seek implicit visual cues. Each question in Human-R is augmented with expert-annotated Chain-of-Thought (CoT) rationales that detail all key visual evidence.

## Design Philosophy
1.  **Sufficient Fine-Grained Tasks with Comprehensive Coverage and Low Redundancy**: A critical principle is to construct a sufficient number of fine-grained tasks with comprehensive coverage and low redundancy.
2.  **Rich and Varied Data Sources**: Each task should be supported by data sources as rich and varied as possible. An iterative approach, grounded in diverse data from daily to professional scenes, mitigates single-source bias and ensures broad capability coverage.
3.  **Criteria for Reasoning Level (Human-R)**: The evaluation of Reasoning should satisfy three criteria:
    *   **Visual Complexity**: Questions should require sufficient visual evidence, and exclude redundant content, going beyond simple concept retrieval.
    *   **Reasoning Necessity and Diversity**: Questions should engage diverse reasoning chains rather than be limited to a few reasoning patterns.
    *   **Proactivity**: Questions should demand proactive extraction of visual evidence over the abundant contexts, rather than relying solely on the referred evidence in the question.

# 4. Core Innovation Points

1.  **Hierarchical Taxonomy for Human-Centric Evaluation**: HumanPCR probes MLLMs’ capacity in human-centric visual contexts across three hierarchical levels: Perception (Human-P), Comprehension (Human-C), and Reasoning (Human-R), providing a structured and systematic evaluation framework.
2.  **Large-Scale Fine-Grained Perception and Comprehension Benchmark**: Human-P and Human-C consist of over 6,000 multiple-choice questions evaluating 34 fine-grained tasks covering 9 essential dimensions (Spatiality, Posture, Appearance, Contact, Identity, Behavior, Procedure, Relation, Scene), enabling detailed ability diagnosis.
3.  **Paradigm for Video Reasoning with Multiple and Proactive Evidence**: Human-R presents a manually curated challenging video reasoning test that uniquely requires integrating multiple, disparate visual evidence and proactively extracting implicit context beyond question cues, addressing the failure of existing benchmarks that implicitly reveal required evidence.
4.  **Expert-Annotated Chain-of-Thought with Visual Evidence**: Each question in Human-R includes human-annotated Chain-of-Thought (CoT) rationales with key visual evidence to support further research and precise diagnosis of reasoning failures.
5.  **Rigorous Quality Control Filtering for Proactive Reasoning**: The construction pipeline rigorously filters out annotations that fall below the required reasoning complexity (specifically those relying on reference shortcuts), ensuring that every question requires integrating multiple visual evidence and relies on at least one essential proactive visual evidence.
6.  **Diagnostic Insights on Proactive Evidence Extraction**: Through extensive evaluations and intervention studies, the paper identifies a critical failure in current MLLMs: models struggle to proactively gather necessary visual evidence, instead showing a faulty reliance on query-prompted cues, and that merely scaling visual contexts offers little gains for complex reasoning.

# 5. Overview of the Overall Technical Solution

The overall technical solution for constructing the HumanPCR benchmark follows a comprehensive pipeline including Taxonomy and Data Source, Annotation, and Quality Control.

1.  **Taxonomy and Data Source**: The taxonomy is defined by surveying a wide range of human-centric perception and understanding works. Tasks are matched with rich and varied datasets (iterative approach). For the Reasoning level (Human-R), public academic datasets are supplemented with web videos collected from 11 defined domains (ranging from daily life to professional scenarios) via domain-relevant tags, followed by manual review for content richness and safety.
2.  **Annotation**:
    *   **Human-P and Human-C**: Leverage annotations from existing datasets. Templates- and LLM-based generation (using GPT-4o) are used to create Multi-Choice questions and options. For under-explored tasks, manual generation is done with the assistance of domain-specific expert annotators.
    *   **Human-R**: Domain experts are recruited to annotate open-ended questions encompassing 5 distinct types of reasoning: Causal Reasoning, Prediction, Counter-Factual Reasoning, Assessment, and Planning. Answering requires integrating multiple pieces of visual evidence, engaging diverse reasoning chains, and extracting at least one proactive visual evidence. Detailed reasoning steps and necessary visual evidence are also annotated.
3.  **Quality Control and Verification**:
    *   **Human-P & Human-C**: QA pairs are first filtered by LLMs to eliminate those solvable without visual input (Blind Filtering), followed by human verification for linguistic quality, answer accuracy, distractor plausibility, and reliance on visual context.
    *   **Human-R**: Reviewers manually evaluate for objectivity, factual correctness, conciseness, and reasoning complexity, specifically eliminating reference redundancy. Meta-reviewers further assess question complexity, ensuring every question requires integrating multiple visual evidence (cannot be fully determined from question alone) and relies on at least one essential proactive visual evidence.
4.  **Evaluation Setup**:
    *   **Configuration**: Direct Answer prompts for multi-choice questions and CoT for open-ended ones. Video inputs are processed by sampling 32 frames for multi-choice tasks and the maximum allowable frames for open-ended tasks.
    *   **Metrics**: Accuracy for multi-choice is determined by matching responses to the correct option. Macro-average accuracy of tasks for each dimension or level is reported. For open-ended questions, a proprietary model (o3-mini) is used as a judge, which demonstrates high agreement with human evaluations.

# 6. Detailed Module Design

## Level 1: Perception (Human-P)
Perception evaluates visual recognition across 5 dimensions and 17 tasks:
*   **Spatiality**: Perceiving existence of people, objects, and their spatial relations. Tasks: Spatial Relation, Object Existence, Human Presence.
*   **Posture**: Recognizing physical position and orientation of body parts, hands, and gaze. Tasks: Body Posture, Hand State, Hand-Object Interaction, Body Orientation, Gaze Estimation.
*   **Appearance**: Identifying human appearances, including inherent attributes and attirement. Tasks: Clothing Attribute, Accessory Recognition, Bodypart Visibility, Physical Attribute.
*   **Contact**: Recognizing detailed interaction regions between people and objects, or themselves. Tasks: Human-Object Contact, Human-Human Contact, Human Self-Contact.
*   **Identity**: Recognizing people’s identity. Tasks: Face Recognition, Identity Clustering.

## Level 2: Comprehension (Human-C)
Comprehension assesses visual concepts comprehension, from 4 dimensions and 17 tasks, based on commonsense or domain-specific cues:
*   **Behavior**: Understanding human actions and body movements, such as gestures and emotions. Tasks: Gesture, Emotion, Basic Action, Knowledge-Based Action.
*   **Procedure**: Thoroughly understanding long-term activities, including underlying intentions and dependence among action sequences. Tasks: Sequential Action, Goal Planning, Procedure Dependence, Irrelevant Action, Multiple Human Sequential Action.
*   **Relation**: Analyzing relations, roles and differences among individuals. Tasks: Human Comparison, Social Relation.
*   **Scene**: Interpreting group dynamics or human activities within broader contexts. Tasks: Group Activity, Struggle Detection, Crowd Event, Cultural Event, Crime Abnormal, Disaster Understanding.

## Level 3: Reasoning (Human-R)
Reasoning examines whether models can integrate continuous, tightly coupled human dynamics within complex scenes for reasoning.
*   **Domains**: Electric & Crafting, Outdoor Leisure & Recreation, Outdoor Adventure, Household, Sports, Transport, Indoor Leisure & Self Care, Daily Spending, Performance & Exhibition, Education, Others.
*   **Reasoning Types**:
    *   **Causal Reasoning**: Involves identifying cause-effect relations in the video.
    *   **Prediction**: Involves anticipating what might happen next.
    *   **Counter-Factual Reasoning**: Asks about hypothetical alternatives.
    *   **Assessment**: Asks for judgment based on visual evidence and criteria.
    *   **Planning**: Requires proposing a viable plan grounded in context.
*   **Intervention Study Mechanism**: To probe the role of proactive evidence, the study progressively lowers the difficulty of visual evidence extraction while keeping the question and answer fixed, enriching prompts with three levels of guidance:
    *   **Level 1 (Relation awareness)**: Giving generic relation-type hints (e.g., “check surrounding context”).
    *   **Level 2 (Logic awareness)**: Additionally highlighting which referred cues are logically linked to potential proactive evidence.
    *   **Level 3 (Proactive guidance)**: Adding vague descriptions that loosely point to the proactive evidence without revealing the reasoning steps or answer.

## Annotation Pipeline Details
*   **Blind Filtering Prompt**: Prompt the LLM to provide its optimal answer to each question in the absence of the corresponding image or video. Any question answered correctly across multiple runs with randomly permuted answer choices is discarded.
*   **Annotator Qualification**: Annotators for specialized domains hold relevant undergraduate degrees or have at least three years of experience. Annotators formulate open-ended questions meeting complexity criteria and provide concise answers and detailed CoT with essential visual evidence. All annotations are first created in the native language and later translated professionally.
*   **Meta-Review Checklist**: Validating objectivity, factual accuracy, non-redundancy, and complexity; ensuring each annotation incorporates at least two distinct pieces of visual evidence and requires the integration of external knowledge triggered by visual content.

# 7. All Mathematical Formulas and Symbol Definitions
The paper does not contain mathematical formulas or symbol definitions.

# 8. Algorithm Pseudocode
The paper does not contain algorithm pseudocode.