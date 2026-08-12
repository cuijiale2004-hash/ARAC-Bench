## ABSTRACT

Learning how to reach goals in an environment is a longstanding challenge in AI, yet reasoning over long horizons remains a challenge for modern methods. The key question is how to estimate the temporal distance between pairs of observations. While temporal difference methods leverage local updates to provide optimality guarantees, they often perform worse than Monte Carlo methods that perform global updates (e.g., with multi-step returns), which lack such guarantees. We show how these approaches can be integrated into a practical offline GCRL method that fits a quasimetric distance using a multistep Monte-Carlo return. We show our method outperforms existing offline GCRL methods on long-horizon simulated tasks with up to 4000 steps, even with visual observations. We also demonstrate that our method can enable stitching in the real-world robotic manipulation domain (Bridge setup). Our approach is the first end-to-end offline GCRL method that enables multistep stitching in this real-world manipulation domain from an unlabeled offline dataset of visual observations and demonstrate robust horizon generalization.

## 1 INTRODUCTION

![](images/e7fccda14a3248883a88058ebb2b0b597488138efe47a16419e81775f572b1ed.jpg)  
Figure 1: In this paper, we present Multistep Quasimetric Estimation (MQE). Unlike prior work in quasimetric distance learning that use single-step TD updates (Wang et al., 2023) or contrastive learning-based Monte-Carlo updates (Myers et al., 2024), MQE is the first work to incorporate multistep returns with real-world success.

It is natural for humans to use inherent ideas of distances to represent task progress: a GPS will tell you how far you are from the destination and a cookbook will tell you how long a recipe will take. Humans can solve tasks by taking the shortest route possible (stitching) and combining several learned tasks together in sequence in a new environment (combinatorial generalization). The AI problem of reaching goals presents a rich structure (formally, an optimal substructure property) that can be exploited to decompose hard problems into easy problems, and reinforcement learning (RL) has been used to address such problems. However, many past attempts at leveraging this property in modern high-dimensional, stochastic settings still need much work. Current offline approaches tend to separate TD learning (local updates) (Mnih et al., 2013; Kostrikov et al., 2022; Kumar et al., 2020)

and MC learning (global value propagation) (Eysenbach et al., 2022; Myers et al., 2024); as TD learning can theoretically recover the optimal Q function $( Q ^ { * } )$ , while MC methods can only recover the behavior Q function $( Q _ { \beta } )$ , but tend to work well in practice.

However, the effectiveness of both the temporal difference and Monte Carlo methods degrades, often from a combination of increasing the horizon length for TD methods (Park et al., 2025b) or difficulties in finding the optimal temporal distance (Park et al., 2024a). These combined challenges present an intriguing opportunity to find methods that can leverage the advantages of both approaches, where one can perform both local and global value propagation at once.

Our main contribution lies in Multistep Quasimetric Estimation (MQE), an offline GCRL method that incorporates both multistep value learning and quasimetric architectures without needing explicit hierarchy. By leveraging such a unique combinations of the benefits of TD and MC methods, MQE allows the learned policy to (i): display a much stronger level of horizon generalization compared to previous methods, which allow us to demonstrate the desired “stitching” behavior, (ii): provide a stable method that can extract strong goal-reaching policies even from noisy data, and (iii): such stability in training allows it to be applied in real-world robot learning problems without additional design choices. To our knowledge, MQE is the first method capable of taking advantage of multistep TD returns with global value propagation through quasimetric architectures. MQE achieves SoTA performance on tasks that require complex control and long-horizon reasoning (up to 21 DoF and 4000 timesteps respectively), and in real world robotic settings, MQE displays compositional generalization behaviors that are not seen in previous RL algorithms.

## 2 RELATED WORK

Our work build upon previous works in offline RL and temporal distance learning.

Goal-conditioned Reinforcement Learning. Recent work has studied the stitching and horizon generalization capabilities of GCRL. Park et al. (2025b) show that while offline RL is easy to scale on short-horizon tasks, it is difficult to learn long-horizon tasks that require more complex reasoning within the agent, which can easily deviate from the optimal distance due to compounding TD errors. Other findings have shown that combinatorial generalization and stitching do not necessarily need dynamic programming (Ghugare et al., 2024; Brandfonbrener et al., 2023; Garg et al., 2023), which gives promise to using simpler methods for learning stitchable policies.

Offline RL for Robotics. We demonstrate offline GCRL scaling to real-world robotic tasks. Although reinforcement learning has been used to obtain highly capable specialist policies in various embodiments (Luo et al., 2025; Ball et al., 2023; Seo et al., 2025; Smith et al., 2023), behavior cloning still remains the most capable method for training generalist policies (O’Neill et al., 2024). Efforts have been made to allow self-supervised and offline RL in robotics (Zheng et al., 2024), however, directly training a policy with offline RL remains difficult, as many researchers have instead used other ways to incorporate RL, such as rejection sampling (Nakamoto et al., 2025; Wagenmaker et al., 2025) or data set curation (Mark et al., 2024; Xu et al., 2024). We show that our method can use multistep quasimetric learning to obtain effective policies for real-world robotic tasks, outperforming non-GCRL methods on long-horizon manipulation.

Multistep RL. RL using multistep returns has been widely used in online RL and offline-to-online RL settings (Li et al., 2025; Tian et al., 2025). This is desirable because in on-policy settings, performing RL with multistep return is theoretically correct and yields better performance (Munos et al., 2016; Asis et al., 2018; Schulman et al., 2016). However, in an offline setting, the theory behind such correctness breaks down (Watkins, 1989), as the learning paradigm becomes an inherently off-policy manner. Methods that use the entire trajectory in a Monte Carlo manner and attempt to recover the Q function can also be seen as a multistep RL algorithm (Eysenbach et al., 2022). However, such approach can only recover the behavior Q function $Q _ { \beta }$ (Myers et al., 2024; Eysenbach et al., 2022).

Temporal Distance Learning. Our work is closely related by works that focus on successor representations (Dayan, 1993; Blier et al., 2021) and uses them to learn the probability of reaching goals. Previous works have shown that by using contrastive learning (Myers et al., 2024), one can recover a temporal distance suitable for goal-reaching. Other works have also shown that using contrastive learning as a way to parameterize distance learning may also recover behavior Q function (Eysenbach et al., 2022). Previous works (Eysenbach et al., 2021; Myers et al., 2025c) have shown that it is possible to learn an optimal goal-reaching policy and Q function in theory. Our work differs in that we learn a Bellman optimal Q function under the bias of behavioral dynamics. This tradeoff helps us achieve considerable empirical gains in long-horizon goal-reaching tasks, and shows compositionality in real-world robotic learning problems.

## 3 PRELIMINARIES

In this section, we define temporal distances and our learning objective.

Notation. We consider a controlled Markov process (CMP) M with state space S, action space ${ \mathcal { A } } ,$ transition dynamics $\mathbf { P } ( s ^ { \prime } \mid s , a )$ , and discount factor as γ. We consider goal-reaching policies $\pi ( a \mid s , g ) : S \times S \to A \in \Pi$ . We denote the behavior policy as $\pi _ { \beta }$ . In lieu of rewards, we optimize the maximum discounted likelihood of a policy reaching the goal, in which we can represent the goal-conditioned Q function and the value function as:

$$
Q _ {g} ^ {\pi} (s, a) \triangleq \mathbb {E} _ {\{\mathfrak {s} _ {t}, \mathfrak {a} _ {t} \} \sim \pi} \left[ \sum_ {t = 0} ^ {\infty} \gamma^ {t} \mathrm{P} (\mathfrak {s} _ {t} = g \mid \mathfrak {s} _ {0} = s, \mathfrak {a} _ {0} = a) \right].\tag{1}
$$

$$
V _ {g} ^ {\pi} (s) \triangleq \mathbb {E} _ {\{\mathfrak {s} _ {t} \} \sim \pi} \Big [ \sum_ {t = 0} ^ {\infty} \gamma^ {t} \mathrm{P} (\mathfrak {s} _ {t} = g \mid \mathfrak {s} _ {0} = s) \Big ].\tag{2}
$$

Equivalently, we can define the optimal Q-function as $Q _ { q } ^ { * } ( s , a ) \triangleq \operatorname* { m a x } _ { \pi \in \Pi } Q _ { q } ^ { \pi } ( s , a )$ . Previous work have shown that using contrastive leaning (Eysenbach et al., 2022; van den Oord et al., 2019) can recover the behavior distance, but not the optimal distance.

Similarly, prior work on MC learning (Myers et al., 2024; Eysenbach et al., 2022; 2021) has demonstrated that future states can be used as goals. We use geometric distribution as described in Eq. (3). This allows us to classify any future state in trajectory as goals, providing a robust way of learning goal-reaching policies.

$$
\mathfrak {s} _ {t} ^ {+} \triangleq \mathfrak {s} _ {t + K}, K \sim \operatorname{Geom} (1 - \gamma).\tag{3}
$$

Quasimetric distance representations. Traditionally, offline RL algorithms use neural networks to represent the critic and value function $Q _ { g } ( s , a )$ and $V _ { g } ( s )$ (Kumar et al., 2020; Kostrikov et al., 2022). Separately, other works have been using dot products $Q ( s , a , g ) \triangleq \varphi ( s , a ) ^ { \mathsf { T } } \psi ( g )$ (Eysenbach et al., 2022; Zheng et al., 2024) or geometric norms $\lVert \varphi ( s ) - \dot { \psi } ( g ) \rVert _ { k }$ for suitable values of k (Eysenbach et al., 2024; Park et al., 2024c). We use quasimetric architectures (Wang et al., 2023; Pitis et al., 2020) to parameterize our value functions.

Formally, the space of quasimetric distances $\mathcal { Q } _ { \mathcal { X } }$ over a set X is defined as the set of functions $d : \mathcal { X } \times \overset { } { \mathcal { X } } \to \bar { \mathbb { R } } _ { > 0 }$ that satisfy the following properties for all $x , y , z \in { \mathcal { X } }$ (Cobza¸s, 2013):

(i) Non-negativity: $d ( x , y ) \geq 0$

(ii) Identity: $d ( x , x ) = 0 .$

(iii) Triangle Inequality: $d ( x , z ) \leq d ( x , y ) + d ( y , z )$

We will also use the notation $\mathcal { D } _ { \mathcal { X } }$ to denote the more general class of distances that satisfy only (i) and (ii).

To enforce the quasimetric properties of the successor distances, we use the Metric Residual Network (MRN) architecture (Liu et al., 2023), which parameterizes a space of quasimetrics in terms of a learned representation, as shown in Eq. (4). MRN splits the representation into N equally sized components, and in each part, takes the sum of an asymmetric component (maximum of ReLU) and a symmetric component $( l _ { 2 }$ norm) of the difference between the two embeddings x and y.

$$
d _ {\mathrm{MRN}} (x, y) \triangleq \frac {1}{N} \sum_ {k = 1} ^ {N} \max _ {m = 1 \dots M} \max (0, x _ {k M + m} - y _ {k M + m}) + \| x _ {k M + m} - y _ {k M + m} \| _ {2}\tag{4}
$$

Given any function $f : \mathcal { X } \to \mathbb { R } ^ { N M } , \operatorname { E q } .$ . (4) defines a quasimetric distance on X. Liu et al. (2023) show that for sufficiently large $M ,$ this parameterization is universal. Thus, fitting a quasimetric distance function can be accomplished by fitting $d _ { \mathrm { M R N } } ( f ( x ) , f ( y ) )$ for a sufficiently expressive class of representations $f \in \Phi$ . We make use of this architecture with a learned representation over $S \times \bar { A } \cup \mathcal { S }$ to express our goal-conditioned Q function $Q _ { g } ( s , a )$ and value function $\dot { V } _ { g } ( s )$ as distances between states/state-action pairs and goals, as demonstrated in (Myers et al., 2025c).

## 4 MULTISTEP QUASIMETRIC ESTIMATION (MQE)

In this section, we develop MQE framework based on quasimetric architectures, which can be broken down into 3 parts: (1) multistep backup under a quasimetric architecture, (2) imposing action invariance as a form of value learning, (3) practical implementation details for extracting the goal-reaching policy. To the best of our knowledge, MQE is the first method to effectively combine multistep returns with a quasimetric distance parameterization, and achieves superior results to previous methods that use TD returns, contrastive learning for value estimation, or hierarchical methods.

Definitions. We will define a distance (quasi)metric over states and state-action pairs that is proportional to the goal-conditioned $Q$ and V functions (Myers et al., 2024):

$$
Q _ {g} (s, a) = V _ {g} (g) e ^ {- d ((s, a), g)}, \qquad V _ {g} (s) = V _ {g} (g) e ^ {- d (s, g)}.\tag{5}
$$

We learn a parameterized form of this distance with learned state and state-action $\psi ( s ) \in \mathbb { R } ^ { N M }$ and $\phi ( s , a ) \in \mathring { \mathbb { R } } ^ { N M }$ representations and a quasimetric distance function (4): :

$$
d \bigl ((s, a), g \bigr) \triangleq d _ {\mathrm{MRN}} \bigl (\varphi (s, a), \psi (g) \bigr), \quad d (s, g) \triangleq d _ {\mathrm{MRN}} \bigl (\psi (s), \psi (g) \bigr).\tag{6}
$$

## 4.1 MULTISTEP RETURNS WITH QUASIMETRIC ARCHITECTURE

The first design principle of MQE is to use multistep returns under a quasimetric architecture. To that end, we start with the fitted onestep Q iteration that operates on quasimetric distance representations, in which ← denotes regressing from the LHS to the RHS:

$$
e ^ {- d ((s, a), g)} \leftarrow \mathbb {E} _ {\{s ^ {\prime} \sim \mathrm{P} (s ^ {\prime} | s, a) \} \sim \mathcal {D}} \big [ \gamma \cdot e ^ {- d (s ^ {\prime}, g)} \big ].\tag{7}
$$

This is similar to optimizing the critic in traditional RL (Kostrikov et al., 2022; Lillicrap et al., 2019), the main difference being that we use a quasimetric architecture to represent the Q and value function. Due to the quasimetric architecture of the network, the reward signal is embedded within the parameterized distance, as $V _ { g } ( s )$ is defined to have a value of $V _ { g } ( g ) > 0$ when $s = g$ . We denote this regression objective as $\tau$

Our insight for the T objective described in Eq. (7) is that, instead of applying this invariance to only the future state $s ^ { \prime } \sim \mathrm { P } ( \mathcal { \bar { s } } ^ { \prime } \mid s , a )$ , we can extend this principle to any state between the current state and goal. This transforms a onestep optimization into a on-policy multistep optimization procedure. To do so, we first define the shorthand $\mathfrak { s } _ { t } ^ { w }$ , which refers to a “waypoint” between the current state and goal state. Empirically, we find that using a combination of Bernoulli distribution and geometric distribution (described in Eq. (8)), capped at the index of the future state work the best.

$$
s _ {t} ^ {w} \leftarrow \mathfrak {s} _ {t + k ^ {\prime}} \quad \text { for } \quad \left\{ \begin{array}{l l} k ^ {\prime} \sim \min (\operatorname{Geom} (1 - \lambda), K) & \text { with   probability   1 - p }, \\ k ^ {\prime} = 1 & \text { with   probability   p }. \end{array} \right.\tag{8}
$$

We now optimize the same objective as in Eq. (7), but across any such waypoint we sample. To account for the multistep nature of this objective, we modify Eq. (7) below to accommodate such changes.

$$
e ^ {- d ((s, a), g)} \leftarrow \mathbb {E} _ {\{(s _ {t}, a _ {t}), s _ {t} ^ {w} \} \sim \mathcal {D}} [ \gamma^ {k ^ {\prime}} \cdot e ^ {- d (s _ {t} ^ {w}, g)} ].\tag{9}
$$

We denote this new objective as $\tau _ { \beta }$ , as the future state of the sample is restricted by trajectories generated by $\pi _ { \beta }$ . This is similar to the n-step returns in prior work (Sutton, 1988; Munos et al., 2016; Li et al., 2025), although we do not sample future states with a fixed number of steps. In practice, we can use any loss function to make the LHS equal to the RHS (in expectation) concerning Eq. (7) and Eq. (9). We use a form of Bregman divergence with LINEX losses (Parsian and Kirmani, 2002), as it does not incur vanishing gradients when the two distances have become close in value (Banerjee et al., 2004; Myers et al., 2025c):

$$
D _ {T} (d, d ^ {\prime}) \triangleq \exp (d - d ^ {\prime}) - d ^ {\prime}\tag{10}
$$

Using this loss, we can concretely define both ${ \mathcal { L } } _ { T _ { \beta } }$ and $\mathcal { L } _ { T }$ that can optimize the objectives of $\tau$ and $\tau _ { \beta }$

$$
\mathcal {L} _ {\mathcal {T} _ {\beta}} \big (\phi , \psi ; \{s _ {i}, a _ {i}, s _ {i} ^ {w}, g _ {i}, k _ {i} ^ {\prime} \} _ {i = 1} ^ {N} \big) = \sum_ {i = 1} ^ {N} \sum_ {j = 1} ^ {N} D _ {T} \big (d ((s _ {i}, a _ {i}), g _ {j}), d (s _ {i} ^ {w}, g _ {j})) - k _ {i} ^ {\prime} \log \gamma \big).\tag{11}
$$

These two losses, when applied to our quasimetric network, will allow us to propagate the value in either a onestep or multistep manner.

Remark: relationship between $p ,$ multistep backup $\tau _ { \beta }$ , and onestep backup $\tau .$ . By changing $p ,$ we can adjust the distance (in expectation) from the waypoint from the current state, as $p = 1$ means that $\tau _ { \beta }$ is equivalent to $\tau$ (as $\lceil k ^ { \prime } = 1 )$ . We demonstrate in Section 5.3 why only using $\mathcal { L } _ { T }$ is insufficient for learning a good distance representation and goal-conditioned policy.

## 4.2 LEARNING VALUE FUNCTIONS VIA ENFORCING ACTION INVARIANCE

With respect to quasimetric networks, (Myers et al., 2024) has demonstrated that if the training data is collected using a Markovian policy, then the optimal critic should observe the property of Action Invariance. As a result, the optimal critic should observe the following property (Sutton and Barto, 2018):

$$
V _ {g} ^ {*} (s) = \max _ {a \in \mathcal {A}} Q _ {g} ^ {*} (s, a).\tag{12}
$$

This can be satisfied, under our construction, if $d ( s , ( s , a ) ) = 0$ for each action $a \in { \mathcal { A } }$ (Myers et al., 2025c) (action invariance). Since our construction of Q and value function do not observe this property, we need to enforce it using gradient descent, as described in Eq. (13).

$$
d (\psi (s), \varphi (s, a)) \leftarrow 0\tag{13}
$$

We remark that this is also similar to the value loss seen in other offline RL methods that employ both a value and Q function, namely IQL (Kostrikov et al., 2022) and TMD (Myers et al., 2025c), and the desired outcome of Eq. (13) is similar to the value loss described in IQL when $\tau \approx 1$ . One common failure case of basic regressions, such as using $\mathcal { L } _ { 1 }$ or $\mathcal { L } _ { 2 }$ regression for $\operatorname { E q . }$ (13), is that the algorithm quickly converges to trivial solution in which $\varphi ( s , a ) = \psi ( s ) = 0$ , which fails to learn any valuable distance measures. As a result, TMD requires a hyperparameter ζ to regulate the magnitude of gradients of the invariance loss. To counteract this, we use a smoother formulation that allows more relaxed enforcement of Eq. (13) when the violation is low in magnitude. As a result, we can define $\mathcal { L } _ { \mathcal { I } }$ as:

$$
\mathcal {L} _ {\mathcal {I}} \big (\varphi , \psi ; \{s _ {i}, a _ {i} \} _ {i = 1} ^ {N} \big) = \sum_ {i, j = 1} ^ {N} \big (e ^ {- d (\psi (s _ {i}), \varphi (s _ {i}, a _ {j}))} - 1 \big) ^ {2}.\tag{14}
$$

Now the loss will scale with the magnitude of such deviation, which removes the need of hyperparameter tuning for an appropriate multiplier as well as stabilizing training dynamics.

## 4.3 POLICY EXTRACTION

We extract the goal-conditioned policy $\pi ( s , g ) : { \mathcal { S } } ^ { 2 } \to { \mathcal { A } }$ using the learned distance using the behavior-regularized deep-deterministic policy gradient (DDPG + BC) (Fujimoto and Gu, 2021):

$$
\mathcal {L} _ {\mu} (\pi ; \{s _ {i}, a _ {i}, g _ {i} \} _ {i = 1} ^ {N}) = \mathbb {E} \Big [ \sum_ {i, j = 1} ^ {N} d \big ((s _ {i}, \pi (s _ {i}, g _ {j})), g _ {j} \big) - \alpha \log \pi (a _ {i} \mid s _ {i}, g _ {i}) \Big ].\tag{15}
$$

Given that a smaller d measure correspond to a higher Q value, as defined in Eq. (5), we can maximize the Q values by minimizing the distance produced by our quasimetric network. We tune the BC coefficient α per environment. We provide more hyperparameter details in Section D.

## 4.4 IMPLEMENTATION DETAILS & ALGORITHM

We concisely define our final learning objective in Algorithm 1. Unlike previous works in hierarchical RL (Nachum et al., 2018; Park et al., 2024b), we randomly sample these waypoints, and we learn a single critic Q that operates on $s$ and A and a single goal-reaching policy $\pi _ { \mu } .$ . As a result, our method does not contain any additional components and is simpler to implement than these hierarchical methods.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1: Multistep Quasimetric Estimation
Require: Dataset D, Batch size B, training iteration T, Probability p
1: Initialize quasimetric network Q with parameters ( $\varphi, \psi$ ), goal-reaching policy  $\pi_{\mu}$ 
2: for  $t = 1...T$  do
3:    Sample  $\{s_i, a_i, s'_i, s_i^w, g_i\}_{i=1}^B \sim D$ 
4:    Update Q with multistep backup by minimizing  $\mathcal{L}_{\mathcal{T}_{\beta}}(\varphi, \psi; \{s_i, a_i, s_i^w, k_i'\}_{i=1}^B)$ 
5:    Update Q with action invariance constraints by minimizing  $\mathcal{L}_{\mathcal{I}}(\varphi, \psi; \{s_i, a_i\}_{i=1}^B)$ 
6:    Update policy  $\pi_{\mu}$  with DDPG+BC by minimizing  $\mathcal{L}_{\mu}(\pi_{\mu}; \{s_i, a_i, g_i\}_{i=1}^B)$ 
7: return  $\pi_{\mu}$
</div>

(8)

(9)

(15)

## 4.5 ANALYSIS

The key theoretical result we show is that this algorithm is able to perform policy improvement on top of the behavior policy $\pi _ { \beta }$ in a tabular setting under standard assumptions. The tabular version of MQE can be viewed as executing three steps:

1. Minimize Eqs. (11) and (14) under the data distribution from $\pi _ { \beta } \colon$

$$
\min _ {\hat {d} \in \mathcal {D} _ {S \times A \cup S}} \mathbb {E} _ {(s, a) \sim p ^ {\pi_ {\beta}}, g \sim p ^ {\pi_ {\beta}}} \Big [ D _ {T} \big (d ((s, a), g), \mathbb {E} _ {p ^ {\pi_ {\beta}} (s ^ {w} | s, a)} [ \gamma^ {k ^ {\prime}} e ^ {- d (s ^ {w}, g)} ] \big) + (e ^ {- d (s, (s, a))} - 1) ^ {2} \Big ].\tag{16}
$$

2. Constrain the distance to be a quasimetric. Mathematically, this can be expressed as projecting the distance into a quasimetric space via a path relaxation operator $\mathcal { P } : \mathcal { D } _ { \mathcal { X } } \overset { \cdot } {  } \mathcal { Q } _ { \mathcal { X } }$ (Myers et al., 2025a), defined as

$$
\mathcal {P} (d) (x, z) \triangleq \min _ {y \in \mathcal {X}} \bigl [ d (x, y) + d (y, z) \bigr ].\tag{17}
$$

Applying this to the learned distance, we construct $\tilde { d } = \mathcal { P } \hat { d } .$

3. Extract the policy via Eq. (15):

$$
\min _ {\pi} \mathbb {E} _ {(s, a) \sim p ^ {\pi_ {\beta}}, g \sim p ^ {\pi_ {\beta}}} \left[ \tilde {d} ((s, \pi (s, g)), g) \right].\tag{18}
$$

This statement is formalized in Theorem 1 below.

Theorem 1. Suppose behavior policy $\pi _ { \beta }$ induces full support over state action pairs $( s , a ) \sim d ^ { \pi _ { \beta } }$ in a tabular setting. We fit a distance by minimizing Eq. (16) and extract a policy by minimizing $E q .$ (18). Then the extracted policy π satisfies $V _ { g } ^ { \pi } ( s ) \geq \breve { V } _ { g } ^ { \pi _ { \beta } ^ { * } } ( s )$ for all $s , g \in S$

The proof is provided in Section C.

antmaze-colossal

visual-antmaze-giant

puzzle-4x4

![](images/45ad993fc107fed1d2d9480ccefe41ac0a58c7c95e97842d02b9e3f92887769d.jpg)

![](images/f967193a36391e53396170d07e98b2e3dde8e31e0aa29b9a168fd1fdd276ab54.jpg)

![](images/0e85f243f35e2936824474a425264d6cb171ef29be4f31c07e600a70edb8dbf3.jpg)

![](images/1929da6b3a9698afba6f1a5f2a7bcb98be660f58eb0d09c1a78911bc4a7d5f13.jpg)  
Figure 2: Tasks from various state and pixel-based environments for OGBench. Antmaze-colossal is 50% larger than any other mazes available on OGBench, and in stitch datasets, test the agent’s ability to generalize over horizon that is up to 1000% longer.

scene-play  
Aggregate OGBench Success Rates  
![](images/60a69a89ff6bce9042c92987a0e45855504a5e9390b59cb36f8b96a78a5340c9.jpg)  
(a) The overall success rate across a set of most challenging OGBench evaluation tasks (90 tasks in total). MQE achieves by far the best performance over these challenging tasks.

Horizon Generalization in AntMaze  
![](images/c2f2131e4755e9f990f5721a7f3b0a31536ec72e4dd03921002a12204140c848.jpg)  
(b) Horizon generalization on antmaze-\*-stitch dataset with training trajectories that are only 4m long. This means that colossal mazes have tasks that have horizons that are 1000% longer than those seen in the training set. MQE also retains the best horizon generalization skills compared to previous methods.

Figure 3: Comparisons of MQE against prior methods on OGBench.

## 5 EXPERIMENTS

Our goal of experiments is to understand the benefits MQE brings when it comes to enabling a policy to generalize compositionally (execute multiple tasks seen in training separately together) and in terms of horizon (generalize over a longer task when a similar but shorter task was seen in training set). To that end, we pose the following questions:

1. How much does MQE improve the horizon generalization abilities of agents?

2. What qualitative improvements does MQE bring in terms of compositional generalization?

3. What are the design decisions to ensure the success of MQE?

Experiment setup Our experiments use challenging, long-horizon problems in offline RL benchmarks as well as real-world settings. We use OGBench (Park et al., 2025a) for our experiments on simulated benchmarks and the BridgeData setup (Walke et al., 2024) for our real-world evaluation.

## 5.1 SIMULATED EVALUATION ON OFFLINE GOAL-REACHING TASKS

We evaluate MQE in both locomotion and manipulation in OGBench (Park et al., 2025a). For locomotion tasks, in addition to the three standard sized mazes, we also designed a “colossal”-sized

observation

maze. This maze is 50% larger than that of the “giant”-sized mazes currently available on OGBench, and it requires as many as 4000 steps for an agent to traverse through the entire maze (see Fig. 2). We employ 13 state-based environments and 5 pixel-based environments (each pixel-based environment takes in a $6 4 \times 6 4 \times 3$ observation) with 5 tasks each, bringing a total of 95 tasks to evaluate in our OGBench setup.

We compare against the following baselines: GCIQL (Kostrikov et al., 2022), CRL (Eysenbach et al., 2022), QRL (Wang et al., 2023), HIQL (Park et al., 2024b), nSAC+BC (Park et al., 2025b; Haarnoja et al., 2018), CMD (Myers et al., 2024), and TMD (Myers et al., 2025c). These methods use either only TD learning (GCIQL, QRL, nSAC+BC), MC value estimation via contrastive learning (CRL, CMD, TMD), use horizon reduction techniques (nSAC+BC with value horizon, HIQL with policy horizon) or use a quasimetric architecture for distance learning (QRL, CMD, TMD). We detail how these methods are implemented in Section D. By comparing against these methods, we can gain a better outlook on what advantage MQE has over other works that learn distances only locally, globally, or in a hierarchical manner.

Table 4 and Fig. 3 show the performance of MQE across state- and pixel-based environments on OGBench. In general, MQE exhibits considerably better capabilities of extracting goal-reaching poli cies, and in some instances (such as humanoidmaze-giant-stitch, exhibits a 10× improvement over the previous best methods, including HIQL and n-SAC+BC, which does explicit policy horizon reduction. The only exception to this is vision-based manipulation, where HIQL performed better. We also compared the performance of MQE against TMD, CRL, and HIQL on antmaze of various sizes, with the training set fixed to only 4 meters long via the usage of stitch datasets. We show that as the evaluation horizon becomes longer, MQE still exhibits strong horizon generalization, and it is the only method that still exhibits nonzero success rates in colossal mazes.

## 5.2 REAL-WORLD EVALUATION OF MQE

While OGBench environments focus on learning long-horizon tasks using mixed quality data in a single environment, we can use real-world BridgeData tasks to evaluate a more sophisticated kind of compositionality. BridgeData tasks consist of individual object manipulation primitives (e.g. picking up a banana and placing it on a plate), and our evaluation tasks are significantly longer (up to 4x), requiring the composition of multiple tasks in the dataset (e.g. placing four different objects on the plate) (Walke et al., 2024). The dataset does not contain any example trajectories that compose multiple tasks in this way. Accomplishing this sort of temporal composition is an important goal in offline RL, because it allows “stitching” long behaviors out of shorter chunks.

![](images/0f00cd648bd607d1661fb84e9d332e95e18a86f92965ef4193f520462d7f8c4d.jpg)

Quadruple Pick and Place  
![](images/a1cbe20d2aad6ab289fa91466e3e78f6a5009f87008b78479063fc5f0a7535d6.jpg)  
Figure 4: We evaluate MQE on multi-stage manipulation tasks on BridgeData. Below are examples of the starting observations and goal images.

We designed the tasks on BridgeData to test whether policies can compose multiple tasks at once without external guidance. Instead of instructing the policy to complete a single pick

and place (abbreviated as PnP), we evaluate a policy’s performance with PnP of up to 4 objects in sequence. To our knowledge, such a task in BridgeData has never been completed without the use of hierarchical policies or high-level planners. We also evaluate the policy on tasks requiring dependencies (i.e. the second task can only succeed when the first task succeeds), with the policy being tasked with opening a drawer and then placing the item within the drawer all conditioned by one image of an opened drawer with an item inside. Only one previous work (Myers et al., 2025b) has shown consistent success when using an end-to-end policy, noting the challenges involved with this type of policy. Examples of the tasks are shown in Fig. 4.

Table 1: Ablation results  
(a) $\mathcal { L } _ { \mathcal { I } }$ Ablation

<table><tr><td>Configuration</td><td>Success Rate (%)</td></tr><tr><td>With  $\mathcal{L}_{\mathcal{I}}$ </td><td> $26.5^{(\pm 1.3)}$ </td></tr><tr><td>Without  $\mathcal{L}_{\mathcal{I}}$ </td><td> $7.9^{(\pm 0.7)}$ </td></tr><tr><td>Using expectile  $\kappa = 0.7$ </td><td> $11.3^{(\pm 1.1)}$ </td></tr><tr><td>Using expectile  $\kappa = 0.9$ </td><td> $8.8^{(\pm 0.7)}$ </td></tr></table>

(b) s<sup>w</sup><sub>t</sub> Sampling Ablation

<table><tr><td>Configuration</td><td>Success Rate (%)</td></tr><tr><td> $k' \sim Eq. (8)$ </td><td> $26.5^{(\pm 1.3)}$ </td></tr><tr><td> $k' \sim Geom(1 - \lambda)$ </td><td> $22.1^{(\pm 1.1)}$ </td></tr><tr><td> $k' \sim Unif[1, K]$ </td><td> $18.9^{(\pm 0.9)}$ </td></tr><tr><td> $k' \sim Unif[1, 50]$ </td><td> $17.8^{(\pm 1.3)}$ </td></tr><tr><td> $k' = 50$ </td><td> $1.7^{(\pm 0.5)}$ </td></tr><tr><td> $k' = 1$ </td><td> $0^{(\pm 0.0)}$ </td></tr></table>

We use a 6DoF, 5Hz, WidowX250 manipulator for our robot learning tasks and we train and deploy a policy $\pi ( \boldsymbol { a } \mid s , g )$ conditioned on observations and goal images. We compare against the following methods: GCBC (Ding et al., 2020), GCIQL, and TRA (Myers et al., 2025b). We compare MQE against GCIQL, a commonly used offline RL method, and we compare MQE against both GCBC (Ding et al., 2020) and TRA (Myers et al., 2025b). TRA is designed for following both goal images and language instructions, but we only use goal images as the modality to test. We provide more details on policy training in Section E.1.

As in prior work that focused on long-horizon manipulation tasks (Black et al., 2024; Shi et al., 2025), we use task progress to measure the effectiveness of these policies due to the long-horizon nature of these tasks. We detail more on the experimental setup and how we assign progress in Section E.1.

Fig. 5 reports the overall task progress on 2 single-stage tasks and 4 tasks that require compositionality, and Table 6 reports the binary success rate of each task. We provide further analysis of policy rollout in Section E. Here, we observe that while MQE helps with single-stage tasks (single PnP, open drawer) against GCBC, both TRA and GCIQL can still perform competitively. However, as the number

![](images/abf9f46717818a6c03016ba32b9443927fa6c7a4c74102c38ee3b85d321df832.jpg)  
Figure 5: Task progress on BridgeData tasks based on how many consecutive tasks the agent is required to perform, plotted with both the mean and the standard error bars.

of tasks needed to be performed in sequence increases, we see that MQE is able to retain a relatively high task progress, while both GCIQL and TRA’s performance regressed.

Taking a look at the two most difficult tasks, we have quadruple PnP and drawer open and place. These tasks are the most challenging since quadruple PnP required the agent to reason 4 consecutive primitives together, and drawer open and place required the agent to complete the first task (open the drawer) before completing the second task (putting the mushroom in the drawer). Among all methods that we have tested, only MQE and TRA displayed positive success rate, as demonstrated in Table 6. We provide more details on policy rollouts in Section E.

## 5.3 ABLATION STUDIES

In this section, we explore the design choices involved for MQE. To that end, we investigate the heuristics needed for MQE. We use the humanoidmaze-giant-stitch environment and dataset, and explore the following design questions:

• How do different distributions for sampling the waypoint $s _ { t } ^ { w }$ affect MQE’s success?

• Is the objective of action invariance I necessary in MQE?

• How do λ and p, the two hyperparameters that affect the geometric sampling of $s _ { t } ^ { w }$ , change MQE’s performance?

Should we use action invariance for value learning? Table 1a shows that when using the same set of hyperparameters, imposing action invariance as an explicit term helps to learn a much better critic, as compared to only using $\bar { \mathcal { L } } _ { \mathcal { T } _ { \beta } }$ . Additionally, we also implemented value learning via expectile loss, described in (Kostrikov et al., 2022). While expectile loss performed well as a value learning objective, it performed considerably worse than action invariance, which validates the theoretical results shown in Section C.

![](images/5498f53253c996590c659c907695ddf0b5c401ea6394c54f456ba66412fb9967.jpg)  
Figure 6: Sucess rate of MQE on humanoidmaze\_giant\_stitch using $\alpha = 0 . 0 1$ , averaged over 4 seeds.

How should we sample our waypoints? We first evaluate the best distribution to sample future waypoints. Table 1b shows that when using a geometric distribution, we achieved much better performance than using a uniform distribution or using a fixed interval. We believe that since the goals are sampled via a geometric distribution, match ing the waypoint with another geometric distribution helps the network to generalize better between the two separate observations.

We then investigate how to combine the hyperparameters λ and p for best performance. Fig. 6 provides an illustration of success rate over pairs of $( p , \lambda )$ . The figure suggests that both hyperparameters need to be relatively high in value. This indicates that MQE needs: (1) a waypoint far enough for the value to rapidly propagate and (2) a high enough p to ensure that local consistencies are being respected. We also see that if we increase the value of the sampling coefficient p, MQE cannot learn a good policy for goal-reaching. This shows that T is not sufficient to learn a good distance for policy learning, as the agent did not learn a good distance representation.

## 6 CONCLUSION

We introduced Multistep Quasimetric Estimation (MQE), a novel method that combines the benefits of fast value propagation via multistep backup and the global constraint of quasimetric distances. MQE is able to solve challenging and long-horizon tasks in simulated benchmarks and on a real-world robotic manipulator.

Limitations and Future Work. While MQE achieves strong performance, we sample the waypoints based on heuristics. This could incur more computation costs when finding the optimal way of sampling the waypoint for environments that are outside of our evaluation range. Future work can investigate the theoretical connection between sampling waypoints and successor distances, investigate the effect of such policy learning on different policy classes such as action-chunking policies, and apply the same method across methods beyond offline RL in scenarios such as offline-to-online RL or online RL.

Acknowledgements We would like to thank Seohong Park, Qiyang Li, Michael Psenka, and Kwanyoung Park for helpful discussions. This material is based upon work supported by the National Science Foundation under Award No. 2441665 and ONR N00014-25-1-2060. Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the NSF and ONR.