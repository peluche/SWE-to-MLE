---
title: "Grimoire"
date: 2023-09-05T11:37:46+02:00
draft: false
author: "peluche"
description: "Grimoire"
comment:
  enable: false
math:
  enable: true
---

## Alignment
Is related to AGI. It's about "building a __nice__ AI". Quoting George Hotz it's also about making 
a "do what I __mean__ machine" as opposed to a what I said machine. When discussed people tend to use `Utility function` as our model for influencing AI's behavior. It's a very hard problem, currently unsolved problem, possibly unsolvable problem? This is a subjet causing a lot of anguish. How to avoid the `Roko's basilisk` and the `paperclip maximizer` doom. Anecdotally, humans are not aligned (within natural selection) with our "inclusive genetic fitness" because we use contraceptives which goes against what "natural selection would want us to do". The most vocal advocate for alignment over the years has been Eliezer Yudkowsky [[yt]](https://youtu.be/EUjc1WuyPT8).

## Argmax
Set the largest value to `1` and everything else to `0`. This is convenient to make a decision, but has a terrible derivative for backpropagation (see. [[softmax]](#softmax)).

## Bayes Theorem
$P(H|E) = \frac{P(H) . P(E|H)}{P(E)} = \frac{P(H) . P(E|H)}{P(H).P(E|H) + P(\lnot H).P(E|\lnot H)}$ with $P(H)$ proba hypothesis is true, $P(E|H)$ proba of seeing the evidence if the hypothesis is true, $P(E)$ proba of seeing the evidence, $P(H|E)$ proba a hypothesis is true given some evidence. e.g. the chance someone shy is a librarian instead of a farmer is: $\frac{\\#shy\\_ librarians}{\\#shy\\_ librarians + \\#shy\\_ farmers}$. See [[yt]](https://youtu.be/HZGCoVF3YvM).

## Binomial Distribution
$pre(x|n,p) = (\frac{n!}{x!(n-x)!})p^x(1-p)^{n-x}$ where $(\frac{n!}{x!(n-x)!})$ is "n choose k" (nCk). See [[yt]](https://youtu.be/J8jNoF-K8E8) [[yt]](https://youtu.be/8idr1WZ1A7Q).

## BLEU
Metric for machine translation based on (translated vs reference) N-gram comparison.

## CoT
Chain of Thought. A way to solve a problem by breaking it down into a sequence of reasoning steps.

## Cross Entropy 
$CE(X) = -\sum_{X=1}^{M}{observed_{X} \times log(predicted_{X})}$ but in practice observed is a vector of [0]s with a single [1], so it simplify to $CE(X) = -log(predicted_{X})$. It's used to compute the loss of a [[softmax]](#softmax) (because it provides steeper gradient in the `[0, 1]` range than the [[squared residual]](#squared-residual)).

## Curse of Dimensionality
(Told ya the Grimoire would have curse and hexes in it!). It means that algorithms working for small examples become impractical on real world examples because they grow exponentially with input size. In CS it often means that runtime become unmanageable. In ML it sometimes also means that we would need exponential amount of data fot the model to learn.

## e/acc
Effective accelerationism is a (satirical?) movement on the opposite side of the [[alignment]](#alignment) doomer crowd. It argues for a set of several super human AGI forming a stalemate equilibrium that would let human reap benefits of AI while not being endengered by it. [[yt]](https://youtu.be/4xvvenRLtY0).

## Eigenvector / Eigenvalue
Eigenvector is a vector that does not change direction for a given transformation (change of basis). The Eigenvalue is the magnitude of the vector after the transformation (e.g. 1 unchanged, 2 doubled in size, -1 reversed direction). Note: for a given change of basis there can only be at most `matrix rank` Eigenvectors so a 2x2 matrix can have 0, 1, or 2 Eigenvectors.

## EMA
Exponential Moving Average. A streaming way to compute the moving average of a sequence of numbers by giving more weight to recent values. $EMA(t) = discount * val_t + (1 - discount) * EMA(t-1)$.

## Entropy
$Entropy = \sum_{i} { Probability_i . Surprise_i} = \sum {p(x)log({1 \over p(x)})} = - \sum {p(x)log(p(x))}$. See [[Surprise]](#surprise) [[yt]](https://youtu.be/YtebGVx-Fxw) [[yt]](https://youtu.be/ErfnhcEV1O8?t=144)

## Geometric Mean
$(\prod_i^n x_i)^{1 \over n} = \sqrt[n]{\prod_i^n x_i}$. Used to compute the mean change of a multiplicative sequence (e.g. given 100€, year 1 gets 10% increase, year 2 gets 10% decrease, now the toral is 99€ because $\sqrt[2]{1.1 * 0.9} = 0.995$ and $100€ * 0.995 * 0.995 = 0.99$ the mean interrest rate was a decrease of 0.5%). [[yt]](https://youtu.be/bEUbfBlZDmo).

## Harmonic Mean
${n \over \sum {1 \over x}} = ({\sum{1 \over x} \over n})^{-1}$. Used when dealing with rates (e.g. speed (aka. km/h)). E.g. drive half distance at 30km/h then half at 50km/h then my average speed was $({{1 \over 30} + {1 \over 50} \over 2})^{-1} = 37.5$. [[yt]](https://youtu.be/jXKYI7wyqp0).

## KL divergence
$D_{KL}(P || Q) = \sum{P(x) . log(\frac{P(x)}{Q(x)})}$. KL divergence is a mesure of similarity between 2 distributions. It is not symetrical $D_{KL}(P || Q) \neq D_{KL}(Q || P)$. It's asking how surprise are we to observe a distribution if we think it should be a different one. Or how suboptimal an encoding scheme designed to encode $Q$ would be at encoding $P$ (in that sense it relates to [[entropy]](#entropy)).

## L1 Regularization
Loss penalty term based on the sum of the absolute value of the weights $\sum |weight|$.

## L2 Regularization
Loss penalty term based on the sum of que squared value of the weights $\sum weight^2$

## Lasso Regression
See [[L1 regularization]](#l2-regularization).

## Logit
LOGistic unIT. Some raw value converted into a proba (see [[softmax]](#softmax)). E.g. an image classifier output layer is a vector of logit

## LoRA
Low Rank Adaptation (LoRA) is a [[PEFT]](#peft). It's used to fine tune a large model while only touching a minimal amount of weights. In practice it emulates a `[N, M]` matrice, by using two `[N, k] @ [k, M]` matricies as a proxy. The insight is that matrices information is sparse and similar results can be used with fewer dimentions, it's a form of dimention reduction.

## MI
Mechanistic Interpretability (MI) is reverse engineering NN. Understand how AI works internally. How does it "think". Probing individual neurons. How does it compute features from earlier features.

## MOE
Mixture Of Experts (MOE) is a mechanism for dynamically rooting (aka. gating mechanism) queries of different types to different experts. Only a subset of the expert models is active at any given time, which reduces the overall memory usage because the entire set of models isn't loaded or running simultaneously.

## NER
Named entity recognition -
Mark each word in a sentence as corresponding to a particular entity (such as persons, locations, or organizations, etc.) or "no entity".

## P-value
$\text{p-value} = P(event) \sum{P({events\\_as\\_likely})} + \sum{P(events\\_less\\_likely)}$. The p-value is used to mesure if something is special / out of the ordinary, a common threshold is 0.05 (aka. 5%). Another way to think, the p-value is used for "Hypothesis testing" (aka. how confident are we that 2 things are different) to test the "Null hypothesis" (aka. 2 things are the same) a p-value of 0 == totally different, 1 == exacly the same. See [[yt]](https://youtu.be/vemZtEM63GY) [[yt]](https://youtu.be/JQc3yx0-Q9E).

## PEFT
Parameter Efficient Fine Tuning (PEFT) is an umbrella term for methods of fine tuning preventing to touch the entirety of the model weights (e.g. [[LoRA]](#lora)).

## Perplexity
$PPL(X) = P(X_1 X_2 ... X_N)^{-{1 \over N}}$ metric (kinda bad one) to measure the performance of language model (the lower the value, the better the perf). Perplexity is analog to the branching factor normalized by the length of the sequence. Say we guess a `N` digits number, all digits have equal probability, $PPL(X) = P(X_1 X_2 ... X_N)^{-{1 \over N}} = ({1 \over 10} . {1 \over 10} ...)^{-{1 \over N}} = {({1 \over 10})^{N}}^{-{1 \over N}} = 10$. Perplexity ~= Branching Factor = 10 (digits from 0 to 9).

## Poisson Distribution
$P(X=x) = \frac{\mu^{x} e^{-\mu}}{x!}$. It's used to count how many event happen in an interval (of time / space ...) (e.g. how many radioactive decay in 2sec). To be a poisson distribution events must be random and independant (aka. probability doens't change over time, and previous events do not influence future events). It is define on `[0, inf]` (because you can't have negative interval of time / space).

## POS
Part of speech - Mark each word in a sentence as corresponding to a particular part of speech (such as noun, verb, adjective, etc.).

## RAG
Retrieval-Augmented Generation (RAG). Add a content store (live internet query / some static DB / ...) and let the LLM query the store as a set of references (and concatenate them to the prompt). It allows to add knowledge without retraining the model, and help with providing sources.

## Ridge Regression
See [[L2 regularization]](#l2-regularization).

## RLHF
Reinforcement Learning from Human Feedback (RLHF). (1) Generate some outputs. (2) Humans label (score) them. (3) Train a second model on these labels. (4) Use this new model as scoring for RL. Intuition: The LLM learn the distribution of outcome, and RL focus the answer on specific groups of proba (narrowing the creativity/diversity by enforcing specific outcomes. aka. biaising the model toward human preference). [[yt]](https://youtu.be/PBH2nImUM5c).

## ROUGE
Metric for text summarization based on (summerized vs reference) N-grams comparison.

## Softmax
$softmax(x) = {e^x \over {\sum_i e^i}}$. It normalize raw values into [[logit]](#logit) (aka. Probabilities summing to 1). The derivative simplify to ${d_{softmax_p} \over d_{raw_p}} = softmax_p . (1 - softmax_p)$. [[yt]](https://youtu.be/KpKog-L9veg).

## Squared Residual
$SR = (1 - p)^2$ loss function used for backpropagation. Works best on `[1, inf]`, for `[0, 1]` see [[cross entropy]](#cross-entropy).

## STE
Straight-Through Estimator. A way to pass gradients through a non-differentiable function (often used with Quantization).

## Surprise
$Surprise = log({1 \over Probability}) = log(1) - log(Probability)$ Note: the surprise of something with probability 0 is undefined (division by zero, or log(0)).


---
