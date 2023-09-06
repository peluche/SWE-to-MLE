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

## Argmax
Set the largest value to `1` and everything else to `0`. This is convenient to make a decision, but has a terrible derivative for backpropagation (see. [softmax](#softmax)).

## BLEU
Metric for machine translation based on (translated vs reference) N-gram comparison.

## Cross Entropy 
$CE(X) = -\sum_{X=1}^{M}{observed_{X} \times log(predicted_{X})}$ but in practice observed is a vector of [0]s with a single [1], so it simplify to $CE(X) = -log(predicted_{X})$. It's used to compute the loss of a [softmax](#softmax) (because it provides steeper gradient in the `[0, 1]` range than the [squared residual](#squared-residual)).

## Entropy
$Entropy = \sum_{i} { Probability_i . Surprise_i} = \sum {p(x)log({1 \over p(x)})} = - \sum {p(x)log(p(x))}$. See [Surprise](#surprise). See [yt](https://youtu.be/YtebGVx-Fxw).

## Geometric Mean
$(\prod_i^n x_i)^{1 \over n} = \sqrt[n]{\prod_i^n x_i}$. Used to compute the mean change of a multiplicative sequence (e.g. given 100€, year 1 gets 10% increase, year 2 gets 10% decrease, now the toral is 99€ because $\sqrt[2]{1.1 * 0.9} = 0.995$ and $100€ * 0.995 * 0.995 = 0.99$ the mean interrest rate was a decrease of 0.5%). [yt](https://youtu.be/bEUbfBlZDmo).

## Harmonic Mean
${n \over \sum {1 \over x}} = ({\sum{1 \over x} \over n})^{-1}$. Used when dealing with rates (e.g. speed (aka. km/h)). E.g. drive half distance at 30km/h then half at 50km/h then my average speed was $({{1 \over 30} + {1 \over 50} \over 2})^{-1} = 37.5$. [yt](https://youtu.be/jXKYI7wyqp0).

## Logit
LOGistic unIT. Some raw value converted into a proba (see [softmax](#softmax)). E.g. an image classifier output layer is a vector of logit

## NER
Named entity recognition -
Mark each word in a sentence as corresponding to a particular entity (such as persons, locations, or organizations, etc.) or "no entity".

## Perplexity
$PPL(X) = P(X_1 X_2 ... X_N)^{-{1 \over N}}$ metric (kinda bad one) to measure the performance of language model (the lower the value, the better the perf). Perplexity is analog to the branching factor normalized by the length of the sequence. Say we guess a `N` digits number, all digits have equal probability, $PPL(X) = P(X_1 X_2 ... X_N)^{-{1 \over N}} = ({1 \over 10} . {1 \over 10} ...)^{-{1 \over N}} = {({1 \over 10})^{N}}^{-{1 \over N}} = 10$. Perplexity ~= Branching Factor = 10 (digits from 0 to 9).

## POS
Part of speech - Mark each word in a sentence as corresponding to a particular part of speech (such as noun, verb, adjective, etc.).

## ROUGE
Metric for text summarization based on (summerized vs reference) N-grams comparison.

## Softmax
$softmax(x) = {e^x \over {\sum_i e^i}}$. It normalize raw values into [logit](#logit) (aka. Probabilities summing to 1). The derivative simplify to ${d_{softmax_p} \over d_{raw_p}} = softmax_p . (1 - softmax_p)$. [yt](https://youtu.be/KpKog-L9veg).

## Squared Residual
$SR = (1 - p)^2$ loss function used for backpropagation. Works best on `[1, inf]`, for `[0, 1]` see [cross entropy](#cross-entropy).

## Surprise
$Surprise = log({1 \over Probability}) = log(1) - log(Probability)$ Note: the surprise of something with probability 0 is undefined (division by zero, or log(0)).