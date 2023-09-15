---
title: "Makemore3 Internals of MLP and Visualization"
date: 2023-09-15T21:58:58+02:00
draft: false
author: "peluche"
authorLink: "https://github.com/peluche"
description: "Makemore3 Internals of MLP and Visualization"
tags: ['NLP', 'LM', 'MLP', 'python', 'pytorch', 'initialization', 'non-linearity pitfall', 'batch normalization', 'Andrej Karpathy']
categories: ['learning']
---

A look at episode #4: [The spelled-out intro to language modeling: Building makemore Part 3: Activations & Gradients, BatchNorm](https://youtu.be/P6sfmUTpUmc?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) from [Andrej Karpathy](https://karpathy.ai/) amazing tutorial series.

{{< youtube P6sfmUTpUmc >}}

It re-uses the N-gram character-level MLP from session #3 and discuss three kind of incremental improvements to training

## Initial weights
While the model was training even with totally random weights this episode gives an intuition of why normally distributed values lead to faster training. Assigning clever weight at initialization time improve the loss of the first batches from `27` to `3.3` in our case. We waste less cycles reducing the weights of the net, and spend more cycles actually training.

## Tanh (and other sigmoid, ReLU, ...) are destroying your backprop gradients
Most non linear layers tend to regularize values inside a range (e.g. `tanh()` clip values between `[-1, 1]`). Because of that it's very dangerous to feed large (negative / positive) values to such layer.

The danger appear during backpropagation because the derivative of `tanh()` at `-1` and `1` is `0`. So the backpropagation gradients gets nullified. Intuitively this happen because `tanh(500) ~= 1.0` and `tanh(50) ~= 1`. Why bother changing the value from `500` to `50` if it will output `1.0` regardless and it's not going to improve the loss.

### Dead neuron
An interesting titbit here. If every output of a `tanh()` become close to `1` (e.g. `neuron.out.min() > 0.99`) then all the derivatives feeding into the neuron become `0` and there is no way to train this neuron anymore because all gradients comming to it are no-op. Effectively making it "dead".

## Batch normalization
Batch normalization introduced in [BatchNorm paper](https://arxiv.org/abs/1502.03167) aims to prevent the non-linear-layers-being-mean-to-gradients™ behaviors by matching the values in the input layer of the `tanh()` layer with a Gaussian distribution.

This creates other problems (we now need to feed information about the distribution of our dataset) to run the network instead of a single element. But it also accidentally solves other problems (by making it harder to overfit the dataset).

Using batch normalization seems to be a cause of great frustration for the community, and Andrej suggests looking into "group normalization" and "layer normalization" instead.

## Torchify the code and visualize the training
Finally the code gets re-written in PyTorch style and used to visualize how the weights of the neural net and training gradients behave based on the magnitude of the NN values. And why you should use some normalization to make your life easier.

## The code

Here's my take on the tutorial with additional notes. You can get the code on [GitHub](https://github.com/peluche/makemore) or bellow.

{{< gist peluche 2d1c539ab4937a534dc588e0b2aec059 >}}