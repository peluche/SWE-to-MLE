---
title: "Makemore Implement a Bigram Character-level Language Model"
date: 2023-09-09T21:14:06+02:00
draft: false
author: "peluche"
authorLink: "https://github.com/peluche"
description: "Makemore Implement a Bigram Character-level Language Model"
tags: ['NLP', 'LM', 'backpropagation', 'gradient', 'neuron', 'python', 'pytorch', 'jupyter', 'Andrej Karpathy']
categories: ['learning']
---

Let's look at episode #2: [The spelled-out intro to language modeling: building makemore](https://youtu.be/PaCmpygFfXo?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) from [Andrej Karpathy](https://karpathy.ai/) amazing tutorial series.

{{< youtube PaCmpygFfXo >}}

It covers an intro to Language Model using a very barebone from scratch approch using a Bigram Character-level Language Model. It means: "given a single character, guess the next character". For this session the NN is trained on a list of names to produce new unique name-sounding words.

The lecture goes from calculating the probabilities of each letters by hand, to automatically generating the probablilities as the set of weight of a very simple one layer NN that produce the exact same results.

The video is a treat from start to finish. To highlight one specific point, Andrej goes off on a tangent about the importance of [understanding tensor broadcasting](https://pytorch.org/docs/stable/notes/broadcasting.html) and how easy it is to shoot yourself in the foot otherwise.

The basics rules of broadcasting go as follow:
- align the dimensions to the right
- the dimensions must be equal
- or one of them must be 1
- or one of them must not exist

Here's my take on the tutorial with additional notes. You can get the code on [GitHub](https://github.com/peluche/makemore) or bellow.

{{< gist peluche ab934dc35d3768b385ae6f5ba63cf3a8 >}}
