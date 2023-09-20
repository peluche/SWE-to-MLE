---
title: "Makemore4 Becoming a Backprop Ninja"
date: 2023-09-20T17:12:56+02:00
draft: false
author: "peluche"
authorLink: "https://github.com/peluche"
description: "Makemore4 Becoming a Backprop Ninja"
tags: ['MLP', 'python', 'pytorch', 'gradient', 'backpropagation', 'derivative', 'Andrej Karpathy']
categories: ['learning']
---

A look at episode #5: [The spelled-out intro to language modeling: Building makemore Part 4: Becoming a Backprop Ninja](https://youtu.be/q8SA3rM6ckI?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) from [Andrej Karpathy](https://karpathy.ai/) amazing tutorial series.

{{< youtube q8SA3rM6ckI >}}

We go back to the previous N-gram character-level MLP model from session #4 and dive into a hands-on manual backpropagation session.

## Computing Gradients by Hand
This lesson is a bit of a different format. It's a lot more exercise centric and less of a type-along lecture. The tasks feel a bit repetitives after a while but it gives a good sense of what micrograd/PyTorch are doing in the background for us.

Andrej intentionnaly cut the code into very minimal atomic operation to make the job of differentiating easier. And later refactor some of the many operations into single higher level blocks (e.g. `cross_entropy` goes from a block of 8 gradients to a one-liner).

## Sticking point
To me the least intuitive line is around converting the minibatch entries into vector space embeddings `C[Xb]` and it leads to nested loops code that sticks out from the rest of the code.

```python
# <???> this is the least intuitive line to me, the `C[Xb]`
# with C.shape = [27, 10], Xb.shape = [32, 3], and C[Xb].shape = [32, 3, 10]
# component: emb = C[Xb]
dC = torch.zeros_like(C)
for k in range(Xb.shape[0]):
    for j in range(Xb.shape[1]):
        ix = Xb[k, j]
        dC[ix] += demb[k, j]
cmp('C', dC, C)
```

I postulate that reformulating `C[Xb]` as a matrix dot product instead `F.one_hot(Xb).float() @ C` leads to a more elegant solution.

```python
# Hypothesis: I think it would be easier to follow if `C[Xb]` was rewritten as
# matrix dot product instead C[Xb] = F.one_hot(Xb).float() @ C
# with emb.shape = [32, 3, 10], F.one_hot(Xb).float() = [32, 3, 27], C = [27, 10]
# so instead we get:
# component: emb = F.one_hot(Xb).float() @ C
dC = torch.tensordot(F.one_hot(Xb, num_classes=vocab_size).float(), demb, dims=([0, 1], [0, 1]))
cmp('C', dC, C)
```

Computing the gradient become a 3D tensor dot product and we avoid manual loops totally.

## The code

Here's my take on the tutorial with additional notes. You can get the code on [GitHub](https://github.com/peluche/makemore) or bellow.

{{< gist peluche af17c77f439b7a0e77ffc492cb67eb5e >}}