---
title: "Quantization a Wizard's Treaty on Bag of Holding Construction"
date: 2024-08-19T20:02:19+02:00
draft: false
author: "peluche"
authorLink: "https://github.com/peluche"
description: "Quantization a Wizard's Treaty on Bag of Holding Construction"
images:
- "/posts/quantization-a-wizards-treaty-on-bag-of-holding-construction/bag-of-holding.png"
tags: ['quantization', 'symmetric quantization', 'asymmetric quantization', 'linear quantization', 'per-tensor quantization', 'triton', 'kernel']
categories: ['bestiary']
resources:
- name: "bag-of-holding"
  src: "bag-of-holding.png"
- name: "symmetric-quantization"
  src: "symmetric-quantization.svg"
- name: "asymmetric-quantization"
  src: "asymmetric-quantization.svg"
- name: "symmetric-matmul"
  src: "symmetric-matmul.svg"
- name: "asymmetric-matmul"
  src: "asymmetric-matmul.svg"
math:
  enable: true
---

*In the hallowed halls of the Arcanum, apprentice quantizers gather around ancient scrolls, their eager hands weaving complex patterns of magic, each fold compressing vast models into the nebulous depths of the Bag of Holding. As they manipulate these arcane energies, the fabric of reality thins, threatening to fray at the edges of their understanding. Those who delve too recklessly into its powers may find their meticulously crafted models reduced to incomprehensible noise, lost in the echoing void of the bag's mysterious expanse.*

![bag-of-holding](bag-of-holding.png "Bag of Holding")

## The Quest
Make the fifth circle model runnable by a low level machine by squeezing its essence into a smaller form factor.

## Why Quantization?
Quantization reduces the size of a model by decreasing the number of bits used to represent its weights. This reduction has two main benefits:
- It alleviates the memory bandwidth bottleneck by reducing the amount of data transferred between memory and compute units.
- It decreases the amount of computation needed. Fewer bits require fewer transistors, and simpler types like integers require fewer transistors than more complex types like floats.

## Quantize and Unquantize
For all upcoming examples, we will use fp32 as the original precision, and int8 as the quantized precision. This gives us a 4x reduction in size, and at the time of writing int8 is the smallest type supported by torch and triton.

And we'll focus on:
- **Linear Quantization**: quantized values are evenly spaced.
- **Per-tensor Quantization**: each tensor is quantized independently, and each element for a given tensor is quantized with the same scale.

### Symmetric Quantization
Let's start with the simplest case, symmetric quantization. In order to convert the original range of values to the quantized one, we find the biggest absolute value and define the original range as `[-max_abs, max_abs]` then map it to the quantized range `[-127, 127]` (using restricted range here for simplicity so we ignore -128).

$scale = \frac{max\\_abs}{q\\_max}$

{{< figure src="symmetric-quantization.svg" alt="Symmetric Quantization" >}}

Note: The piece of the range highlighted in orange is wasted, because we don't have any values in the `[-max_abs, min[` range for this example.

```python
def quantize(weights, bits=8):
    assert bits <= 8 # keep my life simple
    maxi = weights.abs().max()
    q_maxi = 2 ** (bits - 1) - 1
    scale = maxi / q_maxi
    quantized_weights = t.clamp(t.round(weights / scale), -q_maxi, q_maxi).to(t.int8)
    return quantized_weights, scale

def unquantize(quantized_weights, scale):
    return quantized_weights * scale
```

### Asymmetric Quantization
To prevent wasting a piece of the range we can use asymmetric quantization. In this case we define the original range as `[min, max]` and map it to the quantized range `[0, 255]`. In exchange for representing the full range, we have to introduce a zero point to remember the offset during unquantization.

$scale = \frac{max - min}{q\\_max}$

$zero\\_point = \left\lfloor\frac{-min}{scale}\right\rceil$

{{< figure src="asymmetric-quantization.svg" alt="Asymmetric Quantization" >}}

Note: The scale is a floating point number, but the zero point is rounded to the nearest integer.

```python
def quantize(weights, bits=8):
    ''' using the min-max strategy, this is vulnerable to outliers '''
    assert bits <= 8 # keep my life simple
    maxi = weights.max()
    mini = weights.min()
    qmaxi = 2 ** bits - 1
    scale = (maxi - mini) / qmaxi
    zero = int(t.round(-mini / scale))
    quantized_weights = t.clamp(t.round(weights / scale) + zero, 0, qmaxi).to(t.uint8)
    return quantized_weights, scale, zero

def unquantize(quantized_weights, scale, zero):
    quantized_weights = quantized_weights.to(t.int32)
    return (quantized_weights - zero) * scale
```

## Matmul
We need a few more ingredients. We need a way to do a matmul between two int8 tensors, and we need a way to unquantize the result.

### Triton Kernel
For the matmul, we'll use a Triton kernel. Triton is a DSL for writing GPU kernels. It's similar to CUDA but simpler. In practice, I just took the tiled matmul from the Triton tutorial and changed the types to int8 with int32 accumulators. The goal is to perform int8 by int8 products but avoid overflow.

```python
@triton.jit
def matmul_kernel(...):
    ...
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.int32)
    ...

def matmul_i8i32(a, b):
    ''' matmul for int8 with int32 accumulators '''
    ...
    c = t.empty((M, N), device=a.device, dtype=t.int32)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](...)
    return c
```

### Symmetric Quantized Matmul
The symmetric quantized matmul is pretty straightforward. We do a matmul between two int8 tensors using our handmade kernel, and then unquantize it by multiplying by the product of scales.

{{< figure src="symmetric-matmul.svg" alt="Symmetric Matmul" >}}

```python
def symmetric_quantized_matmul(xq, wq, scale_x, scale_w):
    yq = matmul_i8i32(xq, wq)
    scale_y = scale_x * scale_w
    return unquantize(yq, scale_y)
```

### Asymmetric Quantized Matmul
This one is a bit more involved because of the zero points.

{{< figure src="asymmetric-matmul.svg" alt="Asymmetric Matmul" >}}

```python
def asymmetric_quantized_matmul(xq, wq, scale_x, scale_w, zero_x, zero_w):
    unscaled_y = (
        matmul_ui8i32(xq, wq)
        - xq.sum(1, keepdim=True) * zero_w
        - zero_x * wq.sum(0, keepdim=True)
        + xq.shape[1] * zero_x * zero_w)
    return scale_x * scale_w * unscaled_y
```

Note: in practice some of the terms can be precomputed.

## Quantization of the Network
First we make a QuantizedLinear module.

```python
class QuantizedLinear(nn.Module):
    def __init__(self, linear):
        super().__init__()
        w, scale = quantize(linear.weight.T)
        self.w = w
        self.register_buffer('w_matrix', self.w)
        self.scale_w = scale
        self.bias = linear.bias # keep bias unquantized

    def forward(self, x):
        xq, scale_x = quantize(x)
        yq = matmul_i8i32(xq, self.w)
        scale_y = self.scale_w * scale_x
        y = unquantize(yq, scale_y)
        y = y + self.bias
        return y
```

We have all the pieces. Let's write the code to quantize a network. We recursively search for `nn.Linear` modules and replace them with our quantized version.

```python
def quantize_module(module):
    for name, node in module.named_children():
        if isinstance(node, nn.Linear):
            setattr(module, name, QuantizedLinear(node))
        else:
            quantize_module(node)
```

Quantize a model and test it for accuracy.

```python
weights = t.load('weights/mnist.pt')
mnist_base = Mnist().to(device)
mnist_base.load_state_dict(weights)
mnist = Mnist().to(device)
mnist.load_state_dict(weights)
quantize_module(mnist)

print(f'base fp32: {eval(mnist_base)[1]}')
print(f'quantized int8: {eval(mnist)[1]}')
```

And for size.

```python
def model_size(model):
    return sum(p.numel() * p.element_size() for p in model.parameters()) + \
        sum(b.numel() * b.element_size() for b in model.buffers())

print(f'base fp32 size: {model_size(mnist_base)}')
print(f'quantized int8 size: {model_size(mnist)}')
print(f'quantize / base ratio: {model_size(mnist) / model_size(mnist_base):.2f}')
```

| Model                | Accuracy | Size (bytes) |
|----------------------|----------|--------------|
| Base fp32            | 0.9468   | 560424       |
| Quantized int8       | 0.9464   | 140712       |

This gives us a 4x reduction in size for a 0.04% accuracy drop.

## The code
You can get the code at https://github.com/peluche/bag-of-holding

## Sources
- Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference: https://arxiv.org/abs/1712.05877
- A White Paper on Neural Network Quantization: https://arxiv.org/abs/2106.08295
- Triton matmul: https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
