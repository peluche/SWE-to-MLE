---
title: "Einsum for Tensor Manipulation"
date: 2024-03-17T15:12:48+01:00
draft: false
author: "peluche"
authorLink: "https://github.com/peluche"
description: "Einsum for Tensor Manipulation"
tags: ['einsum', 'einops', 'tensor', 'pytorch']
categories: ['bestiary']
resources:
- name: "foo"
  src: "foo.png"
math:
  enable: true
---


*In the ethereal dance of the cosmos, where the arcane whispers intertwine with the silent echoes of unseen dimensions, the Ioun Stone of Mastery emerges as a beacon of unparalleled prowess. This luminescent orb, orbiting its bearer's head, is a testament to the mastery of both magical and mathematical realms, offering a bridge between the manipulation of arcane energies and the intricate ballet of tensor mathematics. As the stone orbits, it casts a subtle glow, its presence a constant reminder of the dual dominion it grants over the spellbinding complexities of magic and the abstract elegance of multidimensional calculations, making the wielder a maestro of both mystical incantations and the unseen algebra of the universe.*

![ioun-stone](ioun-stone.png "Ioun Stone of Mastery")

## The Quest
Study how the Ioun Stone powers work. Understand how Einsum operates over tensors.

## Einsum Uses
Einsum (and einops in general) is a great tool for manipulating tensors. In ML it is often used to implement matrix multiplication or dot products. The simplest case would look like:
```python
x = torch.rand((4, 5))
y = torch.rand((5, 3))

# the torch way
res = x @ y

# einsum way
res = einsum(x, y, 'a b, b c -> a c')
```

Right now it looks like a verbose way of doing the same thing, but it sometimes presents the following advantages:
- documenting the tensor dimensions for ease of reading
- implicit reordering of dimensions

```python
query = torch.rand((100, 20, 32))
key   = torch.rand((100, 20, 32))

# the torch way
keyT = key.permute((0, 2, 1))
res = query @ keyT

# einsum way
res2 = dumbsum(query, key, 'batch seq_q d_model, batch seq_k d_model -> batch seq_q seq_k')
```

## Einsum the Iterative way
Conceptually it's possible to think of einsum as bunch of nested loops:
- the first set of nested loops is used to index into the inputs and output.
- the second set of nested loops for summing all the left over dimensions that are getting reduced.

It could be written by hand as:
```python
result = torch.zeros((10, 20, 20))
for batch in range(10):
  for seq_q in range(20):
    for seq_k in range(20):
      tot = 0
      for d_model in range(32):
        tot += query[batch, seq_q, d_model] * key[batch, seq_k, d_model]
      result[batch, seq_q, seq_k] = tot
```

One way to generate these nested loops is to use recursion:
```python
def dumbsum(x, y, shapes):
  '''
  dumb implem for my own intuition building sake, with absolutely no value for real life use.
  not vectorized, and do not handle splitting / merging / creating extra dim.
  
  the main idea is to:
  1- generate nested loops for indexing for each dim in the output
  2- generate nexted loops for summing everything else
  e.g. 'a b c d e, a c e -> a d b'
  for a in range(x.shape[0]):
    for d in range(x.shape[3]):
      for b in range(x.shape[1]):
        tot = 0
        for c in range(x.shape[2]):
          for e in range(x.shape[4]):
            tot += x[a, b, c, d, e] * y[a, c, e]
        res[a, d, b] = tot

  in practice I initialize res to a tensor of zero, and update it in place instead of accumulating in a tot
  res[a, d, b] += x[a, b, c, d, e] * y[a, c, e]
  '''
  def split_shape(shape):
    return [x for x in shape.split(' ') if x]
  def parse(shapes):
    assert shapes.count(',') == 1
    assert shapes.count('->') == 1
    shapes, res_shape = shapes.split('->')
    x_shape, y_shape = shapes.split(',')
    x_shape, y_shape, res_shape = (split_shape(s) for s in (x_shape, y_shape, res_shape))
    sum_shape = list(set(x_shape + y_shape) - set(res_shape))
    assert set(res_shape).issubset(set(x_shape + y_shape))
    return x_shape, y_shape, res_shape, sum_shape
  def build_dim_lookup(t, t_shape, lookup=None):
    if not lookup: lookup = {}
    dims = t.shape
    for dim, letter in zip(dims, t_shape):
      assert lookup.get(letter, dim) == dim
      lookup[letter] = dim
    return lookup
  def iterate(shape, sum_shape, fn, lookup, indexes):
    if not shape:
      iterate_sum(sum_shape[:], fn, lookup, indexes)
      return
    dim = shape.pop(-1)
    # print(f'iterate over → {dim}')
    for i in range(lookup[dim]):
      indexes[dim] = i
      iterate(shape[:], sum_shape, fn, lookup, indexes)
  def iterate_sum(sum_shape, fn, lookup, indexes):
    if not sum_shape:
      fn(indexes)
      return
    dim = sum_shape.pop(-1)
    # print(f'sum over → {dim}')
    for i in range(lookup[dim]):
      indexes[dim] = i
      iterate_sum(sum_shape[:], fn, lookup, indexes)
  def ind(t_shape, indexes):
    return (indexes[dim] for dim in t_shape)
  def close_sum(x, y, res, x_shape, y_shape, res_shape):
    def fn(indexes):
      # print(f'res[{tuple(ind(res_shape, indexes))}] += x[{tuple(ind(x_shape, indexes))}] * y[{tuple(ind(y_shape, indexes))}]')
      res[*ind(res_shape, indexes)] += x[*ind(x_shape, indexes)] * y[*ind(y_shape, indexes)]
    return fn

  x_shape, y_shape, res_shape, sum_shape = parse(shapes)
  assert len(x_shape) == x.dim()
  assert len(y_shape) == y.dim()
  lookup = build_dim_lookup(x, x_shape)
  lookup = build_dim_lookup(y, y_shape, lookup=lookup)
  res = t.zeros(tuple(lookup[s] for s in res_shape))
  fn = close_sum(x, y, res, x_shape, y_shape, res_shape)
  iterate(res_shape[:], sum_shape[:], fn, lookup, {})
  return res
```

## Einsum Vectorized
The loop version is great for intuition building, but it is extremely slow. Another way to implement einsum is to compose vectorized torch operations.

By hand it would look something like:
```python
query = query[..., None] # add a seq_k dimension
key = key[..., None]     # add a seq_q dimension
query = query.permute((0, 1, 3, 2)) # align the dimensions as: batch, seq_q, seq_k, d_model
key = key.permute((0, 3, 1, 2))     # align the dimensions as: batch, seq_q, seq_k, d_model 
product = query * key # multiply element wise using implicit broadcasting
result = product.sum((3)) # reduce the extra dimension out
```

Which in code could look a little something like:
```python
def dumbsum_vectorized(x, y, shapes):
  '''
  vectorize it, still do not handle splitting / merging / creating extra dim.
  my vectorized also does not handle repeated dim (e.g. 'a a b, a a c -> a a').
  
  the main idea is to:
  1- align the dimensions of x and y, completing the holes with fake `1` dimensions
  2- multiply x and y
  3- sum out the extra dims
  e.g. 'a c d e, a c e -> a d b'
  # align dims
  x = reshape('a c d e -> a 1 c d e')
  y = reshape('a c e   -> a 1 c 1 e')
  # order dims
  x = reshape('a 1 c d e -> a d 1 c e')
  y = reshape('a 1 c 1 e -> a 1 1 c e')
  # mult and sum
  res = x * y
  res = res.sum((3, 4))
  '''
  def split_shape(shape):
    return [x for x in shape.split(' ') if x]
  def parse(shapes):
    assert shapes.count(',') == 1
    assert shapes.count('->') == 1
    shapes, res_shape = shapes.split('->')
    x_shape, y_shape = shapes.split(',')
    x_shape, y_shape, res_shape = (split_shape(s) for s in (x_shape, y_shape, res_shape))
    sum_shape = list(set(x_shape + y_shape) - set(res_shape))
    assert set(res_shape).issubset(set(x_shape + y_shape))
    return x_shape, y_shape, res_shape, sum_shape
  def build_dim_pos_lookup(t_shape):
    return {letter: dim for dim, letter in enumerate(t_shape)}
  def expand(t, t_shape, merged):
    lookup = build_dim_pos_lookup(t_shape)
    ind = len(lookup)
    for dim in merged:
      if dim not in lookup:
        t = t.unsqueeze(-1)
        lookup[dim] = ind
        ind += 1
    return t, lookup
  def align(t, lookup, res_lookup):
    # rely on dict being ordered (python >= 3.7)
    permuted_dims = tuple(lookup[dim] for dim in res_lookup)
    return t.permute(permuted_dims)
  def dims_to_sum(res_shape, res_lookup):
    return tuple(range(len(res_shape), len(res_lookup)))

  x_shape, y_shape, res_shape, sum_shape = parse(shapes)
  assert len(x_shape) == x.dim()
  assert len(y_shape) == y.dim()
  merged = set(x_shape + y_shape)
  x, x_lookup = expand(x, x_shape, merged)
  y, y_lookup = expand(y, y_shape, merged)
  _, res_lookup = expand(t.zeros((0)), res_shape, merged)
  x = align(x, x_lookup, res_lookup)
  y = align(y, y_lookup, res_lookup)
  res = x * y
  dims = dims_to_sum(res_shape, res_lookup)
  if dims: res = res.sum(dims)
  return res
```

## Compare both

### Correctness
We can verify that both versions are producing the same results as the original einsum:
```python
import torch, einops

def einops_test(x, y, pattern):
  a = dumbsum(x, y, pattern)
  b = dumbsum_vectorized(x, y, pattern)
  c = einops.einsum(x, y, pattern)
  assert a.allclose(c)
  assert b.allclose(c)

x = torch.rand((10, 5, 2, 3))
y = torch.rand((3, 10, 5, 7))
einops_test(x, y, 'a b c d, d a b e -> b e c')
einops_test(x, y, 'a b c d, d a b e -> a b c d e')
einops_test(x, y, 'a b c d, d a b e -> e d c b a')
einops_test(x, y, 'a b c d, d a b e -> a')
einops_test(x, y, 'a b c d, d a b e ->')
einops_test(x, y, 'a b c d, d a b e -> a e')
```

### Speed
Timing the iterative version:
```python
%%time
query = torch.rand((100, 20, 32))
key = torch.rand((100, 20, 32))
_ = dumbsum(query, key, 'batch seq_q d_model, batch seq_k d_model -> batch seq_q seq_k')
```

```
CPU times: total: 9.58 s
Wall time: 31.3 s
```

Against the vectorized version:
```python
%%time
query = torch.rand((100, 20, 32))
key = torch.rand((100, 20, 32))
_ = dumbsum_vectorized(query, key, 'batch seq_q d_model, batch seq_k d_model -> batch seq_q seq_k')
```

```
CPU times: total: 0 ns
Wall time: 975 µs
```

Demonstrates the significant speedup brought by using vectorized code.

## The code
You can get the code at https://github.com/peluche/ml-misc/blob/master/einsum-intuition.ipynb
