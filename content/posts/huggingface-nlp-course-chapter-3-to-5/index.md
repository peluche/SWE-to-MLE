---
title: "Huggingface Nlp Course Chapter 3 to 5"
date: 2023-09-02T19:23:41+02:00
draft: false
author: "peluche"
authorLink: "https://github.com/peluche"
description: "Hugging Face NLP Course Chapter 3 to 5"
resources:
- name: "cosine similarity"
  src: "cosine_similarity.png"
tags: ['huggingface', 'transformer', 'tokenizer', 'pytorch', 'python', 'jupyter', 'FAISS', 'embedding']
categories: ['learning']
---

Continuing with [🤗 Hugging Face: NLP Course](https://huggingface.co/learn/nlp-course).

## Chapter 3: Fine-Tuning a Pretrained Model
Goes over fine-tuning. It's a OK as a source of copy/pastable snippets, but there isn't much insight to glean from here.

## Chapter 4: Sharing Models and Tokenizers
Nothing to see here. Just an advertisement for the 🤗 platform.

## Chapter 5: The 🤗 Datasets Library
Another OK source of copy/pastable snippets. But this time we also get a treat, an intro to [FAISS (Facebook AI Similarity Search)](https://ai.meta.com/tools/faiss/).

The end of the chapter introduce embeddings. And how to find the sementically closest neighbour using [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity).

![Cosine Similarity](cosine_similarity.png "Cosine Similarity")

Go get a look at the original at https://huggingface.co/learn/nlp-course/chapter5/6

{{< youtube OATCgQtNX2o >}}

My follow along version of the code is on [GitHub](https://github.com/peluche/huggingface-NLP-course) or bellow.

{{< gist peluche 5e2d8a5938924e6dff1e98b06af4d3d2 >}}
