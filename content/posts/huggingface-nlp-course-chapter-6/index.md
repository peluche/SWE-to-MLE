---
title: "Huggingface NLP Course Chapter 6"
date: 2023-09-03T17:22:14+02:00
draft: false
author: "peluche"
authorLink: "https://github.com/peluche"
description: "Hugging Face NLP Course Chapter 6"
resources:
- name: "tokenization pipeline"
  src: "tokenization_pipeline.svg"
tags: ['huggingface', 'transformer', 'tokenizer', 'pytorch', 'python', 'jupyter', 'BPE', 'wordpiece', 'unigram']
categories: ['learning']
---

Continuing with [Chapter 6: The 🤗 Tokenizer Library](https://huggingface.co/learn/nlp-course/chapter6/).

## Theory
This is a good dense chapter covering the theory behind tokenizers. It covers their architecture:

![tokenization pipeline](tokenization_pipeline.svg "Tokenization Pipeline")

The tradeoffs happening during the normalization phase. Followed by a tour of the 3 most popular subwords tokenization algorithms. I highly recommend going over the videos to get a good feel of the implementations.

### BPE (aka. GPT-2)

{{< youtube HEikzVL-lZU >}}

### WordPiece (aka. BERT)

{{< youtube qpv6ms_t_1A >}}

### Unigram (aka. T5)

{{< youtube TGZfZVuF9Yc >}}

## Practical Coding

The chapter also go over how question answering pipeline manage contexts that are bigger than the allowed amount of tokens. (Spoiler alert) split the context into overlapping chunks and attach the question to each of them. Grade all of them and pick the highest confidence answer.

And how to write your own tokenizer pipeline from scratch using their library.

My follow along version of the code is on [GitHub](https://github.com/peluche/huggingface-NLP-course) or bellow.


{{< gist peluche 6e0e7c125e90754d6f6e955ab6984fab >}}
