---
title: "What is BPE: Byte-Pair Encoding?"
source: "https://medium.com/@parklize/what-is-bpe-byte-pair-encoding-5f1ea76ea01f"
author:
  - "[[GUANGYUAN PIAO]]"
published: 2025-05-02
created: 2026-04-15
description: "What is BPE: Byte-Pair Encoding? What is BPE? Byte-Pair Encoding (BPE) was originally invented as a text compression algorithm [1]. It was later adopted in the context of machine translation first in …"
type: "youtube"
tags:
  - "clippings"
  - "youtube"
status: "inbox"
---
## 핵심 요약

> 클리핑 후 직접 작성하거나 비워두세요.

---

## 영상 내용

## What is BPE?

Byte-Pair Encoding (BPE) was originally invented as a text compression algorithm \[1\]. It was later adopted in the context of machine translation first in Natural Language Processing (NLP) \[2\], and then used by OpenAI for tokenization during the pretraining the GPT model, as well as many other Transformer models.

In this post, we will borrow the example from the LLM course \[3,4\] of [HuggingFace](https://huggingface.co/) to explain how BPE works.

## Starting point

BPE is a *subword* tokenization algorithm that, after training, **breaks text into smaller pieces** called **tokens**.

For example: “hugs” → (trained) BPE → “hug”, “s”

> Training a tokenizer is a statistical process that tries to identify which subwords are the best to pick for a given corpus, and the exact rules used to pick them depend on the tokenization algorithm. — [LLM Course](https://huggingface.co/learn/llm-course/chapter6/2?fw=pt), HuggingFace

BPE training starts with a standardized and pre-tokenized corpus.

As an example, consider the following corpus. The raw corpus can be represented as the convention `frequency x distinct word` instead, which we will follow throughout the rest of this post.

![](https://miro.medium.com/v2/resize:fit:1386/format:webp/1*HQZQzHqyPx4zfoAGsC5d6g.png)

## Initial splits and vocabulary

Given a corpus following the above-mentioned convention, as illustrated in the figure below, we first split each distinct word into characters. The set of distinct characters that appear in these words forms the intial vocabulary (vocab). For example, the vocab shown below contains 12 characters.

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*77c4Qy8-yeBpVXZoQirEDQ.png)

## Iteration: calculate pair frequencies > deriving top merging rule > update splits and vocab

BPE training starts with the intial vocabulary and gradually expands it to reach the desired vocabulary size, denoted as *n*.

Let’s say we want to have a vocabulary size of *n* = 20.

Traning proceeds by repeatedly performing the following steps:

1. Calculate pair frequencies, by sliding over each pair of consecutive characters in the current splits.
2. Choose the most frequent pair from this set; this becomes the next merging rule.
3. Update the splits and vocab by applying the new merging rule.
![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*TmgysOLFAh8BEkenOO-_qw.png)

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*86WifoVJZuo32d06fzDTwQ.png)

This iterative process continues until the vocabulary reaches the desired size, e.g., *n* = 20 as shown in the image below.

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*zWvroDCCvfO24xT7-ou9MA.png)

That’s it for BPE training! Now we can apply our (trained) BPE to new text to break it into smaller subword units/tokens.

## Applying the derived BPE to new text

Let’s assume we have a new text, “hugs”, which we want to tokenize using our trained BPE model.

## Get GUANGYUAN PIAO’s stories in your inbox

Join Medium for free to get updates from this writer.

We start by splitting the text into individual characters:

- Initial split: ‘h’, ‘u’, ‘g’, ‘s’

We then apply the learned BPE merging rules sequentially:

- Applying the first merging rule (h + u = hu): ‘hu’, ‘g’, ‘s’
- Applying the second merging rule (hu + g = hug): ‘hug’, ‘s’

At this point, no further merging rules apply, so we are finished.

- Final tokenized output: ‘hug’, ‘s’
![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*kziQwFDbpHvszicSR0fzrQ.png)

## If you are more of a video person

That’s it!

In this post, we explored what BPE is and how it works, using on an intuitive example.

The explanation is based on the awesome video from HuggingFace that breaks down the concept of BPE in a clear and visual way.

Highly recommended if you prefer learning through video over text!

Hope you enjoyed this gentle introduction to BPE!

See you next time!

## References

1. Gage, Philip. “A new algorithm for data compression.” *The C Users Journal* 12.2 (1994): 23–38.
2. [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909), 2016
3. [Byte-Pair Encoding tokenization](https://huggingface.co/learn/llm-course/chapter6/5?fw=pt) from HuggingFace
4. [LLM Course](https://huggingface.co/learn/llm-course/chapter1/1), HuggingFace

[![GUANGYUAN PIAO](https://miro.medium.com/v2/resize:fill:96:96/0*-fYdj9Db7nGPjt2J.)](https://medium.com/@parklize?source=post_page---post_author_info--5f1ea76ea01f---------------------------------------)[17 following](https://medium.com/@parklize/following?source=post_page---post_author_info--5f1ea76ea01f---------------------------------------)