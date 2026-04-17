---
title: "RAttention: Towards the Minimal Sliding Window Size in Local-Global Attention Models"
source: "https://ar5iv.labs.arxiv.org/html/2506.15545v2"
author:
published:
created: 2026-04-15
description: "Local-global attention models characterAI2024 ; gemmateam2024gemma2  have recently emerged as compelling alternatives to standard Transformers, promising improvements in both training and inference efficiency. However,…"
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

Bailin Wang   Chang Lan   Chong Wang   Ruoming Pang  
Apple  
{bwang47,c\_lan,mr.chongwang,rpang}@apple.com

###### Abstract

Local-global attention models [^28] [^30] have recently emerged as compelling alternatives to standard Transformers, promising improvements in both training and inference efficiency. However, the crucial choice of window size presents a Pareto tradeoff: larger windows maintain performance akin to full attention but offer minimal efficiency gains in short-context scenarios, while smaller windows can lead to performance degradation. Current models, such as Gemma2 and Mistral, adopt conservative window sizes (e.g., 4096 out of an 8192 pretraining length) to preserve performance. This work investigates strategies to shift this Pareto frontier, enabling local-global models to achieve efficiency gains even in short-context regimes. Our core motivation is to address the intrinsic limitation of local attention—its complete disregard for tokens outside the defined window. We explore RAttention, a variant of local attention integrated with a specialized linear attention mechanism designed to capture information from these out-of-window tokens. Pretraining experiments at the 3B and 12B scales demonstrate that RAttention achieves a superior Pareto tradeoff between performance and efficiency. As a sweetspot, RAttention with a window size of just 512 consistently matches the performance of full-attention models across diverse settings. Furthermore, the recurrent nature inherent in the linear attention component of RAttention contributes to enhanced long-context performance, as validated on the RULER benchmark [^14]. Crucially, these improvements do not compromise training efficiency; thanks to a specialized kernel implementation and the reduced window size, RAttention maintains training speeds comparable to existing state-of-the-art approaches. <sup>1</sup>

## 1 Introduction

Improving the efficiency of standard Transformers has been a central focus of architectural design. Sliding Window Attention (SWA) [^7], a natural variant of standard attention, has been widely adopted to reduce both memory transfer and computational costs, as demonstrated in prominent models like Mistral [^15] and Gemma [^30]. Its constant memory consumption during decoding is particularly appealing for decoding time which is bounded by memory transfers rather than computation. However, SWA presents an inherent Pareto tradeoff between model capacity and efficiency: increasing the window size improves performance but diminishes efficiency gains. Recent models have often adopted conservative window sizes; for instance, both Mistral and Gemma utilize a 4096-token window out of a 8192-token pretraining context. Consequently, the efficiency benefits of SWA only become substantial for relatively long sequences. To illustrate, a 12B parameter local-global attention model we tested, using a 4K window size, apparently offers no KV cache savings for $\leq$ 4K context tasks. However, reducing the window size to 1k could yield approximately 56% KV cache savings at 4K length. In this work, we investigate whether this Pareto frontier can be shifted – achieving comparable performance with a significantly smaller window size.

Our primary hypothesis is that the performance degradation observed when reducing the SWA window size stems from an intrinsic limitation of local attention: simply ensuring that number\_of\_layers $\times$ window\_size $\geq$ context\_length is often an inadequate heuristic for determining the optimal window size. To address this, we explore the integration of a Residual Linear Attention (RLA) module. RLA, in our design, is a specialized linear attention model [^17] aimed at capturing information from "residual tokens" – those lying beyond the immediate SWA window. This RLA module inherently enhances the capabilities of SWA. The recurrent nature of RLA offers two significant advantages: first, it preserves the constant-memory property crucial for efficient decoding with SWA; second, it lessens the over-reliance on positional embeddings for length extrapolation, leading to markedly better zero-shot length generalization compared to SWA alone. As depicted in Figure 1, the resulting hybrid architecture, which we term RAttention (Sliding Window Attention with Residual Linear Attention), effectively captures information from all tokens within the context.

The key contribution of this work is demonstrating that RAttention can serve as a drop-in replacement for standard local attention mechanisms, positioning local-global models with RAttention as practical alternatives to full-attention Transformers. First, we show that RAttention models <sup>2</sup> can match, and in some cases outperform, SWA models that use much larger sliding window sizes, while offering superior inference efficiency. Second, we demonstrate that this improved efficiency does not compromise training speed; with our dedicated kernel implementations and the reduced window size, RAttention models maintain training efficiency comparable to SWA models with larger windows. While the concept of combining SWA with linear attention has been explored (e.g., in [^2] [^40], where a tiny SWA window complements linear attention), those approaches did not match the performance of full-attention models. We show that local-global models incorporating RAttention can be reliable alternatives to standard full-attention Transformers.

We conduct extensive experiments comparing RAttention with SWA within a local-global hybrid attention framework. Our results demonstrate that RAttention consistently matches or surpasses the performance of SWA-based models while requiring significantly less memory due to smaller window sizes. Furthermore, RAttention yields substantial performance improvements on long-context reasoning tasks, as measured by the RULER benchmark [^14]. On the training efficiency side, we develop dedicated kernels for RAttention so that the training efficiency is not compromised compared with full-attention models.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2506.15545/assets/figures/teaser.png)

Figure 1: RAttention combines Sliding Window Attention (SWA) for local context with Residual Linear Attention (RLA) to gather information from out-of-window tokens. Apart from 4 in-window tokens, RLA compresses the information of token 1 for query token 5; token 1,2 for query token 6.

## 2 Background

We first briefly introduce standard attention and linear attention mechanisms. For notation, we use uppercase letters for matrices, lowercase letters for row vectors.

### 2.1 Global Attention and Local Attention

In standard Transformers, an input sequence ${\mathbf{X}}\in\mathbb{R}^{L\times d}$ (here $L$ is the length and $d$ is the hidden dimension) in the attention layer is processed through,

$$
\displaystyle{\bm{q}}_{t},\ {\bm{k}}_{t},\ {\bm{v}}_{t}
$$
 
$$
\displaystyle={\bm{x}}_{t}{\bm{W}}_{Q},\ {\bm{x}}_{t}{\bm{W}}_{K},\ {\bm{x}}_{t}{\bm{W}}_{V},
$$
$$
\displaystyle{\bm{o}}_{t}
$$
 
$$
\displaystyle=\frac{\sum_{i=1}^{t}\exp({\bm{q}}_{t}{\bm{k}}_{i}^{\intercal}){\bm{v}}_{i}}{\sum_{i=1}^{t}\exp({\bm{q}}_{t}{\bm{k}}_{i}^{\intercal})},
$$

which computes the query (${\bm{q}}_{t}$), key (${\bm{k}}_{t}$), and value (${\bm{v}}_{t}$) vectors given the current token’s representation ${\bm{x}}_{t}\in\mathbb{R}^{1\times d}$. Then attention is performed over the all the previous keys $\{{\bm{k}}_{1},\dots,{\bm{k}}_{t}\}$ and values $\{{\bm{v}}_{1},\dots,{\bm{v}}_{t}\}$. During inference, as the $t$ grow, the set of keys and values (i.e., KV cache) also grows linearly, leading to heavy memory consumption if $t$ is large. Local attention (i.e., sliding window attention) [^7] is then introduced to achieve constant-memory consumption during inference.

The basic idea of SWA is to limit the attention to only recent $w$ tokens through,

$$
\displaystyle{\bm{o}}^{\text{\scriptsize{swa}}}_{t}
$$
 
$$
\displaystyle=\frac{\sum_{i=\max(1,t-w)}^{t}\exp({\bm{q}}_{t}{\bm{k}}_{i}^{\intercal}){\bm{v}}_{i}}{\sum_{i=\max(1,t-w)}^{t}\exp({\bm{q}}_{t}{\bm{k}}_{i}^{\intercal})}.
$$

As a result, at most $w+1$ tokens (including the current token) need to be considered as KV cache during inference. The choice of $w$ is based on the Pareto tradeoff between downstream performance and efficiency, and we aim to shift the Pareto frontier in this work.

### 2.2 Linear Attention

Linear attention mechanisms [^17] replace $\exp({\bm{q}}_{t}{\bm{k}}_{i}^{\intercal})$ with a kernel $k({\bm{x}},{\bm{y}})$ with an associated feature map $\phi$ (i.e., $k({\bm{x}},{\bm{y}})=\langle\phi({\bm{x}}),\phi({\bm{y}})\rangle$). This simplifies the calculation of ${\bm{o}}_{t}$ since we have

$$
\displaystyle{\bm{o}}^{\text{\scriptsize{la}}}_{t}
$$
 
$$
\displaystyle=\frac{\sum_{i=1}^{t}\phi({\bm{q}}_{t})\phi({\bm{k}}_{i})^{\intercal}{\bm{v}}_{i}}{\sum_{i=1}^{t}\phi({\bm{q}}_{t})\phi({\bm{k}}_{i})^{\intercal}}=\frac{\phi({\bm{q}}_{t})\sum_{i=1}^{t}\phi({\bm{k}}_{i})^{\intercal}{\bm{v}}_{i}}{\phi({\bm{q}}_{t})\sum_{i=1}^{t}\phi({\bm{k}}_{i})^{\intercal}}.
$$

Letting ${\mathbf{S}}_{t}=\sum_{i=1}^{t}\phi({\bm{k}}_{i})^{\intercal}{\bm{v}}_{i}$ and ${\bm{z}}_{t}=\sum_{i=1}^{t}\phi({\bm{k}}_{i})^{\intercal}$ where ${\mathbf{S}}_{t}\in\mathbb{R}^{d^{\prime}\times d},{\bm{z}}_{t}\in\mathbb{R}^{d^{\prime}\times 1}$, we can rewrite the above as an RNN,

$$
\displaystyle{\mathbf{S}}_{t}={\mathbf{S}}_{t-1}
$$
 
$$
\displaystyle+\phi({\bm{k}}_{t})^{\intercal}{\bm{v}}_{t},\hskip 2.84526pt{\bm{z}}_{t}={\bm{z}}_{t-1}+\phi({\bm{k}}_{t})^{\intercal},\hskip 2.84526pt{\bm{o}}^{\text{\scriptsize{la}}}_{t}=\frac{\phi({\bm{q}}_{t}){\mathbf{S}}_{t}}{\phi({\bm{q}}_{t}){\bm{z}}_{t}}.
$$

where $d^{\prime}$ denotes the output size of feature function $\phi$, initial state ${\mathbf{S}}_{0}=\bm{0}$. Recent work has found that the normalization terms can be replaced with a simpler RMSNorm on the output.

$$
\displaystyle{\mathbf{S}}_{t}={\mathbf{S}}_{t-1}+\phi({\bm{k}}_{t})^{\intercal}{\bm{v}}_{t},\quad{\bm{o}}^{\text{\scriptsize{la}}}_{t}=\phi({\bm{q}}_{t}){\mathbf{S}}_{t}.
$$

Eq. 1 makes it clear that a linear attention layer is essentially a linear recurrent layer with matrix-valued hidden states ${\mathbf{S}}_{t}$ that is updated via the outer-product $\phi({\bm{k}}_{t})^{\intercal}{\bm{v}}_{t}$.

#### Chunkwise Parallel Form

The recurrent form in Eq.1 highlights the inherently sequential nature of linear attention, which makes it advantageous during the decoding stage. During training or prefilling, we rely on an equivalent chunkwise parallel formulation to compute the output efficiently [^37].

Let the tilded input ${\mathbf{Q}}_{[i]}:={\mathbf{Q}}_{iC+1:(i+1)C+1}\in\mathbb{R}^{C\times d}$ represent the query vectors for the $i$ -th chunk, and define ${\mathbf{K}}_{[i]}$, ${\mathbf{V}}_{[i]}$, and ${\mathbf{O}}_{[i]}$ similarly. The output for the chunk can be computed as:

$$
\displaystyle{\mathbf{S}}_{[i+1]}={\mathbf{S}}_{[i]}+\tilde{\mathbf{K}}^{\intercal}_{[i]}{\mathbf{V}}_{[i]}\quad\hskip 2.84526pt\in\mathbb{R}^{d^{\prime}\times d}
$$
 
$$
\displaystyle{\mathbf{O}}_{[i]}=\underbrace{{\mathbf{Q}}_{[i]}{\mathbf{S}}_{[i-1]}}_{{\mathbf{O}}^{\text{inter}}_{[i]}}+\underbrace{\big{(}((\tilde{\mathbf{Q}}_{[i]})\tilde{\mathbf{K}}_{[i]}^{\intercal})\odot{\mathbf{M}}\big{)}{\mathbf{V}}_{[i]}}_{{\mathbf{O}}^{\text{intra}}_{[i]}},
$$

Here, $C$ denotes the chunk (tiling) size, and $\tilde{\mathbf{Q}}=\phi({\mathbf{Q}})$, $\tilde{\mathbf{K}}=\phi({\mathbf{K}})$. The key insight is that, given the chunk-level state ${\mathbf{S}}_{[i]}$, the output within each chunk can be computed in parallel using matrix multiplications – an operation that modern accelerators are highly optimized for.

Compared to global attention, linear attention requires maintaining only a fixed-size state S during inference, similar to local attention. Specifically, its cache is equivalent to using a window size of $\frac{d^{\prime}\times d}{2\times d}$ in SWA <sup>3</sup>. While this leads to significant inference efficiency, pure linear attention models still fall short of Transformers, particularly in recall-intensive tasks [^1].

## 3 Method

### 3.1 Residual Linear Attention (RLA)

To address the limitation of SWA, we use a specialized linear attention (RLA) that processes out-of-window tokens, as illustrated in the middle figure in Figure 1. Concretely, the recurrence of RLA is as follwows,

$$
\displaystyle{\mathbf{S}}_{t}={\mathbf{S}}_{t-1}+\phi({\bm{k}}_{t})^{\intercal}{\bm{v}}_{t},\quad{\bm{o}}^{\text{\scriptsize{rla}}}_{t}=\phi({\bm{q}}_{t}){\mathbf{S}}_{t-w-1}.
$$

where instead of reading out from ${\mathbf{S}}_{t}$ as in Eq 1, ${\bm{o}}_{t}$ is obtained via reading out from ${\mathbf{S}}_{t-w-1}$ – the hidden states that captures the contextual information ends at token $(t-w-1)$. To integrate with SWA, we combine he outputs from RLA and SWA, as demonstrated in Figure 1.

#### Parameterization

Employing separate parameters (i.e., projections for query/key/value/output) for RLA and SWA would double the parameter size of token-mixing layers. In this work, we find that simply shares all parameters is sufficient, provided that an appropriate feature map is used. As we will show in the experiments later, the softmax-based feature map [^40] yields the best performance, whereas the identity feature map, commonly used in pure linear attention models, is less effective.

#### Group-Query Variant

We extend RLA to support Group-Query Attention (GQA), a popular variant of Multi-Head Attention that reduces KV memory usage. Specifically, we share key and value vectors within each query group, maintaining efficiency while preserving model quality. When coupled with SWA, RLA will share the same grouping structures as SWA as well.

### 3.2 Hybrid RAttention Models

We combine the results from RLA and SWA as follows:

$$
\displaystyle{\bm{o}}_{t}=\text{RMS}({\bm{o}}^{\text{\scriptsize{swa}}}_{t})+\text{RMS}({\bm{o}}^{\text{\scriptsize{rla}}}_{t})
$$

where two separate RMS norms are used upon the output from SWA and RLA respectively. In the group-query head variant, different heads will use separate parameters for RMS, which is a common strategy used in linear attention models [^27] [^37] [^10]. Collectively, we call the resulting hybrid attention module RAttention.

#### Parameter Efficiency

Our design introduces no additional parameters over standard SWA modules, as RAttention fully reuses the existing query/key/value projections. In contrast, recent work [^12] that combines state-space models (SSM) with SWA requires separate parameter sets for SSM and SWA. This results in a significantly larger portion of the model’s parameters being allocated to token-mixing layers, which potentially limit the capacity of the feedforward layers when scaling up.

#### Kernel Design

<svg id="S3.F2.pic1" height="259.66" overflow="visible" version="1.1" viewBox="0 0 595.6 259.66" width="595.6"><g transform="translate(0,259.66) matrix(1 0 0 -1 0 0) translate(197.4,0) translate(0,211.86)" fill="#000000" stroke="#000000"><g stroke-width="0.8pt"><g stroke="#000000" fill="#FAFAFA"><path d="M -180.76 -169.85 h 566.58 v 217.1 h -566.58 Z"></path></g><g transform="matrix(1.0 0.0 0.0 1.0 -161.07 -61.3)" fill="#000000" stroke="#000000"></g></g><g stroke="#4D4DFF" fill="#D9D9FF" fill-opacity="0.5" stroke-dasharray="3.0pt,3.0pt" stroke-dashoffset="0.0pt" stroke-opacity="0.5" stroke-width="0.8pt"><path d="M -172.88 -122.61 h 168.2 v 149.74 h -168.2 Z"></path></g><g stroke-width="0.8pt" fill="#000000" fill-opacity="0.5" stroke="#000000" stroke-dasharray="3.0pt,3.0pt" stroke-dashoffset="0.0pt" stroke-opacity="0.5" transform="matrix(1.0 0.0 0.0 1.0 -161.07 -47.74)"></g><g stroke-width="0.8pt" fill="#000000" stroke="#000000" transform="matrix(1.0 0.0 0.0 1.0 -168.27 -118)"><foreignObject width="63.55" height="8.65" transform="matrix(1 0 0 -1 0 8.65)" overflow="visible" style="--fo_width :4.86em;--fo_height:0.66em;--fo_depth :0em;"><span style="font-size:90%;">inter-chunk</span></foreignObject></g> <g stroke="#4D4DFF" fill="#D9D9FF" fill-opacity="0.5" stroke-dasharray="3.0pt,3.0pt" stroke-dashoffset="0.0pt" stroke-opacity="0.5" stroke-width="0.8pt"><path d="M 122.39 -126.16 h 148.52 v 153.29 h -148.52 Z"></path></g><g stroke-width="0.8pt" fill="#000000" fill-opacity="0.5" stroke="#000000" stroke-dasharray="3.0pt,3.0pt" stroke-dashoffset="0.0pt" stroke-opacity="0.5" transform="matrix(1.0 0.0 0.0 1.0 134.2 -49.52)"></g><g stroke-width="0.8pt" fill="#000000" stroke="#000000" transform="matrix(1.0 0.0 0.0 1.0 127.01 -121.55)"><foreignObject width="63.55" height="8.65" transform="matrix(1 0 0 -1 0 8.65)" overflow="visible" style="--fo_width :4.86em;--fo_height:0.66em;--fo_depth :0em;"><span style="font-size:90%;">inter-chunk</span></foreignObject></g> <g stroke="#4DFF4D" fill="#D9FFD9" fill-opacity="0.5" stroke-dasharray="3.0pt,3.0pt" stroke-dashoffset="0.0pt" stroke-opacity="0.5" stroke-width="0.8pt"><path d="M -74.06 -165.35 h 142.22 v 180.67 h -142.22 Z"></path></g><g stroke-width="0.8pt" fill="#000000" fill-opacity="0.5" stroke="#000000" stroke-dasharray="3.0pt,3.0pt" stroke-dashoffset="0.0pt" stroke-opacity="0.5" transform="matrix(1.0 0.0 0.0 1.0 -62.25 -75.02)"></g><g stroke-width="0.8pt" fill="#000000" stroke="#000000" transform="matrix(1.0 0.0 0.0 1.0 0 -160.74)"><foreignObject width="63.55" height="8.65" transform="matrix(1 0 0 -1 0 8.65)" overflow="visible" style="--fo_width :4.86em;--fo_height:0.66em;--fo_depth :0em;"><span style="font-size:90%;">intra-chunk</span></foreignObject></g> <g stroke="#4DFF4D" fill="#D9FFD9" fill-opacity="0.5" stroke-dasharray="3.0pt,3.0pt" stroke-dashoffset="0.0pt" stroke-opacity="0.5" stroke-width="0.8pt"><path d="M 201.53 -165.35 h 161.91 v 180.67 h -161.91 Z"></path></g><g stroke-width="0.8pt" fill="#000000" fill-opacity="0.5" stroke="#000000" stroke-dasharray="3.0pt,3.0pt" stroke-dashoffset="0.0pt" stroke-opacity="0.5" transform="matrix(1.0 0.0 0.0 1.0 213.34 -75.02)"></g><g stroke-width="0.8pt" fill="#000000" stroke="#000000" transform="matrix(1.0 0.0 0.0 1.0 295.28 -160.74)"><foreignObject width="63.55" height="8.65" transform="matrix(1 0 0 -1 0 8.65)" overflow="visible" style="--fo_width :4.86em;--fo_height:0.66em;--fo_depth :0em;"><span style="font-size:90%;">intra-chunk</span></foreignObject></g> <g stroke-width="0.8pt"><g stroke="#000000" fill="#FFF2E6"><path d="M -142.41 -74.8 h 48.59 v 31.5 h -48.59 Z"></path></g><g transform="matrix(1.0 0.0 0.0 1.0 -137.8 -61.34)" fill="#000000" stroke="#000000"><foreignObject width="39.37" height="14.41" transform="matrix(1 0 0 -1 0 9.49)" overflow="visible" style="--fo_width :2.85em;--fo_height:0.69em;--fo_depth :0.36em;"><span style="width:2.85em;"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\mathbf{S}_{[i-1]}"><semantics><msub><mi>𝐒</mi> <mrow><mo stretchy="false">[</mo><mrow><mi>i</mi> <mo>−</mo> <mn>1</mn></mrow><mo stretchy="false">]</mo></mrow></msub> <annotation encoding="application/x-tex">\mathbf{S}_{[i-1]}</annotation></semantics></math> </span></foreignObject></g></g><g stroke-width="0.8pt"><g stroke="#000000" fill="#BFDFDF"><path d="M 7.2 -74.8 h 48.59 v 31.5 h -48.59 Z"></path></g><g transform="matrix(1.0 0.0 0.0 1.0 11.81 -61.34)" fill="#000000" stroke="#000000"><foreignObject width="39.37" height="14.41" transform="matrix(1 0 0 -1 0 9.49)" overflow="visible" style="--fo_width :2.85em;--fo_height:0.69em;--fo_depth :0.36em;"><span style="width:2.85em;"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="{\mathbf{S}}_{[i]}"><semantics><msub><mi>𝐒</mi> <mrow><mo stretchy="false">[</mo><mi>i</mi><mo stretchy="false">]</mo></mrow></msub> <annotation encoding="application/x-tex">{\mathbf{S}}_{[i]}</annotation></semantics></math> </span></foreignObject></g></g><g stroke-width="0.8pt" fill="#000000" stroke="#000000" transform="matrix(1.0 0.0 0.0 1.0 91.27 -82.2)"><foreignObject width="10.38" height="12.45" transform="matrix(1 0 0 -1 0 9.69)" overflow="visible" style="--fo_width :0.75em;--fo_height:0.7em;--fo_depth :0.2em;"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\ldots"><semantics><mi mathvariant="normal">…</mi> <annotation encoding="application/x-tex">\ldots</annotation></semantics></math></foreignObject></g> <g stroke-width="0.8pt"><g stroke="#000000" fill="#BFDFDF"><path d="M 143.03 -74.8 h 52.53 v 31.5 h -52.53 Z"></path></g><g transform="matrix(1.0 0.0 0.0 1.0 147.64 -61.34)" fill="#000000" stroke="#000000"><foreignObject width="43.31" height="14.41" transform="matrix(1 0 0 -1 0 9.49)" overflow="visible" style="--fo_width :3.13em;--fo_height:0.69em;--fo_depth :0.36em;"><span style="width:3.13em;"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="{\mathbf{S}}_{[\scriptscriptstyle i+m-1]}"><semantics><msub><mi>𝐒</mi> <mrow><mo stretchy="false">[</mo><mrow><mrow><mi mathsize="0.710em">i</mi> <mo mathsize="0.710em">+</mo> <mi mathsize="0.710em">m</mi></mrow> <mo mathsize="0.710em">−</mo> <mn mathsize="0.710em">1</mn></mrow><mo maxsize="0.710em" minsize="0.710em">]</mo></mrow></msub> <annotation encoding="application/x-tex">{\mathbf{S}}_{[\scriptscriptstyle i+m-1]}</annotation></semantics></math> </span></foreignObject></g></g><g stroke-width="0.8pt"><g stroke="#000000" fill="#F5D9E2"><path d="M 302.47 -74.8 h 48.59 v 31.5 h -48.59 Z"></path></g><g transform="matrix(1.0 0.0 0.0 1.0 307.09 -61.34)" fill="#000000" stroke="#000000"><foreignObject width="39.37" height="14.41" transform="matrix(1 0 0 -1 0 9.49)" overflow="visible" style="--fo_width :2.85em;--fo_height:0.69em;--fo_depth :0.36em;"><span style="width:2.85em;"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="{\mathbf{S}}_{[i+m]}"><semantics><msub><mi>𝐒</mi> <mrow><mo stretchy="false">[</mo><mrow><mi>i</mi> <mo>+</mo> <mi>m</mi></mrow><mo stretchy="false">]</mo></mrow></msub> <annotation encoding="application/x-tex">{\mathbf{S}}_{[i+m]}</annotation></semantics></math> </span></foreignObject></g></g><g stroke-width="0.8pt"><g stroke="#DF809F" fill="#F5D9E2"><path d="M -160.52 -14.76 h 45.44 v 29.53 h -45.44 Z"></path></g><g transform="matrix(1.0 0.0 0.0 1.0 -155.91 -3.34)" fill="#000000" stroke="#000000"><foreignObject width="36.22" height="16.52" transform="matrix(1 0 0 -1 0 11.6)" overflow="visible" style="--fo_width :2.62em;--fo_height:0.84em;--fo_depth :0.36em;"><span style="width:2.62em;"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="{\mathbf{O}}^{\text{inter}}_{[i]}"><semantics><msubsup><mi>𝐎</mi> <mrow><mo stretchy="false">[</mo><mi>i</mi><mo stretchy="false">]</mo></mrow> <mtext>inter</mtext></msubsup> <annotation encoding="application/x-tex">{\mathbf{O}}^{\text{inter}}_{[i]}</annotation></semantics></math> </span></foreignObject></g></g><g stroke-width="0.8pt"><g stroke="#DF809F" fill="#F5D9E2"><path d="M 8.77 -26.57 h 45.44 v 29.53 h -45.44 Z"></path></g><g transform="matrix(1.0 0.0 0.0 1.0 13.39 -15.15)" fill="#000000" stroke="#000000"><foreignObject width="36.22" height="16.52" transform="matrix(1 0 0 -1 0 11.6)" overflow="visible" style="--fo_width :2.62em;--fo_height:0.84em;--fo_depth :0.36em;"><span style="width:2.62em;"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="{\mathbf{O}}^{\text{intra}}_{[i]}"><semantics><msubsup><mi>𝐎</mi> <mrow><mo stretchy="false">[</mo><mi>i</mi><mo stretchy="false">]</mo></mrow> <mtext>intra</mtext></msubsup> <annotation encoding="application/x-tex">{\mathbf{O}}^{\text{intra}}_{[i]}</annotation></semantics></math> </span></foreignObject></g></g><g stroke-width="0.8pt"><g stroke="#DF809F" fill="#F5D9E2"><path d="M 134.76 -14.76 h 45.44 v 29.53 h -45.44 Z"></path></g><g transform="matrix(1.0 0.0 0.0 1.0 139.37 -3.34)" fill="#000000" stroke="#000000"><foreignObject width="36.22" height="16.52" transform="matrix(1 0 0 -1 0 11.6)" overflow="visible" style="--fo_width :2.62em;--fo_height:0.84em;--fo_depth :0.36em;"><span style="width:2.62em;"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="{\mathbf{O}}^{\text{inter}}_{[i+m]}"><semantics><msubsup><mi>𝐎</mi> <mrow><mo stretchy="false">[</mo><mrow><mi>i</mi> <mo>+</mo> <mi>m</mi></mrow><mo stretchy="false">]</mo></mrow> <mtext>inter</mtext></msubsup> <annotation encoding="application/x-tex">{\mathbf{O}}^{\text{inter}}_{[i+m]}</annotation></semantics></math> </span></foreignObject></g></g><g stroke-width="0.8pt"><g stroke="#DF809F" fill="#F5D9E2"><path d="M 304.05 -26.57 h 45.44 v 29.53 h -45.44 Z"></path></g><g transform="matrix(1.0 0.0 0.0 1.0 308.66 -15.15)" fill="#000000" stroke="#000000"><foreignObject width="36.22" height="16.52" transform="matrix(1 0 0 -1 0 11.6)" overflow="visible" style="--fo_width :2.62em;--fo_height:0.84em;--fo_depth :0.36em;"><span style="width:2.62em;"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="{\mathbf{O}}^{\text{intra}}_{[i+m]}"><semantics><msubsup><mi>𝐎</mi> <mrow><mo stretchy="false">[</mo><mrow><mi>i</mi> <mo>+</mo> <mi>m</mi></mrow><mo stretchy="false">]</mo></mrow> <mtext>intra</mtext></msubsup> <annotation encoding="application/x-tex">{\mathbf{O}}^{\text{intra}}_{[i+m]}</annotation></semantics></math> </span></foreignObject></g></g><g stroke-width="0.8pt"><g stroke="#FFB366" fill="#FFF2E6"><path d="M -61.7 -110.24 h 44.66 v 23.64 h -44.66 Z"></path></g><g transform="matrix(1.0 0.0 0.0 1.0 -57.09 -100.71)" fill="#000000" stroke="#000000"><foreignObject width="35.43" height="14.41" transform="matrix(1 0 0 -1 0 9.49)" overflow="visible" style="--fo_width :2.56em;--fo_height:0.69em;--fo_depth :0.36em;"><span style="width:2.56em;"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="{\mathbf{Q}}_{[i]}"><semantics><msub><mi>𝐐</mi> <mrow><mo stretchy="false">[</mo><mi>i</mi><mo stretchy="false">]</mo></mrow></msub> <annotation encoding="application/x-tex">{\mathbf{Q}}_{[i]}</annotation></semantics></math> </span></foreignObject></g></g><g stroke-width="0.8pt"><g stroke="#FFB366" fill="#FFF2E6"><path d="M -57.2 -149.61 h 75.03 v 23.64 h -75.03 Z M -19.46 -125.98 L -19.46 -149.61"></path></g><g transform="matrix(1.0 0.0 0.0 1.0 -52.59 -140.08)" fill="#000000" stroke="#000000"><g transform="matrix(1 0 0 -1 0 9.49)"><g transform="matrix(1 0 0 1 0 9.49)"><g transform="matrix(1 0 0 -1 0 0)"><foreignObject width="27.96" height="14.41" transform="matrix(1 0 0 -1 0 9.49)" overflow="visible" style="--fo_width :2.02em;--fo_height:0.69em;--fo_depth :0.36em;"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="{\mathbf{K}}_{[i]}"><semantics><msub><mi>𝐊</mi> <mrow><mo stretchy="false">[</mo><mi>i</mi><mo stretchy="false">]</mo></mrow></msub> <annotation encoding="application/x-tex">{\mathbf{K}}_{[i]}</annotation></semantics></math> </foreignObject></g></g></g></g><g stroke="#FFB366" fill="#FFF2E6" transform="matrix(1.0 0.0 0.0 1.0 -14.3 -140.08)"><g transform="matrix(1 0 0 -1 0 9.49)"><g transform="matrix(1 0 0 1 0 9.49)"><g transform="matrix(1 0 0 -1 0 0)"><foreignObject width="27.51" height="14.41" transform="matrix(1 0 0 -1 0 9.49)" overflow="visible" style="--fo_width :1.99em;--fo_height:0.69em;--fo_depth :0.36em;"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="{\mathbf{V}}_{[i]}"><semantics><msub><mi>𝐕</mi> <mrow><mo stretchy="false">[</mo><mi>i</mi><mo stretchy="false">]</mo></mrow></msub> <annotation encoding="application/x-tex">{\mathbf{V}}_{[i]}</annotation></semantics></math> </foreignObject></g></g></g></g></g><g stroke-width="0.8pt"><g stroke="#FFB366" fill="#FFF2E6"><path d="M 213.89 -113.8 h 44.66 v 30.75 h -44.66 Z"></path></g><g transform="matrix(1.0 0.0 0.0 1.0 218.5 -87.66)" fill="#000000" stroke="#000000"><foreignObject width="35.43" height="21.52" transform="matrix(1 0 0 -1 0 0)" overflow="visible" style="--fo_width :2.56em;--fo_height:0em;--fo_depth :1.56em;"><span style="width:2.56em;"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="{\mathbf{Q}}_{[\scriptscriptstyle i+m]}"><semantics><msub><mi>𝐐</mi> <mrow><mo stretchy="false">[</mo><mrow><mi mathsize="0.710em">i</mi> <mo mathsize="0.710em">+</mo> <mi mathsize="0.710em">m</mi></mrow><mo maxsize="0.710em" minsize="0.710em">]</mo></mrow></msub> <annotation encoding="application/x-tex">{\mathbf{Q}}_{[\scriptscriptstyle i+m]}</annotation></semantics></math> </span></foreignObject></g></g><g stroke-width="0.8pt"><g stroke="#FFB366" fill="#FFF2E6"><path d="M 230.91 -149.61 h 105.1 v 23.64 h -105.1 Z M 283.69 -125.98 L 283.69 -149.61"></path></g><g transform="matrix(1.0 0.0 0.0 1.0 235.53 -140.08)" fill="#000000" stroke="#000000"><g transform="matrix(1 0 0 -1 0 9.49)"><g transform="matrix(1 0 0 1 0 9.49)"><g transform="matrix(1 0 0 -1 0 0)"><foreignObject width="42.99" height="14.41" transform="matrix(1 0 0 -1 0 9.49)" overflow="visible" style="--fo_width :3.11em;--fo_height:0.69em;--fo_depth :0.36em;"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="{\mathbf{K}}_{[\scriptscriptstyle i+m]}"><semantics><msub><mi>𝐊</mi> <mrow><mo stretchy="false">[</mo><mrow><mi mathsize="0.710em">i</mi> <mo mathsize="0.710em">+</mo> <mi mathsize="0.710em">m</mi></mrow><mo maxsize="0.710em" minsize="0.710em">]</mo></mrow></msub> <annotation encoding="application/x-tex">{\mathbf{K}}_{[\scriptscriptstyle i+m]}</annotation></semantics></math> </foreignObject></g></g></g></g><g stroke="#FFB366" fill="#FFF2E6" transform="matrix(1.0 0.0 0.0 1.0 288.85 -140.08)"><g transform="matrix(1 0 0 -1 0 9.49)"><g transform="matrix(1 0 0 1 0 9.49)"><g transform="matrix(1 0 0 -1 0 0)"><foreignObject width="42.55" height="14.41" transform="matrix(1 0 0 -1 0 9.49)" overflow="visible" style="--fo_width :3.08em;--fo_height:0.69em;--fo_depth :0.36em;"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="{\mathbf{V}}_{[\scriptscriptstyle i+m]}"><semantics><msub><mi>𝐕</mi> <mrow><mo stretchy="false">[</mo><mrow><mi mathsize="0.710em">i</mi> <mo mathsize="0.710em">+</mo> <mi mathsize="0.710em">m</mi></mrow><mo maxsize="0.710em" minsize="0.710em">]</mo></mrow></msub> <annotation encoding="application/x-tex">{\mathbf{V}}_{[\scriptscriptstyle i+m]}</annotation></semantics></math> </foreignObject></g></g></g></g></g><g stroke-width="0.8pt"><g stroke="#000000" fill="#FFF2E6"><path d="M -171.93 -211.3 h 28.91 v 28.91 h -28.91 Z"></path></g><g transform="matrix(1.0 0.0 0.0 1.0 -167.32 -206.69)" fill="#000000" stroke="#000000"></g><g transform="matrix(1.0 0.0 0.0 1.0 -137.31 -201.65)" fill="#000000" stroke="#000000"><foreignObject width="95.44" height="9.61" transform="matrix(1 0 0 -1 0 9.61)" overflow="visible" style="--fo_width :6.9em;--fo_height:0.69em;--fo_depth :0em;">Load from HBM</foreignObject></g></g> <g stroke-width="0.8pt"><g stroke="#000000" fill="#F5D9E2"><path d="M 5.23 -211.3 h 28.91 v 28.91 h -28.91 Z"></path></g><g transform="matrix(1.0 0.0 0.0 1.0 9.84 -206.69)" fill="#000000" stroke="#000000"></g><g transform="matrix(1.0 0.0 0.0 1.0 39.86 -201.65)" fill="#000000" stroke="#000000"><foreignObject width="82.37" height="9.61" transform="matrix(1 0 0 -1 0 9.61)" overflow="visible" style="--fo_width :5.95em;--fo_height:0.69em;--fo_depth :0em;">Store to HBM</foreignObject></g></g> <g stroke-width="0.8pt"><g stroke="#000000" fill="#BFDFDF"><path d="M 150.9 -211.3 h 28.91 v 28.91 h -28.91 Z"></path></g><g transform="matrix(1.0 0.0 0.0 1.0 155.51 -206.69)" fill="#000000" stroke="#000000"></g><g transform="matrix(1.0 0.0 0.0 1.0 185.53 -200.31)" fill="#000000" stroke="#000000"><foreignObject width="45.7" height="12.3" transform="matrix(1 0 0 -1 0 9.61)" overflow="visible" style="--fo_width :3.3em;--fo_height:0.69em;--fo_depth :0.19em;">On-chip</foreignObject></g></g> <g stroke-width="0.8pt"><g transform="matrix(1.0 0.0 0.0 1.0 -167.32 -171.26)" fill="#000000" stroke="#000000"></g><g transform="matrix(1.0 0.0 0.0 1.0 -97.94 -165.35)" fill="#000000" stroke="#000000"></g></g><g stroke-width="0.8pt" stroke-dasharray="3.0pt,3.0pt" stroke-dashoffset="0.0pt"><path d="M 267.72 -196.85 L 295.62 -196.85" style="fill:none"></path><g transform="matrix(1.0 0.0 0.0 1.0 295.62 -196.85)"><path d="M 3.6 0 L -2.16 2.88 L 0 0 L -2.16 -2.88" style="stroke:none"></path></g></g><g stroke-width="0.8pt" fill="#000000" stroke="#000000" transform="matrix(1.0 0.0 0.0 1.0 307.09 -200.31)"><foreignObject width="78.74" height="12.3" transform="matrix(1 0 0 -1 0 9.61)" overflow="visible" style="--fo_width :5.69em;--fo_height:0.69em;--fo_depth :0.19em;"><span style="width:5.69em;">Sequential</span></foreignObject></g><g stroke-width="0.8pt" stroke-dasharray="3.0pt,3.0pt" stroke-dashoffset="0.0pt"><path d="M -93.26 -59.06 L 3.05 -59.06" style="fill:none"></path><g transform="matrix(1.0 0.0 0.0 1.0 3.05 -59.06)"><path d="M 3.6 0 L -2.16 2.88 L 0 0 L -2.16 -2.88" style="stroke:none"></path></g></g><g stroke-width="0.8pt" stroke-dasharray="3.0pt,3.0pt" stroke-dashoffset="0.0pt"><path d="M 196.11 -59.06 L 298.32 -59.06" style="fill:none"></path><g transform="matrix(1.0 0.0 0.0 1.0 298.32 -59.06)"><path d="M 3.6 0 L -2.16 2.88 L 0 0 L -2.16 -2.88" style="stroke:none"></path></g></g><g stroke-width="0.8pt" stroke-dasharray="3.0pt,3.0pt" stroke-dashoffset="0.0pt"><path d="M -196.85 -59.06 L -146.56 -59.06" style="fill:none"></path><g transform="matrix(1.0 0.0 0.0 1.0 -146.56 -59.06)"><path d="M 3.6 0 L -2.16 2.88 L 0 0 L -2.16 -2.88" style="stroke:none"></path></g></g><g stroke-width="0.8pt" stroke-dasharray="3.0pt,3.0pt" stroke-dashoffset="0.0pt"><path d="M 110.24 -59.06 L 138.88 -59.06" style="fill:none"></path><g transform="matrix(1.0 0.0 0.0 1.0 138.88 -59.06)"><path d="M 3.6 0 L -2.16 2.88 L 0 0 L -2.16 -2.88" style="stroke:none"></path></g></g><g stroke-width="0.8pt" stroke-dasharray="3.0pt,3.0pt" stroke-dashoffset="0.0pt"><path d="M 56.35 -59.06 L 75.14 -59.06" style="fill:none"></path><g transform="matrix(1.0 0.0 0.0 1.0 75.14 -59.06)"><path d="M 3.6 0 L -2.16 2.88 L 0 0 L -2.16 -2.88" style="stroke:none"></path></g></g><g stroke-width="0.8pt" stroke-dasharray="3.0pt,3.0pt" stroke-dashoffset="0.0pt"><path d="M 351.62 -59.06 L 394.04 -59.06" style="fill:none"></path><g transform="matrix(1.0 0.0 0.0 1.0 394.04 -59.06)"><path d="M 3.6 0 L -2.16 2.88 L 0 0 L -2.16 -2.88" style="stroke:none"></path></g></g></g></svg>

Figure 2: Interleaved state-saving pattern used in our training kernels. For every $m$ chunks, only the state of the last chunk (${\mathbf{S}}_{[m]}$) is stored in HBM. The intermediate chunk states are recomputed on-chip as needed, using the most recent stored state ${\mathbf{S}}_{[i-1]}$ from the previous group of $m$ chunks. This approach make it more flexible to balances memory I/O cost and matmul computation.

Our linear attention kernels incorporate two key optimizations—fused operations and flexible state saving. To reduce memory I/O cost, we first fuse the computation of the feature map $\phi({\bm{k}}_{t},{\bm{v}}_{t})$ directly within the kernel, eliminating the need to store intermediate values in HBM and transfer them to VMEM, thereby lowering memory overhead.

Beyond reducing memory I/O, another critical factor for efficiency is maximizing the overlap between memory access and computation. In chunk-wise training, there is an inherent tradeoff in how much intermediate state (${\mathbf{S}}_{[i]}$) to store: storing more leads to faster backward passes (which depend on these states) but comes at a higher memory cost. Typically, chunk size is tuned according to two strategies – either storing all intermediate states or recomputing them during the backward pass. In this work, we introduce a more flexible state saving scheme. As illustrated in Figure 2, we store states every $m$ chunks while recomputing the states for the intermediate chunks. This hybrid approach allows compilers (e.g., using Triton or Pallas) to more effectively schedule operations and overlap memory I/O with computation. By enumerative search over $m$ and chunk size, the best configuration can achieve around 15% speedup compared with $m=1$ and a typical chunk size 256.

### 3.3 Local-Global Attention Models

We evaluate RAttention within a local-global attention framework that stacks multiple blocks in a repeating pattern of \[local, local, local, global\]. <sup>4</sup> The computation in each block follows:

$$
\displaystyle{\mathbf{Y}}^{(l)}=\operatorname{Attention}^{(l)}(\text{RMS}({\mathbf{X}}^{(l)}))+{\mathbf{X}}^{(l)}\
$$
 
$$
\displaystyle{\mathbf{X}}^{(l+1)}=\operatorname{SwiGLU}(\text{RMS}({\mathbf{Y}}^{(l)}))+{\mathbf{X}}^{(l)},
$$

where $l$ indexes the layer starting from $1$. The attention module $\operatorname{Attention}^{(l)}$ is instantiated as RAttention when $l\mod 4\neq 0$, and as standard global attention otherwise. Following the LLaMA architecture [^32], we adopt pre-norm and SwiGLU feed-forward layers.

While the combination of linear attention and sliding window attention was previously investigated by [^2], their models, despite outperforming other purely linear architectures, did not achieve performance parity with full-attention Transformers, even at the 1B scale. RAttention local-global models, in contrast, successfully bridges this performance gap.

## 4 Experiment

### 4.1 Main Results

The central question we address in our experiments is: what is the Pareto trade-off between downstream performance and efficiency (i.e., sliding window size) when replacing SWA with RAttention? Our evaluation consists of two stages:

1. we establish the trade-off curve between performance and window size through extensive experiments at the 3B scale. From this curve, we identify the minimal window size range for RAttention that achieves performance comparable to full attention;
2. we further validate these window sizes in settings where models are scaled by in three axes: the number of tokens, the model size or the pretraining context length.

The short answer to the question is that RAttention with a window size of $\geq 512$ can reliably match (or exceed) the performance of full attention across different settings.

#### Pretraining Setup

All the models are implemented in Jax [^6] and trained on v6e Cloud TPU clusters. We use 512/1024 chips provided as 2/4 $\times$ 256 chip slices to train 3B/12B models, respectively. Data parallelism along with activation recomputation is used for distributed training. We use a variant of RMSProp [^31] with momentum as the optimizer. We use our internal web-crawled data with a mixture similar to Llama models [^32]. For evaluation, we consider a set of standard tasks: SciQ [^34], TriviaQA [^16], WebQ [^3], MMLU [^13], GSM8k [^9] LAMBADA [^22], PiQA [^5], HellaSwag [^39], WinoGrande [^25], ARC-easy (ARC-E) and ARC-challenge (ARC-C) [^8].

| Parameters | 3B | 12B |
| --- | --- | --- |
| d\_model | 2048 | 5120 |
| layers | 56 | 40 |
| num heads | 16 | 40 |
| num kv heads | 4 | 8 |
| qk-norm | yes | yes |
| head type | GQA | GQA |
| head size | 128 | 128 |
| non-linearity | GeGLU | GeGLU |
| feedforward dim | 6656 | 16384 |
| pre-norm | yes | yes |
| global-local ratio | 1:3 | 1:3 |

Figure 3: Model specifications of our 3B and 12B models.

#### Model Setup

Our base model is a Transformer model with interleaved global-local attention layers. The mixing ratio of global and local attention is always fixed at 1:3 with the first three layer being local and the final layer being global layer. The global layers do not use rotary positional embedding [^26] whereas the local layers still use it with $\theta=5e5$. Such interleaved patterns are better in long context generalization based on our experience and literature [^35]. Apart from the positional embeddings, all the attention hyperparameters are shared between local and global attention.

In RAttention, all the attention hyperparameters for residual linear attention are shared from SWA (i.e., number of heads, number of kv heads, head\_dim) since query/key/value projections are directly inherited from SWA for parameter efficiency. We emphasize again that RAttention does not require extra parameters compared with SWA or full attention.

<svg id="S4.F4.pic1" height="224.49" overflow="visible" version="1.1" viewBox="0 0 486.3 224.49" width="486.3"><g transform="translate(0,224.49) matrix(1 0 0 -1 0 0) translate(59.74,0) translate(0,37.68) matrix(1.0 0.0 0.0 1.0 -59.74 -37.68)" fill="#000000" stroke="#000000" stroke-width="0.4pt"><g transform="matrix(1 0 0 1 0 0) translate(-165.04,0) translate(0,37.68)"><g fill="#BFBFBF" stroke="#BFBFBF" stroke-width="0.2pt" color="#BFBFBF"><path d="M 230.14 0 L 230.14 177.73 M 287.68 0 L 287.68 177.73 M 345.21 0 L 345.21 177.73 M 402.75 0 L 402.75 177.73 M 460.28 0 L 460.28 177.73 M 517.82 0 L 517.82 177.73 M 575.35 0 L 575.35 177.73 M 632.89 0 L 632.89 177.73" style="fill:none"></path></g><g fill="#BFBFBF" stroke="#BFBFBF" stroke-width="0.2pt" color="#BFBFBF"><path d="M 224.78 41.02 L 642.52 41.02 M 224.78 109.38 L 642.52 109.38 M 224.78 177.73 L 642.52 177.73" style="fill:none"></path></g><g stroke-width="0.2pt" fill="#808080" stroke="#808080" color="#808080"><path d="M 230.14 0 L 230.14 5.91 M 287.68 0 L 287.68 5.91 M 345.21 0 L 345.21 5.91 M 402.75 0 L 402.75 5.91 M 460.28 0 L 460.28 5.91 M 517.82 0 L 517.82 5.91 M 575.35 0 L 575.35 5.91 M 632.89 0 L 632.89 5.91 M 230.14 177.73 L 230.14 171.83 M 287.68 177.73 L 287.68 171.83 M 345.21 177.73 L 345.21 171.83 M 402.75 177.73 L 402.75 171.83 M 460.28 177.73 L 460.28 171.83 M 517.82 177.73 L 517.82 171.83 M 575.35 177.73 L 575.35 171.83 M 632.89 177.73 L 632.89 171.83" style="fill:none"></path></g><g stroke-width="0.2pt" fill="#808080" stroke="#808080" color="#808080"><path d="M 224.78 41.02 L 230.69 41.02 M 224.78 109.38 L 230.69 109.38 M 224.78 177.73 L 230.69 177.73 M 642.52 41.02 L 636.62 41.02 M 642.52 109.38 L 636.62 109.38 M 642.52 177.73 L 636.62 177.73" style="fill:none"></path></g><g stroke="#000000" fill="#000000" stroke-width="0.4pt"><path d="M 224.78 0 L 224.78 177.73 L 642.52 177.73 L 642.52 0 L 224.78 0 Z" style="fill:none"></path><g transform="matrix(1.0 0.0 0.0 1.0 226.68 -13.96)" fill="#000000" stroke="#000000"><foreignObject width="6.92" height="9.07" transform="matrix(1 0 0 -1 0 9.07)" overflow="visible" style="--fo_width :0.5em;--fo_height:0.66em;--fo_depth :0em;">0</foreignObject></g> <g transform="matrix(1.0 0.0 0.0 1.0 280.76 -13.96)" fill="#000000" stroke="#000000"><foreignObject width="13.84" height="9.07" transform="matrix(1 0 0 -1 0 9.07)" overflow="visible" style="--fo_width :1em;--fo_height:0.66em;--fo_depth :0em;">32</foreignObject></g> <g transform="matrix(1.0 0.0 0.0 1.0 338.29 -13.96)" fill="#000000" stroke="#000000"><foreignObject width="13.84" height="9.07" transform="matrix(1 0 0 -1 0 9.07)" overflow="visible" style="--fo_width :1em;--fo_height:0.66em;--fo_depth :0em;">64</foreignObject></g> <g transform="matrix(1.0 0.0 0.0 1.0 392.37 -13.96)" fill="#000000" stroke="#000000"><foreignObject width="20.76" height="9.07" transform="matrix(1 0 0 -1 0 9.07)" overflow="visible" style="--fo_width :1.5em;--fo_height:0.66em;--fo_depth :0em;">128</foreignObject></g> <g transform="matrix(1.0 0.0 0.0 1.0 449.9 -13.96)" fill="#000000" stroke="#000000"><foreignObject width="20.76" height="9.07" transform="matrix(1 0 0 -1 0 9.07)" overflow="visible" style="--fo_width :1.5em;--fo_height:0.66em;--fo_depth :0em;">256</foreignObject></g> <g transform="matrix(1.0 0.0 0.0 1.0 507.44 -13.96)" fill="#000000" stroke="#000000"><foreignObject width="20.76" height="9.07" transform="matrix(1 0 0 -1 0 9.07)" overflow="visible" style="--fo_width :1.5em;--fo_height:0.66em;--fo_depth :0em;">512</foreignObject></g> <g transform="matrix(1.0 0.0 0.0 1.0 561.52 -13.96)" fill="#000000" stroke="#000000"><foreignObject width="27.67" height="9.07" transform="matrix(1 0 0 -1 0 9.07)" overflow="visible" style="--fo_width :2em;--fo_height:0.66em;--fo_depth :0em;">1024</foreignObject></g> <g transform="matrix(1.0 0.0 0.0 1.0 619.05 -13.96)" fill="#000000" stroke="#000000"><foreignObject width="27.67" height="9.07" transform="matrix(1 0 0 -1 0 9.07)" overflow="visible" style="--fo_width :2em;--fo_height:0.66em;--fo_depth :0em;">2048</foreignObject></g> <g transform="matrix(1.0 0.0 0.0 1.0 192.99 36.56)" fill="#000000" stroke="#000000"><foreignObject width="26.91" height="8.92" transform="matrix(1 0 0 -1 0 8.92)" overflow="visible" style="--fo_width :1.94em;--fo_height:0.64em;--fo_depth :0em;"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="0.35"><semantics><mn>0.35</mn> <annotation encoding="application/x-tex">0.35</annotation></semantics></math></foreignObject></g> <g transform="matrix(1.0 0.0 0.0 1.0 199.91 104.92)" fill="#000000" stroke="#000000"><foreignObject width="19.99" height="8.92" transform="matrix(1 0 0 -1 0 8.92)" overflow="visible" style="--fo_width :1.44em;--fo_height:0.64em;--fo_depth :0em;"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="0.4"><semantics><mn>0.4</mn> <annotation encoding="application/x-tex">0.4</annotation></semantics></math></foreignObject></g> <g transform="matrix(1.0 0.0 0.0 1.0 192.99 173.28)" fill="#000000" stroke="#000000"><foreignObject width="26.91" height="8.92" transform="matrix(1 0 0 -1 0 8.92)" overflow="visible" style="--fo_width :1.94em;--fo_height:0.64em;--fo_depth :0em;"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="0.45"><semantics><mn>0.45</mn> <annotation encoding="application/x-tex">0.45</annotation></semantics></math></foreignObject></g> <clipPath id="pgfcp9"><path d="M 224.78 0 L 642.52 0 L 642.52 177.73 L 224.78 177.73 Z"></path></clipPath><g clip-path="url(#pgfcp9)"><g stroke="#BF0040" fill="#BF0040" stroke-dasharray="3.0pt,3.0pt" stroke-dashoffset="0.0pt" stroke-width="1.2pt" color="#BF0040"><path d="M 230.14 65.63 L 632.89 65.63" style="fill:none"></path></g><g></g><g stroke="#0000FF" fill="#0000FF" stroke-width="1.2pt" color="#0000FF"><path d="M 632.89 66.99 L 575.35 27.34 L 517.82 19.14 L 460.28 10.94" style="fill:none"></path></g><g></g><g stroke="#FF0000" fill="#FF0000" stroke-width="1.2pt" color="#FF0000"><path d="M 575.35 164.06 L 517.82 139.45 L 460.28 46.48 L 402.75 75.2 L 345.21 24.61 L 287.68 27.34 L 230.14 8.2" style="fill:none"></path></g><g></g><g stroke="#BF8040" fill="#BF8040" stroke-dasharray="3.0pt,2.0pt,0.4pt,2.0pt" stroke-dashoffset="0.0pt" stroke-width="1.2pt" color="#BF8040"><path d="M 230.14 8.2 L 632.89 8.2" style="fill:none"></path></g><g></g></g><g stroke="#0000FF" fill="#0000FF" stroke-width="1.2pt" color="#0000FF"><path d="M 635.65 66.99 C 635.65 68.52 634.42 69.76 632.89 69.76 C 631.36 69.76 630.12 68.52 630.12 66.99 C 630.12 65.46 631.36 64.22 632.89 64.22 C 634.42 64.22 635.65 65.46 635.65 66.99 Z M 632.89 66.99" style="fill:none"></path><path d="M 578.12 27.34 C 578.12 28.87 576.88 30.11 575.35 30.11 C 573.83 30.11 572.59 28.87 572.59 27.34 C 572.59 25.82 573.83 24.58 575.35 24.58 C 576.88 24.58 578.12 25.82 578.12 27.34 Z M 575.35 27.34" style="fill:none"></path><path d="M 520.59 19.14 C 520.59 20.67 519.35 21.91 517.82 21.91 C 516.29 21.91 515.05 20.67 515.05 19.14 C 515.05 17.61 516.29 16.37 517.82 16.37 C 519.35 16.37 520.59 17.61 520.59 19.14 Z M 517.82 19.14" style="fill:none"></path><path d="M 463.05 10.94 C 463.05 12.47 461.81 13.7 460.28 13.7 C 458.75 13.7 457.51 12.47 457.51 10.94 C 457.51 9.41 458.75 8.17 460.28 8.17 C 461.81 8.17 463.05 9.41 463.05 10.94 Z M 460.28 10.94" style="fill:none"></path></g><g stroke="#FF0000" fill="#FF0000" stroke-width="1.2pt" color="#FF0000"><path d="M 578.12 164.06 C 578.12 165.59 576.88 166.83 575.35 166.83 C 573.83 166.83 572.59 165.59 572.59 164.06 C 572.59 162.53 573.83 161.3 575.35 161.3 C 576.88 161.3 578.12 162.53 578.12 164.06 Z M 575.35 164.06" style="fill:none"></path><path d="M 520.59 139.45 C 520.59 140.98 519.35 142.22 517.82 142.22 C 516.29 142.22 515.05 140.98 515.05 139.45 C 515.05 137.92 516.29 136.69 517.82 136.69 C 519.35 136.69 520.59 137.92 520.59 139.45 Z M 517.82 139.45" style="fill:none"></path><path d="M 463.05 46.48 C 463.05 48.01 461.81 49.25 460.28 49.25 C 458.75 49.25 457.51 48.01 457.51 46.48 C 457.51 44.96 458.75 43.72 460.28 43.72 C 461.81 43.72 463.05 44.96 463.05 46.48 Z M 460.28 46.48" style="fill:none"></path><path d="M 405.51 75.2 C 405.51 76.72 404.28 77.96 402.75 77.96 C 401.22 77.96 399.98 76.72 399.98 75.2 C 399.98 73.67 401.22 72.43 402.75 72.43 C 404.28 72.43 405.51 73.67 405.51 75.2 Z M 402.75 75.2" style="fill:none"></path><path d="M 347.98 24.61 C 347.98 26.14 346.74 27.38 345.21 27.38 C 343.68 27.38 342.44 26.14 342.44 24.61 C 342.44 23.08 343.68 21.84 345.21 21.84 C 346.74 21.84 347.98 23.08 347.98 24.61 Z M 345.21 24.61" style="fill:none"></path><path d="M 290.44 27.34 C 290.44 28.87 289.2 30.11 287.68 30.11 C 286.15 30.11 284.91 28.87 284.91 27.34 C 284.91 25.82 286.15 24.58 287.68 24.58 C 289.2 24.58 290.44 25.82 290.44 27.34 Z M 287.68 27.34" style="fill:none"></path><path d="M 232.91 8.2 C 232.91 9.73 231.67 10.97 230.14 10.97 C 228.61 10.97 227.37 9.73 227.37 8.2 C 227.37 6.67 228.61 5.44 230.14 5.44 C 231.67 5.44 232.91 6.67 232.91 8.2 Z M 230.14 8.2" style="fill:none"></path></g><g transform="matrix(1.0 0.0 0.0 1.0 396.44 -33.07)" fill="#000000" stroke="#000000"><foreignObject width="74.82" height="9.61" transform="matrix(1 0 0 -1 0 9.61)" overflow="visible" style="--fo_width :5.41em;--fo_height:0.69em;--fo_depth :0em;">Window Size</foreignObject></g> <g transform="matrix(0.0 1.0 -1.0 0.0 180.03 24.5)" fill="#000000" stroke="#000000"><foreignObject width="128.36" height="13.84" transform="matrix(1 0 0 -1 0 10.38)" overflow="visible" style="--fo_width :9.28em;--fo_height:0.75em;--fo_depth :0.25em;">MMLU Score (5-shot)</foreignObject></g> <g fill="#FFFFFF" stroke="#000000"><path d="M 233.42 103.33 h 214.14 v 70.57 h -214.14 Z"></path></g><g fill="#FFFFFF" stroke="#000000" transform="matrix(1.0 0.0 0.0 1.0 237.57 114.23)"><g transform="matrix(1 0 0 -1 0 56.9)"><g transform="matrix(1 0 0 1 0 8.13)"><g transform="matrix(1 0 0 -1 0 0) translate(0.83,0)" fill="#BF0040" stroke="#BF0040" stroke-dasharray="3.0pt,3.0pt" stroke-dashoffset="0.0pt" stroke-width="1.2pt" color="#BF0040"><path d="M 0 0 L 11.81 0 L 23.62 0" style="fill:none"></path></g><g transform="matrix(1 0 0 -1 25.56 0) translate(-0.28,0) matrix(1.0 0.0 0.0 1.0 3.04 -3.29)" fill="#000000" stroke="#000000"><foreignObject width="117.79" height="11.07" transform="matrix(1 0 0 -1 0 8.65)" overflow="visible" style="--fo_width :9.2em;--fo_height:0.68em;--fo_depth :0.19em;"><span style="font-size:90%;">Global Attention Only</span></foreignObject></g></g> <g transform="matrix(1 0 0 1 0 24.39)"><g transform="matrix(1 0 0 -1 0 0) translate(0.83,0)" fill="#0000FF" stroke="#0000FF" stroke-width="1.2pt" color="#0000FF"><path d="M 0 0 L 11.81 0 L 23.62 0" style="fill:none"></path><path d="M 14.58 0 C 14.58 1.53 13.34 2.77 11.81 2.77 C 10.28 2.77 9.04 1.53 9.04 0 C 9.04 -1.53 10.28 -2.77 11.81 -2.77 C 13.34 -2.77 14.58 -1.53 14.58 0 Z M 11.81 0" style="fill:none"></path></g><g transform="matrix(1 0 0 -1 25.56 0) translate(-0.28,0) matrix(1.0 0.0 0.0 1.0 3.04 -3.29)" fill="#000000" stroke="#000000"><foreignObject width="114.23" height="8.65" transform="matrix(1 0 0 -1 0 8.65)" overflow="visible" style="--fo_width :8.92em;--fo_height:0.68em;--fo_depth :0em;"><span style="font-size:90%;">Local-Global w. SWA</span></foreignObject></g></g> <g transform="matrix(1 0 0 1 0 40.65)"><g transform="matrix(1 0 0 -1 0 0) translate(0.83,0)" fill="#FF0000" stroke="#FF0000" stroke-width="1.2pt" color="#FF0000"><path d="M 0 0 L 11.81 0 L 23.62 0" style="fill:none"></path><path d="M 14.58 0 C 14.58 1.53 13.34 2.77 11.81 2.77 C 10.28 2.77 9.04 1.53 9.04 0 C 9.04 -1.53 10.28 -2.77 11.81 -2.77 C 13.34 -2.77 14.58 -1.53 14.58 0 Z M 11.81 0" style="fill:none"></path></g><g transform="matrix(1 0 0 -1 25.56 0) translate(-0.28,0) matrix(1.0 0.0 0.0 1.0 3.04 -3.29)" fill="#000000" stroke="#000000"><foreignObject width="146.25" height="8.65" transform="matrix(1 0 0 -1 0 8.65)" overflow="visible" style="--fo_width :11.43em;--fo_height:0.68em;--fo_depth :0em;"><span style="font-size:90%;">Local-Global w. RAttention</span></foreignObject></g></g> <g transform="matrix(1 0 0 1 0 56.91)"><g transform="matrix(1 0 0 -1 0 0) translate(0.83,0)" fill="#BF8040" stroke="#BF8040" stroke-dasharray="3.0pt,2.0pt,0.4pt,2.0pt" stroke-dashoffset="0.0pt" stroke-width="1.2pt" color="#BF8040"><path d="M 0 0 L 11.81 0 L 23.62 0" style="fill:none"></path></g><g transform="matrix(1 0 0 -1 25.56 0) translate(-0.28,0) matrix(1.0 0.0 0.0 1.0 3.04 -3.29)" fill="#000000" stroke="#000000"><foreignObject width="174.72" height="8.65" transform="matrix(1 0 0 -1 0 8.65)" overflow="visible" style="--fo_width :13.65em;--fo_height:0.68em;--fo_depth :0em;"><span style="font-size:90%;">Local-Global w. Linear Attention</span></foreignObject></g></g></g></g></g></g></g></svg>

Figure 4: Comparison of MMLU 5-shot performance scores across different window sizes at 3B scale with pretraining context length 4096. The horizontal purple dashed line represents the baseline using only global attention. The blue line shows Local-Global with sliding window attention (SWA), while the red line demonstrates the performance of Local-Global with RAttention. When window size $=0$, Local-Global with RAttention reduces to Local-Global with only linear attention.

#### Pareto Curve at 3B

We train 3B-parameter SWA and RAttention models using various sliding window sizes on 400B tokens with a batch size of 1024. A full attention model is also trained as a baseline. As shown in Figure 4, there is a clear tradeoff between window size and MMLU performance—smaller window sizes lead to reduced performance.

Importantly, RAttention models achieve a better tradeoff curve compared to SWA models. With a window size of $\geq 512$, RAttention already matches or even surpasses the performance of the full attention baseline. As we will show later, these gains persist across certain benchmarks when scaling up RAttention models. This observation is consistent with prior work suggesting that hybrid attention models can, in some cases, outperform standard Transformers [^23].

#### Selective Pretraining at 3B and 12B

We further verify the effectiveness of RAttention by scaling both the number of tokens, model parameters and context length.

| Metric | Full 8k | SWA 4K | SWA 2K | RAttn-512 |
| --- | --- | --- | --- | --- |
| Average (0/1-shot) | 62.67 | 62.64 | 62.18 | 62.72 |
| MMLU (5-shot) | 52.40 | 50.80 | 48.60 | 52.94 |
| GSM8K (8-shot) | 36.69 | 35.71 | 33.28 | 37.39 |

Figure 5: Main results at 12B scale with pretraining context length 8192. Performance of zero- and one-shot tasks are summarized in Average (0/1-shot).

First, we selectively pretrain 3B-parameter SWA and RAttention models on 2T tokens. <sup>5</sup> As shown in Table 1, RAttention with a window size of 512 outperforms SWA models using window sizes up to 2048. Second, we pretrain 12B-parameter SWA and RAttention models on 600B tokens and again evaluate performance across a range of window sizes. As shown in Table 2, RAttention with a window size of 512 continues to outperform SWA models, further validating its scalability. Finally, we assess RAttention with window size 512 in the setting of pretraining context length 8k and 600B tokens. The summary results shown in Table 5 indicates that RAttention remains strong compared with full attention models.

| Metric | Full 4k | SWA 2k | SWA 1k | SWA 512 | RAtt 1k | RAtt 512 |
| --- | --- | --- | --- | --- | --- | --- |
| ARC-C | 46.25 | 47.01 | 45.39 | 44.97 | 44.28 | 45.39 |
| ARC-E | 78.32 | 78.83 | 78.45 | 77.44 | 77.82 | 77.31 |
| HellaSwag | 56.61 | 56.02 | 56.10 | 56.21 | 56.43 | 56.43 |
| LAMBADA | 72.27 | 72.21 | 72.77 | 71.78 | 72.54 | 72.04 |
| PIQA | 78.56 | 78.24 | 78.56 | 78.89 | 78.45 | 78.24 |
| SciQ | 94.80 | 95.70 | 95.40 | 95.80 | 94.60 | 95.20 |
| WinoGrande | 68.98 | 68.82 | 68.67 | 71.03 | 71.27 | 70.80 |
| TriviaQA (1-shot) | 41.49 | 42.21 | 41.98 | 40.61 | 41.57 | 42.23 |
| WebQS (1-shot) | 20.37 | 19.34 | 20.37 | 21.70 | 18.31 | 18.55 |
| Average (0/1-shot) | 62.00 | 62.00 | 62.00 | 62.00 | 61.70 | 61.80 |
| MMLU (5-shot) | 56.70 | 55.70 | 55.70 | 55.50 | 56.22 | 55.62 |
| GSM8K (8-shot) | 34.19 | 29.49 | 32.90 | 27.98 | 33.13 | 33.74 |

Table 1: Main results at 3B scale with pretraining context length 4096.

| Metric | Full 4k | SWA 2K | RAttn-512 | RAttn-256 | RAttn-128 | RAttn-64 | RAttn-32 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ARC-C | 47.61 | 47.61 | 48.81 | 48.29 | 50.17 | 49.57 | 47.44 |
| ARC-E | 79.21 | 80.01 | 79.12 | 79.25 | 79.97 | 79.88 | 79.29 |
| HellaSwag | 57.76 | 57.99 | 57.72 | 57.79 | 58.11 | 58.25 | 58.06 |
| LAMBADA | 73.55 | 73.06 | 73.14 | 73.88 | 73.65 | 72.99 | 73.51 |
| PIQA | 79.27 | 80.03 | 78.51 | 79.54 | 79.00 | 79.38 | 79.43 |
| SciQ | 95.60 | 96.40 | 95.40 | 95.90 | 95.70 | 95.40 | 95.30 |
| WinoGrande | 70.17 | 72.69 | 71.03 | 71.74 | 72.38 | 71.90 | 70.09 |
| TriviaQA (1-shot) | 41.51 | 41.22 | 41.85 | 41.34 | 42.56 | 40.97 | 42.05 |
| WebQS (1-shot) | 21.21 | 21.65 | 24.66 | 22.64 | 22.00 | 20.42 | 20.23 |
| Average (0/1-shot) | 62.90 | 63.40 | 63.40 | 63.40 | 63.70 | 63.20 | 62.80 |
| MMLU (5-shot) | 52.96 | 49.52 | 51.50 | 53.77 | 51.74 | 49.17 | 49.66 |
| GSM8K (8-shot) | 30.33 | 24.26 | 29.57 | 29.34 | 26.61 | 30.93 | 26.61 |

Table 2: Main results at 12B scale with pretraining context length 4096.

### 4.2 Long-Context Results

We then evaluate zero-shot generalization capability on the RULER [^14] benchmark, testing them directly after pretraining at context length 4k. The average results are shown in Table 6.

| Model | 4k | 8k | 16k | 32k |
| --- | --- | --- | --- | --- |
| Full-4k | 80.38 | 2.89 | 0.00 | 0.08 |
| SWA-2k | 73.49 | 6.85 | 0.79 | 0.41 |
| RAttn-1k | 73.87 | 53.90 | 40.00 | 20.84 |
| RAttn-512 | 80.79 | 66.26 | 50.80 | 29.59 |

Figure 6: Average zero-shot RULER performance at 3B scale with pretraining context length 4K.

RAttention models generalize reasonably well beyond the 4k training context, whereas other models fail to do so. Interestingly, RAttention models with smaller window sizes exhibit better generalization. We believe that smaller window sizes place greater pressure on the local attention module to generalize beyond the local window during pretraining, resulting in improved length generalization.

### 4.3 Training Efficiency

We next demonstrate that the training efficiency of RAttention is not compromised. Specifically, we benchmark training efficiency in terms of step time using a batch size of 1024 and context lengths of 4k and 8k on TPU v5p-1024 (which is more suitable for larger-scale pretraining than v6e). As shown in Table 3, RAttention matches the training speed of both full attention and SWA models. Although RAttention introduces an additional RLA kernel to dispatch in the local attention layers compared to SWA, its small window size and the use of highly optimized RLA kernels collectively allow it to achieve comparable training speeds.

<table><thead><tr><th>Model Size</th><th>Pretraining Length</th><th>Full Attention</th><th>SWA</th><th>RAttention</th></tr></thead><tbody><tr><th rowspan="2">3B</th><th>4096</th><td>0.84 (4k)</td><td>0.80 (2k)</td><td>0.87 (512)</td></tr><tr><th>8192</th><td>1.20 (8k)</td><td>1.05 (4k)</td><td>1.08 (1k)</td></tr><tr><th rowspan="2">12B</th><th>4096</th><td>2.21 (4k)</td><td>2.10 (2k)</td><td>2.26 (512)</td></tr><tr><th>8192</th><td>3.99 (8k)</td><td>3.89 (4k)</td><td>3.97 (1k)</td></tr></tbody></table>

Table 3: Training speed comparison in terms of step time (seconds) at both 3B and 12B scale. Numbers in parentheses indicates window size of models.

### 4.4 Inference Efficiency

Next, we analyze the inference gains achievable by RAttention when using a smaller sliding window size. Since the prefilling stage has a similar efficiency profile to the training stage, we focus our analysis on the step time during the generation stage. In this phase, the attention modules are typically memory-bound, while the feedforward modules can be either compute-bound or memory-bound depending on the batch size. In general, the theoretical step time can be approximated by:

$$
T_{\text{step}}=\frac{B\times S_{\text{KV}}}{BW}+\max\left(\frac{2\times B\times P_{\text{count}}}{F},\frac{P_{\text{size}}}{BW}\right)
$$

where $T_{\text{step}}$ is the theoretical step time, $B$ is the batch size, $S_{\text{KV}}$ is the KV cache size, $BW$ is the total memory bandwidth, $P_{\text{count}}$ is the parameter count, $P_{\text{size}}$ is the parameter size (in bytes), and $F$ is the total FLOPs per second. As a case study, we apply this analysis to our 3B and 12B models using H100 hardware specifications and bfloat16 precision. Figure 7 shows the step time speedup as a function of context length across different batch sizes. As the batch size increases, the theoretical speedup of RAttention grows, reaching up to approximately 60%. Moreover, the speedup ultimately converges to the same point regardless of model size, since the KV cache size increasingly dominates the memory cost relative to the model parameter size.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2506.15545/assets/figures/llm_step_time_speedup_by_batch_size.png)

Figure 7: Step time speedup (%) of local-global models using RAttention (window size 512) compared to SWA (window size 4k). As batch size increases, the theoretical speedup of increases and converges, since the KV cache size increasingly dominates the memory cost relative to the model parameter size.

### 4.5 Ablation Study

| Metric | Full-4k | RAttn-512 | w. ReLU | w. Identity | w. Mamba2 | w. Hymba | \-SWA | \-GroupNorm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Average (0/1-shot) | 68.40 | 68.20 | 68.30 | 67.30 | 68.00 | 68.10 | 68.20 | 68.10 |
| MMLU (5-shot) | 36.80 | 42.22 | 37.90 | 35.70 | 39.21 | 36.53 | 32.60 | 39.81 |

Table 4: Ablation study on 3B models with 400B tokens and context length 4096.

We conducted ablation studies during 3B model training on the 400B token setting to identify the best configuration for RAttention. Results are shown in Table 4. For the feature map choice, we adopt $\mathrm{softmax}$ following [^40], which we found to outperform other alternatives such as ReLU and Identity.

We also experimented with adding more complex gating mechanism to linear attention, specifically using Mamba2 [^10] and Gated DeltaNet [^36]. However, no gain (or slight performance drop) is observed. We suspect that the introduction of more advanced linear models brings up optimization challenges: in our setup, the hybrid model already incorporates three token-mixing modules – full attention, sliding window attention, and residual linear attention – with the latter two sharing parameters. Introducing more complex forms of linear attention appears harder to optimize in this hybrid framework. On the other hand, we believe that it opens up a practical direction to explore for further research: designing better parameter-efficient linear models in the hybrid frameworks.

Additionally, we attempted to stack linear attention and sliding window attention across different heads, following the approach of Hymba [^12]. However, this configuration proved suboptimal, likely because both mechanisms exhibit strong recency bias toward tokens within the sliding window. We also verified that linear attention alone cannot retrain the performance. Finally, we found that applying group normalization improves the overall performance.

### 4.6 Related Work

Overall, there are two main approaches in designing efficient language models. The first relies on constant-memory modules, while the second focuses on leveraging sparsity in attention computation.

#### Constant-Memory Models and Their Hybrids

Recurrent models and SWA models serve as the primary building blocks for hybrid architectures due to their constant-memory properties. However, pure constant-memory models often underperform standard Transformers [^33]. Early hybrid models integrated these modules by interleaving them with standard attention. For example, Gemma2/Gemma3 [^30] [^29] alternate SWA with global attention, while Jamba [^19] and Samba [^23] combine Mamba with either SWA or global attention. Similarly, Griffin [^11] integrates gated linear recurrences with SWA. Another line of research seeks to fuse attention and recurrent mechanisms within the same layer. Megalodon [^21] uses a recurrent model to refine query/key representations within attention, while Hymba [^12] runs both Mamba and attention in parallel within each layer. Our approach, RAttention, advances this direction by achieving better parameter efficiency than Hymba—completely sharing parameters between linear attention and SWA.

#### Sparse Attention Models

To improve efficiency in long-context settings, another approach focuses on sparse attention, where the core challenge is designing effective sparsity patterns for KV-cache access. Early methods [^24] [^18] [^4] use non-parametric techniques (e.g., k-nearest neighbors, locality-sensitive hashing) to select relevant query-key pairs. More recently, parametric methods that learn sparsity patterns have proven effective, such as Native Sparse Attention [^38] and Mixture of Block Attention (MoBA) [^20]. With hardware-aligned implementations, these modules can be trained more efficiently than global attention. However, unlike linear attention and SWA, sparse attention still requires storing the KV cache for all context tokens, same as global attention. This raises an ongoing research question: Is it more effective to sparsely access context within attention (as in sparse attention) or to rely on recurrent modules for context compression? We leave the investigation of this question for future work.

## 5 Conclusion and Future Work

In this work, we explore using RAttention to replace sliding window attention in local-global models. Our results show that residual linear attention enables a substantial reduction in sliding window size—from 4K/8K to 512—without loss in performance. Through both analytical and empirical studies on training and inference efficiency, we demonstrate that RAttention offers significant advantage over SWA: training efficiency is maintained, while inference efficiency is significantly improved. In future work, we plan to focus on engineering efforts to realize the theoretical efficiency gains within current inference frameworks. Fineuning existing pretrained full attention models into RAttention models is also promising direction to explore.

[^1]: E. Akyürek, B. Wang, Y. Kim, and J. Andreas. In-context language learning: Architectures and algorithms, 2024.

[^2]: S. Arora, S. Eyuboglu, M. Zhang, A. Timalsina, S. Alberti, D. Zinsley, J. Zou, A. Rudra, and C. Ré. Simple linear attention language models balance the recall-throughput tradeoff, 2025.

[^3]: J. Berant, A. Chou, R. Frostig, and P. Liang. Semantic parsing on Freebase from question-answer pairs. In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, pages 1533–1544, Seattle, Washington, USA, Oct. 2013. Association for Computational Linguistics.

[^4]: A. Bertsch, U. Alon, G. Neubig, and M. R. Gormley. Unlimiformer: Long-range transformers with unlimited length input, 2023.

[^5]: Y. Bisk, R. Zellers, J. Gao, Y. Choi, et al. Piqa: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pages 7432–7439, 2020.

[^6]: J. Bradbury, R. Frostig, P. Hawkins, M. J. Johnson, C. Leary, D. Maclaurin, G. Necula, A. Paszke, J. VanderPlas, S. Wanderman-Milne, and Q. Zhang. JAX: composable transformations of Python+NumPy programs, 2018.

[^7]: R. Child, S. Gray, A. Radford, and I. Sutskever. Generating long sequences with sparse transformers, 2019.

[^8]: P. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick, and O. Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.

[^9]: K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

[^10]: T. Dao and A. Gu. Transformers are ssms: Generalized models and efficient algorithms through structured state space duality, 2024.

[^11]: S. De, S. L. Smith, A. Fernando, A. Botev, G. Cristian-Muraru, A. Gu, R. Haroun, L. Berrada, Y. Chen, S. Srinivasan, G. Desjardins, A. Doucet, D. Budden, Y. W. Teh, R. Pascanu, N. D. Freitas, and C. Gulcehre. Griffin: Mixing gated linear recurrences with local attention for efficient language models, 2024.

[^12]: X. Dong, Y. Fu, S. Diao, W. Byeon, Z. Chen, A. S. Mahabaleshwarkar, S.-Y. Liu, M. V. Keirsbilck, M.-H. Chen, Y. Suhara, Y. Lin, J. Kautz, and P. Molchanov. Hymba: A hybrid-head architecture for small language models, 2024.

[^13]: D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.

[^14]: C.-P. Hsieh, S. Sun, S. Kriman, S. Acharya, D. Rekesh, F. Jia, Y. Zhang, and B. Ginsburg. Ruler: What’s the real context size of your long-context language models?, 2024.

[^15]: A. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford, D. S. Chaplot, D. de las Casas, F. Bressand, G. Lengyel, G. Lample, L. Saulnier, L. R. Lavaud, M.-A. Lachaux, P. Stock, T. L. Scao, T. Lavril, T. Wang, T. Lacroix, and W. E. Sayed. Mistral 7b, 2023.

[^16]: M. Joshi, E. Choi, D. S. Weld, and L. Zettlemoyer. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. arXiv preprint arXiv:1705.03551, 2017.

[^17]: A. Katharopoulos, A. Vyas, N. Pappas, and F. Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention, 2020.

[^18]: N. Kitaev, Łukasz Kaiser, and A. Levskaya. Reformer: The efficient transformer, 2020.

[^19]: O. Lieber, B. Lenz, H. Bata, G. Cohen, J. Osin, I. Dalmedigos, E. Safahi, S. Meirom, Y. Belinkov, S. Shalev-Shwartz, O. Abend, R. Alon, T. Asida, A. Bergman, R. Glozman, M. Gokhman, A. Manevich, N. Ratner, N. Rozen, E. Shwartz, M. Zusman, and Y. Shoham. Jamba: A hybrid transformer-mamba language model, 2024.

[^20]: E. Lu, Z. Jiang, J. Liu, Y. Du, T. Jiang, C. Hong, S. Liu, W. He, E. Yuan, Y. Wang, Z. Huang, H. Yuan, S. Xu, X. Xu, G. Lai, Y. Chen, H. Zheng, J. Yan, J. Su, Y. Wu, N. Y. Zhang, Z. Yang, X. Zhou, M. Zhang, and J. Qiu. Moba: Mixture of block attention for long-context llms, 2025.

[^21]: X. Ma, X. Yang, W. Xiong, B. Chen, L. Yu, H. Zhang, J. May, L. Zettlemoyer, O. Levy, and C. Zhou. Megalodon: Efficient llm pretraining and inference with unlimited context length, 2024.

[^22]: D. Paperno, G. Kruszewski, A. Lazaridou, Q. N. Pham, R. Bernardi, S. Pezzelle, M. Baroni, G. Boleda, and R. Fernández. The lambada dataset: Word prediction requiring a broad discourse context. arXiv preprint arXiv:1606.06031, 2016.

[^23]: L. Ren, Y. Liu, Y. Lu, Y. Shen, C. Liang, and W. Chen. Samba: Simple hybrid state space models for efficient unlimited context language modeling, 2025.

[^24]: A. Roy, M. Saffar, A. Vaswani, and D. Grangier. Efficient content-based sparse attention with routing transformers, 2020.

[^25]: K. Sakaguchi, R. L. Bras, C. Bhagavatula, and Y. Choi. Winogrande: An adversarial winograd schema challenge at scale. Communications of the ACM, 64(9):99–106, 2021.

[^26]: J. Su, M. Ahmed, Y. Lu, S. Pan, W. Bo, and Y. Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

[^27]: Y. Sun, L. Dong, S. Huang, S. Ma, Y. Xia, J. Xue, J. Wang, and F. Wei. Retentive network: A successor to transformer for large language models. arXiv preprint arXiv:2307.08621, 2023.

[^28]: C. A. R. Team. Optimizing inference, 2024. Accessed: March 19, 2025.

[^29]: G. Team. Gemma 3 technical report, 2025.

[^30]: G. Team, M. Riviere, S. Pathak, P. G. Sessa, C. Hardin, S. Bhupatiraju, L. Hussenot, T. Mesnard, B. Shahriari, A. Ramé, J. Ferret, P. Liu, P. Tafti, A. Friesen, M. Casbon, S. Ramos, R. Kumar, C. L. Lan, S. Jerome, A. Tsitsulin, N. Vieillard, P. Stanczyk, S. Girgin, N. Momchev, M. Hoffman, S. Thakoor, J.-B. Grill, B. Neyshabur, O. Bachem, A. Walton, A. Severyn, A. Parrish, A. Ahmad, A. Hutchison, A. Abdagic, A. Carl, A. Shen, A. Brock, A. Coenen, A. Laforge, A. Paterson, B. Bastian, B. Piot, B. Wu, B. Royal, C. Chen, C. Kumar, C. Perry, C. Welty, C. A. Choquette-Choo, D. Sinopalnikov, D. Weinberger, D. Vijaykumar, D. Rogozińska, D. Herbison, E. Bandy, E. Wang, E. Noland, E. Moreira, E. Senter, E. Eltyshev, F. Visin, G. Rasskin, G. Wei, G. Cameron, G. Martins, H. Hashemi, H. Klimczak-Plucińska, H. Batra, H. Dhand, I. Nardini, J. Mein, J. Zhou, J. Svensson, J. Stanway, J. Chan, J. P. Zhou, J. Carrasqueira, J. Iljazi, J. Becker, J. Fernandez, J. van Amersfoort, J. Gordon, J. Lipschultz, J. Newlan, J. yeong Ji, K. Mohamed, K. Badola, K. Black, K. Millican, K. McDonell, K. Nguyen, K. Sodhia, K. Greene, L. L. Sjoesund, L. Usui, L. Sifre, L. Heuermann, L. Lago, L. McNealus, L. B. Soares, L. Kilpatrick, L. Dixon, L. Martins, M. Reid, M. Singh, M. Iverson, M. Görner, M. Velloso, M. Wirth, M. Davidow, M. Miller, M. Rahtz, M. Watson, M. Risdal, M. Kazemi, M. Moynihan, M. Zhang, M. Kahng, M. Park, M. Rahman, M. Khatwani, N. Dao, N. Bardoliwalla, N. Devanathan, N. Dumai, N. Chauhan, O. Wahltinez, P. Botarda, P. Barnes, P. Barham, P. Michel, P. Jin, P. Georgiev, P. Culliton, P. Kuppala, R. Comanescu, R. Merhej, R. Jana, R. A. Rokni, R. Agarwal, R. Mullins, S. Saadat, S. M. Carthy, S. Cogan, S. Perrin, S. M. R. Arnold, S. Krause, S. Dai, S. Garg, S. Sheth, S. Ronstrom, S. Chan, T. Jordan, T. Yu, T. Eccles, T. Hennigan, T. Kocisky, T. Doshi, V. Jain, V. Yadav, V. Meshram, V. Dharmadhikari, W. Barkley, W. Wei, W. Ye, W. Han, W. Kwon, X. Xu, Z. Shen, Z. Gong, Z. Wei, V. Cotruta, P. Kirk, A. Rao, M. Giang, L. Peran, T. Warkentin, E. Collins, J. Barral, Z. Ghahramani, R. Hadsell, D. Sculley, J. Banks, A. Dragan, S. Petrov, O. Vinyals, J. Dean, D. Hassabis, K. Kavukcuoglu, C. Farabet, E. Buchatskaya, S. Borgeaud, N. Fiedel, A. Joulin, K. Kenealy, R. Dadashi, and A. Andreev. Gemma 2: Improving open language models at a practical size, 2024.

[^31]: T. Tieleman and G. Hinton. Divide the gradient by a running average of its recent magnitude. coursera: Neural networks for machine learning. Technical report, 2017.

[^32]: H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

[^33]: A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin. Attention is all you need, 2023.

[^34]: J. Welbl, N. F. Liu, and M. Gardner. Crowdsourcing multiple choice science questions. arXiv preprint arXiv:1707.06209, 2017.

[^35]: B. Yang, B. Venkitesh, D. Talupuru, H. Lin, D. Cairuz, P. Blunsom, and A. Locatelli. Rope to nope and back again: A new hybrid attention strategy. arXiv preprint arXiv:2501.18795, 2025.

[^36]: S. Yang, J. Kautz, and A. Hatamizadeh. Gated delta networks: Improving mamba2 with delta rule. arXiv preprint arXiv:2412.06464, 2024.

[^37]: S. Yang, B. Wang, Y. Shen, R. Panda, and Y. Kim. Gated linear attention transformers with hardware-efficient training, 2024.

[^38]: J. Yuan, H. Gao, D. Dai, J. Luo, L. Zhao, Z. Zhang, Z. Xie, Y. X. Wei, L. Wang, Z. Xiao, Y. Wang, C. Ruan, M. Zhang, W. Liang, and W. Zeng. Native sparse attention: Hardware-aligned and natively trainable sparse attention, 2025.

[^39]: R. Zellers, A. Holtzman, Y. Bisk, A. Farhadi, and Y. Choi. Hellaswag: Can a machine really finish your sentence? arXiv preprint arXiv:1905.07830, 2019.

[^40]: M. Zhang, S. Arora, R. Chalamala, A. Wu, B. Spector, A. Singhal, K. Ramesh, and C. Ré. Lolcats: On low-rank linearizing of large language models, 2025.