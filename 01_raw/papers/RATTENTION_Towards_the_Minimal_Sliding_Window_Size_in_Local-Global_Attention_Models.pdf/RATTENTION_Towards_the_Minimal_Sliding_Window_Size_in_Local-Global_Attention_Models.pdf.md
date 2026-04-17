# RATTENTION: Towards the Minimal Sliding Window Size in Local-Global Attention Models

Bailin Wang Chang Lan Chong Wang Ruoming Pang Apple {bwang47,c\_lan,mr.chongwang,rpang}@apple.com

# Abstract

Local-global attention models [\[28,](#page-10-0) [30\]](#page-10-1) have recently emerged as compelling alternatives to standard Transformers, promising improvements in both training and inference efficiency. However, the crucial choice of window size presents a Pareto tradeoff: larger windows maintain performance akin to full attention but offer minimal efficiency gains in short-context scenarios, while smaller windows can lead to performance degradation. Current models, such as Gemma2 and Mistral, adopt conservative window sizes (e.g., 4096 out of an 8192 pretraining length) to preserve performance. This work investigates strategies to shift this Pareto frontier, enabling local-global models to achieve efficiency gains even in short-context regimes. Our core motivation is to address the intrinsic limitation of local attention—its complete disregard for tokens outside the defined window. We explore RATTENTION, a variant of local attention integrated with a specialized linear attention mechanism designed to capture information from these out-of-window tokens. Pretraining experiments at the 3B and 12B scales demonstrate that RATTENTION achieves a superior Pareto tradeoff between performance and efficiency. As a sweetspot, RATTENTION with a window size of just 512 consistently matches the performance of full-attention models across diverse settings. Furthermore, the recurrent nature inherent in the linear attention component of RATTENTION contributes to enhanced long-context performance, as validated on the RULER benchmark [\[14\]](#page-9-0). Crucially, these improvements do not compromise training efficiency; thanks to a specialized kernel implementation and the reduced window size, RATTENTION maintains training speeds comparable to existing state-of-the-art approaches. [1](#page-0-0)

# 1 Introduction

Improving the efficiency of standard Transformers has been a central focus of architectural design. Sliding Window Attention (SWA) [\[7\]](#page-9-1), a natural variant of standard attention, has been widely adopted to reduce both memory transfer and computational costs, as demonstrated in prominent models like Mistral [\[15\]](#page-9-2) and Gemma [\[30\]](#page-10-1). Its constant memory consumption during decoding is particularly appealing for decoding time which is bounded by memory transfers rather than computation. However, SWA presents an inherent Pareto tradeoff between model capacity and efficiency: increasing the window size improves performance but diminishes efficiency gains. Recent models have often adopted conservative window sizes; for instance, both Mistral and Gemma utilize a 4096-token window out of a 8192-token pretraining context. Consequently, the efficiency benefits of SWA only become substantial for relatively long sequences. To illustrate, a 12B parameter local-global attention model we tested, using a 4K window size, apparently offers no KV cache savings for ≤ 4K context tasks. However, reducing the window size to 1k could yield approximately 56% KV cache savings at 4K length. In this work, we investigate whether this Pareto frontier can be shifted – achieving comparable performance with a significantly smaller window size.

<span id="page-0-0"></span><sup>1</sup>Our JAX implementation of RATTENTION using Pallas kernels are open-sourced at [https://github.](https://github.com/apple/axlearn/tree/main/axlearn/common/rattention) [com/apple/axlearn/tree/main/axlearn/common/rattention](https://github.com/apple/axlearn/tree/main/axlearn/common/rattention).

![](_page_1_Figure_0.jpeg)

<span id="page-1-0"></span>Figure 1: RATTENTION combines Sliding Window Attention (SWA) for local context with Residual Linear Attention (RLA) to gather information from out-of-window tokens. Apart from 4 in-window tokens, RLA compresses the information of token 1 for query token 5; token 1,2 for query token 6.

Our primary hypothesis is that the performance degradation observed when reducing the SWA window size stems from an intrinsic limitation of local attention: simply ensuring that *number\_of\_layers* × *window\_size* ≥ *context\_length* is often an inadequate heuristic for determining the optimal window size. To address this, we explore the integration of a Residual Linear Attention (RLA) module. RLA, in our design, is a specialized linear attention model [\[17\]](#page-10-2) aimed at capturing information from "residual tokens" – those lying beyond the immediate SWA window. This RLA module inherently enhances the capabilities of SWA. The recurrent nature of RLA offers two significant advantages: first, it preserves the constant-memory property crucial for efficient decoding with SWA; second, it lessens the over-reliance on positional embeddings for length extrapolation, leading to markedly better zero-shot length generalization compared to SWA alone. As depicted in Figure [1,](#page-1-0) the resulting hybrid architecture, which we term RATTENTION (Sliding Window *Attention* with *R*esidual Linear Attention), effectively captures information from all tokens within the context.

The key contribution of this work is demonstrating that RATTENTION can serve as a drop-in replacement for standard local attention mechanisms, positioning local-global models with RATTENTION as practical alternatives to full-attention Transformers. First, we show that RATTENTION models [2](#page-1-1) can match, and in some cases outperform, SWA models that use much larger sliding window sizes, while offering superior inference efficiency. Second, we demonstrate that this improved efficiency does not compromise training speed; with our dedicated kernel implementations and the reduced window size, RATTENTION models maintain training efficiency comparable to SWA models with larger windows. While the concept of combining SWA with linear attention has been explored (e.g., in [\[2,](#page-9-3) [40\]](#page-11-0), where a tiny SWA window complements linear attention), those approaches did not match the performance of full-attention models. We show that local-global models incorporating RATTENTION can be reliable alternatives to standard full-attention Transformers.

We conduct extensive experiments comparing RATTENTION with SWA within a local-global hybrid attention framework. Our results demonstrate that RATTENTION consistently matches or surpasses the performance of SWA-based models while requiring significantly less memory due to smaller window sizes. Furthermore, RATTENTION yields substantial performance improvements on longcontext reasoning tasks, as measured by the RULER benchmark [\[14\]](#page-9-0). On the training efficiency side, we develop dedicated kernels for RATTENTION so that the training efficiency is not compromised compared with full-attention models.

# 2 Background

We first briefly introduce standard attention and linear attention mechanisms. For notation, we use uppercase letters for matrices, lowercase letters for row vectors.

<span id="page-1-1"></span><sup>2</sup>Throughout this paper, "SWA models" and "RATTENTION models" refer to hybrid architectures that combine global attention with either SWA or RATTENTION, respectively, unless specified otherwise.

#### 2.1 Global Attention and Local Attention

In standard Transformers, an input sequence  $\mathbf{X} \in \mathbb{R}^{L \times d}$  (here L is the length and d is the hidden dimension) in the attention layer is processed through,

$$\begin{aligned} \boldsymbol{q}_t, \ \boldsymbol{k}_t, \ \boldsymbol{v}_t &= \boldsymbol{x}_t \boldsymbol{W}_Q, \ \boldsymbol{x}_t \boldsymbol{W}_K, \ \boldsymbol{x}_t \boldsymbol{W}_V, \ \boldsymbol{o}_t &= \frac{\sum_{i=1}^t \exp(\boldsymbol{q}_t \boldsymbol{k}_i^\intercal) \boldsymbol{v}_i}{\sum_{i=1}^t \exp(\boldsymbol{q}_t \boldsymbol{k}_i^\intercal)}, \end{aligned}$$

which computes the query  $(q_t)$ , key  $(k_t)$ , and value  $(v_t)$  vectors given the current token's representation  $x_t \in \mathbb{R}^{1 \times d}$ . Then attention is performed over the all the previous keys  $\{k_1, \ldots, k_t\}$  and values  $\{v_1, \ldots, v_t\}$ . During inference, as the t grow, the set of keys and values (i.e., KV cache) also grows linearly, leading to heavy memory consumption if t is large. Local attention (i.e., sliding window attention) [7] is then introduced to achieve constant-memory consumption during inference.

The basic idea of SWA is to limit the attention to only recent w tokens through,

$$\boldsymbol{o}_t^{\text{swa}} = \frac{\sum_{i=\max(1,t-w)}^{t} \exp(\boldsymbol{q}_t \boldsymbol{k}_i^\intercal) \boldsymbol{v}_i}{\sum_{i=\max(1,t-w)}^{t} \exp(\boldsymbol{q}_t \boldsymbol{k}_i^\intercal)}.$$

As a result, at most w+1 tokens (including the current token) need to be considered as KV cache during inference. The choice of w is based on the Pareto tradeoff between downstream performance and efficiency, and we aim to shift the Pareto frontier in this work.

#### 2.2 Linear Attention

Linear attention mechanisms [17] replace  $\exp(q_t k_i^{\mathsf{T}})$  with a kernel  $k(\boldsymbol{x}, \boldsymbol{y})$  with an associated feature map  $\phi$  (i.e.,  $k(\boldsymbol{x}, \boldsymbol{y}) = \langle \phi(\boldsymbol{x}), \phi(\boldsymbol{y}) \rangle$ ). This simplifies the calculation of  $\boldsymbol{o}_t$  since we have

$$o_t^{\text{la}} = \frac{\sum_{i=1}^t \phi(\boldsymbol{q}_t) \phi(\boldsymbol{k}_i)^\intercal \boldsymbol{v}_i}{\sum_{i=1}^t \phi(\boldsymbol{q}_t) \phi(\boldsymbol{k}_i)^\intercal} = \frac{\phi(\boldsymbol{q}_t) \sum_{i=1}^t \phi(\boldsymbol{k}_i)^\intercal \boldsymbol{v}_i}{\phi(\boldsymbol{q}_t) \sum_{i=1}^t \phi(\boldsymbol{k}_i)^\intercal}.$$

Letting  $\mathbf{S}_t = \sum_{i=1}^t \phi(\mathbf{k}_i)^\intercal \mathbf{v}_i$  and  $\mathbf{z}_t = \sum_{i=1}^t \phi(\mathbf{k}_i)^\intercal$  where  $\mathbf{S}_t \in \mathbb{R}^{d' \times d}, \mathbf{z}_t \in \mathbb{R}^{d' \times 1}$ , we can rewrite the above as an RNN,

$$\mathbf{S}_t = \mathbf{S}_{t-1} + \phi(\mathbf{k}_t)^\intercal \mathbf{v}_t, \ \mathbf{z}_t = \mathbf{z}_{t-1} + \phi(\mathbf{k}_t)^\intercal, \ \mathbf{o}_t^{\mathrm{la}} = \frac{\phi(\mathbf{q}_t) \mathbf{S}_t}{\phi(\mathbf{q}_t) \mathbf{z}_t}.$$

where d' denotes the output size of feature function  $\phi$ , initial state  $S_0 = 0$ . Recent work has found that the normalization terms can be replaced with a simpler RMSNorm on the output.

<span id="page-2-0"></span>
$$\mathbf{S}_t = \mathbf{S}_{t-1} + \phi(\mathbf{k}_t)^{\mathsf{T}} \mathbf{v}_t, \quad \mathbf{o}_t^{\mathsf{Ia}} = \phi(\mathbf{q}_t) \mathbf{S}_t. \tag{1}$$

Eq. 1 makes it clear that a linear attention layer is essentially a linear recurrent layer with matrix-valued hidden states  $\mathbf{S}_t$  that is updated via the outer-product  $\phi(\mathbf{k}_t)^{\mathsf{T}}\mathbf{v}_t$ .

**Chunkwise Parallel Form** The recurrent form in Eq.1 highlights the inherently sequential nature of linear attention, which makes it advantageous during the decoding stage. During training or prefilling, we rely on an equivalent chunkwise parallel formulation to compute the output efficiently[37].

Let the tilded input  $\mathbf{Q}_{[i]} := \mathbf{Q}_{iC+1:(i+1)C+1} \in \mathbb{R}^{C \times d}$  represent the query vectors for the i-th chunk, and define  $\mathbf{K}_{[i]}$ ,  $\mathbf{V}_{[i]}$ , and  $\mathbf{O}_{[i]}$  similarly. The output for the chunk can be computed as:

$$\mathbf{S}_{[i+1]} = \mathbf{S}_{[i]} + \tilde{\mathbf{K}}_{[i]}^{\mathsf{T}} \mathbf{V}_{[i]} \qquad \in \mathbb{R}^{d' \times d}$$
 (2)

$$\mathbf{O}_{[i]} = \underbrace{\mathbf{Q}_{[i]}\mathbf{S}_{[i-1]}}_{\mathbf{O}_{[i]}^{\text{inter}}} + \underbrace{\left(\left(\left(\tilde{\mathbf{Q}}_{[i]}\right)\tilde{\mathbf{K}}_{[i]}^{\mathsf{T}}\right)\odot\mathbf{M}\right)\mathbf{V}_{[i]}}_{\mathbf{O}_{[i]}^{\text{intra}}},\tag{3}$$

Here, C denotes the chunk (tiling) size, and  $\tilde{\mathbf{Q}} = \phi(\mathbf{Q})$ ,  $\tilde{\mathbf{K}} = \phi(\mathbf{K})$ . The key insight is that, given the chunk-level state  $\mathbf{S}_{[i]}$ , the output within each chunk can be computed in parallel using matrix multiplications – an operation that modern accelerators are highly optimized for.

Compared to global attention, linear attention requires maintaining only a fixed-size state S during inference, similar to local attention. Specifically, its cache is equivalent to using a window size of d ′×d 2×d in SWA [3](#page-3-0) . While this leads to significant inference efficiency, pure linear attention models still fall short of Transformers, particularly in recall-intensive tasks [\[1\]](#page-9-4).

# 3 Method

# 3.1 Residual Linear Attention (RLA)

To address the limitation of SWA, we use a specialized linear attention (RLA) that processes out-ofwindow tokens, as illustrated in the middle figure in Figure [1.](#page-1-0) Concretely, the recurrence of RLA is as follwows,

$$\mathbf{S}_t = \mathbf{S}_{t-1} + \phi(\mathbf{k}_t)^{\mathsf{T}} \mathbf{v}_t, \quad \mathbf{o}_t^{\mathsf{rla}} = \phi(\mathbf{q}_t) \mathbf{S}_{t-w-1}. \tag{4}$$

where instead of reading out from S<sup>t</sup> as in Eq [1,](#page-2-0) o<sup>t</sup> is obtained via reading out from St−w−<sup>1</sup> – the hidden states that captures the contextual information ends at token (t − w − 1). To integrate with SWA, we combine he outputs from RLA and SWA, as demonstrated in Figure [1.](#page-1-0)

Parameterization Employing separate parameters (i.e., projections for query/key/value/output) for RLA and SWA would double the parameter size of token-mixing layers. In this work, we find that simply *shares all parameters is sufficient*, provided that an appropriate feature map is used. As we will show in the experiments later, the softmax-based feature map [\[40\]](#page-11-0) yields the best performance, whereas the identity feature map, commonly used in pure linear attention models, is less effective.

Group-Query Variant We extend RLA to support Group-Query Attention (GQA), a popular variant of Multi-Head Attention that reduces KV memory usage. Specifically, we share key and value vectors within each query group, maintaining efficiency while preserving model quality. When coupled with SWA, RLA will share the same grouping structures as SWA as well.

## 3.2 Hybrid RATTENTION Models

We combine the results from RLA and SWA as follows:

$$o_t = \text{RMS}(o_t^{\text{swa}}) + \text{RMS}(o_t^{\text{rla}})$$

where two separate RMS norms are used upon the output from SWA and RLA respectively. In the group-query head variant, different heads will use separate parameters for RMS, which is a common strategy used in linear attention models [\[27,](#page-10-3) [37,](#page-11-1) [10\]](#page-9-5). Collectively, we call the resulting hybrid attention module RATTENTION.

Parameter Efficiency Our design introduces no additional parameters over standard SWA modules, as RATTENTION fully reuses the existing query/key/value projections. In contrast, recent work [\[12\]](#page-9-6) that combines state-space models (SSM) with SWA requires separate parameter sets for SSM and SWA. This results in a significantly larger portion of the model's parameters being allocated to token-mixing layers, which potentially limit the capacity of the feedforward layers when scaling up.

Kernel Design Our linear attention kernels incorporate two key optimizations—fused operations and flexible state saving. To reduce memory I/O cost, we first fuse the computation of the feature map ϕ(kt, vt) directly within the kernel, eliminating the need to store intermediate values in HBM and transfer them to VMEM, thereby lowering memory overhead.

Beyond reducing memory I/O, another critical factor for efficiency is maximizing the overlap between memory access and computation. In chunk-wise training, there is an inherent tradeoff in how much intermediate state (S[i] ) to store: storing more leads to faster backward passes (which depend on these states) but comes at a higher memory cost. Typically, chunk size is tuned according to two strategies – either storing all intermediate states or recomputing them during the backward pass. In this work, we introduce a more flexible state saving scheme. As illustrated in Figure [2,](#page-4-0) we store states every

<span id="page-3-0"></span><sup>3</sup>The factor of 2 in the denominator accounts for both key and value storage. d = 128 in a multi-head setting.

<span id="page-4-0"></span>![](_page_4_Figure_0.jpeg)

Figure 2: Interleaved state-saving pattern used in our training kernels. For every m chunks, only the state of the last chunk  $(\mathbf{S}_{[m]})$  is stored in HBM. The intermediate chunk states are recomputed on-chip as needed, using the most recent stored state  $\mathbf{S}_{[i-1]}$  from the previous group of m chunks. This approach make it more flexible to balances memory I/O cost and matmul computation.

m chunks while recomputing the states for the intermediate chunks. This hybrid approach allows compilers (e.g., using Triton or Pallas) to more effectively schedule operations and overlap memory I/O with computation. By enumerative search over m and chunk size, the best configuration can achieve around 15% speedup compared with m=1 and a typical chunk size 256.

#### 3.3 Local-Global Attention Models

We evaluate RATTENTION within a local-global attention framework that stacks multiple blocks in a repeating pattern of [local, local, global]. <sup>4</sup> The computation in each block follows:

$$\mathbf{Y}^{(l)} = \operatorname{Attention}^{(l)}(\operatorname{RMS}(\mathbf{X}^{(l)})) + \mathbf{X}^{(l)} \qquad \mathbf{X}^{(l+1)} = \operatorname{SwiGLU}(\operatorname{RMS}(\mathbf{Y}^{(l)})) + \mathbf{X}^{(l)},$$

where l indexes the layer starting from 1. The attention module  $Attention^{(l)}$  is instantiated as RATTENTION when  $l \mod 4 \neq 0$ , and as standard global attention otherwise. Following the LLaMA architecture [32], we adopt pre-norm and SwiGLU feed-forward layers.

While the combination of linear attention and sliding window attention was previously investigated by [2], their models, despite outperforming other purely linear architectures, did not achieve performance parity with full-attention Transformers, even at the 1B scale. RATTENTION local-global models, in contrast, successfully bridges this performance gap.

## 4 Experiment

#### 4.1 Main Results

The central question we address in our experiments is: what is the Pareto trade-off between downstream performance and efficiency (i.e., sliding window size) when replacing SWA with RATTENTION? Our evaluation consists of two stages:

- 1. we establish the trade-off curve between performance and window size through extensive experiments at the 3B scale. From this curve, we identify the minimal window size range for RATTENTION that achieves performance comparable to full attention;
- 2. we further validate these window sizes in settings where models are scaled by in three axes: the number of tokens, the model size or the pretraining context length.

The short answer to the question is that RATTENTION with a window size of  $\geq 512$  can reliably match (or exceed) the performance of full attention across different settings.

<span id="page-4-1"></span><sup>&</sup>lt;sup>4</sup>The ration between local and global attention is another axis to consider when exploring the tradeoff between downstream performance and efficiency of local-global models. We leave this direction for future work.

![](_page_5_Figure_0.jpeg)

Figure 4: Comparison of MMLU 5-shot performance scores across different window sizes at 3B scale with pretraining context length 4096. The horizontal purple dashed line represents the baseline using only global attention. The blue line shows Local-Global with sliding window attention (SWA), while the red line demonstrates the performance of Local-Global with RATTENTION. When window size = 0, Local-Global with RATTENTION reduces to Local-Global with only linear attention.

Pretraining Setup All the models are implemented in Jax [\[6\]](#page-9-7) and trained on v6e Cloud TPU clusters. We use 512/1024 chips provided as 2/4 × 256 chip slices to train 3B/12B models, respectively. Data parallelism along with activation recomputation is used for distributed training. We use a variant of RMSProp [\[31\]](#page-11-3) with momentum as the optimizer. We use our internal web-crawled data with a mixture similar to Llama models [\[32\]](#page-11-2). For evaluation, we consider a set of standard tasks: SciQ [\[34\]](#page-11-4), TriviaQA [\[16\]](#page-9-8), WebQ [\[3\]](#page-9-9), MMLU [\[13\]](#page-9-10), GSM8k [\[9\]](#page-9-11) LAMBADA [\[22\]](#page-10-4), PiQA [\[5\]](#page-9-12), HellaSwag [\[39\]](#page-11-5), WinoGrande [\[25\]](#page-10-5), ARC-easy (ARC-E) and ARC-challenge (ARC-C) [\[8\]](#page-9-13).

| Parameters         | 3B    | 12B   |
|--------------------|-------|-------|
| d_model            | 2048  | 5120  |
| layers             | 56    | 40    |
| num heads          | 16    | 40    |
| num kv heads       | 4     | 8     |
| qk-norm            | yes   | yes   |
| head type          | GQA   | GQA   |
| head size          | 128   | 128   |
| non-linearity      | GeGLU | GeGLU |
| feedforward dim    | 6656  | 16384 |
| pre-norm           | yes   | yes   |
| global-local ratio | 1:3   | 1:3   |

Figure 3: Model specifications of our 3B and 12B models.

<span id="page-5-0"></span>Model Setup Our base model is a Transformer model with interleaved global-local attention layers. The mixing ratio of global and local attention is always fixed at 1:3 with the first three layer being local and the final layer being global layer. The global layers do not use rotary positional embedding [\[26\]](#page-10-6) whereas the local layers still use it with θ = 5e5. Such interleaved patterns are better in long context generalization based on our experience and literature [\[35\]](#page-11-6). Apart from the positional embeddings, all the attention hyperparameters are shared between local and global attention.

In RATTENTION, all the attention hyperparameters for residual linear attention are shared from SWA

(i.e., number of heads, number of kv heads, head\_dim) since query/key/value projections are directly inherited from SWA for parameter efficiency. We emphasize again that RATTENTION does not require extra parameters compared with SWA or full attention.

Pareto Curve at 3B We train 3B-parameter SWA and RATTENTION models using various sliding window sizes on 400B tokens with a batch size of 1024. A full attention model is also trained as a baseline. As shown in Figure [4,](#page-5-0) there is a clear tradeoff between window size and MMLU performance—smaller window sizes lead to reduced performance.

Importantly, RATTENTION models achieve a better tradeoff curve compared to SWA models. With a window size of ≥ 512, RATTENTION already matches or even surpasses the performance of the full attention baseline. As we will show later, these gains persist across certain benchmarks when scaling up RATTENTION models. This observation is consistent with prior work suggesting that hybrid attention models can, in some cases, outperform standard Transformers [\[23\]](#page-10-7).

Selective Pretraining at 3B and 12B We further verify the effectiveness of RATTENTION by scaling both the number of tokens, model parameters and context length.

| Metric             | Full 4k | SWA 2k | SWA 1k | SWA 512 | RAtt 1k | <b>RAtt 512</b> |
|--------------------|---------|--------|--------|---------|---------|-----------------|
| ARC-C              | 46.25   | 47.01  | 45.39  | 44.97   | 44.28   | 45.39           |
| ARC-E              | 78.32   | 78.83  | 78.45  | 77.44   | 77.82   | 77.31           |
| HellaSwag          | 56.61   | 56.02  | 56.10  | 56.21   | 56.43   | 56.43           |
| LAMBADA            | 72.27   | 72.21  | 72.77  | 71.78   | 72.54   | 72.04           |
| PIQA               | 78.56   | 78.24  | 78.56  | 78.89   | 78.45   | 78.24           |
| SciQ               | 94.80   | 95.70  | 95.40  | 95.80   | 94.60   | 95.20           |
| WinoGrande         | 68.98   | 68.82  | 68.67  | 71.03   | 71.27   | 70.80           |
| TriviaQA (1-shot)  | 41.49   | 42.21  | 41.98  | 40.61   | 41.57   | 42.23           |
| WebQS (1-shot)     | 20.37   | 19.34  | 20.37  | 21.70   | 18.31   | 18.55           |
| Average (0/1-shot) | 62.00   | 62.00  | 62.00  | 62.00   | 61.70   | 61.80           |
| MMLU (5-shot)      | 56.70   | 55.70  | 55.70  | 55.50   | 56.22   | 55.62           |
| GSM8K (8-shot)     | 34.19   | 29.49  | 32.90  | 27.98   | 33.13   | 33.74           |

Table 1: Main results at 3B scale with pretraining context length 4096.

| Metric             | Full 4k | SWA 2K | RAttn-512 | RAttn-256 | RAttn-128 | RAttn-64 | RAttn-32 |
|--------------------|---------|--------|-----------|-----------|-----------|----------|----------|
| ARC-C              | 47.61   | 47.61  | 48.81     | 48.29     | 50.17     | 49.57    | 47.44    |
| ARC-E              | 79.21   | 80.01  | 79.12     | 79.25     | 79.97     | 79.88    | 79.29    |
| HellaSwag          | 57.76   | 57.99  | 57.72     | 57.79     | 58.11     | 58.25    | 58.06    |
| LAMBADA            | 73.55   | 73.06  | 73.14     | 73.88     | 73.65     | 72.99    | 73.51    |
| PIQA               | 79.27   | 80.03  | 78.51     | 79.54     | 79.00     | 79.38    | 79.43    |
| SciQ               | 95.60   | 96.40  | 95.40     | 95.90     | 95.70     | 95.40    | 95.30    |
| WinoGrande         | 70.17   | 72.69  | 71.03     | 71.74     | 72.38     | 71.90    | 70.09    |
| TriviaQA (1-shot)  | 41.51   | 41.22  | 41.85     | 41.34     | 42.56     | 40.97    | 42.05    |
| WebQS (1-shot)     | 21.21   | 21.65  | 24.66     | 22.64     | 22.00     | 20.42    | 20.23    |
| Average (0/1-shot) | 62.90   | 63.40  | 63.40     | 63.40     | 63.70     | 63.20    | 62.80    |
| MMLU (5-shot)      | 52.96   | 49.52  | 51.50     | 53.77     | 51.74     | 49.17    | 49.66    |
| GSM8K (8-shot)     | 30.33   | 24.26  | 29.57     | 29.34     | 26.61     | 30.93    | 26.61    |

<span id="page-6-3"></span>Table 2: Main results at 12B scale with pretraining context length 4096.

| Metric             | Full 8k | SWA 4K | SWA 2K | RAttn-512 |
|--------------------|---------|--------|--------|-----------|
| Average (0/1-shot) | 62.67   | 62.64  | 62.18  | 62.72     |
| MMLU (5-shot)      | 52.40   | 50.80  | 48.60  | 52.94     |
| GSM8K (8-shot)     | 36.69   | 35.71  | 33.28  | 37.39     |

Figure 5: Main results at 12B scale with pretraining context length 8192. Performance of zero- and one-shot tasks are summarized in **Average (0/1-shot)**.

<span id="page-6-2"></span><span id="page-6-1"></span>First, we selectively pretrain 3B-parameter SWA and RATTENTION models on 2T tokens. <sup>5</sup> As shown in Table 1, RATTENTION with a window size of 512 outperforms SWA models using window sizes up to 2048. Second, we pretrain 12B-parameter SWA

and RATTENTION models on 600B tokens and again evaluate performance across a range of window sizes. As shown in Table 2, RATTENTION with a window size of 512 continues to outperform SWA models, further validating its scalability. Finally, we assess RATTENTION with window size 512 in the setting of pretraining context length 8k and 600B tokens. The summary results shown in Table 5 indicates that RATTENTION remains strong compared with full attention models.

### <span id="page-6-4"></span>4.2 Long-Context Results

We then evaluate zero-shot generalization capability on the RULER [14] benchmark, testing them directly after pretraining at context length 4k. The average results are shown in Table 4.2.

| Model     | 4k    | 8k    | 16k   | 32k   |
|-----------|-------|-------|-------|-------|
| Full-4k   | 80.38 | 2.89  | 0.00  | 0.08  |
| SWA-2k    | 73.49 | 6.85  | 0.79  | 0.41  |
| RAttn-1k  | 73.87 | 53.90 | 40.00 | 20.84 |
| RAttn-512 | 80.79 | 66.26 | 50.80 | 29.59 |

Figure 6: Average zero-shot RULER performance at 3B scale with pretraining context length 4K.

RATTENTION models generalize reasonably well beyond the 4k training context, whereas other models fail to do so. Interestingly, RATTENTION models with smaller window sizes exhibit better generalization. We believe that smaller window sizes place greater pressure on the local attention module to generalize beyond the local window during pretraining, resulting in improved length generalization.

<span id="page-6-0"></span><sup>&</sup>lt;sup>5</sup>Our 400B/600B-token experiments takes 2-3 days to finish, and 2T-token experiments take 9 days.

![](_page_7_Figure_0.jpeg)

<span id="page-7-1"></span>Figure 7: Step time speedup (%) of local-global models using RATTENTION (window size 512) compared to SWA (window size 4k). As batch size increases, the theoretical speedup of RATTENTION increases and converges, since the KV cache size increasingly dominates the memory cost relative to the model parameter size.

## 4.3 Training Efficiency

We next demonstrate that the training efficiency of RATTENTION is not compromised. Specifically, we benchmark training efficiency in terms of step time using a batch size of 1024 and context lengths of 4k and 8k on TPU v5p-1024 (which is more suitable for larger-scale pretraining than v6e). As shown in Table [3,](#page-7-0) RATTENTION matches the training speed of both full attention and SWA models. Although RATTENTION introduces an additional RLA kernel to dispatch in the local attention layers compared to SWA, its small window size and the use of highly optimized RLA kernels collectively allow it to achieve comparable training speeds.

| Model Size | Pretraining Length | Full Attention | SWA       | RATTENTION |
|------------|--------------------|----------------|-----------|------------|
|            | 4096               | 0.84 (4k)      | 0.80 (2k) | 0.87 (512) |
| 3B         | 8192               | 1.20 (8k)      | 1.05 (4k) | 1.08 (1k)  |
|            | 4096               | 2.21 (4k)      | 2.10 (2k) | 2.26 (512) |
| 12B        | 8192               | 3.99 (8k)      | 3.89 (4k) | 3.97 (1k)  |

Table 3: Training speed comparison in terms of step time (seconds) at both 3B and 12B scale. Numbers in parentheses indicates window size of models.

#### 4.4 Inference Efficiency

Next, we analyze the inference gains achievable by RATTENTION when using a smaller sliding window size. Since the prefilling stage has a similar efficiency profile to the training stage, we focus our analysis on the step time during the generation stage. In this phase, the attention modules are typically memory-bound, while the feedforward modules can be either compute-bound or memorybound depending on the batch size. In general, the theoretical step time can be approximated by:

<span id="page-7-0"></span>
$$T_{\text{step}} = \frac{B \times S_{\text{KV}}}{BW} + \max\left(\frac{2 \times B \times P_{\text{count}}}{F}, \frac{P_{\text{size}}}{BW}\right)$$

where Tstep is the theoretical step time, B is the batch size, SKV is the KV cache size, BW is the total memory bandwidth, Pcount is the parameter count, Psize is the parameter size (in bytes), and F is the total FLOPs per second. As a case study, we apply this analysis to our 3B and 12B models using H100 hardware specifications and bfloat16 precision. Figure [7](#page-7-1) shows the step time speedup as a function of context length across different batch sizes. As the batch size increases, the theoretical speedup of RATTENTION grows, reaching up to approximately 60%. Moreover, the speedup ultimately converges to the same point regardless of model size, since the KV cache size increasingly dominates the memory cost relative to the model parameter size.

## 4.5 Ablation Study

We conducted ablation studies during 3B model training on the 400B token setting to identify the best configuration for RATTENTION. Results are shown in Table [4.](#page-8-0) For the feature map choice, we adopt softmax following [\[40\]](#page-11-0), which we found to outperform other alternatives such as ReLU and Identity.

| Metric             | Full-4k | RAttn-512 | w. ReLU | w. Identity | w. Mamba2 | w. Hymba | -SWA  | -GroupNorm |
|--------------------|---------|-----------|---------|-------------|-----------|----------|-------|------------|
| Average (0/1-shot) | 68.40   | 68.20     | 68.30   | 67.30       | 68.00     | 68.10    | 68.20 | 68.10      |
| MMLU (5-shot)      | 36.80   | 42.22     | 37.90   | 35.70       | 39.21     | 36.53    | 32.60 | 39.81      |

<span id="page-8-0"></span>Table 4: Ablation study on 3B models with 400B tokens and context length 4096.

We also experimented with adding more complex gating mechanism to linear attention, specifically using Mamba2 [10] and Gated DeltaNet [36]. However, no gain (or slight performance drop) is observed. We suspect that the introduction of more advanced linear models brings up optimization challenges: in our setup, the hybrid model already incorporates three token-mixing modules – full attention, sliding window attention, and residual linear attention – with the latter two sharing parameters. Introducing more complex forms of linear attention appears harder to optimize in this hybrid framework. On the other hand, we believe that it opens up a practical direction to explore for further research: designing better parameter-efficient linear models in the hybrid frameworks.

Additionally, we attempted to stack linear attention and sliding window attention across different heads, following the approach of Hymba [12]. However, this configuration proved suboptimal, likely because both mechanisms exhibit strong recency bias toward tokens within the sliding window. We also verified that linear attention alone cannot retrain the performance. Finally, we found that applying group normalization improves the overall performance.

#### 4.6 Related Work

Overall, there are two main approaches in designing efficient language models. The first relies on constant-memory modules, while the second focuses on leveraging sparsity in attention computation.

Constant-Memory Models and Their Hybrids Recurrent models and SWA models serve as the primary building blocks for hybrid architectures due to their constant-memory properties. However, pure constant-memory models often underperform standard Transformers [33]. Early hybrid models integrated these modules by interleaving them with standard attention. For example, Gemma2/Gemma3 [30, 29] alternate SWA with global attention, while Jamba [19] and Samba [23] combine Mamba with either SWA or global attention. Similarly, Griffin [11] integrates gated linear recurrences with SWA. Another line of research seeks to fuse attention and recurrent mechanisms within the same layer. Megalodon [21] uses a recurrent model to refine query/key representations within attention, while Hymba [12] runs both Mamba and attention in parallel within each layer. Our approach, RATTENTION, advances this direction by achieving better parameter efficiency than Hymba—completely sharing parameters between linear attention and SWA.

**Sparse Attention Models** To improve efficiency in long-context settings, another approach focuses on sparse attention, where the core challenge is designing effective sparsity patterns for KV-cache access. Early methods [24, 18, 4] use non-parametric techniques (e.g., k-nearest neighbors, locality-sensitive hashing) to select relevant query-key pairs. More recently, parametric methods that learn sparsity patterns have proven effective, such as Native Sparse Attention [38] and Mixture of Block Attention (MoBA) [20]. With hardware-aligned implementations, these modules can be trained more efficiently than global attention. However, unlike linear attention and SWA, sparse attention still requires storing the KV cache for all context tokens, same as global attention. This raises an ongoing research question: Is it more effective to sparsely access context within attention (as in sparse attention) or to rely on recurrent modules for context compression? We leave the investigation of this question for future work.

## 5 Conclusion and Future Work

In this work, we explore using RATTENTION to replace sliding window attention in local-global models. Our results show that residual linear attention enables a substantial reduction in sliding window size—from 4K/8K to 512—without loss in performance. Through both analytical and empirical studies on training and inference efficiency, we demonstrate that RATTENTION offers significant advantage over SWA: training efficiency is maintained, while inference efficiency is significantly improved. In future work, we plan to focus on engineering efforts to realize the theoretical efficiency gains within current inference frameworks. Fineuning existing pretrained full attention models into RATTENTION models is also promising direction to explore.

# Acknowledgment

We thank Sam Wiseman, Tao Lei, Aonan Zhang, Jianyu Wang, Karen Yang for their valuable feedback.

# References

- <span id="page-9-4"></span>[1] E. Akyürek, B. Wang, Y. Kim, and J. Andreas. In-context language learning: Architectures and algorithms, 2024.
- <span id="page-9-3"></span>[2] S. Arora, S. Eyuboglu, M. Zhang, A. Timalsina, S. Alberti, D. Zinsley, J. Zou, A. Rudra, and C. Ré. Simple linear attention language models balance the recall-throughput tradeoff, 2025.
- <span id="page-9-9"></span>[3] J. Berant, A. Chou, R. Frostig, and P. Liang. Semantic parsing on Freebase from questionanswer pairs. In *Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing*, pages 1533–1544, Seattle, Washington, USA, Oct. 2013. Association for Computational Linguistics.
- <span id="page-9-15"></span>[4] A. Bertsch, U. Alon, G. Neubig, and M. R. Gormley. Unlimiformer: Long-range transformers with unlimited length input, 2023.
- <span id="page-9-12"></span>[5] Y. Bisk, R. Zellers, J. Gao, Y. Choi, et al. Piqa: Reasoning about physical commonsense in natural language. In *Proceedings of the AAAI conference on artificial intelligence*, volume 34, pages 7432–7439, 2020.
- <span id="page-9-7"></span>[6] J. Bradbury, R. Frostig, P. Hawkins, M. J. Johnson, C. Leary, D. Maclaurin, G. Necula, A. Paszke, J. VanderPlas, S. Wanderman-Milne, and Q. Zhang. JAX: composable transformations of Python+NumPy programs, 2018.
- <span id="page-9-1"></span>[7] R. Child, S. Gray, A. Radford, and I. Sutskever. Generating long sequences with sparse transformers, 2019.
- <span id="page-9-13"></span>[8] P. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick, and O. Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. *arXiv preprint arXiv:1803.05457*, 2018.
- <span id="page-9-11"></span>[9] K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, et al. Training verifiers to solve math word problems. *arXiv preprint arXiv:2110.14168*, 2021.
- <span id="page-9-5"></span>[10] T. Dao and A. Gu. Transformers are ssms: Generalized models and efficient algorithms through structured state space duality, 2024.
- <span id="page-9-14"></span>[11] S. De, S. L. Smith, A. Fernando, A. Botev, G. Cristian-Muraru, A. Gu, R. Haroun, L. Berrada, Y. Chen, S. Srinivasan, G. Desjardins, A. Doucet, D. Budden, Y. W. Teh, R. Pascanu, N. D. Freitas, and C. Gulcehre. Griffin: Mixing gated linear recurrences with local attention for efficient language models, 2024.
- <span id="page-9-6"></span>[12] X. Dong, Y. Fu, S. Diao, W. Byeon, Z. Chen, A. S. Mahabaleshwarkar, S.-Y. Liu, M. V. Keirsbilck, M.-H. Chen, Y. Suhara, Y. Lin, J. Kautz, and P. Molchanov. Hymba: A hybrid-head architecture for small language models, 2024.
- <span id="page-9-10"></span>[13] D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt. Measuring massive multitask language understanding. *arXiv preprint arXiv:2009.03300*, 2020.
- <span id="page-9-0"></span>[14] C.-P. Hsieh, S. Sun, S. Kriman, S. Acharya, D. Rekesh, F. Jia, Y. Zhang, and B. Ginsburg. Ruler: What's the real context size of your long-context language models?, 2024.
- <span id="page-9-2"></span>[15] A. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford, D. S. Chaplot, D. de las Casas, F. Bressand, G. Lengyel, G. Lample, L. Saulnier, L. R. Lavaud, M.-A. Lachaux, P. Stock, T. L. Scao, T. Lavril, T. Wang, T. Lacroix, and W. E. Sayed. Mistral 7b, 2023.
- <span id="page-9-8"></span>[16] M. Joshi, E. Choi, D. S. Weld, and L. Zettlemoyer. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. *arXiv preprint arXiv:1705.03551*, 2017.

- <span id="page-10-2"></span>[17] A. Katharopoulos, A. Vyas, N. Pappas, and F. Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention, 2020.
- <span id="page-10-12"></span>[18] N. Kitaev, Łukasz Kaiser, and A. Levskaya. Reformer: The efficient transformer, 2020.
- <span id="page-10-9"></span>[19] O. Lieber, B. Lenz, H. Bata, G. Cohen, J. Osin, I. Dalmedigos, E. Safahi, S. Meirom, Y. Belinkov, S. Shalev-Shwartz, O. Abend, R. Alon, T. Asida, A. Bergman, R. Glozman, M. Gokhman, A. Manevich, N. Ratner, N. Rozen, E. Shwartz, M. Zusman, and Y. Shoham. Jamba: A hybrid transformer-mamba language model, 2024.
- <span id="page-10-13"></span>[20] E. Lu, Z. Jiang, J. Liu, Y. Du, T. Jiang, C. Hong, S. Liu, W. He, E. Yuan, Y. Wang, Z. Huang, H. Yuan, S. Xu, X. Xu, G. Lai, Y. Chen, H. Zheng, J. Yan, J. Su, Y. Wu, N. Y. Zhang, Z. Yang, X. Zhou, M. Zhang, and J. Qiu. Moba: Mixture of block attention for long-context llms, 2025.
- <span id="page-10-10"></span>[21] X. Ma, X. Yang, W. Xiong, B. Chen, L. Yu, H. Zhang, J. May, L. Zettlemoyer, O. Levy, and C. Zhou. Megalodon: Efficient llm pretraining and inference with unlimited context length, 2024.
- <span id="page-10-4"></span>[22] D. Paperno, G. Kruszewski, A. Lazaridou, Q. N. Pham, R. Bernardi, S. Pezzelle, M. Baroni, G. Boleda, and R. Fernández. The lambada dataset: Word prediction requiring a broad discourse context. *arXiv preprint arXiv:1606.06031*, 2016.
- <span id="page-10-7"></span>[23] L. Ren, Y. Liu, Y. Lu, Y. Shen, C. Liang, and W. Chen. Samba: Simple hybrid state space models for efficient unlimited context language modeling, 2025.
- <span id="page-10-11"></span>[24] A. Roy, M. Saffar, A. Vaswani, and D. Grangier. Efficient content-based sparse attention with routing transformers, 2020.
- <span id="page-10-5"></span>[25] K. Sakaguchi, R. L. Bras, C. Bhagavatula, and Y. Choi. Winogrande: An adversarial winograd schema challenge at scale. *Communications of the ACM*, 64(9):99–106, 2021.
- <span id="page-10-6"></span>[26] J. Su, M. Ahmed, Y. Lu, S. Pan, W. Bo, and Y. Liu. Roformer: Enhanced transformer with rotary position embedding. *Neurocomputing*, 568:127063, 2024.
- <span id="page-10-3"></span>[27] Y. Sun, L. Dong, S. Huang, S. Ma, Y. Xia, J. Xue, J. Wang, and F. Wei. Retentive network: A successor to transformer for large language models. *arXiv preprint arXiv:2307.08621*, 2023.
- <span id="page-10-0"></span>[28] C. A. R. Team. Optimizing inference, 2024. Accessed: March 19, 2025.
- <span id="page-10-8"></span>[29] G. Team. Gemma 3 technical report, 2025.
- <span id="page-10-1"></span>[30] G. Team, M. Riviere, S. Pathak, P. G. Sessa, C. Hardin, S. Bhupatiraju, L. Hussenot, T. Mesnard, B. Shahriari, A. Ramé, J. Ferret, P. Liu, P. Tafti, A. Friesen, M. Casbon, S. Ramos, R. Kumar, C. L. Lan, S. Jerome, A. Tsitsulin, N. Vieillard, P. Stanczyk, S. Girgin, N. Momchev, M. Hoffman, S. Thakoor, J.-B. Grill, B. Neyshabur, O. Bachem, A. Walton, A. Severyn, A. Parrish, A. Ahmad, A. Hutchison, A. Abdagic, A. Carl, A. Shen, A. Brock, A. Coenen, A. Laforge, A. Paterson, B. Bastian, B. Piot, B. Wu, B. Royal, C. Chen, C. Kumar, C. Perry, C. Welty, C. A. Choquette-Choo, D. Sinopalnikov, D. Weinberger, D. Vijaykumar, D. Rogozinska, D. Herbison, ´ E. Bandy, E. Wang, E. Noland, E. Moreira, E. Senter, E. Eltyshev, F. Visin, G. Rasskin, G. Wei, G. Cameron, G. Martins, H. Hashemi, H. Klimczak-Plucinska, H. Batra, H. Dhand, I. Nardini, ´ J. Mein, J. Zhou, J. Svensson, J. Stanway, J. Chan, J. P. Zhou, J. Carrasqueira, J. Iljazi, J. Becker, J. Fernandez, J. van Amersfoort, J. Gordon, J. Lipschultz, J. Newlan, J. yeong Ji, K. Mohamed, K. Badola, K. Black, K. Millican, K. McDonell, K. Nguyen, K. Sodhia, K. Greene, L. L. Sjoesund, L. Usui, L. Sifre, L. Heuermann, L. Lago, L. McNealus, L. B. Soares, L. Kilpatrick, L. Dixon, L. Martins, M. Reid, M. Singh, M. Iverson, M. Görner, M. Velloso, M. Wirth, M. Davidow, M. Miller, M. Rahtz, M. Watson, M. Risdal, M. Kazemi, M. Moynihan, M. Zhang, M. Kahng, M. Park, M. Rahman, M. Khatwani, N. Dao, N. Bardoliwalla, N. Devanathan, N. Dumai, N. Chauhan, O. Wahltinez, P. Botarda, P. Barnes, P. Barham, P. Michel, P. Jin, P. Georgiev, P. Culliton, P. Kuppala, R. Comanescu, R. Merhej, R. Jana, R. A. Rokni, R. Agarwal, R. Mullins, S. Saadat, S. M. Carthy, S. Cogan, S. Perrin, S. M. R. Arnold, S. Krause, S. Dai, S. Garg, S. Sheth, S. Ronstrom, S. Chan, T. Jordan, T. Yu, T. Eccles, T. Hennigan, T. Kocisky, T. Doshi, V. Jain, V. Yadav, V. Meshram, V. Dharmadhikari, W. Barkley, W. Wei, W. Ye, W. Han, W. Kwon, X. Xu, Z. Shen, Z. Gong, Z. Wei, V. Cotruta, P. Kirk, A. Rao, M. Giang, L. Peran, T. Warkentin,

- E. Collins, J. Barral, Z. Ghahramani, R. Hadsell, D. Sculley, J. Banks, A. Dragan, S. Petrov, O. Vinyals, J. Dean, D. Hassabis, K. Kavukcuoglu, C. Farabet, E. Buchatskaya, S. Borgeaud, N. Fiedel, A. Joulin, K. Kenealy, R. Dadashi, and A. Andreev. Gemma 2: Improving open language models at a practical size, 2024.
- <span id="page-11-3"></span>[31] T. Tieleman and G. Hinton. Divide the gradient by a running average of its recent magnitude. coursera: Neural networks for machine learning. *Technical report*, 2017.
- <span id="page-11-2"></span>[32] H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. *arXiv preprint arXiv:2307.09288*, 2023.
- <span id="page-11-8"></span>[33] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin. Attention is all you need, 2023.
- <span id="page-11-4"></span>[34] J. Welbl, N. F. Liu, and M. Gardner. Crowdsourcing multiple choice science questions. *arXiv preprint arXiv:1707.06209*, 2017.
- <span id="page-11-6"></span>[35] B. Yang, B. Venkitesh, D. Talupuru, H. Lin, D. Cairuz, P. Blunsom, and A. Locatelli. Rope to nope and back again: A new hybrid attention strategy. *arXiv preprint arXiv:2501.18795*, 2025.
- <span id="page-11-7"></span>[36] S. Yang, J. Kautz, and A. Hatamizadeh. Gated delta networks: Improving mamba2 with delta rule. *arXiv preprint arXiv:2412.06464*, 2024.
- <span id="page-11-1"></span>[37] S. Yang, B. Wang, Y. Shen, R. Panda, and Y. Kim. Gated linear attention transformers with hardware-efficient training, 2024.
- <span id="page-11-9"></span>[38] J. Yuan, H. Gao, D. Dai, J. Luo, L. Zhao, Z. Zhang, Z. Xie, Y. X. Wei, L. Wang, Z. Xiao, Y. Wang, C. Ruan, M. Zhang, W. Liang, and W. Zeng. Native sparse attention: Hardware-aligned and natively trainable sparse attention, 2025.
- <span id="page-11-5"></span>[39] R. Zellers, A. Holtzman, Y. Bisk, A. Farhadi, and Y. Choi. Hellaswag: Can a machine really finish your sentence? *arXiv preprint arXiv:1905.07830*, 2019.
- <span id="page-11-0"></span>[40] M. Zhang, S. Arora, R. Chalamala, A. Wu, B. Spector, A. Singhal, K. Ramesh, and C. Ré. Lolcats: On low-rank linearizing of large language models, 2025.