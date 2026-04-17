---
title: 트랜스포머 (Transformer)
created: 2026-04-15
updated: 2026-04-15
tags: [개념, AI, 아키텍처]
sources: [01_raw/transcripts/2026-04-15_RAttention Towards the Minimal Sliding Window Size in Local-Global Attention Mo.md]
---

# 트랜스포머 (Transformer)

2017년 구글이 발표한 "Attention Is All You Need" 기반 아키텍처. 현대 LLM의 표준.

## Kernel

RNN/LSTM의 순차 처리 한계(긴 문장에서 앞부분 망각)를 셀프 어텐션의 병렬 행렬 연산으로 극복. 모든 단어가 모든 단어를 동시에 바라본다.

## 아키텍처 구성

### 핵심 블록 (Decoder-only, GPT 계열)

```
입력 토큰 → [토큰화] → [임베딩] → [위치 인코딩]
    ↓
┌─────────────────────────┐ × N 레이어
│  Self-Attention          │
│  ↓                       │
│  Feed-Forward Network    │
│  ↓                       │
│  Layer Normalization     │
│  + Residual Connection   │
└─────────────────────────┘
    ↓
출력 확률 분포 → [다음 토큰 선택]
```

### 핵심 키워드

- **셀프 어텐션 (Self-Attention)**: 문장 내 모든 토큰이 서로를 참조. [[attention]] 참조.
- **Feed-Forward Network (FFN)**: 어텐션 출력을 비선형 변환. 각 토큰 독립 처리.
- **Residual Connection**: 입력을 출력에 더함. 깊은 네트워크에서 그래디언트 소실 방지.
- **Layer Normalization**: 레이어 출력을 정규화. 학습 안정성.
- **위치 인코딩 (Positional Encoding)**: 토큰의 순서 정보 주입. 어텐션은 순서를 모르므로 필수.
  - RoPE (Rotary Position Embedding): 현대 LLM 표준. 상대적 위치를 회전 행렬로 인코딩.
  - Dual RoPE: Gemma 4가 사용. sliding 레이어와 global 레이어에 다른 RoPE 적용.
- **Context Window**: 한 번에 처리 가능한 최대 토큰 수. O(N²) 제약.

### 병렬 처리의 핵심

행렬 곱셈(QK^T, V 곱하기)은 GPU의 CUDA 코어에서 대규모 병렬 처리에 최적화. 이것이 수십억 파라미터 모델이 빠르게 추론할 수 있는 이유.

## 진화 방향

| 세대 | 특징 | 예시 |
|------|------|------|
| 원본 Transformer (2017) | Encoder-Decoder, Full Attention | BERT, T5 |
| GPT 계열 | Decoder-only, Autoregressive | GPT-4, Claude |
| 효율화 | SWA, MoE, 양자화 | Gemma 4, Mistral |
| 멀티모달 | Vision/Audio Encoder + Text Decoder | Gemma 4, GPT-4o |

> [!causal] 인과 관계
> [[transformer|트랜스포머]]의 병렬 행렬 연산 →(가능하게 함)→ 수십억 파라미터 모델의 실시간 추론
> 신뢰도: 높음 | 출처: [[gemma4-huggingface]]

## 관련

[[attention]], [[sliding-window-attention]], [[tokenization]], [[embedding]], [[quantization]]
- [[rattention-sliding-window]], [[gemma4-huggingface]]
