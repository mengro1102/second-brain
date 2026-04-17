---
title: 어텐션 (Attention)
created: 2026-04-15
updated: 2026-04-15
tags: [개념, AI, Transformer]
sources: [01_raw/transcripts/2026-04-15_RAttention Towards the Minimal Sliding Window Size in Local-Global Attention Mo.md]
---

# 어텐션 (Attention)

문장 내 특정 단어를 이해할 때 어떤 단어에 얼마나 집중할지를 수학적으로 계산하는 메커니즘. 트랜스포머의 심장.

## Kernel

"배가 부르다"와 "배를 타다"에서 '배'의 의미는 주변 단어에 의해 결정된다. 어텐션은 이 "주변 단어에 대한 집중도"를 수학적으로 계산하는 것.

## 방법론: Scaled Dot-Product Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### 핵심 키워드

- **Query (Q)**: 현재 이해하고자 하는 기준 단어의 벡터. "검색어" 역할.
- **Key (K)**: 문장 내 다른 단어들의 특징 벡터. "문서 제목/키워드" 역할.
- **Value (V)**: 문장 내 다른 단어들의 실제 의미 벡터. "문서 내용" 역할.
- **내적 (Dot Product, QK^T)**: Q와 K의 유사도 계산. 값이 클수록 관련성 높음.
- **스케일링 (√d_k)**: 내적 값이 차원 수에 비례해 커지는 것을 방지. 학습 안정성 확보.
- **Softmax**: 유사도 값을 0~1 확률로 변환. 전체 합 = 1. 가중치 분배.
- **KV Cache**: 추론 시 이전 토큰의 K, V를 메모리에 저장하여 재사용. 생성 속도 향상의 핵심.

### 연산 복잡도

- Full Attention: O(N²) — 모든 토큰이 서로를 참조. 토큰 2배 → 연산 4배.
- 이것이 Context Window 한계와 OOM(Out Of Memory)의 근본 원인.

## 변형

| 변형 | 특징 | 사용 모델 |
|------|------|----------|
| Self-Attention | 문장 내 모든 단어가 서로 참조 | 모든 Transformer |
| Multi-Head Attention | 여러 "관점"에서 동시에 어텐션 수행 | 모든 Transformer |
| [[sliding-window-attention\|SWA]] | 고정 윈도우 내에서만 어텐션 → O(N×W) | Gemma, Mistral |
| Cross-Attention | 서로 다른 모달리티(텍스트↔이미지) 간 어텐션 | 멀티모달 모델 |
| Linear Attention | 커널 트릭으로 O(N) 달성 | RAttention의 RLA |
| RAttention | SWA + Residual Linear Attention | Apple 연구 |

> [!causal] 인과 관계
> [[attention|어텐션]]의 O(N²) 복잡도 →(성능 저하)→ 긴 문맥에서의 메모리/속도 병목
> 신뢰도: 높음 | 출처: [[rattention-sliding-window]]

## Gemma 4에서의 적용

- Local-Global Attention: sliding window(512/1024) + full context 레이어 교대
- Dual RoPE: sliding 레이어는 표준 RoPE, global 레이어는 proportional RoPE → 장문맥 지원
- Shared KV Cache: 마지막 N개 레이어가 이전 레이어의 KV 재사용 → 메모리 절감

## 관련

[[transformer]], [[sliding-window-attention]], [[embedding]]
- [[rattention-sliding-window]], [[gemma4-huggingface]]
