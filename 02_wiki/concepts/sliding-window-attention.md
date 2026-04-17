---
title: Sliding Window Attention (SWA)
created: 2026-04-15
updated: 2026-04-15
tags: [개념, Transformer, 효율성]
sources: [01_raw/transcripts/2026-04-15_RAttention Towards the Minimal Sliding Window Size in Local-Global Attention Mo.md]
---

# Sliding Window Attention (SWA)

고정 크기 윈도우 내에서만 어텐션을 수행하여 메모리/연산 비용을 줄이는 기법.

## Kernel

Full Attention의 O(N²) → SWA의 O(N×W). 디코딩 시 일정한 메모리 소비(KV Cache 크기 = W). 단, 윈도우 밖 토큰 정보를 완전히 무시하는 것이 근본 한계.

## 방법론

### Local-Global Hybrid Attention

현대 모델(Gemma 4, Mistral)은 SWA만 쓰지 않고, SWA 레이어와 Full Attention 레이어를 교대 배치:

```
Layer 1: SWA (윈도우 512)    ← 로컬 문맥
Layer 2: Full Attention       ← 글로벌 문맥
Layer 3: SWA (윈도우 512)    ← 로컬 문맥
Layer 4: Full Attention       ← 글로벌 문맥
...
```

### 핵심 키워드

- **윈도우 크기 (Window Size, W)**: SWA가 참조하는 토큰 범위. Gemma 4는 512/1024.
- **KV Cache 절감**: Full Attention은 전체 시퀀스의 KV 저장 필요. SWA는 W개만 저장.
- **Pareto 트레이드오프**: 윈도우 크기 ↑ → 성능 ↑ but 효율 ↓. 최적점 찾기가 핵심.
- **layers × window_size ≥ context_length**: SWA의 경험적 규칙. 하지만 RAttention 논문이 이것이 불충분함을 증명.

### RAttention의 해결책

SWA의 근본 한계(윈도우 밖 정보 손실)를 Residual Linear Attention(RLA)으로 보완:

- RLA: 윈도우 밖 토큰 정보를 선형 어텐션의 recurrent 상태로 캡처
- 결과: 윈도우 512로도 Full Attention 성능 매칭 (3B, 12B 스케일)
- 추가 이점: recurrent 특성 → 장문맥 일반화 성능 향상

> [!causal] 인과 관계
> [[sliding-window-attention|SWA]]의 윈도우 밖 정보 손실 →(성능 저하)→ 소형 윈도우 모델의 품질
> 신뢰도: 높음 | 출처: [[rattention-sliding-window]]

> [!causal] 인과 관계
> RAttention(SWA+RLA) →(성능 향상)→ 소형 윈도우에서의 Full Attention 수준 성능
> 신뢰도: 높음 | 출처: [[rattention-sliding-window]]

## Gemma 4에서의 적용

- 소형 dense 모델: 윈도우 512
- 대형 모델: 윈도우 1024
- Dual RoPE: sliding 레이어는 표준 RoPE, global 레이어는 proportional RoPE
- Shared KV Cache: 마지막 N개 레이어가 이전 레이어의 KV 재사용

## 관련

[[attention]], [[transformer]]
- [[rattention-sliding-window]], [[gemma4-huggingface]]
