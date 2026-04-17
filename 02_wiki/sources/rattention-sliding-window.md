---
title: "RAttention: Towards the Minimal Sliding Window Size in Local-Global Attention Models"
created: 2026-04-15
updated: 2026-04-15
tags: [논문, 어텐션, Transformer, 효율성]
sources: [01_raw/transcripts/2026-04-15_RAttention Towards the Minimal Sliding Window Size in Local-Global Attention Mo.md, 01_raw/papers/RATTENTION_Towards_the_Minimal_Sliding_Window_Size_in_Local-Global_Attention_Models.pdf.pdf]
type: paper
status: ingested
---

# RAttention: Towards the Minimal Sliding Window Size

- 저자: Bailin Wang, Chang Lan, Chong Wang, Ruoming Pang
- 소속: Apple
- 출처: https://ar5iv.labs.arxiv.org/html/2506.15545v2

## Kernel

Sliding Window Attention(SWA)의 윈도우 크기를 극단적으로 줄이면서도 성능을 유지하는 방법. Residual Linear Attention(RLA)을 SWA에 통합하여, 윈도우 밖 토큰 정보를 캡처. 윈도우 512로도 Full Attention 성능 매칭.

## 핵심 주장

1. SWA의 근본 한계: 윈도우 밖 토큰을 완전히 무시 → 윈도우 축소 시 성능 저하
2. RAttention = SWA + Residual Linear Attention(RLA): 윈도우 밖 토큰 정보를 선형 어텐션으로 캡처
3. 윈도우 512로 Full Attention 성능 매칭 (3B, 12B 스케일)
4. RLA의 recurrent 특성 → 장문맥 성능 향상 (RULER 벤치마크)
5. 전용 커널 구현으로 학습 효율 유지

> [!causal] 인과 관계
> [[sliding-window-attention|SWA]]의 윈도우 밖 정보 손실 →(성능 저하)→ 소형 윈도우 모델의 품질
> 신뢰도: 높음 | 출처: [[rattention-sliding-window]]

> [!causal] 인과 관계
> RAttention(SWA+RLA) →(성능 향상)→ 소형 윈도우에서의 Full Attention 수준 성능
> 신뢰도: 높음 | 출처: [[rattention-sliding-window]]

## Gemma 4와의 연결

Gemma 4는 Local-Global Attention(sliding window 512/1024 + full context)을 사용. RAttention 논문은 이 아키텍처의 이론적 배경과 윈도우 크기 선택의 근거를 제공.

## 캡처 맥락

- 목적: Gemma 4 블로그 챕터 2(어텐션 메커니즘) 참조 소스
- 연결: Gemma 4의 Local-Global Attention 설계 근거 이해

## 관련 개념

[[attention]], [[sliding-window-attention]], [[transformer]]

## 관련 엔티티

[[gemma4]]
