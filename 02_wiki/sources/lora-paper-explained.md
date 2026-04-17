---
title: "LoRA 논문 쉽게 설명하기"
created: 2026-04-16
updated: 2026-04-16
tags: [아티클, LoRA, Fine-Tuning, PEFT, LLM]
sources: [01_raw/transcripts/2026-04-16_LoRA 논문 쉽게 설명하기.md]
type: article
status: ingested
---

# LoRA 논문 쉽게 설명하기

- 저자: beeny-ds
- 출처: https://beeny-ds.tistory.com/entry/LORA-논문-쉽게-설명하기
- 원 논문: LoRA: Low-Rank Adaptation of Large Language Models (2021)

## Kernel

LLM의 Fully Fine-Tuning은 가중치의 2~3배 VRAM이 필요하여 비현실적. LoRA는 모델 가중치를 Freeze하고, 저랭크 행렬(LoRA_A, LoRA_B)만 학습하여 간접적으로 가중치를 업데이트. VRAM 대폭 절감 + 추론 속도 동일 + 원복 용이.

## 핵심 주장

1. Fully Fine-Tuning의 한계: Forward/Backward + gradient + optimizer 텐서 → 가중치 수 × 2~3배 VRAM 필요
2. LoRA 핵심 사상: 모델 가중치 Freeze + LoRA_A(d×r) · LoRA_B(r×k) 저랭크 행렬만 학습 (r << d)
3. 적용 위치: Transformer의 Q, K, V, O(self-attention) 레이어에 (LoRA_B × LoRA_A)를 단순 덧셈
4. 최적 적용: Query + Key 레이어에 적용 시 최고 성능
5. 장점 4가지:
   - Fully Fine-Tuning과 비슷하거나 더 좋은 성능
   - VRAM 대폭 절감 (학습 파라미터 = LoRA_A + LoRA_B만)
   - 추론 시 연산량 동일 (학습된 행렬을 원본에 합산)
   - 원복 용이 (더한 만큼 빼면 됨)

## 방법론 상세

### 저랭크 분해 (Low-Rank Decomposition)

```
원본 가중치: W (d × k)  ← Freeze
LoRA_A: (d × r)          ← 학습
LoRA_B: (r × k)          ← 학습
최종 가중치: W + (LoRA_B × LoRA_A)
```

- r (rank): 사용자 설정. r << d. 작을수록 파라미터 적음.
- 18M 파라미터만으로 Downstream task 학습 가능

> [!causal] 인과 관계
> [[lora|LoRA]]의 저랭크 분해 →(가능하게 함)→ LLM의 효율적 Fine-Tuning (VRAM 절감)
> 신뢰도: 높음 | 출처: [[lora-paper-explained]]

> [!causal] 인과 관계
> Fully Fine-Tuning의 VRAM 한계 →(기반이 됨)→ [[lora|LoRA]] 등 PEFT 기법의 등장
> 신뢰도: 높음 | 출처: [[lora-paper-explained]]

## 캡처 맥락

- 목적: LLM Fine-Tuning에 대한 흥미, 로컬 오픈소스 LLM 파인튜닝 사전학습
- 연결: EH R&C 정보에 특화된 파인튜닝 LLM 구축 가능성 탐색
- 다음 단계: LoRA 실습 코드 리뷰, 도메인 특화 파인튜닝 시나리오 설계

## 관련 개념

[[lora]], [[fine-tuning]], [[quantization]], [[transformer]], [[attention]]

## 관련 엔티티

[[gemma4]]
