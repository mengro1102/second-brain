---
title: LoRA (Low-Rank Adaptation)
created: 2026-04-16
updated: 2026-04-16
tags: [개념, Fine-Tuning, PEFT, LLM]
sources: [01_raw/transcripts/2026-04-16_LoRA 논문 쉽게 설명하기.md]
---

# LoRA (Low-Rank Adaptation)

2021년 등장한 Parameter-Efficient Fine-Tuning(PEFT) 기법. 모델 가중치를 Freeze하고 저랭크 행렬만 학습.

## Kernel

LLM의 가중치 행렬 W(d×k)를 직접 업데이트하지 않고, 저랭크 행렬 LoRA_A(d×r) · LoRA_B(r×k)만 학습하여 W + (LoRA_B × LoRA_A)로 간접 업데이트. r << d이므로 학습 파라미터가 극소.

## 방법론

### 동작 원리

```
입력 x → [Frozen W] → h₁
       → [LoRA_A → LoRA_B] → h₂
       → h = h₁ + h₂  (단순 덧셈)
```

### 핵심 키워드

- **Rank (r)**: 저랭크 행렬의 차원. r=8, 16 등. 작을수록 파라미터 적지만 표현력 제한.
- **Freeze**: 원본 모델 가중치를 고정. gradient/optimizer 텐서가 GPU에 로드되지 않음 → VRAM 절감.
- **LoRA_A (d×r)**: 입력을 저차원으로 투영하는 다운프로젝션 행렬.
- **LoRA_B (r×k)**: 저차원에서 원래 차원으로 복원하는 업프로젝션 행렬.
- **Weight Type**: Q, K, V, O 중 어디에 적용할지. Q+K가 최적.
- **PEFT (Parameter-Efficient Fine-Tuning)**: LoRA를 포함한 효율적 파인튜닝 기법의 총칭.

### 왜 저랭크가 작동하는가?

대형 모델의 가중치 변화(ΔW)는 실제로 저랭크 구조를 가진다는 가설. 즉, Fine-Tuning 시 변하는 정보의 "본질적 차원"이 전체 파라미터 수보다 훨씬 작다.

### 장점

1. Fully Fine-Tuning과 동등하거나 우수한 성능
2. VRAM 대폭 절감 (학습 파라미터 = LoRA_A + LoRA_B만)
3. 추론 시 연산량 동일 (합산 후 단일 행렬로 동작)
4. 원복 용이 (더한 만큼 빼면 원본 복원)

> [!causal] 인과 관계
> [[lora|LoRA]]의 저랭크 분해 →(가능하게 함)→ LLM의 효율적 Fine-Tuning
> 신뢰도: 높음 | 출처: [[lora-paper-explained]]

> [!causal] 인과 관계
> [[quantization|양자화]] + [[lora|LoRA]] →(가능하게 함)→ 로컬 GPU에서의 도메인 특화 LLM 구축
> 신뢰도: 중간 | 출처: [[lora-paper-explained]]

## Gemma 4와의 연결

Gemma 4는 Apache 2 라이선스 오픈소스. LoRA로 도메인 특화 파인튜닝 가능. Hugging Face TRL, Unsloth 등으로 실습 가능.

## 관련

[[fine-tuning]], [[quantization]], [[transformer]], [[attention]]
- [[lora-paper-explained]], [[gemma4-huggingface]]
