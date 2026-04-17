---
title: Fine-Tuning
created: 2026-04-16
updated: 2026-04-16
tags: [개념, LLM, 학습]
sources: [01_raw/transcripts/2026-04-16_LoRA 논문 쉽게 설명하기.md]
---

# Fine-Tuning

사전 학습된 모델을 특정 도메인/태스크에 맞게 추가 학습하는 과정.

## Kernel

사전 학습(Pre-training)이 "세상의 일반 지식"을 배우는 것이라면, Fine-Tuning은 "특정 업무에 맞게 전문화"하는 것. 의사 면허를 딴 후 전문의 수련을 받는 것과 유사.

## 유형

| 유형 | 설명 | VRAM 요구 |
|------|------|----------|
| Fully Fine-Tuning | 모든 가중치 업데이트 | 가중치 × 2~3배 |
| [[lora\|LoRA]] | 저랭크 행렬만 학습 | 극소 |
| QLoRA | 양자화 + LoRA | 더 극소 |
| SFT | 지도 미세 조정 (질문-답변 쌍) | 방법에 따라 다름 |
| RLHF/DPO | 인간 선호도 기반 정렬 | 방법에 따라 다름 |

> [!causal] 인과 관계
> Fully Fine-Tuning의 VRAM 한계 →(기반이 됨)→ [[lora|LoRA]] 등 PEFT 기법의 등장
> 신뢰도: 높음 | 출처: [[lora-paper-explained]]

## 관련

[[lora]], [[quantization]], [[transformer]]
- [[lora-paper-explained]]
