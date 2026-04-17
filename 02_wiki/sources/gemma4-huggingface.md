---
title: "Welcome Gemma 4: Frontier multimodal intelligence on device"
created: 2026-04-15
updated: 2026-04-15
tags: [아티클, Gemma, 멀티모달, SLM, DeepMind]
sources: [01_raw/transcripts/2026-04-15_Welcome Gemma 4 Frontier multimodal intelligence on device.md]
type: article
status: ingested
---

# Welcome Gemma 4: Frontier multimodal intelligence on device

- 출처: https://huggingface.co/blog/gemma4
- 소속: [[google-deepmind|Google DeepMind]], Hugging Face

## Kernel

Gemma 4는 Google DeepMind의 오픈소스 멀티모달 SLM 패밀리. Apache 2 라이선스. 이미지/오디오/비디오 입력 → 텍스트 출력. MoE(26B/4B active)로 로컬 GPU에서 구동 가능하면서 LMArena 1441 달성.

## 핵심 주장

1. 4개 사이즈: E2B(2.3B), E4B(4.5B), 31B(dense), 26B-A4B(MoE)
2. 네이티브 멀티모달: 이미지, 오디오, 비디오를 외부 변환기 없이 직접 처리
3. 256K 컨텍스트 윈도우 (31B, 26B-A4B)
4. 핵심 아키텍처: Local-Global Attention(sliding window + full context), Dual RoPE, PLE, Shared KV Cache
5. [[moe|MoE]]: 26B 파라미터 중 4B만 활성화 → 로컬 추론 효율 극대화
6. Function Calling, Thinking 모드 지원 → 에이전틱 워크플로우

> [!causal] 인과 관계
> [[knowledge-distillation|지식 증류]] →(가능하게 함)→ Gemma 4의 SLM 수준 지능
> 신뢰도: 높음 | 출처: [[gemma4-huggingface]]

> [!causal] 인과 관계
> [[moe|MoE]] →(성능 향상)→ Gemma 4의 로컬 추론 효율
> 신뢰도: 높음 | 출처: [[gemma4-huggingface]]

## 아키텍처 상세

- Sliding Window (512/1024 토큰) + Global Full-Context Attention 교대
- Per-Layer Embeddings (PLE): 레이어별 토큰 특화 신호 주입
- Shared KV Cache: 마지막 N개 레이어가 이전 레이어의 KV 재사용
- Vision Encoder: 가변 종횡비, 토큰 버짓 조절 (70~1120)
- Audio Encoder: USM-style conformer

## 캡처 맥락

- 목적: Gemma 4 블로그 포스트 작성을 위한 핵심 소스
- 연결: 블로그 챕터 5(멀티모달), 6(Gemma 4 해부)의 주요 참조
- 다음 단계: output/blog/에 Gemma 4 블로그 포스트 완성

## 관련 개념

[[transformer]], [[attention]], [[quantization]], [[knowledge-distillation]], [[moe]], [[sliding-window-attention]]

## 관련 엔티티

[[google-deepmind|Google DeepMind]], [[gemma4]]
