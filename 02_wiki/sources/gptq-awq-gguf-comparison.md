---
title: "GPTQ, AWQ, GGUF 비교"
created: 2026-04-15
updated: 2026-04-15
tags: [아티클, 양자화, LLM, 최적화]
sources: [01_raw/transcripts/2026-04-15_6.3.2 GPTQ, AWQ, GGUF 비교.md]
type: article
status: ingested
---

# GPTQ, AWQ, GGUF 비교

- 출처: https://wikidocs.net/338712

## Kernel

LLM 양자화의 3대 기법 비교. GPTQ는 열 단위 순차 보상으로 GPU 최적화, AWQ는 활성화 기반 중요 가중치 보존, GGUF는 CPU/크로스플랫폼 호환 포맷. 4비트 양자화로 원본 성능 95% 이상 유지.

## 핵심 주장

1. GPTQ: 가중치 행렬을 열 단위로 순차 양자화, 앞선 열의 오차를 뒤 열에서 보상 → GPU 전용
2. AWQ: 활성화 값 기반으로 중요 가중치 식별, 중요 채널은 높은 정밀도 유지 → GPU 전용
3. GGUF: llama.cpp 생태계의 표준 포맷, CPU+GPU 혼합 추론 지원, 크로스플랫폼
4. 70B 모델도 4비트 양자화 시 24GB VRAM에서 구동 가능

> [!causal] 인과 관계
> [[quantization|양자화]] →(가능하게 함)→ SLM의 로컬 GPU 구동
> 신뢰도: 높음 | 출처: [[gptq-awq-gguf-comparison]]

## 캡처 맥락

- 목적: Gemma 4 블로그 챕터 4(양자화/SLM) 참조 소스
- 연결: GPTQ/AWQ/GGUF의 원리와 트레이드오프 설명에 활용

## 관련 개념

[[quantization]], [[transformer]]

## 관련 엔티티

[[gemma4]]
