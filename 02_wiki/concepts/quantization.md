---
title: 양자화 (Quantization)
created: 2026-04-15
updated: 2026-04-15
tags: [개념, LLM, 최적화]
sources: [01_raw/transcripts/2026-04-15_6.3.2 GPTQ, AWQ, GGUF 비교.md]
---

# 양자화 (Quantization)

모델 가중치의 숫자 정밀도를 낮춰(FP16→INT4) 메모리 사용량을 줄이는 기술. 로컬 AI 시대의 핵심 기술.

## Kernel

70B 모델은 FP16 기준 140GB VRAM 필요 → 로컬 24GB GPU에서 구동 불가. 양자화로 4비트 압축 시 ~35GB로 축소. 놀랍게도 원본 성능의 95% 이상 유지.

## 방법론: 선형 양자화 기본 수식

$$W_q = \text{round}\left(\frac{W}{S}\right) + Z$$

### 핵심 키워드

- **W (원본 가중치)**: FP16 등 고정밀도 실수값.
- **S (Scale Factor)**: 데이터 범위를 압축하는 스케일링 팩터.
- **Z (Zero-point)**: 비대칭 분포 보정을 위한 영점.
- **round()**: 반올림. 실수를 정수 버킷에 매핑.
- **비트 폭 (Bit Width)**: INT4(16가지), INT8(256가지). 낮을수록 압축률 높지만 정밀도 손실.

## 3대 기법 상세 비교

### GPTQ (GPU 전용)

- 원리: 가중치 행렬을 열(column) 단위로 순차 양자화. 앞선 열의 오차를 뒤 열에서 보상.
- 보정 데이터: ~128개 텍스트 샘플로 활성화 값 수집.
- 장점: 4비트에서도 원본과 유사한 품질. 7B 모델 수 분 내 양자화.
- 단점: GPU 전용. 추론 시 역양자화 필요.

### AWQ (GPU 전용)

- 원리: 활성화(Activation) 값 기반으로 중요 가중치 채널 식별. 중요 채널은 높은 정밀도 유지.
- 핵심 통찰: 모든 가중치가 동등하지 않다. 활성화 크기가 큰 채널이 출력에 더 큰 영향.
- 장점: GPTQ보다 빠른 양자화 속도. 동등한 품질.
- 단점: GPU 전용.

### GGUF (크로스플랫폼)

- 원리: llama.cpp 생태계의 표준 포맷. CPU+GPU 혼합 추론 지원.
- 장점: Mac, Windows, Linux 어디서든 구동. Ollama 등 로컬 런타임과 호환.
- 단점: GPU 전용 기법 대비 추론 속도 다소 느림.

### 비교 테이블

| 기법 | 타겟 | 보상 전략 | 속도 | 호환성 |
|------|------|----------|------|--------|
| GPTQ | GPU | 열 단위 순차 보상 | 빠름 | vLLM, TGI |
| AWQ | GPU | 활성화 기반 채널 보존 | 매우 빠름 | vLLM, TGI |
| GGUF | CPU+GPU | 혼합 정밀도 | 보통 | llama.cpp, Ollama |

## 왜 95% 성능이 유지되는가?

뉴런 간 연결 강도(가중치)의 미세한 소수점 값이 무뎌져도, 거대한 네트워크가 만들어내는 전체적인 문맥 파악의 논리 구조는 쉽게 붕괴되지 않기 때문. %%주석: MS의 BitNet과의 연결점 탐구 필요%%

> [!causal] 인과 관계
> [[quantization|양자화]] →(가능하게 함)→ SLM의 로컬 GPU 구동
> 신뢰도: 높음 | 출처: [[gptq-awq-gguf-comparison]]

## Gemma 4에서의 적용

Gemma 4는 양자화에 최적화된 아키텍처 설계. Shared KV Cache와 MoE(4B active)로 양자화 시 메모리 효율 극대화.

## 관련

[[transformer]], [[gemma4-huggingface]], [[gptq-awq-gguf-comparison]]
