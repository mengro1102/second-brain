---
title: "NVIDIA Nemotron 3 Super"
created: 2026-04-30
updated: 2026-04-30
tags: [아티클, NVIDIA, LLM, SLM]
sources: [01_raw/articles/2026-04-30_엔비디아 네모트론 3 슈퍼(NVIDIA Nem..  네이버블로그.md, 01_raw/transcripts/2026-04-30_🤖🚀 Nvidia , Nemotron 3 Super, 무엇이 다른가 핵심만 빠르게 정리.md]
type: article
status: ingested
---

# NVIDIA Nemotron 3 Super

## Kernel

NVIDIA의 Nemotron 3 Super는 Llama 3.1 8B 기반으로 지식 증류(Knowledge Distillation) + NAS(Neural Architecture Search) + 정렬(Alignment)을 거쳐 만든 효율적 SLM. 동급 대비 높은 성능을 달성하면서도 경량화에 성공.

## 핵심 기법

1. **지식 증류**: 대형 모델의 지식을 소형 모델로 전이
2. **NAS (Neural Architecture Search)**: 최적 아키텍처 자동 탐색
3. **정렬 (Alignment)**: 인간 선호도에 맞춘 모델 조정

## 캡처 맥락

- 목적: SLM 효율화 기법 학습
- 연결: EH R&C 온프레미스 서버에서 구동 가능한 모델 탐색
- 다음 단계: 실제 온프레미스 환경에서의 벤치마크 비교

## 관련 개념

[[knowledge-distillation]], [[fine-tuning]], [[quantization]]
