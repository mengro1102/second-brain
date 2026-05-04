---
title: "Gemma 4 비주얼 가이드"
created: 2026-04-30
updated: 2026-04-30
tags: [아티클, Gemma, 멀티모달, 아키텍처]
sources: [01_raw/articles/2026-04-30_Gemma 4 비주얼 가이드.md]
type: article
status: ingested
---

# Gemma 4 비주얼 가이드

## Kernel

Gemma 4의 아키텍처를 시각적으로 해설. Local-Global Attention 교차 배치, GQA(Grouped Query Attention), Shared KV Cache 등 구조 상세.

## 핵심 내용

1. **Local-Global Attention 교차 배치**: 로컬과 글로벌 어텐션을 번갈아 배치하여 효율성과 성능 균형
2. **GQA (Grouped Query Attention)**: 쿼리 그룹이 KV 헤드를 공유하여 메모리 절감
3. **Shared KV Cache**: KV 캐시 공유로 추론 효율 향상
4. **시각적 해설**: 복잡한 아키텍처를 직관적 다이어그램으로 설명

## 캡처 맥락

- 목적: Gemma 4 블로그 포스트 보강 자료
- 연결: 기존 [[gemma4-huggingface]] 소스와 상호 보완
- 다음 단계: 블로그 포스트에 아키텍처 상세 섹션 추가

## 관련 개념

[[attention]], [[sliding-window-attention]], [[transformer]]
