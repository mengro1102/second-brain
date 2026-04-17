---
title: "What is BPE: Byte-Pair Encoding?"
created: 2026-04-15
updated: 2026-04-15
tags: [아티클, 토큰화, NLP]
sources: [01_raw/transcripts/2026-04-15_What is BPE Byte-Pair Encoding.md]
type: article
status: ingested
---

# What is BPE: Byte-Pair Encoding?

- 저자: Guangyuan Piao
- 출처: https://medium.com/@parklize/what-is-bpe-byte-pair-encoding-5f1ea76ea01f

## Kernel

BPE는 텍스트 압축 알고리즘에서 유래한 서브워드 토큰화 기법. 자주 등장하는 글자 조합을 반복적으로 병합하여 어휘를 구축. GPT 등 대부분의 LLM이 사용.

## 핵심 주장

1. BPE는 단어를 문자 단위로 분할한 뒤, 가장 빈도 높은 쌍을 반복 병합
2. 초기 어휘 = 개별 문자 집합 → 병합 반복 → 서브워드 어휘 확장
3. 미등록 단어(OOV) 문제를 해결: 어떤 단어든 서브워드 조합으로 표현 가능
4. OpenAI GPT, 대부분의 Transformer 모델이 BPE 기반 토크나이저 사용

> [!causal] 인과 관계
> [[tokenization|토큰화(BPE)]] →(기반이 됨)→ [[embedding|임베딩]]의 입력 단위 결정
> 신뢰도: 높음 | 출처: [[bpe-byte-pair-encoding]]

## 캡처 맥락

- 목적: Gemma 4 블로그 챕터 1(토큰화/임베딩) 참조 소스
- 연결: 블로그에서 BPE 알고리즘의 직관적 설명에 활용

## 관련 개념

[[tokenization]], [[embedding]], [[transformer]]
