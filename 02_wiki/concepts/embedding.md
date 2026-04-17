---
title: 임베딩 (Embedding)
created: 2026-04-15
updated: 2026-04-15
tags: [개념, NLP, LLM]
sources: [01_raw/transcripts/2026-04-15_What is BPE Byte-Pair Encoding.md]
---

# 임베딩 (Embedding)

단어의 의미를 고차원 벡터 공간의 좌표로 변환하는 기술. 언어 모델이 세상을 인식하는 기저 감각 기관.

## Kernel

토큰에 부여된 ID(숫자)는 그 자체로 의미가 없다. 임베딩은 이 ID를 수천 차원의 벡터 공간 상의 좌표로 변환하여, 의미가 비슷한 단어는 가까이, 다른 단어는 멀리 배치한다.

## 방법론

### 벡터 공간 표현

- 각 토큰 → 고차원 벡터 (예: 768차원, 1024차원)
- 의미적 유사도 = 벡터 간 코사인 유사도
- 벡터 연산으로 의미 관계 표현 가능

### 유명한 예시

```
[King] - [Man] + [Woman] ≈ [Queen]
```

이 연산이 성립하는 이유: 임베딩 공간에서 "성별" 방향과 "왕족" 방향이 독립적인 축으로 인코딩되어 있기 때문.

### 핵심 키워드

- **벡터 공간 (Vector Space)**: 단어가 배치되는 고차원 좌표계. 차원 수가 높을수록 표현력 증가.
- **코사인 유사도 (Cosine Similarity)**: 두 벡터 간 각도로 의미적 거리 측정. 1에 가까울수록 유사.
- **임베딩 테이블 (Embedding Table)**: 어휘의 각 토큰 ID → 벡터 매핑을 저장하는 룩업 테이블. 학습 가능한 파라미터.
- **PLE (Per-Layer Embeddings)**: Gemma 4의 혁신. 단일 임베딩이 아닌 레이어별 토큰 특화 신호를 주입. 각 레이어가 필요한 시점에 토큰 정보를 받음.

> [!causal] 인과 관계
> [[tokenization|토큰화]] →(기반이 됨)→ [[embedding|임베딩]]의 입력 단위 결정
> 신뢰도: 높음 | 출처: [[bpe-byte-pair-encoding]]

## Gemma 4에서의 적용

Gemma 4는 표준 임베딩 + PLE(Per-Layer Embeddings)를 사용. PLE는 레이어별로 토큰 특화 잔차 신호를 주입하여, 단일 임베딩에 모든 정보를 압축하는 한계를 극복.

## 관련

[[tokenization]], [[transformer]], [[attention]]
- [[bpe-byte-pair-encoding]], [[gemma4-huggingface]]
