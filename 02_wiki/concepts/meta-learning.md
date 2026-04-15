---
title: 메타러닝 (Meta-Learning)
created: 2026-04-15
updated: 2026-04-15
tags: [개념, 강화학습, AI]
sources: [raw/papers/discovering-sota-rl-algorithms/discovering-sota-rl-algorithms.md]
---

# 메타러닝 (Meta-Learning)

"학습하는 방법을 학습하는 것." 느린 메타학습 프로세스가 빠른 학습/적응 프로세스를 최적화.

## Kernel

일반 학습: 데이터 → 모델. 메타러닝: 태스크들 → 학습 알고리즘. 학습 규칙 자체가 최적화 대상.

## DiscoRL에서의 역할

[[discorl]]은 메타러닝의 극단적 적용 — RL 업데이트 규칙 자체를 메타네트워크로 표현하고 다양한 환경 경험으로부터 발견.

> [!causal] 인과 관계
> [[메타러닝]] →(가능하게 함)→ [[discorl]]의 RL 규칙 자동 발견
> 신뢰도: 높음 | 출처: [[02_wiki/sources/discovering-sota-rl-algorithms]]

## 관련

[[강화학습]], [[부트스트래핑]], [[02_wiki/sources/discovering-sota-rl-algorithms]]
