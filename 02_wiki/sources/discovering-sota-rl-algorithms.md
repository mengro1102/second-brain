---
title: "Discovering State-of-the-Art Reinforcement Learning Algorithms"
created: 2026-04-15
updated: 2026-04-15
tags: [논문, 강화학습, 메타러닝, DeepMind]
sources: [raw/papers/discovering-sota-rl-algorithms/discovering-sota-rl-algorithms.md]
type: paper
status: ingested
---

# Discovering State-of-the-Art RL Algorithms

- 저자: [[junhyuk-oh|Junhyuk Oh]], Gregory Farquhar, Iurii Kemaev, Dan A. Calian 등
- 소속: [[google-deepmind|Google DeepMind]]
- 출처: Nature (2025), DOI: 10.1038/s41586-025-09761-x

## Kernel

메타러닝만으로 SOTA 강화학습 알고리즘을 자동 발견할 수 있다. 발견된 규칙([[discorl]])은 Atari 57에서 MuZero 포함 모든 기존 알고리즘을 능가하며, 학습 시 본 적 없는 환경에도 일반화된다.

## 핵심 주장

1. 메타네트워크가 에이전트의 예측과 정책을 업데이트하는 "RL 규칙" 자체를 학습
2. [[discorl]]은 Atari 57에서 [[muzero|MuZero]], Dreamer 등 모든 기존 알고리즘 능가
3. 미학습 환경(ProcGen, Crafter, NetHack)에도 일반화
4. 발견된 예측(y, z)은 기존 가치함수와 다른 고유한 의미를 가짐
5. [[bootstrapping|부트스트래핑]]이 자연스럽게 출현

> [!causal] 인과 관계
> [[meta-learning|메타러닝]] →(가능하게 함)→ [[discorl]]의 RL 규칙 자동 발견
> 신뢰도: 높음 | 출처: [[discovering-sota-rl-algorithms]]

> [!causal] 인과 관계
> [[bootstrapping|부트스트래핑]] →(성능 향상)→ [[discorl]]의 예측 품질
> 신뢰도: 높음 | 출처: [[discovering-sota-rl-algorithms]]

## 캡처 맥락

- 목적: 롤모델 [[junhyuk-oh|Junhyuk Oh]]의 최근 연구 follow-up + 세컨드 브레인 테스트
- 연결: 개인 공부 — 강화학습 역량 심화를 통한 성장
- 다음 단계: 심층 논문 리뷰 → 블로그 포스트 작성

## 관련 개념

[[meta-learning|메타러닝]], [[reinforcement-learning|강화학습]], [[bootstrapping|부트스트래핑]]

## 관련 엔티티

[[junhyuk-oh|Junhyuk Oh]], [[google-deepmind|Google DeepMind]], [[discorl]], [[muzero|MuZero]]
