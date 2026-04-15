---
title: 부트스트래핑 (Bootstrapping)
created: 2026-04-15
updated: 2026-04-15
tags: [개념, 강화학습]
sources: [raw/papers/discovering-sota-rl-algorithms/discovering-sota-rl-algorithms.md]
---

# 부트스트래핑 (Bootstrapping)

현재 예측의 타겟을 구성할 때 미래의 예측값을 사용하는 메커니즘. RL의 핵심 아이디어.

## Kernel

완전한 리턴을 기다리지 않고, 현재 추정치를 미래 추정치로 업데이트. TD Learning의 근간.

## DiscoRL에서의 발견

[[discorl]]에서 부트스트래핑이 자연스럽게 출현. 제거 시 성능 크게 하락.

## 관련

[[강화학습]], [[메타러닝]], [[02_wiki/sources/discovering-sota-rl-algorithms]]
