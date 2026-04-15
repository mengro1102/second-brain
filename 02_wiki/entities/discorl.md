---
title: DiscoRL
created: 2026-04-15
updated: 2026-04-15
tags: [모델, 강화학습, 메타러닝]
sources: [raw/papers/discovering-sota-rl-algorithms/discovering-sota-rl-algorithms.md]
---

# DiscoRL

메타러닝으로 자동 발견된 RL 규칙. [[Junhyuk Oh]] 등, [[Google DeepMind]].

## Kernel

에이전트의 예측과 정책을 업데이트하는 "RL 규칙" 자체를 메타네트워크로 학습. 사전 정의 없이 예측(y, z)의 의미를 자동 발견.

## 변형

- Disco57: Atari 57에서 메타학습
- Disco103: Atari + ProcGen + DMLab-30 (103개 환경). 더 범용적.

## 출처

- [[02_wiki/sources/discovering-sota-rl-algorithms]]
