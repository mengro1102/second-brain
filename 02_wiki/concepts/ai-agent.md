---
title: "AI 에이전트 (AI Agent)"
created: 2026-04-30
updated: 2026-04-30
tags: [개념, LLM, 에이전트, 자동화]
sources: [hermes-agent, agentic-researcher, auto-research-claw, openharness, mirofish, claude-code-prompt-caching, kimi-k25-visual-agentic]
---

# AI 에이전트 (AI Agent)

## Kernel

LLM이 도구를 호출하고, 계획을 세우고, 자율적으로 태스크를 수행하는 시스템. 단순 챗봇을 넘어 능동적 행위자.

## 핵심 구성 요소

1. **계획 (Planning)**: 복잡한 태스크를 하위 단계로 분해
2. **도구 사용 (Tool Use)**: 외부 API, 코드 실행, 파일 시스템 등 호출
3. **메모리 (Memory)**: 단기(컨텍스트) + 장기(세션 간) 기억 유지
4. **자율성 (Autonomy)**: 인간 개입 없이 판단·실행·검증 루프 수행

## 유형

| 유형 | 설명 | 예시 |
|------|------|------|
| 코딩 에이전트 | 코드 작성·수정·테스트 자동화 | Claude Code, Cursor, OpenHarness |
| 연구 에이전트 | 문헌 조사·실험·논문 작성 보조 | Agentic Researcher, AutoResearchClaw |
| 범용 에이전트 | 다양한 도구 연동 범용 태스크 | Hermes Agent |
| 시뮬레이션 에이전트 | 다수 에이전트 상호작용 시뮬레이션 | MiroFish |
| 비전 에이전트 | 시각 정보 기반 자율 행동 | Kimi K2.5 |

## 관련

[[mcp]], [[transformer]]

## 소스

- [[hermes-agent]] — 오픈소스 AI 에이전트, 메모리 기반
- [[agentic-researcher]] — AI 도구 활용 연구 방법론
- [[auto-research-claw]] — 완전 자율 연구 에이전트
- [[openharness]] — 경량 에이전트 하네스 프레임워크
- [[mirofish]] — 집단 지능 시뮬레이션 엔진
- [[claude-code-prompt-caching]] — 에이전트 아키텍처 최적화
- [[kimi-k25-visual-agentic]] — 비전 기반 에이전트 모델
