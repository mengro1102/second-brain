---
title: "LLM-Wiki: LLM을 활용하여 개인 지식저장소 구축하기"
created: 2026-04-15
updated: 2026-04-15
tags: [아티클, LLM-Wiki, Karpathy, PKM]
sources: [raw/articles/2026-04-14_LLM-Wiki - LLM을 활용하여 개인 지식저장소 구축 하기.md]
type: article
status: ingested
---

# LLM-Wiki: LLM을 활용하여 개인 지식저장소 구축하기

- 출처: https://news.hada.io/topic?id=28208 (GeekNews)
- 원저자: [[Andrej Karpathy]] (GitHub Gist)

## Kernel

LLM이 직접 위키를 작성·관리하는 영속적 위키(persistent wiki) 패턴. RAG처럼 매번 원문에서 재추출하는 것이 아니라, 지식을 한 번 컴파일하고 점진적으로 축적. 유지보수 비용을 LLM이 거의 0으로 낮춤.

## 핵심 주장

1. 3레이어 아키텍처: Raw Sources(불변) → Wiki(LLM 소유) → Schema(설정)
2. 3대 워크플로우: Ingest(원본→위키), Query(질의→답변), Lint(건강점검)
3. index.md + log.md로 ~100개 소스 규모에서 벡터 DB 없이 작동
4. 위키 유지보수의 핵심 장벽(북키핑 비용)을 LLM이 해결
5. Karpathy의 4가지 장점: 명시성, 데이터 소유권, 파일 우선, AI 선택 자유

> [!causal] 인과 관계
> LLM 유지보수 비용 ≈ 0 →(가능하게 함)→ 영속적 위키의 장기 생존
> 신뢰도: 높음 | 출처: [[llm-wiki-karpathy-guide]]

> [!causal] 인과 관계
> [[Andrej Karpathy]]의 LLM Wiki 패턴 →(기반이 됨)→ 이명로의 세컨드 브레인 아키텍처
> 신뢰도: 높음 | 출처: [[llm-wiki-karpathy-guide]]

## 캡처 맥락

- 목적: 세컨드 브레인 구축의 이론적 기반 문서
- 연결: 현재 볼트의 아키텍처가 이 가이드를 직접 구현한 것
- 다음 단계: 이 패턴 위에 인과 관계 컨벤션, Graphify 등 확장

## 관련 개념

[[rag]], [[mcp]], [[pkm]]

## 관련 엔티티

[[Andrej Karpathy]], [[Obsidian]]
