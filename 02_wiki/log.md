---
title: 위키 로그
created: 2026-04-15
updated: 2026-04-15
---

# 위키 로그

시간순 작업 기록. Append-only.

## [2026-04-15] init | 위키 초기화 (클린 리셋)

- 세컨드 브레인 위키 구조 확립 완료
- 폴더: raw/, wiki/, output/, templates/ 세팅
- steering: context.md (인과 관계 컨벤션 포함), ingest.md, query.md, lint.md
- 테스트 인제스트 데이터 정리, 워크플로우 시나리오 문서 보존
- raw/ 원본 데이터 보존 (papers, transcripts, articles)

## [2026-04-15] ingest | 3개 원본 일괄 인제스트

- 원본 1: `raw/papers/discovering-sota-rl-algorithms/discovering-sota-rl-algorithms.md` (Nature 논문)
- 원본 2: `raw/transcripts/카파시도 못 말한 LLM-Wiki의 진실...` (NEWIT 유튜브)
- 원본 3: `raw/articles/2026-04-14_LLM-Wiki - LLM을 활용하여...` (GeekNews 아티클)
- 생성된 페이지:
  - sources: 3개 (discovering-sota-rl-algorithms, llm-wiki-truth-why-pkm-fails, llm-wiki-karpathy-guide)
  - entities: 3개 (junhyuk-oh, discorl, andrej-karpathy)
  - concepts: 6개 (meta-learning, reinforcement-learning, bootstrapping, rag, mcp, pkm)
- 인과 관계 callout 5개 삽입
- index.md 갱신 완료

## [2026-04-15] setup | Graphify 그래프 탐색 통합

- `scripts/build_graph.py` 생성 — wiki/ 마크다운에서 위키링크+인과 callout 파싱 → graph.json 생성
- `scripts/graph_tools.py` 생성 — explain, path, causal 명령 지원
- graph.json 생성 완료: 27 노드, 63 엣지 (위키링크 61 + 인과 2)
- query 스킬에 그래프 기반 탐색 단계(2.5단계) 추가
- Graphify CLI의 query 명령도 정상 작동 확인

## [2026-04-15] refactor | 볼트 구조 정리 + GRAPH_REPORT 추가

- 숨김 폴더 전환: scripts/ → .scripts/, graphify-out/ → .graphify-out/, templates/ → .templates/
- 옵시디언에서 보이는 폴더: raw/, wiki/, output/ 3개로 집중
- .scripts/build_graph.py에 GRAPH_REPORT.md 자동 생성 기능 추가
- 현재 그래프: 27 노드, 63 엣지 (위키링크 61 + 인과 2), 미해결 노드 13개
- 모든 steering 스킬의 경로 업데이트 완료
