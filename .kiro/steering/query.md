---
inclusion: manual
---

# /query — 위키 기반 질의응답

02_wiki/ 문서를 참고하여 사용자의 질문에 답변하는 워크플로우.

## 트리거

사용자가 "/query" 또는 위키 내용에 대한 질문을 할 때 실행.

## 프로세스

### 1단계: 질의 분류

- **탐색형**: 단일 페이지 검색
- **연결형**: 복수 페이지 교차 참조 + 인과 관계 추적
- **합성형**: 전체 위키 종합

### 2단계: 관련 페이지 탐색

1. `02_wiki/index.md`를 읽어 전체 위키 구조를 파악한다
2. 질문 유형에 따라 관련 페이지를 선별하고 읽는다

### 2.5단계: 그래프 기반 탐색 (연결형/합성형)

`00_graphify-out/graph.json`이 존재하면 그래프 도구를 활용:

- **노드 설명**: `python3 .scripts/graph_tools.py explain "노드명"`
- **경로 탐색**: `python3 .scripts/graph_tools.py path "A" "B"`
- **인과 체인**: `python3 .scripts/graph_tools.py causal "노드명"`
- **Graphify query**: `graphify query "질문" --graph 00_graphify-out/graph.json`

graph.json이 오래된 경우: `python3 .scripts/build_graph.py`로 재빌드.

### 3단계: 답변 생성

- 02_wiki/ 페이지 내용 기반으로 답변 생성
- 위키에 없는 내용은 "위키에 해당 정보가 없습니다"라고 명시
- 모든 주장에 출처 위키 페이지를 `[[페이지명]]`으로 인용
- 인과 관계는 `> [!causal]` callout 형식으로 표시

### 4단계: 답변 보존 판단

보존 시 `02_wiki/syntheses/`에 저장하고 index.md, log.md 갱신.

## 규칙

- 02_wiki/ 페이지만 참조 (01_raw/는 직접 참조하지 않음)
- 위키에 없는 내용을 지어내지 않는다
- 출처를 반드시 인용한다

## 참조 파일

- #[[file:02_wiki/index.md]]
- #[[file:02_wiki/log.md]]
- #[[file:.kiro/steering/context.md]]
