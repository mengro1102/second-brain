# brainforge-mcp Phase 2 구현 로드맵

## 프로젝트 현황

- 저장소: `/mnt/d/Obsidian/llm-wiki-mcp/`
- PyPI: `brainforge-mcp` v0.1.0 배포 완료
- GitHub: `mengro1102/brainforge-mcp` (push 완료)
- 커밋 4개: initial → rename → docs+examples → link fix

## 현재 파일 구조

```
/mnt/d/Obsidian/llm-wiki-mcp/
├── pyproject.toml              # name="brainforge-mcp", entry="llm_wiki_mcp.server:main"
├── README.md                   # A-to-Z 튜토리얼 포함
├── LICENSE                     # MIT
├── examples/wiki/              # 샘플 위키 (LoRA 예제)
│   ├── sources/lora-explained.md
│   ├── concepts/lora.md, fine-tuning.md, transformer.md
│   └── index.md
└── src/llm_wiki_mcp/
    ├── __init__.py             # v0.1.0
    ├── graph.py                # WikiGraph 클래스 (빌드, 파싱, 탐색)
    └── server.py               # MCP 서버 (5개 도구 + init 명령)
```

## 현재 도구 (Phase 1 완료)

| 도구 | 상태 | 설명 |
|------|------|------|
| `explain_node` | ✅ | 노드 프로파일 + 의미 해석 |
| `find_path` | ✅ | BFS 경로 탐색 + 연결 강도 해석 |
| `causal_chain` | ✅ | 인과 네트워크 상류/하류 분석 |
| `graph_summary` | ✅ | 위키 건강 리포트 |
| `rebuild_graph` | ✅ | 그래프 재빌드 |

---

## Phase 2 구현 계획

### 목표
brainforge-mcp만 설치하면 누구나 Karpathy LLM Wiki 패턴의 세컨드 브레인을 갖출 수 있도록:
1. ingest/query/lint/output 워크플로우를 MCP 도구로 추가
2. CONTEXT.md 템플릿으로 사용자 맞춤형 위키 운영
3. Obsidian Web Clipper 템플릿 포함
4. README를 완전한 가이드로 재작성

### Task 1: CONTEXT.md 템플릿 생성

`init` 시 생성되는 `CONTEXT.md` 파일. 사용자가 커스터마이징하는 설정 파일 역할.

**파일 위치**: 볼트 루트 (`~/my-brain/CONTEXT.md`)

**내용 구조**:
```markdown
# 나의 세컨드 브레인 컨텍스트

## 나는 누구인가
- 이름: (이름)
- 분야: (관심 분야)
- 목표: (이 위키로 달성하고 싶은 것)

## 위키 운영 규칙
- 언어: 한국어 (기술 용어는 영어 병기)
- 톤: 간결하고 핵심 중심
- 핵심 우선: Kernel(핵심)을 먼저, 세부사항은 그 다음

## 인제스트 규칙
- 모든 wiki 페이지에 YAML frontmatter 포함 (title, created, updated, tags, sources)
- 위키링크 [[페이지명]]으로 페이지 간 연결 유지
- 새 정보가 기존 내용과 모순되면 `> [!warning] 모순 발견`으로 표기
- 인과 관계는 `> [!causal]` callout으로 명시

## 인과 관계 표기
> [!causal] 인과 관계
> [[원인]] →(관계 동사)→ [[결과]]
> 신뢰도: 높음|중간|낮음 | 출처: [[소스 페이지]]

허용 관계 유형:
- →(가능하게 함)→, →(성능 향상)→, →(성능 저하)→
- →(기반이 됨)→, →(발전시킴)→, →(대체함)→
- →(포함함)→, →(적용됨)→

## 아웃풋 설정
- 블로그 프레임워크: Jekyll / Hugo / 없음
- 블로그 URL: (GitHub Pages URL 등)
- 포스트 카테고리: 공부, 포트폴리오, 개인
```

**구현**: `server.py`의 `INIT_STRUCTURE`에 추가. `ingest`/`query`/`output` 도구가 이 파일을 읽어서 LLM에게 컨텍스트로 제공.

### Task 2: ingest 도구 구현

**도구명**: `ingest`
**하는 일**: raw/ 스캔 → 미인제스트 파일 탐지 → LLM이 위키 페이지를 생성할 수 있도록 구조화된 가이드 반환

**로직**:
1. raw/ 하위 모든 .md 파일 스캔
2. wiki/log.md를 읽어 이미 인제스트된 파일 경로 추출
3. log에 없는 새 파일만 선별
4. 각 새 파일에 대해:
   - 파일 내용 읽기 (frontmatter + 본문)
   - CONTEXT.md 읽기
   - 인제스트 가이드 반환:
     - 원본 내용 요약
     - 생성해야 할 wiki 페이지 목록 (sources/, concepts/, entities/)
     - frontmatter 템플릿
     - 인과 관계 삽입 가이드
     - index.md, log.md 갱신 지침

**반환 형태**: LLM이 바로 실행할 수 있는 구조화된 지침 (마크다운)

**핵심**: MCP 도구는 "무엇을 써야 하는지" 알려주고, LLM이 실제로 파일을 생성.

### Task 3: query 도구 구현

**도구명**: `query`
**하는 일**: 질문에 관련된 위키 페이지를 탐색하여 LLM에게 답변 소스 제공

**로직**:
1. 질문 키워드로 index.md에서 관련 페이지 탐색
2. graph.json에서 관련 노드 + 연결 탐색
3. 관련 페이지 내용을 읽어서 반환
4. CONTEXT.md의 규칙에 따라 답변 형식 가이드 포함

**반환 형태**:
- 관련 페이지 목록 + 내용
- 인과 관계 체인
- 답변 시 인용해야 할 출처
- syntheses/ 저장 여부 판단 기준

### Task 4: lint 도구 구현

**도구명**: `lint`
**하는 일**: 위키 건강 점검 → 문제 목록 + 자동 수정 가이드 반환

**로직**:
1. index.md의 등록 페이지 vs 실제 파일 비교 → 누락/불일치 탐지
2. 모든 위키링크 검증 → 깨진 링크 탐지
3. frontmatter 필수 필드 누락 확인
4. 인과 관계 순환/상충 탐지
5. 고아 페이지 (인바운드 링크 없음) 탐지
6. 미해결 노드 (위키링크는 있지만 페이지 없음) 탐지

**반환 형태**:
- 문제 목록 (자동 수정 가능 / 사용자 판단 필요 분류)
- 각 문제에 대한 수정 지침
- log.md 기록 지침

### Task 5: output 도구 구현

**도구명**: `output`
**하는 일**: wiki/ 페이지 기반으로 블로그/포트폴리오 마크다운 생성 가이드

**로직**:
1. 요청된 주제에 관련된 wiki/ 페이지 수집
2. CONTEXT.md의 아웃풋 설정 읽기
3. 인과 관계 체인 추적하여 논리 구조 파악
4. 아웃풋 유형별 (논문 리뷰, 개념 딥다이브, 비교 분석) 구조 가이드 반환

**반환 형태**:
- 참조할 wiki 페이지 목록 + 내용
- 포스트 구조 제안 (섹션 구성)
- frontmatter 템플릿 (Jekyll/Hugo 호환)
- 위키링크 → URL 변환 가이드

### Task 6: Obsidian Web Clipper 템플릿

`init` 시 `.templates/` 폴더에 생성. Obsidian Web Clipper에서 바로 사용 가능.

**포함할 템플릿**:
- `article-clipper.json` — 웹 기사 클리핑 → raw/articles/
- `research-clipper.json` — 논문 클리핑 (arxiv, ieee 등) → raw/papers/
- `youtube-clipper.json` — 유튜브 영상 → raw/transcripts/

**각 템플릿 구조** (현재 내 볼트의 .templates/ 기반, 간소화):
```json
{
  "schemaVersion": "0.1.0",
  "name": "Article",
  "behavior": "create",
  "noteContentFormat": "## 핵심 요약\n\n> 클리핑 후 직접 작성하거나 비워두세요.\n\n---\n\n## 원문\n\n{{content}}",
  "properties": [
    {"name": "title", "value": "{{title}}", "type": "text"},
    {"name": "source", "value": "{{url}}", "type": "text"},
    {"name": "created", "value": "{{date}}", "type": "date"},
    {"name": "type", "value": "article", "type": "text"},
    {"name": "tags", "value": "clippings", "type": "multitext"},
    {"name": "status", "value": "inbox", "type": "text"}
  ],
  "noteNameFormat": "{{date|date:\"YYYY-MM-DD\"}}_{{title|slice:0,80}}",
  "path": "raw/articles"
}
```

### Task 7: init 명령 확장

현재 `INIT_STRUCTURE`에 추가할 파일:
- `CONTEXT.md` — 사용자 컨텍스트 템플릿
- `.templates/article-clipper.json`
- `.templates/research-clipper.json`
- `.templates/youtube-clipper.json`

### Task 8: README 최종 재작성

추가할 섹션:
- **Obsidian Web Clipper 설정 가이드** (스크린샷 없이 텍스트로)
- **CONTEXT.md 커스터마이징 가이드**
- **워크플로우 도구 사용 예시** (ingest, query, lint, output)
- **MCP 도구 vs LLM 역할 명확화** 표
- **FAQ**: "옵시디언 없이도 쓸 수 있나요?", "영어로도 쓸 수 있나요?" 등

### Task 9: 버전 업 + PyPI 재배포

- `__init__.py` 버전을 `0.2.0`으로 업
- `pyproject.toml` 버전 업
- `python3 -m build && python3 -m twine upload dist/*`

---

## 구현 순서 (의존성 기반)

```
Task 1 (CONTEXT.md) ──┐
Task 6 (Clipper 템플릿) ──┤
Task 7 (init 확장) ────────┤── Task 2 (ingest) ──┐
                           │                      ├── Task 8 (README)
                           ├── Task 3 (query) ────┤
                           ├── Task 4 (lint) ─────┤
                           └── Task 5 (output) ───┘── Task 9 (배포)
```

**권장 실행 순서**:
1. Task 1 + 6 + 7 (init 확장 — CONTEXT.md + 클리퍼 템플릿)
2. Task 2 (ingest)
3. Task 3 (query)
4. Task 4 (lint)
5. Task 5 (output)
6. Task 8 (README 최종)
7. Task 9 (v0.2.0 배포)

---

## 새 채팅에서 이어가기 위한 프롬프트

```
brainforge-mcp (PyPI: brainforge-mcp v0.1.0) 프로젝트의 Phase 2를 구현해줘.

프로젝트 위치: /mnt/d/Obsidian/llm-wiki-mcp/
로드맵: /mnt/d/Obsidian/llm-wiki-mcp/ROADMAP.md (또는 .scripts/llm-wiki-mcp/ROADMAP.md)

현재 상태:
- Phase 1 완료: explain_node, find_path, causal_chain, graph_summary, rebuild_graph
- PyPI v0.1.0 배포 완료
- GitHub push 완료

Phase 2 구현 대상 (ROADMAP.md 참조):
- Task 1: CONTEXT.md 템플릿
- Task 2: ingest 도구
- Task 3: query 도구
- Task 4: lint 도구
- Task 5: output 도구
- Task 6: Obsidian Web Clipper 템플릿
- Task 7: init 명령 확장
- Task 8: README 최종 재작성
- Task 9: v0.2.0 PyPI 배포

순서대로 진행해줘.
```

---

## 참고: 내 볼트의 steering 파일 (구현 참고용)

현재 내 볼트(.kiro/steering/)에 있는 워크플로우 정의가 MCP 도구 구현의 레퍼런스:
- `ingest.md` → Task 2의 로직 참고
- `query.md` → Task 3의 로직 참고
- `lint.md` → Task 4의 로직 참고
- `output.md` → Task 5의 로직 참고
- `context.md` → Task 1의 템플릿 참고

이 파일들은 현재 워크스페이스(`/mnt/d/Obsidian/second-brain/.kiro/steering/`)에 있음.
