---
title: "LLM Wiki 동작 시나리오: 논문 PDF에서 지식 네트워크까지"
created: 2026-04-15
updated: 2026-04-15
tags: [종합, 워크플로우, LLM-Wiki]
sources: []
---

# LLM Wiki 동작 시나리오: 논문 PDF에서 지식 네트워크까지

## 상황 설정

명로가 흥미로운 논문 PDF를 발견했다. 예를 들어 Junhyuk Oh의 "Discovering State-of-the-Art RL Algorithms" (Nature, 2025). 이 논문을 세컨드 브레인에 넣고, 기존 지식과 연결하고, 나중에 블로그로 쓰고 싶다.

---

## Phase 1: 원본 수집 (Capture)

```
[논문 PDF 다운로드]
       ↓
  raw/papers/ 에 저장
       ↓
  marker_single 로 마크다운 변환
       ↓
  raw/papers/논문명/논문명.md 생성 (+ 이미지 추출)
```

**무슨 일이 일어나는가:**
- PDF를 `raw/papers/`에 드롭한다
- 터미널에서 `marker_single "raw/papers/논문.pdf" --output_dir raw/papers/` 실행
- marker가 OCR + 레이아웃 분석으로 마크다운 변환 (표, 수식, 이미지 포함)
- 원본 PDF와 변환된 마크다운이 모두 `raw/`에 보존된다

**핵심 원칙:** `raw/`는 불변. AI도 사람도 절대 수정하지 않는다. 이것이 정보 무결성의 출발점.

---

## Phase 2: 인제스트 (Ingest)

```
  사용자: "#ingest 인제스트해줘"
       ↓
  ┌─────────────────────────────────┐
  │  1단계: 새 파일 탐지             │
  │  - raw/ 스캔                    │
  │  - wiki/log.md와 대조           │
  │  - 미인제스트 파일 선별           │
  └─────────────┬───────────────────┘
                ↓
  ┌─────────────────────────────────┐
  │  2단계: 원본 읽기 + 소스 요약     │
  │  - 핵심 주장 (Key Claims) 추출   │
  │  - 엔티티 식별                   │
  │  - 개념 (Concepts) 추출          │
  │  - 사용자에게 3가지 질문          │
  └─────────────┬───────────────────┘
                ↓
       사용자 답변 대기
       (왜 캡처? 어떻게 연결? 뭘 하고 싶어?)
                ↓
  ┌─────────────────────────────────┐
  │  3단계: wiki/ 반영               │
  │  - sources/ 소스 요약 페이지 생성 │
  │  - entities/ 엔티티 페이지 생성   │
  │  - concepts/ 개념 페이지 생성     │
  │  - 인과 관계 callout 삽입        │
  │  - index.md 갱신                │
  │  - log.md 기록                  │
  └─────────────────────────────────┘
```

**무슨 일이 일어나는가 (실제 예시):**

AI가 논문을 읽고 다음을 생성한다:

1. **`wiki/sources/discovering-sota-rl-algorithms.md`** — 논문 소스 요약
   - Kernel: "메타러닝으로 SOTA RL 알고리즘을 자동 발견"
   - 핵심 주장 5개, 방법론, 실험 결과 정리
   - 사용자의 캡처 맥락 반영

2. **`wiki/entities/junhyuk-oh.md`** — 인물 엔티티
   - 소속, 연구 분야, 주요 논문 목록

3. **`wiki/entities/discorl.md`** — 모델 엔티티
   - Disco57, Disco103 변형, 성과 비교

4. **`wiki/concepts/meta-learning.md`** — 개념 페이지
   - Kernel 먼저, 이 논문에서의 역할, 역사, 관련 개념 링크

5. **인과 관계 삽입:**
   ```markdown
   > [!causal] 인과 관계
   > [[메타러닝]] →(가능하게 함)→ [[DiscoRL]]의 RL 규칙 자동 발견
   > 신뢰도: 높음 | 출처: [[discovering-sota-rl-algorithms]]
   ```

---

## Phase 3: 지식 네트워크 형성 (Knowledge Graph)

```
인제스트 후 옵시디언 그래프 뷰:

        [메타러닝] ←──── [DiscoRL 논문]
            │                  │
            ↓                  ↓
      [부트스트래핑]      [Junhyuk Oh]
            │                  │
            ↓                  ↓
    [TD Learning]        [Google DeepMind]
                               │
                               ↓
                          [MuZero]
```

**무슨 일이 일어나는가:**
- 모든 `[[위키링크]]`가 옵시디언 그래프 뷰에서 노드와 엣지로 시각화된다
- 인과 관계 callout은 마크다운 안에서 방향성을 명시한다
- 새 논문을 인제스트할 때마다 기존 노드와 자동으로 연결된다

**예시: 두 번째 논문 인제스트 시**

만약 "Attention Is All You Need" 논문을 인제스트하면:
- `[[메타러닝]]` 개념 페이지에 새 출처가 추가되지는 않지만
- `[[Transformer]]` 개념 페이지가 새로 생성되고
- 기존 `[[DiscoRL]]` 엔티티에 "Agent Network에 Transformer 아키텍처 사용 가능" 같은 연결이 추가될 수 있다
- 그래프가 점점 밀도 있게 성장한다 → **지식의 복리 효과**

---

## Phase 4: 활용 (Query & Output)

### 질의 (Query)
```
사용자: "DiscoRL이 기존 RL 알고리즘과 다른 점이 뭐야?"
       ↓
  AI가 index.md에서 관련 페이지 탐색
       ↓
  wiki/sources/discovering-sota-rl-algorithms.md
  wiki/entities/discorl.md
  wiki/concepts/meta-learning.md
  wiki/concepts/bootstrapping.md
       ↓
  종합 답변 생성 (위키 페이지 인용 포함)
       ↓
  가치 있는 답변이면 wiki/syntheses/에 저장
```

### 아웃풋 (Output)
```
사용자: "DiscoRL 논문 리뷰를 블로그 포스트로 만들어줘"
       ↓
  wiki/sources/ + wiki/concepts/ 기반으로
       ↓
  output/blog/discorl-paper-review.md 생성
       ↓
  GitHub Pages 블로그에 게시
```

---

## Phase 5: 건강 점검 (Lint)

```
사용자: "위키 린트해줘"
       ↓
  ┌─────────────────────────────────┐
  │  점검 항목:                      │
  │  ✓ 페이지 간 모순 확인            │
  │  ✓ 고아 페이지 탐지               │
  │  ✓ 언급되었으나 페이지 없는 개념    │
  │  ✓ 누락된 교차 참조               │
  │  ✓ 인과 관계 순환 참조 탐지        │
  │  ✓ 상충 인과 관계 격리            │
  └─────────────┬───────────────────┘
                ↓
  결과를 log.md에 기록
  필요 시 페이지 갱신/생성
```

---

## 전체 데이터 흐름 요약

```
[외부 세계]                    [raw/ 불변 원본]              [wiki/ AI 위키]           [output/ 결과물]
                                                                                    
 논문 PDF ──marker──→ raw/papers/논문.md                                              
 웹 기사 ──clipper──→ raw/articles/기사.md    ──ingest──→  sources/요약.md             
 유튜브 ──clipper──→ raw/transcripts/영상.md               concepts/개념.md  ──query──→ blog/포스트.md
 메모 ────직접작성──→ raw/notes/메모.md                     entities/엔티티.md           portfolio/프로젝트.md
                                                          syntheses/종합.md           slides/발표.md
                                                          index.md (목차)
                                                          log.md (기록)
```

**핵심 설계 철학:**
1. **원본 불변**: raw/는 절대 수정 안 함 → 정보 무결성 보장
2. **AI가 유지보수**: 사람은 소스를 넣고 질문하고 방향을 잡음. 요약, 연결, 갱신은 AI가 함
3. **복리 성장**: 인제스트할수록 위키링크 밀도 증가 → 그래프가 풍부해짐 → 더 좋은 답변
4. **인과성 명시**: 단순 연결이 아닌 "왜 연결되는가"를 callout으로 기록 → 논리 추론 가능
