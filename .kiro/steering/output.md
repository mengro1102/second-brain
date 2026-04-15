---
inclusion: manual
---

# /output — 위키에서 결과물 생성

02_wiki/ 문서를 기반으로 블로그 포스트, 포트폴리오, 발표자료 등 최종 결과물을 생성하는 워크플로우.

## 트리거

사용자가 "/output", "블로그 써줘", "포스트 만들어줘" 등을 요청할 때 실행.

## 아웃풋 유형

| 유형 | 소스 | 저장 경로 |
|------|------|----------|
| 논문 리뷰 포스트 | 02_wiki/sources/ + concepts/ | 03_output/blog/ |
| 개념 딥다이브 | 02_wiki/concepts/ + sources/ | 03_output/blog/ |
| 프로젝트 기록 | 02_wiki/sources/ + syntheses/ | 03_output/portfolio/ |
| 비교/종합 분석 | 02_wiki/syntheses/ 또는 교차 | 03_output/blog/ |
| 발표자료 (Marp) | 위 유형 중 하나 | 03_output/slides/ |

## 프로세스

### 1단계: 요청 분석

1. 아웃풋 유형 판별
2. 관련 02_wiki/ 페이지 식별 (index.md 참조)

### 2단계: 소스 수집

1. 해당 02_wiki/ 페이지들을 읽는다
2. 인과 관계 callout 추적하여 관련 개념 체인 파악

### 3단계: 결과물 생성

- 유형별 구조에 맞춰 마크다운 작성
- 위키링크는 일반 텍스트/링크로 변환 (외부 공개용)
- GitHub Pages 호환 마크다운

### 4단계: 저장 및 기록

1. 03_output/ 하위 적절한 폴더에 저장
2. 02_wiki/log.md에 기록

## 규칙

- 02_wiki/ 페이지를 기반으로만 작성 (01_raw/ 직접 참조하지 않음)
- 위키에 없는 내용을 지어내지 않는다
- 한국어 기본, 기술 용어 영어 병기
- 핵심(Kernel)을 먼저, 세부사항은 그 다음

## 참조 파일

- #[[file:02_wiki/index.md]]
- #[[file:02_wiki/log.md]]
- #[[file:.kiro/steering/context.md]]
- #[[file:03_output/.steering.md]]
