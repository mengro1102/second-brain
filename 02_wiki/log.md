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

## [2026-04-15] ingest | Gemma 4 블로그 관련 4개 소스 인제스트

- 원본 1: `01_raw/transcripts/2026-04-15_Welcome Gemma 4...` (Gemma 4 HF 블로그)
- 원본 2: `01_raw/transcripts/2026-04-15_What is BPE...` (BPE 토큰화)
- 원본 3: `01_raw/transcripts/2026-04-15_6.3.2 GPTQ, AWQ, GGUF 비교.md` (양자화 비교)
- 원본 4: `01_raw/transcripts/2026-04-15_RAttention...` + PDF (RAttention 논문)
- 생성된 페이지:
  - sources: 4개 (gemma4-huggingface, bpe-byte-pair-encoding, gptq-awq-gguf-comparison, rattention-sliding-window)
  - concepts: 6개 (transformer, attention, quantization, tokenization, embedding, sliding-window-attention)
- 인과 관계 callout 6개 삽입
- 캡처 맥락: Gemma 4 블로그 포스트 작성을 위한 참조 소스 수집

## [2026-04-15] output | Gemma 4를 이해하는 과정 블로그 포스트 생성

- 유형: 개념 딥다이브
- 저장: 03_output/blog/gemma4-understanding.md
- 참조 wiki 페이지: gemma4-huggingface, bpe-byte-pair-encoding, gptq-awq-gguf-comparison, rattention-sliding-window, tokenization, embedding, attention, transformer, quantization, sliding-window-attention
- 구조: 7개 챕터 (토큰화/임베딩 → 트랜스포머/어텐션 → 학습 파이프라인 → 양자화/SLM → 멀티모달 → Gemma 4 해부 → 결론)

## [2026-04-15] output | Gemma 4 블로그 — Jekyll 구조 + 개념 링크 포스트 생성

- Jekyll 호환 구조 세팅: _config.yml, _posts/, _concepts/ 컬렉션
- 메인 포스트: _posts/2026-04-15-gemma4-understanding.md (키워드 링크 삽입)
- 개념 독립 포스트 6개: tokenization, embedding, attention, transformer, quantization, sliding-window-attention
- 전파거북이 스타일: 메인 포스트 → 개념 포스트 → 상호 링크 체인
- GitHub Pages 이식 준비 완료: git push만 하면 배포

## [2026-04-16] ingest | LoRA 논문 쉽게 설명하기

- 원본: `01_raw/transcripts/2026-04-16_LoRA 논문 쉽게 설명하기.md`
- 생성된 페이지:
  - `02_wiki/sources/lora-paper-explained.md` — 소스 요약
  - `02_wiki/concepts/lora.md` — LoRA 개념 (저랭크 분해, PEFT, Rank, Freeze 등)
  - `02_wiki/concepts/fine-tuning.md` — Fine-Tuning 개념 (유형 비교 테이블)
- 인과 관계 callout 4개 삽입
- 캡처 맥락: LLM Fine-Tuning 흥미, 로컬 오픈소스 LLM 파인튜닝 사전학습, EH R&C 도메인 특화 LLM 구축 탐색

## [2026-04-30] ingest | 신규 raw 데이터 13건 일괄 인제스트

- 주제별: Nemotron 3 Super, Hermes Agent, AI 연구 에이전트(3건), MiroFish, Gemma 4 비주얼 가이드, 파르카에, Claude Code 캐싱, Qwen3.6, UNIST XAI, 커서 AI 인수, Kimi K2.5 논문
- 생성: sources 13개, concepts 1개(ai-agent)
- 캡처 맥락: AI 에이전트 생태계, SLM 효율화, LLM 경쟁 동향, 연구 자동화 도구 탐색
