---
title: "OpenHarness: Claude Code보다 44배 가벼운 Python 기반 오픈소스 AI 에이전트 하네스 프레임워크 (feat. HKUDS)"
source: "https://discuss.pytorch.kr/t/openharness-claude-code-44-python-ai-feat-hkuds/9559"
author:
  - "[[9bow]]"
published: 2026-04-04
created: 2026-04-30
description: "OpenHarness: Claude Code보다 44배 가벼운 Python 기반 오픈소스 AI 에이전트 하네스 프레임워크1405×520 72.4 KB OpenHarness 소개 AI 에이전트가 실제로 작동하려면 단순히 강력한 언어 모델(LLM) 하나만"
type: "article"
tags:
  - "clippings"
  - "article"
status: "inbox"
---
## 핵심 요약

> 클리핑 후 직접 작성하거나 비워두세요.

---

## 원문

[![OpenHarness: Claude Code보다 44배 가벼운 Python 기반 오픈소스 AI 에이전트 하네스 프레임워크](https://discuss.pytorch.kr/uploads/default/optimized/3X/6/b/6be85233f3e9d9627e6950073ba68ef644dc292d_2_1028x380.png)

OpenHarness: Claude Code보다 44배 가벼운 Python 기반 오픈소스 AI 에이전트 하네스 프레임워크1405×520 72.4 KB

](https://discuss.pytorch.kr/uploads/default/original/3X/6/b/6be85233f3e9d9627e6950073ba68ef644dc292d.png "OpenHarness: Claude Code보다 44배 가벼운 Python 기반 오픈소스 AI 에이전트 하네스 프레임워크")

## OpenHarness 소개

AI 에이전트가 실제로 작동하려면 단순히 강력한 언어 모델(LLM) 하나만으로는 부족합니다. LLM이 파일을 읽고, 셸 명령어를 실행하고, 웹을 검색하고, 다른 에이전트를 생성하고, 오류로부터 복구하려면 이를 안전하게 조율하는 인프라가 필요하며, 이러한 인프라를 **에이전트 하네스(Agent Harness)** 라고 합니다. 홍콩대학교 데이터 지능 시스템(HKUDS) 그룹이 공개한 **OpenHarness**는 이 에이전트 하네스를 Python으로 깔끔하게 구현한 오픈소스 프레임워크입니다. OpenHarness는 Claude Code(TypeScript, 512,664줄)와 동등한 수준의 기능을 단 11,733줄의 Python 코드로 구현하였으며, 동일한 anthropics/skills 및 claude-code/plugins 생태계와 완벽하게 호환됩니다.

[![image](https://discuss.pytorch.kr/uploads/default/optimized/3X/0/1/01be939e4ed5f4119679ff6cd8a7dce408b2c6bf_2_1028x573.jpeg)

image1376×768 144 KB

](https://discuss.pytorch.kr/uploads/default/original/3X/0/1/01be939e4ed5f4119679ff6cd8a7dce408b2c6bf.jpeg "image")

OpenHarness의 철학은 간결함과 이해 가능성에 있습니다. 프로덕션 AI 에이전트가 내부적으로 어떻게 동작하는지 이해하고 싶은 연구자, 빌더, 교육자들이 코드베이스를 읽고 이해하면서 자신만의 에이전트를 구축할 수 있도록 설계되었습니다. 실제로 OpenHarness는 Claude Code와 비교하여 98%의 도구(43개 vs 44개)와 61%의 명령어(54개 vs 88개)를 구현하였으며, 114개의 유닛 테스트와 6개의 E2E 테스트 스위트를 포함하고 있습니다. 2026년 4월 1일에 v0.1.0으로 첫 공개된 이후, 빠르게 커뮤니티의 관심을 받고 있습니다.

## OpenHarness의 5대 핵심 서브시스템

[![image](https://discuss.pytorch.kr/uploads/default/optimized/3X/3/8/38093918cfb6a173ce29ff14147454076d21f58f_2_1028x339.jpeg)

image1792×592 320 KB

](https://discuss.pytorch.kr/uploads/default/original/3X/3/8/38093918cfb6a173ce29ff14147454076d21f58f.jpeg "image")

OpenHarness는 에이전트 하네스를 5가지 핵심 서브시스템으로 구조화하여 구현합니다.

**에이전트 루프(Engine)** 는 OpenHarness의 심장부로, 스트리밍 방식의 툴 호출 사이클과 API 재시도, 지수 백오프 로직을 포함합니다. 병렬 툴 실행, 토큰 카운팅, 비용 추적 기능을 지원합니다. 에이전트 루프의 핵심 패턴은 다음과 같습니다.

```python
while True:
    response = await api.stream(messages, tools)

    if response.stop_reason != "tool_use":
        break  # 모델이 완료됨

    for tool_call in response.tool_uses:
        # 권한 확인 → 훅 실행 → 툴 실행 → 훅 실행 → 결과 반환
        result = await harness.execute_tool(tool_call)

    messages.append(tool_results)
    # 루프 계속 — 모델이 결과를 보고 다음 행동 결정
```

**하네스 툴킷(Toolkit)** 은 43개 이상의 도구를 제공합니다. 파일 I/O(Bash, Read, Write, Edit, Glob, Grep), 검색(WebFetch, WebSearch, ToolSearch, LSP), 에이전트(서브에이전트 생성, 팀 조율), 태스크(백그라운드 태스크 관리), MCP(Model Context Protocol) 통합, 스케줄(크론 및 원격 트리거), 메타(스킬 로딩, 설정, 사용자 상호작용) 등 다양한 카테고리로 구성됩니다.

**컨텍스트 및 메모리(Context & Memory)** 는 CLAUDE.md 발견 및 주입, 자동 컨텍스트 압축(auto-compact), MEMORY.md를 통한 세션 간 영속적 지식, 세션 재개 및 히스토리 기능을 제공합니다.

**거버넌스(Permissions)** 는 다단계 권한 모드(기본/자동/계획)와 경로 수준 및 명령어 거부 규칙, PreToolUse/PostToolUse 훅, 대화형 승인 다이얼로그를 통해 에이전트가 의도하지 않은 동작을 하지 않도록 안전하게 제어합니다.

**스웜 조율(Swarm Coordination)** 은 서브에이전트 생성 및 위임, 팀 레지스트리와 태스크 관리, 백그라운드 태스크 라이프사이클을 관리하여 복잡한 멀티 에이전트 워크플로우를 지원합니다.

## 다양한 LLM 프로바이더 지원

OpenHarness는 특정 LLM에 종속되지 않습니다. Anthropic(기본), Moonshot/Kimi, Google Vertex, AWS Bedrock, 그리고 Anthropic 호환 API를 제공하는 어떤 프로바이더든 `ANTHROPIC_BASE_URL` 환경 변수 설정만으로 즉시 연결할 수 있습니다. 이를 통해 OpenAI, Gemini, DeepSeek, Ollama 등 200개 이상의 모델을 OpenHarness와 함께 사용할 수 있습니다.

## 스킬 및 플러그인 시스템

OpenHarness는 `anthropics/skills` 와 호환되는 스킬 시스템을 지원합니다. commit, review, debug, plan, test, simplify, pdf, xlsx 등 40개 이상의 스킬이 필요 시에만 온디맨드로 로딩됩니다. 또한, claude-code 플러그인과도 호환되어 commit-commands, security-guidance, hookify, feature-dev, code-review, pr-review-toolkit 등 12개의 공식 플러그인이 테스트되어 있습니다.

## Claude Code와의 비교

| 항목 | Claude Code | OpenHarness |
| --- | --- | --- |
| 코드 줄 수 | 512,664줄 | 11,733줄 (44배 경량) |
| 파일 수 | 1,884개 | 163개 |
| 언어 | TypeScript | Python |
| 도구 수 | ~44개 | 43개 (98%) |
| 명령어 수 | ~88개 | 54개 (61%) |
| Skills 호환 | ![:white_check_mark:](https://discuss.pytorch.kr/images/emoji/fluentui/white_check_mark.png?v=15 ":white_check_mark:") | ![:white_check_mark:](https://discuss.pytorch.kr/images/emoji/fluentui/white_check_mark.png?v=15 ":white_check_mark:") anthropics/skills |
| Plugin 호환 | ![:white_check_mark:](https://discuss.pytorch.kr/images/emoji/fluentui/white_check_mark.png?v=15 ":white_check_mark:") | ![:white_check_mark:](https://discuss.pytorch.kr/images/emoji/fluentui/white_check_mark.png?v=15 ":white_check_mark:") claude-code/plugins |
| 테스트 | — | 114 유닛 + 6 E2E |

## OpenHarnerss 프로젝트 구성

```graphql
Gemini said
openharness/
  engine/          # 🧠 에이전트 루프 — 질의 → 스트림 → 도구 호출 → 루프
  tools/           # 🔧 43가지 도구 — 파일 입출력, 셸, 검색, 웹, MCP
  skills/          # 📚 지식 — 필요 시 스킬 로딩 (.md 파일)
  plugins/         # 🔌 확장 기능 — 명령, 후크, 에이전트, MCP 서버
  permissions/     # 🛡️ 안전 — 다단계 모드, 경로 규칙, 명령 거부
  hooks/           # ⚡ 생명 주기 — PreToolUse/PostToolUse 이벤트 후크
  commands/        # 💬 54가지 명령 — /help, /commit, /plan, /resume, ...
  mcp/             # 🌐 MCP — Model Context Protocol 클라이언트
  memory/          # 🧠 메모리 — 세션 간 지속되는 지식
  tasks/           # 📋 작업 — 백그라운드 작업 관리
  coordinator/     # 🤝 멀티 에이전트 — 하위 에이전트 생성, 팀 조정
  prompts/         # 📝 컨텍스트 — 시스템 프롬프트 조립, CLAUDE.md, 스킬
  config/          # ⚙️ 설정 — 다층 구성, 마이그레이션
  ui/              # 🖥️ React TUI — 백엔드 프로토콜 + 프론트엔드
```

## OpenHarness 설치 및 빠른 시작

[![image](https://discuss.pytorch.kr/uploads/default/original/3X/1/0/1085c58059d28e6c42aa0d122a5d85a5ba85ccbd.png)

image820×380 15.4 KB

](https://discuss.pytorch.kr/uploads/default/original/3X/1/0/1085c58059d28e6c42aa0d122a5d85a5ba85ccbd.png "image")

OpenHarness는 Python 3.10 이상과 UV 패키지 매니저가 필요합니다. 설치는 저장소를 복제한 뒤, `uv sync` 명령어로 간단히 가능합니다:

```bash
# 저장소 복제(clone) 및 설치
git clone https://github.com/HKUDS/OpenHarness.git
cd OpenHarness
uv sync --extra dev

# 예시: Kimi(Moonshot AI)를 백엔드로 사용
export ANTHROPIC_BASE_URL=https://api.moonshot.cn/anthropic
export ANTHROPIC_API_KEY=your_kimi_api_key
export ANTHROPIC_MODEL=kimi-k2.5

# 실행 (venv 활성화 시)
oh

# venv 없이 실행
uv run oh
```

다음과 같이 비대화형(non-interactive) 모드로 단일 프롬프트를 실행하거나 JSON 형식으로 출력을 받을 수도 있습니다:

```bash
# 단일 프롬프트 실행
oh -p "Inspect this repository and list the top 3 refactors"

# JSON 출력
oh -p "List all functions in main.py" --output-format json

# 스트리밍 JSON 이벤트
oh -p "Fix the bug" --output-format stream-json
```

## 라이선스

OpenHarness 프로젝트는 [MIT 라이선스](https://github.com/HKUDS/OpenHarness/blob/main/LICENSE?utm_source=pytorchkr&ref=pytorchkr)로 공개되어 있어 개인 및 상업적 목적으로 자유롭게 사용, 수정, 배포할 수 있습니다.

## ![:github:](https://discuss.pytorch.kr/uploads/default/original/2X/7/70a6220c603eed42089b4f67366225849e119e20.svg?v=15 ":github:") OpenHarness 프로젝트 GitHub 저장소

[github.com](https://github.com/HKUDS/OpenHarness?utm_source=pytorchkr&ref=pytorchkr)

![](https://opengraph.githubassets.com/7910e4d42d10657f416cc21a2b8b2ad1/HKUDS/OpenHarness)

### [GitHub - HKUDS/OpenHarness: "OpenHarness: Open Agent Harness"](https://github.com/HKUDS/OpenHarness?utm_source=pytorchkr&ref=pytorchkr)

"OpenHarness: Open Agent Harness"

## 더 읽어보기

- [OpenSpace: AI 에이전트가 스스로 학습하고 진화하는 자율 스킬 진화 프레임워크 (feat. HKUDS)](https://discuss.pytorch.kr/t/openspace-ai-feat-hkuds/9476)
- [CatchMe: 나의 모든 디지털 활동을 기억하고 AI 에이전트를 개인화하는 오픈소스 메모리 시스템 (feat. HKUDS)](https://discuss.pytorch.kr/t/catchme-ai-feat-hkuds/9515)
- [https://discuss.pytorch.kr/t/open-multi-agent-typescript-ai/9516](https://discuss.pytorch.kr/t/open-multi-agent-typescript-ai/9516)
- [https://discuss.pytorch.kr/t/emdash-ai-git/9495](https://discuss.pytorch.kr/t/emdash-ai-git/9495)
- [ARIS: 아이디어에서 출판 준비 논문까지, 크로스 모델 협업 자율 ML 연구 파이프라인](https://discuss.pytorch.kr/t/aris-ml/9419)
- [Paperclip: AI 에이전트들로 100% 무인 기업을 운영하는 것을 목표로 하는 오픈소스 '컴퍼니 OS' 프로젝트](https://discuss.pytorch.kr/t/paperclip-ai-100-os/9167)

  
  

---

*이 글은 GPT 모델로 정리한 글을 바탕으로 한 것으로, 원문의 내용 또는 의도와 다르게 정리된 내용이 있을 수 있습니다. 관심있는 내용이시라면 원문도 함께 참고해주세요! 읽으시면서 어색하거나 잘못된 내용을 발견하시면 덧글로 알려주시기를 부탁드립니다.* ![:hugs:](https://discuss.pytorch.kr/images/emoji/fluentui/hugs.png?v=15 ":hugs:")

[![:pytorch:](https://discuss.pytorch.kr/uploads/default/original/2X/f/fa98c2196c22febe7475e503792febf39ba7a0de.svg?v=15 ":pytorch:")파이토치 한국 사용자 모임![:south_korea:](https://discuss.pytorch.kr/images/emoji/fluentui/south_korea.png?v=15 ":south_korea:")](https://pytorch.kr/)이 정리한 이 글이 유용하셨나요? [회원으로 가입](https://discuss.pytorch.kr/signup)하시면 주요 글들을 이메일![:love_letter:](https://discuss.pytorch.kr/images/emoji/fluentui/love_letter.png?v=15 ":love_letter:")로 보내드립니다! (기본은 Weekly지만 [Daily로 변경도 가능](https://discuss.pytorch.kr/my/preferences/emails)합니다.)

![:wrapped_gift:](https://discuss.pytorch.kr/images/emoji/fluentui/wrapped_gift.png?v=15 ":wrapped_gift:") 아래![:down_right_arrow:](https://discuss.pytorch.kr/images/emoji/fluentui/down_right_arrow.png?v=15 ":down_right_arrow:")쪽에 좋아요![:+1:](https://discuss.pytorch.kr/images/emoji/fluentui/+1.png?v=15 ":+1:")를 눌러주시면 새로운 소식들을 정리하고 공유하는데 힘이 됩니다~ ![:star_struck:](https://discuss.pytorch.kr/images/emoji/fluentui/star_struck.png?v=15 ":star_struck:")