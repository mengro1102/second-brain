"""
위키 지식 그래프 MCP 서버.
graph.json 기반으로 노드 프로파일, 경로 탐색, 인과 체인 해석, 위키 건강 진단을 제공한다.
모든 도구는 raw 데이터가 아닌 "의미 해석 + 행동 제안"을 포함한 출력을 반환한다.

실행: uv run --with fastmcp python3 .scripts/wiki_graph_mcp.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import deque
from pathlib import Path

from fastmcp import FastMCP

# ── 경로 설정 ──────────────────────────────────────────────
WORKSPACE = Path(__file__).resolve().parent.parent
GRAPH_PATH = WORKSPACE / "00_graphify-out" / "graph.json"
BUILD_SCRIPT = WORKSPACE / ".scripts" / "build_graph.py"

mcp = FastMCP(name="Wiki Knowledge Graph")


# ── 유틸리티 ───────────────────────────────────────────────

def _load_graph():
    """graph.json을 로드하여 (nodes_dict, links, label_to_id) 반환."""
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(
            "graph.json이 없습니다. rebuild_graph 도구를 먼저 실행하세요."
        )
    data = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in data["nodes"]}
    label_to_id: dict[str, str] = {}
    for n in data["nodes"]:
        label_to_id[n["label"].lower()] = n["id"]
        label_to_id[n["id"].lower()] = n["id"]
    return nodes, data["links"], label_to_id


def _resolve(query: str, label_to_id: dict[str, str]) -> str | None:
    q = query.lower().strip()
    if q in label_to_id:
        return label_to_id[q]
    for label, nid in label_to_id.items():
        if q in label:
            return nid
    return None


def _degree_map(links: list[dict]) -> dict[str, int]:
    """각 노드의 연결 수(degree)를 계산."""
    deg: dict[str, int] = {}
    for l in links:
        deg[l["source"]] = deg.get(l["source"], 0) + 1
        deg[l["target"]] = deg.get(l["target"], 0) + 1
    return deg


def _percentile_rank(value: int, all_values: list[int]) -> int:
    """value가 all_values 중 상위 몇 %인지 반환."""
    if not all_values:
        return 0
    count_below = sum(1 for v in all_values if v < value)
    return round(count_below / len(all_values) * 100)


# ── 도구: explain_node ─────────────────────────────────────

@mcp.tool
def explain_node(query: str) -> str:
    """노드의 상세 정보와 연결 관계를 조회한다.
    노드 이름이나 ID의 일부를 입력하면 해당 노드의 카테고리, 파일 경로,
    태그, 나가는/들어오는 엣지 목록을 반환한다.

    Args:
        query: 검색할 노드 이름 또는 ID (부분 매칭 지원)
    """
    nodes, links, label_to_id = _load_graph()
    nid = _resolve(query, label_to_id)
    if not nid:
        return f"노드를 찾을 수 없습니다: {query}"

    node = nodes[nid]
    degree = _degree_map(links)
    all_degrees = list(degree.values())
    node_degree = degree.get(nid, 0)
    pct = _percentile_rank(node_degree, all_degrees)

    outgoing = [l for l in links if l["source"] == nid]
    incoming = [l for l in links if l["target"] == nid]
    causal_out = [l for l in outgoing if l["type"] == "causal"]
    causal_in = [l for l in incoming if l["type"] == "causal"]

    # 카테고리별 연결 분포
    cat_dist: dict[str, int] = {}
    for l in outgoing + incoming:
        other = l["target"] if l["source"] == nid else l["source"]
        other_node = nodes.get(other, {})
        cat = other_node.get("category", "unresolved")
        cat_dist[cat] = cat_dist.get(cat, 0) + 1
    top_cat = sorted(cat_dist.items(), key=lambda x: x[1], reverse=True)

    # 역할 해석
    if node_degree == 0:
        role = "고립 노드 — 다른 개념과 연결이 없음"
    elif pct >= 80:
        role = f"핵심 허브 — 연결도 상위 {100 - pct}%, 위키의 중심 개념"
    elif pct >= 50:
        role = f"중간 연결자 — 연결도 상위 {100 - pct}%, 여러 개념을 잇는 역할"
    else:
        role = f"말단 노드 — 연결도 하위 {100 - pct}%, 특정 소스에 종속"

    # 인과 역할 해석
    if causal_out and causal_in:
        causal_role = f"매개자 — 원인 역할 {len(causal_out)}개, 결과 역할 {len(causal_in)}개"
    elif causal_out:
        causal_role = f"기반 기술 — 다른 {len(causal_out)}개 개념을 가능하게 함"
    elif causal_in:
        causal_role = f"파생 결과 — {len(causal_in)}개 개념으로부터 영향받음"
    else:
        causal_role = "인과 관계 미등록 — 인과 callout 추가 권장"

    lines = [
        f"## {node['label']}",
        f"카테고리: {node.get('category', '')} | 태그: {node.get('tags', '')}",
        f"파일: {node.get('file', 'N/A')}",
        "",
        "### 위치 분석",
        f"- 연결도: {node_degree} (상위 {100 - pct}%) → {role}",
        f"- 인과 역할: {causal_role}",
    ]

    if top_cat:
        cat_str = ", ".join(f"{c}({n})" for c, n in top_cat[:3])
        lines.append(f"- 최다 연결 카테고리: {cat_str}")

    # 인과 요약
    if causal_out or causal_in:
        lines.append("\n### 인과 요약")
        for l in causal_out:
            tgt = nodes.get(l["target"], {}).get("label", l["target"])
            lines.append(f"- {node['label']} →({l.get('relation', '?')})→ {tgt}")
        for l in causal_in:
            src = nodes.get(l["source"], {}).get("label", l["source"])
            lines.append(f"- {src} →({l.get('relation', '?')})→ {node['label']}")

    # 연결 맥락 (카테고리별 그룹)
    lines.append("\n### 연결 맥락")
    for cat_name in ["concepts", "sources", "entities", "syntheses", "unresolved"]:
        cat_links = []
        for l in outgoing + incoming:
            other = l["target"] if l["source"] == nid else l["source"]
            other_node = nodes.get(other, {})
            if other_node.get("category", "unresolved") == cat_name:
                cat_links.append(other_node.get("label", other))
        if cat_links:
            unique = list(dict.fromkeys(cat_links))  # 중복 제거, 순서 유지
            lines.append(f"- {cat_name}: {', '.join(unique[:8])}")

    # 성장 제안
    unresolved_neighbors = [
        nodes.get(l["target"], {}).get("label", l["target"])
        for l in outgoing
        if nodes.get(l["target"], {}).get("category") == "unresolved"
    ]
    suggestions = []
    if unresolved_neighbors:
        suggestions.append(
            f"미해결 연결 {len(unresolved_neighbors)}개 → 페이지 생성 권장: "
            + ", ".join(unresolved_neighbors[:5])
        )
    if not causal_out and not causal_in:
        suggestions.append("인과 관계 callout이 없음 → 관련 소스에서 인과 관계 추출 권장")
    if node_degree <= 2:
        suggestions.append("연결이 적음 → 관련 concepts/entities 페이지와 위키링크 추가 권장")

    if suggestions:
        lines.append("\n### 성장 제안")
        for s in suggestions:
            lines.append(f"- {s}")

    return "\n".join(lines)


# ── 도구: find_path ────────────────────────────────────────

@mcp.tool
def find_path(start: str, end: str) -> str:
    """두 노드 사이의 최단 경로를 BFS로 탐색한다.
    위키링크와 인과 관계를 모두 양방향으로 탐색하여 연결 경로를 찾는다.

    Args:
        start: 시작 노드 이름 또는 ID
        end: 도착 노드 이름 또는 ID
    """
    nodes, links, label_to_id = _load_graph()
    start_id = _resolve(start, label_to_id)
    end_id = _resolve(end, label_to_id)

    if not start_id:
        return f"시작 노드를 찾을 수 없습니다: {start}"
    if not end_id:
        return f"도착 노드를 찾을 수 없습니다: {end}"
    if start_id == end_id:
        return f"동일한 노드입니다: {nodes.get(start_id, {}).get('label', start_id)}"

    # 인접 리스트 (양방향)
    adj: dict[str, list[dict]] = {}
    for l in links:
        adj.setdefault(l["source"], []).append(l)
        adj.setdefault(l["target"], []).append(
            {**l, "source": l["target"], "target": l["source"], "_reverse": True}
        )

    visited = {start_id}
    queue = deque([(start_id, [(start_id, None)])])

    while queue:
        current, path = queue.popleft()
        if current == end_id:
            start_label = nodes.get(start_id, {}).get("label", start_id)
            end_label = nodes.get(end_id, {}).get("label", end_id)
            hops = len(path) - 1

            # 연결 강도 해석
            if hops == 1:
                strength = "직접 연결 — 강한 관계"
            elif hops == 2:
                strength = "1단계 매개 — 중간 개념을 통해 연결"
            elif hops <= 4:
                strength = f"{hops - 1}단계 매개 — 간접적 관계"
            else:
                strength = f"{hops - 1}단계 매개 — 매우 먼 관계"

            # 경로에 인과 관계가 포함되어 있는지
            has_causal = any(
                e and e.get("type") == "causal" for _, e in path
            )

            lines = [
                f"## {start_label} ↔ {end_label}",
                f"거리: {hops}홉 | {strength}",
            ]
            if has_causal:
                lines.append("⚡ 경로에 인과 관계 포함 — 논리적 연결이 존재")

            lines.append("\n### 경로")
            for nid, edge in path:
                n = nodes.get(nid, {})
                label = n.get("label", nid)
                cat = n.get("category", "")
                if edge:
                    rel = edge.get("relation", edge["type"])
                    direction = "←" if edge.get("_reverse") else "→"
                    edge_type = "🔗" if edge["type"] == "wikilink" else "⚡"
                    lines.append(f"  {direction} {edge_type} [{rel}] {label} ({cat})")
                else:
                    lines.append(f"  ● {label} ({cat})")

            # 경유 노드 해석
            intermediates = [
                nodes.get(nid, {}).get("label", nid)
                for nid, _ in path[1:-1]
            ]
            if intermediates:
                lines.append(f"\n### 매개 노드 해석")
                lines.append(
                    f"- {start_label}과 {end_label}은 "
                    f"{'→'.join(intermediates)}을(를) 통해 연결됨"
                )

            return "\n".join(lines)

        for link in adj.get(current, []):
            neighbor = link["target"]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [(neighbor, link)]))

    start_label = nodes.get(start_id, {}).get("label", start_id)
    end_label = nodes.get(end_id, {}).get("label", end_id)
    return (
        f"## {start_label} ↔ {end_label}\n"
        f"경로 없음 — 두 개념이 현재 위키에서 연결되지 않음\n"
        f"→ 두 개념을 잇는 중간 개념 페이지를 생성하면 연결 가능"
    )


# ── 도구: causal_chain ─────────────────────────────────────

@mcp.tool
def causal_chain(query: str) -> str:
    """특정 노드와 관련된 모든 인과 관계를 조회한다.
    해당 노드가 원인이거나 결과인 모든 인과 관계 callout을 반환한다.

    Args:
        query: 검색할 노드 이름 또는 ID
    """
    nodes, links, label_to_id = _load_graph()
    nid = _resolve(query, label_to_id)
    if not nid:
        return f"노드를 찾을 수 없습니다: {query}"

    causal_links = [l for l in links if l["type"] == "causal"]
    upstream = [l for l in causal_links if l["target"] == nid]  # 이 노드에 영향을 주는 것
    downstream = [l for l in causal_links if l["source"] == nid]  # 이 노드가 영향을 주는 것

    node = nodes.get(nid, {})
    label = node.get("label", nid)

    if not upstream and not downstream:
        return (
            f"## {label}의 인과 관계\n\n"
            f"인과 관계가 등록되지 않음.\n"
            f"→ 관련 소스 페이지에서 이 개념의 원인/결과를 식별하여 "
            f"`> [!causal]` callout 추가 권장"
        )

    lines = [f"## {label}의 인과 네트워크\n"]

    # 상류 (이 노드를 가능하게 한 것)
    if upstream:
        lines.append("### 상류 (이 개념을 가능하게 한 것)")
        for l in upstream:
            src = nodes.get(l["source"], {}).get("label", l["source"])
            rel = l.get("relation", "?")
            lines.append(f"- {src} →({rel})→ {label}")
            lines.append(f"  해석: {src}이(가) {label}의 {_relation_meaning(rel)}")
    else:
        lines.append("### 상류")
        lines.append(f"- 없음 → {label}의 기반이 되는 개념을 추적하면 인과 체인 확장 가능")

    lines.append("")

    # 하류 (이 노드가 영향을 준 것)
    if downstream:
        lines.append("### 하류 (이 개념이 영향을 준 것)")
        for l in downstream:
            tgt = nodes.get(l["target"], {}).get("label", l["target"])
            rel = l.get("relation", "?")
            lines.append(f"- {label} →({rel})→ {tgt}")
            lines.append(f"  해석: {label}이(가) {tgt}의 {_relation_meaning(rel)}")
    else:
        lines.append("### 하류")
        lines.append(f"- 없음 → {label}이 영향을 준 후속 개념/기술을 추적하면 확장 가능")

    # 인과 깊이 분석
    total = len(upstream) + len(downstream)
    lines.append(f"\n### 인과 깊이: {total}단계")
    if total <= 2:
        lines.append("- 얕은 인과 체인 → 상류/하류 확장으로 더 깊은 논리 구조 구축 가능")
    elif total <= 5:
        lines.append("- 적절한 인과 체인 → 핵심 논리 흐름이 형성됨")
    else:
        lines.append("- 풍부한 인과 체인 → 이 개념 중심의 논리 추론이 가능")

    return "\n".join(lines)


def _relation_meaning(relation: str) -> str:
    """인과 관계 동사를 자연어 해석으로 변환."""
    meanings = {
        "가능하게 함": "존재/작동을 가능하게 하는 핵심 기반",
        "성능 향상": "성능을 높이는 보강 요인",
        "성능 저하": "성능을 낮추는 제약 요인",
        "기반이 됨": "이론적/기술적 토대",
        "발전시킴": "확장·발전의 원동력",
        "대체함": "기존 방식을 대체하는 새로운 접근",
        "포함함": "상위 개념으로서 포함하는 관계",
        "적용됨": "실제 적용되는 대상",
    }
    return meanings.get(relation, f"'{relation}' 관계")


# ── 도구: graph_summary ────────────────────────────────────

@mcp.tool
def graph_summary() -> str:
    """현재 위키 그래프의 전체 요약 통계를 반환한다.
    노드 수, 엣지 수, 카테고리별 분포, 허브 노드, 미해결 노드 목록을 포함한다.
    """
    nodes, links, label_to_id = _load_graph()
    degree = _degree_map(links)

    # 카테고리별 분포
    categories: dict[str, int] = {}
    for n in nodes.values():
        cat = n.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    # 허브 노드 (상위 5)
    hubs = sorted(degree.items(), key=lambda x: x[1], reverse=True)[:5]

    # 미해결 노드 (연결 수 기준 정렬)
    unresolved = [
        (nid, degree.get(nid, 0))
        for nid, n in nodes.items()
        if n.get("category") == "unresolved"
    ]
    unresolved.sort(key=lambda x: x[1], reverse=True)

    causal_count = sum(1 for l in links if l["type"] == "causal")
    wikilink_count = sum(1 for l in links if l["type"] == "wikilink")
    total_nodes = len(nodes)
    total_edges = len(links)
    resolved_nodes = total_nodes - len(unresolved)

    # 밀도 계산
    density = round(total_edges / total_nodes, 2) if total_nodes > 0 else 0
    causal_ratio = round(causal_count / total_edges * 100, 1) if total_edges > 0 else 0
    unresolved_ratio = round(len(unresolved) / total_nodes * 100, 1) if total_nodes > 0 else 0

    # 규모 판정
    if resolved_nodes < 20:
        scale = "초기 단계 — 핵심 개념 축적 중"
    elif resolved_nodes < 50:
        scale = "성장 단계 — 개념 간 연결이 형성되는 중"
    elif resolved_nodes < 100:
        scale = "확장 단계 — 지식 네트워크가 밀도를 갖추는 중"
    else:
        scale = "성숙 단계 — 풍부한 지식 그래프"

    lines = [
        "## 위키 건강 리포트\n",
        f"### 규모: {scale}",
        f"- 실제 페이지: {resolved_nodes}개 (concepts {categories.get('concepts', 0)}, "
        f"sources {categories.get('sources', 0)}, "
        f"entities {categories.get('entities', 0)}, "
        f"syntheses {categories.get('syntheses', 0)})",
        f"- 총 엣지: {total_edges} (밀도: {density} 엣지/노드)",
    ]

    # 밀도 해석
    if density >= 4.0:
        lines.append(f"  → 높은 밀도 — 개념 간 연결이 풍부함")
    elif density >= 2.5:
        lines.append(f"  → 양호한 밀도 — 적절히 연결됨")
    else:
        lines.append(f"  → 낮은 밀도 — 위키링크 추가로 연결 강화 필요")

    # 강점
    lines.append("\n### 강점")
    if hubs:
        hub_strs = [
            f"{nodes[nid].get('label', nid)}({deg}연결)"
            for nid, deg in hubs[:3]
        ]
        lines.append(f"- 허브 노드: {', '.join(hub_strs)} → 이 개념들이 위키의 중심축")
    if causal_count > 0:
        lines.append(
            f"- 인과 관계 {causal_count}개 존재 → '왜'를 추적하는 구조가 형성됨"
        )

    # 약점
    lines.append("\n### 약점")
    weaknesses = []
    if unresolved_ratio > 25:
        weaknesses.append(
            f"미해결 노드 {len(unresolved)}개 ({unresolved_ratio}%) → "
            f"참조는 되지만 페이지가 없는 개념이 많음"
        )
    elif unresolved:
        weaknesses.append(f"미해결 노드 {len(unresolved)}개 ({unresolved_ratio}%)")

    if causal_ratio < 5:
        weaknesses.append(
            f"인과 관계 비율 {causal_ratio}% → 위키링크 대비 인과 관계가 부족. "
            f"인제스트 시 `> [!causal]` callout 적극 삽입 권장"
        )
    if categories.get("syntheses", 0) == 0:
        weaknesses.append(
            "syntheses/ 페이지 0개 → 질의 결과 중 가치 있는 답변을 syntheses/에 저장 시작 권장"
        )
    if not weaknesses:
        weaknesses.append("특별한 약점 없음")
    for w in weaknesses:
        lines.append(f"- {w}")

    # 다음 행동 제안
    lines.append("\n### 다음 행동 제안")
    actions = []
    if unresolved:
        top_unresolved = [
            f"{nid}({deg}연결)" for nid, deg in unresolved[:5]
        ]
        actions.append(f"미해결 노드 페이지 생성 (우선순위): {', '.join(top_unresolved)}")
    if causal_ratio < 5:
        actions.append("기존 concepts 페이지에 인과 관계 callout 보강")
    if categories.get("syntheses", 0) == 0:
        actions.append("다음 질의 시 가치 있는 답변을 syntheses/에 저장")
    if not actions:
        actions.append("현재 상태 양호 — 새 소스 인제스트로 위키 확장 계속")
    for i, a in enumerate(actions, 1):
        lines.append(f"{i}. {a}")

    return "\n".join(lines)


# ── 도구: rebuild_graph ────────────────────────────────────

@mcp.tool
def rebuild_graph() -> str:
    """02_wiki/ 마크다운을 파싱하여 graph.json을 재빌드한다.
    위키 페이지 변경 후 그래프를 최신 상태로 갱신할 때 사용한다.
    """
    try:
        result = subprocess.run(
            [sys.executable, str(BUILD_SCRIPT)],
            capture_output=True,
            text=True,
            cwd=str(WORKSPACE),
            timeout=30,
        )
        output = result.stdout.strip()
        if result.returncode != 0:
            return f"재빌드 실패:\n{result.stderr}"
        return f"그래프 재빌드 완료:\n{output}"
    except Exception as e:
        return f"재빌드 중 오류: {e}"


# ── 리소스 정의 ────────────────────────────────────────────

@mcp.resource("wiki://graph/metadata")
def graph_metadata() -> dict:
    """graph.json의 메타데이터를 반환한다."""
    if not GRAPH_PATH.exists():
        return {"error": "graph.json not found"}
    data = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    return data.get("metadata", {})


# ── 엔트리포인트 ───────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
