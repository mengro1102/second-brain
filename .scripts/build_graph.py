"""
wiki/ 폴더의 마크다운 파일에서 위키링크와 인과 관계를 파싱하여
Graphify 호환 graph.json + GRAPH_REPORT.md를 생성하는 스크립트.

사용법: python3 .scripts/build_graph.py
출력: .graphify-out/graph.json, .graphify-out/GRAPH_REPORT.md
"""

import json
import re
import os
from pathlib import Path
from datetime import date

WIKI_DIR = Path("02_wiki")
OUTPUT_DIR = Path("00_graphify-out")

# 위키링크 패턴: [[페이지명]] 또는 [[페이지명|표시텍스트]]
WIKILINK_RE = re.compile(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]')

# 인과 관계 패턴: [[A]] →(관계)→ [[B]]
CAUSAL_RE = re.compile(
    r'\[\[([^\]]+)\]\]\s*→\(([^)]+)\)→\s*\[\[([^\]]+)\]\]'
)

# YAML frontmatter 파싱 (간단 버전)
FRONTMATTER_RE = re.compile(r'^---\s*\n(.*?)\n---', re.DOTALL)
YAML_FIELD_RE = re.compile(r'^(\w+):\s*(.+)$', re.MULTILINE)


def parse_frontmatter(content: str) -> dict:
    match = FRONTMATTER_RE.match(content)
    if not match:
        return {}
    fm = {}
    for m in YAML_FIELD_RE.finditer(match.group(1)):
        key, val = m.group(1), m.group(2).strip().strip('"').strip("'")
        fm[key] = val
    return fm


def extract_nodes_and_edges(wiki_dir: Path):
    nodes = {}
    edges = []

    md_files = list(wiki_dir.rglob("*.md"))
    md_files = [f for f in md_files if f.name != ".steering.md"]

    # Pass 1: 노드 수집
    file_contents = {}
    for md_file in md_files:
        content = md_file.read_text(encoding="utf-8")
        fm = parse_frontmatter(content)
        rel_path = md_file.relative_to(wiki_dir)
        slug = md_file.stem
        parts = rel_path.parts
        category = parts[0] if len(parts) > 1 else "root"
        title = fm.get("title", slug)
        tags = fm.get("tags", "")
        nodes[slug] = {
            "label": title,
            "file": str(rel_path),
            "category": category,
            "tags": tags,
        }
        file_contents[slug] = content

    # 별칭 매핑 생성
    alias_map = build_alias_map(nodes)

    # Pass 2: 엣지 추출 (별칭 매핑 적용)
    for slug, content in file_contents.items():
        for link_match in WIKILINK_RE.finditer(content):
            raw_target = link_match.group(1).strip()
            target_slug = resolve_target(raw_target, alias_map)
            if target_slug != slug:
                edges.append({
                    "source": slug,
                    "target": target_slug,
                    "type": "wikilink",
                    "key": f"wikilink_{slug}_{target_slug}",
                    "weight": 1.0,
                })

        for causal_match in CAUSAL_RE.finditer(content):
            raw_cause = causal_match.group(1).strip()
            relation = causal_match.group(2).strip()
            raw_effect = causal_match.group(3).strip()
            cause = resolve_target(raw_cause, alias_map)
            effect = resolve_target(raw_effect, alias_map)
            edges.append({
                "source": cause,
                "target": effect,
                "type": "causal",
                "relation": relation,
                "key": f"causal_{cause}_{effect}",
                "weight": 2.0,
            })

    return nodes, edges


def normalize_to_slug(text: str) -> str:
    """위키링크 텍스트를 파일명 슬러그로 정규화"""
    s = text.strip()
    # 소문자 변환
    s = s.lower()
    # 공백, 언더스코어를 하이픈으로
    s = re.sub(r'[\s_]+', '-', s)
    # 연속 하이픈 정리
    s = re.sub(r'-+', '-', s).strip('-')
    return s


def build_alias_map(nodes: dict) -> dict:
    """노드 ID와 label에서 별칭 매핑 테이블 생성"""
    alias = {}
    for nid, n in nodes.items():
        # 정확한 ID
        alias[nid] = nid
        alias[nid.lower()] = nid
        # label에서 파생
        label = n.get("label", "")
        if label:
            alias[label.lower()] = nid
            alias[normalize_to_slug(label)] = nid
            # 괄호 안 영어명 추출: "메타러닝 (Meta-Learning)" → "meta-learning"
            paren = re.search(r'\(([^)]+)\)', label)
            if paren:
                alias[paren.group(1).lower()] = nid
                alias[normalize_to_slug(paren.group(1))] = nid
                # 괄호 앞 한국어명
                korean = label[:label.index('(')].strip()
                alias[korean.lower()] = nid
    return alias


def resolve_target(target_text: str, alias_map: dict) -> str:
    """위키링크 대상을 기존 노드 ID로 해석. 못 찾으면 원본 반환."""
    t = target_text.strip()
    # 1. 정확 매칭
    if t in alias_map:
        return alias_map[t]
    # 2. 소문자 매칭
    if t.lower() in alias_map:
        return alias_map[t.lower()]
    # 3. 슬러그 변환 매칭
    slug = normalize_to_slug(t)
    if slug in alias_map:
        return alias_map[slug]
    return t


def deduplicate_edges(edges):
    seen = set()
    unique = []
    for e in edges:
        key = (e["source"], e["target"], e["type"])
        if key not in seen:
            seen.add(key)
            unique.append(e)
    return unique


def build_graph():
    nodes, edges = extract_nodes_and_edges(WIKI_DIR)
    edges = deduplicate_edges(edges)

    # 엣지에서 참조되지만 노드로 등록되지 않은 대상 추가
    all_node_ids = set(nodes.keys())
    for e in edges:
        for key in ["source", "target"]:
            if e[key] not in all_node_ids:
                nodes[e[key]] = {
                    "label": e[key],
                    "file": None,
                    "category": "unresolved",
                    "tags": "",
                }
                all_node_ids.add(e[key])

    graph = {
        "nodes": [{"id": k, **v} for k, v in nodes.items()],
        "links": edges,
        "metadata": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "causal_edges": len([e for e in edges if e["type"] == "causal"]),
            "wikilink_edges": len([e for e in edges if e["type"] == "wikilink"]),
        },
    }

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "graph.json"
    output_path.write_text(
        json.dumps(graph, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Graph built: {output_path}")
    print(f"  Nodes: {graph['metadata']['total_nodes']}")
    print(f"  Edges: {graph['metadata']['total_edges']}")
    print(f"    Wikilinks: {graph['metadata']['wikilink_edges']}")
    print(f"    Causal: {graph['metadata']['causal_edges']}")

    # GRAPH_REPORT.md 생성
    generate_report(nodes, edges, graph["metadata"])

    # graph.html 생성
    generate_html(nodes, edges)


def generate_report(nodes, edges, metadata):
    """그래프 분석 리포트 생성"""
    today = date.today().isoformat()

    # 허브 노드 (연결 수 상위 5개)
    degree = {}
    for e in edges:
        degree[e["source"]] = degree.get(e["source"], 0) + 1
        degree[e["target"]] = degree.get(e["target"], 0) + 1
    hub_nodes = sorted(degree.items(), key=lambda x: x[1], reverse=True)[:5]

    # 고아 노드 (연결 0개)
    connected = set()
    for e in edges:
        connected.add(e["source"])
        connected.add(e["target"])
    orphans = [nid for nid in nodes if nid not in connected]

    # 미해결 노드 (위키링크는 있지만 페이지가 없는 노드)
    unresolved = [nid for nid, n in nodes.items() if n.get("category") == "unresolved"]

    # 인과 관계 목록
    causal_edges = [e for e in edges if e["type"] == "causal"]

    # 카테고리별 노드 수
    categories = {}
    for n in nodes.values():
        cat = n.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    # 리포트 작성
    lines = [
        f"---",
        f"title: Graph Report",
        f"generated: {today}",
        f"---",
        f"",
        f"# Graph Report",
        f"",
        f"생성일: {today}",
        f"",
        f"## 요약",
        f"",
        f"| 지표 | 값 |",
        f"|------|-----|",
        f"| 총 노드 | {metadata['total_nodes']} |",
        f"| 총 엣지 | {metadata['total_edges']} |",
        f"| 위키링크 | {metadata['wikilink_edges']} |",
        f"| 인과 관계 | {metadata['causal_edges']} |",
        f"| 고아 노드 | {len(orphans)} |",
        f"| 미해결 노드 | {len(unresolved)} |",
        f"",
        f"## 카테고리별 분포",
        f"",
        f"| 카테고리 | 노드 수 |",
        f"|---------|--------|",
    ]
    for cat, count in sorted(categories.items()):
        lines.append(f"| {cat} | {count} |")

    lines += [
        f"",
        f"## 허브 노드 (연결 수 상위 5)",
        f"",
    ]
    for nid, deg in hub_nodes:
        label = nodes[nid].get("label", nid)
        lines.append(f"- **{label}** — {deg}개 연결")

    if causal_edges:
        lines += [
            f"",
            f"## 인과 관계 체인",
            f"",
        ]
        for e in causal_edges:
            src = nodes.get(e["source"], {}).get("label", e["source"])
            tgt = nodes.get(e["target"], {}).get("label", e["target"])
            rel = e.get("relation", "?")
            lines.append(f"- {src} →({rel})→ {tgt}")

    if unresolved:
        lines += [
            f"",
            f"## 미해결 노드 (페이지 없음)",
            f"",
            f"다음 노드는 위키링크로 참조되지만 자체 페이지가 없습니다:",
            f"",
        ]
        for nid in unresolved:
            lines.append(f"- `{nid}`")

    if orphans:
        lines += [
            f"",
            f"## 고아 노드 (연결 없음)",
            f"",
        ]
        for nid in orphans:
            label = nodes[nid].get("label", nid)
            lines.append(f"- {label}")

    report_path = OUTPUT_DIR / "GRAPH_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report: {report_path}")


def generate_html(nodes, edges):
    """인터랙티브 그래프 시각화 HTML 생성 (D3.js force-directed)"""

    # 노드/엣지를 JSON으로 직렬화
    vis_nodes = []
    color_map = {
        "concepts": "#4ecdc4",
        "entities": "#ff6b6b",
        "sources": "#45b7d1",
        "syntheses": "#f9ca24",
        "root": "#95afc0",
        "unresolved": "#dfe6e9",
    }
    for nid, n in nodes.items():
        cat = n.get("category", "unknown")
        vis_nodes.append({
            "id": nid,
            "label": n.get("label", nid),
            "category": cat,
            "color": color_map.get(cat, "#b2bec3"),
        })

    vis_links = []
    for e in edges:
        vis_links.append({
            "source": e["source"],
            "target": e["target"],
            "type": e.get("type", "wikilink"),
            "relation": e.get("relation", ""),
        })

    data_json = json.dumps({"nodes": vis_nodes, "links": vis_links}, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>Wiki Knowledge Graph</title>
<style>
  body {{ margin: 0; background: #1a1a2e; font-family: sans-serif; overflow: hidden; }}
  svg {{ width: 100vw; height: 100vh; }}
  .link {{ stroke-opacity: 0.4; }}
  .link.wikilink {{ stroke: #636e72; }}
  .link.causal {{ stroke: #e17055; stroke-width: 2.5; marker-end: url(#arrow); }}
  .node circle {{ stroke: #2d3436; stroke-width: 1.5; cursor: pointer; }}
  .node text {{ fill: #dfe6e9; font-size: 10px; pointer-events: none; }}
  .tooltip {{ position: absolute; background: #2d3436; color: #dfe6e9; padding: 8px 12px;
    border-radius: 6px; font-size: 12px; pointer-events: none; display: none; }}
  .legend {{ position: absolute; top: 16px; left: 16px; background: rgba(45,52,54,0.9);
    padding: 12px; border-radius: 8px; color: #dfe6e9; font-size: 12px; }}
  .legend div {{ margin: 4px 0; }}
  .legend span {{ display: inline-block; width: 12px; height: 12px; border-radius: 50%;
    margin-right: 6px; vertical-align: middle; }}
</style>
</head>
<body>
<div class="legend">
  <div><b>Wiki Knowledge Graph</b></div>
  <div><span style="background:#4ecdc4"></span>Concepts</div>
  <div><span style="background:#ff6b6b"></span>Entities</div>
  <div><span style="background:#45b7d1"></span>Sources</div>
  <div><span style="background:#f9ca24"></span>Syntheses</div>
  <div><span style="background:#dfe6e9"></span>Unresolved</div>
  <div style="margin-top:8px;color:#e17055">→ Causal</div>
</div>
<div class="tooltip" id="tooltip"></div>
<svg>
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="20" refY="5"
      markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#e17055"/>
    </marker>
  </defs>
</svg>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const data = {data_json};
const svg = d3.select("svg");
const width = window.innerWidth, height = window.innerHeight;
const tooltip = d3.select("#tooltip");

const sim = d3.forceSimulation(data.nodes)
  .force("link", d3.forceLink(data.links).id(d => d.id).distance(80))
  .force("charge", d3.forceManyBody().strength(-200))
  .force("center", d3.forceCenter(width/2, height/2))
  .force("collision", d3.forceCollide().radius(20));

const link = svg.append("g").selectAll("line")
  .data(data.links).join("line")
  .attr("class", d => "link " + d.type);

const node = svg.append("g").selectAll("g")
  .data(data.nodes).join("g").attr("class", "node")
  .call(d3.drag().on("start", ds).on("drag", dd).on("end", de));

node.append("circle")
  .attr("r", d => d.category === "unresolved" ? 4 : 8)
  .attr("fill", d => d.color);

node.append("text").attr("dx", 12).attr("dy", 4).text(d => d.label);

node.on("mouseover", (e, d) => {{
  tooltip.style("display", "block")
    .html("<b>" + d.label + "</b><br>Category: " + d.category);
}}).on("mousemove", e => {{
  tooltip.style("left", (e.pageX+12)+"px").style("top", (e.pageY-12)+"px");
}}).on("mouseout", () => tooltip.style("display", "none"));

sim.on("tick", () => {{
  link.attr("x1",d=>d.source.x).attr("y1",d=>d.source.y)
    .attr("x2",d=>d.target.x).attr("y2",d=>d.target.y);
  node.attr("transform", d => "translate("+d.x+","+d.y+")");
}});

function ds(e,d){{ if(!e.active) sim.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; }}
function dd(e,d){{ d.fx=e.x; d.fy=e.y; }}
function de(e,d){{ if(!e.active) sim.alphaTarget(0); d.fx=null; d.fy=null; }}
</script>
</body>
</html>"""

    html_path = OUTPUT_DIR / "graph.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"HTML: {html_path}")


if __name__ == "__main__":
    build_graph()
