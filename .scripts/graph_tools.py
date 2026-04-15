"""
graph.json 기반 그래프 탐색 도구.
Graphify CLI의 path/explain 버그를 우회하는 자체 구현.

사용법:
  python3 scripts/graph_tools.py explain "DiscoRL"
  python3 scripts/graph_tools.py path "메타러닝" "MuZero"
  python3 scripts/graph_tools.py causal "DiscoRL"
"""

import json
import sys
from collections import deque
from pathlib import Path

GRAPH_PATH = Path("00_graphify-out/graph.json")


def load_graph():
    data = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in data["nodes"]}
    # label → id 매핑 (검색용)
    label_to_id = {}
    for n in data["nodes"]:
        label_to_id[n["label"].lower()] = n["id"]
        label_to_id[n["id"].lower()] = n["id"]
    return nodes, data["links"], label_to_id


def resolve_node(query, label_to_id):
    q = query.lower().strip()
    if q in label_to_id:
        return label_to_id[q]
    # 부분 매칭
    for label, nid in label_to_id.items():
        if q in label:
            return nid
    return None


def explain(query):
    nodes, links, label_to_id = load_graph()
    nid = resolve_node(query, label_to_id)
    if not nid:
        print(f"Node not found: {query}")
        return

    node = nodes[nid]
    print(f"Node: {node['label']}")
    print(f"  ID: {nid}")
    print(f"  Category: {node.get('category', '')}")
    print(f"  File: {node.get('file', '')}")
    print(f"  Tags: {node.get('tags', '')}")

    # 연결된 엣지
    outgoing = [l for l in links if l["source"] == nid]
    incoming = [l for l in links if l["target"] == nid]

    print(f"\nOutgoing ({len(outgoing)}):")
    for l in outgoing:
        target = nodes.get(l["target"], {})
        rel = l.get("relation", l["type"])
        print(f"  → [{rel}] {target.get('label', l['target'])}")

    print(f"\nIncoming ({len(incoming)}):")
    for l in incoming:
        source = nodes.get(l["source"], {})
        rel = l.get("relation", l["type"])
        print(f"  ← [{rel}] {source.get('label', l['source'])}")


def find_path(start_q, end_q):
    nodes, links, label_to_id = load_graph()
    start = resolve_node(start_q, label_to_id)
    end = resolve_node(end_q, label_to_id)

    if not start:
        print(f"Start node not found: {start_q}")
        return
    if not end:
        print(f"End node not found: {end_q}")
        return

    # BFS
    adj = {}
    for l in links:
        adj.setdefault(l["source"], []).append(l)
        adj.setdefault(l["target"], []).append(
            {**l, "source": l["target"], "target": l["source"], "_reverse": True}
        )

    visited = {start}
    queue = deque([(start, [(start, None)])])

    while queue:
        current, path = queue.popleft()
        if current == end:
            print(f"Path ({len(path)-1} hops):\n")
            for i, (nid, edge) in enumerate(path):
                node = nodes.get(nid, {})
                label = node.get("label", nid)
                if edge:
                    rel = edge.get("relation", edge["type"])
                    direction = "←" if edge.get("_reverse") else "→"
                    print(f"  {direction} [{rel}] {label}")
                else:
                    print(f"  ● {label}")
            return

        for link in adj.get(current, []):
            neighbor = link["target"]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [(neighbor, link)]))

    print(f"No path found between {start_q} and {end_q}")


def causal_chain(query):
    nodes, links, label_to_id = load_graph()
    nid = resolve_node(query, label_to_id)
    if not nid:
        print(f"Node not found: {query}")
        return

    causal_links = [l for l in links if l["type"] == "causal"]
    related = [l for l in causal_links if l["source"] == nid or l["target"] == nid]

    if not related:
        print(f"No causal relations found for: {query}")
        return

    node = nodes.get(nid, {})
    print(f"Causal relations for: {node.get('label', nid)}\n")
    for l in related:
        src = nodes.get(l["source"], {}).get("label", l["source"])
        tgt = nodes.get(l["target"], {}).get("label", l["target"])
        rel = l.get("relation", "?")
        print(f"  {src} →({rel})→ {tgt}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print('  python3 scripts/graph_tools.py explain "DiscoRL"')
        print('  python3 scripts/graph_tools.py path "메타러닝" "MuZero"')
        print('  python3 scripts/graph_tools.py causal "DiscoRL"')
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "explain":
        explain(sys.argv[2])
    elif cmd == "path":
        find_path(sys.argv[2], sys.argv[3])
    elif cmd == "causal":
        causal_chain(sys.argv[2])
    else:
        print(f"Unknown command: {cmd}")
