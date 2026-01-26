import argparse
import pandas as pd
from collections import defaultdict, deque

def build_index(nodes, edges):
    id_to_synset = nodes.set_index("node_id")["synset"].to_dict()
    id_to_def = nodes.set_index("node_id")["definition"].to_dict()
    children = defaultdict(list)
    parents = defaultdict(list)
    for p, c in edges[["parent_id", "child_id"]].itertuples(index=False):
        children[p].append(c)
        parents[c].append(p)
    return id_to_synset, id_to_def, parents, children

def print_tree(start_id, id_to_synset, id_to_def, adj, depth=2, direction="down"):
    seen = set()
    def rec(node, lvl):
        if node in seen:
            return
        seen.add(node)
        indent = "  " * lvl
        label = f"{id_to_synset[node]} — {id_to_def[node]}"
        print(f"{indent}{label}")
        if lvl >= depth:
            return
        for nxt in adj.get(node, []):
            rec(nxt, lvl + 1)
    print(f"[{direction}]")
    rec(start_id, 0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--word", required=True, help="input word, e.g. dog")
    parser.add_argument("--nodes", default="data/graph/wordnet/wn_nodes.parquet")
    parser.add_argument("--edges", default="data/graph/wordnet/wn_edges.parquet")
    parser.add_argument("--depth", type=int, default=2, help="tree depth for up/down")
    args = parser.parse_args()

    nodes = pd.read_parquet(args.nodes)
    edges = pd.read_parquet(args.edges)
    id_to_synset, id_to_def, parents, children = build_index(nodes, edges)

    # 匹配包含该词的 synset，如 dog.n.01
    matches = nodes[nodes["synset"].str.startswith(args.word + ".")]
    if matches.empty:
        print("No synsets found for word:", args.word)
        return

    for _, row in matches.iterrows():
        sid = int(row["node_id"])
        print("\n=== Synset:", row["synset"], "===")
        print_tree(sid, id_to_synset, id_to_def, parents, depth=args.depth, direction="up")
        print_tree(sid, id_to_synset, id_to_def, children, depth=args.depth, direction="down")

if __name__ == "__main__":
    main()
# python wikikg/common/query_KG.py --word dog --depth 3
