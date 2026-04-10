"""
category_path_lookup.py

Given a category name, find all paths from that category up to the root
("Main_topic_classifications") in the semantic Wikipedia category graph.

Usage:
    python category_path_lookup.py "Physics"
    python category_path_lookup.py "Physics" --data-dir /cephfs/liuxinyu/wikiKG/tests/semantic_category
    python category_path_lookup.py "Physics" --max-paths 20 --max-depth 15
"""

import argparse
import sys
from collections import defaultdict

import pyarrow.parquet as pq

DEFAULT_DATA_DIR = "/cephfs/liuxinyu/wikiKG/WorldKnowledgeGraph/"
ROOT_TITLE = "Main_topic_classifications"


def load_graph(data_dir):
    cats = pq.read_table(f"{data_dir}/semantic_categories.parquet").to_pandas()
    edges = pq.read_table(f"{data_dir}/semantic_category_edges.parquet").to_pandas()

    id_to_title = dict(zip(cats["category_id"], cats["title"]))
    title_to_id = {v: k for k, v in id_to_title.items()}

    # parent graph: child -> list of parents
    parents_of = defaultdict(list)
    for parent_id, child_id in edges[["parent_id", "child_id"]].itertuples(index=False):
        parents_of[child_id].append(parent_id)

    return id_to_title, title_to_id, parents_of


def find_paths_to_root(start_id, root_id, parents_of, max_paths=50, max_depth=20):
    """
    BFS/DFS upward from start_id to root_id.
    Returns a list of paths, each path being a list of category IDs
    from start_id to root_id (inclusive).
    """
    paths = []
    # Stack entries: (current_id, path_so_far, visited_on_this_path)
    stack = [(start_id, [start_id], {start_id})]

    while stack and len(paths) < max_paths:
        node, path, visited = stack.pop()

        if node == root_id:
            paths.append(path)
            continue

        if len(path) >= max_depth:
            continue

        for parent in parents_of.get(node, []):
            if parent not in visited:
                stack.append((parent, path + [parent], visited | {parent}))

    return paths


def lookup(category_name, data_dir=DEFAULT_DATA_DIR, max_paths=50, max_depth=20):
    print(f"Loading graph from {data_dir} ...")
    id_to_title, title_to_id, parents_of = load_graph(data_dir)

    # Normalise: spaces -> underscores
    query = category_name.replace(" ", "_")
    query_lower = query.lower()
    node_id = title_to_id.get(query)

    # Case-insensitive exact match
    if node_id is None:
        for title, tid in title_to_id.items():
            if title.lower() == query_lower:
                node_id = tid
                query = title
                break

    # Substring match: pick the shortest title containing the query
    if node_id is None:
        matches = [(t, tid) for t, tid in title_to_id.items() if query_lower in t.lower()]
        if len(matches) == 1:
            query, node_id = matches[0]
            print(f"Exact match not found; using closest match: '{query}'")
        elif matches:
            # Prefer shortest title (most specific single match)
            matches.sort(key=lambda x: len(x[0]))
            print(f"Exact match not found. Multiple matches ({len(matches)}) — showing paths to root for best match.")
            print("Other matches:")
            for t, _ in matches[1:11]:
                print(f"  {t}")
            query, node_id = matches[0]
            print(f"Using: '{query}'\n")

    if node_id is None:
        print(f"Category '{category_name}' not found in the semantic graph.")
        sys.exit(1)

    root_id = title_to_id.get(ROOT_TITLE)
    if root_id is None:
        print(f"Root category '{ROOT_TITLE}' not found in graph.")
        sys.exit(1)

    if node_id == root_id:
        print(f"'{query}' IS the root category.")
        return

    print(f"\nCategory : {query} (id={node_id})")
    print(f"Root     : {ROOT_TITLE} (id={root_id})")
    print(f"Searching for paths (max_paths={max_paths}, max_depth={max_depth}) ...\n")

    paths = find_paths_to_root(node_id, root_id, parents_of, max_paths, max_depth)

    if not paths:
        print("No path found to root (the category may be disconnected or depth limit reached).")
        return

    print(f"Found {len(paths)} path(s):\n")
    for i, path in enumerate(paths, 1):
        titles = [id_to_title.get(nid, str(nid)) for nid in path]
        print(f"  Path {i} (length {len(path)}):")
        print("    " + " -> ".join(titles))
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Find all paths from a Wikipedia category up to the root."
    )
    parser.add_argument("category", help="Category name to look up (spaces or underscores)")
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help=f"Directory with semantic_*.parquet files (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--max-paths",
        type=int,
        default=50,
        help="Maximum number of paths to return (default: 50)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=20,
        help="Maximum path depth to explore (default: 20)",
    )
    args = parser.parse_args()
    lookup(args.category, args.data_dir, args.max_paths, args.max_depth)


if __name__ == "__main__":
    main()
