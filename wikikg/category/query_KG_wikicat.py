import argparse
import os
import pandas as pd
from collections import defaultdict

def build_index(nodes, edges):
    """构建维基百科分类层级索引"""
    print("Building title lookup table...")
    # 维基数据列名: category_id, title, page_count
    id_to_title = nodes.set_index("category_id")["title"].to_dict()
    
    print("Building adjacency lists (this may take a minute)...")
    children = defaultdict(list)
    parents = defaultdict(list)
    
    # 维基 edges 列名: parent_id, child_id
    for p, c in edges[["parent_id", "child_id"]].itertuples(index=False):
        children[p].append(c)
        parents[c].append(p)
        
    return id_to_title, parents, children

def print_tree(start_id, id_to_title, adj, depth=2, direction="down"):
    """递归打印分类树"""
    seen = set()
    
    def rec(node, lvl):
        if node in seen:
            return
        seen.add(node)
        
        indent = "  " * lvl
        # 处理可能在 edges 中存在但在 nodes 映射中缺失的 ID
        title = id_to_title.get(node, f"[Unknown_ID_{node}]")
        
        # 打印样式优化
        prefix = "└── " if lvl > 0 else ""
        print(f"{indent}{prefix}{title}")
        
        if lvl >= depth:
            return
            
        for nxt in adj.get(node, []):
            rec(nxt, lvl + 1)
            
    print(f"\n[{direction.upper()} TREE]")
    rec(start_id, 0)

def main():
    parser = argparse.ArgumentParser(description="Query Wikipedia Category Hierarchy")
    parser.add_argument("--word", required=True, help="Category keyword to search (e.g., 'Artificial_intelligence')")
    parser.add_argument("--nodes", required=True, help="Path to wiki_categories.parquet")
    parser.add_argument("--edges", required=True, help="Path to wiki_category_edges.parquet")
    parser.add_argument("--depth", type=int, default=2, help="How many levels to explore up/down")
    parser.add_argument("--limit", type=int, default=5, help="Limit number of matching categories to display")
    args = parser.parse_args()

    # 检查文件是否存在
    for f in [args.nodes, args.edges]:
        if not os.path.exists(f):
            print(f"Error: File not found: {f}")
            return

    # 加载数据
    print(f"Loading nodes from {args.nodes}...")
    nodes = pd.read_parquet(args.nodes)
    print(f"Loading edges from {args.edges}...")
    edges = pd.read_parquet(args.edges)
    
    id_to_title, parents, children = build_index(nodes, edges)

    # 搜索匹配的分类
    # 维基百科分类通常使用下划线代替空格，如 "Artificial_intelligence"
    search_term = args.word.replace(" ", "_")
    # 精确匹配标题（不区分大小写建议保留，因为维基标题首字母通常大写）
    matches = nodes[nodes["title"].str.lower() == search_term.lower()]
    
    if matches.empty:
        print(f"No categories found matching: '{args.word}'")
        return

    print(f"\nFound {len(matches)} matches. Showing top {args.limit}:")

    for _, row in matches.head(args.limit).iterrows():
        sid = int(row["category_id"])
        print("\n" + "="*60)
        print(f"CATEGORY: {row['title']} (ID: {sid})")
        print(f"Direct Pages in this category: {row.get('page_count', 'N/A')}")
        print("="*60)
        
        # 向上查父分类 (Super-categories)
        print_tree(sid, id_to_title, parents, depth=args.depth, direction="up")
        
        # # 向下查子分类 (Sub-categories)
        print_tree(sid, id_to_title, children, depth=args.depth, direction="down")

if __name__ == "__main__":
    main()

# python wikikg/category/query_KG_wikicat.py \
#     --word "Machine_learning" \
#     --nodes data/graph/category/wiki_categories.parquet \
#     --edges data/graph/category/wiki_category_edges.parquet \
#     --depth 3