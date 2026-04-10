"""Wikipedia Hyperlink Graph Pipeline.

This pipeline builds a knowledge graph from Wikipedia page hyperlinks:
1. parse_dumps - Extract nodes (pages) and edges (links) from SQL dumps
2. compute_pagerank - Calculate importance scores for each page
3. prune_graph - Remove low-importance nodes and their edges

Data sources:
- page.sql.gz: Page metadata
- pagelinks.sql.gz: Hyperlink relationships
- redirect.sql.gz: Redirect mappings
"""

from .parse_dumps import (
    load_linktarget_map,    # 2026 新增：Linktarget ID 桥接
    build_page_maps, 
    build_redirect_map, 
    write_edges_parallel    # 升级为并行版本
)
from .compute_pagerank import (
    load_nodes, 
    load_edges, 
    pagerank, 
    write_pagerank
)
from .prune_graph import (
    load_pagerank, 
    write_pruned_nodes, 
    write_pruned_edges
)

__all__ = [
    # 解析与转换 (Parse & Extract)
    "load_linktarget_map",
    "build_page_maps",
    "build_redirect_map",
    "write_edges_parallel",
    
    # 算法 (Algorithms)
    "load_nodes",
    "load_edges",
    "pagerank",
    "write_pagerank",
    
    # 修剪 (Pruning)
    "load_pagerank",
    "write_pruned_nodes",
    "write_pruned_edges",
]
