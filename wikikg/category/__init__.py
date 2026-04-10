"""Wikipedia Category Hierarchy Pipeline.

This pipeline builds a category hierarchy from Wikipedia:
1. parse_categories - Extract category nodes and edges from SQL dumps
2. compute_pagerank - Calculate importance scores for each category
3. prune_graph - Remove low-importance categories

Data sources (2026 Schema):
- page.sql.gz: Page metadata
- category.sql.gz: Category metadata
- linktarget.sql.gz: Link targets (The bridge for category parents)
- categorylinks.sql.gz: Category relationships
"""

from .parse_categories import (
    load_page_maps,
    load_categories,
    load_linktarget_map,  # 新增：适配 2026 Schema 的关键映射
    process_categorylinks, # 核心：合并后的高效解析函数
)

__all__ = [
    "load_page_maps",
    "load_categories",
    "load_linktarget_map",
    "process_categorylinks",
]