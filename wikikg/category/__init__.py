"""Wikipedia Category Hierarchy Pipeline.

This pipeline builds a category hierarchy from Wikipedia:
1. parse_categories - Extract category nodes and edges from SQL dumps
2. compute_pagerank - Calculate importance scores for each category
3. prune_graph - Remove low-importance categories

Data sources:
- page.sql.gz: Page metadata (for category pages, ns=14)
- category.sql.gz: Category metadata
- categorylinks.sql.gz: Category relationships
"""

from .parse_categories import (
    load_page_maps,
    load_categories,
    write_categories,
    write_category_edges,
    write_page_categories,
)

__all__ = [
    "load_page_maps",
    "load_categories",
    "write_categories",
    "write_category_edges",
    "write_page_categories",
]
