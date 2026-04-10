import pyarrow.parquet as pq
import pyarrow as pa
from collections import deque

def extract_semantic_subgraph(out_dir, output_dir, root_title="Main_topic_classifications"):
    # Load the data
    categories = pq.read_table(f"{out_dir}/wiki_categories.parquet").to_pandas()
    edges = pq.read_table(f"{out_dir}/wiki_category_edges.parquet").to_pandas()
    page_cats = pq.read_table(f"{out_dir}/wiki_page_categories.parquet").to_pandas()
    
    # Build lookups
    title_to_id = dict(zip(categories["title"], categories["category_id"]))
    
    # Build adjacency list (parent -> children)
    children_of = {}
    for _, row in edges.iterrows():
        children_of.setdefault(row["parent_id"], []).append(row["child_id"])
    
    # Find root
    root_id = title_to_id.get(root_title)
    if root_id is None:
        raise ValueError(f"Category '{root_title}' not found")
    
    # BFS to find all reachable categories
    reachable = {root_id}
    queue = deque([root_id])
    
    while queue:
        parent_id = queue.popleft()
        for child_id in children_of.get(parent_id, []):
            if child_id not in reachable:
                reachable.add(child_id)
                queue.append(child_id)
    
    print(f"Found {len(reachable)} categories under '{root_title}'")
    print(f"Original total: {len(categories)} categories")
    print(f"Filtered out: {len(categories) - len(reachable)} non-semantic categories")
    
    # Filter categories
    filtered_categories = categories[categories["category_id"].isin(reachable)]
    
    # Filter edges (both parent AND child must be in reachable set)
    filtered_edges = edges[
        edges["parent_id"].isin(reachable) & 
        edges["child_id"].isin(reachable)
    ]
    
    # Filter page-category relationships
    filtered_page_cats = page_cats[page_cats["category_id"].isin(reachable)]
    
    print(f"Filtered edges: {len(filtered_edges)} (from {len(edges)})")
    print(f"Filtered page-category links: {len(filtered_page_cats)} (from {len(page_cats)})")
    
    # Save filtered data
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    pq.write_table(
        pa.Table.from_pandas(filtered_categories, preserve_index=False),
        f"{output_dir}/semantic_categories.parquet"
    )
    pq.write_table(
        pa.Table.from_pandas(filtered_edges, preserve_index=False),
        f"{output_dir}/semantic_category_edges.parquet"
    )
    pq.write_table(
        pa.Table.from_pandas(filtered_page_cats, preserve_index=False),
        f"{output_dir}/semantic_page_categories.parquet"
    )
    
    return reachable

# Usage
reachable = extract_semantic_subgraph(
    "data/graph/category",
    "data/tests/semantic_category"
)