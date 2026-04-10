import argparse
import sys
import time
from collections import defaultdict

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def load_visual_nodes(visual_nodes_path: str) -> dict[int, str]:
    """Load visualizable node IDs and titles from VLM-based filtering."""
    
    visual_nodes = {}
    
    log(f"Loading VLM-filtered visual nodes from {visual_nodes_path}")
    table = pq.read_table(visual_nodes_path)
    for pid, title in zip(table.column("page_id").to_pylist(),
                          table.column("title").to_pylist()):
        visual_nodes[pid] = title
    log(f"  Loaded {len(visual_nodes):,} visual nodes")
    
    return visual_nodes


def load_page_categories(page_categories_path: str, visual_ids: set[int]) -> dict[int, set[int]]:
    """Load page -> category mappings, filtered to visual nodes only."""
    
    log(f"Loading page-category mappings from {page_categories_path}")
    table = pq.read_table(page_categories_path)
    page_ids = table.column("page_id").to_numpy()
    cat_ids = table.column("category_id").to_numpy()
    
    page_to_cats = defaultdict(set)
    kept = 0
    
    for pid, cid in tqdm(zip(page_ids, cat_ids), total=len(page_ids), 
                         desc="Filtering mappings", unit="row"):
        if pid in visual_ids:
            page_to_cats[pid].add(cid)
            kept += 1
    
    log(f"  Kept {kept:,} mappings for {len(page_to_cats):,} visual nodes")
    return dict(page_to_cats)


def load_category_hierarchy(
    categories_path: str,
    category_edges_path: str
) -> tuple[dict[int, str], dict[int, set[int]], dict[int, set[int]]]:
    """Load category metadata and hierarchy."""
    
    # Load category metadata
    log(f"Loading categories from {categories_path}")
    table = pq.read_table(categories_path)
    cat_ids = table.column("category_id").to_pylist()
    cat_titles = table.column("title").to_pylist()
    cat_id_to_title = dict(zip(cat_ids, cat_titles))
    log(f"  Loaded {len(cat_id_to_title):,} categories")
    
    # Load hierarchy edges
    log(f"Loading category edges from {category_edges_path}")
    table = pq.read_table(category_edges_path)
    parent_ids = table.column("parent_id").to_numpy()
    child_ids = table.column("child_id").to_numpy()
    
    # Build parent -> children and child -> parents maps
    parent_to_children = defaultdict(set)
    child_to_parents = defaultdict(set)
    
    for pid, cid in tqdm(zip(parent_ids, child_ids), total=len(parent_ids),
                         desc="Building hierarchy", unit="edge"):
        parent_to_children[pid].add(cid)
        child_to_parents[cid].add(pid)
    
    log(f"  Loaded {len(parent_ids):,} hierarchy edges")
    log(f"  Categories with children: {len(parent_to_children):,}")
    log(f"  Categories with parents: {len(child_to_parents):,}")
    
    return cat_id_to_title, dict(parent_to_children), dict(child_to_parents)


def find_relevant_categories(
    page_to_cats: dict[int, set[int]],
    child_to_parents: dict[int, set[int]],
    max_depth: int = 5
) -> set[int]:
    """Find all categories reachable from visual nodes up to max_depth."""
    
    log(f"Finding relevant categories (max_depth={max_depth})...")
    
    # Start with direct categories of visual nodes
    relevant = set()
    for cats in page_to_cats.values():
        relevant.update(cats)
    
    log(f"  Direct categories: {len(relevant):,}")
    
    # Walk up the hierarchy
    frontier = relevant.copy()
    for depth in range(1, max_depth + 1):
        next_frontier = set()
        for cat_id in frontier:
            parents = child_to_parents.get(cat_id, set())
            next_frontier.update(parents - relevant)
        
        if not next_frontier:
            log(f"  Depth {depth}: no new categories, stopping")
            break
        
        relevant.update(next_frontier)
        frontier = next_frontier
        log(f"  Depth {depth}: +{len(next_frontier):,} categories (total: {len(relevant):,})")
    
    return relevant


def compute_category_stats(
    page_to_cats: dict[int, set[int]],
    relevant_cats: set[int]
) -> dict[int, int]:
    """Count how many visual nodes are directly under each category."""
    
    log("Computing category statistics...")
    
    cat_counts = defaultdict(int)
    for cats in tqdm(page_to_cats.values(), desc="Counting", unit="node"):
        for cat_id in cats:
            if cat_id in relevant_cats:
                cat_counts[cat_id] += 1
    
    return dict(cat_counts)


def compute_descendant_counts(
    cat_counts: dict[int, int],
    parent_to_children: dict[int, set[int]],
    relevant_cats: set[int]
) -> dict[int, int]:
    """Compute total concepts under each category (including descendants).
    
    Uses bottom-up aggregation: leaf counts propagate up to parents.
    """
    
    log("Computing descendant concept counts...")
    
    # Filter hierarchy to relevant categories
    filtered_children = {
        cat: children & relevant_cats 
        for cat, children in parent_to_children.items() 
        if cat in relevant_cats
    }
    
    # Compute in-degree for topological sort
    in_degree = defaultdict(int)
    for cat in relevant_cats:
        in_degree[cat] = 0
    for children in filtered_children.values():
        for child in children:
            in_degree[child] += 1
    
    # Find roots (no parents in relevant set)
    roots = [cat for cat in relevant_cats if in_degree[cat] == 0]
    
    # BFS from roots, but we need reverse order (leaves first)
    # So we do topological sort and reverse it
    order = []
    queue = list(roots)
    visited = set()
    
    while queue:
        cat = queue.pop(0)
        if cat in visited:
            continue
        visited.add(cat)
        order.append(cat)
        for child in filtered_children.get(cat, set()):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)
    
    # Process in reverse order (leaves to roots)
    descendant_counts = dict(cat_counts)  # Start with direct counts
    
    for cat in tqdm(reversed(order), total=len(order), desc="Aggregating", unit="cat"):
        for child in filtered_children.get(cat, set()):
            descendant_counts[cat] = descendant_counts.get(cat, 0) + descendant_counts.get(child, 0)
    
    return descendant_counts


def filter_empty_categories(
    relevant_cats: set[int],
    cat_counts: dict[int, int],
    descendant_counts: dict[int, int],
    parent_to_children: dict[int, set[int]],
    child_to_parents: dict[int, set[int]],
    cat_id_to_title: dict[int, str],
    keep_internal: bool = True,
    min_concepts: int = 1
) -> set[int]:
    """Remove categories without sufficient concepts.
    
    Args:
        relevant_cats: Set of category IDs to filter
        cat_counts: Direct concept counts per category
        descendant_counts: Total concepts (including descendants) per category
        parent_to_children: Category hierarchy (parent -> children)
        child_to_parents: Category hierarchy (child -> parents)
        cat_id_to_title: Category ID to title mapping
        keep_internal: If True, keep categories with no direct concepts but populated children
        min_concepts: Minimum concepts required (direct or descendant based on keep_internal)
    
    Returns:
        Filtered set of category IDs
    """
    
    log(f"Filtering empty categories (keep_internal={keep_internal}, min_concepts={min_concepts})...")
    
    before_count = len(relevant_cats)
    
    if keep_internal:
        # Keep if has enough descendants (direct + children's concepts)
        filtered = {
            cat for cat in relevant_cats 
            if descendant_counts.get(cat, 0) >= min_concepts
        }
    else:
        # Keep only if has enough direct concepts
        filtered = {
            cat for cat in relevant_cats 
            if cat_counts.get(cat, 0) >= min_concepts
        }
    
    removed = before_count - len(filtered)
    log(f"  Removed {removed:,} empty categories ({before_count:,} -> {len(filtered):,})")
    
    # Show some examples of removed categories
    removed_cats = relevant_cats - filtered
    if removed_cats:
        log("  Sample removed categories:")
        for cat_id in list(removed_cats)[:10]:
            title = cat_id_to_title.get(cat_id, "Unknown")
            direct = cat_counts.get(cat_id, 0)
            total = descendant_counts.get(cat_id, 0)
            log(f"    - {title[:50]:50s} (direct={direct}, total={total})")
    
    return filtered


def prune_category_hierarchy(
    relevant_cats: set[int],
    parent_to_children: dict[int, set[int]],
    child_to_parents: dict[int, set[int]],
    cat_id_to_title: dict[int, str]
) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    """Prune hierarchy to only include relevant categories."""
    
    log("Pruning category hierarchy...")
    
    pruned_parent_to_children = {}
    pruned_child_to_parents = {}
    
    for cat_id in tqdm(relevant_cats, desc="Pruning", unit="cat"):
        # Filter children to only relevant ones
        children = parent_to_children.get(cat_id, set())
        relevant_children = children & relevant_cats
        if relevant_children:
            pruned_parent_to_children[cat_id] = relevant_children
        
        # Filter parents to only relevant ones
        parents = child_to_parents.get(cat_id, set())
        relevant_parents = parents & relevant_cats
        if relevant_parents:
            pruned_child_to_parents[cat_id] = relevant_parents
    
    # Count edges
    total_edges = sum(len(children) for children in pruned_parent_to_children.values())
    log(f"  Pruned hierarchy: {len(relevant_cats):,} categories, {total_edges:,} edges")
    
    return pruned_parent_to_children, pruned_child_to_parents


def compute_node_depths(
    relevant_cats: set[int],
    pruned_child_to_parents: dict[int, set[int]],
    page_to_cats: dict[int, set[int]]
) -> tuple[dict[int, int], dict[int, int], int]:
    """Compute depth from root for all nodes.
    
    Depth 0 = root categories (no parents)
    Concepts get depth = min(parent category depths) + 1
    
    For DAG handling, we use minimum depth (most general path to root).
    
    Returns:
        category_depths: category_id -> depth
        concept_depths: concept_id -> depth
        max_depth: maximum depth in the graph
    """
    
    log("Computing node depths...")
    
    # Find root categories (no parents in pruned hierarchy)
    cats_with_parents = set(pruned_child_to_parents.keys())
    roots = relevant_cats - cats_with_parents
    log(f"  Found {len(roots):,} root categories")
    
    # BFS from roots to compute category depths
    category_depths = {r: 0 for r in roots}
    queue = list(roots)
    
    # Build parent -> children for traversal
    parent_to_children = defaultdict(set)
    for child, parents in pruned_child_to_parents.items():
        for parent in parents:
            parent_to_children[parent].add(child)
    
    while queue:
        cat = queue.pop(0)
        current_depth = category_depths[cat]
        
        for child in parent_to_children.get(cat, set()):
            if child not in category_depths:
                category_depths[child] = current_depth + 1
                queue.append(child)
            else:
                # Already visited - keep minimum depth (DAG handling)
                category_depths[child] = min(category_depths[child], current_depth + 1)
    
    # Compute concept depths (min parent category depth + 1)
    concept_depths = {}
    for concept_id, cats in page_to_cats.items():
        relevant_cat_depths = [
            category_depths[c] for c in cats 
            if c in category_depths
        ]
        if relevant_cat_depths:
            concept_depths[concept_id] = min(relevant_cat_depths) + 1
        else:
            concept_depths[concept_id] = 0  # No category = treat as root level
    
    # Compute max depth
    max_cat_depth = max(category_depths.values()) if category_depths else 0
    max_concept_depth = max(concept_depths.values()) if concept_depths else 0
    max_depth = max(max_cat_depth, max_concept_depth)
    
    log(f"  Category depth range: 0-{max_cat_depth}")
    log(f"  Concept depth range: 0-{max_concept_depth}")
    
    # Depth histogram for categories
    cat_depth_hist = defaultdict(int)
    for d in category_depths.values():
        cat_depth_hist[d] += 1
    log("  Category depth distribution:")
    for d in sorted(cat_depth_hist.keys()):
        log(f"    Depth {d}: {cat_depth_hist[d]:,} categories")
    
    return category_depths, concept_depths, max_depth


def compute_ancestor_paths(
    relevant_cats: set[int],
    pruned_child_to_parents: dict[int, set[int]],
    page_to_cats: dict[int, set[int]],
    category_depths: dict[int, int]
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """Precompute ancestor paths for all nodes.
    
    For sampling weight computation, we need to propagate concept matches
    up to ancestor categories. This precomputes the path to root for each node.
    
    Returns:
        category_ancestors: category_id -> list of ancestor category_ids (ordered root to parent)
        concept_ancestors: concept_id -> list of ancestor category_ids (ordered root to direct parent)
    """
    
    log("Computing ancestor paths...")
    
    # For each category, compute ancestors (BFS upward)
    category_ancestors = {}
    
    for cat_id in tqdm(relevant_cats, desc="Category ancestors", unit="cat"):
        ancestors = []
        visited = set()
        queue = list(pruned_child_to_parents.get(cat_id, set()))
        
        while queue:
            parent = queue.pop(0)
            if parent in visited:
                continue
            visited.add(parent)
            ancestors.append(parent)
            queue.extend(pruned_child_to_parents.get(parent, set()))
        
        # Sort by depth (root first)
        ancestors.sort(key=lambda x: category_depths.get(x, 0))
        category_ancestors[cat_id] = ancestors
    
    # For each concept, compute ancestors through its direct categories
    concept_ancestors = {}
    
    for concept_id, cats in tqdm(page_to_cats.items(), desc="Concept ancestors", unit="concept"):
        all_ancestors = set()
        direct_cats = [c for c in cats if c in relevant_cats]
        
        # Add direct categories
        all_ancestors.update(direct_cats)
        
        # Add all ancestors of direct categories
        for cat_id in direct_cats:
            all_ancestors.update(category_ancestors.get(cat_id, []))
        
        # Sort by depth (root first)
        ancestor_list = list(all_ancestors)
        ancestor_list.sort(key=lambda x: category_depths.get(x, 0))
        concept_ancestors[concept_id] = ancestor_list
    
    # Stats
    avg_cat_ancestors = np.mean([len(a) for a in category_ancestors.values()]) if category_ancestors else 0
    avg_concept_ancestors = np.mean([len(a) for a in concept_ancestors.values()]) if concept_ancestors else 0
    log(f"  Average ancestors per category: {avg_cat_ancestors:.1f}")
    log(f"  Average ancestors per concept: {avg_concept_ancestors:.1f}")
    
    return category_ancestors, concept_ancestors


def write_unified_knowledge_graph(
    visual_nodes: dict[int, str],
    cat_id_to_title: dict[int, str],
    relevant_cats: set[int],
    page_to_cats: dict[int, set[int]],
    pruned_parent_to_children: dict[int, set[int]],
    pruned_child_to_parents: dict[int, set[int]],
    cat_counts: dict[int, int],
    descendant_counts: dict[int, int],
    category_depths: dict[int, int],
    concept_depths: dict[int, int],
    concept_ancestors: dict[int, list[int]],
    max_depth: int,
    out_dir: str
):
    """Write unified knowledge graph to parquet files.
    
    Output:
        nodes.parquet - all nodes (concepts + categories) with unified schema
        edges.parquet - all edges (belongs_to + child_of) with unified schema
        title_index.parquet - title -> node_id mapping for fast lookup
        ancestors.parquet - node_id -> ancestor list for weight propagation
        metadata.json - graph statistics and configuration
    """
    
    import os
    import json
    os.makedirs(out_dir, exist_ok=True)
    
    # Check for ID collisions between concepts and categories
    concept_ids = set(visual_nodes.keys())
    category_ids = set(relevant_cats)
    collision = concept_ids & category_ids
    
    if collision:
        log(f"Warning: {len(collision):,} ID collisions between concepts and categories")
        log("  Applying offset to category IDs")
        offset = max(concept_ids) + 1
    else:
        offset = 0
        log("No ID collisions detected")
    
    # Build unified nodes table
    log("Building unified nodes table...")
    
    node_ids = []
    original_ids = []  # Keep track of original Wikipedia IDs
    titles = []
    node_types = []
    depths = []
    normalized_depths = []  # 0-1 scale for curriculum
    direct_concept_counts = []
    total_concept_counts = []
    specificity_scores = []  # Precomputed inverse frequency
    
    # Add concept nodes
    for concept_id, title in tqdm(visual_nodes.items(), desc="Adding concepts", unit="node"):
        node_ids.append(concept_id)
        original_ids.append(concept_id)
        titles.append(title)
        node_types.append("concept")
        
        d = concept_depths.get(concept_id, 0)
        depths.append(d)
        normalized_depths.append(d / max_depth if max_depth > 0 else 0.0)
        
        direct_concept_counts.append(1)  # Each concept counts as 1
        total_concept_counts.append(1)
        
        # Specificity = 1.0 for concepts (most specific)
        specificity_scores.append(1.0)
    
    # Add category nodes
    for cat_id in tqdm(relevant_cats, desc="Adding categories", unit="cat"):
        node_ids.append(cat_id + offset)
        original_ids.append(cat_id)
        titles.append(cat_id_to_title.get(cat_id, f"Category_{cat_id}"))
        node_types.append("category")
        
        d = category_depths.get(cat_id, 0)
        depths.append(d)
        normalized_depths.append(d / max_depth if max_depth > 0 else 0.0)
        
        direct_count = cat_counts.get(cat_id, 0)
        total_count = descendant_counts.get(cat_id, 0)
        direct_concept_counts.append(direct_count)
        total_concept_counts.append(total_count)
        
        # Specificity = inverse log of total concepts (larger categories = less specific)
        specificity_scores.append(1.0 / np.log1p(total_count) if total_count > 0 else 1.0)
    
    nodes_table = pa.table({
        "node_id": node_ids,
        "original_id": original_ids,
        "title": titles,
        "node_type": node_types,
        "depth": depths,
        "normalized_depth": normalized_depths,
        "direct_concept_count": direct_concept_counts,
        "total_concept_count": total_concept_counts,
        "specificity_score": specificity_scores
    })
    
    nodes_path = os.path.join(out_dir, "nodes.parquet")
    pq.write_table(nodes_table, nodes_path, compression="zstd")
    log(f"  Wrote {len(node_ids):,} nodes to {nodes_path}")
    
    # Build unified edges table
    log("Building unified edges table...")
    
    src_ids = []
    dst_ids = []
    edge_types = []
    
    # Concept -> Category edges (belongs_to)
    for concept_id, cats in tqdm(page_to_cats.items(), desc="Adding belongs_to edges", unit="concept"):
        for cat_id in cats:
            if cat_id in relevant_cats:
                src_ids.append(concept_id)
                dst_ids.append(cat_id + offset)
                edge_types.append("belongs_to")
    
    # Category -> Parent edges (child_of)
    for child_id, parents in tqdm(pruned_child_to_parents.items(), desc="Adding child_of edges", unit="cat"):
        for parent_id in parents:
            src_ids.append(child_id + offset)
            dst_ids.append(parent_id + offset)
            edge_types.append("child_of")
    
    edges_table = pa.table({
        "src_id": src_ids,
        "dst_id": dst_ids,
        "edge_type": edge_types
    })
    
    edges_path = os.path.join(out_dir, "edges.parquet")
    pq.write_table(edges_table, edges_path, compression="zstd")
    log(f"  Wrote {len(src_ids):,} edges to {edges_path}")
    
    # Build title index for fast lookup during sampling
    log("Building title index...")
    
    # Lowercase titles for case-insensitive matching
    title_lower = [t.lower() for t in titles]
    
    title_index_table = pa.table({
        "title_lower": title_lower,
        "title": titles,
        "node_id": node_ids,
        "node_type": node_types
    })
    
    title_index_path = os.path.join(out_dir, "title_index.parquet")
    pq.write_table(title_index_table, title_index_path, compression="zstd")
    log(f"  Wrote {len(titles):,} title entries to {title_index_path}")
    
    # Build ancestors table for weight propagation
    log("Building ancestors table...")
    
    ancestor_node_ids = []
    ancestor_lists = []
    
    # For concepts, store their category ancestors (with offset applied)
    for concept_id in visual_nodes.keys():
        ancestor_node_ids.append(concept_id)
        ancestors = concept_ancestors.get(concept_id, [])
        # Apply offset to category IDs
        ancestor_lists.append([a + offset for a in ancestors])
    
    # For categories, we can derive ancestors from edges, so skip here
    # (keeps file smaller, ancestors can be computed at runtime if needed)
    
    ancestors_table = pa.table({
        "node_id": ancestor_node_ids,
        "ancestors": ancestor_lists  # List of category node_ids from root to direct parent
    })
    
    ancestors_path = os.path.join(out_dir, "ancestors.parquet")
    pq.write_table(ancestors_table, ancestors_path, compression="zstd")
    log(f"  Wrote {len(ancestor_node_ids):,} ancestor entries to {ancestors_path}")
    
    # Write metadata
    metadata = {
        "id_offset": offset,
        "num_concepts": len(visual_nodes),
        "num_categories": len(relevant_cats),
        "num_nodes": len(node_ids),
        "num_belongs_to_edges": sum(1 for e in edge_types if e == "belongs_to"),
        "num_child_of_edges": sum(1 for e in edge_types if e == "child_of"),
        "max_depth": max_depth,
        "max_category_depth": max(category_depths.values()) if category_depths else 0,
        "max_concept_depth": max(concept_depths.values()) if concept_depths else 0,
        "depth_distribution": {
            str(d): sum(1 for x in depths if x == d) 
            for d in range(max_depth + 1)
        }
    }
    
    metadata_path = os.path.join(out_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log(f"  Wrote metadata to {metadata_path}")
    
    # Print summary
    log("\n" + "=" * 60)
    log("UNIFIED KNOWLEDGE GRAPH SUMMARY")
    log("=" * 60)
    log(f"Total nodes:              {len(node_ids):,}")
    log(f"  - Concept nodes:        {len(visual_nodes):,}")
    log(f"  - Category nodes:       {len(relevant_cats):,}")
    log(f"Total edges:              {len(src_ids):,}")
    log(f"  - belongs_to edges:     {metadata['num_belongs_to_edges']:,}")
    log(f"  - child_of edges:       {metadata['num_child_of_edges']:,}")
    log(f"ID offset applied:        {offset}")
    
    # Find and report root categories
    cats_with_parents = set(pruned_child_to_parents.keys())
    roots = relevant_cats - cats_with_parents
    log(f"Root categories:          {len(roots):,}")
    
    # Depth statistics
    log(f"\nDepth statistics:")
    log(f"  Max depth:              {max_depth}")
    log(f"  Max category depth:     {metadata['max_category_depth']}")
    log(f"  Max concept depth:      {metadata['max_concept_depth']}")
    
    # Top categories by total concept count
    log("\nTop 20 categories by total concept count:")
    sorted_cats = sorted(
        [(c, descendant_counts.get(c, 0)) for c in relevant_cats],
        key=lambda x: -x[1]
    )[:20]
    for cat_id, count in sorted_cats:
        title = cat_id_to_title.get(cat_id, "Unknown")[:40]
        depth = category_depths.get(cat_id, 0)
        spec = 1.0 / np.log1p(count) if count > 0 else 1.0
        log(f"  {title:40s} count={count:6,}  depth={depth}  spec={spec:.3f}")
    
    # Sample root categories
    log(f"\nSample root categories (depth=0):")
    for cat_id in list(roots)[:10]:
        title = cat_id_to_title.get(cat_id, "Unknown")[:50]
        count = descendant_counts.get(cat_id, 0)
        log(f"  {title:50s} count={count:,}")
    
    # Output files summary
    log(f"\nOutput files:")
    log(f"  {nodes_path}")
    log(f"  {edges_path}")
    log(f"  {title_index_path}")
    log(f"  {ancestors_path}")
    log(f"  {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build unified knowledge graph from visual nodes and Wikipedia categories",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input files
    parser.add_argument("--visual-nodes", required=True,
                        help="Visual nodes from VLM-based filtering (parquet with page_id, title)")
    parser.add_argument("--page-categories", required=True,
                        help="wiki_page_categories.parquet")
    parser.add_argument("--categories", required=True,
                        help="wiki_categories.parquet")
    parser.add_argument("--category-edges", required=True,
                        help="wiki_category_edges.parquet")
    
    # Output
    parser.add_argument("--out-dir", required=True,
                        help="Output directory for knowledge graph files")
    
    # Options
    parser.add_argument("--max-depth", type=int, default=5,
                        help="Maximum depth to traverse up category hierarchy")
    parser.add_argument("--min-concepts", type=int, default=1,
                        help="Minimum concepts required to keep a category")
    parser.add_argument("--keep-internal", action="store_true", default=True,
                        help="Keep internal categories with populated children but no direct concepts")
    parser.add_argument("--no-keep-internal", action="store_false", dest="keep_internal",
                        help="Remove categories without direct concepts (even if children have concepts)")
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Load visual nodes
    visual_nodes = load_visual_nodes(args.visual_nodes)
    visual_ids = set(visual_nodes.keys())
    
    # Load page-category mappings
    page_to_cats = load_page_categories(args.page_categories, visual_ids)
    
    # Load category hierarchy
    cat_id_to_title, parent_to_children, child_to_parents = load_category_hierarchy(
        args.categories, args.category_edges
    )
    
    # Find relevant categories
    relevant_cats = find_relevant_categories(
        page_to_cats, child_to_parents, args.max_depth
    )
    
    # Compute statistics (before filtering)
    cat_counts = compute_category_stats(page_to_cats, relevant_cats)
    
    # Compute descendant counts for internal node filtering
    descendant_counts = compute_descendant_counts(
        cat_counts, parent_to_children, relevant_cats
    )
    
    # Filter empty categories
    relevant_cats = filter_empty_categories(
        relevant_cats=relevant_cats,
        cat_counts=cat_counts,
        descendant_counts=descendant_counts,
        parent_to_children=parent_to_children,
        child_to_parents=child_to_parents,
        cat_id_to_title=cat_id_to_title,
        keep_internal=args.keep_internal,
        min_concepts=args.min_concepts
    )
    
    # Recompute descendant counts after filtering
    descendant_counts = compute_descendant_counts(
        cat_counts, parent_to_children, relevant_cats
    )
    
    # Prune hierarchy
    pruned_parent_to_children, pruned_child_to_parents = prune_category_hierarchy(
        relevant_cats, parent_to_children, child_to_parents, cat_id_to_title
    )
    
    # Compute node depths
    category_depths, concept_depths, max_depth = compute_node_depths(
        relevant_cats, pruned_child_to_parents, page_to_cats
    )
    
    # Compute ancestor paths for weight propagation
    category_ancestors, concept_ancestors = compute_ancestor_paths(
        relevant_cats, pruned_child_to_parents, page_to_cats, category_depths
    )
    
    # Write unified output
    write_unified_knowledge_graph(
        visual_nodes=visual_nodes,
        cat_id_to_title=cat_id_to_title,
        relevant_cats=relevant_cats,
        page_to_cats=page_to_cats,
        pruned_parent_to_children=pruned_parent_to_children,
        pruned_child_to_parents=pruned_child_to_parents,
        cat_counts=cat_counts,
        descendant_counts=descendant_counts,
        category_depths=category_depths,
        concept_depths=concept_depths,
        concept_ancestors=concept_ancestors,
        max_depth=max_depth,
        out_dir=args.out_dir
    )
    
    total_time = time.time() - start_time
    log(f"\nTotal time: {total_time:.1f} seconds")


if __name__ == "__main__":
    main()