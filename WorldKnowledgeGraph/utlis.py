import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
import os
from tqdm import tqdm


# Patterns indicating non-semantic categories
# Keep list and collections and geographic information
EXCLUDE_PATTERNS = [
    # Administrative/maintenance
    "_stubs",
    "Wikipedia_",
    "Articles_",
    "Pages_",
    "Redirects_",
    "WikiProject_",
    "User_",
    "Template_",
    "All_",
    "CS1_",
    "Webarchive_",
    "Use_dmy_dates",
    "Use_mdy_dates",
    
    # Lists and collections
    # "Lists_of_",
    # "List_of_",
    
    # Geographic/temporal slicing
    # "_by_country",
    # "_by_continent",
    # "_by_region",
    # "_by_year",
    # "_by_decade",
    # "_by_century",
    # "_by_nationality",
    # "_by_origin",
    
    # Individual instances (not concepts)
    # "Individual_",
    
    # Cultural/media (often tangential)
    # "_in_popular_culture",
    # "_in_fiction",
    # "_in_literature",
    # "_in_film",
    # "_in_television",
    # "_in_art",
    # "_in_music",
    # "_in_video_games",
    # "Fictional_",
    # "Films_about_",
    # "Books_about_",
    # "Songs_about_",
    # "Television_shows_about_",
    # "Video_games_about_",
    # "Comics_about_",
    
    # Organizational
    # "_organizations",
    # "_professionals",
    # "_people",
    # "_companies",
    # "_awards",
]


def is_semantic_category(title):
    """Check if a category title represents a semantic concept."""
    for pattern in EXCLUDE_PATTERNS:
        if pattern in title:
            return False
    return True


def extract_semantic_subgraph(
    out_dir, 
    output_dir, 
    root_title="Main_topic_classifications", 
    apply_pattern_filter=True,
    remove_isolated=True,
    min_pages_for_leaf=10,
):
    """
    Extract a semantic subgraph rooted at root_title.
    
    Args:
        out_dir: Input directory with wiki_*.parquet files
        output_dir: Output directory for filtered files
        root_title: Root category to start from
        apply_pattern_filter: Remove categories matching non-semantic patterns
        remove_isolated: Remove leaf categories with few pages
        min_pages_for_leaf: Minimum pages required for a leaf category to be kept
    """
    # Load the data
    print("Loading categories...")
    categories = pq.read_table(f"{out_dir}/wiki_categories.parquet").to_pandas()
    print(f"  Loaded {len(categories)} categories")
    
    print("Loading edges...")
    edges = pq.read_table(f"{out_dir}/wiki_category_edges.parquet").to_pandas()
    print(f"  Loaded {len(edges)} edges")
    
    print("Loading page-category relationships...")
    page_cats = pq.read_table(f"{out_dir}/wiki_page_categories.parquet").to_pandas()
    print(f"  Loaded {len(page_cats)} page-category links")
    
    # Step 1: Pattern-based filtering
    if apply_pattern_filter:
        print("\n[Step 1] Applying pattern-based filtering...")
        original_count = len(categories)
        semantic_mask = categories["title"].apply(is_semantic_category)
        categories = categories[semantic_mask]
        semantic_ids = set(categories["category_id"])
        print(f"  Removed {original_count - len(categories)} non-semantic categories")
        print(f"  Remaining: {len(categories)} categories")
        
        # Filter edges to only include semantic categories
        original_edge_count = len(edges)
        edges = edges[
            edges["parent_id"].isin(semantic_ids) & 
            edges["child_id"].isin(semantic_ids)
        ]
        print(f"  Removed {original_edge_count - len(edges)} edges")
    else:
        print("\n[Step 1] Skipping pattern-based filtering")
    
    # Step 2: BFS to find reachable categories from root
    print("\n[Step 2] Finding categories reachable from root...")
    title_to_id = dict(zip(categories["title"], categories["category_id"]))
    
    # Build children adjacency
    children_of = defaultdict(list)
    for p, c in edges[["parent_id", "child_id"]].itertuples(index=False):
        children_of[p].append(c)
    
    root_id = title_to_id.get(root_title)
    if root_id is None:
        raise ValueError(f"Category '{root_title}' not found")
    print(f"  Root: '{root_title}' (id: {root_id})")
    
    reachable = {root_id}
    queue = deque([root_id])
    
    with tqdm(desc="  BFS traversal", unit=" nodes") as pbar:
        while queue:
            parent_id = queue.popleft()
            for child_id in children_of.get(parent_id, []):
                if child_id not in reachable:
                    reachable.add(child_id)
                    queue.append(child_id)
                    pbar.update(1)
    
    print(f"  Found {len(reachable)} reachable categories")
    
    # Filter to reachable
    categories = categories[categories["category_id"].isin(reachable)]
    edges = edges[
        edges["parent_id"].isin(reachable) & 
        edges["child_id"].isin(reachable)
    ]
    
    # Step 3: Remove isolated leaf categories
    if remove_isolated:
        print(f"\n[Step 3] Removing isolated leaf categories (min_pages={min_pages_for_leaf})...")
        
        # Iterate until no more removals (removing a leaf might create new leaves)
        iteration = 0
        while True:
            iteration += 1
            
            # Rebuild children_of with current edges
            children_of = defaultdict(list)
            parents_of = defaultdict(list)
            for p, c in edges[["parent_id", "child_id"]].itertuples(index=False):
                children_of[p].append(c)
                parents_of[c].append(p)
            
            # Count pages per category
            current_cat_ids = set(categories["category_id"])
            page_counts = (
                page_cats[page_cats["category_id"].isin(current_cat_ids)]
                .groupby("category_id")
                .size()
                .to_dict()
            )
            
            # Find isolated leaves: no children AND below page threshold
            to_remove = set()
            for cat_id in current_cat_ids:
                has_children = len(children_of.get(cat_id, [])) > 0
                page_count = page_counts.get(cat_id, 0)
                
                if not has_children and page_count < min_pages_for_leaf:
                    # Don't remove root
                    if cat_id != root_id:
                        to_remove.add(cat_id)
            
            if not to_remove:
                print(f"  Iteration {iteration}: No more isolated categories to remove")
                break
            
            print(f"  Iteration {iteration}: Removing {len(to_remove)} isolated categories")
            
            # Remove from categories and edges
            categories = categories[~categories["category_id"].isin(to_remove)]
            edges = edges[
                ~edges["parent_id"].isin(to_remove) & 
                ~edges["child_id"].isin(to_remove)
            ]
        
        print(f"  Final count after isolation removal: {len(categories)} categories")
    else:
        print("\n[Step 3] Skipping isolated category removal")
    
    # Step 4: Filter page-category relationships
    print("\n[Step 4] Filtering page-category links...")
    final_cat_ids = set(categories["category_id"])
    filtered_page_cats = page_cats[page_cats["category_id"].isin(final_cat_ids)]
    
    # Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"  Categories: {len(categories)}")
    print(f"  Edges: {len(edges)}")
    print(f"  Page-category links: {len(filtered_page_cats)}")
    
    # Compute some stats
    children_of = defaultdict(list)
    for p, c in edges[["parent_id", "child_id"]].itertuples(index=False):
        children_of[p].append(c)
    
    leaf_count = sum(1 for cat_id in final_cat_ids if len(children_of.get(cat_id, [])) == 0)
    internal_count = len(categories) - leaf_count
    print(f"  Leaf categories: {leaf_count}")
    print(f"  Internal categories: {internal_count}")
    
    # Save filtered data
    print("\n[Step 5] Saving filtered data...")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"  Writing semantic_categories.parquet...")
    pq.write_table(
        pa.Table.from_pandas(categories, preserve_index=False),
        f"{output_dir}/semantic_categories.parquet"
    )
    
    print(f"  Writing semantic_category_edges.parquet...")
    pq.write_table(
        pa.Table.from_pandas(edges, preserve_index=False),
        f"{output_dir}/semantic_category_edges.parquet"
    )
    
    print(f"  Writing semantic_page_categories.parquet...")
    pq.write_table(
        pa.Table.from_pandas(filtered_page_cats, preserve_index=False),
        f"{output_dir}/semantic_page_categories.parquet"
    )
    
    print("\nDone!")
    return final_cat_ids


class CategoryIndex:
    """Pre-built index for fast category queries."""
    
    def __init__(self, data_dir):
        print(f"Loading data from {data_dir}...")
        self.nodes = pd.read_parquet(f"{data_dir}/semantic_categories.parquet")
        self.edges = pd.read_parquet(f"{data_dir}/semantic_category_edges.parquet")
        
        print("Building index...")
        self.id_to_title = self.nodes.set_index("category_id")["title"].to_dict()
        self.title_to_id = {v: k for k, v in self.id_to_title.items()}
        
        self.children = defaultdict(list)
        self.parents = defaultdict(list)
        
        for p, c in self.edges[["parent_id", "child_id"]].itertuples(index=False):
            self.children[p].append(c)
            self.parents[c].append(p)
        
        print(f"Index ready: {len(self.nodes)} categories, {len(self.edges)} edges")
    
    def query(self, category_title, depth=3):
        """Query a category and show its neighborhood."""
        search_term = category_title.replace(" ", "_")
        matches = self.nodes[self.nodes["title"].str.lower() == search_term.lower()]
        
        if matches.empty:
            print(f"No categories found matching: '{category_title}'")
            # Suggest similar
            similar = [t for t in self.title_to_id.keys() if search_term.lower() in t.lower()]
            if similar:
                print(f"Did you mean: {similar[:10]}")
            return
        
        for _, row in matches.iterrows():
            cat_id = int(row["category_id"])
            print("\n" + "=" * 60)
            print(f"CATEGORY: {row['title']} (ID: {cat_id})")
            print(f"Direct pages: {row.get('page_count', 'N/A')}")
            print("=" * 60)
            
            self._print_tree(cat_id, self.parents, depth, "up")
            # self._print_tree(cat_id, self.children, depth, "down")
    
    def _print_tree(self, start_id, adj, depth, direction):
        """Print tree using simple recursion with global seen set."""
        seen = set()
        
        def rec(node, lvl):
            if node in seen:
                return
            seen.add(node)
            
            indent = "  " * lvl
            title = self.id_to_title.get(node, f"[Unknown_{node}]")
            prefix = "└── " if lvl > 0 else ""
            print(f"{indent}{prefix}{title}")
            
            if lvl >= depth:
                return
            
            for nxt in adj.get(node, []):
                rec(nxt, lvl + 1)
        
        print(f"\n[{direction.upper()} TREE]")
        rec(start_id, 0)