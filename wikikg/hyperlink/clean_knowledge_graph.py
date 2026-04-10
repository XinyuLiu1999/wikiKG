#!/usr/bin/env python3
"""
Clean Knowledge Graph - Comprehensive Semantic Cleaning

This script performs thorough cleaning of a Wikipedia-derived knowledge graph to:
1. Remove maintenance/administrative categories (non-semantic)
2. Break cycles in the category hierarchy
3. Remove orphaned nodes
4. Recompute depths correctly using BFS from true roots
5. Recalculate specificity scores
6. Prune low-value root categories
7. Rebuild all derived data (ancestors, title index)

Usage:
    python clean_knowledge_graph.py <input_dir> <output_dir> [options]

Options:
    --dry-run           Show what would be cleaned without writing output
    --min-root-concepts N  Minimum concepts for a root category (default: 5)
    --verbose           Show detailed progress
    --keep-patterns     File with patterns to keep (one regex per line)
    --remove-patterns   File with additional patterns to remove
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# =============================================================================
# CONFIGURATION: Patterns for identifying non-semantic categories
# =============================================================================

# Categories matching these patterns are Wikipedia maintenance/administrative
# and should be removed for a semantically clean graph
MAINTENANCE_PATTERNS = [
    # Wikipedia/Wikimedia administrative categories
    r"^Articles_",
    r"^All_articles_",
    r"^Pages_",
    r"^All_pages_",
    r"^Wikipedia_",
    r"^Wikipedians_",
    r"^Wikidata_",
    r"^Commons_",
    r"^Webarchive_",
    r"^CS1_",  # Citation Style 1 errors
    r"^Use_dmy_dates",
    r"^Use_mdy_dates",
    r"^Use_American_English",
    r"^Use_British_English",
    r"^Use_Indian_English",
    r"^Use_Canadian_English",
    r"^Use_Australian_English",
    r"^Engvar",
    r"^Short_description",
    r"^Long_short_description",
    r"^Good_articles",
    r"^Featured_articles",
    r"^Spoken_articles",
    r"^Cleanup_tagged_articles",
    r"^Accuracy_disputes",
    r"^Articles_needing",
    r"^Articles_lacking",
    r"^Articles_containing",
    r"^Articles_to_be",
    r"^Articles_with",
    r"^All_stub_articles",
    r"^Stub_message_templates",
    r"_stubs$",
    r"^Redirects_",
    r"^All_redirects",
    r"^Template_",
    r"^Templates_",
    r"^Category_templates",
    r"^Infobox_templates",
    r"^Navbox_templates",
    r"^Sidebar_templates",
    r"^Portal_",
    r"^WikiProject_",
    r"^Project_",
    r"^Noindexed_pages",
    r"^Tracked_pages",
    r"^Dynamic_lists",
    r"^Incomplete_lists",
    r"^Pages_using",
    r"^Pages_with",
    r"^EngvarB",
    r"^DISPLAYTITLE",
    r"^DEFAULTSORT",
    r"^Hidden_categories",
    r"^Tracking_categories",
    r"^Container_categories",
    r"^Maintenance_categories",
    r"^Administration_categories",
    r"^Eponymous_categories",
    
    # Date-based maintenance
    r"^All_articles_from_",
    r"_from_\w+_\d{4}$",  # e.g., "Articles_from_January_2020"
    r"_since_\w+_\d{4}$",
    r"^Pending_",
    r"^Proposed_",
    
    # Quality assessment
    r"^[A-Z]-Class_",
    r"^FA-Class_",
    r"^GA-Class_",
    r"^B-Class_",
    r"^C-Class_",
    r"^Start-Class_",
    r"^Stub-Class_",
    r"^Unassessed_",
    r"^Low-importance_",
    r"^Mid-importance_",
    r"^High-importance_",
    r"^Top-importance_",
    r"^Unknown-importance_",
    r"^Taxonbars_",
    r"^CS1:",
    r"^All_Wikipedia_",
    
    # External link categories
    r"^AC_with_",  # Authority control
    r"^ISNI_",
    r"^VIAF_",
    r"^LCCN_",
    r"^GND_",
    r"^BNF_",
    r"^BIBSYS_",
    r"^IMDb_",
    r"^MusicBrainz_",
    r"^Discogs_",
    r"^SNAC-ID_",
    r"^Worldcat_",
    r"^NLA_",
    r"^SUDOC_",
    
    # Bot-related
    r"^Bot-",
    r"^Bots_",
    r"^AWB_",
    
    # Disputed/problematic content markers
    r"^Disputed_",
    r"^Controversial_",
    r"^POV_",
    r"^NPOV_",
    r"^Orphaned_articles",
    r"^Dead-end_pages",
    
    # Language/translation markers  
    r"_in_translation$",
    r"^Interlanguage_",
    r"^Language_articles_",
]

# Compile patterns for efficiency
MAINTENANCE_REGEX = re.compile("|".join(MAINTENANCE_PATTERNS), re.IGNORECASE)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class CleaningStats:
    """Track statistics about the cleaning process."""
    original_nodes: int = 0
    original_edges: int = 0
    original_concepts: int = 0
    original_categories: int = 0
    
    removed_maintenance_categories: int = 0
    removed_orphan_categories: int = 0
    removed_orphan_concepts: int = 0
    removed_weak_roots: int = 0
    removed_cycle_edges: int = 0
    
    final_nodes: int = 0
    final_edges: int = 0
    final_concepts: int = 0
    final_categories: int = 0
    
    cycles_found: int = 0
    max_depth_before: int = 0
    max_depth_after: int = 0
    
    removed_category_examples: list = field(default_factory=list)
    cycle_examples: list = field(default_factory=list)


@dataclass  
class GraphData:
    """Container for graph data."""
    # Node data
    node_ids: list
    original_ids: list
    titles: list
    node_types: list
    
    # Edge data
    edge_src: list
    edge_dst: list
    edge_types: list
    
    # Metadata
    metadata: dict
    
    # Computed after cleaning
    depths: Optional[list] = None
    normalized_depths: Optional[list] = None
    direct_counts: Optional[list] = None
    total_counts: Optional[list] = None
    specificity_scores: Optional[list] = None
    ancestors: Optional[dict] = None


def log(msg: str, level: str = "INFO"):
    """Log a message with timestamp."""
    symbols = {"INFO": "ℹ", "OK": "✓", "WARN": "⚠", "ERROR": "✗", "HEAD": "═", "CLEAN": "🧹"}
    symbol = symbols.get(level, "•")
    print(f"[{time.strftime('%H:%M:%S')}] {symbol} {msg}", file=sys.stderr, flush=True)


def log_header(msg: str):
    """Log a section header."""
    print(f"\n{'=' * 70}", file=sys.stderr)
    log(msg, "HEAD")
    print('=' * 70, file=sys.stderr)


# =============================================================================
# LOADING
# =============================================================================

def load_graph(input_dir: str) -> GraphData:
    """Load graph from parquet files."""
    log_header("LOADING GRAPH")
    
    input_path = Path(input_dir)
    
    # Load nodes
    log(f"Loading nodes from {input_path / 'nodes.parquet'}")
    nodes = pq.read_table(input_path / "nodes.parquet")
    
    node_ids = nodes.column("node_id").to_pylist()
    original_ids = nodes.column("original_id").to_pylist()
    titles = nodes.column("title").to_pylist()
    node_types = nodes.column("node_type").to_pylist()
    
    log(f"  Loaded {len(node_ids):,} nodes")
    
    # Load edges
    log(f"Loading edges from {input_path / 'edges.parquet'}")
    edges = pq.read_table(input_path / "edges.parquet")
    
    edge_src = edges.column("src_id").to_pylist()
    edge_dst = edges.column("dst_id").to_pylist()
    edge_types = edges.column("edge_type").to_pylist()
    
    log(f"  Loaded {len(edge_src):,} edges")
    
    # Load metadata
    log("Loading metadata")
    with open(input_path / "metadata.json") as f:
        metadata = json.load(f)
    
    return GraphData(
        node_ids=node_ids,
        original_ids=original_ids,
        titles=titles,
        node_types=node_types,
        edge_src=edge_src,
        edge_dst=edge_dst,
        edge_types=edge_types,
        metadata=metadata
    )


# =============================================================================
# CLEANING FUNCTIONS
# =============================================================================

def is_maintenance_category(title: str, extra_patterns: Optional[list] = None) -> bool:
    """Check if a category title matches maintenance patterns."""
    if MAINTENANCE_REGEX.search(title):
        return True
    
    if extra_patterns:
        for pattern in extra_patterns:
            if re.search(pattern, title, re.IGNORECASE):
                return True
    
    return False


def remove_maintenance_categories(
    graph: GraphData, 
    stats: CleaningStats,
    extra_patterns: Optional[list] = None,
    keep_patterns: Optional[list] = None
) -> set:
    """
    Identify and mark maintenance categories for removal.
    Returns set of node_ids to remove.
    """
    log_header("REMOVING MAINTENANCE CATEGORIES")
    
    to_remove = set()
    
    for node_id, title, node_type in zip(graph.node_ids, graph.titles, graph.node_types):
        if node_type != "category":
            continue
            
        # Check if should be kept (whitelist)
        if keep_patterns:
            should_keep = False
            for pattern in keep_patterns:
                if re.search(pattern, title, re.IGNORECASE):
                    should_keep = True
                    break
            if should_keep:
                continue
        
        # Check if matches maintenance patterns
        if is_maintenance_category(title, extra_patterns):
            to_remove.add(node_id)
            if len(stats.removed_category_examples) < 20:
                stats.removed_category_examples.append(title)
    
    stats.removed_maintenance_categories = len(to_remove)
    log(f"Identified {len(to_remove):,} maintenance categories for removal", "CLEAN")
    
    if stats.removed_category_examples:
        log("Examples of removed categories:")
        for ex in stats.removed_category_examples[:10]:
            log(f"    • {ex[:60]}")
    
    return to_remove


def build_indices(graph: GraphData, removed_nodes: set) -> dict:
    """Build lookup indices for the graph, excluding removed nodes."""
    log("Building indices...")
    
    valid_nodes = set(graph.node_ids) - removed_nodes
    
    idx = {
        "valid_nodes": valid_nodes,
        "node_to_type": {},
        "node_to_title": {},
        "node_to_idx": {},
        "concept_ids": set(),
        "category_ids": set(),
        "child_to_parents": defaultdict(set),
        "parent_to_children": defaultdict(set),
        "concept_to_categories": defaultdict(set),
        "category_to_concepts": defaultdict(set),
    }
    
    # Build node lookups
    for i, (node_id, title, node_type) in enumerate(zip(
        graph.node_ids, graph.titles, graph.node_types
    )):
        if node_id not in valid_nodes:
            continue
            
        idx["node_to_type"][node_id] = node_type
        idx["node_to_title"][node_id] = title
        idx["node_to_idx"][node_id] = i
        
        if node_type == "concept":
            idx["concept_ids"].add(node_id)
        else:
            idx["category_ids"].add(node_id)
    
    # Build edge lookups (only for valid nodes)
    for src, dst, etype in zip(graph.edge_src, graph.edge_dst, graph.edge_types):
        if src not in valid_nodes or dst not in valid_nodes:
            continue
            
        if etype == "child_of":
            idx["child_to_parents"][src].add(dst)
            idx["parent_to_children"][dst].add(src)
        elif etype == "belongs_to":
            idx["concept_to_categories"][src].add(dst)
            idx["category_to_concepts"][dst].add(src)
    
    log(f"  Valid nodes: {len(valid_nodes):,}")
    log(f"  Categories: {len(idx['category_ids']):,}")
    log(f"  Concepts: {len(idx['concept_ids']):,}")
    
    return idx


def detect_and_break_cycles(idx: dict, stats: CleaningStats) -> set:
    """
    Detect cycles in the category hierarchy and identify edges to remove.
    Uses Tarjan's algorithm to find strongly connected components (SCCs).
    Returns set of (src, dst) edge tuples to remove.
    """
    log_header("DETECTING AND BREAKING CYCLES")
    
    # Tarjan's SCC algorithm
    index_counter = [0]
    stack = []
    lowlinks = {}
    index = {}
    on_stack = {}
    sccs = []
    
    def strongconnect(node):
        index[node] = index_counter[0]
        lowlinks[node] = index_counter[0]
        index_counter[0] += 1
        stack.append(node)
        on_stack[node] = True
        
        for parent in idx["child_to_parents"].get(node, set()):
            if parent not in idx["category_ids"]:
                continue
            if parent not in index:
                strongconnect(parent)
                lowlinks[node] = min(lowlinks[node], lowlinks[parent])
            elif on_stack.get(parent, False):
                lowlinks[node] = min(lowlinks[node], index[parent])
        
        if lowlinks[node] == index[node]:
            scc = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                scc.append(w)
                if w == node:
                    break
            if len(scc) > 1:
                sccs.append(scc)
    
    # Run on all categories
    for cat_id in idx["category_ids"]:
        if cat_id not in index:
            strongconnect(cat_id)
    
    stats.cycles_found = len(sccs)
    
    if not sccs:
        log("No cycles detected!", "OK")
        return set()
    
    log(f"Found {len(sccs)} strongly connected components (cycles)", "WARN")
    
    # For each SCC, identify edges to remove to break the cycle
    # Strategy: Remove edges that go from lower-degree to higher-degree nodes
    # (preserving more important structural edges)
    edges_to_remove = set()
    
    for scc in sccs:
        scc_set = set(scc)
        
        if len(stats.cycle_examples) < 5:
            titles = [idx["node_to_title"].get(n, "?")[:30] for n in scc[:5]]
            stats.cycle_examples.append(f"Cycle of {len(scc)} nodes: {', '.join(titles)}")
        
        # Find all internal edges in this SCC
        internal_edges = []
        for node in scc:
            for parent in idx["child_to_parents"].get(node, set()):
                if parent in scc_set:
                    # Calculate "importance" as number of children
                    parent_children = len(idx["parent_to_children"].get(parent, set()))
                    node_children = len(idx["parent_to_children"].get(node, set()))
                    internal_edges.append((node, parent, parent_children, node_children))
        
        # Sort by parent importance (descending) - remove edges TO less important parents
        internal_edges.sort(key=lambda x: x[2])
        
        # Remove edges until cycle is broken (minimum feedback arc set approximation)
        # For simplicity, remove all but the edge to the most important parent for each node
        nodes_processed = set()
        for node, parent, _, _ in internal_edges:
            if node not in nodes_processed:
                nodes_processed.add(node)
            else:
                edges_to_remove.add((node, parent))
    
    stats.removed_cycle_edges = len(edges_to_remove)
    log(f"Identified {len(edges_to_remove)} edges to remove to break cycles", "CLEAN")
    
    for ex in stats.cycle_examples:
        log(f"    {ex}", "WARN")
    
    return edges_to_remove


def remove_orphans(idx: dict, stats: CleaningStats) -> set:
    """
    Remove orphaned nodes:
    - Categories with no concepts (directly or through descendants) 
    - Concepts with no categories
    Returns set of node_ids to remove.
    """
    log_header("REMOVING ORPHAN NODES")
    
    to_remove = set()
    
    # Find concepts with no categories
    orphan_concepts = []
    for concept_id in idx["concept_ids"]:
        if not idx["concept_to_categories"].get(concept_id):
            orphan_concepts.append(concept_id)
            to_remove.add(concept_id)
    
    stats.removed_orphan_concepts = len(orphan_concepts)
    log(f"Found {len(orphan_concepts)} orphan concepts (no categories)", "CLEAN")
    
    # Find categories with no concepts (even through descendants)
    # BFS from concepts upward to mark reachable categories
    reachable_categories = set()
    
    for concept_id in idx["concept_ids"]:
        if concept_id in to_remove:
            continue
        for cat_id in idx["concept_to_categories"].get(concept_id, set()):
            if cat_id in reachable_categories:
                continue
            # BFS upward
            queue = deque([cat_id])
            while queue:
                current = queue.popleft()
                if current in reachable_categories:
                    continue
                reachable_categories.add(current)
                for parent in idx["child_to_parents"].get(current, set()):
                    if parent not in reachable_categories:
                        queue.append(parent)
    
    orphan_categories = idx["category_ids"] - reachable_categories
    stats.removed_orphan_categories = len(orphan_categories)
    to_remove.update(orphan_categories)
    
    log(f"Found {len(orphan_categories)} orphan categories (no concepts)", "CLEAN")
    
    return to_remove


def prune_weak_roots(idx: dict, stats: CleaningStats, min_concepts: int = 5) -> set:
    """
    Remove root categories that have very few concepts.
    These are often overly specific or maintenance-related categories that slipped through.
    Returns set of node_ids to remove.
    """
    log_header(f"PRUNING WEAK ROOT CATEGORIES (min_concepts={min_concepts})")
    
    # First, compute total concept counts for all categories
    total_counts = defaultdict(int)
    
    # Start from concepts and propagate counts upward
    for concept_id in idx["concept_ids"]:
        visited = set()
        queue = deque(idx["concept_to_categories"].get(concept_id, set()))
        
        while queue:
            cat_id = queue.popleft()
            if cat_id in visited:
                continue
            visited.add(cat_id)
            total_counts[cat_id] += 1
            
            for parent in idx["child_to_parents"].get(cat_id, set()):
                if parent not in visited:
                    queue.append(parent)
    
    # Find root categories (no parents)
    roots = [
        cat_id for cat_id in idx["category_ids"]
        if not idx["child_to_parents"].get(cat_id)
    ]
    
    log(f"Found {len(roots):,} root categories")
    
    # Identify weak roots
    weak_roots = set()
    for root_id in roots:
        count = total_counts.get(root_id, 0)
        if count < min_concepts:
            weak_roots.add(root_id)
    
    stats.removed_weak_roots = len(weak_roots)
    log(f"Identified {len(weak_roots):,} weak root categories for removal", "CLEAN")
    
    # Also remove categories whose only path to root goes through weak roots
    # (they become orphans after weak root removal)
    # This is handled in subsequent orphan removal pass
    
    return weak_roots


def compute_depths_bfs(idx: dict) -> dict:
    """
    Compute depths using BFS from root categories.
    Handles DAG structure correctly by using minimum depth.
    """
    log_header("COMPUTING DEPTHS (BFS)")
    
    depths = {}
    
    # Find roots (categories with no parents)
    roots = [
        cat_id for cat_id in idx["category_ids"]
        if not idx["child_to_parents"].get(cat_id)
    ]
    
    log(f"Starting BFS from {len(roots):,} root categories")
    
    # BFS from roots
    queue = deque()
    for root_id in roots:
        depths[root_id] = 0
        queue.append((root_id, 0))
    
    while queue:
        node_id, depth = queue.popleft()
        
        # Process children (categories)
        for child_id in idx["parent_to_children"].get(node_id, set()):
            child_depth = depth + 1
            if child_id not in depths or depths[child_id] > child_depth:
                depths[child_id] = child_depth
                queue.append((child_id, child_depth))
    
    # Compute concept depths (parent depth + 1)
    for concept_id in idx["concept_ids"]:
        categories = idx["concept_to_categories"].get(concept_id, set())
        if categories:
            min_cat_depth = min(depths.get(c, 0) for c in categories)
            depths[concept_id] = min_cat_depth + 1
        else:
            depths[concept_id] = 0  # Orphan concept
    
    max_depth = max(depths.values()) if depths else 0
    log(f"Computed depths for {len(depths):,} nodes (max_depth={max_depth})")
    
    return depths


def compute_concept_counts(idx: dict) -> tuple[dict, dict]:
    """Compute direct and total concept counts for each category."""
    log("Computing concept counts...")
    
    direct_counts = {}
    total_counts = defaultdict(int)
    
    # Direct counts
    for cat_id in idx["category_ids"]:
        direct_counts[cat_id] = len(idx["category_to_concepts"].get(cat_id, set()))
    
    # Total counts (propagate from concepts upward)
    for concept_id in idx["concept_ids"]:
        visited = set()
        queue = deque(idx["concept_to_categories"].get(concept_id, set()))
        
        while queue:
            cat_id = queue.popleft()
            if cat_id in visited:
                continue
            visited.add(cat_id)
            total_counts[cat_id] += 1
            
            for parent in idx["child_to_parents"].get(cat_id, set()):
                if parent not in visited:
                    queue.append(parent)
    
    # Concepts have counts of 1
    for concept_id in idx["concept_ids"]:
        direct_counts[concept_id] = 0
        total_counts[concept_id] = 1
    
    return direct_counts, dict(total_counts)


def compute_specificity_scores(idx: dict, total_counts: dict) -> dict:
    """Compute specificity scores for all nodes."""
    log("Computing specificity scores...")
    
    scores = {}
    
    # Concepts have specificity 1.0
    for concept_id in idx["concept_ids"]:
        scores[concept_id] = 1.0
    
    # Categories: 1 / log(1 + total_count)
    for cat_id in idx["category_ids"]:
        count = total_counts.get(cat_id, 0)
        if count > 0:
            scores[cat_id] = 1.0 / np.log1p(count)
        else:
            scores[cat_id] = 1.0  # Empty category (will likely be removed)
    
    return scores


def compute_ancestors(idx: dict, depths: dict) -> dict:
    """Compute ancestor list for each concept, sorted by depth (root first)."""
    log("Computing ancestors...")
    
    ancestors = {}
    
    for concept_id in idx["concept_ids"]:
        # BFS upward from concept's categories
        concept_ancestors = set()
        queue = deque(idx["concept_to_categories"].get(concept_id, set()))
        
        while queue:
            cat_id = queue.popleft()
            if cat_id in concept_ancestors:
                continue
            concept_ancestors.add(cat_id)
            
            for parent in idx["child_to_parents"].get(cat_id, set()):
                if parent not in concept_ancestors:
                    queue.append(parent)
        
        # Sort by depth (ascending = root first)
        sorted_ancestors = sorted(
            concept_ancestors,
            key=lambda x: depths.get(x, 0)
        )
        ancestors[concept_id] = sorted_ancestors
    
    return ancestors


# =============================================================================
# MAIN CLEANING PIPELINE
# =============================================================================

def clean_graph(
    input_dir: str,
    output_dir: str,
    min_root_concepts: int = 5,
    extra_remove_patterns: Optional[list] = None,
    keep_patterns: Optional[list] = None,
    dry_run: bool = False
) -> CleaningStats:
    """Main cleaning pipeline."""
    
    stats = CleaningStats()
    
    # Load graph
    graph = load_graph(input_dir)
    
    stats.original_nodes = len(graph.node_ids)
    stats.original_edges = len(graph.edge_src)
    stats.original_concepts = sum(1 for t in graph.node_types if t == "concept")
    stats.original_categories = sum(1 for t in graph.node_types if t == "category")
    stats.max_depth_before = graph.metadata.get("max_depth", 0)
    
    log_header("ORIGINAL GRAPH STATISTICS")
    log(f"Nodes: {stats.original_nodes:,}")
    log(f"Edges: {stats.original_edges:,}")
    log(f"Concepts: {stats.original_concepts:,}")
    log(f"Categories: {stats.original_categories:,}")
    log(f"Max depth: {stats.max_depth_before}")
    
    # Step 1: Remove maintenance categories
    removed_nodes = remove_maintenance_categories(
        graph, stats, extra_remove_patterns, keep_patterns
    )
    
    # Build indices with remaining nodes
    idx = build_indices(graph, removed_nodes)
    
    # Step 2: Detect and break cycles
    cycle_edges = detect_and_break_cycles(idx, stats)
    
    # Remove cycle edges from index
    for src, dst in cycle_edges:
        idx["child_to_parents"][src].discard(dst)
        idx["parent_to_children"][dst].discard(src)
    
    # Step 3: Prune weak roots
    weak_roots = prune_weak_roots(idx, stats, min_root_concepts)
    removed_nodes.update(weak_roots)
    
    # Update category set
    idx["category_ids"] -= weak_roots
    for node_id in weak_roots:
        idx["valid_nodes"].discard(node_id)
    
    # Step 4: Remove orphans (iteratively until stable)
    iteration = 0
    while True:
        iteration += 1
        orphans = remove_orphans(idx, stats)
        
        if not orphans:
            break
        
        removed_nodes.update(orphans)
        idx["valid_nodes"] -= orphans
        idx["concept_ids"] -= orphans
        idx["category_ids"] -= orphans
        
        # Clean up edge indices
        for node_id in orphans:
            if node_id in idx["child_to_parents"]:
                for parent in idx["child_to_parents"][node_id]:
                    idx["parent_to_children"][parent].discard(node_id)
                del idx["child_to_parents"][node_id]
            
            if node_id in idx["parent_to_children"]:
                for child in idx["parent_to_children"][node_id]:
                    idx["child_to_parents"][child].discard(node_id)
                del idx["parent_to_children"][node_id]
            
            if node_id in idx["concept_to_categories"]:
                for cat in idx["concept_to_categories"][node_id]:
                    idx["category_to_concepts"][cat].discard(node_id)
                del idx["concept_to_categories"][node_id]
            
            if node_id in idx["category_to_concepts"]:
                del idx["category_to_concepts"][node_id]
        
        log(f"  Orphan removal iteration {iteration}: removed {len(orphans)} nodes")
        
        if iteration > 100:
            log("Warning: Orphan removal exceeded 100 iterations", "WARN")
            break
    
    # Step 5: Recompute all derived data
    depths = compute_depths_bfs(idx)
    max_depth = max(depths.values()) if depths else 0
    stats.max_depth_after = max_depth
    
    direct_counts, total_counts = compute_concept_counts(idx)
    specificity_scores = compute_specificity_scores(idx, total_counts)
    ancestors = compute_ancestors(idx, depths)
    
    # Compute final statistics
    stats.final_nodes = len(idx["valid_nodes"])
    stats.final_concepts = len(idx["concept_ids"])
    stats.final_categories = len(idx["category_ids"])
    
    # Count final edges
    final_edge_count = 0
    for src in idx["child_to_parents"]:
        final_edge_count += len(idx["child_to_parents"][src])
    for src in idx["concept_to_categories"]:
        final_edge_count += len(idx["concept_to_categories"][src])
    stats.final_edges = final_edge_count
    
    # Print summary
    log_header("CLEANING SUMMARY")
    log(f"Removed maintenance categories: {stats.removed_maintenance_categories:,}")
    log(f"Removed cycle edges: {stats.removed_cycle_edges:,}")
    log(f"Removed weak root categories: {stats.removed_weak_roots:,}")
    log(f"Removed orphan concepts: {stats.removed_orphan_concepts:,}")
    log(f"Removed orphan categories: {stats.removed_orphan_categories:,}")
    log(f"")
    log(f"Final nodes: {stats.final_nodes:,} (was {stats.original_nodes:,}, -{stats.original_nodes - stats.final_nodes:,})")
    log(f"Final edges: {stats.final_edges:,} (was {stats.original_edges:,}, -{stats.original_edges - stats.final_edges:,})")
    log(f"Final concepts: {stats.final_concepts:,}")
    log(f"Final categories: {stats.final_categories:,}")
    log(f"Max depth: {stats.max_depth_after} (was {stats.max_depth_before})")
    
    if dry_run:
        log_header("DRY RUN - NO OUTPUT WRITTEN")
        return stats
    
    # Step 6: Write cleaned graph
    write_cleaned_graph(
        graph, idx, depths, max_depth, direct_counts, total_counts,
        specificity_scores, ancestors, cycle_edges, output_dir, stats
    )
    
    return stats


def write_cleaned_graph(
    graph: GraphData,
    idx: dict,
    depths: dict,
    max_depth: int,
    direct_counts: dict,
    total_counts: dict,
    specificity_scores: dict,
    ancestors: dict,
    removed_edges: set,
    output_dir: str,
    stats: CleaningStats
):
    """Write the cleaned graph to output directory."""
    log_header("WRITING CLEANED GRAPH")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Build node table
    log("Writing nodes.parquet...")
    
    node_ids_out = []
    original_ids_out = []
    titles_out = []
    node_types_out = []
    depths_out = []
    normalized_depths_out = []
    direct_counts_out = []
    total_counts_out = []
    specificity_out = []
    
    # Create mapping from old node_id to new index
    old_to_new_idx = {}
    
    for i, (node_id, original_id, title, node_type) in enumerate(zip(
        graph.node_ids, graph.original_ids, graph.titles, graph.node_types
    )):
        if node_id not in idx["valid_nodes"]:
            continue
        
        old_to_new_idx[node_id] = len(node_ids_out)
        
        node_ids_out.append(node_id)
        original_ids_out.append(original_id)
        titles_out.append(title)
        node_types_out.append(node_type)
        
        depth = depths.get(node_id, 0)
        depths_out.append(depth)
        normalized_depths_out.append(depth / max_depth if max_depth > 0 else 0.0)
        direct_counts_out.append(direct_counts.get(node_id, 0))
        total_counts_out.append(total_counts.get(node_id, 0))
        specificity_out.append(specificity_scores.get(node_id, 1.0))
    
    nodes_table = pa.table({
        "node_id": pa.array(node_ids_out, type=pa.int64()),
        "original_id": pa.array(original_ids_out, type=pa.int64()),
        "title": pa.array(titles_out, type=pa.string()),
        "node_type": pa.array(node_types_out, type=pa.string()),
        "depth": pa.array(depths_out, type=pa.int32()),
        "normalized_depth": pa.array(normalized_depths_out, type=pa.float32()),
        "direct_concept_count": pa.array(direct_counts_out, type=pa.int32()),
        "total_concept_count": pa.array(total_counts_out, type=pa.int32()),
        "specificity_score": pa.array(specificity_out, type=pa.float32()),
    })
    
    pq.write_table(nodes_table, output_path / "nodes.parquet")
    log(f"  Wrote {len(node_ids_out):,} nodes")
    
    # Build edge table
    log("Writing edges.parquet...")
    
    edge_src_out = []
    edge_dst_out = []
    edge_types_out = []
    
    for src, dst, etype in zip(graph.edge_src, graph.edge_dst, graph.edge_types):
        if src not in idx["valid_nodes"] or dst not in idx["valid_nodes"]:
            continue
        if (src, dst) in removed_edges:
            continue
        
        edge_src_out.append(src)
        edge_dst_out.append(dst)
        edge_types_out.append(etype)
    
    edges_table = pa.table({
        "src_id": pa.array(edge_src_out, type=pa.int64()),
        "dst_id": pa.array(edge_dst_out, type=pa.int64()),
        "edge_type": pa.array(edge_types_out, type=pa.string()),
    })
    
    pq.write_table(edges_table, output_path / "edges.parquet")
    log(f"  Wrote {len(edge_src_out):,} edges")
    
    # Build title index
    log("Writing title_index.parquet...")
    
    title_index_table = pa.table({
        "title_lower": pa.array([t.lower() for t in titles_out], type=pa.string()),
        "title": pa.array(titles_out, type=pa.string()),
        "node_id": pa.array(node_ids_out, type=pa.int64()),
        "node_type": pa.array(node_types_out, type=pa.string()),
    })
    
    pq.write_table(title_index_table, output_path / "title_index.parquet")
    log(f"  Wrote {len(titles_out):,} title index entries")
    
    # Build ancestors table
    log("Writing ancestors.parquet...")
    
    ancestor_node_ids = []
    ancestor_lists = []
    
    for concept_id in idx["concept_ids"]:
        ancestor_node_ids.append(concept_id)
        ancestor_lists.append(ancestors.get(concept_id, []))
    
    ancestors_table = pa.table({
        "node_id": pa.array(ancestor_node_ids, type=pa.int64()),
        "ancestors": pa.array(ancestor_lists, type=pa.list_(pa.int64())),
    })
    
    pq.write_table(ancestors_table, output_path / "ancestors.parquet")
    log(f"  Wrote {len(ancestor_node_ids):,} ancestor entries")
    
    # Build depth distribution
    depth_dist = defaultdict(int)
    for d in depths_out:
        depth_dist[d] += 1
    
    # Write metadata
    log("Writing metadata.json...")
    
    num_belongs_to = sum(1 for e in edge_types_out if e == "belongs_to")
    num_child_of = sum(1 for e in edge_types_out if e == "child_of")
    
    metadata = {
        "num_nodes": len(node_ids_out),
        "num_concepts": stats.final_concepts,
        "num_categories": stats.final_categories,
        "num_edges": len(edge_src_out),
        "num_belongs_to_edges": num_belongs_to,
        "num_child_of_edges": num_child_of,
        "max_depth": max_depth,
        "depth_distribution": {str(k): v for k, v in sorted(depth_dist.items())},
        "cleaning_stats": {
            "original_nodes": stats.original_nodes,
            "original_edges": stats.original_edges,
            "removed_maintenance_categories": stats.removed_maintenance_categories,
            "removed_cycle_edges": stats.removed_cycle_edges,
            "removed_weak_roots": stats.removed_weak_roots,
            "removed_orphan_concepts": stats.removed_orphan_concepts,
            "removed_orphan_categories": stats.removed_orphan_categories,
            "cycles_found": stats.cycles_found,
        },
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "clean_knowledge_graph.py",
    }
    
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    log(f"  Wrote metadata.json")
    log_header("CLEANING COMPLETE")
    log(f"Output written to: {output_dir}", "OK")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Clean Wikipedia knowledge graph for semantic use",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic cleaning
    python clean_knowledge_graph.py data/knowledge_graph/ data/knowledge_graph_clean/
    
    # Dry run to see what would be removed
    python clean_knowledge_graph.py data/knowledge_graph/ data/clean/ --dry-run
    
    # Stricter root pruning
    python clean_knowledge_graph.py data/knowledge_graph/ data/clean/ --min-root-concepts 20
    
    # Add custom patterns to remove
    python clean_knowledge_graph.py data/knowledge_graph/ data/clean/ --remove-patterns patterns.txt
"""
    )
    
    parser.add_argument("input_dir", help="Directory containing input knowledge graph")
    parser.add_argument("output_dir", help="Directory for cleaned output graph")
    
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be cleaned without writing output")
    parser.add_argument("--min-root-concepts", type=int, default=5,
                        help="Minimum concepts for a root category to be kept (default: 5)")
    parser.add_argument("--remove-patterns", type=str, default=None,
                        help="File with additional regex patterns to remove (one per line)")
    parser.add_argument("--keep-patterns", type=str, default=None,
                        help="File with regex patterns to keep (one per line, overrides removal)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show detailed progress")
    
    args = parser.parse_args()
    
    # Load custom patterns if provided
    extra_remove = None
    if args.remove_patterns:
        with open(args.remove_patterns) as f:
            extra_remove = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        log(f"Loaded {len(extra_remove)} additional removal patterns")
    
    keep_patterns = None
    if args.keep_patterns:
        with open(args.keep_patterns) as f:
            keep_patterns = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        log(f"Loaded {len(keep_patterns)} keep patterns")
    
    # Run cleaning
    stats = clean_graph(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        min_root_concepts=args.min_root_concepts,
        extra_remove_patterns=extra_remove,
        keep_patterns=keep_patterns,
        dry_run=args.dry_run
    )
    
    # Exit with error if significant issues remain
    if stats.cycles_found > 0 and not args.dry_run:
        log("Warning: Cycles were found and edges removed", "WARN")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())