#!/usr/bin/env python3
"""
Validate the output of build_knowledge_graph.py

This script performs comprehensive checks on the knowledge graph to ensure:
1. Structural integrity (no orphans, valid edges, etc.)
2. Hierarchy correctness (depths are consistent with edges)
3. Score validity (specificity scores follow expected patterns)
4. Data consistency across files
5. No cycles in the category hierarchy
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from typing import Optional, TextIO

import numpy as np
import pyarrow.parquet as pq


# Global output file handle
_output_file: Optional[TextIO] = None


def log(msg: str, level: str = "INFO"):
    """Log a message with timestamp."""
    symbol = {"INFO": "ℹ", "OK": "✓", "WARN": "⚠", "ERROR": "✗", "HEAD": "═"}
    line = f"[{time.strftime('%H:%M:%S')}] {symbol.get(level, '•')} {msg}"
    print(line, file=sys.stderr, flush=True)
    if _output_file is not None:
        _output_file.write(line + "\n")


def log_header(msg: str):
    """Log a section header."""
    separator = f"\n{'=' * 60}"
    print(separator, file=sys.stderr)
    if _output_file is not None:
        _output_file.write(separator + "\n")
    log(msg, "HEAD")
    print('=' * 60, file=sys.stderr)
    if _output_file is not None:
        _output_file.write('=' * 60 + "\n")


class ValidationResult:
    """Tracks validation results."""
    
    def __init__(self):
        self.checks_passed = 0
        self.checks_failed = 0
        self.warnings = 0
        self.errors = []
        self.warnings_list = []
    
    def ok(self, msg: str):
        self.checks_passed += 1
        log(msg, "OK")
    
    def warn(self, msg: str):
        self.warnings += 1
        self.warnings_list.append(msg)
        log(msg, "WARN")
    
    def error(self, msg: str):
        self.checks_failed += 1
        self.errors.append(msg)
        log(msg, "ERROR")
    
    def summary(self):
        log_header("VALIDATION SUMMARY")
        log(f"Checks passed: {self.checks_passed}")
        log(f"Checks failed: {self.checks_failed}")
        log(f"Warnings: {self.warnings}")
        
        if self.errors:
            log("\nErrors:", "ERROR")
            for e in self.errors:
                log(f"  - {e}", "ERROR")
        
        if self.warnings_list:
            log("\nWarnings:", "WARN")
            for w in self.warnings_list:
                log(f"  - {w}", "WARN")
        
        return self.checks_failed == 0


def load_graph(graph_dir: str) -> dict:
    """Load all graph files into memory."""
    
    log(f"Loading graph from {graph_dir}")
    
    data = {}
    
    # Load nodes
    nodes = pq.read_table(f"{graph_dir}/nodes.parquet")
    data["nodes"] = {
        "node_id": nodes.column("node_id").to_pylist(),
        "original_id": nodes.column("original_id").to_pylist(),
        "title": nodes.column("title").to_pylist(),
        "node_type": nodes.column("node_type").to_pylist(),
        "depth": nodes.column("depth").to_pylist(),
        "normalized_depth": nodes.column("normalized_depth").to_pylist(),
        "direct_concept_count": nodes.column("direct_concept_count").to_pylist(),
        "total_concept_count": nodes.column("total_concept_count").to_pylist(),
        "specificity_score": nodes.column("specificity_score").to_pylist(),
    }
    log(f"  Loaded {len(data['nodes']['node_id']):,} nodes")
    
    # Load edges
    edges = pq.read_table(f"{graph_dir}/edges.parquet")
    data["edges"] = {
        "src_id": edges.column("src_id").to_pylist(),
        "dst_id": edges.column("dst_id").to_pylist(),
        "edge_type": edges.column("edge_type").to_pylist(),
    }
    log(f"  Loaded {len(data['edges']['src_id']):,} edges")
    
    # Load title index
    title_index = pq.read_table(f"{graph_dir}/title_index.parquet")
    data["title_index"] = {
        "title_lower": title_index.column("title_lower").to_pylist(),
        "title": title_index.column("title").to_pylist(),
        "node_id": title_index.column("node_id").to_pylist(),
        "node_type": title_index.column("node_type").to_pylist(),
    }
    log(f"  Loaded {len(data['title_index']['node_id']):,} title index entries")
    
    # Load ancestors
    ancestors = pq.read_table(f"{graph_dir}/ancestors.parquet")
    data["ancestors"] = {
        "node_id": ancestors.column("node_id").to_pylist(),
        "ancestors": ancestors.column("ancestors").to_pylist(),
    }
    log(f"  Loaded {len(data['ancestors']['node_id']):,} ancestor entries")
    
    # Load metadata
    with open(f"{graph_dir}/metadata.json") as f:
        data["metadata"] = json.load(f)
    log(f"  Loaded metadata")
    
    return data


def build_index(data: dict) -> dict:
    """Build lookup indices for validation."""
    
    log("Building indices...")
    
    idx = {}
    
    # Node ID -> index
    idx["node_id_to_idx"] = {
        nid: i for i, nid in enumerate(data["nodes"]["node_id"])
    }
    
    # Node ID -> properties
    idx["node_id_to_type"] = {
        nid: data["nodes"]["node_type"][i] 
        for i, nid in enumerate(data["nodes"]["node_id"])
    }
    idx["node_id_to_depth"] = {
        nid: data["nodes"]["depth"][i] 
        for i, nid in enumerate(data["nodes"]["node_id"])
    }
    idx["node_id_to_title"] = {
        nid: data["nodes"]["title"][i] 
        for i, nid in enumerate(data["nodes"]["node_id"])
    }
    idx["node_id_to_total_count"] = {
        nid: data["nodes"]["total_concept_count"][i] 
        for i, nid in enumerate(data["nodes"]["node_id"])
    }
    idx["node_id_to_specificity"] = {
        nid: data["nodes"]["specificity_score"][i] 
        for i, nid in enumerate(data["nodes"]["node_id"])
    }
    
    # All node IDs
    idx["all_node_ids"] = set(data["nodes"]["node_id"])
    idx["concept_ids"] = {
        nid for nid, t in idx["node_id_to_type"].items() if t == "concept"
    }
    idx["category_ids"] = {
        nid for nid, t in idx["node_id_to_type"].items() if t == "category"
    }
    
    # Edge indices
    idx["child_to_parents"] = defaultdict(set)  # category -> parent categories
    idx["parent_to_children"] = defaultdict(set)  # category -> child categories
    idx["concept_to_categories"] = defaultdict(set)  # concept -> categories
    idx["category_to_concepts"] = defaultdict(set)  # category -> concepts
    
    for src, dst, etype in zip(
        data["edges"]["src_id"],
        data["edges"]["dst_id"],
        data["edges"]["edge_type"]
    ):
        if etype == "child_of":
            idx["child_to_parents"][src].add(dst)
            idx["parent_to_children"][dst].add(src)
        elif etype == "belongs_to":
            idx["concept_to_categories"][src].add(dst)
            idx["category_to_concepts"][dst].add(src)
    
    # Ancestors index
    idx["node_to_ancestors"] = {
        nid: ancestors 
        for nid, ancestors in zip(
            data["ancestors"]["node_id"],
            data["ancestors"]["ancestors"]
        )
    }
    
    log(f"  Built indices for {len(idx['all_node_ids']):,} nodes")
    
    return idx


def validate_basic_structure(data: dict, idx: dict, result: ValidationResult):
    """Validate basic structural properties."""
    
    log_header("BASIC STRUCTURE VALIDATION")
    
    # Check node counts match metadata
    num_concepts = sum(1 for t in data["nodes"]["node_type"] if t == "concept")
    num_categories = sum(1 for t in data["nodes"]["node_type"] if t == "category")
    
    if num_concepts == data["metadata"]["num_concepts"]:
        result.ok(f"Concept count matches metadata: {num_concepts:,}")
    else:
        result.error(f"Concept count mismatch: {num_concepts} vs metadata {data['metadata']['num_concepts']}")
    
    if num_categories == data["metadata"]["num_categories"]:
        result.ok(f"Category count matches metadata: {num_categories:,}")
    else:
        result.error(f"Category count mismatch: {num_categories} vs metadata {data['metadata']['num_categories']}")
    
    # Check edge counts
    num_belongs_to = sum(1 for e in data["edges"]["edge_type"] if e == "belongs_to")
    num_child_of = sum(1 for e in data["edges"]["edge_type"] if e == "child_of")
    
    if num_belongs_to == data["metadata"]["num_belongs_to_edges"]:
        result.ok(f"belongs_to edge count matches metadata: {num_belongs_to:,}")
    else:
        result.error(f"belongs_to edge count mismatch: {num_belongs_to} vs metadata {data['metadata']['num_belongs_to_edges']}")
    
    if num_child_of == data["metadata"]["num_child_of_edges"]:
        result.ok(f"child_of edge count matches metadata: {num_child_of:,}")
    else:
        result.error(f"child_of edge count mismatch: {num_child_of} vs metadata {data['metadata']['num_child_of_edges']}")
    
    # Check for duplicate node IDs
    node_ids = data["nodes"]["node_id"]
    if len(node_ids) == len(set(node_ids)):
        result.ok("No duplicate node IDs")
    else:
        duplicates = len(node_ids) - len(set(node_ids))
        result.error(f"Found {duplicates} duplicate node IDs")
    
    # Check all edges reference valid nodes
    all_node_ids = idx["all_node_ids"]
    invalid_src = sum(1 for s in data["edges"]["src_id"] if s not in all_node_ids)
    invalid_dst = sum(1 for d in data["edges"]["dst_id"] if d not in all_node_ids)
    
    if invalid_src == 0:
        result.ok("All edge sources are valid nodes")
    else:
        result.error(f"Found {invalid_src} edges with invalid source node IDs")
    
    if invalid_dst == 0:
        result.ok("All edge destinations are valid nodes")
    else:
        result.error(f"Found {invalid_dst} edges with invalid destination node IDs")
    
    # Check title index consistency
    if len(data["title_index"]["node_id"]) == len(data["nodes"]["node_id"]):
        result.ok("Title index has same number of entries as nodes")
    else:
        result.error(f"Title index size mismatch: {len(data['title_index']['node_id'])} vs {len(data['nodes']['node_id'])} nodes")
    
    # Check ancestors table covers all concepts
    ancestor_concept_ids = set(data["ancestors"]["node_id"])
    if ancestor_concept_ids == idx["concept_ids"]:
        result.ok("Ancestors table covers all concepts")
    else:
        missing = idx["concept_ids"] - ancestor_concept_ids
        extra = ancestor_concept_ids - idx["concept_ids"]
        if missing:
            result.error(f"Ancestors table missing {len(missing)} concepts")
        if extra:
            result.error(f"Ancestors table has {len(extra)} extra entries")


def validate_edge_types(data: dict, idx: dict, result: ValidationResult):
    """Validate edge type constraints."""
    
    log_header("EDGE TYPE VALIDATION")
    
    # belongs_to: concept -> category
    wrong_belongs_to = 0
    for src, dst, etype in zip(
        data["edges"]["src_id"],
        data["edges"]["dst_id"],
        data["edges"]["edge_type"]
    ):
        if etype == "belongs_to":
            src_type = idx["node_id_to_type"].get(src)
            dst_type = idx["node_id_to_type"].get(dst)
            if src_type != "concept" or dst_type != "category":
                wrong_belongs_to += 1
    
    if wrong_belongs_to == 0:
        result.ok("All belongs_to edges are concept -> category")
    else:
        result.error(f"Found {wrong_belongs_to} belongs_to edges with wrong node types")
    
    # child_of: category -> category
    wrong_child_of = 0
    for src, dst, etype in zip(
        data["edges"]["src_id"],
        data["edges"]["dst_id"],
        data["edges"]["edge_type"]
    ):
        if etype == "child_of":
            src_type = idx["node_id_to_type"].get(src)
            dst_type = idx["node_id_to_type"].get(dst)
            if src_type != "category" or dst_type != "category":
                wrong_child_of += 1
    
    if wrong_child_of == 0:
        result.ok("All child_of edges are category -> category")
    else:
        result.error(f"Found {wrong_child_of} child_of edges with wrong node types")
    
    # No self-loops
    self_loops = sum(1 for s, d in zip(data["edges"]["src_id"], data["edges"]["dst_id"]) if s == d)
    if self_loops == 0:
        result.ok("No self-loop edges")
    else:
        result.error(f"Found {self_loops} self-loop edges")


def validate_hierarchy_depths(data: dict, idx: dict, result: ValidationResult):
    """Validate that depths are consistent with hierarchy."""
    
    log_header("HIERARCHY DEPTH VALIDATION")
    
    # Find root categories (depth 0, no parents)
    roots = [
        nid for nid in idx["category_ids"]
        if idx["node_id_to_depth"][nid] == 0
    ]
    
    roots_without_parents = [
        nid for nid in roots
        if len(idx["child_to_parents"].get(nid, set())) == 0
    ]
    
    if len(roots) == len(roots_without_parents):
        result.ok(f"All {len(roots)} depth-0 categories have no parents")
    else:
        with_parents = len(roots) - len(roots_without_parents)
        result.error(f"{with_parents} depth-0 categories have parents")
    
    # Check parent depth < child depth for all child_of edges
    depth_violations = 0
    depth_violation_examples = []
    
    for child_id, parents in idx["child_to_parents"].items():
        child_depth = idx["node_id_to_depth"].get(child_id, -1)
        for parent_id in parents:
            parent_depth = idx["node_id_to_depth"].get(parent_id, -1)
            if parent_depth >= child_depth:
                depth_violations += 1
                if len(depth_violation_examples) < 5:
                    child_title = idx["node_id_to_title"].get(child_id, "?")[:30]
                    parent_title = idx["node_id_to_title"].get(parent_id, "?")[:30]
                    depth_violation_examples.append(
                        f"{child_title}(d={child_depth}) -> {parent_title}(d={parent_depth})"
                    )
    
    if depth_violations == 0:
        result.ok("All child_of edges have parent_depth < child_depth")
    else:
        result.warn(f"Found {depth_violations} edges where parent_depth >= child_depth (DAG structure)")
        for ex in depth_violation_examples:
            log(f"    Example: {ex}", "WARN")
    
    # Verify concept depths are parent_depth + 1
    concept_depth_errors = 0
    for concept_id in idx["concept_ids"]:
        concept_depth = idx["node_id_to_depth"].get(concept_id, -1)
        categories = idx["concept_to_categories"].get(concept_id, set())
        
        if categories:
            min_cat_depth = min(idx["node_id_to_depth"].get(c, 999) for c in categories)
            expected_depth = min_cat_depth + 1
            if concept_depth != expected_depth:
                concept_depth_errors += 1
    
    if concept_depth_errors == 0:
        result.ok("All concept depths = min(parent category depths) + 1")
    else:
        result.error(f"Found {concept_depth_errors} concepts with incorrect depth")
    
    # Check normalized depth
    max_depth = data["metadata"]["max_depth"]
    norm_depth_errors = 0
    
    for i, nid in enumerate(data["nodes"]["node_id"]):
        depth = data["nodes"]["depth"][i]
        norm_depth = data["nodes"]["normalized_depth"][i]
        expected_norm = depth / max_depth if max_depth > 0 else 0.0
        
        if abs(norm_depth - expected_norm) > 1e-6:
            norm_depth_errors += 1
    
    if norm_depth_errors == 0:
        result.ok("All normalized_depth values are correct")
    else:
        result.error(f"Found {norm_depth_errors} incorrect normalized_depth values")


def validate_no_cycles(data: dict, idx: dict, result: ValidationResult):
    """Validate that category hierarchy has no cycles."""
    
    log_header("CYCLE DETECTION")
    
    # DFS-based cycle detection
    visited = set()
    rec_stack = set()
    has_cycle = False
    cycle_node = None
    
    def dfs(node):
        nonlocal has_cycle, cycle_node
        
        visited.add(node)
        rec_stack.add(node)
        
        for parent in idx["child_to_parents"].get(node, set()):
            if parent not in visited:
                if dfs(parent):
                    return True
            elif parent in rec_stack:
                has_cycle = True
                cycle_node = parent
                return True
        
        rec_stack.remove(node)
        return False
    
    for cat_id in idx["category_ids"]:
        if cat_id not in visited:
            if dfs(cat_id):
                break
    
    if not has_cycle:
        result.ok("No cycles detected in category hierarchy")
    else:
        title = idx["node_id_to_title"].get(cycle_node, "?")
        result.error(f"Cycle detected involving category: {title}")


def validate_specificity_scores(data: dict, idx: dict, result: ValidationResult):
    """Validate specificity scores follow expected patterns."""
    
    log_header("SPECIFICITY SCORE VALIDATION")
    
    # All concepts should have specificity = 1.0
    concept_spec_errors = 0
    for nid in idx["concept_ids"]:
        spec = idx["node_id_to_specificity"].get(nid, 0)
        if abs(spec - 1.0) > 1e-6:
            concept_spec_errors += 1
    
    if concept_spec_errors == 0:
        result.ok("All concepts have specificity_score = 1.0")
    else:
        result.error(f"Found {concept_spec_errors} concepts with specificity != 1.0")
    
    # Category specificity should be 1/log(total_count)
    category_spec_errors = 0
    for nid in idx["category_ids"]:
        spec = idx["node_id_to_specificity"].get(nid, 0)
        total_count = idx["node_id_to_total_count"].get(nid, 0)
        
        if total_count > 0:
            expected = 1.0 / np.log1p(total_count)
        else:
            expected = 1.0
        
        if abs(spec - expected) > 1e-6:
            category_spec_errors += 1
    
    if category_spec_errors == 0:
        result.ok("All category specificity scores match formula")
    else:
        result.error(f"Found {category_spec_errors} categories with incorrect specificity")
    
    # Specificity should be inversely correlated with total_count
    cat_specs = []
    cat_counts = []
    for nid in idx["category_ids"]:
        cat_specs.append(idx["node_id_to_specificity"].get(nid, 0))
        cat_counts.append(idx["node_id_to_total_count"].get(nid, 0))
    
    if len(cat_specs) > 10:
        correlation = np.corrcoef(cat_specs, cat_counts)[0, 1]
        if correlation < 0:
            result.ok(f"Specificity negatively correlated with total_count (r={correlation:.3f})")
        else:
            result.warn(f"Unexpected positive correlation between specificity and count (r={correlation:.3f})")
    
    # Check score ranges
    all_specs = data["nodes"]["specificity_score"]
    min_spec = min(all_specs)
    max_spec = max(all_specs)
    
    if min_spec > 0:
        result.ok(f"All specificity scores are positive (min={min_spec:.4f})")
    else:
        result.error(f"Found non-positive specificity scores (min={min_spec})")
    
    if max_spec <= 1.0:
        result.ok(f"All specificity scores are <= 1.0 (max={max_spec:.4f})")
    else:
        result.warn(f"Some specificity scores > 1.0 (max={max_spec:.4f})")


def validate_ancestors(data: dict, idx: dict, result: ValidationResult):
    """Validate ancestor lists are correct."""
    
    log_header("ANCESTOR VALIDATION")
    
    # Sample check: verify ancestors for some concepts
    sample_size = min(1000, len(idx["concept_ids"]))
    sampled_concepts = list(idx["concept_ids"])[:sample_size]
    
    ancestor_errors = 0
    ordering_errors = 0
    
    for concept_id in sampled_concepts:
        stored_ancestors = set(idx["node_to_ancestors"].get(concept_id, []))
        
        # Compute expected ancestors by traversing hierarchy
        expected_ancestors = set()
        direct_cats = idx["concept_to_categories"].get(concept_id, set())
        expected_ancestors.update(direct_cats)
        
        queue = list(direct_cats)
        while queue:
            cat = queue.pop(0)
            parents = idx["child_to_parents"].get(cat, set())
            for p in parents:
                if p not in expected_ancestors:
                    expected_ancestors.add(p)
                    queue.append(p)
        
        if stored_ancestors != expected_ancestors:
            ancestor_errors += 1
        
        # Check ordering (should be sorted by depth, root first)
        ancestor_list = idx["node_to_ancestors"].get(concept_id, [])
        if len(ancestor_list) > 1:
            depths = [idx["node_id_to_depth"].get(a, 0) for a in ancestor_list]
            if depths != sorted(depths):
                ordering_errors += 1
    
    if ancestor_errors == 0:
        result.ok(f"Ancestor lists are correct (sampled {sample_size} concepts)")
    else:
        result.error(f"Found {ancestor_errors}/{sample_size} concepts with incorrect ancestors")
    
    if ordering_errors == 0:
        result.ok("Ancestor lists are sorted by depth (root first)")
    else:
        result.warn(f"Found {ordering_errors}/{sample_size} concepts with unsorted ancestors")


def validate_concept_coverage(data: dict, idx: dict, result: ValidationResult):
    """Validate concept-category relationships."""
    
    log_header("CONCEPT COVERAGE VALIDATION")
    
    # Every concept should have at least one category
    orphan_concepts = [
        nid for nid in idx["concept_ids"]
        if len(idx["concept_to_categories"].get(nid, set())) == 0
    ]
    
    if len(orphan_concepts) == 0:
        result.ok("All concepts belong to at least one category")
    else:
        result.warn(f"Found {len(orphan_concepts)} concepts with no categories")
        for nid in orphan_concepts[:5]:
            title = idx["node_id_to_title"].get(nid, "?")
            log(f"    Example: {title}", "WARN")
    
    # Check direct_concept_count matches actual edges
    count_errors = 0
    for cat_id in idx["category_ids"]:
        actual_count = len(idx["category_to_concepts"].get(cat_id, set()))
        stored_count = data["nodes"]["direct_concept_count"][
            idx["node_id_to_idx"][cat_id]
        ]
        if actual_count != stored_count:
            count_errors += 1
    
    if count_errors == 0:
        result.ok("All direct_concept_count values match edge counts")
    else:
        result.error(f"Found {count_errors} categories with incorrect direct_concept_count")


def validate_title_index(data: dict, idx: dict, result: ValidationResult):
    """Validate title index consistency."""
    
    log_header("TITLE INDEX VALIDATION")
    
    # Check lowercase consistency
    case_errors = 0
    for title, title_lower in zip(
        data["title_index"]["title"],
        data["title_index"]["title_lower"]
    ):
        if title.lower() != title_lower:
            case_errors += 1
    
    if case_errors == 0:
        result.ok("All title_lower values are correct lowercase")
    else:
        result.error(f"Found {case_errors} title_lower mismatches")
    
    # Check node_id consistency
    id_errors = 0
    for i, nid in enumerate(data["title_index"]["node_id"]):
        expected_title = idx["node_id_to_title"].get(nid)
        actual_title = data["title_index"]["title"][i]
        if expected_title != actual_title:
            id_errors += 1
    
    if id_errors == 0:
        result.ok("Title index node_ids match node titles")
    else:
        result.error(f"Found {id_errors} title index inconsistencies")
    
    # Check for duplicate titles (may be valid but worth noting)
    titles = data["title_index"]["title_lower"]
    unique_titles = len(set(titles))
    if unique_titles == len(titles):
        result.ok("All titles are unique")
    else:
        duplicates = len(titles) - unique_titles
        result.warn(f"Found {duplicates} duplicate titles (may be valid for different node types)")


def validate_statistics(data: dict, idx: dict, result: ValidationResult):
    """Validate statistical properties of the graph."""
    
    log_header("STATISTICAL VALIDATION")
    
    # Depth distribution should match metadata
    actual_dist = defaultdict(int)
    for d in data["nodes"]["depth"]:
        actual_dist[d] += 1
    
    metadata_dist = data["metadata"].get("depth_distribution", {})
    
    dist_match = True
    for d, count in actual_dist.items():
        meta_count = metadata_dist.get(str(d), 0)
        if count != meta_count:
            dist_match = False
            break
    
    if dist_match:
        result.ok("Depth distribution matches metadata")
    else:
        result.error("Depth distribution mismatch with metadata")
    
    # Log some statistics
    log(f"\nGraph Statistics:")
    log(f"  Total nodes: {len(data['nodes']['node_id']):,}")
    log(f"  Concepts: {len(idx['concept_ids']):,}")
    log(f"  Categories: {len(idx['category_ids']):,}")
    log(f"  Total edges: {len(data['edges']['src_id']):,}")
    log(f"  Max depth: {data['metadata']['max_depth']}")
    
    # Avg categories per concept
    cats_per_concept = [
        len(idx["concept_to_categories"].get(c, set()))
        for c in idx["concept_ids"]
    ]
    log(f"  Avg categories per concept: {np.mean(cats_per_concept):.2f}")
    
    # Avg children per category
    children_per_cat = [
        len(idx["parent_to_children"].get(c, set()))
        for c in idx["category_ids"]
    ]
    log(f"  Avg children per category: {np.mean(children_per_cat):.2f}")
    
    # Specificity score distribution
    specs = data["nodes"]["specificity_score"]
    log(f"  Specificity score range: [{min(specs):.4f}, {max(specs):.4f}]")
    log(f"  Specificity score mean: {np.mean(specs):.4f}")


def find_node_by_title(title: str, idx: dict, exact: bool = False) -> Optional[int]:
    """Find a node ID by title (case-insensitive)."""
    title_lower = title.lower()
    
    for nid, node_title in idx["node_id_to_title"].items():
        if exact:
            if node_title.lower() == title_lower:
                return nid
        else:
            if title_lower in node_title.lower():
                return nid
    return None


def get_hierarchy_path(node_id: int, idx: dict, max_depth: int = 10) -> list[list[dict]]:
    """
    Get all paths from a node to root categories.
    
    Returns list of paths, where each path is a list of dicts with node info.
    """
    node_type = idx["node_id_to_type"].get(node_id)
    
    if node_type == "concept":
        # Start from direct categories
        direct_cats = list(idx["concept_to_categories"].get(node_id, set()))
        if not direct_cats:
            return []
        
        all_paths = []
        for cat_id in direct_cats:
            cat_paths = get_hierarchy_path(cat_id, idx, max_depth)
            if cat_paths:
                for path in cat_paths:
                    all_paths.append(path)
            else:
                # Category is a root
                all_paths.append([{
                    "node_id": cat_id,
                    "title": idx["node_id_to_title"].get(cat_id, "?"),
                    "depth": idx["node_id_to_depth"].get(cat_id, 0),
                    "type": "category"
                }])
        return all_paths
    
    elif node_type == "category":
        # BFS to find all paths to roots
        paths = []
        
        def dfs_paths(current_id: int, current_path: list, visited: set, depth: int):
            if depth > max_depth:
                return
            
            parents = idx["child_to_parents"].get(current_id, set())
            
            if not parents:
                # Reached a root
                paths.append(current_path.copy())
                return
            
            for parent_id in parents:
                if parent_id in visited:
                    continue
                
                parent_info = {
                    "node_id": parent_id,
                    "title": idx["node_id_to_title"].get(parent_id, "?"),
                    "depth": idx["node_id_to_depth"].get(parent_id, 0),
                    "type": "category"
                }
                
                new_visited = visited | {parent_id}
                dfs_paths(parent_id, current_path + [parent_info], new_visited, depth + 1)
        
        start_info = {
            "node_id": node_id,
            "title": idx["node_id_to_title"].get(node_id, "?"),
            "depth": idx["node_id_to_depth"].get(node_id, 0),
            "type": "category"
        }
        dfs_paths(node_id, [start_info], {node_id}, 0)
        
        return paths
    
    return []


def print_hierarchy_tree(node_id: int, idx: dict, data: dict, max_paths: int = 5):
    """Print a visual representation of a node's hierarchy."""
    
    node_title = idx["node_id_to_title"].get(node_id, "?")
    node_type = idx["node_id_to_type"].get(node_id, "?")
    node_depth = idx["node_id_to_depth"].get(node_id, 0)
    node_spec = idx["node_id_to_specificity"].get(node_id, 0)
    
    log(f"\n{'─' * 50}")
    log(f"Node: {node_title}")
    log(f"  ID: {node_id}")
    log(f"  Type: {node_type}")
    log(f"  Depth: {node_depth}")
    log(f"  Specificity: {node_spec:.4f}")
    
    if node_type == "concept":
        # Show direct categories
        direct_cats = idx["concept_to_categories"].get(node_id, set())
        log(f"  Direct categories ({len(direct_cats)}):")
        for cat_id in list(direct_cats)[:10]:
            cat_title = idx["node_id_to_title"].get(cat_id, "?")
            cat_depth = idx["node_id_to_depth"].get(cat_id, 0)
            log(f"    • {cat_title} (depth={cat_depth})")
        if len(direct_cats) > 10:
            log(f"    ... and {len(direct_cats) - 10} more")
    
    elif node_type == "category":
        # Show direct concepts
        direct_concepts = idx["category_to_concepts"].get(node_id, set())
        total_count = idx["node_id_to_total_count"].get(node_id, 0)
        log(f"  Direct concepts: {len(direct_concepts)}")
        log(f"  Total concepts (including descendants): {total_count}")
        
        # Show children categories
        children = idx["parent_to_children"].get(node_id, set())
        if children:
            log(f"  Child categories ({len(children)}):")
            for child_id in list(children)[:5]:
                child_title = idx["node_id_to_title"].get(child_id, "?")
                log(f"    ↓ {child_title}")
            if len(children) > 5:
                log(f"    ... and {len(children) - 5} more")
    
    # Get hierarchy paths
    paths = get_hierarchy_path(node_id, idx)
    
    if not paths:
        log("  Hierarchy: (no paths to root)")
        return
    
    # Deduplicate and limit paths
    unique_paths = []
    seen_roots = set()
    for path in paths:
        if path:
            root_id = path[-1]["node_id"]
            if root_id not in seen_roots:
                seen_roots.add(root_id)
                unique_paths.append(path)
    
    log(f"\n  Hierarchy paths to root ({len(unique_paths)} unique roots):")
    
    for i, path in enumerate(unique_paths[:max_paths]):
        # Reverse to show root -> leaf
        path_reversed = list(reversed(path))
        
        log(f"\n  Path {i + 1}:")
        for j, node in enumerate(path_reversed):
            indent = "    " + "  " * j
            arrow = "└─" if j == len(path_reversed) - 1 else "├─"
            depth_str = f"d={node['depth']}"
            log(f"{indent}{arrow} {node['title'][:40]} ({depth_str})")
        
        # Add the original node at the end if it's a concept
        if node_type == "concept":
            indent = "    " + "  " * len(path_reversed)
            log(f"{indent}└─ [{node_title}] (d={node_depth}) ← CONCEPT")
    
    if len(unique_paths) > max_paths:
        log(f"\n  ... and {len(unique_paths) - max_paths} more paths")


def visual_inspection(data: dict, idx: dict, result: ValidationResult, sample_concepts: list[str]):
    """Perform visual inspection of sample concept hierarchies."""
    
    log_header("VISUAL HIERARCHY INSPECTION")
    
    log("This section displays hierarchy structures for manual verification.")
    log("Check that the paths make semantic sense (e.g., Dog -> Mammals -> Animals)")
    
    found_count = 0
    not_found = []
    
    for concept_name in sample_concepts:
        # Try exact match first
        node_id = find_node_by_title(concept_name, idx, exact=True)
        
        # Fall back to partial match
        if node_id is None:
            node_id = find_node_by_title(concept_name, idx, exact=False)
        
        if node_id is not None:
            found_count += 1
            print_hierarchy_tree(node_id, idx, data)
        else:
            not_found.append(concept_name)
    
    if not_found:
        log(f"\n{'─' * 50}")
        log(f"Concepts not found in graph ({len(not_found)}):")
        for name in not_found:
            log(f"  • {name}")
    
    log(f"\n{'─' * 50}")
    log(f"Found {found_count}/{len(sample_concepts)} sample concepts")
    
    if found_count > 0:
        result.ok(f"Visual inspection completed for {found_count} concepts")
    else:
        result.warn("No sample concepts found for visual inspection")


def inspect_depth_samples(data: dict, idx: dict, result: ValidationResult):
    """Show sample nodes at each depth level for verification."""
    
    log_header("DEPTH LEVEL SAMPLES")
    
    max_depth = data["metadata"]["max_depth"]
    
    # Group nodes by depth
    nodes_by_depth = defaultdict(list)
    for nid in idx["all_node_ids"]:
        depth = idx["node_id_to_depth"].get(nid, 0)
        node_type = idx["node_id_to_type"].get(nid, "?")
        title = idx["node_id_to_title"].get(nid, "?")
        nodes_by_depth[depth].append((nid, node_type, title))
    
    log("Sample nodes at each depth level:\n")
    
    for depth in range(max_depth + 1):
        nodes = nodes_by_depth[depth]
        
        # Separate concepts and categories
        concepts = [(nid, t) for nid, nt, t in nodes if nt == "concept"]
        categories = [(nid, t) for nid, nt, t in nodes if nt == "category"]
        
        log(f"Depth {depth}: {len(categories)} categories, {len(concepts)} concepts")
        
        # Show sample categories
        if categories:
            log(f"  Categories:")
            for nid, title in categories[:3]:
                total = idx["node_id_to_total_count"].get(nid, 0)
                log(f"    • {title[:45]:45s} (total_count={total:,})")
        
        # Show sample concepts
        if concepts:
            log(f"  Concepts:")
            for nid, title in concepts[:3]:
                log(f"    • {title[:45]}")
        
        log("")
    
    result.ok("Depth level samples displayed for inspection")


def inspect_root_categories(data: dict, idx: dict, result: ValidationResult):
    """Inspect root categories (depth 0) for semantic validity."""
    
    log_header("ROOT CATEGORY INSPECTION")
    
    roots = [
        nid for nid in idx["category_ids"]
        if idx["node_id_to_depth"].get(nid) == 0
    ]
    
    log(f"Found {len(roots)} root categories\n")
    
    # Sort by total concept count
    root_info = []
    for nid in roots:
        title = idx["node_id_to_title"].get(nid, "?")
        total = idx["node_id_to_total_count"].get(nid, 0)
        num_children = len(idx["parent_to_children"].get(nid, set()))
        root_info.append((nid, title, total, num_children))
    
    root_info.sort(key=lambda x: -x[2])  # Sort by total count descending
    
    log("Top 20 root categories by total concept count:")
    log(f"{'Title':<40} {'Total':>10} {'Children':>10}")
    log("-" * 62)
    
    for nid, title, total, num_children in root_info[:20]:
        log(f"{title[:40]:<40} {total:>10,} {num_children:>10}")
    
    if len(root_info) > 20:
        log(f"\n... and {len(root_info) - 20} more root categories")
    
    # Check for suspicious roots (very small)
    small_roots = [(nid, t, c) for nid, t, c, _ in root_info if c < 10]
    if small_roots:
        log(f"\nWarning: {len(small_roots)} root categories have < 10 total concepts:")
        for nid, title, count in small_roots[:10]:
            log(f"  • {title[:40]} ({count} concepts)")
        result.warn(f"{len(small_roots)} root categories have very few concepts")
    else:
        result.ok("All root categories have reasonable concept counts")


def inspect_largest_categories(data: dict, idx: dict, result: ValidationResult):
    """Inspect the largest categories for semantic validity."""
    
    log_header("LARGEST CATEGORY INSPECTION")
    
    # Get all categories with their counts
    cat_info = []
    for nid in idx["category_ids"]:
        title = idx["node_id_to_title"].get(nid, "?")
        total = idx["node_id_to_total_count"].get(nid, 0)
        direct = len(idx["category_to_concepts"].get(nid, set()))
        depth = idx["node_id_to_depth"].get(nid, 0)
        spec = idx["node_id_to_specificity"].get(nid, 0)
        cat_info.append((nid, title, total, direct, depth, spec))
    
    cat_info.sort(key=lambda x: -x[2])  # Sort by total count
    
    log("Top 20 categories by total concept count:")
    log(f"{'Title':<35} {'Total':>8} {'Direct':>8} {'Depth':>6} {'Spec':>8}")
    log("-" * 70)
    
    for nid, title, total, direct, depth, spec in cat_info[:20]:
        log(f"{title[:35]:<35} {total:>8,} {direct:>8,} {depth:>6} {spec:>8.4f}")
    
    # Verify that larger categories have lower specificity
    top_20_specs = [spec for _, _, _, _, _, spec in cat_info[:20]]
    bottom_20_specs = [spec for _, _, _, _, _, spec in cat_info[-20:] if cat_info[-20:]]
    
    if np.mean(top_20_specs) < np.mean(bottom_20_specs):
        result.ok("Larger categories have lower specificity scores (as expected)")
    else:
        result.warn("Specificity scores don't follow expected pattern with category size")


def main():
    global _output_file
    
    parser = argparse.ArgumentParser(
        description="Validate knowledge graph output from build_knowledge_graph.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("graph_dir", help="Directory containing knowledge graph files")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output file path for validation report (default: <graph_dir>/validation_report.txt)")
    parser.add_argument("--quick", action="store_true", 
                        help="Skip expensive validations (cycle detection, full ancestor check)")
    parser.add_argument("--no-inspect", action="store_true",
                        help="Skip visual inspection sections")
    parser.add_argument("--sample-concepts", type=str, nargs="+",
                        default=[
                            # Animals
                            "Dog", "Golden_Retriever", "Cat", "Lion", "Eagle",
                            # Technology  
                            "Smartphone", "iPhone", "Computer", "Tesla_Model_S",
                            # Places
                            "Paris", "Eiffel_Tower", "Mount_Everest",
                            # Objects
                            "Chair", "Piano", "Bicycle",
                            # Food
                            "Pizza", "Apple", "Coffee",
                            # People (may not be in visual nodes)
                            "Albert_Einstein",
                            # Abstract (should NOT be in visual nodes)
                            "Democracy", "Love",
                        ],
                        help="Concept names to inspect hierarchy for")
    
    args = parser.parse_args()
    
    # Set up output file
    output_path = args.output
    if output_path is None:
        output_path = f"{args.graph_dir}/validation_report.txt"
    
    _output_file = open(output_path, "w", encoding="utf-8")
    
    # Write header
    header = f"""Knowledge Graph Validation Report
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
Graph directory: {args.graph_dir}
{'=' * 60}
"""
    _output_file.write(header)
    print(header, file=sys.stderr)
    
    result = ValidationResult()
    
    try:
        # Load data
        data = load_graph(args.graph_dir)
        idx = build_index(data)
        
        # Run validations
        validate_basic_structure(data, idx, result)
        validate_edge_types(data, idx, result)
        validate_hierarchy_depths(data, idx, result)
        
        if not args.quick:
            validate_no_cycles(data, idx, result)
        else:
            log("Skipping cycle detection (--quick mode)")
        
        validate_specificity_scores(data, idx, result)
        validate_ancestors(data, idx, result)
        validate_concept_coverage(data, idx, result)
        validate_title_index(data, idx, result)
        validate_statistics(data, idx, result)
        
        # Visual inspection
        if not args.no_inspect:
            inspect_depth_samples(data, idx, result)
            inspect_root_categories(data, idx, result)
            inspect_largest_categories(data, idx, result)
            visual_inspection(data, idx, result, args.sample_concepts)
        else:
            log("Skipping visual inspection (--no-inspect mode)")
        
    except FileNotFoundError as e:
        result.error(f"Missing file: {e}")
    except Exception as e:
        result.error(f"Unexpected error: {e}")
        _output_file.close()
        raise
    
    # Print summary
    success = result.summary()
    
    # Write final message and close output file
    _output_file.write(f"\n\nReport saved to: {output_path}\n")
    _output_file.close()
    _output_file = None  # Prevent further writes
    
    print(f"\nReport saved to: {output_path}", file=sys.stderr)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()