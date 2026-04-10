"""Add category info to filtered nodes and apply category-based filtering.

Features:
- Multiprocessing for parallel category lookups
- Progress bars with tqdm
- Chunked processing for memory efficiency
- Detailed statistics reporting
"""

import argparse
import sys
import time
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


# ============================================================================
# Category rule definitions
# ============================================================================

VISUAL_CATEGORIES = {
    # Living things
    "Living_people", "Animals", "Plants", "Fungi", "Bacteria",
    "Mammals", "Birds", "Fish", "Reptiles", "Amphibians", "Insects",
    "Dogs", "Cats", "Horses",
    
    # Man-made objects
    "Buildings_and_structures", "Vehicles", "Aircraft", "Ships", "Automobiles",
    "Furniture", "Clothing", "Tools", "Weapons", "Machines",
    "Foods", "Beverages", "Dishes",
    
    # Places
    "Cities", "Countries", "Mountains", "Rivers", "Lakes", "Islands",
    "Landforms", "Parks", "Beaches",
    
    # Art and media
    "Paintings", "Sculptures", "Photographs", "Films", "Album_covers",
    
    # Events with visual presence
    "Battles", "Festivals", "Sports_events", "Ceremonies",
}

ABSTRACT_CATEGORIES = {
    # Pure abstractions
    "Mathematical_concepts", "Philosophical_concepts", "Abstract_algebra",
    "Theorems", "Conjectures", "Lemmas", "Mathematical_axioms",
    "Logical_expressions", "Mathematical_notation",
    
    # Non-visual concepts
    "Economic_theories", "Political_theories", "Legal_concepts",
    "Sociological_theories", "Psychological_concepts",
    "Linguistic_concepts", "Grammar", "Syntax", "Semantics",
    
    # Meta/structural
    "Wikipedia_categories", "Disambiguation_pages", "Wikipedia_administration",
    "Hidden_categories", "Tracking_categories",
    "Lists", "Indexes", "Outlines", "Timelines", "Chronologies",
    
    # Time periods and eras (not directly visual)
    "Centuries", "Decades", "Years", "Historical_eras",
    
    # Languages and scripts (the concept, not visual text)
    "Languages", "Writing_systems", "Alphabets",
}

# Patterns in category names that suggest visual/abstract
VISUAL_PATTERNS = [
    "photographs_of", "images_of", "pictures_of",
    "_buildings", "_vehicles", "_aircraft", "_ships",
    "_people", "_animals", "_plants",
    "Countries", "Sovereign_states", "Member_states_of_the_United_Nations",
    "Countries_in_Europe", "Countries_in_Asia", "Countries_in_Africa",
    "Oceans", "Seas", "Rivers", "Mountains", "Deserts",
    "_in_europe", "_in_asia", "_in_africa", "_in_north_america",
    "member_states_of_",
    
    # Objects
    "Counting_instruments",  # catches Abacus
    "Mathematical_instruments",
    "Writing_implements",
]

ABSTRACT_PATTERNS = [
    "_theory", "_theories", "_theorem", "_theorems",
    "_concept", "_concepts", "_principles",
    "history_of_", "timeline_of_", "list_of_", "index_of_",
    "_languages", "_language_family",
]


def category_matches_patterns(cat_title: str, patterns: list) -> bool:
    """Check if category title matches any pattern."""
    lower = cat_title.lower()
    return any(p in lower for p in patterns)


def classify_by_categories(cats: set) -> str | None:
    """
    Classify a page based on its categories.
    
    Returns:
        "visual" - definitely visualizable
        "abstract" - definitely not visualizable
        None - uncertain, needs LLM
    """
    if not cats:
        return None
    
    # Check direct membership
    has_visual = bool(cats & VISUAL_CATEGORIES)
    has_abstract = bool(cats & ABSTRACT_CATEGORIES)
    
    # Check patterns
    for cat in cats:
        if category_matches_patterns(cat, VISUAL_PATTERNS):
            has_visual = True
        if category_matches_patterns(cat, ABSTRACT_PATTERNS):
            has_abstract = True
    
    # Decision logic
    if has_abstract and not has_visual:
        return "abstract"
    if has_visual and not has_abstract:
        return "visual"
    if has_visual and has_abstract:
        # Conflicting signals - needs LLM
        return None
    
    return None


# ============================================================================
# Worker functions for multiprocessing
# ============================================================================

# Global for worker processes
_worker_page_to_cats = None
_worker_cat_id_to_title = None


def _init_worker(page_to_cats: dict, cat_id_to_title: dict):
    """Initialize worker process with shared data."""
    global _worker_page_to_cats, _worker_cat_id_to_title
    _worker_page_to_cats = page_to_cats
    _worker_cat_id_to_title = cat_id_to_title


def _process_page(page_id: int) -> tuple[int, str | None, int]:
    """
    Process a single page.
    
    Returns:
        (page_id, classification, num_categories)
    """
    cat_ids = _worker_page_to_cats.get(page_id, set())
    cat_titles = {_worker_cat_id_to_title.get(cid) for cid in cat_ids}
    cat_titles.discard(None)
    
    classification = classify_by_categories(cat_titles)
    return (page_id, classification, len(cat_titles))


def _process_chunk(page_ids: list[int]) -> list[tuple[int, str | None, int]]:
    """Process a chunk of pages."""
    return [_process_page(pid) for pid in page_ids]


# ============================================================================
# Main processing logic
# ============================================================================

def load_category_mappings(page_categories_path: str, categories_path: str):
    """Load and index category data."""
    
    log(f"Loading categories from {categories_path}")
    categories = pq.read_table(categories_path)
    cat_id_to_title = dict(zip(
        categories.column("category_id").to_pylist(),
        categories.column("title").to_pylist()
    ))
    log(f"  Loaded {len(cat_id_to_title):,} categories")
    
    log(f"Loading page-category mappings from {page_categories_path}")
    page_cats = pq.read_table(page_categories_path)
    
    # Build page_id -> set of category_ids
    log("  Building page-to-categories index...")
    page_ids = page_cats.column("page_id").to_numpy()
    cat_ids = page_cats.column("category_id").to_numpy()
    
    page_to_cats = {}
    for pid, cid in tqdm(zip(page_ids, cat_ids), total=len(page_ids), 
                         desc="  Indexing", unit="row"):
        if pid not in page_to_cats:
            page_to_cats[pid] = set()
        page_to_cats[pid].add(cid)
    
    log(f"  Indexed {len(page_to_cats):,} pages with category assignments")
    
    return page_to_cats, cat_id_to_title


def process_nodes_parallel(
    nodes_path: str,
    page_to_cats: dict,
    cat_id_to_title: dict,
    num_workers: int = None,
    chunk_size: int = 10_000
) -> pd.DataFrame:
    """Process nodes in parallel to classify by category."""
    
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    log(f"Loading nodes from {nodes_path}")
    nodes_table = pq.read_table(nodes_path)
    page_ids = nodes_table.column("page_id").to_pylist()
    titles = nodes_table.column("title").to_pylist()
    
    log(f"  Loaded {len(page_ids):,} nodes")
    log(f"Processing with {num_workers} workers, chunk_size={chunk_size:,}")
    
    # Split into chunks
    chunks = [page_ids[i:i + chunk_size] for i in range(0, len(page_ids), chunk_size)]
    
    # Process in parallel
    results = []
    with Pool(num_workers, initializer=_init_worker, 
              initargs=(page_to_cats, cat_id_to_title)) as pool:
        
        for chunk_result in tqdm(
            pool.imap(_process_chunk, chunks),
            total=len(chunks),
            desc="Classifying",
            unit="chunk"
        ):
            results.extend(chunk_result)
    
    # Build result DataFrame
    log("Building result DataFrame...")
    result_df = pd.DataFrame({
        "page_id": page_ids,
        "title": titles,
        "classification": [r[1] for r in results],
        "num_categories": [r[2] for r in results],
    })
    
    return result_df


def print_statistics(df: pd.DataFrame):
    """Print detailed classification statistics."""
    
    log("\n" + "=" * 60)
    log("CLASSIFICATION STATISTICS")
    log("=" * 60)
    
    total = len(df)
    visual = (df["classification"] == "visual").sum()
    abstract = (df["classification"] == "abstract").sum()
    uncertain = df["classification"].isna().sum()
    
    log(f"\nTotal nodes: {total:,}")
    log(f"  Visual:    {visual:,} ({100*visual/total:.1f}%)")
    log(f"  Abstract:  {abstract:,} ({100*abstract/total:.1f}%)")
    log(f"  Uncertain: {uncertain:,} ({100*uncertain/total:.1f}%) <- need LLM")
    
    # Category coverage
    has_cats = (df["num_categories"] > 0).sum()
    log(f"\nCategory coverage:")
    log(f"  With categories:    {has_cats:,} ({100*has_cats/total:.1f}%)")
    log(f"  Without categories: {total - has_cats:,} ({100*(total-has_cats)/total:.1f}%)")
    
    # Category count distribution
    log(f"\nCategories per page:")
    log(f"  Min:    {df['num_categories'].min()}")
    log(f"  Median: {df['num_categories'].median():.0f}")
    log(f"  Mean:   {df['num_categories'].mean():.1f}")
    log(f"  Max:    {df['num_categories'].max()}")
    
    # Sample outputs
    log("\n" + "-" * 60)
    log("Sample VISUAL classifications:")
    visual_samples = df[df["classification"] == "visual"].head(10)
    for _, row in visual_samples.iterrows():
        log(f"  {row['title'][:50]:50s} (cats={row['num_categories']})")
    
    log("\nSample ABSTRACT classifications:")
    abstract_samples = df[df["classification"] == "abstract"].head(10)
    for _, row in abstract_samples.iterrows():
        log(f"  {row['title'][:50]:50s} (cats={row['num_categories']})")
    
    log("\nSample UNCERTAIN (need LLM):")
    uncertain_samples = df[df["classification"].isna()].head(10)
    for _, row in uncertain_samples.iterrows():
        log(f"  {row['title'][:50]:50s} (cats={row['num_categories']})")


def main():
    parser = argparse.ArgumentParser(
        description="Classify nodes by visual generatability using Wikipedia categories",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--nodes", required=True, 
                        help="Filtered nodes.parquet (after PageRank pruning)")
    parser.add_argument("--page-categories", required=True,
                        help="wiki_page_categories.parquet from category extraction")
    parser.add_argument("--categories", required=True,
                        help="wiki_categories.parquet from category extraction")
    parser.add_argument("--out-visual", required=True,
                        help="Output: nodes classified as visual")
    parser.add_argument("--out-uncertain", required=True,
                        help="Output: nodes needing LLM classification")
    parser.add_argument("--out-all", type=str, default=None,
                        help="Output: all nodes with classifications (optional)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of worker processes")
    parser.add_argument("--chunk-size", type=int, default=10_000,
                        help="Chunk size for parallel processing")
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Load category data
    page_to_cats, cat_id_to_title = load_category_mappings(
        args.page_categories, args.categories
    )
    
    # Process nodes
    result_df = process_nodes_parallel(
        args.nodes,
        page_to_cats,
        cat_id_to_title,
        num_workers=args.workers,
        chunk_size=args.chunk_size
    )
    
    # Print statistics
    print_statistics(result_df)
    
    # Save outputs
    log("\nSaving outputs...")
    
    # Visual nodes (keep these)
    visual_df = result_df[result_df["classification"] == "visual"][["page_id", "title"]]
    visual_table = pa.Table.from_pandas(visual_df, preserve_index=False)  # pa, not pq
    pq.write_table(visual_table, args.out_visual, compression="zstd")
    log(f"  Wrote {len(visual_df):,} visual nodes to {args.out_visual}")
    
    # Uncertain nodes (need LLM)
    uncertain_df = result_df[result_df["classification"].isna()][["page_id", "title"]]
    uncertain_table = pa.Table.from_pandas(uncertain_df, preserve_index=False)  # pa, not pq
    pq.write_table(uncertain_table, args.out_uncertain, compression="zstd")
    log(f"  Wrote {len(uncertain_df):,} uncertain nodes to {args.out_uncertain}")
    
    # Optionally save all with classifications
    if args.out_all:
        all_table = pa.Table.from_pandas(result_df, preserve_index=False)  # pa, not pq
        pq.write_table(all_table, args.out_all, compression="zstd")
        log(f"  Wrote {len(result_df):,} total nodes to {args.out_all}")
    
    total_time = time.time() - start_time
    log(f"\nTotal time: {total_time:.1f} seconds")
    
    # Summary
    log("\n" + "=" * 60)
    log("NEXT STEPS")
    log("=" * 60)
    log(f"1. Visual nodes ready: {args.out_visual}")
    log(f"2. Run LLM classification on: {args.out_uncertain}")
    log(f"3. Merge results to get final visual node set")


if __name__ == "__main__":
    main()