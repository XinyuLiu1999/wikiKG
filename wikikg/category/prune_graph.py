"""Prune low-PageRank categories from the category hierarchy."""

import argparse
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from wikikg.common import ParquetBatchWriter, iter_parquet_batches, get_keep_set


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def load_category_pagerank(pagerank_path):
    """Load category PageRank scores from Parquet file."""
    table = pq.read_table(pagerank_path, columns=["category_id", "pagerank"])
    category_ids = table.column("category_id").to_numpy()
    ranks = table.column("pagerank").to_numpy()
    return category_ids, ranks


def write_pruned_categories(categories_path, keep_set, out_path, batch_size=1_000_000):
    """Write pruned categories to Parquet file."""
    schema = pa.schema([
        ("category_id", pa.int64()),
        ("title", pa.string()),
        ("page_count", pa.int64()),
    ])

    kept = 0
    with ParquetBatchWriter(out_path, schema) as writer:
        for batch in iter_parquet_batches(
            categories_path, ["category_id", "title", "page_count"], batch_size
        ):
            ids = batch.column("category_id").to_numpy()
            titles = batch.column("title").to_pylist()
            counts = batch.column("page_count").to_numpy()

            out_ids = []
            out_titles = []
            out_counts = []
            for cid, title, count in zip(ids, titles, counts):
                if int(cid) in keep_set:
                    out_ids.append(int(cid))
                    out_titles.append(title)
                    out_counts.append(int(count))
                    kept += 1
            if out_ids:
                writer.write({
                    "category_id": out_ids,
                    "title": out_titles,
                    "page_count": out_counts,
                })

    log(f"categories kept: {kept}")


def write_pruned_category_edges(edges_path, keep_set, out_path, batch_size=1_000_000):
    """Write pruned category edges to Parquet file."""
    schema = pa.schema([
        ("parent_id", pa.int64()),
        ("child_id", pa.int64()),
    ])

    kept = 0
    with ParquetBatchWriter(out_path, schema) as writer:
        for batch in iter_parquet_batches(edges_path, ["parent_id", "child_id"], batch_size):
            parent = batch.column("parent_id").to_numpy()
            child = batch.column("child_id").to_numpy()

            out_parent = []
            out_child = []
            for p, c in zip(parent, child):
                p_id = int(p)
                c_id = int(c)
                if p_id in keep_set and c_id in keep_set:
                    out_parent.append(p_id)
                    out_child.append(c_id)
                    kept += 1
            if out_parent:
                writer.write({"parent_id": out_parent, "child_id": out_child})

    log(f"category edges kept: {kept}")


def write_pruned_page_categories(page_categories_path, keep_set, out_path, batch_size=1_000_000):
    """Write pruned page-category mappings to Parquet file.

    Only keeps mappings where the category is in keep_set.
    """
    schema = pa.schema([
        ("page_id", pa.int64()),
        ("category_id", pa.int64()),
    ])

    kept = 0
    with ParquetBatchWriter(out_path, schema) as writer:
        for batch in iter_parquet_batches(page_categories_path, ["page_id", "category_id"], batch_size):
            page_ids = batch.column("page_id").to_numpy()
            cat_ids = batch.column("category_id").to_numpy()

            out_pages = []
            out_cats = []
            for pid, cid in zip(page_ids, cat_ids):
                if int(cid) in keep_set:
                    out_pages.append(int(pid))
                    out_cats.append(int(cid))
                    kept += 1
            if out_pages:
                writer.write({"page_id": out_pages, "category_id": out_cats})

    log(f"page-category mappings kept: {kept}")


def main():
    parser = argparse.ArgumentParser(
        description="Prune low-PageRank categories from the hierarchy"
    )
    parser.add_argument("--categories", required=True, help="Path to wiki_categories.parquet")
    parser.add_argument("--edges", required=True, help="Path to wiki_category_edges.parquet")
    parser.add_argument("--page-categories", required=True, help="Path to wiki_page_categories.parquet")
    parser.add_argument("--pagerank", required=True, help="Path to category_pagerank.parquet")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--min-pagerank", type=float, help="Minimum PageRank threshold")
    parser.add_argument("--drop-bottom-pct", type=float, help="Drop bottom X%% of categories")
    parser.add_argument("--batch-size", type=int, default=1_000_000, help="Batch size")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    categories_out = os.path.join(args.out_dir, "wiki_categories_pruned.parquet")
    edges_out = os.path.join(args.out_dir, "wiki_category_edges_pruned.parquet")
    page_categories_out = os.path.join(args.out_dir, "wiki_page_categories_pruned.parquet")

    log("loading category pagerank...")
    category_ids, ranks = load_category_pagerank(args.pagerank)
    keep_set, threshold = get_keep_set(
        category_ids, ranks,
        min_pagerank=args.min_pagerank,
        drop_bottom_pct=args.drop_bottom_pct,
    )
    log(f"pagerank threshold: {threshold:.6e}")
    log(f"categories to keep: {len(keep_set)}")

    log("writing pruned categories...")
    write_pruned_categories(args.categories, keep_set, categories_out, args.batch_size)

    log("writing pruned category edges...")
    write_pruned_category_edges(args.edges, keep_set, edges_out, args.batch_size)

    log("writing pruned page-category mappings...")
    write_pruned_page_categories(args.page_categories, keep_set, page_categories_out, args.batch_size)


if __name__ == "__main__":
    main()
