"""Prune low-PageRank nodes from the hyperlink graph."""

import argparse
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from wikikg.common import ParquetBatchWriter, iter_parquet_batches


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def load_pagerank(pagerank_path):
    """Load PageRank scores from Parquet file.

    Returns:
        tuple: (page_ids array, ranks array)
    """
    table = pq.read_table(pagerank_path, columns=["page_id", "pagerank"])
    page_ids = table.column("page_id").to_numpy()
    ranks = table.column("pagerank").to_numpy()
    return page_ids, ranks


def determine_threshold(ranks, min_pagerank=None, drop_bottom_pct=None):
    """Determine the PageRank threshold for pruning.

    Args:
        ranks: Array of PageRank scores
        min_pagerank: Absolute minimum threshold (mutually exclusive with drop_bottom_pct)
        drop_bottom_pct: Drop bottom X% of nodes (mutually exclusive with min_pagerank)

    Returns:
        Threshold value
    """
    if min_pagerank is not None and drop_bottom_pct is not None:
        raise ValueError("Choose only one of --min-pagerank or --drop-bottom-pct")
    if min_pagerank is not None:
        return float(min_pagerank)
    if drop_bottom_pct is not None:
        q = float(drop_bottom_pct) / 100.0
        return float(np.quantile(ranks, q))
    raise ValueError("Provide --min-pagerank or --drop-bottom-pct")


def write_pruned_nodes(nodes_path, keep_set, out_path, batch_size=1_000_000):
    """Write pruned nodes to Parquet file."""
    schema = pa.schema([
        ("page_id", pa.int64()),
        ("title", pa.string()),
    ])

    kept = 0
    with ParquetBatchWriter(out_path, schema) as writer:
        for batch in iter_parquet_batches(nodes_path, ["page_id", "title"], batch_size):
            ids = batch.column("page_id").to_numpy()
            titles = batch.column("title").to_pylist()

            out_ids = []
            out_titles = []
            for pid, title in zip(ids, titles):
                if int(pid) in keep_set:
                    out_ids.append(int(pid))
                    out_titles.append(title)
                    kept += 1
            if out_ids:
                writer.write({"page_id": out_ids, "title": out_titles})

    log(f"nodes kept: {kept}")


def write_pruned_edges(edges_path, keep_set, out_path, batch_size=1_000_000):
    """Write pruned edges to Parquet file."""
    schema = pa.schema([
        ("src_id", pa.int64()),
        ("dst_id", pa.int64()),
    ])

    kept = 0
    with ParquetBatchWriter(out_path, schema) as writer:
        for batch in iter_parquet_batches(edges_path, ["src_id", "dst_id"], batch_size):
            src = batch.column("src_id").to_numpy()
            dst = batch.column("dst_id").to_numpy()

            out_src = []
            out_dst = []
            for s, d in zip(src, dst):
                s_id = int(s)
                d_id = int(d)
                if s_id in keep_set and d_id in keep_set:
                    out_src.append(s_id)
                    out_dst.append(d_id)
                    kept += 1
            if out_src:
                writer.write({"src_id": out_src, "dst_id": out_dst})

    log(f"edges kept: {kept}")


def main():
    parser = argparse.ArgumentParser(
        description="Prune low-PageRank nodes from the hyperlink graph"
    )
    parser.add_argument("--nodes", required=True, help="Path to nodes.parquet")
    parser.add_argument("--edges", required=True, help="Path to edges.parquet")
    parser.add_argument("--pagerank", required=True, help="Path to pagerank.parquet")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--min-pagerank", type=float, help="Minimum PageRank threshold")
    parser.add_argument("--drop-bottom-pct", type=float, help="Drop bottom X%% of nodes")
    parser.add_argument("--batch-size", type=int, default=1_000_000, help="Batch size")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    nodes_out = os.path.join(args.out_dir, "nodes_pruned.parquet")
    edges_out = os.path.join(args.out_dir, "edges_pruned.parquet")

    log("loading pagerank...")
    page_ids, ranks = load_pagerank(args.pagerank)
    threshold = determine_threshold(ranks, args.min_pagerank, args.drop_bottom_pct)
    log(f"pagerank threshold: {threshold:.6e}")

    keep_mask = ranks >= threshold
    keep_ids = page_ids[keep_mask]
    keep_set = set(int(x) for x in keep_ids)
    log(f"kept nodes: {len(keep_set)}")

    log("writing pruned nodes...")
    write_pruned_nodes(args.nodes, keep_set, nodes_out, args.batch_size)

    log("writing pruned edges...")
    write_pruned_edges(args.edges, keep_set, edges_out, args.batch_size)


if __name__ == "__main__":
    main()
