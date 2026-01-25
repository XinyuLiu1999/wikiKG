"""Compute PageRank scores for the WordNet semantic hierarchy."""

import argparse
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from wikikg.common import (
    ParquetBatchWriter,
    load_nodes_generic,
    load_edges_generic,
    compute_pagerank,
    build_adjacency_matrix,
)


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Compute PageRank for WordNet semantic hierarchy"
    )
    parser.add_argument("--nodes", required=True, help="Path to wn_nodes.parquet")
    parser.add_argument("--edges", required=True, help="Path to wn_edges.parquet")
    parser.add_argument("--out", required=True, help="Output path for wn_pagerank.parquet")
    parser.add_argument("--damping", type=float, default=0.85, help="Damping factor")
    parser.add_argument("--max-iter", type=int, default=30, help="Max iterations")
    parser.add_argument("--tol", type=float, default=1e-6, help="Convergence tolerance")
    parser.add_argument("--edge-batch", type=int, default=1_000_000, help="Edge batch size")
    args = parser.parse_args()

    log("loading nodes...")
    node_ids, id_to_index = load_nodes_generic(args.nodes, id_column="node_id")
    log(f"nodes: {len(node_ids)}")

    log("loading edges...")
    # WordNet edges are (parent_id, child_id), we treat parent->child as the edge direction
    rows, cols = load_edges_generic(
        args.edges,
        id_to_index,
        src_column="parent_id",
        dst_column="child_id",
        batch_size=args.edge_batch,
    )
    log(f"edges: {len(rows)}")

    n = len(node_ids)
    adj = build_adjacency_matrix(rows, cols, n)

    log("computing pagerank...")
    rank = compute_pagerank(adj, args.damping, args.max_iter, args.tol)

    log("writing pagerank...")
    schema = pa.schema([
        ("node_id", pa.int64()),
        ("pagerank", pa.float64()),
    ])

    batch_size = 1_000_000
    with ParquetBatchWriter(args.out, schema) as writer:
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_ids = [int(nid) for nid in node_ids[start:end]]
            batch_rank = [float(r) for r in rank[start:end]]
            writer.write({"node_id": batch_ids, "pagerank": batch_rank})

    log("done")


if __name__ == "__main__":
    main()
