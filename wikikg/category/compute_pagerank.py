"""Compute PageRank scores for the Wikipedia category hierarchy."""

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
        description="Compute PageRank for Wikipedia category hierarchy"
    )
    parser.add_argument("--categories", required=True, help="Path to wiki_categories.parquet")
    parser.add_argument("--edges", required=True, help="Path to wiki_category_edges.parquet")
    parser.add_argument("--out", required=True, help="Output path for category_pagerank.parquet")
    parser.add_argument("--damping", type=float, default=0.85, help="Damping factor")
    parser.add_argument("--max-iter", type=int, default=30, help="Max iterations")
    parser.add_argument("--tol", type=float, default=1e-6, help="Convergence tolerance")
    parser.add_argument("--edge-batch", type=int, default=1_000_000, help="Edge batch size")
    args = parser.parse_args()

    log("loading categories...")
    category_ids, id_to_index = load_nodes_generic(args.categories, id_column="category_id")
    log(f"categories: {len(category_ids)}")

    log("loading category edges...")
    rows, cols = load_edges_generic(
        args.edges,
        id_to_index,
        src_column="child_id",  # 原来是 parent_id
        dst_column="parent_id",  # 原来是 child_id
        batch_size=args.edge_batch,
    )
    log(f"edges: {len(rows)}")

    n = len(category_ids)
    adj = build_adjacency_matrix(rows, cols, n)

    log("computing pagerank...")
    rank = compute_pagerank(adj, args.damping, args.max_iter, args.tol)

    log("writing pagerank...")
    schema = pa.schema([
        ("category_id", pa.int64()),
        ("pagerank", pa.float64()),
    ])

    batch_size = 1_000_000
    with ParquetBatchWriter(args.out, schema) as writer:
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_ids = [int(cid) for cid in category_ids[start:end]]
            batch_rank = [float(r) for r in rank[start:end]]
            writer.write({"category_id": batch_ids, "pagerank": batch_rank})

    log("done")


if __name__ == "__main__":
    main()
# python -m wikikg.category.compute_pagerank \
#   --categories data/graph/category/wiki_categories.parquet \
#   --edges data/graph/category/wiki_category_edges.parquet \
#   --out data/graph/category/category_pagerank.parquet
# python -c "import pandas as pd; df = pd.read_parquet('data/graph/category/category_pagerank.parquet').merge(pd.read_parquet('data/graph/category/wiki_categories.parquet'), on='category_id'); print(df.sort_values('pagerank', ascending=True).head(10)[['title', 'pagerank']])"