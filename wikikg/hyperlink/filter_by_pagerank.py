"""Filter graph nodes by PageRank threshold."""

import argparse
import pyarrow.parquet as pq
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", required=True, help="Original nodes.parquet")
    parser.add_argument("--edges", required=True, help="Original edges.parquet")
    parser.add_argument("--pagerank", required=True, help="pagerank.parquet from previous step")
    parser.add_argument("--out-nodes", required=True, help="Output filtered nodes")
    parser.add_argument("--out-edges", required=True, help="Output filtered edges")
    parser.add_argument("--threshold", type=float, default=None, help="Minimum PageRank score")
    parser.add_argument("--percentile", type=float, default=75, help="Keep top N percent (alternative to threshold)")
    args = parser.parse_args()

    # Load PageRank scores
    pr_table = pq.read_table(args.pagerank)
    page_ids = pr_table.column("page_id").to_numpy()
    scores = pr_table.column("pagerank").to_numpy()

    # Determine threshold
    if args.threshold is not None:
        threshold = args.threshold
    else:
        threshold = np.percentile(scores, args.percentile)
    
    print(f"Threshold: {threshold:.6e} (percentile {args.percentile})")

    # Filter nodes
    keep_mask = scores >= threshold
    keep_ids = set(page_ids[keep_mask].tolist())
    print(f"Keeping {len(keep_ids):,} / {len(page_ids):,} nodes ({100*len(keep_ids)/len(page_ids):.1f}%)")

    # Filter nodes.parquet
    nodes_table = pq.read_table(args.nodes)
    node_ids = nodes_table.column("page_id").to_numpy()
    node_mask = np.isin(node_ids, list(keep_ids))
    filtered_nodes = nodes_table.filter(node_mask)
    pq.write_table(filtered_nodes, args.out_nodes, compression="zstd")
    print(f"Wrote {filtered_nodes.num_rows:,} nodes to {args.out_nodes}")

    # Filter edges.parquet (both endpoints must be in keep_ids)
    edges_table = pq.read_table(args.edges)
    src_ids = edges_table.column("src_id").to_numpy()
    dst_ids = edges_table.column("dst_id").to_numpy()
    edge_mask = np.isin(src_ids, list(keep_ids)) & np.isin(dst_ids, list(keep_ids))
    filtered_edges = edges_table.filter(edge_mask)
    pq.write_table(filtered_edges, args.out_edges, compression="zstd")
    print(f"Wrote {filtered_edges.num_rows:,} edges to {args.out_edges}")

if __name__ == "__main__":
    main()