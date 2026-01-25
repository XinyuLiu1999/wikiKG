"""Compute PageRank scores for the hyperlink graph."""

import argparse
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp

from wikikg.common import ParquetBatchWriter, iter_parquet_batches


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def load_nodes(nodes_path):
    """Load node IDs and build index mapping.

    Returns:
        tuple: (page_ids array, id_to_index dict)
    """
    table = pq.read_table(nodes_path, columns=["page_id"])
    page_ids = table.column("page_id").to_numpy()
    id_to_index = {int(pid): i for i, pid in enumerate(page_ids)}
    return page_ids, id_to_index


def load_edges(edges_path, id_to_index, batch_size=1_000_000):
    """Load edges with vectorized processing for better performance.

    Uses numpy operations instead of Python loops where possible.

    Returns:
        tuple: (row indices, column indices) as numpy arrays
    """
    all_rows = []
    all_cols = []
    total_kept = 0

    # Pre-create lookup arrays for vectorized mapping
    max_id = max(id_to_index.keys()) + 1
    id_lookup = np.full(max_id, -1, dtype=np.int64)
    for pid, idx in id_to_index.items():
        id_lookup[pid] = idx

    for batch in iter_parquet_batches(edges_path, ["src_id", "dst_id"], batch_size):
        src = batch.column("src_id").to_numpy()
        dst = batch.column("dst_id").to_numpy()

        # Vectorized bounds check
        valid_src_bounds = (src >= 0) & (src < max_id)
        valid_dst_bounds = (dst >= 0) & (dst < max_id)
        valid_bounds = valid_src_bounds & valid_dst_bounds

        # Apply bounds filter first
        src_bounded = src[valid_bounds]
        dst_bounded = dst[valid_bounds]

        # Vectorized index lookup
        src_idx = id_lookup[src_bounded]
        dst_idx = id_lookup[dst_bounded]

        # Filter out invalid mappings (-1 means not in graph)
        valid_mask = (src_idx >= 0) & (dst_idx >= 0)
        src_valid = src_idx[valid_mask]
        dst_valid = dst_idx[valid_mask]

        all_rows.append(src_valid)
        all_cols.append(dst_valid)
        total_kept += len(src_valid)

        if total_kept % 5_000_000 == 0 and total_kept > 0:
            log(f"edges: loaded {total_kept}")

    # Concatenate all batches
    rows = np.concatenate(all_rows) if all_rows else np.array([], dtype=np.int64)
    cols = np.concatenate(all_cols) if all_cols else np.array([], dtype=np.int64)

    return rows, cols


def pagerank(adj, damping=0.85, max_iter=30, tol=1e-6):
    """Compute PageRank using power iteration.

    Handles dangling nodes (nodes with no outgoing edges) properly.

    Args:
        adj: Sparse adjacency matrix (CSR format)
        damping: Damping factor (default: 0.85)
        max_iter: Maximum iterations (default: 30)
        tol: Convergence tolerance (default: 1e-6)

    Returns:
        numpy array of PageRank scores
    """
    n = adj.shape[0]
    out_degree = np.array(adj.sum(axis=1)).ravel()
    inv_out = np.zeros(n, dtype=np.float64)
    mask = out_degree > 0
    inv_out[mask] = 1.0 / out_degree[mask]

    rank = np.full(n, 1.0 / n, dtype=np.float64)
    teleport = (1.0 - damping) / n

    for i in range(max_iter):
        # Dangling nodes contribute their rank to all nodes
        dangling = rank[~mask].sum()
        # Contribution from links
        contrib = adj.transpose().dot(rank * inv_out)
        # New rank = random jump + link contribution + dangling contribution
        new_rank = teleport + damping * (contrib + dangling / n)
        delta = np.abs(new_rank - rank).sum()
        rank = new_rank
        log(f"iter {i + 1}: delta={delta:.6e}")
        if delta < tol:
            break

    return rank


def write_pagerank(page_ids, rank, out_path, batch_size=1_000_000):
    """Write PageRank scores to Parquet file."""
    schema = pa.schema([
        ("page_id", pa.int64()),
        ("pagerank", pa.float64()),
    ])

    with ParquetBatchWriter(out_path, schema) as writer:
        # Process in batches to avoid memory issues
        n = len(page_ids)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_ids = [int(pid) for pid in page_ids[start:end]]
            batch_rank = [float(r) for r in rank[start:end]]
            writer.write({"page_id": batch_ids, "pagerank": batch_rank})


def main():
    parser = argparse.ArgumentParser(
        description="Compute PageRank for Wikipedia hyperlink graph"
    )
    parser.add_argument("--nodes", required=True, help="Path to nodes.parquet")
    parser.add_argument("--edges", required=True, help="Path to edges.parquet")
    parser.add_argument("--out", required=True, help="Output path for pagerank.parquet")
    parser.add_argument("--damping", type=float, default=0.85, help="Damping factor")
    parser.add_argument("--max-iter", type=int, default=30, help="Max iterations")
    parser.add_argument("--tol", type=float, default=1e-6, help="Convergence tolerance")
    parser.add_argument("--edge-batch", type=int, default=1_000_000, help="Edge batch size")
    args = parser.parse_args()

    log("loading nodes...")
    page_ids, id_to_index = load_nodes(args.nodes)
    log(f"nodes: {len(page_ids)}")

    log("loading edges...")
    rows, cols = load_edges(args.edges, id_to_index, args.edge_batch)
    log(f"edges: {len(rows)}")

    n = len(page_ids)
    data = np.ones(len(rows), dtype=np.float64)
    adj = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()

    log("computing pagerank...")
    rank = pagerank(adj, args.damping, args.max_iter, args.tol)

    log("writing pagerank...")
    write_pagerank(page_ids, rank, args.out)


if __name__ == "__main__":
    main()
