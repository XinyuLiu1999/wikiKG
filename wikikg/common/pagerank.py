"""Common PageRank computation utilities shared across all pipelines."""

import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp

from .io_utils import ParquetBatchWriter, iter_parquet_batches


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def load_nodes_generic(nodes_path, id_column="node_id"):
    """Load node IDs and build index mapping.

    Args:
        nodes_path: Path to nodes parquet file
        id_column: Name of the ID column

    Returns:
        tuple: (node_ids array, id_to_index dict)
    """
    table = pq.read_table(nodes_path, columns=[id_column])
    node_ids = table.column(id_column).to_numpy()
    id_to_index = {int(nid): i for i, nid in enumerate(node_ids)}
    return node_ids, id_to_index


def load_edges_generic(edges_path, id_to_index, src_column="src_id", dst_column="dst_id", batch_size=1_000_000):
    """Load edges with vectorized processing.

    Args:
        edges_path: Path to edges parquet file
        id_to_index: Mapping from node ID to index
        src_column: Name of source ID column
        dst_column: Name of destination ID column
        batch_size: Batch size for reading

    Returns:
        tuple: (row indices, column indices) as numpy arrays
    """
    all_rows = []
    all_cols = []
    total_kept = 0

    if not id_to_index:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    # Pre-create lookup arrays for vectorized mapping
    max_id = max(id_to_index.keys()) + 1
    id_lookup = np.full(max_id, -1, dtype=np.int64)
    for nid, idx in id_to_index.items():
        id_lookup[nid] = idx

    for batch in iter_parquet_batches(edges_path, [src_column, dst_column], batch_size):
        src = batch.column(src_column).to_numpy()
        dst = batch.column(dst_column).to_numpy()

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


def compute_pagerank(adj, damping=0.85, max_iter=30, tol=1e-6):
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
    if n == 0:
        return np.array([], dtype=np.float64)

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


def build_adjacency_matrix(rows, cols, n):
    """Build sparse adjacency matrix from edge indices.

    Args:
        rows: Source node indices
        cols: Destination node indices
        n: Number of nodes

    Returns:
        CSR sparse matrix
    """
    if len(rows) == 0:
        return sp.csr_matrix((n, n), dtype=np.float64)
    data = np.ones(len(rows), dtype=np.float64)
    return sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


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
        raise ValueError("Choose only one of min_pagerank or drop_bottom_pct")
    if min_pagerank is not None:
        return float(min_pagerank)
    if drop_bottom_pct is not None:
        q = float(drop_bottom_pct) / 100.0
        return float(np.quantile(ranks, q))
    raise ValueError("Provide min_pagerank or drop_bottom_pct")


def get_keep_set(node_ids, ranks, min_pagerank=None, drop_bottom_pct=None):
    """Get set of node IDs to keep based on PageRank threshold.

    Args:
        node_ids: Array of node IDs
        ranks: Array of PageRank scores
        min_pagerank: Absolute minimum threshold
        drop_bottom_pct: Drop bottom X% of nodes

    Returns:
        tuple: (keep_set, threshold)
    """
    threshold = determine_threshold(ranks, min_pagerank, drop_bottom_pct)
    keep_mask = ranks >= threshold
    keep_ids = node_ids[keep_mask]
    keep_set = set(int(x) for x in keep_ids)
    return keep_set, threshold
