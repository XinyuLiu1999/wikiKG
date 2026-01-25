"""Prune low-PageRank nodes from the WordNet graph."""

import argparse
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq

from wikikg.common import ParquetBatchWriter, iter_parquet_batches, get_keep_set


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def load_wordnet_pagerank(pagerank_path):
    """Load WordNet PageRank scores from Parquet file."""
    table = pq.read_table(pagerank_path, columns=["node_id", "pagerank"])
    node_ids = table.column("node_id").to_numpy()
    ranks = table.column("pagerank").to_numpy()
    return node_ids, ranks


def write_pruned_nodes(nodes_path, keep_set, out_path, batch_size=1_000_000):
    """Write pruned nodes to Parquet file."""
    schema = pa.schema([
        ("node_id", pa.int64()),
        ("synset", pa.string()),
        ("definition", pa.string()),
        ("pos", pa.string()),
    ])

    kept = 0
    with ParquetBatchWriter(out_path, schema) as writer:
        for batch in iter_parquet_batches(
            nodes_path, ["node_id", "synset", "definition", "pos"], batch_size
        ):
            ids = batch.column("node_id").to_numpy()
            synsets = batch.column("synset").to_pylist()
            definitions = batch.column("definition").to_pylist()
            pos_list = batch.column("pos").to_pylist()

            out_ids = []
            out_synsets = []
            out_defs = []
            out_pos = []
            for nid, synset, definition, pos in zip(ids, synsets, definitions, pos_list):
                if int(nid) in keep_set:
                    out_ids.append(int(nid))
                    out_synsets.append(synset)
                    out_defs.append(definition)
                    out_pos.append(pos)
                    kept += 1
            if out_ids:
                writer.write({
                    "node_id": out_ids,
                    "synset": out_synsets,
                    "definition": out_defs,
                    "pos": out_pos,
                })

    log(f"nodes kept: {kept}")


def write_pruned_edges(edges_path, keep_set, out_path, batch_size=1_000_000):
    """Write pruned edges to Parquet file."""
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

    log(f"edges kept: {kept}")


def main():
    parser = argparse.ArgumentParser(
        description="Prune low-PageRank nodes from the WordNet graph"
    )
    parser.add_argument("--nodes", required=True, help="Path to wn_nodes.parquet")
    parser.add_argument("--edges", required=True, help="Path to wn_edges.parquet")
    parser.add_argument("--pagerank", required=True, help="Path to wn_pagerank.parquet")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--min-pagerank", type=float, help="Minimum PageRank threshold")
    parser.add_argument("--drop-bottom-pct", type=float, help="Drop bottom X%% of nodes")
    parser.add_argument("--batch-size", type=int, default=1_000_000, help="Batch size")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    nodes_out = os.path.join(args.out_dir, "wn_nodes_pruned.parquet")
    edges_out = os.path.join(args.out_dir, "wn_edges_pruned.parquet")

    log("loading pagerank...")
    node_ids, ranks = load_wordnet_pagerank(args.pagerank)
    keep_set, threshold = get_keep_set(
        node_ids, ranks,
        min_pagerank=args.min_pagerank,
        drop_bottom_pct=args.drop_bottom_pct,
    )
    log(f"pagerank threshold: {threshold:.6e}")
    log(f"nodes to keep: {len(keep_set)}")

    log("writing pruned nodes...")
    write_pruned_nodes(args.nodes, keep_set, nodes_out, args.batch_size)

    log("writing pruned edges...")
    write_pruned_edges(args.edges, keep_set, edges_out, args.batch_size)

    log("done")


if __name__ == "__main__":
    main()
