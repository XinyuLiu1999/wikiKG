"""Build complete WordNet graph with hypernym/hyponym edges."""

import argparse
import os
import sys

import pyarrow as pa
from tqdm import tqdm   # ← 新增

from wikikg.common import ParquetBatchWriter


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def ensure_wordnet():
    """Ensure WordNet data is available.

    Returns:
        wordnet module

    Raises:
        RuntimeError if WordNet data is not available
    """
    try:
        from nltk.corpus import wordnet as wn
        _ = wn.synsets("dog")
        return wn
    except Exception as exc:
        raise RuntimeError(
            "WordNet data not available. Download with: python -m nltk.downloader wordnet omw-1.4"
        ) from exc


def build_graph(pos=None):
    """Build complete WordNet graph from all synsets.

    Only includes hypernym/hyponym relationships (parent -> child edges).

    Args:
        pos: Part of speech filter ('n', 'v', 'a', 'r', 's') or None for all

    Returns:
        tuple: (nodes list, edges list)
        - nodes: [(node_id, synset_name, definition, pos), ...]
        - edges: [(parent_id, child_id), ...] - hypernym -> hyponym edges
    """
    wn = ensure_wordnet()

    # Get all synsets
    if pos:
        all_synsets = list(wn.all_synsets(pos=pos))
    else:
        all_synsets = list(wn.all_synsets())

    log(f"total synsets in WordNet: {len(all_synsets)}")

    # Build node_id mapping
    node_ids = {}
    nodes = []

    for synset in tqdm(all_synsets, desc="Building nodes"):
        name = synset.name()
        node_id = len(node_ids)
        node_ids[name] = node_id
        nodes.append((node_id, name, synset.definition(), synset.pos()))

    log(f"nodes created: {len(nodes)}")

    # Build edges (hypernym -> hyponym)
    edges_set = set()

    for synset in tqdm(all_synsets, desc="Building edges"):
        child_id = node_ids[synset.name()]

        # Add edges from each hypernym to this synset
        for hypernym in synset.hypernyms():
            parent_id = node_ids[hypernym.name()]
            edges_set.add((parent_id, child_id))

        # Also include instance hypernyms
        for hypernym in synset.instance_hypernyms():
            parent_id = node_ids[hypernym.name()]
            edges_set.add((parent_id, child_id))

    # Convert to sorted list for deterministic output
    edges = sorted(edges_set)

    log(f"edges created: {len(edges)}")

    return nodes, edges


def write_parquet(path, schema, rows, batch_size=100000):
    """Write rows to Parquet file in batches."""
    with ParquetBatchWriter(path, schema) as writer:
        batch_cols = {name: [] for name in schema.names}

        for row in tqdm(rows, desc=f"Writing {os.path.basename(path)}"):
            for name, value in zip(schema.names, row):
                batch_cols[name].append(value)

            if len(next(iter(batch_cols.values()))) >= batch_size:
                writer.write(batch_cols)
                batch_cols = {name: [] for name in schema.names}

        if batch_cols and len(next(iter(batch_cols.values()))) > 0:
            writer.write(batch_cols)


def main():
    parser = argparse.ArgumentParser(
        description="Build complete WordNet graph with hypernym/hyponym edges"
    )
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument(
        "--pos",
        choices=["n", "v", "a", "r", "s"],
        help="Filter by part of speech (n=noun, v=verb, a=adj, r=adv, s=adj satellite)"
    )
    args = parser.parse_args()

    nodes, edges = build_graph(pos=args.pos)

    nodes_schema = pa.schema([
        ("node_id", pa.int64()),
        ("synset", pa.string()),
        ("definition", pa.string()),
        ("pos", pa.string()),
    ])
    edges_schema = pa.schema([
        ("parent_id", pa.int64()),
        ("child_id", pa.int64()),
    ])

    os.makedirs(args.out_dir, exist_ok=True)
    write_parquet(os.path.join(args.out_dir, "wn_nodes.parquet"), nodes_schema, nodes)
    write_parquet(os.path.join(args.out_dir, "wn_edges.parquet"), edges_schema, edges)

    log("done")


if __name__ == "__main__":
    main()
