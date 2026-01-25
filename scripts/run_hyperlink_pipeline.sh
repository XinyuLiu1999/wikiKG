#!/usr/bin/env bash
# Wikipedia Hyperlink Graph Pipeline
# Builds a knowledge graph from Wikipedia page hyperlinks and computes PageRank.
#
# Usage:
#   ./scripts/run_hyperlink_pipeline.sh [PAGE_SQL] [PAGELINKS_SQL] [REDIRECT_SQL] [OUT_DIR] [PRUNED_DIR]
#
# Example:
#   ./scripts/run_hyperlink_pipeline.sh data/raw/page.sql.gz data/raw/pagelinks.sql.gz data/raw/redirect.sql.gz data/graph/hyperlink data/graph/hyperlink/pruned

set -euo pipefail

PAGE_SQL=${1:-data/raw/page.sql.gz}
PAGELINKS_SQL=${2:-data/raw/pagelinks.sql.gz}
REDIRECT_SQL=${3:-data/raw/redirect.sql.gz}
OUT_DIR=${4:-data/graph/hyperlink}
PRUNED_DIR=${5:-data/graph/hyperlink/pruned}

echo "=== Wikipedia Hyperlink Graph Pipeline ==="
echo "Input: $PAGE_SQL, $PAGELINKS_SQL, $REDIRECT_SQL"
echo "Output: $OUT_DIR"

# Step 1: Parse dumps
echo ""
echo "[1/3] Parsing Wikipedia dumps..."
python -m wikikg.hyperlink.parse_dumps \
  --page-sql "$PAGE_SQL" \
  --pagelinks-sql "$PAGELINKS_SQL" \
  --redirect-sql "$REDIRECT_SQL" \
  --out-dir "$OUT_DIR"

# Step 2: Compute PageRank
echo ""
echo "[2/3] Computing PageRank..."
python -m wikikg.hyperlink.compute_pagerank \
  --nodes "$OUT_DIR/nodes.parquet" \
  --edges "$OUT_DIR/edges.parquet" \
  --out "$OUT_DIR/pagerank.parquet" \
  --max-iter 30 \
  --tol 1e-6

# Step 3: Prune graph
echo ""
echo "[3/3] Pruning low-PageRank nodes..."
python -m wikikg.hyperlink.prune_graph \
  --nodes "$OUT_DIR/nodes.parquet" \
  --edges "$OUT_DIR/edges.parquet" \
  --pagerank "$OUT_DIR/pagerank.parquet" \
  --out-dir "$PRUNED_DIR" \
  --drop-bottom-pct 20

echo ""
echo "=== Done ==="
echo "Outputs:"
echo "  - $OUT_DIR/nodes.parquet"
echo "  - $OUT_DIR/edges.parquet"
echo "  - $OUT_DIR/pagerank.parquet"
echo "  - $PRUNED_DIR/nodes_pruned.parquet"
echo "  - $PRUNED_DIR/edges_pruned.parquet"
