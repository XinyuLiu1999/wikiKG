#!/usr/bin/env bash
# WordNet Graph Pipeline
# Builds complete WordNet graph with hypernym/hyponym edges,
# computes PageRank, and prunes low-importance synsets.
#
# Usage:
#   ./scripts/run_wordnet_pipeline.sh [OUT_DIR] [POS]
#
# Arguments:
#   OUT_DIR  Output directory (default: data/graph/wordnet)
#   POS      Part of speech filter: n, v, a, r, s (default: all)
#
# Examples:
#   ./scripts/run_wordnet_pipeline.sh                          # All synsets
#   ./scripts/run_wordnet_pipeline.sh data/graph/wordnet       # All synsets
#   ./scripts/run_wordnet_pipeline.sh data/graph/wordnet n     # Nouns only

set -euo pipefail

OUT_DIR=${1:-data/graph/wordnet}
POS=${2:-}
PRUNED_DIR=${OUT_DIR}/pruned

echo "=== WordNet Graph Pipeline ==="
echo "Output: $OUT_DIR"
if [ -n "$POS" ]; then
    echo "POS filter: $POS"
else
    echo "POS filter: all"
fi
echo ""
echo "NOTE: This pipeline uses NLTK WordNet, NOT Wikipedia data."

# Ensure WordNet data is available
echo ""
echo "Checking WordNet data..."
python -c "import nltk; nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"

# Step 1: Build graph
echo ""
echo "[1/3] Building complete WordNet graph..."
if [ -n "$POS" ]; then
    python -m wikikg.wordnet.build_graph \
      --out-dir "$OUT_DIR" \
      --pos "$POS"
else
    python -m wikikg.wordnet.build_graph \
      --out-dir "$OUT_DIR"
fi

# Step 2: Compute PageRank
echo ""
echo "[2/3] Computing PageRank for synsets..."
python -m wikikg.wordnet.compute_pagerank \
  --nodes "$OUT_DIR/wn_nodes.parquet" \
  --edges "$OUT_DIR/wn_edges.parquet" \
  --out "$OUT_DIR/wn_pagerank.parquet" \
  --max-iter 30 \
  --tol 1e-6

# Step 3: Prune graph
echo ""
echo "[3/3] Pruning low-PageRank synsets..."
python -m wikikg.wordnet.prune_graph \
  --nodes "$OUT_DIR/wn_nodes.parquet" \
  --edges "$OUT_DIR/wn_edges.parquet" \
  --pagerank "$OUT_DIR/wn_pagerank.parquet" \
  --out-dir "$PRUNED_DIR" \
  --drop-bottom-pct 20

echo ""
echo "=== Done ==="
echo "Outputs:"
echo "  - $OUT_DIR/wn_nodes.parquet"
echo "  - $OUT_DIR/wn_edges.parquet (hypernym -> hyponym)"
echo "  - $OUT_DIR/wn_pagerank.parquet"
echo "  - $PRUNED_DIR/wn_nodes_pruned.parquet"
echo "  - $PRUNED_DIR/wn_edges_pruned.parquet"
