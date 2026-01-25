#!/usr/bin/env bash
# Wikipedia Category Hierarchy Pipeline
# Builds a category hierarchy from Wikipedia category relationships,
# computes PageRank, and prunes low-importance categories.
#
# Usage:
#   ./scripts/run_category_pipeline.sh [PAGE_SQL] [CATEGORY_SQL] [CATEGORYLINKS_SQL] [OUT_DIR]
#
# Example:
#   ./scripts/run_category_pipeline.sh data/raw/page.sql.gz data/raw/category.sql.gz data/raw/categorylinks.sql.gz data/graph/category

set -euo pipefail

PAGE_SQL=${1:-data/raw/page.sql.gz}
CATEGORY_SQL=${2:-data/raw/category.sql.gz}
CATEGORYLINKS_SQL=${3:-data/raw/categorylinks.sql.gz}
OUT_DIR=${4:-data/graph/category}
PRUNED_DIR=${OUT_DIR}/pruned

echo "=== Wikipedia Category Hierarchy Pipeline ==="
echo "Input: $PAGE_SQL, $CATEGORY_SQL, $CATEGORYLINKS_SQL"
echo "Output: $OUT_DIR"

# Step 1: Parse categories
echo ""
echo "[1/3] Parsing category hierarchy..."
python -m wikikg.category.parse_categories \
  --page-sql "$PAGE_SQL" \
  --category-sql "$CATEGORY_SQL" \
  --categorylinks-sql "$CATEGORYLINKS_SQL" \
  --out-dir "$OUT_DIR"

# Step 2: Compute PageRank
echo ""
echo "[2/3] Computing PageRank for categories..."
python -m wikikg.category.compute_pagerank \
  --categories "$OUT_DIR/wiki_categories.parquet" \
  --edges "$OUT_DIR/wiki_category_edges.parquet" \
  --out "$OUT_DIR/category_pagerank.parquet" \
  --max-iter 30 \
  --tol 1e-6

# Step 3: Prune graph
echo ""
echo "[3/3] Pruning low-PageRank categories..."
python -m wikikg.category.prune_graph \
  --categories "$OUT_DIR/wiki_categories.parquet" \
  --edges "$OUT_DIR/wiki_category_edges.parquet" \
  --page-categories "$OUT_DIR/wiki_page_categories.parquet" \
  --pagerank "$OUT_DIR/category_pagerank.parquet" \
  --out-dir "$PRUNED_DIR" \
  --drop-bottom-pct 20

echo ""
echo "=== Done ==="
echo "Outputs:"
echo "  - $OUT_DIR/wiki_categories.parquet"
echo "  - $OUT_DIR/wiki_category_edges.parquet (parent -> child)"
echo "  - $OUT_DIR/wiki_page_categories.parquet (page -> category)"
echo "  - $OUT_DIR/category_pagerank.parquet"
echo "  - $PRUNED_DIR/wiki_categories_pruned.parquet"
echo "  - $PRUNED_DIR/wiki_category_edges_pruned.parquet"
echo "  - $PRUNED_DIR/wiki_page_categories_pruned.parquet"
