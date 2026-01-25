#!/usr/bin/env bash
set -euo pipefail

DUMP_DATE=${1:-"20240101"}
LANG=${2:-"enwiki"}
OUT_DIR=${3:-"data/raw"}

mkdir -p "$OUT_DIR"
BASE="https://dumps.wikimedia.org/${LANG}/${DUMP_DATE}"

curl -L -o "$OUT_DIR/page.sql.gz" "$BASE/${LANG}-${DUMP_DATE}-page.sql.gz"
curl -L -o "$OUT_DIR/pagelinks.sql.gz" "$BASE/${LANG}-${DUMP_DATE}-pagelinks.sql.gz"
curl -L -o "$OUT_DIR/redirect.sql.gz" "$BASE/${LANG}-${DUMP_DATE}-redirect.sql.gz"
curl -L -o "$OUT_DIR/category.sql.gz" "$BASE/${LANG}-${DUMP_DATE}-category.sql.gz"
curl -L -o "$OUT_DIR/categorylinks.sql.gz" "$BASE/${LANG}-${DUMP_DATE}-categorylinks.sql.gz"
