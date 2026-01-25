# wikiKG

A Python toolkit for building knowledge graphs from Wikipedia and WordNet.

## Overview

wikiKG provides **three independent pipelines** for constructing knowledge graphs. Each pipeline:
1. Builds a graph from its data source
2. Computes PageRank on its own graph
3. Prunes low-importance nodes

| Pipeline | Data Source | Graph Type |
|----------|-------------|------------|
| **Hyperlink** | Wikipedia SQL dumps | Page hyperlinks |
| **Category** | Wikipedia SQL dumps | Category hierarchy |
| **WordNet** | NLTK WordNet | Semantic hierarchy |

## Project Structure

```
wikiKG/
├── wikikg/                      # Python package
│   ├── common/                  # Shared utilities
│   │   ├── io_utils.py          # Parquet I/O
│   │   ├── sql_utils.py         # SQL dump parsing
│   │   └── pagerank.py          # Common PageRank utilities
│   ├── hyperlink/               # Pipeline 1: Hyperlink graph
│   │   ├── parse_dumps.py       # Extract nodes/edges
│   │   ├── compute_pagerank.py  # PageRank calculation
│   │   └── prune_graph.py       # Remove low-rank nodes
│   ├── category/                # Pipeline 2: Category hierarchy
│   │   ├── parse_categories.py  # Build category tree
│   │   ├── compute_pagerank.py  # PageRank calculation
│   │   └── prune_graph.py       # Remove low-rank categories
│   └── wordnet/                 # Pipeline 3: WordNet hierarchy
│       ├── build_graph.py       # Build from NLTK WordNet
│       ├── compute_pagerank.py  # PageRank calculation
│       └── prune_graph.py       # Remove low-rank synsets
├── scripts/                     # Shell scripts
│   ├── download_dumps.sh
│   ├── run_hyperlink_pipeline.sh
│   ├── run_category_pipeline.sh
│   └── run_wordnet_pipeline.sh
├── config.yaml
└── requirements.txt
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Required Wikipedia Dumps

Download these files to `data/raw/`:

```bash
./scripts/download_dumps.sh 20240101 enwiki data/raw
```

---

## Pipeline 1: Wikipedia Hyperlink Graph

Builds a directed graph from Wikipedia page hyperlinks.

### Data Flow

```
page.sql.gz ─────┐
pagelinks.sql.gz ─┼─→ parse_dumps ─→ nodes.parquet ─┐
redirect.sql.gz ──┘                   edges.parquet ─┼─→ compute_pagerank ─→ pagerank.parquet
                                                     │
                                                     └─→ prune_graph ─→ *_pruned.parquet
```

### Usage

```bash
./scripts/run_hyperlink_pipeline.sh
```

Or step by step:
```bash
python -m wikikg.hyperlink.parse_dumps \
  --page-sql data/raw/page.sql.gz \
  --pagelinks-sql data/raw/pagelinks.sql.gz \
  --redirect-sql data/raw/redirect.sql.gz \
  --out-dir data/graph/hyperlink

python -m wikikg.hyperlink.compute_pagerank \
  --nodes data/graph/hyperlink/nodes.parquet \
  --edges data/graph/hyperlink/edges.parquet \
  --out data/graph/hyperlink/pagerank.parquet

python -m wikikg.hyperlink.prune_graph \
  --nodes data/graph/hyperlink/nodes.parquet \
  --edges data/graph/hyperlink/edges.parquet \
  --pagerank data/graph/hyperlink/pagerank.parquet \
  --out-dir data/graph/hyperlink/pruned \
  --drop-bottom-pct 20
```

---

## Pipeline 2: Wikipedia Category Hierarchy

Builds a category tree and computes PageRank on the category graph.

### Data Flow

```
page.sql.gz ──────────┐
category.sql.gz ──────┼─→ parse_categories ─→ wiki_categories.parquet
categorylinks.sql.gz ─┘                       wiki_category_edges.parquet
                                              wiki_page_categories.parquet
                                                        │
                                                        ├─→ compute_pagerank ─→ category_pagerank.parquet
                                                        │
                                                        └─→ prune_graph ─→ *_pruned.parquet
```

### Usage

```bash
./scripts/run_category_pipeline.sh
```

Or step by step:
```bash
python -m wikikg.category.parse_categories \
  --page-sql data/raw/page.sql.gz \
  --category-sql data/raw/category.sql.gz \
  --categorylinks-sql data/raw/categorylinks.sql.gz \
  --out-dir data/graph/category

python -m wikikg.category.compute_pagerank \
  --categories data/graph/category/wiki_categories.parquet \
  --edges data/graph/category/wiki_category_edges.parquet \
  --out data/graph/category/category_pagerank.parquet

python -m wikikg.category.prune_graph \
  --categories data/graph/category/wiki_categories.parquet \
  --edges data/graph/category/wiki_category_edges.parquet \
  --page-categories data/graph/category/wiki_page_categories.parquet \
  --pagerank data/graph/category/category_pagerank.parquet \
  --out-dir data/graph/category/pruned \
  --drop-bottom-pct 20
```

---

## Pipeline 3: WordNet Graph

Builds a complete WordNet graph with hypernym/hyponym edges.

**Note: This pipeline does NOT use Wikipedia data. It only requires NLTK WordNet.**

### Data Flow

```
NLTK WordNet ─→ build_graph ─→ wn_nodes.parquet
                                wn_edges.parquet
                                       │
                                       ├─→ compute_pagerank ─→ wn_pagerank.parquet
                                       │
                                       └─→ prune_graph ─→ *_pruned.parquet
```

### Prerequisites

```bash
python -m nltk.downloader wordnet omw-1.4
```

### How the Graph is Built

The `build_graph.py` script constructs the complete WordNet graph:

1. **Load all synsets**: Enumerate all synsets from NLTK WordNet
   - WordNet contains ~117,000 synsets (nouns, verbs, adjectives, adverbs)
   - Optionally filter by part of speech with `--pos`

2. **Create nodes**: Each synset becomes a node with:
   - `node_id`: Unique integer identifier
   - `synset`: Synset name (e.g., `dog.n.01`)
   - `definition`: Human-readable definition
   - `pos`: Part of speech (n=noun, v=verb, a=adjective, r=adverb, s=adjective satellite)

3. **Build edges**: Only hypernym/hyponym relationships are included
   - Direction: hypernym (parent) → hyponym (child)
   - Includes both regular hypernyms and instance hypernyms
   - Example: `canine.n.02` → `dog.n.01` (dog is a type of canine)

**WordNet hierarchy example:**
```
entity.n.01
    └── physical_entity.n.01
            └── object.n.01
                    └── living_thing.n.01
                            └── organism.n.01
                                    └── animal.n.01
                                            └── canine.n.02
                                                    └── dog.n.01
```

### Usage

Run the complete pipeline:
```bash
# Build graph with all synsets
./scripts/run_wordnet_pipeline.sh

# Build graph with only nouns
./scripts/run_wordnet_pipeline.sh data/graph/wordnet n
```

Or step by step:
```bash
# Step 1: Build the complete WordNet graph
python -m wikikg.wordnet.build_graph \
  --out-dir data/graph/wordnet

# Step 2: Compute PageRank on the graph
python -m wikikg.wordnet.compute_pagerank \
  --nodes data/graph/wordnet/wn_nodes.parquet \
  --edges data/graph/wordnet/wn_edges.parquet \
  --out data/graph/wordnet/wn_pagerank.parquet

# Step 3: Prune low-importance nodes
python -m wikikg.wordnet.prune_graph \
  --nodes data/graph/wordnet/wn_nodes.parquet \
  --edges data/graph/wordnet/wn_edges.parquet \
  --pagerank data/graph/wordnet/wn_pagerank.parquet \
  --out-dir data/graph/wordnet/pruned \
  --drop-bottom-pct 20
```

### build_graph.py Options

| Option | Default | Description |
|--------|---------|-------------|
| `--out-dir` | (required) | Output directory for parquet files |
| `--pos` | all | Filter by part of speech: `n` (noun), `v` (verb), `a` (adj), `r` (adv), `s` (adj satellite) |

---

## Output Files Summary

### Hyperlink Pipeline
| File | Schema |
|------|--------|
| `nodes.parquet` | (page_id, title) |
| `edges.parquet` | (src_id, dst_id) |
| `pagerank.parquet` | (page_id, pagerank) |
| `pruned/*` | Same schemas, filtered |

### Category Pipeline
| File | Schema |
|------|--------|
| `wiki_categories.parquet` | (category_id, title, page_count) |
| `wiki_category_edges.parquet` | (parent_id, child_id) |
| `wiki_page_categories.parquet` | (page_id, category_id) |
| `category_pagerank.parquet` | (category_id, pagerank) |
| `pruned/*` | Same schemas, filtered |

### WordNet Pipeline
| File | Schema |
|------|--------|
| `wn_nodes.parquet` | (node_id, synset, definition, pos) |
| `wn_edges.parquet` | (parent_id, child_id) |
| `wn_pagerank.parquet` | (node_id, pagerank) |
| `pruned/*` | Same schemas, filtered |

---

## Notes

- Each pipeline computes PageRank on its own graph structure
- Pruning removes the bottom X% nodes by PageRank (default: 20%)
- All pipelines use the same PageRank algorithm with configurable parameters
- The parser only keeps main-namespace pages (namespace=0)
- Redirects are resolved so edges point to canonical target pages

## License

MIT
