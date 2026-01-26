"""WordNet Semantic Hierarchy Pipeline.

This pipeline builds a semantic hierarchy from WordNet (independent of Wikipedia):
1. build_graph - Load tags and traverse hypernym relationships
2. compute_pagerank - Calculate importance scores for each synset
3. prune_graph - Remove low-importance synsets

Data sources:
- NLTK WordNet corpus (Princeton WordNet)
- User-provided tags.txt file

Note: This pipeline does NOT use Wikipedia data.
"""

from .build_graph import build_graph, ensure_wordnet

__all__ = [
    "build_graph",
    "ensure_wordnet",
]
