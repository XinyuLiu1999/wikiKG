"""Analyze Wikipedia knowledge graph to identify nodes for filtering.

This script helps you discover:
1. High in-degree nodes that might be citation infrastructure
2. Dangling nodes (no outgoing links) - often meta-pages
3. Nodes with suspicious title patterns
4. Potential namespace leakage

Run this BEFORE computing PageRank to decide what to filter.
"""

import argparse
import re
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm


def load_graph(nodes_path, edges_path):
    """Load nodes and edges."""
    print("Loading nodes...")
    nodes = pq.read_table(nodes_path).to_pandas()
    
    print("Loading edges...")
    edges = pq.read_table(edges_path).to_pandas()
    
    print(f"Loaded {len(nodes):,} nodes and {len(edges):,} edges")
    return nodes, edges


def analyze_degree_distribution(nodes, edges):
    """Analyze in-degree and out-degree distributions."""
    print("\n" + "="*60)
    print("DEGREE ANALYSIS")
    print("="*60)
    
    # Compute degrees
    in_degree = edges['dst_id'].value_counts()
    out_degree = edges['src_id'].value_counts()
    
    # Map to all nodes (some may have 0 degree)
    nodes = nodes.copy()
    nodes['in_degree'] = nodes['page_id'].map(in_degree).fillna(0).astype(int)
    nodes['out_degree'] = nodes['page_id'].map(out_degree).fillna(0).astype(int)
    
    # Statistics
    print(f"\nIn-degree statistics:")
    print(f"  Mean: {nodes['in_degree'].mean():.1f}")
    print(f"  Median: {nodes['in_degree'].median():.0f}")
    print(f"  Max: {nodes['in_degree'].max():,}")
    print(f"  Nodes with 0 in-degree: {(nodes['in_degree'] == 0).sum():,}")
    
    print(f"\nOut-degree statistics:")
    print(f"  Mean: {nodes['out_degree'].mean():.1f}")
    print(f"  Median: {nodes['out_degree'].median():.0f}")
    print(f"  Max: {nodes['out_degree'].max():,}")
    print(f"  Nodes with 0 out-degree (dangling): {(nodes['out_degree'] == 0).sum():,}")
    
    return nodes


def find_high_indegree_dangling(nodes, top_n=100):
    """Find nodes with high in-degree but zero out-degree.
    
    These are often meta-pages like ISBN, DOI that receive many links
    but don't link to anything (or link outside the main namespace).
    """
    print("\n" + "="*60)
    print("HIGH IN-DEGREE DANGLING NODES (likely meta-pages)")
    print("="*60)
    
    dangling = nodes[nodes['out_degree'] == 0].copy()
    dangling = dangling.sort_values('in_degree', ascending=False).head(top_n)
    
    print(f"\nTop {top_n} nodes with HIGH in-degree but ZERO out-degree:")
    print("(These are strong candidates for filtering)\n")
    
    for i, row in dangling.iterrows():
        print(f"  {row['in_degree']:>10,} links → {row['title'][:60]}")
    
    return dangling


def find_high_indegree_low_outdegree(nodes, in_threshold=10000, out_threshold=10, top_n=100):
    """Find nodes with high in-degree but very low out-degree."""
    print("\n" + "="*60)
    print(f"HIGH IN-DEGREE (>{in_threshold:,}) + LOW OUT-DEGREE (<{out_threshold})")
    print("="*60)
    
    suspicious = nodes[
        (nodes['in_degree'] > in_threshold) & 
        (nodes['out_degree'] < out_threshold)
    ].copy()
    suspicious = suspicious.sort_values('in_degree', ascending=False).head(top_n)
    
    print(f"\nFound {len(suspicious):,} suspicious nodes:\n")
    for i, row in suspicious.iterrows():
        print(f"  in={row['in_degree']:>8,}  out={row['out_degree']:>4}  {row['title'][:50]}")
    
    return suspicious


def find_top_hubs(nodes, top_n=50):
    """Find top nodes by in-degree (knowledge hubs)."""
    print("\n" + "="*60)
    print(f"TOP {top_n} KNOWLEDGE HUBS (by in-degree)")
    print("="*60)
    
    hubs = nodes.sort_values('in_degree', ascending=False).head(top_n)
    
    print("\nReview these - meta-pages should be filtered:\n")
    for i, (_, row) in enumerate(hubs.iterrows(), 1):
        marker = "⚠️ " if row['out_degree'] < 50 else "   "
        print(f"{i:3}. {marker}in={row['in_degree']:>8,}  out={row['out_degree']:>5,}  {row['title'][:45]}")
    
    return hubs


def analyze_title_patterns(nodes):
    """Identify potential issues based on title patterns."""
    print("\n" + "="*60)
    print("TITLE PATTERN ANALYSIS")
    print("="*60)
    
    titles = nodes['title'].tolist()
    
    # Check for namespace prefixes
    namespace_prefixes = [
        'Category:', 'File:', 'Wikipedia:', 'Template:', 'Help:', 
        'Portal:', 'Draft:', 'Module:', 'MediaWiki:', 'User:', 'Talk:'
    ]
    
    print("\nNamespace prefix check:")
    for prefix in namespace_prefixes:
        count = sum(1 for t in titles if t and t.startswith(prefix))
        if count > 0:
            print(f"  ❌ {prefix:15} {count:,} pages (SHOULD BE FILTERED)")
            # Show examples
            examples = [t for t in titles if t and t.startswith(prefix)][:3]
            for ex in examples:
                print(f"      → {ex[:60]}")
    
    # Check for disambiguation pages
    disambig_count = sum(1 for t in titles if t and '(disambiguation)' in t.lower())
    print(f"\n  Disambiguation pages: {disambig_count:,}")
    
    # Check for list pages
    list_count = sum(1 for t in titles if t and t.startswith('List_of_'))
    print(f"  'List_of_' pages: {list_count:,}")
    
    # Check for identifier patterns (often meta)
    identifier_patterns = [
        (r'^[A-Z]{2,6}$', 'Short acronyms (ISBN, ISSN, ASCII...)'),
        (r'_\(identifier\)$', 'Identifier pages'),
        (r'^[A-Z][a-z]+_\(disambiguation\)$', 'Disambiguation'),
    ]
    
    print("\nPotential meta-page patterns:")
    for pattern, desc in identifier_patterns:
        matches = [t for t in titles if t and re.match(pattern, t)]
        if matches:
            print(f"\n  {desc}: {len(matches)}")
            for m in matches[:5]:
                print(f"      → {m}")


def find_citation_infrastructure(nodes):
    """Identify likely citation/reference infrastructure pages."""
    print("\n" + "="*60)
    print("CITATION INFRASTRUCTURE DETECTION")
    print("="*60)
    
    # Known patterns for citation infrastructure
    citation_keywords = [
        'identifier', 'doi', 'isbn', 'issn', 'pmid', 'arxiv', 'bibcode',
        'oclc', 'jstor', 'wayback', 'archive', 'wikidata', 'pubmed',
        'semantic_scholar', 'citeseer', 'handle_system', 'lccn', 'viaf'
    ]
    
    found = []
    for _, row in nodes.iterrows():
        title_lower = row['title'].lower() if row['title'] else ''
        for kw in citation_keywords:
            if kw in title_lower:
                found.append({
                    'title': row['title'],
                    'in_degree': row['in_degree'],
                    'out_degree': row['out_degree'],
                    'keyword': kw
                })
                break
    
    found_df = pd.DataFrame(found).drop_duplicates('title')
    found_df = found_df.sort_values('in_degree', ascending=False)
    
    print(f"\nFound {len(found_df)} potential citation infrastructure pages:\n")
    for _, row in found_df.head(30).iterrows():
        print(f"  in={row['in_degree']:>8,}  {row['title'][:50]}")
    
    return found_df


def generate_exclusion_list(nodes, output_path, 
                            min_in_degree=1000,
                            max_out_degree=10,
                            include_dangling_hubs=True):
    """Generate a recommended exclusion list."""
    print("\n" + "="*60)
    print("GENERATING EXCLUSION LIST")
    print("="*60)
    
    exclude_titles = set()
    
    # 1. High in-degree dangling nodes
    if include_dangling_hubs:
        dangling_hubs = nodes[
            (nodes['out_degree'] == 0) & 
            (nodes['in_degree'] > min_in_degree)
        ]['title'].tolist()
        exclude_titles.update(dangling_hubs)
        print(f"  Added {len(dangling_hubs)} dangling hubs (in>{min_in_degree}, out=0)")
    
    # 2. High in-degree, very low out-degree
    suspicious = nodes[
        (nodes['in_degree'] > min_in_degree * 5) & 
        (nodes['out_degree'] < max_out_degree)
    ]['title'].tolist()
    exclude_titles.update(suspicious)
    print(f"  Added {len(suspicious)} suspicious nodes (in>{min_in_degree*5}, out<{max_out_degree})")
    
    # 3. Known citation patterns
    citation_keywords = [
        'identifier', '_id)', 'isbn', 'issn', 'pmid', 'arxiv', 
        'wayback_machine', 'wikidata', 'pubmed', 'bibcode'
    ]
    for _, row in nodes.iterrows():
        if row['title']:
            title_lower = row['title'].lower()
            if any(kw in title_lower for kw in citation_keywords):
                if row['in_degree'] > 1000:  # Only if significant
                    exclude_titles.add(row['title'])
    
    # Write to file
    exclude_titles = sorted(exclude_titles)
    with open(output_path, 'w', encoding='utf-8') as f:
        for title in exclude_titles:
            if title:  # Skip None
                f.write(title + '\n')
    
    print(f"\n✅ Wrote {len(exclude_titles)} titles to {output_path}")
    print("\nTop entries in exclusion list:")
    for title in exclude_titles[:20]:
        in_deg = nodes[nodes['title'] == title]['in_degree'].values
        in_deg = in_deg[0] if len(in_deg) > 0 else 0
        print(f"    {in_deg:>8,}  {title[:50]}")
    
    return exclude_titles


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Wikipedia KG to identify nodes for filtering",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--nodes", required=True, help="Path to nodes.parquet")
    parser.add_argument("--edges", required=True, help="Path to edges.parquet")
    parser.add_argument("--output", default="exclude_titles.txt",
                        help="Output path for recommended exclusion list")
    parser.add_argument("--min-in-degree", type=int, default=1000,
                        help="Minimum in-degree threshold for exclusion")
    parser.add_argument("--top-n", type=int, default=50,
                        help="Number of top items to show in each category")
    args = parser.parse_args()
    
    # Load data
    nodes, edges = load_graph(args.nodes, args.edges)
    
    # Analyze degrees
    nodes = analyze_degree_distribution(nodes, edges)
    
    # Find problematic nodes
    find_top_hubs(nodes, args.top_n)
    find_high_indegree_dangling(nodes, args.top_n)
    find_high_indegree_low_outdegree(nodes, top_n=args.top_n)
    
    # Analyze titles
    analyze_title_patterns(nodes)
    find_citation_infrastructure(nodes)
    
    # Generate exclusion list
    generate_exclusion_list(
        nodes, 
        args.output,
        min_in_degree=args.min_in_degree
    )
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print(f"""
1. Review the generated exclusion list: {args.output}
2. Manually add/remove entries as needed
3. Run PageRank with filtering:
   
   python pagerank_improved.py \\
       --nodes {args.nodes} \\
       --edges {args.edges} \\
       --out pagerank.parquet \\
       --exclude-titles {args.output} \\
       --stats
""")


if __name__ == "__main__":
    main()