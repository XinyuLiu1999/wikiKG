"""Generate comprehensive exclusion list for Wikipedia Knowledge Graph.

This script combines:
1. Curated list of known meta-pages
2. Auto-detection by degree ratio
3. Auto-detection by title patterns

Run this BEFORE computing PageRank.

Usage:
    python generate_exclusions.py \
        --nodes data/graph/hyperlink/nodes.parquet \
        --edges data/graph/hyperlink/edges.parquet \
        --output data/graph/hyperlink/exclude_titles.txt
"""

import argparse
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm


# =============================================================================
# CURATED EXCLUSION LIST (based on your analysis)
# =============================================================================

CURATED_EXCLUSIONS = {
    # =========================================================================
    # CITATION IDENTIFIERS (Top Priority)
    # =========================================================================
    "ISBN",
    "ISSN",
    "Digital_object_identifier",
    "Bibcode",
    "ArXiv",
    "PubMed",
    "PubMed_Central",
    "PubMed_Identifier",
    "OCLC",
    "JSTOR",
    "CiteSeerX",
    "Semantic_Scholar",
    "S2CID",
    "S2CID_(identifier)",
    "PMC_(identifier)",
    "PMID_(identifier)",
    "Handle_System",
    "LCCN_(identifier)",
    "VIAF_(identifier)",
    "WorldCat",
    "Zentralblatt_MATH",
    "Mathematical_Reviews",
    "ZooBank",
    "International_Chemical_Identifier",
    "Unique_Ingredient_Identifier",
    "Location_identifier",
    
    # =========================================================================
    # WEB ARCHIVES & EXTERNAL DATABASES
    # =========================================================================
    "Wayback_Machine",
    "Internet_Archive",
    "Archive.today",
    "Wikidata",
    "Wikispecies",
    "IMDb",
    "AllMusic",
    "Discogs",
    "MusicBrainz",
    "Rotten_Tomatoes",
    "Metacritic",
    "Box_Office_Mojo",
    "Find_a_Grave",
    "Goodreads",
    "Rate_Your_Music",
    "HanCinema",
    "British_Newspaper_Archive",
    "Google_News_Archive",
    "CricketArchive",
    "MacTutor_History_of_Mathematics_Archive",
    "National_Film_and_Sound_Archive",
    "Library_and_Archives_Canada",
    "The_National_Archives_(United_Kingdom)",
    "National_Archives_and_Records_Administration",
    "20th_Century_Press_Archives",
    
    # =========================================================================
    # BIODIVERSITY DATABASES (template-linked from taxoboxes)
    # =========================================================================
    "Global_Biodiversity_Information_Facility",
    "Open_Tree_of_Life",
    "Catalogue_of_Life",
    "Interim_Register_of_Marine_and_Nonmarine_Genera",
    "INaturalist",
    "iNaturalist",
    "Encyclopedia_of_Life",
    "Integrated_Taxonomic_Information_System",
    "National_Biodiversity_Network",
    "Australian_Faunal_Directory",
    "FloraBase",
    "Plants_of_the_World_Online",
    "World_Register_of_Marine_Species",
    "Tropicos",
    "IPNI",
    "The_Plant_List",
    "GBIF",
    "ITIS",
    "NCBI",
    "National_Center_for_Biotechnology_Information",
    "BOLD_Systems",
    "Invasive_Species_Compendium",
    "IntEnz",
    "GEOnet_Names_Server",
    
    # =========================================================================
    # TAXONOMY TEMPLATES (linked from taxoboxes)
    # Only the meta-reference pages, not actual biology concepts
    # =========================================================================
    "Taxonomy_(biology)",
    "Binomial_nomenclature",
    "Synonym_(taxonomy)",
    
    # =========================================================================
    # CHARACTER ENCODING / TYPOGRAPHY (template-linked)
    # These have very high in-degree from templates but are legitimate topics
    # Only exclude if they have low out-degree in your graph
    # =========================================================================
    "ASCII",
    "Diacritic",
    "Ligature_(writing)",
    "Greek_alphabet",
    "HTML_element",
    "ISO_4",
    "Daylight_saving_time",

    
    # =========================================================================
    # GEOGRAPHIC TEMPLATES
    # =========================================================================
    "Geographic_coordinate_system",
    "GEOnet_Names_Server",
    
    # =========================================================================
    # TEMPLATE SHORTCUTS / HELP PAGES
    # =========================================================================
    "H:S",
    "H:L",
    "H:MW",
    "H:A",
    "H:B",
    "H:C",
    "H:D",
    "H:E",
    "H:F",
    "H:G",
    
    # =========================================================================
    # MUSIC CHART DATABASES
    # =========================================================================
    "Dutch_Album_Top_100",
    "Billboard_Hot_100",
    "UK_Singles_Chart",
    "ARIA_Charts",
    "Oricon",
    "Gaon_Chart",
    "Recording_Industry_Association_of_America",
    "British_Phonographic_Industry",
    
    # =========================================================================
    # CENSUS / DEMOGRAPHIC TEMPLATES  
    # =========================================================================
    "Race_and_ethnicity_in_the_United_States_census",
}


def load_data(nodes_path, edges_path):
    """Load nodes and compute degree statistics."""
    print("Loading nodes...")
    nodes = pq.read_table(nodes_path).to_pandas()
    
    print("Loading edges...")  
    edges = pq.read_table(edges_path).to_pandas()
    
    print(f"Loaded {len(nodes):,} nodes and {len(edges):,} edges")
    
    # Compute degrees
    print("Computing degree statistics...")
    in_degree = edges['dst_id'].value_counts()
    out_degree = edges['src_id'].value_counts()
    
    nodes['in_degree'] = nodes['page_id'].map(in_degree).fillna(0).astype(int)
    nodes['out_degree'] = nodes['page_id'].map(out_degree).fillna(0).astype(int)
    nodes['in_out_ratio'] = nodes['in_degree'] / (nodes['out_degree'] + 1)
    
    return nodes, edges


def detect_by_degree_ratio(nodes, min_in_degree=5000, min_ratio=100, max_out_degree=200):
    """
    Detect meta-pages by high in/out degree ratio AND low absolute out-degree.
    
    Key insight: Real articles like "United States" have high ratio but ALSO
    have thousands of outgoing links. Meta-pages have high ratio AND low out-degree.
    
    We require BOTH:
    - High in/out ratio (receives much more than it gives)
    - Low absolute out-degree (doesn't link to many pages)
    """
    print(f"\n[1] Detecting by degree ratio (in>={min_in_degree}, ratio>={min_ratio}, out<={max_out_degree})...")
    
    candidates = nodes[
        (nodes['in_degree'] >= min_in_degree) &
        (nodes['in_out_ratio'] >= min_ratio) &
        (nodes['out_degree'] <= max_out_degree)  # KEY: must have low out-degree
    ].copy()
    
    # Sort by in_degree
    candidates = candidates.sort_values('in_degree', ascending=False)
    
    print(f"    Found {len(candidates)} candidates")
    
    # Show top 20
    print("\n    Top 20 by in-degree:")
    for _, row in candidates.head(20).iterrows():
        print(f"      in={row['in_degree']:>9,}  out={row['out_degree']:>5}  ratio={row['in_out_ratio']:>7.1f}  {row['title'][:40]}")
    
    return set(candidates['title'].dropna().tolist())


def detect_by_title_patterns(nodes, min_in_degree=1000):
    """Detect meta-pages by title patterns."""
    print(f"\n[2] Detecting by title patterns (in>={min_in_degree})...")
    
    candidates = set()
    
    # Pattern 1: Identifier pages
    pattern_identifier = nodes[
        (nodes['title'].str.contains(r'_\(identifier\)$', regex=True, na=False)) &
        (nodes['in_degree'] >= min_in_degree)
    ]
    candidates.update(pattern_identifier['title'].tolist())
    print(f"    Identifier pattern: {len(pattern_identifier)}")
    
    # Pattern 2: Help shortcuts (H:X)
    pattern_help = nodes[
        (nodes['title'].str.match(r'^H:[A-Z]+$', na=False)) &
        (nodes['in_degree'] >= min_in_degree)
    ]
    candidates.update(pattern_help['title'].tolist())
    print(f"    Help shortcuts: {len(pattern_help)}")
    
    # Pattern 3: Known database keywords in high in-degree pages
    db_keywords = [
        'database', 'archive', 'registry', 'catalogue', 'catalog',
        'repository', 'identifier', 'biodiversity', 'compendium'
    ]
    
    for _, row in nodes[nodes['in_degree'] >= min_in_degree * 5].iterrows():
        if row['title'] and row['out_degree'] < 200:
            title_lower = row['title'].lower()
            if any(kw in title_lower for kw in db_keywords):
                candidates.add(row['title'])
    
    print(f"    Database keywords: {len(candidates)} total")
    
    return candidates


def detect_citation_infrastructure(nodes, min_in_degree=1000):
    """Detect citation/reference infrastructure pages."""
    print(f"\n[3] Detecting citation infrastructure (in>={min_in_degree})...")
    
    citation_keywords = [
        'identifier', 'doi', 'isbn', 'issn', 'pmid', 'arxiv', 'bibcode',
        'oclc', 'jstor', 'wayback', 'wikidata', 'pubmed', 'semantic_scholar',
        'citeseer', 'handle_system', 'lccn', 'viaf', 'worldcat', 'archive'
    ]
    
    candidates = set()
    
    for _, row in nodes[nodes['in_degree'] >= min_in_degree].iterrows():
        if row['title']:
            title_lower = row['title'].lower()
            # Must have keyword AND low out-degree (relative to in-degree)
            if any(kw in title_lower for kw in citation_keywords):
                if row['in_out_ratio'] > 10:
                    candidates.add(row['title'])
    
    print(f"    Found {len(candidates)} citation infrastructure pages")
    
    return candidates


def detect_high_indegree_dangling(nodes, min_in_degree=500):
    """Detect dangling nodes (out_degree=0) with significant in-degree."""
    print(f"\n[4] Detecting dangling hubs (in>={min_in_degree}, out=0)...")
    
    dangling = nodes[
        (nodes['out_degree'] == 0) &
        (nodes['in_degree'] >= min_in_degree)
    ]
    
    print(f"    Found {len(dangling)} dangling hubs")
    
    return set(dangling['title'].dropna().tolist())


def validate_exclusions(nodes, exclusions):
    """Validate exclusions and show statistics."""
    print("\n" + "="*70)
    print("VALIDATION")
    print("="*70)
    
    # SAFETY: Never exclude these obvious real articles
    # These are legitimate knowledge hubs that might match patterns
    NEVER_EXCLUDE = {
        # Countries
        "United_States", "France", "Germany", "United_Kingdom", "Italy",
        "Spain", "China", "Japan", "India", "Australia", "Canada", "Russia",
        "Brazil", "Mexico", "South_Korea", "Netherlands", "Sweden", "Poland",
        
        # Major topics
        "World_War_II", "World_War_I", "Animal", "Plant", "Human",
        "Association_football", "Baseball", "Basketball", "Cricket",
        "Christianity", "Islam", "Buddhism", "Hinduism", "Judaism",
        "Olympic_Games", "FIFA_World_Cup",
        
        # Major cities
        "New_York_City", "London", "Paris", "Tokyo", "Beijing", "Berlin",
        "Los_Angeles", "Chicago", "Sydney", "Toronto", "Mumbai",
        
        # Entertainment
        "The_New_York_Times", "BBC", "CNN", "Netflix", "YouTube",
        
        # Lists that are actually useful
        "List_of_sovereign_states",
    }
    
    protected = exclusions & NEVER_EXCLUDE
    if protected:
        print(f"\n🛡️  Removing {len(protected)} protected articles from exclusions:")
        for title in sorted(protected):
            print(f"      - {title}")
        exclusions = exclusions - NEVER_EXCLUDE
    
    # Check which curated exclusions exist in the graph
    existing = exclusions & set(nodes['title'].dropna())
    missing = exclusions - existing
    
    if missing:
        print(f"\n⚠️  {len(missing)} exclusions not found in graph (may be redirects or typos):")
        for title in sorted(missing)[:20]:
            print(f"      - {title}")
        if len(missing) > 20:
            print(f"      ... and {len(missing) - 20} more")
    
    # Statistics on excluded nodes
    excluded_nodes = nodes[nodes['title'].isin(existing)]
    
    total_in_degree = excluded_nodes['in_degree'].sum()
    total_edges = nodes['in_degree'].sum()
    
    print(f"\n📊 Exclusion Statistics:")
    print(f"    Total exclusions: {len(existing):,}")
    print(f"    Total incoming edges to excluded nodes: {total_in_degree:,}")
    print(f"    Percentage of all edges: {total_in_degree / total_edges * 100:.2f}%")
    
    # Top excluded by in-degree
    print(f"\n🔝 Top 30 excluded nodes by in-degree:")
    top_excluded = excluded_nodes.sort_values('in_degree', ascending=False).head(30)
    for _, row in top_excluded.iterrows():
        print(f"      {row['in_degree']:>10,}  {row['title'][:50]}")
    
    return existing


def verify_top_hubs_after_exclusion(nodes, exclusions, top_n=30):
    """Show what the top hubs will look like after exclusion."""
    print("\n" + "="*70)
    print(f"TOP {top_n} HUBS AFTER EXCLUSION (preview)")
    print("="*70)
    
    remaining = nodes[~nodes['title'].isin(exclusions)]
    top_remaining = remaining.sort_values('in_degree', ascending=False).head(top_n)
    
    print("\nThese should be REAL knowledge centers:\n")
    for i, (_, row) in enumerate(top_remaining.iterrows(), 1):
        marker = "✅" if row['out_degree'] > 200 else "⚠️"
        print(f"  {i:2}. {marker} in={row['in_degree']:>9,}  out={row['out_degree']:>5,}  {row['title'][:45]}")


def save_exclusion_list(exclusions, nodes, output_path):
    """Save exclusion list sorted by in-degree."""
    print(f"\n💾 Saving exclusion list to {output_path}...")
    
    # Get in-degree for sorting
    title_to_indegree = dict(zip(nodes['title'], nodes['in_degree']))
    
    # Sort by in-degree (highest first)
    sorted_exclusions = sorted(
        [t for t in exclusions if t],  # Filter None
        key=lambda x: title_to_indegree.get(x, 0),
        reverse=True
    )
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Wikipedia Knowledge Graph - Exclusion List\n")
        f.write(f"# Total: {len(sorted_exclusions)} pages\n")
        f.write("# Generated by generate_exclusions.py\n")
        f.write("#\n")
        f.write("# These are meta-pages, citation infrastructure, and database references\n")
        f.write("# that should be excluded before computing PageRank.\n")
        f.write("#\n\n")
        
        for title in sorted_exclusions:
            in_deg = title_to_indegree.get(title, 0)
            f.write(f"{title}\n")
    
    print(f"✅ Saved {len(sorted_exclusions)} exclusions")


def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive exclusion list for PageRank",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--nodes", required=True, help="Path to nodes.parquet")
    parser.add_argument("--edges", required=True, help="Path to edges.parquet")
    parser.add_argument("--output", required=True, help="Output path for exclusion list")
    parser.add_argument("--min-in-degree", type=int, default=5000,
                       help="Minimum in-degree for auto-detection")
    parser.add_argument("--min-ratio", type=float, default=100,
                       help="Minimum in/out ratio for auto-detection")
    parser.add_argument("--max-out-degree", type=int, default=200,
                       help="Maximum out-degree for ratio-based detection (higher = real article)")
    parser.add_argument("--no-curated", action="store_true",
                       help="Skip curated exclusions (auto-detect only)")
    args = parser.parse_args()
    
    # Load data
    nodes, edges = load_data(args.nodes, args.edges)
    
    # Collect exclusions from all sources
    all_exclusions = set()
    
    # 1. Curated list
    if not args.no_curated:
        print(f"\n[0] Using curated exclusion list: {len(CURATED_EXCLUSIONS)} entries")
        all_exclusions.update(CURATED_EXCLUSIONS)
    
    # 2. Degree ratio detection
    ratio_exclusions = detect_by_degree_ratio(
        nodes, 
        min_in_degree=args.min_in_degree,
        min_ratio=args.min_ratio,
        max_out_degree=args.max_out_degree
    )
    all_exclusions.update(ratio_exclusions)
    
    # 3. Title pattern detection
    pattern_exclusions = detect_by_title_patterns(nodes, min_in_degree=1000)
    all_exclusions.update(pattern_exclusions)
    
    # 4. Citation infrastructure detection
    citation_exclusions = detect_citation_infrastructure(nodes, min_in_degree=1000)
    all_exclusions.update(citation_exclusions)
    
    # 5. Dangling hubs
    dangling_exclusions = detect_high_indegree_dangling(nodes, min_in_degree=500)
    all_exclusions.update(dangling_exclusions)
    
    # Validate and show statistics
    print("\n" + "="*70)
    print(f"TOTAL EXCLUSIONS: {len(all_exclusions)}")
    print("="*70)
    
    validated = validate_exclusions(nodes, all_exclusions)
    
    # Preview top hubs after exclusion
    verify_top_hubs_after_exclusion(nodes, all_exclusions)
    
    # Save
    save_exclusion_list(validated, nodes, args.output)
    
    # Print next steps
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print(f"""
1. Review the exclusion list if needed:
   head -50 {args.output}

2. Run PageRank with filtering:
   python pagerank_improved.py \\
       --nodes {args.nodes} \\
       --edges {args.edges} \\
       --out pagerank.parquet \\
       --exclude-titles {args.output} \\
       --stats
""")


if __name__ == "__main__":
    main()