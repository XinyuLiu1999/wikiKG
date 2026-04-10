import argparse
import pandas as pd
import pyarrow.parquet as pq
import scipy.stats as stats
import sys

def get_pagerank_with_percentile(nodes_path, pagerank_path, target_title):
    print(f"--- Querying: {target_title} ---")

    # 1. Load nodes and find the ID
    nodes = pq.read_table(nodes_path, columns=["page_id", "title"]).to_pandas()
    target_clean = target_title.replace(" ", "_")
    match = nodes[nodes['title'] == target_clean]

    if match.empty:
        print(f"Error: Title '{target_title}' not found.")
        return

    page_id = int(match.iloc[0]['page_id'])
    
    # 2. Load PageRank scores
    pr_table = pq.read_table(pagerank_path)
    pr_scores = pr_table.column("pagerank").to_numpy()
    page_ids = pr_table.column("page_id").to_numpy()
    
    # 3. Find the specific score for our ID
    try:
        idx = (page_ids == page_id).argmax()
        target_score = pr_scores[idx]
    except Exception:
        print(f"Error: Page ID {page_id} has no calculated score.")
        return

    # 4. Calculate Percentile and Rank
    # We use 'weak' kind to say "what % of pages are <= this score"
    percentile = stats.percentileofscore(pr_scores, target_score, kind='weak')
    
    # Calculate numerical rank (1st, 2nd, etc.)
    # Higher scores are better, so we count how many are strictly greater
    rank = (pr_scores > target_score).sum() + 1
    total_pages = len(pr_scores)

    # Output Results
    print(f"Page ID:      {page_id}")
    print(f"PR Score:     {target_score:.12f}")
    print(f"Global Rank:  #{rank:,} of {total_pages:,}")
    print(f"Percentile:   {percentile:.4f}%")
    
    if percentile > 99:
        print("Status:       Elite (Top 1% of the graph)")
    elif percentile > 90:
        print("Status:       Highly Influential")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", required=True)
    parser.add_argument("--pagerank", required=True)
    parser.add_argument("--title", required=True)
    args = parser.parse_args()
    
    get_pagerank_with_percentile(args.nodes, args.pagerank, args.title)