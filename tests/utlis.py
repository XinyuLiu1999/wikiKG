import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
import os
from tqdm import tqdm


# Patterns indicating non-semantic categories
EXCLUDE_PATTERNS = [
    # Administrative/maintenance
    "_stubs",
    "Wikipedia_",
    "Articles_",
    "Pages_",
    "Redirects_",
    "WikiProject_",
    "User_",
    "Template_",
    "All_",
    "CS1_",
    "Webarchive_",
    "Use_dmy_dates",
    "Use_mdy_dates",
    
    # Lists and collections
    "Lists_of_",
    "List_of_",
    
    # Geographic/temporal slicing
    "_by_country",
    "_by_continent",
    "_by_region",
    "_by_year",
    "_by_decade",
    "_by_century",
    "_by_nationality",
    "_by_origin",
    
    # Individual instances (not concepts)
    "Individual_",
    
    # Cultural/media (often tangential)
    "_in_popular_culture",
    "_in_fiction",
    "_in_literature",
    "_in_film",
    "_in_television",
    "_in_art",
    "_in_music",
    "_in_video_games",
    "Fictional_",
    "Films_about_",
    "Books_about_",
    "Songs_about_",
    "Television_shows_about_",
    "Video_games_about_",
    "Comics_about_",
    
    # Organizational
    "_organizations",
    "_professionals",
    "_people",
    "_companies",
    "_awards",
]


def is_semantic_category(title):
    """Check if a category title represents a semantic concept."""
    for pattern in EXCLUDE_PATTERNS:
        if pattern in title:
            return False
    return True


def extract_semantic_subgraph(
    out_dir, 
    output_dir, 
    root_title="Main_topic_classifications", 
    apply_pattern_filter=True,
    remove_isolated=True,
    min_pages_for_leaf=10,
):
    """
    Extract a semantic subgraph rooted at root_title.
    
    Args:
        out_dir: Input directory with wiki_*.parquet files
        output_dir: Output directory for filtered files
        root_title: Root category to start from
        apply_pattern_filter: Remove categories matching non-semantic patterns
        remove_isolated: Remove leaf categories with few pages
        min_pages_for_leaf: Minimum pages required for a leaf category to be kept
    """
    # Load the data
    print("Loading categories...")
    categories = pq.read_table(f"{out_dir}/wiki_categories.parquet").to_pandas()
    print(f"  Loaded {len(categories)} categories")
    
    print("Loading edges...")
    edges = pq.read_table(f"{out_dir}/wiki_category_edges.parquet").to_pandas()
    print(f"  Loaded {len(edges)} edges")
    
    print("Loading page-category relationships...")
    page_cats = pq.read_table(f"{out_dir}/wiki_page_categories.parquet").to_pandas()
    print(f"  Loaded {len(page_cats)} page-category links")
    
    # Step 1: Pattern-based filtering
    if apply_pattern_filter:
        print("\n[Step 1] Applying pattern-based filtering...")
        original_count = len(categories)
        semantic_mask = categories["title"].apply(is_semantic_category)
        categories = categories[semantic_mask]
        semantic_ids = set(categories["category_id"])
        print(f"  Removed {original_count - len(categories)} non-semantic categories")
        print(f"  Remaining: {len(categories)} categories")
        
        # Filter edges to only include semantic categories
        original_edge_count = len(edges)
        edges = edges[
            edges["parent_id"].isin(semantic_ids) & 
            edges["child_id"].isin(semantic_ids)
        ]
        print(f"  Removed {original_edge_count - len(edges)} edges")
    else:
        print("\n[Step 1] Skipping pattern-based filtering")
    
    # Step 2: BFS to find reachable categories from root
    print("\n[Step 2] Finding categories reachable from root...")
    title_to_id = dict(zip(categories["title"], categories["category_id"]))
    
    # Build children adjacency
    children_of = defaultdict(list)
    for p, c in edges[["parent_id", "child_id"]].itertuples(index=False):
        children_of[p].append(c)
    
    root_id = title_to_id.get(root_title)
    if root_id is None:
        raise ValueError(f"Category '{root_title}' not found")
    print(f"  Root: '{root_title}' (id: {root_id})")
    
    reachable = {root_id}
    queue = deque([root_id])
    
    with tqdm(desc="  BFS traversal", unit=" nodes") as pbar:
        while queue:
            parent_id = queue.popleft()
            for child_id in children_of.get(parent_id, []):
                if child_id not in reachable:
                    reachable.add(child_id)
                    queue.append(child_id)
                    pbar.update(1)
    
    print(f"  Found {len(reachable)} reachable categories")
    
    # Filter to reachable
    categories = categories[categories["category_id"].isin(reachable)]
    edges = edges[
        edges["parent_id"].isin(reachable) & 
        edges["child_id"].isin(reachable)
    ]
    
    # Step 3: Remove isolated leaf categories
    if remove_isolated:
        print(f"\n[Step 3] Removing isolated leaf categories (min_pages={min_pages_for_leaf})...")
        
        # Iterate until no more removals (removing a leaf might create new leaves)
        iteration = 0
        while True:
            iteration += 1
            
            # Rebuild children_of with current edges
            children_of = defaultdict(list)
            parents_of = defaultdict(list)
            for p, c in edges[["parent_id", "child_id"]].itertuples(index=False):
                children_of[p].append(c)
                parents_of[c].append(p)
            
            # Count pages per category
            current_cat_ids = set(categories["category_id"])
            page_counts = (
                page_cats[page_cats["category_id"].isin(current_cat_ids)]
                .groupby("category_id")
                .size()
                .to_dict()
            )
            
            # Find isolated leaves: no children AND below page threshold
            to_remove = set()
            for cat_id in current_cat_ids:
                has_children = len(children_of.get(cat_id, [])) > 0
                page_count = page_counts.get(cat_id, 0)
                
                if not has_children and page_count < min_pages_for_leaf:
                    # Don't remove root
                    if cat_id != root_id:
                        to_remove.add(cat_id)
            
            if not to_remove:
                print(f"  Iteration {iteration}: No more isolated categories to remove")
                break
            
            print(f"  Iteration {iteration}: Removing {len(to_remove)} isolated categories")
            
            # Remove from categories and edges
            categories = categories[~categories["category_id"].isin(to_remove)]
            edges = edges[
                ~edges["parent_id"].isin(to_remove) & 
                ~edges["child_id"].isin(to_remove)
            ]
        
        print(f"  Final count after isolation removal: {len(categories)} categories")
    else:
        print("\n[Step 3] Skipping isolated category removal")
    
    # Step 4: Filter page-category relationships
    print("\n[Step 4] Filtering page-category links...")
    final_cat_ids = set(categories["category_id"])
    filtered_page_cats = page_cats[page_cats["category_id"].isin(final_cat_ids)]
    
    # Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"  Categories: {len(categories)}")
    print(f"  Edges: {len(edges)}")
    print(f"  Page-category links: {len(filtered_page_cats)}")
    
    # Compute some stats
    children_of = defaultdict(list)
    for p, c in edges[["parent_id", "child_id"]].itertuples(index=False):
        children_of[p].append(c)
    
    leaf_count = sum(1 for cat_id in final_cat_ids if len(children_of.get(cat_id, [])) == 0)
    internal_count = len(categories) - leaf_count
    print(f"  Leaf categories: {leaf_count}")
    print(f"  Internal categories: {internal_count}")
    
    # Save filtered data
    print("\n[Step 5] Saving filtered data...")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"  Writing semantic_categories.parquet...")
    pq.write_table(
        pa.Table.from_pandas(categories, preserve_index=False),
        f"{output_dir}/semantic_categories.parquet"
    )
    
    print(f"  Writing semantic_category_edges.parquet...")
    pq.write_table(
        pa.Table.from_pandas(edges, preserve_index=False),
        f"{output_dir}/semantic_category_edges.parquet"
    )
    
    print(f"  Writing semantic_page_categories.parquet...")
    pq.write_table(
        pa.Table.from_pandas(filtered_page_cats, preserve_index=False),
        f"{output_dir}/semantic_page_categories.parquet"
    )
    
    print("\nDone!")
    return final_cat_ids


class CategoryIndex:
    """Pre-built index for fast category queries."""
    
    def __init__(self, data_dir):
        print(f"Loading data from {data_dir}...")
        self.nodes = pd.read_parquet(f"{data_dir}/semantic_categories.parquet")
        self.edges = pd.read_parquet(f"{data_dir}/semantic_category_edges.parquet")
        
        print("Building index...")
        self.id_to_title = self.nodes.set_index("category_id")["title"].to_dict()
        self.title_to_id = {v: k for k, v in self.id_to_title.items()}
        
        self.children = defaultdict(list)
        self.parents = defaultdict(list)
        
        for p, c in self.edges[["parent_id", "child_id"]].itertuples(index=False):
            self.children[p].append(c)
            self.parents[c].append(p)
        
        print(f"Index ready: {len(self.nodes)} categories, {len(self.edges)} edges")
    
    def query(self, category_title, depth=3):
        """Query a category and show its neighborhood."""
        search_term = category_title.replace(" ", "_")
        matches = self.nodes[self.nodes["title"].str.lower() == search_term.lower()]
        
        if matches.empty:
            print(f"No categories found matching: '{category_title}'")
            # Suggest similar
            similar = [t for t in self.title_to_id.keys() if search_term.lower() in t.lower()]
            if similar:
                print(f"Did you mean: {similar[:10]}")
            return
        
        for _, row in matches.iterrows():
            cat_id = int(row["category_id"])
            print("\n" + "=" * 60)
            print(f"CATEGORY: {row['title']} (ID: {cat_id})")
            print(f"Direct pages: {row.get('page_count', 'N/A')}")
            print("=" * 60)
            
            self._print_tree(cat_id, self.parents, depth, "up")
            # self._print_tree(cat_id, self.children, depth, "down")
    
    def _print_tree(self, start_id, adj, depth, direction):
        """Print tree using simple recursion with global seen set."""
        seen = set()
        
        def rec(node, lvl):
            if node in seen:
                return
            seen.add(node)
            
            indent = "  " * lvl
            title = self.id_to_title.get(node, f"[Unknown_{node}]")
            prefix = "└── " if lvl > 0 else ""
            print(f"{indent}{prefix}{title}")
            
            if lvl >= depth:
                return
            
            for nxt in adj.get(node, []):
                rec(nxt, lvl + 1)
        
        print(f"\n[{direction.upper()} TREE]")
        rec(start_id, 0)


@dataclass
class SemanticTreeConfig:
    """Configuration for semantic tree building."""
    
    k_parents: int = 2
    root_title: str = "Main_topic_classifications"
    pattern_penalty_weight: int = 10
    domain_stickiness: Dict[str, List[str]] = field(default_factory=lambda: {
        # Countries should stay in country hierarchy
        "countries": ["countries", "Countries"],
        
        # Animals should stay in animal hierarchy
        "animals": ["animals", "Animals", "fauna"],
        
        # Plants should stay in plant hierarchy  
        "plants": ["plants", "Plants", "flora", "Flora"],
        
        # Films should stay in film hierarchy
        "films": ["films", "Films", "cinema", "Cinema"],
        
        # Books should stay in book/literature hierarchy
        "books": ["books", "Books", "literature", "Literature"],
        
        # Music should stay in music hierarchy
        "music": ["music", "Music", "musical"],
        "musicians": ["music", "Music", "musicians"],
        "songs": ["music", "Music", "songs"],
        "albums": ["music", "Music", "albums"],
        
        # Sports should stay in sports hierarchy
        "sports": ["sports", "Sports"],
        "athletes": ["sports", "Sports", "athletes"],
        "players": ["sports", "Sports", "players"],
        
        # Science subcategories
        "physics": ["physics", "Physics", "science", "Science"],
        "chemistry": ["chemistry", "Chemistry", "science", "Science"],
        "biology": ["biology", "Biology", "science", "Science"],
        
        # Technology subcategories
        "software": ["software", "Software", "computing", "Computing", "technology"],
        "programming": ["programming", "Programming", "computing", "Computing"],
        "computers": ["computer", "Computer", "computing", "Computing"],
    })
    
    # Stickiness bonus (negative penalty = bonus)
    domain_stickiness_bonus: int = 5
    
    preferred_anchors: Set[str] = field(default_factory=lambda: {
        # Natural world
        "Animals", "Plants", "Fungi", "Organisms", "Life", "Nature", 
        "Biology", "Ecology",
        
        # Science & Technology
        "Science", "Technology", "Mathematics", "Medicine", "Engineering", 
        "Computing", "Computer_science",
        
        # Human world
        "People", "Society", "Culture", "History", "Geography", "Places",
        "Countries", "Nation",
        
        # Arts & Knowledge
        "Arts", "Music", "Literature", "Philosophy", "Religion", "Education",
        
        # Applied domains
        "Sports", "Games", "Food", "Food_and_drink", "Business", 
        "Economics", "Politics", "Law", "Agriculture",
        
        # Objects & Concepts
        "Objects", "Concepts", "Events", "Activities",
    })
    
    domain_anchors: Dict[str, Set[str]] = field(default_factory=lambda: {
        # Living things
        "Animals": {"Animals", "Organisms", "Life", "Biology", "Nature"},
        "Plants": {"Plants", "Organisms", "Life", "Biology", "Nature"},
        "Fungi": {"Fungi", "Organisms", "Life", "Biology", "Nature"},
        
        # Sciences
        "Science": {"Science", "Nature", "Knowledge"},
        "Physics": {"Physics", "Physical_sciences", "Natural_sciences", "Science"},
        "Chemistry": {"Chemistry", "Physical_sciences", "Natural_sciences", "Science"},
        "Biology": {"Biology", "Life_sciences", "Natural_sciences", "Science"},
        
        # Technology
        "Technology": {"Technology", "Science", "Computing", "Computer_science", "Engineering"},
        "Computing": {"Computing", "Computer_science", "Technology", "Science"},
        
        # Culture/Society
        "Culture": {"Culture", "Society", "Humanities"},
        "Arts": {"Arts", "Culture", "Humanities"},
        "Society": {"Society", "Culture"},
        "Sports": {"Sports", "Culture", "Activities", "Recreation"},
        
        # Geography
        "Geography": {"Geography", "Places", "Earth"},
        "Countries": {"Countries", "Geography", "Places", "Nation", "Society"},
        
        # Food
        "Food": {"Food", "Food_and_drink", "Plants", "Culture"},
    })
    
    domain_keywords: Dict[str, str] = field(default_factory=lambda: {
        # Animals
        "animal": "Animals", "mammal": "Animals", "bird": "Animals",
        "fish": "Animals", "reptile": "Animals", "amphibian": "Animals",
        "insect": "Animals", "canid": "Animals", "feline": "Animals",
        "dog": "Animals", "cat": "Animals", "horse": "Animals",
        "elephant": "Animals", "whale": "Animals", "dolphin": "Animals",
        "domesticated": "Animals", "livestock": "Animals", "pet": "Animals",
        "vertebrate": "Animals", "invertebrate": "Animals",
        "primate": "Animals", "rodent": "Animals", "carnivore": "Animals",
        
        # Plants
        "plant": "Plants", "flower": "Plants", "tree": "Plants",
        "flora": "Plants", "botan": "Plants", "shrub": "Plants",
        "vegetables": "Edible_plants", "fruit": "Plants", "crop": "Plants",
        "herb": "Plants", "grass": "Plants", "fern": "Plants",
        "seed": "Plants", "leaf": "Plants", "root": "Plants",
        
        # Fungi
        "fung": "Fungi", "mushroom": "Fungi", "yeast": "Fungi",
        
        # Technology
        "machine_learning": "Technology",
        "comput": "Technology", "software": "Technology",
        "hardware": "Technology", "programming": "Technology",
        "algorithm": "Technology", "internet": "Technology",
        "digital": "Technology", "robot": "Technology",
        "automat": "Technology", "cyber": "Technology",
        "neural_network": "Technology", "deep_learning": "Technology",
        "data_science": "Technology", "encryption": "Technology",
        
        # Sciences
        "science": "Science", "scientific": "Science",
        "physics": "Physics", "physical": "Physics",
        "chemi": "Chemistry", "biolog": "Biology",
        "genetic": "Biology", "quantum": "Physics",
        "mathem": "Science", "statistic": "Science",
        
        # Geography/Countries
        "countr": "Countries", "republic": "Countries",
        "kingdom": "Countries", "nation": "Countries",
        "federal": "Countries", "sovereign": "Countries",
        "territor": "Geography", "continent": "Geography",
        "island": "Geography", "region": "Geography",
        
        # Culture/Arts
        "culture": "Culture", "cultural": "Culture",
        "art": "Arts", "artist": "Arts",
        "music": "Arts", "musician": "Arts",
        "literature": "Arts", "literary": "Arts",
        "paint": "Arts", "sculpt": "Arts",
        "film": "Arts", "cinema": "Arts",
        "theat": "Arts", "drama": "Arts",
        
        # Sports
        "sport": "Sports", "athlet": "Sports",
        "olymp": "Sports", "championship": "Sports",
        "team": "Sports", "player": "Sports",
        "football": "Sports", "basketball": "Sports",
        "tennis": "Sports", "golf": "Sports",
        
        # Food
        "food": "Food", "cuisine": "Food",
        "dish": "Food", "meal": "Food",
        "cook": "Food", "recipe": "Food",
        "beverage": "Food", "drink": "Food",
    })
    
    penalized_patterns: List[str] = field(default_factory=lambda: [
        # Historical/temporal shortcuts
        "History_of_", "Historical_", "Timeline_of_", "Chronology_of_",
        
        # Cross-cutting slices
        "_in_culture", "_in_fiction", "_in_art", "_in_literature",
        "_in_film", "_in_music", "_in_religion", "_in_mythology",
        "_in_popular_culture",
        
        # Geographic/demographic slices
        "_by_country", "_by_continent", "_by_region",
        "_by_nationality", "_by_ethnicity", "_by_location",
        
        # Temporal slices
        "_by_year", "_by_decade", "_by_century", "_by_period",
        
        # Meta/organizational
        "_terminology", "_concepts", "_stubs",
        "Wikipedia_", "Articles_", "Pages_",
        
        # Tangential connections
        "_and_society", "_and_culture", "_and_politics", "_and_religion",
        
        # Taxonomic ranks
        "oidea", "idae", "inae", "iformes", "morpha", "phyta",
        "mycota", "opsida", "ales", "aceae",
        "_by_classification", "_by_taxonomy", "Taxonomic_", "Taxa_",
        
        # Philosophical/abstract shortcuts
        "Personhood", "Artificial_objects", "Human_activities",
        "Human_behavior", "Abstract_", "Abstraction",
        
        # Organizational shortcuts
        "_by_type", "_state_types", "Constitutional_", "Places_by_",
        "Concepts_in_", "Theories_of_", "Philosophy_of_",
        
        # Study-of patterns
        "Ethnobiology", "Ethno", "Study_of_",
    ])
    
    # Penalty weight for bad patterns
    pattern_penalty_weight: int = 10


class SemanticTreeBuilder:
    """
    Builds a semantic tree/DAG from Wikipedia category graph.
    
    Supports keeping top-k parents per node to create either:
    - k=1: A strict tree
    - k>=2: A DAG with limited fan-in
    """
    
    def __init__(self, config: SemanticTreeConfig = None):
        self.config = config or SemanticTreeConfig()
        
        # Data storage
        self.categories: Optional[pd.DataFrame] = None
        self.edges: Optional[pd.DataFrame] = None
        self.page_cats: Optional[pd.DataFrame] = None
        
        # Lookups (built during load)
        self.id_to_title: Dict[int, str] = {}
        self.title_to_id: Dict[str, int] = {}
        self.root_id: Optional[int] = None
        
        # Graph structures
        self.children_of: Dict[int, List[int]] = defaultdict(list)
        self.parents_of: Dict[int, List[int]] = defaultdict(list)
        
        # Computed during pruning
        self.depths: Dict[int, int] = {}
        self.anchor_ids: Set[int] = set()
        self.domain_anchor_ids: Dict[str, Set[int]] = {}
        self.best_paths: Dict[int, List[List[int]]] = {}  # node -> list of k best paths
    
    def load(self, input_dir: str) -> 'SemanticTreeBuilder':
        """Load data from parquet files."""
        print("Loading data...")
        
        self.categories = pd.read_parquet(f"{input_dir}/semantic_categories.parquet")
        self.edges = pd.read_parquet(f"{input_dir}/semantic_category_edges.parquet")
        self.page_cats = pd.read_parquet(f"{input_dir}/semantic_page_categories.parquet")
        
        print(f"  Categories: {len(self.categories)}")
        print(f"  Edges: {len(self.edges)}")
        print(f"  Page-category links: {len(self.page_cats)}")
        
        self._build_lookups()
        return self
    
    def _build_lookups(self):
        """Build internal lookup structures."""
        print("Building lookups...")
        
        # ID <-> Title mappings
        self.id_to_title = dict(zip(self.categories["category_id"], self.categories["title"]))
        self.title_to_id = {v: k for k, v in self.id_to_title.items()}
        
        # Find root
        self.root_id = self.title_to_id.get(self.config.root_title)
        if self.root_id is None:
            raise ValueError(f"Root category '{self.config.root_title}' not found")
        print(f"  Root: '{self.config.root_title}' (id: {self.root_id})")
        
        # Build adjacency lists
        self.children_of = defaultdict(list)
        self.parents_of = defaultdict(list)
        for p, c in self.edges[["parent_id", "child_id"]].itertuples(index=False):
            self.children_of[p].append(c)
            self.parents_of[c].append(p)
        
        # Build anchor ID sets
        self.anchor_ids = {
            self.title_to_id[a] 
            for a in self.config.preferred_anchors 
            if a in self.title_to_id
        }
        print(f"  Anchor categories found: {len(self.anchor_ids)}")
        
        # Build domain anchor ID sets
        self.domain_anchor_ids = {}
        for domain, anchors in self.config.domain_anchors.items():
            self.domain_anchor_ids[domain] = {
                self.title_to_id[a] 
                for a in anchors 
                if a in self.title_to_id
            }
    
    def _compute_depths(self):
        """Compute shortest path depths from root via BFS."""
        print("Computing depths from root...")
        
        self.depths = {self.root_id: 0}
        queue = deque([self.root_id])
        
        while queue:
            node = queue.popleft()
            for child in self.children_of[node]:
                if child not in self.depths:
                    self.depths[child] = self.depths[node] + 1
                    queue.append(child)
        
        print(f"  Computed depths for {len(self.depths)} nodes")
        print(f"  Max depth: {max(self.depths.values()) if self.depths else 0}")
    
    def _has_penalized_pattern(self, title: str) -> bool:
        """Check if a category title contains a penalized pattern."""
        for pattern in self.config.penalized_patterns:
            if pattern in title:
                return True
        return False
    
    def _infer_domain(self, title: str) -> Optional[str]:
        """Infer which domain a category likely belongs to based on its title."""
        title_lower = title.lower()
        for keyword, domain in self.config.domain_keywords.items():
            if keyword in title_lower:
                return domain
        return None
    
    def _compute_path_penalty(self, path: List[int]) -> int:
        """Compute penalty for a path based on penalized patterns."""
        penalty = 0
        for node_id in path:
            title = self.id_to_title.get(node_id, "")
            if self._has_penalized_pattern(title):
                penalty += self.config.pattern_penalty_weight
        return penalty
    
    def _path_has_domain_anchor(self, path: List[int], domain: Optional[str]) -> bool:
        """Check if path contains an anchor for the given domain."""
        if domain is None or domain not in self.domain_anchor_ids:
            return False
        return len(set(path).intersection(self.domain_anchor_ids[domain])) > 0
    
    def _count_anchors_in_path(self, path: List[int]) -> int:
        """Count how many anchor categories are in the path."""
        return len(set(path).intersection(self.anchor_ids))
    
    def _score_path(self, path: List[int], node_domain: Optional[str]) -> Tuple:
        """
        Score a path (lower is better).
        Returns tuple for lexicographic comparison.
        """
        penalty = self._compute_path_penalty(path)
        has_domain = 0 if self._path_has_domain_anchor(path, node_domain) else 1
        anchor_count = -self._count_anchors_in_path(path)  # Negative: more is better
        first_is_anchor = 0 if (path and path[0] in self.anchor_ids) else 1
        length = len(path)
        
        return (penalty, has_domain, anchor_count, first_is_anchor, length)
    
    def _get_top_k_paths(
        self, 
        candidate_paths: List[Tuple[int, List[int], Tuple]], 
        k: int
    ) -> List[Tuple[int, List[int]]]:
        """
        Get top-k paths from candidates, ensuring diversity.
        
        Returns list of (parent_id, full_path) tuples.
        """
        # Sort by score
        candidate_paths.sort(key=lambda x: x[2])
        
        # Take top k
        result = []
        seen_parents = set()
        
        for parent, path, score in candidate_paths:
            if parent not in seen_parents:
                result.append((parent, path))
                seen_parents.add(parent)
                if len(result) >= k:
                    break
        
        return result
    
    def prune(self) -> 'SemanticTreeBuilder':
        """
        Prune the DAG to keep top-k parents per node.
        
        Selection criteria (in order of priority):
        1. Avoid parents whose path contains penalized patterns
        2. Prefer parents whose path goes through the inferred domain anchor
        3. Prefer parents whose path goes through any preferred anchor categories
        4. Tiebreaker: shorter path to root
        """
        k = self.config.k_parents
        print(f"\nPruning to top-{k} semantic parents per node...")
        
        # Compute depths first
        self._compute_depths()
        
        # Initialize best paths for root
        self.best_paths = {self.root_id: [[self.root_id]]}
        
        # Process nodes in depth order (parents before children)
        nodes_by_depth = sorted(self.depths.keys(), key=lambda n: self.depths[n])
        
        stats = {
            "single_parent": 0,
            "penalized_avoided": 0,
            "domain_preferred": 0,
            "anchor_preferred": 0,
            "depth_tiebreak": 0,
        }
        
        for node in tqdm(nodes_by_depth[1:], desc="  Selecting best paths"):
            parents = self.parents_of.get(node, [])
            
            # Only consider parents that have paths computed
            valid_parents = [p for p in parents if p in self.best_paths]
            
            if not valid_parents:
                continue
            
            # Infer domain for this node
            node_title = self.id_to_title.get(node, "")
            node_domain = self._infer_domain(node_title)
            
            # Generate all candidate paths (through each parent's best paths)
            candidates = []
            for parent in valid_parents:
                for parent_path in self.best_paths[parent]:
                    candidate_path = [node] + parent_path
                    score = self._score_path(candidate_path, node_domain)
                    candidates.append((parent, candidate_path, score))
            
            # Select top-k paths
            top_k = self._get_top_k_paths(candidates, k)
            self.best_paths[node] = [path for _, path in top_k]
            
            # Track stats
            if len(valid_parents) == 1:
                stats["single_parent"] += 1
            elif len(candidates) > 1:
                best_score = candidates[0][2] if candidates else None
                # Find first candidate with different parent
                for _, _, score in candidates[1:]:
                    if score != best_score:
                        if best_score[0] < score[0]:
                            stats["penalized_avoided"] += 1
                        elif best_score[1] < score[1]:
                            stats["domain_preferred"] += 1
                        elif best_score[2] < score[2]:
                            stats["anchor_preferred"] += 1
                        else:
                            stats["depth_tiebreak"] += 1
                        break
        
        # Build pruned edges from best paths
        print("  Extracting parent choices...")
        kept_edges = set()
        for node, paths in self.best_paths.items():
            for path in paths:
                if len(path) > 1:
                    parent = path[1]
                    kept_edges.add((parent, node))
        
        self.edges = pd.DataFrame(list(kept_edges), columns=["parent_id", "child_id"])
        
        # Rebuild adjacency after pruning
        self.children_of = defaultdict(list)
        self.parents_of = defaultdict(list)
        for p, c in self.edges[["parent_id", "child_id"]].itertuples(index=False):
            self.children_of[p].append(c)
            self.parents_of[c].append(p)
        
        print(f"\n  Selection statistics:")
        print(f"    Single parent (no choice): {stats['single_parent']}")
        print(f"    Avoided penalized paths: {stats['penalized_avoided']}")
        print(f"    Preferred domain paths: {stats['domain_preferred']}")
        print(f"    Preferred anchor paths: {stats['anchor_preferred']}")
        print(f"    Depth tiebreaker: {stats['depth_tiebreak']}")
        print(f"  Pruned edges: {len(self.edges)}")
        
        return self
    
    def verify_connectivity(self) -> 'SemanticTreeBuilder':
        """Remove categories not reachable from root."""
        print("\nVerifying connectivity from root...")
        
        reachable = {self.root_id}
        queue = deque([self.root_id])
        
        while queue:
            node = queue.popleft()
            for child in self.children_of[node]:
                if child not in reachable:
                    reachable.add(child)
                    queue.append(child)
        
        original_count = len(self.categories)
        lost_cats = self.categories[~self.categories["category_id"].isin(reachable)]
        
        if len(lost_cats) > 0:
            print(f"  Warning: {len(lost_cats)} categories unreachable. Examples:")
            for title in lost_cats["title"].head(20).tolist():
                print(f"    - {title}")
        
        # Filter to reachable
        self.categories = self.categories[self.categories["category_id"].isin(reachable)]
        self.edges = self.edges[
            self.edges["parent_id"].isin(reachable) &
            self.edges["child_id"].isin(reachable)
        ]
        self.page_cats = self.page_cats[self.page_cats["category_id"].isin(reachable)]
        
        # Update lookups
        self.id_to_title = {k: v for k, v in self.id_to_title.items() if k in reachable}
        self.title_to_id = {v: k for k, v in self.id_to_title.items()}
        
        # Rebuild adjacency
        self.children_of = defaultdict(list)
        self.parents_of = defaultdict(list)
        for p, c in self.edges[["parent_id", "child_id"]].itertuples(index=False):
            self.children_of[p].append(c)
            self.parents_of[c].append(p)
        
        # Filter best_paths
        self.best_paths = {k: v for k, v in self.best_paths.items() if k in reachable}
        
        print(f"  Reachable: {len(reachable)}")
        print(f"  Removed: {original_count - len(self.categories)} categories")
        
        return self
    
    def compute_final_depths(self) -> 'SemanticTreeBuilder':
        """Compute final depths after pruning."""
        print("\nComputing final depths...")
        
        self.depths = {self.root_id: 0}
        queue = deque([self.root_id])
        
        while queue:
            node = queue.popleft()
            for child in self.children_of[node]:
                if child not in self.depths:
                    self.depths[child] = self.depths[node] + 1
                    queue.append(child)
        
        # Add depth to categories
        self.categories = self.categories.copy()
        self.categories["depth"] = self.categories["category_id"].map(self.depths)
        
        print(f"  Max depth: {self.categories['depth'].max()}")
        
        return self
    
    def print_summary(self) -> 'SemanticTreeBuilder':
        """Print summary statistics."""
        print("\n" + "=" * 60)
        print("FINAL SUMMARY")
        print("=" * 60)
        print(f"  Categories: {len(self.categories)}")
        print(f"  Edges: {len(self.edges)}")
        print(f"  Page-category links: {len(self.page_cats)}")
        print(f"  Max depth: {self.categories['depth'].max()}")
        
        print(f"  Depth distribution:")
        depth_dist = self.categories["depth"].value_counts().sort_index()
        for d, count in depth_dist.head(15).items():
            print(f"    Depth {d}: {count} categories")
        if len(depth_dist) > 15:
            print(f"    ... ({len(depth_dist)} total depth levels)")
        
        # Parent statistics
        parents_per_node = self.edges.groupby("child_id").size()
        if len(parents_per_node) > 0:
            print(f"  Parents per category:")
            print(f"    Max: {parents_per_node.max()}")
            print(f"    Mean: {parents_per_node.mean():.2f}")
            parent_dist = parents_per_node.value_counts().sort_index()
            for n_parents, count in parent_dist.items():
                print(f"    {n_parents} parent(s): {count} categories")
        
        return self
    
    def print_sample_paths(self, titles: List[str] = None) -> 'SemanticTreeBuilder':
        """Print sample paths for debugging."""
        if titles is None:
            titles = ["Dogs", "Cats", "Domesticated_animals", "Mammals", "Animals", 
                     "Roses", "Physics", "France", "Jazz", "Python_(programming_language)"]
        
        print("\n  Sample paths:")
        for title in titles:
            if title in self.title_to_id:
                node_id = self.title_to_id[title]
                if node_id in self.best_paths:
                    domain = self._infer_domain(title)
                    paths = self.best_paths[node_id]
                    print(f"\n    {title} [domain={domain}]:")
                    for i, path in enumerate(paths[:self.config.k_parents]):
                        path_titles = [self.id_to_title.get(n, f"[{n}]") for n in path[:10]]
                        suffix = "..." if len(path) > 10 else ""
                        print(f"      Path {i+1}: {' → '.join(path_titles)}{suffix}")
        
        return self
    
    def save(self, output_dir: str) -> 'SemanticTreeBuilder':
        """Save results to parquet files."""
        print(f"\nSaving to {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"  Writing semantic_categories.parquet...")
        pq.write_table(
            pa.Table.from_pandas(self.categories, preserve_index=False),
            f"{output_dir}/semantic_categories.parquet"
        )
        
        print(f"  Writing semantic_category_edges.parquet...")
        pq.write_table(
            pa.Table.from_pandas(self.edges, preserve_index=False),
            f"{output_dir}/semantic_category_edges.parquet"
        )
        
        print(f"  Writing semantic_page_categories.parquet...")
        pq.write_table(
            pa.Table.from_pandas(self.page_cats, preserve_index=False),
            f"{output_dir}/semantic_page_categories.parquet"
        )
        
        print("Done!")
        return self
    
    def build(self, input_dir: str, output_dir: str) -> 'SemanticTreeBuilder':
        """Full pipeline: load, prune, verify, compute, save."""
        print("=" * 60)
        print(f"Building Semantic Tree (k={self.config.k_parents} parents)")
        print("=" * 60)
        
        return (
            self
            .load(input_dir)
            .prune()
            .verify_connectivity()
            .compute_final_depths()
            .print_summary()
            .print_sample_paths()
            .save(output_dir)
        )

class KnowledgeGraphBuilder:
    """
    Builds a unified knowledge graph where categories and pages are both "concept nodes".
    
    - Categories and pages are stored in a single node table with `node_type` field
    - All edges are stored in a single edge table
    - Pages attach only to categories (remain as leaves)
    """
    
    def __init__(self, config: SemanticTreeConfig = None):
        self.config = config or SemanticTreeConfig()
        
        # Category tree (loaded from SemanticTreeBuilder output)
        self.categories: Optional[pd.DataFrame] = None
        self.category_edges: Optional[pd.DataFrame] = None
        
        # Category lookups (using string node_ids for unification)
        self.cat_id_to_title: Dict[str, str] = {}
        self.cat_title_to_id: Dict[str, str] = {}
        self.cat_children: Dict[str, List[str]] = defaultdict(list)
        self.cat_parent: Dict[str, Optional[str]] = {}
        self.cat_depths: Dict[str, int] = {}
        self.cat_paths_to_root: Dict[str, List[str]] = {}
        self.root_id: Optional[str] = None
        
        # Anchor lookups (using string node_ids)
        self.anchor_ids: Set[str] = set()
        self.domain_anchor_ids: Dict[str, Set[str]] = {}
        
        # Visual pages
        self.visual_nodes: Optional[pd.DataFrame] = None
        self.page_categories_raw: Optional[pd.DataFrame] = None
        
        # Page data
        self.page_id_to_title: Dict[str, str] = {}
        self.page_to_categories: Dict[str, List[str]] = defaultdict(list)
        
        # Final unified output
        self.final_nodes: Optional[pd.DataFrame] = None
        self.final_edges: Optional[pd.DataFrame] = None
    @staticmethod
    def _make_category_node_id(cat_id: int) -> str:
        """Create unified node_id for a category."""
        return f"cat_{cat_id}"
    @staticmethod
    def _make_page_node_id(page_id: int) -> str:
        """Create unified node_id for a page."""
        return f"page_{page_id}"
    
    # -------------------------------------------------------------------------
    # Loading
    # -------------------------------------------------------------------------
    
    def load_semantic_tree(self, semantic_tree_dir: str) -> 'KnowledgeGraphBuilder':
        """Load the pre-built semantic category tree from SemanticTreeBuilder output."""
        print("=" * 60)
        print("LOADING SEMANTIC TREE")
        print("=" * 60)
        
        # Load categories
        cat_path = f"{semantic_tree_dir}/semantic_categories.parquet"
        print(f"Loading categories from {cat_path}...")
        self.categories = pd.read_parquet(cat_path)
        print(f"  {len(self.categories)} categories")
        
        # Load edges
        edges_path = f"{semantic_tree_dir}/semantic_category_edges.parquet"
        print(f"Loading edges from {edges_path}...")
        self.category_edges = pd.read_parquet(edges_path)
        print(f"  {len(self.category_edges)} edges")
        
        # Build lookups with unified node_ids
        print("Building lookups...")
        for _, row in self.categories.iterrows():
            node_id = self._make_category_node_id(row["category_id"])
            title = row["title"]
            self.cat_id_to_title[node_id] = title
            self.cat_title_to_id[title.lower()] = node_id
        
        # Build adjacency with unified node_ids
        self.cat_children = defaultdict(list)
        self.cat_parent = {}
        
        for _, row in self.category_edges.iterrows():
            parent_node_id = self._make_category_node_id(row["parent_id"])
            child_node_id = self._make_category_node_id(row["child_id"])
            self.cat_children[parent_node_id].append(child_node_id)
            self.cat_parent[child_node_id] = parent_node_id
        
        # Find root (node with no parent)
        all_children = set(self._make_category_node_id(c) for c in self.category_edges["child_id"])
        all_parents = set(self._make_category_node_id(p) for p in self.category_edges["parent_id"])
        roots = all_parents - all_children
        
        if len(roots) == 1:
            self.root_id = roots.pop()
            print(f"  Root: '{self.cat_id_to_title.get(self.root_id)}' (id: {self.root_id})")
        elif len(roots) == 0:
            if "main_topic_classifications" in self.cat_title_to_id:
                self.root_id = self.cat_title_to_id["main_topic_classifications"]
                print(f"  Root (by name): 'Main_topic_classifications' (id: {self.root_id})")
            else:
                raise ValueError("Could not identify root category")
        else:
            print(f"  Warning: Multiple roots found: {roots}")
            if "main_topic_classifications" in self.cat_title_to_id:
                self.root_id = self.cat_title_to_id["main_topic_classifications"]
            else:
                self.root_id = min(roots)
            print(f"  Using root: '{self.cat_id_to_title.get(self.root_id)}' (id: {self.root_id})")
        
        # Compute depths via BFS from root
        print("Computing depths...")
        self.cat_depths = {self.root_id: 0}
        queue = deque([self.root_id])
        while queue:
            node = queue.popleft()
            for child in self.cat_children[node]:
                if child not in self.cat_depths:
                    self.cat_depths[child] = self.cat_depths[node] + 1
                    queue.append(child)
        
        print(f"  Max depth: {max(self.cat_depths.values()) if self.cat_depths else 0}")
        
        # Compute paths to root
        print("Computing paths to root...")
        self._compute_paths_to_root()
        
        # Build anchor lookups with unified node_ids
        self.anchor_ids = set()
        for anchor_title in self.config.preferred_anchors:
            if anchor_title.lower() in self.cat_title_to_id:
                self.anchor_ids.add(self.cat_title_to_id[anchor_title.lower()])
        print(f"  Found {len(self.anchor_ids)} anchor categories")
        
        self.domain_anchor_ids = {}
        for domain, anchors in self.config.domain_anchors.items():
            self.domain_anchor_ids[domain] = set()
            for anchor_title in anchors:
                if anchor_title.lower() in self.cat_title_to_id:
                    self.domain_anchor_ids[domain].add(self.cat_title_to_id[anchor_title.lower()])
        
        return self
    
    def _compute_paths_to_root(self):
        """Compute path to root for each category."""
        self.cat_paths_to_root = {}
        
        for node_id in self.cat_depths.keys():
            path = [node_id]
            current = node_id
            
            while current != self.root_id and current in self.cat_parent:
                current = self.cat_parent[current]
                path.append(current)
            
            self.cat_paths_to_root[node_id] = path
    
    def load_visual_nodes(self, visual_nodes_path: str) -> 'KnowledgeGraphBuilder':
        """Load pre-filtered visual nodes."""
        print(f"\nLoading visual nodes from {visual_nodes_path}...")
        self.visual_nodes = pd.read_parquet(visual_nodes_path)
        print(f"  {len(self.visual_nodes)} visual pages")
        print(f"  Columns: {self.visual_nodes.columns.tolist()}")
        return self
    
    def load_page_categories(self, page_categories_path: str) -> 'KnowledgeGraphBuilder':
        """Load page-category mappings."""
        print(f"\nLoading page categories from {page_categories_path}...")
        self.page_categories_raw = pd.read_parquet(page_categories_path)
        print(f"  {len(self.page_categories_raw)} page-category links")
        return self
    
    # -------------------------------------------------------------------------
    # Path Scoring
    # -------------------------------------------------------------------------
    
    def _has_penalized_pattern(self, title: str) -> bool:
        """Check if a title contains a penalized pattern."""
        for pattern in self.config.penalized_patterns:
            if pattern in title:
                return True
        return False
    
    def _infer_domain(self, title: str) -> Optional[str]:
        """Infer which domain a title likely belongs to."""
        title_lower = title.lower()
        for keyword, domain in self.config.domain_keywords.items():
            if keyword in title_lower:
                return domain
        return None
    
    def _compute_path_penalty(self, path: List[str]) -> int:
        """Compute penalty for a path based on penalized patterns."""
        penalty = 0
        for node_id in path:
            title = self.cat_id_to_title.get(node_id, "")
            if self._has_penalized_pattern(title):
                penalty += self.config.pattern_penalty_weight
        return penalty
    
    def _path_has_domain_anchor(self, path: List[str], domain: Optional[str]) -> bool:
        """Check if path contains an anchor for the given domain."""
        if domain is None or domain not in self.domain_anchor_ids:
            return False
        return len(set(path).intersection(self.domain_anchor_ids[domain])) > 0
    
    def _count_anchors_in_path(self, path: List[str]) -> int:
        """Count how many anchor categories are in the path."""
        return len(set(path).intersection(self.anchor_ids))
    
    def _score_path(self, path: List[str], node_domain: Optional[str]) -> Tuple:
        """
        Score a path (lower is better).
        
        Returns tuple for lexicographic comparison:
        (penalty, no_domain_anchor, neg_anchor_count, first_not_anchor, length)
        """
        penalty = self._compute_path_penalty(path)
        has_domain = 0 if self._path_has_domain_anchor(path, node_domain) else 1
        anchor_count = -self._count_anchors_in_path(path)
        first_is_anchor = 0 if (path and path[0] in self.anchor_ids) else 1
        length = len(path)
        
        return (penalty, has_domain, anchor_count, first_is_anchor, length)
    def _prune_empty_categories(self):
        """
        Recursively removes categories that have neither child categories 
        nor attached pages.
        """
        print("  Pruning empty categories...")
        
        while True:
            # 1. Identify categories that have pages attached
            cats_with_pages = set(self.final_nodes[self.final_nodes['node_type'] == 'page']['parent_id'])
            
            # 2. Identify categories that are parents to other categories
            cats_with_subcats = set(self.final_edges[
                self.final_edges['child_id'].str.startswith('cat_')
            ]['parent_id'])
            
            # 3. Combine to find "useful" categories
            useful_cats = cats_with_pages.union(cats_with_subcats)
            
            # 4. Identify categories to drop (must be a category and NOT in the useful list)
            # Note: We never prune the root_id
            all_cats = set(self.final_nodes[self.final_nodes['node_type'] == 'category']['node_id'])
            to_drop = (all_cats - useful_cats) - {self.root_id}
            
            if not to_drop:
                break
                
            print(f"    Removing {len(to_drop)} empty leaf categories...")
            
            # 5. Filter the nodes and edges
            self.final_nodes = self.final_nodes[~self.final_nodes['node_id'].isin(to_drop)]
            self.final_edges = self.final_edges[
                ~self.final_edges['child_id'].isin(to_drop) & 
                ~self.final_edges['parent_id'].isin(to_drop)
            ]
    
    # -------------------------------------------------------------------------
    # Build Unified Graph
    # -------------------------------------------------------------------------
    
    def build_graph(self) -> 'KnowledgeGraphBuilder':
        """Build the unified knowledge graph with categories and pages as concept nodes."""
        print("\n" + "=" * 60)
        print("BUILDING UNIFIED KNOWLEDGE GRAPH")
        print("=" * 60)
        
        # Determine page ID and title columns
        if 'page_id' in self.visual_nodes.columns:
            page_id_col = 'page_id'
        elif 'id' in self.visual_nodes.columns:
            page_id_col = 'id'
        else:
            raise ValueError(f"Cannot find page ID column. Columns: {self.visual_nodes.columns.tolist()}")
        
        if 'title' in self.visual_nodes.columns:
            title_col = 'title'
        elif 'name' in self.visual_nodes.columns:
            title_col = 'name'
        else:
            title_col = None
            print("  Warning: No title column found, using IDs as titles")
        
        # Build page lookups
        visual_page_ids = set(self.visual_nodes[page_id_col])
        print(f"  Visual pages to attach: {len(visual_page_ids)}")
        
        if title_col:
            for _, row in self.visual_nodes.iterrows():
                node_id = self._make_page_node_id(row[page_id_col])
                self.page_id_to_title[node_id] = row[title_col]
        else:
            for pid in visual_page_ids:
                node_id = self._make_page_node_id(pid)
                self.page_id_to_title[node_id] = str(pid)
        
        # Build set of category titles (normalized) for duplicate detection
        cat_titles_normalized = set()
        for title in self.cat_id_to_title.values():
            normalized = title.lower().replace(' ', '_').replace('-', '_')
            cat_titles_normalized.add(normalized)
        print(f"  Category titles for dedup: {len(cat_titles_normalized)}")
        
        # Build page -> categories mapping
        valid_cat_node_ids = set(self.cat_id_to_title.keys())
        
        print("  Building page-category mappings...")
        self.page_to_categories = defaultdict(list)
        
        for page_id, cat_id in tqdm(
            self.page_categories_raw[["page_id", "category_id"]].itertuples(index=False),
            total=len(self.page_categories_raw),
            desc="  Filtering"
        ):
            if page_id in visual_page_ids:
                cat_node_id = self._make_category_node_id(cat_id)
                if cat_node_id in valid_cat_node_ids:
                    page_node_id = self._make_page_node_id(page_id)
                    self.page_to_categories[page_node_id].append(cat_node_id)
        
        pages_with_categories = len(self.page_to_categories)
        pages_without = len(visual_page_ids) - pages_with_categories
        print(f"  Pages with valid category mappings: {pages_with_categories}")
        print(f"  Pages without valid categories: {pages_without}")
        
        # Select best category for each page
        print("\n  Selecting best category for each page...")
        
        page_assignments = {}  # page_node_id -> parent_cat_node_id
        stats = {
            "single_cat": 0,
            "penalized_avoided": 0,
            "domain_preferred": 0,
            "anchor_preferred": 0,
            "depth_preferred": 0,
            "length_tiebreak": 0,
            "skipped_same_name": 0,
            "skipped_no_valid_cat": 0,
        }
        
        for page_node_id, cat_node_ids in tqdm(self.page_to_categories.items(), desc="  Scoring"):
            if not cat_node_ids:
                stats["skipped_no_valid_cat"] += 1
                continue
            
            page_title = self.page_id_to_title.get(page_node_id, "")
            page_title_normalized = page_title.lower().replace(' ', '_').replace('-', '_')

            # Check if a category with the same name exists
            if page_title_normalized in cat_titles_normalized:
                # Find the matching category ID
                matching_cat_id = self.cat_title_to_id.get(page_title_normalized)
                
                # If this page can attach to the matching category, prioritize it
                if matching_cat_id and matching_cat_id in cat_node_ids:
                    page_assignments[page_node_id] = matching_cat_id
                    stats["same_name_attached"] = stats.get("same_name_attached", 0) + 1
                    continue
                else:
                    # Category exists but page can't attach to it - skip the page
                    stats["skipped_same_name"] += 1
                    continue
            
            page_domain = self._infer_domain(page_title)
            
            # Score each candidate category
            candidates = []
            for cat_node_id in cat_node_ids:
                if cat_node_id not in self.cat_paths_to_root:
                    continue
                
                cat_path = self.cat_paths_to_root[cat_node_id]
                base_score = self._score_path(cat_path, page_domain)
                
                # Prefer deeper (more specific) categories
                depth = self.cat_depths.get(cat_node_id, 0)
                score = base_score + (-depth,)
                
                candidates.append((cat_node_id, score))
            
            if not candidates:
                stats["skipped_no_valid_cat"] += 1
                continue
            
            # Sort and take best
            candidates.sort(key=lambda x: x[1])
            best_cat, best_score = candidates[0]
            
            page_assignments[page_node_id] = best_cat
            
            # Track stats
            if len(cat_node_ids) == 1:
                stats["single_cat"] += 1
            elif len(candidates) > 1:
                second_score = candidates[1][1]
                if best_score[0] < second_score[0]:
                    stats["penalized_avoided"] += 1
                elif best_score[1] < second_score[1]:
                    stats["domain_preferred"] += 1
                elif best_score[2] < second_score[2]:
                    stats["anchor_preferred"] += 1
                elif best_score[5] < second_score[5]:
                    stats["depth_preferred"] += 1
                else:
                    stats["length_tiebreak"] += 1
        
        print(f"\n  Page attachment statistics:")
        print(f"    Single category (no choice): {stats['single_cat']}")
        print(f"    Avoided penalized paths: {stats['penalized_avoided']}")
        print(f"    Preferred domain paths: {stats['domain_preferred']}")
        print(f"    Preferred anchor paths: {stats['anchor_preferred']}")
        print(f"    Preferred deeper category: {stats['depth_preferred']}")
        print(f"    Length tiebreaker: {stats['length_tiebreak']}")
        print(f"  Skipped:")
        print(f"    Same name as category (attached to it): {stats.get('same_name_attached', 0)}")
        print(f"    Same name as category (couldn't attach): {stats['skipped_same_name']}")
        print(f"    No valid category: {stats['skipped_no_valid_cat']}")

        
        # Build unified node table
        print("\n  Building unified node table...")
        nodes = []
        
        # Add category nodes
        for cat_node_id, title in self.cat_id_to_title.items():
            nodes.append({
                'node_id': cat_node_id,
                'title': title,
                'node_type': 'category',
                'depth': self.cat_depths.get(cat_node_id, 0),
                'parent_id': self.cat_parent.get(cat_node_id),
            })
        
        # Add page nodes
        for page_node_id, parent_cat_id in page_assignments.items():
            nodes.append({
                'node_id': page_node_id,
                'title': self.page_id_to_title.get(page_node_id, ""),
                'node_type': 'page',
                'depth': self.cat_depths.get(parent_cat_id, 0) + 1,
                'parent_id': parent_cat_id,
            })
        
        self.final_nodes = pd.DataFrame(nodes)
        
        # Build unified edge table
        print("  Building unified edge table...")
        edges = []
        
        # Category -> category edges
        for child_id, parent_id in self.cat_parent.items():
            edges.append({
                'parent_id': parent_id,
                'child_id': child_id,
            })
        
        # Category -> page edges
        for page_node_id, parent_cat_id in page_assignments.items():
            edges.append({
                'parent_id': parent_cat_id,
                'child_id': page_node_id,
            })
        
        self.final_edges = pd.DataFrame(edges)
        self._prune_empty_categories()
        
        return self
    
    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------
    
    def print_summary(self) -> 'KnowledgeGraphBuilder':
        """Print summary statistics."""
        print("\n" + "=" * 60)
        print("FINAL SUMMARY")
        print("=" * 60)
        
        n_categories = len(self.final_nodes[self.final_nodes['node_type'] == 'category'])
        n_pages = len(self.final_nodes[self.final_nodes['node_type'] == 'page'])
        
        print(f"\nUnified Knowledge Graph:")
        print(f"  Total nodes: {len(self.final_nodes)}")
        print(f"    - Categories: {n_categories}")
        print(f"    - Pages: {n_pages}")
        print(f"  Total edges: {len(self.final_edges)}")
        
        print(f"\n  Max depth: {self.final_nodes['depth'].max()}")
        
        print(f"\n  Depth distribution:")
        depth_dist = self.final_nodes['depth'].value_counts().sort_index()
        for d, count in depth_dist.head(15).items():
            cat_count = len(self.final_nodes[(self.final_nodes['depth'] == d) & 
                                              (self.final_nodes['node_type'] == 'category')])
            page_count = len(self.final_nodes[(self.final_nodes['depth'] == d) & 
                                               (self.final_nodes['node_type'] == 'page')])
            print(f"    Depth {d}: {count} nodes ({cat_count} cats, {page_count} pages)")
        if len(depth_dist) > 15:
            print(f"    ... ({len(depth_dist)} total levels)")
        
        # Pages per category stats
        page_nodes = self.final_nodes[self.final_nodes['node_type'] == 'page']
        if len(page_nodes) > 0:
            pages_per_cat = page_nodes.groupby('parent_id').size()
            print(f"\n  Pages per category:")
            print(f"    Min: {pages_per_cat.min()}")
            print(f"    Max: {pages_per_cat.max()}")
            print(f"    Mean: {pages_per_cat.mean():.2f}")
            print(f"    Median: {pages_per_cat.median():.0f}")
            print(f"    Categories with pages: {len(pages_per_cat)}")
        
        
        return self
    
    def print_sample_paths(self, n_samples: int = 10) -> 'KnowledgeGraphBuilder':
        """Print sample page paths for verification."""
        print("\n" + "=" * 60)
        print("SAMPLE PAGE PATHS")
        print("=" * 60)
        
        page_nodes = self.final_nodes[self.final_nodes['node_type'] == 'page']
        
        if len(page_nodes) == 0:
            print("  No pages attached")
            return self
        
        sample_pages = page_nodes.sample(min(n_samples, len(page_nodes)))
        
        for _, row in sample_pages.iterrows():
            page_node_id = row['node_id']
            page_title = row['title']
            parent_cat_id = row['parent_id']
            
            if parent_cat_id in self.cat_paths_to_root:
                path = self.cat_paths_to_root[parent_cat_id]
                path_titles = [self.cat_id_to_title.get(n, f"[{n}]") for n in path]
                
                domain = self._infer_domain(page_title)
                domain_str = f" [domain={domain}]" if domain else ""
                
                print(f"\n  {page_title.replace('_', ' ')}{domain_str}:")
                print(f"    → {' → '.join(t.replace('_', ' ') for t in path_titles[:8])}", end="")
                if len(path_titles) > 8:
                    print(" → ...")
                else:
                    print()
        
        return self
    
    def save(self, output_dir: str) -> 'KnowledgeGraphBuilder':
        """Save the unified knowledge graph."""
        print("\n" + "=" * 60)
        print(f"SAVING TO {output_dir}")
        print("=" * 60)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save unified nodes
        nodes_path = f"{output_dir}/nodes.parquet"
        print(f"  Writing {nodes_path}...")
        pq.write_table(
            pa.Table.from_pandas(self.final_nodes, preserve_index=False),
            nodes_path
        )
        
        # Save unified edges
        edges_path = f"{output_dir}/edges.parquet"
        print(f"  Writing {edges_path}...")
        pq.write_table(
            pa.Table.from_pandas(self.final_edges, preserve_index=False),
            edges_path
        )
        
        print("\nDone!")
        return self
    
    # -------------------------------------------------------------------------
    # Full Pipeline
    # -------------------------------------------------------------------------
    
    def build(
        self,
        semantic_tree_dir: str,
        visual_nodes_path: str,
        page_categories_path: str,
        output_dir: str,
    ) -> 'KnowledgeGraphBuilder':
        """Run the full pipeline."""
        return (
            self
            .load_semantic_tree(semantic_tree_dir)
            .load_visual_nodes(visual_nodes_path)
            .load_page_categories(page_categories_path)
            .build_graph()
            .print_summary()
            .print_sample_paths()
            .save(output_dir)
        )

class KnowledgeGraphIndex:
    """
    Index for fast queries on the knowledge graph.
    
    Supports:
    - Looking up pages and categories by name
    - Displaying hierarchy paths to root
    - Showing siblings and children
    """
    
    def __init__(self, knowledge_graph_dir: str):
        print(f"Loading unified knowledge graph from {knowledge_graph_dir}...")
        
        # 1. 加载统一的节点和边表
        self.nodes = pd.read_parquet(f"{knowledge_graph_dir}/nodes.parquet")
        self.edges = pd.read_parquet(f"{knowledge_graph_dir}/edges.parquet")
        
        # 2. 构建节点属性查找表 (node_id -> {title, node_type, depth, parent_id})
        # 将 nodes 转换为 dict 极大提高查询速度
        self.node_data = self.nodes.set_index('node_id').to_dict('index')
        
        # 3. 构建名称到 ID 的反向索引 (用于搜索)
        self.title_to_id = defaultdict(list)
        for node_id, data in self.node_data.items():
            title_norm = str(data['title']).lower().replace(" ", "_")
            self.title_to_id[title_norm].append(node_id)
            
        # 4. 构建邻接表 (父 -> 子)
        self.children_map = defaultdict(list)
        for _, row in self.edges.iterrows():
            self.children_map[row['parent_id']].append(row['child_id'])
            
        # 5. 识别根节点
        root_nodes = self.nodes[self.nodes['parent_id'].isna()]
        self.root_id = root_nodes.iloc[0]['node_id'] if not root_nodes.empty else None
        
        print(f"  Total Nodes: {len(self.nodes)}")
        print(f"  - Categories: {len(self.nodes[self.nodes['node_type'] == 'category'])}")
        print(f"  - Pages: {len(self.nodes[self.nodes['node_type'] == 'page'])}")
        print(f"  Ready.\n")

    def _get_path_to_root(self, node_id: str) -> List[str]:
        """追溯到根节点的路径"""
        path = []
        current = node_id
        while current and current in self.node_data:
            path.append(current)
            current = self.node_data[current].get('parent_id')
        return path

    def search(self, query: str, node_type: Optional[str] = None, limit: int = 5) -> List[str]:
        """通用搜索：支持按类型过滤"""
        query_norm = query.lower().replace(" ", "_")
        results = []
        
        # 优先匹配完全一致的名字
        if query_norm in self.title_to_id:
            results.extend(self.title_to_id[query_norm])
            
        # 模糊匹配
        if len(results) < limit:
            for title, ids in self.title_to_id.items():
                if query_norm in title and ids[0] not in results:
                    results.extend(ids)
                if len(results) >= limit: break
        
        # 按类型过滤
        if node_type:
            results = [rid for rid in results if self.node_data[rid]['node_type'] == node_type]
            
        return results[:limit]

    def query(self, concept: str):
        """主查询接口：展示概念的层级结构"""
        # 搜索匹配的节点
        match_ids = self.search(concept)
        
        if not match_ids:
            print(f"No results found for: '{concept}'")
            return

        for node_id in match_ids:
            data = self.node_data[node_id]
            node_type = data['node_type'].upper()
            title = data['title'].replace('_', ' ')
            
            print("=" * 70)
            print(f"{node_type}: {title}")
            print("=" * 70)
            
            # 展示路径
            path = self._get_path_to_root(node_id)
            print("\nHIERARCHY PATH:")
            print("-" * 50)
            for i, step_id in enumerate(reversed(path)):
                step_data = self.node_data[step_id]
                indent = "  " * i
                prefix = "└── " if i > 0 else ""
                icon = "📁" if step_data['node_type'] == 'category' else "📄"
                print(f"{indent}{prefix}{icon} {step_data['title']} [L{step_data['depth']}]")
            
            # 展示子节点 (如果是分类)
            if data['node_type'] == 'category':
                children = self.children_map.get(node_id, [])
                if children:
                    print("\nDIRECT CHILDREN:")
                    print("-" * 50)
                    cats = [c for c in children if self.node_data[c]['node_type'] == 'category']
                    pages = [c for c in children if self.node_data[c]['node_type'] == 'page']
                    
                    for c in cats[:10]:
                        print(f"  📁 {self.node_data[c]['title']}")
                    if len(pages) > 0:
                        print(f"  ... and {len(pages)} pages")
            
            # 展示兄弟节点 (可选)
            parent_id = data.get('parent_id')
            if parent_id:
                siblings = [s for s in self.children_map.get(parent_id, []) if s != node_id]
                if siblings:
                    print(f"\nSIBLINGS (under {self.node_data[parent_id]['title']}):")
                    print("-" * 50)
                    for s in siblings[:5]:
                        s_data = self.node_data[s]
                        icon = "📁" if s_data['node_type'] == 'category' else "📄"
                        print(f"  {icon} {s_data['title']}")
            print("\n")