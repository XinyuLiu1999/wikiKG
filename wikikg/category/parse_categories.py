"""Parse Wikipedia category hierarchy from SQL dumps."""

import argparse
import os
import sys

import pyarrow as pa

from wikikg.common import ParquetBatchWriter, iter_insert_tuples, to_int, to_str


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def load_page_maps(page_sql, max_pages=None):
    """Load page ID mappings for main and category namespaces.

    Args:
        page_sql: Path to page.sql.gz
        max_pages: Limit number of main namespace pages

    Returns:
        tuple: (main_page_ids set, cat_page_id_to_title dict)
    """
    main_page_ids = set()
    cat_page_id_to_title = {}
    count = 0

    for fields in iter_insert_tuples(page_sql, "page"):
        page_id = to_int(fields[0])
        ns = to_int(fields[1])
        title = to_str(fields[2])
        if page_id is None or ns is None or title is None:
            continue
        if ns == 0:
            if max_pages is None or count < max_pages:
                main_page_ids.add(page_id)
                count += 1
        elif ns == 14:
            cat_page_id_to_title[page_id] = title

        if count % 1_000_000 == 0 and count > 0:
            log(f"page: loaded {count} main-namespace pages")

    return main_page_ids, cat_page_id_to_title


def load_categories(category_sql, max_categories=None):
    """Load category metadata from category.sql.gz.

    Returns:
        tuple: (categories list, cat_title_to_id dict)
    """
    cat_title_to_id = {}
    categories = []
    count = 0

    for fields in iter_insert_tuples(category_sql, "category"):
        cat_id = to_int(fields[0])
        cat_title = to_str(fields[1])
        cat_pages = to_int(fields[2])
        if cat_id is None or cat_title is None:
            continue
        cat_title_to_id[cat_title] = cat_id
        categories.append((cat_id, cat_title, cat_pages))
        count += 1
        if max_categories is not None and count >= max_categories:
            break
        if count % 1_000_000 == 0:
            log(f"category: loaded {count}")

    return categories, cat_title_to_id


def write_categories(categories, out_path, batch_size=1_000_000):
    """Write category nodes to Parquet file."""
    schema = pa.schema([
        ("category_id", pa.int64()),
        ("title", pa.string()),
        ("page_count", pa.int64()),
    ])

    with ParquetBatchWriter(out_path, schema) as writer:
        batch_id = []
        batch_title = []
        batch_count = []

        for cat_id, title, page_count in categories:
            batch_id.append(cat_id)
            batch_title.append(title)
            batch_count.append(page_count if page_count is not None else 0)
            if len(batch_id) >= batch_size:
                writer.write({
                    "category_id": batch_id,
                    "title": batch_title,
                    "page_count": batch_count,
                })
                batch_id = []
                batch_title = []
                batch_count = []

        if batch_id:
            writer.write({
                "category_id": batch_id,
                "title": batch_title,
                "page_count": batch_count,
            })


def process_categorylinks(
    categorylinks_sql,
    cat_title_to_id,
    cat_page_id_to_title,
    main_page_ids,
    category_edges_path,
    page_categories_path,
    max_category_edges=None,
    max_page_categories=None,
    batch_size=1_000_000,
):
    """Process categorylinks in a single pass to extract both edges and mappings.

    This is an optimized version that reads the file only once instead of twice.
    """
    # Schemas
    edges_schema = pa.schema([
        ("parent_id", pa.int64()),
        ("child_id", pa.int64()),
    ])
    page_cat_schema = pa.schema([
        ("page_id", pa.int64()),
        ("category_id", pa.int64()),
    ])

    # Writers
    edges_writer = ParquetBatchWriter(category_edges_path, edges_schema)
    page_cat_writer = ParquetBatchWriter(page_categories_path, page_cat_schema)

    # Batches for category edges
    batch_parent = []
    batch_child = []
    edge_count = 0
    edge_limit_reached = False

    # Batches for page-category mappings
    batch_page = []
    batch_cat = []
    page_cat_count = 0
    page_cat_limit_reached = False

    try:
        for fields in iter_insert_tuples(categorylinks_sql, "categorylinks"):
            cl_from = to_int(fields[0])
            cl_to = to_str(fields[1])
            cl_type = to_str(fields[6])
            if cl_from is None or cl_to is None or cl_type is None:
                continue

            # Process subcategory edges
            if cl_type == "subcat" and not edge_limit_reached:
                child_title = cat_page_id_to_title.get(cl_from)
                if child_title is not None:
                    child_id = cat_title_to_id.get(child_title)
                    parent_id = cat_title_to_id.get(cl_to)
                    if child_id is not None and parent_id is not None:
                        batch_parent.append(parent_id)
                        batch_child.append(child_id)
                        edge_count += 1

                        if max_category_edges is not None and edge_count >= max_category_edges:
                            edge_limit_reached = True

                        if len(batch_parent) >= batch_size:
                            edges_writer.write({"parent_id": batch_parent, "child_id": batch_child})
                            batch_parent = []
                            batch_child = []
                            log(f"categorylinks: wrote {edge_count} category edges")

            # Process page-category mappings
            elif cl_type == "page" and not page_cat_limit_reached:
                if cl_from in main_page_ids:
                    cat_id = cat_title_to_id.get(cl_to)
                    if cat_id is not None:
                        batch_page.append(cl_from)
                        batch_cat.append(cat_id)
                        page_cat_count += 1

                        if max_page_categories is not None and page_cat_count >= max_page_categories:
                            page_cat_limit_reached = True

                        if len(batch_page) >= batch_size:
                            page_cat_writer.write({"page_id": batch_page, "category_id": batch_cat})
                            batch_page = []
                            batch_cat = []
                            log(f"categorylinks: wrote {page_cat_count} page-category pairs")

            # Early exit if both limits reached
            if edge_limit_reached and page_cat_limit_reached:
                break

        # Write remaining batches
        if batch_parent:
            edges_writer.write({"parent_id": batch_parent, "child_id": batch_child})
        if batch_page:
            page_cat_writer.write({"page_id": batch_page, "category_id": batch_cat})

    finally:
        edges_writer.close()
        page_cat_writer.close()

    log(f"categorylinks: total category edges {edge_count}")
    log(f"categorylinks: total page-category pairs {page_cat_count}")


# Keep old functions for backwards compatibility
def write_category_edges(categorylinks_sql, cat_title_to_id, cat_page_id_to_title, out_path, max_edges=None, batch_size=1_000_000):
    """Write category parent-child edges to Parquet file.

    DEPRECATED: Use process_categorylinks() for better performance.
    """
    schema = pa.schema([
        ("parent_id", pa.int64()),
        ("child_id", pa.int64()),
    ])

    count = 0
    with ParquetBatchWriter(out_path, schema) as writer:
        batch_parent = []
        batch_child = []

        for fields in iter_insert_tuples(categorylinks_sql, "categorylinks"):
            cl_from = to_int(fields[0])
            cl_to = to_str(fields[1])
            cl_type = to_str(fields[6])
            if cl_from is None or cl_to is None or cl_type is None:
                continue
            if cl_type != "subcat":
                continue

            child_title = cat_page_id_to_title.get(cl_from)
            if child_title is None:
                continue
            child_id = cat_title_to_id.get(child_title)
            parent_id = cat_title_to_id.get(cl_to)
            if child_id is None or parent_id is None:
                continue

            batch_parent.append(parent_id)
            batch_child.append(child_id)
            count += 1
            if max_edges is not None and count >= max_edges:
                break

            if len(batch_parent) >= batch_size:
                writer.write({"parent_id": batch_parent, "child_id": batch_child})
                batch_parent = []
                batch_child = []
                log(f"categorylinks: wrote {count} category edges")

        if batch_parent:
            writer.write({"parent_id": batch_parent, "child_id": batch_child})

    log(f"categorylinks: total category edges {count}")


def write_page_categories(categorylinks_sql, cat_title_to_id, main_page_ids, out_path, max_pairs=None, batch_size=1_000_000):
    """Write page-to-category mappings to Parquet file.

    DEPRECATED: Use process_categorylinks() for better performance.
    """
    schema = pa.schema([
        ("page_id", pa.int64()),
        ("category_id", pa.int64()),
    ])

    count = 0
    with ParquetBatchWriter(out_path, schema) as writer:
        batch_page = []
        batch_cat = []

        for fields in iter_insert_tuples(categorylinks_sql, "categorylinks"):
            cl_from = to_int(fields[0])
            cl_to = to_str(fields[1])
            cl_type = to_str(fields[6])
            if cl_from is None or cl_to is None or cl_type is None:
                continue
            if cl_type != "page":
                continue
            if cl_from not in main_page_ids:
                continue
            cat_id = cat_title_to_id.get(cl_to)
            if cat_id is None:
                continue

            batch_page.append(cl_from)
            batch_cat.append(cat_id)
            count += 1
            if max_pairs is not None and count >= max_pairs:
                break

            if len(batch_page) >= batch_size:
                writer.write({"page_id": batch_page, "category_id": batch_cat})
                batch_page = []
                batch_cat = []
                log(f"categorylinks: wrote {count} page-category pairs")

        if batch_page:
            writer.write({"page_id": batch_page, "category_id": batch_cat})

    log(f"categorylinks: total page-category pairs {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Parse Wikipedia category hierarchy"
    )
    parser.add_argument("--page-sql", required=True, help="Path to page.sql.gz")
    parser.add_argument("--category-sql", required=True, help="Path to category.sql.gz")
    parser.add_argument("--categorylinks-sql", required=True, help="Path to categorylinks.sql.gz")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--max-pages", type=int, help="Limit main namespace pages")
    parser.add_argument("--max-categories", type=int, help="Limit categories")
    parser.add_argument("--max-category-edges", type=int, help="Limit category edges")
    parser.add_argument("--max-page-categories", type=int, help="Limit page-category pairs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    categories_path = os.path.join(args.out_dir, "wiki_categories.parquet")
    category_edges_path = os.path.join(args.out_dir, "wiki_category_edges.parquet")
    page_categories_path = os.path.join(args.out_dir, "wiki_page_categories.parquet")

    log("loading page maps...")
    main_page_ids, cat_page_id_to_title = load_page_maps(args.page_sql, args.max_pages)
    log(f"main pages: {len(main_page_ids)}")
    log(f"category pages: {len(cat_page_id_to_title)}")

    log("loading categories...")
    categories, cat_title_to_id = load_categories(args.category_sql, args.max_categories)
    log(f"categories: {len(categories)}")

    log("writing wiki_categories.parquet...")
    write_categories(categories, categories_path)

    log("processing categorylinks (single pass)...")
    process_categorylinks(
        args.categorylinks_sql,
        cat_title_to_id,
        cat_page_id_to_title,
        main_page_ids,
        category_edges_path,
        page_categories_path,
        max_category_edges=args.max_category_edges,
        max_page_categories=args.max_page_categories,
    )


if __name__ == "__main__":
    main()
