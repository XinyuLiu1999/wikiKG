"""Parse Wikipedia SQL dumps to extract hyperlink graph."""

import argparse
import os
import sys

import pyarrow as pa

from wikikg.common import ParquetBatchWriter, iter_insert_tuples, to_int, to_str


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def build_page_maps(page_sql, max_pages=None):
    """Build page ID <-> title mappings from page.sql.gz.

    Only keeps main namespace (ns=0) pages.

    Returns:
        tuple: (id_to_title dict, title_to_id dict)
    """
    id_to_title = {}
    title_to_id = {}
    count = 0

    for fields in iter_insert_tuples(page_sql, "page"):
        page_id = to_int(fields[0])
        ns = to_int(fields[1])
        title = to_str(fields[2])
        if page_id is None or ns is None or title is None:
            continue
        if ns != 0:
            continue
        id_to_title[page_id] = title
        title_to_id[title] = page_id
        count += 1
        if max_pages is not None and count >= max_pages:
            break
        if count % 1_000_000 == 0:
            log(f"page: loaded {count} main-namespace pages")

    return id_to_title, title_to_id


def build_redirect_map(redirect_sql, title_to_id, max_redirects=None):
    """Build redirect source -> target mapping from redirect.sql.gz.

    Only keeps main namespace (ns=0) redirects.

    Returns:
        dict: source_id -> target_id mapping
    """
    redirect_map = {}
    count = 0

    for fields in iter_insert_tuples(redirect_sql, "redirect"):
        rd_from = to_int(fields[0])
        rd_namespace = to_int(fields[1])
        rd_title = to_str(fields[2])
        if rd_from is None or rd_namespace is None or rd_title is None:
            continue
        if rd_namespace != 0:
            continue
        target_id = title_to_id.get(rd_title)
        if target_id is None:
            continue
        redirect_map[rd_from] = target_id
        count += 1
        if max_redirects is not None and count >= max_redirects:
            break
        if count % 1_000_000 == 0:
            log(f"redirect: loaded {count} entries")

    return redirect_map


def resolve_redirect(target_id, redirect_map, max_hops=10):
    """Resolve redirect chains up to max_hops.

    Args:
        target_id: Starting page ID
        redirect_map: source_id -> target_id mapping
        max_hops: Maximum chain length to follow

    Returns:
        Final target page ID
    """
    hops = 0
    while target_id in redirect_map and hops < max_hops:
        target_id = redirect_map[target_id]
        hops += 1
    return target_id


def write_nodes(id_to_title, out_path, batch_size=1_000_000):
    """Write nodes to Parquet file."""
    schema = pa.schema([
        ("page_id", pa.int64()),
        ("title", pa.string()),
    ])

    with ParquetBatchWriter(out_path, schema) as writer:
        batch_ids = []
        batch_titles = []

        for page_id, title in id_to_title.items():
            batch_ids.append(page_id)
            batch_titles.append(title)
            if len(batch_ids) >= batch_size:
                writer.write({"page_id": batch_ids, "title": batch_titles})
                batch_ids = []
                batch_titles = []

        if batch_ids:
            writer.write({"page_id": batch_ids, "title": batch_titles})


def write_edges(
    pagelinks_sql,
    title_to_id,
    redirect_map,
    id_to_title,
    out_path,
    batch_size=2_000_000,
    max_edges=None,
):
    """Write edges to Parquet file.

    Resolves redirects and filters invalid edges.
    """
    schema = pa.schema([
        ("src_id", pa.int64()),
        ("dst_id", pa.int64()),
    ])

    with ParquetBatchWriter(out_path, schema) as writer:
        batch_src = []
        batch_dst = []
        count = 0

        for fields in iter_insert_tuples(pagelinks_sql, "pagelinks"):
            pl_from = to_int(fields[0])
            pl_namespace = to_int(fields[1])
            pl_title = to_str(fields[2])

            if pl_from is None or pl_namespace is None or pl_title is None:
                continue
            if pl_namespace != 0:
                continue
            if pl_from not in id_to_title:
                continue

            dst_id = title_to_id.get(pl_title)
            if dst_id is None:
                continue
            dst_id = resolve_redirect(dst_id, redirect_map)
            if dst_id not in id_to_title:
                continue

            batch_src.append(pl_from)
            batch_dst.append(dst_id)
            count += 1
            if max_edges is not None and count >= max_edges:
                break

            if len(batch_src) >= batch_size:
                writer.write({"src_id": batch_src, "dst_id": batch_dst})
                batch_src = []
                batch_dst = []
                log(f"pagelinks: wrote {count} edges")

        if batch_src:
            writer.write({"src_id": batch_src, "dst_id": batch_dst})

    log(f"pagelinks: total edges written {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Parse Wikipedia SQL dumps into hyperlink graph"
    )
    parser.add_argument("--page-sql", required=True, help="Path to page.sql.gz")
    parser.add_argument("--pagelinks-sql", required=True, help="Path to pagelinks.sql.gz")
    parser.add_argument("--redirect-sql", required=True, help="Path to redirect.sql.gz")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--max-pages", type=int, help="Limit number of pages")
    parser.add_argument("--max-redirects", type=int, help="Limit number of redirects")
    parser.add_argument("--max-edges", type=int, help="Limit number of edges")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    nodes_path = os.path.join(args.out_dir, "nodes.parquet")
    edges_path = os.path.join(args.out_dir, "edges.parquet")

    log("building page maps...")
    id_to_title, title_to_id = build_page_maps(args.page_sql, args.max_pages)
    log(f"main-namespace pages: {len(id_to_title)}")

    log("building redirect map...")
    redirect_map = build_redirect_map(args.redirect_sql, title_to_id, args.max_redirects)
    log(f"redirect entries: {len(redirect_map)}")

    log("writing nodes.parquet...")
    write_nodes(id_to_title, nodes_path)

    log("writing edges.parquet...")
    write_edges(
        args.pagelinks_sql,
        title_to_id,
        redirect_map,
        id_to_title,
        edges_path,
        max_edges=args.max_edges,
    )


if __name__ == "__main__":
    main()
