import argparse
import os
import subprocess
import sys
from multiprocessing import Pool, cpu_count

import pyarrow as pa

from wikikg.common import ParquetBatchWriter, iter_insert_tuples, to_int, to_str
from tqdm import tqdm

# Global variables for worker processes
_worker_cat_title_to_id = None
_worker_cat_page_id_to_title = None
_worker_cat_page_id_to_cat_id = None
_worker_ltid_to_cat_id = None  
_worker_main_page_ids = None


def _init_worker(cat_title_to_id, cat_page_id_to_title, cat_page_id_to_cat_id, ltid_to_cat_id, main_page_ids):
    global _worker_cat_title_to_id, _worker_cat_page_id_to_title
    global _worker_cat_page_id_to_cat_id, _worker_ltid_to_cat_id, _worker_main_page_ids
    _worker_cat_title_to_id = cat_title_to_id
    _worker_cat_page_id_to_title = cat_page_id_to_title
    _worker_cat_page_id_to_cat_id = cat_page_id_to_cat_id
    _worker_ltid_to_cat_id = ltid_to_cat_id
    _worker_main_page_ids = main_page_ids


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def iter_insert_tuples_pigz(path, table_name, encoding="utf-8", desc=None):
    prefix = f"INSERT INTO `{table_name}` VALUES"
    try:
        proc = subprocess.Popen(["pigz", "-dc", path], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        proc = subprocess.Popen(["gzip", "-dc", path], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    with tqdm(desc=desc, unit="line", dynamic_ncols=True) as pbar:
        try:
            for line in proc.stdout:
                line = line.decode(encoding, errors="replace")
                pbar.update(1)
                if not line.startswith(prefix):
                    continue
                values_str = line.split("VALUES", 1)[1].strip()
                if values_str.endswith(";"):
                    values_str = values_str[:-1]
                for tup in _iter_values_fast(values_str):
                    yield tup
        finally:
            proc.stdout.close()
            proc.wait()


def _iter_values_fast(values_str):
    in_string, escape, in_tuple = False, False, False
    field_chars, fields = [], []
    for ch in values_str:
        if in_string:
            if escape:
                field_chars.append(ch); escape = False
            elif ch == "\\": escape = True
            elif ch == "'": in_string = False
            else: field_chars.append(ch)
            continue
        if ch == "'": in_string = True; continue
        if ch == "(": in_tuple = True; fields = []; field_chars = []
        elif ch == ")":
            if in_tuple:
                fields.append("".join(field_chars))
                yield tuple(fields)
                in_tuple = False
        elif ch == ",":
            if in_tuple:
                fields.append("".join(field_chars))
                field_chars = []
        elif in_tuple and ch not in " \n\r\t":
            field_chars.append(ch)


def load_page_maps(page_sql, max_pages=None, use_pigz=True):
    main_page_ids = set()
    cat_page_id_to_title = {}
    iterator = iter_insert_tuples_pigz(page_sql, "page", desc="Loading Pages") if use_pigz else iter_insert_tuples(page_sql, "page", desc="Loading Pages")
    for fields in iterator:
        page_id = to_int(fields[0])
        ns = to_int(fields[1])
        title = to_str(fields[2])
        if page_id is None or ns is None: continue
        if ns == 0:
            if max_pages is None or len(main_page_ids) < max_pages:
                main_page_ids.add(page_id)
        elif ns == 14:
            cat_page_id_to_title[page_id] = title
    return main_page_ids, cat_page_id_to_title


def load_linktarget_map(lt_sql, cat_title_to_id, use_pigz=True):
    ltid_to_cat_id = {}
    iterator = iter_insert_tuples_pigz(lt_sql, "linktarget", desc="Loading LinkTargets") if use_pigz else iter_insert_tuples(lt_sql, "linktarget")
    for fields in iterator:
        lt_id = to_int(fields[0])
        ns = to_int(fields[1])
        title = to_str(fields[2])
        if ns == 14:
            cat_id = cat_title_to_id.get(title)
            if cat_id:
                ltid_to_cat_id[lt_id] = cat_id
    return ltid_to_cat_id


def load_categories(category_sql, max_categories=None, use_pigz=True):
    cat_title_to_id = {}
    categories = []
    iterator = iter_insert_tuples_pigz(category_sql, "category", desc="Loading Categories") if use_pigz else iter_insert_tuples(category_sql, "category")
    for fields in iterator:
        cat_id = to_int(fields[0])
        cat_title = to_str(fields[1])
        cat_pages = to_int(fields[2])
        if cat_id is None or cat_title is None: continue
        cat_title_to_id[cat_title] = cat_id
        categories.append((cat_id, cat_title, cat_pages))
    return categories, cat_title_to_id


# 新增：保存分类节点的函数
def write_categories(categories, out_path, batch_size=1_000_000):
    schema = pa.schema([
        ("category_id", pa.int64()),
        ("title", pa.string()),
        ("page_count", pa.int64()),
    ])

    with ParquetBatchWriter(out_path, schema) as writer:
        for i in range(0, len(categories), batch_size):
            batch = categories[i : i + batch_size]
            writer.write({
                "category_id": [c[0] for c in batch],
                "title": [c[1] for c in batch],
                "page_count": [c[2] if c[2] is not None else 0 for c in batch],
            })


def _process_categorylinks_chunk(tuples_chunk):
    edges = []
    page_cats = []
    for fields in tuples_chunk:
        try:
            cl_from = to_int(fields[0])
            cl_type = to_str(fields[4])
            cl_target_id = to_int(fields[6])
        except (IndexError, ValueError): continue

        if cl_from is None or cl_target_id is None: continue
        parent_cat_id = _worker_ltid_to_cat_id.get(cl_target_id)
        if parent_cat_id is None: continue

        if cl_type == "subcat":
            child_cat_id = _worker_cat_page_id_to_cat_id.get(cl_from)
            if child_cat_id:
                edges.append((parent_cat_id, child_cat_id))
        elif cl_type == "page":
            if cl_from in _worker_main_page_ids:
                page_cats.append((cl_from, parent_cat_id))
    return edges, page_cats


def process_categorylinks(categorylinks_sql, cat_title_to_id, cat_page_id_to_title, 
                          cat_page_id_to_cat_id, ltid_to_cat_id, main_page_ids, 
                          category_edges_path, page_categories_path, num_workers=None, use_pigz=True):
    
    if num_workers is None: num_workers = max(1, cpu_count() - 1)

    edges_schema = pa.schema([("parent_id", pa.int64()), ("child_id", pa.int64())])
    page_cat_schema = pa.schema([("page_id", pa.int64()), ("category_id", pa.int64())])

    edges_writer = ParquetBatchWriter(category_edges_path, edges_schema)
    page_cat_writer = ParquetBatchWriter(page_categories_path, page_cat_schema)

    try:
        with Pool(num_workers, initializer=_init_worker, 
                  initargs=(cat_title_to_id, cat_page_id_to_title, cat_page_id_to_cat_id, ltid_to_cat_id, main_page_ids)) as pool:
            
            iterator = iter_insert_tuples_pigz(categorylinks_sql, "categorylinks", desc="Processing Links") if use_pigz else iter_insert_tuples(categorylinks_sql, "categorylinks")
            chunk, pending = [], []
            
            for fields in iterator:
                chunk.append(fields)
                if len(chunk) >= 100_000:
                    pending.append(pool.apply_async(_process_categorylinks_chunk, (chunk,)))
                    chunk = []
                    if len(pending) >= num_workers * 2:
                        for future in pending:
                            edges, page_cats = future.get()
                            if edges: edges_writer.write({"parent_id": [e[0] for e in edges], "child_id": [e[1] for e in edges]})
                            if page_cats: page_cat_writer.write({"page_id": [p[0] for p in page_cats], "category_id": [p[1] for p in page_cats]})
                        pending = []
            
            if chunk: pending.append(pool.apply_async(_process_categorylinks_chunk, (chunk,)))
            for future in pending:
                edges, page_cats = future.get()
                if edges: edges_writer.write({"parent_id": [e[0] for e in edges], "child_id": [e[1] for e in edges]})
                if page_cats: page_cat_writer.write({"page_id": [p[0] for p in page_cats], "category_id": [p[1] for p in page_cats]})

    finally:
        edges_writer.close()
        page_cat_writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--page-sql", required=True)
    parser.add_argument("--category-sql", required=True)
    parser.add_argument("--linktarget-sql", required=True)
    parser.add_argument("--categorylinks-sql", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    use_pigz = True

    # 1. 加载并保存分类节点
    log("1/5 Loading categories...")
    categories, cat_title_to_id = load_categories(args.category_sql, use_pigz=use_pigz)
    
    log("Saving wiki_categories.parquet...")
    categories_path = os.path.join(args.out_dir, "wiki_categories.parquet")
    write_categories(categories, categories_path)
    log(f"Total categories saved: {len(categories)}")
    
    # 2. 加载页面映射
    log("2/5 Loading page maps...")
    main_page_ids, cat_page_id_to_title = load_page_maps(args.page_sql, use_pigz=use_pigz)
    
    # 3. 加载 Linktarget (2026 结构的关键桥梁)
    log("3/5 Loading linktarget maps (The Bridge)...")
    ltid_to_cat_id = load_linktarget_map(args.linktarget_sql, cat_title_to_id, use_pigz=use_pigz)

    # 4. 构建 PageID 到 CategoryID 的映射
    log("4/5 Building page-to-category ID bridge...")
    cat_page_id_to_cat_id = {}
    for pid, title in cat_page_id_to_title.items():
        if title in cat_title_to_id: 
            cat_page_id_to_cat_id[pid] = cat_title_to_id[title]

    # 5. 处理关系并保存
    log("5/5 Processing categorylinks with 3-table linkage...")
    process_categorylinks(
        args.categorylinks_sql, cat_title_to_id, cat_page_id_to_title,
        cat_page_id_to_cat_id, ltid_to_cat_id, main_page_ids,
        os.path.join(args.out_dir, "wiki_category_edges.parquet"),
        os.path.join(args.out_dir, "wiki_page_categories.parquet"),
        num_workers=args.workers
    )
    log("Done!")

if __name__ == "__main__":
    main()

# #
# python -m wikikg.category.parse_categories \
#   --page-sql data/raw/page.sql.gz \
#   --category-sql data/raw/category.sql.gz \
#   --linktarget-sql data/raw/linktarget.sql.gz \
#   --categorylinks-sql data/raw/categorylinks.sql.gz \
#   --out-dir data/graph/category \
#   --workers 8