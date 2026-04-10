import argparse
import os
import subprocess
import sys
from multiprocessing import Pool, cpu_count
import pyarrow as pa
from tqdm import tqdm

# 依赖 Common 工具函数
from wikikg.common import ParquetBatchWriter, iter_insert_tuples, to_int, to_str

# --- 全局变量：用于子进程共享内存 ---
_worker_title_to_id = None
_worker_redirect_map = None
_worker_id_to_title = None
_worker_ltid_to_title = None

def _init_worker(title_to_id, redirect_map, id_to_title, ltid_to_title):
    global _worker_title_to_id, _worker_redirect_map, _worker_id_to_title, _worker_ltid_to_title
    _worker_title_to_id = title_to_id
    _worker_redirect_map = redirect_map
    _worker_id_to_title = id_to_title
    _worker_ltid_to_title = ltid_to_title

class ByteProgressReader:
    """监控底层解压流字节进度的包装器"""
    def __init__(self, fileobj, pbar):
        self.fileobj = fileobj
        self.pbar = pbar

    def read(self, size=-1):
        chunk = self.fileobj.read(size)
        self.pbar.update(len(chunk))
        return chunk

    def __iter__(self):
        for line in self.fileobj:
            self.pbar.update(len(line))
            yield line

def open_decompressed(path, pbar):
    """支持 pigz 并行解压"""
    try:
        proc = subprocess.Popen(["pigz", "-dc", path], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        proc = subprocess.Popen(["gzip", "-dc", path], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return proc, ByteProgressReader(proc.stdout, pbar)

# --- 核心加载逻辑 ---

def load_linktarget_map(lt_sql):
    ltid_to_title = {}
    total_size = os.path.getsize(lt_sql)
    
    with tqdm(total=total_size, unit='B', unit_scale=True, desc="1/4 Loading LinkTargets") as pbar:
        # 1. 这里启动外部解压
        proc, reader = open_decompressed(lt_sql, pbar)
        try:
            # 2. 将解压后的 reader 传给解析函数
            for fields in iter_insert_tuples(reader, "linktarget"):
                lt_id = to_int(fields[0])
                ns = to_int(fields[1])
                title = to_str(fields[2])
                if ns == 0:
                    ltid_to_title[lt_id] = title
        finally:
            proc.terminate() # 确保关闭子进程
    return ltid_to_title

def build_page_maps(page_sql):
    id_to_title, title_to_id = {}, {}
    total_size = os.path.getsize(page_sql)
    with tqdm(total=total_size, unit='B', unit_scale=True, desc="2/4 Loading Pages") as pbar:
        proc, reader = open_decompressed(page_sql, pbar)
        for fields in iter_insert_tuples(reader, "page"):
            pid, ns, title = to_int(fields[0]), to_int(fields[1]), to_str(fields[2])
            if ns == 0:
                id_to_title[pid] = title
                title_to_id[title] = pid
        proc.wait()
    return id_to_title, title_to_id

def build_redirect_map(redirect_sql, title_to_id):
    redirect_map = {}
    total_size = os.path.getsize(redirect_sql)
    with tqdm(total=total_size, unit='B', unit_scale=True, desc="3/4 Loading Redirects") as pbar:
        proc, reader = open_decompressed(redirect_sql, pbar)
        for fields in iter_insert_tuples(reader, "redirect"):
            rd_from, rd_ns, rd_title = to_int(fields[0]), to_int(fields[1]), to_str(fields[2])
            if rd_ns == 0:
                target_id = title_to_id.get(rd_title)
                if target_id: redirect_map[rd_from] = target_id
        proc.wait()
    return redirect_map

# --- 并行处理逻辑 ---

def _process_chunk_worker(chunk):
    """2026 Schema 修正版：索引对齐 pl_from(0), pl_from_ns(1), pl_target_id(2)"""
    batch_src, batch_dst = [], []
    for fields in chunk:
        try:
            # 2026 关键索引修复
            pl_from = to_int(fields[0])
            pl_from_ns = to_int(fields[1]) 
            pl_target_id = to_int(fields[2]) # 必须是索引 2

            # 过滤：我们只关心从百科条目（NS=0）发出的链接
            if pl_from_ns != 0:
                continue

            # 1. 通过 Linktarget 找到目标标题
            target_title = _worker_ltid_to_title.get(pl_target_id)
            if not target_title:
                continue

            # 2. 映射标题到 Page ID
            dst_id = _worker_title_to_id.get(target_title)
            if dst_id is None:
                continue

            # 3. 解析重定向
            dst_id = _worker_redirect_map.get(dst_id, dst_id)
            
            # 4. 验证：源和目标都在我们的 nodes 范围内
            if pl_from in _worker_id_to_title and dst_id in _worker_id_to_title:
                batch_src.append(pl_from)
                batch_dst.append(dst_id)
        except (IndexError, ValueError):
            continue
    return batch_src, batch_dst

def write_edges_parallel(pagelinks_sql, title_to_id, redirect_map, id_to_title, ltid_to_title, out_path, num_workers=None):
    if num_workers is None: num_workers = max(1, cpu_count() - 1)
    schema = pa.schema([("src_id", pa.int64()), ("dst_id", pa.int64())])
    writer = ParquetBatchWriter(out_path, schema)
    
    total_size = os.path.getsize(pagelinks_sql)
    with tqdm(total=total_size, unit='B', unit_scale=True, desc="4/4 Parsing Pagelinks (2026)") as pbar:
        proc, reader = open_decompressed(pagelinks_sql, pbar)
        with Pool(processes=num_workers, initializer=_init_worker, 
                  initargs=(title_to_id, redirect_map, id_to_title, ltid_to_title)) as pool:
            chunk, futures = [], []
            for fields in iter_insert_tuples(reader, "pagelinks"):
                chunk.append(fields)
                if len(chunk) >= 100000:
                    futures.append(pool.apply_async(_process_chunk_worker, (chunk,)))
                    chunk = []
                    if len(futures) >= num_workers * 2:
                        for f in futures:
                            s, d = f.get()
                            if s: writer.write({"src_id": s, "dst_id": d})
                        futures = []
            if chunk: futures.append(pool.apply_async(_process_chunk_worker, (chunk,)))
            for f in futures:
                s, d = f.get()
                if s: writer.write({"src_id": s, "dst_id": d})
        writer.close()
        proc.wait()

def main():
    parser = argparse.ArgumentParser(description="Wikipedia 2026 Hyperlink Graph Parser")
    parser.add_argument("--page-sql", required=True)
    parser.add_argument("--linktarget-sql", required=True, help="New for 2026")
    parser.add_argument("--pagelinks-sql", required=True)
    parser.add_argument("--redirect-sql", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    # 执行流水线
    ltid_to_title = load_linktarget_map(args.linktarget_sql)
    id_to_title, title_to_id = build_page_maps(args.page_sql)
    redirect_map = build_redirect_map(args.redirect_sql, title_to_id)
    
    # 保存节点
    pa.parquet.write_table(pa.Table.from_pydict({
        "page_id": list(id_to_title.keys()), 
        "title": list(id_to_title.values())
    }), os.path.join(args.out_dir, "nodes.parquet"))

    # 并行处理边
    write_edges_parallel(args.pagelinks_sql, title_to_id, redirect_map, id_to_title, ltid_to_title, 
                         os.path.join(args.out_dir, "edges.parquet"), num_workers=args.workers)

if __name__ == "__main__":
    main()