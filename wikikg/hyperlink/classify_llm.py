"""Classify nodes by visual generatability using Qwen API.

Qwen-Turbo is chosen for:
- Low cost (~$0.0008/1K tokens)
- Fast inference
- Good instruction following for simple classification tasks
"""

import argparse
import json
import sys
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from openai import OpenAI


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


SYSTEM_PROMPT = """You evaluate whether concepts can be clearly depicted in a single image.

A concept is VISUALIZABLE if:
- Concrete object, person, animal, plant, place, building, vehicle, food, etc.
- A scene or action that can be photographed
- Something a human would recognize from an image alone

A concept is NOT VISUALIZABLE if:
- Abstract ideas (theories, concepts, principles, emotions)
- Meta/structural pages (lists, indexes, timelines, categories)
- Languages, alphabets, time periods
- Processes or relationships that require diagrams
- Awards, titles, or honors (the concept itself, not a physical trophy)

Respond with a JSON array. Each element: {"title": "...", "visual": true/false}

Be concise. No explanations needed."""


def create_client(api_key: str = None, base_url: str = None) -> OpenAI:
    """Create Qwen API client."""
    api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("Set DASHSCOPE_API_KEY environment variable or pass --api-key")
    
    base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    
    return OpenAI(api_key=api_key, base_url=base_url)


def classify_batch(
    client: OpenAI,
    titles: list[str],
    model: str = "qwen-turbo",
    max_retries: int = 3,
    verbose: bool = False
) -> list[dict]:
    """Classify a batch of titles."""
    
    # Format titles for prompt
    prompt = "Classify these concepts:\n\n"
    for i, title in enumerate(titles, 1):
        clean_title = title.replace("_", " ")
        prompt += f"{i}. {clean_title}\n"
    
    if verbose:
        log(f" Sending batch of {len(titles)} titles...")
        
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2048,
            )
            
            text = response.choices[0].message.content.strip()
            
            if verbose:
                log(f" Response: {text[:200]}...")
            
            # Extract JSON array
            start = text.find("[")
            end = text.rfind("]") + 1
            if start == -1 or end == 0:
                raise ValueError(f"No JSON array found: {text[:200]}")
            
            results = json.loads(text[start:end])
            
            # Validate and align with input titles
            if len(results) != len(titles):
                # Try to match by title
                result_map = {r.get("title", "").replace(" ", "_"): r for r in results}
                aligned = []
                for t in titles:
                    clean = t.replace("_", " ")
                    match = result_map.get(t) or result_map.get(clean)
                    if match:
                        aligned.append({"title": t, "visual": match.get("visual", False)})
                    else:
                        aligned.append({"title": t, "visual": None})
                return aligned
            
            # Add original titles back
            for r, t in zip(results, titles):
                r["title"] = t
            
            return results
            
        except json.JSONDecodeError as e:
            log(f" JSON parse error (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                return [{"title": t, "visual": None} for t in titles]
            time.sleep(1)
            
        except Exception as e:
            log(f" API error (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                return [{"title": t, "visual": None} for t in titles]
            time.sleep(2 ** attempt)
            
    return [{"title": t, "visual": None} for t in titles]


def classify_all(
    client: OpenAI,
    titles: list[str],
    model: str = "qwen-turbo",
    batch_size: int = 30,
    max_workers: int = 8,
    checkpoint_dir: str = None,
    checkpoint_freq: int = 100,
    verbose: bool = False
) -> dict[str, bool | None]:
    """Classify all titles with parallel processing and checkpointing."""
    
    results = {}
    start_batch = 0
    
    if checkpoint_dir:
        checkpoint_path = Path(checkpoint_dir) / "classify_checkpoint.json"
        if checkpoint_path.exists():
            log(f"Loading checkpoint from {checkpoint_path}")
            with open(checkpoint_path, "r") as f:
                ckpt = json.load(f)
                results = ckpt.get("results", {})
                start_batch = ckpt.get("next_batch", 0)
            log(f" Resuming from batch {start_batch}, {len(results)} already classified")
    
    remaining_titles = [t for t in titles if t not in results]
    log(f"Titles to classify: {len(remaining_titles):,}")
    
    if not remaining_titles:
        log("All titles already classified!")
        return results
    
    batches = [remaining_titles[i:i + batch_size]
               for i in range(0, len(remaining_titles), batch_size)]
    
    log(f"Processing {len(batches):,} batches with {max_workers} workers")
    
    processed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i, batch in enumerate(batches):
            future = executor.submit(classify_batch, client, batch, model, 3, verbose)
            futures[future] = (i, batch)
        
        with tqdm(total=len(batches), desc="Classifying", unit="batch") as pbar:
            for future in as_completed(futures):
                batch_idx, batch_titles = futures[future]
                try:
                    batch_results = future.result()
                    for r in batch_results:
                        results[r["title"]] = r.get("visual")
                    if verbose:
                        for r in batch_results:
                            status = "✓" if r.get("visual") else "✗" if r.get("visual") is False else "?"
                            log(f" {status} {r['title']}")
                except Exception as e:
                    log(f" Batch {batch_idx} failed: {e}")
                    for t in batch_titles:
                        results[t] = None
                
                processed += 1
                pbar.update(1)
                
                if checkpoint_dir and processed % checkpoint_freq == 0:
                    checkpoint_path = Path(checkpoint_dir) / "classify_checkpoint.json"
                    with open(checkpoint_path, "w") as f:
                        json.dump({
                            "results": results,
                            "next_batch": batch_idx + 1,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        }, f)
    
    if checkpoint_dir:
        checkpoint_path = Path(checkpoint_dir) / "classify_checkpoint.json"
        with open(checkpoint_path, "w") as f:
            json.dump({
                "results": results,
                "complete": True,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }, f)
        log(f"Saved final checkpoint to {checkpoint_path}")
        
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Classify nodes by visual generatability using Qwen API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--nodes", required=True,
                        help="Input nodes.parquet (uncertain nodes needing classification)")
    parser.add_argument("--out-visual", required=True,
                        help="Output: nodes classified as visual")
    parser.add_argument("--out-nonvisual", required=True,
                        help="Output: nodes classified as non-visual")
    parser.add_argument("--out-failed", type=str, default=None,
                        help="Output: nodes that failed classification (optional)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Qwen API key (or set DASHSCOPE_API_KEY)")
    parser.add_argument("--model", type=str, default="qwen-turbo",
                        choices=["qwen-turbo", "qwen-plus", "qwen-max"],
                        help="Qwen model to use")
    parser.add_argument("--batch-size", type=int, default=30,
                        help="Titles per API call")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel API workers")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Directory for checkpoints")
    
    # Testing options
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of nodes to process (for testing)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed output for each classification")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load data and show what would be processed, but don't call API")
    
    args = parser.parse_args()
    start_time = time.time()
    
    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    log(f"Loading nodes from {args.nodes}")
    table = pq.read_table(args.nodes)
    page_ids = table.column("page_id").to_pylist()
    titles = table.column("title").to_pylist()
    log(f" Loaded {len(titles):,} nodes total")
    
    if args.limit:
        page_ids = page_ids[:args.limit]
        titles = titles[:args.limit]
        log(f" Limited to {len(titles):,} nodes (--limit {args.limit})")
    
    title_to_pid = dict(zip(titles, page_ids))
    
    if args.dry_run:
        log("\n=== DRY RUN ===")
        log(f"Would process: {len(titles):,} nodes")
        log(f"Batches: {(len(titles) + args.batch_size - 1) // args.batch_size:,}")
        log(f"Model: {args.model}")
        log(f"Workers: {args.workers}")
        log("\nSample titles:")
        for t in titles[:20]:
            log(f" - {t}")
        
        est_input_tokens = len(titles) * 25
        est_output_tokens = len(titles) * 15
        pricing = {
            "qwen-turbo": (0.0008, 0.002),
            "qwen-plus": (0.004, 0.012),
            "qwen-max": (0.02, 0.06),
        }
        input_price, output_price = pricing.get(args.model, (0.001, 0.002))
        est_cost = (est_input_tokens / 1000 * input_price +
                    est_output_tokens / 1000 * output_price)
        log(f"\nEstimated cost: ~${est_cost:.4f}")
        return
    
    client = create_client(args.api_key)
    log(f"Using model: {args.model}")
    
    results = classify_all(
        client=client,
        titles=titles,
        model=args.model,
        batch_size=args.batch_size,
        max_workers=args.workers,
        checkpoint_dir=args.checkpoint_dir,
        verbose=args.verbose
    )
    
    visual_titles = [t for t, v in results.items() if v is True]
    nonvisual_titles = [t for t, v in results.items() if v is False]
    failed_titles = [t for t, v in results.items() if v is None]
    
    log(f"\nResults:")
    log(f" Visual: {len(visual_titles):,} ({100*len(visual_titles)/len(titles):.1f}%)")
    log(f" Non-visual: {len(nonvisual_titles):,} ({100*len(nonvisual_titles)/len(titles):.1f}%)")
    log(f" Failed: {len(failed_titles):,} ({100*len(failed_titles)/len(titles):.1f}%)")
    
    if args.verbose:
        log("\nSample VISUAL:")
        for t in visual_titles[:10]:
            log(f" ✓ {t}")
        log("\nSample NON-VISUAL:")
        for t in nonvisual_titles[:10]:
            log(f" ✗ {t}")
            
    log("\nSaving outputs...")
    
    def save_nodes(titles_list, out_path):
        pids = [title_to_pid[t] for t in titles_list if t in title_to_pid]
        df_titles = [t for t in titles_list if t in title_to_pid]
        out_table = pa.table({"page_id": pids, "title": df_titles})
        pq.write_table(out_table, out_path, compression="zstd")
        log(f" Wrote {len(pids):,} nodes to {out_path}")
        
    save_nodes(visual_titles, args.out_visual)
    save_nodes(nonvisual_titles, args.out_nonvisual)
    
    if args.out_failed and failed_titles:
        save_nodes(failed_titles, args.out_failed)
        
    total_time = time.time() - start_time
    log(f"\nTotal time: {total_time:.1f} seconds")


if __name__ == "__main__":
    main()