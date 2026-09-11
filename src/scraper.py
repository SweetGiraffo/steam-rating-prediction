"""Steam Games Data Acquisition Module.

Scrapes game catalog and detailed metadata (tags, genres, reviews, playtime,
pricing, developers, publishers) from SteamSpy API with rate-limiting,
concurrency, retry logic, and continuous checkpointing.
"""

import argparse
import csv
import json
import os
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def fetch_json(url, max_retries=3, backoff_factor=1.5):
    """Fetch JSON with retry and exponential backoff."""
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=12) as response:
                content = response.read().decode("utf-8")
                return json.loads(content)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            if attempt == max_retries - 1:
                return None
            time.sleep(backoff_factor * (attempt + 1))
    return None


def fetch_catalog(num_pages=4, min_reviews=20):
    """Fetch broad catalog of games across multiple SteamSpy pages.
    
    Each page returns ~1000 games.
    """
    catalog = {}
    print(f"[*] Fetching base catalog across {num_pages} SteamSpy pages...")
    for page in range(num_pages):
        url = f"https://steamspy.com/api.php?request=all&page={page}"
        data = fetch_json(url)
        if not data:
            print(f"    [!] Warning: Failed to retrieve catalog page {page}")
            continue
        
        added = 0
        for appid_str, item in data.items():
            try:
                aid = int(appid_str)
                pos = int(item.get("positive", 0))
                neg = int(item.get("negative", 0))
                if pos + neg >= min_reviews:
                    catalog[aid] = item
                    added += 1
            except (ValueError, TypeError):
                continue
        print(f"    -> Page {page}: fetched {len(data)} items, {added} meet min_reviews threshold. Total so far: {len(catalog)}")
        time.sleep(0.5)

    return catalog


def load_checkpoint(checkpoint_path):
    """Load previously saved enriched game details from disk."""
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Ensure keys are ints
                return {int(k): v for k, v in data.items()}
        except Exception as e:
            print(f"[!] Warning: Could not read checkpoint file ({e}). Starting fresh.")
    return {}


def save_checkpoint(enriched_data, checkpoint_path):
    """Save enriched game details to disk atomically."""
    temp_path = f"{checkpoint_path}.tmp"
    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(enriched_data, f, ensure_ascii=False)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    os.rename(temp_path, checkpoint_path)


def enrich_appids(appids, checkpoint_path, max_workers=3, delay=0.15):
    """Fetch detailed tags, genre, and metadata for a list of appids using a thread pool."""
    enriched = load_checkpoint(checkpoint_path)
    print(f"[*] Loaded {len(enriched)} previously enriched games from checkpoint.")
    
    pending_appids = [aid for aid in appids if aid not in enriched]
    print(f"[*] {len(pending_appids)} games need enrichment...")

    if not pending_appids:
        return enriched

    lock = Lock()
    completed_count = 0
    total_needed = len(pending_appids)
    start_time = time.time()

    def worker(appid):
        time.sleep(delay)
        url = f"https://steamspy.com/api.php?request=appdetails&appid={appid}"
        details = fetch_json(url)
        return appid, details

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, aid): aid for aid in pending_appids}
        for future in as_completed(futures):
            aid = futures[future]
            try:
                appid, details = future.result()
                if details and isinstance(details, dict):
                    with lock:
                        enriched[appid] = details
                        completed_count += 1
                        if completed_count % 25 == 0 or completed_count == total_needed:
                            save_checkpoint(enriched, checkpoint_path)
                            elapsed = time.time() - start_time
                            rate = completed_count / max(1.0, elapsed)
                            rem_seconds = (total_needed - completed_count) / max(0.01, rate)
                            print(f"    [+] Enriched {completed_count}/{total_needed} ({completed_count/total_needed*100:.1f}%) | {rate:.1f} games/s | ETA: {rem_seconds/60:.1f}m")
            except Exception as exc:
                print(f"    [!] Error enriching appid {aid}: {exc}")

    # Final checkpoint save
    save_checkpoint(enriched, checkpoint_path)
    return enriched


def parse_owners_range(owners_str):
    """Parse owner ranges like '100,000 .. 200,000' into min and max integers."""
    if not owners_str:
        return 0, 0
    parts = [p.strip().replace(",", "") for p in owners_str.split("..")]
    try:
        min_owners = int(parts[0]) if len(parts) > 0 else 0
        max_owners = int(parts[1]) if len(parts) > 1 else min_owners
        return min_owners, max_owners
    except (ValueError, IndexError):
        return 0, 0


def format_records(catalog, enriched):
    """Merge catalog items with detailed tags and clean schema."""
    records = []
    for appid, item in catalog.items():
        if appid not in enriched:
            continue
        
        detail = enriched[appid]
        pos = int(detail.get("positive", item.get("positive", 0)))
        neg = int(detail.get("negative", item.get("negative", 0)))
        total_rev = pos + neg
        score_pct = round(pos / total_rev, 4) if total_rev > 0 else 0.0

        # Tags: stored as list of tag names sorted by vote count
        raw_tags = detail.get("tags")
        if isinstance(raw_tags, dict):
            # Sort tags by vote count descending
            sorted_tags = sorted(raw_tags.keys(), key=lambda t: raw_tags[t] if isinstance(raw_tags[t], int) else 0, reverse=True)
            tags_list = sorted_tags
            tags_str = ";".join(sorted_tags)
        elif isinstance(raw_tags, list):
            tags_list = raw_tags
            tags_str = ";".join(raw_tags)
        else:
            tags_list = []
            tags_str = ""

        # Genres: comma-separated
        genre_str = detail.get("genre") or item.get("genre") or ""

        # Pricing: SteamSpy price is in cents (e.g. 999 = $9.99) or string
        def parse_price(val):
            try:
                v = float(val)
                return round(v / 100.0, 2)
            except (ValueError, TypeError):
                return 0.0

        price = parse_price(detail.get("price", item.get("price", 0)))
        initial_price = parse_price(detail.get("initialprice", item.get("initialprice", 0)))
        discount = float(detail.get("discount", item.get("discount", 0)) or 0)

        # Playtime in minutes
        avg_forever = int(detail.get("average_forever", item.get("average_forever", 0)))
        med_forever = int(detail.get("median_forever", item.get("median_forever", 0)))
        avg_2weeks = int(detail.get("average_2weeks", item.get("average_2weeks", 0)))

        owners_str = detail.get("owners", item.get("owners", ""))
        min_owners, max_owners = parse_owners_range(owners_str)
        owners_midpoint = (min_owners + max_owners) // 2

        record = {
            "appid": appid,
            "name": detail.get("name", item.get("name", "")).strip(),
            "developer": detail.get("developer", item.get("developer", "")).strip(),
            "publisher": detail.get("publisher", item.get("publisher", "")).strip(),
            "genres": genre_str,
            "tags": tags_str,
            "tag_count": len(tags_list),
            "price": price,
            "initial_price": initial_price,
            "discount": discount,
            "positive": pos,
            "negative": neg,
            "total_reviews": total_rev,
            "rating_pct": score_pct,
            "owners": owners_str,
            "owners_midpoint": owners_midpoint,
            "average_forever": avg_forever,
            "median_forever": med_forever,
            "average_2weeks": avg_2weeks,
            "ccu": int(detail.get("ccu", item.get("ccu", 0)))
        }
        records.append(record)

    return records


def save_dataset(records, csv_path, json_path):
    """Save dataset to both CSV and JSON formats."""
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)

    if not records:
        print("[!] No records to save.")
        return

    # Write JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"[+] Saved {len(records)} games to JSON: {json_path}")

    # Write CSV
    fieldnames = list(records[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"[+] Saved {len(records)} games to CSV: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Steam Games Scraper & Dataset Builder")
    parser.add_argument("--target", type=int, default=2500, help="Target number of games to scrape (default: 2500)")
    parser.add_argument("--pages", type=int, default=4, help="Number of catalog pages to fetch (default: 4)")
    parser.add_argument("--min-reviews", type=int, default=25, help="Minimum total reviews for a game (default: 25)")
    parser.add_argument("--workers", type=int, default=3, help="Concurrent workers for enrichment (default: 3)")
    parser.add_argument("--delay", type=float, default=0.15, help="Delay between requests per worker (default: 0.15s)")
    parser.add_argument("--output-csv", type=str, default="data/steam_games_raw.csv", help="Output CSV path")
    parser.add_argument("--output-json", type=str, default="data/steam_games_raw.json", help="Output JSON path")
    parser.add_argument("--checkpoint", type=str, default="data/enrichment_checkpoint.json", help="Checkpoint file path")
    parser.add_argument("--test", action="store_true", help="Run quick 10-game test")

    args = parser.parse_args()

    if args.test:
        print("[*] Running in TEST mode (10 games)...")
        args.target = 10
        args.pages = 1
        args.output_csv = "data/test_games.csv"
        args.output_json = "data/test_games.json"
        args.checkpoint = "data/test_checkpoint.json"

    # Step 1: Catalog
    catalog = fetch_catalog(num_pages=args.pages, min_reviews=args.min_reviews)
    print(f"[*] Total eligible games in catalog: {len(catalog)}")

    # Prioritize games by engagement (review count)
    sorted_appids = sorted(
        catalog.keys(),
        key=lambda aid: int(catalog[aid].get("positive", 0)) + int(catalog[aid].get("negative", 0)),
        reverse=True
    )
    
    target_appids = sorted_appids[:args.target]
    print(f"[*] Selected top {len(target_appids)} games by review count for enrichment.")

    # Step 2: Enrichment
    enriched = enrich_appids(
        target_appids,
        checkpoint_path=args.checkpoint,
        max_workers=args.workers,
        delay=args.delay
    )

    # Step 3: Formatting & Export
    records = format_records(catalog, enriched)
    print(f"[*] Formatted {len(records)} fully enriched game records.")

    save_dataset(records, args.output_csv, args.output_json)

    # Summary Stats
    if records:
        avg_reviews = sum(r["total_reviews"] for r in records) / len(records)
        avg_tags = sum(r["tag_count"] for r in records) / len(records)
        print("\n=== Dataset Summary ===")
        print(f"Total Games Scraped: {len(records)}")
        print(f"Average Reviews per Game: {avg_reviews:.1f}")
        print(f"Average Tags per Game: {avg_tags:.1f}")
        print(f"Sample Game: {records[0]['name']} (Rating: {records[0]['rating_pct']*100:.1f}%, Tags: {records[0]['tags'][:40]}...)")
        print("=======================\n")


if __name__ == "__main__":
    main()
