"""Steam Games Data Cleaning & Target Construction Module.

Transforms raw scraped Steam metadata into a cleaned, noise-reduced dataset:
- Formulates binary target: 1 if (positive / total_reviews) > 0.70 else 0
- Filters low-volume noise games
- Normalizes string fields (developers, publishers, titles)
- Parses list-like multilabels (tags, genres) into iterables
- Validates and sanitizes numeric features (price, playtime, owners, engagement)
- Exports cleaned CSV and JSON datasets
"""

import argparse
import json
import os
import pandas as pd
import numpy as np


def parse_owners(val):
    """Parse owner ranges like '100,000 .. 200,000' into min, max, midpoint integers."""
    if not isinstance(val, str) or not val.strip():
        return 0, 0, 0
    clean_val = val.replace(",", "").strip()
    if ".." in clean_val:
        parts = clean_val.split("..")
        try:
            low = int(parts[0].strip())
            high = int(parts[1].strip()) if len(parts) > 1 and parts[1].strip() else low
            return low, high, (low + high) // 2
        except ValueError:
            return 0, 0, 0
    try:
        single = int(clean_val)
        return single, single, single
    except ValueError:
        return 0, 0, 0


def clean_dataset(raw_csv_path="data/steam_games_raw.csv", min_reviews=50):
    """Clean the raw Steam games dataset and construct the classification target."""
    print(f"[*] Loading raw dataset from: {raw_csv_path}")
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"Raw data file not found at {raw_csv_path}")

    df = pd.read_csv(raw_csv_path)
    initial_count = len(df)
    print(f"    Raw records loaded: {initial_count}")

    # 1. Clean string identifiers
    df["name"] = df["name"].fillna("Unknown Title").astype(str).str.strip()
    df["developer"] = df["developer"].fillna("Unknown").astype(str).str.strip()
    df["publisher"] = df["publisher"].fillna("Unknown").astype(str).str.strip()
    df["developer"] = df["developer"].replace("", "Unknown")
    df["publisher"] = df["publisher"].replace("", "Unknown")

    # 2. Review counts and noise reduction
    df["positive"] = pd.to_numeric(df["positive"], errors="coerce").fillna(0).astype(int)
    df["negative"] = pd.to_numeric(df["negative"], errors="coerce").fillna(0).astype(int)
    df["total_reviews"] = df["positive"] + df["negative"]

    # Filter games with negligible reviews to reduce statistical noise
    df_clean = df[df["total_reviews"] >= min_reviews].copy()
    dropped_noise = initial_count - len(df_clean)
    print(f"    Filtered {dropped_noise} records with < {min_reviews} reviews ({len(df_clean)} remaining)")

    # 3. Target Formulation
    # target = 1 if (positive / total_reviews) > 0.70 else 0
    df_clean["review_score_pct"] = (df_clean["positive"] / df_clean["total_reviews"]).round(4)
    df_clean["target"] = (df_clean["review_score_pct"] > 0.70).astype(int)

    # 4. Parse Multilabel Features (Genres & Tags)
    def parse_genres(g_str):
        if not isinstance(g_str, str) or not g_str.strip():
            return []
        return [g.strip() for g in g_str.split(",") if g.strip()]

    def parse_tags(t_str):
        if not isinstance(t_str, str) or not t_str.strip():
            return []
        return [t.strip() for t in t_str.split(";") if t.strip()]

    df_clean["genres_list"] = df_clean["genres"].apply(parse_genres)
    df_clean["tags_list"] = df_clean["tags"].apply(parse_tags)
    df_clean["genre_count"] = df_clean["genres_list"].apply(len)
    df_clean["tag_count"] = df_clean["tags_list"].apply(len)
    
    # Clean serialized representations for CSV storage
    df_clean["genres"] = df_clean["genres_list"].apply(lambda lst: ", ".join(lst))
    df_clean["tags"] = df_clean["tags_list"].apply(lambda lst: ";".join(lst))

    # 5. Pricing & Discounts
    df_clean["price"] = pd.to_numeric(df_clean["price"], errors="coerce").fillna(0.0).round(2)
    df_clean["initial_price"] = pd.to_numeric(df_clean["initial_price"], errors="coerce").fillna(df_clean["price"]).round(2)
    df_clean["discount"] = pd.to_numeric(df_clean["discount"], errors="coerce").fillna(0.0).round(1)
    df_clean["is_free"] = (df_clean["price"] == 0.0).astype(int)

    # 6. Playtime & Engagement (impute negative or missing with median)
    df_clean["average_forever"] = pd.to_numeric(df_clean["average_forever"], errors="coerce").fillna(0).astype(int)
    df_clean["median_forever"] = pd.to_numeric(df_clean["median_forever"], errors="coerce").fillna(0).astype(int)
    df_clean["average_2weeks"] = pd.to_numeric(df_clean["average_2weeks"], errors="coerce").fillna(0).astype(int)
    df_clean["ccu"] = pd.to_numeric(df_clean["ccu"], errors="coerce").fillna(0).astype(int)

    # 7. Owners Parsing
    owner_bounds = df_clean["owners"].apply(parse_owners)
    df_clean["owners_min"] = [b[0] for b in owner_bounds]
    df_clean["owners_max"] = [b[1] for b in owner_bounds]
    df_clean["owners_midpoint"] = [b[2] for b in owner_bounds]

    # Reorder columns logically
    columns_order = [
        "appid", "name", "developer", "publisher", "genres", "tags",
        "genre_count", "tag_count", "price", "initial_price", "discount", "is_free",
        "positive", "negative", "total_reviews", "review_score_pct", "target",
        "owners", "owners_min", "owners_max", "owners_midpoint",
        "average_forever", "median_forever", "average_2weeks", "ccu"
    ]
    df_clean = df_clean[[c for c in columns_order if c in df_clean.columns]]

    # Print summary profile
    pos_count = (df_clean["target"] == 1).sum()
    neg_count = (df_clean["target"] == 0).sum()
    print("\n=== Cleaned Dataset Profile ===")
    print(f"Total Cleaned Records: {len(df_clean)}")
    print(f"Target = 1 (> 70% positive): {pos_count} ({pos_count/len(df_clean)*100:.1f}%)")
    print(f"Target = 0 (<= 70% positive): {neg_count} ({neg_count/len(df_clean)*100:.1f}%)")
    print(f"Class Imbalance Ratio (1:0): {pos_count / max(1, neg_count):.2f} to 1")
    print(f"Free-to-Play Games: {(df_clean['is_free'] == 1).sum()}")
    print(f"Paid Games: {(df_clean['is_free'] == 0).sum()} (Median Price: ${df_clean[df_clean['is_free'] == 0]['price'].median():.2f})")
    print(f"Average Playtime (median forever): {df_clean['median_forever'].median():.1f} mins")
    print("===============================\n")

    return df_clean


def save_cleaned_dataset(df_clean, out_csv="data/steam_games_cleaned.csv", out_json="data/steam_games_cleaned.json"):
    """Export cleaned data to CSV and JSON formats."""
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)

    df_clean.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[+] Saved cleaned CSV: {out_csv} ({len(df_clean)} rows)")

    # For JSON, include parsed lists for tags and genres
    records = df_clean.to_dict(orient="records")
    for r in records:
        r["genres_list"] = [g.strip() for g in r["genres"].split(",") if g.strip()] if isinstance(r.get("genres"), str) else []
        r["tags_list"] = [t.strip() for t in r["tags"].split(";") if t.strip()] if isinstance(r.get("tags"), str) else []

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"[+] Saved cleaned JSON: {out_json} ({len(records)} records)")


def main():
    parser = argparse.ArgumentParser(description="Steam Games Data Cleaner")
    parser.add_argument("--input", type=str, default="data/steam_games_raw.csv", help="Input raw CSV path")
    parser.add_argument("--output-csv", type=str, default="data/steam_games_cleaned.csv", help="Output cleaned CSV path")
    parser.add_argument("--output-json", type=str, default="data/steam_games_cleaned.json", help="Output cleaned JSON path")
    parser.add_argument("--min-reviews", type=int, default=50, help="Minimum total reviews threshold")

    args = parser.parse_args()
    df_clean = clean_dataset(raw_csv_path=args.input, min_reviews=args.min_reviews)
    save_cleaned_dataset(df_clean, out_csv=args.output_csv, out_json=args.output_json)


if __name__ == "__main__":
    main()
