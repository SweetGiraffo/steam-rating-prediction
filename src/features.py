"""Steam Games Feature Engineering & Selection Module.

Constructs feature matrices for modeling:
1. Multi-Label Binarization for tags and genres
2. Derived economic and engagement ratios:
   - discount_percentage
   - price_per_hour_playtime
   - engagement_rate (total_reviews / estimated owners)
   - ccu_to_owners_ratio
   - log-transformed continuous distributions (playtime, CCU, owners)
3. Variance Thresholding: drops low-frequency tags (<1% presence)
4. Feature Selection: Preliminary Random Forest / L1 feature importance ranking
5. Exports both full feature matrix and selected feature matrix
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def parse_tag_list(val):
    """Safely parse semicolon-delimited tags into a list."""
    if not isinstance(val, str) or not val.strip():
        return []
    return [t.strip() for t in val.split(";") if t.strip()]


def parse_genre_list(val):
    """Safely parse comma-delimited genres into a list."""
    if not isinstance(val, str) or not val.strip():
        return []
    return [g.strip() for g in val.split(",") if g.strip()]


def engineer_features(cleaned_csv_path="data/steam_games_cleaned.csv"):
    """Transform cleaned game records into an engineered numerical feature matrix."""
    print(f"[*] Loading cleaned data from: {cleaned_csv_path}")
    df = pd.read_csv(cleaned_csv_path)
    total_games = len(df)
    print(f"    Loaded {total_games} games.")

    # 1. MultiLabel Binarization for Tags
    print("[*] Performing Multi-Label Binarization on community tags...")
    tags_series = df["tags"].apply(parse_tag_list)
    mlb_tags = MultiLabelBinarizer()
    tag_matrix = mlb_tags.fit_transform(tags_series)
    tag_cols = [f"tag_{t.replace(' ', '_').replace('-', '_')}" for t in mlb_tags.classes_]
    df_tags = pd.DataFrame(tag_matrix, columns=tag_cols, index=df.index)
    print(f"    Created {len(tag_cols)} binary tag features.")

    # 2. MultiLabel Binarization for Genres
    print("[*] Performing Multi-Label Binarization on genres...")
    genres_series = df["genres"].apply(parse_genre_list)
    mlb_genres = MultiLabelBinarizer()
    genre_matrix = mlb_genres.fit_transform(genres_series)
    genre_cols = [f"genre_{g.replace(' ', '_').replace('-', '_')}" for g in mlb_genres.classes_]
    df_genres = pd.DataFrame(genre_matrix, columns=genre_cols, index=df.index)
    print(f"    Created {len(genre_cols)} binary genre features.")

    # 3. Derived Engineering Features
    print("[*] Computing derived domain features and ratios...")
    df_engineered = pd.DataFrame(index=df.index)

    # Core identification & target (kept for indexing and tracking)
    df_engineered["appid"] = df["appid"]
    df_engineered["name"] = df["name"]
    df_engineered["target"] = df["target"].astype(int)

    # Pricing & Economic Features
    price = df["price"].fillna(0.0).clip(lower=0.0)
    initial_price = df["initial_price"].fillna(price).clip(lower=0.0)
    df_engineered["price"] = price
    df_engineered["initial_price"] = initial_price
    df_engineered["is_free"] = (price == 0.0).astype(int)
    
    # Discount percentage: (initial - current) / initial
    denom = initial_price.replace(0.0, np.nan)
    df_engineered["discount_percentage"] = ((initial_price - price) / denom).fillna(0.0).clip(0.0, 1.0)
    df_engineered["has_discount"] = (df_engineered["discount_percentage"] > 0.0).astype(int)

    # Playtime & Engagement Ratios
    playtime_hours = (df["average_forever"].fillna(0) / 60.0).clip(lower=0.0)
    df_engineered["price_per_hour_playtime"] = (price / (playtime_hours + 1.0)).round(4)
    
    # Engagement Rate = Total Reviews / Estimated Owners
    owners_midpoint = df["owners_midpoint"].replace(0, 1000).clip(lower=1)
    df_engineered["engagement_rate"] = (df["total_reviews"] / owners_midpoint).round(6)
    df_engineered["ccu_to_owners_ratio"] = (df["ccu"] / owners_midpoint).round(6)

    # Log-transformed continuous metrics to normalize heavy right-tails
    df_engineered["log_playtime_forever"] = np.log1p(df["average_forever"].fillna(0).clip(lower=0))
    df_engineered["log_median_playtime"] = np.log1p(df["median_forever"].fillna(0).clip(lower=0))
    df_engineered["log_ccu"] = np.log1p(df["ccu"].fillna(0).clip(lower=0))
    df_engineered["log_owners"] = np.log1p(owners_midpoint)

    # Tag & Genre volume features
    df_engineered["tag_count"] = tags_series.apply(len)
    df_engineered["genre_count"] = genres_series.apply(len)

    # Top Entity Indicator (Top 30 developers / publishers)
    top_devs = set(df["developer"].value_counts().head(30).index)
    top_pubs = set(df["publisher"].value_counts().head(30).index)
    df_engineered["is_top_developer"] = df["developer"].isin(top_devs).astype(int)
    df_engineered["is_top_publisher"] = df["publisher"].isin(top_pubs).astype(int)

    # 4. Concatenate All Features
    df_all = pd.concat([df_engineered, df_genres, df_tags], axis=1)
    
    feature_cols = [c for c in df_all.columns if c not in ["appid", "name", "target"]]
    print(f"[+] Total engineered feature space: {len(feature_cols)} features across {len(df_all)} games.")
    
    return df_all, feature_cols


def apply_variance_threshold(df_all, feature_cols, threshold=0.01):
    """Drop features with variance below threshold (e.g. tags used by < 1% of games)."""
    print(f"[*] Applying VarianceThreshold ({threshold=})...")
    X = df_all[feature_cols].copy()
    
    vt = VarianceThreshold(threshold=threshold)
    vt.fit(X)
    
    kept_mask = vt.get_support()
    kept_features = [col for col, kept in zip(feature_cols, kept_mask) if kept]
    dropped_count = len(feature_cols) - len(kept_features)
    
    print(f"    Dropped {dropped_count} low-variance features.")
    print(f"    Kept {len(kept_features)} features with variance >= {threshold}.")
    
    return kept_features


def rank_and_select_features(df_all, candidate_features, top_k=60, fig_path="reports/figures/feature_importance_preliminary.png"):
    """Rank candidate features using Random Forest importance and select top K."""
    print(f"[*] Ranking features using Random Forest importance on {len(candidate_features)} candidates...")
    X = df_all[candidate_features]
    y = df_all["target"]

    # Use Random Forest with balanced class weights
    rf = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, class_weight="balanced", n_jobs=-1)
    rf.fit(X, y)

    importances = rf.feature_importances_
    feat_df = pd.DataFrame({
        "feature": candidate_features,
        "importance": importances
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    selected_features = feat_df.head(top_k)["feature"].tolist()
    print(f"[+] Selected top {len(selected_features)} features.")

    # Generate Feature Importance Visualization
    os.makedirs(os.path.dirname(os.path.abspath(fig_path)), exist_ok=True)
    top_25 = feat_df.head(25)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=top_25, x="importance", y="feature", palette="mako")
    plt.title("Top 25 Preliminary Feature Importances (Random Forest)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Gini Feature Importance", fontsize=11)
    plt.ylabel("Feature", fontsize=11)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[+] Saved preliminary feature importance figure: {fig_path}")

    return selected_features, feat_df


def save_datasets(df_all, selected_features, feat_importance_df,
                  all_csv="data/steam_features_all.csv",
                  selected_csv="data/steam_features_selected.csv",
                  meta_json="data/feature_metadata.json"):
    """Export complete and selected feature datasets."""
    os.makedirs(os.path.dirname(os.path.abspath(all_csv)), exist_ok=True)

    # Save full matrix
    df_all.to_csv(all_csv, index=False, encoding="utf-8")
    print(f"[+] Saved full feature matrix: {all_csv} ({df_all.shape[0]} rows x {df_all.shape[1]} cols)")

    # Save selected matrix
    cols_to_save = ["appid", "name", "target"] + selected_features
    df_selected = df_all[cols_to_save].copy()
    df_selected.to_csv(selected_csv, index=False, encoding="utf-8")
    print(f"[+] Saved selected feature matrix: {selected_csv} ({df_selected.shape[0]} rows x {df_selected.shape[1]} cols)")

    # Save metadata dictionary
    meta = {
        "total_records": len(df_all),
        "total_engineered_features": len([c for c in df_all.columns if c not in ["appid", "name", "target"]]),
        "selected_features_count": len(selected_features),
        "selected_features": selected_features,
        "feature_importances_top30": feat_importance_df.head(30).to_dict(orient="records")
    }
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[+] Saved feature metadata: {meta_json}")


def main():
    parser = argparse.ArgumentParser(description="Steam Games Feature Engineering & Selection")
    parser.add_argument("--input", type=str, default="data/steam_games_cleaned.csv", help="Input cleaned CSV path")
    parser.add_argument("--threshold", type=float, default=0.01, help="VarianceThreshold (default: 0.01)")
    parser.add_argument("--top-k", type=int, default=60, help="Number of features to select (default: 60)")
    parser.add_argument("--out-all", type=str, default="data/steam_features_all.csv", help="Full feature matrix path")
    parser.add_argument("--out-selected", type=str, default="data/steam_features_selected.csv", help="Selected feature matrix path")
    parser.add_argument("--out-meta", type=str, default="data/feature_metadata.json", help="Feature metadata path")
    parser.add_argument("--fig", type=str, default="reports/figures/feature_importance_preliminary.png", help="Figure output path")

    args = parser.parse_args()

    df_all, feature_cols = engineer_features(args.input)
    candidate_features = apply_variance_threshold(df_all, feature_cols, threshold=args.threshold)
    selected_features, feat_importance_df = rank_and_select_features(df_all, candidate_features, top_k=args.top_k, fig_path=args.fig)
    
    save_datasets(
        df_all,
        selected_features,
        feat_importance_df,
        all_csv=args.out_all,
        selected_csv=args.out_selected,
        meta_json=args.out_meta
    )

    print("\n=== Feature Engineering & Selection Complete ===")
    print(f"Input Records: {len(df_all)}")
    print(f"Total Features Generated: {len(feature_cols)}")
    print(f"Features after Variance Threshold ({args.threshold}): {len(candidate_features)}")
    print(f"Selected Top Features: {len(selected_features)}")
    print("Top 10 Ranked Features:")
    for rank, r in enumerate(feat_importance_df.head(10).itertuples(), 1):
        print(f"  {rank}. {r.feature:<30} (importance: {r.importance:.4f})")
    print("================================================\n")


if __name__ == "__main__":
    main()
