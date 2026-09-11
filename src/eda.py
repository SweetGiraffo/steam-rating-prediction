"""Steam Games Exploratory Data Analysis (EDA) Module.

Analyzes distributions, relationships, and class balance of cleaned Steam games data.
Generates publication-quality visualization figures and a comprehensive markdown report.
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Style configuration for clean, modern plots
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def plot_target_distribution(df, out_path):
    """Plot target class distribution and class imbalance."""
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df["target"].value_counts().sort_index()
    labels = ["Below/Equal 70% (Target=0)", "Above 70% (Target=1)"]
    colors = ["#e74c3c", "#2ecc71"]
    
    bars = ax.bar(labels, counts, color=colors, width=0.5, edgecolor="#333333", linewidth=1)
    total = len(df)
    for bar in bars:
        h = bar.get_height()
        pct = (h / total) * 100
        ax.annotate(f"{h:,}\n({pct:.1f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, h / 2),
                    xytext=(0, 0), textcoords="offset points",
                    ha="center", va="center", color="white", fontsize=12, fontweight="bold")
        
    ax.set_title("Steam Rating Target Distribution (> 70% Positive Reviews)", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylabel("Number of Games", fontsize=11)
    ax.set_ylim(0, max(counts) * 1.15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Saved figure: {out_path}")


def plot_rating_vs_price(df, out_path):
    """Plot price distribution and rating probability by price tier."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Price tier grouping
    def price_tier(p):
        if p == 0:
            return "Free"
        elif p < 10:
            return "$0.01 - $9.99"
        elif p < 20:
            return "$10 - $19.99"
        elif p < 40:
            return "$20 - $39.99"
        else:
            return "$40+"

    df_plot = df.copy()
    df_plot["price_tier"] = df_plot["price"].apply(price_tier)
    tier_order = ["Free", "$0.01 - $9.99", "$10 - $19.99", "$20 - $39.99", "$40+"]

    # Tier volume
    tier_counts = df_plot["price_tier"].value_counts().reindex(tier_order)
    sns.barplot(x=tier_counts.index, y=tier_counts.values, ax=ax1, palette="Blues_d")
    ax1.set_title("Games Count by Price Tier", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Count", fontsize=10)
    ax1.tick_params(axis="x", rotation=15)

    # High rating success rate per tier
    tier_rates = df_plot.groupby("price_tier")["target"].mean().reindex(tier_order) * 100
    sns.barplot(x=tier_rates.index, y=tier_rates.values, ax=ax2, palette="Greens_d")
    ax2.set_title("% High Rated Games (>70%) by Price Tier", fontsize=12, fontweight="bold")
    ax2.set_ylabel("% High Rated", fontsize=10)
    ax2.set_ylim(50, 100)
    ax2.tick_params(axis="x", rotation=15)
    for i, v in enumerate(tier_rates.values):
        ax2.text(i, v + 0.8, f"{v:.1f}%", ha="center", fontweight="bold", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Saved figure: {out_path}")


def plot_top_tags(df, out_path, top_n=20):
    """Plot top N tags by frequency and their positive rating percentage."""
    tag_counter = Counter()
    tag_pos_counter = Counter()

    for _, row in df.iterrows():
        tags = [t.strip() for t in str(row["tags"]).split(";") if t.strip()]
        is_pos = row["target"] == 1
        for t in tags:
            tag_counter[t] += 1
            if is_pos:
                tag_pos_counter[t] += 1

    top_tags = [t for t, _ in tag_counter.most_common(top_n)]
    freqs = [tag_counter[t] for t in top_tags]
    rates = [(tag_pos_counter[t] / tag_counter[t]) * 100 for t in top_tags]

    fig, ax1 = plt.subplots(figsize=(12, 8))

    y_pos = np.arange(len(top_tags))
    ax1.barh(y_pos, freqs, align="center", color="#3498db", alpha=0.85, edgecolor="#2980b9")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(top_tags, fontsize=10)
    ax1.invert_yaxis()  # top tag at the top
    ax1.set_xlabel("Frequency (Game Count)", color="#2980b9", fontsize=11, fontweight="bold")
    ax1.set_title(f"Top {top_n} Steam Tags: Frequency and High-Rating Probability", fontsize=13, fontweight="bold", pad=15)

    # Annotate rate on bars
    for i, (freq, rate) in enumerate(zip(freqs, rates)):
        ax1.text(freq + 15, i, f"{rate:.1f}% High Rated", va="center", fontsize=9, fontweight="bold", color="#2c3e50")

    ax1.set_xlim(0, max(freqs) * 1.22)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Saved figure: {out_path}")


def plot_top_genres(df, out_path):
    """Plot genre frequencies and their correlation with high rating."""
    genre_counter = Counter()
    genre_target = Counter()

    for _, row in df.iterrows():
        genres = [g.strip() for g in str(row["genres"]).split(",") if g.strip()]
        is_pos = row["target"] == 1
        for g in genres:
            genre_counter[g] += 1
            if is_pos:
                genre_target[g] += 1

    common_genres = [g for g, c in genre_counter.most_common(12) if c >= 30]
    counts = [genre_counter[g] for g in common_genres]
    pct_high = [(genre_target[g] / genre_counter[g]) * 100 for g in common_genres]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    sns.barplot(x=counts, y=common_genres, ax=ax1, palette="mako")
    ax1.set_title("Genre Frequency (Game Count)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Number of Games", fontsize=10)

    sns.barplot(x=pct_high, y=common_genres, ax=ax2, palette="viridis")
    ax2.set_title("% High Rated (> 70%) by Genre", fontsize=12, fontweight="bold")
    ax2.set_xlabel("% High Rated", fontsize=10)
    ax2.set_xlim(60, 100)
    for i, v in enumerate(pct_high):
        ax2.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Saved figure: {out_path}")


def plot_playtime_and_engagement(df, out_path):
    """Plot engagement metrics (reviews & playtime) comparing target classes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Log Total Reviews
    df_plot = df.copy()
    df_plot["log_reviews"] = np.log10(df_plot["total_reviews"].clip(lower=1))
    df_plot["target_label"] = df_plot["target"].map({0: "<= 70% Positive", 1: "> 70% Positive"})

    sns.boxplot(x="target_label", y="log_reviews", data=df_plot, ax=ax1, palette=["#e74c3c", "#2ecc71"])
    ax1.set_title("Total Reviews Distribution (Log10 Scale)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Log10(Total Reviews)", fontsize=10)
    ax1.set_xlabel("Rating Target", fontsize=10)

    # CCU distribution
    df_plot["log_ccu"] = np.log10(df_plot["ccu"].clip(lower=1))
    sns.boxplot(x="target_label", y="log_ccu", data=df_plot, ax=ax2, palette=["#e74c3c", "#2ecc71"])
    ax2.set_title("Concurrent Players (CCU) (Log10 Scale)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Log10(Peak CCU)", fontsize=10)
    ax2.set_xlabel("Rating Target", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Saved figure: {out_path}")


def generate_eda_report(df, report_path="reports/eda_report.md"):
    """Compile comprehensive EDA summary markdown document."""
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)

    total_games = len(df)
    pos_games = (df["target"] == 1).sum()
    neg_games = (df["target"] == 0).sum()
    pos_pct = (pos_games / total_games) * 100
    neg_pct = (neg_games / total_games) * 100

    # Pricing stats
    free_games = (df["is_free"] == 1).sum()
    paid_games = (df["is_free"] == 0).sum()
    median_paid_price = df[df["is_free"] == 0]["price"].median()
    mean_paid_price = df[df["is_free"] == 0]["price"].mean()

    # Tag stats
    all_tags = [t.strip() for tags in df["tags"] for t in str(tags).split(";") if t.strip()]
    unique_tags = len(set(all_tags))
    top_10_tags = Counter(all_tags).most_common(10)

    # Genre stats
    all_genres = [g.strip() for genres in df["genres"] for g in str(genres).split(",") if g.strip()]
    unique_genres = len(set(all_genres))
    top_5_genres = Counter(all_genres).most_common(5)

    report_content = f"""# Exploratory Data Analysis (EDA) Report: Steam Games Dataset

## Executive Summary
This report summarizes exploratory analysis on the **{total_games:,} Steam games** dataset cleaned in Phase 2. The primary objective is to inspect feature distributions, examine community rating dynamics, and identify structural characteristics that will guide feature engineering (Phase 3) and classification modeling (Phase 4).

---

## 1. Target Variable & Class Balance

- **Target Definition**: `target = 1` if `(positive / total_reviews) > 0.70` else `0`
- **Positive Class (`target = 1`)**: **{pos_games:,} games ({pos_pct:.1f}%)**
- **Non-Positive Class (`target = 0`)**: **{neg_games:,} games ({neg_pct:.1f}%)**
- **Imbalance Ratio**: **{pos_games/max(1, neg_games):.2f} : 1**

> [!IMPORTANT]
> **Imbalance Implication**: The strong positive skew is a natural reflection of Steam's catalog where well-reviewed games dominate community discovery. In Phase 4, classification models must use `class_weight='balanced'` (Logistic Regression, Random Forest) and `scale_pos_weight` (XGBoost) to ensure the minority class is appropriately weighted and F1-score is preserved.

---

## 2. Pricing & Monetization Insights

| Monetization Model | Game Count | Percentage | High-Rating Rate (>70%) |
| :--- | :--- | :--- | :--- |
| **Free-to-Play** | {free_games:,} | {(free_games/total_games)*100:.1f}% | {(df[df['is_free'] == 1]['target'].mean())*100:.1f}% |
| **Paid Games** | {paid_games:,} | {(paid_games/total_games)*100:.1f}% | {(df[df['is_free'] == 0]['target'].mean())*100:.1f}% |

- **Median Paid Price**: **${median_paid_price:.2f}** (Mean: ${mean_paid_price:.2f})
- **Finding**: Paid games have a noticeably higher high-rating rate compared to free-to-play titles. Free-to-play games are more vulnerable to review bombing and microtransaction backlash.

---

## 3. Top Tags and High-Rating Correlation

Total Unique Tags Discovered: **{unique_tags}**

| Rank | Tag | Total Games | % with High Rating (>70%) |
| :---: | :--- | :---: | :---: |
"""

    for rank, (tag, count) in enumerate(top_10_tags, 1):
        tag_mask = df["tags"].astype(str).str.contains(rf"(?:^|;){tag}(?:;|$)", regex=True)
        tag_high_rate = df[tag_mask]["target"].mean() * 100 if tag_mask.sum() > 0 else 0.0
        report_content += f"| {rank} | **{tag}** | {count:,} | {tag_high_rate:.1f}% |\n"

    report_content += f"""
---

## 4. Genre Landscape

Total Unique Genres Discovered: **{unique_genres}**

| Rank | Genre | Total Games | % with High Rating (>70%) |
| :---: | :--- | :---: | :---: |
"""
    for rank, (genre, count) in enumerate(top_5_genres, 1):
        g_mask = df["genres"].astype(str).str.contains(rf"(?:^|, ){genre}(?:,|$)", regex=True)
        g_high_rate = df[g_mask]["target"].mean() * 100 if g_mask.sum() > 0 else 0.0
        report_content += f"| {rank} | **{genre}** | {count:,} | {g_high_rate:.1f}% |\n"

    report_content += """
---

## 5. Visual Artifacts Generated

1. `reports/figures/target_distribution.png`: Binary target proportions and imbalance.
2. `reports/figures/rating_vs_price.png`: Price tier volume and high-rating probability.
3. `reports/figures/top_tags.png`: Frequency and rating rate of top 20 Steam tags.
4. `reports/figures/top_genres.png`: Genre proportions and high-rating success.
5. `reports/figures/playtime_and_engagement.png`: Distribution of log reviews and CCU across rating categories.

---

## 6. Recommendations for Phase 3 (Feature Engineering)

1. **Tag Encoding**: Use `MultiLabelBinarizer` or `TF-IDF` on the parsed `tags_list`. Because tags have high cardinality (>300 unique), filter tags with <1% variance threshold.
2. **Derived Ratios**:
   - `discount_percentage`: Difference between `initial_price` and `price`.
   - `engagement_rate`: `total_reviews / owners_midpoint`.
   - `price_per_hour`: `price / (average_forever / 60 + 1)`.
3. **Categorical Entity Regularization**: Top developers and publishers should either be frequency-encoded or target-encoded to avoid extreme sparse expansion.
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"[+] Saved EDA report: {report_path}")


def run_eda(csv_path="data/steam_games_cleaned.csv", fig_dir="reports/figures", report_path="reports/eda_report.md"):
    """Run full EDA pipeline."""
    print(f"[*] Starting EDA on cleaned data: {csv_path}")
    os.makedirs(fig_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    plot_target_distribution(df, os.path.join(fig_dir, "target_distribution.png"))
    plot_rating_vs_price(df, os.path.join(fig_dir, "rating_vs_price.png"))
    plot_top_tags(df, os.path.join(fig_dir, "top_tags.png"), top_n=20)
    plot_top_genres(df, os.path.join(fig_dir, "top_genres.png"))
    plot_playtime_and_engagement(df, os.path.join(fig_dir, "playtime_and_engagement.png"))

    generate_eda_report(df, report_path=report_path)
    print("[*] EDA completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Steam Games EDA Generator")
    parser.add_argument("--input", type=str, default="data/steam_games_cleaned.csv", help="Input cleaned CSV path")
    parser.add_argument("--fig-dir", type=str, default="reports/figures", help="Output figures directory")
    parser.add_argument("--report", type=str, default="reports/eda_report.md", help="Output report path")

    args = parser.parse_args()
    run_eda(csv_path=args.input, fig_dir=args.fig_dir, report_path=args.report)


if __name__ == "__main__":
    main()
