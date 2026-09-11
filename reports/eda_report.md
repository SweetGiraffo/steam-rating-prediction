# Exploratory Data Analysis (EDA) Report: Steam Games Dataset

## Executive Summary
This report summarizes exploratory analysis on the **2,500 Steam games** dataset cleaned in Phase 2. The primary objective is to inspect feature distributions, examine community rating dynamics, and identify structural characteristics that will guide feature engineering (Phase 3) and classification modeling (Phase 4).

---

## 1. Target Variable & Class Balance

- **Target Definition**: `target = 1` if `(positive / total_reviews) > 0.70` else `0`
- **Positive Class (`target = 1`)**: **2,219 games (88.8%)**
- **Non-Positive Class (`target = 0`)**: **281 games (11.2%)**
- **Imbalance Ratio**: **7.90 : 1**

> [!IMPORTANT]
> **Imbalance Implication**: The strong positive skew is a natural reflection of Steam's catalog where well-reviewed games dominate community discovery. In Phase 4, classification models must use `class_weight='balanced'` (Logistic Regression, Random Forest) and `scale_pos_weight` (XGBoost) to ensure the minority class is appropriately weighted and F1-score is preserved.

---

## 2. Pricing & Monetization Insights

| Monetization Model | Game Count | Percentage | High-Rating Rate (>70%) |
| :--- | :--- | :--- | :--- |
| **Free-to-Play** | 467 | 18.7% | 74.7% |
| **Paid Games** | 2,033 | 81.3% | 92.0% |

- **Median Paid Price**: **$19.99** (Mean: $21.24)
- **Finding**: Paid games have a noticeably higher high-rating rate compared to free-to-play titles. Free-to-play games are more vulnerable to review bombing and microtransaction backlash.

---

## 3. Top Tags and High-Rating Correlation

Total Unique Tags Discovered: **434**

| Rank | Tag | Total Games | % with High Rating (>70%) |
| :---: | :--- | :---: | :---: |
| 1 | **Singleplayer** | 2,065 | 89.8% |
| 2 | **Action** | 1,697 | 87.2% |
| 3 | **Adventure** | 1,450 | 89.2% |
| 4 | **Multiplayer** | 1,440 | 83.2% |
| 5 | **Atmospheric** | 1,041 | 91.7% |
| 6 | **Indie** | 986 | 94.0% |
| 7 | **Co-op** | 927 | 82.8% |
| 8 | **Open World** | 883 | 83.0% |
| 9 | **Story Rich** | 850 | 93.4% |
| 10 | **RPG** | 829 | 86.2% |

---

## 4. Genre Landscape

Total Unique Genres Discovered: **25**

| Rank | Genre | Total Games | % with High Rating (>70%) |
| :---: | :--- | :---: | :---: |
| 1 | **Action** | 1,462 | 86.4% |
| 2 | **Indie** | 1,154 | 93.7% |
| 3 | **Adventure** | 1,046 | 90.0% |
| 4 | **RPG** | 680 | 85.7% |
| 5 | **Simulation** | 598 | 87.8% |

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
