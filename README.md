# Steam Game Rating Prediction & SHAP Explainability Pipeline

An end-to-end machine learning system that scrapes 2,500+ Steam titles, engineers multi-label tags and economic playtime metrics, benchmarks classification algorithms to predict high community ratings (>70% positive), and explains feature contributions via SHAP (SHapley Additive exPlanations).

Live Showcase Dashboard: https://sweetgiraffo.github.io/steam-rating-prediction/

---

## Performance Summary

Evaluated on a stratified 20% holdout test set (500 games: 444 positive, 56 non-positive) with zero target leakage:

| Model | Accuracy | Positive Class F1 | Minority Class F1 | Weighted F1 | Macro F1 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression (Balanced, Scaled) | 78.0% | 0.8635 | 0.4330 | 0.8153 | 0.6483 | 0.8598 |
| Random Forest (250 Trees, Balanced) | 90.0% | 0.9434 | 0.5686 | 0.9015 | 0.7560 | 0.8804 |
| XGBoost Classifier (Balanced Weights) | 87.8% | 0.9306 | 0.4960 | 0.8819 | 0.7132 | 0.8844 |

- Best Discrimination: XGBoost (0.8844 ROC-AUC)
- Best Overall Accuracy: Random Forest (90.0% Accuracy, 0.9015 Weighted F1)

---

## Pipeline Architecture

1. Data Acquisition: Automated extraction of 2,500+ games from the SteamSpy API with rate-limiting and continuous atomic checkpointing.
2. Data Cleaning & EDA: Binary target formulation (`target = 1` if `positive / total_reviews > 0.70` else `0`), review noise filtering (`total_reviews >= 50`), and class imbalance profiling (88.8% positive vs. 11.2% non-positive).
3. Feature Engineering: MultiLabel binarization of 434 community tags and 24 genres, derived economic/playtime ratios (`price_per_hour_playtime`, `engagement_rate`, `discount_percentage`), variance thresholding (`threshold=0.01` to drop tags present in <1% of games), and preliminary Gini importance ranking (top 60 selected features). Raw review counts excluded to prevent data leakage.
4. Modeling & Benchmark: Stratified 80/20 train/test split. Feature scaling for Logistic Regression. Balanced class weights across all models to handle the 7.9:1 class ratio.
5. SHAP Explainability: TreeExplainer analysis of the trained XGBoost model across global summary beeswarm plots, feature importance rankings, and interaction dependence plots.

---

## Key Explainability Findings

- Indie Advantage: `genre_Indie` and `tag_Indie` provide an immediate positive push to rating log-odds (mean |SHAP| = 0.263). Player sentiment demonstrates higher tolerance and active goodwill toward indie projects.
- Pricing Dynamics: `price`, `initial_price`, and `price_per_hour_playtime` rank among the top 4 global decision drivers. Free-to-play titles suffer from negative sentiment drag due to monetization models and smurfing, whereas titles priced between $10 and $30 with substantial playtime per dollar achieve strong positive contributions.
- Multiplayer Review Drag: `tag_Multiplayer` and `tag_PvP` carry negative rating bias unless counterbalanced by high concurrent player counts (`log_ccu`), reflecting community scrutiny over matchmaking, server latency, and cheat prevention.

---

## Repository Structure

```
steam_project/
├── index.html                        # Minimalist black showcase dashboard
├── requirements.txt                  # Pinned dependencies
├── README.md                         # Project documentation
├── GEMINI.md                         # Persistent style and development guidelines
├── data/
│   ├── steam_games_raw.csv           # 2,500 raw scraped games
│   ├── steam_games_cleaned.csv       # Cleaned dataset (0 missing values)
│   ├── steam_features_all.csv        # Full 474 engineered feature matrix
│   ├── steam_features_selected.csv   # Top 60 curated features + target
│   └── feature_metadata.json         # Feature importance dictionary
├── models/
│   └── best_xgboost_model.pkl        # Serialized best XGBoost model
├── reports/
│   ├── eda_report.md                 # Exploratory data analysis report
│   ├── model_evaluation_report.md    # Model benchmark report
│   ├── shap_analysis_report.md       # SHAP explainability report
│   └── figures/                      # 13 high-resolution analytical plots
│       ├── target_distribution.png
│       ├── rating_vs_price.png
│       ├── top_tags.png
│       ├── top_genres.png
│       ├── playtime_and_engagement.png
│       ├── feature_importance_preliminary.png
│       ├── confusion_matrices.png
│       ├── roc_curves.png
│       ├── shap_summary_beeswarm.png
│       ├── shap_feature_importance.png
│       ├── shap_dependence_price.png
│       ├── shap_dependence_indie.png
│       └── shap_dependence_multiplayer.png
└── src/
    ├── __init__.py
    ├── scraper.py                    # Phase 1: Data Acquisition
    ├── cleaner.py                    # Phase 2: Data Cleaning
    ├── eda.py                        # Phase 2: Exploratory Analysis
    ├── features.py                   # Phase 3: Feature Engineering & Selection
    ├── models.py                     # Phase 4: Modeling & Benchmark
    └── explain.py                    # Phase 5: SHAP Analysis
```

---

## Quickstart

### Setup
```bash
git clone https://github.com/SweetGiraffo/steam-rating-prediction.git
cd steam-rating-prediction
pip install -r requirements.txt
```

### Execution
```powershell
# Phase 1: Scrape dataset
python -m src.scraper --target 2500 --workers 4

# Phase 2: Clean data and generate EDA report
python -m src.cleaner
python -m src.eda

# Phase 3: Engineer features and select top 60 predictors
python -m src.features --threshold 0.01 --top-k 60

# Phase 4: Train and benchmark models
python -m src.models

# Phase 5: Run SHAP TreeExplainer
python -m src.explain

# Launch local dashboard
python -m http.server 8000
```

---

## License
MIT License. See LICENSE for details.
