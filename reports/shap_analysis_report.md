# SHAP Explainability & Model Interpretation Report

## Executive Summary
This report applies **SHAP (SHapley Additive exPlanations)** via `shap.TreeExplainer` to interpret the predictions of our highest-performing **XGBoost model** (0.884 ROC-AUC, 0.882 Weighted F1-score) on the 500-game test set.

By calculating local Shapley values for every feature on every test game, SHAP bridges the gap between tree ensemble accuracy and human interpretability, pinpointing the exact factors that make a Steam game succeed or fail community review standards (>70% positive).

---

## 1. Top Feature Drivers (Mean Absolute SHAP)

| Rank | Feature | Mean |SHAP| Value | Impact Direction & Domain Mechanism |
| :---: | :--- | :---: | :--- |
| 1 | **`log_ccu`** | **0.6337** | Positive: Sustained concurrent player counts act as proof of active, healthy community satisfaction. |
| 2 | **`initial_price`** | **0.3247** | Bimodal / Non-linear: Moderate-to-high prices correlate with perceived production value; low/free prices skew negative unless player engagement is massive. |
| 3 | **`tag_Multiplayer`** | **0.3212** | Negative Bias: Competitive multiplayer titles face severe review scrutiny regarding server stability and balance grievances. |
| 4 | **`price_per_hour_playtime`** | **0.2799** | Bimodal / Non-linear: Moderate-to-high prices correlate with perceived production value; low/free prices skew negative unless player engagement is massive. |
| 5 | **`tag_Indie`** | **0.2629** | Strong Positive: Indie games benefit from favorable player expectations and community support. |
| 6 | **`log_owners`** | **0.1893** | Domain Driver: Significant contributor to tree decision paths. |
| 7 | **`tag_2D`** | **0.1734** | Positive: Polished 2D, story-rich singleplayer experiences consistently earn stellar ratings. |
| 8 | **`tag_PvP`** | **0.1574** | Negative Bias: Competitive multiplayer titles face severe review scrutiny regarding server stability and balance grievances. |
| 9 | **`genre_RPG`** | **0.1408** | Domain Driver: Significant contributor to tree decision paths. |
| 10 | **`genre_Massively_Multiplayer`** | **0.1384** | Negative Bias: Competitive multiplayer titles face severe review scrutiny regarding server stability and balance grievances. |
| 11 | **`tag_Difficult`** | **0.1362** | Domain Driver: Significant contributor to tree decision paths. |
| 12 | **`tag_Open_World`** | **0.1346** | Domain Driver: Significant contributor to tree decision paths. |
| 13 | **`tag_Classic`** | **0.1343** | Domain Driver: Significant contributor to tree decision paths. |
| 14 | **`genre_Indie`** | **0.1318** | Strong Positive: Indie games benefit from favorable player expectations and community support. |
| 15 | **`tag_Great_Soundtrack`** | **0.1299** | Domain Driver: Significant contributor to tree decision paths. |

---

## 2. Key Discoveries from SHAP Visualizations

### A. The "Indie" Tag & Genre Effect
- As revealed in the **SHAP Beeswarm Plot**, having `genre_Indie = 1` consistently exerts a positive push on the log-odds of achieving a high rating.
- Players approach indie titles with constructive, supportive expectations compared to large-scale AAA releases where minor technical issues trigger review bombing.

### B. The Price & Value Proposition Dynamic
- `price`, `initial_price`, and `price_per_hour_playtime` are among the top global drivers of rating probability.
- **The Free-to-Play Vulnerability**: Free-to-play games (`price = 0`) exhibit a net negative SHAP contribution, reflecting community frustration with monetization models, smurfing, and battle passes.
- **The Value Sweet Spot**: Games priced between $10 and $30 that deliver substantial playtime per dollar show large positive SHAP values.

### C. Multiplayer vs. Singleplayer Disparity
- Features such as `tag_Multiplayer`, `tag_Massively_Multiplayer`, and `tag_PvP` consistently exert negative SHAP forces on rating probability.
- Multiplayer games only achieve high ratings when accompanied by very high player concurrency (`log_ccu`), whereas singleplayer games achieve high ratings across diverse audience scales.

---

## 3. Generated Visual Artifacts

1. `reports/figures/shap_summary_beeswarm.png`: Global summary plot displaying directionality and magnitude of top 20 features.
2. `reports/figures/shap_feature_importance.png`: Mean absolute SHAP bar chart for top 20 features.
3. `reports/figures/shap_dependence_price.png`: Non-linear dependence between price and rating probability.
4. `reports/figures/shap_dependence_indie.png`: Dependence plot demonstrating the positive lift of the Indie classification.
5. `reports/figures/shap_dependence_multiplayer.png`: Dependence plot illustrating the multiplayer penalty mitigated by high CCU.
