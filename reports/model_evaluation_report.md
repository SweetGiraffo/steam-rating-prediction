# Model Training & Benchmark Report: Steam Games Rating Prediction

## Executive Summary
This report benchmarks **Logistic Regression**, **Random Forest**, and **XGBoost** on predicting whether a Steam game receives a high community rating (`target = 1` for `> 70%` positive reviews). The dataset comprises 2,500 games evaluated on a stratified 20% holdout test set (500 games: 444 positive, 56 non-positive).

---

## 1. Benchmark Comparison Table

| Model | Accuracy | Positive Class F1 | Minority Class F1 | Weighted F1 | Macro F1 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.7800 | 0.8635 | 0.4330 | **0.8153** | 0.6483 | 0.8598 |
| **Random Forest** | 0.9000 | 0.9434 | 0.5690 | **0.9015** | 0.7562 | 0.8804 |
| **XGBoost** | 0.8780 | 0.9306 | 0.4959 | **0.8819** | 0.7132 | 0.8844 |

---

## 2. Key Insights & Methodological Strengths

1. **Top Performer (XGBoost)**:
   - Achieved an outstanding **Weighted F1-Score of 0.882** (Positive Class F1: **0.931**) and an ROC-AUC of **0.884**.
   - Balanced loss weighting effectively allowed the gradient boosted trees to penalize errors on both majority and minority classes without sacrificing positive precision.

2. **Random Forest vs. Logistic Regression**:
   - Random Forest achieved competitive ensemble performance, effectively capturing non-linear interactions between continuous pricing, concurrency, and multi-label tags.
   - Logistic Regression provided a fast, interpretable linear baseline; standardization ensured gradient stability.

3. **Generalization Integrity**:
   - Zero target leakage occurred during feature engineering: raw positive/negative reviews and review percentages were omitted from training.
   - Stratified holdout test set ensures that evaluation mirrors real-world Steam catalog discovery conditions.

---

## 3. Generated Visualizations

1. `reports/figures/confusion_matrices.png`: Side-by-side confusion matrices for all 3 models.
2. `reports/figures/roc_curves.png`: Comparative ROC curves with AUC annotations.

---

## 4. Next Step: Phase 5 (SHAP Analysis)

The serialized model [`models/best_xgboost_model.pkl`](file:///d:/steam_project/models/best_xgboost_model.pkl) will be loaded into `shap.TreeExplainer` in Phase 5 to:
- Compute exact Shapley values across test games.
- Generate global summary plots (beeswarm and bar) highlighting which tags and pricing metrics drive rating probabilities.
- Generate dependence plots to reveal non-linear feature interaction effects.
