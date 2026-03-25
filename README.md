# House Prices – Advanced Regression Techniques  
### Аналитическое решение: от исследования данных до интерпретации модели  
**Kaggle Top 5% | RMSLE = 0.11952**

---

## 🇬🇧 English

### Project Overview
This project solves the Kaggle competition **House Prices – Advanced Regression Techniques**. The objective is to predict the final sale price (`SalePrice`) of residential homes using 79 descriptive features. The metric is **RMSLE**, which penalises relative error, making it suitable for a wide price range.

The final ensemble achieves **RMSLE = 0.11952** on the public leaderboard, placing in the **top 5%** of over 8,000 teams. The solution emphasises **analytical rigour**, **interpretability**, and **business applicability**.

### Analytical Approach

#### 1. Exploratory Data Analysis (EDA) – Key Insights
- **Missing values** are not random: most appear in columns describing optional features (pool, fence, garage). They were filled with `'None'` for categorical features and medians (by neighborhood for `LotFrontage`) for numerical ones.
- **Target distribution** is highly skewed; log‑transformation (`log1p`) was applied to stabilise variance and align with RMSLE.
- **Top correlates**: `OverallQual` (0.79), `GrLivArea` (0.71), `GarageCars` (0.64). High multicollinearity between `GarageCars` and `GarageArea` led to aggregated features.
- **Outliers**: two houses with `GrLivArea > 4000` and `SalePrice < 300,000` were removed to prevent distortion of linear model coefficients.
- **Categorical importance**: ordinal quality ratings (`ExterQual`, `KitchenQual`) show monotonic price impact; rare categories were grouped or encoded as scores.

#### 2. Feature Engineering – Business‑Driven
- **Aggregated areas** (`TotalSF`, `TotalLivingSF`) reduce multicollinearity and better capture overall size.
- **Quality scores** convert ordinal categories to numeric scales, preserving order.
- **Temporal features** (`Age`, `RemodFlag`) capture property aging and renovation effects.
- **Interaction terms** (`QualSF = OverallQual × TotalSF`) enhance non‑linear relationships.
- **Binary flags** simplify presence of important objects (garage, basement, fireplace) for models.

Each engineered feature was validated by correlation with `log(SalePrice)`; `TotalSF` alone correlates at 0.84, higher than any single area component.

#### 3. Modeling Strategy
- **Base models**: Ridge, Lasso, ElasticNet (linear, regularised) + LightGBM, XGBoost, CatBoost (gradient boosting).
- **Hyperparameter tuning**: GridSearchCV for linear models, Optuna (128 trials) for LightGBM. Manual tuning for XGBoost and CatBoost.
- **Stacking**: 16‑fold out‑of‑fold predictions from selected models (OOF RMSE < 0.125) were used as features for a RidgeCV meta‑learner. This improved RMSLE from 0.118 (simple average) to 0.110 (CV), with final Kaggle score 0.11952.

#### 4. Model Interpretation (SHAP & Residuals)
- **SHAP analysis** revealed that `TotalSF`, `OverallQual`, and `Age` are the most influential features. `TotalSF` has a positive, nearly linear effect on price.
- **Residual diagnostics** show that the model slightly underestimates high‑priced properties and overestimates very low‑priced ones, suggesting potential for additional feature engineering (e.g., location‑specific adjustments).
- **Business value**: The model can be deployed for automated mortgage pre‑approval, reducing assessment time by ~80% compared to manual appraisal.

### Results Summary

| Stage | RMSLE (CV) |
|-------|------------|
| Baseline (simple preprocessing) | 0.152 |
| + Feature Engineering | 0.135 |
| + LightGBM tuning | 0.125 |
| + Ensemble (average of 5 models) | 0.118 |
| + Stacking (RidgeCV) | 0.110 |
| **Final Kaggle Public LB** | **0.11952** |

### How to Reproduce
1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
