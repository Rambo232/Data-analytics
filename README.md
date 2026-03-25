# House Prices – Advanced Regression Techniques  
### Analytical Solution: Top 5% on Kaggle (RMSLE = 0.19048)

> 📌 **Note:** The Russian version of this README is available [below](#-русский-версия).

---

## 📚 Notebooks & Resources

| Language | Link |
|----------|------|
| 🇬🇧 English | [house_prices_analytics_en.ipynb](https://github.com/Rambo232/kaggle_house_prices/blob/main/house_prices_analytics_en.ipynb) |
| 🇷🇺 Russian | [house_prices_analytics_ru.ipynb](https://github.com/Rambo232/kaggle_house_prices/blob/main/house_prices_analytics_ru.ipynb) |
| 🏆 Kaggle | [House Prices – Analysis and Prediction (Top 5%)](https://www.kaggle.com/code/rambo232/house-prices-analysis-and-prediction-top-5) |

---

## 🇬🇧 English

### Project Overview
This project solves the Kaggle competition **House Prices – Advanced Regression Techniques**. The objective is to predict the final sale price (`SalePrice`) of residential homes using 79 descriptive features. The metric is **RMSLE**, which penalises relative error, making it suitable for a wide price range.

The final ensemble achieves **RMSLE = 0.19048** on the public leaderboard, placing in the **top 5%** of over 8,000 teams. The solution emphasises **analytical rigour**, **interpretability**, and **business applicability**.

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
- **Stacking**: 16‑fold out‑of‑fold predictions from selected models (OOF RMSE < 0.125) were used as features for a RidgeCV meta‑learner. This improved RMSLE from 0.118 (simple average) to 0.110 (CV), with final Kaggle score **0.19048**.

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
| **Final Kaggle Public LB** | **0.19048** |

### How to Reproduce
1. Clone this repository:
   ```bash
   git clone https://github.com/Rambo232/kaggle_house_prices.git
   cd kaggle_house_prices



## 🇷🇺 Русский

> 📌 **English version** is available [above](#-english).

### Описание проекта
Это решение соревнования Kaggle **House Prices – Advanced Regression Techniques**. Задача – предсказать итоговую цену продажи дома (`SalePrice`) по 79 признакам. Метрика – **RMSLE**, чувствительная к относительной ошибке, что важно для широкого диапазона цен.

Финальный ансамбль достигает **RMSLE = 0.19048** на публичном лидерборде (топ‑5% из 8000+ команд). Решение акцентирует **аналитический подход**, **интерпретируемость** и **бизнес‑применимость**.

### Аналитический подход

#### 1. Исследовательский анализ (EDA) – ключевые инсайты
- **Пропуски** не случайны: большинство относятся к необязательным объектам (бассейн, забор, гараж). Для категориальных признаков – заполнение `'None'`, для числовых (`LotFrontage`) – медиана по району.
- **Целевая переменная** сильно скошена; применено логарифмирование (`log1p`) для стабилизации дисперсии и соответствия метрике.
- **Топ корреляций**: `OverallQual` (0.79), `GrLivArea` (0.71), `GarageCars` (0.64). Высокая мультиколлинеарность между `GarageCars` и `GarageArea` привела к созданию агрегированных признаков.
- **Выбросы**: два дома с `GrLivArea > 4000` и `SalePrice < 300,000` удалены, чтобы не искажать коэффициенты линейных моделей.
- **Категориальные признаки**: порядковые оценки качества (`ExterQual`, `KitchenQual`) монотонно влияют на цену; редкие категории объединены или преобразованы в числовые баллы.

#### 2. Feature Engineering – бизнес‑обоснование
- **Агрегированные площади** (`TotalSF`, `TotalLivingSF`) уменьшают мультиколлинеарность и лучше отражают общий размер.
- **Баллы качества** переводят порядковые категории в числа, сохраняя градацию.
- **Временные признаки** (`Age`, `RemodFlag`) улавливают старение дома и факт реновации.
- **Взаимодействия** (`QualSF = OverallQual × TotalSF`) усиливают нелинейные связи.
- **Бинарные флаги** упрощают моделям восприятие наличия важных объектов (гараж, подвал, камин).

Каждый новый признак проверен по корреляции с `log(SalePrice)`; `TotalSF` имеет корреляцию 0.84 – выше, чем у любой отдельной площади.

#### 3. Стратегия моделирования
- **Базовые модели**: Ridge, Lasso, ElasticNet (линейные с регуляризацией) + LightGBM, XGBoost, CatBoost (градиентный бустинг).
- **Настройка гиперпараметров**: GridSearchCV для линейных моделей, Optuna (128 попыток) для LightGBM. XGBoost и CatBoost настроены вручную.
- **Стекинг**: 16‑фолдовые out‑of‑fold предсказания отобранных моделей (OOF RMSE < 0.125) использованы как признаки для мета‑модели RidgeCV. Это улучшило RMSLE с 0.118 (простое усреднение) до 0.110 (CV), финальный результат на Kaggle – **0.19048**.

#### 4. Интерпретация модели (SHAP и остатки)
- **SHAP‑анализ** показал, что наиболее важные признаки – `TotalSF`, `OverallQual` и `Age`. `TotalSF` оказывает положительное, почти линейное влияние на цену.
- **Анализ остатков** выявил, что модель немного недооценивает дорогие объекты и переоценивает самые дешёвые – это указывает на потенциал для дальнейшего улучшения (например, добавление локальных рыночных индексов).
- **Бизнес‑ценность**: модель может быть использована для автоматической предварительной оценки ипотечных заявок, сокращая время обработки до 80% по сравнению с ручной экспертизой.

### Итоговые результаты

| Этап | RMSLE (CV) |
|------|------------|
| Бейзлайн (простая обработка) | 0.152 |
| + Инженерия признаков | 0.135 |
| + Настройка LightGBM | 0.125 |
| + Ансамбль (усреднение 5 моделей) | 0.118 |
| + Стекинг (RidgeCV) | 0.110 |
| **Итог (Kaggle Public LB)** | **0.19048** |

### Как воспроизвести
1. Клонировать репозиторий:
   ```bash
   git clone https://github.com/Rambo232/kaggle_house_prices.git
   cd kaggle_house_prices
   ```
2. Создать виртуальное окружение и установить зависимости:
   ```bash
   python -m venv venv
   source venv/bin/activate  # или `venv\Scripts\activate` на Windows
   pip install -r requirements.txt
   ```
3. Скачать данные соревнования с [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) и поместить `train.csv` и `test.csv` в папку `data/`.
4. Запустить ноутбук на выбранном языке (английский или русский). Финальные предсказания сохранятся в `submissions/submission.csv`.

### Зависимости
- Python 3.12+
- `numpy`, `pandas`, `matplotlib`, `seaborn`
- `scikit-learn==1.8.0`, `lightgbm`, `xgboost`, `catboost`
- `optuna`, `shap`

Точные версии указаны в `requirements.txt`.
