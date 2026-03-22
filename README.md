# Titanic — Kaggle Competition

**Результат: 0.80861 | Место 462 из 12 000+ команд | Топ 4%**

---
📓 [Открыть ноутбук (RU)](./titanic_analysis_ru.ipynb) | [Open notebook (EN)](./titanic_analysis_en.ipynb)

🏆 [Kaggle notebook (с графиками)](https://www.kaggle.com/code/rambo232/titanic-feature-engineering-ensemble)

## О проекте

Это решение соревнования [Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic).

**Цель** — не просто получить высокое место используя модель - "черные ящики", а понять, какие факторы на самом деле
влияли на выживание пассажиров и построить прозрачную модель, подтверждающую эти выводы.

Основной упор сделан на анализ данных, исторический контекст и логическое обоснование
каждого решения. В качестве моделей намеренно выбраны **Логистическая регрессия**
и **Случайный лес** — достаточно прозрачные, чтобы объяснить каждое предсказание.

---

## Ключевые инсайты из EDA

| Наблюдение | Что это значит |
|-----------|----------------|
| Женщины выживали в 3.5 раза чаще мужчин | Правило "женщины и дети вперёд" соблюдалось строго |
| Выживаемость: 1 класс 63%, 3 класс 24% | Физическое расположение кают — ближе к шлюпкам |
| U-образная кривая по размеру семьи | Оптимум 2–4 человека, одиночки и большие семьи — хуже |
| Наличие каюты в записях → вдвое выше выживаемость | Прокси богатства и расположения на корабле |
| Шербур (C) даёт 55% выживших — но это confounding | На самом деле эффект объясняется классом билета |

---

## От инсайтов — к признакам

| Гипотеза | Признак | Обоснование |
|----------|---------|-------------|
| Семья эвакуировалась как единое целое | `Family_Survival` | LOO-кодирование исхода группы |
| Титул = возраст + социальный статус | `Title_TE` | Таргет-энкодинг вместо произвольных меток |
| U-образная кривая по семье | `FamilyBin` | Три группы вместо числа |
| Молодые пассажиры 1-го класса — особая группа | `AgePclass_v2` | Age / Pclass² |
| Наличие каюты = близость к шлюпкам | `HasCabin`, `Deck` | Прокси расположения |
| Тариф зависит от класса, нужна нормализация | `Fare_Pclass` | Убирает межклассовый эффект |

---

## Прогресс результата

| Этап | Изменение | Скор |
|------|-----------|------|
| Бейзлайн (10 признаков) | — | 0.79665 |
| + Family_Survival | LOO-сигнал выживаемости группы | +0.005 |
| + Title_TE | Таргет-энкодинг | +0.002 |
| + AgePclass_v2, Fare_Pclass | Нелинейные взаимодействия | +0.002 |
| + CabinSide | Сторона борта | +0.001 |
| Ансамбль + порог 0.51 | Финальная калибровка | +0.0016 |
| **Итог** | | **0.80861** |

---

## Структура ноутбука

```
1. Импорты
2. Загрузка данных
3. Анализ пропусков
4. EDA — анализ выживаемости (графики, исторический контекст)
5. Предобработка (Title, Age, KS-тест стабильности)
6. Бейзлайн — простая модель как точка отсчёта
7. Инженерия признаков (Family_Survival, Title_TE, взаимодействия, каюта)
8. Финальная модель — ансамбль RF + LR
9. Предсказание и отправка
10. Итоги и выводы
```

---

## Ключевые технические решения

**Family_Survival (LOO по билету и фамилии)**
Если пассажир ехал с кем-то кто выжил → 1.0, погиб → 0.0, нет данных → 0.5.
Собственный исход пассажира не используется — нет утечки данных.

**Title_TE (Leave-One-Out Target Encoding)**
Каждый титул заменяется средней выживаемостью, вычисленной без самого пассажира.
Для теста используется глобальная статистика из train.

**Проверка стабильности (KS-тест)**
Критерий Колмогорова–Смирнова подтвердил, что распределения Age, Fare и FamilySize
в train и test статистически неразличимы — признаки применимы к новым данным.

**Ансамбль и порог**
Два набора признаков (с CabinSide и без) → две модели → среднее вероятностей.
Порог 0.51 вместо 0.50 — калибровка под реальную долю выживших (~32%).

---

## Воспроизведение

```bash
# Клонировать репозиторий
git clone <repo_url>
cd titanic

# Установить зависимости
pip install -r requirements.txt

# Запустить ноутбук
jupyter notebook titanic_ru.ipynb
```

**Зависимости:** pandas, numpy, matplotlib, seaborn, scikit-learn, scipy



---

## English version

**Score: 0.80861 | Rank 462 out of 12,000+ teams | Top 4%**

### About
This is my solution to the [Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic) competition.

**Goal** — not just to get a high score, but to understand which factors actually influenced passenger survival and build a transparent model that confirms these insights.

Models chosen intentionally: **Logistic Regression** + **Random Forest** — interpretable enough to explain every prediction without a "black box".

### Key insights from EDA

| Observation | Meaning |
|-------------|---------|
| Women survived 3.5x more often than men | "Women and children first" was strictly enforced |
| Survival: 1st class 63%, 3rd class 24% | Cabin location = physical proximity to lifeboats |
| U-shaped curve by family size | Optimal 2–4 people — solo and large families fared worse |
| Cherbourg (C) effect is a confounder | Explained entirely by higher 1st class proportion |

### From insights to features

| Hypothesis | Feature | Rationale |
|------------|---------|-----------|
| Family evacuated as one unit | `Family_Survival` | LOO group survival signal |
| Title = age + social status | `Title_TE` | Target encoding instead of arbitrary labels |
| Young 1st class passengers — priority group | `AgePclass_v2` | Age / Pclass² |
| Cabin presence = proximity to lifeboats | `HasCabin`, `Deck` | Physical location proxy |

### Score progression

| Step | Change | Score |
|------|--------|-------|
| Baseline (10 features) | — | 0.79665 |
| + Family_Survival | LOO group signal | +0.005 |
| + Title_TE | Target encoding | +0.002 |
| + AgePclass_v2, Fare_Pclass | Nonlinear interactions | +0.002 |
| + CabinSide | Side of the ship | +0.001 |
| Ensemble + threshold 0.51 | Final calibration | +0.0016 |
| **Final** | | **0.80861** |

### Stack
Python, pandas, numpy, scikit-learn, matplotlib, seaborn, scipy
