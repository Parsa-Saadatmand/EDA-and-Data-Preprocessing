# Titanic EDA & Data Preprocessing

Exploratory analysis and data cleaning pipeline for the classic Titanic dataset. The goal here isn't to build a prediction model — it's to take a messy real-world dataset and turn it into something a model could actually be trained on, while pulling out the story the data tells along the way.

## What's in here

The notebook is split into two halves:

**Exploration** — first pass through the data: shape, types, summary stats, and the questions that actually matter for this dataset — who survived, and why. I looked at survival broken down by sex, class, age, and fare, and pulled out the patterns that show up (spoiler: being a woman or a first-class passenger mattered a lot).

**Cleaning & preprocessing** — handling missing values (`Age`, `Embarked`, dropping `Cabin`), capping fare outliers at the 95th percentile, removing duplicates, one-hot encoding categorical columns, and comparing normalization (MinMax) vs standardization (Z-score) on `Age` and `Fare`, wrapped up with a correlation check on the final feature set.

## Key findings

- Overall survival rate was around 38%
- Women survived at roughly 74%, men at around 19% — by far the biggest predictor
- Survival dropped consistently by class: ~63% (1st) → ~47% (2nd) → ~24% (3rd)
- Most passengers were male, 20–40 years old, and traveling third class
- Kids under 10 had noticeably better odds than the rest of the passengers, older passengers had worse odds

## Dataset

The classic Titanic dataset (`train.csv` from [Kaggle's Titanic competition](https://www.kaggle.com/c/titanic/data)), 891 passengers with features like class, sex, age, fare, and survival outcome. Not included in this repo — download it from Kaggle and drop it in as `titanic.csv`.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook titanic_eda_preprocessing.ipynb
```

## Stack

pandas, numpy, matplotlib, seaborn, scikit-learn (for scaling)

## What I'd add next

A model. This notebook stops right before the fun part — the cleaned, encoded, scaled dataset is sitting there ready for logistic regression / random forest / whatever comes next.
