# 🫀 Heart Attack Risk Prediction

This project focuses on predicting whether a patient is at risk of a heart attack based on medical attributes using machine learning models.

---

## 📌 Project Overview

The goal of this project is to build a classification model that can accurately predict heart attack risk using patient medical data.

The workflow includes:

- Data loading and exploration (EDA)
- Data preprocessing and feature encoding
- Model training with hyperparameter tuning
- Model evaluation and comparison

---

## 📋 Dataset

The dataset contains **7,000 patients** with the following features:

- `age` — patient age
- `gender` — male / female
- `resting_bp` — resting blood pressure
- `cholesterol` — serum cholesterol
- `fasting_blood_sugar` — fasting blood sugar > 120 mg/dl
- `max_heart_rate` — maximum heart rate achieved
- `exercise_angina` — exercise induced angina
- `oldpeak` — ST depression induced by exercise
- `num_major_vessels` — number of major vessels colored by fluoroscopy
- `thalassemia` — thalassemia type
- `bmi` — body mass index
- `smoking_status` — never / former / current
- `alcohol_consumption` — non-drinker / moderate / heavy
- `physical_activity` — low / moderate / high
- `heart_attack_risk` — **target variable** (0 = not at risk, 1 = at risk)

---

## 🔍 EDA Insight

Key findings from exploratory data analysis:

- `age`, `resting_bp`, and `cholesterol` are **positively correlated** with heart attack risk
- `max_heart_rate` is **negatively correlated** — higher heart rate indicates a healthier heart
- Dataset is **imbalanced** — one class dominates significantly
- No dangerous multicollinearity found (max inter-feature correlation: 0.62)
- `oldpeak` and `num_major_vessels` show **right-skewed** distributions

---

## ⚙️ Preprocessing

Steps applied to clean and prepare the data:

- Dropped `patient_id` — no predictive value
- Filled missing numerical values with **median** (`resting_bp`, `max_heart_rate`, `oldpeak`)
- Filled missing binary values with **mode** (`fasting_blood_sugar`)
- Dropped rows with `"unknown"` values in categorical columns (~7% of data)
- Applied **Label Encoding** for binary features (`gender`, `exercise_angina`)
- Applied **Ordinal Encoding** for ordered features (`alcohol_consumption`, `physical_activity`, `thalassemia`)
- Applied **One-Hot Encoding** for nominal features (`chest_pain_type`, `resting_ecg`, `smoking_status`, `st_slope`)
- Final dataset: **~6,495 rows** after cleaning

---

## 🤖 Modeling

All models were tuned using **GridSearchCV** with `cv=5` and optimized for **Recall**, prioritizing the detection of true heart attack cases and minimizing false negatives.

| Model | Accuracy | Recall (class 1) |
|---|---|---|
| ✅ Logistic Regression | 79.5% | 0.74 |
| Random Forest | 78.1% | 0.70 |
| Gaussian Naive Bayes | 77.3% | 0.74 |
| Decision Tree | 62.6% | 0.53 |

**Best model: Logistic Regression** — `C=1`, `solver=liblinear`

> ⚠️ In a medical context, **Recall is the most critical metric**. A False Negative (predicting a sick patient as healthy) is far more dangerous than a False Positive.

---

## 📁 File Structure

```
heart_attack_project/
│
├── data/
│   ├── raw/
│   │   └── heart_attack.csv          ← original data, never modified
│   └── processed/
│       └── heart_clean.csv           ← cleaned & encoded data
│
├── MODEL/
│   └── heart_risk_model.joblib       ← saved trained model
│
├── notebooks/
│   ├── 01_EDA.ipynb                  ← exploratory analysis & visualization
│   ├── 02_preprocessing.ipynb        ← cleaning & encoding
│   └── 03_modeling.ipynb             ← training, tuning & evaluation
│
└── README.md
```

---

## 🛠️ Dependencies

```bash
pip install pandas scikit-learn joblib matplotlib seaborn
```

---

## 🚀 How to Run

1. Clone the repository
2. Install dependencies
3. Run notebooks in order:
   - `01_EDA.ipynb`
   - `02_preprocessing.ipynb`
   - `03_modeling.ipynb`
4. Trained model is saved as `MODEL/heart_risk_model.joblib`

---

## 📦 Load Saved Model

```python
import joblib

model = joblib.load("MODEL/heart_risk_model.joblib")
prediction = model.predict(X_new)
```
