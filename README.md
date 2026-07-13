🫀 Heart Attack Risk Prediction

This project focuses on predicting whether a patient is at risk of a heart attack based on medical attributes using multiple machine learning models with hyperparameter tuning.


📊 Data Insight


Dataset contains 7,000 patients with 15+ medical features
Target variable: heart_attack_risk (0 = not at risk, 1 = at risk)
Dataset is imbalanced — one class dominates significantly
Features include demographics (age, gender), vitals (resting_bp, max_heart_rate, cholesterol), and lifestyle factors (smoking_status, alcohol_consumption, physical_activity)



🔍 EDA Insight


age shows a bell-curve distribution (18–90 years, majority 40–65)
cholesterol and resting_bp are right-skewed with notable outliers above 280–300 and 155–160 respectively
oldpeak and num_major_vessels are heavily right-skewed — majority of values near zero
Binary features (fasting_blood_sugar, exercise_angina, family_history, diabetes) are imbalanced with one value dominating
stress_level shows a uniform distribution across all levels


Key Correlations to Target (heart_attack_risk):

FeatureCorrelationDirectionage0.35Higher age → higher riskmax_heart_rate-0.32Higher max HR → lower riskresting_bp0.29Higher BP → higher riskcholesterol0.25Higher cholesterol → higher risknum_major_vessels0.25More vessels → higher riskdiabetes0.22Presence → higher risk

No dangerous multicollinearity (>0.8) found between features. Highest inter-feature correlation is age vs max_heart_rate (-0.62), which is medically expected.


⚙️ Preprocessing Insight


Dropped: patient_id (unique identifier, no predictive value)
Missing values handled:

Numerical columns (resting_bp, max_heart_rate, oldpeak) → filled with median
Binary column (fasting_blood_sugar) → filled with mode
Categorical columns with "unknown" values (smoking_status, physical_activity, alcohol_consumption) → rows dropped (~7% of data)



Encoding:

Binary features (gender, exercise_angina) → Label Encoding (map to 0/1)
Ordinal features (alcohol_consumption, physical_activity) → Manual Mapping (preserving order)
Ordinal medical features (thalassemia, st_slope) → Manual Mapping
Nominal features (chest_pain_type, resting_ecg, smoking_status) → One-Hot Encoding (pd.get_dummies)



Final dataset: ~6,495 rows after cleaning



🤖 Modeling Insight

All models were tuned using GridSearchCV with cv=5 and optimized for Recall (prioritizing detection of true heart attack cases — minimizing false negatives).

Model Comparison (after tuning):

ModelAccuracyRecall (class 1)Logistic Regression~79.5% 🥇0.74 🥇Random Forest~78.1% 🥈0.70 🥈Gaussian Naive Bayes~77.3% 🥉0.74 🥇Decision Tree~62.6% ❌0.53 ❌

Best Model: Logistic Regression


Best params: C=1, solver=liblinear
Chosen for highest accuracy, strong recall, and interpretability
Recall of 0.74 means 74% of actual heart attack patients are correctly identified



⚠️ In a medical context, Recall is the most critical metric — a False Negative (predicting a sick patient as healthy) is far more dangerous than a False Positive.




📁 File Structure

heart_attack_project/
│
├── data/
│   ├── raw/                        ← original data, never modified
│   │   └── heart_attack.csv
│   └── processed/                  ← cleaned & encoded data
│       └── heart_clean.csv
│
├── MODEL/                          ← saved trained model
│   └── heart_risk_model.joblib
│
├── notebooks/
│   ├── 01_EDA.ipynb                ← exploratory data analysis & visualization
│   ├── 02_preprocessing.ipynb      ← data cleaning & encoding
│   └── 03_modeling.ipynb           ← model training, tuning & evaluation
│
└── README.md


🛠️ Dependencies

bashpip install pandas scikit-learn joblib matplotlib seaborn


🚀 How to Run


Clone the repository
Install dependencies
Run notebooks in order: 01_EDA → 02_preprocessing → 03_modeling
Trained model is saved as MODEL/heart_risk_model.joblib