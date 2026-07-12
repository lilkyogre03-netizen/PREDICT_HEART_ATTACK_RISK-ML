heart_attack_project/
│
├── data/
│   ├── raw/                    ← raw data
│   │   └── heart_attack.csv
│   └── processed/              ← data after cleaning
│       └── heart_clean.csv
│
├── notebooks/
│   ├── 01_EDA.ipynb            ← eksplorasi & visualisasi
│   ├── 02_preprocessing.ipynb  ← cleaning & encoding
│   └── 03_modeling.ipynb       ← training & evaluasi model
│
├── src/                        ← kalau udah advanced
│   ├── preprocessing.py
│   └── model.py
│
├── models/                     ← final model 
│   └── heart_risk_model.joblib
│
└── README.md                  