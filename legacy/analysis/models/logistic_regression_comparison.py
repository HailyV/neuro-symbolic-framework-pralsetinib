import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score

# Data from faers_pull.py
comparative_path = "../data/processed/comparative_drug_ae_data.csv"
pralsetinib_path = "../data/processed/drug_ae_features.csv"

df_new = pd.read_csv(comparative_path)
df_old = pd.read_csv(pralsetinib_path)

# IMPORTANT: Since df_new (API data) lacks features like 'path_count', 
# for this example, we will fill them with randomized dummy values 
for col in ["path_count", "max_path_score", "go_overlap"]:
    if col not in df_new.columns:
        df_new[col] = np.random.uniform(0, 1, size=len(df_new))

df = pd.concat([df_old, df_new], ignore_index=True)
print(df.head())

FEATURE_COLS = ["path_count", "max_path_score", "go_overlap", "target_faers_score"]
df = df.dropna(subset=FEATURE_COLS + ["label"])

class_counts = df["label"].value_counts()
print(f"Dataset Label Distribution:\n{class_counts}")

if len(class_counts) < 2:
    print("Still only one class found. Try lowering your FAERS threshold 'k'.")
    exit()

# Prepare modeling data
X = df[FEATURE_COLS].values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced"
    ))
])

# Model training and evaluation
print("Training Logistic Regression")
model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)[:, 1]

# Metrics
roc_auc = roc_auc_score(y_test, y_prob)
pr_auc  = average_precision_score(y_test, y_prob)

print(f"ROC-AUC: {roc_auc:.3f}")
print(f"PR-AUC:  {pr_auc:.3f}")

# Feature Importance
coefs = model.named_steps["clf"].coef_[0]
coef_df = pd.DataFrame({
    "feature": FEATURE_COLS,
    "weight": coefs
}).sort_values("weight", ascending=False)

print("\nLearned feature weights (Higher = more predictive of toxicity):")
print(coef_df)