# Logistic Regression for Drug → AE (or Toxicity Theme) Prediction
# Supervised using FAERS-derived labels

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score

#Load feature table
# Expected columns:
# drug_id | ae | path_count | max_path_score | go_overlap | target_faers_score | label

df = pd.read_csv("../data/processed/drug_ae_features.csv")

FEATURE_COLS = [
    "path_count",
    "max_path_score",
    "go_overlap",
    "target_faers_score"
]
df = df.dropna(subset=FEATURE_COLS + ["label"])

X = df[FEATURE_COLS].values

y = df["label"].values  # binary label based on FAERS threshold k
print(df.head())
# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Logistic regression model
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

# Train
model.fit(X_train, y_train)


# Evaluate
y_prob = model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_prob)
pr_auc  = average_precision_score(y_test, y_prob)

print(f"ROC-AUC: {roc_auc:.3f}")
print(f"PR-AUC:  {pr_auc:.3f}")

# Inspect learned feature weights

coefs = model.named_steps["clf"].coef_[0]

coef_df = pd.DataFrame({
    "feature": FEATURE_COLS,
    "weight": coefs
}).sort_values("weight", ascending=False)

print("\nLearned feature weights:")
print(coef_df)
