import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

# Load data
df = pd.read_csv("../data/processed/drug_ae_features.csv")

FEATURE_COLS = [
    "path_count",
    "max_path_score",
    "go_overlap",
    "target_faers_score"
]

# Preprocessing

X = df[FEATURE_COLS].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define and Train One-Class SVM

model = OneClassSVM(kernel="rbf", gamma=0.1, nu=0.05)
model.fit(X_scaled)

# Predict
# Returns 1 for inliers, -1 for outliers
predictions = model.predict(X_scaled)
scores = model.decision_function(X_scaled) # Distance from the boundary

df['is_inlier'] = predictions
df['anomaly_score'] = scores

print("\nResults (Positive score = similar to training data):")
print(df[['ae', 'is_inlier', 'anomaly_score']])

test_data = pd.DataFrame([
    # Test Case 1: Very similar to Pralsetinib 'Neurological' row
    {"path_count": 15, "max_path_score": 0.5, "go_overlap": 0.1, "target_faers_score": 20, "desc": "Similar to known"},
    
    # Test Case 2: Extreme outlier
    {"path_count": 500, "max_path_score": 5.0, "go_overlap": 0.9, "target_faers_score": 10000, "desc": "Extreme Outlier"},
    
    # Test Case 3: Near-zero
    {"path_count": 0, "max_path_score": 0.0, "go_overlap": 0.0, "target_faers_score": 0, "desc": "Empty/Safe Pair"}
])

# Scale and Predict
X_test = test_data[["path_count", "max_path_score", "go_overlap", "target_faers_score"]].values
X_test_scaled = scaler.transform(X_test)
preds = model.predict(X_test_scaled)
scores = model.decision_function(X_test_scaled)

test_data['prediction'] = ["Inlier (1)" if p == 1 else "Anomaly (-1)" for p in preds]
test_data['score'] = scores

print("Synthetic Test Results")
print(test_data[['desc', 'prediction', 'score']])