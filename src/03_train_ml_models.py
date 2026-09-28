# ============================================================
# CREDIT CARD FRAUD DETECTION
# STEP 3 - RANDOM FOREST + XGBOOST
# ============================================================

import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from xgboost import XGBClassifier


# ============================================================
# 1. CREATE MODEL DIRECTORY
# ============================================================

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# ============================================================
# 2. LOAD PREPROCESSED DATA
# ============================================================

X_train = np.load("outputs/X_train_scaled.npy")
X_test = np.load("outputs/X_test_scaled.npy")

y_train = np.load("outputs/y_train.npy")
y_test = np.load("outputs/y_test.npy")


print("=" * 70)
print("CREDIT CARD FRAUD DETECTION")
print("RANDOM FOREST + XGBOOST")
print("=" * 70)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape :", X_test.shape)

print("\nTraining class distribution:")
print(pd.Series(y_train).value_counts())

print("\nTesting class distribution:")
print(pd.Series(y_test).value_counts())


# ============================================================
# 3. CALCULATE CLASS WEIGHT
# ============================================================

negative_count = np.sum(y_train == 0)
positive_count = np.sum(y_train == 1)

scale_pos_weight = negative_count / positive_count

print("\nScale positive weight:",
      round(scale_pos_weight, 2))


# ============================================================
# 4. RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)

start_time = time.time()

rf_model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    max_depth=None,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_training_time = time.time() - start_time

print(
    f"\nRandom Forest training time: "
    f"{rf_training_time:.2f} seconds"
)


# ============================================================
# 5. RANDOM FOREST PREDICTIONS
# ============================================================

rf_probability = rf_model.predict_proba(X_test)[:, 1]

rf_prediction = (rf_probability >= 0.5).astype(int)


# ============================================================
# 6. RANDOM FOREST METRICS
# ============================================================

rf_precision = precision_score(
    y_test,
    rf_prediction,
    zero_division=0
)

rf_recall = recall_score(
    y_test,
    rf_prediction,
    zero_division=0
)

rf_f1 = f1_score(
    y_test,
    rf_prediction,
    zero_division=0
)

rf_roc_auc = roc_auc_score(
    y_test,
    rf_probability
)

rf_pr_auc = average_precision_score(
    y_test,
    rf_probability
)


print("\n" + "-" * 70)
print("RANDOM FOREST RESULTS")
print("-" * 70)

print(f"Precision : {rf_precision:.4f}")
print(f"Recall    : {rf_recall:.4f}")
print(f"F1 Score  : {rf_f1:.4f}")
print(f"ROC-AUC   : {rf_roc_auc:.4f}")
print(f"PR-AUC    : {rf_pr_auc:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        rf_prediction,
        target_names=["Legitimate", "Fraud"],
        zero_division=0
    )
)


# ============================================================
# 7. RANDOM FOREST CONFUSION MATRIX
# ============================================================

rf_cm = confusion_matrix(
    y_test,
    rf_prediction
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    rf_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Legitimate", "Fraud"],
    yticklabels=["Legitimate", "Fraud"]
)

plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "outputs/random_forest_confusion_matrix.png",
    dpi=300
)

plt.show()


# ============================================================
# 8. SAVE RANDOM FOREST
# ============================================================

joblib.dump(
    rf_model,
    "models/random_forest.joblib"
)

print("\nRandom Forest saved.")


# ============================================================
# 9. XGBOOST
# ============================================================

print("\n" + "=" * 70)
print("TRAINING XGBOOST")
print("=" * 70)

start_time = time.time()

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,

    scale_pos_weight=scale_pos_weight,

    objective="binary:logistic",

    eval_metric="aucpr",

    random_state=42,

    n_jobs=-1,

    tree_method="hist"
)

xgb_model.fit(
    X_train,
    y_train
)

xgb_training_time = time.time() - start_time

print(
    f"\nXGBoost training time: "
    f"{xgb_training_time:.2f} seconds"
)


# ============================================================
# 10. XGBOOST PREDICTIONS
# ============================================================

xgb_probability = xgb_model.predict_proba(
    X_test
)[:, 1]

xgb_prediction = (
    xgb_probability >= 0.5
).astype(int)


# ============================================================
# 11. XGBOOST METRICS
# ============================================================

xgb_precision = precision_score(
    y_test,
    xgb_prediction,
    zero_division=0
)

xgb_recall = recall_score(
    y_test,
    xgb_prediction,
    zero_division=0
)

xgb_f1 = f1_score(
    y_test,
    xgb_prediction,
    zero_division=0
)

xgb_roc_auc = roc_auc_score(
    y_test,
    xgb_probability
)

xgb_pr_auc = average_precision_score(
    y_test,
    xgb_probability
)


print("\n" + "-" * 70)
print("XGBOOST RESULTS")
print("-" * 70)

print(f"Precision : {xgb_precision:.4f}")
print(f"Recall    : {xgb_recall:.4f}")
print(f"F1 Score  : {xgb_f1:.4f}")
print(f"ROC-AUC   : {xgb_roc_auc:.4f}")
print(f"PR-AUC    : {xgb_pr_auc:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        xgb_prediction,
        target_names=["Legitimate", "Fraud"],
        zero_division=0
    )
)


# ============================================================
# 12. XGBOOST CONFUSION MATRIX
# ============================================================

xgb_cm = confusion_matrix(
    y_test,
    xgb_prediction
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    xgb_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Legitimate", "Fraud"],
    yticklabels=["Legitimate", "Fraud"]
)

plt.title("XGBoost - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "outputs/xgboost_confusion_matrix.png",
    dpi=300
)

plt.show()


# ============================================================
# 13. SAVE XGBOOST
# ============================================================

joblib.dump(
    xgb_model,
    "models/xgboost.joblib"
)

print("\nXGBoost saved.")


# ============================================================
# 14. MODEL COMPARISON
# ============================================================

comparison = pd.DataFrame({
    "Model": [
        "Random Forest",
        "XGBoost"
    ],

    "Precision": [
        rf_precision,
        xgb_precision
    ],

    "Recall": [
        rf_recall,
        xgb_recall
    ],

    "F1": [
        rf_f1,
        xgb_f1
    ],

    "ROC_AUC": [
        rf_roc_auc,
        xgb_roc_auc
    ],

    "PR_AUC": [
        rf_pr_auc,
        xgb_pr_auc
    ],

    "Training_Time_Seconds": [
        rf_training_time,
        xgb_training_time
    ]
})


# ============================================================
# 15. DISPLAY COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    comparison.to_string(
        index=False
    )
)


# ============================================================
# 16. SAVE COMPARISON
# ============================================================

comparison.to_csv(
    "outputs/ml_model_comparison.csv",
    index=False
)


# ============================================================
# 17. SAVE TEST PROBABILITIES
# ============================================================

np.save(
    "outputs/rf_probability.npy",
    rf_probability
)

np.save(
    "outputs/xgb_probability.npy",
    xgb_probability
)


print("\n" + "=" * 70)
print("ML MODEL TRAINING COMPLETED")
print("=" * 70)

print("\nSaved files:")

print("models/random_forest.joblib")
print("models/xgboost.joblib")
print("outputs/random_forest_confusion_matrix.png")
print("outputs/xgboost_confusion_matrix.png")
print("outputs/ml_model_comparison.csv")