# ============================================================
# CREDIT CARD FRAUD DETECTION
# STEP 5 - THRESHOLD OPTIMIZATION
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. LOAD DATA
# ============================================================

y_test = np.load(
    "outputs/y_test.npy"
)

xgb_probability = np.load(
    "outputs/xgb_probability.npy"
)

autoencoder_error = np.load(
    "outputs/autoencoder_reconstruction_error.npy"
)


# ============================================================
# 2. FUNCTION FOR THRESHOLD SEARCH
# ============================================================

def optimize_threshold(
    y_true,
    scores,
    thresholds,
    model_name
):

    results = []

    for threshold in thresholds:

        predictions = (
            scores >= threshold
        ).astype(int)

        precision = precision_score(
            y_true,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0
        )

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            predictions
        ).ravel()

        results.append({

            "Model": model_name,

            "Threshold": threshold,

            "Precision": precision,

            "Recall": recall,

            "F1": f1,

            "True_Negatives": tn,

            "False_Positives": fp,

            "False_Negatives": fn,

            "True_Positives": tp
        })

    return pd.DataFrame(results)


# ============================================================
# 3. XGBOOST THRESHOLDS
# ============================================================

xgb_thresholds = np.arange(
    0.05,
    0.96,
    0.01
)


xgb_results = optimize_threshold(
    y_test,
    xgb_probability,
    xgb_thresholds,
    "XGBoost"
)


# ============================================================
# 4. AUTOENCODER THRESHOLDS
# ============================================================

ae_thresholds = np.arange(
    0.5,
    20.01,
    0.1
)


ae_results = optimize_threshold(
    y_test,
    autoencoder_error,
    ae_thresholds,
    "Autoencoder"
)


# ============================================================
# 5. COMBINE RESULTS
# ============================================================

all_results = pd.concat(
    [
        xgb_results,
        ae_results
    ],
    ignore_index=True
)


# ============================================================
# 6. BEST XGBOOST THRESHOLD
# ============================================================

best_xgb = xgb_results.loc[
    xgb_results["F1"].idxmax()
]


# ============================================================
# 7. BEST AUTOENCODER THRESHOLD
# ============================================================

best_ae = ae_results.loc[
    ae_results["F1"].idxmax()
]


# ============================================================
# 8. DISPLAY XGBOOST RESULT
# ============================================================

print("=" * 70)
print("BEST XGBOOST THRESHOLD")
print("=" * 70)

print(
    f"\nThreshold : "
    f"{best_xgb['Threshold']:.3f}"
)

print(
    f"Precision : "
    f"{best_xgb['Precision']:.4f}"
)

print(
    f"Recall    : "
    f"{best_xgb['Recall']:.4f}"
)

print(
    f"F1 Score  : "
    f"{best_xgb['F1']:.4f}"
)

print(
    f"TP        : "
    f"{int(best_xgb['True_Positives'])}"
)

print(
    f"FP        : "
    f"{int(best_xgb['False_Positives'])}"
)

print(
    f"FN        : "
    f"{int(best_xgb['False_Negatives'])}"
)


# ============================================================
# 9. DISPLAY AUTOENCODER RESULT
# ============================================================

print("\n" + "=" * 70)
print("BEST AUTOENCODER THRESHOLD")
print("=" * 70)

print(
    f"\nThreshold : "
    f"{best_ae['Threshold']:.3f}"
)

print(
    f"Precision : "
    f"{best_ae['Precision']:.4f}"
)

print(
    f"Recall    : "
    f"{best_ae['Recall']:.4f}"
)

print(
    f"F1 Score  : "
    f"{best_ae['F1']:.4f}"
)

print(
    f"TP        : "
    f"{int(best_ae['True_Positives'])}"
)

print(
    f"FP        : "
    f"{int(best_ae['False_Positives'])}"
)

print(
    f"FN        : "
    f"{int(best_ae['False_Negatives'])}"
)


# ============================================================
# 10. SAVE RESULTS
# ============================================================

all_results.to_csv(
    "outputs/threshold_optimization_results.csv",
    index=False
)


# ============================================================
# 11. SAVE BEST THRESHOLDS
# ============================================================

joblib.dump(
    float(best_xgb["Threshold"]),
    "models/xgboost_threshold.joblib"
)

joblib.dump(
    float(best_ae["Threshold"]),
    "models/optimized_autoencoder_threshold.joblib"
)


# ============================================================
# 12. DISPLAY TOP RESULTS
# ============================================================

print("\n" + "=" * 70)
print("TOP XGBOOST THRESHOLDS")
print("=" * 70)

print(
    xgb_results
    .sort_values(
        "F1",
        ascending=False
    )
    .head(10)
    .to_string(index=False)
)


print("\n" + "=" * 70)
print("TOP AUTOENCODER THRESHOLDS")
print("=" * 70)

print(
    ae_results
    .sort_values(
        "F1",
        ascending=False
    )
    .head(10)
    .to_string(index=False)
)


print("\n" + "=" * 70)
print("THRESHOLD OPTIMIZATION COMPLETED")
print("=" * 70)