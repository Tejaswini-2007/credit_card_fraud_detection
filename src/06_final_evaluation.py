# ============================================================
# CREDIT CARD FRAUD DETECTION
# STEP 6 - FINAL MODEL EVALUATION
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


# ============================================================
# 1. DIRECTORIES
# ============================================================

os.makedirs("outputs", exist_ok=True)


# ============================================================
# 2. LOAD TEST LABELS
# ============================================================

y_test = np.load(
    "outputs/y_test.npy"
)


# ============================================================
# 3. LOAD SAVED PREDICTION SCORES
# ============================================================

# Random Forest probabilities
rf_probability = np.load(
    "outputs/rf_probability.npy"
)

# XGBoost probabilities
xgb_probability = np.load(
    "outputs/xgb_probability.npy"
)

# Autoencoder reconstruction errors
ae_error = np.load(
    "outputs/autoencoder_reconstruction_error.npy"
)


# ============================================================
# 4. LOAD OPTIMIZED THRESHOLDS
# ============================================================

xgb_threshold = joblib.load(
    "models/xgboost_threshold.joblib"
)

ae_threshold = joblib.load(
    "models/optimized_autoencoder_threshold.joblib"
)


# ============================================================
# 5. DISPLAY INFORMATION
# ============================================================

print("=" * 75)
print("CREDIT CARD FRAUD DETECTION")
print("FINAL MODEL EVALUATION")
print("=" * 75)

print("\nTest samples:", len(y_test))

print(
    "Actual legitimate:",
    np.sum(y_test == 0)
)

print(
    "Actual fraud:",
    np.sum(y_test == 1)
)

print("\nOptimized thresholds:")

print(
    f"XGBoost     : {xgb_threshold:.3f}"
)

print(
    f"Autoencoder : {ae_threshold:.3f}"
)


# ============================================================
# 6. CREATE PREDICTIONS
# ============================================================

# ------------------------------------------------------------
# Random Forest
# ------------------------------------------------------------

rf_prediction = (
    rf_probability >= 0.50
).astype(int)


# ------------------------------------------------------------
# XGBoost
# ------------------------------------------------------------

xgb_prediction = (
    xgb_probability >= xgb_threshold
).astype(int)


# ------------------------------------------------------------
# Autoencoder
# ------------------------------------------------------------

ae_prediction = (
    ae_error >= ae_threshold
).astype(int)


# ============================================================
# 7. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model_name,
    y_true,
    predictions,
    scores,
    score_type="probability"
):

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

    roc_auc = roc_auc_score(
        y_true,
        scores
    )

    pr_auc = average_precision_score(
        y_true,
        scores
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions
    ).ravel()

    return {
        "Model": model_name,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "True_Negatives": tn,
        "False_Positives": fp,
        "False_Negatives": fn,
        "True_Positives": tp
    }


# ============================================================
# 8. EVALUATE ALL MODELS
# ============================================================

rf_results = evaluate_model(
    "Random Forest",
    y_test,
    rf_prediction,
    rf_probability
)


xgb_results = evaluate_model(
    "XGBoost",
    y_test,
    xgb_prediction,
    xgb_probability
)


ae_results = evaluate_model(
    "Autoencoder",
    y_test,
    ae_prediction,
    ae_error,
    score_type="reconstruction_error"
)


# ============================================================
# 9. CREATE FINAL COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame([
    rf_results,
    xgb_results,
    ae_results
])


# ============================================================
# 10. DISPLAY FINAL COMPARISON
# ============================================================

print("\n" + "=" * 75)
print("FINAL MODEL COMPARISON")
print("=" * 75)

display_columns = [
    "Model",
    "Precision",
    "Recall",
    "F1",
    "ROC_AUC",
    "PR_AUC",
    "False_Positives",
    "False_Negatives",
    "True_Positives"
]

print(
    comparison[
        display_columns
    ].to_string(index=False)
)


# ============================================================
# 11. SAVE COMPARISON
# ============================================================

comparison.to_csv(
    "outputs/final_model_comparison.csv",
    index=False
)

print(
    "\nSaved:"
    "\noutputs/final_model_comparison.csv"
)


# ============================================================
# 12. DETERMINE BEST MODEL
# ============================================================

best_model = comparison.loc[
    comparison["F1"].idxmax()
]

print("\n" + "=" * 75)
print("BEST MODEL BASED ON F1 SCORE")
print("=" * 75)

print(
    f"\nModel     : {best_model['Model']}"
)

print(
    f"Precision : {best_model['Precision']:.4f}"
)

print(
    f"Recall    : {best_model['Recall']:.4f}"
)

print(
    f"F1 Score  : {best_model['F1']:.4f}"
)

print(
    f"ROC-AUC   : {best_model['ROC_AUC']:.4f}"
)

print(
    f"PR-AUC    : {best_model['PR_AUC']:.4f}"
)


# ============================================================
# 13. CONFUSION MATRICES
# ============================================================

models = [
    ("Random Forest", rf_prediction),
    ("XGBoost", xgb_prediction),
    ("Autoencoder", ae_prediction)
]


for model_name, predictions in models:

    cm = confusion_matrix(
        y_test,
        predictions
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Legitimate",
            "Fraud"
        ]
    )

    display.plot(
        ax=ax,
        values_format="d",
        cmap="Blues"
    )

    ax.set_title(
        f"{model_name} - Confusion Matrix"
    )

    plt.tight_layout()

    filename = (
        model_name.lower()
        .replace(" ", "_")
        + "_final_confusion_matrix.png"
    )

    plt.savefig(
        f"outputs/{filename}",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


print(
    "\nConfusion matrices saved."
)


# ============================================================
# 14. METRIC COMPARISON GRAPH
# ============================================================

metrics = [
    "Precision",
    "Recall",
    "F1",
    "ROC_AUC",
    "PR_AUC"
]

x = np.arange(
    len(metrics)
)

width = 0.25

fig, ax = plt.subplots(
    figsize=(11, 6)
)


for i, model_name in enumerate(
    comparison["Model"]
):

    values = comparison.loc[
        comparison["Model"] == model_name,
        metrics
    ].values[0]

    ax.bar(
        x + (i - 1) * width,
        values,
        width,
        label=model_name
    )


ax.set_xlabel(
    "Evaluation Metric"
)

ax.set_ylabel(
    "Score"
)

ax.set_title(
    "Fraud Detection Model Comparison"
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    metrics
)

ax.set_ylim(
    0,
    1
)

ax.legend()

plt.tight_layout()

plt.savefig(
    "outputs/final_model_metric_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 15. PRECISION vs RECALL
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 6)
)

models_names = comparison["Model"]

precision_values = comparison["Precision"]

recall_values = comparison["Recall"]

x = np.arange(
    len(models_names)
)

width = 0.35

ax.bar(
    x - width / 2,
    precision_values,
    width,
    label="Precision"
)

ax.bar(
    x + width / 2,
    recall_values,
    width,
    label="Recall"
)

ax.set_xlabel(
    "Model"
)

ax.set_ylabel(
    "Score"
)

ax.set_title(
    "Precision vs Recall"
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    models_names
)

ax.set_ylim(
    0,
    1
)

ax.legend()

plt.tight_layout()

plt.savefig(
    "outputs/precision_recall_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 16. FALSE POSITIVE / FALSE NEGATIVE COMPARISON
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 6)
)

fp_values = comparison[
    "False_Positives"
]

fn_values = comparison[
    "False_Negatives"
]

x = np.arange(
    len(models_names)
)

width = 0.35

ax.bar(
    x - width / 2,
    fp_values,
    width,
    label="False Positives"
)

ax.bar(
    x + width / 2,
    fn_values,
    width,
    label="False Negatives"
)

ax.set_xlabel(
    "Model"
)

ax.set_ylabel(
    "Number of Transactions"
)

ax.set_title(
    "False Positive vs False Negative Comparison"
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    models_names
)

ax.legend()

plt.tight_layout()

plt.savefig(
    "outputs/error_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 17. SAVE FINAL MODEL INFORMATION
# ============================================================

final_model_info = pd.DataFrame({

    "Selected_Model": [
        best_model["Model"]
    ],

    "Precision": [
        best_model["Precision"]
    ],

    "Recall": [
        best_model["Recall"]
    ],

    "F1": [
        best_model["F1"]
    ],

    "ROC_AUC": [
        best_model["ROC_AUC"]
    ],

    "PR_AUC": [
        best_model["PR_AUC"]
    ]
})


final_model_info.to_csv(
    "outputs/final_selected_model.csv",
    index=False
)


# ============================================================
# 18. CLASSIFICATION REPORTS
# ============================================================

print("\n" + "=" * 75)
print("RANDOM FOREST CLASSIFICATION REPORT")
print("=" * 75)

print(
    classification_report(
        y_test,
        rf_prediction,
        target_names=[
            "Legitimate",
            "Fraud"
        ],
        zero_division=0
    )
)


print("\n" + "=" * 75)
print("XGBOOST CLASSIFICATION REPORT")
print("=" * 75)

print(
    classification_report(
        y_test,
        xgb_prediction,
        target_names=[
            "Legitimate",
            "Fraud"
        ],
        zero_division=0
    )
)


print("\n" + "=" * 75)
print("AUTOENCODER CLASSIFICATION REPORT")
print("=" * 75)

print(
    classification_report(
        y_test,
        ae_prediction,
        target_names=[
            "Legitimate",
            "Fraud"
        ],
        zero_division=0
    )
)


# ============================================================
# 19. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 75)
print("FINAL EVALUATION COMPLETED")
print("=" * 75)

print("\nGenerated files:")

print("1. outputs/final_model_comparison.csv")
print("2. outputs/final_selected_model.csv")
print("3. outputs/random_forest_final_confusion_matrix.png")
print("4. outputs/xgboost_final_confusion_matrix.png")
print("5. outputs/autoencoder_final_confusion_matrix.png")
print("6. outputs/final_model_metric_comparison.png")
print("7. outputs/precision_recall_comparison.png")
print("8. outputs/error_comparison.png")

print("\nProject evaluation complete.")