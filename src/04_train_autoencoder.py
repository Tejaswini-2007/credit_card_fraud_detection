# ============================================================
# CREDIT CARD FRAUD DETECTION
# STEP 4 - AUTOENCODER ANOMALY DETECTION
# ============================================================

import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense
)
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

np.random.seed(42)
tf.random.set_seed(42)


# ============================================================
# 2. DIRECTORIES
# ============================================================

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# ============================================================
# 3. LOAD DATA
# ============================================================

X_train = np.load(
    "outputs/X_train_scaled.npy"
)

X_test = np.load(
    "outputs/X_test_scaled.npy"
)

y_train = np.load(
    "outputs/y_train.npy"
)

y_test = np.load(
    "outputs/y_test.npy"
)


print("=" * 70)
print("AUTOENCODER FRAUD DETECTION")
print("=" * 70)

print("\nTraining data:", X_train.shape)
print("Testing data :", X_test.shape)


# ============================================================
# 4. SELECT ONLY LEGITIMATE TRAINING TRANSACTIONS
# ============================================================

X_train_normal = X_train[y_train == 0]

print("\nLegitimate training samples:")
print(X_train_normal.shape)

print(
    "\nFraud samples excluded from Autoencoder training:",
    np.sum(y_train == 1)
)


# ============================================================
# 5. BUILD AUTOENCODER
# ============================================================

input_dim = X_train.shape[1]

input_layer = Input(
    shape=(input_dim,)
)


# Encoder

encoded = Dense(
    16,
    activation="relu"
)(input_layer)

encoded = Dense(
    8,
    activation="relu"
)(encoded)


# Decoder

decoded = Dense(
    16,
    activation="relu"
)(encoded)

decoded = Dense(
    input_dim,
    activation="linear"
)(decoded)


autoencoder = Model(
    input_layer,
    decoded
)


# ============================================================
# 6. COMPILE
# ============================================================

autoencoder.compile(
    optimizer="adam",
    loss="mse"
)


print("\nAutoencoder architecture:")

autoencoder.summary()


# ============================================================
# 7. EARLY STOPPING
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


# ============================================================
# 8. TRAIN
# ============================================================

print("\n" + "=" * 70)
print("TRAINING AUTOENCODER")
print("=" * 70)

start_time = time.time()

history = autoencoder.fit(
    X_train_normal,
    X_train_normal,

    epochs=50,

    batch_size=256,

    validation_split=0.20,

    shuffle=True,

    callbacks=[early_stopping],

    verbose=1
)

training_time = time.time() - start_time

print(
    f"\nAutoencoder training time: "
    f"{training_time:.2f} seconds"
)


# ============================================================
# 9. RECONSTRUCTION ERROR
# ============================================================

print("\nCalculating reconstruction errors...")

X_test_reconstructed = autoencoder.predict(
    X_test,
    batch_size=1024,
    verbose=0
)

reconstruction_error = np.mean(
    np.square(
        X_test - X_test_reconstructed
    ),
    axis=1
)


# ============================================================
# 10. INSPECT ERROR DISTRIBUTION
# ============================================================

normal_errors = reconstruction_error[
    y_test == 0
]

fraud_errors = reconstruction_error[
    y_test == 1
]


print("\n" + "=" * 70)
print("RECONSTRUCTION ERROR")
print("=" * 70)

print(
    "\nLegitimate mean error:",
    np.mean(normal_errors)
)

print(
    "Fraud mean error:",
    np.mean(fraud_errors)
)

print(
    "\nLegitimate median error:",
    np.median(normal_errors)
)

print(
    "Fraud median error:",
    np.median(fraud_errors)
)


# ============================================================
# 11. INITIAL THRESHOLD
# ============================================================

threshold = np.percentile(
    normal_errors,
    99
)

print(
    "\nInitial anomaly threshold (99th percentile):",
    threshold
)


# ============================================================
# 12. AUTOENCODER PREDICTIONS
# ============================================================

ae_prediction = (
    reconstruction_error >= threshold
).astype(int)


# ============================================================
# 13. METRICS
# ============================================================

ae_precision = precision_score(
    y_test,
    ae_prediction,
    zero_division=0
)

ae_recall = recall_score(
    y_test,
    ae_prediction,
    zero_division=0
)

ae_f1 = f1_score(
    y_test,
    ae_prediction,
    zero_division=0
)

ae_roc_auc = roc_auc_score(
    y_test,
    reconstruction_error
)

ae_pr_auc = average_precision_score(
    y_test,
    reconstruction_error
)


print("\n" + "=" * 70)
print("AUTOENCODER RESULTS")
print("=" * 70)

print(
    f"Precision : {ae_precision:.4f}"
)

print(
    f"Recall    : {ae_recall:.4f}"
)

print(
    f"F1 Score  : {ae_f1:.4f}"
)

print(
    f"ROC-AUC   : {ae_roc_auc:.4f}"
)

print(
    f"PR-AUC    : {ae_pr_auc:.4f}"
)


# ============================================================
# 14. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

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
# 15. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    ae_prediction
)

print("\nConfusion Matrix:")

print(cm)

plt.figure(
    figsize=(6, 5)
)

plt.imshow(
    cm
)

plt.title(
    "Autoencoder - Confusion Matrix"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.colorbar()

plt.xticks(
    [0, 1],
    ["Legitimate", "Fraud"]
)

plt.yticks(
    [0, 1],
    ["Legitimate", "Fraud"]
)

for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    "outputs/autoencoder_confusion_matrix.png",
    dpi=300
)

plt.show()


# ============================================================
# 16. RECONSTRUCTION ERROR DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.hist(
    normal_errors,
    bins=100,
    alpha=0.7,
    label="Legitimate"
)

plt.hist(
    fraud_errors,
    bins=100,
    alpha=0.7,
    label="Fraud"
)

plt.axvline(
    threshold,
    linestyle="--",
    label="Threshold"
)

plt.xlabel(
    "Reconstruction Error"
)

plt.ylabel(
    "Frequency"
)

plt.title(
    "Autoencoder Reconstruction Error Distribution"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "outputs/autoencoder_error_distribution.png",
    dpi=300
)

plt.show()


# ============================================================
# 17. SAVE MODEL
# ============================================================

autoencoder.save(
    "models/autoencoder.keras"
)


# ============================================================
# 18. SAVE RECONSTRUCTION ERRORS
# ============================================================

np.save(
    "outputs/autoencoder_reconstruction_error.npy",
    reconstruction_error
)


# ============================================================
# 19. SAVE THRESHOLD
# ============================================================

joblib.dump(
    threshold,
    "models/autoencoder_threshold.joblib"
)


# ============================================================
# 20. SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(
    history.history
)

history_df.to_csv(
    "outputs/autoencoder_training_history.csv",
    index=False
)


# ============================================================
# 21. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("AUTOENCODER TRAINING COMPLETED")
print("=" * 70)

print("\nSaved files:")

print("models/autoencoder.keras")

print("models/autoencoder_threshold.joblib")

print("outputs/autoencoder_confusion_matrix.png")

print("outputs/autoencoder_error_distribution.png")

print(
    "outputs/autoencoder_reconstruction_error.npy"
)

print(
    "outputs/autoencoder_training_history.csv"
)