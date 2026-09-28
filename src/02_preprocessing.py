# ============================================================
# CREDIT CARD FRAUD DETECTION
# STEP 2 - PREPROCESSING
# ============================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = "data/creditcard.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("CREDIT CARD FRAUD DETECTION - PREPROCESSING")
print("=" * 70)

print("\nOriginal dataset shape:")
print(df.shape)


# ============================================================
# 2. CHECK DUPLICATES
# ============================================================

duplicate_count = df.duplicated().sum()

print("\nDuplicate rows found:", duplicate_count)


# ============================================================
# 3. REMOVE EXACT DUPLICATES
# ============================================================

df = df.drop_duplicates().reset_index(drop=True)

print("\nDataset shape after removing duplicates:")
print(df.shape)


# ============================================================
# 4. CHECK CLASS DISTRIBUTION AFTER DUPLICATE REMOVAL
# ============================================================

print("\nClass distribution after duplicate removal:")

print(df["Class"].value_counts())

print("\nClass percentages:")

print(
    df["Class"]
    .value_counts(normalize=True)
    .mul(100)
)


# ============================================================
# 5. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop("Class", axis=1)

y = df["Class"]


print("\nFeature matrix shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)


print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 7. CHECK CLASS DISTRIBUTION
# ============================================================

print("\nTraining class distribution:")

print(y_train.value_counts())

print("\nTesting class distribution:")

print(y_test.value_counts())


# ============================================================
# 8. SCALE FEATURES
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


print("\nScaling completed.")


# ============================================================
# 9. CALCULATE CLASS IMBALANCE RATIO
# ============================================================

negative_count = (y_train == 0).sum()

positive_count = (y_train == 1).sum()

scale_pos_weight = negative_count / positive_count


print("\n" + "=" * 70)
print("CLASS IMBALANCE")
print("=" * 70)

print("Legitimate training samples:", negative_count)

print("Fraud training samples     :", positive_count)

print(
    "Scale positive weight       :",
    round(scale_pos_weight, 2)
)


# ============================================================
# 10. SAVE SCALER
# ============================================================

joblib.dump(
    scaler,
    "models/scaler.joblib"
)

print("\nScaler saved to:")
print("models/scaler.joblib")


# ============================================================
# 11. SAVE PROCESSED DATA
# ============================================================

np.save(
    "outputs/X_train_scaled.npy",
    X_train_scaled
)

np.save(
    "outputs/X_test_scaled.npy",
    X_test_scaled
)

np.save(
    "outputs/y_train.npy",
    y_train.to_numpy()
)

np.save(
    "outputs/y_test.npy",
    y_test.to_numpy()
)


# ============================================================
# 12. SAVE ORIGINAL TRAIN/TEST FEATURES
# ============================================================

X_train.to_csv(
    "outputs/X_train.csv",
    index=False
)

X_test.to_csv(
    "outputs/X_test.csv",
    index=False
)

y_train.to_csv(
    "outputs/y_train.csv",
    index=False
)

y_test.to_csv(
    "outputs/y_test.csv",
    index=False
)


# ============================================================
# 13. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nFiles created:")
print("1. models/scaler.joblib")
print("2. outputs/X_train_scaled.npy")
print("3. outputs/X_test_scaled.npy")
print("4. outputs/y_train.npy")
print("5. outputs/y_test.npy")
print("6. outputs/X_train.csv")
print("7. outputs/X_test.csv")
print("8. outputs/y_train.csv")
print("9. outputs/y_test.csv")