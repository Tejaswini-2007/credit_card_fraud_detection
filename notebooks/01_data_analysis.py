# ============================================================
# CREDIT CARD FRAUD DETECTION
# STEP 1: DATA UNDERSTANDING & EXPLORATORY DATA ANALYSIS
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. LOAD DATASET
# ============================================================

DATA_PATH = "data/creditcard.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("CREDIT CARD FRAUD DETECTION")
print("=" * 70)

print("\nDataset loaded successfully!")


# ============================================================
# 2. BASIC INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DATASET SHAPE")
print("=" * 70)

print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")


# ============================================================
# 3. FIRST FIVE ROWS
# ============================================================

print("\n" + "=" * 70)
print("FIRST 5 ROWS")
print("=" * 70)

print(df.head())


# ============================================================
# 4. COLUMN NAMES
# ============================================================

print("\n" + "=" * 70)
print("COLUMNS")
print("=" * 70)

print(df.columns.tolist())


# ============================================================
# 5. DATA TYPES
# ============================================================

print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)

print(df.dtypes)


# ============================================================
# 6. MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing_values = df.isnull().sum()

print(missing_values)

print("\nTotal missing values:",
      missing_values.sum())


# ============================================================
# 7. DUPLICATES
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATES")
print("=" * 70)

duplicate_count = df.duplicated().sum()

print("Duplicate rows:", duplicate_count)


# ============================================================
# 8. STATISTICAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STATISTICAL SUMMARY")
print("=" * 70)

print(df.describe())


# ============================================================
# 9. CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

class_counts = df["Class"].value_counts()

print(class_counts)


# ============================================================
# 10. CLASS PERCENTAGE
# ============================================================

print("\n" + "=" * 70)
print("CLASS PERCENTAGE")
print("=" * 70)

class_percentage = (
    df["Class"]
    .value_counts(normalize=True)
    .mul(100)
)

print(class_percentage)


# ============================================================
# 11. FRAUD VS LEGITIMATE COUNT
# ============================================================

legitimate = (df["Class"] == 0).sum()
fraud = (df["Class"] == 1).sum()

print("\n" + "=" * 70)
print("TRANSACTION SUMMARY")
print("=" * 70)

print(f"Legitimate transactions : {legitimate}")
print(f"Fraudulent transactions : {fraud}")

print(f"Fraud percentage        : "
      f"{(fraud / len(df)) * 100:.4f}%")


# ============================================================
# 12. VISUALIZE CLASS DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Class"
)

plt.title("Legitimate vs Fraudulent Transactions")
plt.xlabel("Class (0 = Legitimate, 1 = Fraud)")
plt.ylabel("Number of Transactions")

plt.tight_layout()
plt.show()


# ============================================================
# 13. TRANSACTION AMOUNT DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 5))

sns.histplot(
    data=df,
    x="Amount",
    bins=50
)

plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# ============================================================
# 14. AMOUNT BY CLASS
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Class",
    y="Amount"
)

plt.title("Transaction Amount by Class")
plt.xlabel("Class (0 = Legitimate, 1 = Fraud)")
plt.ylabel("Amount")

plt.tight_layout()
plt.show()


print("\n" + "=" * 70)
print("EDA COMPLETED")
print("=" * 70)