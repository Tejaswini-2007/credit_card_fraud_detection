 🛡️ FraudShield AI — Credit Card Fraud Detection

> An intelligent credit card fraud detection system combining Machine Learning and Deep Learning to identify fraudulent transactions, handle severe class imbalance, optimize detection thresholds, and provide real-time predictions through a web interface.


 📌 Overview

Credit card fraud is a highly imbalanced classification problem where fraudulent transactions represent only a very small fraction of all transactions.

FraudShield AI is a machine learning-based fraud detection system designed to distinguish between:

- ✅ Legitimate transactions
- 🚨 Fraudulent transactions

The project compares multiple approaches and selects the best-performing model based on fraud-focused evaluation metrics rather than accuracy alone.

The final system combines:

- Random Forest
- XGBoost
- Autoencoder-based anomaly detection
- Threshold optimization
- FastAPI backend
- Interactive web frontend

The primary classification model selected by the evaluation pipeline is **XGBoost**.

---

 🎯 Project Objectives

The main objectives of this project are:

1. Detect fraudulent credit card transactions.
2. Handle severe class imbalance.
3. Preprocess and scale transaction data.
4. Train and compare multiple ML/DL models.
5. Evaluate models using fraud-focused metrics.
6. Optimize the fraud detection threshold.
7. Compare supervised and unsupervised approaches.
8. Deploy the selected model through an API.
9. Provide an interactive web interface for transaction analysis.


🧠 Models Used

 1. Random Forest

Random Forest is used as a baseline supervised classification model.

It combines multiple decision trees and makes predictions based on the aggregated output of the trees.

 Role

- Baseline comparison
- Handles nonlinear relationships
- Provides strong classification performance


 2. XGBoost ⭐

XGBoost is the primary fraud classification model.

It uses gradient boosting to build an ensemble of decision trees sequentially.

After threshold optimization, XGBoost achieved the best F1 score among the evaluated models.

