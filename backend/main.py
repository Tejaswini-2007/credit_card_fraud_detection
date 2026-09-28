import os
import joblib
import numpy as np

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from tensorflow.keras.models import load_model


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="ML + Deep Learning fraud detection system",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# FEATURE ORDER
# ============================================================

FEATURE_COLUMNS = [
    "Time",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",
    "Amount"
]


# ============================================================
# LOAD MODELS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "scaler.joblib"
)

XGB_PATH = os.path.join(
    BASE_DIR,
    "models",
    "xgboost.joblib"
)

AE_PATH = os.path.join(
    BASE_DIR,
    "models",
    "autoencoder.keras"
)

XGB_THRESHOLD_PATH = os.path.join(
    BASE_DIR,
    "models",
    "xgboost_threshold.joblib"
)

AE_THRESHOLD_PATH = os.path.join(
    BASE_DIR,
    "models",
    "optimized_autoencoder_threshold.joblib"
)


scaler = joblib.load(
    SCALER_PATH
)

xgb_model = joblib.load(
    XGB_PATH
)

autoencoder = load_model(
    AE_PATH
)

xgb_threshold = joblib.load(
    XGB_THRESHOLD_PATH
)

ae_threshold = joblib.load(
    AE_THRESHOLD_PATH
)


# ============================================================
# REQUEST SCHEMA
# ============================================================

class Transaction(BaseModel):

    Time: float

    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    Amount: float


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():

    return {
        "status": "running",
        "message": "Credit Card Fraud Detection API"
    }


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.get("/model-info")
def model_info():

    return {

        "primary_model": "XGBoost",

        "xgboost_threshold": float(
            xgb_threshold
        ),

        "autoencoder_threshold": float(
            ae_threshold
        ),

        "features": FEATURE_COLUMNS,

        "models": [
            "XGBoost",
            "Random Forest",
            "Autoencoder"
        ]
    }


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
def predict(transaction: Transaction):

    # --------------------------------------------------------
    # Create feature array
    # --------------------------------------------------------

    values = [
        getattr(
            transaction,
            feature
        )
        for feature in FEATURE_COLUMNS
    ]

    X = np.array(
        [values],
        dtype=float
    )


    # --------------------------------------------------------
    # Scaling
    # --------------------------------------------------------

    X_scaled = scaler.transform(
        X
    )


    # --------------------------------------------------------
    # XGBoost prediction
    # --------------------------------------------------------

    fraud_probability = float(
        xgb_model.predict_proba(
            X_scaled
        )[0][1]
    )


    xgb_prediction = (
        fraud_probability >= xgb_threshold
    )


    # --------------------------------------------------------
    # Autoencoder anomaly score
    # --------------------------------------------------------

    reconstruction = autoencoder.predict(
        X_scaled,
        verbose=0
    )

    reconstruction_error = float(
        np.mean(
            np.square(
                X_scaled - reconstruction
            )
        )
    )


    ae_prediction = (
        reconstruction_error >= ae_threshold
    )


    # --------------------------------------------------------
    # Risk level
    # --------------------------------------------------------

    if fraud_probability >= 0.80:

        risk_level = "HIGH"

    elif fraud_probability >= 0.53:

        risk_level = "MEDIUM"

    else:

        risk_level = "LOW"


    # --------------------------------------------------------
    # Final response
    # --------------------------------------------------------

    return {

        "prediction": (
            "Fraudulent"
            if xgb_prediction
            else "Legitimate"
        ),

        "is_fraud": bool(
            xgb_prediction
        ),

        "fraud_probability": round(
            fraud_probability * 100,
            2
        ),

        "xgboost_threshold": round(
            float(xgb_threshold) * 100,
            2
        ),

        "risk_level": risk_level,

        "autoencoder": {

            "reconstruction_error":
                round(
                    reconstruction_error,
                    4
                ),

            "threshold":
                round(
                    float(ae_threshold),
                    4
                ),

            "anomaly_detected":
                bool(ae_prediction)
        }
    }