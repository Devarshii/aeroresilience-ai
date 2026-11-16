
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


DATA_FILE = Path("data/processed/flights_clean_2026_08.parquet")
MODEL_DIR = Path("outputs/models")
REPORT_DIR = Path("outputs/reports")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORICAL_FEATURES = [
    "Reporting_Airline",
    "Origin",
    "Dest",
]

NUMERIC_FEATURES = [
    "DayOfWeek",
    "Month",
    "ScheduledDepHour",
    "Distance",
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "Disrupted"


def main():
    print("=" * 60)
    print("AERORESILIENCE AI — FLIGHT DISRUPTION PREDICTION")
    print("=" * 60)

    df = pd.read_parquet(DATA_FILE)
    df["FlightDate"] = pd.to_datetime(df["FlightDate"])

    df = df.sort_values("FlightDate").reset_index(drop=True)

    # Chronological split prevents training on later dates
    # and testing on earlier dates.
    cutoff = pd.Timestamp("2026-08-25")

    train = df[df["FlightDate"] < cutoff].copy()
    test = df[df["FlightDate"] >= cutoff].copy()

    if train.empty or test.empty:
        raise ValueError("Training or testing data is empty.")

    X_train = train[FEATURES]
    y_train = train[TARGET].astype(int)

    X_test = test[FEATURES]
    y_test = test[TARGET].astype(int)

    print(f"\nTraining flights: {len(train):,}")
    print(f"Testing flights: {len(test):,}")
    print(f"Training disruption rate: {y_train.mean():.2%}")
    print(f"Testing disruption rate: {y_test.mean():.2%}")

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="UNKNOWN",
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    model = XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    print("\nTraining XGBoost model...")
    pipeline.fit(X_train, y_train)

    print("Evaluating model on held-out flights...")

    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(
            average_precision_score(y_test, probabilities)
        ),
        "test_disruption_rate": float(y_test.mean()),
    }

    print("\nMODEL PERFORMANCE")
    print("-" * 60)

    for name, value in metrics.items():
        if name.endswith("_rows"):
            print(f"{name}: {value:,}")
        else:
            print(f"{name}: {value:.4f}")

    print("\nCLASSIFICATION REPORT")
    print(classification_report(y_test, predictions, zero_division=0))

    print("\nCONFUSION MATRIX")
    print(confusion_matrix(y_test, predictions))

    model_path = MODEL_DIR / "flight_disruption_xgboost.joblib"
    joblib.dump(pipeline, model_path)

    metrics_path = REPORT_DIR / "disruption_model_metrics.json"

    with open(metrics_path, "w") as file:
        json.dump(metrics, file, indent=4)

    predictions_df = test[
        [
            "FlightDate",
            "Reporting_Airline",
            "Origin",
            "Dest",
            "Disrupted",
        ]
    ].copy()

    predictions_df["PredictedRisk"] = probabilities
    predictions_df["PredictedDisrupted"] = predictions

    predictions_df.to_csv(
        REPORT_DIR / "flight_disruption_predictions.csv",
        index=False,
    )

    print(f"\nModel saved: {model_path}")
    print(f"Metrics saved: {metrics_path}")
    print("Predictions saved successfully.")

    print("\nAERORESILIENCE AI — MODEL TRAINING COMPLETE")


if __name__ == "__main__":
    main()
