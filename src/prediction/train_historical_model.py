
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


# ====================================================
# AERORESILIENCE AI
# STEP 10 — HISTORICAL MODEL TRAINING
# ====================================================

DATA_FILE = Path(
    "data/processed/flights_historical_2026_06_08.parquet"
)

MODEL_DIR = Path("outputs/models")
REPORT_DIR = Path("outputs/reports")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORICAL = [
    "Reporting_Airline",
    "Origin",
    "Dest",
]

NUMERIC = [
    "DayOfWeek",
    "Month",
    "ScheduledDepHour",
    "Distance",
]

FEATURES = CATEGORICAL + NUMERIC
TARGET = "Disrupted"


def build_pipeline():

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(strategy="median"),
                NUMERIC,
            ),
            (
                "categorical",
                Pipeline([
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
                        ),
                    ),
                ]),
                CATEGORICAL,
            ),
        ]
    )

    model = XGBClassifier(
        n_estimators=300,
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

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def get_metrics(actual, probabilities, threshold):

    predicted = (
        probabilities >= threshold
    ).astype(int)

    return {
        "accuracy": float(
            accuracy_score(actual, predicted)
        ),
        "precision": float(
            precision_score(
                actual, predicted, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                actual, predicted, zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                actual, predicted, zero_division=0
            )
        ),
        "roc_auc": float(
            roc_auc_score(actual, probabilities)
        ),
        "pr_auc": float(
            average_precision_score(actual, probabilities)
        ),
        "alerts": int(predicted.sum()),
        "missed_disruptions": int(
            ((actual == 1) & (predicted == 0)).sum()
        ),
    }


def main():

    print("=" * 65)
    print("AERORESILIENCE AI — HISTORICAL ML TRAINING")
    print("=" * 65)

    df = pd.read_parquet(DATA_FILE)

    train = df[df["Month"] == 6].copy()
    validation = df[df["Month"] == 7].copy()
    test = df[df["Month"] == 8].copy()

    if any(
        dataset.empty
        for dataset in [train, validation, test]
    ):
        raise ValueError(
            "Training, validation, or evaluation data is empty."
        )

    print(f"\nJune training records: {len(train):,}")
    print(f"July validation records: {len(validation):,}")
    print(f"August evaluation records: {len(test):,}")

    print("\nTraining initial model on June...")

    pipeline = build_pipeline()

    pipeline.fit(
        train[FEATURES],
        train[TARGET],
    )

    print("Evaluating on July validation data...")

    val_probabilities = pipeline.predict_proba(
        validation[FEATURES]
    )[:, 1]

    val_actual = validation[TARGET].astype(int).to_numpy()

    thresholds = np.arange(0.15, 0.651, 0.05)

    results = []

    for threshold in thresholds:

        metrics = get_metrics(
            val_actual,
            val_probabilities,
            float(threshold),
        )

        results.append({
            "threshold": round(float(threshold), 2),
            **metrics,
        })

    threshold_df = pd.DataFrame(results)

    threshold_df.to_csv(
        REPORT_DIR / "historical_validation_thresholds.csv",
        index=False,
    )

    # Select the threshold with the highest validation F1.
    # This is a statistical starting point, not a final
    # airline business-cost-optimized alert policy.
    best_row = threshold_df.sort_values(
        ["f1", "precision"],
        ascending=False,
    ).iloc[0]

    best_threshold = float(best_row["threshold"])

    print("\nJULY VALIDATION RESULTS")
    print(threshold_df[
        ["threshold", "precision", "recall", "f1", "alerts"]
    ].to_string(index=False))

    print(f"\nSelected validation threshold: {best_threshold:.2f}")

    # Refit model using June and July together.
    print("\nRetraining on June + July...")

    full_train = pd.concat(
        [train, validation],
        ignore_index=True,
    )

    final_pipeline = build_pipeline()

    final_pipeline.fit(
        full_train[FEATURES],
        full_train[TARGET],
    )

    print("\nEvaluating on August...")

    test_probabilities = final_pipeline.predict_proba(
        test[FEATURES]
    )[:, 1]

    test_actual = test[TARGET].astype(int).to_numpy()

    final_metrics = get_metrics(
        test_actual,
        test_probabilities,
        best_threshold,
    )

    baseline_accuracy = float(
        max(test_actual.mean(), 1 - test_actual.mean())
    )

    print("\n" + "=" * 65)
    print("AUGUST HISTORICAL EVALUATION RESULTS")
    print("=" * 65)

    print(f"Baseline majority-class accuracy: {baseline_accuracy:.4f}")
    print(f"Selected threshold: {best_threshold:.2f}")

    for name, value in final_metrics.items():
        if name in ["alerts", "missed_disruptions"]:
            print(f"{name}: {value:,}")
        else:
            print(f"{name}: {value:.4f}")

    final_predictions = (
        test_probabilities >= best_threshold
    ).astype(int)

    print("\nCONFUSION MATRIX")
    print(confusion_matrix(
        test_actual,
        final_predictions,
    ))

    # Save model and metadata.
    model_file = MODEL_DIR / "historical_disruption_xgboost.joblib"

    joblib.dump(
        final_pipeline,
        model_file,
    )

    report = {
        "training_months": ["2026-06", "2026-07"],
        "validation_month": "2026-07",
        "evaluation_month": "2026-08",
        "evaluation_status": (
            "exploratory: August previously inspected"
        ),
        "selected_threshold": best_threshold,
        "baseline_accuracy": baseline_accuracy,
        "metrics": final_metrics,
    }

    with open(
        REPORT_DIR / "historical_model_metrics.json",
        "w",
    ) as file:
        json.dump(report, file, indent=4)

    predictions = test[
        [
            "FlightDate",
            "Reporting_Airline",
            "Origin",
            "Dest",
            "Disrupted",
        ]
    ].copy()

    predictions["PredictedRisk"] = test_probabilities
    predictions["PredictedDisrupted"] = final_predictions

    predictions.to_parquet(
        REPORT_DIR / "historical_predictions.parquet",
        index=False,
    )

    print(f"\nModel saved: {model_file}")
    print("Evaluation reports saved successfully.")

    print("\nAERORESILIENCE AI — HISTORICAL MODEL COMPLETE")


if __name__ == "__main__":
    main()
