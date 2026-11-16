
from pathlib import Path
import json

import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# =====================================================
# AERORESILIENCE AI
# STEP 11 — OPERATIONAL ALERT CAPACITY ANALYSIS
# =====================================================

REPORT_DIR = Path("outputs/reports")

VALIDATION_FILE = (
    REPORT_DIR / "historical_validation_thresholds.csv"
)

PREDICTIONS_FILE = (
    REPORT_DIR / "historical_predictions.parquet"
)

METRICS_FILE = (
    REPORT_DIR / "historical_model_metrics.json"
)

ALERT_CAPACITY = 0.20


def evaluate(actual, predictions):
    matrix = confusion_matrix(
        actual, predictions, labels=[0, 1]
    )

    tn, fp, fn, tp = matrix.ravel()

    return {
        "total_flights": len(actual),
        "alerts": int(predictions.sum()),
        "alert_rate": float(predictions.mean()),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_negative": int(tn),
        "precision": float(
            precision_score(actual, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(actual, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(actual, predictions, zero_division=0)
        ),
    }


def main():
    print("=" * 65)
    print("AERORESILIENCE AI — ALERT CAPACITY ANALYSIS")
    print("=" * 65)

    validation = pd.read_csv(VALIDATION_FILE)
    august = pd.read_parquet(PREDICTIONS_FILE)

    with open(METRICS_FILE) as file:
        model_metadata = json.load(file)

    july_records = 631970

    validation["alert_rate"] = (
        validation["alerts"] / july_records
    )

    feasible = validation[
        validation["alert_rate"] <= ALERT_CAPACITY
    ].copy()

    if feasible.empty:
        raise ValueError(
            "No validation threshold satisfies alert capacity. "
            "Expand the threshold grid and rerun Step 10."
        )

    # Choose the highest-recall feasible threshold
    # using JULY validation data, not August.
    chosen = feasible.sort_values(
        ["recall", "precision"],
        ascending=False,
    ).iloc[0]

    selected_threshold = float(chosen["threshold"])

    print("\nJULY VALIDATION — CAPACITY POLICY")
    print(f"Assumed alert capacity: {ALERT_CAPACITY:.0%}")
    print(f"Selected threshold: {selected_threshold:.2f}")
    print(f"Validation alert rate: {chosen['alert_rate']:.2%}")
    print(f"Validation recall: {chosen['recall']:.2%}")

    actual = august["Disrupted"].astype(int)

    probabilities = august["PredictedRisk"]

    old_threshold = float(
        model_metadata["selected_threshold"]
    )

    original_predictions = (
        probabilities >= old_threshold
    ).astype(int)

    capacity_predictions = (
        probabilities >= selected_threshold
    ).astype(int)

    original = evaluate(
        actual, original_predictions
    )

    capacity = evaluate(
        actual, capacity_predictions
    )

    comparison = pd.DataFrame([
        {"policy": "Original F1 threshold", **original},
        {"policy": "Capacity-constrained threshold", **capacity},
    ])

    comparison.to_csv(
        REPORT_DIR / "alert_capacity_comparison.csv",
        index=False,
    )

    print("\nAUGUST — EXPLORATORY POLICY COMPARISON")
    print(
        comparison[
            [
                "policy",
                "alerts",
                "alert_rate",
                "precision",
                "recall",
                "false_positive",
                "false_negative",
            ]
        ].to_string(index=False)
    )

    print("\nBUSINESS INTERPRETATION")
    print(
        "Alerts avoided:",
        original["alerts"] - capacity["alerts"]
    )

    print(
        "Additional missed disruptions:",
        capacity["false_negative"]
        - original["false_negative"]
    )

    print("\nIMPORTANT")
    print(
        "The 20% alert capacity is hypothetical. "
        "August is an exploratory evaluation, not "
        "a completely untouched final test."
    )

    print("\nAERORESILIENCE AI — ALERT ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
