
from pathlib import Path

import pandas as pd
import plotly.express as px

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

PREDICTIONS_FILE = Path(
    "outputs/reports/flight_disruption_predictions.csv"
)

OUTPUT_DIR = Path("outputs/reports")
FIGURE_DIR = Path("outputs/figures")

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 60)
    print("AERORESILIENCE AI — MODEL THRESHOLD ANALYSIS")
    print("=" * 60)

    df = pd.read_csv(PREDICTIONS_FILE)

    y_true = df["Disrupted"].astype(int)
    probabilities = df["PredictedRisk"]

    thresholds = [
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
    ]

    results = []

    for threshold in thresholds:
        predictions = (
            probabilities >= threshold
        ).astype(int)

        results.append({
            "Threshold": threshold,
            "Accuracy": accuracy_score(
                y_true, predictions
            ),
            "Precision": precision_score(
                y_true, predictions,
                zero_division=0
            ),
            "Recall": recall_score(
                y_true, predictions,
                zero_division=0
            ),
            "F1": f1_score(
                y_true, predictions,
                zero_division=0
            ),
            "Alerts": int(predictions.sum()),
            "MissedDisruptions": int(
                ((y_true == 1) & (predictions == 0)).sum()
            ),
        })

    results_df = pd.DataFrame(results)

    print("\nTHRESHOLD COMPARISON")
    print(results_df.to_string(index=False))

    results_df.to_csv(
        OUTPUT_DIR / "threshold_comparison.csv",
        index=False,
    )

    chart_data = results_df.melt(
        id_vars="Threshold",
        value_vars=["Precision", "Recall", "F1"],
        var_name="Metric",
        value_name="Score",
    )

    fig = px.line(
        chart_data,
        x="Threshold",
        y="Score",
        color="Metric",
        markers=True,
        title="Disruption Prediction — Threshold Trade-offs",
        template="plotly_dark",
    )

    fig.update_layout(
        title_x=0.5,
        yaxis_range=[0, 1],
    )

    fig.write_html(
        FIGURE_DIR / "model_threshold_analysis.html"
    )

    baseline_accuracy = float((y_true == 0).mean())

    print("\nBASELINE COMPARISON")
    print(
        f"Always-predict-normal accuracy: "
        f"{baseline_accuracy:.2%}"
    )
    print(
        f"Actual disruption prevalence: "
        f"{y_true.mean():.2%}"
    )

    print("\nNOTE")
    print(
        "These are exploratory results on the test set. "
        "Do not select a final threshold using this test set. "
        "Final tuning requires a separate validation period."
    )

    print("\nAERORESILIENCE AI — EVALUATION COMPLETE")


if __name__ == "__main__":
    main()
