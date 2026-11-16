
from pathlib import Path
import numpy as np
import pandas as pd

# =====================================================
# AERORESILIENCE AI
# STEP 12 — DAILY TOP-K FLIGHT RISK PRIORITIZATION
# =====================================================

INPUT_FILE = Path(
    "outputs/reports/historical_predictions.parquet"
)

REPORT_DIR = Path("outputs/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CAPACITY = 0.20


def main():
    print("=" * 65)
    print("AERORESILIENCE AI — DAILY TOP-K ALERT PRIORITIZATION")
    print("=" * 65)

    df = pd.read_parquet(INPUT_FILE)

    required = [
        "FlightDate",
        "Reporting_Airline",
        "Origin",
        "Dest",
        "Disrupted",
        "PredictedRisk",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["FlightDate"] = pd.to_datetime(df["FlightDate"])
    df["Disrupted"] = df["Disrupted"].astype(int)

    if df["PredictedRisk"].isna().any():
        raise ValueError("PredictedRisk contains missing values.")

    if not df["PredictedRisk"].between(0, 1).all():
        raise ValueError("PredictedRisk must be between 0 and 1.")

    # Assign a stable identifier before sorting.
    df["FlightRecordID"] = np.arange(len(df))

    # Rank flights independently for each scheduled day.
    # The model uses only predeparture schedule features.
    df = df.sort_values(
        ["FlightDate", "PredictedRisk", "FlightRecordID"],
        ascending=[True, False, True],
        kind="stable",
    ).copy()

    df["DailyRank"] = (
        df.groupby("FlightDate").cumcount() + 1
    )

    df["DailyFlights"] = (
        df.groupby("FlightDate")["FlightRecordID"]
        .transform("size")
    )

    # Floor guarantees that selected alerts never exceed
    # 20% of scheduled flights for any individual day.
    df["DailyAlertCapacity"] = np.floor(
        df["DailyFlights"] * CAPACITY
    ).astype(int)

    df["PriorityAlert"] = (
        df["DailyRank"] <= df["DailyAlertCapacity"]
    ).astype(int)

    actual = df["Disrupted"].to_numpy()
    alerts = df["PriorityAlert"].to_numpy()

    tp = int(((actual == 1) & (alerts == 1)).sum())
    fp = int(((actual == 0) & (alerts == 1)).sum())
    fn = int(((actual == 1) & (alerts == 0)).sum())
    tn = int(((actual == 0) & (alerts == 0)).sum())

    total = len(df)
    selected = tp + fp
    disrupted = tp + fn

    precision = tp / selected if selected else 0
    recall = tp / disrupted if disrupted else 0

    print("\nAUGUST 2026 — DAILY PRIORITIZATION")
    print(f"Total flights: {total:,}")
    print(f"Daily alert capacity: {CAPACITY:.0%}")
    print(f"Selected alerts: {selected:,}")
    print(f"Actual alert rate: {selected / total:.2%}")

    print("\nALERT PERFORMANCE")
    print(f"True positives: {tp:,}")
    print(f"False positives: {fp:,}")
    print(f"False negatives: {fn:,}")
    print(f"True negatives: {tn:,}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")

    # Daily results for an operational dashboard.
    daily = (
        df.groupby("FlightDate")
        .agg(
            TotalFlights=("Disrupted", "size"),
            DisruptedFlights=("Disrupted", "sum"),
            SelectedAlerts=("PriorityAlert", "sum"),
        )
        .reset_index()
    )

    daily["AlertRate"] = (
        daily["SelectedAlerts"] / daily["TotalFlights"]
    )

    assert (daily["AlertRate"] <= CAPACITY + 1e-12).all()

    # Actual outcomes are retained for retrospective evaluation,
    # not as inputs into the priority-ranking algorithm.
    df.to_parquet(
        REPORT_DIR / "daily_prioritized_flights.parquet",
        index=False,
    )

    daily.to_csv(
        REPORT_DIR / "daily_alert_capacity_summary.csv",
        index=False,
    )

    pd.DataFrame([{
        "Capacity": CAPACITY,
        "TotalFlights": total,
        "SelectedAlerts": selected,
        "ActualAlertRate": selected / total,
        "Precision": precision,
        "Recall": recall,
        "TruePositives": tp,
        "FalsePositives": fp,
        "FalseNegatives": fn,
        "TrueNegatives": tn,
    }]).to_csv(
        REPORT_DIR / "topk_alert_metrics.csv",
        index=False,
    )

    print("\nReports saved successfully.")
    print(
        "\nNOTE: Exploratory August analysis; "
        "not a new independent test."
    )
    print(
        "\nAERORESILIENCE AI — TOP-K PRIORITIZATION COMPLETE"
    )


if __name__ == "__main__":
    main()
