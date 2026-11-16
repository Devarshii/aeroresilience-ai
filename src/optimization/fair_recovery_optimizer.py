
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

# =====================================================
# AERORESILIENCE AI
# STEP 23 — FAIRNESS-CONSTRAINED RECOVERY OPTIMIZER
# =====================================================

REPORT_DIR = Path("outputs/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = REPORT_DIR / "optimized_recovery_flights.csv"

MAX_ADDITIONAL_DELAY = 60
STOP_END = 13 * 60
DEPARTURES_PER_SLOT = 8


def load_data():
    print("\nLoading recovery optimization results...")

    df = pd.read_csv(INPUT_FILE)

    required = [
        "RecoveryFlightID",
        "Reporting_Airline",
        "ScheduledMinute",
        "RecoveryDepartureMinute",
        "BaselineDelayMinutes",
        "OptimizedDepartureMinute",
        "OptimizedDelayMinutes",
        "PriorityWeight",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df["RecoveryFlightID"].duplicated().any():
        raise ValueError("Duplicate recovery flight IDs.")

    if df[required].isna().any().any():
        raise ValueError("Required fields contain missing values.")

    return df


def optimize_with_fairness(df):
    print("\nRunning fairness-constrained optimization...")

    n = len(df)

    if n == 0:
        raise ValueError("No flights to optimize.")

    slots = np.sort(
        df["RecoveryDepartureMinute"].to_numpy(
            dtype=float
        )
    )

    scheduled = df[
        "ScheduledMinute"
    ].to_numpy(dtype=float)

    baseline_delay = df[
        "BaselineDelayMinutes"
    ].to_numpy(dtype=float)

    weights = df[
        "PriorityWeight"
    ].to_numpy(dtype=float)

    baseline_slots = df[
        "RecoveryDepartureMinute"
    ].to_numpy(dtype=float)

    if not np.all(
        baseline_slots - scheduled >= -1e-9
    ):
        raise ValueError("Invalid baseline schedule.")

    # Candidate delay for each flight-slot assignment.
    delay_matrix = (
        slots[None, :] - scheduled[:, None]
    )

    # A valid assignment must meet:
    # 1. No departure before ground-stop end.
    # 2. No departure before the original schedule.
    # 3. No more than 60 additional minutes versus FCFS.
    feasible = (
        (slots[None, :] >= STOP_END)
        & (delay_matrix >= 0)
        & (
            delay_matrix
            <= baseline_delay[:, None]
            + MAX_ADDITIONAL_DELAY
        )
    )

    # Minimize weighted delay across feasible assignments.
    weighted_cost = (
        delay_matrix * weights[:, None]
    )

    # A sufficiently large penalty makes forbidden
    # assignments more expensive than any feasible
    # complete schedule.
    max_feasible_cost = float(
        np.max(weighted_cost[feasible])
    )

    big_m = (
        n * max(max_feasible_cost, 1.0) + 1.0
    )

    cost_matrix = np.where(
        feasible,
        weighted_cost,
        big_m
    )

    row_ind, col_ind = linear_sum_assignment(
        cost_matrix
    )

    if not feasible[row_ind, col_ind].all():
        raise RuntimeError(
            "No feasible fair recovery assignment found."
        )

    assigned = np.empty(n, dtype=float)
    assigned[row_ind] = slots[col_ind]

    result = df.copy()

    result["FairDepartureMinute"] = assigned.astype(int)

    result["FairDelayMinutes"] = (
        result["FairDepartureMinute"]
        - result["ScheduledMinute"]
    )

    result["AdditionalDelayVsFCFS"] = (
        result["FairDelayMinutes"]
        - result["BaselineDelayMinutes"]
    )

    # ---------------------------------------------
    # Feasibility validation
    # ---------------------------------------------

    assert (
        result["AdditionalDelayVsFCFS"]
        <= MAX_ADDITIONAL_DELAY
    ).all()

    assert (
        result["FairDelayMinutes"] >= 0
    ).all()

    assert (
        result["FairDepartureMinute"] >= STOP_END
    ).all()

    original_counts = (
        result["RecoveryDepartureMinute"]
        .value_counts().sort_index()
    )

    fair_counts = (
        result["FairDepartureMinute"]
        .value_counts().sort_index()
    )

    assert original_counts.equals(fair_counts)

    assert (
        fair_counts <= DEPARTURES_PER_SLOT
    ).all()

    return result


def calculate_metrics(df, delay_col, policy):
    delay = df[delay_col].to_numpy(dtype=float)
    weights = df["PriorityWeight"].to_numpy(dtype=float)

    return {
        "Policy": policy,
        "Flights": len(df),
        "TotalDelayMinutes": float(delay.sum()),
        "AverageDelayMinutes": float(delay.mean()),
        "MaximumDelayMinutes": float(delay.max()),
        "P90DelayMinutes": float(np.percentile(delay, 90)),
        "WeightedDelayScore": float(
            np.sum(delay * weights)
        ),
    }


def main():
    print("=" * 65)
    print("AERORESILIENCE AI — FAIR RECOVERY OPTIMIZATION")
    print("=" * 65)

    df = load_data()
    result = optimize_with_fairness(df)

    # ---------------------------------------------
    # Compare all three recovery policies
    # ---------------------------------------------

    summary = pd.DataFrame([
        calculate_metrics(
            result,
            "BaselineDelayMinutes",
            "FCFS baseline",
        ),
        calculate_metrics(
            result,
            "OptimizedDelayMinutes",
            "Unconstrained optimization",
        ),
        calculate_metrics(
            result,
            "FairDelayMinutes",
            "Fairness-constrained optimization",
        ),
    ])

    baseline_score = summary.iloc[0][
        "WeightedDelayScore"
    ]

    fair_score = summary.iloc[2][
        "WeightedDelayScore"
    ]

    improvement = (
        100 * (baseline_score - fair_score)
        / baseline_score
        if baseline_score > 0 else 0
    )

    # ---------------------------------------------
    # Airline-level impact
    # ---------------------------------------------

    airline_summary = (
        result.groupby("Reporting_Airline")
        .agg(
            Flights=("RecoveryFlightID", "size"),
            AverageBaselineDelay=(
                "BaselineDelayMinutes", "mean"
            ),
            AverageFairDelay=(
                "FairDelayMinutes", "mean"
            ),
            MaximumAddedDelay=(
                "AdditionalDelayVsFCFS", "max"
            ),
        )
        .reset_index()
    )

    airline_summary["AverageDelayChange"] = (
        airline_summary["AverageFairDelay"]
        - airline_summary["AverageBaselineDelay"]
    )

    # ---------------------------------------------
    # Save reports
    # ---------------------------------------------

    result.to_csv(
        REPORT_DIR / "fair_recovery_flights.csv",
        index=False,
    )

    summary.to_csv(
        REPORT_DIR / "fair_recovery_comparison.csv",
        index=False,
    )

    airline_summary.to_csv(
        REPORT_DIR / "fair_recovery_airline_impact.csv",
        index=False,
    )

    # ---------------------------------------------
    # Display results
    # ---------------------------------------------

    print("\nRECOVERY POLICY COMPARISON")
    print(summary.to_string(index=False))

    print("\nFAIRNESS-CONSTRAINED RESULTS")

    print(
        f"Weighted delay improvement vs FCFS: "
        f"{improvement:.2f}%"
    )

    print(
        f"Maximum permitted added delay: "
        f"{MAX_ADDITIONAL_DELAY} minutes"
    )

    print(
        "Actual maximum added delay:",
        f"{result['AdditionalDelayVsFCFS'].max():.0f} minutes"
    )

    print(
        "Flights delayed further than FCFS:",
        int((result["AdditionalDelayVsFCFS"] > 0).sum())
    )

    print(
        "Flights departing earlier than FCFS:",
        int((result["AdditionalDelayVsFCFS"] < 0).sum())
    )

    print("\nAIRLINE-LEVEL FAIRNESS")
    print(airline_summary.to_string(index=False))

    print("\nIMPORTANT LIMITATIONS")
    print(
        "The fairness rule limits additional delays "
        "relative to the hypothetical FCFS schedule."
    )
    print(
        "It does not guarantee equal outcomes across airlines."
    )
    print(
        "The model still uses simulated departure capacity "
        "and proxy network-priority weights."
    )
    print(
        "Aircraft and crew feasibility constraints "
        "remain outside this optimization."
    )

    print("\nReports saved successfully.")

    print(
        "\nAERORESILIENCE AI — FAIR OPTIMIZATION COMPLETE"
    )


if __name__ == "__main__":
    main()
