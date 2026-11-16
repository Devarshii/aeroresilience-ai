
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

# =====================================================
# AERORESILIENCE AI
# STEP 21 — INTELLIGENT AIRLINE RECOVERY OPTIMIZATION
# =====================================================

REPORT_DIR = Path("outputs/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_FILE = (
    REPORT_DIR / "recovery_baseline_flights.csv"
)

HISTORICAL_FILE = Path(
    "data/processed/flights_historical_2026_06_08.parquet"
)

STOP_END = 13 * 60
SLOT_MINUTES = 15
DEPARTURES_PER_SLOT = 8


def load_data():
    print("\nLoading baseline recovery schedule...")

    flights = pd.read_csv(BASELINE_FILE)

    historical = pd.read_parquet(
        HISTORICAL_FILE,
        columns=["Origin", "Dest"],
    )

    required = [
        "RecoveryFlightID",
        "Origin",
        "Dest",
        "ScheduledMinute",
        "RecoveryDepartureMinute",
        "BaselineDelayMinutes",
    ]

    missing = [
        col for col in required
        if col not in flights.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if flights["RecoveryFlightID"].duplicated().any():
        raise ValueError("Duplicate recovery flight IDs.")

    return flights, historical


def calculate_priority_weights(flights, historical):
    """
    Estimate relative network importance based on
    the number of distinct onward routes available
    from each destination.

    This is an experimental network proxy.
    It is NOT actual passenger connection demand.
    """

    onward_routes = (
        historical.groupby("Origin")["Dest"]
        .nunique()
        .rename("OnwardRouteCount")
        .reset_index()
    )

    flights = flights.merge(
        onward_routes,
        left_on="Dest",
        right_on="Origin",
        how="left",
        suffixes=("", "_network"),
    )

    if "Origin_network" in flights.columns:
        flights = flights.drop(columns=["Origin_network"])

    flights["OnwardRouteCount"] = (
        flights["OnwardRouteCount"]
        .fillna(0)
        .astype(int)
    )

    # Larger destination networks receive more
    # importance in the experimental objective.
    flights["PriorityWeight"] = (
        1.0
        + np.minimum(
            flights["OnwardRouteCount"], 150
        ) / 25.0
    )

    return flights


def optimize_schedule(flights):
    print("\nRunning recovery assignment optimization...")

    n = len(flights)

    if n == 0:
        raise ValueError("No recovery flights available.")

    # Use exactly the same departure positions
    # as the FCFS baseline for fair comparison.
    slots = np.sort(
        flights["RecoveryDepartureMinute"]
        .to_numpy(dtype=float)
    )

    scheduled = flights[
        "ScheduledMinute"
    ].to_numpy(dtype=float)

    weights = flights[
        "PriorityWeight"
    ].to_numpy(dtype=float)

    # Every assignment must respect each flight's
    # scheduled departure and the ground-stop end.
    delay_matrix = np.maximum(
        slots[None, :] - scheduled[:, None],
        0.0,
    )

    cost_matrix = (
        delay_matrix * weights[:, None]
    )

    # Solve minimum total weighted delay.
    flight_indices, slot_indices = (
        linear_sum_assignment(cost_matrix)
    )

    result = flights.copy()

    assigned_slots = np.empty(n, dtype=float)
    assigned_slots[flight_indices] = slots[slot_indices]

    result["OptimizedDepartureMinute"] = (
        assigned_slots.astype(int)
    )

    result["OptimizedDelayMinutes"] = (
        result["OptimizedDepartureMinute"]
        - result["ScheduledMinute"]
    )

    # ---------------------------------------------
    # Feasibility checks
    # ---------------------------------------------

    if not (
        result["OptimizedDepartureMinute"] >= STOP_END
    ).all():
        raise RuntimeError(
            "Assignment violates ground-stop restriction."
        )

    if not (
        result["OptimizedDelayMinutes"] >= 0
    ).all():
        raise RuntimeError(
            "Assignment contains negative delays."
        )

    baseline_counts = (
        flights["RecoveryDepartureMinute"]
        .value_counts().sort_index()
    )

    optimized_counts = (
        result["OptimizedDepartureMinute"]
        .value_counts().sort_index()
    )

    if not baseline_counts.equals(optimized_counts):
        raise RuntimeError(
            "Optimized schedule changed available slot capacity."
        )

    if (
        optimized_counts > DEPARTURES_PER_SLOT
    ).any():
        raise RuntimeError(
            "Recovery slot capacity exceeded."
        )

    return result


def main():
    print("=" * 65)
    print("AERORESILIENCE AI — RECOVERY OPTIMIZATION")
    print("=" * 65)

    flights, historical = load_data()

    flights = calculate_priority_weights(
        flights, historical
    )

    optimized = optimize_schedule(flights)

    # ---------------------------------------------
    # Compare FCFS and optimized policies
    # ---------------------------------------------

    weights = optimized[
        "PriorityWeight"
    ].to_numpy()

    baseline_delays = optimized[
        "BaselineDelayMinutes"
    ].to_numpy()

    optimized_delays = optimized[
        "OptimizedDelayMinutes"
    ].to_numpy()

    baseline_weighted = float(
        np.sum(weights * baseline_delays)
    )

    optimized_weighted = float(
        np.sum(weights * optimized_delays)
    )

    baseline_total = float(
        baseline_delays.sum()
    )

    optimized_total = float(
        optimized_delays.sum()
    )

    improvement = (
        (baseline_weighted - optimized_weighted)
        / baseline_weighted * 100
        if baseline_weighted > 0 else 0.0
    )

    summary = pd.DataFrame([
        {
            "Policy": "FCFS baseline",
            "Flights": len(optimized),
            "TotalDelayMinutes": baseline_total,
            "WeightedDelayScore": baseline_weighted,
            "AverageDelayMinutes": float(
                baseline_delays.mean()
            ),
            "MaximumDelayMinutes": float(
                baseline_delays.max()
            ),
        },
        {
            "Policy": "Network-priority optimized",
            "Flights": len(optimized),
            "TotalDelayMinutes": optimized_total,
            "WeightedDelayScore": optimized_weighted,
            "AverageDelayMinutes": float(
                optimized_delays.mean()
            ),
            "MaximumDelayMinutes": float(
                optimized_delays.max()
            ),
        },
    ])

    optimized["DelayChangeMinutes"] = (
        optimized["OptimizedDelayMinutes"]
        - optimized["BaselineDelayMinutes"]
    )

    # ---------------------------------------------
    # Save reports
    # ---------------------------------------------

    optimized.to_csv(
        REPORT_DIR / "optimized_recovery_flights.csv",
        index=False,
    )

    summary.to_csv(
        REPORT_DIR / "recovery_optimization_comparison.csv",
        index=False,
    )

    # ---------------------------------------------
    # Terminal summary
    # ---------------------------------------------

    print("\nRECOVERY POLICY COMPARISON")
    print(summary.to_string(index=False))

    print("\nOPTIMIZATION RESULTS")
    print(f"Flights optimized: {len(optimized):,}")

    print(
        f"Baseline weighted delay score: "
        f"{baseline_weighted:,.2f}"
    )

    print(
        f"Optimized weighted delay score: "
        f"{optimized_weighted:,.2f}"
    )

    print(
        f"Weighted delay improvement: "
        f"{improvement:.2f}%"
    )

    print(
        f"Flights scheduled earlier: "
        f"{(optimized['DelayChangeMinutes'] < 0).sum():,}"
    )

    print(
        f"Flights scheduled later: "
        f"{(optimized['DelayChangeMinutes'] > 0).sum():,}"
    )

    print(
        f"Flights unchanged: "
        f"{(optimized['DelayChangeMinutes'] == 0).sum():,}"
    )

    print("\nIMPORTANT LIMITATIONS")
    print(
        "Priority weights are derived from historical "
        "onward route counts, not actual passenger bookings."
    )
    print(
        "The optimization uses exactly the same departure "
        "slots as the FCFS baseline."
    )
    print(
        "Crew, aircraft, gate, runway interactions, "
        "passenger connections, and existing traffic "
        "after the ground stop are not modeled."
    )
    print(
        "Weighted delay improvement does not represent "
        "verified passenger time savings or financial savings."
    )

    print("\nOptimization reports saved successfully.")

    print(
        "\nAERORESILIENCE AI — RECOVERY OPTIMIZATION COMPLETE"
    )


if __name__ == "__main__":
    main()
