
from pathlib import Path

import numpy as np
import pandas as pd

from src.optimization.aircraft_aware_recovery import (
    load_flights,
    create_utc_schedule,
    simulate_policy,
)

# =====================================================
# AERORESILIENCE AI
# STEP 29 — NETWORK RECOVERY VALIDATION
# =====================================================

REPORT_DIR = Path("outputs/reports")

SCHEDULE_FILE = (
    REPORT_DIR / "aircraft_aware_optimized_schedule.csv"
)

KEYS = [
    "FlightDate",
    "Reporting_Airline",
    "Tail_Number",
    "Flight_Number_Reporting_Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
]

POLICIES = {
    "FCFS": "BaselineDelayMinutes",
    "Network Priority": "OptimizedDelayMinutes",
    "Fair": "FairDelayMinutes",
    "Aircraft Aware": "AircraftAwareDelayMinutes",
}


def main():

    print("=" * 65)
    print("AERORESILIENCE AI — NETWORK RECOVERY VALIDATION")
    print("=" * 65)

    if not SCHEDULE_FILE.exists():
        raise FileNotFoundError(SCHEDULE_FILE)

    print("\nLoading recovery schedules...")

    df = create_utc_schedule(load_flights())
    optimized = pd.read_csv(SCHEDULE_FILE)

    for frame in (df, optimized):
        frame["FlightDate"] = pd.to_datetime(
            frame["FlightDate"], errors="coerce"
        )

        for col in [
            "Reporting_Airline",
            "Tail_Number",
            "Origin",
            "Dest",
        ]:
            frame[col] = (
                frame[col].astype("string").str.strip()
            )

        for col in [
            "Flight_Number_Reporting_Airline",
            "CRSDepTime",
        ]:
            frame[col] = pd.to_numeric(
                frame[col], errors="coerce"
            )

    if optimized.duplicated(KEYS).any():
        raise ValueError("Duplicate optimized flight keys.")

    if optimized["AircraftAwareDelayMinutes"].isna().any():
        raise ValueError("Missing optimized delays.")

    # Add new optimizer decisions to the complete
    # aircraft schedule, matching by flight identity.
    df = df.merge(
        optimized[
            KEYS + ["AircraftAwareDelayMinutes"]
        ],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )

    seeds = df["RecoverySeed"].fillna(False)

    if df.loc[seeds, "AircraftAwareDelayMinutes"].isna().any():
        raise ValueError(
            "Some recovered flights did not match."
        )

    if int(seeds.sum()) != len(optimized):
        raise ValueError(
            "Mismatch in number of recovered flights."
        )

    df["AircraftAwareDelayMinutes"] = (
        df["AircraftAwareDelayMinutes"].fillna(0.0)
    )

    if (df["AircraftAwareDelayMinutes"] < 0).any():
        raise ValueError("Negative optimized delay detected.")

    print(f"Recovery flights matched: {int(seeds.sum()):,}")

    # -------------------------------------------------
    # 1. NO-GROUND-STOP COUNTERFACTUAL
    # -------------------------------------------------

    print("\nEvaluating no-ground-stop baseline...")

    df["NoStopDelay"] = 0.0

    no_stop = simulate_policy(
        df, "No Stop", "NoStopDelay"
    )

    no_stop_total = float(
        no_stop["TotalDelay"].sum()
    )

    print(
        f"No-stop modeled delay: {no_stop_total:,.0f}"
    )

    # -------------------------------------------------
    # 2. EVALUATE ALL FOUR POLICIES
    # -------------------------------------------------

    results = []

    for name, column in POLICIES.items():

        print(f"\nEvaluating {name}...")

        simulated = simulate_policy(
            df, name, column
        )

        total = float(simulated["TotalDelay"].sum())

        incremental = total - no_stop_total

        if incremental < -1e-6:
            raise RuntimeError(
                f"Negative incremental delay: {name}"
            )

        results.append({
            "Policy": name,
            "TotalModeledDelayMinutes": total,
            "NoStopDelayMinutes": no_stop_total,
            "IncrementalDelayMinutes": max(
                incremental, 0.0
            ),
            "DelayedFlights": int(
                (simulated["TotalDelay"] > 0).sum()
            ),
            "DownstreamDelayedFlights": int(
                (
                    (simulated["TotalDelay"] > 0)
                    & ~simulated["RecoverySeed"]
                ).sum()
            ),
        })

    comparison = pd.DataFrame(results)

    fcfs = float(
        comparison.loc[
            comparison["Policy"] == "FCFS",
            "IncrementalDelayMinutes"
        ].iloc[0]
    )

    comparison["IncrementalImprovementPercent"] = (
        100
        * (
            fcfs - comparison["IncrementalDelayMinutes"]
        )
        / fcfs
        if fcfs > 0 else np.nan
    )

    # -------------------------------------------------
    # 3. SAVE RESULTS
    # -------------------------------------------------

    output_file = (
        REPORT_DIR
        / "validated_network_recovery_comparison.csv"
    )

    comparison.to_csv(output_file, index=False)

    print("\n" + "=" * 65)
    print("VALIDATED POLICY COMPARISON")
    print("=" * 65)

    print(comparison.to_string(index=False))

    print("\nIMPORTANT")
    print(
        "This validates counterfactual consistency, "
        "not real-world airline performance."
    )
    print(
        "The optimizer was developed against this "
        "same simulator, so independent scenario "
        "testing is still required."
    )

    print(f"\nReport saved: {output_file}")

    print(
        "\nAERORESILIENCE AI — "
        "NETWORK RECOVERY VALIDATION COMPLETE"
    )


if __name__ == "__main__":
    main()
