
from pathlib import Path

import numpy as np
import pandas as pd

from src.optimization.aircraft_aware_recovery import (
    load_flights,
    create_utc_schedule,
    simulate_policy,
    POLICIES,
)

# =====================================================
# AERORESILIENCE AI
# STEP 27 — INCREMENTAL GROUND-STOP IMPACT
# =====================================================

REPORT_DIR = Path("outputs/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

FLIGHT_KEYS = [
    "Airline",
    "TailNumber",
    "FlightDate",
    "Origin",
    "Dest",
    "ScheduledDepUTC",
]


def main():
    print("=" * 65)
    print("AERORESILIENCE AI — INCREMENTAL RECOVERY ANALYSIS")
    print("=" * 65)

    # Use exactly the same aircraft schedules
    # and model assumptions for every comparison.
    flights = load_flights()
    flights = create_utc_schedule(flights)

    # ---------------------------------------------
    # 1. No-ground-stop counterfactual
    # ---------------------------------------------

    flights["NoStopDelay"] = 0.0

    print("\nSimulating no-ground-stop baseline...")

    no_stop = simulate_policy(
        flights,
        "No Ground Stop",
        "NoStopDelay",
    )

    if no_stop.duplicated(FLIGHT_KEYS).any():
        raise ValueError(
            "Duplicate flight identifiers in baseline."
        )

    baseline = no_stop[
        FLIGHT_KEYS + ["TotalDelay"]
    ].rename(
        columns={
            "TotalDelay": "NoStopDelayMinutes"
        }
    )

    print(
        "No-stop modeled delay minutes:",
        f"{baseline['NoStopDelayMinutes'].sum():,.0f}",
    )

    # ---------------------------------------------
    # 2. Compare each recovery strategy
    # ---------------------------------------------

    summaries = []
    details = []

    for policy, delay_column in POLICIES.items():
        print(f"\nEvaluating {policy}...")

        simulated = simulate_policy(
            flights,
            policy,
            delay_column,
        )

        if simulated.duplicated(FLIGHT_KEYS).any():
            raise ValueError(
                f"Duplicate flight identifiers: {policy}"
            )

        merged = simulated.merge(
            baseline,
            on=FLIGHT_KEYS,
            how="left",
            validate="one_to_one",
        )

        if merged["NoStopDelayMinutes"].isna().any():
            raise ValueError(
                "No-stop comparison contains unmatched flights."
            )

        # Isolate additional delay caused by recovery
        # relative to the same model without a shutdown.
        merged["IncrementalDelayMinutes"] = (
            merged["TotalDelay"]
            - merged["NoStopDelayMinutes"]
        )

        if (
            merged["IncrementalDelayMinutes"] < -1e-6
        ).any():
            raise RuntimeError(
                "Negative incremental delay detected. "
                "Review simulation assumptions."
            )

        merged["IncrementalDelayMinutes"] = (
            merged["IncrementalDelayMinutes"].clip(lower=0)
        )

        additional = (
            merged["IncrementalDelayMinutes"] > 0
        )

        downstream = (
            additional & ~merged["RecoverySeed"]
        )

        summaries.append({
            "Policy": policy,
            "OperatingFlights": len(merged),
            "NoStopDelayMinutes": float(
                merged["NoStopDelayMinutes"].sum()
            ),
            "RecoveryTotalDelayMinutes": float(
                merged["TotalDelay"].sum()
            ),
            "IncrementalDelayMinutes": float(
                merged["IncrementalDelayMinutes"].sum()
            ),
            "IncrementallyDelayedFlights": int(
                additional.sum()
            ),
            "DownstreamIncrementallyDelayedFlights": int(
                downstream.sum()
            ),
            "DownstreamIncrementalDelayMinutes": float(
                merged.loc[
                    downstream, "IncrementalDelayMinutes"
                ].sum()
            ),
        })

        details.append(merged)

    summary = pd.DataFrame(summaries)

    fcfs_delay = float(
        summary.loc[
            summary["Policy"] == "FCFS",
            "IncrementalDelayMinutes",
        ].iloc[0]
    )

    if fcfs_delay > 0:
        summary["ImprovementVsFCFSPercent"] = (
            100
            * (
                fcfs_delay
                - summary["IncrementalDelayMinutes"]
            )
            / fcfs_delay
        )
    else:
        summary["ImprovementVsFCFSPercent"] = np.nan

    # ---------------------------------------------
    # 3. Save the corrected comparisons
    # ---------------------------------------------

    summary.to_csv(
        REPORT_DIR / "incremental_recovery_comparison.csv",
        index=False,
    )

    pd.concat(
        details,
        ignore_index=True,
    ).to_parquet(
        REPORT_DIR / "incremental_recovery_details.parquet",
        index=False,
    )

    print("\nINCREMENTAL GROUND-STOP IMPACT")
    print(summary.to_string(index=False))

    print("\nINTERPRETATION")
    print(
        "Incremental delay = simulated recovery delay "
        "minus simulated no-stop delay."
    )
    print(
        "All policies use identical aircraft schedules "
        "and turnaround assumptions."
    )
    print(
        "The results are counterfactual estimates, "
        "not observed airline delays."
    )
    print(
        "This model still excludes crew, airport capacity, "
        "aircraft substitutions, and passenger rebooking."
    )

    print("\nReports saved successfully.")

    print(
        "\nAERORESILIENCE AI — "
        "INCREMENTAL ANALYSIS COMPLETE"
    )


if __name__ == "__main__":
    main()
