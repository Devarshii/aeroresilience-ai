
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

import src.optimization.aircraft_aware_recovery as engine

# =====================================================
# AERORESILIENCE AI
# STEP 30 — RECOVERY ROBUSTNESS & SENSITIVITY
# =====================================================

REPORT_DIR = Path("outputs/reports")
FIGURE_DIR = Path("outputs/figures")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

SCHEDULE_FILE = (
    REPORT_DIR / "aircraft_aware_optimized_schedule.csv"
)

TURNAROUND_SCENARIOS = [10, 20, 30, 40]

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


def load_schedules():

    print("\nLoading aircraft schedules...")

    df = engine.create_utc_schedule(
        engine.load_flights()
    )

    optimized = pd.read_csv(SCHEDULE_FILE)

    for frame in [df, optimized]:

        frame["FlightDate"] = pd.to_datetime(
            frame["FlightDate"]
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
            "CRSDepTime",
            "Flight_Number_Reporting_Airline",
        ]:
            frame[col] = pd.to_numeric(
                frame[col], errors="coerce"
            )

    if optimized.duplicated(KEYS).any():
        raise ValueError("Duplicate optimized flight keys.")

    df = df.merge(
        optimized[
            KEYS + ["AircraftAwareDelayMinutes"]
        ],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )

    seed_mask = df["RecoverySeed"].fillna(False)

    if (
        df.loc[seed_mask, "AircraftAwareDelayMinutes"]
        .isna().any()
    ):
        raise ValueError(
            "Some recovery flights were not matched."
        )

    if int(seed_mask.sum()) != len(optimized):
        raise ValueError(
            "Unexpected number of recovery matches."
        )

    df["AircraftAwareDelayMinutes"] = (
        df["AircraftAwareDelayMinutes"].fillna(0.0)
    )

    df["NoStopDelay"] = 0.0

    return df


def main():

    print("=" * 65)
    print("AERORESILIENCE AI — RECOVERY SENSITIVITY ANALYSIS")
    print("=" * 65)

    df = load_schedules()

    results = []

    original_turn = engine.MIN_TURN

    try:

        for minimum_turn in TURNAROUND_SCENARIOS:

            print(
                f"\nTesting minimum turnaround: "
                f"{minimum_turn} minutes"
            )

            # Use the same simulation engine while
            # changing only the turnaround assumption.
            engine.MIN_TURN = minimum_turn

            no_stop = engine.simulate_policy(
                df,
                "No Stop",
                "NoStopDelay",
            )

            no_stop_total = float(
                no_stop["TotalDelay"].sum()
            )

            for policy, column in POLICIES.items():

                simulated = engine.simulate_policy(
                    df, policy, column
                )

                total_delay = float(
                    simulated["TotalDelay"].sum()
                )

                incremental = (
                    total_delay - no_stop_total
                )

                results.append({
                    "MinimumTurnMinutes": minimum_turn,
                    "Policy": policy,
                    "NoStopDelayMinutes": no_stop_total,
                    "TotalModeledDelayMinutes": total_delay,
                    "IncrementalDelayMinutes": incremental,
                })

    finally:
        # Restore the project's normal assumption.
        engine.MIN_TURN = original_turn

    summary = pd.DataFrame(results)

    # ---------------------------------------------
    # COMPARE EVERY POLICY AGAINST FCFS
    # ---------------------------------------------

    baseline = (
        summary[summary["Policy"] == "FCFS"][
            ["MinimumTurnMinutes", "IncrementalDelayMinutes"]
        ]
        .rename(
            columns={
                "IncrementalDelayMinutes": "FCFSIncrementalDelay"
            }
        )
    )

    summary = summary.merge(
        baseline,
        on="MinimumTurnMinutes",
        how="left",
        validate="many_to_one",
    )

    summary["ImprovementVsFCFSPercent"] = np.where(
        summary["FCFSIncrementalDelay"] > 0,
        100 * (
            summary["FCFSIncrementalDelay"]
            - summary["IncrementalDelayMinutes"]
        ) / summary["FCFSIncrementalDelay"],
        np.nan,
    )

    # Verify that the original 20-minute scenario
    # reproduces our previous Step 29 results.
    original = summary[
        summary["MinimumTurnMinutes"] == 20
    ].set_index("Policy")

    expected = {
        "FCFS": 103710.0,
        "Aircraft Aware": 96092.0,
    }

    for policy, expected_delay in expected.items():

        actual = float(
            original.loc[
                policy, "IncrementalDelayMinutes"
            ]
        )

        if not np.isclose(
            actual, expected_delay, atol=0.01
        ):
            raise RuntimeError(
                f"Reproduction check failed for {policy}: "
                f"expected {expected_delay}, got {actual}"
            )

    # ---------------------------------------------
    # SAVE REPORT
    # ---------------------------------------------

    report_file = (
        REPORT_DIR / "recovery_sensitivity_comparison.csv"
    )

    summary.to_csv(report_file, index=False)

    # ---------------------------------------------
    # INTERACTIVE VISUALIZATION
    # ---------------------------------------------

    fig = px.line(
        summary,
        x="MinimumTurnMinutes",
        y="IncrementalDelayMinutes",
        color="Policy",
        markers=True,
        title="Aircraft Recovery Sensitivity to Turnaround Time",
        labels={
            "MinimumTurnMinutes":
                "Minimum Turnaround Assumption (Minutes)",
            "IncrementalDelayMinutes":
                "Incremental Modeled Delay Minutes",
        },
        template="plotly_dark",
    )

    fig.update_layout(title_x=0.5)

    chart_file = (
        FIGURE_DIR / "recovery_sensitivity_analysis.html"
    )

    fig.write_html(chart_file)

    # ---------------------------------------------
    # DISPLAY RESULTS
    # ---------------------------------------------

    print("\n" + "=" * 65)
    print("RECOVERY ROBUSTNESS COMPARISON")
    print("=" * 65)

    print(
        summary[
            [
                "MinimumTurnMinutes",
                "Policy",
                "IncrementalDelayMinutes",
                "ImprovementVsFCFSPercent",
            ]
        ].to_string(index=False)
    )

    aircraft_aware = summary[
        summary["Policy"] == "Aircraft Aware"
    ]

    print("\nAIRCRAFT-AWARE IMPROVEMENT BY SCENARIO")

    print(
        aircraft_aware[
            [
                "MinimumTurnMinutes",
                "ImprovementVsFCFSPercent",
            ]
        ].to_string(index=False)
    )

    print("\nIMPORTANT LIMITATIONS")
    print(
        "The same recovery assignments are evaluated "
        "under different turnaround assumptions."
    )
    print(
        "These are sensitivity tests, not independently "
        "optimized or held-out scenarios."
    )
    print(
        "Real aircraft-specific turnaround restrictions, "
        "crew, gates, and airport capacity are not included."
    )

    print(f"\nReport saved: {report_file}")
    print(f"Chart saved: {chart_file}")

    print(
        "\nAERORESILIENCE AI — SENSITIVITY ANALYSIS COMPLETE"
    )


if __name__ == "__main__":
    main()
