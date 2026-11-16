
from pathlib import Path
import shutil

import pandas as pd
import plotly.express as px

import src.optimization.multi_airport_recovery_test as engine

# =====================================================
# AERORESILIENCE AI
# STEP 33 — FULL 24-SCENARIO RECOVERY STRESS TEST
# =====================================================

REPORT_DIR = Path("outputs/reports")
FIGURE_DIR = Path("outputs/figures")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

DATES = [
    "2026-08-12",
    "2026-08-19",
]

DURATIONS = [2, 3, 4]

AIRPORTS = ["ORD", "DFW", "DEN", "ATL"]

BATCH_RESULT = (
    REPORT_DIR / "multi_airport_recovery_results.csv"
)

BATCH_SCHEDULE = (
    REPORT_DIR / "multi_airport_recovery_schedules.csv"
)


def main():

    print("=" * 65)
    print("AERORESILIENCE AI — 24-SCENARIO STRESS TEST")
    print("=" * 65)

    # Preserve existing Step 32 results.
    for source, name in [
        (BATCH_RESULT, "step32_recovery_results.csv"),
        (BATCH_SCHEDULE, "step32_recovery_schedules.csv"),
    ]:
        if source.exists():
            destination = REPORT_DIR / name

            if not destination.exists():
                shutil.copy2(source, destination)

    original_date = engine.SCENARIO_DATE
    original_start = engine.START_HOUR
    original_end = engine.STOP_HOUR
    original_airports = engine.AIRPORTS.copy()

    results = []
    schedules = []

    try:

        engine.AIRPORTS = AIRPORTS
        engine.START_HOUR = 10

        for date in DATES:

            for duration in DURATIONS:

                print("\n" + "=" * 65)
                print(
                    f"TESTING DATE: {date} "
                    f"| DURATION: {duration} HOURS"
                )
                print("=" * 65)

                engine.SCENARIO_DATE = date
                engine.STOP_HOUR = (
                    engine.START_HOUR + duration
                )

                # Reuse our existing aircraft-aware
                # optimizer without changing its
                # objective or operational assumptions.
                engine.main()

                batch = pd.read_csv(BATCH_RESULT)
                batch_schedule = pd.read_csv(BATCH_SCHEDULE)

                if len(batch) != len(AIRPORTS):
                    raise RuntimeError(
                        "Unexpected number of airports "
                        "in scenario batch."
                    )

                if not (
                    batch["ScenarioDate"].eq(date).all()
                ):
                    raise RuntimeError(
                        "Scenario date mismatch."
                    )

                if not (
                    batch["GroundStopHours"].eq(duration).all()
                ):
                    raise RuntimeError(
                        "Scenario duration mismatch."
                    )

                results.append(batch)

                batch_schedule["ScenarioDate"] = date
                batch_schedule["GroundStopHours"] = duration

                schedules.append(batch_schedule)

    finally:

        # Restore the original engine settings.
        engine.SCENARIO_DATE = original_date
        engine.START_HOUR = original_start
        engine.STOP_HOUR = original_end
        engine.AIRPORTS = original_airports

    # =================================================
    # COMBINE SCENARIOS
    # =================================================

    summary = pd.concat(
        results,
        ignore_index=True,
    )

    all_schedules = pd.concat(
        schedules,
        ignore_index=True,
    )

    expected = (
        len(DATES)
        * len(DURATIONS)
        * len(AIRPORTS)
    )

    if len(summary) != expected:
        raise RuntimeError(
            f"Expected {expected} scenarios, "
            f"got {len(summary)}."
        )

    if summary.duplicated(
        ["Airport", "ScenarioDate", "GroundStopHours"]
    ).any():
        raise RuntimeError(
            "Duplicate scenarios detected."
        )

    # Calculate absolute simulated delay savings.
    summary["ModeledMinutesAvoided"] = (
        summary["FCFSIncrementalDelayMinutes"]
        - summary["OptimizedIncrementalDelayMinutes"]
    )

    summary["Improved"] = (
        summary["ModeledMinutesAvoided"] > 0
    )

    # =================================================
    # SAVE COMPLETE RESULTS
    # =================================================

    summary_file = (
        REPORT_DIR / "full_24_scenario_results.csv"
    )

    schedules_file = (
        REPORT_DIR / "full_24_scenario_schedules.csv"
    )

    summary.to_csv(
        summary_file,
        index=False,
    )

    all_schedules.to_csv(
        schedules_file,
        index=False,
    )

    # =================================================
    # SUMMARIZE RESULTS BY AIRPORT
    # =================================================

    airport_summary = (
        summary.groupby("Airport")
        .agg(
            Scenarios=("Improved", "size"),
            ImprovedScenarios=("Improved", "sum"),
            AverageImprovementPercent=(
                "ImprovementPercent", "mean"
            ),
            MinimumImprovementPercent=(
                "ImprovementPercent", "min"
            ),
            MaximumImprovementPercent=(
                "ImprovementPercent", "max"
            ),
            TotalModeledMinutesAvoided=(
                "ModeledMinutesAvoided", "sum"
            ),
        )
        .reset_index()
    )

    airport_summary.to_csv(
        REPORT_DIR / "stress_test_airport_summary.csv",
        index=False,
    )

    # =================================================
    # VISUALIZE ROBUSTNESS
    # =================================================

    fig = px.bar(
        summary,
        x="Airport",
        y="ImprovementPercent",
        color="GroundStopHours",
        facet_col="ScenarioDate",
        barmode="group",
        title=(
            "Aircraft-Aware Recovery — "
            "24-Scenario Stress Test"
        ),
        labels={
            "ImprovementPercent": "Modeled Delay Reduction (%)",
            "GroundStopHours": "Ground Stop (Hours)",
        },
        template="plotly_dark",
    )

    fig.update_layout(title_x=0.5)

    chart_file = (
        FIGURE_DIR / "full_24_scenario_stress_test.html"
    )

    fig.write_html(chart_file)

    # =================================================
    # FINAL RESULTS
    # =================================================

    print("\n" + "=" * 65)
    print("FULL STRESS TEST RESULTS")
    print("=" * 65)

    print(f"Scenarios evaluated: {len(summary)}")
    print(
        "Scenarios showing improvement:",
        int(summary["Improved"].sum())
    )

    print(
        "Average scenario improvement:",
        f"{summary['ImprovementPercent'].mean():.2f}%"
    )

    print(
        "Minimum scenario improvement:",
        f"{summary['ImprovementPercent'].min():.2f}%"
    )

    print(
        "Maximum scenario improvement:",
        f"{summary['ImprovementPercent'].max():.2f}%"
    )

    print("\nAIRPORT ROBUSTNESS SUMMARY")
    print(airport_summary.to_string(index=False))

    print("\nINDIVIDUAL SCENARIO RESULTS")
    print(
        summary[
            [
                "Airport",
                "ScenarioDate",
                "GroundStopHours",
                "RecoveryFlights",
                "ImprovementPercent",
            ]
        ].to_string(index=False)
    )

    print("\nIMPORTANT LIMITATIONS")
    print(
        "All scenarios use hypothetical ground stops "
        "and historical flight schedules."
    )
    print(
        "The same simulator is used for optimization "
        "and evaluation in every scenario."
    )
    print(
        "Each scenario is optimized independently; "
        "this is robustness testing, not a "
        "held-out predictive validation."
    )
    print(
        "Savings across scenarios are not additive "
        "because simulations overlap in time and flights."
    )
    print(
        "Crew, gates, passenger connections, and actual "
        "airport capacity remain unmodeled."
    )

    print(f"\nResults saved: {summary_file}")
    print(f"Schedules saved: {schedules_file}")
    print(f"Chart saved: {chart_file}")

    print(
        "\nAERORESILIENCE AI — "
        "FULL STRESS TEST COMPLETE"
    )


if __name__ == "__main__":
    main()
