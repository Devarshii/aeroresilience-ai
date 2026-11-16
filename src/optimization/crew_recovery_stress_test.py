
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px

from src.optimization.integrated_aircraft_crew_recovery import (
    load_inputs,
    evaluate_policy,
    POLICIES,
)

# =====================================================
# AERORESILIENCE AI
# STEP 37 — CREW RECOVERY STRESS TEST
# =====================================================

REPORT_DIR = Path("outputs/reports")
FIGURE_DIR = Path("outputs/figures")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

SCENARIOS = [
    "Normal availability",
    "50% reserve availability",
    "Reserves arrive 60 min late",
    "Reserve duty reduced by 60 min",
]


def prepare_scenario(crews, scenario):
    """
    Modify synthetic reserve crew assumptions.

    Primary crews remain unchanged, allowing
    a controlled comparison across scenarios.
    """

    adjusted = crews.copy(deep=True)

    if scenario == "Normal availability":
        return adjusted

    if scenario == "50% reserve availability":

        primary = adjusted[
            adjusted["CrewType"] == "Primary"
        ].copy()

        reserve = adjusted[
            adjusted["CrewType"] == "Reserve"
        ].copy()

        # Deterministically retain approximately
        # half of reserves within each airline.
        reserve = (
            reserve.sort_values("CrewID")
            .groupby(
                "Reporting_Airline",
                group_keys=False,
            )
            .apply(
                lambda group: group.iloc[
                    :int(np.ceil(len(group) * 0.5))
                ],
                include_groups=False,
            )
        )

        # Restore grouping field because apply
        # excludes grouping columns.
        # Use the original indexed subset instead.
        reserve_ids = set(reserve["CrewID"])

        adjusted = adjusted[
            (adjusted["CrewType"] == "Primary")
            | adjusted["CrewID"].isin(reserve_ids)
        ].copy()

    elif scenario == "Reserves arrive 60 min late":

        mask = adjusted["CrewType"] == "Reserve"

        adjusted.loc[
            mask, "AvailableMinute"
        ] += 60

    elif scenario == "Reserve duty reduced by 60 min":

        mask = adjusted["CrewType"] == "Reserve"

        adjusted.loc[
            mask, "RemainingDutyMinutes"
        ] = np.maximum(
            adjusted.loc[
                mask, "RemainingDutyMinutes"
            ] - 60,
            0,
        )

    else:
        raise ValueError(
            f"Unknown scenario: {scenario}"
        )

    if adjusted["CrewID"].duplicated().any():
        raise RuntimeError(
            "Duplicate synthetic crew identifiers."
        )

    return adjusted


def main():
    print("=" * 65)
    print("AERORESILIENCE AI — CREW RECOVERY STRESS TEST")
    print("=" * 65)

    flights, crews = load_inputs()

    results = []
    details = []

    for scenario in SCENARIOS:

        print("\n" + "-" * 65)
        print(f"SCENARIO: {scenario}")
        print("-" * 65)

        scenario_crews = prepare_scenario(
            crews, scenario
        )

        available_reserves = int(
            (
                scenario_crews["CrewType"]
                == "Reserve"
            ).sum()
        )

        for policy, time_column in POLICIES.items():

            policy_details, summary = evaluate_policy(
                flights,
                scenario_crews,
                policy,
                time_column,
            )

            summary["Scenario"] = scenario
            summary["AvailableReserveCrews"] = (
                available_reserves
            )

            results.append(summary)

            policy_details["StressScenario"] = scenario
            details.append(policy_details)

            print(
                f"{policy}: "
                f"{summary['CoveredFlights']}/"
                f"{summary['RecoveryFlights']} covered, "
                f"reserves used={summary['ReserveCrewsUsed']}, "
                f"uncovered={summary['UncoveredFlights']}"
            )

    comparison = pd.DataFrame(results)

    all_details = pd.concat(
        details,
        ignore_index=True,
    )

    # -------------------------------------------------
    # VERIFY ORIGINAL RESULTS
    # -------------------------------------------------

    original = comparison[
        comparison["Scenario"]
        == "Normal availability"
    ]

    for row in original.itertuples(index=False):

        if (
            row.RecoveryFlights != 177
            or row.CoveredFlights != 177
            or row.ReserveCrewsUsed != 3
            or row.UncoveredFlights != 0
        ):
            raise RuntimeError(
                "Original Step 36 results were not "
                "reproduced. Review input consistency."
            )

    if len(comparison) != (
        len(SCENARIOS) * len(POLICIES)
    ):
        raise RuntimeError(
            "Unexpected stress-test scenario count."
        )

    # -------------------------------------------------
    # SAVE RESULTS
    # -------------------------------------------------

    comparison_file = (
        REPORT_DIR / "crew_recovery_stress_test.csv"
    )

    details_file = (
        REPORT_DIR / "crew_recovery_stress_details.csv"
    )

    comparison.to_csv(
        comparison_file,
        index=False,
    )

    all_details.to_csv(
        details_file,
        index=False,
    )

    # -------------------------------------------------
    # VISUALIZATION
    # -------------------------------------------------

    fig = px.bar(
        comparison,
        x="Scenario",
        y="UncoveredFlights",
        color="Policy",
        barmode="group",
        title="Synthetic Crew Recovery — Staffing Stress Test",
        labels={
            "Scenario": "Crew Availability Scenario",
            "UncoveredFlights": "Uncovered Flights",
        },
        template="plotly_dark",
        text="UncoveredFlights",
    )

    fig.update_layout(
        title_x=0.5,
        xaxis_tickangle=-20,
    )

    chart_file = (
        FIGURE_DIR / "crew_recovery_stress_test.html"
    )

    fig.write_html(chart_file)

    # -------------------------------------------------
    # FINAL TERMINAL RESULTS
    # -------------------------------------------------

    print("\n" + "=" * 65)
    print("CREW RECOVERY STRESS TEST RESULTS")
    print("=" * 65)

    print(
        comparison[
            [
                "Scenario",
                "Policy",
                "AvailableReserveCrews",
                "PrimaryCrewConflicts",
                "ReserveCrewsUsed",
                "CoveredFlights",
                "UncoveredFlights",
                "CoverageRatePercent",
            ]
        ].to_string(index=False)
    )

    print("\nIMPORTANT LIMITATIONS")
    print(
        "These tests use synthetic crew units "
        "and hypothetical staffing restrictions."
    )
    print(
        "Crew qualification and regulatory legality "
        "are not represented."
    )
    print(
        "Uncovered flights are flagged, not "
        "automatically delayed or cancelled."
    )
    print(
        "Reserve reassignment is evaluated separately "
        "for each policy and scenario."
    )
    print(
        "This analysis tests staffing sensitivity, "
        "not real airline staffing reliability."
    )

    print(f"\nComparison saved: {comparison_file}")
    print(f"Details saved: {details_file}")
    print(f"Chart saved: {chart_file}")

    print(
        "\nAERORESILIENCE AI — "
        "CREW STRESS TEST COMPLETE"
    )


if __name__ == "__main__":
    main()
