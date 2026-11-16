
from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# AERORESILIENCE AI
# MODULE 5 — RESILIENCE & WHAT-IF SIMULATOR
# STEP 51 — FINAL INTEGRATION VALIDATION
# ============================================================

REPORT_DIR = Path("outputs/reports")
FIGURE_DIR = Path("outputs/figures")

REPORT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "integrated": REPORT_DIR / "integrated_resilience_scenarios.csv",
    "integrated_comparison": REPORT_DIR / "integrated_resilience_comparison.csv",
    "what_if": REPORT_DIR / "resilience_what_if_scenarios.csv",
    "what_if_summary": REPORT_DIR / "resilience_what_if_summary.csv",
    "passenger": REPORT_DIR / "passenger_connection_resilience_summary.csv",
    "passenger_comparison": REPORT_DIR / "passenger_connection_resilience_comparison.csv",
    "passenger_validation": REPORT_DIR / "passenger_connection_resilience_validation.csv",
    "crew": REPORT_DIR / "crew_resilience_summary.csv",
    "crew_comparison": REPORT_DIR / "crew_resilience_comparison.csv",
    "crew_validation": REPORT_DIR / "crew_resilience_validation.csv",
    "financial_assumptions": REPORT_DIR / "financial_cost_assumptions.json",
}

FIGURES = [
    FIGURE_DIR / "integrated_resilience_turnaround.html",
    FIGURE_DIR / "resilience_what_if_heatmap.html",
    FIGURE_DIR / "passenger_connection_resilience.html",
    FIGURE_DIR / "crew_resilience_dashboard.html",
]

POLICIES = {"FCFS", "AircraftAware"}
TOLERANCE = 0.01

CHECKS = []


# ============================================================
# 1. VALIDATION UTILITIES
# ============================================================

def check(name, condition, details=""):

    passed = bool(condition)

    CHECKS.append({
        "Check": name,
        "Status": "PASS" if passed else "FAIL",
        "Details": str(details),
    })

    print(
        f"[{'PASS' if passed else 'FAIL'}] {name}"
    )


def close(actual, expected):

    return bool(
        np.isclose(
            float(actual),
            float(expected),
            atol=TOLERANCE,
            rtol=0,
        )
    )


def require_columns(data, columns, source):

    missing = [
        column for column in columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"{source} missing columns: {missing}"
        )


def verify_policy_pairs(data, group_column):

    for _, group in data.groupby(group_column):

        if (
            len(group) != 2
            or set(group["Policy"]) != POLICIES
        ):
            return False

    return True


# ============================================================
# 2. LOAD EXISTING RESULTS
# ============================================================

def load_data():

    print("\nChecking Module 5 outputs...")

    for name, path in FILES.items():

        check(
            f"Required report: {name}",
            path.is_file() and path.stat().st_size > 0
            if path.exists() else False,
            path,
        )

    for path in FIGURES:

        check(
            f"Visualization: {path.name}",
            path.is_file() and path.stat().st_size > 0
            if path.exists() else False,
            path,
        )

    missing = [
        path for path in list(FILES.values()) + FIGURES
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing required Module 5 outputs:\n"
            + "\n".join(map(str, missing))
        )

    data = {}

    for name, path in FILES.items():

        if path.suffix == ".json":

            with open(path, encoding="utf-8") as file:
                data[name] = json.load(file)

        else:
            data[name] = pd.read_csv(path)

    return data


# ============================================================
# 3. VALIDATE AIRCRAFT RESILIENCE
# ============================================================

def validate_aircraft(data):

    print("\nVALIDATING AIRCRAFT RESILIENCE")

    integrated = data["integrated"]
    comparison = data["integrated_comparison"]

    require_columns(
        integrated,
        [
            "MinimumTurnMinutes",
            "Policy",
            "NoStopDelayMinutes",
            "TotalModeledDelayMinutes",
            "AircraftIncrementalDelayMinutes",
            "IllustrativeCombinedCostUSD",
            "PassengersRebookedBaseline",
            "PassengersUnaccommodatedBaseline",
            "PassengersAtRiskBaseline",
        ],
        "Integrated resilience",
    )

    require_columns(
        comparison,
        [
            "MinimumTurnMinutes",
            "DelayMinutesAvoided",
            "IllustrativeCostReductionUSD",
        ],
        "Integrated comparison",
    )

    check(
        "Matched aircraft recovery policy pairs",
        verify_policy_pairs(
            integrated,
            "MinimumTurnMinutes",
        ),
    )

    check(
        "Unique aircraft policy-scenario records",
        not integrated.duplicated(
            ["MinimumTurnMinutes", "Policy"]
        ).any(),
    )

    calculated_delay = (
        integrated["TotalModeledDelayMinutes"]
        - integrated["NoStopDelayMinutes"]
    )

    check(
        "Aircraft incremental delays reconcile",
        np.allclose(
            calculated_delay,
            integrated["AircraftIncrementalDelayMinutes"],
            atol=TOLERANCE,
            rtol=0,
        ),
    )

    check(
        "Integrated passenger totals reconcile",
        (
            integrated["PassengersRebookedBaseline"]
            + integrated["PassengersUnaccommodatedBaseline"]
            == integrated["PassengersAtRiskBaseline"]
        ).all(),
    )

    all_comparisons_match = True

    for row in comparison.itertuples(index=False):

        selected = integrated[
            integrated["MinimumTurnMinutes"]
            == row.MinimumTurnMinutes
        ].set_index("Policy")

        if set(selected.index) != POLICIES:
            all_comparisons_match = False
            continue

        expected_delay = (
            selected.loc[
                "FCFS", "AircraftIncrementalDelayMinutes"
            ]
            - selected.loc[
                "AircraftAware",
                "AircraftIncrementalDelayMinutes",
            ]
        )

        expected_cost = (
            selected.loc[
                "FCFS", "IllustrativeCombinedCostUSD"
            ]
            - selected.loc[
                "AircraftAware", "IllustrativeCombinedCostUSD"
            ]
        )

        all_comparisons_match &= (
            close(row.DelayMinutesAvoided, expected_delay)
            and close(
                row.IllustrativeCostReductionUSD,
                expected_cost,
            )
        )

    check(
        "Aircraft and cost policy comparisons reconcile",
        all_comparisons_match,
    )


# ============================================================
# 4. VALIDATE FINANCIAL WHAT-IF SCENARIOS
# ============================================================

def validate_what_if(data):

    print("\nVALIDATING FINANCIAL WHAT-IF SCENARIOS")

    scenarios = data["what_if"]
    summary = data["what_if_summary"]

    require_columns(
        scenarios,
        [
            "ScenarioID",
            "MinimumTurnMinutes",
            "FCFSModeledCostUSD",
            "AircraftAwareModeledCostUSD",
            "CostReductionUSD",
            "LowerCostPolicy",
        ],
        "What-if scenarios",
    )

    expected_count = (
        data["integrated"]["MinimumTurnMinutes"].nunique()
        * 5 * 3 * 3
    )

    check(
        "Expected financial scenario count",
        len(scenarios) == expected_count,
        f"Expected {expected_count}, found {len(scenarios)}",
    )

    check(
        "Unique financial scenario IDs",
        scenarios["ScenarioID"].is_unique,
    )

    difference = (
        scenarios["FCFSModeledCostUSD"]
        - scenarios["AircraftAwareModeledCostUSD"]
    )

    check(
        "All what-if cost differences reconcile",
        np.allclose(
            difference,
            scenarios["CostReductionUSD"],
            atol=TOLERANCE,
            rtol=0,
        ),
    )

    expected_winner = np.where(
        difference > 0.000001,
        "AircraftAware",
        np.where(
            difference < -0.000001,
            "FCFS",
            "Tie",
        ),
    )

    check(
        "What-if preferred policies reconcile",
        (
            expected_winner
            == scenarios["LowerCostPolicy"].to_numpy()
        ).all(),
    )

    check(
        "Scenario summary counts reconcile",
        int(summary["Scenarios"].sum()) == len(scenarios),
    )

    # Verify the cost formula against the integrated
    # scenario quantities and hypothetical unit rates.
    integrated = data["integrated"]

    reference = integrated.set_index(
        ["MinimumTurnMinutes", "Policy"]
    )

    all_costs_match = True

    required_cost_fields = [
        "DelayCostPerMinuteUSD",
        "RebookingCostPerPassengerUSD",
        "AssistanceCostPerPassengerUSD",
        "ReserveCrewAssignmentCostUSD",
    ]

    require_columns(
        scenarios,
        required_cost_fields,
        "Financial what-if",
    )

    for row in scenarios.itertuples(index=False):

        for policy, recorded in [
            ("FCFS", row.FCFSModeledCostUSD),
            ("AircraftAware", row.AircraftAwareModeledCostUSD),
        ]:

            source = reference.loc[
                (row.MinimumTurnMinutes, policy)
            ]

            expected = (
                source["AircraftIncrementalDelayMinutes"]
                * row.DelayCostPerMinuteUSD

                + source["PassengersRebookedBaseline"]
                * row.RebookingCostPerPassengerUSD

                + source["PassengersUnaccommodatedBaseline"]
                * row.AssistanceCostPerPassengerUSD

                + source["ReserveCrewAssignmentsBaseline"]
                * row.ReserveCrewAssignmentCostUSD
            )

            if not close(expected, recorded):
                all_costs_match = False

    check(
        "What-if costs reproduce the stated formula",
        all_costs_match,
    )


# ============================================================
# 5. VALIDATE PASSENGER RESILIENCE
# ============================================================

def validate_passengers(data):

    print("\nVALIDATING PASSENGER RESILIENCE")

    passenger = data["passenger"]
    comparison = data["passenger_comparison"]
    validation = data["passenger_validation"]

    require_columns(
        passenger,
        [
            "MinimumConnectionMinutes",
            "Policy",
            "SimulatedPassengers",
            "PassengersAtRisk",
            "PassengersNotAtRisk",
        ],
        "Passenger resilience",
    )

    check(
        "Passenger scenarios have both policies",
        verify_policy_pairs(
            passenger,
            "MinimumConnectionMinutes",
        ),
    )

    check(
        "Passenger risk totals reconcile",
        (
            passenger["PassengersAtRisk"]
            + passenger["PassengersNotAtRisk"]
            == passenger["SimulatedPassengers"]
        ).all(),
    )

    monotonic = True

    for _, group in passenger.groupby("Policy"):

        ordered = group.sort_values(
            "MinimumConnectionMinutes"
        )

        monotonic &= bool(
            (
                ordered["PassengersAtRisk"]
                .diff()
                .fillna(0)
                >= 0
            ).all()
        )

    check(
        "Risk never decreases as connection minimum rises",
        monotonic,
    )

    check(
        "Original 45-minute baseline reproduced",
        (
            validation["BaselineMatches"]
            .astype(str)
            .str.lower()
            .eq("true")
        ).all(),
    )

    check(
        "Passenger policy comparison count",
        len(comparison)
        == passenger["MinimumConnectionMinutes"].nunique(),
    )


# ============================================================
# 6. VALIDATE CREW RESILIENCE
# ============================================================

def validate_crew(data):

    print("\nVALIDATING CREW RESILIENCE")

    crew = data["crew"]
    comparison = data["crew_comparison"]
    validation = data["crew_validation"]

    require_columns(
        crew,
        [
            "Scenario",
            "Policy",
            "TotalFlights",
            "CoveredFlights",
            "UncoveredFlights",
            "CrewCoveragePercent",
        ],
        "Crew resilience",
    )

    check(
        "Crew stress scenarios have both policies",
        verify_policy_pairs(
            crew,
            "Scenario",
        ),
    )

    check(
        "Crew coverage counts reconcile",
        (
            crew["CoveredFlights"]
            + crew["UncoveredFlights"]
            == crew["TotalFlights"]
        ).all(),
    )

    calculated_coverage = (
        100
        * crew["CoveredFlights"]
        / crew["TotalFlights"]
    )

    check(
        "Crew coverage percentages reconcile",
        np.allclose(
            calculated_coverage,
            crew["CrewCoveragePercent"],
            atol=TOLERANCE,
            rtol=0,
        ),
    )

    check(
        "Existing crew validation checks passed",
        validation["Status"].eq("PASS").all(),
    )

    check(
        "Crew comparison scenario count",
        len(comparison)
        == crew["Scenario"].nunique(),
    )


# ============================================================
# 7. SAVE FINAL AUDIT AND METHODOLOGY
# ============================================================

def save_results():

    audit = pd.DataFrame(CHECKS)

    audit_file = (
        REPORT_DIR
        / "resilience_module_validation.csv"
    )

    audit.to_csv(audit_file, index=False)

    passed = int(
        audit["Status"].eq("PASS").sum()
    )

    failed = int(
        audit["Status"].eq("FAIL").sum()
    )

    methodology = {
        "Project": "AeroResilience AI",
        "Module": 5,
        "Step": 51,
        "Scope": (
            "Cross-report consistency validation of "
            "aircraft, passenger, crew and illustrative "
            "financial resilience analyses."
        ),
        "TotalValidationChecks": len(audit),
        "Passed": passed,
        "Failed": failed,
        "OperationalModelStatus": (
            "Integrated analysis prototype; "
            "not a fully coupled operational simulator"
        ),
        "ImportantLimitations": [
            "All crew and passenger groups are synthetic.",
            "Financial unit costs are hypothetical.",
            "Historical flights are used as scheduling inputs.",
            "Aircraft turnaround scenarios reuse "
            "previously simulated schedules.",
            "Passenger connection thresholds are varied "
            "without rerunning aircraft recovery.",
            "Crew stress outputs reuse earlier "
            "crew recovery simulations.",
            "Passenger and crew baselines are held fixed "
            "within financial what-if scenarios.",
            "Passing checks validates internal consistency, "
            "not prediction accuracy or real-world feasibility.",
            "No actual airline savings are established.",
        ],
    }

    methodology_file = (
        REPORT_DIR
        / "resilience_module_final_methodology.json"
    )

    with open(
        methodology_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            methodology,
            file,
            indent=4,
        )

    print("\n" + "=" * 65)
    print("MODULE 5 RESILIENCE VALIDATION SUMMARY")
    print("=" * 65)

    print(f"Total checks: {len(audit)}")
    print(f"Passed checks: {passed}")
    print(f"Failed checks: {failed}")

    print(f"\nAudit saved: {audit_file}")
    print(f"Methodology saved: {methodology_file}")

    if failed:

        print("\nFAILED CHECKS")

        print(
            audit[
                audit["Status"] == "FAIL"
            ].to_string(index=False)
        )

        raise RuntimeError(
            f"Module 5 validation failed: "
            f"{failed} checks"
        )

    print(
        "\nAERORESILIENCE AI — "
        "RESILIENCE MODULE VALIDATION COMPLETE"
    )


# ============================================================
# 8. MAIN
# ============================================================

def main():

    print("=" * 65)
    print("AERORESILIENCE AI")
    print("STEP 51 — FINAL RESILIENCE MODULE VALIDATION")
    print("=" * 65)

    data = load_data()

    validate_aircraft(data)

    validate_what_if(data)

    validate_passengers(data)

    validate_crew(data)

    save_results()


if __name__ == "__main__":
    main()
