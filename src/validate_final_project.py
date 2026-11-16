
"""
AeroResilience AI
Step 56 — Final Project Validation

Checks:
1. Required analytical reports
2. Report schemas and basic data integrity
3. Reconciliation of aircraft, passenger, crew
   and financial results
4. Saved flight network source
5. Streamlit application syntax
6. Simulation methodology disclosures

Does not modify existing project data or models.
"""

from pathlib import Path
import ast
import json
import math
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "outputs" / "reports"
OUTPUT = REPORTS / "final_project_validation.csv"
SUMMARY = REPORTS / "final_project_validation_summary.json"
APP = ROOT / "app.py"

RESULTS = []


def check(category, name, passed, details=""):
    status = "PASS" if passed else "FAIL"

    RESULTS.append({
        "Category": category,
        "Check": name,
        "Status": status,
        "Details": str(details),
    })

    print(f"[{status}] {name}")

    if details and not passed:
        print(f"       {details}")


def read_report(filename, required_columns=None):
    path = REPORTS / filename

    if not path.is_file():
        check(
            "File",
            filename,
            False,
            "Required report is missing",
        )
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)

        check(
            "File",
            filename,
            not df.empty,
            f"{len(df):,} rows",
        )

        if required_columns:
            missing = [
                col for col in required_columns
                if col not in df.columns
            ]

            check(
                "Schema",
                f"{filename}: required columns",
                not missing,
                f"Missing: {missing}" if missing else "All present",
            )

        return df

    except Exception as exc:
        check(
            "File",
            filename,
            False,
            str(exc),
        )
        return pd.DataFrame()


def row_for_policy(df, policy):
    if "Policy" not in df.columns:
        return None

    rows = df[
        df["Policy"].astype(str).str.strip() == policy
    ]

    return rows.iloc[0] if not rows.empty else None


def numeric(row, column):
    if row is None or column not in row.index:
        return None

    value = pd.to_numeric(row[column], errors="coerce")

    if pd.isna(value):
        return None

    return float(value)


def close(a, b, tolerance=0.02):
    return (
        a is not None
        and b is not None
        and math.isclose(
            a,
            b,
            abs_tol=tolerance,
            rel_tol=0,
        )
    )


def validate_passengers(df):
    if df.empty:
        return

    required = [
        "SimulatedPassengersAtRisk",
        "PassengersRebooked",
        "PassengersUnaccommodated",
        "RebookingRatePercent",
    ]

    for policy in ["FCFS", "AircraftAware"]:
        row = row_for_policy(df, policy)

        check(
            "Passenger",
            f"{policy}: policy record",
            row is not None,
        )

        if row is None:
            continue

        values = {
            key: numeric(row, key)
            for key in required
        }

        if any(value is None for value in values.values()):
            check(
                "Passenger",
                f"{policy}: required metrics",
                False,
                "Missing or nonnumeric metrics",
            )
            continue

        at_risk = values["SimulatedPassengersAtRisk"]
        rebooked = values["PassengersRebooked"]
        uncovered = values["PassengersUnaccommodated"]
        rate = values["RebookingRatePercent"]

        check(
            "Passenger",
            f"{policy}: passenger conservation",
            close(at_risk, rebooked + uncovered),
            f"{at_risk:g} = {rebooked:g} + {uncovered:g}",
        )

        expected_rate = (
            100 * rebooked / at_risk
            if at_risk > 0 else 0
        )

        check(
            "Passenger",
            f"{policy}: rebooking rate",
            close(expected_rate, rate),
            f"Calculated {expected_rate:.4f}%, saved {rate:.4f}%",
        )


def validate_finance(df):
    if df.empty:
        return

    components = [
        "OperationalDelayCostUSD",
        "PassengerRebookingCostUSD",
        "PassengerAssistanceCostUSD",
        "ReserveCrewAssignmentCostUSD",
    ]

    for policy in ["FCFS", "AircraftAware"]:
        row = row_for_policy(df, policy)

        check(
            "Finance",
            f"{policy}: policy record",
            row is not None,
        )

        if row is None:
            continue

        costs = [
            numeric(row, col)
            for col in components
        ]

        total = numeric(
            row,
            "TotalModeledRecoveryCostUSD",
        )

        if None in costs or total is None:
            check(
                "Finance",
                f"{policy}: cost inputs complete",
                False,
            )
            continue

        calculated = sum(costs)

        check(
            "Finance",
            f"{policy}: cost components reconcile",
            close(calculated, total),
            f"Components ${calculated:,.2f}; "
            f"saved total ${total:,.2f}",
        )

    fcfs = row_for_policy(df, "FCFS")
    aware = row_for_policy(df, "AircraftAware")

    if fcfs is None or aware is None:
        return

    fcfs_cost = numeric(
        fcfs,
        "TotalModeledRecoveryCostUSD",
    )

    aware_cost = numeric(
        aware,
        "TotalModeledRecoveryCostUSD",
    )

    check(
        "Finance",
        "Cost comparison is calculable",
        fcfs_cost is not None and aware_cost is not None,
        (
            f"FCFS minus AircraftAware = "
            f"${fcfs_cost - aware_cost:,.2f}"
            if fcfs_cost is not None
            and aware_cost is not None
            else "Missing cost values"
        ),
    )


def validate_aircraft(df):
    if df.empty:
        return

    required = [
        "MinimumTurnMinutes",
        "FCFSIncrementalDelayMinutes",
        "AircraftAwareIncrementalDelayMinutes",
        "DelayMinutesAvoided",
        "AircraftDelayImprovementPercent",
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        check(
            "Aircraft",
            "Aircraft comparison columns",
            False,
            f"Missing: {missing}",
        )
        return

    for _, row in df.iterrows():
        turn = numeric(row, "MinimumTurnMinutes")
        fcfs = numeric(row, "FCFSIncrementalDelayMinutes")
        aware = numeric(
            row,
            "AircraftAwareIncrementalDelayMinutes",
        )

        avoided = numeric(row, "DelayMinutesAvoided")
        improvement = numeric(
            row,
            "AircraftDelayImprovementPercent",
        )

        if None in [turn, fcfs, aware, avoided, improvement]:
            check(
                "Aircraft",
                "Aircraft scenario numeric values",
                False,
            )
            continue

        expected_avoided = fcfs - aware

        expected_percent = (
            100 * expected_avoided / fcfs
            if fcfs > 0 else 0
        )

        check(
            "Aircraft",
            f"{turn:g}-minute turn: delay reconciliation",
            close(expected_avoided, avoided),
        )

        check(
            "Aircraft",
            f"{turn:g}-minute turn: percent reconciliation",
            close(expected_percent, improvement),
        )


def validate_global_passengers(df):
    if df.empty:
        return

    required = [
        "PassengersAtRisk",
        "GreedyRebooked",
        "OptimizedRebooked",
        "GreedyUnaccommodated",
        "OptimizedUnaccommodated",
    ]

    if any(col not in df.columns for col in required):
        check(
            "Optimization",
            "Global passenger optimization schema",
            False,
        )
        return

    for policy in ["FCFS", "AircraftAware"]:
        row = row_for_policy(df, policy)

        if row is None:
            check(
                "Optimization",
                f"{policy}: optimization record",
                False,
            )
            continue

        at_risk = numeric(row, "PassengersAtRisk")
        greedy = numeric(row, "GreedyRebooked")
        optimized = numeric(row, "OptimizedRebooked")
        greedy_uncovered = numeric(
            row,
            "GreedyUnaccommodated",
        )
        optimized_uncovered = numeric(
            row,
            "OptimizedUnaccommodated",
        )

        check(
            "Optimization",
            f"{policy}: greedy conservation",
            close(
                at_risk,
                greedy + greedy_uncovered
                if greedy is not None
                and greedy_uncovered is not None
                else None,
            ),
        )

        check(
            "Optimization",
            f"{policy}: optimized conservation",
            close(
                at_risk,
                optimized + optimized_uncovered
                if optimized is not None
                and optimized_uncovered is not None
                else None,
            ),
        )

        check(
            "Optimization",
            f"{policy}: optimized coverage not worse",
            (
                optimized is not None
                and greedy is not None
                and optimized >= greedy
            ),
        )


def validate_network_source():
    path = REPORTS / "recovery_baseline_flights.csv"

    df = read_report(
        "recovery_baseline_flights.csv",
        ["Origin", "Dest"],
    )

    if df.empty:
        return

    pairs = (
        df[["Origin", "Dest"]]
        .dropna()
        .astype(str)
        .drop_duplicates()
    )

    check(
        "Network",
        "Flight network contains route pairs",
        len(pairs) > 0,
        f"{len(pairs):,} unique airport pairs",
    )


def validate_streamlit():
    if not APP.is_file():
        check(
            "Dashboard",
            "app.py exists",
            False,
        )
        return

    check(
        "Dashboard",
        "app.py exists",
        True,
    )

    try:
        source = APP.read_text(encoding="utf-8")
        ast.parse(source)

        check(
            "Dashboard",
            "app.py Python syntax",
            True,
        )

    except Exception as exc:
        check(
            "Dashboard",
            "app.py Python syntax",
            False,
            str(exc),
        )
        return

    sections = [
        "Executive Overview",
        "Flight Disruption Intelligence",
        "Aircraft Recovery",
        "Crew Recovery",
        "Passenger Intelligence",
        "Financial Analytics",
        "What-If Simulator",
    ]

    for section in sections:
        check(
            "Dashboard",
            f"Navigation: {section}",
            section in source,
        )

    check(
        "Dashboard",
        "Simulation limitations disclosed",
        "RESEARCH SIMULATION" in source,
    )

    check(
        "Dashboard",
        "Aircraft banner title",
        "AeroResilience AI</div>" in source,
    )


def main():
    REPORTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 65)
    print("AERORESILIENCE AI — FINAL PROJECT VALIDATION")
    print("=" * 65)

    print("\n[1] DASHBOARD VALIDATION")
    validate_streamlit()

    print("\n[2] PASSENGER RECOVERY")
    passengers = read_report(
        "passenger_rebooking_summary.csv",
        [
            "Policy",
            "SimulatedPassengersAtRisk",
            "PassengersRebooked",
            "PassengersUnaccommodated",
            "RebookingRatePercent",
        ],
    )
    validate_passengers(passengers)

    print("\n[3] AIRCRAFT RECOVERY")
    aircraft = read_report(
        "integrated_resilience_comparison.csv",
    )
    validate_aircraft(aircraft)

    print("\n[4] FINANCIAL RECOVERY")
    finance = read_report(
        "recovery_financial_cost_baseline.csv",
        [
            "Policy",
            "TotalModeledRecoveryCostUSD",
        ],
    )
    validate_finance(finance)

    print("\n[5] PASSENGER OPTIMIZATION")
    global_passengers = read_report(
        "global_passenger_optimization_summary.csv",
    )
    validate_global_passengers(global_passengers)

    print("\n[6] CREW RECOVERY")
    crew = read_report(
        "crew_resilience_comparison.csv",
        ["Scenario"],
    )

    if not crew.empty:
        check(
            "Crew",
            "Crew resilience scenarios present",
            len(crew) > 0,
            f"{len(crew)} comparison rows",
        )

    print("\n[7] WHAT-IF SIMULATION")
    what_if = read_report(
        "resilience_what_if_scenarios.csv",
        [
            "MinimumTurnMinutes",
            "DelayCostPerMinuteUSD",
            "FCFSModeledCostUSD",
            "AircraftAwareModeledCostUSD",
            "CostReductionUSD",
        ],
    )

    if not what_if.empty:
        correct = (
            pd.to_numeric(
                what_if["FCFSModeledCostUSD"],
                errors="coerce",
            )
            - pd.to_numeric(
                what_if["AircraftAwareModeledCostUSD"],
                errors="coerce",
            )
            - pd.to_numeric(
                what_if["CostReductionUSD"],
                errors="coerce",
            )
        ).abs()

        check(
            "What-If",
            "Scenario cost differences reconcile",
            bool(correct.notna().all() and (correct < 0.02).all()),
        )

    print("\n[8] FLIGHT NETWORK")
    validate_network_source()

    print("\n[9] FINANCIAL SENSITIVITY")
    sensitivity = read_report(
        "financial_sensitivity_scenarios.csv",
        ["LowerCostPolicy"],
    )

    if not sensitivity.empty:
        check(
            "Sensitivity",
            "Financial scenario policies present",
            sensitivity["LowerCostPolicy"].notna().all(),
        )

    print("\n" + "=" * 65)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 65)

    result_df = pd.DataFrame(RESULTS)

    passed = int(
        (result_df["Status"] == "PASS").sum()
    )

    failed = int(
        (result_df["Status"] == "FAIL").sum()
    )

    total = len(result_df)

    print(f"Total checks: {total}")
    print(f"Passed checks: {passed}")
    print(f"Failed checks: {failed}")

    result_df.to_csv(
        OUTPUT,
        index=False,
    )

    summary = {
        "Project": "AeroResilience AI",
        "ValidationStep": 56,
        "TotalChecks": total,
        "PassedChecks": passed,
        "FailedChecks": failed,
        "OverallStatus": (
            "PASS" if failed == 0 else "FAIL"
        ),
        "Scope": (
            "File availability, schemas, financial and "
            "operational reconciliation, and dashboard syntax"
        ),
        "Limitation": (
            "This validation does not establish real-world "
            "operational accuracy, simulation calibration, "
            "or airline production readiness."
        ),
    }

    SUMMARY.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Validation report saved:")
    print(OUTPUT.relative_to(ROOT))

    print()
    print("Validation summary saved:")
    print(SUMMARY.relative_to(ROOT))

    if failed == 0:
        print()
        print(
            "AERORESILIENCE AI — "
            "FINAL PROJECT VALIDATION PASSED"
        )
    else:
        print()
        print(
            "AERORESILIENCE AI — "
            "VALIDATION NEEDS REVIEW"
        )

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
