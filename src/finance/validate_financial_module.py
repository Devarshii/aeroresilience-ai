
from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# AERORESILIENCE AI
# MODULE 4 — FINANCIAL IMPACT & RECOVERY ANALYTICS
# STEP 45 — FINANCIAL MODEL VALIDATION
# ============================================================

REPORT_DIR = Path("outputs/reports")
FIGURE_DIR = Path("outputs/figures")

POLICIES = ["FCFS", "AircraftAware"]
TOLERANCE = 0.01

FILES = {
    "baseline": (
        REPORT_DIR / "recovery_financial_cost_baseline.csv"
    ),
    "comparison": (
        REPORT_DIR / "recovery_financial_comparison.csv"
    ),
    "assumptions": (
        REPORT_DIR / "financial_cost_assumptions.json"
    ),
    "scenarios": (
        REPORT_DIR / "financial_sensitivity_scenarios.csv"
    ),
    "summary": (
        REPORT_DIR / "financial_sensitivity_summary.csv"
    ),
    "break_even": (
        REPORT_DIR / "financial_break_even_analysis.csv"
    ),
    "drivers": (
        REPORT_DIR / "financial_cost_drivers.csv"
    ),
    "executive_report": (
        REPORT_DIR / "financial_executive_decision_report.md"
    ),
    "dashboard": (
        FIGURE_DIR / "financial_decision_dashboard.html"
    ),
}


# ============================================================
# 1. VALIDATION FRAMEWORK
# ============================================================

checks = []


def check(name, condition, description):

    passed = bool(condition)

    checks.append({
        "Check": name,
        "Status": "PASS" if passed else "FAIL",
        "Details": description,
    })

    print(
        f"[{'PASS' if passed else 'FAIL'}] "
        f"{name}"
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


# ============================================================
# 2. VERIFY OUTPUT FILES
# ============================================================

def verify_files():

    print("\nChecking Module 4 output files...")

    for name, path in FILES.items():

        check(
            f"File exists: {name}",
            path.is_file() and path.stat().st_size > 0,
            str(path),
        )

    missing = [
        str(path)
        for path in FILES.values()
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing required reports:\n"
            + "\n".join(missing)
        )


# ============================================================
# 3. LOAD RESULTS
# ============================================================

def load_results():

    baseline = pd.read_csv(FILES["baseline"])
    comparison = pd.read_csv(FILES["comparison"])
    scenarios = pd.read_csv(FILES["scenarios"])
    summary = pd.read_csv(FILES["summary"])
    break_even = pd.read_csv(FILES["break_even"])
    drivers = pd.read_csv(FILES["drivers"])

    with open(
        FILES["assumptions"],
        "r",
        encoding="utf-8",
    ) as file:
        configuration = json.load(file)

    assumptions = configuration["FinancialAssumptions"]

    return {
        "baseline": baseline,
        "comparison": comparison,
        "scenarios": scenarios,
        "summary": summary,
        "break_even": break_even,
        "drivers": drivers,
        "assumptions": assumptions,
    }


# ============================================================
# 4. VERIFY BASELINE COST CALCULATIONS
# ============================================================

def verify_baseline(data):

    print("\nValidating baseline recovery costs...")

    baseline = data["baseline"]
    assumptions = data["assumptions"]

    check(
        "Exactly two recovery policies",
        (
            len(baseline) == 2
            and set(baseline["Policy"]) == set(POLICIES)
            and baseline["Policy"].is_unique
        ),
        "Expected FCFS and AircraftAware",
    )

    rates = {
        "OperationalDelayCostUSD": (
            "IncrementalNetworkDelayMinutes",
            "OperationalCostPerIncrementalDelayMinute",
        ),
        "PassengerRebookingCostUSD": (
            "PassengersRebooked",
            "PassengerRebookingProcessingCost",
        ),
        "PassengerAssistanceCostUSD": (
            "PassengersUnaccommodated",
            "UnaccommodatedPassengerAssistanceCost",
        ),
        "ReserveCrewAssignmentCostUSD": (
            "SyntheticReserveCrewAssignments",
            "ReserveCrewAssignmentCost",
        ),
    }

    for row in baseline.itertuples(index=False):

        for cost_column, (
            quantity_column,
            rate_name,
        ) in rates.items():

            expected = (
                float(getattr(row, quantity_column))
                * float(assumptions[rate_name])
            )

            actual = float(
                getattr(row, cost_column)
            )

            check(
                f"{row.Policy}: {cost_column}",
                close(actual, expected),
                f"Expected ${expected:,.2f}",
            )

        components = sum(
            float(getattr(row, column))
            for column in rates
        )

        check(
            f"{row.Policy}: total cost reconciles",
            close(
                row.TotalModeledRecoveryCostUSD,
                components,
            ),
            f"Components total ${components:,.2f}",
        )

        passenger_total = (
            row.PassengersRebooked
            + row.PassengersUnaccommodated
        )

        check(
            f"{row.Policy}: passengers reconcile",
            passenger_total
            == row.SyntheticPassengersAtRisk,
            "Rebooked plus unaccommodated equals at risk",
        )


# ============================================================
# 5. VERIFY POLICY COMPARISON
# ============================================================

def verify_comparison(data):

    print("\nValidating policy comparison...")

    baseline = data["baseline"].set_index("Policy")
    comparison = data["comparison"]

    fcfs = baseline.loc["FCFS"]
    aircraft = baseline.loc["AircraftAware"]

    expected_difference = (
        fcfs["TotalModeledRecoveryCostUSD"]
        - aircraft["TotalModeledRecoveryCostUSD"]
    )

    check(
        "One baseline financial comparison",
        len(comparison) == 1,
        "Exactly one FCFS versus AircraftAware comparison",
    )

    recorded_difference = comparison.iloc[0][
        "NetModeledCostReductionUSD"
    ]

    check(
        "Net cost difference reconciles",
        close(
            recorded_difference,
            expected_difference,
        ),
        f"Expected difference ${expected_difference:,.2f}",
    )

    expected_percentage = (
        100 * expected_difference
        / fcfs["TotalModeledRecoveryCostUSD"]
    )

    recorded_percentage = comparison.iloc[0][
        "NetModeledCostReductionPercent"
    ]

    check(
        "Cost reduction percentage reconciles",
        close(
            recorded_percentage,
            expected_percentage,
        ),
        f"Expected {expected_percentage:.4f}%",
    )


# ============================================================
# 6. VERIFY FINANCIAL SENSITIVITY SCENARIOS
# ============================================================

def verify_scenarios(data):

    print("\nValidating sensitivity scenarios...")

    scenarios = data["scenarios"]
    baseline = data["baseline"].set_index("Policy")
    assumptions = data["assumptions"]

    check(
        "72 unique financial scenarios",
        (
            len(scenarios) == 72
            and scenarios["ScenarioID"].is_unique
        ),
        f"Found {len(scenarios)} scenarios",
    )

    actual_grid = set(
        zip(
            scenarios["DelayCostPerMinuteUSD"],
            scenarios["RebookingCostPerPassengerUSD"],
            scenarios["AssistanceCostPerPassengerUSD"],
        )
    )

    check(
        "Financial scenario grid has no duplicates",
        len(actual_grid) == len(scenarios),
        "Each unit-cost combination should be unique",
    )

    all_costs_valid = True
    winners_valid = True

    for row in scenarios.itertuples(index=False):

        calculated = {}

        for policy in POLICIES:

            policy_data = baseline.loc[policy]

            calculated[policy] = (
                policy_data[
                    "IncrementalNetworkDelayMinutes"
                ]
                * row.DelayCostPerMinuteUSD

                + policy_data["PassengersRebooked"]
                * row.RebookingCostPerPassengerUSD

                + policy_data["PassengersUnaccommodated"]
                * row.AssistanceCostPerPassengerUSD

                + policy_data[
                    "SyntheticReserveCrewAssignments"
                ]
                * assumptions["ReserveCrewAssignmentCost"]
            )

        difference = (
            calculated["FCFS"]
            - calculated["AircraftAware"]
        )

        valid = (
            close(
                row.FCFSTotalCostUSD,
                calculated["FCFS"],
            )
            and close(
                row.AircraftAwareTotalCostUSD,
                calculated["AircraftAware"],
            )
            and close(
                row.NetCostReductionUSD,
                difference,
            )
        )

        all_costs_valid = all_costs_valid and valid

        if difference > 0.000001:
            winner = "AircraftAware"
        elif difference < -0.000001:
            winner = "FCFS"
        else:
            winner = "Tie"

        winners_valid = (
            winners_valid
            and winner == row.LowerCostPolicy
        )

    check(
        "All sensitivity cost calculations",
        all_costs_valid,
        "Recomputed every scenario from input quantities",
    )

    check(
        "All scenario policy winners",
        winners_valid,
        "Verified lower-cost policy for each scenario",
    )

    counts = scenarios[
        "LowerCostPolicy"
    ].value_counts()

    recorded_counts = (
        data["summary"]
        .set_index("LowerCostPolicy")["ScenarioCount"]
        .to_dict()
    )

    count_match = all(
        int(counts.get(policy, 0))
        == int(recorded_counts.get(policy, 0))
        for policy in ["FCFS", "AircraftAware", "Tie"]
    )

    check(
        "Sensitivity summary reconciles",
        count_match,
        "Summary counts match detailed scenarios",
    )


# ============================================================
# 7. VERIFY BREAK-EVEN AND COST DRIVERS
# ============================================================

def verify_break_even(data):

    print("\nValidating break-even analysis...")

    baseline = data["baseline"].set_index("Policy")
    assumptions = data["assumptions"]

    fcfs = baseline.loc["FCFS"]
    aircraft = baseline.loc["AircraftAware"]

    delay_avoided = (
        fcfs["IncrementalNetworkDelayMinutes"]
        - aircraft["IncrementalNetworkDelayMinutes"]
    )

    non_delay_difference = (
        (
            fcfs["PassengersRebooked"]
            - aircraft["PassengersRebooked"]
        )
        * assumptions["PassengerRebookingProcessingCost"]

        + (
            fcfs["PassengersUnaccommodated"]
            - aircraft["PassengersUnaccommodated"]
        )
        * assumptions["UnaccommodatedPassengerAssistanceCost"]

        + (
            fcfs["SyntheticReserveCrewAssignments"]
            - aircraft["SyntheticReserveCrewAssignments"]
        )
        * assumptions["ReserveCrewAssignmentCost"]
    )

    recorded = data["break_even"].iloc[0]

    check(
        "Delay minutes avoided",
        close(
            recorded["DelayMinutesAvoided"],
            delay_avoided,
        ),
        f"Expected {delay_avoided:,.0f}",
    )

    check(
        "Non-delay cost difference",
        close(
            recorded["NonDelayCostDifferenceUSD"],
            non_delay_difference,
        ),
        f"Expected ${non_delay_difference:,.2f}",
    )

    if delay_avoided != 0:

        expected_threshold = (
            -non_delay_difference / delay_avoided
        )

        check(
            "Break-even delay rate",
            close(
                recorded["BreakEvenDelayCostPerMinuteUSD"],
                expected_threshold,
            ),
            f"Expected ${expected_threshold:.4f}/minute",
        )

    drivers = data["drivers"]

    check(
        "Four financial cost drivers",
        len(drivers) == 4,
        "Operational, rebooking, assistance, crew",
    )

    check(
        "Cost-driver differences reconcile",
        close(
            drivers["CostReductionUSD"].sum(),
            fcfs["TotalModeledRecoveryCostUSD"]
            - aircraft["TotalModeledRecoveryCostUSD"],
        ),
        "Cost categories explain the total policy difference",
    )


# ============================================================
# 8. SAVE FINAL VALIDATION REPORT
# ============================================================

def save_audit():

    audit = pd.DataFrame(checks)

    output_file = (
        REPORT_DIR / "financial_module_validation.csv"
    )

    audit.to_csv(
        output_file,
        index=False,
    )

    passed = int(
        (audit["Status"] == "PASS").sum()
    )

    failed = int(
        (audit["Status"] == "FAIL").sum()
    )

    print("\n" + "=" * 65)
    print("MODULE 4 FINANCIAL VALIDATION SUMMARY")
    print("=" * 65)

    print(f"Total checks: {len(audit)}")
    print(f"Passed checks: {passed}")
    print(f"Failed checks: {failed}")

    print(f"\nValidation report saved: {output_file}")

    if failed > 0:

        print("\nFAILED VALIDATIONS")

        print(
            audit[
                audit["Status"] == "FAIL"
            ].to_string(index=False)
        )

        raise RuntimeError(
            f"Financial module audit failed "
            f"{failed} validation checks."
        )

    print(
        "\nAERORESILIENCE AI — "
        "FINANCIAL MODULE VALIDATION COMPLETE"
    )


# ============================================================
# 9. MAIN PROGRAM
# ============================================================

def main():

    print("=" * 65)
    print("AERORESILIENCE AI")
    print("STEP 45 — FINANCIAL MODEL VALIDATION")
    print("=" * 65)

    verify_files()

    data = load_results()

    verify_baseline(data)

    verify_comparison(data)

    verify_scenarios(data)

    verify_break_even(data)

    save_audit()


if __name__ == "__main__":
    main()
