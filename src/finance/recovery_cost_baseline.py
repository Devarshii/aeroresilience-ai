
from pathlib import Path
import json

import pandas as pd


# ============================================================
# AERORESILIENCE AI
# MODULE 4 — FINANCIAL IMPACT & RECOVERY ANALYTICS
# STEP 42 — RECOVERY COST BASELINE
# ============================================================

REPORT_DIR = Path("outputs/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

PASSENGER_FILE = (
    REPORT_DIR / "passenger_rebooking_summary.csv"
)

POLICIES = ["FCFS", "AircraftAware"]


# ============================================================
# 1. HISTORICAL SIMULATION RESULTS
# ============================================================
# These are previously calculated scenario outputs,
# NOT observed financial outcomes.
#
# Incremental delay minutes are from our August 12,
# 2026 ORD ground-stop recovery scenario.
#
# Crew assignments are from our synthetic crew model.
# ============================================================

SIMULATION_RESULTS = {
    "FCFS": {
        "IncrementalNetworkDelayMinutes": 103710,
        "SyntheticReserveCrewAssignments": 3,
    },
    "AircraftAware": {
        "IncrementalNetworkDelayMinutes": 96092,
        "SyntheticReserveCrewAssignments": 3,
    },
}


# ============================================================
# 2. HYPOTHETICAL FINANCIAL ASSUMPTIONS
# ============================================================
# All dollar values below are EXAMPLE assumptions.
#
# They are NOT verified airline costs or industry
# averages.
#
# Operational delay cost is modeled separately from
# passenger assistance costs to avoid knowingly
# counting those same passenger expenses twice.
#
# Real airline accounting is more complex.
# ============================================================

COST_ASSUMPTIONS = {
    "Currency": "USD",

    "OperationalCostPerIncrementalDelayMinute": 45.0,

    "PassengerRebookingProcessingCost": 80.0,

    "UnaccommodatedPassengerAssistanceCost": 250.0,

    "ReserveCrewAssignmentCost": 300.0,

    "Scenario": (
        "Hypothetical August 12, 2026 ORD "
        "ground-stop recovery"
    ),

    "AssumptionType": (
        "Illustrative financial scenario; "
        "not actual airline expenses"
    ),
}


# ============================================================
# 3. LOAD PASSENGER REBOOKING RESULTS
# ============================================================

def load_passenger_results():

    print("\nLoading Module 3 passenger results...")

    if not PASSENGER_FILE.exists():
        raise FileNotFoundError(
            f"Missing required report: {PASSENGER_FILE}\n"
            "Run Step 40 first."
        )

    passengers = pd.read_csv(PASSENGER_FILE)

    required_columns = [
        "Policy",
        "SimulatedPassengersAtRisk",
        "PassengersRebooked",
        "PassengersUnaccommodated",
    ]

    missing = [
        column
        for column in required_columns
        if column not in passengers.columns
    ]

    if missing:
        raise ValueError(
            f"Passenger summary missing columns: {missing}"
        )

    passengers = passengers[
        passengers["Policy"].isin(POLICIES)
    ].copy()

    if len(passengers) != len(POLICIES):
        raise ValueError(
            "Expected exactly two passenger policy rows."
        )

    if passengers["Policy"].duplicated().any():
        raise ValueError(
            "Duplicate recovery policies found."
        )

    numeric_columns = [
        "SimulatedPassengersAtRisk",
        "PassengersRebooked",
        "PassengersUnaccommodated",
    ]

    for column in numeric_columns:
        passengers[column] = pd.to_numeric(
            passengers[column],
            errors="raise",
        )

        if passengers[column].isna().any():
            raise ValueError(
                f"Missing values found in {column}"
            )

        if (passengers[column] < 0).any():
            raise ValueError(
                f"Negative values found in {column}"
            )

    for row in passengers.itertuples(index=False):

        total = (
            row.PassengersRebooked
            + row.PassengersUnaccommodated
        )

        if total != row.SimulatedPassengersAtRisk:
            raise ValueError(
                f"Passenger totals do not reconcile "
                f"for {row.Policy}"
            )

    print("\nPassenger results loaded successfully.")

    print(
        passengers[
            required_columns
        ].to_string(index=False)
    )

    return passengers


# ============================================================
# 4. COMPUTE FINANCIAL COSTS
# ============================================================

def calculate_financial_costs(passengers):

    print("\nCalculating hypothetical recovery costs...")

    records = []

    for policy in POLICIES:

        passenger_row = passengers[
            passengers["Policy"] == policy
        ].iloc[0]

        simulation = SIMULATION_RESULTS[policy]

        network_delay = int(
            simulation["IncrementalNetworkDelayMinutes"]
        )

        reserve_assignments = int(
            simulation["SyntheticReserveCrewAssignments"]
        )

        rebooked = int(
            passenger_row["PassengersRebooked"]
        )

        unaccommodated = int(
            passenger_row["PassengersUnaccommodated"]
        )

        at_risk = int(
            passenger_row["SimulatedPassengersAtRisk"]
        )

        # --------------------------------------------
        # COST CATEGORY 1: AIRCRAFT NETWORK DELAY
        # --------------------------------------------

        delay_cost = (
            network_delay
            * COST_ASSUMPTIONS[
                "OperationalCostPerIncrementalDelayMinute"
            ]
        )

        # --------------------------------------------
        # COST CATEGORY 2: PASSENGER REBOOKING
        # --------------------------------------------

        rebooking_cost = (
            rebooked
            * COST_ASSUMPTIONS[
                "PassengerRebookingProcessingCost"
            ]
        )

        # --------------------------------------------
        # COST CATEGORY 3: PASSENGER ASSISTANCE
        # --------------------------------------------

        assistance_cost = (
            unaccommodated
            * COST_ASSUMPTIONS[
                "UnaccommodatedPassengerAssistanceCost"
            ]
        )

        # --------------------------------------------
        # COST CATEGORY 4: RESERVE CREW ASSIGNMENT
        # --------------------------------------------

        crew_cost = (
            reserve_assignments
            * COST_ASSUMPTIONS[
                "ReserveCrewAssignmentCost"
            ]
        )

        # --------------------------------------------
        # TOTAL MODEL COST
        # --------------------------------------------

        total_cost = (
            delay_cost
            + rebooking_cost
            + assistance_cost
            + crew_cost
        )

        records.append({
            "Policy": policy,
            "IncrementalNetworkDelayMinutes": network_delay,
            "SyntheticPassengersAtRisk": at_risk,
            "PassengersRebooked": rebooked,
            "PassengersUnaccommodated": unaccommodated,
            "SyntheticReserveCrewAssignments": reserve_assignments,
            "OperationalDelayCostUSD": delay_cost,
            "PassengerRebookingCostUSD": rebooking_cost,
            "PassengerAssistanceCostUSD": assistance_cost,
            "ReserveCrewAssignmentCostUSD": crew_cost,
            "TotalModeledRecoveryCostUSD": total_cost,
        })

    return pd.DataFrame(records)


# ============================================================
# 5. COMPARE FINANCIAL PERFORMANCE
# ============================================================

def compare_policies(costs):

    indexed = costs.set_index("Policy")

    fcfs = indexed.loc["FCFS"]

    optimized = indexed.loc["AircraftAware"]

    financial_difference = (
        fcfs["TotalModeledRecoveryCostUSD"]
        - optimized["TotalModeledRecoveryCostUSD"]
    )

    delay_difference = (
        fcfs["IncrementalNetworkDelayMinutes"]
        - optimized["IncrementalNetworkDelayMinutes"]
    )

    delay_cost_difference = (
        fcfs["OperationalDelayCostUSD"]
        - optimized["OperationalDelayCostUSD"]
    )

    rebooking_difference = (
        fcfs["PassengerRebookingCostUSD"]
        - optimized["PassengerRebookingCostUSD"]
    )

    assistance_difference = (
        fcfs["PassengerAssistanceCostUSD"]
        - optimized["PassengerAssistanceCostUSD"]
    )

    crew_difference = (
        fcfs["ReserveCrewAssignmentCostUSD"]
        - optimized["ReserveCrewAssignmentCostUSD"]
    )

    baseline_cost = fcfs[
        "TotalModeledRecoveryCostUSD"
    ]

    percent_difference = (
        100 * financial_difference / baseline_cost
        if baseline_cost > 0
        else 0.0
    )

    if financial_difference > 0:
        lower_cost_policy = "AircraftAware"
    elif financial_difference < 0:
        lower_cost_policy = "FCFS"
    else:
        lower_cost_policy = "Tie"

    comparison = {
        "BaselinePolicy": "FCFS",
        "AlternativePolicy": "AircraftAware",

        "FCFSTotalCostUSD": float(
            fcfs["TotalModeledRecoveryCostUSD"]
        ),

        "AircraftAwareTotalCostUSD": float(
            optimized["TotalModeledRecoveryCostUSD"]
        ),

        "IncrementalDelayMinutesAvoided": int(
            delay_difference
        ),

        "OperationalDelayCostDifferenceUSD": float(
            delay_cost_difference
        ),

        "RebookingCostDifferenceUSD": float(
            rebooking_difference
        ),

        "PassengerAssistanceCostDifferenceUSD": float(
            assistance_difference
        ),

        "CrewCostDifferenceUSD": float(
            crew_difference
        ),

        "NetModeledCostReductionUSD": float(
            financial_difference
        ),

        "NetModeledCostReductionPercent": float(
            percent_difference
        ),

        "LowerModeledCostPolicy": lower_cost_policy,
    }

    component_sum = (
        delay_cost_difference
        + rebooking_difference
        + assistance_difference
        + crew_difference
    )

    if abs(component_sum - financial_difference) > 0.01:
        raise RuntimeError(
            "Financial components do not reconcile."
        )

    return pd.DataFrame([comparison])


# ============================================================
# 6. SAVE REPORTS
# ============================================================

def save_reports(costs, comparison):

    cost_file = (
        REPORT_DIR / "recovery_financial_cost_baseline.csv"
    )

    comparison_file = (
        REPORT_DIR / "recovery_financial_comparison.csv"
    )

    assumption_file = (
        REPORT_DIR / "financial_cost_assumptions.json"
    )

    costs.to_csv(
        cost_file,
        index=False,
    )

    comparison.to_csv(
        comparison_file,
        index=False,
    )

    with open(
        assumption_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                "FinancialAssumptions": COST_ASSUMPTIONS,
                "SimulationInputs": SIMULATION_RESULTS,
                "HistoricalPassengerInput": str(
                    PASSENGER_FILE
                ),
                "Limitations": [
                    "Dollar assumptions are illustrative.",
                    "Passengers and seat inventory are synthetic.",
                    "Network delay is from a simulated scenario.",
                    "The operational delay rate excludes "
                    "separately modeled passenger expenses.",
                    "Crew assignment counts are synthetic.",
                    "This is not audited airline cost data.",
                    "Financial savings are scenario estimates, "
                    "not realized cash savings.",
                ],
            },
            file,
            indent=4,
        )

    return (
        cost_file,
        comparison_file,
        assumption_file,
    )


# ============================================================
# 7. PRINT EXECUTIVE FINANCIAL SUMMARY
# ============================================================

def display_results(costs, comparison):

    print("\n" + "=" * 70)
    print("FINANCIAL RECOVERY COST COMPARISON")
    print("=" * 70)

    display_columns = [
        "Policy",
        "OperationalDelayCostUSD",
        "PassengerRebookingCostUSD",
        "PassengerAssistanceCostUSD",
        "ReserveCrewAssignmentCostUSD",
        "TotalModeledRecoveryCostUSD",
    ]

    print(
        costs[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:,.2f}",
        )
    )

    result = comparison.iloc[0]

    print("\n" + "=" * 70)
    print("EXECUTIVE FINANCIAL INSIGHTS")
    print("=" * 70)

    print(
        "FCFS modeled recovery cost: "
        f"${result['FCFSTotalCostUSD']:,.2f}"
    )

    print(
        "Aircraft-aware modeled recovery cost: "
        f"${result['AircraftAwareTotalCostUSD']:,.2f}"
    )

    print(
        "Incremental network delay avoided: "
        f"{result['IncrementalDelayMinutesAvoided']:,} minutes"
    )

    print(
        "Net modeled cost reduction: "
        f"${result['NetModeledCostReductionUSD']:,.2f}"
    )

    print(
        "Net modeled cost reduction percentage: "
        f"{result['NetModeledCostReductionPercent']:.2f}%"
    )

    print(
        "Lower modeled cost strategy: "
        f"{result['LowerModeledCostPolicy']}"
    )

    print("\nIMPORTANT INTERPRETATION")

    print(
        "These dollar estimates use hypothetical "
        "unit costs, not verified airline expenses."
    )

    print(
        "The financial comparison uses outputs from "
        "our simulated aircraft and passenger recovery."
    )

    print(
        "The network-delay cost assumption is "
        "defined to exclude passenger assistance "
        "and rebooking expenses."
    )

    print(
        "Results are sensitive to the assumptions; "
        "they do not demonstrate realized savings."
    )


# ============================================================
# 8. MAIN PROGRAM
# ============================================================

def main():

    print("=" * 70)
    print("AERORESILIENCE AI")
    print("MODULE 4 — FINANCIAL IMPACT & RECOVERY ANALYTICS")
    print("STEP 42 — FINANCIAL RECOVERY COST BASELINE")
    print("=" * 70)

    passengers = load_passenger_results()

    costs = calculate_financial_costs(
        passengers
    )

    comparison = compare_policies(
        costs
    )

    files = save_reports(
        costs,
        comparison,
    )

    display_results(
        costs,
        comparison,
    )

    print("\nREPORTS GENERATED SUCCESSFULLY")

    for file in files:
        print(f"Saved: {file}")

    print(
        "\nAERORESILIENCE AI — "
        "FINANCIAL COST BASELINE COMPLETE"
    )


if __name__ == "__main__":
    main()
