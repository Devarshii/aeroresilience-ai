
from pathlib import Path
from itertools import product
import json

import numpy as np
import pandas as pd


# ============================================================
# AERORESILIENCE AI
# MODULE 4 — FINANCIAL IMPACT & RECOVERY ANALYTICS
# STEP 43 — FINANCIAL SENSITIVITY ANALYSIS
# ============================================================

REPORT_DIR = Path("outputs/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_FILE = (
    REPORT_DIR / "recovery_financial_cost_baseline.csv"
)

ASSUMPTIONS_FILE = (
    REPORT_DIR / "financial_cost_assumptions.json"
)

# Experimental scenario assumptions, not airline benchmarks.
DELAY_RATES = [0, 10, 25, 45, 70, 100]
REBOOKING_RATES = [40, 80, 120]
ASSISTANCE_RATES = [100, 250, 500, 1000]

POLICIES = ["FCFS", "AircraftAware"]


# ============================================================
# 1. LOAD STEP 42 OUTPUTS
# ============================================================

def load_baseline():

    print("\nLoading Step 42 financial results...")

    if not BASELINE_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {BASELINE_FILE}"
        )

    if not ASSUMPTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {ASSUMPTIONS_FILE}"
        )

    baseline = pd.read_csv(BASELINE_FILE)

    with open(
        ASSUMPTIONS_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        configuration = json.load(file)

    assumptions = configuration["FinancialAssumptions"]

    required = [
        "Policy",
        "IncrementalNetworkDelayMinutes",
        "PassengersRebooked",
        "PassengersUnaccommodated",
        "SyntheticReserveCrewAssignments",
    ]

    missing = [
        column for column in required
        if column not in baseline.columns
    ]

    if missing:
        raise ValueError(
            f"Missing baseline columns: {missing}"
        )

    if (
        len(baseline) != 2
        or set(baseline["Policy"]) != set(POLICIES)
        or baseline["Policy"].duplicated().any()
    ):
        raise ValueError(
            "Expected one row for FCFS and AircraftAware."
        )

    numeric_columns = required[1:]

    for column in numeric_columns:
        baseline[column] = pd.to_numeric(
            baseline[column],
            errors="raise",
        )

        if (
            baseline[column].isna().any()
            or (baseline[column] < 0).any()
        ):
            raise ValueError(
                f"Invalid values in {column}"
            )

    print("Baseline loaded successfully.")

    return (
        baseline.set_index("Policy"),
        assumptions,
    )


# ============================================================
# 2. CALCULATE SCENARIO FINANCIAL COST
# ============================================================

def calculate_cost(
    policy_data,
    delay_rate,
    rebooking_rate,
    assistance_rate,
    crew_rate,
):

    operational = (
        policy_data["IncrementalNetworkDelayMinutes"]
        * delay_rate
    )

    rebooking = (
        policy_data["PassengersRebooked"]
        * rebooking_rate
    )

    assistance = (
        policy_data["PassengersUnaccommodated"]
        * assistance_rate
    )

    crew = (
        policy_data["SyntheticReserveCrewAssignments"]
        * crew_rate
    )

    return float(
        operational
        + rebooking
        + assistance
        + crew
    )


# ============================================================
# 3. RUN SCENARIO GRID
# ============================================================

def run_sensitivity(baseline, assumptions):

    crew_rate = float(
        assumptions["ReserveCrewAssignmentCost"]
    )

    records = []

    scenario_number = 0

    for delay_rate, rebooking_rate, assistance_rate in product(
        DELAY_RATES,
        REBOOKING_RATES,
        ASSISTANCE_RATES,
    ):

        scenario_number += 1

        fcfs_cost = calculate_cost(
            baseline.loc["FCFS"],
            delay_rate,
            rebooking_rate,
            assistance_rate,
            crew_rate,
        )

        aircraft_cost = calculate_cost(
            baseline.loc["AircraftAware"],
            delay_rate,
            rebooking_rate,
            assistance_rate,
            crew_rate,
        )

        difference = fcfs_cost - aircraft_cost

        if difference > 0.000001:
            winner = "AircraftAware"
        elif difference < -0.000001:
            winner = "FCFS"
        else:
            winner = "Tie"

        records.append({
            "ScenarioID": scenario_number,
            "DelayCostPerMinuteUSD": delay_rate,
            "RebookingCostPerPassengerUSD": rebooking_rate,
            "AssistanceCostPerPassengerUSD": assistance_rate,
            "ReserveCrewCostUSD": crew_rate,
            "FCFSTotalCostUSD": fcfs_cost,
            "AircraftAwareTotalCostUSD": aircraft_cost,
            "NetCostReductionUSD": difference,
            "NetCostReductionPercent": (
                100 * difference / fcfs_cost
                if fcfs_cost > 0 else np.nan
            ),
            "LowerCostPolicy": winner,
        })

    return pd.DataFrame(records)


# ============================================================
# 4. BREAK-EVEN ANALYSIS
# ============================================================

def break_even_analysis(baseline, assumptions):

    fcfs = baseline.loc["FCFS"]
    aircraft = baseline.loc["AircraftAware"]

    delay_minutes_avoided = float(
        fcfs["IncrementalNetworkDelayMinutes"]
        - aircraft["IncrementalNetworkDelayMinutes"]
    )

    rebooking_difference = float(
        fcfs["PassengersRebooked"]
        - aircraft["PassengersRebooked"]
    )

    assistance_difference = float(
        fcfs["PassengersUnaccommodated"]
        - aircraft["PassengersUnaccommodated"]
    )

    crew_difference = float(
        fcfs["SyntheticReserveCrewAssignments"]
        - aircraft["SyntheticReserveCrewAssignments"]
    )

    rebooking_rate = float(
        assumptions["PassengerRebookingProcessingCost"]
    )

    assistance_rate = float(
        assumptions["UnaccommodatedPassengerAssistanceCost"]
    )

    crew_rate = float(
        assumptions["ReserveCrewAssignmentCost"]
    )

    other_cost_difference = (
        rebooking_difference * rebooking_rate
        + assistance_difference * assistance_rate
        + crew_difference * crew_rate
    )

    # Net cost reduction =
    # delay_minutes_avoided * delay_rate
    # + other_cost_difference
    #
    # Break-even occurs when reduction = 0.

    if abs(delay_minutes_avoided) < 1e-12:
        threshold = np.nan
        interpretation = (
            "No delay-minute difference; "
            "a delay-rate break-even is undefined."
        )
    else:
        threshold = (
            -other_cost_difference
            / delay_minutes_avoided
        )

        if threshold < 0:
            interpretation = (
                "Aircraft-aware has lower modeled costs "
                "for every nonnegative delay rate "
                "under these fixed other assumptions."
            )
        else:
            interpretation = (
                "At this delay cost per minute, "
                "the two policies have equal "
                "modeled total costs."
            )

    return pd.DataFrame([{
        "DelayMinutesAvoided": delay_minutes_avoided,
        "NonDelayCostDifferenceUSD": other_cost_difference,
        "BreakEvenDelayCostPerMinuteUSD": threshold,
        "Interpretation": interpretation,
    }])


# ============================================================
# 5. SUMMARIZE RESULTS
# ============================================================

def summarize_scenarios(scenarios):

    summary = (
        scenarios.groupby(
            "LowerCostPolicy",
            dropna=False,
        )
        .agg(
            ScenarioCount=("ScenarioID", "count"),
            MinimumNetReductionUSD=(
                "NetCostReductionUSD",
                "min",
            ),
            MaximumNetReductionUSD=(
                "NetCostReductionUSD",
                "max",
            ),
        )
        .reset_index()
    )

    return summary


# ============================================================
# 6. MAIN PROGRAM
# ============================================================

def main():

    print("=" * 70)
    print("AERORESILIENCE AI")
    print("MODULE 4 — FINANCIAL SENSITIVITY ANALYSIS")
    print("=" * 70)

    baseline, assumptions = load_baseline()

    scenarios = run_sensitivity(
        baseline,
        assumptions,
    )

    break_even = break_even_analysis(
        baseline,
        assumptions,
    )

    summary = summarize_scenarios(scenarios)

    # Save outputs.
    scenario_file = (
        REPORT_DIR / "financial_sensitivity_scenarios.csv"
    )

    summary_file = (
        REPORT_DIR / "financial_sensitivity_summary.csv"
    )

    break_even_file = (
        REPORT_DIR / "financial_break_even_analysis.csv"
    )

    scenarios.to_csv(
        scenario_file,
        index=False,
    )

    summary.to_csv(
        summary_file,
        index=False,
    )

    break_even.to_csv(
        break_even_file,
        index=False,
    )

    print("\n" + "=" * 70)
    print("FINANCIAL SENSITIVITY RESULTS")
    print("=" * 70)

    print(
        f"\nTotal financial scenarios tested: "
        f"{len(scenarios):,}"
    )

    print("\nLOWER-COST STRATEGY BY SCENARIO")

    print(
        summary.to_string(index=False)
    )

    print("\nNET MODELED COST REDUCTION RANGE")

    print(
        f"Minimum: "
        f"${scenarios['NetCostReductionUSD'].min():,.2f}"
    )

    print(
        f"Maximum: "
        f"${scenarios['NetCostReductionUSD'].max():,.2f}"
    )

    print("\nBREAK-EVEN ANALYSIS")

    result = break_even.iloc[0]

    print(
        "Delay minutes avoided by aircraft-aware: "
        f"{result['DelayMinutesAvoided']:,.0f}"
    )

    print(
        "Non-delay cost difference (FCFS minus aircraft-aware): "
        f"${result['NonDelayCostDifferenceUSD']:,.2f}"
    )

    threshold = result[
        "BreakEvenDelayCostPerMinuteUSD"
    ]

    if pd.notna(threshold):
        print(
            "Break-even delay cost per minute: "
            f"${threshold:,.4f}"
        )
    else:
        print(
            "Break-even delay cost per minute: undefined"
        )

    print(result["Interpretation"])

    print("\nIMPORTANT LIMITATIONS")
    print(
        "All scenario unit costs are hypothetical, "
        "not verified airline costs."
    )
    print(
        "Aircraft delays and passenger recovery outcomes "
        "are held constant across sensitivity scenarios."
    )
    print(
        "Only financial assumptions change; "
        "the recovery strategies are not re-optimized."
    )
    print(
        "Crew, passenger, and aircraft constraints "
        "remain simplified."
    )
    print(
        "The resulting dollar differences are "
        "modeled estimates, not realized savings."
    )

    print("\nREPORTS SAVED")
    print(scenario_file)
    print(summary_file)
    print(break_even_file)

    print(
        "\nAERORESILIENCE AI — "
        "FINANCIAL SENSITIVITY ANALYSIS COMPLETE"
    )


if __name__ == "__main__":
    main()
