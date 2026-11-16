
from pathlib import Path
from itertools import product
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ============================================================
# AERORESILIENCE AI
# MODULE 5 — RESILIENCE & WHAT-IF SIMULATOR
# STEP 48 — WHAT-IF SCENARIO ENGINE
# ============================================================

REPORT_DIR = Path("outputs/reports")
FIGURE_DIR = Path("outputs/figures")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = (
    REPORT_DIR / "integrated_resilience_scenarios.csv"
)

ASSUMPTIONS_FILE = (
    REPORT_DIR / "financial_cost_assumptions.json"
)

POLICIES = ["FCFS", "AircraftAware"]

# These are illustrative cost scenarios,
# NOT verified airline industry cost estimates.
DELAY_RATES = [10, 25, 45, 70, 100]
REBOOKING_RATES = [40, 80, 120]
ASSISTANCE_RATES = [100, 250, 500]


# ============================================================
# 1. LOAD AND VALIDATE PREVIOUS RESULTS
# ============================================================

def load_inputs():

    print("\nLoading integrated resilience results...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Run Step 47 first: {INPUT_FILE}"
        )

    if not ASSUMPTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Missing assumptions: {ASSUMPTIONS_FILE}"
        )

    data = pd.read_csv(INPUT_FILE)

    with open(
        ASSUMPTIONS_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        configuration = json.load(file)

    assumptions = configuration["FinancialAssumptions"]

    required = [
        "MinimumTurnMinutes",
        "Policy",
        "AircraftIncrementalDelayMinutes",
        "PassengersAtRiskBaseline",
        "PassengersRebookedBaseline",
        "PassengersUnaccommodatedBaseline",
        "ReserveCrewAssignmentsBaseline",
        "PassengerOutcomesRecomputed",
        "CrewAssignmentsRecomputed",
    ]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing integrated columns: {missing}"
        )

    numeric = [
        "MinimumTurnMinutes",
        "AircraftIncrementalDelayMinutes",
        "PassengersAtRiskBaseline",
        "PassengersRebookedBaseline",
        "PassengersUnaccommodatedBaseline",
        "ReserveCrewAssignmentsBaseline",
    ]

    for column in numeric:
        data[column] = pd.to_numeric(
            data[column],
            errors="raise",
        )

    if (
        data[numeric].isna().any().any()
        or (data[numeric] < 0).any().any()
    ):
        raise ValueError(
            "Invalid integrated scenario quantities."
        )

    if data.duplicated(
        ["MinimumTurnMinutes", "Policy"]
    ).any():
        raise ValueError(
            "Duplicate turnaround-policy combinations."
        )

    # This step must not mistake baseline passenger
    # or crew data for recomputed scenario outcomes.
    for column in [
        "PassengerOutcomesRecomputed",
        "CrewAssignmentsRecomputed",
    ]:
        values = (
            data[column]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        if values.eq("true").any():
            raise ValueError(
                f"{column} unexpectedly contains True. "
                "Review the integration methodology."
            )

    passenger_total = (
        data["PassengersRebookedBaseline"]
        + data["PassengersUnaccommodatedBaseline"]
    )

    if not (
        passenger_total
        == data["PassengersAtRiskBaseline"]
    ).all():
        raise ValueError(
            "Passenger totals do not reconcile."
        )

    for turnaround, group in data.groupby(
        "MinimumTurnMinutes"
    ):
        if (
            len(group) != 2
            or set(group["Policy"]) != set(POLICIES)
        ):
            raise ValueError(
                "Incomplete recovery policy pair for "
                f"turnaround {turnaround}."
            )

    crew_rate = float(
        assumptions["ReserveCrewAssignmentCost"]
    )

    if not np.isfinite(crew_rate) or crew_rate < 0:
        raise ValueError(
            "Invalid crew assignment cost assumption."
        )

    print(
        "Matched turnaround settings:",
        sorted(data["MinimumTurnMinutes"].unique()),
    )

    print(
        "Integrated source records:",
        len(data),
    )

    return data, crew_rate


# ============================================================
# 2. COMPUTE A HYPOTHETICAL COST SCENARIO
# ============================================================

def calculate_cost(
    row,
    delay_rate,
    rebooking_rate,
    assistance_rate,
    crew_rate,
):

    return float(
        row["AircraftIncrementalDelayMinutes"]
        * delay_rate

        + row["PassengersRebookedBaseline"]
        * rebooking_rate

        + row["PassengersUnaccommodatedBaseline"]
        * assistance_rate

        + row["ReserveCrewAssignmentsBaseline"]
        * crew_rate
    )


# ============================================================
# 3. GENERATE WHAT-IF SCENARIOS
# ============================================================

def generate_scenarios(data, crew_rate):

    print("\nGenerating what-if financial scenarios...")

    records = []
    scenario_id = 0

    for turnaround, group in data.groupby(
        "MinimumTurnMinutes"
    ):

        lookup = group.set_index("Policy")

        fcfs = lookup.loc["FCFS"]
        aircraft = lookup.loc["AircraftAware"]

        for (
            delay_rate,
            rebooking_rate,
            assistance_rate,
        ) in product(
            DELAY_RATES,
            REBOOKING_RATES,
            ASSISTANCE_RATES,
        ):

            scenario_id += 1

            fcfs_cost = calculate_cost(
                fcfs,
                delay_rate,
                rebooking_rate,
                assistance_rate,
                crew_rate,
            )

            aircraft_cost = calculate_cost(
                aircraft,
                delay_rate,
                rebooking_rate,
                assistance_rate,
                crew_rate,
            )

            reduction = (
                fcfs_cost - aircraft_cost
            )

            if reduction > 0.000001:
                preferred = "AircraftAware"
            elif reduction < -0.000001:
                preferred = "FCFS"
            else:
                preferred = "Tie"

            records.append({
                "ScenarioID": scenario_id,
                "MinimumTurnMinutes": turnaround,
                "DelayCostPerMinuteUSD": delay_rate,
                "RebookingCostPerPassengerUSD": (
                    rebooking_rate
                ),
                "AssistanceCostPerPassengerUSD": (
                    assistance_rate
                ),
                "ReserveCrewAssignmentCostUSD": (
                    crew_rate
                ),
                "FCFSModeledCostUSD": fcfs_cost,
                "AircraftAwareModeledCostUSD": (
                    aircraft_cost
                ),
                "CostReductionUSD": reduction,
                "CostReductionPercent": (
                    100 * reduction / fcfs_cost
                    if fcfs_cost > 0 else np.nan
                ),
                "LowerCostPolicy": preferred,
                "AircraftDelayMinutesAvoided": (
                    fcfs["AircraftIncrementalDelayMinutes"]
                    - aircraft[
                        "AircraftIncrementalDelayMinutes"
                    ]
                ),
                "PassengerOutcomesRecomputed": False,
                "CrewAssignmentsRecomputed": False,
            })

    result = pd.DataFrame(records)

    expected_count = (
        data["MinimumTurnMinutes"].nunique()
        * len(DELAY_RATES)
        * len(REBOOKING_RATES)
        * len(ASSISTANCE_RATES)
    )

    if len(result) != expected_count:
        raise RuntimeError(
            "What-if scenario count does not reconcile."
        )

    print(
        f"What-if scenarios generated: {len(result):,}"
    )

    return result


# ============================================================
# 4. SUMMARIZE ROBUSTNESS
# ============================================================

def summarize_results(scenarios):

    summary = (
        scenarios.groupby(
            "MinimumTurnMinutes"
        )
        .agg(
            Scenarios=("ScenarioID", "count"),
            MinCostReductionUSD=(
                "CostReductionUSD", "min"
            ),
            AverageCostReductionUSD=(
                "CostReductionUSD", "mean"
            ),
            MaxCostReductionUSD=(
                "CostReductionUSD", "max"
            ),
            AircraftAwarePreferred=(
                "LowerCostPolicy",
                lambda values: int(
                    (values == "AircraftAware").sum()
                ),
            ),
            FCFSPreferred=(
                "LowerCostPolicy",
                lambda values: int(
                    (values == "FCFS").sum()
                ),
            ),
            Ties=(
                "LowerCostPolicy",
                lambda values: int(
                    (values == "Tie").sum()
                ),
            ),
        )
        .reset_index()
    )

    return summary


# ============================================================
# 5. CREATE INTERACTIVE HEATMAP
# ============================================================

def create_heatmap(scenarios):

    # Hold passenger-related unit costs at their
    # Step 42 illustrative baseline values.
    selected = scenarios[
        (
            scenarios[
                "RebookingCostPerPassengerUSD"
            ] == 80
        )
        & (
            scenarios[
                "AssistanceCostPerPassengerUSD"
            ] == 250
        )
    ].copy()

    heatmap = selected.pivot(
        index="MinimumTurnMinutes",
        columns="DelayCostPerMinuteUSD",
        values="CostReductionUSD",
    ).sort_index()

    fig = go.Figure(
        data=go.Heatmap(
            x=heatmap.columns.tolist(),
            y=heatmap.index.tolist(),
            z=heatmap.to_numpy(),
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(
                title="FCFS minus aircraft-aware (USD)"
            ),
            hovertemplate=(
                "Minimum turn: %{y} minutes"
                "<br>Delay rate: $%{x}/minute"
                "<br>Cost reduction: $%{z:,.0f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=(
            "AeroResilience AI — "
            "Recovery Strategy What-If Heatmap"
        ),
        xaxis_title=(
            "Hypothetical Operational Delay Cost "
            "(USD per Minute)"
        ),
        yaxis_title=(
            "Minimum Aircraft Turnaround (Minutes)"
        ),
        template="plotly_white",
        height=600,
        annotations=[
            dict(
                x=0.5,
                y=-0.22,
                xref="paper",
                yref="paper",
                text=(
                    "Passenger and crew outcomes held "
                    "fixed. Positive values favor "
                    "aircraft-aware recovery."
                ),
                showarrow=False,
            )
        ],
        margin=dict(
            l=90,
            r=100,
            t=90,
            b=140,
        ),
    )

    output_file = (
        FIGURE_DIR
        / "resilience_what_if_heatmap.html"
    )

    fig.write_html(
        str(output_file),
        include_plotlyjs=True,
    )

    return output_file


# ============================================================
# 6. SAVE METHODOLOGY
# ============================================================

def save_methodology(
    scenarios,
    crew_rate,
):

    methodology = {
        "Project": "AeroResilience AI",
        "Module": 5,
        "Step": 48,
        "Method": (
            "Financial what-if sensitivity across "
            "existing aircraft turnaround simulations."
        ),
        "TotalScenarios": len(scenarios),
        "DelayRatesUSDPerMinute": DELAY_RATES,
        "RebookingRatesUSD": REBOOKING_RATES,
        "AssistanceRatesUSD": ASSISTANCE_RATES,
        "ReserveCrewRateUSD": crew_rate,
        "Limitations": [
            "All financial unit rates are hypothetical.",
            "Aircraft outcomes are reused from "
            "previous simulations.",
            "Passenger outcomes are fixed "
            "policy-specific baseline values.",
            "Crew assignments are fixed "
            "policy-specific baseline values.",
            "The scenarios do not model changes "
            "in passenger itineraries or crew legality.",
            "Financial sensitivities do not "
            "re-optimize recovery schedules.",
            "No actual airline savings are claimed.",
        ],
    }

    output_file = (
        REPORT_DIR
        / "resilience_what_if_methodology.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            methodology,
            file,
            indent=4,
        )

    return output_file


# ============================================================
# 7. MAIN PROGRAM
# ============================================================

def main():

    print("=" * 70)
    print("AERORESILIENCE AI")
    print("MODULE 5 — RESILIENCE & WHAT-IF SIMULATOR")
    print("STEP 48 — WHAT-IF SCENARIO ENGINE")
    print("=" * 70)

    data, crew_rate = load_inputs()

    scenarios = generate_scenarios(
        data,
        crew_rate,
    )

    summary = summarize_results(scenarios)

    scenario_file = (
        REPORT_DIR
        / "resilience_what_if_scenarios.csv"
    )

    summary_file = (
        REPORT_DIR
        / "resilience_what_if_summary.csv"
    )

    scenarios.to_csv(
        scenario_file,
        index=False,
    )

    summary.to_csv(
        summary_file,
        index=False,
    )

    chart_file = create_heatmap(scenarios)

    methodology_file = save_methodology(
        scenarios,
        crew_rate,
    )

    print("\nWHAT-IF SCENARIO RESULTS")

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:,.2f}",
        )
    )

    print("\nOVERALL FINANCIAL ROBUSTNESS")

    print(
        "Total scenarios:",
        len(scenarios),
    )

    print(
        "Aircraft-aware lower-cost scenarios:",
        int(
            (
                scenarios["LowerCostPolicy"]
                == "AircraftAware"
            ).sum()
        ),
    )

    print(
        "FCFS lower-cost scenarios:",
        int(
            (
                scenarios["LowerCostPolicy"]
                == "FCFS"
            ).sum()
        ),
    )

    print(
        "Tied scenarios:",
        int(
            (
                scenarios["LowerCostPolicy"]
                == "Tie"
            ).sum()
        ),
    )

    print("\nIMPORTANT LIMITATIONS")

    print(
        "Passenger and crew outcomes remain fixed "
        "within each policy."
    )

    print(
        "This analysis varies hypothetical "
        "cost assumptions rather than "
        "rerunning operational recovery."
    )

    print(
        "Results are illustrative model comparisons, "
        "not actual airline savings."
    )

    print("\nFILES SAVED")

    for path in [
        scenario_file,
        summary_file,
        chart_file,
        methodology_file,
    ]:
        print(path)

    print(
        "\nAERORESILIENCE AI — "
        "WHAT-IF SCENARIO ENGINE COMPLETE"
    )


if __name__ == "__main__":
    main()
