
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ============================================================
# AERORESILIENCE AI
# MODULE 5 — RESILIENCE & WHAT-IF SIMULATOR
# STEP 47 — INTEGRATED RESILIENCE ANALYSIS
# COMPLETE CORRECTED VERSION
# ============================================================

REPORT_DIR = Path("outputs/reports")
FIGURE_DIR = Path("outputs/figures")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

AIRCRAFT_FILE = (
    REPORT_DIR / "recovery_sensitivity_comparison.csv"
)

PASSENGER_FILE = (
    REPORT_DIR / "passenger_rebooking_summary.csv"
)

FINANCIAL_FILE = (
    REPORT_DIR / "recovery_financial_cost_baseline.csv"
)

ASSUMPTIONS_FILE = (
    REPORT_DIR / "financial_cost_assumptions.json"
)

STRESS_FILE = (
    REPORT_DIR / "stress_test_airport_summary.csv"
)

POLICIES = ["FCFS", "AircraftAware"]

AIRCRAFT_REQUIRED = [
    "MinimumTurnMinutes",
    "Policy",
    "NoStopDelayMinutes",
    "TotalModeledDelayMinutes",
    "IncrementalDelayMinutes",
]

PASSENGER_REQUIRED = [
    "Policy",
    "SimulatedPassengersAtRisk",
    "PassengersRebooked",
    "PassengersUnaccommodated",
]

FINANCIAL_REQUIRED = [
    "Policy",
    "SyntheticReserveCrewAssignments",
    "TotalModeledRecoveryCostUSD",
]

FINANCIAL_RATE_FIELDS = [
    "OperationalCostPerIncrementalDelayMinute",
    "PassengerRebookingProcessingCost",
    "UnaccommodatedPassengerAssistanceCost",
    "ReserveCrewAssignmentCost",
]


# ============================================================
# 1. NORMALIZE RECOVERY POLICY NAMES
# ============================================================

def normalize_policy(value):
    """
    Standardize policy labels across previous modules.

    Examples:
        FCFS -> FCFS
        First Come First Served -> FCFS
        AircraftAware -> AircraftAware
        Aircraft-Aware Optimized -> AircraftAware

    Other recovery policies retain their original names.
    """

    original = str(value).strip()

    normalized = re.sub(
        r"[^a-z0-9]+",
        "",
        original.lower(),
    )

    if normalized in {
        "fcfs",
        "firstcomefirstserved",
        "firstinfirstout",
    }:
        return "FCFS"

    if (
        "aircraftaware" in normalized
        or normalized in {
            "aircraftoptimized",
            "aircraftawareoptimized",
        }
    ):
        return "AircraftAware"

    return original


def standardize_policy_column(
    dataframe,
    source_name,
):

    dataframe = dataframe.copy()

    if "Policy" not in dataframe.columns:
        raise ValueError(
            f"{source_name} is missing Policy column."
        )

    dataframe["OriginalPolicyLabel"] = (
        dataframe["Policy"].astype(str)
    )

    dataframe["Policy"] = (
        dataframe["Policy"]
        .map(normalize_policy)
    )

    print(f"\n{source_name} recovery policies:")

    labels = (
        dataframe[
            [
                "OriginalPolicyLabel",
                "Policy",
            ]
        ]
        .drop_duplicates()
        .sort_values("OriginalPolicyLabel")
    )

    print(labels.to_string(index=False))

    return dataframe


# ============================================================
# 2. LOAD EXISTING PROJECT REPORTS
# ============================================================

def load_inputs():

    print("\nLoading existing AeroResilience outputs...")

    required_files = {
        "Aircraft sensitivity": AIRCRAFT_FILE,
        "Passenger recovery": PASSENGER_FILE,
        "Financial baseline": FINANCIAL_FILE,
        "Financial assumptions": ASSUMPTIONS_FILE,
    }

    for label, path in required_files.items():

        if not path.exists():
            raise FileNotFoundError(
                f"{label} not found: {path}"
            )

    aircraft = pd.read_csv(AIRCRAFT_FILE)

    passengers = pd.read_csv(PASSENGER_FILE)

    financial = pd.read_csv(FINANCIAL_FILE)

    with open(
        ASSUMPTIONS_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        configuration = json.load(file)

    assumptions = configuration["FinancialAssumptions"]

    print(
        f"\nAircraft sensitivity records: {len(aircraft):,}"
    )

    print(
        f"Passenger policy records: {len(passengers):,}"
    )

    print(
        f"Financial policy records: {len(financial):,}"
    )

    aircraft = standardize_policy_column(
        aircraft,
        "Aircraft sensitivity",
    )

    passengers = standardize_policy_column(
        passengers,
        "Passenger recovery",
    )

    financial = standardize_policy_column(
        financial,
        "Financial baseline",
    )

    return (
        aircraft,
        passengers,
        financial,
        assumptions,
    )


# ============================================================
# 3. VALIDATE AND PREPARE DATA
# ============================================================

def prepare_inputs(
    aircraft,
    passengers,
    financial,
    assumptions,
):

    print("\nValidating integrated scenario inputs...")

    for column in AIRCRAFT_REQUIRED:

        if column not in aircraft.columns:
            raise ValueError(
                f"Aircraft report missing: {column}"
            )

    for column in PASSENGER_REQUIRED:

        if column not in passengers.columns:
            raise ValueError(
                f"Passenger report missing: {column}"
            )

    for column in FINANCIAL_REQUIRED:

        if column not in financial.columns:
            raise ValueError(
                f"Financial report missing: {column}"
            )

    numeric_aircraft = [
        "MinimumTurnMinutes",
        "NoStopDelayMinutes",
        "TotalModeledDelayMinutes",
        "IncrementalDelayMinutes",
    ]

    for column in numeric_aircraft:

        aircraft[column] = pd.to_numeric(
            aircraft[column],
            errors="raise",
        )

    if aircraft[numeric_aircraft].isna().any().any():
        raise ValueError(
            "Aircraft sensitivity data contains missing values."
        )

    if (aircraft[numeric_aircraft] < 0).any().any():
        raise ValueError(
            "Aircraft sensitivity data contains negative values."
        )

    expected_incremental = (
        aircraft["TotalModeledDelayMinutes"]
        - aircraft["NoStopDelayMinutes"]
    )

    if not np.allclose(
        expected_incremental,
        aircraft["IncrementalDelayMinutes"],
        atol=0.01,
        rtol=0,
    ):
        raise ValueError(
            "Aircraft delay calculations do not reconcile."
        )

    aircraft = aircraft[
        aircraft["Policy"].isin(POLICIES)
    ].copy()

    print(
        "\nTarget aircraft-policy records:",
        len(aircraft),
    )

    if aircraft.empty:
        raise ValueError(
            "No recognized FCFS or aircraft-aware "
            "recovery policies were found. "
            "Review the printed original policy labels."
        )

    if aircraft.duplicated(
        ["MinimumTurnMinutes", "Policy"]
    ).any():

        duplicates = aircraft.loc[
            aircraft.duplicated(
                ["MinimumTurnMinutes", "Policy"],
                keep=False,
            ),
            [
                "MinimumTurnMinutes",
                "OriginalPolicyLabel",
                "Policy",
            ],
        ]

        print("\nDuplicate scenarios:")
        print(duplicates.to_string(index=False))

        raise ValueError(
            "Duplicate turnaround-policy combinations "
            "after policy normalization."
        )

    # Each turnaround setting must have both policies.
    policy_counts = (
        aircraft.groupby("MinimumTurnMinutes")["Policy"]
        .nunique()
    )

    complete_turns = policy_counts[
        policy_counts == len(POLICIES)
    ].index

    print(
        "Matched turnaround settings:",
        list(complete_turns),
    )

    aircraft = aircraft[
        aircraft["MinimumTurnMinutes"].isin(
            complete_turns
        )
    ].copy()

    if aircraft.empty:

        print("\nAvailable turnaround-policy pairs:")

        print(
            policy_counts.to_string()
        )

        raise ValueError(
            "No turnaround setting contains both "
            "FCFS and AircraftAware policies. "
            "Check the policy mapping above."
        )

    # Validate passenger and financial policies.
    for dataset, label in [
        (passengers, "Passenger"),
        (financial, "Financial"),
    ]:

        selected = dataset[
            dataset["Policy"].isin(POLICIES)
        ]

        if (
            len(selected) != 2
            or selected["Policy"].duplicated().any()
            or set(selected["Policy"]) != set(POLICIES)
        ):

            raise ValueError(
                f"{label} data must have exactly "
                "one FCFS and one AircraftAware row."
            )

    numeric_passenger = PASSENGER_REQUIRED[1:]

    for column in numeric_passenger:

        passengers[column] = pd.to_numeric(
            passengers[column],
            errors="raise",
        )

    financial["SyntheticReserveCrewAssignments"] = (
        pd.to_numeric(
            financial["SyntheticReserveCrewAssignments"],
            errors="raise",
        )
    )

    for field in FINANCIAL_RATE_FIELDS:

        if field not in assumptions:
            raise ValueError(
                f"Missing financial assumption: {field}"
            )

        rate = float(assumptions[field])

        if not np.isfinite(rate) or rate < 0:
            raise ValueError(
                f"Invalid financial assumption: {field}"
            )

    for dataset, columns in [
        (passengers, numeric_passenger),
        (
            financial,
            ["SyntheticReserveCrewAssignments"],
        ),
    ]:

        if dataset[columns].isna().any().any():
            raise ValueError(
                "Missing passenger or crew data."
            )

        if (dataset[columns] < 0).any().any():
            raise ValueError(
                "Negative passenger or crew values."
            )

    for row in passengers.itertuples(index=False):

        if (
            row.PassengersRebooked
            + row.PassengersUnaccommodated
            != row.SimulatedPassengersAtRisk
        ):

            raise ValueError(
                f"Passenger counts do not reconcile "
                f"for {row.Policy}"
            )

    print("\nAll integrated input checks passed.")

    return (
        aircraft,
        passengers,
        financial,
    )


# ============================================================
# 4. BUILD INTEGRATED RESILIENCE SCENARIOS
# ============================================================

def build_integrated_scenarios(
    aircraft,
    passengers,
    financial,
    assumptions,
):

    print("\nBuilding integrated resilience scenarios...")

    passenger_lookup = (
        passengers.set_index("Policy")
    )

    financial_lookup = (
        financial.set_index("Policy")
    )

    delay_rate = float(
        assumptions[
            "OperationalCostPerIncrementalDelayMinute"
        ]
    )

    rebooking_rate = float(
        assumptions[
            "PassengerRebookingProcessingCost"
        ]
    )

    assistance_rate = float(
        assumptions[
            "UnaccommodatedPassengerAssistanceCost"
        ]
    )

    crew_rate = float(
        assumptions[
            "ReserveCrewAssignmentCost"
        ]
    )

    records = []

    for row in aircraft.itertuples(index=False):

        policy = row.Policy

        passenger = passenger_lookup.loc[policy]

        finance = financial_lookup.loc[policy]

        incremental_delay = float(
            row.IncrementalDelayMinutes
        )

        passengers_at_risk = int(
            passenger["SimulatedPassengersAtRisk"]
        )

        rebooked = int(
            passenger["PassengersRebooked"]
        )

        unaccommodated = int(
            passenger["PassengersUnaccommodated"]
        )

        crew_assignments = int(
            finance["SyntheticReserveCrewAssignments"]
        )

        operational_cost = (
            incremental_delay * delay_rate
        )

        rebooking_cost = (
            rebooked * rebooking_rate
        )

        assistance_cost = (
            unaccommodated * assistance_rate
        )

        crew_cost = (
            crew_assignments * crew_rate
        )

        total_cost = (
            operational_cost
            + rebooking_cost
            + assistance_cost
            + crew_cost
        )

        records.append({
            "MinimumTurnMinutes": float(
                row.MinimumTurnMinutes
            ),
            "Policy": policy,
            "OriginalAircraftPolicyLabel": (
                row.OriginalPolicyLabel
            ),
            "NoStopDelayMinutes": float(
                row.NoStopDelayMinutes
            ),
            "TotalModeledDelayMinutes": float(
                row.TotalModeledDelayMinutes
            ),
            "AircraftIncrementalDelayMinutes": (
                incremental_delay
            ),
            "PassengersAtRiskBaseline": (
                passengers_at_risk
            ),
            "PassengersRebookedBaseline": rebooked,
            "PassengersUnaccommodatedBaseline": (
                unaccommodated
            ),
            "ReserveCrewAssignmentsBaseline": (
                crew_assignments
            ),
            "OperationalDelayCostUSD": (
                operational_cost
            ),
            "PassengerRebookingCostUSD": (
                rebooking_cost
            ),
            "PassengerAssistanceCostUSD": (
                assistance_cost
            ),
            "ReserveCrewAssignmentCostUSD": crew_cost,
            "IllustrativeCombinedCostUSD": total_cost,
            "PassengerOutcomesRecomputed": False,
            "CrewAssignmentsRecomputed": False,
            "ScenarioType": (
                "Turnaround sensitivity with fixed "
                "passenger and crew baselines"
            ),
        })

    integrated = pd.DataFrame(records)

    if integrated.empty:
        raise ValueError(
            "No integrated scenario records produced."
        )

    integrated = integrated.sort_values(
        ["MinimumTurnMinutes", "Policy"]
    ).reset_index(drop=True)

    print(
        f"Integrated policy-scenario records: "
        f"{len(integrated):,}"
    )

    return integrated


# ============================================================
# 5. COMPARE FCFS VS AIRCRAFT-AWARE
# ============================================================

def compare_scenarios(integrated):

    print("\nComparing recovery policies...")

    records = []

    for turnaround, group in integrated.groupby(
        "MinimumTurnMinutes"
    ):

        group_policies = set(group["Policy"])

        if group_policies != set(POLICIES):
            raise ValueError(
                "Incomplete policy comparison at "
                f"turnaround {turnaround}"
            )

        selected = group.set_index("Policy")

        fcfs = selected.loc["FCFS"]
        optimized = selected.loc["AircraftAware"]

        fcfs_delay = float(
            fcfs["AircraftIncrementalDelayMinutes"]
        )

        optimized_delay = float(
            optimized["AircraftIncrementalDelayMinutes"]
        )

        delay_avoided = (
            fcfs_delay - optimized_delay
        )

        fcfs_cost = float(
            fcfs["IllustrativeCombinedCostUSD"]
        )

        optimized_cost = float(
            optimized["IllustrativeCombinedCostUSD"]
        )

        cost_difference = (
            fcfs_cost - optimized_cost
        )

        if cost_difference > 0.000001:
            preferred = "AircraftAware"

        elif cost_difference < -0.000001:
            preferred = "FCFS"

        else:
            preferred = "Tie"

        records.append({
            "MinimumTurnMinutes": turnaround,
            "FCFSIncrementalDelayMinutes": (
                fcfs_delay
            ),
            "AircraftAwareIncrementalDelayMinutes": (
                optimized_delay
            ),
            "DelayMinutesAvoided": (
                delay_avoided
            ),
            "AircraftDelayImprovementPercent": (
                100 * delay_avoided / fcfs_delay
                if fcfs_delay > 0
                else np.nan
            ),
            "FCFSIllustrativeCostUSD": fcfs_cost,
            "AircraftAwareIllustrativeCostUSD": (
                optimized_cost
            ),
            "IllustrativeCostReductionUSD": (
                cost_difference
            ),
            "IllustrativeCostReductionPercent": (
                100 * cost_difference / fcfs_cost
                if fcfs_cost > 0
                else np.nan
            ),
            "LowerIllustrativeCostPolicy": preferred,
            "PassengerAndCrewBaselinesHeldFixed": True,
        })

    comparison = pd.DataFrame(records)

    if comparison.empty:
        raise ValueError(
            "No valid matched scenarios produced."
        )

    comparison = comparison.sort_values(
        "MinimumTurnMinutes"
    ).reset_index(drop=True)

    return comparison


# ============================================================
# 6. INSPECT EXISTING AIRPORT STRESS TESTS
# ============================================================

def inspect_airport_stress_results():

    print("\nInspecting existing airport stress tests...")

    if not STRESS_FILE.exists():

        print(
            "Airport stress-test report not found; "
            "skipping inventory."
        )

        return pd.DataFrame()

    stress = pd.read_csv(STRESS_FILE)

    print(
        f"Existing airport stress-test records: "
        f"{len(stress):,}"
    )

    if stress.empty:
        return stress

    stress = stress.copy()

    stress["SourceFile"] = STRESS_FILE.name

    return stress


# ============================================================
# 7. GENERATE INTERACTIVE FIGURE
# ============================================================

def create_resilience_chart(comparison):

    print("\nCreating resilience sensitivity chart...")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=comparison["MinimumTurnMinutes"],
            y=comparison[
                "FCFSIncrementalDelayMinutes"
            ],
            mode="lines+markers",
            name="FCFS",
            line=dict(
                color="#64748b",
                width=3,
            ),
            marker=dict(size=9),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=comparison["MinimumTurnMinutes"],
            y=comparison[
                "AircraftAwareIncrementalDelayMinutes"
            ],
            mode="lines+markers",
            name="Aircraft-aware",
            line=dict(
                color="#0d9488",
                width=3,
            ),
            marker=dict(size=9),
        )
    )

    fig.update_layout(
        title=(
            "AeroResilience AI — "
            "Aircraft Turnaround Resilience"
        ),
        xaxis_title=(
            "Minimum Aircraft Turnaround (Minutes)"
        ),
        yaxis_title=(
            "Incremental Network Delay (Minutes)"
        ),
        template="plotly_white",
        height=600,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            y=1.12,
            x=0,
        ),
        margin=dict(
            l=70,
            r=40,
            t=120,
            b=70,
        ),
    )

    output_file = (
        FIGURE_DIR
        / "integrated_resilience_turnaround.html"
    )

    fig.write_html(
        str(output_file),
        include_plotlyjs=True,
    )

    return output_file


# ============================================================
# 8. SAVE METHODOLOGY
# ============================================================

def save_methodology(
    integrated,
    comparison,
    assumptions,
):

    methodology = {
        "Project": "AeroResilience AI",
        "Module": 5,
        "Step": 47,
        "AnalysisType": (
            "Integrated aircraft sensitivity and "
            "fixed passenger/crew baseline comparison"
        ),
        "IntegratedPolicyScenarioRecords": int(
            len(integrated)
        ),
        "MatchedTurnaroundSettings": (
            comparison["MinimumTurnMinutes"]
            .astype(float)
            .tolist()
        ),
        "FinancialAssumptions": assumptions,
        "Inputs": [
            str(AIRCRAFT_FILE),
            str(PASSENGER_FILE),
            str(FINANCIAL_FILE),
            str(ASSUMPTIONS_FILE),
        ],
        "Limitations": [
            "Aircraft sensitivity results are reused "
            "from earlier simulations.",
            "Passenger connection and rebooking outcomes "
            "are held fixed within each policy.",
            "Crew reserve assignments are held fixed "
            "within each policy.",
            "Financial outputs use hypothetical unit costs.",
            "The aircraft turnaround scenarios may not "
            "share the same no-stop reference as the "
            "original financial baseline.",
            "Only matching FCFS and aircraft-aware "
            "turnaround scenarios are compared.",
            "An integrated decision-analysis view is "
            "not equivalent to a fully coupled simulator.",
            "No actual airline savings are claimed.",
        ],
    }

    output_file = (
        REPORT_DIR
        / "integrated_resilience_methodology.json"
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
# 9. SAVE OUTPUTS
# ============================================================

def save_outputs(
    integrated,
    comparison,
    stress,
    assumptions,
):

    integrated_file = (
        REPORT_DIR
        / "integrated_resilience_scenarios.csv"
    )

    comparison_file = (
        REPORT_DIR
        / "integrated_resilience_comparison.csv"
    )

    integrated.to_csv(
        integrated_file,
        index=False,
    )

    comparison.to_csv(
        comparison_file,
        index=False,
    )

    if not stress.empty:

        stress_file = (
            REPORT_DIR
            / "integrated_airport_stress_inventory.csv"
        )

        stress.to_csv(
            stress_file,
            index=False,
        )

        print(f"Saved: {stress_file}")

    figure_file = create_resilience_chart(
        comparison
    )

    methodology_file = save_methodology(
        integrated,
        comparison,
        assumptions,
    )

    return [
        integrated_file,
        comparison_file,
        figure_file,
        methodology_file,
    ]


# ============================================================
# 10. PRINT RESULTS
# ============================================================

def display_results(comparison):

    print("\n" + "=" * 70)

    print("INTEGRATED RESILIENCE SCENARIO RESULTS")

    print("=" * 70)

    columns = [
        "MinimumTurnMinutes",
        "DelayMinutesAvoided",
        "AircraftDelayImprovementPercent",
        "IllustrativeCostReductionUSD",
        "LowerIllustrativeCostPolicy",
    ]

    print(
        comparison[columns].to_string(
            index=False,
            float_format=lambda value: f"{value:,.2f}",
        )
    )

    print("\nRESILIENCE SUMMARY")

    print(
        "Matched turnaround scenarios:",
        len(comparison),
    )

    print(
        "Aircraft-aware lower-cost scenarios:",
        int(
            (
                comparison[
                    "LowerIllustrativeCostPolicy"
                ] == "AircraftAware"
            ).sum()
        ),
    )

    print(
        "FCFS lower-cost scenarios:",
        int(
            (
                comparison[
                    "LowerIllustrativeCostPolicy"
                ] == "FCFS"
            ).sum()
        ),
    )

    print(
        "Positive aircraft-aware delay improvements:",
        int(
            (
                comparison["DelayMinutesAvoided"] > 0
            ).sum()
        ),
    )

    print("\nIMPORTANT LIMITATIONS")

    print(
        "Aircraft recovery sensitivity values "
        "were reused from existing simulations."
    )

    print(
        "Passenger outcomes are fixed baseline "
        "values, not recalculated for every scenario."
    )

    print(
        "Crew reserve assignments are fixed baseline "
        "values, not recalculated for every scenario."
    )

    print(
        "Financial calculations use illustrative "
        "cost assumptions, not actual airline expenses."
    )

    print(
        "This is an integrated comparison, not yet "
        "a fully coupled recovery simulator."
    )


# ============================================================
# 11. MAIN
# ============================================================

def main():

    print("=" * 70)

    print("AERORESILIENCE AI")

    print("MODULE 5 — RESILIENCE & WHAT-IF SIMULATOR")

    print("STEP 47 — INTEGRATED RESILIENCE ANALYSIS")

    print("=" * 70)

    (
        aircraft,
        passengers,
        financial,
        assumptions,
    ) = load_inputs()

    (
        aircraft,
        passengers,
        financial,
    ) = prepare_inputs(
        aircraft,
        passengers,
        financial,
        assumptions,
    )

    integrated = build_integrated_scenarios(
        aircraft,
        passengers,
        financial,
        assumptions,
    )

    comparison = compare_scenarios(
        integrated
    )

    stress = inspect_airport_stress_results()

    output_files = save_outputs(
        integrated,
        comparison,
        stress,
        assumptions,
    )

    display_results(comparison)

    print("\nREPORTS GENERATED SUCCESSFULLY")

    for path in output_files:
        print(f"Saved: {path}")

    print(
        "\nAERORESILIENCE AI — "
        "INTEGRATED RESILIENCE ANALYSIS COMPLETE"
    )


# ============================================================
# RUN STEP 47
# ============================================================

if __name__ == "__main__":
    main()
