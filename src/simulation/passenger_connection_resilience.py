
from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ============================================================
# AERORESILIENCE AI
# MODULE 5 — RESILIENCE & WHAT-IF SIMULATOR
# STEP 49 — PASSENGER CONNECTION RESILIENCE
# ============================================================

REPORT_DIR = Path("outputs/reports")
FIGURE_DIR = Path("outputs/figures")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

DETAILS_FILE = (
    REPORT_DIR / "missed_connection_details.csv"
)

BASELINE_FILE = (
    REPORT_DIR / "passenger_rebooking_summary.csv"
)

POLICIES = ["FCFS", "AircraftAware"]

# Hypothetical minimum connection time requirements.
# These are experimental thresholds, not official
# airline or airport minimum connection times.
CONNECTION_THRESHOLDS = [30, 45, 60, 75, 90]

BASELINE_THRESHOLD = 45

REQUIRED_COLUMNS = [
    "Policy",
    "ItineraryID",
    "SimulatedPassengers",
    "PredictedInboundArrivalUTC",
    "OutboundScheduledDepartureUTC",
]


# ============================================================
# 1. LOAD PASSENGER CONNECTION RESULTS
# ============================================================

def load_data():

    print("\nLoading existing passenger connection results...")

    if not DETAILS_FILE.exists():
        raise FileNotFoundError(
            f"Missing Step 39 report: {DETAILS_FILE}"
        )

    if not BASELINE_FILE.exists():
        raise FileNotFoundError(
            f"Missing Step 40 report: {BASELINE_FILE}"
        )

    details = pd.read_csv(DETAILS_FILE)
    baseline = pd.read_csv(BASELINE_FILE)

    missing = [
        column for column in REQUIRED_COLUMNS
        if column not in details.columns
    ]

    if missing:
        raise ValueError(
            f"Missing passenger detail columns: {missing}"
        )

    details = details[
        details["Policy"].isin(POLICIES)
    ].copy()

    if details.empty:
        raise ValueError(
            "No FCFS or AircraftAware passenger records."
        )

    if details.duplicated(
        ["Policy", "ItineraryID"]
    ).any():
        raise ValueError(
            "Duplicate itinerary-policy records detected."
        )

    details["SimulatedPassengers"] = pd.to_numeric(
        details["SimulatedPassengers"],
        errors="raise",
    )

    if (
        details["SimulatedPassengers"].isna().any()
        or (details["SimulatedPassengers"] <= 0).any()
    ):
        raise ValueError(
            "Invalid synthetic passenger counts."
        )

    if not np.allclose(
        details["SimulatedPassengers"],
        np.rint(details["SimulatedPassengers"]),
    ):
        raise ValueError(
            "Passenger counts must be whole numbers."
        )

    details["SimulatedPassengers"] = (
        details["SimulatedPassengers"].astype(int)
    )

    for column in [
        "PredictedInboundArrivalUTC",
        "OutboundScheduledDepartureUTC",
    ]:
        details[column] = (
            pd.to_datetime(
                details[column],
                utc=True,
                errors="coerce",
            )
            .dt.tz_localize(None)
        )

    if details[
        [
            "PredictedInboundArrivalUTC",
            "OutboundScheduledDepartureUTC",
        ]
    ].isna().any().any():
        raise ValueError(
            "Invalid connection arrival/departure timestamps."
        )

    if (
        len(baseline) != 2
        or set(baseline["Policy"]) != set(POLICIES)
    ):
        raise ValueError(
            "Expected two baseline passenger policies."
        )

    print(
        f"Passenger itinerary-policy records: "
        f"{len(details):,}"
    )

    return details, baseline


# ============================================================
# 2. CALCULATE AVAILABLE CONNECTION TIME
# ============================================================

def calculate_connection_minutes(details):

    details = details.copy()

    details["AvailableConnectionMinutes"] = (
        (
            details["OutboundScheduledDepartureUTC"]
            - details["PredictedInboundArrivalUTC"]
        )
        .dt.total_seconds()
        / 60
    )

    if not np.isfinite(
        details["AvailableConnectionMinutes"]
    ).all():
        raise ValueError(
            "Invalid available connection time values."
        )

    print("\nCalculated itinerary connection times.")

    return details


# ============================================================
# 3. RECALCULATE MISSED CONNECTIONS
# ============================================================

def calculate_scenarios(details):

    print("\nTesting minimum connection time assumptions...")

    summary_records = []
    detail_records = []

    for threshold in CONNECTION_THRESHOLDS:

        for policy in POLICIES:

            selected = details[
                details["Policy"] == policy
            ].copy()

            # A missed connection occurs when
            # available time is below the chosen
            # experimental minimum connection time.
            selected["MissedConnectionScenario"] = (
                selected["AvailableConnectionMinutes"]
                < threshold
            )

            selected["MinimumConnectionMinutes"] = (
                threshold
            )

            passengers_at_risk = int(
                selected.loc[
                    selected["MissedConnectionScenario"],
                    "SimulatedPassengers",
                ].sum()
            )

            total_passengers = int(
                selected["SimulatedPassengers"].sum()
            )

            safe_passengers = (
                total_passengers - passengers_at_risk
            )

            missed_groups = int(
                selected["MissedConnectionScenario"].sum()
            )

            risk_rate = (
                100 * passengers_at_risk
                / total_passengers
                if total_passengers > 0
                else np.nan
            )

            summary_records.append({
                "MinimumConnectionMinutes": threshold,
                "Policy": policy,
                "SimulatedPassengers": total_passengers,
                "PassengersAtRisk": passengers_at_risk,
                "PassengersNotAtRisk": safe_passengers,
                "MissedItineraryGroups": missed_groups,
                "RiskRatePercent": risk_rate,
            })

            detail_records.append(
                selected[
                    [
                        "MinimumConnectionMinutes",
                        "Policy",
                        "ItineraryID",
                        "SimulatedPassengers",
                        "AvailableConnectionMinutes",
                        "MissedConnectionScenario",
                    ]
                ]
            )

    summary = pd.DataFrame(summary_records)

    scenario_details = pd.concat(
        detail_records,
        ignore_index=True,
    )

    return summary, scenario_details


# ============================================================
# 4. VERIFY 45-MINUTE ORIGINAL BASELINE
# ============================================================

def validate_baseline(summary, baseline):

    print("\nValidating original 45-minute baseline...")

    original = baseline.set_index("Policy")

    checks = []

    for policy in POLICIES:

        scenario = summary[
            (summary["Policy"] == policy)
            & (
                summary["MinimumConnectionMinutes"]
                == BASELINE_THRESHOLD
            )
        ]

        if len(scenario) != 1:
            raise ValueError(
                f"Missing 45-minute result: {policy}"
            )

        calculated = int(
            scenario.iloc[0]["PassengersAtRisk"]
        )

        expected = int(
            original.loc[
                policy,
                "SimulatedPassengersAtRisk",
            ]
        )

        matched = calculated == expected

        checks.append({
            "Policy": policy,
            "MinimumConnectionMinutes": (
                BASELINE_THRESHOLD
            ),
            "CalculatedPassengersAtRisk": calculated,
            "OriginalPassengersAtRisk": expected,
            "BaselineMatches": matched,
        })

        print(
            f"{policy}: calculated={calculated:,}, "
            f"original={expected:,}, "
            f"match={matched}"
        )

    validation = pd.DataFrame(checks)

    if not validation["BaselineMatches"].all():
        raise RuntimeError(
            "The 45-minute scenario does not match "
            "the original passenger analysis. "
            "Check the Step 39 connection definition."
        )

    print("Original passenger baseline validated.")

    return validation


# ============================================================
# 5. COMPARE RECOVERY POLICIES
# ============================================================

def compare_policies(summary):

    records = []

    for threshold, group in summary.groupby(
        "MinimumConnectionMinutes"
    ):

        indexed = group.set_index("Policy")

        fcfs = indexed.loc["FCFS"]
        aircraft = indexed.loc["AircraftAware"]

        difference = (
            int(fcfs["PassengersAtRisk"])
            - int(aircraft["PassengersAtRisk"])
        )

        if difference > 0:
            preferred = "AircraftAware"
        elif difference < 0:
            preferred = "FCFS"
        else:
            preferred = "Tie"

        records.append({
            "MinimumConnectionMinutes": threshold,
            "FCFSPassengersAtRisk": int(
                fcfs["PassengersAtRisk"]
            ),
            "AircraftAwarePassengersAtRisk": int(
                aircraft["PassengersAtRisk"]
            ),
            "RiskReductionPassengers": difference,
            "RiskReductionPercentVsFCFS": (
                100 * difference
                / fcfs["PassengersAtRisk"]
                if fcfs["PassengersAtRisk"] > 0
                else np.nan
            ),
            "LowerPassengerRiskPolicy": preferred,
        })

    return pd.DataFrame(records).sort_values(
        "MinimumConnectionMinutes"
    )


# ============================================================
# 6. RESILIENCE VALIDATION
# ============================================================

def validate_scenarios(summary):

    print("\nValidating scenario consistency...")

    for policy in POLICIES:

        selected = (
            summary[
                summary["Policy"] == policy
            ]
            .sort_values("MinimumConnectionMinutes")
        )

        # Increasing required connection time
        # cannot decrease missed connection risk.
        if not (
            selected["PassengersAtRisk"]
            .diff()
            .fillna(0)
            >= 0
        ).all():
            raise RuntimeError(
                f"Risk monotonicity failed: {policy}"
            )

        if not (
            selected["PassengersAtRisk"]
            + selected["PassengersNotAtRisk"]
            == selected["SimulatedPassengers"]
        ).all():
            raise RuntimeError(
                f"Passenger totals failed: {policy}"
            )

    print("All passenger scenario checks passed.")


# ============================================================
# 7. INTERACTIVE VISUALIZATION
# ============================================================

def create_chart(summary):

    fig = go.Figure()

    colors = {
        "FCFS": "#64748b",
        "AircraftAware": "#0d9488",
    }

    for policy in POLICIES:

        selected = summary[
            summary["Policy"] == policy
        ].sort_values(
            "MinimumConnectionMinutes"
        )

        fig.add_trace(
            go.Scatter(
                x=selected[
                    "MinimumConnectionMinutes"
                ],
                y=selected["PassengersAtRisk"],
                mode="lines+markers",
                name=policy,
                line=dict(
                    color=colors[policy],
                    width=3,
                ),
                marker=dict(size=10),
                hovertemplate=(
                    "Minimum connection: %{x} minutes"
                    "<br>Passengers at risk: %{y:,}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=(
            "AeroResilience AI — "
            "Passenger Connection Resilience"
        ),
        xaxis_title=(
            "Assumed Minimum Connection Time (Minutes)"
        ),
        yaxis_title=(
            "Simulated Passengers at Risk"
        ),
        template="plotly_white",
        height=580,
        hovermode="x unified",
    )

    output_file = (
        FIGURE_DIR
        / "passenger_connection_resilience.html"
    )

    fig.write_html(
        str(output_file),
        include_plotlyjs=True,
    )

    return output_file


# ============================================================
# 8. MAIN PROGRAM
# ============================================================

def main():

    print("=" * 70)
    print("AERORESILIENCE AI")
    print("MODULE 5 — RESILIENCE & WHAT-IF SIMULATOR")
    print("STEP 49 — PASSENGER CONNECTION RESILIENCE")
    print("=" * 70)

    details, baseline = load_data()

    details = calculate_connection_minutes(details)

    summary, scenario_details = calculate_scenarios(
        details
    )

    validation = validate_baseline(
        summary,
        baseline,
    )

    validate_scenarios(summary)

    comparison = compare_policies(summary)

    output_files = {
        "summary": (
            REPORT_DIR
            / "passenger_connection_resilience_summary.csv"
        ),
        "details": (
            REPORT_DIR
            / "passenger_connection_resilience_details.csv"
        ),
        "comparison": (
            REPORT_DIR
            / "passenger_connection_resilience_comparison.csv"
        ),
        "validation": (
            REPORT_DIR
            / "passenger_connection_resilience_validation.csv"
        ),
    }

    summary.to_csv(
        output_files["summary"],
        index=False,
    )

    scenario_details.to_csv(
        output_files["details"],
        index=False,
    )

    comparison.to_csv(
        output_files["comparison"],
        index=False,
    )

    validation.to_csv(
        output_files["validation"],
        index=False,
    )

    chart_file = create_chart(summary)

    methodology_file = (
        REPORT_DIR
        / "passenger_connection_resilience_methodology.json"
    )

    with open(
        methodology_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                "Module": 5,
                "Step": 49,
                "MinimumConnectionThresholdsMinutes": (
                    CONNECTION_THRESHOLDS
                ),
                "BaselineThresholdMinutes": (
                    BASELINE_THRESHOLD
                ),
                "Input": str(DETAILS_FILE),
                "PassengerDemandType": "Synthetic",
                "ImportantLimitations": [
                    "Passenger counts are synthetic.",
                    "Aircraft recovery arrival times "
                    "are held fixed within each policy.",
                    "Minimum connection time is varied "
                    "as a hypothetical scenario parameter.",
                    "Outbound flights are assumed to "
                    "depart on their original schedules.",
                    "Passenger risk is recalculated, "
                    "but rebooking assignments are not.",
                    "This does not establish official "
                    "airport minimum connection times.",
                    "No real airline passenger outcomes "
                    "are claimed.",
                ],
            },
            file,
            indent=4,
        )

    print("\nPASSENGER CONNECTION RESILIENCE RESULTS")
    print(
        comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print("\nVALIDATION RESULTS")
    print(validation.to_string(index=False))

    print("\nIMPORTANT LIMITATIONS")
    print(
        "Connection requirements are hypothetical."
    )
    print(
        "Passenger risk is recalculated for each "
        "connection-time threshold."
    )
    print(
        "Aircraft arrival times and outbound schedules "
        "are unchanged within each policy."
    )
    print(
        "Passenger rebooking is not rerun in this step."
    )

    print("\nFILES SAVED")

    for path in output_files.values():
        print(path)

    print(chart_file)
    print(methodology_file)

    print(
        "\nAERORESILIENCE AI — "
        "PASSENGER CONNECTION RESILIENCE COMPLETE"
    )


if __name__ == "__main__":
    main()
