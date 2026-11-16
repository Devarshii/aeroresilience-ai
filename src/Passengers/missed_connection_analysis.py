
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

# =====================================================
# AERORESILIENCE AI
# STEP 39 — MISSED PASSENGER CONNECTION ANALYSIS
# =====================================================

DATA_FILE = Path(
    "data/processed/synthetic_passenger_itineraries.parquet"
)

RECOVERY_FILE = Path(
    "outputs/reports/integrated_crew_recovery_details.csv"
)

REPORT_DIR = Path("outputs/reports")
FIGURE_DIR = Path("outputs/figures")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

MIN_CONNECTION_MINUTES = 45

POLICIES = ["FCFS", "AircraftAware"]


# =====================================================
# 1. LOAD AND VALIDATE DATA
# =====================================================

def load_data():

    print("\nLoading passenger itineraries...")

    if not DATA_FILE.exists():
        raise FileNotFoundError(DATA_FILE)

    if not RECOVERY_FILE.exists():
        raise FileNotFoundError(RECOVERY_FILE)

    itineraries = pd.read_parquet(DATA_FILE)
    recovery = pd.read_csv(RECOVERY_FILE)

    itinerary_columns = [
        "ItineraryID",
        "RecoveryFlightID",
        "Airline",
        "ConnectionAirport",
        "FinalDestination",
        "InboundScheduledArrivalUTC",
        "OutboundScheduledDepartureUTC",
        "ConnectionMinutes",
        "SimulatedPassengers",
    ]

    recovery_columns = [
        "RecoveryFlightID",
        "Policy",
        "Reporting_Airline",
        "ScheduledMinute",
        "PolicyDepartureMinute",
    ]

    for name, frame, columns in [
        ("Itineraries", itineraries, itinerary_columns),
        ("Recovery", recovery, recovery_columns),
    ]:

        missing = [
            col for col in columns
            if col not in frame.columns
        ]

        if missing:
            raise ValueError(
                f"{name} missing columns: {missing}"
            )

    if itineraries["ItineraryID"].duplicated().any():
        raise ValueError("Duplicate itinerary IDs.")

    if recovery.duplicated(
        ["RecoveryFlightID", "Policy"]
    ).any():
        raise ValueError(
            "Duplicate recovery flight-policy pairs."
        )

    if itineraries[itinerary_columns].isna().any().any():
        raise ValueError("Incomplete passenger itineraries.")

    if recovery[recovery_columns].isna().any().any():
        raise ValueError("Incomplete recovery schedules.")

    for column in [
        "InboundScheduledArrivalUTC",
        "OutboundScheduledDepartureUTC",
    ]:
        itineraries[column] = pd.to_datetime(
            itineraries[column],
            utc=True,
            errors="coerce",
        )

    if itineraries[
        [
            "InboundScheduledArrivalUTC",
            "OutboundScheduledDepartureUTC",
        ]
    ].isna().any().any():
        raise ValueError("Invalid itinerary timestamps.")

    itineraries["SimulatedPassengers"] = pd.to_numeric(
        itineraries["SimulatedPassengers"],
        errors="raise",
    )

    if not (
        itineraries["SimulatedPassengers"] > 0
    ).all():
        raise ValueError("Passenger counts must be positive.")

    recovery["DepartureDelayMinutes"] = (
        recovery["PolicyDepartureMinute"]
        - recovery["ScheduledMinute"]
    )

    if (
        recovery["DepartureDelayMinutes"] < -1e-6
    ).any():
        raise ValueError("Negative recovery delay found.")

    print(f"Itineraries: {len(itineraries):,}")
    print(
        "Simulated passenger count:",
        f"{itineraries['SimulatedPassengers'].sum():,.0f}"
    )

    return itineraries, recovery


# =====================================================
# 2. CALCULATE CONNECTION RISK
# =====================================================

def evaluate_policy(itineraries, recovery, policy):

    print(f"\nEvaluating {policy}...")

    selected = recovery[
        recovery["Policy"] == policy
    ].copy()

    if selected.empty:
        raise ValueError(
            f"No recovery records found for {policy}"
        )

    selected = selected[
        [
            "RecoveryFlightID",
            "Reporting_Airline",
            "DepartureDelayMinutes",
        ]
    ]

    result = itineraries.merge(
        selected,
        on="RecoveryFlightID",
        how="left",
        validate="many_to_one",
    )

    if result["DepartureDelayMinutes"].isna().any():
        raise ValueError(
            f"Missing inbound recovery delay for {policy}"
        )

    if not (
        result["Airline"]
        == result["Reporting_Airline"]
    ).all():
        raise ValueError(
            f"Airline mismatch in {policy} itineraries"
        )

    result["Policy"] = policy

    # Simplifying assumptions:
    # - Inbound flight duration stays unchanged.
    # - Inbound arrival shifts by departure delay.
    # - Outbound flight stays on schedule.
    # - Every synthetic passenger in a group
    #   shares the same connection outcome.

    result["PredictedInboundArrivalUTC"] = (
        result["InboundScheduledArrivalUTC"]
        + pd.to_timedelta(
            result["DepartureDelayMinutes"],
            unit="m",
        )
    )

    result["AvailableConnectionMinutes"] = (
        (
            result["OutboundScheduledDepartureUTC"]
            - result["PredictedInboundArrivalUTC"]
        ).dt.total_seconds() / 60
    )

    result["MissedConnection"] = (
        result["AvailableConnectionMinutes"]
        < MIN_CONNECTION_MINUTES
    )

    result["SimulatedPassengersAtRisk"] = np.where(
        result["MissedConnection"],
        result["SimulatedPassengers"],
        0,
    )

    result["AdditionalTimeNeededMinutes"] = np.maximum(
        MIN_CONNECTION_MINUTES
        - result["AvailableConnectionMinutes"],
        0,
    )

    return result


# =====================================================
# 3. BUILD POLICY SUMMARY
# =====================================================

def summarize_policy(result, policy):

    missed_groups = int(
        result["MissedConnection"].sum()
    )

    passengers_at_risk = int(
        result["SimulatedPassengersAtRisk"].sum()
    )

    total_passengers = int(
        result["SimulatedPassengers"].sum()
    )

    return {
        "Policy": policy,
        "CandidateItineraries": len(result),
        "SimulatedPassengers": total_passengers,
        "MissedItineraryGroups": missed_groups,
        "SimulatedPassengersAtRisk": passengers_at_risk,
        "PassengerRiskRatePercent": (
            100 * passengers_at_risk / total_passengers
            if total_passengers else 0.0
        ),
        "AverageConnectionMinutes": float(
            result["AvailableConnectionMinutes"].mean()
        ),
        "UniqueInboundFlightsWithMisses": int(
            result.loc[
                result["MissedConnection"],
                "RecoveryFlightID",
            ].nunique()
        ),
    }


# =====================================================
# 4. MAIN ANALYSIS
# =====================================================

def main():

    print("=" * 65)
    print("AERORESILIENCE AI — MISSED CONNECTION ANALYSIS")
    print("=" * 65)

    itineraries, recovery = load_data()

    results = []
    details = []

    # Reference scenario with no newly imposed
    # inbound flight delay.
    reference = itineraries.copy()
    reference["Policy"] = "NoStopReference"
    reference["DepartureDelayMinutes"] = 0.0

    reference["AvailableConnectionMinutes"] = (
        (
            reference["OutboundScheduledDepartureUTC"]
            - reference["InboundScheduledArrivalUTC"]
        ).dt.total_seconds() / 60
    )

    reference["MissedConnection"] = (
        reference["AvailableConnectionMinutes"]
        < MIN_CONNECTION_MINUTES
    )

    reference["SimulatedPassengersAtRisk"] = np.where(
        reference["MissedConnection"],
        reference["SimulatedPassengers"],
        0,
    )

    results.append(
        summarize_policy(reference, "NoStopReference")
    )

    details.append(reference)

    for policy in POLICIES:

        evaluated = evaluate_policy(
            itineraries,
            recovery,
            policy,
        )

        results.append(
            summarize_policy(evaluated, policy)
        )

        details.append(evaluated)

    summary = pd.DataFrame(results)

    all_details = pd.concat(
        details,
        ignore_index=True,
    )

    # =================================================
    # 5. COMPARE FCFS AND AIRCRAFT-AWARE STRATEGIES
    # =================================================

    fcfs = summary.loc[
        summary["Policy"] == "FCFS",
        "SimulatedPassengersAtRisk",
    ].iloc[0]

    aircraft_aware = summary.loc[
        summary["Policy"] == "AircraftAware",
        "SimulatedPassengersAtRisk",
    ].iloc[0]

    difference = int(fcfs - aircraft_aware)

    improvement = (
        100 * difference / fcfs
        if fcfs > 0 else np.nan
    )

    # =================================================
    # 6. AIRPORT-LEVEL BREAKDOWN
    # =================================================

    airport_summary = (
        all_details[
            all_details["Policy"].isin(POLICIES)
        ]
        .groupby(
            ["Policy", "ConnectionAirport"]
        )
        .agg(
            Itineraries=("ItineraryID", "size"),
            SimulatedPassengers=(
                "SimulatedPassengers", "sum"
            ),
            PassengersAtRisk=(
                "SimulatedPassengersAtRisk", "sum"
            ),
        )
        .reset_index()
    )

    airport_summary["RiskRatePercent"] = (
        100
        * airport_summary["PassengersAtRisk"]
        / airport_summary["SimulatedPassengers"]
    )

    # =================================================
    # 7. SAVE REPORTS
    # =================================================

    summary_file = (
        REPORT_DIR / "missed_connection_summary.csv"
    )

    details_file = (
        REPORT_DIR / "missed_connection_details.csv"
    )

    airport_file = (
        REPORT_DIR / "missed_connection_by_airport.csv"
    )

    summary.to_csv(summary_file, index=False)
    all_details.to_csv(details_file, index=False)
    airport_summary.to_csv(airport_file, index=False)

    # =================================================
    # 8. VISUALIZATION
    # =================================================

    fig = px.bar(
        summary,
        x="Policy",
        y="SimulatedPassengersAtRisk",
        text="SimulatedPassengersAtRisk",
        title=(
            "Missed Connection Risk — "
            "Recovery Policy Comparison"
        ),
        labels={
            "SimulatedPassengersAtRisk":
                "Synthetic Passengers at Risk",
        },
        template="plotly_dark",
    )

    fig.update_layout(title_x=0.5)

    chart_file = (
        FIGURE_DIR / "missed_connection_comparison.html"
    )

    fig.write_html(chart_file)

    # =================================================
    # 9. TERMINAL OUTPUT
    # =================================================

    print("\n" + "=" * 65)
    print("PASSENGER CONNECTION RISK COMPARISON")
    print("=" * 65)

    print(summary.to_string(index=False))

    print("\nFCFS VS AIRCRAFT-AWARE")

    print(
        f"Difference in simulated passengers at risk: "
        f"{difference:+,}"
    )

    if np.isfinite(improvement):
        print(
            f"Risk reduction vs FCFS: "
            f"{improvement:.2f}%"
        )
    else:
        print(
            "Risk reduction percentage undefined "
            "because FCFS has zero at-risk passengers."
        )

    print("\nTOP CONNECTION AIRPORTS BY PASSENGER RISK")

    print(
        airport_summary.sort_values(
            "PassengersAtRisk",
            ascending=False,
        ).head(10).to_string(index=False)
    )

    print("\nIMPORTANT LIMITATIONS")
    print(
        "All passenger groups are synthetic."
    )
    print(
        "Connection risk assumes the outbound flight "
        "departs at its original scheduled time."
    )
    print(
        "Inbound flight duration is assumed unchanged."
    )
    print(
        "A 45-minute minimum connection time is "
        "an experimental assumption."
    )
    print(
        "This is a deterministic connection feasibility "
        "indicator, not a calibrated probability."
    )
    print(
        "Aircraft-aware recovery may improve network "
        "delay without reducing missed connections."
    )
    print(
        "Seat availability, passenger rebooking, "
        "outbound delays and protected connections "
        "are not yet modeled."
    )

    print(f"\nSummary saved: {summary_file}")
    print(f"Details saved: {details_file}")
    print(f"Airport report saved: {airport_file}")
    print(f"Chart saved: {chart_file}")

    print(
        "\nAERORESILIENCE AI — "
        "MISSED CONNECTION ANALYSIS COMPLETE"
    )


if __name__ == "__main__":
    main()
