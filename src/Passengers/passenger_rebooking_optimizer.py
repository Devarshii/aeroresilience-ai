
from pathlib import Path
from collections import defaultdict
import hashlib

import numpy as np
import pandas as pd

from src.optimization.aircraft_aware_recovery import (
    create_utc_schedule,
)

# =====================================================
# AERORESILIENCE AI
# STEP 40 — PASSENGER REBOOKING OPTIMIZER
# CORRECTED COMPLETE VERSION
# =====================================================

RAW_FILE = Path(
    "data/raw/bts_flights_2026_08.parquet"
)

RISK_FILE = Path(
    "outputs/reports/missed_connection_details.csv"
)

REPORT_DIR = Path("outputs/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

POLICIES = ["FCFS", "AircraftAware"]

MIN_CONNECTION = 45
MAX_REBOOKING_HOURS = 36

# Synthetic seat availability assumptions.
MIN_AVAILABLE_SEATS = 5
MAX_AVAILABLE_SEATS = 25

# Tail_Number is required by create_utc_schedule().
FLIGHT_COLUMNS = [
    "FlightDate",
    "Reporting_Airline",
    "Tail_Number",
    "Flight_Number_Reporting_Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
    "CRSElapsedTime",
    "Cancelled",
    "Diverted",
]

ASSIGNMENT_COLUMNS = [
    "Policy",
    "ItineraryID",
    "Airline",
    "ConnectionAirport",
    "FinalDestination",
    "AlternativeFlightID",
    "RebookedPassengers",
    "AlternativeDepartureUTC",
    "AlternativeArrivalUTC",
    "SyntheticSeatCapacity",
    "RecoveryWaitMinutes",
]

OUTCOME_COLUMNS = [
    "Policy",
    "ItineraryID",
    "RequestedPassengers",
    "RebookedPassengers",
    "UnaccommodatedPassengers",
    "FullyRebooked",
]


# =====================================================
# 1. DETERMINISTIC SYNTHETIC SEAT INVENTORY
# =====================================================

def simulated_seats(flight_id):
    """
    Generate reproducible synthetic seat availability.

    The same alternative flight receives the same
    capacity under both recovery policies.
    """

    digest = hashlib.sha256(
        str(flight_id).encode("utf-8")
    ).hexdigest()

    number = int(digest[:12], 16)

    return (
        MIN_AVAILABLE_SEATS
        + number
        % (
            MAX_AVAILABLE_SEATS
            - MIN_AVAILABLE_SEATS
            + 1
        )
    )


# =====================================================
# 2. LOAD HISTORICAL FLIGHT SCHEDULES
# =====================================================

def load_schedules():

    print("\nLoading BTS alternative flight schedules...")

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Missing flight data: {RAW_FILE}"
        )

    raw = pd.read_parquet(
        RAW_FILE,
        columns=FLIGHT_COLUMNS,
    )

    raw["FlightDate"] = pd.to_datetime(
        raw["FlightDate"],
        errors="coerce",
    )

    raw = raw[
        raw["FlightDate"].between(
            "2026-08-12",
            "2026-08-15",
        )
    ].copy()

    numeric_columns = [
        "Flight_Number_Reporting_Airline",
        "CRSDepTime",
        "CRSElapsedTime",
        "Cancelled",
        "Diverted",
    ]

    for column in numeric_columns:
        raw[column] = pd.to_numeric(
            raw[column],
            errors="coerce",
        )

    text_columns = [
        "Reporting_Airline",
        "Tail_Number",
        "Origin",
        "Dest",
    ]

    for column in text_columns:
        raw[column] = (
            raw[column]
            .astype("string")
            .str.strip()
        )

    raw = raw[
        raw["Cancelled"].eq(0)
        & raw["Diverted"].eq(0)
        & raw["CRSElapsedTime"].gt(0)
    ].copy()

    raw = raw.dropna(
        subset=[
            "FlightDate",
            "Reporting_Airline",
            "Tail_Number",
            "Flight_Number_Reporting_Airline",
            "Origin",
            "Dest",
            "CRSDepTime",
            "CRSElapsedTime",
        ]
    ).copy()

    raw = raw[
        raw["Tail_Number"].ne("")
        & raw["Origin"].ne("")
        & raw["Dest"].ne("")
    ].copy()

    if raw.empty:
        raise ValueError(
            "No valid alternative flight schedules found."
        )

    print("Converting scheduled times to UTC...")

    # This helper requires Tail_Number because it
    # sorts schedules by aircraft rotation.
    raw = create_utc_schedule(raw)

    raw = raw.sort_values(
        [
            "DepUTC",
            "Reporting_Airline",
            "Flight_Number_Reporting_Airline",
        ]
    ).reset_index(drop=True)

    # Construct a reproducible identifier for each
    # airline, flight, route, and departure.
    raw["AlternativeFlightID"] = (
        raw["Reporting_Airline"].astype(str)
        + "|"
        + raw[
            "Flight_Number_Reporting_Airline"
        ].astype("Int64").astype(str)
        + "|"
        + raw["Origin"].astype(str)
        + "|"
        + raw["Dest"].astype(str)
        + "|"
        + raw["DepUTC"].astype(str)
    )

    # Avoid counting an alternative flight twice.
    raw = raw.drop_duplicates(
        subset=["AlternativeFlightID"]
    ).copy()

    raw["AvailableSeats"] = (
        raw["AlternativeFlightID"]
        .map(simulated_seats)
    )

    if raw["AvailableSeats"].isna().any():
        raise ValueError(
            "Synthetic seat availability contains missing values."
        )

    print(
        f"Scheduled candidate flights: {len(raw):,}"
    )

    print(
        "Synthetic seats available:",
        f"{raw['AvailableSeats'].sum():,}"
    )

    return raw


# =====================================================
# 3. LOAD MISSED CONNECTION RESULTS
# =====================================================

def load_passenger_risk():

    print("\nLoading missed connection results...")

    if not RISK_FILE.exists():
        raise FileNotFoundError(
            f"Missing connection analysis: {RISK_FILE}"
        )

    risk = pd.read_csv(RISK_FILE)

    required = [
        "ItineraryID",
        "Policy",
        "Airline",
        "ConnectionAirport",
        "FinalDestination",
        "SimulatedPassengers",
        "MissedConnection",
        "PredictedInboundArrivalUTC",
        "OutboundScheduledDepartureUTC",
        "OutboundFlightNumber",
    ]

    missing = [
        column for column in required
        if column not in risk.columns
    ]

    if missing:
        raise ValueError(
            f"Missing passenger risk columns: {missing}"
        )

    risk = risk[
        risk["Policy"].isin(POLICIES)
    ].copy()

    if risk.empty:
        raise ValueError(
            "No passenger risk records found "
            "for the required recovery policies."
        )

    if risk.duplicated(
        ["Policy", "ItineraryID"]
    ).any():
        raise ValueError(
            "Duplicate policy-itinerary combinations."
        )

    risk["MissedConnection"] = (
        risk["MissedConnection"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("true")
    )

    for column in [
        "PredictedInboundArrivalUTC",
        "OutboundScheduledDepartureUTC",
    ]:

        risk[column] = (
            pd.to_datetime(
                risk[column],
                utc=True,
                errors="coerce",
            )
            .dt.tz_localize(None)
        )

    risk["SimulatedPassengers"] = (
        pd.to_numeric(
            risk["SimulatedPassengers"],
            errors="raise",
        ).astype(int)
    )

    risk["OutboundFlightNumber"] = pd.to_numeric(
        risk["OutboundFlightNumber"],
        errors="coerce",
    )

    if risk[
        [
            "PredictedInboundArrivalUTC",
            "OutboundScheduledDepartureUTC",
        ]
    ].isna().any().any():

        raise ValueError(
            "Passenger connection timestamps are invalid."
        )

    if (
        risk["SimulatedPassengers"] <= 0
    ).any():

        raise ValueError(
            "Synthetic passenger counts must be positive."
        )

    if risk.loc[
        risk["MissedConnection"],
        "OutboundFlightNumber",
    ].isna().any():

        raise ValueError(
            "Missed itineraries have missing outbound flight numbers."
        )

    print(
        "Missed itinerary-policy records:",
        int(risk["MissedConnection"].sum())
    )

    return risk


# =====================================================
# 4. BUILD ALTERNATIVE FLIGHT LOOKUP
# =====================================================

def build_flight_lookup(schedules):

    lookup = {}

    for key, group in schedules.groupby(
        [
            "Reporting_Airline",
            "Origin",
            "Dest",
        ]
    ):

        lookup[key] = group.sort_values(
            ["ArrUTC", "DepUTC"]
        ).copy()

    print(
        f"Alternative route groups: {len(lookup):,}"
    )

    return lookup


# =====================================================
# 5. PASSENGER REBOOKING FOR ONE POLICY
# =====================================================

def rebook_policy(risk, lookup, policy):

    print(
        f"\nOptimizing passenger recovery: {policy}"
    )

    missed = risk[
        (risk["Policy"] == policy)
        & risk["MissedConnection"]
    ].copy()

    missed = missed.sort_values(
        [
            "OutboundScheduledDepartureUTC",
            "ItineraryID",
        ]
    )

    # Each policy starts with identical synthetic
    # seat inventory. Within one policy, passenger
    # groups share the available seat pool.
    used_seats = defaultdict(int)

    assignments = []
    outcomes = []

    for itinerary in missed.itertuples(index=False):

        requested = int(
            itinerary.SimulatedPassengers
        )

        remaining = requested

        ready_time = (
            itinerary.PredictedInboundArrivalUTC
            + pd.Timedelta(
                minutes=MIN_CONNECTION
            )
        )

        latest_time = (
            itinerary.OutboundScheduledDepartureUTC
            + pd.Timedelta(
                hours=MAX_REBOOKING_HOURS
            )
        )

        key = (
            itinerary.Airline,
            itinerary.ConnectionAirport,
            itinerary.FinalDestination,
        )

        candidates = lookup.get(key)

        if candidates is not None:

            options = candidates[
                (candidates["DepUTC"] >= ready_time)
                & (candidates["DepUTC"] <= latest_time)
            ]

            # Earliest feasible arrival is preferred.
            for flight in options.itertuples(index=False):

                if remaining <= 0:
                    break

                flight_id = flight.AlternativeFlightID

                # Exclude the original missed flight.
                original_flight = (
                    flight.DepUTC
                    == itinerary.OutboundScheduledDepartureUTC
                    and
                    flight.Flight_Number_Reporting_Airline
                    == itinerary.OutboundFlightNumber
                )

                if original_flight:
                    continue

                available = (
                    int(flight.AvailableSeats)
                    - used_seats[flight_id]
                )

                if available <= 0:
                    continue

                allocated = min(
                    available,
                    remaining,
                )

                used_seats[flight_id] += allocated
                remaining -= allocated

                assignments.append({
                    "Policy": policy,
                    "ItineraryID": itinerary.ItineraryID,
                    "Airline": itinerary.Airline,
                    "ConnectionAirport": (
                        itinerary.ConnectionAirport
                    ),
                    "FinalDestination": (
                        itinerary.FinalDestination
                    ),
                    "AlternativeFlightID": flight_id,
                    "RebookedPassengers": int(allocated),
                    "AlternativeDepartureUTC": (
                        flight.DepUTC
                    ),
                    "AlternativeArrivalUTC": (
                        flight.ArrUTC
                    ),
                    "SyntheticSeatCapacity": int(
                        flight.AvailableSeats
                    ),
                    "RecoveryWaitMinutes": float(
                        (
                            flight.DepUTC - ready_time
                        ).total_seconds() / 60
                    ),
                })

        outcomes.append({
            "Policy": policy,
            "ItineraryID": itinerary.ItineraryID,
            "RequestedPassengers": requested,
            "RebookedPassengers": requested - remaining,
            "UnaccommodatedPassengers": remaining,
            "FullyRebooked": remaining == 0,
        })

    return (
        pd.DataFrame(
            assignments,
            columns=ASSIGNMENT_COLUMNS,
        ),
        pd.DataFrame(
            outcomes,
            columns=OUTCOME_COLUMNS,
        ),
    )


# =====================================================
# 6. VALIDATE AND SUMMARIZE REBOOKING
# =====================================================

def summarize(risk, assignments, outcomes, policy):

    selected = risk[
        risk["Policy"] == policy
    ]

    missed = selected[
        selected["MissedConnection"]
    ]

    total_at_risk = int(
        missed["SimulatedPassengers"].sum()
    )

    rebooked = int(
        assignments["RebookedPassengers"].sum()
    )

    unaccommodated = int(
        outcomes["UnaccommodatedPassengers"].sum()
    )

    if rebooked + unaccommodated != total_at_risk:

        raise RuntimeError(
            "Passenger recovery totals do not reconcile."
        )

    # Check group-level passenger conservation.
    if not outcomes.empty:

        if not (
            outcomes["RequestedPassengers"]
            ==
            (
                outcomes["RebookedPassengers"]
                + outcomes["UnaccommodatedPassengers"]
            )
        ).all():

            raise RuntimeError(
                "Passenger group conservation failed."
            )

    # Validate simulated seat constraints.
    if not assignments.empty:

        allocated = assignments.groupby(
            "AlternativeFlightID"
        ).agg(
            UsedSeats=(
                "RebookedPassengers",
                "sum",
            ),
            AvailableSeats=(
                "SyntheticSeatCapacity",
                "first",
            ),
        )

        if (
            allocated["UsedSeats"]
            > allocated["AvailableSeats"]
        ).any():

            raise RuntimeError(
                "Synthetic alternative flight seat "
                "capacity exceeded."
            )

    rebooking_rate = (
        100 * rebooked / total_at_risk
        if total_at_risk > 0 else np.nan
    )

    return {
        "Policy": policy,
        "SimulatedPassengersAtRisk": total_at_risk,
        "PassengersRebooked": rebooked,
        "PassengersUnaccommodated": unaccommodated,
        "RebookingRatePercent": rebooking_rate,
        "FullyRebookedGroups": int(
            outcomes["FullyRebooked"].sum()
        ),
        "AlternativeFlightsUsed": int(
            assignments["AlternativeFlightID"].nunique()
        ),
    }


# =====================================================
# 7. MAIN PROGRAM
# =====================================================

def main():

    print("=" * 65)
    print("AERORESILIENCE AI — PASSENGER REBOOKING")
    print("=" * 65)

    risk = load_passenger_risk()

    schedules = load_schedules()

    lookup = build_flight_lookup(schedules)

    all_assignments = []
    all_outcomes = []
    summary_rows = []

    for policy in POLICIES:

        assignments, outcomes = rebook_policy(
            risk,
            lookup,
            policy,
        )

        result = summarize(
            risk,
            assignments,
            outcomes,
            policy,
        )

        summary_rows.append(result)

        all_assignments.append(assignments)
        all_outcomes.append(outcomes)

    summary = pd.DataFrame(summary_rows)

    assignments = pd.concat(
        all_assignments,
        ignore_index=True,
    )

    outcomes = pd.concat(
        all_outcomes,
        ignore_index=True,
    )

    # -------------------------------------------------
    # SAVE OUTPUT FILES
    # -------------------------------------------------

    summary_file = (
        REPORT_DIR
        / "passenger_rebooking_summary.csv"
    )

    assignments_file = (
        REPORT_DIR
        / "passenger_rebooking_assignments.csv"
    )

    outcomes_file = (
        REPORT_DIR
        / "passenger_rebooking_outcomes.csv"
    )

    summary.to_csv(
        summary_file,
        index=False,
    )

    assignments.to_csv(
        assignments_file,
        index=False,
    )

    outcomes.to_csv(
        outcomes_file,
        index=False,
    )

    # -------------------------------------------------
    # TERMINAL RESULTS
    # -------------------------------------------------

    print("\n" + "=" * 65)
    print("PASSENGER REBOOKING COMPARISON")
    print("=" * 65)

    print(
        summary.to_string(index=False)
    )

    print("\nPASSENGER RECOVERY VALIDATION")

    for row in summary.itertuples(index=False):

        print(f"\nPolicy: {row.Policy}")

        print(
            f"Passengers at risk: "
            f"{row.SimulatedPassengersAtRisk:,}"
        )

        print(
            f"Passengers rebooked: "
            f"{row.PassengersRebooked:,}"
        )

        print(
            f"Passengers unaccommodated: "
            f"{row.PassengersUnaccommodated:,}"
        )

        print(
            f"Rebooking rate: "
            f"{row.RebookingRatePercent:.2f}%"
        )

        print(
            f"Alternative flights used: "
            f"{row.AlternativeFlightsUsed:,}"
        )

    print("\nIMPORTANT LIMITATIONS")

    print(
        "Passenger groups and seat availability are synthetic."
    )

    print(
        "Only same-airline nonstop alternative flights "
        "are considered."
    )

    print(
        "BTS flights provide historical schedules, "
        "not live airline seat inventory."
    )

    print(
        "Alternative flights are assumed to operate "
        "according to their original schedules."
    )

    print(
        "This is a deterministic greedy allocation "
        "heuristic, not a globally optimal solution."
    )

    print(
        "Passenger groups may be split across "
        "multiple alternative flights."
    )

    print(
        "Real ticketing rules, fare classes, "
        "accessibility requirements, aircraft changes, "
        "and crew restrictions are not modeled."
    )

    print(
        "The results are simulated passenger recovery "
        "outcomes, not verified airline performance."
    )

    print("\nReports saved successfully.")

    print(
        f"Summary saved: {summary_file}"
    )

    print(
        f"Assignments saved: {assignments_file}"
    )

    print(
        f"Outcomes saved: {outcomes_file}"
    )

    print(
        "\nAERORESILIENCE AI — "
        "PASSENGER REBOOKING COMPLETE"
    )


# =====================================================
# RUN STEP 40
# =====================================================

if __name__ == "__main__":
    main()
