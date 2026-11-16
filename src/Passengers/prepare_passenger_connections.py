
from pathlib import Path

import numpy as np
import pandas as pd

from src.optimization.aircraft_aware_recovery import (
    create_utc_schedule,
)

# =====================================================
# AERORESILIENCE AI
# STEP 38 — PASSENGER CONNECTION DATA PREPARATION
# =====================================================

RAW_FILE = Path(
    "data/raw/bts_flights_2026_08.parquet"
)

BASELINE_FILE = Path(
    "outputs/reports/recovery_baseline_flights.csv"
)

OUTPUT_DIR = Path("data/processed")
REPORT_DIR = Path("outputs/reports")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
MIN_CONNECTION_MINUTES = 45
MAX_CONNECTION_MINUTES = 240

SCENARIO_DATE = pd.Timestamp("2026-08-12")

FLIGHT_KEYS = [
    "FlightDate",
    "Reporting_Airline",
    "Tail_Number",
    "Flight_Number_Reporting_Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
]


def standardize(df):
    df = df.copy()

    df["FlightDate"] = pd.to_datetime(
        df["FlightDate"], errors="coerce"
    ).dt.normalize()

    for col in [
        "Reporting_Airline",
        "Tail_Number",
        "Origin",
        "Dest",
    ]:
        df[col] = df[col].astype("string").str.strip()

    for col in [
        "Flight_Number_Reporting_Airline",
        "CRSDepTime",
    ]:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    return df


def load_data():
    print("\nLoading BTS flight schedules...")

    if not RAW_FILE.exists():
        raise FileNotFoundError(RAW_FILE)

    if not BASELINE_FILE.exists():
        raise FileNotFoundError(BASELINE_FILE)

    columns = FLIGHT_KEYS + [
        "CRSElapsedTime",
        "Cancelled",
        "Diverted",
    ]

    raw = pd.read_parquet(
        RAW_FILE, columns=columns
    )

    baseline = pd.read_csv(BASELINE_FILE)

    raw = standardize(raw)
    baseline = standardize(baseline)

    raw["CRSElapsedTime"] = pd.to_numeric(
        raw["CRSElapsedTime"],
        errors="coerce",
    )

    for col in ["Cancelled", "Diverted"]:
        raw[col] = pd.to_numeric(
            raw[col], errors="coerce"
        )

    raw = raw[
        raw["FlightDate"].isin([
            SCENARIO_DATE,
            SCENARIO_DATE + pd.Timedelta(days=1),
        ])
        & raw["Cancelled"].eq(0)
        & raw["Diverted"].eq(0)
        & raw["CRSElapsedTime"].gt(0)
    ].copy()

    if raw.duplicated(FLIGHT_KEYS).any():
        raise ValueError(
            "Duplicate BTS flight identifiers found."
        )

    if baseline.duplicated(FLIGHT_KEYS).any():
        raise ValueError(
            "Duplicate baseline flight identifiers found."
        )

    print("Converting scheduled times to UTC...")

    raw = create_utc_schedule(raw)

    seeds = baseline[
        [
            "RecoveryFlightID",
            *FLIGHT_KEYS,
        ]
    ].merge(
        raw[
            FLIGHT_KEYS + ["DepUTC", "ArrUTC"]
        ],
        on=FLIGHT_KEYS,
        how="left",
        validate="one_to_one",
    )

    if seeds[["DepUTC", "ArrUTC"]].isna().any().any():
        raise ValueError(
            "One or more recovery flights did not match "
            "the BTS schedules."
        )

    if len(seeds) != 177:
        raise ValueError(
            f"Expected 177 recovery flights, got {len(seeds)}"
        )

    return raw, seeds


def build_connections(raw, seeds):
    rng = np.random.default_rng(RANDOM_SEED)

    print("\nBuilding candidate passenger connections...")

    # A connecting flight must depart from the
    # arrival airport of the preceding flight.
    # For this prototype, same-airline connections
    # are considered, without interline itineraries.

    onward_lookup = {
        key: group.sort_values("DepUTC")
        for key, group in raw.groupby(
            ["Reporting_Airline", "Origin"]
        )
    }

    records = []
    itinerary_number = 1

    for flight in seeds.itertuples(index=False):
        key = (
            flight.Reporting_Airline,
            flight.Dest,
        )

        onward = onward_lookup.get(key)

        if onward is None:
            continue

        candidates = onward[
            (onward["DepUTC"] > flight.ArrUTC)
            & (onward["Dest"] != flight.Origin)
        ].copy()

        candidates["ConnectionMinutes"] = (
            candidates["DepUTC"] - flight.ArrUTC
        ).dt.total_seconds() / 60

        candidates = candidates[
            candidates["ConnectionMinutes"].between(
                MIN_CONNECTION_MINUTES,
                MAX_CONNECTION_MINUTES,
            )
        ]

        # Select up to two plausible onward flights
        # per disrupted inbound flight.
        candidates = candidates.head(2)

        for next_flight in candidates.itertuples(index=False):
            simulated_passengers = int(
                rng.integers(5, 26)
            )

            records.append({
                "ItineraryID": f"ITIN-{itinerary_number:06d}",
                "RecoveryFlightID": int(
                    flight.RecoveryFlightID
                ),
                "Airline": str(flight.Reporting_Airline),
                "InitialOrigin": str(flight.Origin),
                "ConnectionAirport": str(flight.Dest),
                "FinalDestination": str(next_flight.Dest),
                "InboundScheduledArrivalUTC": flight.ArrUTC,
                "OutboundScheduledDepartureUTC": (
                    next_flight.DepUTC
                ),
                "ConnectionMinutes": float(
                    next_flight.ConnectionMinutes
                ),
                "SimulatedPassengers": (
                    simulated_passengers
                ),
                "OutboundFlightNumber": (
                    next_flight.Flight_Number_Reporting_Airline
                ),
                "PassengerDataType": "Synthetic",
            })

            itinerary_number += 1

    return pd.DataFrame(records)


def main():
    print("=" * 65)
    print("AERORESILIENCE AI — PASSENGER CONNECTIONS")
    print("=" * 65)

    raw, seeds = load_data()
    itineraries = build_connections(raw, seeds)

    if itineraries.empty:
        raise ValueError(
            "No plausible connections found "
            "for the selected scenario."
        )

    if itineraries["ItineraryID"].duplicated().any():
        raise RuntimeError("Duplicate itinerary IDs.")

    if not itineraries["ConnectionMinutes"].between(
        MIN_CONNECTION_MINUTES,
        MAX_CONNECTION_MINUTES,
    ).all():
        raise RuntimeError(
            "Invalid connection duration detected."
        )

    total_passengers = int(
        itineraries["SimulatedPassengers"].sum()
    )

    connected_flights = int(
        itineraries["RecoveryFlightID"].nunique()
    )

    summary = pd.DataFrame([{
        "RecoveryFlights": len(seeds),
        "RecoveryFlightsWithConnections": connected_flights,
        "CandidateItineraries": len(itineraries),
        "SimulatedPassengers": total_passengers,
        "MinimumConnectionMinutes": (
            MIN_CONNECTION_MINUTES
        ),
        "MaximumConnectionMinutes": (
            MAX_CONNECTION_MINUTES
        ),
    }])

    itinerary_file = (
        OUTPUT_DIR / "synthetic_passenger_itineraries.parquet"
    )

    report_file = (
        REPORT_DIR / "passenger_connection_summary.csv"
    )

    itineraries.to_parquet(
        itinerary_file, index=False
    )

    summary.to_csv(
        report_file, index=False
    )

    print("\nPASSENGER CONNECTION SUMMARY")
    print(summary.to_string(index=False))

    print("\nSAMPLE CONNECTION ITINERARIES")
    print(
        itineraries[
            [
                "ItineraryID",
                "Airline",
                "ConnectionAirport",
                "FinalDestination",
                "ConnectionMinutes",
                "SimulatedPassengers",
            ]
        ].head(10).to_string(index=False)
    )

    print("\nIMPORTANT LIMITATIONS")
    print(
        "Passenger groups and itineraries are synthetic."
    )
    print(
        "Flight schedules come from BTS records, "
        "but these are only plausible connections."
    )
    print(
        "Actual bookings, seat inventory, protected "
        "connections, and ticketing are unavailable."
    )
    print(
        "Only same-airline onward connections are modeled."
    )
    print(
        "The itinerary data is for research simulation, "
        "not actual passenger demand estimation."
    )

    print(f"\nItineraries saved: {itinerary_file}")
    print(f"Summary saved: {report_file}")

    print(
        "\nAERORESILIENCE AI — "
        "PASSENGER CONNECTION DATA COMPLETE"
    )


if __name__ == "__main__":
    main()
