
from pathlib import Path

import airportsdata
import numpy as np
import pandas as pd

# =====================================================
# AERORESILIENCE AI
# STEP 24 — AIRCRAFT-AWARE RECOVERY FEASIBILITY AUDIT
# =====================================================

RAW_FILE = Path("data/raw/bts_flights_2026_08.parquet")
RECOVERY_FILE = Path("outputs/reports/fair_recovery_flights.csv")
REPORT_DIR = Path("outputs/reports")

REPORT_DIR.mkdir(parents=True, exist_ok=True)

MIN_TURN_MINUTES = 20

KEYS = [
    "FlightDate",
    "Reporting_Airline",
    "Tail_Number",
    "Flight_Number_Reporting_Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
]


def prepare_flights():
    print("\nLoading aircraft schedules...")

    columns = KEYS + [
        "CRSElapsedTime",
        "Cancelled",
        "Diverted",
    ]

    raw = pd.read_parquet(RAW_FILE, columns=columns)
    recovery = pd.read_csv(RECOVERY_FILE)

    required = KEYS + [
        "RecoveryFlightID",
        "ScheduledMinute",
        "BaselineDelayMinutes",
        "OptimizedDelayMinutes",
        "FairDelayMinutes",
    ]

    missing = [c for c in required if c not in recovery.columns]

    if missing:
        raise ValueError(f"Missing recovery columns: {missing}")

    for frame in [raw, recovery]:
        frame["FlightDate"] = pd.to_datetime(
            frame["FlightDate"], errors="coerce"
        )

        for col in [
            "Reporting_Airline",
            "Tail_Number",
            "Origin",
            "Dest",
        ]:
            frame[col] = frame[col].astype("string").str.strip()

        for col in [
            "CRSDepTime",
            "Flight_Number_Reporting_Airline",
        ]:
            frame[col] = pd.to_numeric(
                frame[col], errors="coerce"
            )

    raw["CRSElapsedTime"] = pd.to_numeric(
        raw["CRSElapsedTime"], errors="coerce"
    )

    raw = raw.dropna(
        subset=KEYS + ["CRSElapsedTime"]
    ).copy()

    # Work with aircraft involved in the recovery.
    affected = recovery[
        ["Reporting_Airline", "Tail_Number"]
    ].drop_duplicates()

    raw = raw.merge(
        affected,
        on=["Reporting_Airline", "Tail_Number"],
        how="inner",
    )

    # Preserve unique matching to scheduled records.
    duplicate_keys = raw.duplicated(KEYS, keep=False)

    if duplicate_keys.any():
        raise ValueError(
            "Ambiguous raw flight identifiers found. "
            "Review duplicate schedule records before matching."
        )

    return raw, recovery


def add_utc_times(df):
    airport_db = airportsdata.load("IATA")

    zones = {
        code: item["tz"]
        for code, item in airport_db.items()
        if item.get("tz")
    }

    df = df.copy()

    hours = df["CRSDepTime"] // 100
    minutes = df["CRSDepTime"] % 100

    valid = (
        hours.between(0, 24)
        & minutes.between(0, 59)
        & ((hours < 24) | (minutes == 0))
    )

    df = df.loc[valid].copy()

    local_minutes = (
        (df["CRSDepTime"] // 100) * 60
        + (df["CRSDepTime"] % 100)
    )

    df["LocalDeparture"] = (
        df["FlightDate"]
        + pd.to_timedelta(local_minutes, unit="m")
    )

    df["Timezone"] = df["Origin"].map(zones)
    df["DepartureUTC"] = pd.NaT

    for timezone, group in df.groupby("Timezone"):
        converted = (
            group["LocalDeparture"]
            .dt.tz_localize(
                timezone,
                ambiguous="NaT",
                nonexistent="NaT",
            )
            .dt.tz_convert("UTC")
            .dt.tz_localize(None)
        )

        df.loc[group.index, "DepartureUTC"] = (
            converted.to_numpy()
        )

    df = df.dropna(subset=["DepartureUTC"]).copy()

    df["ArrivalUTC"] = (
        df["DepartureUTC"]
        + pd.to_timedelta(
            df["CRSElapsedTime"], unit="m"
        )
    )

    return df


def main():
    print("=" * 65)
    print("AERORESILIENCE AI — AIRCRAFT FEASIBILITY AUDIT")
    print("=" * 65)

    raw, recovery = prepare_flights()
    raw = add_utc_times(raw)

    raw = raw.sort_values(
        [
            "Reporting_Airline",
            "Tail_Number",
            "DepartureUTC",
        ]
    ).reset_index(drop=True)

    groups = raw.groupby(
        ["Reporting_Airline", "Tail_Number"],
        sort=False,
    )

    # Identify the next scheduled flight of each aircraft.
    for source, target in [
        ("Origin", "NextOrigin"),
        ("DepartureUTC", "NextDepartureUTC"),
        ("Cancelled", "NextCancelled"),
        ("Diverted", "NextDiverted"),
    ]:
        raw[target] = groups[source].shift(-1)

    matched = recovery.merge(
        raw[
            KEYS + [
                "DepartureUTC",
                "ArrivalUTC",
                "NextOrigin",
                "NextDepartureUTC",
                "NextCancelled",
                "NextDiverted",
            ]
        ],
        on=KEYS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    matched["ScheduleMatched"] = (
        matched["_merge"] == "both"
    )

    # Check whether the next scheduled flight starts
    # from the airport where this aircraft arrives.
    matched["AirportContinuity"] = (
        matched["Dest"] == matched["NextOrigin"]
    )

    matched["NextFlightKnown"] = (
        matched["NextDepartureUTC"].notna()
    )

    matched["NextFlightOperating"] = (
        matched["NextCancelled"].eq(0)
        & matched["NextDiverted"].eq(0)
    )

    # This audit only evaluates unambiguous,
    # operating next-flight connections.
    matched["AuditableConnection"] = (
        matched["ScheduleMatched"]
        & matched["NextFlightKnown"]
        & matched["AirportContinuity"]
        & matched["NextFlightOperating"]
    )

    policies = {
        "FCFS": "BaselineDelayMinutes",
        "Unconstrained": "OptimizedDelayMinutes",
        "Fair": "FairDelayMinutes",
    }

    summary = []

    for name, delay_column in policies.items():

        adjusted_arrival = (
            matched["ArrivalUTC"]
            + pd.to_timedelta(
                matched[delay_column], unit="m"
            )
        )

        available_turn = (
            matched["NextDepartureUTC"]
            - adjusted_arrival
        ).dt.total_seconds() / 60

        matched[f"{name}AvailableTurn"] = available_turn

        conflict = (
            matched["AuditableConnection"]
            & (available_turn < MIN_TURN_MINUTES)
        )

        matched[f"{name}TurnConflict"] = conflict

        summary.append({
            "Policy": name,
            "RecoveredFlights": len(matched),
            "MatchedFlights": int(
                matched["ScheduleMatched"].sum()
            ),
            "AuditableNextConnections": int(
                matched["AuditableConnection"].sum()
            ),
            "PotentialTurnConflicts": int(conflict.sum()),
            "ConflictRatePercent": (
                100 * conflict.sum()
                / matched["AuditableConnection"].sum()
                if matched["AuditableConnection"].any()
                else np.nan
            ),
        })

    results = pd.DataFrame(summary)

    matched.to_csv(
        REPORT_DIR / "aircraft_feasibility_details.csv",
        index=False,
    )

    results.to_csv(
        REPORT_DIR / "aircraft_feasibility_comparison.csv",
        index=False,
    )

    print("\nRECOVERY AIRCRAFT FEASIBILITY")
    print(results.to_string(index=False))

    print("\nDATA COVERAGE")
    print(f"Recovered flights: {len(matched):,}")
    print(
        "Matched schedules:",
        f"{matched['ScheduleMatched'].sum():,}"
    )
    print(
        "Auditable next-flight connections:",
        f"{matched['AuditableConnection'].sum():,}"
    )

    print("\nIMPORTANT LIMITATIONS")
    print(
        "This is an aircraft-turnaround audit, "
        "not a complete recovery feasibility certification."
    )
    print(
        "Next flights retain their original scheduled times."
    )
    print(
        "The audit assumes unchanged flight duration "
        "and aircraft assignment."
    )
    print(
        "Unknown, cancelled, diverted, or mismatched "
        "connections are excluded from conflict rates."
    )
    print(
        "Crew, gates, runway capacity, and downstream "
        "rescheduling are not yet evaluated."
    )

    print("\nReports saved successfully.")
    print(
        "\nAERORESILIENCE AI — "
        "AIRCRAFT FEASIBILITY AUDIT COMPLETE"
    )


if __name__ == "__main__":
    main()
