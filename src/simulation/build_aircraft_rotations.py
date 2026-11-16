
from pathlib import Path

import airportsdata
import numpy as np
import pandas as pd

# =====================================================
# AERORESILIENCE AI
# STEP 16 — AIRCRAFT ROTATION CHAIN BUILDER
# =====================================================

RAW_FILE = Path("data/raw/bts_flights_2026_08.parquet")
REPORT_DIR = Path("outputs/reports")
PROCESSED_DIR = Path("data/processed")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

MIN_TURN_MINUTES = 20
MAX_CONNECTION_GAP_MINUTES = 600

COLUMNS = [
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


def main():

    print("=" * 65)
    print("AERORESILIENCE AI — AIRCRAFT ROTATION BUILDER")
    print("=" * 65)

    df = pd.read_parquet(
        RAW_FILE,
        columns=COLUMNS,
    )

    print(f"\nOriginal flights: {len(df):,}")

    # ---------------------------------------------
    # 1. Validate essential information
    # ---------------------------------------------

    df["FlightDate"] = pd.to_datetime(
        df["FlightDate"], errors="coerce"
    )

    for col in ["CRSDepTime", "CRSElapsedTime"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Tail_Number"] = (
        df["Tail_Number"].astype("string").str.strip()
    )

    df = df.dropna(
        subset=[
            "FlightDate",
            "Reporting_Airline",
            "Tail_Number",
            "Origin",
            "Dest",
            "CRSDepTime",
            "CRSElapsedTime",
        ]
    ).copy()

    df = df[
        df["Tail_Number"].ne("")
        & df["CRSElapsedTime"].between(1, 1440)
    ].copy()

    hours = (df["CRSDepTime"] // 100)
    minutes = (df["CRSDepTime"] % 100)

    valid_time = (
        hours.between(0, 24)
        & minutes.between(0, 59)
        & ((hours < 24) | (minutes == 0))
    )

    df = df.loc[valid_time].copy()

    # ---------------------------------------------
    # 2. Calculate scheduled local departures
    # ---------------------------------------------

    departure_minutes = (
        (df["CRSDepTime"] // 100) * 60
        + (df["CRSDepTime"] % 100)
    )

    df["ScheduledDepLocal"] = (
        df["FlightDate"]
        + pd.to_timedelta(departure_minutes, unit="m")
    )

    # ---------------------------------------------
    # 3. Convert departure timestamps to UTC
    # ---------------------------------------------

    airport_db = airportsdata.load("IATA")

    timezone_lookup = {
        code: record["tz"]
        for code, record in airport_db.items()
        if record.get("tz")
    }

    df["OriginTimezone"] = df["Origin"].map(
        timezone_lookup
    )

    df["ScheduledDepUTC"] = pd.NaT

    for timezone, group in df.groupby("OriginTimezone"):
        try:
            utc_times = (
                group["ScheduledDepLocal"]
                .dt.tz_localize(
                    timezone,
                    ambiguous="NaT",
                    nonexistent="NaT",
                )
                .dt.tz_convert("UTC")
                .dt.tz_localize(None)
            )

            df.loc[group.index, "ScheduledDepUTC"] = (
                utc_times.to_numpy()
            )

        except Exception as error:
            print(
                f"Timezone conversion skipped: "
                f"{timezone}: {error}"
            )

    missing_utc = df["ScheduledDepUTC"].isna().sum()

    print(f"Flights without usable UTC time: {missing_utc:,}")

    df = df.dropna(
        subset=["ScheduledDepUTC"]
    ).copy()

    # Scheduled arrival in UTC is calculated using
    # official scheduled elapsed duration.
    df["ScheduledArrUTC"] = (
        df["ScheduledDepUTC"]
        + pd.to_timedelta(
            df["CRSElapsedTime"], unit="m"
        )
    )

    # ---------------------------------------------
    # 4. Identify consecutive aircraft flights
    # ---------------------------------------------

    df = df.sort_values(
        [
            "Reporting_Airline",
            "Tail_Number",
            "ScheduledDepUTC",
        ]
    ).reset_index(drop=True)

    df["FlightID"] = np.arange(len(df))

    aircraft_groups = df.groupby(
        ["Reporting_Airline", "Tail_Number"],
        sort=False,
    )

    previous_columns = {
        "FlightID": "PreviousFlightID",
        "Dest": "PreviousDestination",
        "ScheduledArrUTC": "PreviousScheduledArrival",
    }

    for source, target in previous_columns.items():
        df[target] = aircraft_groups[source].shift(1)

    df["TurnaroundMinutes"] = (
        (
            df["ScheduledDepUTC"]
            - df["PreviousScheduledArrival"]
        ).dt.total_seconds() / 60
    )

    df["AirportMatch"] = (
        df["Origin"] == df["PreviousDestination"]
    )

    df["ValidRotation"] = (
        df["PreviousFlightID"].notna()
        & df["AirportMatch"]
        & df["TurnaroundMinutes"].between(
            MIN_TURN_MINUTES,
            MAX_CONNECTION_GAP_MINUTES,
        )
    )

    # A separate flag shows scheduling conflicts
    # rather than silently treating them as valid.
    df["ShortTurnOrOverlap"] = (
        df["PreviousFlightID"].notna()
        & df["AirportMatch"]
        & (df["TurnaroundMinutes"] < MIN_TURN_MINUTES)
    )

    rotations = df[df["ValidRotation"]].copy()

    # ---------------------------------------------
    # 5. Save results
    # ---------------------------------------------

    output_file = (
        PROCESSED_DIR / "aircraft_rotations_august_2026.parquet"
    )

    rotations.to_parquet(
        output_file, index=False
    )

    summary = pd.DataFrame([{
        "TotalUsableFlights": len(df),
        "UniqueAircraft": df["Tail_Number"].nunique(),
        "ValidRotationLinks": int(df["ValidRotation"].sum()),
        "ShortTurnOrOverlap": int(
            df["ShortTurnOrOverlap"].sum()
        ),
        "MedianTurnaroundMinutes": (
            rotations["TurnaroundMinutes"].median()
            if len(rotations) else np.nan
        ),
        "AverageTurnaroundMinutes": (
            rotations["TurnaroundMinutes"].mean()
            if len(rotations) else np.nan
        ),
    }])

    summary.to_csv(
        REPORT_DIR / "aircraft_rotation_summary.csv",
        index=False,
    )

    print("\nAIRCRAFT ROTATION RESULTS")
    print(summary.to_string(index=False))

    print("\nSAMPLE VALID CONNECTIONS")

    display_cols = [
        "Tail_Number",
        "PreviousDestination",
        "Origin",
        "Dest",
        "TurnaroundMinutes",
    ]

    print(
        rotations[display_cols]
        .head(10)
        .to_string(index=False)
    )

    print("\nIMPORTANT")
    print(
        "These are inferred scheduled aircraft connections. "
        "They are not verified operational rotations."
    )
    print(
        "Cancelled flights and aircraft substitutions may "
        "make historical tail-number sequences incomplete."
    )
    print(
        "Cross-month connections are not modeled yet."
    )

    print(f"\nSaved: {output_file}")

    print(
        "\nAERORESILIENCE AI — AIRCRAFT ROTATIONS COMPLETE"
    )


if __name__ == "__main__":
    main()
