
from pathlib import Path
import pandas as pd

# AeroResilience AI — Flight Data Cleaning

RAW_FILE = Path("data/raw/bts_flights_2026_08.parquet")
OUTPUT_FILE = Path("data/processed/flights_clean_2026_08.parquet")

REQUIRED_COLUMNS = [
    "FlightDate",
    "Reporting_Airline",
    "Tail_Number",
    "Flight_Number_Reporting_Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
    "CRSArrTime",
    "DepDelay",
    "ArrDelay",
    "Cancelled",
    "Diverted",
    "Distance",
]


def main():
    print("=" * 60)
    print("AERORESILIENCE AI — FLIGHT DATA CLEANING")
    print("=" * 60)

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Raw data not found: {RAW_FILE}"
        )

    df = pd.read_parquet(RAW_FILE)

    print(f"\nOriginal records: {len(df):,}")

    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in BTS dataset: {missing}"
        )

    df = df[REQUIRED_COLUMNS].copy()

    df["FlightDate"] = pd.to_datetime(
        df["FlightDate"], errors="coerce"
    )

    numeric_cols = [
        "CRSDepTime",
        "CRSArrTime",
        "DepDelay",
        "ArrDelay",
        "Cancelled",
        "Diverted",
        "Distance",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    df = df.dropna(
        subset=[
            "FlightDate",
            "Origin",
            "Dest",
            "Cancelled",
            "Diverted",
        ]
    )

    # Missing delays are meaningful for cancelled or diverted flights.
    # Keep them missing rather than filling them with zero.

    df["Cancelled"] = df["Cancelled"].astype(int)
    df["Diverted"] = df["Diverted"].astype(int)

    df["Departure_Delayed_15"] = (
        df["DepDelay"] >= 15
    ).astype(int)

    df["Arrival_Delayed_15"] = (
        df["ArrDelay"] >= 15
    ).astype(int)

    df["Disrupted"] = (
        (df["Cancelled"] == 1)
        | (df["Diverted"] == 1)
        | (df["Arrival_Delayed_15"] == 1)
    ).astype(int)

    df["DayOfWeek"] = df["FlightDate"].dt.day_of_week
    df["Month"] = df["FlightDate"].dt.month

    df["ScheduledDepHour"] = (
        (df["CRSDepTime"] // 100) % 24
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )

    df.to_parquet(
        OUTPUT_FILE, index=False
    )

    print(f"Cleaned records: {len(df):,}")
    print(f"Cancelled flights: {df['Cancelled'].sum():,}")
    print(f"Diverted flights: {df['Diverted'].sum():,}")
    print(f"Disrupted flights: {df['Disrupted'].sum():,}")
    print(f"Disruption rate: {df['Disrupted'].mean():.2%}")

    print(f"\nSaved cleaned data: {OUTPUT_FILE}")

    print("\nAERORESILIENCE AI — DATA CLEANING COMPLETE")


if __name__ == "__main__":
    main()
