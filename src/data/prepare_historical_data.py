
from pathlib import Path
import pandas as pd

# =====================================================
# AERORESILIENCE AI
# STEP 9 — MULTI-MONTH HISTORICAL DATA PREPARATION
# =====================================================

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
REPORT_DIR = Path("outputs/reports")

MONTHS = [6, 7, 8]
YEAR = 2026

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


def clean_month(month):
    file_path = (
        RAW_DIR / f"bts_flights_{YEAR}_{month:02d}.parquet"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Missing dataset: {file_path}"
        )

    print(f"\nProcessing {YEAR}-{month:02d}...")

    df = pd.read_parquet(file_path)

    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = df[REQUIRED_COLUMNS].copy()

    original_count = len(df)

    df["FlightDate"] = pd.to_datetime(
        df["FlightDate"], errors="coerce"
    )

    numeric_columns = [
        "CRSDepTime",
        "CRSArrTime",
        "DepDelay",
        "ArrDelay",
        "Cancelled",
        "Diverted",
        "Distance",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    # Preserve missing delay values for cancelled
    # or diverted flights.
    df = df.dropna(
        subset=[
            "FlightDate",
            "Origin",
            "Dest",
            "Reporting_Airline",
            "Cancelled",
            "Diverted",
            "CRSDepTime",
        ]
    )

    # Remove invalid cancellation/diversion flags.
    df = df[
        df["Cancelled"].isin([0, 1])
        & df["Diverted"].isin([0, 1])
    ].copy()

    # Scheduled departure times use HHMM format.
    hours = df["CRSDepTime"] // 100
    minutes = df["CRSDepTime"] % 100

    valid_time = (
        (hours >= 0)
        & (hours <= 24)
        & (minutes >= 0)
        & (minutes < 60)
        & ((hours < 24) | (minutes == 0))
    )

    df = df.loc[valid_time].copy()

    df["Cancelled"] = df["Cancelled"].astype(int)
    df["Diverted"] = df["Diverted"].astype(int)

    df["ScheduledDepHour"] = (
        (df["CRSDepTime"] // 100) % 24
    ).astype(int)

    df["DayOfWeek"] = (
        df["FlightDate"].dt.day_of_week
    )

    df["Month"] = df["FlightDate"].dt.month

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

    # Ensure each file contains the requested month.
    expected_period = f"{YEAR}-{month:02d}"

    invalid_dates = (
        df["FlightDate"].dt.strftime("%Y-%m")
        != expected_period
    )

    if invalid_dates.any():
        raise ValueError(
            f"Unexpected flight dates in {file_path}"
        )

    print(f"Original records: {original_count:,}")
    print(f"Cleaned records: {len(df):,}")
    print(f"Removed records: {original_count - len(df):,}")
    print(f"Disruption rate: {df['Disrupted'].mean():.2%}")

    return df


def main():
    print("=" * 65)
    print("AERORESILIENCE AI — HISTORICAL DATA PREPARATION")
    print("=" * 65)

    PROCESSED_DIR.mkdir(
        parents=True, exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True, exist_ok=True
    )

    datasets = []

    for month in MONTHS:
        df = clean_month(month)
        datasets.append(df)

    combined = pd.concat(
        datasets,
        ignore_index=True
    )

    combined = combined.sort_values(
        ["FlightDate", "CRSDepTime"]
    ).reset_index(drop=True)

    output_file = (
        PROCESSED_DIR
        / "flights_historical_2026_06_08.parquet"
    )

    combined.to_parquet(
        output_file,
        index=False
    )

    monthly_summary = (
        combined.groupby("Month")
        .agg(
            TotalFlights=("Disrupted", "size"),
            DisruptedFlights=("Disrupted", "sum"),
            CancelledFlights=("Cancelled", "sum"),
            DivertedFlights=("Diverted", "sum"),
        )
        .reset_index()
    )

    monthly_summary["DisruptionRate"] = (
        monthly_summary["DisruptedFlights"]
        / monthly_summary["TotalFlights"] * 100
    )

    summary_file = (
        REPORT_DIR / "historical_monthly_summary.csv"
    )

    monthly_summary.to_csv(
        summary_file,
        index=False
    )

    print("\n" + "=" * 65)
    print("COMBINED DATASET SUMMARY")
    print("=" * 65)

    print(f"Total flights: {len(combined):,}")
    print(f"Unique airlines: {combined['Reporting_Airline'].nunique()}")
    print(f"Unique airports: {combined['Origin'].nunique()}")
    print(f"Disrupted flights: {combined['Disrupted'].sum():,}")
    print(f"Overall disruption rate: {combined['Disrupted'].mean():.2%}")

    print("\nMONTHLY COMPARISON")
    print(monthly_summary.to_string(index=False))

    print(f"\nCombined dataset saved: {output_file}")
    print(f"Monthly summary saved: {summary_file}")

    print("\nAERORESILIENCE AI — HISTORICAL DATA READY")


if __name__ == "__main__":
    main()
