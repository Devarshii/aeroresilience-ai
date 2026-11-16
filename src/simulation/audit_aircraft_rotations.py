
from pathlib import Path
import pandas as pd

# =====================================================
# AERORESILIENCE AI
# STEP 15 — AIRCRAFT ROTATION DATA AUDIT
# =====================================================

RAW_DIR = Path("data/raw")
REPORT_DIR = Path("outputs/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MONTHS = [6, 7, 8]
YEAR = 2026

IMPORTANT_FIELDS = [
    "FlightDate",
    "Reporting_Airline",
    "Tail_Number",
    "Flight_Number_Reporting_Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
    "CRSArrTime",
    "CRSElapsedTime",
    "DepTime",
    "ArrTime",
    "DepDelay",
    "ArrDelay",
    "Cancelled",
    "Diverted",
]


def audit_month(month):
    path = RAW_DIR / f"bts_flights_{YEAR}_{month:02d}.parquet"

    if not path.exists():
        raise FileNotFoundError(path)

    print(f"\nAuditing {YEAR}-{month:02d}...")

    df = pd.read_parquet(path)

    available = [c for c in IMPORTANT_FIELDS if c in df.columns]
    missing = [c for c in IMPORTANT_FIELDS if c not in df.columns]

    print(f"Total flight records: {len(df):,}")
    print(f"Available audit fields: {len(available)}")

    if missing:
        print(f"Missing fields: {missing}")

    reports = []

    for col in IMPORTANT_FIELDS:
        if col in df.columns:
            non_missing = int(df[col].notna().sum())
            coverage = non_missing / len(df) if len(df) else 0
        else:
            non_missing = 0
            coverage = 0

        reports.append({
            "Month": month,
            "Field": col,
            "Available": col in df.columns,
            "NonMissingRecords": non_missing,
            "CoveragePercent": round(coverage * 100, 2),
        })

    # Examine aircraft identification coverage.
    if "Tail_Number" in df.columns:
        tails = df["Tail_Number"].astype("string").str.strip()
        valid_tails = tails.notna() & tails.ne("")

        print(f"Flights with aircraft ID: {valid_tails.sum():,}")
        print(f"Aircraft ID coverage: {valid_tails.mean():.2%}")
        print(f"Unique aircraft IDs: {tails[valid_tails].nunique():,}")

        # Count aircraft that appear more than once per day.
        if "FlightDate" in df.columns:
            rotations = pd.DataFrame({
                "FlightDate": df["FlightDate"],
                "Tail_Number": tails,
            })

            rotations = rotations.loc[valid_tails]

            counts = (
                rotations.groupby(["FlightDate", "Tail_Number"])
                .size()
            )

            multi_flight = counts[counts >= 2]

            print(
                "Aircraft-day groups with 2+ flight records:",
                f"{len(multi_flight):,}"
            )

            print(
                "Flights in those groups:",
                f"{int(multi_flight.sum()):,}"
            )

    return pd.DataFrame(reports)


def main():
    print("=" * 65)
    print("AERORESILIENCE AI — AIRCRAFT ROTATION DATA AUDIT")
    print("=" * 65)

    reports = []

    for month in MONTHS:
        reports.append(audit_month(month))

    combined = pd.concat(reports, ignore_index=True)

    output = REPORT_DIR / "aircraft_rotation_data_audit.csv"

    combined.to_csv(output, index=False)

    print("\nKEY FIELD COVERAGE")
    key_fields = [
        "Tail_Number",
        "CRSDepTime",
        "CRSArrTime",
        "CRSElapsedTime",
    ]

    print(
        combined[combined["Field"].isin(key_fields)]
        [["Month", "Field", "CoveragePercent"]]
        .to_string(index=False)
    )

    print("\nINTERPRETATION")
    print(
        "Repeated aircraft IDs indicate potential rotation "
        "records, not verified chronological flight chains."
    )
    print(
        "Flight times use airport-local clocks. "
        "Time zones and overnight flights must be handled "
        "before estimating aircraft turnaround dependencies."
    )
    print(
        "Cancelled flights can have missing actual times. "
        "Do not replace these missing values with zero."
    )

    print(f"\nAudit saved: {output}")

    print("\nAERORESILIENCE AI — ROTATION AUDIT COMPLETE")


if __name__ == "__main__":
    main()
