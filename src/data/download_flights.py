
from pathlib import Path
import argparse
import io
import zipfile

import pandas as pd
import requests

BASE_URL = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present"
)

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def download_month(year, month):
    output_file = RAW_DIR / f"bts_flights_{year}_{month:02d}.parquet"

    if output_file.exists():
        print(f"Already downloaded: {output_file}")
        return

    url = f"{BASE_URL}_{year}_{month}.zip"

    print(f"\nDownloading BTS flights: {year}-{month:02d}")

    response = requests.get(url, timeout=180)
    response.raise_for_status()

    print("Download successful. Extracting data...")

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        csv_files = [
            name for name in archive.namelist()
            if name.lower().endswith(".csv")
        ]

        if not csv_files:
            raise ValueError("No CSV found inside downloaded ZIP.")

        with archive.open(csv_files[0]) as csv_file:
            df = pd.read_csv(csv_file, low_memory=False)

    df = df.loc[
        :, ~df.columns.str.startswith("Unnamed")
    ]

    required = [
        "FlightDate", "Origin", "Dest",
        "DepDelay", "ArrDelay",
        "Cancelled", "Diverted"
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df.to_parquet(output_file, index=False)

    print(f"Records: {len(df):,}")
    print(f"Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--months", type=int, nargs="+", default=[8])

    args = parser.parse_args()

    for month in args.months:
        if not 1 <= month <= 12:
            raise ValueError(f"Invalid month: {month}")

        download_month(args.year, month)

    print("\nAERORESILIENCE AI — HISTORICAL DOWNLOAD COMPLETE")


if __name__ == "__main__":
    main()
