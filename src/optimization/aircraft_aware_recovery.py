
from pathlib import Path
import airportsdata
import numpy as np
import pandas as pd

# =====================================================
# AERORESILIENCE AI
# STEP 26 — MULTI-FLIGHT AIRCRAFT RECOVERY SIMULATOR
# =====================================================

RAW_FILE = Path("data/raw/bts_flights_2026_08.parquet")
RECOVERY_FILE = Path(
    "outputs/reports/fair_recovery_flights.csv"
)

REPORT_DIR = Path("outputs/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MIN_TURN = 20
MAX_GAP = 600

KEYS = [
    "FlightDate",
    "Reporting_Airline",
    "Tail_Number",
    "Flight_Number_Reporting_Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
]

POLICIES = {
    "FCFS": "BaselineDelayMinutes",
    "Optimized": "OptimizedDelayMinutes",
    "Fair": "FairDelayMinutes",
}


def load_flights():
    print("Loading aircraft schedules...")

    columns = KEYS + [
        "CRSElapsedTime",
        "Cancelled",
        "Diverted",
    ]

    raw = pd.read_parquet(RAW_FILE, columns=columns)
    recovery = pd.read_csv(RECOVERY_FILE)

    for frame in (raw, recovery):
        frame["FlightDate"] = pd.to_datetime(
            frame["FlightDate"], errors="coerce"
        )

        for col in [
            "Reporting_Airline",
            "Tail_Number",
            "Origin",
            "Dest",
        ]:
            frame[col] = (
                frame[col].astype("string").str.strip()
            )

        for col in [
            "CRSDepTime",
            "Flight_Number_Reporting_Airline",
        ]:
            frame[col] = pd.to_numeric(
                frame[col], errors="coerce"
            )

    for col in [
        "CRSElapsedTime",
        "Cancelled",
        "Diverted",
    ]:
        raw[col] = pd.to_numeric(
            raw[col], errors="coerce"
        )

    for col in POLICIES.values():
        recovery[col] = pd.to_numeric(
            recovery[col], errors="coerce"
        )

    if recovery[
        list(POLICIES.values())
    ].isna().any().any():
        raise ValueError("Recovery delays contain missing values.")

    if recovery[KEYS].isna().any().any():
        raise ValueError("Recovery flight keys are incomplete.")

    if recovery.duplicated(KEYS).any():
        raise ValueError("Duplicate recovery flight keys.")

    affected = recovery[
        ["Reporting_Airline", "Tail_Number"]
    ].drop_duplicates()

    raw = raw.merge(
        affected,
        on=["Reporting_Airline", "Tail_Number"],
        how="inner",
    )

    raw = raw.dropna(
        subset=KEYS + ["CRSElapsedTime"]
    ).copy()

    if raw.duplicated(KEYS).any():
        raise ValueError(
            "Ambiguous raw schedule records. "
            "Resolve duplicate keys before continuing."
        )

    raw = raw.merge(
        recovery[
            KEYS + list(POLICIES.values())
        ],
        on=KEYS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    matched = int((raw["_merge"] == "both").sum())

    if matched != len(recovery):
        raise ValueError(
            f"Matched {matched} recovery flights, "
            f"expected {len(recovery)}."
        )

    raw["RecoverySeed"] = raw["_merge"].eq("both")
    raw = raw.drop(columns=["_merge"])

    for col in POLICIES.values():
        raw[col] = raw[col].fillna(0.0)

    raw["Operating"] = (
        raw["Cancelled"].eq(0)
        & raw["Diverted"].eq(0)
    )

    print(f"Recovery flights matched: {matched:,}")
    print(f"Affected-aircraft schedule rows: {len(raw):,}")

    return raw


def create_utc_schedule(df):
    print("Converting scheduled times to UTC...")

    airport_db = airportsdata.load("IATA")

    timezone_lookup = {
        code: record["tz"]
        for code, record in airport_db.items()
        if record.get("tz")
    }

    df = df.copy()

    hours = df["CRSDepTime"] // 100
    minutes = df["CRSDepTime"] % 100

    valid_time = (
        hours.between(0, 24)
        & minutes.between(0, 59)
        & ((hours < 24) | (minutes == 0))
    )

    if not valid_time.all():
        raise ValueError(
            "Invalid scheduled times in aircraft records."
        )

    local_minutes = hours * 60 + minutes

    df["LocalDeparture"] = (
        df["FlightDate"]
        + pd.to_timedelta(local_minutes, unit="m")
    )

    df["Timezone"] = df["Origin"].map(timezone_lookup)
    df["DepUTC"] = pd.NaT

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

        df.loc[group.index, "DepUTC"] = (
            converted.to_numpy()
        )

    if df["DepUTC"].isna().any():
        missing_seeds = int(
            df.loc[df["DepUTC"].isna(), "RecoverySeed"].sum()
        )
        if missing_seeds:
            raise ValueError(
                "UTC conversion failed for recovery flights."
            )
        print(
            "Non-seed records excluded due to missing UTC:",
            int(df["DepUTC"].isna().sum())
        )

    df = df.dropna(subset=["DepUTC"]).copy()

    df["ArrUTC"] = (
        df["DepUTC"]
        + pd.to_timedelta(
            df["CRSElapsedTime"], unit="m"
        )
    )

    return df.sort_values(
        [
            "Reporting_Airline",
            "Tail_Number",
            "DepUTC",
        ]
    ).reset_index(drop=True)


def simulate_policy(df, name, delay_column):
    results = []

    for (airline, tail), group in df.groupby(
        ["Reporting_Airline", "Tail_Number"],
        sort=False,
    ):
        previous_arrival = None
        previous_dest = None
        previous_delay = 0.0
        previous_hop = 0

        for flight in group.itertuples(index=False):

            if not flight.Operating:
                previous_arrival = None
                previous_dest = None
                previous_delay = 0.0
                previous_hop = 0
                continue

            direct_delay = float(
                getattr(flight, delay_column)
            )

            inherited = 0.0
            hop = 0

            if previous_arrival is not None:
                gap = (
                    flight.DepUTC - previous_arrival
                ).total_seconds() / 60

                valid_link = (
                    previous_dest == flight.Origin
                    and 0 <= gap <= MAX_GAP
                )

                if valid_link:
                    inherited = max(
                        0.0,
                        previous_delay
                        + MIN_TURN
                        - gap
                    )

                    if inherited > 0:
                        hop = previous_hop + 1

            total_delay = max(
                direct_delay, inherited
            )

            # All rows, including unaffected flights,
            # are retained for policy comparison.
            results.append({
                "Policy": name,
                "Airline": airline,
                "TailNumber": tail,
                "FlightDate": flight.FlightDate,
                "Origin": flight.Origin,
                "Dest": flight.Dest,
                "ScheduledDepUTC": flight.DepUTC,
                "RecoverySeed": bool(flight.RecoverySeed),
                "DirectDelay": round(direct_delay, 2),
                "InheritedDelay": round(inherited, 2),
                "TotalDelay": round(total_delay, 2),
                "PropagationHop": (
                    0 if direct_delay >= inherited
                    else hop
                ),
            })

            previous_arrival = flight.ArrUTC
            previous_dest = flight.Dest
            previous_delay = total_delay

            previous_hop = (
                0 if direct_delay >= inherited
                else hop
            )

    return pd.DataFrame(results)


def main():
    print("=" * 65)
    print("AERORESILIENCE AI — AIRCRAFT-AWARE RECOVERY")
    print("=" * 65)

    df = load_flights()
    df = create_utc_schedule(df)

    all_results = []
    summaries = []

    for policy, column in POLICIES.items():
        print(f"\nSimulating {policy} recovery...")

        result = simulate_policy(df, policy, column)
        all_results.append(result)

        delayed = result["TotalDelay"] > 0

        downstream = (
            delayed & ~result["RecoverySeed"]
        )

        summaries.append({
            "Policy": policy,
            "OperatingFlights": len(result),
            "DelayedFlights": int(delayed.sum()),
            "DownstreamDelayedFlights": int(
                downstream.sum()
            ),
            "TotalDelayMinutes": float(
                result["TotalDelay"].sum()
            ),
            "DownstreamDelayMinutes": float(
                result.loc[
                    downstream, "TotalDelay"
                ].sum()
            ),
            "MaximumDelayMinutes": float(
                result["TotalDelay"].max()
            ),
        })

    details = pd.concat(
        all_results, ignore_index=True
    )

    summary = pd.DataFrame(summaries)

    baseline = summary.loc[
        summary["Policy"] == "FCFS",
        "TotalDelayMinutes"
    ].iloc[0]

    summary["ImprovementVsFCFSPercent"] = (
        100 * (
            baseline - summary["TotalDelayMinutes"]
        ) / baseline
        if baseline > 0 else 0.0
    )

    details.to_parquet(
        REPORT_DIR / "aircraft_aware_recovery_details.parquet",
        index=False
    )

    summary.to_csv(
        REPORT_DIR / "aircraft_aware_recovery_comparison.csv",
        index=False
    )

    print("\nMULTI-FLIGHT RECOVERY COMPARISON")
    print(summary.to_string(index=False))

    print("\nSIMULATION LIMITATIONS")
    print(
        "Historical aircraft assignments are assumed fixed."
    )
    print(
        "Delays follow consecutive scheduled flights "
        "on the same aircraft where airport continuity holds."
    )
    print(
        "Cancelled or diverted flights break rotation chains."
    )
    print(
        "Twenty-minute minimum turnaround is an assumption."
    )
    print(
        "Recovered departures may be pushed beyond their "
        "assigned slots by upstream aircraft conflicts."
    )
    print(
        "Downstream airport slot capacity, gates, crews, "
        "passengers, and substitutions are not modeled."
    )
    print(
        "This is a diagnostic propagation simulation, "
        "not a globally feasible recovery schedule."
    )

    print("\nReports saved successfully.")
    print(
        "\nAERORESILIENCE AI — "
        "AIRCRAFT-AWARE RECOVERY COMPLETE"
    )


if __name__ == "__main__":
    main()
