
from pathlib import Path

import airportsdata
import numpy as np
import pandas as pd

# ==================================================
# AERORESILIENCE AI
# STEP 18 — MULTI-FLIGHT CASCADING DELAY SIMULATOR
# ==================================================

RAW_FILE = Path("data/raw/bts_flights_2026_08.parquet")
REPORT_DIR = Path("outputs/reports")

REPORT_DIR.mkdir(parents=True, exist_ok=True)

AIRPORT = "ORD"
SCENARIO_DATE = "2026-08-12"
START_HOUR = 10
END_HOUR = 13

MIN_TURN_MINUTES = 20
MAX_CONNECTION_GAP = 600

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


def prepare_flights():
    print("Loading August flight records...")

    df = pd.read_parquet(RAW_FILE, columns=COLUMNS)

    df["FlightDate"] = pd.to_datetime(
        df["FlightDate"], errors="coerce"
    )

    for col in [
        "CRSDepTime", "CRSElapsedTime",
        "Cancelled", "Diverted"
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in [
        "Tail_Number", "Reporting_Airline",
        "Origin", "Dest"
    ]:
        df[col] = df[col].astype("string").str.strip()

    df = df.dropna(subset=[
        "FlightDate", "Tail_Number", "Reporting_Airline",
        "Origin", "Dest", "CRSDepTime",
        "CRSElapsedTime", "Cancelled", "Diverted"
    ]).copy()

    df = df[
        df["Tail_Number"].ne("")
        & df["CRSElapsedTime"].between(1, 1440)
    ].copy()

    hours = df["CRSDepTime"] // 100
    minutes = df["CRSDepTime"] % 100

    valid = (
        hours.between(0, 24)
        & minutes.between(0, 59)
        & ((hours < 24) | (minutes == 0))
    )

    df = df.loc[valid].copy()

    departure_minutes = hours.loc[df.index] * 60 + minutes.loc[df.index]

    df["DepLocal"] = (
        df["FlightDate"]
        + pd.to_timedelta(departure_minutes, unit="m")
    )

    airport_db = airportsdata.load("IATA")

    timezone_lookup = {
        code: record["tz"]
        for code, record in airport_db.items()
        if record.get("tz")
    }

    df["Timezone"] = df["Origin"].map(timezone_lookup)
    df["DepUTC"] = pd.NaT

    for timezone, group in df.groupby("Timezone"):
        try:
            converted = (
                group["DepLocal"]
                .dt.tz_localize(
                    timezone,
                    ambiguous="NaT",
                    nonexistent="NaT"
                )
                .dt.tz_convert("UTC")
                .dt.tz_localize(None)
            )

            df.loc[group.index, "DepUTC"] = converted.to_numpy()

        except (ValueError, TypeError) as error:
            print(f"Timezone issue for {timezone}: {error}")

    df = df.dropna(subset=["DepUTC"]).copy()

    df["ArrUTC"] = (
        df["DepUTC"]
        + pd.to_timedelta(df["CRSElapsedTime"], unit="m")
    )

    df["Eligible"] = (
        (df["Cancelled"] == 0)
        & (df["Diverted"] == 0)
    )

    df["GroundStopSeed"] = (
        (df["Origin"] == AIRPORT)
        & (df["FlightDate"] == pd.Timestamp(SCENARIO_DATE))
        & (hours.reindex(df.index) >= START_HOUR)
        & (hours.reindex(df.index) < END_HOUR)
        & df["Eligible"]
    )

    return df


def run_simulation(df):
    from zoneinfo import ZoneInfo

    local_end = pd.Timestamp(
        f"{SCENARIO_DATE} {END_HOUR:02d}:00:00"
    ).tz_localize(ZoneInfo("America/Chicago"))

    stop_end_utc = (
        local_end.tz_convert("UTC").tz_localize(None)
    )

    seeds = df[df["GroundStopSeed"]].copy()

    if seeds.empty:
        raise ValueError("No eligible ORD departures in scenario.")

    print(f"Ground-stop seed flights: {len(seeds):,}")

    # Only process aircraft involved in the ground stop.
    affected_aircraft = seeds[
        ["Reporting_Airline", "Tail_Number"]
    ].drop_duplicates()

    df = df.merge(
        affected_aircraft,
        on=["Reporting_Airline", "Tail_Number"],
        how="inner"
    )

    df = df.sort_values([
        "Reporting_Airline", "Tail_Number", "DepUTC"
    ]).reset_index(drop=True)

    results = []

    for (_, _), group in df.groupby(
        ["Reporting_Airline", "Tail_Number"],
        sort=False
    ):
        previous = None
        previous_delay = 0.0
        previous_hop = 0

        for flight in group.itertuples(index=False):
            if not flight.Eligible:
                previous = None
                previous_delay = 0.0
                previous_hop = 0
                continue

            inherited_delay = 0.0
            inherited_hop = 0

            if previous is not None:
                gap = (
                    flight.DepUTC - previous.ArrUTC
                ).total_seconds() / 60

                valid_link = (
                    previous.Dest == flight.Origin
                    and MIN_TURN_MINUTES <= gap <= MAX_CONNECTION_GAP
                )

                if valid_link and previous_delay > 0:
                    slack = max(gap - MIN_TURN_MINUTES, 0)

                    inherited_delay = max(
                        previous_delay - slack, 0
                    )

                    if inherited_delay > 0:
                        inherited_hop = previous_hop + 1

            direct_delay = 0.0

            if flight.GroundStopSeed:
                direct_delay = max(
                    (stop_end_utc - flight.DepUTC).total_seconds() / 60,
                    0
                )

            delay = max(direct_delay, inherited_delay)

            if delay > 0:
                direct = bool(flight.GroundStopSeed)

                results.append({
                    "FlightDate": flight.FlightDate,
                    "Airline": flight.Reporting_Airline,
                    "TailNumber": flight.Tail_Number,
                    "Origin": flight.Origin,
                    "Dest": flight.Dest,
                    "ScheduledDepUTC": flight.DepUTC,
                    "SimulatedDelayMinutes": round(delay, 2),
                    "DirectGroundStop": direct,
                    "PropagationHop": (
                        0 if direct and direct_delay >= inherited_delay
                        else inherited_hop
                    ),
                })

            previous = flight
            previous_delay = delay
            previous_hop = (
                0 if flight.GroundStopSeed and direct_delay >= inherited_delay
                else inherited_hop
            )

    return pd.DataFrame(results)


def main():
    print("=" * 65)
    print("AERORESILIENCE AI — CASCADING DISRUPTION SIMULATION")
    print("=" * 65)

    if not 0 <= START_HOUR < END_HOUR <= 24:
        raise ValueError("Invalid ground-stop window.")

    df = prepare_flights()
    results = run_simulation(df)

    if results.empty:
        raise ValueError("Simulation produced no delayed flights.")

    direct = results[results["DirectGroundStop"]]
    downstream = results[~results["DirectGroundStop"]]

    total_delay = results["SimulatedDelayMinutes"].sum()

    print("\nSCENARIO RESULTS")
    print(f"Airport: {AIRPORT}")
    print(f"Date: {SCENARIO_DATE}")
    print(f"Ground stop: {START_HOUR}:00–{END_HOUR}:00 local")
    print(f"Directly delayed departures: {len(direct):,}")
    print(f"Downstream delayed flights: {len(downstream):,}")
    print(f"Total modeled delayed flights: {len(results):,}")
    print(f"Total modeled delay minutes: {total_delay:,.0f}")
    print(
        "Maximum propagation hop:",
        results["PropagationHop"].max()
    )

    summary = pd.DataFrame([{
        "Airport": AIRPORT,
        "ScenarioDate": SCENARIO_DATE,
        "DirectDelayedFlights": len(direct),
        "DownstreamDelayedFlights": len(downstream),
        "TotalDelayedFlights": len(results),
        "TotalModeledDelayMinutes": total_delay,
        "MaxPropagationHop": int(results["PropagationHop"].max())
    }])

    results.to_csv(
        REPORT_DIR / "cascade_flight_details.csv",
        index=False
    )

    summary.to_csv(
        REPORT_DIR / "cascade_simulation_summary.csv",
        index=False
    )

    if not downstream.empty:
        impact = (
            downstream.groupby("Origin")
            .agg(
                DownstreamFlights=("Origin", "size"),
                TotalDelayMinutes=("SimulatedDelayMinutes", "sum")
            )
            .sort_values("TotalDelayMinutes", ascending=False)
            .reset_index()
        )

        impact.to_csv(
            REPORT_DIR / "cascade_airport_impact.csv",
            index=False
        )

        print("\nTOP DOWNSTREAM AIRPORTS")
        print(impact.head(10).to_string(index=False))

    print("\nSIMULATION ASSUMPTIONS")
    print("- Ground stop applies to ORD departures only.")
    print("- Affected departures cannot leave before 1 PM.")
    print("- Same-aircraft assignments are assumed.")
    print("- Scheduled flight durations remain unchanged.")
    print("- Minimum turnaround is assumed to be 20 minutes.")
    print("- No crew, airspace, gate, or passenger constraints.")
    print("- Cancelled and diverted flights break chains.")
    print("- Results are simulated, not observed disruptions.")

    print("\nAERORESILIENCE AI — CASCADE SIMULATION COMPLETE")


if __name__ == "__main__":
    main()
