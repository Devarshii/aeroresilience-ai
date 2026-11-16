
from pathlib import Path
import pandas as pd
import numpy as np

# =====================================================
# AERORESILIENCE AI
# STEP 20 — AIRLINE RECOVERY BASELINE
# =====================================================

DATA_FILE = Path(
    "data/raw/bts_flights_2026_08.parquet"
)

REPORT_DIR = Path("outputs/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

AIRPORT = "ORD"
SCENARIO_DATE = "2026-08-12"

STOP_START = 10 * 60
STOP_END = 13 * 60

SLOT_MINUTES = 15
DEPARTURES_PER_SLOT = 8

COLUMNS = [
    "FlightDate",
    "Reporting_Airline",
    "Flight_Number_Reporting_Airline",
    "Tail_Number",
    "Origin",
    "Dest",
    "CRSDepTime",
    "Cancelled",
    "Diverted",
]


def prepare_exposed_flights():

    print("\nLoading official BTS flight records...")

    df = pd.read_parquet(
        DATA_FILE,
        columns=COLUMNS,
    )

    df["FlightDate"] = pd.to_datetime(
        df["FlightDate"]
    )

    for col in ["CRSDepTime", "Cancelled", "Diverted"]:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    df = df[
        (df["FlightDate"] == pd.Timestamp(SCENARIO_DATE))
        & (df["Origin"] == AIRPORT)
        & (df["Cancelled"] == 0)
        & (df["Diverted"] == 0)
    ].copy()

    df = df.dropna(
        subset=["CRSDepTime"]
    )

    hours = df["CRSDepTime"] // 100
    minutes = df["CRSDepTime"] % 100

    df["ScheduledMinute"] = (
        hours * 60 + minutes
    )

    exposed = df[
        (df["ScheduledMinute"] >= STOP_START)
        & (df["ScheduledMinute"] < STOP_END)
    ].copy()

    exposed = exposed.sort_values(
        [
            "ScheduledMinute",
            "Reporting_Airline",
            "Flight_Number_Reporting_Airline",
        ],
        kind="stable",
    ).reset_index(drop=True)

    exposed["RecoveryFlightID"] = np.arange(
        1, len(exposed) + 1
    )

    return exposed


def build_baseline(df):

    print("\nBuilding first-come, first-served recovery...")

    if df.empty:
        raise ValueError(
            "No eligible flights in the ground-stop window."
        )

    result = df.copy()

    # Each 15-minute window has 8 hypothetical
    # departure positions.
    #
    # We assign flights in original scheduled order.
    positions = np.arange(len(result))

    slot_number = (
        positions // DEPARTURES_PER_SLOT
    )

    result["RecoveryDepartureMinute"] = (
        STOP_END + slot_number * SLOT_MINUTES
    )

    result["BaselineDelayMinutes"] = (
        result["RecoveryDepartureMinute"]
        - result["ScheduledMinute"]
    )

    result["RecoverySlot"] = slot_number + 1

    assert (
        result["RecoveryDepartureMinute"] >= STOP_END
    ).all()

    assert (
        result["BaselineDelayMinutes"] >= 0
    ).all()

    slot_counts = result.groupby(
        "RecoverySlot"
    ).size()

    assert (
        slot_counts <= DEPARTURES_PER_SLOT
    ).all()

    return result


def main():

    print("=" * 65)
    print("AERORESILIENCE AI — RECOVERY BASELINE")
    print("=" * 65)

    exposed = prepare_exposed_flights()
    baseline = build_baseline(exposed)

    total_flights = len(baseline)

    total_delay = float(
        baseline["BaselineDelayMinutes"].sum()
    )

    avg_delay = float(
        baseline["BaselineDelayMinutes"].mean()
    )

    max_delay = float(
        baseline["BaselineDelayMinutes"].max()
    )

    recovery_minutes = float(
        baseline["RecoveryDepartureMinute"].max()
        - STOP_END + SLOT_MINUTES
    )

    summary = pd.DataFrame([{
        "Airport": AIRPORT,
        "ScenarioDate": SCENARIO_DATE,
        "GroundStopHours": (
            STOP_END - STOP_START
        ) / 60,
        "DeparturesPerSlot": DEPARTURES_PER_SLOT,
        "SlotMinutes": SLOT_MINUTES,
        "FlightsRecovered": total_flights,
        "TotalDelayMinutes": total_delay,
        "AverageDelayMinutes": avg_delay,
        "MaximumDelayMinutes": max_delay,
        "RecoveryWindowMinutes": recovery_minutes,
    }])

    baseline.to_csv(
        REPORT_DIR / "recovery_baseline_flights.csv",
        index=False
    )

    summary.to_csv(
        REPORT_DIR / "recovery_baseline_summary.csv",
        index=False
    )

    print("\nSCENARIO CONFIGURATION")
    print(f"Airport: {AIRPORT}")
    print(f"Date: {SCENARIO_DATE}")
    print("Ground stop: 10:00 AM – 1:00 PM")
    print(
        f"Capacity: {DEPARTURES_PER_SLOT} "
        f"departures every {SLOT_MINUTES} minutes"
    )

    print("\nBASELINE RECOVERY RESULTS")
    print(f"Flights scheduled for recovery: {total_flights:,}")
    print(f"Total modeled delay minutes: {total_delay:,.0f}")
    print(f"Average delay per flight: {avg_delay:.2f} min")
    print(f"Maximum delay: {max_delay:.2f} min")
    print(f"Recovery window: {recovery_minutes:.0f} min")

    print("\nFIRST 10 RECOVERY ASSIGNMENTS")

    print(
        baseline[
            [
                "RecoveryFlightID",
                "Reporting_Airline",
                "Origin",
                "Dest",
                "ScheduledMinute",
                "RecoveryDepartureMinute",
                "BaselineDelayMinutes",
            ]
        ].head(10).to_string(index=False)
    )

    print("\nIMPORTANT LIMITATIONS")
    print(
        "This is a hypothetical FCFS departure-slot baseline."
    )
    print(
        "It does not represent an actual ORD recovery plan."
    )
    print(
        "Existing departures after 1 PM, runway capacity, "
        "crew, gates, and aircraft repositioning are "
        "not yet modeled."
    )
    print(
        "All selected flights are assumed recoverable. "
        "Historical cancellations and diversions are excluded."
    )

    print("\nReports saved successfully.")

    print(
        "\nAERORESILIENCE AI — RECOVERY BASELINE COMPLETE"
    )


if __name__ == "__main__":
    main()
