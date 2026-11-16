
from pathlib import Path
from math import ceil

import pandas as pd
import plotly.express as px

# =====================================================
# AERORESILIENCE AI
# STEP 31 — MULTI-AIRPORT GROUND-STOP SCENARIOS
# =====================================================

DATA_FILE = Path(
    "data/raw/bts_flights_2026_08.parquet"
)

REPORT_DIR = Path("outputs/reports")
FIGURE_DIR = Path("outputs/figures")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

AIRPORTS = ["ORD", "DFW", "DEN", "ATL"]
DATES = ["2026-08-12", "2026-08-19"]
DURATIONS = [2, 3, 4]

START_HOUR = 10

# Experimental recovery assumptions.
SLOT_MINUTES = 15
DEPARTURES_PER_SLOT = 8


def main():

    print("=" * 65)
    print("AERORESILIENCE AI — MULTI-AIRPORT SCENARIOS")
    print("=" * 65)

    if not DATA_FILE.exists():
        raise FileNotFoundError(DATA_FILE)

    df = pd.read_parquet(
        DATA_FILE,
        columns=[
            "FlightDate",
            "Reporting_Airline",
            "Tail_Number",
            "Origin",
            "Dest",
            "CRSDepTime",
            "Cancelled",
            "Diverted",
        ],
    )

    df["FlightDate"] = pd.to_datetime(
        df["FlightDate"], errors="coerce"
    )

    for col in ["CRSDepTime", "Cancelled", "Diverted"]:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    df = df.dropna(
        subset=[
            "FlightDate",
            "Origin",
            "Dest",
            "CRSDepTime",
            "Cancelled",
            "Diverted",
        ]
    ).copy()

    hours = df["CRSDepTime"] // 100
    minutes = df["CRSDepTime"] % 100

    valid_time = (
        hours.between(0, 24)
        & minutes.between(0, 59)
        & ((hours < 24) | (minutes == 0))
    )

    df = df.loc[valid_time].copy()

    df["ScheduledMinute"] = (
        hours.loc[df.index] * 60
        + minutes.loc[df.index]
    )

    results = []
    scenario_id = 0

    for airport in AIRPORTS:
        for date in DATES:
            for duration in DURATIONS:

                scenario_id += 1

                start = START_HOUR * 60
                end = start + duration * 60

                if end > 1440:
                    raise ValueError(
                        "Ground stop exceeds calendar day."
                    )

                exposed = df[
                    (df["Origin"] == airport)
                    & (df["FlightDate"] == pd.Timestamp(date))
                    & (df["ScheduledMinute"] >= start)
                    & (df["ScheduledMinute"] < end)
                ].copy()

                # Match the eligibility definition
                # from our existing recovery baseline.
                eligible = exposed[
                    (exposed["Cancelled"] == 0)
                    & (exposed["Diverted"] == 0)
                ].copy()

                capacity_slots = ceil(
                    len(eligible) / DEPARTURES_PER_SLOT
                )

                backlog_minutes = (
                    capacity_slots * SLOT_MINUTES
                )

                results.append({
                    "ScenarioID": scenario_id,
                    "Airport": airport,
                    "Date": date,
                    "StartHourLocal": START_HOUR,
                    "DurationHours": duration,
                    "ScheduledExposedFlights": len(exposed),
                    "EligibleRecoveryFlights": len(eligible),
                    "HistoricallyCancelled": int(
                        exposed["Cancelled"].sum()
                    ),
                    "HistoricallyDiverted": int(
                        exposed["Diverted"].sum()
                    ),
                    "AffectedAircraftIDs": (
                        eligible["Tail_Number"].nunique()
                    ),
                    "DestinationAirports": (
                        eligible["Dest"].nunique()
                    ),
                    "Airlines": (
                        eligible["Reporting_Airline"].nunique()
                    ),
                    "MinimumRecoverySlots": capacity_slots,
                    "IdealizedBacklogMinutes": backlog_minutes,
                })

    summary = pd.DataFrame(results)

    if summary.empty:
        raise ValueError("No scenarios generated.")

    # Verify the original ORD recovery scenario.
    original = summary[
        (summary["Airport"] == "ORD")
        & (summary["Date"] == "2026-08-12")
        & (summary["DurationHours"] == 3)
    ]

    if len(original) != 1:
        raise RuntimeError(
            "Original ORD scenario missing."
        )

    actual_count = int(
        original["EligibleRecoveryFlights"].iloc[0]
    )

    if actual_count != 177:
        raise RuntimeError(
            "Original scenario mismatch: "
            f"expected 177 eligible flights, got {actual_count}."
        )

    report_file = (
        REPORT_DIR / "multi_airport_scenarios.csv"
    )

    summary.to_csv(report_file, index=False)

    fig = px.bar(
        summary,
        x="Airport",
        y="EligibleRecoveryFlights",
        color="DurationHours",
        facet_col="Date",
        barmode="group",
        title="Ground-Stop Scenarios — Eligible Flights",
        labels={
            "EligibleRecoveryFlights":
                "Eligible Recovery Flights",
            "DurationHours": "Stop Duration (Hours)",
        },
        template="plotly_dark",
    )

    fig.update_layout(title_x=0.5)

    chart_file = (
        FIGURE_DIR / "multi_airport_scenarios.html"
    )

    fig.write_html(chart_file)

    print("\nSCENARIO COVERAGE")
    print(f"Scenarios generated: {len(summary)}")
    print(f"Airports evaluated: {len(AIRPORTS)}")
    print(f"Dates evaluated: {len(DATES)}")

    print("\nFLIGHT EXPOSURE COMPARISON")
    print(
        summary[
            [
                "Airport",
                "Date",
                "DurationHours",
                "ScheduledExposedFlights",
                "EligibleRecoveryFlights",
                "AffectedAircraftIDs",
                "IdealizedBacklogMinutes",
            ]
        ].to_string(index=False)
    )

    print("\nORIGINAL ORD SCENARIO VERIFIED")
    print(f"Eligible flights: {actual_count}")

    print("\nIMPORTANT LIMITATIONS")
    print(
        "These are hypothetical disruption scenarios "
        "applied to historical schedules."
    )
    print(
        "Historical cancellations and diversions are "
        "not attributed to the simulated ground stop."
    )
    print(
        "Backlog minutes assume an isolated departure "
        "queue with 8 positions per 15-minute slot."
    )
    print(
        "Existing airport traffic, aircraft connections, "
        "crews and gates are not yet included."
    )
    print(
        "This step prepares scenarios; it does not "
        "evaluate optimized recovery performance."
    )

    print(f"\nReport saved: {report_file}")
    print(f"Chart saved: {chart_file}")

    print(
        "\nAERORESILIENCE AI — "
        "MULTI-AIRPORT SCENARIOS COMPLETE"
    )


if __name__ == "__main__":
    main()
