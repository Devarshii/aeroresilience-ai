
from pathlib import Path
import pandas as pd
import plotly.express as px

DATA_FILE = Path("data/processed/flights_clean_2026_08.parquet")
OUTPUT_DIR = Path("outputs/figures")
REPORT_DIR = Path("outputs/reports")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def save_chart(fig, filename):
    fig.update_layout(
        template="plotly_dark",
        title_x=0.5,
        font=dict(size=13),
        margin=dict(l=50, r=30, t=80, b=50),
    )
    fig.write_html(OUTPUT_DIR / filename)
    print(f"Saved chart: {filename}")


def main():
    print("=" * 60)
    print("AERORESILIENCE AI — DISRUPTION ANALYSIS")
    print("=" * 60)

    df = pd.read_parquet(DATA_FILE)

    print(f"\nTotal flights: {len(df):,}")
    print(f"Disrupted flights: {df['Disrupted'].sum():,}")
    print(f"Disruption rate: {df['Disrupted'].mean():.2%}")

    # 1. Airport disruption analysis
    airports = (
        df.groupby("Origin")
        .agg(
            TotalFlights=("Disrupted", "size"),
            DisruptedFlights=("Disrupted", "sum"),
            CancelledFlights=("Cancelled", "sum"),
        )
        .reset_index()
    )

    airports["DisruptionRate"] = (
        airports["DisruptedFlights"]
        / airports["TotalFlights"] * 100
    )

    # Only compare airports with at least 1,000 flights
    major_airports = airports[
        airports["TotalFlights"] >= 1000
    ].sort_values("DisruptionRate", ascending=False)

    major_airports.to_csv(
        REPORT_DIR / "airport_disruptions.csv",
        index=False,
    )

    fig = px.bar(
        major_airports.head(15),
        x="Origin",
        y="DisruptionRate",
        title="Top 15 Major Airports by Flight Disruption Rate",
        labels={
            "Origin": "Origin Airport",
            "DisruptionRate": "Disruption Rate (%)",
        },
        hover_data=["TotalFlights", "DisruptedFlights"],
    )
    save_chart(fig, "airport_disruptions.html")

    # 2. Airline disruption analysis
    airlines = (
        df.groupby("Reporting_Airline")
        .agg(
            TotalFlights=("Disrupted", "size"),
            DisruptedFlights=("Disrupted", "sum"),
            CancelledFlights=("Cancelled", "sum"),
        )
        .reset_index()
    )

    airlines["DisruptionRate"] = (
        airlines["DisruptedFlights"]
        / airlines["TotalFlights"] * 100
    )

    airlines = airlines.sort_values(
        "DisruptionRate", ascending=False
    )

    airlines.to_csv(
        REPORT_DIR / "airline_disruptions.csv",
        index=False,
    )

    fig = px.bar(
        airlines,
        x="Reporting_Airline",
        y="DisruptionRate",
        title="Flight Disruption Rate by Reporting Airline",
        labels={
            "Reporting_Airline": "Airline",
            "DisruptionRate": "Disruption Rate (%)",
        },
        hover_data=["TotalFlights", "CancelledFlights"],
    )
    save_chart(fig, "airline_disruptions.html")

    # 3. Hourly disruption patterns
    hourly_df = df.dropna(
        subset=["ScheduledDepHour"]
    ).copy()

    hourly = (
        hourly_df.groupby("ScheduledDepHour")
        .agg(
            TotalFlights=("Disrupted", "size"),
            DisruptedFlights=("Disrupted", "sum"),
        )
        .reset_index()
    )

    hourly["DisruptionRate"] = (
        hourly["DisruptedFlights"]
        / hourly["TotalFlights"] * 100
    )

    hourly.to_csv(
        REPORT_DIR / "hourly_disruptions.csv",
        index=False,
    )

    fig = px.line(
        hourly,
        x="ScheduledDepHour",
        y="DisruptionRate",
        markers=True,
        title="Flight Disruption Rate by Scheduled Departure Hour",
        labels={
            "ScheduledDepHour": "Scheduled Departure Hour (Local)",
            "DisruptionRate": "Disruption Rate (%)",
        },
    )
    save_chart(fig, "hourly_disruptions.html")

    # 4. Route disruption analysis
    df["Route"] = df["Origin"] + " → " + df["Dest"]

    routes = (
        df.groupby("Route")
        .agg(
            TotalFlights=("Disrupted", "size"),
            DisruptedFlights=("Disrupted", "sum"),
        )
        .reset_index()
    )

    routes["DisruptionRate"] = (
        routes["DisruptedFlights"]
        / routes["TotalFlights"] * 100
    )

    routes = routes[
        routes["TotalFlights"] >= 100
    ].sort_values("DisruptionRate", ascending=False)

    routes.to_csv(
        REPORT_DIR / "route_disruptions.csv",
        index=False,
    )

    fig = px.bar(
        routes.head(15),
        x="Route",
        y="DisruptionRate",
        title="Top 15 High-Disruption Routes (100+ Flights)",
        hover_data=["TotalFlights", "DisruptedFlights"],
    )
    save_chart(fig, "route_disruptions.html")

    # 5. Arrival delay analysis
    completed = df[
        (df["Cancelled"] == 0)
        & (df["Diverted"] == 0)
        & df["ArrDelay"].notna()
    ].copy()

    completed["PositiveDelayMinutes"] = (
        completed["ArrDelay"].clip(lower=0)
    )

    delay_by_airport = (
        completed.groupby("Origin")
        .agg(
            TotalPositiveDelayMinutes=(
                "PositiveDelayMinutes", "sum"
            ),
            AverageArrivalDelay=("ArrDelay", "mean"),
        )
        .reset_index()
        .sort_values(
            "TotalPositiveDelayMinutes", ascending=False
        )
    )

    delay_by_airport.to_csv(
        REPORT_DIR / "airport_delay_impact.csv",
        index=False,
    )

    fig = px.bar(
        delay_by_airport.head(15),
        x="Origin",
        y="TotalPositiveDelayMinutes",
        title="Top 15 Origin Airports by Total Arrival Delay Minutes",
    )
    save_chart(fig, "airport_delay_impact.html")

    print("\nTOP 10 MAJOR AIRPORTS BY DISRUPTION RATE")
    print(major_airports.head(10).to_string(index=False))

    print("\nTOP 10 ROUTES BY DISRUPTION RATE")
    print(routes.head(10).to_string(index=False))

    print("\nAERORESILIENCE AI — EDA COMPLETE")
    print(f"Charts saved to: {OUTPUT_DIR}")
    print(f"Reports saved to: {REPORT_DIR}")


if __name__ == "__main__":
    main()
