
from pathlib import Path
import pandas as pd
import networkx as nx
import plotly.express as px

# =====================================================
# AERORESILIENCE AI
# STEP 14 — AIRPORT GROUND-STOP EXPOSURE SIMULATOR
# =====================================================

DATA_FILE = Path(
    "data/processed/flights_historical_2026_06_08.parquet"
)

REPORT_DIR = Path("outputs/reports")
FIGURE_DIR = Path("outputs/figures")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

AIRPORT = "ORD"
SCENARIO_DATE = "2026-08-12"
START_HOUR = 10
DURATION_HOURS = 3


def main():

    print("=" * 65)
    print("AERORESILIENCE AI — AIRPORT GROUND-STOP SIMULATION")
    print("=" * 65)

    df = pd.read_parquet(
        DATA_FILE,
        columns=[
            "FlightDate",
            "Origin",
            "Dest",
            "ScheduledDepHour",
            "Cancelled",
            "Diverted",
            "Disrupted",
        ],
    )

    df["FlightDate"] = pd.to_datetime(df["FlightDate"])

    scenario_day = df[
        df["FlightDate"] == pd.Timestamp(SCENARIO_DATE)
    ].copy()

    if scenario_day.empty:
        raise ValueError("No flights found for scenario date.")

    if not 0 <= START_HOUR < 24:
        raise ValueError("Invalid start hour.")

    end_hour = START_HOUR + DURATION_HOURS

    if DURATION_HOURS <= 0 or end_hour > 24:
        raise ValueError(
            "Scenario must fit within one local calendar day."
        )

    # Flight departure hours are local to the origin airport.
    # Restrict the ground-stop window to ORD departures.
    exposed = scenario_day[
        (scenario_day["Origin"] == AIRPORT)
        & (scenario_day["ScheduledDepHour"] >= START_HOUR)
        & (scenario_day["ScheduledDepHour"] < end_hour)
    ].copy()

    if exposed.empty:
        raise ValueError("No exposed departures found.")

    print(f"\nScenario airport: {AIRPORT}")
    print(f"Scenario date: {SCENARIO_DATE}")
    print(f"Ground stop duration: {DURATION_HOURS} hours")
    print(f"Scheduled departures directly exposed: {len(exposed):,}")

    # These are actual historical outcomes, not effects
    # caused by our hypothetical ground stop.
    print(
        f"Historically cancelled among exposed flights: "
        f"{exposed['Cancelled'].sum():,}"
    )

    print(
        f"Historically disrupted among exposed flights: "
        f"{exposed['Disrupted'].sum():,}"
    )

    # ---------------------------------------------
    # 1. Direct destination exposure
    # ---------------------------------------------

    direct = (
        exposed.groupby("Dest")
        .agg(
            ExposedDepartures=("Dest", "size"),
            HistoricallyCancelled=("Cancelled", "sum"),
            HistoricallyDisrupted=("Disrupted", "sum"),
        )
        .reset_index()
        .sort_values("ExposedDepartures", ascending=False)
    )

    direct.to_csv(
        REPORT_DIR / "ground_stop_direct_exposure.csv",
        index=False,
    )

    # ---------------------------------------------
    # 2. Network reachability analysis
    # ---------------------------------------------

    # Use the full historical route network to find
    # possible onward destinations.
    #
    # This is reachability, NOT an estimate of actual
    # cascading delays or missed connections.

    routes = (
        df.groupby(["Origin", "Dest"])
        .size()
        .reset_index(name="HistoricalFlights")
    )

    graph = nx.DiGraph()

    for row in routes.itertuples(index=False):
        graph.add_edge(
            row.Origin,
            row.Dest,
            weight=int(row.HistoricalFlights),
        )

    first_hop = set(direct["Dest"])

    second_hop = set()

    for airport in first_hop:
        if airport in graph:
            second_hop.update(graph.successors(airport))

    second_hop.discard(AIRPORT)

    network_summary = pd.DataFrame([
        {
            "Metric": "Directly exposed departures",
            "Value": len(exposed),
        },
        {
            "Metric": "Direct destination airports",
            "Value": len(first_hop),
        },
        {
            "Metric": "Potential second-hop airports",
            "Value": len(second_hop),
        },
    ])

    network_summary.to_csv(
        REPORT_DIR / "ground_stop_network_summary.csv",
        index=False,
    )

    onward = routes[
        routes["Origin"].isin(first_hop)
    ].copy()

    onward = onward[
        onward["Dest"] != AIRPORT
    ]

    onward = onward.sort_values(
        "HistoricalFlights",
        ascending=False,
    )

    onward.to_csv(
        REPORT_DIR / "ground_stop_onward_network.csv",
        index=False,
    )

    # ---------------------------------------------
    # 3. Direct exposure visualization
    # ---------------------------------------------

    fig = px.bar(
        direct.head(20),
        x="Dest",
        y="ExposedDepartures",
        title=(
            f"ORD Ground Stop — Top Exposed Destinations "
            f"({SCENARIO_DATE})"
        ),
        labels={
            "Dest": "Destination Airport",
            "ExposedDepartures": "Scheduled Departures Exposed",
        },
        template="plotly_dark",
    )

    fig.update_layout(title_x=0.5)

    fig.write_html(
        FIGURE_DIR / "ground_stop_direct_exposure.html"
    )

    # ---------------------------------------------
    # 4. Report
    # ---------------------------------------------

    print("\nTOP DIRECTLY EXPOSED DESTINATIONS")
    print(direct.head(15).to_string(index=False))

    print("\nNETWORK REACHABILITY")
    print(network_summary.to_string(index=False))

    print("\nINTERPRETATION")
    print(
        "Direct exposure represents scheduled departures "
        "within the hypothetical ground-stop window."
    )
    print(
        "Second-hop reachability describes network "
        "connections, not predicted disrupted flights."
    )
    print(
        "No aircraft, crew, passenger, or recovery "
        "decisions are modeled in this initial scenario."
    )

    print("\nReports and chart saved successfully.")
    print(
        "\nAERORESILIENCE AI — GROUND STOP SIMULATION COMPLETE"
    )


if __name__ == "__main__":
    main()
