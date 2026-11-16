
"""
AeroResilience AI
Step 67 — Ground-Stop Recovery Decision Scorecard

Uses the existing validated four-policy recovery comparison.

IMPORTANT:
This analysis represents the project's hypothetical ORD
ground-stop experiment, NOT the actual November 13, 2026
nationwide FAA ground-stop event.

Does not modify existing source data or the Streamlit app.
"""

from pathlib import Path
import json

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT / "outputs" / "reports"
    / "validated_network_recovery_comparison.csv"
)

REPORTS = ROOT / "outputs" / "reports"
FIGURES = ROOT / "outputs" / "figures"

POLICIES = [
    "FCFS",
    "Network Priority",
    "Fair",
    "Aircraft Aware",
]

COLORS = {
    "FCFS": "#7B91AC",
    "Network Priority": "#E7A75B",
    "Fair": "#A78BFA",
    "Aircraft Aware": "#25D4C0",
}

REQUIRED = [
    "Policy",
    "TotalModeledDelayMinutes",
    "NoStopDelayMinutes",
    "IncrementalDelayMinutes",
    "DelayedFlights",
    "DownstreamDelayedFlights",
    "IncrementalImprovementPercent",
]


def load_results():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing source report: {INPUT}"
        )

    df = pd.read_csv(INPUT)

    missing = [
        col for col in REQUIRED
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = df[REQUIRED].copy()

    df["Policy"] = df["Policy"].astype(str).str.strip()

    if df["Policy"].duplicated().any():
        raise ValueError("Duplicate policy records found.")

    unavailable = [
        policy for policy in POLICIES
        if policy not in df["Policy"].values
    ]

    if unavailable:
        raise ValueError(
            f"Missing policies: {unavailable}"
        )

    df = df.set_index("Policy").loc[POLICIES].reset_index()

    numeric_columns = [
        col for col in REQUIRED
        if col != "Policy"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="raise",
        )

    if df[numeric_columns].isna().any().any():
        raise ValueError("Missing numeric data.")

    if (df[numeric_columns] < 0).any().any():
        # Negative improvement percentages are valid.
        nonnegative = [
            col for col in numeric_columns
            if col != "IncrementalImprovementPercent"
        ]
        if (df[nonnegative] < 0).any().any():
            raise ValueError("Negative delay or flight counts.")

    difference = (
        df["TotalModeledDelayMinutes"]
        - df["NoStopDelayMinutes"]
        - df["IncrementalDelayMinutes"]
    ).abs()

    if not (difference < 0.01).all():
        raise ValueError(
            "Incremental delay does not reconcile "
            "with total and no-stop delay."
        )

    baseline = float(
        df.loc[
            df["Policy"] == "FCFS",
            "IncrementalDelayMinutes",
        ].iloc[0]
    )

    if baseline <= 0:
        raise ValueError("FCFS baseline must be positive.")

    expected = (
        100 * (
            baseline - df["IncrementalDelayMinutes"]
        ) / baseline
    )

    if not (
        (
            expected - df["IncrementalImprovementPercent"]
        ).abs() < 0.02
    ).all():
        raise ValueError(
            "Saved improvement percentages do not "
            "match calculated values."
        )

    return df


def make_scorecard(df):
    baseline = float(
        df.loc[
            df["Policy"] == "FCFS",
            "IncrementalDelayMinutes",
        ].iloc[0]
    )

    baseline_delayed = int(
        df.loc[
            df["Policy"] == "FCFS",
            "DelayedFlights",
        ].iloc[0]
    )

    baseline_downstream = int(
        df.loc[
            df["Policy"] == "FCFS",
            "DownstreamDelayedFlights",
        ].iloc[0]
    )

    df = df.copy()

    df["DelayMinutesAvoidedVsFCFS"] = (
        baseline - df["IncrementalDelayMinutes"]
    )

    df["DelayedFlightsReducedVsFCFS"] = (
        baseline_delayed - df["DelayedFlights"]
    )

    df["DownstreamFlightsReducedVsFCFS"] = (
        baseline_downstream
        - df["DownstreamDelayedFlights"]
    )

    df["LowestNetworkDelay"] = (
        df["IncrementalDelayMinutes"]
        == df["IncrementalDelayMinutes"].min()
    )

    df["FewestDelayedFlights"] = (
        df["DelayedFlights"]
        == df["DelayedFlights"].min()
    )

    df["FewestDownstreamDelayedFlights"] = (
        df["DownstreamDelayedFlights"]
        == df["DownstreamDelayedFlights"].min()
    )

    return df


def make_figure(df):
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "Incremental Network Delay",
            "Number of Delayed Flights",
        ],
        horizontal_spacing=0.18,
    )

    colors = [
        COLORS[policy]
        for policy in df["Policy"]
    ]

    fig.add_trace(
        go.Bar(
            x=df["IncrementalDelayMinutes"],
            y=df["Policy"],
            orientation="h",
            marker_color=colors,
            text=[
                f"{value:,.0f}"
                for value in df["IncrementalDelayMinutes"]
            ],
            textposition="outside",
            name="Modeled delay minutes",
            hovertemplate=(
                "%{y}<br>"
                "Incremental delay: %{x:,.0f} min"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=df["DelayedFlights"],
            y=df["Policy"],
            orientation="h",
            marker_color=colors,
            text=df["DelayedFlights"],
            textposition="outside",
            name="Delayed flights",
            hovertemplate=(
                "%{y}<br>"
                "Delayed flights: %{x:,.0f}"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title={
            "text": (
                "AeroResilience AI"
                "<br><sup>Ground-Stop Recovery "
                "Decision Scorecard</sup>"
            ),
            "x": 0.04,
        },
        template="plotly_dark",
        paper_bgcolor="#07111F",
        plot_bgcolor="#122238",
        font=dict(color="#E5F2FA"),
        height=540,
        showlegend=False,
        margin=dict(
            l=50,
            r=75,
            t=120,
            b=110,
        ),
        bargap=0.35,
    )

    fig.update_xaxes(
        title_text="Simulated delay minutes",
        row=1,
        col=1,
        gridcolor="#2A4056",
        range=[
            0,
            df["IncrementalDelayMinutes"].max() * 1.25,
        ],
    )

    fig.update_xaxes(
        title_text="Flight count",
        row=1,
        col=2,
        gridcolor="#2A4056",
        range=[
            0,
            df["DelayedFlights"].max() * 1.25,
        ],
    )

    fig.update_yaxes(
        autorange="reversed",
        row=1,
        col=1,
    )

    fig.update_yaxes(
        autorange="reversed",
        row=1,
        col=2,
    )

    fig.add_annotation(
        text=(
            "Historical BTS inputs | "
            "Hypothetical ORD ground-stop simulation | "
            "Not actual November 13 incident results"
        ),
        x=0.5,
        y=-0.24,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            color="#92AFC4",
            size=12,
        ),
    )

    return fig


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    df = load_results()
    scorecard = make_scorecard(df)

    csv_path = (
        REPORTS / "ground_stop_decision_scorecard.csv"
    )

    html_path = (
        FIGURES / "ground_stop_decision_scorecard.html"
    )

    json_path = (
        REPORTS / "ground_stop_scorecard_methodology.json"
    )

    scorecard.to_csv(csv_path, index=False)

    fig = make_figure(scorecard)
    fig.write_html(
        html_path,
        include_plotlyjs=True,
        full_html=True,
    )

    methodology = {
        "Project": "AeroResilience AI",
        "Step": 67,
        "Experiment": (
            "Hypothetical ORD ground-stop recovery"
        ),
        "HistoricalData": (
            "BTS June-August 2026 flight records"
        ),
        "PolicyComparison": POLICIES,
        "BaselinePolicy": "FCFS",
        "Interpretation": (
            "Lower delay and lower delayed-flight counts "
            "are desirable but may favor different policies."
        ),
        "NotRepresented": (
            "Actual November 13, 2026 nationwide "
            "ground-stop recovery outcomes"
        ),
        "Limitations": [
            "Simulated recovery schedules",
            "Simplified operational constraints",
            "No reconstruction of nationwide event",
            "Passenger and crew outcomes are not "
            "evaluated for all four policies",
            "Financial costs are not measured "
            "in this four-policy scorecard",
        ],
    }

    json_path.write_text(
        json.dumps(methodology, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("AERORESILIENCE AI — GROUND-STOP DECISION SCORECARD")
    print("=" * 72)

    columns = [
        "Policy",
        "IncrementalDelayMinutes",
        "DelayedFlights",
        "DownstreamDelayedFlights",
        "IncrementalImprovementPercent",
    ]

    print()
    print(scorecard[columns].to_string(index=False))

    best_delay = scorecard.loc[
        scorecard["IncrementalDelayMinutes"].idxmin()
    ]

    fewest_flights = scorecard.loc[
        scorecard["DelayedFlights"].idxmin()
    ]

    best_downstream = scorecard.loc[
        scorecard["DownstreamDelayedFlights"].idxmin()
    ]

    print()
    print("LOWEST MODELED NETWORK DELAY")
    print(
        f"{best_delay['Policy']}: "
        f"{best_delay['IncrementalDelayMinutes']:,.0f} minutes"
    )

    print()
    print("FEWEST DELAYED FLIGHTS")
    print(
        f"{fewest_flights['Policy']}: "
        f"{fewest_flights['DelayedFlights']:,.0f} flights"
    )

    print()
    print("FEWEST DOWNSTREAM DELAYED FLIGHTS")
    print(
        f"{best_downstream['Policy']}: "
        f"{best_downstream['DownstreamDelayedFlights']:,.0f}"
    )

    print()
    print("IMPORTANT INTERPRETATION")
    print(
        "No policy is assumed to be superior "
        "on every operational metric."
    )
    print(
        "These are hypothetical ORD simulation outcomes, "
        "not results from the November 13 incident."
    )

    print()
    print("FILES SAVED")
    for path in [csv_path, html_path, json_path]:
        print(path.relative_to(ROOT))

    print()
    print(
        "AERORESILIENCE AI — "
        "GROUND-STOP DECISION SCORECARD COMPLETE"
    )


if __name__ == "__main__":
    main()
