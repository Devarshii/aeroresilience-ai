
"""
AeroResilience AI
Step 65 — U.S. Ground Stop Recovery Timeline

Real incident milestones are based on FAA and DOT statements.
Recovery stages are conceptual and are NOT actual observed
airline recovery timings.

Outputs:
- outputs/reports/ground_stop_event_timeline.csv
- outputs/figures/ground_stop_recovery_timeline.html
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "outputs" / "reports"
FIGURES = ROOT / "outputs" / "figures"

FAA_SOURCE = (
    "https://www.faa.gov/newsroom/"
    "nationwide-ground-stop-questions-and-answers"
)

DOT_SOURCE = (
    "https://www.transportation.gov/briefing-room/"
    "icymi-us-transportation-secretary-sean-p-duffy-"
    "provides-update-restoration-airspace"
)


def build_event_data():

    events = [
        {
            "EventDate": "2026-11-13",
            "EventTimeET": "15:30",
            "Event": "Nationwide Ground Stop",
            "Description": (
                "FAA technical instability prompted "
                "a nationwide ground stop."
            ),
            "EvidenceType": "Reported incident",
            "Source": FAA_SOURCE,
        },
        {
            "EventDate": "2026-11-13",
            "EventTimeET": "20:15",
            "Event": "Air Traffic Resumes",
            "Description": (
                "FAA reported that the system had "
                "stabilized and air traffic resumed."
            ),
            "EvidenceType": "Reported incident",
            "Source": FAA_SOURCE,
        },
        {
            "EventDate": "2026-11-14",
            "EventTimeET": "",
            "Event": "Recovery Continues",
            "Description": (
                "DOT reported that airspace was open "
                "but delays and cancellations could continue."
            ),
            "EvidenceType": "Date-level official update",
            "Source": DOT_SOURCE,
        },
    ]

    return pd.DataFrame(events)


def build_timeline_figure(events):

    start = pd.Timestamp("2026-11-13 15:30")
    resume = pd.Timestamp("2026-11-13 20:15")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=[start, resume],
            y=[1, 1],
            mode="lines",
            line=dict(
                color="#F5B44B",
                width=9,
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[start, resume],
            y=[1, 1],
            mode="markers+text",
            marker=dict(
                size=21,
                color=["#F5B44B", "#27D9C0"],
                line=dict(
                    color="#FFFFFF",
                    width=2,
                ),
            ),
            text=[
                "Ground Stop Issued",
                "Air Traffic Resumes",
            ],
            textposition=[
                "top center",
                "top center",
            ],
            textfont=dict(
                size=15,
                color="#F1F7FF",
            ),
            customdata=[
                [
                    "Approximately 3:30 PM ET",
                    "FAA ground stop",
                ],
                [
                    "8:15 PM ET",
                    "FAA reports traffic resumed",
                ],
            ],
            hovertemplate=(
                "%{customdata[1]}"
                "<br>%{customdata[0]}"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )

    fig.add_annotation(
        x=start + (resume - start) / 2,
        y=0.76,
        text="Approximately 4 hours 45 minutes",
        showarrow=False,
        font=dict(
            size=15,
            color="#F5B44B",
        ),
    )

    fig.add_annotation(
        x=start + (resume - start) / 2,
        y=0.28,
        text=(
            "After reopening: aircraft, crew, "
            "passenger and financial recovery challenges"
        ),
        showarrow=False,
        font=dict(
            size=13,
            color="#A7F3E8",
        ),
        bgcolor="#13293E",
        bordercolor="#315367",
        borderpad=12,
    )

    fig.update_layout(
        title=dict(
            text=(
                "November 13, 2026 — "
                "U.S. Nationwide Ground Stop"
            ),
            font=dict(size=22),
        ),
        template="plotly_dark",
        paper_bgcolor="#07111F",
        plot_bgcolor="#07111F",
        font=dict(color="#F1F7FF"),
        height=440,
        margin=dict(
            l=45,
            r=45,
            t=100,
            b=70,
        ),
        xaxis=dict(
            title="November 13, 2026 (Eastern Time)",
            range=[
                pd.Timestamp("2026-11-13 14:45"),
                pd.Timestamp("2026-11-13 21:00"),
            ],
            tickformat="%I:%M %p",
            showgrid=True,
            gridcolor="#234057",
        ),
        yaxis=dict(
            range=[0, 1.65],
            visible=False,
        ),
    )

    return fig


def main():

    REPORTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES.mkdir(
        parents=True,
        exist_ok=True,
    )

    events = build_event_data()

    report_path = (
        REPORTS / "ground_stop_event_timeline.csv"
    )

    figure_path = (
        FIGURES / "ground_stop_recovery_timeline.html"
    )

    events.to_csv(
        report_path,
        index=False,
    )

    figure = build_timeline_figure(events)

    figure.write_html(
        figure_path,
        include_plotlyjs=True,
        full_html=True,
    )

    print()
    print("=" * 65)
    print("AERORESILIENCE AI — GROUND STOP TIMELINE")
    print("=" * 65)

    print()
    print("OFFICIAL INCIDENT MILESTONES")
    print(
        events[
            ["EventDate", "EventTimeET", "Event"]
        ].to_string(index=False)
    )

    print()
    print("INTERPRETATION")
    print(
        "The FAA reported a nationwide ground stop "
        "beginning at approximately 3:30 PM ET."
    )
    print(
        "Air traffic resumed at approximately "
        "8:15 PM ET."
    )
    print(
        "Airline network recovery continued "
        "after the airspace reopened."
    )

    print()
    print("IMPORTANT LIMITATION")
    print(
        "This timeline describes reported milestones."
    )
    print(
        "It does not reconstruct individual flight, "
        "crew, or passenger operations."
    )
    print(
        "AeroResilience AI recovery results remain "
        "separate hypothetical simulations."
    )

    print()
    print("FILES SAVED")
    print(report_path.relative_to(ROOT))
    print(figure_path.relative_to(ROOT))

    print()
    print(
        "AERORESILIENCE AI — "
        "GROUND STOP TIMELINE COMPLETE"
    )


if __name__ == "__main__":
    main()
