
"""
AeroResilience AI — Ground Stop Case Study Page

Displays:
- November 13, 2026 incident context
- Verified event timeline from saved project data
- Four-policy hypothetical ORD recovery comparison
- Operational trade-offs and research limitations

Does not modify any existing model outputs.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.dashboard.ground_stop_timeline import build_timeline_figure
from src.dashboard.ground_stop_decision_scorecard import (
    make_figure,
    make_scorecard,
    load_results,
)


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "outputs" / "reports"

TIMELINE = REPORTS / "ground_stop_event_timeline.csv"

FAA_SOURCE = (
    "https://www.faa.gov/newsroom/"
    "nationwide-ground-stop-questions-and-answers"
)


def render_ground_stop_case_study():

    st.title("✈️ Ground Stop: From Crisis to Recovery")

    st.caption(
        "Real-world incident context | "
        "Historical BTS data | "
        "Hypothetical airline recovery experiments"
    )

    st.markdown(
        """
        ### The problem does not end when flights resume

        The November 13, 2026 nationwide FAA ground stop
        illustrates how a major airspace disruption can
        create challenges beyond the initial departure halt.

        Once departures are authorized to resume, airlines
        must recover aircraft rotations, crew assignments,
        passenger connections, and operating schedules.

        **AeroResilience AI explores how recovery strategies
        can be evaluated across these competing objectives.**
        """
    )

    st.info(
        "The November incident is the real-world motivation. "
        "The recovery results below come from a separate "
        "hypothetical Chicago O'Hare experiment, not from "
        "an actual reconstruction of November 13."
    )

    st.divider()

    st.subheader("1. U.S. Ground Stop — Incident Timeline")

    if TIMELINE.exists():
        events = pd.read_csv(
            TIMELINE,
            keep_default_na=False,
        )

        figure = build_timeline_figure(events)

        st.plotly_chart(
            figure,
            use_container_width=True,
        )

        with st.expander("View reported incident milestones"):
            st.dataframe(
                events,
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.warning(
            "Incident timeline data is unavailable. "
            "Run the ground_stop_timeline module "
            "to generate the report."
        )

    st.markdown(
        f"[FAA incident information and updates]({FAA_SOURCE})"
    )

    st.divider()

    st.subheader("2. Four-Policy Recovery Decision Scorecard")

    st.caption(
        "Experiment: hypothetical ORD ground stop | "
        "August 12, 2026 | "
        "20-minute minimum aircraft turnaround"
    )

    try:
        results = make_scorecard(load_results())

    except (FileNotFoundError, ValueError) as error:
        st.warning(
            f"Recovery comparison is unavailable: {error}"
        )
        results = None

    if results is not None:

        lowest_delay = results.loc[
            results["IncrementalDelayMinutes"].idxmin()
        ]

        fewest_delayed = results.loc[
            results["DelayedFlights"].idxmin()
        ]

        fewest_downstream = results.loc[
            results["DownstreamDelayedFlights"].idxmin()
        ]

        a, b, c = st.columns(3)

        a.metric(
            "Lowest network delay",
            str(lowest_delay["Policy"]),
            help="Lowest simulated incremental network delay, in minutes.",
        )

        b.metric(
            "Fewest delayed flights",
            str(fewest_delayed["Policy"]),
            help="Fewest flights delayed in the modeled recovery scenario.",
        )

        c.metric(
            "Fewest downstream delayed flights",
            str(fewest_downstream["Policy"]),
            help="Fewest downstream flights delayed in the modeled recovery scenario.",
        )

        st.plotly_chart(
            make_figure(results),
            use_container_width=True,
        )

        columns = [
            "Policy",
            "IncrementalDelayMinutes",
            "DelayedFlights",
            "DownstreamDelayedFlights",
            "IncrementalImprovementPercent",
        ]

        st.dataframe(
            results[columns],
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "⬇ Download recovery scorecard",
            data=results.to_csv(index=False),
            file_name="ground_stop_decision_scorecard.csv",
            mime="text/csv",
        )

    st.divider()

    st.subheader("3. The Airline Recovery Challenge")

    with st.container(border=True):
        st.markdown(
            """
            **Aircraft recovery**

            A grounded aircraft may be required for later
            flights. Rescheduling one departure can change
            downstream aircraft availability.

            **Crew recovery**

            Airlines must consider available crews and
            operational duty constraints when restoring
            flight schedules.

            **Passenger recovery**

            Missed onward connections may require rebooking.
            Minimizing flight delay alone may not maximize
            passenger recovery.

            **Financial recovery**

            Different schedules can produce different
            operational, rebooking, assistance, and crew costs.
            """
        )

    st.subheader("4. What Our Research Demonstrates")

    st.markdown(
        """
        - Recovery-policy performance depends on the metric
          used to evaluate it.
        - Aircraft-aware scheduling produced the lowest
          modeled incremental network delay in the
          existing ORD experiment.
        - The fairness-oriented policy produced fewer
          downstream delayed flights in that experiment.
        - Passenger and financial results introduce
          additional trade-offs between recovery objectives.
        """
    )

    st.divider()

    st.warning(
        "RESEARCH LIMITATIONS: Historical BTS flight data "
        "are real, but the ORD ground-stop scenario, recovery "
        "schedules, passenger data, crew resources, and "
        "financial estimates are simulated. Results do not "
        "represent the actual November 13, 2026 event. "
        "The platform is not a live airline dispatch system."
    )
