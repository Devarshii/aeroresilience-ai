
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

# =====================================================
# AERORESILIENCE AI
# STEP 17 — AIRCRAFT DELAY PROPAGATION ENGINE
# =====================================================

INPUT_FILE = Path(
    "data/processed/aircraft_rotations_august_2026.parquet"
)

REPORT_DIR = Path("outputs/reports")
FIGURE_DIR = Path("outputs/figures")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Experimental assumptions, not airline-specific standards.
MIN_TURN_MINUTES = 20
INCOMING_DELAY_SCENARIOS = [30, 60, 90, 120]


def simulate_single_connection(df, incoming_delay):
    """
    Calculate the possible delay inherited by the
    next flight when its incoming aircraft is late.

    Scheduled buffer = scheduled turnaround - minimum turn.

    Propagated delay = max(
        0, incoming delay - available buffer
    )

    Assumes no aircraft swaps or recovery actions.
    """

    buffer = np.maximum(
        df["TurnaroundMinutes"].to_numpy()
        - MIN_TURN_MINUTES,
        0,
    )

    return np.maximum(
        incoming_delay - buffer,
        0,
    )


def main():
    print("=" * 65)
    print("AERORESILIENCE AI — AIRCRAFT DELAY PROPAGATION")
    print("=" * 65)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing rotation dataset: {INPUT_FILE}"
        )

    df = pd.read_parquet(INPUT_FILE)

    required = [
        "FlightID",
        "PreviousFlightID",
        "Tail_Number",
        "Origin",
        "Dest",
        "TurnaroundMinutes",
        "ValidRotation",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[
        df["ValidRotation"].fillna(False)
    ].copy()

    df["TurnaroundMinutes"] = pd.to_numeric(
        df["TurnaroundMinutes"], errors="coerce"
    )

    df = df.dropna(
        subset=["TurnaroundMinutes"]
    ).copy()

    if df.empty:
        raise ValueError("No usable aircraft rotation links.")

    print(f"\nValid rotation links: {len(df):,}")
    print(f"Minimum turnaround assumption: {MIN_TURN_MINUTES} min")

    results = []

    for delay in INCOMING_DELAY_SCENARIOS:

        propagated = simulate_single_connection(
            df, delay
        )

        affected = propagated > 0

        results.append({
            "IncomingDelayMinutes": delay,
            "RotationLinks": len(df),
            "AffectedConnections": int(affected.sum()),
            "AffectedPercent": float(
                affected.mean() * 100
            ),
            "TotalPropagatedMinutes": float(
                propagated.sum()
            ),
            "AverageDelayAllLinks": float(
                propagated.mean()
            ),
            "AverageDelayAffectedLinks": (
                float(propagated[affected].mean())
                if affected.any() else 0.0
            ),
        })

        if delay == 90:
            df["Scenario90_PropagatedDelay"] = propagated
            df["Scenario90_Affected"] = affected.astype(int)

    summary = pd.DataFrame(results)

    summary.to_csv(
        REPORT_DIR / "delay_propagation_scenarios.csv",
        index=False,
    )

    # Save a detailed 90-minute scenario for later analysis.
    detail_columns = [
        "FlightID",
        "PreviousFlightID",
        "Tail_Number",
        "Origin",
        "Dest",
        "TurnaroundMinutes",
        "Scenario90_PropagatedDelay",
        "Scenario90_Affected",
    ]

    df[detail_columns].to_parquet(
        REPORT_DIR / "delay_propagation_90min.parquet",
        index=False,
    )

    # Visualize increasing delay exposure.
    fig = px.bar(
        summary,
        x="IncomingDelayMinutes",
        y="AffectedPercent",
        title="Aircraft Rotation Exposure Under Incoming Delay Scenarios",
        labels={
            "IncomingDelayMinutes": "Incoming Aircraft Delay (Minutes)",
            "AffectedPercent": "Connections With Inherited Delay (%)",
        },
        template="plotly_dark",
        text="AffectedPercent",
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
    )

    fig.update_layout(
        title_x=0.5,
        yaxis_range=[0, 105],
    )

    fig.write_html(
        FIGURE_DIR / "delay_propagation_scenarios.html"
    )

    print("\nDELAY PROPAGATION SCENARIOS")
    print(summary.to_string(index=False))

    print("\nIMPORTANT LIMITATIONS")
    print(
        "Each rotation link is independently stressed "
        "with the same hypothetical incoming delay."
    )
    print(
        "This is a one-connection sensitivity analysis, "
        "not a complete multi-flight cascading simulation."
    )
    print(
        "Affected connections are modeled exposures, "
        "not observed additional flight delays."
    )
    print(
        "Aircraft substitutions, crew constraints, airport "
        "capacity, and recovery actions are not included."
    )

    print("\nReports and visualization saved successfully.")
    print(
        "\nAERORESILIENCE AI — DELAY PROPAGATION COMPLETE"
    )


if __name__ == "__main__":
    main()
