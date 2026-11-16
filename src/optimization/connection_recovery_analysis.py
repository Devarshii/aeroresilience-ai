
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

# =====================================================
# AERORESILIENCE AI
# STEP 25 — AIRCRAFT CONNECTION RECOVERY ANALYSIS
# =====================================================

REPORT_DIR = Path("outputs/reports")
FIGURE_DIR = Path("outputs/figures")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = (
    REPORT_DIR / "aircraft_feasibility_details.csv"
)

MIN_TURN_MINUTES = 20

POLICIES = {
    "FCFS": "FCFSAvailableTurn",
    "Unconstrained": "UnconstrainedAvailableTurn",
    "Fair": "FairAvailableTurn",
}


def main():

    print("=" * 65)
    print("AERORESILIENCE AI — CONNECTION RECOVERY ANALYSIS")
    print("=" * 65)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required = [
        "RecoveryFlightID",
        "Reporting_Airline",
        "AuditableConnection",
        *POLICIES.values(),
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df["AuditableConnection"] = (
        df["AuditableConnection"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    auditable = df[
        df["AuditableConnection"]
    ].copy()

    if auditable.empty:
        raise ValueError(
            "No auditable aircraft connections found."
        )

    print(f"\nRecovered flights: {len(df):,}")
    print(f"Auditable connections: {len(auditable):,}")
    print(
        f"Minimum turnaround assumption: "
        f"{MIN_TURN_MINUTES} minutes"
    )

    results = []

    for policy, column in POLICIES.items():

        available = pd.to_numeric(
            auditable[column],
            errors="coerce"
        )

        if available.isna().any():
            raise ValueError(
                f"Missing turnaround values for {policy}"
            )

        # Required delay of the next scheduled flight
        # to restore the assumed minimum turnaround.
        required_shift = np.maximum(
            MIN_TURN_MINUTES - available,
            0
        )

        auditable[
            f"{policy}RequiredNextFlightDelay"
        ] = required_shift

        affected = required_shift > 0

        results.append({
            "Policy": policy,
            "AuditableConnections": len(auditable),
            "ConnectionsRequiringAdjustment": int(
                affected.sum()
            ),
            "ConflictRatePercent": float(
                100 * affected.mean()
            ),
            "TotalRequiredNextFlightDelayMinutes": float(
                required_shift.sum()
            ),
            "AverageRequiredDelayAllConnections": float(
                required_shift.mean()
            ),
            "AverageRequiredDelayAffected": (
                float(required_shift[affected].mean())
                if affected.any() else 0
            ),
            "MaximumRequiredNextFlightDelay": float(
                required_shift.max()
            ),
        })

    summary = pd.DataFrame(results)

    # ---------------------------------------------
    # Compare all three recovery strategies
    # ---------------------------------------------

    print("\nNEXT-FLIGHT RECOVERY REQUIREMENTS")
    print(summary.to_string(index=False))

    summary.to_csv(
        REPORT_DIR / "connection_recovery_comparison.csv",
        index=False
    )

    auditable.to_csv(
        REPORT_DIR / "connection_recovery_details.csv",
        index=False
    )

    # ---------------------------------------------
    # Visualize required next-flight adjustments
    # ---------------------------------------------

    fig = px.bar(
        summary,
        x="Policy",
        y="TotalRequiredNextFlightDelayMinutes",
        title=(
            "Required Next-Flight Delay "
            "by Recovery Strategy"
        ),
        text_auto=".0f",
        labels={
            "Policy": "Recovery Strategy",
            "TotalRequiredNextFlightDelayMinutes":
                "Required Next-Flight Delay (Minutes)",
        },
        template="plotly_dark",
    )

    fig.update_layout(
        title_x=0.5
    )

    fig.write_html(
        FIGURE_DIR / "connection_recovery_comparison.html"
    )

    # ---------------------------------------------
    # Inspect the most severe connection problems
    # ---------------------------------------------

    worst = auditable.sort_values(
        "FairRequiredNextFlightDelay",
        ascending=False
    ).head(10)

    print("\nTOP 10 NEXT-FLIGHT ADJUSTMENTS — FAIR POLICY")

    display = [
        "RecoveryFlightID",
        "Reporting_Airline",
        "Origin",
        "Dest",
        "FairRequiredNextFlightDelay",
    ]

    print(
        worst[display].to_string(index=False)
    )

    print("\nIMPORTANT LIMITATIONS")
    print(
        "Required delay is calculated independently "
        "for each first downstream aircraft connection."
    )
    print(
        "The calculation assumes unchanged aircraft "
        "assignments, flight durations, and minimum turn."
    )
    print(
        "Next-flight departure adjustments are not "
        "scheduled or propagated further in this step."
    )
    print(
        "Additional runway slots, crews, gates, "
        "and passenger consequences are not modeled."
    )
    print(
        "The summed minutes are diagnostic requirements, "
        "not a feasible network-wide recovery schedule."
    )

    print("\nReports and visualization saved successfully.")

    print(
        "\nAERORESILIENCE AI — "
        "CONNECTION RECOVERY ANALYSIS COMPLETE"
    )


if __name__ == "__main__":
    main()
