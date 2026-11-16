
from pathlib import Path

import pandas as pd
import plotly.express as px

# =====================================================
# AERORESILIENCE AI
# STEP 19 — CASCADE SIMULATION VALIDATION
# =====================================================

REPORT_DIR = Path("outputs/reports")
FIGURE_DIR = Path("outputs/figures")

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

DETAIL_FILE = REPORT_DIR / "cascade_flight_details.csv"
SUMMARY_FILE = REPORT_DIR / "cascade_simulation_summary.csv"


def main():
    print("=" * 65)
    print("AERORESILIENCE AI — CASCADE VALIDATION")
    print("=" * 65)

    details = pd.read_csv(DETAIL_FILE)
    summary = pd.read_csv(SUMMARY_FILE)

    required = [
        "Airline",
        "TailNumber",
        "Origin",
        "Dest",
        "ScheduledDepUTC",
        "SimulatedDelayMinutes",
        "DirectGroundStop",
        "PropagationHop",
    ]

    missing = [c for c in required if c not in details.columns]

    if missing:
        raise ValueError(f"Missing fields: {missing}")

    details["DirectGroundStop"] = (
        details["DirectGroundStop"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    details["ScheduledDepUTC"] = pd.to_datetime(
        details["ScheduledDepUTC"],
        errors="coerce"
    )

    print("\nRUNNING INTEGRITY CHECKS...")

    checks = {
        "No missing departure times":
            details["ScheduledDepUTC"].notna().all(),

        "All modeled delays positive":
            details["SimulatedDelayMinutes"].gt(0).all(),

        "Propagation hops nonnegative":
            details["PropagationHop"].ge(0).all(),

        "Direct delays use hop zero":
            details.loc[
                details["DirectGroundStop"],
                "PropagationHop"
            ].eq(0).all(),

        "Downstream delays use positive hops":
            details.loc[
                ~details["DirectGroundStop"],
                "PropagationHop"
            ].gt(0).all(),

        "No duplicate flight identifiers":
            ~details.duplicated(
                subset=[
                    "Airline",
                    "TailNumber",
                    "ScheduledDepUTC",
                    "Origin",
                    "Dest",
                ]
            ).any(),
    }

    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"{status}: {name}")

    direct = details[details["DirectGroundStop"]]
    downstream = details[~details["DirectGroundStop"]]

    calculated = {
        "DirectDelayedFlights": len(direct),
        "DownstreamDelayedFlights": len(downstream),
        "TotalDelayedFlights": len(details),
        "TotalModeledDelayMinutes":
            details["SimulatedDelayMinutes"].sum(),
        "MaxPropagationHop":
            details["PropagationHop"].max(),
    }

    for field, value in calculated.items():
        reported = float(summary.iloc[0][field])
        passed = abs(float(value) - reported) < 0.01

        checks[f"Summary matches: {field}"] = passed

        print(
            f"{'PASS' if passed else 'FAIL'}: "
            f"{field} = {value:,.2f}"
        )

    # ---------------------------------------------
    # Airport-level impact
    # ---------------------------------------------

    airport_impact = (
        downstream.groupby("Origin")
        .agg(
            AffectedFlights=("Origin", "size"),
            DelayMinutes=("SimulatedDelayMinutes", "sum"),
        )
        .reset_index()
        .sort_values("DelayMinutes", ascending=False)
    )

    airport_impact.to_csv(
        REPORT_DIR / "validated_cascade_airports.csv",
        index=False
    )

    # ---------------------------------------------
    # Delay propagation by hop
    # ---------------------------------------------

    hop_summary = (
        details.groupby("PropagationHop")
        .agg(
            Flights=("PropagationHop", "size"),
            DelayMinutes=("SimulatedDelayMinutes", "sum"),
        )
        .reset_index()
        .sort_values("PropagationHop")
    )

    hop_summary.to_csv(
        REPORT_DIR / "cascade_hop_summary.csv",
        index=False
    )

    fig = px.bar(
        hop_summary,
        x="PropagationHop",
        y="DelayMinutes",
        title="Simulated Delay Propagation by Aircraft Rotation Hop",
        labels={
            "PropagationHop": "Propagation Hop",
            "DelayMinutes": "Simulated Delay Minutes",
        },
        template="plotly_dark",
    )

    fig.update_layout(title_x=0.5)

    fig.write_html(
        FIGURE_DIR / "cascade_propagation_hops.html"
    )

    # ---------------------------------------------
    # Save validation report
    # ---------------------------------------------

    validation_report = pd.DataFrame([
        {"Check": name, "Passed": bool(passed)}
        for name, passed in checks.items()
    ])

    validation_report.to_csv(
        REPORT_DIR / "cascade_validation_report.csv",
        index=False
    )

    print("\nPROPAGATION BY HOP")
    print(hop_summary.to_string(index=False))

    print("\nTOP DOWNSTREAM AIRPORTS")
    print(airport_impact.head(10).to_string(index=False))

    all_passed = all(checks.values())

    print("\nVALIDATION RESULT")
    print("PASS" if all_passed else "FAIL — REVIEW REQUIRED")

    if not all_passed:
        raise RuntimeError(
            "One or more simulation integrity checks failed."
        )

    print("\nIMPORTANT:")
    print(
        "Passing these checks confirms internal consistency, "
        "not real-world predictive accuracy."
    )

    print(
        "\nAERORESILIENCE AI — CASCADE VALIDATION COMPLETE"
    )


if __name__ == "__main__":
    main()
