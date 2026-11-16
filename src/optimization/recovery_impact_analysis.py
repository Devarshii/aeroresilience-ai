
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

# =====================================================
# AERORESILIENCE AI
# STEP 22 — RECOVERY STRATEGY & FAIRNESS ANALYSIS
# =====================================================

INPUT_FILE = Path(
    "outputs/reports/optimized_recovery_flights.csv"
)

REPORT_DIR = Path("outputs/reports")
FIGURE_DIR = Path("outputs/figures")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def save_chart(fig, filename):
    fig.update_layout(
        template="plotly_dark",
        title_x=0.5,
        margin=dict(l=50, r=30, t=80, b=50),
    )
    fig.write_html(FIGURE_DIR / filename)
    print(f"Saved chart: {filename}")


def main():
    print("=" * 65)
    print("AERORESILIENCE AI — RECOVERY IMPACT ANALYSIS")
    print("=" * 65)

    df = pd.read_csv(INPUT_FILE)

    required = [
        "RecoveryFlightID",
        "Reporting_Airline",
        "PriorityWeight",
        "BaselineDelayMinutes",
        "OptimizedDelayMinutes",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df["RecoveryFlightID"].duplicated().any():
        raise ValueError("Duplicate flight identifiers.")

    if df[required].isna().any().any():
        raise ValueError("Missing values in required fields.")

    if not (df["PriorityWeight"] > 0).all():
        raise ValueError("Priority weights must be positive.")

    if not (
        df[["BaselineDelayMinutes", "OptimizedDelayMinutes"]]
        .ge(0).all().all()
    ):
        raise ValueError("Negative delay detected.")

    df["DelayChange"] = (
        df["OptimizedDelayMinutes"]
        - df["BaselineDelayMinutes"]
    )

    df["DelayReduction"] = -df["DelayChange"]

    df["WeightedBaseline"] = (
        df["BaselineDelayMinutes"] * df["PriorityWeight"]
    )

    df["WeightedOptimized"] = (
        df["OptimizedDelayMinutes"] * df["PriorityWeight"]
    )

    df["WeightedBenefit"] = (
        df["WeightedBaseline"] - df["WeightedOptimized"]
    )

    # ------------------------------------------------
    # 1. Overall performance
    # ------------------------------------------------

    baseline_score = df["WeightedBaseline"].sum()
    optimized_score = df["WeightedOptimized"].sum()

    improvement = (
        100 * (baseline_score - optimized_score)
        / baseline_score
        if baseline_score > 0 else 0
    )

    overall = pd.DataFrame([{
        "TotalFlights": len(df),
        "FlightsImproved": int((df["DelayChange"] < 0).sum()),
        "FlightsWorsened": int((df["DelayChange"] > 0).sum()),
        "FlightsUnchanged": int((df["DelayChange"] == 0).sum()),
        "BaselineDelayMinutes": df["BaselineDelayMinutes"].sum(),
        "OptimizedDelayMinutes": df["OptimizedDelayMinutes"].sum(),
        "WeightedDelayImprovementPercent": improvement,
        "BaselineMaximumDelay": df["BaselineDelayMinutes"].max(),
        "OptimizedMaximumDelay": df["OptimizedDelayMinutes"].max(),
        "BaselineP90Delay": df["BaselineDelayMinutes"].quantile(0.90),
        "OptimizedP90Delay": df["OptimizedDelayMinutes"].quantile(0.90),
        "FlightsWorsenedBy60PlusMinutes": int(
            (df["DelayChange"] >= 60).sum()
        ),
        "FlightsWorsenedBy120PlusMinutes": int(
            (df["DelayChange"] >= 120).sum()
        ),
    }])

    overall.to_csv(
        REPORT_DIR / "recovery_impact_summary.csv",
        index=False,
    )

    # ------------------------------------------------
    # 2. Airline-level fairness comparison
    # ------------------------------------------------

    airline = (
        df.groupby("Reporting_Airline")
        .agg(
            Flights=("RecoveryFlightID", "size"),
            AverageBaselineDelay=("BaselineDelayMinutes", "mean"),
            AverageOptimizedDelay=("OptimizedDelayMinutes", "mean"),
            MaximumOptimizedDelay=("OptimizedDelayMinutes", "max"),
            TotalDelayChange=("DelayChange", "sum"),
            TotalWeightedBenefit=("WeightedBenefit", "sum"),
            ImprovedFlights=("DelayChange", lambda s: int((s < 0).sum())),
            WorsenedFlights=("DelayChange", lambda s: int((s > 0).sum())),
        )
        .reset_index()
    )

    airline["AverageDelayChange"] = (
        airline["AverageOptimizedDelay"]
        - airline["AverageBaselineDelay"]
    )

    airline = airline.sort_values(
        "AverageDelayChange",
        ascending=False,
    )

    airline.to_csv(
        REPORT_DIR / "airline_recovery_fairness.csv",
        index=False,
    )

    # ------------------------------------------------
    # 3. Priority-group evaluation
    # ------------------------------------------------

    df["PriorityGroup"] = pd.qcut(
        df["PriorityWeight"].rank(method="first"),
        q=4,
        labels=[
            "Q1 - Lower",
            "Q2",
            "Q3",
            "Q4 - Higher",
        ],
    )

    priority = (
        df.groupby("PriorityGroup", observed=True)
        .agg(
            Flights=("RecoveryFlightID", "size"),
            AverageBaselineDelay=("BaselineDelayMinutes", "mean"),
            AverageOptimizedDelay=("OptimizedDelayMinutes", "mean"),
            TotalWeightedBenefit=("WeightedBenefit", "sum"),
        )
        .reset_index()
    )

    priority["AverageDelayChange"] = (
        priority["AverageOptimizedDelay"]
        - priority["AverageBaselineDelay"]
    )

    priority.to_csv(
        REPORT_DIR / "recovery_priority_impact.csv",
        index=False,
    )

    # ------------------------------------------------
    # 4. Visualizations
    # ------------------------------------------------

    comparison = pd.DataFrame({
        "Policy": ["FCFS Baseline", "Optimized"],
        "WeightedDelayScore": [
            baseline_score, optimized_score
        ],
    })

    fig = px.bar(
        comparison,
        x="Policy",
        y="WeightedDelayScore",
        title="Recovery Strategies — Weighted Delay Comparison",
        text_auto=".0f",
    )

    save_chart(fig, "recovery_weighted_comparison.html")

    fig = px.bar(
        airline,
        x="Reporting_Airline",
        y="AverageDelayChange",
        title="Change in Average Delay by Airline",
        labels={
            "Reporting_Airline": "Airline",
            "AverageDelayChange": "Average Delay Change (Minutes)",
        },
    )

    save_chart(fig, "recovery_airline_fairness.html")

    fig = px.bar(
        priority,
        x="PriorityGroup",
        y="AverageDelayChange",
        title="Recovery Impact by Priority Group",
        labels={
            "PriorityGroup": "Flight Priority Group",
            "AverageDelayChange": "Average Delay Change (Minutes)",
        },
    )

    save_chart(fig, "recovery_priority_impact.html")

    # ------------------------------------------------
    # 5. Terminal results
    # ------------------------------------------------

    print("\nOVERALL RECOVERY RESULTS")
    print(overall.to_string(index=False))

    print("\nAIRLINE FAIRNESS RESULTS")
    print(
        airline[
            [
                "Reporting_Airline",
                "Flights",
                "AverageDelayChange",
                "WorsenedFlights",
            ]
        ].to_string(index=False)
    )

    print("\nPRIORITY GROUP IMPACT")
    print(priority.to_string(index=False))

    print("\nIMPORTANT")
    print(
        "This analysis measures scheduling trade-offs, "
        "not verified financial or passenger savings."
    )
    print(
        "Priority groups are based on proxy network weights, "
        "not actual passenger importance."
    )
    print(
        "Airline fairness is descriptive; differences may "
        "reflect unequal flight counts and route mixes."
    )

    print("\nAERORESILIENCE AI — IMPACT ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
