
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =====================================================
# AERORESILIENCE AI
# MODULE 4 — FINANCIAL IMPACT & RECOVERY ANALYTICS
# STEP 44 — EXECUTIVE FINANCIAL DECISION REPORT
# =====================================================

REPORT_DIR = Path("outputs/reports")
FIGURE_DIR = Path("outputs/figures")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_FILE = (
    REPORT_DIR / "recovery_financial_cost_baseline.csv"
)

SENSITIVITY_FILE = (
    REPORT_DIR / "financial_sensitivity_scenarios.csv"
)

BREAK_EVEN_FILE = (
    REPORT_DIR / "financial_break_even_analysis.csv"
)

COST_COMPONENTS = {
    "OperationalDelayCostUSD": "Aircraft network delay",
    "PassengerRebookingCostUSD": "Passenger rebooking",
    "PassengerAssistanceCostUSD": "Passenger assistance",
    "ReserveCrewAssignmentCostUSD": "Reserve crew assignment",
}


# =====================================================
# 1. LOAD AND VERIFY REPORTS
# =====================================================

def load_reports():

    required_files = [
        BASELINE_FILE,
        SENSITIVITY_FILE,
        BREAK_EVEN_FILE,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required report not found: {path}"
            )

    baseline = pd.read_csv(BASELINE_FILE)
    sensitivity = pd.read_csv(SENSITIVITY_FILE)
    break_even = pd.read_csv(BREAK_EVEN_FILE)

    if (
        len(baseline) != 2
        or set(baseline["Policy"])
        != {"FCFS", "AircraftAware"}
    ):
        raise ValueError(
            "Expected one financial baseline row "
            "for each recovery policy."
        )

    required_costs = [
        "TotalModeledRecoveryCostUSD",
        *COST_COMPONENTS.keys(),
    ]

    for column in required_costs:
        if column not in baseline.columns:
            raise ValueError(
                f"Missing financial column: {column}"
            )

        baseline[column] = pd.to_numeric(
            baseline[column],
            errors="raise",
        )

    if baseline[required_costs].isna().any().any():
        raise ValueError(
            "Missing baseline financial values."
        )

    components_total = baseline[
        list(COST_COMPONENTS)
    ].sum(axis=1)

    if not (
        (
            components_total
            - baseline["TotalModeledRecoveryCostUSD"]
        ).abs() <= 0.01
    ).all():
        raise ValueError(
            "Baseline cost components do not reconcile."
        )

    if sensitivity.empty or break_even.empty:
        raise ValueError(
            "Financial analysis reports are empty."
        )

    return (
        baseline.set_index("Policy"),
        sensitivity,
        break_even,
    )


# =====================================================
# 2. PREPARE FINANCIAL COST DRIVERS
# =====================================================

def build_cost_drivers(baseline):

    rows = []

    for column, label in COST_COMPONENTS.items():

        fcfs = float(
            baseline.loc["FCFS", column]
        )

        aircraft_aware = float(
            baseline.loc["AircraftAware", column]
        )

        rows.append({
            "CostCategory": label,
            "FCFSCostUSD": fcfs,
            "AircraftAwareCostUSD": aircraft_aware,
            "CostReductionUSD": fcfs - aircraft_aware,
        })

    return pd.DataFrame(rows)


# =====================================================
# 3. BUILD EXECUTIVE CHART
# =====================================================

def build_chart(baseline, sensitivity, drivers):

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Total Modeled Recovery Cost",
            "Cost Components by Recovery Policy",
            "Lower-Cost Strategy Across Scenarios",
            "Cost Reduction vs Delay Minute Cost",
        ],
        vertical_spacing=0.17,
        horizontal_spacing=0.15,
    )

    policies = ["FCFS", "AircraftAware"]

    policy_colors = {
        "FCFS": "#64748b",
        "AircraftAware": "#0d9488",
    }

    # Chart 1 — Total recovery cost.
    for policy in policies:

        fig.add_trace(
            go.Bar(
                x=[policy],
                y=[
                    float(
                        baseline.loc[
                            policy,
                            "TotalModeledRecoveryCostUSD",
                        ]
                    )
                ],
                name=policy,
                marker_color=policy_colors[policy],
                legendgroup=policy,
                showlegend=True,
                hovertemplate=(
                    "%{x}: $%{y:,.0f}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    # Chart 2 — Financial cost breakdown.
    for column, label in COST_COMPONENTS.items():

        fig.add_trace(
            go.Bar(
                x=policies,
                y=[
                    float(baseline.loc[p, column])
                    for p in policies
                ],
                name=label,
                showlegend=True,
                hovertemplate=(
                    "%{x}: $%{y:,.0f}<extra>"
                    + label
                    + "</extra>"
                ),
            ),
            row=1,
            col=2,
        )

    # Chart 3 — Scenario results.
    counts = (
        sensitivity["LowerCostPolicy"]
        .value_counts()
    )

    categories = [
        "AircraftAware",
        "FCFS",
        "Tie",
    ]

    fig.add_trace(
        go.Bar(
            x=categories,
            y=[
                int(counts.get(category, 0))
                for category in categories
            ],
            name="Scenario count",
            marker_color=[
                "#0d9488",
                "#64748b",
                "#eab308",
            ],
            showlegend=False,
            hovertemplate=(
                "%{x}: %{y} scenarios<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )

    # Chart 4 — Financial sensitivity.
    grouped = (
        sensitivity.groupby(
            "DelayCostPerMinuteUSD",
            as_index=False,
        )
        .agg(
            MinimumReduction=(
                "NetCostReductionUSD",
                "min",
            ),
            AverageReduction=(
                "NetCostReductionUSD",
                "mean",
            ),
            MaximumReduction=(
                "NetCostReductionUSD",
                "max",
            ),
        )
        .sort_values("DelayCostPerMinuteUSD")
    )

    fig.add_trace(
        go.Scatter(
            x=grouped["DelayCostPerMinuteUSD"],
            y=grouped["MaximumReduction"],
            mode="lines",
            name="Maximum reduction",
            line=dict(
                color="#99f6e4",
                width=1,
            ),
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=grouped["DelayCostPerMinuteUSD"],
            y=grouped["MinimumReduction"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(13,148,136,0.14)",
            name="Scenario cost range",
            line=dict(
                color="#99f6e4",
                width=1,
            ),
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=grouped["DelayCostPerMinuteUSD"],
            y=grouped["AverageReduction"],
            mode="lines+markers",
            name="Average reduction",
            line=dict(
                color="#0d9488",
                width=3,
            ),
            showlegend=False,
            hovertemplate=(
                "Delay rate: $%{x:,.0f}/min"
                "<br>Average reduction: $%{y:,.0f}"
                "<extra></extra>"
            ),
        ),
        row=2,
        col=2,
    )

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="#ef4444",
        row=2,
        col=2,
    )

    fig.update_layout(
        title=(
            "AeroResilience AI | "
            "Financial Recovery Decision Report"
        ),
        template="plotly_white",
        height=950,
        width=1400,
        barmode="stack",
        margin=dict(
            l=70,
            r=50,
            t=120,
            b=80,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.16,
            xanchor="center",
            x=0.5,
        ),
    )

    fig.update_yaxes(
        title_text="USD",
        tickprefix="$",
        row=1,
        col=1,
    )

    fig.update_yaxes(
        title_text="USD",
        tickprefix="$",
        row=1,
        col=2,
    )

    fig.update_yaxes(
        title_text="Scenarios",
        row=2,
        col=1,
    )

    fig.update_xaxes(
        title_text="Delay cost per minute (USD)",
        row=2,
        col=2,
    )

    fig.update_yaxes(
        title_text="FCFS cost minus aircraft-aware cost (USD)",
        row=2,
        col=2,
    )

    output_file = (
        FIGURE_DIR / "financial_decision_dashboard.html"
    )

    fig.write_html(
        str(output_file),
        include_plotlyjs=True,
    )

    return output_file


# =====================================================
# 4. CREATE EXECUTIVE MARKDOWN REPORT
# =====================================================

def build_executive_report(
    baseline,
    sensitivity,
    break_even,
    drivers,
):

    fcfs = float(
        baseline.loc[
            "FCFS",
            "TotalModeledRecoveryCostUSD",
        ]
    )

    aircraft = float(
        baseline.loc[
            "AircraftAware",
            "TotalModeledRecoveryCostUSD",
        ]
    )

    difference = fcfs - aircraft

    percent = (
        100 * difference / fcfs
        if fcfs > 0 else 0.0
    )

    if difference > 0:
        preferred = "Aircraft-aware"
    elif difference < 0:
        preferred = "FCFS"
    else:
        preferred = "Tie"

    counts = (
        sensitivity["LowerCostPolicy"]
        .value_counts()
    )

    threshold = float(
        break_even.iloc[0][
            "BreakEvenDelayCostPerMinuteUSD"
        ]
    )

    lines = [
        "# AeroResilience AI — Financial Decision Report",
        "",
        "## Executive Summary",
        "",
        "This report evaluates hypothetical financial outcomes "
        "for two simulated airline ground-stop recovery strategies.",
        "",
        f"- FCFS modeled cost: **${fcfs:,.2f}**",
        f"- Aircraft-aware modeled cost: **${aircraft:,.2f}**",
        f"- FCFS minus aircraft-aware: **${difference:,.2f}**",
        f"- Relative difference: **{percent:.2f}%**",
        f"- Lower baseline modeled cost: **{preferred}**",
        "",
        "## Financial Sensitivity",
        "",
        f"- Scenarios tested: **{len(sensitivity):,}**",
        f"- Aircraft-aware lower-cost scenarios: "
        f"**{counts.get('AircraftAware', 0)}**",
        f"- FCFS lower-cost scenarios: "
        f"**{counts.get('FCFS', 0)}**",
        f"- Ties: **{counts.get('Tie', 0)}**",
        f"- Break-even delay cost per minute, holding other "
        f"baseline assumptions fixed: **${threshold:,.4f}**",
        "",
        "## Cost Drivers",
        "",
        "| Category | FCFS (USD) | Aircraft-aware (USD) "
        "| FCFS minus aircraft-aware (USD) |",
        "|---|---:|---:|---:|",
    ]

    for row in drivers.itertuples(index=False):
        lines.append(
            f"| {row.CostCategory} "
            f"| ${row.FCFSCostUSD:,.2f} "
            f"| ${row.AircraftAwareCostUSD:,.2f} "
            f"| ${row.CostReductionUSD:,.2f} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "Financial preference depends on the assumed unit "
        "costs and the particular simulated recovery scenario.",
        "",
        "A lower modeled cost does not establish a real-world "
        "operational advantage or actual financial savings.",
        "",
        "## Limitations",
        "",
        "- All monetary assumptions are illustrative.",
        "- Passenger demand and seat capacity are synthetic.",
        "- Flight delays originate from a simulated ground stop.",
        "- Airline fare rules, real crew legality, and real "
        "seat inventory are not modeled.",
        "- The scenario analysis varies financial inputs "
        "without rerunning recovery optimization.",
        "- This analysis is not a real airline financial forecast.",
    ])

    report_file = (
        REPORT_DIR / "financial_executive_decision_report.md"
    )

    report_file.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    return report_file


# =====================================================
# 5. MAIN
# =====================================================

def main():

    print("=" * 65)
    print("AERORESILIENCE AI")
    print("STEP 44 — FINANCIAL DECISION REPORT")
    print("=" * 65)

    baseline, sensitivity, break_even = load_reports()

    drivers = build_cost_drivers(baseline)

    driver_file = (
        REPORT_DIR / "financial_cost_drivers.csv"
    )

    drivers.to_csv(
        driver_file,
        index=False,
    )

    chart_file = build_chart(
        baseline,
        sensitivity,
        drivers,
    )

    report_file = build_executive_report(
        baseline,
        sensitivity,
        break_even,
        drivers,
    )

    print("\nFINANCIAL COST DRIVERS")
    print(
        drivers.to_string(
            index=False,
            float_format=lambda x: f"{x:,.2f}",
        )
    )

    print("\nSENSITIVITY RESULTS")
    print(
        sensitivity["LowerCostPolicy"]
        .value_counts()
        .to_string()
    )

    print("\nFILES SAVED")
    print(driver_file)
    print(report_file)
    print(chart_file)

    print(
        "\nAERORESILIENCE AI — "
        "FINANCIAL DECISION REPORT COMPLETE"
    )


if __name__ == "__main__":
    main()
