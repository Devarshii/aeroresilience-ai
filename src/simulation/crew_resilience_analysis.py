
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ============================================================
# AERORESILIENCE AI
# MODULE 5 — RESILIENCE & WHAT-IF SIMULATOR
# STEP 50 — CREW RESILIENCE ANALYSIS
# ============================================================

REPORT_DIR = Path("outputs/reports")
FIGURE_DIR = Path("outputs/figures")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = (
    REPORT_DIR / "crew_recovery_stress_test.csv"
)

SUMMARY_FILE = (
    REPORT_DIR / "crew_resilience_summary.csv"
)

COMPARISON_FILE = (
    REPORT_DIR / "crew_resilience_comparison.csv"
)

VALIDATION_FILE = (
    REPORT_DIR / "crew_resilience_validation.csv"
)

CHART_FILE = (
    FIGURE_DIR / "crew_resilience_dashboard.html"
)

METHODOLOGY_FILE = (
    REPORT_DIR / "crew_resilience_methodology.json"
)

POLICIES = ["FCFS", "AircraftAware"]


# ============================================================
# 1. COLUMN UTILITIES
# ============================================================

def normalize_name(value):

    return re.sub(
        r"[^a-z0-9]",
        "",
        str(value).lower(),
    )


def find_column(data, candidates, required=True):

    lookup = {
        normalize_name(column): column
        for column in data.columns
    }

    for candidate in candidates:
        key = normalize_name(candidate)

        if key in lookup:
            return lookup[key]

    if required:
        raise ValueError(
            "Could not identify required column.\n"
            f"Expected one of: {candidates}\n"
            f"Available columns: {list(data.columns)}"
        )

    return None


def normalize_policy(value):

    name = normalize_name(value)

    if name in {
        "fcfs",
        "firstcomefirstserved",
    }:
        return "FCFS"

    if "aircraftaware" in name:
        return "AircraftAware"

    return str(value).strip()


# ============================================================
# 2. LOAD EXISTING CREW STRESS RESULTS
# ============================================================

def load_crew_results():

    print("=" * 65)
    print("AERORESILIENCE AI")
    print("STEP 50 — CREW RESILIENCE ANALYSIS")
    print("=" * 65)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing existing crew stress report: {INPUT_FILE}"
        )

    data = pd.read_csv(INPUT_FILE)

    if data.empty:
        raise ValueError(
            "Existing crew stress report is empty."
        )

    print("\nEXISTING CREW STRESS-TEST COLUMNS")

    for column in data.columns:
        print(" -", column)

    policy_column = find_column(
        data,
        ["Policy", "RecoveryPolicy"],
    )

    scenario_column = find_column(
        data,
        [
            "Scenario",
            "ScenarioName",
            "StressScenario",
            "StressTestScenario",
            "CrewScenario",
            "Condition",
        ],
    )

    total_column = find_column(
        data,
        [
            "TotalFlights",
            "Flights",
            "FlightsEvaluated",
            "RecoveryFlights",
            "TotalRecoveryFlights",
            "FlightCount",
        ],
    )

    covered_column = find_column(
        data,
        [
            "CoveredFlights",
            "FlightsCovered",
            "CrewCoveredFlights",
            "FlightsWithCrew",
            "CrewCovered",
            "CoveredFlightCount",
        ],
        required=False,
    )

    uncovered_column = find_column(
        data,
        [
            "UncoveredFlights",
            "FlightsUncovered",
            "FlightsWithoutCrew",
            "CrewUncoveredFlights",
        ],
        required=False,
    )

    coverage_column = find_column(
        data,
        [
            "CrewCoveragePercent",
            "CoveragePercent",
            "CrewCoverageRatePercent",
            "CrewCoveragePercentage",
            "CoverageRatePercent",
        ],
        required=False,
    )

    if (
        covered_column is None
        and uncovered_column is None
        and coverage_column is None
    ):
        raise ValueError(
            "No crew coverage measure was found. "
            "Review the displayed column names."
        )

    print("\nIDENTIFIED FIELDS")
    print("Policy:", policy_column)
    print("Scenario:", scenario_column)
    print("Total flights:", total_column)
    print("Covered flights:", covered_column)
    print("Uncovered flights:", uncovered_column)
    print("Coverage percentage:", coverage_column)

    records = []

    for row in data.to_dict("records"):

        policy = normalize_policy(
            row[policy_column]
        )

        if policy not in POLICIES:
            continue

        scenario = str(
            row[scenario_column]
        ).strip()

        total = int(
            pd.to_numeric(
                row[total_column],
                errors="raise",
            )
        )

        if total <= 0:
            raise ValueError(
                f"Invalid total flight count: {total}"
            )

        covered = None
        uncovered = None

        if covered_column is not None:

            covered = int(
                pd.to_numeric(
                    row[covered_column],
                    errors="raise",
                )
            )

        if uncovered_column is not None:

            uncovered = int(
                pd.to_numeric(
                    row[uncovered_column],
                    errors="raise",
                )
            )

        if covered is None and uncovered is not None:
            covered = total - uncovered

        if uncovered is None and covered is not None:
            uncovered = total - covered

        if covered is None:

            reported_percent = float(
                pd.to_numeric(
                    row[coverage_column],
                    errors="raise",
                )
            )

            covered_estimate = (
                total * reported_percent / 100
            )

            if not np.isclose(
                covered_estimate,
                round(covered_estimate),
                atol=0.02,
            ):
                raise ValueError(
                    "Coverage percentage cannot be "
                    "reconciled to an integer flight count."
                )

            covered = int(
                round(covered_estimate)
            )

            uncovered = total - covered

        if (
            covered < 0
            or uncovered < 0
            or covered + uncovered != total
        ):
            raise ValueError(
                f"Crew coverage does not reconcile: "
                f"{scenario}, {policy}"
            )

        coverage = (
            100 * covered / total
        )

        if coverage_column is not None:

            reported = float(
                pd.to_numeric(
                    row[coverage_column],
                    errors="raise",
                )
            )

            if not np.isclose(
                coverage,
                reported,
                atol=0.02,
            ):
                raise ValueError(
                    f"Reported crew coverage mismatch: "
                    f"{scenario}, {policy}"
                )

        records.append({
            "Scenario": scenario,
            "Policy": policy,
            "TotalFlights": total,
            "CoveredFlights": covered,
            "UncoveredFlights": uncovered,
            "CrewCoveragePercent": coverage,
        })

    summary = pd.DataFrame(records)

    if summary.empty:
        raise ValueError(
            "No matching crew recovery policies found."
        )

    if summary.duplicated(
        ["Scenario", "Policy"]
    ).any():
        raise ValueError(
            "Duplicate crew scenario-policy records."
        )

    return summary


# ============================================================
# 3. COMPARE RECOVERY POLICIES
# ============================================================

def compare_policies(summary):

    print("\nComparing crew resilience outcomes...")

    records = []

    for scenario, group in summary.groupby(
        "Scenario",
        sort=True,
    ):

        if set(group["Policy"]) != set(POLICIES):
            raise ValueError(
                f"Missing recovery policy in: {scenario}"
            )

        indexed = group.set_index("Policy")

        fcfs = indexed.loc["FCFS"]
        aircraft = indexed.loc["AircraftAware"]

        coverage_difference = (
            aircraft["CrewCoveragePercent"]
            - fcfs["CrewCoveragePercent"]
        )

        if coverage_difference > 0.000001:
            preferred = "AircraftAware"
        elif coverage_difference < -0.000001:
            preferred = "FCFS"
        else:
            preferred = "Tie"

        records.append({
            "Scenario": scenario,
            "TotalFlights": int(
                fcfs["TotalFlights"]
            ),
            "FCFSCoveredFlights": int(
                fcfs["CoveredFlights"]
            ),
            "AircraftAwareCoveredFlights": int(
                aircraft["CoveredFlights"]
            ),
            "FCFSUncoveredFlights": int(
                fcfs["UncoveredFlights"]
            ),
            "AircraftAwareUncoveredFlights": int(
                aircraft["UncoveredFlights"]
            ),
            "FCFSCoveragePercent": float(
                fcfs["CrewCoveragePercent"]
            ),
            "AircraftAwareCoveragePercent": float(
                aircraft["CrewCoveragePercent"]
            ),
            "AircraftAwareCoverageDifferencePoints": (
                coverage_difference
            ),
            "BetterCrewCoveragePolicy": preferred,
        })

    return pd.DataFrame(records)


# ============================================================
# 4. VALIDATE CREW RESILIENCE
# ============================================================

def validate_results(summary, comparison):

    checks = []

    def record_check(name, condition):

        passed = bool(condition)

        checks.append({
            "Check": name,
            "Status": "PASS" if passed else "FAIL",
        })

        print(
            f"[{'PASS' if passed else 'FAIL'}] "
            f"{name}"
        )

    print("\nCREW RESILIENCE VALIDATION")

    record_check(
        "Unique scenario-policy records",
        not summary.duplicated(
            ["Scenario", "Policy"]
        ).any(),
    )

    record_check(
        "Flight counts reconcile",
        (
            summary["CoveredFlights"]
            + summary["UncoveredFlights"]
            == summary["TotalFlights"]
        ).all(),
    )

    record_check(
        "Coverage percentages within 0–100",
        summary[
            "CrewCoveragePercent"
        ].between(0, 100).all(),
    )

    record_check(
        "All scenarios contain both policies",
        comparison["Scenario"].nunique()
        == summary["Scenario"].nunique(),
    )

    record_check(
        "Same total flights per policy in each scenario",
        all(
            group["TotalFlights"].nunique() == 1
            for _, group in summary.groupby("Scenario")
        ),
    )

    validation = pd.DataFrame(checks)

    validation.to_csv(
        VALIDATION_FILE,
        index=False,
    )

    if (
        validation["Status"] == "FAIL"
    ).any():
        raise RuntimeError(
            "Crew resilience validation failed."
        )

    return validation


# ============================================================
# 5. CREATE INTERACTIVE DASHBOARD
# ============================================================

def create_dashboard(comparison):

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=comparison["Scenario"],
            x=comparison["FCFSCoveragePercent"],
            name="FCFS",
            orientation="h",
            marker_color="#64748b",
            text=comparison[
                "FCFSCoveragePercent"
            ].round(2),
            textposition="inside",
            hovertemplate=(
                "%{y}<br>FCFS coverage: %{x:.2f}%"
                "<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Bar(
            y=comparison["Scenario"],
            x=comparison[
                "AircraftAwareCoveragePercent"
            ],
            name="Aircraft-aware",
            orientation="h",
            marker_color="#0d9488",
            text=comparison[
                "AircraftAwareCoveragePercent"
            ].round(2),
            textposition="inside",
            hovertemplate=(
                "%{y}<br>Aircraft-aware coverage: "
                "%{x:.2f}%<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=(
            "AeroResilience AI — "
            "Crew Recovery Resilience"
        ),
        xaxis_title="Crew Coverage (%)",
        yaxis_title="Crew Stress Scenario",
        xaxis=dict(
            range=[0, 105],
        ),
        template="plotly_white",
        barmode="group",
        height=650,
        margin=dict(
            l=250,
            r=50,
            t=100,
            b=70,
        ),
    )

    fig.write_html(
        str(CHART_FILE),
        include_plotlyjs=True,
    )


# ============================================================
# 6. SAVE METHODOLOGY
# ============================================================

def save_methodology(summary):

    methodology = {
        "Project": "AeroResilience AI",
        "Module": 5,
        "Step": 50,
        "Source": str(INPUT_FILE),
        "NumberOfPolicyScenarioRecords": (
            len(summary)
        ),
        "NumberOfCrewScenarios": int(
            summary["Scenario"].nunique()
        ),
        "Methods": [
            "Reuse existing simulated crew stress tests",
            "Compare FCFS and aircraft-aware crew coverage",
            "Validate crew coverage and flight counts",
            "Generate an interactive resilience dashboard",
        ],
        "Limitations": [
            "Crew identities, reserves, and duty "
            "conditions are synthetic.",
            "This analysis reuses previous crew "
            "simulation results.",
            "No new crew assignment optimization "
            "is performed in this step.",
            "Crew coverage is based on simplified "
            "duty and availability assumptions.",
            "Real crew qualification, contractual "
            "rules and full legal restrictions "
            "are not established.",
            "Results do not represent observed "
            "airline staffing outcomes.",
        ],
    }

    with open(
        METHODOLOGY_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            methodology,
            file,
            indent=4,
        )


# ============================================================
# 7. MAIN PROGRAM
# ============================================================

def main():

    summary = load_crew_results()

    comparison = compare_policies(summary)

    validation = validate_results(
        summary,
        comparison,
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    comparison.to_csv(
        COMPARISON_FILE,
        index=False,
    )

    create_dashboard(comparison)

    save_methodology(summary)

    print("\n" + "=" * 65)
    print("CREW RESILIENCE COMPARISON")
    print("=" * 65)

    columns = [
        "Scenario",
        "FCFSCoveredFlights",
        "AircraftAwareCoveredFlights",
        "FCFSUncoveredFlights",
        "AircraftAwareUncoveredFlights",
        "FCFSCoveragePercent",
        "AircraftAwareCoveragePercent",
    ]

    print(
        comparison[columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print("\nVALIDATION")
    print(
        f"Checks passed: {len(validation)}"
    )

    print("\nIMPORTANT LIMITATIONS")
    print(
        "Crew stress-test outcomes are reused "
        "from the existing synthetic crew model."
    )
    print(
        "Crew feasibility is not recalculated "
        "in this reporting step."
    )
    print(
        "This is not a real airline crew legality "
        "or rostering system."
    )

    print("\nFILES SAVED")

    for path in [
        SUMMARY_FILE,
        COMPARISON_FILE,
        VALIDATION_FILE,
        CHART_FILE,
        METHODOLOGY_FILE,
    ]:
        print(path)

    print(
        "\nAERORESILIENCE AI — "
        "CREW RESILIENCE ANALYSIS COMPLETE"
    )


if __name__ == "__main__":
    main()
