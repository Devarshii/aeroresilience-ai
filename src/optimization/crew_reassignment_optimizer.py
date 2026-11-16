
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

# =====================================================
# AERORESILIENCE AI
# STEP 35 — CREW REASSIGNMENT OPTIMIZATION
# =====================================================

CREW_FILE = Path(
    "data/processed/synthetic_crew_roster.parquet"
)

FLIGHT_FILE = Path(
    "outputs/reports/recovery_baseline_flights.csv"
)

REPORT_DIR = Path("outputs/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

DUMMY_COST = 100000.0
INFEASIBLE_COST = 10000000.0

# Synthetic crew units are assigned to at most
# one recovery flight in this simplified model.


def load_data():
    print("\nLoading crew and recovery flight data...")

    crews = pd.read_parquet(CREW_FILE)
    flights = pd.read_csv(FLIGHT_FILE)

    crew_columns = [
        "CrewID",
        "CrewType",
        "Reporting_Airline",
        "AssignedFlightID",
        "AvailableMinute",
        "DutyUsedMinutes",
        "DutyLimitMinutes",
        "RemainingDutyMinutes",
        "ProjectedDutyMinutes",
    ]

    flight_columns = [
        "RecoveryFlightID",
        "Reporting_Airline",
        "RecoveryDepartureMinute",
    ]

    for name, frame, required in [
        ("crew", crews, crew_columns),
        ("flight", flights, flight_columns),
    ]:
        missing = [
            col for col in required
            if col not in frame.columns
        ]

        if missing:
            raise ValueError(
                f"Missing {name} columns: {missing}"
            )

    if crews["CrewID"].duplicated().any():
        raise ValueError("Duplicate crew IDs.")

    if flights["RecoveryFlightID"].duplicated().any():
        raise ValueError("Duplicate flight IDs.")

    if crews[[
        "CrewID",
        "CrewType",
        "Reporting_Airline",
        "AvailableMinute",
        "RemainingDutyMinutes",
    ]].isna().any().any():
        raise ValueError("Incomplete crew resource data.")

    if flights[flight_columns].isna().any().any():
        raise ValueError("Incomplete recovery flight data.")

    print(f"Flights: {len(flights):,}")
    print(f"Synthetic crew units: {len(crews):,}")

    return crews, flights


def identify_at_risk_flights(crews, flights):
    print("\nChecking primary crew duty feasibility...")

    primary = crews[
        crews["CrewType"] == "Primary"
    ].copy()

    if primary["AssignedFlightID"].isna().any():
        raise ValueError(
            "Primary crew without assigned flight."
        )

    if primary["AssignedFlightID"].duplicated().any():
        raise ValueError(
            "Multiple primary crews assigned to one flight."
        )

    primary = primary.rename(
        columns={
            "CrewID": "PrimaryCrewID",
            "Reporting_Airline": "PrimaryAirline",
            "ProjectedDutyMinutes": "RequiredDutyMinutes",
            "AvailableMinute": "PrimaryAvailableMinute",
            "RemainingDutyMinutes": "PrimaryRemainingDuty",
        }
    )

    selected_columns = [
        "AssignedFlightID",
        "PrimaryCrewID",
        "PrimaryAirline",
        "RequiredDutyMinutes",
        "PrimaryAvailableMinute",
        "PrimaryRemainingDuty",
    ]

    merged = flights.merge(
        primary[selected_columns],
        left_on="RecoveryFlightID",
        right_on="AssignedFlightID",
        how="left",
        validate="one_to_one",
    )

    if merged["PrimaryCrewID"].isna().any():
        raise ValueError(
            "Some flights have no primary crew."
        )

    if not (
        merged["Reporting_Airline"]
        == merged["PrimaryAirline"]
    ).all():
        raise ValueError(
            "Primary crew airline mismatch."
        )

    if merged[[
        "RequiredDutyMinutes",
        "PrimaryAvailableMinute",
        "PrimaryRemainingDuty",
    ]].isna().any().any():
        raise ValueError("Missing primary crew constraints.")

    merged["PrimaryFeasible"] = (
        (
            merged["PrimaryAvailableMinute"]
            <= merged["RecoveryDepartureMinute"]
        )
        & (
            merged["PrimaryRemainingDuty"]
            >= merged["RequiredDutyMinutes"]
        )
    )

    at_risk = merged[
        ~merged["PrimaryFeasible"]
    ].copy()

    print(
        f"Primary crew conflicts: {len(at_risk):,}"
    )

    return merged, at_risk


def optimize_reserve_assignments(crews, at_risk):
    print("\nOptimizing reserve crew assignments...")

    reserves = crews[
        crews["CrewType"] == "Reserve"
    ].copy().reset_index(drop=True)

    at_risk = at_risk.reset_index(drop=True)

    n = len(at_risk)
    m = len(reserves)

    if n == 0:
        return pd.DataFrame(
            columns=[
                "RecoveryFlightID",
                "ReplacementCrewID",
                "ReplacementStatus",
                "ReserveAvailableMinute",
                "ReserveDutyRemaining",
            ]
        )

    # One dummy choice per at-risk flight allows
    # the solver to leave flights uncovered.
    #
    # Every real reserve crew can serve only one
    # flight in this experiment.
    cost = np.full(
        (n, m + n),
        INFEASIBLE_COST,
        dtype=float,
    )

    cost[:, m:] = DUMMY_COST

    for i, flight in enumerate(
        at_risk.itertuples(index=False)
    ):
        for j, reserve in enumerate(
            reserves.itertuples(index=False)
        ):

            airline_match = (
                flight.Reporting_Airline
                == reserve.Reporting_Airline
            )

            available_in_time = (
                reserve.AvailableMinute
                <= flight.RecoveryDepartureMinute
            )

            enough_duty = (
                reserve.RemainingDutyMinutes
                >= flight.RequiredDutyMinutes
            )

            if (
                airline_match
                and available_in_time
                and enough_duty
            ):
                # Prefer a feasible crew with less
                # unused duty capacity remaining
                # after completing the flight.
                #
                # This is a synthetic utilization
                # objective, not real crew economics.
                remaining_after = (
                    reserve.RemainingDutyMinutes
                    - flight.RequiredDutyMinutes
                )

                cost[i, j] = float(
                    remaining_after
                )

    rows, cols = linear_sum_assignment(cost)

    assignments = []

    for i, j in zip(rows, cols):
        flight_id = at_risk.loc[
            i, "RecoveryFlightID"
        ]

        if j < m:
            if cost[i, j] >= INFEASIBLE_COST:
                raise RuntimeError(
                    "Solver selected an infeasible crew."
                )

            reserve = reserves.iloc[j]

            assignments.append({
                "RecoveryFlightID": flight_id,
                "ReplacementCrewID": reserve["CrewID"],
                "ReplacementStatus": "Covered by reserve",
                "ReserveAvailableMinute": float(
                    reserve["AvailableMinute"]
                ),
                "ReserveDutyRemaining": float(
                    reserve["RemainingDutyMinutes"]
                ),
            })
        else:
            assignments.append({
                "RecoveryFlightID": flight_id,
                "ReplacementCrewID": None,
                "ReplacementStatus": "Uncovered",
                "ReserveAvailableMinute": np.nan,
                "ReserveDutyRemaining": np.nan,
            })

    return pd.DataFrame(assignments)


def main():
    print("=" * 65)
    print("AERORESILIENCE AI — CREW REASSIGNMENT")
    print("=" * 65)

    crews, flights = load_data()

    all_flights, at_risk = identify_at_risk_flights(
        crews, flights
    )

    reassigned = optimize_reserve_assignments(
        crews, at_risk
    )

    final = all_flights.merge(
        reassigned,
        on="RecoveryFlightID",
        how="left",
        validate="one_to_one",
    )

    final["FinalCrewID"] = np.where(
        final["ReplacementCrewID"].notna(),
        final["ReplacementCrewID"],
        final["PrimaryCrewID"],
    )

    final["CrewCovered"] = (
        final["PrimaryFeasible"]
        | final["ReplacementCrewID"].notna()
    )

    final["CrewRecoveryStatus"] = np.select(
        [
            final["PrimaryFeasible"],
            final["ReplacementCrewID"].notna(),
        ],
        [
            "Primary crew retained",
            "Reserve crew assigned",
        ],
        default="Uncovered",
    )

    # An uncovered flight has no feasible crew
    # under this simplified fixed-departure policy.
    final.loc[
        ~final["CrewCovered"], "FinalCrewID"
    ] = None

    covered = int(final["CrewCovered"].sum())
    uncovered = int((~final["CrewCovered"]).sum())

    reserve_used = int(
        final["ReplacementCrewID"].notna().sum()
    )

    at_risk_count = len(at_risk)

    # -------------------------------------------------
    # VALIDATION
    # -------------------------------------------------

    used_crews = final.loc[
        final["CrewCovered"], "FinalCrewID"
    ]

    if used_crews.duplicated().any():
        raise RuntimeError(
            "A crew was assigned to multiple flights."
        )

    assigned_reserves = final[
        final["ReplacementCrewID"].notna()
    ]

    if not (
        assigned_reserves["ReserveAvailableMinute"]
        <= assigned_reserves["RecoveryDepartureMinute"]
    ).all():
        raise RuntimeError(
            "Reserve availability violation."
        )

    if not (
        assigned_reserves["ReserveDutyRemaining"]
        >= assigned_reserves["RequiredDutyMinutes"]
    ).all():
        raise RuntimeError(
            "Reserve duty limit violation."
        )

    if covered + uncovered != len(flights):
        raise RuntimeError(
            "Crew coverage reconciliation failed."
        )

    # -------------------------------------------------
    # SUMMARY
    # -------------------------------------------------

    summary = pd.DataFrame([{
        "RecoveryFlights": len(final),
        "InitialCrewConflicts": at_risk_count,
        "ReserveCrewsUsed": reserve_used,
        "UncoveredFlights": uncovered,
        "CoveredFlights": covered,
        "CoverageRatePercent": (
            100 * covered / len(final)
            if len(final) else 0
        ),
    }])

    airline_summary = (
        final.groupby("Reporting_Airline")
        .agg(
            Flights=("RecoveryFlightID", "size"),
            CoveredFlights=("CrewCovered", "sum"),
        )
        .reset_index()
    )

    airline_summary["UncoveredFlights"] = (
        airline_summary["Flights"]
        - airline_summary["CoveredFlights"]
    )

    # -------------------------------------------------
    # SAVE REPORTS
    # -------------------------------------------------

    final.to_csv(
        REPORT_DIR / "crew_reassignment_details.csv",
        index=False,
    )

    summary.to_csv(
        REPORT_DIR / "crew_reassignment_summary.csv",
        index=False,
    )

    airline_summary.to_csv(
        REPORT_DIR / "crew_reassignment_by_airline.csv",
        index=False,
    )

    # -------------------------------------------------
    # TERMINAL OUTPUT
    # -------------------------------------------------

    print("\nCREW REASSIGNMENT RESULTS")
    print(summary.to_string(index=False))

    print("\nAIRLINE COVERAGE")
    print(airline_summary.to_string(index=False))

    print("\nAT-RISK FLIGHT DECISIONS")
    print(
        final.loc[
            ~final["PrimaryFeasible"],
            [
                "RecoveryFlightID",
                "Reporting_Airline",
                "PrimaryCrewID",
                "ReplacementCrewID",
                "CrewRecoveryStatus",
            ],
        ].to_string(index=False)
    )

    print("\nIMPORTANT LIMITATIONS")
    print(
        "This optimization uses synthetic abstract "
        "crew units, not real personnel assignments."
    )
    print(
        "Crew qualifications, aircraft type ratings, "
        "legal rest requirements, and actual FAA "
        "duty-time regulations are not modeled."
    )
    print(
        "Only one flight per crew unit is allowed."
    )
    print(
        "Flight departure times remain fixed. "
        "Uncovered flights are not automatically "
        "delayed or cancelled."
    )
    print(
        "Reserve assignments are not yet integrated "
        "with aircraft-aware recovery schedules."
    )
    print(
        "Coverage is a simulated feasibility result, "
        "not a claim about actual airline staffing."
    )

    print("\nReports saved successfully.")
    print(
        "\nAERORESILIENCE AI — "
        "CREW REASSIGNMENT COMPLETE"
    )


if __name__ == "__main__":
    main()
