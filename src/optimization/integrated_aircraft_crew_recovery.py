
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

# =====================================================
# AERORESILIENCE AI
# STEP 36 — INTEGRATED AIRCRAFT & CREW RECOVERY
# CORRECTED COMPLETE VERSION
# =====================================================

REPORT_DIR = Path("outputs/reports")
PROCESSED_DIR = Path("data/processed")

REPORT_DIR.mkdir(parents=True, exist_ok=True)

FLIGHT_FILE = (
    REPORT_DIR / "recovery_baseline_flights.csv"
)

OPTIMIZED_FILE = (
    REPORT_DIR / "aircraft_aware_optimized_schedule.csv"
)

CREW_FILE = (
    PROCESSED_DIR / "synthetic_crew_roster.parquet"
)

NETWORK_FILE = (
    REPORT_DIR / "validated_network_recovery_comparison.csv"
)

UNASSIGNED_COST = 1_000_000.0
INFEASIBLE_COST = 10_000_000.0

MAX_EXTRA_DELAY = 60
GROUND_STOP_END = 13 * 60

FLIGHT_KEYS = [
    "FlightDate",
    "Reporting_Airline",
    "Tail_Number",
    "Flight_Number_Reporting_Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
]

POLICIES = {
    "FCFS": "RecoveryDepartureMinute",
    "AircraftAware": "AircraftAwareDepartureMinute",
}


# =====================================================
# 1. STANDARDIZE FLIGHT IDENTIFIERS
# =====================================================

def standardize_flight_keys(df):

    df = df.copy()

    df["FlightDate"] = pd.to_datetime(
        df["FlightDate"],
        errors="coerce",
    ).dt.normalize()

    for column in [
        "Reporting_Airline",
        "Tail_Number",
        "Origin",
        "Dest",
    ]:
        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )

    for column in [
        "Flight_Number_Reporting_Airline",
        "CRSDepTime",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    return df


# =====================================================
# 2. RECOVER MISSING RECOVERY FLIGHT IDS
# =====================================================

def restore_recovery_ids(flights, optimized):

    print("\nChecking optimized flight identifiers...")

    if "RecoveryFlightID" in optimized.columns:

        if optimized["RecoveryFlightID"].notna().all():

            print(
                "RecoveryFlightID already available."
            )

            return optimized

    print(
        "RecoveryFlightID missing. "
        "Matching original flight identifiers..."
    )

    baseline_missing = [
        column for column in FLIGHT_KEYS
        if column not in flights.columns
    ]

    optimized_missing = [
        column for column in FLIGHT_KEYS
        if column not in optimized.columns
    ]

    if baseline_missing or optimized_missing:

        raise ValueError(
            "Cannot restore recovery IDs. "
            f"Missing baseline columns: {baseline_missing}. "
            f"Missing optimized columns: {optimized_missing}."
        )

    flights = standardize_flight_keys(flights)
    optimized = standardize_flight_keys(optimized)

    if flights[FLIGHT_KEYS].isna().any().any():

        raise ValueError(
            "Baseline flight identifiers contain missing values."
        )

    if optimized[FLIGHT_KEYS].isna().any().any():

        raise ValueError(
            "Optimized flight identifiers contain missing values."
        )

    if flights.duplicated(FLIGHT_KEYS).any():

        raise ValueError(
            "Duplicate baseline flight identities found."
        )

    if optimized.duplicated(FLIGHT_KEYS).any():

        raise ValueError(
            "Duplicate optimized flight identities found."
        )

    if "RecoveryFlightID" in optimized.columns:
        optimized = optimized.drop(
            columns=["RecoveryFlightID"]
        )

    optimized = optimized.merge(
        flights[
            FLIGHT_KEYS + ["RecoveryFlightID"]
        ],
        on=FLIGHT_KEYS,
        how="left",
        validate="one_to_one",
    )

    if optimized["RecoveryFlightID"].isna().any():

        raise ValueError(
            "Some optimized flights could not be matched "
            "to the original recovery schedule."
        )

    if len(optimized) != len(flights):

        raise ValueError(
            "Optimized and baseline flight counts differ."
        )

    print(
        f"Successfully restored "
        f"{len(optimized):,} recovery flight IDs."
    )

    return optimized


# =====================================================
# 3. LOAD AND VALIDATE INPUTS
# =====================================================

def load_inputs():

    print("\nLoading recovery and crew data...")

    for path in [
        FLIGHT_FILE,
        OPTIMIZED_FILE,
        CREW_FILE,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"Required project file missing: {path}"
            )

    flights = pd.read_csv(FLIGHT_FILE)
    optimized = pd.read_csv(OPTIMIZED_FILE)
    crews = pd.read_parquet(CREW_FILE)

    # Restore optimized flight identifiers when
    # they were not preserved in Step 28.
    optimized = restore_recovery_ids(
        flights,
        optimized,
    )

    required_flights = [
        "RecoveryFlightID",
        "Reporting_Airline",
        "ScheduledMinute",
        "RecoveryDepartureMinute",
        "BaselineDelayMinutes",
    ]

    required_optimized = [
        "RecoveryFlightID",
        "Reporting_Airline",
        "AircraftAwareDelayMinutes",
    ]

    required_crews = [
        "CrewID",
        "CrewType",
        "Reporting_Airline",
        "AssignedFlightID",
        "AvailableMinute",
        "RemainingDutyMinutes",
        "ProjectedDutyMinutes",
    ]

    for label, frame, required in [
        ("Baseline", flights, required_flights),
        ("Optimized", optimized, required_optimized),
        ("Crew", crews, required_crews),
    ]:

        missing = [
            column for column in required
            if column not in frame.columns
        ]

        if missing:

            raise ValueError(
                f"{label} missing fields: {missing}"
            )

    if flights["RecoveryFlightID"].duplicated().any():

        raise ValueError(
            "Duplicate baseline recovery IDs."
        )

    if optimized["RecoveryFlightID"].duplicated().any():

        raise ValueError(
            "Duplicate optimized recovery IDs."
        )

    if crews["CrewID"].duplicated().any():

        raise ValueError(
            "Duplicate crew identifiers."
        )

    if flights[required_flights].isna().any().any():

        raise ValueError(
            "Baseline recovery data is incomplete."
        )

    if optimized[required_optimized].isna().any().any():

        raise ValueError(
            "Optimized recovery data is incomplete."
        )

    # -------------------------------------------------
    # MATCH OPTIMIZED DELAYS TO BASELINE FLIGHTS
    # -------------------------------------------------

    flights = flights.merge(
        optimized[
            [
                "RecoveryFlightID",
                "Reporting_Airline",
                "AircraftAwareDelayMinutes",
            ]
        ],
        on=[
            "RecoveryFlightID",
            "Reporting_Airline",
        ],
        how="left",
        validate="one_to_one",
    )

    if flights["AircraftAwareDelayMinutes"].isna().any():

        raise ValueError(
            "Some optimized recovery flights were not matched."
        )

    if len(flights) != len(optimized):

        raise ValueError(
            "Recovery schedule flight count mismatch."
        )

    # -------------------------------------------------
    # CALCULATE AIRCRAFT-AWARE DEPARTURE TIMES
    # -------------------------------------------------

    flights["AircraftAwareDepartureMinute"] = (
        flights["ScheduledMinute"]
        + flights["AircraftAwareDelayMinutes"]
    )

    for column in POLICIES.values():

        if (
            flights[column] < GROUND_STOP_END
        ).any():

            raise ValueError(
                f"Departure before ground-stop end: {column}"
            )

    additional_delay = (
        flights["AircraftAwareDelayMinutes"]
        - flights["BaselineDelayMinutes"]
    )

    if (
        additional_delay > MAX_EXTRA_DELAY + 1e-6
    ).any():

        raise ValueError(
            "Aircraft-aware fairness constraint exceeded."
        )

    if (
        flights["AircraftAwareDelayMinutes"] < 0
    ).any():

        raise ValueError(
            "Negative optimized delay detected."
        )

    print(f"\nRecovery flights: {len(flights):,}")

    print(
        f"Synthetic crew units: {len(crews):,}"
    )

    print(
        "Optimized recovery schedules matched successfully."
    )

    return flights, crews


# =====================================================
# 4. EVALUATE ONE RECOVERY POLICY
# =====================================================

def evaluate_policy(
    flights,
    crews,
    policy,
    time_column,
):

    print(
        f"\nEvaluating {policy} crew feasibility..."
    )

    work = flights.copy()

    work["Policy"] = policy

    work["PolicyDepartureMinute"] = (
        pd.to_numeric(
            work[time_column],
            errors="coerce",
        )
    )

    primary = crews[
        crews["CrewType"] == "Primary"
    ].copy()

    reserves = crews[
        crews["CrewType"] == "Reserve"
    ].copy().reset_index(drop=True)

    if primary["AssignedFlightID"].isna().any():

        raise ValueError(
            "Primary crew assigned flight IDs missing."
        )

    if primary["AssignedFlightID"].duplicated().any():

        raise ValueError(
            "Multiple primary crews assigned to one flight."
        )

    # -------------------------------------------------
    # PREPARE PRIMARY CREW INFORMATION
    # -------------------------------------------------

    primary = primary.rename(
        columns={
            "CrewID": "PrimaryCrewID",
            "Reporting_Airline": "PrimaryAirline",
            "AvailableMinute": "PrimaryAvailableMinute",
            "RemainingDutyMinutes": "PrimaryRemainingDuty",
            "ProjectedDutyMinutes": "RequiredDutyMinutes",
        }
    )

    work = work.merge(
        primary[
            [
                "AssignedFlightID",
                "PrimaryCrewID",
                "PrimaryAirline",
                "PrimaryAvailableMinute",
                "PrimaryRemainingDuty",
                "RequiredDutyMinutes",
            ]
        ],
        left_on="RecoveryFlightID",
        right_on="AssignedFlightID",
        how="left",
        validate="one_to_one",
    )

    if work["PrimaryCrewID"].isna().any():

        raise ValueError(
            "Some recovery flights have no primary crew."
        )

    if not (
        work["Reporting_Airline"]
        == work["PrimaryAirline"]
    ).all():

        raise ValueError(
            "Primary crew airline compatibility mismatch."
        )

    # -------------------------------------------------
    # CHECK PRIMARY CREW FEASIBILITY
    # -------------------------------------------------

    work["PrimaryFeasible"] = (
        (
            work["PrimaryAvailableMinute"]
            <= work["PolicyDepartureMinute"]
        )
        &
        (
            work["PrimaryRemainingDuty"]
            >= work["RequiredDutyMinutes"]
        )
    )

    at_risk = work[
        ~work["PrimaryFeasible"]
    ].copy().reset_index(drop=True)

    print(
        f"Primary crew conflicts: {len(at_risk):,}"
    )

    # -------------------------------------------------
    # RESERVE CREW ASSIGNMENT OPTIMIZATION
    # -------------------------------------------------

    replacements = {}

    n = len(at_risk)
    m = len(reserves)

    if n > 0:

        cost = np.full(
            (n, m + n),
            INFEASIBLE_COST,
            dtype=float,
        )

        # Dummy assignments represent uncovered flights.
        cost[:, m:] = UNASSIGNED_COST

        for i, flight in enumerate(
            at_risk.itertuples(index=False)
        ):

            for j, crew in enumerate(
                reserves.itertuples(index=False)
            ):

                airline_match = (
                    flight.Reporting_Airline
                    == crew.Reporting_Airline
                )

                available_in_time = (
                    crew.AvailableMinute
                    <= flight.PolicyDepartureMinute
                )

                enough_duty = (
                    crew.RemainingDutyMinutes
                    >= flight.RequiredDutyMinutes
                )

                if (
                    airline_match
                    and available_in_time
                    and enough_duty
                ):

                    cost[i, j] = float(
                        crew.RemainingDutyMinutes
                        - flight.RequiredDutyMinutes
                    )

        rows, cols = linear_sum_assignment(cost)

        for i, j in zip(rows, cols):

            flight_id = int(
                at_risk.loc[
                    i, "RecoveryFlightID"
                ]
            )

            if j < m:

                if cost[i, j] >= INFEASIBLE_COST:

                    raise RuntimeError(
                        "Infeasible reserve crew assignment."
                    )

                replacements[flight_id] = (
                    reserves.loc[j, "CrewID"]
                )

            else:

                replacements[flight_id] = None

    # -------------------------------------------------
    # FINAL CREW ASSIGNMENTS
    # -------------------------------------------------

    work["ReplacementCrewID"] = (
        work["RecoveryFlightID"].map(replacements)
    )

    replaced = (
        work["ReplacementCrewID"].notna()
    )

    work["CrewCovered"] = (
        work["PrimaryFeasible"]
        | replaced
    )

    work["FinalCrewID"] = (
        work["PrimaryCrewID"].astype("string")
    )

    work.loc[
        replaced, "FinalCrewID"
    ] = work.loc[
        replaced, "ReplacementCrewID"
    ]

    work.loc[
        ~work["CrewCovered"],
        "FinalCrewID",
    ] = pd.NA

    work["CrewStatus"] = np.select(
        [
            work["PrimaryFeasible"],
            replaced,
        ],
        [
            "Primary retained",
            "Reserve assigned",
        ],
        default="Uncovered",
    )

    # -------------------------------------------------
    # VALIDATE FINAL ASSIGNMENTS
    # -------------------------------------------------

    assigned = work.loc[
        work["CrewCovered"],
        "FinalCrewID",
    ]

    if assigned.duplicated().any():

        raise RuntimeError(
            "A synthetic crew unit was assigned "
            "to multiple flights."
        )

    reserve_lookup = reserves.set_index("CrewID")

    for row in work.loc[
        replaced
    ].itertuples(index=False):

        reserve = reserve_lookup.loc[
            row.ReplacementCrewID
        ]

        if (
            reserve["Reporting_Airline"]
            != row.Reporting_Airline
        ):

            raise RuntimeError(
                "Reserve airline compatibility failure."
            )

        if (
            reserve["AvailableMinute"]
            > row.PolicyDepartureMinute
        ):

            raise RuntimeError(
                "Reserve crew availability violation."
            )

        if (
            reserve["RemainingDutyMinutes"]
            < row.RequiredDutyMinutes
        ):

            raise RuntimeError(
                "Reserve crew duty violation."
            )

    # -------------------------------------------------
    # POLICY SUMMARY
    # -------------------------------------------------

    summary = {
        "Policy": policy,
        "RecoveryFlights": len(work),
        "PrimaryCrewConflicts": int(
            (~work["PrimaryFeasible"]).sum()
        ),
        "ReserveCrewsUsed": int(
            replaced.sum()
        ),
        "CoveredFlights": int(
            work["CrewCovered"].sum()
        ),
        "UncoveredFlights": int(
            (~work["CrewCovered"]).sum()
        ),
        "CoverageRatePercent": float(
            work["CrewCovered"].mean() * 100
        ),
        "TotalInitialDepartureDelayMinutes": float(
            (
                work["PolicyDepartureMinute"]
                - work["ScheduledMinute"]
            ).sum()
        ),
    }

    return work, summary


# =====================================================
# 5. COMPARE AIRCRAFT & CREW RECOVERY POLICIES
# =====================================================

def main():

    print("=" * 65)
    print(
        "AERORESILIENCE AI — INTEGRATED AIRCRAFT & CREW"
    )
    print("=" * 65)

    flights, crews = load_inputs()

    all_details = []
    summaries = []

    for policy, column in POLICIES.items():

        details, summary = evaluate_policy(
            flights,
            crews,
            policy,
            column,
        )

        all_details.append(details)
        summaries.append(summary)

    comparison = pd.DataFrame(summaries)

    details = pd.concat(
        all_details,
        ignore_index=True,
    )

    # -------------------------------------------------
    # ADD PREVIOUS AIRCRAFT-CHAIN SIMULATION RESULTS
    # -------------------------------------------------

    if NETWORK_FILE.exists():

        network = pd.read_csv(NETWORK_FILE)

        required_network = [
            "Policy",
            "IncrementalDelayMinutes",
        ]

        missing = [
            column for column in required_network
            if column not in network.columns
        ]

        if missing:

            raise ValueError(
                f"Missing network metrics: {missing}"
            )

        network["PolicyKey"] = (
            network["Policy"].map(
                {
                    "FCFS": "FCFS",
                    "Aircraft Aware": "AircraftAware",
                }
            )
        )

        network = network.dropna(
            subset=["PolicyKey"]
        )

        comparison = comparison.merge(
            network[
                [
                    "PolicyKey",
                    "IncrementalDelayMinutes",
                ]
            ],
            left_on="Policy",
            right_on="PolicyKey",
            how="left",
            validate="one_to_one",
        ).drop(columns=["PolicyKey"])

        if comparison[
            "IncrementalDelayMinutes"
        ].isna().any():

            raise ValueError(
                "Missing aircraft-chain delay comparison."
            )

    # -------------------------------------------------
    # SAVE REPORTS
    # -------------------------------------------------

    details_file = (
        REPORT_DIR
        / "integrated_crew_recovery_details.csv"
    )

    comparison_file = (
        REPORT_DIR
        / "integrated_recovery_comparison.csv"
    )

    details.to_csv(
        details_file,
        index=False,
    )

    comparison.to_csv(
        comparison_file,
        index=False,
    )

    # -------------------------------------------------
    # TERMINAL RESULTS
    # -------------------------------------------------

    print("\n" + "=" * 65)
    print("INTEGRATED RECOVERY COMPARISON")
    print("=" * 65)

    print(
        comparison.to_string(index=False)
    )

    print("\nCREW COVERAGE BY POLICY")

    for row in comparison.itertuples(index=False):

        print(
            f"{row.Policy}: "
            f"{row.CoveredFlights}/"
            f"{row.RecoveryFlights} flights covered, "
            f"reserves used={row.ReserveCrewsUsed}, "
            f"uncovered={row.UncoveredFlights}"
        )

    print("\nIMPORTANT LIMITATIONS")

    print(
        "Crew units are synthetic, not real airline personnel."
    )

    print(
        "Crew availability and duty limits are hypothetical."
    )

    print(
        "Aircraft-chain delay comes from a separate "
        "simulation, not a joint optimization."
    )

    print(
        "Crew assignments do not alter the departure "
        "schedule in this experiment."
    )

    print(
        "Legal crew duty/rest rules, aircraft qualifications, "
        "crew repositioning, gates and airport capacity "
        "are not modeled."
    )

    print(
        "This is a research feasibility comparison, "
        "not a deployable airline recovery schedule."
    )

    print("\nReports saved successfully.")

    print(
        f"Details saved: {details_file}"
    )

    print(
        f"Comparison saved: {comparison_file}"
    )

    print(
        "\nAERORESILIENCE AI — "
        "INTEGRATED RECOVERY ANALYSIS COMPLETE"
    )


# =====================================================
# 6. RUN PROGRAM
# =====================================================

if __name__ == "__main__":
    main()
