
from pathlib import Path
import numpy as np
import pandas as pd

# =====================================================
# AERORESILIENCE AI
# STEP 34 — SYNTHETIC CREW DATA PREPARATION
# =====================================================

REPORT_DIR = Path("outputs/reports")
PROCESSED_DIR = Path("data/processed")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

RECOVERY_FILE = (
    REPORT_DIR / "recovery_baseline_flights.csv"
)

RANDOM_SEED = 42
RESERVE_RATIO = 0.20

# Experimental assumptions only.
SIMULATED_DUTY_LIMIT_MIN = 600
GROUND_STOP_END_MIN = 13 * 60


def main():
    print("=" * 65)
    print("AERORESILIENCE AI — CREW DATA PREPARATION")
    print("=" * 65)

    rng = np.random.default_rng(RANDOM_SEED)

    if not RECOVERY_FILE.exists():
        raise FileNotFoundError(
            f"Missing recovery schedule: {RECOVERY_FILE}"
        )

    flights = pd.read_csv(RECOVERY_FILE)

    required = [
        "RecoveryFlightID",
        "Reporting_Airline",
        "Origin",
        "Dest",
        "ScheduledMinute",
        "RecoveryDepartureMinute",
    ]

    missing = [
        col for col in required
        if col not in flights.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if flights["RecoveryFlightID"].duplicated().any():
        raise ValueError("Duplicate recovery flight IDs.")

    if flights[required].isna().any().any():
        raise ValueError("Missing required flight data.")

    flights = flights.sort_values(
        "RecoveryFlightID"
    ).reset_index(drop=True)

    if flights.empty:
        raise ValueError("No recovery flights available.")

    print(f"\nRecovery flights: {len(flights):,}")
    print(
        "Airlines:",
        flights["Reporting_Airline"].nunique()
    )

    # -------------------------------------------------
    # 1. GENERATE SYNTHETIC PRIMARY CREWS
    # -------------------------------------------------

    primary = flights[
        [
            "RecoveryFlightID",
            "Reporting_Airline",
        ]
    ].copy()

    primary["CrewID"] = [
        f"CREW-{i:05d}"
        for i in range(1, len(primary) + 1)
    ]

    primary["CrewType"] = "Primary"
    primary["AssignedFlightID"] = (
        primary["RecoveryFlightID"]
    )

    # Model one abstract crew unit per flight.
    # This is NOT an individual pilot or real roster.
    primary["BaseAirport"] = "ORD"
    primary["AvailableMinute"] = GROUND_STOP_END_MIN

    # Hypothetical duty already used before recovery.
    primary["DutyUsedMinutes"] = rng.integers(
        180, 421, size=len(primary)
    )

    # Synthetic projected operating duty.
    primary["ProjectedDutyMinutes"] = rng.integers(
        90, 241, size=len(primary)
    )

    # -------------------------------------------------
    # 2. GENERATE RESERVE CREWS
    # -------------------------------------------------

    reserve_records = []
    next_id = len(primary) + 1

    for airline, group in flights.groupby(
        "Reporting_Airline"
    ):
        reserve_count = max(
            1,
            int(np.ceil(len(group) * RESERVE_RATIO))
        )

        for _ in range(reserve_count):

            reserve_records.append({
                "RecoveryFlightID": np.nan,
                "Reporting_Airline": airline,
                "CrewID": f"CREW-{next_id:05d}",
                "CrewType": "Reserve",
                "AssignedFlightID": np.nan,
                "BaseAirport": "ORD",
                "AvailableMinute": int(
                    GROUND_STOP_END_MIN
                    + rng.choice([0, 30, 60, 90, 120])
                ),
                "DutyUsedMinutes": int(
                    rng.integers(60, 301)
                ),
                "ProjectedDutyMinutes": 0,
            })

            next_id += 1

    reserves = pd.DataFrame(reserve_records)

    # -------------------------------------------------
    # 3. COMBINE CREW DATA
    # -------------------------------------------------

    crews = pd.concat(
        [primary, reserves],
        ignore_index=True,
    )

    crews["DutyLimitMinutes"] = (
        SIMULATED_DUTY_LIMIT_MIN
    )

    crews["RemainingDutyMinutes"] = (
        crews["DutyLimitMinutes"]
        - crews["DutyUsedMinutes"]
    )

    crews["PrimaryDutyFeasible"] = (
        crews["DutyUsedMinutes"]
        + crews["ProjectedDutyMinutes"]
        <= crews["DutyLimitMinutes"]
    )

    crews["AvailableAfterGroundStop"] = (
        crews["AvailableMinute"]
        >= GROUND_STOP_END_MIN
    )

    # Prevent accidental duplicate crew identifiers.
    if crews["CrewID"].duplicated().any():
        raise ValueError("Duplicate crew identifiers.")

    if len(primary) != len(flights):
        raise RuntimeError("Primary crew coverage mismatch.")

    # -------------------------------------------------
    # 4. IDENTIFY POTENTIAL CREW CONSTRAINTS
    # -------------------------------------------------

    assigned = crews[
        crews["CrewType"] == "Primary"
    ].copy()

    reserve = crews[
        crews["CrewType"] == "Reserve"
    ].copy()

    duty_risk = assigned[
        ~assigned["PrimaryDutyFeasible"]
    ].copy()

    reserve["AvailableAt1300"] = (
        reserve["AvailableMinute"]
        <= GROUND_STOP_END_MIN
    )

    summary = pd.DataFrame([{
        "RecoveryFlights": len(flights),
        "PrimaryCrews": len(assigned),
        "ReserveCrews": len(reserve),
        "TotalSyntheticCrews": len(crews),
        "PrimaryDutyConflicts": len(duty_risk),
        "ReserveCrewsAvailableAt1300": int(
            reserve["AvailableAt1300"].sum()
        ),
        "AssumedDutyLimitMinutes":
            SIMULATED_DUTY_LIMIT_MIN,
    }])

    # -------------------------------------------------
    # 5. SAVE DATASETS
    # -------------------------------------------------

    crew_file = (
        PROCESSED_DIR / "synthetic_crew_roster.parquet"
    )

    summary_file = (
        REPORT_DIR / "crew_data_summary.csv"
    )

    risk_file = (
        REPORT_DIR / "crew_duty_risk.csv"
    )

    crews.to_parquet(
        crew_file, index=False
    )

    summary.to_csv(
        summary_file, index=False
    )

    duty_risk.to_csv(
        risk_file, index=False
    )

    # -------------------------------------------------
    # 6. TERMINAL OUTPUT
    # -------------------------------------------------

    print("\nCREW RESOURCE SUMMARY")
    print(summary.to_string(index=False))

    print("\nAIRLINE-LEVEL CREW DISTRIBUTION")

    distribution = (
        crews.groupby(
            ["Reporting_Airline", "CrewType"]
        )
        .size()
        .unstack(fill_value=0)
    )

    print(distribution.to_string())

    print("\nIMPORTANT MODELING LIMITATIONS")
    print(
        "All crews are synthetic and do not represent "
        "real airline employees."
    )
    print(
        "Each CrewID represents an abstract operating "
        "crew unit, not a verified qualified crew."
    )
    print(
        "Airline affiliation is the only compatibility "
        "attribute modeled at this stage."
    )
    print(
        "The 600-minute duty limit is hypothetical, "
        "not an FAA crew legality rule."
    )
    print(
        "Aircraft type ratings, rest requirements, "
        "crew positioning and actual duty regulations "
        "are not yet modeled."
    )
    print(
        "This step creates data; it does not "
        "perform crew reassignment."
    )

    print(f"\nCrew roster saved: {crew_file}")
    print(f"Summary saved: {summary_file}")
    print(f"Duty risk report saved: {risk_file}")

    print(
        "\nAERORESILIENCE AI — "
        "CREW DATA PREPARATION COMPLETE"
    )


if __name__ == "__main__":
    main()
