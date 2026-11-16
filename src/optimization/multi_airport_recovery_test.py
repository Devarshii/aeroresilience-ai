
from pathlib import Path
from zoneinfo import ZoneInfo

import airportsdata
import numpy as np
import pandas as pd

from src.optimization.aircraft_aware_recovery import (
    create_utc_schedule,
)
from src.optimization.network_recovery_optimizer import (
    build_chains,
    simulate_chain,
)

# =====================================================
# AERORESILIENCE AI
# STEP 32 — MULTI-AIRPORT RECOVERY TESTING
# =====================================================

RAW_FILE = Path("data/raw/bts_flights_2026_08.parquet")
REPORT_DIR = Path("outputs/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

AIRPORTS = ["ORD", "DFW", "DEN", "ATL"]

SCENARIO_DATE = "2026-08-19"
START_HOUR = 10
STOP_HOUR = 13

SLOT_MINUTES = 15
DEPARTURES_PER_SLOT = 8

MAX_EXTRA_DELAY = 60
MAX_PASSES = 2

MINUTE_NS = 60_000_000_000

COLUMNS = [
    "FlightDate",
    "Reporting_Airline",
    "Tail_Number",
    "Flight_Number_Reporting_Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
    "CRSElapsedTime",
    "Cancelled",
    "Diverted",
]


# =====================================================
# 1. LOAD HISTORICAL FLIGHTS
# =====================================================

def load_historical_flights():

    print("\nLoading August 2026 flight records...")

    df = pd.read_parquet(
        RAW_FILE,
        columns=COLUMNS,
    )

    df["FlightDate"] = pd.to_datetime(
        df["FlightDate"], errors="coerce"
    )

    for col in [
        "CRSDepTime",
        "CRSElapsedTime",
        "Cancelled",
        "Diverted",
    ]:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    for col in [
        "Reporting_Airline",
        "Tail_Number",
        "Origin",
        "Dest",
    ]:
        df[col] = df[col].astype("string").str.strip()

    df = df.dropna(
        subset=[
            "FlightDate",
            "Reporting_Airline",
            "Tail_Number",
            "Origin",
            "Dest",
            "CRSDepTime",
            "CRSElapsedTime",
            "Cancelled",
            "Diverted",
        ]
    ).copy()

    df = df[
        df["Tail_Number"].ne("")
        & df["CRSElapsedTime"].between(1, 1440)
    ].copy()

    hours = df["CRSDepTime"] // 100
    minutes = df["CRSDepTime"] % 100

    valid = (
        hours.between(0, 24)
        & minutes.between(0, 59)
        & ((hours < 24) | (minutes == 0))
    )

    df = df.loc[valid].copy()

    df["ScheduledMinute"] = (
        (df["CRSDepTime"] // 100) * 60
        + (df["CRSDepTime"] % 100)
    )

    df["Operating"] = (
        df["Cancelled"].eq(0)
        & df["Diverted"].eq(0)
    )

    print(f"Usable records: {len(df):,}")

    return df


# =====================================================
# 2. PREPARE AIRPORT SCENARIO
# =====================================================

def prepare_scenario(raw, airport):

    start_minute = START_HOUR * 60
    end_minute = STOP_HOUR * 60

    exposed = raw[
        (raw["Origin"] == airport)
        & (
            raw["FlightDate"]
            == pd.Timestamp(SCENARIO_DATE)
        )
        & (
            raw["ScheduledMinute"] >= start_minute
        )
        & (
            raw["ScheduledMinute"] < end_minute
        )
        & raw["Operating"]
    ].copy()

    if exposed.empty:
        raise ValueError(
            f"No eligible departures for {airport}"
        )

    affected_aircraft = exposed[
        ["Reporting_Airline", "Tail_Number"]
    ].drop_duplicates()

    affected = raw.merge(
        affected_aircraft,
        on=["Reporting_Airline", "Tail_Number"],
        how="inner",
    ).copy()

    # Preserve original row identity to distinguish
    # the selected ground-stop flights.
    affected["RecoverySeed"] = affected.index.isin(
        []
    )

    # Identify seeds using the same flight fields
    # and the same local-time scenario window.
    affected["RecoverySeed"] = (
        (affected["Origin"] == airport)
        & (
            affected["FlightDate"]
            == pd.Timestamp(SCENARIO_DATE)
        )
        & (
            affected["ScheduledMinute"] >= start_minute
        )
        & (
            affected["ScheduledMinute"] < end_minute
        )
        & affected["Operating"]
    )

    # Convert all affected-aircraft flights to UTC.
    affected = create_utc_schedule(affected)

    affected = affected.reset_index(drop=True)

    affected["DepMinuteUTC"] = (
        affected["DepUTC"].astype("int64")
        // MINUTE_NS
    )

    affected["ArrMinuteUTC"] = (
        affected["ArrUTC"].astype("int64")
        // MINUTE_NS
    )

    recovery = affected[
        affected["RecoverySeed"]
    ].copy()

    if len(recovery) != len(exposed):
        raise ValueError(
            f"Scenario coverage mismatch for {airport}: "
            f"{len(recovery)} UTC-valid seeds versus "
            f"{len(exposed)} eligible departures."
        )

    recovery = recovery.sort_values(
        [
            "DepMinuteUTC",
            "Reporting_Airline",
            "Tail_Number",
        ],
        kind="stable",
    )

    # Construct the hypothetical local ground-stop end.
    airport_data = airportsdata.load("IATA")
    timezone = airport_data[airport]["tz"]

    stop_end_local = pd.Timestamp(
        f"{SCENARIO_DATE} {STOP_HOUR:02d}:00:00"
    ).tz_localize(ZoneInfo(timezone))

    stop_end_utc = stop_end_local.tz_convert("UTC")

    end_utc_minute = (
        stop_end_utc.value // MINUTE_NS
    )

    slots = (
        end_utc_minute
        + (
            np.arange(len(recovery))
            // DEPARTURES_PER_SLOT
        ) * SLOT_MINUTES
    ).astype(float)

    seed_indices = recovery.index.to_numpy()

    scheduled = recovery[
        "DepMinuteUTC"
    ].to_numpy(dtype=float)

    baseline_delays = slots - scheduled

    if (baseline_delays < 0).any():
        raise RuntimeError(
            f"Negative baseline delay for {airport}"
        )

    return (
        affected,
        recovery,
        seed_indices,
        scheduled,
        slots,
        baseline_delays,
    )


# =====================================================
# 3. BUILD RECOVERY COST FUNCTION
# =====================================================

def make_cost_function(affected, seed_indices, scheduled):

    chains, seed_info = build_chains(affected)

    positions = [
        seed_info[int(index)]
        for index in seed_indices
    ]

    chain_seed_rows = {}

    for i, (key, local_position) in enumerate(positions):
        chain_seed_rows.setdefault(key, []).append(
            (i, local_position)
        )

    def chain_cost(key, assignment):

        direct_delays = {}

        for i, local_position in chain_seed_rows.get(
            key, []
        ):
            direct_delays[local_position] = max(
                0.0,
                float(assignment[i] - scheduled[i]),
            )

        return simulate_chain(
            chains[key],
            direct_delays,
        )

    return chains, positions, chain_cost


# =====================================================
# 4. OPTIMIZE RECOVERY SLOTS
# =====================================================

def optimize_scenario(
    chains,
    positions,
    chain_cost,
    scheduled,
    baseline_slots,
    baseline_delays,
):

    assignment = baseline_slots.copy()

    costs = {
        key: chain_cost(key, assignment)
        for key in chains
    }

    initial_total = sum(costs.values())

    accepted_swaps = 0

    for iteration in range(MAX_PASSES):

        accepted_this_pass = 0

        for i in range(len(assignment)):

            for j in range(i + 1, len(assignment)):

                if assignment[i] == assignment[j]:
                    continue

                delay_i = assignment[j] - scheduled[i]
                delay_j = assignment[i] - scheduled[j]

                if not (
                    0 <= delay_i
                    <= baseline_delays[i] + MAX_EXTRA_DELAY
                ):
                    continue

                if not (
                    0 <= delay_j
                    <= baseline_delays[j] + MAX_EXTRA_DELAY
                ):
                    continue

                affected_keys = {
                    positions[i][0],
                    positions[j][0],
                }

                old_cost = sum(
                    costs[key] for key in affected_keys
                )

                assignment[i], assignment[j] = (
                    assignment[j], assignment[i]
                )

                new_costs = {
                    key: chain_cost(key, assignment)
                    for key in affected_keys
                }

                new_cost = sum(new_costs.values())

                if new_cost < old_cost - 1e-7:

                    costs.update(new_costs)

                    accepted_swaps += 1
                    accepted_this_pass += 1

                else:

                    assignment[i], assignment[j] = (
                        assignment[j], assignment[i]
                    )

        print(
            f"  Pass {iteration + 1}: "
            f"{accepted_this_pass} accepted swaps"
        )

        if accepted_this_pass == 0:
            break

    final_total = sum(
        chain_cost(key, assignment)
        for key in chains
    )

    if final_total > initial_total + 1e-6:
        raise RuntimeError(
            "Optimized delay exceeds baseline."
        )

    final_delay = assignment - scheduled

    if (
        (final_delay < 0).any()
        or (
            final_delay
            > baseline_delays + MAX_EXTRA_DELAY
        ).any()
    ):
        raise RuntimeError(
            "Recovery fairness constraint failed."
        )

    if not np.array_equal(
        np.sort(assignment),
        np.sort(baseline_slots),
    ):
        raise RuntimeError(
            "Recovery departure capacity changed."
        )

    return assignment, initial_total, final_total, accepted_swaps


# =====================================================
# 5. EVALUATE ALL AIRPORTS
# =====================================================

def main():

    print("=" * 65)
    print("AERORESILIENCE AI — MULTI-AIRPORT RECOVERY TEST")
    print("=" * 65)

    raw = load_historical_flights()

    results = []
    schedules = []

    for airport in AIRPORTS:

        print("\n" + "-" * 65)
        print(f"Testing {airport} — {SCENARIO_DATE}")
        print("-" * 65)

        (
            affected,
            recovery,
            seed_indices,
            scheduled,
            baseline_slots,
            baseline_delays,
        ) = prepare_scenario(raw, airport)

        chains, positions, chain_cost = (
            make_cost_function(
                affected,
                seed_indices,
                scheduled,
            )
        )

        # Counterfactual with zero directly imposed delays.
        no_stop_assignment = scheduled.copy()

        no_stop_total = sum(
            chain_cost(key, no_stop_assignment)
            for key in chains
        )

        (
            optimized_slots,
            fcfs_total,
            optimized_total,
            swaps,
        ) = optimize_scenario(
            chains,
            positions,
            chain_cost,
            scheduled,
            baseline_slots,
            baseline_delays,
        )

        fcfs_incremental = (
            fcfs_total - no_stop_total
        )

        optimized_incremental = (
            optimized_total - no_stop_total
        )

        improvement = (
            100 * (
                fcfs_incremental - optimized_incremental
            ) / fcfs_incremental
            if fcfs_incremental > 0
            else np.nan
        )

        results.append({
            "Airport": airport,
            "ScenarioDate": SCENARIO_DATE,
            "GroundStopHours": STOP_HOUR - START_HOUR,
            "RecoveryFlights": len(recovery),
            "AffectedAircraft": len(chains),
            "NoStopDelayMinutes": no_stop_total,
            "FCFSTotalDelayMinutes": fcfs_total,
            "OptimizedTotalDelayMinutes": optimized_total,
            "FCFSIncrementalDelayMinutes": fcfs_incremental,
            "OptimizedIncrementalDelayMinutes": (
                optimized_incremental
            ),
            "ImprovementPercent": improvement,
            "AcceptedSwaps": swaps,
        })

        schedule = recovery[
            [
                "FlightDate",
                "Reporting_Airline",
                "Tail_Number",
                "Origin",
                "Dest",
                "CRSDepTime",
            ]
        ].copy()

        schedule["ScenarioAirport"] = airport
        schedule["FCFSDelayMinutes"] = baseline_delays
        schedule["OptimizedDelayMinutes"] = (
            optimized_slots - scheduled
        )

        schedules.append(schedule)

        print(f"  Eligible recovery flights: {len(recovery):,}")
        print(f"  FCFS incremental delay: {fcfs_incremental:,.0f}")
        print(
            f"  Optimized incremental delay: "
            f"{optimized_incremental:,.0f}"
        )
        print(f"  Improvement: {improvement:.2f}%")

    # =================================================
    # 6. SAVE RESULTS
    # =================================================

    summary = pd.DataFrame(results)

    summary.to_csv(
        REPORT_DIR / "multi_airport_recovery_results.csv",
        index=False,
    )

    pd.concat(
        schedules,
        ignore_index=True,
    ).to_csv(
        REPORT_DIR / "multi_airport_recovery_schedules.csv",
        index=False,
    )

    print("\n" + "=" * 65)
    print("MULTI-AIRPORT RECOVERY COMPARISON")
    print("=" * 65)

    print(
        summary[
            [
                "Airport",
                "RecoveryFlights",
                "FCFSIncrementalDelayMinutes",
                "OptimizedIncrementalDelayMinutes",
                "ImprovementPercent",
                "AcceptedSwaps",
            ]
        ].to_string(index=False)
    )

    print("\nIMPORTANT LIMITATIONS")
    print(
        "Results come from hypothetical ground stops "
        "applied to August 2026 historical schedules."
    )
    print(
        "The 8-departures-per-15-minute capacity "
        "is assumed, not actual airport capacity."
    )
    print(
        "The same simplified propagation model "
        "is used for optimization and evaluation."
    )
    print(
        "Crew, gates, other airport traffic and "
        "aircraft substitutions are not included."
    )
    print(
        "These new scenarios test transfer across airports "
        "and dates, not real-world predictive accuracy."
    )

    print("\nReports saved successfully.")

    print(
        "\nAERORESILIENCE AI — "
        "MULTI-AIRPORT RECOVERY TEST COMPLETE"
    )


if __name__ == "__main__":
    main()
