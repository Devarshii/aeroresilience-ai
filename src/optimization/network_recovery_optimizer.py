
from pathlib import Path

import numpy as np
import pandas as pd

from src.optimization.aircraft_aware_recovery import (
    load_flights,
    create_utc_schedule,
)

# =====================================================
# AERORESILIENCE AI
# STEP 28 — AIRCRAFT-AWARE NETWORK RECOVERY OPTIMIZER
# =====================================================

REPORT_DIR = Path("outputs/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MIN_TURN = 20
MAX_GAP = 600
MAX_EXTRA_DELAY = 60
MAX_PASSES = 3

MINUTE_NS = 60_000_000_000


# =====================================================
# 1. BUILD AIRCRAFT ROTATION CHAINS
# =====================================================

def build_chains(df):
    """
    Convert aircraft schedules into chronological
    rotation chains using UTC timestamps.
    """

    df = df.copy().reset_index(drop=True)

    if "DepMinuteUTC" not in df.columns:
        df["DepMinuteUTC"] = (
            df["DepUTC"].astype("int64") // MINUTE_NS
        )

    if "ArrMinuteUTC" not in df.columns:
        df["ArrMinuteUTC"] = (
            df["ArrUTC"].astype("int64") // MINUTE_NS
        )

    chains = {}
    seed_info = {}

    for key, group in df.groupby(
        ["Reporting_Airline", "Tail_Number"],
        sort=False,
    ):

        indices = group.index.to_numpy()

        chains[key] = {
            "dep": group["DepMinuteUTC"].to_numpy(),
            "arr": group["ArrMinuteUTC"].to_numpy(),
            "origin": group["Origin"].astype(str).to_numpy(),
            "dest": group["Dest"].astype(str).to_numpy(),
            "operating": group["Operating"]
            .fillna(False)
            .to_numpy(dtype=bool),
            "indices": indices,
        }

        for local_position, global_index in enumerate(indices):

            if bool(df.loc[global_index, "RecoverySeed"]):
                seed_info[int(global_index)] = (
                    key,
                    local_position,
                )

    return chains, seed_info


# =====================================================
# 2. SIMULATE AIRCRAFT DELAY PROPAGATION
# =====================================================

def simulate_chain(chain, direct_delays):
    """
    Simulate delays across one aircraft rotation chain.

    Scheduled aircraft assignments remain fixed.
    """

    previous_arrival = None
    previous_destination = None
    previous_delay = 0.0

    total_delay = 0.0

    n = len(chain["dep"])

    for i in range(n):

        if not chain["operating"][i]:

            previous_arrival = None
            previous_destination = None
            previous_delay = 0.0

            continue

        inherited_delay = 0.0

        if previous_arrival is not None:

            gap = (
                chain["dep"][i] - previous_arrival
            )

            valid_connection = (
                previous_destination == chain["origin"][i]
                and 0 <= gap <= MAX_GAP
            )

            if valid_connection:

                inherited_delay = max(
                    0.0,
                    previous_delay + MIN_TURN - gap,
                )

        direct_delay = float(
            direct_delays.get(i, 0.0)
        )

        delay = max(
            direct_delay,
            inherited_delay,
        )

        total_delay += delay

        previous_arrival = chain["arr"][i]
        previous_destination = chain["dest"][i]
        previous_delay = delay

    return total_delay


# =====================================================
# 3. MAIN NETWORK RECOVERY OPTIMIZER
# =====================================================

def main():

    print("=" * 65)
    print("AERORESILIENCE AI — NETWORK RECOVERY OPTIMIZER")
    print("=" * 65)

    # -------------------------------------------------
    # LOAD AIRCRAFT DATA
    # -------------------------------------------------

    print("\nLoading aircraft schedules...")

    df = load_flights()

    df = create_utc_schedule(df)

    # Create UTC numerical timestamps BEFORE
    # selecting the recovery flights.
    df["DepMinuteUTC"] = (
        df["DepUTC"].astype("int64") // MINUTE_NS
    )

    df["ArrMinuteUTC"] = (
        df["ArrUTC"].astype("int64") // MINUTE_NS
    )

    # Build aircraft rotation chains.
    chains, seed_info = build_chains(df)

    # Use matching row positions for all calculations.
    df = df.reset_index(drop=True)

    recovery = df[
        df["RecoverySeed"].fillna(False)
    ].copy()

    if recovery.empty:
        raise ValueError(
            "No recovery flights found."
        )

    # -------------------------------------------------
    # PREPARE RECOVERY DATA
    # -------------------------------------------------

    seed_indices = recovery.index.to_numpy()

    scheduled = recovery[
        "DepMinuteUTC"
    ].to_numpy(dtype=float)

    baseline_delays = recovery[
        "BaselineDelayMinutes"
    ].to_numpy(dtype=float)

    baseline_slots = scheduled + baseline_delays

    if not np.all(np.isfinite(scheduled)):
        raise ValueError(
            "Scheduled departure times contain invalid values."
        )

    if not np.all(np.isfinite(baseline_delays)):
        raise ValueError(
            "Baseline delays contain invalid values."
        )

    if not np.all(baseline_delays >= 0):
        raise ValueError(
            "Baseline delays cannot be negative."
        )

    print(f"\nRecovery flights: {len(recovery):,}")
    print(f"Aircraft chains: {len(chains):,}")

    # -------------------------------------------------
    # MAP RECOVERY FLIGHTS TO AIRCRAFT CHAINS
    # -------------------------------------------------

    positions = []

    for index in seed_indices:

        if int(index) not in seed_info:
            raise ValueError(
                f"Missing aircraft chain for flight {index}"
            )

        positions.append(
            seed_info[int(index)]
        )

    chain_seed_rows = {}

    for flight_index, (key, local_pos) in enumerate(positions):

        chain_seed_rows.setdefault(
            key, []
        ).append(
            (flight_index, local_pos)
        )

    # -------------------------------------------------
    # CALCULATE AIRCRAFT CHAIN COST
    # -------------------------------------------------

    def chain_cost(key, assignment):

        direct_delays = {}

        for flight_index, local_pos in chain_seed_rows.get(
            key, []
        ):

            delay = (
                assignment[flight_index]
                - scheduled[flight_index]
            )

            direct_delays[local_pos] = max(
                0.0, float(delay)
            )

        return simulate_chain(
            chains[key],
            direct_delays,
        )

    def evaluate_all(assignment):

        return sum(
            chain_cost(key, assignment)
            for key in chains
        )

    # -------------------------------------------------
    # START WITH FCFS BASELINE
    # -------------------------------------------------

    assignment = baseline_slots.copy()

    current_costs = {
        key: chain_cost(key, assignment)
        for key in chains
    }

    initial_cost = sum(
        current_costs.values()
    )

    print("\nINITIAL RECOVERY PERFORMANCE")

    print(
        f"FCFS total modeled delay: "
        f"{initial_cost:,.0f} minutes"
    )

    # -------------------------------------------------
    # AIRCRAFT-AWARE LOCAL SEARCH OPTIMIZATION
    # -------------------------------------------------

    print("\nStarting aircraft-aware optimization...")

    accepted_swaps = 0
    n = len(assignment)

    for iteration in range(MAX_PASSES):

        improvements_this_pass = 0

        for i in range(n):

            for j in range(i + 1, n):

                # Ignore identical departure slots.
                if assignment[i] == assignment[j]:
                    continue

                # Test swapping departure slots.
                new_delay_i = (
                    assignment[j] - scheduled[i]
                )

                new_delay_j = (
                    assignment[i] - scheduled[j]
                )

                # -------------------------------------
                # FAIRNESS CONSTRAINTS
                # -------------------------------------

                feasible_i = (
                    0 <= new_delay_i
                    <= baseline_delays[i] + MAX_EXTRA_DELAY
                )

                feasible_j = (
                    0 <= new_delay_j
                    <= baseline_delays[j] + MAX_EXTRA_DELAY
                )

                if not (feasible_i and feasible_j):
                    continue

                # -------------------------------------
                # FIND AFFECTED AIRCRAFT CHAINS
                # -------------------------------------

                key_i = positions[i][0]
                key_j = positions[j][0]

                affected_keys = {
                    key_i,
                    key_j,
                }

                old_cost = sum(
                    current_costs[key]
                    for key in affected_keys
                )

                # -------------------------------------
                # TEST SWAP
                # -------------------------------------

                assignment[i], assignment[j] = (
                    assignment[j],
                    assignment[i],
                )

                new_costs = {
                    key: chain_cost(key, assignment)
                    for key in affected_keys
                }

                new_cost = sum(
                    new_costs.values()
                )

                # -------------------------------------
                # ACCEPT ONLY IMPROVING SWAPS
                # -------------------------------------

                if new_cost < old_cost - 1e-7:

                    for key, cost in new_costs.items():
                        current_costs[key] = cost

                    accepted_swaps += 1
                    improvements_this_pass += 1

                else:

                    # Restore previous assignments.
                    assignment[i], assignment[j] = (
                        assignment[j],
                        assignment[i],
                    )

        current_total = sum(
            current_costs.values()
        )

        print(
            f"Pass {iteration + 1}: "
            f"Accepted swaps = {improvements_this_pass:,}, "
            f"Total delay = {current_total:,.0f} minutes"
        )

        if improvements_this_pass == 0:
            break

    # -------------------------------------------------
    # FINAL VERIFICATION
    # -------------------------------------------------

    print("\nValidating optimized schedule...")

    final_total = evaluate_all(assignment)

    if final_total > initial_cost + 1e-6:
        raise RuntimeError(
            "Optimized schedule is worse than FCFS."
        )

    final_delays = assignment - scheduled

    if not np.all(final_delays >= 0):
        raise RuntimeError(
            "Negative departure delay detected."
        )

    if not np.all(
        final_delays <= baseline_delays + MAX_EXTRA_DELAY
    ):
        raise RuntimeError(
            "Fairness restriction violated."
        )

    if not np.array_equal(
        np.sort(assignment),
        np.sort(baseline_slots),
    ):
        raise RuntimeError(
            "Departure slot allocation changed."
        )

    print("Schedule validation passed.")

    # -------------------------------------------------
    # COMPARE RECOVERY STRATEGIES
    # -------------------------------------------------

    print("\nComparing recovery strategies...")

    results = []

    policies = [
        ("FCFS", "BaselineDelayMinutes"),
        ("Network-priority", "OptimizedDelayMinutes"),
        ("Fairness-constrained", "FairDelayMinutes"),
    ]

    for policy_name, column in policies:

        policy_delays = recovery[
            column
        ].to_numpy(dtype=float)

        policy_slots = (
            scheduled + policy_delays
        )

        total_delay = evaluate_all(
            policy_slots
        )

        results.append({
            "Policy": policy_name,
            "TotalModeledDelayMinutes": total_delay,
        })

    results.append({
        "Policy": "Aircraft-aware local search",
        "TotalModeledDelayMinutes": final_total,
    })

    comparison = pd.DataFrame(results)

    if initial_cost > 0:

        comparison["ImprovementVsFCFSPercent"] = (
            100
            * (
                initial_cost
                - comparison["TotalModeledDelayMinutes"]
            )
            / initial_cost
        )

    else:

        comparison["ImprovementVsFCFSPercent"] = 0.0

    # -------------------------------------------------
    # SAVE OPTIMIZED FLIGHT SCHEDULE
    # -------------------------------------------------

    output = recovery.copy()

    output["AircraftAwareDepartureUTC"] = (
        pd.to_datetime(
            assignment * 60,
            unit="s",
            utc=True,
        )
    )

    output["AircraftAwareDelayMinutes"] = (
        final_delays
    )

    output["AdditionalDelayVsFCFS"] = (
        final_delays - baseline_delays
    )

    output_file = (
        REPORT_DIR
        / "aircraft_aware_optimized_schedule.csv"
    )

    comparison_file = (
        REPORT_DIR
        / "network_recovery_optimization_comparison.csv"
    )

    output.to_csv(
        output_file,
        index=False,
    )

    comparison.to_csv(
        comparison_file,
        index=False,
    )

    # -------------------------------------------------
    # DISPLAY RESULTS
    # -------------------------------------------------

    print("\n" + "=" * 65)
    print("RECOVERY STRATEGY COMPARISON")
    print("=" * 65)

    print(
        comparison.to_string(index=False)
    )

    improvement = (
        100 * (initial_cost - final_total) / initial_cost
        if initial_cost > 0 else 0.0
    )

    print("\nAIRCRAFT-AWARE OPTIMIZATION RESULTS")

    print(f"Accepted swaps: {accepted_swaps:,}")

    print(
        f"Initial modeled delay: "
        f"{initial_cost:,.0f} minutes"
    )

    print(
        f"Final modeled delay: "
        f"{final_total:,.0f} minutes"
    )

    print(
        f"Improvement vs FCFS: "
        f"{improvement:.2f}%"
    )

    print(
        "Maximum additional delay vs FCFS:",
        f"{output['AdditionalDelayVsFCFS'].max():.0f} minutes"
    )

    print(
        "Flights scheduled earlier:",
        int(
            (output["AdditionalDelayVsFCFS"] < 0).sum()
        ),
    )

    print(
        "Flights scheduled later:",
        int(
            (output["AdditionalDelayVsFCFS"] > 0).sum()
        ),
    )

    print(
        "Flights unchanged:",
        int(
            (output["AdditionalDelayVsFCFS"] == 0).sum()
        ),
    )

    # -------------------------------------------------
    # LIMITATIONS
    # -------------------------------------------------

    print("\nIMPORTANT LIMITATIONS")

    print(
        "This optimizer uses deterministic local search."
    )

    print(
        "Global optimality is not guaranteed."
    )

    print(
        "Aircraft assignments and scheduled flight "
        "durations are assumed fixed."
    )

    print(
        "The 60-minute additional-delay limit is hypothetical."
    )

    print(
        "Downstream airport capacity, crew assignments, "
        "gates, and passenger rebooking are not modeled."
    )

    print(
        "The objective measures total simulated aircraft "
        "delay, not verified airline cost savings."
    )

    print(
        "\nReports saved successfully."
    )

    print(
        f"Optimized schedule: {output_file}"
    )

    print(
        f"Comparison report: {comparison_file}"
    )

    print(
        "\nAERORESILIENCE AI — "
        "NETWORK RECOVERY OPTIMIZATION COMPLETE"
    )


# =====================================================
# RUN PROGRAM
# =====================================================

if __name__ == "__main__":
    main()
