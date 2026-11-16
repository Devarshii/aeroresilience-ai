
from pathlib import Path
import numpy as np
import pandas as pd

from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix

from src.Passengers.passenger_rebooking_optimizer import (
    load_passenger_risk,
    load_schedules,
    build_flight_lookup,
    rebook_policy,
)

# =====================================================
# AERORESILIENCE AI
# STEP 41 — GLOBAL PASSENGER ALLOCATION OPTIMIZATION
# =====================================================

REPORT_DIR = Path("outputs/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

POLICIES = ["FCFS", "AircraftAware"]

MIN_CONNECTION_MINUTES = 45
MAX_REBOOKING_HOURS = 36

TOLERANCE = 1e-5


# =====================================================
# 1. CREATE FEASIBLE PASSENGER-FLIGHT OPTIONS
# =====================================================

def build_problem(risk, lookup, policy):

    missed = risk[
        (risk["Policy"] == policy)
        & risk["MissedConnection"]
    ].copy()

    missed = missed.sort_values(
        ["OutboundScheduledDepartureUTC", "ItineraryID"]
    ).reset_index(drop=True)

    groups = []
    edges = []
    flight_capacities = {}

    for group_index, itinerary in enumerate(
        missed.itertuples(index=False)
    ):

        passengers = int(itinerary.SimulatedPassengers)

        ready_time = (
            itinerary.PredictedInboundArrivalUTC
            + pd.Timedelta(
                minutes=MIN_CONNECTION_MINUTES
            )
        )

        cutoff = (
            itinerary.OutboundScheduledDepartureUTC
            + pd.Timedelta(
                hours=MAX_REBOOKING_HOURS
            )
        )

        route = (
            itinerary.Airline,
            itinerary.ConnectionAirport,
            itinerary.FinalDestination,
        )

        group = {
            "GroupIndex": group_index,
            "ItineraryID": itinerary.ItineraryID,
            "Passengers": passengers,
            "Airline": itinerary.Airline,
            "ConnectionAirport": itinerary.ConnectionAirport,
            "FinalDestination": itinerary.FinalDestination,
        }

        groups.append(group)

        candidates = lookup.get(route)

        if candidates is None:
            continue

        # Find the originally missed onward flight.
        original = candidates[
            (
                candidates["DepUTC"]
                == itinerary.OutboundScheduledDepartureUTC
            )
            & (
                candidates["Flight_Number_Reporting_Airline"]
                == itinerary.OutboundFlightNumber
            )
        ]

        if len(original) != 1:
            raise ValueError(
                "Cannot uniquely identify original onward "
                f"flight for {itinerary.ItineraryID}"
            )

        original_arrival = original.iloc[0]["ArrUTC"]

        alternatives = candidates[
            (candidates["DepUTC"] >= ready_time)
            & (candidates["DepUTC"] <= cutoff)
        ]

        for flight in alternatives.itertuples(index=False):

            # Exclude the original flight.
            if (
                flight.DepUTC
                == itinerary.OutboundScheduledDepartureUTC
                and
                flight.Flight_Number_Reporting_Airline
                == itinerary.OutboundFlightNumber
            ):
                continue

            flight_id = flight.AlternativeFlightID

            capacity = int(flight.AvailableSeats)

            if capacity <= 0:
                continue

            flight_capacities[flight_id] = capacity

            arrival_delay = max(
                0.0,
                (
                    flight.ArrUTC - original_arrival
                ).total_seconds() / 60,
            )

            edges.append({
                "GroupIndex": group_index,
                "ItineraryID": itinerary.ItineraryID,
                "FlightID": flight_id,
                "Capacity": capacity,
                "ArrivalDelayMinutes": arrival_delay,
                "AlternativeDepartureUTC": flight.DepUTC,
                "AlternativeArrivalUTC": flight.ArrUTC,
            })

    return groups, edges, flight_capacities


# =====================================================
# 2. GLOBAL LINEAR PROGRAMMING OPTIMIZER
# =====================================================

def optimize_allocation(groups, edges, capacities):

    if not edges:
        return pd.DataFrame(), 0

    flight_ids = sorted(capacities)
    flight_index = {
        flight_id: index
        for index, flight_id in enumerate(flight_ids)
    }

    n_groups = len(groups)
    n_flights = len(flight_ids)
    n_edges = len(edges)

    # Each edge is a feasible assignment of passengers
    # from one itinerary group to one alternative flight.
    #
    # Constraints:
    # 1. Group allocations <= group passenger demand.
    # 2. Flight allocations <= synthetic available seats.
    # 3. All allocations >= 0.

    rows = []
    cols = []
    values = []

    for edge_index, edge in enumerate(edges):

        rows.extend([
            edge["GroupIndex"],
            n_groups + flight_index[edge["FlightID"]],
        ])

        cols.extend([
            edge_index,
            edge_index,
        ])

        values.extend([1.0, 1.0])

    A_ub = coo_matrix(
        (values, (rows, cols)),
        shape=(n_groups + n_flights, n_edges),
    ).tocsr()

    b_ub = np.concatenate([
        np.array(
            [group["Passengers"] for group in groups],
            dtype=float,
        ),
        np.array(
            [capacities[flight_id] for flight_id in flight_ids],
            dtype=float,
        ),
    ])

    # -------------------------------------------------
    # PHASE 1: MAXIMIZE PASSENGERS REBOOKED
    # -------------------------------------------------

    print("  Phase 1: maximizing passenger coverage...")

    phase1 = linprog(
        c=-np.ones(n_edges),
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=(0, None),
        method="highs",
    )

    if not phase1.success:
        raise RuntimeError(
            f"Coverage optimization failed: {phase1.message}"
        )

    maximum_rebooked = int(
        round(float(phase1.x.sum()))
    )

    # -------------------------------------------------
    # PHASE 2: MINIMIZE ARRIVAL DELAY
    # -------------------------------------------------

    print("  Phase 2: minimizing passenger arrival delay...")

    delay_cost = np.array(
        [edge["ArrivalDelayMinutes"] for edge in edges],
        dtype=float,
    )

    # Small deterministic tie-break preference.
    # It does not replace the arrival-delay objective.
    delay_cost = delay_cost + (
        np.arange(n_edges) / max(n_edges, 1)
    ) * 1e-6

    A_eq = csr_matrix(
        np.ones((1, n_edges), dtype=float)
    )

    phase2 = linprog(
        c=delay_cost,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=[maximum_rebooked],
        bounds=(0, None),
        method="highs",
    )

    if not phase2.success:
        raise RuntimeError(
            f"Delay optimization failed: {phase2.message}"
        )

    # Transportation-network linear programs with
    # integer supplies/capacities admit integer solutions.
    # Verify that property before converting passenger
    # allocations to whole numbers.
    if not np.allclose(
        phase2.x,
        np.rint(phase2.x),
        atol=TOLERANCE,
        rtol=0,
    ):
        raise RuntimeError(
            "Non-integer passenger allocation detected."
        )

    allocation = np.rint(phase2.x).astype(int)

    if allocation.sum() != maximum_rebooked:
        raise RuntimeError(
            "Maximum passenger coverage not preserved."
        )

    records = []

    for edge, passengers in zip(edges, allocation):

        if passengers <= 0:
            continue

        records.append({
            "ItineraryID": edge["ItineraryID"],
            "AlternativeFlightID": edge["FlightID"],
            "RebookedPassengers": int(passengers),
            "ArrivalDelayMinutes": edge["ArrivalDelayMinutes"],
            "AlternativeDepartureUTC": (
                edge["AlternativeDepartureUTC"]
            ),
            "AlternativeArrivalUTC": (
                edge["AlternativeArrivalUTC"]
            ),
        })

    assignments = pd.DataFrame(records)

    # -------------------------------------------------
    # VALIDATE ALLOCATION
    # -------------------------------------------------

    if not assignments.empty:

        used = assignments.groupby(
            "AlternativeFlightID"
        )["RebookedPassengers"].sum()

        for flight_id, count in used.items():
            if count > capacities[flight_id]:
                raise RuntimeError(
                    f"Seat capacity exceeded: {flight_id}"
                )

        group_usage = assignments.groupby(
            "ItineraryID"
        )["RebookedPassengers"].sum()

        demand = {
            group["ItineraryID"]: group["Passengers"]
            for group in groups
        }

        for itinerary_id, count in group_usage.items():
            if count > demand[itinerary_id]:
                raise RuntimeError(
                    f"Passenger demand exceeded: {itinerary_id}"
                )

    return assignments, maximum_rebooked


# =====================================================
# 3. COMPARE GREEDY AND GLOBAL ALLOCATION
# =====================================================

def evaluate_policy(risk, lookup, policy):

    print("\n" + "-" * 65)
    print(f"POLICY: {policy}")
    print("-" * 65)

    # Reproduce the existing Step 40 greedy baseline.
    greedy_assignments, greedy_outcomes = rebook_policy(
        risk,
        lookup,
        policy,
    )

    groups, edges, capacities = build_problem(
        risk,
        lookup,
        policy,
    )

    optimized, optimized_count = optimize_allocation(
        groups,
        edges,
        capacities,
    )

    total_passengers = sum(
        group["Passengers"] for group in groups
    )

    greedy_count = (
        int(greedy_assignments["RebookedPassengers"].sum())
        if not greedy_assignments.empty else 0
    )

    if optimized_count < greedy_count:
        raise RuntimeError(
            "Global allocation covered fewer passengers "
            "than the feasible greedy baseline."
        )

    if not greedy_outcomes.empty:

        greedy_uncovered = int(
            greedy_outcomes["UnaccommodatedPassengers"].sum()
        )

        if greedy_count + greedy_uncovered != total_passengers:
            raise RuntimeError(
                "Greedy passenger totals do not reconcile."
            )

    # Compute greedy arrival delay using the SAME
    # original scheduled onward arrivals used by
    # the optimization model.
    edge_delay = {
        (
            edge["ItineraryID"],
            edge["FlightID"],
        ): edge["ArrivalDelayMinutes"]
        for edge in edges
    }

    greedy_delay = 0.0

    for row in greedy_assignments.itertuples(index=False):

        key = (
            row.ItineraryID,
            row.AlternativeFlightID,
        )

        if key not in edge_delay:
            raise ValueError(
                f"Greedy assignment not feasible in LP: {key}"
            )

        greedy_delay += (
            row.RebookedPassengers * edge_delay[key]
        )

    optimized_delay = (
        float(
            (
                optimized["RebookedPassengers"]
                * optimized["ArrivalDelayMinutes"]
            ).sum()
        )
        if not optimized.empty else 0.0
    )

    print(f"  Passengers at risk: {total_passengers:,}")
    print(f"  Greedy rebooked: {greedy_count:,}")
    print(f"  Globally optimized rebooked: {optimized_count:,}")
    print(
        "  Additional passengers accommodated:",
        f"{optimized_count - greedy_count:,}"
    )

    return {
        "Policy": policy,
        "PassengersAtRisk": total_passengers,
        "GreedyRebooked": greedy_count,
        "OptimizedRebooked": optimized_count,
        "GreedyUnaccommodated": (
            total_passengers - greedy_count
        ),
        "OptimizedUnaccommodated": (
            total_passengers - optimized_count
        ),
        "AdditionalPassengersAccommodated": (
            optimized_count - greedy_count
        ),
        "GreedyPassengerArrivalDelayMinutes": greedy_delay,
        "OptimizedPassengerArrivalDelayMinutes": optimized_delay,
        "GreedyAvgArrivalDelayMinutes": (
            greedy_delay / greedy_count
            if greedy_count else np.nan
        ),
        "OptimizedAvgArrivalDelayMinutes": (
            optimized_delay / optimized_count
            if optimized_count else np.nan
        ),
    }, optimized


# =====================================================
# 4. MAIN PROGRAM
# =====================================================

def main():

    print("=" * 65)
    print("AERORESILIENCE AI — GLOBAL PASSENGER OPTIMIZATION")
    print("=" * 65)

    risk = load_passenger_risk()
    schedules = load_schedules()
    lookup = build_flight_lookup(schedules)

    results = []
    all_assignments = []

    for policy in POLICIES:

        summary, assignments = evaluate_policy(
            risk,
            lookup,
            policy,
        )

        results.append(summary)

        if not assignments.empty:
            assignments["Policy"] = policy
            all_assignments.append(assignments)

    comparison = pd.DataFrame(results)

    optimized_assignments = (
        pd.concat(all_assignments, ignore_index=True)
        if all_assignments
        else pd.DataFrame(
            columns=[
                "ItineraryID",
                "AlternativeFlightID",
                "RebookedPassengers",
                "ArrivalDelayMinutes",
                "AlternativeDepartureUTC",
                "AlternativeArrivalUTC",
                "Policy",
            ]
        )
    )

    summary_file = (
        REPORT_DIR / "global_passenger_optimization_summary.csv"
    )

    assignment_file = (
        REPORT_DIR / "global_passenger_optimized_assignments.csv"
    )

    comparison.to_csv(
        summary_file,
        index=False,
    )

    optimized_assignments.to_csv(
        assignment_file,
        index=False,
    )

    print("\n" + "=" * 65)
    print("GLOBAL REBOOKING OPTIMIZATION RESULTS")
    print("=" * 65)

    print(comparison.to_string(index=False))

    print("\nIMPORTANT LIMITATIONS")
    print(
        "Passenger demand and available seat inventory "
        "are synthetic."
    )
    print(
        "Passenger groups may be split across alternative flights."
    )
    print(
        "Only same-airline nonstop alternatives are modeled."
    )
    print(
        "The optimization maximizes rebooked passengers "
        "before minimizing arrival delay."
    )
    print(
        "Arrival delays are measured relative to each "
        "original onward flight's scheduled arrival."
    )
    print(
        "Alternative flights are assumed to operate on time."
    )
    print(
        "Real passenger priorities, airport capacity, "
        "ticketing, accessibility, fares and disruptions "
        "to alternative flights are not included."
    )
    print(
        "Optimality applies to this simplified mathematical "
        "allocation model, not real airline operations."
    )

    print(f"\nSummary saved: {summary_file}")
    print(f"Assignments saved: {assignment_file}")

    print(
        "\nAERORESILIENCE AI — "
        "GLOBAL PASSENGER OPTIMIZATION COMPLETE"
    )


if __name__ == "__main__":
    main()
