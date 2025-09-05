from __future__ import annotations

import argparse
import json
import math
import random
from typing import Dict, List, Mapping

from .api_client import ApiClient
from .strategy import (
    build_attribute_order_constrained,
    compute_constraint_sets,
    correlate_binary_joint,
    correlate_binary_joint_vectorized_parallel,
    make_online_policy,
    make_online_policy_from_primal,
    multiplicative_weights_lp,
    solve_lp_primal,
)


def run(base_url: str, scenario: int, player_id: str, N: int = 1000, debug: bool = False, use_lp: bool = False) -> None:
    api = ApiClient(base_url)

    # Start game and fetch stats
    init = api.new_game(scenario=scenario, player_id=player_id)
    constraints_min_count = {c.attribute: c.minCount for c in init.constraints}

    attribute_order = build_attribute_order_constrained(init.relativeFrequencies, constraints_min_count)
    # Build joint distribution approximation using parallel processing
    type_probs = correlate_binary_joint_vectorized_parallel(
        attribute_order, init.relativeFrequencies, init.correlations, num_samples=150000
    )

    # Use the original constraints (no marginal capping). LP will throttle types as needed.
    constraint_sets = compute_constraint_sets(attribute_order, constraints_min_count, N)

    # Comprehensive feasibility analysis
    print(f"[info] Scenario {scenario} analysis:")
    for attr, min_count in constraints_min_count.items():
        pop_share = sum(type_probs[t] for t in constraint_sets[attr][1])
        alpha = constraint_sets[attr][0]
        print(f"  {attr}: requested {min_count}/{N} ({alpha:.3f}), pop_share={pop_share:.3f}")

    if debug:
        print("[debug] Joint distribution (top 10 types by frequency):")
        sorted_types = sorted(type_probs.items(), key=lambda x: x[1], reverse=True)[:10]
        for t, prob in sorted_types:
            attrs = [a for a, val in zip(attribute_order, t) if val == 1]
            print(f"    {attrs or ['none']}: {prob:.6f}")
        print(f"[debug] Total types: {len(type_probs)}")

    # Compute base acceptance probabilities
    if use_lp:
        print("[info] Using primal LP solver with exact r_t rates")
        r_by_type, A_rate, lambdas = solve_lp_primal(type_probs, constraint_sets)
        base_accept = r_by_type
        print(f"[info] LP solution: A* = {A_rate:.3f}")
    else:
        print("[info] Using multiplicative weights heuristic")
        base_accept = multiplicative_weights_lp(type_probs, constraint_sets, iterations=450)

    # Estimate expected rejections using A* = sum_t p_t a_t
    A_star = 0.0
    for t, p_t in type_probs.items():
        A_star += p_t * float(base_accept.get(t, 0.0))

    print(f"[info] Base acceptance rate A* = {A_star:.3f}")

    if A_star <= 1e-9:
        print("[warn] Acceptance rate collapsed to ~0 under constraints; scenario likely impossible under reject cap.")
        print("[info] Proceeding anyway with endgame guard to minimize chance of failure.")
    else:
        rej_exp = int(round(N * ((1.0 / A_star) - 1.0)))
        if rej_exp > 20000:
            print(f"[warn] Expected rejections {rej_exp} exceed cap 20000; run may fail early.")
        else:
            print(f"[info] Expected rejections: {rej_exp}")

    # Show top types by acceptance probability and their population frequency
    top_types = sorted(base_accept.items(), key=lambda x: x[1], reverse=True)[:5]
    print("[info] Top 5 types by acceptance probability:")
    for t, prob in top_types:
        attrs = [a for a, val in zip(attribute_order, t) if val == 1]
        pop_freq = type_probs.get(t, 0.0)
        leverage = len(attrs)  # How many constraints this type satisfies
        print(f"  {attrs or ['none']}: accept={prob:.3f}, freq={pop_freq:.4f}, leverage={leverage}")

    # Also show types with highest population frequency for comparison
    freq_types = sorted(type_probs.items(), key=lambda x: x[1], reverse=True)[:5]
    print("[info] Top 5 types by population frequency:")
    for t, freq in freq_types:
        attrs = [a for a, val in zip(attribute_order, t) if val == 1]
        accept_prob = base_accept.get(t, 0.0)
        print(f"  {attrs or ['none']}: freq={freq:.4f}, accept={accept_prob:.3f}")

    if use_lp:
        # Set fixed seed for reproducible LP results
        rng = random.Random(42)
        step, policy_state = make_online_policy_from_primal(
            N=N,
            attribute_order=attribute_order,
            r_by_type=base_accept,
            constraint_sets=constraint_sets,
            rng=rng,
        )
    else:
        step, policy_state = make_online_policy(
            N=N,
            attribute_order=attribute_order,
            base_accept=base_accept,
            constraint_sets=constraint_sets,
            lambda_nudge=0.5,
            safety_beta=1.0,
        )

    # Iterate game
    state = api.decide_and_next(init.gameId, person_index=0, accept=None)
    if getattr(state, "status", None) != "running":
        print("Game finished immediately.")
        return

    person_index = state.nextPerson.personIndex
    next_attributes = state.nextPerson.attributes

    while True:
        accept, _ = step(next_attributes)
        state = api.decide_and_next(init.gameId, person_index=person_index, accept=accept)
        if state.status != "running":
            print(json.dumps({"status": state.status, "rejectedCount": state.rejectedCount}))
            break
        person_index = state.nextPerson.personIndex
        next_attributes = state.nextPerson.attributes


def main() -> None:
    parser = argparse.ArgumentParser(description="Berghain Bouncer Algorithm CLI")
    parser.add_argument("--scenario", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--N", type=int, default=1000)
    parser.add_argument("--debug", action="store_true", help="Show debug information about joint distribution")
    parser.add_argument("--use-lp", action="store_true", help="Use primal LP solver instead of multiplicative weights (default: MW)")
    args = parser.parse_args()
    base_url = "https://berghain.challenges.listenlabs.ai"
    player_id = "15f0e870-99e8-4572-a1b2-0cf0dcef4d8d"
    run(base_url=base_url, scenario=args.scenario, player_id=player_id, N=args.N, debug=args.debug, use_lp=args.use_lp)


if __name__ == "__main__":
    main()


