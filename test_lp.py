#!/usr/bin/env python3
"""Quick test of the LP implementation"""

import sys
sys.path.insert(0, '.')

from bba_cli.strategy import solve_lp_primal, compute_constraint_sets, build_attribute_order_constrained
from bba_cli.models import AttributeId, Constraint

# Simple test case
def test_lp():
    # Mock some basic data
    type_probs = {
        (0, 0): 0.4,  # No attributes
        (1, 0): 0.3,  # Has attribute A
        (0, 1): 0.2,  # Has attribute B
        (1, 1): 0.1,  # Has both attributes
    }

    attribute_order = ['A', 'B']

    # Mock constraints
    constraints_min_count = {'A': 50, 'B': 30}
    N = 100
    constraint_sets = compute_constraint_sets(attribute_order, constraints_min_count, N)

    print("Testing LP solver...")
    print(f"Type probabilities: {type_probs}")
    print(f"Constraints: {constraints_min_count}")
    print(f"N: {N}")

    try:
        r_by_type, A_rate, lambdas = solve_lp_primal(type_probs, constraint_sets)
        print("\nLP solution successful!")
        print(f"Acceptance rate A*: {A_rate:.3f}")
        print("Type acceptance rates:")
        for t, r in r_by_type.items():
            print(f"  {t}: {r:.3f}")
        print(f"Dual values (lambdas): {lambdas}")
        return True
    except Exception as e:
        print(f"LP failed: {e}")
        return False

if __name__ == "__main__":
    success = test_lp()
    sys.exit(0 if success else 1)
