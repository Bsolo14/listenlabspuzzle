#!/usr/bin/env python3
"""
Comprehensive test suite to verify that parallel implementations produce
identical results to the original algorithm across various scenarios.
"""
import sys
import os
import time
import numpy as np

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(__file__))

from bba_cli.strategy import (
    correlate_binary_joint,
    correlate_binary_joint_parallel,
    correlate_binary_joint_vectorized_parallel,
    build_attribute_order,
    build_attribute_order_constrained,
    compute_constraint_sets,
    multiplicative_weights_lp
)


def test_with_different_sample_sizes():
    """Test with various sample sizes to ensure consistency."""
    print("🔬 Testing different sample sizes...")

    # Fixed test data
    attribute_order = ["attr_A", "attr_B", "attr_C", "attr_D"]
    relative_frequencies = {
        "attr_A": 0.7,
        "attr_B": 0.5,
        "attr_C": 0.3,
        "attr_D": 0.2
    }
    correlations = {
        "attr_A": {"attr_B": 0.4, "attr_C": 0.2, "attr_D": 0.1},
        "attr_B": {"attr_A": 0.4, "attr_C": 0.3, "attr_D": 0.2},
        "attr_C": {"attr_A": 0.2, "attr_B": 0.3, "attr_D": 0.4},
        "attr_D": {"attr_A": 0.1, "attr_B": 0.2, "attr_C": 0.4}
    }

    sample_sizes = [1000, 5000, 10000, 25000, 50000]

    for num_samples in sample_sizes:
        print(f"  Testing with {num_samples} samples...")

        # Set seed for reproducible results
        np.random.seed(42)

        # Test original
        start = time.time()
        result_orig = correlate_binary_joint(
            attribute_order, relative_frequencies, correlations, num_samples
        )
        time_orig = time.time() - start

        # Set seed again for parallel
        np.random.seed(42)

        # Test parallel
        start = time.time()
        result_par = correlate_binary_joint_parallel(
            attribute_order, relative_frequencies, correlations, num_samples
        )
        time_par = time.time() - start

        # Set seed again for vectorized parallel
        np.random.seed(42)

        # Test vectorized parallel
        start = time.time()
        result_vec = correlate_binary_joint_vectorized_parallel(
            attribute_order, relative_frequencies, correlations, num_samples
        )
        time_vec = time.time() - start

        # Verify identical results
        tolerance = 1e-10
        for key in result_orig:
            if abs(result_orig[key] - result_par[key]) > tolerance:
                print(f"❌ Parallel result differs for {key}")
                return False
            if abs(result_orig[key] - result_vec[key]) > tolerance:
                print(f"❌ Vectorized result differs for {key}")
                return False

        print(f"    ✅ {num_samples} samples: all results identical")
    return True


def test_with_different_attributes():
    """Test with different numbers of attributes."""
    print("🔬 Testing different numbers of attributes...")

    test_configs = [
        (["attr_A"], {"attr_A": 0.6}, {}),  # Single attribute
        (["attr_A", "attr_B"], {"attr_A": 0.6, "attr_B": 0.4},
         {"attr_A": {"attr_B": 0.3}, "attr_B": {"attr_A": 0.3}}),  # Two attributes
        (["A", "B", "C", "D", "E"], {"A": 0.8, "B": 0.6, "C": 0.4, "D": 0.3, "E": 0.2},
         {"A": {"B": 0.5, "C": 0.3, "D": 0.2, "E": 0.1},
          "B": {"A": 0.5, "C": 0.4, "D": 0.3, "E": 0.2},
          "C": {"A": 0.3, "B": 0.4, "D": 0.5, "E": 0.3},
          "D": {"A": 0.2, "B": 0.3, "C": 0.5, "E": 0.4},
          "E": {"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.4}})  # Five attributes
    ]

    for i, (attr_order, freqs, corrs) in enumerate(test_configs):
        print(f"  Testing with {len(attr_order)} attributes...")

        num_samples = 5000
        np.random.seed(42)

        # Test all three implementations
        result_orig = correlate_binary_joint(attr_order, freqs, corrs, num_samples)
        np.random.seed(42)
        result_par = correlate_binary_joint_parallel(attr_order, freqs, corrs, num_samples)
        np.random.seed(42)
        result_vec = correlate_binary_joint_vectorized_parallel(attr_order, freqs, corrs, num_samples)

        # Verify identical results
        tolerance = 1e-10
        for key in result_orig:
            if abs(result_orig[key] - result_par[key]) > tolerance:
                print(f"❌ Config {i+1}: Parallel result differs for {key}")
                return False
            if abs(result_orig[key] - result_vec[key]) > tolerance:
                print(f"❌ Config {i+1}: Vectorized result differs for {key}")
                return False

        print(f"  ✅ Config {i+1} passed")

    return True


def test_edge_cases():
    """Test edge cases like empty attributes, zero correlations, etc."""
    print("🔬 Testing edge cases...")

    # Test with no correlations
    attr_order = ["attr_A", "attr_B", "attr_C"]
    freqs = {"attr_A": 0.6, "attr_B": 0.4, "attr_C": 0.2}
    corrs = {}  # No correlations

    print("  Testing with no correlations...")
    np.random.seed(42)
    result_orig = correlate_binary_joint(attr_order, freqs, corrs, 5000)
    np.random.seed(42)
    result_par = correlate_binary_joint_parallel(attr_order, freqs, corrs, 5000)
    np.random.seed(42)
    result_vec = correlate_binary_joint_vectorized_parallel(attr_order, freqs, corrs, 5000)

    tolerance = 1e-10
    for key in result_orig:
        if abs(result_orig[key] - result_par[key]) > tolerance:
            print(f"❌ No correlations test: Parallel result differs for {key}")
            return False
        if abs(result_orig[key] - result_vec[key]) > tolerance:
            print(f"❌ No correlations test: Vectorized result differs for {key}")
            return False

    print("  ✅ No correlations test passed")

    # Test with extreme frequencies (very high/low)
    freqs_extreme = {"attr_A": 0.95, "attr_B": 0.05, "attr_C": 0.5}
    corrs_extreme = {
        "attr_A": {"attr_B": 0.8, "attr_C": 0.1},
        "attr_B": {"attr_A": 0.8, "attr_C": 0.2},
        "attr_C": {"attr_A": 0.1, "attr_B": 0.2}
    }

    print("  Testing with extreme frequencies...")
    np.random.seed(42)
    result_orig = correlate_binary_joint(attr_order, freqs_extreme, corrs_extreme, 5000)
    np.random.seed(42)
    result_par = correlate_binary_joint_parallel(attr_order, freqs_extreme, corrs_extreme, 5000)
    np.random.seed(42)
    result_vec = correlate_binary_joint_vectorized_parallel(attr_order, freqs_extreme, corrs_extreme, 5000)

    for key in result_orig:
        if abs(result_orig[key] - result_par[key]) > tolerance:
            print(f"❌ Extreme frequencies test: Parallel result differs for {key}")
            return False
        if abs(result_orig[key] - result_vec[key]) > tolerance:
            print(f"❌ Extreme frequencies test: Vectorized result differs for {key}")
            return False

    print("  ✅ Extreme frequencies test passed")

    return True


def test_full_algorithm_pipeline():
    """Test the complete algorithm pipeline to ensure end-to-end correctness."""
    print("🔬 Testing full algorithm pipeline...")

    # Mock game initialization data
    relative_frequencies = {
        "local": 0.6,
        "black": 0.4,
        "regular": 0.3,
        "cool": 0.5
    }
    correlations = {
        "local": {"black": 0.3, "regular": 0.4, "cool": 0.2},
        "black": {"local": 0.3, "regular": 0.5, "cool": 0.1},
        "regular": {"local": 0.4, "black": 0.5, "cool": 0.3},
        "cool": {"local": 0.2, "black": 0.1, "regular": 0.3}
    }

    # Mock constraints
    constraints = [
        {"attribute": "local", "minCount": 400},
        {"attribute": "black", "minCount": 300},
        {"attribute": "regular", "minCount": 200}
    ]

    # Use constrained attribute order (only constrained attributes)
    constraints_min_count = {c["attribute"]: c["minCount"] for c in constraints}
    attribute_order = build_attribute_order_constrained(relative_frequencies, constraints_min_count)

    N = 1000
    num_samples = 25000

    print("  Running complete pipeline...")

    # Test all three implementations through the full pipeline
    for impl_name, impl_func in [
        ("Original", correlate_binary_joint),
        ("Parallel", correlate_binary_joint_parallel),
        ("Vectorized", correlate_binary_joint_vectorized_parallel)
    ]:
        np.random.seed(42)

        # 1. Generate joint distribution
        type_probs = impl_func(attribute_order, relative_frequencies, correlations, num_samples)

        # 2. Build constraint sets
        constraints_min_count = {c["attribute"]: c["minCount"] for c in constraints}
        constraint_sets = compute_constraint_sets(attribute_order, constraints_min_count, N)

        # 3. Run LP solver
        base_accept = multiplicative_weights_lp(type_probs, constraint_sets, iterations=200)

        # Verify we get reasonable results
        total_acceptance = sum(type_probs[t] * base_accept.get(t, 0.0) for t in type_probs)
        if not (0.1 <= total_acceptance <= 1.0):  # Should be reasonable acceptance rate (constrained optimization may achieve 100%)
            print(f"❌ {impl_name}: Unreasonable acceptance rate: {total_acceptance}")
            return False

        # Verify constraints are satisfied (approximately)
        for attr, (alpha, satisfying_types) in constraint_sets.items():
            expected_count = sum(type_probs[t] * base_accept.get(t, 0.0) for t in satisfying_types)
            if expected_count < alpha * 0.95:  # Allow 5% tolerance
                print(f"❌ {impl_name}: Constraint {attr} not satisfied: {expected_count} < {alpha}")
                return False

        print(f"  ✅ {impl_name} pipeline completed successfully")

    return True


def main():
    """Run all comprehensive tests."""
    print("🧪 COMPREHENSIVE ALGORITHM INTEGRITY TEST SUITE")
    print("=" * 60)

    tests = [
        ("Sample Size Variations", test_with_different_sample_sizes),
        ("Attribute Count Variations", test_with_different_attributes),
        ("Edge Cases", test_edge_cases),
        ("Full Pipeline", test_full_algorithm_pipeline)
    ]

    all_passed = True

    for test_name, test_func in tests:
        print(f"\n🔬 Running: {test_name}")
        try:
            if not test_func():
                all_passed = False
                print(f"❌ {test_name} FAILED")
            else:
                print(f"✅ {test_name} PASSED")
        except Exception as e:
            print(f"❌ {test_name} CRASHED: {e}")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Parallel implementations are functionally identical to original!")
        print("✅ Algorithm integrity is maintained across all scenarios!")
        return 0
    else:
        print("❌ SOME TESTS FAILED!")
        print("⚠️  Parallel implementations may have altered the algorithm!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
