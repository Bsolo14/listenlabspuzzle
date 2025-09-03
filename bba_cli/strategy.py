from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Callable, Dict, List, Mapping, Set, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
from scipy.special import erfinv

from .models import AttributeId, Constraint, TypeKey


def process_samples_chunk(chunk: np.ndarray, thresholds_array: np.ndarray, K: int) -> Dict[TypeKey, int]:
    """
    Process a chunk of samples and return type counts.
    This function is defined at module level to work with multiprocessing.
    """
    chunk_counts = defaultdict(int)
    for sample in chunk:
        type_tuple = tuple(1 if sample[i] > thresholds_array[i] else 0 for i in range(K))
        chunk_counts[type_tuple] += 1
    return dict(chunk_counts)


def build_attribute_order(relative_frequencies: Mapping[AttributeId, float]) -> List[AttributeId]:
    """Build ordered list of attributes sorted by relative frequency (highest first)."""
    return sorted(relative_frequencies.keys(), key=lambda x: relative_frequencies[x], reverse=True)


def compute_constraint_sets(
    attribute_order: List[AttributeId],
    constraints_min_count: Mapping[AttributeId, int],
    N: int
) -> Dict[AttributeId, Tuple[float, Set[TypeKey]]]:
    """
    Compute constraint sets for each attribute.

    Returns a dict mapping attribute -> (alpha, set_of_types_that_satisfy_this_constraint)
    where alpha = min_count / N
    """
    constraint_sets = {}

    for attr, min_count in constraints_min_count.items():
        alpha = min_count / N
        attr_index = attribute_order.index(attr)
        satisfying_types = set()

        # Generate all possible type combinations (2^K where K=len(attribute_order))
        for type_bits in range(2 ** len(attribute_order)):
            type_tuple = tuple((type_bits >> i) & 1 for i in range(len(attribute_order)))
            # Check if this type satisfies the constraint (has the attribute)
            if type_tuple[attr_index] == 1:
                satisfying_types.add(type_tuple)

        constraint_sets[attr] = (alpha, satisfying_types)

    return constraint_sets


def correlate_binary_joint(
    attribute_order: List[AttributeId],
    relative_frequencies: Mapping[AttributeId, float],
    correlations: Mapping[AttributeId, Mapping[AttributeId, float]],
    num_samples: int = 150000
) -> Dict[TypeKey, float]:
    """
    Approximate joint type probabilities using correlations.

    Uses Gaussian copula to generate correlated binary variables.
    """
    K = len(attribute_order)
    if K == 0:
        return {}

    # Build correlation matrix
    corr_matrix = np.eye(K)
    for i, attr_i in enumerate(attribute_order):
        for j, attr_j in enumerate(attribute_order):
            if i != j and attr_j in correlations.get(attr_i, {}):
                corr_matrix[i, j] = correlations[attr_i][attr_j]
                corr_matrix[j, i] = correlations[attr_i][attr_j]  # Ensure symmetry

    # Ensure positive semi-definite
    eigenvalues = np.linalg.eigvals(corr_matrix)
    if np.any(eigenvalues < -1e-10):
        # Adjust negative eigenvalues
        corr_matrix = corr_matrix + np.eye(K) * (1e-10 - np.min(eigenvalues))

    # Generate samples using multivariate normal
    mean = np.zeros(K)
    np.random.seed(42)  # Fixed seed for reproducible results
    samples = np.random.multivariate_normal(mean, corr_matrix, num_samples)

    # Convert to binary using marginal frequencies
    thresholds = {}
    for i, attr in enumerate(attribute_order):
        freq = relative_frequencies[attr]
        # For correlated Gaussians, threshold at the inverse CDF of the frequency
        thresholds[i] = np.sqrt(2) * erfinv(2 * freq - 1)

    # Generate binary types and count frequencies
    type_counts = defaultdict(int)
    for sample in samples:
        type_tuple = tuple(1 if sample[i] > thresholds[i] else 0 for i in range(K))
        type_counts[type_tuple] += 1

    # Convert to probabilities
    type_probs = {t: count / num_samples for t, count in type_counts.items()}

    # Ensure we have all possible types (even if not sampled)
    all_types = [tuple((bits >> i) & 1 for i in range(K)) for bits in range(2**K)]
    for t in all_types:
        if t not in type_probs:
            type_probs[t] = 1e-10  # Small probability for unobserved types

    return type_probs


def correlate_binary_joint_parallel(
    attribute_order: List[AttributeId],
    relative_frequencies: Mapping[AttributeId, float],
    correlations: Mapping[AttributeId, Mapping[AttributeId, float]],
    num_samples: int = 150000
) -> Dict[TypeKey, float]:
    """
    Approximate joint type probabilities using correlations with parallel processing.

    Uses Gaussian copula to generate correlated binary variables.
    """
    K = len(attribute_order)
    if K == 0:
        return {}

    # Build correlation matrix
    corr_matrix = np.eye(K)
    for i, attr_i in enumerate(attribute_order):
        for j, attr_j in enumerate(attribute_order):
            if i != j and attr_j in correlations.get(attr_i, {}):
                corr_matrix[i, j] = correlations[attr_i][attr_j]
                corr_matrix[j, i] = correlations[attr_i][attr_j]  # Ensure symmetry

    # Ensure positive semi-definite
    eigenvalues = np.linalg.eigvals(corr_matrix)
    if np.any(eigenvalues < -1e-10):
        # Adjust negative eigenvalues
        corr_matrix = corr_matrix + np.eye(K) * (1e-10 - np.min(eigenvalues))

    # Get CPU count for optimal parallelization
    num_workers = min(mp.cpu_count(), 8)  # Cap at 8 to avoid overhead
    samples_per_worker = num_samples // num_workers

    # Prepare arguments for parallel processing
    thresholds = [np.sqrt(2) * erfinv(2 * relative_frequencies[attr] - 1)
                  for attr in attribute_order]
    thresholds_array = np.array(thresholds)

    # Generate all samples at once (faster than per-worker generation)
    mean = np.zeros(K)
    np.random.seed(42)  # Fixed seed for reproducible results
    all_samples = np.random.multivariate_normal(mean, corr_matrix, num_samples)

    # Split samples across workers
    sample_chunks = np.array_split(all_samples, num_workers)

    # Process chunks in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_samples_chunk, chunk, thresholds_array, K) for chunk in sample_chunks]

        # Combine results from all workers
        combined_counts = defaultdict(int)
        for future in as_completed(futures):
            chunk_counts = future.result()
            for type_tuple, count in chunk_counts.items():
                combined_counts[type_tuple] += count

    # Convert to probabilities
    type_probs = {t: count / num_samples for t, count in combined_counts.items()}

    # Ensure we have all possible types (even if not sampled)
    all_types = [tuple((bits >> i) & 1 for i in range(K)) for bits in range(2**K)]
    for t in all_types:
        if t not in type_probs:
            type_probs[t] = 1e-10  # Small probability for unobserved types

    return type_probs


def correlate_binary_joint_vectorized_parallel(
    attribute_order: List[AttributeId],
    relative_frequencies: Mapping[AttributeId, float],
    correlations: Mapping[AttributeId, Mapping[AttributeId, float]],
    num_samples: int = 150000
) -> Dict[TypeKey, float]:
    """
    Fully vectorized parallel implementation for maximum speed.
    """
    K = len(attribute_order)
    if K == 0:
        return {}

    # Build correlation matrix
    corr_matrix = np.eye(K)
    for i, attr_i in enumerate(attribute_order):
        for j, attr_j in enumerate(attribute_order):
            if i != j and attr_j in correlations.get(attr_i, {}):
                corr_matrix[i, j] = correlations[attr_i][attr_j]
                corr_matrix[j, i] = correlations[attr_i][attr_j]

    # Ensure positive semi-definite
    eigenvalues = np.linalg.eigvals(corr_matrix)
    if np.any(eigenvalues < -1e-10):
        corr_matrix = corr_matrix + np.eye(K) * (1e-10 - np.min(eigenvalues))

    # Generate all samples at once
    mean = np.zeros(K)
    np.random.seed(42)  # Fixed seed for reproducible results
    samples = np.random.multivariate_normal(mean, corr_matrix, num_samples)

    # Vectorized threshold calculation
    thresholds_array = np.array([np.sqrt(2) * erfinv(2 * relative_frequencies[attr] - 1)
                                for attr in attribute_order])

    # Vectorized binary conversion (much faster!)
    binary_samples = (samples > thresholds_array).astype(int)

    # Use numpy's unique for counting (extremely fast)
    unique_types, counts = np.unique(binary_samples, axis=0, return_counts=True)

    # Convert to dictionary format
    type_counts = {tuple(row): count for row, count in zip(unique_types, counts)}

    # Convert to probabilities
    type_probs = {t: count / num_samples for t, count in type_counts.items()}

    # Ensure we have all possible types
    all_types = [tuple((bits >> i) & 1 for i in range(K)) for bits in range(2**K)]
    for t in all_types:
        if t not in type_probs:
            type_probs[t] = 1e-10

    return type_probs


def multiplicative_weights_lp(
    type_probs: Dict[TypeKey, float],
    constraint_sets: Dict[AttributeId, Tuple[float, Set[TypeKey]]],
    iterations: int = 450,
    eta: float = 0.1
) -> Dict[TypeKey, float]:
    """
    Solve the LP using multiplicative weights method.

    Maximizes sum_t p_t * a_t subject to constraints and 0 <= a_t <= 1.
    """
    # Initialize acceptance probabilities
    a = {t: 1.0 for t in type_probs.keys()}

    # Adjust eta based on number of constraints
    K = len(constraint_sets)
    eta = min(eta, 0.5 / K) if K > 0 else eta

    for _ in range(iterations):
        # Compute slack for each constraint
        slacks = {}
        for attr, (alpha, satisfying_types) in constraint_sets.items():
            slack = 0.0
            for t, p_t in type_probs.items():
                indicator = 1.0 if t in satisfying_types else 0.0
                slack += (indicator - alpha) * p_t * a[t]
            slacks[attr] = slack

        # Check if all constraints are satisfied
        if all(s >= 0 for s in slacks.values()):
            break

        # Update acceptance probabilities for violated constraints
        for t in a.keys():
            update_factor = 1.0
            for attr, (alpha, satisfying_types) in constraint_sets.items():
                if slacks[attr] < 0:  # Violated constraint
                    if t not in satisfying_types:  # Type doesn't satisfy this constraint
                        update_factor *= math.exp(-eta * type_probs[t])

            a[t] *= update_factor

            # Clip to [0, 1]
            a[t] = max(0.0, min(1.0, a[t]))

    return a


def test_parallel_functionality():
    """
    Test function to verify that parallel implementations produce identical results.
    """
    import time

    # Mock data for testing
    attribute_order = ["attr_A", "attr_B", "attr_C"]
    relative_frequencies = {
        "attr_A": 0.6,
        "attr_B": 0.4,
        "attr_C": 0.2
    }
    correlations = {
        "attr_A": {"attr_B": 0.3, "attr_C": 0.1},
        "attr_B": {"attr_A": 0.3, "attr_C": 0.2},
        "attr_C": {"attr_A": 0.1, "attr_B": 0.2}
    }

    num_samples = 10000  # Smaller sample size for faster testing

    print("Testing parallel functionality with mock data...")
    print(f"Attributes: {attribute_order}")
    print(f"Sample size: {num_samples}")

    # Test original function
    start_time = time.time()
    result_original = correlate_binary_joint(
        attribute_order, relative_frequencies, correlations, num_samples
    )
    original_time = time.time() - start_time

    # Test parallel function
    start_time = time.time()
    result_parallel = correlate_binary_joint_parallel(
        attribute_order, relative_frequencies, correlations, num_samples
    )
    parallel_time = time.time() - start_time

    # Test vectorized parallel function
    start_time = time.time()
    result_vectorized = correlate_binary_joint_vectorized_parallel(
        attribute_order, relative_frequencies, correlations, num_samples
    )
    vectorized_time = time.time() - start_time

    # Verify results are identical (within numerical precision)
    tolerance = 1e-10
    differences_found = 0

    for key in result_original:
        if key not in result_parallel or key not in result_vectorized:
            print(f"❌ Key {key} missing in parallel results")
            differences_found += 1
            continue

        orig_val = result_original[key]
        par_val = result_parallel[key]
        vec_val = result_vectorized[key]

        if abs(orig_val - par_val) > tolerance:
            print(f"❌ Parallel result differs for {key}: {orig_val} vs {par_val}")
            differences_found += 1

        if abs(orig_val - vec_val) > tolerance:
            print(f"❌ Vectorized result differs for {key}: {orig_val} vs {vec_val}")
            differences_found += 1

    # Performance comparison
    print("\n📊 Performance Results:")
    print(".2f")
    print(".2f")
    print(".2f")
    print(".2f")
    print(".2f")
    # Success/failure summary
    if differences_found == 0:
        print("✅ All results are identical within numerical precision!")
        print("✅ Parallel implementations are functionally equivalent to original.")
    else:
        print(f"❌ Found {differences_found} differences between implementations.")

    return differences_found == 0


def attributes_to_type_key(attributes: Mapping[AttributeId, bool], attribute_order: List[AttributeId]) -> TypeKey:
    """Convert attribute mapping to type tuple."""
    return tuple(1 if attributes.get(attr, False) else 0 for attr in attribute_order)


def make_online_policy(
    N: int,
    attribute_order: List[AttributeId],
    base_accept: Dict[TypeKey, float],
    constraint_sets: Dict[AttributeId, Tuple[float, Set[TypeKey]]],
    lambda_nudge: float = 0.5,
    safety_beta: float = 1.0,
) -> Tuple[Callable[[Mapping[AttributeId, bool]], Tuple[bool, Dict]], Dict]:
    """
    Create the online policy function.

    Returns (step_function, initial_policy_state)
    """
    # Initialize state
    policy_state = {
        'n': 0,  # Total accepted
        'c': {attr: 0 for attr in constraint_sets.keys()},  # Count per constraint
        'rejected': 0,
    }

    def attributes_to_type(attributes: Mapping[AttributeId, bool]) -> TypeKey:
        """Convert attribute mapping to type tuple."""
        return attributes_to_type_key(attributes, attribute_order)

    def step(attributes: Mapping[AttributeId, bool]) -> Tuple[bool, Dict]:
        """Process one arrival and return (accept_decision, updated_state)."""
        nonlocal policy_state

        t = attributes_to_type(attributes)
        R = N - policy_state['n']  # Remaining capacity
        r = {}  # Remaining required for each constraint

        for attr, (alpha, _) in constraint_sets.items():
            required = max(0, math.ceil(alpha * N) - policy_state['c'][attr])
            r[attr] = required

        # If all minimum requirements are already satisfied, accept everyone to fill remaining capacity
        if sum(r.values()) == 0 and R > 0:
            policy_state['n'] += 1
            for attr in constraint_sets.keys():
                if attributes.get(attr, False):
                    policy_state['c'][attr] += 1
            return True, policy_state

        # Endgame hard guard
        total_required = sum(r.values())
        # Only trigger forced mode when remaining requirements exceed remaining capacity
        # or when a single requirement consumes all remaining capacity
        forced_mode = (total_required > R) or any(rj == R for rj in r.values())

        if forced_mode:
            # In forced mode, only accept if they can help satisfy constraints that still need people
            # Check which needed constraints this person can actually help with
            can_help_needed = []
            for attr in constraint_sets.keys():
                if r[attr] > 0 and attributes.get(attr, False):
                    can_help_needed.append(attr)

            if can_help_needed:
                # Accept this person - they can help with at least one needed constraint
                policy_state['n'] += 1
                # Only increment counters for constraints that still need more people
                for attr in constraint_sets.keys():
                    if attributes.get(attr, False) and r[attr] > 0:
                        policy_state['c'][attr] += 1
                return True, policy_state
            else:
                # Reject - this person cannot help with any remaining constraints
                policy_state['rejected'] += 1
                return False, policy_state

        # Deficit-aware scoring
        use_buffer = policy_state['n'] < 0.8 * N
        w = {}

        for attr, (alpha, _) in constraint_sets.items():
            buffer_alpha = alpha + safety_beta / math.sqrt(N) if use_buffer else alpha
            deficit = max(0.0, buffer_alpha - (policy_state['c'][attr] / max(1, policy_state['n'])))
            w[attr] = deficit

        # Compute score for this type
        s = sum(w[attr] for attr in constraint_sets.keys()
                if attributes.get(attr, False) is True)

        # Probabilistic admission with nudge
        base_prob = base_accept.get(t, 0.0)
        pi = min(1.0, base_prob * (1.0 + lambda_nudge * s))

        # Make decision
        if random.random() < pi:
            policy_state['n'] += 1
            for attr in constraint_sets.keys():
                if attributes.get(attr, False):
                    policy_state['c'][attr] += 1
            return True, policy_state
        else:
            policy_state['rejected'] += 1
            return False, policy_state

    return step, policy_state
