from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Callable, Dict, List, Mapping, Set, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
from scipy.special import erfinv
from scipy.optimize import linprog

from .models import AttributeId, Constraint, TypeKey


def process_samples_chunk(chunk: np.ndarray, thresholds_array: np.ndarray, K: int) -> Dict[TypeKey, int]:
    """
    Process a chunk of samples and return type counts.
    This function is defined at module level to work with multiprocessing.
    """
    chunk_counts = defaultdict(int)
    for sample in chunk:
        type_tuple = tuple(1 if sample[i] <= thresholds_array[i] else 0 for i in range(K))
        chunk_counts[type_tuple] += 1
    return dict(chunk_counts)


def build_attribute_order(relative_frequencies: Mapping[AttributeId, float]) -> List[AttributeId]:
    """Build ordered list of attributes sorted by relative frequency (highest first)."""
    return sorted(relative_frequencies.keys(), key=lambda x: relative_frequencies[x], reverse=True)


def build_attribute_order_constrained(
    relative_frequencies: Mapping[AttributeId, float],
    constraints_min_count: Mapping[AttributeId, int],
) -> List[AttributeId]:
    """
    Return ONLY the constrained attributes, sorted for determinism (e.g., by freq desc).
    """
    constrained_attrs = list(constraints_min_count.keys())
    return sorted(constrained_attrs, key=lambda a: relative_frequencies[a], reverse=True)


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
        type_tuple = tuple(1 if sample[i] <= thresholds[i] else 0 for i in range(K))
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
    binary_samples = (samples <= thresholds_array).astype(int)

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


def two_attr_joint(qA: float, qB: float, rho_binary: float) -> Dict[TypeKey, float]:
    varA = qA * (1 - qA)
    varB = qB * (1 - qB)
    if varA == 0 or varB == 0:
        # Degenerate marginals
        p11 = 1.0 if (qA == 1.0 and qB == 1.0) else 0.0
        p10 = 1.0 if (qA == 1.0 and qB == 0.0) else 0.0
        p01 = 1.0 if (qA == 0.0 and qB == 1.0) else 0.0
        p00 = 1.0 if (qA == 0.0 and qB == 0.0) else 0.0
        return {(0,0): p00, (0,1): p01, (1,0): p10, (1,1): p11}

    cov = rho_binary * math.sqrt(varA * varB)
    p11 = qA * qB + cov
    lower = max(0.0, qA + qB - 1.0)
    upper = min(qA, qB)
    p11 = min(max(p11, lower), upper)

    p10 = qA - p11
    p01 = qB - p11
    p00 = 1.0 - qA - qB + p11
    return {(0,0): p00, (0,1): p01, (1,0): p10, (1,1): p11}


def solve_lp_primal(
    type_probs: Dict[TypeKey, float],
    constraint_sets: Dict[AttributeId, Tuple[float, Set[TypeKey]]],
):
    """
    Solve the primal LP using SciPy HiGHS solver.

    maximize Σ_t p_t r_t
    s.t.     Σ_t p_t r_t (1[t∈S_j] - α_j) ≥ 0  for each j
             0 ≤ r_t ≤ 1

    Returns:
      r_by_type: Dict[TypeKey, float]
      A_rate   : float  (#accepted per arrival, i.e., expected accept rate)
      lambdas  : Dict[AttributeId, float]  (optional, for logging)
    """
    types = list(type_probs.keys())
    p = np.array([type_probs[t] for t in types], dtype=float)
    T = len(types)
    attrs = list(constraint_sets.keys())
    m = len(attrs)

    # ≥ to ≤ : multiply by -1
    A_ub = np.zeros((m, T))
    b_ub = np.zeros(m)
    for j, a in enumerate(attrs):
        alpha, S = constraint_sets[a]
        for i, t in enumerate(types):
            ind = 1.0 if t in S else 0.0
            A_ub[j, i] = - p[i] * (ind - alpha)   # ≤ 0
    c = -p
    bounds = [(0.0, 1.0)] * T

    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"LP infeasible or failed: {res.message}")

    r = np.clip(res.x, 0.0, 1.0)
    A_rate = float((p * r).sum())

    # optional: duals for logging
    lambdas = {}
    try:
        lam = np.array(res.ineqlin.marginals, dtype=float)
        lam[lam < 0] = 0.0
        for j, a in enumerate(attrs):
            lambdas[a] = float(lam[j])
    except Exception:
        lambdas = {a: 0.0 for a in attrs}

    r_by_type = {types[i]: float(r[i]) for i in range(T)}
    return r_by_type, A_rate, lambdas


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
    enable_neither_wiggle: bool = False,
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
        'both_young_wd': 0,  # Track how many admitted are both young and well_dressed
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
        # Lock as soon as the needs fill the remaining slots
        forced_mode = (total_required >= R) or any(rj == R for rj in r.values())

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
                # Always increment all attributes the person actually has
                for attr in constraint_sets.keys():
                    if attributes.get(attr, False):
                        policy_state['c'][attr] += 1
                # Update both counter if applicable
                if enable_neither_wiggle and ('young' in attribute_order and 'well_dressed' in attribute_order):
                    if attributes.get('young', False) and attributes.get('well_dressed', False):
                        policy_state['both_young_wd'] += 1
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

        # Special handling: if this person is neither young nor well_dressed, optionally accept with alpha
        if enable_neither_wiggle and ('young' in attribute_order and 'well_dressed' in attribute_order):
            is_young = bool(attributes.get('young', False))
            is_wd = bool(attributes.get('well_dressed', False))
            is_neither = (not is_young) and (not is_wd)
            if is_neither and policy_state['n'] > 0:
                both_frac = policy_state['both_young_wd'] / max(1, policy_state['n'])
                alpha = max(both_frac - (200.0 / float(N)), 0.0)
                # Ensure alpha within [0,1]
                alpha = min(alpha, 1.0)
                # Elevate probability to at least alpha
                pi = max(pi, alpha)

        # Make decision
        if random.random() < pi:
            policy_state['n'] += 1
            for attr in constraint_sets.keys():
                if attributes.get(attr, False):
                    policy_state['c'][attr] += 1
            # Update both counter if applicable
            if enable_neither_wiggle and ('young' in attribute_order and 'well_dressed' in attribute_order):
                if attributes.get('young', False) and attributes.get('well_dressed', False):
                    policy_state['both_young_wd'] += 1
            return True, policy_state
        else:
            policy_state['rejected'] += 1
            return False, policy_state

    return step, policy_state


def make_online_policy_from_primal(
    N: int,
    attribute_order: List[AttributeId],
    r_by_type: Dict[TypeKey, float],
    constraint_sets: Dict[AttributeId, Tuple[float, Set[TypeKey]]],
    rng: random.Random,
    enable_neither_wiggle: bool = False,
    reserve_trigger_n: Optional[int] = None,
):
    """
    Create online policy using primal LP rates r_t.

    Returns (step_function, initial_policy_state)
    """
    state = {'n': 0, 'rejected': 0, 'c': {a: 0 for a in constraint_sets}, 'both_young_wd': 0}

    def to_type_key(attrs: Mapping[AttributeId, bool]) -> TypeKey:
        return tuple(1 if attrs.get(a, False) else 0 for a in attribute_order)

    def step(attrs: Mapping[AttributeId, bool]):
        nonlocal state
        t = to_type_key(attrs)

        # --- endgame lock (use the fixed version from change #2) ---
        R = N - state['n']
        r_need = {a: max(0, math.ceil(alpha * N) - state['c'][a]) for a, (alpha, _) in constraint_sets.items()}
        forced_mode = (sum(r_need.values()) >= R) or any(v == R for v in r_need.values())
        # Engage reserve mode early once we have spent the budget implied by A_max
        if reserve_trigger_n is not None and state['n'] >= reserve_trigger_n:
            forced_mode = True

        if forced_mode:
            helps = any(r_need[a] > 0 and attrs.get(a, False) for a in constraint_sets)
            if helps:
                state['n'] += 1
                for a in constraint_sets:
                    if attrs.get(a, False): state['c'][a] += 1
                return True, state
            else:
                state['rejected'] += 1
                return False, state

        # --- stationary thinning from the LP ---
        rt = r_by_type.get(t, 0.0)
        pi = rt
        # Special handling: if this person is neither young nor well_dressed, optionally accept with alpha
        if enable_neither_wiggle and ('young' in attribute_order and 'well_dressed' in attribute_order):
            is_young = bool(attrs.get('young', False))
            is_wd = bool(attrs.get('well_dressed', False))
            is_neither = (not is_young) and (not is_wd)
            if is_neither and state['n'] > 0:
                both_frac = state['both_young_wd'] / max(1, state['n'])
                alpha = max(both_frac - (200.0 / float(N)), 0.0)
                alpha = min(alpha, 1.0)
                pi = max(pi, alpha)

        if pi >= 1.0 - 1e-9:
            accept = True
        elif pi <= 1e-9:
            accept = False
        else:
            accept = (rng.random() < pi)

        if accept:
            state['n'] += 1
            for a in constraint_sets:
                if attrs.get(a, False): state['c'][a] += 1
            if enable_neither_wiggle and ('young' in attribute_order and 'well_dressed' in attribute_order):
                if attrs.get('young', False) and attrs.get('well_dressed', False):
                    state['both_young_wd'] += 1
            return True, state
        else:
            state['rejected'] += 1
            return False, state

    return step, state
