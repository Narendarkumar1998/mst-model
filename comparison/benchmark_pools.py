"""
Candidate pools for the comparison with adapted baselines.

The admissible-item attributes are generated with the marginal-preserving
generator (version "marginal_preserving_v2") used for the paper's primary
benchmark, with the same seeds, so every complete candidate pool I_t is
identical to the pool of the corresponding primary-benchmark instance.

The only addition is a hard-admissibility flag. In each matched size and
replicate group, ceil(rho * n) item positions are marked as prohibited by the
disclosure policy, chosen uniformly at random with a seed derived from the
matched seed. The same positions are prohibited in the three structural
regimes, so the regimes stay matched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

GLOBAL_SEED = 2026
REGIME_STRENGTH = 0.75
EXACT_BENCHMARK_SIZES = [8, 10, 12, 14, 16]
TAU_LEVELS = [0.70, 0.80, 0.90, 0.95]
REPLICATES = 100
INSTANCE_REGIMES = [
    "independent",
    "utility_burden_conflict",
    "utility_burden_alignment",
]
WEIGHT_PROFILES = {
    "balanced": (1 / 3, 1 / 3, 1 / 3),
    "privacy_priority": (0.60, 0.20, 0.20),
    "interpretation_priority": (0.20, 0.60, 0.20),
    "parsimony_priority": (0.20, 0.20, 0.60),
}
PROHIBITED_SHARE = 0.25
ADMISSIBILITY_SEED_OFFSET = 777


@dataclass(frozen=True)
class Pool:
    ids: tuple
    utility: np.ndarray
    privacy: np.ndarray
    cost: np.ndarray
    admissible: np.ndarray


def matched_seed(size_index: int, replicate: int) -> int:
    """Seed formula of the primary benchmark (notebook 02, corrected v2)."""
    return GLOBAL_SEED + 50_000_000 + size_index * 1_000_000 + replicate


def _assign_burden_preserving_marginal(base_values, utility_values, regime, rng, regime_strength):
    """Verbatim logic of the primary-benchmark generator (notebook 02, cell 50)."""
    base_values = np.asarray(base_values, dtype=float)
    utility_values = np.asarray(utility_values, dtype=float)
    number_of_items = len(base_values)
    if number_of_items <= 1:
        return base_values.copy()
    if regime == "independent":
        return base_values.copy()
    utility_order = np.argsort(utility_values, kind="mergesort")
    utility_ranks = np.empty(number_of_items, dtype=float)
    utility_ranks[utility_order] = np.arange(number_of_items, dtype=float)
    utility_rank_std = utility_ranks.std(ddof=0)
    if utility_rank_std > 0:
        standardized = (utility_ranks - utility_ranks.mean()) / utility_rank_std
    else:
        standardized = np.zeros(number_of_items, dtype=float)
    independent_noise = rng.normal(loc=0.0, scale=1.0, size=number_of_items)
    residual_strength = np.sqrt(max(0.0, 1.0 - regime_strength ** 2))
    if regime == "utility_burden_conflict":
        direction = 1.0
    elif regime == "utility_burden_alignment":
        direction = -1.0
    else:
        raise ValueError(f"Unknown regime: {regime}")
    ordering_score = direction * regime_strength * standardized + residual_strength * independent_noise
    score_order = np.argsort(ordering_score, kind="mergesort")
    sorted_burden_values = np.sort(base_values)
    assigned_values = np.empty_like(base_values, dtype=float)
    assigned_values[score_order] = sorted_burden_values
    return assigned_values


def generate_items(n_items: int, regime: str, seed: int, regime_strength: float = REGIME_STRENGTH):
    """Return (ids, utility, privacy, cost) exactly as the primary-benchmark generator."""
    rng = np.random.default_rng(seed)
    utility_latent = rng.beta(a=2.0, b=2.0, size=n_items)
    utility_values = 0.08 + 0.52 * utility_latent
    base_privacy_latent = rng.beta(a=2.0, b=2.0, size=n_items)
    base_cognitive_latent = rng.beta(a=2.0, b=2.0, size=n_items)
    privacy_latent = _assign_burden_preserving_marginal(
        base_privacy_latent, utility_latent, regime, rng, regime_strength)
    cognitive_latent = _assign_burden_preserving_marginal(
        base_cognitive_latent, utility_latent, regime, rng, regime_strength)
    privacy_values = 0.01 + 0.24 * privacy_latent
    cognitive_values = 0.10 + 0.90 * cognitive_latent
    ids = tuple(f"z{i + 1:03d}" for i in range(n_items))
    return (ids,
            np.array([float(v) for v in utility_values]),
            np.array([float(v) for v in privacy_values]),
            np.array([float(v) for v in cognitive_values]))


def admissibility_flags(n_items: int, seed: int, share: float = PROHIBITED_SHARE) -> np.ndarray:
    """Mark ceil(share * n) item positions as prohibited; True means admissible."""
    flags = np.ones(n_items, dtype=bool)
    k = math.ceil(share * n_items) if share > 0 else 0
    if k:
        rng = np.random.default_rng(seed + ADMISSIBILITY_SEED_OFFSET)
        flags[rng.choice(n_items, size=k, replace=False)] = False
    return flags


def make_pool(n_items: int, regime: str, seed: int, share: float = PROHIBITED_SHARE) -> Pool:
    ids, u, p, c = generate_items(n_items, regime, seed)
    return Pool(ids, u, p, c, admissibility_flags(n_items, seed, share))
