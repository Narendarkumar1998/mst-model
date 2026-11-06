"""Minimum Sufficient Transparency (MST): authoritative Python implementation."""
from .mst_core import (
    InformationItem,
    mst_utility,
    mst_privacy,
    mst_cognitive_cost,
    mst_normalized_cognitive_cost,
    mst_disclosure_cost,
    evaluate_subset,
    exact_mst,
    greedy_mst,
    progressive_mst_step,
    progressive_mst_sequence,
    compute_optimality_gap,
)

__all__ = [
    "InformationItem", "mst_utility", "mst_privacy", "mst_cognitive_cost",
    "mst_normalized_cognitive_cost", "mst_disclosure_cost", "evaluate_subset",
    "exact_mst", "greedy_mst", "progressive_mst_step",
    "progressive_mst_sequence", "compute_optimality_gap",
]
