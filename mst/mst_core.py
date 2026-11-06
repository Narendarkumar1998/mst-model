from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations


@dataclass(frozen=True)
class InformationItem:
    """
    A privacy-admissible information item considered by MST.

    Parameters
    ----------
    item_id : str
        Unique identifier of the information item.
    name : str
        Human-readable description.
    utility : float
        Modeled decision-support contribution in [0, 1].
    privacy : float
        Modeled privacy exposure in [0, 1].
    cognitive_cost : float
        Modeled interpretation cost, constrained to be non-negative.
    """

    item_id: str
    name: str
    utility: float
    privacy: float
    cognitive_cost: float

    def __post_init__(self) -> None:

        if not self.item_id:
            raise ValueError(
                "item_id must be a non-empty string."
            )

        if not 0.0 <= self.utility <= 1.0:
            raise ValueError(
                f"utility must be in [0, 1], "
                f"received {self.utility}"
            )

        if not 0.0 <= self.privacy <= 1.0:
            raise ValueError(
                f"privacy must be in [0, 1], "
                f"received {self.privacy}"
            )

        if self.cognitive_cost < 0.0:
            raise ValueError(
                "cognitive_cost must be non-negative."
            )


def _validate_unique_ids(
    items: Iterable[InformationItem],
) -> None:
    """
    Ensure that information identifiers are unique.
    """

    item_list = list(items)

    ids = [
        item.item_id
        for item in item_list
    ]

    if len(ids) != len(set(ids)):
        raise ValueError(
            "All information items must have unique item_id values."
        )


def mst_utility(
    items: Iterable[InformationItem],
) -> float:
    """
    Bounded cumulative modeled decision-support utility.

    U(S) = 1 - product(1 - u(z))
    """

    product = 1.0

    for item in items:
        product *= (
            1.0 - item.utility
        )

    return 1.0 - product


def mst_privacy(
    items: Iterable[InformationItem],
) -> float:
    """
    Bounded cumulative privacy exposure.

    P(S) = 1 - product(1 - p(z))
    """

    product = 1.0

    for item in items:
        product *= (
            1.0 - item.privacy
        )

    return 1.0 - product


def mst_cognitive_cost(
    items: Iterable[InformationItem],
) -> float:
    """
    Additive modeled interpretation cost.

    C(S) = sum c(z)
    """

    return sum(
        item.cognitive_cost
        for item in items
    )


# Reference scales of the revised model (paper, Eqs. 11-12): when set, interpretation
# cost and cardinality are divided by fixed references C_ref and K_ref instead of the
# totals of the pool passed as all_items. The benchmarks of the revised article use
# C_ref = 5.5 and K_ref = 10 (optimization/scales.py).
REFERENCE_SCALES = None          # None, or (C_ref, K_ref)


def set_reference_scales(cost_reference=None, cardinality_reference=None):
    """Use fixed reference scales (both values) or the pool totals (no arguments)."""
    global REFERENCE_SCALES
    REFERENCE_SCALES = None if cost_reference is None else (float(cost_reference), float(cardinality_reference))


def mst_normalized_cognitive_cost(
    selected_items: Iterable[InformationItem],
    all_items: Iterable[InformationItem],
) -> float:
    """
    Normalized modeled interpretation cost.

    C_bar(S) = C(S) / C(D), if C(D) > 0,
               0 otherwise.
    """

    selected = list(selected_items)
    admissible = list(all_items)

    selected_cost = (
        mst_cognitive_cost(selected)
    )

    total_cost = (
        mst_cognitive_cost(admissible)
        if REFERENCE_SCALES is None
        else REFERENCE_SCALES[0]
    )

    if total_cost == 0.0:
        return 0.0

    return (
        selected_cost
        / total_cost
    )


def mst_disclosure_cost(
    selected_items: Iterable[InformationItem],
    all_items: Iterable[InformationItem],
    lambda_privacy: float = 1.0,
    lambda_cognitive: float = 1.0,
    lambda_cardinality: float = 0.01,
) -> float:
    """
    Official normalized MST disclosure objective.

    J(S) =
        lambda_privacy * P(S)
        + lambda_cognitive * C_bar(S)
        + lambda_cardinality * |S| / |D|
    """

    selected = list(
        selected_items
    )

    admissible = list(
        all_items
    )

    total_items = len(
        admissible
    )

    if total_items == 0:
        raise ValueError(
            "all_items must contain at least one "
            "admissible information item."
        )

    if any(
        weight < 0.0
        for weight in (
            lambda_privacy,
            lambda_cognitive,
            lambda_cardinality,
        )
    ):
        raise ValueError(
            "All lambda weights must be non-negative."
        )

    privacy_component = (
        mst_privacy(selected)
    )

    cognitive_component = (
        mst_normalized_cognitive_cost(
            selected_items=selected,
            all_items=admissible,
        )
    )

    cardinality_component = (
        len(selected)
        / (total_items if REFERENCE_SCALES is None else REFERENCE_SCALES[1])
    )

    return (
        lambda_privacy
        * privacy_component
        +
        lambda_cognitive
        * cognitive_component
        +
        lambda_cardinality
        * cardinality_component
    )


def evaluate_subset(
    subset: Iterable[InformationItem],
    all_items: Iterable[InformationItem],
    lambda_privacy: float = 1.0,
    lambda_cognitive: float = 1.0,
    lambda_cardinality: float = 0.01,
) -> dict:
    """
    Evaluate one disclosure subset under the official
    normalized MST formulation.
    """

    selected = list(
        subset
    )

    admissible = list(
        all_items
    )

    return {
        "items": tuple(
            item.item_id
            for item in selected
        ),
        "utility":
            mst_utility(selected),
        "privacy":
            mst_privacy(selected),
        "cognitive_cost":
            mst_cognitive_cost(selected),
        "normalized_cognitive_cost":
            mst_normalized_cognitive_cost(
                selected_items=selected,
                all_items=admissible,
            ),
        "disclosure_cost":
            mst_disclosure_cost(
                selected_items=selected,
                all_items=admissible,
                lambda_privacy=lambda_privacy,
                lambda_cognitive=lambda_cognitive,
                lambda_cardinality=lambda_cardinality,
            ),
    }


def exact_mst(
    items: list[InformationItem],
    tau: float,
    lambda_privacy: float = 1.0,
    lambda_cognitive: float = 1.0,
    lambda_cardinality: float = 0.01,
) -> dict:
    """
    Solve MST exactly by exhaustive subset enumeration.

    Returns the globally minimum-cost disclosure subset
    satisfying U(S) >= tau.
    """

    if not 0.0 < tau <= 1.0:
        raise ValueError(
            "tau must be in the interval (0, 1]."
        )

    _validate_unique_ids(
        items
    )

    total_items = len(
        items
    )

    if total_items == 0:
        return {
            "status": "INFEASIBLE",
            "selected_items": tuple(),
            "utility": 0.0,
            "privacy": 0.0,
            "cognitive_cost": 0.0,
            "normalized_cognitive_cost": 0.0,
            "disclosure_cost": float("inf"),
        }

    if mst_utility(items) < tau:
        return {
            "status": "INFEASIBLE",
            "selected_items": tuple(),
            "utility":
                mst_utility(items),
            "privacy":
                mst_privacy(items),
            "cognitive_cost":
                mst_cognitive_cost(items),
            "normalized_cognitive_cost":
                mst_normalized_cognitive_cost(
                    selected_items=items,
                    all_items=items,
                ),
            "disclosure_cost":
                float("inf"),
        }

    best_result = None

    for subset_size in range(
        total_items + 1
    ):

        for subset in combinations(
            items,
            subset_size,
        ):

            result = evaluate_subset(
                subset=subset,
                all_items=items,
                lambda_privacy=lambda_privacy,
                lambda_cognitive=lambda_cognitive,
                lambda_cardinality=lambda_cardinality,
            )

            if (
                result["utility"]
                < tau
            ):
                continue

            if best_result is None:
                best_result = result
                continue

            current_key = (
                result["disclosure_cost"],
                result["privacy"],
                result[
                    "normalized_cognitive_cost"
                ],
                len(result["items"]),
                result["items"],
            )

            best_key = (
                best_result[
                    "disclosure_cost"
                ],
                best_result[
                    "privacy"
                ],
                best_result[
                    "normalized_cognitive_cost"
                ],
                len(
                    best_result["items"]
                ),
                best_result[
                    "items"
                ],
            )

            if current_key < best_key:
                best_result = result

    return {
        "status": "OPTIMAL",
        "selected_items":
            best_result["items"],
        "utility":
            best_result["utility"],
        "privacy":
            best_result["privacy"],
        "cognitive_cost":
            best_result[
                "cognitive_cost"
            ],
        "normalized_cognitive_cost":
            best_result[
                "normalized_cognitive_cost"
            ],
        "disclosure_cost":
            best_result[
                "disclosure_cost"
            ],
    }


def greedy_mst(
    items: list[InformationItem],
    tau: float,
    lambda_privacy: float = 1.0,
    lambda_cognitive: float = 1.0,
    lambda_cardinality: float = 0.01,
    epsilon: float = 1e-12,
) -> dict:
    """
    Solve MST greedily using marginal modeled utility
    per marginal normalized disclosure cost.
    """

    if not 0.0 < tau <= 1.0:
        raise ValueError(
            "tau must be in the interval (0, 1]."
        )

    _validate_unique_ids(
        items
    )

    total_items = len(
        items
    )

    if total_items == 0:
        return {
            "status": "INFEASIBLE",
            "selected_items": tuple(),
            "utility": 0.0,
            "privacy": 0.0,
            "cognitive_cost": 0.0,
            "normalized_cognitive_cost": 0.0,
            "disclosure_cost": float("inf"),
        }

    if mst_utility(items) < tau:
        return {
            "status": "INFEASIBLE",
            "selected_items": tuple(),
            "utility":
                mst_utility(items),
            "privacy":
                mst_privacy(items),
            "cognitive_cost":
                mst_cognitive_cost(items),
            "normalized_cognitive_cost":
                mst_normalized_cognitive_cost(
                    selected_items=items,
                    all_items=items,
                ),
            "disclosure_cost":
                float("inf"),
        }

    selected = []
    remaining = list(
        items
    )

    while (
        mst_utility(selected)
        < tau
    ):

        current_utility = (
            mst_utility(selected)
        )

        current_cost = (
            mst_disclosure_cost(
                selected_items=selected,
                all_items=items,
                lambda_privacy=lambda_privacy,
                lambda_cognitive=lambda_cognitive,
                lambda_cardinality=lambda_cardinality,
            )
        )

        best_item = None
        best_score = float(
            "-inf"
        )

        for item in remaining:

            candidate = (
                selected + [item]
            )

            candidate_utility = (
                mst_utility(
                    candidate
                )
            )

            candidate_cost = (
                mst_disclosure_cost(
                    selected_items=candidate,
                    all_items=items,
                    lambda_privacy=lambda_privacy,
                    lambda_cognitive=lambda_cognitive,
                    lambda_cardinality=lambda_cardinality,
                )
            )

            delta_utility = (
                candidate_utility
                - current_utility
            )

            delta_cost = (
                candidate_cost
                - current_cost
            )

            if delta_utility <= 0.0:
                continue

            score = (
                delta_utility
                / (
                    delta_cost
                    + epsilon
                )
            )

            if (
                score > best_score
                or (
                    abs(
                        score
                        - best_score
                    )
                    <= 1e-12
                    and best_item
                    is not None
                    and item.item_id
                    < best_item.item_id
                )
            ):
                best_item = item
                best_score = score

        if best_item is None:
            return {
                "status":
                    "INFEASIBLE",
                "selected_items":
                    tuple(
                        item.item_id
                        for item
                        in selected
                    ),
                "utility":
                    mst_utility(
                        selected
                    ),
                "privacy":
                    mst_privacy(
                        selected
                    ),
                "cognitive_cost":
                    mst_cognitive_cost(
                        selected
                    ),
                "normalized_cognitive_cost":
                    mst_normalized_cognitive_cost(
                        selected_items=selected,
                        all_items=items,
                    ),
                "disclosure_cost":
                    float("inf"),
            }

        selected.append(
            best_item
        )

        remaining.remove(
            best_item
        )

    final_result = (
        evaluate_subset(
            subset=selected,
            all_items=items,
            lambda_privacy=lambda_privacy,
            lambda_cognitive=lambda_cognitive,
            lambda_cardinality=lambda_cardinality,
        )
    )

    return {
        "status": "FEASIBLE",
        "selected_items":
            final_result["items"],
        "utility":
            final_result["utility"],
        "privacy":
            final_result["privacy"],
        "cognitive_cost":
            final_result[
                "cognitive_cost"
            ],
        "normalized_cognitive_cost":
            final_result[
                "normalized_cognitive_cost"
            ],
        "disclosure_cost":
            final_result[
                "disclosure_cost"
            ],
    }


def progressive_mst_step(
    selected_items: list[InformationItem],
    all_items: list[InformationItem],
    lambda_privacy: float = 1.0,
    lambda_cognitive: float = 1.0,
    lambda_cardinality: float = 0.01,
    epsilon: float = 1e-12,
) -> dict:
    """
    Perform one Progressive-MST disclosure step.
    """

    _validate_unique_ids(
        all_items
    )

    selected_ids = {
        item.item_id
        for item
        in selected_items
    }

    remaining_items = [
        item
        for item in all_items
        if item.item_id
        not in selected_ids
    ]

    if not remaining_items:
        return {
            "status":
                "NO_ADDITIONAL_INFORMATION",
            "selected_items":
                tuple(
                    item.item_id
                    for item
                    in selected_items
                ),
        }

    current_utility = (
        mst_utility(
            selected_items
        )
    )

    current_cost = (
        mst_disclosure_cost(
            selected_items=selected_items,
            all_items=all_items,
            lambda_privacy=lambda_privacy,
            lambda_cognitive=lambda_cognitive,
            lambda_cardinality=lambda_cardinality,
        )
    )

    best_item = None
    best_score = float(
        "-inf"
    )

    for item in remaining_items:

        expanded_subset = (
            selected_items
            + [item]
        )

        new_utility = (
            mst_utility(
                expanded_subset
            )
        )

        new_cost = (
            mst_disclosure_cost(
                selected_items=expanded_subset,
                all_items=all_items,
                lambda_privacy=lambda_privacy,
                lambda_cognitive=lambda_cognitive,
                lambda_cardinality=lambda_cardinality,
            )
        )

        delta_utility = (
            new_utility
            - current_utility
        )

        delta_cost = (
            new_cost
            - current_cost
        )

        if delta_utility <= 0.0:
            continue

        score = (
            delta_utility
            / (
                delta_cost
                + epsilon
            )
        )

        if (
            score > best_score
            or (
                abs(
                    score
                    - best_score
                )
                <= 1e-12
                and best_item
                is not None
                and item.item_id
                < best_item.item_id
            )
        ):
            best_item = item
            best_score = score

    if best_item is None:
        return {
            "status":
                "NO_ADDITIONAL_INFORMATION",
            "selected_items":
                tuple(
                    item.item_id
                    for item
                    in selected_items
                ),
        }

    expanded_items = (
        selected_items
        + [best_item]
    )

    result = (
        evaluate_subset(
            subset=expanded_items,
            all_items=all_items,
            lambda_privacy=lambda_privacy,
            lambda_cognitive=lambda_cognitive,
            lambda_cardinality=lambda_cardinality,
        )
    )

    return {
        "status": "EXPANDED",
        "added_item":
            best_item.item_id,
        "selected_items":
            result["items"],
        "utility":
            result["utility"],
        "privacy":
            result["privacy"],
        "cognitive_cost":
            result[
                "cognitive_cost"
            ],
        "normalized_cognitive_cost":
            result[
                "normalized_cognitive_cost"
            ],
        "disclosure_cost":
            result[
                "disclosure_cost"
            ],
        "selection_score":
            best_score,
    }


def progressive_mst_sequence(
    initial_items: list[InformationItem],
    all_items: list[InformationItem],
    lambda_privacy: float = 1.0,
    lambda_cognitive: float = 1.0,
    lambda_cardinality: float = 0.01,
) -> list[dict]:
    """
    Generate the complete Progressive-MST disclosure sequence.
    """

    _validate_unique_ids(
        all_items
    )

    current_items = list(
        initial_items
    )

    history = []

    initial_result = (
        evaluate_subset(
            subset=current_items,
            all_items=all_items,
            lambda_privacy=lambda_privacy,
            lambda_cognitive=lambda_cognitive,
            lambda_cardinality=lambda_cardinality,
        )
    )

    history.append(
        {
            "step": 0,
            "added_item": None,
            **initial_result,
        }
    )

    step = 1

    while True:

        result = (
            progressive_mst_step(
                selected_items=current_items,
                all_items=all_items,
                lambda_privacy=lambda_privacy,
                lambda_cognitive=lambda_cognitive,
                lambda_cardinality=lambda_cardinality,
            )
        )

        if (
            result["status"]
            != "EXPANDED"
        ):
            break

        added_id = (
            result["added_item"]
        )

        added_item = next(
            item
            for item in all_items
            if item.item_id
            == added_id
        )

        current_items.append(
            added_item
        )

        history.append(
            {
                "step":
                    step,
                "added_item":
                    added_id,
                "items":
                    result[
                        "selected_items"
                    ],
                "utility":
                    result[
                        "utility"
                    ],
                "privacy":
                    result[
                        "privacy"
                    ],
                "cognitive_cost":
                    result[
                        "cognitive_cost"
                    ],
                "normalized_cognitive_cost":
                    result[
                        "normalized_cognitive_cost"
                    ],
                "disclosure_cost":
                    result[
                        "disclosure_cost"
                    ],
            }
        )

        step += 1

    return history


def compute_optimality_gap(
    exact_result: dict,
    approximate_result: dict,
) -> float:
    """
    Compute the relative optimality gap.

    Gap =
        (J_approx - J_exact)
        / J_exact
    """

    if (
        exact_result["status"]
        != "OPTIMAL"
    ):
        raise ValueError(
            "The exact result must contain "
            "an optimal solution."
        )

    if (
        approximate_result[
            "status"
        ]
        not in {
            "FEASIBLE",
            "OPTIMAL",
        }
    ):
        raise ValueError(
            "The approximate result must "
            "contain a feasible solution."
        )

    exact_cost = (
        exact_result[
            "disclosure_cost"
        ]
    )

    approximate_cost = (
        approximate_result[
            "disclosure_cost"
        ]
    )

    if exact_cost <= 0.0:

        if (
            approximate_cost
            == exact_cost
        ):
            return 0.0

        raise ValueError(
            "Relative optimality gap is "
            "undefined when the exact "
            "optimal cost is zero."
        )

    return (
        approximate_cost
        - exact_cost
    ) / exact_cost
