"""
Reference scales of the MST objective (revised model).

Interpretation cost and cardinality are normalized by fixed reference values set by
the policy owner, the size and the cost of a typical acceptable disclosure, so that a
weight profile has the same meaning whatever the size of the pool:
    C(S) = sum_{i in S} c_i / C_ref,     K(S) = |S| / K_ref.
For the synthetic benchmarks, K_ref = 10 items and C_ref = 10 times the mean item
cost of the generator (Beta(2,2) scaled to [0.10, 1.00], mean 0.55), i.e. 5.5.
"""

REF_ITEMS = 10.0
REF_ITEM_COST = 0.55
REF_COST = REF_ITEMS * REF_ITEM_COST
