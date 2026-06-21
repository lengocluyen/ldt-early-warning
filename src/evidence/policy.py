from __future__ import annotations


def should_present_evidence(
    evidence_type: str,
    selected_types: set[str],
    presented_count: int,
    min_items: int = 3,
) -> bool:
    """Diversity-aware selection: fill a minimum set, then prefer unseen types."""
    return presented_count < min_items or evidence_type not in selected_types
