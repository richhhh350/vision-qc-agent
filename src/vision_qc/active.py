from __future__ import annotations

import numpy as np

from .model import coreset_indices


def select_batch(items: list[dict], budget: int, strategy: str, seed: int = 42) -> list[str]:
    if budget < 0:
        raise ValueError("Budget must be nonnegative")
    if strategy not in {"random", "uncertainty", "coreset"}:
        raise ValueError("Unknown sampling strategy")
    if not items or budget == 0:
        return []
    budget = min(budget, len(items))
    if strategy == "random":
        indices = np.random.default_rng(seed).choice(len(items), budget, replace=False)
    elif strategy == "uncertainty":
        # Rank by ambiguity of the two conformal p-values, NOT by treating them as probabilities.
        values = [min(x["p_normal"], x["p_defect"]) for x in items]
        indices = np.argsort(-np.asarray(values), kind="stable")[:budget]
    else:
        features = np.asarray([x["embedding"] for x in items], dtype=float)
        if not np.isfinite(features).all():
            raise ValueError("Non-finite embedding")
        features = (features - features.mean(0)) / np.maximum(features.std(0), 1e-6)
        indices = coreset_indices(features, budget, seed)
    return [items[int(i)]["id"] for i in indices]
