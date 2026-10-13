from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


@dataclass
class ScoreProbability:
    center: float
    scale: float
    slope: float
    bias: float

    @classmethod
    def fit(cls, scores: list[float], labels: list[int]) -> "ScoreProbability":
        y = np.asarray(labels, dtype=float)
        if len(scores) != len(y) or set(y) != {0.0, 1.0}:
            raise ValueError("Independent probability-fit set requires both classes")
        center, scale = float(np.mean(scores)), max(float(np.std(scores)), 1e-6)
        x = (np.asarray(scores) - center) / scale

        def objective(w):
            z = w[0] * x + w[1]
            return float(np.mean(np.logaddexp(0, z) - y * z) + 0.002 * w[0] ** 2)

        result = minimize(objective, [1.0, 0.0], bounds=[(0, None), (None, None)])
        if not result.success:
            raise RuntimeError(f"Probability calibration failed: {result.message}")
        return cls(center, scale, float(result.x[0]), float(result.x[1]))

    def predict(self, score):
        return expit(self.slope * (np.asarray(score) - self.center) / self.scale + self.bias)

    def to_dict(self):
        return asdict(self)


@dataclass
class ConformalRouter:
    normal_scores: list[float]
    defect_scores: list[float]
    alpha_normal: float = 0.05
    alpha_defect: float = 0.01

    def __post_init__(self):
        if not 0 < self.alpha_normal < 1 or not 0 < self.alpha_defect < 1:
            raise ValueError("Alpha must be between zero and one")
        if not np.isfinite(self.normal_scores + self.defect_scores).all():
            raise ValueError("Calibration scores must be finite")

    def predict(self, score: float) -> dict:
        if not np.isfinite(score):
            return dict(route="review", prediction_set=[0, 1], p_normal=1.0, p_defect=1.0, reason="invalid_score")
        # Nonconformity is s for class 0 and -s for class 1. Ties included conservatively.
        p0 = (1 + int(np.sum(np.asarray(self.normal_scores) >= score))) / (len(self.normal_scores) + 1)
        p1 = (1 + int(np.sum(np.asarray(self.defect_scores) <= score))) / (len(self.defect_scores) + 1)
        labels = ([0] if p0 > self.alpha_normal else []) + ([1] if p1 > self.alpha_defect else [])
        route = "pass" if labels == [0] else "alarm" if labels == [1] else "review"
        reason = "singleton" if len(labels) == 1 else "ambiguous" if labels else "empty_prediction_set"
        return dict(route=route, prediction_set=labels, p_normal=p0, p_defect=p1, reason=reason)

    def to_dict(self):
        return asdict(self)
