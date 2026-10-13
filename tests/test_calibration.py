import numpy as np
import pytest

from vision_qc.calibration import ConformalRouter, ScoreProbability
from vision_qc.metrics import auroc, binomial_upper, ece, pixel_metrics


def test_normal_only_cannot_release():
    router = ConformalRouter([1, 2, 3], [])
    assert all(router.predict(s)["route"] != "pass" for s in [-100, 0, 1, 100])


def test_resolution_and_ties_are_conservative():
    assert ConformalRouter([1] * 100, [10] * 98).predict(0)["route"] == "review"
    assert ConformalRouter([1] * 100, [10] * 99).predict(0)["route"] == "pass"
    assert ConformalRouter([1] * 100, [10] * 99).predict(10)["p_defect"] == 1
    assert ConformalRouter([1] * 100, [10] * 99).predict(np.nan)["route"] == "review"


def test_empty_prediction_set_is_reviewed():
    result = ConformalRouter([1] * 100, [10] * 100).predict(5)
    assert result["prediction_set"] == []
    assert result["route"] == "review"


def test_exact_risk_bound_needs_299_zero_miss_defects():
    assert binomial_upper(0, 298) > 0.01
    assert binomial_upper(0, 299) < 0.01
    assert binomial_upper(0, 0) == 1
    assert binomial_upper(5, 5) == 1


def test_probability_is_separate_and_monotone():
    model = ScoreProbability.fit([0, 1, 2, 3, 4, 5], [0, 0, 0, 1, 1, 1])
    assert np.all(np.diff(model.predict(np.arange(6))) >= 0)
    assert 0 <= ece([0, 1], model.predict([0, 5])) <= 1
    with pytest.raises(ValueError):
        ScoreProbability.fit([1, 2], [0, 0])


def test_known_metric_values():
    assert auroc([0, 0, 1, 1], [0, 1, 2, 3]) == 1
    assert auroc([0, 1], [1, 1]) == 0.5
    assert auroc([1, 1], [0, 1]) is None
    mask = np.array([[0, 0], [1, 1]])
    metrics = pixel_metrics([mask], [mask.astype(float)])
    assert metrics["pixel_auroc"] == 1
    assert metrics["aupro_0_30_approx"] == pytest.approx(1)
