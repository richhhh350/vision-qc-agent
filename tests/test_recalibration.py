import json

import pytest

from vision_qc.cli import recalibrate
from vision_qc.data import make_demo
from vision_qc.pipeline import train


def test_reused_holdout_is_rejected(tmp_path):
    manifest = make_demo(tmp_path / "data", seed=21)
    train(manifest, tmp_path / "run", synthetic=True)
    with pytest.raises(ValueError, match="Fresh calibration"):
        recalibrate(tmp_path / "run", manifest, tmp_path / "reused")
    fresh_manifest = make_demo(tmp_path / "fresh", seed=22)
    result = recalibrate(tmp_path / "run", fresh_manifest, tmp_path / "calibrated")
    assert "release_gate" in result
    original = json.loads((tmp_path / "run" / "model.json").read_text())
    updated = json.loads((tmp_path / "calibrated" / "model.json").read_text())
    assert original["weights_sha256"] == updated["weights_sha256"]
