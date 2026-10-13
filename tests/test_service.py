import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from vision_qc.api import create_app
from vision_qc.data import make_demo, synthetic_image
from vision_qc.pipeline import Inspector, train
from vision_qc.cli import retrain


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("integration")
    manifest = make_demo(root / "data")
    train(manifest, root / "run", synthetic=True)
    return root


def image_bytes(seed=99):
    image, _ = synthetic_image(np.random.default_rng(seed), False)
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def test_end_to_end_feedback_conflict_and_selection(trained, tmp_path):
    client = TestClient(create_app(trained / "run", tmp_path))
    assert client.get("/health").json()["status"] == "ready"
    assert client.post("/inspect", files={"file": ("bad.png", b"not an image")}, data={"category": "x"}).status_code == 422
    response = client.post("/inspect", files={"file": ("ok.png", image_bytes())}, data={"category": "unknown"})
    assert response.status_code == 200
    row = response.json()
    assert row["result"]["route"] == "review"
    assert "unknown_category" in row["result"]["reason"]
    assert client.get(f'/reviews/{row["id"]}/image?heatmap=true').status_code == 200
    duplicate = client.post("/inspect", files={"file": ("other.png", image_bytes())}, data={"category": "unknown"}).json()
    assert duplicate["id"] == row["id"]
    assert client.get("/active-learning").json()["ids"] == []  # Unknown category needs its own model.
    body = dict(label=0, reviewer="tester", note="normal", revision=0)
    assert client.post(f'/reviews/{row["id"]}/feedback', json=body).status_code == 200
    assert client.post(f'/reviews/{row["id"]}/feedback', json=body).status_code == 409
    assert client.get("/stats").json()["labeled"] == 1
    history = client.get(f'/reviews/{row["id"]}/history').json()
    assert len(history) == 1 and history[0]["reviewer"] == "tester"


def test_release_gate_and_drift(trained):
    inspector = Inspector(trained / "run")
    image, _ = synthetic_image(np.random.default_rng(11), False)
    inspector.meta["release_gate"]["approved"] = False
    decision, _ = inspector.inspect(image, "demo_texture")
    assert decision["route"] == "review"
    assert "insufficient_validation_evidence" in decision["reason"]
    from PIL import ImageEnhance
    shifted, _ = inspector.inspect(ImageEnhance.Brightness(image).enhance(0.3), "demo_texture")
    assert shifted["drift"]
    assert shifted["route"] == "review"


def test_retrain_uses_new_feedback_and_preserves_holdout(trained, tmp_path):
    state = tmp_path / "state"
    client = TestClient(create_app(trained / "run", state))
    for seed in range(3):
        row = client.post("/inspect", files={"file": ("x.png", image_bytes(seed + 100))}, data={"category": "demo_texture"}).json()
        client.post(f'/reviews/{row["id"]}/feedback', json=dict(label=0, reviewer="test", note="", revision=0))
    before = (trained / "run" / "model.json").read_bytes()
    result = retrain(trained / "run", state, tmp_path / "candidate")
    assert result["accepted_labels"] == 3
    assert before == (trained / "run" / "model.json").read_bytes()
    after = json.loads((tmp_path / "candidate" / "model.json").read_text())
    original = json.loads(before)
    assert after["counts"]["calibration"] == original["counts"]["calibration"]
    assert after["release_gate"]["approved"] is False


def test_api_key_and_missing_model(tmp_path, monkeypatch):
    monkeypatch.setenv("QC_API_KEY", "test-key")
    client = TestClient(create_app(tmp_path / "absent", tmp_path / "state"))
    assert client.get("/health").status_code == 401
    assert client.get("/model", headers={"X-API-Key": "test-key"}).status_code == 503
