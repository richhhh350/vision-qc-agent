import numpy as np
import pytest
from PIL import Image

from vision_qc.active import select_batch
from vision_qc.data import load_manifest, write_manifest
from vision_qc.model import coreset_indices


def test_group_and_content_leakage_rejected(tmp_path):
    Image.new("RGB", (8, 8), (1, 2, 3)).save(tmp_path / "a.png")
    Image.new("RGB", (8, 8), (4, 5, 6)).save(tmp_path / "b.png")
    rows = [dict(id="a", path="a.png", category="x", split="train", label=0, group="part1", mask=""),
            dict(id="b", path="b.png", category="x", split="test", label=1, group="part1", mask="")]
    path = tmp_path / "manifest.csv"
    write_manifest(path, rows)
    with pytest.raises(ValueError, match="Group leakage"):
        load_manifest(path)
    rows[1]["group"] = "part2"
    (tmp_path / "b.png").write_bytes((tmp_path / "a.png").read_bytes())
    write_manifest(path, rows)
    with pytest.raises(ValueError, match="Identical"):
        load_manifest(path)


@pytest.mark.parametrize("strategy", ["random", "uncertainty", "coreset"])
def test_acquisition_unique_and_deterministic(strategy):
    items = [dict(id=str(i), p_normal=0.2, p_defect=i / 10, embedding=[float(i), 1.0]) for i in range(10)]
    chosen = select_batch(items, 20, strategy)
    assert len(chosen) == len(set(chosen)) == 10
    assert chosen == select_batch(items, 20, strategy)
    assert select_batch(items, 0, strategy) == []


def test_degenerate_coreset_no_duplicates():
    assert len(set(coreset_indices(np.zeros((10, 3)), 5))) == 5
