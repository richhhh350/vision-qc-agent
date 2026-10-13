import importlib.util
from pathlib import Path
from PIL import Image


def test_official_test_is_never_used_for_calibration(tmp_path):
    spec = importlib.util.spec_from_file_location("importer", Path(__file__).parents[1] / "scripts/import_dataset.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    records = [dict(path=tmp_path / f"{origin}-{label}-{i}.png", label=label, origin=origin, mask=None)
               for origin in ("train", "test") for label in (0, 1) for i in range(30)]
    for protocol in ("official", "routing"):
        rows = module.allocate(records, "part", protocol, 42)
        assert sum(r["split"] == "test" for r in rows) == 60
        assert all(r["split"] == "test" for r in rows if Path(r["path"]).name.startswith("test"))
        assert all(r["label"] == 0 for r in rows if r["split"] == "train")


def test_ksdd_archive_duplicate_is_documented(tmp_path):
    spec = importlib.util.spec_from_file_location("importer", Path(__file__).parents[1] / "scripts/import_dataset.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    train = tmp_path / "train"
    train.mkdir()
    Image.new("RGB", (4, 4), "gray").save(train / "10301.png")
    Image.new("L", (4, 4), 0).save(train / "10301_GT.png")
    (train / "10301 (copy).png").write_bytes((train / "10301.png").read_bytes())
    (train / "10301_GT (copy).png").write_bytes((train / "10301_GT.png").read_bytes())
    exclusions = []
    records = module.collect("ksdd2", tmp_path, "ksdd2", tmp_path / "derived", exclusions)
    assert len(records) == 1
    assert len(exclusions) == 1
