from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SPLITS = {"train", "prob_fit", "calibration", "validation", "test", "pool"}


@dataclass(frozen=True)
class Sample:
    id: str
    path: Path
    category: str
    split: str
    label: int | None
    group: str
    mask: Path | None = None


def load_manifest(path: str | Path, check_hashes: bool = True) -> list[Sample]:
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("Manifest is empty")
    samples, ids, files, groups, hashes = [], set(), set(), {}, {}
    for row in rows:
        split = row["split"]
        label = int(row["label"]) if row.get("label", "") != "" else None
        if split not in SPLITS or label not in (0, 1, None):
            raise ValueError("Invalid split or label")
        if split != "pool" and label is None:
            raise ValueError("Only pool samples may have unknown labels")
        if split == "train" and label != 0:
            raise ValueError("Anomaly detector training must contain normal images only")
        file = (path.parent / row["path"]).resolve()
        group = row.get("group") or row["id"]
        if row["id"] in ids or file in files:
            raise ValueError("Duplicate sample ID or image path")
        if group in groups and groups[group] != split:
            raise ValueError(f"Group leakage across splits: {group}")
        if not file.is_file():
            raise ValueError(f"Missing image: {file}")
        if check_hashes:
            digest = hashlib.sha256(file.read_bytes()).hexdigest()
            if digest in hashes and hashes[digest] != split:
                raise ValueError("Identical image content appears across splits")
            hashes[digest] = split
        mask = (path.parent / row["mask"]).resolve() if row.get("mask") else None
        if mask and not mask.is_file():
            raise ValueError(f"Missing mask: {mask}")
        samples.append(Sample(row["id"], file, row["category"], split, label, group, mask))
        ids.add(row["id"])
        files.add(file)
        groups[group] = split
    return samples


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "path", "category", "split", "label", "group", "mask"])
        writer.writeheader()
        writer.writerows(rows)


def synthetic_image(rng: np.random.Generator, defect: bool, size: int = 96) -> tuple[Image.Image, Image.Image]:
    """Procedural texture, used ONLY to exercise the pipeline, never as benchmark data."""
    yy, xx = np.mgrid[:size, :size]
    texture = 128 + 8 * np.sin(xx / 3.5) + 7 * np.cos(yy / 4.5)
    texture = texture + rng.normal(0, 2, (size, size)) + rng.normal(0, 3)
    rgb = np.repeat(texture[..., None], 3, axis=-1)
    rgb += np.array([6, 1, -5])
    image = Image.fromarray(np.uint8(np.clip(rgb, 0, 255)))
    mask = Image.new("L", (size, size))
    if defect:
        x, y = rng.integers(18, size - 30, 2).tolist()
        width = int(rng.integers(3, 10))
        points = (x, y, x + int(rng.integers(10, 25)), y + int(rng.integers(10, 25)))
        if rng.random() < 0.5:
            ImageDraw.Draw(mask).line(points, fill=255, width=width)
        else:
            ImageDraw.Draw(mask).ellipse(points, fill=255)
        color = (int(rng.integers(15, 55)),) * 3
        image.paste(color, mask=mask)
    return image, mask


def make_demo(root: Path, seed: int = 42) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    target = root / "manifest.csv"
    if target.exists():
        raise FileExistsError(f"Dataset already exists: {target}. Choose a new directory.")
    rng, rows = np.random.default_rng(seed), []
    # >= 299 independent defects are needed for a 95% one-sided upper bound < 1% at zero misses.
    counts = {"train": (64, 0), "prob_fit": (40, 40), "calibration": (120, 120),
              "validation": (320, 320), "test": (80, 80), "pool": (60, 60)}
    for split, (normal, defect) in counts.items():
        directory = root / split
        directory.mkdir(exist_ok=True)
        for i in range(normal + defect):
            label = int(i >= normal)
            image, mask = synthetic_image(rng, bool(label))
            sample_id = f"{split}-{i:04d}"
            image.save(directory / f"{sample_id}.png")
            mask.save(directory / f"{sample_id}-mask.png")
            rows.append(dict(id=sample_id, path=f"{split}/{sample_id}.png", category="demo_texture",
                             split=split, label=label, group=sample_id, mask=f"{split}/{sample_id}-mask.png"))
    write_manifest(target, rows)
    return target
