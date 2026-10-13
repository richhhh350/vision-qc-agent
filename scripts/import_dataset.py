"""Import public datasets while preserving the complete official test split.

official: all official normal training images train the detector; training defects go to pool.
routing: create calibration/validation/pool from official TRAIN only.
"""
import argparse
import csv
import json
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from vision_qc.data import load_manifest, write_manifest


def collect(kind, root, category, derived, exclusions=None):
    records = []
    if kind in {"mvtec", "btad", "loco"}:
        folder = root / category
        for origin in ("train", "validation", "test"):
            for path in sorted((folder / origin).glob("*/*")):
                if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
                    continue
                defect = path.parent.name
                label = int(defect not in {"good", "ok"})
                mask = None
                if label:
                    if kind == "loco":
                        parts = sorted((folder / "ground_truth" / defect / path.stem).glob("*.png"))
                        if not parts:
                            raise ValueError(f"Missing LOCO masks: {path}")
                        combined = None
                        for part in parts:
                            with Image.open(part) as im:
                                binary = np.asarray(im.convert("L")) > 0
                            combined = binary if combined is None else combined | binary
                        derived.mkdir(parents=True, exist_ok=True)
                        mask = derived / f"{category}-{defect}-{path.stem}.png"
                        Image.fromarray(combined.astype(np.uint8) * 255).save(mask)
                    else:
                        suffix = "_mask.png" if kind == "mvtec" else ".png"
                        mask = folder / "ground_truth" / defect / (path.stem + suffix)
                records.append(dict(path=path, label=label, origin=origin, stratum=defect, mask=mask))
    elif kind == "visa":
        with (root / "split_csv" / "1cls.csv").open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if row["object"] == category:
                    records.append(dict(path=root / row["image"], label=int(row["label"] != "normal"),
                                        origin=row["split"], stratum=row["label"],
                                        mask=root / row["mask"] if row.get("mask") else None))
    else:
        for origin in ("train", "test"):
            for path in sorted((root / origin).glob("*.png")):
                if path.stem.removesuffix(" (copy)").endswith("_GT"):
                    continue
                mask = path.with_name(path.stem + "_GT.png")
                if not mask.exists() and path.stem.endswith(" (copy)"):
                    original = path.with_name(path.stem.removesuffix(" (copy)") + path.suffix)
                    if original.is_file() and original.with_name(original.stem + "_GT.png").is_file():
                        digest = hashlib.sha256(path.read_bytes()).hexdigest()
                        if digest == hashlib.sha256(original.read_bytes()).hexdigest():
                            if exclusions is not None:
                                exclusions.append(dict(path=str(path), duplicate_of=str(original), sha256=digest))
                            continue
                with Image.open(mask) as im:
                    label = int(np.any(np.asarray(im) > 0))
                records.append(dict(path=path, label=label, origin=origin, stratum=str(label), mask=mask))
    if not records:
        raise ValueError("No matching images found. Check dataset root/category.")
    return records


def allocate(records, category, protocol, seed):
    rng, strata, rows = np.random.default_rng(seed), defaultdict(list), []
    for index, record in enumerate(records):
        record["id"] = f"{category}-{index:06d}"
        strata[(record["origin"], record["label"])].append(record)
    for (origin, label), group in strata.items():
        order = rng.permutation(len(group))
        if origin == "test":
            allocations = [("test", 1.0)]
        elif origin == "validation":
            allocations = [("validation", 1.0)]
        elif protocol == "official":
            allocations = [("pool" if label else "train", 1.0)]
        elif label == 0:
            allocations = [("train", 0.7), ("prob_fit", 0.1), ("calibration", 0.1), ("validation", 0.1)]
        else:
            allocations = [("prob_fit", 0.15), ("calibration", 0.45), ("validation", 0.3), ("pool", 0.1)]
        offset = 0
        for i, (split, ratio) in enumerate(allocations):
            count = len(group) - offset if i == len(allocations) - 1 else int(len(group) * ratio)
            for index in order[offset:offset + count]:
                record = group[index]
                rows.append(dict(id=record["id"], path=str(record["path"].resolve()), category=category,
                                 split=split, label=record["label"], group=record["id"],
                                 mask=str(record["mask"].resolve()) if record["mask"] else ""))
            offset += count
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["mvtec", "visa", "btad", "ksdd2", "loco"], required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", choices=["official", "routing"], default="official")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    exclusions = []
    records = collect(args.kind, args.root, args.category, args.output.parent / "derived_masks", exclusions)
    rows = allocate(records, args.category, args.protocol, args.seed)
    write_manifest(args.output, rows)
    load_manifest(args.output)
    metadata = dict(dataset=args.kind, category=args.category, protocol=args.protocol, seed=args.seed,
                    official_test_preserved=True, counts=dict(Counter(f'{r["split"]}:{r["label"]}' for r in rows)),
                    excluded_exact_duplicate_files=exclusions,
                    note="One-class detector uses normal training only; routing calibration uses official training only.",
                    pixel_note="LOCO union masks support diagnostic AUROC only; official sPRO needs the publisher evaluator.")
    args.output.with_suffix(".protocol.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
