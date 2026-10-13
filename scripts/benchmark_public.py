"""One-class CPU baseline on the complete official test split, without test-driven fitting."""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
from PIL import Image

from vision_qc.data import load_manifest
from vision_qc.metrics import auroc, pixel_metrics
from vision_qc.model import PatchMemory
from vision_qc.pipeline import dump_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = json.loads(args.manifest.with_suffix(".protocol.json").read_text(encoding="utf-8"))
    if protocol["protocol"] != "official" or not protocol["official_test_preserved"]:
        raise ValueError("Use an official-protocol manifest for the main benchmark")
    samples = load_manifest(args.manifest)
    train = [s for s in samples if s.split == "train"]
    test = [s for s in samples if s.split == "test"]
    if not train or not test:
        raise ValueError("Training and test images are required")
    start = time.perf_counter()
    model = PatchMemory(seed=args.seed).fit([s.path for s in train])
    training_seconds = time.perf_counter() - start
    predictions, maps, masks, times = [], [], [], []
    for sample in test:
        with Image.open(sample.path) as im:
            image = im.convert("RGB")
        start = time.perf_counter()
        detected = model.predict(image)
        times.append((time.perf_counter() - start) * 1000)
        predictions.append(dict(id=sample.id, label=sample.label, score=detected.score, path=str(sample.path)))
        if sample.mask:
            with Image.open(sample.mask) as mask:
                masks.append(np.asarray(mask.convert("L").resize((model.size, model.size), Image.Resampling.NEAREST)) > 0)
            maps.append(detected.heatmap)
        elif sample.label == 0:
            masks.append(np.zeros((model.size, model.size), dtype=bool))
            maps.append(detected.heatmap)
    args.output.mkdir(parents=True)
    model.save(args.output / "model.npz")
    report = dict(dataset=protocol["dataset"], category=protocol["category"], synthetic=False,
                  protocol="official train normals only; complete official test; fixed hyperparameters",
                  n_train_normal=len(train), n_test=len(test), n_test_defect=sum(s.label for s in test),
                  excluded_training_defects=sum(s.label == 1 and s.split != "test" for s in samples),
                  image_auroc=auroc([s.label for s in test], [p["score"] for p in predictions]),
                  **pixel_metrics(masks, maps), pixel_evaluation_images=len(masks),
                  pixel_resolution=[model.size, model.size],
                  pixel_note="96x96 nearest-neighbor masks; small defects can vanish. Not native-resolution official localization score.",
                  backend=model.backend, model_bytes=(args.output / "model.npz").stat().st_size,
                  train_seconds=training_seconds, p50_inference_ms=float(np.percentile(times, 50)),
                  p95_inference_ms=float(np.percentile(times, 95)), timing_scope="resize/features/nearest-neighbor, excludes decode and routing",
                  seed=args.seed, python=platform.python_version(), platform=platform.platform(),
                  manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                  risk_certification="None: ranking benchmark only; route calibration is a separate experiment")
    dump_json(args.output / "metrics.json", report)
    dump_json(args.output / "predictions.json", predictions)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
