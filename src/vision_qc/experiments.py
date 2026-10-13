from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from .active import select_batch
from .calibration import ConformalRouter
from .data import load_manifest
from .metrics import auroc, routing_metrics
from .model import PatchMemory
from .pipeline import Inspector, dump_json


def save_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score_samples(model, samples):
    results = []
    for sample in samples:
        with Image.open(sample.path) as image:
            results.append(model.predict(image))
    return results


def learning_curves(inspector: Inspector, samples, seed: int) -> list[dict]:
    """Offline oracle simulation. Pool labels are revealed only AFTER choosing IDs."""
    grouped = {k: [s for s in samples if s.split == k] for k in ("train", "calibration", "test", "pool")}
    if not grouped["pool"] or any(s.label is None for s in grouped["pool"]):
        return []
    results = []
    for strategy in ("random", "uncertainty", "coreset"):
        remaining = {s.id: s for s in grouped["pool"]}
        selected, normal_additions = [], []
        model = inspector.model
        for budget in (0, 10, 20, 40):
            if budget and remaining:
                cal_detections = score_samples(model, grouped["calibration"])
                router = ConformalRouter(
                    [d.score for s, d in zip(grouped["calibration"], cal_detections) if s.label == 0],
                    [d.score for s, d in zip(grouped["calibration"], cal_detections) if s.label == 1],
                    inspector.router.alpha_normal, inspector.router.alpha_defect)
                candidates = list(remaining.values())
                detections = score_samples(model, candidates)
                items = [dict(id=s.id, embedding=d.embedding.tolist(), **router.predict(d.score)) for s, d in zip(candidates, detections)]
                ids = select_batch(items, budget - len(selected), strategy, seed)
                # No pool label is exposed to the acquisition function.
                for sample_id in ids:
                    sample = remaining.pop(sample_id)
                    selected.append(sample)
                    if sample.label == 0:
                        normal_additions.append(sample.path)
                model = PatchMemory(seed=seed).fit([s.path for s in grouped["train"]] + normal_additions)
            cal_detections = score_samples(model, grouped["calibration"])
            router = ConformalRouter(
                [d.score for s, d in zip(grouped["calibration"], cal_detections) if s.label == 0],
                [d.score for s, d in zip(grouped["calibration"], cal_detections) if s.label == 1],
                inspector.router.alpha_normal, inspector.router.alpha_defect)
            detected = score_samples(model, grouped["test"])
            labels = [s.label for s in grouped["test"]]
            metrics = routing_metrics(labels, [router.predict(d.score) for d in detected])
            results.append(dict(strategy=strategy, seed=seed, queried_labels=len(selected),
                                normal_labels_used=len(normal_additions), defect_labels_collected=sum(s.label for s in selected),
                                initial_calibration_labels=len(grouped["calibration"]),
                                image_auroc=auroc(labels, [d.score for d in detected]),
                                review_rate=metrics["review_rate"], auto_release_fnr=metrics["auto_release_fnr"],
                                fnr_upper_95=metrics["fnr_upper_95"]))
    return results


def run_experiments(run: Path, manifest: Path, seed: int = 42) -> dict:
    inspector = Inspector(run)
    samples = load_manifest(manifest)
    validation = [s for s in samples if s.split == "validation"]
    test = [s for s in samples if s.split == "test"]
    val_detection = score_samples(inspector.model, validation)
    test_detection = score_samples(inspector.model, test)
    labels, scores = [s.label for s in test], [d.score for d in test_detection]
    # Threshold grid derived ONLY from validation normals. Test is descriptive, never used to select a policy.
    val_normal = [d.score for s, d in zip(validation, val_detection) if s.label == 0]
    val_defect = [d.score for s, d in zip(validation, val_detection) if s.label == 1]
    if not test or not val_normal or not val_defect:
        raise ValueError("Experiments require test and validation with both classes")
    curves = []
    for quantile in (0.8, 0.9, 0.95, 0.99):
        high = float(np.quantile(val_normal, quantile))
        low = min(high, float(np.quantile(val_defect, 0.01)))
        alarm_threshold = max(high, float(np.quantile(val_defect, 0.01)))
        for policy in ("fixed", "double"):
            decisions = []
            for s in scores:
                route = ("alarm" if s >= high else "pass") if policy == "fixed" else ("pass" if s < low else "alarm" if s >= alarm_threshold else "review")
                decisions.append(dict(route=route, prediction_set=[0] if route == "pass" else [1] if route == "alarm" else [0, 1]))
            curves.append(dict(policy=policy, setting=quantile, **routing_metrics(labels, decisions)))
    for alpha in (0.005, 0.01, 0.02, 0.05, 0.1):
        router = ConformalRouter(inspector.router.normal_scores, inspector.router.defect_scores, 0.05, alpha)
        curves.append(dict(policy="conformal", setting=alpha, **routing_metrics(labels, [router.predict(s) for s in scores])))
    robust = []
    for transform in ("clean", "brightness_0.65", "brightness_1.35", "blur_1.5", "noise_0.06", "rotation_8"):
        rng, decisions, detected_scores = np.random.default_rng(seed), [], []
        for sample in test:
            with Image.open(sample.path) as opened:
                image = opened.convert("RGB")
            if transform.startswith("brightness"):
                image = ImageEnhance.Brightness(image).enhance(float(transform.split("_")[1]))
            elif transform.startswith("blur"):
                image = image.filter(ImageFilter.GaussianBlur(1.5))
            elif transform.startswith("noise"):
                pixels = np.asarray(image, dtype=float) + rng.normal(0, 255 * 0.06, (*image.size[::-1], 3))
                image = Image.fromarray(np.uint8(np.clip(pixels, 0, 255)))
            elif transform.startswith("rotation"):
                image = image.rotate(8, resample=Image.Resampling.BILINEAR)
            result, _ = inspector.inspect(image, sample.category)
            decisions.append(result)
            detected_scores.append(result["score"])
        robust.append(dict(transform=transform, image_auroc=auroc(labels, detected_scores),
                           drift_review_rate=float(np.mean([r["drift"] for r in decisions])), **routing_metrics(labels, decisions)))
    learning = learning_curves(inspector, samples, seed)
    output = run / "experiments" / f"seed-{seed}"
    output.mkdir(parents=True, exist_ok=True)
    save_csv(output / "routing.csv", curves)
    save_csv(output / "robustness.csv", robust)
    save_csv(output / "learning.csv", learning)
    result = dict(synthetic=inspector.meta["synthetic"], seed=seed, output=str(output),
                  routing_rows=len(curves), robustness_rows=len(robust), learning_rows=len(learning),
                  notes=["Curves are exploratory; no threshold is selected on test.",
                         "Active learning updates normal memory; defect labels are collected but not used by this one-class learner.",
                         "Calibration labels are a separate initial cost. Acquisition uses calibration, so reused-calibration learning curves have NO conformal guarantee.",
                         "API audit sampling is not included in offline routing rates."])
    dump_json(output / "metadata.json", result)
    return result


def benchmark(run: Path, image_path: Path, repeats: int = 100) -> dict:
    if repeats < 5:
        raise ValueError("Use at least 5 measurements")
    inspector = Inspector(run)
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    for _ in range(10):
        inspector.inspect(image, inspector.meta["category"])
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        inspector.inspect(image, inspector.meta["category"])
        times.append((time.perf_counter() - start) * 1000)
    report = dict(backend=inspector.model.backend, repeats=repeats, warmup=10, batch_size=1, device="cpu",
                  timing_scope="PIL resize + model + calibration + routing + localization; excludes decode/network/storage",
                  p50_ms=float(np.percentile(times, 50)), p95_ms=float(np.percentile(times, 95)),
                  serial_images_per_second=1000 / float(np.mean(times)),
                  model_bytes=(run / "model.npz").stat().st_size, gpu_memory_bytes=None)
    dump_json(run / "benchmark.json", report)
    return report
