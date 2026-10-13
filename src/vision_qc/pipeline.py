from __future__ import annotations

import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import label as components, find_objects

from .calibration import ConformalRouter, ScoreProbability
from .data import load_manifest
from .metrics import auroc, ece, pixel_metrics, routing_metrics
from .model import PatchMemory


def dump_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def image_signature(image: Image.Image) -> np.ndarray:
    x = np.asarray(image.convert("RGB").resize((32, 32)), dtype=np.float32) / 255
    return np.array([x.mean(), x.std(), np.mean((x < 0.02) | (x > 0.98))])


def localization(heatmap: np.ndarray, threshold: float, width: int, height: int) -> list[list[int]]:
    regions, _ = components(heatmap > threshold)
    boxes = []
    for index, region in enumerate(find_objects(regions), 1):
        if region is None or np.sum(regions[region] == index) < 4:
            continue
        yy, xx = region
        boxes.append([int(xx.start * width / heatmap.shape[1]), int(yy.start * height / heatmap.shape[0]),
                      int(np.ceil(xx.stop * width / heatmap.shape[1])), int(np.ceil(yy.stop * height / heatmap.shape[0]))])
    return boxes[:30]


def overlay(image: Image.Image, heatmap: np.ndarray, threshold: float, boxes: list[list[int]]) -> Image.Image:
    base = image.convert("RGB")
    scaled = np.asarray(Image.fromarray(heatmap.astype(np.float32)).resize(base.size))
    strength = np.clip(scaled / max(threshold * 2, 1e-6), 0, 1)[..., None] * 0.55
    rgb = np.asarray(base).astype(float)
    rgb = rgb * (1 - strength) + np.array([255, 55, 45]) * strength
    result = Image.fromarray(np.uint8(np.clip(rgb, 0, 255)))
    draw = ImageDraw.Draw(result)
    for box in boxes:
        draw.rectangle(box, outline=(255, 190, 30), width=2)
    return result


class Inspector:
    def __init__(self, run: str | Path):
        self.run = Path(run).resolve()
        self.meta = json.loads((self.run / "model.json").read_text(encoding="utf-8"))
        self.version = self.meta["version"]
        model_path = self.run / "model.npz"
        if hashlib.sha256(model_path.read_bytes()).hexdigest() != self.meta["weights_sha256"]:
            raise ValueError("Model checksum mismatch")
        self.model = PatchMemory.load(model_path)
        self.router = ConformalRouter(**self.meta["conformal"])
        self.probability = ScoreProbability(**self.meta["probability"]) if self.meta["probability"] else None

    def inspect(self, image: Image.Image, category: str, enforce_gate: bool = True) -> tuple[dict, np.ndarray]:
        start = time.perf_counter()
        detected = self.model.predict(image)
        decision = self.router.predict(detected.score)
        decision["raw_route"] = decision["route"]
        reasons = []
        if category != self.meta["category"]:
            reasons.append("unknown_category")
        signature = image_signature(image)
        drift = np.any(signature < self.meta["signature_low"]) or np.any(signature > self.meta["signature_high"])
        if drift:
            reasons.append("image_distribution_shift")
        if enforce_gate and decision["route"] == "pass" and not self.meta["release_gate"]["approved"]:
            reasons.append("insufficient_validation_evidence")
        if reasons:
            decision.update(route="review", reason=",".join(reasons))
        boxes = localization(detected.heatmap, self.meta["pixel_threshold"], *image.size)
        result = dict(**decision, score=detected.score,
                      defect_probability=float(self.probability.predict(detected.score)) if self.probability else None,
                      category=category, model_version=self.version, boxes=boxes,
                      embedding=detected.embedding.tolist(), drift=bool(drift),
                      latency_ms=(time.perf_counter() - start) * 1000,
                      explanation=f"检测到 {len(boxes)} 个候选区域；位置由热力图阈值生成，需结合原图复核。",
                      explanation_source="deterministic_heatmap_summary", synthetic=self.meta["synthetic"])
        return result, detected.heatmap


def train(manifest: Path, output: Path, *, alpha_defect: float = 0.01, alpha_normal: float = 0.05,
          seed: int = 42, synthetic: bool = False, extra_training: list[Path] | None = None,
          extra_probability: list[tuple[Path, int]] | None = None, frozen_model=None) -> dict:
    if output.exists():
        raise FileExistsError(f"Run directory already exists: {output}")
    samples = load_manifest(manifest)
    categories = {s.category for s in samples}
    if len(categories) != 1:
        raise ValueError("Train one category per run; use separate runs for different products")
    grouped = {split: [s for s in samples if s.split == split] for split in ("train", "prob_fit", "calibration", "validation")}
    if any(not grouped[k] for k in ("train", "calibration", "validation")):
        raise ValueError("train/calibration/validation splits are required")
    start = time.perf_counter()
    model = frozen_model or PatchMemory(seed=seed).fit([s.path for s in grouped["train"]] + (extra_training or []))
    scores, signatures, train_maps = {}, [], []
    for split, rows in grouped.items():
        scores[split] = []
        for sample in rows:
            with Image.open(sample.path) as image:
                detected = model.predict(image)
                scores[split].append(detected.score)
                if split == "train":
                    signatures.append(image_signature(image))
                    train_maps.append(detected.heatmap)
    probability_labels = [s.label for s in grouped["prob_fit"]]
    for path, label in extra_probability or []:
        with Image.open(path) as image:
            scores["prob_fit"].append(model.predict(image).score)
        probability_labels.append(label)
    probability = (ScoreProbability.fit(scores["prob_fit"], probability_labels)
                   if set(probability_labels) == {0, 1} else None)
    cal = grouped["calibration"]
    router = ConformalRouter([v for s, v in zip(cal, scores["calibration"]) if s.label == 0],
                             [v for s, v in zip(cal, scores["calibration"]) if s.label == 1],
                             alpha_normal, alpha_defect)
    validation = routing_metrics([s.label for s in grouped["validation"]], [router.predict(s) for s in scores["validation"]])
    release_gate = {"approved": validation["fnr_upper_95"] <= alpha_defect,
                    "target_fnr": alpha_defect, "confidence": 0.95,
                    "scope": "fixed-model independent validation; not a drift or repeated-selection guarantee",
                    "validation": validation}
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + hashlib.sha256(
        model.bank.tobytes() + str(seed).encode() + manifest.read_bytes()
        + json.dumps(router.to_dict(), sort_keys=True).encode()
        + json.dumps(probability.to_dict() if probability else None, sort_keys=True).encode()).hexdigest()[:8]
    output.mkdir(parents=True)
    model.save(output / "model.npz")
    signature = np.stack(signatures)
    margins = np.array([0.16, 0.12, 0.10])
    meta = dict(version=version, category=next(iter(categories)), backend=model.backend, synthetic=synthetic,
                manifest=str(manifest.resolve()), manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                weights_sha256=hashlib.sha256((output / "model.npz").read_bytes()).hexdigest(),
                probability=probability.to_dict() if probability else None, conformal=router.to_dict(),
                release_gate=release_gate, seed=seed, signature_low=(signature.min(0) - margins).tolist(),
                signature_high=(signature.max(0) + margins).tolist(),
                pixel_threshold=float(np.quantile(np.stack(train_maps), 0.995)),
                train_seconds=time.perf_counter() - start, python=platform.python_version(), platform=platform.platform(),
                counts={k: len(v) for k, v in grouped.items()},
                extra_training_count=len(extra_training or []), extra_probability_count=len(extra_probability or []))
    dump_json(output / "model.json", meta)
    dump_json(output / "provenance.json", [dict(id=s.id, split=s.split, category=s.category, label=s.label,
              path=str(s.path), sha256=hashlib.sha256(s.path.read_bytes()).hexdigest()) for s in samples])
    return meta


def evaluate(run: Path, manifest: Path, split: str = "test") -> dict:
    inspector = Inspector(run)
    samples = [s for s in load_manifest(manifest) if s.split == split]
    if not samples or any(s.label is None for s in samples):
        raise ValueError("Evaluation requires a labeled, non-empty split")
    predictions, maps, masks = [], [], []
    for sample in samples:
        with Image.open(sample.path) as image:
            result, heatmap = inspector.inspect(image, sample.category)
        predictions.append(dict(id=sample.id, label=sample.label, **result))
        if sample.mask:
            with Image.open(sample.mask) as mask:
                masks.append(np.asarray(mask.convert("L").resize(heatmap.shape[::-1], Image.Resampling.NEAREST)) > 0)
            maps.append(heatmap)
        elif sample.label == 0:
            masks.append(np.zeros_like(heatmap, dtype=bool))
            maps.append(heatmap)
    labels, scores = [s.label for s in samples], [p["score"] for p in predictions]
    latency = [p["latency_ms"] for p in predictions]
    report = dict(model_version=inspector.version, split=split, synthetic=inspector.meta["synthetic"],
                  image_auroc=auroc(labels, scores), **routing_metrics(labels, predictions),
                  **pixel_metrics(masks, maps),
                  pixel_evaluation_images=len(masks), pixel_missing_defect_masks=sum(s.label == 1 and s.mask is None for s in samples),
                  ece=ece(labels, [p["defect_probability"] for p in predictions]) if inspector.probability else None,
                  latency_p50_ms=float(np.percentile(latency, 50)), latency_p95_ms=float(np.percentile(latency, 95)),
                  serial_inference_images_per_second=1000 / float(np.mean(latency)),
                  model_bytes=(run / "model.npz").stat().st_size,
                  release_gate=inspector.meta["release_gate"],
                  caveat="Synthetic pipeline check only" if inspector.meta["synthetic"] else "Held-out results; read sample counts and uncertainty bounds")
    dump_json(run / f"metrics-{split}.json", report)
    dump_json(run / f"predictions-{split}.json", predictions)
    return report
