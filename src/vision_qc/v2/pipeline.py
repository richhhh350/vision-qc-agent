from __future__ import annotations

import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from ..calibration import ConformalRouter, ScoreProbability
from ..data import load_manifest
from ..metrics import auroc, ece, routing_metrics
from ..pipeline import dump_json, image_signature, localization
from .detectors import DeepDetector, LegacyDetector, load_detector
from .evidence import certificate, check_certificate, check_weights, fingerprints, hash_json, policy_hash
from .metrics import NativePixelMetrics, prevalence_cost


class InspectorV2:
    def __init__(self, run):
        self.run = Path(run).resolve()
        self.meta = json.loads((self.run/"model.json").read_text(encoding="utf-8"))
        check_weights(self.run, self.meta)
        self.version = self.meta["version"]
        self.model = load_detector(self.run, self.meta["detector"])
        self.router = ConformalRouter(**self.meta["conformal"])
        self.probability = ScoreProbability(**self.meta["probability"]) if self.meta["probability"] else None
        self.secondary = None
        if self.meta.get("secondary"):
            from .secondary import Segmenter
            self.secondary = Segmenter.load(self.run / "secondary.pt")
        approved, reason = check_certificate(self.meta)
        self.meta["release_gate"].update(approved=approved, reason=reason)

    def inspect(self, image, category, enforce_gate=True):
        start = time.perf_counter()
        detected = self.model.predict(image)
        detected_at = time.perf_counter()
        decision = self.router.predict(detected.score)
        decision["raw_route"] = decision["route"]
        reasons = []
        if category != self.meta["category"]:
            reasons.append("unknown_category")
        signature = image_signature(image)
        drift = bool(np.any(signature < self.meta["signature_low"]) or np.any(signature > self.meta["signature_high"]))
        if drift:
            reasons.append("image_distribution_shift")
        approved, gate_reason = check_certificate(self.meta)
        if enforce_gate and decision["route"] == "pass" and not approved:
            reasons.append(gate_reason)
        if reasons:
            decision.update(route="review", reason=",".join(reasons))
        decision["policy_route"] = decision["route"]
        image_hash = hashlib.sha256(image.convert("RGB").tobytes() + str(image.size).encode()).hexdigest()
        if decision["route"] != "review" and int(image_hash[:8], 16) / 2**32 < self.meta["audit_fraction"]:
            decision.update(route="review", reason="random_audit")
        secondary, secondary_ms = None, 0.0
        if decision["route"] == "review" and self.secondary:
            stage = time.perf_counter()
            secondary = self.secondary.predict(image)
            secondary_ms = (time.perf_counter()-stage)*1000
        boxes = localization(detected.heatmap, self.meta["pixel_threshold"], *image.size)
        routing_ms = (time.perf_counter()-detected_at)*1000-secondary_ms
        return dict(**decision, score=detected.score, category=category, model_version=self.version,
                    defect_probability=float(self.probability.predict(detected.score)) if self.probability else None,
                    embedding=detected.embedding.tolist(), boxes=boxes, drift=drift,
                    backend=self.meta["backend"], secondary=secondary,
                    latency_ms=(time.perf_counter()-start)*1000,
                    stage_timing={**getattr(self.model, "last_timing", {}), "secondary_ms": secondary_ms,
                                  "routing_localization_ms": max(0.0,routing_ms)},
                    policy_sha256=policy_hash(self.meta), heatmap_coordinates="original_image",
                    explanation=f"原图坐标下发现 {len(boxes)} 个候选区域；二级结果仅辅助人工复核。",
                    explanation_source="deterministic_heatmap_summary", synthetic=self.meta["synthetic"]), detected.heatmap


def train_v2(manifest, output, config, *, detector=None, independent=False):
    manifest, output = Path(manifest), Path(output)
    if output.exists():
        raise FileExistsError(output)
    samples = load_manifest(manifest)
    if len({s.category for s in samples}) != 1:
        raise ValueError("One category per model")
    if any(s.split == "validation" for s in samples) and any(s.split == "acceptance" for s in samples):
        raise ValueError("Use acceptance or legacy validation, not both")
    groups = {split: [s for s in samples if s.split == split] for split in {s.split for s in samples}}
    train = groups.get("train", [])
    if not train:
        raise ValueError("Normal train images required")
    for split in ("prob_fit", "calibration", "acceptance", "validation"):
        rows = groups.get(split, [])
        if len({s.group for s in rows}) != len(rows):
            raise ValueError(f"{split} requires one image per independent group; repeated views cannot add evidence")
    records, manifest_hash = fingerprints(samples)
    start = time.perf_counter()
    config = json.loads(json.dumps(config))
    detector_config = config["detector"]
    if detector is None:
        if detector_config["backend"] == "patch_memory_cpu":
            detector = LegacyDetector(**{k:v for k,v in detector_config.items() if k != "backend"})
        else:
            detector = DeepDetector(**detector_config)
        if detector.backend == "efficientad":
            from .efficient import fit_efficientad
            fit_efficientad(detector, [s.path for s in train], config.get("efficientad", {}))
        else:
            detector.fit([s.path for s in train])
    output.mkdir(parents=True)
    names = detector.save(output)
    weights = {name: hashlib.sha256((output/name).read_bytes()).hexdigest() for name in names}
    dump_json(output/"provenance.json", [dict(**r, path=str(s.path)) for r,s in zip(records,samples)])
    # Fixed train-only subsample estimates display threshold and simple drift guard.
    signature, pixels = [], []
    for idx in np.linspace(0, len(train)-1, min(64, len(train)), dtype=int):
        with Image.open(train[int(idx)].path) as image:
            signature.append(image_signature(image))
            heatmap = detector.predict(image).heatmap
            pixels.extend(heatmap.ravel()[::max(1, heatmap.size//512)].tolist())
    signatures = np.stack(signature)
    scored = {}
    for split in ("prob_fit", "calibration"):
        scored[split] = []
        for sample in groups.get(split, []):
            with Image.open(sample.path) as image:
                scored[split].append(detector.predict(image).score)
        print(f"Scored {split}: {len(scored[split])}", flush=True)
    labels = [s.label for s in groups.get("prob_fit", [])]
    probability = ScoreProbability.fit(scored["prob_fit"], labels) if set(labels) == {0, 1} else None
    calibration = groups.get("calibration", [])
    router = ConformalRouter([score for s,score in zip(calibration, scored["calibration"]) if s.label == 0],
                             [score for s,score in zip(calibration, scored["calibration"]) if s.label == 1],
                             config.get("alpha_normal", 0.05), config.get("alpha_defect", 0.01))
    meta = dict(schema_version=2, category=samples[0].category, backend=detector.backend, detector=detector.metadata(),
                weights=weights, manifest=str(manifest.resolve()), manifest_sha256=manifest_hash,
                synthetic=config.get("synthetic", False), probability=probability.to_dict() if probability else None,
                conformal=router.to_dict(), signature_low=(signatures.min(0)-[.16,.12,.10]).tolist(),
                signature_high=(signatures.max(0)+[.16,.12,.10]).tolist(), pixel_threshold=float(np.quantile(pixels,.995)),
                secondary=None, routing_policy="conformal+category+image_drift; secondary_advice_only",
                audit_fraction=.05, counts={k:len(v) for k,v in groups.items()},
                release_gate=dict(approved=False, target_fnr=router.alpha_defect, validation={}),
                python=platform.python_version(), platform=platform.platform(), config=config)
    meta["version"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"-"+policy_hash(meta)[:10]
    dump_json(output/"model.json",meta)
    # Reuse the fitted model for validation; no second load or accidental refit.
    inspector = object.__new__(InspectorV2)
    inspector.run, inspector.meta, inspector.model = output, meta, detector
    inspector.version, inspector.router, inspector.probability, inspector.secondary = meta["version"],router,probability,None
    acceptance = groups.get("acceptance", groups.get("validation", []))
    decisions = []
    for sample in acceptance:
        with Image.open(sample.path) as image:
            decisions.append(inspector.inspect(image,sample.category,enforce_gate=False)[0])
    metrics = routing_metrics([s.label for s in acceptance], decisions) if acceptance else dict(
        n=0,n_defect=0,missed_defects=0,auto_release_fnr=None,fnr_upper_95=1.0)
    cal_hash = hash_json([r for r in records if r["split"] == "calibration"])
    accept_hash = hash_json([r for r in records if r["split"] in {"acceptance", "validation"}])
    meta["evidence"] = dict(calibration_sha256=cal_hash, acceptance_sha256=accept_hash)
    meta["certificate"] = certificate(meta,metrics,cal_hash,accept_hash,independent=independent)
    approved, reason = check_certificate(meta)
    meta["release_gate"].update(approved=approved, reason=reason, validation=metrics)
    meta["train_seconds"] = time.perf_counter()-start
    dump_json(output/"model.json",meta)
    dump_json(output/"config.json",config)
    return meta


def evaluate_v2(run, manifest, split="test"):
    inspector = InspectorV2(run)
    samples = [s for s in load_manifest(manifest) if s.split == split]
    if not samples or any(s.label is None for s in samples):
        raise ValueError("Labeled nonempty evaluation split required")
    pixel = NativePixelMetrics()
    predictions = []
    for index, sample in enumerate(samples):
        with Image.open(sample.path) as image:
            result, heatmap = inspector.inspect(image,sample.category)
        predictions.append(dict(id=sample.id,label=sample.label,**result))
        if sample.mask:
            with Image.open(sample.mask) as mask:
                mask = np.asarray(mask.convert("L")) > 0
            pixel.update(mask,heatmap,inspector.meta["pixel_threshold"])
        elif sample.label == 0:
            pixel.update(np.zeros_like(heatmap,bool),heatmap,inspector.meta["pixel_threshold"])
        if (index+1)%100 == 0:
            print(f"Evaluate {split} {index+1}/{len(samples)}",flush=True)
    labels = [s.label for s in samples]
    latency = [r["latency_ms"] for r in predictions]
    report = dict(schema_version=2,model_version=inspector.version,backend=inspector.meta["backend"],split=split,
                  synthetic=inspector.meta["synthetic"],image_auroc=auroc(labels,[r["score"] for r in predictions]),
                  **routing_metrics(labels,predictions),**pixel.result(),
                  review_cost_scenarios=prevalence_cost(labels,predictions),
                  ece=ece(labels,[r["defect_probability"] for r in predictions]) if inspector.probability else None,
                  latency_p50_ms=float(np.percentile(latency,50)),latency_p95_ms=float(np.percentile(latency,95)),
                  timing_scope="preprocess+detectors+geometry+route; excludes decoding, HTTP and disk",
                  release_gate=inspector.meta["release_gate"],certificate=inspector.meta["certificate"],
                  pixel_missing_defect_masks=sum(s.label == 1 and s.mask is None for s in samples),
                  secondary_calls=sum(r["secondary"] is not None for r in predictions),
                  caveat="Repeated public benchmark; not new independent production acceptance")
    report["raw_conformal_routing"] = routing_metrics(labels,[dict(r,route=r["raw_route"]) for r in predictions])
    dump_json(Path(run)/f"metrics-{split}.json",report)
    dump_json(Path(run)/f"predictions-{split}.json",predictions)
    return report
