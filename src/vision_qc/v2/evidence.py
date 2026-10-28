from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

from PIL import Image

from ..metrics import binomial_upper


def hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def pixel_hash(path):
    with Image.open(path) as image:
        image = image.convert("RGB")
        return hashlib.sha256(image.tobytes() + str(image.size).encode()).hexdigest()


def fingerprints(samples):
    records = [dict(id=s.id, split=s.split, category=s.category, group=s.group, label=s.label,
                    pixels=pixel_hash(s.path), mask=hashlib.sha256(s.mask.read_bytes()).hexdigest() if s.mask else None)
               for s in samples]
    seen = {}
    for row in records:
        previous = seen.setdefault(row["pixels"], row)
        if previous is not row and (previous["split"] != row["split"] or previous["group"] != row["group"]):
            raise ValueError("Decoded image duplication across splits or independent groups")
    # Unlike the original CSV hash, independent of the filesystem root.
    return records, hash_json(sorted(records, key=lambda x: x["id"]))


POLICY_FIELDS = ("schema_version", "backend", "category", "weights", "detector", "conformal", "probability",
                 "signature_low", "signature_high", "pixel_threshold", "secondary", "routing_policy", "audit_fraction")


def policy_hash(meta):
    return hash_json({k: meta.get(k) for k in POLICY_FIELDS})


def certificate(meta, metrics, calibration_hash, acceptance_hash, *, independent=False, days=30):
    now = datetime.now(timezone.utc)
    return dict(policy_sha256=policy_hash(meta), calibration_sha256=calibration_hash,
                acceptance_sha256=acceptance_hash, category=meta["category"],
                created_at=now.isoformat(), expires_at=(now+timedelta(days=days)).isoformat(),
                independent_sampling_attested=independent, synthetic=meta.get("synthetic", False),
                confidence=0.95, target_fnr=meta["conformal"]["alpha_defect"],
                n_defect=metrics["n_defect"], missed_defects=metrics["missed_defects"],
                false_alarm_rate=metrics.get("false_alarm_rate"),
                fnr_upper_95=metrics["fnr_upper_95"], scope="frozen whole policy; one-time independent acceptance")


def check_certificate(meta, now=None):
    try:
        return _check_certificate(meta, now)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False, "invalid_certificate"


def _check_certificate(meta, now=None):
    cert = meta.get("certificate")
    if not cert:
        return False, "missing_certificate"
    now = now or datetime.now(timezone.utc)
    if cert["policy_sha256"] != policy_hash(meta) or cert["category"] != meta["category"]:
        return False, "policy_changed"
    if meta.get("schema_version") != 2:
        return False, "unsupported_schema"
    for key in ("calibration_sha256", "acceptance_sha256"):
        if len(cert.get(key,"")) != 64 or cert[key] != meta.get("evidence", {}).get(key):
            return False, "evidence_changed"
    if datetime.fromisoformat(cert["created_at"]) > now:
        return False, "invalid_certificate_date"
    if datetime.fromisoformat(cert["expires_at"]) <= now:
        return False, "certificate_expired"
    if not cert.get("independent_sampling_attested") or cert.get("synthetic") or meta.get("synthetic"):
        return False, "independent_production_evidence_missing"
    if cert["confidence"] != 0.95 or cert["target_fnr"] != meta["conformal"]["alpha_defect"]:
        return False, "risk_definition_changed"
    if type(cert["missed_defects"]) is not int or type(cert["n_defect"]) is not int:
        return False, "invalid_counts"
    upper = binomial_upper(cert["missed_defects"], cert["n_defect"])
    if not math.isfinite(cert["fnr_upper_95"]) or abs(upper-cert["fnr_upper_95"]) > 1e-10 or upper > cert["target_fnr"]:
        return False, "insufficient_acceptance_evidence"
    false_alarm = cert.get("false_alarm_rate")
    if false_alarm is None or not math.isfinite(false_alarm) or not 0 <= false_alarm <= .1:
        return False, "false_alarm_limit_exceeded"
    return True, "approved"


def check_weights(run: Path, meta):
    if meta.get("schema_version") != 2 or not meta.get("weights"):
        raise ValueError("Invalid v2 model schema or missing weight checksums")
    required = {"detector.npz" if meta["backend"] == "patch_memory_cpu" else "detector.pt"}
    if meta.get("secondary"):
        required.add("secondary.pt")
    if not required.issubset(meta["weights"]):
        raise ValueError("Missing required artifact checksums")
    for name, expected in meta["weights"].items():
        path = (run / name).resolve()
        if not path.is_relative_to(run.resolve()) or not path.is_file():
            raise ValueError("Invalid model artifact path")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Model artifact checksum mismatch: {name}")
