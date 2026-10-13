from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .data import load_manifest, make_demo
from .pipeline import Inspector, dump_json, evaluate, train


def retrain(run: Path, state: Path, output: Path) -> dict:
    from .store import ReviewStore
    inspector = Inspector(run)
    meta = inspector.meta
    manifest = Path(meta["manifest"])
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != meta["manifest_sha256"]:
        raise ValueError("Original manifest changed; make an explicit new experiment")
    provenance = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
    protected = [p for p in provenance if p["split"] != "pool"]
    heldout = {p["sha256"] for p in protected}
    # Compare decoded pixels as uploads are re-encoded to PNG by the API.
    from PIL import Image
    def pixel_hash(path):
        with Image.open(path) as image:
            image = image.convert("RGB")
            return hashlib.sha256(image.tobytes() + str(image.size).encode()).hexdigest()
    protected_pixels = {pixel_hash(Path(p["path"])) for p in protected}
    for p in provenance:
        if hashlib.sha256(Path(p["path"]).read_bytes()).hexdigest() != p["sha256"]:
            raise ValueError("Original data changed since training")
    rows = ReviewStore(state / "reviews.sqlite").list("labeled", 100000)
    normal, probability, accepted, seen = [], [], [], set()
    for row in rows:
        if row["result"]["category"] != meta["category"]:
            continue
        path = Path(row["image_path"])
        digest = pixel_hash(path)
        if digest in seen or digest in protected_pixels or hashlib.sha256(path.read_bytes()).hexdigest() in heldout:
            continue
        seen.add(digest)
        # Disjoint feedback roles: normal images for memory OR probability fit, never calibration.
        if row["label"] == 0 and int(digest[:8], 16) % 2 == 0:
            normal.append(path)
        else:
            probability.append((path, row["label"]))
        accepted.append({"id": row["id"], "label": row["label"], "revision": row["revision"], "sha256_pixels": digest})
    if not accepted:
        raise ValueError("No new eligible labels. Original dataset images and unknown categories cannot enter feedback training.")
    result = train(manifest, output, alpha_defect=meta["conformal"]["alpha_defect"],
                   alpha_normal=meta["conformal"]["alpha_normal"], seed=meta["seed"], synthetic=meta["synthetic"],
                   extra_training=normal, extra_probability=probability)
    # Feedback acquisition may depend on the old calibration set; after adapting the model,
    # those same heldouts are diagnostic only, NOT independent calibration evidence.
    result["release_gate"]["approved"] = False
    result["release_gate"]["scope"] = "adaptive candidate: requires fresh independent calibration and validation"
    result["release_gate"]["reason"] = "feedback_selection_depends_on_previous_calibration"
    dump_json(output / "model.json", result)
    dump_json(output / "feedback-snapshot.json", accepted)
    return {"candidate": str(output), "accepted_labels": len(accepted), "release_gate": result["release_gate"],
            "note": "Candidate only. Automatic release disabled until fresh independent calibration and validation."}


def recalibrate(run: Path, manifest: Path, output: Path) -> dict:
    """Freeze the model and require disjoint new calibration/validation images."""
    from PIL import Image
    inspector = Inspector(run)
    previous = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
    def digest(path):
        with Image.open(path) as image:
            image = image.convert("RGB")
            return hashlib.sha256(image.tobytes() + str(image.size).encode()).hexdigest()
    used = {digest(Path(row["path"])) for row in previous}
    history = run / "used-image-fingerprints.json"
    if history.exists():
        used.update(json.loads(history.read_text(encoding="utf-8")))
    feedback = run / "feedback-snapshot.json"
    if feedback.exists():
        used.update(row["sha256_pixels"] for row in json.loads(feedback.read_text(encoding="utf-8")))
    new_samples = load_manifest(manifest)
    if {row.category for row in new_samples} != {inspector.meta["category"]}:
        raise ValueError("Recalibration must use the same product category as the frozen model")
    for row in new_samples:
        if row.split in {"calibration", "validation"} and digest(row.path) in used:
            raise ValueError("Fresh calibration/validation must not overlap any previous experiment or feedback image")
    result = train(manifest, output, frozen_model=inspector.model, seed=inspector.meta["seed"],
                   synthetic=inspector.meta["synthetic"], alpha_normal=inspector.router.alpha_normal,
                   alpha_defect=inspector.router.alpha_defect)
    result["parent_version"] = inspector.version
    dump_json(output / "model.json", result)
    dump_json(output / "used-image-fingerprints.json", sorted(used | {digest(row.path) for row in new_samples}))
    return {"run": str(output), "release_gate": result["release_gate"]}


def main():
    parser = argparse.ArgumentParser(description="视觉质检 Agent")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Generate synthetic data, train and evaluate")
    demo.add_argument("--data", type=Path, default=Path("data/demo"))
    demo.add_argument("--output", type=Path, default=Path("runs/demo"))
    demo.add_argument("--seed", type=int, default=42)
    fit = sub.add_parser("train")
    fit.add_argument("--manifest", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--alpha-defect", type=float, default=0.01)
    fit.add_argument("--alpha-normal", type=float, default=0.05)
    fit.add_argument("--seed", type=int, default=42)
    test = sub.add_parser("evaluate")
    test.add_argument("--run", type=Path, required=True)
    test.add_argument("--manifest", type=Path, required=True)
    test.add_argument("--split", choices=["validation", "test"], default="test")
    candidate = sub.add_parser("retrain")
    candidate.add_argument("--run", type=Path, required=True)
    candidate.add_argument("--state", type=Path, required=True)
    candidate.add_argument("--output", type=Path, required=True)
    fresh = sub.add_parser("recalibrate")
    fresh.add_argument("--run", type=Path, required=True)
    fresh.add_argument("--manifest", type=Path, required=True)
    fresh.add_argument("--output", type=Path, required=True)
    experiments = sub.add_parser("experiments")
    experiments.add_argument("--run", type=Path, required=True)
    experiments.add_argument("--manifest", type=Path, required=True)
    experiments.add_argument("--seed", type=int, default=42)
    benchmark = sub.add_parser("benchmark")
    benchmark.add_argument("--run", type=Path, required=True)
    benchmark.add_argument("--image", type=Path, required=True)
    benchmark.add_argument("--repeats", type=int, default=100)
    export = sub.add_parser("export-onnx")
    export.add_argument("--run", type=Path, required=True)
    export.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "demo":
            manifest = args.data / "manifest.csv"
            if not manifest.exists():
                manifest = make_demo(args.data, args.seed)
            if not args.output.exists():
                train(manifest, args.output, seed=args.seed, synthetic=True)
            result = evaluate(args.output, manifest)
        elif args.command == "train":
            result = train(args.manifest, args.output, alpha_defect=args.alpha_defect, alpha_normal=args.alpha_normal, seed=args.seed)
        elif args.command == "evaluate":
            result = evaluate(args.run, args.manifest, args.split)
        elif args.command == "retrain":
            result = retrain(args.run, args.state, args.output)
        elif args.command == "recalibrate":
            result = recalibrate(args.run, args.manifest, args.output)
        elif args.command == "experiments":
            from .experiments import run_experiments
            result = run_experiments(args.run, args.manifest, args.seed)
        elif args.command == "benchmark":
            from .experiments import benchmark
            result = benchmark(args.run, args.image, args.repeats)
        else:
            from .export import export_onnx
            result = export_onnx(args.run, args.image)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
