"""Optional genuine PatchCore/EfficientAD training entry point; not run in the CPU demo.

Uses ONLY train and prob_fit splits during fitting. Calibration/validation/test stay untouched.
Designed for anomalib 2.x; use a separate Python 3.11/3.12 environment.
"""
import argparse
import json
import shutil
from pathlib import Path

from vision_qc.data import load_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=["patchcore", "efficientad"], default="patchcore")
    parser.add_argument("--accelerator", choices=["cpu", "gpu", "auto"], default="auto")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    from anomalib.data import Folder
    from anomalib.engine import Engine
    from anomalib.models import EfficientAd, Patchcore
    from anomalib.data.utils import TestSplitMode, ValSplitMode
    from lightning.pytorch import seed_everything
    from PIL import Image
    seed_everything(42, workers=True)
    samples = load_manifest(args.manifest)
    if len({s.category for s in samples}) != 1:
        raise ValueError("Use one category per model")
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.output / "fit_data"
    for directory in ("train/good", "tuning/good", "tuning/bad", "tuning/masks"):
        (root / directory).mkdir(parents=True)
    used = []
    for index, sample in enumerate(samples):
        if sample.split not in {"train", "prob_fit"}:
            continue
        role = "train/good" if sample.split == "train" else "tuning/bad" if sample.label else "tuning/good"
        filename = f"{index:07d}.png"
        with Image.open(sample.path) as image:
            image.convert("RGB").save(root / role / filename)
        if sample.label:
            if sample.mask is None:
                raise ValueError("This training adapter requires defect masks in prob_fit")
            with Image.open(sample.mask) as mask:
                mask.convert("L").save(root / "tuning/masks" / filename)
        used.append(dict(id=sample.id, split=sample.split))
    data = Folder(name=next(iter({s.category for s in samples})), root=root, normal_dir="train/good",
                  normal_test_dir="tuning/good", abnormal_dir="tuning/bad", mask_dir="tuning/masks",
                  train_batch_size=1 if args.model == "efficientad" else 16, eval_batch_size=1,
                  num_workers=0, test_split_mode=TestSplitMode.FROM_DIR,
                  val_split_mode=ValSplitMode.SAME_AS_TEST, seed=42)
    model = Patchcore() if args.model == "patchcore" else EfficientAd()
    engine = Engine(default_root_dir=str(args.output), accelerator=args.accelerator, devices=1,
                    max_epochs=1 if args.model == "patchcore" else args.epochs)
    engine.fit(model=model, datamodule=data)
    # Engine's tuning data is NOT the project's risk validation or final test.
    exported = engine.export(model=model, export_type="onnx", export_root=str(args.output / "export"))
    (args.output / "training-protocol.json").write_text(json.dumps(dict(
        model=args.model, fitting_samples=used, export=str(exported),
        note="Official model trained; fit new score calibration and risk gate before routing."), indent=2), encoding="utf-8")
    print(f"Exported {exported}. This optional adapter has not been validated on your real dataset.")


if __name__ == "__main__":
    main()
