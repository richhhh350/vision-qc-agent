"""Few-shot NEW-category experiment, using validation only to pick the normal-image budget."""
import argparse
import json
from pathlib import Path

import numpy as np

from vision_qc.data import load_manifest, write_manifest
from vision_qc.pipeline import Inspector, dump_json, evaluate, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 5, 10, 20, 50])
    parser.add_argument("--max-review-rate", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    reference = Inspector(args.reference_run)
    samples = load_manifest(args.manifest)
    categories = {s.category for s in samples}
    if len(categories) != 1 or reference.meta["category"] in categories:
        raise ValueError("Provide exactly one NEW category, absent from the reference model")
    normal = [s for s in samples if s.split == "train"]
    other = [s for s in samples if s.split != "train"]
    order = np.random.default_rng(args.seed).permutation(len(normal))
    args.output.mkdir(parents=True, exist_ok=False)
    curve, chosen = [], None
    for budget in sorted(set(args.budgets)):
        if budget <= 0 or budget > len(normal):
            continue
        subset = [normal[i] for i in order[:budget]] + other
        manifest = args.output / f"budget-{budget}.csv"
        write_manifest(manifest, [dict(id=s.id, path=str(s.path), category=s.category, split=s.split,
                                      label=s.label if s.label is not None else "", group=s.group,
                                      mask=str(s.mask) if s.mask else "") for s in subset])
        run = args.output / f"budget-{budget}"
        model = train(manifest, run, seed=args.seed)
        val = model["release_gate"]["validation"]
        # This sweep is research only. Do not promote a winner on a repeatedly queried holdout.
        model["release_gate"]["approved"] = False
        model["release_gate"]["scope"] = "few-shot budget sweep: fresh final holdout required for promotion"
        dump_json(run / "model.json", model)
        curve.append(dict(normal_budget=budget, initial_nonpool_labels=sum(s.split not in {"pool", "test"} for s in other),
                          validation_fnr=val["auto_release_fnr"], validation_upper_95=val["fnr_upper_95"],
                          validation_review_rate=val["review_rate"]))
        if chosen is None and val["fnr_upper_95"] <= 0.01 and val["review_rate"] <= args.max_review_rate:
            chosen = budget
    result = dict(category=next(iter(categories)), curve=curve, validation_selected_budget=chosen,
                  note="Different budgets reuse validation: selection is exploratory; all candidate release gates remain disabled.")
    # Test is queried once, AFTER budget selection; absent a qualifying budget report failure.
    if chosen is not None:
        result["selected_test"] = evaluate(args.output / f"budget-{chosen}", args.output / f"budget-{chosen}.csv")
    dump_json(args.output / "adaptation.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
